"""Tests for §2.3 — TRUE per-game pitcher logs from MLB StatsAPI.

Pitcher projections used to be synthesized from ESPN season splits: GP identical
clones with no game_date and no per-game variance. `mlb_starters.get_pitcher_gamelog`
replaces that with real StatsAPI per-game logs, and a fail-open helper
(`_pitcher_gamelog_or_synth`) is threaded (via an optional `player_name`) through the
gamelog fetch chokepoints so the durable store, real-line calibration, and the
projection backtest all see dated pitcher rows.

Fully hermetic: StatsAPI (`mlb_starters._player_gamelog_splits` / `find_player_id`),
ESPN (`espn_client.get_athlete_gamelog` / `get_pitcher_stats`), and the SQL store
(in-memory SQLite) are mocked/stubbed — no live network or Azure I/O.
"""
from datetime import datetime, timezone
import unittest
from unittest.mock import patch, MagicMock

import db_store
import espn_client
import gamelog_store
import mlb_starters


def _split(date, ip="6.1", k="7", er="2", h="5", bb="1", opp="Yankees",
           is_home=True, game_pk=1000):
    """A StatsAPI pitching gameLog split (stat values are STRINGS, as the API
    returns them; inningsPitched is base-3 notation)."""
    return {
        "date": date,
        "isHome": is_home,
        "opponent": {"name": opp},
        "game": {"gamePk": game_pk},
        "stat": {"inningsPitched": ip, "strikeOuts": k, "earnedRuns": er,
                 "hits": h, "baseOnBalls": bb},
    }


class GetPitcherGamelogTransformTests(unittest.TestCase):
    """The split → row transform: field mapping, IP coercion, ordering, fail-open."""

    def _run(self, splits, found=("42", True), name="Gerrit Cole", season=2024):
        with patch.object(mlb_starters, "find_player_id", return_value=found) \
                as fpi, \
             patch.object(mlb_starters, "_player_gamelog_splits",
                          return_value=splits) as pgs:
            rows = mlb_starters.get_pitcher_gamelog(name, season)
        return rows, fpi, pgs

    def test_field_mapping_and_ip_is_base3_float(self):
        rows, _fpi, _pgs = self._run([_split("2024-07-04", ip="6.1", k="7",
                                             er="2", opp="Yankees",
                                             is_home=True)])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        # IP is a base-3 FLOAT (6.1), NOT the raw string and NOT decimalized 6.33.
        self.assertIsInstance(r["IP"], float)
        self.assertEqual(r["IP"], 6.1)
        self.assertEqual(espn_client.ip_to_outs(r["IP"]), 19)   # 6 IP + 1 out
        self.assertEqual(r["K"], 7.0)
        self.assertEqual(r["ER"], 2.0)
        self.assertEqual(r["game_date"], "2024-07-04")
        self.assertEqual(r["opponent"], "Yankees")
        self.assertTrue(r["is_home"])
        self.assertTrue(r["completed"])       # a past date is final
        self.assertNotIn("_gamePk", r)        # local tiebreak stripped

    def test_uses_pitching_group_and_passed_season(self):
        _rows, fpi, pgs = self._run([_split("2024-07-04")], season=2023)
        fpi.assert_called_once_with("Gerrit Cole", 2023)
        pgs.assert_called_once_with("42", "pitching", 2023)

    def test_newest_first_with_doubleheader_tiebreak(self):
        splits = [
            _split("2024-07-01", game_pk=1),
            _split("2024-07-05", game_pk=2),
            # doubleheader on 07-03: higher gamePk sorts first (deterministic).
            _split("2024-07-03", game_pk=50),
            _split("2024-07-03", game_pk=90),
        ]
        rows, _fpi, _pgs = self._run(splits)
        self.assertEqual([r["game_date"] for r in rows],
                         ["2024-07-05", "2024-07-03", "2024-07-03", "2024-07-01"])

    def test_classified_as_pitcher(self):
        rows, _fpi, _pgs = self._run([_split("2024-07-04")])
        self.assertEqual(gamelog_store._classify("baseball", rows, False),
                         "pitcher")

    def test_today_dated_row_marked_incomplete(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows, _fpi, _pgs = self._run([_split(today)])
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["completed"])   # same-day = possible live partial

    def test_split_without_innings_is_skipped(self):
        no_ip = {"date": "2024-07-04", "isHome": True,
                 "opponent": {"name": "Yankees"}, "game": {"gamePk": 1},
                 "stat": {"strikeOuts": "0"}}   # no inningsPitched -> not a game
        rows, _fpi, _pgs = self._run([no_ip, _split("2024-07-05")])
        self.assertEqual([r["game_date"] for r in rows], ["2024-07-05"])

    def test_empty_on_ambiguous_name(self):
        rows, _fpi, pgs = self._run([_split("2024-07-04")], found=None)
        self.assertEqual(rows, [])
        pgs.assert_not_called()          # never fetch a log we can't attribute

    def test_empty_on_non_pitcher(self):
        rows, _fpi, pgs = self._run([_split("2024-07-04")], found=("42", False))
        self.assertEqual(rows, [])       # never bind a pitcher prop to a batter
        pgs.assert_not_called()

    def test_empty_on_statsapi_miss(self):
        rows, _fpi, _pgs = self._run([])
        self.assertEqual(rows, [])

    def test_season_none_defaults_to_current_utc_year(self):
        # Production always calls with season=None (chokepoints pass None); the
        # `season or now().year` default must resolve to the current UTC year.
        year = datetime.now(timezone.utc).year
        with patch.object(mlb_starters, "find_player_id",
                          return_value=("42", True)) as fpi, \
             patch.object(mlb_starters, "_player_gamelog_splits",
                          return_value=[]) as pgs:
            mlb_starters.get_pitcher_gamelog("Gerrit Cole")   # season omitted
        fpi.assert_called_once_with("Gerrit Cole", year)
        pgs.assert_called_once_with("42", "pitching", year)


class PitcherGamelogOrSynthTests(unittest.TestCase):
    """The fail-open helper: real -> synth -> []; None name is byte-identical."""

    def test_real_hit_skips_synth(self):
        real = [{"IP": 6.0, "K": 5.0, "ER": 2.0, "game_date": "2024-07-04"}]
        with patch.object(mlb_starters, "get_pitcher_gamelog",
                          return_value=real), \
             patch.object(espn_client, "get_pitcher_stats") as synth:
            out = mlb_starters._pitcher_gamelog_or_synth("mlb", "42", "Ace", 2024)
        self.assertEqual(out, real)
        synth.assert_not_called()

    def test_real_miss_falls_back_to_synth(self):
        synth_rows = [{"IP": 5.0, "K": 6.0, "ER": 3.0}]
        with patch.object(mlb_starters, "get_pitcher_gamelog", return_value=[]), \
             patch.object(espn_client, "get_pitcher_stats",
                          return_value=synth_rows) as synth:
            out = mlb_starters._pitcher_gamelog_or_synth("mlb", "42", "Ace", 2024)
        self.assertEqual(out, synth_rows)
        synth.assert_called_once_with("mlb", "42", season=2024)

    def test_none_name_is_byte_identical_to_synth(self):
        synth_rows = [{"IP": 5.0, "K": 6.0, "ER": 3.0}]
        with patch.object(mlb_starters, "get_pitcher_gamelog") as real, \
             patch.object(espn_client, "get_pitcher_stats",
                          return_value=synth_rows) as synth:
            out = mlb_starters._pitcher_gamelog_or_synth("mlb", "42", None, 2024)
        self.assertEqual(out, synth_rows)
        real.assert_not_called()                      # StatsAPI path skipped
        synth.assert_called_once_with("mlb", "42", season=2024)

    def test_synth_exception_returns_empty(self):
        with patch.object(mlb_starters, "get_pitcher_gamelog", return_value=[]), \
             patch.object(espn_client, "get_pitcher_stats",
                          side_effect=RuntimeError("boom")):
            out = mlb_starters._pitcher_gamelog_or_synth("mlb", "42", "Ace", 2024)
        self.assertEqual(out, [])

    def test_real_raise_falls_back_to_synth(self):
        # StatsAPI down (get_pitcher_gamelog raises) must fall through to synth,
        # not propagate — the fail-open contract get_pitcher_stats provided.
        synth_rows = [{"IP": 5.0, "K": 6.0, "ER": 3.0}]
        with patch.object(mlb_starters, "get_pitcher_gamelog",
                          side_effect=RuntimeError("statsapi 503")), \
             patch.object(espn_client, "get_pitcher_stats",
                          return_value=synth_rows) as synth:
            out = mlb_starters._pitcher_gamelog_or_synth("mlb", "42", "Ace", 2024)
        self.assertEqual(out, synth_rows)
        synth.assert_called_once_with("mlb", "42", season=2024)

    def test_real_raise_and_synth_raise_returns_empty(self):
        with patch.object(mlb_starters, "get_pitcher_gamelog",
                          side_effect=RuntimeError("statsapi 503")), \
             patch.object(espn_client, "get_pitcher_stats",
                          side_effect=RuntimeError("espn down")):
            out = mlb_starters._pitcher_gamelog_or_synth("mlb", "42", "Ace", 2024)
        self.assertEqual(out, [])


def _real_rows(n=3, start_day=20):
    """Dated real pitcher rows (newest-first), shaped like get_pitcher_gamelog."""
    return [{"IP": 6.0, "K": 7.0, "ER": 2.0,
             "game_date": f"2024-07-{start_day - i:02d}",
             "opponent": "Yankees", "is_home": True, "completed": True}
            for i in range(n)]


class _SqliteBackend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        gamelog_store.create_all()
        gamelog_store._KEY_LOCKS.clear()

    def tearDown(self):
        db_store.configure_engine(None)
        gamelog_store._KEY_LOCKS.clear()


class NameThreadingTests(_SqliteBackend, unittest.TestCase):
    """player_name flows through the durable store to the real StatsAPI log, while
    omitting it stays byte-identical to the synthesized-splits behavior."""

    def test_named_fetch_stores_real_dated_pitcher_log(self):
        real = _real_rows()
        with patch.object(espn_client, "get_athlete_gamelog", return_value=[]), \
             patch.object(mlb_starters, "get_pitcher_gamelog", return_value=real):
            served = gamelog_store.get_gamelog("baseball", "mlb", "ace1",
                                               player_name="Ace McReal")
        self.assertEqual(len(served), 3)
        self.assertTrue(all(r.get("game_date") for r in served))   # dated
        self.assertEqual(served[0]["game_date"], "2024-07-20")     # newest-first
        with db_store.get_engine().connect() as conn:
            meta = gamelog_store._read_meta(conn, "baseball", "mlb", "ace1", 0)
        self.assertEqual(meta["player_type"], "pitcher")

    def test_omitted_name_uses_synth_unchanged(self):
        with patch.object(espn_client, "get_athlete_gamelog", return_value=[]), \
             patch.object(espn_client, "get_pitcher_stats",
                          return_value=[{"IP": 5.0, "K": 6.0, "ER": 3.0}]):
            served = gamelog_store.get_gamelog("baseball", "mlb", "syn1")
        self.assertEqual(len(served), 1)
        self.assertEqual(served[0]["IP"], 5.0)
        with db_store.get_engine().connect() as conn:
            meta = gamelog_store._read_meta(conn, "baseball", "mlb", "syn1", 0)
        self.assertEqual(meta["player_type"], "pitcher")

    def test_migration_synth_dateless_replaced_by_real_dated(self):
        # First fetch: synthesized (dateless) rows persisted.
        with patch.object(espn_client, "get_athlete_gamelog", return_value=[]), \
             patch.object(espn_client, "get_pitcher_stats",
                          return_value=[{"IP": 5.0, "K": 6.0, "ER": 3.0}]):
            gamelog_store.get_gamelog("baseball", "mlb", "mig1", ttl_hours=0)
        with db_store.get_engine().connect() as conn:
            stored = gamelog_store._read_rows(conn, "pitcher", "mig1", 0)
        self.assertTrue(stored and stored[0].get("game_date") is None)  # dateless

        # Re-fetch WITH a name: real dated rows swap in (different season vs None).
        real = _real_rows()
        with patch.object(espn_client, "get_athlete_gamelog", return_value=[]), \
             patch.object(mlb_starters, "get_pitcher_gamelog", return_value=real):
            served = gamelog_store.get_gamelog("baseball", "mlb", "mig1",
                                               ttl_hours=0, player_name="Ace")
        self.assertEqual(len(served), 3)
        self.assertTrue(all(r.get("game_date") for r in served))

    def test_get_player_stat_history_threads_name_end_to_end(self):
        # Whole chain: name -> get_athlete_id -> get_gamelog(player_name=...) ->
        # helper -> get_pitcher_gamelog -> dated rows -> pitcher_outs values.
        gamelog_store.seed_athlete_id("baseball", "mlb", "Ace McReal", "ace9")
        real = _real_rows()   # IP 6.0 each -> 18 outs
        # P2.5b makes baseball history WAREHOUSE-ONLY on the live path; this ESPN/
        # StatsAPI pitcher name-threading chain is now reached via allow_warehouse=False
        # (the parity/backtest seam), which the machinery still serves.
        with patch.object(espn_client, "get_athlete_gamelog", return_value=[]), \
             patch.object(mlb_starters, "get_pitcher_gamelog", return_value=real):
            hist = espn_client.get_player_stat_history(
                "baseball", "mlb", "Ace McReal", "pitcher_outs", n=20,
                allow_warehouse=False)
        self.assertTrue(hist["found"])
        self.assertEqual(hist["stat_label"], "IP")
        self.assertEqual(hist["values"], [18.0, 18.0, 18.0])   # ip_to_outs(6.0)
        self.assertTrue(all(hist["game_dates"]))


class ShouldReplaceGuardTests(unittest.TestCase):
    """The clobber guard: a dateless synth refetch must never overwrite a stored
    DATED real log, while every legitimate refresh/migration still replaces."""

    def _dateless(self, n=3):
        return [{"IP": 5.0, "K": 6.0, "ER": 3.0} for _ in range(n)]

    def test_dateless_new_does_not_clobber_dated_stored(self):
        # The core §2.3 defect: synth fallback (season None) must not delete the
        # dated real log real-line calibration + as-of slicing depend on.
        self.assertFalse(
            gamelog_store._should_replace(_real_rows(3), self._dateless(3)))

    def test_synth_to_real_migration_replaces(self):
        # stored dateless -> new dated: different season (None vs 2024) => replace.
        self.assertTrue(
            gamelog_store._should_replace(self._dateless(3), _real_rows(3)))

    def test_both_dateless_refresh_replaces(self):
        # Neither side dated: falls through to the completed-count compare (0>=0).
        self.assertTrue(
            gamelog_store._should_replace(self._dateless(2), self._dateless(3)))

    def test_same_season_fewer_completed_does_not_replace(self):
        # Transient partial fetch (fewer finals, same season) must be rejected.
        self.assertFalse(
            gamelog_store._should_replace(_real_rows(3), _real_rows(2)))

    def test_same_season_more_completed_replaces(self):
        self.assertTrue(
            gamelog_store._should_replace(_real_rows(3), _real_rows(4)))

    def test_different_dated_season_replaces(self):
        next_season = [dict(r, game_date=r["game_date"].replace("2024", "2025"))
                       for r in _real_rows(2)]
        self.assertTrue(
            gamelog_store._should_replace(_real_rows(3), next_season))


class PlayerStatSeriesInProgressTests(unittest.TestCase):
    """backtest._player_stat_series must exclude an in-progress (completed=False)
    game so a same-day partial box score isn't graded as a final observation."""

    def test_in_progress_game_excluded(self):
        import backtest
        gl = [
            {"game_date": "2024-07-24", "ER": 3.0, "IP": 6.0, "completed": False},
            {"game_date": "2024-07-20", "ER": 2.0, "IP": 6.0, "completed": True},
            {"game_date": "2024-07-15", "ER": 1.0, "IP": 6.0, "completed": True},
        ]
        with patch.object(backtest, "cached_athlete_id", return_value="42"), \
             patch.object(backtest, "cached_gamelog", return_value=gl):
            out = backtest._player_stat_series("baseball", "mlb", "Ace",
                                               "pitcher_earned_runs")
        self.assertEqual(out, [("2024-07-15", 1.0), ("2024-07-20", 2.0)])


class EspnCacheNonSqlPitcherFallbackTests(unittest.TestCase):
    """The file-cache (non-SQL) branch of cached_gamelog must thread player_name
    into the fail-open helper — the one gamelog chokepoint the SQL-backed store
    tests never exercise."""

    def test_non_sql_branch_threads_player_name(self):
        import os
        import tempfile
        import espn_cache
        real = _real_rows()
        path = os.path.join(tempfile.mkdtemp(), "gl.json")
        with patch.object(espn_cache, "_sql", return_value=False), \
             patch.object(espn_cache, "_cache_key", return_value=path), \
             patch.object(espn_cache, "_read_cache_file", return_value=None), \
             patch.object(espn_cache, "get_athlete_gamelog", return_value=[]), \
             patch.object(mlb_starters, "_pitcher_gamelog_or_synth",
                          return_value=real) as helper:
            out = espn_cache.cached_gamelog("baseball", "mlb", "p1",
                                            player_name="Ace McReal")
        self.assertEqual(out, real)
        helper.assert_called_once_with("mlb", "p1", "Ace McReal", None)


class PitcherRealLineJoinTests(unittest.TestCase):
    """Dated pitcher logs now join to book lines (dateless synth never did)."""

    def _book_row(self, gd="2024-07-24", line=2.5):
        return {"sport_key": "baseball_mlb", "game_date": gd,
                "commence_time": f"{gd}T23:00:00Z", "event_id": "e1",
                "home_team": "Reds", "away_team": "Guardians",
                "player": "A. Pitcher", "player_mlb_id": "123",
                "prop_key": "pitcher_earned_runs",
                "line": line, "over_price": -110, "under_price": -110}

    def _join(self, book_rows, gamelog):
        import book_line_calibration as blc
        # MLB calibration is warehouse-only (no ESPN): source the per-game log from
        # get_calib_gamelog.
        with patch("mlb_warehouse.get_calib_gamelog", return_value=gamelog):
            return blc.join_book_lines_to_actuals(book_rows, "baseball", "mlb")

    def test_dated_pitcher_log_joins(self):
        gl = [{"game_date": f"2024-07-{d:02d}T23:00:00Z", "ER": 2.0, "IP": 6.0}
              for d in range(13, 24)]                         # 11 prior games
        gl = [{"game_date": "2024-07-24T23:00:00Z", "ER": 3.0, "IP": 6.0}] + gl
        out = self._join([self._book_row()], gl)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["stat_label"], "ER")
        self.assertEqual(out[0]["actual"], 3.0)
        self.assertGreaterEqual(len(out[0]["prior_games"]), 10)

    def test_dateless_synth_log_does_not_join(self):
        # Synthesized rows carry no game_date -> nothing to match a dated book line.
        synth = [{"ER": 2.0, "IP": 6.0} for _ in range(12)]
        self.assertEqual(self._join([self._book_row()], synth), [])


if __name__ == "__main__":
    unittest.main()
