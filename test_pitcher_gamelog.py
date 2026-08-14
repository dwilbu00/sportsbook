"""Pitcher-gamelog store guarantees that survive the P4b ESPN teardown.

MLB is warehouse-only post-P4b (the ESPN-synth pitcher path + get_pitcher_gamelog were
removed), so these cover: the clobber guard (gamelog_store._should_replace), the
completed-game filter in backtest._player_stat_series (now warehouse-sourced for MLB),
and the real-line join onto dated pitcher logs.

Hermetic: the warehouse readers + SQL store (in-memory SQLite) are mocked/stubbed — no
live network or Azure I/O.
"""
import unittest
from unittest.mock import patch

import db_store
import gamelog_store
import mlb_starters


def _real_rows(n=3, start_day=20):
    """Dated real pitcher rows (newest-first), the reduced per-game shape."""
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
        # P4: MLB player logs come from the warehouse (get_calib_gamelog); the shared
        # (date,value) tail still drops the in-progress (completed=False) game.
        import backtest
        gl = [
            {"game_date": "2024-07-24", "ER": 3.0, "IP": 6.0, "completed": False},
            {"game_date": "2024-07-20", "ER": 2.0, "IP": 6.0, "completed": True},
            {"game_date": "2024-07-15", "ER": 1.0, "IP": 6.0, "completed": True},
        ]
        with patch.object(mlb_starters, "resolve_mlbam_id",
                          return_value=(543037, True)), \
             patch("mlb_warehouse.get_calib_gamelog", return_value=gl), \
             patch("mlb_warehouse._current_season", return_value=2024):
            out = backtest._player_stat_series("baseball", "mlb", "Ace",
                                               "pitcher_earned_runs")
        self.assertEqual(out, [("2024-07-15", 1.0), ("2024-07-20", 2.0)])


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
