"""Tests for the durable rolling ESPN gamelog store (gamelog_store, Phase C),
exercised against in-memory SQLite so pymssql and the live Azure DB are never
touched. Mirrors test_db_store.py::_SqliteBackend: the SQL backend is enabled via
configure_engine (tests never set SQL_* env), and tearDown clears the override.
"""

import threading
import time
import unittest
from unittest.mock import patch

import db_store
import gamelog_store
import espn_client


def _nba_row(game_date, pts=25.0, reb=8.0, ast=6.0, minutes=34.0):
    return {"MIN": minutes, "FG": 9.0, "3PT": 3.0, "FT": 4.0, "REB": reb,
            "AST": ast, "BLK": 1.0, "STL": 2.0, "PF": 3.0, "TO": 2.0,
            "PTS": pts, "opponent": "Celtics", "is_home": True, "team_id": "2",
            "game_date": game_date, "completed": True}


def _nfl_row(game_date, yds=88.0, td=1.0, opp="Bears", home=True, team="3",
             completed=True):
    """A realistic ESPN NFL gamelog row: ONE position-dependent row per game with
    ~18 labels; the store keeps only the two the app reads (YDS, TD)."""
    return {"CMP": 22.0, "ATT": 33.0, "YDS": yds, "CMP%": 66.7, "AVG": 4.0,
            "TD": td, "INT": 0.0, "LNG": 40.0, "SACK": 1.0, "RTG": 105.0,
            "QBR": 70.0, "CAR": 3.0, "opponent": opp, "is_home": home,
            "team_id": team, "game_date": game_date, "completed": completed}


class _Backend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        gamelog_store.create_all()
        gamelog_store._KEY_LOCKS.clear()

    def tearDown(self):
        db_store.configure_engine(None)   # → enabled() False (no SQL_* env)
        gamelog_store._KEY_LOCKS.clear()


class RoundTripTests(_Backend, unittest.TestCase):
    # (MLB batter/pitcher roundtrips removed in P4b — MLB is warehouse-only; the
    # generic store round-trip is covered by the NBA/NFL cases below.)

    def test_nba_roundtrip(self):
        rows = [_nba_row("2026-01-15T00:00:00.000+00:00")]
        with patch.object(espn_client, "get_athlete_gamelog", return_value=rows):
            gamelog_store.get_gamelog("basketball", "nba", "7")
        with patch.object(espn_client, "get_athlete_gamelog") as mock:
            served = gamelog_store.get_gamelog("basketball", "nba", "7")
            mock.assert_not_called()
        self.assertEqual(set(served[0]),
                         {"MIN", "PTS", "REB", "AST", "opponent", "is_home",
                          "team_id", "game_date", "completed"})
        self.assertEqual(served[0]["PTS"], 25.0)

    def test_nfl_roundtrip_reduced_shape(self):
        rows = [_nfl_row("2025-09-14T18:00:00.000+00:00", yds=305.0, td=2.0),
                _nfl_row("2025-09-07T18:00:00.000+00:00", yds=210.0, td=1.0)]
        with patch.object(espn_client, "get_athlete_gamelog", return_value=rows):
            first = gamelog_store.get_gamelog("football", "nfl", "n1")
            self.assertEqual(len(first), 2)          # fetch path returns raw rows
        with patch.object(espn_client, "get_athlete_gamelog") as mock:
            served = gamelog_store.get_gamelog("football", "nfl", "n1")
            mock.assert_not_called()                  # 2nd served from SQL
        self.assertEqual(len(served), 2)
        # Only the two consumer-read labels survive; the ~16 others are dropped.
        self.assertEqual(set(served[0]),
                         {"YDS", "TD", "opponent", "is_home", "team_id",
                          "game_date", "completed"})
        self.assertEqual(served[0]["YDS"], 305.0)
        self.assertEqual(served[0]["TD"], 2.0)


class TtlGateTests(_Backend, unittest.TestCase):
    # NBA fixtures (the store is NBA/NFL-only post-P4b; MLB is warehouse-only).

    def test_fresh_meta_serves_from_sql(self):
        rows = [_nba_row("2026-01-15T00:00:00.000+00:00")]
        with patch.object(espn_client, "get_athlete_gamelog",
                          return_value=rows) as mock:
            gamelog_store.get_gamelog("basketball", "nba", "1")
            gamelog_store.get_gamelog("basketball", "nba", "1")
            self.assertEqual(mock.call_count, 1)   # 2nd served from SQL

    def test_stale_ttl_triggers_refetch(self):
        rows = [_nba_row("2026-01-15T00:00:00.000+00:00")]
        with patch.object(espn_client, "get_athlete_gamelog",
                          return_value=rows) as mock:
            gamelog_store.get_gamelog("basketball", "nba", "1")
            gamelog_store.get_gamelog("basketball", "nba", "1", ttl_hours=0)
            self.assertEqual(mock.call_count, 2)

    def test_empty_fetch_keeps_negative_ttl(self):
        with patch.object(espn_client, "get_athlete_gamelog",
                          return_value=[]) as gl:
            out = gamelog_store.get_gamelog("basketball", "nba", "5")
            self.assertEqual(out, [])
            # Second call within the negative TTL serves [] from the fast path
            # (player_type is None on a not-found meta) without a re-fetch/crash.
            out2 = gamelog_store.get_gamelog("basketball", "nba", "5")
            self.assertEqual(out2, [])
            self.assertEqual(gl.call_count, 1)
        # A not-found row exists with game_count 0 so the negative TTL governs.
        with db_store.get_engine().connect() as conn:
            meta = gamelog_store._read_meta(conn, "basketball", "nba", "5", 0)
        self.assertEqual(meta["game_count"], 0)


class ClassificationTests(_Backend, unittest.TestCase):

    def _player_type(self, aid, sport="basketball", league="nba"):
        with db_store.get_engine().connect() as conn:
            meta = gamelog_store._read_meta(conn, sport, league, aid, 0)
        return meta["player_type"] if meta else None

    # (MLB batter/pitcher classification removed in P4b — _classify is NBA/NFL-only.)

    def test_nfl_classified_as_nfl(self):
        with patch.object(espn_client, "get_athlete_gamelog",
                          return_value=[_nfl_row("2025-09-14T18:00:00Z")]):
            gamelog_store.get_gamelog("football", "nfl", "n2")
        self.assertEqual(self._player_type("n2", "football", "nfl"), "nfl")

    def test_tableless_sport_passthrough_no_persist(self):
        # A sport with no fact table (e.g. NHL) still passes through to direct
        # ESPN with no persistence -- nothing regresses when SQL is on.
        rows = [{"G": 1.0, "A": 2.0, "opponent": "Bruins", "is_home": True,
                 "team_id": "3", "game_date": "2026-01-14T18:00:00Z"}]
        with patch.object(espn_client, "get_athlete_gamelog", return_value=rows):
            out = gamelog_store.get_gamelog("hockey", "nhl", "h1")
        self.assertEqual(out, rows)            # passthrough returns raw
        with db_store.get_engine().connect() as conn:
            meta = gamelog_store._read_meta(conn, "hockey", "nhl", "h1", 0)
        self.assertIsNone(meta)


class AthleteIdCacheTests(_Backend, unittest.TestCase):

    def test_lookup_caches_and_disambiguates_by_team(self):
        athlete = {"id": "123", "name": "Will Smith", "team_id": "5"}
        with patch.object(espn_client, "search_athlete",
                          return_value=athlete) as mock:
            a1 = gamelog_store.get_athlete_id("baseball", "mlb", "Will Smith")
            a2 = gamelog_store.get_athlete_id("baseball", "mlb", "Will Smith")
            self.assertEqual(mock.call_count, 1)          # 2nd from cache
        self.assertEqual(a1["id"], "123")
        self.assertEqual(a1["team_id"], "5")
        self.assertEqual(a2["id"], "123")
        # A different matchup (team_ids) is a distinct cache key → new lookup.
        other = {"id": "456", "name": "Will Smith", "team_id": "9"}
        with patch.object(espn_client, "search_athlete",
                          return_value=other) as mock:
            a3 = gamelog_store.get_athlete_id("baseball", "mlb", "Will Smith",
                                              team_ids=[9])
            self.assertEqual(mock.call_count, 1)
        self.assertEqual(a3["id"], "456")

    def test_not_found_returns_none(self):
        with patch.object(espn_client, "search_athlete", return_value=None):
            self.assertIsNone(
                gamelog_store.get_athlete_id("baseball", "mlb", "Nobody"))

    def test_seed_pins_id_and_serves_without_search(self):
        gamelog_store.seed_athlete_id("baseball", "mlb", "Mookie Betts", "808")
        with patch.object(espn_client, "search_athlete") as mock:
            got = gamelog_store.get_athlete_id("baseball", "mlb", "Mookie Betts")
            mock.assert_not_called()          # seeded → no lossy search
        self.assertEqual(got["id"], "808")

    def test_seed_overwrites_existing_key(self):
        gamelog_store.seed_athlete_id("baseball", "mlb", "Dupe", "1")
        gamelog_store.seed_athlete_id("baseball", "mlb", "Dupe", "2")
        with patch.object(espn_client, "search_athlete") as mock:
            got = gamelog_store.get_athlete_id("baseball", "mlb", "Dupe")
            mock.assert_not_called()
        self.assertEqual(got["id"], "2")      # authoritative overwrite

    def test_reseed_is_surgical_keeps_surrogate_id(self):
        # WS15: re-seeding the same natural key is an UPDATE-in-place, not a
        # delete + insert — the surrogate id is preserved and no duplicate row
        # is left behind.
        from sqlalchemy import select

        def _row():
            with db_store.get_engine().connect() as conn:
                return [dict(r._mapping) for r in conn.execute(
                    select(gamelog_store.athlete_id_cache).where(
                        gamelog_store.athlete_id_cache.c.player_name_lower
                        == "dupe"))]

        gamelog_store.seed_athlete_id("baseball", "mlb", "Dupe", "1")
        before = _row()
        self.assertEqual(len(before), 1)
        gamelog_store.seed_athlete_id("baseball", "mlb", "Dupe", "2")
        after = _row()
        self.assertEqual(len(after), 1)                 # no orphan/dup row
        self.assertEqual(after[0]["id"], before[0]["id"])   # surrogate id stable
        self.assertEqual(after[0]["athlete_id"], "2")       # data updated

    def test_seed_noop_on_falsy_id(self):
        gamelog_store.seed_athlete_id("baseball", "mlb", "Ghost", None)
        with patch.object(espn_client, "search_athlete",
                          return_value={"id": "9", "name": "Ghost",
                                        "team_id": "1"}) as mock:
            got = gamelog_store.get_athlete_id("baseball", "mlb", "Ghost")
            mock.assert_called_once()          # nothing seeded → real lookup
        self.assertEqual(got["id"], "9")


class ConcurrencyTests(_Backend, unittest.TestCase):

    def test_same_athlete_fetched_once(self):
        rows = [_nba_row("2026-01-15T00:00:00.000+00:00")]
        calls = []

        def slow(*a, **k):
            calls.append(1)
            time.sleep(0.05)
            return rows

        barrier = threading.Barrier(2)
        results = []

        def worker():
            barrier.wait()
            results.append(gamelog_store.get_gamelog("basketball", "nba", "z1"))

        with patch.object(espn_client, "get_athlete_gamelog", side_effect=slow):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(calls), 1)         # per-key lock → one fetch
        self.assertEqual(len(results), 2)
        # No duplicate rows from a double insert.
        with db_store.get_engine().connect() as conn:
            stored = gamelog_store._read_rows(conn, "nba", "z1", 0)
        self.assertEqual(len(stored), 1)


class DispatchTests(_Backend, unittest.TestCase):

    def test_cached_gamelog_routes_to_sql(self):
        import espn_cache
        rows = [_nba_row("2026-01-15T00:00:00.000+00:00")]
        with patch.object(espn_client, "get_athlete_gamelog", return_value=rows):
            espn_cache.cached_gamelog("basketball", "nba", "d1")
        with db_store.get_engine().connect() as conn:
            meta = gamelog_store._read_meta(conn, "basketball", "nba", "d1", 0)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["player_type"], "nba")

    def test_get_player_stat_history_routes_to_sql(self):
        # NBA still uses the durable SQL gamelog store (MLB is warehouse-only post-P4b).
        rows = [_nba_row("2026-01-15T00:00:00.000+00:00", pts=25.0),
                _nba_row("2026-01-13T00:00:00.000+00:00", pts=18.0)]
        athlete = {"id": "555", "name": "Star", "team_id": "10"}
        with patch.object(espn_client, "search_athlete", return_value=athlete), \
             patch.object(espn_client, "get_athlete_gamelog",
                          return_value=rows) as gl:
            hist = espn_client.get_player_stat_history(
                "basketball", "nba", "Star", "player_points", n=20)
            # Second call serves from SQL (no second ESPN gamelog fetch).
            espn_client.get_player_stat_history(
                "basketball", "nba", "Star", "player_points", n=20)
            self.assertEqual(gl.call_count, 1)
        self.assertTrue(hist["found"])
        self.assertEqual(hist["values"], [25.0, 18.0])
        self.assertEqual(hist["team_id"], "10")

    def test_seed_athlete_id_routes_to_sql(self):
        import espn_cache
        espn_cache.seed_athlete_id("baseball", "mlb", "Seeded Star", "4242")
        # cached_athlete_id (SQL path) now resolves without a lossy search.
        with patch.object(espn_client, "search_athlete") as mock:
            aid = espn_cache.cached_athlete_id("baseball", "mlb", "Seeded Star")
            mock.assert_not_called()
        self.assertEqual(aid, "4242")


class SchemaParityTests(_Backend, unittest.TestCase):
    """Guard the fact-table + bookkeeping columns against drift (sql/schema.sql
    mirrors these for the hand-run Azure DDL)."""

    def test_batter_columns(self):
        self.assertEqual(
            {c.name for c in gamelog_store.mlb_batter_gamelog.columns},
            set(gamelog_store._FACT_META_COLS) | set(gamelog_store._BATTER_STATS))

    def test_pitcher_columns(self):
        self.assertEqual(
            {c.name for c in gamelog_store.mlb_pitcher_gamelog.columns},
            set(gamelog_store._FACT_META_COLS) | set(gamelog_store._PITCHER_STATS))

    def test_nba_columns(self):
        self.assertEqual(
            {c.name for c in gamelog_store.nba_gamelog.columns},
            set(gamelog_store._FACT_META_COLS) | set(gamelog_store._NBA_STATS))

    def test_nfl_columns(self):
        self.assertEqual(
            {c.name for c in gamelog_store.nfl_gamelog.columns},
            set(gamelog_store._FACT_META_COLS) | set(gamelog_store._NFL_STATS))

    def test_meta_columns(self):
        self.assertEqual(
            {c.name for c in gamelog_store.gamelog_fetch_meta.columns},
            {"id", "sport", "league", "athlete_id", "season_bucket",
             "player_type", "last_fetched_at", "game_count"})

    def test_athlete_id_cache_columns(self):
        self.assertEqual(
            {c.name for c in gamelog_store.athlete_id_cache.columns},
            {"id", "sport", "league", "player_name_lower", "team_key",
             "athlete_id", "name", "team_id", "fetched_at"})


if __name__ == "__main__":
    unittest.main()
