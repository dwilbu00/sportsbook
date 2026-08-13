"""Tests for the SQL-durable Platt recalibration overlay (accuracy-roadmap 0.1b)
and doubleheader-safe outcome resolution (0.2).

The durable store is Azure SQL in prod; here it runs against an in-memory SQLite
engine (db_store.configure_engine("sqlite://")). The local git-committed
recalibration file is the seed/prior the SQL overlay merges onto, so CALIB_DIR is
redirected to a temp dir per test (empty = no prior). The MLB statsapi client
(_get) is patched so nothing touches live services — important because
.streamlit/secrets.toml is present locally.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import db_store
import mlb_starters
import recalibration

_VALID_FIT = {"batter_hits": {"a": 0.5, "b": 0.1, "n_fit": 120, "validated": True}}


class SqlRecalibrationTests(unittest.TestCase):
    """The SQL overlay round-trips validated fits and merges onto the local seed.

    Runs against an in-memory SQLite engine; CALIB_DIR is a temp dir so the local
    seed (the prior) is controlled per test — empty means no prior to blend."""

    def setUp(self):
        recalibration._LOAD_CACHE.pop("baseball_mlb", None)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        self.addCleanup(db_store.configure_engine, None)
        self.addCleanup(recalibration._LOAD_CACHE.pop, "baseball_mlb", None)

    def _write_seed(self, props, ts="2026-07-20T00:00:00+00:00"):
        path = os.path.join(self._tmp.name, "recalibration_baseball_mlb.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fit_timestamp": ts, "props": props}, f)

    def test_save_then_load_sql_round_trip(self):
        # Empty seed dir → no prior to blend → the SQL fit applies verbatim.
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name):
            recalibration.save_recalibration("baseball_mlb", _VALID_FIT)
            props = recalibration.load_recalibration("baseball_mlb")
        self.assertIn("batter_hits", props)
        self.assertEqual(props["batter_hits"]["a"], 0.5)
        # A runtime SQL refit persists to SQL only, never the committed seed file.
        self.assertFalse(os.path.exists(
            os.path.join(self._tmp.name, "recalibration_baseball_mlb.json")))

    def test_unvalidated_fit_is_not_applied(self):
        fit = {"pitcher_strikeouts": {"a": 1.0, "b": 0.0, "validated": False}}
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name):
            recalibration.save_recalibration("baseball_mlb", fit)
            props = recalibration.load_recalibration("baseball_mlb")
        self.assertEqual(props, {})

    def test_sql_empty_falls_back_to_local_seed(self):
        # An empty SQL overlay must not hide the git-committed seed.
        self._write_seed(_VALID_FIT)
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name):
            props = recalibration.load_recalibration("baseball_mlb")
        self.assertIn("batter_hits", props)

    def test_load_degrades_to_cache_on_db_error(self):
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name):
            recalibration.save_recalibration("baseball_mlb", _VALID_FIT)
            first = recalibration.load_recalibration("baseball_mlb")
            self.assertIn("batter_hits", first)
            # Expire the TTL, then make the DB read fail: serve the last cache.
            recalibration._LOAD_CACHE["baseball_mlb"]["fetched_at"] = 0
            with patch.object(recalibration._db, "load_recal",
                              side_effect=Exception("db down")):
                degraded = recalibration.load_recalibration("baseball_mlb")
        self.assertEqual(degraded, first)

    def test_malformed_props_degrades_to_empty_without_raising(self):
        # A bad `props` shape from the store must degrade to {} on the (unwrapped)
        # free-loop load path (props.py:348), never raise.
        for cfg in ({"props": {"batter_hits": None}},
                    {"props": [1, 2, 3]},
                    {"props": {"batter_hits": "oops"}}):
            recalibration._LOAD_CACHE.pop("baseball_mlb", None)
            with patch.object(recalibration, "CALIB_DIR", self._tmp.name), \
                 patch.object(recalibration._db, "load_recal", return_value=cfg):
                self.assertEqual(
                    recalibration.load_recalibration("baseball_mlb"), {})

    def test_seed_save_stays_local_only(self):
        # to_blob=False (offline seeding) writes the local seed and never touches
        # the SQL overlay, keeping the committed file a pristine prior.
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name), \
             patch.object(recalibration._db, "save_recal") as save_recal:
            recalibration.save_recalibration("baseball_mlb", _VALID_FIT,
                                             to_blob=False)
        save_recal.assert_not_called()
        self.assertTrue(os.path.exists(
            os.path.join(self._tmp.name, "recalibration_baseball_mlb.json")))

    def test_sql_fit_blends_toward_seed_and_keeps_seed_only_props(self):
        # Seed holds two props; the SQL overlay has a fit for only one. The fit
        # blends toward its seed prior; the seed-only prop survives untouched
        # (the per-key overlay, not the old all-or-nothing fallback).
        seed = {
            "batter_hits": {"a": 0.4, "b": 0.2, "n_fit": 100, "validated": True},
            "pitcher_outs": {"a": 0.6, "b": -0.1, "n_fit": 80, "validated": True},
        }
        self._write_seed(seed)
        sql_fit = {"batter_hits":
                   {"a": 0.8, "b": 0.0, "n_fit": 300, "validated": True}}
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name):
            recalibration.save_recalibration("baseball_mlb", sql_fit)  # SQL only
            props = recalibration.load_recalibration("baseball_mlb")
        # Seed-only prop passes through unchanged...
        self.assertEqual(props["pitcher_outs"]["a"], 0.6)
        # ...and the fit prop is a shrinkage blend strictly between seed and loop.
        self.assertIn("blend_weight", props["batter_hits"])
        self.assertGreater(props["batter_hits"]["a"], 0.4)
        self.assertLess(props["batter_hits"]["a"], 0.8)


class StorageBackendStringTests(unittest.TestCase):
    """With no SQL configured, the human-readable backend strings name local."""

    def test_local_storage_strings_when_no_sql(self):
        import warehouse
        with patch.object(recalibration, "_sql", return_value=False):
            self.assertEqual(recalibration.prediction_log_storage(), "Local cache")
        with patch.object(warehouse, "_sql", return_value=False):
            self.assertEqual(warehouse.storage_backend(), "Local warehouse/")


class RefitGateTests(unittest.TestCase):
    """maintain_sport gate keyed on the durable fit_timestamp, not file mtime."""

    def _run(self, last_fit_ts, resolved_since):
        with patch.object(recalibration, "resolve_pending_outcomes", return_value=0), \
             patch.object(recalibration, "_load_recal_cached",
                          return_value=(last_fit_ts, {})), \
             patch.object(recalibration, "_count_resolved_since",
                          return_value=resolved_since), \
             patch.object(recalibration, "compact_prediction_log", return_value=0), \
             patch.object(recalibration, "refit_sport",
                          return_value={"p": (1, 0, 90)}) as refit:
            recalibration.maintain_sport("baseball_mlb")
        return refit.called

    def test_no_fit_yet_triggers_refit(self):
        self.assertTrue(self._run(None, 0))

    def test_fresh_fit_skips_refit(self):
        self.assertFalse(self._run(time.time(), 999))

    def test_stale_fit_with_enough_new_triggers_refit(self):
        old = time.time() - (recalibration.MIN_REFIT_INTERVAL_HOURS + 1) * 3600
        self.assertTrue(self._run(old, recalibration.MIN_NEW_FOR_REFIT))

    def test_stale_fit_without_enough_new_skips_refit(self):
        old = time.time() - (recalibration.MIN_REFIT_INTERVAL_HOURS + 1) * 3600
        self.assertFalse(self._run(old, recalibration.MIN_NEW_FOR_REFIT - 1))


class DoubleheaderPickTests(unittest.TestCase):
    def test_pick_nearest_commence(self):
        cands = [("2024-07-04T17:10:00Z", 0), ("2024-07-04T23:10:00Z", 1)]
        self.assertEqual(
            recalibration._pick_candidate(cands, "2024-07-04T23:05:00Z"), 1)
        self.assertEqual(
            recalibration._pick_candidate(cands, "2024-07-04T17:30:00Z"), 0)

    def test_date_only_or_no_commence_falls_back_first(self):
        cands = [("2024-07-04", 0), ("2024-07-04", 1)]
        self.assertEqual(recalibration._pick_candidate(cands, None), 0)
        self.assertEqual(
            recalibration._pick_candidate(cands, "2024-07-04T23:00:00Z"), 0)

    def test_doubleheader_disambiguated_via_statsapi_not_espn(self):
        # MLB doubleheaders are disambiguated by the statsapi hard-ID (gamePk) path
        # (mlb_starters.resolve_player_game_stat via _resolve_mlb_actual), NOT the
        # ESPN gamelog — MLB grading is warehouse+statsapi only in P6, so ESPN is
        # never consulted (the statsapi value already pins the correct game).
        row = {
            "ts": "2024-07-04T10:00:00Z", "sport_key": "baseball_mlb",
            "prop_key": "batter_hits", "player": "Player One",
            "game_date": "2024-07-04", "commence_time": "2024-07-04T23:05:00Z",
            "line": 0.5, "resolved": False,
        }

        def mutate(mutator, where=None):
            return mutator([row])

        with patch.object(recalibration, "_read_log", return_value=[row]), \
             patch.object(recalibration, "_resolve_mlb_actual",
                          return_value=2.0) as sa, \
             patch("espn_cache.cached_gamelog") as esp, \
             patch.object(recalibration, "mutate_prediction_log", side_effect=mutate):
            resolved = recalibration.resolve_pending_outcomes("baseball_mlb")
        self.assertEqual(resolved, 1)
        self.assertEqual(row["actual"], 2.0)   # statsapi nightcap value
        self.assertEqual(row["outcome"], 1)
        sa.assert_called()                     # statsapi hard-ID path used
        esp.assert_not_called()                # ESPN never consulted for MLB

    def test_statsapi_resolves_when_espn_unavailable(self):
        # The hard-ID path must run independently of ESPN: a player ESPN can't
        # name-match (aid=None) is still graded via statsapi.
        row = {
            "ts": "2024-07-04T10:00:00Z", "sport_key": "baseball_mlb",
            "prop_key": "pitcher_strikeouts", "player": "Ace Pitcher",
            "game_date": "2024-07-04", "commence_time": "2024-07-04T23:05:00Z",
            "line": 6.5, "resolved": False,
        }

        def mutate(mutator, where=None):
            return mutator([row])

        with patch.object(recalibration, "_read_log", return_value=[row]), \
             patch.object(recalibration, "_resolve_mlb_actual", return_value=8.0), \
             patch("espn_cache.cached_athlete_id", return_value=None), \
             patch.object(recalibration, "mutate_prediction_log", side_effect=mutate):
            resolved = recalibration.resolve_pending_outcomes("baseball_mlb")
        self.assertEqual(resolved, 1)
        self.assertEqual(row["actual"], 8.0)
        self.assertEqual(row["outcome"], 1)

    def test_live_game_sentinel_stays_pending_and_skips_espn(self):
        # A statsapi GAME_NOT_FINAL sentinel must keep the bet pending and NOT
        # fall through to the un-gated ESPN partial-stat path.
        with patch.object(recalibration, "_resolve_mlb_actual",
                          return_value=mlb_starters.GAME_NOT_FINAL), \
             patch("espn_cache.cached_gamelog") as mock_gamelog:
            result = recalibration.resolve_one_prop(
                "baseball_mlb", "Live Bat", "batter_hits", 0.5,
                "2024-07-04", "2024-07-04T23:10:00Z")
        self.assertIsNone(result)
        mock_gamelog.assert_not_called()


class StatsapiResolverTests(unittest.TestCase):
    def setUp(self):
        mlb_starters._PLAYER_INDEX_CACHE.clear()
        self.addCleanup(mlb_starters._PLAYER_INDEX_CACHE.clear)
        p1 = patch.object(mlb_starters, "_read_cache", return_value=None)
        p2 = patch.object(mlb_starters, "_write_cache", return_value=None)
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def _fake_get(self, path, params=None, **kw):
        if path == "sports/1/players":
            return {"people": [
                {"id": 100, "fullName": "Test Pitcher",
                 "primaryPosition": {"abbreviation": "P", "type": "Pitcher"}},
                {"id": 200, "fullName": "Test Batter",
                 "primaryPosition": {"abbreviation": "CF", "type": "Outfielder"}},
                {"id": 300, "fullName": "Dup Name",
                 "primaryPosition": {"abbreviation": "P"}},
                {"id": 301, "fullName": "Dup Name",
                 "primaryPosition": {"abbreviation": "1B"}},
            ]}
        if path.startswith("people/") and path.endswith("/stats"):
            return {"stats": [{"splits": [
                {"game": {"gamePk": 1}, "date": "2024-07-04",
                 "stat": {"strikeOuts": 5}},
                {"game": {"gamePk": 2}, "date": "2024-07-04",
                 "stat": {"strikeOuts": 9}},
            ]}]}
        if path == "schedule" and (params or {}).get("date") == "2024-07-04":
            return {"dates": [{"games": [
                {"gamePk": 1, "gameDate": "2024-07-04T17:10:00Z",
                 "status": {"abstractGameState": "Final"}, "teams": {}},
                {"gamePk": 2, "gameDate": "2024-07-04T23:10:00Z",
                 "status": {"abstractGameState": "Final"}, "teams": {}},
            ]}]}
        return {"dates": []}

    def test_find_player_id_unique_and_ambiguous(self):
        with patch.object(mlb_starters, "_get", side_effect=self._fake_get):
            self.assertEqual(
                mlb_starters.find_player_id("Test Pitcher", 2024), (100, True))
            self.assertIsNone(mlb_starters.find_player_id("Dup Name", 2024))

    def test_resolve_picks_nearest_gamepk(self):
        with patch.object(mlb_starters, "_get", side_effect=self._fake_get):
            val = mlb_starters.resolve_player_game_stat(
                "Test Pitcher", "2024-07-04T23:05:00Z", "2024-07-04",
                "pitching", "strikeOuts", 2024)
        self.assertEqual(val, 9.0)  # nightcap gamePk 2

    def test_position_group_mismatch_skips(self):
        with patch.object(mlb_starters, "_get", side_effect=self._fake_get):
            self.assertIsNone(mlb_starters.resolve_player_game_stat(
                "Test Batter", "2024-07-04T23:05:00Z", "2024-07-04",
                "pitching", "strikeOuts", 2024))

    def test_utc_date_shift_binds_to_true_game_not_next_day(self):
        # A late West-coast game officially dated 07-11 is LOGGED game_date=07-12
        # (UTC of first pitch). The everyday hitter also plays on 07-12; the
        # resolver must pick the 07-11 game (nearest commence), not the 07-12 one.
        def fake_get(path, params=None, **kw):
            if path == "sports/1/players":
                return {"people": [{"id": 55, "fullName": "Everyday Bat",
                                    "primaryPosition": {"abbreviation": "CF"}}]}
            if path.startswith("people/") and path.endswith("/stats"):
                return {"stats": [{"splits": [
                    {"game": {"gamePk": 10}, "date": "2024-07-11",
                     "stat": {"hits": 2}},   # true (forecast) game
                    {"game": {"gamePk": 11}, "date": "2024-07-12",
                     "stat": {"hits": 0}},   # following day
                ]}]}
            if path == "schedule":
                d = (params or {}).get("date")
                if d == "2024-07-11":
                    return {"dates": [{"games": [
                        {"gamePk": 10, "gameDate": "2024-07-12T02:10:00Z",
                         "status": {"abstractGameState": "Final"},
                         "teams": {}}]}]}
                if d == "2024-07-12":
                    return {"dates": [{"games": [
                        {"gamePk": 11, "gameDate": "2024-07-12T20:10:00Z",
                         "status": {"abstractGameState": "Final"},
                         "teams": {}}]}]}
            return {"dates": []}

        with patch.object(mlb_starters, "_get", side_effect=fake_get):
            val = mlb_starters.resolve_player_game_stat(
                "Everyday Bat", "2024-07-12T02:10:00Z", "2024-07-12",
                "hitting", "hits", 2024)
        self.assertEqual(val, 2.0)  # the 07-11 night game, not 07-12's 0 hits

    def test_live_game_returns_not_final_sentinel(self):
        # The player's forecast game is still in progress: the resolver must
        # return the GAME_NOT_FINAL sentinel (not a partial stat) so the bet
        # stays pending instead of grading off an incomplete line.
        def fake_get(path, params=None, **kw):
            if path == "sports/1/players":
                return {"people": [{"id": 77, "fullName": "Live Bat",
                                    "primaryPosition": {"abbreviation": "CF"}}]}
            if path.startswith("people/") and path.endswith("/stats"):
                return {"stats": [{"splits": [
                    {"game": {"gamePk": 20}, "date": "2024-07-04",
                     "stat": {"hits": 0}}]}]}  # 0 hits SO FAR (3rd inning)
            if path == "schedule":
                return {"dates": [{"games": [
                    {"gamePk": 20, "gameDate": "2024-07-04T23:10:00Z",
                     "status": {"abstractGameState": "Live"}, "teams": {}}]}]}
            return {"dates": []}

        with patch.object(mlb_starters, "_get", side_effect=fake_get):
            val = mlb_starters.resolve_player_game_stat(
                "Live Bat", "2024-07-04T23:10:00Z", "2024-07-04",
                "hitting", "hits", 2024)
        self.assertIs(val, mlb_starters.GAME_NOT_FINAL)


class NdjsonReadCacheTests(unittest.TestCase):
    """The SQL read-cache (use_cache=True) must serve repeat reads without a new
    DB read, and every write through mutate_ndjson_log must invalidate it."""

    def setUp(self):
        recalibration._NDJSON_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        self.addCleanup(db_store.configure_engine, None)
        self.addCleanup(recalibration._NDJSON_CACHE.clear)

    def _seed(self, **row):
        def add(rows):
            rows.append(row)
            return 1
        recalibration.mutate_ndjson_log("wagers.jsonl", add)

    def test_cached_read_then_write_invalidates(self):
        self._seed(wager_id="w1", status="pending")
        # Spy on the DB read (mutate uses _select_rows, so it never bumps this).
        with patch.object(recalibration._db, "read_rows",
                          side_effect=db_store.read_rows) as spy:
            rows1, _ = recalibration._read_ndjson_blob("wagers.jsonl", use_cache=True)
            self.assertEqual(len(rows1), 1)
            self.assertEqual(spy.call_count, 1)

            # Second read within TTL is served from cache — no new DB read.
            rows2, _ = recalibration._read_ndjson_blob("wagers.jsonl", use_cache=True)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(len(rows2), 1)

            # A write pops the cache; the next cached read re-queries.
            def add(rows):
                rows.append({"wager_id": "w2", "status": "pending"})
                return 1
            recalibration.mutate_ndjson_log("wagers.jsonl", add)
            calls_before = spy.call_count
            rows3, _ = recalibration._read_ndjson_blob("wagers.jsonl", use_cache=True)
            self.assertGreater(spy.call_count, calls_before)  # cache invalidated
            self.assertEqual(len(rows3), 2)

    def test_mutated_rows_do_not_poison_cache(self):
        # A cached read returns a deep copy, so a caller mutating rows in place
        # cannot corrupt the snapshot served to the next reader.
        self._seed(wager_id="w1", status="pending", close_price=None)
        rows1, _ = recalibration._read_ndjson_blob("wagers.jsonl", use_cache=True)
        rows1[0]["close_price"] = -120  # caller mutates its copy
        rows2, _ = recalibration._read_ndjson_blob("wagers.jsonl", use_cache=True)
        self.assertIsNone(rows2[0]["close_price"])  # cache untouched


class StaleDnpVoidTests(unittest.TestCase):
    """A stale, confirmed scratch/DNP prediction is voided out of pending;
    anything not-confirmed-DNP (data outage / too recent) keeps retrying."""

    def _row(self):
        return {"ts": "2024-07-04T10:00:00Z", "sport_key": "baseball_mlb",
                "prop_key": "batter_hits", "player": "Scratch Sam",
                "game_date": "2024-07-04", "commence_time": "2024-07-04T23:05:00Z",
                "line": 0.5, "resolved": False}

    def _run(self, row, is_dnp):
        def mutate(mutator, where=None):
            return mutator([row])
        with patch.object(recalibration, "_read_log", return_value=[row]), \
             patch.object(recalibration, "resolve_one_prop", return_value=None), \
             patch.object(recalibration, "_is_stale_dnp", return_value=is_dnp), \
             patch.object(recalibration, "mutate_prediction_log",
                          side_effect=mutate):
            return recalibration.resolve_pending_outcomes("baseball_mlb")

    def test_stale_dnp_is_voided(self):
        row = self._row()
        ret = self._run(row, is_dnp=True)
        self.assertTrue(row["resolved"])          # cleared out of pending
        self.assertIsNone(row["outcome"])         # no label -> excluded from calib
        self.assertIsNone(row["actual"])
        self.assertEqual(ret, 0)                  # voids don't count as resolved

    def test_unresolvable_but_not_dnp_stays_pending(self):
        row = self._row()
        ret = self._run(row, is_dnp=False)        # e.g. data outage / not stale yet
        self.assertFalse(row.get("resolved"))     # keeps retrying next pass
        self.assertNotIn("outcome", row)          # untouched
        self.assertEqual(ret, 0)

    def test_is_stale_dnp_age_gate(self):
        # Non-MLB and non-stale games are never voided (unit-level gate check).
        self.assertFalse(recalibration._is_stale_dnp(
            "basketball_nba", "player_points", "X", "2024-07-04",
            "2024-07-04T23:05:00Z"))
        from datetime import datetime, timezone
        recent = datetime.now(timezone.utc).isoformat()
        self.assertFalse(recalibration._is_stale_dnp(
            "baseball_mlb", "batter_hits", "X",
            recent[:10], recent))                 # game just now -> under the gate


if __name__ == "__main__":
    unittest.main()
