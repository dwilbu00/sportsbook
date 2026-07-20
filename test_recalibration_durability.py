"""Tests for accuracy-roadmap 0.1b (Blob-durable Platt recalibration) and 0.2
(hard-ID / doubleheader-safe outcome resolution).

All external I/O is mocked: the Azure blob (requests.get/put) and the MLB
statsapi client (_get) are patched, and _prediction_log_blob_url is either
patched to a fake URL or to "" so nothing touches live services — important
because .streamlit/secrets.toml is present locally.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import mlb_starters
import recalibration

_FAKE_URL = ("https://acct.blob.core.windows.net/cont/predictions/"
             "prediction_log.jsonl?sig=abc&sr=c")


class _FakeBlobStore:
    """Minimal in-memory Azure blob honoring If-None-Match:* and If-Match."""

    def __init__(self, body=None, etag=None):
        self.body = body
        self.etag = etag

    def get(self, url, headers=None, timeout=None):
        inm = (headers or {}).get("If-None-Match")
        if self.body is None:
            return Mock(status_code=404)
        if inm and inm == self.etag:
            return Mock(status_code=304)
        return Mock(status_code=200, text=self.body, headers={"ETag": self.etag})

    def put(self, url, data=None, headers=None, timeout=None):
        inm = (headers or {}).get("If-None-Match")
        ifm = (headers or {}).get("If-Match")
        if inm == "*" and self.body is not None:
            return Mock(status_code=412)
        if ifm is not None and ifm != self.etag:
            return Mock(status_code=412)
        self.body = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        self.etag = f'"{abs(hash(self.body)) & 0xffff}"'
        return Mock(status_code=201)


_VALID_FIT = {"batter_hits": {"a": 0.5, "b": 0.1, "n_fit": 120, "validated": True}}


class BlobRecalibrationTests(unittest.TestCase):
    def setUp(self):
        recalibration._LOAD_CACHE.pop("baseball_mlb", None)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(recalibration._LOAD_CACHE.pop, "baseball_mlb", None)

    def test_save_then_load_blob_round_trip(self):
        store = _FakeBlobStore()
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=_FAKE_URL), patch(
                "requests.get", side_effect=store.get), patch(
                "requests.put", side_effect=store.put):
            recalibration.save_recalibration("baseball_mlb", _VALID_FIT)
            props = recalibration.load_recalibration("baseball_mlb")
        self.assertIn("batter_hits", props)
        self.assertEqual(props["batter_hits"]["a"], 0.5)
        self.assertIsNotNone(store.body)  # persisted to the blob, not just local

    def test_unvalidated_fit_is_not_applied(self):
        store = _FakeBlobStore()
        fit = {"pitcher_strikeouts": {"a": 1.0, "b": 0.0, "validated": False}}
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=_FAKE_URL), patch(
                "requests.get", side_effect=store.get), patch(
                "requests.put", side_effect=store.put):
            recalibration.save_recalibration("baseball_mlb", fit)
            props = recalibration.load_recalibration("baseball_mlb")
        self.assertEqual(props, {})

    def test_load_falls_back_to_local_baseline_on_404(self):
        # Empty blob (404) must not hide the git-committed seed.
        store = _FakeBlobStore()  # body None -> GET 404
        path = os.path.join(self._tmp.name, "recalibration_baseball_mlb.json")
        with open(path, "w") as f:
            json.dump({"fit_timestamp": "2026-07-20T00:00:00+00:00",
                       "props": _VALID_FIT}, f)
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=_FAKE_URL), patch(
                "requests.get", side_effect=store.get):
            props = recalibration.load_recalibration("baseball_mlb")
        self.assertIn("batter_hits", props)

    def test_load_degrades_to_cache_on_network_error(self):
        store = _FakeBlobStore(body=json.dumps({"props": _VALID_FIT}), etag='"e1"')
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=_FAKE_URL):
            with patch("requests.get", side_effect=store.get):
                first = recalibration.load_recalibration("baseball_mlb")
            self.assertIn("batter_hits", first)
            # Expire the TTL, then make the network fail.
            recalibration._LOAD_CACHE["baseball_mlb"]["fetched_at"] = 0
            import requests
            with patch("requests.get", side_effect=requests.RequestException("down")):
                degraded = recalibration.load_recalibration("baseball_mlb")
        self.assertEqual(degraded, first)

    def test_malformed_props_degrades_to_empty_without_raising(self):
        # Valid JSON but a bad `props` shape must NOT raise on the free-loop
        # load path (props.py:348 is unwrapped) — it must degrade to {}.
        for body in ('{"props": {"batter_hits": null}}',
                     '{"props": [1, 2, 3]}',
                     '{"props": {"batter_hits": "oops"}}'):
            store = _FakeBlobStore(body=body, etag='"e"')
            recalibration._LOAD_CACHE.pop("baseball_mlb", None)
            with patch.object(recalibration, "CALIB_DIR", self._tmp.name), patch.object(
                    recalibration, "_prediction_log_blob_url", return_value=_FAKE_URL), patch(
                    "requests.get", side_effect=store.get):
                self.assertEqual(
                    recalibration.load_recalibration("baseball_mlb"), {})

    def test_save_overwrites_an_unreadable_blob(self):
        # A present-but-corrupt blob must be overwritable (If-Match), not loop
        # forever on If-None-Match:* -> 412 (durability silently lost).
        store = _FakeBlobStore(body="corrupt not json", etag='"e0"')
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=_FAKE_URL), patch(
                "requests.get", side_effect=store.get), patch(
                "requests.put", side_effect=store.put):
            recalibration.save_recalibration("baseball_mlb", _VALID_FIT)
            props = recalibration.load_recalibration("baseball_mlb")
        self.assertIn("batter_hits", props)

    def test_seed_save_stays_local_only(self):
        # to_blob=False must never PUT to the production blob.
        put_calls = []
        with patch.object(recalibration, "CALIB_DIR", self._tmp.name), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=_FAKE_URL), patch(
                "requests.put", side_effect=lambda *a, **k: put_calls.append(a)):
            recalibration.save_recalibration("baseball_mlb", _VALID_FIT,
                                             to_blob=False)
        self.assertEqual(put_calls, [])
        self.assertTrue(os.path.exists(
            os.path.join(self._tmp.name, "recalibration_baseball_mlb.json")))


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

    def test_resolve_disambiguates_doubleheader_via_espn(self):
        row = {
            "ts": "2024-07-04T10:00:00Z", "sport_key": "baseball_mlb",
            "prop_key": "batter_hits", "player": "Player One",
            "game_date": "2024-07-04", "commence_time": "2024-07-04T23:05:00Z",
            "line": 0.5, "resolved": False,
        }
        gamelog = [
            {"game_date": "2024-07-04T17:10:00Z", "H": 0},   # game 1
            {"game_date": "2024-07-04T23:10:00Z", "H": 2},   # game 2 (nightcap)
        ]

        def mutate(mutator):
            return mutator([row])

        with patch.object(recalibration, "_read_log", return_value=[row]), \
             patch.object(recalibration, "_resolve_mlb_actual", return_value=None), \
             patch("espn_cache.cached_athlete_id", return_value="1"), \
             patch("espn_cache.cached_gamelog", return_value=gamelog), \
             patch.object(recalibration, "_stat_label", return_value="H"), \
             patch.object(recalibration, "mutate_prediction_log", side_effect=mutate):
            resolved = recalibration.resolve_pending_outcomes("baseball_mlb")
        self.assertEqual(resolved, 1)
        self.assertEqual(row["actual"], 2.0)   # nightcap, not the 17:10 opener
        self.assertEqual(row["outcome"], 1)

    def test_statsapi_resolves_when_espn_unavailable(self):
        # The hard-ID path must run independently of ESPN: a player ESPN can't
        # name-match (aid=None) is still graded via statsapi.
        row = {
            "ts": "2024-07-04T10:00:00Z", "sport_key": "baseball_mlb",
            "prop_key": "pitcher_strikeouts", "player": "Ace Pitcher",
            "game_date": "2024-07-04", "commence_time": "2024-07-04T23:05:00Z",
            "line": 6.5, "resolved": False,
        }

        def mutate(mutator):
            return mutator([row])

        with patch.object(recalibration, "_read_log", return_value=[row]), \
             patch.object(recalibration, "_resolve_mlb_actual", return_value=8.0), \
             patch("espn_cache.cached_athlete_id", return_value=None), \
             patch.object(recalibration, "mutate_prediction_log", side_effect=mutate):
            resolved = recalibration.resolve_pending_outcomes("baseball_mlb")
        self.assertEqual(resolved, 1)
        self.assertEqual(row["actual"], 8.0)
        self.assertEqual(row["outcome"], 1)


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
                 "teams": {}},
                {"gamePk": 2, "gameDate": "2024-07-04T23:10:00Z",
                 "teams": {}},
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
                         "teams": {}}]}]}
                if d == "2024-07-12":
                    return {"dates": [{"games": [
                        {"gamePk": 11, "gameDate": "2024-07-12T20:10:00Z",
                         "teams": {}}]}]}
            return {"dates": []}

        with patch.object(mlb_starters, "_get", side_effect=fake_get):
            val = mlb_starters.resolve_player_game_stat(
                "Everyday Bat", "2024-07-12T02:10:00Z", "2024-07-12",
                "hitting", "hits", 2024)
        self.assertEqual(val, 2.0)  # the 07-11 night game, not 07-12's 0 hits


if __name__ == "__main__":
    unittest.main()
