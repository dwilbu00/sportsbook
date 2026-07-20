"""Tests for roadmap 0.4 — the durable odds/line warehouse.

Fully hermetic: no live Odds API / Azure I/O. The blob path is exercised with an
in-memory fake container (honoring If-None-Match:* write-once and If-Match RMW);
the local-fallback path uses a tempdir. capture -> flush -> read is verified end
to end, along with write-once immutability, seeding, and closing-line lookup.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import warehouse

_FAKE_URL = ("https://acct.blob.core.windows.net/cont/predictions/"
             "prediction_log.jsonl?sig=abc&sr=c")


def _payload(event_id="E1", home="Boston Red Sox", away="New York Yankees",
             commence="2026-07-16T18:00:00Z"):
    return {
        "id": event_id,
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "bookmakers": [{
            "key": "draftkings",
            "title": "DraftKings",
            "markets": [{
                "key": "h2h",
                "outcomes": [{"name": home, "price": -120},
                             {"name": away, "price": 110}],
            }],
        }],
    }


class _FakeContainer:
    """In-memory Azure container honoring If-None-Match:* and If-Match."""

    def __init__(self):
        self.blobs = {}   # path -> (body, etag)
        self.puts = []

    @staticmethod
    def _path(url):
        return urlsplit(url).path

    def get(self, url, headers=None, timeout=None):
        path = self._path(url)
        if path not in self.blobs:
            return Mock(status_code=404)
        body, etag = self.blobs[path]
        return Mock(status_code=200, text=body, headers={"ETag": etag})

    def put(self, url, data=None, headers=None, timeout=None):
        path = self._path(url)
        headers = headers or {}
        exists = path in self.blobs
        if headers.get("If-None-Match") == "*" and exists:
            return Mock(status_code=412)
        ifm = headers.get("If-Match")
        if ifm is not None and (not exists or self.blobs[path][1] != ifm):
            return Mock(status_code=412)
        body = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        etag = f'"{abs(hash(body)) & 0xffff}"'
        self.blobs[path] = (body, etag)
        self.puts.append(path)
        return Mock(status_code=201)


class LocalFallbackTests(unittest.TestCase):
    def setUp(self):
        warehouse._accumulator.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(warehouse._accumulator.clear)
        p1 = patch.object(warehouse, "SCRIPT_DIR", self._tmp.name)
        p2 = patch.object(warehouse, "_blob_base", return_value="")
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def test_capture_flush_list_round_trip(self):
        warehouse.capture_event_odds(
            "baseball_mlb", "E1", "us", "h2h", None, _payload(),
            captured_at="2026-07-16T14:00:00Z")
        warehouse.capture_event_odds(
            "baseball_mlb", "E1", "us", "h2h", None, _payload(),
            captured_at="2026-07-16T15:00:00Z")
        flushed = warehouse.flush()
        self.assertEqual(flushed, 2)
        snaps = warehouse.list_snapshots("baseball_mlb", "2026-07-16")
        self.assertEqual(len(snaps), 2)
        env = warehouse.read_snapshot(snaps[0]["name"])
        self.assertEqual(env["event_id"], "E1")
        self.assertEqual(env["payload"]["home_team"], "Boston Red Sox")

    def test_same_hour_capture_is_write_once(self):
        for _ in range(2):
            warehouse.capture_event_odds(
                "baseball_mlb", "E1", "us", "h2h", None, _payload(),
                captured_at="2026-07-16T14:30:00Z")
        warehouse.flush()
        snaps = warehouse.list_snapshots("baseball_mlb", "2026-07-16")
        self.assertEqual(len(snaps), 1)  # one hour bucket -> one immutable blob

    def test_no_commence_is_skipped(self):
        warehouse.capture_event_odds(
            "baseball_mlb", "E1", "us", "h2h", None,
            {"id": "E1", "home_team": "H", "away_team": "A"})
        self.assertEqual(warehouse.flush(), 0)

    def test_closing_line_for_moneyline(self):
        warehouse.capture_event_odds(
            "baseball_mlb", "E1", "us", "h2h", None, _payload(),
            captured_at="2026-07-16T17:00:00Z")
        warehouse.flush()
        close = warehouse.closing_line_for(
            "baseball_mlb", "2026-07-16", "E1", "moneyline",
            selection="Boston Red Sox", commence_time="2026-07-16T18:00:00Z")
        self.assertIsNotNone(close)
        self.assertEqual(close["price"], -120)
        self.assertAlmostEqual(close["implied_prob"], 120 / 220, places=4)


class BlobStoreTests(unittest.TestCase):
    def setUp(self):
        warehouse._accumulator.clear()
        self.addCleanup(warehouse._accumulator.clear)
        self.container = _FakeContainer()
        p1 = patch.object(warehouse, "_blob_base", return_value=_FAKE_URL)
        p2 = patch("requests.get", side_effect=self.container.get)
        p3 = patch("requests.put", side_effect=self.container.put)
        p1.start(); p2.start(); p3.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop); self.addCleanup(p3.stop)

    def test_snapshot_lands_under_container_root(self):
        warehouse.capture_event_odds(
            "baseball_mlb", "E1", "us", "h2h", None, _payload(),
            captured_at="2026-07-16T14:00:00Z")
        warehouse.flush()
        # Blob paths are container-root-relative (…/cont/warehouse/…), not nested
        # under predictions/.
        snap_paths = [p for p in self.container.puts if "/warehouse/" in p
                      and not p.endswith("_manifest.json")]
        self.assertTrue(snap_paths)
        self.assertIn("/cont/warehouse/baseball_mlb/2026-07-16/", snap_paths[0])
        snaps = warehouse.list_snapshots("baseball_mlb", "2026-07-16")
        self.assertEqual(len(snaps), 1)

    def test_reput_same_name_is_412_swallowed(self):
        for _ in range(2):
            warehouse.capture_event_odds(
                "baseball_mlb", "E1", "us", "h2h", None, _payload(),
                captured_at="2026-07-16T14:00:00Z")
        # Two eager PUTs to the same immutable name: the second returns 412 and
        # must be swallowed (no raise); the manifest still holds exactly one.
        warehouse.flush()
        snaps = warehouse.list_snapshots("baseball_mlb", "2026-07-16")
        self.assertEqual(len(snaps), 1)

    def test_manifest_merges_two_events(self):
        warehouse.capture_event_odds(
            "baseball_mlb", "E1", "us", "h2h", None, _payload("E1"),
            captured_at="2026-07-16T14:00:00Z")
        warehouse.capture_event_odds(
            "baseball_mlb", "E2", "us", "h2h", None,
            _payload("E2", home="Chicago Cubs", away="St. Louis Cardinals"),
            captured_at="2026-07-16T14:00:00Z")
        warehouse.flush()
        snaps = warehouse.list_snapshots("baseball_mlb", "2026-07-16")
        self.assertEqual({s["event_id"] for s in snaps}, {"E1", "E2"})


class SeedAndJoinTests(unittest.TestCase):
    def setUp(self):
        warehouse._accumulator.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(warehouse._accumulator.clear)
        self._store_dir = os.path.join(self._tmp.name, "historical_odds")
        os.makedirs(self._store_dir, exist_ok=True)
        p1 = patch.object(warehouse, "SCRIPT_DIR", self._tmp.name)
        p2 = patch.object(warehouse, "_blob_base", return_value="")
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def _write_store(self):
        import historical_odds as store_mod
        store = {
            "sport_key": "baseball_mlb",
            "games": {
                "2026-07-16|New York Yankees @ Boston Red Sox": {
                    "commence_time": "2026-07-16T18:00:00Z",
                    "event_id": "E1",
                    "home_team": "Boston Red Sox",
                    "away_team": "New York Yankees",
                    "moneyline": {
                        "Boston Red Sox": [
                            {"book": "DK", "price": -120, "implied_prob": 0.545}],
                        "New York Yankees": [
                            {"book": "DK", "price": 110, "implied_prob": 0.476}],
                    },
                    "spreads": {}, "totals": {}, "props": {},
                },
            },
        }
        with patch.object(store_mod, "STORE_DIR", self._store_dir):
            store_mod.save_store("baseball_mlb", store)

    def test_seed_from_store_then_list_and_join(self):
        import historical_odds as store_mod
        self._write_store()
        with patch.object(store_mod, "STORE_DIR", self._store_dir):
            written = warehouse.seed_from_store("baseball_mlb")
        self.assertEqual(written, 1)
        snaps = warehouse.list_snapshots("baseball_mlb", "2026-07-16")
        self.assertEqual(len(snaps), 1)
        env = warehouse.read_snapshot(snaps[0]["name"])
        self.assertEqual(env["format"], "historical_odds_store")

        # Closing line reads straight from the seeded (parsed) payload.
        close = warehouse.closing_line_for(
            "baseball_mlb", "2026-07-16", "E1", "moneyline",
            selection="Boston Red Sox")
        self.assertEqual(close["price"], -120)

        # Join a resolved prediction-log row to the event's snapshot span.
        log_rows = [{
            "sport_key": "baseball_mlb", "event_id": "E1",
            "prop_key": "batter_hits", "player": "Some Batter",
            "game_date": "2026-07-16", "line": 0.5, "raw_prob": 0.6,
            "final_prob": 0.58, "outcome": 1, "resolved": True,
        }]
        with patch("recalibration.read_prediction_log", return_value=log_rows):
            joined = warehouse.join_predictions_to_lines(
                "baseball_mlb", ["2026-07-16"])
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["outcome"], 1)
        self.assertEqual(joined[0]["n_snapshots"], 1)
        self.assertIsNotNone(joined[0]["close_captured_at"])


class KindClassificationTests(unittest.TestCase):
    def test_kind_for_markets(self):
        self.assertEqual(warehouse._kind_for_markets("h2h,spreads,totals"), "team")
        self.assertEqual(warehouse._kind_for_markets("batter_hits"), "props")
        self.assertEqual(
            warehouse._kind_for_markets("batter_hits_alternate"), "alt")

    def test_hour_bucket_is_colon_free(self):
        self.assertEqual(
            warehouse._hour_bucket("2026-07-16T14:23:01Z"), "20260716T14Z")


if __name__ == "__main__":
    unittest.main()
