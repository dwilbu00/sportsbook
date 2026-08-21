"""Tests for roadmap 0.4 — the durable odds/line warehouse.

Fully hermetic: no live Odds API / Azure I/O. These exercise the local-fallback
path (SCRIPT_DIR redirected to a tempdir, _sql() forced off); capture -> flush ->
read is verified end to end, along with write-once immutability, seeding, and
closing-line lookup.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import warehouse


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


class LocalFallbackTests(unittest.TestCase):
    def setUp(self):
        warehouse._accumulator.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(warehouse._accumulator.clear)
        p1 = patch.object(warehouse, "SCRIPT_DIR", self._tmp.name)
        p2 = patch.object(warehouse, "_sql", return_value=False)
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

    def test_source_defaults_live_and_can_override(self):
        # provenance tag: unmarked callers (live analysis fetch) -> 'live';
        # the backfill passes source='backfill'.
        warehouse.capture_event_odds(
            "baseball_mlb", "E1", "us", "h2h", None, _payload(),
            captured_at="2026-07-16T14:00:00Z")
        warehouse.capture_event_odds(
            "baseball_mlb", "E2", "us", "h2h", None, _payload(event_id="E2"),
            captured_at="2026-07-16T14:00:00Z", source="backfill")
        warehouse.flush()
        env = {}
        for s in warehouse.list_snapshots("baseball_mlb", "2026-07-16"):
            e = warehouse.read_snapshot(s["name"])
            env[e["event_id"]] = e.get("source")
        self.assertEqual(env["E1"], "live")
        self.assertEqual(env["E2"], "backfill")

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


class SeedAndJoinTests(unittest.TestCase):
    def setUp(self):
        warehouse._accumulator.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(warehouse._accumulator.clear)
        self._store_dir = os.path.join(self._tmp.name, "historical_odds")
        os.makedirs(self._store_dir, exist_ok=True)
        p1 = patch.object(warehouse, "SCRIPT_DIR", self._tmp.name)
        p2 = patch.object(warehouse, "_sql", return_value=False)
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
        # Self-tag: a seed reload must stamp source='seed' (not the 'live' default),
        # so a later retag never has to fix it.
        self.assertEqual(env["source"], "seed")

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
