"""Tests for the WS1b database_failure telemetry (audit P20 subset).

Two layers:
* The ops_telemetry helper itself — counters bump, structured log line, and the
  hard invariant that it NEVER raises (telemetry must not break a fail-open path).
* The wiring: each durable-store swallow site (save_recalibration SQL write,
  warehouse capture + the two refit-critical reads) records a `database_failure`
  event when the underlying db_store call raises, WITHOUT propagating the error.

Run on the in-memory SQLite engine so _sql() is True and the DB code paths are
actually reached; the db_store method under test is patched to raise.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_ops_telemetry -v
"""
import logging
import unittest
from unittest.mock import patch

import db_store
import ops_telemetry
import recalibration
import warehouse

_FIT = {"batter_hits": {"a": 0.5, "b": 0.1, "n_fit": 120, "validated": True}}


class OpsEventHelperTests(unittest.TestCase):

    def setUp(self):
        ops_telemetry.reset_counters()
        self.addCleanup(ops_telemetry.reset_counters)

    def test_event_bumps_counter(self):
        ops_telemetry.ops_event("database_failure", op="x")
        ops_telemetry.ops_event("database_failure", op="y")
        ops_telemetry.ops_event("identity_failure")
        self.assertEqual(ops_telemetry.count("database_failure"), 2)
        self.assertEqual(ops_telemetry.count("identity_failure"), 1)
        self.assertEqual(ops_telemetry.count("never_seen"), 0)
        self.assertEqual(ops_telemetry.counters(),
                         {"database_failure": 2, "identity_failure": 1})

    def test_event_logs_structured_line(self):
        with self.assertLogs("sportsbook.ops", level="WARNING") as cm:
            ops_telemetry.ops_event("database_failure", op="save_recal",
                                    sport="baseball_mlb")
        self.assertEqual(len(cm.records), 1)
        msg = cm.records[0].getMessage()
        self.assertIn("kind=database_failure", msg)
        self.assertIn("op='save_recal'", msg)
        self.assertIn("sport='baseball_mlb'", msg)

    def test_event_never_raises_on_bad_fields(self):
        class Boom:
            def __repr__(self):
                raise ValueError("no repr")
        # Must not propagate even if a field's repr() blows up.
        ops_telemetry.ops_event("database_failure", bad=Boom())
        # Counter still bumped (the bump precedes the format).
        self.assertEqual(ops_telemetry.count("database_failure"), 1)

    def test_reset_clears(self):
        ops_telemetry.ops_event("database_failure")
        ops_telemetry.reset_counters()
        self.assertEqual(ops_telemetry.counters(), {})


class _SqlEnv:
    """In-memory SQLite engine so _sql() is True; counters reset each use."""

    def __enter__(self):
        recalibration._LOAD_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        ops_telemetry.reset_counters()
        return self

    def __exit__(self, *exc):
        db_store.configure_engine(None)
        recalibration._LOAD_CACHE.clear()
        ops_telemetry.reset_counters()


class DbFailureWiringTests(unittest.TestCase):

    def test_save_recalibration_records_db_failure(self):
        with _SqlEnv():
            with patch.object(db_store, "save_recal",
                              side_effect=RuntimeError("db down")):
                # Must NOT raise (best-effort), and must record the event.
                recalibration.save_recalibration(
                    "baseball_mlb", _FIT, to_blob=True)
            self.assertEqual(
                ops_telemetry.count("database_failure"), 1)

    def test_load_prop_lines_records_db_failure(self):
        with _SqlEnv():
            with patch.object(db_store, "player_prop_lines",
                              side_effect=RuntimeError("db down")):
                out = warehouse.load_prop_lines("baseball_mlb")
            self.assertEqual(out, [])
            self.assertEqual(ops_telemetry.count("database_failure"), 1)

    def test_load_team_market_store_records_db_failure(self):
        with _SqlEnv():
            with patch.object(db_store, "team_market_lines",
                              side_effect=RuntimeError("db down")):
                store = warehouse.load_team_market_store("baseball_mlb")
            self.assertEqual(store["games"], {})
            self.assertEqual(ops_telemetry.count("database_failure"), 1)

    def test_capture_event_odds_records_db_failure(self):
        payload = {"commence_time": "2026-08-01T18:00:00Z",
                   "home_team": "Boston Red Sox",
                   "away_team": "New York Yankees", "bookmakers": []}
        with _SqlEnv():
            # Isolate the DB call from the parse helpers so the test pins the
            # telemetry wiring, not payload-enrichment internals.
            with patch.object(warehouse, "_enrich_ids", return_value=({}, [])), \
                 patch.object(warehouse, "_enumerate_lines", return_value=[]), \
                 patch.object(db_store, "capture_odds_snapshot",
                              side_effect=RuntimeError("db down")):
                # Best-effort: must not raise.
                warehouse.capture_event_odds(
                    "baseball_mlb", "E1", "us", "h2h", ["draftkings"], payload)
            self.assertEqual(ops_telemetry.count("database_failure"), 1)

    def test_healthy_write_records_nothing(self):
        # A successful SQL write must not log a spurious failure.
        with _SqlEnv():
            recalibration.save_recalibration("baseball_mlb", _FIT, to_blob=True)
            self.assertEqual(ops_telemetry.count("database_failure"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
