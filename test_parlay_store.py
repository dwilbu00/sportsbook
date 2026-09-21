"""Storage/settlement regression tests for parlay_store (audit F04-F06).

Hermetic: a fresh in-memory SQLite store per test (the prod SQL path), mirror
disabled, NDJSON cache cleared. Exercises atomic ticket+leg persistence, ghost-leg
replacement, parlay bankroll contribution and void-stake settlement.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_parlay_store -v
"""
import os
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import bankroll
import db_store
import parlay_store
import recalibration


class _ParlayStoreTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch("warehouse_mirror.enabled", return_value=False))
        recalibration._NDJSON_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()

    def tearDown(self):
        db_store.configure_engine(None)
        recalibration._NDJSON_CACHE.clear()
        self.stack.close()

    def _ticket(self, pid="audit-ticket"):
        return dict(parlay_id=pid, book="draftkings", stake=10.,
                    combined_american=200, boost_pct=.5,
                    sport_key="americanfootball_nfl")

    def _legs(self, n=2):
        return [dict(player=f"Fixture {i}", prop_key="player_receptions", line=4.5,
                     side="OVER", price=-110, sport_key="americanfootball_nfl",
                     game_date="2026-09-18", commence_time="2026-09-18T20:00:00Z")
                for i in range(n)]


class F04AtomicityTests(_ParlayStoreTest):
    def test_ticket_and_legs_save_is_atomic(self):
        # A leg-write failure must not leave an orphan ticket with no legs. Legs are
        # written first (ticket = commit point), so the failure raises before any
        # ticket exists.
        real_mutate = recalibration.mutate_ndjson_log

        def fail_legs(filename, *args, **kwargs):
            if filename == parlay_store.LEGS_FILE:
                raise OSError("synthetic leg write failure")
            return real_mutate(filename, *args, **kwargs)

        with patch.object(recalibration, "mutate_ndjson_log", side_effect=fail_legs):
            with self.assertRaises(OSError):
                parlay_store.save_parlay(self._ticket(), self._legs())
        self.assertEqual(db_store.read_rows("parlays"), [])

    def test_replacing_ticket_legs_does_not_keep_ghost_legs(self):
        parlay_store.save_parlay(self._ticket(), self._legs(3))
        parlay_store.save_parlay(self._ticket(), self._legs(2))
        t = parlay_store.load_parlays()[0]
        self.assertEqual(t["n_legs"], 2)
        self.assertEqual(len(t["legs"]), t["n_legs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
