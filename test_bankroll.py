"""Tests for the durable bankroll ledger + Kelly-settings KV store (bankroll.py).

Hermetic: no live Azure. Every behavior is exercised on BOTH backends the store
supports -- an in-memory SQLite engine (the prod SQL path, incl. the surgical
INSERT/UPDATE/DELETE diff) and a tempdir NDJSON file (the Blob/local fallback) --
via two context managers a shared mixin runs its assertions under. That mirrors
test_wagers.py's _SqlLedger / _LocalLedger split and guarantees the derived
balance, the difference-transaction adjustment rule, the idempotent bet-P/L
reconcile sweep, and the settings round-trip behave identically on each.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_bankroll -v
"""
import tempfile
import unittest
from unittest.mock import patch

import bankroll
import db_store
import recalibration


class _SqlBankroll:
    """Route bankroll/recalibration onto a fresh in-memory SQL store (exercises
    the surgical diff writer + typed reads)."""

    def __enter__(self):
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        return self

    def __exit__(self, *exc):
        db_store.configure_engine(None)
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()


class _LocalBankroll:
    """Force bankroll/recalibration onto a tempdir NDJSON store (Blob/local
    fallback: ``where`` ignored, full-file read-modify-write)."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()

    def __enter__(self):
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        self._p1 = patch.object(recalibration, "PRED_DIR", self._tmp.name)
        self._p2 = patch.object(recalibration, "_prediction_log_blob_url",
                                return_value="")
        self._p1.start(); self._p2.start()
        return self

    def __exit__(self, *exc):
        self._p1.stop(); self._p2.stop()
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        self._tmp.cleanup()


def _w(wager_id, status, profit):
    """Minimal settled/pending wager row (only the fields reconcile reads)."""
    return {"wager_id": wager_id, "status": status, "profit": profit,
            "resolved_at": "2026-08-07T00:00:00+00:00"}


class _BankrollBehaviorMixin:
    """Backend-agnostic assertions; subclasses set ``ledger`` to a CM factory."""

    ledger = None

    # -- derived balance + adjustments ------------------------------------------

    def test_empty_ledger_is_zero(self):
        with self.ledger():
            self.assertEqual(bankroll.read_ledger(), [])
            self.assertEqual(bankroll.current_balance(), 0.0)
            s = bankroll.summary()
            self.assertEqual(s["balance"], 0.0)
            self.assertEqual(s["n_txns"], 0)

    def test_adjustment_from_zero_writes_full_target(self):
        with self.ledger():
            self.assertEqual(bankroll.record_adjustment(1000.0), 1000.0)
            self.assertEqual(bankroll.current_balance(), 1000.0)
            txns = bankroll.read_ledger()
            self.assertEqual(len(txns), 1)
            self.assertEqual(txns[0]["txn_type"], "adjustment")

    def test_withdraw_then_redeposit_by_target(self):
        # The user's own example: ledger 700, withdraw to 500 -> -200; back to
        # 700 -> +200. Each adjustment is the SIGNED difference target-current.
        with self.ledger():
            bankroll.record_adjustment(700.0)
            self.assertEqual(bankroll.record_adjustment(500.0), -200.0)
            self.assertEqual(bankroll.current_balance(), 500.0)
            self.assertEqual(bankroll.record_adjustment(700.0), 200.0)
            self.assertEqual(bankroll.current_balance(), 700.0)

    def test_equal_target_is_a_noop(self):
        with self.ledger():
            bankroll.record_adjustment(500.0)
            self.assertEqual(bankroll.record_adjustment(500.0), 0.0)
            self.assertEqual(bankroll.record_adjustment(500.004), 0.0)  # <half cent
            self.assertEqual(len(bankroll.read_ledger()), 1)  # no extra txn
            self.assertEqual(bankroll.current_balance(), 500.0)

    def test_bad_target_is_ignored(self):
        with self.ledger():
            self.assertEqual(bankroll.record_adjustment("not a number"), 0.0)
            self.assertEqual(bankroll.record_adjustment(None), 0.0)
            self.assertEqual(bankroll.read_ledger(), [])

    # -- reconcile bet P/L ------------------------------------------------------

    def test_reconcile_settled_bets_sum_to_profit(self):
        with self.ledger():
            rows = [_w("a", "won", 9.09), _w("b", "lost", -10.0),
                    _w("c", "push", 0.0), _w("d", "pending", None)]
            self.assertEqual(bankroll.reconcile_bet_txns(rows), 3)  # d excluded
            self.assertAlmostEqual(bankroll.current_balance(), -0.91)
            # Idempotent: no wager change -> no write.
            self.assertEqual(bankroll.reconcile_bet_txns(rows), 0)

    def test_reconcile_regrade_to_pending_removes_txn(self):
        with self.ledger():
            bankroll.reconcile_bet_txns([_w("a", "won", 9.09)])
            self.assertEqual(bankroll.current_balance(), 9.09)
            self.assertEqual(
                bankroll.reconcile_bet_txns([_w("a", "pending", None)]), 1)
            self.assertEqual(bankroll.current_balance(), 0.0)

    def test_reconcile_deleted_wager_removes_txn(self):
        with self.ledger():
            bankroll.reconcile_bet_txns([_w("a", "won", 5.0)])
            self.assertEqual(bankroll.reconcile_bet_txns([]), 1)  # wager gone
            self.assertEqual(bankroll.current_balance(), 0.0)

    def test_reconcile_changed_profit_updates_amount(self):
        with self.ledger():
            bankroll.reconcile_bet_txns([_w("a", "won", 5.0)])
            self.assertEqual(bankroll.reconcile_bet_txns([_w("a", "won", 7.5)]), 1)
            self.assertEqual(bankroll.current_balance(), 7.5)

    def test_reconcile_leaves_adjustments_untouched(self):
        with self.ledger():
            bankroll.record_adjustment(700.0)
            bankroll.reconcile_bet_txns([_w("a", "lost", -10.0)])
            self.assertEqual(bankroll.current_balance(), 690.0)
            bankroll.reconcile_bet_txns([])       # bet gone; adjustment stays
            self.assertEqual(bankroll.current_balance(), 700.0)
            s = bankroll.summary()
            self.assertEqual(s["adjustments_total"], 700.0)
            self.assertEqual(s["bets_total"], 0.0)

    def test_end_to_end_target_then_new_settlement(self):
        with self.ledger():
            settled = [_w("a", "won", 20.0), _w("b", "lost", -10.0)]
            bankroll.reconcile_bet_txns(settled)
            self.assertEqual(bankroll.current_balance(), 10.0)
            # User sets their real bankroll once.
            self.assertEqual(bankroll.record_adjustment(700.0), 690.0)
            self.assertEqual(bankroll.current_balance(), 700.0)
            # A later settlement moves the balance by exactly its profit, and the
            # one-time adjustment is NOT double-counted.
            bankroll.reconcile_bet_txns(settled + [_w("c", "won", 45.0)])
            self.assertEqual(bankroll.current_balance(), 745.0)

    def test_summary_splits_bets_and_adjustments(self):
        with self.ledger():
            bankroll.record_adjustment(700.0)
            bankroll.reconcile_bet_txns([_w("a", "won", 9.09)])
            s = bankroll.summary()
            self.assertEqual(s["adjustments_total"], 700.0)
            self.assertEqual(s["bets_total"], 9.09)
            self.assertEqual(s["balance"], 709.09)
            self.assertEqual(s["n_txns"], 2)
            # Newest txn first.
            self.assertEqual(len(s["txns"]), 2)

    # -- Kelly settings KV round-trip --------------------------------------------

    def test_kelly_settings_default_when_unset(self):
        with self.ledger():
            d = bankroll.load_kelly_settings()
            self.assertEqual(d["kelly_fraction"], 0.5)
            self.assertEqual(d["kelly_cap_pct"], 5.0)
            self.assertEqual(d["kelly_slate_cap_pct"], 25.0)

    def test_kelly_settings_round_trip_and_partial_update(self):
        with self.ledger():
            self.assertEqual(bankroll.save_kelly_settings(0.25, 3.0, 15.0), 3)
            d = bankroll.load_kelly_settings()
            self.assertEqual(d["kelly_fraction"], 0.25)
            self.assertEqual(d["kelly_cap_pct"], 3.0)
            self.assertEqual(d["kelly_slate_cap_pct"], 15.0)
            # Re-saving identical values is a no-op; changing one updates one.
            self.assertEqual(bankroll.save_kelly_settings(0.25, 3.0, 15.0), 0)
            self.assertEqual(bankroll.save_kelly_settings(0.5, 3.0, 15.0), 1)
            self.assertEqual(bankroll.load_kelly_settings()["kelly_fraction"], 0.5)


class BankrollSqlTests(_BankrollBehaviorMixin, unittest.TestCase):
    ledger = _SqlBankroll


class BankrollLocalTests(_BankrollBehaviorMixin, unittest.TestCase):
    ledger = _LocalBankroll


class ReconcileReadsWagersLedgerTests(unittest.TestCase):
    """When called with no rows, reconcile pulls the wagers ledger itself. Verify
    the lazy import + read path end-to-end on the SQL store (submit + grade a
    real wager, then reconcile picks up its realized profit)."""

    def test_reconcile_reads_settled_wagers_from_ledger(self):
        from datetime import datetime, timezone
        import wagers

        with _SqlBankroll():
            row = wagers.build_wager_row("player_prop", None, {
                "player": "Rafael Devers", "prop": "batter_hits",
                "prop_label": "Hits", "line": 1.5, "direction": "OVER",
                "over_price": -110, "over_rate": 60.0, "edge_pct": 7.0,
                "matchup": "NYY @ BOS", "team": "Boston Red Sox",
                "event_id": "E1"}, {
                "sport_key": "baseball_mlb", "event_id": "E1",
                "commence_time": "2026-07-16T18:00:00Z", "game_date": "2026-07-16",
                "home_team": "Boston Red Sox", "away_team": "New York Yankees",
                "stake": 10.0, "placed_at": "2026-07-16T12:00:00+00:00", "seq": 0})
            wagers.submit_wagers([row])
            # Pending wager contributes nothing yet.
            self.assertEqual(bankroll.reconcile_bet_txns(), 0)
            self.assertEqual(bankroll.current_balance(), 0.0)
            # Grade it to a loss, then reconcile off the ledger (no rows passed).
            now = datetime(2026, 7, 20, tzinfo=timezone.utc)
            with patch.object(recalibration, "resolve_one_prop", return_value=0.0):
                wagers.resolve_pending_wagers(now=now)
            self.assertEqual(bankroll.reconcile_bet_txns(), 1)
            self.assertEqual(bankroll.current_balance(), -10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
