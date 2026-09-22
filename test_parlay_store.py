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


class F05BankrollContributionTests(_ParlayStoreTest):
    def test_settled_parlay_loss_contributes_to_bankroll(self):
        bankroll.record_adjustment(100.)
        pid = parlay_store.save_parlay(self._ticket(), self._legs())
        parlay_store.settle_manual(pid, "lost")
        bankroll.reconcile_bet_txns()
        self.assertEqual(bankroll.current_balance(), 90.)

    def test_settled_parlay_win_contributes_boosted_profit(self):
        bankroll.record_adjustment(100.)
        pid = parlay_store.save_parlay(self._ticket(), self._legs())
        parlay_store.settle_manual(pid, "won")
        bankroll.reconcile_bet_txns()
        # +200 -> dec 3.0; profit = stake*(dec-1)*(1+boost) = 10*2.0*1.5 = 30.
        self.assertEqual(bankroll.current_balance(), 130.)

    def test_reconcile_is_idempotent_for_parlays(self):
        bankroll.record_adjustment(100.)
        pid = parlay_store.save_parlay(self._ticket(), self._legs())
        parlay_store.settle_manual(pid, "lost")
        bankroll.reconcile_bet_txns()
        self.assertEqual(bankroll.reconcile_bet_txns(), 0)  # second call writes nothing
        self.assertEqual(bankroll.current_balance(), 90.)

    def test_failed_parlay_read_does_not_erase_parlay_bet_txns(self):
        # F03-class protection extended to parlays: an unavailable parlay store must
        # not delete existing bet:parlay:* txns as stale.
        bankroll.record_adjustment(100.)
        pid = parlay_store.save_parlay(self._ticket(), self._legs())
        parlay_store.settle_manual(pid, "lost")
        bankroll.reconcile_bet_txns()
        self.assertEqual(bankroll.current_balance(), 90.)
        real = recalibration._read_ndjson_blob

        def fail_parlays(filename, *a, **k):
            if filename == parlay_store.PARLAYS_FILE:
                raise OSError("synthetic parlay read failure")
            return real(filename, *a, **k)

        with patch.object(recalibration, "_read_ndjson_blob", side_effect=fail_parlays):
            bankroll.reconcile_bet_txns()
        self.assertEqual(bankroll.current_balance(), 90.)


class F06StakeEditSettlementTests(_ParlayStoreTest):
    def test_void_stake_edit_recomputes_full_refund(self):
        pid = parlay_store.save_parlay(self._ticket(), self._legs())
        parlay_store.settle_manual(pid, "void")
        parlay_store.update_stake(pid, 20.)
        self.assertEqual(parlay_store.load_parlays()[0]["payout"], 20.)

    def test_lost_stake_edit_recomputes_loss(self):
        pid = parlay_store.save_parlay(self._ticket(), self._legs())
        parlay_store.settle_manual(pid, "lost")
        parlay_store.update_stake(pid, 25.)
        t = parlay_store.load_parlays()[0]
        self.assertEqual(t["payout"], 0.0)
        self.assertEqual(t["profit"], -25.)

    def test_won_stake_edit_recomputes_boosted_payout(self):
        pid = parlay_store.save_parlay(self._ticket(), self._legs())
        parlay_store.settle_manual(pid, "won")
        parlay_store.update_stake(pid, 20.)
        t = parlay_store.load_parlays()[0]
        # +200 -> dec 3.0; profit = 20*(dec-1)*(1+boost) = 20*2.0*1.5 = 60; payout 80.
        self.assertEqual(t["profit"], 60.)
        self.assertEqual(t["payout"], 80.)


class ManualTicketBuildTests(unittest.TestCase):
    """parlay_store.build_manual_ticket — the pure assembly behind the manual-add form
    (book-built tickets the optimizer never surfaced, so analytics see the whole slate)."""

    def _rows(self, n=2, our_p=None):
        return [{"player": f"P{i}", "prop_key": "player_receptions", "side": "over",
                 "line": 4.5, "price": -110, "game_date": "2026-09-18", "our_p": our_p}
                for i in range(n)]

    def test_two_legs_with_probs_compute_joint_and_ev(self):
        ticket, legs = parlay_store.build_manual_ticket(
            "DraftKings", "parlay", "americanfootball_nfl", 0.5, 200, 10.0, "50% boost",
            self._rows(2, our_p=0.6))
        self.assertEqual(len(legs), 2)
        self.assertAlmostEqual(ticket["our_joint_prob"], 0.36)      # 0.6 * 0.6
        self.assertIsNotNone(ticket["our_boosted_ev_pct"])
        self.assertFalse(ticket["is_same_game"])
        self.assertEqual(legs[0]["side"], "OVER")                   # normalized upper
        self.assertEqual(legs[0]["corr_category"], "cross_game")
        self.assertEqual(ticket["combined_american"], 200)

    def test_missing_one_prob_leaves_joint_and_ev_none(self):
        rows = self._rows(2, our_p=0.6)
        rows[1]["our_p"] = None
        ticket, _ = parlay_store.build_manual_ticket(
            "DraftKings", "parlay", "americanfootball_nfl", 0.5, 200, 10.0, "", rows)
        self.assertIsNone(ticket["our_joint_prob"])
        self.assertIsNone(ticket["our_boosted_ev_pct"])
        self.assertIsNone(ticket["bonus_label"])                   # "" -> None

    def test_blank_rows_skipped_and_min_two_enforced(self):
        rows = self._rows(1, our_p=0.6) + [{"player": "  ", "prop_key": "x",
                                            "side": "OVER", "line": 1.5, "price": -110}]
        with self.assertRaises(ValueError):
            parlay_store.build_manual_ticket(
                "DraftKings", "parlay", "americanfootball_nfl", 0.0, 200, 10.0, "", rows)

    def test_invalid_combined_price_rejected(self):
        with self.assertRaises(ValueError):
            parlay_store.build_manual_ticket(
                "DraftKings", "parlay", "americanfootball_nfl", 0.0, 50, 10.0, "",
                self._rows(2))

    def test_sgp_sets_same_game_and_corr(self):
        ticket, legs = parlay_store.build_manual_ticket(
            "FanDuel", "sgp", "americanfootball_nfl", 0.3, -120, 5.0, "", self._rows(2))
        self.assertTrue(ticket["is_same_game"])
        self.assertEqual(legs[0]["corr_category"], "sgp")


class ManualTicketRoundTripTests(_ParlayStoreTest):
    def test_build_save_load_round_trip(self):
        rows = [{"player": "Amon-Ra St. Brown", "prop_key": "player_receptions",
                 "side": "OVER", "line": 5.5, "price": -115, "game_date": "2026-09-20",
                 "our_p": 0.62},
                {"player": "Jahmyr Gibbs", "prop_key": "player_rush_yds", "side": "OVER",
                 "line": 60.5, "price": -110, "game_date": "2026-09-20", "our_p": 0.55}]
        ticket, legs = parlay_store.build_manual_ticket(
            "DraftKings", "parlay", "americanfootball_nfl", 0.5, 264, 8.0, "50% boost", rows)
        pid = parlay_store.save_parlay(ticket, legs)
        loaded = parlay_store.load_parlays()
        self.assertEqual(len(loaded), 1)
        t = loaded[0]
        self.assertEqual(t["parlay_id"], pid)
        self.assertEqual(t["n_legs"], 2)
        self.assertEqual(len(t["legs"]), 2)
        self.assertEqual(t["book"], "DraftKings")
        self.assertAlmostEqual(t["our_joint_prob"], 0.62 * 0.55)
        self.assertEqual({l["player"] for l in t["legs"]},
                         {"Amon-Ra St. Brown", "Jahmyr Gibbs"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
