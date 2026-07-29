"""Pick-rules ROI lens (pickrules_roi.py).

The lens grades the RE-DERIVED recommended slate (bet_selector rules replayed
over every logged is_value forecast, grouped by ET game-date) against the raw
is_value pool, flat 1u/pick. These tests exercise the re-derivation semantics
(esp. the user-caught "a rule only fires on markets analyzed together" case),
the identity-membership dedup-safety, the priced-count-weighted combination, and
the team-field-availability reporting.
"""

import contextlib
import io
import unittest

import pickrules_roi
from recalibration import market_prediction_identity, prediction_identity


def _quiet_report(result):
    """Render the report to a throwaway buffer (smoke test: must not raise)."""
    with contextlib.redirect_stdout(io.StringIO()):
        pickrules_roi.print_report(result)


def _prow(prop, direction, event_id, player="P", team="Home", line=0.5,
          over_prob=0.7, price=-110, resolved=True, outcome=1, is_value=True,
          batting_order=1, game_date="2026-07-20",
          commence="2026-07-20T23:00:00Z", sport="baseball_mlb",
          ts="2026-07-20T12:00:00Z"):
    """A prediction_log row. `outcome` is over/under of actual vs line (1=over)."""
    return {
        "sport_key": sport, "event_id": event_id, "game_date": game_date,
        "commence_time": commence, "prop_key": prop, "player": player,
        "team": team, "line": line, "direction": direction,
        "final_prob": over_prob, "raw_prob": over_prob, "price": price,
        "outcome": outcome, "is_value": is_value, "resolved": resolved,
        "batting_order": batting_order, "ts": ts,
    }


def _mrow(bet_type, side, event_id, model_prob=0.6, price=-110, resolved=True,
          outcome=1, is_value=True, team=None, game_date="2026-07-20",
          commence="2026-07-20T23:00:00Z", sport="baseball_mlb",
          ts="2026-07-20T12:00:00Z"):
    """A market_prediction_log row. `outcome` is side-aware (1=pick won)."""
    return {
        "sport_key": sport, "event_id": event_id, "game_date": game_date,
        "commence_time": commence, "bet_type": bet_type, "side": side,
        "team": team, "model_prob": model_prob, "price": price,
        "outcome": outcome, "is_value": is_value, "resolved": resolved, "ts": ts,
    }


MLB = "baseball_mlb"


class RederiveConsistencyTests(unittest.TestCase):
    """The exact scenario the user flagged: a cross-market rule (ER/K) may only
    fire when BOTH markets are in the analyzed pool for that date."""

    def test_er_dropped_only_when_k_also_in_pool(self):
        # Date D1 (event e1): same pitcher ER-over + K-over both is_value.
        # K has higher EV (0.7 vs 0.6) → the ER-over is the one dropped.
        er1 = _prow("pitcher_earned_runs", "OVER", "e1", over_prob=0.6)
        k1 = _prow("pitcher_strikeouts", "OVER", "e1", over_prob=0.7)
        slate_pred, _mkt, _ra, _rs = pickrules_roi.rederive_slate([er1, k1], [], MLB)
        self.assertIn(prediction_identity(k1), slate_pred)
        self.assertNotIn(prediction_identity(er1), slate_pred)
        self.assertEqual(len(slate_pred), 1)

    def test_er_kept_when_k_not_analyzed(self):
        # Same ER-over, but K was never analyzed that date → rule cannot fire.
        er = _prow("pitcher_earned_runs", "OVER", "e2", over_prob=0.6,
                   game_date="2026-07-21", commence="2026-07-21T23:00:00Z")
        slate_pred, _mkt, _ra, _rs = pickrules_roi.rederive_slate([er], [], MLB)
        self.assertIn(prediction_identity(er), slate_pred)

    def test_doubleheader_distinct_events_not_paired(self):
        # Same pitcher name across two events same date: different event_id →
        # the ER/K rule must NOT cross-pair them; both stay on the slate.
        er = _prow("pitcher_earned_runs", "OVER", "dh1", over_prob=0.6)
        k = _prow("pitcher_strikeouts", "OVER", "dh2", over_prob=0.7)
        slate_pred, _m, _ra, _rs = pickrules_roi.rederive_slate([er, k], [], MLB)
        self.assertIn(prediction_identity(er), slate_pred)
        self.assertIn(prediction_identity(k), slate_pred)


class RuleOfThreeTests(unittest.TestCase):
    def test_caps_at_three_when_team_present(self):
        hits = [_prow("batter_hits", "OVER", "e1", player=f"B{i}", team="Yankees",
                      line=0.5, over_prob=0.65) for i in range(4)]
        slate_pred, _m, applied, skipped = pickrules_roi.rederive_slate(hits, [], MLB)
        on = [prediction_identity(h) for h in hits if prediction_identity(h) in slate_pred]
        self.assertEqual(len(on), 3)
        self.assertIn("Rule-of-3", applied)
        self.assertEqual(skipped, [])

    def test_team_none_exempts_and_is_reported_skipped(self):
        hits = [_prow("batter_hits", "OVER", "e1", player=f"B{i}", team=None,
                      line=0.5, over_prob=0.65) for i in range(4)]
        slate_pred, _m, applied, skipped = pickrules_roi.rederive_slate(hits, [], MLB)
        on = [prediction_identity(h) for h in hits if prediction_identity(h) in slate_pred]
        self.assertEqual(len(on), 4)  # cap exempts team=None
        self.assertNotIn("Rule-of-3", applied)
        self.assertTrue(any("Rule-of-3" in note for note in skipped))


class TotalOverPropUnderTests(unittest.TestCase):
    def test_total_over_drops_opposing_prop_under(self):
        total = _mrow("total", "over", "e1", model_prob=0.7)      # higher EV
        under = _prow("batter_hits", "UNDER", "e1", line=1.5, over_prob=0.4)
        slate_pred, slate_mkt, _ra, _rs = pickrules_roi.rederive_slate(
            [under], [total], MLB)
        self.assertIn(market_prediction_identity(total), slate_mkt)
        self.assertNotIn(prediction_identity(under), slate_pred)


class SlateVsPoolTests(unittest.TestCase):
    def test_slate_subset_of_pool_and_n_dropped(self):
        er = _prow("pitcher_earned_runs", "OVER", "e1", over_prob=0.6)
        k = _prow("pitcher_strikeouts", "OVER", "e1", over_prob=0.7)
        res = pickrules_roi.slate_vs_pool([er, k], [], MLB)
        self.assertEqual(res["props"]["pool"]["total"], 2)
        self.assertEqual(res["props"]["slate"]["total"], 1)  # ER dropped
        self.assertEqual(res["n_dropped"], 1)

    def test_dedup_safety_resolved_sibling_still_graded(self):
        # A resolved row + an unresolved re-log sibling share one identity. The
        # slate must still grade the resolved outcome (identity membership, not a
        # row-level flag that could pick the unresolved sibling).
        won = _prow("batter_hits", "OVER", "e1", price=100, outcome=1,
                    resolved=True, ts="2026-07-20T12:00:00Z")
        pending = dict(won)
        pending.update(resolved=False, outcome=None, actual=None,
                       ts="2026-07-20T18:00:00Z")  # newer, unresolved re-log
        res = pickrules_roi.slate_vs_pool([won, pending], [], MLB)
        self.assertEqual(res["props"]["slate"]["graded"], 1)
        self.assertEqual(res["props"]["slate"]["priced_resolved"], 1)
        self.assertAlmostEqual(res["props"]["slate"]["realized_roi"], 1.0)

    def test_combined_roi_is_priced_count_weighted(self):
        # Props: 1 win + 1 loss at +100 → roi 0.0 over 2 priced.
        # Market: 1 win at +100 → roi 1.0 over 1 priced.
        # Combined = (0*2 + 1*1)/3 = 0.3333; hit-rate = (0.5*2 + 1*1)/3 = 0.6667.
        p_win = _prow("batter_hits", "OVER", "e1", player="A", price=100, outcome=1)
        p_loss = _prow("total_bases", "OVER", "e1", player="B", price=100, outcome=0)
        m_win = _mrow("moneyline", "home", "e1", model_prob=0.6, price=100, outcome=1)
        res = pickrules_roi.slate_vs_pool([p_win, p_loss], [m_win], MLB)
        self.assertAlmostEqual(res["combined"]["pool"]["roi"], 1.0 / 3.0, places=4)
        self.assertAlmostEqual(res["combined"]["pool"]["hit_rate"], 2.0 / 3.0, places=4)

    def test_report_renders_without_error(self):
        er = _prow("pitcher_earned_runs", "OVER", "e1", over_prob=0.6)
        k = _prow("pitcher_strikeouts", "OVER", "e1", over_prob=0.7)
        # Smoke: printing must not raise on a populated result.
        _quiet_report(pickrules_roi.slate_vs_pool([er, k], [], MLB))

    def test_empty_inputs_are_safe(self):
        res = pickrules_roi.slate_vs_pool([], [], MLB)
        self.assertEqual(res["n_dropped"], 0)
        self.assertIsNone(res["delta"]["roi"])
        _quiet_report(res)  # must not raise on empty


if __name__ == "__main__":
    unittest.main()
