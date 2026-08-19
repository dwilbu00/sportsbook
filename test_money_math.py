"""Oracle and regression tests for the core money math.

Covers the primitives that decide which bets are shown as +EV — odds
conversion, expected ROI, the player-prop value gate (P1.1), the same-game
parlay payout neutralization (P1.2), the Gaussian-copula joint probability, and
the live Platt application. These are the highest-leverage surfaces in the app:
a regression here mislabels real-money bets, so each assertion pins a
hand-computed expected value rather than re-deriving through the code under test.
"""

import math
import unittest

import analysis
import odds_client
import wagers
from odds_client import american_to_decimal, american_to_implied_prob
from pricing_common import (kelly_fraction, kelly_stake, scale_to_slate_cap,
                            prob_interval_low, kelly_fraction_uncertain,
                            kelly_stake_uncertain)
from recalibration import apply_platt


class OddsConversionTests(unittest.TestCase):
    def test_american_to_decimal_table(self):
        self.assertAlmostEqual(american_to_decimal(-110), 1.9090909, places=6)
        self.assertEqual(american_to_decimal(100), 2.0)
        self.assertEqual(american_to_decimal(150), 2.5)
        self.assertEqual(american_to_decimal(-200), 1.5)
        self.assertEqual(american_to_decimal(-500), 1.2)
        self.assertEqual(american_to_decimal(300), 4.0)

    def test_implied_prob_is_inverse_of_decimal(self):
        # american_to_implied_prob(x) must equal 1 / american_to_decimal(x);
        # this exact-inverse property is what makes the ML/totals/spreads EV
        # gates safe.
        for a in (-500, -200, -110, 120, 150, 300):
            self.assertAlmostEqual(
                american_to_implied_prob(a),
                1.0 / american_to_decimal(a),
                places=9,
                msg=f"mismatch at {a}",
            )

    def test_decimal_to_american_roundtrip(self):
        for a in (-500, -200, -110, 120, 150, 300):
            d = american_to_decimal(a)
            self.assertEqual(odds_client._decimal_to_american(d), a)
            self.assertEqual(analysis._decimal_to_american(d), a)

    def test_expected_roi(self):
        # Fair coin at +100 is break-even; 55% at +100 returns +10%.
        self.assertAlmostEqual(analysis._expected_roi(0.5, 100), 0.0, places=9)
        self.assertAlmostEqual(analysis._expected_roi(0.55, 100), 0.10, places=9)
        # No executable price -> no ROI.
        self.assertIsNone(analysis._expected_roi(0.6, None))
        # Heavy favorite: 74.23% at -300 is actually -EV.
        self.assertLess(analysis._expected_roi(0.7423, -300), 0.0)


class PropValueGateTests(unittest.TestCase):
    """P1.1 — a prop must be +EV at the executable price, not just beat the
    de-vigged edge threshold."""

    def test_edge_clears_threshold_but_negative_ev_is_not_value(self):
        # Failure scenario: Over -300 (break-even 0.75). De-vigged fair 0.68,
        # model over-rate 0.74 -> edge +6.0% clears the 5% threshold, but the
        # bet is -EV at the price actually bettable.
        fair_over = 0.68
        over_rate = 0.74
        threshold = 0.05
        edge = over_rate - fair_over
        roi = analysis._expected_roi(over_rate, -300)
        self.assertGreaterEqual(edge, threshold)   # would pass an edge-only gate
        self.assertLess(roi, 0.0)                   # but is a losing bet
        self.assertFalse(analysis._prop_is_value(edge, threshold, roi))

    def test_positive_ev_and_edge_is_value(self):
        edge = 0.1077
        roi = analysis._expected_roi(0.80, -300)   # 0.80 * 1.3333 - 1 > 0
        self.assertGreater(roi, 0.0)
        self.assertTrue(analysis._prop_is_value(edge, 0.05, roi))

    def test_missing_price_is_not_value(self):
        self.assertFalse(analysis._prop_is_value(0.20, 0.05, None))

    def test_positive_ev_but_edge_below_threshold_is_not_value(self):
        roi = analysis._expected_roi(0.55, 100)     # +10% ROI
        self.assertGreater(roi, 0.0)
        self.assertFalse(analysis._prop_is_value(0.02, 0.05, roi))


class ParlayValueJointTests(unittest.TestCase):
    """P1.2 — same-game parlays must not credit the copula correlation benefit
    against the naive (independent) payout the book will not pay."""

    def test_sgp_prices_against_independent_joint(self):
        best_joint = 0.17          # copula, correlation-inflated
        independent = 0.125        # product of leg probs
        decimal_product = american_to_decimal(-110) ** 3  # ~6.96

        # Naive figure would look like strong value...
        self.assertGreater(best_joint * decimal_product - 1.0, 0.0)

        # ...but an SGP is priced against the independent joint, which is -EV.
        vj = analysis._parlay_value_joint(best_joint, independent, has_sgp=True)
        self.assertEqual(vj, independent)
        self.assertLess(vj * decimal_product - 1.0, 0.0)

    def test_cross_game_parlay_uses_copula_joint(self):
        best_joint = 0.17
        independent = 0.125
        vj = analysis._parlay_value_joint(best_joint, independent, has_sgp=False)
        self.assertEqual(vj, best_joint)


class DevigFairBaselineTests(unittest.TestCase):
    """P2a — every market measures edge against the de-vigged fair prob."""

    def test_symmetric_market_removes_hold(self):
        raw = american_to_implied_prob(-110)          # ~0.5238
        fair = analysis._devig_fair(raw, raw)
        self.assertAlmostEqual(fair, 0.5, places=6)
        self.assertLess(fair, raw)                     # hold removed

    def test_asymmetric_market(self):
        over = american_to_implied_prob(-200)          # 0.6667
        under = american_to_implied_prob(150)          # 0.40
        fair = analysis._devig_fair(over, under)
        self.assertAlmostEqual(fair, over / (over + under), places=9)

    def test_one_sided_market_falls_back_to_raw(self):
        raw = american_to_implied_prob(-110)
        self.assertEqual(analysis._devig_fair(raw, None), raw)

    def test_missing_side_is_none(self):
        self.assertIsNone(analysis._devig_fair(None, 0.5))


class CopulaInvariantTests(unittest.TestCase):
    N = 40000

    def _identity(self, n):
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def test_independence_recovers_product(self):
        joint = analysis._gaussian_copula_joint_prob(
            [0.5, 0.5, 0.5], self._identity(3), n_samples=self.N, seed=42)
        self.assertAlmostEqual(joint, 0.125, delta=0.01)

    def test_positive_correlation_raises_negative_lowers(self):
        probs = [0.5, 0.5]
        ident = self._identity(2)
        pos = [[1.0, 0.5], [0.5, 1.0]]
        neg = [[1.0, -0.5], [-0.5, 1.0]]
        j_ind = analysis._gaussian_copula_joint_prob(
            probs, ident, n_samples=self.N, seed=7)
        j_pos = analysis._gaussian_copula_joint_prob(
            probs, pos, n_samples=self.N, seed=7)
        j_neg = analysis._gaussian_copula_joint_prob(
            probs, neg, n_samples=self.N, seed=7)
        self.assertAlmostEqual(j_ind, 0.25, delta=0.01)
        self.assertGreater(j_pos, j_ind + 0.02)
        self.assertLess(j_neg, j_ind - 0.02)

    def test_boundaries(self):
        self.assertEqual(analysis._gaussian_copula_joint_prob([], []), 1.0)
        self.assertEqual(
            analysis._gaussian_copula_joint_prob([0.7], [[1.0]]), 0.7)
        self.assertEqual(
            analysis._gaussian_copula_joint_prob(
                [0.0, 0.5], self._identity(2)), 0.0)
        self.assertEqual(
            analysis._gaussian_copula_joint_prob(
                [1.0, 1.0], self._identity(2)), 1.0)


class ApplyPlattTests(unittest.TestCase):
    """The live per-prediction recalibration transform."""

    def test_identity_recovers_input(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            self.assertAlmostEqual(apply_platt(p, 1.0, 0.0), p, places=7)

    def test_none_passthrough(self):
        self.assertIsNone(apply_platt(None, 1.0, 0.0))
        self.assertEqual(apply_platt(0.7, None, None), 0.7)

    def test_monotonic_in_p(self):
        prev = -1.0
        for i in range(1, 20):
            v = apply_platt(0.05 * i, 1.3, -0.2)
            self.assertGreater(v, prev)
            prev = v

    def test_slope_sharpens_or_shrinks(self):
        # a > 1 pushes a >0.5 probability further from 0.5; a < 1 pulls toward.
        self.assertGreater(apply_platt(0.7, 2.0, 0.0), 0.7)
        self.assertLess(apply_platt(0.7, 0.5, 0.0), 0.7)
        self.assertGreater(apply_platt(0.7, 0.5, 0.0), 0.5)

    def test_boundary_probs_stay_finite(self):
        for p in (0.0, 1.0):
            v = apply_platt(p, 1.0, 0.0)
            self.assertTrue(math.isfinite(v))
            self.assertTrue(0.0 < v < 1.0)


class KellySizingTests(unittest.TestCase):
    """Vig-aware fractional-Kelly bet sizing (P-Kelly). Each assertion pins a
    hand-computed stake/fraction rather than re-deriving through the code, since a
    regression here mis-sizes real-money bets."""

    def test_kelly_fraction_basic_half_and_full(self):
        # p=0.55 at +100 (decimal 2.0, b=1.0): expected ROI 0.10, so full-Kelly
        # f* = 0.10; half-Kelly = 0.05. cap=1.0 leaves them uncapped.
        self.assertAlmostEqual(kelly_fraction(0.55, 100, 0.5, 1.0), 0.05, places=9)
        self.assertAlmostEqual(kelly_fraction(0.55, 100, 1.0, 1.0), 0.10, places=9)

    def test_kelly_fraction_reuses_american_to_decimal(self):
        # Pin b = decimal - 1 for both American signs (proves reuse of the odds
        # helper, not a hand-rolled conversion). p chosen so every leg is +EV.
        p = 0.65
        for a in (-150, 150, -110, 200):
            b = american_to_decimal(a) - 1.0
            er = p * (b + 1.0) - 1.0
            expected = 0.5 * er / b  # cap 1.0, all +EV so no clamp
            self.assertAlmostEqual(
                kelly_fraction(p, a, 0.5, 1.0), expected, places=9,
                msg=f"mismatch at {a}")

    def test_kelly_fraction_non_positive_ev_is_zero(self):
        # 0.40 at +100 is -EV -> no stake (mirrors the _prop_is_value EV gate).
        self.assertEqual(kelly_fraction(0.40, 100, 0.5, 0.05), 0.0)
        # Break-even (0.50 at +100, ROI exactly 0) also sizes to 0.
        self.assertEqual(kelly_fraction(0.50, 100, 0.5, 0.05), 0.0)

    def test_kelly_fraction_none_safe(self):
        self.assertEqual(kelly_fraction(0.60, None, 0.5, 0.05), 0.0)
        self.assertEqual(kelly_fraction(None, 100, 0.5, 0.05), 0.0)
        # Probability boundaries must not raise.
        for p in (0.0, 1.0):
            self.assertIsInstance(kelly_fraction(p, 100, 0.5, 0.05), float)

    def test_kelly_fraction_cap_clamp(self):
        # 0.90 at +100: full f* = 0.80, half = 0.40, both clamped to the 5% cap.
        self.assertAlmostEqual(kelly_fraction(0.90, 100, 0.5, 0.05), 0.05, places=9)
        self.assertAlmostEqual(kelly_fraction(0.90, 100, 1.0, 0.05), 0.05, places=9)

    def test_kelly_fraction_scales_linearly(self):
        # 0.60 at +100: full f* = 0.20 (cap 1.0). Fraction scales it linearly.
        self.assertAlmostEqual(kelly_fraction(0.60, 100, 0.25, 1.0), 0.05, places=9)
        self.assertAlmostEqual(kelly_fraction(0.60, 100, 0.50, 1.0), 0.10, places=9)
        self.assertAlmostEqual(kelly_fraction(0.60, 100, 1.00, 1.0), 0.20, places=9)
        self.assertEqual(kelly_fraction(0.60, 100, 0.0, 1.0), 0.0)

    def test_kelly_stake_dollars_and_rounding(self):
        # 0.52 at +100: full f* = 0.04, half = 0.02 (under the 5% cap).
        self.assertAlmostEqual(
            kelly_stake(0.52, 100, 1000.0, 0.5, 0.05), 20.00, places=2)
        # Rounds to cents: 333.33 * 0.02 = 6.6666 -> 6.67.
        self.assertAlmostEqual(
            kelly_stake(0.52, 100, 333.33, 0.5, 0.05), 6.67, places=2)

    def test_kelly_stake_non_positive_bankroll_or_ev(self):
        self.assertEqual(kelly_stake(0.60, 100, 0.0, 0.5, 0.05), 0.0)
        self.assertEqual(kelly_stake(0.60, 100, -500.0, 0.5, 0.05), 0.0)
        self.assertEqual(kelly_stake(0.40, 100, 1000.0, 0.5, 0.05), 0.0)
        self.assertEqual(kelly_stake(0.60, None, 1000.0, 0.5, 0.05), 0.0)

    def test_prob_interval_low_shrinks_with_thin_sample(self):
        # Wald interval: prob_low = p - z*sqrt(p(1-p)/n). Thinner n -> wider -> lower.
        p = 0.60
        wide = prob_interval_low(p, 25, z=1.0)    # se=sqrt(.24/25)=0.098 -> 0.502
        tight = prob_interval_low(p, 400, z=1.0)   # se=sqrt(.24/400)=0.0245 -> 0.5755
        self.assertAlmostEqual(wide, 0.60 - math.sqrt(0.24 / 25), places=6)
        self.assertAlmostEqual(tight, 0.60 - math.sqrt(0.24 / 400), places=6)
        self.assertLess(wide, tight)              # thin sample -> more conservative
        self.assertLess(tight, p)                 # always <= point estimate

    def test_prob_interval_low_fails_open(self):
        # Missing/degenerate n_eff or prob -> return the point estimate unchanged.
        self.assertEqual(prob_interval_low(0.60, 0), 0.60)
        self.assertEqual(prob_interval_low(0.60, None), 0.60)
        self.assertEqual(prob_interval_low(1.0, 100), 1.0)   # boundary, no raise
        self.assertEqual(prob_interval_low(None, 100), None)

    def test_uncertain_kelly_sizes_off_low_bound(self):
        # p=0.60 @ +100 point-Kelly (half) = 0.10; using prob_low=0.55 -> 0.05.
        self.assertAlmostEqual(
            kelly_fraction_uncertain(0.60, 0.55, 100, 0.5, 1.0), 0.05, places=9)
        # Never exceeds the point estimate (prob_low <= prob).
        self.assertLessEqual(
            kelly_fraction_uncertain(0.60, 0.55, 100, 0.5, 1.0),
            kelly_fraction(0.60, 100, 0.5, 1.0))

    def test_uncertain_kelly_abstains_when_interval_spans_breakeven(self):
        # Point prob 0.55 @ +100 is +EV (would bet), but if the low bound is 0.50
        # (break-even) or below, the interval spans break-even -> ABSTAIN (0.0).
        self.assertGreater(kelly_fraction(0.55, 100, 0.5, 1.0), 0.0)
        self.assertEqual(kelly_fraction_uncertain(0.55, 0.50, 100, 0.5, 1.0), 0.0)
        self.assertEqual(kelly_fraction_uncertain(0.55, 0.48, 100, 0.5, 1.0), 0.0)

    def test_uncertain_kelly_none_low_falls_back_to_point(self):
        # prob_low=None -> byte-identical to the point-estimate kelly_fraction.
        for p, a in ((0.55, 100), (0.62, -130), (0.40, 150)):
            self.assertEqual(
                kelly_fraction_uncertain(p, None, a, 0.5, 0.05),
                kelly_fraction(p, a, 0.5, 0.05))

    def test_uncertain_kelly_stake_dollars(self):
        # bankroll 1000, prob_low 0.55 @ +100 half-Kelly frac 0.05 -> $50.
        self.assertAlmostEqual(
            kelly_stake_uncertain(0.60, 0.55, 100, 1000.0, 0.5, 1.0), 50.00, places=2)
        self.assertEqual(
            kelly_stake_uncertain(0.55, 0.50, 100, 1000.0, 0.5, 1.0), 0.0)  # abstain
        self.assertEqual(
            kelly_stake_uncertain(0.60, 0.55, 100, 0.0, 0.5, 1.0), 0.0)  # no bankroll

    def test_scale_to_slate_cap_scales_down_proportionally(self):
        # Sum 60 > cap 25 (25% of 100) -> scale by 25/60.
        out = scale_to_slate_cap([10, 20, 30], 100.0, 0.25)
        self.assertAlmostEqual(sum(out), 25.00, places=2)
        self.assertAlmostEqual(out[0], 4.17, places=2)
        self.assertAlmostEqual(out[1], 8.33, places=2)
        self.assertAlmostEqual(out[2], 12.50, places=2)

    def test_scale_to_slate_cap_noop_within_cap(self):
        # Sum 10 <= cap 25 -> unchanged (just rounded).
        self.assertEqual(scale_to_slate_cap([5, 5], 100.0, 0.25), [5.0, 5.0])

    def test_scale_to_slate_cap_degenerate_inputs(self):
        # Zero bankroll -> cap 0 -> no scaling, stakes merely rounded.
        self.assertEqual(scale_to_slate_cap([10, 20], 0.0, 0.25), [10.0, 20.0])
        # None entries coerce to 0.0 and never raise.
        out = scale_to_slate_cap([10, None, 20], 100.0, 0.25)
        self.assertAlmostEqual(sum(out), 25.00, places=2)
        self.assertEqual(out[1], 0.0)

    def test_build_wager_row_kelly_sizes_stake(self):
        # End-to-end through the per-bet hook: model_prob 55% (-> 0.55) at the DK
        # price +100, bankroll 1000, half-Kelly, 5% cap -> f* 0.05 -> $50.00.
        cand = {
            "event_id": "E1", "direction": "OVER", "line": 0.5,
            "dk_over_price": 100, "over_price": 120,  # DK preferred over best-book
            "over_rate": 55.0, "player": "Test Batter",
            "prop": "batter_hits", "prop_label": "Hits", "edge_pct": 6.0,
            "matchup": "AWY @ HOM", "team": "HOM",
        }
        meta = {
            "sport_key": "americanfootball_nfl",  # skips MLB id-enrichment
            "event_id": "E1", "stake": 0.0, "placed_at": "2026-08-07T00:00:00+00:00",
            "seq": 0, "kelly": True, "bankroll": 1000.0,
            "kelly_fraction": 0.5, "kelly_cap": 0.05,
        }
        row = wagers.build_wager_row("player_prop", "OVER", cand, meta)
        self.assertIsNotNone(row)
        self.assertEqual(row["executed_price"], 100)  # DK, not the +120 best-book
        self.assertAlmostEqual(row["model_prob"], 0.55, places=9)
        self.assertAlmostEqual(row["stake"], 50.00, places=2)

    def test_build_wager_row_flat_when_kelly_off(self):
        # No meta['kelly'] -> the flat _blank_row stake is preserved (fail-open).
        cand = {
            "event_id": "E1", "direction": "OVER", "line": 0.5,
            "dk_over_price": 100, "over_rate": 55.0, "player": "Test Batter",
            "prop": "batter_hits", "prop_label": "Hits", "edge_pct": 6.0,
            "matchup": "AWY @ HOM", "team": "HOM",
        }
        meta = {
            "sport_key": "americanfootball_nfl", "event_id": "E1", "stake": 10.0,
            "placed_at": "2026-08-07T00:00:00+00:00", "seq": 0,
        }
        row = wagers.build_wager_row("player_prop", "OVER", cand, meta)
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["stake"], 10.0, places=2)


if __name__ == "__main__":
    unittest.main()
