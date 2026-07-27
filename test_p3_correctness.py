"""P3 correctness fixes:

1. Unbounded output-side opponent-defense multiplier (props) — a sparse/garbage
   opp points-allowed or a calibrated strength > 1 could drive the projection to
   an extreme or negative value; now bounded to [0.5, 1.5].
2. Copula shrink was silent (parlay) — a non-PSD correlation matrix silently
   collapsed toward independence with no signal; the applied shrink is now
   surfaced.
3. Missing-price parlay legs (parlay) — value/safe_value parlays priced a leg
   with no executable price at a fabricated -110/1.91, which could clear the EV
   gate on a price the book never offered; such legs are now excluded.
"""

import unittest

import parlay
import pricing_common
import props


class TolerantTeamDefenseLookupTests(unittest.TestCase):
    """The runtime opp-defense lookup must tolerate name drift (feed name vs
    team_defense key) instead of failing open — parity with the backtest sweep."""

    DEF = {"New York Yankees": 4.1, "Los Angeles Angels": 5.2,
           "Boston Red Sox": 4.7}

    def test_exact_match(self):
        self.assertEqual(
            pricing_common._resolve_team_defense("New York Yankees", self.DEF), 4.1)

    def test_last_token_match(self):
        # feed says just "Yankees"; key is the full "New York Yankees".
        self.assertEqual(
            pricing_common._resolve_team_defense("Yankees", self.DEF), 4.1)

    def test_substring_and_case_insensitive(self):
        self.assertEqual(
            pricing_common._resolve_team_defense("la angels", self.DEF), 5.2)

    def test_no_match_returns_none(self):
        self.assertIsNone(
            pricing_common._resolve_team_defense("Toronto Blue Jays", self.DEF))

    def test_empty_inputs_return_none(self):
        self.assertIsNone(pricing_common._resolve_team_defense("", self.DEF))
        self.assertIsNone(pricing_common._resolve_team_defense("Yankees", None))
        self.assertIsNone(pricing_common._resolve_team_defense("Yankees", {}))

    def test_backtest_alias_is_shared_impl(self):
        import backtest
        self.assertIs(backtest._resolve_opp_pts_allowed,
                      pricing_common._resolve_team_defense)


class OutputDefenseMultiplierClampTests(unittest.TestCase):
    def test_normal_input_is_unclamped(self):
        # 5% softer defense, full strength -> 1.05, inside the bound.
        self.assertAlmostEqual(
            props._output_defense_multiplier(105.0, 100.0, 1.0), 1.05)

    def test_high_opp_pa_is_capped_at_1_5(self):
        self.assertEqual(
            props._output_defense_multiplier(10_000.0, 100.0, 1.0), 1.5)

    def test_strength_gt_1_cannot_go_negative(self):
        # opp_pa half the league avg with strength 5 would be 1+5*(-0.5)=-1.5
        # unclamped; must floor at 0.5 so the projection can't invert.
        self.assertEqual(
            props._output_defense_multiplier(50.0, 100.0, 5.0), 0.5)

    def test_missing_or_disabled_returns_neutral(self):
        self.assertEqual(props._output_defense_multiplier(None, 100.0, 1.0), 1.0)
        self.assertEqual(props._output_defense_multiplier(105.0, 0.0, 1.0), 1.0)
        self.assertEqual(props._output_defense_multiplier(105.0, 100.0, 0.0), 1.0)

    def test_always_within_bounds_across_extremes(self):
        for opp_pa in (1.0, 50.0, 99.0, 100.0, 150.0, 5000.0):
            for strength in (0.5, 1.0, 3.0):
                m = props._output_defense_multiplier(opp_pa, 100.0, strength)
                self.assertGreaterEqual(m, 0.5)
                self.assertLessEqual(m, 1.5)


# A valid-per-pair but jointly impossible correlation triangle (A~B +, A~C +,
# B~C -): its determinant is negative, so it is not positive semi-definite.
_NON_PSD = [[1.0, 0.8, 0.8], [0.8, 1.0, -0.8], [0.8, -0.8, 1.0]]
_PSD = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class CopulaShrinkVisibilityTests(unittest.TestCase):
    def test_make_psd_returns_shrink_below_one_for_non_psd(self):
        _L, shrink = parlay._make_psd_cholesky(_NON_PSD)
        self.assertLess(shrink, 1.0)

    def test_make_psd_returns_shrink_one_for_psd(self):
        _L, shrink = parlay._make_psd_cholesky(_PSD)
        self.assertEqual(shrink, 1.0)

    def test_gaussian_copula_default_return_is_scalar(self):
        # Backward-compat guard: without return_shrink the result stays a float
        # (test_money_math and _copula_joint_hit_prob rely on this).
        out = parlay._gaussian_copula_joint_prob([0.6, 0.6], _PSD)
        self.assertIsInstance(out, float)

    def test_gaussian_copula_reports_shrink_when_requested(self):
        prob, shrink = parlay._gaussian_copula_joint_prob(
            [0.5, 0.5, 0.5], _NON_PSD, n_samples=200, return_shrink=True)
        self.assertIsInstance(prob, float)
        self.assertLess(shrink, 1.0)

    def test_gaussian_copula_shrink_one_for_psd(self):
        _prob, shrink = parlay._gaussian_copula_joint_prob(
            [0.5, 0.5, 0.5], _PSD, n_samples=200, return_shrink=True)
        self.assertEqual(shrink, 1.0)


def _ml(team, opp, price, edge=8.0, hist=70.0, implied=50.0):
    """Minimal moneyline value candidate for _normalize_legs. Each distinct
    (team, opp) is a distinct game, so no two conflict."""
    return {
        "is_value": True, "edge_pct": edge, "home_away": "HOME",
        "team": team, "opponent": opp,
        "best_edge_pct": edge, "best_price": price,
        "blended_prob": hist, "hist_prob": hist,
        "best_book_implied_prob": implied, "book_implied_prob": implied,
    }


class MissingPriceParlayTests(unittest.TestCase):
    def _all_legs(self, results):
        return [leg for p in results.values() for leg in p["legs"]]

    def test_priceless_leg_excluded_from_value_parlays(self):
        # 3 priced value legs + 1 priceless. The priceless leg must never appear
        # in a value parlay, and no surfaced parlay may rely on a default price.
        ml = [_ml("A", "a", -110), _ml("B", "b", -110),
              _ml("C", "c", -110), _ml("D", "d", None)]
        results = parlay.generate_parlays(ml, [], [], [], "basketball_nba",
                                          mode="value")
        self.assertTrue(results, "expected at least one value parlay")
        for parlay_dict in results.values():
            self.assertFalse(parlay_dict["payout_uses_default_price"])
            for leg in parlay_dict["legs"]:
                self.assertIsNotNone(leg["odds_price"])
                self.assertNotEqual(leg["team"], "D")

    def test_no_value_parlay_when_only_combo_has_a_priceless_leg(self):
        # Exactly 3 legs, one priceless -> the only 3-combo is rejected -> no
        # fabricated-price parlay is surfaced.
        ml = [_ml("A", "a", -110), _ml("B", "b", -110), _ml("C", "c", None)]
        results = parlay.generate_parlays(ml, [], [], [], "basketball_nba",
                                          mode="value")
        self.assertNotIn(3, results)


if __name__ == "__main__":
    unittest.main()
