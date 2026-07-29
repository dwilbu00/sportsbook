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


def _ml_cand(team, opp, event_id, home_away="HOME", price=-110, edge=8.0):
    return {
        "is_value": True, "edge_pct": edge, "home_away": home_away,
        "team": team, "opponent": opp, "event_id": event_id,
        "best_edge_pct": edge, "best_price": price,
        "blended_prob": 70.0, "hist_prob": 70.0,
        "best_book_implied_prob": 50.0, "book_implied_prob": 50.0,
    }


def _prop_cand(player, prop, direction, event_id, team, matchup="Away @ Home",
               price=-110, edge=8.0, over_rate=65.0, batting_order=1, line=0.5):
    return {
        "is_value": True, "no_history": False, "edge_pct": edge,
        "games_sampled": 20, "direction": direction, "prop": prop,
        "player": player, "matchup": matchup, "team": team,
        "event_id": event_id, "prop_label": prop, "line": line,
        "over_rate": over_rate, "over_implied": 50.0, "under_implied": 50.0,
        "best_price": price, "over_price": price, "under_price": price,
        "batting_order": batting_order,
    }


def _total_cand(event_id, matchup, side="OVER", price=-110, over_hit=60.0):
    return {
        "matchup": matchup, "event_id": event_id, "line": 8.5,
        "is_over_value": side == "OVER", "is_under_value": side == "UNDER",
        "over_hit_rate": over_hit, "over_edge_pct": 8.0, "under_edge_pct": 8.0,
        "over_price": price, "under_price": price,
        "over_implied": 50.0, "under_implied": 50.0,
    }


class ParlayRuleAlignmentTests(unittest.TestCase):
    """The parlay generator must obey the SAME cross-bet rules as the single-bet
    auto-pick (bet_selector): L2 anti-correlation, L3 MLB contradictions, the
    Rule-of-3 team cap, the batting-order gate, event_id doubleheader identity,
    and the ER-over/K-over contradiction. Each test builds a 3-leg pool whose
    only combo either is or isn't rule-legal (mirrors MissingPriceParlayTests)."""

    def _gen(self, ml=None, spreads=None, totals=None, props=None):
        return parlay.generate_parlays(ml or [], spreads or [], totals or [],
                                       props or [], "baseball_mlb", mode="value")

    def test_anti_correlation_pair_blocked(self):
        # pitcher K OVER + game total OVER (same game) = -0.30 → blocked.
        k = _prop_cand("P", "pitcher_strikeouts", "OVER", "e1", "Home")
        tot = _total_cand("e1", "Away @ Home", side="OVER")
        ml = _ml_cand("C", "c", "e2")
        self.assertNotIn(3, self._gen(ml=[ml], totals=[tot], props=[k]))
        # Control: K OVER + total UNDER = +0.35 → allowed.
        tot_u = _total_cand("e1", "Away @ Home", side="UNDER")
        self.assertIn(3, self._gen(ml=[ml], totals=[tot_u], props=[k]))

    def test_l3_total_over_plus_prop_under_blocked(self):
        tot = _total_cand("e1", "Away @ Home", side="OVER")
        under = _prop_cand("B", "batter_hits", "UNDER", "e1", "Home", line=1.5)
        ml = _ml_cand("C", "c", "e2")
        self.assertNotIn(3, self._gen(ml=[ml], totals=[tot], props=[under]))

    def test_l3_pitcher_under_plus_opposing_hitter_under_blocked(self):
        pu = _prop_cand("P", "pitcher_strikeouts", "UNDER", "e1", "HomeTeam")
        hu = _prop_cand("B", "batter_hits", "UNDER", "e1", "AwayTeam", line=1.5)
        ml = _ml_cand("C", "c", "e2")
        self.assertNotIn(3, self._gen(ml=[ml], props=[pu, hu]))

    def test_er_over_plus_k_over_same_pitcher_blocked(self):
        er = _prop_cand("P", "pitcher_earned_runs", "OVER", "e1", "Home")
        k = _prop_cand("P", "pitcher_strikeouts", "OVER", "e1", "Home")
        ml = _ml_cand("C", "c", "e2")
        self.assertNotIn(3, self._gen(ml=[ml], props=[er, k]))
        # Control: different pitchers (both starters) → allowed.
        k_other = _prop_cand("Q", "pitcher_strikeouts", "OVER", "e1", "Away")
        self.assertIn(3, self._gen(ml=[ml], props=[er, k_other]))

    def test_rule_of_three_caps_batter_hits_overs(self):
        # 4 batter_hits OVER on one team → no 4-leg parlay; a 3-leg is fine.
        hits = [_prop_cand(f"B{i}", "batter_hits", "OVER", f"e{i}", "Yankees")
                for i in range(4)]
        results = parlay.generate_parlays([], [], [], hits, "baseball_mlb",
                                          mode="value")
        self.assertIn(3, results)
        self.assertNotIn(4, results)

    def test_batting_order_gate_drops_off_slot_hits_over(self):
        # A confirmed slot-7 batter_hits OVER is dropped → only 2 legs → nothing.
        off = _prop_cand("Deep", "batter_hits", "OVER", "e1", "T", batting_order=7)
        a = _ml_cand("A", "a", "e2")
        b = _ml_cand("B", "b", "e3")
        self.assertFalse(self._gen(ml=[a, b], props=[off]))
        # Control: slot 3 stays → 3 legs → a parlay is produced.
        ok = _prop_cand("Top", "batter_hits", "OVER", "e1", "T", batting_order=3)
        self.assertIn(3, self._gen(ml=[a, b], props=[ok]))

    def test_doubleheader_not_collapsed_by_event_id(self):
        # Same matchup string, two events (doubleheader): ML on each side must
        # NOT be treated as a same-game opposite-ML conflict.
        # Both legs normalize to the SAME matchup string "Red Sox @ Yankees"
        # (HOME → "opp @ team"; AWAY → "team @ opp"); only event_id separates
        # them. With team-name game keys this pair would be a false opposite-ML
        # conflict and no 3-leg parlay could form.
        g1 = _ml_cand("Yankees", "Red Sox", "g1", home_away="HOME")
        g2 = _ml_cand("Red Sox", "Yankees", "g2", home_away="AWAY")
        c = _ml_cand("Cubs", "Sox", "e3")
        self.assertIn(3, self._gen(ml=[g1, g2, c]))

    def test_pair_correlation_er_k_is_blocking(self):
        er = {"game_key": "e1", "bet_type": "player_prop_over", "team": "H",
              "player": "P", "prop_key": "pitcher_earned_runs"}
        k = {"game_key": "e1", "bet_type": "player_prop_over", "team": "H",
             "player": "P", "prop_key": "pitcher_strikeouts"}
        self.assertLessEqual(
            parlay._pair_correlation(er, k, "baseball_mlb"), -0.20)


class SameGamePairCountTests(unittest.TestCase):
    """`_same_game_pair_count` scales with how concentrated a parlay is in one
    game; `_same_game_penalty` turns that into a mode-scaled soft score haircut."""

    @staticmethod
    def _legs(*keys):
        return [{"game_key": k} for k in keys]

    def test_all_cross_game_is_zero(self):
        self.assertEqual(parlay._same_game_pair_count(self._legs("a", "b", "c")), 0)

    def test_two_in_one_game_is_one_pair(self):
        self.assertEqual(parlay._same_game_pair_count(self._legs("a", "a", "c")), 1)

    def test_three_in_one_game_is_three_pairs(self):
        self.assertEqual(parlay._same_game_pair_count(self._legs("a", "a", "a")), 3)

    def test_two_games_two_each_is_two_pairs(self):
        self.assertEqual(
            parlay._same_game_pair_count(self._legs("a", "a", "b", "b")), 2)

    def test_penalty_zero_for_cross_game(self):
        self.assertEqual(parlay._same_game_penalty(self._legs("a", "b"), "value"), 0.0)

    def test_penalty_scales_and_is_mode_aware(self):
        one_pair = self._legs("a", "a", "c")
        self.assertEqual(parlay._same_game_penalty(one_pair, "value"),
                         -parlay.SGP_PAIR_PENALTY)
        self.assertEqual(parlay._same_game_penalty(one_pair, "safe_value"),
                         -parlay.SGP_PAIR_PENALTY)
        self.assertEqual(parlay._same_game_penalty(one_pair, "safe"),
                         -parlay.SGP_SAFE_PENALTY)


class SameGamePenaltyBehaviorTests(unittest.TestCase):
    """The generator prefers diversified cross-game parlays but never hard-blocks
    a same-game parlay when it is the only valid option."""

    def test_cross_game_preferred_over_same_game_tie(self):
        # Pool: two batter_hits OVERs in one game (e1) + three cross-game MLs.
        # Every leg has identical value, so the ONLY thing separating a
        # same-game 3-combo from a cross-game one is the diversification
        # penalty. The winning 3-leg parlay must be cross-game.
        b1 = _prop_cand("B1", "batter_hits", "OVER", "e1", "Home",
                        batting_order=1, over_rate=70.0)
        b2 = _prop_cand("B2", "batter_hits", "OVER", "e1", "Home",
                        batting_order=2, over_rate=70.0)
        m2 = _ml_cand("T2", "o2", "e2")
        m3 = _ml_cand("T3", "o3", "e3")
        m4 = _ml_cand("T4", "o4", "e4")
        results = parlay.generate_parlays(
            [m2, m3, m4], [], [], [b1, b2], "baseball_mlb", mode="value")
        self.assertIn(3, results)
        self.assertFalse(results[3]["has_sgp"],
                         "expected the cross-game parlay to win the 3-leg slot")

    def test_same_game_parlay_still_allowed_when_only_option(self):
        # Only three legs, all in one game: the penalty applies but there is no
        # cross-game alternative, so the +value SGP is still returned (soft, not
        # a hard block).
        b1 = _prop_cand("B1", "batter_hits", "OVER", "e1", "Home",
                        batting_order=1, over_rate=70.0)
        b2 = _prop_cand("B2", "batter_hits", "OVER", "e1", "Home",
                        batting_order=2, over_rate=70.0)
        b3 = _prop_cand("B3", "batter_hits", "OVER", "e1", "Home",
                        batting_order=3, over_rate=70.0)
        results = parlay.generate_parlays(
            [], [], [], [b1, b2, b3], "baseball_mlb", mode="value")
        self.assertIn(3, results)
        self.assertTrue(results[3]["has_sgp"])


if __name__ == "__main__":
    unittest.main()
