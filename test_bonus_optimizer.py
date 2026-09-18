"""Tests for the bonus optimizer's market-scoping + leg-count controls
(nfl_bonus_optimizer). The leg builders themselves need live data; these cover the
pure combinatorial/scoping logic that Phase 2 added."""
import unittest

import nfl_bonus_optimizer as opt
from bonus import Bonus


def _leg(gid, prop, player, P, odds):
    fav = P >= 0.5
    return {"gid": gid, "prop": prop, "player": player, "team": "X", "opp": "Y",
            "line": 0.5, "side": "OVER" if fav else "UNDER",
            "P": P, "fair_over": fav, "odds": odds}


class ScopeTests(unittest.TestCase):
    def test_empty_scope_returns_all(self):
        legs = [_leg(1, "batter_hits", "A", 0.6, -150),
                _leg(2, "batter_home_runs", "B", 0.55, -120)]
        self.assertEqual(len(opt._scope_legs(legs, ())), 2)

    def test_scope_filters_by_prop(self):
        legs = [_leg(1, "batter_hits", "A", 0.6, -150),
                _leg(2, "batter_home_runs", "B", 0.55, -120)]
        out = opt._scope_legs(legs, ("batter_home_runs",))
        self.assertEqual([l["prop"] for l in out], ["batter_home_runs"])

    def test_scope_with_no_matches_is_empty(self):
        legs = [_leg(1, "batter_hits", "A", 0.6, -150)]
        self.assertEqual(opt._scope_legs(legs, ("anytime_td",)), [])


class LegCountTests(unittest.TestCase):
    def _three_cross_game(self):
        return [_leg(1, "batter_hits", "A", 0.7, -150),
                _leg(2, "batter_hits", "B", 0.7, -150),
                _leg(3, "batter_hits", "C", 0.7, -150)]

    def test_leg_count_forces_exact_size(self):
        b = Bonus(bet_type="parlay", boost_pct=0.5, min_legs=2)
        plays = opt.cross_game_plays(self._three_cross_game(), b, 1000.0, leg_count=2)
        self.assertTrue(plays)
        self.assertTrue(all(len(combo) == 2 for _ev, _r, combo in plays))

    def test_leg_count_three(self):
        b = Bonus(bet_type="parlay", boost_pct=0.5, min_legs=2)
        plays = opt.cross_game_plays(self._three_cross_game(), b, 1000.0, leg_count=3)
        self.assertTrue(all(len(combo) == 3 for _ev, _r, combo in plays))

    def test_leg_count_below_type_min_yields_nothing(self):
        b = Bonus(bet_type="parlay", boost_pct=0.5, min_legs=2)   # parlay needs >=2
        self.assertEqual(
            opt.cross_game_plays(self._three_cross_game(), b, 1000.0, leg_count=1), [])

    def test_sgp_indep_leg_count(self):
        # two same-game legs -> a 2-leg SGP stack when leg_count=2
        legs = [_leg(9, "batter_hits", "A", 0.7, -150),
                _leg(9, "batter_total_bases", "A", 0.65, -140),
                _leg(9, "batter_rbis", "B", 0.6, -130)]
        b = Bonus(bet_type="sgp", boost_pct=0.5, min_legs=2)
        stacks = opt.sgp_stacks_indep(legs, b, 1000.0, leg_count=2)
        self.assertTrue(stacks)
        for _jp, _mx, combo, _need in stacks:
            self.assertEqual(len(combo), 2)


class EvaluateSlateScopeTests(unittest.TestCase):
    def test_per_bonus_market_scope_and_count(self):
        legs = {"draftkings": [
            _leg(1, "batter_hits", "A", 0.7, -150),
            _leg(2, "batter_home_runs", "B", 0.7, -150),
            _leg(3, "batter_home_runs", "C", 0.7, -150)]}
        hr = Bonus(bet_type="parlay", boost_pct=0.5, min_legs=2,
                   book="draftkings", markets=("batter_home_runs",))

        def _indep(lg, bn, leg_count=None):
            return opt.sgp_stacks_indep(lg, bn, 1000.0, leg_count)[:opt.TOP_K]

        res = opt.evaluate_slate(legs, [hr], None, 1000.0, sgp_fn=_indep, leg_count=2)
        r = res[0]
        self.assertEqual(r["n_legs"], 2)                  # only the 2 HR legs in scope
        self.assertTrue(r["cross"])
        for _ev, _rr, combo in r["cross"]:
            self.assertTrue(all(l["prop"] == "batter_home_runs" for l in combo))
            self.assertEqual(len(combo), 2)


if __name__ == "__main__":
    unittest.main()
