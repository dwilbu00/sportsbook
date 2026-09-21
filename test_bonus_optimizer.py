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


class F12DeterministicConflictTests(unittest.TestCase):
    @staticmethod
    def _mk(prop, player, P, side, line):
        return {"gid": "g", "prop": prop, "player": player, "team": "A", "opp": "B",
                "line": line, "side": side, "P": P, "fair_over": side == "OVER",
                "odds": -110}

    def test_hits_over_and_total_bases_under_same_player_rejected(self):
        # >=2 hits forces >=2 total bases, so UNDER 0.5 TB is impossible. [F12]
        b = Bonus("sgp", .5, min_legs=2)
        legs = [self._mk("batter_hits", "A", .7, "OVER", 1.5),
                self._mk("batter_total_bases", "A", .6, "UNDER", 0.5)]
        self.assertEqual(opt.sgp_stacks_indep(legs, b, 1000.), [])

    def test_same_stat_opposite_sides_rejected(self):
        b = Bonus("sgp", .5, min_legs=2)
        legs = [self._mk("batter_hits", "A", .7, "OVER", 1.5),
                self._mk("batter_hits", "A", .6, "UNDER", 0.5)]
        self.assertEqual(opt.sgp_stacks_indep(legs, b, 1000.), [])

    def test_compatible_same_player_combo_builds(self):
        b = Bonus("sgp", .5, min_legs=2)
        legs = [self._mk("batter_hits", "A", .7, "OVER", 0.5),
                self._mk("batter_total_bases", "A", .6, "OVER", 1.5)]
        self.assertEqual(len(opt.sgp_stacks_indep(legs, b, 1000.)), 1)

    def test_different_players_never_conflict(self):
        b = Bonus("sgp", .5, min_legs=2)
        legs = [self._mk("batter_hits", "A", .7, "OVER", 1.5),
                self._mk("batter_total_bases", "B", .6, "UNDER", 0.5)]
        self.assertEqual(len(opt.sgp_stacks_indep(legs, b, 1000.)), 1)


class F10FrontierTests(unittest.TestCase):
    def test_game_diversity_survives_top_probability_prefilter(self):
        # F10: 18 legs from one game + 1 from another. A plain top-N-by-P prefilter
        # fills the budget with the single game and returns no cross-game ticket; the
        # game-diverse frontier keeps the second game feasible.
        b = Bonus("parlay", .5, min_legs=2)
        legs = [_leg("g1", "batter_hits", str(i), .80, -150) for i in range(18)]
        legs.append(_leg("g2", "batter_hits", "Other", .79, -150))
        out = opt.cross_game_plays(legs, b, 1000., leg_count=2)
        self.assertTrue(out)
        self.assertTrue(any(len({l["gid"] for l in combo}) == 2
                            for _ev, _r, combo in out))


class F09SgpxCompositionTests(unittest.TestCase):
    def _slate(self, bt):
        b = Bonus(bt, .5, min_legs=2)
        legs = {"draftkings": [_leg("g1", "batter_hits", "A", .7, -150),
                               _leg("g2", "batter_hits", "B", .7, -150)]}
        return opt.evaluate_slate(legs, [b], None, 1000., sgp_fn=lambda *a: [])

    def test_sgpx_does_not_emit_plain_cross_game(self):
        # F09: an SGP+ promo must not surface plain cross-game tickets (no same-game
        # component) — those aren't valid SGP+ construction.
        self.assertEqual(self._slate("sgp_sgpx")[0]["cross"], [])

    def test_plain_parlay_still_builds_cross_game(self):
        self.assertTrue(self._slate("parlay")[0]["cross"])


class F08SgpEligibilityTests(unittest.TestCase):
    def test_forced_leg_count_below_min_legs_is_rejected(self):
        # F08: an SGP promo requiring 3 legs must not yield a 2-leg ticket, even when
        # the leg count is forced to 2.
        b = Bonus("sgp", .5, min_legs=3)
        legs = [_leg("g", "batter_hits", "A", .7, -150),
                _leg("g", "batter_hits", "B", .7, -150)]
        self.assertEqual(opt.sgp_stacks_indep(legs, b, 1000., leg_count=2), [])

    def test_valid_forced_count_at_min_legs_builds(self):
        b = Bonus("sgp", .5, min_legs=3)
        legs = [_leg("g", "batter_hits", p, .7, -150) for p in ("A", "B", "C")]
        out = opt.sgp_stacks_indep(legs, b, 1000., leg_count=3)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0][2]), 3)

    def test_copula_sgp_also_honors_min_legs(self):
        b = Bonus("sgp", .5, min_legs=3)
        legs = [_leg("g", "batter_hits", "A", .7, -150),
                _leg("g", "batter_hits", "B", .7, -150)]
        self.assertEqual(opt.sgp_stacks(legs, b, None, 1000., leg_count=2), [])


class F07LegIdentityTests(unittest.TestCase):
    def test_mlb_reuse_preserves_settlement_time_and_date(self):
        # F07: the MLB candidate-reuse path must carry commence_time/game_date onto
        # the leg so a logged parlay leg is gradable (leg_to_store persists them).
        from unittest.mock import patch
        c = dict(type="player_prop", prop="batter_hits", player="Fixture",
                 event_id="e", line=.5, batting_order=1, lineup_status="in",
                 over_implied=70., dk_over_price=-150,
                 commence_time="2026-09-20T00:10:00Z", game_date="2026-09-19",
                 team="A")
        with patch("book_calibration.load_maps", return_value={}):
            legs = opt.legs_from_candidates([c], "draftkings")
        self.assertTrue(legs)
        row = opt.leg_to_store(legs[0], sport="baseball_mlb")
        self.assertEqual(row["commence_time"], c["commence_time"])
        self.assertEqual(row["game_date"], c["game_date"])


if __name__ == "__main__":
    unittest.main()
