"""Tests for the bonus/promo EV engine (bonus.py) — the actual +EV mechanism, so
its math + qualification + naming get real coverage."""
import unittest

import bonus
from bonus import Bonus


class MathTests(unittest.TestCase):
    def test_boosted_ev_pays_on_winnings_only(self):
        # coin flip at -110 (dec 1.909) with a 30% boost -> ~+9.1% (the canonical no.)
        ev = bonus.boosted_ev_per_dollar(0.50, bonus.american_to_dec(-110), 0.30)
        self.assertAlmostEqual(ev, 0.091, places=3)

    def test_no_boost_parlay_is_negative(self):
        dec = bonus.combined_decimal([-110, -110, -110])
        ev = bonus.boosted_ev_per_dollar(0.52 ** 3, dec, 0.0)
        self.assertLess(ev, 0.0)                       # vig compounds without a boost

    def test_kelly_zero_when_not_ev(self):
        # a true coin flip at -110 with NO boost is -EV -> Kelly 0
        self.assertEqual(bonus.kelly_fraction(0.50, bonus.american_to_dec(-110), 0.0),
                         0.0)

    def test_kelly_positive_when_boost_flips_ev(self):
        f = bonus.kelly_fraction(0.50, bonus.american_to_dec(-110), 0.30)
        self.assertGreater(f, 0.0)

    def test_american_dec_roundtrip(self):
        for a in (-250, -110, 100, 150, 300):
            self.assertAlmostEqual(bonus.dec_to_american(bonus.american_to_dec(a)), a,
                                   places=6)


class EvaluateTests(unittest.TestCase):
    def test_min_legs_gate_rejects_short_ticket(self):
        sgp = Bonus(bet_type="sgp", boost_pct=0.30, min_odds_leg=-250, min_legs=3)
        self.assertFalse(bonus.evaluate([(0.70, -180)] * 2, sgp)["qualifies"])
        self.assertTrue(bonus.evaluate([(0.70, -180)] * 3, sgp)["qualifies"])

    def test_single_type_rejects_parlay(self):
        b = Bonus(bet_type="single", boost_pct=0.30)
        self.assertFalse(bonus.evaluate([(0.6, -140), (0.6, -140)], b)["type_ok"])
        self.assertTrue(bonus.evaluate([(0.6, -140)], b)["type_ok"])

    def test_min_odds_leg_floor(self):
        # a -400 leg is SHORTER than a -300 floor -> disqualified
        b = Bonus(bet_type="any", boost_pct=0.25, min_odds_leg=-300)
        self.assertFalse(bonus.evaluate([(0.8, -400)], b)["legs_ok"])
        self.assertTrue(bonus.evaluate([(0.7, -250)], b)["legs_ok"])

    def test_joint_prob_override_used_for_sgp(self):
        b = Bonus(bet_type="sgp", boost_pct=0.30, min_legs=2)
        legs = [(0.6, -150), (0.6, -150)]
        indep = bonus.evaluate(legs, b)
        corr = bonus.evaluate(legs, b, joint_prob=0.50)   # correlated joint > product
        self.assertAlmostEqual(indep["joint_P"], 0.36, places=6)   # 0.6*0.6
        self.assertAlmostEqual(corr["joint_P"], 0.50, places=6)
        self.assertGreater(corr["boosted_ev_pct"], indep["boosted_ev_pct"])

    def test_stake_clamped_to_wager_bounds(self):
        b = Bonus(bet_type="any", boost_pct=0.50, max_wager=10.0, min_wager=1.0)
        r = bonus.evaluate([(0.7, -150), (0.7, -150)], b, bankroll=1e6)
        self.assertLessEqual(r["kelly_stake"], 10.0)      # capped
        self.assertGreater(r["kelly_stake"], 0.0)

    def test_negative_ev_stakes_zero(self):
        b = Bonus(bet_type="any", boost_pct=0.0, min_wager=5.0)   # no boost -> -EV
        r = bonus.evaluate([(0.5, -110)], b, bankroll=1000.0)
        self.assertEqual(r["kelly_stake"], 0.0)           # min_wager not forced on -EV


class NamingTests(unittest.TestCase):
    def test_display_name_with_scope(self):
        b = Bonus(bet_type="sgp", boost_pct=0.50, book="draftkings",
                  sport="baseball_mlb", markets=("batter_home_runs",))
        self.assertEqual(bonus.display_name(b), "DK · MLB · SGP 50% · HR")

    def test_display_name_no_scope(self):
        b = Bonus(bet_type="parlay", boost_pct=0.25, book="fanduel",
                  sport="americanfootball_nfl")
        self.assertEqual(bonus.display_name(b), "FD · NFL · PARLAY 25%")

    def test_scope_tag_caps_long_lists(self):
        tag = bonus._market_scope_tag(
            ("batter_hits", "batter_total_bases", "batter_rbis", "pitcher_strikeouts"))
        self.assertTrue(tag.endswith("+…"))

    def test_markets_default_empty(self):
        self.assertEqual(Bonus(bet_type="any", boost_pct=0.25).markets, ())


class F15OddsValidationTests(unittest.TestCase):
    def test_is_valid_american(self):
        for bad in (0, 50, -99, 99, float("nan"), float("inf"), float("-inf"), None):
            self.assertFalse(bonus.is_valid_american(bad), bad)
        for good in (-110, 100, -100, 250, -100000):
            self.assertTrue(bonus.is_valid_american(good), good)

    def test_american_to_dec_rejects_invalid_instead_of_zerodiv(self):
        with self.assertRaises(ValueError):
            bonus.american_to_dec(0)      # was ZeroDivisionError -> page crash [F15]
        with self.assertRaises(ValueError):
            bonus.american_to_dec(50)

    def test_evaluate_survives_invalid_stored_min_odds(self):
        b = Bonus(bet_type="single", boost_pct=.5, min_odds_leg=0.,
                  min_odds_overall=0., min_wager=1., max_wager=10.)
        r = bonus.evaluate([(.6, -110)], b, bankroll=100.)
        self.assertTrue(r["qualifies"])   # invalid min_odds = no constraint, not a crash

    def test_evaluate_survives_invalid_leg_odds(self):
        b = Bonus(bet_type="single", boost_pct=.5, min_wager=1., max_wager=10.)
        r = bonus.evaluate([(.6, 0)], b, bankroll=100.)
        self.assertFalse(r["qualifies"])
        self.assertEqual(r["kelly_stake"], 0.)


class F11SizingBoundsTests(unittest.TestCase):
    def test_zero_bankroll_yields_no_stake(self):
        # F11: min_wager must not override a zero bankroll.
        b = Bonus(bet_type="single", boost_pct=0.5, min_wager=10., max_wager=100.)
        self.assertEqual(bonus.evaluate([(.6, 100)], b, bankroll=0.)["kelly_stake"], 0.)

    def test_unavailable_or_nonfinite_bankroll_yields_no_stake(self):
        b = Bonus(bet_type="single", boost_pct=0.5, min_wager=10., max_wager=100.)
        for bad in (None, float("nan"), float("inf"), -50.):
            self.assertEqual(
                bonus.evaluate([(.6, 100)], b, bankroll=bad)["kelly_stake"], 0.)

    def test_bankroll_below_min_wager_cannot_afford_ticket(self):
        b = Bonus(bet_type="single", boost_pct=0.5, min_wager=10., max_wager=100.)
        self.assertEqual(bonus.evaluate([(.6, 100)], b, bankroll=5.)["kelly_stake"], 0.)

    def test_sufficient_bankroll_sizes_within_bounds(self):
        b = Bonus(bet_type="single", boost_pct=0.5, min_wager=10., max_wager=100.)
        stake = bonus.evaluate([(.6, 100)], b, bankroll=1000.)["kelly_stake"]
        self.assertGreaterEqual(stake, 10.)
        self.assertLessEqual(stake, 100.)


if __name__ == "__main__":
    unittest.main()
