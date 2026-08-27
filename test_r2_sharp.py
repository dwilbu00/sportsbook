"""Unit tests for r2_sharp — the Pinnacle-fair reference + cross-line projector.

Pure math: no warehouse, no I/O. Ground truth is built from the SAME survival
function the projector inverts (stats.negbin_at_least), so the round-trips are exact
up to the solver tolerance.
"""
import unittest

import r2_sharp as r2
from stats import negbin_at_least


class FairTwoWayTests(unittest.TestCase):
    def test_removes_vig_and_sums_to_one(self):
        fo, fu = r2.fair_two_way(-110, -110)
        self.assertAlmostEqual(fo + fu, 1.0, places=9)
        self.assertAlmostEqual(fo, 0.5, places=6)   # symmetric juice -> 50/50

    def test_favorite_keeps_higher_prob(self):
        fo, fu = r2.fair_two_way(-200, +170)
        self.assertGreater(fo, fu)
        self.assertAlmostEqual(fo + fu, 1.0, places=9)
        # fair favorite prob is below the vigged implied (vig inflated it)
        self.assertLess(fo, 200 / 300)

    def test_missing_side_returns_none(self):
        self.assertEqual(r2.fair_two_way(-120, None), (None, None))
        self.assertEqual(r2.fair_two_way(None, None), (None, None))


class SolveCountMeanTests(unittest.TestCase):
    def test_round_trip_poisson(self):
        for mean in (0.4, 1.2, 3.5, 8.5):
            for k in (1, 2, 3):
                s = negbin_at_least(k, mean, 0.0)
                got = r2.solve_count_mean(k, s, 0.0)
                self.assertAlmostEqual(got, mean, places=4)

    def test_round_trip_negbin(self):
        for mean in (1.0, 2.5, 9.0):
            for disp in (0.1, 0.5):
                for k in (1, 2, 4):
                    s = negbin_at_least(k, mean, disp)
                    got = r2.solve_count_mean(k, s, disp)
                    self.assertAlmostEqual(got, mean, places=3)

    def test_degenerate_targets(self):
        self.assertEqual(r2.solve_count_mean(1, 0.0, 0.0), 0.0)
        self.assertEqual(r2.solve_count_mean(1, -0.1, 0.0), 0.0)
        self.assertEqual(r2.solve_count_mean(1, 1.0, 0.0), float("inf"))
        self.assertEqual(r2.solve_count_mean(0, 0.5, 0.0), 0.0)   # k<=0 degenerate


class SameLineTests(unittest.TestCase):
    def test_same_line_is_pure_devig(self):
        offers = [{"point": 0.5, "over_price": -120, "under_price": +100}]
        sf = r2.fair_prob_at_line(offers, 0.5)
        self.assertFalse(sf.projected)
        self.assertEqual(sf.distance, 0.0)
        self.assertEqual(sf.n_lines, 1)
        expected, _ = r2.fair_two_way(-120, +100)
        self.assertAlmostEqual(sf.prob, expected, places=9)

    def test_no_offers_returns_none(self):
        self.assertIsNone(r2.fair_prob_at_line([], 0.5).prob)
        # one-sided offer can't be deviged -> no usable line
        self.assertIsNone(
            r2.fair_prob_at_line([{"point": 0.5, "over_price": -120}], 0.5).prob)


class CrossLineProjectionTests(unittest.TestCase):
    def test_bullseye_pinnacle_1p5_to_dk_0p5(self):
        # Pinnacle posts hits 1.5; DK posts 0.5. Project sharp fair to DK's line.
        offers = [{"point": 1.5, "over_price": +150, "under_price": -180}]
        at_1p5 = r2.fair_prob_at_line(offers, 1.5)     # exact
        at_0p5 = r2.fair_prob_at_line(offers, 0.5)     # projected
        self.assertFalse(at_1p5.projected)
        self.assertTrue(at_0p5.projected)
        self.assertEqual(at_0p5.distance, 1.0)
        # Survival decreases in the threshold: P(X>=1) > P(X>=2).
        self.assertGreater(at_0p5.prob, at_1p5.prob)
        # And it equals the survival at k=1 under the mean the 1.5 line implies.
        mean = r2.solve_count_mean(2, at_1p5.prob, 0.0)   # 1.5 -> k=2
        self.assertAlmostEqual(at_0p5.prob, negbin_at_least(1, mean, 0.0), places=6)

    def test_projection_is_monotone_across_lines(self):
        offers = [{"point": 1.5, "over_price": +100, "under_price": -120}]
        probs = [r2.fair_prob_at_line(offers, L).prob for L in (0.5, 1.5, 2.5, 3.5)]
        for a, b in zip(probs, probs[1:]):
            self.assertGreater(a, b)   # strictly decreasing in the line

    def test_multiline_shape_recovery(self):
        # Build two sharp lines from a KNOWN NegBin (mean=1.6, disp=0.3), then check
        # the fit recovers that shape and projects a third line to the true survival.
        mean, disp = 1.6, 0.3
        s0 = negbin_at_least(1, mean, disp)   # line 0.5
        s1 = negbin_at_least(2, mean, disp)   # line 1.5
        points = [(0.5, 1, s0), (1.5, 2, s1)]
        fit_mean, fit_disp = r2.fit_count_shape(points)
        self.assertAlmostEqual(fit_mean, mean, places=2)
        self.assertAlmostEqual(fit_disp, disp, places=2)
        # project to line 2.5 (k=3) and compare to the true survival
        proj = negbin_at_least(3, fit_mean, fit_disp)
        self.assertAlmostEqual(proj, negbin_at_least(3, mean, disp), places=3)

    def test_single_line_uses_default_dispersion(self):
        points = [(0.5, 1, 0.6)]
        mean, disp = r2.fit_count_shape(points, default_dispersion=0.2)
        self.assertEqual(disp, 0.2)
        self.assertAlmostEqual(negbin_at_least(1, mean, 0.2), 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
