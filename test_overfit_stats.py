"""Tests for overfit_stats.py — the backtest-overfit brakes."""

import unittest

import overfit_stats as ofs


class ExpectedMaxZTests(unittest.TestCase):
    def test_monotonic_and_endpoints(self):
        self.assertEqual(ofs.expected_max_z(1), 0.0)
        self.assertEqual(ofs.expected_max_z(0), 0.0)
        vals = [ofs.expected_max_z(n) for n in (2, 10, 100, 1000)]
        self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])))  # increasing
        # ~2.5 for N=100, ~2.6 for N~120 (the t-bar our 121-cell combo must clear)
        self.assertAlmostEqual(ofs.expected_max_z(100), 2.5, delta=0.2)
        self.assertGreater(ofs.expected_max_z(121), 2.5)


class DeflatedRoiTests(unittest.TestCase):
    def test_haircut_kills_a_best_of_many_winner(self):
        # a modest edge: 200 bets, mean +0.10/unit, unit-ish dispersion -> t ~ 1.4
        returns = ([0.91] * 110) + ([-1.0] * 90)   # 110 wins / 90 losses at ~ -110
        one = ofs.deflated_roi(returns, n_trials=1)
        many = ofs.deflated_roi(returns, n_trials=121)
        self.assertGreater(one["deflated_prob"], many["deflated_prob"])  # haircut bites
        self.assertEqual(one["noise_bar"], 0.0)
        self.assertGreater(many["noise_bar"], 2.5)
        # the SAME returns look credible as a single test but not as best-of-121
        self.assertFalse(many["credible"])

    def test_strong_edge_survives_haircut(self):
        # a big, real edge (mean +0.5/unit over 400 bets) clears even N=121
        returns = ([0.91] * 300) + ([-1.0] * 100)
        r = ofs.deflated_roi(returns, n_trials=121)
        self.assertTrue(r["credible"])
        self.assertGreater(r["deflated_prob"], 0.95)

    def test_none_on_thin(self):
        self.assertIsNone(ofs.deflated_roi([0.5], 10))


class PboTests(unittest.TestCase):
    def test_robust_config_low_pbo(self):
        # config 0 is better in EVERY block -> selection is not overfit -> PBO 0
        m = [[2.0, 1.0]] * 6
        r = ofs.pbo_cscv(m)
        self.assertEqual(r["pbo"], 0.0)
        self.assertEqual(r["n_blocks"], 6)

    def test_antagonistic_config_high_pbo(self):
        # each config wins only its own half of the blocks -> IS-best always craters
        # OOS -> PBO 1.0
        m = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        r = ofs.pbo_cscv(m)
        self.assertEqual(r["pbo"], 1.0)

    def test_guards(self):
        self.assertIsNone(ofs.pbo_cscv([[1, 2], [3, 4]]))          # S<4
        self.assertIsNone(ofs.pbo_cscv([[1, 2]] * 5))              # odd S
        self.assertIsNone(ofs.pbo_cscv([[1]] * 4))                 # <2 configs


if __name__ == "__main__":
    unittest.main()
