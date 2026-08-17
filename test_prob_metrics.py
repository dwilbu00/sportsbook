"""Tests for prob_metrics.py — richer probability + betting metrics."""

import unittest

import prob_metrics as pm


class BrierSkillTests(unittest.TestCase):
    def test_bss_positive_when_model_beats_reference(self):
        outcomes = [1, 0, 1, 0]
        good = [0.9, 0.1, 0.8, 0.2]      # sharp + correct
        market = [0.6, 0.4, 0.55, 0.45]  # tamer
        self.assertGreater(pm.brier_skill_score(good, outcomes, market), 0)

    def test_bss_negative_when_model_worse(self):
        outcomes = [1, 0, 1, 0]
        bad = [0.4, 0.6, 0.45, 0.55]     # wrong-leaning
        market = [0.6, 0.4, 0.55, 0.45]
        self.assertLess(pm.brier_skill_score(bad, outcomes, market), 0)


class EceTests(unittest.TestCase):
    def test_perfect_calibration_zero(self):
        # 10 obs at 0.5 with exactly 5 wins -> conf==acc in that bin -> ECE 0
        probs = [0.5] * 10
        outcomes = [1, 0] * 5
        self.assertAlmostEqual(pm.ece(probs, outcomes, bins=10), 0.0, places=6)

    def test_overconfident_positive(self):
        probs = [0.95] * 10
        outcomes = [1, 0] * 5   # says 95%, hits 50%
        self.assertGreater(pm.ece(probs, outcomes), 0.4)


class CalibrationSlopeTests(unittest.TestCase):
    def test_calibrated_slope_near_one(self):
        probs, outcomes = [], []
        for p, wins in ((0.3, 30), (0.5, 50), (0.7, 70)):
            for k in range(100):
                probs.append(p)
                outcomes.append(1 if k < wins else 0)
        s = pm.calibration_slope(probs, outcomes)["slope"]
        self.assertTrue(0.7 <= s <= 1.3, f"slope {s} not ~1")

    def test_overconfident_slope_below_one(self):
        # predicts extreme 0.1/0.9 but only hits 0.3/0.7 -> overconfident -> slope<1
        probs, outcomes = [], []
        for p, wins in ((0.1, 30), (0.9, 70)):
            for k in range(100):
                probs.append(p)
                outcomes.append(1 if k < wins else 0)
        self.assertLess(pm.calibration_slope(probs, outcomes)["slope"], 1.0)


class EquityAndTierTests(unittest.TestCase):
    def test_equity_drawdown(self):
        # +1, -1, -1, +1 -> peak 1 at bet0, trough -1 at bet2 -> max dd 2
        r = [1.0, -1.0, -1.0, 1.0]
        s = pm.equity_stats(r)
        self.assertAlmostEqual(s["final_units"], 0.0)
        self.assertAlmostEqual(s["max_drawdown_units"], 2.0)

    def test_tier_monotonicity(self):
        up = pm.tier_monotonicity([-0.02, 0.01, 0.05, 0.09])
        self.assertTrue(up["monotonic"])
        self.assertAlmostEqual(up["rank_corr"], 1.0)
        self.assertGreater(up["top_minus_bottom"], 0)
        noisy = pm.tier_monotonicity([0.05, -0.03, 0.06, -0.02])
        self.assertFalse(noisy["monotonic"])


if __name__ == "__main__":
    unittest.main()
