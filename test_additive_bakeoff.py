"""Tests for the additive bake-off wiring in backtest_starters (Tier A #1b).

Covers the pure/logic pieces that don't need SQL or a cached Statcast corpus:
label orientation, projector orientation, exp-IP clamping, grader shape.
"""
import unittest

import backtest_starters as bs


class ExpIpTests(unittest.TestCase):
    def test_clamp_and_default(self):
        self.assertEqual(bs._exp_ip(6.0), 6.0)
        self.assertEqual(bs._exp_ip(10.0), 7.0)      # clamp hi
        self.assertEqual(bs._exp_ip(1.0), 3.5)       # clamp lo
        self.assertEqual(bs._exp_ip(None), 5.2)      # default


class AdditiveTrainingRowsTests(unittest.TestCase):
    def test_label_orientation_and_drops(self):
        asof = {
            ("H", "2024-05-01"): {"xwobacon": 0.30, "k9": 8.0, "n_bbe": 100},
            ("A", "2024-05-01"): {"xwobacon": 0.40, "k9": 6.0, "n_bbe": 100},
            ("H2", "2024-05-02"): {"xwobacon": None, "k9": 7.0, "n_bbe": 5},  # dropped
        }
        rows = [
            {"home_sp": "H", "away_sp": "A", "date": "2024-05-01",
             "home_runs": 2.0, "away_runs": 5.0},
            {"home_sp": "H2", "away_sp": "A", "date": "2024-05-02",
             "home_runs": 3.0, "away_runs": 4.0},
        ]
        train = bs._additive_training_rows(rows, asof, ("xwobacon", "k9"))
        # Game 1: home SP "H" -> label = away_runs (5.0); away SP "A" -> home_runs (2.0).
        # Game 2: "H2" has null xwobacon -> dropped; "A" (2024-05-02) not in asof -> dropped.
        labels = sorted(r["label"] for r in train)
        self.assertEqual(labels, [2.0, 5.0])
        h = next(r for r in train if r["label"] == 5.0)   # the home SP row
        self.assertAlmostEqual(h["xwobacon"], 0.30)


class AdditiveProjectorTests(unittest.TestCase):
    def test_orientation_better_away_starter_suppresses_home(self):
        asof = {
            ("A", "2024-05-01"): {"xwobacon": 0.30, "k9": 8.0, "n_bbe": 200},  # better
            ("H", "2024-05-01"): {"xwobacon": 0.40, "k9": 6.0, "n_bbe": 200},  # worse
        }
        # rate9 rises with xwobacon (worse pitcher -> more runs).
        model = {"feature_keys": ["xwobacon", "k9"], "intercept": 0.0,
                 "coef": [10.0, 0.0], "league_rate9": 4.0, "n": 1000}
        proj = bs._make_additive_projector(asof, model, 4.0, ("xwobacon", "k9"))
        row = {"home_sp": "H", "away_sp": "A", "date": "2024-05-01",
               "a_ip": 6.0, "h_ip": 6.0, "a_off_faced": 1.0, "h_off_faced": 1.0}
        home_runs, away_runs = proj(row)
        # Home bats vs the BETTER away starter (A) -> fewer home runs than away runs.
        self.assertLess(home_runs, away_runs)

    def test_missing_features_fall_back_to_league(self):
        model = {"feature_keys": ["xwobacon", "k9"], "intercept": 0.0,
                 "coef": [10.0, 0.0], "league_rate9": 4.3, "n": 1000}
        proj = bs._make_additive_projector({}, model, 4.3, ("xwobacon", "k9"))
        row = {"home_sp": "H", "away_sp": "A", "date": "2024-05-01",
               "a_ip": None, "h_ip": None, "a_off_faced": 1.0, "h_off_faced": 1.0}
        hr, ar = proj(row)   # both starters missing -> league_bp both sides
        self.assertAlmostEqual(hr, ar)          # symmetric fallback
        self.assertAlmostEqual(hr, 4.3, places=3)


class GraderShapeTests(unittest.TestCase):
    def test_variant_metrics_projfn_shape(self):
        rows = [
            {"home_win": 1, "margin": 2.0, "home_runs": 5.0, "away_runs": 3.0,
             "total_runs": 8.0},
            {"home_win": 0, "margin": -1.0, "home_runs": 3.0, "away_runs": 4.0,
             "total_runs": 7.0},
            {"home_win": 1, "margin": 3.0, "home_runs": 6.0, "away_runs": 3.0,
             "total_runs": 9.0},
        ]
        m = bs._variant_metrics_projfn(
            rows, rows, lambda r: (r["home_runs"], r["away_runs"]))
        for key in ("ml", "spread", "margin", "total_rmse", "score_nll"):
            self.assertIn(key, m)
        self.assertIn("brier", m["ml"])
        self.assertIn("rmse", m["margin"])
        self.assertGreaterEqual(m["score_nll"], 0.0)


if __name__ == "__main__":
    unittest.main()
