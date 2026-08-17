"""Tests for xera_lite (the runs/9 fitter feeding expected_runs_additive)."""
import unittest

import xera_lite as xl


class OlsTests(unittest.TestCase):
    def test_recovers_linear_coeffs(self):
        X = [[1, 1], [2, 1], [3, 2], [4, 3], [5, 5], [1, 4], [2, 2], [6, 1],
             [3, 3], [4, 1]]
        y = [2.0 + 3.0 * a - 1.0 * b for a, b in X]
        intercept, coef = xl.fit_ols_multi(X, y, ridge=0.0)
        self.assertAlmostEqual(intercept, 2.0, places=4)
        self.assertAlmostEqual(coef[0], 3.0, places=4)
        self.assertAlmostEqual(coef[1], -1.0, places=4)

    def test_singular_returns_none(self):
        # Perfectly collinear feature (== intercept) with no ridge -> singular.
        X = [[1.0], [1.0], [1.0], [1.0]]
        y = [1.0, 2.0, 3.0, 4.0]
        self.assertIsNone(xl.fit_ols_multi(X, y, ridge=0.0))


class FitTests(unittest.TestCase):
    def _rows(self, n=40):
        rows = []
        for i in range(n):
            x = 0.30 + 0.003 * (i % 13)          # xwOBAcon spread
            k = 6.0 + 0.2 * (i % 11)             # K/9 spread
            rows.append({"xwobacon": x, "k9": k,
                         "label": 5.0 + 10.0 * x - 0.1 * k})
        return rows

    def test_fit_and_predict_recovers(self):
        rows = self._rows()
        rows.append({"xwobacon": None, "k9": 8.0, "label": 4.0})   # dropped
        rows.append({"xwobacon": 0.31, "k9": 8.0, "label": None})  # dropped
        m = xl.fit(rows, ["xwobacon", "k9"], ridge=1e-6)
        self.assertIsNotNone(m)
        self.assertEqual(m["n"], 40)
        # Predict near a training point; with tiny ridge it recovers the plane.
        p = xl.predict({"xwobacon": 0.33, "k9": 8.0}, m, n_sample=10_000)
        self.assertAlmostEqual(p, 5.0 + 10.0 * 0.33 - 0.1 * 8.0, places=1)

    def test_too_few_rows_returns_none(self):
        self.assertIsNone(xl.fit(self._rows(5), ["xwobacon", "k9"]))


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.m = {"feature_keys": ["x"], "intercept": 0.0, "coef": [10.0],
                  "league_rate9": 4.0, "n": 100}

    def test_sample_shrinkage(self):
        self.assertAlmostEqual(xl.predict({"x": 0.6}, self.m, n_sample=0), 4.0)
        self.assertAlmostEqual(
            xl.predict({"x": 0.6}, self.m, n_sample=10_000), 6.0, places=1)

    def test_missing_feature_returns_none(self):
        self.assertIsNone(xl.predict({"y": 1.0}, self.m))

    def test_clamped_range(self):
        hi = {"feature_keys": ["x"], "intercept": 50.0, "coef": [0.0],
              "league_rate9": 4.0, "n": 100}
        lo = {"feature_keys": ["x"], "intercept": -50.0, "coef": [0.0],
              "league_rate9": 4.0, "n": 100}
        self.assertEqual(xl.predict({"x": 1.0}, hi, n_sample=10_000), 9.0)
        self.assertEqual(xl.predict({"x": 1.0}, lo, n_sample=10_000), 1.5)


if __name__ == "__main__":
    unittest.main()
