"""Platt-fit outcome classification (audit F25).

fit_platt returns None for several DISTINCT reasons; a no-signal model must not be
conflated with a failed optimizer or a thin sample. Serving behavior is unchanged
(fit_platt still returns None for any non-'ok' reason).

Run: PYTHONIOENCODING=utf-8 python -m unittest test_recalibration_fit -v
"""
import random
import unittest

import recalibration as rc


class F25PlattFitReasonTests(unittest.TestCase):
    def test_insufficient_sample(self):
        self.assertEqual(rc.platt_fit_reason([0.6, 0.4], [1, 0]), "insufficient_sample")

    def test_no_class_variation(self):
        self.assertEqual(rc.platt_fit_reason([0.6] * 400, [1] * 400),
                         "no_class_variation")

    def test_no_signal_is_low_slope_not_optimizer_failure(self):
        random.seed(0)
        raw = [random.uniform(0.3, 0.7) for _ in range(400)]
        out = [random.randint(0, 1) for _ in range(400)]
        reason = rc.platt_fit_reason(raw, out)
        self.assertEqual(reason, "low_slope_weak_signal")
        self.assertNotEqual(reason, "optimizer_failure")   # the F25 conflation
        self.assertIsNone(rc.fit_platt(raw, out))          # serving unchanged: no map

    def test_real_signal_fits(self):
        raw = [0.2 if i % 5 else 0.85 for i in range(400)]
        out = [0 if i % 5 else 1 for i in range(400)]
        self.assertEqual(rc.platt_fit_reason(raw, out), "ok")
        self.assertIsNotNone(rc.fit_platt(raw, out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
