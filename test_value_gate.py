"""
Recommendation-gate decision (props._prop_gate_is_value) + value_gate IO.

The ROI-primary gate (EV >= ev_floor AND edge >= edge_floor) replaces the legacy
flat edge-threshold gate for sports with a calibrated value_gate. It was chosen
by refit_calibration --gate-diag because a flat edge floor wrongly suppressed
high-ROI / long-odds bets. These tests pin the decision logic + the DK-alone
edge bump + per-prop suppression + the legacy fallback, and the calibration IO.
"""
import tempfile
import unittest

import calibration_loader as cl
from props import (_prop_gate_is_value, _DK_SELFDEVIG_EDGE_MULT, _gate_floor_mult,
                   _DEFAULT_GATE_TIME_BANDS, _DEFAULT_GATE_LONGSHOT,
                   _recency_weighted_cv, _CV_MIN_PRIOR)
from stats import _recency_weights, _weighted_mean


class PropGateDecisionTests(unittest.TestCase):
    # ── ROI-primary gate (ev_floor configured) ─────────────────────────
    def _roi(self, edge, ev, prop="batter_hits", suppress=(), dk_alone=False):
        return _prop_gate_is_value(
            edge, ev, prop, ev_floor=0.04, edge_floor=0.01, suppress=suppress,
            legacy_threshold=0.05, dk_alone=dk_alone)

    def test_roi_gate_passes_high_ev_low_edge(self):
        # The user's case: a +10% EV bet at only 3% edge — suppressed by the old
        # 5% edge floor, allowed here.
        self.assertTrue(self._roi(edge=0.03, ev=0.10))

    def test_roi_gate_rejects_below_ev_floor(self):
        # Big edge but EV under the floor → not value (profit-led).
        self.assertFalse(self._roi(edge=0.20, ev=0.03))

    def test_roi_gate_rejects_below_edge_floor(self):
        # Big EV but the model barely agrees (edge < 1%) → not value.
        self.assertFalse(self._roi(edge=0.005, ev=0.20))

    def test_roi_gate_rejects_none_ev(self):
        self.assertFalse(self._roi(edge=0.10, ev=None))

    def test_roi_gate_boundary_inclusive(self):
        self.assertTrue(self._roi(edge=0.01, ev=0.04))       # exactly at floors

    def test_roi_gate_dk_alone_doubles_edge_floor(self):
        # edge 1.5% clears the 1% floor normally, but not the doubled 2% DK-alone
        # floor.
        self.assertTrue(self._roi(edge=0.015, ev=0.10, dk_alone=False))
        self.assertFalse(self._roi(edge=0.015, ev=0.10, dk_alone=True))
        self.assertTrue(self._roi(edge=0.025, ev=0.10, dk_alone=True))

    def test_roi_gate_dk_alone_doubles_ev_floor(self):
        # EV 5% clears the 4% floor normally, but not the doubled 8% DK-alone floor
        # (edge 5% clears both the 1% and doubled 2% edge floors, isolating the EV bump).
        self.assertTrue(self._roi(edge=0.05, ev=0.05, dk_alone=False))
        self.assertFalse(self._roi(edge=0.05, ev=0.05, dk_alone=True))
        self.assertTrue(self._roi(edge=0.05, ev=0.09, dk_alone=True))

    def test_suppress_blocks_value(self):
        self.assertFalse(self._roi(edge=0.20, ev=0.20, prop="pitcher_outs",
                                   suppress=("pitcher_outs",)))

    # ── legacy edge-threshold gate (ev_floor=None) ─────────────────────
    def _legacy(self, edge, ev, prop="batter_hits", suppress=(), dk_alone=False):
        return _prop_gate_is_value(
            edge, ev, prop, ev_floor=None, edge_floor=0.0, suppress=suppress,
            legacy_threshold=0.05, dk_alone=dk_alone)

    def test_legacy_requires_edge_and_positive_ev(self):
        self.assertTrue(self._legacy(edge=0.06, ev=0.02))
        self.assertFalse(self._legacy(edge=0.04, ev=0.02))     # edge < 5%
        self.assertFalse(self._legacy(edge=0.06, ev=-0.01))    # not +EV
        self.assertFalse(self._legacy(edge=0.06, ev=None))     # no price

    def test_legacy_dk_alone_doubles_threshold(self):
        # 6% edge clears 5% but not the doubled 10% DK-alone threshold.
        self.assertTrue(self._legacy(edge=0.06, ev=0.02, dk_alone=False))
        self.assertFalse(self._legacy(edge=0.06, ev=0.02, dk_alone=True))
        self.assertTrue(self._legacy(edge=0.11, ev=0.02, dk_alone=True))

    def test_legacy_suppress_blocks_value(self):
        self.assertFalse(self._legacy(edge=0.20, ev=0.20, prop="pitcher_outs",
                                      suppress=("pitcher_outs",)))

    def test_mult_constant_is_two(self):
        self.assertEqual(_DK_SELFDEVIG_EDGE_MULT, 2.0)


class R5SelectivityTests(unittest.TestCase):
    """R5: the gate scales its edge/EV floors UP the earlier the bet (early edges
    revert) and on longshot plus-money legs. hours_to_pitch/price None or no config
    → multiplier 1.0 (pre-R5 behavior)."""

    def test_mult_no_inputs_is_one(self):
        self.assertEqual(_gate_floor_mult(None, None, None, None), 1.0)
        self.assertEqual(_gate_floor_mult(None, None, _DEFAULT_GATE_TIME_BANDS,
                                          _DEFAULT_GATE_LONGSHOT), 1.0)

    def test_mult_time_bands(self):
        tb = _DEFAULT_GATE_TIME_BANDS   # [(2,1.0),(6,1.5),(inf,2.0)]
        self.assertEqual(_gate_floor_mult(1.0, None, tb, None), 1.0)   # <2h
        self.assertEqual(_gate_floor_mult(4.0, None, tb, None), 1.5)   # 2-6h
        self.assertEqual(_gate_floor_mult(9.0, None, tb, None), 2.0)   # >6h (early)

    def test_mult_longshot_surcharge(self):
        ls = _DEFAULT_GATE_LONGSHOT   # (150, 1.5)
        self.assertEqual(_gate_floor_mult(None, 200, None, ls), 1.5)   # +200 longshot
        self.assertEqual(_gate_floor_mult(None, -150, None, ls), 1.0)  # favorite
        self.assertEqual(_gate_floor_mult(None, 120, None, ls), 1.0)   # +120 < +150

    def test_mult_stacks_early_and_longshot(self):
        self.assertEqual(_gate_floor_mult(9.0, 200, _DEFAULT_GATE_TIME_BANDS,
                                          _DEFAULT_GATE_LONGSHOT), 3.0)  # 2.0 * 1.5

    def test_gate_early_demands_more_edge(self):
        base = dict(prop_key="batter_hits", ev_floor=0.04, edge_floor=0.01,
                    suppress=(), legacy_threshold=0.05, dk_alone=False,
                    time_bands=_DEFAULT_GATE_TIME_BANDS, longshot=None)
        # <2h: mult 1.0 → 1% edge / 4% EV floors → a 1.5%-edge, 5%-EV bet passes.
        self.assertTrue(_prop_gate_is_value(0.015, 0.05, hours_to_pitch=1.0, **base))
        # 9h out: floors x2 → 2% edge floor rejects the same 1.5%-edge bet.
        self.assertFalse(_prop_gate_is_value(0.015, 0.05, hours_to_pitch=9.0, **base))

    def test_gate_backward_compatible_when_unscaled(self):
        # No R5 inputs → identical to the pre-R5 decision.
        self.assertTrue(_prop_gate_is_value(
            0.02, 0.05, "batter_hits", 0.04, 0.01, (), 0.05, False))


class ValueGateIOTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._orig = cl.CALIBRATION_DIR
        cl.CALIBRATION_DIR = self._dir
        cl.set_candidate_mode(False)

    def tearDown(self):
        cl.CALIBRATION_DIR = self._orig
        cl.set_candidate_mode(False)

    def test_roundtrip_and_default_empty(self):
        self.assertEqual(cl.load_value_gate("baseball_mlb"), {})
        gate = {"ev_floor": 0.04, "edge_floor": 0.01, "suppress": ["pitcher_outs"]}
        cl.save_value_gate("baseball_mlb", gate)
        self.assertEqual(cl.load_value_gate("baseball_mlb"), gate)

    def test_load_coerces_bad_field_types(self):
        import json
        # A hand/mis-configured block: string suppress, string ev_floor, bool edge.
        cl.save_value_gate("baseball_mlb", {"ev_floor": 0.04})  # seed a valid blob
        blob = json.load(open(cl.calibration_path("baseball_mlb")))
        blob["value_gate"] = {"ev_floor": "0.04", "edge_floor": True,
                              "suppress": "pitcher_outs"}
        json.dump(blob, open(cl.calibration_path("baseball_mlb"), "w"))
        gate = cl.load_value_gate("baseball_mlb")
        # string ev_floor + bool edge_floor dropped (fail safe → legacy gate);
        # bare-string suppress becomes a single-key list (not per-character).
        self.assertNotIn("ev_floor", gate)
        self.assertNotIn("edge_floor", gate)
        self.assertEqual(gate["suppress"], ["pitcher_outs"])

    def test_load_non_dict_is_empty(self):
        import json
        cl.save_value_gate("baseball_mlb", {"ev_floor": 0.04})
        blob = json.load(open(cl.calibration_path("baseball_mlb")))
        blob["value_gate"] = ["not", "a", "dict"]
        json.dump(blob, open(cl.calibration_path("baseball_mlb"), "w"))
        self.assertEqual(cl.load_value_gate("baseball_mlb"), {})

    def test_save_preserves_other_blocks(self):
        cl.save_calibration("baseball_mlb", {"batter_hits": {"method": "E"}})
        cl.save_prob_shrink("baseball_mlb", {"moneyline": 0.25})
        cl.save_value_gate("baseball_mlb", {"ev_floor": 0.04})
        import json
        blob = json.load(open(cl.calibration_path("baseball_mlb")))
        self.assertEqual(blob["props"]["batter_hits"]["method"], "E")
        self.assertEqual(blob["prob_shrink"], {"moneyline": 0.25})
        self.assertEqual(blob["value_gate"], {"ev_floor": 0.04})

    def test_load_parses_r5_config(self):
        import json
        cl.save_value_gate("baseball_mlb", {"ev_floor": 0.04})
        blob = json.load(open(cl.calibration_path("baseball_mlb")))
        blob["value_gate"]["time_bands"] = [[2, 1.0], [6, 1.5], ["bad", 2]]
        blob["value_gate"]["longshot_surcharge"] = [150, 1.5]
        json.dump(blob, open(cl.calibration_path("baseball_mlb"), "w"))
        gate = cl.load_value_gate("baseball_mlb")
        self.assertEqual(gate["time_bands"], [(2.0, 1.0), (6.0, 1.5)])  # bad row dropped
        self.assertEqual(gate["longshot_surcharge"], (150.0, 1.5))

    def test_load_parses_cv_floor(self):
        import json
        cl.save_value_gate("baseball_mlb", {"ev_floor": 0.04})
        blob = json.load(open(cl.calibration_path("baseball_mlb")))
        # good numeric floor kept; non-numeric entry dropped (fail safe → not CV-gated)
        blob["value_gate"]["cv_floor"] = {"pitcher_earned_runs": 1.3, "bad": "x"}
        json.dump(blob, open(cl.calibration_path("baseball_mlb"), "w"))
        gate = cl.load_value_gate("baseball_mlb")
        self.assertEqual(gate["cv_floor"], {"pitcher_earned_runs": 1.3})

    def test_cv_floor_absent_is_inert(self):
        cl.save_value_gate("baseball_mlb", {"ev_floor": 0.04})
        self.assertNotIn("cv_floor", cl.load_value_gate("baseball_mlb"))


class RecencyWeightedCVTests(unittest.TestCase):
    """props._recency_weighted_cv must reproduce backtest.py's inline volatility
    formula EXACTLY (the lens the earned_runs CV>=1.3 edge was validated on) and
    fail CLOSED (None) on inputs where a high-CV floor can't be confirmed."""

    @staticmethod
    def _backtest_cv(prior, hl):
        # mirror of backtest.py:2639-2648
        wts = _recency_weights(len(prior), hl)
        raw_mean = _weighted_mean(prior, wts)
        sw = sum(wts) or 1.0
        var = sum(w * (v - raw_mean) ** 2 for v, w in zip(prior, wts)) / sw
        return (var ** 0.5 / raw_mean) if raw_mean > 1e-9 else 0.0

    def test_matches_backtest_formula(self):
        for hl in (None, 7, 15):
            for series in ([3, 0, 5, 2, 8, 1, 4, 6, 0, 7],
                           [4, 4, 4, 4, 4],
                           [2.5, 3.1, 0.0, 9.0, 1.0, 2.0]):
                got = _recency_weighted_cv(series, hl)
                self.assertIsNotNone(got)
                self.assertAlmostEqual(got, self._backtest_cv(series, hl), places=12)

    def test_fails_closed_on_short_or_degenerate(self):
        self.assertIsNone(_recency_weighted_cv([1, 2, 3, 4], None))   # < _CV_MIN_PRIOR
        self.assertIsNone(_recency_weighted_cv([], None))
        self.assertIsNone(_recency_weighted_cv(None, None))
        self.assertIsNone(_recency_weighted_cv([0, 0, 0, 0, 0, 0], None))  # zero mean

    def test_min_prior_boundary(self):
        self.assertEqual(_CV_MIN_PRIOR, 5)
        self.assertIsNone(_recency_weighted_cv([1, 2, 3, 4], None))       # 4 < 5
        self.assertIsNotNone(_recency_weighted_cv([1, 2, 3, 4, 5], None))  # 5 ok


if __name__ == "__main__":
    unittest.main()
