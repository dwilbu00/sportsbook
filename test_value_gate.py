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
from props import _prop_gate_is_value, _DK_SELFDEVIG_EDGE_MULT


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


if __name__ == "__main__":
    unittest.main()
