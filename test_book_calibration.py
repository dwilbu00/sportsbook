"""Tests for book_calibration — the per-market shrink applied to bonus leg P."""
import os
import tempfile
import unittest
from unittest import mock

import book_calibration as bc


class ApplyTests(unittest.TestCase):
    def test_identity_when_no_map(self):
        self.assertEqual(bc.apply("baseball_mlb", "batter_hits", 0.60, maps={}), 0.60)

    def test_none_passthrough(self):
        self.assertIsNone(bc.apply("baseball_mlb", "x", None, maps={}))

    def test_platt_shifts(self):
        maps = {"pitcher_earned_runs": {"kind": "platt", "a": 1.046, "b": -0.117}}
        p = bc.apply("baseball_mlb", "pitcher_earned_runs", 0.55, maps=maps)
        self.assertLess(p, 0.55)                     # over-bias correction shifts down
        self.assertGreater(p, 0.45)

    def test_unmapped_market_identity(self):
        maps = {"pitcher_earned_runs": {"kind": "platt", "a": 1.0, "b": -0.1}}
        self.assertEqual(bc.apply("baseball_mlb", "batter_hits", 0.60, maps=maps), 0.60)

    def test_isotonic(self):
        maps = {"m": {"kind": "isotonic", "kx": [0.1, 0.5, 0.9],
                      "ky": [0.08, 0.47, 0.85]}}
        p = bc.apply("x", "m", 0.5, maps=maps)
        self.assertAlmostEqual(p, 0.47, places=2)

    def test_bad_entry_is_identity(self):
        maps = {"m": {"kind": "platt"}}              # missing a/b
        self.assertEqual(bc.apply("x", "m", 0.6, maps=maps), 0.6)

    def test_clamped_to_unit(self):
        maps = {"m": {"kind": "platt", "a": 5.0, "b": 5.0}}
        p = bc.apply("x", "m", 0.99, maps=maps)
        self.assertLessEqual(p, 1.0)
        self.assertGreaterEqual(p, 0.0)


class SaveLoadTests(unittest.TestCase):
    def test_save_load_merges_markets(self):
        d = tempfile.mkdtemp()
        with mock.patch.object(bc, "_path",
                               return_value=os.path.join(d, "bc.json")):
            self.assertEqual(bc.load_maps("baseball_mlb"), {})     # absent = {}
            bc.save_map("baseball_mlb", "pitcher_earned_runs",
                        {"kind": "platt", "a": 1.0, "b": -0.1, "n": 4456})
            bc.save_map("baseball_mlb", "batter_home_runs",
                        {"kind": "isotonic", "kx": [0.1, 0.9], "ky": [0.1, 0.85]})
            m = bc.load_maps("baseball_mlb")
        self.assertEqual(set(m), {"pitcher_earned_runs", "batter_home_runs"})
        self.assertEqual(m["pitcher_earned_runs"]["a"], 1.0)


if __name__ == "__main__":
    unittest.main()
