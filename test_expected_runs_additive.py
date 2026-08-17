"""Tests for mlb_starters.expected_runs_additive (the xERA-lite additive projector)."""
import unittest

import mlb_starters as ms


class ExpectedRunsAdditiveTests(unittest.TestCase):
    def test_basic_innings_split(self):
        # 6 IP starter (2/3 of the game) + bullpen for the other 1/3.
        v = ms.expected_runs_additive(4.0, 4.5, 6.0)
        self.assertAlmostEqual(v, 4.0 * (6 / 9) + 4.5 * (3 / 9))   # 4.1667

    def test_offense_and_run_env_scale(self):
        base = 4.0 * (6 / 9) + 4.5 * (3 / 9)
        v = ms.expected_runs_additive(4.0, 4.5, 6.0, offense_factor=1.1,
                                      run_env=1.05)
        self.assertAlmostEqual(v, base * 1.1 * 1.05)

    def test_exp_ip_clamped_to_full_game(self):
        # A complete game (>=9 exp_ip) is all starter, no bullpen.
        v = ms.expected_runs_additive(3.0, 9.0, 12.0)
        self.assertAlmostEqual(v, 3.0)

    def test_zero_ip_is_all_bullpen(self):
        v = ms.expected_runs_additive(3.0, 5.0, 0.0)
        self.assertAlmostEqual(v, 5.0)

    def test_clamps_range(self):
        self.assertEqual(ms.expected_runs_additive(50.0, 50.0, 6.0), 12.0)   # cap
        self.assertEqual(ms.expected_runs_additive(0.1, 0.1, 6.0), 0.5)      # floor

    def test_bad_input_returns_none(self):
        self.assertIsNone(ms.expected_runs_additive(None, 4.0, 6.0))
        self.assertIsNone(ms.expected_runs_additive(4.0, -1.0, 6.0))
        self.assertIsNone(ms.expected_runs_additive(4.0, 4.0, 6.0, offense_factor=0))
        self.assertIsNone(ms.expected_runs_additive("x", 4.0, 6.0))

    def test_better_starter_lowers_opponent_runs(self):
        # A lower starter_rate9 (better pitcher) => fewer expected runs.
        good = ms.expected_runs_additive(3.0, 4.5, 6.0)
        bad = ms.expected_runs_additive(6.0, 4.5, 6.0)
        self.assertLess(good, bad)


if __name__ == "__main__":
    unittest.main()
