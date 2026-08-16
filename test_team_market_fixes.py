"""Tests for the team-market #1 leakage fix (1a): as-of-date bounding of starter
quality so backtests don't see the pitcher's full-season line. Offline (mocks
StatsAPI).

(1b, a margin/total variance-std floor, was tried and REVERTED: a direct
measurement showed the model's pred_std already averages ~5.9 — at/above the 4.52
empirical margin std — so the floor was a no-op. The overconfidence is a
calibration/regression-to-mean problem in the point estimate, addressed in step 2,
not a variance problem.)
"""

import unittest
from unittest.mock import patch

import mlb_starters as ms


_STAT_PAYLOAD = {"people": [{
    "fullName": "Test Pitcher", "pitchHand": {"code": "R"},
    "stats": [{"splits": [{"stat": {
        "era": "3.00", "battersFaced": 300, "strikeOuts": 80,
        "baseOnBalls": 25, "inningsPitched": "70.0", "gamesStarted": 12}}]}],
}]}


class AsOfPitcherQualityTests(unittest.TestCase):
    def test_asof_uses_bydaterange_and_skips_xera(self):
        cap = {}

        def fake_get(path, params):
            cap["hydrate"] = params.get("hydrate")
            return _STAT_PAYLOAD

        with patch.object(ms, "_get", side_effect=fake_get), \
                patch.object(ms, "_read_cache", return_value=None), \
                patch.object(ms, "_write_cache"), \
                patch.object(ms, "get_pitcher_expected_stats") as xmock:
            q = ms.get_pitcher_quality(543037, 2024, as_of_date="2024-06-15")
        self.assertIn("byDateRange", cap["hydrate"])
        self.assertIn("endDate=2024-06-14", cap["hydrate"])  # as_of minus 1 day
        xmock.assert_not_called()                             # xERA skipped (leaky)
        self.assertIsNone(q["xera"])
        self.assertEqual(q["run_suppression_basis"], "era")

    def test_live_mode_unchanged_uses_season_and_xera(self):
        cap = {}

        def fake_get(path, params):
            cap["hydrate"] = params.get("hydrate")
            return _STAT_PAYLOAD

        with patch.object(ms, "_get", side_effect=fake_get), \
                patch.object(ms, "_read_cache", return_value=None), \
                patch.object(ms, "_write_cache"), \
                patch.object(ms, "get_pitcher_expected_stats",
                             return_value={}) as xmock:
            ms.get_pitcher_quality(543037, 2024)
        self.assertIn("type=season", cap["hydrate"])
        xmock.assert_called()                                 # xERA fetched live

    def test_asof_cache_key_differs_from_season(self):
        # Distinct cache namespaces so as-of and live results never cross-pollute.
        captured_keys = []

        def fake_read(key, *a, **k):
            captured_keys.append(key)
            return None

        with patch.object(ms, "_get", side_effect=lambda p, q: _STAT_PAYLOAD), \
                patch.object(ms, "_read_cache", side_effect=fake_read), \
                patch.object(ms, "_write_cache"), \
                patch.object(ms, "get_pitcher_expected_stats", return_value={}):
            ms.get_pitcher_quality(543037, 2024)
            ms.get_pitcher_quality(543037, 2024, as_of_date="2024-06-15")
        self.assertEqual(captured_keys[0], "pitcher_543037_2024")
        self.assertIn("asof_2024-06-14", captured_keys[1])


if __name__ == "__main__":
    unittest.main()
