"""Local matchup-feature cache (feature_store) — version marker + round-trip +
staleness. Hermetic: CACHE_DIR redirected to a temp dir per test."""
import os
import tempfile
import unittest

import feature_store as fs


class FeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self._orig = fs.CACHE_DIR
        fs.CACHE_DIR = tempfile.mkdtemp()

    def tearDown(self):
        fs.CACHE_DIR = self._orig

    def test_season_version_count_and_max_date(self):
        games = [{"date": "2025-04-01"}, {"date": "2025-04-03"},
                 {"date": "2025-04-02"}]
        self.assertEqual(fs.season_version(games), "3:2025-04-03")
        # A new completed game bumps the version (count AND max date change).
        games2 = games + [{"date": "2025-04-05"}]
        self.assertEqual(fs.season_version(games2), "4:2025-04-05")
        self.assertNotEqual(fs.season_version(games), fs.season_version(games2))
        self.assertEqual(fs.season_version([]), "0:")

    def test_save_load_round_trip(self):
        feats = {("2025-04-01", "NYY", "BOS"): {"home_sp_id": 1, "edge": 0.12},
                 ("2025-04-01", "LAD", "SFG"): {"home_sp_id": 2, "edge": None}}
        self.assertTrue(fs.save("baseball_mlb", 2025, "2:2025-04-01", feats))
        got = fs.load("baseball_mlb", 2025, "2:2025-04-01")
        self.assertEqual(got, feats)

    def test_load_stale_version_is_miss(self):
        feats = {("2025-04-01", "NYY", "BOS"): {"x": 1}}
        fs.save("baseball_mlb", 2025, "1:2025-04-01", feats)
        # Same season, but the version advanced -> treat as a miss (recompute).
        self.assertIsNone(fs.load("baseball_mlb", 2025, "2:2025-04-02"))
        # Matching version still hits.
        self.assertIsNotNone(fs.load("baseball_mlb", 2025, "1:2025-04-01"))

    def test_load_missing_file_is_none(self):
        self.assertIsNone(fs.load("baseball_mlb", 2024, "anything"))

    def test_numpy_scalars_serialize(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        feats = {("2025-04-01", "NYY", "BOS"): {"f": np.float64(0.371), "k": np.int64(5)}}
        self.assertTrue(fs.save("baseball_mlb", 2025, "v", feats))
        got = fs.load("baseball_mlb", 2025, "v")
        self.assertAlmostEqual(got[("2025-04-01", "NYY", "BOS")]["f"], 0.371)
        self.assertEqual(got[("2025-04-01", "NYY", "BOS")]["k"], 5)

    def test_clear_removes_season(self):
        fs.save("baseball_mlb", 2024, "v", {("d", "h", "a"): {}})
        fs.save("baseball_mlb", 2025, "v", {("d", "h", "a"): {}})
        self.assertEqual(fs.clear("baseball_mlb", 2024), 1)
        self.assertIsNone(fs.load("baseball_mlb", 2024, "v"))
        self.assertIsNotNone(fs.load("baseball_mlb", 2025, "v"))   # untouched

    def test_corrupt_file_is_miss_not_raise(self):
        os.makedirs(fs.CACHE_DIR, exist_ok=True)
        with open(fs._path("baseball_mlb", 2025), "w", encoding="utf-8") as fh:
            fh.write("{ not valid json")
        self.assertIsNone(fs.load("baseball_mlb", 2025, "v"))


if __name__ == "__main__":
    unittest.main()
