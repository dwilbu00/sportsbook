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


class PrewarmCacheTests(unittest.TestCase):
    """backtest._prewarm_matchup_features cache integration: hit/miss, None never
    cached, version bump rebuilds, bypass env. compute_fn is the test seam so no
    warehouse/SQL is touched."""

    def setUp(self):
        import backtest
        self.backtest = backtest
        self._orig = fs.CACHE_DIR
        fs.CACHE_DIR = tempfile.mkdtemp()
        os.environ.pop("ODI_NO_FEATURE_CACHE", None)

    def tearDown(self):
        fs.CACHE_DIR = self._orig
        os.environ.pop("ODI_NO_FEATURE_CACHE", None)

    def _sched_pw(self, extra=None):
        games = [{"date": "2025-04-01", "home_team": "NYY", "away_team": "BOS",
                  "game_pk": 1},
                 {"date": "2025-04-02", "home_team": "LAD", "away_team": "SFG",
                  "game_pk": 2}]
        if extra:
            games.append(extra)
        schedules = {1: games}
        pw = [dict(g) for g in games]
        return schedules, pw

    def _compute(self, calls, none_for=()):
        def c(home, away, d10, sport_key):
            calls.append((home, away, d10))
            return None if home in none_for else {"home": home, "edge": 0.1}
        return c

    def test_second_run_hits_cache(self):
        sched, pw = self._sched_pw()
        calls = []
        r1 = self.backtest._prewarm_matchup_features(
            pw, sched, "baseball_mlb", True, compute_fn=self._compute(calls))
        self.assertEqual(len(calls), 2)          # first run builds both
        calls.clear()
        r2 = self.backtest._prewarm_matchup_features(
            pw, sched, "baseball_mlb", True, compute_fn=self._compute(calls))
        self.assertEqual(len(calls), 0)          # second run: all cached
        self.assertEqual(r1, r2)

    def test_none_features_never_cached(self):
        # A game whose features are None (unresolved starter / transient error) must
        # recompute every run — never poison the cache.
        extra = {"date": "2025-04-03", "home_team": "XXX", "away_team": "YYY",
                 "game_pk": 3}
        sched, pw = self._sched_pw(extra)
        calls = []
        self.backtest._prewarm_matchup_features(
            pw, sched, "baseball_mlb", True,
            compute_fn=self._compute(calls, none_for=("XXX",)))
        calls.clear()
        r2 = self.backtest._prewarm_matchup_features(
            pw, sched, "baseball_mlb", True,
            compute_fn=self._compute(calls, none_for=("XXX",)))
        self.assertEqual(calls, [("XXX", "YYY", "2025-04-03")])  # only None recomputed
        self.assertIsNone(r2[("2025-04-03", "XXX", "YYY")])

    def test_version_bump_rebuilds(self):
        sched, pw = self._sched_pw()
        calls = []
        self.backtest._prewarm_matchup_features(
            pw, sched, "baseball_mlb", True, compute_fn=self._compute(calls))
        # A newly-completed game changes the season version -> full rebuild.
        extra = {"date": "2025-04-03", "home_team": "TOR", "away_team": "TBR",
                 "game_pk": 3}
        sched2, pw2 = self._sched_pw(extra)
        calls.clear()
        self.backtest._prewarm_matchup_features(
            pw2, sched2, "baseball_mlb", True, compute_fn=self._compute(calls))
        self.assertEqual(len(calls), 3)          # version bumped -> recompute all

    def test_bypass_env_disables_cache(self):
        sched, pw = self._sched_pw()
        os.environ["ODI_NO_FEATURE_CACHE"] = "1"
        calls = []
        self.backtest._prewarm_matchup_features(
            pw, sched, "baseball_mlb", True, compute_fn=self._compute(calls))
        calls.clear()
        self.backtest._prewarm_matchup_features(
            pw, sched, "baseball_mlb", True, compute_fn=self._compute(calls))
        self.assertEqual(len(calls), 2)          # no cache -> recompute every run


if __name__ == "__main__":
    unittest.main()
