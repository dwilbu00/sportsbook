"""Tests for coherence_offset_store — durable KV round-trip + the incremental,
self-maintaining offset math (baseline + delta fold + watermark advance)."""
import unittest
from unittest import mock

import coherence_offset_store as cos


class StoreKVTests(unittest.TestCase):
    """load_stats/save_stats over a fake app_settings blob (the durable KV)."""

    def setUp(self):
        self.rows = []
        self._p1 = mock.patch(
            "recalibration._read_ndjson_blob",
            side_effect=lambda f, use_cache=False: (list(self.rows), None))
        self._p2 = mock.patch(
            "recalibration.mutate_ndjson_log",
            side_effect=lambda f, fn: fn(self.rows))
        self._p1.start()
        self._p2.start()

    def tearDown(self):
        self._p1.stop()
        self._p2.stop()

    def test_save_then_load_roundtrip(self):
        self.assertIsNone(cos.load_stats("baseball_mlb"))
        cos.save_stats("baseball_mlb", {"sum": -12.3456789, "n": 6671,
                                        "through_date": "2026-09-16", "dispersion": 0.0})
        got = cos.load_stats("baseball_mlb")
        self.assertEqual(got["n"], 6671)
        self.assertEqual(got["through_date"], "2026-09-16")
        self.assertAlmostEqual(got["sum"], -12.3456789, places=6)
        # Fits the app_settings.setting_value String(256) column.
        self.assertLessEqual(len(self.rows[0]["setting_value"]), 256)

    def test_per_sport_keys_coexist(self):
        cos.save_stats("baseball_mlb", {"sum": 1.0, "n": 10,
                                        "through_date": "2026-01-01", "dispersion": 0.0})
        cos.save_stats("americanfootball_nfl", {"sum": 2.0, "n": 20,
                                                "through_date": "2026-02-02",
                                                "dispersion": 0.0})
        self.assertEqual(cos.load_stats("baseball_mlb")["n"], 10)
        self.assertEqual(cos.load_stats("americanfootball_nfl")["n"], 20)
        self.assertEqual(len({r["setting_key"] for r in self.rows}), 2)

    def test_save_noop_when_unchanged(self):
        stats = {"sum": 1.0, "n": 10, "through_date": "2026-01-01", "dispersion": 0.0}
        self.assertEqual(cos.save_stats("baseball_mlb", stats), 1)
        self.assertEqual(cos.save_stats("baseball_mlb", stats), 0)   # identical → no write


class CurrentOffsetTests(unittest.TestCase):
    """current_offset: bootstrap, exact incremental fold, watermark advance, no-op
    when current, and fail-open."""

    def test_bootstrap_when_no_baseline(self):
        with mock.patch.object(cos, "load_stats", return_value=None), \
             mock.patch.object(cos, "seed_stats",
                               return_value=(0.05, 500, "2026-09-16")) as seed:
            res = cos.current_offset("baseball_mlb", today="2026-09-17")
        self.assertEqual(res, (0.05, 500))
        seed.assert_called_once()

    def test_incremental_fold_is_exact(self):
        base = {"sum": 10.0, "n": 100, "through_date": "2026-09-10", "dispersion": 0.0}
        saved = {}
        with mock.patch.object(cos, "load_stats", return_value=base), \
             mock.patch.object(cos, "save_stats",
                               side_effect=lambda sp, st: saved.update(st)), \
             mock.patch("r2_data.load_team_triad_range",
                        return_value=(["t1", "t2"], {})), \
             mock.patch.object(cos.coherence_flags, "_triad_offset_stats",
                               return_value=(2.0, 20)):
            res = cos.current_offset("baseball_mlb", today="2026-09-17")
        self.assertEqual(res, (0.1, 120))                 # (10+2)/(100+20)
        self.assertAlmostEqual(saved["sum"], 12.0)
        self.assertEqual(saved["n"], 120)
        self.assertEqual(saved["through_date"], "2026-09-16")   # cutoff = yesterday

    def test_no_read_when_already_current(self):
        base = {"sum": 10.0, "n": 100, "through_date": "2026-09-16", "dispersion": 0.0}
        with mock.patch.object(cos, "load_stats", return_value=base), \
             mock.patch.object(cos, "save_stats") as save, \
             mock.patch("r2_data.load_team_triad_range") as rd:
            res = cos.current_offset("baseball_mlb", today="2026-09-17")  # cutoff 09-16
        self.assertEqual(res, (0.1, 100))
        rd.assert_not_called()
        save.assert_not_called()

    def test_watermark_advances_even_with_no_new_triads(self):
        base = {"sum": 10.0, "n": 100, "through_date": "2026-09-05", "dispersion": 0.0}
        saved = {}
        with mock.patch.object(cos, "load_stats", return_value=base), \
             mock.patch.object(cos, "save_stats",
                               side_effect=lambda sp, st: saved.update(st)), \
             mock.patch("r2_data.load_team_triad_range", return_value=([], {})), \
             mock.patch.object(cos.coherence_flags, "_triad_offset_stats",
                               return_value=(0.0, 0)):
            res = cos.current_offset("baseball_mlb", today="2026-09-17")
        self.assertEqual(res, (0.1, 100))                 # mean unchanged
        self.assertEqual(saved["through_date"], "2026-09-16")   # but watermark moved

    def test_failure_returns_none(self):
        with mock.patch.object(cos, "load_stats",
                               side_effect=RuntimeError("azure down")):
            self.assertIsNone(cos.current_offset("baseball_mlb", today="2026-09-17"))


if __name__ == "__main__":
    unittest.main()
