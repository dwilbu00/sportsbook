"""Tests for recalibration._statcast_maintenance (cron-free statcast freshness)."""
import datetime as dt
import unittest
from unittest.mock import patch

import recalibration


class StatcastMaintenanceTests(unittest.TestCase):
    def setUp(self):
        recalibration._last_statcast_derived_build = 0.0   # reset the daily gate
        # Keep the KV-store watermark hermetic: no real read/write in these tests.
        self._wm_read = patch.object(
            recalibration, "_read_statcast_watermark", return_value=None)
        self._wm_write = patch.object(recalibration, "_write_statcast_watermark")
        self.m_read = self._wm_read.start()
        self.m_write = self._wm_write.start()
        self.addCleanup(self._wm_read.stop)
        self.addCleanup(self._wm_write.stop)

    def test_ensure_every_call_but_derived_build_is_gated(self):
        with patch("savant_history.enabled", return_value=True), \
             patch("savant_history.ensure_days", return_value=(1, 1)) as m_ensure, \
             patch("statcast_asof.build") as m_build:
            recalibration._statcast_maintenance()
            recalibration._statcast_maintenance()
        # Raw ensure_days runs EVERY call (cheap/idempotent).
        self.assertEqual(m_ensure.call_count, 2)
        # The heavier derived snapshot rebuild is daily-gated -> once.
        self.assertEqual(m_build.call_count, 1)
        # Called with a capped recent window (start < end).
        args, kw = m_ensure.call_args
        self.assertEqual(kw.get("cap"), 12)
        self.assertLess(args[0], args[1])              # lo < today

    def test_noop_when_sql_off(self):
        with patch("savant_history.enabled", return_value=False), \
             patch("savant_history.ensure_days") as m_ensure, \
             patch("statcast_asof.build") as m_build:
            recalibration._statcast_maintenance()
        m_ensure.assert_not_called()
        m_build.assert_not_called()

    def test_ensure_failure_never_raises(self):
        # A Savant/network hiccup in ensure_days must not propagate.
        with patch("savant_history.enabled", return_value=True), \
             patch("savant_history.ensure_days", side_effect=RuntimeError("boom")), \
             patch("statcast_asof.build") as m_build:
            recalibration._statcast_maintenance()          # must not raise
        # Derived build still attempted (independent step).
        self.assertEqual(m_build.call_count, 1)
        # A failed ensure must NOT advance the completeness watermark.
        self.m_write.assert_not_called()


class StatcastWatermarkHealTests(unittest.TestCase):
    """The self-healing watermark: widen the lookback over an idle gap; advance the
    watermark only when the whole range came back complete."""

    def setUp(self):
        recalibration._last_statcast_derived_build = 0.0

    def _run(self, watermark, ensure_ret):
        with patch.object(recalibration, "_read_statcast_watermark",
                          return_value=watermark), \
             patch.object(recalibration, "_write_statcast_watermark") as m_write, \
             patch("savant_history.enabled", return_value=True), \
             patch("savant_history.ensure_days",
                   return_value=ensure_ret) as m_ensure, \
             patch("statcast_asof.build"):
            recalibration._statcast_maintenance()
        return m_ensure, m_write

    def test_idle_gap_widens_lookback_to_watermark_plus_one(self):
        today = dt.date.today()
        wm = today - dt.timedelta(days=15)             # idle ~15 days
        m_ensure, _ = self._run(wm, (5, 5))
        args, _kw = m_ensure.call_args
        # lo is pulled back to the day AFTER the watermark, well beyond the 4-day window
        self.assertEqual(args[0], (wm + dt.timedelta(days=1)).isoformat())
        self.assertEqual(args[1], today.isoformat())
        # and that is strictly older than the default trailing window start.
        self.assertLess(args[0], (today - dt.timedelta(days=4)).isoformat())

    def test_fresh_watermark_uses_trailing_window_only(self):
        today = dt.date.today()
        wm = today - dt.timedelta(days=1)              # up to date -> no widening
        m_ensure, _ = self._run(wm, (1, 1))
        args, _kw = m_ensure.call_args
        self.assertEqual(args[0], (today - dt.timedelta(days=4)).isoformat())

    def test_watermark_advances_when_range_complete(self):
        today = dt.date.today()
        _m_ensure, m_write = self._run(today - dt.timedelta(days=20), (10, 10))
        m_write.assert_called_once()
        (written,), _kw = m_write.call_args
        self.assertEqual(written, today)               # advanced to today

    def test_watermark_holds_when_cap_leaves_days_behind(self):
        # A gap bigger than the cap: fetched < missing -> do NOT advance, so the next
        # hourly call retries the still-missing older days.
        _m_ensure, m_write = self._run(dt.date.today() - dt.timedelta(days=40),
                                       (12, 40))
        m_write.assert_not_called()

    def test_no_watermark_behaves_like_trailing_window(self):
        today = dt.date.today()
        m_ensure, m_write = self._run(None, (2, 2))
        args, _kw = m_ensure.call_args
        self.assertEqual(args[0], (today - dt.timedelta(days=4)).isoformat())
        m_write.assert_called_once()                   # complete -> stamps today


class StatcastWatermarkKVTests(unittest.TestCase):
    """The watermark read/write round-trips through the app_settings KV store and is
    fail-open, without clobbering other settings."""

    def test_read_fail_open_returns_none(self):
        with patch.object(recalibration, "_read_ndjson_blob",
                          side_effect=RuntimeError("db down")):
            self.assertIsNone(recalibration._read_statcast_watermark())

    def test_read_parses_stored_date(self):
        rows = [{"setting_key": "kelly_fraction", "setting_value": "0.5"},
                {"setting_key": "statcast_last_ensured",
                 "setting_value": "2026-08-10"}]
        with patch.object(recalibration, "_read_ndjson_blob",
                          return_value=(rows, None)):
            self.assertEqual(recalibration._read_statcast_watermark(),
                             dt.date(2026, 8, 10))

    def test_read_bad_value_returns_none(self):
        rows = [{"setting_key": "statcast_last_ensured", "setting_value": "not-a-date"}]
        with patch.object(recalibration, "_read_ndjson_blob",
                          return_value=(rows, None)):
            self.assertIsNone(recalibration._read_statcast_watermark())

    def test_write_upserts_only_the_watermark_key(self):
        captured = {}

        def fake_mutate(filename, mutator, *a, **k):
            rows = [{"setting_key": "kelly_fraction", "setting_value": "0.5"}]
            changed = mutator(rows)
            captured["rows"] = rows
            captured["changed"] = changed
            return changed

        with patch.object(recalibration, "mutate_ndjson_log",
                          side_effect=fake_mutate):
            recalibration._write_statcast_watermark(dt.date(2026, 8, 17))
        keys = {r["setting_key"]: r for r in captured["rows"]}
        self.assertEqual(keys["kelly_fraction"]["setting_value"], "0.5")   # preserved
        self.assertEqual(keys["statcast_last_ensured"]["setting_value"],
                         "2026-08-17")
        self.assertEqual(captured["changed"], 1)

    def test_write_fail_open_returns_zero(self):
        with patch.object(recalibration, "mutate_ndjson_log",
                          side_effect=RuntimeError("db down")):
            self.assertEqual(
                recalibration._write_statcast_watermark(dt.date(2026, 8, 17)), 0)


if __name__ == "__main__":
    unittest.main()
