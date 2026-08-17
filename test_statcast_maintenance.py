"""Tests for recalibration._statcast_maintenance (cron-free statcast freshness)."""
import unittest
from unittest.mock import patch

import recalibration


class StatcastMaintenanceTests(unittest.TestCase):
    def setUp(self):
        recalibration._last_statcast_derived_build = 0.0   # reset the daily gate

    def test_ensure_every_call_but_derived_build_is_gated(self):
        with patch("savant_history.enabled", return_value=True), \
             patch("savant_history.ensure_days") as m_ensure, \
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


if __name__ == "__main__":
    unittest.main()
