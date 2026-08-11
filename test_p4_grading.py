"""P4 unified resolution: prediction/wager grading resolves the actual FROM THE
WAREHOUSE via mlb_warehouse.resolve_actual (which refreshes an active game and
reads a settled game frozen); the live per-player path is the fail-open fallback,
and the per-pass cap bounds ONLY that fallback. mlb_warehouse / the live path are
monkeypatched (no DB or network touched)."""

import unittest
from unittest.mock import patch

import recalibration


class ResolveOnePropTests(unittest.TestCase):
    def _resolve(self, **kw):
        return recalibration.resolve_one_prop(
            "baseball_mlb", "X", "batter_hits", 0.5, "2026-08-09",
            "2026-08-09T23:00:00Z", **kw)

    def test_warehouse_hit_short_circuits_live(self):
        with patch("mlb_warehouse.resolve_actual", return_value=2.0) as ra, \
             patch("recalibration._resolve_mlb_actual") as live:
            v = self._resolve(game_pk=700, mlb_player_id="1")
        self.assertEqual(v, 2.0)
        ra.assert_called_once()
        live.assert_not_called()

    def test_warehouse_zero_is_returned(self):
        with patch("mlb_warehouse.resolve_actual", return_value=0.0), \
             patch("recalibration._resolve_mlb_actual") as live:
            v = self._resolve(game_pk=700, mlb_player_id="1")
        self.assertEqual(v, 0.0)               # a real 0, not a miss
        live.assert_not_called()

    def test_warehouse_miss_falls_through_to_live(self):
        with patch("mlb_warehouse.resolve_actual", return_value=None), \
             patch("recalibration._resolve_mlb_actual", return_value=3.0) as live:
            v = self._resolve(game_pk=700, mlb_player_id="1")
        self.assertEqual(v, 3.0)
        live.assert_called_once()

    def test_no_ids_skips_warehouse(self):
        with patch("mlb_warehouse.resolve_actual") as ra, \
             patch("recalibration._resolve_mlb_actual", return_value=1.0):
            v = self._resolve()                # no game_pk / mlb_player_id
        ra.assert_not_called()
        self.assertEqual(v, 1.0)

    def test_use_warehouse_false_skips_warehouse(self):
        # The sweep passes this for its explicit live fallback (already tried WH).
        with patch("mlb_warehouse.resolve_actual") as ra, \
             patch("recalibration._resolve_mlb_actual", return_value=1.0):
            v = self._resolve(game_pk=700, mlb_player_id="1", _use_warehouse=False)
        ra.assert_not_called()
        self.assertEqual(v, 1.0)


class ResolvePendingCapTests(unittest.TestCase):
    """The per-pass cap bounds ONLY the live per-player fallback; warehouse
    resolutions (historical frozen reads + cache-bounded active refreshes) are
    unbounded, so a fully-ingested backlog resolves in one pass."""

    def _rows(self, n, with_ids=True):
        return [{"sport_key": "baseball_mlb", "player": f"P{i}",
                 "prop_key": "batter_hits", "line": 0.5,
                 "game_date": "2020-08-09",
                 "commence_time": "2020-08-09T23:00:00Z",
                 "game_pk": (700 + i) if with_ids else None,
                 "player_mlb_id": str(i) if with_ids else None,
                 "resolved": False} for i in range(n)]

    def test_warehouse_rows_resolve_beyond_cap(self):
        rows = self._rows(5, with_ids=True)
        with patch("recalibration._read_log", return_value=rows), \
             patch("mlb_warehouse.resolve_actual", return_value=2.0), \
             patch("recalibration.resolve_one_prop") as live, \
             patch("recalibration.mutate_prediction_log"):
            n = recalibration.resolve_pending_outcomes("baseball_mlb",
                                                       max_to_resolve=1)
        self.assertEqual(n, 5)                 # all 5 resolved despite cap=1
        live.assert_not_called()               # no live fallback used

    def test_live_fallback_rows_are_capped(self):
        rows = self._rows(5, with_ids=False)   # no ids → warehouse skipped → live
        with patch("recalibration._read_log", return_value=rows), \
             patch("recalibration.resolve_one_prop", return_value=2.0) as live, \
             patch("recalibration._is_stale_dnp", return_value=False), \
             patch("recalibration.mutate_prediction_log"):
            n = recalibration.resolve_pending_outcomes("baseball_mlb",
                                                       max_to_resolve=2)
        self.assertEqual(live.call_count, 2)   # live fallback capped at 2
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
