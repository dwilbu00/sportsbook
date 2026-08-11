"""P4 grading slice: resolve_one_prop reads the warehouse facts first (0 network)
when it has the P3-stamped (game_pk, mlb_player_id), and resolve_pending_outcomes
exempts those free warehouse hits from the per-pass network cap. mlb_warehouse /
the live path are monkeypatched so no DB or network is touched."""

import unittest
from unittest.mock import patch

import recalibration


class ResolveOnePropWarehouseFirstTests(unittest.TestCase):
    def _resolve(self, **kw):
        # A long-past commence → clears the WAREHOUSE_GRADE_MIN_AGE_HOURS gate.
        return recalibration.resolve_one_prop(
            "baseball_mlb", "X", "batter_hits", 0.5, "2020-08-09",
            "2020-08-09T23:00:00Z", **kw)

    def test_warehouse_hit_short_circuits_live(self):
        with patch("mlb_warehouse.get_actual_stat", return_value=2.0) as gas, \
             patch("recalibration._resolve_mlb_actual") as live:
            v = self._resolve(game_pk=700, mlb_player_id="1")
        self.assertEqual(v, 2.0)
        gas.assert_called_once()
        live.assert_not_called()               # never hit the network path

    def test_warehouse_zero_is_returned(self):
        with patch("mlb_warehouse.get_actual_stat", return_value=0.0), \
             patch("recalibration._resolve_mlb_actual") as live:
            v = self._resolve(game_pk=700, mlb_player_id="1")
        self.assertEqual(v, 0.0)               # a real 0, not a miss
        live.assert_not_called()

    def test_warehouse_miss_falls_through_to_live(self):
        with patch("mlb_warehouse.get_actual_stat", return_value=None), \
             patch("recalibration._resolve_mlb_actual", return_value=3.0) as live:
            v = self._resolve(game_pk=700, mlb_player_id="1")
        self.assertEqual(v, 3.0)
        live.assert_called_once()

    def test_no_ids_skips_warehouse(self):
        with patch("mlb_warehouse.get_actual_stat") as gas, \
             patch("recalibration._resolve_mlb_actual", return_value=1.0):
            v = self._resolve()                # no game_pk / mlb_player_id
        gas.assert_not_called()
        self.assertEqual(v, 1.0)

    def test_recent_game_skips_warehouse_grades_live(self):
        # A too-recent game (corrections may still land) must NOT use the frozen
        # fact — it grades off the fresh live read. A future commence is < the age
        # gate deterministically.
        with patch("mlb_warehouse.get_actual_stat") as gas, \
             patch("recalibration._resolve_mlb_actual", return_value=1.0):
            v = recalibration.resolve_one_prop(
                "baseball_mlb", "X", "batter_hits", 0.5, "2099-01-01",
                "2099-01-01T00:00:00Z", game_pk=700, mlb_player_id="1")
        gas.assert_not_called()
        self.assertEqual(v, 1.0)


class ResolvePendingCapTests(unittest.TestCase):
    """The per-pass cap counts LIVE fetches only — warehouse hits are unbounded."""

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
             patch("mlb_warehouse.get_actual_stat", return_value=2.0), \
             patch("recalibration.resolve_one_prop") as live, \
             patch("recalibration.mutate_prediction_log"):
            n = recalibration.resolve_pending_outcomes("baseball_mlb",
                                                       max_to_resolve=1)
        self.assertEqual(n, 5)                 # all 5 resolved despite cap=1
        live.assert_not_called()               # no network path used

    def test_network_rows_are_capped(self):
        rows = self._rows(5, with_ids=False)   # no ids → warehouse skipped
        with patch("recalibration._read_log", return_value=rows), \
             patch("recalibration.resolve_one_prop", return_value=2.0) as live, \
             patch("recalibration._is_stale_dnp", return_value=False), \
             patch("recalibration.mutate_prediction_log"):
            n = recalibration.resolve_pending_outcomes("baseball_mlb",
                                                       max_to_resolve=2)
        self.assertEqual(live.call_count, 2)   # live fetches capped at 2
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
