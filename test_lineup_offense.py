"""Tests for the #3 lineup-offense edge (team predictions from today's 9 batters).

Warehouse as-of OPS (leakage-safe cumulative) + the home-minus-away lineup edge.
Offline (mocks SQL). The margin shift is inert by default (DEFAULT_LINEUP_WEIGHT=0);
these test the feature computation, not the (backtest-swept) weight.
"""

import unittest
from unittest.mock import patch

import mlb_starters as ms
import mlb_warehouse as mw


class AsofBatterOpsTests(unittest.TestCase):
    # (official_date, ab, h, bb, hbp, sf, tb)
    _IDX = {"b1": [
        ("2024-04-01", 4, 2, 1, 0, 0, 4),   # 4 AB, 2 H, 1 BB, 4 TB
        ("2024-04-08", 3, 1, 0, 0, 0, 1),   # 3 AB, 1 H, 1 TB
        ("2024-04-15", 4, 4, 0, 0, 0, 8),   # ON the as-of date → excluded
    ]}

    def test_cumulative_and_leakage(self):
        with patch.object(mw, "enabled", return_value=True), \
                patch.object(mw, "_batter_game_index", return_value=self._IDX):
            st = mw.asof_batter_ops("b1", "2024-04-15")   # strictly before
        # AB=7 H=3 BB=1 HBP=0 SF=0 TB=5 → OBP=(3+1)/(7+1)=.5, SLG=5/7, OPS=.5+5/7
        self.assertAlmostEqual(st["pa"], 8.0)
        self.assertAlmostEqual(st["obp"], 4 / 8)
        self.assertAlmostEqual(st["slg"], 5 / 7)
        self.assertAlmostEqual(st["ops"], 4 / 8 + 5 / 7)

    def test_none_when_no_prior_or_unknown(self):
        with patch.object(mw, "enabled", return_value=True), \
                patch.object(mw, "_batter_game_index", return_value=self._IDX):
            self.assertIsNone(mw.asof_batter_ops("b1", "2024-04-01"))  # first game
            self.assertIsNone(mw.asof_batter_ops("x", "2024-07-01"))   # unknown

    def test_none_when_sql_off(self):
        with patch.object(mw, "enabled", return_value=False):
            self.assertIsNone(mw.asof_batter_ops("b1", "2024-07-01"))


class LineupOffenseEdgeTests(unittest.TestCase):
    def _run(self, home_ops, away_ops, game_pk=123):
        def _tid(name, idx):
            return {"id": "H"} if name == "Home" else {"id": "A"}

        def _ops(aid, d):
            return {"ops": home_ops if aid.startswith("h") else away_ops}
        with patch.object(ms, "_match_team_id", side_effect=_tid), \
                patch.object(mw, "_game_pk_index",
                             return_value={("2024-07-01", "H", "A"): game_pk}), \
                patch.object(mw, "_game_lineup_index",
                             return_value={game_pk: {"H": ["h1", "h2"],
                                                     "A": ["a1", "a2"]}}), \
                patch.object(mw, "asof_batter_ops", side_effect=_ops):
            return ms.lineup_offense_edge("Home", "Away", "2024-07-01", {}, 2024)

    def test_home_stronger_positive_edge(self):
        self.assertGreater(self._run(0.900, 0.700), 0)

    def test_away_stronger_negative_edge(self):
        self.assertLess(self._run(0.700, 0.900), 0)

    def test_equal_lineups_zero_edge(self):
        self.assertAlmostEqual(self._run(0.750, 0.750), 0.0, places=6)

    def test_none_when_game_not_resolved(self):
        with patch.object(ms, "_match_team_id",
                          side_effect=lambda n, i: {"id": "H"}), \
                patch.object(mw, "_game_pk_index", return_value={}):
            self.assertIsNone(
                ms.lineup_offense_edge("Home", "Away", "2024-07-01", {}, 2024))


if __name__ == "__main__":
    unittest.main()
