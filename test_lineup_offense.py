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
    def _run(self, home, away, game_pk=123):
        # home/away = {aid_prefix: (ops, pa)} lookups
        def _tid(name, idx):
            return {"id": "H"} if name == "Home" else {"id": "A"}

        def _ops(aid, d):
            ops, pa = home if aid.startswith("h") else away
            return {"ops": ops, "pa": pa}
        with patch.object(ms, "_match_team_id", side_effect=_tid), \
                patch.object(mw, "_game_pk_index",
                             return_value={("2024-07-01", "H", "A"): game_pk}), \
                patch.object(mw, "_game_lineup_index",
                             return_value={game_pk: {"H": ["h1", "h2"],
                                                     "A": ["a1", "a2"]}}), \
                patch.object(mw, "asof_batter_ops", side_effect=_ops):
            return ms.lineup_offense_edge("Home", "Away", "2024-07-01", {}, 2024)

    def test_home_stronger_positive_edge(self):
        # ample PA → little shrinkage → edge ≈ home-away OPS diff > 0
        self.assertGreater(self._run((0.900, 400), (0.700, 400)), 0.1)

    def test_away_stronger_negative_edge(self):
        self.assertLess(self._run((0.700, 400), (0.900, 400)), -0.1)

    def test_equal_lineups_zero_edge(self):
        self.assertAlmostEqual(self._run((0.750, 400), (0.750, 400)), 0.0, places=6)

    def test_small_sample_batter_is_shrunk_not_explosive(self):
        # A 2-PA callup batting 4.000 must NOT blow up the edge — PA-shrinkage pulls
        # it toward league (~0.711), so vs a league-average opponent the edge stays
        # small, not ~+3.3.
        edge = self._run((4.000, 2), (0.711, 400))
        self.assertLess(edge, 0.15)

    def test_edge_clamped_to_range(self):
        self.assertLessEqual(self._run((2.000, 400), (0.400, 400)), 0.3)

    def test_slot_pa_weighting_favors_top_of_order(self):
        # Strong leadoff + 8 weak hitters: the slot-PA weighting pulls the side mean
        # ABOVE a flat mean (the top of the order counts more).
        def _tid(n, i):
            return {"id": "H"} if n == "Home" else {"id": "A"}

        def _ops(aid, d):
            if aid == "h1":
                return {"ops": 1.000, "pa": 600}   # star leadoff
            if aid.startswith("h"):
                return {"ops": 0.650, "pa": 600}   # weak rest
            return {"ops": 0.711, "pa": 600}       # league-avg away
        home_ids = [f"h{i}" for i in range(1, 10)]
        with patch.object(ms, "_match_team_id", side_effect=_tid), \
                patch.object(mw, "_game_pk_index",
                             return_value={("2024-07-01", "H", "A"): 1}), \
                patch.object(mw, "_game_lineup_index",
                             return_value={1: {"H": home_ids,
                                               "A": [f"a{i}" for i in range(1, 10)]}}), \
                patch.object(mw, "asof_batter_ops", side_effect=_ops):
            edge = ms.lineup_offense_edge("Home", "Away", "2024-07-01", {}, 2024)
        # flat-mean home ≈ (1.000 + 8*0.650)/9 = 0.689 → flat edge ≈ -0.022;
        # slot-weighted lifts the star (slot 1, highest PA) → home mean higher →
        # edge strictly greater than the flat-mean edge.
        flat_home = (1.000 + 8 * 0.650) / 9
        # shrink is negligible at pa=600, so compare against flat_home - league
        self.assertGreater(edge, flat_home - 0.711)

    def test_none_when_game_not_resolved(self):
        with patch.object(ms, "_match_team_id",
                          side_effect=lambda n, i: {"id": "H"}), \
                patch.object(mw, "_game_pk_index", return_value={}):
            self.assertIsNone(
                ms.lineup_offense_edge("Home", "Away", "2024-07-01", {}, 2024))


class LineupOffenseFactorsTests(unittest.TestCase):
    """The 1.0-centered per-team factor that feeds the expected-runs projection."""
    def _run(self, home_ops, away_ops):
        def _tid(n, i):
            return {"id": "H"} if n == "Home" else {"id": "A"}

        def _ops(aid, d):
            ops = home_ops if aid.startswith("h") else away_ops
            return {"ops": ops, "pa": 600}
        with patch.object(ms, "_match_team_id", side_effect=_tid), \
                patch.object(mw, "_game_pk_index",
                             return_value={("2024-07-01", "H", "A"): 1}), \
                patch.object(mw, "_game_lineup_index",
                             return_value={1: {"H": ["h1", "h2"],
                                               "A": ["a1", "a2"]}}), \
                patch.object(mw, "asof_batter_ops", side_effect=_ops):
            return ms.lineup_offense_factors("Home", "Away", "2024-07-01", {}, 2024)

    def test_league_average_lineup_is_one(self):
        f = self._run(0.711, 0.711)
        self.assertAlmostEqual(f["home"], 1.0, places=2)
        self.assertAlmostEqual(f["away"], 1.0, places=2)

    def test_strong_above_one_weak_below(self):
        f = self._run(0.900, 0.600)
        self.assertGreater(f["home"], 1.0)
        self.assertLess(f["away"], 1.0)

    def test_none_when_no_game(self):
        with patch.object(ms, "_match_team_id",
                          side_effect=lambda n, i: {"id": "H"}), \
                patch.object(mw, "_game_pk_index", return_value={}):
            self.assertIsNone(
                ms.lineup_offense_factors("Home", "Away", "2024-07-01", {}, 2024))


if __name__ == "__main__":
    unittest.main()
