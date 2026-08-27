"""Unit tests for r2_backtest — grading, haircut EV/profit, hardened gate, BH.
Pure: synthetic PropLegs + a synthetic outcome index, no warehouse."""
import unittest

import r2_backtest as bt
import r2_data
from odds_client import american_to_decimal


def _leg(prop="batter_hits", dk_point=0.5, dk_over=+100, dk_under=-130,
         pin_over=-200, pin_under=+170, mlbid="683002", gpk=700, season="2024"):
    return r2_data.PropLeg(
        event_id="E1", game_date=f"{season}-06-26", commence_time=f"{season}-06-26T23:10:00Z",
        captured_at=f"{season}-06-26T23:05:00Z", snapshot_id=2, game_pk=gpk,
        player="Gunnar Henderson", player_mlb_id=mlbid, prop_key=prop,
        dk_point=dk_point, dk_over_price=dk_over, dk_under_price=dk_under,
        pinnacle_offers=[{"point": dk_point, "over_price": pin_over, "under_price": pin_under}])


class HaircutTests(unittest.TestCase):
    def test_haircut_shrinks_ev_and_profit(self):
        raw = bt.ev_haircut(0.60, +100, 0.0)
        hc = bt.ev_haircut(0.60, +100, 0.05)
        self.assertGreater(raw, hc)
        self.assertAlmostEqual(bt.profit_haircut(+100, "win", 0.0), 1.0, places=9)
        self.assertAlmostEqual(bt.profit_haircut(+100, "win", 0.05),
                               american_to_decimal(100) * 0.95 - 1, places=9)
        self.assertEqual(bt.profit_haircut(-110, "loss", 0.05), -1.0)
        self.assertEqual(bt.profit_haircut(-110, "push", 0.05), 0.0)


class GradeLegsTests(unittest.TestCase):
    def test_plus_ev_over_graded_win(self):
        legs = {"2024": [_leg()]}
        idx = {"batter": {("683002", 700): {"H": 1.0}}}   # got a hit -> OVER 0.5 wins
        rows, cov = bt.grade_legs(legs, idx, haircut=0.02, ev_floor=0.03)
        overs = [r for r in rows if r["side"] == "OVER"]
        self.assertEqual(len(overs), 1)
        self.assertEqual(overs[0]["result"], "win")
        self.assertGreater(overs[0]["profit"], 0)
        self.assertEqual(overs[0]["arm"], "same_line")
        self.assertEqual(cov["graded"], len(rows))

    def test_missing_actual_dropped_not_zero(self):
        legs = {"2024": [_leg()]}
        rows, cov = bt.grade_legs(legs, {"batter": {}}, haircut=0.02, ev_floor=0.03)
        self.assertEqual(rows, [])
        self.assertGreaterEqual(cov["dropped_no_actual"], 1)

    def test_null_id_resolved_then_graded(self):
        legs = {"2024": [_leg(mlbid=None)]}
        idx = {"batter": {("683002", 700): {"H": 1.0}}}
        rows, cov = bt.grade_legs(legs, idx, haircut=0.02, ev_floor=0.03,
                                  resolve_fn=lambda name, season, prop: "683002")
        self.assertTrue(any(r["side"] == "OVER" for r in rows))

    def test_null_id_unresolved_dropped(self):
        legs = {"2024": [_leg(mlbid=None)]}
        idx = {"batter": {("683002", 700): {"H": 1.0}}}
        rows, cov = bt.grade_legs(legs, idx, haircut=0.02, ev_floor=0.03,
                                  resolve_fn=lambda *a: None)
        self.assertEqual(rows, [])
        self.assertGreaterEqual(cov["dropped_null_id"], 1)

    def test_below_floor_not_selected(self):
        legs = {"2024": [_leg()]}
        idx = {"batter": {("683002", 700): {"H": 1.0}}}
        rows, cov = bt.grade_legs(legs, idx, haircut=0.02, ev_floor=0.95)
        self.assertEqual(rows, [])
        self.assertEqual(cov.get("selected", 0), 0)


class GateTests(unittest.TestCase):
    def _season_rows(self, season, wins, losses):
        return ([{"season": season, "result": "win", "profit": 0.9} for _ in range(wins)] +
                [{"season": season, "result": "loss", "profit": -1.0} for _ in range(losses)])

    def test_single_season_fails(self):
        rows = self._season_rows("2024", 100, 50)
        passed, reason, _per, _p = bt.hardened_gate(rows, min_n=100, min_seasons=2)
        self.assertFalse(passed)
        self.assertIn("season", reason)

    def test_two_positive_seasons_pass(self):
        rows = self._season_rows("2024", 100, 50) + self._season_rows("2025", 100, 50)
        passed, reason, _per, pooled = bt.hardened_gate(rows, min_n=100, min_seasons=2, min_t=2.0)
        self.assertTrue(passed, reason)
        self.assertGreater(pooled.t_stat, 2.0)

    def test_one_negative_season_fails(self):
        rows = self._season_rows("2024", 100, 50) + self._season_rows("2025", 40, 110)
        passed, reason, _per, _p = bt.hardened_gate(rows, min_n=100, min_seasons=2)
        self.assertFalse(passed)
        self.assertIn("not positive", reason)


class DistBucketTests(unittest.TestCase):
    def test_line_gap_buckets(self):
        self.assertEqual(bt.dist_bucket(0.0), "<=0.5")
        self.assertEqual(bt.dist_bucket(0.5), "<=0.5")
        self.assertEqual(bt.dist_bucket(1.0), "(0.5,1.0]")
        self.assertEqual(bt.dist_bucket(1.5), "(1.0,1.5]")
        self.assertEqual(bt.dist_bucket(2.0), ">1.5")
        self.assertEqual(bt.dist_bucket(None), "n/a")


class BHTests(unittest.TestCase):
    def test_bh_selects_expected(self):
        pvals = [0.001, 0.5, 0.9, 0.02]
        self.assertEqual(bt.benjamini_hochberg(pvals, alpha=0.05), {0, 3})

    def test_bh_none_pass(self):
        self.assertEqual(bt.benjamini_hochberg([0.9, 0.8, 0.7], alpha=0.05), set())


class ReportSmokeTests(unittest.TestCase):
    def test_build_report_runs(self):
        rows = ([{"season": "2024", "prop_key": "pitcher_earned_runs", "side": "OVER",
                  "arm": "same_line", "result": "win", "profit": 0.5,
                  "ev": 0.1, "ev_bucket": "5%-10%"} for _ in range(120)] +
                [{"season": "2025", "prop_key": "pitcher_earned_runs", "side": "OVER",
                  "arm": "same_line", "result": "loss", "profit": -1.0,
                  "ev": 0.1, "ev_bucket": "5%-10%"} for _ in range(120)])
        verdict = bt.build_report(rows, {"legs": 1, "graded": 240}, min_n=100)
        self.assertIn("primary_passed", verdict)
        self.assertIn("text", verdict)


if __name__ == "__main__":
    unittest.main()
