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


class SharpnessTests(unittest.TestCase):
    def test_brier_paired_t_detects_sharper_book(self):
        # Pinnacle probs track the outcome (with per-leg confidence variation so the
        # Brier gap has real variance); DK is flat 0.5 -> Pinnacle sharper.
        rows = []
        for i in range(200):
            over = 1.0 if i % 2 == 0 else 0.0
            conf = 0.85 if i % 4 < 2 else 0.70    # two confidence tiers -> variance
            rows.append({"dk_fair": 0.5, "over": over,
                         "pin_fair": conf if over else (1 - conf)})
        md, t = bt._paired_brier_t(rows)
        self.assertGreater(md, 0)        # DK_brier - Pin_brier > 0
        self.assertGreater(t, 2)         # significantly: Pinnacle sharper
        db, _ = bt._brier_logloss(rows, "dk_fair")
        pb, _ = bt._brier_logloss(rows, "pin_fair")
        self.assertGreater(db, pb)       # DK worse (higher) Brier

    def test_sharpness_rows_same_line_only(self):
        # A same-line leg (Pinnacle posts DK's exact point) yields a row; a projected
        # one (different point) does not.
        same = _leg(prop="pitcher_earned_runs", dk_point=2.5,
                    pin_over=-110, pin_under=-110)  # pinnacle offer at 2.5
        idx = {"pitcher": {("683002", 700): {"ER": 3.0}}}   # over 2.5
        rows = bt.sharpness_rows({"2024": [same]}, idx)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["over"], 1.0)


class SideSplitTests(unittest.TestCase):
    def test_fade_detected_when_unders_win_overs_lose(self):
        # Build legs where the UNDER always wins (actual below the line) across 2
        # seasons -> _fade_verdict flags a DK-over bias.
        legs = {}
        idx_all = {}
        legs_list_by_season = {}
        for si, season in enumerate(("2024", "2025")):
            legs_s = []
            for i in range(150):
                gpk = 1000 * (si + 1) + i
                legs_s.append(_leg(prop="batter_total_bases", dk_point=1.5,
                                   dk_over=-110, dk_under=-110,
                                   pin_over=-110, pin_under=-110,
                                   mlbid=str(gpk), gpk=gpk, season=season))
                idx_all[(str(gpk), gpk)] = {"TB": 0.0}    # actual 0 -> UNDER 1.5 wins
            legs[season] = legs_s
        idx = {"batter": idx_all}
        rows = bt.side_split_rows(legs, idx, haircut=0.0)
        pr = [r for r in rows if r["prop_key"] == "batter_total_bases"]
        v = bt._fade_verdict(pr, min_n=100)
        self.assertIsNotNone(v)
        self.assertIn("UNDER positive", v)


class TeamSharpnessTests(unittest.TestCase):
    def test_team_sharpness_rows_grade_home_win(self):
        import r2_data
        legs = {"2024": [r2_data.TeamMLLeg(
            event_id="G1", game_date="2024-06-26", commence_time="x", snapshot_id=2,
            game_pk=555, home="NYM", away="NYY",
            dk_home=+105, dk_away=-125, pin_home=+108, pin_away=-120)]}
        finals = {555: 1.0}   # home (NYM) won
        rows = bt.team_sharpness_rows(legs, finals, label="moneyline_team")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["over"], 1.0)
        self.assertEqual(rows[0]["prop_key"], "moneyline_team")
        self.assertTrue(0 < rows[0]["dk_fair"] < 1 and 0 < rows[0]["pin_fair"] < 1)

    def test_missing_final_dropped(self):
        import r2_data
        legs = {"2024": [r2_data.TeamMLLeg(
            event_id="G1", game_date="x", commence_time="x", snapshot_id=2,
            game_pk=999, home="NYM", away="NYY",
            dk_home=+105, dk_away=-125, pin_home=+108, pin_away=-120)]}
        self.assertEqual(bt.team_sharpness_rows(legs, {555: 1.0}), [])   # 999 not in finals


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
