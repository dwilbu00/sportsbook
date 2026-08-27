"""Unit tests for coherence_backtest — grading mechanics + bucketing.
The coherence math itself is covered in test_coherence; here we check the run-line
grade (home_covered logic) and side win/loss mapping."""
import unittest

import coherence_backtest as cb
import r2_data


def _triad(gpk=500, rl_point=-1.5, ml_home=-150, ml_away=+130,
           rl_home=+120, rl_away=-140, total_over=-110, total_under=-110,
           total_line=8.5, season="2024"):
    return r2_data.TeamTriad(
        event_id="G1", game_date=f"{season}-06-26", commence_time="x",
        snapshot_id=2, game_pk=gpk, home="NYM", away="NYY",
        ml_home=ml_home, ml_away=ml_away, rl_home_point=rl_point,
        rl_home=rl_home, rl_away=rl_away, total_line=total_line,
        total_over=total_over, total_under=total_under)


class IncohBucketTests(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(cb._incoh_bucket(0.01), "<=0.02")
        self.assertEqual(cb._incoh_bucket(-0.04), "(0.02,0.05]")
        self.assertEqual(cb._incoh_bucket(0.08), "(0.05,0.10]")
        self.assertEqual(cb._incoh_bucket(-0.2), ">0.10")


class GradeCoherenceTests(unittest.TestCase):
    def test_home_cover_grading(self):
        # Home wins 5-2 -> home covers -1.5 (5-1.5=3.5 > 2). ev_floor very low so
        # both run-line sides are graded; home should WIN, away should LOSE.
        triads = {"2024": [_triad(gpk=500, rl_point=-1.5)]}
        scores = {500: (5.0, 2.0)}
        rows, cov = cb.grade_coherence(triads, scores, haircut=0.0, ev_floor=-10.0)
        by = {r["side"]: r for r in rows}
        self.assertEqual(by["home"]["result"], "win")
        self.assertEqual(by["away"]["result"], "loss")
        self.assertGreater(by["home"]["profit"], 0)
        self.assertEqual(by["away"]["profit"], -1.0)

    def test_home_wins_by_one_does_not_cover(self):
        # Home wins 3-2 -> does NOT cover -1.5 -> home RL loses, away RL wins.
        triads = {"2024": [_triad(gpk=501, rl_point=-1.5)]}
        scores = {501: (3.0, 2.0)}
        rows, cov = cb.grade_coherence(triads, scores, haircut=0.0, ev_floor=-10.0)
        by = {r["side"]: r for r in rows}
        self.assertEqual(by["home"]["result"], "loss")
        self.assertEqual(by["away"]["result"], "win")

    def test_missing_score_dropped(self):
        triads = {"2024": [_triad(gpk=999)]}
        rows, cov = cb.grade_coherence(triads, {500: (5.0, 2.0)}, ev_floor=-10.0)
        self.assertEqual(rows, [])
        self.assertGreaterEqual(cov["dropped_no_score"], 1)

    def test_incoh_recorded(self):
        triads = {"2024": [_triad(gpk=500)]}
        rows, _ = cb.grade_coherence(triads, {500: (5.0, 2.0)}, ev_floor=-10.0)
        self.assertTrue(rows)
        self.assertIn("incoh", rows[0])
        self.assertIn("implied", rows[0])


if __name__ == "__main__":
    unittest.main()
