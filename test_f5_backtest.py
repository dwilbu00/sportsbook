"""Unit tests for f5_backtest — F5 moneyline grading + edge selection."""
import unittest

import f5_backtest as f5
import r2_data


def _leg(gpk=700, dk_home=-120, dk_away=-110, pin_home=+102, pin_away=-114,
         fd_home=+105, fd_away=-125, season="2024"):
    return r2_data.F5MLLeg(
        event_id="F1", game_date=f"{season}-06-26", commence_time="x", snapshot_id=2,
        game_pk=gpk, home="Mets", away="Yankees", dk_home=dk_home, dk_away=dk_away,
        pin_home=pin_home, pin_away=pin_away, fd_home=fd_home, fd_away=fd_away)


class GradeF5MLTests(unittest.TestCase):
    def test_win_loss_push(self):
        self.assertEqual(f5.grade_f5_ml("home", 3, 1), "win")
        self.assertEqual(f5.grade_f5_ml("home", 1, 3), "loss")
        self.assertEqual(f5.grade_f5_ml("away", 3, 1), "loss")
        self.assertEqual(f5.grade_f5_ml("away", 1, 3), "win")
        self.assertEqual(f5.grade_f5_ml("home", 2, 2), "push")
        self.assertEqual(f5.grade_f5_ml("away", 2, 2), "push")


class GradeEdgeTests(unittest.TestCase):
    def test_bets_both_books_and_grades(self):
        # Low floor so DK + FD both fire; home led F5 4-1 (home win).
        legs = {"2024": [_leg(gpk=700)]}
        scores = {700: (4.0, 1.0)}
        rows, cov = f5.grade_edge(legs, scores, haircut=0.0, ev_floor=-10.0)
        books = {r["book"] for r in rows}
        self.assertEqual(books, {"dk", "fd"})
        # every home bet won, every away bet lost
        for r in rows:
            self.assertEqual(r["result"], "win" if r["side"] == "home" else "loss")

    def test_tie_is_push(self):
        legs = {"2024": [_leg(gpk=701)]}
        rows, cov = f5.grade_edge(legs, {701: (2.0, 2.0)}, haircut=0.0, ev_floor=-10.0)
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["result"], "push")
            self.assertEqual(r["profit"], 0.0)

    def test_missing_score_dropped(self):
        legs = {"2024": [_leg(gpk=999)]}
        rows, cov = f5.grade_edge(legs, {700: (4.0, 1.0)}, ev_floor=-10.0)
        self.assertEqual(rows, [])
        self.assertGreaterEqual(cov["dropped_no_score"], 1)

    def test_high_floor_selects_nothing(self):
        legs = {"2024": [_leg(gpk=700)]}
        rows, cov = f5.grade_edge(legs, {700: (4.0, 1.0)}, ev_floor=0.95)
        self.assertEqual(rows, [])


class SharpnessRowsTests(unittest.TestCase):
    def test_excludes_ties_and_pairs_books(self):
        legs = {"2024": [_leg(gpk=700), _leg(gpk=701)]}
        scores = {700: (4.0, 1.0), 701: (2.0, 2.0)}   # 701 is a tie -> excluded
        rows = f5.sharpness_rows(legs, scores)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["over"], 1.0)         # home won game 700
        self.assertIn("dk_fair", rows[0])
        self.assertIn("pin_fair", rows[0])


if __name__ == "__main__":
    unittest.main()
