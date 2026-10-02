"""NBA Step-1 calibration join (grade the real-line odds corpus off nba_warehouse, not ESPN):
  * nba_warehouse.calib_gamelogs_bulk — bulk per-player gamelogs for the join.
  * book_line_calibration._nba_grade_book_lines / join_book_lines_to_actuals dispatch.
SQLite warehouse; no network.
"""
import unittest

import db_store
import book_line_calibration as bkc
import nba_warehouse as nw


def _line(player, prop, date, point=20.5):
    return {"player": player, "player_mlb_id": None, "prop_key": prop, "game_date": date,
            "point": point, "price": -110, "direction": "over", "implied_prob": 0.524}


class CalibJoinTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        nw.create_all()
        with db_store.get_engine().begin() as c:
            c.execute(nw.nba_player.insert().values(
                athlete_id="20", name_norm="shooter guy", full_name="Shooter Guy"))
            # 12 played games (2024-01-01, 03, ... 23) + 1 DNP (2024-01-25)
            for k in range(1, 13):
                c.execute(nw.nba_player_game.insert().values(
                    athlete_id="20", game_id=f"g{k}", season=2024,
                    game_date=f"2024-01-{2 * k - 1:02d}", home_away="home",
                    opponent_team_id="99", played=True, minutes=30.0,
                    points=float(20 + k), rebounds=5.0, assists=4.0,
                    three_point_field_goals_attempted=8.0, field_goals_attempted=18.0,
                    free_throws_attempted=6.0))
            c.execute(nw.nba_player_game.insert().values(
                athlete_id="20", game_id="gdnp", season=2024, game_date="2024-01-25",
                home_away="home", opponent_team_id="99", played=False, minutes=None,
                points=0.0, rebounds=0.0, assists=0.0))

    def tearDown(self):
        db_store.configure_engine(None)

    def test_bulk_reader_shape(self):
        bulk = nw.calib_gamelogs_bulk([2024])
        self.assertIn("shooter guy", bulk)
        games = bulk["shooter guy"]
        self.assertEqual(games[0]["game_date"], "2024-01-25")      # most-recent first (DNP)
        g = next(g for g in games if g["game_date"] == "2024-01-23")
        self.assertEqual(g["points"], 32.0)
        self.assertEqual(g["three_point_field_goals_attempted"], 8.0)  # attempt stat present
        self.assertTrue(g["is_home"])
        self.assertEqual(g["MIN"], 30.0)

    def test_single_prop_grade(self):
        enr = bkc.join_book_lines_to_actuals(
            [_line("Shooter Guy", "player_points", "2024-01-23")], "basketball", "nba")
        self.assertEqual(len(enr), 1)
        o = enr[0]
        self.assertEqual(o["actual"], 32.0)              # points on 2024-01-23
        self.assertEqual(o["stat_label"], "points")
        self.assertGreaterEqual(len(o["prior_games"]), 10)
        # priors carry BOTH the prop stat and the attempt stat (for the opportunity gate)
        self.assertIn("points", o["prior_games"][0])
        self.assertIn("three_point_field_goals_attempted", o["prior_games"][0])

    def test_combo_prop_grade(self):
        enr = bkc.join_book_lines_to_actuals(
            [_line("Shooter Guy", "player_points_rebounds_assists", "2024-01-23")],
            "basketball", "nba")
        self.assertEqual(len(enr), 1)
        o = enr[0]
        self.assertEqual(o["actual"], 41.0)              # 32 + 5 + 4
        self.assertEqual(o["stat_label"], "player_points_rebounds_assists")
        self.assertEqual(o["prior_games"][0]["player_points_rebounds_assists"],
                         o["prior_games"][0]["points"] + o["prior_games"][0]["rebounds"]
                         + o["prior_games"][0]["assists"])

    def test_dnp_test_game_dropped(self):
        enr = bkc.join_book_lines_to_actuals(
            [_line("Shooter Guy", "player_points", "2024-01-25")], "basketball", "nba")
        self.assertEqual(enr, [])                        # DNP → not gradeable

    def test_insufficient_priors_dropped(self):
        enr = bkc.join_book_lines_to_actuals(
            [_line("Shooter Guy", "player_points", "2024-01-03")], "basketball", "nba")
        self.assertEqual(enr, [])                        # only 1 prior < 10

    def test_unmapped_prop_dropped(self):
        enr = bkc.join_book_lines_to_actuals(
            [_line("Shooter Guy", "player_double_double", "2024-01-23")], "basketball", "nba")
        self.assertEqual(enr, [])

    def test_exact_date_only_no_off_by_one(self):
        enr = bkc.join_book_lines_to_actuals(
            [_line("Shooter Guy", "player_points", "2024-01-24")], "basketball", "nba")
        self.assertEqual(enr, [])                        # no game on the 24th; no +/-1 borrow

    def test_warehouse_disabled_returns_none(self):
        db_store.configure_engine(None)                  # warehouse off
        try:
            self.assertIsNone(bkc._nba_grade_book_lines(
                [_line("Shooter Guy", "player_points", "2024-01-23")]))
        finally:
            db_store.configure_engine("sqlite://")       # restore for tearDown


if __name__ == "__main__":
    unittest.main()
