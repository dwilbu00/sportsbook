"""NBA Step-1 calibration join (grade the real-line odds corpus off nba_warehouse, not ESPN):
  * nba_warehouse.calib_gamelogs_bulk — bulk per-player gamelogs for the join.
  * book_line_calibration._nba_grade_book_lines / join_book_lines_to_actuals dispatch.
SQLite warehouse; no network.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import db_store
import book_line_calibration as bkc
import nba_warehouse as nw


def _line(player, prop, date, point=20.5):
    return {"player": player, "player_mlb_id": None, "prop_key": prop, "game_date": date,
            "point": point, "price": -110, "direction": "over", "implied_prob": 0.524}


class CalibJoinTests(unittest.TestCase):
    def setUp(self):
        self._mdir = tempfile.mkdtemp()                  # empty mirror -> grade off SQLite
        os.environ["ODI_MIRROR_DIR"] = self._mdir
        db_store.configure_engine("sqlite://")
        nw.create_all()
        with db_store.get_engine().begin() as c:
            c.execute(nw.nba_player.insert().values(
                person_id=20, name_norm="shooter guy", full_name="Shooter Guy"))
            # 12 played games (2024-01-01, 03, ... 23) + 1 DNP (2024-01-25)
            for k in range(1, 13):
                c.execute(nw.nba_player_game.insert().values(
                    person_id=20, game_id=f"g{k}", season=2024,
                    game_date=f"2024-01-{2 * k - 1:02d}", team_id=13, home_away="home",
                    opponent_team_id=99, played=True, minutes=30.0,
                    points=float(20 + k), rebounds=5.0, assists=4.0,
                    three_point_field_goals_attempted=8.0, field_goals_attempted=18.0,
                    free_throws_attempted=6.0))
            c.execute(nw.nba_player_game.insert().values(
                person_id=20, game_id="gdnp", season=2024, game_date="2024-01-25",
                team_id=13, home_away="home", opponent_team_id=99, played=False, minutes=None,
                points=0.0, rebounds=0.0, assists=0.0))

    def tearDown(self):
        db_store.configure_engine(None)
        os.environ.pop("ODI_MIRROR_DIR", None)
        shutil.rmtree(self._mdir, ignore_errors=True)

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

    def test_no_sql_no_mirror_grades_nothing_and_never_espn(self):
        # ESPN retired 2026-10: with SQL off and no mirror there's simply no data — an empty
        # result, and the join never falls back to the ESPN gamelog.
        db_store.configure_engine(None)                  # warehouse off
        try:
            with patch("book_line_calibration.cached_gamelog") as espn:
                out = bkc.join_book_lines_to_actuals(
                    [_line("Shooter Guy", "player_points", "2024-01-23")], "basketball", "nba")
            self.assertEqual(out, [])
            espn.assert_not_called()
        finally:
            db_store.configure_engine("sqlite://")       # restore for tearDown


class MirrorTests(unittest.TestCase):
    """export_mirror -> parquet; calib_gamelogs_bulk reads it MIRROR-FIRST at 0 DTU (even with
    Azure off), NaN-safe (a null stat stays None through parquet, never becomes 0)."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        os.environ["ODI_MIRROR_DIR"] = self._dir
        db_store.configure_engine("sqlite://")
        nw.create_all()
        with db_store.get_engine().begin() as c:
            c.execute(nw.nba_player.insert().values(
                person_id=30, name_norm="mirror guy", full_name="Mirror Guy"))
            c.execute(nw.nba_player_game.insert().values(
                person_id=30, game_id="m1", season=2024, game_date="2024-02-01",
                team_id=13, home_away="home", opponent_team_id=99, played=True,
                minutes=28.0, points=18.0, rebounds=5.0, assists=3.0, plus_minus=None))

    def tearDown(self):
        db_store.configure_engine(None)
        os.environ.pop("ODI_MIRROR_DIR", None)
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_export_then_read_mirror_first_without_azure(self):
        written = nw.export_mirror([2024], verbose=False)
        self.assertEqual(written[2024], 1)
        self.assertTrue(os.path.exists(nw._mirror_file(2024)))
        db_store.configure_engine(None)                  # Azure OFF — must serve from parquet
        bulk = nw.calib_gamelogs_bulk([2024])
        self.assertIn("mirror guy", bulk)
        g = bulk["mirror guy"][0]
        self.assertEqual(g["points"], 18.0)
        self.assertEqual(g["team_id"], 13)
        self.assertTrue(g["is_home"])
        self.assertTrue(g["played"])
        self.assertIsNone(g["plus_minus"])               # None survives parquet (not NaN/0)


if __name__ == "__main__":
    unittest.main()
