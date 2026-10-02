"""NBA backtest-engine warehouse integration: the player-props sweep (synthetic refit /
recency sweep / eval_min_streak all route through run_player_props_backtest -> fetch_player_data)
grades NBA off nba_warehouse (all 16 props incl. combos) instead of ESPN's 3. SQLite; no network.
"""
import os
import unittest

import db_store
import backtest
import nba_warehouse as nw


class SweepModeTests(unittest.TestCase):
    def tearDown(self):
        db_store.configure_engine(None)
        os.environ.pop("ODI_NBA_SWEEP_SDV", None)

    def test_off_when_sql_disabled(self):
        db_store.configure_engine(None)
        self.assertFalse(backtest._nba_sweep_mode("basketball"))

    def test_on_when_enabled(self):
        db_store.configure_engine("sqlite://")
        self.assertTrue(backtest._nba_sweep_mode("basketball"))

    def test_kill_switch(self):
        db_store.configure_engine("sqlite://")
        os.environ["ODI_NBA_SWEEP_SDV"] = "0"
        self.assertFalse(backtest._nba_sweep_mode("basketball"))

    def test_non_basketball_off(self):
        db_store.configure_engine("sqlite://")
        self.assertFalse(backtest._nba_sweep_mode("baseball"))
        self.assertFalse(backtest._nba_sweep_mode("football"))


class SweepStatLabelTests(unittest.TestCase):
    def _gl(self):
        return [{"points": 30.0, "rebounds": 8.0, "assists": 9.0, "played": True},
                {"points": 0.0, "rebounds": 0.0, "assists": 0.0, "played": False}]  # DNP

    def test_single_prop_uses_column(self):
        self.assertEqual(backtest._nba_sweep_stat_label("player_points", self._gl()), "points")

    def test_combo_injects_sum_and_dnp_none(self):
        gl = self._gl()
        label = backtest._nba_sweep_stat_label("player_points_rebounds_assists", gl)
        self.assertEqual(label, "player_points_rebounds_assists")
        self.assertEqual(gl[0][label], 47.0)                 # 30+8+9
        self.assertIsNone(gl[1][label])                      # DNP -> None (skipped, no phantom 0)

    def test_unmapped_prop_none(self):
        self.assertIsNone(backtest._nba_sweep_stat_label("player_double_double", self._gl()))


class FetchPlayerDataNbaTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        nw.create_all()
        with db_store.get_engine().begin() as c:
            c.execute(nw.nba_player.insert().values(
                athlete_id="20", name_norm="shooter guy", full_name="Shooter Guy"))
            for k in range(1, 4):
                c.execute(nw.nba_player_game.insert().values(
                    athlete_id="20", game_id=f"g{k}", season=2024,
                    game_date=f"2024-01-0{k}", team_id="13", home_away="home",
                    opponent_team_id="99", played=True, minutes=30.0,
                    points=float(20 + k), rebounds=5.0, assists=4.0,
                    three_point_field_goals_attempted=8.0))

    def tearDown(self):
        db_store.configure_engine(None)

    def test_fetch_routes_to_warehouse(self):
        data = backtest.fetch_player_data("basketball", "nba", ["Shooter Guy"],
                                          season_year=2024, quiet=True)
        self.assertIn("Shooter Guy", data)
        gl = data["Shooter Guy"]
        self.assertEqual(len(gl), 3)
        self.assertEqual(gl[0]["game_date"], "2024-01-03")    # newest first
        self.assertEqual(gl[0]["points"], 23.0)
        self.assertEqual(gl[0]["team_id"], "13")              # team_id present (layoff detect)
        self.assertIn("three_point_field_goals_attempted", gl[0])
        # the sweep's actual-extraction then resolves all props off these rows
        self.assertEqual(backtest._nba_sweep_stat_label("player_threes", gl),
                         "three_point_field_goals_made")

    def test_fetch_missing_player_skipped(self):
        data = backtest.fetch_player_data("basketball", "nba", ["Nobody Here"],
                                          season_year=2024, quiet=True)
        self.assertEqual(data, {})


if __name__ == "__main__":
    unittest.main()
