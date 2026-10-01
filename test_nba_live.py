"""NBA Phase 4d live-analysis cutover (player history off ESPN):
  * nba_warehouse.player_history — recent played games, combo sum, namesake + as-of.
  * espn_client._nba_warehouse_history / get_player_stat_history — contract dict + gating.
SQLite warehouse; no network.
"""
import unittest
from unittest.mock import patch

import db_store
import espn_client
import nba_warehouse as nw


def _player(aid, name_norm, team_id="13"):
    return {"athlete_id": aid, "name_norm": name_norm, "full_name": name_norm.title(),
            "team_id": team_id}


def _game(aid, gid, date, played=True, team_id="13", opp="14", home_away="home", **stats):
    row = {"athlete_id": aid, "game_id": gid, "season": 2024, "game_date": date,
           "team_id": team_id, "opponent_team_id": opp, "home_away": home_away,
           "played": played}
    row.update(stats)
    return row


class PlayerHistoryTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        nw.create_all()
        eng = db_store.get_engine()
        with eng.begin() as c:
            for p in (_player("10", "star guard", "13"),
                      _player("11", "star guard", "15")):   # namesake on another team
                c.execute(nw.nba_player.insert().values(**p))
            rows = [
                _game("10", "g1", "2024-01-05", points=10.0, assists=4.0),
                _game("10", "g2", "2024-01-07", points=20.0, assists=6.0),
                _game("10", "g3", "2024-01-09", points=30.0, assists=8.0),
                _game("10", "g4", "2024-01-11", played=False, points=0.0, assists=0.0),  # DNP
                _game("11", "g5", "2024-01-08", team_id="15", points=5.0, assists=1.0),  # namesake
            ]
            for r in rows:
                c.execute(nw.nba_player_game.insert().values(**r))

    def tearDown(self):
        db_store.configure_engine(None)

    def test_recent_played_only_most_recent_first(self):
        h = nw.player_history("Star Guard", "points", n=5, team_ids=["13"])
        self.assertEqual([r["value"] for r in h], [30.0, 20.0, 10.0])   # DNP g4 excluded
        self.assertEqual(h[0]["game_date"], "2024-01-09")
        self.assertEqual(h[0]["athlete_id"], "10")

    def test_namesake_disambiguation_by_team(self):
        h = nw.player_history("Star Guard", "points", n=5, team_ids=["15"])
        self.assertEqual([r["value"] for r in h], [5.0])
        self.assertEqual(h[0]["athlete_id"], "11")

    def test_combo_prop_sums(self):
        h = nw.player_history("Star Guard", ("points", "assists"), n=5, team_ids=["13"])
        self.assertEqual([r["value"] for r in h], [38.0, 26.0, 14.0])

    def test_as_of_date_caps_strictly_before(self):
        h = nw.player_history("Star Guard", "points", n=5, as_of_date="2024-01-08",
                              team_ids=["13"])
        self.assertEqual([r["value"] for r in h], [20.0, 10.0])

    def test_n_limit(self):
        h = nw.player_history("Star Guard", "points", n=2, team_ids=["13"])
        self.assertEqual(len(h), 2)

    def test_unknown_player_empty(self):
        self.assertEqual(nw.player_history("Nobody Here", "points", n=5), [])


class NbaWarehouseHistoryContractTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        nw.create_all()
        with db_store.get_engine().begin() as c:
            c.execute(nw.nba_player.insert().values(**_player("10", "star guard", "13")))
            c.execute(nw.nba_player_game.insert().values(
                **_game("10", "g1", "2024-01-09", points=30.0, assists=8.0)))

    def tearDown(self):
        db_store.configure_engine(None)

    def test_contract_dict_shape(self):
        d = espn_client._nba_warehouse_history("Star Guard", "player_points", 5,
                                               team_ids=["13"])
        self.assertIsNotNone(d)
        self.assertEqual(d["source"], "nba_warehouse")
        self.assertTrue(d["found"])
        self.assertEqual(d["values"], [30.0])
        self.assertEqual(d["stat_label"], "player_points")
        self.assertEqual(d["athlete_id"], "10")
        for k in ("opponents", "home_aways", "game_dates", "plate_appearances", "at_bats"):
            self.assertEqual(len(d[k]), 1)

    def test_get_player_stat_history_routes_to_warehouse(self):
        d = espn_client.get_player_stat_history("basketball", "nba", "Star Guard",
                                                "player_points", 5)
        self.assertEqual(d["source"], "nba_warehouse")
        self.assertEqual(d["values"], [30.0])

    def test_unmapped_prop_falls_through(self):
        self.assertIsNone(
            espn_client._nba_warehouse_history("Star Guard", "player_double_double", 5))

    def test_kill_switch_disables(self):
        with patch("nba_source.live_from_sdv", return_value=False):
            self.assertIsNone(
                espn_client._nba_warehouse_history("Star Guard", "player_points", 5))


if __name__ == "__main__":
    unittest.main()
