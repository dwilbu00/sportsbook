"""Tests for nba_warehouse (v1 core schema). Guards:
  * SchemaParity — the column-name SPECs match the SQLAlchemy Table definitions.
  * DdlParity — the hand-run sql/schema.sql DDL defines the SAME columns as _META
    (prod DDL is hand-applied, so this catches code<->DDL drift).
  * CreateAll — the tables build on SQLite (the test-only create_all path).
"""
import os
import re
import unittest
from unittest.mock import patch

import pandas as pd
from sqlalchemy import inspect, select

import db_store
import nba_warehouse as nw


class SchemaParityTests(unittest.TestCase):
    def test_team_cols(self):
        self.assertEqual({c.name for c in nw.nba_team.columns}, set(nw._TEAM_COLS))

    def test_game_cols(self):
        self.assertEqual({c.name for c in nw.nba_game.columns}, set(nw._GAME_COLS))

    def test_player_cols(self):
        self.assertEqual({c.name for c in nw.nba_player.columns}, set(nw._PLAYER_COLS))

    def test_player_game_cols(self):
        self.assertEqual({c.name for c in nw.nba_player_game.columns},
                         set(nw._PLAYER_GAME_COLS))

    def test_player_game_natural_key(self):
        uqs = [c for c in nw.nba_player_game.constraints
               if c.__class__.__name__ == "UniqueConstraint"]
        self.assertTrue(any({col.name for col in u.columns} == {"athlete_id", "game_id"}
                            for u in uqs))


class DdlParityTests(unittest.TestCase):
    """The hand-run DDL must define the same columns as the _META Tables."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql", "schema.sql")
        with open(path, encoding="utf-8") as f:
            cls.sql = f.read()

    def _ddl_cols(self, table):
        m = re.search(r"CREATE TABLE dbo\." + re.escape(table) + r"\s*\((.*?)\n\);",
                      self.sql, re.S)
        self.assertIsNotNone(m, f"no CREATE TABLE for {table}")
        body = re.sub(r"--[^\n]*", "", m.group(1))        # drop line comments
        # a column def is `<name> <TYPE...>`; constraints/keys have no leading name+type
        return set(re.findall(r"\b([a-z_][a-z0-9_]*)\s+(?:NVARCHAR|INT|FLOAT|BIT)\b",
                              body, re.I))

    def test_team_ddl(self):
        self.assertEqual(self._ddl_cols("nba_team"), set(nw._TEAM_COLS))

    def test_game_ddl(self):
        self.assertEqual(self._ddl_cols("nba_game"), set(nw._GAME_COLS))

    def test_player_ddl(self):
        self.assertEqual(self._ddl_cols("nba_player"), set(nw._PLAYER_COLS))

    def test_player_game_ddl(self):
        self.assertEqual(self._ddl_cols("nba_player_game"), set(nw._PLAYER_GAME_COLS))


class CreateAllTests(unittest.TestCase):
    def test_sqlite_create_all(self):
        db_store.configure_engine("sqlite://")
        try:
            nw.create_all()
            names = set(inspect(db_store.get_engine()).get_table_names())
            for t in ("nba_team", "nba_game", "nba_player", "nba_player_game"):
                self.assertIn(t, names)
        finally:
            db_store.configure_engine(None)


class CoercerTests(unittest.TestCase):
    def test_numeric_coercers(self):
        self.assertIsNone(nw._f(float("nan")))
        self.assertEqual(nw._f("+5"), 5.0)
        self.assertEqual(nw._f("-3"), -3.0)
        self.assertIsNone(nw._f(""))
        self.assertEqual(nw._i("12.0"), 12)
        self.assertEqual(nw._s("  LAL "), "LAL")
        self.assertIsNone(nw._s("  "))

    def test_bool_coercer(self):
        self.assertIsNone(nw._b(None))
        self.assertIsNone(nw._b(float("nan")))
        self.assertTrue(nw._b(True))
        self.assertFalse(nw._b(0))

    def test_current_season_is_end_year(self):
        self.assertIsInstance(nw._current_season(), int)


class IngestTests(unittest.TestCase):
    """Phase 2 ingest against a SQLite warehouse (nba_source loaders mocked — no network)."""

    def setUp(self):
        db_store.configure_engine("sqlite://")
        nw.create_all()
        nw._FRAME_CACHE.clear()

    def tearDown(self):
        db_store.configure_engine(None)
        nw._FRAME_CACHE.clear()

    def _rows(self, table):
        with db_store.get_engine().connect() as c:
            return [dict(r._mapping) for r in c.execute(select(table))]

    def _schedule_df(self):
        return pd.DataFrame([
            {"game_id": 100, "season": 2024, "season_type": 2,
             "game_date": "2023-10-24", "game_date_time": "2023-10-24T19:30-04:00",
             "home_id": 13, "home_abbreviation": "LAL", "home_score": 110.0,
             "away_id": 14, "away_abbreviation": "DEN", "away_score": 105.0,
             "status_type_completed": True, "neutral_site": False, "venue_id": 3900},
            {"game_id": 200, "season": 2024, "season_type": 2,
             "game_date": "2023-11-05", "game_date_time": "2023-11-05T18:00-05:00",
             "home_id": 15, "home_abbreviation": "BOS", "home_score": None,
             "away_id": 13, "away_abbreviation": "LAL", "away_score": None,
             "status_type_completed": False, "neutral_site": False, "venue_id": 3800},
        ])

    def _player_box_df(self):
        base = {"athlete_short_name": "", "athlete_jersey": "0",
                "athlete_position_abbreviation": "G", "season": 2024, "season_type": 2,
                "field_goals_made": 0.0, "field_goals_attempted": 0.0,
                "three_point_field_goals_made": 0.0, "three_point_field_goals_attempted": 0.0,
                "free_throws_made": 0.0, "free_throws_attempted": 0.0,
                "offensive_rebounds": 0.0, "defensive_rebounds": 0.0, "steals": 0.0,
                "blocks": 0.0, "turnovers": 0.0, "fouls": 0.0}
        return pd.DataFrame([
            {**base, "athlete_id": 1, "game_id": 100,
             "athlete_display_name": "LeBron James", "team_id": 13,
             "team_abbreviation": "LAL", "team_display_name": "Los Angeles Lakers",
             "team_name": "Lakers", "team_location": "Los Angeles",
             "game_date": "2023-10-24", "home_away": "home", "opponent_team_id": 14,
             "team_score": 110.0, "opponent_team_score": 105.0, "minutes": 35.0,
             "points": 28.0, "rebounds": 8.0, "assists": 9.0, "plus_minus": "+5",
             # active=False but 35 min played — the real ESPN quirk; played must key off minutes.
             "starter": True, "did_not_play": False, "active": False},
            {**base, "athlete_id": 2, "game_id": 100,
             "athlete_display_name": "Bench Warmer", "team_id": 13,
             "team_abbreviation": "LAL", "team_display_name": "Los Angeles Lakers",
             "team_name": "Lakers", "team_location": "Los Angeles",
             "game_date": "2023-10-24", "home_away": "home", "opponent_team_id": 14,
             "team_score": 110.0, "opponent_team_score": 105.0, "minutes": 0.0,
             "points": 0.0, "rebounds": 0.0, "assists": 0.0, "plus_minus": None,
             "starter": False, "did_not_play": True, "active": True},
            {**base, "athlete_id": 3, "game_id": 200,
             "athlete_display_name": "Later Guy", "team_id": 15,
             "team_abbreviation": "BOS", "team_display_name": "Boston Celtics",
             "team_name": "Celtics", "team_location": "Boston",
             "game_date": "2023-11-05", "home_away": "home", "opponent_team_id": 13,
             "team_score": 99.0, "opponent_team_score": 101.0, "minutes": 20.0,
             "points": 12.0, "rebounds": 5.0, "assists": 2.0, "plus_minus": "-2",
             "starter": True, "did_not_play": False, "active": True},
        ])

    def test_ingest_games(self):
        with patch("nba_source.load_schedule", return_value=self._schedule_df()) as ls:
            self.assertEqual(nw.ingest_games([2024]), (2, 0))
            ls.assert_called_with(2024, ttl=0)
            self.assertEqual(nw.ingest_games([2024]), (0, 0))      # idempotent
        rows = {r["game_id"]: r for r in self._rows(nw.nba_game)}
        self.assertEqual(rows["100"]["home_score"], 110.0)
        self.assertEqual(rows["100"]["season"], 2024)
        self.assertEqual(rows["100"]["home_abbr"], "LAL")
        self.assertTrue(rows["100"]["completed"])
        self.assertIsNone(rows["200"]["home_score"])              # unplayed

    def test_ingest_player_games(self):
        with patch("nba_source.load_player_box", return_value=self._player_box_df()):
            self.assertEqual(nw.ingest_player_games(2024), (3, 0))
            self.assertEqual(nw.ingest_player_games(2024), (0, 0))  # idempotent
        pg = {r["athlete_id"]: r for r in self._rows(nw.nba_player_game)}
        self.assertEqual(pg["1"]["points"], 28.0)
        self.assertEqual(pg["1"]["plus_minus"], 5.0)              # "+5" -> 5.0
        self.assertEqual(pg["1"]["played"], 1)                    # 35 min (active flag ignored)
        self.assertEqual(pg["2"]["played"], 0)                    # 0 min / DNP -> void signal
        self.assertEqual(pg["1"]["game_date"], "2023-10-24")
        # player dim bootstrapped
        players = {r["athlete_id"]: r for r in self._rows(nw.nba_player)}
        self.assertEqual(players["1"]["name_norm"], "lebron james")
        self.assertEqual(players["1"]["position"], "G")
        # team dim bootstrapped + conf/div enriched
        teams = {r["team_id"]: r for r in self._rows(nw.nba_team)}
        self.assertEqual(teams["13"]["abbreviation"], "LAL")
        self.assertEqual(teams["13"]["location"], "Los Angeles")
        self.assertEqual(teams["13"]["conference"], "West")
        self.assertEqual(teams["13"]["division"], "Pacific")
        self.assertEqual(teams["15"]["conference"], "East")

    def test_player_games_since_filter(self):
        with patch("nba_source.load_player_box", return_value=self._player_box_df()):
            nw.ingest_player_games(2024, since_date="2023-11-01")  # only the Nov game
        pg = self._rows(nw.nba_player_game)
        self.assertEqual({r["athlete_id"] for r in pg}, {"3"})
        self.assertEqual({r["game_date"] for r in pg}, {"2023-11-05"})

    def test_player_game_frame(self):
        with patch("nba_source.load_player_box", return_value=self._player_box_df()):
            nw.ingest_player_games(2024)
        nw._FRAME_CACHE.clear()
        df = nw.player_game_frame(2024)
        self.assertIsNotNone(df)
        self.assertIn("player_norm", df.columns)
        self.assertIn("points", df.columns)
        row = df[(df["player_norm"] == "lebron james")
                 & (df["game_date"] == "2023-10-24")].iloc[0]
        self.assertEqual(float(row["points"]), 28.0)
        self.assertEqual(float(row["assists"]), 9.0)
        self.assertIsNone(nw.player_game_frame(2099))             # empty season -> None


if __name__ == "__main__":
    unittest.main()
