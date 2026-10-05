"""Tests for nba_warehouse (stats.nba.com schema). Guards:
  * SchemaParity — the column-name SPECs match the SQLAlchemy Table definitions.
  * DdlParity — the hand-run sql/schema.sql DDL defines the SAME columns as _META.
  * CreateAll — the tables build on SQLite.
  * Ingest — stats.nba.com player box + schedule-join enrichment, canonical stat mapping,
    minutes/DNP handling, conf/div from standings (loaders mocked, no network).
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
        self.assertTrue(any({col.name for col in u.columns} == {"person_id", "game_id"}
                            for u in uqs))


class DdlParityTests(unittest.TestCase):
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
        # BIGINT listed before INT so the alternation consumes it whole (not INT within BIGINT).
        return set(re.findall(r"\b([a-z_][a-z0-9_]*)\s+(?:NVARCHAR|BIGINT|INT|FLOAT|BIT)\b",
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
        self.assertEqual(nw._i("1610612747"), 1610612747)
        self.assertEqual(nw._s("  LAL "), "LAL")
        self.assertEqual(nw._date("2024-10-22 00:00:00"), "2024-10-22")

    def test_current_season_is_end_year(self):
        self.assertIsInstance(nw._current_season(), int)


# ───────────────────────────── stats.nba.com fixtures (source-shaped) ─────────────
_LAL, _GSW = 1610612747, 1610612744


def _schedule():
    return pd.DataFrame([
        {"game_id": "0022400001", "game_date": "2024-10-22", "home_team_id": _LAL,
         "away_team_id": _GSW, "home_pts": 120, "away_pts": 110,
         "home_team_abbreviation": "LAL", "away_team_abbreviation": "GSW",
         "home_team_name": "Lakers", "away_team_name": "Warriors",
         "season_type": "regular-season"},
        {"game_id": "0012400009", "game_date": "2024-10-05", "home_team_id": _LAL,
         "away_team_id": _GSW, "home_pts": 100, "away_pts": 98,
         "home_team_abbreviation": "LAL", "away_team_abbreviation": "GSW",
         "home_team_name": "Lakers", "away_team_name": "Warriors",
         "season_type": "preseason"},     # must be EXCLUDED from the spine
    ])


def _player_box():
    zero = {k: 0 for k in ("points", "rebounds_total", "rebounds_offensive",
            "rebounds_defensive", "assists", "three_pointers_made",
            "three_pointers_attempted", "field_goals_made", "field_goals_attempted",
            "free_throws_made", "free_throws_attempted", "steals", "blocks",
            "turnovers", "fouls_personal", "plus_minus_points")}
    return pd.DataFrame([
        {**zero, "person_id": 2544, "first_name": "LeBron", "family_name": "James",
         "team_id": _LAL, "team_tricode": "LAL", "team_city": "Los Angeles",
         "team_name": "Lakers", "side": "home", "game_id": "0022400001",
         "minutes": "35:56", "comment": "", "player_slug": "lebron-james", "position": "F",
         "points": 28, "rebounds_total": 8, "rebounds_offensive": 1,
         "rebounds_defensive": 7, "assists": 9, "three_pointers_made": 2,
         "three_pointers_attempted": 5, "field_goals_made": 10, "field_goals_attempted": 18,
         "free_throws_made": 6, "free_throws_attempted": 7, "steals": 1, "blocks": 1,
         "turnovers": 3, "fouls_personal": 2, "plus_minus_points": 5},
        {**zero, "person_id": 201939, "first_name": "Stephen", "family_name": "Curry",
         "team_id": _GSW, "team_tricode": "GSW", "team_city": "Golden State",
         "team_name": "Warriors", "side": "away", "game_id": "0022400001",
         "minutes": "", "comment": "DNP - Coach's Decision",   # DNP
         "player_slug": "stephen-curry", "position": "G"},
    ])


def _standings():
    return pd.DataFrame([
        {"team_id": _LAL, "conference": "West", "division": "Pacific"},
        {"team_id": _GSW, "conference": "West", "division": "Pacific"},
    ])


class IngestTests(unittest.TestCase):
    def setUp(self):
        self._mdir = os.environ.get("ODI_MIRROR_DIR")
        import tempfile
        os.environ["ODI_MIRROR_DIR"] = tempfile.mkdtemp()
        db_store.configure_engine("sqlite://")
        nw.create_all()
        nw._FRAME_CACHE.clear()

    def tearDown(self):
        db_store.configure_engine(None)
        nw._FRAME_CACHE.clear()
        if self._mdir is None:
            os.environ.pop("ODI_MIRROR_DIR", None)
        else:
            os.environ["ODI_MIRROR_DIR"] = self._mdir

    def _rows(self, table):
        with db_store.get_engine().connect() as c:
            return [dict(r._mapping) for r in c.execute(select(table))]

    def test_ingest_games_filters_preseason(self):
        with patch("nba_stats_source.load_schedule", return_value=_schedule()):
            self.assertEqual(nw.ingest_games([2025]), (1, 0))   # preseason excluded
            self.assertEqual(nw.ingest_games([2025]), (0, 0))   # idempotent
        rows = {r["game_id"]: r for r in self._rows(nw.nba_game)}
        self.assertIn("0022400001", rows)
        self.assertNotIn("0012400009", rows)                    # preseason not in spine
        self.assertEqual(rows["0022400001"]["home_team_id"], _LAL)
        self.assertEqual(rows["0022400001"]["season_type"], "regular-season")
        self.assertTrue(rows["0022400001"]["completed"])

    def test_ingest_player_games_enriches_and_maps(self):
        with patch("nba_stats_source.load_player_box", return_value=_player_box()), \
             patch("nba_stats_source.load_schedule", return_value=_schedule()), \
             patch("nba_stats_source.load_standings", return_value=_standings()):
            self.assertEqual(nw.ingest_player_games(2025), (2, 0))
            self.assertEqual(nw.ingest_player_games(2025), (0, 0))   # idempotent
        pg = {r["person_id"]: r for r in self._rows(nw.nba_player_game)}
        lj = pg[2544]
        self.assertEqual(lj["points"], 28.0)
        self.assertEqual(lj["rebounds"], 8.0)                   # rebounds_total -> rebounds
        self.assertEqual(lj["three_point_field_goals_made"], 2.0)  # three_pointers_made ->
        self.assertEqual(lj["offensive_rebounds"], 1.0)
        self.assertEqual(lj["minutes"], 35.93)                  # "35:56" -> float
        self.assertEqual(lj["played"], 1)
        self.assertEqual(lj["game_date"], "2024-10-22")         # from schedule join
        self.assertEqual(lj["opponent_team_id"], _GSW)          # derived from side+schedule
        self.assertEqual(lj["home_away"], "home")
        self.assertEqual(lj["team_score"], 120.0)
        self.assertEqual(lj["opponent_team_score"], 110.0)
        sc = pg[201939]
        self.assertEqual(sc["played"], 0)                       # minutes "" -> DNP
        self.assertEqual(sc["dnp_reason"], "DNP - Coach's Decision")
        self.assertEqual(sc["opponent_team_id"], _LAL)
        # player dim
        players = {r["person_id"]: r for r in self._rows(nw.nba_player)}
        self.assertEqual(players[2544]["name_norm"], "lebron james")
        self.assertEqual(players[2544]["position"], "F")
        # team dim + real conf/div from standings
        teams = {r["team_id"]: r for r in self._rows(nw.nba_team)}
        self.assertEqual(teams[_LAL]["tricode"], "LAL")
        self.assertEqual(teams[_LAL]["city"], "Los Angeles")
        self.assertEqual(teams[_LAL]["conference"], "West")
        self.assertEqual(teams[_LAL]["division"], "Pacific")

    def test_player_game_frame(self):
        with patch("nba_stats_source.load_player_box", return_value=_player_box()), \
             patch("nba_stats_source.load_schedule", return_value=_schedule()), \
             patch("nba_stats_source.load_standings", return_value=_standings()):
            nw.ingest_player_games(2025)
        nw._FRAME_CACHE.clear()
        df = nw.player_game_frame(2025)
        self.assertIsNotNone(df)
        self.assertIn("player_norm", df.columns)
        row = df[df["player_norm"] == "lebron james"].iloc[0]
        self.assertEqual(float(row["points"]), 28.0)
        self.assertEqual(float(row["rebounds"]), 8.0)
        self.assertIsNone(nw.player_game_frame(2099))


if __name__ == "__main__":
    unittest.main()
