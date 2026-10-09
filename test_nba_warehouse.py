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

    def test_team_game_cols(self):
        self.assertEqual({c.name for c in nw.nba_team_game.columns}, set(nw._TEAM_GAME_COLS))

    def test_team_game_natural_key(self):
        uqs = [c for c in nw.nba_team_game.constraints
               if c.__class__.__name__ == "UniqueConstraint"]
        self.assertTrue(any({col.name for col in u.columns} == {"game_id", "team_id"}
                            for u in uqs))

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

    def test_team_game_ddl(self):
        self.assertEqual(self._ddl_cols("nba_team_game"), set(nw._TEAM_GAME_COLS))

    def test_additive_team_game_script_matches(self):
        # the standalone owner-run script must define the same columns as schema.sql/_META
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql",
                            "add_nba_team_game.sql")
        with open(path, encoding="utf-8") as f:
            sql = f.read()
        m = re.search(r"CREATE TABLE dbo\.nba_team_game\s*\((.*?)\n\);", sql, re.S)
        self.assertIsNotNone(m)
        body = re.sub(r"--[^\n]*", "", m.group(1))
        cols = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s+(?:NVARCHAR|BIGINT|INT|FLOAT|BIT)\b",
                              body, re.I))
        self.assertEqual(cols, set(nw._TEAM_GAME_COLS))
        self.assertNotIn("DROP TABLE", sql.upper())               # additive only


class CreateAllTests(unittest.TestCase):
    def test_sqlite_create_all(self):
        db_store.configure_engine("sqlite://")
        try:
            nw.create_all()
            names = set(inspect(db_store.get_engine()).get_table_names())
            for t in ("nba_team", "nba_game", "nba_player", "nba_player_game",
                      "nba_team_game"):
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


def _team_box_row(team_id, side, gid="0022400001", minutes="240:00", **kw):
    base = {"team_id": team_id, "side": side, "game_id": gid, "minutes": minutes,
            "season_type_id": "2", "points": 0, "field_goals_made": 40,
            "field_goals_attempted": 88, "three_pointers_made": 12,
            "three_pointers_attempted": 35, "free_throws_made": 15,
            "free_throws_attempted": 20, "rebounds_offensive": 10,
            "rebounds_defensive": 33, "rebounds_total": 43, "assists": 25, "steals": 7,
            "blocks": 5, "turnovers": 13, "fouls_personal": 19, "plus_minus_points": 0}
    base.update(kw)
    return base


def _team_box():
    return pd.DataFrame([
        _team_box_row(_LAL, "home", points=120, plus_minus_points=10),
        _team_box_row(_GSW, "away", points=110, plus_minus_points=-10,
                      rebounds_offensive=12, rebounds_defensive=30, turnovers=15),
        # preseason game: in the team box, excluded via the schedule's season_type
        _team_box_row(_LAL, "home", gid="0012400009", points=100, season_type_id="1"),
        _team_box_row(_GSW, "away", gid="0012400009", points=98, season_type_id="1"),
        # orphan single row (no opponent row) -> skipped
        _team_box_row(_LAL, "home", gid="0022400099", points=101),
    ])


class PossessionTests(unittest.TestCase):
    def test_bbref_formula_both_team_average(self):
        tm = {"field_goals_attempted": 88, "field_goals_made": 40, "free_throws_attempted": 20,
              "offensive_rebounds": 10, "defensive_rebounds": 33, "turnovers": 13}
        op = {"field_goals_attempted": 88, "field_goals_made": 40, "free_throws_attempted": 20,
              "offensive_rebounds": 12, "defensive_rebounds": 30, "turnovers": 15}
        side_t = 88 + 8 - 1.07 * (10 / 40) * 48 + 13         # ORB% = 10/(10+30)
        side_o = 88 + 8 - 1.07 * (12 / 45) * 48 + 15         # ORB% = 12/(12+33)
        self.assertAlmostEqual(nw.estimate_possessions(tm, op), 0.5 * (side_t + side_o))
        self.assertEqual(nw.estimate_possessions(tm, op), nw.estimate_possessions(op, tm))

    def test_missing_input_none(self):
        self.assertIsNone(nw.estimate_possessions({"field_goals_attempted": 80}, {}))


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


    def test_ingest_team_games_pairs_and_derives(self):
        with patch("nba_stats_source.load_team_box", return_value=_team_box()), \
             patch("nba_stats_source.load_schedule", return_value=_schedule()):
            self.assertEqual(nw.ingest_team_games(2025), (2, 0))   # 1 game x 2 teams
            self.assertEqual(nw.ingest_team_games(2025), (0, 0))   # idempotent
        rows = {r["team_id"]: r for r in self._rows(nw.nba_team_game)}
        self.assertEqual(set(rows), {_LAL, _GSW})               # preseason + orphan excluded
        lal, gsw = rows[_LAL], rows[_GSW]
        self.assertEqual(lal["game_id"], "0022400001")
        self.assertEqual(lal["game_date"], "2024-10-22")        # from the schedule join
        self.assertEqual(lal["season_type"], "regular-season")
        self.assertEqual(lal["opponent_team_id"], _GSW)
        self.assertEqual(lal["home_away"], "home")
        self.assertEqual(lal["points"], 120.0)
        self.assertEqual(lal["opponent_points"], 110.0)
        self.assertEqual(gsw["opponent_points"], 120.0)
        self.assertEqual(lal["three_point_field_goals_made"], 12.0)   # canonical mapping
        self.assertEqual(gsw["offensive_rebounds"], 12.0)
        self.assertEqual(lal["game_minutes"], 48.0)             # "240:00" / 5
        self.assertIsNotNone(lal["possessions"])
        self.assertEqual(lal["possessions"], gsw["possessions"])  # same game, same value

    def test_team_games_reader_mirror_roundtrip(self):
        with patch("nba_stats_source.load_team_box", return_value=_team_box()), \
             patch("nba_stats_source.load_schedule", return_value=_schedule()):
            nw.ingest_team_games(2025)
        azure = nw.team_games([2025])                           # no mirror -> Azure (SQLite)
        self.assertEqual(len(azure), 2)
        pd.DataFrame(azure, columns=list(nw._TEAM_MIRROR_COLS)).to_parquet(
            nw._team_mirror_file(2025), index=False)
        db_store.configure_engine(None)                         # mirror only now
        mirrored = nw.team_games([2025])
        self.assertEqual(len(mirrored), 2)
        self.assertEqual({r["team_id"] for r in mirrored}, {_LAL, _GSW})
        self.assertEqual(mirrored[0]["possessions"], azure[0]["possessions"])

    def test_validate_team_games_vs_official(self):
        with patch("nba_stats_source.load_team_box", return_value=_team_box()), \
             patch("nba_stats_source.load_schedule", return_value=_schedule()):
            nw.ingest_team_games(2025)
        poss = nw.team_games([2025])[0]["possessions"]
        pace = 48.0 * poss / 48.0
        official = pd.DataFrame([
            {"team_id": str(_LAL), "measure_type": "advanced", "season_type": "regular-season",
             "per_mode": "totals", "pace": pace + 1.0, "off_rating": 100 * 120 / poss + 1.0,
             "def_rating": 100 * 110 / poss + 1.0},
            {"team_id": str(_GSW), "measure_type": "advanced", "season_type": "regular-season",
             "per_mode": "totals", "pace": pace + 1.5, "off_rating": 100 * 110 / poss + 1.0,
             "def_rating": 100 * 120 / poss + 1.0}])
        with patch("nba_stats_source.load_team_season_stats", return_value=official):
            v = nw.validate_team_games(2025)
        self.assertEqual(v["n_teams"], 2)
        self.assertAlmostEqual(v["off_rating"]["mean_err"], -1.0, places=6)
        self.assertAlmostEqual(v["off_rating"]["corr"], 1.0, places=6)
        self.assertAlmostEqual(v["pace"]["mean_err"], -1.25, places=6)
        self.assertIsNone(v["pace"]["corr"])        # one game: no spread -> undefined, not NaN


if __name__ == "__main__":
    unittest.main()
