"""Tests for nfl_warehouse (v1 core schema). Guards:
  * SchemaParity — the column-name SPECs match the SQLAlchemy Table definitions.
  * DdlParity — the hand-run sql/schema.sql DDL defines the SAME columns as _META
    (prod DDL is hand-applied, so this catches code↔DDL drift).
  * CreateAll — the tables build on SQLite (the test-only create_all path).
"""
import os
import re
import unittest
from unittest.mock import patch

import pandas as pd
from sqlalchemy import select

import db_store
import nfl_warehouse as nw


class SchemaParityTests(unittest.TestCase):
    def test_team_cols(self):
        self.assertEqual({c.name for c in nw.nfl_team.columns}, set(nw._TEAM_COLS))

    def test_game_cols(self):
        self.assertEqual({c.name for c in nw.nfl_game.columns}, set(nw._GAME_COLS))

    def test_player_cols(self):
        self.assertEqual({c.name for c in nw.nfl_player.columns}, set(nw._PLAYER_COLS))

    def test_player_week_cols(self):
        self.assertEqual({c.name for c in nw.nfl_player_week.columns},
                         set(nw._PLAYER_WEEK_COLS))

    def test_player_week_natural_key(self):
        uqs = [c for c in nw.nfl_player_week.constraints
               if c.__class__.__name__ == "UniqueConstraint"]
        self.assertTrue(any({col.name for col in u.columns} == {"player_id", "season", "week"}
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
        self.assertEqual(self._ddl_cols("nfl_team"), set(nw._TEAM_COLS))

    def test_game_ddl(self):
        self.assertEqual(self._ddl_cols("nfl_game"), set(nw._GAME_COLS))

    def test_player_ddl(self):
        self.assertEqual(self._ddl_cols("nfl_player"), set(nw._PLAYER_COLS))

    def test_player_week_ddl(self):
        self.assertEqual(self._ddl_cols("nfl_player_week"), set(nw._PLAYER_WEEK_COLS))


class CreateAllTests(unittest.TestCase):
    def test_sqlite_create_all(self):
        db_store.configure_engine("sqlite://")
        try:
            nw.create_all()
            from sqlalchemy import inspect
            names = set(inspect(db_store.get_engine()).get_table_names())
            for t in ("nfl_team", "nfl_game", "nfl_player", "nfl_player_week"):
                self.assertIn(t, names)
        finally:
            db_store.configure_engine(None)


class IngestTests(unittest.TestCase):
    """Phase 2 ingest against a SQLite warehouse (fetchers mocked — no network)."""

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

    def test_seed_teams(self):
        n_ins, _ = nw.seed_teams()
        rows = self._rows(nw.nfl_team)
        self.assertEqual(n_ins, 32)
        self.assertEqual(len(rows), 32)
        kc = next(r for r in rows if r["team_abbr"] == "KC")
        self.assertEqual(kc["name"], "Kansas City Chiefs")
        self.assertEqual(kc["conference"], "AFC")
        # idempotent: a second pass inserts nothing.
        self.assertEqual(nw.seed_teams(), (0, 0))

    def test_ingest_games(self):
        games = [
            {"game_id": "2026_01_NYG_DAL", "season": "2026", "week": "1",
             "game_type": "REG", "gameday": "2026-09-07", "gametime": "16:25",
             "home_team": "DAL", "away_team": "NYG", "home_score": 24, "away_score": 20,
             "location": "Home", "result": 4, "total": 44, "espn": "401"},
            {"game_id": "2026_02_DAL_PHI", "season": "2026", "week": "2",
             "game_type": "REG", "gameday": "2026-09-14", "gametime": "20:20",
             "home_team": "PHI", "away_team": "DAL", "home_score": None,
             "away_score": None, "location": "Home", "result": None, "total": None,
             "espn": "402"},
        ]
        with patch("nfl_schedule.load_games", return_value=games):
            self.assertEqual(nw.ingest_games(["2026"]), (2, 0))
            # idempotent re-ingest
            self.assertEqual(nw.ingest_games(["2026"]), (0, 0))
        rows = {r["game_id"]: r for r in self._rows(nw.nfl_game)}
        self.assertEqual(rows["2026_01_NYG_DAL"]["home_score"], 24.0)
        self.assertEqual(rows["2026_01_NYG_DAL"]["week"], 1)
        self.assertIsNone(rows["2026_02_DAL_PHI"]["home_score"])   # unplayed

    def _pw_df(self):
        return pd.DataFrame([
            {"player_id": "00-1", "player_display_name": "Dak Prescott", "season": 2026,
             "week": 1, "team": "DAL", "opponent_team": "NYG", "position": "QB",
             "position_group": "QB", "season_type": "REG", "attempts": 30.0,
             "completions": 22.0, "passing_yards": 290.0, "passing_tds": 2.0,
             "carries": 2.0, "rushing_yards": 5.0, "rushing_tds": 0.0, "receptions": 0.0,
             "targets": 0.0, "receiving_yards": 0.0, "receiving_tds": 0.0,
             "special_teams_tds": 0.0},
            {"player_id": "00-2", "player_display_name": "CeeDee Lamb", "season": 2026,
             "week": 1, "team": "DAL", "opponent_team": "NYG", "position": "WR",
             "position_group": "WR", "season_type": "REG", "receptions": 8.0,
             "targets": 11.0, "receiving_yards": 120.0, "receiving_tds": 1.0,
             "carries": 0.0, "rushing_yards": 0.0, "passing_yards": 0.0},
        ])

    def _snaps_df(self):
        import nfl_props_scan as scan
        return pd.DataFrame([
            {"player_norm": scan._norm("Dak Prescott"), "week": 1,
             "offense_snaps": 65.0, "offense_pct": 1.0, "defense_snaps": 0.0,
             "defense_pct": 0.0, "st_snaps": 0.0, "st_pct": 0.0},
        ])

    def test_ingest_player_weeks(self):
        snaps = self._snaps_df()
        with patch("pandas.read_parquet", return_value=self._pw_df()), \
             patch("nfl_opportunity_serving._load_snaps", return_value=snaps):
            self.assertEqual(nw.ingest_player_weeks(2026), (2, 0))
            # idempotent
            self.assertEqual(nw.ingest_player_weeks(2026), (0, 0))
        pw = {r["player_id"]: r for r in self._rows(nw.nfl_player_week)}
        self.assertEqual(pw["00-1"]["passing_yards"], 290.0)
        self.assertEqual(pw["00-1"]["played"], 1)           # snap row present (True)
        self.assertEqual(pw["00-1"]["offense_snaps"], 65.0)
        self.assertEqual(pw["00-2"]["receiving_yards"], 120.0)
        self.assertEqual(pw["00-2"]["played"], 0)           # posted week, no snap row → DNP
        # player dim bootstrapped
        players = {r["player_id"]: r for r in self._rows(nw.nfl_player)}
        self.assertEqual(players["00-2"]["position"], "WR")
        self.assertEqual(players["00-2"]["name_norm"], "ceedee lamb")

    def test_player_week_frame(self):
        import nfl_props_scan as scan
        with patch("pandas.read_parquet", return_value=self._pw_df()), \
             patch("nfl_opportunity_serving._load_snaps", return_value=self._snaps_df()):
            nw.ingest_player_weeks(2026)
        nw._FRAME_CACHE.clear()
        df = nw.player_week_frame(2026)
        self.assertIsNotNone(df)
        self.assertIn("player_norm", df.columns)
        self.assertIn("passing_yards", df.columns)
        row = df[(df["player_norm"] == scan._norm("Dak Prescott"))
                 & (df["week"] == 1)].iloc[0]
        self.assertEqual(float(row["passing_yards"]), 290.0)
        self.assertIsNone(nw.player_week_frame(2099))     # empty season -> None

    def test_player_weeks_week_filter(self):
        df = self._pw_df()
        df.loc[len(df)] = {**{c: 0.0 for c in df.columns}, "player_id": "00-3",
                           "player_display_name": "Week2 Guy", "season": 2026, "week": 2,
                           "team": "DAL", "position": "RB", "position_group": "RB",
                           "season_type": "REG"}
        with patch("pandas.read_parquet", return_value=df), \
             patch("nfl_opportunity_serving._load_snaps", return_value=None):
            nw.ingest_player_weeks(2026, weeks=[2])          # only week 2
        pw = self._rows(nw.nfl_player_week)
        self.assertEqual({r["week"] for r in pw}, {2})
        self.assertEqual({r["player_id"] for r in pw}, {"00-3"})


class ResolveNflActualWarehouseTests(unittest.TestCase):
    """recalibration._resolve_nfl_actual grades from the warehouse frame FIRST, falling
    back to the live nflverse feed for weeks the warehouse hasn't ingested."""

    def _frame(self, pnorm, week, **stats):
        row = {"player_norm": pnorm, "week": week}
        row.update(stats)
        return pd.DataFrame([row])

    def _resolve(self, prop, name, wh, live, sw=(2026, 3)):
        import recalibration as rc
        with patch("recalibration._nfl_player_week_frame", return_value=wh), \
             patch("nfl_opportunity_serving._load", return_value=live), \
             patch("nfl_schedule.season_week_for_date", return_value=sw):
            return rc._resolve_nfl_actual(prop, name, "2026-09-21")

    def test_warehouse_used_when_present(self):
        wh = self._frame("dak prescott", 3, passing_yards=312.0)
        self.assertEqual(self._resolve("player_pass_yds", "Dak Prescott", wh, None), 312.0)

    def test_falls_back_to_live_when_warehouse_none(self):
        live = self._frame("dak prescott", 3, passing_yards=280.0)
        self.assertEqual(self._resolve("player_pass_yds", "Dak Prescott", None, live), 280.0)

    def test_warehouse_missing_week_falls_to_live(self):
        wh = self._frame("dak prescott", 2, passing_yards=999.0)      # wrong week only
        live = self._frame("dak prescott", 3, passing_yards=280.0)
        self.assertEqual(self._resolve("player_pass_yds", "Dak Prescott", wh, live), 280.0)

    def test_anytime_td_sums_components(self):
        wh = self._frame("ceedee lamb", 3, rushing_tds=0.0, receiving_tds=2.0,
                         special_teams_tds=1.0)
        self.assertEqual(self._resolve("player_anytime_td", "CeeDee Lamb", wh, None), 3.0)


if __name__ == "__main__":
    unittest.main()
