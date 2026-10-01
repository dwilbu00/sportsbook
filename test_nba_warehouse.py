"""Tests for nba_warehouse (v1 core schema). Guards:
  * SchemaParity — the column-name SPECs match the SQLAlchemy Table definitions.
  * DdlParity — the hand-run sql/schema.sql DDL defines the SAME columns as _META
    (prod DDL is hand-applied, so this catches code<->DDL drift).
  * CreateAll — the tables build on SQLite (the test-only create_all path).
"""
import os
import re
import unittest

from sqlalchemy import inspect

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


if __name__ == "__main__":
    unittest.main()
