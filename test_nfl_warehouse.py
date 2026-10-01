"""Tests for nfl_warehouse (v1 core schema). Guards:
  * SchemaParity — the column-name SPECs match the SQLAlchemy Table definitions.
  * DdlParity — the hand-run sql/schema.sql DDL defines the SAME columns as _META
    (prod DDL is hand-applied, so this catches code↔DDL drift).
  * CreateAll — the tables build on SQLite (the test-only create_all path).
"""
import os
import re
import unittest

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


if __name__ == "__main__":
    unittest.main()
