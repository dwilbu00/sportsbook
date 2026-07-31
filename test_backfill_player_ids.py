"""Tests for backfill_player_ids.py (Phase 4 data step), exercised against
in-memory SQLite so pymssql and the live Azure DB are never touched.

Both the db_store durable tables AND the SFBB map tables live on the same
configure_engine("sqlite://") backend; the map is seeded from the same CSV
fixture test_player_id_map uses (Mike Trout / Jose Ramirez / Ohtani / teams).
"""

import unittest
from unittest.mock import patch

from sqlalchemy import insert, select

import backfill_player_ids as backfill
import db_store
import player_id_map
import recalibration
from test_player_id_map import _PLAYERS_CSV, _TEAMS_CSV, _FakeResp


class _Backend:
    def setUp(self):
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        player_id_map.create_all()
        self._reset_map_state()
        self._load_map()

    def tearDown(self):
        db_store.configure_engine(None)
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        self._reset_map_state()

    @staticmethod
    def _reset_map_state():
        player_id_map._invalidate_index("player")
        player_id_map._invalidate_index("team")
        player_id_map._LAST_FRESH_CHECK = {"player": 0.0, "team": 0.0}
        player_id_map._KEY_LOCKS.clear()

    def _load_map(self):
        def fake_get(url, headers=None, timeout=None, allow_redirects=None):
            return _FakeResp(_PLAYERS_CSV if "PLAYER" in url.upper() else _TEAMS_CSV)
        with patch.object(player_id_map.requests, "get", side_effect=fake_get):
            player_id_map.refresh_players()
            player_id_map.refresh_teams()
        self._reset_map_state()
        # Warm both in-process indexes from SQL now, OUTSIDE any write txn. Under
        # StaticPool the whole suite shares ONE sqlite connection, so a lazy index
        # build fired mid-backfill (e.g. the first _mlb_id call in _backfill_odds,
        # after the snapshot UPDATE) opens a nested SQLAlchemy connection whose
        # close rolls back the outer engine.begin() transaction's un-committed
        # writes. Production pools hand the reader a distinct physical connection,
        # so this is a test-only artifact; pre-warming sidesteps it while still
        # exercising the real SQL round-trip.
        player_id_map.mlb_id_for_name("Mike Trout")
        player_id_map.team_code_for_name("ARI")

    # Direct insert (bypasses db_store.mutate so we can seed the exact pre-backfill
    # column state, including an un-enriched legacy row with a name:<norm> key).
    def _insert_predictions(self, rows):
        with db_store.get_engine().begin() as conn:
            for r in rows:
                params = {name: fn(r.get(name))
                          for name, fn in db_store._PREDICTION_SPEC}
                params["event_key"] = r.get("event_id") or r.get("game_date") or ""
                conn.execute(insert(db_store.prediction_log), params)

    def _read_predictions(self):
        with db_store.get_engine().connect() as conn:
            return [dict(m._mapping)
                    for m in conn.execute(select(db_store.prediction_log)).all()]


class PredictionCollisionMergeTests(_Backend, unittest.TestCase):

    def _seed_collision(self):
        # Two rows for one forecast, split under the OLD raw-name identity: one
        # already enriched (Phase-3 going-forward write, mlb:608070), one legacy
        # accent variant still name-keyed. They collide on mlb:608070 post-backfill.
        self._insert_predictions([
            {"sport_key": "baseball_mlb", "event_id": "e1",
             "prop_key": "batter_hits", "player": "Jose Ramirez", "team": "CLE",
             "line": 0.5, "raw_prob": 0.6, "ts": "t1", "resolved": True,
             "outcome": 1, "actual": 2.0, "player_mlb_id": "608070",
             "player_key": "mlb:608070"},
            {"sport_key": "baseball_mlb", "event_id": "e1",
             "prop_key": "batter_hits", "player": "José Ramírez", "team": "CLE",
             "line": 0.5, "raw_prob": 0.62, "ts": "t2", "resolved": False,
             "player_key": "name:jose ramirez"},
        ])

    def test_two_spellings_collapse_to_one_mlb_keyed_row(self):
        self._seed_collision()
        with db_store.get_engine().begin() as conn:
            _, deleted = backfill._backfill_predictions(conn, dry_run=False)
        rows = self._read_predictions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["player_key"], "mlb:608070")
        self.assertEqual(rows[0]["player_mlb_id"], "608070")
        self.assertEqual(rows[0]["team_code"], "CLE")
        # The resolved outcome survives the merge (never drop a graded result).
        self.assertTrue(rows[0]["resolved"])
        self.assertEqual(rows[0]["outcome"], 1)
        self.assertEqual(deleted, 1)

    def test_idempotent_second_run_merges_nothing(self):
        self._seed_collision()
        with db_store.get_engine().begin() as conn:
            backfill._backfill_predictions(conn, dry_run=False)
        with db_store.get_engine().begin() as conn:
            _, deleted = backfill._backfill_predictions(conn, dry_run=False)
        self.assertEqual(deleted, 0)
        self.assertEqual(len(self._read_predictions()), 1)

    def test_dry_run_writes_nothing(self):
        self._seed_collision()
        with db_store.get_engine().begin() as conn:
            upd, deleted = backfill._backfill_predictions(conn, dry_run=True)
        self.assertEqual(deleted, 1)          # reports the planned merge
        rows = self._read_predictions()
        self.assertEqual(len(rows), 2)        # ...but nothing was written
        self.assertIsNone([r for r in rows
                           if r["player"] == "José Ramírez"][0]["player_mlb_id"])


class PredictionEnrichmentTests(_Backend, unittest.TestCase):

    def test_baseball_row_enriched_nba_row_name_keyed(self):
        self._insert_predictions([
            {"sport_key": "baseball_mlb", "event_id": "e9",
             "prop_key": "batter_hits", "player": "Mike Trout", "team": "LAA",
             "line": 1.5, "raw_prob": 0.55, "ts": "t1", "player_key": "name:x"},
            {"sport_key": "basketball_nba", "event_id": "n1",
             "prop_key": "points", "player": "Some Guy", "line": 20.5,
             "raw_prob": 0.5, "ts": "t2", "player_key": "name:some guy"},
        ])
        with db_store.get_engine().begin() as conn:
            backfill._backfill_predictions(conn, dry_run=False)
        by_player = {r["player"]: r for r in self._read_predictions()}
        trout = by_player["Mike Trout"]
        self.assertEqual(trout["player_mlb_id"], "545361")
        self.assertEqual(trout["team_code"], "LAA")
        self.assertEqual(trout["player_key"], "mlb:545361")
        nba = by_player["Some Guy"]
        self.assertIsNone(nba["player_mlb_id"])          # non-MLB → no id
        self.assertEqual(nba["player_key"], "name:some guy")

    def test_ambiguous_namesake_stays_name_keyed(self):
        # Will Smith is two active namesakes in the fixture → mlb_id_for_name None
        # → the row must keep its name key (matches today's drop-ambiguous safety).
        self._insert_predictions([
            {"sport_key": "baseball_mlb", "event_id": "e2",
             "prop_key": "batter_hits", "player": "Will Smith", "line": 0.5,
             "raw_prob": 0.5, "ts": "t1", "player_key": "name:will smith"},
        ])
        with db_store.get_engine().begin() as conn:
            backfill._backfill_predictions(conn, dry_run=False)
        row = self._read_predictions()[0]
        self.assertIsNone(row["player_mlb_id"])
        self.assertEqual(row["player_key"], "name:will smith")


class WagerAndMarketEnrichmentTests(_Backend, unittest.TestCase):

    def _insert(self, table, rows):
        with db_store.get_engine().begin() as conn:
            for r in rows:
                conn.execute(insert(table), r)

    def test_wager_team_codes_and_player_id(self):
        self._insert(db_store.wagers, [{
            "wager_id": "w1", "sport_key": "baseball_mlb", "bet_type": "player_prop",
            "home_team": "Cleveland Guardians", "away_team": "Arizona Diamondbacks",
            "team": "Cleveland Guardians", "opponent": "Arizona Diamondbacks",
            "player": "Mike Trout", "line": 0.5, "stake": 10.0,
        }])
        with db_store.get_engine().begin() as conn:
            n = backfill._backfill_team_codes(conn, db_store.wagers, dry_run=False,
                                              player_col="player")
        self.assertEqual(n, 1)
        with db_store.get_engine().connect() as conn:
            r = dict(conn.execute(select(db_store.wagers)).first()._mapping)
        self.assertEqual(r["home_code"], "CLE")
        self.assertEqual(r["away_code"], "ARI")
        self.assertEqual(r["team_code"], "CLE")
        self.assertEqual(r["opponent_code"], "ARI")
        self.assertEqual(r["player_mlb_id"], "545361")

    def test_nba_wager_untouched(self):
        self._insert(db_store.wagers, [{
            "wager_id": "w2", "sport_key": "basketball_nba", "bet_type": "moneyline",
            "home_team": "Lakers", "team": "Lakers", "line": 0.0, "stake": 5.0,
        }])
        with db_store.get_engine().begin() as conn:
            n = backfill._backfill_team_codes(conn, db_store.wagers, dry_run=False,
                                              player_col="player")
        self.assertEqual(n, 0)               # non-baseball skipped
        with db_store.get_engine().connect() as conn:
            r = dict(conn.execute(select(db_store.wagers)).first()._mapping)
        self.assertIsNone(r["home_code"])


class OddsBackfillTests(_Backend, unittest.TestCase):

    def test_odds_snapshot_and_line_codes(self):
        with db_store.get_engine().begin() as conn:
            res = conn.execute(insert(db_store.odds_snapshot), {
                "sport": "baseball_mlb", "game_date": "2026-07-20",
                "event_id": "e1", "kind": "props", "snapshot_hour": "x",
                "home": "Cleveland Guardians", "away": "Arizona Diamondbacks",
            })
            sid = res.inserted_primary_key[0]
            # Both dicts must carry the same keys: SQLAlchemy's executemany binds
            # params from the first row, so the moneyline row spells out the
            # player-prop-only columns as None rather than omitting them.
            conn.execute(insert(db_store.odds_line), [
                {"snapshot_id": sid, "bet_type": "player_prop",
                 "selection": "Mike Trout", "player": "Mike Trout",
                 "prop_key": "batter_hits", "direction": "OVER"},
                {"snapshot_id": sid, "bet_type": "moneyline",
                 "selection": "Cleveland Guardians", "player": None,
                 "prop_key": None, "direction": None},
            ])
        with db_store.get_engine().begin() as conn:
            snaps, lines = backfill._backfill_odds(conn, dry_run=False)
        self.assertEqual((snaps, lines), (1, 2))
        with db_store.get_engine().connect() as conn:
            snap = dict(conn.execute(select(db_store.odds_snapshot)).first()._mapping)
            line_rows = [dict(m._mapping) for m in
                         conn.execute(select(db_store.odds_line)).all()]
        self.assertEqual(snap["home_code"], "CLE")
        self.assertEqual(snap["away_code"], "ARI")
        prop = [r for r in line_rows if r["bet_type"] == "player_prop"][0]
        team = [r for r in line_rows if r["bet_type"] == "moneyline"][0]
        self.assertEqual(prop["player_mlb_id"], "545361")
        self.assertEqual(team["team_code"], "CLE")


if __name__ == "__main__":
    unittest.main()
