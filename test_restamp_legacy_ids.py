"""Tests for the Commit C Phase 4 legacy identity re-stamp (restamp_legacy_ids).

Runs on in-memory SQLite; mlb_starters (the resolver + season index) is mocked so
no StatsAPI/SFBB is touched. Covers the owner-chosen policy: overwrite-drifted,
fill-gains, never-null-on-None; game_pk-null-on-change; the prediction_log
collision-merge (DELETE loser, fold outcome forward); dry-run writes nothing; the
cold-index abort; the both-teams JOIN hint; and the wagers path.
"""

import unittest
from unittest import mock

from sqlalchemy import insert, select

import db_store
import mlb_starters
import mlb_warehouse
import restamp_legacy_ids as rs


def _idx_nonempty(_season):
    return {"someone": [(1, False)]}          # non-empty → not cold


class _Backend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        mlb_warehouse.create_all()

    def tearDown(self):
        db_store.configure_engine(None)

    # ---- insert helpers -----------------------------------------------------
    def _pred(self, **kw):
        row = {"sport_key": "baseball_mlb", "event_key": kw.get("event_id") or "2026-08-10",
               "prop_key": "batter_hits", "player": "Luis Garcia Jr.",
               "game_date": "2026-08-10", "line": 0.5, "resolved": False,
               "team": "New York Yankees"}
        row.update(kw)
        row["player_key"] = db_store.player_key(row)
        with db_store.get_engine().begin() as c:
            c.execute(insert(db_store.prediction_log), row)

    def _wager(self, **kw):
        row = {"wager_id": kw.pop("wager_id", "w1"), "sport_key": "baseball_mlb",
               "player": "Luis Garcia Jr.", "prop_key": "batter_hits",
               "home_team": "New York Yankees", "away_team": "Boston Red Sox",
               "game_date": "2026-08-10"}
        row.update(kw)
        with db_store.get_engine().begin() as c:
            c.execute(insert(db_store.wagers), row)

    def _rows(self, table):
        with db_store.get_engine().connect() as c:
            return [dict(r._mapping) for r in c.execute(select(table))]

    def _restamp(self, resolve_map, dry_run=False, do_odds=False, index=_idx_nonempty):
        def _resolve(name, season, prop_key=None, teams=None, **k):
            return resolve_map.get((name, prop_key))
        with mock.patch.object(mlb_starters, "resolve_mlbam_id", side_effect=_resolve), \
             mock.patch.object(mlb_starters, "warm_player_index"), \
             mock.patch.object(mlb_starters, "_player_index", side_effect=index):
            return rs.restamp(dry_run=dry_run, do_odds=do_odds, samples=50)


class PredictionPolicyTests(_Backend, unittest.TestCase):
    def test_dry_run_writes_nothing(self):
        self._pred(player_mlb_id="677651", game_pk=999)
        s = self._restamp({("Luis Garcia Jr.", "batter_hits"): (671277, False)},
                          dry_run=True)
        self.assertEqual(s["prediction_log"]["updated"], 1)
        self.assertEqual(self._rows(db_store.prediction_log)[0]["player_mlb_id"],
                         "677651")                       # unchanged on disk

    def test_overwrite_drifted_and_null_game_pk(self):
        self._pred(player_mlb_id="677651", game_pk=999)   # a PITCHER id on a batter prop
        self._restamp({("Luis Garcia Jr.", "batter_hits"): (671277, False)})
        row = self._rows(db_store.prediction_log)[0]
        self.assertEqual(row["player_mlb_id"], "671277")
        self.assertEqual(row["player_key"], "mlb:671277")
        self.assertIsNone(row["game_pk"])                 # stale P5 pin dropped

    def test_fill_gain_from_null(self):
        self._pred(player_mlb_id=None, player="Aaron Judge", event_id="g1")
        self._restamp({("Aaron Judge", "batter_hits"): (592450, False)})
        row = self._rows(db_store.prediction_log)[0]
        self.assertEqual(row["player_mlb_id"], "592450")
        self.assertEqual(row["player_key"], "mlb:592450")

    def test_never_null_a_good_id_on_none(self):
        self._pred(player_mlb_id="123", game_pk=999)
        s = self._restamp({("Luis Garcia Jr.", "batter_hits"): None})   # resolver misses
        self.assertEqual(s["prediction_log"]["updated"], 0)             # no-op
        row = self._rows(db_store.prediction_log)[0]
        self.assertEqual(row["player_mlb_id"], "123")                   # preserved
        self.assertEqual(row["game_pk"], 999)                          # untouched

    def test_collision_merge_folds_outcome_and_deletes_loser(self):
        # A: already-correct id, RESOLVED. B: a name-keyed gain that re-stamps onto A.
        self._pred(player="Aaron Judge", player_mlb_id="592450", event_id="g9",
                   resolved=True, actual=2.0, outcome=1, resolved_at="2026-08-11",
                   ts="2026-08-11")
        self._pred(player="Aaron Judge", player_mlb_id=None, event_id="g9",
                   resolved=False, ts="2026-08-09")
        s = self._restamp({("Aaron Judge", "batter_hits"): (592450, False)})
        self.assertEqual(s["prediction_log"]["merged_deleted"], 1)
        rows = self._rows(db_store.prediction_log)
        self.assertEqual(len(rows), 1)                                  # loser removed
        self.assertEqual(rows[0]["player_key"], "mlb:592450")
        self.assertTrue(rows[0]["resolved"])
        self.assertEqual(rows[0]["outcome"], 1)                         # outcome kept

    def test_cold_index_aborts_no_write(self):
        self._pred(player_mlb_id="677651", game_pk=999)
        s = self._restamp({("Luis Garcia Jr.", "batter_hits"): (671277, False)},
                          index=lambda _s: {})            # empty index → cold
        self.assertIn("aborted", s)
        self.assertEqual(self._rows(db_store.prediction_log)[0]["player_mlb_id"],
                         "677651")                        # untouched

    def test_both_teams_recovered_via_market_join(self):
        # prediction_log carries only the own team; market_prediction_log supplies both.
        self._pred(player_mlb_id="677651", event_id="evtX")
        with db_store.get_engine().begin() as c:
            c.execute(insert(db_store.market_prediction_log), {
                "sport_key": "baseball_mlb", "event_key": "evtX", "bet_type": "moneyline",
                "side": "home", "home_team": "New York Yankees",
                "away_team": "Boston Red Sox"})
        seen = {}

        def _resolve(name, season, prop_key=None, teams=None, **k):
            seen["teams"] = teams
            return (671277, False)
        with mock.patch.object(mlb_starters, "resolve_mlbam_id", side_effect=_resolve), \
             mock.patch.object(mlb_starters, "warm_player_index"), \
             mock.patch.object(mlb_starters, "_player_index", side_effect=_idx_nonempty):
            rs.restamp(dry_run=True)
        self.assertEqual(seen["teams"], ["New York Yankees", "Boston Red Sox"])


class WagerRestampTests(_Backend, unittest.TestCase):
    def test_wager_drift_corrected_and_game_pk_nulled(self):
        self._wager(player_mlb_id="677651", game_pk=999)
        self._restamp({("Luis Garcia Jr.", "batter_hits"): (671277, False)})
        row = self._rows(db_store.wagers)[0]
        self.assertEqual(row["player_mlb_id"], "671277")
        self.assertIsNone(row["game_pk"])


if __name__ == "__main__":
    unittest.main()
