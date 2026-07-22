"""Tests for the Azure SQL durable-state backend (db_store) and the
recalibration/wagers dispatch, exercised against in-memory SQLite so pymssql and
the live Azure database are never touched.

The SQL backend is env-gated (db_store enables only when SQL_* env vars are set,
which tests never do) plus a test override (configure_engine). tearDown clears the
override AND the recalibration in-memory caches so SQL never leaks into the other
hermetic storage tests (test_wagers, test_recalibration_durability, ...).
"""

import tempfile
import unittest
from unittest.mock import patch

import db_store
import recalibration
import wagers


class _SqliteBackend:
    """Fresh in-memory SQLite DB with the schema created, per test."""

    def setUp(self):
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()

    def tearDown(self):
        db_store.configure_engine(None)  # → enabled() False (no SQL_* env)
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()


class DbStoreOpsTests(_SqliteBackend, unittest.TestCase):

    def test_enabled_with_override(self):
        self.assertTrue(db_store.enabled())

    def test_disabled_after_teardown(self):
        db_store.configure_engine(None)
        self.assertFalse(db_store.enabled())  # no SQL_* env in tests

    def test_wager_column_roundtrip_and_types(self):
        def add(rows):
            rows.append({"wager_id": "w1", "status": "pending",
                         "sport_key": "baseball_mlb", "bet_type": "player_prop",
                         "stake": 10.0, "line": 1.5, "executed_price": -115,
                         "direction": "OVER", "player": "Slugger"})
            return 1
        self.assertEqual(db_store.mutate("wagers", add), 1)
        r = db_store.read_rows("wagers")[0]
        # Full standard key set present; correct types on the round-trip.
        self.assertEqual(set(r), {name for name, _ in db_store._WAGER_SPEC})
        self.assertIsInstance(r["stake"], float)
        self.assertIsInstance(r["executed_price"], int)
        self.assertEqual(r["line"], 1.5)
        self.assertIsNone(r["close_price"])       # unset column → None

    def test_prediction_tristate_and_int_zero_preserved(self):
        def add(rows):
            rows.append({"sport_key": "baseball_mlb", "event_id": "e1",
                         "prop_key": "batter_hits", "player": "Under Won",
                         "game_date": "2026-07-20", "line": 0.5, "raw_prob": 0.4,
                         "is_value": False, "resolved": True, "actual": 0.0,
                         "outcome": 0})   # under won: outcome 0 must NOT become NULL
            rows.append({"sport_key": "baseball_mlb", "event_id": "e1",
                         "prop_key": "batter_hits", "player": "Unresolved",
                         "game_date": "2026-07-20", "line": 1.5, "raw_prob": 0.4,
                         "is_value": None, "resolved": False,
                         "actual": None, "outcome": None})
            return 2
        db_store.mutate("prediction_log", add)
        by_player = {r["player"]: r for r in db_store.read_rows("prediction_log")}
        won = by_player["Under Won"]
        self.assertIs(won["is_value"], False)     # falsy, not NULL
        self.assertEqual(won["outcome"], 0)       # 0 preserved, not NULL
        self.assertEqual(won["actual"], 0.0)
        self.assertIsInstance(won["price"], type(None))
        pending = by_player["Unresolved"]
        self.assertIsNone(pending["is_value"])    # tri-state NULL
        self.assertIsNone(pending["outcome"])
        self.assertIs(pending["resolved"], False)

    def test_status_check_constraint_rolls_back(self):
        def bad(rows):
            rows.append({"wager_id": "x", "status": "bogus", "stake": 1.0})
            return 1
        with self.assertRaises(Exception):
            db_store.mutate("wagers", bad)
        self.assertEqual(db_store.read_rows("wagers"), [])  # transaction rolled back

    def test_negative_stake_rejected(self):
        def bad(rows):
            rows.append({"wager_id": "x", "status": "pending", "stake": -5})
            return 1
        with self.assertRaises(Exception):
            db_store.mutate("wagers", bad)

    def test_zero_stake_accepted(self):
        # The app's default/paper unit stake is 0.0; the Blob path accepted it,
        # so SQL must too (CHECK is stake >= 0, not > 0).
        def add(rows):
            rows.append({"wager_id": "z", "status": "pending", "stake": 0.0})
            return 1
        self.assertEqual(db_store.mutate("wagers", add), 1)
        self.assertEqual(db_store.read_rows("wagers")[0]["stake"], 0.0)

    def test_resolved_defaults_false_when_absent(self):
        # `resolved` is NOT NULL; a row lacking it must store False, not crash.
        def add(rows):
            rows.append({"sport_key": "baseball_mlb", "event_id": "e9",
                         "prop_key": "batter_hits", "player": "NoResolvedKey",
                         "game_date": "2026-07-20", "line": 0.5, "raw_prob": 0.5})
            return 1
        db_store.mutate("prediction_log", add)
        self.assertIs(db_store.read_rows("prediction_log")[0]["resolved"], False)

    def test_falsy_mutator_skips_write(self):
        self.assertEqual(db_store.mutate("wagers", lambda rows: 0), 0)
        self.assertEqual(db_store.read_rows("wagers"), [])

    def test_unknown_store_raises(self):
        with self.assertRaises(KeyError):
            db_store.read_rows("not_a_table")

    def test_ordering_is_insertion_order(self):
        def add(rows):
            rows.extend([{"wager_id": f"w{i}", "status": "pending", "stake": 1.0}
                         for i in range(5)])
            return 1
        db_store.mutate("wagers", add)
        ids = [r["wager_id"] for r in db_store.read_rows("wagers")]
        self.assertEqual(ids, ["w0", "w1", "w2", "w3", "w4"])

    def test_recal_roundtrip_with_folds_child_table(self):
        cfg = {
            "sport_key": "baseball_mlb",
            "fit_timestamp": "2026-07-22T00:00:00+00:00",
            "props": {"batter_hits": {
                "a": 0.5, "b": 0.2, "n_fit": 938, "n_validation": 402,
                "validated": True, "source": "seed",
                "validation_folds": [
                    {"holdout_start": "2026-07-17", "n_validation": 210,
                     "raw_brier": 0.25, "calibrated_brier": 0.24},
                    {"holdout_start": "2026-07-19", "n_validation": 192,
                     "raw_brier": 0.24, "calibrated_brier": 0.23},
                ]}},
            "meta": {"source": "test"},
        }
        db_store.save_recal("baseball_mlb", cfg)
        got = db_store.load_recal("baseball_mlb")
        self.assertEqual(got["fit_timestamp"], cfg["fit_timestamp"])
        self.assertEqual(got["meta"], {"source": "test"})
        prop = got["props"]["batter_hits"]
        self.assertEqual(prop["a"], 0.5)
        self.assertIs(prop["validated"], True)
        # Folds reconstructed from the child table, in order.
        self.assertEqual(len(prop["validation_folds"]), 2)
        self.assertEqual(prop["validation_folds"][0]["holdout_start"],
                         "2026-07-17")
        self.assertEqual(prop["validation_folds"][1]["n_validation"], 192)

    def test_recal_no_folds_omits_key(self):
        db_store.save_recal("baseball_mlb", {"props": {
            "pitcher_strikeouts": {"a": 0.7, "validated": True}}})
        prop = db_store.load_recal("baseball_mlb")["props"]["pitcher_strikeouts"]
        self.assertNotIn("validation_folds", prop)

    def test_recal_save_replaces_sport(self):
        db_store.save_recal("baseball_mlb", {"props": {
            "batter_hits": {"a": 1.0, "validated": True}}})
        db_store.save_recal("baseball_mlb", {"props": {
            "pitcher_strikeouts": {"a": 2.0, "validated": True}}})
        got = db_store.load_recal("baseball_mlb")
        self.assertNotIn("batter_hits", got["props"])
        self.assertIn("pitcher_strikeouts", got["props"])

    def test_load_recal_missing_returns_none(self):
        self.assertIsNone(db_store.load_recal("no_such_sport"))


class SqlDispatchTests(_SqliteBackend, unittest.TestCase):

    def test_recalibration_sql_flag_on(self):
        self.assertTrue(recalibration._sql())
        self.assertEqual(recalibration.prediction_log_storage(), "Azure SQL")

    def test_prediction_log_routes_to_sql(self):
        row = {"sport_key": "baseball_mlb", "event_id": "e1",
               "prop_key": "batter_hits", "player": "X",
               "game_date": "2026-07-20", "line": 0.5, "raw_prob": 0.6,
               "final_prob": 0.6, "projected": 1.2, "direction": "OVER",
               "price": -110, "book": "DK", "is_value": True,
               "resolved": False, "actual": None, "outcome": None}
        self.assertEqual(recalibration.log_prediction_rows([row]), 1)
        self.assertEqual(len(db_store.read_rows("prediction_log")), 1)  # in SQL
        got = recalibration.read_prediction_log()
        self.assertEqual(got[0]["player"], "X")
        self.assertIs(got[0]["is_value"], True)
        # Re-logging the same identity stays a single row (upsert semantics).
        recalibration.log_prediction_rows([row])
        self.assertEqual(len(db_store.read_rows("prediction_log")), 1)

    def test_wagers_submit_read_update_delete_via_sql(self):
        row = {"wager_id": "w1", "placed_at": "2026-07-20T00:00:00+00:00",
               "sport_key": "baseball_mlb", "bet_type": "player_prop",
               "status": "pending", "stake": 10.0, "line": 0.5,
               "direction": "OVER", "player": "X"}
        self.assertEqual(wagers.submit_wagers([row]), 1)
        got = wagers.read_wagers()
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["wager_id"], "w1")
        # Submit dedups by wager_id.
        self.assertEqual(wagers.submit_wagers([row]), 0)
        # Edit a pending field, then confirm the fresh read (cache invalidated).
        self.assertEqual(wagers.update_wagers({"w1": {"stake": 20.0}}), 1)
        self.assertEqual(wagers.read_wagers()[0]["stake"], 20.0)
        # Delete.
        self.assertEqual(wagers.delete_wagers(["w1"]), 1)
        self.assertEqual(wagers.read_wagers(), [])

    def test_recal_overlay_and_sql_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(recalibration, "CALIB_DIR", tmp):
                # Empty SQL recal → overlay the git-committed local baseline.
                recalibration._LOAD_CACHE.pop("baseball_mlb", None)
                baseline = (1.0, {"batter_hits": {"a": 1.0, "b": 0.0,
                                                  "validated": True}})
                with patch.object(recalibration, "_read_local_recal",
                                  return_value=baseline):
                    props = recalibration.load_recalibration("baseball_mlb")
                self.assertIn("batter_hits", props)
                # After a validated SQL fit, SQL wins over the baseline.
                recalibration._LOAD_CACHE.pop("baseball_mlb", None)
                per_prop = {"pitcher_strikeouts": {"a": 0.7, "b": 0.1,
                                                   "validated": True}}
                recalibration.save_recalibration("baseball_mlb", per_prop,
                                                 to_blob=True)
                props2 = recalibration.load_recalibration("baseball_mlb")
                self.assertIn("pitcher_strikeouts", props2)


class SchemaParityTests(unittest.TestCase):
    """Guard db_store's write/read column SPECs against the Table definitions, so
    the columns can't silently drift from the schema (which sql/schema.sql mirrors
    for the hand-run Azure DDL)."""

    def test_prediction_spec_matches_table(self):
        table_cols = {c.name for c in db_store.prediction_log.columns}
        spec_cols = ({n for n, _ in db_store._PREDICTION_SPEC}
                     | {"id", "event_key"})
        self.assertEqual(spec_cols, table_cols)

    def test_wager_spec_matches_table(self):
        table_cols = {c.name for c in db_store.wagers.columns}
        spec_cols = {n for n, _ in db_store._WAGER_SPEC} | {"id"}
        self.assertEqual(spec_cols, table_cols)

    def test_recal_param_spec_matches_table(self):
        table_cols = {c.name for c in db_store.recalibration_params.columns}
        spec_cols = ({n for n, _ in db_store._RECAL_PARAM_SPEC}
                     | {"sport_key", "prop_key"})
        self.assertEqual(spec_cols, table_cols)

    def test_recal_fold_spec_matches_table(self):
        table_cols = {c.name for c in db_store.recalibration_folds.columns}
        spec_cols = ({n for n, _ in db_store._RECAL_FOLD_SPEC}
                     | {"sport_key", "prop_key", "fold_index"})
        self.assertEqual(spec_cols, table_cols)


if __name__ == "__main__":
    unittest.main()
