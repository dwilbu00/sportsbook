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

    def test_read_rows_retries_transient_operational_error(self):
        # A cold Azure SQL serverless resume can throw OperationalError on the
        # first read; read_rows must retry (after a backoff) instead of letting
        # the caller surface an empty store.
        from sqlalchemy.exc import OperationalError
        real_select = db_store._select_rows
        calls = {"n": 0}

        def flaky(conn, cfg, where=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OperationalError("SELECT 1", {}, Exception("resuming"))
            return real_select(conn, cfg, where)

        with patch.object(db_store, "_select_rows", side_effect=flaky), \
                patch.object(db_store.time, "sleep") as sleep:
            rows = db_store.read_rows("wagers")
        self.assertEqual(rows, [])
        self.assertEqual(calls["n"], 2)      # retried once after the failure
        sleep.assert_called_once()           # backed off before the retry

    def test_read_rows_reraises_after_exhausting_retries(self):
        from sqlalchemy.exc import OperationalError
        err = OperationalError("stmt", {}, Exception("down"))
        with patch.object(db_store, "_select_rows", side_effect=err), \
                patch.object(db_store.time, "sleep"):
            with self.assertRaises(OperationalError):
                db_store.read_rows("wagers", max_retries=2)


class SurgicalMutateTests(_SqliteBackend, unittest.TestCase):
    """mutate() writes only the delta (INSERT/UPDATE/DELETE), never delete-all +
    insert-all. Proven by autoincrement id stability: a replace-all would renumber
    every row; a surgical write keeps untouched rows' ids."""

    def _add_wagers(self, n):
        def add(rows):
            rows.extend([{"wager_id": f"w{i}", "status": "pending", "stake": 1.0}
                         for i in range(n)])
            return n
        db_store.mutate("wagers", add)

    def _wager_pk(self):
        """{wager_id: autoincrement id}."""
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(db_store.select(
                db_store.wagers.c.wager_id, db_store.wagers.c.id)).all()
        return {r[0]: r[1] for r in rows}

    def test_update_is_surgical_keeps_other_ids(self):
        self._add_wagers(3)
        before = self._wager_pk()

        def edit(rows):
            for r in rows:
                if r["wager_id"] == "w1":
                    r["stake"] = 99.0
            return 1
        db_store.mutate("wagers", edit)
        self.assertEqual(self._wager_pk(), before)       # every id stable
        got = {r["wager_id"]: r["stake"] for r in db_store.read_rows("wagers")}
        self.assertEqual(got["w1"], 99.0)
        self.assertEqual(got["w0"], 1.0)                 # untouched row unchanged

    def test_delete_is_surgical_keeps_other_ids(self):
        self._add_wagers(3)
        before = self._wager_pk()

        def prune(rows):
            rows[:] = [r for r in rows if r["wager_id"] != "w1"]
            return 1
        db_store.mutate("wagers", prune)
        after = self._wager_pk()
        self.assertNotIn("w1", after)
        self.assertEqual(after["w0"], before["w0"])      # survivors keep ids
        self.assertEqual(after["w2"], before["w2"])

    def test_insert_appends_and_keeps_existing_ids(self):
        self._add_wagers(2)
        before = self._wager_pk()

        def add_one(rows):
            rows.append({"wager_id": "w9", "status": "pending", "stake": 5.0})
            return 1
        db_store.mutate("wagers", add_one)
        after = self._wager_pk()
        self.assertEqual(after["w0"], before["w0"])
        self.assertEqual(after["w1"], before["w1"])
        self.assertGreater(after["w9"], max(before.values()))   # appended after

    def test_truthy_but_unchanged_writes_nothing(self):
        # A mutator that reports a change but leaves every field equal emits no
        # UPDATE (change detection compares coerced params) — ids untouched.
        self._add_wagers(1)
        before = self._wager_pk()
        db_store.mutate("wagers", lambda rows: 1)
        self.assertEqual(self._wager_pk(), before)

    def test_duplicate_identity_in_mutation_raises_and_rolls_back(self):
        # The diff must not silently collapse two same-identity rows (last-writer-
        # wins); it fails loudly like the old delete-all+insert-all UNIQUE path.
        def add_dupes(rows):
            rows.append({"wager_id": "dup", "status": "pending", "stake": 1.0})
            rows.append({"wager_id": "dup", "status": "won", "stake": 2.0})
            return 1
        with self.assertRaises(ValueError):
            db_store.mutate("wagers", add_dupes)
        self.assertEqual(db_store.read_rows("wagers"), [])   # rolled back

    def test_read_rows_where_filters(self):
        def add(rows):
            rows.append({"wager_id": "p1", "status": "pending", "stake": 1.0})
            rows.append({"wager_id": "s1", "status": "won", "stake": 1.0})
            return 1
        db_store.mutate("wagers", add)
        pend = db_store.read_rows("wagers", where={"status": "pending"})
        self.assertEqual([r["wager_id"] for r in pend], ["p1"])
        both = db_store.read_rows("wagers", where={"status": ["pending", "won"]})
        self.assertEqual({r["wager_id"] for r in both}, {"p1", "s1"})

    def test_where_filtered_mutate_only_sees_and_touches_subset(self):
        def add(rows):
            rows.append({"wager_id": "p1", "status": "pending", "stake": 1.0})
            rows.append({"wager_id": "s1", "status": "won", "stake": 1.0})
            return 1
        db_store.mutate("wagers", add)
        before = self._wager_pk()

        seen = {}

        def grade(rows):
            seen["ids"] = [r["wager_id"] for r in rows]   # only the subset
            for r in rows:
                r["status"] = "won"
                r["profit"] = 2.0
            return 1
        db_store.mutate("wagers", grade, where={"status": "pending"})
        self.assertEqual(seen["ids"], ["p1"])             # settled row not read
        self.assertEqual(self._wager_pk(), before)        # no id churn
        got = {r["wager_id"]: r for r in db_store.read_rows("wagers")}
        self.assertEqual(got["p1"]["status"], "won")
        self.assertEqual(got["p1"]["profit"], 2.0)

    def test_prediction_resolve_updates_in_place(self):
        def add(rows):
            rows.append({"sport_key": "baseball_mlb", "event_id": "e1",
                         "prop_key": "batter_hits", "player": "X",
                         "game_date": "2026-07-20", "line": 0.5, "raw_prob": 0.6,
                         "resolved": False})
            return 1
        db_store.mutate("prediction_log", add)
        with db_store.get_engine().connect() as conn:
            id_before = conn.execute(
                db_store.select(db_store.prediction_log.c.id)).scalar()

        def resolve(rows):
            for r in rows:
                r["actual"] = 2.0
                r["outcome"] = 1
                r["resolved"] = True
            return 1
        db_store.mutate("prediction_log", resolve,
                        where={"resolved": False})
        with db_store.get_engine().connect() as conn:
            ids = conn.execute(
                db_store.select(db_store.prediction_log.c.id)).all()
        self.assertEqual([r[0] for r in ids], [id_before])   # updated, not reinserted
        got = db_store.read_rows("prediction_log")[0]
        self.assertEqual(got["outcome"], 1)
        self.assertIs(got["resolved"], True)

    def test_prediction_identity_null_event_id_uses_game_date(self):
        # event_id NULL → identity keys on game_date; a later update must MATCH the
        # existing row (UPDATE), not insert a duplicate.
        def add(rows):
            rows.append({"sport_key": "baseball_mlb", "event_id": None,
                         "prop_key": "batter_hits", "player": "Y",
                         "game_date": "2026-07-21", "line": 1.5, "raw_prob": 0.5,
                         "resolved": False})
            return 1
        db_store.mutate("prediction_log", add)
        with db_store.get_engine().connect() as conn:
            id_before = conn.execute(
                db_store.select(db_store.prediction_log.c.id)).scalar()

        def resolve(rows):
            for r in rows:
                r["resolved"] = True
                r["outcome"] = 0
                r["actual"] = 1.0
            return 1
        db_store.mutate("prediction_log", resolve)
        with db_store.get_engine().connect() as conn:
            ids = conn.execute(
                db_store.select(db_store.prediction_log.c.id)).all()
        self.assertEqual([r[0] for r in ids], [id_before])   # no duplicate insert


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

    def test_market_prediction_log_routes_to_sql(self):
        ar = {
            "events": {"e1": {"commence_time": "2020-05-01T23:00:00Z",
                              "game_date": "2020-05-01", "home_team": "Rockies",
                              "away_team": "Astros"}},
            "all_ml": [{"event_id": "e1", "team": "Astros", "opponent": "Rockies",
                        "home_away": "AWAY", "blended_prob": 62.0,
                        "model_prob": 60.0, "best_price": -140, "best_book": "DK",
                        "is_value": True}],
            "all_spreads": [{"event_id": "e1", "team": "Astros",
                             "opponent": "Rockies", "home_away": "AWAY",
                             "spread": -1.5, "cover_rate": 55.0,
                             "model_cover_rate": 53.0, "price": -110,
                             "is_value": False}],
            "all_totals": [{"event_id": "e1", "matchup": "Astros @ Rockies",
                            "line": 9.5, "over_hit_rate": 58.0,
                            "model_over_hit_rate": 56.0, "over_price": -105,
                            "under_price": -115, "is_over_value": True,
                            "is_under_value": False}],
        }
        rows = recalibration.build_market_prediction_rows(ar, "baseball_mlb")
        self.assertEqual(len(rows), 3)
        recalibration.log_market_prediction_rows(rows)
        stored = db_store.read_rows("market_prediction_log")
        self.assertEqual(len(stored), 3)                       # in SQL
        by_type = {r["bet_type"]: r for r in stored}
        self.assertEqual(by_type["moneyline"]["side"], "away")
        self.assertIs(by_type["moneyline"]["is_value"], True)
        self.assertEqual(by_type["spread"]["point"], -1.5)
        # Re-logging the same slate stays one row per (sport, event, market).
        recalibration.log_market_prediction_rows(rows)
        self.assertEqual(len(db_store.read_rows("market_prediction_log")), 3)

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

    def test_market_prediction_spec_matches_table(self):
        table_cols = {c.name for c in db_store.market_prediction_log.columns}
        spec_cols = ({n for n, _ in db_store._MARKET_PREDICTION_SPEC}
                     | {"id", "event_key"})
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

    def test_odds_snapshot_columns(self):
        self.assertEqual(
            {c.name for c in db_store.odds_snapshot.columns},
            {"id", "sport", "game_date", "event_id", "kind", "snapshot_hour",
             "captured_at", "commence_time", "home", "away", "regions",
             "markets", "bookmakers"})

    def test_odds_line_columns(self):
        self.assertEqual(
            {c.name for c in db_store.odds_line.columns},
            {"id", "snapshot_id", "bet_type", "selection", "point", "player",
             "prop_key", "direction", "price", "implied_prob"})


class WarehouseSqlTests(_SqliteBackend, unittest.TestCase):
    """Phase B: normalized odds warehouse on SQL (capture + closing_line_for)."""

    def _meta(self, hour, kind="team"):
        return {"sport": "baseball_mlb", "game_date": "2026-07-22",
                "event_id": "e1", "kind": kind, "snapshot_hour": hour,
                "captured_at": f"2026-07-22T{hour[-3:-1]}:00:00Z",
                "commence_time": "2026-07-22T23:00:00Z",
                "home": "Rockies", "away": "Astros", "regions": "us",
                "markets": "h2h,spreads,totals", "bookmakers": None}

    def test_capture_write_once_and_lookup(self):
        lines = [
            {"bet_type": "moneyline", "selection": "Rockies", "price": 120,
             "implied_prob": 0.45},
            {"bet_type": "total", "selection": "Over", "point": 9.5,
             "price": -105, "implied_prob": 0.51},
        ]
        self.assertTrue(db_store.capture_odds_snapshot(self._meta("20260722T18Z"), lines))
        # Write-once: same (sport,date,event,kind,hour) is rejected.
        self.assertFalse(db_store.capture_odds_snapshot(self._meta("20260722T18Z"), lines))
        snaps = db_store.odds_snapshots_for_event("baseball_mlb", "2026-07-22", "e1")
        self.assertEqual(len(snaps), 1)
        self.assertEqual(
            db_store.odds_line_lookup(snaps[0]["id"], "moneyline",
                                      selection="Rockies")["price"], 120)
        # Total point-exact + h2h alias.
        self.assertEqual(
            db_store.odds_line_lookup(snaps[0]["id"], "totals",
                                      selection="Over", point=9.5)["price"], -105)

    def test_spread_point_fallback(self):
        db_store.capture_odds_snapshot(self._meta("20260722T18Z"), [
            {"bet_type": "spread", "selection": "Rockies", "point": 1.5,
             "price": -110, "implied_prob": 0.52}])
        sid = db_store.odds_snapshots_for_event(
            "baseball_mlb", "2026-07-22", "e1")[0]["id"]
        # Exact point missing → best price for the selection.
        self.assertEqual(
            db_store.odds_line_lookup(sid, "spread", selection="Rockies",
                                      point=2.5)["price"], -110)

    def test_closing_line_picks_nearest_at_or_before_commence(self):
        import warehouse
        # Two snapshots: 18Z (before 23Z commence) and 22Z (closer, before).
        db_store.capture_odds_snapshot(self._meta("20260722T18Z"), [
            {"bet_type": "moneyline", "selection": "Rockies", "price": 100,
             "implied_prob": 0.5}])
        db_store.capture_odds_snapshot(self._meta("20260722T22Z"), [
            {"bet_type": "moneyline", "selection": "Rockies", "price": 130,
             "implied_prob": 0.43}])
        close = warehouse.closing_line_for(
            "baseball_mlb", "2026-07-22", "e1", "moneyline",
            selection="Rockies", commence_time="2026-07-22T23:00:00Z")
        self.assertEqual(close["price"], 130)          # the 22Z (nearest) snapshot
        self.assertEqual(close["captured_at"], "2026-07-22T22:00:00Z")

    def test_capture_event_odds_parses_payload(self):
        import warehouse
        payload = {
            "id": "e1", "home_team": "Rockies", "away_team": "Astros",
            "commence_time": "2026-07-22T23:00:00Z",
            "bookmakers": [
                {"key": "draftkings", "title": "DraftKings", "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Rockies", "price": 118},
                        {"name": "Astros", "price": -140}]}]},
                {"key": "fanduel", "title": "FanDuel", "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Rockies", "price": 122},  # best → stored
                        {"name": "Astros", "price": -145}]}]},
            ],
        }
        warehouse.capture_event_odds(
            "baseball_mlb", "e1", "us", "h2h", None, payload)
        close = warehouse.closing_line_for(
            "baseball_mlb", "2026-07-22", "e1", "moneyline",
            selection="Rockies", commence_time="2026-07-22T23:00:00Z")
        self.assertEqual(close["price"], 122)          # best across books
        self.assertEqual(warehouse.storage_backend(), "Azure SQL")


class TeamMarketLinesSqlTests(_SqliteBackend, unittest.TestCase):
    """Phase B: bulk team-market reader (db_store.team_market_lines) + the
    warehouse store assembler (warehouse.load_team_market_store)."""

    def _meta(self, hour):
        return {"sport": "baseball_mlb", "game_date": "2026-07-22",
                "event_id": "e1", "kind": "team", "snapshot_hour": hour,
                "captured_at": f"2026-07-22T{hour[-3:-1]}:00:00Z",
                "commence_time": "2026-07-22T23:00:00Z",
                "home": "Rockies", "away": "Astros", "regions": "us",
                "markets": "h2h,spreads,totals", "bookmakers": None}

    def _team_lines(self, ml_home, ml_away):
        return [
            {"bet_type": "moneyline", "selection": "Rockies", "price": ml_home,
             "implied_prob": 0.5},
            {"bet_type": "moneyline", "selection": "Astros", "price": ml_away,
             "implied_prob": 0.5},
            {"bet_type": "spread", "selection": "Rockies", "point": 1.5,
             "price": -110, "implied_prob": 0.52},
            {"bet_type": "spread", "selection": "Astros", "point": -1.5,
             "price": -110, "implied_prob": 0.52},
            {"bet_type": "total", "selection": "Over", "point": 9.5,
             "price": -105, "implied_prob": 0.51},
            {"bet_type": "total", "selection": "Under", "point": 9.5,
             "price": -105, "implied_prob": 0.51},
            # A prop line — MUST be excluded by team_market_lines.
            {"bet_type": "player_prop", "selection": "Kris Bryant",
             "player": "Kris Bryant", "prop_key": "batter_hits",
             "direction": "OVER", "point": 0.5, "price": -120,
             "implied_prob": 0.55},
        ]

    def test_excludes_props_and_filters_dates(self):
        db_store.capture_odds_snapshot(self._meta("20260722T18Z"),
                                       self._team_lines(100, -120))
        rows = db_store.team_market_lines("baseball_mlb")
        self.assertTrue(rows)
        self.assertEqual({r["bet_type"] for r in rows},
                         {"moneyline", "spread", "total"})   # no player_prop
        self.assertTrue(all(r["event_id"] == "e1" for r in rows))
        # Explicit-date filter.
        self.assertEqual(
            db_store.team_market_lines("baseball_mlb", dates=["2025-01-01"]), [])
        self.assertTrue(
            db_store.team_market_lines("baseball_mlb", dates=["2026-07-22"]))
        # Range filter.
        self.assertTrue(db_store.team_market_lines(
            "baseball_mlb", date_from="2026-07-01", date_to="2026-07-31"))
        self.assertEqual(
            db_store.team_market_lines("baseball_mlb", date_from="2026-08-01"), [])

    def test_store_shape_closing_pick_and_backtest_parity(self):
        import warehouse
        import backtest
        # Early snapshot (18Z) then the closing one (22Z, nearest before 23Z).
        db_store.capture_odds_snapshot(self._meta("20260722T18Z"),
                                       self._team_lines(100, -120))
        db_store.capture_odds_snapshot(self._meta("20260722T22Z"),
                                       self._team_lines(130, -150))
        store = warehouse.load_team_market_store("baseball_mlb")
        self.assertEqual(len(store["games"]), 1)
        entry = next(iter(store["games"].values()))
        # Moneyline: both teams, from the CLOSING (22Z) snapshot.
        self.assertEqual(entry["moneyline"]["Rockies"][0]["price"], 130)
        self.assertEqual(entry["moneyline"]["Astros"][0]["price"], -150)
        # Spread: mirrored home +1.5 / away -1.5.
        self.assertEqual(entry["spreads"]["Rockies"][0]["spread"], 1.5)
        self.assertEqual(entry["spreads"]["Astros"][0]["spread"], -1.5)
        # Total: same line on both sides.
        self.assertEqual(entry["totals"]["Over"][0]["line"], 9.5)
        self.assertEqual(entry["totals"]["Under"][0]["line"], 9.5)
        # Parity: the backtest market readers consume the entry unchanged.
        self.assertIsNotNone(backtest._moneyline_market(entry))
        self.assertIsNotNone(backtest._spread_market(entry))
        self.assertIsNotNone(backtest._total_market(entry))
        # Every offer must carry the parse_game_odds fields the LIVE analyzers
        # read — notably "book" (analyze_moneyline_value uses best_offer["book"]).
        for team in (entry["moneyline"]["Rockies"] + entry["moneyline"]["Astros"]):
            self.assertEqual(set(team), {"book", "price", "implied_prob"})
        for team in (entry["spreads"]["Rockies"] + entry["spreads"]["Astros"]):
            self.assertEqual(set(team), {"book", "spread", "price"})
        for side in (entry["totals"]["Over"] + entry["totals"]["Under"]):
            self.assertEqual(set(side), {"book", "line", "price"})

    def test_store_empty_when_sql_off(self):
        import warehouse
        db_store.configure_engine(None)   # SQL disabled
        try:
            self.assertEqual(
                warehouse.load_team_market_store("baseball_mlb")["games"], {})
        finally:
            db_store.configure_engine("sqlite://")
            db_store.create_all()

    def test_partial_entry_keeps_all_three_market_keys(self):
        # A snapshot with ONLY a total (no moneyline, no mirrored spread) must
        # still assemble an entry carrying moneyline/spreads/totals keys (={} for
        # the absent markets) — the live analyzers hard-subscript all three, so a
        # missing key would KeyError-abort the backtest.
        import warehouse
        db_store.capture_odds_snapshot(self._meta("20260722T18Z"), [
            {"bet_type": "total", "selection": "Over", "point": 9.5,
             "price": -105, "implied_prob": 0.51},
            {"bet_type": "total", "selection": "Under", "point": 9.5,
             "price": -105, "implied_prob": 0.51}])
        entry = next(iter(
            warehouse.load_team_market_store("baseball_mlb")["games"].values()))
        self.assertEqual(entry["moneyline"], {})
        self.assertEqual(entry["spreads"], {})
        self.assertTrue(entry["totals"])
        for key in ("moneyline", "spreads", "totals"):
            self.assertIn(key, entry)


class PlayerPropLinesSqlTests(_SqliteBackend, unittest.TestCase):
    """Player-prop bulk reader (db_store.player_prop_lines) + warehouse
    closing-line assembler (load_prop_lines) + doubleheader detection."""

    def _meta(self, hour, event_id="e1", commence="2026-07-22T23:00:00Z",
              home="Rockies", away="Astros", game_date="2026-07-22"):
        return {"sport": "baseball_mlb", "game_date": game_date,
                "event_id": event_id, "kind": "props", "snapshot_hour": hour,
                "captured_at": f"2026-07-22T{hour[-3:-1]}:00:00Z",
                "commence_time": commence, "home": home, "away": away,
                "regions": "us", "markets": "batter_hits", "bookmakers": None}

    def _prop_lines(self, over_price, under_price, line=0.5,
                    player="Kris Bryant", prop_key="batter_hits"):
        return [
            {"bet_type": "player_prop", "selection": player, "player": player,
             "prop_key": prop_key, "direction": "OVER", "point": line,
             "price": over_price, "implied_prob": 0.55},
            {"bet_type": "player_prop", "selection": player, "player": player,
             "prop_key": prop_key, "direction": "UNDER", "point": line,
             "price": under_price, "implied_prob": 0.45},
            # a team line the prop reader MUST exclude
            {"bet_type": "moneyline", "selection": "Rockies", "price": 120,
             "implied_prob": 0.45},
        ]

    def test_reader_excludes_team_and_carries_prop_fields(self):
        db_store.capture_odds_snapshot(self._meta("20260722T18Z"),
                                       self._prop_lines(-110, -110))
        rows = db_store.player_prop_lines("baseball_mlb")
        self.assertTrue(rows)
        self.assertTrue(all(r["prop_key"] == "batter_hits" for r in rows))
        self.assertEqual({r["direction"] for r in rows}, {"OVER", "UNDER"})
        self.assertTrue(all(r["player"] == "Kris Bryant" for r in rows))

    def test_load_prop_lines_closing_pick_combine_and_et_date(self):
        import warehouse
        db_store.capture_odds_snapshot(self._meta("20260722T18Z"),
                                       self._prop_lines(-105, -115))
        db_store.capture_odds_snapshot(self._meta("20260722T22Z"),
                                       self._prop_lines(120, -140))
        rows = warehouse.load_prop_lines("baseball_mlb")
        self.assertEqual(len(rows), 1)          # one (event, player, prop)
        r = rows[0]
        self.assertEqual(r["line"], 0.5)
        self.assertEqual(r["over_price"], 120)  # from the CLOSING (22Z) snapshot
        self.assertEqual(r["under_price"], -140)
        self.assertEqual(r["event_id"], "e1")
        self.assertEqual(r["game_date"], "2026-07-22")   # ET (23:00Z → 19:00 EDT)

    def test_load_prop_lines_et_date_crosses_utc_midnight(self):
        import warehouse
        # commence 00:30Z on the 23rd = 20:30 EDT on the 22nd → ET date 07-22,
        # while the stored (UTC) game_date is 07-23 (UTC at rest, ET on read).
        db_store.capture_odds_snapshot(
            self._meta("20260722T20Z", commence="2026-07-23T00:30:00Z",
                       game_date="2026-07-23"),
            self._prop_lines(-110, -110))
        rows = warehouse.load_prop_lines("baseball_mlb")
        self.assertEqual(rows[0]["game_date"], "2026-07-22")

    def test_doubleheader_detection_vs_consecutive_day(self):
        import warehouse
        dh = [  # same ET date + teams, two event_ids → doubleheader
            {"game_date": "2026-07-28", "home_team": "Reds",
             "away_team": "Guardians", "event_id": "g1"},
            {"game_date": "2026-07-28", "home_team": "Reds",
             "away_team": "Guardians", "event_id": "g2"}]
        self.assertEqual(warehouse.doubleheader_event_ids(dh), {"g1", "g2"})
        consec = [  # distinct ET dates → NOT a doubleheader
            {"game_date": "2026-07-24", "home_team": "Rangers",
             "away_team": "Mariners", "event_id": "a1"},
            {"game_date": "2026-07-25", "home_team": "Rangers",
             "away_team": "Mariners", "event_id": "a2"}]
        self.assertEqual(warehouse.doubleheader_event_ids(consec), set())


class RefitPerformedTests(_SqliteBackend, unittest.TestCase):
    """refit_performed flag + count_rows + recalibration count/mark helpers that
    power the app's 'time to refit' banner."""

    def _seed(self):
        rows = [("A", True, False), ("B", True, False),
                ("C", False, False), ("D", True, True)]

        def add(existing):
            for player, resolved, refit in rows:
                existing.append({
                    "sport_key": "baseball_mlb", "event_id": "E1",
                    "prop_key": "batter_hits", "player": player, "line": 0.5,
                    "resolved": resolved, "refit_performed": refit})
            return len(rows)
        db_store.mutate("prediction_log", add)

    def test_new_row_defaults_false(self):
        self._seed()
        rows = {r["player"]: r for r in db_store.read_rows("prediction_log")}
        self.assertFalse(rows["A"]["refit_performed"])   # not-null default 0

    def test_count_rows_filtered(self):
        self._seed()
        self.assertEqual(
            db_store.count_rows("prediction_log",
                                where={"resolved": True, "refit_performed": False}),
            2)                                            # A, B only

    def test_count_pending_and_mark(self):
        import recalibration
        self._seed()
        self.assertEqual(recalibration.count_pending_refit("baseball_mlb"), 2)
        self.assertEqual(recalibration.mark_predictions_refit("baseball_mlb"), 2)
        self.assertEqual(recalibration.count_pending_refit("baseball_mlb"), 0)
        rows = {r["player"]: r for r in db_store.read_rows("prediction_log")}
        self.assertFalse(rows["C"]["refit_performed"])   # unresolved untouched
        self.assertTrue(rows["D"]["refit_performed"])    # already-flagged kept


if __name__ == "__main__":
    unittest.main()
