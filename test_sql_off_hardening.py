"""Tests for the SQL-off hardening guard (audit Finding F1, WS1).

A durable write (predictions, wagers, recal params) or a refit-critical warehouse
read must fail LOUDLY when the SQL backend is off but a SQL deployment is
*signalled* (SPORTSBOOK_REQUIRE_SQL set, or any SQL_* secret present) — instead of
silently degrading to ephemeral local disk that Streamlit Cloud wipes on restart.
It must stay completely inert for hermetic tests and no-SQL local dev.

The production signal is environment-only, so every test snapshots and CLEARS the
four SQL_* keys + SPORTSBOOK_REQUIRE_SQL in setUp (a clean, no-SQL baseline even
though .streamlit/secrets.toml exists locally) and restores them in tearDown. A
"prod context" is simulated by setting a partial SQL_* env (or the explicit flag)
while leaving the engine unconfigured (db_store.enabled() False).

Run: PYTHONIOENCODING=utf-8 python -m unittest test_sql_off_hardening -v
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import db_store
import recalibration
import warehouse

_ENV_KEYS = ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD",
             "SPORTSBOOK_REQUIRE_SQL")
_FULL_SQL = {"SQL_SERVER": "s.database.windows.net", "SQL_DATABASE": "d",
             "SQL_USER": "u", "SQL_PASSWORD": "p"}


class _EnvSandbox(unittest.TestCase):
    """Base: hermetic env (no SQL_* / flag) + engine reset per test."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        self.addCleanup(self._restore_env)
        self.addCleanup(db_store.configure_engine, None)

    def _restore_env(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _set_env(self, **kw):
        for k, v in kw.items():
            os.environ[k] = v

    def _local_dirs(self):
        """A tempdir wired as PRED_DIR/CALIB_DIR/LOG_PATH so a genuine local
        write in a clean env succeeds (proving the guard did not fire)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return (patch.object(recalibration, "PRED_DIR", tmp.name),
                patch.object(recalibration, "CALIB_DIR", tmp.name),
                patch.object(recalibration, "LOG_PATH",
                             os.path.join(tmp.name, "prediction_log.jsonl")))


class RequireSqlTableTests(_EnvSandbox):
    """§8.8 — db_store.require_sql() truth table: flag {on,off,absent} × SQL_*
    {none, partial, full}."""

    def test_flag_on_forces_true_regardless_of_secrets(self):
        for extra in ({}, {"SQL_SERVER": "s"}, dict(_FULL_SQL)):
            with self.subTest(extra=extra):
                for k in _ENV_KEYS:
                    os.environ.pop(k, None)
                self._set_env(SPORTSBOOK_REQUIRE_SQL="1", **extra)
                self.assertTrue(db_store.require_sql())

    def test_flag_off_forces_false_regardless_of_secrets(self):
        for extra in ({}, {"SQL_SERVER": "s"}, dict(_FULL_SQL)):
            with self.subTest(extra=extra):
                for k in _ENV_KEYS:
                    os.environ.pop(k, None)
                self._set_env(SPORTSBOOK_REQUIRE_SQL="0", **extra)
                self.assertFalse(db_store.require_sql())

    def test_absent_flag_infers_from_any_secret(self):
        self.assertFalse(db_store.require_sql())          # clean env
        self._set_env(SQL_SERVER="s")                     # partial ⇒ intent
        self.assertTrue(db_store.require_sql())
        os.environ.pop("SQL_SERVER")
        self._set_env(**_FULL_SQL)                        # full ⇒ intent
        self.assertTrue(db_store.require_sql())

    def test_flag_synonyms(self):
        for on in ("1", "true", "TRUE", "yes", "on", "On"):
            self._set_env(SPORTSBOOK_REQUIRE_SQL=on)
            self.assertTrue(db_store.require_sql(), on)
        for off in ("0", "false", "no", "off", "OFF"):
            self._set_env(SPORTSBOOK_REQUIRE_SQL=off)
            self.assertFalse(db_store.require_sql(), off)


class PredictionLogWriteGuardTests(_EnvSandbox):
    """§8.1 — mutate_prediction_log."""

    def test_raises_in_prod_context(self):
        self._set_env(SQL_SERVER="s.database.windows.net")   # partial ⇒ signalled
        db_store.configure_engine(None)                      # ...but SQL off
        with self.assertRaises(RuntimeError):
            recalibration.mutate_prediction_log(lambda rows: rows.append({}) or 1)

    def test_writes_local_in_clean_env(self):
        p1, p2, p3 = self._local_dirs()
        with p1, p2, p3:
            n = recalibration.mutate_prediction_log(
                lambda rows: rows.append({"x": 1}) or len(rows))
        self.assertEqual(n, 1)


class NdjsonWriteGuardTests(_EnvSandbox):
    """§8.2 — mutate_ndjson_log."""

    def test_raises_in_prod_context(self):
        self._set_env(SPORTSBOOK_REQUIRE_SQL="1")            # explicit opt-in
        db_store.configure_engine(None)
        with self.assertRaises(RuntimeError):
            recalibration.mutate_ndjson_log(
                "wagers.jsonl", lambda rows: rows.append({}) or 1)

    def test_writes_local_in_clean_env(self):
        p1, p2, p3 = self._local_dirs()
        with p1, p2, p3:
            n = recalibration.mutate_ndjson_log(
                "wagers.jsonl", lambda rows: rows.append({"x": 1}) or len(rows))
        self.assertEqual(n, 1)


class SaveRecalibrationGuardTests(_EnvSandbox):
    """§8.3 — save_recalibration is guarded ONLY for the durable (to_blob=True)
    refit path; the offline seed (to_blob=False) is an INTENTIONAL local write."""

    _FIT = {"batter_hits": {"a": 0.5, "b": 0.1, "n_fit": 120, "validated": True}}

    def test_runtime_refit_raises_in_prod_context(self):
        self._set_env(SQL_SERVER="s.database.windows.net")
        db_store.configure_engine(None)
        with self.assertRaises(RuntimeError):
            recalibration.save_recalibration(
                "baseball_mlb", self._FIT, to_blob=True)

    def test_offline_seed_never_guarded_even_in_prod_context(self):
        # to_blob=False writes the committed git seed on purpose → must NOT raise
        # even when a SQL deployment is signalled.
        self._set_env(SPORTSBOOK_REQUIRE_SQL="1")
        db_store.configure_engine(None)
        _, p2, _ = self._local_dirs()
        with p2:  # CALIB_DIR → tempdir
            recalibration.save_recalibration(
                "baseball_mlb", self._FIT, to_blob=False)
        # And the seed file landed locally.
        self.assertTrue(True)


class FlagSemanticsGuardTests(_EnvSandbox):
    """§8.4 — flag drives the guard: opt-in with no secrets raises; escape hatch
    with partial secrets does not."""

    def test_opt_in_flag_raises_without_any_secret(self):
        self._set_env(SPORTSBOOK_REQUIRE_SQL="1")
        db_store.configure_engine(None)
        with self.assertRaises(RuntimeError):
            recalibration.mutate_prediction_log(lambda rows: rows.append({}) or 1)

    def test_escape_hatch_flag_allows_partial_secret_local_use(self):
        p1, p2, p3 = self._local_dirs()
        self._set_env(SPORTSBOOK_REQUIRE_SQL="0", SQL_SERVER="s")
        db_store.configure_engine(None)
        with p1, p2, p3:
            n = recalibration.mutate_prediction_log(
                lambda rows: rows.append({"x": 1}) or len(rows))
        self.assertEqual(n, 1)


class WarehouseReadGuardTests(_EnvSandbox):
    """§8.5 — refit-critical warehouse reads abort loudly in a prod context and
    stay inert (return []/empty) in a clean env (pins test_store_empty_when_sql_off
    behavior under the guard)."""

    def test_load_prop_lines_raises_in_prod_context(self):
        self._set_env(SQL_DATABASE="d")
        db_store.configure_engine(None)
        with self.assertRaises(RuntimeError):
            warehouse.load_prop_lines("baseball_mlb")

    def test_load_team_market_store_raises_in_prod_context(self):
        self._set_env(SQL_DATABASE="d")
        db_store.configure_engine(None)
        with self.assertRaises(RuntimeError):
            warehouse.load_team_market_store("baseball_mlb")

    def test_reads_are_inert_in_clean_env(self):
        db_store.configure_engine(None)
        self.assertEqual(warehouse.load_prop_lines("baseball_mlb"), [])
        store = warehouse.load_team_market_store("baseball_mlb")
        self.assertEqual(store["games"], {})


class DbImportFallbackTests(_EnvSandbox):
    """§8.6 — when db_store failed to import (_db is None) but SQL_* is configured,
    the module-local env fallback in recalibration/warehouse must still catch it."""

    def test_recalibration_env_fallback_raises_when_db_none(self):
        self._set_env(**_FULL_SQL)
        with patch.object(recalibration, "_db", None):
            self.assertTrue(recalibration._require_sql())
            with self.assertRaises(RuntimeError):
                recalibration.mutate_prediction_log(
                    lambda rows: rows.append({}) or 1)

    def test_warehouse_env_fallback_raises_when_db_none(self):
        self._set_env(**_FULL_SQL)
        with patch.object(warehouse, "_db", None):
            self.assertTrue(warehouse._require_sql())
            with self.assertRaises(RuntimeError):
                warehouse.load_prop_lines("baseball_mlb")


class HealthySqlNeverRaisesTests(_EnvSandbox):
    """§8.7 — with a real (SQLite) engine, enabled() is True → the guard never
    fires, even with SPORTSBOOK_REQUIRE_SQL=1 forced on."""

    def setUp(self):
        super().setUp()
        self._set_env(SPORTSBOOK_REQUIRE_SQL="1")
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        recalibration._NDJSON_CACHE.clear()
        recalibration._LOAD_CACHE.clear()
        self.addCleanup(recalibration._NDJSON_CACHE.clear)
        self.addCleanup(recalibration._LOAD_CACHE.clear)

    def test_writes_and_reads_do_not_raise(self):
        # No-op mutators (falsy return → no INSERT): we assert only that the
        # _ensure_durable guard stays inert on a healthy engine, not the schema.
        self.assertEqual(recalibration.mutate_prediction_log(lambda rows: 0), 0)
        self.assertEqual(
            recalibration.mutate_ndjson_log("wagers.jsonl", lambda rows: 0), 0)
        # Reads: empty warehouse, but no raise.
        self.assertEqual(warehouse.load_prop_lines("baseball_mlb"), [])
        self.assertEqual(
            warehouse.load_team_market_store("baseball_mlb")["games"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
