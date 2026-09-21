"""Prediction provenance stamp (audit F28 P1).

Run: PYTHONIOENCODING=utf-8 python -m unittest test_prediction_provenance -v
"""
import os
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import db_store
import prediction_provenance as pp
import recalibration


class ContextBuildTests(unittest.TestCase):
    def test_mlb_season_from_commence_no_week(self):
        c = pp.build_context("baseball_mlb", "2026-07-16T18:00:00Z", method="C")
        self.assertEqual(c["event_season"], 2026)
        self.assertIsNone(c["event_week"])
        self.assertEqual(c["method"], "C")
        self.assertTrue(c["context_fingerprint"])

    def test_fingerprint_excludes_serve_time(self):
        a = pp.build_context("baseball_mlb", "2026-07-16T18:00:00Z", method="C")
        b = pp.build_context("baseball_mlb", "2026-07-16T18:00:00Z", method="C")
        self.assertNotEqual(a["as_of"], "")            # both stamped
        self.assertEqual(a["context_fingerprint"], b["context_fingerprint"])

    def test_fingerprint_changes_with_method(self):
        a = pp.build_context("baseball_mlb", "2026-07-16T18:00:00Z", method="C")
        b = pp.build_context("baseball_mlb", "2026-07-16T18:00:00Z", method="E")
        self.assertNotEqual(a["context_fingerprint"], b["context_fingerprint"])

    def test_nfl_season_year_derivation(self):
        # A January game belongs to the prior NFL season.
        self.assertEqual(pp.event_season_week(
            "americanfootball_nfl", "2025-01-05T18:00:00Z")[0], 2024)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch("warehouse_mirror.enabled", return_value=False))
        recalibration._NDJSON_CACHE.clear()
        db_store.configure_engine("sqlite://")
        db_store.create_all()

    def tearDown(self):
        db_store.configure_engine(None)
        recalibration._NDJSON_CACHE.clear()
        self.stack.close()

    def test_context_persists_to_prediction_log(self):
        ctx = pp.build_context("baseball_mlb", "2026-07-16T18:00:00Z", method="C",
                               reference_book_policy="two_way_devig")
        recalibration.log_prediction(
            "baseball_mlb", "batter_hits", "Star", "2026-07-16", 1.5, 0.6,
            final_prob=0.58, commence_time="2026-07-16T18:00:00Z", event_id="e1",
            context=ctx, write=True)
        r = db_store.read_rows("prediction_log")[0]
        self.assertEqual(r["event_season"], 2026)
        self.assertEqual(r["method"], "C")
        self.assertTrue(r["calibration_hash"])
        self.assertTrue(r["model_schema"])
        self.assertEqual(r["context_fingerprint"], ctx["context_fingerprint"])
        self.assertTrue(r["as_of"])

    def test_pre_ddl_schema_tolerated(self):
        # The code can ship before the owner runs the ADD COLUMN DDL: simulate a live
        # table WITHOUT the provenance columns; logging must still succeed (provenance
        # skipped) rather than erroring the write.
        prov = {"event_season", "event_week", "as_of", "calibration_hash",
                "model_schema", "method", "reference_book_policy", "context_fingerprint"}
        db_store._LIVE_COLS["prediction_log"] = {
            c.name for c in db_store.prediction_log.columns} - prov
        ctx = pp.build_context("baseball_mlb", "2026-07-16T18:00:00Z", method="C")
        recalibration.log_prediction(
            "baseball_mlb", "batter_hits", "PreDDL", "2026-07-16", 1.5, 0.6,
            final_prob=0.58, commence_time="2026-07-16T18:00:00Z", event_id="e9",
            context=ctx, write=True)
        r = [x for x in db_store.read_rows("prediction_log")
             if x["player"] == "PreDDL"][0]
        self.assertEqual(r["final_prob"], 0.58)        # core write succeeded
        self.assertIsNone(r.get("event_season"))       # provenance skipped, no crash

    def test_legacy_row_without_context_is_null_provenance(self):
        recalibration.log_prediction(
            "baseball_mlb", "batter_hits", "NoCtx", "2026-07-16", 1.5, 0.6,
            commence_time="2026-07-16T18:00:00Z", event_id="e2", write=True)
        r = [x for x in db_store.read_rows("prediction_log") if x["player"] == "NoCtx"][0]
        self.assertIsNone(r["event_season"])
        self.assertIsNone(r["context_fingerprint"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
