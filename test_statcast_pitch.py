"""Tests for the durable Azure-SQL Statcast raw-pitch store (savant_history).

Covers the SQL cutover of the previously-local per-day pitch cache: day-atomic
ingest + manifest, SQL-only load_days round-trip, the empty-vs-unfetched manifest
distinction, missing_days / ensure_days incremental gap-fill (capped), one-shot
file migration, and the schema-parity drift guard. Runs on in-memory SQLite; the
Savant network fetch is monkeypatched (no HTTP).
"""

import os
import tempfile
import unittest
from unittest import mock

import db_store
import savant_history as sh


def _row(day="2024-07-03", pitcher="657006", batter="592450", xwoba=0.35,
         xba=0.28, ptype="X"):
    return {"game_date": day, "pitcher": pitcher, "batter": batter,
            "p_throws": "R", "batting_team": "NYY", "stand": "R",
            "xwoba": xwoba, "xba": xba, "description": "hit_into_play",
            "type": ptype, "launch_speed": 95.1, "launch_speed_angle": 6,
            "launch_angle": 22.0, "bb_type": "line_drive"}


class _Backend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        sh.create_all()

    def tearDown(self):
        db_store.configure_engine(None)


class IngestLoadTests(_Backend, unittest.TestCase):
    def test_ingest_and_load_roundtrip(self):
        rows = [_row(batter="1"), _row(batter="2"), _row(batter="3")]
        n = sh.ingest_day("2024-07-03", rows)
        self.assertEqual(n, 3)
        got = sh.load_days("2024-07-03", "2024-07-03")
        self.assertEqual(len(got), 3)
        # Same dict shape as fetch_statcast_day emits (every PITCH_COLS key present).
        self.assertEqual(set(got[0].keys()), set(sh.PITCH_COLS))
        self.assertEqual({r["batter"] for r in got}, {"1", "2", "3"})
        self.assertAlmostEqual(got[0]["xwoba"], 0.35)
        self.assertEqual(got[0]["type"], "X")

    def test_ingest_is_day_atomic_replace(self):
        sh.ingest_day("2024-07-03", [_row(batter="a"), _row(batter="b")])
        sh.ingest_day("2024-07-03", [_row(batter="c")])   # replace, not append
        got = sh.load_days("2024-07-03", "2024-07-03")
        self.assertEqual([r["batter"] for r in got], ["c"])

    def test_load_days_range_filter_and_order(self):
        for d in ("2024-07-05", "2024-07-03", "2024-07-04"):
            sh.ingest_day(d, [_row(day=d)])
        got = sh.load_days("2024-07-03", "2024-07-04")
        self.assertEqual([r["game_date"] for r in got],
                         ["2024-07-03", "2024-07-04"])   # in-range, ordered

    def test_empty_day_manifested_and_not_missing(self):
        # An ingested EMPTY day (offseason) must be distinguishable from unfetched:
        # load returns nothing, but it is NOT reported missing (manifest present).
        sh.ingest_day("2024-01-15", [])
        self.assertEqual(sh.load_days("2024-01-15", "2024-01-15"), [])
        self.assertNotIn("2024-01-15", sh.missing_days("2024-01-15", "2024-01-15"))

    def test_missing_days_reports_gaps(self):
        sh.ingest_day("2024-07-03", [_row()])
        sh.ingest_day("2024-07-05", [_row(day="2024-07-05")])
        miss = sh.missing_days("2024-07-03", "2024-07-06")
        self.assertEqual(miss, ["2024-07-04", "2024-07-06"])   # gaps only, oldest-first

    def test_missing_days_skips_offseason(self):
        # Off-season days (mid-Nov..mid-Feb) are NEVER reported missing → never
        # re-fetched from Savant every run. In-season un-ingested days still are.
        miss = sh.missing_days("2023-12-20", "2024-03-02")
        self.assertNotIn("2024-01-15", miss)   # deep off-season
        self.assertNotIn("2024-02-14", miss)   # last off-season day
        self.assertNotIn("2023-12-25", miss)
        self.assertIn("2024-02-15", miss)      # spring-training window opens
        self.assertIn("2024-03-01", miss)      # in-season, un-ingested
        self.assertTrue(sh._in_mlb_season(sh._date.fromisoformat("2024-11-10")))
        self.assertFalse(sh._in_mlb_season(sh._date.fromisoformat("2024-11-11")))


class SqlOffTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine(None)

    def test_sql_off_is_a_noop(self):
        self.assertFalse(sh.enabled())
        self.assertEqual(sh.load_days("2024-07-03", "2024-07-03"), [])
        self.assertIs(sh.ingest_day("2024-07-03", [_row()]), False)
        self.assertEqual(sh.ingested_days("2024-07-03", "2024-07-03"), set())
        self.assertEqual(sh.ensure_days("2024-07-03", "2024-07-03", verbose=False),
                         (0, 0))


class EnsureDaysTests(_Backend, unittest.TestCase):
    def test_ensure_fetches_only_missing(self):
        sh.ingest_day("2024-07-03", [_row()])            # already present

        def _fake_fetch(day, force=False):
            sh.ingest_day(day, [_row(day=day)])          # stand-in for the Savant pull
            return []

        with mock.patch.object(sh, "fetch_statcast_day",
                               side_effect=_fake_fetch) as ff:
            n_new, n_missing = sh.ensure_days(
                "2024-07-03", "2024-07-05", sleep=0, verbose=False)
        self.assertEqual((n_new, n_missing), (2, 2))     # 07-04 + 07-05 fetched
        self.assertEqual(sorted(c.args[0] for c in ff.call_args_list),
                         ["2024-07-04", "2024-07-05"])
        self.assertEqual(sh.missing_days("2024-07-03", "2024-07-05"), [])

    def test_ensure_cap_fetches_most_recent_and_reports_remainder(self):
        def _fake_fetch(day, force=False):
            sh.ingest_day(day, [_row(day=day)])
            return []

        with mock.patch.object(sh, "fetch_statcast_day", side_effect=_fake_fetch) as ff:
            n_new, n_missing = sh.ensure_days(
                "2024-07-01", "2024-07-05", cap=2, sleep=0, verbose=False)
        self.assertEqual((n_new, n_missing), (2, 5))     # only 2 of 5 pulled
        # The most-RECENT cap days (the fresh gap) are the ones fetched.
        self.assertEqual(sorted(c.args[0] for c in ff.call_args_list),
                         ["2024-07-04", "2024-07-05"])


class MigrationTests(_Backend, unittest.TestCase):
    def test_migrate_files_to_sql_ingests_existing_files(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sh, "CACHE_DIR", d):
                with open(os.path.join(d, "2024-07-03.json"), "w") as f:
                    json.dump([_row(batter="x"), _row(batter="y")], f)
                # 07-04 has no file → skipped (stays missing for ensure_days later).
                n_days, n_rows = sh.migrate_files_to_sql(
                    "2024-07-03", "2024-07-04", verbose=False)
        self.assertEqual((n_days, n_rows), (1, 2))
        self.assertEqual(len(sh.load_days("2024-07-03", "2024-07-03")), 2)
        self.assertIn("2024-07-04", sh.missing_days("2024-07-03", "2024-07-04"))


class SchemaParityTests(_Backend, unittest.TestCase):
    def test_pitch_cols_match_table_data_columns(self):
        # PITCH_COLS (the emitted/loaded dict shape) must equal the statcast_pitch
        # DATA columns (all but the surrogate id), same order.
        data_cols = [c.name for c in sh.statcast_pitch.columns if c.name != "id"]
        self.assertEqual(list(sh.PITCH_COLS), data_cols)


if __name__ == "__main__":
    unittest.main()
