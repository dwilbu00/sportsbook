"""Tests for the clean-slate archive-then-reset of the app's forward-tracking +
bet ledger (archive_app_data.py). Non-destructive: archive-first, verify, then
delete; all four target tables cleared; other app_settings preserved."""
import io
import os
import json
import glob
import tempfile
import contextlib
import unittest

from sqlalchemy import insert, select, func

import db_store
import archive_app_data as aad


class ArchiveAppDataTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        self.eng = db_store.get_engine()
        # archive_and_reset's _engine() calls promote_secrets_from_toml(), which
        # leaks SQL_* into os.environ and breaks later SQL-off tests. Point it at
        # the test engine instead (same pattern as test_odds_provenance).
        self._orig_engine = aad._engine
        aad._engine = db_store.get_engine
        self.addCleanup(lambda: setattr(aad, "_engine", self._orig_engine))
        self._arch = tempfile.mkdtemp()
        self._seed_rows()

    def _seed_rows(self):
        with self.eng.begin() as c:
            c.execute(insert(db_store.prediction_log).values(
                sport_key="baseball_mlb", event_key="E1", prop_key="batter_hits",
                player="Batter A", line=0.5, player_key="mlb:1", resolved=True))
            c.execute(insert(db_store.market_prediction_log).values(
                sport_key="baseball_mlb", event_key="E1", bet_type="moneyline",
                side="home", resolved=True))
            c.execute(insert(db_store.wagers).values(
                wager_id="W1", sport_key="baseball_mlb", status="won", stake=10.0))
            c.execute(insert(db_store.bankroll_ledger).values(
                txn_id="adj:1", txn_type="adjustment", amount=100.0))
            c.execute(insert(db_store.bankroll_ledger).values(
                txn_id="bet:W1", txn_type="bet", amount=9.09, wager_id="W1"))
            # a NON-target app setting that must survive the reset
            c.execute(insert(db_store.app_settings).values(
                setting_key="kelly_fraction", setting_value="0.5", updated_at="x"))

    def _count(self, t):
        with self.eng.connect() as c:
            return c.execute(select(func.count()).select_from(t)).scalar()

    def _setting(self, key):
        s = db_store.app_settings
        with self.eng.connect() as c:
            return c.execute(select(s.c.setting_value)
                             .where(s.c.setting_key == key)).scalar()

    def test_dry_run_deletes_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()):
            aad.archive_and_reset(apply=False, yes=False, archive_dir=self._arch)
        self.assertEqual(self._count(db_store.prediction_log), 1)
        self.assertEqual(self._count(db_store.wagers), 1)
        self.assertEqual(self._count(db_store.bankroll_ledger), 2)
        self.assertEqual(glob.glob(os.path.join(self._arch, "*.json")), [])

    def test_apply_without_yes_aborts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                aad.archive_and_reset(apply=True, yes=False, archive_dir=self._arch)
        self.assertEqual(self._count(db_store.wagers), 1)   # untouched

    def test_apply_archives_then_clears_all_four(self):
        with contextlib.redirect_stdout(io.StringIO()):
            aad.archive_and_reset(apply=True, yes=True, archive_dir=self._arch)
        # all four target tables emptied
        self.assertEqual(self._count(db_store.prediction_log), 0)
        self.assertEqual(self._count(db_store.market_prediction_log), 0)
        self.assertEqual(self._count(db_store.wagers), 0)
        self.assertEqual(self._count(db_store.bankroll_ledger), 0)  # balance -> 0
        # non-target app setting preserved
        self.assertEqual(self._setting("kelly_fraction"), "0.5")
        # epoch marker written with the per-table counts
        epoch = json.loads(self._setting("app_data_epoch"))
        self.assertEqual(epoch["counts"]["wagers"], 1)
        self.assertEqual(epoch["counts"]["bankroll_ledger"], 2)

    def test_archive_file_round_trips_all_rows(self):
        with contextlib.redirect_stdout(io.StringIO()):
            aad.archive_and_reset(apply=True, yes=True, archive_dir=self._arch)
        files = glob.glob(os.path.join(self._arch, "app_data_epoch_*.json"))
        self.assertEqual(len(files), 1)
        with open(files[0], encoding="utf-8") as f:
            payload = json.load(f)
        # every cleared row is recoverable from the archive
        self.assertEqual(len(payload["tables"]["prediction_log"]), 1)
        self.assertEqual(payload["tables"]["wagers"][0]["wager_id"], "W1")
        self.assertEqual(len(payload["tables"]["bankroll_ledger"]), 2)

    def test_idempotent_second_run_is_noop(self):
        with contextlib.redirect_stdout(io.StringIO()):
            aad.archive_and_reset(apply=True, yes=True, archive_dir=self._arch)
            aad.archive_and_reset(apply=True, yes=True, archive_dir=self._arch)
        # only the first run wrote an archive (second saw empty tables -> returned)
        self.assertEqual(
            len(glob.glob(os.path.join(self._arch, "*.json"))), 1)

    def test_subset_tables_only_clears_named(self):
        with contextlib.redirect_stdout(io.StringIO()):
            aad.archive_and_reset(apply=True, yes=True, tables=["wagers"],
                                  archive_dir=self._arch)
        self.assertEqual(self._count(db_store.wagers), 0)
        self.assertEqual(self._count(db_store.prediction_log), 1)   # untouched
        self.assertEqual(self._count(db_store.bankroll_ledger), 2)  # untouched


if __name__ == "__main__":
    unittest.main()
