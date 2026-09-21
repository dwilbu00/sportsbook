"""Durable bonus-store regression tests (audit F16): stable ids + safe seeding.

Hermetic: fresh in-memory SQLite per test, mirror disabled, NDJSON cache cleared.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_bonus_store -v
"""
import json
import os
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import bonus as bl
import bonus_store
import db_store
import recalibration


class _StoreTest(unittest.TestCase):
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

    def _write_raw(self, value):
        payload = json.dumps(value)

        def up(rows):
            for r in rows:
                if r.get("setting_key") == "active_bonuses":
                    r["setting_value"] = payload
                    return 1
            rows.append({"setting_key": "active_bonuses", "setting_value": payload,
                         "updated_at": "t"})
            return 1
        recalibration.mutate_ndjson_log("app_settings.jsonl", up)


class F16IdentityTests(_StoreTest):
    def test_new_bonus_id_is_unique(self):
        self.assertNotEqual(bl.new_bonus_id(), bl.new_bonus_id())

    def test_round_trip_preserves_distinct_ids_for_same_label(self):
        b1 = bl.Bonus("sgp", .5, label="Promo", bonus_id=bl.new_bonus_id())
        b2 = bl.Bonus("parlay", .5, label="Promo", bonus_id=bl.new_bonus_id())
        bonus_store.save_bonuses([b1, b2])
        loaded = bonus_store.load_bonuses()
        self.assertEqual({x.bonus_id for x in loaded}, {b1.bonus_id, b2.bonus_id})

    def test_legacy_entry_gets_id_assigned_and_persisted(self):
        self._write_raw([{"bet_type": "sgp", "boost_pct": .5, "label": "Legacy"}])
        first = bonus_store.load_bonuses()
        self.assertTrue(first[0].bonus_id)
        # persisted: a second load returns the SAME id (not a fresh one each time)
        self.assertEqual(bonus_store.load_bonuses()[0].bonus_id, first[0].bonus_id)

    def test_failed_read_returns_ephemeral_without_persisting(self):
        # A committed real promo, then a forced read failure: load must NOT persist
        # example promos over it (never clobber unreadable durable data). [F16]
        real = bl.Bonus("sgp", .5, label="Real", bonus_id=bl.new_bonus_id())
        bonus_store.save_bonuses([real])
        orig = recalibration._read_ndjson_blob

        def boom(filename, *a, **k):
            if filename == bonus_store._SETTINGS_FILE:
                raise OSError("synthetic store read failure")
            return orig(filename, *a, **k)

        with patch.object(recalibration, "_read_ndjson_blob", side_effect=boom):
            got = bonus_store.load_bonuses()          # ephemeral seeds, no write
        self.assertTrue(got)                          # returns something usable
        # the real promo is intact once the store is readable again
        after = bonus_store.load_bonuses()
        self.assertEqual([b.label for b in after], ["Real"])
        self.assertEqual(after[0].bonus_id, real.bonus_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
