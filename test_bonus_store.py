"""bonus_store durability — the fix for "bonuses don't survive a reload".

Root cause: app_settings.setting_value was String(256); the active-bonus JSON exceeds
256 chars, so SQL Server rejected the write and the list reverted to seeds on reload.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_bonus_store -v
"""
import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

import bonus as bonuslib
import bonus_store
import db_store


class SchemaGuardTests(unittest.TestCase):
    def test_setting_value_is_unbounded(self):
        # The active-bonus JSON exceeds 256 chars -> the column must be unbounded
        # (Text / NVARCHAR(max)), never a bounded String, or writes truncate/fail.
        col = db_store.app_settings.c.setting_value
        self.assertIsNone(getattr(col.type, "length", None),
                          "setting_value must be unbounded Text, not a bounded String")


class RoundTripTests(unittest.TestCase):
    def _big_list(self, n=8):
        return [bonuslib.Bonus(
                    bet_type="parlay", boost_pct=0.5, min_odds_leg=-300.0,
                    min_odds_overall=1000.0, min_legs=4, max_wager=25.0, min_wager=1.0,
                    book="DraftKings",
                    markets=("player_pass_yds", "player_rush_yds", "player_receptions"),
                    bonus_id=bonuslib.new_bonus_id(),
                    label=f"Promo #{i} — a reasonably long descriptive label here")
                for i in range(n)]

    def test_large_bonus_list_round_trips(self):
        big = self._big_list()
        payload = json.dumps([{k: asdict(b)[k] for k in bonus_store._FIELDS} for b in big])
        self.assertGreater(len(payload), 256)   # exactly the size the old column rejected

        store = {"rows": []}                     # in-memory app_settings KV

        def _read(filename, use_cache=False, where=None):
            return [dict(r) for r in store["rows"]], None

        def _mutate(filename, mutator, **k):
            return mutator(store["rows"])

        with patch("recalibration._read_ndjson_blob", side_effect=_read), \
                patch("recalibration.mutate_ndjson_log", side_effect=_mutate):
            wrote = bonus_store.save_bonuses(big)
            loaded = bonus_store.load_bonuses()

        self.assertEqual(wrote, 1)
        self.assertEqual(len(loaded), len(big))
        self.assertEqual([b.bonus_id for b in loaded], [b.bonus_id for b in big])
        self.assertEqual(loaded[0].markets, big[0].markets)


if __name__ == "__main__":
    unittest.main(verbosity=2)
