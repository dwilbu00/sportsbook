"""Tests for pitcher_asof.py — the per-(pitcher, game-date) as-of feature store."""
import unittest
from unittest.mock import patch

import db_store
import pitcher_asof as pa


class WarehouseAsofCurveTests(unittest.TestCase):
    def test_strict_before_and_cumulative(self):
        games = [("2024-04-01", 18, 2.0, 6.0),   # 6 IP, 2 ER, 6 K
                 ("2024-04-07", 15, 4.0, 5.0)]
        curve = pa._warehouse_asof_curve(games)
        # First start: nothing strictly before -> None.
        self.assertIsNone(curve["2024-04-01"])
        # Second start: only the 04-01 line (strict <).
        c = curve["2024-04-07"]
        self.assertAlmostEqual(c["ip"], 6.0)
        self.assertAlmostEqual(c["era"], 3.0)      # (2/6)*9
        self.assertAlmostEqual(c["k9"], 9.0)       # (6/6)*9
        self.assertAlmostEqual(c["avg_ip"], 6.0)
        self.assertEqual(c["games"], 1)

    def test_same_date_doubleheader_shares_pre_date_line(self):
        # Two appearances on 04-07: BOTH exclude same-date games (strict <), so both
        # collapse to one distinct-date entry = the pre-04-07 (04-01) line only.
        games = [("2024-04-01", 18, 3.0, 6.0),
                 ("2024-04-07", 9, 1.0, 2.0),
                 ("2024-04-07", 12, 2.0, 3.0)]
        curve = pa._warehouse_asof_curve(games)
        self.assertEqual(set(curve), {"2024-04-01", "2024-04-07"})
        self.assertEqual(curve["2024-04-07"]["games"], 1)   # only 04-01, not the DH
        self.assertAlmostEqual(curve["2024-04-07"]["ip"], 6.0)


class StatcastAsOfTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"pitcher": "1", "game_date": "2024-04-01", "xwoba": 0.300, "type": "X"},
            {"pitcher": "1", "game_date": "2024-04-05", "xwoba": 0.400, "type": "X"},
            {"pitcher": "1", "game_date": "2024-04-10", "xwoba": 0.200, "type": "X"},
            {"pitcher": "2", "game_date": "2024-04-01", "xwoba": 0.500, "type": "X"},
            {"pitcher": "1", "game_date": "2024-04-05", "xwoba": None,   # not a BBE
             "type": "S", "description": "swinging_strike"},
        ]
        self.sc = pa._StatcastAsOf(self.rows)

    def test_strict_before(self):
        self.assertIsNone(self.sc.asof("1", "2024-04-01"))
        r = self.sc.asof("1", "2024-04-05")        # only 04-01 BBE
        self.assertAlmostEqual(r["xwobacon"], 0.300); self.assertEqual(r["n_bbe"], 1)
        r = self.sc.asof("1", "2024-04-10")        # 04-01 + 04-05 BBE
        self.assertAlmostEqual(r["xwobacon"], 0.350); self.assertEqual(r["n_bbe"], 2)
        r = self.sc.asof("1", "2024-04-11")        # all three BBE
        self.assertAlmostEqual(r["xwobacon"], 0.300); self.assertEqual(r["n_bbe"], 3)

    def test_unknown_pitcher(self):
        self.assertIsNone(self.sc.asof("999", "2024-04-05"))


class BuildAndReadSqliteTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        pa.create_all()

    def tearDown(self):
        db_store.configure_engine(None)

    def test_build_then_read_mixed_key_rows_with_rates(self):
        # HETEROGENEOUS rows in one bulk insert (the executemany bug): warehouse
        # games start 04-05, but statcast pitches exist 04-01..03 (before the first
        # warehouse game). So the 04-05 row has statcast but NO warehouse line, the
        # 04-11 row has BOTH. Every row must be full-keyed. Also verifies the
        # statcast rate columns (whiff/barrel/gb/xwobacon) populate.
        pit_idx = {"1": [("2024-04-05", 18, 2.0, 6.0),
                         ("2024-04-11", 15, 4.0, 5.0)]}
        savant_rows = [   # two BBE + one whiff, all before both warehouse dates
            {"pitcher": "1", "game_date": "2024-04-01", "xwoba": 0.35,
             "description": "hit_into_play", "type": "X", "launch_speed": 100.0,
             "launch_speed_angle": 6, "bb_type": "fly_ball"},          # barrel + hard-hit
            {"pitcher": "1", "game_date": "2024-04-02", "xwoba": 0.30,
             "description": "hit_into_play", "type": "X", "launch_speed": 80.0,
             "launch_speed_angle": 3, "bb_type": "ground_ball"},       # GB, soft
            {"pitcher": "1", "game_date": "2024-04-03", "xwoba": None,
             "description": "swinging_strike", "type": "S"},           # whiff
        ]
        with patch("mlb_warehouse._pitcher_game_index", return_value=pit_idx), \
             patch("savant_history.load_days", return_value=savant_rows):
            n = pa.build_season(2024, verbose=False)
        self.assertEqual(n, 2)                             # both SP rows written

        r5 = pa.asof_pitcher_features("1", "2024-04-05")   # statcast only, no wh line
        self.assertIsNone(r5["era"])
        self.assertAlmostEqual(r5["xwobacon"], 0.325)      # mean(0.35, 0.30) over BBE
        r11 = pa.asof_pitcher_features("1", "2024-04-11")   # both
        self.assertAlmostEqual(r11["era"], 3.0)            # from the 04-05 game
        self.assertEqual(r11["games"], 1)
        self.assertAlmostEqual(r11["xwobacon"], 0.325)
        self.assertEqual(r11["n_bbe"], 2)
        self.assertEqual(r11["n_pitches"], 3)
        self.assertAlmostEqual(r11["whiff_pct"], 1 / 3)    # 1 whiff / 3 swings
        self.assertAlmostEqual(r11["barrel_pct"], 0.5)     # 1 barrel / 2 BBE
        self.assertAlmostEqual(r11["hard_hit_pct"], 0.5)   # 1 hard-hit / 2 BBE
        self.assertAlmostEqual(r11["gb_pct"], 0.5)         # 1 GB / 2 BBE
        self.assertEqual(r11["role"], "SP")
        self.assertIsNone(pa.asof_pitcher_features("1", "2024-04-01"))

    def test_rebuild_is_idempotent(self):
        pit_idx = {"1": [("2024-04-01", 18, 2.0, 6.0),
                         ("2024-04-07", 15, 4.0, 5.0)]}
        with patch("mlb_warehouse._pitcher_game_index", return_value=pit_idx), \
             patch("savant_history.load_days", return_value=[]):
            n1 = pa.build_season(2024, verbose=False)
            n2 = pa.build_season(2024, verbose=False)   # replace-write, no dup
        self.assertEqual(n1, n2)
        from sqlalchemy import select, func
        with db_store.get_engine().connect() as c:
            total = c.execute(select(func.count()).select_from(
                pa.pitcher_asof_daily)).scalar()
        self.assertEqual(total, n1)


class SchemaParityTests(unittest.TestCase):
    def test_cols_spec_matches_table(self):
        # _COLS (the drift SPEC) must equal the Table's columns in order — mirrors
        # statcast_asof/db_store SchemaParity; guards sql/schema.sql from drifting.
        self.assertEqual(list(pa._COLS),
                         [c.name for c in pa.pitcher_asof_daily.columns])


class GetOrFillSqliteTests(unittest.TestCase):
    def setUp(self):
        import savant_history as sh
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        pa.create_all()
        sh.create_all()
        with db_store.get_engine().begin() as c:
            c.execute(sh.statcast_pitch.insert(), [
                {"game_date": "2024-04-01", "pitcher": "1", "xwoba": 0.30},
                {"game_date": "2024-04-03", "pitcher": "1", "xwoba": 0.40},
                {"game_date": "2024-04-09", "pitcher": "1", "xwoba": 0.99},  # after as_of
            ])

    def tearDown(self):
        db_store.configure_engine(None)

    def test_fill_on_miss_then_hit(self):
        wh = {"era": 3.0, "ip": 6.0, "k": 6.0, "games": 1, "avg_ip": 6.0}
        with patch("mlb_warehouse.asof_pitcher_stats", return_value=wh) as m:
            # Miss -> compute + persist.
            row = pa.get_or_fill("1", "2024-04-07")
            self.assertAlmostEqual(row["era"], 3.0)
            self.assertAlmostEqual(row["k9"], 9.0)          # (6/6)*9
            self.assertAlmostEqual(row["xwobacon"], 0.35)   # mean(0.30,0.40), 04-09 excluded
            self.assertEqual(row["n_bbe"], 2)
            self.assertEqual(m.call_count, 1)
            # Hit -> served from the table, NO recompute (asof_pitcher_stats not called).
            row2 = pa.get_or_fill("1", "2024-04-07")
            self.assertAlmostEqual(row2["era"], 3.0)
            self.assertEqual(m.call_count, 1)

        from sqlalchemy import select, func
        with db_store.get_engine().connect() as c:
            total = c.execute(select(func.count()).select_from(
                pa.pitcher_asof_daily)).scalar()
        self.assertEqual(total, 1)                           # persisted exactly once

    def test_no_asof_returns_none(self):
        # A first-ever start (no prior warehouse line, no prior BBE) -> None, no row.
        with patch("mlb_warehouse.asof_pitcher_stats", return_value=None):
            self.assertIsNone(pa.get_or_fill("1", "2024-04-01"))


if __name__ == "__main__":
    unittest.main()
