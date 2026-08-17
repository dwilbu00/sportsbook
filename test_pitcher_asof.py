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


class XwobaconAsOfTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"pitcher": "1", "game_date": "2024-04-01", "xwoba": 0.300},
            {"pitcher": "1", "game_date": "2024-04-05", "xwoba": 0.400},
            {"pitcher": "1", "game_date": "2024-04-10", "xwoba": 0.200},
            {"pitcher": "2", "game_date": "2024-04-01", "xwoba": 0.500},
            {"pitcher": "1", "game_date": "2024-04-05", "xwoba": None},   # ignored
        ]
        self.xw = pa._XwobaconAsOf(self.rows)

    def test_strict_before(self):
        self.assertEqual(self.xw.asof("1", "2024-04-01"), (None, 0))
        m, n = self.xw.asof("1", "2024-04-05")     # only 04-01 BBE
        self.assertAlmostEqual(m, 0.300); self.assertEqual(n, 1)
        m, n = self.xw.asof("1", "2024-04-10")     # 04-01 + 04-05
        self.assertAlmostEqual(m, 0.350); self.assertEqual(n, 2)
        m, n = self.xw.asof("1", "2024-04-11")     # all three
        self.assertAlmostEqual(m, 0.300); self.assertEqual(n, 3)

    def test_unknown_pitcher(self):
        self.assertEqual(self.xw.asof("999", "2024-04-05"), (None, 0))


class BuildAndReadSqliteTests(unittest.TestCase):
    def setUp(self):
        db_store.configure_engine("sqlite://")
        db_store.create_all()
        pa.create_all()

    def tearDown(self):
        db_store.configure_engine(None)

    def test_build_then_read(self):
        pit_idx = {"1": [("2024-04-01", 18, 2.0, 6.0),
                         ("2024-04-07", 15, 4.0, 5.0)]}
        savant_rows = [
            {"pitcher": "1", "game_date": "2024-04-01", "xwoba": 0.35},
            {"pitcher": "1", "game_date": "2024-04-03", "xwoba": 0.30},
        ]
        with patch("mlb_warehouse._pitcher_game_index", return_value=pit_idx), \
             patch("savant_history.load_days", return_value=savant_rows):
            n = pa.build_season(2024, verbose=False)
        # 04-01 has no as-of warehouse line AND no BBE before it -> skipped;
        # only the 04-07 row is written.
        self.assertEqual(n, 1)

        row = pa.asof_pitcher_features("1", "2024-04-07")
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["era"], 3.0)
        self.assertEqual(row["games"], 1)
        self.assertAlmostEqual(row["xwobacon"], 0.325)   # mean(0.35, 0.30)
        self.assertEqual(row["n_bbe"], 2)
        self.assertEqual(row["role"], "SP")
        # A date with no row reads back None (fail-open).
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
