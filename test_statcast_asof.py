"""Tests for the P2.4a Statcast expected-BA blend.

Covers the durable statcast_asof store (against in-memory SQLite, mirroring
test_gamelog_store — Azure/pymssql never touched), the savant_history batter
as-of aggregation, the props blend math + full-pipeline integration, and the
backtest batter-xBA index. All hermetic — no live network.
"""

import unittest
from unittest.mock import patch

import db_store
import statcast_asof
import savant_history as sh
import props
import backtest_props


class _SqlBackend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        statcast_asof.create_all()

    def tearDown(self):
        db_store.configure_engine(None)   # → enabled() False (no SQL_* env)


class StoreRoundTripTests(_SqlBackend, unittest.TestCase):

    def test_put_and_get(self):
        rates = {
            "111": {"xba": 0.310, "n_ab": 200, "xwoba": 0.360, "n_bbe": 150},
            "222": {"xba": 0.240, "n_ab": 50, "xwoba": 0.300, "n_bbe": 40},
        }
        self.assertEqual(
            statcast_asof.put_rates(2026, "2026-07-24", rates, "bat"), 2)
        self.assertEqual(statcast_asof.get_batter_xba("111", 2026), (0.310, 200))
        # int id + int season are coerced to the stored string/int keys.
        self.assertEqual(statcast_asof.get_batter_xba(222, 2026), (0.240, 50))

    def test_missing_row_season_and_none_fail_open(self):
        statcast_asof.put_rates(2026, "d", {
            "111": {"xba": 0.3, "n_ab": 100, "xwoba": None, "n_bbe": 0}}, "bat")
        self.assertEqual(statcast_asof.get_batter_xba("999", 2026), (None, 0))
        self.assertEqual(statcast_asof.get_batter_xba("111", 2025), (None, 0))
        self.assertEqual(statcast_asof.get_batter_xba(None, 2026), (None, 0))

    def test_replace_all_per_season(self):
        statcast_asof.put_rates(2026, "d1", {
            "111": {"xba": 0.3, "n_ab": 100, "xwoba": None, "n_bbe": 0}}, "bat")
        statcast_asof.put_rates(2026, "d2", {
            "222": {"xba": 0.25, "n_ab": 100, "xwoba": None, "n_bbe": 0}}, "bat")
        self.assertEqual(statcast_asof.get_batter_xba("111", 2026), (None, 0))
        self.assertEqual(statcast_asof.get_batter_xba("222", 2026)[0], 0.25)

    def test_role_separation(self):
        # A pitcher row and a batter row can share the same MLBAM id (role key).
        statcast_asof.put_rates(2026, "d", {
            "500": {"xba": 0.28, "n_ab": 120}}, "bat")
        statcast_asof.put_rates(2026, "d", {
            "500": {"csw_pct": 0.32, "n_pitches": 900}}, "pit")
        self.assertEqual(statcast_asof.get_batter_xba("500", 2026), (0.28, 120))
        self.assertEqual(statcast_asof.get_pitcher_csw("500", 2026), (0.32, 900))
        # Rewriting the batter role does NOT clobber the pitcher role.
        statcast_asof.put_rates(2026, "d2", {
            "500": {"xba": 0.30, "n_ab": 130}}, "bat")
        self.assertEqual(statcast_asof.get_pitcher_csw("500", 2026), (0.32, 900))

    def test_get_pitcher_csw_missing(self):
        self.assertEqual(statcast_asof.get_pitcher_csw("999", 2026), (None, 0))


class SqlOffTests(unittest.TestCase):

    def test_get_fails_open_when_sql_off(self):
        db_store.configure_engine(None)
        self.assertEqual(statcast_asof.get_batter_xba("111", 2026), (None, 0))


class BatterAsofRatesTests(unittest.TestCase):

    def _row(self, batter, date, xba=None, xwoba=None):
        return {"batter": batter, "game_date": date, "xba": xba, "xwoba": xwoba}

    def test_asof_filter_and_means(self):
        rows = [
            self._row("1", "2026-06-01", xba=0.5, xwoba=0.4),
            self._row("1", "2026-06-02", xba=0.0, xwoba=None),   # strikeout AB
            self._row("1", "2026-07-01", xba=1.0, xwoba=0.9),    # ON cutoff → excluded
        ]
        out = sh.batter_asof_rates(rows, "2026-07-01", min_ab=1)
        self.assertIn("1", out)
        self.assertAlmostEqual(out["1"]["xba"], 0.25)   # (0.5 + 0.0) / 2
        self.assertEqual(out["1"]["n_ab"], 2)
        self.assertAlmostEqual(out["1"]["xwoba"], 0.4)  # one BBE pre-cutoff
        self.assertEqual(out["1"]["n_bbe"], 1)

    def test_min_ab_gate_excludes_thin_samples(self):
        rows = [self._row("1", "2026-06-01", xba=0.5)]
        self.assertEqual(sh.batter_asof_rates(rows, "2026-07-01", min_ab=40), {})


class BlendMathTests(unittest.TestCase):

    def test_linear_blend(self):
        proj, xm = props._xstats_blend(1.0, 0.3, 4.0, 0.5)
        self.assertAlmostEqual(xm, 1.2)            # 0.3 xBA × 4 AB/game
        self.assertAlmostEqual(proj, 1.1)          # 0.5·1.0 + 0.5·1.2

    def test_weight_clamped(self):
        self.assertAlmostEqual(props._xstats_blend(1.0, 0.3, 4.0, 2.0)[0], 1.2)
        self.assertAlmostEqual(props._xstats_blend(1.0, 0.3, 4.0, -1.0)[0], 1.0)


class BuildIndexTests(unittest.TestCase):

    def test_batter_xba_index_asof(self):
        rows = [{"batter": "5", "game_date": f"2026-06-0{i}", "xba": 0.3}
                for i in range(1, 6)]
        idx = backtest_props.build_batter_xba_index(rows)
        self.assertAlmostEqual(idx.asof_mean("5", "2026-07-01", min_bbe=1), 0.3)
        self.assertIsNone(idx.asof_mean("5", "2026-06-01", min_bbe=1))  # none prior


class XstatsProjectionTests(unittest.TestCase):
    """End-to-end through analyze_player_props_value. sport_key=None + patched
    default strength keeps the MLB calibration/statsapi paths out (like the park/
    weather projection tests); find_player_id / get_batter_xba are patched."""

    def _prop_data(self):
        return {
            "commence_time": "2026-07-20T23:10:00Z",
            "home_team": "Chicago Cubs",
            "away_team": "Houston Astros",
            "game_id": "evt-xs",
            "props": {"batter_hits": {"Sticky Sam": {
                "line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
                "over_price": -110, "under_price": -110,
                "over_book": "DK", "under_book": "DK"}}},
        }

    def _histories(self, n=12):
        dates = [f"2026-06-{d:02d}" for d in range(1, 1 + n)]
        return {"Sticky Sam": {"batter_hits": {
            "found": True,
            "values": [1.0] * n,                 # 1 hit/game → base_proj 1.0
            "opponents": ["Minnesota Twins"] * n,
            "home_aways": [False] * n,
            "at_bats": [4.0] * n,                # 4 AB/game
            "game_dates": list(reversed(dates)),
        }}}

    def _run(self, strength, xba, n_ab, pid=("12345", False)):
        with patch.object(props, "DEFAULT_PLAYER_PROP_XSTATS_STRENGTH", strength), \
             patch("mlb_starters.find_player_id", return_value=pid), \
             patch("statcast_asof.get_batter_xba", return_value=(xba, n_ab)):
            cands = props.analyze_player_props_value(
                self._prop_data(), self._histories(),
                threshold_pct=1.0, sport_key=None)
        return cands[0]

    def test_off_is_neutral(self):
        off = self._run(0.0, 0.400, 300)
        self.assertIsNone(off["xstats"])
        self.assertAlmostEqual(off["avg_stat"], 1.0, places=2)

    def test_blend_shifts_projection_toward_xba(self):
        # xBA 0.200 × 4 AB = 0.8 expected hits; w=1.0 → projection = 0.8.
        on = self._run(1.0, 0.200, 300)
        self.assertIsNotNone(on["xstats"])
        self.assertEqual(on["xstats"]["n_ab"], 300)
        self.assertAlmostEqual(on["avg_stat"], 0.8, places=2)

    def test_low_sample_fails_open(self):
        on = self._run(1.0, 0.200, 10)          # n_ab < XSTATS_MIN_N (40)
        self.assertIsNone(on["xstats"])
        self.assertAlmostEqual(on["avg_stat"], 1.0, places=2)

    def test_unknown_id_fails_open(self):
        on = self._run(1.0, 0.200, 300, pid=None)
        self.assertIsNone(on["xstats"])
        self.assertAlmostEqual(on["avg_stat"], 1.0, places=2)

    def test_pitcher_id_skipped(self):
        on = self._run(1.0, 0.200, 300, pid=("999", True))  # is_pitcher → skip
        self.assertIsNone(on["xstats"])
        self.assertAlmostEqual(on["avg_stat"], 1.0, places=2)


class SchemaParityTests(unittest.TestCase):
    """Guard the Table <-> _COLS <-> sql/schema.sql from drifting (mirror
    test_gamelog_store / test_db_store SchemaParityTests)."""

    def test_columns_match_spec(self):
        cols = {c.name for c in statcast_asof.statcast_player_asof.columns}
        self.assertEqual(cols, set(statcast_asof._COLS))


class AsofRatesTests(unittest.TestCase):
    """savant_history.asof_rates — plate-discipline / contact-quality (v5)."""

    def _pitch(self, desc, typ="S", ls=None, lsa=None,
               pid="P", bid="B", date="2026-06-01"):
        return {"pitcher": pid, "batter": bid, "game_date": date,
                "description": desc, "type": typ,
                "launch_speed": ls, "launch_speed_angle": lsa}

    def _rows(self):
        # 10 pitches before the cutoff: 3 whiffs, 2 fouls, 2 called, 1 ball, 2 BIP
        # (one 100mph barrel, one 80mph non-barrel); +1 pitch ON the cutoff.
        rows = ([self._pitch("swinging_strike") for _ in range(3)]
                + [self._pitch("foul") for _ in range(2)]
                + [self._pitch("called_strike") for _ in range(2)]
                + [self._pitch("ball", typ="B")]
                + [self._pitch("hit_into_play", typ="X", ls=100.0, lsa=6),
                   self._pitch("hit_into_play", typ="X", ls=80.0, lsa=3)])
        rows.append(self._pitch("swinging_strike", date="2026-07-01"))  # excluded
        return rows

    def test_pitcher_rates(self):
        out = sh.asof_rates(self._rows(), "2026-07-01", "pitcher", min_pitches=1)
        self.assertIn("P", out)
        r = out["P"]
        self.assertEqual(r["n_pitches"], 10)          # cutoff pitch excluded
        self.assertEqual(r["n_bip"], 2)
        self.assertAlmostEqual(r["whiff_pct"], 3 / 7)  # 3 whiffs / 7 swings
        self.assertAlmostEqual(r["csw_pct"], 0.5)      # (2 called + 3 whiff)/10
        self.assertAlmostEqual(r["hard_hit_pct"], 0.5)  # 1 of 2 BIP >= 95
        self.assertAlmostEqual(r["barrel_pct"], 0.5)    # 1 of 2 BIP lsa==6

    def test_batter_key(self):
        out = sh.asof_rates(self._rows(), "2026-07-01", "batter", min_pitches=1)
        self.assertIn("B", out)
        self.assertEqual(out["B"]["n_pitches"], 10)

    def test_min_pitches_gate(self):
        self.assertEqual(
            sh.asof_rates(self._rows(), "2026-07-01", "pitcher", min_pitches=50),
            {})


if __name__ == "__main__":
    unittest.main()
