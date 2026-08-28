"""Unit tests for warehouse_mirror — the local parquet backtest mirror.

Exercises the reader round-trip on synthetic parquet (no Azure): shape parity with
the db_store readers, the critical NaN->None coercion, the index builders, and the
missing-file -> None fallback contract that keeps the mirror safe-by-default.
"""
import os
import shutil
import tempfile
import unittest

import warehouse_mirror as wm

SPORT = "baseball_mlb"


class WarehouseMirrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mirror_test_")
        self._orig_dir = wm.MIRROR_DIR
        wm.MIRROR_DIR = self.tmp
        os.environ["ODI_BACKTEST_MIRROR"] = "1"
        # team lines (note point=None -> must survive as None, not NaN)
        wm._write([{
            "event_id": "E1", "game_date": "2024-05-01", "commence_time": "x",
            "home": "Mets", "away": "Cubs", "home_code": "NYM", "away_code": "CHC",
            "captured_at": "x", "kind": "team", "snapshot_id": 1,
            "bet_type": "moneyline", "selection": "Mets", "point": None,
            "price": -150, "implied_prob": 0.6, "team_code": "NYM", "game_pk": 777,
        }], wm._team_file(SPORT, "draftkings", "2024"))
        wm._write([{
            "event_id": "E1", "game_date": "2024-05-01", "commence_time": "x",
            "home": "Mets", "away": "Cubs", "captured_at": "x", "kind": "props",
            "snapshot_id": 1, "selection": "P", "player": "P",
            "prop_key": "batter_hits", "direction": "UNDER", "point": 1.5,
            "price": -120, "implied_prob": 0.54, "player_mlb_id": "111", "game_pk": 777,
        }], wm._prop_file(SPORT, "draftkings", "2024"))
        wm._write([{
            "game_pk": 777, "official_date": "2024-05-01", "season": 2024,
            "game_type": "R", "home_score": 5.0, "away_score": 3.0,
            "home_score_f5": 2.0, "away_score_f5": 1.0,
            "home_team_id": "121", "away_team_id": "112",
        }], wm._game_file(SPORT))
        wm._write([
            {"athlete_id": "999", "game_pk": 777, "season_bucket": 2024,
             "team_id": "121", "GS": 1.0, "IP": 6.0, "ER": 2.0, "K": 7.0, "BB": 1.0,
             "BF": 24.0, "official_date": "2024-05-01", "game_type": "R"},
            # a spring (S) start: kept by pitcher_game_index, EXCLUDED from calib bulk
            {"athlete_id": "999", "game_pk": 555, "season_bucket": 2024,
             "team_id": "121", "GS": 1.0, "IP": 3.0, "ER": 5.0, "K": 2.0, "BB": 2.0,
             "BF": 15.0, "official_date": "2024-03-10", "game_type": "S"},
        ], wm._pitcher_file(SPORT, "2024"))
        wm._write([{
            "athlete_id": "111", "game_pk": 777, "season_bucket": 2024, "H": 1.0,
            "SO": 1.0, "TB": 2.0, "RBI": 0.0, "official_date": "2024-05-01",
            "game_type": "R",
        }], wm._batter_file(SPORT, "2024"))

    def tearDown(self):
        wm.MIRROR_DIR = self._orig_dir
        os.environ.pop("ODI_BACKTEST_MIRROR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_enabled_gating(self):
        self.assertTrue(wm.enabled())
        os.environ["ODI_BACKTEST_MIRROR"] = "0"
        self.assertFalse(wm.enabled())

    def test_team_market_lines_shape_and_none_coercion(self):
        rows = wm.team_market_lines(SPORT, date_from="2024-01-01",
                                    date_to="2024-12-31", bookmaker="draftkings")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["bet_type"], "moneyline")
        self.assertEqual(r["price"], -150)
        self.assertIsNone(r["point"])         # NaN->None parity (critical)
        self.assertEqual(r["game_pk"], 777)

    def test_prop_lines_prop_key_filter(self):
        rows = wm.player_prop_lines(SPORT, date_from="2024-01-01",
                                    date_to="2024-12-31", prop_keys=["batter_hits"],
                                    bookmaker="draftkings")
        self.assertEqual(rows[0]["prop_key"], "batter_hits")
        self.assertEqual(rows[0]["point"], 1.5)
        # a prop not present -> empty (filter applied)
        self.assertEqual(wm.player_prop_lines(
            SPORT, date_from="2024-01-01", date_to="2024-12-31",
            prop_keys=["pitcher_strikeouts"], bookmaker="draftkings"), [])

    def test_only_early_not_mirrored_falls_back(self):
        # only_early / exclude_early are not mirrored -> None so the caller hits Azure
        self.assertIsNone(wm.team_market_lines(
            SPORT, date_from="2024-01-01", date_to="2024-12-31", only_early=True))
        self.assertIsNone(wm.player_prop_lines(
            SPORT, date_from="2024-01-01", date_to="2024-12-31", exclude_early=True))

    def test_index_builders(self):
        self.assertEqual(wm.build_team_scores_index(), {777: (5.0, 3.0)})
        self.assertEqual(wm.build_f5_scores_index(), {777: (2.0, 1.0)})
        self.assertEqual(wm.build_team_finals_index(), {777: 1.0})
        self.assertEqual(wm.game_teams_index(), {777: ("121", "112")})
        pt = wm.pitcher_team_index(seasons=["2024"])
        self.assertEqual(pt[("999", 777)], ("121", 1.0))
        self.assertIn(("999", 555), pt)       # spring start also indexed (no type filter)

    def test_pitcher_game_index_ip_to_outs(self):
        idx = wm.pitcher_game_index(2024)
        self.assertIn("999", idx)
        d, outs, er, k, bb, bf = idx["999"][1]   # [1] = regular 05-01 (spring 03-10 sorts first)
        self.assertEqual(d, "2024-05-01")
        self.assertEqual(outs, 18)               # 6.0 IP -> 18 outs
        self.assertEqual((er, k, bb, bf), (2.0, 7.0, 1.0, 24.0))

    def test_calib_gamelogs_bulk_stat_cols(self):
        pit = wm.calib_gamelogs_bulk("pitcher", 2024)
        self.assertEqual(pit["999"][0]["ER"], 2.0)
        self.assertEqual(pit["999"][0]["game_pk"], 777)
        bat = wm.calib_gamelogs_bulk("batter", 2024)
        self.assertEqual(bat["111"][0]["H"], 1.0)
        self.assertEqual(bat["111"][0]["TB"], 2.0)

    def test_game_type_exclusion_parity(self):
        # calib bulk drops S/A/E (matches get_calib_gamelogs_bulk); the as-of pitcher
        # index keeps ALL game types (matches _pitcher_game_index).
        cg = wm.calib_gamelogs_bulk("pitcher", 2024)
        self.assertEqual({r["game_pk"] for r in cg["999"]}, {777})   # spring 555 dropped
        idx = wm.pitcher_game_index(2024)
        self.assertEqual(len(idx["999"]), 2)                          # R + S both kept
        self.assertEqual(idx["999"][0][0], "2024-03-10")             # date-ordered (spring first)

    def test_read_prefers_valid_then_base(self):
        name = wm._team_file(SPORT, "draftkings", "2024")
        wm._mark_valid(name)                                  # base -> _valid
        self.assertTrue(wm._is_valid(name))
        self.assertFalse(os.path.exists(wm._path(name)))      # base gone
        rows = wm.team_market_lines(SPORT, date_from="2024-01-01",
                                    date_to="2024-12-31", bookmaker="draftkings")
        self.assertEqual(len(rows), 1)                        # read via _valid copy

    def test_write_invalidates_stale_valid(self):
        name = wm._team_file(SPORT, "draftkings", "2024")
        wm._mark_valid(name)
        self.assertTrue(wm._is_valid(name))
        wm._write([{"game_pk": 1}], name)                     # fresh data
        self.assertFalse(wm._is_valid(name))                  # stale _valid dropped
        self.assertTrue(os.path.exists(wm._path(name)))       # base present, needs re-verify

    def test_demote(self):
        name = wm._team_file(SPORT, "draftkings", "2024")
        wm._mark_valid(name)
        wm._demote(name)                                      # _valid -> base
        self.assertFalse(wm._is_valid(name))
        self.assertTrue(os.path.exists(wm._path(name)))

    def test_ensure_fast_path_needs_no_azure(self):
        # all needed files marked _valid -> ensure() returns True via the fast path,
        # never importing db_store / touching Azure.
        for f in wm._needed_files(SPORT, ["2024"]):
            if not (wm._is_valid(f) or os.path.exists(wm._path(f))):
                wm._write([{"game_pk": 1}], f)
            wm._mark_valid(f)
        self.assertTrue(wm.ensure(SPORT, ["2024"]))

    def test_flag_default_on_and_explicit_off(self):
        # ON by default (unset), OFF only when explicitly falsy.
        os.environ.pop("ODI_BACKTEST_MIRROR", None)
        self.assertTrue(wm.flag_on())                         # default ON
        os.environ["ODI_BACKTEST_MIRROR"] = "0"
        self.assertFalse(wm.flag_on())
        self.assertFalse(wm.autobuild(SPORT, ["2024"]))       # off -> no-op, no Azure

    def test_missing_season_returns_none(self):
        # a season with no parquet -> None (per-call Azure fallback), not empty/wrong
        self.assertIsNone(wm.team_market_lines(
            SPORT, date_from="2025-01-01", date_to="2025-12-31", bookmaker="draftkings"))
        self.assertIsNone(wm.pitcher_game_index(2025))
        self.assertIsNone(wm.calib_gamelogs_bulk("pitcher", 2025))


if __name__ == "__main__":
    unittest.main()
