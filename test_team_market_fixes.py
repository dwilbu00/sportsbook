"""Tests for the team-market #1 leakage fix (1a) + its warehouse-native, zero-network
as-of starter-quality source.

as-of (backtest) mode sources the starter line from the warehouse mlb_pitcher_game
facts (O(1 query)/season, leakage-safe) and falls back to the StatsAPI byDateRange
only when the warehouse has no in-season games yet. Live mode (as_of_date=None) is
unchanged (season + Savant xERA). Offline (mocks SQL + StatsAPI).

(1b, a margin/total variance-std floor, was tried and REVERTED: the model's pred_std
already averages ~5.9 — at/above the 4.52 empirical margin std — so the floor was a
no-op. The overconfidence is a calibration/regression-to-mean problem in the point
estimate, addressed in step 2, not a variance problem.)
"""

import unittest
from unittest.mock import patch

import mlb_starters as ms
import mlb_warehouse as mw


_STAT_PAYLOAD = {"people": [{
    "fullName": "Test Pitcher", "pitchHand": {"code": "R"},
    "stats": [{"splits": [{"stat": {
        "era": "3.00", "battersFaced": 300, "strikeOuts": 80,
        "baseOnBalls": 25, "inningsPitched": "70.0", "gamesStarted": 12}}]}],
}]}


class WarehouseAsofPitcherTests(unittest.TestCase):
    # index rows are (official_date, outs, er, k)
    _IDX = {"111": [
        ("2024-04-01", 18, 2.0, 6.0),   # 6.0 IP, 2 ER
        ("2024-04-08", 21, 1.0, 8.0),   # 7.0 IP, 1 ER
        ("2024-04-15", 15, 3.0, 5.0),   # 5.0 IP, 3 ER — ON the as-of date
    ]}

    def test_asof_cumulative_and_leakage_cutoff(self):
        with patch.object(mw, "enabled", return_value=True), \
                patch.object(mw, "_pitcher_game_index", return_value=self._IDX):
            st = mw.asof_pitcher_stats("111", "2024-04-15")   # strictly before
        self.assertEqual(st["games"], 2)                       # 04-15 excluded
        self.assertAlmostEqual(st["ip"], (18 + 21) / 3.0)      # 13.0 IP
        self.assertAlmostEqual(st["era"], 3.0 / 13.0 * 9.0)    # (2+1)ER / 13IP *9
        self.assertAlmostEqual(st["avg_ip"], 13.0 / 2)

    def test_asof_none_when_no_prior_games(self):
        with patch.object(mw, "enabled", return_value=True), \
                patch.object(mw, "_pitcher_game_index", return_value=self._IDX):
            # only game is ON the date (excluded) → no prior games
            self.assertIsNone(mw.asof_pitcher_stats("111", "2024-04-01"))
            self.assertIsNone(mw.asof_pitcher_stats("999", "2024-07-01"))  # unknown

    def test_asof_none_when_sql_off(self):
        with patch.object(mw, "enabled", return_value=False):
            self.assertIsNone(mw.asof_pitcher_stats("111", "2024-07-01"))


class GetPitcherQualityAsofTests(unittest.TestCase):
    def test_asof_prefers_warehouse_zero_network(self):
        with patch.object(mw, "asof_pitcher_stats", return_value={
                    "era": 3.0, "ip": 60.0, "k": 50, "games": 10, "avg_ip": 6.0}), \
                patch.object(mw, "pitcher_throws", return_value="L"), \
                patch.object(ms, "_read_cache", return_value=None), \
                patch.object(ms, "_write_cache"), \
                patch.object(ms, "_get") as getmock, \
                patch.object(ms, "get_pitcher_expected_stats") as xmock:
            q = ms.get_pitcher_quality(543037, 2024, as_of_date="2024-06-15")
        getmock.assert_not_called()                 # ZERO StatsAPI network
        xmock.assert_not_called()                   # no Savant xERA (leaky)
        self.assertEqual(q["run_suppression_basis"], "warehouse_era")
        self.assertEqual(q["throws"], "L")
        self.assertIsNone(q["xera"])
        self.assertAlmostEqual(q["run_suppression"],
                               max(0.5, min(2.0, ms.LEAGUE_AVG["era"] / 3.0)))

    def test_asof_falls_back_to_bydaterange_when_warehouse_empty(self):
        with patch.object(mw, "asof_pitcher_stats", return_value=None), \
                patch.object(ms, "_read_cache", return_value=None), \
                patch.object(ms, "_write_cache"), \
                patch.object(ms, "_get", return_value=_STAT_PAYLOAD) as getmock, \
                patch.object(ms, "get_pitcher_expected_stats") as xmock:
            ms.get_pitcher_quality(543037, 2024, as_of_date="2024-06-15")
        getmock.assert_called()                     # fell back to StatsAPI
        self.assertIn("byDateRange", getmock.call_args[0][1]["hydrate"])
        self.assertIn("endDate=2024-06-14", getmock.call_args[0][1]["hydrate"])
        xmock.assert_not_called()                   # still skips xERA in as-of mode

    def test_live_mode_unchanged_uses_season_and_xera(self):
        cap = {}

        def fake_get(path, params):
            cap["hydrate"] = params.get("hydrate")
            return _STAT_PAYLOAD

        with patch.object(ms, "_get", side_effect=fake_get), \
                patch.object(ms, "_read_cache", return_value=None), \
                patch.object(ms, "_write_cache"), \
                patch.object(mw, "asof_pitcher_stats") as whmock, \
                patch.object(ms, "get_pitcher_expected_stats",
                             return_value={}) as xmock:
            ms.get_pitcher_quality(543037, 2024)     # as_of_date=None → live
        whmock.assert_not_called()                   # warehouse not consulted live
        self.assertIn("type=season", cap["hydrate"])
        xmock.assert_called()                        # xERA fetched live

    def test_asof_cache_key_differs_from_season(self):
        keys = []

        def fake_read(key, *a, **k):
            keys.append(key)
            return None

        with patch.object(ms, "_get", side_effect=lambda p, q: _STAT_PAYLOAD), \
                patch.object(ms, "_read_cache", side_effect=fake_read), \
                patch.object(ms, "_write_cache"), \
                patch.object(mw, "asof_pitcher_stats", return_value=None), \
                patch.object(ms, "get_pitcher_expected_stats", return_value={}):
            ms.get_pitcher_quality(543037, 2024)
            ms.get_pitcher_quality(543037, 2024, as_of_date="2024-06-15")
        self.assertEqual(keys[0], "pitcher_543037_2024")
        self.assertIn("asof_2024-06-14", keys[1])


if __name__ == "__main__":
    unittest.main()
