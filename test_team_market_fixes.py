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


class AsofSeasonRunsTests(unittest.TestCase):
    """Harness-faithfulness fix: the odds backtest now feeds as-of season runs so
    the Pythagorean blend is actually graded (it was silently inert before)."""
    _TEAMS = {"Rockies": {"id": "COL"}, "Brewers": {"id": "MIL"}}
    _SCHED = {"COL": [
        {"date": "2025-04-01", "home_team": "Rockies", "away_team": "Brewers",
         "home_score": 5, "away_score": 3},   # Rockies home: +5 scored, +3 allowed
        {"date": "2025-04-02", "home_team": "Brewers", "away_team": "Rockies",
         "home_score": 2, "away_score": 7},   # Rockies away: +7 scored, +2 allowed
        {"date": "2025-04-10", "home_team": "Rockies", "away_team": "Brewers",
         "home_score": 1, "away_score": 9},   # ON the as-of date → excluded
    ]}

    def test_cumulative_and_leakage(self):
        import backtest
        rs_ra = backtest._asof_season_runs(
            "Rockies", self._SCHED, self._TEAMS, "2025-04-10")   # strictly before
        self.assertEqual(rs_ra, (12, 5))                          # (5+7, 3+2)

    def test_none_when_no_prior_games(self):
        import backtest
        self.assertIsNone(backtest._asof_season_runs(
            "Rockies", self._SCHED, self._TEAMS, "2025-04-01"))   # first game
        self.assertIsNone(backtest._asof_season_runs(
            "Padres", self._SCHED, self._TEAMS, "2025-07-01"))    # unknown team

    def test_live_stats_populates_season_runs_only_when_given(self):
        import backtest
        s = backtest._live_stats([], (12, 5))
        self.assertEqual(s["season"]["runs_scored"], 12)
        self.assertEqual(s["season"]["runs_allowed"], 5)
        # unchanged (no runs keys) when None → pythag safely skipped (NBA/NFL path)
        self.assertNotIn("runs_scored", backtest._live_stats([])["season"])


class UnleashSweepTests(unittest.TestCase):
    """The risky part of unleash_sweep is the per-variant override of two GLOBALS
    (analysis.DEFAULT_PYTHAG_WEIGHT + pricing_common._PROB_SHRINK_CACHE) — they
    MUST be restored even though run_odds_backtest is a no-op here. Mocks the
    backtest so no data/network is needed."""

    def test_overrides_applied_per_variant_then_restored(self):
        import backtest
        import analysis
        import pricing_common
        sport = "baseball_mlb"
        analysis.DEFAULT_PYTHAG_WEIGHT = 0.35
        pricing_common._PROB_SHRINK_CACHE.pop(sport, None)   # absent before
        seen = []

        def fake_run(*a, **k):
            # snapshot the globals AS THE ANALYZERS WOULD SEE THEM mid-run
            seen.append((analysis.DEFAULT_PYTHAG_WEIGHT,
                         dict(pricing_common._PROB_SHRINK_CACHE.get(sport, {}))))

        live = {"moneyline": 0.25, "spreads": 0.25, "totals": 0.1}
        with patch.object(backtest, "run_odds_backtest", side_effect=fake_run), \
                patch.object(pricing_common, "_shrink_factor",
                             side_effect=lambda sk, m: live[m]):
            backtest.unleash_sweep(sport, "baseball", "mlb")

        # 3 variants: baseline, pythag-off, spreads-unshrunk
        self.assertEqual(len(seen), 3)
        pyths = [s[0] for s in seen]
        self.assertEqual(pyths.count(0.35), 2)               # baseline + spreads-unshrunk
        self.assertIn(0.0, pyths)                            # pythag-off variant
        # exactly one variant unshrinks spreads to 1.0 (challenger held fixed)
        self.assertEqual(
            sum(1 for _, c in seen if abs(c.get("spreads", 0.25) - 1.0) < 1e-9), 1)
        # baseline variant kept spreads at the live 0.25
        self.assertTrue(any(abs(c.get("spreads") - 0.25) < 1e-9
                            for p, c in seen if p == 0.35))
        # GLOBALS restored to their pre-call state (no leakage into live pricing)
        self.assertEqual(analysis.DEFAULT_PYTHAG_WEIGHT, 0.35)
        self.assertNotIn(sport, pricing_common._PROB_SHRINK_CACHE)

    def test_pythag_sweep_regrades_per_weight_and_restores(self):
        import backtest
        import analysis
        import pricing_common
        sport = "baseball_mlb"
        analysis.DEFAULT_PYTHAG_WEIGHT = 0.35
        seen_weights = []

        def fake_run(*a, **k):
            seen_weights.append(analysis.DEFAULT_PYTHAG_WEIGHT)

        with patch.object(backtest, "run_odds_backtest", side_effect=fake_run), \
                patch.object(pricing_common, "_shrink_factor",
                             side_effect=lambda sk, m: 0.25):
            backtest.pythag_sweep(sport, "baseball", "mlb")
        # one re-grade per swept weight, and it actually varied the weight
        self.assertEqual(seen_weights, [0.0, 0.15, 0.25, 0.35, 0.50, 0.70, 1.0])
        # DEFAULT_PYTHAG_WEIGHT restored to its pre-sweep value
        self.assertEqual(analysis.DEFAULT_PYTHAG_WEIGHT, 0.35)

    def test_combo_sweep_one_regrade_per_weight_and_restores(self):
        import backtest
        import analysis
        import pricing_common
        analysis.DEFAULT_PYTHAG_WEIGHT = 0.35
        seen = []

        def fake_run(*a, **k):
            seen.append(analysis.DEFAULT_PYTHAG_WEIGHT)

        with patch.object(backtest, "run_odds_backtest", side_effect=fake_run), \
                patch.object(pricing_common, "_shrink_factor",
                             side_effect=lambda sk, m: 0.25):
            backtest.pythag_shrink_combo(
                "baseball_mlb", "baseball", "mlb",
                weights=[0.0, 0.5, 1.0], shrinks=[0.1, 0.25, 1.0])
        # ONE re-grade per pythag weight (shrink axis is offline → free)
        self.assertEqual(seen, [0.0, 0.5, 1.0])
        self.assertEqual(analysis.DEFAULT_PYTHAG_WEIGHT, 0.35)   # restored

    def test_globals_restored_even_when_backtest_raises(self):
        import backtest
        import analysis
        import pricing_common
        sport = "baseball_mlb"
        analysis.DEFAULT_PYTHAG_WEIGHT = 0.35
        pricing_common._PROB_SHRINK_CACHE.pop(sport, None)
        live = {"moneyline": 0.25, "spreads": 0.25, "totals": 0.1}
        with patch.object(backtest, "run_odds_backtest",
                          side_effect=RuntimeError("boom")), \
                patch.object(pricing_common, "_shrink_factor",
                             side_effect=lambda sk, m: live[m]):
            with self.assertRaises(RuntimeError):
                backtest.unleash_sweep(sport, "baseball", "mlb")
        # a mid-run failure must NOT leave the overrides installed
        self.assertEqual(analysis.DEFAULT_PYTHAG_WEIGHT, 0.35)
        self.assertNotIn(sport, pricing_common._PROB_SHRINK_CACHE)


if __name__ == "__main__":
    unittest.main()
