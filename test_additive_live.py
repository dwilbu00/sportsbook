"""Tests for the LIVE additive expected-runs wiring (Tier A #1d, commit 4).

Headline = the fit==serve GOLDEN: mlb_starters.live_additive_runs reproduces the
bake-off's number for the same as-of rows + model, because both run the shared
additive_runs.* spine. Plus fail-safe fallbacks, the ODI_MLB_ADDITIVE_RUNS truth
table, and analysis-seam INERTNESS (flag OFF -> byte-identical multiplicative).
Hermetic: pitcher_asof readers + the calibration loaders are monkeypatched; no SQL.
"""
import contextlib
import os
import unittest
from unittest.mock import patch

import additive_runs as ar          # importing all three at module top also proves
import analysis                     # the mlb_starters<->additive_runs import chain is
import calibration_loader           # cycle-free (a cycle would fail here at collection)
import mlb_starters
import pitcher_asof

FEATURE_KEYS = ["xwobacon", "k9"]
MODEL = {"feature_keys": FEATURE_KEYS, "intercept": 0.5, "coef": [8.0, -0.1],
         "league_rate9": 4.5, "n": 500}
CFG = {"enabled": True, "feature_keys": FEATURE_KEYS, "model": MODEL,
       "blend": {"mode": "blend", "blend_k": 200.0, "n_starts": 10},
       "bullpen": {"league_rp_era": 4.2, "league_bp": 4.5}}

SP_H = [{"as_of_date": "2025-09-28", "season_bucket": 2025, "n_bbe": 400, "ip": 180.0,
         "xwobacon": 0.34, "k9": 9.0},
        {"as_of_date": "2026-05-30", "season_bucket": 2026, "n_bbe": 40, "ip": 50.0,
         "xwobacon": 0.30, "k9": 10.0}]
SP_A = [{"as_of_date": "2025-09-28", "season_bucket": 2025, "n_bbe": 300, "ip": 150.0,
         "xwobacon": 0.36, "k9": 7.5},
        {"as_of_date": "2026-05-29", "season_bucket": 2026, "n_bbe": 30, "ip": 40.0,
         "xwobacon": 0.33, "k9": 8.0}]
RP_HT = [{"as_of_date": "2025-09-28", "era": 4.3}, {"as_of_date": "2026-05-25", "era": 3.9}]
RP_AT = [{"as_of_date": "2026-05-26", "era": 4.6}]

FACTORS = {"complete": True, "home_offense_factor": 1.1, "away_offense_factor": 0.95,
           "home_sp_id": "H", "away_sp_id": "A", "home_team_id": "HT",
           "away_team_id": "AT", "game_date": "2026-06-01",
           "home_avg_ip": 5.8, "away_avg_ip": 5.2}


def _sp(eid, season):
    return SP_H if str(eid) == "H" else SP_A


def _rp(tid, season):
    return RP_HT if str(tid) == "HT" else RP_AT


def _live_patches(cfg=CFG):
    return [patch.object(pitcher_asof, "load_sp_series", side_effect=_sp),
            patch.object(pitcher_asof, "load_rp_series", side_effect=_rp),
            patch.object(pitcher_asof, "get_or_fill", return_value=None),
            patch.object(calibration_loader, "load_expected_runs_additive",
                         return_value=cfg)]


class FitServeGoldenTests(unittest.TestCase):
    def test_live_reproduces_bakeoff_number(self):
        with contextlib.ExitStack() as s:
            for p in _live_patches():
                s.enter_context(p)
            live = mlb_starters.live_additive_runs("baseball_mlb", FACTORS)
        # Independent recomputation via the SHARED helpers + the bake-off's row
        # semantics: home_runs faces the AWAY starter + AWAY pen + HOME-lineup offense
        # (a_off_faced=home_offense_factor, a_ip=away_avg_ip, away_abbr=away_team_id).
        fg = ar.make_feat_getter({"H": SP_H, "A": SP_A}, "blend",
                                 tuple(FEATURE_KEYS), n_starts=10, blend_k=200.0)
        bpg = ar.make_bp_getter({"HT": RP_HT, "AT": RP_AT}, str, 4.2, 4.5)
        proj = ar.make_additive_projector(fg, MODEL, 4.5, tuple(FEATURE_KEYS), bpg)
        expected = proj({"date": "2026-06-01", "home_sp": "H", "away_sp": "A",
                         "home_abbr": "HT", "away_abbr": "AT",
                         "a_ip": 5.2, "h_ip": 5.8,
                         "a_off_faced": 1.1, "h_off_faced": 0.95})
        self.assertIsNotNone(live)
        self.assertAlmostEqual(live[0], expected[0], places=9)
        self.assertAlmostEqual(live[1], expected[1], places=9)
        # Sanity: the bullpen term actually moved the numbers off flat-league_bp.
        self.assertNotAlmostEqual(live[0], live[1], places=3)


class LiveAdditiveFallbackTests(unittest.TestCase):
    def test_non_mlb_returns_none(self):
        self.assertIsNone(mlb_starters.live_additive_runs("basketball_nba", FACTORS))

    def test_none_factors_returns_none(self):
        self.assertIsNone(mlb_starters.live_additive_runs("baseball_mlb", None))

    def test_disabled_cfg_returns_none(self):
        with patch.object(calibration_loader, "load_expected_runs_additive",
                          return_value={"enabled": False}):
            self.assertIsNone(
                mlb_starters.live_additive_runs("baseball_mlb", FACTORS))

    def test_missing_surfaced_key_returns_none(self):
        f = dict(FACTORS)
        f.pop("home_sp_id")
        with contextlib.ExitStack() as s:
            for p in _live_patches():
                s.enter_context(p)
            self.assertIsNone(mlb_starters.live_additive_runs("baseball_mlb", f))

    def test_no_league_rate9_returns_none(self):
        bad = dict(CFG, model={"feature_keys": FEATURE_KEYS, "coef": [8.0, -0.1]})
        with contextlib.ExitStack() as s:
            for p in _live_patches(cfg=bad):
                s.enter_context(p)
            self.assertIsNone(mlb_starters.live_additive_runs("baseball_mlb", FACTORS))


class FlagTests(unittest.TestCase):
    def test_flag_truth_table(self):
        for val, exp in [("1", True), ("true", True), ("on", True), ("yes", True),
                         ("TRUE", True), (" 1 ", True), ("", False), ("0", False),
                         ("off", False), ("garbage", False)]:
            with patch.dict(os.environ, {"ODI_MLB_ADDITIVE_RUNS": val}):
                self.assertEqual(mlb_starters._mlb_additive_runs_enabled(), exp,
                                 f"value={val!r}")

    def test_unset_is_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ODI_MLB_ADDITIVE_RUNS", None)
            self.assertFalse(mlb_starters._mlb_additive_runs_enabled())

    def test_import_chain_cycle_free(self):
        self.assertTrue(hasattr(mlb_starters, "live_additive_runs"))
        self.assertTrue(hasattr(ar, "make_additive_projector"))


class AnalysisSeamTests(unittest.TestCase):
    """The analysis seam swaps ONLY the two run scalars; OFF is byte-identical."""
    _CHALLENGER = {"enabled": True, "live_markets": {"spreads": True},
                   "final_2025_validation": {
                       "model": {"offense_weight": 1.25, "pitching_weight": 0.75,
                                 "home_base_runs": 4.4, "away_base_runs": 4.4},
                       "ensemble_challenger_share": {"home_minus_1_5": 0.7,
                                                     "margin": 0.9}}}
    _MF = {"expected_runs": {"complete": True, "home_offense_factor": 1.1,
                             "away_offense_factor": 0.95,
                             "home_staff_suppression": 1.05,
                             "away_staff_suppression": 0.98}}

    def test_flag_off_is_multiplicative_and_never_calls_additive(self):
        analysis._EXPECTED_RUNS_CACHE.clear()
        with patch.object(mlb_starters, "_mlb_additive_runs_enabled",
                          return_value=False), \
             patch.object(mlb_starters, "live_additive_runs") as spy, \
             patch.object(analysis, "load_expected_runs_challenger",
                          return_value=self._CHALLENGER):
            out = analysis._mlb_expected_runs_projection("baseball_mlb", self._MF)
        spy.assert_not_called()
        hr = mlb_starters.expected_runs_from_factors(4.4, 1.1, 0.98, 1.25, 0.75)
        ar_ = mlb_starters.expected_runs_from_factors(4.4, 0.95, 1.05, 1.25, 0.75)
        self.assertAlmostEqual(out["home_runs"], hr)
        self.assertAlmostEqual(out["away_runs"], ar_)
        self.assertAlmostEqual(out["spread_share"], 0.7)

    def test_flag_on_substitutes_additive_scalars_only(self):
        analysis._EXPECTED_RUNS_CACHE.clear()
        with patch.object(mlb_starters, "_mlb_additive_runs_enabled",
                          return_value=True), \
             patch.object(mlb_starters, "live_additive_runs",
                          return_value=(6.1, 3.2)) as spy, \
             patch.object(analysis, "load_expected_runs_challenger",
                          return_value=self._CHALLENGER):
            out = analysis._mlb_expected_runs_projection("baseball_mlb", self._MF)
        spy.assert_called_once()
        self.assertEqual(out["home_runs"], 6.1)
        self.assertEqual(out["away_runs"], 3.2)
        self.assertAlmostEqual(out["margin"], 2.9)
        self.assertAlmostEqual(out["spread_share"], 0.7)   # downstream unchanged

    def test_flag_on_but_additive_none_falls_back_to_multiplicative(self):
        analysis._EXPECTED_RUNS_CACHE.clear()
        with patch.object(mlb_starters, "_mlb_additive_runs_enabled",
                          return_value=True), \
             patch.object(mlb_starters, "live_additive_runs", return_value=None), \
             patch.object(analysis, "load_expected_runs_challenger",
                          return_value=self._CHALLENGER):
            out = analysis._mlb_expected_runs_projection("baseball_mlb", self._MF)
        hr = mlb_starters.expected_runs_from_factors(4.4, 1.1, 0.98, 1.25, 0.75)
        self.assertAlmostEqual(out["home_runs"], hr)     # multiplicative fallback


if __name__ == "__main__":
    unittest.main()
