"""Hermetic tests for the §2.6 `platoon` candidate feature (batter vs opposing-
starter handedness). Three layers, all pure stdlib (no live Statcast / SQL /
secrets):

  * prop_features math — _platoon_fn (strength scaling, cap bounds, strength-0
    and absent-factor no-ops), registry applies-to + runtime-knob plumbing, the
    added projection_multiplier kwarg (optional -> legacy callers byte-identical).
  * projection threading — the REAL book_line_calibration.project_and_empirical
    consumes obs["platoon_factor"]: strength 0 == production, an on-strength
    shifts BOTH the projection (methods B/C/E) and the empirical line (method A),
    an off-registry prop is a hard no-op.
  * _attach_platoon — hermetic (savant_history.load_days + find_player_id
    patched): leakage safety (the graded game's own BBE never enter the as-of
    means, cross-checked against the O(n) asof_batter_xwoba_vs_hand oracle),
    max-pitch starter-hand selection, fail-open on every missing piece, the
    batter_hits perf short-circuit, and non-baseball / load-failure no-ops.

Run: PYTHONIOENCODING=utf-8 python test_platoon.py
"""

import unittest
from unittest.mock import patch

import backtest_props as bp
import book_line_calibration as blc
import mlb_starters
import prop_features as pf
import savant_history as sh


# ── prop_features math ───────────────────────────────────────────────────────
class PlatoonFnTests(unittest.TestCase):
    def test_applies_only_to_batter_hits(self):
        self.assertTrue(pf.feature_applies("platoon", "batter_hits"))
        for pk in ("pitcher_outs", "pitcher_strikeouts",
                   "pitcher_earned_runs", "player_points"):
            self.assertFalse(pf.feature_applies("platoon", pk))

    def test_strengths_from_params_reads_platoon_knob(self):
        # The knob exists in the registry but is never set in prod (INERT); the
        # only live injector is the --feature-diag features map.
        self.assertEqual(
            pf.strengths_from_params({"platoon_strength": 1.0}),
            {"platoon": 1.0})
        self.assertEqual(
            pf.strengths_from_params({"platoon_strength": 0.0}), {})
        self.assertEqual(
            pf.strengths_from_params({"features": {"platoon": 0.5}}),
            {"platoon": 0.5})

    def test_scales_with_strength(self):
        factor = 1.05
        for s in (0.5, 1.0):
            m = pf.projection_multiplier(
                "batter_hits", {"platoon": s}, [], "2025-06-01",
                platoon_factor=factor)
            self.assertAlmostEqual(m, 1.0 + s * (factor - 1.0), places=12)

    def test_strength_zero_and_absent_factor_are_noop(self):
        # strength 0 -> production even with a factor present.
        self.assertEqual(pf.projection_multiplier(
            "batter_hits", {"platoon": 0.0}, [], "2025-06-01",
            platoon_factor=1.05), 1.0)
        # a present strength but no factor -> 1.0.
        self.assertEqual(pf.projection_multiplier(
            "batter_hits", {"platoon": 1.0}, [], "2025-06-01",
            platoon_factor=None), 1.0)
        # a factor of exactly 1.0 is neutral.
        self.assertEqual(pf.projection_multiplier(
            "batter_hits", {"platoon": 1.0}, [], "2025-06-01",
            platoon_factor=1.0), 1.0)

    def test_cap_bounds_both_directions(self):
        hi = pf.projection_multiplier(
            "batter_hits", {"platoon": 1.0}, [], "2025-06-01", platoon_factor=5.0)
        self.assertAlmostEqual(hi, 1.0 + pf.PLATOON_FEAT_CAP, places=12)
        lo = pf.projection_multiplier(
            "batter_hits", {"platoon": 1.0}, [], "2025-06-01", platoon_factor=0.1)
        self.assertAlmostEqual(lo, 1.0 - pf.PLATOON_FEAT_CAP, places=12)

    def test_excluded_prop_is_noop(self):
        self.assertEqual(pf.projection_multiplier(
            "pitcher_outs", {"platoon": 1.0}, [], "2025-06-01",
            platoon_factor=1.05), 1.0)

    def test_legacy_call_still_works(self):
        # The added kwarg is optional -> rest/gamecontext callers are unchanged.
        dates = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23",
                 "2025-05-24"]
        self.assertEqual(
            pf.projection_multiplier("batter_hits", {"rest": 1.0}, dates,
                                     "2025-05-30"),
            pf.projection_multiplier("batter_hits", {"rest": 1.0}, dates,
                                     "2025-05-30", platoon_factor=None))
        # gamecontext + platoon can both be threaded without collision.
        m = pf.projection_multiplier(
            "batter_hits", {"platoon": 1.0}, [], "2025-06-01",
            gamecontext_factors={"full": 1.03}, platoon_factor=1.02)
        self.assertAlmostEqual(m, 1.02, places=12)   # only platoon applies here


# ── projection threading (real project_and_empirical) ────────────────────────
class ProjectionThreadingTests(unittest.TestCase):
    DATES = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23",
             "2025-05-24", "2025-05-25", "2025-05-26", "2025-05-27",
             "2025-05-28", "2025-05-29"]
    VALUES = [1, 2, 1, 2, 3, 2, 1, 2, 3, 2]        # mean 1.9; straddles line 2.0
    GRADED = "2025-06-05"

    def _obs(self, prop_key="batter_hits", stat_label="H", line=2.0,
             platoon_factor=None):
        prior_games = [{"game_date": d, stat_label: v, "MIN": 0.0,
                        "is_home": None, "opponent": None}
                       for d, v in zip(self.DATES, self.VALUES)]
        o = {"prop_key": prop_key, "stat_label": stat_label, "line": line,
             "game_date": self.GRADED, "prior_games": prior_games,
             "test_game": {"is_home": None}}
        if platoon_factor is not None:
            o["platoon_factor"] = platoon_factor
        return o

    def _pe(self, features=None, platoon_factor=None, prop_key="batter_hits",
            stat_label="H", line=2.0):
        params = {"half_life": None}
        if features is not None:
            params["features"] = features
        obs = self._obs(prop_key, stat_label, line, platoon_factor)
        return blc.project_and_empirical(obs, params, "baseball_mlb")

    def test_no_features_key_is_production(self):
        base = self._pe(None, platoon_factor=1.05)
        zero = self._pe({"platoon": 0.0}, platoon_factor=1.05)
        self.assertEqual(base, zero)   # strength 0 == no-features == production

    def test_absent_factor_is_production(self):
        base = self._pe(None)
        on = self._pe({"platoon": 1.0}, platoon_factor=None)
        self.assertEqual(base, on)     # no factor -> feature no-ops

    def test_on_strength_shifts_projection_and_empirical(self):
        proj0, emp0 = self._pe(None, platoon_factor=1.05)
        proj1, emp1 = self._pe({"platoon": 1.0}, platoon_factor=1.05)
        exp_mult = pf._platoon_fn({"platoon_factor": 1.05}, 1.0)
        self.assertAlmostEqual(exp_mult, 1.05, places=12)   # within the cap
        # projection scaled by the multiplier (moves methods B/C/E) ...
        self.assertAlmostEqual(proj1, proj0 * exp_mult, places=9)
        # ... and the effective line (line / mult = 1.905) drops below 2.0, so the
        # 2-hit games now count as overs -> the empirical rate rises (moves A).
        self.assertGreater(emp1, emp0)

    def test_excluded_prop_is_noop_end_to_end(self):
        base = self._pe(None, platoon_factor=1.05,
                        prop_key="pitcher_earned_runs", stat_label="ER")
        on = self._pe({"platoon": 1.0}, platoon_factor=1.05,
                      prop_key="pitcher_earned_runs", stat_label="ER")
        self.assertEqual(base, on)


# ── _attach_platoon (hermetic) ────────────────────────────────────────────────
def _pitch(game_date, batter=None, batting_team=None, pitcher=None,
           p_throws=None, xwoba=None):
    return {"game_date": game_date, "batter": batter,
            "batting_team": batting_team, "pitcher": pitcher,
            "p_throws": p_throws, "xwoba": xwoba, "type": "X"}


class AttachPlatoonTests(unittest.TestCase):
    GRADED = "2024-07-01"
    BATTER = "111"
    TEAM = "AAA"
    VS_L_XWOBA = 0.330
    VS_R_XWOBA = 0.300
    N_L = 30            # >= PLATOON_MIN_BBE_VS
    N_R = 30

    def setUp(self):
        blc._PLATOON_CACHE.clear()     # memoized by year-set; isolate each test

    def _rows(self, l_dom_hand="L", n_vs_l=None, n_vs_r=None):
        """Synthetic pitch rows: prior vs-L/vs-R BBE for the batter (before the
        graded date) + graded-day rows that set the batter's team and the
        opposing starter's hand. A graded-day BBE with a wild 0.900 xwOBA probes
        leakage — it must never enter the as-of means."""
        n_vs_l = self.N_L if n_vs_l is None else n_vs_l
        n_vs_r = self.N_R if n_vs_r is None else n_vs_r
        rows = []
        for _ in range(n_vs_l):
            rows.append(_pitch("2024-05-01", self.BATTER, self.TEAM,
                               "900L", "L", self.VS_L_XWOBA))
        for _ in range(n_vs_r):
            rows.append(_pitch("2024-06-01", self.BATTER, self.TEAM,
                               "900R", "R", self.VS_R_XWOBA))
        # Graded day: the L starter "999L" throws the most pitches vs AAA (the
        # max-pitch proxy), incl. one BBE @0.900 the batter put in play (leakage
        # bait); the R reliever "998R" throws only a few. Which hand dominates is
        # controlled by l_dom_hand.
        big, small = ("999L", "L"), ("998R", "R")
        if l_dom_hand == "R":
            big, small = ("998R", "R"), ("999L", "L")
        rows.append(_pitch(self.GRADED, self.BATTER, self.TEAM,
                           big[0], big[1], 0.900))          # graded BBE (leak bait)
        for _ in range(19):
            rows.append(_pitch(self.GRADED, self.BATTER, self.TEAM,
                               big[0], big[1], None))       # non-BBE pitches
        for _ in range(3):
            rows.append(_pitch(self.GRADED, self.BATTER, self.TEAM,
                               small[0], small[1], None))
        return rows

    def _attach(self, rows, obs_list, sport="baseball",
                find_ret=(111, False), load_side=None):
        fp = patch.object(mlb_starters, "find_player_id", return_value=find_ret)
        if load_side is not None:
            ld = patch.object(sh, "load_days", side_effect=load_side)
        else:
            ld = patch.object(sh, "load_days", return_value=rows)
        with fp, ld as load_mock:
            blc._attach_platoon(obs_list, sport)
        return load_mock

    def _obs(self, **over):
        o = {"prop_key": "batter_hits", "player": "Test Batter",
             "game_date": self.GRADED}
        o.update(over)
        return o

    def test_leakage_safe_ratio_matches_oracle(self):
        rows = self._rows()
        obs = self._obs()
        self._attach(rows, [obs])
        # vs-L leg equals the independent O(n) oracle (graded 0.900 excluded).
        vs_oracle, n = sh.asof_batter_xwoba_vs_hand(
            self.BATTER, "L", rows, self.GRADED,
            min_bbe=blc.PLATOON_MIN_BBE_VS)
        self.assertEqual(n, self.N_L)
        self.assertAlmostEqual(vs_oracle, self.VS_L_XWOBA, places=12)
        # base = all-hands as-of mean, also excluding the graded 0.900.
        base = ((self.N_L * self.VS_L_XWOBA + self.N_R * self.VS_R_XWOBA)
                / (self.N_L + self.N_R))
        self.assertAlmostEqual(
            obs["platoon_factor"], vs_oracle / base, places=12)
        # sanity: leakage would have pulled vs-L up toward 0.900.
        leaky = (self.N_L * self.VS_L_XWOBA + 0.900) / (self.N_L + 1)
        self.assertNotAlmostEqual(obs["platoon_factor"], leaky / base, places=6)

    def test_max_pitch_pitcher_sets_hand(self):
        # R starter dominates the graded day -> the vs-R leg drives the factor.
        rows = self._rows(l_dom_hand="R")
        obs = self._obs()
        self._attach(rows, [obs])
        base = ((self.N_L * self.VS_L_XWOBA + self.N_R * self.VS_R_XWOBA)
                / (self.N_L + self.N_R))
        self.assertAlmostEqual(
            obs["platoon_factor"], self.VS_R_XWOBA / base, places=12)
        self.assertLess(obs["platoon_factor"], 1.0)   # weak vs RHP

    def test_pre_clamp_bounds_extreme_ratio(self):
        # A batter who mashes the faced hand far past the pre-clamp band.
        rows = []
        for _ in range(self.N_L):
            rows.append(_pitch("2024-05-01", self.BATTER, self.TEAM,
                               "900L", "L", 0.600))
        for _ in range(self.N_R):
            rows.append(_pitch("2024-06-01", self.BATTER, self.TEAM,
                               "900R", "R", 0.100))
        for _ in range(20):
            rows.append(_pitch(self.GRADED, self.BATTER, self.TEAM,
                               "999L", "L", None))
        obs = self._obs()
        self._attach(rows, [obs])
        self.assertAlmostEqual(
            obs["platoon_factor"], 1.0 + blc.PLATOON_RUN_CAP, places=12)

    def test_thin_vs_hand_fails_open(self):
        rows = self._rows(n_vs_l=10)      # < PLATOON_MIN_BBE_VS (25)
        obs = self._obs()
        self._attach(rows, [obs])
        self.assertNotIn("platoon_factor", obs)

    def test_thin_base_fails_open(self):
        # Enough vs-L for its leg but < PLATOON_MIN_BBE_BASE (40) all-hands.
        rows = self._rows(n_vs_l=26, n_vs_r=5)
        obs = self._obs()
        self._attach(rows, [obs])
        self.assertNotIn("platoon_factor", obs)

    def test_unknown_batter_fails_open(self):
        rows = self._rows()
        obs = self._obs()
        self._attach(rows, [obs], find_ret=None)
        self.assertNotIn("platoon_factor", obs)

    def test_pitcher_id_fails_open(self):
        rows = self._rows()
        obs = self._obs()
        self._attach(rows, [obs], find_ret=(111, True))   # is_pitcher
        self.assertNotIn("platoon_factor", obs)

    def test_missing_team_fails_open(self):
        # The batter never batted on the graded date -> batter_team missing.
        rows = self._rows()
        obs = self._obs(game_date="2024-08-15")
        self._attach(rows, [obs])
        self.assertNotIn("platoon_factor", obs)

    def test_non_baseball_is_noop_without_loading(self):
        rows = self._rows()
        obs = self._obs()
        load_mock = self._attach(rows, [obs], sport="basketball")
        self.assertNotIn("platoon_factor", obs)
        load_mock.assert_not_called()

    def test_no_batter_hits_short_circuits_before_load(self):
        obs = {"prop_key": "pitcher_outs", "player": "P", "game_date": self.GRADED}
        load_mock = self._attach(self._rows(), [obs])
        load_mock.assert_not_called()      # perf short-circuit off the hot path

    def test_load_failure_fails_open_and_caches(self):
        obs = self._obs()
        self._attach(None, [obs], load_side=RuntimeError("boom"))  # must not raise
        self.assertNotIn("platoon_factor", obs)
        # the failure is remembered so a repeat join doesn't retry the load.
        self.assertIs(blc._PLATOON_CACHE.get(frozenset({"2024"})), False)


if __name__ == "__main__":
    unittest.main()
