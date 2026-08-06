"""Tests for calibration-refit leakage safety (P1.3) and selection gate (P1.4).

P1.3: the prop-calibration sweep must resolve opponent defense strictly as-of
the game date (matching the runtime model), never from a full-season aggregate
that peeks at future results.

P1.4: a fancier calibration method (pooled Gaussian / ECDF) may only be selected
over the empirical baseline if it beats it by a margin on the holdout AND
confirms out-of-sample in two expanding chronological folds — otherwise the
argmin-Brier search over ~250 candidates ships winner's-curse noise.
"""

from datetime import date, timedelta
import unittest

import backtest
import refit_calibration


class AsOfDefenseTests(unittest.TestCase):
    # Series are sorted most-recent-first, as _team_defense_lookup emits them.
    SERIES = {
        "Lakers": [("2025-01-10", 100), ("2025-01-05", 110), ("2025-01-01", 120)],
    }

    def test_season_to_date_excludes_future_games(self):
        # As-of 2025-01-08: only the 01-05 and 01-01 games count (not 01-10).
        self.assertEqual(
            backtest._resolve_opp_pa_asof("Lakers", "2025-01-08", self.SERIES),
            115.0,
        )

    def test_no_leakage_at_earlier_cutoff(self):
        self.assertEqual(
            backtest._resolve_opp_pa_asof("Lakers", "2025-01-06", self.SERIES),
            115.0,
        )

    def test_trailing_window(self):
        # window=1 -> only the most-recent game strictly before the cutoff.
        self.assertEqual(
            backtest._resolve_opp_pa_asof("Lakers", "2025-01-08", self.SERIES, 1),
            110.0,
        )

    def test_none_when_no_prior_games(self):
        self.assertIsNone(
            backtest._resolve_opp_pa_asof("Lakers", "2024-12-31", self.SERIES))

    def test_tolerant_name_match(self):
        self.assertEqual(
            backtest._resolve_opp_pa_asof("lakers", "2025-02-01", self.SERIES),
            110.0,  # mean of all three games
        )


def _dated(i):
    return (date(2025, 1, 1) + timedelta(days=i)).isoformat()


def _make_obs(n, emp_mode, separate):
    """Build calib_obs rows: (name, projected, line, actual, empirical_over, date).

    separate=True  -> projection cleanly separates the outcome (Gaussian wins).
    emp_mode='perfect' -> empirical prob equals the outcome (method A wins);
    emp_mode='flat'    -> empirical prob is 0.5 (uninformative).
    """
    obs = []
    line = 10.0
    for i in range(n):
        high = (i % 2 == 0)
        proj = (line + 5 if high else line - 5) if separate else line
        actual = line + 3 if high else line - 3
        emp = (1.0 if high else 0.0) if emp_mode == "perfect" else 0.5
        obs.append(("Player", proj, line, actual, emp, _dated(i)))
    return obs


class ChronologicalFoldsTests(unittest.TestCase):
    def test_two_disjoint_later_folds(self):
        folds = refit_calibration._chronological_folds(_make_obs(150, "flat", True))
        self.assertEqual(len(folds), 2)
        for fit_obs, score_obs in folds:
            latest_train = max(o[5] for o in fit_obs)
            earliest_test = min(o[5] for o in score_obs)
            self.assertLessEqual(latest_train, earliest_test)  # no leakage

    def test_too_little_data_returns_empty(self):
        self.assertEqual(
            refit_calibration._chronological_folds(_make_obs(30, "flat", True)), [])


class SelectionGateTests(unittest.TestCase):
    PROP = "points"

    def _winner(self, obs):
        results = {"hl10/defadj0.0/ven0.0": {self.PROP: {"calib_obs": obs}}}
        winners = refit_calibration._best_per_prop(results, [self.PROP])
        return winners.get(self.PROP)

    def test_confirmed_method_is_selected(self):
        # Gaussian separates outcomes perfectly; empirical is uninformative (0.5).
        winner = self._winner(_make_obs(150, "flat", separate=True))
        self.assertIsNotNone(winner)
        self.assertIn(winner["method"], ("B", "C"))
        self.assertTrue(winner["confirmed"])
        self.assertIsNotNone(winner["cv_brier"])
        # It genuinely beat the empirical baseline.
        self.assertLess(winner["brier"], winner["baseline_brier"])

    def test_unconfirmed_method_falls_back_to_empirical(self):
        # Empirical is near-perfect; the Gaussian cannot beat it, so the safe
        # empirical baseline (method A) must be selected, not a fancier method.
        winner = self._winner(_make_obs(150, "perfect", separate=False))
        self.assertIsNotNone(winner)
        self.assertEqual(winner["method"], "A")
        self.assertFalse(winner["confirmed"])


class SweepGridTests(unittest.TestCase):
    """P2.1b: the expanded props sweep grid + variant-name round-trip."""

    def setUp(self):
        self.grid = backtest._build_props_sweep_grid()

    def test_size_and_baseline_cell(self):
        # 4 half_lives × 3 opp × 3 def_adj × 4 shrink × 2 venue × 2 rest = 576
        # (§2.6 appended the rest/days-off candidate-feature axis {0.0, 1.0}).
        self.assertEqual(len(self.grid), 576)
        self.assertIn("none/opp0.0/defadj0.0/shrink0/ven0.0/rest0.0", self.grid)

    def test_contains_current_shipped_selections(self):
        # Baseline-is-the-floor requires the grid to still contain every knob
        # combo the live MLB calibration currently ships — else §2.1b/§2.6 could
        # regress a prop by dropping its winner from the grid. §2.6 appends the
        # rest axis, so each shipped combo survives as its rest0.0 variant.
        for cell in (
            "none/opp0.0/defadj0.0/shrink0/ven0.0/rest0.0",   # batter_hits, pitcher_outs
            "none/opp0.0/defadj0.0/shrink0/ven0.25/rest0.0",  # pitcher_K, batter_K
            "hl15/opp0.0/defadj0.0/shrink0/ven0.25/rest0.0",  # pitcher_earned_runs
        ):
            self.assertIn(cell, self.grid)

    def test_only_runtime_backed_knobs_are_set(self):
        # No NBA-only preset knob (use_minutes / pace_adj / rest_adj / def_window)
        # is ever turned on — those have no props.py runtime, so selecting one for
        # MLB would be a silent no-op (the trap P2.1 exists to avoid). NB the
        # §2.6 rest/days-off feature IS runtime-backed (props.py rest_strength),
        # so its axis is legitimately swept — distinct from the NBA rest_adj knob.
        for name, preset in self.grid.items():
            self.assertFalse(preset["use_minutes"], name)
            self.assertEqual(preset["pace_adj"], 0.0, name)
            self.assertEqual(preset["rest_adj"], 0.0, name)
            self.assertIsNone(preset["def_window"], name)

    def test_preset_values_match_label(self):
        p = self.grid["hl10/opp0.5/defadj1.0/shrink5/ven0.25/rest0.0"]
        self.assertEqual(p["half_life"], 10)
        self.assertEqual(p["opp_defense_strength"], 0.5)
        self.assertEqual(p["def_adj"], 1.0)
        self.assertEqual(p["shrink_k"], 5)
        self.assertEqual(p["venue_strength"], 0.25)
        self.assertEqual(p["rest_strength"], 0.0)

    def test_every_label_parses_and_roundtrips(self):
        for name, preset in self.grid.items():
            parsed = refit_calibration._parse_variant_name(name)
            self.assertIsNotNone(parsed, name)
            self.assertEqual(parsed["half_life"], preset["half_life"], name)
            self.assertEqual(parsed["opp_defense_strength"],
                             preset["opp_defense_strength"], name)
            self.assertEqual(parsed["output_def_strength"], preset["def_adj"], name)
            self.assertEqual(parsed["shrink_k"], preset["shrink_k"], name)
            self.assertEqual(parsed["venue_strength"],
                             preset["venue_strength"], name)


class ParseVariantNameTests(unittest.TestCase):
    """Variant-name parser: legacy 3-part + P2.1b 5-part + §2.6 6-part."""

    def test_five_part_new_format(self):
        # A 5-part label carries no rest token → rest_strength defaults to 0.0.
        p = refit_calibration._parse_variant_name(
            "hl15/opp0.5/defadj1.0/shrink10/ven0.25")
        self.assertEqual(p, {
            "half_life": 15, "opp_defense_strength": 0.5,
            "output_def_strength": 1.0, "shrink_k": 10.0,
            "venue_strength": 0.25, "rest_strength": 0.0})

    def test_six_part_rest_feature_format(self):
        # §2.6 appends an optional /rest<r> candidate-feature token.
        p = refit_calibration._parse_variant_name(
            "hl15/opp0.5/defadj1.0/shrink10/ven0.25/rest1.0")
        self.assertEqual(p, {
            "half_life": 15, "opp_defense_strength": 0.5,
            "output_def_strength": 1.0, "shrink_k": 10.0,
            "venue_strength": 0.25, "rest_strength": 1.0})

    def test_bad_rest_token_returns_none(self):
        self.assertIsNone(refit_calibration._parse_variant_name(
            "hl15/opp0.5/defadj1.0/shrink10/ven0.25/xxx1.0"))

    def test_legacy_three_part_defers_shrink_to_cli(self):
        # Legacy label carries no shrink token → shrink_k is None (unspecified),
        # opp defaults to 0.0 (it has no CLI fallback).
        p = refit_calibration._parse_variant_name("hl15/defadj1.0/ven0.25")
        self.assertEqual(p["half_life"], 15)
        self.assertEqual(p["opp_defense_strength"], 0.0)
        self.assertEqual(p["output_def_strength"], 1.0)
        self.assertEqual(p["venue_strength"], 0.25)
        self.assertIsNone(p["shrink_k"])

    def test_none_half_life(self):
        self.assertIsNone(refit_calibration._parse_variant_name(
            "none/opp0.0/defadj0.0/shrink0/ven0.0")["half_life"])

    def test_malformed_returns_none(self):
        for bad in (
            "garbage",
            "hl15/defadj1.0",                        # 2 parts
            "hl15/xyz0.5/defadj1.0/shrink5/ven0.25",  # bad opp prefix
            "zz15/opp0/defadj0/shrink0/ven0",         # bad hl prefix
            "hl15/opp0.5/defadj1.0/shrink5/xxx0.25",  # bad ven prefix
        ):
            self.assertIsNone(refit_calibration._parse_variant_name(bad), bad)


class BuildPropCfgKnobTests(unittest.TestCase):
    """P2.1b: swept opp_defense_strength + shrinkage_k persist into the cfg."""

    def _cfg(self, vname, variant_confirmed, shrinkage_k_default=0):
        obs = _make_obs(60, "flat", separate=True)
        results = {vname: {"points": {"calib_obs": obs}}}
        winner = {"variant": vname, "method": "A", "brier": 0.2, "hit": 0.5,
                  "baseline_brier": 0.22, "cv_brier": 0.21, "confirmed": False,
                  "variant_confirmed": variant_confirmed}
        return refit_calibration._build_prop_cfg(
            winner, results, "points", shrinkage_k_default)

    def test_swept_knobs_persist_from_five_part_label(self):
        cfg = self._cfg("hl10/opp0.5/defadj1.0/shrink5/ven0.25", True)
        self.assertEqual(cfg["half_life"], 10)
        self.assertEqual(cfg["opp_defense_strength"], 0.5)
        self.assertEqual(cfg["output_def_strength"], 1.0)
        self.assertEqual(cfg["shrinkage_k"], 5)
        self.assertEqual(cfg["venue_strength"], 0.25)
        self.assertTrue(cfg["variant_confirmed"])

    def test_swept_shrink_zero_is_honored_over_cli_default(self):
        # A 5-part winner that CHOSE shrink0 keeps 0 even if --shrinkage-k is set;
        # the gate's choice wins.
        cfg = self._cfg("none/opp0.0/defadj0.0/shrink0/ven0.0", False,
                        shrinkage_k_default=7)
        self.assertEqual(cfg["shrinkage_k"], 0)
        self.assertEqual(cfg["opp_defense_strength"], 0.0)
        self.assertFalse(cfg["variant_confirmed"])

    def test_legacy_label_falls_back_to_cli_shrinkage_default(self):
        cfg = self._cfg("hl10/defadj0.0/ven0.0", False, shrinkage_k_default=7)
        self.assertEqual(cfg["shrinkage_k"], 7)

    def test_fit_basis_defaults_to_synthetic_sweep(self):
        # Every prop is fit at the SYNTHETIC season-average line here; the
        # real-line pass promotes provenance to "real_line" only on a genuine
        # book-line flip. The offline builder must always start synthetic.
        cfg = self._cfg("hl10/opp0.5/defadj1.0/shrink5/ven0.25", True)
        self.assertEqual(cfg["fit_basis"], "synthetic_sweep")


if __name__ == "__main__":
    unittest.main()
