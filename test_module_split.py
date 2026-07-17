"""Contract locks for the P3 analysis.py -> stats/pricing_common/props/parlay split.

These tests guard the refactor's invariants so a future change can't silently:
  (a) reintroduce an import cycle or leave a new module unimportable,
  (b) drop a name that one of analysis's 8 importers still resolves via the
      backward-compatible re-export facade,
  (c) re-create the _norm_cdf duplicate the split removed, or
  (d) shadow a re-exported cache with a fresh object (which would break the
      in-place cache mutation test_modeling relies on).
"""

import subprocess
import sys
import unittest

import analysis
import calibration_loader
import parlay
import pricing_common
import props
import stats


# Every name at least one of analysis's importers resolves as analysis.<name>,
# grouped by the module that now owns it. If any is missing, an importer breaks.
_NATIVE = (
    "analyze_moneyline_value", "analyze_totals_value", "analyze_spreads_value",
    "format_moneyline_report", "format_spreads_report", "format_totals_report",
    "make_bet_checklist_entry", "_predict_margin", "_mlb_expected_runs_projection",
    "_apply_starter_logit", "_EXPECTED_RUNS_CACHE",
)
_FROM_STATS = (
    "_norm_cdf", "_norm_ppf", "_normal_inv_cdf", "_recency_weights",
    "_weighted_mean", "_weighted_rate", "_weighted_quantile", "_weighted_std",
    "_half_life_for", "_SQRT2", "RECENCY_HALF_LIFE", "DEFAULT_HALF_LIFE",
)
_FROM_PRICING_COMMON = (
    "_decimal_to_american", "_consensus_price_for_line", "_expected_roi",
    "_prop_is_value", "_devig_fair", "_starter_adjustment", "_shrink_factor",
    "_apply_shrink", "_blend_weight", "_venue_match_multiplier",
    "_opponent_defense_multiplier", "VENUE_MATCH_WEIGHTS", "DEFAULT_VENUE_WEIGHTS",
    "_STARTER_ADJ_CACHE", "_PROB_SHRINK_CACHE", "_MARKET_BLEND_CACHE",
)
_FROM_PROPS = (
    "analyze_player_props_value", "format_props_report", "_mlb_prop_matchup_mult",
    "_lineup_exposure_mult", "_mlb_lineup_exposure_mult", "_log5_rate",
    "_player_prop_half_life", "_MLB_LEAGUE", "_LINEUP_ADJ_CACHE",
)
_FROM_PARLAY = (
    "generate_parlays", "_parlay_value_joint", "_gaussian_copula_joint_prob",
    "_make_psd_cholesky", "_box_muller_pairs", "_cholesky", "_normalize_legs",
    "_has_hard_conflict", "_pair_correlation", "_build_corr_matrix",
    "_copula_joint_hit_prob", "_correlation_penalty", "_same_team_prop_count",
    "_score_parlay",
)


class CleanImportTests(unittest.TestCase):
    def test_each_new_module_imports_first_in_a_fresh_interpreter(self):
        # Importing a leaf/sibling module *before* anything else in a clean
        # interpreter is the real cycle test (in-process, analysis is already
        # loaded by other tests, which would mask a cycle).
        for mod in ("stats", "pricing_common", "props", "parlay"):
            with self.subTest(module=mod):
                r = subprocess.run(
                    [sys.executable, "-c", f"import {mod}"],
                    capture_output=True, text=True)
                self.assertEqual(r.returncode, 0,
                                 msg=f"`import {mod}` failed:\n{r.stderr}")


class FacadeCompletenessTests(unittest.TestCase):
    def test_all_contract_names_resolve_on_analysis(self):
        for name in (_NATIVE + _FROM_STATS + _FROM_PRICING_COMMON
                     + _FROM_PROPS + _FROM_PARLAY):
            with self.subTest(name=name):
                self.assertTrue(hasattr(analysis, name),
                                msg=f"analysis.{name} missing after split")

    def test_reexports_point_at_the_owning_module(self):
        for name in _FROM_STATS:
            self.assertIs(getattr(analysis, name), getattr(stats, name), name)
        for name in _FROM_PRICING_COMMON:
            self.assertIs(getattr(analysis, name),
                          getattr(pricing_common, name), name)
        for name in _FROM_PROPS:
            self.assertIs(getattr(analysis, name), getattr(props, name), name)
        for name in _FROM_PARLAY:
            self.assertIs(getattr(analysis, name), getattr(parlay, name), name)


class NormCdfDedupTests(unittest.TestCase):
    def test_single_canonical_norm_cdf(self):
        # The duplicate in calibration_loader was removed; all three names must
        # be the one object defined in stats.
        self.assertIs(analysis._norm_cdf, stats._norm_cdf)
        self.assertIs(calibration_loader._norm_cdf, stats._norm_cdf)


class SharedMutableObjectIdentityTests(unittest.TestCase):
    """Re-exported caches must be the SAME object as in the owning module, so
    tests that mutate them in place through the analysis namespace are seen by
    the moved functions that read them."""

    def test_caches_are_shared_objects(self):
        self.assertIs(analysis._STARTER_ADJ_CACHE,
                      pricing_common._STARTER_ADJ_CACHE)
        self.assertIs(analysis._MARKET_BLEND_CACHE,
                      pricing_common._MARKET_BLEND_CACHE)
        self.assertIs(analysis._PROB_SHRINK_CACHE,
                      pricing_common._PROB_SHRINK_CACHE)
        self.assertIs(analysis._LINEUP_ADJ_CACHE, props._LINEUP_ADJ_CACHE)
        self.assertIs(analysis._MLB_LEAGUE, props._MLB_LEAGUE)

    def test_starter_adjustment_is_shared(self):
        self.assertIs(analysis._starter_adjustment,
                      pricing_common._starter_adjustment)


if __name__ == "__main__":
    unittest.main()
