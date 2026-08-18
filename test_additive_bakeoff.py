"""Tests for the additive bake-off wiring in backtest_starters (Tier A #1b).

Covers the pure/logic pieces that don't need SQL or a cached Statcast corpus:
label orientation, projector orientation, exp-IP clamping, grader shape.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import additive_runs as ar
import backtest_starters as bs
import calibration_loader as cl
import pitcher_asof


class FeatureListInvariantTests(unittest.TestCase):
    """The fit==serve spine: the live single-entity series (pitcher_asof) and the
    offline bulk series (backtest_starters) MUST carry the same feature columns, and
    every bake-off family must be a subset of what the loaders pull. These were only
    documented in comments before; enforce them so any future feature addition that
    forgets one side is caught here, not by a silent live/offline projection drift."""

    def test_series_features_match_all_asof_features(self):
        self.assertEqual(set(pitcher_asof._SERIES_FEATURES),
                         set(bs._ALL_ASOF_FEATURES))

    def test_every_family_is_subset_of_loaded_features(self):
        loaded = set(bs._ALL_ASOF_FEATURES)
        for name, keys in bs._ADDITIVE_FEATURE_SETS.items():
            missing = set(keys) - loaded
            self.assertFalse(
                missing, f"family {name!r} needs unloaded features {missing}")

    def test_all_asof_features_are_stored_columns(self):
        # Every loadable feature must be a real column populated by build_season
        # (_STAT_COLS) or the warehouse curve (k9/k_pct/bb_pct), else the series
        # loader would KeyError / return NULL for it.
        stored = set(pitcher_asof._STAT_COLS) | {"k9", "k_pct", "bb_pct"}
        self.assertTrue(set(bs._ALL_ASOF_FEATURES) <= stored)


class CswFeatureFamilyTests(unittest.TestCase):
    """Batch A #25 — activate the inert csw_pct as an additive bake-off feature."""

    def test_csw_family_registered_and_carries_csw(self):
        self.assertIn("csw", bs._ADDITIVE_FEATURE_SETS)
        self.assertIn("csw_pct", bs._ADDITIVE_FEATURE_SETS["csw"])
        # csw must be loadable on BOTH the offline and live series or fit != serve.
        self.assertIn("csw_pct", bs._ALL_ASOF_FEATURES)
        self.assertIn("csw_pct", pitcher_asof._SERIES_FEATURES)

    def test_csw_is_marginal_over_contact(self):
        # The csw family = contact + csw_pct (marginal-then-joint read); the ONLY extra
        # key must be csw_pct, so the bake-off measures CSW's marginal cleanly.
        extra = set(bs._ADDITIVE_FEATURE_SETS["csw"]) - set(
            bs._ADDITIVE_FEATURE_SETS["contact"])
        self.assertEqual(extra, {"csw_pct"})

    def test_csw_family_avoids_null_until_backfill_columns(self):
        # csw must NOT inherit k_pct/bb_pct (NULL until the #1c-a BB/BF re-backfill) or
        # feat_from_row's any-null-key drop would auto-skip every row -> ungradable on
        # current data. Guards the adversarial-review fix.
        self.assertNotIn("k_pct", bs._ADDITIVE_FEATURE_SETS["csw"])
        self.assertNotIn("bb_pct", bs._ADDITIVE_FEATURE_SETS["csw"])

    def test_extra_series_column_does_not_perturb_a_non_csw_config(self):
        # The live byte-identical mechanism: a config whose feature_keys omit csw_pct
        # reads the SAME features whether or not the series row carries csw_pct, so
        # adding the column to the series lists cannot move a non-csw projection.
        keys = ("xwobacon", "k9")
        without = {"xwobacon": 0.32, "k9": 8.0, "n_bbe": 150}
        with_csw = dict(without, csw_pct=0.31)
        f_without, n_without = ar.feat_from_row(without, keys)
        f_with, n_with = ar.feat_from_row(with_csw, keys)
        self.assertEqual(f_without, f_with)
        self.assertEqual(n_without, n_with)
        self.assertNotIn("csw_pct", f_with)


class ExpIpTests(unittest.TestCase):
    def test_clamp_and_default(self):
        self.assertEqual(bs._exp_ip(6.0), 6.0)
        self.assertEqual(bs._exp_ip(10.0), 7.0)      # clamp hi
        self.assertEqual(bs._exp_ip(1.0), 3.5)       # clamp lo
        self.assertEqual(bs._exp_ip(None), 5.2)      # default


class AdditiveTrainingRowsTests(unittest.TestCase):
    def test_label_orientation_and_drops(self):
        asof = {
            ("H", "2024-05-01"): {"xwobacon": 0.30, "k9": 8.0, "n_bbe": 100},
            ("A", "2024-05-01"): {"xwobacon": 0.40, "k9": 6.0, "n_bbe": 100},
            ("H2", "2024-05-02"): {"xwobacon": None, "k9": 7.0, "n_bbe": 5},  # dropped
        }
        rows = [
            {"home_sp": "H", "away_sp": "A", "date": "2024-05-01",
             "home_runs": 2.0, "away_runs": 5.0},
            {"home_sp": "H2", "away_sp": "A", "date": "2024-05-02",
             "home_runs": 3.0, "away_runs": 4.0},
        ]
        train = bs._additive_training_rows(
            rows, bs._dict_feat_getter(asof, ("xwobacon", "k9")),
            ("xwobacon", "k9"))
        # Game 1: home SP "H" -> label = away_runs (5.0); away SP "A" -> home_runs (2.0).
        # Game 2: "H2" has null xwobacon -> dropped; "A" (2024-05-02) not in asof -> dropped.
        labels = sorted(r["label"] for r in train)
        self.assertEqual(labels, [2.0, 5.0])
        h = next(r for r in train if r["label"] == 5.0)   # the home SP row
        self.assertAlmostEqual(h["xwobacon"], 0.30)


class AdditiveProjectorTests(unittest.TestCase):
    def test_orientation_better_away_starter_suppresses_home(self):
        asof = {
            ("A", "2024-05-01"): {"xwobacon": 0.30, "k9": 8.0, "n_bbe": 200},  # better
            ("H", "2024-05-01"): {"xwobacon": 0.40, "k9": 6.0, "n_bbe": 200},  # worse
        }
        # rate9 rises with xwobacon (worse pitcher -> more runs).
        model = {"feature_keys": ["xwobacon", "k9"], "intercept": 0.0,
                 "coef": [10.0, 0.0], "league_rate9": 4.0, "n": 1000}
        proj = bs._make_additive_projector(
            bs._dict_feat_getter(asof, ("xwobacon", "k9")), model, 4.0,
            ("xwobacon", "k9"))
        row = {"home_sp": "H", "away_sp": "A", "date": "2024-05-01",
               "a_ip": 6.0, "h_ip": 6.0, "a_off_faced": 1.0, "h_off_faced": 1.0}
        home_runs, away_runs = proj(row)
        # Home bats vs the BETTER away starter (A) -> fewer home runs than away runs.
        self.assertLess(home_runs, away_runs)

    def test_missing_features_fall_back_to_league(self):
        model = {"feature_keys": ["xwobacon", "k9"], "intercept": 0.0,
                 "coef": [10.0, 0.0], "league_rate9": 4.3, "n": 1000}
        proj = bs._make_additive_projector(
            bs._dict_feat_getter({}, ("xwobacon", "k9")), model, 4.3,
            ("xwobacon", "k9"))
        row = {"home_sp": "H", "away_sp": "A", "date": "2024-05-01",
               "a_ip": None, "h_ip": None, "a_off_faced": 1.0, "h_off_faced": 1.0}
        hr, ar = proj(row)   # both starters missing -> league_bp both sides
        self.assertAlmostEqual(hr, ar)          # symmetric fallback
        self.assertAlmostEqual(hr, 4.3, places=3)


class BullpenGetterTests(unittest.TestCase):
    def setUp(self):
        self.series = {"10": [{"as_of_date": "2024-04-01", "era": 4.5},
                              {"as_of_date": "2024-04-10", "era": 5.0}]}
        self.abbr_to_id = {"NYY": "10"}
        self.g = bs._make_bp_getter(self.series, self.abbr_to_id.get,
                                    league_rp_era=4.0, league_bp=4.3)

    def test_league_relative_scaling(self):
        # as-of era 4.5 vs league 4.0 -> ratio 1.125 -> 4.3 * 1.125.
        self.assertAlmostEqual(self.g("NYY", "2024-04-05"), 4.3 * (4.5 / 4.0))

    def test_strict_before(self):
        # 04-01 is not strictly before 04-01 -> no prior line -> flat league_bp.
        self.assertAlmostEqual(self.g("NYY", "2024-04-01"), 4.3)

    def test_unknown_team_falls_back(self):
        self.assertAlmostEqual(self.g("BOS", "2024-04-05"), 4.3)  # no abbr->id

    def test_ratio_is_clamped(self):
        series = {"10": [{"as_of_date": "2024-04-01", "era": 100.0}]}
        g = bs._make_bp_getter(series, {"NYY": "10"}.get, 4.0, 4.3)
        self.assertAlmostEqual(g("NYY", "2024-04-05"), 4.3 * 2.0)  # clamp hi

    def test_no_league_era_falls_back(self):
        g = bs._make_bp_getter(self.series, self.abbr_to_id.get, None, 4.3)
        self.assertAlmostEqual(g("NYY", "2024-04-05"), 4.3)

    def test_resolve_id_identity_callable(self):
        # #1d: the live path passes an IDENTITY resolver (team_id -> team_id) instead
        # of a dict.get — the getter must work for any callable, same result.
        g = bs._make_bp_getter(self.series, lambda k: k, 4.0, 4.3)  # series keyed "10"
        self.assertAlmostEqual(g("10", "2024-04-05"), 4.3 * (4.5 / 4.0))
        self.assertAlmostEqual(g("99", "2024-04-05"), 4.3)          # unknown -> league


class AdditiveRunsExtractionTests(unittest.TestCase):
    """#1d: the pure helpers moved to additive_runs.py; backtest_starters aliases them
    so the OFFLINE bake-off and the LIVE path run the SAME code (fit==serve spine)."""
    def test_aliases_are_the_extracted_functions(self):
        import additive_runs as ar
        self.assertIs(bs._exp_ip, ar.exp_ip)
        self.assertIs(bs._feat_from_row, ar.feat_from_row)
        self.assertIs(bs._window_diff, ar.window_diff)
        self.assertIs(bs._make_feat_getter, ar.make_feat_getter)
        self.assertIs(bs._make_bp_getter, ar.make_bp_getter)
        self.assertIs(bs._make_additive_projector, ar.make_additive_projector)


class ProjectorBullpenTests(unittest.TestCase):
    def test_away_bullpen_backs_home_runs(self):
        # Equal starters -> the bullpen term decides. home_runs is backed by the AWAY
        # team's bullpen; a worse away pen (higher rate9) should raise home_runs.
        asof = {("A", "2024-05-01"): {"xwobacon": 0.35, "k9": 8.0, "n_bbe": 200},
                ("H", "2024-05-01"): {"xwobacon": 0.35, "k9": 8.0, "n_bbe": 200}}
        model = {"feature_keys": ["xwobacon", "k9"], "intercept": 4.0,
                 "coef": [0.0, 0.0], "league_rate9": 4.0, "n": 1000}
        bp = {"AWY": 6.0, "HOM": 2.0}
        proj = bs._make_additive_projector(
            bs._dict_feat_getter(asof, ("xwobacon", "k9")), model, 4.0,
            ("xwobacon", "k9"), bp_getter=lambda team, date: bp[team])
        row = {"home_sp": "H", "away_sp": "A", "date": "2024-05-01",
               "home_abbr": "HOM", "away_abbr": "AWY",
               "a_ip": 6.0, "h_ip": 6.0, "a_off_faced": 1.0, "h_off_faced": 1.0}
        home_runs, away_runs = proj(row)
        self.assertGreater(home_runs, away_runs)   # bad away pen -> more home runs


class WindowingTests(unittest.TestCase):
    def setUp(self):
        # One pitcher "P": a 2023 prior-season final + three 2024 as-of rows.
        self.series = {"P": [
            {"as_of_date": "2023-09-30", "season_bucket": 2023,
             "xwobacon": 0.30, "n_bbe": 400, "k9": 9.0, "ip": 180.0},
            {"as_of_date": "2024-04-01", "season_bucket": 2024,
             "xwobacon": 0.40, "n_bbe": 10, "k9": 6.0, "ip": 5.0},
            {"as_of_date": "2024-04-08", "season_bucket": 2024,
             "xwobacon": 0.38, "n_bbe": 20, "k9": 7.0, "ip": 11.0},
            {"as_of_date": "2024-04-15", "season_bucket": 2024,
             "xwobacon": 0.36, "n_bbe": 30, "k9": 8.0, "ip": 17.0},
        ]}
        self.keys = ("xwobacon", "k9")

    def test_cumulative(self):
        g = bs._make_feat_getter(self.series, "cumulative", self.keys)
        feats, n = g("P", "2024-04-15")
        self.assertAlmostEqual(feats["xwobacon"], 0.36)
        self.assertEqual(n, 30)

    def test_window_last_1_start_differences(self):
        g = bs._make_feat_getter(self.series, "window", self.keys, n_starts=1)
        feats, n = g("P", "2024-04-15")
        # (0.36*30 - 0.38*20) / (30-20) = 3.2/10 = 0.32
        self.assertAlmostEqual(feats["xwobacon"], 0.32, places=4)
        self.assertEqual(n, 10)

    def test_blend_with_prior_season(self):
        g = bs._make_feat_getter(self.series, "blend", self.keys, blend_k=200.0)
        feats, n = g("P", "2024-04-15")
        w = 30 / (30 + 200)
        self.assertAlmostEqual(feats["xwobacon"], w * 0.36 + (1 - w) * 0.30, places=5)
        self.assertEqual(n, 30)

    def test_window_diff_k9(self):
        old = {"k9": 7.0, "ip": 11.0, "xwobacon": 0.38, "n_bbe": 20}
        new = {"k9": 8.0, "ip": 17.0, "xwobacon": 0.36, "n_bbe": 30}
        d = bs._window_diff(old, new, ("k9",))
        # ((8*17/9) - (7*11/9)) / (17-11) * 9
        self.assertAlmostEqual(d["k9"], ((8 * 17 / 9) - (7 * 11 / 9)) / 6 * 9, places=4)

    def test_window_no_new_bbe_returns_none(self):
        old = {"xwobacon": 0.36, "n_bbe": 30}
        new = {"xwobacon": 0.36, "n_bbe": 30}
        self.assertIsNone(bs._window_diff(old, new, ("xwobacon",)))


class GraderShapeTests(unittest.TestCase):
    def test_variant_metrics_projfn_shape(self):
        rows = [
            {"home_win": 1, "margin": 2.0, "home_runs": 5.0, "away_runs": 3.0,
             "total_runs": 8.0},
            {"home_win": 0, "margin": -1.0, "home_runs": 3.0, "away_runs": 4.0,
             "total_runs": 7.0},
            {"home_win": 1, "margin": 3.0, "home_runs": 6.0, "away_runs": 3.0,
             "total_runs": 9.0},
        ]
        m = bs._variant_metrics_projfn(
            rows, rows, lambda r: (r["home_runs"], r["away_runs"]))
        for key in ("ml", "spread", "margin", "total_rmse", "score_nll"):
            self.assertIn(key, m)
        self.assertIn("brier", m["ml"])
        self.assertIn("rmse", m["margin"])
        self.assertGreaterEqual(m["score_nll"], 0.0)


class SaveAdditiveModelTests(unittest.TestCase):
    """#1d commit 5: save_additive_model fits + STAGES the candidate block
    (never live). Heavy data loaders + the fit are mocked; only the assembly +
    candidate-staging is exercised."""
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._od, self._oa = cl.CALIBRATION_DIR, cl.ARCHIVE_DIR
        cl.CALIBRATION_DIR = self._dir
        cl.ARCHIVE_DIR = os.path.join(self._dir, "archive")
        cl.set_candidate_mode(False)

    def tearDown(self):
        cl.CALIBRATION_DIR, cl.ARCHIVE_DIR = self._od, self._oa
        cl.set_candidate_mode(False)

    def test_stages_candidate_block_never_live(self):
        rows = [{"date": "2026-05-01", "home_sp": "H", "away_sp": "A",
                 "home_runs": 4.0, "away_runs": 3.0}]
        model = {"feature_keys": ["xwobacon", "k9"], "intercept": 0.5,
                 "coef": [8.0, -0.1], "league_rate9": 4.5, "n": 300}
        with patch.object(bs, "get_season_games", return_value=[{}]), \
             patch.object(bs, "build_dataset", return_value=(rows, 0.32)), \
             patch.object(bs, "_load_pitcher_asof_series",
                          return_value={"H": [{}], "A": [{}]}), \
             patch.object(bs, "_additive_training_rows",
                          return_value=[{"xwobacon": 0.3, "k9": 9.0, "label": 4.0}] * 20), \
             patch.object(bs.xera_lite, "fit", return_value=model), \
             patch.object(bs, "_load_bullpen_asof_series", return_value=({}, 4.2)):
            block = bs.save_additive_model([2026])
        self.assertTrue(block["enabled"])
        self.assertEqual(block["model"], model)
        self.assertEqual(block["feature_keys"], ["xwobacon", "k9"])
        self.assertEqual(block["blend"]["mode"], "blend")
        self.assertEqual(block["bullpen"]["league_rp_era"], 4.2)
        self.assertEqual(block["bullpen"]["league_bp"], 4.5)   # == model league_rate9
        # Wrote a CANDIDATE, never live.
        self.assertTrue(os.path.exists(cl.candidate_path("baseball_mlb")))
        self.assertFalse(os.path.exists(cl.calibration_path("baseball_mlb")))
        staged = cl._read_json(cl.candidate_path("baseball_mlb"))
        self.assertEqual(
            staged["expected_runs_additive"]["model"]["league_rate9"], 4.5)


if __name__ == "__main__":
    unittest.main()
