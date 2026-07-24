"""Tests for roadmap 0.3 — real-line calibration method re-selection.

The Layer-1 calibration method (A/B/C) is chosen OFFLINE at a synthetic
season-average line but applied at the REAL book line. These tests cover the
real-line re-selector (`book_line_calibration.select_method_at_real_lines`) and
the offline orchestration (`refit_calibration.refit_sport_real_lines`) that
merges the re-selected method into calibration/<sport>.json.

Fully hermetic: no live ESPN / Odds API / Azure I/O. The selector is exercised
on hand-crafted rows; the orchestration's store/join/save calls are mocked.
"""
from datetime import date, timedelta
import unittest
from unittest.mock import patch, MagicMock

import book_line_calibration as blc
import refit_calibration


def _day(i):
    return (date(2026, 1, 1) + timedelta(days=i)).isoformat()


def _signal_rows(n_main=114, n_outlier=6):
    """Rows where the residual model (C) beats empirical (A): a clean projected
    signal (residual ~ 0) with a few balanced ±50 outliers interspersed to
    inflate the Gaussian sigma (mushing method B) while leaving the ECDF (C)
    sharp. `empirical_over` is uninformative (0.5), so method A ~ Brier 0.25.
    Outcome is `actual > line` with line = 0.
    """
    total = n_main + n_outlier
    rows = []
    main_i = out_i = 0
    # place one outlier roughly every (total // n_outlier) positions
    every = max(1, total // n_outlier)
    for pos in range(total):
        if pos % every == every - 1 and out_i < n_outlier:
            sign = 1.0 if out_i % 2 == 0 else -1.0
            rows.append({"player": f"O{out_i}", "projected": 0.0, "line": 0.0,
                         "actual": sign * 50.0, "empirical_over": 0.5,
                         "game_date": _day(pos)})
            out_i += 1
        else:
            over = (main_i % 2 == 0)
            proj = 1.0 if over else -1.0
            rows.append({"player": f"P{main_i % 10}", "projected": proj,
                         "line": 0.0, "actual": proj, "empirical_over": 0.5,
                         "game_date": _day(pos)})
            main_i += 1
    return rows


class SelectMethodAtRealLinesTests(unittest.TestCase):

    def test_confirmed_residual_method_beats_empirical(self):
        # C beats A by a wide margin and confirms in both folds -> selected.
        rows = _signal_rows()
        sel = blc.select_method_at_real_lines(rows)
        self.assertIsNotNone(sel)
        self.assertEqual(sel["method"], "C")
        self.assertTrue(sel["confirmed"])
        # baseline is empirical (0.5 -> Brier 0.25) and the winner beats it.
        self.assertAlmostEqual(sel["baseline_brier"], 0.25, places=2)
        self.assertLess(sel["fit_brier"], sel["baseline_brier"])
        # residuals fit on ALL usable obs (deployed distribution).
        self.assertEqual(sel["n_obs"], len(rows))
        self.assertEqual(len(sel["residual_ecdf"]), len(rows))
        self.assertIsNotNone(sel["cv_brier"])

    def test_empirical_wins_when_unbeatable(self):
        # empirical_over == outcome -> method A is perfect; no B/C can clear it.
        rows = []
        for i in range(120):
            over = (i % 2 == 0)
            rows.append({"player": f"P{i}", "projected": 0.0, "line": 0.0,
                         "actual": 1.0 if over else -1.0,
                         "empirical_over": 1.0 if over else 0.0,
                         "game_date": _day(i)})
        sel = blc.select_method_at_real_lines(rows)
        self.assertIsNotNone(sel)
        self.assertEqual(sel["method"], "A")
        self.assertFalse(sel["confirmed"])
        self.assertAlmostEqual(sel["baseline_brier"], 0.0, places=3)

    def test_stays_A_when_folds_unavailable(self):
        # Same clean signal, but only 40 usable obs (< 3*20): no confirmation
        # folds can form, so a non-empirical method cannot be confirmed and the
        # safe empirical baseline is kept. (This is the pitcher_strikeouts case.)
        rows = []
        for i in range(40):
            over = (i % 2 == 0)
            proj = 1.0 if over else -1.0
            rows.append({"player": f"P{i}", "projected": proj, "line": 0.0,
                         "actual": proj, "empirical_over": 0.5,
                         "game_date": _day(i)})
        sel = blc.select_method_at_real_lines(rows)
        self.assertIsNotNone(sel)
        self.assertEqual(sel["method"], "A")
        self.assertFalse(sel["confirmed"])
        self.assertIsNone(sel["cv_brier"])
        self.assertEqual(sel["n_obs"], 40)

    def test_returns_none_below_min_obs(self):
        rows = [{"player": "P", "projected": 1.0, "line": 0.0, "actual": 1.0,
                 "empirical_over": 0.5, "game_date": _day(i)} for i in range(10)]
        self.assertIsNone(blc.select_method_at_real_lines(rows))

    def test_pushes_excluded_from_usable_count(self):
        # actual == line rows are pushes and must not count toward the 20 floor.
        rows = [{"player": "P", "projected": 0.0, "line": 5.0, "actual": 5.0,
                 "empirical_over": 0.5, "game_date": _day(i)} for i in range(30)]
        self.assertIsNone(blc.select_method_at_real_lines(rows))


class RealLineFoldsTests(unittest.TestCase):

    def test_no_folds_below_threshold(self):
        rows = [{"game_date": _day(i)} for i in range(59)]
        self.assertEqual(blc._real_line_folds(rows), [])

    def test_two_expanding_folds(self):
        rows = [{"game_date": _day(i)} for i in range(120)]
        folds = blc._real_line_folds(rows)
        self.assertEqual(len(folds), 2)
        (tr1, te1), (tr2, te2) = folds
        # expanding train; each set >= 20; strictly-earlier train than test.
        self.assertGreaterEqual(len(tr1), 20)
        self.assertGreaterEqual(len(te1), 20)
        self.assertGreater(len(tr2), len(tr1))
        self.assertLess(max(r["game_date"] for r in tr1),
                        min(r["game_date"] for r in te1))


class RefitSportRealLinesTests(unittest.TestCase):

    def _existing(self):
        return {
            "batter_hits": {
                "method": "A", "half_life": None, "venue_strength": 0.0,
                "opp_defense_strength": 0.0, "shrinkage_k": 0,
                "variant_label": "none/defadj0.0/ven0.0",
                "residual_mu": 0.01, "residual_sigma": 0.9,
                "residual_ecdf": [-1.0, 0.0, 1.0], "n_obs": 3638,
                "fit_brier": 0.2141, "warmup_games": 10,
                "warmup": {"method": "A", "residual_mu": 0.0,
                           "residual_sigma": 1.0, "residual_ecdf": [-1.0, 0.0, 1.0],
                           "n_obs": 3741},
            },
            "pitcher_strikeouts": {
                "method": "A", "half_life": None, "venue_strength": 0.25,
                "opp_defense_strength": 0.0, "shrinkage_k": 0,
                "warmup_games": 10, "warmup": {"method": "A"},
            },
        }

    def _run(self, sel_map, harvest=None, join=None):
        existing = self._existing()
        save_mock = MagicMock()
        # build returns a per-prop marker list so the patched selector can key
        # off the prop; the actual projection math is not exercised here.
        def _build(enriched, params, sport_key, prop_key, td=None, la=None,
                   xstats_strength=0.0, xba_index=None):
            return [{"prop_key": prop_key}]
        def _select(rows, shrinkage_k=15):
            pk = rows[0]["prop_key"] if rows else None
            return sel_map.get(pk)
        with patch.object(refit_calibration, "load_calibration",
                          return_value=existing), \
             patch.object(refit_calibration, "save_calibration", save_mock), \
             patch.object(blc, "harvest_book_lines_from_store",
                          return_value=(harvest if harvest is not None
                                        else [{"prop_key": "batter_hits"}])), \
             patch.object(blc, "join_book_lines_to_actuals",
                          return_value=(join if join is not None else ["x"])), \
             patch.object(blc, "build_real_line_obs", side_effect=_build), \
             patch.object(blc, "select_method_at_real_lines", side_effect=_select):
            refit_calibration.refit_sport_real_lines("mlb")
        return existing, save_mock

    def test_flip_writes_only_reselected_prop_and_preserves_warmup(self):
        sel = {"method": "C", "fit_brier": 0.2424, "baseline_brier": 0.2472,
               "cv_brier": 0.2411, "confirmed": True, "residual_mu": 0.05,
               "residual_sigma": 0.8, "residual_ecdf": [-2.0, 0.0, 2.0],
               "n_obs": 938}
        # pitcher_strikeouts is re-evaluated but the real-line eval CONFIRMS its
        # shipped method A -> it must NOT be rewritten (only genuine flips are).
        keep_a = {"method": "A", "fit_brier": 0.2764, "baseline_brier": 0.2764,
                  "cv_brier": None, "confirmed": False, "residual_mu": 0.0,
                  "residual_sigma": 1.0, "residual_ecdf": [-1.0, 0.0, 1.0],
                  "n_obs": 90}
        # batter_hits flips A->C and is written; pitcher_strikeouts stays A.
        existing, save_mock = self._run({"batter_hits": sel,
                                         "pitcher_strikeouts": keep_a})
        self.assertEqual(save_mock.call_count, 1)
        args, kwargs = save_mock.call_args
        sport_key, changed = args[0], args[1]
        self.assertEqual(sport_key, "baseball_mlb")
        self.assertTrue(kwargs.get("merge_props"))
        # only the re-selected prop is written -> merge leaves the rest intact.
        self.assertEqual(set(changed.keys()), {"batter_hits"})
        bh = changed["batter_hits"]
        self.assertEqual(bh["method"], "C")
        self.assertEqual(bh["residual_ecdf"], [-2.0, 0.0, 2.0])
        self.assertEqual(bh["n_obs"], 938)
        self.assertTrue(bh["confirmed"])
        # variant params + warmup carried forward from the existing cfg.
        self.assertIsNone(bh["half_life"])
        self.assertEqual(bh["venue_strength"], 0.0)
        self.assertEqual(bh["shrinkage_k"], 0)
        self.assertEqual(bh["variant_label"], "none/defadj0.0/ven0.0")
        self.assertEqual(bh["warmup"], existing["batter_hits"]["warmup"])
        self.assertTrue(bh["real_line_fit"]["fit_at_real_lines"])

    def test_nothing_written_when_no_prop_has_data(self):
        _, save_mock = self._run({"batter_hits": None,
                                  "pitcher_strikeouts": None})
        save_mock.assert_not_called()

    def test_nothing_written_when_all_methods_confirmed_unchanged(self):
        keep_a = {"method": "A", "fit_brier": 0.25, "baseline_brier": 0.25,
                  "cv_brier": None, "confirmed": False, "residual_mu": 0.0,
                  "residual_sigma": 1.0, "residual_ecdf": [0.0], "n_obs": 100}
        _, save_mock = self._run({"batter_hits": dict(keep_a),
                                  "pitcher_strikeouts": dict(keep_a)})
        save_mock.assert_not_called()

    def test_nothing_written_when_store_empty(self):
        _, save_mock = self._run({"batter_hits": None}, harvest=[])
        save_mock.assert_not_called()

    def test_no_existing_calibration_is_noop(self):
        save_mock = MagicMock()
        with patch.object(refit_calibration, "load_calibration", return_value={}), \
             patch.object(refit_calibration, "save_calibration", save_mock):
            refit_calibration.refit_sport_real_lines("mlb")
        save_mock.assert_not_called()


class VariantGateTests(unittest.TestCase):
    """P2.1: the variant confirmation gate (refit_calibration._variant_confirms)."""

    def _score(self, method, brier):
        return [{"method": method, "k": None, "brier": brier, "hit": 0.5}]

    def test_accepts_when_candidate_beats_baseline_in_both_folds(self):
        # 2 folds; per fold _variant_confirms scores candidate then baseline.
        folds = [(["t"], ["s"]), (["t"], ["s"])]
        seq = [self._score("C", 0.20), self._score("A", 0.25),   # fold 1: cand<base
               self._score("C", 0.21), self._score("A", 0.24)]   # fold 2: cand<base
        with patch.object(refit_calibration, "_chronological_folds",
                          return_value=folds), \
             patch.object(refit_calibration, "_score_calibration_methods",
                          side_effect=seq):
            self.assertTrue(refit_calibration._variant_confirms(
                ["cand"], ["base"], "C"))

    def test_rejects_when_candidate_loses_one_fold(self):
        folds = [(["t"], ["s"]), (["t"], ["s"])]
        seq = [self._score("C", 0.20), self._score("A", 0.25),   # fold 1: cand<base
               self._score("C", 0.26), self._score("A", 0.24)]   # fold 2: cand>=base
        with patch.object(refit_calibration, "_chronological_folds",
                          return_value=folds), \
             patch.object(refit_calibration, "_score_calibration_methods",
                          side_effect=seq):
            self.assertFalse(refit_calibration._variant_confirms(
                ["cand"], ["base"], "C"))

    def test_rejects_when_no_baseline_or_thin_data(self):
        self.assertFalse(refit_calibration._variant_confirms(["c"], [], "C"))
        with patch.object(refit_calibration, "_chronological_folds",
                          return_value=[]):
            self.assertFalse(refit_calibration._variant_confirms(
                ["c"], ["b"], "C"))

    def test_is_baseline_variant(self):
        self.assertTrue(refit_calibration._is_baseline_variant(
            "none/defadj0.0/ven0.0"))
        self.assertFalse(refit_calibration._is_baseline_variant(
            "hl15/defadj0.0/ven0.25"))
        self.assertFalse(refit_calibration._is_baseline_variant(
            "none/defadj1.0/ven0.0"))


class VenueParityTests(unittest.TestCase):
    """P2.1: runtime venue multiplier honors the numeric strength (== backtest)."""

    def test_numeric_strength_matches_backtest_spread(self):
        import pricing_common
        f = pricing_common._venue_match_multiplier
        # strength 0.25 -> (1.25 match, 0.75 mismatch), identical to backtest.venue_mult
        self.assertAlmostEqual(f(True, True, "baseball_mlb", strength=0.25), 1.25)
        self.assertAlmostEqual(f(True, False, "baseball_mlb", strength=0.25), 0.75)
        self.assertAlmostEqual(f(False, False, "baseball_mlb", strength=0.40), 1.40)

    def test_none_strength_keeps_legacy_fixed_weights(self):
        import pricing_common
        f = pricing_common._venue_match_multiplier
        # No strength -> the fixed per-sport VENUE_MATCH_WEIGHTS (MLB 1.40/0.60).
        self.assertAlmostEqual(f(True, True, "baseball_mlb"), 1.40)
        self.assertAlmostEqual(f(True, False, "baseball_mlb"), 0.60)

    def test_unknown_venue_is_neutral(self):
        import pricing_common
        self.assertEqual(
            pricing_common._venue_match_multiplier(None, True, "baseball_mlb",
                                                   strength=0.25), 1.0)


if __name__ == "__main__":
    unittest.main()
