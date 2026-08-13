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
import os
import unittest
from unittest.mock import patch, MagicMock

import book_line_calibration as blc
import player_id_map
import refit_calibration
from test_backfill_player_ids import _Backend


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


def _bet_rows(n=20, price=-110):
    """n priced test rows for a direct `_roi_tiebreak` call. Only over/under
    price matter to the helper; the P(over) vector and outcomes are supplied
    separately, positionally aligned by index."""
    return [{"over_price": price, "under_price": price} for _ in range(n)]


def _priced_rows(n=120, price=-110):
    """n dated rows carrying identical book prices, used by the tiebreak
    integration tests which patch `_score_abc_real` to control the Brier scene.
    All are usable (actual != line) and n>=100 so two confirmation folds form;
    row *content* is irrelevant under the patch — prices + dates are what the
    tiebreak path reads."""
    return [{"player": f"P{i}", "projected": 1.0, "line": 0.0, "actual": 1.0,
             "empirical_over": 0.5, "game_date": _day(i),
             "over_price": price, "under_price": price} for i in range(n)]


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

    def _run(self, sel_map, harvest=None, join=None, existing=None):
        existing = existing if existing is not None else self._existing()
        save_mock = MagicMock()
        # build returns a per-prop marker list so the patched selector can key
        # off the prop; the actual projection math is not exercised here.
        def _build(enriched, params, sport_key, prop_key, td=None, la=None,
                   xstats_strength=0.0, xba_index=None):
            return [{"prop_key": prop_key}]
        def _select(rows, shrinkage_k=15, negbin_eligible=False,
                    roi_tiebreak=True):
            pk = rows[0]["prop_key"] if rows else None
            return sel_map.get(pk)
        _harvest = (harvest if harvest is not None
                    else [{"prop_key": "batter_hits"}])
        with patch.object(refit_calibration, "load_calibration",
                          return_value=existing), \
             patch.object(refit_calibration, "save_calibration", save_mock), \
             patch.object(blc, "harvest_real_line_book_lines",
                          return_value=(_harvest, len(_harvest), 0)), \
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
        # provenance flips synthetic -> real on a genuine real-line flip.
        self.assertEqual(bh["fit_basis"], "real_line")

    def test_confirmed_unchanged_prop_keeps_synthetic_fit_basis(self):
        # A prop re-evaluated on real lines but NOT flipped (method confirmed
        # unchanged) must NOT be rewritten -> its synthetic provenance survives and
        # it gains no real_line_fit. Guards against falsely labeling stale
        # synthetic pitcher numbers as real-line fits.
        existing = self._existing()
        existing["batter_hits"]["fit_basis"] = "synthetic_sweep"
        keep_a = {"method": "A", "fit_brier": 0.2141, "baseline_brier": 0.2141,
                  "cv_brier": None, "confirmed": False, "residual_mu": 0.0,
                  "residual_sigma": 1.0, "residual_ecdf": [-1.0, 0.0, 1.0],
                  "n_obs": 900}
        _, save_mock = self._run({"batter_hits": keep_a}, existing=existing)
        save_mock.assert_not_called()
        self.assertEqual(existing["batter_hits"]["fit_basis"], "synthetic_sweep")
        self.assertNotIn("real_line_fit", existing["batter_hits"])

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

    # ── Incumbent protection (anti-churn on thin real-line samples) ──

    def test_thin_flip_is_suppressed(self):
        # pitcher_strikeouts ships A; the selector proposes A->B but on only 300
        # real-line obs (< MIN_REAL_LINE_OVERRIDE_OBS): too thin to trust the
        # 2-fold gate, so the incumbent is kept and NOTHING is written.
        thin_b = {"method": "B", "fit_brier": 0.2612, "baseline_brier": 0.2774,
                  "cv_brier": 0.2650, "confirmed": True, "residual_mu": 0.0,
                  "residual_sigma": 1.0, "residual_ecdf": [0.0], "n_obs": 300,
                  "single_split": {"A": 0.2774, "B": 0.2612}}
        _, save_mock = self._run({"batter_hits": None,
                                  "pitcher_strikeouts": thin_b})
        save_mock.assert_not_called()

    def test_deep_flip_writes_while_thin_flip_suppressed(self):
        # batter_hits flips A->C on a DEEP sample (>=500) -> written; the thin
        # pitcher_strikeouts A->B flip is suppressed in the SAME run.
        deep_c = {"method": "C", "fit_brier": 0.2424, "baseline_brier": 0.2472,
                  "cv_brier": 0.2411, "confirmed": True, "residual_mu": 0.05,
                  "residual_sigma": 0.8, "residual_ecdf": [-2.0, 0.0, 2.0],
                  "n_obs": 938, "single_split": {"A": 0.2472, "C": 0.2424}}
        thin_b = {"method": "B", "fit_brier": 0.26, "baseline_brier": 0.277,
                  "cv_brier": 0.265, "confirmed": True, "residual_mu": 0.0,
                  "residual_sigma": 1.0, "residual_ecdf": [0.0], "n_obs": 300,
                  "single_split": {"A": 0.277, "B": 0.26}}
        _, save_mock = self._run({"batter_hits": deep_c,
                                  "pitcher_strikeouts": thin_b})
        self.assertEqual(save_mock.call_count, 1)
        changed = save_mock.call_args[0][1]
        self.assertEqual(set(changed.keys()), {"batter_hits"})
        self.assertEqual(changed["batter_hits"]["method"], "C")

    def test_worse_than_incumbent_flip_suppressed(self):
        # A deep sample where the selector lands on the safe baseline A, but the
        # incumbent B still scores BETTER than A on the single split (B beat A
        # pooled and only lost a confirmation fold). Flipping B->A would drop the
        # stronger method for no gain -> keep B, write nothing.
        existing = self._existing()
        existing["pitcher_strikeouts"]["method"] = "B"
        pick_a = {"method": "A", "fit_brier": 0.278, "baseline_brier": 0.278,
                  "cv_brier": None, "confirmed": False, "residual_mu": 0.0,
                  "residual_sigma": 1.0, "residual_ecdf": [0.0], "n_obs": 800,
                  "single_split": {"A": 0.278, "B": 0.265}}
        _, save_mock = self._run({"batter_hits": None,
                                  "pitcher_strikeouts": pick_a},
                                 existing=existing)
        save_mock.assert_not_called()


class IncumbentProtectedUnitTests(unittest.TestCase):
    """Pure-function tests of refit_calibration._incumbent_protected — the
    anti-churn guard that suppresses a thin / not-better real-line method flip."""

    P = staticmethod(refit_calibration._incumbent_protected)

    def test_no_flip_returns_none(self):
        self.assertIsNone(self.P(
            {"method": "E", "n_obs": 3000, "single_split": {}}, "E"))

    def test_no_incumbent_returns_none(self):
        self.assertIsNone(self.P({"method": "C", "n_obs": 3000}, None))
        self.assertIsNone(self.P({"method": "C", "n_obs": 3000}, ""))

    def test_falsy_sel_returns_none(self):
        self.assertIsNone(self.P(None, "A"))

    def test_thin_flip_protected(self):
        note = self.P(
            {"method": "B", "n_obs": 300,
             "single_split": {"A": 0.277, "B": 0.261}}, "A")
        self.assertIsNotNone(note)
        self.assertIn("too thin", note)

    def test_deep_flip_pick_better_allowed(self):
        self.assertIsNone(self.P(
            {"method": "C", "n_obs": 900,
             "single_split": {"A": 0.247, "C": 0.242}}, "A"))

    def test_deep_flip_pick_worse_protected(self):
        note = self.P(
            {"method": "A", "n_obs": 900,
             "single_split": {"A": 0.278, "B": 0.265}}, "B")
        self.assertIsNotNone(note)
        self.assertIn("not better", note)

    def test_deep_flip_incumbent_unscored_allowed(self):
        # incumbent method absent from single_split -> can't compare the pick to
        # it -> allow the (deep, gate-confirmed) flip rather than block blindly.
        self.assertIsNone(self.P(
            {"method": "C", "n_obs": 900,
             "single_split": {"A": 0.247, "C": 0.242}}, "D"))

    def test_custom_min_override_obs(self):
        sel = {"method": "B", "n_obs": 300,
               "single_split": {"A": 0.28, "B": 0.26}}
        self.assertIsNotNone(self.P(sel, "A"))                       # default 500
        self.assertIsNone(self.P(sel, "A", min_override_obs=100))    # lowered


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


class HarvestUnionTests(unittest.TestCase):
    """Real-line obs now union the historical_odds backfill store with the app's
    RESOLVED prediction log (grows with usage), deduped by player/prop/game/line."""

    def _pred_rows(self):
        return [
            {"sport_key": "baseball_mlb", "prop_key": "batter_hits",
             "player": "A", "game_date": "2026-07-25", "line": 0.5,
             "resolved": True},
            {"sport_key": "baseball_mlb", "prop_key": "batter_hits",
             "player": "B", "game_date": "2026-07-25", "line": 1.5,
             "resolved": True},
            {"sport_key": "baseball_mlb", "prop_key": "pitcher_strikeouts",
             "player": "C", "game_date": "2026-07-25", "line": 5.5,
             "resolved": True},                                   # wrong prop
            {"sport_key": "baseball_mlb", "prop_key": "batter_hits",
             "player": "D", "game_date": "2026-07-26", "line": 0.5,
             "resolved": False},                                  # unresolved
        ]

    def test_prediction_log_harvest_filters(self):
        with patch("recalibration._read_log", return_value=self._pred_rows()):
            out = blc.harvest_book_lines_from_prediction_log(
                "baseball_mlb", ["batter_hits"])
        self.assertEqual({(r["player"], r["line"]) for r in out},
                         {("A", 0.5), ("B", 1.5)})   # C wrong-prop, D unresolved
        self.assertIsNone(out[0]["over_price"])       # prices absent (unused)

    def test_union_dedups_store_preferred(self):
        store = [
            {"sport_key": "baseball_mlb", "game_date": "2026-07-25", "player": "A",
             "prop_key": "batter_hits", "line": 0.5, "over_price": -110,
             "under_price": -110, "home_team": "H", "away_team": "X"},   # collides
            {"sport_key": "baseball_mlb", "game_date": "2026-07-25", "player": "E",
             "prop_key": "batter_hits", "line": 0.5, "over_price": -120,
             "under_price": 100, "home_team": "H", "away_team": "X"},     # unique
        ]
        with patch.object(blc, "harvest_book_lines_from_store", return_value=store), \
             patch("recalibration._read_log", return_value=self._pred_rows()):
            out, n_store, n_pred = blc.harvest_real_line_book_lines(
                "baseball_mlb", ["batter_hits"])
        self.assertEqual(n_store, 2)
        self.assertEqual(n_pred, 1)                    # only B is new (A collides)
        a = next(r for r in out if r["player"] == "A")
        self.assertEqual(a["over_price"], -110)        # store row preferred


class WarehouseHarvestTests(unittest.TestCase):
    """harvest_real_line_book_lines with the Azure warehouse as primary source:
    prediction-log backstop, event_id-based doubleheader drop, alt-line survival.
    Mocks the warehouse/pred-log seams (no SQL/network)."""

    def _wh_row(self, event_id, player, line, gd="2026-07-24",
                home="Reds", away="Guardians", pk="batter_hits"):
        return {"sport_key": "baseball_mlb", "game_date": gd,
                "commence_time": f"{gd}T23:00:00Z", "home_team": home,
                "away_team": away, "event_id": event_id, "player": player,
                "prop_key": pk, "line": line, "over_price": -110,
                "under_price": -110}

    def _pred_row(self, event_id, player, line, gd="2026-07-24",
                  pk="batter_hits"):
        return {"sport_key": "baseball_mlb", "game_date": gd,
                "commence_time": f"{gd}T23:00:00Z", "event_id": event_id,
                "home_team": None, "away_team": None, "player": player,
                "prop_key": pk, "line": line, "over_price": None,
                "under_price": None}

    def _harvest(self, wh_rows, pred_rows):
        with patch("db_store.enabled", return_value=True), \
             patch("warehouse.load_prop_lines", return_value=wh_rows), \
             patch("book_line_calibration.harvest_book_lines_from_prediction_log",
                   return_value=pred_rows):
            return blc.harvest_real_line_book_lines("baseball_mlb",
                                                    ["batter_hits"])

    def test_warehouse_primary_pred_backstop_and_dedup(self):
        wh = [self._wh_row("e1", "A", 0.5)]
        pred = [self._pred_row("e1", "A", 0.5),   # dup of wh → deduped away
                self._pred_row("e2", "B", 0.5)]   # backstop (not in wh) → kept
        out, n_primary, n_pred = self._harvest(wh, pred)
        self.assertEqual(sorted(r["player"] for r in out), ["A", "B"])
        self.assertEqual(n_primary, 1)
        self.assertEqual(n_pred, 1)

    def test_doubleheader_dropped_from_both_sources(self):
        wh = [self._wh_row("g1", "A", 0.5), self._wh_row("g2", "A", 0.5)]
        pred = [self._pred_row("g2", "A", 0.5)]   # same dh event in the log too
        out, _n_primary, _n_pred = self._harvest(wh, pred)
        self.assertEqual(out, [])                 # both games dropped

    def test_alt_lines_both_survive(self):
        wh = [self._wh_row("e1", "A", 0.5), self._wh_row("e1", "A", 1.5)]
        out, _n_primary, _n_pred = self._harvest(wh, [])
        self.assertEqual(sorted(r["line"] for r in out), [0.5, 1.5])

    def test_id_key_dedups_accent_variants_across_sources(self):
        # Same player, two spellings, SAME player_mlb_id, one game/prop/line. The
        # name key keeps them distinct (accents differ), but the id key collapses
        # them → a single pooled obs (no double-count of one game).
        wh = [self._wh_row("e1", "Jose Ramirez", 0.5)]
        wh[0]["player_mlb_id"] = "608070"
        pred = [self._pred_row("e1", "José Ramírez", 0.5)]
        pred[0]["player_mlb_id"] = "608070"
        out, n_primary, n_pred = self._harvest(wh, pred)
        self.assertEqual(len(out), 1)              # id collapses the variant
        self.assertEqual(n_primary, 1)
        self.assertEqual(n_pred, 0)               # pred row was the id-dup
        self.assertEqual(out[0]["player"], "Jose Ramirez")   # primary preferred

    def test_name_key_still_dedups_when_only_one_source_has_id(self):
        # Asymmetric enrichment (only the pred row carries an id): the id key can't
        # match (wh id-key is None), but the shared odds-feed NAME still dedups →
        # no regression during the manual-runbook transition window.
        wh = [self._wh_row("e1", "A", 0.5)]                    # un-enriched
        pred = [self._pred_row("e1", "A", 0.5)]
        pred[0]["player_mlb_id"] = "608070"                   # only pred enriched
        out, n_primary, n_pred = self._harvest(wh, pred)
        self.assertEqual(len(out), 1)
        self.assertEqual(n_pred, 0)


class JoinToActualsTests(unittest.TestCase):
    """Exercise the REAL join_book_lines_to_actuals (not mocked): idx binding,
    prior_games, and the doubleheader date guard. Mocks only the ESPN fetch."""

    def _gamelog(self, dates_hits):
        return [{"game_date": f"{d}T23:00:00Z", "H": h} for d, h in dates_hits]

    def _book_row(self, gd="2026-07-24", line=0.5):
        return {"sport_key": "baseball_mlb", "game_date": gd,
                "commence_time": f"{gd}T23:00:00Z", "event_id": "e1",
                "home_team": "Reds", "away_team": "Guardians",
                "player": "A. Batter", "prop_key": "batter_hits", "line": line,
                "over_price": -110, "under_price": -110}

    def _join(self, book_rows, gamelog):
        with patch("book_line_calibration.cached_athlete_id",
                   return_value="123"), \
             patch("book_line_calibration.cached_gamelog", return_value=gamelog):
            return blc.join_book_lines_to_actuals(book_rows, "baseball", "mlb")

    def test_join_attaches_actual_and_prior_games(self):
        gl = self._gamelog([(f"2026-07-{d:02d}", 1) for d in range(13, 24)]
                           + [("2026-07-24", 2)])
        out = self._join([self._book_row()], gl)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["actual"], 2.0)
        self.assertEqual(out[0]["stat_label"], "H")
        self.assertGreaterEqual(len(out[0]["prior_games"]), 10)

    def test_join_skips_doubleheader_date(self):
        gl = self._gamelog([(f"2026-07-{d:02d}", 1) for d in range(12, 24)]
                           + [("2026-07-24", 2), ("2026-07-24", 0)])
        self.assertEqual(self._join([self._book_row()], gl), [])

    def _pitcher_gamelog(self, dates_k):
        # Pitcher rows carry IP (the pitcher discriminator) plus K/SO.
        return [{"game_date": f"{d}T23:00:00Z", "IP": 6.0, "K": k, "SO": k}
                for d, k in dates_k]

    def test_join_drops_pitcher_prop_on_batter_gamelog(self):
        # A pitcher_strikeouts book line pooled (via a namesake / id-map slip)
        # onto a BATTER's gamelog: the shared "SO" label would bind and grade the
        # bet off the batter's whiffs. The role gate drops the row instead.
        gl = self._gamelog([(f"2026-07-{d:02d}", 1) for d in range(12, 24)]
                           + [("2026-07-24", 2)])
        for g in gl:
            g["SO"] = 1                       # would let _stat_label_for match
        row = self._book_row(line=5.5)
        row["prop_key"] = "pitcher_strikeouts"
        self.assertEqual(self._join([row], gl), [])

    def test_join_grades_pitcher_prop_on_pitcher_gamelog(self):
        # Role matches -> the gate is transparent; the pitcher prop grades on "K".
        gl = self._pitcher_gamelog(
            [(f"2026-07-{d:02d}", 7) for d in range(12, 24)] + [("2026-07-24", 9)])
        row = self._book_row(line=5.5)
        row["prop_key"] = "pitcher_strikeouts"
        out = self._join([row], gl)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["actual"], 9.0)
        self.assertEqual(out[0]["stat_label"], "K")

    def test_join_drops_tb_rbi_on_espn_gamelog(self):
        # batter_total_bases / batter_rbis are WAREHOUSE-ONLY. With no mlb_id the
        # warehouse is skipped and we fall open to the ESPN gamelog — which must NOT
        # grade these off its uncalibrated 'RBI' (or a phantom 'TB'). The row is
        # dropped rather than poison the calibration corpus.
        gl = [{"game_date": f"2026-07-{d:02d}T23:00:00Z", "H": 1, "RBI": 2, "TB": 3}
              for d in range(12, 25)]
        for prop in ("batter_rbis", "batter_total_bases"):
            row = self._book_row(line=0.5)
            row["prop_key"] = prop
            self.assertEqual(self._join([row], gl), [], prop)

    def _join_warehouse(self, book_rows, gamelog):
        # CALIB on + book line carrying an mlb_id → the gamelog is warehouse-sourced.
        with patch.dict(os.environ, {blc._MLB_WAREHOUSE_CALIB_ENV: "1"}), \
             patch("player_id_map.espn_id_for_mlb_id", return_value="123"), \
             patch("book_line_calibration.cached_athlete_id", return_value="123"), \
             patch("mlb_warehouse.get_calib_gamelog", return_value=gamelog), \
             patch("book_line_calibration.cached_gamelog", return_value=[]):
            return blc.join_book_lines_to_actuals(book_rows, "baseball", "mlb")

    def test_tb_rbi_grade_off_warehouse(self):
        # The guard is SOURCE-aware, not a blanket drop: when the warehouse serves
        # the gamelog (CALIB on + mlb_id), batter_rbis grades normally off its 'RBI'.
        wh = [{"game_date": f"2026-07-{d:02d}T23:00:00Z", "RBI": 1, "completed": True}
              for d in range(12, 24)] + [
              {"game_date": "2026-07-24T23:00:00Z", "RBI": 3, "completed": True}]
        row = self._book_row(line=0.5)
        row["prop_key"] = "batter_rbis"
        row["player_mlb_id"] = "592450"
        out = self._join_warehouse([row], wh)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["actual"], 3.0)
        self.assertEqual(out[0]["stat_label"], "RBI")


class OfflineParkProjectionTests(unittest.TestCase):
    """`project_and_empirical` reconstructs production's park-factor road-context
    delta (props.py combined_mult) so the re-fit residual basis matches the LIVE
    projection. These pin the three behaviours the "measure, then decide" dry-run
    established: (1) a park-eligible prop folds park into BOTH the projection and
    the comparison line; (2) a park-neutral prop is byte-identical to the no-park
    path (the pitcher_outs control); (3) a prediction-log row (home_team None)
    fails open to no shift. Park never changes a shipped method today, but it must
    keep behaving this way as the warehouse grows."""

    def _obs(self, prop="batter_hits", stat="H", home="Reds", away="Guardians",
             upcoming_home=True, line=1.0, values=None):
        # 12 prior games, all played at home (past_parks -> player's own park),
        # so a non-1.0 upcoming park produces a genuine road-context delta. MLB
        # rows carry MIN 0.0 (no minutes filter). Values straddle line/1.10 so
        # the line-shift is observable in the empirical over-rate.
        values = values if values is not None else [1, 2, 0, 1, 2, 1, 0, 1, 2, 1, 1, 0]
        prior = [{stat: float(v), "is_home": True, "opponent": away,
                  "game_date": f"2026-06-{i + 1:02d}", "MIN": 0.0}
                 for i, v in enumerate(values)]
        return {
            "prop_key": prop, "stat_label": stat, "line": line,
            "home_team": home, "away_team": away,
            "test_game": {"is_home": upcoming_home},
            "prior_games": prior,
            "game_date": "2026-07-01", "player": "A. Batter",
        }

    def test_park_folds_into_projection_and_line(self):
        obs = self._obs()
        params = {"half_life": 5.0}
        with patch("props._park_factor_mult", return_value=(1.10, {"kind": "hits"})):
            proj_on, emp_on = blc.project_and_empirical(obs, params, "baseball_mlb")
        with patch("props._park_factor_mult", return_value=(1.0, None)):
            proj_off, emp_off = blc.project_and_empirical(obs, params, "baseball_mlb")
        # Projection scales by the park multiplier (mean-scale).
        self.assertAlmostEqual(proj_on, proj_off * 1.10, places=9)
        # The comparison line shifts by the inverse (line/1.10 < line), so at least
        # one prior value (the 1.0s) now clears it -> a strictly higher over-rate.
        self.assertGreater(emp_on, emp_off)

    def test_park_neutral_prop_is_byte_identical(self):
        # batter_strikeouts is not in park_factors.PROP_PARK_KIND -> park_mult 1.0
        # even with a home team set. Must match an explicitly-neutralized run
        # exactly (the pitcher_outs 0.3056/0.3056 control from the dry-run).
        obs = self._obs(prop="batter_strikeouts", stat="SO",
                        values=[0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 0], line=1.5)
        params = {"half_life": 5.0}
        proj_real, emp_real = blc.project_and_empirical(obs, params, "baseball_mlb")
        with patch("props._park_factor_mult", return_value=(1.0, None)):
            proj_neut, emp_neut = blc.project_and_empirical(obs, params, "baseball_mlb")
        self.assertEqual((proj_real, emp_real), (proj_neut, emp_neut))

    def test_park_fails_open_on_prediction_log_row(self):
        # Prediction-log rows carry no game frame (home_team None, is_home None).
        # The upcoming park is unknown -> _park_factor_mult fails closed to 1.0,
        # so the projection is unshifted (identical to a neutralized run).
        obs = self._obs()
        obs["home_team"] = None
        obs["away_team"] = None
        obs["test_game"] = {"is_home": None}
        params = {"half_life": 5.0}
        proj_real, emp_real = blc.project_and_empirical(obs, params, "baseball_mlb")
        with patch("props._park_factor_mult", return_value=(1.0, None)):
            proj_neut, emp_neut = blc.project_and_empirical(obs, params, "baseball_mlb")
        self.assertEqual((proj_real, emp_real), (proj_neut, emp_neut))


class JoinIdBridgeTests(_Backend, unittest.TestCase):
    """join_book_lines_to_actuals pivots athlete-id resolution onto the book
    line's player_mlb_id via the SFBB bridge (MLBAM→ESPN), bypassing the
    name-based cache entirely; same-id spellings pool into ONE gamelog fetch.
    Uses the map-seeding mixin so espn_id_for_mlb_id resolves for real."""

    def _gamelog(self, dates_hits):
        return [{"game_date": f"{d}T23:00:00Z", "H": h} for d, h in dates_hits]

    def _row(self, player, line=0.5, mlb_id="608070", gd="2026-07-24"):
        return {"sport_key": "baseball_mlb", "game_date": gd,
                "commence_time": f"{gd}T23:00:00Z", "event_id": "e1",
                "home_team": "Reds", "away_team": "Guardians",
                "player": player, "player_mlb_id": mlb_id,
                "prop_key": "batter_hits", "line": line,
                "over_price": -110, "under_price": -110}

    def test_id_resolves_aid_and_skips_name_search(self):
        # The book row's name is a garbage spelling the name cache would miss, but
        # the correct player_mlb_id resolves the ESPN athlete_id via the bridge.
        gl = self._gamelog([(f"2026-07-{d:02d}", 1) for d in range(13, 24)]
                           + [("2026-07-24", 2)])
        seen = {}

        def fake_gamelog(sport, league, aid, **kw):
            seen["aid"] = aid
            return gl

        with patch("book_line_calibration.cached_athlete_id") as m_name, \
             patch("book_line_calibration.cached_gamelog",
                   side_effect=fake_gamelog):
            out = blc.join_book_lines_to_actuals(
                [self._row("Totally Wrong Spelling")], "baseball", "mlb")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["actual"], 2.0)
        m_name.assert_not_called()                 # name cache bypassed
        self.assertEqual(seen["aid"],
                         player_id_map.espn_id_for_mlb_id("608070"))

    def test_same_id_two_spellings_pool_one_fetch(self):
        # Two alt-line rows for the same player under different spellings: grouped
        # by id → a single ESPN gamelog fetch, both lines still join.
        gl = self._gamelog([(f"2026-07-{d:02d}", 1) for d in range(12, 24)]
                           + [("2026-07-24", 2)])
        calls = []

        def fake_gamelog(sport, league, aid, **kw):
            calls.append(aid)
            return gl

        rows = [self._row("Jose Ramirez", line=0.5),
                self._row("José Ramírez", line=1.5)]
        with patch("book_line_calibration.cached_athlete_id") as m_name, \
             patch("book_line_calibration.cached_gamelog",
                   side_effect=fake_gamelog):
            out = blc.join_book_lines_to_actuals(rows, "baseball", "mlb")
        self.assertEqual(len(calls), 1)            # pooled: ONE fetch
        m_name.assert_not_called()
        self.assertEqual(sorted(r["line"] for r in out), [0.5, 1.5])

    def test_falls_back_to_name_when_id_unresolved(self):
        # An unmapped id must not strand the row: the join falls back to the
        # name-based cache (zero regression for un-enriched / unknown players).
        gl = self._gamelog([(f"2026-07-{d:02d}", 1) for d in range(13, 24)]
                           + [("2026-07-24", 2)])
        with patch("book_line_calibration.cached_athlete_id",
                   return_value="999") as m_name, \
             patch("book_line_calibration.cached_gamelog", return_value=gl):
            out = blc.join_book_lines_to_actuals(
                [self._row("A. Batter", mlb_id="000404")], "baseball", "mlb")
        self.assertEqual(len(out), 1)
        m_name.assert_called_once()                # id miss → name path


class RoiTiebreakUnitTests(unittest.TestCase):
    """Direct tests of the `_roi_tiebreak` decision logic (guardrail branches).

    At -110/-110 the de-vigged consensus is 0.50 and a bet fires when
    |p - 0.5| >= 0.05 and the leg is +EV (p >= ~0.524). So p=0.6 backs the OVER
    (wins when o==1) and p=0.4 backs the UNDER (wins when o==0); p=0.51 clears
    neither edge nor is placed. Winning pays decimal-1 (~0.909), losing -1.
    """

    def _out(self, n=20):
        return [1, 0] * (n // 2)

    def test_override_applies(self):
        out = self._out(20)
        # C always backs the winning side (100% hit); A always bets OVER (~50%).
        probs = {"A": [0.6] * 20, "C": [0.6 if o else 0.4 for o in out]}
        rec = blc._roi_tiebreak(_bet_rows(20), probs, out, ["A", "C"], "A",
                                threshold=0.05, min_bets=15, min_roi_gain=0.02)
        self.assertTrue(rec["applied"])
        self.assertEqual(rec["winner"], "C")
        self.assertEqual(rec["brier_leader"], "A")
        self.assertEqual(rec["n_bets"]["A"], 20)
        self.assertEqual(rec["n_bets"]["C"], 20)
        self.assertGreater(rec["rois"]["C"], rec["rois"]["A"])

    def test_no_override_when_roi_margin_not_met(self):
        out = self._out(20)
        probs = {"A": [0.6] * 20, "C": [0.6 if o else 0.4 for o in out]}
        # Same clear ROI edge, but demand an impossibly large margin.
        rec = blc._roi_tiebreak(_bet_rows(20), probs, out, ["A", "C"], "A",
                                threshold=0.05, min_bets=15, min_roi_gain=2.0)
        self.assertFalse(rec["applied"])
        self.assertEqual(rec["winner"], "A")

    def test_no_override_when_winner_below_bet_floor(self):
        out = self._out(20)
        # C never clears the edge (0.51 -> no bets); leader A trades fine.
        probs = {"A": [0.6] * 20, "C": [0.51] * 20}
        rec = blc._roi_tiebreak(_bet_rows(20), probs, out, ["A", "C"], "A",
                                threshold=0.05, min_bets=15, min_roi_gain=0.02)
        self.assertFalse(rec["applied"])
        self.assertEqual(rec["n_bets"]["C"], 0)
        self.assertEqual(rec["winner"], "A")

    def test_no_override_when_leader_untradeable(self):
        out = self._out(20)
        # Leader A places no bets on this split -> keep the Brier pick, no flip.
        probs = {"A": [0.51] * 20, "C": [0.6 if o else 0.4 for o in out]}
        rec = blc._roi_tiebreak(_bet_rows(20), probs, out, ["A", "C"], "A",
                                threshold=0.05, min_bets=15, min_roi_gain=0.02)
        self.assertFalse(rec["applied"])
        self.assertEqual(rec["n_bets"]["A"], 0)
        self.assertEqual(rec["winner"], "A")

    def test_no_prices_returns_none(self):
        out = self._out(20)
        probs = {"A": [0.6] * 20, "C": [0.6 if o else 0.4 for o in out]}
        unpriced = [{"over_price": None, "under_price": None} for _ in range(20)]
        self.assertIsNone(
            blc._roi_tiebreak(unpriced, probs, out, ["A", "C"], "A",
                              threshold=0.05, min_bets=15, min_roi_gain=0.02))


class RoiTiebreakSelectionTests(unittest.TestCase):
    """`select_method_at_real_lines` wiring: tie_set construction (confirmed +
    within-band), the override, the `roi_tiebreak=False` opt-out, and the
    "clear Brier winner -> ROI never consulted" case. `_score_abc_real` is
    patched to script the Brier scene deterministically (call 0 = single
    holdout, calls 1..2 = the two confirmation folds)."""

    def _patch(self, single_scores, single_probs, single_out, fold_scores,
               fold_confirm=True):
        """Script `_score_abc_real`: call 0 = single holdout (the given scene),
        calls 1..2 = the two confirmation folds. Each fold now returns REAL
        per-fold probs/out sized to its own `len(test)` so the ROI cross-fold
        guard can re-run `_roi_tiebreak` on it (the same C-backs-the-winner scene
        the single holdout uses -> C confirms). With ``fold_confirm=False`` the
        SECOND fold's C is made inert (P(over)~market -> no value bets -> C
        untradeable there), so the winner fails one fold and the guard suppresses
        the override."""
        calls = {"n": 0}

        def _se(train, test, negbin_eligible=False):
            i = calls["n"]
            calls["n"] += 1
            if i == 0:
                return (single_scores, (0.0, 1.0, []), single_probs, single_out)
            m = len(test)
            fo = [1, 0] * (m // 2) + ([1] if m % 2 else [])
            if not fold_confirm and i == 2:
                c = [0.51] * m           # edge ~0.01 < threshold -> no C bets
            else:
                c = [0.6 if o else 0.4 for o in fo]   # C backs the winning side
            fp = {"A": [0.6] * m, "B": [0.5] * m, "C": c}
            return (fold_scores, (0.0, 1.0, []), fp, fo)

        return patch.object(blc, "_score_abc_real", side_effect=_se)

    def _scene(self):
        # 60 test rows (single holdout of 120 usable). C backs the winning side
        # (100% hit); A always bets OVER (~50%). C's single-split Brier misses
        # the 0.002 margin over A (0.001) but confirms in both folds -> ROI can
        # rescue it (tie_set == [A, C]).
        out = [1, 0] * 30
        probs = {"A": [0.6] * 60, "B": [0.5] * 60,
                 "C": [0.6 if o else 0.4 for o in out]}
        single = {"A": 0.250, "B": 0.260, "C": 0.249}
        folds = {"A": 0.25, "B": 0.26, "C": 0.24}   # C<A both folds; B does not
        return single, probs, out, folds

    def test_roi_rescues_confirmed_within_band_of_A(self):
        single, probs, out, folds = self._scene()
        with self._patch(single, probs, out, folds):
            sel = blc.select_method_at_real_lines(_priced_rows(120))
        self.assertEqual(sel["method"], "C")
        self.assertTrue(sel["confirmed"])
        rt = sel["roi_tiebreak"]
        self.assertTrue(rt["applied"])
        self.assertEqual(rt["winner"], "C")
        self.assertEqual(rt["brier_leader"], "A")
        self.assertCountEqual(rt["tie_set"], ["A", "C"])

    def test_opt_out_keeps_brier_pick(self):
        single, probs, out, folds = self._scene()
        with self._patch(single, probs, out, folds):
            sel = blc.select_method_at_real_lines(_priced_rows(120),
                                                  roi_tiebreak=False)
        self.assertEqual(sel["method"], "A")     # Brier gate keeps A (0.001<margin)
        self.assertIsNone(sel["roi_tiebreak"])

    def test_clear_brier_winner_not_consulted(self):
        # C beats A by 0.05 (well past the margin) -> C wins on Brier alone and
        # A falls out of the band, so the tiebreak never runs.
        out = [1, 0] * 30
        probs = {"A": [0.6] * 60, "B": [0.5] * 60,
                 "C": [0.6 if o else 0.4 for o in out]}
        single = {"A": 0.250, "B": 0.260, "C": 0.200}
        folds = {"A": 0.25, "B": 0.26, "C": 0.20}
        with self._patch(single, probs, out, folds):
            sel = blc.select_method_at_real_lines(_priced_rows(120))
        self.assertEqual(sel["method"], "C")
        self.assertIsNone(sel["roi_tiebreak"])   # tie_set == [C]; never ran

    def test_roi_override_requires_both_folds(self):
        # C wins the single-split ROI AND clears the ROI margin in BOTH folds ->
        # the cross-fold guard confirms it and the override is honored.
        single, probs, out, folds = self._scene()
        with self._patch(single, probs, out, folds, fold_confirm=True):
            sel = blc.select_method_at_real_lines(_priced_rows(120))
        self.assertEqual(sel["method"], "C")
        rt = sel["roi_tiebreak"]
        self.assertTrue(rt["applied"])
        self.assertIs(rt["fold_confirmed"], True)
        self.assertEqual(len(rt["fold_recs"]), 2)
        self.assertTrue(all(fr["applied"] and fr["winner"] == "C"
                            for fr in rt["fold_recs"]))

    def test_roi_override_suppressed_when_fold_fails(self):
        # C wins the single-split ROI but is untradeable in the SECOND fold ->
        # the cross-fold guard keeps the Brier pick (A) and marks the override
        # unconfirmed. Mirrors the 2-fold winner's-curse guard the Brier gate
        # already enforces via _confirms.
        single, probs, out, folds = self._scene()
        with self._patch(single, probs, out, folds, fold_confirm=False):
            sel = blc.select_method_at_real_lines(_priced_rows(120))
        self.assertEqual(sel["method"], "A")
        rt = sel["roi_tiebreak"]
        self.assertFalse(rt["applied"])          # forced False by the guard
        self.assertIs(rt["fold_confirmed"], False)
        self.assertEqual(rt["winner"], "C")      # single-split winner recorded
        # first fold confirmed C, second fold did not (C untradeable there).
        self.assertEqual(rt["fold_recs"][0]["winner"], "C")
        self.assertNotEqual(rt["fold_recs"][1]["winner"], "C")

    def test_guard_reaches_per_bucket_path(self):
        # The guard lives inside select_method_at_real_lines behind
        # `if roi_tiebreak:`. The per-bucket path (_select_line_methods) must
        # forward roi_tiebreak=True so the guard is live there too, not just on
        # the pooled path. Spy the selector and assert the forwarded flag.
        obs = [{"prop_key": "batter_hits", "player": f"P{i}", "line": 0.5,
                "actual": 1.0, "game_date": _day(i),
                "over_price": -110, "under_price": -110} for i in range(120)]
        spy = MagicMock(return_value=None)
        with patch.object(blc, "project_and_empirical", return_value=(1.0, 0.6)), \
                patch.object(blc, "project_distributional", return_value=0.6), \
                patch.object(blc, "select_method_at_real_lines", spy):
            refit_calibration._select_line_methods(
                "batter_hits", obs, params={}, sport_key="baseball_mlb",
                team_defense={}, league_avg_def={}, pooled_method="A",
                xba_index=None, quality_index=None)
        self.assertTrue(spy.called)
        _args, kwargs = spy.call_args
        self.assertIs(kwargs.get("roi_tiebreak"), True)


class CalibWarehouseCutoverTests(unittest.TestCase):
    """P4 calibration cutover: join_book_lines_to_actuals grades off the warehouse
    per-game facts when ODI_MLB_WAREHOUSE_CALIB is on (MLB + mlb_id), else the ESPN
    cached_gamelog path (flag OFF = byte-identical). The sweep engine is untouched."""

    def test_calib_role_majority(self):
        self.assertEqual(blc._calib_role(
            [{"prop_key": "batter_hits"}, {"prop_key": "batter_strikeouts"}]),
            "batter")
        self.assertEqual(blc._calib_role(
            [{"prop_key": "pitcher_strikeouts"}, {"prop_key": "pitcher_outs"}]),
            "pitcher")
        self.assertEqual(blc._calib_role(                        # tie → batter
            [{"prop_key": "pitcher_outs"}, {"prop_key": "batter_hits"}]),
            "batter")

    def _book_line(self):
        return {"player": "Aaron Judge", "player_mlb_id": "592450",
                "prop_key": "batter_hits", "line": 0.5,
                "game_date": "2024-07-04T18:00:00Z"}

    def _run(self, flag):
        # warehouse log dated OFF the book line → row is skipped after the fetch,
        # so we assert only the SOURCE routing (no downstream projection).
        wh_log = [{"game_date": "2024-06-01T18:00:00Z", "H": 1.0, "AB": 4.0,
                   "opponent": "Boston Red Sox", "is_home": True, "completed": True}]
        with patch.dict(os.environ, {blc._MLB_WAREHOUSE_CALIB_ENV: flag}), \
             patch.object(blc, "cached_athlete_id", return_value="e1"), \
             patch("player_id_map.espn_id_for_mlb_id", return_value=None), \
             patch("mlb_warehouse.get_calib_gamelog", return_value=wh_log) as wh, \
             patch.object(blc, "cached_gamelog", return_value=[]) as esp:
            blc.join_book_lines_to_actuals([self._book_line()], "baseball", "mlb")
        return wh, esp

    def test_flag_on_uses_warehouse(self):
        wh, esp = self._run("1")
        wh.assert_called_once_with("592450", "batter")
        esp.assert_not_called()

    def test_flag_off_uses_espn(self):
        wh, esp = self._run("")
        wh.assert_not_called()
        esp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
