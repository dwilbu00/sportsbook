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
        def _select(rows, shrinkage_k=15, negbin_eligible=False):
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


if __name__ == "__main__":
    unittest.main()
