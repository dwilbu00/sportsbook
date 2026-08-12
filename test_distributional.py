"""§2.4b-2 distributional batter_hits model.

Covers the pure binomial survival helper, the shared contact-quality composite,
the runtime `_distributional_over_rate` branch (fail-open + statcast lookup), the
leakage-safe as-of quality index, the offline `project_distributional`, and the
`diagnose_distributional` reporting wiring — all hermetic (no live Statcast/ESPN).
"""

import io
import unittest
from contextlib import redirect_stdout
from math import comb
from unittest.mock import patch

import backtest_props
import book_line_calibration as blc
import mlb_starters
import props
import refit_calibration
import statcast_asof
import stats


def _binom_ge(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


class HitsAtLeastTests(unittest.TestCase):
    def test_k1_is_one_minus_miss_all(self):
        for n in (1, 3, 4, 6):
            for p in (0.1, 0.25, 0.4):
                self.assertAlmostEqual(stats.hits_at_least(1, n, p),
                                       1 - (1 - p) ** n)

    def test_matches_bruteforce_binomial(self):
        for n in (2, 4, 5, 7):
            for k in range(0, n + 2):
                for p in (0.05, 0.3, 0.55, 0.9):
                    self.assertAlmostEqual(stats.hits_at_least(k, n, p),
                                           max(0.0, min(1.0, _binom_ge(k, n, p))),
                                           places=9)

    def test_edges(self):
        self.assertEqual(stats.hits_at_least(0, 4, 0.3), 1.0)   # k<=0
        self.assertEqual(stats.hits_at_least(1, 0, 0.3), 0.0)   # n<=0
        self.assertEqual(stats.hits_at_least(1, 4, 0.0), 0.0)   # p=0
        self.assertEqual(stats.hits_at_least(1, 4, 1.0), 1.0)   # p=1
        self.assertEqual(stats.hits_at_least(5, 4, 0.3), 0.0)   # k>n
        self.assertEqual(stats.hits_at_least(3, 2, 1.0), 0.0)   # k>n even at p=1

    def test_fractional_n_is_rounded(self):
        self.assertEqual(stats.hits_at_least(1, 3.8, 0.25),
                         stats.hits_at_least(1, 4, 0.25))


class DistPOverCompositeTests(unittest.TestCase):
    def test_level_blend_and_quality_nudge(self):
        # level = (1-.5)*.28 + .5*.30 = .29; q_adj = 1 + .1*(.45/.39-1)
        #                                              + .1*(.09/.075-1) = 1.03538..
        p, meta = props._dist_p_over(0.28, 3.9, 0.30, 0.45, 0.09, 1.0, 1.0,
                                     0.5, 0.5)
        self.assertEqual(meta["k"], 1)
        self.assertEqual(meta["n_ab_expected"], 4)
        self.assertAlmostEqual(meta["p_ab"], round(0.29 * (
            1 + 0.1 * (0.45 / 0.39 - 1) + 0.1 * (0.09 / 0.075 - 1)), 4), places=4)
        # meta["p_ab"] is rounded to 4dp, so compare the survival at 3 places.
        self.assertAlmostEqual(p, stats.hits_at_least(1, 4, meta["p_ab"]),
                               places=3)

    def test_missing_xba_drops_blend(self):
        # xba None -> s forced to 0 -> level = r_emp regardless of strength.
        p, meta = props._dist_p_over(0.25, 4.0, None, None, None, 1.0, 1.0,
                                     0.5, 0.5)
        self.assertEqual(meta["xba_weight"], 0.0)
        self.assertAlmostEqual(meta["p_ab"], 0.25)
        self.assertAlmostEqual(p, 1 - 0.75 ** 4)

    def test_line_maps_to_k(self):
        _, m05 = props._dist_p_over(0.25, 4.0, None, None, None, 1, 1, 0.5, 0.0)
        _, m15 = props._dist_p_over(0.25, 4.0, None, None, None, 1, 1, 1.5, 0.0)
        self.assertEqual((m05["k"], m15["k"]), (1, 2))

    def test_rate_and_exposure_multipliers(self):
        _, m = props._dist_p_over(0.25, 4.0, None, None, None, 1.2, 1.5, 0.5, 0.0)
        self.assertAlmostEqual(m["p_ab"], 0.30)          # 0.25 * 1.2
        self.assertEqual(m["n_ab_expected"], 6)          # round(4.0 * 1.5)

    def test_pab_bounds(self):
        _, m = props._dist_p_over(0.99, 4.0, None, None, None, 2.0, 1.0, 0.5, 0.0)
        self.assertLessEqual(m["p_ab"], props._DIST_PAB_BOUNDS[1])


class DistributionalOverRateTests(unittest.TestCase):
    def _series(self):
        values = [1, 2, 0, 1, 1, 0, 2, 1, 1, 0, 1, 2]
        at_bats = [4, 4, 3, 4, 5, 4, 4, 3, 4, 4, 5, 4]
        weights = [1.0] * len(values)
        return values, at_bats, weights

    def test_not_whitelisted_returns_none(self):
        v, ab, w = self._series()
        p, meta = props._distributional_over_rate(
            "pitcher_strikeouts", 0.5, v, ab, w, 1.0, 1.0, "P",
            "2024-07-01T18:00Z", {}, 0.5)
        self.assertIsNone(p)
        self.assertIsNone(meta)

    def test_no_usable_ab_returns_none(self):
        v, _, w = self._series()
        p, meta = props._distributional_over_rate(
            "batter_hits", 0.5, v, [None] * len(v), w, 1.0, 1.0, "B",
            "2024-07-01T18:00Z", {}, 0.5)
        self.assertIsNone(p)

    def test_statcast_lookup_and_kmapping(self):
        v, ab, w = self._series()
        rates = {"xba": 0.31, "hard_hit_pct": 0.44, "barrel_pct": 0.09,
                 "n_ab": 120}
        with patch.object(mlb_starters, "find_player_id",
                          return_value=("12345", False)), \
             patch.object(statcast_asof, "get_rates", return_value=rates):
            p05, m05 = props._distributional_over_rate(
                "batter_hits", 0.5, v, ab, w, 1.0, 1.0, "Bat",
                "2024-07-01T18:00Z", {}, 0.5)
            p15, m15 = props._distributional_over_rate(
                "batter_hits", 1.5, v, ab, w, 1.0, 1.0, "Bat",
                "2024-07-01T18:00Z", {}, 0.5)
        self.assertEqual((m05["k"], m15["k"]), (1, 2))
        self.assertEqual(m05["xba"], 0.31)
        self.assertEqual(m05["n_ab_sample"], 120)
        self.assertGreater(p05, p15)     # P(>=1) > P(>=2)
        self.assertAlmostEqual(          # p_ab rounded to 4dp in meta
            p05, stats.hits_at_least(1, m05["n_ab_expected"], m05["p_ab"]),
            places=3)

    def test_thin_statcast_sample_falls_back_to_empirical(self):
        v, ab, w = self._series()
        with patch.object(mlb_starters, "find_player_id",
                          return_value=("12345", False)), \
             patch.object(statcast_asof, "get_rates",
                          return_value={"xba": 0.31, "n_ab": 5}):   # < XSTATS_MIN_N
            p, meta = props._distributional_over_rate(
                "batter_hits", 0.5, v, ab, w, 1.0, 1.0, "Bat",
                "2024-07-01T18:00Z", {}, 0.5)
        self.assertIsNone(meta["xba"])            # xBA not trusted
        self.assertEqual(meta["xba_weight"], 0.0)
        self.assertIsNotNone(p)                    # still returns a prob

    def test_statcast_exception_fails_open(self):
        v, ab, w = self._series()
        with patch.object(mlb_starters, "find_player_id",
                          side_effect=RuntimeError("boom")):
            p, meta = props._distributional_over_rate(
                "batter_hits", 0.5, v, ab, w, 1.0, 1.0, "Bat",
                "2024-07-01T18:00Z", {}, 0.5)
        self.assertIsNotNone(p)                    # empirical fallback, no crash
        self.assertIsNone(meta["xba"])


class BatterQualityIndexTests(unittest.TestCase):
    def _raw(self):
        rows = []
        for d in range(1, 51):                     # 50 hard-hit barrels
            rows.append({"type": "X", "batter": "B1",
                         "game_date": f"2024-06-{d:02d}",
                         "launch_speed": 99.0, "launch_speed_angle": 6})
        for d in range(1, 51):                     # 50 soft non-barrels
            rows.append({"type": "X", "batter": "B1",
                         "game_date": f"2024-06-{d:02d}",
                         "launch_speed": 70.0, "launch_speed_angle": 1})
        rows.append({"type": "S", "batter": "B1",   # non-batted-ball: ignored
                     "game_date": "2024-06-01", "launch_speed": None})
        return rows

    def test_rates_over_batted_balls(self):
        idx = backtest_props.build_batter_quality_index(self._raw())
        q = idx.asof("B1", "2024-07-01")
        self.assertAlmostEqual(q["hard_hit_pct"], 0.5)
        self.assertAlmostEqual(q["barrel_pct"], 0.5)

    def test_leakage_safe_as_of(self):
        idx = backtest_props.build_batter_quality_index(self._raw())
        # As of 2024-06-02, only the 2 balls dated 06-01 are visible (< min_bbe).
        self.assertIsNone(idx.asof("B1", "2024-06-02", min_bbe=40))

    def test_unknown_batter_is_none(self):
        idx = backtest_props.build_batter_quality_index(self._raw())
        self.assertIsNone(idx.asof("NOBODY", "2024-07-01"))


class AsOfWindowMeanTests(unittest.TestCase):
    def _idx(self):
        idx = backtest_props.AsOfIndex()
        for d in range(1, 11):                  # values 1..10 on dates 06-01..10
            idx.add("k", f"2024-06-{d:02d}", float(d))
        return idx

    def test_trailing_window_vs_cumulative(self):
        idx = self._idx()
        self.assertAlmostEqual(idx.asof_mean("k", "2024-07-01", min_bbe=1), 5.5)
        self.assertAlmostEqual(               # last 3 = (8+9+10)/3
            idx.asof_window_mean("k", "2024-07-01", 3, 1), 9.0)

    def test_window_leakage_safe(self):
        # as_of 2024-06-06 sees only dates < 06 (values 1..5); last 2 = (4+5)/2.
        self.assertAlmostEqual(
            self._idx().asof_window_mean("k", "2024-06-06", 2, 1), 4.5)

    def test_window_min_count_gate(self):
        idx = backtest_props.AsOfIndex()
        for d in range(1, 4):
            idx.add("k", f"2024-06-{d:02d}", 1.0)
        self.assertIsNone(
            idx.asof_window_mean("k", "2024-07-01", 50, min_count=40))


class ProjectDistributionalTests(unittest.TestCase):
    def _obs(self, line=0.5, hits=1, ab=4):
        return {"prop_key": "batter_hits", "line": line, "game_date": "2024-07-01",
                "player": "Bat", "stat_label": "H",
                "test_game": {"is_home": True},
                "prior_games": [{"H": hits, "AB": ab, "is_home": True,
                                 "opponent": "X"} for _ in range(12)]}

    def test_not_batter_hits_returns_none(self):
        obs = dict(self._obs(), prop_key="pitcher_strikeouts")
        self.assertIsNone(
            blc.project_distributional(obs, {"half_life": None}, "baseball_mlb"))

    def test_empirical_rate_only(self):
        p = blc.project_distributional(self._obs(hits=1, ab=4),
                                       {"half_life": None}, "baseball_mlb")
        self.assertAlmostEqual(p, 1 - 0.75 ** 4)      # rate .25, n 4, P(>=1)

    def test_xba_index_shifts_prob(self):
        class _Idx:
            def asof_mean(self, pid, as_of, min_bbe=40):
                return 0.35                            # xBA well above .25
        with patch.object(mlb_starters, "find_player_id",
                          return_value=("999", False)):
            base = blc.project_distributional(
                self._obs(), {"half_life": None}, "baseball_mlb",
                xstats_strength=0.0)
            blended = blc.project_distributional(
                self._obs(), {"half_life": None}, "baseball_mlb",
                xba_index=_Idx(), xstats_strength=1.0)
        self.assertGreater(blended, base)              # higher xBA -> higher P

    def test_no_ab_returns_none(self):
        obs = self._obs()
        for g in obs["prior_games"]:
            g["AB"] = None
        self.assertIsNone(
            blc.project_distributional(obs, {"half_life": None}, "baseball_mlb"))

    def test_rolling_window_uses_windowed_lookup(self):
        class _Idx:
            def asof_window_mean(self, pid, as_of, window, min_count=1):
                return 0.40                        # hot rolling window
            def asof_mean(self, pid, as_of, min_bbe=40):
                return 0.20                        # cold season-to-date
        with patch.object(mlb_starters, "find_player_id",
                          return_value=("9", False)):
            season = blc.project_distributional(
                self._obs(), {"half_life": None}, "baseball_mlb",
                xba_index=_Idx(), xstats_strength=1.0)
            rolling = blc.project_distributional(
                self._obs(), {"half_life": None}, "baseball_mlb",
                xba_index=_Idx(), xstats_strength=1.0,
                xba_window=50, xba_min_count=1)
        self.assertGreater(rolling, season)        # 0.40 rolling > 0.20 season

    def test_home_ab_delta_only_affects_home_games(self):
        home = dict(self._obs(), test_game={"is_home": True})
        away = dict(self._obs(), test_game={"is_home": False})
        off = blc.project_distributional(home, {"half_life": None},
                                         "baseball_mlb", home_ab_delta=0.0)
        p_home = blc.project_distributional(home, {"half_life": None},
                                            "baseball_mlb", home_ab_delta=-1.0)
        p_away = blc.project_distributional(away, {"half_life": None},
                                            "baseball_mlb", home_ab_delta=-1.0)
        self.assertLess(p_home, off)      # -1 AB at home -> fewer chances -> lower P
        self.assertEqual(p_away, off)     # away game unaffected by the home delta


class DiagnoseDistributionalTests(unittest.TestCase):
    def _enriched(self):
        rows = []
        for i in range(80):
            actual = i % 3                              # 0,1,2 (line 0.5 => no push)
            rows.append({
                "prop_key": "batter_hits", "line": 0.5, "actual": actual,
                "game_date": f"2024-07-{(i % 28) + 1:02d}",
                "player": f"Bat{i % 9}", "stat_label": "H",
                "test_game": {"is_home": True},
                "prior_games": [{"H": (j % 2), "AB": 4, "is_home": True,
                                 "opponent": "X"} for j in range(12)],
            })
        return rows

    def test_runs_and_reports_without_writing(self):
        cfg = {"batter_hits": {"method": "C", "half_life": None,
                               "venue_strength": 0.0, "opp_defense_strength": 0.0}}
        with patch.object(refit_calibration, "load_calibration", return_value=cfg), \
             patch.object(blc, "harvest_real_line_book_lines",
                          return_value=([{"x": 1}], 1, 0)), \
             patch.object(blc, "join_book_lines_to_actuals",
                          return_value=self._enriched()), \
             patch("savant_history.load_days", return_value=[]), \
             patch.object(mlb_starters, "find_player_id", return_value=None):
            out = io.StringIO()
            with redirect_stdout(out):
                refit_calibration.diagnose_distributional("mlb")
        text = out.getvalue()
        self.assertIn("distributional diagnostic", text)
        self.assertIn("residual ECDF (shipped)", text)   # method C row
        self.assertIn("B pooled Gaussian", text)          # method B row
        self.assertIn("line 0.5", text)                   # bucketed columns
        self.assertIn("Diagnostic only", text)            # footer, no write


class DispatchIntegrationTests(unittest.TestCase):
    """End-to-end: analyze_player_props_value routes a batter_hits calibration
    with method "D" through the distributional path (and only then)."""

    def _prop_data(self):
        return {
            "commence_time": "2026-07-20T18:10:00Z", "home_team": "Home Nine",
            "away_team": "Away Nine", "game_id": "evt1",
            "props": {"batter_hits": {"Bat Man": {
                "line": 0.5, "over_implied": 0.55, "under_implied": 0.45,
                "over_price": -120, "under_price": 100,
                "over_book": "DK", "under_book": "DK"}}},
        }

    def _histories(self):
        dates = [f"2026-07-{d:02d}" for d in range(1, 15)]   # 14 consecutive days
        hits = [1, 0, 1, 2, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1]
        return {"Bat Man": {"batter_hits": {
            "found": True, "values": list(reversed(hits)),
            "game_dates": list(reversed(dates)),
            "at_bats": [4] * 14}}}

    def _run(self, cfg):
        rates = {"xba": 0.30, "hard_hit_pct": 0.42, "barrel_pct": 0.08,
                 "n_ab": 120}
        with patch.object(props, "load_calibration",
                          return_value={"batter_hits": cfg}), \
             patch.object(props, "load_recalibration", return_value={}), \
             patch.object(props, "maybe_auto_refit"), \
             patch.object(props, "log_prediction_rows"), \
             patch.object(props, "log_prediction"), \
             patch.object(mlb_starters, "find_player_id",
                          return_value=("1", False)), \
             patch.object(statcast_asof, "get_rates", return_value=rates):
            cands = props.analyze_player_props_value(
                self._prop_data(), self._histories(), threshold_pct=1.0,
                sport_key="baseball_mlb")
        return cands[0]

    def test_method_D_routes_through_distributional(self):
        cand = self._run({"method": "D", "half_life": None,
                          "xstats_strength": 0.5, "warmup_games": 10})
        self.assertEqual(cand["calibration"]["method"], "D")
        dist = cand["distributional"]
        self.assertIsNotNone(dist)
        self.assertEqual(dist["k"], 1)
        # over_rate (a %) is the binomial survival of the reported p_ab/n.
        self.assertAlmostEqual(
            cand["over_rate"] / 100.0,
            stats.hits_at_least(1, dist["n_ab_expected"], dist["p_ab"]),
            places=2)

    def test_non_D_method_does_not_use_distributional(self):
        cand = self._run({"method": "A", "half_life": None, "warmup_games": 10})
        self.assertIsNone(cand["distributional"])


class UnderHalfSuppressionTests(unittest.TestCase):
    """Model UNDER picks on a batter_hits 0.5 line (~43% realized win rate) are
    demoted from recommendations (is_value forced False), config-gated."""

    def test_suppress_under_helper(self):
        self.assertTrue(props._suppress_under("batter_hits", 0.5))
        self.assertFalse(props._suppress_under("batter_hits", 1.5))
        self.assertFalse(props._suppress_under("batter_strikeouts", 0.5))
        self.assertFalse(props._suppress_under("batter_hits", None))

    def _prop_data(self, line=0.5, over_implied=0.75, under_implied=0.25,
                   dk_under=-110):
        return {
            "commence_time": "2026-07-20T18:00:00Z", "home_team": "H",
            "away_team": "A", "game_id": "e1",
            "props": {"batter_hits": {"Cold Carl": {
                "line": line, "over_implied": over_implied,
                "under_implied": under_implied, "over_price": -110,
                "under_price": dk_under, "over_book": "DK", "under_book": "DK",
                "dk_over_price": -110, "dk_under_price": dk_under,
                "dk_over_book": "DK", "dk_under_book": "DK"}}}}

    def _hist(self, values):
        dates = [f"2026-07-{d:02d}" for d in range(1, 1 + len(values))]
        return {"Cold Carl": {"batter_hits": {
            "found": True, "values": list(values),
            "game_dates": list(reversed(dates))}}}

    def _run(self, prop_data, hist):
        return props.analyze_player_props_value(
            prop_data, hist, threshold_pct=1.0, sport_key=None)[0]

    def test_under_half_value_is_suppressed(self):
        cand = self._run(self._prop_data(), self._hist([0.0] * 15))
        self.assertEqual(cand["direction"], "UNDER")
        self.assertFalse(cand["is_value"])          # demoted from recs
        self.assertTrue(cand["under_suppressed"])

    def test_config_off_restores_value(self):
        # With the filter disabled the same bet IS value -> suppression was the
        # only thing blocking it (proves the filter is what flipped it).
        with patch.object(props, "SUPPRESS_UNDER_MAX_LINE", {}):
            cand = self._run(self._prop_data(), self._hist([0.0] * 15))
        self.assertEqual(cand["direction"], "UNDER")
        self.assertTrue(cand["is_value"])
        self.assertFalse(cand["under_suppressed"])

    def test_over_half_value_unaffected(self):
        cand = self._run(
            self._prop_data(over_implied=0.25, under_implied=0.75),
            self._hist([1.0] * 15))
        self.assertEqual(cand["direction"], "OVER")
        self.assertTrue(cand["is_value"])
        self.assertFalse(cand["under_suppressed"])

    def test_under_at_higher_line_not_suppressed(self):
        cand = self._run(self._prop_data(line=1.5), self._hist([0.0] * 15))
        self.assertEqual(cand["direction"], "UNDER")
        self.assertTrue(cand["is_value"])           # line 1.5 > cap 0.5
        self.assertFalse(cand["under_suppressed"])


class NonHitBatterMarketAnalyzeTests(unittest.TestCase):
    """The new batter_total_bases (line 1.5) and batter_rbis (line 0.5) props
    project via the line-agnostic empirical over-rate (method A / sport_key=None),
    NOT the batter_hits binomial model — so a candidate is produced with the right
    line and distributional stays None."""

    def _prop_data(self, prop, line, over_implied):
        return {
            "commence_time": "2026-07-20T18:00:00Z", "home_team": "H",
            "away_team": "A", "game_id": "e1",
            "props": {prop: {"Slugger Sam": {
                "line": line, "over_implied": over_implied,
                "under_implied": 1.0 - over_implied, "over_price": -110,
                "under_price": -110, "over_book": "DK", "under_book": "DK",
                "dk_over_price": -110, "dk_under_price": -110,
                "dk_over_book": "DK", "dk_under_book": "DK"}}}}

    def _hist(self, prop, values):
        dates = [f"2026-07-{d:02d}" for d in range(1, 1 + len(values))]
        return {"Slugger Sam": {prop: {
            "found": True, "values": list(values),
            "game_dates": list(reversed(dates))}}}

    def _run(self, prop_data, hist):
        return props.analyze_player_props_value(
            prop_data, hist, threshold_pct=1.0, sport_key=None)[0]

    def test_total_bases_over_is_value(self):
        # TB multi-valued per game; ~67% clear the 1.5 line, book implies only 45%.
        cand = self._run(
            self._prop_data("batter_total_bases", 1.5, 0.45),
            self._hist("batter_total_bases", [2.0] * 10 + [1.0] * 5))
        self.assertEqual(cand["prop"], "batter_total_bases")
        self.assertEqual(cand["line"], 1.5)
        self.assertEqual(cand["direction"], "OVER")
        self.assertTrue(cand["is_value"])
        self.assertIsNone(cand["distributional"])   # binomial model is batter_hits-only

    def test_rbis_over_is_value(self):
        cand = self._run(
            self._prop_data("batter_rbis", 0.5, 0.45),
            self._hist("batter_rbis", [1.0] * 10 + [0.0] * 5))
        self.assertEqual(cand["prop"], "batter_rbis")
        self.assertEqual(cand["line"], 0.5)
        self.assertEqual(cand["direction"], "OVER")
        self.assertTrue(cand["is_value"])
        self.assertIsNone(cand["distributional"])


class LineConditionalRuntimeTests(unittest.TestCase):
    """props._method_cfg_for_line resolves the per-line bucket method + sub-cfg."""

    def test_no_line_methods_is_pooled(self):
        cfg = {"method": "C", "residual_mu": 0.1}
        self.assertEqual(props._method_cfg_for_line(cfg, 0.5), ("C", cfg))

    def test_bucket_routing(self):
        cfg = {"method": "C", "line_methods": [
            {"max_line": 0.5, "method": "C", "residual_mu": 0.2},
            {"max_line": None, "method": "D", "xstats_strength": 0.5}]}
        m05, c05 = props._method_cfg_for_line(cfg, 0.5)
        m15, c15 = props._method_cfg_for_line(cfg, 1.5)
        m25, _ = props._method_cfg_for_line(cfg, 2.5)
        self.assertEqual((m05, c05["residual_mu"]), ("C", 0.2))
        self.assertEqual((m15, c15["xstats_strength"]), ("D", 0.5))
        self.assertEqual(m25, "D")                       # open top bucket

    def test_malformed_or_unusable_falls_back(self):
        bad = {"method": "C", "line_methods": [{"max_line": 0.5}]}  # no method
        self.assertEqual(props._method_cfg_for_line(bad, 0.5), ("C", bad))
        cfg = {"method": "C", "line_methods": [{"max_line": None, "method": "D"}]}
        self.assertEqual(props._method_cfg_for_line(cfg, None)[0], "C")  # bad line

    def test_inherited_bucket_merges_pooled_residuals(self):
        # A bare inherited bucket {max_line, method} (the real written shape when a
        # SIBLING bucket adopts) must reuse the pooled residuals + warmup, else
        # method C collapses to a constant 0.5. The merge {**pooled, **bucket}
        # supplies them.
        cfg = {"method": "C", "residual_mu": 0.04, "residual_sigma": 0.87,
               "residual_ecdf": [-1.0, 0.0, 1.0], "warmup": {"method": "A"},
               "warmup_games": 10, "line_methods": [
                   {"max_line": 0.5, "method": "C"},              # bare inherit
                   {"max_line": None, "method": "D", "xstats_strength": 0.5}]}
        _, c05 = props._method_cfg_for_line(cfg, 0.5)
        self.assertEqual(c05["residual_ecdf"], [-1.0, 0.0, 1.0])  # pooled residuals
        self.assertEqual(c05["residual_mu"], 0.04)
        self.assertEqual(c05["warmup"], {"method": "A"})          # pooled warmup

    def test_inherited_bucket_calibrates_like_pooled_not_flat(self):
        import calibration_loader as cl
        pooled = {"method": "C", "residual_mu": 0.04, "residual_sigma": 0.87,
                  "residual_ecdf": sorted([-1.5, -0.5, 0.0, 0.3, 0.9, 1.4]),
                  "warmup_games": 10}
        cfg = dict(pooled, line_methods=[
            {"max_line": 0.5, "method": "C"},
            {"max_line": None, "method": "D", "xstats_strength": 0.5}])
        _, c05 = props._method_cfg_for_line(cfg, 0.5)
        p_pooled = cl.apply_calibration_with_warmup(pooled, 1.0, 0.5, 20,
                                                    empirical_over=0.6)
        p_bucket = cl.apply_calibration_with_warmup(c05, 1.0, 0.5, 20,
                                                    empirical_over=0.6)
        self.assertIsNotNone(p_bucket)
        self.assertNotAlmostEqual(p_bucket, 0.5)          # NOT the collapse bug
        self.assertAlmostEqual(p_bucket, p_pooled)        # inherits pooled calib


class SelectMethodDCandidateTests(unittest.TestCase):
    """select_method_at_real_lines admits D only when rows carry p_dist."""

    def _rows(self, with_pdist):
        rows = []
        for i in range(160):
            actual = 1 if i % 2 == 0 else 2          # under/over the 1.5 line
            o = 1 if actual > 1.5 else 0
            r = {"player": f"P{i % 20}", "projected": 1.5, "line": 1.5,
                 "actual": actual, "empirical_over": 0.5,
                 "game_date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"}
            if with_pdist:
                r["p_dist"] = 0.98 if o else 0.02      # near-perfect
            rows.append(r)
        return rows

    def test_d_selected_when_pdist_present_and_wins(self):
        sel = blc.select_method_at_real_lines(self._rows(True))
        self.assertEqual(sel["method"], "D")
        self.assertTrue(sel["confirmed"])
        self.assertIn("D", sel["single_split"])

    def test_backcompat_no_pdist_no_d(self):
        sel = blc.select_method_at_real_lines(self._rows(False))
        self.assertNotIn("D", sel["single_split"])       # D not scored
        self.assertEqual(sel["method"], "A")             # nothing beats empirical


class SelectLineMethodsTests(unittest.TestCase):
    """refit_calibration._select_line_methods adopts a bucket method only when a
    deep-enough bucket's gate-confirmed winner beats the pooled method there."""

    def _enriched(self, n_top):
        obs = []
        for i in range(50):        # line-0.5 bucket (small -> inherits pooled)
            obs.append({"prop_key": "batter_hits", "line": 0.5,
                        "actual": i % 2, "player": "P",
                        "game_date": f"2026-01-{1 + i % 27:02d}",
                        "_proj": 0.5, "_emp": 0.5, "_pdist": 0.5})
        for i in range(n_top):     # line-1.5 bucket where D is near-perfect
            actual = 1 if i % 2 == 0 else 2
            o = 1 if actual > 1.5 else 0
            obs.append({"prop_key": "batter_hits", "line": 1.5, "actual": actual,
                        "player": f"P{i % 20}",
                        "game_date": f"2026-{2 + i // 28:02d}-{1 + i % 28:02d}",
                        "_proj": 1.5, "_emp": 0.5, "_pdist": 0.98 if o else 0.02})
        return obs

    def _run(self, enriched, pooled="C"):
        def _pe(o, params, sk, td=None, la=None):
            return o["_proj"], o["_emp"]
        def _pd(o, params, sk, td=None, la=None, xba_index=None,
                quality_index=None, xstats_strength=0.0):
            return o["_pdist"]
        with patch.object(blc, "project_and_empirical", side_effect=_pe), \
             patch.object(blc, "project_distributional", side_effect=_pd):
            return refit_calibration._select_line_methods(
                "batter_hits", enriched, {}, "baseball_mlb", {}, None,
                pooled, object(), object())

    def test_deep_bucket_adopts_d(self):
        lm = self._run(self._enriched(n_top=160))
        self.assertIsNotNone(lm)
        self.assertEqual(len(lm), 2)
        self.assertEqual(lm[0]["max_line"], 0.5)
        self.assertEqual(lm[0]["method"], "C")           # small 0.5 bucket inherits
        self.assertIsNone(lm[1]["max_line"])
        self.assertEqual(lm[1]["method"], "D")           # deep top bucket flips
        self.assertTrue(lm[1]["confirmed"])
        self.assertEqual(lm[1]["xstats_strength"],
                         refit_calibration.LINE_COND_XSTATS_STRENGTH)

    def test_thin_bucket_returns_none(self):
        self.assertIsNone(self._run(self._enriched(n_top=40)))   # < MIN_BUCKET_OBS

    def test_bucket_ready_gate(self):
        deep = self._enriched(n_top=160)
        thin = self._enriched(n_top=40)
        self.assertTrue(refit_calibration._lc_bucket_ready(deep, {"batter_hits"}))
        self.assertFalse(refit_calibration._lc_bucket_ready(thin, {"batter_hits"}))
        self.assertFalse(refit_calibration._lc_bucket_ready(deep, {"pitcher_outs"}))

    def test_pooled_E_does_not_drop_confirmed_override(self):
        """Audit finding #1 regression: when the POOLED method is E, the bucket
        selector must still score E per-bucket (negbin_eligible threaded) so the
        deep bucket's confirmed D winner is adopted. Without the flag, single.get
        ('E') is None -> the adopt guard fails for every bucket -> line_methods
        returns None -> the merge SILENTLY DROPS a live override."""
        lm = self._run(self._enriched(n_top=160), pooled="E")
        self.assertIsNotNone(lm)                       # NOT dropped
        self.assertEqual(lm[1]["method"], "D")         # deep bucket keeps its winner
        self.assertTrue(lm[1]["confirmed"])

    def test_bucket_adopts_E_stores_negbin_params(self):
        """A bucket that adopts method E persists mean_scale + dispersion (and no
        residual block), so the runtime `_method_cfg_for_line` merge can dispatch
        it. Force an E winner via the selector to keep the assertion deterministic."""
        e_win = {
            "method": "E", "confirmed": True, "n_obs": 160,
            "fit_brier": 0.10, "baseline_brier": 0.25, "cv_brier": 0.10,
            "mean_scale": 1.05, "dispersion": 0.30,
            "single_split": {"A": 0.25, "C": 0.24, "E": 0.10},
        }

        def _pe(o, params, sk, td=None, la=None):
            return o["_proj"], o["_emp"]

        def _pd(o, params, sk, td=None, la=None, xba_index=None,
                quality_index=None, xstats_strength=0.0):
            return o["_pdist"]

        with patch.object(blc, "project_and_empirical", side_effect=_pe), \
             patch.object(blc, "project_distributional", side_effect=_pd), \
             patch.object(blc, "select_method_at_real_lines", return_value=e_win):
            lm = refit_calibration._select_line_methods(
                "batter_hits", self._enriched(n_top=160), {}, "baseball_mlb",
                {}, None, "C", object(), object())
        self.assertIsNotNone(lm)
        top = lm[1]
        self.assertEqual(top["method"], "E")
        self.assertEqual(top["mean_scale"], 1.05)
        self.assertEqual(top["dispersion"], 0.30)
        self.assertNotIn("residual_ecdf", top)         # E carries no residual block


class LineupGatingTests(unittest.TestCase):
    """§2.5A: a confirmed-OUT player's prop is demoted from recommendations AND
    skipped from the calibration log (the label would never resolve); "in" and
    "unknown" leave both behaviors unchanged. The status rides in on `history`,
    exactly like `batting_order`."""

    def _prop_data(self):
        # over_implied LOW so a strong hitter's OVER clears the value gate (mirrors
        # UnderHalfSuppressionTests.test_over_half_value_unaffected).
        return {
            "commence_time": "2026-07-20T18:00:00Z", "home_team": "H",
            "away_team": "A", "game_id": "e1",
            "props": {"batter_hits": {"Hot Hal": {
                "line": 0.5, "over_implied": 0.25, "under_implied": 0.75,
                "over_price": -110, "under_price": -110,
                "over_book": "DK", "under_book": "DK",
                "dk_over_price": -110, "dk_under_price": -110,
                "dk_over_book": "DK", "dk_under_book": "DK"}}},
        }

    def _hist(self, status):
        dates = [f"2026-07-{d:02d}" for d in range(1, 16)]
        h = {"found": True, "values": [1.0] * 15,
             "game_dates": list(reversed(dates))}
        if status is not None:
            h["lineup_status"] = status
        return {"Hot Hal": {"batter_hits": h}}

    def _run(self, status, sport_key="baseball_mlb"):
        rates = {"xba": 0.30, "hard_hit_pct": 0.42, "barrel_pct": 0.08,
                 "n_ab": 120}
        with patch.object(props, "load_calibration", return_value={}), \
             patch.object(props, "load_recalibration", return_value={}), \
             patch.object(props, "maybe_auto_refit"), \
             patch.object(props, "log_prediction_rows"), \
             patch.object(props, "log_prediction") as log, \
             patch.object(mlb_starters, "find_player_id",
                          return_value=("1", False)), \
             patch.object(statcast_asof, "get_rates", return_value=rates):
            cand = props.analyze_player_props_value(
                self._prop_data(), self._hist(status), threshold_pct=1.0,
                sport_key=sport_key)[0]
        return cand, log

    def test_out_demotes_and_skips_log(self):
        cand, log = self._run("out")
        self.assertEqual(cand["direction"], "OVER")
        self.assertFalse(cand["is_value"])          # demoted from recs
        self.assertEqual(cand["lineup_status"], "out")
        log.assert_not_called()                      # no corrupt label logged

    def test_unknown_is_value_and_logged(self):
        cand, log = self._run("unknown")
        self.assertTrue(cand["is_value"])            # fail open
        self.assertEqual(cand["lineup_status"], "unknown")
        log.assert_called()

    def test_in_is_value_and_logged(self):
        cand, log = self._run("in")
        self.assertTrue(cand["is_value"])
        self.assertEqual(cand["lineup_status"], "in")
        log.assert_called()

    def test_gate_disabled_restores_value(self):
        # With MLB removed from the gate map the same "out" is ignored -> the gate
        # was the only thing flipping value (and logging resumes).
        with patch.object(props, "PLAYER_PROP_LINEUP_GATING", {}):
            cand, log = self._run("out")
        self.assertTrue(cand["is_value"])
        log.assert_called()


if __name__ == "__main__":
    unittest.main()
