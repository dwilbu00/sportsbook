"""Hermetic tests for the §2.6 candidate-feature harness (rest / days-off).

Four layers, all pure stdlib (no live ESPN / Statcast / SQL / secrets):
  * prop_features math — rest_multiplier (cadence adaptation, bounds, leakage,
    strength-0 no-op), the registry helpers, projection_multiplier's applies-to.
  * projection threading — the REAL book_line_calibration.project_and_empirical
    with vs without a feature: strength 0 is byte-identical to production, an
    on-strength shifts BOTH the projection (methods B/C/E) and the empirical line
    (method A), and an off-registry prop is a hard no-op.
  * sweep round-trip — a 6-part /rest label parses, is non-baseline when on and
    baseline when off, persists rest_strength through _build_prop_cfg, and the
    backtest grid carries the rest axis with rest0.0 inserted before its rest1.0
    twin (the tie-break that keeps excluded props at 0).
  * diagnose_features — end-to-end, hermetic: per-prop×strength Brier table, the
    incumbent gain + gate verdict, the pick-flip line, the consensus-ROI column,
    and the invariant that a DIAGNOSTIC writes nothing.

Run: PYTHONIOENCODING=utf-8 python test_feature_diag.py
"""

import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import backtest
import book_line_calibration as blc
import prop_features as pf
import refit_calibration as rc


# ── prop_features math ───────────────────────────────────────────────────────
class RestMultiplierTests(unittest.TestCase):
    def test_strength_zero_is_noop(self):
        dates = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23"]
        self.assertEqual(pf.rest_multiplier(dates, "2025-06-01", 0.0), 1.0)
        self.assertEqual(pf.rest_multiplier(dates, "2025-06-01", None), 1.0)

    def test_too_few_prior_dates_is_noop(self):
        # < REST_MIN_PRIOR (3) usable dates -> no stable cadence -> 1.0.
        self.assertEqual(
            pf.rest_multiplier(["2025-05-20", "2025-05-25"], "2025-06-01", 1.0),
            1.0)

    def test_on_cadence_is_neutral(self):
        # A 5-day-cadence starter resting exactly 5 days -> delta 0 -> mult 1.0.
        dates = ["2025-05-01", "2025-05-06", "2025-05-11", "2025-05-16"]
        self.assertAlmostEqual(
            pf.rest_multiplier(dates, "2025-05-21", 1.0), 1.0, places=9)

    def test_more_rest_than_cadence_boosts(self):
        # Daily cadence (gap 1), 6 days off -> delta capped at +4 -> mult > 1,
        # hard-capped at 1 + REST_CAP.
        dates = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23",
                 "2025-05-24"]
        m = pf.rest_multiplier(dates, "2025-05-30", 1.0)
        self.assertGreater(m, 1.0)
        self.assertLessEqual(m, 1.0 + pf.REST_CAP + 1e-12)

    def test_less_rest_than_cadence_dampens(self):
        # 5-day cadence but only 1 day of rest -> delta -4 -> mult < 1.
        dates = ["2025-05-01", "2025-05-06", "2025-05-11", "2025-05-16"]
        m = pf.rest_multiplier(dates, "2025-05-17", 1.0)
        self.assertLess(m, 1.0)
        self.assertGreaterEqual(m, 1.0 - pf.REST_CAP - 1e-12)

    def test_cadence_self_adapts(self):
        # SAME 5 days of rest reads as "on schedule" for a 5-day starter but as a
        # big layoff for a daily batter -> the one form adapts with no position
        # logic. graded is 5 days after each player's last game.
        starter = ["2025-05-01", "2025-05-06", "2025-05-11", "2025-05-16"]
        batter = ["2025-05-12", "2025-05-13", "2025-05-14", "2025-05-15",
                  "2025-05-16"]
        m_starter = pf.rest_multiplier(starter, "2025-05-21", 1.0)  # delta 0
        m_batter = pf.rest_multiplier(batter, "2025-05-21", 1.0)    # delta +4
        self.assertAlmostEqual(m_starter, 1.0, places=9)
        self.assertGreater(m_batter, m_starter)

    def test_strength_scales_effect(self):
        dates = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23",
                 "2025-05-24"]
        half = pf.rest_multiplier(dates, "2025-05-30", 0.5)
        full = pf.rest_multiplier(dates, "2025-05-30", 1.0)
        # both boost; the stronger knob boosts at least as much (cap permitting).
        self.assertGreater(half, 1.0)
        self.assertGreaterEqual(full, half)

    def test_future_and_equal_dates_ignored_no_leakage(self):
        # A prior date == graded and one AFTER graded must not feed the estimate:
        # the result must equal the same history with those rows removed.
        base = ["2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04"]
        leaky = base + ["2025-06-10", "2025-07-01"]   # == graded, and future
        self.assertEqual(
            pf.rest_multiplier(leaky, "2025-06-10", 1.0),
            pf.rest_multiplier(base, "2025-06-10", 1.0))

    def test_bad_graded_date_is_noop(self):
        dates = ["2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04"]
        self.assertEqual(pf.rest_multiplier(dates, "not-a-date", 1.0), 1.0)
        self.assertEqual(pf.rest_multiplier(dates, None, 1.0), 1.0)

    def test_datetime_strings_parsed(self):
        # ESPN game_date is ISO datetime; only the date part should matter.
        dates = ["2025-05-20T18:05Z", "2025-05-21T18:05Z", "2025-05-22T18:05Z",
                 "2025-05-23T18:05Z", "2025-05-24T18:05Z"]
        m = pf.rest_multiplier(dates, "2025-05-30T23:05Z", 1.0)
        self.assertGreater(m, 1.0)


class RegistryTests(unittest.TestCase):
    def test_feature_applies_only_to_registered_props(self):
        for pk in ("pitcher_outs", "pitcher_strikeouts", "batter_hits"):
            self.assertTrue(pf.feature_applies("rest", pk))
        for pk in ("pitcher_earned_runs", "batter_strikeouts", "player_points"):
            self.assertFalse(pf.feature_applies("rest", pk))
        self.assertFalse(pf.feature_applies("no_such_feature", "batter_hits"))

    def test_strengths_from_params_reads_knob_and_map(self):
        self.assertEqual(pf.strengths_from_params({"rest_strength": 0.5}),
                         {"rest": 0.5})
        self.assertEqual(pf.strengths_from_params({"features": {"rest": 1.0}}),
                         {"rest": 1.0})
        # explicit 0 in the map removes it (off), even if the knob set it.
        self.assertEqual(
            pf.strengths_from_params({"rest_strength": 1.0,
                                      "features": {"rest": 0.0}}),
            {})
        self.assertEqual(pf.strengths_from_params({}), {})
        self.assertEqual(pf.strengths_from_params({"rest_strength": 0.0}), {})

    def test_projection_multiplier_applies_to_filter(self):
        dates = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23",
                 "2025-05-24"]
        on = pf.projection_multiplier("batter_hits", {"rest": 1.0}, dates,
                                      "2025-05-30")
        self.assertGreater(on, 1.0)
        # excluded prop: hard no-op even at full strength.
        off = pf.projection_multiplier("pitcher_earned_runs", {"rest": 1.0},
                                       dates, "2025-05-30")
        self.assertEqual(off, 1.0)
        # empty strengths -> 1.0.
        self.assertEqual(
            pf.projection_multiplier("batter_hits", {}, dates, "2025-05-30"), 1.0)


# ── projection threading (real project_and_empirical) ────────────────────────
def _obs(prop_key, stat_label, line, graded, prior_dates, values):
    prior_games = [{"game_date": d, stat_label: v, "MIN": 0.0,
                    "is_home": None, "opponent": None}
                   for d, v in zip(prior_dates, values)]
    return {"prop_key": prop_key, "stat_label": stat_label, "line": line,
            "game_date": graded, "prior_games": prior_games,
            "test_game": {"is_home": None}}


class ProjectionThreadingTests(unittest.TestCase):
    # daily cadence + a 7-day layoff -> mult hits the +REST_CAP ceiling.
    DATES = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23",
             "2025-05-24", "2025-05-25", "2025-05-26", "2025-05-27",
             "2025-05-28", "2025-05-29"]
    VALUES = [1, 2, 1, 2, 3, 2, 1, 2, 3, 2]        # mean 1.9; straddles line 2.0
    GRADED = "2025-06-05"

    def _pe(self, features=None, prop_key="batter_hits", stat_label="H",
            line=2.0):
        params = {"half_life": None}
        if features is not None:
            params["features"] = features
        obs = _obs(prop_key, stat_label, line, self.GRADED, self.DATES,
                   self.VALUES)
        return blc.project_and_empirical(obs, params, "baseball_mlb")

    def test_no_features_key_is_production(self):
        base = self._pe(None)
        zero = self._pe({"rest": 0.0})
        self.assertEqual(base, zero)   # strength 0 == no-features == production

    def test_on_strength_shifts_projection_and_empirical(self):
        proj0, emp0 = self._pe(None)
        proj1, emp1 = self._pe({"rest": 1.0})
        exp_mult = pf.rest_multiplier(self.DATES, self.GRADED, 1.0)
        self.assertGreater(exp_mult, 1.0)
        # projection scaled by the multiplier (moves methods B/C/E) ...
        self.assertAlmostEqual(proj1, proj0 * exp_mult, places=9)
        # ... and the empirical over-rate rises because the effective line
        # (line / mult) drops below 2.0, so the 2-hit games now count as overs
        # (moves method A). This is the exact split production uses.
        self.assertGreater(emp1, emp0)

    def test_excluded_prop_is_noop_end_to_end(self):
        base = self._pe(None, prop_key="pitcher_earned_runs", stat_label="ER")
        on = self._pe({"rest": 1.0}, prop_key="pitcher_earned_runs",
                      stat_label="ER")
        self.assertEqual(base, on)


# ── sweep round-trip ─────────────────────────────────────────────────────────
class SweepRoundTripTests(unittest.TestCase):
    def test_parse_6part_rest_token(self):
        p = rc._parse_variant_name(
            "hl15/opp0.5/defadj1.0/shrink5/ven0.25/rest1.0")
        self.assertEqual(p["half_life"], 15)
        self.assertEqual(p["opp_defense_strength"], 0.5)
        self.assertEqual(p["output_def_strength"], 1.0)
        self.assertEqual(p["shrink_k"], 5.0)
        self.assertEqual(p["venue_strength"], 0.25)
        self.assertEqual(p["rest_strength"], 1.0)

    def test_legacy_labels_default_rest_off(self):
        for name in ("hl15/defadj1.0/ven0.25",
                     "hl15/opp0.5/defadj1.0/shrink5/ven0.25"):
            self.assertEqual(rc._parse_variant_name(name)["rest_strength"], 0.0)

    def test_bad_rest_token_rejected(self):
        self.assertIsNone(rc._parse_variant_name(
            "hl15/opp0.5/defadj1.0/shrink5/ven0.25/zzz1.0"))

    def test_baseline_recognizes_rest_off_and_rejects_rest_on(self):
        self.assertTrue(rc._is_baseline_variant(
            "none/opp0.0/defadj0.0/shrink0/ven0.0/rest0.0"))
        self.assertFalse(rc._is_baseline_variant(
            "none/opp0.0/defadj0.0/shrink0/ven0.0/rest1.0"))

    def test_build_prop_cfg_persists_rest_strength(self):
        vname = "none/opp0.0/defadj0.0/shrink0/ven0.0/rest1.0"
        # calib_obs schema: (player, projected, line, actual, empirical, date)
        obs = [("p", 1.0 + (i % 3) * 0.1, 1.5, i % 2, 0.5,
                f"2026-05-{1 + i:02d}") for i in range(30)]
        results = {vname: {"batter_hits": {"calib_obs": obs}}}
        winner = {"variant": vname, "method": "A", "brier": 0.24, "hit": 55.0,
                  "baseline_brier": 0.25, "cv_brier": 0.245, "confirmed": False,
                  "variant_confirmed": True}
        cfg = rc._build_prop_cfg(winner, results, "batter_hits", 0)
        self.assertEqual(cfg["rest_strength"], 1.0)
        self.assertEqual(cfg["variant_label"], vname)

    def test_backtest_grid_has_rest_axis_ordered(self):
        grid = backtest._build_props_sweep_grid()
        self.assertEqual(len(grid), 576)                 # 4×3×3×4×2×2
        keys = list(grid)
        for k in keys:
            self.assertEqual(len(k.split("/")), 6)
            self.assertIn("/rest", k)
        base0 = "none/opp0.0/defadj0.0/shrink0/ven0.0/rest0.0"
        base1 = "none/opp0.0/defadj0.0/shrink0/ven0.0/rest1.0"
        self.assertIn(base0, grid)
        self.assertIn(base1, grid)
        # rest0.0 twin inserted immediately before rest1.0 -> strict-`<` tie-break
        # keeps an excluded prop's duplicate cells at rest_strength 0.0.
        self.assertLess(keys.index(base0), keys.index(base1))
        self.assertEqual(grid[base0]["rest_strength"], 0.0)
        self.assertEqual(grid[base1]["rest_strength"], 1.0)
        self.assertTrue(rc._is_baseline_variant(base0))


# ── diagnose_features: end-to-end, hermetic ──────────────────────────────────
def _fake_build(enriched, params, *a, **k):
    """Rows tagged with the injected strength so the fake selector can vary."""
    s = (params.get("features") or {}).get("rest", 0.0)
    return [{"_s": s}]


def _fake_select(rows, **k):
    """strength 0 -> incumbent E middling, gate pick A; strength > 0 -> E improves
    past the 0.002 gate AND becomes the pick (a flip)."""
    s = rows[0]["_s"] if rows else 0.0
    if s == 0.0:
        return {"method": "A",
                "single_split": {"A": 0.250, "B": 0.248, "E": 0.246}}
    return {"method": "E",
            "single_split": {"A": 0.249, "B": 0.245, "E": 0.240}}


def _fake_roi(*a, **k):
    return {"A": {"roi": 0.03, "n_bets": 30, "hit": 0.5, "avg_edge": 0.07},
            "E": {"roi": 0.11, "n_bets": 28, "hit": 0.6, "avg_edge": 0.09}}


class DiagnoseFeaturesEndToEndTests(unittest.TestCase):
    def _run(self, existing, feature=None, prop_filter=None):
        buf = io.StringIO()
        with patch.object(rc, "load_calibration", return_value=existing), \
             patch.object(rc, "save_calibration") as save_mock, \
             patch.object(rc, "_roi_by_method", side_effect=_fake_roi), \
             patch.object(blc, "harvest_real_line_book_lines",
                          return_value=([{}], 1, 0)), \
             patch.object(blc, "join_book_lines_to_actuals",
                          return_value=[{"prop_key": "batter_hits"}]), \
             patch.object(blc, "build_real_line_obs", side_effect=_fake_build), \
             patch.object(blc, "select_method_at_real_lines",
                          side_effect=_fake_select), \
             redirect_stdout(buf):
            rc.diagnose_features("mlb", feature=feature, prop_filter=prop_filter)
        return buf.getvalue(), save_mock

    def test_end_to_end_table_gate_roi_and_no_write(self):
        out, save_mock = self._run({"batter_hits": {"method": "E",
                                                    "half_life": None}})
        self.assertIn("Candidate-feature diagnostic", out)
        self.assertIn("holdout Brier by method", out)
        self.assertIn("incumbent=E", out)
        # gate verdict: E improves 0.246 -> 0.240 = +0.006 >= 0.002 -> YES.
        self.assertIn("gate: YES", out)
        # the selection flips A -> E when the feature is on.
        self.assertIn("selection would change", out)
        # consensus-ROI co-signal column.
        self.assertIn("consensus-ROI", out)
        self.assertIn("ROI%", out)
        # the invariant: a diagnostic writes NOTHING.
        self.assertIn("nothing written", out)
        save_mock.assert_not_called()

    def test_off_strength_row_equals_production_selfcheck(self):
        # The strength-0.00 Brier row must equal the all-off selection (the
        # built-in "strength 0 == production" check). _fake_select returns the
        # 0.250/0.248/0.246 row for s==0.
        out, _ = self._run({"batter_hits": {"method": "E", "half_life": None}})
        self.assertIn("0.246", out)   # incumbent E at strength 0 (production)

    def test_unknown_feature_reports_and_returns(self):
        out, save_mock = self._run({"batter_hits": {"method": "E"}},
                                   feature="teleport")
        self.assertIn("Unknown feature", out)
        save_mock.assert_not_called()

    def test_no_applicable_props_short_circuits(self):
        # only an excluded prop is calibrated -> no registered feature applies.
        out, save_mock = self._run({"pitcher_earned_runs": {"method": "A"}})
        self.assertIn("No calibrated props", out)
        save_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
