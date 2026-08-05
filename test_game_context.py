"""§3.1 GameContext — the shared, cached, leakage-safe per-game run/hits object
plus its one gate-facing per-batter mean scalar (``gc_factor``).

Covers Phase 1 (inert foundation) only: the ``_run_pmf`` byte-parity refactor,
the run/total joint math, and ``build_game_context`` / ``gamecontext_factor``
(leakage cutoff, fail-open completeness, cap bounds, cache). All hermetic — the
network sources (``get_team_index`` / ``get_expected_runs_team_factors``) are
patched, and the LEAKY season-to-date sources are patched to explode if the
build path ever touches them.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

import book_line_calibration as blc
import mlb_starters
import prop_features as pf


# ── byte-parity reference recursions (mirror the documented _run_pmf formula) ──
def _ref_poisson_pmf(mu, max_runs=30):
    values = [math.exp(-mu)]
    for score in range(1, max_runs + 1):
        values.append(values[-1] * mu / score)
    values[-1] += max(0.0, 1.0 - sum(values))
    return values


def _ref_negbin_pmf(mu, dispersion, max_runs=30):
    size = 1.0 / dispersion
    success = size / (size + mu)
    failure = 1.0 - success
    values = [success ** size]
    for score in range(1, max_runs + 1):
        values.append(values[-1] * (score - 1.0 + size) / score * failure)
    values[-1] += max(0.0, 1.0 - sum(values))
    return values


def _fake_inputs(**over):
    """A complete get_expected_runs_team_factors payload (HOM vs AWY)."""
    base = {
        "league_xwoba": 0.320,
        "league_bullpen_xwoba": 0.315,
        "offense_vs_hand": {
            "L": {"HOM": 0.340, "AWY": 0.300},
            "R": {"HOM": 0.345, "AWY": 0.305},
        },
        "bullpen_xwoba": {"HOM": 0.330, "AWY": 0.300},
    }
    base.update(over)
    return base


def _fake_match(team_name, team_index):
    return {"HomeTown": {"abbr": "HOM"},
            "AwayTown": {"abbr": "AWY"}}.get(team_name)


class RunPmfParityTest(unittest.TestCase):
    """_run_pmf must reproduce the old spread inner arithmetic exactly."""

    def test_poisson_pmf_byte_identical(self):
        for mu in (0.5, 3.1, 4.3, 8.0):
            got = mlb_starters._run_pmf(mu, dispersion=0.0, max_runs=30)
            ref = _ref_poisson_pmf(mu, max_runs=30)
            self.assertEqual(got, ref, f"poisson mismatch at mu={mu}")

    def test_negbin_pmf_byte_identical(self):
        for mu in (3.1, 4.3, 8.0):
            for disp in (0.05, 0.2):
                got = mlb_starters._run_pmf(mu, dispersion=disp, max_runs=30)
                ref = _ref_negbin_pmf(mu, disp, max_runs=30)
                self.assertEqual(got, ref, f"negbin mismatch mu={mu} d={disp}")

    def test_pmf_sums_to_one(self):
        for mu in (0.5, 4.3, 8.0):
            self.assertAlmostEqual(sum(mlb_starters._run_pmf(mu)), 1.0, places=12)
            self.assertAlmostEqual(
                sum(mlb_starters._run_pmf(mu, dispersion=0.2)), 1.0, places=12)

    def test_terminal_bucket_absorbs_tail(self):
        # A tiny max_runs forces the tail into the last bucket; still sums to 1.
        pmf = mlb_starters._run_pmf(8.0, max_runs=3)
        self.assertEqual(len(pmf), 4)
        self.assertAlmostEqual(sum(pmf), 1.0, places=12)
        self.assertGreater(pmf[-1], 0.0)

    def test_margin_probability_still_consistent(self):
        # The refactored poisson_margin_probability equals a brute-force sum over
        # the very PMFs _run_pmf returns (i.e. the refactor changed nothing).
        h = mlb_starters._run_pmf(4.3, max_runs=30)
        a = mlb_starters._run_pmf(4.0, max_runs=30)
        brute = sum(hp * ap
                    for hs, hp in enumerate(h)
                    for as_, ap in enumerate(a)
                    if hs + 0.5 > as_)
        got = mlb_starters.poisson_margin_probability(4.3, 4.0, 0.5, max_runs=30)
        self.assertAlmostEqual(got, brute, places=12)


class JointMathTest(unittest.TestCase):
    def test_convolve_moments_are_additive(self):
        h = mlb_starters._run_pmf(4.3, max_runs=30)
        a = mlb_starters._run_pmf(4.0, max_runs=30)
        total = mlb_starters._convolve_pmf(h, a)
        self.assertAlmostEqual(sum(total), 1.0, places=10)
        mean, std = mlb_starters._pmf_moments(total)
        # Independent Poissons: total mean ≈ 8.3, total var ≈ 8.3.
        self.assertAlmostEqual(mean, 8.3, delta=0.01)
        self.assertAlmostEqual(std, math.sqrt(8.3), delta=0.02)

    def test_pmf_moments_known_distribution(self):
        pmf = [0.25, 0.25, 0.25, 0.25]  # uniform on 0..3
        mean, std = mlb_starters._pmf_moments(pmf)
        self.assertAlmostEqual(mean, 1.5, places=12)
        self.assertAlmostEqual(std, math.sqrt(1.25), places=12)


class GcFactorTermsTest(unittest.TestCase):
    def test_forms_and_bounds(self):
        # own_off high, opposing bullpen weak -> factor pushed to the upper cap.
        hi = mlb_starters._gc_factor_from_terms(2.0, 0.5, "full")
        self.assertAlmostEqual(hi, 1.0 + mlb_starters.GC_RUN_CAP, places=12)
        # own_off low, opposing bullpen strong -> lower cap.
        lo = mlb_starters._gc_factor_from_terms(0.5, 2.0, "full")
        self.assertAlmostEqual(lo, 1.0 - mlb_starters.GC_RUN_CAP, places=12)
        # own-only ablation ignores the bullpen term entirely.
        own = mlb_starters._gc_factor_from_terms(1.10, 5.0, "own")
        self.assertAlmostEqual(own, 1.10 ** mlb_starters.GC_EPS, places=12)
        # opp-only ablation ignores own offense entirely.
        opp = mlb_starters._gc_factor_from_terms(5.0, 1.05, "opp")
        eff = (mlb_starters.GC_STARTER_SHARE
               + (1 - mlb_starters.GC_STARTER_SHARE) * 1.05)
        self.assertAlmostEqual(opp, (1.0 / eff) ** mlb_starters.GC_EPS, places=12)

    def test_missing_terms_fail_open(self):
        self.assertEqual(mlb_starters._gc_factor_from_terms(None, 1.05, "full"), 1.0)
        self.assertEqual(mlb_starters._gc_factor_from_terms(1.1, None, "full"), 1.0)
        self.assertEqual(mlb_starters._gc_factor_from_terms(None, 1.05, "own"), 1.0)
        self.assertEqual(mlb_starters._gc_factor_from_terms(1.1, None, "opp"), 1.0)


class BuildGameContextTest(unittest.TestCase):
    def setUp(self):
        mlb_starters._GC_CACHE.clear()

    def _build(self, inputs_ret, season=2024, date="2024-07-01"):
        eri = MagicMock(return_value=inputs_ret)
        boom = MagicMock(side_effect=AssertionError("LEAK: season-to-date source"))
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "_match_team_id", side_effect=_fake_match), \
             patch.object(mlb_starters, "get_expected_runs_team_factors", eri), \
             patch.object(mlb_starters, "get_team_offense_splits", boom), \
             patch.object(mlb_starters, "get_team_bullpen_quality", boom), \
             patch.object(mlb_starters, "get_pitcher_quality", boom):
            ctx = mlb_starters.build_game_context(
                "HomeTown", "AwayTown", date, season)
        return ctx, eri

    def test_complete_build_is_leakage_safe_and_bounded(self):
        ctx, eri = self._build(_fake_inputs())
        self.assertTrue(ctx["complete"])
        # Only the as-of source was consulted, with as_of == the game date.
        self.assertEqual(eri.call_args.args[0], 2024)
        self.assertEqual(eri.call_args.args[1], "2024-07-01")
        # gc_factor present per side/form and inside the hard cap band.
        lo, hi = 1.0 - mlb_starters.GC_RUN_CAP, 1.0 + mlb_starters.GC_RUN_CAP
        for side in ("home", "away"):
            for form in ("full", "own", "opp"):
                self.assertTrue(lo <= ctx["gc_factor"][side][form] <= hi)
        # Foundation fields populated and internally consistent.
        self.assertEqual(len(ctx["run_pmf"]["home"]), 31)
        self.assertAlmostEqual(sum(ctx["total_pmf"]), 1.0, places=10)
        self.assertGreater(ctx["team_hits_mean"]["home"], 0.0)
        conv_mean, _ = mlb_starters._pmf_moments(ctx["total_pmf"])
        self.assertAlmostEqual(ctx["total_mean"], conv_mean, places=9)

    def test_missing_inputs_fail_open_to_neutral(self):
        ctx, _ = self._build(None)
        self.assertFalse(ctx["complete"])
        for side in ("home", "away"):
            for form in ("full", "own", "opp"):
                self.assertEqual(ctx["gc_factor"][side][form], 1.0)
        self.assertIsNone(ctx["mu_runs"]["home"])
        self.assertIsNone(ctx["total_pmf"])

    def test_unmatched_team_fail_open(self):
        eri = MagicMock(return_value=_fake_inputs())
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "_match_team_id", return_value=None), \
             patch.object(mlb_starters, "get_expected_runs_team_factors", eri):
            ctx = mlb_starters.build_game_context(
                "Nope", "AlsoNope", "2024-07-01", 2024)
        self.assertFalse(ctx["complete"])
        self.assertEqual(ctx["gc_factor"]["home"]["full"], 1.0)
        # No abbr resolved -> the as-of source is never even queried.
        eri.assert_not_called()

    def test_cache_hit_skips_refetch(self):
        eri = MagicMock(return_value=_fake_inputs())
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "_match_team_id", side_effect=_fake_match), \
             patch.object(mlb_starters, "get_expected_runs_team_factors", eri):
            first = mlb_starters.build_game_context(
                "HomeTown", "AwayTown", "2024-07-01", 2024)
            second = mlb_starters.build_game_context(
                "HomeTown", "AwayTown", "2024-07-01", 2024)
        self.assertIs(first, second)
        eri.assert_called_once()

    def test_stale_incomplete_cache_is_rebuilt(self):
        # An incomplete (fail-open) entry must not be trusted past its short TTL.
        key = ("HOM", "AWY", "2024-07-01")
        stale = {"complete": False, "gc_factor": {"home": {"full": 1.0}}}
        mlb_starters._GC_CACHE[key] = (
            0.0, stale)  # built_at=0 -> older than _GC_LIVE_TTL
        ctx, eri = self._build(_fake_inputs())
        self.assertTrue(ctx["complete"])
        self.assertIsNot(ctx, stale)
        eri.assert_called_once()


class GamecontextFactorTest(unittest.TestCase):
    def setUp(self):
        mlb_starters._GC_CACHE.clear()

    def test_matches_build_game_context_side(self):
        eri = MagicMock(return_value=_fake_inputs())
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "_match_team_id", side_effect=_fake_match), \
             patch.object(mlb_starters, "get_expected_runs_team_factors", eri):
            ctx = mlb_starters.build_game_context(
                "HomeTown", "AwayTown", "2024-07-01", 2024)
            # HomeTown as own vs AwayTown opp == the object's home side.
            for form in ("full", "own", "opp"):
                scalar = mlb_starters.gamecontext_factor(
                    "HomeTown", "AwayTown", "2024-07-01", 2024, form=form)
                self.assertAlmostEqual(
                    scalar, ctx["gc_factor"]["home"][form], places=12)

    def test_fail_open_on_unmatched(self):
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "_match_team_id", return_value=None):
            self.assertEqual(
                mlb_starters.gamecontext_factor("x", "y", "2024-07-01", 2024), 1.0)

    def test_fail_open_on_no_team_index(self):
        with patch.object(mlb_starters, "get_team_index", return_value=None):
            self.assertEqual(
                mlb_starters.gamecontext_factor("x", "y", "2024-07-01", 2024), 1.0)


# ── Phase 2: prop_features gamecontext registry + projection threading ────────
class GamecontextRegistryTest(unittest.TestCase):
    FACTORS = {"full": 1.06, "own": 1.04, "opp": 0.97}

    def test_applies_only_to_batter_hits(self):
        for name in ("gamecontext", "gamecontext_own", "gamecontext_opp"):
            self.assertTrue(pf.feature_applies(name, "batter_hits"))
            for pk in ("pitcher_outs", "pitcher_strikeouts",
                       "pitcher_earned_runs", "player_points"):
                self.assertFalse(pf.feature_applies(name, pk))

    def test_strengths_from_params_reads_gc_knobs(self):
        self.assertEqual(
            pf.strengths_from_params({"gamecontext_strength": 1.0}),
            {"gamecontext": 1.0})
        self.assertEqual(
            pf.strengths_from_params({"gamecontext_own_strength": 0.5}),
            {"gamecontext_own": 0.5})
        self.assertEqual(
            pf.strengths_from_params({"gamecontext_opp_strength": 0.0}), {})

    def test_each_form_reads_its_own_key_and_scales(self):
        for name, key in (("gamecontext", "full"),
                          ("gamecontext_own", "own"),
                          ("gamecontext_opp", "opp")):
            gc = self.FACTORS[key]
            for s in (0.5, 1.0):
                m = pf.projection_multiplier(
                    "batter_hits", {name: s}, [], "2025-06-01",
                    gamecontext_factors=self.FACTORS)
                self.assertAlmostEqual(m, 1.0 + s * (gc - 1.0), places=12)

    def test_strength_zero_and_absent_factors_are_noop(self):
        # strength 0 -> production; missing form / None factors -> 1.0.
        self.assertEqual(pf.projection_multiplier(
            "batter_hits", {"gamecontext": 0.0}, [], "2025-06-01",
            gamecontext_factors=self.FACTORS), 1.0)
        self.assertEqual(pf.projection_multiplier(
            "batter_hits", {"gamecontext": 1.0}, [], "2025-06-01",
            gamecontext_factors=None), 1.0)
        self.assertEqual(pf.projection_multiplier(
            "batter_hits", {"gamecontext": 1.0}, [], "2025-06-01",
            gamecontext_factors={"own": 1.05}), 1.0)  # no "full" key

    def test_cap_bounds(self):
        blown = {"full": 5.0}  # far past the cap
        m = pf.projection_multiplier(
            "batter_hits", {"gamecontext": 1.0}, [], "2025-06-01",
            gamecontext_factors=blown)
        self.assertAlmostEqual(m, 1.0 + pf.GC_FEAT_CAP, places=12)

    def test_excluded_prop_is_noop(self):
        self.assertEqual(pf.projection_multiplier(
            "pitcher_outs", {"gamecontext": 1.0}, [], "2025-06-01",
            gamecontext_factors=self.FACTORS), 1.0)

    def test_legacy_4arg_call_still_works(self):
        # The added kwarg is optional -> old callers (rest) are byte-identical.
        dates = ["2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23",
                 "2025-05-24"]
        self.assertEqual(
            pf.projection_multiplier("batter_hits", {"rest": 1.0}, dates,
                                     "2025-05-30"),
            pf.projection_multiplier("batter_hits", {"rest": 1.0}, dates,
                                     "2025-05-30", gamecontext_factors=None))


# ── Phase 2: _attach_gamecontext fail-open plumbing ───────────────────────────
def _gc_ret(complete=True):
    forms = {"full": 1.05, "own": 1.03, "opp": 0.98}
    return {"complete": complete,
            "gc_factor": {"home": dict(forms),
                          "away": {"full": 0.96, "own": 0.97, "opp": 1.01}}}


class AttachGamecontextTest(unittest.TestCase):
    def _enriched(self, **over):
        row = {"prop_key": "batter_hits", "home_team": "HomeTown",
               "away_team": "AwayTown", "game_date": "2024-07-01",
               "test_game": {"is_home": True}}
        row.update(over)
        return [row]

    def test_complete_attaches_own_side(self):
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "build_game_context",
                          return_value=_gc_ret(True)):
            enr = self._enriched(**{"test_game": {"is_home": True}})
            blc._attach_gamecontext(enr, "baseball")
            self.assertEqual(enr[0]["gc_factor"]["full"], 1.05)   # home side
            enr = self._enriched(**{"test_game": {"is_home": False}})
            blc._attach_gamecontext(enr, "baseball")
            self.assertEqual(enr[0]["gc_factor"]["full"], 0.96)   # away side

    def test_none_teams_fail_open(self):
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "build_game_context",
                          return_value=_gc_ret(True)) as bgc:
            enr = self._enriched(home_team=None, away_team=None)
            blc._attach_gamecontext(enr, "baseball")
            self.assertNotIn("gc_factor", enr[0])
            bgc.assert_not_called()

    def test_incomplete_context_not_attached(self):
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "build_game_context",
                          return_value=_gc_ret(False)):
            enr = self._enriched()
            blc._attach_gamecontext(enr, "baseball")
            self.assertNotIn("gc_factor", enr[0])

    def test_non_baseball_is_noop(self):
        with patch.object(mlb_starters, "build_game_context",
                          return_value=_gc_ret(True)) as bgc:
            enr = self._enriched()
            blc._attach_gamecontext(enr, "basketball")
            self.assertNotIn("gc_factor", enr[0])
            bgc.assert_not_called()

    def test_build_exception_fails_open(self):
        with patch.object(mlb_starters, "get_team_index", return_value={"x": 1}), \
             patch.object(mlb_starters, "build_game_context",
                          side_effect=RuntimeError("boom")):
            enr = self._enriched()
            blc._attach_gamecontext(enr, "baseball")  # must not raise
            self.assertNotIn("gc_factor", enr[0])


if __name__ == "__main__":
    unittest.main()
