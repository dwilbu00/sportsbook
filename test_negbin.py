"""§2.2 Negative-Binomial count-distribution calibration (method "E").

Covers the pure NegBin primitives in ``stats`` (survival, NLL drift-guard vs
``backtest_starters``, dispersion MLE), the runtime ``_negbin_over_rate`` branch
plus its end-to-end dispatch through ``analyze_player_props_value``, and the
offline real-line selection gate admitting E only when eligible and it wins. All
hermetic (no live ESPN / Statcast / SQL).
"""

import math
import random
import unittest
from math import lgamma
from unittest.mock import patch

import backtest_starters
import book_line_calibration as blc
import mlb_starters
import props
import stats


# ── brute-force reference distributions (test-only) ──────────────────────────
def _poisson_ge(k, mean):
    """P(X >= k) for a Poisson(mean), summed the slow honest way."""
    cdf = sum(math.exp(-mean) * mean ** x / math.factorial(x) for x in range(k))
    return max(0.0, min(1.0, 1.0 - cdf))


def _negbin_pmf(x, mean, dispersion):
    """NegBin P(X = x) with variance = mean + dispersion*mean^2 via lgamma."""
    size = 1.0 / dispersion
    p = size / (size + mean)                       # success prob
    return (math.exp(lgamma(x + size) - lgamma(size) - lgamma(x + 1.0))
            * p ** size * (1.0 - p) ** x)


def _negbin_ge(k, mean, dispersion):
    cdf = sum(_negbin_pmf(x, mean, dispersion) for x in range(k))
    return max(0.0, min(1.0, 1.0 - cdf))


class NegBinAtLeastTests(unittest.TestCase):
    def test_k_le_zero_is_certain(self):
        for mean in (0.5, 2.0, 6.0):
            self.assertEqual(stats.negbin_at_least(0, mean, 0.3), 1.0)
            self.assertEqual(stats.negbin_at_least(-3, mean, 0.3), 1.0)

    def test_zero_mean_never_clears_a_positive_line(self):
        self.assertEqual(stats.negbin_at_least(1, 0.0, 0.3), 0.0)
        self.assertEqual(stats.negbin_at_least(2, -1.0, 0.3), 0.0)

    def test_fractional_k_is_truncated(self):
        # int(k) mirrors hits_at_least's rounding contract.
        self.assertEqual(stats.negbin_at_least(2.9, 4.0, 0.4),
                         stats.negbin_at_least(2, 4.0, 0.4))

    def test_monotone_decreasing_in_k(self):
        for mean in (1.0, 3.0, 7.0):
            for disp in (0.0, 0.25, 1.0):
                prev = 1.0
                for k in range(0, 15):
                    cur = stats.negbin_at_least(k, mean, disp)
                    self.assertLessEqual(cur, prev + 1e-12)
                    prev = cur

    def test_poisson_limit_matches_bruteforce(self):
        # dispersion <= 0 => Poisson survival.
        for mean in (0.4, 1.0, 2.5, 5.0):
            for k in range(0, 12):
                self.assertAlmostEqual(stats.negbin_at_least(k, mean, 0.0),
                                       _poisson_ge(k, mean), places=9)
        # negative dispersion takes the same Poisson branch.
        self.assertAlmostEqual(stats.negbin_at_least(3, 2.0, -0.5),
                               _poisson_ge(3, 2.0), places=9)

    def test_matches_bruteforce_negbin(self):
        for mean in (0.6, 1.5, 3.0, 6.0):
            for disp in (0.1, 0.4, 1.0, 1.8):
                for k in range(0, 12):
                    self.assertAlmostEqual(
                        stats.negbin_at_least(k, mean, disp),
                        _negbin_ge(k, mean, disp), places=9)

    def test_overdispersion_fattens_the_upper_tail(self):
        # For a fixed mean below the line, more dispersion => MORE upper-tail mass
        # (a heavier right tail), so P(over) rises with dispersion.
        mean, k = 3.0, 6            # line 5.5
        p_lo = stats.negbin_at_least(k, mean, 0.05)
        p_hi = stats.negbin_at_least(k, mean, 1.5)
        self.assertGreater(p_hi, p_lo)


class NegBinNllDriftGuardTests(unittest.TestCase):
    """The inlined stats._negbin_nll must equal backtest_starters' NLL so stats
    stays a pure leaf without cross-importing backtest_starters."""

    def test_matches_backtest_starters_on_a_grid(self):
        for actual in (0.0, 1.0, 2.0, 5.0, 9.0):
            for mean in (0.5, 1.0, 3.0, 7.0):
                for disp in (0.0, 0.2, 0.75, 1.5):
                    self.assertAlmostEqual(
                        stats._negbin_nll(actual, mean, disp),
                        backtest_starters._negative_binomial_score_nll(
                            actual, mean, disp),
                        places=10)


class FitNegbinDispersionTests(unittest.TestCase):
    def test_recovers_a_known_dispersion(self):
        # Simulate over-dispersed counts (gamma-Poisson) with a known phi and check
        # the MLE lands close. Fixed-seed Mersenne Twister => deterministic.
        rng = random.Random(1234)
        true_phi, mean = 0.5, 4.0
        shape = 1.0 / true_phi
        pairs = []
        for _ in range(4000):
            lam = rng.gammavariate(shape, mean * true_phi)   # E[lam]=mean
            # Poisson(lam) via Knuth
            L, k, p = math.exp(-lam), 0, 1.0
            while True:
                k += 1
                p *= rng.random()
                if p <= L:
                    break
            pairs.append((mean, float(k - 1)))
        phi = stats.fit_negbin_dispersion(pairs)
        self.assertAlmostEqual(phi, true_phi, delta=0.15)

    def test_underdispersed_or_degenerate_returns_zero(self):
        # actual == mean everywhere => zero sample variance => not over-dispersed.
        self.assertEqual(stats.fit_negbin_dispersion([(3.0, 3.0)] * 50), 0.0)
        self.assertEqual(stats.fit_negbin_dispersion([]), 0.0)          # empty
        self.assertEqual(stats.fit_negbin_dispersion([(4.0, 5.0)]), 0.0)  # n<2
        # non-numeric / bad input fails to Poisson, never raises.
        self.assertEqual(stats.fit_negbin_dispersion([(None, 1.0),
                                                      (2.0, None)]), 0.0)

    def test_respects_cap(self):
        # Wildly over-dispersed data pins the estimate at the cap, not beyond.
        rng = random.Random(7)
        pairs = [(2.0, float(rng.choice([0, 0, 0, 0, 20, 30]))) for _ in range(300)]
        self.assertLessEqual(stats.fit_negbin_dispersion(pairs, cap=2.0), 2.0)


class NegbinOverRateTests(unittest.TestCase):
    def test_applies_mean_scale_and_continuity(self):
        p = props._negbin_over_rate(6.0, 1.1, 0.3, 5.5)
        self.assertAlmostEqual(p, stats.negbin_at_least(6, 6.6, 0.3), places=12)

    def test_none_mean_scale_defaults_to_one(self):
        p = props._negbin_over_rate(4.0, None, 0.4, 3.5)
        self.assertAlmostEqual(p, stats.negbin_at_least(4, 4.0, 0.4), places=12)

    def test_dispersion_none_is_poisson(self):
        p = props._negbin_over_rate(4.0, 1.0, None, 3.5)
        self.assertAlmostEqual(p, stats.negbin_at_least(4, 4.0, 0.0), places=12)

    def test_fails_open_on_unusable_mean(self):
        self.assertIsNone(props._negbin_over_rate(0.0, 1.0, 0.3, 0.5))
        self.assertIsNone(props._negbin_over_rate("x", 1.0, 0.3, 0.5))


class MethodEDispatchTests(unittest.TestCase):
    """analyze_player_props_value routes a method-"E" pitcher_strikeouts cfg
    through the NegBin count model (and non-E methods do not)."""

    def _prop_data(self, line=5.5):
        return {
            "commence_time": "2026-07-20T18:10:00Z", "home_team": "Home Nine",
            "away_team": "Away Nine", "game_id": "evt1",
            "props": {"pitcher_strikeouts": {"Ace Arm": {
                "line": line, "over_implied": 0.5, "under_implied": 0.5,
                "over_price": -110, "under_price": -110,
                "over_book": "DK", "under_book": "DK"}}},
        }

    def _histories(self):
        # All-equal values => base_proj == 6.0 regardless of half_life/shrinkage,
        # and pitcher_strikeouts carries no park/weather/lineup/matchup multiplier,
        # so avg_stat == 6.0 exactly (predictable NegBin mean).
        dates = [f"2026-07-{d:02d}" for d in range(1, 15)]
        return {"Ace Arm": {"pitcher_strikeouts": {
            "found": True, "values": [6.0] * 14,
            "game_dates": list(reversed(dates))}}}

    def _run(self, cfg):
        with patch.object(props, "load_calibration",
                          return_value={"pitcher_strikeouts": cfg}), \
             patch.object(props, "load_recalibration", return_value={}), \
             patch.object(props, "maybe_auto_refit"), \
             patch.object(props, "log_prediction_rows"), \
             patch.object(props, "log_prediction"), \
             patch.object(mlb_starters, "find_player_id",
                          return_value=("1", False)):
            cands = props.analyze_player_props_value(
                self._prop_data(), self._histories(), threshold_pct=1.0,
                sport_key="baseball_mlb")
        return cands[0]

    def test_method_E_routes_through_negbin(self):
        cand = self._run({"method": "E", "half_life": None,
                          "mean_scale": 1.1, "dispersion": 0.3})
        self.assertEqual(cand["calibration"]["method"], "E")
        self.assertAlmostEqual(cand["calibration"]["mean_scale"], 1.1)
        self.assertAlmostEqual(cand["calibration"]["dispersion"], 0.3)
        # over_rate (a %, rounded to 2 dp) is the NegBin survival at k=int(line)+1
        # of avg_stat*scale.
        self.assertAlmostEqual(
            cand["over_rate"] / 100.0,
            stats.negbin_at_least(6, 6.0 * 1.1, 0.3), places=3)

    def test_method_E_fails_open_without_params(self):
        # No mean_scale/dispersion => negbin mean is still avg_stat (scale->1),
        # dispersion->Poisson; still a valid E probability, not the raw empirical.
        cand = self._run({"method": "E", "half_life": None})
        self.assertEqual(cand["calibration"]["method"], "E")
        self.assertAlmostEqual(
            cand["over_rate"] / 100.0,
            stats.negbin_at_least(6, 6.0, 0.0), places=3)

    def test_non_E_method_untouched(self):
        cand = self._run({"method": "A", "half_life": None})
        self.assertEqual(cand["calibration"]["method"], "A")
        self.assertNotIn("dispersion", cand["calibration"])


def _overdispersed_rows(seed=4, n=180):
    """Deterministic over-dispersed count obs where the NegBin count model (E)
    separates over/under better than the symmetric Gaussian (B) or coarse residual
    ECDF (C). Uses only Mersenne-Twister .choice()/.random() (stable across CPython
    versions). Lines sit near each row's mean so the Gaussian floor miscalibration
    bites. empirical_over is a constant 0.5 => method A is uninformative (Brier
    0.25). Verified: at seed 4, E is the strict gate winner and confirms."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        mean = rng.choice([0.8, 1.2, 2.5, 4.0])
        mult = rng.choice([0.15, 0.4, 0.7, 1.0, 1.4, 2.2, 3.5])
        lam = mean * mult
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                break
        rows.append({"player": f"P{i % 20}", "projected": mean,
                     "line": round(mean) + 0.5, "actual": float(k - 1),
                     "empirical_over": 0.5,
                     "game_date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"})
    return rows


class SelectMethodECandidateTests(unittest.TestCase):
    """select_method_at_real_lines admits E only when negbin_eligible=True."""

    def test_e_selected_when_eligible_and_wins(self):
        sel = blc.select_method_at_real_lines(
            _overdispersed_rows(), negbin_eligible=True)
        self.assertEqual(sel["method"], "E")
        self.assertTrue(sel["confirmed"])
        self.assertIn("E", sel["single_split"])
        # deployed count-model params are fit on ALL usable obs and returned.
        self.assertIn("mean_scale", sel)
        self.assertIn("dispersion", sel)
        self.assertGreater(sel["dispersion"], 0.0)
        self.assertTrue(0.5 <= sel["mean_scale"] <= 2.0)
        # E genuinely beat the incumbents on the holdout.
        ss = sel["single_split"]
        self.assertLess(ss["E"], ss["A"])
        self.assertLess(ss["E"], ss["B"])
        self.assertLess(ss["E"], ss["C"])

    def test_backcompat_flag_off_no_e(self):
        sel = blc.select_method_at_real_lines(_overdispersed_rows())  # flag off
        self.assertNotIn("E", sel["single_split"])   # E never scored
        self.assertNotIn("mean_scale", sel)           # nor its params
        self.assertNotEqual(sel["method"], "E")

    def test_too_few_obs_returns_none(self):
        self.assertIsNone(blc.select_method_at_real_lines(
            _overdispersed_rows(n=10), negbin_eligible=True))


if __name__ == "__main__":
    unittest.main()
