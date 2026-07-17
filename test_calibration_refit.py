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


if __name__ == "__main__":
    unittest.main()
