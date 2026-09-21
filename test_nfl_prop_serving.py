"""Concurrency regression for nfl_prop_serving (audit F13).

The live analysis path projects many props in a 16-worker thread pool. The shrink
strength must be passed per-call, never assigned to a module global, or parallel
projections race and become nondeterministic.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_nfl_prop_serving -v
"""
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import nfl_prop_serving as serving

_MODELS = "calibration/nfl_prop_models.json"


@unittest.skipUnless(os.path.exists(_MODELS), "frozen NFL prop models not present")
class F13ProjectionRaceTests(unittest.TestCase):
    def test_parallel_projection_matches_serial(self):
        obs = dict(n_prior=10, cv=.5, exp_vol=15., conv_own=7., vw=20., adot=None)
        with patch.object(serving, "build_obs", return_value=obs):
            expected = serving.project("a", 2026, 3, "player_rush_yds", 70.5)["proj"]

        a_inside = threading.Event()
        b_done = threading.Event()

        def interleave(player, *args):
            # Force projection "a" to sit inside build_obs while "b" runs fully — the
            # exact interleaving that leaked b's shrink K into a's mean via the global.
            if player == "a":
                a_inside.set()
                if not b_done.wait(5):
                    raise RuntimeError("fixture sync failed")
            else:
                if not a_inside.wait(5):
                    raise RuntimeError("fixture sync failed")
            return dict(obs)

        def other():
            try:
                return serving.project("b", 2026, 3, "player_reception_yds", 70.5)
            finally:
                b_done.set()

        with patch.object(serving, "build_obs", side_effect=interleave):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(serving.project, "a", 2026, 3,
                                    "player_rush_yds", 70.5)
                self.assertTrue(a_inside.wait(5))
                second = pool.submit(other)
                observed = first.result(timeout=6)["proj"]
                second.result(timeout=6)
        self.assertAlmostEqual(observed, expected, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
