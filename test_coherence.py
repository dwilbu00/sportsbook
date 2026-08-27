"""Unit tests for coherence — the run-distribution cross-market translation.
Ground truth is self-consistent by construction (derive the 3 market probs from a
known pair of run means, then check the solver round-trips them)."""
import unittest

import coherence as c


class BasicShapeTests(unittest.TestCase):
    def test_symmetric_means_are_a_coinflip(self):
        self.assertAlmostEqual(c.p_home_win(4.5, 4.5), 0.5, places=6)

    def test_more_home_runs_more_wins(self):
        self.assertGreater(c.p_home_win(6.0, 4.0), c.p_home_win(4.0, 6.0))
        self.assertGreater(c.p_home_win(6.0, 4.0), 0.5)

    def test_total_over_increases_in_total_mean(self):
        self.assertGreater(c.p_total_over(6.0, 6.0, 8.5),
                           c.p_total_over(4.0, 4.0, 8.5))

    def test_covering_is_harder_than_winning(self):
        # P(win by 2+) < P(win) for a home favorite at -1.5
        mh, ma = 5.5, 4.0
        self.assertLess(c.p_home_cover(mh, ma, -1.5), c.p_home_win(mh, ma))


class SolveRoundTripTests(unittest.TestCase):
    def _check(self, mh, ma, tl, disp):
        mlf = c.p_home_win(mh, ma, disp)
        tof = c.p_total_over(mh, ma, tl, disp)
        got = c.solve_run_means(mlf, tl, tof, dispersion=disp)
        self.assertIsNotNone(got)
        # Means recover to within a fraction of a run.
        self.assertAlmostEqual(got[0], mh, delta=0.25)
        self.assertAlmostEqual(got[1], ma, delta=0.25)
        # THE key property: when the 3 prices ARE coherent, the ML+total-implied
        # run-line cover matches the true cover (residual ~0). A real residual in
        # the backtest therefore means a genuine DK mispricing.
        true_cover = c.p_home_cover(mh, ma, -1.5, disp)
        impl = c.implied_home_cover(mlf, tl, tof, -1.5, dispersion=disp)
        self.assertIsNotNone(impl)
        self.assertAlmostEqual(impl, true_cover, delta=0.01)

    def test_roundtrip_poisson_home_fav(self):
        self._check(5.2, 4.0, 8.5, 0.0)

    def test_roundtrip_poisson_away_fav(self):
        self._check(3.8, 5.0, 9.5, 0.0)

    def test_roundtrip_negbin(self):
        self._check(4.8, 4.2, 8.5, 0.15)

    def test_roundtrip_low_total(self):
        self._check(3.5, 3.2, 7.0, 0.0)


class DegenerateInputTests(unittest.TestCase):
    def test_bad_targets_return_none(self):
        self.assertIsNone(c.solve_run_means(0.0, 8.5, 0.5))
        self.assertIsNone(c.solve_run_means(0.6, 8.5, 1.0))
        self.assertIsNone(c.solve_run_means(0.6, 0.0, 0.5))
        self.assertIsNone(c.implied_home_cover(1.0, 8.5, 0.5, -1.5))


if __name__ == "__main__":
    unittest.main()
