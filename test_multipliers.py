"""Characterization test locking ``backtest.opp_strength_mult`` to the (now
deleted) ``analysis._opponent_strength_multiplier`` formula.

``analysis._opponent_strength_multiplier`` was dead code (zero callers): the
live, parameterized implementation is ``backtest.opp_strength_mult(opp_win_pct,
strength)``. At the reference strength 0.5 the two are identical —
``opp_strength_mult(p, 0.5) == 0.5 + clamp(p, 0, 1)`` — so this test documents
and locks that equivalence now that the duplicate has been removed (P3 dedup).
"""

import unittest

import backtest


def _legacy_formula(opp_win_pct):
    """The deleted ``analysis._opponent_strength_multiplier`` body verbatim."""
    if opp_win_pct is None:
        return 1.0
    clamped = max(0.0, min(1.0, opp_win_pct))
    return 0.5 + clamped


class OppStrengthMultTests(unittest.TestCase):
    def test_matches_legacy_formula_at_reference_strength(self):
        for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            self.assertAlmostEqual(
                backtest.opp_strength_mult(p, 0.5), _legacy_formula(p),
                places=12, msg=f"diverged at opp_win_pct={p}")

    def test_clamps_out_of_range_win_pct(self):
        # Below 0 and above 1 clamp to the [0.5, 1.5] endpoints.
        self.assertEqual(backtest.opp_strength_mult(-0.3, 0.5), 0.5)
        self.assertEqual(backtest.opp_strength_mult(1.4, 0.5), 1.5)

    def test_none_is_neutral(self):
        self.assertEqual(backtest.opp_strength_mult(None, 0.5), 1.0)

    def test_strength_zero_is_off(self):
        # strength<=0 disables the adjustment regardless of opponent.
        self.assertEqual(backtest.opp_strength_mult(0.9, 0.0), 1.0)

    def test_wider_strength_widens_range(self):
        # strength=0.75 -> range [0.25, 1.75].
        self.assertAlmostEqual(backtest.opp_strength_mult(0.0, 0.75), 0.25)
        self.assertAlmostEqual(backtest.opp_strength_mult(1.0, 0.75), 1.75)


if __name__ == "__main__":
    unittest.main()
