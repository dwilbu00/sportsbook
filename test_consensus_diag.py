"""Test the cross-method agreement diagnostic core (_consensus_prop_stats).

Pure/offline — no SQL, no network. Feeds synthetic held-out test rows (per-method
P(over), outcome, de-vigged market prob, decimal payouts) and asserts the agree-vs-
split ROI/hit split + the value-side agreement rate.
"""

import unittest

import refit_calibration as rc


def _row(m, mkt, outcome, dec=2.0):
    """One priced test row in the shape _roi_sim_method + _consensus_prop_stats read:
    m = {method: P(over)}, mkt = de-vigged market P(over), outcome = 1 (over) / 0."""
    return {"m": dict(m), "mkt_over": mkt, "o": outcome,
            "over_price": 100, "under_price": 100, "over_dec": dec, "under_dec": dec}


class ConsensusStatsTests(unittest.TestCase):
    def test_agree_beats_split(self):
        test = [
            # All of A/B/C see value on OVER (p > mkt 0.50); over WINS.
            _row({"A": 0.62, "B": 0.61, "C": 0.63}, 0.50, 1),
            _row({"A": 0.64, "B": 0.62, "C": 0.60}, 0.50, 1),
            # A/C say over, B says under → SPLIT on the value side; consensus backs
            # over (mean > mkt) and over LOSES.
            _row({"A": 0.70, "B": 0.40, "C": 0.60}, 0.50, 0),
            _row({"A": 0.72, "B": 0.38, "C": 0.58}, 0.50, 0),
        ]
        st = rc._consensus_prop_stats(test, 0.05)
        self.assertEqual(st["methods"], ["A", "B", "C"])
        self.assertEqual(st["n_priced"], 4)
        self.assertAlmostEqual(st["edge_agree_pct"], 50.0)   # 2 of 4 rows all-agree
        # Both buckets are bettable (|consensus edge| >= 5%), and agreement wins.
        self.assertEqual(st["agree"]["n_bets"], 2)
        self.assertEqual(st["split"]["n_bets"], 2)
        self.assertEqual(st["agree"]["hit"], 1.0)
        self.assertEqual(st["split"]["hit"], 0.0)
        self.assertGreater(st["agree"]["roi"], st["split"]["roi"])

    def test_no_priced_rows_returns_none(self):
        test = [{"m": {"A": 0.6, "B": 0.6, "C": 0.6}, "mkt_over": None, "o": 1}]
        self.assertIsNone(rc._consensus_prop_stats(test, 0.05))


if __name__ == "__main__":
    unittest.main()
