"""Unit tests for r2_grade — grading, profit, and ROI/significance aggregation."""
import unittest

import r2_grade as g
from odds_client import american_to_decimal


class GradeTests(unittest.TestCase):
    def test_over_under_half_line_never_pushes(self):
        self.assertEqual(g.grade_over_under(2, 1.5, "OVER"), "win")
        self.assertEqual(g.grade_over_under(1, 1.5, "OVER"), "loss")
        self.assertEqual(g.grade_over_under(1, 1.5, "UNDER"), "win")
        self.assertEqual(g.grade_over_under(2, 1.5, "UNDER"), "loss")

    def test_integer_line_pushes_on_equal(self):
        self.assertEqual(g.grade_over_under(9, 9, "OVER"), "push")
        self.assertEqual(g.grade_over_under(9, 9, "UNDER"), "push")
        self.assertEqual(g.grade_over_under(10, 9, "OVER"), "win")
        self.assertEqual(g.grade_over_under(8, 9, "UNDER"), "win")

    def test_grade_missing_inputs(self):
        self.assertIsNone(g.grade_over_under(None, 1.5, "OVER"))
        self.assertIsNone(g.grade_over_under(2, None, "OVER"))
        self.assertIsNone(g.grade_over_under(2, 1.5, ""))

    def test_moneyline(self):
        self.assertEqual(g.grade_moneyline("NYY", "NYY"), "win")
        self.assertEqual(g.grade_moneyline("NYY", "BOS"), "loss")
        self.assertIsNone(g.grade_moneyline("NYY", None))


class ProfitTests(unittest.TestCase):
    def test_win_loss_push(self):
        self.assertAlmostEqual(g.profit(+150, "win"), 1.5, places=9)
        self.assertAlmostEqual(g.profit(-110, "win"),
                               american_to_decimal(-110) - 1, places=9)
        self.assertEqual(g.profit(-110, "loss"), -1.0)
        self.assertEqual(g.profit(-110, "push"), 0.0)

    def test_bad_inputs(self):
        self.assertIsNone(g.profit(-110, "n/a"))
        self.assertIsNone(g.profit(None, "win"))


class SummarizeTests(unittest.TestCase):
    def test_roi_and_hit_rate(self):
        # 2 wins at +100 (+1 each), 2 losses (-1 each) -> ROI 0, hit 50%.
        rows = [
            {"result": "win", "profit": 1.0}, {"result": "win", "profit": 1.0},
            {"result": "loss", "profit": -1.0}, {"result": "loss", "profit": -1.0},
        ]
        s = g.summarize(rows)
        self.assertEqual((s.n, s.decided, s.wins, s.pushes), (4, 4, 2, 0))
        self.assertAlmostEqual(s.roi, 0.0, places=9)
        self.assertAlmostEqual(s.hit_rate, 0.5, places=9)

    def test_push_excluded_from_hit_rate_denominator(self):
        rows = [
            {"result": "win", "profit": 1.0},
            {"result": "push", "profit": 0.0},
            {"result": "loss", "profit": -1.0},
        ]
        s = g.summarize(rows)
        self.assertEqual((s.n, s.decided, s.pushes), (3, 2, 1))
        self.assertAlmostEqual(s.hit_rate, 0.5, places=9)   # 1 win / 2 decided

    def test_t_stat_positive_for_consistent_edge(self):
        # A realistic winning book (varying payouts) -> large positive t; a
        # zero-mean set -> ~0 t. (Constant profits have zero variance -> t=0 by
        # design, a safe non-false-positive; real profits always vary.)
        pos = [{"result": "win", "profit": 0.05 if i % 2 else 0.15}
               for i in range(50)]                       # mean 0.10, small variance
        self.assertGreater(g.summarize(pos).t_stat, 5.0)
        bal = [{"result": "win", "profit": 1.0} for _ in range(25)] + \
              [{"result": "loss", "profit": -1.0} for _ in range(25)]
        self.assertAlmostEqual(g.summarize(bal).t_stat, 0.0, places=6)

    def test_zero_variance_is_safe_zero_t(self):
        # Degenerate: identical profits -> zero sample variance -> t=0 (never a
        # spurious "infinitely significant" edge on constant data).
        self.assertEqual(g.summarize([{"result": "win", "profit": 0.1}] * 10).t_stat,
                         0.0)

    def test_empty(self):
        s = g.summarize([])
        self.assertEqual(s.n, 0)
        self.assertEqual(s.roi, 0.0)


class BucketingTests(unittest.TestCase):
    def test_ev_bucket_labels(self):
        self.assertEqual(g.ev_bucket(-0.01), "<0")
        self.assertEqual(g.ev_bucket(0.0), "0%-2%")
        self.assertEqual(g.ev_bucket(0.03), "2%-5%")
        self.assertEqual(g.ev_bucket(0.25), ">=20%")
        self.assertEqual(g.ev_bucket(None), "n/a")

    def test_by_key_groups_and_summarizes(self):
        rows = [
            {"season": "2024", "result": "win", "profit": 1.0},
            {"season": "2024", "result": "loss", "profit": -1.0},
            {"season": "2025", "result": "win", "profit": 1.0},
        ]
        out = g.by_key(rows, lambda r: r["season"])
        self.assertEqual(set(out), {"2024", "2025"})
        self.assertAlmostEqual(out["2024"].roi, 0.0, places=9)
        self.assertAlmostEqual(out["2025"].roi, 1.0, places=9)

    def test_replicates_per_season_gate(self):
        # 2024 positive, 2025 negative -> gate fails (must replicate EVERY season).
        rows = ([{"season": "2024", "result": "win", "profit": 0.1} for _ in range(40)] +
                [{"season": "2025", "result": "loss", "profit": -0.1} for _ in range(40)])
        ok, per = g.replicates_per_season(rows, lambda r: r["season"], min_n=30)
        self.assertFalse(ok)
        self.assertEqual(set(per), {"2024", "2025"})
        # both positive -> gate passes
        rows2 = ([{"season": "2024", "result": "win", "profit": 0.1} for _ in range(40)] +
                 [{"season": "2025", "result": "win", "profit": 0.1} for _ in range(40)])
        ok2, _ = g.replicates_per_season(rows2, lambda r: r["season"], min_n=30)
        self.assertTrue(ok2)

    def test_replicates_ignores_thin_seasons(self):
        # 2026 has too few bets to judge -> ignored by the gate, still reported.
        rows = ([{"season": "2024", "result": "win", "profit": 0.1} for _ in range(40)] +
                [{"season": "2026", "result": "loss", "profit": -0.1} for _ in range(5)])
        ok, per = g.replicates_per_season(rows, lambda r: r["season"], min_n=30)
        self.assertTrue(ok)                 # only 2024 judged, and it's positive
        self.assertIn("2026", per)          # thin season still in the report


if __name__ == "__main__":
    unittest.main()
