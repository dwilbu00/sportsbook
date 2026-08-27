"""Unit tests for r2_edge — DK-vs-sharp EV computation. Pure math, no warehouse."""
import unittest

import r2_edge as edge
from odds_client import american_to_decimal
from r2_sharp import fair_two_way


class PropLegEdgeTests(unittest.TestCase):
    def test_same_line_ev_signs(self):
        # Sharp posts hits 0.5 at -140/+120; DK posts the SAME line but cheaper OVER
        # (+100) -> DK over should be +EV; DK under at -110 vs a rich sharp under -EV.
        pin = [{"point": 0.5, "over_price": -140, "under_price": +120}]
        legs = edge.prop_leg_edges(0.5, +100, -110, pin)
        by = {lg.side: lg for lg in legs}
        self.assertIn("OVER", by)
        self.assertIn("UNDER", by)
        for lg in legs:
            self.assertFalse(lg.projected)
            self.assertEqual(lg.distance, 0.0)
        # sharp fair over from the deviged sharp price
        fair_over, _ = fair_two_way(-140, +120)
        self.assertAlmostEqual(by["OVER"].sharp_fair, fair_over, places=9)
        self.assertAlmostEqual(by["UNDER"].sharp_fair, 1 - fair_over, places=9)
        # EV = fair * decimal - 1
        self.assertAlmostEqual(
            by["OVER"].ev, fair_over * american_to_decimal(100) - 1, places=9)
        # DK over is richer than sharp fair price -> +EV; the mirrored under -> -EV
        self.assertGreater(by["OVER"].ev, 0.0)
        self.assertLess(by["UNDER"].ev, 0.0)

    def test_projected_flag_on_line_gap(self):
        # Sharp posts only 1.5; DK posts 0.5 -> projected across a 1.0 gap.
        pin = [{"point": 1.5, "over_price": +150, "under_price": -180}]
        legs = edge.prop_leg_edges(0.5, -120, +100, pin)
        self.assertTrue(all(lg.projected for lg in legs))
        self.assertTrue(all(lg.distance == 1.0 for lg in legs))
        # sharp fair over at 0.5 (P>=1) exceeds fair over at 1.5 (P>=2)
        over = next(lg for lg in legs if lg.side == "OVER")
        self.assertGreater(over.sharp_fair, 0.5)

    def test_unpriceable_sharp_returns_empty(self):
        self.assertEqual(edge.prop_leg_edges(0.5, -110, -110, []), [])
        # one-sided sharp -> can't devig -> no line
        self.assertEqual(
            edge.prop_leg_edges(0.5, -110, -110,
                                [{"point": 0.5, "over_price": -110}]), [])

    def test_missing_dk_side_skips_that_leg(self):
        pin = [{"point": 0.5, "over_price": -120, "under_price": +100}]
        legs = edge.prop_leg_edges(0.5, -110, None, pin)   # no DK under price
        self.assertEqual({lg.side for lg in legs}, {"OVER"})


class MoneylineEdgeTests(unittest.TestCase):
    def test_positive_when_dk_beats_sharp_fair(self):
        # Sharp: home -150 / away +130 -> fair home ~0.585. DK home at +120 (much
        # richer than fair price) -> strongly +EV.
        lg = edge.moneyline_edge(dk_price=+120, pin_price=-150, pin_other_price=+130)
        self.assertEqual(lg.side, "ML")
        self.assertIsNone(lg.point)
        fair, _ = fair_two_way(-150, +130)
        self.assertAlmostEqual(lg.sharp_fair, fair, places=9)
        self.assertGreater(lg.ev, 0.0)

    def test_negative_when_dk_worse_than_sharp(self):
        # DK home -200 while sharp fair says ~0.585 -> paying too much -> -EV.
        lg = edge.moneyline_edge(dk_price=-200, pin_price=-150, pin_other_price=+130)
        self.assertLess(lg.ev, 0.0)

    def test_unpriceable_returns_none(self):
        self.assertIsNone(edge.moneyline_edge(+120, -150, None))


class BestLegTests(unittest.TestCase):
    def test_picks_highest_positive_ev(self):
        pin = [{"point": 0.5, "over_price": -140, "under_price": +120}]
        legs = edge.prop_leg_edges(0.5, +100, -110, pin)
        best = edge.best_positive_leg(legs)
        self.assertEqual(best.side, "OVER")
        self.assertGreater(best.ev, 0.0)

    def test_none_when_no_positive(self):
        # DK strictly worse than sharp on both sides -> no +EV leg.
        pin = [{"point": 0.5, "over_price": +100, "under_price": +100}]
        legs = edge.prop_leg_edges(0.5, -130, -130, pin)
        self.assertIsNone(edge.best_positive_leg(legs))


if __name__ == "__main__":
    unittest.main()
