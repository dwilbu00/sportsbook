"""Unit tests for under_flags — the DK-vs-Pinnacle-F5 shape flagger (pure logic)."""
import unittest

import under_flags as uf


def _game(eid="E1", home="Red Sox", away="Mariners", dk_pt=9.0, pf_pt=4.5,
          dk_over=-110, dk_under=-110, with_dk=True, with_pin=True):
    books = []
    if with_dk:
        books.append({"key": "draftkings", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": dk_over, "point": dk_pt},
            {"name": "Under", "price": dk_under, "point": dk_pt}]}]})
    if with_pin:
        books.append({"key": "pinnacle", "markets": [{"key": "totals_1st_5_innings",
            "outcomes": [{"name": "Over", "price": -105, "point": pf_pt},
                         {"name": "Under", "price": -115, "point": pf_pt}]}]})
    return {"id": eid, "home_team": home, "away_team": away,
            "commence_time": "2026-09-01T23:00:00Z", "bookmakers": books}


class ClassifyGapTests(unittest.TestCase):
    def test_zones(self):
        self.assertEqual(uf.classify_gap(4.7)[0], "UNDER")   # strong
        self.assertEqual(uf.classify_gap(4.2)[0], "UNDER")   # lean
        self.assertEqual(uf.classify_gap(3.2)[0], "OVER")    # speculative
        self.assertEqual(uf.classify_gap(3.7), (None, None)) # weak middle -> skip
        self.assertEqual(uf.classify_gap(5.3), (None, None)) # tail -> skip
        self.assertEqual(uf.classify_gap(None), (None, None))

    def test_boundaries(self):
        self.assertEqual(uf.classify_gap(4.5)[1][:6], "STRONG")  # 4.5 inclusive -> strong
        self.assertEqual(uf.classify_gap(4.0)[1][:4], "lean")    # 4.0 inclusive -> lean
        self.assertIsNone(uf.classify_gap(5.0)[0])               # 5.0 excluded


class FlagFromPairsTests(unittest.TestCase):
    def test_under_flag_prices_at_dk_under(self):
        pairs = uf.pairs_from_upcoming([_game(dk_pt=9.0, pf_pt=4.5, dk_under=-120)])
        flags = uf.flag_from_pairs(pairs)
        self.assertEqual(len(flags), 1)
        f = flags[0]
        self.assertEqual(f["side"], "UNDER")
        self.assertEqual(f["gap"], 4.5)
        self.assertEqual(f["dk_price"], -120)          # UNDER -> DK's under price
        self.assertEqual(f["pin_f5_total"], 4.5)

    def test_over_flag_prices_at_dk_over(self):
        pairs = uf.pairs_from_upcoming([_game(dk_pt=7.5, pf_pt=4.5, dk_over=+100)])
        flags = uf.flag_from_pairs(pairs)
        self.assertEqual(flags[0]["side"], "OVER")
        self.assertEqual(flags[0]["dk_price"], +100)

    def test_skip_zone_no_flag(self):
        pairs = uf.pairs_from_upcoming([_game(dk_pt=8.5, pf_pt=4.5)])  # gap 4.0? -> lean
        self.assertTrue(uf.flag_from_pairs(pairs))                     # 4.0 flags (lean)
        pairs2 = uf.pairs_from_upcoming([_game(dk_pt=8.3, pf_pt=4.5)]) # gap 3.8 -> skip
        self.assertEqual(uf.flag_from_pairs(pairs2), [])

    def test_strong_sorted_first(self):
        pairs = uf.pairs_from_upcoming([
            _game(eid="spec", dk_pt=7.5, pf_pt=4.5),    # gap 3.0 OVER speculative
            _game(eid="strong", dk_pt=9.2, pf_pt=4.5)]) # gap 4.7 UNDER strong
        flags = uf.flag_from_pairs(pairs)
        self.assertEqual(flags[0]["event_id"], "strong")

    def test_missing_market_dropped(self):
        self.assertEqual(uf.pairs_from_upcoming([_game(with_pin=False)]), [])
        self.assertEqual(uf.pairs_from_upcoming([_game(with_dk=False)]), [])


if __name__ == "__main__":
    unittest.main()
