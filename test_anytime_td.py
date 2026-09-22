"""Tests for the anytime-TD (Yes/No) parse fix in odds_client.

The anytime-TD market is two-way Yes/No, not Over/Under. Before the fix both
sides were dropped by the Over/Under filter, so the prop never priced. The fix
normalizes Yes -> Over (scores 0.5+ TDs) and No -> Under via _prop_side, so the
existing two-sided de-vig machinery treats it like any other 0.5-line prop. A
book that posts only the Yes side is still dropped by the two-sided requirement
(no fictitious one-sided market) -- a strict improvement, zero regression.

Run: PYTHONIOENCODING=utf-8 python test_anytime_td.py
"""
import unittest

from odds_client import (parse_player_props, dk_prop_lines, _prop_side,
                         american_to_implied_prob)


def _oc(player, side, price, point=None):
    oc = {"description": player, "name": side, "price": price}
    if point is not None:
        oc["point"] = point
    return oc


def _dk_book(outcomes, mkey="player_anytime_td"):
    return {"key": "draftkings", "title": "DraftKings",
            "markets": [{"key": mkey, "outcomes": outcomes}]}


def _game(bookmakers, gid="NFL1"):
    return {"id": gid, "home_team": "Bears", "away_team": "Packers",
            "sport_key": "americanfootball_nfl", "bookmakers": bookmakers}


class PropSideNormalizationTests(unittest.TestCase):

    def test_anytime_td_yes_maps_to_over(self):
        self.assertEqual(_prop_side("player_anytime_td", "Yes"), "Over")

    def test_anytime_td_no_maps_to_under(self):
        self.assertEqual(_prop_side("player_anytime_td", "No"), "Under")

    def test_anytime_td_over_under_passthrough(self):
        # Already-normalized sides are untouched (defensive; markets are Yes/No).
        self.assertEqual(_prop_side("player_anytime_td", "Over"), "Over")
        self.assertEqual(_prop_side("player_anytime_td", "Under"), "Under")

    def test_other_market_unchanged(self):
        # Non-anytime markets never remap -- Yes on a real O/U market stays Yes
        # (and is then correctly dropped by the caller's O/U filter).
        self.assertEqual(_prop_side("batter_hits", "Over"), "Over")
        self.assertEqual(_prop_side("batter_hits", "Yes"), "Yes")


class ParsePlayerPropsAnytimeTests(unittest.TestCase):

    def test_two_sided_yes_no_parses_at_line_half(self):
        game = _game([_dk_book([_oc("Star RB", "Yes", -110),
                                _oc("Star RB", "No", -110)])])
        props = parse_player_props(game)["props"]
        self.assertIn("player_anytime_td", props)
        entry = props["player_anytime_td"]["Star RB"]
        self.assertEqual(entry["line"], 0.5)                 # pinned, not None
        self.assertEqual(entry["over_price"], -110)          # Yes price
        self.assertEqual(entry["under_price"], -110)         # No price
        # Symmetric -110/-110 de-vigs to a fair 0.5 P(scores a TD).
        self.assertAlmostEqual(entry["over_implied"], 0.5, places=6)
        self.assertEqual(entry["dk_over_price"], -110)       # DK executable

    def test_yes_favored_gives_over_half_prob(self):
        game = _game([_dk_book([_oc("Star RB", "Yes", -200),
                                _oc("Star RB", "No", 160)])])
        entry = parse_player_props(game)["props"]["player_anytime_td"]["Star RB"]
        self.assertGreater(entry["over_implied"], 0.5)       # favored to score

    def test_yes_only_is_dropped_no_one_sided_market(self):
        # Only the Yes side posted -> two-sided requirement drops it (== today's
        # behavior for a one-sided book; no fictitious market invented).
        game = _game([_dk_book([_oc("Star RB", "Yes", -110)])])
        self.assertEqual(parse_player_props(game)["props"], {})

    def test_normal_over_under_prop_unaffected(self):
        # Regression guard: _prop_side passthrough leaves real O/U props intact.
        game = _game([_dk_book([_oc("Bat", "Over", -115, point=1.5),
                                _oc("Bat", "Under", -105, point=1.5)],
                               mkey="batter_hits")])
        props = parse_player_props(game)["props"]
        self.assertEqual(props["batter_hits"]["Bat"]["line"], 1.5)


class DkPropLinesAnytimeTests(unittest.TestCase):

    def test_two_sided_yes_no(self):
        game = _game([_dk_book([_oc("Star RB", "Yes", -120),
                                _oc("Star RB", "No", 100)])])
        out = dk_prop_lines(game, "player_anytime_td")
        self.assertEqual(out, [{"player": "Star RB", "line": 0.5,
                                "over_price": -120, "under_price": 100}])

    def test_yes_only_keeps_over_side(self):
        # dk_prop_lines does not require both sides (CLV backfill wants whatever
        # DK posted); Yes-only yields over_price set, under_price None at 0.5.
        game = _game([_dk_book([_oc("Star RB", "Yes", -120)])])
        out = dk_prop_lines(game, "player_anytime_td")
        self.assertEqual(out, [{"player": "Star RB", "line": 0.5,
                                "over_price": -120, "under_price": None}])


class AnytimeTdGradingTests(unittest.TestCase):
    """recalibration._anytime_td_actual — grades on TDs the player SCORED (rush +
    reception), never the ambiguous shared 'TD' label that collides with a QB's
    passing TDs."""

    def setUp(self):
        import recalibration
        self.rc = recalibration

    def test_sums_rush_and_receiving_excludes_passing(self):
        row = {"rushingTouchdowns": 1.0, "receivingTouchdowns": 1.0,
               "passingTouchdowns": 3.0, "TD": 3.0}
        self.assertEqual(self.rc._anytime_td_actual(row), 2.0)

    def test_scored_via_reception_only(self):
        row = {"rushingTouchdowns": 0.0, "receivingTouchdowns": 1.0,
               "passingTouchdowns": 0.0}
        self.assertEqual(self.rc._anytime_td_actual(row), 1.0)

    def test_qb_passing_tds_are_not_an_anytime_td(self):
        # The bug this fixes: a QB with 3 passing TDs but no rush/rec TD must grade
        # 0 (an anytime-TD OVER 0.5 LOSES), not 3 off the shared 'TD' label.
        row = {"rushingTouchdowns": 0.0, "receivingTouchdowns": 0.0,
               "passingTouchdowns": 3.0, "TD": 3.0}
        self.assertEqual(self.rc._anytime_td_actual(row), 0.0)

    def test_falls_back_to_shared_td_when_no_split_names(self):
        self.assertEqual(self.rc._anytime_td_actual({"TD": 1.0}), 1.0)

    def test_none_when_no_touchdown_field(self):
        self.assertIsNone(self.rc._anytime_td_actual({"YDS": 55.0}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
