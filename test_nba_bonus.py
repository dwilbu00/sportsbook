"""NBA bonus-engine wiring (nfl_bonus_optimizer NBA branch):
  * NBA market universe + dispatch;
  * leg P = the bonus book's OWN two-way de-vig at the line + its per-book over-bias map
    (not the board's all-book consensus over_implied); a book with no two-sided price is
    skipped; NBA legs get the odds-feed team (home/away) for composition rules;
  * NBA same-game rules in the shared conflict check (same player opposite sides; teammates
    both OVER on the same stat) — NFL/MLB legs unaffected.
Synthetic boards; no network/DB.
"""
import unittest
from unittest.mock import patch

import nfl_bonus_optimizer as opt
from odds_client import american_to_implied_prob, devig_two_way


def _board(props):
    return [{"game_id": "evt1", "home_team": "Home Team", "away_team": "Away Team",
             "commence_time": "2025-01-15T00:30:00Z", "props": props}]


def _p(dk_o, dk_u, fd_o=None, fd_u=None, consensus=0.70, line=24.5):
    return {"line": line, "over_implied": consensus,
            "dk_over_price": dk_o, "dk_under_price": dk_u,
            "fd_over_price": fd_o, "fd_under_price": fd_u}


class MarketsTests(unittest.TestCase):
    def test_nba_universe(self):
        m = opt.bonus_markets("basketball_nba")
        self.assertEqual(len(m), 10)
        self.assertIn("player_points_rebounds_assists", m)
        self.assertNotIn("batter_hits", m)                      # no longer falls to MLB
        self.assertEqual(opt.bonus_markets("baseball_mlb"), opt.MLB_BONUS_MARKETS)
        self.assertEqual(set(opt.NBA_ABBR), set(opt.NBA_BONUS_MARKETS))

    def test_value_finder_offers_every_bonus_market(self):
        # owner 2026-10-16: the Value Finder offers all 10 calibrated NBA props, so a board
        # built there can feed every NBA bonus market (credits only on props ticked per run)
        import odds_client
        self.assertEqual(set(odds_client.PLAYER_PROPS_BY_SPORT["basketball_nba"]),
                         set(opt.NBA_BONUS_MARKETS))
        self.assertTrue(all(p in odds_client.PROP_LABELS for p in opt.NBA_BONUS_MARKETS))


class LegsTests(unittest.TestCase):
    def _legs(self, props, book, maps=None, teams=None, resolve=None):
        with patch("book_calibration.load_maps", return_value=maps or {}), \
             patch("nba_warehouse.player_teams", return_value=teams or {}), \
             patch("nba_warehouse.resolve_team_id",
                   side_effect=(resolve or {"Home Team": 1, "Away Team": 2}).get):
            return opt.legs_from_scoped_board(_board(props), book, None, "basketball_nba")

    def test_own_book_devig_not_consensus(self):
        legs = self._legs({"player_points": {"Star Guard": _p(-130, 110)}}, "draftkings")
        own = devig_two_way(american_to_implied_prob(-130), american_to_implied_prob(110))[0]
        self.assertEqual(len(legs), 1)
        self.assertAlmostEqual(legs[0]["P"], own)               # not the 0.70 consensus
        self.assertEqual((legs[0]["side"], legs[0]["odds"]), ("OVER", -130))

    def test_book_map_applied_per_book(self):
        maps = {"player_points@draftkings": {"kind": "platt", "a": 1.0, "b": -0.5}}
        dk = self._legs({"player_points": {"G": _p(-115, -105, -115, -105)}}, "draftkings",
                        maps=maps)[0]
        fd = self._legs({"player_points": {"G": _p(-115, -105, -115, -105)}}, "fanduel",
                        maps=maps)[0]
        raw = devig_two_way(american_to_implied_prob(-115), american_to_implied_prob(-105))[0]
        self.assertAlmostEqual(fd["P"], raw)                    # FD: no map -> raw de-vig
        # DK map shifts the over below 0.5 -> the UNDER becomes the favorite
        self.assertEqual(dk["side"], "UNDER")
        self.assertEqual(dk["odds"], -105)
        self.assertGreater(dk["P"], 0.5)

    def test_missing_book_price_skipped(self):
        self.assertEqual(self._legs({"player_points": {"G": _p(-130, 110)}}, "fanduel"), [])

    def test_team_tagging(self):
        legs = self._legs({"player_points": {"Star Guard": _p(-130, 110),
                                             "Role Guy": _p(-120, 100)}}, "draftkings",
                          teams={"star guard": 1, "role guy": 2})
        by = {l["player"]: l for l in legs}
        self.assertEqual(by["Star Guard"]["team"], "Home Team")
        self.assertEqual(by["Star Guard"]["opp"], "Away Team")
        self.assertEqual(by["Role Guy"]["team"], "Away Team")


class CompositionTests(unittest.TestCase):
    def _leg(self, player, prop, side, team="Home Team"):
        return {"player": player, "prop": prop, "side": side, "team": team, "line": 10.5}

    def test_same_player_opposite_sides_excluded(self):
        a = self._leg("A", "player_points", "OVER")
        b = self._leg("A", "player_assists", "UNDER")
        self.assertTrue(opt._pair_impossible(a, b))

    def test_same_player_same_side_allowed(self):
        a = self._leg("A", "player_points", "OVER")
        b = self._leg("A", "player_points_rebounds_assists", "OVER")
        self.assertFalse(opt._pair_impossible(a, b))

    def test_teammates_both_over_same_stat_excluded(self):
        a = self._leg("A", "player_points", "OVER")
        b = self._leg("B", "player_points", "OVER")
        self.assertTrue(opt._pair_impossible(a, b))
        self.assertFalse(opt._pair_impossible(self._leg("A", "player_points", "UNDER"),
                                              self._leg("B", "player_points", "UNDER")))
        self.assertFalse(opt._pair_impossible(a, self._leg("B", "player_rebounds", "OVER")))

    def test_opponents_both_over_allowed(self):
        a = self._leg("A", "player_points", "OVER", team="Home Team")
        b = self._leg("B", "player_points", "OVER", team="Away Team")
        self.assertFalse(opt._pair_impossible(a, b))

    def test_non_nba_legs_unaffected(self):
        a = {"player": "QB", "prop": "player_pass_yds", "side": "OVER", "team": "X",
             "line": 250.5}
        b = {"player": "QB", "prop": "player_pass_attempts", "side": "UNDER", "team": "X",
             "line": 35.5}
        self.assertFalse(opt._pair_impossible(a, b))            # NFL rules unchanged


if __name__ == "__main__":
    unittest.main()
