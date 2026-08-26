"""Unit tests for ingest_multibook_cache — the per-book parse/tag/source logic.

No DB, no real cache: synthetic Odds-API-shaped payloads exercise the single-book
feeding (parity path), the bookmaker tagging, close/open detection, and the
double-/single-nested cache unwrap.
"""
import json
import os
import tempfile
import unittest

import ingest_multibook_cache as img
import warehouse as wh


def _team_game():
    return {
        "id": "evt_team_1", "sport_key": "baseball_mlb",
        "home_team": "Baltimore Orioles", "away_team": "Seattle Mariners",
        "commence_time": "2024-05-19T22:00:00Z",
        "bookmakers": [
            {"key": "draftkings", "title": "DraftKings", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Baltimore Orioles", "price": -150},
                    {"name": "Seattle Mariners", "price": 130}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 8.5, "price": -110},
                    {"name": "Under", "point": 8.5, "price": -105}]}]},
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Baltimore Orioles", "price": -145},
                    {"name": "Seattle Mariners", "price": 128}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 8.5, "price": -108},
                    {"name": "Under", "point": 8.5, "price": -108}]}]},
        ],
    }


def _prop_game():
    return {
        "id": "evt_prop_1", "sport_key": "baseball_mlb",
        "home_team": "Baltimore Orioles", "away_team": "Seattle Mariners",
        "commence_time": "2024-05-19T22:00:00Z",
        "bookmakers": [
            {"key": "draftkings", "title": "DraftKings", "markets": [
                {"key": "batter_hits", "outcomes": [
                    {"name": "Over", "description": "Gunnar Henderson", "point": 1.5, "price": 120},
                    {"name": "Under", "description": "Gunnar Henderson", "point": 1.5, "price": -150}]}]},
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "batter_hits", "outcomes": [
                    {"name": "Over", "description": "Gunnar Henderson", "point": 1.5, "price": 115},
                    {"name": "Under", "description": "Gunnar Henderson", "point": 1.5, "price": -145}]}]},
        ],
    }


class PerBookLinesTests(unittest.TestCase):
    def test_team_lines_tagged_per_book(self):
        lines = img._per_book_lines(_team_game(), "team")
        # per book: 2 moneyline + 2 totals (Over/Under @8.5) = 4; x2 books = 8
        self.assertEqual(len(lines), 8)
        self.assertEqual({ln["bookmaker"] for ln in lines}, {"draftkings", "pinnacle"})
        self.assertEqual(sum(ln["bookmaker"] == "draftkings" for ln in lines), 4)
        # DK moneyline for the Orioles keeps its own price (single-book -> no cross-book max)
        dk_ml = [ln for ln in lines if ln["bookmaker"] == "draftkings"
                 and ln["bet_type"] == "moneyline" and ln["selection"] == "Baltimore Orioles"]
        self.assertEqual(len(dk_ml), 1)
        self.assertEqual(dk_ml[0]["price"], -150)
        # Pinnacle keeps ITS price (proves per-book separation, not a collapse)
        pin_ml = [ln for ln in lines if ln["bookmaker"] == "pinnacle"
                  and ln["bet_type"] == "moneyline" and ln["selection"] == "Baltimore Orioles"]
        self.assertEqual(pin_ml[0]["price"], -145)

    def test_prop_lines_tagged_per_book(self):
        lines = img._per_book_lines(_prop_game(), "props")
        # per book: Over + Under = 2; x2 books = 4
        self.assertEqual(len(lines), 4)
        self.assertEqual({ln["bookmaker"] for ln in lines}, {"draftkings", "pinnacle"})
        for ln in lines:
            self.assertEqual(ln["bet_type"], "player_prop")
            self.assertEqual(ln["prop_key"], "batter_hits")
            self.assertEqual(ln["player"], "Gunnar Henderson")
            self.assertEqual(ln["point"], 1.5)
            self.assertIn(ln["direction"], ("OVER", "UNDER"))
        dk_over = [ln for ln in lines if ln["bookmaker"] == "draftkings"
                   and ln["direction"] == "OVER"]
        self.assertEqual(len(dk_over), 1)
        self.assertEqual(dk_over[0]["price"], 120)

    def test_region_left_null(self):
        lines = img._per_book_lines(_prop_game(), "props")
        self.assertTrue(all(ln["region"] is None for ln in lines))


class KindAndSourceTests(unittest.TestCase):
    def test_kind_derivation(self):
        self.assertEqual(
            wh._kind_for_markets(",".join(sorted(img._game_market_keys(_team_game())))),
            "team")
        self.assertEqual(
            wh._kind_for_markets(",".join(sorted(img._game_market_keys(_prop_game())))),
            "props")

    def test_classify_snapshot(self):
        EARLY, WIN = 12, 30
        # ts ~5 min before first pitch -> that game's close
        self.assertEqual(
            img._classify_snapshot("2024-05-19T21:55:00Z", "2024-05-19T22:00:00Z", EARLY, WIN),
            "multibook_close")
        # ts at the fixed early hour, hours before -> morning open
        self.assertEqual(
            img._classify_snapshot("2024-05-19T12:00:00Z", "2024-05-19T22:00:00Z", EARLY, WIN),
            "multibook_open")
        # ts is an intraday pre-close (5pm for a 10pm game), not the early hour -> skip
        self.assertIsNone(
            img._classify_snapshot("2024-05-19T17:00:00Z", "2024-05-19T22:00:00Z", EARLY, WIN))
        # unparseable ts -> skip (never silently mislabeled)
        self.assertIsNone(
            img._classify_snapshot(None, "2024-05-19T22:00:00Z", EARLY, WIN))


class IterCacheGamesTests(unittest.TestCase):
    def test_unwraps_both_nestings_and_filters_sport(self):
        with tempfile.TemporaryDirectory() as d:
            # double-nested historical (featured list)
            with open(os.path.join(d, "a.json"), "w") as f:
                json.dump({"cached_at": 1.0,
                           "data": {"timestamp": "2024-05-19T22:00:00Z",
                                    "data": [_team_game()]}}, f)
            # single-nested live (one event dict)
            with open(os.path.join(d, "b.json"), "w") as f:
                json.dump({"cached_at": 2.0, "data": _prop_game()}, f)
            # wrong sport -> filtered out
            with open(os.path.join(d, "c.json"), "w") as f:
                nba = dict(_team_game()); nba["sport_key"] = "basketball_nba"
                json.dump({"cached_at": 3.0,
                           "data": {"timestamp": "x", "data": [nba]}}, f)
            got = list(img._iter_cache_games(d, "baseball_mlb"))
        ids = {g["id"] for g, _ts, _p in got}
        self.assertEqual(ids, {"evt_team_1", "evt_prop_1"})
        # the double-nested file carries its snapshot ts; the live file has none
        ts_by_id = {g["id"]: ts for g, ts, _p in got}
        self.assertEqual(ts_by_id["evt_team_1"], "2024-05-19T22:00:00Z")
        self.assertIsNone(ts_by_id["evt_prop_1"])


if __name__ == "__main__":
    unittest.main()
