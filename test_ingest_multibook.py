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


def _f5_game():
    """First-5-innings payload as the event endpoint returns it: F5-only market
    keys. DK posts F5 moneyline ONLY (audit-confirmed); Pinnacle posts all three."""
    return {
        "id": "evt_f5_1", "sport_key": "baseball_mlb",
        "home_team": "New York Mets", "away_team": "New York Yankees",
        "commence_time": "2024-06-26T23:10:00Z",
        "bookmakers": [
            {"key": "draftkings", "title": "DraftKings", "markets": [
                {"key": "h2h_1st_5_innings", "outcomes": [
                    {"name": "New York Mets", "price": 105},
                    {"name": "New York Yankees", "price": -125}]}]},
            {"key": "pinnacle", "title": "Pinnacle", "markets": [
                {"key": "h2h_1st_5_innings", "outcomes": [
                    {"name": "New York Mets", "price": 108},
                    {"name": "New York Yankees", "price": -120}]},
                {"key": "spreads_1st_5_innings", "outcomes": [
                    {"name": "New York Mets", "point": 0.5, "price": -130},
                    {"name": "New York Yankees", "point": -0.5, "price": 110}]},
                {"key": "totals_1st_5_innings", "outcomes": [
                    {"name": "Over", "point": 4.5, "price": -105},
                    {"name": "Under", "point": 4.5, "price": -105}]}]},
        ],
    }


class F5Tests(unittest.TestCase):
    def test_kind_is_first_five(self):
        keys = ",".join(sorted(img._game_market_keys(_f5_game())))
        self.assertEqual(wh._kind_for_markets(keys), "first_five")

    def test_f5_does_not_collide_with_team_kind(self):
        # An F5 payload and a full-game team payload for the same event/hour must
        # resolve to DIFFERENT kinds, so uq_odds_snapshot never drops one.
        self.assertNotEqual(
            wh._kind_for_markets(",".join(sorted(img._game_market_keys(_f5_game())))),
            wh._kind_for_markets(",".join(sorted(img._game_market_keys(_team_game())))))

    def test_f5_lines_parsed_per_book_as_team_shapes(self):
        lines = img._per_book_lines(_f5_game(), "first_five")
        self.assertEqual({ln["bookmaker"] for ln in lines}, {"draftkings", "pinnacle"})
        # F5 keys are shimmed to base shapes -> emitted as moneyline/total/spread.
        self.assertTrue(all(ln["bet_type"] in ("moneyline", "spread", "total")
                            for ln in lines))
        # DK posts F5 moneyline only -> exactly its 2 moneyline sides, no totals.
        dk = [ln for ln in lines if ln["bookmaker"] == "draftkings"]
        self.assertEqual(len(dk), 2)
        self.assertEqual({ln["bet_type"] for ln in dk}, {"moneyline"})
        dk_mets = [ln for ln in dk if ln["selection"] == "New York Mets"]
        self.assertEqual(dk_mets[0]["price"], 105)  # DK's own F5 price, un-collapsed
        # Pinnacle posts all three F5 markets: 2 ml + 2 spread + 2 total = 6 lines.
        pin = [ln for ln in lines if ln["bookmaker"] == "pinnacle"]
        self.assertEqual(len(pin), 6)
        self.assertEqual({ln["bet_type"] for ln in pin},
                         {"moneyline", "spread", "total"})
        pin_tot = [ln for ln in pin if ln["bet_type"] == "total"]
        self.assertEqual({ln["point"] for ln in pin_tot}, {4.5})

    def test_scan_tags_f5_as_own_kind(self):
        # An F5 file and a team file for distinct events both scan with their own
        # kind; the F5 event keys on ("evt_f5_1", "first_five").
        def _write(d, name, ts, game):
            with open(os.path.join(d, name), "w") as f:
                json.dump({"cached_at": 1.0, "data": {"timestamp": ts, "data": [game]}}, f)
        with tempfile.TemporaryDirectory() as d:
            _write(d, "f5_close.json", "2024-06-26T23:05:00Z", _f5_game())
            chosen, stats = img._scan_snapshots(d, "baseball_mlb", 12, progress_every=0)
        self.assertIn(("evt_f5_1", "first_five"), chosen)
        self.assertEqual(chosen[("evt_f5_1", "first_five")]["close"],
                         wh._hour_bucket("2024-06-26T23:05:00Z"))

    def test_kinds_filter_scans_only_matching_kind(self):
        # With kinds={'first_five'}, a team file is skipped in pass 1 -> only the F5
        # event lands in `chosen` (this is the fast F5-only top-up path).
        def _write(d, name, ts, game):
            with open(os.path.join(d, name), "w") as f:
                json.dump({"cached_at": 1.0, "data": {"timestamp": ts, "data": [game]}}, f)
        with tempfile.TemporaryDirectory() as d:
            _write(d, "f5.json", "2024-06-26T23:05:00Z", _f5_game())
            _write(d, "team.json", "2024-05-19T21:55:00Z", _team_game())
            chosen, _stats = img._scan_snapshots(
                d, "baseball_mlb", 12, kinds={"first_five"}, progress_every=0)
        self.assertEqual(set(chosen), {("evt_f5_1", "first_five")})


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

    def _write(self, d, name, ts, game):
        with open(os.path.join(d, name), "w") as f:
            json.dump({"cached_at": 1.0, "data": {"timestamp": ts, "data": [game]}}, f)

    def test_scan_picks_open_and_nearest_close(self):
        # One event seen 3x: 12Z (open), 21:55 (nearest commence 22:00 -> close),
        # 17:00 (intraday, further from commence -> neither). _scan_snapshots must
        # choose 12Z open + 21:55 close, and mark the event as having both.
        g = _team_game()  # id evt_team_1, commence 2024-05-19T22:00:00Z
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "a_open.json", "2024-05-19T12:00:00Z", g)
            self._write(d, "b_close.json", "2024-05-19T21:55:00Z", g)
            self._write(d, "c_mid.json", "2024-05-19T17:00:00Z", g)
            chosen, stats = img._scan_snapshots(d, "baseball_mlb", 12, progress_every=0)
        key = ("evt_team_1", "team")
        self.assertEqual(chosen[key]["open"], wh._hour_bucket("2024-05-19T12:00:00Z"))
        self.assertEqual(chosen[key]["close"], wh._hour_bucket("2024-05-19T21:55:00Z"))
        self.assertNotEqual(chosen[key]["close"], wh._hour_bucket("2024-05-19T17:00:00Z"))
        self.assertEqual((stats["n_both"], stats["n_open_only"], stats["n_close_only"]), (1, 0, 0))

    def test_scan_completeness_open_only(self):
        # An event with ONLY an early snapshot -> open-only (no close invented).
        g = _prop_game()  # id evt_prop_1
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "o.json", "2024-05-19T12:00:00Z", g)
            chosen, stats = img._scan_snapshots(d, "baseball_mlb", 12, progress_every=0)
        self.assertEqual((stats["n_both"], stats["n_open_only"], stats["n_close_only"]), (0, 1, 0))
        self.assertIsNone(chosen[("evt_prop_1", "props")]["close"])


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

    def test_seasons_filter_excludes_purged_year(self):
        g2023 = _team_game()
        g2023["id"] = "evt_2023"
        g2023["commence_time"] = "2023-05-19T22:00:00Z"
        g2024 = _team_game()  # id evt_team_1, commence 2024-05-19
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.json"), "w") as f:
                json.dump({"cached_at": 1.0,
                           "data": {"timestamp": "2024-05-19T22:00:00Z",
                                    "data": [g2023, g2024]}}, f)
            got = list(img._iter_cache_games(d, "baseball_mlb", {"2024", "2025", "2026"}))
        self.assertEqual({g["id"] for g, _ts, _p in got}, {"evt_team_1"})  # 2023 dropped


if __name__ == "__main__":
    unittest.main()
