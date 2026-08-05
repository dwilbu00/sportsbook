"""Tests for the DraftKings closing-line CLV backfill (backfill_dk_clv.py) and
the odds_client DK extractors (dk_prop_lines for props, dk_game_lines for the
featured team markets).

Run: PYTHONIOENCODING=utf-8 python test_backfill_dk_clv.py
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch, Mock

import backfill_dk_clv as bk
import db_store
import wagers
from odds_client import dk_prop_lines, dk_game_lines, american_to_implied_prob


# ── fixtures ────────────────────────────────────────────────────────────────

def _mkt(key, outcomes):
    return {"key": key, "outcomes": outcomes}


def _oc(player, side, point, price):
    return {"description": player, "name": side, "point": point, "price": price}


DK_BOOK = {
    "key": "draftkings", "title": "DraftKings",
    "markets": [
        _mkt("batter_hits", [
            _oc("Bat", "Over", 1.5, -115), _oc("Bat", "Under", 1.5, -105)]),
        _mkt("pitcher_outs", [
            _oc("Pit", "Over", 17.5, -120), _oc("Pit", "Under", 17.5, 100)]),
    ],
}
FD_BOOK = {  # a non-DK book that must never leak into dk_prop_lines
    "key": "fanduel", "title": "FanDuel",
    "markets": [_mkt("batter_hits", [
        _oc("Bat", "Over", 2.5, -140), _oc("Bat", "Under", 2.5, 110)])],
}


def _game(bookmakers, gid="E1"):
    return {"id": gid, "home_team": "H", "away_team": "A",
            "bookmakers": bookmakers}


# ── dk_prop_lines (odds_client) ─────────────────────────────────────────────

class DkPropLinesTests(unittest.TestCase):
    def test_both_sides_at_a_line(self):
        rows = dk_prop_lines(_game([DK_BOOK]), "batter_hits")
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["player"], r["line"]), ("Bat", 1.5))
        self.assertEqual((r["over_price"], r["under_price"]), (-115, -105))

    def test_ignores_non_dk_books(self):
        # FanDuel posts a different (2.5) line — dk_prop_lines must return only
        # DraftKings' 1.5, never FanDuel's.
        rows = dk_prop_lines(_game([FD_BOOK, DK_BOOK]), "batter_hits")
        self.assertEqual([(r["line"], r["over_price"]) for r in rows],
                         [(1.5, -115)])

    def test_multiple_lines_all_kept(self):
        dk = {"key": "draftkings", "title": "DraftKings", "markets": [
            _mkt("batter_hits", [
                _oc("Bat", "Over", 0.5, -200), _oc("Bat", "Under", 0.5, 160),
                _oc("Bat", "Over", 1.5, -115), _oc("Bat", "Under", 1.5, -105)])]}
        rows = dk_prop_lines(_game([dk]), "batter_hits")
        self.assertEqual(sorted(r["line"] for r in rows), [0.5, 1.5])

    def test_absent_book_or_market_is_empty(self):
        self.assertEqual(dk_prop_lines(_game([FD_BOOK]), "batter_hits"), [])
        self.assertEqual(dk_prop_lines(_game([DK_BOOK]), "batter_walks"), [])
        self.assertEqual(dk_prop_lines({}, "batter_hits"), [])


# ── dk_game_lines (odds_client, team markets) ───────────────────────────────

def _team_oc(name, price, point=None):
    oc = {"name": name, "price": price}
    if point is not None:
        oc["point"] = point
    return oc


DK_TEAM_BOOK = {
    "key": "draftkings", "title": "DraftKings",
    "markets": [
        _mkt("h2h", [_team_oc("Home Team", -130), _team_oc("Away Team", 110)]),
        _mkt("spreads", [_team_oc("Home Team", -105, -1.5),
                         _team_oc("Away Team", -115, 1.5)]),
        _mkt("totals", [_team_oc("Over", -110, 8.5),
                        _team_oc("Under", -105, 8.5)]),
    ],
}
FD_TEAM_BOOK = {  # a non-DK book that must never leak into dk_game_lines
    "key": "fanduel", "title": "FanDuel",
    "markets": [_mkt("h2h", [_team_oc("Home Team", -150),
                             _team_oc("Away Team", 130)])],
}


class DkGameLinesTests(unittest.TestCase):
    def test_splits_all_three_markets(self):
        out = dk_game_lines(_game([DK_TEAM_BOOK]))
        self.assertEqual(out["moneyline"],
                         [{"team": "Home Team", "price": -130},
                          {"team": "Away Team", "price": 110}])
        self.assertEqual(out["spreads"],
                         [{"team": "Home Team", "point": -1.5, "price": -105},
                          {"team": "Away Team", "point": 1.5, "price": -115}])
        self.assertEqual(out["totals"],
                         [{"side": "Over", "point": 8.5, "price": -110},
                          {"side": "Under", "point": 8.5, "price": -105}])

    def test_ignores_non_dk_books(self):
        # FanDuel posts a different moneyline — dk_game_lines returns only DK's.
        out = dk_game_lines(_game([FD_TEAM_BOOK, DK_TEAM_BOOK]))
        self.assertEqual([o["price"] for o in out["moneyline"]], [-130, 110])

    def test_spread_without_point_dropped(self):
        dk = {"key": "draftkings", "title": "DraftKings", "markets": [
            _mkt("spreads", [{"name": "Home Team", "price": -110}])]}
        self.assertEqual(dk_game_lines(_game([dk]))["spreads"], [])

    def test_total_non_over_under_dropped(self):
        dk = {"key": "draftkings", "title": "DraftKings", "markets": [
            _mkt("totals", [{"name": "Yes", "point": 8.5, "price": -110}])]}
        self.assertEqual(dk_game_lines(_game([dk]))["totals"], [])

    def test_absent_book_is_empty(self):
        self.assertEqual(dk_game_lines(_game([FD_TEAM_BOOK])),
                         {"moneyline": [], "spreads": [], "totals": []})
        self.assertEqual(dk_game_lines({}),
                         {"moneyline": [], "spreads": [], "totals": []})


# ── dk_close_for_wager (selection + CLV math) ───────────────────────────────

class DkCloseForWagerTests(unittest.TestCase):
    def _offers(self):
        return [{"player": "Bat", "line": 1.5,
                 "over_price": -115, "under_price": -105}]

    def test_exact_line_over_computes_clv(self):
        row = {"player": "Bat", "direction": "OVER", "line": 1.5,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_wager(self._offers(), row)
        self.assertEqual((price, line), (-115, 1.5))
        expect = round((american_to_implied_prob(-115)
                        - american_to_implied_prob(-110)) * 100.0, 2)
        self.assertEqual(clv, expect)   # ~ +1.11 (executed had better odds)
        self.assertGreater(clv, 0)

    def test_exact_line_under_picks_under_side(self):
        row = {"player": "Bat", "direction": "UNDER", "line": 1.5,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_wager(self._offers(), row)
        self.assertEqual((price, line), (-105, 1.5))
        self.assertLess(clv, 0)         # -105 close is worse than -110 executed

    def test_line_moved_returns_none(self):
        # Bet was placed at 0.5 but DK's standard close is 1.5 — no exact-line
        # match. We DON'T stamp a mismatched line (the settled table shows only
        # clv_pct, and a stamped close_price would strand the row forever); we
        # leave it unfilled so it's retried free and picked up by a future
        # alternate-line pass.
        row = {"player": "Bat", "direction": "OVER", "line": 0.5,
               "executed_price": -110}
        self.assertIsNone(bk.dk_close_for_wager(self._offers(), row))

    def test_player_absent_returns_none(self):
        row = {"player": "Someone Else", "direction": "OVER", "line": 1.5,
               "executed_price": -110}
        self.assertIsNone(bk.dk_close_for_wager(self._offers(), row))

    def test_side_not_posted_returns_none(self):
        offers = [{"player": "Bat", "line": 1.5,
                   "over_price": -115, "under_price": None}]
        row = {"player": "Bat", "direction": "UNDER", "line": 1.5,
               "executed_price": -110}
        self.assertIsNone(bk.dk_close_for_wager(offers, row))

    def test_name_match_folds_accents_and_case(self):
        offers = [{"player": "José Ramírez", "line": 1.5,
                   "over_price": -115, "under_price": -105}]
        row = {"player": "jose ramirez", "direction": "OVER", "line": 1.5,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_wager(offers, row)
        self.assertEqual((price, line), (-115, 1.5))


# ── dk_close_for_team_wager (team-market selection + CLV math) ───────────────

class DkCloseForTeamWagerTests(unittest.TestCase):
    def _lines(self):
        return {
            "moneyline": [{"team": "Home Team", "price": -130},
                          {"team": "Away Team", "price": 110}],
            "spreads": [{"team": "Home Team", "point": -1.5, "price": -105},
                        {"team": "Away Team", "point": 1.5, "price": -115}],
            "totals": [{"side": "Over", "point": 8.5, "price": -110},
                       {"side": "Under", "point": 8.5, "price": -105}],
        }

    def test_moneyline_matches_team_no_line(self):
        row = {"bet_type": "moneyline", "team": "Home Team", "line": None,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_team_wager(self._lines(), row)
        self.assertEqual((price, line), (-130, None))  # moneyline has no line
        expect = round((american_to_implied_prob(-130)
                        - american_to_implied_prob(-110)) * 100.0, 2)
        self.assertEqual(clv, expect)

    def test_moneyline_folds_name_case_and_accents(self):
        row = {"bet_type": "moneyline", "team": "home team",
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_team_wager(self._lines(), row)
        self.assertEqual(price, -130)

    def test_moneyline_team_absent_returns_none(self):
        row = {"bet_type": "moneyline", "team": "Nobody FC",
               "executed_price": -110}
        self.assertIsNone(bk.dk_close_for_team_wager(self._lines(), row))

    def test_spread_exact_point_matches(self):
        row = {"bet_type": "spread", "team": "Away Team", "line": 1.5,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_team_wager(self._lines(), row)
        self.assertEqual((price, line), (-115, 1.5))

    def test_spread_line_moved_returns_none(self):
        # DK's featured close carries Home at -1.5, not the -2.5 that was bet.
        row = {"bet_type": "spread", "team": "Home Team", "line": -2.5,
               "executed_price": -110}
        self.assertIsNone(bk.dk_close_for_team_wager(self._lines(), row))

    def test_total_over_exact_point(self):
        row = {"bet_type": "total", "direction": "OVER", "line": 8.5,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_team_wager(self._lines(), row)
        self.assertEqual((price, line), (-110, 8.5))

    def test_total_under_uses_direction(self):
        row = {"bet_type": "total", "direction": "UNDER", "line": 8.5,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_team_wager(self._lines(), row)
        self.assertEqual((price, line), (-105, 8.5))

    def test_total_falls_back_to_side_when_no_direction(self):
        row = {"bet_type": "total", "side": "under", "line": 8.5,
               "executed_price": -110}
        price, line, clv = bk.dk_close_for_team_wager(self._lines(), row)
        self.assertEqual(price, -105)

    def test_total_line_moved_returns_none(self):
        row = {"bet_type": "total", "direction": "OVER", "line": 9.5,
               "executed_price": -110}
        self.assertIsNone(bk.dk_close_for_team_wager(self._lines(), row))

    def test_unknown_bet_type_returns_none(self):
        row = {"bet_type": "parlay", "executed_price": -110}
        self.assertIsNone(bk.dk_close_for_team_wager(self._lines(), row))


# ── tiny helpers ────────────────────────────────────────────────────────────

class HelperTests(unittest.TestCase):
    def test_same_sport(self):
        self.assertTrue(bk._same_sport("baseball_mlb", "baseball_mlb"))
        self.assertTrue(bk._same_sport("mlb", "baseball_mlb"))
        self.assertFalse(bk._same_sport("basketball_nba", "baseball_mlb"))
        self.assertFalse(bk._same_sport(None, "baseball_mlb"))

    def test_lines_equal(self):
        self.assertTrue(bk._lines_equal(1.5, "1.5"))
        self.assertFalse(bk._lines_equal(0.5, 1.5))
        self.assertFalse(bk._lines_equal(None, 1.5))

    def test_market_key(self):
        self.assertEqual(bk._market_key({"bet_type": "moneyline"}), "h2h")
        self.assertEqual(bk._market_key({"bet_type": "spread"}), "spreads")
        self.assertEqual(bk._market_key({"bet_type": "total"}), "totals")
        self.assertEqual(
            bk._market_key({"bet_type": "player_prop",
                            "prop_key": "batter_hits"}), "batter_hits")
        # A prop missing its prop_key can't be fetched -> None (skipped upstream).
        self.assertIsNone(bk._market_key({"bet_type": "player_prop"}))
        self.assertIsNone(bk._market_key({"bet_type": "parlay"}))


# ── CLI main() end-to-end (no network) ──────────────────────────────────────

def _wager(wid, eid, prop, player, direction, line, commence):
    return {"wager_id": wid, "bet_type": "player_prop",
            "sport_key": "baseball_mlb", "event_id": eid, "prop_key": prop,
            "player": player, "direction": direction, "line": line,
            "executed_price": -110, "commence_time": commence,
            "close_price": None}


def _ml_wager(wid, eid, team, commence):
    return {"wager_id": wid, "bet_type": "moneyline",
            "sport_key": "baseball_mlb", "event_id": eid, "team": team,
            "line": None, "point": None, "executed_price": -110,
            "commence_time": commence, "close_price": None}


def _spread_wager(wid, eid, team, line, commence):
    return {"wager_id": wid, "bet_type": "spread", "sport_key": "baseball_mlb",
            "event_id": eid, "team": team, "line": line, "point": line,
            "executed_price": -110, "commence_time": commence,
            "close_price": None}


def _total_wager(wid, eid, direction, line, commence):
    return {"wager_id": wid, "bet_type": "total", "sport_key": "baseball_mlb",
            "event_id": eid, "team": "Both teams", "direction": direction,
            "side": direction.lower(), "line": line, "point": line,
            "executed_price": -110, "commence_time": commence,
            "close_price": None}


# A DraftKings book carrying BOTH a prop market and the featured team markets,
# so a mixed prop+team event resolves from one payload.
DK_BOOK_ALL = {
    "key": "draftkings", "title": "DraftKings",
    "markets": DK_BOOK["markets"] + DK_TEAM_BOOK["markets"],
}


# All commence times are safely in the past so _commence_passed is True.
_C1 = "2024-07-20T18:00:00Z"
_C2 = "2024-07-21T18:00:00Z"


class MainTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            _wager("w1", "E1", "batter_hits", "Bat", "OVER", 1.5, _C1),
            _wager("w2", "E1", "pitcher_outs", "Pit", "UNDER", 17.5, _C1),
            _wager("w3", "E2", "batter_hits", "Slug", "OVER", 0.5, _C2),
        ]
        self.data = {
            "E1": _game([DK_BOOK], "E1"),
            "E2": _game([{"key": "draftkings", "title": "DraftKings", "markets": [
                _mkt("batter_hits", [
                    _oc("Slug", "Over", 0.5, 120),
                    _oc("Slug", "Under", 0.5, -150)])]}], "E2"),
        }

    def _run(self, argv, remaining=100000, cached=False):
        self.calls = []
        self.upcoming = Mock()

        def fake_fetch(api_key, sport, eid, date=None, regions=None,
                       markets=None, bookmakers=None):
            self.calls.append({"eid": eid, "date": date, "markets": markets,
                               "bookmakers": bookmakers, "sport": sport})
            return self.data.get(eid), "snap-ts"

        writer = Mock(side_effect=lambda filled: len(filled))
        with patch.object(bk, "load_config",
                          return_value={"odds_api_key": "k"}), \
             patch.object(db_store, "promote_secrets_from_toml",
                          return_value=True), \
             patch.object(db_store, "enabled", return_value=True), \
             patch.object(wagers, "read_wagers_with_status",
                          return_value=(self.rows, None)), \
             patch.object(wagers, "read_wagers", return_value=self.rows), \
             patch.object(bk, "get_historical_event_odds",
                          side_effect=fake_fetch), \
             patch.object(bk, "is_historical_event_cached",
                          return_value=cached), \
             patch.object(bk, "get_upcoming_events", self.upcoming), \
             patch.object(bk, "get_remaining_credits", return_value=remaining), \
             patch.object(wagers, "apply_clv_updates", writer), \
             patch.object(sys, "argv", argv):
            out = io.StringIO()
            with redirect_stdout(out):
                bk.main()
        self.writer = writer
        return out.getvalue()

    def test_one_call_per_event_with_market_union(self):
        self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        by_eid = {c["eid"]: c for c in self.calls}
        self.assertEqual(set(by_eid), {"E1", "E2"})
        self.assertEqual(by_eid["E1"]["markets"], "batter_hits,pitcher_outs")
        self.assertEqual(by_eid["E2"]["markets"], "batter_hits")
        for c in self.calls:
            self.assertEqual(c["bookmakers"], ["draftkings"])
            self.assertEqual(c["sport"], "baseball_mlb")
        # date passed is the wager's commence_time (nearest snapshot = the close).
        self.assertEqual(by_eid["E1"]["date"], _C1)
        self.assertEqual(by_eid["E2"]["date"], _C2)

    def test_fills_expected_clv(self):
        self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        filled = self.writer.call_args[0][0]
        # w1: exact line -> DK-vs-DK same-line CLV.
        self.assertEqual(filled["w1"]["close_price"], -115)
        self.assertEqual(filled["w1"]["close_line"], 1.5)
        self.assertIsNotNone(filled["w1"]["clv_pct"])
        # w2: pitcher_outs UNDER at 17.5 -> DK under price 100.
        self.assertEqual(filled["w2"]["close_price"], 100)
        self.assertEqual(filled["w2"]["close_line"], 17.5)
        # w3: exact 0.5 OVER -> DK over price 120.
        self.assertEqual(filled["w3"]["close_price"], 120)

    def test_budget_cap_trims_events(self):
        # E1 = 2 markets (20 cr), E2 = 1 market (10 cr). Freshest first is E2
        # (later commence); cap 15 fits E2 only, E1 (20) then busts the budget.
        self._run(["backfill_dk_clv.py", "--sport", "mlb",
                   "--max-credits", "15"])
        self.assertEqual([c["eid"] for c in self.calls], ["E2"])
        filled = self.writer.call_args[0][0]
        self.assertIn("w3", filled)
        self.assertNotIn("w1", filled)

    def test_cached_events_are_free_and_ignore_the_cap(self):
        # Everything already cached -> a re-run (or --refresh) must not bill the
        # budget, so even a tiny --max-credits still processes every event.
        out = self._run(["backfill_dk_clv.py", "--sport", "mlb",
                         "--max-credits", "5"], cached=True)
        self.assertEqual({c["eid"] for c in self.calls}, {"E1", "E2"})
        self.assertIn("Spent ~0 credits", out)

    def test_stable_market_set_includes_filled_sibling(self):
        # E1 has an unfilled batter_hits bet and an ALREADY-filled pitcher_outs
        # bet. The requested market set must still be the full union so the
        # permanent-cache key is identical to the first run (no re-bill).
        sibling = _wager("wf", "E1", "pitcher_outs", "Pit", "UNDER", 17.5, _C1)
        sibling["close_price"] = 100  # already has CLV
        self.rows = [
            _wager("w1", "E1", "batter_hits", "Bat", "OVER", 1.5, _C1),
            sibling,
        ]
        self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        by_eid = {c["eid"]: c for c in self.calls}
        self.assertEqual(by_eid["E1"]["markets"], "batter_hits,pitcher_outs")
        filled = self.writer.call_args[0][0]
        self.assertIn("w1", filled)
        self.assertNotIn("wf", filled)  # already filled — never reconsidered

    def test_already_filled_props_excluded(self):
        # The Blob/local read ignores the SQL where-filter and returns ALL rows;
        # collect must still drop already-filled props (idempotent re-runs).
        done = _wager("wdone", "E2", "batter_hits", "Slug", "OVER", 0.5, _C2)
        done["close_price"] = -110
        self.rows = [
            _wager("w1", "E1", "batter_hits", "Bat", "OVER", 1.5, _C1),
            done,
        ]
        self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        eids = {c["eid"] for c in self.calls}
        self.assertIn("E1", eids)
        self.assertNotIn("E2", eids)  # its only prop is already filled

    def test_line_moved_left_unfilled_end_to_end(self):
        # Bet at 2.5 but DK's standard close is 1.5 -> no exact match -> the row
        # is left blank (not written), not stamped with a mismatched line.
        self.rows = [_wager("w1", "E1", "batter_hits", "Bat", "OVER", 2.5, _C1)]
        out = self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        filled = self.writer.call_args[0][0]
        self.assertEqual(filled, {})
        self.assertIn("left blank", out)

    def test_reserve_triggers_free_preflight(self):
        # With a reserve floor and an unknown balance, a free events call runs
        # first so the reserve binds from the very first event.
        self._run(["backfill_dk_clv.py", "--sport", "mlb", "--reserve", "50"],
                  remaining=None)
        self.upcoming.assert_called_once()

    def test_no_preflight_without_reserve(self):
        self._run(["backfill_dk_clv.py", "--sport", "mlb"], remaining=None)
        self.upcoming.assert_not_called()

    def test_read_error_exits_nonzero(self):
        with patch.object(bk, "load_config",
                          return_value={"odds_api_key": "k"}), \
             patch.object(db_store, "promote_secrets_from_toml",
                          return_value=True), \
             patch.object(db_store, "enabled", return_value=True), \
             patch.object(wagers, "read_wagers_with_status",
                          return_value=([], RuntimeError("sql down"))), \
             patch.object(sys, "argv",
                          ["backfill_dk_clv.py", "--sport", "mlb"]):
            out = io.StringIO()
            with redirect_stdout(out):
                with self.assertRaises(SystemExit):
                    bk.main()
        self.assertIn("Could not read", out.getvalue())

    def test_dry_run_makes_no_calls(self):
        out = self._run(["backfill_dk_clv.py", "--sport", "mlb", "--dry-run"])
        self.assertEqual(self.calls, [])
        self.writer.assert_not_called()
        self.assertIn("dry-run", out.lower())

    def test_no_candidates_exits_clean(self):
        self.rows = []
        out = self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        self.assertEqual(self.calls, [])
        self.assertIn("Nothing to do", out)

    def test_prop_and_team_share_one_event_call(self):
        # A prop + moneyline + total on the SAME event -> ONE fetch whose market
        # union spans all three families, and each family gets its DK close.
        self.rows = [
            _wager("w1", "E1", "batter_hits", "Bat", "OVER", 1.5, _C1),
            _ml_wager("w2", "E1", "Home Team", _C1),
            _total_wager("w3", "E1", "OVER", 8.5, _C1),
        ]
        self.data = {"E1": _game([DK_BOOK_ALL], "E1")}
        self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        by_eid = {c["eid"]: c for c in self.calls}
        self.assertEqual(set(by_eid), {"E1"})
        # union of the three wagers' market keys (sorted), not the whole book.
        self.assertEqual(by_eid["E1"]["markets"], "batter_hits,h2h,totals")
        filled = self.writer.call_args[0][0]
        self.assertEqual(filled["w1"]["close_price"], -115)       # prop
        self.assertEqual((filled["w2"]["close_price"],
                          filled["w2"]["close_line"]), (-130, None))  # moneyline
        self.assertEqual((filled["w3"]["close_price"],
                          filled["w3"]["close_line"]), (-110, 8.5))   # total

    def test_spread_exact_line_filled(self):
        self.rows = [_spread_wager("w1", "E1", "Away Team", 1.5, _C1)]
        self.data = {"E1": _game([DK_TEAM_BOOK], "E1")}
        self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        by_eid = {c["eid"]: c for c in self.calls}
        self.assertEqual(by_eid["E1"]["markets"], "spreads")
        filled = self.writer.call_args[0][0]
        self.assertEqual((filled["w1"]["close_price"],
                          filled["w1"]["close_line"]), (-115, 1.5))

    def test_team_line_moved_left_unfilled(self):
        # Spread bet at -2.5 but DK's featured close is -1.5 -> no exact match ->
        # left blank (not stamped with a mismatched line).
        self.rows = [_spread_wager("w1", "E1", "Home Team", -2.5, _C1)]
        self.data = {"E1": _game([DK_TEAM_BOOK], "E1")}
        out = self._run(["backfill_dk_clv.py", "--sport", "mlb"])
        self.assertEqual(self.writer.call_args[0][0], {})
        self.assertIn("left blank", out)


if __name__ == "__main__":
    unittest.main()
