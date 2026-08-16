"""Tests for the SBR team-market ingest transform (ingest_sbr_odds).

Pure/offline — no SQL, no network. Exercises the DK currentLine extraction, the
Odds-API v4 payload shape (so warehouse.capture_event_odds/parse_game_odds read
it), the game-type / year / final-status filtering, the A's rebrand cleanup, and
the doubleheader id guard. Also round-trips a built payload through the real
odds_client.parse_game_odds + warehouse._emit_team_lines to prove the shape.
"""

import unittest

import ingest_sbr_odds as ing


def _game(away_full="Toronto Blue Jays", away_short="TOR",
          home_full="New York Yankees", home_short="NYY",
          start="2023-06-12T17:05:00+00:00", gtype="R", status="Final",
          dk_ml=True, dk_ps=True, dk_tt=True, other_book=True):
    ml, ps, tt = [], [], []
    if other_book:
        ml.append({"sportsbook": "fanduel",
                   "currentLine": {"homeOdds": -190, "awayOdds": 160}})
    if dk_ml:
        ml.append({"sportsbook": "draftkings",
                   "openingLine": {"homeOdds": -175, "awayOdds": 148},
                   "currentLine": {"homeOdds": -195, "awayOdds": 165}})
    if dk_ps:
        ps.append({"sportsbook": "draftkings",
                   "currentLine": {"homeOdds": 100, "awayOdds": -118,
                                   "homeSpread": -1.5, "awaySpread": 1.5}})
    if dk_tt:
        tt.append({"sportsbook": "draftkings",
                   "currentLine": {"overOdds": -105, "underOdds": -117,
                                   "total": 8.5}})
    return {"gameView": {"startDate": start, "gameType": gtype,
                         "gameStatusText": status,
                         "awayTeam": {"fullName": away_full, "shortName": away_short},
                         "homeTeam": {"fullName": home_full, "shortName": home_short}},
            "odds": {"moneyline": ml, "pointspread": ps, "totals": tt}}


class TransformTests(unittest.TestCase):
    def test_dk_current_picks_draftkings(self):
        offers = [{"sportsbook": "fanduel", "currentLine": {"homeOdds": -100}},
                  {"sportsbook": "draftkings", "currentLine": {"homeOdds": -195}}]
        self.assertEqual(ing._dk_current(offers), {"homeOdds": -195})
        self.assertIsNone(ing._dk_current([{"sportsbook": "caesars",
                                            "currentLine": {}}]))
        self.assertIsNone(ing._dk_current([]))

    def test_num_rejects_bool_and_str(self):
        self.assertEqual(ing._num(-195), -195)
        self.assertEqual(ing._num(1.5), 1.5)
        self.assertIsNone(ing._num(True))
        self.assertIsNone(ing._num("−195"))
        self.assertIsNone(ing._num(None))

    def test_clean_team_athletics_rebrand(self):
        self.assertEqual(ing._clean_team("Athletics Athletics"), "Athletics")
        self.assertEqual(ing._clean_team("Oakland Athletics"), "Oakland Athletics")
        self.assertEqual(ing._clean_team("New York Yankees"), "New York Yankees")

    def test_build_payload_shape_all_markets(self):
        gv = _game()["gameView"]
        ml = {"homeOdds": -195, "awayOdds": 165}
        ps = {"homeOdds": 100, "awayOdds": -118, "homeSpread": -1.5, "awaySpread": 1.5}
        tt = {"overOdds": -105, "underOdds": -117, "total": 8.5}
        payload, present = ing.build_payload("sbr-x", gv, ml, ps, tt)
        self.assertEqual(present, ["moneyline", "spread", "total"])
        self.assertEqual(payload["id"], "sbr-x")
        self.assertEqual(payload["home_team"], "New York Yankees")
        self.assertEqual(payload["away_team"], "Toronto Blue Jays")
        self.assertEqual(payload["commence_time"], "2023-06-12T17:05:00+00:00")
        books = payload["bookmakers"]
        self.assertEqual([b["key"] for b in books], ["draftkings"])
        mkeys = {m["key"] for m in books[0]["markets"]}
        self.assertEqual(mkeys, {"h2h", "spreads", "totals"})

    def test_build_payload_skips_incomplete_market(self):
        gv = _game()["gameView"]
        # spread missing awaySpread → dropped; totals missing → dropped; ml ok.
        payload, present = ing.build_payload(
            "sbr-x", gv,
            {"homeOdds": -195, "awayOdds": 165},
            {"homeOdds": 100, "awayOdds": -118, "homeSpread": -1.5},
            None)
        self.assertEqual(present, ["moneyline"])
        self.assertEqual({m["key"] for m in payload["bookmakers"][0]["markets"]},
                         {"h2h"})

    def test_build_payload_none_when_no_market(self):
        gv = _game()["gameView"]
        payload, present = ing.build_payload("sbr-x", gv, None, None, None)
        self.assertIsNone(payload)
        self.assertEqual(present, [])

    def test_scan_filters_year_type_final_and_dk(self):
        data = {
            "2022-05-01": [_game(start="2022-05-01T17:00:00+00:00")],      # pre-year
            "2023-05-01": [_game(start="2023-05-01T17:00:00+00:00")],      # keep
            "2023-05-02": [_game(start="2023-05-02T17:00:00+00:00",
                                 gtype="S")],                             # spring
            "2023-05-03": [_game(start="2023-05-03T17:00:00+00:00",
                                 status="Postponed")],                     # not final
            "2023-05-04": [_game(start="2023-05-04T17:00:00+00:00",
                                 dk_ml=False, dk_ps=False, dk_tt=False)],  # no DK
            "2024-10-01": [_game(start="2024-10-01T17:00:00+00:00",
                                 gtype="W")],                             # postseason keep
        }
        cands, skips = ing.scan(data, min_year=2023)
        years = sorted(c["year"] for c in cands)
        self.assertEqual(years, [2023, 2024])
        self.assertEqual(skips["pre_year"], 1)
        self.assertEqual(skips["wrong_type"], 1)
        self.assertEqual(skips["not_final"], 1)
        self.assertEqual(skips["no_dk_line"], 1)

    def test_scan_game_types_all(self):
        data = {"2023-03-01": [_game(start="2023-03-01T17:00:00+00:00",
                                     gtype="S")]}
        cands, _ = ing.scan(data, min_year=2023, allowed_types=None)
        self.assertEqual(len(cands), 1)

    def test_doubleheader_id_guard(self):
        # Same matchup + date twice → the second gets a -g2 suffix, deterministically.
        g = _game(start="2023-07-04T17:00:00+00:00")
        g2 = _game(start="2023-07-04T20:00:00+00:00")
        data = {"2023-07-04": [g, g2]}
        cands, _ = ing.scan(data, min_year=2023)
        ids = [c["event_id"] for c in cands]
        self.assertEqual(ids, ["sbr-2023-07-04-TOR-NYY", "sbr-2023-07-04-TOR-NYY-g2"])

    def test_payload_roundtrips_through_parse_and_emit(self):
        """The built payload must parse + emit the exact team lines the warehouse
        write path (capture_event_odds → _enumerate_lines) depends on."""
        import odds_client
        import warehouse
        gv = _game()["gameView"]
        payload, _ = ing.build_payload(
            "sbr-rt", gv,
            {"homeOdds": -195, "awayOdds": 165},
            {"homeOdds": 100, "awayOdds": -118, "homeSpread": -1.5, "awaySpread": 1.5},
            {"overOdds": -105, "underOdds": -117, "total": 8.5})
        parsed = odds_client.parse_game_odds(payload)
        lines = []
        warehouse._emit_team_lines(parsed, lines)
        by = {(ln["bet_type"], ln["selection"]): ln for ln in lines}
        self.assertEqual(by[("moneyline", "New York Yankees")]["price"], -195)
        self.assertEqual(by[("moneyline", "Toronto Blue Jays")]["price"], 165)
        self.assertEqual(by[("spread", "New York Yankees")]["point"], -1.5)
        self.assertEqual(by[("total", "Over")]["point"], 8.5)
        self.assertEqual(by[("total", "Over")]["price"], -105)
        self.assertEqual(by[("total", "Under")]["price"], -117)


if __name__ == "__main__":
    unittest.main()
