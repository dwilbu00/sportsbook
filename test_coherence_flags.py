"""Unit tests for coherence_flags — pure daily-flag selection + the in-app adapter.

Most tests disable the moderate-favorite band (fav_min=0.0, fav_max=1.0) so they
exercise the underlying EV / offset / shape mechanics independent of the sharpening
gate; SharpeningGateTests covers the band + dog-only gate itself.
"""
import unittest

import coherence_flags as cf
import r2_data

# Disable the favorite band so a flag appears regardless of the fixture's ML.
_NO_BAND = dict(fav_min=0.0, fav_max=1.0)


def _triad(ml_home=-150, ml_away=+130, rl_point=-1.5, rl_home=+120, rl_away=-140,
           total_over=-110, total_under=-110, total_line=8.5):
    return r2_data.TeamTriad(
        event_id="G1", game_date="2026-08-27", commence_time="x", snapshot_id=2,
        game_pk=500, home="Mets", away="Yankees", ml_home=ml_home, ml_away=ml_away,
        rl_home_point=rl_point, rl_home=rl_home, rl_away=rl_away,
        total_line=total_line, total_over=total_over, total_under=total_under)


class FlagGamesTests(unittest.TestCase):
    def test_flags_have_actionable_fields(self):
        # Low floor so a side flags; check the record is bettable.
        flags = cf.flag_games([_triad()], offset=0.0, ev_floor=-10.0, **_NO_BAND)
        self.assertTrue(flags)
        f = flags[0]
        for k in ("side", "team", "point", "dk_price", "ev", "coherent_fair"):
            self.assertIn(k, f)
        self.assertIn(f["side"], ("home", "away"))
        # Sharpened: only the underdog +1.5 side is ever flagged.
        self.assertTrue(all(x["point"] > 0 for x in flags))

    def test_high_floor_flags_nothing(self):
        self.assertEqual(
            cf.flag_games([_triad()], offset=0.0, ev_floor=0.95, **_NO_BAND), [])

    def test_offset_shifts_fair(self):
        # A positive offset lowers implied home cover -> raises the away-side fair by
        # the same amount. Assert the calibrated fair actually moves with the offset.
        base = cf.flag_games([_triad()], offset=0.0, ev_floor=-10.0, **_NO_BAND)
        shifted = cf.flag_games([_triad()], offset=0.10, ev_floor=-10.0, **_NO_BAND)
        b_away = next(f for f in base if f["side"] == "away")
        s_away = next(f for f in shifted if f["side"] == "away")
        self.assertAlmostEqual(s_away["coherent_fair"] - b_away["coherent_fair"],
                               0.10, places=3)

    def test_sorted_by_ev_desc(self):
        flags = cf.flag_games([_triad(), _triad(rl_home=+180, rl_away=-220)],
                              offset=0.0, ev_floor=-10.0, **_NO_BAND)
        evs = [f["ev"] for f in flags]
        self.assertEqual(evs, sorted(evs, reverse=True))


def _game(book="draftkings", home="Mets", away="Yankees"):
    return {
        "id": "E1", "commence_time": "2026-08-27T23:05:00Z",
        "home_team": home, "away_team": away,
        "bookmakers": [{"key": book, "markets": [
            {"key": "h2h", "outcomes": [
                {"name": home, "price": -150}, {"name": away, "price": +130}]},
            {"key": "spreads", "outcomes": [
                {"name": home, "price": +120, "point": -1.5},
                {"name": away, "price": -140, "point": +1.5}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": -110, "point": 8.5},
                {"name": "Under", "price": -110, "point": 8.5}]},
        ]}]}


def _game_odds(home="Mets", away="Yankees", ml_home=-150, ml_away=+130,
               rl_home=(-1.5, +120), rl_away=(+1.5, -140),
               over=(8.5, -110), under=(8.5, -110)):
    """odds_client.parse_game_odds shape (DK-only view)."""
    return {
        "game_id": "E1", "home_team": home, "away_team": away,
        "commence_time": "2026-08-27T23:05:00Z",
        "moneyline": {home: [{"book": "draftkings", "price": ml_home}],
                      away: [{"book": "draftkings", "price": ml_away}]},
        "spreads": {home: [{"book": "draftkings", "spread": rl_home[0], "price": rl_home[1]}],
                    away: [{"book": "draftkings", "spread": rl_away[0], "price": rl_away[1]}]},
        "totals": {"Over": [{"book": "draftkings", "line": over[0], "price": over[1]}],
                   "Under": [{"book": "draftkings", "line": under[0], "price": under[1]}]},
    }


class RunLineCandidatesTests(unittest.TestCase):
    def test_flags_positive_ev_side_with_spread_shape(self):
        # Very low floor so a side flags; check it's spread-shaped + coherence-tagged.
        cands = cf.run_line_candidates(_game_odds(), offset=0.0, ev_floor=-10.0,
                                       **_NO_BAND)
        self.assertTrue(cands)
        c = cands[0]
        self.assertEqual(c["type"], "runline_coherence")
        self.assertEqual(c["model_source"], "coherence")
        for k in ("team", "opponent", "home_away", "spread", "cover_rate",
                  "implied_prob", "edge_pct", "expected_roi_pct", "price", "matchup"):
            self.assertIn(k, c)
        self.assertIn(c["home_away"], ("HOME", "AWAY"))
        self.assertTrue(c["is_value"])
        self.assertEqual(abs(c["spread"]), 1.5)
        # Sharpened: only the underdog +1.5 side.
        self.assertTrue(all(x["spread"] > 0 for x in cands))

    def test_high_floor_flags_nothing(self):
        self.assertEqual(cf.run_line_candidates(_game_odds(), offset=0.0,
                                                ev_floor=0.95, **_NO_BAND), [])

    def test_missing_market_returns_empty(self):
        g = _game_odds()
        del g["totals"]["Over"]        # no total -> can't solve run means
        self.assertEqual(
            cf.run_line_candidates(g, offset=0.0, ev_floor=-10.0, **_NO_BAND), [])

    def test_non_runline_spread_ignored(self):
        # A mismatched pair (home -1.5, away +2.5) is rejected.
        g = _game_odds(rl_away=(+2.5, -140))
        self.assertEqual(
            cf.run_line_candidates(g, offset=0.0, ev_floor=-10.0, **_NO_BAND), [])

    def test_offset_shifts_fair(self):
        base = cf.run_line_candidates(_game_odds(), offset=0.0, ev_floor=-10.0,
                                      **_NO_BAND)
        shifted = cf.run_line_candidates(_game_odds(), offset=0.10, ev_floor=-10.0,
                                         **_NO_BAND)
        b_away = next(c for c in base if c["home_away"] == "AWAY")
        s_away = next(c for c in shifted if c["home_away"] == "AWAY")
        # +offset lowers implied home cover -> raises away cover by the same amount.
        self.assertAlmostEqual((s_away["cover_rate"] - b_away["cover_rate"]) / 100.0,
                               0.10, places=2)

    def test_candidate_feeds_checklist_and_selector(self):
        import analysis
        import bet_selector
        c = cf.run_line_candidates(_game_odds(), offset=0.0, ev_floor=-10.0,
                                   **_NO_BAND)[0]
        c["event_id"] = "E1"
        entry = analysis.make_bet_checklist_entry(c, "runline_coherence")
        self.assertTrue(entry["selection_key"].startswith("bet_selection:"))
        self.assertIn("runline_coherence", entry["selection_key"])
        # selector can rank + build a conflict leg without raising
        self.assertIsNotNone(bet_selector._prob("runline_coherence", None, c))
        leg = bet_selector._leg("runline_coherence", None, c)
        self.assertEqual(leg["bet_type"], "spread")   # conflicts as a spread


class SharpeningGateTests(unittest.TestCase):
    """The 2026-08-28 sharpening: dog +1.5 side only, moderate-favorite band."""

    def test_dog_only_never_flags_favorite_side(self):
        # Band disabled + low floor: still only the underdog +1.5 side is returned.
        flags = cf.flag_games([_triad()], offset=0.0, ev_floor=-10.0, **_NO_BAND)
        self.assertTrue(flags)
        self.assertTrue(all(f["point"] > 0 for f in flags))
        self.assertTrue(all(f["side"] == "away" for f in flags))  # away is the +1.5 dog
        cands = cf.run_line_candidates(_game_odds(), offset=0.0, ev_floor=-10.0,
                                       **_NO_BAND)
        self.assertTrue(cands)
        self.assertTrue(all(c["spread"] > 0 for c in cands))

    def test_in_band_moderate_favorite_flags(self):
        # -190/+160 devigs to a ~63% favorite -> inside the default [0.60,0.70) band.
        t = _triad(ml_home=-190, ml_away=+160)
        self.assertTrue(cf.flag_games([t], offset=0.0, ev_floor=-10.0))
        g = _game_odds(ml_home=-190, ml_away=+160)
        self.assertTrue(cf.run_line_candidates(g, offset=0.0, ev_floor=-10.0))

    def test_heavy_favorite_out_of_band_dropped(self):
        # -350/+280 devigs to a ~75% favorite -> above the band -> nothing (default).
        t = _triad(ml_home=-350, ml_away=+280)
        self.assertEqual(cf.flag_games([t], offset=0.0, ev_floor=-10.0), [])
        g = _game_odds(ml_home=-350, ml_away=+280)
        self.assertEqual(cf.run_line_candidates(g, offset=0.0, ev_floor=-10.0), [])

    def test_near_pickem_out_of_band_dropped(self):
        # -120/+100 devigs to a ~52% favorite -> below the band -> nothing (default).
        t = _triad(ml_home=-120, ml_away=+100)
        self.assertEqual(cf.flag_games([t], offset=0.0, ev_floor=-10.0), [])

    def test_band_can_be_disabled(self):
        # A heavy favorite that the default band drops is flagged when the band is off.
        t = _triad(ml_home=-350, ml_away=+280)
        self.assertTrue(cf.flag_games([t], offset=0.0, ev_floor=-10.0, **_NO_BAND))


class TriadsFromUpcomingTests(unittest.TestCase):
    def test_parses_complete_triad(self):
        triads, stats = cf.triads_from_upcoming([_game()])
        self.assertEqual(len(triads), 1)
        t = triads[0]
        self.assertEqual((t.home, t.away), ("Mets", "Yankees"))
        self.assertEqual((t.ml_home, t.ml_away), (-150, +130))
        self.assertEqual((t.rl_home_point, t.rl_home, t.rl_away), (-1.5, +120, -140))
        self.assertEqual((t.total_line, t.total_over, t.total_under), (8.5, -110, -110))
        self.assertEqual(t.game_date, "2026-08-27")
        self.assertEqual(stats["triads_built"], 1)

    def test_drops_when_book_absent(self):
        triads, stats = cf.triads_from_upcoming([_game(book="fanduel")])
        self.assertEqual(triads, [])
        self.assertEqual(stats["events_dropped_no_book"], 1)

    def test_drops_incomplete_triad(self):
        g = _game()
        g["bookmakers"][0]["markets"] = g["bookmakers"][0]["markets"][:2]  # no totals
        triads, stats = cf.triads_from_upcoming([g])
        self.assertEqual(triads, [])
        self.assertEqual(stats["events_dropped_incomplete_triad"], 1)

    def test_flags_run_on_parsed_triads(self):
        # End-to-end: parsed live triads feed flag_games unchanged (band disabled).
        triads, _ = cf.triads_from_upcoming([_game()])
        flags = cf.flag_games(triads, offset=0.0, ev_floor=-10.0, **_NO_BAND)
        self.assertTrue(flags)


if __name__ == "__main__":
    unittest.main()
