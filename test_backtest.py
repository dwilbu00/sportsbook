"""Tests for backtest.py's team-market join, exercised against in-memory SQLite
so the live Azure DB is never touched.

The SFBB team map is seeded from the same CSV fixture the other suites use
(reusing test_backfill_player_ids._Backend), so player_id_map.team_code_for_name
resolves real codes (CLE/ARI/…). Covers the id-preferring join that fixes the
lossy exact→ci→substring name match (e.g. Indians→Guardians)."""

import unittest

import backtest
from test_backfill_player_ids import _Backend


def _entry(home, away, commence="2026-07-22T23:00:00Z"):
    return {"commence_time": commence, "home_team": home, "away_team": away,
            "moneyline": {home: [{"book": "b", "price": -110}]},
            "spreads": {}, "totals": {}}


class BacktestTeamCodeJoinTests(_Backend, unittest.TestCase):

    def test_lookup_prefers_code_across_franchise_rename(self):
        # Odds store carries the legacy spelling "Cleveland Indians"; the ESPN
        # schedule uses "Cleveland Guardians". The name match (exact→ci→substring)
        # can't bridge that, but both resolve to code CLE → the id key joins them.
        store = {"games": {"g1": _entry("Cleveland Indians",
                                        "Arizona Diamondbacks")}}
        espn_teams = {"Cleveland Guardians": "5", "Arizona Diamondbacks": "29"}
        lookup, unmatched = backtest._build_odds_lookup(store, espn_teams)
        self.assertEqual(unmatched, 1)                 # name path missed
        self.assertIn(("id", "2026-07-22", "CLE", "ARI"), lookup)
        hit = backtest._lookup_game_odds(
            lookup, "2026-07-22", "Cleveland Guardians", "Arizona Diamondbacks")
        self.assertIsNotNone(hit)                       # recovered via code
        self.assertEqual(hit["home_team"], "Cleveland Indians")

    def test_falls_back_to_name_when_code_unresolved(self):
        # Non-MLB teams don't resolve to a SFBB code → no id key; the join must
        # still work through the exact name key (zero regression).
        store = {"games": {"g1": _entry("Sacramento Kings", "Toronto Raptors")}}
        espn_teams = {"Sacramento Kings": "1", "Toronto Raptors": "2"}
        lookup, unmatched = backtest._build_odds_lookup(store, espn_teams)
        self.assertEqual(unmatched, 0)
        self.assertFalse(any(k[0] == "id" for k in lookup))   # no code key
        hit = backtest._lookup_game_odds(
            lookup, "2026-07-22", "Sacramento Kings", "Toronto Raptors")
        self.assertIsNotNone(hit)

    def test_build_emits_both_key_kinds(self):
        store = {"games": {"g1": _entry("Cleveland Guardians",
                                        "Arizona Diamondbacks")}}
        espn_teams = {"Cleveland Guardians": "5", "Arizona Diamondbacks": "29"}
        lookup, unmatched = backtest._build_odds_lookup(store, espn_teams)
        self.assertEqual(unmatched, 0)
        self.assertIn(
            ("2026-07-22", "Cleveland Guardians", "Arizona Diamondbacks"), lookup)
        self.assertIn(("id", "2026-07-22", "CLE", "ARI"), lookup)

    def test_surfaced_codes_take_precedence(self):
        # When the SQL warehouse already surfaced home_code/away_code on the entry
        # they're used directly (no name resolution needed).
        entry = _entry("Whatever A", "Whatever B")
        entry["home_code"], entry["away_code"] = "CLE", "ARI"
        store = {"games": {"g1": entry}}
        lookup, _ = backtest._build_odds_lookup(store, {})
        self.assertIn(("id", "2026-07-22", "CLE", "ARI"), lookup)


if __name__ == "__main__":
    unittest.main()
