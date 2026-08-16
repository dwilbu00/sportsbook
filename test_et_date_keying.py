"""Tests for ET-normalized game keying (finding 7).

A night game rolls to the next UTC day, so a night game on ET-day-1 and the
afternoon game of the SAME matchup on ET-day-2 share a UTC calendar date. Keying
on the UTC prefix collapses them to one key, dropping one game's closing line.
game_key + backtest._et_date10 + _build_odds_lookup must key on the ET play date
so the two stay distinct. Offline (no SQL/network).
"""

import unittest

import historical_odds
import backtest


class EtDateKeyingTests(unittest.TestCase):
    def test_game_key_uses_et_play_date(self):
        # 02:00Z on the 12th = 10pm ET on the 11th → ET date is the 11th.
        night = historical_odds.game_key("2023-06-12T02:00:00Z", "Yankees", "Jays")
        self.assertTrue(night.startswith("2023-06-11|"), night)
        # 17:05Z on the 12th = 1pm ET on the 12th → ET date is the 12th.
        day = historical_odds.game_key("2023-06-12T17:05:00Z", "Yankees", "Jays")
        self.assertTrue(day.startswith("2023-06-12|"), day)

    def test_consecutive_day_same_matchup_do_not_collapse(self):
        # Same matchup, adjacent ET days, SAME UTC date (06-12) → distinct keys.
        a = historical_odds.game_key("2023-06-12T02:00:00Z", "Yankees", "Jays")  # ET 06-11
        b = historical_odds.game_key("2023-06-12T17:05:00Z", "Yankees", "Jays")  # ET 06-12
        self.assertNotEqual(a, b)

    def test_et_date10_helper(self):
        self.assertEqual(backtest._et_date10("2023-06-12T02:00:00Z"), "2023-06-11")
        self.assertEqual(backtest._et_date10("2023-06-12T17:05:00Z"), "2023-06-12")
        # Fallback: a bare date / unparseable value returns its own prefix.
        self.assertEqual(backtest._et_date10("2023-06-12"), "2023-06-12")

    def test_build_odds_lookup_distinct_id_keys_for_adjacent_games(self):
        # Two store entries, same matchup + same UTC date, different ET dates →
        # _build_odds_lookup must emit TWO distinct id-keyed entries, not overwrite.
        store = {"games": {
            "k1": {"commence_time": "2023-06-12T02:00:00Z",  # ET 06-11
                   "home_team": "Yankees", "away_team": "Jays",
                   "home_code": "NYY", "away_code": "TOR",
                   "moneyline": {"Yankees": [{"price": -150}]},
                   "spreads": {}, "totals": {}, "event_id": "e_night"},
            "k2": {"commence_time": "2023-06-12T17:05:00Z",  # ET 06-12
                   "home_team": "Yankees", "away_team": "Jays",
                   "home_code": "NYY", "away_code": "TOR",
                   "moneyline": {"Yankees": [{"price": -120}]},
                   "spreads": {}, "totals": {}, "event_id": "e_day"},
        }}
        lookup, _ = backtest._build_odds_lookup(store, {})
        id_keys = [k for k in lookup if isinstance(k, tuple) and k[0] == "id"]
        self.assertIn(("id", "2023-06-11", "NYY", "TOR"), id_keys)
        self.assertIn(("id", "2023-06-12", "NYY", "TOR"), id_keys)
        self.assertEqual(lookup[("id", "2023-06-11", "NYY", "TOR")]["event_id"],
                         "e_night")
        self.assertEqual(lookup[("id", "2023-06-12", "NYY", "TOR")]["event_id"],
                         "e_day")


if __name__ == "__main__":
    unittest.main()
