"""_load_player_gamelog in-process memo — guards the forward-grading speedup.

The resolve loop calls _load_player_gamelog once per ROW; without the memo each call is
a fresh Azure/ESPN read of the whole gamelog (~0.7s), so grading N rows for a player paid
N reads (the 113s->0.1s bottleneck). This pins: (a) a 2nd load within the TTL hits the memo
(one underlying fetch), and (b) resolve_pending_outcomes clears the memo per pass.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_gamelog_memo -v
"""
import unittest
from unittest.mock import patch

import recalibration as rc


class GamelogMemoTests(unittest.TestCase):
    def setUp(self):
        rc._GAMELOG_MEMO.clear()

    def tearDown(self):
        rc._GAMELOG_MEMO.clear()

    def test_second_load_hits_memo(self):
        rows = [{"game_date": "2026-09-21T18:00:00Z", "receptions": 5.0,
                 "receivingYards": 62.0, "completed": True}]
        with patch("espn_cache.cached_athlete_id", return_value="a1"), \
                patch("espn_cache.cached_gamelog", return_value=rows) as cg:
            gl1, bd1 = rc._load_player_gamelog("football", "nfl", "Player One")
            gl2, bd2 = rc._load_player_gamelog("football", "nfl", "Player One")   # memo
        self.assertEqual(cg.call_count, 1)          # underlying fetch happened ONCE
        self.assertEqual(gl1, gl2)
        self.assertEqual(bd1["2026-09-21"], bd2["2026-09-21"])

    def test_miss_is_memoized_too(self):
        with patch("espn_cache.cached_athlete_id", return_value=None) as cid:
            rc._load_player_gamelog("football", "nfl", "Ghost")
            rc._load_player_gamelog("football", "nfl", "Ghost")
        self.assertEqual(cid.call_count, 1)         # a not-found player isn't re-looked-up

    def test_distinct_players_are_separate(self):
        with patch("espn_cache.cached_athlete_id", side_effect=lambda *a: a[-1]), \
                patch("espn_cache.cached_gamelog",
                      return_value=[{"game_date": "2026-09-21T18:00:00Z", "TD": 1.0,
                                     "completed": True}]) as cg:
            rc._load_player_gamelog("football", "nfl", "A")
            rc._load_player_gamelog("football", "nfl", "B")
        self.assertEqual(cg.call_count, 2)          # different players -> separate fetches


if __name__ == "__main__":
    unittest.main(verbosity=2)
