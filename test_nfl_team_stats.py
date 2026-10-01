"""Tests for nfl_team_stats — NFL team-market {season, recent, recent_games} from the
nflverse games spine (no ESPN). Hermetic: nfl_schedule.load_games is mocked; nfl_epa's
static 32-team name↔abbr map is real."""
import unittest
from unittest.mock import patch

import nfl_team_stats as nts


def _g(season, gameday, home_ab, away_ab, hs, as_):
    return {"season": season, "gameday": gameday, "week": 1,
            "home_team": home_ab, "away_team": away_ab,
            "home_score": hs, "away_score": as_}


class TeamStatsTests(unittest.TestCase):
    def _games(self):
        return [
            _g("2026", "2026-09-07", "DAL", "NYG", 24, 20),    # DAL home win
            _g("2026", "2026-09-14", "PHI", "DAL", 17, 31),    # DAL away win
            _g("2026", "2026-09-21", "DAL", "WAS", 10, 27),    # DAL home loss
            _g("2026", "2026-09-28", "GB", "DAL", 28, 14),     # DAL away loss
            _g("2025", "2025-12-01", "DAL", "NYG", 30, 10),    # prior season (form only)
            _g("2026", "2026-10-05", "DAL", "KC", None, None),  # unplayed → excluded
        ]

    def test_shape_record_and_keying(self):
        with patch("nfl_schedule.load_games", return_value=self._games()):
            s = nts.team_stats("Dallas Cowboys", recent_n=4, seasons=["2026", "2025"])
        self.assertIsNotNone(s)
        # Current-season (2026) completed: W NYG, W PHI, L WAS, L GB = 2-2.
        self.assertEqual(s["season"]["record"], "2-2")
        self.assertAlmostEqual(s["season"]["win_pct"], 0.5)
        self.assertIsNone(s["season"]["runs_scored"])          # NFL has no runs
        # Most-recent-first; the queried team is keyed by its ODDS name in every game.
        self.assertEqual(s["recent_games"][0]["date"], "2026-09-28")
        for g in s["recent_games"]:
            self.assertIn("Dallas Cowboys", (g["home_team"], g["away_team"]))
            self.assertEqual(g["total_score"], g["home_score"] + g["away_score"])
        # Unplayed game excluded.
        self.assertTrue(all(g["date"] != "2026-10-05" for g in s["recent_games"]))

    def test_recent_window_respected(self):
        with patch("nfl_schedule.load_games", return_value=self._games()):
            s = nts.team_stats("Dallas Cowboys", recent_n=4, seasons=["2026", "2025"])
        self.assertEqual(s["recent"]["games"], 4)              # last 4 of 5 completed
        # last 4 (2026 only): 2 W / 2 L.
        self.assertEqual((s["recent"]["wins"], s["recent"]["losses"]), (2, 2))

    def test_opponent_gets_full_name(self):
        with patch("nfl_schedule.load_games", return_value=self._games()):
            s = nts.team_stats("Dallas Cowboys", recent_n=4, seasons=["2026"])
        latest = s["recent_games"][0]                          # GB @ DAL, DAL away
        self.assertEqual(latest["away_team"], "Dallas Cowboys")
        self.assertEqual(latest["home_team"], "Green Bay Packers")  # abbr→full

    def test_unresolved_team_is_none(self):
        with patch("nfl_schedule.load_games", return_value=self._games()):
            self.assertIsNone(nts.team_stats("Nonexistent Team"))

    def test_no_games_is_none(self):
        with patch("nfl_schedule.load_games", return_value=[]):
            self.assertIsNone(nts.team_stats("Dallas Cowboys", seasons=["2026"]))


if __name__ == "__main__":
    unittest.main()
