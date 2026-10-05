"""P1 of the stats.nba.com migration: the dep-free NBA-Stats loaders. No network (mocked)."""
import unittest
from unittest.mock import patch

import pandas as pd

import nba_stats_source as src


class HelperTests(unittest.TestCase):
    def test_minutes_to_float(self):
        self.assertEqual(src.minutes_to_float("35:56"), 35.93)
        self.assertEqual(src.minutes_to_float("0:00"), 0.0)
        self.assertEqual(src.minutes_to_float("40:00"), 40.0)
        self.assertIsNone(src.minutes_to_float(""))          # DNP
        self.assertIsNone(src.minutes_to_float(None))
        self.assertIsNone(src.minutes_to_float("nan"))
        self.assertEqual(src.minutes_to_float("37"), 37.0)   # plain numeric fallback

    def test_end_year_for_date(self):
        self.assertEqual(src.end_year_for_date("2024-11-01"), 2025)   # 2024-25 -> _2025
        self.assertEqual(src.end_year_for_date("2025-03-01"), 2025)
        self.assertEqual(src.end_year_for_date("2024-06-17"), 2024)   # Finals -> ending yr
        self.assertEqual(src.end_year_for_date("2024-08-01"), 2025)   # Aug rollover
        self.assertIsNone(src.end_year_for_date(None))

    def test_norm(self):
        self.assertEqual(src._norm("LeBron James"), "lebron james")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        src._CACHE.clear()

    def tearDown(self):
        src._CACHE.clear()

    def _pbox(self):
        return pd.DataFrame([
            {"person_id": 2544, "first_name": "LeBron", "family_name": "James",
             "team_id": 1610612747, "team_tricode": "LAL", "side": "home",
             "game_id": "0022400015", "minutes": "35:56", "points": 28},
            {"person_id": 201939, "first_name": "Stephen", "family_name": "Curry",
             "team_id": 1610612744, "team_tricode": "GSW", "side": "away",
             "game_id": "0022400015", "minutes": "", "points": 0},   # DNP
        ])

    def test_load_player_box_adds_norm(self):
        with patch("pandas.read_parquet", return_value=self._pbox()):
            df = src.load_player_box(2025, ttl=0)
        self.assertIn("player_norm", df.columns)
        self.assertEqual(list(df["player_norm"]), ["lebron james", "stephen curry"])
        self.assertEqual(df.iloc[0]["game_id"], "0022400015")       # string, zero-padded

    def test_missing_season_returns_none(self):
        with patch("nba_stats_source._fetch", return_value=None):
            self.assertIsNone(src.load_player_box(2099, ttl=0))
            self.assertIsNone(src.load_schedule(2099, ttl=0))

    def test_cache_hit_avoids_refetch(self):
        with patch("pandas.read_parquet", return_value=self._pbox()) as rp:
            src.load_player_box(2025)        # ttl default -> caches
            src.load_player_box(2025)        # served from cache
            self.assertEqual(rp.call_count, 1)

    def test_url_patterns(self):
        self.assertIn("nba_stats_player_boxscores/player_boxscores_{season}", src.PLAYER_BOX_URL)
        self.assertIn("nba_stats_schedules/nba_schedule_{season}", src.SCHEDULE_URL)
        self.assertIn("nba_stats_team_season_stats", src.TEAM_SEASON_URL)


if __name__ == "__main__":
    unittest.main()
