"""NBA Phase 4 grading/analysis cutover (off ESPN):
  * recalibration._resolve_nba_actual — warehouse-first, live-fallback, DNP-safe.
  * recalibration._nba_dnp_row / _nba_is_dnp — confirmed-DNP void signal.
  * game_results._nba_final_score / final_score — team scores off the schedule spine.
  * nba_team_stats.team_stats — ESPN-free team form from the schedule spine.
All fetchers mocked — no network.
"""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import game_results as gr
import nba_team_stats as nts
import recalibration as rc


def _wh_frame(pnorm, game_date, played=True, **stats):
    row = {"player_norm": pnorm, "game_date": game_date, "played": played}
    row.update(stats)
    return pd.DataFrame([row])


def _live_box(pnorm, game_date, minutes=30.0, **stats):
    row = {"player_norm": pnorm, "game_date": game_date, "minutes": minutes}
    row.update(stats)
    return pd.DataFrame([row])


class ResolveNbaActualTests(unittest.TestCase):
    def _resolve(self, prop, name, wh, live, game_date="2024-01-15"):
        with patch("recalibration._nba_player_game_frame", return_value=wh), \
             patch("nba_source.load_player_box", return_value=live):
            return rc._resolve_nba_actual(prop, name, game_date)

    def test_warehouse_used_when_present(self):
        wh = _wh_frame("lebron james", "2024-01-15", points=30.0)
        self.assertEqual(self._resolve("player_points", "LeBron James", wh, None), 30.0)

    def test_combo_prop_sums(self):
        wh = _wh_frame("lebron james", "2024-01-15", points=30.0, rebounds=8.0, assists=10.0)
        self.assertEqual(
            self._resolve("player_points_rebounds_assists", "LeBron James", wh, None), 48.0)

    def test_dnp_returns_none_not_phantom_zero(self):
        wh = _wh_frame("role player", "2024-01-15", played=False, points=0.0)
        self.assertIsNone(self._resolve("player_points", "Role Player", wh, None))

    def test_falls_back_to_live_when_warehouse_none(self):
        live = _live_box("lebron james", "2024-01-15", minutes=33.0, points=28.0)
        self.assertEqual(self._resolve("player_points", "LeBron James", None, live), 28.0)

    def test_live_null_minutes_is_dnp(self):
        live = _live_box("role player", "2024-01-15", minutes=float("nan"), points=0.0)
        self.assertIsNone(self._resolve("player_points", "Role Player", None, live))

    def test_wrong_date_no_match(self):
        wh = _wh_frame("lebron james", "2024-01-14", points=30.0)   # off by a day
        self.assertIsNone(self._resolve("player_points", "LeBron James", wh, None))

    def test_unmapped_prop(self):
        wh = _wh_frame("lebron james", "2024-01-15", points=30.0)
        self.assertIsNone(self._resolve("player_double_double", "LeBron James", wh, None))


class NbaDnpTests(unittest.TestCase):
    def test_dnp_row_true_false_none(self):
        with patch("recalibration._nba_player_game_frame",
                   return_value=_wh_frame("x", "2024-01-15", played=False)), \
             patch("nba_source.load_player_box", return_value=None):
            self.assertTrue(rc._nba_dnp_row("X", "2024-01-15"))
        with patch("recalibration._nba_player_game_frame",
                   return_value=_wh_frame("x", "2024-01-15", played=True)), \
             patch("nba_source.load_player_box", return_value=None):
            self.assertFalse(rc._nba_dnp_row("X", "2024-01-15"))
        with patch("recalibration._nba_player_game_frame", return_value=None), \
             patch("nba_source.load_player_box", return_value=None):
            self.assertIsNone(rc._nba_dnp_row("X", "2024-01-15"))

    def test_is_dnp_age_gated(self):
        old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with patch("recalibration._nba_dnp_row", return_value=True):
            self.assertTrue(rc._nba_is_dnp("player_points", "X", "2024-01-15", old))
            self.assertFalse(rc._nba_is_dnp("player_points", "X", "2024-01-15", recent))
        with patch("recalibration._nba_dnp_row", return_value=False):
            self.assertFalse(rc._nba_is_dnp("player_points", "X", "2024-01-15", old))


class NbaFinalScoreTests(unittest.TestCase):
    def _sched(self):
        return pd.DataFrame([
            {"status_type_completed": True, "home_score": 110, "away_score": 105,
             "home_display_name": "Los Angeles Lakers", "away_display_name": "Denver Nuggets",
             "game_date": "2024-01-10",
             "game_date_time": pd.Timestamp("2024-01-10 22:00:00-05:00")},
            {"status_type_completed": True, "home_score": 99, "away_score": 101,
             "home_display_name": "Los Angeles Lakers", "away_display_name": "Denver Nuggets",
             "game_date": "2024-03-02",
             "game_date_time": pd.Timestamp("2024-03-02 22:30:00-05:00")},
        ])

    def test_disambiguate_by_commence(self):
        # 2024-03-02 22:30 ET == 2024-03-03 03:30 UTC → the second game (99-101).
        with patch("nba_source.load_schedule", return_value=self._sched()):
            sc = gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets",
                                     "2024-03-03T03:30:00Z", "2024-03-02")
        self.assertEqual(sc, (99.0, 101.0))

    def test_disambiguate_by_game_date_no_commence(self):
        with patch("nba_source.load_schedule", return_value=self._sched()):
            sc = gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets", None, "2024-01-10")
        self.assertEqual(sc, (110.0, 105.0))

    def test_ambiguous_series_without_commence_or_date(self):
        with patch("nba_source.load_schedule", return_value=self._sched()):
            self.assertIsNone(
                gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets", None, None))

    def test_commence_far_from_any_game_stays_pending(self):
        with patch("nba_source.load_schedule", return_value=self._sched()):
            self.assertIsNone(
                gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets",
                                    "2024-06-01T00:00:00Z", "2024-06-01"))

    def test_final_score_routes_to_nba_spine(self):
        with patch("nba_source.load_schedule", return_value=self._sched()):
            sc = gr.final_score("basketball_nba", "2024-01-10", "Los Angeles Lakers",
                                "Denver Nuggets", "2024-01-11T03:00:00Z")
        self.assertEqual(sc, (110.0, 105.0))


class NbaTeamStatsTests(unittest.TestCase):
    def _sched(self):
        return pd.DataFrame([
            {"status_type_completed": True, "home_score": 110, "away_score": 105,
             "home_display_name": "Los Angeles Lakers", "away_display_name": "Denver Nuggets",
             "game_date": "2024-01-10"},
            {"status_type_completed": True, "home_score": 120, "away_score": 118,
             "home_display_name": "Boston Celtics", "away_display_name": "Los Angeles Lakers",
             "game_date": "2024-01-12"},
            {"status_type_completed": False, "home_score": None, "away_score": None,
             "home_display_name": "Los Angeles Lakers", "away_display_name": "Miami Heat",
             "game_date": "2024-01-20"},
        ])

    def test_team_form(self):
        with patch("nba_source.load_schedule", return_value=self._sched()), \
             patch("nba_source.current_end_year", return_value=2024):
            st = nts.team_stats("Los Angeles Lakers", recent_n=5, seasons=[2024])
        self.assertIsNotNone(st)
        self.assertEqual(st["recent"]["games"], 2)                # 2 completed, upcoming excluded
        self.assertEqual(st["recent"]["wins"], 1)                 # won @home 110-105, lost @BOS
        self.assertEqual(st["recent"]["losses"], 1)
        self.assertEqual(st["season"]["record"], "1-1")
        # queried team keyed to its own side; most-recent first
        self.assertEqual(st["recent_games"][0]["date"], "2024-01-12")
        self.assertEqual(st["recent_games"][0]["away_team"], "Los Angeles Lakers")

    def test_unknown_team_none(self):
        with patch("nba_source.load_schedule", return_value=self._sched()), \
             patch("nba_source.current_end_year", return_value=2024):
            self.assertIsNone(nts.team_stats("Nonexistent Team", seasons=[2024]))


if __name__ == "__main__":
    unittest.main()
