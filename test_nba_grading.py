"""NBA grading/analysis cutover (stats.nba.com, off ESPN):
  * recalibration._resolve_nba_actual — WAREHOUSE-ONLY (frame), DNP-safe, exact-date.
  * recalibration._nba_dnp_row / _nba_is_dnp — confirmed-DNP void signal.
  * game_results._nba_final_score / final_score — team scores off the warehouse spine.
  * nba_team_stats.team_stats — team form off the warehouse spine.
Loaders/warehouse seeded in SQLite; no network.
"""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import db_store
import game_results as gr
import nba_team_stats as nts
import nba_warehouse as nw
import recalibration as rc

_LAL, _DEN, _BOS, _MIA = 1610612747, 1610612743, 1610612738, 1610612748


def _wh_frame(pnorm, game_date, played=True, **stats):
    row = {"player_norm": pnorm, "game_date": game_date, "played": played}
    row.update(stats)
    return pd.DataFrame([row])


class ResolveNbaActualTests(unittest.TestCase):
    """WAREHOUSE-ONLY: the frame is the sole grading source (no live-box fallback — the
    stats.nba.com box lacks game_date). A miss returns None (-> ESPN burn-in fallback upstream)."""

    def _resolve(self, prop, name, wh, game_date="2024-01-15"):
        with patch("recalibration._nba_player_game_frame", return_value=wh):
            return rc._resolve_nba_actual(prop, name, game_date)

    def test_warehouse_used_when_present(self):
        wh = _wh_frame("lebron james", "2024-01-15", points=30.0)
        self.assertEqual(self._resolve("player_points", "LeBron James", wh), 30.0)

    def test_combo_prop_sums(self):
        wh = _wh_frame("lebron james", "2024-01-15", points=30.0, rebounds=8.0, assists=10.0)
        self.assertEqual(
            self._resolve("player_points_rebounds_assists", "LeBron James", wh), 48.0)

    def test_dnp_returns_none_not_phantom_zero(self):
        wh = _wh_frame("role player", "2024-01-15", played=False, points=0.0)
        self.assertIsNone(self._resolve("player_points", "Role Player", wh))

    def test_no_warehouse_row_returns_none(self):
        self.assertIsNone(self._resolve("player_points", "LeBron James", None))

    def test_wrong_date_no_match(self):
        wh = _wh_frame("lebron james", "2024-01-14", points=30.0)   # off by a day
        self.assertIsNone(self._resolve("player_points", "LeBron James", wh))

    def test_unmapped_prop(self):
        wh = _wh_frame("lebron james", "2024-01-15", points=30.0)
        self.assertIsNone(self._resolve("player_double_double", "LeBron James", wh))


class NbaDnpTests(unittest.TestCase):
    def test_dnp_row_true_false_none(self):
        with patch("recalibration._nba_player_game_frame",
                   return_value=_wh_frame("x", "2024-01-15", played=False)):
            self.assertTrue(rc._nba_dnp_row("X", "2024-01-15"))
        with patch("recalibration._nba_player_game_frame",
                   return_value=_wh_frame("x", "2024-01-15", played=True)):
            self.assertFalse(rc._nba_dnp_row("X", "2024-01-15"))
        with patch("recalibration._nba_player_game_frame", return_value=None):
            self.assertIsNone(rc._nba_dnp_row("X", "2024-01-15"))

    def test_is_dnp_age_gated(self):
        old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        with patch("recalibration._nba_dnp_row", return_value=True):
            self.assertTrue(rc._nba_is_dnp("player_points", "X", "2024-01-15", old))
            self.assertFalse(rc._nba_is_dnp("player_points", "X", "2024-01-15", recent))
        with patch("recalibration._nba_dnp_row", return_value=False):
            self.assertFalse(rc._nba_is_dnp("player_points", "X", "2024-01-15", old))


class _SpineTestBase(unittest.TestCase):
    """Seeds nba_team + nba_game (the spine) in SQLite for the warehouse-backed team graders."""

    def setUp(self):
        db_store.configure_engine("sqlite://")
        nw.create_all()
        with db_store.get_engine().begin() as c:
            for tid, name in ((_LAL, "Los Angeles Lakers"), (_DEN, "Denver Nuggets"),
                              (_BOS, "Boston Celtics"), (_MIA, "Miami Heat")):
                c.execute(nw.nba_team.insert().values(
                    team_id=tid, name=name, name_norm=db_store.normalize_name(name)))
            for gm in self.GAMES:
                c.execute(nw.nba_game.insert().values(**gm))

    def tearDown(self):
        db_store.configure_engine(None)


class NbaFinalScoreTests(_SpineTestBase):
    GAMES = [
        {"game_id": "0022300001", "season": 2024, "season_type": "regular-season",
         "game_date": "2024-01-10", "home_team_id": _LAL, "away_team_id": _DEN,
         "home_score": 110.0, "away_score": 105.0, "completed": True},
        {"game_id": "0022300500", "season": 2024, "season_type": "regular-season",
         "game_date": "2024-03-02", "home_team_id": _LAL, "away_team_id": _DEN,
         "home_score": 99.0, "away_score": 101.0, "completed": True},   # series: same matchup
    ]

    def test_match_by_game_date(self):
        self.assertEqual(
            gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets", None, "2024-01-10"),
            (110.0, 105.0))
        self.assertEqual(
            gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets", None, "2024-03-02"),
            (99.0, 101.0))

    def test_no_game_on_date_none(self):
        self.assertIsNone(
            gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets", None, "2024-06-01"))

    def test_final_score_routes_to_spine(self):
        self.assertEqual(
            gr.final_score("basketball_nba", "2024-03-02", "Los Angeles Lakers",
                           "Denver Nuggets", "2024-03-03T03:30:00Z"),
            (99.0, 101.0))

    def test_date_slippage_plus_minus_one(self):
        # bet game_date off by a day (UTC/ET) -> ±1 still resolves (no same-matchup back-to-back)
        self.assertEqual(
            gr._nba_final_score("Los Angeles Lakers", "Denver Nuggets", None, "2024-01-11"),
            (110.0, 105.0))


class NbaTeamStatsTests(_SpineTestBase):
    GAMES = [
        {"game_id": "0022300001", "season": 2024, "game_date": "2024-01-10",
         "home_team_id": _LAL, "away_team_id": _DEN, "home_score": 110.0,
         "away_score": 105.0, "completed": True},                       # LAL home win
        {"game_id": "0022300050", "season": 2024, "game_date": "2024-01-12",
         "home_team_id": _BOS, "away_team_id": _LAL, "home_score": 120.0,
         "away_score": 118.0, "completed": True},                       # LAL away loss
        {"game_id": "0022300090", "season": 2024, "game_date": "2024-01-20",
         "home_team_id": _LAL, "away_team_id": _MIA, "home_score": None,
         "away_score": None, "completed": False},                       # upcoming (excluded)
    ]

    def test_team_form(self):
        st = nts.team_stats("Los Angeles Lakers", recent_n=5, seasons=[2024])
        self.assertIsNotNone(st)
        self.assertEqual(st["recent"]["games"], 2)                # 2 completed, upcoming excluded
        self.assertEqual(st["recent"]["wins"], 1)
        self.assertEqual(st["recent"]["losses"], 1)
        self.assertEqual(st["season"]["record"], "1-1")
        self.assertEqual(st["recent_games"][0]["date"], "2024-01-12")   # most-recent first
        self.assertEqual(st["recent_games"][0]["away_team"], "Los Angeles Lakers")

    def test_unknown_team_none(self):
        self.assertIsNone(nts.team_stats("Nonexistent Team", seasons=[2024]))


if __name__ == "__main__":
    unittest.main()
