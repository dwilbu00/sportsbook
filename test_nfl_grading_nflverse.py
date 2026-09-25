"""NFL prop grading via the nflverse player-week feed (phase 1 of the cutover).

Guards recalibration._resolve_nfl_actual + nfl_schedule.season_week_for_date + the
resolve_one_prop wiring (nflverse-first, ESPN fallback). See
notes/NFL_GRADING_NFLVERSE_CUTOVER_2026-09-25.md.

Run: PYTHONIOENCODING=utf-8 python -m unittest test_nfl_grading_nflverse -v
"""
import unittest
from unittest.mock import patch

import pandas as pd

import recalibration as rc
import nfl_schedule
import nfl_props_scan as scan


def _df(rows):
    return pd.DataFrame(rows)


class ResolveNflActualTests(unittest.TestCase):
    def _run(self, prop_key, player, df, sw=("2026", 3), game_date="2026-09-21"):
        with patch("nfl_schedule.season_week_for_date", return_value=sw), \
                patch("nfl_opportunity_serving._load", return_value=df):
            return rc._resolve_nfl_actual(prop_key, player, game_date)

    def test_yardage_reads_the_mapped_column(self):
        df = _df([{"player_norm": scan._norm("Jared Goff"), "week": 3,
                   "passing_yards": 327.0, "rushing_yards": 5.0}])
        self.assertEqual(self._run("player_pass_yds", "Jared Goff", df), 327.0)

    def test_receptions_column(self):
        df = _df([{"player_norm": scan._norm("Malik Nabers"), "week": 3,
                   "receptions": 7.0}])
        self.assertEqual(self._run("player_receptions", "Malik Nabers", df), 7.0)

    def test_anytime_td_sums_rush_rec_and_return(self):
        # DK/FD anytime-TD pays rushing + receiving + return TDs -> sum all three.
        df = _df([{"player_norm": scan._norm("Kyren Williams"), "week": 3,
                   "rushing_tds": 1.0, "receiving_tds": 1.0, "special_teams_tds": 1.0}])
        self.assertEqual(self._run("player_anytime_td", "Kyren Williams", df), 3.0)

    def test_zero_is_a_real_actual_not_pending(self):
        # a player who PLAYED but caught 0 has a row -> actual 0.0, NOT None(pending).
        df = _df([{"player_norm": scan._norm("Player One"), "week": 3, "receptions": 0.0}])
        self.assertEqual(self._run("player_receptions", "Player One", df), 0.0)

    def test_unmapped_prop_is_none(self):
        df = _df([{"player_norm": scan._norm("X"), "week": 3, "passing_yards": 1.0}])
        self.assertIsNone(self._run("player_field_goals_made", "X", df))

    def test_dnp_no_row_is_none(self):
        df = _df([{"player_norm": scan._norm("Someone Else"), "week": 3, "receptions": 5.0}])
        self.assertIsNone(self._run("player_receptions", "Ghost Player", df))

    def test_wrong_week_is_none(self):
        # sw resolves to week 3 but the player's only row is week 2 -> pending.
        df = _df([{"player_norm": scan._norm("Player One"), "week": 2, "receptions": 5.0}])
        self.assertIsNone(self._run("player_receptions", "Player One", df))

    def test_unresolvable_week_is_none(self):
        df = _df([{"player_norm": scan._norm("Player One"), "week": 3, "receptions": 5.0}])
        self.assertIsNone(self._run("player_receptions", "Player One", df, sw=None))

    def test_none_feed_is_none(self):
        with patch("nfl_schedule.season_week_for_date", return_value=("2026", 3)), \
                patch("nfl_opportunity_serving._load", return_value=None):
            self.assertIsNone(rc._resolve_nfl_actual("player_receptions", "X", "2026-09-21"))


class SeasonWeekForDateTests(unittest.TestCase):
    def _patch_games(self, recs):
        return patch("nfl_schedule.load_games", side_effect=lambda seasons=None, **k: recs)

    def test_exact_gameday(self):
        recs = [{"season": "2026", "week": "3", "gameday": "2026-09-21"},
                {"season": "2026", "week": "2", "gameday": "2026-09-14"}]
        with self._patch_games(recs):
            self.assertEqual(nfl_schedule.season_week_for_date("2026-09-21"), ("2026", 3))

    def test_plus_minus_one_day_slippage(self):
        recs = [{"season": "2026", "week": "3", "gameday": "2026-09-21"}]
        with self._patch_games(recs):
            self.assertEqual(nfl_schedule.season_week_for_date("2026-09-22"), ("2026", 3))

    def test_no_game_near_date_is_none(self):
        recs = [{"season": "2026", "week": "3", "gameday": "2026-09-21"}]
        with self._patch_games(recs):
            self.assertIsNone(nfl_schedule.season_week_for_date("2026-10-15"))

    def test_january_belongs_to_prior_season(self):
        # a Jan 2027 playoff game is season 2026, week 19.
        recs = [{"season": "2026", "week": "19", "gameday": "2027-01-10"}]
        with self._patch_games(recs):
            self.assertEqual(nfl_schedule.season_week_for_date("2027-01-10"), ("2026", 19))

    def test_bad_date_is_none(self):
        with self._patch_games([]):
            self.assertIsNone(nfl_schedule.season_week_for_date("not-a-date"))


class ResolveOnePropWiringTests(unittest.TestCase):
    """resolve_one_prop wiring: gated OFF by default (ESPN-primary); when the gate is
    ON, nflverse is preferred with ESPN as the fallback on a miss."""

    def _resolve(self):
        return rc.resolve_one_prop(
            "americanfootball_nfl", "Player", "player_receptions", 5.5,
            "2026-09-21", "2026-09-21T17:00:00Z")

    def test_gate_off_by_default_skips_nflverse_uses_espn(self):
        # Phase 1 ships dormant: default gate OFF -> nflverse never consulted.
        self.assertFalse(rc._nfl_grade_from_nflverse())     # documents the default
        with patch("recalibration._resolve_mlb_actual", return_value=None), \
                patch("recalibration._resolve_nfl_actual", return_value=42.0) as nfl, \
                patch("recalibration._load_player_gamelog", return_value=(None, {})) as espn:
            out = self._resolve()
        self.assertIsNone(out)            # ESPN path (empty) -> pending
        nfl.assert_not_called()           # gate off -> nflverse skipped entirely
        espn.assert_called_once()

    def test_gate_on_prefers_nflverse_and_skips_espn_on_hit(self):
        with patch("recalibration._nfl_grade_from_nflverse", return_value=True), \
                patch("recalibration._resolve_mlb_actual", return_value=None), \
                patch("recalibration._resolve_nfl_actual", return_value=42.0) as nfl, \
                patch("recalibration._load_player_gamelog") as espn:
            out = self._resolve()
        self.assertEqual(out, 42.0)
        nfl.assert_called_once()
        espn.assert_not_called()          # nflverse hit -> never touches the ESPN path

    def test_gate_on_falls_back_to_espn_on_nflverse_miss(self):
        with patch("recalibration._nfl_grade_from_nflverse", return_value=True), \
                patch("recalibration._resolve_mlb_actual", return_value=None), \
                patch("recalibration._resolve_nfl_actual", return_value=None) as nfl, \
                patch("recalibration._load_player_gamelog", return_value=(None, {})) as espn:
            out = self._resolve()
        self.assertIsNone(out)            # ESPN also empty -> stays pending
        nfl.assert_called_once()
        espn.assert_called_once()         # fell through to the ESPN fallback

    def test_env_var_overrides_default(self):
        import os
        with patch.dict(os.environ, {"ODI_NFL_GRADE_NFLVERSE": "1"}):
            self.assertTrue(rc._nfl_grade_from_nflverse())
        with patch.dict(os.environ, {"ODI_NFL_GRADE_NFLVERSE": "0"}):
            self.assertFalse(rc._nfl_grade_from_nflverse())


if __name__ == "__main__":
    unittest.main(verbosity=2)
