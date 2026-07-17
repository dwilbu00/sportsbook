"""Tests for the generalized low-participation reliability filter (P2 batch:
DNP / zero-participation games polluting non-NBA projections).

Before this change the filter derived its threshold only from `MIN`, which
exists for NBA/NHL but not MLB/NFL — so DNP / near-zero-participation MLB games
(rendered as 0.0 stats) stayed in the sample and dragged batter projections
down. The filter now uses a sport-appropriate participation metric (plate
appearances / at-bats for MLB) while leaving NBA behavior and NFL (no reliable
signal) unchanged.
"""

import unittest

from prop_filter import filter_player_gamelog


def _dates(eligible):
    return {g["game_date"] for g in eligible}


class MlbParticipationTests(unittest.TestCase):
    def test_only_zero_pa_games_excluded_using_raw_pa_keys(self):
        # Backtest passes raw ESPN gamelog dicts (PA/AB present). MLB uses a
        # floor-only threshold (fraction 0.0): ONLY true DNPs (0 PA) are dropped;
        # a 1-PA cameo is a real game and is kept (backtest-validated choice).
        games = [
            {"game_date": "2024-07-01", "PA": 4, "H": 2},
            {"game_date": "2024-07-02", "PA": 0, "H": 0},   # DNP → dropped
            {"game_date": "2024-07-03", "PA": 5, "H": 1},
            {"game_date": "2024-07-04", "PA": 4, "H": 3},
            {"game_date": "2024-07-05", "PA": 1, "H": 0},   # 1-PA cameo → kept
            {"game_date": "2024-07-06", "PA": 4, "H": 2},
        ]
        result = filter_player_gamelog(
            games, None, "baseball_mlb", min_streak=1)
        self.assertEqual(result["n_excluded_low_min"], 1)
        self.assertEqual(len(result["eligible_games"]), 5)
        self.assertNotIn("2024-07-02", _dates(result["eligible_games"]))
        self.assertIn("2024-07-05", _dates(result["eligible_games"]))

    def test_low_pa_games_excluded_using_runtime_underscore_keys(self):
        # analysis.py (runtime) threads _pa/_ab into the filter dicts.
        games = [
            {"game_date": "2024-07-01", "_pa": 4},
            {"game_date": "2024-07-02", "_pa": 0},          # DNP
            {"game_date": "2024-07-03", "_pa": 5},
            {"game_date": "2024-07-04", "_pa": 4},
        ]
        result = filter_player_gamelog(
            games, None, "baseball_mlb", min_streak=1)
        self.assertEqual(result["n_excluded_low_min"], 1)
        self.assertNotIn("2024-07-02", _dates(result["eligible_games"]))

    def test_at_bats_used_when_plate_appearances_absent(self):
        games = [
            {"game_date": "2024-07-01", "AB": 4},
            {"game_date": "2024-07-02", "AB": 0},           # DNP
            {"game_date": "2024-07-03", "AB": 4},
            {"game_date": "2024-07-04", "AB": 4},
        ]
        result = filter_player_gamelog(
            games, None, "baseball_mlb", min_streak=1)
        self.assertEqual(result["n_excluded_low_min"], 1)
        self.assertNotIn("2024-07-02", _dates(result["eligible_games"]))

    def test_pitcher_games_without_pa_ab_are_not_filtered(self):
        # MLB pitcher gamelogs (splits) carry no PA/AB → participation None →
        # the low-participation filter stays disabled (status quo).
        games = [
            {"game_date": "2024-07-01", "K": 7, "IP": 6.0},
            {"game_date": "2024-07-07", "K": 0, "IP": 0.0},
            {"game_date": "2024-07-13", "K": 9, "IP": 7.0},
        ]
        result = filter_player_gamelog(
            games, None, "baseball_mlb", min_streak=1)
        self.assertEqual(result["n_excluded_low_min"], 0)
        self.assertEqual(len(result["eligible_games"]), 3)


class NbaBehaviorPreservedTests(unittest.TestCase):
    def test_min_based_low_participation_still_applies(self):
        games = [
            {"game_date": "2024-04-01", "MIN": 30},
            {"game_date": "2024-04-02", "MIN": 2},          # low minutes
            {"game_date": "2024-04-03", "MIN": 30},
            {"game_date": "2024-04-04", "MIN": 30},
            {"game_date": "2024-04-05", "MIN": 30},
        ]
        result = filter_player_gamelog(
            games, None, "basketball_nba", min_streak=1)
        # median MIN = 30 → threshold max(10, 15) = 15; MIN=2 dropped.
        self.assertEqual(result["n_excluded_low_min"], 1)
        self.assertEqual(len(result["eligible_games"]), 4)
        self.assertNotIn("2024-04-02", _dates(result["eligible_games"]))
        self.assertFalse(result["skip_prediction"])


class NflNoParticipationSignalTests(unittest.TestCase):
    def test_nfl_games_are_not_participation_filtered(self):
        # NFL gamelogs lack a reliable per-game participation metric, so even a
        # zero-stat game is retained (unchanged behavior).
        games = [
            {"game_date": "2024-09-01", "YDS": 80},
            {"game_date": "2024-09-08", "YDS": 0},
            {"game_date": "2024-09-15", "YDS": 90},
        ]
        result = filter_player_gamelog(
            games, None, "americanfootball_nfl", min_streak=1)
        self.assertEqual(result["n_excluded_low_min"], 0)
        self.assertEqual(len(result["eligible_games"]), 3)


if __name__ == "__main__":
    unittest.main()
