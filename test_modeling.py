"""Focused regression tests for sportsbook model correctness boundaries."""

import unittest

import analysis
import mlb_starters
from backtest_props import _rolling_splits
from odds_client import parse_player_props
from prop_filter import filter_player_gamelog
from recalibration import fit_platt_chronological, summarize_prediction_rows
from savant_history import _at_bat_xba


class StarterAdjustmentTests(unittest.TestCase):
    def tearDown(self):
        analysis._STARTER_ADJ_CACHE.clear()
        analysis._LINEUP_ADJ_CACHE.clear()

    def test_missing_calibration_fails_closed(self):
        analysis._STARTER_ADJ_CACHE["missing"] = {}
        self.assertEqual(
            analysis._starter_adjustment(
                "missing", "props", "batter_strikeouts"),
            0.0,
        )

    def test_prop_weights_are_independent(self):
        analysis._STARTER_ADJ_CACHE["mapped"] = {
            "enabled": True,
            "props": {"batter_hits": 0.0, "batter_strikeouts": 0.5},
        }
        self.assertEqual(
            analysis._starter_adjustment("mapped", "props", "batter_hits"),
            0.0,
        )
        self.assertEqual(
            analysis._starter_adjustment(
                "mapped", "props", "batter_strikeouts"),
            0.5,
        )
        self.assertEqual(
            analysis._starter_adjustment("mapped", "props", "pitcher_outs"),
            0.0,
        )

    def test_log5_is_league_neutral_and_responds_to_pitcher(self):
        league = analysis._MLB_LEAGUE["k_pct"]
        self.assertAlmostEqual(
            analysis._log5_rate(league, league, league), league)
        self.assertGreater(
            analysis._log5_rate(0.25, 0.30, league),
            analysis._log5_rate(0.25, 0.20, league),
        )

    def test_batter_k_projection_weights_starter_exposure(self):
        features = {
            "away": {
                "starter": {"k_pct": 0.30, "bf": 100, "avg_ip": 6.0},
                "bullpen": {"k_pct": 0.20},
            }
        }
        multiplier = analysis._mlb_prop_matchup_mult(
            "batter_strikeouts",
            upcoming_is_home=True,
            matchup_features=features,
            weight=1.0,
            player_context={"base_projection": 1.0, "expected_exposure": 4.0},
        )
        self.assertGreater(multiplier, 1.0)
        self.assertLess(multiplier, 1.4)
        features["away"]["bullpen"]["k_pct"] = 0.40
        self.assertEqual(
            multiplier,
            analysis._mlb_prop_matchup_mult(
                "batter_strikeouts", True, features, 1.0,
                player_context={
                    "base_projection": 1.0, "expected_exposure": 4.0,
                },
            ),
        )

    def test_statcast_xba_uses_at_bat_denominator(self):
        self.assertEqual(_at_bat_xba({"events": "strikeout"}), 0.0)
        self.assertIsNone(_at_bat_xba({
            "events": "walk", "estimated_ba_using_speedangle": "0.900",
        }))
        self.assertEqual(_at_bat_xba({
            "events": "single", "estimated_ba_using_speedangle": "0.700",
        }), 0.7)

    def test_batter_hit_projection_uses_xba(self):
        features = {
            "away": {
                "starter": {"xba": 0.290, "avg_ip": 6.0},
                "bullpen": {"avg_allowed": 0.260},
            }
        }
        multiplier = analysis._mlb_prop_matchup_mult(
            "batter_hits",
            upcoming_is_home=True,
            matchup_features=features,
            weight=1.0,
            player_context={"base_projection": 1.0, "expected_exposure": 4.0},
        )
        self.assertGreater(multiplier, 1.0)
        self.assertLess(multiplier, 1.4)

    def test_lineup_order_adjusts_hits_but_not_strikeouts(self):
        analysis._LINEUP_ADJ_CACHE["baseball_mlb"] = {
            "enabled": True,
            "props": {"batter_hits": 0.75, "batter_strikeouts": 0.0},
            "slot_expected_exposure": {
                "batter_hits": {"1": 4.1, "9": 3.4},
            },
        }
        context = {"expected_exposure": 3.6, "batting_order": 1}
        self.assertGreater(
            analysis._mlb_lineup_exposure_mult("batter_hits", context),
            1.0,
        )
        self.assertEqual(
            analysis._mlb_lineup_exposure_mult(
                "batter_strikeouts", context),
            1.0,
        )
        self.assertEqual(
            analysis._mlb_lineup_exposure_mult(
                "batter_hits", {"expected_exposure": 3.6}),
            1.0,
        )

    def test_only_complete_announced_lineups_return_player_context(self):
        game = {
            "lineups": {
                "homePlayers": [
                    {"id": slot, "fullName": (
                        "José Ramírez" if slot == 1 else f"Home Player {slot}")}
                    for slot in range(1, 10)
                ],
                "awayPlayers": [
                    {"id": 100 + slot, "fullName": f"Away Player {slot}"}
                    for slot in range(1, 9)
                ],
            },
        }
        players = mlb_starters._lineup_players(game)
        lineup = {
            "home_confirmed": True,
            "away_confirmed": False,
            "players": players,
        }
        self.assertEqual(
            mlb_starters.lineup_player_context(
                lineup, "Jose Ramirez")["batting_order"],
            1,
        )
        self.assertIsNone(
            mlb_starters.lineup_player_context(lineup, "Away Player 1"))


class AsOfReliabilityTests(unittest.TestCase):
    def test_future_games_do_not_complete_an_earlier_streak(self):
        games = [
            {"game_date": f"2024-04-{day:02d}", "MIN": 30}
            for day in range(1, 9)
        ]
        schedule = [
            {"date": f"2024-04-{day:02d}"}
            for day in range(1, 12)
        ]
        early = filter_player_gamelog(
            games, schedule, "basketball_nba",
            min_streak=5, as_of_date="2024-04-05",
        )
        later = filter_player_gamelog(
            games, schedule, "basketball_nba",
            min_streak=5, as_of_date="2024-04-07",
        )
        self.assertTrue(early["skip_prediction"])
        self.assertEqual(early["current_streak"], 4)
        self.assertFalse(later["skip_prediction"])
        self.assertEqual(len(later["eligible_games"]), 6)

    def test_future_return_does_not_retroactively_mark_pre_layoff_game(self):
        games = [
            {"game_date": "2024-04-01", "MIN": 30},
            {"game_date": "2024-04-02", "MIN": 30},
            {"game_date": "2024-04-03", "MIN": 30},
            {"game_date": "2024-04-10", "MIN": 30},
        ]
        schedule = [
            {"date": f"2024-04-{day:02d}"}
            for day in range(1, 12)
        ]
        result = filter_player_gamelog(
            games, schedule, "basketball_nba",
            min_streak=1, as_of_date="2024-04-05",
        )
        eligible_dates = {
            game["game_date"] for game in result["eligible_games"]
        }
        self.assertIn("2024-04-03", eligible_dates)


class MarketParsingTests(unittest.TestCase):
    def test_props_use_devigged_consensus_and_best_side_prices(self):
        game = {
            "id": "game",
            "home_team": "Home",
            "away_team": "Away",
            "bookmakers": [
                {
                    "title": "Book 1",
                    "markets": [{
                        "key": "batter_hits",
                        "outcomes": [
                            {"description": "Player", "name": "Over",
                             "price": -120, "point": 1.5},
                            {"description": "Player", "name": "Under",
                             "price": 100, "point": 1.5},
                        ],
                    }],
                },
                {
                    "title": "Book 2",
                    "markets": [{
                        "key": "batter_hits",
                        "outcomes": [
                            {"description": "Player", "name": "Over",
                             "price": 105, "point": 1.5},
                            {"description": "Player", "name": "Under",
                             "price": -125, "point": 1.5},
                        ],
                    }],
                },
                {
                    "title": "Bad Cross-Line Book",
                    "markets": [{
                        "key": "batter_hits",
                        "outcomes": [
                            {"description": "Player", "name": "Over",
                             "price": 150, "point": 1.5},
                            {"description": "Player", "name": "Under",
                             "price": 500, "point": 2.5},
                        ],
                    }],
                },
                {
                    "title": "Best One-Sided Over",
                    "markets": [{
                        "key": "batter_hits",
                        "outcomes": [
                            {"description": "Player", "name": "Over",
                             "price": 200, "point": 1.5},
                        ],
                    }],
                },
            ],
        }
        info = parse_player_props(game)["props"]["batter_hits"]["Player"]
        self.assertEqual(info["over_price"], 200)
        self.assertEqual(info["under_price"], 100)
        self.assertEqual(info["books_sampled"], 2)
        self.assertEqual(info["over_prices_sampled"], 4)
        self.assertAlmostEqual(
            info["over_implied"] + info["under_implied"], 1.0)


class RecalibrationTests(unittest.TestCase):
    def test_platt_requires_and_passes_later_holdout(self):
        records = []
        for block in range(10):
            date = f"2024-{block + 1:02d}"
            records.extend([
                (f"{date}-01", 0.8, 1),
                (f"{date}-02", 0.8, 1),
                (f"{date}-03", 0.8, 1),
                (f"{date}-04", 0.8, 0),
                (f"{date}-05", 0.8, 0),
                (f"{date}-06", 0.2, 1),
                (f"{date}-07", 0.2, 1),
                (f"{date}-08", 0.2, 0),
                (f"{date}-09", 0.2, 0),
                (f"{date}-10", 0.2, 0),
            ])
        result = fit_platt_chronological(records)
        self.assertIsNotNone(result)
        self.assertTrue(result["validated"])
        self.assertEqual(result["n_validation_folds"], 2)
        self.assertLess(
            result["holdout_calibrated_brier"],
            result["holdout_raw_brier"],
        )
        self.assertLess(
            result["holdout_calibrated_log_loss"],
            result["holdout_raw_log_loss"],
        )

    def test_matchup_rolling_folds_keep_dates_intact(self):
        observations = []
        for month in range(1, 11):
            date = f"2024-{month:02d}-01"
            observations.extend(
                [(1.0, 1.0, True, {}, date)] * 60
            )
        folds = _rolling_splits(observations)
        self.assertEqual(len(folds), 2)
        for train, holdout in folds:
            self.assertTrue({row[4] for row in train}.isdisjoint(
                {row[4] for row in holdout}))
            self.assertLess(max(row[4] for row in train),
                            min(row[4] for row in holdout))

    def test_forward_summary_deduplicates_and_scores_direction(self):
        rows = [
            {
                "ts": "2024-04-01T10:00:00Z",
                "sport_key": "baseball_mlb",
                "prop_key": "batter_hits",
                "player": "Player One",
                "game_date": "2024-04-01",
                "line": 1.5,
                "raw_prob": 0.7,
                "direction": "OVER",
                "resolved": False,
                "outcome": None,
            },
            {
                "ts": "2024-04-01T10:05:00Z",
                "sport_key": "baseball_mlb",
                "prop_key": "batter_hits",
                "player": "Player One",
                "game_date": "2024-04-01",
                "line": 1.5,
                "raw_prob": 0.8,
                "final_prob": 0.9,
                "direction": "OVER",
                "price": 100,
                "resolved": True,
                "outcome": 1,
            },
            {
                "ts": "2024-04-02T10:00:00Z",
                "sport_key": "baseball_mlb",
                "prop_key": "batter_hits",
                "player": "Player Two",
                "game_date": "2024-04-02",
                "line": 0.5,
                "raw_prob": 0.3,
                "direction": "UNDER",
                "price": -110,
                "resolved": True,
                "outcome": 0,
            },
        ]
        summary = summarize_prediction_rows(rows)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["resolved"], 2)
        self.assertEqual(summary["direction_hit_rate"], 1.0)
        self.assertAlmostEqual(summary["probability_brier"], 0.05)
        self.assertEqual(summary["priced_resolved"], 2)
        self.assertAlmostEqual(summary["realized_roi"], (1 + 10 / 11) / 2)


class ParlayCorrelationTests(unittest.TestCase):
    def test_moneyline_prop_synergy_requires_same_team(self):
        moneyline = {
            "game_key": "Away @ Home", "bet_type": "moneyline",
            "team": "Home",
        }
        same_team_hit = {
            "game_key": "Away @ Home", "bet_type": "player_prop_over",
            "team": "Home", "prop_key": "batter_hits",
        }
        opponent_hit = dict(same_team_hit, team="Away")
        self.assertEqual(
            analysis._pair_correlation(
                moneyline, same_team_hit, "baseball_mlb"),
            0.25,
        )
        self.assertEqual(
            analysis._pair_correlation(
                moneyline, opponent_hit, "baseball_mlb"),
            0.05,
        )
        self.assertEqual(
            analysis._correlation_penalty(
                moneyline, same_team_hit, "baseball_mlb"),
            5.0 * analysis._pair_correlation(
                moneyline, same_team_hit, "baseball_mlb"),
        )


if __name__ == "__main__":
    unittest.main()
