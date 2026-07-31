"""Focused regression tests for sportsbook model correctness boundaries."""

import math
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import requests

import analysis
import backtest_market_consensus
import backtest_starters
import espn_client
import mlb_starters
import odds_client
import park_factors
import props
import recalibration
import weather_factors
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


class PlayerStartStatusTests(unittest.TestCase):
    """§2.5A tri-state pre-game availability gate (mlb_starters.player_start_status).

    Only a confident "out" acts; everything uncertain fails open to "unknown".
    Batter props gate on the confirmed lineup, pitcher props on the announced
    probable. ``season=None`` exercises the pure name-based logic; a positive
    season enables the id-confirmation arms (find_player_id patched)."""

    def _lineup(self, home_names, away_names, home_conf=True, away_conf=True):
        game = {"lineups": {
            "homePlayers": [{"id": i, "fullName": n}
                            for i, n in enumerate(home_names, 1)],
            "awayPlayers": [{"id": 100 + i, "fullName": n}
                            for i, n in enumerate(away_names, 1)],
        }}
        return {"home_confirmed": home_conf, "away_confirmed": away_conf,
                "players": mlb_starters._lineup_players(game)}

    def _probables(self, home_name=None, away_name=None,
                   home_id=111, away_id=222):
        out = {}
        if home_name:
            out[mlb_starters._norm("Guardians")] = {
                "pitcher_id": home_id, "name": home_name, "team_id": 1}
        if away_name:
            out[mlb_starters._norm("Tigers")] = {
                "pitcher_id": away_id, "name": away_name, "team_id": 2}
        return out

    # ---- batter arm ----
    def test_batter_in_confirmed_side(self):
        lineup = self._lineup(
            ["José Ramírez"] + [f"H{i}" for i in range(2, 10)],
            [f"A{i}" for i in range(1, 10)])
        self.assertEqual(mlb_starters.player_start_status(
            "batter_hits", "Jose Ramirez", "Guardians", "Tigers",
            lineup, {}, season=None), "in")

    def test_batter_present_but_side_unconfirmed_is_unknown(self):
        lineup = self._lineup(
            ["José Ramírez"] + [f"H{i}" for i in range(2, 10)],
            [f"A{i}" for i in range(1, 10)], home_conf=False)
        self.assertEqual(mlb_starters.player_start_status(
            "batter_hits", "Jose Ramirez", "Guardians", "Tigers",
            lineup, {}, season=None), "unknown")

    def test_batter_absent_both_confirmed_is_out(self):
        lineup = self._lineup([f"H{i}" for i in range(1, 10)],
                              [f"A{i}" for i in range(1, 10)])
        self.assertEqual(mlb_starters.player_start_status(
            "batter_hits", "Benched Regular", "Guardians", "Tigers",
            lineup, {}, season=None), "out")

    def test_batter_absent_one_side_unconfirmed_is_unknown(self):
        lineup = self._lineup([f"H{i}" for i in range(1, 10)],
                              [f"A{i}" for i in range(1, 9)], away_conf=False)
        self.assertEqual(mlb_starters.player_start_status(
            "batter_hits", "Benched Regular", "Guardians", "Tigers",
            lineup, {}, season=None), "unknown")

    def test_batter_absent_by_name_but_id_matches_is_in(self):
        # Odds-feed spelling differs, but the id resolves to a posted player
        # (home slot 1, id=1) -> NOT out. Guards a false out on spelling drift.
        lineup = self._lineup(
            ["José Ramírez"] + [f"H{i}" for i in range(2, 10)],
            [f"A{i}" for i in range(1, 10)])
        with patch.object(mlb_starters, "find_player_id",
                          return_value=(1, False)):
            self.assertEqual(mlb_starters.player_start_status(
                "batter_hits", "J Ram odds spelling", "Guardians", "Tigers",
                lineup, {}, season=2025), "in")

    # ---- pitcher arm ----
    def test_pitcher_matches_probable_is_in(self):
        probs = self._probables("Shane Bieber", "Tarik Skubal")
        self.assertEqual(mlb_starters.player_start_status(
            "pitcher_strikeouts", "Shane Bieber", "Guardians", "Tigers",
            {}, probs, season=None), "in")

    def test_pitcher_not_announced_both_sides_is_out(self):
        probs = self._probables("Shane Bieber", "Tarik Skubal")
        with patch.object(mlb_starters, "find_player_id",
                          return_value=(999, True)):
            self.assertEqual(mlb_starters.player_start_status(
                "pitcher_strikeouts", "Some Reliever", "Guardians", "Tigers",
                {}, probs, season=2025), "out")

    def test_pitcher_one_side_tbd_is_unknown(self):
        probs = self._probables("Shane Bieber", None)  # away starter TBD
        with patch.object(mlb_starters, "find_player_id",
                          return_value=(999, True)):
            self.assertEqual(mlb_starters.player_start_status(
                "pitcher_strikeouts", "Some Reliever", "Guardians", "Tigers",
                {}, probs, season=2025), "unknown")

    def test_pitcher_id_match_when_name_differs_is_in(self):
        probs = self._probables("Shane Bieber", "Tarik Skubal")
        with patch.object(mlb_starters, "find_player_id",
                          return_value=(111, True)):
            self.assertEqual(mlb_starters.player_start_status(
                "pitcher_strikeouts", "S Bieber odds", "Guardians", "Tigers",
                {}, probs, season=2025), "in")

    def test_pitcher_no_probables_is_unknown(self):
        self.assertEqual(mlb_starters.player_start_status(
            "pitcher_strikeouts", "Shane Bieber", "Guardians", "Tigers",
            {}, {}, season=None), "unknown")

    def test_pitcher_unresolvable_name_not_ruled_out(self):
        # Both probables announced but this pitcher's id can't be resolved
        # (find_player_id -> None) -> stay "unknown", never a false out.
        probs = self._probables("Shane Bieber", "Tarik Skubal")
        with patch.object(mlb_starters, "find_player_id", return_value=None):
            self.assertEqual(mlb_starters.player_start_status(
                "pitcher_strikeouts", "Mystery Arm", "Guardians", "Tigers",
                {}, probs, season=2025), "unknown")

    # ---- fail-open ----
    def test_empty_inputs_are_unknown(self):
        self.assertEqual(mlb_starters.player_start_status(
            "batter_hits", "Nobody", "Guardians", "Tigers",
            {}, {}, season=None), "unknown")

    def test_missing_player_name_is_unknown(self):
        self.assertEqual(mlb_starters.player_start_status(
            "batter_hits", "", "Guardians", "Tigers",
            {}, {}, season=None), "unknown")


class ExpectedRunsTests(unittest.TestCase):
    def setUp(self):
        analysis._EXPECTED_RUNS_CACHE.clear()

    def tearDown(self):
        analysis._EXPECTED_RUNS_CACHE.clear()

    @staticmethod
    def _calibration():
        return {
            "enabled": True,
            "live_markets": {
                "moneyline": False, "spreads": True, "totals": False,
            },
            "final_2025_validation": {
                "model": {
                    "offense_weight": 1.25,
                    "pitching_weight": 0.75,
                    "home_base_runs": 4.423,
                    "away_base_runs": 4.429,
                },
                "ensemble_challenger_share": {
                    "moneyline": 0.75,
                    "home_minus_1_5": 0.70,
                    "margin": 0.90,
                },
            },
        }

    @staticmethod
    def _team_stats():
        games = [
            {"home_team": "Home", "away_team": "Away",
             "home_score": 6, "away_score": 3},
            {"home_team": "Away", "away_team": "Home",
             "home_score": 2, "away_score": 4},
            {"home_team": "Home", "away_team": "Away",
             "home_score": 1, "away_score": 5},
            {"home_team": "Away", "away_team": "Home",
             "home_score": 3, "away_score": 2},
        ]
        base = {
            "season": {"win_pct": 0.5},
            "recent": {"win_pct": 0.5},
            "recent_games": games,
        }
        return dict(base), dict(base)

    @staticmethod
    def _game_odds():
        return {
            "home_team": "Home",
            "away_team": "Away",
            "spreads": {
                "Home": [{"spread": -1.5, "price": -110}],
                "Away": [{"spread": 1.5, "price": -110}],
            },
            "moneyline": {
                "Home": [{"implied_prob": 0.5, "price": 100,
                          "book": "Test"}],
                "Away": [{"implied_prob": 0.5, "price": 100,
                          "book": "Test"}],
            },
        }

    @staticmethod
    def _matchup_features(complete=True):
        return {
            "starter_edge": 0.15,
            "expected_runs": {
                "complete": complete,
                "home_offense_factor": 1.15,
                "away_offense_factor": 0.90,
                "home_staff_suppression": 1.10,
                "away_staff_suppression": 0.85,
            },
        }

    def test_pythagorean_uses_modern_baseball_exponent(self):
        probability = mlb_starters.pythagorean_win_probability(5.0, 4.0)
        expected = 5.0 ** 1.83 / (5.0 ** 1.83 + 4.0 ** 1.83)
        self.assertAlmostEqual(probability, expected)
        self.assertAlmostEqual(
            probability + mlb_starters.pythagorean_win_probability(4.0, 5.0),
            1.0,
        )

    def test_expected_runs_respond_to_offense_and_run_prevention(self):
        neutral = mlb_starters.expected_runs_from_factors(4.5, 1.0, 1.0)
        self.assertEqual(neutral, 4.5)
        self.assertGreater(
            mlb_starters.expected_runs_from_factors(4.5, 1.2, 1.0),
            neutral,
        )
        self.assertLess(
            mlb_starters.expected_runs_from_factors(4.5, 1.0, 1.2),
            neutral,
        )

    def test_run_line_probabilities_are_complementary(self):
        favorite_cover = mlb_starters.poisson_margin_probability(
            4.5, 4.5, -1.5)
        underdog_cover = mlb_starters.poisson_margin_probability(
            4.5, 4.5, 1.5)
        self.assertLess(favorite_cover, 0.5)
        self.assertGreater(underdog_cover, 0.5)
        self.assertAlmostEqual(favorite_cover + underdog_cover, 1.0)

    def test_zero_dispersion_matches_poisson_run_line(self):
        poisson = mlb_starters.poisson_margin_probability(
            5.1, 3.8, -1.5)
        negative_binomial = (
            mlb_starters.negative_binomial_margin_probability(
                5.1, 3.8, -1.5, 0.0)
        )
        self.assertAlmostEqual(poisson, negative_binomial)

    def test_negative_binomial_run_lines_are_complementary(self):
        favorite_cover = (
            mlb_starters.negative_binomial_margin_probability(
                4.5, 4.5, -1.5, 0.2)
        )
        underdog_cover = (
            mlb_starters.negative_binomial_margin_probability(
                4.5, 4.5, 1.5, 0.2)
        )
        self.assertLess(favorite_cover, 0.5)
        self.assertGreater(underdog_cover, 0.5)
        self.assertAlmostEqual(favorite_cover + underdog_cover, 1.0)

    def test_challenger_maps_each_offense_to_opposing_staff(self):
        row = {
            "h_sp_sup": 0.9,
            "a_sp_sup": 1.1,
            "h_off_faced": 0.8,
            "a_off_faced": 1.2,
        }
        model = {
            "home_base_runs": 4.0,
            "away_base_runs": 4.0,
            "offense_weight": 1.0,
            "pitching_weight": 1.0,
        }
        home_runs, away_runs = backtest_starters.project_expected_runs(
            row, model)
        self.assertAlmostEqual(home_runs, 4.0 * 1.2 / 1.1)
        self.assertAlmostEqual(away_runs, 4.0 * 0.8 / 0.9)

    def test_park_and_opposing_bullpen_workload_adjust_expected_runs(self):
        row = {
            "home_team": "TST",
            "venue_id": "99",
            "h_sp_sup": 1.0,
            "a_sp_sup": 1.0,
            "h_off_faced": 1.0,
            "a_off_faced": 1.0,
            "h_bp_workload": 75.0,
            "a_bp_workload": 150.0,
        }
        model = {
            "home_base_runs": 4.0,
            "away_base_runs": 4.0,
            "offense_weight": 1.0,
            "pitching_weight": 1.0,
            "park_factors": {"99": 1.1},
            "park_strength": 1.0,
            "fatigue_weight": 0.2,
            "workload_center": 100.0,
        }
        home_runs, away_runs = backtest_starters.project_expected_runs(
            row, model)
        self.assertAlmostEqual(home_runs, 4.0 * 1.1 * math.exp(0.1))
        self.assertAlmostEqual(away_runs, 4.0 * 1.1 * math.exp(-0.05))

    def test_ensemble_weight_prefers_more_accurate_candidate(self):
        current = [0.4, 0.6, 0.4, 0.6]
        challenger = [0.8, 0.2, 0.8, 0.2]
        outcomes = [1, 0, 1, 0]
        self.assertEqual(
            backtest_starters._fit_blend_weight(
                current, challenger, outcomes),
            1.0,
        )

    def test_expected_runs_team_factors_use_savant_aggregates(self):
        offense_left = [
            {"player_name": "HME", "xwoba": "0.330", "pa": "100"},
            {"player_name": "AWY", "xwoba": "0.300", "pa": "100"},
        ]
        offense_right = [
            {"player_name": "HME", "xwoba": "0.350", "pa": "200"},
            {"player_name": "AWY", "xwoba": "0.310", "pa": "200"},
        ]
        bullpens = [
            {"player_name": "HME", "xwoba": "0.300", "pa": "150"},
            {"player_name": "AWY", "xwoba": "0.360", "pa": "150"},
        ]
        # Savant team keys are normalized against the season's StatsAPI abbrs;
        # give a fake index using the same placeholder abbrs so normalization is
        # a no-op (and the test stays hermetic — no live team-index fetch).
        fake_index = {
            "home": {"id": 1, "name": "Home", "abbr": "HME"},
            "away": {"id": 2, "name": "Away", "abbr": "AWY"},
        }
        with tempfile.TemporaryDirectory() as cache_dir, patch.object(
                mlb_starters, "CACHE_DIR", cache_dir), patch.object(
                mlb_starters, "get_team_index", return_value=fake_index), \
                patch.object(
                mlb_starters, "_get_savant_csv",
                side_effect=[offense_left, offense_right, bullpens]) as fetch:
            factors = mlb_starters.get_expected_runs_team_factors(
                2026, "2026-07-15")

        self.assertEqual(fetch.call_count, 3)
        for call in fetch.call_args_list:
            self.assertEqual(call.args[1]["game_date_lt"], "2026-07-14")
            self.assertEqual(call.args[1]["hfGT"], "R|")
        self.assertAlmostEqual(factors["league_xwoba"], 0.325)
        self.assertAlmostEqual(factors["league_bullpen_xwoba"], 0.330)
        self.assertEqual(factors["offense_vs_hand"]["L"]["HME"], 0.330)
        self.assertEqual(factors["offense_vs_hand"]["R"]["AWY"], 0.310)
        self.assertEqual(factors["bullpen_xwoba"]["HME"], 0.300)

    def test_live_matchup_features_expose_separate_run_factors(self):
        team_index = {
            "home": {"id": 1, "name": "Home", "abbr": "HME"},
            "away": {"id": 2, "name": "Away", "abbr": "AWY"},
        }
        probables = {
            "home": {"pitcher_id": 11},
            "away": {"pitcher_id": 22},
        }
        qualities = {
            11: {"throws": "R", "run_suppression": 1.2,
                 "run_suppression_basis": "xera", "avg_ip": 6.0,
                 "xwoba": 0.280},
            22: {"throws": "L", "run_suppression": 0.8,
                 "run_suppression_basis": "era", "avg_ip": 4.5,
                 "xwoba": 0.360},
        }
        offense_splits = {
            1: {"vL": {"ops": 0.780}, "vR": {"ops": 0.750}},
            2: {"vL": {"ops": 0.700}, "vR": {"ops": 0.640}},
        }
        bullpens = {
            1: {"bullpen_suppression": 1.1},
            2: {"bullpen_suppression": 0.9},
        }
        expected_inputs = {
            "league_xwoba": 0.320,
            "league_bullpen_xwoba": 0.330,
            "offense_vs_hand": {
                "L": {"HME": 0.352, "AWY": 0.310},
                "R": {"HME": 0.330, "AWY": 0.288},
            },
            "bullpen_xwoba": {"HME": 0.300, "AWY": 0.360},
        }
        with patch.object(
                mlb_starters, "get_probable_starters",
                return_value=probables), patch.object(
                mlb_starters, "get_pitcher_quality",
                side_effect=lambda pitcher_id, season: qualities[pitcher_id]), patch.object(
                mlb_starters, "get_team_offense_splits",
                side_effect=lambda team_id, season: offense_splits[team_id]), patch.object(
                mlb_starters, "get_team_bullpen_quality",
                side_effect=lambda team_id, season: bullpens[team_id]), patch.object(
                mlb_starters, "get_expected_runs_team_factors",
                side_effect=[expected_inputs, None]):
            features = mlb_starters.build_matchup_features(
                "Home", "Away", "2026-07-15", 2026,
                team_index=team_index)
            fallback_features = mlb_starters.build_matchup_features(
                "Home", "Away", "2026-07-15", 2026,
                team_index=team_index)

        factors = features["expected_runs"]
        self.assertTrue(factors["complete"])
        legacy_home_staff = (6.0 / 9.0) * 1.2 + (3.0 / 9.0) * 1.1
        legacy_away_staff = 0.5 * 0.8 + 0.5 * 0.9
        self.assertAlmostEqual(
            features["starter_edge"],
            math.tanh(
                legacy_home_staff / (0.640 / 0.711)
                - legacy_away_staff / (0.780 / 0.711)))
        self.assertAlmostEqual(
            fallback_features["starter_edge"], features["starter_edge"])
        self.assertFalse(fallback_features["expected_runs"]["complete"])
        self.assertAlmostEqual(
            factors["home_offense_factor"], 0.352 / 0.320)
        self.assertAlmostEqual(
            factors["away_offense_factor"], 0.288 / 0.320)
        self.assertAlmostEqual(
            factors["home_staff_suppression"],
            (6.0 / 9.0) * (0.320 / 0.280)
            + (3.0 / 9.0) * (0.330 / 0.300))
        self.assertAlmostEqual(
            factors["away_staff_suppression"],
            0.5 * (0.320 / 0.360) + 0.5 * (0.330 / 0.360))

    def test_mlb_spreads_use_validated_expected_runs_ensemble(self):
        game_odds = self._game_odds()
        home_stats, away_stats = self._team_stats()
        features = self._matchup_features()
        with patch.object(
                analysis, "load_expected_runs_challenger",
                return_value=self._calibration()), patch.object(
                analysis, "_shrink_factor", return_value=0.25), patch.object(
                analysis, "_blend_weight", return_value=1.0):
            current_margin, pred_std, _, _ = analysis._predict_margin(
                game_odds, home_stats, away_stats,
                "baseball_mlb", features)
            candidates = analysis.analyze_spreads_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb", matchup_features=features)

        home_runs = mlb_starters.expected_runs_from_factors(
            4.423, 1.15, 0.85, 1.25, 0.75)
        away_runs = mlb_starters.expected_runs_from_factors(
            4.429, 0.90, 1.10, 1.25, 0.75)
        current_cover = analysis._norm_cdf(
            (current_margin - 1.5) / pred_std)
        current_adjusted = 0.5 + 0.25 * (current_cover - 0.5)
        expected_cover = mlb_starters.poisson_margin_probability(
            home_runs, away_runs, -1.5)
        ensemble_cover = (
            0.30 * current_adjusted + 0.70 * expected_cover)
        ensemble_margin = (
            0.10 * current_margin + 0.90 * (home_runs - away_runs))
        home = next(c for c in candidates if c["home_away"] == "HOME")
        away = next(c for c in candidates if c["home_away"] == "AWAY")
        self.assertEqual(home["model_source"], "expected_runs_ensemble")
        self.assertEqual(home["cover_rate"], round(ensemble_cover * 100, 2))
        self.assertEqual(home["pred_game_margin"], round(ensemble_margin, 2))
        self.assertEqual(home["expected_home_runs"], round(home_runs, 2))
        self.assertEqual(home["expected_away_runs"], round(away_runs, 2))
        self.assertAlmostEqual(
            home["cover_rate"] + away["cover_rate"], 100.0)

    def test_incomplete_mlb_inputs_fall_back_to_current_spread_model(self):
        game_odds = self._game_odds()
        home_stats, away_stats = self._team_stats()
        baseline_features = {"starter_edge": 0.15}
        incomplete_features = self._matchup_features(complete=False)
        with patch.object(analysis, "_blend_weight", return_value=1.0):
            baseline = analysis.analyze_spreads_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb",
                matchup_features=baseline_features)
            fallback = analysis.analyze_spreads_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb",
                matchup_features=incomplete_features)
        self.assertEqual(fallback, baseline)
        self.assertTrue(all(
            candidate["model_source"] == "current_margin_model"
            for candidate in fallback))

    def test_expected_runs_inputs_do_not_change_moneyline_or_other_sports(self):
        game_odds = self._game_odds()
        home_stats, away_stats = self._team_stats()
        baseline_features = {"starter_edge": 0.15}
        complete_features = self._matchup_features()
        with patch.object(analysis, "_blend_weight", return_value=1.0):
            baseline_ml = analysis.analyze_moneyline_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb",
                matchup_features=baseline_features)
            expected_runs_ml = analysis.analyze_moneyline_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb",
                matchup_features=complete_features)
            baseline_other = analysis.analyze_spreads_value(
                game_odds, home_stats, away_stats,
                sport_key="basketball_nba",
                matchup_features=baseline_features)
            expected_runs_other = analysis.analyze_spreads_value(
                game_odds, home_stats, away_stats,
                sport_key="basketball_nba",
                matchup_features=complete_features)
        self.assertEqual(expected_runs_ml, baseline_ml)
        self.assertEqual(expected_runs_other, baseline_other)

    @patch("backtest_starters._season_venue_index")
    @patch("backtest_props.season_schedule")
    def test_game_enrichment_attaches_actual_venue(
            self, schedule, venue_index):
        schedule.return_value = {
            "2025-04-01": [{
                "home_abbr": "HME", "away_abbr": "AWY",
                "home_sp": 10, "away_sp": 20,
            }],
        }
        venue_index.return_value = (
            {("2025-04-01", "10", "20"): "123"}, {})
        games = [{
            "date": "2025-04-01", "home_sp": 10, "away_sp": 20,
            "home_win": 1, "total_runs": 7, "margin": 3,
        }]
        enriched = backtest_starters._enrich_games(
            games, 2025, include_venues=True)
        self.assertEqual(enriched[0]["venue_id"], "123")

    @patch("backtest_props.season_schedule")
    def test_bullpen_workload_uses_only_prior_relief_pitches(self, schedule):
        schedule.return_value = {
            "2024-04-01": [{
                "home_abbr": "HME", "away_abbr": "AWY",
                "home_sp": 10, "away_sp": 20,
            }],
            "2024-04-02": [{
                "home_abbr": "AWY", "away_abbr": "HME",
                "home_sp": 20, "away_sp": 10,
            }],
        }
        rows = [
            # AWY pitches while HME bats. Only pitcher 21 is a reliever.
            {"game_date": "2024-04-01", "batting_team": "HME",
             "pitcher": "20"},
            {"game_date": "2024-04-01", "batting_team": "HME",
             "pitcher": "21"},
            {"game_date": "2024-04-01", "batting_team": "HME",
             "pitcher": "21"},
        ]
        workload = backtest_starters._bullpen_workload_features(rows, 2024)
        self.assertEqual(workload[("2024-04-01", "AWY")], 0.0)
        self.assertEqual(workload[("2024-04-02", "AWY")], 2.0)


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
    def test_player_history_keeps_current_team_when_gamelog_omits_it(self):
        with patch.object(espn_client, "search_athlete", return_value={
                "id": "3934672", "name": "Jalen Brunson", "team_id": "18",
        }), patch.object(espn_client, "get_athlete_gamelog", return_value=[
                {"PTS": 25.0, "opponent": "Indiana Pacers",
                 "is_home": False, "team_id": None},
        ]):
            history = espn_client.get_player_stat_history(
                "basketball", "nba", "Jalen Brunson", "player_points")

        self.assertTrue(history["found"])
        self.assertEqual(history["team_id"], "18")

    def test_bet_checklist_entries_include_instructions_not_pricing(self):
        common = {"event_id": "game-1", "edge_pct": 8.5,
                  "best_price": -110, "expected_roi_pct": 12.0}
        entries = [
            analysis.make_bet_checklist_entry({
                **common,
                "team": "Indiana Pacers",
                "opponent": "New York Knicks",
                "home_away": "HOME",
            }, "moneyline"),
            analysis.make_bet_checklist_entry({
                **common,
                "team": "New York Knicks",
                "opponent": "Indiana Pacers",
                "home_away": "AWAY",
                "spread": 3.5,
            }, "spread"),
            analysis.make_bet_checklist_entry({
                **common,
                "matchup": "New York Knicks @ Indiana Pacers",
                "line": 224.5,
            }, "total", side="UNDER"),
            analysis.make_bet_checklist_entry({
                **common,
                "matchup": "New York Knicks @ Indiana Pacers",
                "player": "Jalen Brunson",
                "team": "New York Knicks",
                "prop": "player_points",
                "prop_label": "Points",
                "direction": "OVER",
                "line": 24.5,
            }, "player_prop"),
            analysis.make_bet_checklist_entry({
                **common,
                "matchup": "New York Knicks @ Indiana Pacers",
                "player": "Jalen Brunson",
                "team": "New York Knicks",
                "prop": "player_points",
                "prop_label": "Points",
                "direction": "OVER",
                "line": 24.5,
                "safe_mode": True,
                "safe_threshold": 20,
            }, "player_prop"),
        ]

        self.assertEqual(entries[0]["bet"], "Indiana Pacers moneyline")
        self.assertEqual(
            entries[0]["matchup"], "New York Knicks @ Indiana Pacers")
        self.assertEqual(entries[1]["bet"], "New York Knicks +3.5")
        self.assertEqual(entries[2]["bet"], "UNDER 224.5")
        self.assertEqual(entries[2]["team"], "Both teams")
        self.assertEqual(
            entries[3]["bet"], "Jalen Brunson — Points OVER 24.5")
        self.assertEqual(entries[3]["team"], "New York Knicks")
        self.assertEqual(entries[4]["bet"], "Jalen Brunson — Points 20+")
        for entry in entries:
            self.assertEqual(set(entry), {
                "selection_key", "type", "bet", "matchup", "team"})
            self.assertTrue(entry["selection_key"].startswith("bet_selection:"))

    def test_market_comparison_keeps_dk_separate_from_peer_medians(self):
        def book(key, spread, spread_price, total, total_price,
                 complete=True):
            spread_outcomes = [
                {"name": "Home", "point": spread,
                 "price": spread_price},
            ]
            if complete:
                spread_outcomes.append({
                    "name": "Away", "point": -spread, "price": -110})
            return {
                "key": key,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Home", "price": -110},
                            {"name": "Away", "price": -110},
                        ],
                    },
                    {"key": "spreads", "outcomes": spread_outcomes},
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": total,
                             "price": total_price},
                            {"name": "Under", "point": total,
                             "price": -110},
                        ],
                    },
                ],
            }

        game = {
            "home_team": "Home",
            "away_team": "Away",
            "bookmakers": [
                book("draftkings", 1.5, -190, 8.5, -115),
                book("peer-one", -1.5, 150, 9.0, -105),
                book("peer-two", -1.5, 155, 9.0, -110),
                book("peer-three", -1.5, 160, 9.0, -115),
                book("incomplete", -2.5, 250, 10.0, 200,
                     complete=False),
            ],
        }
        comparisons = odds_client.build_market_comparisons(game)

        spread = comparisons["spreads"]["Home"]
        self.assertEqual(spread["primary_line"], 1.5)
        self.assertEqual(spread["primary_price"], -190)
        self.assertEqual(spread["peer_median_line"], -1.5)
        self.assertEqual(spread["peer_median_price"], 155)
        self.assertEqual(spread["peer_count"], 3)
        self.assertEqual(spread["line_advantage"], 3.0)
        self.assertFalse(spread["dominates_peer_offer"])
        self.assertNotIn("edge", spread)

        over = comparisons["totals"]["Over"]
        under = comparisons["totals"]["Under"]
        self.assertEqual(over["line_advantage"], 0.5)
        self.assertEqual(under["line_advantage"], -0.5)

        game["bookmakers"] = [
            book("draftkings", 3.5, -115, 8.5, -110),
            book("peer-one", 2.5, -110, 8.5, -110),
            book("peer-two", 2.5, -105, 8.5, -110),
            book("peer-three", 2.5, -115, 8.5, -110),
        ]
        key_number = odds_client.build_market_comparisons(
            game)["spreads"]["Home"]
        self.assertEqual(key_number["key_numbers"], [3.0])

    def test_market_consensus_excludes_draftkings_and_cross_line_peers(self):
        def book(key, home_price, away_price, home_point=-1.5):
            return {
                "key": key,
                "markets": [{
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Home", "price": home_price,
                         "point": home_point},
                        {"name": "Away", "price": away_price,
                         "point": -home_point},
                    ],
                }],
            }

        game = {
            "id": "game",
            "home_team": "Home",
            "away_team": "Away",
            "bookmakers": [
                book("draftkings", 110, -130),
                book("peer-one", -110, -110),
                book("peer-two", -105, -115),
                book("peer-three", -115, -105),
                book("cross-line-outlier", 200, -250, home_point=-2.5),
            ],
        }
        rows = backtest_market_consensus._build_market_observations(
            game,
            {"date": "2025-07-01", "home_score": 5, "away_score": 3},
            "spreads",
            min_peer_books=3,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["peer_count"], 3)
        self.assertNotIn("draftkings", rows[0]["peer_books"])
        self.assertNotIn("cross-line-outlier", rows[0]["peer_books"])
        self.assertAlmostEqual(
            sum(row["peer_probability"] for row in rows), 1.0)
        self.assertEqual([row["result"] for row in rows], [1, -1])

    def test_market_consensus_grades_totals_and_run_lines_by_side(self):
        spread = {"point_a": -1.5, "point_b": 1.5}
        total = {"point_a": 7.5, "point_b": 7.5}
        self.assertEqual(
            backtest_market_consensus._grade_side(
                "spreads", "a", spread, 5, 3),
            1,
        )
        self.assertEqual(
            backtest_market_consensus._grade_side(
                "spreads", "b", spread, 5, 3),
            -1,
        )
        self.assertEqual(
            backtest_market_consensus._grade_side(
                "totals", "a", total, 5, 3),
            1,
        )
        self.assertEqual(
            backtest_market_consensus._grade_side(
                "totals", "b", total, 5, 3),
            -1,
        )

    def test_line_advantage_keeps_draftkings_price_in_the_grade(self):
        def book(key, home_point, home_price, away_price):
            return {
                "key": key,
                "markets": [{
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Home", "point": home_point,
                         "price": home_price},
                        {"name": "Away", "point": -home_point,
                         "price": away_price},
                    ],
                }],
            }

        game = {
            "id": "game",
            "home_team": "Home",
            "away_team": "Away",
            "bookmakers": [
                book("draftkings", 1.5, -190, 155),
                book("peer-one", -1.5, 155, -185),
                book("peer-two", -1.5, 160, -190),
                book("peer-three", -1.5, 150, -180),
            ],
        }
        rows = backtest_market_consensus._build_line_advantage_observations(
            game,
            {"date": "2025-07-01", "home_score": 4, "away_score": 3},
            "spreads",
            min_peer_books=3,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["side"], "home")
        self.assertEqual(rows[0]["point_advantage"], 3.0)
        self.assertEqual(rows[0]["price"], -190)
        self.assertFalse(rows[0]["dominates_peer_offer"])
        self.assertAlmostEqual(rows[0]["profit"], 100 / 190)

        game["bookmakers"] = [
            book("draftkings", 3.5, -115, -105),
            book("peer-one", 2.5, -110, -110),
            book("peer-two", 2.5, -105, -115),
            book("peer-three", 2.5, -115, -105),
        ]
        rows = backtest_market_consensus._build_line_advantage_observations(
            game,
            {"date": "2025-07-01", "home_score": 4, "away_score": 3},
            "spreads",
            min_peer_books=3,
        )
        self.assertEqual(rows[0]["crossed_key_numbers"], [3.0])

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

    def test_maintenance_resolves_before_testing_refit_gate(self):
        # blob-url -> "" keeps compact_prediction_log off live Azure (secrets.toml
        # is present locally); a None last-fit forces the refit branch the way the
        # old os.path.exists=False mock did before the gate became blob-aware.
        with patch.object(
                recalibration, "resolve_pending_outcomes", return_value=25), patch.object(
                recalibration, "resolve_pending_market_outcomes", return_value=0), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=""), patch.object(
                recalibration, "_load_recal_cached", return_value=(None, {})), patch.object(
                recalibration, "compact_prediction_log", return_value=0), patch.object(
                recalibration, "refit_sport", return_value={"prop": (1, 0, 90)}) as refit:
            result = recalibration.maintain_sport("baseball_mlb")
        # Team-market resolution rides along but stays OUT of newly_resolved (that
        # gates the prop Platt refit).
        self.assertEqual(
            result,
            {"newly_resolved": 25, "newly_resolved_markets": 0, "refit": True})
        refit.assert_called_once_with(
            "baseball_mlb", resolve_first=False, newly_resolved=25)

    def test_outcome_resolution_refreshes_recent_gamelogs(self):
        rows = [{
            "ts": "2024-04-01T10:00:00Z",
            "sport_key": "baseball_mlb",
            "prop_key": "batter_hits",
            "player": "Player One",
            "game_date": "2024-04-01",
            "line": 0.5,
            "resolved": False,
        }]

        def mutate(mutator, where=None):
            return mutator(rows)

        # _resolve_mlb_actual -> None forces the ESPN fallback (and keeps this
        # MLB row off live statsapi, which the hard-ID path would otherwise hit).
        with patch.object(recalibration, "_read_log", return_value=rows), patch.object(
                recalibration, "_resolve_mlb_actual", return_value=None), patch(
                "espn_cache.cached_athlete_id", return_value="123"), patch(
                "espn_cache.cached_gamelog",
                return_value=[{"game_date": "2024-04-01", "H": 1}],
        ) as gamelog, patch.object(
                recalibration, "_stat_label", return_value="H"), patch.object(
                recalibration, "mutate_prediction_log", side_effect=mutate):
            resolved = recalibration.resolve_pending_outcomes("baseball_mlb")

        self.assertEqual(resolved, 1)
        self.assertTrue(rows[0]["resolved"])
        gamelog.assert_called_once_with(
            "baseball", "mlb", "123", ttl_hours=6)

    def test_forced_odds_refresh_never_uses_expired_cache(self):
        response = requests.Response()
        response.status_code = 429
        response.url = "https://example.test/odds"
        with patch.object(
                odds_client, "_get_with_retry", return_value=response), patch.object(
                odds_client, "_read_cache_expired", return_value={"stale": True},
        ) as expired:
            with self.assertRaises(requests.HTTPError):
                odds_client.get_event_odds(
                    "key", "baseball_mlb", "event-1",
                    markets="batter_hits", force_refresh=True)
        expired.assert_not_called()

    def test_local_prediction_log_transaction(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
                recalibration, "PRED_DIR", temp_dir), patch.object(
                recalibration, "CALIB_DIR", temp_dir), patch.object(
                recalibration, "LOG_PATH",
                os.path.join(temp_dir, "prediction_log.jsonl")), patch.object(
                recalibration, "_prediction_log_blob_url", return_value=""):
            written = recalibration.log_prediction_rows([{"test": 1}])
            rows = recalibration.read_prediction_log()
        self.assertEqual(written, 1)
        self.assertEqual(rows, [{"test": 1}])


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


class BestPriceHybridTests(unittest.TestCase):
    """P1.1b: props line-shop the BEST price across U.S. books for edge/EV, but
    carry the DraftKings price separately for staking/display."""

    def _game(self, books):
        return {
            "id": "g1", "home_team": "H", "away_team": "A",
            "commence_time": "2026-07-20T23:10:00Z", "sport_key": "baseball_mlb",
            "bookmakers": books,
        }

    def _book(self, title, over, under):
        outcomes = [{"description": "Player", "name": "Over",
                     "price": over, "point": 1.5}]
        if under is not None:
            outcomes.append({"description": "Player", "name": "Under",
                             "price": under, "point": 1.5})
        return {"title": title, "markets": [{"key": "batter_hits",
                                             "outcomes": outcomes}]}

    def test_dk_price_carved_out_alongside_best(self):
        game = self._game([
            self._book("DraftKings", over=100, under=-120),
            self._book("BetMGM", over=150, under=-110),
        ])
        info = parse_player_props(game)["props"]["batter_hits"]["Player"]
        # Best across books drives value/EV.
        self.assertEqual(info["over_price"], 150)
        self.assertEqual(info["over_book"], "BetMGM")
        self.assertEqual(info["under_price"], -110)
        # DraftKings carved out for staking/display.
        self.assertEqual(info["dk_over_price"], 100)
        self.assertEqual(info["dk_over_book"], "DraftKings")
        self.assertEqual(info["dk_under_price"], -120)

    def test_dk_absent_is_none(self):
        game = self._game([
            self._book("BetMGM", over=150, under=-110),
            self._book("Caesars", over=140, under=-115),
        ])
        info = parse_player_props(game)["props"]["batter_hits"]["Player"]
        self.assertEqual(info["over_price"], 150)  # best still resolves
        self.assertIsNone(info["dk_over_price"])
        self.assertIsNone(info["dk_under_price"])
        self.assertIsNone(info["dk_over_book"])


class SharpWeightedConsensusTests(unittest.TestCase):
    """P1.1c: the prop de-vig consensus up-weights sharp books (Pinnacle/Circa)
    and drops stale quotes, reducing to the plain arithmetic mean when neither
    a sharp book nor timestamps are present."""

    def _game(self, books):
        return {
            "id": "g1", "home_team": "H", "away_team": "A",
            "commence_time": "2026-07-20T23:10:00Z", "sport_key": "baseball_mlb",
            "bookmakers": books,
        }

    def _book(self, title, over, under, last_update=None):
        market = {"key": "batter_hits", "outcomes": [
            {"description": "P", "name": "Over", "price": over, "point": 1.5},
            {"description": "P", "name": "Under", "price": under, "point": 1.5},
        ]}
        if last_update is not None:
            market["last_update"] = last_update
        return {"title": title, "markets": [market]}

    def test_sharp_book_pulls_consensus_toward_it(self):
        game = self._game([
            self._book("Book A", -110, -110),      # fair_over 0.500
            self._book("Book B", -110, -110),      # fair_over 0.500
            self._book("Pinnacle", 200, -250),     # fair_over ~0.318
        ])
        info = parse_player_props(game)["props"]["batter_hits"]["P"]
        # Weighted (Pinnacle x3): (0.5+0.5+3*0.3182)/5 ≈ 0.391, vs plain mean
        # ≈ 0.439. Consensus is pulled toward the sharp book.
        self.assertAlmostEqual(info["over_implied"], 0.3909, places=3)
        self.assertLess(info["over_implied"], 0.439)

    def test_stale_book_dropped(self):
        game = self._game([
            self._book("Fresh", -110, -110, last_update="2026-07-20T18:00:00Z"),
            self._book("Stale", -300, 250, last_update="2026-07-20T17:00:00Z"),
        ])
        info = parse_player_props(game)["props"]["batter_hits"]["P"]
        # Stale quote (>600s behind) is excluded; only the fresh 0.5 survives.
        self.assertAlmostEqual(info["over_implied"], 0.5, places=3)

    def test_no_sharp_no_timestamp_matches_plain_mean(self):
        game = self._game([
            self._book("Book A", -110, -110),     # fair_over 0.5000
            self._book("Book B", 100, -120),      # fair_over ~0.4783
        ])
        info = parse_player_props(game)["props"]["batter_hits"]["P"]
        self.assertAlmostEqual(info["over_implied"], (0.5 + 0.4783) / 2, places=3)


class MarketPriorShrinkageTests(unittest.TestCase):
    """P1.1a: blend the calibrated OVER prob toward the de-vigged market prob
    with w = n/(n+k). Thin samples lean on the market prior (collapsing false
    edges); large samples keep the model.

    Uses sport_key=None so calibration/recalibration/refit/statsapi/logging are
    all bypassed (method-A passthrough: over_rate == empirical over-rate), and
    the reliability filter's participation/layoff gates are disabled — leaving a
    clean, hermetic path through the runtime prop pipeline.
    """

    def _prop_data(self, line=0.5, over_implied=0.45, under_implied=0.55):
        return {
            "commence_time": "2026-07-20T23:10:00Z",  # late US game (next-day UTC)
            "home_team": "Home Nine",
            "away_team": "Away Nine",
            "game_id": "evt1",
            "props": {
                "batter_hits": {
                    "Slumping Sammy": {
                        "line": line,
                        "over_implied": over_implied,
                        "under_implied": under_implied,
                        "over_price": -110,
                        "under_price": -110,
                        "over_book": "DK",
                        "under_book": "DK",
                    }
                }
            },
        }

    def _histories(self, n_games, value=0.0):
        # n_games consecutive daily games (no layoffs), all the same stat value
        # so the weighted over-rate is deterministic regardless of decay.
        dates = [f"2026-06-{d:02d}" for d in range(1, 1 + n_games)]
        return {
            "Slumping Sammy": {
                "batter_hits": {
                    "found": True,
                    "values": [value] * n_games,
                    "game_dates": list(reversed(dates)),  # newest-first
                }
            }
        }

    def _run(self, prop_data, histories, k):
        # k is injected via the per-sport default (sport_key=None reads
        # DEFAULT_PLAYER_PROP_MARKET_PRIOR_K) so no calibration file is needed.
        with patch.object(props, "DEFAULT_PLAYER_PROP_MARKET_PRIOR_K", k):
            cands = props.analyze_player_props_value(
                prop_data, histories, threshold_pct=1.0, sport_key=None)
        return cands[0]

    def test_thin_sample_edge_collapses_toward_market(self):
        # A 0-for-15 hitter at a 0.5 line: raw model says P(over)=0 -> a huge
        # false UNDER edge. The market prior should pull the prob toward 45%.
        pd_ = self._prop_data(over_implied=0.45, under_implied=0.55)
        hist = self._histories(15, value=0.0)

        base = self._run(pd_, hist, k=0)
        self.assertIsNone(base["market_prior"])
        self.assertEqual(base["over_rate"], 0.0)  # pure model
        self.assertEqual(base["direction"], "UNDER")
        base_edge = base["edge_pct"]

        shrunk = self._run(pd_, hist, k=15)
        meta = shrunk["market_prior"]
        self.assertIsNotNone(meta)
        self.assertEqual(meta["k"], 15)
        self.assertEqual(meta["n"], 15)
        self.assertAlmostEqual(meta["w"], 15 / 30, places=3)
        self.assertEqual(meta["pre_blend"], 0.0)  # raw model unchanged by blend
        # over_rate blended 0.5*0 + 0.5*45% = 22.5%, moving toward the market.
        self.assertAlmostEqual(shrunk["over_rate"], 22.5, places=1)
        # The false UNDER edge shrinks substantially.
        self.assertLess(shrunk["edge_pct"], base_edge)
        self.assertAlmostEqual(shrunk["edge_pct"], 22.5, places=1)

    def test_large_sample_edge_preserved(self):
        # With 200 games the model is trusted: w ~ 0.98, prob barely moves.
        pd_ = self._prop_data(over_implied=0.45, under_implied=0.55)
        hist = self._histories(200, value=0.0)

        base = self._run(pd_, hist, k=0)
        shrunk = self._run(pd_, hist, k=15)
        self.assertEqual(base["market_prior"], None)
        self.assertGreater(shrunk["market_prior"]["w"], 0.9)
        # Movement is tiny: over_rate stays near the model's 0.
        self.assertLess(shrunk["over_rate"], 5.0)
        self.assertAlmostEqual(
            shrunk["edge_pct"], base["edge_pct"], delta=5.0)

    def test_k_zero_is_a_no_op(self):
        pd_ = self._prop_data()
        hist = self._histories(15, value=1.0)  # 15-for-15 -> P(over 0.5)=1
        base = self._run(pd_, hist, k=0)
        self.assertIsNone(base["market_prior"])
        self.assertEqual(base["over_rate"], 100.0)


class BestMarketPriorKSweepTests(unittest.TestCase):
    """P1.1a backtest k-sweep: _best_market_prior_k should pick k>0 when a
    thin-sample noisy model is corrected by an accurate market, and report the
    false-positive (bet-count) collapse."""

    def test_prefers_shrinkage_when_market_is_accurate(self):
        from backtest import _best_market_prior_k
        # Noisy thin-sample model (n=6) that is confidently wrong half the time,
        # vs a market pinned at the true 50/50. Outcomes alternate.
        obs = []
        for i in range(40):
            outcome = i % 2
            model_p = 0.05 if outcome == 1 else 0.95  # wrong-way confident
            obs.append((model_p, 0.5, outcome, 6))
        res = _best_market_prior_k(obs)
        self.assertIsNotNone(res)
        best_k, best_brier, model_brier, bets_k0, bets_best = res
        self.assertGreater(best_k, 0)
        self.assertLess(best_brier, model_brier)
        # The wrong-way "edges" the model flagged collapse under the prior.
        self.assertLess(bets_best, bets_k0)

    def test_keeps_model_when_it_is_sharp_and_sample_large(self):
        from backtest import _best_market_prior_k
        # Large-sample model that is always right; shrinking toward a coin-flip
        # market only hurts, so k=0 should win.
        obs = []
        for i in range(40):
            outcome = i % 2
            model_p = 0.98 if outcome == 1 else 0.02
            obs.append((model_p, 0.5, outcome, 400))
        res = _best_market_prior_k(obs)
        best_k, best_brier, model_brier, bets_k0, bets_best = res
        self.assertEqual(best_k, 0)

    def test_empty_returns_none(self):
        from backtest import _best_market_prior_k
        self.assertIsNone(_best_market_prior_k([]))


class ParkFactorTableTests(unittest.TestCase):
    """P1.2: static MLB park-factor table + name normalization."""

    def test_park_key_normalizes_and_aliases(self):
        self.assertEqual(park_factors._park_key("Colorado Rockies"),
                         "coloradorockies")
        # Relocation / rename aliases collapse to one canonical key.
        self.assertEqual(park_factors._park_key("Oakland Athletics"),
                         "athletics")
        self.assertEqual(park_factors._park_key("Sacramento Athletics"),
                         "athletics")
        self.assertEqual(park_factors._park_key("Cleveland Indians"),
                         "clevelandguardians")
        self.assertEqual(park_factors._park_key("Arizona D-backs"),
                         "arizonadiamondbacks")
        self.assertEqual(park_factors._park_key(None), "")

    def test_park_factor_lookup(self):
        # Coors is hitter/run friendly; a marine pitcher park suppresses.
        self.assertGreater(park_factors.park_factor("Colorado Rockies", "hits"),
                           1.0)
        self.assertGreater(park_factors.park_factor("Colorado Rockies", "runs"),
                           park_factors.park_factor("Colorado Rockies", "hits"))
        self.assertLess(park_factors.park_factor("Seattle Mariners", "hits"),
                        1.0)
        # Unknown team or kind → neutral 1.0 (fail closed).
        self.assertEqual(park_factors.park_factor("Nowhere FC", "hits"), 1.0)
        self.assertEqual(park_factors.park_factor("Colorado Rockies", "steals"),
                         1.0)
        # Aliases resolve to the same factor as the canonical name.
        self.assertEqual(park_factors.park_factor("Oakland Athletics", "hits"),
                         park_factors.park_factor("Athletics", "hits"))


class ParkFactorMultTests(unittest.TestCase):
    """P1.2: road-context delta multiplier (props._park_factor_mult)."""

    def test_neutral_sample_to_coors_raises(self):
        mult, meta = props._park_factor_mult(
            "batter_hits", ["Houston Astros", "Houston Astros"], [1.0, 1.0],
            "Colorado Rockies", 1.0)
        self.assertGreater(mult, 1.0)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["kind"], "hits")
        self.assertAlmostEqual(meta["pf_base"], 1.0, places=3)
        self.assertAlmostEqual(
            mult, park_factors.park_factor("Colorado Rockies", "hits"),
            places=3)

    def test_coors_heavy_sample_no_double_count(self):
        # A Rockies hitter (home logs already at Coors) going to Coors: the
        # delta is ≈1.0 — the point of the road-context formulation.
        mult, meta = props._park_factor_mult(
            "batter_hits",
            ["Colorado Rockies", "Colorado Rockies", "Colorado Rockies"],
            [1.0, 1.0, 1.0], "Colorado Rockies", 1.0)
        self.assertAlmostEqual(mult, 1.0, places=3)
        self.assertIsNone(meta)

    def test_earned_runs_uses_runs_kind(self):
        mult, meta = props._park_factor_mult(
            "pitcher_earned_runs", ["Houston Astros"], [1.0],
            "Colorado Rockies", 1.0)
        self.assertEqual(meta["kind"], "runs")
        # Runs factor is capped by PARK_FACTOR_BOUNDS.
        self.assertAlmostEqual(mult, props.PARK_FACTOR_BOUNDS[1], places=3)

    def test_strength_scales_effect(self):
        full, _ = props._park_factor_mult(
            "batter_hits", ["Houston Astros"], [1.0], "Colorado Rockies", 1.0)
        half, _ = props._park_factor_mult(
            "batter_hits", ["Houston Astros"], [1.0], "Colorado Rockies", 0.5)
        self.assertAlmostEqual(half, 1.0 + 0.5 * (full - 1.0), places=3)

    def test_bounds_clamp_extreme_delta(self):
        # Coors baseline → a strong pitcher park would push below the floor.
        mult, _ = props._park_factor_mult(
            "batter_hits", ["Colorado Rockies"], [1.0], "Seattle Mariners", 1.0)
        self.assertEqual(mult, props.PARK_FACTOR_BOUNDS[0])

    def test_missing_data_is_neutral(self):
        # No known past parks → baseline undefined → neutral.
        mult, meta = props._park_factor_mult(
            "batter_hits", [None, None], [1.0, 1.0], "Colorado Rockies", 1.0)
        self.assertEqual((mult, meta), (1.0, None))
        # Unknown upcoming park → neutral.
        mult, meta = props._park_factor_mult(
            "batter_hits", ["Houston Astros"], [1.0], None, 1.0)
        self.assertEqual((mult, meta), (1.0, None))

    def test_non_mapped_prop_is_neutral(self):
        mult, meta = props._park_factor_mult(
            "pitcher_strikeouts", ["Houston Astros"], [1.0],
            "Colorado Rockies", 1.0)
        self.assertEqual((mult, meta), (1.0, None))

    def test_zero_strength_is_neutral(self):
        mult, meta = props._park_factor_mult(
            "batter_hits", ["Houston Astros"], [1.0], "Colorado Rockies", 0.0)
        self.assertEqual((mult, meta), (1.0, None))


class ParkFactorProjectionTests(unittest.TestCase):
    """P1.2: the park delta actually moves the projection through the runtime
    prop pipeline. Uses sport_key=None + a patched default strength so the MLB
    calibration/statsapi/logging paths stay out (like MarketPriorShrinkageTests),
    and all past games are AWAY at a neutral park so no espn_teams lookup is
    needed to resolve the baseline."""

    def _prop_data(self, home_team):
        return {
            "commence_time": "2026-07-20T23:10:00Z",
            "home_team": home_team,
            "away_team": "Houston Astros",  # the player's team (on the road)
            "game_id": "evt-park",
            "props": {
                "batter_hits": {
                    "Roady Rob": {
                        "line": 0.5,
                        "over_implied": 0.5,
                        "under_implied": 0.5,
                        "over_price": -110,
                        "under_price": -110,
                        "over_book": "DK",
                        "under_book": "DK",
                    }
                }
            },
        }

    def _histories(self, n=12):
        dates = [f"2026-06-{d:02d}" for d in range(1, 1 + n)]
        return {
            "Roady Rob": {
                "batter_hits": {
                    "found": True,
                    "values": [1.0] * n,               # 1 hit each → base_proj 1.0
                    "opponents": ["Minnesota Twins"] * n,  # neutral park
                    "home_aways": [False] * n,         # all road → park = opponent
                    "game_dates": list(reversed(dates)),
                }
            }
        }

    def _run(self, home_team, strength):
        with patch.object(props, "DEFAULT_PLAYER_PROP_PARK_STRENGTH", strength):
            cands = props.analyze_player_props_value(
                self._prop_data(home_team), self._histories(),
                threshold_pct=1.0, sport_key=None)
        return cands[0]

    def test_coors_upcoming_raises_projection(self):
        off = self._run("Colorado Rockies", 0.0)
        on = self._run("Colorado Rockies", 1.0)
        self.assertIsNone(off["park_factor"])
        self.assertAlmostEqual(off["avg_stat"], 1.0, places=2)
        self.assertIsNotNone(on["park_factor"])
        # Baseline is a neutral park; upcoming Coors → projection scales up.
        self.assertGreater(on["avg_stat"], off["avg_stat"])
        self.assertAlmostEqual(
            on["avg_stat"],
            round(park_factors.park_factor("Colorado Rockies", "hits"), 2),
            places=2)

    def test_neutral_upcoming_is_a_no_op(self):
        on = self._run("Minnesota Twins", 1.0)  # same neutral park as baseline
        self.assertIsNone(on["park_factor"])
        self.assertAlmostEqual(on["avg_stat"], 1.0, places=2)


class WeatherFactorGeoTests(unittest.TestCase):
    """P1.3: static park geo table + wind out/in-to-CF projection."""

    def test_geo_table_well_formed(self):
        self.assertGreaterEqual(len(weather_factors.MLB_PARK_GEO), 28)
        for name, geo in weather_factors.MLB_PARK_GEO.items():
            self.assertIn(geo["roof"], ("open", "retractable", "dome"), name)
            self.assertTrue(0 <= geo["cf_bearing"] < 360, name)
            self.assertTrue(20.0 <= geo["lat"] <= 50.0, name)
            self.assertTrue(-125.0 <= geo["lon"] <= -66.0, name)

    def test_unsettled_venues_omitted(self):
        # Athletics (Sacramento) + Rays (displaced) → neutral, mirroring park_factors.
        self.assertIsNone(weather_factors.park_geo("Athletics"))
        self.assertIsNone(weather_factors.park_geo("Oakland Athletics"))
        self.assertIsNone(weather_factors.park_geo("Tampa Bay Rays"))

    def test_park_geo_uses_park_factor_aliases(self):
        self.assertIsNotNone(weather_factors.park_geo("Colorado Rockies"))
        # "Arizona D-backs" normalizes (via park_factors._park_key) to the DBacks.
        self.assertEqual(
            weather_factors.park_geo("Arizona D-backs"),
            weather_factors.park_geo("Arizona Diamondbacks"))
        self.assertIsNone(weather_factors.park_geo("Nowhere FC"))

    def test_wind_out_component_sign(self):
        woc = weather_factors.wind_out_component
        # CF bearing 0 (points N). Wind FROM the south blows OUT to a N-facing CF.
        self.assertAlmostEqual(woc(10, 180, 0), 10.0, places=3)
        # Wind FROM the north blows IN.
        self.assertAlmostEqual(woc(10, 0, 0), -10.0, places=3)
        # Crosswind (from due east) ≈ no out/in component.
        self.assertAlmostEqual(woc(10, 90, 0), 0.0, places=3)
        # Missing inputs → None.
        self.assertIsNone(woc(None, 180, 0))
        self.assertIsNone(woc(10, None, 0))
        self.assertIsNone(woc(10, 180, None))


class WeatherFactorMultTests(unittest.TestCase):
    """P1.3: baseline-relative weather multiplier (props._weather_factor_mult)."""

    def _w(self, temp_f=70.0, wind_out=0.0, dome=False):
        return {"temp_f": temp_f, "wind_out_mph": wind_out, "dome": dome}

    def test_hot_raises_cold_lowers(self):
        hot, meta = props._weather_factor_mult("batter_hits", self._w(temp_f=90), 1.0)
        self.assertGreater(hot, 1.0)
        self.assertEqual(meta["kind"], "hits")
        cold, _ = props._weather_factor_mult("batter_hits", self._w(temp_f=50), 1.0)
        self.assertLess(cold, 1.0)

    def test_wind_out_raises_in_lowers(self):
        out, _ = props._weather_factor_mult("batter_hits", self._w(wind_out=10), 1.0)
        self.assertGreater(out, 1.0)
        inn, _ = props._weather_factor_mult("batter_hits", self._w(wind_out=-10), 1.0)
        self.assertLess(inn, 1.0)

    def test_runs_more_sensitive_than_hits(self):
        w = self._w(temp_f=90, wind_out=10)
        runs, _ = props._weather_factor_mult("pitcher_earned_runs", w, 1.0)
        hits, _ = props._weather_factor_mult("batter_hits", w, 1.0)
        self.assertGreater(runs, hits)

    def test_dome_is_neutral(self):
        self.assertEqual(
            props._weather_factor_mult("batter_hits",
                                       self._w(temp_f=95, wind_out=15, dome=True), 1.0),
            (1.0, None))

    def test_unmapped_prop_zero_strength_empty_neutral(self):
        self.assertEqual(
            props._weather_factor_mult("pitcher_strikeouts", self._w(temp_f=95), 1.0),
            (1.0, None))
        self.assertEqual(
            props._weather_factor_mult("batter_hits", self._w(temp_f=95), 0.0),
            (1.0, None))
        self.assertEqual(
            props._weather_factor_mult("batter_hits", None, 1.0), (1.0, None))
        self.assertEqual(
            props._weather_factor_mult("batter_hits", {}, 1.0), (1.0, None))
        self.assertEqual(
            props._weather_factor_mult(
                "batter_hits", {"temp_f": None, "wind_out_mph": None}, 1.0),
            (1.0, None))

    def test_strength_scales(self):
        full, _ = props._weather_factor_mult("batter_hits", self._w(wind_out=10), 1.0)
        half, _ = props._weather_factor_mult("batter_hits", self._w(wind_out=10), 0.5)
        self.assertAlmostEqual(half, 1.0 + 0.5 * (full - 1.0), places=4)

    def test_bounds_clamp(self):
        hi, _ = props._weather_factor_mult(
            "pitcher_earned_runs", self._w(temp_f=200, wind_out=100), 1.0)
        self.assertEqual(hi, props.WEATHER_FACTOR_BOUNDS[1])
        lo, _ = props._weather_factor_mult(
            "pitcher_earned_runs", self._w(temp_f=-100, wind_out=-100), 1.0)
        self.assertEqual(lo, props.WEATHER_FACTOR_BOUNDS[0])


class WeatherFactorProjectionTests(unittest.TestCase):
    """P1.3: the weather nudge moves the projection through the runtime prop
    pipeline. Same offline harness as ParkFactorProjectionTests (sport_key=None +
    patched default strength; park stays off at its 0.0 default)."""

    def _prop_data(self):
        return {
            "commence_time": "2026-07-20T23:10:00Z",
            "home_team": "Chicago Cubs",
            "away_team": "Houston Astros",
            "game_id": "evt-wx",
            "props": {
                "batter_hits": {
                    "Windy Will": {
                        "line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
                        "over_price": -110, "under_price": -110,
                        "over_book": "DK", "under_book": "DK",
                    }
                }
            },
        }

    def _histories(self, n=12):
        dates = [f"2026-06-{d:02d}" for d in range(1, 1 + n)]
        return {
            "Windy Will": {
                "batter_hits": {
                    "found": True,
                    "values": [1.0] * n,
                    "opponents": ["Minnesota Twins"] * n,
                    "home_aways": [False] * n,
                    "game_dates": list(reversed(dates)),
                }
            }
        }

    def _run(self, weather, strength):
        with patch.object(props, "DEFAULT_PLAYER_PROP_WEATHER_STRENGTH", strength):
            cands = props.analyze_player_props_value(
                self._prop_data(), self._histories(),
                threshold_pct=1.0, sport_key=None, weather=weather)
        return cands[0]

    def test_off_is_neutral(self):
        off = self._run({"temp_f": 95, "wind_out_mph": 12, "dome": False}, 0.0)
        self.assertIsNone(off["weather"])
        self.assertAlmostEqual(off["avg_stat"], 1.0, places=2)

    def test_hot_windy_raises_projection(self):
        weather = {"temp_f": 90, "wind_out_mph": 10, "dome": False}
        on = self._run(weather, 1.0)
        self.assertIsNotNone(on["weather"])
        self.assertGreater(on["avg_stat"], 1.0)
        expected_mult = props._weather_factor_mult("batter_hits", weather, 1.0)[0]
        self.assertAlmostEqual(on["avg_stat"], round(expected_mult, 2), places=2)

    def test_no_weather_data_is_neutral(self):
        on = self._run(None, 1.0)
        self.assertIsNone(on["weather"])
        self.assertAlmostEqual(on["avg_stat"], 1.0, places=2)


class WeatherFetchTests(unittest.TestCase):
    """P1.3: weather_factors.get_game_weather (hermetic — requests patched)."""

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    _PAYLOAD = {"hourly": {
        "time": ["2026-07-24T19:00", "2026-07-24T20:00", "2026-07-24T21:00"],
        "temperature_2m": [70, 80, 90],
        "wind_speed_10m": [5, 10, 15],
        "wind_direction_10m": [180, 180, 180],
    }}

    def test_nearest_hour_and_wind_out(self):
        with patch("weather_factors.requests.get",
                   return_value=self._Resp(self._PAYLOAD)) as mock_get:
            # Coors (cf_bearing≈2, roof open); first pitch 20:10Z → 20:00 hour.
            w = weather_factors.get_game_weather(
                "Colorado Rockies", "2026-07-24T20:10:00Z", use_cache=False)
        self.assertTrue(mock_get.called)
        self.assertEqual(w["temp_f"], 80)
        self.assertEqual(w["wind_mph"], 10)
        self.assertFalse(w["dome"])
        # Wind from due south into a ~north-facing CF → ~full 10 mph blowing out.
        self.assertAlmostEqual(w["wind_out_mph"], 10.0, places=1)

    def test_dome_skips_fetch(self):
        with patch.object(weather_factors, "park_geo",
                          return_value={"lat": 0.0, "lon": 0.0,
                                        "cf_bearing": 0, "roof": "dome"}), \
             patch("weather_factors.requests.get") as mock_get:
            w = weather_factors.get_game_weather(
                "Domed Team", "2026-07-24T20:10:00Z", use_cache=False)
        self.assertTrue(w["dome"])
        self.assertIsNone(w["wind_out_mph"])
        self.assertFalse(mock_get.called)

    def test_unknown_park_skips_fetch(self):
        with patch("weather_factors.requests.get") as mock_get:
            w = weather_factors.get_game_weather(
                "Nowhere FC", "2026-07-24T20:10:00Z", use_cache=False)
        self.assertFalse(mock_get.called)
        self.assertIsNone(w["temp_f"])
        self.assertFalse(w["dome"])

    def test_network_error_fails_open(self):
        with patch("weather_factors.requests.get",
                   side_effect=requests.RequestException("boom")):
            w = weather_factors.get_game_weather(
                "Colorado Rockies", "2026-07-24T20:10:00Z", use_cache=False)
        self.assertIsNone(w["temp_f"])
        self.assertIsNone(w["wind_out_mph"])
        self.assertFalse(w["dome"])


if __name__ == "__main__":
    unittest.main()
