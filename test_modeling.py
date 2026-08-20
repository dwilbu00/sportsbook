"""Focused regression tests for sportsbook model correctness boundaries."""

import json
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
import pricing_common
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

    def test_batter_unresolvable_name_not_ruled_out(self):
        # Both lineups posted and he's absent by name, but his id can't be
        # resolved (find_player_id -> None, e.g. a StatsAPI index outage that
        # resolve_mlbam_id swallows to None) -> stay "unknown", never a false
        # "out" that would demote a valid bet AND drop its calibration label.
        # Symmetric with test_pitcher_unresolvable_name_not_ruled_out.
        lineup = self._lineup([f"H{i}" for i in range(1, 10)],
                              [f"A{i}" for i in range(1, 10)])
        with patch.object(mlb_starters, "find_player_id", return_value=None):
            self.assertEqual(mlb_starters.player_start_status(
                "batter_hits", "Ghost Batter", "Guardians", "Tigers",
                lineup, {}, season=2025), "unknown")

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
                side_effect=lambda pitcher_id, season, as_of_date=None: qualities[pitcher_id]), patch.object(
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
        # Isolate from the live prob_shrink value: analyze_spreads_value shrinks the
        # cover via _apply_shrink (NOT _shrink_factor), so pin _apply_shrink to a
        # deterministic 0.25 pull-to-0.5 — the value the assertions below assume. This
        # keeps the test about the ENSEMBLE math, independent of calibration refits
        # (e.g. the spreads-shrink 0.25 -> 0.6 promote that used to break it).
        with patch.object(
                analysis, "load_expected_runs_challenger",
                return_value=self._calibration()), patch.object(
                analysis, "_apply_shrink",
                side_effect=lambda p, sk, mk: 0.5 + 0.25 * (p - 0.5)), patch.object(
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

    # ── Pythagorean strength (WIRING ONLY, inert by default) ──────────────────
    def _stats_with_runs(self):
        home, away = self._team_stats()
        home["season"] = dict(home["season"])   # unshare the nested season dict
        away["season"] = dict(away["season"])
        home["season"].update(runs_scored=600, runs_allowed=450)   # strong
        away["season"].update(runs_scored=450, runs_allowed=600)   # weak
        return home, away

    def _ml_by_team(self, home_stats, away_stats):
        with patch.object(analysis, "_blend_weight", return_value=1.0):
            cands = analysis.analyze_moneyline_value(
                self._game_odds(), home_stats, away_stats, sport_key="baseball_mlb")
        return {c["team"]: c for c in cands}

    def test_pythag_exposed_and_blended_by_default(self):
        with_runs = self._ml_by_team(*self._stats_with_runs())
        no_runs = self._ml_by_team(*self._team_stats())
        # exposed from the warehouse season block; None on the ESPN (run-less) block
        self.assertIsNotNone(with_runs["Home"]["pythag_win_pct"])
        self.assertGreater(with_runs["Home"]["pythag_win_pct"], 50.0)  # strong team
        self.assertIsNone(no_runs["Home"]["pythag_win_pct"])
        # ACTIVE at the default weight: the run-differential Pythagorean now pulls
        # the model probability toward the Pythagorean win% (blend, not replace).
        # The run-less fixture is otherwise identical, so it isolates the blend.
        # model_prob and pythag_win_pct are both percentages (0-100).
        base = no_runs["Home"]["model_prob"]
        pythag = with_runs["Home"]["pythag_win_pct"]
        w = analysis.DEFAULT_PYTHAG_WEIGHT
        self.assertGreater(w, 0.0)  # activated
        # delta covers the 2-dp rounding of base/pythag/model_prob in the output.
        self.assertAlmostEqual(with_runs["Home"]["model_prob"],
                               (1.0 - w) * base + w * pythag, delta=0.02)
        self.assertNotEqual(with_runs["Home"]["model_prob"], base)

    def test_pythag_blends_when_weighted(self):
        with patch.object(analysis, "DEFAULT_PYTHAG_WEIGHT", 1.0):
            home = self._ml_by_team(*self._stats_with_runs())["Home"]
        # full weight → the model prob IS the pythagorean win%
        self.assertEqual(home["model_prob"], home["pythag_win_pct"])

    def test_pythag_skipped_without_runs_even_when_weighted(self):
        weighted = None
        with patch.object(analysis, "DEFAULT_PYTHAG_WEIGHT", 1.0):
            weighted = self._ml_by_team(*self._team_stats())
        baseline = self._ml_by_team(*self._team_stats())
        # no runs on the block → weight has no effect (pythag skipped)
        self.assertEqual(weighted["Home"]["model_prob"],
                         baseline["Home"]["model_prob"])

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


class AdditiveTotalsTests(unittest.TestCase):
    """Tier B: ODI_MLB_ADDITIVE_TOTALS runs-first totals seam in
    analyze_totals_value. Mirrors the #1d spreads substitution — replace the
    recency + starter-shift projection with the additive expected TOTAL, keep
    the Normal-CDF + shrink downstream. Inert (byte-identical) when the flag is
    off; falls back to the recency projection when the additive returns None."""

    @staticmethod
    def _team_stats():
        # recent_games carry total_score (totals model) + per-team scores
        # (recency scoring means). Same list for both teams, mirroring
        # ExpectedRunsTests._team_stats.
        games = [
            {"home_team": "Home", "away_team": "Away",
             "home_score": 6, "away_score": 3, "total_score": 9},
            {"home_team": "Away", "away_team": "Home",
             "home_score": 2, "away_score": 4, "total_score": 6},
            {"home_team": "Home", "away_team": "Away",
             "home_score": 1, "away_score": 5, "total_score": 6},
            {"home_team": "Away", "away_team": "Home",
             "home_score": 3, "away_score": 2, "total_score": 5},
        ]
        base = {
            "season": {"win_pct": 0.5},
            "recent": {"win_pct": 0.5, "avg_scored": 4.0, "avg_allowed": 4.0},
            "recent_games": games,
        }
        return dict(base), dict(base)

    @staticmethod
    def _game_odds():
        return {
            "home_team": "Home",
            "away_team": "Away",
            "totals": {
                "Over": [{"line": 8.5, "price": -110}],
                "Under": [{"line": 8.5, "price": -110}],
            },
        }

    @staticmethod
    def _matchup_features():
        # Truthy so the seam's `and matchup_features` guard passes; the
        # expected_runs payload is only forwarded to (mocked) live_additive_runs.
        return {"expected_runs": {"complete": True}}

    def test_additive_totals_inert_when_flag_off(self):
        # Flag OFF (default): the additive must never be consulted, and the
        # projection must equal the pure recency + starter-shift baseline even
        # if live_additive_runs would return something wildly different.
        game_odds = self._game_odds()
        home_stats, away_stats = self._team_stats()
        features = self._matchup_features()
        with patch.object(mlb_starters, "_mlb_additive_totals_enabled",
                          return_value=False), \
                patch.object(mlb_starters, "live_additive_runs",
                             return_value=(100.0, 100.0)) as live:
            candidates = analysis.analyze_totals_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb", matchup_features=features)
        over = next(c for c in candidates if c["type"] == "total_over")
        live.assert_not_called()
        # Recompute the baseline with the additive fully absent to confirm
        # byte-identical projection.
        with patch.object(mlb_starters, "_mlb_additive_totals_enabled",
                          return_value=False):
            baseline = analysis.analyze_totals_value(
                self._game_odds(), *self._team_stats(),
                sport_key="baseball_mlb",
                matchup_features=self._matchup_features())
        self.assertEqual(over, next(
            c for c in baseline if c["type"] == "total_over"))

    def test_additive_totals_overrides_projection_when_enabled(self):
        # Flag ON + additive fires: projected_total becomes home+away additive
        # runs (a value distinct from the recency baseline), and the higher
        # total pushes model_over_hit_rate up.
        game_odds = self._game_odds()
        home_stats, away_stats = self._team_stats()
        features = self._matchup_features()
        with patch.object(mlb_starters, "_mlb_additive_totals_enabled",
                          return_value=True), \
                patch.object(mlb_starters, "live_additive_runs",
                             return_value=(7.25, 6.75)) as live:
            candidates = analysis.analyze_totals_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb", matchup_features=features)
        live.assert_called_once()
        over = next(c for c in candidates if c["type"] == "total_over")
        self.assertEqual(over["projected_total"], round(7.25 + 6.75, 2))

        # Baseline (flag off) projects well below the additive 14.0 total, so the
        # additive must raise the modeled over probability.
        with patch.object(mlb_starters, "_mlb_additive_totals_enabled",
                          return_value=False):
            baseline = next(
                c for c in analysis.analyze_totals_value(
                    self._game_odds(), *self._team_stats(),
                    sport_key="baseball_mlb",
                    matchup_features=self._matchup_features())
                if c["type"] == "total_over")
        self.assertLess(baseline["projected_total"], over["projected_total"])
        self.assertGreater(over["model_over_hit_rate"],
                           baseline["model_over_hit_rate"])

    def test_additive_totals_none_falls_back_to_recency(self):
        # Flag ON but the additive returns None (non-MLB / thin data): the seam
        # must fall through to the recency projection, byte-identical to off.
        game_odds = self._game_odds()
        home_stats, away_stats = self._team_stats()
        features = self._matchup_features()
        with patch.object(mlb_starters, "_mlb_additive_totals_enabled",
                          return_value=True), \
                patch.object(mlb_starters, "live_additive_runs",
                             return_value=None) as live:
            candidates = analysis.analyze_totals_value(
                game_odds, home_stats, away_stats,
                sport_key="baseball_mlb", matchup_features=features)
        live.assert_called_once()
        over = next(c for c in candidates if c["type"] == "total_over")
        with patch.object(mlb_starters, "_mlb_additive_totals_enabled",
                          return_value=False):
            baseline = next(
                c for c in analysis.analyze_totals_value(
                    self._game_odds(), *self._team_stats(),
                    sport_key="baseball_mlb",
                    matchup_features=self._matchup_features())
                if c["type"] == "total_over")
        self.assertEqual(over, baseline)

    def test_additive_totals_flag_helper_reads_env(self):
        with patch.dict(os.environ, {"ODI_MLB_ADDITIVE_TOTALS": "1"}):
            self.assertTrue(mlb_starters._mlb_additive_totals_enabled())
        with patch.dict(os.environ, {"ODI_MLB_ADDITIVE_TOTALS": "off"}):
            self.assertFalse(mlb_starters._mlb_additive_totals_enabled())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(mlb_starters._mlb_additive_totals_enabled())


class TeamMarketSuppressTests(unittest.TestCase):
    """value_gate.suppress must gate TEAM markets, not just props: a suppressed
    market never flags a value bet, so it can't enter bet selection / the top-N
    picker. Totals is the shipped case (loses at volume, badly overconfident);
    the same guard covers moneyline/spreads. See _market_suppressed +
    analyze_*_value."""

    def setUp(self):
        # _VALUE_GATE_CACHE is a module-level dict in pricing_common; snapshot
        # and restore so seeding here never leaks into other tests.
        self._orig = dict(pricing_common._VALUE_GATE_CACHE)

    def tearDown(self):
        pricing_common._VALUE_GATE_CACHE.clear()
        pricing_common._VALUE_GATE_CACHE.update(self._orig)

    def _seed(self, suppress):
        pricing_common._VALUE_GATE_CACHE["baseball_mlb"] = {"suppress": suppress}

    def test_market_suppressed_membership_and_fail_open(self):
        self._seed(["totals", "pitcher_outs"])
        self.assertTrue(pricing_common._market_suppressed("baseball_mlb", "totals"))
        self.assertFalse(pricing_common._market_suppressed("baseball_mlb", "moneyline"))
        self.assertFalse(pricing_common._market_suppressed("baseball_mlb", "spreads"))
        # Fail OPEN on empty inputs — a config miss must never blank the card.
        self.assertFalse(pricing_common._market_suppressed("", "totals"))
        self.assertFalse(pricing_common._market_suppressed("baseball_mlb", ""))

    def test_suppressed_totals_never_flags_value(self):
        # Force the underlying value gate True so the ONLY thing that can zero the
        # flag is the suppression guard; additive-on gives a clean over (diff>0).
        def _over():
            with patch("analysis._prop_is_value", return_value=True), \
                    patch.object(mlb_starters, "_mlb_additive_totals_enabled",
                                 return_value=True), \
                    patch.object(mlb_starters, "live_additive_runs",
                                 return_value=(7.25, 6.75)):  # 14.0 > 8.5 line
                cands = analysis.analyze_totals_value(
                    AdditiveTotalsTests._game_odds(),
                    *AdditiveTotalsTests._team_stats(),
                    sport_key="baseball_mlb",
                    matchup_features=AdditiveTotalsTests._matchup_features())
            return next(c for c in cands if c["type"] == "total_over")

        self._seed([])                       # not suppressed -> flag CAN be True
        self.assertTrue(_over()["is_over_value"])
        self._seed(["totals"])               # suppressed -> forced False
        self.assertFalse(_over()["is_over_value"])
        self.assertFalse(_over()["is_under_value"])


class AdditiveMoneylineTests(unittest.TestCase):
    """Tier B: ODI_MLB_ADDITIVE_ML runs-first moneyline seam in
    analyze_moneyline_value. When on and the additive fires, P(home win) comes
    from the additive expected runs (symmetric Poisson margin at 0) as the base
    win prob; the season-pythag blend + shrink apply downstream. Inert (byte-
    identical) when off; falls back to the recency margin model on None."""

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
            "season": {"win_pct": 0.5, "runs_scored": 700, "runs_allowed": 680},
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
                "Home": [{"implied_prob": 0.5, "price": 100, "book": "Test"}],
                "Away": [{"implied_prob": 0.5, "price": 100, "book": "Test"}],
            },
        }

    @staticmethod
    def _matchup_features():
        return {"starter_edge": 0.0, "expected_runs": {"complete": True}}

    def _baseline(self):
        with patch.object(mlb_starters, "_mlb_additive_ml_enabled",
                          return_value=False), \
                patch.object(analysis, "_blend_weight", return_value=1.0):
            return analysis.analyze_moneyline_value(
                self._game_odds(), *self._team_stats(),
                sport_key="baseball_mlb",
                matchup_features=self._matchup_features())

    def test_additive_ml_inert_when_flag_off(self):
        with patch.object(mlb_starters, "_mlb_additive_ml_enabled",
                          return_value=False), \
                patch.object(mlb_starters, "live_additive_runs",
                             return_value=(9.0, 1.0)) as live, \
                patch.object(analysis, "_blend_weight", return_value=1.0):
            candidates = analysis.analyze_moneyline_value(
                self._game_odds(), *self._team_stats(),
                sport_key="baseball_mlb",
                matchup_features=self._matchup_features())
        live.assert_not_called()
        self.assertEqual(candidates, self._baseline())

    def test_additive_ml_overrides_win_prob_when_enabled(self):
        # Pin the pythag blend off so model_prob == the base (additive) win prob.
        with patch.object(mlb_starters, "_mlb_additive_ml_enabled",
                          return_value=True), \
                patch.object(mlb_starters, "live_additive_runs",
                             return_value=(6.5, 3.5)) as live, \
                patch.object(analysis, "DEFAULT_PYTHAG_WEIGHT", 0.0), \
                patch.object(analysis, "_blend_weight", return_value=1.0):
            candidates = analysis.analyze_moneyline_value(
                self._game_odds(), *self._team_stats(),
                sport_key="baseball_mlb",
                matchup_features=self._matchup_features())
        live.assert_called()
        p_h = mlb_starters.poisson_margin_probability(6.5, 3.5, 0.0)
        p_a = mlb_starters.poisson_margin_probability(3.5, 6.5, 0.0)
        expected_home = 0.5 * (p_h + (1.0 - p_a))
        home = next(c for c in candidates if c["home_away"] == "HOME")
        away = next(c for c in candidates if c["home_away"] == "AWAY")
        self.assertEqual(home["model_prob"], round(expected_home * 100, 2))
        self.assertEqual(away["model_prob"],
                         round((1.0 - expected_home) * 100, 2))
        # Home is the clear favorite by runs -> above the 50% baseline.
        self.assertGreater(home["model_prob"], 50.0)

    def test_additive_ml_none_falls_back_to_recency(self):
        with patch.object(mlb_starters, "_mlb_additive_ml_enabled",
                          return_value=True), \
                patch.object(mlb_starters, "live_additive_runs",
                             return_value=None) as live, \
                patch.object(analysis, "_blend_weight", return_value=1.0):
            candidates = analysis.analyze_moneyline_value(
                self._game_odds(), *self._team_stats(),
                sport_key="baseball_mlb",
                matchup_features=self._matchup_features())
        live.assert_called()
        self.assertEqual(candidates, self._baseline())

    def test_additive_ml_non_mlb_untouched(self):
        # Flag on but a non-MLB sport: live_additive_runs returns None internally,
        # so the seam is inert and the NBA path is unchanged.
        with patch.object(mlb_starters, "_mlb_additive_ml_enabled",
                          return_value=True), \
                patch.object(analysis, "_blend_weight", return_value=1.0):
            nba = analysis.analyze_moneyline_value(
                self._game_odds(), *self._team_stats(),
                sport_key="basketball_nba",
                matchup_features=self._matchup_features())
        with patch.object(mlb_starters, "_mlb_additive_ml_enabled",
                          return_value=False), \
                patch.object(analysis, "_blend_weight", return_value=1.0):
            nba_baseline = analysis.analyze_moneyline_value(
                self._game_odds(), *self._team_stats(),
                sport_key="basketball_nba",
                matchup_features=self._matchup_features())
        self.assertEqual(nba, nba_baseline)

    def test_additive_ml_flag_helper_reads_env(self):
        with patch.dict(os.environ, {"ODI_MLB_ADDITIVE_ML": "yes"}):
            self.assertTrue(mlb_starters._mlb_additive_ml_enabled())
        with patch.dict(os.environ, {"ODI_MLB_ADDITIVE_ML": "0"}):
            self.assertFalse(mlb_starters._mlb_additive_ml_enabled())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(mlb_starters._mlb_additive_ml_enabled())

    def test_any_additive_enabled_covers_each_flag(self):
        # build_matchup_features surfaces the live-additive keys under this gate; it must
        # trigger on ANY single additive flag, incl. ML alone (the fixed gap where an
        # ML-only run left live_additive_runs starved of ids -> silently inert).
        for flag in ("ODI_MLB_ADDITIVE_RUNS", "ODI_MLB_ADDITIVE_TOTALS",
                     "ODI_MLB_ADDITIVE_ML"):
            with patch.dict(os.environ, {flag: "1"}, clear=True):
                self.assertTrue(mlb_starters._any_additive_enabled(),
                                f"{flag} alone must trigger surfacing")
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(mlb_starters._any_additive_enabled())


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
        # A None last-fit forces the refit branch; compact_prediction_log is mocked
        # out so nothing touches the durable store (secrets.toml is present locally).
        with patch.object(
                recalibration, "resolve_pending_outcomes", return_value=25), patch.object(
                recalibration, "resolve_pending_market_outcomes", return_value=0), patch.object(
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
        # NBA/NFL/NHL grading refreshes the ESPN gamelog (ttl_hours=6). MLB is
        # warehouse+statsapi only in P6 and never reaches this ESPN path, so the
        # refresh behavior is exercised via basketball.
        rows = [{
            "ts": "2024-04-01T10:00:00Z",
            "sport_key": "basketball_nba",
            "prop_key": "player_points",
            "player": "Player One",
            "game_date": "2024-04-01",
            "line": 0.5,
            "resolved": False,
        }]

        def mutate(mutator, where=None):
            return mutator(rows)

        with patch.object(recalibration, "_read_log", return_value=rows), patch.object(
                recalibration, "_resolve_mlb_actual", return_value=None), patch(
                "espn_cache.cached_athlete_id", return_value="123"), patch(
                "espn_cache.cached_gamelog",
                return_value=[{"game_date": "2024-04-01", "PTS": 1}],
        ) as gamelog, patch.object(
                recalibration, "_stat_label", return_value="PTS"), patch.object(
                recalibration, "mutate_prediction_log", side_effect=mutate):
            resolved = recalibration.resolve_pending_outcomes("basketball_nba")

        self.assertEqual(resolved, 1)
        self.assertTrue(rows[0]["resolved"])
        gamelog.assert_called_once_with(
            "basketball", "nba", "123", ttl_hours=6)

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
                os.path.join(temp_dir, "prediction_log.jsonl")):
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


class DkLineAnchorTests(unittest.TestCase):
    """DK-only bettor: parse_player_props anchors the analyzed line on the line DK
    actually posts (not the cross-book modal), so a DK-bettable leg is never dropped
    when the consensus line differs; falls back to consensus when DK is absent."""

    def _game(self, books):
        return {"id": "g1", "home_team": "H", "away_team": "A",
                "commence_time": "2026-07-20T23:10:00Z",
                "sport_key": "baseball_mlb", "bookmakers": books}

    def _book(self, title, line, over=-110, under=-110):
        return {"title": title, "markets": [{"key": "batter_hits", "outcomes": [
            {"description": "P", "name": "Over", "price": over, "point": line},
            {"description": "P", "name": "Under", "price": under, "point": line}]}]}

    def _info(self, books):
        return parse_player_props(self._game(books))["props"]["batter_hits"]["P"]

    def test_anchors_on_dk_line_when_consensus_differs(self):
        # 3 peers post 0.5 (the modal); DK posts only 1.5. Old behavior analyzed 0.5
        # -> DK price None -> silently dropped. New: analyze DK's 1.5.
        info = self._info([
            self._book("BetMGM", 0.5), self._book("Caesars", 0.5),
            self._book("FanDuel", 0.5), self._book("DraftKings", 1.5)])
        self.assertEqual(info["line"], 1.5)
        self.assertEqual(info["line_source"], "dk")
        self.assertEqual(info["consensus_line"], 0.5)
        self.assertIsNotNone(info["dk_over_price"])       # DK now bettable
        self.assertEqual(info["peer_count"], 0)           # DK alone at 1.5
        self.assertEqual(info["market_implied_method"], "dk_selfdevig_fallback")

    def test_peerconsensus_when_peers_quote_dk_line(self):
        info = self._info([
            self._book("DraftKings", 1.5), self._book("BetMGM", 1.5),
            self._book("Caesars", 1.5), self._book("FanDuel", 1.5)])
        self.assertEqual(info["line"], 1.5)
        self.assertEqual(info["line_source"], "dk")
        self.assertEqual(info["peer_count"], 3)
        self.assertEqual(info["market_implied_method"],
                         "two_way_devig_peerconsensus_at_dk_line")

    def test_falls_back_to_consensus_when_dk_absent(self):
        info = self._info([
            self._book("BetMGM", 0.5), self._book("Caesars", 0.5),
            self._book("FanDuel", 1.5)])
        self.assertEqual(info["line"], 0.5)               # modal, DK absent
        self.assertEqual(info["line_source"], "consensus")
        self.assertIsNone(info["dk_over_price"])          # not bettable
        self.assertEqual(info["market_implied_method"],
                         "two_way_devig_sharpweighted_consensus")

    def test_sharp_peer_lifts_selfdevig_fallback(self):
        # DK + a single SHARP book (Pinnacle) at DK's line: peer_count is only 1, but
        # a sharp book is a real independent check, so it is NOT the self-devig
        # fallback — the raised-edge guard must not fire.
        info = self._info([
            self._book("BetMGM", 0.5), self._book("Caesars", 0.5),
            self._book("DraftKings", 1.5), self._book("Pinnacle", 1.5)])
        self.assertEqual(info["line"], 1.5)
        self.assertEqual(info["line_source"], "dk")
        self.assertEqual(info["peer_count"], 1)
        self.assertEqual(info["market_implied_method"],
                         "two_way_devig_peerconsensus_at_dk_line")


class NewBatterMarketParsingTests(unittest.TestCase):
    """batter_total_bases (line 1.5) and batter_rbis (line 0.5) parse through the
    generic parse_player_props once they're in PROP_LABELS — this guards those
    label entries (a missing label silently drops the market at parse)."""

    def _game(self, market_key, point):
        outcomes = [
            {"description": "Slugger Sam", "name": "Over",
             "price": -110, "point": point},
            {"description": "Slugger Sam", "name": "Under",
             "price": -110, "point": point},
        ]
        return {
            "id": "g1", "home_team": "H", "away_team": "A",
            "commence_time": "2026-07-20T23:10:00Z", "sport_key": "baseball_mlb",
            "bookmakers": [{"title": "DraftKings",
                            "markets": [{"key": market_key, "outcomes": outcomes}]}],
        }

    def test_total_bases_parses(self):
        out = parse_player_props(self._game("batter_total_bases", 1.5))["props"]
        self.assertIn("batter_total_bases", out)
        info = out["batter_total_bases"]["Slugger Sam"]
        self.assertEqual(info["line"], 1.5)
        self.assertEqual(info["dk_over_price"], -110)

    def test_rbis_parses(self):
        out = parse_player_props(self._game("batter_rbis", 0.5))["props"]
        self.assertIn("batter_rbis", out)
        self.assertEqual(out["batter_rbis"]["Slugger Sam"]["line"], 0.5)


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


class AirDensityTests(unittest.TestCase):
    """Moist-air density helper (weather_factors.air_density) for the weather feature:
    the physics combiner (temp+humidity+pressure). Denser air (cold/humid/high-pressure)
    suppresses batted-ball carry; thinner air boosts it; graceful on missing data."""

    def test_baseline_and_monotonicity(self):
        b = weather_factors.AIR_DENSITY_BASELINE_KG_M3
        self.assertAlmostEqual(weather_factors.air_density(70, 50, 1013.25), b, places=3)
        self.assertGreater(weather_factors.air_density(40, 90, 1030), b)   # cold+humid+high = denser
        self.assertLess(weather_factors.air_density(95, 20, 1000), b)      # hot+dry+low = thinner
        self.assertLess(weather_factors.air_density(95),
                        weather_factors.air_density(50))                   # hotter -> thinner

    def test_missing_inputs(self):
        b = weather_factors.AIR_DENSITY_BASELINE_KG_M3
        self.assertAlmostEqual(weather_factors.air_density(70), b, places=3)  # humidity/pressure default
        self.assertIsNone(weather_factors.air_density(None))
        self.assertIsNone(weather_factors.air_density("x", None, None))


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


class VisualCrossingHistoricalTests(unittest.TestCase):
    """Batch A weather data layer: the Visual Crossing historical fetch + hour pick."""

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    _DAILY = {"days": [{"datetime": "2024-06-17", "temp": 74, "humidity": 55,
                        "pressure": 1015, "windspeed": 9, "winddir": 210}]}
    _HOURLY = {"days": [{"datetime": "2024-06-17", "hours": [
        {"datetimeEpoch": 1000, "temp": 70, "humidity": 50, "pressure": 1015,
         "windspeed": 8, "winddir": 180},
        {"datetimeEpoch": 5000, "temp": 78, "humidity": 44, "pressure": 1013,
         "windspeed": 12, "winddir": 200}]}]}

    def test_fetch_daily_default(self):
        with patch.dict(os.environ, {"WEATHER_API_KEY": "k"}), \
                patch("weather_factors.requests.get",
                      return_value=self._Resp(self._DAILY)) as mock_get:
            out = weather_factors.fetch_visualcrossing_range(40.0, -75.0, "2024-06-17")
        self.assertTrue(mock_get.called)
        w = out["2024-06-17"]                            # daily -> one dict per date
        self.assertEqual((w["temp_f"], w["wind_mph"], w["pressure_mb"]),
                         (74, 9, 1015))
        self.assertEqual(mock_get.call_args.kwargs["params"]["include"], "days")

    def test_fetch_hourly_when_requested(self):
        with patch.dict(os.environ, {"WEATHER_API_KEY": "k"}), \
                patch("weather_factors.requests.get",
                      return_value=self._Resp(self._HOURLY)) as mock_get:
            out = weather_factors.fetch_visualcrossing_range(
                40.0, -75.0, "2024-06-17", hourly=True)
        self.assertEqual(len(out["2024-06-17"]), 2)      # hourly -> list per date
        h0 = out["2024-06-17"][0]
        self.assertEqual((h0["epoch"], h0["temp_f"], h0["wind_mph"]), (1000, 70, 8))
        self.assertEqual(mock_get.call_args.kwargs["params"]["include"], "hours")

    def test_fetch_without_key_is_empty_no_call(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("weather_factors.requests.get") as mock_get:
            out = weather_factors.fetch_visualcrossing_range(40.0, -75.0, "2024-06-17")
        self.assertEqual(out, {})
        self.assertFalse(mock_get.called)               # no key -> never hits the API

    def test_fetch_fails_open_on_error(self):
        with patch.dict(os.environ, {"WEATHER_API_KEY": "k"}), \
                patch("weather_factors.requests.get", side_effect=RuntimeError("boom")):
            self.assertEqual(
                weather_factors.fetch_visualcrossing_range(40, -75, "2024-06-17"), {})

    def test_pick_hour_by_epoch_nearest(self):
        hours = [{"epoch": 1000, "temp_f": 70}, {"epoch": 5000, "temp_f": 78}]
        self.assertEqual(weather_factors.pick_hour_by_epoch(hours, 4800)["temp_f"], 78)
        self.assertEqual(weather_factors.pick_hour_by_epoch(hours, 1200)["temp_f"], 70)
        self.assertIsNone(weather_factors.pick_hour_by_epoch([], 1000))
        self.assertIsNone(weather_factors.pick_hour_by_epoch(hours, None))


class WeatherRunEnvPhysicsTests(unittest.TestCase):
    """Batch A weather run_env: run_env_from_weather is a BASELINE-RELATIVE, centered-
    on-1.0 deviation (no double-count with the park factor's structural climate)."""

    def test_at_baseline_is_neutral(self):
        self.assertAlmostEqual(
            weather_factors.run_env_from_weather(75, 5, 75, 5), 1.0)

    def test_warmer_than_baseline_raises_runs(self):
        c_temp = weather_factors.WEATHER_RUN_ENV_COEF[0]
        self.assertAlmostEqual(
            weather_factors.run_env_from_weather(85, 5, 75, 5), 1 + c_temp * 10)

    def test_wind_out_over_baseline_raises_runs(self):
        c_wind = weather_factors.WEATHER_RUN_ENV_COEF[1]
        self.assertAlmostEqual(
            weather_factors.run_env_from_weather(75, 10, 75, 5), 1 + c_wind * 5)

    def test_colder_and_wind_in_lowers_runs(self):
        self.assertLess(weather_factors.run_env_from_weather(60, -8, 75, 4), 1.0)

    def test_missing_inputs_are_neutral(self):
        self.assertAlmostEqual(
            weather_factors.run_env_from_weather(None, None, 75, 5), 1.0)
        self.assertAlmostEqual(
            weather_factors.run_env_from_weather(85, 10, None, None), 1.0)


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


class ServeModeGradingTests(unittest.TestCase):
    """Gap A: the odds backtest's serve-mode grades the ALREADY-served probs
    (per-market prob_shrink + model<->market blend applied inside the analyzers),
    so a holdout grades exactly what production would serve. Raw mode (default)
    keeps returning the pre-shrink/pre-blend model prob for the shrink FIT."""

    def _fake_analyzers(self):
        # Distinct raw vs served values so the field selection is unambiguous.
        ml = [{"home_away": "HOME", "model_prob": 70.0, "blended_prob": 58.0,
               "pythag_win_pct": None}]
        sp = [{"home_away": "HOME", "spread": -1.5,
               "model_cover_rate": 65.0, "cover_rate": 55.0}]
        tot = [{"line": 8.5, "model_over_hit_rate": 62.0, "over_hit_rate": 53.0}]
        return ml, sp, tot

    def test_raw_mode_returns_pure_model_fields(self):
        import backtest
        ml, sp, tot = self._fake_analyzers()
        with patch("backtest._live_stats", lambda *a, **k: {}), \
             patch("backtest.analyze_moneyline_value", lambda *a, **k: ml), \
             patch("backtest.analyze_spreads_value", lambda *a, **k: sp), \
             patch("backtest.analyze_totals_value", lambda *a, **k: tot):
            hw, hc, ov, _, _sc = backtest._live_spread_total_probs(
                {}, [], [], 5.0, "baseball_mlb", serve_mode=False)
        self.assertAlmostEqual(hw, 0.70)
        self.assertAlmostEqual(hc[1], 0.65)
        self.assertAlmostEqual(ov[1], 0.62)

    def test_serve_mode_returns_served_fields(self):
        import backtest
        ml, sp, tot = self._fake_analyzers()
        with patch("backtest._live_stats", lambda *a, **k: {}), \
             patch("backtest.analyze_moneyline_value", lambda *a, **k: ml), \
             patch("backtest.analyze_spreads_value", lambda *a, **k: sp), \
             patch("backtest.analyze_totals_value", lambda *a, **k: tot):
            hw, hc, ov, _, _sc = backtest._live_spread_total_probs(
                {}, [], [], 5.0, "baseball_mlb", serve_mode=True)
        self.assertAlmostEqual(hw, 0.58)
        self.assertAlmostEqual(hc[1], 0.55)
        self.assertAlmostEqual(ov[1], 0.53)

    def test_serve_mode_rejects_calibration_fit(self):
        import backtest
        with self.assertRaises(ValueError):
            backtest._run_odds_backtest_impl(
                "baseball_mlb", "baseball", "mlb", 10, 20, {},
                serve_mode=True, write_calibration=True)

    def test_serve_mode_rejects_collect_obs(self):
        import backtest
        with self.assertRaises(ValueError):
            backtest._run_odds_backtest_impl(
                "baseball_mlb", "baseball", "mlb", 10, 20, {},
                serve_mode=True, collect_obs={"moneyline": []})

    def test_serve_mode_requires_live_engine(self):
        import backtest
        with self.assertRaises(ValueError):
            backtest._run_odds_backtest_impl(
                "baseball_mlb", "baseball", "mlb", 10, 20, {},
                serve_mode=True, engine="convolution")


class LiveBlendWriteTests(unittest.TestCase):
    """Gap C: fit + persist the model<->market blend on the LIVE model (not the
    retired convolution engine), ON TOP of the fitted prob_shrink — so one raw
    --write-calibration pass produces both corrections in serve order (shrink
    then blend)."""

    def _bucket(self, obs):
        return {"blend": list(obs), "n": len(obs)}

    def _results(self, ml_obs):
        return {"live": {"moneyline": self._bucket(ml_obs),
                         "spreads": self._bucket([]),
                         "totals": self._bucket([])}}

    def test_shrink_map_transforms_obs_before_fit(self):
        import backtest
        raw = [(0.90, 0.55, 1), (0.90, 0.55, 0), (0.10, 0.45, 0)]
        results = self._results(raw)
        captured = []

        def fake_bbw(obs, step=0.05):
            if not obs:
                return None
            captured.append(list(obs))
            return (0.5, 0.10, 0.20, 0.15)

        with patch("backtest._best_blend_weight", fake_bbw), \
             patch("backtest.save_market_blend"):
            backtest._write_blend_calibration(
                "baseball_mlb", results, shrink_map={"moneyline": 0.5}, min_n=0)
        self.assertEqual(len(captured), 1)  # only the non-empty market
        expected = [(backtest._shrink_prob(pm, 0.5), mk, o) for pm, mk, o in raw]
        for got, exp in zip(captured[0], expected):
            self.assertAlmostEqual(got[0], exp[0])
            self.assertAlmostEqual(got[1], exp[1])
            self.assertEqual(got[2], exp[2])

    def test_no_shrink_map_fits_raw(self):
        import backtest
        raw = [(0.90, 0.55, 1), (0.10, 0.45, 0)]
        results = self._results(raw)
        captured = []

        def fake_bbw(obs, step=0.05):
            if not obs:
                return None
            captured.append(list(obs))
            return (0.5, 0.10, 0.20, 0.15)

        with patch("backtest._best_blend_weight", fake_bbw), \
             patch("backtest.save_market_blend"):
            backtest._write_blend_calibration("baseball_mlb", results, min_n=0)
        self.assertEqual([tuple(x) for x in captured[0]], raw)  # untransformed

    def test_min_n_withholds_thin_sample(self):
        import backtest
        results = self._results([(0.90, 0.55, 1)] * 5)
        with patch("backtest._best_blend_weight",
                   lambda obs, step=0.05: (0.2, 0.10, 0.20, 0.15)), \
             patch("backtest.save_market_blend") as save:
            backtest._write_blend_calibration("baseball_mlb", results, min_n=1000)
        self.assertFalse(save.called)  # thin -> nothing persisted

    def test_min_n_persists_when_sample_sufficient(self):
        import backtest
        results = self._results([(0.90, 0.55, 1)] * 5)
        with patch("backtest._best_blend_weight",
                   lambda obs, step=0.05: (0.2, 0.10, 0.20, 0.15)), \
             patch("backtest.save_market_blend") as save:
            backtest._write_blend_calibration("baseball_mlb", results, min_n=1)
        self.assertTrue(save.called)
        blend = save.call_args[0][1]
        self.assertEqual(blend["moneyline"]["w"], 0.2)
        self.assertTrue(blend["moneyline"]["on_shrunk"] is False)

    def test_chained_shrink_then_blend_writes_both_blocks(self):
        import backtest
        import calibration_loader as cl
        orig_dir = cl.CALIBRATION_DIR
        tmp = tempfile.mkdtemp()
        cl.CALIBRATION_DIR = tmp
        try:
            # Model is a constant 0.85 (no discrimination); the market perfectly
            # separates two subpopulations. Shrink calibrates the level; the blend
            # then captures the discrimination the scalar shrink can't.
            obs = ([(0.85, 0.80, 1)] * 120 + [(0.85, 0.80, 0)] * 30
                   + [(0.85, 0.40, 1)] * 60 + [(0.85, 0.40, 0)] * 90)
            results = self._results(obs)
            fitted = backtest._write_shrink_calibration(
                "baseball_mlb", results, min_shrink_n=1)
            self.assertIn("moneyline", fitted)
            self.assertLess(fitted["moneyline"], 1.0)
            backtest._write_blend_calibration(
                "baseball_mlb", results, shrink_map=fitted, min_n=1)
            with open(cl.calibration_path("baseball_mlb"), encoding="utf-8") as f:
                blob = json.load(f)
            self.assertIn("moneyline", blob["prob_shrink"])
            self.assertIn("moneyline", blob["market_blend"])
            self.assertLess(blob["market_blend"]["moneyline"]["w"], 1.0)
            self.assertTrue(blob["market_blend"]["moneyline"]["on_shrunk"])
            self.assertTrue(blob["meta"]["market_blend"]["on_shrunk_probs"])
        finally:
            cl.CALIBRATION_DIR = orig_dir


class ChallengerShareSaveTests(unittest.TestCase):
    """save_expected_runs_challenger_shares updates ONLY the ensemble blend shares
    and preserves the rest of the challenger block + other calibration blocks."""

    def setUp(self):
        import calibration_loader as cl
        self.cl = cl
        self._dir = tempfile.mkdtemp()
        self._orig = cl.CALIBRATION_DIR
        cl.CALIBRATION_DIR = self._dir

    def tearDown(self):
        self.cl.CALIBRATION_DIR = self._orig

    def test_preserves_model_and_live_markets(self):
        cl = self.cl
        path = cl.calibration_path("baseball_mlb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sport_key": "baseball_mlb", "props": {"x": 1},
                       "expected_runs_challenger": {
                           "enabled": True,
                           "live_markets": {"spreads": True},
                           "final_2025_validation": {
                               "model": {"offense_weight": 0.5},
                               "ensemble_challenger_share": {
                                   "home_minus_1_5": 0.3, "margin": 0.4}}}}, f)
        cl.save_expected_runs_challenger_shares(
            "baseball_mlb", {"home_minus_1_5": 0.7, "margin": 0.6})
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        chal = blob["expected_runs_challenger"]
        self.assertTrue(chal["enabled"])
        self.assertEqual(chal["live_markets"], {"spreads": True})
        self.assertEqual(chal["final_2025_validation"]["model"],
                         {"offense_weight": 0.5})
        self.assertEqual(chal["final_2025_validation"]["ensemble_challenger_share"],
                         {"home_minus_1_5": 0.7, "margin": 0.6})
        self.assertEqual(blob["props"], {"x": 1})

    def test_preserves_sibling_share_keys(self):
        # A sibling key (e.g. a moneyline challenger share) must survive a
        # spreads-only fit (per-key merge, not wholesale replace).
        cl = self.cl
        path = cl.calibration_path("baseball_mlb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"sport_key": "baseball_mlb", "expected_runs_challenger": {
                "final_2025_validation": {"ensemble_challenger_share": {
                    "moneyline": 0.75, "home_minus_1_5": 0.3, "margin": 0.4}}}}, f)
        cl.save_expected_runs_challenger_shares(
            "baseball_mlb", {"home_minus_1_5": 0.7, "margin": 0.6})
        with open(path, encoding="utf-8") as f:
            share = json.load(f)["expected_runs_challenger"][
                "final_2025_validation"]["ensemble_challenger_share"]
        self.assertEqual(share, {"moneyline": 0.75,
                                 "home_minus_1_5": 0.7, "margin": 0.6})


class GenericFitterSkipTests(unittest.TestCase):
    """The generic shrink/blend fitters honor skip_markets (so --fit-shares can own
    the spreads market without the generic fit double-counting its composite cover)."""

    def _results(self, per_market):
        return {"live": {m: {"blend": list(per_market.get(m, [])),
                             "n": len(per_market.get(m, []))} for m in
                         ("moneyline", "spreads", "totals")}}

    def test_shrink_skips_spreads(self):
        import backtest
        # Overconfident obs in every market so shrink WOULD fire if not skipped.
        obs = [(0.9, 0.5, 1)] * 100 + [(0.9, 0.5, 0)] * 100
        results = self._results({"moneyline": obs, "spreads": obs, "totals": obs})
        with patch("backtest.save_prob_shrink") as save:
            backtest._write_shrink_calibration(
                "baseball_mlb", results, min_shrink_n=1, skip_markets={"spreads"})
        written = save.call_args[0][1]
        self.assertIn("moneyline", written)
        self.assertNotIn("spreads", written)

    def test_blend_skips_spreads(self):
        import backtest
        # Overconfident model (0.9) covering only 55% while the market (0.55) is
        # calibrated -> blending toward market helps -> weight WOULD be written if
        # spreads were not skipped.
        obs = [(0.9, 0.55, 1)] * 110 + [(0.9, 0.55, 0)] * 90
        results = self._results({"moneyline": obs, "spreads": obs, "totals": obs})
        with patch("backtest.save_market_blend") as save:
            backtest._write_blend_calibration(
                "baseball_mlb", results, min_n=1, skip_markets={"spreads"})
        written = save.call_args[0][1]
        self.assertIn("moneyline", written)
        self.assertNotIn("spreads", written)


class SharesFitterTests(unittest.TestCase):
    """Gap E: the MLB spreads ensemble fitter recovers the Brier-optimal
    challenger spread_share and writes the full stack (shrink/share/blend) in
    serve order, from RAW components."""

    def setUp(self):
        import calibration_loader as cl
        self.cl = cl
        self._dir = tempfile.mkdtemp()
        self._orig = cl.CALIBRATION_DIR
        cl.CALIBRATION_DIR = self._dir

    def tearDown(self):
        self.cl.CALIBRATION_DIR = self._orig

    def test_recovers_share_when_additive_perfect(self):
        import backtest
        cl = self.cl
        # Recency cover uninformative (0.5); additive cover perfectly separates;
        # display margin equals actual. Optimal share -> 1.0, margin_share -> 1.0,
        # no shrink (0.5 is unshrinkable), no market blend (market uninformative).
        obs = ([(0.5, 0.9, 0.5, 1, 0.0, 2.0, 2.0)] * 100
               + [(0.5, 0.1, 0.5, 0, 0.0, -2.0, -2.0)] * 100)
        wrote = backtest._write_shares_calibration("baseball_mlb", obs, min_n=1)
        self.assertTrue(wrote)
        with open(cl.calibration_path("baseball_mlb"), encoding="utf-8") as f:
            blob = json.load(f)
        share = blob["expected_runs_challenger"]["final_2025_validation"][
            "ensemble_challenger_share"]
        self.assertEqual(share["home_minus_1_5"], 1.0)
        self.assertEqual(share["margin"], 1.0)
        # 0.5 is unshrinkable -> served_s pinned to the no-op 1.0 (fit==serve).
        self.assertEqual(blob["prob_shrink"]["spreads"], 1.0)
        # Market uninformative -> blend pinned to the no-op w=1.0 (not omitted), so
        # an inherited spreads blend can't survive and break fit==serve.
        self.assertEqual(blob["market_blend"]["spreads"]["w"], 1.0)

    def test_inherited_blend_pinned_to_noop(self):
        import backtest
        cl = self.cl
        # A candidate seeded from live carries an inherited spreads blend w=0.6.
        with open(cl.calibration_path("baseball_mlb"), "w", encoding="utf-8") as f:
            json.dump({"sport_key": "baseball_mlb",
                       "market_blend": {"spreads": {"w": 0.6},
                                        "moneyline": {"w": 0.5}}}, f)
        # Data where the market is uninformative -> blend does not beat the model.
        obs = ([(0.5, 0.9, 0.5, 1, 0.0, 2.0, 2.0)] * 100
               + [(0.5, 0.1, 0.5, 0, 0.0, -2.0, -2.0)] * 100)
        backtest._write_shares_calibration("baseball_mlb", obs, min_n=1)
        with open(cl.calibration_path("baseball_mlb"), encoding="utf-8") as f:
            mb = json.load(f)["market_blend"]
        self.assertEqual(mb["spreads"]["w"], 1.0)   # inherited 0.6 neutralized
        self.assertEqual(mb["moneyline"]["w"], 0.5)  # sibling market untouched

    def test_min_n_zero_does_not_crash_on_empty(self):
        import backtest
        # min_n=0 must NOT divide-by-zero on empty obs (max(1,min_n) guard).
        self.assertFalse(
            backtest._write_shares_calibration("baseball_mlb", [], min_n=0))

    def test_shrink_fires_and_persists(self):
        import backtest
        cl = self.cl
        # Overconfident recency cover (0.85, covers 60%); additive uninformative.
        obs = ([(0.85, 0.5, None, 1, 0.0, 0.0, 0.0)] * 120
               + [(0.85, 0.5, None, 0, 0.0, 0.0, 0.0)] * 80)
        backtest._write_shares_calibration("baseball_mlb", obs, min_n=1)
        with open(cl.calibration_path("baseball_mlb"), encoding="utf-8") as f:
            blob = json.load(f)
        self.assertIn("spreads", blob["prob_shrink"])
        self.assertLess(blob["prob_shrink"]["spreads"], 1.0)

    def test_thin_sample_withheld(self):
        import backtest
        cl = self.cl
        obs = [(0.5, 0.9, 0.5, 1, 0.0, 2.0, 2.0)] * 3
        wrote = backtest._write_shares_calibration("baseball_mlb", obs, min_n=1000)
        self.assertFalse(wrote)
        self.assertFalse(os.path.exists(cl.calibration_path("baseball_mlb")))

    def test_fit_matches_serve_formula(self):
        import backtest
        # The fitter's cover formula must equal analysis serve order:
        #   shrink(cc) -> +share*(ec-shrink(cc)).  Cross-check on a hand value.
        cc, ec, s, sig = 0.80, 0.60, 0.50, 0.40
        shrunk = backtest._shrink_prob(cc, s)           # 0.5 + 0.5*(0.8-0.5)=0.65
        model_cover = shrunk + sig * (ec - shrunk)      # 0.65 + 0.4*(-0.05)=0.63
        self.assertAlmostEqual(shrunk, 0.65)
        self.assertAlmostEqual(model_cover, 0.63)


class BankrollSimTests(unittest.TestCase):
    """Batch B1: the chronological bankroll sim (_bankroll_sim) — value-side
    selection (matching _team_gate_tally), the value gate (edge + EV), and
    uncertainty-Kelly abstain. Rows are (date, raw_home_p, fair_home, price_home,
    price_away, home_won, n_eff). shrink=1.0 here so p == raw_home_p (s=1.0 = NO
    shrink; s=0 would collapse to 0.5)."""

    def test_flat_growth_and_drawdown(self):
        import backtest
        # Back HOME (p 0.60 >= fair 0.50) @ +100. bet1 home loses (-> 99, dd 1%),
        # bet2 home wins (-> 100).
        bets = [("2025-04-01", 0.60, 0.50, 100, -120, 0, 100),
                ("2025-04-02", 0.60, 0.50, 100, -120, 1, 100)]
        r = backtest._bankroll_sim(bets, shrink=1.0, edge_gate=0.05, method="flat")
        self.assertEqual(r["n_bets"], 2)
        self.assertAlmostEqual(r["growth_pct"], 0.0, places=6)   # -1 then +1
        self.assertAlmostEqual(r["max_dd_pct"], 1.0, places=6)   # trough 99 vs peak 100

    def test_value_side_is_the_away_dog_when_model_fades_home(self):
        import backtest
        # Model fades home (raw_home 0.40 < fair_home 0.50) -> back AWAY dog @ +130;
        # away wins (home_won 0) -> our away bet wins. Confirms value-side selection.
        bets = [("2025-04-01", 0.40, 0.50, -120, 130, 0, 100)]
        r = backtest._bankroll_sim(bets, shrink=1.0, edge_gate=0.05, method="flat")
        self.assertEqual(r["n_bets"], 1)
        self.assertGreater(r["growth_pct"], 0.0)   # +130 dog hit

    def test_gate_excludes_thin_edge_and_neg_ev(self):
        import backtest
        bets = [
            ("2025-04-01", 0.52, 0.50, 100, -120, 1, 100),   # edge 0.02 < 0.05 skip
            ("2025-04-02", 0.60, 0.50, -200, 150, 1, 100),   # edge 0.10 EV<0 skip
            ("2025-04-03", 0.60, 0.50, 100, -120, 1, 100),   # edge 0.10 +EV placed
        ]
        r = backtest._bankroll_sim(bets, shrink=1.0, edge_gate=0.05, method="flat")
        self.assertEqual(r["n_bets"], 1)

    def test_ukelly_abstains_thin_sample_that_kelly_takes(self):
        import backtest
        # Back home p=0.55 >= fair 0.48 (edge 0.07, +EV @ +100), n_eff=20:
        # prob_low ~0.38 -> EV<0 -> uncertainty-Kelly abstains; plain Kelly bets it.
        bets = [("2025-04-01", 0.55, 0.48, 100, -120, 1, 20)]
        k = backtest._bankroll_sim(bets, shrink=1.0, edge_gate=0.05, method="kelly")
        u = backtest._bankroll_sim(bets, shrink=1.0, edge_gate=0.05,
                                   method="ukelly", z=1.5)
        self.assertEqual(k["n_bets"], 1)
        self.assertEqual(u["n_bets"], 0)   # abstained on uncertainty

    def test_kelly_compounds_more_than_flat_on_wins(self):
        import backtest
        bets = [("2025-04-0%d" % i, 0.60, 0.50, 100, -120, 1, 100)
                for i in range(1, 6)]
        flat = backtest._bankroll_sim(bets, shrink=1.0, method="flat")
        kelly = backtest._bankroll_sim(bets, shrink=1.0, method="kelly")
        self.assertEqual(flat["n_bets"], 5)
        self.assertEqual(kelly["n_bets"], 5)
        self.assertGreater(kelly["growth_pct"], flat["growth_pct"])  # 5%/leg vs 1u


class PropRecentNWindowTests(unittest.TestCase):
    """STEP-1 per-prop history window (recent_n): calibration slices the newest N
    games BEFORE the projection; recent_n=null = full season (matches the sweep),
    absent = per-sport default. Also the accessors + fetch superset."""

    def test_accessors(self):
        self.assertEqual(props._player_prop_recent_n("baseball_mlb"), 20)
        self.assertEqual(props._player_prop_recent_n("basketball_nba"), 10)
        self.assertEqual(props._player_prop_recent_n("americanfootball_nfl"), 8)
        self.assertIsNone(props._player_prop_recent_n(None))       # unknown -> no cap
        self.assertEqual(props.prop_fetch_limit("baseball_mlb"), 200)
        self.assertEqual(props.prop_fetch_limit(None), 100)

    @staticmethod
    def _project(calib):
        # 30 games, newest-first: newest 15 = 2.0, older 15 = 0.0.
        n = 30
        vals = [2.0] * 15 + [0.0] * 15
        gdates = list(reversed([f"2026-06-{d:02d}" for d in range(1, n + 1)]))
        hist = {"P": {"batter_hits": {"found": True, "values": vals,
                      "opponents": ["Y"] * n, "home_aways": [False] * n,
                      "game_dates": gdates}}}
        pdata = {"props": {"batter_hits": {"P": {
            "line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
            "over_price": -110, "under_price": -110,
            "over_book": "DK", "under_book": "DK"}}},
            "home_team": "X", "away_team": "Y"}
        with patch("props.load_calibration", return_value=calib):
            return props.analyze_player_props_value(
                pdata, hist, threshold_pct=1.0, sport_key="baseball_mlb")[0]

    def test_recent_n_slices_newest_games(self):
        # null = full season -> all 30: (15*2 + 15*0)/30 = 1.0
        self.assertAlmostEqual(
            self._project({"batter_hits": {"recent_n": None}})["avg_stat"], 1.0, places=2)
        # 20 -> newest 20: (15*2 + 5*0)/20 = 1.5
        self.assertAlmostEqual(
            self._project({"batter_hits": {"recent_n": 20}})["avg_stat"], 1.5, places=2)
        # 10 -> newest 10 (all 2.0) = 2.0 (MLB streak floor is 8, so 10 survives)
        self.assertAlmostEqual(
            self._project({"batter_hits": {"recent_n": 10}})["avg_stat"], 2.0, places=2)
        # absent -> MLB per-sport default 20 -> 1.5
        self.assertAlmostEqual(
            self._project({"batter_hits": {}})["avg_stat"], 1.5, places=2)


class RecencySweepGridTests(unittest.TestCase):
    """STEP-1 recency sweep: _preset carries recent_n, _build_recency_sweep_grid
    isolates the two recency axes (incumbent n20/none present), and the projection
    contract holds — a length-N weight vector selects the NEWEST N (arrays are
    most-recent-first) via _recency_weights/_weighted_* zip-truncation, which is what
    the base_w cap in run_player_props_backtest relies on to model recent_n."""

    def test_preset_carries_recent_n(self):
        import backtest
        # default = "__calib__" sentinel (refit resolves to the prop's LOCKED window)
        self.assertEqual(backtest._preset(half_life=None)["recent_n"], "__calib__")
        self.assertEqual(backtest._preset(half_life=5, recent_n=20)["recent_n"], 20)
        self.assertIsNone(backtest._preset(half_life=5, recent_n=None)["recent_n"])  # explicit full

    def test_grid_isolates_axes_and_includes_incumbent(self):
        import backtest
        g = backtest._build_recency_sweep_grid([15, 20, None], [None, 7])
        self.assertEqual(set(g), {"n15/none", "n15/hl7", "n20/none", "n20/hl7",
                                  "full/none", "full/hl7"})
        for cell in g.values():          # every other knob off -> axes isolated
            self.assertEqual(cell["opp_defense_strength"], 0.0)
            self.assertEqual(cell["shrink_k"], 0)
            self.assertEqual(cell["def_adj"], 0.0)
        self.assertIn("n20/none", backtest._build_recency_sweep_grid())   # incumbent cell

    def test_recent_n_truncation_selects_newest(self):
        # vals newest-first 10..1; recent_n=3 -> newest {10,9,8}: mean 9.0, and the
        # method-A over-rate at line 8.5 -> 2/3 (10,9). Full history -> mean 5.5.
        import stats
        vals = list(range(10, 0, -1))
        w_full = stats._recency_weights(len(vals), None)
        w_n3 = stats._recency_weights(min(3, len(vals)), None)
        self.assertAlmostEqual(stats._weighted_mean(vals, w_full), 5.5)
        self.assertAlmostEqual(stats._weighted_mean(vals, w_n3), 9.0)
        self.assertAlmostEqual(
            stats._weighted_rate(vals, w_n3, lambda v: v > 8.5), 2 / 3)


class MultiSeasonPoolingTests(unittest.TestCase):
    """STEP-2 multi-season pooling: _merge_props_results pools per-season
    run_player_props_backtest results into one dict (concatenated calib_obs +
    summed tallies) so the residual fit sees the COMBINED sample, and the pool
    unions dedupe players across seasons so a player active in only one season
    still contributes (and the thin pitcher pool widens)."""

    @staticmethod
    def _cell(obs, n=0, hits=0, decisive=0, safe=None, quantile=None):
        return {
            "errors": [], "n": n, "hits": hits, "decisive": decisive,
            "safe": safe or {}, "quantile": quantile or {},
            "calib_obs": list(obs),
        }

    @staticmethod
    def _obs(proj, actual, date):
        # (x, projected, synthetic_line, actual, empirical_over, date) — the shape
        # _fit_residuals (residual = actual-proj) and _chronological_folds (o[5])
        # consume.
        return (None, proj, proj, actual, 1 if actual > proj else 0, date)

    def test_merge_concatenates_obs_and_sums_tallies(self):
        import refit_calibration as rc
        a = {"v": {"batter_hits": self._cell(
            [self._obs(1.0, 2.0, "2024-04-01")], n=3, hits=2, decisive=3,
            safe={0.5: {"hits": 1, "n": 2}})}}
        b = {"v": {"batter_hits": self._cell(
            [self._obs(1.0, 0.0, "2025-04-01")], n=4, hits=1, decisive=4,
            safe={0.5: {"hits": 2, "n": 3}})}}
        merged = rc._merge_props_results(a, b)
        cell = merged["v"]["batter_hits"]
        self.assertEqual(len(cell["calib_obs"]), 2)      # pooled obs
        self.assertEqual(cell["n"], 7)
        self.assertEqual(cell["hits"], 3)
        self.assertEqual(cell["decisive"], 7)
        self.assertEqual(cell["safe"][0.5], {"hits": 3, "n": 5})
        # The residual fit now spans both seasons (residuals +1.0 and -1.0 -> mu 0).
        fit = rc._fit_residuals(cell["calib_obs"])
        self.assertEqual(fit["n_obs"], 2)
        self.assertAlmostEqual(fit["residual_mu"], 0.0)

    def test_merge_adds_missing_variant_or_prop(self):
        import refit_calibration as rc
        a = {"v1": {"batter_hits": self._cell([self._obs(1.0, 2.0, "2024-04-01")])}}
        b = {"v2": {"pitcher_strikeouts":
                    self._cell([self._obs(5.0, 6.0, "2024-04-01")])}}
        merged = rc._merge_props_results(a, b)
        self.assertIn("v1", merged)
        self.assertIn("v2", merged)          # a variant absent from acc is added
        self.assertIn("pitcher_strikeouts", merged["v2"])

    def test_mlb_pool_union_dedupes_by_id_and_role(self):
        import refit_calibration as rc
        pools = {
            2024: [("101", "batter", "A"), ("201", "pitcher", "P")],
            2025: [("101", "batter", "A"),      # dup id+role -> dropped
                   ("102", "batter", "B"),      # new
                   ("201", "pitcher", "P")],    # dup -> dropped
        }
        with patch.object(rc, "_mlb_player_pool",
                          side_effect=lambda sy, **kw: pools[sy]):
            union = rc._mlb_pool_union([2024, 2025])
        self.assertEqual(
            union,
            [("101", "batter", "A"), ("201", "pitcher", "P"),
             ("102", "batter", "B")])

    def test_nba_pool_union_dedupes_by_name(self):
        import refit_calibration as rc
        pools = {2024: ["A", "B"], 2025: ["B", "C"]}
        with patch.object(rc, "_nba_player_pool",
                          side_effect=lambda sy, **kw: pools[sy]):
            self.assertEqual(rc._nba_pool_union([2024, 2025]), ["A", "B", "C"])


class PortfolioSimTests(unittest.TestCase):
    """Batch B1 top-N/day portfolio: per-day best-N-by-EV selection across eligible
    markets, market policy (totals off, spreads high-conviction), per-day (not global)
    cap. Rows: (date, home_side_prob, fair, price_home, price_away, home_won, n_eff)."""

    ML = {"moneyline": {"enabled": True, "shrink": 0.25, "edge_gate": 0.05},
          "spreads": {"enabled": False}, "totals": {"enabled": False}}
    BOTH = {"moneyline": {"enabled": True, "shrink": 0.25, "edge_gate": 0.05},
            "spreads": {"enabled": True, "shrink": 1.0, "edge_gate": 0.10},
            "totals": {"enabled": False}}

    # ML rows (shrink 0.25): hp 0.90/0.80/0.70 -> shrunk 0.60/0.575/0.55 @ +100 ->
    # EV 0.20/0.15/0.10; A,B win, C loses.
    A = ("2025-04-01", 0.90, 0.50, 100, -120, 1, 100)
    B = ("2025-04-01", 0.80, 0.50, 100, -120, 1, 100)
    C = ("2025-04-01", 0.70, 0.50, 100, -120, 0, 100)

    def test_top_n_caps_per_day(self):
        import backtest
        bets = {"moneyline": [self.A, self.B, self.C], "spreads": [], "totals": []}
        self.assertEqual(backtest._portfolio_sim(bets, None, self.ML)["n"], 3)
        r2 = backtest._portfolio_sim(bets, 2, self.ML)
        self.assertEqual(r2["n"], 2)              # top-2 by EV = A,B
        self.assertEqual(r2["win"], 100.0)        # both won (C, the loser, dropped)

    def test_ranks_by_ev_across_markets(self):
        import backtest
        # Spreads bet: served cover 0.62 @ +100 -> edge 0.12, EV 0.24 (tops A's 0.20),
        # but it LOSES (home_covers=0). N=1 must pick it (highest EV) -> win 0%.
        s_lose = ("2025-04-01", 0.62, 0.50, 100, -120, 0, 100)
        bets = {"moneyline": [self.A, self.B, self.C], "spreads": [s_lose],
                "totals": []}
        r = backtest._portfolio_sim(bets, 1, self.BOTH)
        self.assertEqual(r["n"], 1)
        self.assertEqual(r["win"], 0.0)           # confirms the spread (top EV) chosen

    def test_totals_excluded_and_spread_gate(self):
        import backtest
        weak_spread = ("2025-04-01", 0.54, 0.50, 100, -120, 1, 100)  # edge .04<.10
        totals_bet = ("2025-04-01", 0.70, 0.50, 100, -120, 1, 100)   # totals off
        bets = {"moneyline": [self.A], "spreads": [weak_spread],
                "totals": [totals_bet]}
        r = backtest._portfolio_sim(bets, None, self.BOTH)
        self.assertEqual(r["n"], 1)               # only the ML bet A qualifies

    def test_cap_is_per_day_not_global(self):
        import backtest
        day2 = [(("2025-04-02",) + row[1:]) for row in (self.A, self.B, self.C)]
        bets = {"moneyline": [self.A, self.B, self.C] + day2, "spreads": [],
                "totals": []}
        r = backtest._portfolio_sim(bets, 2, self.ML)
        self.assertEqual(r["n"], 4)               # 2/day x 2 days, not 2 global
        self.assertAlmostEqual(r["avg_per_day"], 2.0)


if __name__ == "__main__":
    unittest.main()
