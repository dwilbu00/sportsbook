"""P3/P6: the offline synthetic-sweep calibration engine sources MLB inputs from the
StatsAPI warehouse instead of ESPN.

These lock the gate + the four re-keyed seams WITHOUT touching a DB or ESPN:
  * the env/param gate (ODI_MLB_WAREHOUSE_BACKTEST), MLB-only so NBA/NFL are byte-id
  * fetch_player_data → mlb_warehouse.get_calib_gamelog (by MLBAM id+role), no ESPN
  * _mlb_warehouse_defense_lookup keys on the CANONICAL name the player rows carry
    (opponent-name match by construction — the whole point of the re-key)
  * _mlb_player_pool enriched to (mlb_id, role, name)
"""

import os
import unittest
from unittest import mock

import backtest
import refit_calibration


class GateTests(unittest.TestCase):
    def test_env_flag_reads_truthy_values(self):
        for val in ("1", "true", "on", "yes", "TRUE"):
            with mock.patch.dict(os.environ,
                                 {"ODI_MLB_WAREHOUSE_BACKTEST": val}):
                self.assertTrue(backtest._mlb_warehouse_backtest_enabled())
        for val in ("", "0", "no", "off"):
            with mock.patch.dict(os.environ,
                                 {"ODI_MLB_WAREHOUSE_BACKTEST": val}):
                self.assertFalse(backtest._mlb_warehouse_backtest_enabled())

    def _gate_call(self, espn_sport, env, warehouse_inputs=None):
        """Run run_player_props_backtest just far enough to capture the
        warehouse_inputs fetch_player_data received (empty data → early return)."""
        fpd = mock.Mock(return_value={})
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(backtest, "fetch_player_data", fpd):
            backtest.run_player_props_backtest(
                "mlb" if espn_sport == "baseball" else "nba",
                espn_sport, "x", "sk", players=["A"], props=["batter_hits"],
                games_per_player=5, min_sample=5,
                variants={"base": {"half_life": 10}},
                warehouse_inputs=warehouse_inputs)
        return fpd.call_args.kwargs.get("warehouse_inputs")

    def test_mlb_env_on_uses_warehouse(self):
        self.assertTrue(
            self._gate_call("baseball", {"ODI_MLB_WAREHOUSE_BACKTEST": "1"}))

    def test_mlb_env_off_uses_espn(self):
        self.assertFalse(
            self._gate_call("baseball", {"ODI_MLB_WAREHOUSE_BACKTEST": ""}))

    def test_non_baseball_never_uses_warehouse_even_if_forced(self):
        # NBA with the flag ON AND an explicit force must stay ESPN (byte-identical).
        fpd = mock.Mock(return_value={})
        with mock.patch.dict(os.environ, {"ODI_MLB_WAREHOUSE_BACKTEST": "1"}), \
             mock.patch.object(backtest, "fetch_player_data", fpd):
            backtest.run_player_props_backtest(
                "nba", "basketball", "nba", "basketball_nba",
                players=["A"], props=["player_points"], games_per_player=5,
                min_sample=5, variants={"base": {"half_life": 10}},
                warehouse_inputs=True)
        self.assertFalse(fpd.call_args.kwargs.get("warehouse_inputs"))

    def test_explicit_param_overrides_env(self):
        # warehouse_inputs=False forces ESPN even with the env flag ON.
        self.assertFalse(
            self._gate_call("baseball", {"ODI_MLB_WAREHOUSE_BACKTEST": "1"},
                            warehouse_inputs=False))


class IterPoolPlayersTests(unittest.TestCase):
    def test_tuple_and_string_entries(self):
        out = list(backtest._iter_pool_players(
            [("592450", "batter", "Aaron Judge"), "Bare Name"]))
        self.assertEqual(out[0], ("592450", "batter", "Aaron Judge"))
        self.assertEqual(out[1], (None, None, "Bare Name"))


class FetchPlayerDataWarehouseTests(unittest.TestCase):
    _WH_LOG = [{"H": 2.0, "game_date": "2026-07-20T18:00:00Z",
                "is_home": True, "opponent": "Boston Red Sox", "team_id": "147",
                "completed": True}]

    def test_baseball_warehouse_path_uses_get_calib_gamelog_not_espn(self):
        gcg = mock.Mock(return_value=list(self._WH_LOG))
        with mock.patch.object(backtest, "cached_athlete_id") as caid, \
             mock.patch.object(backtest, "cached_gamelog") as cgl, \
             mock.patch("mlb_warehouse.get_calib_gamelog", gcg):
            data = backtest.fetch_player_data(
                "baseball", "mlb", [("592450", "batter", "Aaron Judge")],
                season_year=2026, warehouse_inputs=True)
        self.assertIn("Aaron Judge", data)
        self.assertEqual(data["Aaron Judge"][0]["H"], 2.0)
        gcg.assert_called_once_with("592450", "batter", season=2026)
        caid.assert_not_called()          # ESPN id lookup bypassed
        cgl.assert_not_called()           # ESPN gamelog bypassed

    def test_bare_name_resolved_via_resolver_under_flag(self):
        gcg = mock.Mock(return_value=list(self._WH_LOG))
        with mock.patch("mlb_starters.resolve_mlbam_id",
                        return_value=(605483, True)) as rz, \
             mock.patch("mlb_warehouse.get_calib_gamelog", gcg), \
             mock.patch("mlb_warehouse._current_season", return_value=2026):
            data = backtest.fetch_player_data(
                "baseball", "mlb", ["Some Pitcher"], warehouse_inputs=True)
        rz.assert_called_once()
        gcg.assert_called_once_with("605483", "pitcher", season=None)
        self.assertIn("Some Pitcher", data)

    def test_warehouse_miss_skips_player(self):
        with mock.patch("mlb_warehouse.get_calib_gamelog", return_value=[]):
            data = backtest.fetch_player_data(
                "baseball", "mlb", [("1", "batter", "Ghost")],
                season_year=2026, warehouse_inputs=True)
        self.assertEqual(data, {})

    def test_flag_off_uses_espn_path_with_name_from_tuple(self):
        with mock.patch.object(backtest, "cached_athlete_id",
                               return_value="e1") as caid, \
             mock.patch.object(backtest, "cached_gamelog",
                               return_value=[{"H": 1.0, "game_date": "2026-07-01"}]), \
             mock.patch("mlb_warehouse.get_calib_gamelog") as gcg:
            data = backtest.fetch_player_data(
                "baseball", "mlb", [("592450", "batter", "Aaron Judge")],
                season_year=2026, warehouse_inputs=False)
        caid.assert_called_once_with("baseball", "mlb", "Aaron Judge")
        gcg.assert_not_called()
        self.assertIn("Aaron Judge", data)

    def test_non_baseball_ignores_warehouse_flag(self):
        with mock.patch.object(backtest, "cached_athlete_id",
                               return_value="e1") as caid, \
             mock.patch.object(backtest, "cached_gamelog",
                               return_value=[{"PTS": 20.0, "game_date": "2026-01-01"}]), \
             mock.patch("mlb_warehouse.get_calib_gamelog") as gcg:
            backtest.fetch_player_data(
                "basketball", "nba", ["Star"], warehouse_inputs=True)
        caid.assert_called_once()          # ESPN path
        gcg.assert_not_called()

    def test_same_name_namesakes_both_survive(self):
        # Fix A: two distinct MLBAM ids sharing a fullName each keep their own gamelog
        # (name collision disambiguated by id) instead of the second overwriting the
        # first — the id-dedup pool's stated intent, now true downstream.
        def _gcg(rid, role, season=None):
            label = "K" if role == "pitcher" else "H"
            return [{label: 1.0, "game_date": "2026-07-20T18:00Z",
                     "is_home": True, "opponent": "X", "team_id": "1",
                     "completed": True}]
        with mock.patch("mlb_warehouse.get_calib_gamelog", side_effect=_gcg):
            data = backtest.fetch_player_data(
                "baseball", "mlb",
                [("669257", "batter", "Will Smith"),
                 ("592858", "pitcher", "Will Smith")],
                season_year=2026, warehouse_inputs=True)
        self.assertEqual(len(data), 2)                     # both survived
        self.assertIn("Will Smith", data)
        self.assertIn("Will Smith (592858)", data)
        self.assertIn("H", data["Will Smith"][0])          # batter log kept
        self.assertIn("K", data["Will Smith (592858)"][0])  # pitcher log kept


class NoneStatFilterTests(unittest.TestCase):
    """Fix B: a legacy warehouse row with a present-but-None served stat (pre-a68f4e6
    TB/RBI capture) must be dropped from prior_games, not poison prior_values/sum()."""

    def test_none_stat_games_dropped_no_crash(self):
        gl = []
        for d in range(1, 10):
            h = None if d in (3, 6) else float(d % 3)   # two None-H legacy rows
            gl.append({"H": h, "game_date": f"2026-07-0{d}T18:00:00Z",
                       "is_home": True, "opponent": "X", "completed": True})
        gl.sort(key=lambda g: g["game_date"], reverse=True)

        def _passthru(prior, sched, sk, **k):
            return {"skip_prediction": False, "skip_reason": None,
                    "eligible_games": prior}

        with mock.patch.object(backtest, "fetch_player_data",
                               return_value={"Guy": gl}), \
             mock.patch("prop_filter.filter_player_gamelog", side_effect=_passthru):
            res = backtest.run_player_props_backtest(
                "mlb", "baseball", "mlb", "baseball_mlb",
                players=["Guy"], props=["batter_hits"],
                games_per_player=80, min_sample=2,
                variants={"base": {"half_life": 10}}, calibrate=True)
        # Completed without a TypeError from summing a None-laden prior_values.
        self.assertIsNotNone(res)
        self.assertIn("base", res)


class WarmupMetaTests(unittest.TestCase):
    """Fix C: a requested warmup that yields nothing (e.g. warehouse lacks the prior
    season) is flagged in meta (warmup_present=False), not silently omitted."""

    def test_missing_warmup_flags_meta_present_false(self):
        captured = {}
        with mock.patch.object(refit_calibration, "run_player_props_backtest",
                               side_effect=[{"base": {}}, None]), \
             mock.patch.object(refit_calibration, "_best_per_prop", return_value={}), \
             mock.patch.object(
                 refit_calibration, "save_calibration",
                 side_effect=lambda sk, cfg, meta=None: captured.update(meta=meta)):
            refit_calibration.refit_sport(
                "mlb", season=2026, prior_season=2025,
                players=[("1", "batter", "Guy")], props=["batter_hits"],
                warehouse_inputs=True)
        self.assertFalse(captured["meta"]["warmup_present"])
        self.assertEqual(captured["meta"]["warmup_season"], 2025)


class WarehouseDefenseLookupTests(unittest.TestCase):
    # Two teams, two games between them: NYY 5-3 home, BOS 6-2 home.
    NAMES = {"147": "New York Yankees", "111": "Boston Red Sox"}
    GAMES = {
        "147": [
            {"date": "2026-07-20T18:00Z", "home_team": "New York Yankees",
             "away_team": "Boston Red Sox", "home_score": 5, "away_score": 3},
            {"date": "2026-07-19T18:00Z", "home_team": "Boston Red Sox",
             "away_team": "New York Yankees", "home_score": 6, "away_score": 2},
        ],
        "111": [
            {"date": "2026-07-20T18:00Z", "home_team": "New York Yankees",
             "away_team": "Boston Red Sox", "home_score": 5, "away_score": 3},
            {"date": "2026-07-19T18:00Z", "home_team": "Boston Red Sox",
             "away_team": "New York Yankees", "home_score": 6, "away_score": 2},
        ],
    }

    def _lookup(self):
        with mock.patch("mlb_warehouse._team_name_map", return_value=self.NAMES), \
             mock.patch("mlb_warehouse._team_final_games",
                        side_effect=lambda tid, season=None: self.GAMES.get(str(tid), [])):
            return backtest._mlb_warehouse_defense_lookup(season_year=2026)

    def test_keys_are_canonical_names_and_allowed_is_opponent_score(self):
        avg, series, league = self._lookup()
        # NYY allowed = away_score(3) when home + home_score(6) when away = 4.5
        self.assertAlmostEqual(avg["New York Yankees"], 4.5)
        # BOS allowed = home_score(5) when away + away_score(2) when home = 3.5
        self.assertAlmostEqual(avg["Boston Red Sox"], 3.5)
        self.assertAlmostEqual(league, 4.0)
        self.assertEqual(set(avg), {"New York Yankees", "Boston Red Sox"})
        # series sorted most-recent-first
        self.assertEqual([d for d, _ in series["New York Yankees"]],
                         ["2026-07-20T18:00Z", "2026-07-19T18:00Z"])

    def test_opponent_names_match_defense_keys_by_construction(self):
        # The load-bearing invariant: the opponent name get_calib_gamelog stamps on a
        # player row comes from _team_name_map — the SAME source as the defense keys —
        # so _resolve_opp_pts_allowed hits without any tolerant/fuzzy matching.
        avg, _, _ = self._lookup()
        opponent_names = set(self.NAMES.values())      # what get_calib_gamelog stamps
        self.assertTrue(opponent_names <= set(avg))


class MlbPlayerPoolTests(unittest.TestCase):
    def test_pool_enriched_with_id_and_role(self):
        people = {"people": [
            {"id": 592450, "fullName": "Aaron Judge"},
            {"id": 605483, "fullName": "Brayan Bello"},
        ]}
        with mock.patch("backtest_props.frequent_batter_ids",
                        return_value=[592450]), \
             mock.patch("backtest_props.starter_ids", return_value=[605483]), \
             mock.patch("mlb_starters._get", return_value=people):
            pool = refit_calibration._mlb_player_pool(2026)
        self.assertEqual(pool, [("592450", "batter", "Aaron Judge"),
                                ("605483", "pitcher", "Brayan Bello")])

    def test_dedupes_by_id(self):
        # Same id in both lists (defensive) → one entry, role = first-seen (batter).
        people = {"people": [{"id": 1, "fullName": "Two Way"}]}
        with mock.patch("backtest_props.frequent_batter_ids", return_value=[1]), \
             mock.patch("backtest_props.starter_ids", return_value=[1]), \
             mock.patch("mlb_starters._get", return_value=people):
            pool = refit_calibration._mlb_player_pool(2026)
        self.assertEqual(pool, [("1", "batter", "Two Way")])


class WarehouseTeamSchedulesTests(unittest.TestCase):
    """P3b: _warehouse_team_schedules mirrors get_all_teams + build_schedules for the
    team-market backtests, pooling seasons, canonical names."""

    NAMES = {"147": "New York Yankees", "111": "Boston Red Sox"}

    def _tfg(self, tid, season=None):
        return [{"date": f"{season}-07-20T18:00Z", "home_team": "New York Yankees",
                 "away_team": "Boston Red Sox", "home_score": 5, "away_score": 3,
                 "total_score": 8}]

    def test_shape_season_pooling_and_win_pct(self):
        with mock.patch("mlb_warehouse._team_name_map", return_value=self.NAMES), \
             mock.patch("mlb_warehouse._team_final_games", side_effect=self._tfg):
            teams, sched = backtest._warehouse_team_schedules([2025, 2026])
        # NYY home 5-3 in every fixture game → 1.0; BOS away, loses → 0.0.
        self.assertEqual(teams["New York Yankees"], {"id": "147", "win_pct": 1.0})
        self.assertEqual(teams["Boston Red Sox"], {"id": "111", "win_pct": 0.0})
        self.assertEqual(len(sched["147"]), 2)          # pooled across both seasons
        self.assertEqual(sched["147"][0]["home_team"], "New York Yankees")

    def test_none_season_maps_to_current_not_all_seasons(self):
        seen = []

        def _tfg(tid, season=None):
            seen.append(season)
            return []

        with mock.patch("mlb_warehouse._team_name_map", return_value={"147": "NYY"}), \
             mock.patch("mlb_warehouse._current_season", return_value=2026), \
             mock.patch("mlb_warehouse._team_final_games", side_effect=_tfg):
            backtest._warehouse_team_schedules([None])
        self.assertEqual(seen, [2026])   # None → current, NOT the unfiltered all-seasons branch

    def test_no_games_win_pct_defaults_neutral(self):
        with mock.patch("mlb_warehouse._team_name_map", return_value={"147": "NYY"}), \
             mock.patch("mlb_warehouse._current_season", return_value=2026), \
             mock.patch("mlb_warehouse._team_final_games", return_value=[]):
            teams, _ = backtest._warehouse_team_schedules([2026])
        self.assertEqual(teams["NYY"]["win_pct"], 0.5)


class TeamMarketGateTests(unittest.TestCase):
    """P3b: run_backtest / run_odds_backtest source MLB team schedules from the
    warehouse when gated, ESPN otherwise; non-baseball is byte-identical."""

    def test_run_backtest_baseball_env_on_uses_warehouse(self):
        wh = mock.Mock(return_value=({}, {}))   # empty → clean "No games" early return
        with mock.patch.dict(os.environ, {"ODI_MLB_WAREHOUSE_BACKTEST": "1"}), \
             mock.patch.object(backtest, "_warehouse_team_schedules", wh), \
             mock.patch.object(backtest, "get_all_teams") as gat:
            backtest.run_backtest("baseball_mlb", "baseball", "mlb", limit=10,
                                  window=10, variants={"base": {"half_life": 10}},
                                  season_year=2026)
        wh.assert_called_once()
        gat.assert_not_called()

    def test_run_backtest_nba_ignores_flag(self):
        with mock.patch.dict(os.environ, {"ODI_MLB_WAREHOUSE_BACKTEST": "1"}), \
             mock.patch.object(backtest, "_warehouse_team_schedules") as wh, \
             mock.patch.object(backtest, "get_all_teams", return_value={}) as gat, \
             mock.patch.object(backtest, "build_schedules", return_value={}):
            backtest.run_backtest("basketball_nba", "basketball", "nba", limit=10,
                                  window=10, variants={"base": {"half_life": 10}},
                                  season_year=2025)
        gat.assert_called_once()
        wh.assert_not_called()

    def test_run_odds_backtest_baseball_env_on_uses_warehouse(self):
        store = {"games": {"g1": {"commence_time": "2026-07-01T00:00:00Z",
                                  "home_team": "A", "away_team": "B"}},
                 "bookmaker": "dk"}
        wh = mock.Mock(return_value=({}, {}))
        with mock.patch.dict(os.environ, {"ODI_MLB_WAREHOUSE_BACKTEST": "1"}), \
             mock.patch.object(backtest, "_load_odds_store",
                               return_value=(store, "store")), \
             mock.patch.object(backtest, "_build_odds_lookup", return_value=({}, 0)), \
             mock.patch.object(backtest, "_warehouse_team_schedules", wh), \
             mock.patch.object(backtest, "get_all_teams") as gat:
            try:
                backtest.run_odds_backtest(
                    "baseball_mlb", "baseball", "mlb", limit=10, window=10,
                    variants={"base": {"half_life": 10}}, season_year=2026,
                    supplement_log=False)
            except Exception:
                pass   # downstream empty-slate handling isn't under test; the
                       # warehouse branch runs before any of it
        wh.assert_called_once()
        gat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
