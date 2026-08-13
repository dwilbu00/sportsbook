"""Pitcher innings-pitched (IP) notation-vs-outs conversion tests.

Baseball records IP so the fractional digit is OUTS, not tenths (6.1 = 6 innings
+ 1 out = 19 outs). `pitcher_outs` is the only prop whose ESPN stat label ('IP')
needs conversion; it was correct on the two live production read paths but was
missing on four others (grading fallback, offline projection backtest, offline
real-line calibration, and the dormant splits fallback). These tests pin the
shared helper and each fixed site.
"""

import unittest
from unittest.mock import patch

import espn_client
import recalibration
import backtest
import book_line_calibration as blc


class IpHelperTests(unittest.TestCase):

    def test_ip_to_outs(self):
        f = espn_client.ip_to_outs
        self.assertEqual(f(6.0), 18)
        self.assertEqual(f(6.1), 19)
        self.assertEqual(f(6.2), 20)
        self.assertEqual(f(5.2), 17)
        self.assertEqual(f(0.0), 0)
        self.assertIsNone(f(None))

    def test_outs_to_ip(self):
        g = espn_client.outs_to_ip
        self.assertEqual(g(18), 6.0)
        self.assertEqual(g(19), 6.1)
        self.assertEqual(g(20), 6.2)
        self.assertEqual(g(0), 0.0)
        self.assertIsNone(g(None))

    def test_roundtrip_for_valid_notation(self):
        for ip in (0.0, 5.0, 5.1, 5.2, 6.0, 6.1, 6.2, 7.1, 9.0):
            self.assertAlmostEqual(
                espn_client.outs_to_ip(espn_client.ip_to_outs(ip)), ip)


class SplitsFallbackTests(unittest.TestCase):
    """Cluster 4: get_pitcher_stats must average IP in OUT space, not decimal."""

    def _resp(self, payload):
        class _R:
            def raise_for_status(self_):
                pass

            def json(self_):
                return payload
        return _R()

    def test_opponent_split_ip_averaged_in_outs(self):
        # 16.1 IP (= 49 outs) over gp=3 -> 16.3 outs/game -> 16 outs -> 5.1 IP.
        # The old decimal path gave round(16.1/3,1)=5.4 -> a bogus 19 outs.
        payload = {"labels": ["GP", "IP", "K"],
                   "splitCategories": [
                       {"displayName": "Opponent",
                        "splits": [{"stats": ["3", "16.1", "24"]}]}]}
        with patch.object(espn_client.requests, "get",
                          return_value=self._resp(payload)):
            rows = espn_client.get_pitcher_stats("mlb", "1")
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(r["IP"], 5.1)
            self.assertEqual(espn_client.ip_to_outs(r["IP"]), 16)

    def test_overall_fallback_ip_averaged_in_outs(self):
        # 100.0 IP (= 300 outs) over GS=15 -> 20 outs/game -> 6.2 IP.
        # The old decimal path gave round(100/15,1)=6.7 -> a bogus 25 outs.
        payload = {"labels": ["GS", "IP"],
                   "splitCategories": [
                       {"displayName": "Overall",
                        "splits": [{"stats": ["15", "100.0"]}]}]}
        with patch.object(espn_client.requests, "get",
                          return_value=self._resp(payload)):
            rows = espn_client.get_pitcher_stats("mlb", "2")
        self.assertEqual(len(rows), 15)
        self.assertEqual(rows[0]["IP"], 6.2)
        self.assertEqual(espn_client.ip_to_outs(rows[0]["IP"]), 20)


class GradingFallbackTests(unittest.TestCase):
    """Cluster 1 (P6 cutover): MLB grades from the WAREHOUSE + statsapi ONLY. When
    both miss, resolve_one_prop stays pending (None) and never touches the ESPN
    gamelog for ANY prop; the statsapi hard-ID value is passed through as-is. (The
    ESPN grading path remains for NBA/NFL/NHL — exercised in the postponed/role-gate
    suites retargeted to basketball.)"""

    def test_mlb_never_grades_off_espn(self):
        # After the statsapi miss, MLB must NOT fall to the ESPN gamelog for ANY
        # prop (incl. the old pitcher_outs IP->outs and the TB/RBI warehouse-only
        # cases) — it stays pending (None) and never even loads the gamelog.
        gamelog = [{"H": 2.0, "IP": 6.1, "RBI": 2.0, "TB": 3.0,
                    "game_date": "2025-07-01T18:00:00Z", "completed": True,
                    "opponent": "X", "is_home": True}]
        by_date = {"2025-07-01": [0]}
        for role, prop in (("P", "pitcher_outs"), ("B", "batter_hits"),
                           ("B", "batter_rbis"), ("B", "batter_total_bases")):
            with patch.object(recalibration, "_resolve_mlb_actual",
                              return_value=None), \
                 patch.object(recalibration, "_load_player_gamelog",
                              return_value=(gamelog, by_date)) as ld, \
                 patch.object(recalibration, "_pick_candidate", return_value=0):
                actual = recalibration.resolve_one_prop(
                    "baseball_mlb", role, prop, 0.5,
                    "2025-07-01", "2025-07-01T18:00:00Z")
            self.assertIsNone(actual, prop)
            ld.assert_not_called()       # ESPN gamelog never loaded for MLB

    def test_statsapi_outs_not_double_converted(self):
        with patch.object(recalibration, "_resolve_mlb_actual",
                          return_value=18):
            actual = recalibration.resolve_one_prop(
                "baseball_mlb", "P", "pitcher_outs", 18.5,
                "2025-07-01", "2025-07-01T18:00:00Z")
        self.assertEqual(actual, 18.0)   # statsapi outs passed through as-is


class RealLineCalibrationTests(unittest.TestCase):
    """Cluster 3: book_line_calibration.project_and_empirical converts IP."""

    def _obs(self, stat_label, val, line, actual):
        prior = [{stat_label: val, "is_home": True, "opponent": "X"}
                 for _ in range(10)]
        return {"prior_games": prior, "stat_label": stat_label, "line": line,
                "test_game": {"is_home": True}, "actual": actual}

    def test_pitcher_outs_projected_in_outs(self):
        obs = self._obs("IP", 6.0, 17.5, 18.0)   # 6.0 IP = 18 outs each
        projected, emp = blc.project_and_empirical(obs, {"half_life": None},
                                                   "baseball_mlb")
        self.assertAlmostEqual(projected, 18.0)  # not 6.0
        self.assertAlmostEqual(emp, 1.0)         # all 18 > 17.5

    def test_batter_hits_unchanged(self):
        obs = self._obs("H", 1.0, 0.5, 1.0)
        projected, emp = blc.project_and_empirical(obs, {"half_life": None},
                                                   "baseball_mlb")
        self.assertAlmostEqual(projected, 1.0)   # H not converted
        self.assertAlmostEqual(emp, 1.0)


class ProjectionBacktestTests(unittest.TestCase):
    """Cluster 2: run_player_props_backtest reads pitcher_outs in outs space."""

    def test_calib_obs_actual_and_line_in_outs(self):
        # gl[0] is the test game (5.2 IP = 17 outs); the rest are priors (6.0 IP
        # = 18 outs). No team_id -> no schedule fetch.
        gl = [{"IP": (5.2 if i == 0 else 6.0),
               "game_date": f"2025-07-{20 - i:02d}T18:00:00Z",
               "is_home": True, "opponent": "NYY"} for i in range(10)]
        off_variant = {"half_life": None, "venue_strength": 0.0,
                       "opp_defense_strength": 0.0, "def_adj": 0.0,
                       "pace_adj": 0.0, "park_strength": 0.0, "shrink_k": 0.0,
                       "rest_adj": 0.0, "use_minutes": False}
        passthrough = lambda prior, sched, sk, **kw: {
            "skip_prediction": False, "skip_reason": None,
            "eligible_games": prior}
        with patch.object(backtest, "fetch_player_data",
                          return_value={"P": gl}), \
             patch("prop_filter.filter_player_gamelog",
                   side_effect=passthrough), \
             patch.object(backtest, "get_all_teams", return_value={}), \
             patch.object(backtest, "_team_defense_lookup",
                          return_value=({}, {}, None)), \
             patch.object(backtest, "_team_pace_lookup",
                          return_value=({}, None)):
            res = backtest.run_player_props_backtest(
                "MLB", "baseball", "mlb", "baseball_mlb",
                players=["P"], props=["pitcher_outs"],
                games_per_player=1, min_sample=3,
                variants={"base": off_variant}, calibrate=True)
        obs = res["base"]["pitcher_outs"]["calib_obs"]
        self.assertTrue(obs)
        for (_name, _projected, synth, actual, _emp, _date) in obs:
            self.assertEqual(actual, 17.0)       # ip_to_outs(5.2), not 5.2
            self.assertAlmostEqual(synth, 18.0)  # mean of 6.0 -> 18 priors


class StrikeoutRoleGateTests(unittest.TestCase):
    """pitcher_strikeouts / batter_strikeouts share the "K"/"SO" labels; the
    props sweep must not leak a batter's strikeouts into the pitcher-K pool
    (or vice-versa). Gated on the pitcher-exclusive 'IP' field."""

    def test_prop_role(self):
        self.assertEqual(backtest._prop_role("pitcher_strikeouts"), "pitching")
        self.assertEqual(backtest._prop_role("pitcher_outs"), "pitching")
        self.assertEqual(backtest._prop_role("batter_hits"), "hitting")
        self.assertEqual(backtest._prop_role("batter_strikeouts"), "hitting")
        self.assertIsNone(backtest._prop_role("player_points"))  # NBA: no role

    def test_gamelog_is_pitcher(self):
        self.assertTrue(backtest._gamelog_is_pitcher([{"IP": 6.0, "K": 7}]))
        self.assertFalse(backtest._gamelog_is_pitcher([{"SO": 2, "H": 1}]))
        self.assertFalse(backtest._gamelog_is_pitcher([]))

    def test_role_matches_gamelog(self):
        pit = [{"IP": 6.0, "K": 7}]
        bat = [{"SO": 2, "H": 1}]
        self.assertTrue(backtest._role_matches_gamelog("pitcher_strikeouts", pit))
        self.assertFalse(backtest._role_matches_gamelog("pitcher_strikeouts", bat))
        self.assertTrue(backtest._role_matches_gamelog("batter_strikeouts", bat))
        self.assertFalse(backtest._role_matches_gamelog("batter_strikeouts", pit))
        # Non-MLB prop: no role concept -> always matches.
        self.assertTrue(backtest._role_matches_gamelog("player_points", bat))

    def test_sweep_does_not_cross_contaminate_strikeout_pools(self):
        # A pitcher (K + IP) and a batter (SO, no IP). Swept for BOTH strikeout
        # props, each pool must contain ONLY its own role's games.
        pit_gl = [{"K": (7 if i == 0 else 6), "IP": 6.0,
                   "game_date": f"2025-07-{20 - i:02d}T18:00:00Z",
                   "is_home": True, "opponent": "NYY"} for i in range(10)]
        bat_gl = [{"SO": (2 if i == 0 else 1), "H": 1,
                   "game_date": f"2025-07-{20 - i:02d}T18:00:00Z",
                   "is_home": True, "opponent": "BOS"} for i in range(10)]
        off = {"half_life": None, "venue_strength": 0.0,
               "opp_defense_strength": 0.0, "def_adj": 0.0, "pace_adj": 0.0,
               "park_strength": 0.0, "shrink_k": 0.0, "rest_adj": 0.0,
               "use_minutes": False}
        passthrough = lambda prior, sched, sk, **kw: {
            "skip_prediction": False, "skip_reason": None,
            "eligible_games": prior}
        with patch.object(backtest, "fetch_player_data",
                          return_value={"Pitcher": pit_gl, "Batter": bat_gl}), \
             patch("prop_filter.filter_player_gamelog", side_effect=passthrough), \
             patch.object(backtest, "get_all_teams", return_value={}), \
             patch.object(backtest, "_team_defense_lookup",
                          return_value=({}, {}, None)), \
             patch.object(backtest, "_team_pace_lookup", return_value=({}, None)):
            res = backtest.run_player_props_backtest(
                "MLB", "baseball", "mlb", "baseball_mlb",
                players=["Pitcher", "Batter"],
                props=["pitcher_strikeouts", "batter_strikeouts"],
                games_per_player=1, min_sample=3,
                variants={"base": off}, calibrate=True)

        pit_obs = res["base"]["pitcher_strikeouts"]["calib_obs"]
        bat_obs = res["base"]["batter_strikeouts"]["calib_obs"]
        self.assertTrue(pit_obs)   # the correct role still resolves
        self.assertTrue(bat_obs)
        self.assertEqual({o[0] for o in pit_obs}, {"Pitcher"})  # no Batter leak
        self.assertEqual({o[0] for o in bat_obs}, {"Batter"})   # no Pitcher leak


if __name__ == "__main__":
    unittest.main()
