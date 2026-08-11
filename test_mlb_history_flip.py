"""P4 MODEL-INPUT FLIP (env-gated, fail-open):

espn_client.get_player_stat_history serves MLB player histories from the StatsAPI
warehouse (mlb_warehouse.get_player_history) when ODI_MLB_WAREHOUSE_HIST is on,
falling open to ESPN otherwise; the parity harness forces the ESPN side. props then
bridges the warehouse dict's MLBAM team_id via the additive team_name (venue/opp-
defense + a name→ESPN-schedule fallback for the layoff filter).

The warehouse + id map + ESPN fetch are monkeypatched; no DB or network is touched.
"""

import os
import unittest
from unittest import mock

import espn_client
import props


WH_DICT = {
    "player": "Aaron Judge", "athlete_id": "592450", "stat_label": "H",
    "values": [2.0, 1.0], "opponents": ["Boston Red Sox", "Boston Red Sox"],
    "home_aways": [True, False], "minutes": [0.0, 0.0],
    "game_dates": ["2026-08-09T18:00:00Z", "2026-08-08T18:00:00Z"],
    "plate_appearances": [4.0, 4.0], "at_bats": [4.0, 4.0],
    "team_id": "147", "team_name": "New York Yankees", "found": True,
}


def _flag_on():
    return mock.patch.dict(os.environ, {espn_client._MLB_WAREHOUSE_HIST_ENV: "1"})


def _flag_off():
    return mock.patch.dict(os.environ, {espn_client._MLB_WAREHOUSE_HIST_ENV: ""})


class WarehouseHistGateTests(unittest.TestCase):
    """_mlb_warehouse_history gating: flag + sport + SQL + servable-prop + a
    globally-unique name resolution, else None (fall open to ESPN)."""

    def _call(self, sport="baseball", prop="batter_hits", name="Aaron Judge",
              mlb_id="592450", wh=WH_DICT, sql_enabled=True):
        with mock.patch.object(
                espn_client, "db_store",
                mock.Mock(enabled=mock.Mock(return_value=sql_enabled))), \
             mock.patch("player_id_map.mlb_id_for_name", return_value=mlb_id), \
             mock.patch("mlb_warehouse.get_player_history", return_value=wh):
            return espn_client._mlb_warehouse_history(sport, name, prop, 20)

    def test_flag_off_returns_none(self):
        with _flag_off():
            self.assertIsNone(self._call())

    def test_flag_on_baseball_hit(self):
        with _flag_on():
            self.assertEqual(self._call(), WH_DICT)

    def test_non_baseball_returns_none(self):
        with _flag_on():
            self.assertIsNone(self._call(sport="basketball"))

    def test_sql_disabled_returns_none(self):
        with _flag_on():
            self.assertIsNone(self._call(sql_enabled=False))

    def test_unsupported_prop_returns_none(self):
        with _flag_on():                       # HR has no fact column
            self.assertIsNone(self._call(prop="batter_home_runs"))

    def test_ambiguous_or_unknown_name_returns_none(self):
        with _flag_on():
            self.assertIsNone(self._call(mlb_id=None))

    def test_warehouse_no_rows_returns_none(self):
        with _flag_on():
            self.assertIsNone(self._call(wh=None))

    def test_never_raises(self):
        with _flag_on(), mock.patch.object(
                espn_client, "db_store",
                mock.Mock(enabled=mock.Mock(side_effect=RuntimeError("boom")))):
            self.assertIsNone(espn_client._mlb_warehouse_history(
                "baseball", "X", "batter_hits", 20))

    def test_scopes_to_current_season(self):
        # Matches the ESPN/gamelog_store baseline (current-season-only), so the
        # recent-N window isn't padded with prior-season games early in the year.
        with _flag_on(), \
             mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=True))), \
             mock.patch("player_id_map.mlb_id_for_name", return_value="592450"), \
             mock.patch("mlb_warehouse._current_season", return_value=2026), \
             mock.patch("mlb_warehouse.get_player_history",
                        return_value=WH_DICT) as gph:
            espn_client._mlb_warehouse_history("baseball", "Aaron Judge",
                                               "batter_hits", 20)
        self.assertEqual(gph.call_args.kwargs.get("season"), 2026)


class GetPlayerStatHistoryBranchTests(unittest.TestCase):
    """The warehouse-first branch short-circuits the ESPN path on a hit, and
    allow_warehouse=False forces ESPN (the parity harness relies on this)."""

    def test_warehouse_hit_short_circuits(self):
        with mock.patch.object(espn_client, "_mlb_warehouse_history",
                               return_value=WH_DICT) as wh:
            r = espn_client.get_player_stat_history(
                "baseball", "mlb", "Aaron Judge", "batter_hits", n=10)
        self.assertEqual(r, WH_DICT)
        wh.assert_called_once()

    def test_allow_warehouse_false_forces_espn(self):
        with mock.patch.object(espn_client, "_mlb_warehouse_history",
                               return_value=WH_DICT) as wh, \
             mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=False))), \
             mock.patch.object(espn_client, "search_athlete", return_value=None):
            r = espn_client.get_player_stat_history(
                "baseball", "mlb", "Aaron Judge", "batter_hits", n=10,
                allow_warehouse=False)
        wh.assert_not_called()                 # warehouse never consulted
        self.assertFalse(r["found"])           # fell to ESPN (no athlete resolved)

    def test_warehouse_miss_falls_open_to_espn(self):
        with mock.patch.object(espn_client, "_mlb_warehouse_history",
                               return_value=None), \
             mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=False))), \
             mock.patch.object(espn_client, "search_athlete", return_value=None):
            r = espn_client.get_player_stat_history(
                "baseball", "mlb", "X", "batter_hits", n=10)
        self.assertFalse(r["found"])


class PropsTeamBridgeTests(unittest.TestCase):
    """props resolves a warehouse dict's team via the additive team_name (its
    team_id is MLBAM, absent from the ESPN-id-keyed id_to_name), so venue/opp-
    defense keep working; without team_name the ESPN-id reverse-map (correctly)
    misses. The bridge is inert for ESPN dicts."""

    ESPN_TEAMS = {"New York Yankees": {"id": "e-nyy"},
                  "Boston Red Sox": {"id": "e-bos"}}

    def _prop_data(self):
        return {
            "commence_time": "2026-08-10T23:10:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "game_id": "evt-bridge",
            "props": {"batter_hits": {"Aaron Judge": {
                "line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
                "over_price": -110, "under_price": -110,
                "over_book": "DK", "under_book": "DK"}}},
        }

    def _history(self, n=12, team_name="New York Yankees", team_id="147"):
        dates = [f"2026-06-{d:02d}" for d in range(1, 1 + n)]
        h = {"found": True, "values": [1.0] * n,
             "opponents": ["Minnesota Twins"] * n, "home_aways": [False] * n,
             "game_dates": list(reversed(dates)),
             "plate_appearances": [4.0] * n, "at_bats": [4.0] * n,
             "team_id": team_id}
        if team_name is not None:
            h["team_name"] = team_name
        return {"Aaron Judge": {"batter_hits": h}}

    def _run(self, history):
        return props.analyze_player_props_value(
            self._prop_data(), history, threshold_pct=1.0, sport_key=None,
            espn_teams=self.ESPN_TEAMS)[0]

    def test_team_name_bridges_mlbam_id(self):
        # team_id "147" (MLBAM) is NOT in id_to_name ("e-nyy"); team_name resolves it.
        cand = self._run(self._history())
        self.assertEqual(cand["team"], "New York Yankees")

    def test_without_team_name_espn_id_map_misses(self):
        # No team_name → id_to_name.get("147") misses → team unresolved (proves the
        # bridge, not id_to_name, did the resolution above).
        cand = self._run(self._history(team_name=None))
        self.assertIsNone(cand["team"])

    def test_espn_dict_unaffected(self):
        # An ESPN-style dict (ESPN team id, no team_name) still resolves via id_to_name.
        cand = self._run(self._history(team_name=None, team_id="e-nyy"))
        self.assertEqual(cand["team"], "New York Yankees")

    def test_divergent_name_tolerant_schedule_bridge(self):
        # Warehouse StatsAPI "Athletics" tolerant-matches ESPN "Oakland Athletics"
        # so the layoff filter still receives that team's schedule (an exact .get
        # would drop it — a regression vs the ESPN id-keyed lookup).
        espn_teams = {"Oakland Athletics": {"id": "e-oak"},
                      "Boston Red Sox": {"id": "e-bos"}}
        sched = [{"date": f"2026-06-{d:02d}"} for d in range(1, 13)]
        team_schedules = {"e-oak": sched}
        dates = [f"2026-06-{d:02d}" for d in range(1, 13)]
        prop_data = {
            "commence_time": "2026-08-10T23:10:00Z",
            "home_team": "Oakland Athletics", "away_team": "Boston Red Sox",
            "game_id": "evt-oak",
            "props": {"batter_hits": {"Some Athletic": {
                "line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
                "over_price": -110, "under_price": -110,
                "over_book": "DK", "under_book": "DK"}}},
        }
        history = {"Some Athletic": {"batter_hits": {
            "found": True, "values": [1.0] * 12,
            "opponents": ["Minnesota Twins"] * 12, "home_aways": [False] * 12,
            "game_dates": list(reversed(dates)),
            "plate_appearances": [4.0] * 12, "at_bats": [4.0] * 12,
            "team_id": "133", "team_name": "Athletics"}}}
        captured = {}
        orig = props.filter_player_gamelog

        def spy(gamelog, team_schedule, *a, **k):
            captured["sched"] = team_schedule
            return orig(gamelog, team_schedule, *a, **k)

        with mock.patch.object(props, "filter_player_gamelog", side_effect=spy):
            props.analyze_player_props_value(
                prop_data, history, threshold_pct=1.0, sport_key=None,
                espn_teams=espn_teams, team_schedules=team_schedules)
        self.assertEqual(captured.get("sched"), sched)  # tolerant name match hit


def _team_on():
    return mock.patch.dict(os.environ, {espn_client._MLB_WAREHOUSE_TEAM_ENV: "1"})


def _team_off():
    return mock.patch.dict(os.environ, {espn_client._MLB_WAREHOUSE_TEAM_ENV: ""})


class TeamMarketFlipTests(unittest.TestCase):
    """mlb_warehouse_team_stats: env-gated, MLB-only, fail-open; rekeys the queried
    team's recent_games to the ODDS name so the analyzers' exact match holds."""

    # Canonical-named games (queried team home in g1, away in g2).
    GAMES = [
        {"date": "2026-08-09T18:00:00Z", "home_team": "Oakland Athletics",
         "away_team": "Boston Red Sox", "home_score": 5, "away_score": 3,
         "total_score": 8},
        {"date": "2026-08-08T18:00:00Z", "home_team": "Tampa Bay Rays",
         "away_team": "Oakland Athletics", "home_score": 2, "away_score": 4,
         "total_score": 6},
    ]
    SEASON = {"record": "70-50", "wins": 70, "losses": 50, "win_pct": 0.583,
              "runs_scored": 600, "runs_allowed": 520}

    _MISSING = object()

    def _call(self, sport="baseball", team="Athletics", canonical="Oakland Athletics",
              season=_MISSING, games=_MISSING, sql=True):
        season = self.SEASON if season is self._MISSING else season
        games = self.GAMES if games is self._MISSING else games
        with mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=sql))), \
             mock.patch("mlb_warehouse.team_name_canonical", return_value=canonical), \
             mock.patch("mlb_warehouse.get_team_standings", return_value=season), \
             mock.patch("mlb_warehouse.get_team_games",
                        return_value=[dict(g) for g in games]), \
             mock.patch("mlb_warehouse._current_season", return_value=2026):
            return espn_client.mlb_warehouse_team_stats(sport, team, recent_n=10)

    def test_flag_off_returns_none(self):
        with _team_off():
            self.assertIsNone(self._call())

    def test_flag_on_rekeys_to_odds_name_and_matches(self):
        # Odds "Athletics" vs canonical "Oakland Athletics" — the divergent case.
        with _team_on():
            s = self._call()
        self.assertIsNotNone(s)
        self.assertEqual(s["season"], self.SEASON)
        # queried team's own name rekeyed to the odds spelling; opponent left canonical
        self.assertEqual(s["recent_games"][0]["home_team"], "Athletics")
        self.assertEqual(s["recent_games"][0]["away_team"], "Boston Red Sox")
        self.assertEqual(s["recent_games"][1]["away_team"], "Athletics")
        # compute_recent_form matched both games (proves the rekey is load-bearing:
        # without it, the canonical/odds gap would zero the form)
        self.assertEqual(s["recent"]["games"], 2)
        self.assertEqual(s["recent"]["wins"], 2)          # 5>3 and 4>2
        self.assertAlmostEqual(s["recent"]["avg_scored"], 4.5)  # (5+4)/2

    def test_non_baseball_returns_none(self):
        with _team_on():
            self.assertIsNone(self._call(sport="basketball"))

    def test_sql_off_returns_none(self):
        with _team_on():
            self.assertIsNone(self._call(sql=False))

    def test_unresolved_name_returns_none(self):
        with _team_on():
            self.assertIsNone(self._call(canonical=None))

    def test_no_standings_returns_none(self):
        with _team_on():
            self.assertIsNone(self._call(season=None))

    def test_no_games_returns_none(self):
        with _team_on():
            self.assertIsNone(self._call(games=[]))

    def test_never_raises(self):
        with _team_on(), mock.patch.object(
                espn_client, "db_store",
                mock.Mock(enabled=mock.Mock(side_effect=RuntimeError("boom")))):
            self.assertIsNone(
                espn_client.mlb_warehouse_team_stats("baseball", "X", 10))

    def test_team_defense_gated_and_fail_open(self):
        with _team_off():
            self.assertIsNone(espn_client.mlb_warehouse_team_defense("baseball"))
        with _team_on(), \
             mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=True))), \
             mock.patch("mlb_warehouse.get_team_defense",
                        return_value={"New York Yankees": 4.0}):
            self.assertEqual(
                espn_client.mlb_warehouse_team_defense("baseball"),
                {"New York Yankees": 4.0})
        with _team_on():                                  # non-baseball → None
            self.assertIsNone(espn_client.mlb_warehouse_team_defense("basketball"))


if __name__ == "__main__":
    unittest.main()
