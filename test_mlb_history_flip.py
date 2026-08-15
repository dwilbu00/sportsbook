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
        # P6: identity now resolves through mlb_starters.resolve_mlbam_id (game-context
        # → season roster → role-verified SFBB), which returns (mlbam_id, is_pitcher);
        # mlb_id=None models an unresolvable/ambiguous name.
        resolved = (mlb_id, False) if mlb_id else None
        with mock.patch.object(
                espn_client, "db_store",
                mock.Mock(enabled=mock.Mock(return_value=sql_enabled))), \
             mock.patch("mlb_starters.resolve_mlbam_id", return_value=resolved), \
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
        with _flag_on():                       # HR has no odds market / not in spec
            self.assertIsNone(self._call(prop="batter_home_runs"))

    def test_new_batter_markets_are_servable(self):
        # TB/RBI are now in _ACTUAL_STAT_SPEC, so the flip gate lets them through to
        # the warehouse reader (HR above stays gated out).
        with _flag_on():
            self.assertEqual(self._call(prop="batter_total_bases"), WH_DICT)
            self.assertEqual(self._call(prop="batter_rbis"), WH_DICT)

    def test_ambiguous_or_unknown_name_returns_none(self):
        with _flag_on():
            self.assertIsNone(self._call(mlb_id=None))

    def test_namesake_resolved_game_context_first(self):
        # P6: the resolver receives the game's two teams AND the prop_key + today's
        # posted lineup/probables, so a namesake (Max Muncy / Luis Garcia Jr.) binds
        # to the id that actually appears in this game (game-context-first) rather than
        # the drift-prone SFBB cross-map, and it resolves off the warehouse.
        lineup = {"players": {}, "home_confirmed": False, "away_confirmed": False}
        probs = {}
        with _flag_on(), mock.patch.object(
                espn_client, "db_store",
                mock.Mock(enabled=mock.Mock(return_value=True))), \
             mock.patch("mlb_starters.resolve_mlbam_id",
                        return_value=("592450", False)) as m_id, \
             mock.patch("mlb_warehouse.get_player_history", return_value=WH_DICT):
            out = espn_client._mlb_warehouse_history(
                "baseball", "Max Muncy", "batter_hits", 20,
                teams=["Athletics", "New York Yankees"],
                confirmed_lineup=lineup, probable_starters=probs)
        self.assertEqual(out, WH_DICT)
        # teams + prop_key + game context all threaded into the single resolver.
        _, kwargs = m_id.call_args
        self.assertEqual(kwargs.get("teams"), ["Athletics", "New York Yankees"])
        self.assertEqual(kwargs.get("prop_key"), "batter_hits")
        self.assertIs(kwargs.get("confirmed_lineup"), lineup)
        self.assertIs(kwargs.get("probable_starters"), probs)

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
             mock.patch("mlb_starters.resolve_mlbam_id",
                        return_value=("592450", False)), \
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
        # Parity-harness exemption: allow_warehouse=False skips the warehouse branch
        # AND the P2.5b baseball fail-closed guard, so the TRUE ESPN side is still
        # reached for diffing (search_athlete consulted).
        with mock.patch.object(espn_client, "_mlb_warehouse_history",
                               return_value=WH_DICT) as wh, \
             mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=False))), \
             mock.patch.object(espn_client, "search_athlete",
                               return_value=None) as sa:
            r = espn_client.get_player_stat_history(
                "baseball", "mlb", "Aaron Judge", "batter_hits", n=10,
                allow_warehouse=False)
        wh.assert_not_called()                 # warehouse never consulted
        sa.assert_called_once()                # P2.5b guard exempt -> ESPN reached
        self.assertFalse(r["found"])           # fell to ESPN (no athlete resolved)

    def test_baseball_warehouse_miss_fails_closed_not_espn(self):
        # P2.5b: on the LIVE path (allow_warehouse defaults True) a baseball warehouse
        # MISS fails CLOSED to an empty history -- it must NOT fall open to ESPN. The
        # ESPN name lookup + gamelog fetch are asserted un-called: the warehouse is the
        # sole MLB source, and an empty history drops the prop rather than serving a
        # wrong ESPN-sourced value.
        with mock.patch.object(espn_client, "_mlb_warehouse_history",
                               return_value=None), \
             mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=True))), \
             mock.patch.object(espn_client, "search_athlete") as sa, \
             mock.patch.object(espn_client, "get_athlete_gamelog") as gg:
            r = espn_client.get_player_stat_history(
                "baseball", "mlb", "X", "batter_hits", n=10)
        self.assertFalse(r["found"])
        self.assertEqual(r["values"], [])
        sa.assert_not_called()                 # ESPN name resolution never consulted
        gg.assert_not_called()                 # ESPN gamelog never fetched

    def test_non_baseball_warehouse_miss_still_falls_open(self):
        # Sport-scoping: the P2.5b fail-closed guard is baseball-only. A basketball
        # miss still falls open to the ESPN path (other sports stay byte-identical).
        with mock.patch.object(espn_client, "_mlb_warehouse_history",
                               return_value=None), \
             mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=False))), \
             mock.patch.object(espn_client, "search_athlete",
                               return_value=None) as sa:
            r = espn_client.get_player_stat_history(
                "basketball", "nba", "X", "player_points", n=10)
        self.assertFalse(r["found"])
        sa.assert_called_once()                # non-baseball still consults ESPN


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


class EnforceIdentityTests(unittest.TestCase):
    """P4 fail-closed identity enforcement (env ODI_MLB_ENFORCE_IDENTITY): an MLB
    player the resolver can't uniquely pin gets NO candidate/prediction when on;
    default OFF keeps the P3 shadow posture; a slate-level circuit breaker fails OPEN
    on a systemic (all-unresolved) failure. entity_resolver + refit/log mocked."""

    def _prop_data(self, players):
        return {"commence_time": "2026-08-10T23:10:00Z",
                "home_team": "New York Yankees", "away_team": "Boston Red Sox",
                "game_id": "evt-enf",
                "props": {"batter_hits": {p: {
                    "line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
                    "over_price": -110, "under_price": -110,
                    "over_book": "DK", "under_book": "DK"} for p in players}}}

    def _run(self, resolved_map, enforce, sport_key="baseball_mlb", hiccup=False):
        def _resolve(name, *a, **k):
            if hiccup:                       # _resolve_ident catches → None (fail-open)
                raise RuntimeError("resolver down")
            return ({"resolved": True, "mlb_player_id": "1", "game_pk": 700}
                    if resolved_map.get(name) else
                    {"resolved": False, "mlb_player_id": None, "game_pk": None})
        env = {props._MLB_ENFORCE_IDENTITY_ENV: "1" if enforce else ""}
        with mock.patch.dict(os.environ, env), \
             mock.patch("entity_resolver.resolve", side_effect=_resolve), \
             mock.patch.object(props, "maybe_auto_refit"), \
             mock.patch.object(props, "load_recalibration", return_value={}), \
             mock.patch.object(props, "log_prediction_rows"):
            cands = props.analyze_player_props_value(
                self._prop_data(list(resolved_map)), {}, threshold_pct=1.0,
                sport_key=sport_key)
        return {c["player"] for c in cands}

    def test_unresolved_dropped_when_enforced(self):
        # 1 of 3 unpinned (33% < 50%) → enforce drops only the unresolved one
        self.assertEqual(
            self._run({"A": True, "B": True, "C": False}, enforce=True), {"A", "B"})

    def test_kept_when_not_enforced(self):                    # P3 shadow (default)
        self.assertEqual(
            self._run({"A": True, "C": False}, enforce=False), {"A", "C"})

    def test_circuit_breaker_all_unresolved_fails_open(self):
        # systemic: 100% unpinned → fail OPEN, keep the whole slate
        self.assertEqual(
            self._run({"A": False, "B": False, "C": False}, enforce=True),
            {"A", "B", "C"})

    def test_non_mlb_never_enforced(self):
        self.assertEqual(
            self._run({"A": False}, enforce=True, sport_key=None), {"A"})

    def test_resolver_hiccup_fails_open(self):
        self.assertEqual(
            self._run({"A": False, "B": False}, enforce=True, hiccup=True),
            {"A", "B"})

    def test_breaker_counts_name_prop_key_pairs(self):
        # The systemic-failure breaker counts the (name, prop_key) unit (the same unit
        # the per-row drop enforces), since the resolver is role-partitioned. Slate:
        # resolved star A in 3 markets + unresolved B in 1 → pairs 1/4 = 25% < 50% →
        # breaker HOLDS → enforce → B dropped (a distinct-NAMES count would be 1/2 =
        # 50% → fail-open → keep B, which is NOT what happens).
        od = {"line": 0.5, "over_implied": 0.5, "under_implied": 0.5,
              "over_price": -110, "under_price": -110,
              "over_book": "DK", "under_book": "DK"}
        prop_data = {
            "commence_time": "2026-08-10T23:10:00Z",
            "home_team": "New York Yankees", "away_team": "Boston Red Sox",
            "game_id": "evt-unit",
            "props": {"batter_hits": {"A": dict(od), "B": dict(od)},
                      "batter_total_bases": {"A": dict(od)},
                      "batter_rbis": {"A": dict(od)}}}
        resolved = {"A": True, "B": False}

        def _resolve(name, *a, **k):
            return ({"resolved": True, "mlb_player_id": "1", "game_pk": 700}
                    if resolved.get(name) else
                    {"resolved": False, "mlb_player_id": None, "game_pk": None})

        with mock.patch.dict(os.environ, {props._MLB_ENFORCE_IDENTITY_ENV: "1"}), \
             mock.patch("entity_resolver.resolve", side_effect=_resolve), \
             mock.patch.object(props, "maybe_auto_refit"), \
             mock.patch.object(props, "load_recalibration", return_value={}), \
             mock.patch.object(props, "log_prediction_rows"):
            kept = {c["player"] for c in props.analyze_player_props_value(
                prop_data, {}, threshold_pct=1.0, sport_key="baseball_mlb")}
        self.assertIn("A", kept)
        self.assertNotIn("B", kept)                        # pairs 1/4 < 50% → enforced


class GateStatusTests(unittest.TestCase):
    """mlb_warehouse_gate_status reflects the live env flags + SQL state — the
    operator's only signal that a flip took effect (predictions record no source)."""

    def _status(self, env, sql=True):
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(espn_client.db_store, "enabled", return_value=sql):
            return espn_client.mlb_warehouse_gate_status()

    def test_all_off_by_default(self):
        s = self._status({})
        self.assertEqual(
            s, {"history": False, "team": False, "calib": False,
                "enforce_identity": False, "sql": True})

    def test_each_flag_reads_its_env_key(self):
        self.assertTrue(self._status({"ODI_MLB_WAREHOUSE_HIST": "1"})["history"])
        self.assertTrue(self._status({"ODI_MLB_WAREHOUSE_TEAM": "true"})["team"])
        self.assertTrue(self._status({"ODI_MLB_WAREHOUSE_CALIB": "on"})["calib"])
        self.assertTrue(
            self._status({"ODI_MLB_ENFORCE_IDENTITY": "yes"})["enforce_identity"])

    def test_falsey_values_stay_off(self):
        self.assertFalse(self._status({"ODI_MLB_WAREHOUSE_HIST": "0"})["history"])
        self.assertFalse(self._status({"ODI_MLB_WAREHOUSE_HIST": "no"})["history"])

    def test_sql_disabled_reported(self):
        self.assertFalse(self._status({}, sql=False)["sql"])


class WarehouseOnlyPropsEspnGuardTests(unittest.TestCase):
    """batter_total_bases / batter_rbis are WAREHOUSE-ONLY: the live ESPN gamelog
    path must return no_history for them even though a RAW ESPN gamelog carries an
    'RBI' label (gamelog_store's slow path returns raw rows). Without the guard an
    uncalibrated RBI over-rate would leak whenever the flag is off / cache is stale."""

    def _gamelog(self):
        return [{"H": 1.0, "RBI": 2.0, "opponent": "BOS", "is_home": True,
                 "game_date": "2026-07-0%d" % d} for d in range(1, 6)]

    def _hist(self, prop):
        with mock.patch.object(espn_client, "db_store",
                               mock.Mock(enabled=mock.Mock(return_value=False))), \
             mock.patch.object(espn_client, "search_athlete",
                               return_value={"id": "1", "team_id": "147"}), \
             mock.patch.object(espn_client, "get_athlete_gamelog",
                               return_value=self._gamelog()):
            return espn_client.get_player_stat_history(
                "baseball", "mlb", "Guy", prop, allow_warehouse=False)

    def test_rbis_not_served_by_espn(self):
        rbi = self._hist("batter_rbis")
        self.assertFalse(rbi["found"])          # guarded: warehouse-only
        self.assertEqual(rbi["values"], [])

    def test_total_bases_not_served_by_espn(self):
        self.assertFalse(self._hist("batter_total_bases")["found"])

    def test_hits_still_served_by_espn(self):   # control: guard is TB/RBI-specific
        hits = self._hist("batter_hits")
        self.assertTrue(hits["found"])
        self.assertEqual(hits["values"][0], 1.0)


SEASON_E2E = 2026


class WarehouseHistoryEndToEndTests(unittest.TestCase):
    """F8: wire the REAL producers (get_confirmed_lineup / get_probable_starters, fed
    a canned StatsAPI /schedule payload) through the REAL, UNMOCKED resolver
    (resolve_mlbam_id) into _mlb_warehouse_history. Every other test above MOCKS the
    resolver, which hides producer↔resolver SHAPE drift: if the lineup/probable dict
    layout the producers emit ever diverged from what _game_context_id reads, those
    tests would still pass while production silently fell open to ESPN. These fail on
    that drift — only the lowest network/DB seams (_get, caches, get_player_history,
    _current_season, SFBB) are stubbed; the identity chain itself runs for real."""

    HOME, AWAY = "New York Yankees", "Boston Red Sox"
    DATE = "2026-08-10"

    def _fake_get(self, path, params):
        # The SAME /schedule endpoint both producers call, branched by requested
        # hydrate (lineups vs probablePitcher) — the real StatsAPI response shape.
        assert path == "schedule", path
        hy = params.get("hydrate")
        if hy == "lineups":
            home_players = ([{"id": 592450, "fullName": "Aaron Judge"}]
                            + [{"id": 600000 + i, "fullName": f"Yankee {i}"}
                               for i in range(1, 9)])
            away_players = [{"id": 610000 + i, "fullName": f"Sock {i}"}
                            for i in range(1, 10)]
            return {"dates": [{"games": [{
                "teams": {"home": {"team": {"name": self.HOME}},
                          "away": {"team": {"name": self.AWAY}}},
                "lineups": {"homePlayers": home_players,
                            "awayPlayers": away_players}}]}]}
        if hy == "probablePitcher":
            return {"dates": [{"games": [{"teams": {
                "home": {"team": {"name": self.HOME, "id": 147},
                         "probablePitcher": {"id": 543037,
                                             "fullName": "Gerrit Cole"}},
                "away": {"team": {"name": self.AWAY, "id": 111},
                         "probablePitcher": {"id": 605483,
                                             "fullName": "Brayan Bello"}}}}]}]}
        return {"dates": []}

    def _producers(self):
        """Run the REAL producers against the canned payload (no network / no cache)."""
        import mlb_starters
        with mock.patch.object(mlb_starters, "_get", side_effect=self._fake_get), \
             mock.patch.object(mlb_starters, "_read_cache", return_value=None), \
             mock.patch.object(mlb_starters, "_write_cache"):
            lineup = mlb_starters.get_confirmed_lineup(self.HOME, self.AWAY, self.DATE)
            probs = mlb_starters.get_probable_starters(self.DATE)
        return lineup, probs

    def _run(self, name, prop_key, lineup, probs, expect_id):
        import mlb_starters

        def _gph(mlb_id, prop_key_, **k):
            # Returns the dict ONLY for the id the real resolver should have produced,
            # so a wrong bind (or an ESPN fall-through) surfaces as None, not WH_DICT.
            return WH_DICT if str(mlb_id) == expect_id else None

        prev_p = mlb_starters._PITCHER_BY_ID_CACHE.get(SEASON_E2E)
        prev_i = mlb_starters._PLAYER_INDEX_CACHE.get(SEASON_E2E)
        # Prime the season caches so tiers 2/3 stay offline: the inverted role index
        # for the batter arm's is_pitcher lookup (the pitcher arm forces True and never
        # consults it), and an EMPTY roster so an unresolvable name falls through to
        # None rather than hitting the network. Tier-1 (lineup/probables) never reads
        # either — this only matters for the absent-name fall-through.
        mlb_starters._PITCHER_BY_ID_CACHE[SEASON_E2E] = {"592450": False}
        mlb_starters._PLAYER_INDEX_CACHE[SEASON_E2E] = {}
        try:
            with _flag_on(), \
                 mock.patch.object(espn_client, "db_store",
                                   mock.Mock(enabled=mock.Mock(return_value=True))), \
                 mock.patch.object(mlb_starters, "_player_id_map",
                                   return_value=None), \
                 mock.patch("mlb_warehouse._current_season",
                            return_value=SEASON_E2E), \
                 mock.patch("mlb_warehouse.get_player_history", side_effect=_gph):
                return espn_client._mlb_warehouse_history(
                    "baseball", name, prop_key, 20, teams=[self.HOME, self.AWAY],
                    confirmed_lineup=lineup, probable_starters=probs)
        finally:
            for cache, prev in ((mlb_starters._PITCHER_BY_ID_CACHE, prev_p),
                                (mlb_starters._PLAYER_INDEX_CACHE, prev_i)):
                if prev is None:
                    cache.pop(SEASON_E2E, None)
                else:
                    cache[SEASON_E2E] = prev

    def test_real_producers_shapes_are_what_the_resolver_reads(self):
        # Guard the contract directly: the real lineup producer emits players keyed by
        # normalized name with a player_id, and the probable producer keys by team.
        lineup, probs = self._producers()
        self.assertTrue(lineup["home_confirmed"] and lineup["away_confirmed"])
        self.assertEqual(lineup["players"]["aaron judge"]["player_id"], 592450)
        self.assertEqual(probs["new york yankees"]["pitcher_id"], 543037)

    def test_batter_binds_off_real_lineup(self):
        lineup, probs = self._producers()
        out = self._run("Aaron Judge", "batter_hits", lineup, probs,
                        expect_id="592450")
        self.assertEqual(out, WH_DICT)           # resolved 592450 off the real lineup

    def test_pitcher_binds_off_real_probables(self):
        lineup, probs = self._producers()
        out = self._run("Gerrit Cole", "pitcher_strikeouts", lineup, probs,
                        expect_id="543037")
        self.assertEqual(out, WH_DICT)           # resolved 543037 off real probables

    def test_absent_name_returns_none(self):
        # A name in neither the real lineup nor probables (and no SFBB) resolves to
        # nothing → _mlb_warehouse_history returns None (a warehouse MISS). Proves the
        # WH_DICT hits above came from a real bind, not an unconditional stub. (Under
        # P2.5b that None now fails CLOSED at the get_player_stat_history level for
        # baseball rather than falling open to ESPN — see GetPlayerStatHistoryBranchTests.)
        lineup, probs = self._producers()
        self.assertIsNone(
            self._run("Ghost Player", "batter_hits", lineup, probs,
                      expect_id="592450"))


if __name__ == "__main__":
    unittest.main()
