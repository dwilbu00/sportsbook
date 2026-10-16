"""One-off edit script (ESPN-for-NBA retirement, phase B tests): delete tests of the removed NBA
switches; move tests of SHARED ESPN code (still used by NFL/NHL) off basketball."""
ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


edit("test_nba_live.py", [(
    '''    def test_kill_switch_disables(self):
        with patch("nba_stats_source.live_from_sdv", return_value=False):
            self.assertIsNone(
                espn_client._nba_warehouse_history("Star Guard", "player_points", 5))

''', "")])

edit("test_nba_team_model.py", [
    ('''        self.ns = _app_funcs("_nba_live_warehouse", "_resolve_team_dim",
                             "_fetch_team_schedule")''',
     '''        self.ns = _app_funcs("_resolve_team_dim", "_fetch_team_schedule")'''),
    ('''        with patch("nba_stats_source.live_from_sdv", return_value=True), \\
             patch("nba_warehouse.resolve_team_id", return_value=1610612747):''',
     '''        with patch("nba_warehouse.resolve_team_id", return_value=1610612747):'''),
    ('''    def test_nba_kill_switch_falls_back_to_espn(self):
        with patch("nba_stats_source.live_from_sdv", return_value=False):
            d = self.ns["_resolve_team_dim"]("basketball_nba", "Los Angeles Lakers", {})
        self.assertEqual(d["id"], "espn")
''', '''    def test_nba_never_uses_espn(self):
        # ESPN retired for NBA: the ESPN find_team stub is never consulted
        with patch("nba_warehouse.resolve_team_id", return_value=None):
            self.assertIsNone(self.ns["_resolve_team_dim"]("basketball_nba", "Lakers", {}))
'''),
    ('''        with patch("nba_stats_source.live_from_sdv", return_value=True), \\
             patch("nba_warehouse.resolve_team_id", return_value=None):''',
     '''        with patch("nba_warehouse.resolve_team_id", return_value=None):'''),
    ('''        with patch("nba_stats_source.live_from_sdv", return_value=True), \\
             patch("nba_team_stats.team_stats",''',
     '''        with patch("nba_team_stats.team_stats",'''),
])

edit("test_nba_sweep.py", [(
    '''class SweepModeTests(unittest.TestCase):
    def tearDown(self):
        db_store.configure_engine(None)
        os.environ.pop("ODI_NBA_SWEEP_SDV", None)

    def test_off_when_sql_disabled(self):
        db_store.configure_engine(None)
        self.assertFalse(backtest._nba_sweep_mode("basketball"))

    def test_on_when_enabled(self):
        db_store.configure_engine("sqlite://")
        self.assertTrue(backtest._nba_sweep_mode("basketball"))

    def test_kill_switch(self):
        db_store.configure_engine("sqlite://")
        os.environ["ODI_NBA_SWEEP_SDV"] = "0"
        self.assertFalse(backtest._nba_sweep_mode("basketball"))

    def test_non_basketball_off(self):
        db_store.configure_engine("sqlite://")
        self.assertFalse(backtest._nba_sweep_mode("baseball"))
        self.assertFalse(backtest._nba_sweep_mode("football"))''',
    '''class SweepModeTests(unittest.TestCase):
    """ESPN retired for NBA: basketball always sweeps off the warehouse (mirror-first, so even
    with SQL off); no kill switch."""

    def tearDown(self):
        db_store.configure_engine(None)

    def test_basketball_always_warehouse(self):
        db_store.configure_engine(None)
        self.assertTrue(backtest._nba_sweep_mode("basketball"))
        db_store.configure_engine("sqlite://")
        self.assertTrue(backtest._nba_sweep_mode("basketball"))

    def test_non_basketball_off(self):
        self.assertFalse(backtest._nba_sweep_mode("baseball"))
        self.assertFalse(backtest._nba_sweep_mode("football"))''')])

edit("test_gamelog_store.py", [(
    '''    def test_get_player_stat_history_routes_to_sql(self):
        # NBA still uses the durable SQL gamelog store (MLB is warehouse-only post-P4b).
        rows = [_nba_row("2026-01-15T00:00:00.000+00:00", pts=25.0),
                _nba_row("2026-01-13T00:00:00.000+00:00", pts=18.0)]
        athlete = {"id": "555", "name": "Star", "team_id": "10"}
        with patch.object(espn_client, "search_athlete", return_value=athlete), \\
             patch.object(espn_client, "get_athlete_gamelog",
                          return_value=rows) as gl:
            hist = espn_client.get_player_stat_history(
                "basketball", "nba", "Star", "player_points", n=20)''',
    '''    def test_get_player_stat_history_routes_to_sql(self):
        # NFL (the ESPN fallback path) uses the durable SQL gamelog store; MLB and NBA are
        # warehouse-only.
        rows = [_nfl_row("2026-01-15T00:00:00.000+00:00", yds=250.0),
                _nfl_row("2026-01-13T00:00:00.000+00:00", yds=180.0)]
        athlete = {"id": "555", "name": "Star", "team_id": "10"}
        with patch.object(espn_client, "search_athlete", return_value=athlete), \\
             patch.object(espn_client, "get_athlete_gamelog",
                          return_value=rows) as gl:
            hist = espn_client.get_player_stat_history(
                "football", "nfl", "Star", "player_pass_yds", n=20)'''),
    ('''            espn_client.get_player_stat_history(
                "basketball", "nba", "Star", "player_points", n=20)
            self.assertGreaterEqual(after_first, 1)''',
     '''            espn_client.get_player_stat_history(
                "football", "nfl", "Star", "player_pass_yds", n=20)
            self.assertGreaterEqual(after_first, 1)'''),
    ('''        self.assertTrue(hist["found"])
        self.assertEqual(hist["values"], [25.0, 18.0])
        self.assertEqual(hist["team_id"], "10")''',
     '''        self.assertTrue(hist["found"])
        self.assertEqual(hist["values"], [250.0, 180.0])
        self.assertEqual(hist["team_id"], "10")'''),
])

edit("test_mlb_history_flip.py", [(
    '''    def test_non_baseball_warehouse_miss_still_falls_open(self):
        # Sport-scoping: the P2.5b fail-closed guard is baseball-only. A basketball
        # miss still falls open to the ESPN path (other sports stay byte-identical).''',
    '''    def test_non_baseball_warehouse_miss_still_falls_open(self):
        # Sport-scoping: the P2.5b fail-closed guard is baseball-only. A football miss
        # still falls open to the ESPN path (NBA is warehouse-only + fails closed too).'''),
    ('''            r = espn_client.get_player_stat_history(
                "basketball", "nba", "X", "player_points", n=10)
        self.assertFalse(r["found"])
        sa.assert_called_once()                # non-baseball still consults ESPN''',
     '''            r = espn_client.get_player_stat_history(
                "football", "nfl", "X", "player_pass_yds", n=10)
        self.assertFalse(r["found"])
        sa.assert_called_once()                # NFL still consults ESPN'''),
])

edit("test_p2_finish.py", [(
    '''            espn_client.get_player_stat_history(
                "basketball", "nba", "Some Player", "player_points",
                team_ids=["1", "2"])''',
    '''            espn_client.get_player_stat_history(
                "football", "nfl", "Some Player", "player_pass_yds",
                team_ids=["1", "2"])''')])

edit("test_p3_warehouse_sweep.py", [(
    '''        with mock.patch.object(backtest, "cached_athlete_id",
                               return_value="e1") as caid, \\
             mock.patch.object(backtest, "cached_gamelog",
                               return_value=[{"PTS": 20.0, "game_date": "2026-01-01"}]), \\
             mock.patch("mlb_warehouse.get_calib_gamelogs_bulk") as gcg:
            backtest.fetch_player_data("basketball", "nba", ["Star"])''',
    '''        with mock.patch.object(backtest, "cached_athlete_id",
                               return_value="e1") as caid, \\
             mock.patch.object(backtest, "cached_gamelog",
                               return_value=[{"passingYards": 250.0,
                                              "game_date": "2026-01-01"}]), \\
             mock.patch("mlb_warehouse.get_calib_gamelogs_bulk") as gcg:
            backtest.fetch_player_data("football", "nfl", ["Star"])''')])

edit("test_modeling.py", [
    ('''        with patch.object(espn_client, "search_athlete", return_value={
                "id": "3934672", "name": "Jalen Brunson", "team_id": "18",
        }), patch.object(espn_client, "get_athlete_gamelog", return_value=[
                {"PTS": 25.0, "opponent": "Indiana Pacers",
                 "is_home": False, "team_id": None},
        ]):
            history = espn_client.get_player_stat_history(
                "basketball", "nba", "Jalen Brunson", "player_points")''',
     '''        with patch.object(espn_client, "search_athlete", return_value={
                "id": "3916387", "name": "Josh Allen", "team_id": "2",
        }), patch.object(espn_client, "get_athlete_gamelog", return_value=[
                {"passingYards": 250.0, "opponent": "Miami Dolphins",
                 "is_home": False, "team_id": None},
        ]):
            history = espn_client.get_player_stat_history(
                "football", "nfl", "Josh Allen", "player_pass_yds")'''),
    ('''        self.assertTrue(history["found"])
        self.assertEqual(history["team_id"], "18")''',
     '''        self.assertTrue(history["found"])
        self.assertEqual(history["team_id"], "2")'''),
    ('''        # NBA/NFL/NHL grading refreshes the ESPN gamelog (ttl_hours=6). MLB is
        # warehouse+statsapi only in P6 and never reaches this ESPN path, so the
        # refresh behavior is exercised via basketball.
        rows = [{
            "ts": "2024-04-01T10:00:00Z",
            "sport_key": "basketball_nba",''',
     '''        # NHL (and the NFL fallback) grading refreshes the ESPN gamelog (ttl_hours=6).
        # MLB and NBA are warehouse-only and never reach this ESPN path, so the refresh
        # behavior is exercised via hockey.
        rows = [{
            "ts": "2024-04-01T10:00:00Z",
            "sport_key": "icehockey_nhl",'''),
    ('''            resolved = recalibration.resolve_pending_outcomes("basketball_nba")

        self.assertEqual(resolved, 1)
        self.assertTrue(rows[0]["resolved"])
        gamelog.assert_called_once_with(
            "basketball", "nba", "123", ttl_hours=6)''',
     '''            resolved = recalibration.resolve_pending_outcomes("icehockey_nhl")

        self.assertEqual(resolved, 1)
        self.assertTrue(rows[0]["resolved"])
        gamelog.assert_called_once_with(
            "hockey", "nhl", "123", ttl_hours=6)'''),
])

edit("test_postponed_grading.py", [
    ('''    slippage (same start time) still resolves. This is the NBA/NFL/NHL grading
    path (MLB is warehouse+statsapi only in P6 and never reaches ESPN), so it's
    exercised via basketball."""''',
     '''    slippage (same start time) still resolves. This is the NHL (and NFL-fallback) grading
    path (MLB and NBA are warehouse-only and never reach ESPN), so it's exercised via
    hockey."""'''),
    ('''            return recalibration.resolve_one_prop(
                "basketball_nba", "P1", "player_points", 0.5, game_date, commence)''',
     '''            return recalibration.resolve_one_prop(
                "icehockey_nhl", "P1", "player_points", 0.5, game_date, commence)'''),
])
print("ok")
