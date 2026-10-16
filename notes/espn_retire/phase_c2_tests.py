"""One-off edit script (ESPN retirement phase C2 tests): matchup backtest + props-odds series are
on the warehouse for NBA; the shared ESPN series path is still covered via football."""
ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


edit("test_p3_warehouse_sweep.py", [
    ('''    """run_backtest / run_odds_backtest source MLB team schedules from the warehouse
    (unconditionally post-P4); non-baseball stays on ESPN."""''',
     '''    """run_backtest / run_odds_backtest source MLB team schedules from the StatsAPI warehouse
    (post-P4) and NBA from the stats.nba.com warehouse (ESPN retired 2026-10); NFL/NHL stay on
    ESPN."""'''),
    ('''    def test_run_backtest_nba_uses_espn(self):
        with mock.patch.object(backtest, "_warehouse_team_schedules") as wh, \\
             mock.patch.object(backtest, "get_all_teams", return_value={}) as gat, \\
             mock.patch.object(backtest, "build_schedules", return_value={}):
            backtest.run_backtest("basketball_nba", "basketball", "nba", limit=10,
                                  window=10, variants={"base": {"half_life": 10}},
                                  season_year=2025)
        gat.assert_called_once()
        wh.assert_not_called()''',
     '''    def test_run_backtest_nba_uses_nba_warehouse(self):
        nwh = mock.Mock(return_value=({}, {}))
        with mock.patch.object(backtest, "_nba_warehouse_team_schedules", nwh), \\
             mock.patch.object(backtest, "_warehouse_team_schedules") as wh, \\
             mock.patch.object(backtest, "get_all_teams") as gat:
            backtest.run_backtest("basketball_nba", "basketball", "nba", limit=10,
                                  window=10, variants={"base": {"half_life": 10}},
                                  season_year=2025)
        nwh.assert_called_once()
        gat.assert_not_called()             # no ESPN
        wh.assert_not_called()

    def test_run_backtest_nfl_still_espn(self):
        with mock.patch.object(backtest, "get_all_teams", return_value={}) as gat, \\
             mock.patch.object(backtest, "build_schedules", return_value={}):
            backtest.run_backtest("americanfootball_nfl", "football", "nfl", limit=10,
                                  window=10, variants={"base": {"half_life": 10}},
                                  season_year=2025)
        gat.assert_called_once()'''),
    ('''    """run_props_odds_backtest's _player_stat_series sources MLB player logs from the
    warehouse (role-verified id → get_calib_gamelog); NBA/NFL use ESPN."""''',
     '''    """run_props_odds_backtest's _player_stat_series sources MLB player logs from the
    warehouse (role-verified id → get_calib_gamelog), NBA from the stats.nba.com corpus; NFL
    uses ESPN."""'''),
    ('''    def test_non_baseball_uses_espn(self):
        with mock.patch.object(backtest, "cached_athlete_id",
                               return_value="e1") as caid, \\
             mock.patch.object(backtest, "cached_gamelog",
                               return_value=[{"PTS": 20.0, "game_date": "2026-01-01",
                                              "completed": True}]), \\
             mock.patch("mlb_warehouse.get_calib_gamelog") as gcg:
            ser = backtest._player_stat_series(
                "basketball", "nba", "Star", "player_points")
        caid.assert_called_once()
        gcg.assert_not_called()
        self.assertEqual(ser, [("2026-01-01", 20.0)])''',
     '''    def test_nfl_uses_espn(self):
        with mock.patch.object(backtest, "cached_athlete_id",
                               return_value="e1") as caid, \\
             mock.patch.object(backtest, "cached_gamelog",
                               return_value=[{"passingYards": 250.0,
                                              "game_date": "2026-01-01",
                                              "completed": True}]), \\
             mock.patch("mlb_warehouse.get_calib_gamelog") as gcg:
            ser = backtest._player_stat_series(
                "football", "nfl", "Star", "player_pass_yds")
        caid.assert_called_once()
        gcg.assert_not_called()
        self.assertEqual(ser, [("2026-01-01", 250.0)])

    def test_nba_uses_warehouse_corpus(self):
        backtest._NBA_SERIES.clear()
        bulk = {"jimmy butler iii": [
            {"game_date": "2025-01-03", "played": True, "points": 20.0, "rebounds": 5.0},
            {"game_date": "2025-01-01", "played": True, "points": 30.0, "rebounds": 7.0},
            {"game_date": "2025-01-02", "played": False, "points": 0.0, "rebounds": 0.0}]}
        try:
            with mock.patch("nba_warehouse.calib_gamelogs_bulk", return_value=bulk), \\
                 mock.patch.object(backtest, "cached_athlete_id") as caid:
                pts = backtest._player_stat_series("basketball", "nba", "Jimmy Butler",
                                                   "player_points")
                pr = backtest._player_stat_series("basketball", "nba", "Jimmy Butler",
                                                  "player_points_rebounds")
            caid.assert_not_called()                     # no ESPN
            self.assertEqual(pts, [("2025-01-01", 30.0), ("2025-01-03", 20.0)])  # DNP dropped
            self.assertEqual(pr, [("2025-01-01", 37.0), ("2025-01-03", 25.0)])   # combo sum
        finally:
            backtest._NBA_SERIES.clear()'''),
])
print("ok")
