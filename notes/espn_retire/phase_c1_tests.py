"""One-off edit script (ESPN retirement phase C1 tests): the NBA refit pool now ranks warehouse
players by minutes; the calibration join no longer gates on SQL (mirror-first, no ESPN)."""
ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


p = ROOT + "test_calibration_pool.py"
s = open(p, encoding="utf-8").read()
start = s.index("class NbaPlayerPoolTests(unittest.TestCase):")
end = s.index("class SeedAthleteIdTests")
new_cls = '''class NbaPlayerPoolTests(unittest.TestCase):
    """The NBA calibration pool = top-N warehouse players by total PLAYED minutes among those
    with >= min_games played games in the season (stats.nba.com; ESPN retired 2026-10)."""

    def _bulk(self):
        def games(n, mins, season=2024, played=True):
            return [{"season": season, "played": played, "MIN": mins} for _ in range(n)]
        # Deliberately NOT in minutes order so the sort is exercised.
        return {
            "rotation c": games(55, 25.5),                  # 1402.5 min
            "cameo d": games(8, 11.3),                      # below min_games -> dropped
            "star b": games(65, 32.3),                      # 2099.5
            "zero e": games(30, 0.0),                       # zero minutes -> no played games
            "star a": games(70, 34.3),                      # 2401
            "dnp f": games(40, 30.0, played=False),         # never played -> dropped
            "old g": games(70, 35.0, season=2023),          # other season -> dropped
        }

    def _pool(self, **kw):
        with patch("nba_warehouse.calib_gamelogs_bulk", return_value=self._bulk()):
            return refit_calibration._nba_player_pool(2024, **kw)

    def test_filters_sorts_limits(self):
        self.assertEqual(self._pool(max_players=2, min_games=15), ["Star A", "Star B"])

    def test_min_games_zero_minutes_dnp_and_season_excluded(self):
        pool = self._pool(max_players=50, min_games=15)
        self.assertEqual(pool, ["Star A", "Star B", "Rotation C"])
        for gone in ("Cameo D", "Zero E", "Dnp F", "Old G"):
            self.assertNotIn(gone, pool)

    def test_names_resolve_back_to_the_warehouse_key(self):
        import db_store
        for name in self._pool(max_players=50, min_games=15):
            self.assertIn(db_store.normalize_name(name), self._bulk())

    def test_empty_source_returns_empty(self):
        with patch("nba_warehouse.calib_gamelogs_bulk", return_value={}):
            self.assertEqual(refit_calibration._nba_player_pool(2024), [])


'''
s = s[:start] + new_cls + s[end:]
open(p, "w", encoding="utf-8").write(s)

edit("test_nba_calibration_join.py", [(
    '''    def test_warehouse_disabled_returns_none(self):
        db_store.configure_engine(None)                  # warehouse off
        try:
            self.assertIsNone(bkc._nba_grade_book_lines(
                [_line("Shooter Guy", "player_points", "2024-01-23")]))
        finally:
            db_store.configure_engine("sqlite://")       # restore for tearDown''',
    '''    def test_no_sql_no_mirror_grades_nothing_and_never_espn(self):
        # ESPN retired 2026-10: with SQL off and no mirror there's simply no data — an empty
        # result, and the join never falls back to the ESPN gamelog.
        db_store.configure_engine(None)                  # warehouse off
        try:
            with patch("book_line_calibration.cached_gamelog") as espn:
                out = bkc.join_book_lines_to_actuals(
                    [_line("Shooter Guy", "player_points", "2024-01-23")], "basketball", "nba")
            self.assertEqual(out, [])
            espn.assert_not_called()
        finally:
            db_store.configure_engine("sqlite://")       # restore for tearDown''')])
print("ok")
