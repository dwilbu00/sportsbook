"""Tests for the data-driven NBA calibration pool (P2 survivorship-bias fix).

Before P2 `refit_calibration.refit_sport` fell back to a hand-picked 18-superstar list for
NBA (`DEFAULT_STARTERS["nba"]`) — survivorship-biased and containing no role players / DNP
games. It builds a usage-representative pool (top-N by minutes), mirroring the MLB
`_mlb_player_pool` — from the stats.nba.com warehouse since the 2026-10 ESPN retirement
(the ESPN season-athlete listing it used to call is gone). Covers:

  * refit_calibration._nba_player_pool — filter / sort / limit / season / name round-trip
  * espn_cache.seed_athlete_id — pins an authoritative id so cached_athlete_id
    resolves without the lossy search_athlete first-name-match (shared ESPN id cache)
"""

import tempfile
import unittest
from unittest.mock import Mock, patch

import espn_cache
import refit_calibration


class NbaPlayerPoolTests(unittest.TestCase):
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


class SeedAthleteIdTests(unittest.TestCase):
    def test_pinned_id_resolves_without_calling_search(self):
        search = Mock(side_effect=AssertionError("search_athlete must not run"))
        with tempfile.TemporaryDirectory() as d, \
                patch.object(espn_cache, "CACHE_DIR", d), \
                patch.object(espn_cache, "search_athlete", search):
            espn_cache.seed_athlete_id("basketball", "nba", "Luka Doncic", "1966")
            aid = espn_cache.cached_athlete_id("basketball", "nba", "Luka Doncic")
        self.assertEqual(aid, "1966")
        search.assert_not_called()

    def test_falsy_id_is_noop(self):
        search = Mock(return_value=None)
        with tempfile.TemporaryDirectory() as d, \
                patch.object(espn_cache, "CACHE_DIR", d), \
                patch.object(espn_cache, "search_athlete", search):
            espn_cache.seed_athlete_id("basketball", "nba", "Ghost", None)
            # No pin written -> cached_athlete_id falls through to the search.
            espn_cache.cached_athlete_id("basketball", "nba", "Ghost")
        search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
