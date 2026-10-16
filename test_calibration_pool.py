"""Tests for the data-driven NBA calibration pool (P2 survivorship-bias fix).

Before this change `refit_calibration.refit_sport` fell back to a hand-picked
18-superstar list for NBA (`DEFAULT_STARTERS["nba"]`) — survivorship-biased and
containing no role players / DNP games. It now builds a usage-representative pool
from ESPN's season `byathlete` statistics (top-N by minutes), mirroring the MLB
`_mlb_player_pool`. These tests cover the three new pieces:

  * espn_client.list_season_athletes — paginated, name-mapped parse, fail-closed
  * refit_calibration._nba_player_pool — filter / sort / limit / dedup / id-pin
  * espn_cache.seed_athlete_id — pins an authoritative id so cached_athlete_id
    resolves without the lossy search_athlete first-name-match
"""

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

import requests

import espn_cache
import espn_client
import refit_calibration


# ── fixtures for the byathlete response shape ────────────────────────────────
def _entry(aid, name, games, avg_min, total_min, pos="G",
           general_names=("gamesPlayed", "avgMinutes", "minutes")):
    """One athlete row. `values` are aligned positionally to `general_names`."""
    lookup = {"gamesPlayed": games, "avgMinutes": avg_min, "minutes": total_min}
    values = [lookup[n] for n in general_names]
    return {
        "athlete": {"id": aid, "displayName": name,
                    "position": {"abbreviation": pos}},
        "categories": [{"name": "general", "values": values}],
    }


def _page(entries, pages, general_names=("gamesPlayed", "avgMinutes", "minutes")):
    return {
        "categories": [{"name": "general", "names": list(general_names)}],
        "athletes": entries,
        "pagination": {"pages": pages},
    }


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class ListSeasonAthletesTests(unittest.TestCase):
    def test_parses_and_paginates(self):
        pages = [
            _Resp(_page([_entry("1", "Luka Doncic", 70, 37.5, 2624),
                         _entry("2", "Jayson Tatum", 74, 36.9, 2732)], pages=2)),
            _Resp(_page([_entry("3", "Role Player", 40, 12.0, 480)], pages=2)),
        ]
        with patch("espn_client.requests.get", side_effect=pages) as get:
            out = espn_client.list_season_athletes("basketball", "nba", 2024)
        self.assertEqual(get.call_count, 2)
        self.assertEqual([a["name"] for a in out],
                         ["Luka Doncic", "Jayson Tatum", "Role Player"])
        luka = out[0]
        self.assertEqual(luka["id"], "1")
        self.assertEqual(luka["games"], 70.0)
        self.assertEqual(luka["minutes"], 2624.0)
        self.assertEqual(luka["avg_minutes"], 37.5)
        self.assertEqual(luka["position"], "G")

    def test_columns_mapped_by_name_not_index(self):
        # A reordered `names` list must not change which value is read.
        order = ("minutes", "gamesPlayed", "avgMinutes")
        payload = _page([_entry("9", "Reordered", 55, 30.0, 1650,
                                general_names=order)],
                        pages=1, general_names=order)
        with patch("espn_client.requests.get", return_value=_Resp(payload)):
            out = espn_client.list_season_athletes("basketball", "nba", 2024)
        self.assertEqual(out[0]["games"], 55.0)
        self.assertEqual(out[0]["minutes"], 1650.0)
        self.assertEqual(out[0]["avg_minutes"], 30.0)

    def test_incomplete_pagination_fails_closed_to_empty(self):
        # Page 1 ok (of 3), page 2 errors mid-pagination. The byathlete feed is
        # ranking-biased, so a page-1-only subset must NOT be returned — the
        # whole fetch fails closed to [] so the caller aborts rather than fitting
        # on a truncated, biased sample.
        pages = [
            _Resp(_page([_entry("1", "Kept", 70, 30.0, 2100)], pages=3)),
            requests.exceptions.ConnectionError("reset"),
        ]
        stderr = io.StringIO()
        with patch("espn_client.requests.get", side_effect=pages), \
                redirect_stderr(stderr):
            out = espn_client.list_season_athletes("basketball", "nba", 2024)
        self.assertEqual(out, [])
        log = stderr.getvalue()
        self.assertIn("list_season_athletes failed", log)
        self.assertIn("incomplete", log)

    def test_empty_athletes_page_stops(self):
        # pages=5 but the first page is empty -> stop after one request.
        with patch("espn_client.requests.get",
                   return_value=_Resp(_page([], pages=5))) as get:
            out = espn_client.list_season_athletes("basketball", "nba", 2024)
        self.assertEqual(out, [])
        self.assertEqual(get.call_count, 1)

    def test_entry_without_id_or_name_skipped(self):
        payload = _page([
            {"athlete": {"displayName": "No Id"},
             "categories": [{"name": "general", "values": [70, 30, 2100]}]},
            _entry("2", "Valid", 70, 30.0, 2100),
        ], pages=1)
        with patch("espn_client.requests.get", return_value=_Resp(payload)):
            out = espn_client.list_season_athletes("basketball", "nba", 2024)
        self.assertEqual([a["name"] for a in out], ["Valid"])

    def test_respects_max_pages_cap_and_fails_closed(self):
        # Every page reports 99 total pages; max_pages bounds the loop AND, since
        # the last page is never reached, the fetch is incomplete -> [].
        stderr = io.StringIO()
        with patch("espn_client.requests.get",
                   return_value=_Resp(_page([_entry("1", "P", 70, 30.0, 2100)],
                                            pages=99))) as get, \
                redirect_stderr(stderr):
            out = espn_client.list_season_athletes(
                "basketball", "nba", 2024, max_pages=2)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(out, [])
        self.assertIn("incomplete", stderr.getvalue())

    def test_non_numeric_stat_values_coerce_to_none(self):
        # ESPN can emit non-numeric markers ("--", "", null). They must coerce to
        # None (not raise), so downstream filtering drops the row cleanly.
        payload = _page([{
            "athlete": {"id": "7", "displayName": "Marked",
                        "position": {"abbreviation": "F"}},
            "categories": [{"name": "general", "values": ["--", "", None]}],
        }], pages=1)
        with patch("espn_client.requests.get", return_value=_Resp(payload)):
            out = espn_client.list_season_athletes("basketball", "nba", 2024)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["games"])
        self.assertIsNone(out[0]["minutes"])
        self.assertIsNone(out[0]["avg_minutes"])


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
