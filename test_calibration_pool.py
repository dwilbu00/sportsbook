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
    def _athletes(self):
        # Deliberately NOT in minutes order (1400, 90, 2100, 0, 2400) so the
        # top-N-by-minutes SORT is actually exercised: if the sort were dropped
        # and ESPN's arbitrary order followed, the selection tests would fail.
        return [
            {"id": "3", "name": "Rotation C", "games": 55, "minutes": 1400,
             "avg_minutes": 25.5},
            {"id": "4", "name": "Cameo D", "games": 8, "minutes": 90,
             "avg_minutes": 11.3},   # below min_games -> dropped
            {"id": "2", "name": "Star B", "games": 65, "minutes": 2100,
             "avg_minutes": 32.3},
            {"id": "5", "name": "Zero E", "games": 30, "minutes": 0,
             "avg_minutes": 0.0},    # zero minutes -> dropped
            {"id": "1", "name": "Star A", "games": 70, "minutes": 2400,
             "avg_minutes": 34.3},
        ]

    def test_filters_sorts_limits(self):
        with patch.object(refit_calibration, "list_season_athletes",
                          return_value=self._athletes()), \
                patch.object(refit_calibration, "seed_athlete_id"):
            pool = refit_calibration._nba_player_pool(
                2024, max_players=2, min_games=15)
        # Cameo (8 g) and Zero-minute players excluded; sorted by minutes desc;
        # capped at 2 -> the two highest-minute eligible players.
        self.assertEqual(pool, ["Star A", "Star B"])

    def test_min_games_and_zero_minutes_excluded(self):
        with patch.object(refit_calibration, "list_season_athletes",
                          return_value=self._athletes()), \
                patch.object(refit_calibration, "seed_athlete_id"):
            pool = refit_calibration._nba_player_pool(
                2024, max_players=50, min_games=15)
        self.assertEqual(pool, ["Star A", "Star B", "Rotation C"])
        self.assertNotIn("Cameo D", pool)
        self.assertNotIn("Zero E", pool)

    def test_dedupes_names_keeping_highest_minutes(self):
        # Two rows share a display name; the HIGHER-minutes id must be the one
        # pinned (it decides whose gamelog feeds the fit). Provided out of order
        # so the sort has to place id "1" ahead of id "2".
        dupes = [
            {"id": "2", "name": "Same Name", "games": 60, "minutes": 1800},
            {"id": "1", "name": "Same Name", "games": 70, "minutes": 2400},
        ]
        seed = Mock()
        with patch.object(refit_calibration, "list_season_athletes",
                          return_value=dupes), \
                patch.object(refit_calibration, "seed_athlete_id", seed):
            pool = refit_calibration._nba_player_pool(2024, min_games=15)
        self.assertEqual(pool, ["Same Name"])
        # Pinned exactly once, with the higher-minutes id.
        seed.assert_called_once()
        self.assertEqual(seed.call_args.args[2:], ("Same Name", "1"))

    def test_pins_ids_for_kept_players(self):
        seed = Mock()
        with patch.object(refit_calibration, "list_season_athletes",
                          return_value=self._athletes()), \
                patch.object(refit_calibration, "seed_athlete_id", seed):
            refit_calibration._nba_player_pool(2024, max_players=50,
                                               min_games=15)
        pinned = {call.args[2]: call.args[3] for call in seed.call_args_list}
        self.assertEqual(pinned, {"Star A": "1", "Star B": "2",
                                  "Rotation C": "3"})

    def test_empty_source_returns_empty(self):
        with patch.object(refit_calibration, "list_season_athletes",
                          return_value=[]), \
                patch.object(refit_calibration, "seed_athlete_id"):
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
