"""Tests for the MLB StatsAPI medallion P1 layer (mlb_warehouse + parity harness).

Covers: SchemaParity (Table columns ↔ column SPECs), the pure parse/derive
functions off fixture StatsAPI payloads, reconcile-based idempotent ingestion on
SQLite, the standings snapshot fact, the bronze lifecycle, and the pure parity
diff core. No network — the fetchers are monkeypatched to return fixtures.
"""

import copy
import json
import unittest
from unittest import mock

from sqlalchemy import select

import db_store
import mlb_warehouse
import mlb_warehouse_parity as parity


# ─────────────────────────────────────────────────────────────────── fixtures
TEAMS = {"teams": [
    {"id": 147, "name": "New York Yankees", "abbreviation": "NYY",
     "sport": {"id": 1}, "league": {"id": 103}, "division": {"id": 201}},
    {"id": 111, "name": "Boston Red Sox", "abbreviation": "BOS",
     "sport": {"id": 1}, "league": {"id": 103}, "division": {"id": 201}},
    {"id": 119, "name": "Los Angeles Dodgers", "abbreviation": "LAD",
     "sport": {"id": 1}, "league": {"id": 104}, "division": {"id": 203}},
    {"id": 137, "name": "San Francisco Giants", "abbreviation": "SF",
     "sport": {"id": 1}, "league": {"id": 104}, "division": {"id": 203}},
    {"id": 159, "name": "AL All-Stars", "sport": {"id": 51}},   # non-MLB → filtered
]}

SCHEDULE = {"dates": [{
    "date": "2024-07-04",
    "games": [
        {
            "gamePk": 745804,
            "gameDate": "2024-07-04T17:10:00Z",
            "officialDate": "2024-07-04",
            "season": "2024",
            "gameNumber": 1,
            "doubleHeader": "N",
            "venue": {"id": 15},
            "status": {"abstractGameState": "Final", "detailedState": "Final"},
            "teams": {
                "home": {"score": 5, "team": {"id": 147, "name": "New York Yankees"}},
                "away": {"score": 3, "team": {"id": 111, "name": "Boston Red Sox"}},
            },
        },
        {   # not final → no boxscore, scores absent
            "gamePk": 745805,
            "gameDate": "2024-07-04T23:10:00Z",
            "officialDate": "2024-07-04",
            "season": "2024",
            "gameNumber": 1,
            "doubleHeader": "N",
            "venue": {"id": 22},
            "status": {"abstractGameState": "Live", "detailedState": "In Progress"},
            "teams": {
                "home": {"team": {"id": 119, "name": "Los Angeles Dodgers"}},
                "away": {"team": {"id": 137, "name": "San Francisco Giants"}},
            },
        },
    ],
}]}

BOXSCORE = {"teams": {
    "home": {
        "team": {"id": 147, "name": "New York Yankees"},
        "players": {
            "ID592450": {
                "person": {"id": 592450, "fullName": "Aaron Judge"},
                "position": {"abbreviation": "RF", "type": "Outfielder", "code": "9"},
                "battingOrder": "200",
                "stats": {"batting": {
                    "atBats": 4, "hits": 2, "strikeOuts": 1, "baseOnBalls": 1,
                    "hitByPitch": 0, "sacFlies": 0, "sacBunts": 0,
                    "plateAppearances": 5}},
            },
            "ID543037": {
                "person": {"id": 543037, "fullName": "Gerrit Cole"},
                "position": {"abbreviation": "P", "type": "Pitcher", "code": "1"},
                "stats": {
                    "pitching": {"inningsPitched": "6.1", "strikeOuts": 8,
                                 "earnedRuns": 2},
                    "batting": {"atBats": 0, "hits": 0, "strikeOuts": 0,
                                "baseOnBalls": 0, "plateAppearances": 0}},
            },
        },
    },
    "away": {
        "team": {"id": 111, "name": "Boston Red Sox"},
        "players": {
            "ID646240": {
                "person": {"id": 646240, "fullName": "Rafael Devers"},
                "position": {"abbreviation": "3B", "type": "Infielder", "code": "5"},
                "battingOrder": "300",
                "stats": {"batting": {
                    "atBats": 3, "hits": 1, "strikeOuts": 2, "baseOnBalls": 0,
                    "hitByPitch": 1, "sacFlies": 0, "sacBunts": 0,
                    "plateAppearances": 4}},
            },
            "ID1": {   # bench player, never batted → excluded from batter rows
                "person": {"id": 1, "fullName": "Bench Guy"},
                "position": {"abbreviation": "1B", "type": "Infielder", "code": "3"},
                "stats": {"batting": {"atBats": 0, "plateAppearances": 0,
                                      "baseOnBalls": 0, "hits": 0,
                                      "strikeOuts": 0}},
            },
        },
    },
}}

STANDINGS = {"records": [
    {"league": {"id": 103}, "teamRecords": [
        {"team": {"id": 147, "name": "New York Yankees"}, "wins": 55,
         "losses": 30, "winningPercentage": ".647"},
        {"team": {"id": 111, "name": "Boston Red Sox"}, "wins": 45,
         "losses": 40, "winningPercentage": ".529"},
    ]},
    {"league": {"id": 104}, "teamRecords": [
        {"team": {"id": 119, "name": "Los Angeles Dodgers"}, "wins": 52,
         "losses": 33, "winningPercentage": ".612"},
    ]},
]}

GAME = {"game_pk": 745804, "home_team_id": "147", "away_team_id": "111",
        "official_date": "2024-07-04", "game_date": "2024-07-04T17:10:00Z"}


def _rows(table):
    with db_store.get_engine().connect() as conn:
        return conn.execute(select(table)).fetchall()


def _count(table):
    return len(_rows(table))


class _Backend:
    def setUp(self):
        db_store.configure_engine("sqlite://")
        mlb_warehouse.create_all()
        mlb_warehouse._TEAMS_ENSURED.clear()

    def tearDown(self):
        db_store.configure_engine(None)
        mlb_warehouse._TEAMS_ENSURED.clear()


# ─────────────────────────────────────────────────────────────── schema parity
class SchemaParityTests(unittest.TestCase):
    """Guard the column-name SPECs against the Table definitions (which
    sql/schema.sql mirrors for the hand-run Azure DDL)."""

    def test_bronze_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.mlb_bronze.columns},
                         set(mlb_warehouse._BRONZE_COLS))

    def test_team_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.mlb_team.columns},
                         set(mlb_warehouse._TEAM_COLS))

    def test_game_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.mlb_game.columns},
                         set(mlb_warehouse._GAME_COLS))

    def test_player_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.mlb_player.columns},
                         set(mlb_warehouse._PLAYER_COLS))

    def test_standings_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.mlb_team_standings.columns},
                         set(mlb_warehouse._STANDINGS_COLS))

    def test_alias_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.player_alias.columns},
                         set(mlb_warehouse._ALIAS_COLS))


# ─────────────────────────────────────────────────────────── pure parse / derive
class ParseTests(unittest.TestCase):
    def test_parse_teams_filters_non_mlb(self):
        rows = mlb_warehouse.parse_teams(TEAMS)
        self.assertEqual(len(rows), 4)   # AL All-Stars (sport 51) excluded
        nyy = next(r for r in rows if r["team_id"] == "147")
        self.assertEqual(nyy["abbreviation"], "NYY")
        self.assertEqual(nyy["league_id"], "103")
        self.assertEqual(nyy["division_id"], "201")
        self.assertEqual(nyy["name_norm"], "new york yankees")

    def test_parse_schedule_games_and_teams(self):
        games, teams = mlb_warehouse.parse_schedule(SCHEDULE, 2024)
        self.assertEqual(len(games), 2)
        self.assertEqual(len(teams), 4)
        g = next(x for x in games if x["game_pk"] == 745804)
        self.assertEqual(g["home_team_id"], "147")
        self.assertEqual(g["away_team_id"], "111")
        self.assertEqual(g["official_date"], "2024-07-04")
        self.assertEqual(g["season"], 2024)
        self.assertEqual(g["status"], "Final")
        self.assertEqual(g["home_score"], 5.0)
        self.assertEqual(g["away_score"], 3.0)
        self.assertEqual(g["venue_id"], "15")
        # live game → scores absent
        live = next(x for x in games if x["game_pk"] == 745805)
        self.assertIsNone(live["home_score"])
        self.assertEqual(live["status"], "Live")

    def test_parse_schedule_linescore_fallback(self):
        raw = copy.deepcopy(SCHEDULE)
        g = raw["dates"][0]["games"][0]
        del g["teams"]["home"]["score"]
        del g["teams"]["away"]["score"]
        g["linescore"] = {"teams": {"home": {"runs": 7}, "away": {"runs": 1}}}
        games, _ = mlb_warehouse.parse_schedule(raw, 2024)
        gg = next(x for x in games if x["game_pk"] == 745804)
        self.assertEqual(gg["home_score"], 7.0)
        self.assertEqual(gg["away_score"], 1.0)

    def test_parse_standings(self):
        rows = mlb_warehouse.parse_standings(STANDINGS, 2024, "2024-07-04")
        self.assertEqual(len(rows), 3)
        nyy = next(r for r in rows if r["team_id"] == "147")
        self.assertEqual(nyy["wins"], 55)
        self.assertEqual(nyy["losses"], 30)
        self.assertAlmostEqual(nyy["win_pct"], 0.647)
        self.assertEqual(nyy["season"], 2024)
        self.assertEqual(nyy["as_of_date"], "2024-07-04")

    def test_parse_boxscore_players(self):
        rows = mlb_warehouse.parse_boxscore_players(BOXSCORE, GAME)
        by_id = {r["player_id"]: r for r in rows}
        self.assertEqual(len(rows), 4)
        self.assertTrue(by_id["543037"]["is_pitcher"])       # Cole
        self.assertFalse(by_id["592450"]["is_pitcher"])      # Judge
        self.assertEqual(by_id["592450"]["name_norm"], "aaron judge")
        self.assertEqual(by_id["592450"]["primary_position"], "RF")


class DeriveTests(unittest.TestCase):
    def test_derive_batter_rows(self):
        rows = mlb_warehouse.derive_batter_rows(BOXSCORE, GAME)
        by_id = {r["athlete_id"]: r for r in rows}
        # Judge + Devers only; Cole (pitcher, 0 PA) and Bench Guy excluded.
        self.assertEqual(set(by_id), {"592450", "646240"})
        judge = by_id["592450"]
        self.assertEqual(judge["H"], 2.0)
        self.assertEqual(judge["AB"], 4.0)
        self.assertEqual(judge["BB"], 1.0)
        self.assertEqual(judge["game_pk"], 745804)
        self.assertEqual(judge["team_id"], "147")
        devers = by_id["646240"]
        self.assertEqual(devers["HBP"], 1.0)
        self.assertEqual(devers["team_id"], "111")

    def test_derive_pitcher_rows(self):
        rows = mlb_warehouse.derive_pitcher_rows(BOXSCORE, GAME)
        self.assertEqual(len(rows), 1)
        cole = rows[0]
        self.assertEqual(cole["athlete_id"], "543037")
        self.assertEqual(cole["IP"], 6.1)      # base-3 float
        self.assertEqual(cole["K"], 8.0)
        self.assertEqual(cole["ER"], 2.0)
        self.assertEqual(cole["team_id"], "147")


# ─────────────────────────────────────────────────────────────────── ingestion
class IngestTests(_Backend, unittest.TestCase):
    def _ingest(self, schedule=SCHEDULE):
        with mock.patch.object(mlb_warehouse, "fetch_teams", return_value=TEAMS), \
             mock.patch.object(mlb_warehouse, "fetch_schedule",
                               return_value=schedule), \
             mock.patch.object(mlb_warehouse, "fetch_boxscore",
                               return_value=BOXSCORE):
            return mlb_warehouse.ingest_date("2024-07-04")

    def test_ingest_populates_dims_and_bronze(self):
        summary = self._ingest()
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["final"], 1)
        self.assertEqual(summary["boxscores"], 1)
        self.assertEqual(summary["players"], 4)

        self.assertEqual(_count(mlb_warehouse.mlb_team), 4)
        self.assertEqual(_count(mlb_warehouse.mlb_game), 2)
        self.assertEqual(_count(mlb_warehouse.mlb_player), 4)

        # full-team enrich survives the minimal schedule upsert (no clobber)
        nyy = next(r._mapping for r in _rows(mlb_warehouse.mlb_team)
                   if r._mapping["team_id"] == "147")
        self.assertEqual(nyy["abbreviation"], "NYY")

        g = next(r._mapping for r in _rows(mlb_warehouse.mlb_game)
                 if r._mapping["game_pk"] == 745804)
        self.assertEqual(g["home_score"], 5.0)

    def test_bronze_lifecycle(self):
        self._ingest()
        bronze = {(r._mapping["kind"], r._mapping["natural_ref"]): r._mapping
                  for r in _rows(mlb_warehouse.mlb_bronze)}
        self.assertIn(("schedule", "2024-07-04"), bronze)
        self.assertIn(("boxscore", "745804"), bronze)
        # schedule stays pending; a processed boxscore is marked
        self.assertIsNone(bronze[("schedule", "2024-07-04")]["processed_at"])
        self.assertIsNotNone(bronze[("boxscore", "745804")]["processed_at"])
        # payload is valid JSON round-trip
        payload = json.loads(bronze[("boxscore", "745804")]["payload"])
        self.assertIn("teams", payload)
        # purge removes only the processed boxscore payloads
        deleted = mlb_warehouse.purge_processed_boxscores()
        self.assertEqual(deleted, 1)
        kinds = {r._mapping["kind"] for r in _rows(mlb_warehouse.mlb_bronze)}
        self.assertNotIn("boxscore", kinds)
        self.assertIn("schedule", kinds)

    def test_ingest_is_idempotent(self):
        self._ingest()
        self._ingest()
        self.assertEqual(_count(mlb_warehouse.mlb_team), 4)
        self.assertEqual(_count(mlb_warehouse.mlb_game), 2)
        self.assertEqual(_count(mlb_warehouse.mlb_player), 4)
        # one live bronze payload per (kind, natural_ref) — no growth
        self.assertEqual(_count(mlb_warehouse.mlb_bronze), 2)

    def test_reingest_updates_changed_score(self):
        self._ingest()
        changed = copy.deepcopy(SCHEDULE)
        changed["dates"][0]["games"][0]["teams"]["home"]["score"] = 9
        self._ingest(schedule=changed)
        g = next(r._mapping for r in _rows(mlb_warehouse.mlb_game)
                 if r._mapping["game_pk"] == 745804)
        self.assertEqual(g["home_score"], 9.0)
        self.assertEqual(_count(mlb_warehouse.mlb_game), 2)   # updated, not duped

    def test_find_game_pk_retro_match(self):
        self._ingest()
        self.assertEqual(
            mlb_warehouse.find_game_pk("2024-07-04", "147", "111"), 745804)
        self.assertIsNone(
            mlb_warehouse.find_game_pk("2024-07-04", "999", "888"))

    def test_ensure_teams_does_not_memoize_empty_load(self):
        # An empty /teams payload must NOT poison the per-process memo, else
        # ingest_standings' FK stays unsatisfiable and never recovers in-process.
        with mock.patch.object(mlb_warehouse, "fetch_teams",
                               return_value={"teams": []}):
            n = mlb_warehouse.ensure_teams(2024)
        self.assertEqual(n, 0)
        self.assertNotIn(2024, mlb_warehouse._TEAMS_ENSURED)
        self.assertEqual(_count(mlb_warehouse.mlb_team), 0)
        # a later good fetch still loads (the memo was not poisoned)
        with mock.patch.object(mlb_warehouse, "fetch_teams", return_value=TEAMS):
            n2 = mlb_warehouse.ensure_teams(2024)
        self.assertEqual(n2, 4)
        self.assertIn(2024, mlb_warehouse._TEAMS_ENSURED)
        self.assertEqual(_count(mlb_warehouse.mlb_team), 4)


class StandingsTests(_Backend, unittest.TestCase):
    def _snapshot(self, as_of):
        with mock.patch.object(mlb_warehouse, "fetch_teams", return_value=TEAMS), \
             mock.patch.object(mlb_warehouse, "fetch_standings",
                               return_value=STANDINGS):
            return mlb_warehouse.ingest_standings(2024, as_of_date=as_of)

    def test_standings_snapshot(self):
        summary = self._snapshot("2024-07-04")
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(_count(mlb_warehouse.mlb_team_standings), 3)
        nyy = next(r._mapping for r in _rows(mlb_warehouse.mlb_team_standings)
                   if r._mapping["team_id"] == "147")
        self.assertAlmostEqual(nyy["win_pct"], 0.647)
        self.assertEqual(nyy["wins"], 55)

    def test_standings_idempotent_same_asof(self):
        self._snapshot("2024-07-04")
        self._snapshot("2024-07-04")
        self.assertEqual(_count(mlb_warehouse.mlb_team_standings), 3)

    def test_standings_new_asof_adds_snapshot(self):
        self._snapshot("2024-07-04")
        self._snapshot("2024-07-05")
        self.assertEqual(_count(mlb_warehouse.mlb_team_standings), 6)


# ───────────────────────────────────────────────────────────── parity diff core
class ParityDiffTests(unittest.TestCase):
    def test_diff_value_maps(self):
        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        b = {"x": 1.0, "y": 2.5, "w": 9.0}
        rep = parity.diff_value_maps(a, b, tol=1e-6)
        self.assertEqual(rep["compared"], 2)
        self.assertEqual(rep["matches"], 1)
        self.assertEqual(rep["mismatches"], 1)
        self.assertEqual(rep["only_statsapi"], 1)
        self.assertEqual(rep["only_espn"], 1)
        self.assertAlmostEqual(rep["match_rate"], 0.5)

    def test_statsapi_player_game_stats_batter(self):
        m = parity.statsapi_player_game_stats(BOXSCORE, GAME, "batter", "H")
        self.assertEqual(m[("aaron judge", "2024-07-04")], 2.0)
        self.assertEqual(m[("rafael devers", "2024-07-04")], 1.0)
        self.assertNotIn(("gerrit cole", "2024-07-04"), m)

    def test_statsapi_player_game_stats_pitcher(self):
        m = parity.statsapi_player_game_stats(BOXSCORE, GAME, "pitcher", "K")
        self.assertEqual(m, {("gerrit cole", "2024-07-04"): 8.0})

    def test_statsapi_standings_winpct(self):
        # /standings gives nicknames; the lens resolves full names via /teams so
        # the keys align with ESPN's displayName — both fetchers are mocked.
        nick = copy.deepcopy(STANDINGS)
        nick["records"][0]["teamRecords"][0]["team"]["name"] = "Yankees"
        nick["records"][1]["teamRecords"][0]["team"]["name"] = "Dodgers"
        with mock.patch.object(mlb_warehouse, "fetch_standings", return_value=nick), \
             mock.patch.object(mlb_warehouse, "fetch_teams", return_value=TEAMS):
            m = parity.statsapi_standings_winpct(2024)
        self.assertAlmostEqual(m["new york yankees"], 0.647)   # resolved from id 147
        self.assertAlmostEqual(m["los angeles dodgers"], 0.612)  # resolved from id 119


class ParityAlignTests(unittest.TestCase):
    """The UTC/local ±1-day + doubleheader realignment that keeps the batter
    parity lens from silently dropping night games out of `compared`."""

    def test_prev_day(self):
        self.assertEqual(parity._prev_day("2024-07-05"), "2024-07-04")
        self.assertEqual(parity._prev_day("2024-07-05T02:10:00Z"), "2024-07-04")
        self.assertIsNone(parity._prev_day(None))

    def test_align_remaps_utc_night_game_to_official(self):
        # ESPN filed a west-coast night game one UTC day ahead of official date.
        keys = {("mookie betts", "2024-07-04")}
        out = parity._align_espn_to_official(
            {("mookie betts", "2024-07-05"): 2.0}, keys)
        self.assertEqual(out, {("mookie betts", "2024-07-04"): 2.0})

    def test_align_keeps_aligned_day_game(self):
        keys = {("aaron judge", "2024-07-04")}
        out = parity._align_espn_to_official(
            {("aaron judge", "2024-07-04"): 1.0}, keys)
        self.assertEqual(out, {("aaron judge", "2024-07-04"): 1.0})

    def test_align_sums_doubleheader_across_utc_dates(self):
        # G1 afternoon (UTC == official) + G2 night (UTC == official+1) → one total.
        keys = {("aaron judge", "2024-07-04")}
        out = parity._align_espn_to_official(
            {("aaron judge", "2024-07-04"): 1.0,
             ("aaron judge", "2024-07-05"): 3.0}, keys)
        self.assertEqual(out, {("aaron judge", "2024-07-04"): 4.0})

    def test_align_bounds_to_window(self):
        out = parity._align_espn_to_official(
            {("x", "2024-07-10"): 5.0}, set(),
            start="2024-07-01", end="2024-07-04")
        self.assertEqual(out, {})

    def test_align_unmatched_date_kept_as_is(self):
        # No official key to snap to and inside window → passes through unchanged.
        out = parity._align_espn_to_official(
            {("y", "2024-07-03"): 2.0}, set(),
            start="2024-07-01", end="2024-07-04")
        self.assertEqual(out, {("y", "2024-07-03"): 2.0})


if __name__ == "__main__":
    unittest.main()
