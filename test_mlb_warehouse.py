"""Tests for the MLB StatsAPI medallion P1 layer (mlb_warehouse + parity harness).

Covers: SchemaParity (Table columns ↔ column SPECs), the pure parse/derive
functions off fixture StatsAPI payloads, reconcile-based idempotent ingestion on
SQLite, the standings snapshot fact, the bronze lifecycle, and the pure parity
diff core. No network — the fetchers are monkeypatched to return fixtures.
"""

import copy
import datetime
import json
import unittest
from unittest import mock

from sqlalchemy import insert, select

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
            "gameType": "R",
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
                    "homeRuns": 1, "totalBases": 5, "rbi": 2,
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
                    "homeRuns": 0, "totalBases": 1, "rbi": 0,
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

    def test_batter_game_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.mlb_batter_game.columns},
                         set(mlb_warehouse._BATTER_GAME_COLS))

    def test_pitcher_game_cols(self):
        self.assertEqual({c.name for c in mlb_warehouse.mlb_pitcher_game.columns},
                         set(mlb_warehouse._PITCHER_GAME_COLS))


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
        self.assertEqual(g["game_type"], "R")            # captured from gameType
        self.assertEqual(g["status"], "Final")
        # live game has no gameType in the fixture → None (faithful, not defaulted)
        self.assertIsNone(
            next(x for x in games if x["game_pk"] == 745805)["game_type"])
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
        self.assertEqual(judge["HR"], 1.0)          # HR/TB/RBI captured
        self.assertEqual(judge["TB"], 5.0)          # HR + single
        self.assertEqual(judge["RBI"], 2.0)
        self.assertEqual(judge["game_pk"], 745804)
        self.assertEqual(judge["team_id"], "147")
        devers = by_id["646240"]
        self.assertEqual(devers["HBP"], 1.0)
        self.assertEqual((devers["HR"], devers["TB"], devers["RBI"]), (0.0, 1.0, 0.0))
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

    def test_batter_same_player_both_teams_deduped(self):
        # The 2024-06-26 Danny Jansen game: suspended, traded, resumed → listed for
        # BOTH clubs. Keep the line where he actually batted (max PA) + its team.
        box = {"teams": {
            "home": {"team": {"id": 111}, "players": {"ID643376": {
                "person": {"id": 643376, "fullName": "Danny Jansen"},
                "position": {"abbreviation": "C"}, "battingOrder": "701",
                "stats": {"batting": {"atBats": 4, "hits": 1, "strikeOuts": 1,
                                      "plateAppearances": 4}}}}},
            "away": {"team": {"id": 141}, "players": {"ID643376": {
                "person": {"id": 643376, "fullName": "Danny Jansen"},
                "position": {"abbreviation": "C"}, "battingOrder": "700",
                "stats": {"batting": {"atBats": 0, "hits": 0, "strikeOuts": 0,
                                      "plateAppearances": 0}}}}}}}
        game = {"game_pk": 746942, "home_team_id": "111", "away_team_id": "141"}
        rows = mlb_warehouse.derive_batter_rows(box, game)
        self.assertEqual(len(rows), 1)                 # deduped
        self.assertEqual(rows[0]["athlete_id"], "643376")
        self.assertEqual(rows[0]["team_id"], "111")    # Boston — where he batted
        self.assertEqual((rows[0]["AB"], rows[0]["H"]), (4.0, 1.0))

    def test_pitcher_same_player_both_teams_deduped(self):
        box = {"teams": {
            "home": {"team": {"id": 111}, "players": {"ID1": {
                "person": {"id": 1, "fullName": "Two Team"},
                "stats": {"pitching": {"inningsPitched": "5.0", "strikeOuts": 6,
                                       "earnedRuns": 2}}}}},
            "away": {"team": {"id": 141}, "players": {"ID1": {
                "person": {"id": 1, "fullName": "Two Team"},
                "stats": {"pitching": {"inningsPitched": "0.1", "strikeOuts": 0,
                                       "earnedRuns": 0}}}}}}}
        game = {"game_pk": 700, "home_team_id": "111", "away_team_id": "141"}
        rows = mlb_warehouse.derive_pitcher_rows(box, game)
        self.assertEqual(len(rows), 1)                 # deduped (max IP)
        self.assertEqual(rows[0]["team_id"], "111")
        self.assertEqual(rows[0]["IP"], 5.0)


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


# ───────────────────────────────────── P2 StatsAPI-native game facts (writer)
class GameFactTests(_Backend, unittest.TestCase):
    """The game-centric writer: ingest_date persists boxscore-derived batter/
    pitcher stat lines into mlb_batter_game / mlb_pitcher_game (dual-run)."""

    def _ingest(self, schedule=SCHEDULE, boxscore=BOXSCORE):
        with mock.patch.object(mlb_warehouse, "fetch_teams", return_value=TEAMS), \
             mock.patch.object(mlb_warehouse, "fetch_schedule",
                               return_value=schedule), \
             mock.patch.object(mlb_warehouse, "fetch_boxscore",
                               return_value=boxscore):
            return mlb_warehouse.ingest_date("2024-07-04")

    def test_ingest_writes_game_facts(self):
        summary = self._ingest()
        self.assertEqual(summary["batter_rows"], 2)   # Judge + Devers (Cole: 0 PA)
        self.assertEqual(summary["pitcher_rows"], 1)   # Cole
        b = {r._mapping["athlete_id"]: r._mapping
             for r in _rows(mlb_warehouse.mlb_batter_game)}
        self.assertEqual(set(b), {"592450", "646240"})
        self.assertEqual(b["592450"]["H"], 2.0)
        self.assertEqual(b["592450"]["HR"], 1.0)           # HR/TB/RBI persisted end-to-end
        self.assertEqual(b["592450"]["TB"], 5.0)
        self.assertEqual(b["592450"]["RBI"], 2.0)
        self.assertEqual(b["592450"]["game_pk"], 745804)   # native from the boxscore
        self.assertEqual(b["592450"]["team_id"], "147")
        self.assertEqual(b["592450"]["season_bucket"], 2024)
        self.assertEqual(b["646240"]["team_id"], "111")
        p = _rows(mlb_warehouse.mlb_pitcher_game)
        self.assertEqual(len(p), 1)
        cole = p[0]._mapping
        self.assertEqual(cole["athlete_id"], "543037")
        self.assertEqual(cole["IP"], 6.1)                  # base-3 float
        self.assertEqual(cole["K"], 8.0)
        self.assertEqual(cole["team_id"], "147")
        # live game 745805 contributed no facts (not genuine-final)
        self.assertTrue(all(r._mapping["game_pk"] == 745804
                            for r in _rows(mlb_warehouse.mlb_batter_game)))

    def test_game_facts_idempotent(self):
        self._ingest()
        self._ingest()
        self.assertEqual(_count(mlb_warehouse.mlb_batter_game), 2)
        self.assertEqual(_count(mlb_warehouse.mlb_pitcher_game), 1)

    def test_game_facts_correction_updates_in_place(self):
        self._ingest()
        changed = copy.deepcopy(BOXSCORE)
        (changed["teams"]["home"]["players"]["ID592450"]
         ["stats"]["batting"]["hits"]) = 3
        self._ingest(boxscore=changed)
        b = {r._mapping["athlete_id"]: r._mapping
             for r in _rows(mlb_warehouse.mlb_batter_game)}
        self.assertEqual(b["592450"]["H"], 3.0)                     # updated
        self.assertEqual(_count(mlb_warehouse.mlb_batter_game), 2)  # not duped


# ─────────────────────────────── P2 leakage-safe reader (the ordering fix)
class GameLogReaderTests(_Backend, unittest.TestCase):
    """get_batter/pitcher_game_log: most-recent-first by JOINED game_date DESC,
    game_pk DESC tiebreak; as_of_date is a strict leakage cutoff."""

    def _seed_game(self, game_pk, official_date, game_date,
                   home="147", away="111", season=2024):
        with db_store.get_engine().begin() as conn:
            for tid in (home, away):
                if not conn.execute(
                        select(mlb_warehouse.mlb_team).where(
                            mlb_warehouse.mlb_team.c.team_id == tid)).first():
                    conn.execute(insert(mlb_warehouse.mlb_team),
                                 {"team_id": tid, "name": tid})
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": game_pk, "game_date": game_date,
                "official_date": official_date, "season": season,
                "home_team_id": home, "away_team_id": away})

    def _seed_batter(self, athlete_id, game_pk, H, team_id="147", season=2024):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_batter_game), {
                "athlete_id": athlete_id, "game_pk": game_pk,
                "team_id": team_id, "season_bucket": season, "H": H})

    def test_orders_by_game_date_not_game_pk(self):
        # Postponement: pk=100 was scheduled early but PLAYED in Sept; pk=200
        # played in April. Recency must follow game_date (play date), not pk.
        self._seed_game(100, "2024-09-01", "2024-09-01T18:00:00Z")
        self._seed_game(200, "2024-04-01", "2024-04-01T18:00:00Z")
        self._seed_batter("1", 100, 2.0)
        self._seed_batter("1", 200, 0.0)
        log = mlb_warehouse.get_batter_game_log("1")
        self.assertEqual([r["game_pk"] for r in log], [100, 200])  # Sept first

    def test_doubleheader_game_pk_tiebreak(self):
        # Identical play timestamp (traditional DH) → deterministic game_pk DESC.
        self._seed_game(300, "2024-07-04", "2024-07-04T18:00:00Z")
        self._seed_game(301, "2024-07-04", "2024-07-04T18:00:00Z")
        self._seed_batter("1", 300, 1.0)
        self._seed_batter("1", 301, 3.0)
        log = mlb_warehouse.get_batter_game_log("1")
        self.assertEqual([r["game_pk"] for r in log], [301, 300])

    def test_as_of_cutoff_is_strict(self):
        self._seed_game(1, "2024-07-01", "2024-07-01T18:00:00Z")
        self._seed_game(2, "2024-07-04", "2024-07-04T18:00:00Z")
        self._seed_game(3, "2024-07-08", "2024-07-08T18:00:00Z")
        for pk in (1, 2, 3):
            self._seed_batter("1", pk, 1.0)
        log = mlb_warehouse.get_batter_game_log("1", as_of_date="2024-07-05")
        self.assertEqual([r["game_pk"] for r in log], [2, 1])   # 07-08 excluded
        # a game ON the cutoff date is excluded (strictly-before)
        log2 = mlb_warehouse.get_batter_game_log("1", as_of_date="2024-07-04")
        self.assertEqual([r["game_pk"] for r in log2], [1])

    def test_opponent_and_is_home_derivation(self):
        self._seed_game(10, "2024-07-04", "2024-07-04T18:00:00Z",
                        home="147", away="111")
        self._seed_batter("home_guy", 10, 1.0, team_id="147")
        self._seed_batter("away_guy", 10, 1.0, team_id="111")
        home = mlb_warehouse.get_batter_game_log("home_guy")[0]
        self.assertTrue(home["is_home"])
        self.assertEqual(home["opponent_team_id"], "111")
        away = mlb_warehouse.get_batter_game_log("away_guy")[0]
        self.assertFalse(away["is_home"])
        self.assertEqual(away["opponent_team_id"], "147")

    def test_limit_and_season_filter(self):
        self._seed_game(1, "2023-07-01", "2023-07-01T18:00:00Z", season=2023)
        self._seed_game(2, "2024-07-04", "2024-07-04T18:00:00Z", season=2024)
        self._seed_batter("1", 1, 1.0, season=2023)
        self._seed_batter("1", 2, 2.0, season=2024)
        self.assertEqual(len(mlb_warehouse.get_batter_game_log("1", limit=1)), 1)
        self.assertEqual(
            [r["game_pk"] for r in
             mlb_warehouse.get_batter_game_log("1", season=2024)], [2])

    def test_pitcher_reader(self):
        self._seed_game(5, "2024-07-04", "2024-07-04T18:00:00Z")
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_pitcher_game), {
                "athlete_id": "p1", "game_pk": 5, "team_id": "147",
                "season_bucket": 2024, "IP": 6.1, "K": 8.0, "ER": 2.0})
        log = mlb_warehouse.get_pitcher_game_log("p1")
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["K"], 8.0)
        self.assertEqual(log[0]["IP"], 6.1)
        self.assertEqual(log[0]["is_home"], True)


# ──────────────────────────────────────────── P3 resolver support (warehouse)
class P3SupportTests(_Backend, unittest.TestCase):
    """team_id_for_name, find_game_pk_by_commence (series/DH-robust), and the
    player_alias writer."""

    def setUp(self):
        super().setUp()
        for tid, nm in (("147", "New York Yankees"), ("111", "Boston Red Sox")):
            with db_store.get_engine().begin() as conn:
                conn.execute(insert(mlb_warehouse.mlb_team), {
                    "team_id": tid, "name": nm,
                    "name_norm": db_store.normalize_name(nm)})

    def _game(self, game_pk, official_date, game_date, home="147", away="111"):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": game_pk, "official_date": official_date,
                "game_date": game_date, "home_team_id": home,
                "away_team_id": away, "season": int(official_date[:4])})

    def test_team_id_for_name(self):
        self.assertEqual(mlb_warehouse.team_id_for_name("New York Yankees"), "147")
        self.assertEqual(mlb_warehouse.team_id_for_name("new york yankees"), "147")
        self.assertIsNone(mlb_warehouse.team_id_for_name("Nonexistent Team"))
        self.assertIsNone(mlb_warehouse.team_id_for_name(None))

    def test_commence_series_picks_right_day(self):
        self._game(1, "2026-08-08", "2026-08-08T23:05:00Z")
        self._game(2, "2026-08-09", "2026-08-09T23:05:00Z")
        self.assertEqual(mlb_warehouse.find_game_pk_by_commence(
            "147", "111", "2026-08-09T23:05:00Z"), 2)
        self.assertEqual(mlb_warehouse.find_game_pk_by_commence(
            "147", "111", "2026-08-08T23:10:00Z"), 1)

    def test_commence_split_dh_picks_by_time(self):
        self._game(10, "2026-08-08", "2026-08-08T17:05:00Z")   # game 1 afternoon
        self._game(11, "2026-08-08", "2026-08-08T23:05:00Z")   # game 2 night
        self.assertEqual(mlb_warehouse.find_game_pk_by_commence(
            "147", "111", "2026-08-08T23:00:00Z"), 11)
        self.assertEqual(mlb_warehouse.find_game_pk_by_commence(
            "147", "111", "2026-08-08T17:10:00Z"), 10)

    def test_commence_traditional_dh_identical_ts_is_ambiguous(self):
        self._game(20, "2026-08-08", "2026-08-08T17:05:00Z")
        self._game(21, "2026-08-08", "2026-08-08T17:05:00Z")   # identical → ambiguous
        self.assertIsNone(mlb_warehouse.find_game_pk_by_commence(
            "147", "111", "2026-08-08T17:05:00Z"))

    def test_commence_out_of_tolerance_and_unknown(self):
        self._game(30, "2026-08-08", "2026-08-08T23:05:00Z")
        self.assertIsNone(mlb_warehouse.find_game_pk_by_commence(
            "147", "111", "2026-08-15T23:05:00Z"))          # far date → out of window
        self.assertIsNone(mlb_warehouse.find_game_pk_by_commence(
            "999", "888", "2026-08-08T23:05:00Z"))          # unknown matchup

    def test_record_player_alias_upsert(self):
        self.assertTrue(
            mlb_warehouse.record_player_alias("oddsapi", "aaron judge", "592450"))
        rows = _rows(mlb_warehouse.player_alias)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]._mapping["mlb_player_id"], "592450")
        self.assertEqual(rows[0]._mapping["resolution_method"], "sfbb_unique")
        # re-record same → no dup
        mlb_warehouse.record_player_alias("oddsapi", "aaron judge", "592450")
        self.assertEqual(_count(mlb_warehouse.player_alias), 1)
        # a changed id updates in place (still one row)
        mlb_warehouse.record_player_alias("oddsapi", "aaron judge", "999999")
        rows = _rows(mlb_warehouse.player_alias)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]._mapping["mlb_player_id"], "999999")

    def test_record_player_alias_guards(self):
        self.assertFalse(mlb_warehouse.record_player_alias("oddsapi", "", "592450"))
        self.assertFalse(mlb_warehouse.record_player_alias("oddsapi", "x", None))
        self.assertEqual(_count(mlb_warehouse.player_alias), 0)


# ─────────────────────────────── P4 grading fast path (actual-stat reader)
class GetActualStatTests(_Backend, unittest.TestCase):
    """get_actual_stat reads a graded prop's actual straight from the game facts."""

    def setUp(self):
        super().setUp()
        with db_store.get_engine().begin() as conn:
            for tid in ("147", "111"):
                conn.execute(insert(mlb_warehouse.mlb_team),
                             {"team_id": tid, "name": tid})
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 700, "official_date": "2026-08-09",
                "game_date": "2026-08-09T23:05:00Z", "home_team_id": "147",
                "away_team_id": "111", "season": 2026})
            conn.execute(insert(mlb_warehouse.mlb_batter_game), {
                "athlete_id": "b1", "game_pk": 700, "team_id": "147",
                "season_bucket": 2026, "H": 2.0, "SO": 1.0})
            conn.execute(insert(mlb_warehouse.mlb_batter_game), {
                "athlete_id": "b0", "game_pk": 700, "team_id": "147",
                "season_bucket": 2026, "H": 0.0, "SO": 0.0})   # a real 0
            conn.execute(insert(mlb_warehouse.mlb_pitcher_game), {
                "athlete_id": "p1", "game_pk": 700, "team_id": "147",
                "season_bucket": 2026, "K": 8.0, "ER": 2.0, "IP": 6.1})

    def test_batter_and_pitcher_stats(self):
        gs = mlb_warehouse.get_actual_stat
        self.assertEqual(gs("b1", 700, "batter_hits"), 2.0)
        self.assertEqual(gs("b1", 700, "batter_strikeouts"), 1.0)
        self.assertEqual(gs("p1", 700, "pitcher_strikeouts"), 8.0)
        self.assertEqual(gs("p1", 700, "pitcher_earned_runs"), 2.0)

    def test_pitcher_outs_converts_ip_base3(self):
        self.assertEqual(mlb_warehouse.get_actual_stat("p1", 700, "pitcher_outs"), 19)

    def test_resolved_zero_is_not_a_miss(self):
        self.assertEqual(mlb_warehouse.get_actual_stat("b0", 700, "batter_hits"), 0.0)

    def test_unsupported_prop_returns_none(self):
        # HR/TB/RBI have no fact column → None → caller falls back to live.
        self.assertIsNone(mlb_warehouse.get_actual_stat("b1", 700, "batter_home_runs"))
        self.assertIsNone(mlb_warehouse.get_actual_stat("b1", 700, "batter_total_bases"))

    def test_missing_row_returns_none(self):
        self.assertIsNone(mlb_warehouse.get_actual_stat("nobody", 700, "batter_hits"))
        self.assertIsNone(mlb_warehouse.get_actual_stat("b1", 999, "batter_hits"))

    def test_none_ids_return_none(self):
        self.assertIsNone(mlb_warehouse.get_actual_stat(None, 700, "batter_hits"))
        self.assertIsNone(mlb_warehouse.get_actual_stat("b1", None, "batter_hits"))


# ───────────────────── P4 unified resolution (refresh active / freeze historical)
class ResolveActualTests(_Backend, unittest.TestCase):
    """resolve_actual: refresh a recent/active game, read a settled game frozen."""

    def _game(self, hours_ago):
        gd = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(hours=hours_ago)).isoformat()
        return {"game_date": gd, "status": "Final", "detailed_state": "Final"}

    def test_historical_reads_frozen_no_refresh(self):
        with mock.patch.object(mlb_warehouse, "get_game",
                               return_value=self._game(100)), \
             mock.patch.object(mlb_warehouse, "_fact_captured_after",
                               return_value=True), \
             mock.patch.object(mlb_warehouse, "refresh_game_facts") as refresh, \
             mock.patch.object(mlb_warehouse, "get_actual_stat", return_value=2.0):
            v = mlb_warehouse.resolve_actual("b1", 700, "batter_hits")
        self.assertEqual(v, 2.0)
        refresh.assert_not_called()             # frozen — no network

    def test_active_recent_refreshes_short_ttl(self):
        with mock.patch.object(mlb_warehouse, "get_game",
                               return_value=self._game(2)), \
             mock.patch.object(mlb_warehouse, "refresh_game_facts") as refresh, \
             mock.patch.object(mlb_warehouse, "get_actual_stat", return_value=1.0):
            v = mlb_warehouse.resolve_actual("b1", 700, "batter_hits")
        self.assertEqual(v, 1.0)
        refresh.assert_called_once()
        self.assertTrue(refresh.call_args.kwargs["active"])   # short-TTL refresh

    def test_crossed_window_not_post_refreshed_forces_fresh(self):
        with mock.patch.object(mlb_warehouse, "get_game",
                               return_value=self._game(100)), \
             mock.patch.object(mlb_warehouse, "_fact_captured_after",
                               return_value=False), \
             mock.patch.object(mlb_warehouse, "refresh_game_facts") as refresh, \
             mock.patch.object(mlb_warehouse, "get_actual_stat", return_value=3.0):
            v = mlb_warehouse.resolve_actual("b1", 700, "batter_hits")
        self.assertEqual(v, 3.0)
        refresh.assert_called_once()
        self.assertFalse(refresh.call_args.kwargs["active"])  # one forced-fresh refresh

    def test_game_not_in_warehouse_returns_none(self):
        with mock.patch.object(mlb_warehouse, "get_game", return_value=None):
            self.assertIsNone(mlb_warehouse.resolve_actual("b1", 700, "batter_hits"))

    def test_unsupported_prop_returns_none(self):
        self.assertIsNone(
            mlb_warehouse.resolve_actual("b1", 700, "batter_home_runs"))


class RefreshGameFactsTests(_Backend, unittest.TestCase):
    def test_refresh_writes_facts_from_boxscore(self):
        with db_store.get_engine().begin() as conn:
            for tid in ("147", "111"):
                conn.execute(insert(mlb_warehouse.mlb_team),
                             {"team_id": tid, "name": tid})
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 745804, "official_date": "2024-07-04",
                "game_date": "2024-07-04T17:10:00Z", "home_team_id": "147",
                "away_team_id": "111", "season": 2024,
                "status": "Final", "detailed_state": "Final"})
        with mock.patch.object(mlb_warehouse, "fetch_boxscore",
                               return_value=BOXSCORE):
            ok = mlb_warehouse.refresh_game_facts(745804, active=True)
        self.assertTrue(ok)
        self.assertEqual(_count(mlb_warehouse.mlb_batter_game), 2)   # Judge + Devers
        self.assertEqual(_count(mlb_warehouse.mlb_pitcher_game), 1)  # Cole

    def test_refresh_non_final_is_noop(self):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 999, "official_date": "2024-07-04",
                "game_date": "2024-07-04T17:10:00Z", "status": "Live",
                "detailed_state": "In Progress"})
        with mock.patch.object(mlb_warehouse, "fetch_boxscore") as fb:
            ok = mlb_warehouse.refresh_game_facts(999, active=True)
        self.assertFalse(ok)
        fb.assert_not_called()                  # never fetch a non-final box

    def _seed_judge(self, fetched_at):
        # game + teams + a pre-existing Judge fact matching the BOXSCORE exactly
        # (so an unchanged re-pull is a genuine no-op) with a KNOWN old fetched_at.
        with db_store.get_engine().begin() as conn:
            for tid in ("147", "111"):
                conn.execute(insert(mlb_warehouse.mlb_team),
                             {"team_id": tid, "name": tid})
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 745804, "official_date": "2024-07-04",
                "game_date": "2024-07-04T17:10:00Z", "home_team_id": "147",
                "away_team_id": "111", "season": 2024,
                "status": "Final", "detailed_state": "Final"})
            conn.execute(insert(mlb_warehouse.mlb_batter_game), {
                "athlete_id": "592450", "game_pk": 745804, "team_id": "147",
                "season_bucket": 2024, "AB": 4.0, "H": 2.0, "SO": 1.0, "BB": 1.0,
                "HBP": 0.0, "SF": 0.0, "SH": 0.0, "HR": 1.0, "TB": 5.0, "RBI": 2.0,
                "fetched_at": fetched_at})

    def _judge(self):
        return next(r._mapping for r in _rows(mlb_warehouse.mlb_batter_game)
                    if r._mapping["athlete_id"] == "592450")

    def test_post_window_refresh_advances_fetched_at(self):
        # active=False must bump fetched_at even on an UNCHANGED box → this is what
        # flips the game to HISTORICAL/frozen in resolve_actual.
        self._seed_judge(1000.0)
        with mock.patch.object(mlb_warehouse, "fetch_boxscore", return_value=BOXSCORE):
            mlb_warehouse.refresh_game_facts(745804, active=False)
        self.assertGreater(self._judge()["fetched_at"], 1000.0)

    def test_active_refresh_keeps_fetched_at_when_unchanged(self):
        # active=True on an unchanged box is a no-op — no fetched_at churn.
        self._seed_judge(1000.0)
        with mock.patch.object(mlb_warehouse, "fetch_boxscore", return_value=BOXSCORE):
            mlb_warehouse.refresh_game_facts(745804, active=True)
        self.assertEqual(self._judge()["fetched_at"], 1000.0)


# ──────────────────────────────── P4 auto-ingest maintenance (window + stragglers)
class IngestMaintenanceTests(_Backend, unittest.TestCase):
    def test_covers_window_and_stragglers(self):
        today = datetime.date.fromisoformat(mlb_warehouse._today())
        straggler = (today - datetime.timedelta(days=5)).isoformat()   # past, not Final
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 50, "official_date": straggler,
                "game_date": straggler + "T18:00:00Z", "status": "Live"})
        called = []
        with mock.patch.object(mlb_warehouse, "ingest_date",
                               side_effect=lambda d, **k: called.append(d) or
                               {"games": 0, "batter_rows": 0, "pitcher_rows": 0}):
            mlb_warehouse.ingest_maintenance(days_back=2, days_forward=2)
        for d in range(-2, 3):                         # rolling window present
            self.assertIn((today + datetime.timedelta(days=d)).isoformat(), called)
        self.assertIn(straggler, called)               # straggler swept in

    def test_final_straggler_not_swept(self):
        today = datetime.date.fromisoformat(mlb_warehouse._today())
        old_final = (today - datetime.timedelta(days=6)).isoformat()
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 51, "official_date": old_final,
                "game_date": old_final + "T18:00:00Z", "status": "Final"})
        called = []
        with mock.patch.object(mlb_warehouse, "ingest_date",
                               side_effect=lambda d, **k: called.append(d) or {}):
            mlb_warehouse.ingest_maintenance(days_back=1, days_forward=0)
        self.assertNotIn(old_final, called)            # already Final → not re-swept

    def test_fail_open(self):
        with mock.patch.object(mlb_warehouse, "ingest_date",
                               side_effect=RuntimeError("boom")):
            summary = mlb_warehouse.ingest_maintenance(days_back=1, days_forward=0)
        self.assertEqual(summary["dates"], 0)          # all failed, but no raise


class GetPlayerHistoryTests(_Backend, unittest.TestCase):
    """get_player_history reproduces the ESPN get_player_stat_history contract dict
    from the StatsAPI facts: most-recent-first, resolved opponent NAMES, derived PA,
    IP→outs, leakage-safe as_of, and None (fall-open) when it can't serve."""

    def setUp(self):
        super().setUp()
        for tid, nm in (("147", "New York Yankees"), ("111", "Boston Red Sox"),
                        ("119", "Los Angeles Dodgers")):
            with db_store.get_engine().begin() as conn:
                conn.execute(insert(mlb_warehouse.mlb_team), {
                    "team_id": tid, "name": nm,
                    "name_norm": db_store.normalize_name(nm)})

    def _game(self, game_pk, official_date, game_date, home="147", away="111",
              season=2024, game_type="R"):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": game_pk, "official_date": official_date,
                "game_date": game_date, "season": season, "game_type": game_type,
                "home_team_id": home, "away_team_id": away})

    def _batter(self, athlete_id, game_pk, team_id="147", season=2024, **stats):
        row = {"athlete_id": athlete_id, "game_pk": game_pk, "team_id": team_id,
               "season_bucket": season}
        row.update(stats)
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_batter_game), row)

    def _pitcher(self, athlete_id, game_pk, team_id="147", season=2024, **stats):
        row = {"athlete_id": athlete_id, "game_pk": game_pk, "team_id": team_id,
               "season_bucket": season}
        row.update(stats)
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_pitcher_game), row)

    def test_batter_history_shape_order_and_names(self):
        self._game(200, "2024-04-01", "2024-04-01T18:00:00Z", home="147", away="111")
        self._game(300, "2024-09-01", "2024-09-01T18:00:00Z", home="119", away="147")
        # Judge (team 147): home vs BOS in April, away @ LAD in Sept.
        self._batter("592450", 200, H=1.0, AB=4.0, BB=1.0, HBP=0.0, SF=0.0, SH=0.0)
        self._batter("592450", 300, H=2.0, AB=3.0, BB=0.0, HBP=1.0, SF=0.0, SH=0.0)
        h = mlb_warehouse.get_player_history("592450", "batter_hits",
                                             player_name="Aaron Judge")
        self.assertTrue(h["found"])
        self.assertEqual(h["player"], "Aaron Judge")
        self.assertEqual(h["athlete_id"], "592450")
        self.assertEqual(h["values"], [2.0, 1.0])                # Sept first (recency)
        self.assertEqual(h["opponents"],
                         ["Los Angeles Dodgers", "Boston Red Sox"])
        self.assertEqual(h["home_aways"], [False, True])         # away @ LAD, home vs BOS
        self.assertEqual(h["at_bats"], [3.0, 4.0])
        self.assertEqual(h["plate_appearances"], [4.0, 5.0])     # AB+BB+HBP+SF+SH
        self.assertEqual(h["team_id"], "147")
        self.assertEqual(h["team_name"], "New York Yankees")
        self.assertEqual(h["minutes"], [0.0, 0.0])
        self.assertEqual([d[:10] for d in h["game_dates"]],
                         ["2024-09-01", "2024-04-01"])

    def test_pitcher_outs_ip_to_outs(self):
        self._game(400, "2024-07-04", "2024-07-04T18:00:00Z")
        self._pitcher("543037", 400, IP=6.1, K=8.0, ER=2.0)
        h = mlb_warehouse.get_player_history("543037", "pitcher_outs")
        self.assertEqual(h["values"], [19.0])                    # 6.1 IP → 19 outs
        self.assertIsNone(h["at_bats"][0])                       # pitcher: no PA/AB
        self.assertIsNone(h["plate_appearances"][0])
        self.assertEqual(h["stat_label"], "IP")

    def test_pitcher_strikeouts_raw(self):
        self._game(401, "2024-07-04", "2024-07-04T18:00:00Z")
        self._pitcher("543037", 401, IP=5.0, K=7.0, ER=1.0)
        h = mlb_warehouse.get_player_history("543037", "pitcher_strikeouts")
        self.assertEqual(h["values"], [7.0])

    def test_unsupported_prop_returns_none(self):
        self._game(500, "2024-07-04", "2024-07-04T18:00:00Z")
        self._batter("1", 500, H=1.0, AB=4.0)
        self.assertIsNone(mlb_warehouse.get_player_history("1", "batter_home_runs"))

    def test_no_rows_returns_none(self):
        self.assertIsNone(mlb_warehouse.get_player_history("nobody", "batter_hits"))

    def test_as_of_is_strict_leakage_cutoff(self):
        self._game(1, "2024-07-01", "2024-07-01T18:00:00Z")
        self._game(2, "2024-07-04", "2024-07-04T18:00:00Z")
        self._batter("1", 1, H=1.0, AB=4.0)
        self._batter("1", 2, H=2.0, AB=4.0)
        h = mlb_warehouse.get_player_history("1", "batter_hits",
                                             as_of_date="2024-07-04")
        self.assertEqual(h["values"], [1.0])                     # 07-04 game excluded

    def test_limit_caps_games(self):
        for pk, d in ((1, "2024-07-01"), (2, "2024-07-02"), (3, "2024-07-03")):
            self._game(pk, d, d + "T18:00:00Z")
            self._batter("1", pk, H=1.0, AB=4.0)
        h = mlb_warehouse.get_player_history("1", "batter_hits", n=2)
        self.assertEqual(len(h["values"]), 2)

    def test_disabled_returns_none(self):
        db_store.configure_engine(None)
        self.assertIsNone(mlb_warehouse.get_player_history("1", "batter_hits"))

    def test_excludes_spring_allstar_exhibition(self):
        # ESPN scope is regular+postseason; spring/all-star/exhibition are dropped
        # (and an All-Star game must not become rows[0] / the team source).
        self._game(1, "2024-07-01", "2024-07-01T18:00:00Z")               # regular
        self._game(2, "2024-07-16", "2024-07-16T18:00:00Z", game_type="A")  # all-star (newest)
        self._game(3, "2024-03-01", "2024-03-01T18:00:00Z", game_type="S")  # spring
        self._batter("592450", 1, H=1.0, AB=4.0)
        self._batter("592450", 2, H=0.0, AB=1.0)
        self._batter("592450", 3, H=3.0, AB=5.0)
        h = mlb_warehouse.get_player_history("592450", "batter_hits")
        self.assertEqual(h["values"], [1.0])                  # only the regular game
        self.assertEqual(h["team_name"], "New York Yankees")

    def test_season_filter_scopes_to_year(self):
        self._game(1, "2023-09-01", "2023-09-01T18:00:00Z", season=2023)
        self._game(2, "2024-04-01", "2024-04-01T18:00:00Z", season=2024)
        self._batter("1", 1, H=3.0, AB=4.0, season=2023)
        self._batter("1", 2, H=1.0, AB=4.0, season=2024)
        h = mlb_warehouse.get_player_history("1", "batter_hits", season=2024)
        self.assertEqual(h["values"], [1.0])                  # prior season excluded


class PlayerInputParityTests(_Backend, unittest.TestCase):
    """The model-input shadow lens reads the STORED facts (not a fresh boxscore)
    and diffs vs ESPN; only_espn also flags backfill gaps. ESPN side monkeypatched."""

    def setUp(self):
        super().setUp()
        for tid, nm in (("147", "New York Yankees"), ("111", "Boston Red Sox")):
            with db_store.get_engine().begin() as conn:
                conn.execute(insert(mlb_warehouse.mlb_team), {
                    "team_id": tid, "name": nm,
                    "name_norm": db_store.normalize_name(nm)})
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_player), {
                "player_id": "592450", "full_name": "Aaron Judge",
                "name_norm": db_store.normalize_name("Aaron Judge")})
        self.nm = db_store.normalize_name("Aaron Judge")

    def _game(self, game_pk, official_date, game_type="R"):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": game_pk, "official_date": official_date,
                "game_date": official_date + "T18:00:00Z", "season": 2024,
                "game_type": game_type, "home_team_id": "147",
                "away_team_id": "111"})

    def _batter(self, game_pk, H, athlete_id="592450"):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_batter_game), {
                "athlete_id": athlete_id, "game_pk": game_pk, "team_id": "147",
                "season_bucket": 2024, "H": H, "AB": 4.0})

    def test_warehouse_map_keys_and_dh_sum(self):
        self._game(1, "2024-07-04")
        self._game(2, "2024-07-04")          # split DH, same official date
        self._game(3, "2024-07-05")
        self._batter(1, 1.0)
        self._batter(2, 2.0)
        self._batter(3, 0.0)
        m, disp = parity._warehouse_player_game_stats(
            "2024-07-04", "2024-07-05", "batter", "batter_hits")
        self.assertEqual(m[(self.nm, "2024-07-04")], 3.0)        # DH summed
        self.assertEqual(m[(self.nm, "2024-07-05")], 0.0)
        self.assertEqual(disp[self.nm], "Aaron Judge")

    def test_window_and_game_type_exclusion(self):
        self._game(1, "2024-07-04")
        self._game(2, "2024-07-10")                              # out of window
        self._game(3, "2024-07-04", game_type="S")              # spring → excluded
        self._batter(1, 1.0)
        self._batter(2, 5.0)
        self._batter(3, 9.0)
        m, _ = parity._warehouse_player_game_stats(
            "2024-07-04", "2024-07-06", "batter", "batter_hits")
        self.assertEqual(m, {(self.nm, "2024-07-04"): 1.0})     # in-window regular only

    def test_pitcher_ip_to_outs(self):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_player), {
                "player_id": "543037", "full_name": "Gerrit Cole",
                "name_norm": db_store.normalize_name("Gerrit Cole")})
        self._game(1, "2024-07-04")
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_pitcher_game), {
                "athlete_id": "543037", "game_pk": 1, "team_id": "147",
                "season_bucket": 2024, "IP": 6.1, "K": 8.0, "ER": 2.0})
        cole = db_store.normalize_name("Gerrit Cole")
        m, _ = parity._warehouse_player_game_stats(
            "2024-07-04", "2024-07-04", "pitcher", "pitcher_strikeouts")
        self.assertEqual(m[(cole, "2024-07-04")], 8.0)          # K raw
        m2, _ = parity._warehouse_player_game_stats(
            "2024-07-04", "2024-07-04", "pitcher", "pitcher_outs")
        self.assertEqual(m2[(cole, "2024-07-04")], 19.0)        # 6.1 IP → 19 outs

    def test_player_input_parity_matches_espn(self):
        self._game(1, "2024-07-04")
        self._batter(1, 2.0)
        with mock.patch.object(parity, "_espn_player_game_stats",
                               return_value={(self.nm, "2024-07-04"): 2.0}):
            rep = parity.player_input_parity("2024-07-04", "2024-07-04")
        self.assertEqual(rep["compared"], 1)
        self.assertEqual(rep["matches"], 1)
        self.assertEqual(rep["only_espn"], 0)
        self.assertEqual(rep["source"], "warehouse_facts")

    def test_player_input_parity_flags_backfill_gap(self):
        # ESPN has 07-10 (prev-day 07-09 also absent, so no ±1-day remap) that the
        # warehouse hasn't ingested → surfaces as only_espn.
        self._game(1, "2024-07-04")
        self._batter(1, 2.0)
        with mock.patch.object(parity, "_espn_player_game_stats",
                               return_value={(self.nm, "2024-07-04"): 2.0,
                                             (self.nm, "2024-07-10"): 1.0}):
            rep = parity.player_input_parity("2024-07-04", "2024-07-10")
        self.assertEqual(rep["matches"], 1)
        self.assertEqual(rep["only_espn"], 1)                   # 07-10 not in warehouse


class TeamMarketReaderTests(_Backend, unittest.TestCase):
    """StatsAPI-native team-market model inputs: recent_games from mlb_game;
    win%/record + team defense from the standings snapshot (cumulative runs)."""

    def setUp(self):
        super().setUp()
        for tid, nm in (("147", "New York Yankees"), ("111", "Boston Red Sox"),
                        ("119", "Los Angeles Dodgers")):
            with db_store.get_engine().begin() as conn:
                conn.execute(insert(mlb_warehouse.mlb_team), {
                    "team_id": tid, "name": nm,
                    "name_norm": db_store.normalize_name(nm)})

    def _game(self, pk, official_date, game_date, home, away, hs, as_,
              status="Final", game_type="R", season=2024):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": pk, "official_date": official_date,
                "game_date": game_date, "season": season, "game_type": game_type,
                "home_team_id": home, "away_team_id": away,
                "home_score": hs, "away_score": as_, "status": status})

    def _stand(self, tid, w, losses, wp, rs, ra, as_of="2024-07-04", season=2024):
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_team_standings), {
                "team_id": tid, "season": season, "as_of_date": as_of,
                "wins": w, "losses": losses, "win_pct": wp,
                "runs_scored": rs, "runs_allowed": ra})

    def test_parse_standings_captures_runs(self):
        raw = {"records": [{"teamRecords": [
            {"team": {"id": 147}, "wins": 55, "losses": 30,
             "winningPercentage": ".647", "runsScored": 420, "runsAllowed": 330}]}]}
        row = mlb_warehouse.parse_standings(raw, 2024, "2024-07-04")[0]
        self.assertEqual(row["runs_scored"], 420)
        self.assertEqual(row["runs_allowed"], 330)

    def test_get_team_games_shape_order_and_exclusions(self):
        self._game(1, "2024-07-01", "2024-07-01T18:00:00Z", "147", "111", 5, 3)
        self._game(2, "2024-07-05", "2024-07-06T00:10:00Z", "119", "147", 4, 2)
        self._game(3, "2024-07-03", "2024-07-03T18:00:00Z", "147", "111", 1, 2,
                   status="Live")                       # not final → excluded
        self._game(4, "2024-03-01", "2024-03-01T18:00:00Z", "147", "111", 9, 0,
                   game_type="S")                       # spring → excluded
        games = mlb_warehouse.get_team_games("New York Yankees")
        self.assertEqual([g["date"][:10] for g in games],
                         ["2024-07-06", "2024-07-01"])  # most-recent-first, final reg
        g0 = games[0]
        self.assertEqual((g0["home_team"], g0["away_team"]),
                         ("Los Angeles Dodgers", "New York Yankees"))
        self.assertEqual((g0["home_score"], g0["away_score"], g0["total_score"]),
                         (4, 2, 6))

    def test_get_team_games_leakage_cutoff(self):
        self._game(1, "2024-07-01", "2024-07-01T18:00:00Z", "147", "111", 5, 3)
        self._game(2, "2024-07-05", "2024-07-05T18:00:00Z", "147", "111", 4, 2)
        past = mlb_warehouse.get_team_games("New York Yankees",
                                            as_of_date="2024-07-05")
        self.assertEqual([g["date"][:10] for g in past], ["2024-07-01"])

    def test_get_team_games_excludes_suspended(self):
        # A suspended game reports abstractGameState 'Final' with a PARTIAL score;
        # detailed_state gates it out (mlb_game holds every scheduled game).
        self._game(1, "2024-07-01", "2024-07-01T18:00:00Z", "147", "111", 5, 3)
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_game), {
                "game_pk": 2, "official_date": "2024-07-02",
                "game_date": "2024-07-02T18:00:00Z", "season": 2024,
                "game_type": "R", "home_team_id": "147", "away_team_id": "111",
                "home_score": 3, "away_score": 1, "status": "Final",
                "detailed_state": "Suspended: Rain"})
        games = mlb_warehouse.get_team_games("New York Yankees")
        self.assertEqual([g["date"][:10] for g in games], ["2024-07-01"])

    def test_team_name_canonical_tolerant(self):
        # tolerant input spelling → canonical mlb_team.name (for the flip's
        # exact-match consumers).
        self.assertEqual(
            mlb_warehouse.team_name_canonical("new york yankees"),
            "New York Yankees")
        self.assertIsNone(mlb_warehouse.team_name_canonical("Nowhere FC"))

    def test_get_team_standings(self):
        self._stand("147", 55, 30, 0.647, 420, 330)
        self.assertEqual(
            mlb_warehouse.get_team_standings("New York Yankees", season=2024),
            {"record": "55-30", "wins": 55, "losses": 30, "win_pct": 0.647,
             "runs_scored": 420, "runs_allowed": 330})

    def test_get_team_standings_latest_snapshot_and_asof(self):
        self._stand("147", 40, 30, 0.571, 300, 280, as_of="2024-06-01")
        self._stand("147", 55, 30, 0.647, 420, 330, as_of="2024-07-04")
        self.assertEqual(mlb_warehouse.get_team_standings(
            "New York Yankees", season=2024)["wins"], 55)           # latest
        self.assertEqual(mlb_warehouse.get_team_standings(
            "New York Yankees", season=2024, as_of_date="2024-06-15")["wins"], 40)

    def test_get_team_defense_from_standings_runs(self):
        self._stand("147", 50, 50, 0.5, 500, 400)   # 100 g, 400 allowed → 4.0
        self._stand("111", 40, 40, 0.5, 360, 480)   #  80 g, 480 allowed → 6.0
        d = mlb_warehouse.get_team_defense(season=2024)
        self.assertAlmostEqual(d["New York Yankees"], 4.0)
        self.assertAlmostEqual(d["Boston Red Sox"], 6.0)

    def test_get_team_defense_uses_latest_snapshot(self):
        self._stand("147", 10, 10, 0.5, 100, 100, as_of="2024-06-01")  # 5.0
        self._stand("147", 50, 50, 0.5, 500, 400, as_of="2024-07-04")  # 4.0
        self.assertAlmostEqual(
            mlb_warehouse.get_team_defense(season=2024)["New York Yankees"], 4.0)


class TeamDefenseParityTests(_Backend, unittest.TestCase):
    """team_defense_parity diffs the standings-derived warehouse map vs ESPN
    (monkeypatched); reuses the pure diff core."""

    def test_diff_warehouse_vs_espn(self):
        for tid, nm in (("147", "New York Yankees"),):
            with db_store.get_engine().begin() as conn:
                conn.execute(insert(mlb_warehouse.mlb_team), {
                    "team_id": tid, "name": nm,
                    "name_norm": db_store.normalize_name(nm)})
        with db_store.get_engine().begin() as conn:
            conn.execute(insert(mlb_warehouse.mlb_team_standings), {
                "team_id": "147", "season": 2024, "as_of_date": "2024-07-04",
                "wins": 50, "losses": 50, "win_pct": 0.5,
                "runs_scored": 500, "runs_allowed": 400})     # → 4.0/game
        nm = db_store.normalize_name("New York Yankees")
        with mock.patch.object(parity, "_espn_team_defense_map",
                               return_value={nm: 4.1}):        # within 0.25 tol
            rep = parity.team_defense_parity(2024)
        self.assertEqual(rep["compared"], 1)
        self.assertEqual(rep["matches"], 1)
        self.assertEqual(rep["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
