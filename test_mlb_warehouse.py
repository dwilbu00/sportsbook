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


if __name__ == "__main__":
    unittest.main()
