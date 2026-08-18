"""mlb_warehouse.py — MLB StatsAPI medallion: silver dims + bronze + game facts.

Read-only DUAL-RUN. Populates durable, StatsAPI-native dims (team / game / player),
a per-season standings fact, a provider→MLBAM alias scaffold, a transient bronze
raw-JSON landing table (all P1), and — P2 — the game-centric per-game batter /
pitcher stat FACTS (mlb_batter_game / mlb_pitcher_game), all in parallel with the
live ESPN path. As of the P6 teardown these tables are the SOLE MLB source (live
history, team markets, grading, calibration, and the offline backtests) — ESPN was
removed for MLB; the read shapes are pinned by golden fixture tests in
test_mlb_warehouse.py (the ESPN-vs-warehouse parity harness has been retired).

House conventions (mirrors gamelog_store.py / player_id_map.py):
  * Owns its OWN SQLAlchemy Core MetaData + Tables + create_all() (create_all is
    TEST-ONLY / SQLite; prod DDL is the hand-run sql/schema.sql, kept in lockstep
    and guarded by test_mlb_warehouse.py::SchemaParityTests).
  * Reuses db_store.get_engine()/reconcile()/normalize_name()/enabled().
  * Reuses mlb_starters._get/_read_cache/_write_cache/_is_genuine_final for the
    StatsAPI HTTP + cache + finalization gate (no new HTTP stack).

Design notes carried from the re-architecture doc (11-mlb-statsapi-rearchitecture):
  * Natural keys from StatsAPI are the dim PKs (team_id / game_pk / player_id);
    the games dim is the spine (home/away FK → team dim).
  * game_date is the FULL ISO timestamp (its time disambiguates split
    doubleheaders); official_date is the YYYY-MM-DD play date used for joins.
  * reconcile() DELETES in-scope rows absent from the desired set, so every
    accumulating-dim upsert here is SCOPED to its single natural key (the
    athlete_id_cache idiom) — a call can never delete a sibling row.
  * P2 game facts are game-centric + MLBAM-native: one row per (athlete, game_pk)
    straight from the boxscore, so game_pk is ALWAYS present → the natural key is a
    plain UNIQUE(athlete_id, game_pk) (no NULL/filtered-index dance). team_id is
    NOT NULL but NOT in the key; season_bucket is derived + indexed, NOT in the
    key; game_date/opponent/is_home are NOT stored (the reader rejoins mlb_game).
  * The reader (_game_log) orders MOST-RECENT-FIRST by the JOINED game_date DESC
    (play date), game_pk DESC tiebreak — surrogate id is NOT recency once the
    writer is a surgical reconcile (the §6 mandatory ordering fix, done natively).

MLB only. NBA/NFL stay on ESPN. The ESPN gamelog_store.py path is UNTOUCHED.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import threading
import time

from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, MetaData, String,
    Table, Text, UniqueConstraint, and_, bindparam, delete, func, insert, not_,
    or_, select, update,
)
from sqlalchemy.exc import OperationalError

import db_store
import mlb_starters

# ─────────────────────────────────────────────────────────── schema (own MetaData)
_META = MetaData()

mlb_bronze = Table(
    "mlb_bronze", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kind", String(16), nullable=False),          # schedule|boxscore|standings|teams
    Column("natural_ref", String(64), nullable=False),   # gamePk | date | season:asof
    Column("payload", Text, nullable=False),             # raw JSON
    Column("fetched_at", Float),
    Column("processed_at", Float),                        # NULL = pending
    UniqueConstraint("kind", "natural_ref", name="uq_mlb_bronze"),
)

mlb_team = Table(
    "mlb_team", _META,
    Column("team_id", String(32), primary_key=True),     # MLBAM (natural PK)
    Column("name", String(160)),
    Column("name_norm", String(160)),
    Column("abbreviation", String(16)),
    Column("league_id", String(16)),
    Column("division_id", String(16)),
    Column("fetched_at", Float),
    Index("ix_mlb_team_name", "name_norm"),
)

# Venue dimension (Batch A run_env): venue_id is the physical park a game is played in
# (already on mlb_game, from the schedule). Keying park + weather on venue_id — not the
# home team's name — is correct for neutral-site games (London/Tokyo/Field of Dreams/
# Sacramento), relocations, and the A's/Rays venue limbo. Populated by build_venue_dim
# (StatsAPI /venues hydrate=location for name+lat/lon) joined to the authored priors
# (park_factors / weather_factors.MLB_PARK_GEO). park_hits/park_runs default 1.0
# (neutral) for an unmatched venue → run_env fails open.
mlb_venue = Table(
    "mlb_venue", _META,
    Column("venue_id", String(16), primary_key=True),    # StatsAPI venue id
    Column("name", String(160)),
    Column("team_id", String(32)),                       # canonical home team (None = neutral)
    Column("team_name", String(160)),                    # for park/geo prior resolution
    Column("lat", Float),
    Column("lon", Float),
    Column("cf_bearing", Float),                          # home plate → CF compass degrees
    Column("park_hits", Float),                           # authored prior (1.0 = neutral)
    Column("park_runs", Float),
    Column("elevation_ft", Float),                        # nullable
    Column("roof", String(16)),                           # open|retractable|dome
    Column("fetched_at", Float),
    Index("ix_mlb_venue_team", "team_id"),
)

# Per-game weather (Batch A run_env), keyed by (venue_id, weather_date) so it joins
# mlb_game on venue_id AND official_date = weather_date. First-pitch-hour conditions
# from Visual Crossing (mlb_warehouse.build_weather), used baseline-relative in the
# weather run_env term. Actual weather is a pre-outcome game condition (not leakage).
weather_game = Table(
    "weather_game", _META,
    Column("venue_id", String(16), primary_key=True),
    Column("weather_date", String(10), primary_key=True),  # YYYY-MM-DD (official_date)
    Column("temp_f", Float),
    Column("humidity", Float),                             # relative %, 0-100
    Column("pressure_mb", Float),                          # sea-level (structural park
                                                          # climate stays in park factor)
    Column("wind_mph", Float),
    Column("wind_dir_deg", Float),                         # direction wind blows FROM
    Column("first_pitch_utc", String(40)),                # the game hour sampled (audit)
    Column("source", String(24)),                         # "visualcrossing"
    Column("fetched_at", Float),
)

mlb_game = Table(
    "mlb_game", _META,
    Column("game_pk", Integer, primary_key=True, autoincrement=False),  # MLBAM (supplied)
    Column("game_date", String(40)),                     # FULL ISO timestamp (UTC)
    Column("official_date", String(10)),                 # YYYY-MM-DD play date
    Column("season", Integer),
    Column("game_type", String(4)),                      # R|S|A|E|D|F|L|W|P (StatsAPI gameType)
    Column("game_number", Integer),                      # doubleheader game #
    Column("double_header", String(4)),                  # N|Y|S
    Column("home_team_id", String(32), ForeignKey("mlb_team.team_id",
                                                   name="fk_mlb_game_home")),
    Column("away_team_id", String(32), ForeignKey("mlb_team.team_id",
                                                   name="fk_mlb_game_away")),
    Column("venue_id", String(16)),
    # Home-plate umpire (Batch A #4): parsed from the boxscore `officials` block during
    # ingest; joins to umpire_asof by hp_umpire_id. NULL on pre-capture / schedule-only
    # rows (fails open to a neutral umpire run_env).
    Column("hp_umpire_id", String(32)),
    Column("hp_umpire_name", String(160)),
    Column("status", String(32)),                        # abstractGameState
    Column("detailed_state", String(64)),                # detailedState
    Column("home_score", Float),
    Column("away_score", Float),
    Column("fetched_at", Float),
    Index("ix_mlb_game_official_date", "official_date"),
    Index("ix_mlb_game_teams", "official_date", "home_team_id", "away_team_id"),
    # (season, status) COVERING scan (Azure missing-index rec): season-final-games reads
    # with scores/teams/dates. Azure keyed away_score too, but scores aren't filter
    # predicates so it's moved into the INCLUDE. mssql_include ignored off SQL Server.
    Index("ix_mlb_game_season_status", "season", "status",
          mssql_include=["away_score", "home_score", "away_team_id", "home_team_id",
                         "detailed_state", "game_date", "game_type", "official_date"]),
)

mlb_player = Table(
    "mlb_player", _META,
    Column("player_id", String(32), primary_key=True),   # MLBAM (natural PK)
    Column("full_name", String(160)),
    Column("name_norm", String(160)),
    Column("primary_position", String(16)),
    Column("is_pitcher", Boolean),
    Column("bats", String(8)),                           # nullable in P1
    Column("throws", String(8)),                         # nullable in P1
    Column("fetched_at", Float),
    Index("ix_mlb_player_name", "name_norm"),
)

mlb_team_standings = Table(
    "mlb_team_standings", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("team_id", String(32),
           ForeignKey("mlb_team.team_id", name="fk_mlb_standings_team"),
           nullable=False),
    Column("season", Integer, nullable=False),
    Column("as_of_date", String(16), nullable=False),    # YYYY-MM-DD cutoff
    Column("wins", Integer),
    Column("losses", Integer),
    Column("win_pct", Float),
    Column("runs_scored", Integer),      # season cumulative (StatsAPI runsScored)
    Column("runs_allowed", Integer),     # season cumulative (StatsAPI runsAllowed)
    Column("fetched_at", Float),
    UniqueConstraint("team_id", "season", "as_of_date",
                     name="uq_mlb_team_standings"),
    Index("ix_mlb_team_standings_asof", "season", "as_of_date"),
)

player_alias = Table(
    "player_alias", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("provider", String(32), nullable=False),      # oddsapi|sfbb|...
    Column("provider_key", String(200), nullable=False), # provider name or id
    Column("mlb_player_id", String(32), nullable=False), # MLBAM (value, not FK)
    Column("confidence", Float),
    Column("resolution_method", String(32)),             # alias|roster_exact|fuzzy_single|seed
    Column("valid_from", String(40)),
    Column("valid_to", String(40)),
    Column("fetched_at", Float),
    UniqueConstraint("provider", "provider_key", name="uq_player_alias"),
    Index("ix_player_alias_mlb", "mlb_player_id"),
)

# ── StatsAPI-native per-game stat facts (P2, game-centric, dual-run) ──────────
# One row per (athlete, game), derived straight from the boxscore, so athlete_id
# is the MLBAM id and game_pk comes from the games dim — game_pk is ALWAYS
# present, so the natural key is a plain UNIQUE(athlete_id, game_pk) (no NULL /
# filtered-index dance). team_id is NOT NULL but NOT in the key: it is an
# attribute functionally dependent on (athlete, game) (§6). season_bucket is
# derived (FD on the game) and kept only as a plain indexed attribute for cheap
# season scans — deliberately NOT in the UNIQUE. Denormalized game_date/opponent/
# is_home are NOT stored: the reader/gold view rejoins fact→mlb_game→mlb_team.
# Surrogate id stays PK. These live ALONGSIDE the ESPN-sourced *_gamelog tables
# in gamelog_store.py (untouched) — nothing app-facing consumes these until P4.
# HR/TB/RBI appended (2026-08-11). TB/RBI are now fact-servable (in _ACTUAL_STAT_SPEC
# → get_actual_stat/get_player_history/resolve_actual) for the batter_total_bases /
# batter_rbis props; get_player_history SKIPS a game whose stat column is NULL so
# pre-backfill (un-populated) rows can't corrupt the model input — a full re-backfill
# just widens coverage, it isn't a correctness prerequisite. batter_home_runs (HR) is
# captured but not yet an odds market, so it stays out of _ACTUAL_STAT_SPEC.
_BATTER_GAME_STATS = ("AB", "H", "SO", "BB", "HBP", "SF", "SH", "HR", "TB", "RBI")
# BB/BF/HR/HBP/GS appended 2026-08-17 (Tier A #1c) to unlock the as-of pitcher
# feature set the additive expected-runs model wants: K%/BB% (per BF), FIP
# components (HR/HBP/BB/K), and GS==0 relief classification for the team-bullpen
# aggregate. NULL on pre-ALTER/pre-re-backfill rows (get_player_history skips NULL
# stat cells, so partial coverage is safe). ⚠ COORDINATED: the prod ALTER (see
# sql/schema.sql) MUST run BEFORE this code deploys — _game_log SELECTs *stat_cols.
_PITCHER_GAME_STATS = ("IP", "K", "ER", "BB", "BF", "HR", "HBP", "GS")


def _game_fact_table(name, stat_cols):
    return Table(
        name, _META,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("athlete_id", String(32), nullable=False),        # MLBAM (natural key part)
        Column("game_pk", Integer,
               ForeignKey("mlb_game.game_pk", name=f"fk_{name}_game"),
               nullable=False),                                  # games dim (natural key part)
        Column("team_id", String(32),
               ForeignKey("mlb_team.team_id", name=f"fk_{name}_team"),
               nullable=False),                                  # MLBAM (attribute, NOT in key)
        Column("season_bucket", Integer),                        # derived; indexed, NOT in key
        *[Column(c, Float) for c in stat_cols],
        Column("fetched_at", Float),
        UniqueConstraint("athlete_id", "game_pk", name=f"uq_{name}"),
        Index(f"ix_{name}_athlete", "athlete_id", "season_bucket"),
        # game_pk-first COVERING index (Azure missing-index rec): the uq is
        # (athlete_id, game_pk) so it can't seek game_pk-first, and grading / as-of
        # reads that filter by game_pk otherwise scan. Covers all stat cols -> index-
        # only. Append-only ingest, so the extra INCLUDE width is cheap. mssql_include
        # is ignored off SQL Server (SQLite tests build the 2-col index).
        Index(f"ix_{name}_gamepk", "game_pk", "season_bucket",
              mssql_include=["athlete_id", *stat_cols]),
    )


mlb_batter_game = _game_fact_table("mlb_batter_game", _BATTER_GAME_STATS)
mlb_pitcher_game = _game_fact_table("mlb_pitcher_game", _PITCHER_GAME_STATS)

# Column-name SPECs — SchemaParityTests asserts these equal the Table columns.
_BRONZE_COLS = ("id", "kind", "natural_ref", "payload", "fetched_at", "processed_at")
_TEAM_COLS = ("team_id", "name", "name_norm", "abbreviation", "league_id",
              "division_id", "fetched_at")
_GAME_COLS = ("game_pk", "game_date", "official_date", "season", "game_type",
              "game_number", "double_header", "home_team_id", "away_team_id",
              "venue_id", "hp_umpire_id", "hp_umpire_name", "status",
              "detailed_state", "home_score", "away_score", "fetched_at")
_VENUE_COLS = ("venue_id", "name", "team_id", "team_name", "lat", "lon",
               "cf_bearing", "park_hits", "park_runs", "elevation_ft", "roof",
               "fetched_at")
_WEATHER_GAME_COLS = ("venue_id", "weather_date", "temp_f", "humidity",
                      "pressure_mb", "wind_mph", "wind_dir_deg", "first_pitch_utc",
                      "source", "fetched_at")
_PLAYER_COLS = ("player_id", "full_name", "name_norm", "primary_position",
                "is_pitcher", "bats", "throws", "fetched_at")
_STANDINGS_COLS = ("id", "team_id", "season", "as_of_date", "wins", "losses",
                   "win_pct", "runs_scored", "runs_allowed", "fetched_at")
_ALIAS_COLS = ("id", "provider", "provider_key", "mlb_player_id", "confidence",
               "resolution_method", "valid_from", "valid_to", "fetched_at")
_BATTER_GAME_COLS = ("id", "athlete_id", "game_pk", "team_id", "season_bucket",
                     *_BATTER_GAME_STATS, "fetched_at")
_PITCHER_GAME_COLS = ("id", "athlete_id", "game_pk", "team_id", "season_bucket",
                      *_PITCHER_GAME_STATS, "fetched_at")


def create_all():
    """Create the warehouse tables. TEST-ONLY (SQLite); prod DDL is hand-run."""
    _META.create_all(db_store.get_engine())


def enabled():
    return db_store.enabled()


# ───────────────────────────────────────────────────────────────────── coercers
def _s(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _now():
    return time.time()


def _today():
    return datetime.date.today().isoformat()


def _current_season():
    return datetime.date.today().year


# ─────────────────────────────────────────────── StatsAPI fetchers (reuse mlb_starters)
_SCHED_LIVE_TTL = 900            # seconds — live/unfinished slate
_SCHED_FINAL_TTL = 24 * 3600     # seconds — a fully-final slate is immutable
_STANDINGS_TTL = 3600            # seconds — standings move daily

# A slate's boxscore fetches are the backfill bottleneck (one StatsAPI round-trip
# each). They are network-bound + independent, so ingest_date fetches them
# CONCURRENTLY (DB writes stay serial under _WRITE_LOCK). StatsAPI is free but be a
# polite client — bounded pool, env-tunable via ODI_MLB_BOXSCORE_WORKERS.
_BOXSCORE_FETCH_WORKERS = 8


def _boxscore_fetch_workers():
    try:
        n = int(os.environ.get("ODI_MLB_BOXSCORE_WORKERS") or _BOXSCORE_FETCH_WORKERS)
    except (TypeError, ValueError):
        return _BOXSCORE_FETCH_WORKERS
    return max(1, min(32, n))


def _schedule_all_final(raw):
    games = [g for d in (raw or {}).get("dates", []) or []
             for g in d.get("games", []) or []]
    if not games:
        return False
    for g in games:
        st = g.get("status") or {}
        if not mlb_starters._is_genuine_final(
                {"status": st.get("abstractGameState"),
                 "detailedState": st.get("detailedState")}):
            return False
    return True


def fetch_teams(season):
    """Raw /teams payload for a season (id/name/abbr/league/division). Cached 7d."""
    cache = f"warehouse_teams_{season}"
    cached = mlb_starters._read_cache(cache, max_age=7 * 24 * 3600)
    if cached is not None:
        return cached
    data = mlb_starters._get("teams", {"sportId": 1, "season": season})
    mlb_starters._write_cache(cache, data)
    return data


def fetch_players(season):
    """Raw /sports/1/players roster for a season. Unlike the boxscore person object
    (id/fullName only), each person here carries batSide.code / pitchHand.code — the
    source for mlb_player handedness (see populate_handedness). Cached 7d."""
    cache = f"warehouse_players_{season}"
    cached = mlb_starters._read_cache(cache, max_age=7 * 24 * 3600)
    if cached is not None:
        return cached
    data = mlb_starters._get("sports/1/players", {"season": season})
    mlb_starters._write_cache(cache, data)
    return data


def fetch_schedule(date):
    """Raw /schedule payload (with linescore hydrate) for one YYYY-MM-DD date.

    Adaptive TTL: a fully-final slate is served from the long cache; otherwise a
    short live cache (mirrors mlb_starters.get_schedule_index)."""
    cache = f"warehouse_schedule_{date}"
    cached = mlb_starters._read_cache(cache, max_age=_SCHED_FINAL_TTL)
    if cached is not None and _schedule_all_final(cached):
        return cached
    fresh = mlb_starters._read_cache(cache, max_age=_SCHED_LIVE_TTL)
    if fresh is not None:
        return fresh
    data = mlb_starters._get(
        "schedule", {"sportId": 1, "date": date, "hydrate": "linescore"})
    mlb_starters._write_cache(cache, data)
    return data


def fetch_boxscore(game_pk, max_age=_SCHED_FINAL_TTL):
    """Raw /game/{gamePk}/boxscore payload. A genuine-final box is immutable → the
    default long cache; grading a recent ("active") game passes a SHORT max_age (or
    0) to pick up official-scorer corrections before they settle."""
    cache = f"warehouse_boxscore_{game_pk}"
    if max_age:
        cached = mlb_starters._read_cache(cache, max_age=max_age)
        if cached is not None:
            return cached
    data = mlb_starters._get(f"game/{game_pk}/boxscore")
    mlb_starters._write_cache(cache, data)
    return data


def fetch_standings(season):
    """Raw /standings payload (both leagues, regular season) for a season."""
    cache = f"warehouse_standings_{season}"
    cached = mlb_starters._read_cache(cache, max_age=_STANDINGS_TTL)
    if cached is not None:
        return cached
    data = mlb_starters._get(
        "standings",
        {"leagueId": "103,104", "season": season,
         "standingsTypes": "regularSeason"})
    mlb_starters._write_cache(cache, data)
    return data


# ───────────────────────────────────────────────────────── pure parse / derive
def parse_teams(raw):
    """Raw /teams payload → mlb_team dim rows (sportId==1 only)."""
    out = []
    for t in (raw or {}).get("teams", []) or []:
        if (t.get("sport") or {}).get("id") != 1:
            continue
        tid = t.get("id")
        if tid is None:
            continue
        name = t.get("name")
        out.append({
            "team_id": str(tid),
            "name": name,
            "name_norm": db_store.normalize_name(name or ""),
            "abbreviation": t.get("abbreviation"),
            "league_id": _s((t.get("league") or {}).get("id")),
            "division_id": _s((t.get("division") or {}).get("id")),
            "fetched_at": _now(),
        })
    return out


def parse_schedule(raw, season):
    """Raw /schedule payload → (mlb_game rows, minimal mlb_team rows).

    The minimal team rows (id + name) guarantee the mlb_game home/away FKs resolve
    even before the full /teams enrich runs."""
    games = []
    teams = {}
    for d in (raw or {}).get("dates", []) or []:
        for g in d.get("games", []) or []:
            pk = g.get("gamePk")
            if pk is None:
                continue
            t = g.get("teams", {}) or {}
            home = t.get("home") or {}
            away = t.get("away") or {}
            home_team = home.get("team") or {}
            away_team = away.get("team") or {}
            for tm in (home_team, away_team):
                if tm.get("id") is not None:
                    teams[str(tm["id"])] = {
                        "team_id": str(tm["id"]),
                        "name": tm.get("name"),
                        "name_norm": db_store.normalize_name(tm.get("name") or ""),
                        "fetched_at": _now(),
                    }
            status = g.get("status") or {}
            hs = home.get("score")
            as_ = away.get("score")
            ls = (g.get("linescore") or {}).get("teams") or {}
            if not isinstance(hs, (int, float)):
                hs = (ls.get("home") or {}).get("runs")
            if not isinstance(as_, (int, float)):
                as_ = (ls.get("away") or {}).get("runs")
            hid = home_team.get("id")
            aid = away_team.get("id")
            games.append({
                "game_pk": int(pk),
                "game_date": g.get("gameDate"),
                "official_date": g.get("officialDate"),
                "season": _i(g.get("season")) or season,
                "game_type": _s(g.get("gameType")),      # R=regular, A=all-star, S=spring, P/D/F/L/W=postseason
                "game_number": _i(g.get("gameNumber")),
                "double_header": g.get("doubleHeader"),
                "home_team_id": str(hid) if hid is not None else None,
                "away_team_id": str(aid) if aid is not None else None,
                "venue_id": _s((g.get("venue") or {}).get("id")),
                "status": status.get("abstractGameState"),
                "detailed_state": status.get("detailedState"),
                "home_score": _f(hs),
                "away_score": _f(as_),
                "fetched_at": _now(),
            })
    return games, list(teams.values())


def parse_standings(raw, season, as_of_date):
    """Raw /standings payload → mlb_team_standings snapshot rows."""
    out = []
    for rec in (raw or {}).get("records", []) or []:
        for tr in rec.get("teamRecords", []) or []:
            tm = tr.get("team") or {}
            tid = tm.get("id")
            if tid is None:
                continue
            out.append({
                "team_id": str(tid),
                "season": int(season),
                "as_of_date": as_of_date,
                "wins": _i(tr.get("wins")),
                "losses": _i(tr.get("losses")),
                "win_pct": _f(tr.get("winningPercentage")),
                "runs_scored": _i(tr.get("runsScored")),
                "runs_allowed": _i(tr.get("runsAllowed")),
                "fetched_at": _now(),
            })
    return out


def _is_pitcher_pos(pos):
    pos = pos or {}
    if pos.get("abbreviation") == "P":
        return True
    if str(pos.get("type") or "").lower() == "pitcher":
        return True
    return str(pos.get("code") or "") == "1"


def parse_boxscore_players(box, game=None):
    """Raw boxscore → mlb_player dim rows (one per person, deduped)."""
    out = {}
    teams = (box or {}).get("teams", {}) or {}
    for side in ("home", "away"):
        players = (teams.get(side) or {}).get("players", {}) or {}
        for _pk, p in players.items():
            person = p.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            name = person.get("fullName")
            pos = p.get("position") or {}
            out[str(pid)] = {
                "player_id": str(pid),
                "full_name": name,
                "name_norm": db_store.normalize_name(name or ""),
                "primary_position": pos.get("abbreviation"),
                "is_pitcher": _is_pitcher_pos(pos),
                "fetched_at": _now(),
            }
    return list(out.values())


def derive_batter_rows(box, game):
    """Raw boxscore → per-batter stat rows (athlete_id (MLBAM), game_pk, team_id,
    AB/H/SO/BB/HBP/SF/SH). Only players who came to the plate (or carry a batting
    order). DEDUPED per athlete_id: a player can legitimately appear under BOTH
    teams in one boxscore — the 2024-06-26 Danny Jansen game (suspended, traded,
    resumed → listed for both clubs) — so keep the line where he actually batted
    (max plate appearances), which also picks the correct team_id. Otherwise the
    (athlete_id, game_pk) natural key would collide."""
    by_ath = {}                                    # athlete_id -> (participation, row)
    teams = (box or {}).get("teams", {}) or {}
    gid = (game or {}).get("game_pk")
    for side in ("home", "away"):
        team_id = (game or {}).get(f"{side}_team_id")
        players = (teams.get(side) or {}).get("players", {}) or {}
        for _pk, p in players.items():
            bat = ((p.get("stats") or {}).get("batting")) or {}
            if not bat:
                continue
            pa = _i(bat.get("plateAppearances")) or 0
            ab = _i(bat.get("atBats")) or 0
            bb = _i(bat.get("baseOnBalls")) or 0
            hbp = _i(bat.get("hitByPitch")) or 0
            sf = _i(bat.get("sacFlies")) or 0
            sh = _i(bat.get("sacBunts")) or 0
            has_order = str(p.get("battingOrder") or "") != ""
            if pa <= 0 and (ab + bb + hbp + sf + sh) <= 0 and not has_order:
                continue
            person = p.get("person") or {}
            aid = str(person.get("id"))
            row = {
                "athlete_id": aid,
                "game_pk": gid,
                "team_id": team_id,
                "AB": _f(bat.get("atBats")),
                "H": _f(bat.get("hits")),
                "SO": _f(bat.get("strikeOuts")),
                "BB": _f(bat.get("baseOnBalls")),
                "HBP": _f(bat.get("hitByPitch")),
                "SF": _f(bat.get("sacFlies")),
                "SH": _f(bat.get("sacBunts")),
                "HR": _f(bat.get("homeRuns")),      # StatsAPI gives these directly
                "TB": _f(bat.get("totalBases")),    # (HR = 4 TB); no derivation needed
                "RBI": _f(bat.get("rbi")),
            }
            participation = pa if pa > 0 else (ab + bb + hbp + sf + sh)
            prev = by_ath.get(aid)
            if prev is None or participation > prev[0]:
                by_ath[aid] = (participation, row)
    return [row for _participation, row in by_ath.values()]


def derive_pitcher_rows(box, game):
    """Raw boxscore → per-pitcher stat rows: athlete_id, game_pk, team_id, IP/K/ER.
    IP is the base-3 float ("6.1" == 6IP + 1 out). DEDUPED per athlete_id (keep the
    max-IP line) for the same both-teams-in-one-game anomaly derive_batter_rows
    guards against, so the (athlete_id, game_pk) natural key can't collide."""
    by_ath = {}                                    # athlete_id -> (outs, row)
    teams = (box or {}).get("teams", {}) or {}
    gid = (game or {}).get("game_pk")
    for side in ("home", "away"):
        team_id = (game or {}).get(f"{side}_team_id")
        players = (teams.get(side) or {}).get("players", {}) or {}
        for _pk, p in players.items():
            pit = ((p.get("stats") or {}).get("pitching")) or {}
            if not pit:
                continue
            ip_raw = pit.get("inningsPitched")
            if ip_raw in (None, ""):
                continue
            person = p.get("person") or {}
            aid = str(person.get("id"))
            row = {
                "athlete_id": aid,
                "game_pk": gid,
                "team_id": team_id,
                "IP": _f(ip_raw),
                "K": _f(pit.get("strikeOuts")),
                "ER": _f(pit.get("earnedRuns")),
                # Tier A #1c: K%/BB% (per BF), FIP components, relief classification.
                "BB": _f(pit.get("baseOnBalls")),
                "BF": _f(pit.get("battersFaced")),
                "HR": _f(pit.get("homeRuns")),
                "HBP": _f(pit.get("hitByPitch")),
                "GS": _f(pit.get("gamesStarted")),   # 1 = started, 0 = relief
            }
            outs = _ip_to_outs(_f(ip_raw)) or 0
            prev = by_ath.get(aid)
            if prev is None or outs > prev[0]:
                by_ath[aid] = (outs, row)
    return [row for _outs, row in by_ath.values()]


# ─────────────────────────────────────────────────────────── bronze + silver writes
_WRITE_LOCK = threading.Lock()
_TEAMS_ENSURED = set()  # in-process memo: seasons whose full team dim is loaded


def _land_bronze(conn, kind, ref, payload_obj):
    """Upsert the one live raw payload for a natural ref (leaves processed_at
    untouched on update; NULL on insert)."""
    db_store.reconcile(
        conn, mlb_bronze,
        [{"kind": kind, "natural_ref": str(ref),
          "payload": json.dumps(payload_obj), "fetched_at": _now()}],
        ("kind", "natural_ref"),
        scope={"kind": kind, "natural_ref": str(ref)},
        ignore_cols=("fetched_at",))


def _mark_bronze_processed(conn, kind, ref):
    conn.execute(
        update(mlb_bronze)
        .where((mlb_bronze.c.kind == kind)
               & (mlb_bronze.c.natural_ref == str(ref)))
        .values(processed_at=_now()))


def purge_processed_boxscores():
    """Maintenance: delete boxscore bronze rows already processed. Bronze is
    semi-temporary; a finalized boxscore's payload is only needed until the silver
    it feeds is written. Returns rows deleted."""
    if not enabled():
        return 0
    engine = db_store.get_engine()
    with engine.begin() as conn:
        res = conn.execute(
            delete(mlb_bronze).where(
                (mlb_bronze.c.kind == "boxscore")
                & (mlb_bronze.c.processed_at.isnot(None))))
    return res.rowcount or 0


def _upsert_rows(table, rows, key_cols, ignore_cols=("fetched_at",)):
    """Batched insert-or-update of an accumulating dim (never deletes a sibling).
    ONE existing-read + one bulk INSERT + per-changed-row UPDATE via
    db_store.upsert_bulk — the set-based replacement for the old per-row reconcile
    loop (which cost a round-trip per row). Returns (n_ins, n_upd, n_del=0)."""
    if not rows or not enabled():
        return (0, 0, 0)
    engine = db_store.get_engine()
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    i, u = db_store.upsert_bulk(
                        conn, table, rows, key_cols, ignore_cols=ignore_cols)
                return (i, u, 0)
            except OperationalError:
                if attempt == 2:
                    raise
    return (0, 0, 0)


def _augment_game_facts(rows, season):
    """Stamp derive_*_rows with the derived season_bucket + a fetched_at."""
    now = _now()
    season = _i(season)
    for r in rows:
        r["season_bucket"] = season
        r["fetched_at"] = now
    return rows


def _valid_game_fact(r):
    """A fact row is writable only with a real MLBAM athlete_id, a game_pk, and a
    team_id (the NOT NULL + FK columns) — guards against a malformed boxscore
    row silently violating a constraint or landing a junk 'None' athlete."""
    aid = r.get("athlete_id")
    return (bool(aid) and aid != "None"
            and r.get("game_pk") is not None and bool(r.get("team_id")))


def _write_game_facts(conn, box, game, touch_fetched_at=False):
    """Persist the StatsAPI-native per-game batter/pitcher stat lines for one
    boxscore into mlb_batter_game / mlb_pitcher_game (game-centric, MLBAM +
    game_pk native). Per-row surgical upsert scoped to (athlete_id, game_pk) — the
    mlb_player idiom, so a call can never delete a sibling and an empty derive
    result is a no-op (never a scope wipe). Runs INSIDE the caller's boxscore
    transaction; the game_pk FK resolves from the earlier schedule txn and the
    team_id FK from the minimal schedule teams. Returns (n_batter, n_pitcher)."""
    gpk = game.get("game_pk")
    season = game.get("season")
    batters = [r for r in _augment_game_facts(derive_batter_rows(box, game), season)
               if _valid_game_fact(r)]
    pitchers = [r for r in _augment_game_facts(derive_pitcher_rows(box, game), season)
                if _valid_game_fact(r)]
    # Batched upsert scoped to this game_pk (all a boxscore's rows share it): one
    # existing-read + one bulk INSERT per table instead of a round-trip per player.
    # Never deletes — matches the P2 per-row behavior, just set-based.
    #
    # ``touch_fetched_at`` — normally fetched_at is an ignore_col (an idempotent
    # re-ingest of an unchanged final box is a no-op, no churn). The P4 post-window
    # "freeze" refresh passes True so fetched_at IS advanced even when the box is
    # unchanged — that advance is what flips the game to HISTORICAL (frozen, 0
    # network) in resolve_actual; without it the freeze would never fire and every
    # >48h grade would re-fetch forever.
    ignore = () if touch_fetched_at else ("fetched_at",)
    for table, rows in ((mlb_batter_game, batters), (mlb_pitcher_game, pitchers)):
        db_store.upsert_bulk(conn, table, rows, ("athlete_id", "game_pk"),
                             scope={"game_pk": gpk}, ignore_cols=ignore)
    return len(batters), len(pitchers)


def _real_franchise_ids():
    """team_ids of the REAL MLB franchises — the /teams-enriched rows (league_id
    populated). Non-franchise schedule entries (all-star squads, postseason bracket-
    seed placeholders) only ever get the minimal parse_schedule row (league_id NULL),
    so this set excludes them. Empty on error → callers fail OPEN (no filtering)."""
    if not enabled():
        return set()
    try:
        with db_store.get_engine().connect() as conn:
            return {r[0] for r in conn.execute(
                select(mlb_team.c.team_id).where(mlb_team.c.league_id.isnot(None)))}
    except (OperationalError, ValueError, TypeError):
        return set()


def ensure_teams(season, force=False):
    """Load/refresh the full 30-team dim for a season (memoized per process)."""
    if not enabled():
        return 0
    if not force and season in _TEAMS_ENSURED:
        return 0
    rows = parse_teams(fetch_teams(season))
    if not rows:
        # An empty /teams payload (transient empty-200 or a stale 7d cache) must
        # NOT memoize the season as "ensured": ingest_standings relies entirely on
        # this call to satisfy fk_mlb_standings_team, so a poisoned memo would skip
        # the reload and keep the FK unsatisfiable. Leave it unmemoized → next call
        # retries the fetch rather than silently trusting a zero-row load.
        return 0
    _upsert_rows(mlb_team, rows, ("team_id",))
    _TEAMS_ENSURED.add(season)
    return len(rows)


# ───────────────────────────────────────────────────────────────── ingestion API
def ingest_date(date, with_boxscores=True, land_bronze=True, skip_game_pks=None):
    """Ingest one YYYY-MM-DD slate → mlb_team/mlb_game (+ mlb_player and the
    StatsAPI-native per-game batter/pitcher stat facts from each genuine-final
    boxscore). Read-only DUAL-RUN: NO ESPN reads, and NOTHING app-facing consumes
    the result until the P4 cutover (the ESPN gamelog_store path stays live).
    Idempotent (surgical upsert).

    ``land_bronze`` — normally the raw schedule/boxscore JSON is upserted into
    mlb_bronze (the audit/reprocess trail). Bronze is transient (purged once its
    silver is written), and for a bulk historical backfill re-landing every large
    boxscore blob is pure write overhead against remote SQL — pass False to skip
    it and write ONLY the silver dims + facts (10x-less write volume per game).
    The maintenance/live paths keep the default (True)."""
    summary = {"date": date, "games": 0, "final": 0, "boxscores": 0,
               "players": 0, "batter_rows": 0, "pitcher_rows": 0,
               "skipped": not enabled()}
    if not enabled():
        return summary
    season = int(str(date)[:4])
    ensure_teams(season)

    raw = fetch_schedule(date)
    games, sched_teams = parse_schedule(raw, season)
    # Prevention: ingest ONLY real-franchise matchups. Drops all-star games (squad
    # "teams") + postseason bracket-seed placeholders (TBD "teams") + their minimal
    # rows, so non-real entries never enter mlb_team/mlb_game/facts. Fail-open — if
    # the franchise roster is unavailable (empty set), ingest everything as before.
    real_ids = _real_franchise_ids()
    if real_ids:
        games = [g for g in games if g.get("home_team_id") in real_ids
                 and g.get("away_team_id") in real_ids]
        sched_teams = [t for t in sched_teams if t.get("team_id") in real_ids]
    summary["games"] = len(games)

    engine = db_store.get_engine()
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    if land_bronze:
                        _land_bronze(conn, "schedule", date, raw)
                    # minimal teams BEFORE games (FK); both batched, never delete.
                    db_store.upsert_bulk(conn, mlb_team, sched_teams, ("team_id",),
                                         ignore_cols=("fetched_at",))
                    db_store.upsert_bulk(conn, mlb_game, games, ("game_pk",),
                                         ignore_cols=("fetched_at",))
                break
            except OperationalError:
                if attempt == 2:
                    raise

    if not with_boxscores:
        return summary

    # Genuine-final games each need one boxscore fetch (the backfill bottleneck).
    # Fetch them CONCURRENTLY (network-bound, free StatsAPI); WRITE serially below
    # under _WRITE_LOCK. The surgical upsert is order-independent and ex.map preserves
    # input order, so this is purely a latency win — byte-identical to the old
    # sequential path. A fetch failure → (g, None) → skipped, matching the prior
    # try/except-continue.
    final_games = [
        g for g in games
        if mlb_starters._is_genuine_final(
            {"status": g["status"], "detailedState": g["detailed_state"]})]
    if skip_game_pks:
        # Resume/incremental: skip games already re-derived (boxscore fetch + write is
        # the whole cost) — the dims upsert above still ran, keeping mlb_game current.
        final_games = [g for g in final_games
                       if g["game_pk"] not in skip_game_pks]
    summary["final"] = len(final_games)

    def _fetch(g):
        try:
            return g, fetch_boxscore(g["game_pk"])
        except Exception:
            return g, None

    seen_players = set()

    def _write(g, box):
        # SERIAL write (main thread, under _WRITE_LOCK) of one fetched boxscore. A
        # failed fetch (box is None) is skipped → re-ingested on the next run.
        if box is None:
            return
        players = parse_boxscore_players(box, g)
        nb = npi = 0
        with _WRITE_LOCK:
            for attempt in range(3):
                try:
                    with engine.begin() as conn:
                        if land_bronze:
                            _land_bronze(conn, "boxscore", str(g["game_pk"]), box)
                        db_store.upsert_bulk(conn, mlb_player, players,
                                             ("player_id",),
                                             ignore_cols=("fetched_at",))
                        nb, npi = _write_game_facts(conn, box, g)
                        if land_bronze:
                            _mark_bronze_processed(conn, "boxscore",
                                                   str(g["game_pk"]))
                    break
                except OperationalError:
                    if attempt == 2:
                        raise
        summary["boxscores"] += 1
        summary["batter_rows"] += nb
        summary["pitcher_rows"] += npi
        seen_players.update(p["player_id"] for p in players)

    # Fetch concurrently but CONSUME in submission order, writing each game as its
    # fetch completes (ex.map yields in order) — so an interrupted backfill still
    # durably commits the prefix it finished, while all fetches overlap (bounded by
    # the pool). Byte-identical results to the old sequential path on a clean run.
    workers = min(_boxscore_fetch_workers(), len(final_games))
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for g, box in ex.map(_fetch, final_games):
                _write(g, box)
    else:
        for g in final_games:
            _write(*_fetch(g))

    summary["players"] = len(seen_players)
    return summary


def ingest_range(start, end, with_boxscores=True, land_bronze=True,
                 skip_game_pks=None):
    """Ingest an inclusive [start, end] date range; yields a per-date summary.
    ``land_bronze=False`` skips the transient raw-JSON audit trail for a fast
    bulk backfill (see ingest_date). ``skip_game_pks`` (a set) skips already-derived
    games — see pitcher_cols_done_game_pks / the --skip-existing resume path."""
    d0 = datetime.date.fromisoformat(str(start))
    d1 = datetime.date.fromisoformat(str(end))
    results = []
    cur = d0
    step = datetime.timedelta(days=1)
    while cur <= d1:
        results.append(ingest_date(cur.isoformat(), with_boxscores=with_boxscores,
                                   land_bronze=land_bronze,
                                   skip_game_pks=skip_game_pks))
        cur += step
    return results


def pitcher_cols_done_game_pks(start, end, col="BF"):
    """Set of game_pks whose mlb_pitcher_game rows already have ``col`` populated, for
    games with official_date in [start, end]. The RESUME set for a #1c re-derive: a
    game's facts are written atomically (one transaction), so any row having ``col``
    means the whole game is done -> skip it. {} on error / SQL off."""
    if not enabled():
        return set()
    try:
        pg, g = mlb_pitcher_game, mlb_game
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(pg.c.game_pk).select_from(
                    pg.join(g, pg.c.game_pk == g.c.game_pk))
                .where((g.c.official_date >= str(start))
                       & (g.c.official_date <= str(end))
                       & (pg.c[col].isnot(None))).distinct()).fetchall()
        return {r[0] for r in rows}
    except (OperationalError, ValueError, TypeError, KeyError):
        return set()


def ingest_maintenance(days_back=2, days_forward=2, straggler_days=14):
    """Keep the warehouse current for grading + game_pk stamping. Ingests a rolling
    [today-days_back .. today+days_forward] window (schedule + boxscores → recent
    finals' facts + upcoming schedule so new predictions can be stamped), PLUS any
    past game still not marked Final within straggler_days (catches a
    suspended/resumed game or an ingestion gap of any recent age). MLB-only,
    idempotent (batched no-op upserts + cached fetches), fail-open per date. Meant
    to be called from the app's already-hourly maintenance. Returns a summary."""
    summary = {"dates": 0, "games": 0, "batter_rows": 0, "pitcher_rows": 0,
               "skipped": not enabled()}
    if not enabled():
        return summary
    today = _today()
    try:
        t = datetime.date.fromisoformat(today)
    except (TypeError, ValueError):
        return summary
    dates = {(t + datetime.timedelta(days=d)).isoformat()
             for d in range(-days_back, days_forward + 1)}
    # Stragglers: past games the warehouse still doesn't have as Final (bounded so
    # a genuinely-terminal postponed/cancelled game isn't re-fetched forever).
    try:
        lo = (t - datetime.timedelta(days=straggler_days)).isoformat()
        engine = db_store.get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                select(mlb_game.c.official_date).where(
                    (mlb_game.c.official_date >= lo)
                    & (mlb_game.c.official_date < today)
                    & (mlb_game.c.status != "Final")).distinct()).fetchall()
        dates.update(r[0] for r in rows if r[0])
    except (OperationalError, ValueError, TypeError):
        pass
    for date in sorted(dates):
        try:
            res = ingest_date(date)
            summary["dates"] += 1
            summary["games"] += res.get("games", 0)
            summary["batter_rows"] += res.get("batter_rows", 0)
            summary["pitcher_rows"] += res.get("pitcher_rows", 0)
        except Exception:
            continue
    summary["handedness_updated"] = populate_handedness()
    return summary


def populate_handedness(season=None):
    """Fill mlb_player.bats / throws from the season roster (/sports/1/players
    batSide.code / pitchHand.code) for players already in mlb_player. The boxscore
    person object carries no handedness, so it can't come from the game-ingest path;
    this is a pure UPDATE of the two columns, which the boxscore player upsert NEVER
    writes (upsert_bulk only touches the keys it's handed), so it survives re-ingest.
    Non-destructive, idempotent (only writes a genuine change), fail-open. Unblocks the
    built-but-dormant platoon feature (which needs a stored batter/pitcher hand).
    Returns the number of mlb_player rows updated."""
    if not enabled():
        return 0
    season = int(season) if season else _current_season()
    try:
        people = (fetch_players(season) or {}).get("people", []) or []
    except Exception:
        return 0
    hand = {}
    for p in people:
        pid = p.get("id")
        if pid is None:
            continue
        b = (p.get("batSide") or {}).get("code")
        t = (p.get("pitchHand") or {}).get("code")
        if b or t:
            hand[str(pid)] = (b, t)
    if not hand:
        return 0
    updated = 0
    try:
        with db_store.get_engine().begin() as conn:
            # mlb_player holds only players who've appeared (small), so read it all and
            # match against the roster rather than a large IN-list.
            rows = conn.execute(select(
                mlb_player.c.player_id, mlb_player.c.bats,
                mlb_player.c.throws)).fetchall()
            for pid, cur_b, cur_t in rows:
                b, t = hand.get(pid, (None, None))
                if (b and b != cur_b) or (t and t != cur_t):
                    conn.execute(
                        update(mlb_player)
                        .where(mlb_player.c.player_id == pid)
                        .values(bats=(b or cur_b), throws=(t or cur_t)))
                    updated += 1
    except (OperationalError, ValueError, TypeError):
        return updated
    return updated


def ingest_standings(season=None, as_of_date=None):
    """Snapshot /standings for a season into mlb_team_standings (as-of today)."""
    season = int(season) if season else _current_season()
    as_of = as_of_date or _today()
    summary = {"season": season, "as_of": as_of, "rows": 0,
               "skipped": not enabled()}
    if not enabled():
        return summary
    ensure_teams(season)
    raw = fetch_standings(season)
    rows = parse_standings(raw, season, as_of)
    engine = db_store.get_engine()
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    _land_bronze(conn, "standings", f"{season}:{as_of}", raw)
                    # Batched: all rows share (season, as_of_date) → scope the read
                    # to this snapshot; upsert-only (never deletes a sibling snapshot).
                    db_store.upsert_bulk(
                        conn, mlb_team_standings, rows,
                        ("team_id", "season", "as_of_date"),
                        scope={"season": season, "as_of_date": as_of},
                        ignore_cols=("fetched_at",))
                break
            except OperationalError:
                if attempt == 2:
                    raise
    summary["rows"] = len(rows)
    return summary


# ─────────────────────────────────────────────────────────────── read helpers
def get_game(game_pk):
    """One mlb_game row as a dict, or None. Fail-open."""
    if not enabled():
        return None
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            r = conn.execute(
                select(mlb_game).where(mlb_game.c.game_pk == int(game_pk))
            ).first()
        return dict(r._mapping) if r else None
    except (OperationalError, ValueError, TypeError):
        return None


def final_game_by_pk(game_pk):
    """Classify one warehouse game for game_pk-based team-market grading (Tier A #2).
    Returns {'state','home_score','away_score','home_team_id','away_team_id',
    'commence_time'} or None. ``state``:
      'final'    — status=='Final', NOT terminal, and BOTH scores present. A legit
                   0-0 grades (the check is ``is None``, not falsy).
      'terminal' — postponed / suspended / cancelled (never a gradable score on this
                   game_pk; the caller falls back so a next-day makeup still grades).
      'live'     — anything else (scheduled / in progress / final-but-score-not-yet).
    Reads get_game (a SNAKE_CASE row) directly — deliberately NOT via
    mlb_starters._is_genuine_final, which reads camelCase status/detailedState and
    would misclassify this snake_case row (e.g. a postponed game would pass as final).
    Fail-open: no row / SQL off / bad id -> None (caller uses the name+date path)."""
    g = get_game(game_pk)
    if not g:
        return None
    import mlb_starters
    detailed = str(g.get("detailed_state") or "").lower()
    terminal = any(bad in detailed for bad in mlb_starters._NON_FINAL_DETAILED)
    hs, as_ = g.get("home_score"), g.get("away_score")
    if terminal:
        state = "terminal"
    elif g.get("status") == "Final" and hs is not None and as_ is not None:
        state = "final"
    else:
        state = "live"
    return {"state": state, "home_score": hs, "away_score": as_,
            "home_team_id": g.get("home_team_id"),
            "away_team_id": g.get("away_team_id"),
            "commence_time": g.get("game_date")}


def find_game_pk(official_date, home_team_id, away_team_id):
    """Retro-match a durable row to its game_pk on (date, home, away). Returns the
    single matching game_pk, or None if 0 or >1 (doubleheader → caller decides)."""
    if not enabled():
        return None
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                select(mlb_game.c.game_pk).where(
                    (mlb_game.c.official_date == str(official_date))
                    & (mlb_game.c.home_team_id == str(home_team_id))
                    & (mlb_game.c.away_team_id == str(away_team_id)))
            ).fetchall()
        return rows[0][0] if len(rows) == 1 else None
    except OperationalError:
        return None


# ───────────────────────────────────────── P3 resolver support (read + alias write)
def team_id_for_name(name):
    """MLBAM team_id for a team display name (odds feed / any source), via the
    mlb_team dim's name_norm. Returns the single match, else None (unknown or
    ambiguous). Fail-open."""
    if not name or not enabled():
        return None
    try:
        nn = db_store.normalize_name(name)
        engine = db_store.get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                select(mlb_team.c.team_id).where(mlb_team.c.name_norm == nn)
            ).fetchall()
        return rows[0][0] if len(rows) == 1 else None
    except (OperationalError, ValueError, TypeError):
        return None


def _parse_ts(v):
    """Parse an ISO-8601 (optionally Z-suffixed) timestamp → aware datetime, or None."""
    if not v:
        return None
    try:
        return datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _score_commence_candidates(cands, commence, tol_hours=20):
    """From {game_pk: game_date} candidates for ONE matchup, the single game_pk whose
    game_date is nearest ``commence`` within tol_hours, or None (no candidate, out of
    tolerance, or a ~tied same-timestamp DH — the 2nd-closest is within 60s). tz-naive
    / mixed-awareness / bad timestamps are skipped, never raised. Shared scorer for
    find_game_pk_by_commence (per-matchup SQL) and the team backfill (in-memory index)
    so both resolve identically."""
    ts = _parse_ts(commence)
    if ts is None or ts.tzinfo is None:      # tz-naive can't be compared → fail-closed
        return None
    scored = []
    for pk, gd in cands.items():
        gts = _parse_ts(gd)
        if gts is None or gts.tzinfo is None:
            continue
        try:
            scored.append((abs((gts - ts).total_seconds()), pk))
        except (TypeError, ValueError):      # mixed awareness / bad ts → skip, never raise
            continue
    if not scored:
        return None
    scored.sort()
    best_delta, best_pk = scored[0]
    if best_delta > tol_hours * 3600:
        return None
    if len(scored) > 1 and (scored[1][0] - best_delta) <= 60:   # ~tied → DH ambiguity
        return None
    return best_pk


def find_game_pk_by_commence(home_team_id, away_team_id, commence, tol_hours=20):
    """Resolve a game_pk for a matchup by the NEAREST game_date to the odds commence
    time — robust to a series (picks the right day) and a SPLIT doubleheader (picks
    the right game by first-pitch time), which a bare official_date match cannot.
    Returns the single closest game_pk within tol_hours, or None if there is no
    match, it is out of tolerance, or it is AMBIGUOUS (a traditional DH whose two
    games share a timestamp → the 2nd-closest is within 60s). Fail-closed."""
    if not (home_team_id and away_team_id) or not enabled():
        return None
    ts = _parse_ts(commence)
    if ts is None or ts.tzinfo is None:      # tz-naive can't be compared → fail-closed
        return None
    day = ts.date()
    cand_dates = {(day + datetime.timedelta(days=d)).isoformat() for d in (-1, 0, 1)}
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                select(mlb_game.c.game_pk, mlb_game.c.game_date).where(
                    (mlb_game.c.home_team_id == str(home_team_id))
                    & (mlb_game.c.away_team_id == str(away_team_id))
                    & (mlb_game.c.official_date.in_(cand_dates)))
            ).fetchall()
    except (OperationalError, ValueError, TypeError):
        return None
    return _score_commence_candidates({pk: gd for pk, gd in rows}, commence, tol_hours)


def record_player_alias(provider, provider_key, mlb_player_id,
                        confidence=1.0, method="sfbb_unique"):
    """Upsert one provider→MLBAM association into player_alias — the P3 audit trail
    of successful resolutions (spec §8). Keyed on (provider, provider_key);
    valid_from records first-seen and is preserved on an unchanged re-resolution
    (it is an ignore_col, so a no-op diff never resets it). Fail-open → False on any
    problem; NEVER raises into the caller's prediction path."""
    if not enabled() or not (provider and provider_key and mlb_player_id):
        return False
    row = {"provider": provider, "provider_key": str(provider_key),
           "mlb_player_id": str(mlb_player_id), "confidence": _f(confidence),
           "resolution_method": method, "valid_from": _today(), "valid_to": None,
           "fetched_at": _now()}
    scope = {"provider": provider, "provider_key": str(provider_key)}
    try:
        engine = db_store.get_engine()
        with _WRITE_LOCK:
            for attempt in range(3):
                try:
                    with engine.begin() as conn:
                        db_store.reconcile(
                            conn, player_alias, [row],
                            ("provider", "provider_key"), scope=scope,
                            ignore_cols=("valid_from", "fetched_at"))
                    return True
                except OperationalError:
                    if attempt == 2:
                        raise
    except Exception:
        return False
    return False


def _game_log(table, stat_cols, athlete_id, season=None, as_of_date=None,
              limit=None, exclude_game_types=None):
    """StatsAPI-native per-game log for one athlete, MOST-RECENT-FIRST, joined to
    the games dim for chronology + opponent/home derivation.

    ``exclude_game_types`` (e.g. spring/all-star/exhibition) is applied BEFORE the
    limit so the recent-N window matches the intended scope; None keeps every
    game_type (the P2 dual-run readers' original behavior).

    This is the reconcile-safe ordering the design mandates (§6): the writer is a
    surgical upsert, so surrogate ``id`` is NOT recency — order by the joined
    ``mlb_game.game_date`` DESC (the PLAY date; ``game_pk`` is assigned at schedule
    time, not play time, so a postponed game keeps a low pk under a late date) with
    ``game_pk`` DESC only as the same-date (doubleheader) tiebreaker. Pass
    ``as_of_date`` for a LEAKAGE-SAFE slice: only games whose ``official_date`` is
    STRICTLY BEFORE it (excludes the game being predicted). Dual-run — nothing
    app-facing consumes this until P4. Fail-open → []."""
    if not enabled():
        return []
    g = mlb_game
    joined = table.join(g, table.c.game_pk == g.c.game_pk)
    cols = [table.c.athlete_id, table.c.game_pk, table.c.team_id,
            table.c.season_bucket, g.c.game_date, g.c.official_date, g.c.season,
            g.c.home_team_id, g.c.away_team_id] + [table.c[c] for c in stat_cols]
    stmt = (select(*cols).select_from(joined)
            .where(table.c.athlete_id == str(athlete_id)))
    if season is not None:
        stmt = stmt.where(table.c.season_bucket == int(season))
    if as_of_date is not None:
        stmt = stmt.where(g.c.official_date < str(as_of_date))
    if exclude_game_types:
        stmt = stmt.where(g.c.game_type.notin_(tuple(exclude_game_types)))
    stmt = stmt.order_by(g.c.game_date.desc(), table.c.game_pk.desc())
    if limit:
        stmt = stmt.limit(int(limit))
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(stmt).fetchall()
    except (OperationalError, ValueError, TypeError):
        return []
    out = []
    for r in rows:
        m = r._mapping
        team_id = m["team_id"]
        is_home = team_id == m["home_team_id"]
        rec = {"athlete_id": m["athlete_id"], "game_pk": m["game_pk"],
               "team_id": team_id, "is_home": is_home,
               "opponent_team_id": m["away_team_id"] if is_home else m["home_team_id"],
               "game_date": m["game_date"], "official_date": m["official_date"],
               "season": m["season"]}
        for c in stat_cols:
            rec[c] = m[c]
        out.append(rec)
    return out


def get_batter_game_log(athlete_id, season=None, as_of_date=None, limit=None,
                        exclude_game_types=None):
    """Most-recent-first StatsAPI-native batter per-game log (P2, dual-run)."""
    return _game_log(mlb_batter_game, _BATTER_GAME_STATS, athlete_id,
                     season=season, as_of_date=as_of_date, limit=limit,
                     exclude_game_types=exclude_game_types)


def get_pitcher_game_log(athlete_id, season=None, as_of_date=None, limit=None,
                         exclude_game_types=None):
    """Most-recent-first StatsAPI-native pitcher per-game log (P2, dual-run)."""
    return _game_log(mlb_pitcher_game, _PITCHER_GAME_STATS, athlete_id,
                     season=season, as_of_date=as_of_date, limit=limit,
                     exclude_game_types=exclude_game_types)


def _ip_to_outs(ip):
    """IP base-3 notation → out count (6.1 IP = 6*3+1 = 19 outs). Mirrors
    espn_client.ip_to_outs; kept local so the StatsAPI module stays ESPN-free."""
    if ip is None:
        return None
    whole = int(ip)
    return whole * 3 + round((ip - whole) * 10)


# Graded/served MLB props whose actual is a stored fact column → (table, column,
# xform). batter_home_runs (HR) is captured as a column but NOT listed here yet (no
# odds market wired), so it's absent → caller falls back to the live per-player
# fetch. ⚠ TB/RBI columns are only populated on rows ingested since the a68f4e6
# capture landed — legacy rows have them NULL; get_player_history SKIPS a NULL-stat
# game (never fabricates a 0) so a pre-backfill row can't corrupt the model input.
_ACTUAL_STAT_SPEC = {
    "batter_hits": (mlb_batter_game, "H", None),
    "batter_strikeouts": (mlb_batter_game, "SO", None),
    "batter_total_bases": (mlb_batter_game, "TB", None),
    "batter_rbis": (mlb_batter_game, "RBI", None),
    "pitcher_strikeouts": (mlb_pitcher_game, "K", None),
    "pitcher_earned_runs": (mlb_pitcher_game, "ER", None),
    "pitcher_outs": (mlb_pitcher_game, "IP", "ip_to_outs"),
}

# StatsAPI gameType codes the ESPN player gamelog does NOT return (spring,
# all-star, exhibition) — excluded from the model-input read so the recent-N
# window matches the ESPN baseline's regular+postseason scope. Also keeps an
# All-Star game out of rows[0], whose team_id is a non-team all-star squad id.
_NON_REGULAR_GAME_TYPES = ("S", "A", "E")


def get_actual_stat(mlb_player_id, game_pk, prop_key):
    """The actual stat value for a graded MLB prop, read straight from the
    game-centric facts (ZERO network) — the P4 grading fast path.

    Returns the value (a resolved 0 comes back as 0.0, NOT None), or None when: the
    prop isn't fact-servable (HR/TB/RBI etc.), the (player, game) has no fact row
    (game not ingested yet, or a DNP), or SQL is off. A fact row exists only for a
    genuine-final boxscore, so a non-None result implies the game is final. The
    caller treats None as "fall back to the live hard-ID/ESPN path." Fail-open."""
    spec = _ACTUAL_STAT_SPEC.get(prop_key)
    if spec is None or not mlb_player_id or game_pk is None or not enabled():
        return None
    table, col, xform = spec
    try:
        gpk = int(game_pk)
    except (TypeError, ValueError):
        return None
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            r = conn.execute(
                select(table.c[col]).where(
                    (table.c.athlete_id == str(mlb_player_id))
                    & (table.c.game_pk == gpk))
            ).first()
    except (OperationalError, ValueError, TypeError):
        return None
    if r is None or r[0] is None:
        return None
    val = float(r[0])
    return _ip_to_outs(val) if xform == "ip_to_outs" else val


def _team_name_map():
    """{team_id (MLBAM): display name} from the team dim — for resolving a fact
    row's own/opponent team_id to the NAME the model's venue + opponent-defense
    lookups key on (the warehouse is MLBAM-keyed; those lookups are name-keyed).
    Tiny (30 rows). Fail-open → {}."""
    if not enabled():
        return {}
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(mlb_team.c.team_id, mlb_team.c.name)).fetchall()
        return {r[0]: r[1] for r in rows}
    except (OperationalError, ValueError, TypeError):
        return {}


# ───────────────────────────────────────────────────── venue dimension (run_env)
_STADIUM_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Major_League_Baseball_Stadiums.csv")


def _stadium_csv_geo(path=_STADIUM_CSV):
    """{park_factors._park_key(team) : (lat, lon)} from the ArcGIS stadium CSV — a
    lat/lon fallback/cross-check for the venue dim. Rows with blank coords (e.g. the
    A's Sutter Health placeholder) are skipped. Fail-open → {} on any read error."""
    import csv
    from park_factors import _park_key
    out = {}
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                team = row.get("TEAM")
                try:
                    lat = float(row.get("LATITUDE"))
                    lon = float(row.get("LONGITUDE"))
                except (TypeError, ValueError):
                    continue
                if team:
                    out[_park_key(team)] = (lat, lon)
    except OSError:
        return {}
    return out


def _venue_dim_row(venue_id, api_rec, team_id, team_name, csv_geo, now):
    """PURE enrichment of one mlb_venue row from: the StatsAPI /venues record
    (name + location.defaultCoordinates + elevation), the venue's modal home team
    (team_id/name, or None for a neutral site), and the CSV lat/lon fallback keyed by
    park_factors._park_key. lat/lon precedence: StatsAPI > CSV(team) > MLB_PARK_GEO(team).
    Park priors + cf_bearing/roof come from the authored team-keyed tables (1.0 / None
    for an unmatched/neutral venue → run_env stays neutral)."""
    import park_factors
    import weather_factors
    from park_factors import _park_key
    api_rec = api_rec or {}
    loc = api_rec.get("location") or {}
    coords = loc.get("defaultCoordinates") or {}
    lat = _f(coords.get("latitude"))
    lon = _f(coords.get("longitude"))
    geo = weather_factors.park_geo(team_name) if team_name else None
    if lat is None and team_name and _park_key(team_name) in (csv_geo or {}):
        lat, lon = csv_geo[_park_key(team_name)]
    if lat is None and geo:
        lat, lon = geo.get("lat"), geo.get("lon")
    return {
        "venue_id": str(venue_id),
        "name": api_rec.get("name"),
        "team_id": str(team_id) if team_id is not None else None,
        "team_name": team_name,
        "lat": lat,
        "lon": lon,
        "cf_bearing": _f((geo or {}).get("cf_bearing")),
        "park_hits": park_factors.park_factor(team_name, "hits") if team_name else 1.0,
        "park_runs": park_factors.park_factor(team_name, "runs") if team_name else 1.0,
        "elevation_ft": _f(loc.get("elevation")),
        "roof": (geo or {}).get("roof"),
        "fetched_at": now,
    }


def _all_venue_ids():
    """Every distinct venue_id in mlb_game (any game type) — the row set for the venue
    dim, so weather/geo cover any park a game was played at. [] on SQL off."""
    if not enabled():
        return []
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(select(mlb_game.c.venue_id).distinct()
                                .where(mlb_game.c.venue_id.isnot(None))).fetchall()
        return sorted({str(r[0]) for r in rows if r[0] is not None})
    except (OperationalError, ValueError, TypeError):
        return {}


def _resident_team_by_venue():
    """{venue_id: team_id} where the venue is that team's PRIMARY REGULAR-season home
    park — i.e. the team's modal regular home venue, per season (so relocations like the
    A's Oakland→Sacramento both count). This is what earns a venue its team's park
    factor; SPRING and NEUTRAL-site venues (a real home team plays there, but it isn't
    their home park) get NO resident team → neutral park priors, not e.g. Coors applied
    to a Scottsdale spring field. {} on SQL off."""
    if not enabled():
        return {}
    from sqlalchemy import func
    g = mlb_game
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(g.c.season, g.c.home_team_id, g.c.venue_id,
                       func.count().label("n"))
                .where((g.c.venue_id.isnot(None))
                       & (g.c.home_team_id.isnot(None))
                       & (g.c.game_type == "R"))
                .group_by(g.c.season, g.c.home_team_id, g.c.venue_id)).fetchall()
    except (OperationalError, ValueError, TypeError):
        return {}
    # Per (season, team) the modal regular home venue = that team's park that season.
    best = {}                       # (season, team_id) -> (venue_id, n)
    for season, tid, vid, n in rows:
        key = (season, str(tid))
        cur = best.get(key)
        if cur is None or (n or 0) > cur[1]:
            best[key] = (str(vid), n or 0)
    return {vid: tid for (season, tid), (vid, _n) in best.items()}


def venue_park_runs_map():
    """{venue_id: park_runs} from mlb_venue — the bulk park run_env lookup. Missing/NULL
    park_runs -> 1.0 (neutral). {} on SQL off/empty (park term then stays neutral)."""
    if not enabled():
        return {}
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(mlb_venue.c.venue_id, mlb_venue.c.park_runs)).fetchall()
        return {str(v): (float(p) if p is not None else 1.0) for v, p in rows}
    except (OperationalError, ValueError, TypeError):
        return {}


def resident_venue_by_team():
    """{team_id: venue_id} — each team's PRIMARY home park (mlb_venue rows that carry a
    resident team). The LIVE venue resolver: a scheduled game's venue_id from its home
    team_id (neutral-site live is a future refinement off the schedule). {} on SQL off."""
    if not enabled():
        return {}
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(mlb_venue.c.team_id, mlb_venue.c.venue_id)
                .where(mlb_venue.c.team_id.isnot(None))).fetchall()
        return {str(t): str(v) for t, v in rows if t is not None and v is not None}
    except (OperationalError, ValueError, TypeError):
        return {}


def game_venue_index(seasons):
    """{(official_date, home_team_id): venue_id} over `seasons` from mlb_game — the
    OFFLINE venue resolver (the ACTUAL venue per game, so neutral-site games key their
    real park, not the home team's usual one). {} on SQL off."""
    if not enabled():
        return {}
    try:
        want = {int(s) for s in seasons}
    except (TypeError, ValueError):
        return {}
    try:
        g = mlb_game
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(g.c.official_date, g.c.home_team_id, g.c.venue_id)
                .where(g.c.season.in_(sorted(want))
                       & g.c.venue_id.isnot(None))).fetchall()
        return {(str(d), str(t)): str(v) for d, t, v in rows
                if d is not None and t is not None and v is not None}
    except (OperationalError, ValueError, TypeError):
        return {}


def build_venue_dim(verbose=True):
    """Populate mlb_venue: one row per distinct venue_id in mlb_game, enriched with the
    StatsAPI /venues location (lat/lon/elevation) + the authored park/geo priors
    (matched via the venue's modal home team). Idempotent upsert (never deletes).
    OPERATOR-run (network + SQL). Returns rows written, or 0 when SQL is off/empty."""
    if not enabled():
        raise RuntimeError("SQL is not configured (SQL_* secrets) — cannot build.")
    venue_ids = _all_venue_ids()
    if not venue_ids:
        if verbose:
            print("  [venue] no venue_ids in mlb_game; nothing to build.")
        return 0
    resident = _resident_team_by_venue()   # venue_id -> team_id (PRIMARY parks only)
    name_of = _team_name_map()
    csv_geo = _stadium_csv_geo()
    # StatsAPI /venues location (one batched call; hydrate gives lat/lon + elevation).
    api = {}
    try:
        raw = mlb_starters._get("venues", params={
            "venueIds": ",".join(venue_ids), "hydrate": "location"})
        for v in (raw or {}).get("venues", []) or []:
            if v.get("id") is not None:
                api[str(v["id"])] = v
    except Exception as exc:                       # noqa: BLE001 — fail-open on network
        if verbose:
            print(f"  [venue] StatsAPI /venues fetch failed ({exc}); "
                  "using priors/CSV geo only.")
    now = _now()
    rows = []
    for vid in venue_ids:
        tid = resident.get(vid)            # None for spring/neutral -> neutral priors
        rows.append(_venue_dim_row(vid, api.get(vid), tid,
                                   name_of.get(tid) if tid else None, csv_geo, now))
    with _WRITE_LOCK:
        with db_store.get_engine().begin() as conn:
            db_store.upsert_bulk(conn, mlb_venue, rows, ("venue_id",),
                                 ignore_cols=("fetched_at",))
    if verbose:
        n_geo = sum(1 for r in rows if r["lat"] is not None)
        n_park = sum(1 for r in rows if (r["park_runs"] or 1.0) != 1.0)
        print(f"  [venue] {len(rows)} venues upserted "
              f"({n_geo} with lat/lon, {n_park} with a non-neutral park factor).")
    return len(rows)


def _weather_targets(seasons):
    """[{venue_id, date, lat, lon, first_pitch_epoch, first_pitch_iso}] — one per
    (venue, official_date) REGULAR-season game in `seasons`, carrying the venue's
    lat/lon (from mlb_venue) + the EARLIEST first-pitch UTC epoch that date (the game
    hour to sample; doubleheaders share the venue-date weather row). Skips venues with
    no lat/lon. [] on SQL off."""
    if not enabled():
        return []
    try:
        want = {int(s) for s in seasons}
    except (TypeError, ValueError):
        return []
    import weather_factors as wf
    g = mlb_game
    try:
        with db_store.get_engine().connect() as conn:
            grows = conn.execute(
                select(g.c.venue_id, g.c.official_date, g.c.game_date)
                .where(g.c.season.in_(sorted(want)) & (g.c.game_type == "R")
                       & g.c.venue_id.isnot(None)
                       & g.c.official_date.isnot(None))).fetchall()
            vrows = conn.execute(
                select(mlb_venue.c.venue_id, mlb_venue.c.lat,
                       mlb_venue.c.lon)).fetchall()
    except (OperationalError, ValueError, TypeError):
        return []
    latlon = {str(v): (la, lo) for v, la, lo in vrows
              if la is not None and lo is not None}
    best = {}                       # (venue, date) -> earliest first-pitch ISO
    for vid, od, gd in grows:
        vid = str(vid)
        if vid not in latlon:
            continue
        key = (vid, str(od)[:10])
        gd = str(gd) if gd else None
        if key not in best or (gd and (best[key] is None or gd < best[key])):
            best[key] = gd
    out = []
    for (vid, od), gd in best.items():
        la, lo = latlon[vid]
        epoch = None
        if gd:
            try:
                epoch = wf._parse_iso_utc(gd).timestamp()
            except (ValueError, TypeError, OSError):
                epoch = None
        out.append({"venue_id": vid, "date": od, "lat": la, "lon": lo,
                    "first_pitch_epoch": epoch, "first_pitch_iso": gd})
    return out


def build_weather(seasons, apply=False, force=False, verbose=True):
    """Backfill weather_game (first-pitch-hour) for regular-season games in `seasons`
    from Visual Crossing (Batch A run_env). DRY-RUN by default: reports the fetch
    volume, NO network / write. apply=True fetches per (venue, season) date-range (one
    batched call each) and upserts. Incremental: skips (venue, date) already present
    unless force. Needs WEATHER_API_KEY + populated mlb_venue lat/lon. Operator-run.
    Returns weather_game rows written (0 on dry-run)."""
    if not enabled():
        raise RuntimeError("SQL is not configured (SQL_* secrets) — cannot build.")
    import weather_factors as wf
    targets = _weather_targets(seasons)
    if not targets:
        if verbose:
            print("  [weather] no targets (need mlb_venue lat/lon + regular games "
                  "for the seasons). Run --build-venues first?")
        return 0
    have = set()
    if not force:
        try:
            with db_store.get_engine().connect() as conn:
                for v, d in conn.execute(select(weather_game.c.venue_id,
                                                weather_game.c.weather_date)):
                    have.add((str(v), str(d)))
        except (OperationalError, ValueError, TypeError):
            pass
    todo = [t for t in targets if (t["venue_id"], t["date"]) not in have]
    groups = {}                     # (venue, YYYY) -> [targets] = one batched fetch
    for t in todo:
        groups.setdefault((t["venue_id"], t["date"][:4]), []).append(t)
    if not apply:
        if verbose:
            print(f"  [weather] DRY-RUN: {len(todo)} venue-dates to fetch across "
                  f"{len(groups)} (venue,season) calls "
                  f"({len(targets) - len(todo)} already present, {len(targets)} total). "
                  f"Re-run with --apply to fetch + write.")
        return 0
    if not wf.promote_weather_key_from_toml():
        print("  [weather] WEATHER_API_KEY not set (secrets.toml / env) — aborting.")
        return 0
    now = _now()
    written = 0
    for (vid, yr), ts in sorted(groups.items()):
        dates = sorted(t["date"] for t in ts)
        lat, lon = ts[0]["lat"], ts[0]["lon"]
        day_hours = wf.fetch_visualcrossing_range(lat, lon, dates[0], dates[-1])
        rows = []
        for t in ts:
            pick = wf.pick_hour_by_epoch(day_hours.get(t["date"]),
                                         t["first_pitch_epoch"])
            if not pick:
                continue
            rows.append({
                "venue_id": vid, "weather_date": t["date"],
                "temp_f": pick.get("temp_f"), "humidity": pick.get("humidity"),
                "pressure_mb": pick.get("pressure_mb"),
                "wind_mph": pick.get("wind_mph"),
                "wind_dir_deg": pick.get("wind_dir_deg"),
                "first_pitch_utc": t.get("first_pitch_iso"),
                "source": "visualcrossing", "fetched_at": now})
        if rows:
            with _WRITE_LOCK:
                with db_store.get_engine().begin() as conn:
                    db_store.upsert_bulk(conn, weather_game, rows,
                                         ("venue_id", "weather_date"),
                                         scope={"venue_id": vid},
                                         ignore_cols=("fetched_at",))
            written += len(rows)
        if verbose:
            print(f"    [weather] {vid} {yr}: {len(rows)}/{len(ts)} dates written.")
    if verbose:
        print(f"  [weather] {written} weather_game rows upserted "
              f"({len(groups)} venue-season fetches).")
    return written


def get_player_history(mlb_player_id, prop_key, n=20, as_of_date=None,
                       season=None, player_name=None):
    """Reproduce the ESPN ``get_player_stat_history`` dict for one MLB player+prop
    straight from the StatsAPI facts — the P4 model-INPUT read (the projection-side
    analog of ``get_actual_stat``'s grading read).

    Returns the SAME contract dict ``get_player_stat_history`` returns (most-recent-
    first, index-aligned parallel lists) so a warehouse-first branch can drop in
    without touching props/backtest: ``player``, ``athlete_id`` (MLBAM),
    ``stat_label``, ``values``, ``opponents`` (opponent DISPLAY NAME),
    ``home_aways``, ``minutes`` (0.0 — no MLB analog), ``game_dates``,
    ``plate_appearances`` (AB+BB+HBP+SF+SH, matching the ESPN reader's own fallback
    formula), ``at_bats``, ``team_id`` (own team, MLBAM), and ``found``. One
    ADDITIVE key: ``team_name`` (own team resolved) — the eventual flip maps
    venue/opponent-defense on NAME because the warehouse is MLBAM-keyed, not
    ESPN-id-keyed, and props' reverse-map is ESPN-id-keyed (extra keys are harmless;
    consumers read via .get()).

    Returns None (NOT an empty dict) when the warehouse can't serve — an
    unsupported prop (HR/TB/RBI have no fact column), SQL off, no id, or no rows —
    so the caller falls open to the live ESPN path, mirroring ``get_actual_stat``'s
    None-means-fallback convention. ``pitcher_outs`` values are IP→outs. A fact row
    exists only for a genuine-final boxscore where the batter came to the plate, so
    a real 0 is present (0.0) and a DNP is simply ABSENT — never a synthesized
    0-row. Leakage-safe via ``as_of_date`` (strict official_date <). Fail-open."""
    spec = _ACTUAL_STAT_SPEC.get(prop_key)
    if spec is None or not mlb_player_id or not enabled():
        return None
    table, col, xform = spec
    is_pitcher = table is mlb_pitcher_game
    stat_cols = _PITCHER_GAME_STATS if is_pitcher else _BATTER_GAME_STATS
    # Exclude spring/all-star/exhibition (BEFORE the limit) so the recent-N window
    # matches the ESPN baseline's regular+postseason scope — see the constant.
    rows = _game_log(table, stat_cols, mlb_player_id, season=season,
                     as_of_date=as_of_date, limit=n,
                     exclude_game_types=_NON_REGULAR_GAME_TYPES)
    if not rows:
        return None
    names = _team_name_map()
    values, opponents, home_aways, game_dates = [], [], [], []
    plate_appearances, at_bats = [], []
    own_team_id = None
    for r in rows:
        v = r.get(col)
        if xform == "ip_to_outs":
            v = _ip_to_outs(v)
        if v is None:
            # The stat wasn't captured for this game (e.g. TB/RBI on a row written
            # before those columns were populated / pre-backfill). EXCLUDE it — a
            # genuine 0 is stored as 0 (StatsAPI always reports the stat for a batter
            # who came to the plate), so None means "no data", not "zero". Fabricating
            # a 0.0 here would silently corrupt the projection with all-zero history.
            continue
        if own_team_id is None:
            own_team_id = r["team_id"]           # newest KEPT game's team (~ current)
        values.append(float(v))
        opponents.append(names.get(r["opponent_team_id"]))
        home_aways.append(bool(r["is_home"]))
        game_dates.append(r["game_date"])
        if is_pitcher:
            plate_appearances.append(None)
            at_bats.append(None)
        else:
            ab = r.get("AB")
            if ab is None:                       # match ESPN: PA None when AB None
                plate_appearances.append(None)
            else:
                plate_appearances.append(
                    (ab or 0.0) + (r.get("BB") or 0.0) + (r.get("HBP") or 0.0)
                    + (r.get("SF") or 0.0) + (r.get("SH") or 0.0))
            at_bats.append(ab)
    if not values:              # every candidate game lacked the stat (pre-backfill)
        return None
    return {
        "player": player_name,
        "athlete_id": str(mlb_player_id),
        "stat_label": col,
        "values": values,
        "opponents": opponents,
        "home_aways": home_aways,
        "minutes": [0.0] * len(values),
        "game_dates": game_dates,
        "plate_appearances": plate_appearances,
        "at_bats": at_bats,
        "team_id": own_team_id,
        "team_name": names.get(own_team_id),
        "found": True,
        "source": "warehouse",     # stamped onto prediction_log.source (see espn_client.get_player_stat_history)
    }


def get_calib_gamelog(mlb_player_id, role, season=None):
    """Per-game log in the ESPN cached_gamelog SHAPE (a list of per-game dicts,
    most-recent-first) — the reshape the backtest/calibration real-line readers
    consume (they walk + index-slice a raw per-game log), sourced from the warehouse
    facts. role ∈ {'pitcher','batter'}.

    Each dict keeps the role's stat-label keys (batter AB/H/SO/BB/HBP/SF/SH/HR/TB/RBI;
    pitcher IP/K/ER — same names the readers key on) + is_home + game_date (ISO
    string) + team_id (MLBAM), and ADDS ``opponent`` (DISPLAY NAME via _team_name_map
    — load-bearing for the opp_defense weighting + park reconstruction) and
    ``completed=True`` (a warehouse row exists only for a genuine-final). Current-
    season by default AND spring/all-star/exhibition excluded (_NON_REGULAR_GAME_
    TYPES), matching the ESPN reader's scope — an all-star game also carries a non-
    team squad id absent from mlb_team, so excluding it avoids a None opponent /
    garbage is_home in the fit. Fail-open → [] (the offline backtest then skips that
    player; there is no ESPN fallback for MLB post-P6).

    The opponent NAME is the canonical mlb_team.name, which the real-line fit's
    park/opp-defense features key on; that resolution is pinned by golden tests in
    test_mlb_warehouse.py (CalibParkNameGoldenTests + the reader shape tests)."""
    if not enabled() or not mlb_player_id:
        return []
    season = season if season is not None else _current_season()
    rows = (get_pitcher_game_log(mlb_player_id, season=season,
                                 exclude_game_types=_NON_REGULAR_GAME_TYPES)
            if role == "pitcher"
            else get_batter_game_log(mlb_player_id, season=season,
                                     exclude_game_types=_NON_REGULAR_GAME_TYPES))
    if not rows:
        return []
    names = _team_name_map()
    out = []
    for r in rows:
        g = dict(r)
        g["opponent"] = names.get(r.get("opponent_team_id"))
        g["completed"] = True
        out.append(g)
    return out


# ── P4 team-market model inputs: recent form / standings / team defense ───────
# StatsAPI-native team readers for the team markets. recent_games come from
# mlb_game (per-game scores — no better source); win%/record + team defense come
# from the /standings snapshot (mlb_team_standings), which carries the cumulative
# runsScored/runsAllowed DIRECTLY — cleaner + more authoritative than ESPN's
# scan-and-average, and it unlocks run differential / Pythagorean strength for a
# future team-market signal. Dual-run: nothing app-facing consumes these until the
# team-market flip. Fail-open; leakage-safe via as_of_date.
# ── As-of starter line from the warehouse facts (backtest, ZERO network) ──────
# Replaces the per-(pitcher, game-date) StatsAPI byDateRange storm: one query per
# season builds an in-memory index, then as-of cumulative stats are computed
# locally — O(1 query)/season regardless of how large the warehouse grows.
_PITCHER_IDX = {}                # season -> {athlete_id: [(official_date, outs, er, k)]}
_PITCHER_IDX_LOCK = threading.Lock()
_THROWS_CACHE = {}
_THROWS_LOADED = [False]


def _pitcher_game_index(season):
    """{athlete_id: [(official_date, outs, er, k, bb, bf), ...] sorted} for a season,
    from mlb_pitcher_game joined to mlb_game for the play date. Loaded ONCE per season
    (thread-safe); an empty/failed load is not cached so it can retry. BB/BF (Tier A
    #1c) are None on pre-re-backfill rows -> coerced to 0.0 here."""
    season = int(season)
    idx = _PITCHER_IDX.get(season)
    if idx is not None:
        return idx
    with _PITCHER_IDX_LOCK:
        idx = _PITCHER_IDX.get(season)
        if idx is not None:
            return idx
        pg, g = mlb_pitcher_game, mlb_game
        joined = pg.join(g, pg.c.game_pk == g.c.game_pk)
        try:
            with db_store.get_engine().connect() as conn:
                rows = conn.execute(
                    select(pg.c.athlete_id, g.c.official_date,
                           pg.c.IP, pg.c.ER, pg.c.K, pg.c.BB, pg.c.BF)
                    .select_from(joined)
                    .where(pg.c.season_bucket == season)).all()
        except (OperationalError, ValueError, TypeError):
            return {}                          # transient — don't cache the miss
        built = {}
        for aid, od, ip, er, k, bb, bf in rows:
            if not od:
                continue
            built.setdefault(str(aid), []).append(
                (str(od)[:10], _ip_to_outs(ip) or 0, float(er or 0),
                 float(k or 0), float(bb or 0), float(bf or 0)))
        for lst in built.values():
            lst.sort(key=lambda x: x[0])
        _PITCHER_IDX[season] = built
        return built


def asof_pitcher_stats(athlete_id, as_of_date):
    """Cumulative pitching line for a starter through the day BEFORE as_of_date
    (leakage-safe), from the warehouse facts — ZERO network. Returns
    {'era','ip','k','games','avg_ip','bb','bf'} or None (no prior games in-season /
    SQL off). IP is summed in OUTS (facts store StatsAPI 6.1 = 6 IP + 1 out). bb/bf
    are cumulative-as-of (0 when unpopulated -> K%/BB% consumers threshold on bf)."""
    if not enabled() or not athlete_id or not as_of_date:
        return None
    cutoff = str(as_of_date)[:10]              # games STRICTLY before the game day
    games = _pitcher_game_index(int(cutoff[:4])).get(str(athlete_id))
    if not games:
        return None
    outs = er = k = bb = bf = 0.0
    n = 0
    for od, o, e, kk, w, f in games:
        if od < cutoff:
            outs += o
            er += e
            k += kk
            bb += w
            bf += f
            n += 1
    if n == 0 or outs <= 0:
        return None
    ip = outs / 3.0
    return {"era": (er / ip) * 9.0, "ip": ip, "k": k, "games": n,
            "avg_ip": ip / n, "bb": bb, "bf": bf}


def pitcher_throws(athlete_id):
    """Handedness ('L'/'R') for a pitcher from mlb_player, batch-loaded once."""
    if not enabled() or not athlete_id:
        return None
    if not _THROWS_LOADED[0]:
        with _PITCHER_IDX_LOCK:
            if not _THROWS_LOADED[0]:
                try:
                    with db_store.get_engine().connect() as conn:
                        for pid, thr in conn.execute(
                                select(mlb_player.c.player_id,
                                       mlb_player.c.throws)).all():
                            _THROWS_CACHE[str(pid)] = thr
                except (OperationalError, ValueError, TypeError):
                    pass
                _THROWS_LOADED[0] = True
    return _THROWS_CACHE.get(str(athlete_id))


# ── As-of LINEUP offense from the warehouse facts (backtest, ZERO network) ────
# The offense analog of the pitcher index: one query/season → in-memory batter
# index → as-of cumulative OBP/SLG/OPS computed locally. Feeds a lineup-offense
# edge for team predictions (today's actual 9, not a team-season blob).
_BATTER_IDX = {}                 # season -> {athlete_id: [(official_date, ab,h,bb,hbp,sf,tb)]}
_BATTER_IDX_LOCK = threading.Lock()


def _batter_game_index(season):
    """{athlete_id: [(official_date, ab, h, bb, hbp, sf, tb), ...] sorted} for a
    season. Loaded ONCE per season (thread-safe); empty/failed loads not cached."""
    season = int(season)
    idx = _BATTER_IDX.get(season)
    if idx is not None:
        return idx
    with _BATTER_IDX_LOCK:
        idx = _BATTER_IDX.get(season)
        if idx is not None:
            return idx
        b, g = mlb_batter_game, mlb_game
        joined = b.join(g, b.c.game_pk == g.c.game_pk)
        try:
            with db_store.get_engine().connect() as conn:
                rows = conn.execute(
                    select(b.c.athlete_id, g.c.official_date, b.c.AB, b.c.H,
                           b.c.BB, b.c.HBP, b.c.SF, b.c.TB)
                    .select_from(joined)
                    .where(b.c.season_bucket == season)).all()
        except (OperationalError, ValueError, TypeError):
            return {}
        built = {}
        for aid, od, ab, h, bb, hbp, sf, tb in rows:
            if not od:
                continue
            built.setdefault(str(aid), []).append(
                (str(od)[:10], float(ab or 0), float(h or 0), float(bb or 0),
                 float(hbp or 0), float(sf or 0), float(tb or 0)))
        for lst in built.values():
            lst.sort(key=lambda x: x[0])
        _BATTER_IDX[season] = built
        return built


def asof_batter_ops(athlete_id, as_of_date):
    """Cumulative OBP/SLG/OPS for a batter through the day BEFORE as_of_date
    (leakage-safe), from the warehouse facts — ZERO network. Returns
    {'obp','slg','ops','pa'} or None (no prior PAs in-season / SQL off)."""
    if not enabled() or not athlete_id or not as_of_date:
        return None
    cutoff = str(as_of_date)[:10]
    games = _batter_game_index(int(cutoff[:4])).get(str(athlete_id))
    if not games:
        return None
    ab = h = bb = hbp = sf = tb = 0.0
    for od, a, hh, w, hp, f, t in games:
        if od < cutoff:
            ab += a
            h += hh
            bb += w
            hbp += hp
            sf += f
            tb += t
    denom = ab + bb + hbp + sf
    if denom <= 0 or ab <= 0:
        return None
    obp = (h + bb + hbp) / denom
    slg = tb / ab
    return {"obp": obp, "slg": slg, "ops": obp + slg, "pa": denom}


_GAME_PK_IDX = {}       # season -> {(official_date, home_id, away_id): game_pk}
_GAME_LINEUP_IDX = {}   # season -> {game_pk: {team_id: [top-9 aids by PA]}}


def _game_pk_index(season):
    """{(official_date, home_id, away_id): game_pk} for a season — one query, so
    the backtest resolves a game_pk without a per-game find_game_pk round trip.
    Doubleheaders (same key, 2 game_pks) are dropped (ambiguous)."""
    season = int(season)
    idx = _GAME_PK_IDX.get(season)
    if idx is not None:
        return idx
    with _BATTER_IDX_LOCK:
        idx = _GAME_PK_IDX.get(season)
        if idx is not None:
            return idx
        g = mlb_game
        try:
            with db_store.get_engine().connect() as conn:
                rows = conn.execute(
                    select(g.c.official_date, g.c.home_team_id,
                           g.c.away_team_id, g.c.game_pk)
                    .where(g.c.season == season)).all()
        except (OperationalError, ValueError, TypeError):
            return {}
        built, dupes = {}, set()
        for od, h, a, pk in rows:
            k = (str(od)[:10], str(h), str(a))
            if k in built:
                dupes.add(k)
            built[k] = pk
        for k in dupes:
            built.pop(k, None)                 # doubleheader → ambiguous, skip
        _GAME_PK_IDX[season] = built
        return built


def _game_lineup_index(season):
    """{game_pk: {team_id: [top-9 athlete_ids by PA]}} for a season — one query.
    De-facto starters = the batters with the most plate appearances in the box
    score (no batting order is stored; the backtest's lineup stand-in)."""
    season = int(season)
    idx = _GAME_LINEUP_IDX.get(season)
    if idx is not None:
        return idx
    with _BATTER_IDX_LOCK:
        idx = _GAME_LINEUP_IDX.get(season)
        if idx is not None:
            return idx
        b = mlb_batter_game
        try:
            with db_store.get_engine().connect() as conn:
                rows = conn.execute(
                    select(b.c.game_pk, b.c.team_id, b.c.athlete_id, b.c.AB,
                           b.c.BB, b.c.HBP, b.c.SF, b.c.SH)
                    .where(b.c.season_bucket == season)).all()
        except (OperationalError, ValueError, TypeError):
            return {}
        acc = {}
        for pk, tid, aid, ab, bb, hbp, sf, sh in rows:
            pa = (float(ab or 0) + float(bb or 0) + float(hbp or 0)
                  + float(sf or 0) + float(sh or 0))
            acc.setdefault(pk, {}).setdefault(str(tid), []).append((pa, str(aid)))
        built = {pk: {tid: [a for _p, a in sorted(lst, reverse=True)[:9]]
                      for tid, lst in teams.items()}
                 for pk, teams in acc.items()}
        _GAME_LINEUP_IDX[season] = built
        return built


def _team_final_games(team_id, as_of_date=None, season=None, limit=None):
    """Most-recent-first FINAL regular/postseason games for a team (home or away),
    from mlb_game joined to mlb_team for the home/away display names. Each dict
    mirrors espn_client.get_team_schedule ({date, home_team, away_team, home_score,
    away_score, total_score}) so compute_recent_form / compute_team_defense /
    annotate_opponent_strength work UNCHANGED on it. Ordered game_date DESC,
    game_pk DESC; leakage-safe via as_of_date (strict official_date <)."""
    if not enabled() or not team_id:
        return []
    g = mlb_game
    home = mlb_team.alias("home_t")
    away = mlb_team.alias("away_t")
    joined = (g.join(home, g.c.home_team_id == home.c.team_id, isouter=True)
              .join(away, g.c.away_team_id == away.c.team_id, isouter=True))
    # Genuine-final only: mlb_game (unlike the fact tables) holds EVERY scheduled
    # game, and a SUSPENDED game reports abstractGameState 'Final' with a PARTIAL
    # score, so gate on detailed_state too — same denylist as
    # mlb_starters._is_genuine_final (NULL trusts the abstract state).
    det = g.c.detailed_state
    genuine_final = or_(det.is_(None), and_(
        *[not_(det.ilike(f"%{b}%")) for b in mlb_starters._NON_FINAL_DETAILED]))
    stmt = (select(g.c.game_date, g.c.game_pk, g.c.home_score, g.c.away_score,
                   home.c.name.label("home_name"), away.c.name.label("away_name"))
            .select_from(joined)
            .where((g.c.home_team_id == str(team_id))
                   | (g.c.away_team_id == str(team_id)))
            .where(g.c.status == "Final")
            .where(genuine_final)
            .where(g.c.home_score.isnot(None))
            .where(g.c.away_score.isnot(None))
            .where(g.c.game_type.notin_(_NON_REGULAR_GAME_TYPES)))
    if season is not None:
        stmt = stmt.where(g.c.season == int(season))
    if as_of_date is not None:
        stmt = stmt.where(g.c.official_date < str(as_of_date))
    stmt = stmt.order_by(g.c.game_date.desc(), g.c.game_pk.desc())
    if limit:
        stmt = stmt.limit(int(limit))
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(stmt).fetchall()
    except (OperationalError, ValueError, TypeError):
        return []
    out = []
    for r in rows:
        m = r._mapping
        try:
            hs, as_ = int(m["home_score"]), int(m["away_score"])
        except (TypeError, ValueError):
            continue
        out.append({"date": m["game_date"], "home_team": m["home_name"],
                    "away_team": m["away_name"], "home_score": hs,
                    "away_score": as_, "total_score": hs + as_,
                    "game_pk": m["game_pk"]})   # #2b: DH-safe backtest dedup/join key
    return out


def get_team_games(team_name, as_of_date=None, season=None, limit=None):
    """espn_client.get_team_schedule analog from the warehouse (final reg/post
    games, most-recent-first, leakage-safe). Fail-open → [].

    ⚠ home_team/away_team in each dict are the CANONICAL mlb_team.name. The
    consumers compute_recent_form / compute_team_defense match the team by EXACT
    string, so the flip must pass THAT canonical name (see team_name_canonical) —
    NOT the raw odds/ESPN spelling — or the exact-match silently yields all-zero
    form. team_name here is resolved tolerantly (team_id_for_name), so the input
    spelling may differ; the OUTPUT names are canonical."""
    return _team_final_games(team_id_for_name(team_name),
                             as_of_date=as_of_date, season=season, limit=limit)


def team_name_canonical(team_name):
    """The canonical mlb_team.name for a (tolerantly-resolved) team name, or None.
    The team-market flip resolves the odds/ESPN name ONCE via this and passes the
    result to BOTH get_team_games and compute_recent_form / compute_team_defense so
    their exact-string team match holds despite StatsAPI/ESPN/odds spelling gaps
    (e.g. 'St Louis Cardinals' vs 'St. Louis Cardinals', Athletics)."""
    tid = team_id_for_name(team_name)
    return _team_name_map().get(tid) if tid else None


def team_id_for_name_tolerant(name):
    """team_id_for_name with a fuzzy fallback ladder — exact name_norm (the fast
    path), then substring, then last-word/nickname — mirroring espn_client.find_team's
    tolerance so the warehouse team-dim resolver matches odds/ESPN spellings the exact
    name_norm misses (e.g. 'Oakland Athletics' vs the canonical 'Athletics'). Returns
    the team_id ONLY on an UNAMBIGUOUS hit (a single distinct team); an ambiguous
    partial like 'Sox' (Red Sox + White Sox) or 'Los Angeles' (two LA clubs) → None,
    so a caller never binds the wrong franchise. Fail-open (None on any error)."""
    if not name or not enabled():
        return None
    tid = team_id_for_name(name)                      # exact name_norm (fast path)
    if tid:
        return tid
    try:
        nn = db_store.normalize_name(name)
    except (TypeError, ValueError):
        return None
    if not nn:
        return None
    nn_last = nn.split()[-1]
    substr, lastword = set(), set()
    for cand_id, disp in _team_name_map().items():
        try:
            dn = db_store.normalize_name(disp or "")
        except (TypeError, ValueError):
            continue
        if not dn:
            continue
        if nn in dn or dn in nn:
            substr.add(cand_id)
        if nn_last and dn.split() and nn_last == dn.split()[-1]:
            lastword.add(cand_id)
    for hits in (substr, lastword):                  # substring is the tighter tier
        if len(hits) == 1:
            return next(iter(hits))
    return None


def warehouse_find_team(team_name, season=None, as_of_date=None):
    """StatsAPI-warehouse replacement for espn_client.find_team: return the same
    ``{id, display_name, abbreviation, short_name, record, wins, losses, win_pct}``
    dict its consumers read — the ``if home_espn and away_espn`` team-market gate,
    the event_team_ids for player disambiguation, and the season block — sourced from
    the mlb_team dim + the latest /standings snapshot, so MLB team markets resolve
    WITHOUT the ESPN teams dict. The tolerance is applied ONCE here; the canonical
    display name then feeds get_team_standings' exact match. None when the name can't
    be resolved unambiguously (or SQL is off). Fail-open — callers keep find_team as a
    transition fallback until the P6 teardown removes ESPN team resolution for MLB."""
    if not enabled():
        return None
    tid = team_id_for_name_tolerant(team_name)
    if not tid:
        return None
    disp = _team_name_map().get(tid) or team_name
    st = get_team_standings(disp, season=season, as_of_date=as_of_date) or {}
    return {
        "id": str(tid),
        "display_name": disp,
        "abbreviation": "",          # not stored/used by the team-market consumers
        "short_name": "",            # (present for shape parity with find_team)
        "record": st.get("record", ""),
        "wins": st.get("wins", 0),
        "losses": st.get("losses", 0),
        "win_pct": st.get("win_pct", 0.0),
    }


def _latest_standings_asof(conn, season, as_of_date=None):
    """The most-recent standings snapshot date for a season (<= as_of_date if
    given), or None. All 30 teams share an as_of_date per ingest, so this pins one
    coherent snapshot to read."""
    s = mlb_team_standings
    stmt = select(s.c.as_of_date).where(s.c.season == season)
    if as_of_date is not None:
        stmt = stmt.where(s.c.as_of_date <= str(as_of_date))
    r = conn.execute(stmt.order_by(s.c.as_of_date.desc()).limit(1)).first()
    return r[0] if r is not None else None


def get_team_standings(team_name, season=None, as_of_date=None):
    """The season block {record, wins, losses, win_pct, runs_scored, runs_allowed}
    from the LATEST mlb_team_standings snapshot for the team (as_of_date <= the
    cutoff, else the overall latest) — mirrors the ESPN get_all_teams season fields
    (runs_* are additive, for a future run-differential/Pythagorean signal). None if
    unknown. Fail-open."""
    if not enabled():
        return None
    tid = team_id_for_name(team_name)
    if not tid:
        return None
    season = int(season) if season else _current_season()
    s = mlb_team_standings
    stmt = (select(s.c.wins, s.c.losses, s.c.win_pct,
                   s.c.runs_scored, s.c.runs_allowed)
            .where(s.c.team_id == str(tid)).where(s.c.season == season))
    if as_of_date is not None:
        stmt = stmt.where(s.c.as_of_date <= str(as_of_date))
    stmt = stmt.order_by(s.c.as_of_date.desc()).limit(1)
    try:
        with db_store.get_engine().connect() as conn:
            r = conn.execute(stmt).first()
    except (OperationalError, ValueError, TypeError):
        return None
    if r is None:
        return None
    w = int(r[0]) if r[0] is not None else 0
    losses = int(r[1]) if r[1] is not None else 0
    wp = (float(r[2]) if r[2] is not None
          else (w / (w + losses) if (w + losses) else 0.0))
    return {"record": f"{w}-{losses}", "wins": w, "losses": losses, "win_pct": wp,
            "runs_scored": (int(r[3]) if r[3] is not None else None),
            "runs_allowed": (int(r[4]) if r[4] is not None else None)}


def get_team_defense(season=None, as_of_date=None):
    """{team display name: avg runs ALLOWED per game} — the StatsAPI-native team-
    defense input, from the latest /standings snapshot's cumulative runs_allowed /
    games (wins+losses). Cleaner + more authoritative than reconstructing it by
    scanning per-game scores (ESPN's method). Optional as_of_date picks the latest
    snapshot <= that date (leakage-safe). Fail-open → {}."""
    if not enabled():
        return {}
    season = int(season) if season else _current_season()
    s = mlb_team_standings
    try:
        with db_store.get_engine().connect() as conn:
            asof = _latest_standings_asof(conn, season, as_of_date)
            if asof is None:
                return {}
            rows = conn.execute(
                select(s.c.team_id, s.c.wins, s.c.losses, s.c.runs_allowed)
                .where(s.c.season == season)
                .where(s.c.as_of_date == asof)).fetchall()
    except (OperationalError, ValueError, TypeError):
        return {}
    names = _team_name_map()
    out = {}
    for tid, w, losses, ra in rows:
        games = (int(w) if w else 0) + (int(losses) if losses else 0)
        nm = names.get(tid)
        if nm and ra is not None and games > 0:
            out[nm] = float(ra) / games
    return out


def get_team_offense(season=None, as_of_date=None):
    """{team display name: avg runs SCORED per game} — the offensive analog of
    get_team_defense, from the latest /standings snapshot's cumulative runs_scored /
    games (wins+losses). Supplies opponent-OFFENSE strength (a strong lineup inflates
    an opposing pitcher's ER/hits/BF) and the run-environment side. Optional as_of_date
    picks the latest snapshot <= that date (leakage-safe). Fail-open → {}."""
    if not enabled():
        return {}
    season = int(season) if season else _current_season()
    s = mlb_team_standings
    try:
        with db_store.get_engine().connect() as conn:
            asof = _latest_standings_asof(conn, season, as_of_date)
            if asof is None:
                return {}
            rows = conn.execute(
                select(s.c.team_id, s.c.wins, s.c.losses, s.c.runs_scored)
                .where(s.c.season == season)
                .where(s.c.as_of_date == asof)).fetchall()
    except (OperationalError, ValueError, TypeError):
        return {}
    names = _team_name_map()
    out = {}
    for tid, w, losses, rs in rows:
        games = (int(w) if w else 0) + (int(losses) if losses else 0)
        nm = names.get(tid)
        if nm and rs is not None and games > 0:
            out[nm] = float(rs) / games
    return out


# ── P4 unified resolution: refresh an ACTIVE game, freeze a HISTORICAL one ─────
_HISTORICAL_MIN_AGE_HOURS = 48   # after this (+ one post-window refresh) → frozen
_BOXSCORE_ACTIVE_TTL = 900       # a recent game's box is re-pulled at most this often


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def _fact_captured_after(game_pk, game_dt, hours):
    """True if this game's facts were captured >= game_dt + hours ago — i.e. the
    one post-correction-window refresh has happened, so the fact is now frozen/
    historical. False when no fact exists yet or it was captured in-window."""
    try:
        cutoff = (game_dt + datetime.timedelta(hours=hours)).timestamp()
        engine = db_store.get_engine()
        with engine.connect() as conn:
            for tbl in (mlb_batter_game, mlb_pitcher_game):
                r = conn.execute(
                    select(tbl.c.fetched_at)
                    .where(tbl.c.game_pk == int(game_pk)).limit(1)).first()
                if r is not None and r[0] is not None:
                    return float(r[0]) >= cutoff
        return False
    except (OperationalError, ValueError, TypeError):
        return False


def refresh_game_facts(game_pk, active=True):
    """Re-pull ONE genuine-final game's boxscore and re-upsert its batter/pitcher
    facts so the warehouse reflects the settled line. ``active`` uses a short cache
    (a recent game, corrections may still land); ``active=False`` forces a fresh
    pull — the single post-window refresh that freezes a game as historical.
    Returns True if facts were (re)written. Fail-open; a no-op if the game isn't in
    mlb_game or isn't genuine-final."""
    if not enabled():
        return False
    try:
        game = get_game(game_pk)
        if not game or not mlb_starters._is_genuine_final(
                {"status": game.get("status"),
                 "detailedState": game.get("detailed_state")}):
            return False
        box = fetch_boxscore(game_pk, max_age=_BOXSCORE_ACTIVE_TTL if active else 0)
        engine = db_store.get_engine()
        with _WRITE_LOCK:
            for attempt in range(3):
                try:
                    with engine.begin() as conn:
                        # The post-window (active=False) refresh advances fetched_at
                        # so the game freezes → subsequent grades are 0-network.
                        _write_game_facts(conn, box, game,
                                          touch_fetched_at=not active)
                    return True
                except OperationalError:
                    if attempt == 2:
                        raise
    except Exception:
        return False
    return False


def resolve_actual(mlb_player_id, game_pk, prop_key):
    """Unified grading read (P4): resolve a prop's actual FROM THE WAREHOUSE, first
    refreshing a still-correctable game from a fresh boxscore so the database is the
    single source of truth.

    * HISTORICAL (>= _HISTORICAL_MIN_AGE_HOURS final AND its fact already captured
      after that window) → read the frozen fact, ZERO network.
    * ACTIVE (recent) or a just-crossed game whose fact predates the window → one
      refresh (fresh boxscore → merged facts), then read.

    Returns the value (a real 0 → 0.0), or None (unsupported prop / game not in the
    warehouse / not final / player DNP) so the caller can fall back to the live
    per-player path. Fail-open (never raises)."""
    if (_ACTUAL_STAT_SPEC.get(prop_key) is None or not mlb_player_id
            or game_pk is None or not enabled()):
        return None
    try:
        game = get_game(game_pk)
        if not game:                         # schedule not ingested → live fallback
            return None
        gd = _parse_ts(game.get("game_date"))
        active = True
        if gd is not None:
            age_h = (_utcnow() - gd).total_seconds() / 3600.0
            if (age_h >= _HISTORICAL_MIN_AGE_HOURS
                    and _fact_captured_after(game_pk, gd, _HISTORICAL_MIN_AGE_HOURS)):
                return get_actual_stat(mlb_player_id, game_pk, prop_key)  # frozen
            active = age_h < _HISTORICAL_MIN_AGE_HOURS
        refresh_game_facts(game_pk, active=active)
        return get_actual_stat(mlb_player_id, game_pk, prop_key)
    except Exception:
        return None


# ── P5: translate-in-place backfill of game_pk onto legacy durable rows ────────
def _legacy_gamepk_index(conn, season=None):
    """{(athlete_id, official_date): {game_pk: game_date}} from the batter+pitcher
    facts joined to the games dim — the PLAYER-ANCHOR lookup for the P5 backfill: a
    player's fact row pins the exact game they appeared in (doubleheader-safe).
    official_date embeds the year, so keys never collide across seasons."""
    idx = {}
    for tbl in (mlb_batter_game, mlb_pitcher_game):
        stmt = (select(tbl.c.athlete_id, tbl.c.game_pk,
                       mlb_game.c.official_date, mlb_game.c.game_date)
                .select_from(tbl.join(mlb_game, tbl.c.game_pk == mlb_game.c.game_pk)))
        if season is not None:
            stmt = stmt.where(mlb_game.c.season == int(season))
        for aid, gpk, offd, gdate in conn.execute(stmt):
            if aid and offd:
                idx.setdefault((str(aid), str(offd)), {})[gpk] = gdate
    return idx


# Two DH games whose start times are within this of EACH OTHER (relative to the
# row's commence) are treated as indistinguishable → leave game_pk NULL rather than
# guess (mirrors find_game_pk_by_commence's ambiguous-DH → None). A traditional
# single-admission DH can carry an identical gameDate; split DHs are hours apart.
_DH_TIE_TOL_SEC = 300


def _pick_legacy_game_pk(cands, commence):
    """From {game_pk: game_date} candidates for one (player, official_date): exactly
    one → it; a doubleheader (>1) → the game whose start is nearest the row's
    commence_time, UNLESS the two closest are within _DH_TIE_TOL_SEC (ambiguous) or
    commence is missing/unparseable/tz-naive → None (leave NULL, additive). Never
    raises (a bad timestamp can't abort the backfill)."""
    if not cands:
        return None
    if len(cands) == 1:
        return next(iter(cands))
    ct = _parse_ts(commence)
    if ct is None or ct.tzinfo is None:          # can't disambiguate a DH → NULL
        return None
    scored = []
    for gpk, gdate in cands.items():
        gt = _parse_ts(gdate)
        if gt is None or gt.tzinfo is None:
            continue
        try:
            scored.append((abs((gt - ct).total_seconds()), gpk))
        except (TypeError, ValueError):          # mixed awareness / bad ts → skip
            continue
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    if len(scored) >= 2 and (scored[1][0] - scored[0][0]) < _DH_TIE_TOL_SEC:
        return None                              # (near-)tie → ambiguous → NULL
    return scored[0][1]


def backfill_legacy_game_pk(dry_run=True, season=None):
    """P5: retro-match game_pk onto legacy MLB prediction_log + wagers rows that
    predate P3 stamping. PLAYER-ANCHOR: a row's player_mlb_id + official_date
    (game_date[:10]) pins the exact game_pk via the fact tables — the fact row IS the
    game the player appeared in, so doubleheaders are handled (a DH-both player is
    disambiguated by commence_time). Unresolvable (DNP → no fact, no player id, or an
    ambiguous DH) → left NULL (additive; the row still grades by name+date).

    NON-DESTRUCTIVE: only fills a row whose game_pk IS NULL, MLB only; idempotent
    (re-runs are no-ops). dry_run=True writes nothing and just reports coverage."""
    summary = {"dry_run": dry_run, "skipped": not enabled()}
    if not enabled():
        return summary
    eng = db_store.get_engine()
    with eng.connect() as conn:
        idx = _legacy_gamepk_index(conn, season=season)
    for name, table in (("prediction_log", db_store.prediction_log),
                        ("wagers", db_store.wagers)):
        with eng.connect() as conn:
            rows = conn.execute(
                select(table.c.id, table.c.player_mlb_id,
                       table.c.game_date, table.c.commence_time)
                .where((table.c.sport_key == "baseball_mlb")
                       & (table.c.game_pk.is_(None))
                       & (table.c.player_mlb_id.isnot(None)))).fetchall()
        matched = []
        for rid, mlb_id, gdate, commence in rows:
            offd = str(gdate)[:10] if gdate else None
            if not offd:
                continue
            gpk = _pick_legacy_game_pk(idx.get((str(mlb_id), offd), {}), commence)
            if gpk is not None:
                matched.append({"rid": rid, "gpk": int(gpk)})
        if matched and not dry_run:
            with _WRITE_LOCK:
                with eng.begin() as conn:
                    conn.execute(
                        update(table)
                        .where((table.c.id == bindparam("rid"))
                               & (table.c.game_pk.is_(None)))   # idempotent + safe
                        .values(game_pk=bindparam("gpk")),
                        matched)
        summary[name] = {"candidates": len(rows), "matched": len(matched)}
    return summary


def _team_game_index():
    """{(home_team_id, away_team_id, official_date): {game_pk: game_date}} over ALL
    mlb_game rows — loaded ONCE so the team-market backfill resolves in memory instead
    of a per-row SQL round-trip (thousands of remote round-trips → one bulk read).
    {} on error."""
    out = {}
    try:
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(select(
                mlb_game.c.home_team_id, mlb_game.c.away_team_id,
                mlb_game.c.official_date, mlb_game.c.game_pk,
                mlb_game.c.game_date)).fetchall()
    except (OperationalError, ValueError, TypeError):
        return {}
    for hid, aid, offd, gpk, gd in rows:
        if hid and aid and offd:
            out.setdefault((str(hid), str(aid), str(offd)[:10]), {})[gpk] = gd
    return out


def _resolve_team_game_pk_indexed(home_team, away_team, commence, game_date,
                                  name_cache, index):
    """DH-safe game_pk for a TEAM row from its team NAMES + commence/date, resolved
    IN MEMORY against `index` (_team_game_index) + a per-run team-name cache — zero
    per-row SQL. Same policy as the live stamp: commence-nearest (split-DH safe) over
    a ±1-day window, then a UNIQUE-official-date fallback (None on any DH). Wrapped so
    a bad row can never abort the backfill."""
    try:
        if home_team not in name_cache:
            name_cache[home_team] = (team_id_for_name_tolerant(home_team)
                                     if home_team else None)
        if away_team not in name_cache:
            name_cache[away_team] = (team_id_for_name_tolerant(away_team)
                                     if away_team else None)
        hid, aid = name_cache[home_team], name_cache[away_team]
        if not (hid and aid):
            return None
        ts = _parse_ts(commence)
        if ts is not None and ts.tzinfo is not None:
            cands = {}
            for d in (-1, 0, 1):
                day = (ts.date() + datetime.timedelta(days=d)).isoformat()
                cands.update(index.get((str(hid), str(aid), day), {}))
            gpk = _score_commence_candidates(cands, commence)
            if gpk is not None:
                return gpk
        if game_date:                        # unique-official-date fallback (DH-safe)
            same = index.get((str(hid), str(aid), str(game_date)[:10]), {})
            if len(same) == 1:
                return next(iter(same))
        return None
    except Exception:
        return None


def backfill_team_game_pk(dry_run=True, season=None):
    """Tier A #2: retro-match game_pk onto legacy MLB TEAM-market rows (wagers +
    odds_line) that predate #2 stamping. TEAM-ANCHOR (no player id): resolve the game
    from the row's team NAMES + commence_time by commence-nearest match (split-DH
    safe), with a unique-official-date fallback. An ambiguous same-timestamp DH /
    unknown team / tz-naive-or-missing commence -> left NULL (additive; the row still
    grades by name+date). All odds lines of one snapshot share the game, so the
    odds_line pass resolves ONCE per snapshot_id.

    PERFORMANCE: the whole mlb_game table is loaded ONCE into an in-memory index
    (_team_game_index) and every row resolves against it with NO per-row SQL — team
    names hit team_id_for_name_tolerant once per DISTINCT name (cached). So the run is
    ~4 queries + in-memory work, not thousands of remote round-trips.

    NON-DESTRUCTIVE + idempotent: only ever WRITES game_pk on a row whose game_pk IS
    NULL (guarded in both the SELECT and the UPDATE .where), MLB team-markets only.
    dry_run=True writes nothing and just reports coverage. Owner runs with --apply."""
    summary = {"dry_run": dry_run, "skipped": not enabled()}
    if not enabled():
        return summary
    eng = db_store.get_engine()
    team_types = ("moneyline", "spread", "total")
    index = _team_game_index()        # ONE bulk load; all resolution is in-memory
    name_cache = {}                   # distinct team name -> id (≈30-60 resolver calls)
    print(f"  [team-backfill] indexed {sum(len(v) for v in index.values())} games; "
          f"resolving in memory (dry_run={dry_run})…")

    def _resolve(home, away, commence, gdate):
        return _resolve_team_game_pk_indexed(home, away, commence, gdate,
                                             name_cache, index)

    # --- wagers pass (team-market money rows; player IS NULL) ---
    w = db_store.wagers
    wpred = ((w.c.sport_key == "baseball_mlb") & (w.c.game_pk.is_(None))
             & (w.c.player.is_(None)) & (w.c.bet_type.in_(team_types)))
    if season:
        wpred = wpred & (w.c.game_date.like(f"{int(season)}%"))
    with eng.connect() as conn:
        wrows = conn.execute(
            select(w.c.id, w.c.home_team, w.c.away_team, w.c.game_date,
                   w.c.commence_time).where(wpred)).fetchall()
    wmatched = [{"rid": rid, "gpk": int(gpk)}
                for rid, home, away, gdate, commence in wrows
                if (gpk := _resolve(home, away, commence, gdate)) is not None]
    if wmatched and not dry_run:
        with _WRITE_LOCK:
            with eng.begin() as conn:
                conn.execute(
                    update(w).where((w.c.id == bindparam("rid"))
                                    & (w.c.game_pk.is_(None)))
                    .values(game_pk=bindparam("gpk")), wmatched)
    summary["wagers"] = {"candidates": len(wrows), "matched": len(wmatched)}
    print(f"  [team-backfill] wagers: {len(wmatched)}/{len(wrows)} matched")

    # --- odds_line pass (team-market lines; resolve ONCE per snapshot, fan out) ---
    ol, os_ = db_store.odds_line, db_store.odds_snapshot
    opred = ((os_.c.sport == "baseball_mlb") & (ol.c.game_pk.is_(None))
             & (ol.c.bet_type.in_(team_types)))
    if season:
        opred = opred & (os_.c.game_date.like(f"{int(season)}%"))
    with eng.connect() as conn:
        orows = conn.execute(
            select(ol.c.id, ol.c.snapshot_id, os_.c.home, os_.c.away,
                   os_.c.game_date, os_.c.commence_time)
            .select_from(ol.join(os_, ol.c.snapshot_id == os_.c.id))
            .where(opred)).fetchall()
    snap_gpk = {}          # snapshot_id -> resolved gpk (a snapshot's lines share it)
    omatched = []
    for rid, sid, home, away, gdate, commence in orows:
        if sid not in snap_gpk:
            snap_gpk[sid] = _resolve(home, away, commence, gdate)
        if snap_gpk[sid] is not None:
            omatched.append({"rid": rid, "gpk": int(snap_gpk[sid])})
    if omatched and not dry_run:
        with _WRITE_LOCK:
            with eng.begin() as conn:
                for i in range(0, len(omatched), 500):     # chunk the large pass
                    conn.execute(
                        update(ol).where((ol.c.id == bindparam("rid"))
                                         & (ol.c.game_pk.is_(None)))
                        .values(game_pk=bindparam("gpk")), omatched[i:i + 500])
    summary["odds_line"] = {"candidates": len(orows), "matched": len(omatched)}
    print(f"  [team-backfill] odds_line: {len(omatched)}/{len(orows)} matched "
          f"({len(snap_gpk)} snapshots)")
    return summary


def purge_non_franchise_teams(dry_run=True):
    """Remove non-real 'teams' that leaked into mlb_team via the schedule — all-star
    squads + postseason bracket-seed placeholders (league_id AND division_id NULL;
    never in the /teams franchise roster) — and everything FK-bound to them, in FK
    order: facts → games → team rows. All REPRODUCIBLE (re-derivable from StatsAPI);
    the 30 real franchises (league_id populated) are untouched. dry_run=True reports
    the counts and writes nothing. Idempotent."""
    summary = {"dry_run": dry_run, "skipped": not enabled()}
    if not enabled():
        return summary
    eng = db_store.get_engine()
    with eng.connect() as conn:
        placeholder_ids = {r[0] for r in conn.execute(
            select(mlb_team.c.team_id).where(
                mlb_team.c.league_id.is_(None)
                & mlb_team.c.division_id.is_(None)))}
        game_pks = set()
        if placeholder_ids:
            game_pks = {r[0] for r in conn.execute(
                select(mlb_game.c.game_pk).where(
                    mlb_game.c.home_team_id.in_(placeholder_ids)
                    | mlb_game.c.away_team_id.in_(placeholder_ids)))}
        bf = pf = 0
        if game_pks:
            bf = conn.execute(select(func.count()).select_from(mlb_batter_game)
                              .where(mlb_batter_game.c.game_pk.in_(game_pks))).scalar()
            pf = conn.execute(select(func.count()).select_from(mlb_pitcher_game)
                              .where(mlb_pitcher_game.c.game_pk.in_(game_pks))).scalar()
    summary.update(teams=len(placeholder_ids), games=len(game_pks),
                   batter_facts=int(bf or 0), pitcher_facts=int(pf or 0))
    if placeholder_ids and not dry_run:
        with _WRITE_LOCK:
            with eng.begin() as conn:
                if game_pks:                      # facts → games (FK order)
                    conn.execute(delete(mlb_batter_game)
                                 .where(mlb_batter_game.c.game_pk.in_(game_pks)))
                    conn.execute(delete(mlb_pitcher_game)
                                 .where(mlb_pitcher_game.c.game_pk.in_(game_pks)))
                    conn.execute(delete(mlb_game)
                                 .where(mlb_game.c.game_pk.in_(game_pks)))
                conn.execute(delete(mlb_team)     # → team rows last
                             .where(mlb_team.c.team_id.in_(placeholder_ids)))
    return summary


def _fmt(summary):
    return json.dumps(summary, default=str)


def _main_cli():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="MLB StatsAPI medallion ingestion (P1 silver dims + bronze, "
                    "read-only dual-run — nothing app-facing consumes it yet).")
    ap.add_argument("--ingest", metavar="YYYY-MM-DD",
                    help="Ingest one date's schedule → dims (+ boxscore players).")
    ap.add_argument("--ingest-range", nargs=2, metavar=("START", "END"),
                    help="Ingest an inclusive date range.")
    ap.add_argument("--standings", nargs="?", const=0, type=int, metavar="SEASON",
                    help="Snapshot standings for SEASON (default: current year).")
    ap.add_argument("--handedness", nargs="?", const=0, type=int, metavar="SEASON",
                    help="Backfill mlb_player.bats/throws from the SEASON roster "
                         "(default: current year). Non-destructive; also runs each "
                         "ingest_maintenance cycle for the current season.")
    ap.add_argument("--no-boxscores", action="store_true",
                    help="Schedule/game dims only; skip boxscore player dims.")
    ap.add_argument("--no-bronze", action="store_true",
                    help="Fast backfill: skip the transient raw-JSON bronze audit "
                         "trail, writing only silver dims + facts (much less write "
                         "volume/round-trips per game). Use for bulk historical "
                         "re-ingest; the live/maintenance path keeps bronze.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="With --ingest-range: RESUME a #1c re-derive by skipping "
                         "games whose pitcher facts already have BB/BF populated (so a "
                         "killed run doesn't redo finished dates). Pair with "
                         "--no-bronze for the fastest catch-up.")
    ap.add_argument("--purge-boxscores", action="store_true",
                    help="Maintenance: delete processed boxscore bronze payloads.")
    ap.add_argument("--backfill-game-pk", action="store_true",
                    help="P5: retro-match game_pk onto legacy MLB prediction_log + "
                         "wagers rows (player-anchor). DRY-RUN unless --apply.")
    ap.add_argument("--backfill-team-game-pk", action="store_true",
                    help="Tier A #2: retro-match game_pk onto legacy MLB TEAM-market "
                         "rows (wagers + odds_line, team-anchor). DRY-RUN unless "
                         "--apply.")
    ap.add_argument("--purge-non-franchise", action="store_true",
                    help="Cleanup: remove non-real teams (all-star squads + "
                         "postseason bracket placeholders) + their games/facts from "
                         "the dim. DRY-RUN unless --apply.")
    ap.add_argument("--build-venues", action="store_true",
                    help="Batch A run_env: populate mlb_venue (StatsAPI /venues "
                         "lat/lon/elevation + park/geo priors via each venue's modal "
                         "home team). Idempotent upsert; run after the mlb_venue DDL.")
    ap.add_argument("--build-weather", metavar="SEASONS", default=None,
                    help="Batch A run_env: backfill weather_game (first-pitch-hour, "
                         "Visual Crossing) for regular games in SEASONS ('2024-2026' | "
                         "'2024,2025' | '2024'). DRY-RUN (reports fetch volume) unless "
                         "--apply. Incremental; needs WEATHER_API_KEY + --build-venues.")
    ap.add_argument("--apply", action="store_true",
                    help="With --backfill-game-pk / --backfill-team-game-pk / "
                         "--purge-non-franchise: WRITE the changes (default is a "
                         "no-write dry-run preview).")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to write.")

    did = False
    box = not args.no_boxscores
    bronze = not args.no_bronze
    if args.ingest:
        did = True
        print(_fmt(ingest_date(args.ingest, with_boxscores=box,
                               land_bronze=bronze)))
    if args.ingest_range:
        did = True
        skip = None
        if args.skip_existing:
            skip = pitcher_cols_done_game_pks(args.ingest_range[0],
                                              args.ingest_range[1])
            print(f"  [skip-existing] {len(skip)} games already have BF — skipping "
                  f"them (resume mode).")
        for res in ingest_range(args.ingest_range[0], args.ingest_range[1],
                                with_boxscores=box, land_bronze=bronze,
                                skip_game_pks=skip):
            print(_fmt(res))
    if args.standings is not None:
        did = True
        season = args.standings if args.standings and args.standings > 0 else None
        print(_fmt(ingest_standings(season)))
    if args.handedness is not None:
        did = True
        season = args.handedness if args.handedness and args.handedness > 0 else None
        print(f"handedness: updated {populate_handedness(season)} mlb_player rows")
    if args.purge_boxscores:
        did = True
        print(f"purged {purge_processed_boxscores()} processed boxscore rows")
    if args.backfill_game_pk:
        did = True
        print(_fmt(backfill_legacy_game_pk(dry_run=not args.apply)))
    if args.backfill_team_game_pk:
        did = True
        print(_fmt(backfill_team_game_pk(dry_run=not args.apply)))
    if args.build_venues:
        did = True
        print(_fmt(build_venue_dim()))
    if args.build_weather:
        did = True
        if "-" in args.build_weather:                 # range 'A-B' -> inclusive list
            lo, hi = args.build_weather.split("-")
            wx_seasons = list(range(int(lo), int(hi) + 1))
        else:
            wx_seasons = [int(s) for s in args.build_weather.split(",") if s.strip()]
        print(_fmt(build_weather(wx_seasons, apply=args.apply)))
    if args.purge_non_franchise:
        did = True
        print(_fmt(purge_non_franchise_teams(dry_run=not args.apply)))
    if not did:
        ap.error("nothing to do — pass --ingest, --ingest-range, --standings, "
                 "--handedness, --purge-boxscores, --backfill-game-pk, "
                 "--backfill-team-game-pk, or --purge-non-franchise")


if __name__ == "__main__":
    _main_cli()
