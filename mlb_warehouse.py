"""mlb_warehouse.py — MLB StatsAPI medallion: silver dims + bronze + game facts.

Read-only DUAL-RUN. Populates durable, StatsAPI-native dims (team / game / player),
a per-season standings fact, a provider→MLBAM alias scaffold, a transient bronze
raw-JSON landing table (all P1), and — P2 — the game-centric per-game batter /
pitcher stat FACTS (mlb_batter_game / mlb_pitcher_game), all in parallel with the
live ESPN path. NOTHING in the app consumes these tables yet (that begins at the
P4 cutover); the parity harness (mlb_warehouse_parity.py) diffs the StatsAPI-
derived shapes against the ESPN path before anything is switched over.

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
import datetime
import json
import threading
import time

from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, MetaData, String,
    Table, Text, UniqueConstraint, and_, delete, insert, not_, or_, select,
    update,
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
    Column("status", String(32)),                        # abstractGameState
    Column("detailed_state", String(64)),                # detailedState
    Column("home_score", Float),
    Column("away_score", Float),
    Column("fetched_at", Float),
    Index("ix_mlb_game_official_date", "official_date"),
    Index("ix_mlb_game_teams", "official_date", "home_team_id", "away_team_id"),
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
# HR/TB/RBI appended (2026-08-11) so batter_home_runs / batter_total_bases /
# batter_rbis become fact-servable LATER — captured now, but deliberately NOT yet in
# _ACTUAL_STAT_SPEC (that + a re-backfill is the "incorporate into the app" step;
# adding them early would read NULL-as-0.0 on un-backfilled rows).
_BATTER_GAME_STATS = ("AB", "H", "SO", "BB", "HBP", "SF", "SH", "HR", "TB", "RBI")
_PITCHER_GAME_STATS = ("IP", "K", "ER")


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
    )


mlb_batter_game = _game_fact_table("mlb_batter_game", _BATTER_GAME_STATS)
mlb_pitcher_game = _game_fact_table("mlb_pitcher_game", _PITCHER_GAME_STATS)

# Column-name SPECs — SchemaParityTests asserts these equal the Table columns.
_BRONZE_COLS = ("id", "kind", "natural_ref", "payload", "fetched_at", "processed_at")
_TEAM_COLS = ("team_id", "name", "name_norm", "abbreviation", "league_id",
              "division_id", "fetched_at")
_GAME_COLS = ("game_pk", "game_date", "official_date", "season", "game_type",
              "game_number", "double_header", "home_team_id", "away_team_id",
              "venue_id", "status", "detailed_state", "home_score", "away_score",
              "fetched_at")
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
def ingest_date(date, with_boxscores=True):
    """Ingest one YYYY-MM-DD slate → mlb_team/mlb_game (+ mlb_player and the
    StatsAPI-native per-game batter/pitcher stat facts from each genuine-final
    boxscore). Read-only DUAL-RUN: NO ESPN reads, and NOTHING app-facing consumes
    the result until the P4 cutover (the ESPN gamelog_store path stays live).
    Idempotent (surgical upsert)."""
    summary = {"date": date, "games": 0, "final": 0, "boxscores": 0,
               "players": 0, "batter_rows": 0, "pitcher_rows": 0,
               "skipped": not enabled()}
    if not enabled():
        return summary
    season = int(str(date)[:4])
    ensure_teams(season)

    raw = fetch_schedule(date)
    games, sched_teams = parse_schedule(raw, season)
    summary["games"] = len(games)

    engine = db_store.get_engine()
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                with engine.begin() as conn:
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

    seen_players = set()
    for g in games:
        if not mlb_starters._is_genuine_final(
                {"status": g["status"], "detailedState": g["detailed_state"]}):
            continue
        summary["final"] += 1
        try:
            box = fetch_boxscore(g["game_pk"])
        except Exception:
            continue
        players = parse_boxscore_players(box, g)
        nb = npi = 0
        with _WRITE_LOCK:
            for attempt in range(3):
                try:
                    with engine.begin() as conn:
                        _land_bronze(conn, "boxscore", str(g["game_pk"]), box)
                        db_store.upsert_bulk(conn, mlb_player, players,
                                             ("player_id",),
                                             ignore_cols=("fetched_at",))
                        nb, npi = _write_game_facts(conn, box, g)
                        _mark_bronze_processed(conn, "boxscore", str(g["game_pk"]))
                    break
                except OperationalError:
                    if attempt == 2:
                        raise
        summary["boxscores"] += 1
        summary["batter_rows"] += nb
        summary["pitcher_rows"] += npi
        seen_players.update(p["player_id"] for p in players)
    summary["players"] = len(seen_players)
    return summary


def ingest_range(start, end, with_boxscores=True):
    """Ingest an inclusive [start, end] date range; yields a per-date summary."""
    d0 = datetime.date.fromisoformat(str(start))
    d1 = datetime.date.fromisoformat(str(end))
    results = []
    cur = d0
    step = datetime.timedelta(days=1)
    while cur <= d1:
        results.append(ingest_date(cur.isoformat(), with_boxscores=with_boxscores))
        cur += step
    return results


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
    return summary


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
    if ts is None:
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
    scored = []
    for pk, gd in rows:
        gts = _parse_ts(gd)
        if gts is not None:
            scored.append((abs((gts - ts).total_seconds()), pk))
    if not scored:
        return None
    scored.sort()
    best_delta, best_pk = scored[0]
    if best_delta > tol_hours * 3600:
        return None
    if len(scored) > 1 and (scored[1][0] - best_delta) <= 60:   # ~tied → DH ambiguity
        return None
    return best_pk


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


def get_batter_game_log(athlete_id, season=None, as_of_date=None, limit=None):
    """Most-recent-first StatsAPI-native batter per-game log (P2, dual-run)."""
    return _game_log(mlb_batter_game, _BATTER_GAME_STATS, athlete_id,
                     season=season, as_of_date=as_of_date, limit=limit)


def get_pitcher_game_log(athlete_id, season=None, as_of_date=None, limit=None):
    """Most-recent-first StatsAPI-native pitcher per-game log (P2, dual-run)."""
    return _game_log(mlb_pitcher_game, _PITCHER_GAME_STATS, athlete_id,
                     season=season, as_of_date=as_of_date, limit=limit)


def _ip_to_outs(ip):
    """IP base-3 notation → out count (6.1 IP = 6*3+1 = 19 outs). Mirrors
    espn_client.ip_to_outs; kept local so the StatsAPI module stays ESPN-free."""
    if ip is None:
        return None
    whole = int(ip)
    return whole * 3 + round((ip - whole) * 10)


# Graded MLB props whose actual is a stored fact column → (table, column, xform).
# HR / total_bases / RBI are NOT stored (no column) → absent here → caller falls
# back to the live per-player fetch.
_ACTUAL_STAT_SPEC = {
    "batter_hits": (mlb_batter_game, "H", None),
    "batter_strikeouts": (mlb_batter_game, "SO", None),
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
    for r in rows:
        v = r.get(col)
        if xform == "ip_to_outs":
            v = _ip_to_outs(v)
        values.append(float(v) if v is not None else 0.0)
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
    own_team_id = rows[0]["team_id"]             # newest game's team (~ current)
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
    }


# ── P4 team-market model inputs: recent form / standings / team defense ───────
# StatsAPI-native team readers for the team markets. recent_games come from
# mlb_game (per-game scores — no better source); win%/record + team defense come
# from the /standings snapshot (mlb_team_standings), which carries the cumulative
# runsScored/runsAllowed DIRECTLY — cleaner + more authoritative than ESPN's
# scan-and-average, and it unlocks run differential / Pythagorean strength for a
# future team-market signal. Dual-run: nothing app-facing consumes these until the
# team-market flip. Fail-open; leakage-safe via as_of_date.
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
                    "away_score": as_, "total_score": hs + as_})
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
    ap.add_argument("--no-boxscores", action="store_true",
                    help="Schedule/game dims only; skip boxscore player dims.")
    ap.add_argument("--purge-boxscores", action="store_true",
                    help="Maintenance: delete processed boxscore bronze payloads.")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to write.")

    did = False
    box = not args.no_boxscores
    if args.ingest:
        did = True
        print(_fmt(ingest_date(args.ingest, with_boxscores=box)))
    if args.ingest_range:
        did = True
        for res in ingest_range(args.ingest_range[0], args.ingest_range[1],
                                with_boxscores=box):
            print(_fmt(res))
    if args.standings is not None:
        did = True
        season = args.standings if args.standings and args.standings > 0 else None
        print(_fmt(ingest_standings(season)))
    if args.purge_boxscores:
        did = True
        print(f"purged {purge_processed_boxscores()} processed boxscore rows")
    if not did:
        ap.error("nothing to do — pass --ingest, --ingest-range, --standings, "
                 "or --purge-boxscores")


if __name__ == "__main__":
    _main_cli()
