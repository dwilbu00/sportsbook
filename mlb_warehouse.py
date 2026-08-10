"""mlb_warehouse.py — MLB StatsAPI medallion: silver dims + bronze landing (P1).

Read-only DUAL-RUN. Populates durable, StatsAPI-native dims (team / game / player),
a per-season standings fact, a provider→MLBAM alias scaffold, and a transient
bronze raw-JSON landing table — in parallel with the live ESPN path. NOTHING in
the app consumes these tables yet (that begins at the P4 cutover); P1 exists so a
parity harness (mlb_warehouse_parity.py) can diff the StatsAPI-derived shapes
against the ESPN path before anything is switched over.

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
  * P1 does NOT write the gamelog FACT tables — that re-key is P2. The
    boxscore-derived batter/pitcher rows (derive_*_rows) are produced here only
    for the parity harness; P1 persists dims + bronze, not facts.

MLB only. NBA/NFL stay on ESPN.
"""

from __future__ import annotations

import argparse
import datetime
import json
import threading
import time

from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, MetaData, String,
    Table, Text, UniqueConstraint, delete, insert, select, update,
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

# Column-name SPECs — SchemaParityTests asserts these equal the Table columns.
_BRONZE_COLS = ("id", "kind", "natural_ref", "payload", "fetched_at", "processed_at")
_TEAM_COLS = ("team_id", "name", "name_norm", "abbreviation", "league_id",
              "division_id", "fetched_at")
_GAME_COLS = ("game_pk", "game_date", "official_date", "season", "game_number",
              "double_header", "home_team_id", "away_team_id", "venue_id",
              "status", "detailed_state", "home_score", "away_score", "fetched_at")
_PLAYER_COLS = ("player_id", "full_name", "name_norm", "primary_position",
                "is_pitcher", "bats", "throws", "fetched_at")
_STANDINGS_COLS = ("id", "team_id", "season", "as_of_date", "wins", "losses",
                   "win_pct", "fetched_at")
_ALIAS_COLS = ("id", "provider", "provider_key", "mlb_player_id", "confidence",
               "resolution_method", "valid_from", "valid_to", "fetched_at")


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


def fetch_boxscore(game_pk):
    """Raw /game/{gamePk}/boxscore payload. Only fetched for genuine-final games
    (immutable) → long cache."""
    cache = f"warehouse_boxscore_{game_pk}"
    cached = mlb_starters._read_cache(cache, max_age=_SCHED_FINAL_TTL)
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
    """Raw boxscore → per-batter stat rows (P1: for the parity harness, not
    persisted). Shape mirrors the mlb_batter_gamelog fact: athlete_id (MLBAM),
    game_pk, team_id, AB/H/SO/BB/HBP/SF/SH. Only players who came to the plate."""
    rows = []
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
            rows.append({
                "athlete_id": str(person.get("id")),
                "game_pk": gid,
                "team_id": team_id,
                "AB": _f(bat.get("atBats")),
                "H": _f(bat.get("hits")),
                "SO": _f(bat.get("strikeOuts")),
                "BB": _f(bat.get("baseOnBalls")),
                "HBP": _f(bat.get("hitByPitch")),
                "SF": _f(bat.get("sacFlies")),
                "SH": _f(bat.get("sacBunts")),
            })
    return rows


def derive_pitcher_rows(box, game):
    """Raw boxscore → per-pitcher stat rows (P1: parity harness only). Shape
    mirrors mlb_pitcher_gamelog: athlete_id, game_pk, team_id, IP/K/ER. IP is the
    base-3 float ("6.1" == 6IP + 1 out), matching get_pitcher_gamelog."""
    rows = []
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
            rows.append({
                "athlete_id": str(person.get("id")),
                "game_pk": gid,
                "team_id": team_id,
                "IP": _f(ip_raw),
                "K": _f(pit.get("strikeOuts")),
                "ER": _f(pit.get("earnedRuns")),
            })
    return rows


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
    """Single-key surgical upsert of each row (scoped to its own natural key so a
    call never deletes a sibling row). Returns (n_ins, n_upd, n_del)."""
    if not rows or not enabled():
        return (0, 0, 0)
    engine = db_store.get_engine()
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                tot = [0, 0, 0]
                with engine.begin() as conn:
                    for row in rows:
                        scope = {k: row[k] for k in key_cols}
                        i, u, d = db_store.reconcile(
                            conn, table, [row], key_cols,
                            scope=scope, ignore_cols=ignore_cols)
                        tot[0] += i
                        tot[1] += u
                        tot[2] += d
                return tuple(tot)
            except OperationalError:
                if attempt == 2:
                    raise
    return (0, 0, 0)


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
    """Ingest one YYYY-MM-DD slate → mlb_team/mlb_game (+ mlb_player from each
    genuine-final boxscore). Read-only dual-run: NO gamelog facts, NO ESPN reads,
    NOTHING app-facing consumes the result. Idempotent (surgical upsert)."""
    summary = {"date": date, "games": 0, "final": 0, "boxscores": 0,
               "players": 0, "skipped": not enabled()}
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
                    for t in sched_teams:        # minimal → FK safety
                        db_store.reconcile(
                            conn, mlb_team, [t], ("team_id",),
                            scope={"team_id": t["team_id"]},
                            ignore_cols=("fetched_at",))
                    for g in games:
                        db_store.reconcile(
                            conn, mlb_game, [g], ("game_pk",),
                            scope={"game_pk": g["game_pk"]},
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
        with _WRITE_LOCK:
            for attempt in range(3):
                try:
                    with engine.begin() as conn:
                        _land_bronze(conn, "boxscore", str(g["game_pk"]), box)
                        for p in players:
                            db_store.reconcile(
                                conn, mlb_player, [p], ("player_id",),
                                scope={"player_id": p["player_id"]},
                                ignore_cols=("fetched_at",))
                        _mark_bronze_processed(conn, "boxscore", str(g["game_pk"]))
                    break
                except OperationalError:
                    if attempt == 2:
                        raise
        summary["boxscores"] += 1
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
                    for r in rows:
                        db_store.reconcile(
                            conn, mlb_team_standings, [r],
                            ("team_id", "season", "as_of_date"),
                            scope={"team_id": r["team_id"], "season": r["season"],
                                   "as_of_date": r["as_of_date"]},
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
