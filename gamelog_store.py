"""Durable rolling ESPN gamelog store (Blob->SQL "Phase C").

When the ``SQL_*`` secrets are configured, ESPN player gamelogs and athlete-id
lookups are cached in Azure SQL instead of the ephemeral file cache
(``espn_cache.py`` -> ``cache/backtest/*.json``), which is wiped on every
Streamlit Cloud restart. Completed games are fetched once and reused forever,
surviving restarts; repeated analyses within a TTL hit SQL (0 ESPN calls); past
seasons are stored once and never re-fetched.

Design
------
* **Reuses ``db_store``'s engine.** ``db_store.get_engine()`` / ``enabled()`` and
  the same SQLite-in-tests / mssql-in-prod plumbing. Own ``MetaData`` +
  ``create_all()`` (TEST-ONLY SQLite; prod DDL is hand-run from
  ``sql/gamelog_schema.sql`` -- appended to ``sql/schema.sql`` here).
* **Fully columnar, per-sport dense fact tables.** Columns = ONLY the stats the
  app actually reads (sourced from the verified ESPN gamelog contracts), so a
  read reconstructs the exact ``get_athlete_gamelog`` dict shape and all
  downstream extraction is untouched. No JSON, no EAV.
* **Replace-all per (athlete, season_bucket).** ESPN returns the whole current
  season with no "since" filter, so there is no ESPN-level incremental fetch;
  the win is the TTL cache (serve from SQL between refetches) + past seasons
  stored once. Each TTL-expired fetch DELETEs + re-INSERTs that athlete's rows
  (portable across SQLite + mssql; no MERGE/upsert), guarded so a transient
  partial response can't clobber a good log.
* **MLB + NBA + NFL.** Other sports (e.g. NHL) have no fact table -> the store
  passes through to the direct ESPN calls without persisting, so nothing
  regresses when SQL is on. Batter vs pitcher is one endpoint now (real
  per-game pitcher gamelogs); the synthesized-splits fallback is dormant.
* **Feature-flagged.** With SQL off this module is never reached; the file-cache
  path in ``espn_cache`` is byte-for-byte unchanged.
"""

import os
import threading
import time
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint, delete, insert, select,
)
from sqlalchemy.exc import OperationalError

import db_store

# 15 minutes -- mirrors espn_cache.NEGATIVE_TTL_HOURS so a transient ESPN miss
# (empty/failed fetch) recovers within minutes instead of sticking for the full
# success TTL.
NEGATIVE_TTL_HOURS = 0.25
# Default freshness for the live-app current-season path: short enough to pick up
# a game that finished EARLIER THE SAME DAY (e.g. game 1 of a doubleheader before
# you analyze game 2), long enough to serve repeated same-session analyses from
# SQL. ESPN is free (only rate-limited), so a few refetches/day/player is cheap.
# Past seasons (explicit season_year) are immutable -> a very long TTL
# (fetched once, never re-fetched). Note: this governs PROJECTION freshness only;
# bet settling is gated on FINAL game status, not this TTL.
LIVE_TTL_HOURS = 4
PAST_SEASON_TTL_HOURS = 24 * 365 * 5
# Athlete-id lookups (name->id) rarely change.
ATHLETE_TTL_HOURS = 24 * 30


# ──────────────────────────────────────────────────────────────────────────────
# Value coercion (mirrors db_store; typed columns convert back on read)
# ──────────────────────────────────────────────────────────────────────────────
def _s(v):
    return None if v is None else str(v)


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _b(v):
    return None if v is None else bool(v)  # tri-state (None preserved)


# ──────────────────────────────────────────────────────────────────────────────
# Schema (mirrors sql/schema.sql's gamelog section exactly)
# ──────────────────────────────────────────────────────────────────────────────
_META = MetaData()

# Stat columns per player type. Column name == the ESPN gamelog label key, so
# reconstruction re-emits the exact keys consumers read.
_BATTER_STATS = ("AB", "H", "SO", "BB", "HBP", "SF", "SH")
_PITCHER_STATS = ("IP", "K", "ER")
_NBA_STATS = ("MIN", "PTS", "REB", "AST")
# NFL gamelogs are ONE position-dependent row per game; across all three NFL props
# the app reads only two labels -- pass/rush yds both resolve to "YDS" (passing
# for a QB, rushing for a RB, receiving for a WR) and anytime-TD to "TD" -- so,
# like _NBA_STATS, this keeps only what's read (verified against the live ESPN
# NFL gamelog contract), not the full ~18-label row.
_NFL_STATS = ("YDS", "TD")
# Metadata keys re-emitted on every reconstructed row.
_META_KEYS = ("opponent", "is_home", "team_id", "game_date", "completed")


def _fact_table(name, stat_cols):
    return Table(
        name, _META,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("athlete_id", String(32), nullable=False),
        Column("season_bucket", Integer, nullable=False),   # 0 = current/None
        Column("game_key", String(220)),                    # synthetic (not unique)
        Column("game_date", String(40)),                    # FULL ISO timestamp
        Column("opponent", String(160)),
        Column("is_home", Boolean),
        Column("team_id", String(32)),
        Column("completed", Boolean),
        *[Column(c, Float) for c in stat_cols],
        Index(f"ix_{name}_athlete", "athlete_id", "season_bucket"),
    )


mlb_batter_gamelog = _fact_table("mlb_batter_gamelog", _BATTER_STATS)
mlb_pitcher_gamelog = _fact_table("mlb_pitcher_gamelog", _PITCHER_STATS)
nba_gamelog = _fact_table("nba_gamelog", _NBA_STATS)
nfl_gamelog = _fact_table("nfl_gamelog", _NFL_STATS)

# (player_type -> (Table, stat columns)). player_type is stored on the meta row.
_TABLE_FOR = {
    "batter": (mlb_batter_gamelog, _BATTER_STATS),
    "pitcher": (mlb_pitcher_gamelog, _PITCHER_STATS),
    "nba": (nba_gamelog, _NBA_STATS),
    "nfl": (nfl_gamelog, _NFL_STATS),
}

# TTL gate + which fact table an athlete lives in.
gamelog_fetch_meta = Table(
    "gamelog_fetch_meta", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sport", String(32), nullable=False),
    Column("league", String(32), nullable=False),
    Column("athlete_id", String(32), nullable=False),
    Column("season_bucket", Integer, nullable=False),
    Column("player_type", String(16)),
    Column("last_fetched_at", Float),                        # epoch seconds
    Column("game_count", Integer),
    UniqueConstraint("sport", "league", "athlete_id", "season_bucket",
                     name="uq_gamelog_fetch_meta"),
)

# Durable name->id (replaces the file cached_athlete_id). Stores team_id too so
# the get_player_stat_history reroute keeps athlete.team_id.
athlete_id_cache = Table(
    "athlete_id_cache", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sport", String(32), nullable=False),
    Column("league", String(32), nullable=False),
    Column("player_name_lower", String(160), nullable=False),
    Column("team_key", String(64), nullable=False),         # sorted team_ids or ""
    Column("athlete_id", String(32)),                       # None = not found
    Column("name", String(160)),
    Column("team_id", String(32)),
    Column("fetched_at", Float),
    UniqueConstraint("sport", "league", "player_name_lower", "team_key",
                     name="uq_athlete_id_cache"),
)

# Column-name SPECs for the schema-parity test (mirror test_db_store style).
_FACT_META_COLS = ("id", "athlete_id", "season_bucket", "game_key", "game_date",
                   "opponent", "is_home", "team_id", "completed")


def create_all():
    """Create the gamelog tables. TEST-ONLY (SQLite); prod DDL is hand-run."""
    _META.create_all(db_store.get_engine())


# ──────────────────────────────────────────────────────────────────────────────
# Per-key in-process lock (serializes fetch-and-store for one athlete/bucket).
# The analyze loop fires a pitcher's 3 prop futures concurrently for the SAME
# athlete; a single-replica Community Cloud makes an in-process lock sufficient.
# ──────────────────────────────────────────────────────────────────────────────
_KEY_LOCKS = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _key_lock(key):
    with _KEY_LOCKS_GUARD:
        lk = _KEY_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _KEY_LOCKS[key] = lk
        return lk


def _now():
    return time.time()


# ──────────────────────────────────────────────────────────────────────────────
# ESPN fetch (raw) + classification
# ──────────────────────────────────────────────────────────────────────────────
def _sport_has_table(sport):
    return sport in ("basketball", "football")


def _fetch_espn(sport, league, athlete_id, season_year):
    """Fetch the raw ESPN gamelog for NBA/NFL. (MLB is warehouse-only post-P6, so the
    old empty-gamelog pitcher/synth fallback is gone.)"""
    from espn_client import get_athlete_gamelog
    return get_athlete_gamelog(sport, league, athlete_id,
                               season_year=season_year) or []


def _classify(sport, rows):
    """player_type for the fetched rows (NBA/NFL only — MLB is warehouse-only)."""
    if sport == "basketball":
        return "nba"
    if sport == "football":
        return "nfl"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Row <-> column mapping
# ──────────────────────────────────────────────────────────────────────────────
def _game_key(row):
    return f"{row.get('game_date')}|{row.get('opponent')}|{row.get('is_home')}"


def _row_params(athlete_id, bucket, row, stat_cols):
    params = {
        "athlete_id": str(athlete_id),
        "season_bucket": bucket,
        "game_key": _game_key(row),
        "game_date": _s(row.get("game_date")),
        "opponent": _s(row.get("opponent")),
        "is_home": _b(row.get("is_home")),
        "team_id": _s(row.get("team_id")),
        "completed": _b(row.get("completed")),
    }
    for c in stat_cols:
        params[c] = _f(row.get(c))
    return params


def _reconstruct(mapping, stat_cols):
    """Rebuild the get_athlete_gamelog dict shape from a stored row, emitting a
    key only when its stored value is not None (mirrors ESPN, which omits absent
    stats; all consumers read via .get())."""
    out = {}
    for c in stat_cols:
        v = mapping[c]
        if v is not None:
            out[c] = v
    for k in _META_KEYS:
        v = mapping[k]
        if v is not None:
            out[k] = v
    return out


def _read_rows(conn, player_type, athlete_id, bucket):
    entry = _TABLE_FOR.get(player_type)
    if entry is None:            # unknown/negative-meta type → nothing stored
        return []
    table, stat_cols = entry
    result = conn.execute(
        select(table)
        .where((table.c.athlete_id == str(athlete_id))
               & (table.c.season_bucket == bucket))
        .order_by(table.c.id)          # insertion order == ESPN order (recent-first)
    )
    return [_reconstruct(r._mapping, stat_cols) for r in result]


# ──────────────────────────────────────────────────────────────────────────────
# Season / clobber-guard helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _rows_season(rows):
    """Season = year of the newest game_date, or None when dateless."""
    years = [dt.year for dt in (_parse_dt(r.get("game_date")) for r in rows) if dt]
    return max(years) if years else None


def _completed_count(rows):
    """Count games that are final (completed flag True, or past-dated when the
    flag is absent). Excludes today's in-progress game so a partial-day fetch
    doesn't look like a regression."""
    now = datetime.now(timezone.utc)
    count = 0
    for r in rows:
        comp = r.get("completed")
        if comp is True:
            count += 1
        elif comp is None:
            dt = _parse_dt(r.get("game_date"))
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < now:
                    count += 1
    return count


def _should_replace(stored_rows, new_rows):
    """True unless the fetch would DEGRADE the durable log:
      * a dateless refetch must never clobber a stored DATED log — dated rows
        are strictly richer (real-line calibration + as-of slicing need
        game_date), and a dateless set is the synth fallback that fires when the
        real StatsAPI log is momentarily unavailable; or
      * a same-season fetch returned FEWER completed games than are stored (a
        transient partial).
    A different (dated) season is a legitimate rollover -> always replace. Note a
    dateless->dated fetch (the synth->real migration) still replaces, since the
    stored side is then dateless and this guard only protects a dated store."""
    new_season = _rows_season(new_rows)
    stored_season = _rows_season(stored_rows)
    if new_season is None and stored_season is not None:
        return False                          # dateless synth must not clobber dated real
    if new_season != stored_season:
        return True
    return _completed_count(new_rows) >= _completed_count(stored_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Meta read / write (DELETE-by-key + INSERT == portable "upsert")
# ──────────────────────────────────────────────────────────────────────────────
def _read_meta(conn, sport, league, athlete_id, bucket):
    row = conn.execute(
        select(gamelog_fetch_meta).where(
            (gamelog_fetch_meta.c.sport == sport)
            & (gamelog_fetch_meta.c.league == league)
            & (gamelog_fetch_meta.c.athlete_id == str(athlete_id))
            & (gamelog_fetch_meta.c.season_bucket == bucket))
    ).first()
    return row._mapping if row is not None else None


def _write_meta(conn, sport, league, athlete_id, bucket, player_type, game_count):
    conn.execute(delete(gamelog_fetch_meta).where(
        (gamelog_fetch_meta.c.sport == sport)
        & (gamelog_fetch_meta.c.league == league)
        & (gamelog_fetch_meta.c.athlete_id == str(athlete_id))
        & (gamelog_fetch_meta.c.season_bucket == bucket)))
    conn.execute(insert(gamelog_fetch_meta), {
        "sport": sport, "league": league, "athlete_id": str(athlete_id),
        "season_bucket": bucket, "player_type": player_type,
        "last_fetched_at": _now(), "game_count": game_count,
    })


def _meta_fresh(meta, ttl_hours):
    if not meta or meta["last_fetched_at"] is None:
        return False
    ttl = ttl_hours if (meta["game_count"] or 0) > 0 else NEGATIVE_TTL_HOURS
    return (_now() - meta["last_fetched_at"]) < ttl * 3600


def _resolve_ttl(ttl_hours, bucket):
    if bucket != 0:                       # a specific past season is immutable
        return PAST_SEASON_TTL_HOURS
    if ttl_hours is None:
        ttl_hours = LIVE_TTL_HOURS
    env_ttl = os.environ.get("ODI_GAMELOG_TTL_HOURS")
    if env_ttl:
        try:
            ttl_hours = float(env_ttl)
        except ValueError:
            pass
    return ttl_hours


# ──────────────────────────────────────────────────────────────────────────────
# Public ops
# ──────────────────────────────────────────────────────────────────────────────
def get_gamelog(sport, league, athlete_id, season_year=None, ttl_hours=None,
                player_name=None):
    """Durable, TTL-gated replacement for espn_cache.cached_gamelog.

    Returns the full gamelog (most-recent-first, same shape as
    get_athlete_gamelog); the caller applies any [:n] slice. Sports without a
    fact table (NHL/other) pass through to the direct ESPN fetch with no
    persistence. ``player_name`` is accepted for call-site compatibility but no
    longer used (MLB is warehouse-only post-P6)."""
    if not _sport_has_table(sport):
        return _fetch_espn(sport, league, athlete_id, season_year)

    bucket = int(season_year) if season_year else 0
    ttl_hours = _resolve_ttl(ttl_hours, bucket)
    engine = db_store.get_engine()

    # Fast path: fresh meta -> serve from SQL (0 ESPN calls).
    with engine.connect() as conn:
        meta = _read_meta(conn, sport, league, athlete_id, bucket)
        if _meta_fresh(meta, ttl_hours):
            return _read_rows(conn, meta["player_type"], athlete_id, bucket)

    # Slow path: serialize per (athlete, bucket), re-check, then fetch + store.
    with _key_lock((sport, league, str(athlete_id), bucket)):
        with engine.connect() as conn:
            meta = _read_meta(conn, sport, league, athlete_id, bucket)
            if _meta_fresh(meta, ttl_hours):
                return _read_rows(conn, meta["player_type"], athlete_id, bucket)

        rows = _fetch_espn(sport, league, athlete_id, season_year)
        player_type = _classify(sport, rows)

        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    existing_type = meta["player_type"] if meta else None
                    if not rows or player_type is None:
                        # Empty/unclassifiable: keep any stored rows, apply the
                        # negative TTL (game_count=0) so retry cadence is short.
                        _write_meta(conn, sport, league, athlete_id, bucket,
                                    existing_type, 0)
                        return (_read_rows(conn, existing_type, athlete_id, bucket)
                                if existing_type else [])

                    if existing_type is not None:
                        stored = _read_rows(conn, existing_type, athlete_id, bucket)
                        if not _should_replace(stored, rows):
                            # Transient partial -> keep stored rows, short retry.
                            _write_meta(conn, sport, league, athlete_id, bucket,
                                        existing_type, 0)
                            return stored

                    table, stat_cols = _TABLE_FOR[player_type]
                    conn.execute(delete(table).where(
                        (table.c.athlete_id == str(athlete_id))
                        & (table.c.season_bucket == bucket)))
                    conn.execute(
                        insert(table),
                        [_row_params(athlete_id, bucket, r, stat_cols)
                         for r in rows])
                    _write_meta(conn, sport, league, athlete_id, bucket,
                                player_type, len(rows))
                return rows
            except OperationalError:      # transient lock/timeout -> retry
                if attempt == 2:
                    raise
    return rows


def _put_athlete_cache(sport, league, name_lower, team_key, row):
    """WS15: surgical upsert of the single athlete_id_cache row keyed by
    (sport, league, player_name_lower, team_key) via db_store.reconcile — an
    UPDATE-in-place (stable surrogate id) or INSERT instead of delete-by-key +
    insert. ``fetched_at`` is the cache-hit TTL gate, so it is refreshed on every
    call (NOT an ignore_col). Serialized per key with the same OperationalError
    retry the delete+insert path used."""
    engine = db_store.get_engine()
    scope = {"sport": sport, "league": league,
             "player_name_lower": name_lower, "team_key": team_key}
    with _key_lock(("athlete", sport, league, name_lower, team_key)):
        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    db_store.reconcile(
                        conn, athlete_id_cache, [row],
                        ("sport", "league", "player_name_lower", "team_key"),
                        scope=scope)
                return
            except OperationalError:
                if attempt == 2:
                    raise


def get_athlete_id(sport, league, name, team_ids=None, ttl_hours=ATHLETE_TTL_HOURS):
    """Durable name->id lookup. Returns {'id','name','team_id'} or None, matching
    espn_client.search_athlete. Works for all sports (sport-agnostic cache); on a
    cache miss it resolves via search_athlete. (MLB is warehouse-only post-P6 and no
    longer reaches here.)"""
    name_lower = (name or "").lower()
    team_key = "|".join(sorted(str(t) for t in team_ids if t)) if team_ids else ""
    engine = db_store.get_engine()

    def _hit(mapping):
        aid = mapping["athlete_id"]
        ttl = ttl_hours if aid else NEGATIVE_TTL_HOURS
        if mapping["fetched_at"] is not None and \
                (_now() - mapping["fetched_at"]) < ttl * 3600:
            return True
        return False

    with engine.connect() as conn:
        row = conn.execute(
            select(athlete_id_cache).where(
                (athlete_id_cache.c.sport == sport)
                & (athlete_id_cache.c.league == league)
                & (athlete_id_cache.c.player_name_lower == name_lower)
                & (athlete_id_cache.c.team_key == team_key))
        ).first()
        if row is not None and _hit(row._mapping):
            m = row._mapping
            return ({"id": m["athlete_id"], "name": m["name"] or name,
                     "team_id": m["team_id"]} if m["athlete_id"] else None)

    from espn_client import search_athlete
    athlete = search_athlete(sport, league, name, team_ids=team_ids)
    aid = athlete["id"] if athlete else None

    _put_athlete_cache(sport, league, name_lower, team_key, {
        "sport": sport, "league": league,
        "player_name_lower": name_lower, "team_key": team_key,
        "athlete_id": _s(aid),
        "name": _s(athlete.get("name")) if athlete else None,
        "team_id": _s(athlete.get("team_id")) if athlete else None,
        "fetched_at": _now(),
    })
    return athlete


def seed_athlete_id(sport, league, name, athlete_id, team_ids=None):
    """Pre-populate the durable name->id cache with a KNOWN id (the SQL analog of
    espn_cache.seed_athlete_id).

    Lets a caller that already holds an authoritative ESPN athlete id (e.g. the
    calibration-pool builder, from the season statistics listing) pin it so a
    later get_athlete_id skips search_athlete's lossy first-name match. Written
    under the same (sport, league, name, team_key) key get_athlete_id reads, with
    a fresh timestamp so it counts as a hit. Overwrites any existing row for that
    key (the seed is authoritative). No-op on a falsy id."""
    if not athlete_id:
        return
    name_lower = (name or "").lower()
    team_key = "|".join(sorted(str(t) for t in team_ids if t)) if team_ids else ""
    _put_athlete_cache(sport, league, name_lower, team_key, {
        "sport": sport, "league": league,
        "player_name_lower": name_lower, "team_key": team_key,
        "athlete_id": _s(athlete_id), "name": _s(name),
        "team_id": None, "fetched_at": _now(),
    })
