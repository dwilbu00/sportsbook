"""Smart Fantasy Baseball ID cross-maps — the data-integrity backbone.

Player and team identity in this app is reconciled BY NAME STRING off one book-feed
name, then fanned out to three independent, lossy name-spaces:

* the betting ``player`` name (``prediction_log`` / ``wagers`` / ``odds_line``),
* the **ESPN** ``athlete_id`` (``mlb_*_gamelog`` / ``athlete_id_cache``), and
* the **MLBAM** ``player_id`` (``statcast_player_asof`` / statsapi).

The two numeric id-spaces the DB already stores (ESPN + MLBAM) are never linked to
each other or to the name. Smart Fantasy Baseball publishes two authoritative
cross-maps that close the gap:

* **Player map** (``PLAYERIDMAPCSV``) — ~3,800 players carrying ``MLBID`` (== the
  MLBAM id used by BOTH statsapi and Statcast), ``ESPNID``, ``IDPLAYER`` (SFBB's
  own stable key), ``DRAFTKINGSNAME``, etc. One numeric key (MLBID) covers the
  whole MLB player pipeline; ESPNID bridges to the ESPN gamelog tables.
* **Team map** (``TEAMMAPLINK``) — 30 clubs; per-source abbreviations/nicknames.
  No numeric ids → canonical key = the stable 3-letter ``SFBBTEAM`` code.

This module stores both maps in Azure SQL and serves fail-open lookups the app's
resolvers route through. It mirrors ``statcast_asof.py`` exactly: own ``MetaData``
+ ``create_all()`` (TEST-ONLY SQLite; prod DDL is hand-run from ``sql/schema.sql``),
fail-open reads, DELETE-all + bulk ``insert()`` replace-writes under a lock with
3× ``OperationalError`` retry, a TTL meta gate, and a ``--refresh`` CLI.

Everything fails open: a map miss / SQL-off / non-MLB player → ``None``/``{}`` and
the caller's existing name-based path is used unchanged. No hard dependency on SFBB
availability — the lazy in-app refresh keeps serving a stale in-memory index when
the source is briefly unreachable; only the CLI raises loudly.
"""

import argparse
import csv
import io
import threading
import time

import requests
from sqlalchemy import (
    Boolean, Column, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint, delete, insert, select,
)
from sqlalchemy.exc import OperationalError

import db_store

# The SFBB CSV endpoints 307-redirect (players) / Google-Sheets-redirect (teams)
# and reject a default library User-Agent (406), so send a browser UA and follow
# redirects — the same reason mlb_starters/savant_history spoof the UA for Savant.
PLAYER_URL = "https://smartfantasybaseball.com/PLAYERIDMAPCSV"
TEAM_URL = "https://www.smartfantasybaseball.com/TEAMMAPLINK"
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
}
_FETCH_TIMEOUT = 60

# Player ids change slowly (a callup here and there); teams change once a decade.
PLAYER_TTL_HOURS = 24
TEAM_TTL_HOURS = 24 * 7
# Don't hit the meta row more than once per this interval per process (the lazy
# freshness check runs on every resolver call; the TTLs above gate the web fetch).
_FRESH_CHECK_INTERVAL_S = 600

# Cleveland's nickname is stale in the map's display column (FANGRAPHSTEAM shows
# "Indians"); the odds feed / box scores say "Guardians". The abbr columns (CLE)
# are correct, so this only patches the nickname → code path.
_TEAM_NICK_OVERRIDES = {"guardians": "CLE"}


# ──────────────────────────────────────────────────────────────────────────────
# Schema (SQLAlchemy Core; mirrors sql/schema.sql exactly —
# test_player_id_map.py::SchemaParityTests enforces it)
# ──────────────────────────────────────────────────────────────────────────────
_META = MetaData()

player_id_map = Table(
    "player_id_map", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sfbb_id", String(32), nullable=False),      # IDPLAYER (SFBB stable key)
    Column("mlb_id", String(32)),                        # MLBID = MLBAM
    Column("espn_id", String(32)),                       # ESPNID
    Column("bref_id", String(32)),                       # BREFID
    Column("fangraphs_id", String(32)),                  # IDFANGRAPHS
    Column("name", String(160)),                         # PLAYERNAME
    Column("name_norm", String(160), nullable=False),    # normalize_name(PLAYERNAME)
    Column("team", String(16)),                          # TEAM (SFBB code)
    Column("pos", String(16)),                           # POS
    Column("allpos", String(64)),                        # ALLPOS
    Column("bats", String(8)),
    Column("throws", String(8)),
    Column("dk_name", String(160)),                      # DRAFTKINGSNAME
    Column("active", Boolean),
    Column("source", String(64)),
    Column("fetched_at", Float),                         # epoch seconds
    UniqueConstraint("sfbb_id", name="uq_player_id_map"),
    Index("ix_player_id_map_mlb", "mlb_id"),
    Index("ix_player_id_map_espn", "espn_id"),
    Index("ix_player_id_map_name", "name_norm"),
)

team_id_map = Table(
    "team_id_map", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sfbb_code", String(16), nullable=False),     # SFBBTEAM (canonical)
    Column("dk_code", String(16)),                       # DKTEAM
    Column("espn_code", String(16)),                     # ESPNTEAM
    Column("bbref_code", String(16)),                    # BBREFTEAM
    Column("fangraphs_abbr", String(16)),                # FANGRAPHSABBR
    Column("retrosheet", String(16)),                    # RETROSHEET
    Column("nickname", String(64)),                      # FANGRAPHSTEAM (display; stale)
    Column("name_norm", String(64), nullable=False),     # normalize_name(nickname)
    Column("source", String(64)),
    Column("fetched_at", Float),
    UniqueConstraint("sfbb_code", name="uq_team_id_map"),
    Index("ix_team_id_map_name", "name_norm"),
)

id_map_meta = Table(
    "id_map_meta", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("map_name", String(16), nullable=False),      # 'player' | 'team'
    Column("last_fetched_at", Float),                    # epoch seconds
    Column("row_count", Integer),
    UniqueConstraint("map_name", name="uq_id_map_meta"),
)

# Column-name SPECs for the schema-parity drift test (mirror statcast_asof._COLS).
_PLAYER_COLS = ("id", "sfbb_id", "mlb_id", "espn_id", "bref_id", "fangraphs_id",
                "name", "name_norm", "team", "pos", "allpos", "bats", "throws",
                "dk_name", "active", "source", "fetched_at")
_TEAM_COLS = ("id", "sfbb_code", "dk_code", "espn_code", "bbref_code",
              "fangraphs_abbr", "retrosheet", "nickname", "name_norm", "source",
              "fetched_at")
_META_COLS = ("id", "map_name", "last_fetched_at", "row_count")

_WRITE_LOCK = threading.Lock()
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


def enabled():
    return db_store.enabled()


def create_all():
    """Create the tables. TEST-ONLY (SQLite); prod DDL is hand-run from schema.sql."""
    _META.create_all(db_store.get_engine())


# ──────────────────────────────────────────────────────────────────────────────
# Coercion helpers (CSV cell → typed DB value; blank → None)
# ──────────────────────────────────────────────────────────────────────────────
def _s(v):
    """Stripped text, or None if empty/blank."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _upper(v):
    s = _s(v)
    return s.upper() if s else None


def _active(v):
    """SFBB ACTIVE is 'Y'/'N'; anything else → None (unknown)."""
    s = _s(v)
    if s is None:
        return None
    u = s.upper()
    if u in ("Y", "YES", "TRUE", "1"):
        return True
    if u in ("N", "NO", "FALSE", "0"):
        return False
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Fetch + parse (stdlib csv; NO pandas)
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_csv(url):
    """GET a SFBB CSV endpoint (browser UA, follow redirects) → list of row dicts.
    Decodes utf-8-sig so a leading BOM is stripped from the first header."""
    resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=_FETCH_TIMEOUT,
                        allow_redirects=True)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _dedup_by_anchor(rows, anchor, prefer="mlb_id"):
    """Collapse rows sharing the same UNIQUE anchor to one so a duplicate id in the
    SFBB feed can't violate the table's UNIQUE constraint. SFBB occasionally ships
    two rows for one IDPLAYER (e.g. a retired player listed twice — the 'burneaj01'
    duplicate that crashed the first bulk load). Keeps the first occurrence, but
    upgrades to a later row that carries ``prefer`` (mlb_id — the key the whole
    pipeline joins on) when the kept one lacks it. Deterministic in CSV order."""
    by, order = {}, []
    for r in rows:
        k = r.get(anchor)
        if k not in by:
            by[k] = r
            order.append(k)
        elif prefer and not by[k].get(prefer) and r.get(prefer):
            by[k] = r
    return [by[k] for k in order]


def _parse_player_rows(csv_rows, fetched_at):
    """SFBB player CSV rows → player_id_map insert params, one per IDPLAYER.

    Anchored on IDPLAYER (never blank); a row with no usable name is dropped
    (name_norm is NOT NULL). Explicitly-inactive rows (ACTIVE=N) are dropped: a
    retired player is noise for a live-slate bettor and, decisively, SFBB reuses
    /duplicates IDPLAYER across some of them (two inactive 'burneaj01' rows), which
    violates uq_player_id_map. A blank/unknown ACTIVE flag is KEPT — never drop a
    possibly-current player on a malformed flag. A final dedup on IDPLAYER makes
    the write bulletproof even against an active/active duplicate."""
    out = []
    for r in csv_rows:
        sfbb_id = _s(r.get("IDPLAYER"))
        if not sfbb_id:
            continue
        active = _active(r.get("ACTIVE"))
        if active is False:                 # drop only confirmed-retired players
            continue
        name = _s(r.get("PLAYERNAME")) or _s(r.get("MLBNAME"))
        name_norm = db_store.normalize_name(name)
        if not name_norm:
            continue
        out.append({
            "sfbb_id": sfbb_id,
            "mlb_id": _s(r.get("MLBID")),
            "espn_id": _s(r.get("ESPNID")),
            "bref_id": _s(r.get("BREFID")),
            "fangraphs_id": _s(r.get("IDFANGRAPHS")),
            "name": name,
            "name_norm": name_norm,
            "team": _upper(r.get("TEAM")),
            "pos": _s(r.get("POS")),
            "allpos": _s(r.get("ALLPOS")),
            "bats": _s(r.get("BATS")),
            "throws": _s(r.get("THROWS")),
            "dk_name": _s(r.get("DRAFTKINGSNAME")),
            "active": active,
            "source": "sfbb",
            "fetched_at": fetched_at,
        })
    return _dedup_by_anchor(out, "sfbb_id")


def _parse_team_rows(csv_rows, fetched_at):
    """SFBB team CSV rows → team_id_map insert params. Anchored on SFBBTEAM."""
    out = []
    for r in csv_rows:
        code = _upper(r.get("SFBBTEAM"))
        if not code:
            continue
        nickname = _s(r.get("FANGRAPHSTEAM"))
        out.append({
            "sfbb_code": code,
            "dk_code": _upper(r.get("DKTEAM")),
            "espn_code": _upper(r.get("ESPNTEAM")),
            "bbref_code": _upper(r.get("BBREFTEAM")),
            "fangraphs_abbr": _upper(r.get("FANGRAPHSABBR")),
            "retrosheet": _upper(r.get("RETROSHEET")),
            "nickname": nickname,
            "name_norm": db_store.normalize_name(nickname or code),
            "source": "sfbb",
            "fetched_at": fetched_at,
        })
    # Dedup on the UNIQUE anchor for symmetry/robustness (teams have no mlb_id, so
    # this is a plain keep-first — 30 unique codes in practice, but cheap insurance).
    return _dedup_by_anchor(out, "sfbb_code", prefer=None)


# ──────────────────────────────────────────────────────────────────────────────
# Replace-write (DELETE-all + bulk INSERT == portable rebuild) + meta upsert
# ──────────────────────────────────────────────────────────────────────────────
def _write_meta(conn, map_name, row_count):
    conn.execute(delete(id_map_meta).where(id_map_meta.c.map_name == map_name))
    conn.execute(insert(id_map_meta), {
        "map_name": map_name, "last_fetched_at": _now(), "row_count": row_count})


def _replace_write(table, params, map_name):
    engine = db_store.get_engine()
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    conn.execute(delete(table))
                    if params:
                        conn.execute(insert(table), params)
                    _write_meta(conn, map_name, len(params))
                return len(params)
            except OperationalError:
                if attempt == 2:
                    raise
    return len(params)


def refresh_players():
    """Fetch + replace the player map from SFBB. Returns rows written. Raises on a
    fetch/DB error (the CLI surfaces it; the lazy in-app path swallows it)."""
    params = _parse_player_rows(_fetch_csv(PLAYER_URL), _now())
    n = _replace_write(player_id_map, params, "player")
    _invalidate_index("player")
    return n


def refresh_teams():
    """Fetch + replace the team map from SFBB. Returns rows written."""
    params = _parse_team_rows(_fetch_csv(TEAM_URL), _now())
    n = _replace_write(team_id_map, params, "team")
    _invalidate_index("team")
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Meta / TTL gate + lazy freshness
# ──────────────────────────────────────────────────────────────────────────────
def _read_meta(conn, map_name):
    row = conn.execute(
        select(id_map_meta).where(id_map_meta.c.map_name == map_name)).first()
    return row._mapping if row is not None else None


def _meta_fresh(meta, ttl_hours):
    if not meta or meta["last_fetched_at"] is None:
        return False
    return (_now() - meta["last_fetched_at"]) < ttl_hours * 3600


_LAST_FRESH_CHECK = {"player": 0.0, "team": 0.0}
_FRESH_LOCK = threading.Lock()


def _ensure_fresh(map_name):
    """Best-effort lazy refresh: if the stored map is past its TTL, re-fetch it
    (serialized per map). Throttled to one meta read per process per interval, and
    ALWAYS returns fast — any error leaves the stale in-memory index in place."""
    if not enabled():
        return
    now = _now()
    with _FRESH_LOCK:
        if now - _LAST_FRESH_CHECK.get(map_name, 0.0) < _FRESH_CHECK_INTERVAL_S:
            return
        _LAST_FRESH_CHECK[map_name] = now
    ttl = PLAYER_TTL_HOURS if map_name == "player" else TEAM_TTL_HOURS
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            if _meta_fresh(_read_meta(conn, map_name), ttl):
                return
    except (OperationalError, ValueError, TypeError, RuntimeError):
        return
    with _key_lock(("refresh", map_name)):
        try:
            engine = db_store.get_engine()
            with engine.connect() as conn:
                if _meta_fresh(_read_meta(conn, map_name), ttl):
                    return
            if map_name == "player":
                refresh_players()
            else:
                refresh_teams()
        except Exception:                      # noqa: BLE001 — never break a lookup
            pass


# ──────────────────────────────────────────────────────────────────────────────
# In-process cached indexes (lazy-loaded from SQL, rebuilt after a write)
# ──────────────────────────────────────────────────────────────────────────────
_PLAYER_INDEX = None
_TEAM_INDEX = None
_INDEX_LOCK = threading.Lock()


def _invalidate_index(map_name):
    global _PLAYER_INDEX, _TEAM_INDEX
    with _INDEX_LOCK:
        if map_name == "player":
            _PLAYER_INDEX = None
        else:
            _TEAM_INDEX = None


def _prefer_active(existing, candidate):
    """True if ``candidate`` should replace ``existing`` in a by-id index — i.e. no
    incumbent, or the candidate is active (so an active row wins over an inactive
    namesake sharing the id)."""
    return existing is None or bool(candidate.get("active"))


def _build_player_index():
    idx = {"by_name": {}, "by_mlb": {}, "by_espn": {}}
    if not enabled():
        return idx
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(select(player_id_map))]
    except (OperationalError, ValueError, TypeError, RuntimeError):
        return idx
    for r in rows:
        nn = r.get("name_norm")
        if nn:
            idx["by_name"].setdefault(nn, []).append(r)
        dk = db_store.normalize_name(r.get("dk_name") or "")
        if dk and dk != nn:
            idx["by_name"].setdefault(dk, []).append(r)
        mid = r.get("mlb_id")
        if mid and _prefer_active(idx["by_mlb"].get(mid), r):
            idx["by_mlb"][mid] = r
        eid = r.get("espn_id")
        if eid and _prefer_active(idx["by_espn"].get(eid), r):
            idx["by_espn"][eid] = r
    return idx


def _build_team_index():
    idx = {"by_code": {}, "by_abbr": {}, "by_name": {}}
    if not enabled():
        return idx
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(select(team_id_map))]
    except (OperationalError, ValueError, TypeError, RuntimeError):
        return idx
    # Canonical code first, so an abbr shared across sources can never displace a
    # team's own SFBB code (and by_code stays authoritative).
    for r in rows:
        idx["by_code"][r["sfbb_code"]] = r
    for r in rows:
        code = r["sfbb_code"]
        for abbr in (r.get("sfbb_code"), r.get("dk_code"), r.get("espn_code"),
                     r.get("bbref_code"), r.get("fangraphs_abbr"),
                     r.get("retrosheet")):
            if abbr:
                idx["by_abbr"].setdefault(abbr.upper(), code)
        nn = r.get("name_norm")
        if nn:
            idx["by_name"].setdefault(nn, code)
    for nick, code in _TEAM_NICK_OVERRIDES.items():
        if code in idx["by_code"]:
            idx["by_name"][nick] = code
    return idx


def _player_idx():
    global _PLAYER_INDEX
    _ensure_fresh("player")
    if _PLAYER_INDEX is None:
        with _INDEX_LOCK:
            if _PLAYER_INDEX is None:
                _PLAYER_INDEX = _build_player_index()
    return _PLAYER_INDEX


def _team_idx():
    global _TEAM_INDEX
    _ensure_fresh("team")
    if _TEAM_INDEX is None:
        with _INDEX_LOCK:
            if _TEAM_INDEX is None:
                _TEAM_INDEX = _build_team_index()
    return _TEAM_INDEX


# ──────────────────────────────────────────────────────────────────────────────
# Fail-open lookups (return None/{} on miss, SQL-off, or error)
# ──────────────────────────────────────────────────────────────────────────────
def _rows_for_name(name):
    idx = _player_idx()
    return idx["by_name"].get(db_store.normalize_name(name)) or []


def _unique_id(rows, field):
    """The single distinct non-empty ``field`` across ``rows`` (preferring active
    rows), or None when zero or ambiguous — mirrors find_player_id's refusal to
    bind a non-unique name. Two rows for one two-way player (same id) still resolve;
    two genuine namesakes (two ids) do not."""
    active = [r for r in rows if r.get("active")]
    pool = active or rows
    ids = {r.get(field) for r in pool if r.get(field)}
    return next(iter(ids)) if len(ids) == 1 else None


def mlb_id_for_name(name):
    """MLBAM id for a name, or None if unknown/ambiguous. Fail-open."""
    return _unique_id(_rows_for_name(name), "mlb_id")


def espn_id_for_name(name):
    """ESPN athlete id for a name, or None if unknown/ambiguous. Fail-open."""
    return _unique_id(_rows_for_name(name), "espn_id")


def get_row(name):
    """Best full map row (dict) for a name — the single active match, or the sole
    match — or None when absent/ambiguous. Fail-open."""
    rows = _rows_for_name(name)
    if not rows:
        return None
    active = [r for r in rows if r.get("active")]
    pool = active or rows
    return pool[0] if len(pool) == 1 else None


def espn_id_for_mlb_id(mlb_id):
    """ESPN athlete id for a MLBAM id, or None. The MLBAM→ESPN bridge. Fail-open."""
    if not mlb_id:
        return None
    row = _player_idx()["by_mlb"].get(str(mlb_id))
    return row.get("espn_id") if row else None


def mlb_id_for_espn_id(espn_id):
    """MLBAM id for an ESPN athlete id, or None. The ESPN→MLBAM bridge that finally
    joins the ESPN gamelog tables to statcast_player_asof. Fail-open."""
    if not espn_id:
        return None
    row = _player_idx()["by_espn"].get(str(espn_id))
    return row.get("mlb_id") if row else None


def team_code_for_abbr(abbr):
    """Canonical SFBBTEAM code for any source's team abbreviation, or None."""
    if not abbr:
        return None
    return _team_idx()["by_abbr"].get(str(abbr).strip().upper())


def team_code_for_name(name):
    """Canonical SFBBTEAM code for a team name/abbr (odds feed or box score), or
    None. Tries, in order: exact abbr → full nickname/name_norm → last word
    (city+nickname → nickname). Fail-open; a miss lets the caller keep its own
    normalization + alias fallback."""
    if not name:
        return None
    idx = _team_idx()
    raw = str(name).strip()
    code = idx["by_abbr"].get(raw.upper())
    if code:
        return code
    norm = db_store.normalize_name(raw)
    if not norm:
        return None
    if norm in idx["by_name"]:
        return idx["by_name"][norm]
    last = norm.rsplit(" ", 1)[-1]
    return idx["by_name"].get(last)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _main_cli():
    from cli_encoding import configure_stdio
    configure_stdio()
    ap = argparse.ArgumentParser(
        description="Refresh the SFBB player/team ID cross-maps in Azure SQL.")
    ap.add_argument("--refresh", action="store_true",
                    help="Fetch both maps from SFBB and replace the SQL tables.")
    ap.add_argument("--players-only", action="store_true",
                    help="With --refresh, refresh only the player map.")
    ap.add_argument("--teams-only", action="store_true",
                    help="With --refresh, refresh only the team map.")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to write.")
    if not args.refresh:
        ap.error("nothing to do — pass --refresh")
    if not args.teams_only:
        print(f"player_id_map: wrote {refresh_players()} rows")
    if not args.players_only:
        print(f"team_id_map: wrote {refresh_teams()} rows")


if __name__ == "__main__":
    _main_cli()
