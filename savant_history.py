"""
Leakage-safe historical Statcast layer for backtesting (as-of-date features).

The live client (mlb_starters.py) uses SEASON-AGGREGATE Savant x-stats, which is
correct for *today* but would leak future information into a backtest of a past
game. This module instead caches per-day Statcast pitch data and computes
"as-of" aggregates from ONLY the pitches thrown BEFORE the game being graded.

Measure used: mean ``estimated_woba_using_speedangle`` over batted-ball events
(xwOBAcon — contact quality). Lower = better pitcher / weaker offense.

Historical pitch data never changes, so day caches are permanent. Full-season
pulls are slow (Savant-rate-limited) but free; run once and reuse.
"""

import csv
import io
import json
import os
import random
import threading
import time
from datetime import date as _date, datetime, timedelta, timezone

import requests

from sqlalchemy import (
    Column, Float, Index, Integer, MetaData, String, Table, delete, insert,
    select,
)
from sqlalchemy.exc import OperationalError

import db_store

SAVANT_BASE = "https://baseballsavant.mlb.com"
SAVANT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsbookValueFinder/1.0)"}
# Bump SCHEMA_VERSION whenever the trimmed row shape changes. v4 stored per-at-bat
# xBA (+ xwOBA). v5 (P2.4b) ADDS the plate-discipline / contact-quality raw columns
# needed for whiff%/CSW%/hard-hit%/barrel% (description, type, launch_speed,
# launch_speed_angle, launch_angle, bb_type). Bumping the version rolls CACHE_DIR to
# a fresh dir → a re-fetch (the v4 cache lacks these columns; it is not reused).
SCHEMA_VERSION = 5
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cache", f"statcast_days_v{SCHEMA_VERSION}")
LEGACY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cache", "statcast_days_v2")

# Minimum prior events before an as-of estimate is usable (batted balls for
# xwOBAcon, official at-bats for xBA).
MIN_BBE = 40
# Minimum prior PITCHES before an as-of plate-discipline rate (whiff/CSW) is usable.
MIN_PITCHES = 100


# ── Durable SQL home for the raw pitch rows (replaces the local day-file cache) ──
# statcast_pitch holds one row per pitch (trimmed v5 shape); statcast_day is the
# ingest manifest so an INGESTED-but-empty offseason day is distinguishable from an
# UNFETCHED one. SQL is the single source of truth: load_days reads it, ingest_day
# writes it (day-atomic), ensure_days gap-fills. The legacy local day files remain
# readable ONLY by the one-shot migrate_files_to_sql. create_all is TEST-ONLY
# (prod DDL is hand-run from sql/schema.sql), mirroring statcast_asof.py.
_META = MetaData()

statcast_pitch = Table(
    "statcast_pitch", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("game_date", String(16), nullable=False),
    Column("pitcher", String(16)),
    Column("batter", String(16)),
    Column("p_throws", String(2)),
    Column("batting_team", String(8)),
    Column("stand", String(2)),
    Column("xwoba", Float),
    Column("xba", Float),
    Column("description", String(40)),
    Column("type", String(2)),
    Column("launch_speed", Float),
    Column("launch_speed_angle", Integer),
    Column("launch_angle", Float),
    Column("bb_type", String(20)),
    Index("ix_statcast_pitch_date", "game_date"),
    # Covering index for the warehouse team-offense aggregate (mlb_starters.
    # _warehouse_team_factors): GROUP BY batting_team, p_throws over a game_date range,
    # AVG/COUNT(xwoba). Group keys first (stream aggregate, no sort), game_date last
    # (range seek per group), xwoba covered (index-only). mssql_include is ignored off
    # SQL Server (SQLite tests just build the 3-col index).
    Index("ix_statcast_pitch_offense", "batting_team", "p_throws", "game_date",
          mssql_include=["xwoba"]),
)

statcast_day = Table(
    "statcast_day", _META,
    Column("game_date", String(16), primary_key=True),
    Column("n_rows", Integer, nullable=False),
    Column("fetched_at", String(32)),
)

# The trimmed per-pitch dict shape == the statcast_pitch DATA columns (order-stable).
# fetch_statcast_day emits these keys; load_days returns them; the schema-parity test
# asserts the SQL columns match. Keep in sync with sql/schema.sql.
PITCH_COLS = ("game_date", "pitcher", "batter", "p_throws", "batting_team",
              "stand", "xwoba", "xba", "description", "type", "launch_speed",
              "launch_speed_angle", "launch_angle", "bb_type")

_WRITE_LOCK = threading.Lock()


def enabled():
    return db_store.enabled()


def create_all():
    """Create the statcast SQL tables. TEST-ONLY (SQLite); prod DDL is hand-run
    from sql/schema.sql."""
    _META.create_all(db_store.get_engine())


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _pf(v):
    """Parse a Statcast CSV float cell, else None."""
    try:
        return float(v) if v not in ("", "null", None) else None
    except (TypeError, ValueError):
        return None


def _pi(v):
    """Parse a Statcast CSV int cell (e.g. launch_speed_angle 1..6), else None."""
    try:
        return int(float(v)) if v not in ("", "null", None) else None
    except (TypeError, ValueError):
        return None


# `description` (Statcast pitch outcome) sets for plate-discipline rates (v5).
_SWING_DESCRIPTIONS = frozenset({
    "hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
})
_WHIFF_DESCRIPTIONS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip",
})
HARD_HIT_MPH = 95.0     # hard-hit ball threshold
BARREL_LSA = 6          # launch_speed_angle == 6 → Savant "Barrel"


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _day_path(day):
    return os.path.join(CACHE_DIR, f"{day}.json")


def _at_bat_xba(row):
    """Return expected hits for an official at-bat, else None."""
    event = row.get("events")
    if event in ("strikeout", "strikeout_double_play"):
        return 0.0
    if event in ("walk", "intent_walk", "hit_by_pitch", "sac_fly",
                 "sac_bunt", "catcher_interf"):
        return None
    value = row.get("estimated_ba_using_speedangle")
    try:
        return float(value) if value not in ("", "null", None) else None
    except (TypeError, ValueError):
        return None


def fetch_statcast_day(day, force=False):
    """
    Fetch all pitches for a single ``day`` (YYYY-MM-DD) from Baseball Savant, trim
    to the v5 shape, INGEST to SQL (statcast_pitch + manifest, day-atomic), and
    return the trimmed rows:
        {game_date, pitcher, batter, p_throws, batting_team, stand, xwoba, xba,
         description, type, launch_speed, launch_speed_angle, launch_angle, bb_type}
    xwOBA is populated for batted balls; xBA once per official at-bat (0.0 for
    strikeouts) else None; the plate-discipline/contact fields drive whiff%/CSW%/
    hard-hit%/barrel% via the asof_rates aggregator.

    SQL is the single store — there is NO file cache write (see migrate_files_to_sql
    for the one-shot import of the legacy day files). ``force`` is retained for
    signature compatibility and no longer gates a cache read (dedup now lives in
    ensure_days / missing_days via the manifest).
    """
    url = f"{SAVANT_BASE}/statcast_search/csv"
    params = {"all": "true", "type": "details",
              "game_date_gt": day, "game_date_lt": day}
    resp = None
    for attempt in range(5):
        resp = requests.get(url, params=params, headers=SAVANT_HEADERS, timeout=90)
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            time.sleep(1.5 ** attempt + random.uniform(0, 0.5))
            continue
        break
    resp.raise_for_status()
    rows = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))

    trimmed = []
    for r in rows:
        xw = r.get("estimated_woba_using_speedangle")
        try:
            xw = float(xw) if xw not in ("", "null", None) else None
        except ValueError:
            xw = None
        xb = _at_bat_xba(r)
        topbot = r.get("inning_topbot")
        # The batting team is the AWAY team in the top half, HOME in the bottom.
        batting_team = r.get("away_team") if topbot == "Top" else r.get("home_team")
        trimmed.append({
            "game_date": r.get("game_date"),
            "pitcher": r.get("pitcher"),
            "batter": r.get("batter"),
            "p_throws": r.get("p_throws"),
            "batting_team": batting_team,
            "stand": r.get("stand"),
            "xwoba": xw,
            "xba": xb,
            # v5 (P2.4b): plate-discipline / contact-quality raw fields.
            "description": r.get("description"),
            "type": r.get("type"),                       # S/B/X (X = batted ball)
            "launch_speed": _pf(r.get("launch_speed")),
            "launch_speed_angle": _pi(r.get("launch_speed_angle")),
            "launch_angle": _pf(r.get("launch_angle")),
            "bb_type": r.get("bb_type"),
        })
    ingest_day(day, trimmed)
    return trimmed


def ingest_day(day, rows):
    """Replace all statcast_pitch rows for ``day`` with ``rows`` (day-atomic,
    idempotent) and upsert the statcast_day manifest. Returns the row count, or
    False when SQL is off (nothing persisted — SQL is the single store).

    Mirrors statcast_asof.put_rates' write discipline: one transaction under
    _WRITE_LOCK with a bounded OperationalError retry. Because a trimmed pitch row
    has no stable natural key, idempotency is at the DAY grain — DELETE the day then
    bulk-INSERT — exactly the per-day-atomic semantics the old file write had. An
    EMPTY day still writes a manifest row (n_rows=0) so it is not re-fetched."""
    if not enabled():
        return False
    params = [{c: r.get(c) for c in PITCH_COLS} for r in (rows or [])]
    for p in params:                     # a single-day pull is all `day`; be explicit
        p["game_date"] = day
    engine = db_store.get_engine()
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    conn.execute(delete(statcast_pitch).where(
                        statcast_pitch.c.game_date == day))
                    if params:
                        conn.execute(insert(statcast_pitch), params)
                    conn.execute(delete(statcast_day).where(
                        statcast_day.c.game_date == day))
                    conn.execute(insert(statcast_day), [{
                        "game_date": day, "n_rows": len(params),
                        "fetched_at": _now_iso()}])
                return len(params)
            except OperationalError:
                if attempt == 2:
                    raise
    return len(params)


def load_days(start, end):
    """All statcast_pitch rows in [start, end] (inclusive, YYYY-MM-DD) as trimmed
    dicts — the SAME shape fetch_statcast_day emits, so every downstream aggregator
    is unchanged. SQL-ONLY: returns [] when SQL is off (local/tests patch this) or
    the range has not been ingested. Lexicographic date compare is correct for the
    zero-padded YYYY-MM-DD strings."""
    if not enabled():
        return []
    engine = db_store.get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            select(*[statcast_pitch.c[c] for c in PITCH_COLS])
            .where((statcast_pitch.c.game_date >= start)
                   & (statcast_pitch.c.game_date <= end))
            .order_by(statcast_pitch.c.game_date))
        return [dict(zip(PITCH_COLS, row)) for row in result]


def ingested_days(start, end):
    """Set of game_dates in [start, end] present in the manifest (INCLUDING empty
    days). Empty set when SQL is off."""
    if not enabled():
        return set()
    engine = db_store.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(statcast_day.c.game_date).where(
                (statcast_day.c.game_date >= start)
                & (statcast_day.c.game_date <= end)))
        return {r[0] for r in rows}


def missing_days(start, end):
    """Calendar days in [start, min(end, today)] NOT yet ingested (per the manifest),
    oldest-first. ``end`` is capped at today — future days have no games."""
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    today = _date.today()
    if d1 > today:
        d1 = today
    have = ingested_days(start, d1.isoformat())
    out = []
    day = d0
    while day <= d1:
        ds = day.isoformat()
        if ds not in have:
            out.append(ds)
        day += timedelta(days=1)
    return out


def ensure_days(start, end, cap=None, sleep=0.5, verbose=True):
    """Fetch+ingest any missing days in [start, end]; the incremental gap-fill.

    ``cap`` bounds how many days a SINGLE call will pull (None = unbounded, for the
    explicit one-time bulk backfill). When more than ``cap`` days are missing, the
    most-RECENT ``cap`` are fetched (the fresh gap since the last run) and the rest
    are left for the operator's bulk backfill. Returns (n_fetched, n_missing_total)
    so the caller can warn when n_fetched < n_missing_total. Never raises for a
    single day's fetch error (logs + continues); requires SQL (no-op (0, 0) off)."""
    if not enabled():
        return (0, 0)
    miss = missing_days(start, end)
    n_missing = len(miss)
    to_fetch = miss[-cap:] if (cap is not None and n_missing > cap) else miss
    n = 0
    for ds in to_fetch:
        try:
            fetch_statcast_day(ds)          # fetches + ingests (manifest updated)
            n += 1
            if verbose:
                print(f"  [statcast] ingested {ds}")
            time.sleep(sleep)               # be polite to Savant
        except Exception as exc:
            if verbose:
                print(f"  [statcast] fetch failed {ds}: {type(exc).__name__}: {exc}")
    if verbose:
        print(f"ensure_days done: {n} day(s) ingested "
              f"({n_missing} missing in {start}..{end}).")
    return (n, n_missing)


def fetch_range(start, end, sleep=0.5, verbose=True):
    """Back-compat wrapper (backtest_starters --fetch / statcast_asof build --fetch):
    ensure every day in [start, end] is ingested to SQL. Uncapped — this is the
    explicit bulk path."""
    ensure_days(start, end, cap=None, sleep=sleep, verbose=verbose)


def _load_day_file(day):
    """Read a legacy local day file (v5, then v2) for ``day`` → its trimmed rows, or
    None when neither file exists. Used ONLY by migrate_files_to_sql — load_days is
    SQL-only. An empty file legitimately returns []."""
    p = _day_path(day)
    if not os.path.exists(p):
        legacy = os.path.join(LEGACY_CACHE_DIR, f"{day}.json")
        p = legacy if os.path.exists(legacy) else p
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def migrate_files_to_sql(start, end, verbose=True):
    """One-shot: ingest the EXISTING local day files in [start, end] into SQL (no
    download), so the durable store is seeded without re-hitting Savant. A day with
    a file (even empty) is ingested + manifested; a day with no file is skipped (it
    stays 'missing' for ensure_days). Returns (n_days, n_rows)."""
    if not enabled():
        raise RuntimeError("SQL is not configured (SQL_* secrets) — nothing to write.")
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    day = d0
    n_days = n_rows = 0
    while day <= d1:
        ds = day.isoformat()
        rows = _load_day_file(ds)
        if rows is not None:
            ingest_day(ds, rows)
            n_days += 1
            n_rows += len(rows)
            if verbose and n_days % 30 == 0:
                print(f"  migrated through {ds} ({n_days} days, {n_rows:,} rows)")
        day += timedelta(days=1)
    if verbose:
        print(f"migrate_files_to_sql done: {n_days} day file(s) → {n_rows:,} rows "
              f"ingested for {start}..{end}.")
    return (n_days, n_rows)


def asof_pitcher_xwoba(pitcher_id, rows, as_of, min_bbe=MIN_BBE):
    """
    Mean xwOBAcon a pitcher ALLOWED using only batted balls before ``as_of``.
    Returns (xwoba, n_bbe) or (None, n) if under the sample threshold.
    """
    pid = str(pitcher_id)
    vals = [r["xwoba"] for r in rows
            if r["pitcher"] == pid and r["xwoba"] is not None
            and r["game_date"] and r["game_date"] < as_of]
    if len(vals) < min_bbe:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def asof_team_xwoba_vs_hand(team_abbr, hand, rows, as_of, min_bbe=MIN_BBE):
    """
    Mean xwOBAcon a team's hitters PRODUCED vs pitchers of a given hand
    ('L'/'R'), using only batted balls before ``as_of``. For batter-side
    (Phase 2) features. Returns (xwoba, n_bbe) or (None, n).
    """
    vals = [r["xwoba"] for r in rows
            if r["batting_team"] == team_abbr and r["p_throws"] == hand
            and r["xwoba"] is not None
            and r["game_date"] and r["game_date"] < as_of]
    if len(vals) < min_bbe:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def asof_batter_xwoba_vs_hand(batter_id, hand, rows, as_of, min_bbe=MIN_BBE):
    """
    Mean xwOBAcon an individual batter PRODUCED vs pitchers of a given hand
    ('L'/'R'), using only batted balls before ``as_of``. For individual batter
    props (batter_hits, etc.). Returns (xwoba, n_bbe) or (None, n).
    """
    bid = str(batter_id)
    vals = [r["xwoba"] for r in rows
            if r.get("batter") == bid and r["p_throws"] == hand
            and r["xwoba"] is not None
            and r["game_date"] and r["game_date"] < as_of]
    if len(vals) < min_bbe:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def batter_asof_rates(rows, as_of, min_ab=MIN_BBE):
    """Per-batter as-of expected-BA (and xwOBAcon) over pitches before ``as_of``.

    Bulk aggregation for the OFFLINE derived-table build (roadmap 2.4a): one
    ``as_of`` (typically today) across every batter, rather than the per-game
    as-of the backtest needs. xBA is the mean at-bat ``xba`` over official ABs
    (batted balls + strikeouts, excluding walks/HBP/sacrifices — see
    ``_at_bat_xba``); xwOBAcon is the mean over batted balls. Leakage-safe by the
    same strict ``game_date < as_of`` filter as the ``asof_*`` primitives.

    Returns ``{batter_id: {"xba", "n_ab", "xwoba", "n_bbe"}}`` for batters with at
    least ``min_ab`` official ABs (xwoba is None when under ``min_ab`` batted
    balls). ``batter_id`` is the MLBAM id string (matches ``find_player_id``).
    """
    xba_agg = {}   # bid -> [sum_xba, n_ab]
    xw_agg = {}    # bid -> [sum_xwoba, n_bbe]
    for r in rows:
        gd = r.get("game_date")
        if not gd or gd >= as_of:
            continue
        bid = r.get("batter")
        if not bid:
            continue
        xb = r.get("xba")
        if xb is not None:
            a = xba_agg.setdefault(bid, [0.0, 0])
            a[0] += xb
            a[1] += 1
        xw = r.get("xwoba")
        if xw is not None:
            a = xw_agg.setdefault(bid, [0.0, 0])
            a[0] += xw
            a[1] += 1
    out = {}
    for bid, (s, n) in xba_agg.items():
        if n < min_ab:
            continue
        xw = xw_agg.get(bid)
        out[bid] = {
            "xba": s / n,
            "n_ab": n,
            "xwoba": (xw[0] / xw[1]) if (xw and xw[1] > 0) else None,
            "n_bbe": xw[1] if xw else 0,
        }
    return out


def _empty_rate_acc():
    return {"n_pitches": 0, "n_swings": 0, "n_whiff": 0, "n_called": 0,
            "n_bip": 0, "n_hardhit": 0, "n_barrel": 0}


def _accumulate_rates(acc, r):
    """Fold one v5 pitch row into a counts accumulator."""
    acc["n_pitches"] += 1
    desc = r.get("description")
    if desc in _SWING_DESCRIPTIONS:
        acc["n_swings"] += 1
    if desc in _WHIFF_DESCRIPTIONS:
        acc["n_whiff"] += 1
    if desc == "called_strike":
        acc["n_called"] += 1
    if r.get("type") == "X":                       # batted ball
        acc["n_bip"] += 1
        ls = r.get("launch_speed")
        if ls is not None and ls >= HARD_HIT_MPH:
            acc["n_hardhit"] += 1
        if r.get("launch_speed_angle") == BARREL_LSA:
            acc["n_barrel"] += 1


def _finalize_rates(acc, min_pitches):
    """Counts → rates; None if under the pitch threshold. Per-rate values are
    None when their own denominator is empty (e.g. no batted balls yet)."""
    if acc["n_pitches"] < min_pitches:
        return None

    def _ratio(num, den):
        return (num / den) if den else None

    return {
        "whiff_pct": _ratio(acc["n_whiff"], acc["n_swings"]),
        "csw_pct": _ratio(acc["n_called"] + acc["n_whiff"], acc["n_pitches"]),
        "hard_hit_pct": _ratio(acc["n_hardhit"], acc["n_bip"]),
        "barrel_pct": _ratio(acc["n_barrel"], acc["n_bip"]),
        "n_pitches": acc["n_pitches"],
        "n_bip": acc["n_bip"],
    }


def asof_rates(rows, as_of, key, min_pitches=MIN_PITCHES):
    """Per-player as-of plate-discipline / contact rates (v5), leakage-safe.

    ``key`` = "pitcher" → rates the pitcher INDUCES (whiff/CSW → strikeout props);
    "batter" → rates the batter ALLOWS (whiff/CSW) + PRODUCES (hard-hit/barrel).
    Returns ``{player_id: {whiff_pct, csw_pct, hard_hit_pct, barrel_pct,
    n_pitches, n_bip}}`` for players with >= ``min_pitches`` pitches strictly
    before ``as_of``. Mirrors ``batter_asof_rates`` but accumulates count pairs
    (numerator/denominator) per rate rather than averaging a single value.
    """
    accs = {}
    for r in rows:
        gd = r.get("game_date")
        if not gd or gd >= as_of:
            continue
        pid = r.get(key)
        if not pid:
            continue
        acc = accs.get(pid)
        if acc is None:
            acc = accs[pid] = _empty_rate_acc()
        _accumulate_rates(acc, r)
    out = {}
    for pid, acc in accs.items():
        fin = _finalize_rates(acc, min_pitches)
        if fin is not None:
            out[pid] = fin
    return out


def _main_cli():
    import argparse
    from cli_encoding import configure_stdio
    configure_stdio()
    ap = argparse.ArgumentParser(
        description="Statcast raw-pitch SQL store: migrate legacy day files + "
                    "bulk/incremental gap-fill from Baseball Savant.")
    ap.add_argument("--migrate-to-sql", dest="migrate", action="store_true",
                    help="Ingest EXISTING local day files into SQL (no download).")
    ap.add_argument("--ensure", action="store_true",
                    help="Fetch+ingest missing days from Savant (bulk when uncapped).")
    ap.add_argument("--season",
                    help="Season year(s), comma-separated (e.g. 2023,2024,2025,2026); "
                         "expands each to Mar 1..Nov 30.")
    ap.add_argument("--start", help="YYYY-MM-DD (overrides --season).")
    ap.add_argument("--end", help="YYYY-MM-DD (overrides --season).")
    ap.add_argument("--cap", type=int, default=None,
                    help="With --ensure, max days to pull this run (default: all).")
    ap.add_argument("--smoke", action="store_true",
                    help="Two-day fetch + as-of smoke test (requires SQL).")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to do.")

    ranges = []
    if args.start and args.end:
        ranges = [(args.start, args.end)]
    elif args.season:
        for y in (s.strip() for s in args.season.split(",") if s.strip()):
            ranges.append((f"{y}-03-01", f"{y}-11-30"))

    if args.migrate:
        if not ranges:
            ap.error("--migrate-to-sql needs --season or --start/--end")
        for s, e in ranges:
            migrate_files_to_sql(s, e)
    elif args.ensure:
        if not ranges:
            ap.error("--ensure needs --season or --start/--end")
        for s, e in ranges:
            ensure_days(s, e, cap=args.cap)
    elif args.smoke:
        # Two days, then an as-of estimate using only the earlier day.
        for d in ("2024-07-03", "2024-07-04"):
            rows = fetch_statcast_day(d)
            print(f"{d}: {len(rows)} pitches, "
                  f"{sum(1 for r in rows if r['xwoba'] is not None)} BBE")
        rows = load_days("2024-07-03", "2024-07-04")
        # Justin Steele pitched around this window; evaluate as-of 2024-07-04.
        xw, n = asof_pitcher_xwoba(657006, rows, "2024-07-04", min_bbe=1)
        print(f"Steele as-of 2024-07-04 xwOBAcon={xw and round(xw,3)} over {n} BBE")
    else:
        ap.error("nothing to do — pass --migrate-to-sql, --ensure, or --smoke")


if __name__ == "__main__":
    _main_cli()
