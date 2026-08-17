"""Durable per-(pitcher, game-date) as-of PITCHER feature store (the per-DATE curve
that statcast_asof.py deliberately does NOT hold).

WHY THIS EXISTS
---------------
The MLB expected-runs model prices team markets off a starter's run-prevention
signal. Today that signal is defined THREE incompatible ways across the pipeline
(the "scale trap"): the offline fitter uses xwOBAcon (~0.37, contact), live serves
full xwOBA (~0.31, est_woba), and the as-of backtest grades on plain ERA — so the
fitted weights are validated on one scale, served on another, graded on a third.

This table collapses all three onto ONE source: a materialized row per pitcher per
game-date holding the RAW as-of features (strictly game_date < as_of, leakage-safe).
The backtest, the fitter, and (eventually) live all read the SAME rows, so
fit == serve == grade by construction. It also removes the per-run in-memory load of
the full ~3M-row statcast_pitch corpus: each lookup becomes one indexed SELECT.

DESIGN
------
* RAW features only — NOT the fitted runs/9 ("xERA-lite") value. The fitted map
  g(features) -> runs/9 is applied IN CODE at read time, so re-fitting the model
  never requires rebuilding this table; only adding a genuinely new *feature* does.
* Grain: (entity_id, as_of_date, role). role='SP' -> entity_id = pitcher MLBAM id
  (one row per pitcher per game-date). role='RP' -> entity_id = team_id (team relief
  as-of aggregate; built in a follow-up — the SP rows are the core).
* v1 features (populated now, NO schema change): era/ip/avg_ip/k9/games from the
  warehouse mlb_pitcher_game facts (as-of), and xwOBAcon/n_bbe from statcast_pitch
  (as-of). The discipline/contact-quality columns (whiff/csw/barrel/hard_hit/gb) and
  the walk-rate columns (k_pct/bb_pct/n_bf) are in the schema as nullable so the v2
  build (after the mlb_pitcher_game BB/BF/HR unlock) needs NO second DDL.

POPULATION — NO CRON. There is no scheduled nightly job. Rows are filled two ways,
both writing the same table:
  * build_season(season): a one-shot BULK backfill (run offline to prime a whole
    season's as-of curve at once — the efficient path for the backtest range).
  * get_or_fill(entity_id, as_of_date): a lazy READ-THROUGH used by live + backtest
    stragglers — reads the row and, on a miss, computes that single (pitcher, date)
    row from the warehouse + statcast and persists it (write-once/idempotent) before
    returning. So the store self-maintains on demand as games are priced/graded.

Prod DDL is hand-run from SCHEMA_SQL below (mirrors savant_history / statcast_asof);
create_all() is TEST-ONLY (SQLite). Reuses db_store's engine + feature flag.
"""

import argparse
import bisect
import threading
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Float, Index, Integer, MetaData, String, Table, UniqueConstraint,
    delete, select,
)
from sqlalchemy.exc import OperationalError

import db_store

_META = MetaData()

# One row per (entity_id, as_of_date, role). Features are the cumulative as-of line
# through the day STRICTLY BEFORE as_of_date. All feature columns are nullable — a
# consumer that finds a NULL (or n below its stabilization floor) falls back to a
# prior / ERA, exactly as the on-the-fly path does today.
pitcher_asof_daily = Table(
    "pitcher_asof_daily", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("entity_id", String(32), nullable=False),   # MLBAM pitcher id (SP) | team_id (RP)
    Column("as_of_date", String(16), nullable=False),  # the game date; features are < this
    Column("role", String(4), nullable=False),         # "SP" | "RP"
    Column("season_bucket", Integer),                  # derived; indexed, not in key
    # --- warehouse mlb_pitcher_game as-of (v1, populated now) ---
    Column("era", Float),                              # earned runs / 9, as-of
    Column("ip", Float),                              # innings pitched, as-of (cumulative)
    Column("avg_ip", Float),                          # ip / games, as-of (starter workload)
    Column("k9", Float),                              # K / 9, as-of
    Column("games", Integer),                         # game count behind the line
    # --- statcast_pitch as-of contact quality (v1: xwobacon now; rest nullable) ---
    Column("xwobacon", Float),                        # mean xwOBAcon allowed (per BBE), as-of
    Column("n_bbe", Integer),                         # batted balls behind xwobacon
    Column("whiff_pct", Float),                       # whiffs / swings
    Column("csw_pct", Float),                         # (called + whiff) / pitches
    Column("barrel_pct", Float),                      # barrels / BBE
    Column("hard_hit_pct", Float),                    # LS>=95 / BBE
    Column("gb_pct", Float),                          # ground balls / BBE
    Column("n_pitches", Integer),                     # pitches behind whiff/csw
    # --- needs the mlb_pitcher_game BB/BF/HR unlock (v2, nullable until then) ---
    Column("k_pct", Float),                           # K / BF
    Column("bb_pct", Float),                          # BB / BF
    Column("n_bf", Integer),                          # batters faced behind k%/bb%
    Column("fetched_at", Float),
    UniqueConstraint("entity_id", "as_of_date", "role",
                     name="uq_pitcher_asof_daily"),
    Index("ix_pitcher_asof_key", "entity_id", "role", "as_of_date"),
    Index("ix_pitcher_asof_season", "season_bucket", "role"),
)

# Column-name SPEC for the schema-parity drift test (mirror statcast_asof style).
_COLS = ("id", "entity_id", "as_of_date", "role", "season_bucket",
         "era", "ip", "avg_ip", "k9", "games",
         "xwobacon", "n_bbe", "whiff_pct", "csw_pct", "barrel_pct",
         "hard_hit_pct", "gb_pct", "n_pitches",
         "k_pct", "bb_pct", "n_bf", "fetched_at")

# Hand-run in prod (Azure SQL). create_all() is TEST-ONLY.
SCHEMA_SQL = """
CREATE TABLE pitcher_asof_daily (
    id             INT IDENTITY(1,1) PRIMARY KEY,
    entity_id      VARCHAR(32)  NOT NULL,
    as_of_date     VARCHAR(16)  NOT NULL,
    role           VARCHAR(4)   NOT NULL,
    season_bucket  INT          NULL,
    era            FLOAT        NULL,
    ip             FLOAT        NULL,
    avg_ip         FLOAT        NULL,
    k9             FLOAT        NULL,
    games          INT          NULL,
    xwobacon       FLOAT        NULL,
    n_bbe          INT          NULL,
    whiff_pct      FLOAT        NULL,
    csw_pct        FLOAT        NULL,
    barrel_pct     FLOAT        NULL,
    hard_hit_pct   FLOAT        NULL,
    gb_pct         FLOAT        NULL,
    n_pitches      INT          NULL,
    k_pct          FLOAT        NULL,
    bb_pct         FLOAT        NULL,
    n_bf           INT          NULL,
    fetched_at     FLOAT        NULL,
    CONSTRAINT uq_pitcher_asof_daily UNIQUE (entity_id, as_of_date, role)
);
CREATE INDEX ix_pitcher_asof_key ON pitcher_asof_daily (entity_id, role, as_of_date);
CREATE INDEX ix_pitcher_asof_season ON pitcher_asof_daily (season_bucket, role);
"""

_WRITE_LOCK = threading.Lock()


def enabled():
    return db_store.enabled()


def create_all():
    """Create the table. TEST-ONLY (SQLite); prod DDL is hand-run from SCHEMA_SQL."""
    _META.create_all(db_store.get_engine())


def _now():
    return datetime.now(timezone.utc).timestamp()


# Every inserted row MUST carry the SAME keys: a bulk executemany compiles the
# INSERT from the first row's keys, so a later row missing e.g. 'era' raises
# "A value is required for bind parameter 'era'". A row with only statcast (no
# warehouse line) and a row with only a warehouse line (no BBE) otherwise differ.
_FEATURE_KEYS = ("era", "ip", "avg_ip", "k9", "games", "xwobacon", "n_bbe",
                 "whiff_pct", "csw_pct", "barrel_pct", "hard_hit_pct", "gb_pct",
                 "n_pitches", "k_pct", "bb_pct", "n_bf")


def _blank_row(entity_id, as_of_date, role, season, now):
    """A full-keyed row (all feature columns None) so every insert is homogeneous."""
    r = {k: None for k in _FEATURE_KEYS}
    r.update({"entity_id": str(entity_id), "as_of_date": str(as_of_date)[:10],
              "role": role, "season_bucket": season, "fetched_at": now})
    return r


# ──────────────────────────────────────────────────────────────────────────────
# As-of computation (pure; unit-tested)
# ──────────────────────────────────────────────────────────────────────────────

def _warehouse_asof_curve(games):
    """games = [(date10, outs, er, k), ...] for ONE pitcher (any order). Returns
    {date10: {era, ip, avg_ip, k9, games}} — the cumulative line over games STRICTLY
    before that date (matches mlb_warehouse.asof_pitcher_stats' `od < cutoff`). One
    entry per DISTINCT date; a same-date second appearance shares the pre-date line."""
    by_date = sorted(games, key=lambda g: g[0])
    out = {}
    cum_outs = cum_er = cum_k = 0.0
    cum_games = 0
    i, n = 0, len(by_date)
    for date10 in sorted({g[0] for g in by_date}):
        # `out[date10]` = everything strictly before date10 (cum_* has absorbed all
        # earlier dates but not this one yet).
        if cum_outs > 0 and cum_games > 0:
            ip = cum_outs / 3.0
            out[date10] = {
                "era": (cum_er / ip) * 9.0, "ip": ip,
                "avg_ip": ip / cum_games, "k9": (cum_k / ip) * 9.0,
                "games": cum_games,
            }
        else:
            out[date10] = None   # no prior in-season games -> caller falls back
        while i < n and by_date[i][0] == date10:      # now absorb this date
            cum_outs += by_date[i][1]
            cum_er += by_date[i][2]
            cum_k += by_date[i][3]
            cum_games += 1
            i += 1
    return out


class _XwobaconAsOf:
    """Per-pitcher prefix-sum of xwOBAcon over batted balls, for O(log n) as-of
    means (strictly game_date < as_of). Built once from statcast_pitch rows."""

    def __init__(self, rows):
        # pid -> (sorted dates, prefix_sum_of_xwoba, count) parallel arrays.
        acc = {}
        for r in rows:
            x = r.get("xwoba")
            pid, d = r.get("pitcher"), r.get("game_date")
            if x is None or not pid or not d:
                continue
            acc.setdefault(str(pid), []).append((d, x))
        self._idx = {}
        for pid, pairs in acc.items():
            pairs.sort(key=lambda p: p[0])
            dates = [p[0] for p in pairs]
            psum, s = [], 0.0
            for _, x in pairs:
                s += x
                psum.append(s)
            self._idx[pid] = (dates, psum)

    def asof(self, pid, as_of):
        """(mean_xwobacon, n_bbe) over BBE strictly before as_of, or (None, n)."""
        entry = self._idx.get(str(pid))
        if not entry:
            return None, 0
        dates, psum = entry
        i = bisect.bisect_left(dates, as_of)   # count of dates strictly < as_of
        if i <= 0:
            return None, 0
        return psum[i - 1] / i, i


# ──────────────────────────────────────────────────────────────────────────────
# Build (offline) + read
# ──────────────────────────────────────────────────────────────────────────────

def build_season(season, verbose=True):
    """Materialize role='SP' as-of rows for every (pitcher, game-date) in `season`.
    Replace-writes the season's SP rows (idempotent rebuild). Returns n rows written.
    Reads mlb_warehouse (warehouse as-of line) + savant_history (xwOBAcon). SQL-only."""
    if not enabled():
        raise RuntimeError("SQL is not configured (SQL_* secrets) — cannot build.")
    import mlb_warehouse
    import savant_history as sh

    season = int(season)
    pit_idx = mlb_warehouse._pitcher_game_index(season)   # {aid: [(date,outs,er,k)]}
    if not pit_idx:
        if verbose:
            print(f"  [build] no mlb_pitcher_game rows for {season}; nothing to build.")
        return 0
    xw = _XwobaconAsOf(sh.load_days(f"{season}-01-01", f"{season}-12-31"))

    now = _now()
    rows = []
    for aid, games in pit_idx.items():
        curve = _warehouse_asof_curve(games)
        for date10, wh in curve.items():
            xwobacon, n_bbe = xw.asof(aid, date10)
            if wh is None and xwobacon is None:
                continue   # nothing as-of -> no row (consumer would fall back anyway)
            row = _blank_row(aid, date10, "SP", season, now)
            row["xwobacon"] = xwobacon
            row["n_bbe"] = n_bbe or None
            if wh:
                row.update(wh)          # era/ip/avg_ip/k9/games
            rows.append(row)

    engine = db_store.get_engine()
    with _WRITE_LOCK:
        with engine.begin() as conn:
            conn.execute(delete(pitcher_asof_daily).where(
                (pitcher_asof_daily.c.season_bucket == season)
                & (pitcher_asof_daily.c.role == "SP")))
            for i in range(0, len(rows), 500):        # chunked insert
                conn.execute(pitcher_asof_daily.insert(), rows[i:i + 500])
    if verbose:
        print(f"  [build] season {season}: wrote {len(rows)} SP as-of rows "
              f"({len(pit_idx)} pitchers).")
    return len(rows)


def asof_pitcher_features(entity_id, as_of_date, role="SP", max_retries=3):
    """The materialized as-of feature row for (entity_id, as_of_date, role), or None.
    Single indexed SELECT; fail-open (None) so a miss never blocks a caller."""
    if not enabled() or not entity_id or not as_of_date:
        return None
    stmt = (select(pitcher_asof_daily)
            .where((pitcher_asof_daily.c.entity_id == str(entity_id))
                   & (pitcher_asof_daily.c.as_of_date == str(as_of_date)[:10])
                   & (pitcher_asof_daily.c.role == role)))
    for attempt in range(max_retries):
        try:
            with db_store.get_engine().connect() as conn:
                r = conn.execute(stmt).mappings().first()
            return dict(r) if r else None
        except OperationalError:
            if attempt == max_retries - 1:
                return None
    return None


def _asof_xwobacon_sql(pitcher_id, as_of, season_start):
    """Single-pitcher as-of xwOBAcon via one SQL aggregation over statcast_pitch
    (in-season, strictly game_date < as_of). Returns (mean, n_bbe) or (None, 0). For
    the on-demand fill path (one row); the bulk build uses the in-memory prefix-sum.
    Same window/filter as the build so both produce identical values."""
    try:
        import savant_history as sh
        from sqlalchemy import func
        sp = sh.statcast_pitch
        with db_store.get_engine().connect() as conn:
            row = conn.execute(
                select(func.avg(sp.c.xwoba), func.count(sp.c.xwoba)).where(
                    (sp.c.pitcher == str(pitcher_id))
                    & (sp.c.xwoba.isnot(None))
                    & (sp.c.game_date >= season_start)
                    & (sp.c.game_date < str(as_of)[:10]))).first()
        if not row or not row[1]:
            return None, 0
        return float(row[0]), int(row[1])
    except (OperationalError, ValueError, TypeError):
        return None, 0


def _compute_asof_row(entity_id, as_of_date, role):
    """Build ONE as-of feature row dict for (entity_id, as_of_date, role), or None if
    there is nothing as-of. Uses the SAME definitions as build_season (warehouse
    asof_pitcher_stats + in-season xwOBAcon), so on-demand rows match bulk rows."""
    if role != "SP":
        return None                    # RP (team relief) fill is a follow-up
    import mlb_warehouse
    d10 = str(as_of_date)[:10]
    season_start = f"{d10[:4]}-01-01"
    wh = mlb_warehouse.asof_pitcher_stats(entity_id, d10)   # era/ip/k/games/avg_ip
    xwobacon, n_bbe = _asof_xwobacon_sql(entity_id, d10, season_start)
    if wh is None and xwobacon is None:
        return None
    row = _blank_row(entity_id, d10, "SP", int(d10[:4]), _now())
    row["xwobacon"] = xwobacon
    row["n_bbe"] = n_bbe or None
    if wh:
        row.update({"era": wh.get("era"), "ip": wh.get("ip"),
                    "avg_ip": wh.get("avg_ip"), "games": wh.get("games"),
                    "k9": ((wh["k"] / wh["ip"]) * 9.0
                           if wh.get("ip") else None)})
    return row


def get_or_fill(entity_id, as_of_date, role="SP"):
    """Read-through: the as-of feature row, computing + persisting it on a miss.
    The lazy self-fill path (no cron) for live pricing + backtest stragglers.
    Write-once/idempotent (uq_pitcher_asof_daily): a concurrent writer just re-reads.
    Fail-open: returns None (or the computed dict even if the persist fails)."""
    hit = asof_pitcher_features(entity_id, as_of_date, role)
    if hit is not None:
        return hit
    if not enabled():
        return None
    row = _compute_asof_row(entity_id, as_of_date, role)
    if row is None:
        return None
    try:
        with db_store.get_engine().begin() as conn:
            conn.execute(pitcher_asof_daily.insert(), [row])
    except Exception:
        # Lost a write race (uq) or transient error — prefer the persisted row.
        persisted = asof_pitcher_features(entity_id, as_of_date, role)
        if persisted is not None:
            return persisted
    return row


def _main_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true", help="Materialize as-of rows.")
    p.add_argument("--season", type=int, action="append",
                   help="Season(s) to build (repeatable). Default: 2023..2026.")
    p.add_argument("--ddl", action="store_true", help="Print the prod DDL and exit.")
    args = p.parse_args()

    if args.ddl:
        print(SCHEMA_SQL)
        return 0
    try:
        db_store.promote_secrets_from_toml()
    except Exception:
        pass
    if not enabled():
        print("SQL not enabled (SQL_* secrets) — aborting.")
        return 2
    if args.build:
        seasons = args.season or [2023, 2024, 2025, 2026]
        total = 0
        for s in seasons:
            total += build_season(s)
        print(f"Done. {total} as-of rows across seasons {seasons}.")
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    raise SystemExit(_main_cli())
