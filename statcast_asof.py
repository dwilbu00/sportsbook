"""Durable per-batter Statcast as-of rate table (Blob->SQL, roadmap 2.4a).

The raw pitch-level Statcast cache (``savant_history`` -> ``cache/statcast_days_v4/``,
~326 MB, gitignored) is ephemeral on Streamlit Cloud, so the live app can't read
it. This module is the small, durable **derived** table the live projection reads:
one row per batter holding season-to-date expected BA (xBA) + xwOBAcon, computed
OFFLINE from the raw cache and written to Azure SQL.

* **Offline build** (``python statcast_asof.py --build --season 2026``): loads the
  raw days, aggregates each batter's as-of xBA via
  ``savant_history.batter_asof_rates``, and replace-writes the season's rows.
* **Live read** (``get_batter_xba``): a single indexed SQL SELECT; fails open to
  ``(None, 0)`` so a miss never blocks a recommendation. props.py shrinks the
  batter_hits projection toward ``xBA x AB/game`` when a row exists (n >= 40).

Reuses ``db_store``'s engine / feature flag (SQL is the single store — no fallback;
the app is cut over). Own ``MetaData`` + ``create_all()`` (TEST-ONLY SQLite; prod
DDL is hand-run from ``sql/schema.sql``). The full per-DATE as-of curve needed for
backtesting is NOT stored here — the backtest reads ``savant_history`` directly
offline.
"""

import argparse
import threading

from sqlalchemy import (
    Column, Float, Index, Integer, MetaData, String, Table, UniqueConstraint,
    select,
)
from sqlalchemy.exc import OperationalError

import db_store

_META = MetaData()

# One row per (player_id, season_bucket, split, role). role = "bat" (rates the
# batter produces/allows → hits/contact) or "pit" (rates the pitcher induces →
# strikeout props); the two share the MLBAM id space, so role joins the key.
# split = "all" for v1 (per-hand "vsL"/"vsR" is a future refinement). Column names
# are plain (no reserved words), mirroring sql/schema.sql exactly.
statcast_player_asof = Table(
    "statcast_player_asof", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", String(32), nullable=False),     # MLBAM id (find_player_id)
    Column("season_bucket", Integer, nullable=False),    # season year, e.g. 2026
    Column("split", String(16), nullable=False),         # "all" | "vsL" | "vsR"
    Column("role", String(4), nullable=False),           # "bat" | "pit"
    Column("as_of_date", String(16)),                    # YYYY-MM-DD the build ran
    Column("xba", Float),                                 # expected BA (per AB) — bat
    Column("xwoba", Float),                               # xwOBAcon (per batted ball)
    Column("n_ab", Integer),                             # official ABs behind xba
    Column("n_bbe", Integer),                            # batted balls behind xwoba
    Column("whiff_pct", Float),                           # whiffs / swings
    Column("csw_pct", Float),                             # (called + whiff) / pitches
    Column("hard_hit_pct", Float),                        # LS>=95 / batted balls
    Column("barrel_pct", Float),                          # barrels / batted balls
    Column("n_pitches", Integer),                         # pitches behind whiff/csw
    Column("n_bip", Integer),                             # batted balls behind hh/brl
    UniqueConstraint("player_id", "season_bucket", "split", "role",
                     name="uq_statcast_player_asof"),
    Index("ix_statcast_player_asof_key", "season_bucket", "split", "role"),
    Index("ix_statcast_player_asof_player", "player_id", "season_bucket", "role"),
)

# Column-name SPEC for the schema-parity drift test (mirror test_db_store style).
_COLS = ("id", "player_id", "season_bucket", "split", "role", "as_of_date",
         "xba", "xwoba", "n_ab", "n_bbe",
         "whiff_pct", "csw_pct", "hard_hit_pct", "barrel_pct",
         "n_pitches", "n_bip")

_WRITE_LOCK = threading.Lock()


def enabled():
    return db_store.enabled()


def create_all():
    """Create the table. TEST-ONLY (SQLite); prod DDL is hand-run."""
    _META.create_all(db_store.get_engine())


# ── live read ──────────────────────────────────────────────────────────────
def get_rates(player_id, season, role, split="all"):
    """Full as-of rate row (dict) for a player, or None. Fails open on anything
    (SQL off / no row / error). Keys: xba, xwoba, n_ab, n_bbe, whiff_pct, csw_pct,
    hard_hit_pct, barrel_pct, n_pitches, n_bip."""
    if player_id is None or not enabled():
        return None
    cols = ("xba", "xwoba", "n_ab", "n_bbe", "whiff_pct", "csw_pct",
            "hard_hit_pct", "barrel_pct", "n_pitches", "n_bip")
    try:
        engine = db_store.get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                select(*[statcast_player_asof.c[c] for c in cols])
                .where((statcast_player_asof.c.player_id == str(player_id))
                       & (statcast_player_asof.c.season_bucket == int(season))
                       & (statcast_player_asof.c.split == split)
                       & (statcast_player_asof.c.role == role))
            ).first()
    except (OperationalError, ValueError, TypeError):
        return None
    return dict(zip(cols, row)) if row is not None else None


def get_batter_xba(player_id, season, split="all"):
    """(xba, n_ab) for a batter's season-to-date expected BA, or (None, 0).
    §2.4a accessor — thin wrapper over get_rates(role="bat")."""
    r = get_rates(player_id, season, "bat", split=split)
    if not r or r.get("xba") is None:
        return (None, 0)
    return (r["xba"], r.get("n_ab") or 0)


def get_pitcher_csw(player_id, season, split="all"):
    """(csw_pct, n_pitches) a pitcher INDUCES, or (None, 0). §2.4b accessor."""
    r = get_rates(player_id, season, "pit", split=split)
    if not r or r.get("csw_pct") is None:
        return (None, 0)
    return (r["csw_pct"], r.get("n_pitches") or 0)


# ── offline write ──────────────────────────────────────────────────────────
def put_rates(season, as_of_date, rates, role, split="all"):
    """Reconcile all rows for (season, split, role) to ``rates``
    ({player_id: {xba,xwoba,n_ab,n_bbe,whiff_pct,csw_pct,hard_hit_pct,
    barrel_pct,n_pitches,n_bip}}; missing keys → NULL). Returns rows in the
    desired end-state.

    WS15: a surgical natural-key upsert (``db_store.reconcile``) replaces the old
    delete-all + insert-all — players in both the store and ``rates`` are UPDATEd
    in place (stable surrogate id), new players INSERTed, absent players DELETEd,
    and the (season, split, role) partition is never momentarily emptied. The
    (player_id, season_bucket, split, role) key is globally unique
    (uq_statcast_player_asof), so a sibling role/season partition is never
    touched (as the role-separation + replace-per-season tests assert)."""
    season = int(season)
    engine = db_store.get_engine()
    _fields = ("xba", "xwoba", "n_ab", "n_bbe", "whiff_pct", "csw_pct",
               "hard_hit_pct", "barrel_pct", "n_pitches", "n_bip")
    params = [dict({
        "player_id": str(pid),
        "season_bucket": season,
        "split": split,
        "role": role,
        "as_of_date": as_of_date,
    }, **{f: r.get(f) for f in _fields}) for pid, r in rates.items()]
    scope = {"season_bucket": season, "split": split, "role": role}
    with _WRITE_LOCK:
        for attempt in range(3):
            try:
                with engine.begin() as conn:
                    db_store.reconcile(
                        conn, statcast_player_asof, params,
                        ("player_id", "season_bucket", "split", "role"),
                        scope=scope)
                return len(params)
            except OperationalError:
                if attempt == 2:
                    raise
    return len(params)


def build(season, start=None, end=None, as_of=None, split="all",
          fetch=False, verbose=True):
    """Compute + store each batter's season-to-date as-of xBA from the raw cache.

    ``as_of`` (default: day after ``end``) is the strict leakage cutoff — every
    pitch strictly before it counts. ``fetch=True`` pulls any missing raw days
    first (slow, Savant-rate-limited); otherwise the days must already be cached
    (``backtest_starters.py --fetch`` / ``savant_history.fetch_range``)."""
    import savant_history as sh
    from datetime import date as _date, timedelta

    season = int(season)
    start = start or f"{season}-03-01"
    end = end or f"{season}-11-30"
    # Never fetch/scan past today: an in-progress season's future days have no
    # games, so pulling them just wastes Savant round-trips (and delays the build).
    today = _date.today().isoformat()
    if end > today:
        end = today
    if as_of is None:
        as_of = (_date.fromisoformat(end) + timedelta(days=1)).isoformat()
    if fetch:
        sh.fetch_range(start, end, verbose=verbose)
    rows = sh.load_days(start, end)
    if not rows:
        raise RuntimeError(
            f"No Statcast days cached for {start}..{end} — run "
            f"`python backtest_starters.py --season {season} --fetch` first "
            f"(or pass --fetch).")

    # Batter role: xBA/xwOBA (§2.4a) merged with batter-side plate-discipline /
    # contact rates (§2.4b). Pitcher role: pitcher-induced whiff/CSW (§2.4b).
    bat_xba = sh.batter_asof_rates(rows, as_of)           # {bid: xba/xwoba/n_ab/n_bbe}
    bat_rates = sh.asof_rates(rows, as_of, "batter")      # {bid: whiff/csw/hh/brl/...}
    bat = dict(bat_rates)
    for bid, x in bat_xba.items():
        bat.setdefault(bid, {}).update(x)
    pit = sh.asof_rates(rows, as_of, "pitcher")           # {pid: whiff/csw/hh/brl/...}

    n_bat = put_rates(season, as_of, bat, "bat", split=split)
    n_pit = put_rates(season, as_of, pit, "pit", split=split)
    if verbose:
        print(f"statcast_asof: wrote {n_bat} batter + {n_pit} pitcher rows for "
              f"season {season} (split={split}, as_of<{as_of}) from {len(rows)} "
              f"pitch rows.")
    return n_bat + n_pit


def _main_cli():
    from cli_encoding import configure_stdio
    configure_stdio()
    ap = argparse.ArgumentParser(
        description="Build the durable per-batter Statcast as-of rate table (SQL).")
    ap.add_argument("--build", action="store_true", help="Compute + write the table.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--start", help="YYYY-MM-DD (default: {season}-03-01).")
    ap.add_argument("--end", help="YYYY-MM-DD (default: {season}-11-30).")
    ap.add_argument("--as-of", dest="as_of",
                    help="Leakage cutoff YYYY-MM-DD (default: day after --end).")
    ap.add_argument("--split", default="all")
    ap.add_argument("--fetch", action="store_true",
                    help="Pull missing raw Statcast days first (slow).")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to write.")
    if args.build:
        build(args.season, start=args.start, end=args.end, as_of=args.as_of,
              split=args.split, fetch=args.fetch)
    else:
        ap.error("nothing to do — pass --build")


if __name__ == "__main__":
    _main_cli()
