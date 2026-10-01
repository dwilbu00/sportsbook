"""nfl_warehouse.py — NFL nflverse warehouse: silver dims + per-player-week facts.

Durable Azure persistence mirroring the MLB warehouse (the project's data-storage
standard): a 32-team dim, the games spine, a player dim, and the per-player-week stat
fact that SUPERSEDES the ESPN ``nfl_gamelog`` cache. Source = nflverse — the core,
gradable layers are all dep-free over HTTP (games.csv + stats_player_week + snap_counts),
so the warehouse can self-maintain on Cloud like MLB; richer layers (rosters/pbp/…) are
backfilled offline via nflreadpy (v2). Live ANALYSIS keeps the fast live-HTTP path; this
warehouse is the durable record for grading / backtests / audit.

House conventions (mirror mlb_warehouse.py):
  * Owns its OWN SQLAlchemy Core MetaData + Tables + create_all() (create_all is
    TEST-ONLY / SQLite; prod DDL is the hand-run sql/schema.sql, kept in lockstep and
    guarded by test_nfl_warehouse.py::SchemaParityTests).
  * Reuses db_store.get_engine()/upsert_bulk()/reconcile()/normalize_name()/enabled().
  * Natural keys from nflverse are the dim PKs (team_abbr / game_id / player_id=gsis);
    the games dim is the spine.

Deviation from MLB, by design: nflverse team ABBREVIATIONS drift across seasons
(OAK→LV, SD→LAC, STL→LA), so the game/player facts carry the abbr as a plain indexed
ATTRIBUTE — NOT an enforced FK to nfl_team (a current-32 dim would reject historical
rows on a multi-season backfill). nfl_team is a standalone lookup dim. game_id on
nfl_player_week is a nullable attribute (derived best-effort from season+week+team;
stats_player_week carries no game_id), not an FK.
"""
from __future__ import annotations

import datetime
import time

from sqlalchemy import (
    Boolean, Column, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint, select,
)

import db_store

# ─────────────────────────────────────────────────────────── schema (own MetaData)
_META = MetaData()

nfl_team = Table(
    "nfl_team", _META,
    Column("team_abbr", String(8), primary_key=True),    # nflverse abbr (natural PK)
    Column("name", String(64)),
    Column("name_norm", String(64)),
    Column("conference", String(4)),                     # AFC|NFC
    Column("division", String(8)),                       # e.g. "AFC West"
    Column("fetched_at", Float),
    Index("ix_nfl_team_name", "name_norm"),
)

nfl_game = Table(
    "nfl_game", _META,
    Column("game_id", String(32), primary_key=True),     # nflverse {season}_{wk}_{AWAY}_{HOME}
    Column("season", Integer),
    Column("week", Integer),
    Column("game_type", String(4)),                      # REG|WC|DIV|CON|SB
    Column("gameday", String(10)),                       # YYYY-MM-DD
    Column("gametime", String(8)),                       # kickoff ET HH:MM
    Column("home_team", String(8)),                      # abbr (attribute, NOT FK — abbr drift)
    Column("away_team", String(8)),
    Column("home_score", Float),                         # NULL until played
    Column("away_score", Float),
    Column("location", String(16)),                      # Home|Neutral
    Column("result", Float),                             # home margin
    Column("total", Float),                              # combined points
    Column("espn", String(32)),                         # ESPN game id cross-ref
    Column("fetched_at", Float),
    Index("ix_nfl_game_gameday", "gameday"),
    Index("ix_nfl_game_season_week", "season", "week"),
    Index("ix_nfl_game_teams", "season", "home_team", "away_team"),
)

nfl_player = Table(
    "nfl_player", _META,
    Column("player_id", String(16), primary_key=True),   # nflverse gsis_id (natural PK)
    Column("full_name", String(96)),
    Column("name_norm", String(96)),
    Column("display_name", String(96)),
    Column("football_name", String(96)),
    Column("position", String(8)),
    Column("position_group", String(8)),
    Column("team_abbr", String(8)),                      # most-recent (attribute)
    Column("jersey_number", Integer),                    # offline (rosters)
    Column("years_exp", Integer),                        # offline (rosters)
    Column("pfr_player_id", String(16)),                 # snap-counts join key (offline)
    Column("fetched_at", Float),
    Index("ix_nfl_player_name", "name_norm"),
)

# ── Per-player-week stat fact — supersedes the ESPN nfl_gamelog cache ──────────
# Grain: one row per (player, season, week). Natural key UNIQUE(player_id, season, week)
# (analog of MLB's UNIQUE(athlete_id, game_pk)). team/opponent are abbr attributes;
# game_id is a nullable attribute derived best-effort (no FK). Stats are the full
# stats_player_week set (prop stats are a subset) + snap counts + a played bit.
_PASS_STATS = ("completions", "attempts", "passing_yards", "passing_tds",
               "passing_interceptions", "sacks_suffered", "passing_air_yards",
               "passing_yards_after_catch", "passing_first_downs", "passing_epa",
               "passing_cpoe")
_RUSH_STATS = ("carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
               "rushing_epa")
_RECV_STATS = ("receptions", "targets", "receiving_yards", "receiving_tds",
               "receiving_air_yards", "receiving_yards_after_catch",
               "receiving_first_downs", "receiving_epa", "target_share",
               "air_yards_share")
_MISC_STATS = ("special_teams_tds",)
_SNAP_STATS = ("offense_snaps", "offense_pct", "defense_snaps", "defense_pct",
               "st_snaps", "st_pct")
_PLAYER_WEEK_STATS = (_PASS_STATS + _RUSH_STATS + _RECV_STATS + _MISC_STATS
                      + _SNAP_STATS)

nfl_player_week = Table(
    "nfl_player_week", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("player_id", String(16), nullable=False),     # gsis (natural key part)
    Column("season", Integer, nullable=False),           # (natural key part)
    Column("week", Integer, nullable=False),             # (natural key part)
    Column("game_id", String(32)),                       # nullable attribute (derived), no FK
    Column("team", String(8)),                           # abbr (attribute)
    Column("opponent_team", String(8)),
    Column("position", String(8)),
    Column("position_group", String(8)),
    Column("season_type", String(4)),                    # REG|POST
    *[Column(c, Float) for c in _PLAYER_WEEK_STATS],
    Column("played", Boolean),                            # snap-row presence (DNP/void signal)
    Column("fetched_at", Float),
    UniqueConstraint("player_id", "season", "week", name="uq_nfl_player_week"),
    Index("ix_nfl_player_week_player", "player_id", "season"),
    # (season, week) COVERING index: the uq is (player_id, season, week) so a
    # season/week-first read otherwise scans. mssql_include ignored off SQL Server.
    Index("ix_nfl_player_week_sw", "season", "week",
          mssql_include=["player_id", *_PLAYER_WEEK_STATS, "played"]),
)

# Column-name SPECs — SchemaParityTests asserts these equal the Table columns.
_TEAM_COLS = ("team_abbr", "name", "name_norm", "conference", "division", "fetched_at")
_GAME_COLS = ("game_id", "season", "week", "game_type", "gameday", "gametime",
              "home_team", "away_team", "home_score", "away_score", "location",
              "result", "total", "espn", "fetched_at")
_PLAYER_COLS = ("player_id", "full_name", "name_norm", "display_name",
                "football_name", "position", "position_group", "team_abbr",
                "jersey_number", "years_exp", "pfr_player_id", "fetched_at")
_PLAYER_WEEK_COLS = ("id", "player_id", "season", "week", "game_id", "team",
                     "opponent_team", "position", "position_group", "season_type",
                     *_PLAYER_WEEK_STATS, "played", "fetched_at")


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
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f               # NaN -> None
    except (TypeError, ValueError):
        return None


def _i(v):
    f = _f(v)
    return int(f) if f is not None else None


def _current_season():
    """nflverse season for today (Aug-rollover), via nfl_epa.season_for_date; fail-safe."""
    try:
        import nfl_epa
        return int(nfl_epa.season_for_date(
            datetime.datetime.now(datetime.timezone.utc).date().isoformat()))
    except Exception:
        d = datetime.datetime.now(datetime.timezone.utc).date()
        return d.year if d.month >= 8 else d.year - 1


# ──────────────────────────────────────────────────── ingest (dep-free, Cloud-safe)
def seed_teams():
    """Upsert the 32-team dim from the static nfl_epa / nfl_data maps (no network).
    Idempotent. Returns (n_insert, n_update)."""
    import nfl_epa
    try:
        import nfl_data
        divisions = getattr(nfl_data, "DIVISIONS", {}) or {}
    except Exception:
        divisions = {}
    now = time.time()
    rows = []
    for abbr, name in nfl_epa.TEAM_ABBR_TO_NAME.items():
        div = divisions.get(abbr)
        conf = div[:3] if div else None              # "AFCW"/"NFCE" -> "AFC"/"NFC"
        rows.append({"team_abbr": abbr, "name": name,
                     "name_norm": db_store.normalize_name(name),
                     "conference": conf, "division": div, "fetched_at": now})
    if not rows:
        return (0, 0)
    with db_store.get_engine().begin() as conn:
        return db_store.upsert_bulk(conn, nfl_team, rows, ("team_abbr",),
                                    ignore_cols=("fetched_at",))


def ingest_games(seasons):
    """Upsert nfl_game from the nflverse games spine (nfl_schedule.load_games — dep-free
    games.csv over HTTP on Cloud). Per-season scoped (bounded read). Idempotent."""
    import nfl_schedule
    n_ins = n_upd = 0
    now = time.time()
    for s in seasons:
        recs = nfl_schedule.load_games([str(s)]) or []
        rows = [{
            "game_id": _s(r.get("game_id")), "season": _i(r.get("season")),
            "week": _i(r.get("week")), "game_type": _s(r.get("game_type")),
            "gameday": _s(r.get("gameday")), "gametime": _s(r.get("gametime")),
            "home_team": _s(r.get("home_team")), "away_team": _s(r.get("away_team")),
            "home_score": _f(r.get("home_score")), "away_score": _f(r.get("away_score")),
            "location": _s(r.get("location")), "result": _f(r.get("result")),
            "total": _f(r.get("total")), "espn": _s(r.get("espn")), "fetched_at": now,
        } for r in recs if r.get("game_id")]
        if not rows:
            continue
        with db_store.get_engine().begin() as conn:
            i, u = db_store.upsert_bulk(conn, nfl_game, rows, ("game_id",),
                                        scope={"season": _i(s)},
                                        ignore_cols=("fetched_at",))
        n_ins += i
        n_upd += u
    return (n_ins, n_upd)


def _snap_index(season):
    """{(player_norm, week): snap_row} + the set of weeks whose snap slate is posted, from
    nflverse snap counts (dep-free). Powers the `played` DNP/void signal + snap columns."""
    try:
        import nfl_opportunity_serving as _opp
        snaps = _opp._load_snaps(season)
    except Exception:
        snaps = None
    idx, posted = {}, set()
    if snaps is None:
        return idx, posted
    try:
        for sr in snaps.to_dict("records"):
            wk = _i(sr.get("week"))
            posted.add(wk)
            idx[(sr.get("player_norm"), wk)] = sr
    except Exception:
        pass
    return idx, posted


def ingest_player_weeks(season, weeks=None):
    """Fetch one season's FULL stats_player_week (+ snap counts), upsert nfl_player_week
    and bootstrap nfl_player — all dep-free HTTP (Cloud-safe). Per-(season,week) scoped so
    maintenance touches a bounded slice; ``weeks`` (iterable) limits which weeks. Idempotent.
    Returns (n_insert, n_update) for the fact."""
    import pandas as pd
    import nfl_opportunity_serving as _opp
    import nfl_props_scan as scan
    s = int(season)
    try:
        df = pd.read_parquet(_opp.NFLVERSE_URL.format(season=s))
    except Exception:
        return (0, 0)
    snap_idx, posted_weeks = _snap_index(s)
    want = {int(w) for w in weeks} if weeks is not None else None
    now = time.time()
    by_week = {}                                      # week -> [fact rows]
    players = {}                                      # player_id -> dim row
    for rec in df.to_dict("records"):
        pid = _s(rec.get("player_id"))
        wk = _i(rec.get("week"))
        if not pid or wk is None or (want is not None and wk not in want):
            continue
        name = _s(rec.get("player_display_name"))
        pnorm = scan._norm(name) if name else None
        sr = snap_idx.get((pnorm, wk))
        played = (sr is not None) if wk in posted_weeks else None
        row = {"player_id": pid, "season": _i(rec.get("season")) or s, "week": wk,
               "game_id": None,                       # derived in a later phase (no game_id in feed)
               "team": _s(rec.get("team")), "opponent_team": _s(rec.get("opponent_team")),
               "position": _s(rec.get("position")),
               "position_group": _s(rec.get("position_group")),
               "season_type": _s(rec.get("season_type")),
               "played": played, "fetched_at": now}
        for c in _PLAYER_WEEK_STATS:
            if c in _SNAP_STATS:
                row[c] = _f(sr.get(c)) if sr is not None else None
            else:
                row[c] = _f(rec.get(c))
        by_week.setdefault(wk, []).append(row)
        if pid not in players:
            players[pid] = {
                "player_id": pid, "full_name": name, "name_norm": pnorm,
                "display_name": name, "football_name": None,
                "position": _s(rec.get("position")),
                "position_group": _s(rec.get("position_group")),
                "team_abbr": _s(rec.get("team")), "jersey_number": None,
                "years_exp": None, "pfr_player_id": None, "fetched_at": now}
    n_ins = n_upd = 0
    for wk, rows in sorted(by_week.items()):
        with db_store.get_engine().begin() as conn:
            i, u = db_store.upsert_bulk(conn, nfl_player_week, rows,
                                        ("player_id", "season", "week"),
                                        scope={"season": s, "week": wk},
                                        ignore_cols=("fetched_at",))
        n_ins += i
        n_upd += u
    # Player dim: single-col identity → chunk so the IN stays well under the ~2100 cap.
    plist = list(players.values())
    for i in range(0, len(plist), 500):
        with db_store.get_engine().begin() as conn:
            db_store.upsert_bulk(conn, nfl_player, plist[i:i + 500], ("player_id",),
                                 ignore_cols=("fetched_at",))
    return (n_ins, n_upd)


def ingest_maintenance(recent_weeks=3):
    """Lazy warehouse maintenance (Cloud, dep-free): seed the team dim + upsert the CURRENT
    season's games and recent player-weeks from nflverse HTTP. Idempotent + fail-open per
    step (a warehouse hiccup must never block grading/refit). Mirrors
    mlb_warehouse.ingest_maintenance; called from recalibration.maintain_sport. Bulk
    history is the offline backfill (the CLI below)."""
    if not enabled():
        return
    cur = _current_season()
    try:
        seed_teams()
    except Exception:
        pass
    try:
        ingest_games([cur])
    except Exception:
        pass
    # Only the recent weeks: bound the per-week upsert reads; a backfill does all weeks.
    try:
        import pandas as pd
        import nfl_opportunity_serving as _opp
        df = pd.read_parquet(_opp.NFLVERSE_URL.format(season=cur))
        wks = [int(w) for w in df["week"].dropna().unique()]
        recent = sorted(wks)[-int(recent_weeks):] if wks else None
        ingest_player_weeks(cur, weeks=recent)
    except Exception:
        pass


# ───────────────────────────────────────────────── gold reads (grading, Phase 4)
# Columns grading needs, named EXACTLY as nfl_opportunity_serving._load emits them, so
# the warehouse frame is a drop-in for the live feed in recalibration._resolve_nfl_actual
# (same _canonical_norm nickname resolver + column extraction).
_FRAME_STATS = ("passing_yards", "rushing_yards", "receiving_yards", "receptions",
                "attempts", "carries", "completions", "passing_tds", "rushing_tds",
                "receiving_tds", "special_teams_tds")
_FRAME_TTL = 1800                                     # 30 min memo (warehouse is durable)
_FRAME_CACHE = {}                                    # season(int) -> (ts, DataFrame|None)


def player_week_frame(season, ttl=_FRAME_TTL):
    """A DataFrame shaped like nfl_opportunity_serving._load — ``player_norm``, ``week`` +
    the grading stat columns — read from the warehouse for ``season``, or None when the
    warehouse is disabled/empty for that season. The durable, frozen-record grading source
    (MLB-parity: grade from the DB); a short per-process memo avoids re-querying per prop.
    Never raises — a failure returns None so the caller falls back to the live feed."""
    if not enabled():
        return None
    s = int(season)
    hit = _FRAME_CACHE.get(s)
    if hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    try:
        import pandas as pd
        pw, pl = nfl_player_week, nfl_player
        stmt = (select(pl.c.name_norm.label("player_norm"), pw.c.week,
                       *[pw.c[c] for c in _FRAME_STATS])
                .select_from(pw.join(pl, pw.c.player_id == pl.c.player_id))
                .where(pw.c.season == s))
        with db_store.get_engine().connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(stmt)]
        df = pd.DataFrame(rows) if rows else None
        _FRAME_CACHE[s] = (time.time(), df)
        return df
    except Exception:
        return hit[1] if hit is not None else None


def _backfill_cli():                                  # pragma: no cover - offline owner tool
    """Offline backfill: `python nfl_warehouse.py --seasons 2016-2025`. Run on the dev box
    (set SQL_DRIVER=pyodbc for fast_executemany). Loads games + ALL player-weeks per season."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="e.g. 2016-2025 or 2023,2024,2025")
    args = ap.parse_args()
    db_store.promote_secrets_from_toml()
    if not enabled():
        raise SystemExit("DB not enabled (no secrets.toml?)")
    spec = args.seasons.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = [int(x) for x in spec.split(",") if x.strip()]
    print(f"seed_teams: {seed_teams()}")
    print(f"ingest_games({seasons}): {ingest_games(seasons)}")
    for s in seasons:
        print(f"ingest_player_weeks({s}): {ingest_player_weeks(s)}")


if __name__ == "__main__":                            # pragma: no cover
    _backfill_cli()
