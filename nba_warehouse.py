"""nba_warehouse.py — NBA sportsdataverse warehouse: silver dims + per-player-game facts.

Durable Azure persistence mirroring the MLB warehouse (the project's data-storage
standard) and the NFL warehouse: a team dim, the games spine, a player dim, and the
per-player-game stat fact. Source = sportsdataverse's ESPN-family per-season parquet
(GitHub release assets) — a plain public HTTPS GET (no auth, not LFS), so the warehouse
is dep-free over HTTP and can self-maintain on Cloud like MLB/NFL. The ``sportsdataverse``
package itself is an OFFLINE/dev dependency ONLY — never imported at runtime (same runtime
rule as the nflverse layer); the Cloud reader hits the deterministic release URL directly.
This SUPERSEDES the ephemeral ESPN ``nba_gamelog`` cache (gamelog_store) and gets NBA
grading + live analysis fully off the live ESPN API.

House conventions (mirror mlb_warehouse.py / nfl_warehouse.py):
  * Owns its OWN SQLAlchemy Core MetaData + Tables + create_all() (create_all is
    TEST-ONLY / SQLite; prod DDL is the hand-run sql/schema.sql, kept in lockstep and
    guarded by test_nba_warehouse.py::SchemaParityTests + DdlParityTests).
  * Reuses db_store.get_engine()/upsert_bulk()/normalize_name()/enabled().
  * Natural keys from the ESPN feed are the dim PKs (team_id / game_id / athlete_id);
    the games dim is the spine. Grain = per-(player, game), keyed UNIQUE(athlete_id,
    game_id) (MLB-style, game-centric; NBA grading disambiguates by date, not week).

Deviation from MLB, by design: team/game references are plain indexed ATTRIBUTES, NOT
enforced FKs. ESPN NBA ids are *generally* stable, but a multi-season backfill (2002→)
crosses franchise relocations/renames (SEA→OKC, NJ→BKN, NOH→NOP) and occasional
schedule/box feed mismatches; attribute refs (the NFL precedent) never reject a historical
or slightly-inconsistent row on backfill. The spine (nba_game) self-contains abbrs + ids +
scores, so team-market grading reads it without the team dim; nba_team is a convenience
lookup enriched with conference/division from a static map (reliable conf/div via the
standings dataset is a v2 enrichment).
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

nba_team = Table(
    "nba_team", _META,
    Column("team_id", String(16), primary_key=True),     # ESPN team id (natural PK)
    Column("name", String(64)),                          # team_display_name
    Column("name_norm", String(64)),
    Column("abbreviation", String(8)),
    Column("location", String(48)),                      # e.g. "Los Angeles"
    Column("conference", String(4)),                     # East|West (static map, best-effort)
    Column("division", String(16)),                      # e.g. "Pacific" (static map)
    Column("fetched_at", Float),
    Index("ix_nba_team_name", "name_norm"),
    Index("ix_nba_team_abbr", "abbreviation"),
)

nba_game = Table(
    "nba_game", _META,
    Column("game_id", String(16), primary_key=True),     # ESPN event id (natural PK, spine)
    Column("season", Integer),                           # season-ENDING year (2024 = 2023-24)
    Column("season_type", Integer),                      # 1=pre 2=reg 3=post (ESPN)
    Column("game_date", String(10)),                     # YYYY-MM-DD (ET)
    Column("game_date_time", String(32)),                # ISO tip-off (ET)
    Column("home_team_id", String(16)),                  # id (attribute, NOT FK)
    Column("away_team_id", String(16)),
    Column("home_abbr", String(8)),                      # point-in-time abbr (attribute)
    Column("away_abbr", String(8)),
    Column("home_score", Float),                         # NULL until played
    Column("away_score", Float),
    Column("completed", Boolean),                        # status_type_completed
    Column("neutral_site", Boolean),
    Column("venue_id", String(16)),
    Column("fetched_at", Float),
    Index("ix_nba_game_date", "game_date"),
    Index("ix_nba_game_season", "season", "season_type"),
    Index("ix_nba_game_teams", "season", "home_team_id", "away_team_id"),
)

nba_player = Table(
    "nba_player", _META,
    Column("athlete_id", String(16), primary_key=True),  # ESPN athlete id (natural PK)
    Column("full_name", String(96)),                     # athlete_display_name
    Column("name_norm", String(96)),
    Column("short_name", String(96)),                    # athlete_short_name
    Column("position", String(8)),                       # athlete_position_abbreviation
    Column("jersey", String(8)),
    Column("team_id", String(16)),                       # most-recent (attribute)
    Column("fetched_at", Float),
    Index("ix_nba_player_name", "name_norm"),
)

# ── Per-player-game stat fact — supersedes the ESPN nba_gamelog cache ───────────
# Grain: one row per (player, game). Natural key UNIQUE(athlete_id, game_id) (MLB-style,
# game-centric). team/opponent/game refs are attributes (no FK — see module docstring).
# `played` = minutes>0 (the DNP/void signal, like NFL snaps). MINUTES is null exactly for
# did_not_play rows; the ESPN `active` flag is unreliable so it is stored but not the signal.
_BOX_STATS = ("minutes", "points", "field_goals_made", "field_goals_attempted",
              "three_point_field_goals_made", "three_point_field_goals_attempted",
              "free_throws_made", "free_throws_attempted", "offensive_rebounds",
              "defensive_rebounds", "rebounds", "assists", "steals", "blocks",
              "turnovers", "fouls", "plus_minus")

nba_player_game = Table(
    "nba_player_game", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("athlete_id", String(16), nullable=False),    # (natural key part)
    Column("game_id", String(16), nullable=False),       # (natural key part); joins spine
    Column("season", Integer, nullable=False),           # season-ENDING year (indexed)
    Column("game_date", String(10)),                     # YYYY-MM-DD (grading key, indexed)
    Column("team_id", String(16)),                       # attribute
    Column("opponent_team_id", String(16)),
    Column("home_away", String(8)),                      # home|away
    Column("team_score", Float),
    Column("opponent_team_score", Float),
    *[Column(c, Float) for c in _BOX_STATS],
    Column("starter", Boolean),
    Column("did_not_play", Boolean),
    Column("active", Boolean),                           # ESPN flag (unreliable; stored, not used)
    Column("played", Boolean),                            # minutes>0 (DNP/null-minutes = void)
    Column("fetched_at", Float),
    UniqueConstraint("athlete_id", "game_id", name="uq_nba_player_game"),
    Index("ix_nba_player_game_athlete", "athlete_id", "season"),
    Index("ix_nba_player_game_game", "game_id"),
    # (season, game_date) COVERING index for player_game_frame reads (uq is athlete-first).
    Index("ix_nba_player_game_season", "season", "game_date",
          mssql_include=["athlete_id", "home_away", "opponent_team_id",
                         *_BOX_STATS, "played"]),
)

# Column-name SPECs — SchemaParityTests asserts these equal the Table columns.
_TEAM_COLS = ("team_id", "name", "name_norm", "abbreviation", "location",
              "conference", "division", "fetched_at")
_GAME_COLS = ("game_id", "season", "season_type", "game_date", "game_date_time",
              "home_team_id", "away_team_id", "home_abbr", "away_abbr",
              "home_score", "away_score", "completed", "neutral_site", "venue_id",
              "fetched_at")
_PLAYER_COLS = ("athlete_id", "full_name", "name_norm", "short_name", "position",
                "jersey", "team_id", "fetched_at")
_PLAYER_GAME_COLS = ("id", "athlete_id", "game_id", "season", "game_date", "team_id",
                     "opponent_team_id", "home_away", "team_score",
                     "opponent_team_score", *_BOX_STATS, "starter", "did_not_play",
                     "active", "played", "fetched_at")

# Best-effort conference/division by ESPN abbreviation (non-critical dim enrichment;
# None on miss — reliable conf/div via load_nba_standings is a v2 enrichment). Includes
# historical abbrs seen on a 2002→ backfill (SEA→OKC, NJ→BKN, NOH/NOK→NOP).
_CONF_DIV = {
    "BOS": ("East", "Atlantic"), "BKN": ("East", "Atlantic"), "NJ": ("East", "Atlantic"),
    "NY": ("East", "Atlantic"), "PHI": ("East", "Atlantic"), "TOR": ("East", "Atlantic"),
    "CHI": ("East", "Central"), "CLE": ("East", "Central"), "DET": ("East", "Central"),
    "IND": ("East", "Central"), "MIL": ("East", "Central"),
    "ATL": ("East", "Southeast"), "CHA": ("East", "Southeast"), "MIA": ("East", "Southeast"),
    "ORL": ("East", "Southeast"), "WSH": ("East", "Southeast"),
    "DEN": ("West", "Northwest"), "MIN": ("West", "Northwest"), "OKC": ("West", "Northwest"),
    "SEA": ("West", "Northwest"), "POR": ("West", "Northwest"), "UTAH": ("West", "Northwest"),
    "GS": ("West", "Pacific"), "LAC": ("West", "Pacific"), "LAL": ("West", "Pacific"),
    "PHX": ("West", "Pacific"), "SAC": ("West", "Pacific"),
    "DAL": ("West", "Southwest"), "HOU": ("West", "Southwest"), "MEM": ("West", "Southwest"),
    "NO": ("West", "Southwest"), "NOH": ("West", "Southwest"), "NOK": ("West", "Southwest"),
    "SA": ("West", "Southwest"),
}


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


def _b(v):
    """Coerce to bool|None (NaN/None -> None). numpy.bool_ and pandas NA safe."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and v != v:        # NaN
            return None
    except (TypeError, ValueError):
        pass
    try:
        import pandas as pd
        if v is pd.NA or (hasattr(pd, "isna") and pd.isna(v)):
            return None
    except (TypeError, ValueError, ImportError):
        pass
    return bool(v)


def _date(v):
    """A YYYY-MM-DD string from a pandas Timestamp / date / ISO string (None on NaT/empty)."""
    try:
        import pandas as pd
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError, ImportError):
        if v is None:
            return None
    s = str(v).strip()
    return s[:10] if len(s) >= 10 else (s or None)


def _current_season():
    """NBA season-ENDING year for today (Aug-rollover; 2024 = 2023-24)."""
    d = datetime.datetime.now(datetime.timezone.utc).date()
    return d.year + 1 if d.month >= 8 else d.year


# ──────────────────────────────────────────────────── ingest (dep-free, Cloud-safe)
def ingest_games(seasons):
    """Upsert nba_game (the spine) from sportsdataverse ESPN schedules (dep-free HTTP).
    Per-season scoped (bounded read). Idempotent. ``season`` here is the season-ENDING
    year (2024 = 2023-24). Returns (n_insert, n_update)."""
    import nba_source as src
    n_ins = n_upd = 0
    now = time.time()
    for end_year in seasons:
        df = src.load_schedule(int(end_year), ttl=0)     # ingest always reads fresh
        if df is None or getattr(df, "empty", True):
            continue
        rows = []
        for r in df.to_dict("records"):
            gid = _s(r.get("game_id")) or _s(r.get("id"))
            if not gid:
                continue
            rows.append({
                "game_id": gid, "season": int(end_year),
                "season_type": _i(r.get("season_type")),
                "game_date": _date(r.get("game_date")),
                "game_date_time": _s(r.get("game_date_time")),
                "home_team_id": _s(r.get("home_id")),
                "away_team_id": _s(r.get("away_id")),
                "home_abbr": _s(r.get("home_abbreviation")),
                "away_abbr": _s(r.get("away_abbreviation")),
                "home_score": _f(r.get("home_score")),
                "away_score": _f(r.get("away_score")),
                "completed": _b(r.get("status_type_completed")),
                "neutral_site": _b(r.get("neutral_site")),
                "venue_id": _s(r.get("venue_id")), "fetched_at": now,
            })
        if not rows:
            continue
        with db_store.get_engine().begin() as conn:
            i, u = db_store.upsert_bulk(conn, nba_game, rows, ("game_id",),
                                        scope={"season": int(end_year)},
                                        ignore_cols=("fetched_at",))
        n_ins += i
        n_upd += u
    return (n_ins, n_upd)


def _team_row(rec, now):
    """A nba_team dim row from a player-box record (richest source: id/abbr/name/location),
    enriched with best-effort conference/division from the static abbr map."""
    tid = _s(rec.get("team_id"))
    if not tid:
        return None
    abbr = _s(rec.get("team_abbreviation"))
    conf, div = _CONF_DIV.get(abbr, (None, None)) if abbr else (None, None)
    name = _s(rec.get("team_display_name")) or _s(rec.get("team_name"))
    return {"team_id": tid, "name": name,
            "name_norm": db_store.normalize_name(name) if name else None,
            "abbreviation": abbr, "location": _s(rec.get("team_location")),
            "conference": conf, "division": div, "fetched_at": now}


def ingest_player_games(season, since_date=None):
    """Fetch one season's ESPN player box, upsert nba_player_game and bootstrap the
    nba_player + nba_team dims — all dep-free HTTP (Cloud-safe). ``season`` = ending year.
    ``since_date`` (YYYY-MM-DD) limits to games on/after it (maintenance bound). Facts are
    upserted per-game (scope={game_id}) under per-DATE transactions (bounded reads, fewer
    commits). Idempotent. Returns (n_insert, n_update) for the fact."""
    import nba_source as src
    end_year = int(season)
    df = src.load_player_box(end_year, ttl=0)            # ingest always reads fresh
    if df is None or getattr(df, "empty", True):
        return (0, 0)
    now = time.time()
    by_date = {}                                         # game_date -> {game_id -> [rows]}
    players = {}                                         # athlete_id -> dim row
    teams = {}                                           # team_id -> dim row
    for rec in df.to_dict("records"):
        aid = _s(rec.get("athlete_id"))
        gid = _s(rec.get("game_id"))
        if not aid or not gid:
            continue
        gdate = _date(rec.get("game_date"))
        if since_date and (gdate is None or gdate < since_date):
            continue
        name = _s(rec.get("athlete_display_name"))
        minutes = _f(rec.get("minutes"))
        active = _b(rec.get("active"))
        dnp = _b(rec.get("did_not_play"))
        # played = the DNP/void signal. MINUTES is the ground truth (it's null exactly for
        # did_not_play rows; >0 for players who logged time). The ESPN `active` flag is
        # UNRELIABLE (False even for a 37-min, 70-pt game) so it is stored but NOT used here.
        played = None
        if minutes is not None:
            played = minutes > 0                      # 0 min (dressed, no time) = void
        elif dnp is not None:
            played = not dnp
        row = {"athlete_id": aid, "game_id": gid, "season": end_year,
               "game_date": gdate, "team_id": _s(rec.get("team_id")),
               "opponent_team_id": _s(rec.get("opponent_team_id")),
               "home_away": _s(rec.get("home_away")),
               "team_score": _f(rec.get("team_score")),
               "opponent_team_score": _f(rec.get("opponent_team_score")),
               "starter": _b(rec.get("starter")), "did_not_play": dnp,
               "active": active, "played": played, "fetched_at": now}
        for c in _BOX_STATS:
            row[c] = _f(rec.get(c))
        by_date.setdefault(gdate, {}).setdefault(gid, []).append(row)
        if aid not in players:
            players[aid] = {
                "athlete_id": aid, "full_name": name,
                "name_norm": db_store.normalize_name(name) if name else None,
                "short_name": _s(rec.get("athlete_short_name")),
                "position": _s(rec.get("athlete_position_abbreviation")),
                "jersey": _s(rec.get("athlete_jersey")),
                "team_id": _s(rec.get("team_id")), "fetched_at": now}
        tid = _s(rec.get("team_id"))
        if tid and tid not in teams:
            tr = _team_row(rec, now)
            if tr:
                teams[tid] = tr
    n_ins = n_upd = 0
    for gdate in sorted(by_date, key=lambda d: (d is None, d or "")):
        with db_store.get_engine().begin() as conn:
            for gid, rows in by_date[gdate].items():
                i, u = db_store.upsert_bulk(conn, nba_player_game, rows,
                                            ("athlete_id", "game_id"),
                                            scope={"game_id": gid},
                                            ignore_cols=("fetched_at",))
                n_ins += i
                n_upd += u
    tlist = list(teams.values())
    if tlist:
        with db_store.get_engine().begin() as conn:
            db_store.upsert_bulk(conn, nba_team, tlist, ("team_id",),
                                 ignore_cols=("fetched_at",))
    # Player dim: single-col identity → chunk so the IN stays well under the ~2100 cap.
    plist = list(players.values())
    for i in range(0, len(plist), 500):
        with db_store.get_engine().begin() as conn:
            db_store.upsert_bulk(conn, nba_player, plist[i:i + 500], ("athlete_id",),
                                 ignore_cols=("fetched_at",))
    return (n_ins, n_upd)


def ingest_maintenance(recent_days=14):
    """Lazy warehouse maintenance (Cloud, dep-free): upsert the CURRENT season's games +
    the last ``recent_days`` of player-games from sportsdataverse HTTP. Idempotent +
    fail-open per step (a warehouse hiccup must never block grading/refit). Mirrors
    mlb/nfl_warehouse.ingest_maintenance; called from recalibration.maintain_sport. Bulk
    history is the offline backfill (the CLI below). Teams/players bootstrap in-line."""
    if not enabled():
        return
    cur = _current_season()
    try:
        ingest_games([cur])
    except Exception:
        pass
    try:
        since = (datetime.datetime.now(datetime.timezone.utc).date()
                 - datetime.timedelta(days=int(recent_days))).isoformat()
        ingest_player_games(cur, since_date=since)
    except Exception:
        pass


# ───────────────────────────────────────────────── gold reads (grading, Phase 4)
# Columns grading needs — the full box stat set, named EXACTLY as the ESPN player box
# (and nba_source) emit them, plus ``player_norm`` + ``game_date`` — so the warehouse
# frame is a drop-in for a live player-box read in recalibration._resolve_nba_actual.
_FRAME_STATS = ("points", "rebounds", "assists", "steals", "blocks", "turnovers",
                "three_point_field_goals_made", "field_goals_made",
                "field_goals_attempted", "free_throws_made", "free_throws_attempted",
                "offensive_rebounds", "defensive_rebounds", "minutes")
_FRAME_TTL = 1800                                    # 30 min memo (warehouse is durable)
_FRAME_CACHE = {}                                   # season(int) -> (ts, DataFrame|None)


def player_game_frame(season, ttl=_FRAME_TTL):
    """A DataFrame of one season's player-games — ``player_norm``, ``game_date`` + the
    grading stat columns + ``played`` — read from the warehouse, or None when the warehouse
    is disabled/empty for that season. The durable, frozen-record grading source (MLB/NFL
    parity: grade from the DB); a short per-process memo avoids re-querying per prop. Never
    raises — a failure returns None so the caller falls back to the live feed."""
    if not enabled():
        return None
    s = int(season)
    hit = _FRAME_CACHE.get(s)
    if hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    try:
        import pandas as pd
        pg, pl = nba_player_game, nba_player
        stmt = (select(pl.c.name_norm.label("player_norm"), pg.c.game_date, pg.c.played,
                       *[pg.c[c] for c in _FRAME_STATS])
                .select_from(pg.join(pl, pg.c.athlete_id == pl.c.athlete_id))
                .where(pg.c.season == s))
        with db_store.get_engine().connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(stmt)]
        df = pd.DataFrame(rows) if rows else None
        _FRAME_CACHE[s] = (time.time(), df)
        return df
    except Exception:
        return hit[1] if hit is not None else None


def _disambiguate_athlete(conn, aids, team_ids):
    """Pick ONE athlete_id from same-normalized-name candidates: prefer whose most-recent game
    is with a team in ``team_ids`` (ESPN ids == warehouse team_id), else the athlete with the
    overall most-recent game. (NBA namesakes are rare; this keeps a live lookup from
    interleaving two players' games.)"""
    rows = conn.execute(
        select(nba_player_game.c.athlete_id, nba_player_game.c.team_id)
        .where(nba_player_game.c.athlete_id.in_(aids))
        .order_by(nba_player_game.c.game_date.desc()))
    seen = {}
    for r in rows:
        if r.athlete_id not in seen:                 # first seen == most-recent game (desc)
            seen[r.athlete_id] = r.team_id
    if team_ids:
        want = {str(t) for t in team_ids}
        for a, tid in seen.items():
            if str(tid) in want:
                return a
    return next(iter(seen), aids[0])


def player_history(player_name, cols, n=20, as_of_date=None, team_ids=None):
    """Recent per-game stat history for one player from the warehouse, most-recent-first,
    PLAYED games only (DNP rows excluded — a prop voids on a DNP, never counts a phantom 0).
    ``cols`` is a stat column or a tuple SUMMED per game (combo props). Returns a list of
    {value, game_date, opponent, home_away, athlete_id, team_id}, or [] when disabled / the
    player has no played games. ``as_of_date`` (YYYY-MM-DD) caps to games strictly before it
    (as-of/backtest); ``team_ids`` disambiguates a namesake. Never raises."""
    if not enabled():
        return []
    cols = cols if isinstance(cols, tuple) else (cols,)
    try:
        nn = db_store.normalize_name(player_name)
        if not nn:
            return []
        pg, pl = nba_player_game, nba_player
        present = [c for c in cols if c in pg.c]
        if not present:
            return []
        with db_store.get_engine().connect() as conn:
            aids = [r[0] for r in conn.execute(
                select(pl.c.athlete_id).where(pl.c.name_norm == nn))]
            if not aids:
                return []
            aid = aids[0] if len(aids) == 1 else _disambiguate_athlete(conn, aids, team_ids)
            stmt = (select(pg.c.athlete_id, pg.c.game_date, pg.c.opponent_team_id,
                           pg.c.home_away, pg.c.team_id, pg.c.played,
                           *[pg.c[c] for c in present])
                    .where(pg.c.athlete_id == aid))
            if as_of_date:
                stmt = stmt.where(pg.c.game_date < str(as_of_date)[:10])
            stmt = stmt.order_by(pg.c.game_date.desc())
            rows = [dict(r._mapping) for r in conn.execute(stmt)]
        out = []
        for r in rows:
            if not r.get("played"):                  # DNP → excluded (void, not a 0)
                continue
            out.append({"value": sum(float(r.get(c) or 0.0) for c in present),
                        "game_date": r.get("game_date"),
                        "opponent": r.get("opponent_team_id"),
                        "home_away": r.get("home_away"),
                        "athlete_id": r.get("athlete_id"), "team_id": r.get("team_id")})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def _backfill_cli():                                 # pragma: no cover - offline owner tool
    """Offline backfill: `python nba_warehouse.py --seasons 2002-2026`. Run on the dev box
    (set SQL_DRIVER=pyodbc for fast_executemany). Loads games + ALL player-games per season."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="season-ENDING years, e.g. 2002-2026 or 2024,2025,2026")
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
    print(f"ingest_games({seasons}): {ingest_games(seasons)}")
    for s in seasons:
        print(f"ingest_player_games({s}): {ingest_player_games(s)}")


if __name__ == "__main__":                           # pragma: no cover
    _backfill_cli()
