"""nba_warehouse.py — NBA stats.nba.com warehouse: silver dims + per-player-game facts.

Durable Azure persistence mirroring the MLB/NFL warehouses (the project's data-storage
standard). Source = sportsdataverse's **NBA-Stats family** (official stats.nba.com data) via
``nba_stats_source`` — dep-free per-season parquet (package never imported at runtime). This is
the 2026-10 MIGRATION off the ESPN-family feed: real stats.nba.com ids (person_id/team_id int64,
game_id zero-padded string), clean DNP reasons, and (later) advanced team defense. Truly off
ESPN at the data origin, matching how nflverse is for NFL.

House conventions (mirror mlb/nfl_warehouse):
  * Owns its OWN MetaData + Tables + create_all() (TEST-ONLY SQLite; prod DDL is the hand-run
    sql/schema.sql, guarded by test_nba_warehouse SchemaParity + DdlParity).
  * Reuses db_store.get_engine()/upsert_bulk()/normalize_name()/enabled().
  * Natural keys from stats.nba.com are the dim PKs (team_id / game_id / person_id); nba_game
    is the spine. Grain = per-(player, game), UNIQUE(person_id, game_id).

KEY design note: the stats.nba.com player box is NOT self-contained (it has `side` but not
game_date / opponent / scores). Ingest builds a game_id→schedule map and SYNTHESIZES
game_date / opponent_team_id / home_away / scores onto each fact row, so the stored fact — and
every downstream reader (player_game_frame / calib_gamelogs_bulk / player_history) — keeps the
SAME canonical shape as before. Source stat columns are mapped to canonical warehouse names
(rebounds_total→rebounds, three_pointers_made→three_point_field_goals_made, …) so the prop map
(nba_source.PROP_STAT_COLS) and the whole re-tune machinery are unchanged.

Deviation from MLB: team/game refs are indexed attributes, not FKs (robust across any feed
gap; the spine self-contains ids+scores). person_id/team_id are BIGINT (int64 stats.nba.com
ids); game_id stays a String (zero-padded — leading zeros are part of the canonical id).
"""
from __future__ import annotations

import datetime
import os
import time

from sqlalchemy import (
    BigInteger, Boolean, Column, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint, select,
)

import db_store

# ─────────────────────────────────────────────────────────── schema (own MetaData)
_META = MetaData()

nba_team = Table(
    "nba_team", _META,
    Column("team_id", BigInteger, primary_key=True, autoincrement=False),  # stats.nba.com id
    Column("name", String(64)),                          # team_city + " " + team_name
    Column("name_norm", String(64)),
    Column("tricode", String(8)),                        # LAL / GSW / NOP
    Column("city", String(48)),
    Column("conference", String(4)),                     # East|West (real, from standings)
    Column("division", String(16)),                      # e.g. "Pacific" (real, from standings)
    Column("fetched_at", Float),
    Index("ix_nba_team_name", "name_norm"),
    Index("ix_nba_team_tricode", "tricode"),
)

nba_game = Table(
    "nba_game", _META,
    Column("game_id", String(16), primary_key=True),     # stats.nba.com id (zero-padded str)
    Column("season", Integer),                           # season-ENDING year (2025 = 2024-25)
    Column("season_type", String(16)),                   # regular-season|playoffs|play-in|nba-cup
    Column("game_date", String(10)),                     # YYYY-MM-DD
    Column("home_team_id", BigInteger),                  # attribute (not FK)
    Column("away_team_id", BigInteger),
    Column("home_tricode", String(8)),
    Column("away_tricode", String(8)),
    Column("home_score", Float),                         # NULL until played
    Column("away_score", Float),
    Column("completed", Boolean),
    Column("fetched_at", Float),
    Index("ix_nba_game_date", "game_date"),
    Index("ix_nba_game_season", "season", "season_type"),
    Index("ix_nba_game_teams", "season", "home_team_id", "away_team_id"),
)

nba_player = Table(
    "nba_player", _META,
    Column("person_id", BigInteger, primary_key=True, autoincrement=False),  # stats.nba.com id
    Column("full_name", String(96)),                     # first_name + " " + family_name
    Column("name_norm", String(96)),
    Column("first_name", String(48)),
    Column("family_name", String(48)),
    Column("player_slug", String(64)),
    Column("position", String(8)),
    Column("team_id", BigInteger),                        # most-recent (attribute)
    Column("fetched_at", Float),
    Index("ix_nba_player_name", "name_norm"),
)

# ── Per-player-game stat fact ───────────────────────────────────────────────────
# Grain: one row per (player, game). Natural key UNIQUE(person_id, game_id). Stat columns use
# CANONICAL warehouse names (mapped from stats.nba.com source names at ingest) so downstream
# readers + nba_stats_source.PROP_STAT_COLS are unchanged. `played` = minutes>0 (DNP/void signal);
# `dnp_reason` carries the stats.nba.com `comment` for a non-played row.
_BOX_STATS = ("minutes", "points", "field_goals_made", "field_goals_attempted",
              "three_point_field_goals_made", "three_point_field_goals_attempted",
              "free_throws_made", "free_throws_attempted", "offensive_rebounds",
              "defensive_rebounds", "rebounds", "assists", "steals", "blocks",
              "turnovers", "fouls", "plus_minus")

# canonical warehouse column -> stats.nba.com player-box source column (minutes special).
_CANON_FROM_SRC = {
    "points": "points", "field_goals_made": "field_goals_made",
    "field_goals_attempted": "field_goals_attempted",
    "three_point_field_goals_made": "three_pointers_made",
    "three_point_field_goals_attempted": "three_pointers_attempted",
    "free_throws_made": "free_throws_made", "free_throws_attempted": "free_throws_attempted",
    "offensive_rebounds": "rebounds_offensive", "defensive_rebounds": "rebounds_defensive",
    "rebounds": "rebounds_total", "assists": "assists", "steals": "steals",
    "blocks": "blocks", "turnovers": "turnovers", "fouls": "fouls_personal",
    "plus_minus": "plus_minus_points",
}

nba_player_game = Table(
    "nba_player_game", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("person_id", BigInteger, nullable=False),     # (natural key part)
    Column("game_id", String(16), nullable=False),       # (natural key part); joins spine
    Column("season", Integer, nullable=False),           # season-ENDING year (indexed)
    Column("game_date", String(10)),                     # YYYY-MM-DD (grading key, indexed)
    Column("team_id", BigInteger),                        # attribute
    Column("opponent_team_id", BigInteger),
    Column("home_away", String(8)),                      # home|away (from `side`)
    Column("team_score", Float),
    Column("opponent_team_score", Float),
    *[Column(c, Float) for c in _BOX_STATS],
    Column("played", Boolean),                            # minutes>0 (DNP/null-minutes = void)
    Column("dnp_reason", String(48)),                    # stats.nba.com `comment` when not played
    Column("fetched_at", Float),
    UniqueConstraint("person_id", "game_id", name="uq_nba_player_game"),
    Index("ix_nba_player_game_person", "person_id", "season"),
    Index("ix_nba_player_game_game", "game_id"),
    Index("ix_nba_player_game_season", "season", "game_date",
          mssql_include=["person_id", "home_away", "opponent_team_id",
                         *_BOX_STATS, "played"]),
)

# ── Per-team-game fact (pace / efficiency inputs) ────────────────────────────────
# Grain: one row per (team, game) from the stats.nba.com TEAM box (which carries team-level
# rebounds/turnovers that player boxes don't attribute). Same canonical stat names as the
# player fact. `possessions` = the Basketball-Reference estimate averaged over BOTH teams (so
# both rows of a game carry the same value); validated 2024-25 vs official stats.nba.com
# (pace corr 0.995, ORtg 0.998, DRtg 0.997 — a near-constant offset that cancels in
# league-relative ratings). `game_minutes` = team minutes / 5 (48 regulation, 53 per OT).
_TEAM_BOX_STATS = tuple(c for c in _BOX_STATS if c != "minutes")
# stats.nba.com team-box season_type_id -> spine season_type (fallback when a game_id isn't in
# the schedule map; preseason '1' / all-star '3' are deliberately absent -> excluded).
_TEAM_SEASON_TYPE_IDS = {"2": "regular-season", "4": "playoffs", "5": "play-in",
                         "6": "nba-cup"}

nba_team_game = Table(
    "nba_team_game", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("game_id", String(16), nullable=False),       # (natural key part); joins spine
    Column("team_id", BigInteger, nullable=False),       # (natural key part)
    Column("season", Integer, nullable=False),           # season-ENDING year
    Column("season_type", String(16)),                   # regular-season|playoffs|play-in|nba-cup
    Column("game_date", String(10)),                     # YYYY-MM-DD (from the schedule join)
    Column("opponent_team_id", BigInteger),
    Column("home_away", String(8)),                      # home|away (from `side`)
    Column("game_minutes", Float),                       # 48 regulation, +5 per OT
    *[Column(c, Float) for c in _TEAM_BOX_STATS],
    Column("opponent_points", Float),
    Column("possessions", Float),                        # BBRef estimate, both-team averaged
    Column("fetched_at", Float),
    UniqueConstraint("game_id", "team_id", name="uq_nba_team_game"),
    Index("ix_nba_team_game_team", "team_id", "season"),
    Index("ix_nba_team_game_season", "season", "game_date"),
)

# Column-name SPECs — SchemaParityTests asserts these equal the Table columns.
_TEAM_COLS = ("team_id", "name", "name_norm", "tricode", "city", "conference",
              "division", "fetched_at")
_GAME_COLS = ("game_id", "season", "season_type", "game_date", "home_team_id",
              "away_team_id", "home_tricode", "away_tricode", "home_score",
              "away_score", "completed", "fetched_at")
_PLAYER_COLS = ("person_id", "full_name", "name_norm", "first_name", "family_name",
                "player_slug", "position", "team_id", "fetched_at")
_PLAYER_GAME_COLS = ("id", "person_id", "game_id", "season", "game_date", "team_id",
                     "opponent_team_id", "home_away", "team_score",
                     "opponent_team_score", *_BOX_STATS, "played", "dnp_reason",
                     "fetched_at")
_TEAM_GAME_COLS = ("id", "game_id", "team_id", "season", "season_type", "game_date",
                   "opponent_team_id", "home_away", "game_minutes", *_TEAM_BOX_STATS,
                   "opponent_points", "possessions", "fetched_at")

# season_type values kept in the spine (countable games; preseason/all-star excluded).
_REAL_SEASON_TYPES = frozenset({"regular-season", "playoffs", "play-in", "nba-cup"})


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
    """NBA season-ENDING year for today (Aug-rollover; 2025 = 2024-25)."""
    d = datetime.datetime.now(datetime.timezone.utc).date()
    return d.year + 1 if d.month >= 8 else d.year


# ──────────────────────────────────────────────────── ingest (dep-free, Cloud-safe)
def _schedule_map(src, end_year):
    """{game_id: {game_date, home_team_id, away_team_id, home_pts, away_pts, season_type}} from
    the stats.nba.com schedule — the join source that enriches the player box (which lacks
    date/opponent/scores). Also returns the list of spine rows for nba_game."""
    df = src.load_schedule(int(end_year), ttl=0)
    gmap, spine = {}, []
    if df is None or getattr(df, "empty", True):
        return gmap, spine
    now = time.time()
    for r in df.to_dict("records"):
        gid = _s(r.get("game_id"))
        if not gid:
            continue
        stype = _s(r.get("season_type"))
        hid, aid = _i(r.get("home_team_id")), _i(r.get("away_team_id"))
        hp, ap = _f(r.get("home_pts")), _f(r.get("away_pts"))
        gdate = _date(r.get("game_date"))
        gmap[gid] = {"game_date": gdate, "home_team_id": hid, "away_team_id": aid,
                     "home_pts": hp, "away_pts": ap, "season_type": stype}
        if stype in _REAL_SEASON_TYPES:              # spine excludes preseason / all-star
            spine.append({
                "game_id": gid, "season": int(end_year), "season_type": stype,
                "game_date": gdate, "home_team_id": hid, "away_team_id": aid,
                "home_tricode": _s(r.get("home_team_abbreviation")),
                "away_tricode": _s(r.get("away_team_abbreviation")),
                "home_score": hp, "away_score": ap,
                "completed": (hp is not None and ap is not None), "fetched_at": now})
    return gmap, spine


def ingest_games(seasons):
    """Upsert nba_game (the spine) from stats.nba.com schedules (dep-free HTTP). Per-season
    scoped. Idempotent. ``season`` = ending year (2025 = 2024-25). Returns (n_insert, n_update)."""
    import nba_stats_source as src
    n_ins = n_upd = 0
    for end_year in seasons:
        _gmap, spine = _schedule_map(src, end_year)
        if not spine:
            continue
        with db_store.get_engine().begin() as conn:
            i, u = db_store.upsert_bulk(conn, nba_game, spine, ("game_id",),
                                        scope={"season": int(end_year)},
                                        ignore_cols=("fetched_at",))
        n_ins += i
        n_upd += u
    return (n_ins, n_upd)


def _conf_div_map(src, end_year):
    """{team_id: (conference, division)} from the stats.nba.com standings (real, not guessed).
    Empty on any miss — conf/div is non-critical dim enrichment."""
    out = {}
    try:
        df = src.load_standings(int(end_year), ttl=0)
        if df is None or getattr(df, "empty", True):
            return out
        cols = set(df.columns)
        if not ({"team_id"} & cols):
            return out
        for r in df.to_dict("records"):
            tid = _i(r.get("team_id"))
            if tid is not None:
                out[tid] = (_s(r.get("conference")), _s(r.get("division")))
    except Exception:
        pass
    return out


def ingest_player_games(season, since_date=None):
    """Fetch one season's stats.nba.com player box, enrich via the schedule join (game_date /
    opponent / home-away / scores), map source stats to canonical names, upsert nba_player_game
    and bootstrap nba_player + nba_team. ``season`` = ending year; ``since_date`` (YYYY-MM-DD)
    bounds maintenance. Facts upserted per-game under per-date transactions. Idempotent.
    Returns (n_insert, n_update) for the fact."""
    import nba_stats_source as src
    end_year = int(season)
    df = src.load_player_box(end_year, ttl=0)            # ingest always reads fresh
    if df is None or getattr(df, "empty", True):
        return (0, 0)
    gmap, _spine = _schedule_map(src, end_year)          # game_id -> date/opponent/scores
    conf_div = _conf_div_map(src, end_year)
    now = time.time()
    by_date = {}                                         # game_date -> {game_id -> [rows]}
    players = {}                                         # person_id -> dim row
    teams = {}                                           # team_id -> dim row
    for rec in df.to_dict("records"):
        pid = _i(rec.get("person_id"))
        gid = _s(rec.get("game_id"))
        if pid is None or not gid:
            continue
        g = gmap.get(gid, {})
        gdate = g.get("game_date") or _date(rec.get("game_date"))
        if since_date and (gdate is None or gdate < since_date):
            continue
        side = _s(rec.get("side"))                       # home|away (on the box row)
        tid = _i(rec.get("team_id"))
        if side == "home":
            opp_id, tscore, oscore = g.get("away_team_id"), g.get("home_pts"), g.get("away_pts")
        elif side == "away":
            opp_id, tscore, oscore = g.get("home_team_id"), g.get("away_pts"), g.get("home_pts")
        else:
            opp_id = tscore = oscore = None
        minutes = src.minutes_to_float(rec.get("minutes"))
        played = (minutes is not None and minutes > 0)   # ""/None/0 min = DNP/void
        comment = _s(rec.get("comment"))
        row = {"person_id": pid, "game_id": gid, "season": end_year, "game_date": gdate,
               "team_id": tid, "opponent_team_id": opp_id, "home_away": side,
               "team_score": tscore, "opponent_team_score": oscore,
               "played": played, "dnp_reason": (comment if not played else None),
               "fetched_at": now}
        for c in _BOX_STATS:
            row[c] = minutes if c == "minutes" else _f(rec.get(_CANON_FROM_SRC[c]))
        by_date.setdefault(gdate, {}).setdefault(gid, []).append(row)
        if pid not in players:
            first, fam = _s(rec.get("first_name")), _s(rec.get("family_name"))
            full = " ".join(x for x in (first, fam) if x) or None
            players[pid] = {
                "person_id": pid, "full_name": full,
                "name_norm": db_store.normalize_name(full) if full else None,
                "first_name": first, "family_name": fam,
                "player_slug": _s(rec.get("player_slug")),
                "position": _s(rec.get("position")), "team_id": tid, "fetched_at": now}
        if tid is not None and tid not in teams:
            city, nm = _s(rec.get("team_city")), _s(rec.get("team_name"))
            full = " ".join(x for x in (city, nm) if x) or None
            conf, div = conf_div.get(tid, (None, None))
            teams[tid] = {"team_id": tid, "name": full,
                          "name_norm": db_store.normalize_name(full) if full else None,
                          "tricode": _s(rec.get("team_tricode")), "city": city,
                          "conference": conf, "division": div, "fetched_at": now}
    n_ins = n_upd = 0
    for gdate in sorted(by_date, key=lambda d: (d is None, d or "")):
        with db_store.get_engine().begin() as conn:
            for gid, rows in by_date[gdate].items():
                i, u = db_store.upsert_bulk(conn, nba_player_game, rows,
                                            ("person_id", "game_id"),
                                            scope={"game_id": gid},
                                            ignore_cols=("fetched_at",))
                n_ins += i
                n_upd += u
    tlist = list(teams.values())
    if tlist:
        with db_store.get_engine().begin() as conn:
            db_store.upsert_bulk(conn, nba_team, tlist, ("team_id",),
                                 ignore_cols=("fetched_at",))
    plist = list(players.values())
    for i in range(0, len(plist), 500):                 # chunk the single-col IN (~2100 cap)
        with db_store.get_engine().begin() as conn:
            db_store.upsert_bulk(conn, nba_player, plist[i:i + 500], ("person_id",),
                                 ignore_cols=("fetched_at",))
    return (n_ins, n_upd)


def estimate_possessions(tm, opp):
    """Basketball-Reference possession estimate for one game, averaged over both teams:
    ``FGA + 0.4·FTA − 1.07·ORB%·(FGA − FGM) + TOV`` per side, ORB% = ORB / (ORB + opp DRB).
    ``tm``/``opp`` are canonical box dicts. None if any input is missing."""
    keys = ("field_goals_attempted", "field_goals_made", "free_throws_attempted",
            "offensive_rebounds", "defensive_rebounds", "turnovers")
    if any(tm.get(k) is None or opp.get(k) is None for k in keys):
        return None

    def _side(a, b):
        orb, odrb = a["offensive_rebounds"], b["defensive_rebounds"]
        orb_pct = orb / (orb + odrb) if (orb + odrb) > 0 else 0.0
        return (a["field_goals_attempted"] + 0.4 * a["free_throws_attempted"]
                - 1.07 * orb_pct * (a["field_goals_attempted"] - a["field_goals_made"])
                + a["turnovers"])

    return 0.5 * (_side(tm, opp) + _side(opp, tm))


def ingest_team_games(season, since_date=None):
    """Fetch one season's stats.nba.com TEAM box, pair each game's two team rows, derive
    possessions + opponent + points-allowed, enrich date/season_type via the schedule join, and
    upsert nba_team_game. Excludes preseason/all-star (same rule as the spine) and any game
    without exactly two team rows. ``season`` = ending year; ``since_date`` bounds maintenance.
    One transaction per season. Idempotent. Returns (n_insert, n_update)."""
    import nba_stats_source as src
    end_year = int(season)
    df = src.load_team_box(end_year, ttl=0)
    if df is None or getattr(df, "empty", True):
        return (0, 0)
    gmap, _spine = _schedule_map(src, end_year)
    now = time.time()
    by_game = {}
    for rec in df.to_dict("records"):
        gid = _s(rec.get("game_id"))
        if gid and _i(rec.get("team_id")) is not None:
            by_game.setdefault(gid, []).append(rec)
    rows = []
    for gid, recs in by_game.items():
        if len(recs) != 2:
            continue
        g = gmap.get(gid, {})
        stype = g.get("season_type") or _TEAM_SEASON_TYPE_IDS.get(
            _s(recs[0].get("season_type_id")))
        if stype not in _REAL_SEASON_TYPES:
            continue
        gdate = g.get("game_date")
        if since_date and (gdate is None or gdate < since_date):
            continue
        box = [{c: _f(r.get(_CANON_FROM_SRC[c])) for c in _TEAM_BOX_STATS} for r in recs]
        poss = estimate_possessions(box[0], box[1])
        for k in (0, 1):
            r, o = recs[k], recs[1 - k]
            tmin = src.minutes_to_float(r.get("minutes"))
            rows.append({
                "game_id": gid, "team_id": _i(r.get("team_id")), "season": end_year,
                "season_type": stype, "game_date": gdate,
                "opponent_team_id": _i(o.get("team_id")), "home_away": _s(r.get("side")),
                "game_minutes": (tmin / 5.0) if tmin else None,
                **box[k],
                "opponent_points": box[1 - k]["points"], "possessions": poss,
                "fetched_at": now})
    if not rows:
        return (0, 0)
    with db_store.get_engine().begin() as conn:
        return db_store.upsert_bulk(conn, nba_team_game, rows, ("game_id", "team_id"),
                                    scope={"season": end_year},
                                    ignore_cols=("fetched_at",))


def ingest_maintenance(recent_days=14):
    """Lazy warehouse maintenance (Cloud, dep-free): upsert the CURRENT season's games + the
    last ``recent_days`` of player-games and team-games from stats.nba.com HTTP. Idempotent +
    fail-open per step. Mirrors mlb/nfl_warehouse; called from recalibration.maintain_sport."""
    if not enabled():
        return
    cur = _current_season()
    try:
        ingest_games([cur])
    except Exception:
        pass
    since = (datetime.datetime.now(datetime.timezone.utc).date()
             - datetime.timedelta(days=int(recent_days))).isoformat()
    try:
        ingest_player_games(cur, since_date=since)
    except Exception:
        pass
    try:
        ingest_team_games(cur, since_date=since)
    except Exception:
        pass


# ───────────────────────────────────────────────── gold reads (grading / sweep / calib)
# Canonical stat columns the frames expose — UNCHANGED from the ESPN-family warehouse, so
# recalibration._resolve_nba_actual / the backtest sweep / the calib join are all unaffected.
_FRAME_STATS = ("points", "rebounds", "assists", "steals", "blocks", "turnovers",
                "three_point_field_goals_made", "field_goals_made",
                "field_goals_attempted", "free_throws_made", "free_throws_attempted",
                "offensive_rebounds", "defensive_rebounds", "minutes")
_FRAME_TTL = 1800
_FRAME_CACHE = {}                                   # season(int) -> (ts, DataFrame|None)


def player_game_frame(season, ttl=_FRAME_TTL):
    """One season's player-games as a DataFrame — ``player_norm``, ``game_date``, ``played`` +
    the canonical grading stats — or None when disabled/empty. Output shape is identical to the
    old warehouse, so grading is source-agnostic. Never raises."""
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
                .select_from(pg.join(pl, pg.c.person_id == pl.c.person_id))
                .where(pg.c.season == s))
        with db_store.get_engine().connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(stmt)]
        df = pd.DataFrame(rows) if rows else None
        _FRAME_CACHE[s] = (time.time(), df)
        return df
    except Exception:
        return hit[1] if hit is not None else None


def _disambiguate_person(conn, pids, team_ids):
    """Pick ONE person_id from same-normalized-name candidates: prefer whose most-recent game is
    with a team in ``team_ids`` (stats.nba.com ids), else the overall most-recent. Namesakes
    are rare; this keeps a live lookup from interleaving two players."""
    rows = conn.execute(
        select(nba_player_game.c.person_id, nba_player_game.c.team_id)
        .where(nba_player_game.c.person_id.in_(pids))
        .order_by(nba_player_game.c.game_date.desc()))
    seen = {}
    for r in rows:
        if r.person_id not in seen:
            seen[r.person_id] = r.team_id
    if team_ids:
        want = {str(t) for t in team_ids}
        for p, tid in seen.items():
            if str(tid) in want:
                return p
    return next(iter(seen), pids[0])


_NAME_RESOLVER = {}                                   # {"ts": t, "r": nba_names.Resolver}


def name_resolver(ttl=1800):
    """nba_names.Resolver over every nba_player name_norm (cached) — maps odds-feed spellings
    ("Jimmy Butler", "Nicolas Claxton") onto the warehouse's. None when disabled / on error."""
    hit = _NAME_RESOLVER.get("r")
    if hit is not None and (time.time() - _NAME_RESOLVER.get("ts", 0)) < ttl:
        return hit
    if not enabled():
        return None
    try:
        import nba_names
        with db_store.get_engine().connect() as conn:
            known = [r[0] for r in conn.execute(select(nba_player.c.name_norm)) if r[0]]
    except Exception:
        return None
    r = nba_names.Resolver(known)
    _NAME_RESOLVER.update({"ts": time.time(), "r": r})
    return r


def player_history(player_name, cols, n=20, as_of_date=None, team_ids=None):
    """Recent per-game stat history for one player, most-recent-first, PLAYED games only (DNP
    excluded). ``cols`` str or tuple (SUMMED per game). Returns [{value, game_date, opponent,
    home_away, athlete_id, team_id}] (``athlete_id`` carries the person_id — the generic
    get_player_stat_history contract field), or [] when disabled / no played games. Never
    raises."""
    if not enabled():
        return []
    cols = cols if isinstance(cols, tuple) else (cols,)
    try:
        nn = db_store.normalize_name(player_name)
        if not nn:
            return []
        res = name_resolver()
        nn = (res.resolve(player_name) if res is not None else None) or nn
        pg, pl = nba_player_game, nba_player
        present = [c for c in cols if c in pg.c]
        if not present:
            return []
        with db_store.get_engine().connect() as conn:
            pids = [r[0] for r in conn.execute(
                select(pl.c.person_id).where(pl.c.name_norm == nn))]
            if not pids:
                return []
            pid = pids[0] if len(pids) == 1 else _disambiguate_person(conn, pids, team_ids)
            stmt = (select(pg.c.person_id, pg.c.game_date, pg.c.opponent_team_id,
                           pg.c.home_away, pg.c.team_id, pg.c.played,
                           *[pg.c[c] for c in present])
                    .where(pg.c.person_id == pid))
            if as_of_date:
                stmt = stmt.where(pg.c.game_date < str(as_of_date)[:10])
            stmt = stmt.order_by(pg.c.game_date.desc())
            rows = [dict(r._mapping) for r in conn.execute(stmt)]
        out = []
        for r in rows:
            if not r.get("played"):
                continue
            out.append({"value": sum(float(r.get(c) or 0.0) for c in present),
                        "game_date": r.get("game_date"),
                        "opponent": r.get("opponent_team_id"),
                        "home_away": r.get("home_away"),
                        "athlete_id": r.get("person_id"), "team_id": r.get("team_id")})
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


# ───────────────────────────────────────── team-market grading (off the spine)
def _resolve_team_id(conn, name):
    """stats.nba.com team_id for an odds-feed team name via the nba_team dim: exact name_norm,
    then a UNIQUE nickname (last token) match (odds 'LA Clippers' vs 'Los Angeles Clippers').
    None if absent/ambiguous."""
    nn = db_store.normalize_name(name)
    if not nn:
        return None
    row = conn.execute(select(nba_team.c.team_id)
                       .where(nba_team.c.name_norm == nn)).fetchone()
    if row:
        return row[0]
    tok = nn.split()[-1]
    rows = conn.execute(select(nba_team.c.team_id)
                        .where(nba_team.c.name_norm.like("% " + tok))).fetchall()
    return rows[0][0] if len(rows) == 1 else None


_TEAM_ID_CACHE = {}                                   # normalized odds name -> team_id


def resolve_team_id(name):
    """Public, connection-owning ``_resolve_team_id`` (odds-feed name -> stats.nba.com
    team_id), memoized per process (team names are stable). None when disabled/absent."""
    nn = db_store.normalize_name(name or "")
    if not nn:
        return None
    if nn in _TEAM_ID_CACHE:
        return _TEAM_ID_CACHE[nn]
    if not enabled():
        return None
    try:
        with db_store.get_engine().connect() as conn:
            tid = _resolve_team_id(conn, name)
    except Exception:
        return None
    if tid is not None:
        _TEAM_ID_CACHE[nn] = int(tid)
    return int(tid) if tid is not None else None


_PLAYER_TEAMS = {}                                    # {"ts": t, "map": {name_norm: team_id}}


def player_teams(ttl=1800):
    """{name_norm: current team_id} from the nba_player dim (most-recent team), cached — the
    bonus optimizer's same-game teammate/opponent tagging. {} when disabled / on error."""
    hit = _PLAYER_TEAMS.get("map")
    if hit is not None and (time.time() - _PLAYER_TEAMS.get("ts", 0)) < ttl:
        return hit
    if not enabled():
        return {}
    try:
        with db_store.get_engine().connect() as conn:
            m = {r[0]: int(r[1]) for r in conn.execute(
                select(nba_player.c.name_norm, nba_player.c.team_id))
                if r[0] and r[1] is not None}
    except Exception:
        return {}
    _PLAYER_TEAMS.update({"ts": time.time(), "map": m})
    return m


def team_names():
    """{team_id: name} from the nba_team dim ({} when disabled / on error)."""
    if not enabled():
        return {}
    try:
        with db_store.get_engine().connect() as conn:
            return {int(r[0]): r[1] for r in conn.execute(
                select(nba_team.c.team_id, nba_team.c.name))}
    except Exception:
        return {}


def final_score(home_team, away_team, game_date, commence=None):
    """(home_score, away_score) for a completed NBA game from the nba_game spine, resolving the
    odds home/away names -> team_id (nba_team dim) and matching on game_date (exact, then ±1 for
    UTC/ET slippage — safe: a team never hosts the same opponent on consecutive days). None when
    disabled / unresolved / not yet final (caller stays pending, same fail-closed contract as the
    ESPN path). Never raises."""
    if not enabled():
        return None
    try:
        with db_store.get_engine().connect() as conn:
            hid = _resolve_team_id(conn, home_team)
            aid = _resolve_team_id(conn, away_team)
            if hid is None or aid is None:
                return None
            d0 = str(game_date)[:10]
            dates = [d0]
            try:
                base = datetime.date.fromisoformat(d0)
                dates += [(base + datetime.timedelta(days=k)).isoformat() for k in (-1, 1)]
            except (TypeError, ValueError):
                pass
            for d in dates:
                r = conn.execute(
                    select(nba_game.c.home_score, nba_game.c.away_score)
                    .where(nba_game.c.home_team_id == hid, nba_game.c.away_team_id == aid,
                           nba_game.c.game_date == d, nba_game.c.home_score.isnot(None),
                           nba_game.c.away_score.isnot(None))).fetchone()
                if r is not None:
                    return (float(r[0]), float(r[1]))
        return None
    except Exception:
        return None


# ── Local parquet MIRROR of the player-game actuals (0-DTU calib/sweep reads) ──────
_MIRROR_COLS = ("name_norm", "season", "game_date", "game_id", "team_id", "home_away",
                "opponent_team_id", "played", *_BOX_STATS)


def _mirror_dir():
    return os.environ.get(
        "ODI_MIRROR_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "warehouse_mirror_data"))


def _mirror_file(season):
    return os.path.join(_mirror_dir(),
                        f"nba_player_game__basketball_nba__{int(season)}.parquet")


def _nn(v):
    """NaN/NA -> None (parquet reloads None floats as NaN). Keeps a real 0 distinct from
    'no data' so a DNP's null stat never becomes 0."""
    try:
        if v is None:
            return None
        if isinstance(v, float) and v != v:
            return None
        import pandas as pd
        if pd.isna(v):
            return None
    except (TypeError, ValueError, ImportError):
        pass
    return v


def _row_to_game(m):
    """A sweep/calib game dict from a mirror OR Azure record (shared shape; NaN-safe)."""
    g = {"season": _nn(m.get("season")), "game_date": m.get("game_date"),
         "game_pk": m.get("game_id"), "team_id": _nn(m.get("team_id")),
         "is_home": (m.get("home_away") == "home"),
         "opponent": _nn(m.get("opponent_team_id")),
         "MIN": _nn(m.get("minutes")), "played": _nn(m.get("played"))}
    for c in _BOX_STATS:
        g[c] = _nn(m.get(c))
    return g


def _query_player_games(seasons):
    """Raw per-game records (dicts) for ``seasons`` from Azure — the mirror source + the Azure
    fallback for any un-mirrored season."""
    pg, pl = nba_player_game, nba_player
    stmt = (select(pl.c.name_norm.label("name_norm"), pg.c.season, pg.c.game_date,
                   pg.c.game_id, pg.c.team_id, pg.c.home_away, pg.c.opponent_team_id,
                   pg.c.played, *[pg.c[c] for c in _BOX_STATS])
            .select_from(pg.join(pl, pg.c.person_id == pl.c.person_id))
            .where(pg.c.season.in_([int(s) for s in seasons])))
    with db_store.get_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def export_mirror(seasons, verbose=True):                # pragma: no cover - offline owner tool
    """Dump nba_player_game (+ name_norm) per season from Azure to the local parquet mirror so
    calib_gamelogs_bulk reads actuals at 0 DTU. Run AFTER a (re-)backfill; completed seasons are
    static. Returns rows written per season."""
    if not enabled():
        raise SystemExit("DB not enabled (no secrets.toml?) — can't export the mirror.")
    import pandas as pd
    os.makedirs(_mirror_dir(), exist_ok=True)
    written = {}
    for s in seasons:
        rows = _query_player_games([int(s)])
        df = pd.DataFrame(rows, columns=list(_MIRROR_COLS)) if rows else pd.DataFrame(
            columns=list(_MIRROR_COLS))
        df.to_parquet(_mirror_file(s), index=False)
        written[int(s)] = len(df)
        if verbose:
            print(f"  nba_player_game mirror {s}: {len(df):,} rows -> {_mirror_file(s)}")
    return written


def calib_gamelogs_bulk(seasons):
    """{name_norm: [per-game dict, most-recent-first]} across ``seasons``, for the offline
    calibration/opportunity join AND the backtest-engine NBA sweep. Each dict: season,
    game_date, game_pk, team_id, MIN, is_home, opponent, played + every box stat (canonical).
    MIRROR-FIRST (0 DTU): parquet per season when present, Azure for un-mirrored. Never raises."""
    from collections import defaultdict
    out = defaultdict(list)

    def _add(records):
        for m in records:
            nm = m.get("name_norm")
            if nm:
                out[nm].append(_row_to_game(m))

    azure_seasons = []
    for s in seasons:
        f = _mirror_file(s)
        if os.path.exists(f):
            try:
                import pandas as pd
                _add(pd.read_parquet(f).to_dict("records"))
                continue
            except Exception:
                pass
        azure_seasons.append(int(s))
    if azure_seasons and enabled():
        try:
            _add(_query_player_games(azure_seasons))
        except Exception:
            pass
    for nm in out:
        out[nm].sort(key=lambda g: g.get("game_date") or "", reverse=True)
    return dict(out)


# ── Team-game MIRROR + reader (pace / efficiency ratings; 0-DTU offline reads) ──────────
_TEAM_MIRROR_COLS = tuple(c for c in _TEAM_GAME_COLS if c not in ("id", "fetched_at"))


def _team_mirror_file(season):
    return os.path.join(_mirror_dir(),
                        f"nba_team_game__basketball_nba__{int(season)}.parquet")


def _query_team_games(seasons):
    tg = nba_team_game
    stmt = (select(*[tg.c[c] for c in _TEAM_MIRROR_COLS])
            .where(tg.c.season.in_([int(s) for s in seasons])))
    with db_store.get_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def export_team_mirror(seasons, verbose=True):           # pragma: no cover - offline owner tool
    """Dump nba_team_game per season from Azure to the local parquet mirror (0-DTU reads for
    the ratings module / team-model study). Run AFTER a backfill. Returns rows per season."""
    if not enabled():
        raise SystemExit("DB not enabled (no secrets.toml?) — can't export the mirror.")
    import pandas as pd
    os.makedirs(_mirror_dir(), exist_ok=True)
    written = {}
    for s in seasons:
        rows = _query_team_games([int(s)])
        df = pd.DataFrame(rows, columns=list(_TEAM_MIRROR_COLS))
        df.to_parquet(_team_mirror_file(s), index=False)
        written[int(s)] = len(df)
        if verbose:
            print(f"  nba_team_game mirror {s}: {len(df):,} rows -> {_team_mirror_file(s)}")
    return written


def team_games(seasons):
    """Team-game dicts (``_TEAM_MIRROR_COLS``) across ``seasons``, ascending by
    (game_date, game_id, team_id). MIRROR-FIRST (parquet per season), Azure for un-mirrored
    seasons. NaN -> None. Never raises (returns what it could read)."""
    out, azure_seasons = [], []
    for s in seasons:
        f = _team_mirror_file(s)
        if os.path.exists(f):
            try:
                import pandas as pd
                out.extend({k: _nn(v) for k, v in m.items()}
                           for m in pd.read_parquet(f).to_dict("records"))
                continue
            except Exception:
                pass
        azure_seasons.append(int(s))
    if azure_seasons and enabled():
        try:
            out.extend(_query_team_games(azure_seasons))
        except Exception:
            pass
    out.sort(key=lambda r: (str(r.get("game_date") or ""), str(r.get("game_id") or ""),
                            r.get("team_id") or 0))
    return out


def validate_team_games(season):
    """Per-team REGULAR-SEASON pace / ORtg / DRtg derived from nba_team_game vs the official
    stats.nba.com advanced season totals. Returns {n_teams, pace|off_rating|def_rating:
    {corr, mean_err, mae}} or None when either side is unavailable. Read-only."""
    try:
        import numpy as np
        import pandas as pd
        import nba_stats_source as src
    except Exception:
        return None
    rows = [r for r in team_games([season]) if r.get("season_type") == "regular-season"]
    off = src.load_team_season_stats(int(season))
    if not rows or off is None or getattr(off, "empty", True):
        return None
    df = pd.DataFrame(rows)
    g = df.groupby("team_id").agg(poss=("possessions", "sum"), mins=("game_minutes", "sum"),
                                  pts=("points", "sum"), opts=("opponent_points", "sum"))
    g["pace"] = 48.0 * g["poss"] / g["mins"]
    g["off_rating"] = 100.0 * g["pts"] / g["poss"]
    g["def_rating"] = 100.0 * g["opts"] / g["poss"]
    adv = off[(off["measure_type"] == "advanced") & (off["season_type"] == "regular-season")
              & (off["per_mode"] == "totals")].copy()
    adv["team_id"] = pd.to_numeric(adv["team_id"])
    g.index = pd.to_numeric(g.index)
    j = adv.merge(g, left_on="team_id", right_index=True, suffixes=("_off", ""))
    if j.empty:
        return None
    res = {"n_teams": int(len(j))}
    for k in ("pace", "off_rating", "def_rating"):
        ours, theirs = j[k].astype(float), pd.to_numeric(j[k + "_off"])
        d = ours - theirs
        corr = (float(np.corrcoef(ours, theirs)[0, 1])
                if ours.std() > 0 and theirs.std() > 0 else None)   # undefined w/o spread
        res[k] = {"corr": corr, "mean_err": float(d.mean()), "mae": float(d.abs().mean())}
    return res


def prop_id_index(seasons):
    """{(name_norm, 'YYYY-MM-DD'): set((person_id, game_id))} over ``seasons`` — the resolver
    the P5b odds-line id stamp (nba_stamp_odds.py) uses to pre-compute the line->actuals join.
    Keyed on the SAME ``db_store.normalize_name`` + EXACT ``game_date`` that the real-line
    calibration join matches on (``book_line_calibration._nba_grade_book_lines``), so a stamped
    (person_id, game_id) equals what that join resolves for the common (one-person-per-name)
    case. Reads Azure directly (the stamp CLI writes Azure); never raises -> {}."""
    from collections import defaultdict
    idx = defaultdict(set)
    pg, pl = nba_player_game, nba_player
    stmt = (select(pl.c.name_norm, pg.c.game_date, pg.c.person_id, pg.c.game_id)
            .select_from(pg.join(pl, pg.c.person_id == pl.c.person_id))
            .where(pg.c.season.in_([int(s) for s in seasons])))
    try:
        with db_store.get_engine().connect() as conn:
            for nn, gd, pid, gid in conn.execute(stmt):
                if nn and gd and pid is not None and gid is not None:
                    idx[(nn, str(gd)[:10])].add((int(pid), str(gid)))
    except Exception:
        return {}
    return dict(idx)


def resolve_prop_ids(index, player_name, et_date, resolver=None):
    """(person_id, game_id) for a prop book line IFF (name_norm, ET-date) resolves to EXACTLY
    one warehouse player-game; else (None, None). UNAMBIGUOUS-ONLY by design: a namesake
    collision (two same-norm players who both have a game that date) or a no-match is left
    unstamped for the name+date fallback — the stamp never guesses. ``index`` = ``prop_id_index``
    output; ``et_date`` = US-Eastern date of the line's commence (``pricing_common.et_local_date``)."""
    nn = db_store.normalize_name(player_name or "")
    if resolver is not None:                       # odds spelling -> warehouse name_norm
        nn = resolver.resolve(player_name) or nn
    cands = index.get((nn, str(et_date or "")[:10]))
    if cands and len(cands) == 1:
        return next(iter(cands))
    return None, None


def _backfill_cli():                                 # pragma: no cover - offline owner tool
    """Offline backfill: `python nba_warehouse.py --seasons 1997-2026` (stats.nba.com; set
    SQL_DRIVER=pyodbc for fast_executemany). `--export-mirror` dumps the 0-DTU parquet mirror
    instead (run it AFTER a backfill)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="season-ENDING years, e.g. 1997-2026 or 2023,2024,2025,2026")
    ap.add_argument("--export-mirror", action="store_true",
                    help="Dump nba_player_game AND nba_team_game to the local parquet mirror "
                         "for these seasons instead of ingesting. Run after a backfill.")
    ap.add_argument("--team-games-only", action="store_true",
                    help="Ingest ONLY nba_team_game (team box + possessions) for these "
                         "seasons — skips the heavy player-game re-ingest.")
    ap.add_argument("--validate-team", action="store_true",
                    help="Compare derived pace/ORtg/DRtg vs official stats.nba.com season "
                         "stats for these seasons (read-only).")
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
    if args.export_mirror:
        print(f"export_mirror({seasons}):")
        export_mirror(seasons)
        export_team_mirror(seasons)
        return
    if args.validate_team:
        for s in seasons:
            v = validate_team_games(s)
            if not v:
                print(f"  {s}: unavailable (no team games or no official stats)")
                continue
            print(f"  {s}: {v['n_teams']} teams  " + "  ".join(
                f"{k} corr {v[k]['corr'] if v[k]['corr'] is None else round(v[k]['corr'], 4)}"
                f" err {v[k]['mean_err']:+.2f}" for k in ("pace", "off_rating", "def_rating")))
        return
    if args.team_games_only:
        for s in seasons:
            print(f"ingest_team_games({s}): {ingest_team_games(s)}")
        return
    print(f"ingest_games({seasons}): {ingest_games(seasons)}")
    for s in seasons:
        print(f"ingest_player_games({s}): {ingest_player_games(s)}")
        print(f"ingest_team_games({s}): {ingest_team_games(s)}")


if __name__ == "__main__":                           # pragma: no cover
    _backfill_cli()
