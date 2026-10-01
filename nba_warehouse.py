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

from sqlalchemy import (
    Boolean, Column, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint,
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
# `played` = active AND not DNP AND minutes>0 (the DNP/void signal, like NFL snaps).
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
    Column("active", Boolean),
    Column("played", Boolean),                            # active & !DNP & minutes>0 (void signal)
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


def _current_season():
    """NBA season-ENDING year for today (Aug-rollover; 2024 = 2023-24)."""
    d = datetime.datetime.now(datetime.timezone.utc).date()
    return d.year + 1 if d.month >= 8 else d.year
