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

from sqlalchemy import (
    Boolean, Column, Float, Index, Integer, MetaData, String, Table,
    UniqueConstraint,
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
