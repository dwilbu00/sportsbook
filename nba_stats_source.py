"""nba_stats_source.py — dep-free sportsdataverse NBA-Stats (stats.nba.com) data layer.

The migration target replacing the ESPN-family nba_source.py (2026-10): OFFICIAL NBA stats —
real person_id / team_id (int64) + game_id (zero-padded string), advanced team defense
(def_rating/pace/opponent splits), and clean DNP reasons — published as per-season parquet
GitHub release assets, read dep-free with pandas (the sportsdataverse package is NEVER imported
at runtime; same rule as nflverse/the ESPN family). Season label = the season-ENDING year
(2024-25 -> _2025 = start_year+1); data back to 1996-97. Same 302-redirect / 404 / no-auth /
not-LFS profile as the ESPN family, so the loader shape is a near 1:1 port.

Deliberately a THIN fetcher: unlike the ESPN box, the stats.nba.com player box is NOT
self-contained — it has `side` (home/away) but NOT game_date / opponent / scores. Those are
synthesized by a lossless game_id -> schedule join at INGEST time (nba_warehouse), not here, so
each loader just returns the raw frame (+ player_norm on the box for name resolution).
"""
import datetime
import time

import pandas as pd

import db_store

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PLAYER_BOX_URL  = BASE + "/nba_stats_player_boxscores/player_boxscores_{season}.parquet"
TEAM_BOX_URL    = BASE + "/nba_stats_team_boxscores/team_boxscores_{season}.parquet"
SCHEDULE_URL    = BASE + "/nba_stats_schedules/nba_schedule_{season}.parquet"
TEAM_SEASON_URL = BASE + "/nba_stats_team_season_stats/team_season_stats_{season}.parquet"
STANDINGS_URL   = BASE + "/nba_stats_standings/standings_{season}.parquet"

MIN_SEASON = 1997            # 1996-97 (player box earliest); below this -> 404 -> None
CACHE_TTL = 6 * 3600         # 6h; a season's parquet refreshes at most ~daily (next-day)
_CACHE = {}                  # (kind, season) -> (fetch_ts, DataFrame|None)


# prop_key -> canonical warehouse stat column(s); a tuple is SUMMED (combo props: PRA, P+R, …).
# The single source of truth for NBA prop<->stat mapping, shared by grading, warehouse history,
# the calib join, and the sweep. Columns are the CANONICAL warehouse names (the ingest maps
# stats.nba.com source names to these), so this is unchanged across the ESPN->stats.nba.com
# migration. Double/triple-double are category-counts (not simple sums) -> intentionally absent.
PROP_STAT_COLS = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "three_point_field_goals_made",
    "player_steals": "steals",
    "player_blocks": "blocks",
    "player_turnovers": "turnovers",
    "player_field_goals": "field_goals_made",
    "player_frees_made": "free_throws_made",
    "player_frees_attempts": "free_throws_attempted",
    "player_points_rebounds_assists": ("points", "rebounds", "assists"),
    "player_points_rebounds": ("points", "rebounds"),
    "player_points_assists": ("points", "assists"),
    "player_rebounds_assists": ("rebounds", "assists"),
    "player_blocks_steals": ("blocks", "steals"),
    "player_steals_blocks": ("steals", "blocks"),
}


def _norm(name):
    return db_store.normalize_name(name)


def end_year_for_date(d):
    """NBA season-ENDING year for a date (== the stats.nba.com file label). The season spans
    Oct–Jun; Aug+ rolls to the next calendar year (the ending year), Jan–Jul stays.
    2024-11-01 -> 2025 (the 2024-25 season = file _2025). Identical convention to the ESPN
    family, so the warehouse's season keying is unchanged."""
    if d is None:
        return None
    if isinstance(d, str):
        try:
            d = datetime.date.fromisoformat(d[:10])
        except ValueError:
            return None
    y, m = d.year, d.month
    return y + 1 if m >= 8 else y


def current_end_year():
    return end_year_for_date(datetime.datetime.now(datetime.timezone.utc).date())


def minutes_to_float(v):
    """stats.nba.com minutes are 'MM:SS' strings ('35:56' -> 35.93). Empty '' / None / NaN =
    did-not-play -> None (the DNP signal; a real 0-minute appearance is '0:00' -> 0.0)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    if ":" in s:
        mm, _, ss = s.partition(":")
        try:
            return round(int(mm) + int(float(ss)) / 60.0, 2)
        except (TypeError, ValueError):
            return None
    try:
        f = float(s)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _fetch(url, retries=3):
    """GET a release-asset parquet directly (no package, no GitHub API). DataFrame, or None on
    a persistent failure (incl. a real 404 = season absent). Short backoff rides out GitHub's
    occasional 403/5xx on an asset under load."""
    for i in range(max(1, retries)):
        try:
            return pd.read_parquet(url)
        except Exception:                            # 404 (absent) or transient 403/5xx
            if i < retries - 1:
                time.sleep(0.5 * (i + 1))
    return None


def _load(kind, url, season, ttl, add_norm=False):
    s = int(season)
    hit = _CACHE.get((kind, s))
    if ttl and hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    df = _fetch(url.format(season=s))
    if df is None:
        return hit[1] if hit is not None else None   # serve stale on a transient miss
    if add_norm and "first_name" in df.columns and "family_name" in df.columns:
        df = df.copy()
        df["player_norm"] = [_norm(f"{a} {b}")
                             for a, b in zip(df["first_name"], df["family_name"])]
    _CACHE[(kind, s)] = (time.time(), df)
    return df


def load_player_box(season, ttl=CACHE_TTL):
    """One season's stats.nba.com player box (per-player-per-game) + a ``player_norm`` column
    (normalize(first_name + ' ' + family_name)). None on a missing season / persistent miss.
    NOT self-contained — game_date/opponent/scores come from the schedule join at ingest."""
    return _load("pbox", PLAYER_BOX_URL, season, ttl, add_norm=True)


def load_team_box(season, ttl=CACHE_TTL):
    """One season's team box (2 rows/game, raw box stats per side). Used to DERIVE per-game
    possessions/pace for the as-of pace-adjusted defense (stats.nba.com has no per-game
    advanced)."""
    return _load("tbox", TEAM_BOX_URL, season, ttl)


def load_schedule(season, ttl=CACHE_TTL):
    """One season's schedule (the spine): game_id, game_date, home/away team_id + pts, status,
    season_type. The join source for enriching the player box with date/opponent/scores."""
    return _load("sched", SCHEDULE_URL, season, ttl)


def load_team_season_stats(season, ttl=CACHE_TTL):
    """One season's advanced team stats (def_rating/pace/net_rating + opponent splits). Grain
    is sparse-by-measure_type — filter/pivot at ingest (see P4)."""
    return _load("tseason", TEAM_SEASON_URL, season, ttl)


def load_standings(season, ttl=CACHE_TTL):
    """One season's standings — real conference/division per team (retires the static map)."""
    return _load("stand", STANDINGS_URL, season, ttl)
