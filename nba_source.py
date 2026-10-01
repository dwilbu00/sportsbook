"""nba_source.py — dep-free sportsdataverse NBA data layer (ESPN-family parquet).

sportsdataverse publishes per-season ESPN NBA data as GitHub RELEASE ASSETS:
    https://github.com/sportsdataverse/sportsdataverse-data/releases/download/
        <tag>/<file>_<endyear>.parquet
A plain public HTTPS GET (one 302 to a short-lived signed blob, auto-followed by
pandas/pyarrow/fsspec) — NOT git-LFS, no auth, no gzip wrapper. So the Cloud runtime
reads it DIRECTLY with pandas; the ``sportsdataverse`` package itself is an OFFLINE/dev
dependency ONLY and is never imported here (same runtime rule as the nflverse layer in
nfl_opportunity_serving). This is the single NBA data source for both the durable
warehouse (nba_warehouse) and live analysis — it replaces the live ESPN API.

Conventions:
  * Season label = the season-ENDING year (2024 = 2023-24). Data back to 2002; the
    current-season asset refreshes next-day during the season.
  * A missing/not-yet-published season 404s → the loader returns None (treat as absent).
  * A short retry/backoff rides out GitHub's occasional 403/5xx on a release asset.
  * Live callers use the TTL cache (6h); the warehouse ingest reads fresh (ttl=0).
"""
import datetime
import time

import pandas as pd

import db_store

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PLAYER_BOX_URL = BASE + "/espn_nba_player_boxscores/player_box_{season}.parquet"
TEAM_BOX_URL = BASE + "/espn_nba_team_boxscores/team_box_{season}.parquet"      # v2
SCHEDULE_URL = BASE + "/espn_nba_schedules/nba_schedule_{season}.parquet"

MIN_SEASON = 2002
CACHE_TTL = 6 * 3600         # 6h — a season's parquet refreshes at most ~daily
_PLAYER_CACHE = {}           # end_year(int) -> (fetch_ts, DataFrame|None)
_SCHED_CACHE = {}            # end_year(int) -> (fetch_ts, DataFrame|None)


def _norm(name):
    return db_store.normalize_name(name)


def end_year_for_date(d):
    """NBA season-ENDING year for a date (YYYY-MM-DD str / date / datetime). The season
    spans Oct–Jun; Aug+ rolls to the next calendar year (the ending year), Jan–Jul stays.
    2023-11-01 -> 2024; 2024-06-17 -> 2024. Mirrors nba_warehouse._current_season."""
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


def _fetch_parquet(url, retries=3):
    """GET a release-asset parquet directly (no package, no GitHub API). Returns a
    DataFrame, or None on a persistent failure (incl. a real 404 = season absent).
    A short backoff rides out GitHub's occasional 403/5xx on an asset under load."""
    last = None
    for i in range(max(1, retries)):
        try:
            return pd.read_parquet(url)
        except Exception as e:                       # 404 (absent) or transient 403/5xx
            last = e
            if i < retries - 1:
                time.sleep(0.5 * (i + 1))
    return None


def load_player_box(season, ttl=CACHE_TTL):
    """One season's ESPN player box (per-player-per-game) + a ``player_norm`` column.
    Cached with TTL; ``ttl=0`` forces a fresh fetch (the warehouse ingest). None on a
    missing season or persistent fetch failure (serves stale on a transient miss)."""
    s = int(season)
    hit = _PLAYER_CACHE.get(s)
    if ttl and hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    df = _fetch_parquet(PLAYER_BOX_URL.format(season=s))
    if df is not None:
        df = df.copy()
        if "athlete_display_name" in df.columns:
            df["player_norm"] = df["athlete_display_name"].map(_norm)
    elif hit is not None:
        return hit[1]                                # serve stale on transient fetch failure
    _PLAYER_CACHE[s] = (time.time(), df)
    return df


def load_schedule(season, ttl=CACHE_TTL):
    """One season's ESPN schedule (per-game spine). Cached with TTL; ``ttl=0`` forces a
    fresh fetch. None on a missing season or persistent fetch failure."""
    s = int(season)
    hit = _SCHED_CACHE.get(s)
    if ttl and hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    df = _fetch_parquet(SCHEDULE_URL.format(season=s))
    if df is None and hit is not None:
        return hit[1]
    _SCHED_CACHE[s] = (time.time(), df)
    return df
