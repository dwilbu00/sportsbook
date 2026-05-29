"""
Tiny file-based cache for ESPN player lookups + gamelogs.

Extracted from backtest.py so runtime code (analysis.py → recalibration.py)
can resolve player outcomes without dragging in the full backtest module.
backtest.py re-exports these names for backward compatibility.

Cache files live under SPORTSBOOK_ODDS/cache/backtest/ (legacy name kept
to avoid invalidating existing caches).
"""
import hashlib
import json
import os
import time

from espn_client import get_athlete_gamelog, search_athlete

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache", "backtest")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(*parts):
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


def cached_athlete_id(espn_sport, espn_league, player_name, ttl_hours=24 * 30):
    """Cached athlete ID lookup (player names rarely change)."""
    path = _cache_key("athlete_id", espn_sport, espn_league, player_name.lower())
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < ttl_hours * 3600:
            with open(path) as f:
                return json.load(f).get("id")
    athlete = search_athlete(espn_sport, espn_league, player_name)
    aid = athlete["id"] if athlete else None
    with open(path, "w") as f:
        json.dump({"id": aid}, f)
    return aid


def cached_gamelog(espn_sport, espn_league, athlete_id, ttl_hours=24 * 30,
                   season_year=None):
    """
    Cached gamelog fetch. Historical games are immutable, so a long TTL
    (default 30 days) is safe; the most recent games may lag by up to
    that window.

    When `season_year` is provided, ESPN is queried for that specific
    season and the cache file is keyed by season so different seasons
    don't collide.

    Set ODI_GAMELOG_TTL_HOURS env var to override (e.g., "8760" for a year).
    """
    env_ttl = os.environ.get("ODI_GAMELOG_TTL_HOURS")
    if env_ttl:
        try:
            ttl_hours = float(env_ttl)
        except ValueError:
            pass
    if season_year:
        path = _cache_key("gamelog", espn_sport, espn_league,
                          f"{athlete_id}_s{season_year}")
    else:
        path = _cache_key("gamelog", espn_sport, espn_league, athlete_id)
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < ttl_hours * 3600:
            with open(path) as f:
                return json.load(f)
    gamelog = get_athlete_gamelog(espn_sport, espn_league, athlete_id,
                                  season_year=season_year)
    # MLB pitcher fallback: the standard gamelog endpoint returns nothing
    # for pitchers; the splits endpoint approximates per-game stats.
    if not gamelog and espn_sport == "baseball":
        try:
            from espn_client import get_pitcher_stats
            gamelog = get_pitcher_stats(espn_league, athlete_id,
                                        season=season_year)
        except Exception:
            gamelog = []
    with open(path, "w") as f:
        json.dump(gamelog, f)
    return gamelog
