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

# Optional SQL backend (Phase C durable gamelog store). Guarded so a missing
# SQLAlchemy install simply leaves SQL disabled and the file cache is used.
try:
    import db_store
except Exception:  # pragma: no cover - import guard
    db_store = None


def _sql():
    return db_store is not None and db_store.enabled()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache", "backtest")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(*parts):
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


# Failed OR genuinely-empty ESPN responses (None id / empty gamelog) are cached
# only briefly. The underlying ESPN helpers swallow network errors and return
# None/[] indistinguishably from a real "no data" result, so a single transient
# outage would otherwise poison a player's lookup for the full 30-day success
# TTL — silently dropping them from projections and outcome resolution. A short
# negative TTL lets a transient miss recover within minutes while still avoiding
# hammering ESPN for a genuinely absent player within one analysis.
NEGATIVE_TTL_HOURS = 0.25  # 15 minutes


def _read_cache_file(path):
    """Return parsed cache contents, or None if missing/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _is_fresh(path, ttl_hours):
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return False
    return age < ttl_hours * 3600


def cached_athlete_id(espn_sport, espn_league, player_name, ttl_hours=24 * 30,
                      team_ids=None):
    """Cached athlete ID lookup (player names rarely change).

    ``team_ids`` (the matchup's ESPN team ids) disambiguates same-name players;
    it is forwarded to search_athlete and folded into the cache key so different
    teams don't collide.
    """
    if _sql():
        import gamelog_store
        athlete = gamelog_store.get_athlete_id(
            espn_sport, espn_league, player_name, team_ids=team_ids,
            ttl_hours=ttl_hours)
        return athlete["id"] if athlete else None
    # Default (no team_ids) keeps the legacy 4-part key so existing caches +
    # seed_athlete_id stay valid; a provided team_ids extends the key so
    # different-team same-name players don't collide.
    if team_ids:
        team_key = "|".join(sorted(str(t) for t in team_ids if t))
        path = _cache_key("athlete_id", espn_sport, espn_league,
                          player_name.lower(), team_key)
    else:
        path = _cache_key("athlete_id", espn_sport, espn_league,
                          player_name.lower())
    cached = _read_cache_file(path)
    if cached is not None:
        aid = cached.get("id")
        # A missing id is trusted only for the short negative TTL.
        if _is_fresh(path, ttl_hours if aid else NEGATIVE_TTL_HOURS):
            return aid
    athlete = search_athlete(espn_sport, espn_league, player_name,
                             team_ids=team_ids)
    aid = athlete["id"] if athlete else None
    with open(path, "w") as f:
        json.dump({"id": aid}, f)
    return aid


def seed_athlete_id(espn_sport, espn_league, player_name, athlete_id):
    """Pre-populate the athlete-id cache with a KNOWN id.

    When a caller already holds an authoritative ESPN athlete id (e.g. from the
    season statistics listing used to build the calibration pool), writing it
    here lets `cached_athlete_id` resolve the exact player and skip
    `search_athlete`'s lossy first-name-match. That matters for a broad pool
    where common names would otherwise silently resolve to the wrong athlete and
    corrupt the fit. Written under the same key/format `cached_athlete_id` reads.
    No-op when `athlete_id` is falsy (nothing authoritative to pin)."""
    if not athlete_id:
        return
    # Under SQL, cached_athlete_id reads from the durable gamelog_store cache, so
    # a local-file seed would be a silent no-op — seed SQL instead to match.
    if _sql():
        import gamelog_store
        gamelog_store.seed_athlete_id(
            espn_sport, espn_league, player_name, athlete_id)
        return
    path = _cache_key("athlete_id", espn_sport, espn_league, player_name.lower())
    with open(path, "w") as f:
        json.dump({"id": str(athlete_id)}, f)


def cached_gamelog(espn_sport, espn_league, athlete_id, ttl_hours=24 * 30,
                   season_year=None, player_name=None):
    """
    Cached gamelog fetch. Historical games are immutable, so a long TTL
    (default 30 days) is safe; the most recent games may lag by up to
    that window.

    When `season_year` is provided, ESPN is queried for that specific
    season and the cache file is keyed by season so different seasons
    don't collide.

    `player_name` is accepted for call-site compatibility but no longer used
    (MLB is warehouse-only post-P6; this path serves NBA/NFL).

    Set ODI_GAMELOG_TTL_HOURS env var to override (e.g., "8760" for a year).
    """
    if _sql():
        import gamelog_store
        return gamelog_store.get_gamelog(
            espn_sport, espn_league, athlete_id, season_year=season_year,
            ttl_hours=ttl_hours, player_name=player_name)
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
    cached = _read_cache_file(path)
    if cached is not None:
        # An empty gamelog (no data OR a swallowed fetch error) is trusted only
        # for the short negative TTL, so a transient miss recovers in minutes
        # instead of sticking for the full success TTL.
        if _is_fresh(path, ttl_hours if cached else NEGATIVE_TTL_HOURS):
            return cached
    gamelog = get_athlete_gamelog(espn_sport, espn_league, athlete_id,
                                  season_year=season_year) or []
    # (MLB is warehouse-only post-P6; NBA/NFL have no pitcher/synth fallback.)
    with open(path, "w") as f:
        json.dump(gamelog, f)
    return gamelog
