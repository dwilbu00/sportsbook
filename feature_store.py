"""Materialized per-game matchup-feature cache for FAST repeat backtests.

The odds backtest's slow phase is the prewarm: building build_matchup_features for
every game (as-of pitcher/offense/park lookups, SQL round-trips). Those features are
DETERMINISTIC per (sport, date, home, away) given the as-of warehouse and are
SEPARABLE from grading (built up front, then consumed by the analyzers). So they can
be materialized once and reused across every subsequent backtest.

Increment 1 (this module) is the LOCAL layer: a per-(sport, season) JSON cache with a
VERSION check so a stale local copy is never used. The version is a warehouse-derived
marker (game_count:max_date for the season) — it bumps when a season's games are
added / completed (2026) or re-ingested, so stable past seasons (2024-25) cache
forever while the current season auto-refreshes. Increment 2 adds an Azure SQL gold
table as the durable source of truth that this local cache syncs from.

fit==serve is preserved: we cache the EXACT output of the live build_matchup_features
(captured with the additive keys surfaced), so a cached backtest grades what
production serves. On any doubt, `clear()` / a version bump forces a rebuild.
"""
import json
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".feature_cache")


def season_version(games):
    """Version marker for a season's feature cache, derived from the games in scope:
    ``"<game_count>:<max_date>"``. Bumps when games are added/completed or re-ingested
    (count or max date changes) so the current season auto-refreshes and stable past
    seasons stay cached. ``games`` = the loaded schedule dicts (need a 'date' key)."""
    dates = [g.get("date") for g in (games or []) if g and g.get("date")]
    return f"{len(games or [])}:{max(dates) if dates else ''}"


def _path(sport_key, season):
    return os.path.join(CACHE_DIR, f"{sport_key}_{season}.json")


def _json_default(o):
    """Coerce numpy scalars (and anything with .item()) to native JSON so a feature
    dict carrying numpy floats still serializes."""
    item = getattr(o, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def load(sport_key, season, version):
    """Return the cached ``{(date, home, away): feature_dict}`` for the season IFF a
    local cache exists AND its stored version matches ``version``; else None (a miss
    or a stale cache — the caller then recomputes). Never raises."""
    try:
        with open(_path(sport_key, season), encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict) or blob.get("version") != version:
        return None
    rows = blob.get("rows")
    if not isinstance(rows, list):
        return None
    out = {}
    for row in rows:
        try:
            key, feats = row
            out[(key[0], key[1], key[2])] = feats
        except (ValueError, TypeError, IndexError):
            continue
    return out


def save(sport_key, season, version, features_map):
    """Persist ``{(date, home, away): feature_dict}`` + ``version`` atomically.
    Best-effort: returns True on success, False on any failure (a cache write must
    never break a backtest)."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        blob = {"version": version,
                "rows": [[list(k), v] for k, v in (features_map or {}).items()]}
        path = _path(sport_key, season)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, default=_json_default)
        os.replace(tmp, path)   # atomic on the same filesystem
        return True
    except (OSError, TypeError, ValueError):
        return False


def clear(sport_key=None, season=None):
    """Delete cached files (all, per-sport, or one season). Returns the count removed.
    The hard rebuild escape hatch for a values-only re-ingest a version wouldn't
    catch."""
    removed = 0
    try:
        names = os.listdir(CACHE_DIR)
    except OSError:
        return 0
    for name in names:
        if not name.endswith(".json"):
            continue
        if sport_key and not name.startswith(f"{sport_key}_"):
            continue
        if season is not None and name != f"{sport_key}_{season}.json":
            continue
        try:
            os.remove(os.path.join(CACHE_DIR, name))
            removed += 1
        except OSError:
            pass
    return removed
