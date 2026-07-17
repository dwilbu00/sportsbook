"""
Shared on-disk store for historical closing-line odds from The Odds API.

The expensive historical pulls (10 credits x markets x regions each) are run
once by backfill_historical_odds.py and persisted here so that the backtest
(and any re-runs) cost zero additional credits.

Layout:
    SPORTSBOOK_ODDS/historical_odds/<sport_key>.json

Schema:
{
  "sport_key":  "basketball_nba",
  "bookmaker":  "draftkings",
  "markets":    "h2h,spreads,totals",
  "updated":    "2026-06-29T12:00:00Z",
  "games": {
     "<commence_date>|<away_team> @ <home_team>": {
        "commence_time":      "2024-01-16T00:30:00Z",
        "snapshot_timestamp": "2024-01-16T00:25:00Z",
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "moneyline": {...}, "spreads": {...}, "totals": {...}
     }, ...
  }
}

The per-game moneyline/spreads/totals blocks are exactly what
odds_client.parse_game_odds() returns, so downstream code can consume them
without any extra parsing.
"""
import json
import os
from datetime import datetime, timezone

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "historical_odds")


def store_path(sport_key, label=""):
    """Path to a sport's store. An optional label writes a separate file
    (e.g. label='morning' -> baseball_mlb__morning.json) so different snapshot
    times (morning vs closing) don't overwrite each other."""
    suffix = f"__{label}" if label else ""
    return os.path.join(STORE_DIR, f"{sport_key}{suffix}.json")


def game_key(commence_time, home_team, away_team):
    """Stable key for a game: '<YYYY-MM-DD>|<away> @ <home>' (UTC date)."""
    date10 = (commence_time or "")[:10]
    return f"{date10}|{away_team} @ {home_team}"


def load_store(sport_key, label=""):
    """Load the store for a sport. Returns a dict; 'games' is {} if missing."""
    path = store_path(sport_key, label)
    if not os.path.exists(path):
        return {"sport_key": sport_key, "games": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return {"sport_key": sport_key, "games": {}}
    blob.setdefault("games", {})
    return blob


def save_store(sport_key, store, label=""):
    """Persist the store; creates historical_odds/ if needed."""
    os.makedirs(STORE_DIR, exist_ok=True)
    store["sport_key"] = sport_key
    store["updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with open(store_path(sport_key, label), "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
