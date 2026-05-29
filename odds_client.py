"""
Client for The Odds API (https://the-odds-api.com)
Fetches upcoming game odds from multiple sportsbooks.
Includes file-based caching to avoid redundant API calls.
"""

import hashlib
import json
import os
import time

import requests


BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_MAX_AGE = 3600  # 1 hour in seconds

# Track remaining API credits (updated on each live API call)
_remaining_credits = None


def _ensure_cache_dir():
    """Create the cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key(*parts):
    """Generate a cache filename from variable key parts."""
    raw = "_".join(str(p) for p in parts if p)
    safe = hashlib.md5(raw.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe}.json")


def _read_cache(cache_path):
    """
    Read cached data if it exists and is less than CACHE_MAX_AGE seconds old.

    Returns:
        tuple: (data, age_seconds) if cache hit, (None, None) if miss
    """
    if not os.path.exists(cache_path):
        return None, None

    try:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        cached_at = cached.get("cached_at", 0)
        age = time.time() - cached_at
        if age < CACHE_MAX_AGE:
            return cached.get("data"), age
    except (json.JSONDecodeError, KeyError, OSError):
        pass

    return None, None


def _read_cache_expired(cache_path):
    """Read cached data even if expired. Used as fallback when API credits run out."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        return cached.get("data")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _write_cache(cache_path, data):
    """Write data to cache with a timestamp."""
    _ensure_cache_dir()
    with open(cache_path, "w") as f:
        json.dump({"cached_at": time.time(), "data": data}, f)


def get_upcoming_events(api_key, sport):
    """
    Fetch upcoming events for a sport (FREE - no credit cost).
    Use this to show the user available games before spending credits on odds.

    Returns:
        list: List of event dicts with id, commence_time, home_team, away_team
    """
    cache_path = _cache_key("events", sport)
    cached, age = _read_cache(cache_path)
    if cached is not None:
        print(f"  [Cache] Using cached events ({int(age)}s old, expires in {int(CACHE_MAX_AGE - age)}s)")
        return cached

    url = f"{BASE_URL}/sports/{sport}/events"
    params = {"apiKey": api_key}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    global _remaining_credits
    remaining = resp.headers.get("x-requests-remaining", "?")
    if remaining != "?":
        _remaining_credits = int(remaining)
    print(f"  [Odds API] Events endpoint (FREE). Credits remaining: {remaining}")

    data = resp.json()
    _write_cache(cache_path, data)
    return data


def get_event_odds(api_key, sport, event_id, regions="us", markets="h2h", bookmakers=None):
    """
    Fetch odds for a single event.
    Cost: number of markets x number of regions.

    Parameters:
        api_key (str): The Odds API key
        sport (str): Sport key
        event_id (str): Specific event ID
        regions (str): Comma-separated regions
        markets (str): Comma-separated markets (h2h, spreads, totals)
        bookmakers (list): Optional bookmaker filter

    Returns:
        dict: Single game dict with odds data
    """
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("event_odds", sport, event_id, regions, markets, books_key)
    cached, age = _read_cache(cache_path)
    if cached is not None:
        print(f"    [Cache] Using cached odds for {event_id} ({int(age)}s old)")
        return cached

    url = f"{BASE_URL}/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Fall back to expired cache if credits exhausted (401/429)
        if resp.status_code in (401, 429):
            expired = _read_cache_expired(cache_path)
            if expired is not None:
                print(f"  [Odds API] Credits exhausted — using expired cache for {event_id}")
                return expired
        raise

    global _remaining_credits
    cost = resp.headers.get("x-requests-last", "?")
    remaining = resp.headers.get("x-requests-remaining", "?")
    if remaining != "?":
        _remaining_credits = int(remaining)
    print(f"  [Odds API] Event odds cost: {cost} credit(s). Remaining: {remaining}")

    data = resp.json()
    _write_cache(cache_path, data)
    return data


def get_upcoming_odds(api_key, sport, regions="us", markets="h2h,spreads,totals", bookmakers=None):
    """
    Fetch upcoming game odds for a given sport.

    Parameters:
        api_key (str): The Odds API key
        sport (str): Sport key (e.g., 'basketball_nba')
        regions (str): Comma-separated regions (us, us2, uk, au, eu)
        markets (str): Comma-separated markets (h2h, spreads, totals)
        bookmakers (list): Optional list of specific bookmaker keys to filter

    Returns:
        list: List of game dicts with odds data
    """
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("all_odds", sport, regions, markets, books_key)
    cached, age = _read_cache(cache_path)
    if cached is not None:
        print(f"  [Cache] Using cached odds ({int(age)}s old, expires in {int(CACHE_MAX_AGE - age)}s)")
        return cached

    url = f"{BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    global _remaining_credits
    remaining = resp.headers.get("x-requests-remaining", "?")
    if remaining != "?":
        _remaining_credits = int(remaining)
    print(f"  [Odds API] Requests remaining this month: {remaining}")

    data = resp.json()
    _write_cache(cache_path, data)
    return data


def get_remaining_credits():
    """Return the last known remaining credits, or None if unknown."""
    return _remaining_credits


def is_event_cached(sport, event_id, regions="us", markets="h2h", bookmakers=None):
    """Check if odds for a specific event/market combo are cached and fresh."""
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("event_odds", sport, event_id, regions, markets, books_key)
    cached, _ = _read_cache(cache_path)
    return cached is not None


def parse_game_odds(game):
    """
    Parse a single game's odds into a structured dict.

    Returns:
        dict with keys: game_id, home_team, away_team, commence_time,
                        moneyline, spreads, totals
    """
    result = {
        "game_id": game["id"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "commence_time": game["commence_time"],
        "moneyline": {},
        "spreads": {},
        "totals": {},
    }

    for bookmaker in game.get("bookmakers", []):
        book_key = bookmaker["key"]
        book_title = bookmaker["title"]

        for market in bookmaker.get("markets", []):
            market_key = market["key"]
            outcomes = market["outcomes"]

            if market_key == "h2h":
                for outcome in outcomes:
                    team = outcome["name"]
                    price = outcome["price"]
                    if team not in result["moneyline"]:
                        result["moneyline"][team] = []
                    result["moneyline"][team].append({
                        "book": book_title,
                        "price": price,
                        "implied_prob": american_to_implied_prob(price),
                    })

            elif market_key == "spreads":
                for outcome in outcomes:
                    team = outcome["name"]
                    point = outcome.get("point", 0)
                    price = outcome["price"]
                    if team not in result["spreads"]:
                        result["spreads"][team] = []
                    result["spreads"][team].append({
                        "book": book_title,
                        "spread": point,
                        "price": price,
                    })

            elif market_key == "totals":
                for outcome in outcomes:
                    label = outcome["name"]  # "Over" or "Under"
                    point = outcome.get("point", 0)
                    price = outcome["price"]
                    if label not in result["totals"]:
                        result["totals"][label] = []
                    result["totals"][label].append({
                        "book": book_title,
                        "line": point,
                        "price": price,
                    })

    return result


def american_to_implied_prob(american_odds):
    """
    Convert American odds to implied probability (0-1).
    Includes the vig so raw probabilities from all outcomes sum > 1.
    """
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    else:
        return 1 + (american_odds / 100)


def consensus_odds(team_odds_list):
    """
    Calculate consensus (average) implied probability across all bookmakers.
    
    Parameters:
        team_odds_list (list): List of dicts with 'implied_prob' key
    
    Returns:
        float: Average implied probability
    """
    if not team_odds_list:
        return 0.0
    probs = [o["implied_prob"] for o in team_odds_list]
    return sum(probs) / len(probs)


PLAYER_PROPS_BY_SPORT = {
    "basketball_nba": ["player_points", "player_assists", "player_rebounds"],
    "americanfootball_nfl": ["player_anytime_td", "player_rush_yds", "player_pass_yds"],
    "baseball_mlb": ["batter_hits", "pitcher_strikeouts", "pitcher_outs", "batter_strikeouts"],
}

PROP_LABELS = {
    "player_points": "Points",
    "player_assists": "Assists",
    "player_rebounds": "Rebounds",
    "player_anytime_td": "Anytime TD",
    "player_rush_yds": "Rushing Yards",
    "player_pass_yds": "Passing Yards",
    "batter_hits": "Hits",
    "pitcher_strikeouts": "Pitcher Strikeouts",
    "pitcher_outs": "Pitcher Outs",
    "batter_strikeouts": "Batter Strikeouts",
}


def parse_player_props(game_data):
    """
    Parse player prop odds from an event odds response.
    Props have outcomes with 'description' (player name), 'name' (Over/Under),
    'price' (American odds), and 'point' (the line).

    Returns:
        dict: Structured prop data with game info and per-player lines
    """
    result = {
        "game_id": game_data["id"],
        "home_team": game_data["home_team"],
        "away_team": game_data["away_team"],
        "commence_time": game_data.get("commence_time"),
        "sport_key": game_data.get("sport_key"),
        "props": {},
    }

    for bookmaker in game_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            market_key = market["key"]
            if market_key not in PROP_LABELS:
                continue

            if market_key not in result["props"]:
                result["props"][market_key] = {}

            # Group outcomes by player description
            players = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                if not player:
                    continue
                if player not in players:
                    players[player] = {}
                side = outcome["name"]  # "Over" or "Under"
                players[player][side] = outcome

            for player, sides in players.items():
                # Skip if already recorded from an earlier bookmaker
                if player in result["props"][market_key]:
                    continue

                if "Over" not in sides or "Under" not in sides:
                    continue

                over = sides["Over"]
                under = sides["Under"]

                if market_key == "player_anytime_td":
                    line = 0.5
                else:
                    line = over.get("point", under.get("point", 0))

                over_price = over["price"]
                under_price = under["price"]

                result["props"][market_key][player] = {
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "over_implied": american_to_implied_prob(over_price),
                    "under_implied": american_to_implied_prob(under_price),
                }

    return result
