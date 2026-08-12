"""
Client for The Odds API (https://the-odds-api.com)
Fetches upcoming game odds from multiple sportsbooks.
Includes file-based caching to avoid redundant API calls.
"""

import hashlib
import json
import os
import random
import statistics
import time
from datetime import datetime

import requests


BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_MAX_AGE = 3600  # 1 hour in seconds
# When credits are exhausted we may fall back to expired cache for interactive
# analysis, but only up to this age. Beyond it, pricing bets against the stale
# line is worse than failing, so we surface the error instead.
STALE_CACHE_MAX_AGE = 6 * 3600  # 6 hours in seconds

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
    """Read cached data even if expired, returning (data, age_seconds).

    Used as a fallback when API credits run out. The caller decides whether the
    age is acceptable (see STALE_CACHE_MAX_AGE) rather than serving arbitrarily
    old odds. Returns (None, None) on miss or when the cache lacks a timestamp.
    """
    if not os.path.exists(cache_path):
        return None, None
    try:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        data = cached.get("data")
        cached_at = cached.get("cached_at", 0)
        if not cached_at:
            return None, None
        return data, time.time() - cached_at
    except (json.JSONDecodeError, KeyError, OSError):
        return None, None


def _is_quota_error(resp):
    """True when an error response is exhausted-usage / rate-limit rather than
    an invalid API key.

    The Odds API returns 429 for rate limits and 401 for BOTH an invalid key
    and exhausted usage credits, distinguished only by the response body. We
    serve (bounded-age) stale cache on the former but never the latter — an
    invalid-key auth error must surface, not be masked by old odds.
    """
    if resp is None:
        return False
    if resp.status_code == 429:
        return True
    if resp.status_code != 401:
        return False
    try:
        body = resp.json()
        code = body.get("error_code") or ""
        message = body.get("message") or ""
    except (ValueError, AttributeError):
        code, message = "", (getattr(resp, "text", "") or "")
    text = f"{code} {message}".upper()
    return any(token in text for token in ("USAGE", "CREDIT", "QUOTA", "LIMIT"))


def _write_cache(cache_path, data):
    """Write data to cache with a timestamp."""
    _ensure_cache_dir()
    with open(cache_path, "w") as f:
        json.dump({"cached_at": time.time(), "data": data}, f)


def _get_with_retry(url, params, timeout=30, max_retries=5, backoff_base=1.5):
    """
    GET a URL with automatic retry + exponential backoff on rate-limit (429)
    and transient server (5xx) errors.

    When many events are analyzed at once the concurrent requests can trip The
    Odds API rate limiter (HTTP 429). Rather than failing the whole analysis,
    we back off and retry the individual request, which also self-throttles the
    burst of parallel calls.

    Returns the final ``requests.Response`` (the caller should still invoke
    ``raise_for_status()`` so existing error handling / cache fallbacks run).
    """
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            # Transient network fault (connection reset, read timeout, DNS
            # blip): no HTTP response was received and no credit was spent, so
            # it is safe to retry. Without this the whole analysis — and, worse,
            # a paid closing-line capture — fails on a momentary blip.
            if attempt < max_retries:
                delay = backoff_base ** attempt + random.uniform(0, 0.5)
                print(
                    f"  [Odds API] network error ({type(exc).__name__}) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue
            raise
        is_retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
        if is_retryable and attempt < max_retries:
            # Honor the server's Retry-After hint when present, otherwise use
            # exponential backoff with a little jitter to de-sync parallel calls.
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else (
                    backoff_base ** attempt + random.uniform(0, 0.5)
                )
            except (TypeError, ValueError):
                delay = backoff_base ** attempt + random.uniform(0, 0.5)
            print(
                f"  [Odds API] HTTP {resp.status_code} — retrying in "
                f"{delay:.1f}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            continue
        return resp
    return resp


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

    resp = _get_with_retry(url, params)
    resp.raise_for_status()

    global _remaining_credits
    remaining = resp.headers.get("x-requests-remaining", "?")
    if remaining != "?":
        _remaining_credits = int(remaining)
    print(f"  [Odds API] Events endpoint (FREE). Credits remaining: {remaining}")

    data = resp.json()
    _write_cache(cache_path, data)
    return data


def get_event_odds(api_key, sport, event_id, regions="us", markets="h2h",
                   bookmakers=None, force_refresh=False):
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
        force_refresh (bool): Skip the one-hour cache (for closing snapshots)

    Returns:
        dict: Single game dict with odds data
    """
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("event_odds", sport, event_id, regions, markets, books_key)
    if not force_refresh:
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
        resp = _get_with_retry(url, params)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        # A bounded-age expired cache is useful for interactive analysis when
        # credits are exhausted / rate-limited, but:
        #   • never for a forced refresh (would mislabel a stale line as a fresh
        #     closing snapshot),
        #   • never for an invalid-key auth error (that must surface), and
        #   • never beyond STALE_CACHE_MAX_AGE (pricing bets against a very old
        #     line is worse than failing).
        # When served, the payload is flagged so the caller can warn the user
        # and avoid trusting the edge.
        if not force_refresh and _is_quota_error(resp):
            expired, age = _read_cache_expired(cache_path)
            if expired is not None and age is not None and age <= STALE_CACHE_MAX_AGE:
                print(f"  [Odds API] Credits exhausted — using stale cache for "
                      f"{event_id} ({int(age)}s old)")
                if isinstance(expired, dict):
                    expired = dict(expired)
                    expired["_stale_cache"] = True
                    expired["_stale_age_seconds"] = int(age)
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
    # Durable archive of the freshly fetched snapshot (roadmap 0.4). Best-effort,
    # lazy-imported to avoid an import cycle, and only on a genuine live fetch
    # (cache hits / stale-cache fallbacks return earlier), so it is free.
    try:
        import warehouse
        warehouse.capture_event_odds(
            sport, event_id, regions, markets, bookmakers, data)
    except Exception:
        pass
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

    resp = _get_with_retry(url, params)
    resp.raise_for_status()

    global _remaining_credits
    remaining = resp.headers.get("x-requests-remaining", "?")
    if remaining != "?":
        _remaining_credits = int(remaining)
    print(f"  [Odds API] Requests remaining this month: {remaining}")

    data = resp.json()
    _write_cache(cache_path, data)
    return data


# ─── Historical odds (paid plan) ──────────────────────────────────────────
#
# Historical snapshots are IMMUTABLE — the odds at a past timestamp never
# change — so they are cached permanently (no TTL) rather than for 1 hour.
# Cost: events = 1 credit; odds/event-odds = 10 x markets x regions.
# The response is wrapped in {timestamp, previous_timestamp, next_timestamp,
# data}; we unwrap `data` so the existing parse_* helpers work unchanged, and
# return the snapshot timestamp alongside it.

HISTORICAL_BASE_URL = f"{BASE_URL}/historical"


def _normalize_snapshot_date(date):
    """
    Coerce a timestamp to the full ISO8601 form the historical API requires
    (YYYY-MM-DDTHH:MM:SSZ). ESPN commence times omit seconds (e.g.
    '2026-06-28T00:40Z'), which the API rejects with HTTP 422. Returns the
    input unchanged if it doesn't look like a date-time.
    """
    import re
    if not date:
        return date
    d = date.strip()
    if d.endswith("Z"):
        d = d[:-1]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?", d)
    if not m:
        return date
    return f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4) or '00'}Z"


def _read_cache_permanent(cache_path):
    """Read cached data with no expiry. Used for immutable historical snapshots."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            cached = json.load(f)
        return cached.get("data")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _update_credits_from_headers(resp, label):
    """Record remaining credits and print the cost of the last call."""
    global _remaining_credits
    cost = resp.headers.get("x-requests-last", "?")
    remaining = resp.headers.get("x-requests-remaining", "?")
    if remaining != "?":
        _remaining_credits = int(remaining)
    print(f"  [Odds API] {label} cost: {cost} credit(s). Remaining: {remaining}")


def get_historical_events(api_key, sport, date, regions=None):
    """
    List events as they appeared at a past timestamp (cost: 1 credit).
    Use this to discover historical event IDs for get_historical_event_odds().

    Parameters:
        date (str): ISO8601 snapshot timestamp, e.g. '2023-11-29T22:45:00Z'.
                    The API returns the closest snapshot at or before `date`.

    Returns:
        tuple: (events_list, snapshot_timestamp). Empty list if none found.
    """
    date = _normalize_snapshot_date(date)
    cache_path = _cache_key("hist_events", sport, date)
    cached = _read_cache_permanent(cache_path)
    if cached is not None:
        return cached.get("data", []), cached.get("timestamp")

    url = f"{HISTORICAL_BASE_URL}/sports/{sport}/events"
    params = {"apiKey": api_key, "date": date}

    resp = _get_with_retry(url, params)
    resp.raise_for_status()
    _update_credits_from_headers(resp, "Historical events")

    body = resp.json()
    snapshot = {"data": body.get("data", []), "timestamp": body.get("timestamp")}
    _write_cache(cache_path, snapshot)
    return snapshot["data"], snapshot["timestamp"]


def get_historical_odds(api_key, sport, date, regions="us",
                        markets="h2h,spreads,totals", bookmakers=None):
    """
    Fetch a featured-market (h2h/spreads/totals) odds snapshot at a past
    timestamp. Cost: 10 x markets x regions.

    To approximate the CLOSING LINE, pass date = the game's commence_time;
    the API returns the latest snapshot at or before that moment.

    Returns:
        tuple: (games_list, snapshot_timestamp). Each game matches the live
               /odds schema, so parse_game_odds() works unchanged.
    """
    date = _normalize_snapshot_date(date)
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("hist_odds", sport, date, regions, markets, books_key)
    cached = _read_cache_permanent(cache_path)
    if cached is not None:
        return cached.get("data", []), cached.get("timestamp")

    url = f"{HISTORICAL_BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "date": date,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    resp = _get_with_retry(url, params)
    resp.raise_for_status()
    _update_credits_from_headers(resp, "Historical odds")

    body = resp.json()
    snapshot = {"data": body.get("data", []), "timestamp": body.get("timestamp")}
    _write_cache(cache_path, snapshot)
    return snapshot["data"], snapshot["timestamp"]


def get_historical_event_odds(api_key, sport, event_id, date, regions="us",
                              markets="h2h", bookmakers=None):
    """
    Fetch a single event's odds snapshot at a past timestamp, including
    additional markets (player props, alternate lines). Available for
    additional markets after 2023-05-03. Cost: 10 x markets x regions.

    Returns:
        tuple: (game_dict_or_None, snapshot_timestamp). The game dict matches
               the live event-odds schema, so parse_player_props() and
               parse_game_odds() work unchanged. Returns (None, None) if the
               event had expired (HTTP 404) at the requested timestamp.
    """
    date = _normalize_snapshot_date(date)
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("hist_event_odds", sport, event_id, date, regions,
                            markets, books_key)
    cached = _read_cache_permanent(cache_path)
    if cached is not None:
        return cached.get("data"), cached.get("timestamp")

    url = f"{HISTORICAL_BASE_URL}/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "date": date,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    try:
        resp = _get_with_retry(url, params)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        # Event had expired (completed/cancelled) at this timestamp.
        if resp.status_code == 404:
            snapshot = {"data": None, "timestamp": None}
            _write_cache(cache_path, snapshot)
            return None, None
        raise

    _update_credits_from_headers(resp, "Historical event odds")

    body = resp.json()
    snapshot = {"data": body.get("data"), "timestamp": body.get("timestamp")}
    _write_cache(cache_path, snapshot)
    return snapshot["data"], snapshot["timestamp"]


def is_historical_event_cached(sport, event_id, date, regions="us",
                               markets="h2h", bookmakers=None):
    """True if this exact historical event-odds snapshot is already in the
    permanent cache — a re-fetch would return instantly and cost 0 credits.

    Mirrors the cache-key construction in get_historical_event_odds so a
    budget-guarded caller can treat cache hits (and cached 404s) as free instead
    of charging them against a credit cap."""
    date = _normalize_snapshot_date(date)
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("hist_event_odds", sport, event_id, date, regions,
                            markets, books_key)
    return os.path.exists(cache_path)


def devig_two_way(implied_a, implied_b):
    """
    Remove the bookmaker margin (vig) from a two-outcome market by normalizing
    the raw implied probabilities so they sum to 1.

    The live american_to_implied_prob() keeps the vig in, so raw two-sided
    probabilities sum to >1 and any edge computed against them is biased low.
    Use this to turn closing-line prices into a true market probability.

    Returns:
        tuple: (fair_prob_a, fair_prob_b). Returns (0.5, 0.5) on degenerate input.
    """
    total = implied_a + implied_b
    if total <= 0:
        return 0.5, 0.5
    return implied_a / total, implied_b / total


def get_remaining_credits():
    """Return the last known remaining credits, or None if unknown."""
    return _remaining_credits


def reset_remaining_credits():
    """Forget the last known remaining-credit balance (e.g. after an API-key
    change) so the pre-flight budget gate treats it as unknown and lets the next
    analysis run. The first live call under the new key repopulates the count
    from the ``x-requests-remaining`` response header. Without this, an exhausted
    old key leaves this module global at 0 and the gate ``st.stop()``s before any
    call can refresh it — deadlocking a freshly-entered, funded key."""
    global _remaining_credits
    _remaining_credits = None


def is_event_cached(sport, event_id, regions="us", markets="h2h", bookmakers=None):
    """Check if odds for a specific event/market combo are cached and fresh.

    Uses the file mtime instead of reading + JSON-parsing the whole payload
    (40-128 KB): the credit-cost estimator calls this per selected game on every
    rerun, and it only needs a yes/no. Cache files are written once and never
    modified, so mtime is a faithful proxy for the stored ``cached_at``."""
    books_key = ",".join(sorted(bookmakers)) if bookmakers else ""
    cache_path = _cache_key("event_odds", sport, event_id, regions, markets, books_key)
    try:
        return (time.time() - os.path.getmtime(cache_path)) < CACHE_MAX_AGE
    except OSError:
        return False


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


def _complete_featured_offer(game, bookmaker, market_key):
    """Return one complete two-sided featured offer from a bookmaker."""
    market = next(
        (entry for entry in bookmaker.get("markets", [])
         if entry.get("key") == market_key),
        None,
    )
    if not market:
        return None

    if market_key in ("h2h", "spreads"):
        side_names = (game.get("home_team"), game.get("away_team"))
    else:
        side_names = ("Over", "Under")

    offer = {}
    for side_name in side_names:
        outcome = next(
            (entry for entry in market.get("outcomes", [])
             if entry.get("name") == side_name),
            None,
        )
        if not outcome:
            return None
        price = outcome.get("price")
        if (not isinstance(price, (int, float)) or isinstance(price, bool)
                or abs(price) < 100):
            return None
        point = outcome.get("point")
        if market_key != "h2h" and not isinstance(point, (int, float)):
            return None
        offer[side_name] = {"line": point, "price": price}

    first, second = (offer[name] for name in side_names)
    if market_key == "spreads" and abs(first["line"] + second["line"]) > 1e-9:
        return None
    if market_key == "totals" and abs(first["line"] - second["line"]) > 1e-9:
        return None
    return offer


def _decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def build_market_comparisons(
        game, primary_bookmaker="draftkings", min_peer_books=3):
    """Compare one book's featured offers with complete peer-book medians.

    This is line-shopping context only. It deliberately returns no probability
    or edge adjustment, so callers cannot accidentally blend it into the model.
    """
    comparisons = {"h2h": {}, "spreads": {}, "totals": {}}
    primary_book = next(
        (book for book in game.get("bookmakers", [])
         if book.get("key") == primary_bookmaker),
        None,
    )
    if not primary_book:
        return comparisons

    for market_key in comparisons:
        primary_offer = _complete_featured_offer(
            game, primary_book, market_key)
        if not primary_offer:
            continue
        peer_offers = []
        for book in game.get("bookmakers", []):
            if book is primary_book or book.get("key") == primary_bookmaker:
                continue
            offer = _complete_featured_offer(game, book, market_key)
            if offer:
                peer_offers.append(offer)
        if len(peer_offers) < min_peer_books:
            continue

        for side_name, primary_side in primary_offer.items():
            peer_decimal_price = statistics.median(
                american_to_decimal(offer[side_name]["price"])
                for offer in peer_offers
            )
            primary_decimal_price = american_to_decimal(primary_side["price"])
            peer_line = None
            line_advantage = None
            key_numbers = []
            if market_key != "h2h":
                peer_line = statistics.median(
                    offer[side_name]["line"] for offer in peer_offers)
                if market_key == "spreads":
                    line_advantage = primary_side["line"] - peer_line
                    key_numbers = sorted({
                        abs(key_number)
                        for key_number in (-7.0, -3.0, 3.0, 7.0)
                        if min(primary_side["line"], peer_line)
                        <= key_number
                        <= max(primary_side["line"], peer_line)
                    })
                elif side_name == "Over":
                    line_advantage = peer_line - primary_side["line"]
                else:
                    line_advantage = primary_side["line"] - peer_line

            price_advantage = primary_decimal_price - peer_decimal_price
            comparisons[market_key][side_name] = {
                "primary_line": primary_side["line"],
                "primary_price": primary_side["price"],
                "peer_median_line": peer_line,
                "peer_median_price": _decimal_to_american(peer_decimal_price),
                "peer_count": len(peer_offers),
                "line_advantage": line_advantage,
                "price_advantage": price_advantage,
                "dominates_peer_offer": (
                    line_advantage is not None
                    and line_advantage > 1e-9
                    and price_advantage >= -1e-9
                ),
                "key_numbers": key_numbers,
            }
    return comparisons


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
    "baseball_mlb": ["batter_hits", "pitcher_strikeouts", "pitcher_outs", "batter_strikeouts", "pitcher_earned_runs", "batter_total_bases", "batter_rbis"],
}

# Mapping of standard player prop market keys → their alt-line market key on
# The Odds API. Entries with no known alt market are omitted (e.g. anytime TD
# is a yes/no with no line).
PLAYER_PROP_ALTS_BY_SPORT = {
    "basketball_nba": {
        "player_points": "player_points_alternate",
        "player_assists": "player_assists_alternate",
        "player_rebounds": "player_rebounds_alternate",
    },
    "americanfootball_nfl": {
        "player_rush_yds": "player_rush_yds_alternate",
        "player_pass_yds": "player_pass_yds_alternate",
    },
    "baseball_mlb": {
        "batter_hits": "batter_hits_alternate",
        "pitcher_strikeouts": "pitcher_strikeouts_alternate",
    },
}

# Team-level alternate markets (per event, regardless of sport).
TEAM_ALT_MARKETS = {
    "spreads": "alternate_spreads",
    "totals": "alternate_totals",
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
    "pitcher_earned_runs": "Pitcher Earned Runs",
    "batter_total_bases": "Total Bases",
    "batter_rbis": "RBIs",
}


def _prop_side(market_key, side):
    """Normalize an outcome's side to Over/Under. The anytime-TD market is a
    two-way Yes/No market where Yes == Over 0.5 TDs (the player scores) and No ==
    Under; map it so the two-sided de-vig/line machinery treats it like any other
    Over/Under prop (its line is pinned to 0.5 alongside). All other markets pass
    through unchanged. If a book posts only the Yes side, the downstream
    two-sided requirement still drops it -- no fictitious one-sided market, so
    this is a strict improvement over silently discarding every anytime-TD."""
    if market_key == "player_anytime_td":
        if side == "Yes":
            return "Over"
        if side == "No":
            return "Under"
    return side


def _dk_offer(side_offers, book_key="draftkings"):
    """First offer whose book title matches DraftKings (case-insensitive), or
    None. Used to carve the executable DK price out of the multi-book set for
    staking/display (P1.1b)."""
    for offer in side_offers or []:
        if book_key in str(offer.get("book", "")).lower():
            return offer
    return None


def dk_prop_lines(game_data, prop_key, book_key="draftkings"):
    """Every line DraftKings posts for one prop market in an event-odds payload,
    WITHOUT the consensus-modal-line collapse parse_player_props applies.

    Returns a list of ``{"player", "line", "over_price", "under_price"}`` — one
    entry per (player, line) DK offers, with a side's price None when DK only
    posts the opposite side at that line. The DK closing-line CLV backfill
    (backfill_dk_clv.py) needs DK's price at the *exact* line a bet was placed
    on, which the single consensus line from parse_player_props can't provide.
    Only the DraftKings bookmaker is read (matched on key or title). Pure and
    hermetic; never raises."""
    out = []
    try:
        for bookmaker in game_data.get("bookmakers", []) or []:
            key = str(bookmaker.get("key") or "").lower()
            title = str(bookmaker.get("title") or "").lower()
            if book_key not in key and book_key not in title:
                continue
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != prop_key:
                    continue
                # (player, line) -> {"Over": price, "Under": price}
                by_pl = {}
                for outcome in market.get("outcomes", []) or []:
                    player = outcome.get("description")
                    side = _prop_side(prop_key, outcome.get("name"))
                    if not player or side not in ("Over", "Under"):
                        continue
                    line = (0.5 if prop_key == "player_anytime_td"
                            else outcome.get("point"))
                    price = outcome.get("price")
                    if line is None or price is None:
                        continue
                    by_pl.setdefault((player, line), {})[side] = price
                for (player, line), sides in by_pl.items():
                    out.append({
                        "player": player,
                        "line": line,
                        "over_price": sides.get("Over"),
                        "under_price": sides.get("Under"),
                    })
    except Exception:
        return out
    return out


def dk_game_lines(game_data, book_key="draftkings"):
    """Every TEAM-market line DraftKings posts in an event-odds payload, split by
    market — the team-market analogue of dk_prop_lines.

    Returns ``{"moneyline": [...], "spreads": [...], "totals": [...]}`` where
      moneyline -> [{"team", "price"}]                (no line)
      spreads   -> [{"team", "point", "price"}]        (signed team spread)
      totals    -> [{"side": "Over"/"Under", "point", "price"}]
    one entry per outcome DK offers. Only the DraftKings bookmaker is read
    (matched on key or title) and only the FEATURED markets (h2h/spreads/totals),
    not the '_alternate' ladders. The DK closing-line CLV backfill
    (backfill_dk_clv.py) needs DK's price at the *exact* line/side a team bet was
    placed on, which the warehouse's best-of-book / de-vigged consensus can't
    provide. Pure and hermetic; never raises."""
    out = {"moneyline": [], "spreads": [], "totals": []}
    try:
        for bookmaker in game_data.get("bookmakers", []) or []:
            key = str(bookmaker.get("key") or "").lower()
            title = str(bookmaker.get("title") or "").lower()
            if book_key not in key and book_key not in title:
                continue
            for market in bookmaker.get("markets", []) or []:
                mkey = market.get("key")
                for outcome in market.get("outcomes", []) or []:
                    name = outcome.get("name")
                    price = outcome.get("price")
                    if name is None or price is None:
                        continue
                    if mkey == "h2h":
                        out["moneyline"].append({"team": name, "price": price})
                    elif mkey == "spreads":
                        point = outcome.get("point")
                        if point is None:
                            continue
                        out["spreads"].append(
                            {"team": name, "point": point, "price": price})
                    elif mkey == "totals":
                        point = outcome.get("point")
                        if point is None or name not in ("Over", "Under"):
                            continue
                        out["totals"].append(
                            {"side": name, "point": point, "price": price})
    except Exception:
        return out
    return out


# ── De-vig consensus quality (P1.1c) ──
# Sharp books whose de-vigged prices are up-weighted when averaging the prop
# consensus. Keys are lowercased substrings matched against the book title.
# NOTE: Pinnacle is usually an EU-region book, so it rarely appears in a
# regions=us payload — the weight is inert then, but kept for when it does /
# if the fetch region widens. Circa is US (Nevada). Weights are conservative
# judgment values (this methodology change is NOT backtestable against existing
# captures, whose warehoused consensus is unweighted).
SHARP_BOOK_WEIGHTS = {"pinnacle": 3.0, "circa": 2.0}
# Drop a book's quote from the consensus when its last_update is this many
# seconds staler than the freshest quote (best-effort; only when timestamps
# parse; never trims below one surviving offer).
MAX_STALE_SECONDS = 600


def _book_weight(book_title):
    t = str(book_title).lower()
    for key, weight in SHARP_BOOK_WEIGHTS.items():
        if key in t:
            return weight
    return 1.0


def _parse_ts(value):
    """Parse an Odds API ISO-8601 timestamp (e.g. '2026-07-20T18:00:00Z') to an
    epoch float, or None when absent/unparseable. Best-effort, never raises."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(
            str(value).strip().replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _trim_stale_offers(offers, max_stale_seconds=MAX_STALE_SECONDS):
    """Drop offers whose last_update is more than max_stale_seconds behind the
    freshest offer. Offers without a parseable timestamp are always kept (we
    can't judge them); if no offer has a timestamp, all are kept. Never returns
    fewer than one offer."""
    stamped = [(offer, _parse_ts(offer.get("last_update"))) for offer in offers]
    parseable = [ts for _, ts in stamped if ts is not None]
    if not parseable:
        return offers
    cutoff = max(parseable) - max_stale_seconds
    kept = [offer for offer, ts in stamped if ts is None or ts >= cutoff]
    return kept or offers


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

    # Keep complete two-sided offers by bookmaker and line. Standard lines can
    # differ across books, so combining an Over from one line with an Under
    # from another would produce a fictitious market and an invalid de-vig.
    grouped = {}
    side_prices = {}
    for bookmaker in game_data.get("bookmakers", []):
        book_title = bookmaker.get("title") or bookmaker.get("key") or "Unknown"
        for market in bookmaker.get("markets", []):
            market_key = market["key"]
            if market_key not in PROP_LABELS:
                continue

            # Group outcomes by player AND line inside each book. A malformed
            # or alternate-like payload can contain multiple points in one
            # market; those sides must never be paired across lines.
            players = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                if not player:
                    continue
                side = _prop_side(market_key, outcome.get("name"))
                if side not in ("Over", "Under"):
                    continue
                line = (0.5 if market_key == "player_anytime_td"
                        else outcome.get("point"))
                price = outcome.get("price")
                if line is None or price is None:
                    continue
                side_prices.setdefault(market_key, {}).setdefault(
                    player, {}).setdefault(line, {}).setdefault(side, []).append({
                        "book": book_title,
                        "price": price,
                    })
                players.setdefault((player, line), {})[side] = outcome

            for (player, line), sides in players.items():
                if "Over" not in sides or "Under" not in sides:
                    continue

                over = sides["Over"]
                under = sides["Under"]

                over_price = over["price"]
                under_price = under["price"]

                grouped.setdefault(market_key, {}).setdefault(
                    player, {}).setdefault(line, []).append({
                        "book": book_title,
                        "over_price": over_price,
                        "under_price": under_price,
                        # Freshness for the staleness trim (P1.1c). Market-level
                        # is more precise than bookmaker-level; may be absent.
                        "last_update": (market.get("last_update")
                                        or bookmaker.get("last_update")),
                    })

    for market_key, by_player in grouped.items():
        result["props"][market_key] = {}
        for player, by_line in by_player.items():
            # Use the modal line as the consensus standard line. On a tie,
            # prefer the line nearest the median of the offered lines.
            lines = sorted(by_line)
            median_line = lines[len(lines) // 2]
            line = max(
                lines,
                key=lambda value: (len(by_line[value]),
                                   -abs(float(value) - float(median_line))),
            )
            offers = by_line[line]
            # De-vig each book first, drop stale quotes, then take a sharp-book-
            # weighted mean of the fair OVER probs (P1.1c). Reduces to the plain
            # arithmetic mean when no sharp book and no timestamps are present.
            survivors = _trim_stale_offers(offers)
            weighted_sum = 0.0
            weight_total = 0.0
            for offer in survivors:
                raw_over = american_to_implied_prob(offer["over_price"])
                raw_under = american_to_implied_prob(offer["under_price"])
                fo, _ = devig_two_way(raw_over, raw_under)
                w = _book_weight(offer.get("book"))
                weighted_sum += fo * w
                weight_total += w

            fair_over = weighted_sum / weight_total if weight_total else 0.5
            fair_under = 1.0 - fair_over
            executable = side_prices[market_key][player][line]
            best_over = max(
                executable["Over"], key=lambda offer: offer["price"])
            best_under = max(
                executable["Under"], key=lambda offer: offer["price"])
            # Best-price line-shopping (P1.1b, hybrid): edge/EV are measured at
            # the BEST executable price across all US books, but staking/display
            # use the DraftKings price (the user bets at DK). dk_* is None when
            # DK doesn't post this prop at the consensus line — callers fall back
            # to the best price.
            dk_over = _dk_offer(executable["Over"])
            dk_under = _dk_offer(executable["Under"])
            result["props"][market_key][player] = {
                "line": line,
                "over_price": best_over["price"],
                "under_price": best_under["price"],
                "over_book": best_over["book"],
                "under_book": best_under["book"],
                "dk_over_price": dk_over["price"] if dk_over else None,
                "dk_under_price": dk_under["price"] if dk_under else None,
                "dk_over_book": dk_over["book"] if dk_over else None,
                "dk_under_book": dk_under["book"] if dk_under else None,
                # Edge is measured against the consensus fair probability;
                # expected ROI still uses the best executable side price.
                "over_implied": fair_over,
                "under_implied": fair_under,
                "offers": offers,
                "books_sampled": len(offers),
                "over_prices_sampled": len(executable["Over"]),
                "under_prices_sampled": len(executable["Under"]),
                "market_implied_method": "two_way_devig_sharpweighted_consensus",
            }

    return result


def parse_alt_player_props(game_data):
    """
    Parse alternate player prop odds from an event odds response.

    Each prop has many lines per player (e.g. Points 5.5 / 6.5 / 7.5 / ...).
    Returns:
        dict keyed by (player_name, base_prop_key) → list of entries
              [{"line": float, "over_price": int|None, "under_price": int|None}, ...]
        sorted by line ascending.
    """
    grouped = {}  # (player, base_prop) -> {line: {"over_price": ..., "under_price": ...}}

    for bookmaker in game_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            mkey = market.get("key", "")
            if not mkey.endswith("_alternate"):
                continue
            base_prop = mkey[: -len("_alternate")]
            for outcome in market.get("outcomes", []):
                player = outcome.get("description") or outcome.get("name")
                point = outcome.get("point")
                price = outcome.get("price")
                side = outcome.get("name")  # "Over" / "Under"
                if not player or point is None or price is None:
                    continue
                key = (player, base_prop)
                if key not in grouped:
                    grouped[key] = {}
                if point not in grouped[key]:
                    grouped[key][point] = {"line": point, "over_price": None, "under_price": None}
                if side == "Over":
                    previous = grouped[key][point]["over_price"]
                    grouped[key][point]["over_price"] = (
                        price if previous is None else max(previous, price))
                elif side == "Under":
                    previous = grouped[key][point]["under_price"]
                    grouped[key][point]["under_price"] = (
                        price if previous is None else max(previous, price))

    return {
        k: sorted(v.values(), key=lambda e: e["line"])
        for k, v in grouped.items()
    }


def parse_alt_team_lines(game_data):
    """
    Parse alternate spreads/totals from an event odds response.

    Returns:
        {
          "spreads": {team_name: [{"line": float, "price": int}, ...]},
          "totals":  {"Over": [...], "Under": [...]},
        }
        with each list sorted by line ascending.
    """
    result = {"spreads": {}, "totals": {"Over": [], "Under": []}}

    for bookmaker in game_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            mkey = market.get("key", "")
            if mkey == "alternate_spreads":
                for outcome in market.get("outcomes", []):
                    team = outcome.get("name")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if team is None or point is None or price is None:
                        continue
                    result["spreads"].setdefault(team, []).append({"line": point, "price": price})
            elif mkey == "alternate_totals":
                for outcome in market.get("outcomes", []):
                    label = outcome.get("name")  # "Over" / "Under"
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if label not in ("Over", "Under") or point is None or price is None:
                        continue
                    result["totals"][label].append({"line": point, "price": price})

    for team in result["spreads"]:
        result["spreads"][team].sort(key=lambda e: e["line"])
    result["totals"]["Over"].sort(key=lambda e: e["line"])
    result["totals"]["Under"].sort(key=lambda e: e["line"])
    return result
