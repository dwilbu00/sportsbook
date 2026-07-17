"""Capture pregame closing prop prices and maintain resolved prediction logs.

Examples:
    python forward_tracker.py --capture-closing
    python forward_tracker.py --capture-closing --dry-run
    python forward_tracker.py --resolve
"""

import argparse
import json
import os
import threading
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

from odds_client import get_event_odds, get_historical_event_odds
from recalibration import (
    maintain_sport,
    mutate_prediction_log,
    prediction_log_storage,
    prediction_row_key,
    read_prediction_log,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
SPORT_ALIASES = {
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}
_capture_lock = threading.Lock()


def _load_settings():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        config = {}
    secrets_path = os.path.join(SCRIPT_DIR, ".streamlit", "secrets.toml")
    try:
        import tomllib
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
    except (ImportError, OSError, ValueError):
        secrets = {}
    api_key = (os.environ.get("ODDS_API_KEY")
               or secrets.get("ODDS_API_KEY")
               or config.get("odds_api_key"))
    return {
        "api_key": api_key,
        "bookmakers": config.get("bookmakers") or None,
    }


def _parse_utc(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    return " ".join(
        "".join(ch for ch in text if not unicodedata.combining(ch))
        .replace(".", "").replace("'", "").replace("-", " ")
        .casefold().split()
    )


def closing_event_groups(rows, now=None, window_minutes=10, sport_key=None,
                         include_attempted=False):
    """Group log rows for events beginning within the closing window."""
    now = now or datetime.now(timezone.utc)
    groups = defaultdict(list)
    attempted_events = {
        (row.get("sport_key"), row.get("event_id"))
        for row in rows
        if row.get("event_id") and (row.get("closing_attempted_at")
                                    or row.get("closing_captured_at"))
    }
    for row in rows:
        if sport_key and row.get("sport_key") != sport_key:
            continue
        if (row.get("resolved") or row.get("closing_captured_at")
                or not row.get("event_id")):
            continue
        event_key = (row.get("sport_key"), row["event_id"])
        if event_key in attempted_events and not include_attempted:
            continue
        if not row.get("prop_key") or not row.get("player"):
            continue
        if (row.get("direction") or "").upper() not in ("OVER", "UNDER"):
            continue
        commence = _parse_utc(row.get("commence_time"))
        if commence is None:
            continue
        minutes_before = (commence - now).total_seconds() / 60.0
        if 0 <= minutes_before <= window_minutes:
            groups[(row.get("sport_key"), row["event_id"])].append(row)
    return groups


def overdue_closing_event_groups(rows, now=None, sport_key=None,
                                 include_attempted=False):
    """Group queued closing captures whose events have already started."""
    now = now or datetime.now(timezone.utc)
    groups = defaultdict(list)
    attempted_events = {
        (row.get("sport_key"), row.get("event_id"))
        for row in rows
        if row.get("event_id") and (row.get("closing_attempted_at")
                                    or row.get("closing_captured_at"))
    }
    for row in rows:
        if sport_key and row.get("sport_key") != sport_key:
            continue
        if (row.get("resolved") or row.get("closing_captured_at")
                or not row.get("event_id")):
            continue
        event_key = (row.get("sport_key"), row["event_id"])
        if event_key in attempted_events and not include_attempted:
            continue
        if not row.get("prop_key") or not row.get("player"):
            continue
        if (row.get("direction") or "").upper() not in ("OVER", "UNDER"):
            continue
        commence = _parse_utc(row.get("commence_time"))
        if commence is not None and commence <= now:
            groups[event_key].append(row)
    return groups


def next_closing_capture(rows, now=None, lead_minutes=5, sport_key=None):
    """Return timing metadata for the next queued closing capture."""
    now = now or datetime.now(timezone.utc)
    events = {}
    attempted_events = {
        (row.get("sport_key"), row.get("event_id"))
        for row in rows
        if row.get("event_id") and (row.get("closing_attempted_at")
                                    or row.get("closing_captured_at"))
    }
    for row in rows:
        if sport_key and row.get("sport_key") != sport_key:
            continue
        if (row.get("resolved") or row.get("closing_captured_at")
                or not row.get("event_id")):
            continue
        if not row.get("prop_key") or not row.get("player"):
            continue
        if (row.get("direction") or "").upper() not in ("OVER", "UNDER"):
            continue
        commence = _parse_utc(row.get("commence_time"))
        if commence is None:
            continue
        key = (row.get("sport_key"), row["event_id"])
        if key in attempted_events:
            continue
        events[key] = commence
    if not events:
        return None
    (event_sport, event_id), commence = min(
        events.items(), key=lambda item: item[1])
    target = commence - timedelta(minutes=lead_minutes)
    return {
        "sport_key": event_sport,
        "event_id": event_id,
        "commence_time": commence.isoformat(),
        "target_time": target.isoformat(),
        "wait_seconds": max(0.0, (target - now).total_seconds()),
    }


def find_closing_offer(game_data, row):
    """Find the best exact-player, exact-line price for the forecasted side."""
    target_player = _normalize_name(row.get("player"))
    target_prop = row.get("prop_key")
    target_side = (row.get("direction") or "").casefold()
    try:
        target_line = float(row.get("line"))
    except (TypeError, ValueError):
        return None

    offers = []
    for bookmaker in game_data.get("bookmakers", []):
        book = bookmaker.get("title") or bookmaker.get("key") or "Unknown"
        for market in bookmaker.get("markets", []):
            if market.get("key") != target_prop:
                continue
            for outcome in market.get("outcomes", []):
                if _normalize_name(outcome.get("description")) != target_player:
                    continue
                if str(outcome.get("name") or "").casefold() != target_side:
                    continue
                point = (0.5 if target_prop == "player_anytime_td"
                         else outcome.get("point"))
                try:
                    exact_line = abs(float(point) - target_line) < 1e-9
                    price = int(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                if exact_line:
                    offers.append({"book": book, "price": price})
    if not offers:
        return None
    best = max(offers, key=lambda offer: offer["price"])
    opening_book = str(row.get("book") or "").casefold()
    same_book = next(
        (offer for offer in offers
         if str(offer["book"]).casefold() == opening_book),
        None,
    )
    return {
        "price": best["price"],
        "book": best["book"],
        "same_book_price": same_book["price"] if same_book else None,
        "books_sampled": len(offers),
    }


def capture_closing_odds(api_key, bookmakers=None, window_minutes=10,
                         sport_key=None, dry_run=False, now=None,
                         force_retry=False):
    """Capture live or queued historical exact-line closing prices.

    Concurrency / single-replica assumption
    ---------------------------------------
    The `_capture_lock` below is *process-local*: it prevents two Streamlit
    sessions in the SAME process from paying for the same event's closing
    snapshot. It does NOT coordinate across processes/replicas. The event is
    also read (`read_prediction_log`) and its "attempted" flag written
    (`mutate_prediction_log`) in two separate steps, so two replicas running
    this concurrently could both see an event as un-attempted and each spend a
    credit fetching its closing odds (a double-spend), even though the log
    write itself is transactional.

    This is safe on Streamlit Community Cloud, which runs a SINGLE replica per
    app — the deployment target here. Before scaling to multiple replicas,
    replace the read-then-write with an atomic claim (e.g. conditionally stamp
    `closing_attempted_at` via the blob's If-Match ETag and only fetch after
    winning the claim), so exactly one replica pays per event.
    """
    now = now or datetime.now(timezone.utc)
    # Serialize the read/fetch/write cycle within this process (see the
    # single-replica note above) so two sessions cannot spend credits on the
    # same event before either one records its attempt.
    with _capture_lock:
        prediction_rows = read_prediction_log()
        live_groups = closing_event_groups(
            prediction_rows, now=now, window_minutes=window_minutes,
            sport_key=sport_key, include_attempted=force_retry)
        historical_groups = overdue_closing_event_groups(
            prediction_rows, now=now, sport_key=sport_key,
            include_attempted=force_retry)
        # At the exact commence-time boundary an event can satisfy both group
        # predicates. It must use the timestamped historical endpoint once the
        # game has started, not the mutable live endpoint.
        for event_key in historical_groups:
            live_groups.pop(event_key, None)
        event_groups = [
            (event_key, rows, False)
            for event_key, rows in live_groups.items()
        ] + [
            (event_key, rows, True)
            for event_key, rows in historical_groups.items()
        ]
        market_count = sum(len({row["prop_key"] for row in rows})
                           for _, rows, _ in event_groups)
        if dry_run:
            return {
                "events": len(event_groups), "markets": market_count,
                "historical_events": len(historical_groups),
                "rows_updated": 0, "exact_line_misses": 0,
                "request_errors": 0, "closing_captured": 0,
                "historical_captured": 0,
            }

        updates = {}
        misses = 0
        request_errors = 0
        captured = 0
        historical_captured = 0
        for (event_sport, event_id), rows, use_historical in event_groups:
            markets = ",".join(sorted({row["prop_key"] for row in rows}))
            attempted_at = now.isoformat()
            snapshot_at = attempted_at
            try:
                if use_historical:
                    commence = _parse_utc(rows[0].get("commence_time"))
                    target = commence - timedelta(minutes=5)
                    target_time = target.isoformat().replace("+00:00", "Z")
                    game_data, snapshot_at = get_historical_event_odds(
                        api_key, event_sport, event_id, date=target_time,
                        markets=markets, bookmakers=bookmakers)
                    if not game_data:
                        raise RuntimeError("historical_snapshot_not_found")
                else:
                    game_data = get_event_odds(
                        api_key, event_sport, event_id, markets=markets,
                        bookmakers=bookmakers, force_refresh=True)
            except Exception as exc:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                error_code = type(exc).__name__
                if status_code is not None:
                    error_code += f"_{status_code}"
                print(f"  [closing] {event_id}: {error_code}")
                request_errors += 1
                # Don't burn the one-shot "attempted" flag on failures that are
                # retryable later and spent no credit: transient network faults
                # (a RequestException with no HTTP response) and quota/rate-limit
                # responses (401/429). Leaving the event un-attempted lets the
                # next run recover the paid closing snapshot instead of dropping
                # it forever. Definitive outcomes (snapshot-not-found = credit
                # spent, or a malformed-request HTTP error) are still marked.
                retry_later = (
                    (isinstance(exc, requests.exceptions.RequestException)
                     and response is None)
                    or status_code in (401, 429)
                )
                if retry_later:
                    continue
                for row in rows:
                    updates[prediction_row_key(row)] = {
                        "closing_attempted_at": attempted_at,
                        "closing_attempt_error": error_code,
                    }
                continue
            for row in rows:
                offer = find_closing_offer(game_data, row)
                update = {
                    "closing_attempted_at": attempted_at,
                    "closing_attempt_error": None,
                }
                if not offer:
                    misses += 1
                    update["closing_attempt_error"] = "exact_line_not_found"
                    updates[prediction_row_key(row)] = update
                    continue
                commence = _parse_utc(row.get("commence_time"))
                snapshot_time = _parse_utc(snapshot_at) or now
                minutes_before = ((commence - snapshot_time).total_seconds() / 60.0
                                  if commence else None)
                update.update({
                    "closing_price": offer["price"],
                    "closing_book": offer["book"],
                    "closing_same_book_price": offer["same_book_price"],
                    "closing_books_sampled": offer["books_sampled"],
                    "closing_captured_at": attempted_at,
                    "closing_snapshot_at": snapshot_time.isoformat(),
                    "closing_minutes_before": (
                        round(minutes_before, 2)
                        if minutes_before is not None else None),
                    "closing_source": (
                        "odds_api_historical_exact_line_pregame"
                        if use_historical
                        else "odds_api_exact_line_pregame"),
                })
                captured += 1
                if use_historical:
                    historical_captured += 1
                updates[prediction_row_key(row)] = update

        def apply_updates(current_rows):
            changed = 0
            for row in current_rows:
                update = updates.get(prediction_row_key(row))
                if update:
                    row.update(update)
                    changed += 1
            return changed

        rows_updated = mutate_prediction_log(apply_updates) if updates else 0
        return {
            "events": len(event_groups), "markets": market_count,
            "historical_events": len(historical_groups),
            "rows_updated": rows_updated or 0, "exact_line_misses": misses,
            "request_errors": request_errors, "closing_captured": captured,
            "historical_captured": historical_captured,
        }


def resolve_and_refit(sport_key=None):
    sports = [sport_key] if sport_key else list(SPORT_ALIASES.values())
    return {key: maintain_sport(key) for key in sports}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-closing", action="store_true",
                        help="Capture exact-line prices shortly before game time")
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve past outcomes and run gated recalibration")
    parser.add_argument("--sport", choices=list(SPORT_ALIASES), default=None,
                        help="Limit work to one sport (default: all logged sports)")
    parser.add_argument("--window-minutes", type=int, default=10,
                        help="How soon an event must start for a closing fetch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show eligible event/market counts without API calls")
    args = parser.parse_args()
    if not args.capture_closing and not args.resolve:
        parser.error("choose --capture-closing and/or --resolve")

    sport_key = SPORT_ALIASES.get(args.sport) if args.sport else None
    print(f"Prediction log storage: {prediction_log_storage()}")
    if args.capture_closing:
        settings = _load_settings()
        if not settings["api_key"] and not args.dry_run:
            parser.error("ODDS_API_KEY is not configured")
        result = capture_closing_odds(
            settings["api_key"], bookmakers=settings["bookmakers"],
            window_minutes=args.window_minutes, sport_key=sport_key,
            dry_run=args.dry_run)
        print("Closing capture: " + ", ".join(
            f"{key}={value}" for key, value in result.items()))
    if args.resolve:
        for key, result in resolve_and_refit(sport_key).items():
            print(f"{key}: resolved={result['newly_resolved']}, "
                  f"refit={result['refit']}")


if __name__ == "__main__":
    _main()
