"""
Backtest whether same-snapshot peer-book consensus identifies value at DraftKings.

Historical featured-market snapshots are immutable and cached permanently by
odds_client, so interrupted or repeated runs do not repay for completed dates.
DraftKings is always excluded from the peer consensus. Spread and total peers
must offer the exact same point as DraftKings, and every contributing book is
de-vigged before the median fair probability is calculated.

Default study design (declared before the API pull):
    - MLB regular-season Tuesdays and Saturdays, April-September 2024 and 2025
    - one 14:00 UTC snapshot per sampled date
    - moneyline, run line, and total
    - games must start at least two hours after the snapshot
    - at least three non-DraftKings books at the exact DraftKings line

Usage:
    python backtest_market_consensus.py --dry-run
    python backtest_market_consensus.py
    python backtest_market_consensus.py --sport nba
    python backtest_market_consensus.py --sport nfl
"""

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from odds_client import (
    BASE_URL,
    _cache_key,
    _normalize_snapshot_date,
    american_to_decimal,
    american_to_implied_prob,
    get_historical_odds,
    get_remaining_credits,
)
from espn_client import get_all_teams, get_team_schedule


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
RESULT_CACHE_DIR = os.path.join(SCRIPT_DIR, "cache", "market_consensus")
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
EASTERN = ZoneInfo("America/New_York")

SPORT_STUDIES = {
    "nba": {
        "sport_key": "basketball_nba",
        "espn_sport": "basketball",
        "espn_league": "nba",
        "ranges": [
            ("2024", "2023-10-24", "2024-04-14", {1}),
            ("2025", "2024-10-22", "2025-04-13", {1}),
        ],
    },
    "nfl": {
        "sport_key": "americanfootball_nfl",
        "espn_sport": "football",
        "espn_league": "nfl",
        "ranges": [
            ("2023", "2023-09-07", "2024-01-07", {6}),
            ("2024", "2024-09-05", "2025-01-05", {6}),
            ("2025", "2025-09-04", "2026-01-04", {6}),
        ],
    },
}

MARKET_SPECS = {
    "h2h": {
        "label": "Moneyline",
        "side_a": "home",
        "side_b": "away",
    },
    "spreads": {
        "label": "Run line",
        "side_a": "home",
        "side_b": "away",
    },
    "totals": {
        "label": "Total",
        "side_a": "over",
        "side_b": "under",
    },
}

WEEKDAYS = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_utc(timestamp):
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sample_dates(years, weekday_numbers, start_month, end_month):
    sampled = []
    for year in years:
        current = date(year, start_month, 1)
        if end_month == 12:
            end = date(year, 12, 31)
        else:
            end = date(year, end_month + 1, 1) - timedelta(days=1)
        while current <= end:
            if current.weekday() in weekday_numbers:
                sampled.append(current.isoformat())
            current += timedelta(days=1)
    return sampled


def _dates_in_range(start_iso, end_iso, weekday_numbers):
    current = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    sampled = []
    while current <= end:
        if current.weekday() in weekday_numbers:
            sampled.append(current.isoformat())
        current += timedelta(days=1)
    return sampled


def _study_dates(sport, years, weekday_numbers, start_month, end_month):
    if sport == "mlb":
        dates = _sample_dates(
            years, weekday_numbers, start_month, end_month)
        return dates, {str(year): year for year in years}

    dates = []
    periods = {}
    for period, start_iso, end_iso, weekdays in SPORT_STUDIES[sport]["ranges"]:
        range_dates = _dates_in_range(start_iso, end_iso, weekdays)
        dates.extend(range_dates)
        periods[period] = int(period)
    return dates, periods


def _snapshot_timestamp(date_iso, wall_time):
    return f"{date_iso}T{wall_time}:00Z"


def _historical_snapshot_cached(sport, timestamp, regions, markets):
    timestamp = _normalize_snapshot_date(timestamp)
    path = _cache_key("hist_odds", sport, timestamp, regions, markets, "")
    return os.path.exists(path)


def _remaining_credits(api_key):
    response = requests.get(
        f"{BASE_URL}/sports",
        params={"apiKey": api_key},
        timeout=30,
    )
    response.raise_for_status()
    value = response.headers.get("x-requests-remaining")
    return int(value) if value is not None else None


def _valid_american_price(price):
    return (
        isinstance(price, (int, float))
        and not isinstance(price, bool)
        and abs(price) >= 100
    )


def _fair_pair(price_a, price_b):
    if not _valid_american_price(price_a) or not _valid_american_price(price_b):
        return None
    implied_a = american_to_implied_prob(price_a)
    implied_b = american_to_implied_prob(price_b)
    total = implied_a + implied_b
    if total <= 0:
        return None
    return implied_a / total, implied_b / total


def _same_point(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def _team_key(name):
    normalized = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    aliases = {
        "oaklandathletics": "athletics",
        "theathletics": "athletics",
        "laclippers": "clippers",
        "losangelesclippers": "clippers",
    }
    return aliases.get(normalized, normalized)


def _market_offer(game, bookmaker, market_key):
    market = next(
        (entry for entry in bookmaker.get("markets", [])
         if entry.get("key") == market_key),
        None,
    )
    if not market:
        return None

    outcomes = market.get("outcomes", [])
    if market_key in ("h2h", "spreads"):
        home = next(
            (outcome for outcome in outcomes
             if _team_key(outcome.get("name")) == _team_key(game.get("home_team"))),
            None,
        )
        away = next(
            (outcome for outcome in outcomes
             if _team_key(outcome.get("name")) == _team_key(game.get("away_team"))),
            None,
        )
        if not home or not away:
            return None
        offer = {
            "price_a": home.get("price"),
            "price_b": away.get("price"),
            "point_a": home.get("point"),
            "point_b": away.get("point"),
        }
    else:
        over = next(
            (outcome for outcome in outcomes
             if str(outcome.get("name", "")).lower() == "over"),
            None,
        )
        under = next(
            (outcome for outcome in outcomes
             if str(outcome.get("name", "")).lower() == "under"),
            None,
        )
        if not over or not under:
            return None
        offer = {
            "price_a": over.get("price"),
            "price_b": under.get("price"),
            "point_a": over.get("point"),
            "point_b": under.get("point"),
        }

    if _fair_pair(offer["price_a"], offer["price_b"]) is None:
        return None
    return offer


def _matches_draftkings_line(market_key, draftkings, peer):
    if market_key == "h2h":
        return True
    return (
        _same_point(draftkings.get("point_a"), peer.get("point_a"))
        and _same_point(draftkings.get("point_b"), peer.get("point_b"))
    )


def _grade_side(market_key, side, draftkings, home_score, away_score):
    if market_key == "h2h":
        value = (
            home_score - away_score
            if side == "a"
            else away_score - home_score
        )
    elif market_key == "spreads":
        point = draftkings["point_a" if side == "a" else "point_b"]
        if point is None:
            return None
        value = (
            home_score - away_score + float(point)
            if side == "a"
            else away_score - home_score + float(point)
        )
    else:
        point = draftkings["point_a" if side == "a" else "point_b"]
        if point is None:
            return None
        value = home_score + away_score - float(point)
        if side == "b":
            value = -value
    if value > 1e-9:
        return 1
    if value < -1e-9:
        return -1
    return 0


def _build_market_observations(game, result, market_key, min_peer_books):
    books = game.get("bookmakers", [])
    draftkings_book = next(
        (book for book in books if book.get("key") == "draftkings"),
        None,
    )
    if not draftkings_book:
        return []
    draftkings = _market_offer(game, draftkings_book, market_key)
    if not draftkings:
        return []

    peer_probabilities = []
    peer_books = []
    for book in books:
        book_key = book.get("key")
        if not book_key or book_key == "draftkings":
            continue
        offer = _market_offer(game, book, market_key)
        if not offer or not _matches_draftkings_line(market_key, draftkings, offer):
            continue
        fair = _fair_pair(offer["price_a"], offer["price_b"])
        if fair is None:
            continue
        peer_probabilities.append(fair)
        peer_books.append(book_key)

    if len(peer_probabilities) < min_peer_books:
        return []

    peer_a = statistics.median(pair[0] for pair in peer_probabilities)
    peer_b = statistics.median(pair[1] for pair in peer_probabilities)
    peer_total = peer_a + peer_b
    peer_a, peer_b = peer_a / peer_total, peer_b / peer_total
    draftkings_fair = _fair_pair(
        draftkings["price_a"], draftkings["price_b"])

    observations = []
    for side, peer_probability, fair_probability, price_key in (
        ("a", peer_a, draftkings_fair[0], "price_a"),
        ("b", peer_b, draftkings_fair[1], "price_b"),
    ):
        price = draftkings[price_key]
        raw_probability = american_to_implied_prob(price)
        result_code = _grade_side(
            market_key,
            side,
            draftkings,
            result["home_score"],
            result["away_score"],
        )
        if result_code is None:
            continue
        profit = (
            american_to_decimal(price) - 1
            if result_code == 1
            else -1 if result_code == -1 else 0
        )
        observations.append({
            "event_id": game.get("id"),
            "date": result["date"],
            "year": int(result["date"][:4]),
            "period": str(result.get("period", result["date"][:4])),
            "market": market_key,
            "side": MARKET_SPECS[market_key][
                "side_a" if side == "a" else "side_b"
            ],
            "price": price,
            "point": draftkings[
                "point_a" if side == "a" else "point_b"
            ],
            "peer_probability": peer_probability,
            "draftkings_fair_probability": fair_probability,
            "draftkings_raw_probability": raw_probability,
            "edge": peer_probability - raw_probability,
            "expected_roi": (
                peer_probability * american_to_decimal(price) - 1
            ),
            "result": result_code,
            "profit": profit,
            "peer_count": len(peer_probabilities),
            "peer_books": sorted(peer_books),
        })
    return observations


def _build_line_advantage_observations(
        game, result, market_key, min_peer_books):
    """Grade a better DK point than the median peer main line at DK's price."""
    if market_key not in ("spreads", "totals"):
        return []
    books = game.get("bookmakers", [])
    draftkings_book = next(
        (book for book in books if book.get("key") == "draftkings"),
        None,
    )
    draftkings = (
        _market_offer(game, draftkings_book, market_key)
        if draftkings_book else None
    )
    if not draftkings:
        return []

    peers = []
    for book in books:
        book_key = book.get("key")
        if not book_key or book_key == "draftkings":
            continue
        offer = _market_offer(game, book, market_key)
        if offer:
            peers.append((book_key, offer))
    if len(peers) < min_peer_books:
        return []

    if market_key == "spreads":
        peer_points = {
            "a": statistics.median(offer["point_a"] for _, offer in peers),
            "b": statistics.median(offer["point_b"] for _, offer in peers),
        }
        candidates = [
            (
                side,
                float(draftkings[f"point_{side}"]) - float(peer_points[side]),
            )
            for side in ("a", "b")
        ]
    else:
        peer_total = statistics.median(
            float(offer["point_a"]) for _, offer in peers)
        draftkings_total = float(draftkings["point_a"])
        candidates = [
            ("a", peer_total - draftkings_total),
            ("b", draftkings_total - peer_total),
        ]
        peer_points = {"a": peer_total, "b": peer_total}

    favorable = [(side, advantage) for side, advantage in candidates
                 if advantage > 1e-9]
    if not favorable:
        return []
    # Opposing spread points and one shared total should yield one favorable
    # side. If malformed source data makes both positive, keep only the larger.
    side, advantage = max(favorable, key=lambda candidate: candidate[1])
    price = draftkings[f"price_{side}"]
    crossed_key_numbers = []
    if market_key == "spreads":
        peer_point = float(peer_points[side])
        draftkings_point = float(draftkings[f"point_{side}"])
        crossed_key_numbers = sorted({
            abs(key_number)
            for key_number in (-7.0, -3.0, 3.0, 7.0)
            if peer_point <= key_number <= draftkings_point
        })
    peer_median_decimal_price = statistics.median(
        american_to_decimal(offer[f"price_{side}"])
        for _, offer in peers
    )
    dominates_peer_offer = (
        american_to_decimal(price) >= peer_median_decimal_price
    )
    result_code = _grade_side(
        market_key,
        side,
        draftkings,
        result["home_score"],
        result["away_score"],
    )
    if result_code is None:
        return []
    profit = (
        american_to_decimal(price) - 1
        if result_code == 1
        else -1 if result_code == -1 else 0
    )
    return [{
        "event_id": game.get("id"),
        "date": result["date"],
        "year": int(result["date"][:4]),
        "period": str(result.get("period", result["date"][:4])),
        "market": market_key,
        "side": MARKET_SPECS[market_key][
            "side_a" if side == "a" else "side_b"
        ],
        "price": price,
        "draftkings_point": draftkings[f"point_{side}"],
        "peer_median_point": peer_points[side],
        "point_advantage": advantage,
        "crossed_key_numbers": crossed_key_numbers,
        "peer_median_decimal_price": peer_median_decimal_price,
        "dominates_peer_offer": dominates_peer_offer,
        "result": result_code,
        "profit": profit,
        "peer_count": len(peers),
        "peer_books": sorted(book_key for book_key, _ in peers),
    }]


def _schedule_cache_path(year):
    return os.path.join(RESULT_CACHE_DIR, f"mlb_schedule_{year}.json")


def _load_mlb_results(year):
    os.makedirs(RESULT_CACHE_DIR, exist_ok=True)
    path = _schedule_cache_path(year)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            body = json.load(handle)
    else:
        response = requests.get(
            MLB_SCHEDULE_URL,
            params={
                "sportId": 1,
                "startDate": f"{year}-04-01",
                "endDate": f"{year}-09-30",
                "hydrate": "linescore",
            },
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(body, handle)

    results = []
    for date_entry in body.get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {})
            if status.get("abstractGameState") != "Final":
                continue
            if game.get("gameType") != "R":
                continue
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_score = home.get("score")
            away_score = away.get("score")
            if not isinstance(home_score, (int, float)):
                home_score = game.get("linescore", {}).get("teams", {}).get(
                    "home", {}).get("runs")
            if not isinstance(away_score, (int, float)):
                away_score = game.get("linescore", {}).get("teams", {}).get(
                    "away", {}).get("runs")
            if not isinstance(home_score, (int, float)) \
                    or not isinstance(away_score, (int, float)):
                continue
            results.append({
                "game_pk": game.get("gamePk"),
                "date": game.get("officialDate") or date_entry.get("date"),
                "commence_time": game.get("gameDate"),
                "home_team": home.get("team", {}).get("name"),
                "away_team": away.get("team", {}).get("name"),
                "home_score": home_score,
                "away_score": away_score,
                "period": str(year),
            })
    return results


def _espn_results_cache_path(sport, season_year):
    return os.path.join(
        RESULT_CACHE_DIR, f"{sport}_schedule_{season_year}.json")


def _load_espn_results(sport, league, season_year):
    os.makedirs(RESULT_CACHE_DIR, exist_ok=True)
    path = _espn_results_cache_path(sport, season_year)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    teams = get_all_teams(sport, league)
    schedules = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(
                get_team_schedule, sport, league, info["id"], season_year)
            for info in teams.values() if info.get("id")
        ]
        for future in as_completed(futures):
            try:
                schedules.extend(future.result())
            except Exception:
                continue

    results = []
    seen = set()
    for game in schedules:
        commence = _parse_utc(game.get("date"))
        if not commence:
            continue
        key = (
            game.get("date"),
            game.get("home_team"),
            game.get("away_team"),
        )
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "game_pk": "|".join(str(value) for value in key),
            "date": commence.astimezone(EASTERN).date().isoformat(),
            "commence_time": game.get("date"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "home_score": game.get("home_score"),
            "away_score": game.get("away_score"),
            "period": str(season_year),
        })
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)
    return results


def _index_results(results):
    indexed = defaultdict(list)
    for result in results:
        key = (
            result["date"],
            _team_key(result["home_team"]),
            _team_key(result["away_team"]),
        )
        indexed[key].append(result)
    return indexed


def _match_result(game, slate_date, indexed, used_game_pks):
    key = (
        slate_date,
        _team_key(game.get("home_team")),
        _team_key(game.get("away_team")),
    )
    candidates = [
        result for result in indexed.get(key, [])
        if result["game_pk"] not in used_game_pks
    ]
    if not candidates:
        return None
    commence = _parse_utc(game.get("commence_time"))
    if commence:
        candidates.sort(key=lambda result: abs(
            (_parse_utc(result["commence_time"]) - commence).total_seconds()
        ))
    matched = candidates[0]
    used_game_pks.add(matched["game_pk"])
    return matched


def _metric(rows):
    count = len(rows)
    if not count:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "roi_pct": None,
            "ci_low_pct": None,
            "ci_high_pct": None,
            "average_expected_roi_pct": None,
            "median_price": None,
        }
    wins = sum(row["result"] == 1 for row in rows)
    losses = sum(row["result"] == -1 for row in rows)
    pushes = count - wins - losses
    profits = [row["profit"] for row in rows]
    average = statistics.mean(profits)
    standard_error = (
        statistics.stdev(profits) / math.sqrt(count) if count > 1 else 0.0
    )
    return {
        "bets": count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "roi_pct": 100 * average,
        "ci_low_pct": 100 * (average - 1.96 * standard_error),
        "ci_high_pct": 100 * (average + 1.96 * standard_error),
        "average_expected_roi_pct": (
            100 * statistics.mean(row["expected_roi"] for row in rows)
            if all("expected_roi" in row for row in rows) else None
        ),
        "median_price": statistics.median(row["price"] for row in rows),
    }


def _best_side_per_event(observations, market_key):
    best = {}
    for row in observations:
        if row["market"] != market_key:
            continue
        event_id = row["event_id"]
        if event_id not in best \
                or row["expected_roi"] > best[event_id]["expected_roi"]:
            best[event_id] = row
    return list(best.values())


def _forecast_metrics(observations, market_key, periods=None):
    rows_by_event = {}
    for row in observations:
        if row["market"] != market_key:
            continue
        if periods is not None and row["period"] not in periods:
            continue
        if row["side"] not in ("home", "over") or row["result"] == 0:
            continue
        rows_by_event[row["event_id"]] = row
    rows = list(rows_by_event.values())
    if not rows:
        return {"games": 0, "draftkings_brier": None, "peer_brier": None}
    outcomes = [1 if row["result"] == 1 else 0 for row in rows]
    draftkings_brier = statistics.mean(
        (row["draftkings_fair_probability"] - outcome) ** 2
        for row, outcome in zip(rows, outcomes)
    )
    peer_brier = statistics.mean(
        (row["peer_probability"] - outcome) ** 2
        for row, outcome in zip(rows, outcomes)
    )
    blend_brier = statistics.mean(
        (
            (row["draftkings_fair_probability"] + row["peer_probability"])
            / 2
            - outcome
        ) ** 2
        for row, outcome in zip(rows, outcomes)
    )
    return {
        "games": len(rows),
        "draftkings_brier": draftkings_brier,
        "peer_brier": peer_brier,
        "blend_brier": blend_brier,
        "peer_minus_draftkings": peer_brier - draftkings_brier,
    }


def _fmt_metric(metric):
    if not metric["bets"]:
        return "n=0"
    return (
        f"n={metric['bets']}, {metric['wins']}-{metric['losses']}-"
        f"{metric['pushes']}, ROI={metric['roi_pct']:+.2f}% "
        f"(95% CI {metric['ci_low_pct']:+.2f}% to "
        f"{metric['ci_high_pct']:+.2f}%)"
    )


def _build_report(observations, line_observations, coverage, study):
    report = {
        "study": study,
        "coverage": dict(coverage),
        "markets": {},
    }
    thresholds = (0.0, 0.01, 0.02, 0.03, 0.05)
    periods = study["periods"]
    for market_key, spec in MARKET_SPECS.items():
        candidates = _best_side_per_event(observations, market_key)
        market_report = {
            "label": spec["label"],
            "eligible_games": len(candidates),
            "forecast": {
                "all": _forecast_metrics(observations, market_key),
                **{
                    period: _forecast_metrics(
                        observations, market_key, {period})
                    for period in periods
                },
            },
            "edge_thresholds": {},
        }
        for threshold in thresholds:
            selected = [row for row in candidates if row["edge"] >= threshold]
            market_report["edge_thresholds"][f"{threshold * 100:.0f}pp"] = {
                "all": _metric(selected),
                **{
                    period: _metric(
                        [row for row in selected if row["period"] == period]
                    )
                    for period in periods
                },
            }
        line_rows = [
            row for row in line_observations if row["market"] == market_key
        ]
        if line_rows:
            market_report["favorable_line_difference"] = {
                "all": _metric(line_rows),
                "dominant_offer": _metric([
                    row for row in line_rows if row["dominates_peer_offer"]
                ]),
                **{
                    period: _metric(
                        [row for row in line_rows if row["period"] == period]
                    )
                    for period in periods
                },
                "by_point_advantage": {
                    str(advantage): _metric([
                        row for row in line_rows
                        if row["point_advantage"] == advantage
                    ])
                    for advantage in sorted({
                        row["point_advantage"] for row in line_rows
                    })
                },
            }
            if study["sport"] == "nfl" and market_key == "spreads":
                market_report["favorable_line_difference"][
                    "key_number_crossing"] = {
                        "any": _metric([
                            row for row in line_rows
                            if row["crossed_key_numbers"]
                        ]),
                        "3": _metric([
                            row for row in line_rows
                            if 3.0 in row["crossed_key_numbers"]
                        ]),
                        "7": _metric([
                            row for row in line_rows
                            if 7.0 in row["crossed_key_numbers"]
                        ]),
                    }
        report["markets"][market_key] = market_report
    return report


def _print_report(report):
    print("\n=== Same-snapshot peer consensus vs DraftKings ===")
    print(
        f"  Snapshots loaded: {report['coverage'].get('snapshots_loaded', 0)}; "
        f"eligible games matched: "
        f"{report['coverage'].get('matched_games', 0)}"
    )
    print(
        f"  Skipped (< minimum lead): "
        f"{report['coverage'].get('too_close_or_started', 0)}; "
        f"unmatched results: {report['coverage'].get('unmatched_results', 0)}"
    )
    for market in report["markets"].values():
        print(f"\n{market['label']} — {market['eligible_games']} exact-line games")
        forecast = market["forecast"]["all"]
        if forecast["games"]:
            print(
                f"  Brier: DK={forecast['draftkings_brier']:.6f}, "
                f"peers={forecast['peer_brier']:.6f}, "
                f"50/50={forecast['blend_brier']:.6f}, "
                f"peer-DK={forecast['peer_minus_draftkings']:+.6f}"
            )
        for threshold, by_period in market["edge_thresholds"].items():
            print(f"  edge >= {threshold:>3}: {_fmt_metric(by_period['all'])}")
            year_parts = [
                f"{period}: {_fmt_metric(by_period[period])}"
                for period in report["study"]["periods"]
            ]
            print(" " * 16 + " | ".join(year_parts))
        line_difference = market.get("favorable_line_difference")
        if line_difference:
            metric = line_difference["all"]
            print(
                "  favorable DK point vs peer median: "
                f"{_fmt_metric(metric)}; median DK price="
                f"{metric['median_price']:+g}"
            )
            print(
                "    DK also has median-or-better payout: "
                f"{_fmt_metric(line_difference['dominant_offer'])}"
            )
            key_crossing = line_difference.get("key_number_crossing")
            if key_crossing:
                print(
                    "    crosses NFL key number 3 or 7: "
                    f"{_fmt_metric(key_crossing['any'])}"
                )
                print(f"      key 3: {_fmt_metric(key_crossing['3'])}")
                print(f"      key 7: {_fmt_metric(key_crossing['7'])}")
            period_parts = [
                f"{period}: {_fmt_metric(line_difference[period])}"
                for period in report["study"]["periods"]
            ]
            print(" " * 4 + " | ".join(period_parts))
            for advantage, bucket in line_difference[
                    "by_point_advantage"].items():
                print(
                    f"    point advantage {advantage}: "
                    f"{_fmt_metric(bucket)}"
                )


def _parse_csv(value):
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Backtest same-snapshot peer-book consensus vs DraftKings")
    parser.add_argument("--sport", choices=("mlb", "nba", "nfl"),
                        default="mlb")
    parser.add_argument("--years", default="2024,2025")
    parser.add_argument("--weekdays", default="tue,sat")
    parser.add_argument("--start-month", type=int, default=4)
    parser.add_argument("--end-month", type=int, default=9)
    parser.add_argument("--snapshot-time", default="14:00")
    parser.add_argument("--markets", default="h2h,spreads,totals")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--min-lead-hours", type=float, default=2.0)
    parser.add_argument("--min-peer-books", type=int, default=3)
    parser.add_argument("--max-credits", type=int, default=3200)
    parser.add_argument("--reserve-credits", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        years = sorted({int(value) for value in _parse_csv(args.years)})
    except ValueError:
        parser.error("--years must be a comma-separated list of years")
    weekday_names = _parse_csv(args.weekdays)
    unknown_weekdays = [name for name in weekday_names if name not in WEEKDAYS]
    if unknown_weekdays:
        parser.error(f"Unknown weekdays: {unknown_weekdays}")
    markets = ",".join(_parse_csv(args.markets))
    unknown_markets = [
        market for market in _parse_csv(markets) if market not in MARKET_SPECS
    ]
    if unknown_markets:
        parser.error(f"Unknown markets: {unknown_markets}")
    try:
        datetime.strptime(args.snapshot_time, "%H:%M")
    except ValueError:
        parser.error("--snapshot-time must be UTC HH:MM")

    sampled_dates, period_seasons = _study_dates(
        args.sport,
        years,
        {WEEKDAYS[name] for name in weekday_names},
        args.start_month,
        args.end_month,
    )
    if args.sport != "mlb":
        preset_weekdays = {
            weekday
            for _, _, _, weekdays in SPORT_STUDIES[args.sport]["ranges"]
            for weekday in weekdays
        }
        weekday_names = [
            name for name, number in WEEKDAYS.items()
            if number in preset_weekdays
        ]
    periods = list(period_seasons)
    sport_key = (
        "baseball_mlb" if args.sport == "mlb"
        else SPORT_STUDIES[args.sport]["sport_key"]
    )
    tasks = [
        _snapshot_timestamp(date_iso, args.snapshot_time)
        for date_iso in sampled_dates
    ]
    uncached = [
        timestamp for timestamp in tasks
        if not _historical_snapshot_cached(
            sport_key, timestamp, args.regions, markets)
    ]
    cost_per_snapshot = 10 * len(_parse_csv(markets)) * len(
        _parse_csv(args.regions))
    estimated_cost = len(uncached) * cost_per_snapshot

    config = _load_config()
    api_key = config["odds_api_key"]
    remaining = _remaining_credits(api_key)
    print(f"\n=== {args.sport.upper()} cross-book consensus study plan ===")
    print(f"  Dates: {len(tasks)} ({sampled_dates[0]} through {sampled_dates[-1]})")
    print(f"  Markets: {markets}; snapshot: {args.snapshot_time}Z")
    print(f"  Cached dates: {len(tasks) - len(uncached)}; new dates: {len(uncached)}")
    print(f"  Estimated new cost: {estimated_cost} credits")
    print(f"  Credits remaining before run: {remaining}")
    if estimated_cost > args.max_credits:
        raise RuntimeError(
            f"Estimated cost {estimated_cost} exceeds --max-credits "
            f"{args.max_credits}")
    if remaining is not None and remaining - estimated_cost < args.reserve_credits:
        raise RuntimeError(
            f"Run would leave {remaining - estimated_cost} credits, below "
            f"--reserve-credits {args.reserve_credits}")
    if args.dry_run:
        return

    results = []
    if args.sport == "mlb":
        for year in years:
            results.extend(_load_mlb_results(year))
    else:
        preset = SPORT_STUDIES[args.sport]
        for period in periods:
            results.extend(_load_espn_results(
                preset["espn_sport"],
                preset["espn_league"],
                period_seasons[period],
            ))
    print(f"  Loaded {len(results)} completed results for matching.")
    indexed_results = _index_results(results)

    snapshots = []
    for index, (slate_date, timestamp) in enumerate(
            zip(sampled_dates, tasks), start=1):
        print(f"\n[{index}/{len(tasks)}] {timestamp}")
        games, actual_timestamp = get_historical_odds(
            api_key,
            sport_key,
            timestamp,
            regions=args.regions,
            markets=markets,
            bookmakers=None,
        )
        snapshots.append((slate_date, actual_timestamp, games))
        current_remaining = get_remaining_credits()
        if current_remaining is not None \
                and current_remaining < args.reserve_credits:
            raise RuntimeError(
                f"Stopped because remaining credits ({current_remaining}) "
                f"fell below reserve ({args.reserve_credits})")
        time.sleep(0.05)

    coverage = Counter()
    coverage["snapshots_loaded"] = len(snapshots)
    observations = []
    line_observations = []
    used_game_pks = set()
    for slate_date, actual_timestamp, games in snapshots:
        snapshot_dt = _parse_utc(actual_timestamp)
        coverage["raw_snapshot_games"] += len(games)
        for game in games:
            commence = _parse_utc(game.get("commence_time"))
            if not commence or not snapshot_dt:
                coverage["invalid_time"] += 1
                continue
            if commence.astimezone(EASTERN).date().isoformat() != slate_date:
                coverage["other_slate_date"] += 1
                continue
            lead_hours = (commence - snapshot_dt).total_seconds() / 3600
            if lead_hours < args.min_lead_hours:
                coverage["too_close_or_started"] += 1
                continue
            result = _match_result(
                game, slate_date, indexed_results, used_game_pks)
            if not result:
                coverage["unmatched_results"] += 1
                continue
            coverage["matched_games"] += 1
            for market_key in _parse_csv(markets):
                rows = _build_market_observations(
                    game, result, market_key, args.min_peer_books)
                if rows:
                    coverage[f"{market_key}_eligible_games"] += 1
                    for row in rows:
                        row["lead_hours"] = lead_hours
                    observations.extend(rows)
                line_rows = _build_line_advantage_observations(
                    game, result, market_key, args.min_peer_books)
                if line_rows:
                    for row in line_rows:
                        row["lead_hours"] = lead_hours
                    line_observations.extend(line_rows)

    study = {
        "sport": args.sport,
        "periods": periods,
        "weekdays": weekday_names,
        "start_month": args.start_month,
        "end_month": args.end_month,
        "snapshot_time_utc": args.snapshot_time,
        "markets": _parse_csv(markets),
        "min_lead_hours": args.min_lead_hours,
        "min_peer_books": args.min_peer_books,
        "planned_snapshot_credits": len(tasks) * cost_per_snapshot,
        "new_api_credits_this_run": estimated_cost,
        "credits_remaining_after": get_remaining_credits() or remaining,
    }
    report = _build_report(
        observations, line_observations, coverage, study)
    os.makedirs(RESULT_CACHE_DIR, exist_ok=True)
    report_path = os.path.join(
        RESULT_CACHE_DIR, f"{args.sport}_consensus_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    _print_report(report)
    print(f"\nReport cached at: {report_path}")
    print(f"Credits remaining: {study['credits_remaining_after']}")


if __name__ == "__main__":
    main()
