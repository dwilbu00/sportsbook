"""
Leakage-safe historical Statcast layer for backtesting (as-of-date features).

The live client (mlb_starters.py) uses SEASON-AGGREGATE Savant x-stats, which is
correct for *today* but would leak future information into a backtest of a past
game. This module instead caches per-day Statcast pitch data and computes
"as-of" aggregates from ONLY the pitches thrown BEFORE the game being graded.

Measure used: mean ``estimated_woba_using_speedangle`` over batted-ball events
(xwOBAcon — contact quality). Lower = better pitcher / weaker offense.

Historical pitch data never changes, so day caches are permanent. Full-season
pulls are slow (Savant-rate-limited) but free; run once and reuse.
"""

import csv
import io
import json
import os
import random
import time
from datetime import date as _date, timedelta

import requests

SAVANT_BASE = "https://baseballsavant.mlb.com"
SAVANT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsbookValueFinder/1.0)"}
# Bump SCHEMA_VERSION whenever the trimmed row shape changes. v4 stores xBA on
# an at-bat basis: batted-ball xBA plus zeroes for strikeouts, while excluding
# walks/HBP/sacrifices. Raw contact-only xBA (v3) is intentionally not a fallback
# because comparing its BABIP-like denominator with batter hits/AB is invalid.
SCHEMA_VERSION = 4
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cache", f"statcast_days_v{SCHEMA_VERSION}")
LEGACY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cache", "statcast_days_v2")

# Minimum prior events before an as-of estimate is usable (batted balls for
# xwOBAcon, official at-bats for xBA).
MIN_BBE = 40


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _day_path(day):
    return os.path.join(CACHE_DIR, f"{day}.json")


def _at_bat_xba(row):
    """Return expected hits for an official at-bat, else None."""
    event = row.get("events")
    if event in ("strikeout", "strikeout_double_play"):
        return 0.0
    if event in ("walk", "intent_walk", "hit_by_pitch", "sac_fly",
                 "sac_bunt", "catcher_interf"):
        return None
    value = row.get("estimated_ba_using_speedangle")
    try:
        return float(value) if value not in ("", "null", None) else None
    except (TypeError, ValueError):
        return None


def fetch_statcast_day(day, force=False):
    """
    Fetch (and permanently cache) all pitches for a single ``day`` (YYYY-MM-DD).

    Returns a list of trimmed dict rows:
        {game_date, pitcher, p_throws, batting_team, stand, xwoba, xba}
    xwOBA is populated for batted balls. xBA is populated once per official
    at-bat (including 0.0 for strikeouts), otherwise None.
    """
    _ensure_dir()
    path = _day_path(day)
    if not force and os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    url = f"{SAVANT_BASE}/statcast_search/csv"
    params = {"all": "true", "type": "details",
              "game_date_gt": day, "game_date_lt": day}
    resp = None
    for attempt in range(5):
        resp = requests.get(url, params=params, headers=SAVANT_HEADERS, timeout=90)
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            time.sleep(1.5 ** attempt + random.uniform(0, 0.5))
            continue
        break
    resp.raise_for_status()
    rows = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))

    trimmed = []
    for r in rows:
        xw = r.get("estimated_woba_using_speedangle")
        try:
            xw = float(xw) if xw not in ("", "null", None) else None
        except ValueError:
            xw = None
        xb = _at_bat_xba(r)
        topbot = r.get("inning_topbot")
        # The batting team is the AWAY team in the top half, HOME in the bottom.
        batting_team = r.get("away_team") if topbot == "Top" else r.get("home_team")
        trimmed.append({
            "game_date": r.get("game_date"),
            "pitcher": r.get("pitcher"),
            "batter": r.get("batter"),
            "p_throws": r.get("p_throws"),
            "batting_team": batting_team,
            "stand": r.get("stand"),
            "xwoba": xw,
            "xba": xb,
        })
    with open(path, "w") as f:
        json.dump(trimmed, f)
    return trimmed


def fetch_range(start, end, sleep=0.5, verbose=True):
    """Ensure every day in [start, end] (inclusive, YYYY-MM-DD) is cached."""
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    day = d0
    n = 0
    while day <= d1:
        ds = day.isoformat()
        if not os.path.exists(_day_path(ds)):
            fetch_statcast_day(ds)
            n += 1
            if verbose:
                print(f"  fetched {ds}")
            time.sleep(sleep)  # be polite to Savant
        day += timedelta(days=1)
    if verbose:
        print(f"fetch_range done: {n} new day(s) cached")


def load_days(start, end):
    """Load all cached day rows in [start, end] into one list (must be fetched)."""
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    out = []
    day = d0
    while day <= d1:
        p = _day_path(day.isoformat())
        if not os.path.exists(p):
            legacy = os.path.join(LEGACY_CACHE_DIR, f"{day.isoformat()}.json")
            p = legacy if os.path.exists(legacy) else p
        if os.path.exists(p):
            with open(p, "r") as f:
                out.extend(json.load(f))
        day += timedelta(days=1)
    return out


def asof_pitcher_xwoba(pitcher_id, rows, as_of, min_bbe=MIN_BBE):
    """
    Mean xwOBAcon a pitcher ALLOWED using only batted balls before ``as_of``.
    Returns (xwoba, n_bbe) or (None, n) if under the sample threshold.
    """
    pid = str(pitcher_id)
    vals = [r["xwoba"] for r in rows
            if r["pitcher"] == pid and r["xwoba"] is not None
            and r["game_date"] and r["game_date"] < as_of]
    if len(vals) < min_bbe:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def asof_team_xwoba_vs_hand(team_abbr, hand, rows, as_of, min_bbe=MIN_BBE):
    """
    Mean xwOBAcon a team's hitters PRODUCED vs pitchers of a given hand
    ('L'/'R'), using only batted balls before ``as_of``. For batter-side
    (Phase 2) features. Returns (xwoba, n_bbe) or (None, n).
    """
    vals = [r["xwoba"] for r in rows
            if r["batting_team"] == team_abbr and r["p_throws"] == hand
            and r["xwoba"] is not None
            and r["game_date"] and r["game_date"] < as_of]
    if len(vals) < min_bbe:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def asof_batter_xwoba_vs_hand(batter_id, hand, rows, as_of, min_bbe=MIN_BBE):
    """
    Mean xwOBAcon an individual batter PRODUCED vs pitchers of a given hand
    ('L'/'R'), using only batted balls before ``as_of``. For individual batter
    props (batter_hits, etc.). Returns (xwoba, n_bbe) or (None, n).
    """
    bid = str(batter_id)
    vals = [r["xwoba"] for r in rows
            if r.get("batter") == bid and r["p_throws"] == hand
            and r["xwoba"] is not None
            and r["game_date"] and r["game_date"] < as_of]
    if len(vals) < min_bbe:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    # Smoke test: two days, then an as-of estimate using only the earlier day.
    for d in ("2024-07-03", "2024-07-04"):
        rows = fetch_statcast_day(d)
        print(f"{d}: {len(rows)} pitches, "
              f"{sum(1 for r in rows if r['xwoba'] is not None)} BBE")
    rows = load_days("2024-07-03", "2024-07-04")
    # Justin Steele pitched around this window; evaluate as-of 2024-07-04.
    xw, n = asof_pitcher_xwoba(657006, rows, "2024-07-04", min_bbe=1)
    print(f"Steele as-of 2024-07-04 xwOBAcon={xw and round(xw,3)} over {n} BBE")
