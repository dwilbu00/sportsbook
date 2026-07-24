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
# Bump SCHEMA_VERSION whenever the trimmed row shape changes. v4 stored per-at-bat
# xBA (+ xwOBA). v5 (P2.4b) ADDS the plate-discipline / contact-quality raw columns
# needed for whiff%/CSW%/hard-hit%/barrel% (description, type, launch_speed,
# launch_speed_angle, launch_angle, bb_type). Bumping the version rolls CACHE_DIR to
# a fresh dir → a re-fetch (the v4 cache lacks these columns; it is not reused).
SCHEMA_VERSION = 5
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cache", f"statcast_days_v{SCHEMA_VERSION}")
LEGACY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cache", "statcast_days_v2")

# Minimum prior events before an as-of estimate is usable (batted balls for
# xwOBAcon, official at-bats for xBA).
MIN_BBE = 40
# Minimum prior PITCHES before an as-of plate-discipline rate (whiff/CSW) is usable.
MIN_PITCHES = 100


def _pf(v):
    """Parse a Statcast CSV float cell, else None."""
    try:
        return float(v) if v not in ("", "null", None) else None
    except (TypeError, ValueError):
        return None


def _pi(v):
    """Parse a Statcast CSV int cell (e.g. launch_speed_angle 1..6), else None."""
    try:
        return int(float(v)) if v not in ("", "null", None) else None
    except (TypeError, ValueError):
        return None


# `description` (Statcast pitch outcome) sets for plate-discipline rates (v5).
_SWING_DESCRIPTIONS = frozenset({
    "hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
})
_WHIFF_DESCRIPTIONS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip",
})
HARD_HIT_MPH = 95.0     # hard-hit ball threshold
BARREL_LSA = 6          # launch_speed_angle == 6 → Savant "Barrel"


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

    Returns a list of trimmed dict rows (v5):
        {game_date, pitcher, batter, p_throws, batting_team, stand, xwoba, xba,
         description, type, launch_speed, launch_speed_angle, launch_angle, bb_type}
    xwOBA is populated for batted balls; xBA once per official at-bat (0.0 for
    strikeouts) else None; the plate-discipline/contact fields drive whiff%/CSW%/
    hard-hit%/barrel% via the asof_rates aggregator.
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
            # v5 (P2.4b): plate-discipline / contact-quality raw fields.
            "description": r.get("description"),
            "type": r.get("type"),                       # S/B/X (X = batted ball)
            "launch_speed": _pf(r.get("launch_speed")),
            "launch_speed_angle": _pi(r.get("launch_speed_angle")),
            "launch_angle": _pf(r.get("launch_angle")),
            "bb_type": r.get("bb_type"),
        })
    with open(path, "w") as f:
        json.dump(trimmed, f)
    return trimmed


def fetch_range(start, end, sleep=0.5, verbose=True):
    """Ensure every day in [start, end] (inclusive, YYYY-MM-DD) is cached.

    ``end`` is capped at today: future days have no games, so fetching them just
    wastes Savant round-trips (and caches empty files)."""
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    today = _date.today()
    if d1 > today:
        d1 = today
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


def batter_asof_rates(rows, as_of, min_ab=MIN_BBE):
    """Per-batter as-of expected-BA (and xwOBAcon) over pitches before ``as_of``.

    Bulk aggregation for the OFFLINE derived-table build (roadmap 2.4a): one
    ``as_of`` (typically today) across every batter, rather than the per-game
    as-of the backtest needs. xBA is the mean at-bat ``xba`` over official ABs
    (batted balls + strikeouts, excluding walks/HBP/sacrifices — see
    ``_at_bat_xba``); xwOBAcon is the mean over batted balls. Leakage-safe by the
    same strict ``game_date < as_of`` filter as the ``asof_*`` primitives.

    Returns ``{batter_id: {"xba", "n_ab", "xwoba", "n_bbe"}}`` for batters with at
    least ``min_ab`` official ABs (xwoba is None when under ``min_ab`` batted
    balls). ``batter_id`` is the MLBAM id string (matches ``find_player_id``).
    """
    xba_agg = {}   # bid -> [sum_xba, n_ab]
    xw_agg = {}    # bid -> [sum_xwoba, n_bbe]
    for r in rows:
        gd = r.get("game_date")
        if not gd or gd >= as_of:
            continue
        bid = r.get("batter")
        if not bid:
            continue
        xb = r.get("xba")
        if xb is not None:
            a = xba_agg.setdefault(bid, [0.0, 0])
            a[0] += xb
            a[1] += 1
        xw = r.get("xwoba")
        if xw is not None:
            a = xw_agg.setdefault(bid, [0.0, 0])
            a[0] += xw
            a[1] += 1
    out = {}
    for bid, (s, n) in xba_agg.items():
        if n < min_ab:
            continue
        xw = xw_agg.get(bid)
        out[bid] = {
            "xba": s / n,
            "n_ab": n,
            "xwoba": (xw[0] / xw[1]) if (xw and xw[1] > 0) else None,
            "n_bbe": xw[1] if xw else 0,
        }
    return out


def _empty_rate_acc():
    return {"n_pitches": 0, "n_swings": 0, "n_whiff": 0, "n_called": 0,
            "n_bip": 0, "n_hardhit": 0, "n_barrel": 0}


def _accumulate_rates(acc, r):
    """Fold one v5 pitch row into a counts accumulator."""
    acc["n_pitches"] += 1
    desc = r.get("description")
    if desc in _SWING_DESCRIPTIONS:
        acc["n_swings"] += 1
    if desc in _WHIFF_DESCRIPTIONS:
        acc["n_whiff"] += 1
    if desc == "called_strike":
        acc["n_called"] += 1
    if r.get("type") == "X":                       # batted ball
        acc["n_bip"] += 1
        ls = r.get("launch_speed")
        if ls is not None and ls >= HARD_HIT_MPH:
            acc["n_hardhit"] += 1
        if r.get("launch_speed_angle") == BARREL_LSA:
            acc["n_barrel"] += 1


def _finalize_rates(acc, min_pitches):
    """Counts → rates; None if under the pitch threshold. Per-rate values are
    None when their own denominator is empty (e.g. no batted balls yet)."""
    if acc["n_pitches"] < min_pitches:
        return None

    def _ratio(num, den):
        return (num / den) if den else None

    return {
        "whiff_pct": _ratio(acc["n_whiff"], acc["n_swings"]),
        "csw_pct": _ratio(acc["n_called"] + acc["n_whiff"], acc["n_pitches"]),
        "hard_hit_pct": _ratio(acc["n_hardhit"], acc["n_bip"]),
        "barrel_pct": _ratio(acc["n_barrel"], acc["n_bip"]),
        "n_pitches": acc["n_pitches"],
        "n_bip": acc["n_bip"],
    }


def asof_rates(rows, as_of, key, min_pitches=MIN_PITCHES):
    """Per-player as-of plate-discipline / contact rates (v5), leakage-safe.

    ``key`` = "pitcher" → rates the pitcher INDUCES (whiff/CSW → strikeout props);
    "batter" → rates the batter ALLOWS (whiff/CSW) + PRODUCES (hard-hit/barrel).
    Returns ``{player_id: {whiff_pct, csw_pct, hard_hit_pct, barrel_pct,
    n_pitches, n_bip}}`` for players with >= ``min_pitches`` pitches strictly
    before ``as_of``. Mirrors ``batter_asof_rates`` but accumulates count pairs
    (numerator/denominator) per rate rather than averaging a single value.
    """
    accs = {}
    for r in rows:
        gd = r.get("game_date")
        if not gd or gd >= as_of:
            continue
        pid = r.get(key)
        if not pid:
            continue
        acc = accs.get(pid)
        if acc is None:
            acc = accs[pid] = _empty_rate_acc()
        _accumulate_rates(acc, r)
    out = {}
    for pid, acc in accs.items():
        fin = _finalize_rates(acc, min_pitches)
        if fin is not None:
            out[pid] = fin
    return out


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
