"""
Backtest harness for the sportsbook analysis model.

For each completed game in a sport's recent schedule, project the matchup
using ONLY games that occurred before it, then compare the projection to
the actual outcome. Run with different feature toggles to measure how much
each enhancement (recency, opp_strength, venue) actually helps.

Usage:
    python backtest.py --sport nba --limit 100
    python backtest.py --sport nba --limit 200 --variants baseline,recency,all
    python backtest.py --sport mlb --limit 500 --half-life 18

The script reports MAE/RMSE for projected totals and margins, plus
directional accuracy and Brier score for win-probability predictions.
No sportsbook odds are required — this measures projection quality, not
betting ROI.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from espn_client import (
    get_all_teams,
    get_team_schedule,
    annotate_opponent_strength,
)
from analysis import (
    _recency_weights,
    _weighted_mean,
    _weighted_rate,
    _half_life_for,
)


# ────────────────────────────────────────────────────────────────
#  Parameterized multiplier variants (for sweeping)
# ────────────────────────────────────────────────────────────────

def opp_strength_mult(opp_win_pct, strength):
    """
    Tunable version of analysis._opponent_strength_multiplier.
    strength is the half-range around 1.0:
      strength=0     → always 1.0 (off)
      strength=0.5   → range [0.5, 1.5] (original)
      strength=0.75  → range [0.25, 1.75] (more aggressive)
    """
    if opp_win_pct is None or strength <= 0:
        return 1.0
    clamped = max(0.0, min(1.0, opp_win_pct))
    # Map [0..1] → [1 - strength .. 1 + strength]
    return (1.0 - strength) + 2 * strength * clamped


def venue_mult(past_is_home, upcoming_is_home, strength):
    """
    Tunable version of analysis._venue_match_multiplier.
    strength is the half-spread:
      strength=0    → always 1.0 (off)
      strength=0.15 → (1.15, 0.85)
      strength=0.25 → (1.25, 0.75)
      strength=0.40 → (1.40, 0.60)
    """
    if past_is_home is None or upcoming_is_home is None or strength <= 0:
        return 1.0
    match = (past_is_home == upcoming_is_home)
    return (1.0 + strength) if match else (1.0 - strength)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache", "backtest")
os.makedirs(CACHE_DIR, exist_ok=True)


# Map our --sport CLI to (espn_sport, espn_league, sport_key) tuples.
SPORT_MAP = {
    "nba": ("basketball", "nba", "basketball_nba"),
    "nfl": ("football", "nfl", "americanfootball_nfl"),
    "mlb": ("baseball", "mlb", "baseball_mlb"),
    "nhl": ("hockey", "nhl", "icehockey_nhl"),
}


# ────────────────────────────────────────────────────────────────
#  Simple file-based cache for ESPN responses
# ────────────────────────────────────────────────────────────────

def _cache_key(*parts):
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


def cached_schedule(espn_sport, espn_league, team_id, season_year=None, ttl_hours=24 * 7):
    """
    Cached wrapper around get_team_schedule. Historical seasons get a long TTL
    (results don't change). Current season uses a shorter TTL.
    """
    path = _cache_key("schedule", espn_sport, espn_league, team_id, season_year or "current")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        # Historical seasons (specified explicitly) are immutable — long TTL
        effective_ttl = (ttl_hours * 24) if season_year else ttl_hours
        if age < effective_ttl * 3600:
            with open(path) as f:
                return json.load(f)
    games = get_team_schedule(espn_sport, espn_league, team_id, season_year=season_year)
    with open(path, "w") as f:
        json.dump(games, f)
    return games


# ────────────────────────────────────────────────────────────────
#  Projection (with feature toggles)
# ────────────────────────────────────────────────────────────────

def project_team_form(team_name, prior_games, upcoming_is_home, params):
    """
    Project a team's win rate, average scoring, average allowed, and
    average margin based on their prior games and the upcoming venue.

    params dict keys:
        half_life:        float or None (None = no recency decay)
        opp_strength:     float, half-range of opp-strength multiplier (0 = off)
        venue_strength:   float, half-spread of venue multiplier (0 = off)
    """
    if not prior_games:
        return None

    # Build per-game series
    wins, scored, allowed, margins, opp_str, past_h = [], [], [], [], [], []
    for g in prior_games:
        if g["home_team"] == team_name:
            scored.append(g["home_score"])
            allowed.append(g["away_score"])
            margins.append(g["home_score"] - g["away_score"])
            wins.append(1 if g["home_score"] > g["away_score"] else 0)
            past_h.append(True)
        elif g["away_team"] == team_name:
            scored.append(g["away_score"])
            allowed.append(g["home_score"])
            margins.append(g["away_score"] - g["home_score"])
            wins.append(1 if g["away_score"] > g["home_score"] else 0)
            past_h.append(False)
        else:
            continue
        opp_str.append(g.get("opponent_win_pct"))

    if not scored:
        return None

    base_w = _recency_weights(len(scored), params.get("half_life"))

    weights = []
    for bw, opp, ph in zip(base_w, opp_str, past_h):
        w = bw
        w *= opp_strength_mult(opp, params.get("opp_strength", 0.0))
        w *= venue_mult(ph, upcoming_is_home, params.get("venue_strength", 0.0))
        weights.append(w)

    return {
        "win_pct": _weighted_mean(wins, weights),
        "avg_scored": _weighted_mean(scored, weights),
        "avg_allowed": _weighted_mean(allowed, weights),
        "avg_margin": _weighted_mean(margins, weights),
        "sample_size": len(scored),
    }


def project_matchup(home_team, away_team, home_prior, away_prior, params):
    """Project the upcoming matchup. Returns dict with projected_total, projected_margin, home_win_prob."""
    home = project_team_form(home_team, home_prior, True, params)
    away = project_team_form(away_team, away_prior, False, params)
    if not home or not away:
        return None

    # Projected total: avg of (offense-based) and (defense-based) estimates
    projected_total = ((home["avg_scored"] + away["avg_scored"])
                       + (home["avg_allowed"] + away["avg_allowed"])) / 2

    # Projected margin (home perspective): blend home's avg margin with negative of away's avg margin
    projected_margin = (home["avg_margin"] - away["avg_margin"]) / 2

    # Home win probability: blend home's wp with (1 - away's wp)
    home_win_prob = (home["win_pct"] + (1 - away["win_pct"])) / 2

    return {
        "projected_total": projected_total,
        "projected_margin": projected_margin,
        "home_win_prob": home_win_prob,
        "home_sample": home["sample_size"],
        "away_sample": away["sample_size"],
    }


# ────────────────────────────────────────────────────────────────
#  Backtest loop
# ────────────────────────────────────────────────────────────────

def build_schedules(espn_sport, espn_league, espn_teams, season_year=None, max_workers=15):
    """Fetch all teams' schedules in parallel; return {team_id: games_list}."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(cached_schedule, espn_sport, espn_league, info["id"], season_year): info["id"]
            for info in espn_teams.values() if info.get("id")
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                results[tid] = fut.result()
            except Exception:
                results[tid] = []
    return results


def all_completed_games(schedules):
    """Flatten all schedules into a deduped list of (date, home, away, home_score, away_score)."""
    seen = set()
    games = []
    for tid, sched in schedules.items():
        for g in sched:
            key = (g.get("date"), g.get("home_team"), g.get("away_team"))
            if key in seen:
                continue
            seen.add(key)
            games.append(g)
    games.sort(key=lambda g: g.get("date") or "")
    return games


def prior_games_for(team_name, schedules, espn_teams, before_date, window):
    """
    Return up to `window` of team's most recent games that ended before before_date.
    Games are returned newest-first (matching analyzer expectations).
    """
    info = espn_teams.get(team_name)
    if not info:
        # try case-insensitive
        for name, i in espn_teams.items():
            if name.lower() == team_name.lower():
                info = i
                break
    if not info:
        return []
    sched = schedules.get(info["id"], [])
    prior = [g for g in sched if (g.get("date") or "") < before_date]
    prior.sort(key=lambda g: g.get("date") or "", reverse=True)  # newest first
    return prior[:window]


def run_backtest(sport_key, espn_sport, espn_league, limit, window, variants,
                 min_sample=5, season_year=None, sweep=False):
    # Resolve "auto" half-life in variants
    variants = {name: _resolve_params(p, sport_key) for name, p in variants.items()}

    print(f"\n=== Loading {sport_key} team list ===")
    espn_teams = get_all_teams(espn_sport, espn_league)
    print(f"Loaded {len(espn_teams)} teams")

    season_label = f"season {season_year}" if season_year else "current season"
    print(f"\n=== Fetching schedules for {season_label} (cached) ===")
    schedules = build_schedules(espn_sport, espn_league, espn_teams, season_year=season_year)
    print(f"Fetched {sum(1 for v in schedules.values() if v)} non-empty schedules")

    all_games = all_completed_games(schedules)
    print(f"Total deduped completed games: {len(all_games)}")
    # Use only games where both teams have enough prior history
    if not all_games:
        print("No games found.")
        return

    # Sample evenly across the season
    if limit and limit < len(all_games):
        # Use the most recent `limit` games (more relevant prior data accumulated)
        sample = all_games[-limit:]
    else:
        sample = all_games

    print(f"Backtesting {len(sample)} games across {len(variants)} variants...")

    # Results per variant
    results = {name: {
        "total_errors": [], "margin_errors": [],
        "correct_winner": 0, "win_prob_brier": [],
        "n": 0,
    } for name in variants}

    skipped = 0
    for game in sample:
        date = game.get("date")
        home = game.get("home_team")
        away = game.get("away_team")
        if not (date and home and away):
            skipped += 1
            continue

        # Get each team's prior games before this date
        home_prior = prior_games_for(home, schedules, espn_teams, date, window)
        away_prior = prior_games_for(away, schedules, espn_teams, date, window)

        if len(home_prior) < min_sample or len(away_prior) < min_sample:
            skipped += 1
            continue

        # Annotate opponent strength on each team's prior games
        annotate_opponent_strength(home_prior, home, espn_teams)
        annotate_opponent_strength(away_prior, away, espn_teams)

        actual_total = game.get("total_score") or (game["home_score"] + game["away_score"])
        actual_margin = game["home_score"] - game["away_score"]
        home_won = 1 if actual_margin > 0 else 0

        for variant_name, params in variants.items():
            proj = project_matchup(home, away, home_prior, away_prior, params)
            if not proj:
                continue
            r = results[variant_name]
            r["total_errors"].append(proj["projected_total"] - actual_total)
            r["margin_errors"].append(proj["projected_margin"] - actual_margin)
            predicted_home_win = 1 if proj["projected_margin"] > 0 else 0
            if predicted_home_win == home_won:
                r["correct_winner"] += 1
            # Brier score: (predicted_prob - actual_outcome)^2
            r["win_prob_brier"].append((proj["home_win_prob"] - home_won) ** 2)
            r["n"] += 1

    print(f"\nSkipped {skipped} games (insufficient prior history or missing data)")

    # Summarize
    print_results(results, sweep=sweep)


def print_results(results, sweep=False):
    rows = []
    for name, r in results.items():
        n = r["n"]
        if n == 0:
            continue
        total_mae = sum(abs(e) for e in r["total_errors"]) / n
        total_rmse = math.sqrt(sum(e * e for e in r["total_errors"]) / n)
        margin_mae = sum(abs(e) for e in r["margin_errors"]) / n
        winner_pct = r["correct_winner"] / n * 100
        brier = sum(r["win_prob_brier"]) / n
        rows.append((name, n, total_mae, total_rmse, margin_mae, winner_pct, brier))

    if sweep:
        # Show TOP 10 by Brier, then TOP 10 by Winner %, then TOP 10 by Margin MAE
        print()
        for metric_name, key, lower_better in [
            ("Brier (lower = better calibration)",   lambda r: r[6], True),
            ("Winner % (higher = better)",            lambda r: -r[5], True),  # negate for "lower=better"
            ("Margin MAE (lower = better)",           lambda r: r[4], True),
        ]:
            print("=" * 100)
            print(f"  TOP 10 by {metric_name}")
            print("=" * 100)
            sorted_rows = sorted(rows, key=key) if lower_better else sorted(rows, key=key, reverse=True)
            _print_table_header()
            for row in sorted_rows[:10]:
                _print_table_row(*row)
            print()
    else:
        print()
        _print_table_header()
        for row in rows:
            _print_table_row(*row)
        print("\nLegend:")
        print("  Total MAE/RMSE: error in projected total points (lower = better)")
        print("  Margin MAE:     error in projected home margin (lower = better)")
        print("  Winner %:       directional accuracy of winner prediction (higher = better)")
        print("  Brier:          calibration of home_win_prob (lower = better; 0.25 = always 50%)")


def _print_table_header():
    print(f"{'Variant':<30} {'N':>5}  {'Total MAE':>10}  {'Total RMSE':>11}  {'Margin MAE':>11}  {'Winner %':>10}  {'Brier':>8}")
    print("-" * 100)


def _print_table_row(name, n, tmae, trmse, mmae, wpct, brier):
    print(f"{name:<30} {n:>5}  {tmae:>10.2f}  {trmse:>11.2f}  {mmae:>11.2f}  {wpct:>9.2f}%  {brier:>8.4f}")


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def _preset(half_life, opp_strength=0.0, venue_strength=0.0):
    return {"half_life": half_life, "opp_strength": opp_strength, "venue_strength": venue_strength}


VARIANT_PRESETS = {
    # baseline: no recency decay, no opp/venue adjustment
    "baseline":   _preset(half_life=None, opp_strength=0.0,  venue_strength=0.0),
    # recency only — uses analysis.py default half-life per sport
    "recency":    _preset(half_life="auto", opp_strength=0.0,  venue_strength=0.0),
    "rec+opp":    _preset(half_life="auto", opp_strength=0.5,  venue_strength=0.0),
    "no_venue":   _preset(half_life="auto", opp_strength=0.5,  venue_strength=0.0),
    "no_opp":     _preset(half_life="auto", opp_strength=0.0,  venue_strength=0.25),
    "all":        _preset(half_life="auto", opp_strength=0.5,  venue_strength=0.25),
}


def _resolve_params(params, sport_key):
    """Replace half_life='auto' with the sport-specific default."""
    p = dict(params)
    if p.get("half_life") == "auto":
        p["half_life"] = _half_life_for(sport_key)
    return p


def _build_sweep_grid():
    """Cross-product of all parameter settings to try in --sweep mode."""
    half_lifes = [None, 3, 5, 7, 10, 15, 20]
    opp_strengths = [0.0, 0.25, 0.5, 0.75]
    venue_strengths = [0.0, 0.15, 0.25, 0.4]

    variants = {}
    for hl in half_lifes:
        for opp in opp_strengths:
            for ven in venue_strengths:
                hl_label = "none" if hl is None else f"hl{hl}"
                name = f"{hl_label}/opp{opp}/ven{ven}"
                variants[name] = _preset(half_life=hl, opp_strength=opp, venue_strength=ven)
    return variants


def main():
    p = argparse.ArgumentParser(description="Backtest the sportsbook projection model")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="nba")
    p.add_argument("--season", type=int, default=None,
                   help="ESPN season year (e.g., 2025 = 2024-25 NBA season). Default: current.")
    p.add_argument("--limit", type=int, default=200, help="Max games to backtest (most recent N)")
    p.add_argument("--window", type=int, default=10, help="Max prior games to use per team")
    p.add_argument("--min-sample", type=int, default=5, help="Skip games where either team has fewer prior games than this")
    p.add_argument("--variants", default="baseline,recency,rec+opp,all",
                   help="Comma-separated subset of: " + ",".join(VARIANT_PRESETS.keys()))
    p.add_argument("--sweep", action="store_true",
                   help="Run a parameter sweep (ignores --variants). "
                        "Tests grid of half_life × opp_strength × venue_strength.")
    args = p.parse_args()

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]

    if args.sweep:
        variants = _build_sweep_grid()
        variant_names = list(variants.keys())
        print(f"\n{'#'*60}")
        print(f"#  SWEEP MODE: {len(variants)} parameter combos")
    else:
        variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in variant_names if v not in VARIANT_PRESETS]
        if unknown:
            print(f"Unknown variants: {unknown}")
            print(f"Available: {list(VARIANT_PRESETS.keys())}")
            sys.exit(1)
        variants = {name: VARIANT_PRESETS[name] for name in variant_names}
        print(f"\n{'#'*60}")

    print(f"#  Backtest: {args.sport.upper()} ({sport_key})")
    print(f"#  Season: {args.season if args.season else 'current'}")
    print(f"#  Sample limit: {args.limit}   Prior window: {args.window}   Min sample: {args.min_sample}")
    if not args.sweep:
        print(f"#  Variants: {variant_names}")
    print(f"{'#'*60}")

    run_backtest(sport_key, espn_sport, espn_league,
                 limit=args.limit, window=args.window, variants=variants,
                 min_sample=args.min_sample, season_year=args.season,
                 sweep=args.sweep)


if __name__ == "__main__":
    main()
