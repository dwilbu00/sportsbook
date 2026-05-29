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
    get_athlete_gamelog,
    search_athlete,
    get_team_pace_factor,
    PROP_STAT_MAP,
)
# Re-export ESPN cache helpers (now live in espn_cache.py) so existing
# imports `from backtest import cached_athlete_id, cached_gamelog` keep
# working in book_line_calibration.py, eval_min_streak.py, etc.
from espn_cache import cached_athlete_id, cached_gamelog, _cache_key
from analysis import (
    _recency_weights,
    _weighted_mean,
    _weighted_rate,
    _half_life_for,
    _norm_cdf,
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


def opp_defense_mult(opp_pts_allowed, league_avg, strength):
    """
    Player-prop defense multiplier. Upweights past games played against
    tougher-than-average defenses (where the player still produced).
    strength is the half-spread:
      strength=0    → always 1.0 (off)
      strength=0.5  → scales the bounded ratio's delta-from-1 by 0.5
    The raw ratio (league_avg / opp_pts_allowed) is clamped to [0.5, 1.5],
    then its distance from 1.0 is scaled by `strength`.
    """
    if not opp_pts_allowed or not league_avg or league_avg <= 0 or strength <= 0:
        return 1.0
    ratio = league_avg / opp_pts_allowed
    bounded = max(0.5, min(1.5, ratio))
    return 1.0 + strength * (bounded - 1.0)


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


# Month each sport's regular season nominally starts. Used to derive the
# "current season cutoff" for a given game-date so cross-season games are
# never mixed into a player's prior-game window unless we explicitly opt in.
SPORT_SEASON_START_MONTH = {
    "basketball_nba": 10,
    "americanfootball_nfl": 9,
    "baseball_mlb": 3,
    "icehockey_nhl": 10,
}


# Default streak threshold (consecutive valid games required to qualify) used
# by the props backtest when filtering player gamelogs. Set to roughly the
# expected production half-life + 1 per sport. Production analysis.py uses
# the per-prop calibrated half-life instead.
SPORT_DEFAULT_MIN_STREAK = {
    "basketball_nba": 8,
    "baseball_mlb": 6,
    "americanfootball_nfl": 4,
    "icehockey_nhl": 8,
}


def _season_start_iso(date_iso, sport_key):
    """
    Return ISO date string for the most recent season-start on/before date_iso.
    e.g., for NBA + "2026-01-15" → "2025-10-01"; for "2025-11-02" → "2025-10-01".
    """
    if not date_iso:
        return None
    from datetime import date as _date
    try:
        d = _date.fromisoformat(date_iso[:10])
    except ValueError:
        return None
    start_month = SPORT_SEASON_START_MONTH.get(sport_key, 1)
    if d.month >= start_month:
        return _date(d.year, start_month, 1).isoformat()
    return _date(d.year - 1, start_month, 1).isoformat()


def _filter_to_current_season(prior_games, test_date, sport_key):
    """Return only prior_games whose game_date falls in the same season as test_date."""
    cutoff = _season_start_iso(test_date, sport_key)
    if not cutoff:
        return prior_games
    return [g for g in prior_games
            if (g.get("game_date") or "")[:10] >= cutoff]


# ────────────────────────────────────────────────────────────────
#  Simple file-based cache for ESPN responses
#  (cached_athlete_id, cached_gamelog, _cache_key now live in espn_cache.py
#   and are re-imported at the top of this file for back-compat.)
# ────────────────────────────────────────────────────────────────


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
        # Per-game series + weights for empirical-distribution / quantile use
        "totals_series": [s + a for s, a in zip(scored, allowed)],
        "margins_series": list(margins),
        "weights": list(weights),
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

    # Per-matchup empirical distribution of plausible (total, margin) values,
    # built by convolving each team's prior-game series with the other team's.
    # Used for per-matchup quantile alt-line analysis (variance auto-scales).
    convolved_totals, convolved_margins, convolved_weights = [], [], []
    h_tot, h_mar, h_w = home["totals_series"], home["margins_series"], home["weights"]
    a_tot, a_mar, a_w = away["totals_series"], away["margins_series"], away["weights"]
    for ht, hm, hw in zip(h_tot, h_mar, h_w):
        for at, am, aw in zip(a_tot, a_mar, a_w):
            convolved_totals.append((ht + at) / 2)
            convolved_margins.append((hm - am) / 2)
            convolved_weights.append(hw * aw)

    return {
        "projected_total": projected_total,
        "projected_margin": projected_margin,
        "home_win_prob": home_win_prob,
        "home_sample": home["sample_size"],
        "away_sample": away["sample_size"],
        "total_samples": convolved_totals,
        "margin_samples": convolved_margins,
        "sample_weights": convolved_weights,
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
                 min_sample=5, season_year=None, sweep=False,
                 quantile_mode=False, safe_target=0.80):
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
    def _empty_q_bucket():
        return {q: {"hits": 0, "n": 0, "cushions": []}
                for q in QUANTILE_THRESHOLDS}
    results = {name: {
        "total_errors": [], "margin_errors": [],
        "correct_winner": 0, "win_prob_brier": [],
        "n": 0,
        "quantile": {
            "total_over":  _empty_q_bucket() if quantile_mode else {},
            "total_under": _empty_q_bucket() if quantile_mode else {},
            "home_cover":  _empty_q_bucket() if quantile_mode else {},
            "away_cover":  _empty_q_bucket() if quantile_mode else {},
        } if quantile_mode else {},
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

            # Per-matchup empirical-distribution quantile alt-line tracking
            if quantile_mode and proj.get("total_samples"):
                tot_samples = proj["total_samples"]
                mar_samples = proj["margin_samples"]
                wts = proj["sample_weights"]
                proj_total = proj["projected_total"]
                proj_margin = proj["projected_margin"]
                for q in QUANTILE_THRESHOLDS:
                    # Total OVER: alt = lower-tail quantile of total distribution
                    alt_to = _weighted_quantile(tot_samples, wts, q)
                    if alt_to is not None:
                        tl = r["quantile"]["total_over"][q]
                        tl["n"] += 1
                        tl["cushions"].append(proj_total - alt_to)
                        if actual_total > alt_to:
                            tl["hits"] += 1
                    # Total UNDER: alt = upper-tail quantile (1-q)
                    alt_tu = _weighted_quantile(tot_samples, wts, 1.0 - q) if q > 0 else _weighted_quantile(tot_samples, wts, 1.0)
                    if alt_tu is not None:
                        tl = r["quantile"]["total_under"][q]
                        tl["n"] += 1
                        tl["cushions"].append(alt_tu - proj_total)
                        if actual_total < alt_tu:
                            tl["hits"] += 1
                    # HOME cover: alt = lower-tail quantile of margin (worst case for home)
                    alt_hc = _weighted_quantile(mar_samples, wts, q)
                    if alt_hc is not None:
                        tl = r["quantile"]["home_cover"][q]
                        tl["n"] += 1
                        tl["cushions"].append(proj_margin - alt_hc)
                        if actual_margin > alt_hc:
                            tl["hits"] += 1
                    # AWAY cover: alt = upper-tail quantile of margin (worst case for away)
                    alt_ac = _weighted_quantile(mar_samples, wts, 1.0 - q) if q > 0 else _weighted_quantile(mar_samples, wts, 1.0)
                    if alt_ac is not None:
                        tl = r["quantile"]["away_cover"][q]
                        tl["n"] += 1
                        tl["cushions"].append(alt_ac - proj_margin)
                        if actual_margin < alt_ac:
                            tl["hits"] += 1

    print(f"\nSkipped {skipped} games (insufficient prior history or missing data)")

    # Summarize
    print_results(results, sweep=sweep)

    if quantile_mode:
        if sweep:
            _print_matchup_quantile_sweep_summary(results, safe_target=safe_target)
        else:
            _print_matchup_quantile_results(results)


def _print_matchup_quantile_sweep_summary(results, safe_target=0.80):
    """
    Compact sweep summary for matchup quantile mode. For each bet type,
    rank variants by the smallest q-level whose actual hit% ≥ safe_target
    and the median cushion at that q. Lower median cushion = better
    (less shading needed to reach safety target).
    """
    bet_labels = {
        "total_over":  "Total OVER",
        "total_under": "Total UNDER",
        "home_cover":  "HOME cover",
        "away_cover":  "AWAY cover",
    }

    def _resolve(qmap):
        """Find smallest q whose hit% ≥ target; return (q, median_cushion, hit%, n)."""
        for q in sorted(qmap.keys(), reverse=True):  # 0.20 → 0.15 → ... → 0.0
            t = qmap[q]
            if t["n"] == 0:
                continue
            hit = t["hits"] / t["n"]
            if hit >= safe_target:
                cu = sorted(t["cushions"])
                med = cu[len(cu) // 2]
                return q, med, hit * 100.0, t["n"]
        return None

    print()
    print("=" * 100)
    print(f"  MATCHUP QUANTILE SWEEP — best variants @ target hit-rate {safe_target*100:.0f}%")
    print("=" * 100)
    print("  For each bet type: smallest q whose actual hit% reaches the target,")
    print("  ranked by median cushion (lower = less shading from projection).")

    for bet_key, bet_label in bet_labels.items():
        rows = []
        for vname, r in results.items():
            qmap = (r.get("quantile") or {}).get(bet_key) or {}
            if not qmap:
                continue
            resolved = _resolve(qmap)
            if resolved is None:
                continue
            q, med, hit, n = resolved
            rows.append((vname, q, med, hit, n))
        if not rows:
            continue
        rows.sort(key=lambda x: x[2])  # lower median cushion = better

        header = f"{'Variant':<30} {'q':>5}  {'targ':>5}  {'hit%':>7}  {'med_cu':>8}  {'n':>5}"
        print()
        print(f"── {bet_label} (target {safe_target*100:.0f}%) " +
              "─" * max(0, 70 - len(bet_label)))
        print(header)
        print("-" * len(header))
        for vname, q, med, hit, n in rows[:15]:
            target = (1.0 - q) * 100.0
            print(f"{vname:<30} {q:>5.2f}  {target:>4.0f}%  {hit:>6.2f}%  "
                  f"{med:>8.2f}  {n:>5}")


def _print_matchup_quantile_results(results):
    """
    Per-matchup empirical-distribution quantile alt-line analysis for game
    totals and spreads. The alt-line is the q-quantile of the empirical
    distribution built by convolving each team's prior-game series with
    the other's — so cushion auto-scales to that specific matchup's
    plausible-outcome spread (high-pace shootouts get wider cushions
    than grind-it-out games).
    """
    bet_labels = {
        "total_over":  ("Total OVER",  "alt = low-tail of total dist; win if actual_total > alt"),
        "total_under": ("Total UNDER", "alt = high-tail of total dist; win if actual_total < alt"),
        "home_cover":  ("HOME cover",  "alt = low-tail of margin dist; win if actual_margin > alt"),
        "away_cover":  ("AWAY cover",  "alt = high-tail of margin dist; win if actual_margin < alt"),
    }

    print()
    print("=" * 100)
    print("  PER-MATCHUP QUANTILE ALT-LINE analysis (game totals + spreads)")
    print("=" * 100)
    print("  Strategy: alt_line = q-quantile of empirical (team_A × team_B) distribution.")
    print("  Cushion auto-scales to each matchup's variance — no flat scalar needed.")

    for vname, r in results.items():
        qmap_all = r.get("quantile") or {}
        if not qmap_all:
            continue
        print()
        print(f"── Variant: {vname} " + "─" * max(0, 80 - len(vname)))
        for bet_key, (bet_label, bet_desc) in bet_labels.items():
            qmap = qmap_all.get(bet_key) or {}
            if not qmap:
                continue
            print(f"  {bet_label}:  ({bet_desc})")
            header = (f"    {'target':>8}  {'q':>5}  {'n':>5}  {'hit%':>7}  "
                      f"{'mean_cu':>9}  {'median_cu':>10}  {'max_cu':>8}")
            print(header)
            print("    " + "-" * (len(header) - 4))
            for q in sorted(qmap.keys(), reverse=True):
                t = qmap[q]
                if t["n"] == 0:
                    continue
                target = (1.0 - q) * 100.0
                hit = t["hits"] / t["n"] * 100.0
                cu = t["cushions"]
                mean_cu = sum(cu) / len(cu)
                sorted_cu = sorted(cu)
                med_cu = sorted_cu[len(sorted_cu) // 2]
                max_cu = max(cu)
                tag = "extreme" if q == 0.0 else f"{target:.0f}%"
                print(f"    {tag:>8}  {q:>5.2f}  {t['n']:>5}  {hit:>6.2f}%  "
                      f"{mean_cu:>9.2f}  {med_cu:>10.2f}  {max_cu:>8.2f}")

    print()
    print("Reading guide:")
    print("  - 'target' = nominal confidence level (1 − q). Compare against achieved hit%.")
    print("  - mean/median/max cushion = how far our alt line sits from our projection.")
    print("  - Small cushion = closer to book lines (better payout); larger = safer hit.")


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

def _preset(half_life, opp_strength=0.0, venue_strength=0.0,
            opp_defense_strength=0.0, use_minutes=False,
            pace_adj=0.0, def_adj=0.0,
            shrink_k=0.0, rest_adj=0.0, def_window=None):
    return {
        "half_life": half_life,
        "opp_strength": opp_strength,
        "venue_strength": venue_strength,
        "opp_defense_strength": opp_defense_strength,
        # Player-props enhancements (output-side adjustments):
        "use_minutes": use_minutes,          # rate-based projection × proj_min
        "pace_adj": pace_adj,                # strength of opponent-pace output mult
        "def_adj": def_adj,                  # strength of opponent-defense output mult
        # New (this round):
        "shrink_k": shrink_k,                # Bayesian shrinkage toward unweighted prior mean
        "rest_adj": rest_adj,                # B2B / rest-days projection scaling
        "def_window": def_window,            # use only last N opp games for defense (None = season)
    }


VARIANT_PRESETS = {
    # ── Matchup-mode presets ──
    "baseline":     _preset(half_life=None,   opp_strength=0.0, venue_strength=0.0),
    "recency":      _preset(half_life="auto", opp_strength=0.0, venue_strength=0.0),
    "rec+opp":      _preset(half_life="auto", opp_strength=0.5, venue_strength=0.0),
    "no_venue":     _preset(half_life="auto", opp_strength=0.5, venue_strength=0.0),
    "no_opp":       _preset(half_life="auto", opp_strength=0.0, venue_strength=0.25),
    "all":          _preset(half_life="auto", opp_strength=0.5, venue_strength=0.25),
    # ── Props-mode presets (use opp_defense_strength instead of opp_strength) ──
    "rec+defense":  _preset(half_life="auto", opp_defense_strength=0.5),
    "props_all":    _preset(half_life="auto", opp_defense_strength=0.5, venue_strength=0.25),
    # ── Props-mode enhancement presets ──
    "min_only":     _preset(half_life="auto", use_minutes=True),
    "pace_only":    _preset(half_life="auto", pace_adj=1.0),
    "def_only":     _preset(half_life="auto", def_adj=1.0),
    "min+pace":     _preset(half_life="auto", use_minutes=True, pace_adj=1.0),
    "min+pace+def": _preset(half_life="auto", use_minutes=True, pace_adj=1.0, def_adj=1.0),
    # ── New feature isolation variants (Bayesian shrink / rest / def-window) ──
    "def+shrink3":  _preset(half_life="auto", def_adj=1.0, shrink_k=3),
    "def+shrink5":  _preset(half_life="auto", def_adj=1.0, shrink_k=5),
    "def+shrink10": _preset(half_life="auto", def_adj=1.0, shrink_k=10),
    "def+rest3":    _preset(half_life="auto", def_adj=1.0, rest_adj=0.03),
    "def+rest5":    _preset(half_life="auto", def_adj=1.0, rest_adj=0.05),
    "def+rest10":   _preset(half_life="auto", def_adj=1.0, rest_adj=0.10),
    "def+defwin5":  _preset(half_life="auto", def_adj=1.0, def_window=5),
    "def+defwin10": _preset(half_life="auto", def_adj=1.0, def_window=10),
    "def+defwin15": _preset(half_life="auto", def_adj=1.0, def_window=15),
    "def+all_new":  _preset(half_life="auto", def_adj=1.0,
                            shrink_k=5, rest_adj=0.05, def_window=10),
}


def _resolve_params(params, sport_key):
    """Replace half_life='auto' with the sport-specific default."""
    p = dict(params)
    if p.get("half_life") == "auto":
        p["half_life"] = _half_life_for(sport_key)
    return p


def _build_sweep_grid():
    """Cross-product of all parameter settings to try in --sweep mode (matchup)."""
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


def _build_props_sweep_grid():
    """
    Cross-product for player-props sweep mode.
    Tunes the three knobs that empirically move the needle:
      • half_life (recency decay)
      • def_adj   (output-side opponent-defense scaling — best single feature)
      • venue     (venue-match weighting)
    """
    half_lifes = [None, 3, 5, 7, 10, 15, 20]
    def_adjs = [0.0, 0.5, 1.0, 1.5]
    venue_strengths = [0.0, 0.15, 0.25]

    variants = {}
    for hl in half_lifes:
        for da in def_adjs:
            for vs in venue_strengths:
                hl_label = "none" if hl is None else f"hl{hl}"
                name = f"{hl_label}/defadj{da}/ven{vs}"
                variants[name] = _preset(half_life=hl, def_adj=da,
                                         venue_strength=vs)
    return variants


# ────────────────────────────────────────────────────────────────
#  Player-props backtest
# ────────────────────────────────────────────────────────────────

# Default starter lists per sport (used when --players is not provided).
DEFAULT_STARTERS = {
    "nba": [
        "Cade Cunningham",
        "Nikola Jokic",
        "Victor Wembanyama",
        "Jalen Brunson",
        "Kevin Durant",
        "Scottie Barnes",
        "Desmond Bane",
        "Mikal Bridges",
        "Shai Gilgeous-Alexander",
        "Luka Doncic",
        "Jayson Tatum",
        "Anthony Edwards",
        "Devin Booker",
        "Donovan Mitchell",
        "Tyrese Haliburton",
        "Damian Lillard",
        "LaMelo Ball",
        "Trae Young",
    ],
    "mlb": [
        # Top hitters (used for batter_hits)
        "Aaron Judge",
        "Shohei Ohtani",
        "Mookie Betts",
        "Juan Soto",
        "Freddie Freeman",
        "Bobby Witt Jr.",
        "Jose Altuve",
        "Vladimir Guerrero Jr.",
        "Yordan Alvarez",
        "Bryce Harper",
        # Top starting pitchers (used for pitcher_strikeouts)
        "Gerrit Cole",
        "Tarik Skubal",
        "Zack Wheeler",
        "Logan Webb",
        "Corbin Burnes",
        "Spencer Strider",
        "Pablo Lopez",
        "Cole Ragans",
    ],
    "nfl": [
        # Pass yards / pass TDs
        "Patrick Mahomes",
        "Josh Allen",
        "Lamar Jackson",
        "Joe Burrow",
        "Jalen Hurts",
        "Jared Goff",
        # Rush yards
        "Christian McCaffrey",
        "Saquon Barkley",
        "Derrick Henry",
        "Bijan Robinson",
        "Jahmyr Gibbs",
        # Receiving yds / anytime TD
        "Tyreek Hill",
        "Justin Jefferson",
        "CeeDee Lamb",
        "Ja'Marr Chase",
        "Amon-Ra St. Brown",
    ],
}

DEFAULT_PROPS = {
    "nba": ["player_points", "player_rebounds", "player_assists"],
    "mlb": ["batter_hits", "pitcher_strikeouts"],
    "nfl": ["player_pass_yds", "player_rush_yds", "player_anytime_td"],
}

PROP_LABELS_SHORT = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "batter_hits": "H",
    "pitcher_strikeouts": "K",
    "pitcher_outs": "IPx3",
    "batter_strikeouts": "BK",
    "player_pass_yds": "PaYd",
    "player_rush_yds": "RuYd",
    "player_anytime_td": "TD",
}


# Per-prop offset grids for safe-mode (alt-line) evaluation. Offsets reflect
# how books typically space alt lines (PTS in 1.0 steps, REB/AST in 0.5 steps).
SAFE_MODE_OFFSETS = {
    "player_points":   [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "player_rebounds": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "player_assists":  [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
}


# Fine-grained offsets for cushion-sweep mode. Smaller step sizes so we can
# identify the sweet spot between cushion and payout per prop.
CUSHION_SWEEP_OFFSETS = {
    "player_points":   [round(x * 0.5, 2) for x in range(0, 21)],   # 0 → 10 in 0.5 steps
    "player_rebounds": [round(x * 0.25, 2) for x in range(0, 17)],  # 0 → 4 in 0.25 steps
    "player_assists":  [round(x * 0.25, 2) for x in range(0, 17)],  # 0 → 4 in 0.25 steps
}


# Per-player quantile thresholds. Each q value = lower-tail quantile of the
# player's own weighted prior-game distribution used as the OVER alt line.
# q=0.20 ⇒ player historically clears 80% of the time, q=0.05 ⇒ 95%, etc.
QUANTILE_THRESHOLDS = [0.20, 0.15, 0.10, 0.05, 0.00]


def _weighted_quantile(values, weights, q):
    """
    Weighted empirical quantile. Returns the smallest value v such that
    the cumulative weight ≤ v is ≥ q · total_weight.
    q=0.0 returns the minimum, q=1.0 returns the maximum.
    """
    if not values:
        return None
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    target = q * total
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            return v
    return pairs[-1][0]


def fetch_player_data(espn_sport, espn_league, players, season_year=None):
    """Resolve each player → (athlete_id, gamelog). Returns {name: gamelog_list}."""
    data = {}
    for name in players:
        aid = cached_athlete_id(espn_sport, espn_league, name)
        if not aid:
            print(f"  [skip] {name}: athlete not found")
            continue
        gamelog = cached_gamelog(espn_sport, espn_league, aid,
                                 season_year=season_year)
        if not gamelog:
            print(f"  [skip] {name}: empty gamelog")
            continue
        # Sort newest-first by game_date
        gamelog.sort(key=lambda g: g.get("game_date") or "", reverse=True)
        data[name] = gamelog
        print(f"  [ok]   {name}: {len(gamelog)} games")
    return data


def _stat_label_for(prop_key, gamelog):
    """Pick the first PROP_STAT_MAP label that actually appears in the gamelog."""
    for label in PROP_STAT_MAP.get(prop_key, []):
        if any(label in g for g in gamelog):
            return label
    return None


def cached_pace_factor(espn_sport, espn_league, team_id, season_year=None,
                       ttl_hours=24 * 7):
    """Cache the per-team pace factor (long TTL — only updated once per game)."""
    path = _cache_key("pace", espn_sport, espn_league, team_id, season_year or "current")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < ttl_hours * 3600:
            with open(path) as f:
                return json.load(f).get("pace")
    pace = get_team_pace_factor(espn_sport, espn_league, team_id, season_year=season_year)
    with open(path, "w") as f:
        json.dump({"pace": pace}, f)
    return pace


def _team_pace_lookup(espn_sport, espn_league, season_year=None):
    """
    Build {team_display_name: pace_factor} from ESPN core stats. Returns
    (lookup, league_avg).
    """
    espn_teams = get_all_teams(espn_sport, espn_league)
    lookup = {}
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {
            pool.submit(cached_pace_factor, espn_sport, espn_league, info["id"], season_year):
                name
            for name, info in espn_teams.items() if info.get("id")
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                pace = fut.result()
                if pace is not None:
                    lookup[name] = pace
            except Exception:
                continue
    league_avg = (sum(lookup.values()) / len(lookup)) if lookup else None
    return lookup, league_avg


def _team_defense_lookup(espn_sport, espn_league, season_year=None):
    """
    Build defense lookup structures from team schedules. Returns
    (avg_lookup, series_lookup, league_avg) where:
      avg_lookup    = {team_name: season_avg_pts_allowed}
      series_lookup = {team_name: [(date_iso, pts_allowed), ...]} sorted desc by date
      league_avg    = float average of season-avg values
    """
    espn_teams = get_all_teams(espn_sport, espn_league)
    schedules = build_schedules(espn_sport, espn_league, espn_teams,
                                season_year=season_year)
    lookup = {}
    series = {}
    for team_name, info in espn_teams.items():
        tid = info.get("id")
        if not tid:
            continue
        sched = schedules.get(tid, [])
        rows = []
        for g in sched:
            try:
                if g["home_team"] == team_name:
                    allowed = g["away_score"]
                elif g["away_team"] == team_name:
                    allowed = g["home_score"]
                else:
                    continue
                date = g.get("date") or g.get("gameDate")
                if allowed is not None:
                    rows.append((date, allowed))
            except (KeyError, TypeError):
                continue
        if rows:
            rows.sort(key=lambda r: r[0] or "", reverse=True)
            allowed_vals = [a for _, a in rows]
            lookup[team_name] = sum(allowed_vals) / len(allowed_vals)
            series[team_name] = rows
    league_avg = (sum(lookup.values()) / len(lookup)) if lookup else None
    return lookup, series, league_avg


def _resolve_opp_pa_windowed(opp_name, test_date, team_series, window):
    """
    Return avg pts allowed by `opp_name` using only their `window` most-recent
    games STRICTLY BEFORE `test_date`. Avoids look-ahead leakage in backtest.
    Returns None if not enough data.
    """
    if not opp_name or not team_series or not window:
        return None
    # Tolerant team-name lookup (same logic as _resolve_opp_pts_allowed)
    rows = team_series.get(opp_name)
    if rows is None:
        lo = opp_name.lower()
        for k, v in team_series.items():
            kl = k.lower()
            if lo in kl or kl in lo or kl.split()[-1] == lo or lo.split()[-1] == kl.split()[-1]:
                rows = v
                break
    if not rows:
        return None
    cutoff = test_date or ""
    prior = [a for d, a in rows if d and d < cutoff]
    if not prior:
        return None
    prior = prior[:window]
    return sum(prior) / len(prior)


def _resolve_opp_pts_allowed(opp_name, team_defense):
    """Tolerant lookup — try exact, then partial substring match."""
    if not opp_name or not team_defense:
        return None
    if opp_name in team_defense:
        return team_defense[opp_name]
    lo = opp_name.lower()
    for k, v in team_defense.items():
        kl = k.lower()
        if lo in kl or kl in lo or kl.split()[-1] == lo or lo.split()[-1] == kl.split()[-1]:
            return v
    return None


def run_player_props_backtest(sport, espn_sport, espn_league, sport_key,
                              players, props, games_per_player, min_sample,
                              variants, sweep=False, season_year=None,
                              safe_mode=False, cushion_sweep=False,
                              safe_target=0.80, quantile_mode=False,
                              calibrate=False, cross_season="strict"):
    variants = {name: _resolve_params(p, sport_key) for name, p in variants.items()}
    # Sweep mode + cushion-sweep mode always use the fine-grained offsets so
    # the cushion-for-target metric is well-resolved. Coarse offsets only
    # apply for a non-sweep, plain --safe-mode invocation.
    use_fine_offsets = cushion_sweep or sweep
    offsets_by_prop = CUSHION_SWEEP_OFFSETS if use_fine_offsets else SAFE_MODE_OFFSETS
    if cushion_sweep:
        safe_mode = True
    # In sweep mode we always need safe-mode tallies (for the cushion ranking).
    if sweep:
        safe_mode = True

    season_label = f" (season {season_year})" if season_year else ""
    print(f"\n=== Fetching gamelogs for {len(players)} players{season_label} ===")
    player_data = fetch_player_data(espn_sport, espn_league, players,
                                    season_year=season_year)
    if not player_data:
        print("No player data resolved. Aborting.")
        return

    # ── Reliability filter (streak-based: see prop_filter.py) ──
    # Drops low-minutes games, the 1 game before a layoff, the 1st-back game
    # after a layoff, and any games sitting inside a too-short consecutive
    # run. All variants in the sweep are evaluated on the SAME filtered set
    # using a sport-specific `min_streak` so Brier comparisons stay apples-
    # to-apples. Production analysis.py applies the per-prop calibrated hl.
    from prop_filter import filter_player_gamelog

    sport_min_streak = SPORT_DEFAULT_MIN_STREAK.get(sport_key, 5)

    unique_team_ids = set()
    for gl in player_data.values():
        for g in gl:
            tid = g.get("team_id")
            if tid:
                unique_team_ids.add(str(tid))
    team_schedules_for_filter = {}
    if unique_team_ids:
        print(f"\n=== Fetching team schedules for reliability filter "
              f"({len(unique_team_ids)} teams) ===")
        for tid in unique_team_ids:
            try:
                team_schedules_for_filter[tid] = cached_schedule(
                    espn_sport, espn_league, tid, season_year=season_year)
            except Exception:
                team_schedules_for_filter[tid] = []

    filter_stats = {"low_min": 0, "pre_layoff": 0, "post_layoff": 0,
                    "short_streak": 0}
    filtered_player_data = {}
    for name, gl in player_data.items():
        # Use most-recent team_id (handles mid-season trades).
        tid = next((g.get("team_id") for g in gl if g.get("team_id")), None)
        sched = team_schedules_for_filter.get(str(tid)) if tid else None
        filt = filter_player_gamelog(gl, sched, sport_key,
                                     min_streak=sport_min_streak)
        filter_stats["low_min"] += filt["n_excluded_low_min"]
        filter_stats["pre_layoff"] += filt["n_excluded_pre_layoff"]
        filter_stats["post_layoff"] += filt["n_excluded_post_layoff"]
        filter_stats["short_streak"] += filt["n_excluded_short_streak"]
        if not filt["eligible_games"]:
            print(f"  [drop] {name}: 0 eligible games (curr_streak="
                  f"{filt['current_streak']} < {sport_min_streak})")
            continue
        filtered_player_data[name] = filt["eligible_games"]
    print(f"=== Reliability filter (min_streak={sport_min_streak}): dropped "
          f"{filter_stats['low_min']} low-min, "
          f"{filter_stats['pre_layoff']} pre-layoff, "
          f"{filter_stats['post_layoff']} post-layoff, "
          f"{filter_stats['short_streak']} short-streak games ===")
    player_data = filtered_player_data

    # Build team-defense lookup if any variant uses defense weighting OR
    # the output-side defense adjustment.
    team_defense, team_defense_series, league_avg_def = {}, {}, None
    needs_defense = any(
        p.get("opp_defense_strength", 0.0) > 0 or p.get("def_adj", 0.0) > 0
        for p in variants.values()
    )
    if needs_defense:
        print("\n=== Fetching team schedules for defense lookup ===")
        team_defense, team_defense_series, league_avg_def = _team_defense_lookup(
            espn_sport, espn_league, season_year=season_year)
        print(f"Built defense lookup for {len(team_defense)} teams "
              f"(league avg pts allowed = {league_avg_def:.1f})" if league_avg_def
              else "Defense lookup empty.")

    # Build team-pace lookup if any variant uses pace adjustment.
    team_pace, league_avg_pace = {}, None
    needs_pace = any(p.get("pace_adj", 0.0) > 0 for p in variants.values())
    if needs_pace:
        print("\n=== Fetching team pace factors ===")
        team_pace, league_avg_pace = _team_pace_lookup(
            espn_sport, espn_league, season_year=season_year)
        print(f"Built pace lookup for {len(team_pace)} teams "
              f"(league avg pace = {league_avg_pace:.2f})" if league_avg_pace
              else "Pace lookup empty.")

    # results[variant][prop] = {errors, n, hits, decisive, safe[offset]={"hits":, "n":}}
    # When calibrate=True, also collect per-observation tuples for residual-
    # calibration analysis: (projected, synthetic_line, actual, empirical_over).
    results = {
        vname: {prop: {
            "errors": [], "n": 0, "hits": 0, "decisive": 0,
            "safe": {off: {"hits": 0, "n": 0}
                     for off in offsets_by_prop.get(prop, [])},
            "quantile": {q: {"hits": 0, "n": 0, "cushions": []}
                         for q in QUANTILE_THRESHOLDS} if quantile_mode else {},
            "calib_obs": [] if calibrate else None,
        } for prop in props}
        for vname in variants
    }

    total_observations = 0
    skipped = 0

    for name, gamelog in player_data.items():
        test_slice = gamelog[:games_per_player]
        for prop_key in props:
            stat_label = _stat_label_for(prop_key, gamelog)
            if not stat_label:
                continue

            for i, test_game in enumerate(test_slice):
                actual = test_game.get(stat_label)
                if actual is None:
                    continue
                # NOTE: the upstream reliability filter has already removed
                # low-minutes and layoff-window games from `gamelog`, so we
                # don't need a per-test min-played check here anymore.
                prior_games = gamelog[i + 1:]

                # Strict-season policy: drop prior games from earlier seasons
                # so we never project a player using stale (different team /
                # role / coach) data. `all` keeps the old cross-season pool.
                if cross_season == "strict":
                    test_date = test_game.get("game_date")
                    prior_games = _filter_to_current_season(
                        prior_games, test_date, sport_key)

                if len(prior_games) < min_sample:
                    skipped += 1
                    continue

                prior_values = [g.get(stat_label, 0.0) for g in prior_games]
                prior_minutes = [g.get("MIN", 0.0) for g in prior_games]
                prior_home_aways = [g.get("is_home") for g in prior_games]
                prior_opponents = [g.get("opponent") for g in prior_games]
                upcoming_is_home = test_game.get("is_home")
                upcoming_opp = test_game.get("opponent")
                upcoming_date = test_game.get("game_date")

                # Days of rest before the upcoming game. ESPN game_date is ISO
                # like "2025-03-14T..."; subtract whole-date components.
                days_rest = None
                if upcoming_date and prior_games:
                    latest_prior_date = prior_games[0].get("game_date")
                    if latest_prior_date:
                        try:
                            from datetime import date as _date
                            d1 = _date.fromisoformat(upcoming_date[:10])
                            d0 = _date.fromisoformat(latest_prior_date[:10])
                            days_rest = (d1 - d0).days
                        except (ValueError, TypeError):
                            days_rest = None

                # Synthetic line = unweighted season avg of prior games. Used
                # to evaluate directional (O/U) hit-rate of each variant.
                synthetic_line = sum(prior_values) / len(prior_values)
                actual_over = 1 if actual > synthetic_line else (0 if actual < synthetic_line else None)

                for vname, params in variants.items():
                    hl = params.get("half_life")
                    base_w = _recency_weights(len(prior_values), hl)
                    venue_s = params.get("venue_strength", 0.0)
                    def_s = params.get("opp_defense_strength", 0.0)
                    weights = []
                    for bw, ph, opp in zip(base_w, prior_home_aways, prior_opponents):
                        w = bw * venue_mult(ph, upcoming_is_home, venue_s)
                        if def_s > 0:
                            opp_pa = _resolve_opp_pts_allowed(opp, team_defense)
                            w *= opp_defense_mult(opp_pa, league_avg_def, def_s)
                        weights.append(w)

                    # ── Base projection ──
                    if params.get("use_minutes"):
                        # Rate-based: per-minute weighted mean × projected minutes.
                        rates = [v / m for v, m in zip(prior_values, prior_minutes) if m and m > 0]
                        rate_weights = [w for w, m in zip(weights, prior_minutes) if m and m > 0]
                        min_weights = weights
                        if rates and sum(rate_weights) > 0 and sum(min_weights) > 0:
                            per_min_rate = _weighted_mean(rates, rate_weights)
                            proj_min = _weighted_mean(prior_minutes, min_weights)
                            projected = per_min_rate * proj_min
                        else:
                            projected = _weighted_mean(prior_values, weights)
                    else:
                        projected = _weighted_mean(prior_values, weights)

                    # ── Bayesian shrinkage toward unweighted prior mean ──
                    # Regularizes the recency-weighted projection toward the
                    # equal-weight (season-long) mean by `shrink_k` pseudo-obs.
                    shrink_k = params.get("shrink_k", 0.0) or 0.0
                    if shrink_k > 0 and prior_values:
                        unweighted_mean = sum(prior_values) / len(prior_values)
                        eff_n = sum(weights) if weights else 0.0
                        if eff_n + shrink_k > 0:
                            projected = ((eff_n * projected) + (shrink_k * unweighted_mean)) / (eff_n + shrink_k)

                    # ── Output-side adjustments based on upcoming opponent ──
                    pace_s = params.get("pace_adj", 0.0)
                    if pace_s > 0 and team_pace and league_avg_pace:
                        opp_pace = _resolve_opp_pts_allowed(upcoming_opp, team_pace)
                        if opp_pace:
                            projected *= 1.0 + pace_s * (opp_pace / league_avg_pace - 1.0)

                    output_def_s = params.get("def_adj", 0.0)
                    def_window = params.get("def_window")
                    if output_def_s > 0 and team_defense and league_avg_def:
                        if def_window and team_defense_series:
                            opp_pa = _resolve_opp_pa_windowed(
                                upcoming_opp, upcoming_date,
                                team_defense_series, def_window)
                            # Fallback to season avg when no windowed data
                            if opp_pa is None:
                                opp_pa = _resolve_opp_pts_allowed(upcoming_opp, team_defense)
                        else:
                            opp_pa = _resolve_opp_pts_allowed(upcoming_opp, team_defense)
                        if opp_pa:
                            projected *= 1.0 + output_def_s * (opp_pa / league_avg_def - 1.0)

                    # ── Rest-days adjustment ──
                    # rest_adj is the size of the B2B penalty (e.g., 0.05 = −5%).
                    # Only apply on B2B (days_rest == 1); leave normal/long rest alone.
                    rest_adj = params.get("rest_adj", 0.0) or 0.0
                    if rest_adj > 0 and days_rest is not None and days_rest <= 1:
                        projected *= (1.0 - rest_adj)

                    err = projected - actual
                    cell = results[vname][prop_key]
                    cell["errors"].append(err)
                    cell["n"] += 1

                    # Hit-rate vs synthetic line
                    if actual_over is not None:
                        predicted_over = 1 if projected > synthetic_line else (
                            0 if projected < synthetic_line else None)
                        if predicted_over is not None:
                            cell["decisive"] += 1
                            if predicted_over == actual_over:
                                cell["hits"] += 1

                    # Safe-mode (alt-line) hit tracking: bet OVER on
                    # (projected - offset). Win if actual > alt_line.
                    for off, tally in cell["safe"].items():
                        alt_line = projected - off
                        tally["n"] += 1
                        if actual > alt_line:
                            tally["hits"] += 1

                    # Calibration data capture: store enough to compare
                    # empirical-CDF vs residual-calibrated forecasters later.
                    if calibrate and cell["calib_obs"] is not None:
                        if sum(weights) > 0:
                            empirical_over = _weighted_rate(
                                prior_values, weights, lambda v: v > synthetic_line)
                        else:
                            empirical_over = 0.5
                        cell["calib_obs"].append(
                            (name, projected, synthetic_line, actual, empirical_over,
                             upcoming_date or ""))

                    # Per-player weighted-quantile alt line. Cushion auto-scales
                    # to that player's own variance instead of a flat scalar.
                    if quantile_mode and cell["quantile"]:
                        for q, qtally in cell["quantile"].items():
                            alt = _weighted_quantile(prior_values, weights, q)
                            if alt is None:
                                continue
                            qtally["n"] += 1
                            qtally["cushions"].append(projected - alt)
                            if actual > alt:
                                qtally["hits"] += 1

                    total_observations += 1

    print(f"\nProcessed {total_observations} (player, prop, game) observations")
    print(f"Skipped {skipped} game-prop observations (insufficient prior history)")

    if sweep:
        _print_props_sweep_results(results, props, top_k=10, safe_target=safe_target)
    else:
        _print_props_results(results, props)

    if safe_mode:
        _print_safe_mode_results(results, props, cushion_sweep=cushion_sweep)

    if quantile_mode:
        _print_quantile_results(results, props)

    if calibrate:
        if sweep:
            # Combined sweep: rank (projection_variant × calibration_method × k)
            _print_combined_sweep_results(results, props,
                                          k_values=(0, 5, 15, 30, 60),
                                          top_n=15)
        else:
            _print_calibration_results(results, props)
            if calibrate == "sweep":
                _print_calibration_k_sweep(results, props,
                                           k_values=(0, 5, 10, 15, 20, 30, 60))

    # Return the full per-variant results dict so callers like
    # refit_calibration.py can fit persistent calibration files.
    return results


# ────────────────────────────────────────────────────────────────
#  Residual-calibration analysis (per (variant, prop))
# ────────────────────────────────────────────────────────────────

def _empirical_cdf(sorted_vals, x):
    """Return F(x) from a sorted list of samples (right-continuous step CDF)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.5
    # Binary search for rightmost index <= x
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / n


def _brier(probs, outcomes):
    if not probs:
        return None
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _logloss(probs, outcomes, eps=1e-6):
    if not probs:
        return None
    s = 0.0
    for p, o in zip(probs, outcomes):
        pp = max(eps, min(1 - eps, p))
        s += -(o * math.log(pp) + (1 - o) * math.log(1 - pp))
    return s / len(probs)


def _hit_rate(probs, outcomes):
    if not probs:
        return None
    hits = sum(1 for p, o in zip(probs, outcomes)
               if (1 if p >= 0.5 else 0) == o)
    return hits / len(probs) * 100.0


def _calibration_buckets(probs, outcomes, n_buckets=10):
    """Reliability-diagram buckets: [(bucket_mid, predicted_avg, actual_avg, n), ...]."""
    if not probs:
        return []
    buckets = [[] for _ in range(n_buckets)]
    for p, o in zip(probs, outcomes):
        idx = min(int(p * n_buckets), n_buckets - 1)
        buckets[idx].append((p, o))
    out = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        ps = [b[0] for b in bucket]
        os_ = [b[1] for b in bucket]
        mid = (i + 0.5) / n_buckets
        out.append((mid, sum(ps) / len(ps), sum(os_) / len(os_), len(bucket)))
    return out


def _per_player_stats(player_residuals):
    """Return {player: (n, mu, sigma)} from {player: [residuals]}."""
    out = {}
    for p, rs in player_residuals.items():
        n = len(rs)
        if n == 0:
            continue
        mu = sum(rs) / n
        var = sum((r - mu) ** 2 for r in rs) / n
        sigma = math.sqrt(var) if var > 0 else 1e-6
        out[p] = (n, mu, sigma)
    return out


def _shrunk(player_stat, pool_stat, n_player, k):
    """James-Stein style shrinkage: blend per-player stat toward pool."""
    lam = n_player / (n_player + k) if (n_player + k) > 0 else 0.0
    return lam * player_stat + (1.0 - lam) * pool_stat


def _print_calibration_results(results, props, shrinkage_k=15):
    """
    Compare five probabilistic forecasters for P(actual > synthetic_line):
      A) empirical    — weighted fraction of prior games > line (current method)
      B) resid_normal — bias-correct + Φ((projected+μ_r − L)/σ_r)  POOLED
      C) resid_ecdf   — bias-correct + 1 − F_r(L − (projected+μ_r))  POOLED
      B*) per-player Gaussian with shrinkage toward pool  (k=shrinkage_k)
      C*) per-player ECDF, mixture-blended with pool       (λ from k)
    """
    print()
    print("=" * 110)
    print(f"  RESIDUAL CALIBRATION (per-player + shrinkage k={shrinkage_k})")
    print("=" * 110)
    print("  Task: predict P(actual > synthetic_line) for each held-out observation.")
    print("  A=empirical   B/C=pooled residual   B*/C*=per-player residual w/ shrinkage")
    print()

    header = (f"{'Variant':<22} {'Prop':<5} {'N':>5}  "
              f"{'BrA':>6} {'BrB':>6} {'BrB*':>6} {'BrC':>6} {'BrC*':>6}  "
              f"{'HitA':>5} {'HitB':>5} {'HitB*':>5} {'HitC':>5} {'HitC*':>5}")
    print(header)
    print("-" * len(header))

    for vname, by_prop in results.items():
        for prop_key in props:
            cell = by_prop[prop_key]
            obs = cell.get("calib_obs") or []
            if len(obs) < 20:
                continue

            # ── Pool stats ──
            all_resid = [actual - proj for _, proj, _, actual, _, *_ in obs]
            n_pool = len(all_resid)
            mu_pool = sum(all_resid) / n_pool
            var_pool = sum((r - mu_pool) ** 2 for r in all_resid) / n_pool
            sigma_pool = math.sqrt(var_pool) if var_pool > 0 else 1e-6
            sorted_pool = sorted(all_resid)

            # ── Per-player stats ──
            player_resid = {}
            for player, proj, _, actual, _, *_ in obs:
                player_resid.setdefault(player, []).append(actual - proj)
            player_stats = _per_player_stats(player_resid)
            player_sorted_resid = {p: sorted(rs) for p, rs in player_resid.items()}

            pA, pB, pBs, pC, pCs, outcomes = [], [], [], [], [], []
            for player, proj, line, actual, emp, *_ in obs:
                if actual == line:
                    continue
                o = 1 if actual > line else 0
                outcomes.append(o)
                pA.append(max(0.0, min(1.0, emp)))

                # B (pooled Gaussian)
                corrected_pool = proj + mu_pool
                z = (corrected_pool - line) / sigma_pool if sigma_pool > 0 else 0.0
                pB.append(_norm_cdf(z))

                # C (pooled ECDF)
                f_le = _empirical_cdf(sorted_pool, line - corrected_pool)
                pC.append(1.0 - f_le)

                # B* (per-player Gaussian with shrinkage)
                if player in player_stats:
                    n_p, mu_p, sigma_p = player_stats[player]
                    mu_s = _shrunk(mu_p, mu_pool, n_p, shrinkage_k)
                    sigma_s = _shrunk(sigma_p, sigma_pool, n_p, shrinkage_k)
                else:
                    mu_s, sigma_s = mu_pool, sigma_pool
                corrected_s = proj + mu_s
                z_s = (corrected_s - line) / sigma_s if sigma_s > 0 else 0.0
                pBs.append(_norm_cdf(z_s))

                # C* (per-player ECDF blended with pool by λ)
                if player in player_sorted_resid:
                    n_p = len(player_sorted_resid[player])
                    lam = n_p / (n_p + shrinkage_k)
                    f_player = _empirical_cdf(player_sorted_resid[player],
                                              line - corrected_s)
                    f_pool   = _empirical_cdf(sorted_pool, line - corrected_s)
                    f_blend  = lam * f_player + (1 - lam) * f_pool
                else:
                    f_blend = _empirical_cdf(sorted_pool, line - corrected_s)
                pCs.append(1.0 - f_blend)

            if not outcomes:
                continue

            short = PROP_LABELS_SHORT.get(prop_key, prop_key)
            print(
                f"{vname:<22} {short:<5} {len(outcomes):>5}  "
                f"{_brier(pA, outcomes):>6.4f} {_brier(pB, outcomes):>6.4f} "
                f"{_brier(pBs, outcomes):>6.4f} {_brier(pC, outcomes):>6.4f} "
                f"{_brier(pCs, outcomes):>6.4f}  "
                f"{_hit_rate(pA, outcomes):>4.1f}% {_hit_rate(pB, outcomes):>4.1f}% "
                f"{_hit_rate(pBs, outcomes):>4.1f}% {_hit_rate(pC, outcomes):>4.1f}% "
                f"{_hit_rate(pCs, outcomes):>4.1f}%"
            )

    print()
    print("Reading guide:")
    print("  A = empirical (weighted prior-game fraction > line); B/C use POOLED residual stats.")
    print(f"  B*/C* compute per-player (μ_r, σ_r) and shrink toward pool by λ=n_player/(n_player+{shrinkage_k}).")
    print("  Brier lower=better.  Hit% higher=better.")
    print("  If B*/C* beats B/C, per-player calibration is worth implementing per (sport, prop).")


def _score_calibration_at_k(obs, k):
    """
    Compute (Brier_B*, Brier_C*, Hit_B*, Hit_C*) at a given shrinkage k.
    Pool stats computed from all `obs`; per-player stats computed per name.
    No train/test split — diagnostic only.
    """
    all_resid = [actual - proj for _, proj, _, actual, _, *_ in obs]
    mu_pool = sum(all_resid) / len(all_resid)
    var_pool = sum((r - mu_pool) ** 2 for r in all_resid) / len(all_resid)
    sigma_pool = math.sqrt(var_pool) if var_pool > 0 else 1e-6
    sorted_pool = sorted(all_resid)

    player_resid = {}
    for player, proj, _, actual, _, *_ in obs:
        player_resid.setdefault(player, []).append(actual - proj)
    player_stats = _per_player_stats(player_resid)
    player_sorted = {p: sorted(rs) for p, rs in player_resid.items()}

    pBs, pCs, outcomes = [], [], []
    for player, proj, line, actual, _, *_ in obs:
        if actual == line:
            continue
        outcomes.append(1 if actual > line else 0)
        if player in player_stats:
            n_p, mu_p, sigma_p = player_stats[player]
            mu_s = _shrunk(mu_p, mu_pool, n_p, k)
            sigma_s = _shrunk(sigma_p, sigma_pool, n_p, k)
        else:
            mu_s, sigma_s = mu_pool, sigma_pool
        corrected = proj + mu_s
        z = (corrected - line) / sigma_s if sigma_s > 0 else 0.0
        pBs.append(_norm_cdf(z))
        if player in player_sorted:
            n_p = len(player_sorted[player])
            lam = n_p / (n_p + k) if (n_p + k) > 0 else 1.0
            f_pl = _empirical_cdf(player_sorted[player], line - corrected)
            f_po = _empirical_cdf(sorted_pool, line - corrected)
            f_blend = lam * f_pl + (1 - lam) * f_po
        else:
            f_blend = _empirical_cdf(sorted_pool, line - corrected)
        pCs.append(1.0 - f_blend)

    return (_brier(pBs, outcomes), _brier(pCs, outcomes),
            _hit_rate(pBs, outcomes), _hit_rate(pCs, outcomes))


def _print_calibration_k_sweep(results, props, k_values=(0, 5, 10, 15, 20, 30, 60)):
    """
    Sweep shrinkage k over a small grid and report Brier_B*/C* + Hit_B*/C* per
    (variant, prop, k). k=0 → fully per-player (no shrinkage to pool);
    very large k → fully pooled. Helps identify the optimal shrinkage strength.
    """
    print()
    print("=" * 100)
    print("  SHRINKAGE-K SWEEP (per-player residual calibration)")
    print("=" * 100)
    print("  k=0 → fully per-player.   k→∞ → fully pooled.   k ≈ n_player_avg/2 is typical sweet spot.")
    print()

    header = (f"{'Variant':<22} {'Prop':<5} {'k':>4}  "
              f"{'BrierB*':>8} {'BrierC*':>8}  {'HitB*':>6} {'HitC*':>6}")
    print(header)
    print("-" * len(header))

    for vname, by_prop in results.items():
        for prop_key in props:
            obs = by_prop[prop_key].get("calib_obs") or []
            if len(obs) < 20:
                continue
            short = PROP_LABELS_SHORT.get(prop_key, prop_key)
            best_brC, best_k = float("inf"), None
            for k in k_values:
                brBs, brCs, hBs, hCs = _score_calibration_at_k(obs, k)
                star = ""
                if brCs is not None and brCs < best_brC:
                    best_brC, best_k = brCs, k
                print(f"{vname:<22} {short:<5} {k:>4}  "
                      f"{brBs:>8.4f} {brCs:>8.4f}  "
                      f"{hBs:>5.2f}% {hCs:>5.2f}%")
            if best_k is not None:
                print(f"{'':<22} {'':<5} {'best k for C*: ' + str(best_k):>30}")
            print()


def _evaluate_calibration_methods(obs, k_values, holdout=False):
    """
    Evaluate calibration methods for one (variant, prop) observation list.
    Returns a list of dicts: {method, k, brier, hit}.
    Methods: A (empirical), B (pooled Gaussian), C (pooled ECDF),
             B*@k, C*@k (per-player with shrinkage) for each k.

    holdout=False: fit and score on the same `obs` (diagnostic / in-sample).
    holdout=True:  sort by game_date, fit on earliest 50%, score on latest 50%.
    """
    if len(obs) < 40 if holdout else 20:
        return []

    # Determine fit vs score sets
    if holdout:
        sorted_obs = sorted(obs, key=lambda o: o[5] if len(o) > 5 else "")
        split = len(sorted_obs) // 2
        fit_obs = sorted_obs[:split]
        score_obs = sorted_obs[split:]
        if len(fit_obs) < 20 or len(score_obs) < 20:
            return []
    else:
        fit_obs = obs
        score_obs = obs

    # Pool stats from fit_obs
    all_resid = [actual - proj for _, proj, _, actual, _, *_ in fit_obs]
    mu_pool = sum(all_resid) / len(all_resid)
    var_pool = sum((r - mu_pool) ** 2 for r in all_resid) / len(all_resid)
    sigma_pool = math.sqrt(var_pool) if var_pool > 0 else 1e-6
    sorted_pool = sorted(all_resid)

    # Per-player stats from fit_obs
    player_resid = {}
    for player, proj, _, actual, _, *_ in fit_obs:
        player_resid.setdefault(player, []).append(actual - proj)
    player_stats = _per_player_stats(player_resid)
    player_sorted = {p: sorted(rs) for p, rs in player_resid.items()}

    # Score rows from score_obs
    rows = [(player, proj, line, actual, emp)
            for player, proj, line, actual, emp, *_ in score_obs
            if actual != line]
    if not rows:
        return []
    outcomes = [1 if actual > line else 0 for _, _, line, actual, _ in rows]

    results = []

    # A: empirical
    pA = [max(0.0, min(1.0, emp)) for _, _, _, _, emp in rows]
    results.append({"method": "A", "k": None,
                    "brier": _brier(pA, outcomes), "hit": _hit_rate(pA, outcomes)})

    # B: pooled Gaussian
    pB = []
    for _, proj, line, _, _ in rows:
        corrected = proj + mu_pool
        z = (corrected - line) / sigma_pool if sigma_pool > 0 else 0.0
        pB.append(_norm_cdf(z))
    results.append({"method": "B", "k": None,
                    "brier": _brier(pB, outcomes), "hit": _hit_rate(pB, outcomes)})

    # C: pooled ECDF
    pC = []
    for _, proj, line, _, _ in rows:
        corrected = proj + mu_pool
        pC.append(1.0 - _empirical_cdf(sorted_pool, line - corrected))
    results.append({"method": "C", "k": None,
                    "brier": _brier(pC, outcomes), "hit": _hit_rate(pC, outcomes)})

    # B*, C* at each k
    for k in k_values:
        pBs, pCs = [], []
        for player, proj, line, _, _ in rows:
            if player in player_stats:
                n_p, mu_p, sigma_p = player_stats[player]
                mu_s = _shrunk(mu_p, mu_pool, n_p, k)
                sigma_s = _shrunk(sigma_p, sigma_pool, n_p, k)
            else:
                mu_s, sigma_s = mu_pool, sigma_pool
            corrected = proj + mu_s
            z = (corrected - line) / sigma_s if sigma_s > 0 else 0.0
            pBs.append(_norm_cdf(z))
            if player in player_sorted:
                n_p = len(player_sorted[player])
                lam = n_p / (n_p + k) if (n_p + k) > 0 else 1.0
                f_pl = _empirical_cdf(player_sorted[player], line - corrected)
                f_po = _empirical_cdf(sorted_pool, line - corrected)
                f_blend = lam * f_pl + (1 - lam) * f_po
            else:
                f_blend = _empirical_cdf(sorted_pool, line - corrected)
            pCs.append(1.0 - f_blend)
        results.append({"method": "B*", "k": k,
                        "brier": _brier(pBs, outcomes), "hit": _hit_rate(pBs, outcomes)})
        results.append({"method": "C*", "k": k,
                        "brier": _brier(pCs, outcomes), "hit": _hit_rate(pCs, outcomes)})

    return results


def _print_combined_sweep_results(results, props, k_values, top_n=15, holdout=True):
    """
    Combined sweep ranking: for each prop, rank (projection_variant × calibration_method × k)
    by Brier (and a separate ranking by Hit %).

    holdout=True (default) fits calibration on chronologically earliest 50% of
    observations per (variant, prop) and scores on the latest 50% — honest
    out-of-sample numbers.
    """
    print()
    print("=" * 110)
    label = "out-of-sample HOLDOUT" if holdout else "IN-SAMPLE"
    print(f"  COMBINED SWEEP ({label}): best (projection × calibration method × shrinkage k) per prop")
    print("=" * 110)
    print()

    # Collect all rows
    by_prop_rows = {prop: [] for prop in props}
    for vname, by_prop in results.items():
        for prop_key in props:
            obs = by_prop[prop_key].get("calib_obs") or []
            evals = _evaluate_calibration_methods(obs, k_values, holdout=holdout)
            for e in evals:
                if e["brier"] is None:
                    continue
                by_prop_rows[prop_key].append({
                    "variant": vname,
                    "method": e["method"],
                    "k": e["k"],
                    "brier": e["brier"],
                    "hit": e["hit"],
                    "n_obs": len(obs),
                })

    for prop_key in props:
        rows = by_prop_rows[prop_key]
        if not rows:
            continue
        short = PROP_LABELS_SHORT.get(prop_key, prop_key)
        header = (f"{'Variant':<26} {'Method':>7} {'k':>4} {'N':>5}  "
                  f"{'Brier':>7}  {'Hit %':>7}")

        print(f"── {short} — TOP {top_n} by Brier (lower = better) " +
              "─" * max(0, 60 - len(short)))
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda x: x["brier"])[:top_n]:
            k_str = "—" if r["k"] is None else str(r["k"])
            hit_str = f"{r['hit']:6.2f}%" if r["hit"] is not None else "    n/a"
            print(f"{r['variant']:<26} {r['method']:>7} {k_str:>4} {r['n_obs']:>5}  "
                  f"{r['brier']:>7.4f}  {hit_str}")

        print()
        print(f"── {short} — TOP {top_n} by Hit % (higher = better) " +
              "─" * max(0, 60 - len(short)))
        print(header)
        print("-" * len(header))
        hit_rows = [r for r in rows if r["hit"] is not None]
        for r in sorted(hit_rows, key=lambda x: -x["hit"])[:top_n]:
            k_str = "—" if r["k"] is None else str(r["k"])
            print(f"{r['variant']:<26} {r['method']:>7} {k_str:>4} {r['n_obs']:>5}  "
                  f"{r['brier']:>7.4f}  {r['hit']:6.2f}%")
        print()


def _cushion_for_target(cell, target):
    """
    Smallest tested offset (cushion) at which the historical OVER hit-rate
    first reaches `target`. Returns None when no offset achieves the target.
    """
    safe = cell.get("safe", {})
    for off in sorted(safe.keys()):
        t = safe[off]
        if t["n"] == 0:
            continue
        if t["hits"] / t["n"] >= target:
            return off
    return None


def _summarize_variant(by_prop, props, safe_target=0.80):
    """
    Aggregate per-prop errors into MAE / RMSE / Bias / HitRate AND the
    per-prop cushion-for-target plus an aggregate "total cushion" score.
    """
    all_errs = []
    total_hits = 0
    total_decisive = 0
    cushion_per_prop = {}
    cushion_total = 0.0
    cushion_resolved = 0
    cushion_missing = 0
    for prop_key in props:
        cell = by_prop[prop_key]
        all_errs.extend(cell["errors"])
        total_hits += cell["hits"]
        total_decisive += cell["decisive"]

        cu = _cushion_for_target(cell, safe_target)
        cushion_per_prop[prop_key] = cu
        if cu is not None:
            cushion_total += cu
            cushion_resolved += 1
        elif cell.get("safe"):
            cushion_missing += 1

    if not all_errs:
        return None
    n = len(all_errs)
    mae = sum(abs(e) for e in all_errs) / n
    rmse = math.sqrt(sum(e * e for e in all_errs) / n)
    bias = sum(all_errs) / n
    hit_rate = (total_hits / total_decisive * 100) if total_decisive else None
    return {
        "n": n, "mae": mae, "rmse": rmse, "bias": bias,
        "hits": total_hits, "decisive": total_decisive, "hit_rate": hit_rate,
        "cushion_per_prop": cushion_per_prop,
        "cushion_total": cushion_total,
        "cushion_resolved": cushion_resolved,
        "cushion_missing": cushion_missing,
        "safe_target": safe_target,
    }


def _print_props_results(results, props):
    """Print MAE/RMSE/Bias/HitRate per variant per prop, plus combined."""
    print()
    header = (f"{'Variant':<30} {'Prop':<5} {'N':>6}  {'MAE':>8}  {'RMSE':>8}  "
              f"{'Bias':>8}  {'Hit %':>8}")
    print(header)
    print("-" * len(header))
    for vname, by_prop in results.items():
        for prop_key in props:
            cell = by_prop[prop_key]
            errs, n = cell["errors"], cell["n"]
            if n == 0:
                continue
            mae = sum(abs(e) for e in errs) / n
            rmse = math.sqrt(sum(e * e for e in errs) / n)
            bias = sum(errs) / n
            hit = (cell["hits"] / cell["decisive"] * 100) if cell["decisive"] else float("nan")
            short = PROP_LABELS_SHORT.get(prop_key, prop_key)
            hit_str = f"{hit:>7.2f}%" if cell["decisive"] else "    n/a"
            print(f"{vname:<30} {short:<5} {n:>6}  {mae:>8.3f}  {rmse:>8.3f}  {bias:>+8.3f}  {hit_str}")
        summary = _summarize_variant(by_prop, props)
        if summary:
            hit_str = f"{summary['hit_rate']:>7.2f}%" if summary["hit_rate"] is not None else "    n/a"
            print(f"{vname:<30} {'ALL':<5} {summary['n']:>6}  {summary['mae']:>8.3f}  "
                  f"{summary['rmse']:>8.3f}  {summary['bias']:>+8.3f}  {hit_str}")
        print("-" * len(header))
    print("\nLegend:")
    print("  MAE:   mean absolute error of projected stat vs actual (lower = better)")
    print("  RMSE:  root-mean-square error (lower = better; penalizes large misses)")
    print("  Bias:  mean signed error; +ve = projection overshoots, -ve = undershoots")
    print("  Hit %: directional hit-rate vs synthetic line = unweighted avg of prior games")
    print("         (>50% means variant's projection correctly distinguishes O/U vs season avg)")


def _print_quantile_results(results, props):
    """
    Per-player weighted-quantile alt-line analysis.

    For each variant × prop × q-threshold, the alt-line is the q-quantile
    of that player's own weighted prior-game distribution. The cushion
    (projection − alt_line) auto-scales to the player's variance, so
    low-volume / steady players need MUCH less cushion than high-variance
    stars at the same nominal confidence level.

    Reports per-prop:
      - target confidence (1 - q)
      - actual achieved hit-rate
      - mean / median / max cushion across all (player, game) observations
    """
    print()
    print("=" * 100)
    print("  PER-PLAYER QUANTILE ALT-LINE analysis")
    print("=" * 100)
    print("  Strategy: alt_line = q-quantile of player's own weighted prior-game distribution.")
    print("  Cushion scales naturally with each player's variance — no flat scalar needed.")
    print()

    for vname, by_prop in results.items():
        print(f"── Variant: {vname} " + "─" * max(0, 80 - len(vname)))
        for prop_key in props:
            cell = by_prop[prop_key]
            qmap = cell.get("quantile") or {}
            if not qmap:
                continue
            short = PROP_LABELS_SHORT.get(prop_key, prop_key)
            print(f"  {short}:")
            header = (f"    {'target':>8}  {'q':>5}  {'n':>5}  {'hit%':>7}  "
                      f"{'mean_cu':>9}  {'median_cu':>10}  {'max_cu':>8}")
            print(header)
            print("    " + "-" * (len(header) - 4))
            for q in sorted(qmap.keys(), reverse=True):
                t = qmap[q]
                if t["n"] == 0:
                    continue
                target = (1.0 - q) * 100.0
                hit = t["hits"] / t["n"] * 100.0
                cu = t["cushions"]
                mean_cu = sum(cu) / len(cu)
                sorted_cu = sorted(cu)
                med_cu = sorted_cu[len(sorted_cu) // 2]
                max_cu = max(cu)
                tag = "min" if q == 0.0 else f"{target:.0f}%"
                print(f"    {tag:>8}  {q:>5.2f}  {t['n']:>5}  {hit:>6.2f}%  "
                      f"{mean_cu:>9.2f}  {med_cu:>10.2f}  {max_cu:>8.2f}")
        print()

    print("Reading guide:")
    print("  - 'target' = the % of the time the player historically clears the alt line")
    print("    based on their weighted recent games (1 − q).")
    print("  - 'hit %' = actual achieved hit-rate on held-out test games. Should track target.")
    print("  - mean/median/max cushion show how much our projection sits ABOVE the alt-line.")
    print("    Small cushion = tighter to projection (closer to book lines), bigger payout.")


def _print_safe_mode_results(results, props, cushion_sweep=False):
    """
    Safe-mode (OVER alt-line) hit-rate analysis. For each variant × prop,
    print the hit-rate at each tested offset (alt_line = projected − offset).
    Higher hit-rate = "safer" alt-line bet. Also reports the smallest offset
    needed to hit each target hit-rate (cushion sweet-spot).

    In cushion_sweep mode the offsets are finer-grained, so we show the
    thresholds prominently and the full curve in a compact wrapped layout.
    """
    targets = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    title = ("CUSHION SWEEP — fine-grained OVER alt-line analysis"
             if cushion_sweep
             else "SAFE-MODE (OVER alt-line) hit-rate analysis")

    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    print("  Strategy: always bet OVER on (projection − offset).")
    print("  Hit means actual > alt_line. Higher offset = safer bet but worse payout.")
    print()

    for vname, by_prop in results.items():
        print(f"── Variant: {vname} " + "─" * (80 - len(vname)))
        for prop_key in props:
            cell = by_prop[prop_key]
            # Pull offsets from cell["safe"] in sorted order so it works for
            # both coarse and fine grids.
            offsets = sorted(cell["safe"].keys())
            if not offsets or cell["n"] == 0:
                continue
            short = PROP_LABELS_SHORT.get(prop_key, prop_key)

            curve = []
            for off in offsets:
                t = cell["safe"][off]
                rate = (t["hits"] / t["n"] * 100) if t["n"] else 0.0
                curve.append((off, t["n"], rate))

            print(f"  {short}:")

            # Print thresholds first (the most actionable info)
            ach_lines = []
            for tgt in targets:
                hit_off = next((o for o, _, r in curve if r / 100 >= tgt), None)
                if hit_off is not None:
                    ach_lines.append(f"{int(tgt*100)}%@{hit_off}")
                else:
                    ach_lines.append(f"{int(tgt*100)}%:—")
            print(f"     Thresholds:  " + "  ".join(ach_lines))

            # Print the full curve, wrapped to ~8 entries per row for fine grids
            chunk = 8
            for i in range(0, len(curve), chunk):
                seg = curve[i:i + chunk]
                hdr = "     " + "  ".join(f"off={o:<5}" for o, _, _ in seg)
                row = "     " + "  ".join(f"{r:>5.1f}%   " for _, _, r in seg)
                print(hdr)
                print(row)
            print()
    print("Reading guide:")
    print("  - off=0 row: hit-rate when betting OVER on the raw projection (50% = perfect calibration).")
    print("  - Each subsequent offset gives more cushion → higher hit-rate, lower payout.")
    print("  - In production, pair this curve with the book's alt-line odds:")
    print("       cushion = (projected − book_alt_line)")
    print("       expected hit-rate ≈ this table's value at the nearest offset")
    print("       bet if: (cushion ≥ X) AND (book's implied prob < this hit-rate)")


def _parse_variant_label(vname):
    """
    Parse variant names like 'hl10/defadj1.0/ven0.25' into a dict of knob values.
    Recognizes prefixes: hl, defadj, def, ven, pace.
    """
    parts = {}
    # Order matters — match longer prefixes first ("defadj" before "def")
    prefixes = ("defadj", "pace", "hl", "def", "ven")
    for tok in vname.split("/"):
        for prefix in prefixes:
            if tok.startswith(prefix):
                parts[prefix] = tok[len(prefix):]
                break
    return parts


def _print_props_sweep_results(results, props, top_k=10, safe_target=0.80):
    """For sweep mode: show top-K variants + marginal-effect tables per knob.

    Adds a ranking by the smallest cushion needed for each prop to reach
    `safe_target` historical OVER hit-rate (so we can pick parameters that
    minimize how much we have to shade our prediction to bet safely).
    """
    rows = []
    for vname, by_prop in results.items():
        summary = _summarize_variant(by_prop, props, safe_target=safe_target)
        if summary:
            rows.append((vname, summary))

    if not rows:
        print("No results to display.")
        return

    header = (f"{'Variant':<30} {'N':>6}  {'MAE':>8}  {'RMSE':>8}  "
              f"{'Bias':>8}  {'Hit %':>8}")

    # ── Top-K by MAE ──
    print()
    print("=" * len(header))
    print(f"  TOP {top_k} by Combined MAE (lower = better projection accuracy)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for vname, s in sorted(rows, key=lambda r: r[1]["mae"])[:top_k]:
        hit_str = f"{s['hit_rate']:>7.2f}%" if s["hit_rate"] is not None else "    n/a"
        print(f"{vname:<30} {s['n']:>6}  {s['mae']:>8.3f}  {s['rmse']:>8.3f}  "
              f"{s['bias']:>+8.3f}  {hit_str}")

    # ── Top-K by Hit % ──
    print()
    print("=" * len(header))
    print(f"  TOP {top_k} by Hit % vs synthetic line (higher = better O/U direction)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    hit_rows = [r for r in rows if r[1]["hit_rate"] is not None]
    for vname, s in sorted(hit_rows, key=lambda r: -r[1]["hit_rate"])[:top_k]:
        hit_str = f"{s['hit_rate']:>7.2f}%"
        print(f"{vname:<30} {s['n']:>6}  {s['mae']:>8.3f}  {s['rmse']:>8.3f}  "
              f"{s['bias']:>+8.3f}  {hit_str}")

    # ── Top-K by Safe-Target Cushion ──
    # Rank by the SUM of cushions required (across PTS/REB/AST) to hit
    # `safe_target`. Lower = our model needs less shading to safely bet OVER.
    safe_rows = [r for r in rows if r[1]["cushion_resolved"] == len(props)]
    cushion_header = (
        f"{'Variant':<30} {'PTS_cu':>7}  {'REB_cu':>7}  {'AST_cu':>7}  "
        f"{'TotalCu':>8}  {'MAE':>8}  {'Hit %':>8}"
    )
    print()
    print("=" * len(cushion_header))
    print(f"  TOP {top_k} by Safe-Mode Cushion (target {safe_target*100:.0f}% hit-rate, "
          f"lower total cushion = better)")
    print("=" * len(cushion_header))
    print(cushion_header)
    print("-" * len(cushion_header))
    if not safe_rows:
        print("  (no variants resolved cushion-for-target on all props; "
              "consider lowering --safe-target)")
    else:
        for vname, s in sorted(safe_rows, key=lambda r: r[1]["cushion_total"])[:top_k]:
            cp = s["cushion_per_prop"]
            def _fmt_cu(p):
                v = cp.get(p)
                return f"{v:>7.2f}" if v is not None else "    n/a"
            hit_str = f"{s['hit_rate']:>7.2f}%" if s["hit_rate"] is not None else "    n/a"
            print(f"{vname:<30} {_fmt_cu('player_points')}  {_fmt_cu('player_rebounds')}  "
                  f"{_fmt_cu('player_assists')}  {s['cushion_total']:>8.2f}  "
                  f"{s['mae']:>8.3f}  {hit_str}")

    # ── Per-prop cushion ranking (since units differ across props) ──
    for prop_key, label in [("player_points", "PTS"),
                            ("player_rebounds", "REB"),
                            ("player_assists", "AST")]:
        if prop_key not in props:
            continue
        ranked = [(v, s["cushion_per_prop"].get(prop_key), s)
                  for v, s in rows if s["cushion_per_prop"].get(prop_key) is not None]
        if not ranked:
            continue
        ranked.sort(key=lambda r: r[1])
        ph = f"{'Variant':<30} {label+'_cushion':>14}  {label+'_MAE':>10}"
        print()
        print("=" * len(ph))
        print(f"  TOP {top_k} {label} variants by Cushion@{safe_target*100:.0f}%")
        print("=" * len(ph))
        print(ph)
        print("-" * len(ph))
        for vname, cu, s in ranked[:top_k]:
            # per-prop MAE
            errs = []
            # results[vname] is by_prop dict
            cell_errs = results[vname][prop_key]["errors"]
            errs = cell_errs
            pmae = (sum(abs(e) for e in errs) / len(errs)) if errs else float("nan")
            print(f"{vname:<30} {cu:>14.2f}  {pmae:>10.3f}")

    # ── Marginal-effect tables (isolate each parameter) ──
    parsed = [(vname, _parse_variant_label(vname), s) for vname, s in rows]

    knob_label = {
        "hl": "Half-life",
        "def": "Defense weight strength",
        "defadj": "Defense output adj. strength",
        "pace": "Pace adj. strength",
        "ven": "Venue strength",
    }
    # Only display knobs that actually appear in the parsed variants
    knobs_present = {k for _, parts, _ in parsed for k in parts}
    for knob in ("hl", "defadj", "def", "pace", "ven"):
        if knob not in knobs_present:
            continue
        # Group rows by value of this knob (averaging across all other knobs)
        groups = {}
        for vname, parts, s in parsed:
            v = parts.get(knob)
            if v is None:
                continue
            groups.setdefault(v, []).append(s)

        if not groups:
            continue

        print()
        print("=" * 70)
        print(f"  MARGINAL EFFECT: {knob_label[knob]} "
              f"(averaged across all other parameter settings)")
        print("=" * 70)
        mhead = f"{knob:<6} {'n_combos':>9} {'avg MAE':>10} {'avg RMSE':>10} {'avg Bias':>10} {'avg Hit %':>11}"
        print(mhead)
        print("-" * len(mhead))

        # Sort by parameter value (numerically when possible)
        def _key(v):
            if v == "none":
                return -1.0
            try:
                return float(v)
            except ValueError:
                return 0.0

        for v in sorted(groups.keys(), key=_key):
            entries = groups[v]
            n = len(entries)
            avg_mae = sum(e["mae"] for e in entries) / n
            avg_rmse = sum(e["rmse"] for e in entries) / n
            avg_bias = sum(e["bias"] for e in entries) / n
            hit_vals = [e["hit_rate"] for e in entries if e["hit_rate"] is not None]
            avg_hit = (sum(hit_vals) / len(hit_vals)) if hit_vals else None
            hit_str = f"{avg_hit:>10.2f}%" if avg_hit is not None else "       n/a"
            print(f"{v:<6} {n:>9} {avg_mae:>10.4f} {avg_rmse:>10.4f} {avg_bias:>+10.4f}  {hit_str}")

    # ── Marginal-effect tables, ranked by Safe-Target Cushion ──
    for knob in ("hl", "defadj", "def", "pace", "ven"):
        if knob not in knobs_present:
            continue
        groups = {}
        for vname, parts, s in parsed:
            v = parts.get(knob)
            if v is None:
                continue
            groups.setdefault(v, []).append(s)
        if not groups:
            continue

        print()
        print("=" * 78)
        print(f"  MARGINAL EFFECT (SAFE MODE): {knob_label[knob]} "
              f"— avg cushion needed @ {safe_target*100:.0f}%")
        print("=" * 78)
        mhead = (f"{knob:<6} {'n_combos':>9} {'avg PTS_cu':>11} "
                 f"{'avg REB_cu':>11} {'avg AST_cu':>11} {'avg TotalCu':>12}")
        print(mhead)
        print("-" * len(mhead))

        def _key(v):
            if v == "none":
                return -1.0
            try:
                return float(v)
            except ValueError:
                return 0.0

        for v in sorted(groups.keys(), key=_key):
            entries = groups[v]
            n = len(entries)
            def _avg_per(prop_key):
                vals = [e["cushion_per_prop"].get(prop_key) for e in entries]
                vals = [x for x in vals if x is not None]
                return (sum(vals) / len(vals)) if vals else None
            pts_a = _avg_per("player_points")
            reb_a = _avg_per("player_rebounds")
            ast_a = _avg_per("player_assists")
            tot_vals = [e["cushion_total"] for e in entries
                        if e["cushion_resolved"] == len(props)]
            tot_a = (sum(tot_vals) / len(tot_vals)) if tot_vals else None
            def _f(x):
                return f"{x:>11.3f}" if x is not None else "        n/a"
            tot_s = f"{tot_a:>12.3f}" if tot_a is not None else "         n/a"
            print(f"{v:<6} {n:>9} {_f(pts_a)} {_f(reb_a)} {_f(ast_a)} {tot_s}")

    print()
    print("How to read the marginal-effect tables:")
    print("  - Each row holds one parameter fixed at the labeled value and")
    print("    averages MAE/Hit%/cushion across every combination of the OTHER parameters.")
    print("  - Lower avg cushion = our prediction needs less shading to hit the")
    print(f"    {safe_target*100:.0f}% safe-mode hit-rate target for that prop.")
    print("  - Compare the 'best by MAE' setting vs the 'best by cushion' setting:")
    print("    if they differ, safe mode may want different modifiers than the")
    print("    accuracy-optimized production model.")


def main():
    p = argparse.ArgumentParser(description="Backtest the sportsbook projection model")
    p.add_argument("--mode", choices=["matchup", "props"], default="matchup",
                   help="matchup = team-level projections; props = player-prop projections")
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
    # Player-props mode arguments
    p.add_argument("--players", default=None,
                   help="Comma-separated player names (props mode). Default: built-in starters per sport.")
    p.add_argument("--props", default=None,
                   help="Comma-separated prop keys (props mode). Default: pts/reb/ast for NBA.")
    p.add_argument("--games-per-player", type=int, default=60,
                   help="Most recent N games per player to backtest (props mode)")
    p.add_argument("--safe-mode", action="store_true",
                   help="Add safe-mode (OVER alt-line) hit-rate analysis to props output")
    p.add_argument("--cushion-sweep", action="store_true",
                   help="Fine-grained cushion sweep — finds the cushion needed "
                        "to hit each target hit-rate. Implies --safe-mode.")
    p.add_argument("--safe-target", type=float, default=0.80,
                   help="Target hit-rate (0-1) used to rank sweep variants by "
                        "the smallest cushion required to reach it. Default 0.80.")
    p.add_argument("--quantile-mode", action="store_true",
                   help="Run per-player weighted-quantile alt-line analysis "
                        "(cushion auto-scales to each player's variance).")
    p.add_argument("--calibrate", nargs="?", const=True, default=False,
                   choices=[True, "sweep"],
                   help="(props mode) Run residual-calibration analysis: "
                        "compare empirical vs residual-Normal vs residual-ECDF "
                        "probabilistic forecasters with Brier/log-loss/hit-rate. "
                        "Use --calibrate=sweep to also sweep shrinkage k.")
    p.add_argument("--cross-season", choices=["strict", "all"], default="strict",
                   help="(props mode) 'strict' (default) keeps only current-season "
                        "prior games per test observation; 'all' uses the full "
                        "ESPN gamelog regardless of season boundary.")
    args = p.parse_args()

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]

    if args.sweep:
        variants = (_build_props_sweep_grid() if args.mode == "props"
                    else _build_sweep_grid())
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

    print(f"#  Backtest: {args.sport.upper()} ({sport_key})  mode={args.mode}")
    print(f"#  Season: {args.season if args.season else 'current'}")
    if args.mode == "matchup":
        print(f"#  Sample limit: {args.limit}   Prior window: {args.window}   Min sample: {args.min_sample}")
    else:
        print(f"#  Games per player: {args.games_per_player}   Min sample: {args.min_sample}")
    if not args.sweep:
        print(f"#  Variants: {variant_names}")
    print(f"{'#'*60}")

    if args.mode == "matchup":
        run_backtest(sport_key, espn_sport, espn_league,
                     limit=args.limit, window=args.window, variants=variants,
                     min_sample=args.min_sample, season_year=args.season,
                     sweep=args.sweep, quantile_mode=args.quantile_mode,
                     safe_target=args.safe_target)
    else:
        # Player-props mode
        if args.players:
            players = [n.strip() for n in args.players.split(",") if n.strip()]
        else:
            players = DEFAULT_STARTERS.get(args.sport, [])
            if not players:
                print(f"No default starter list for sport={args.sport}. Use --players.")
                sys.exit(1)

        if args.props:
            props = [p.strip() for p in args.props.split(",") if p.strip()]
        else:
            props = DEFAULT_PROPS.get(args.sport, [])
            if not props:
                print(f"No default props for sport={args.sport}. Use --props.")
                sys.exit(1)

        print(f"#  Players ({len(players)}): {', '.join(players)}")
        print(f"#  Props: {', '.join(props)}")
        print(f"{'#'*60}")

        run_player_props_backtest(args.sport, espn_sport, espn_league, sport_key,
                                  players=players, props=props,
                                  games_per_player=args.games_per_player,
                                  min_sample=args.min_sample,
                                  variants=variants, sweep=args.sweep,
                                  season_year=args.season,
                                  safe_mode=args.safe_mode,
                                  cushion_sweep=args.cushion_sweep,
                                  safe_target=args.safe_target,
                                  quantile_mode=args.quantile_mode,
                                  calibrate=args.calibrate,
                                  cross_season=args.cross_season)


if __name__ == "__main__":
    main()
