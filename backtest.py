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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from espn_client import (
    get_all_teams,
    get_team_schedule,
    annotate_opponent_strength,
    get_athlete_gamelog,
    search_athlete,
    get_team_pace_factor,
    ip_to_outs,
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
    _weighted_std,
    _half_life_for,
    _norm_cdf,
    _park_factor_mult,
    analyze_moneyline_value,
    analyze_spreads_value,
    analyze_totals_value,
)
from odds_client import (
    american_to_decimal,
    american_to_implied_prob,
    devig_two_way,
)
from pricing_common import _resolve_team_defense
import historical_odds as hist_store
import prop_features  # §2.6 candidate-feature registry (rest/days-off, …)
from calibration_loader import (
    save_market_blend,
    save_prob_shrink,
    save_calibration,
    load_calibration,
    apply_calibration_with_warmup,
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


def cached_schedule(espn_sport, espn_league, team_id, season_year=None,
                    ttl_hours=24 * 7, current_ttl_hours=12):
    """
    Cached wrapper around get_team_schedule. Historical seasons get a long TTL
    (results don't change). The current season uses a SHORT TTL
    (``current_ttl_hours``, default 12h) so recently-completed games surface
    promptly: the team-market odds backtest grades warehoused closing lines
    against these finals, and a multi-day cache would hide the last few days of
    results — leaving freshly-captured (upcoming-then-completed) games
    ungradeable until the cache expired.
    """
    path = _cache_key("schedule", espn_sport, espn_league, team_id, season_year or "current")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        # Historical seasons (specified explicitly) are immutable — long TTL;
        # the current season refreshes within current_ttl_hours.
        effective_ttl = (ttl_hours * 24) if season_year else current_ttl_hours
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


def _warehouse_team_schedules(seasons_list):
    """MLB warehouse analog of get_all_teams + build_schedules for the team-market
    backtests (run_backtest / run_odds_backtest). Returns (teams, schedules):
      teams     = {canonical_name: {"id": mlbam_team_id}}   (get_all_teams shape)
      schedules = {mlbam_team_id: [game dicts]}             (build_schedules shape)
    Each game dict is a mlb_warehouse._team_final_games row ({date, home_team,
    away_team, home_score, away_score, total_score}) with CANONICAL StatsAPI names —
    the same shape get_team_schedule emits — pooled across seasons_list. The
    opponent-strength annotation + the odds-store join key off these structures; the
    odds join prefers SFBB team CODES (team_code_for_name), robust to canonical-vs-
    odds spelling, with the name key a fallback — so both backtests run unchanged."""
    import mlb_warehouse
    # A None season means the CURRENT season (mirroring get_team_schedule(None), the
    # ESPN default). _team_final_games applies NO filter on season=None, so leaving it
    # None would silently pool EVERY warehouse season and broaden the fit -- map it to
    # the concrete current year instead.
    cur = mlb_warehouse._current_season()
    seasons = list(dict.fromkeys(cur if sy is None else sy for sy in seasons_list))
    names = mlb_warehouse._team_name_map()             # {mlbam_tid: canonical_name}
    teams, schedules = {}, {}
    for tid, nm in names.items():
        rows = []
        for sy in seasons:
            rows.extend(mlb_warehouse._team_final_games(tid, season=sy))
        schedules[tid] = rows
        # win_pct (from the fetched games) mirrors the ESPN teams dict so
        # annotate_opponent_strength's opponent-strength weighting isn't neutralized:
        # a missing win_pct defaults to 0.5 and opp_strength_mult(0.5, s) == 1.0.
        wins = sum(1 for g in rows if (
            (g["home_team"] == nm and g["home_score"] > g["away_score"])
            or (g["away_team"] == nm and g["away_score"] > g["home_score"])))
        teams[nm] = {"id": tid, "win_pct": (wins / len(rows)) if rows else 0.5}
    return teams, schedules


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

    # P3b/P4/P6: MLB team schedules come from the StatsAPI warehouse; NBA/NFL stay on
    # ESPN. (ESPN was fully removed for MLB in P4.)
    use_warehouse = espn_sport == "baseball"

    print(f"\n=== Loading {sport_key} team list ===")
    season_label = f"season {season_year}" if season_year else "current season"
    if use_warehouse:
        print("=== team-market inputs: StatsAPI warehouse (ESPN bypassed) ===")
        espn_teams, schedules = _warehouse_team_schedules([season_year])
    else:
        espn_teams = get_all_teams(espn_sport, espn_league)
        print(f"Loaded {len(espn_teams)} teams")
        print(f"\n=== Fetching schedules for {season_label} (cached) ===")
        schedules = build_schedules(espn_sport, espn_league, espn_teams,
                                    season_year=season_year)
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


def _match_espn_name(espn_teams, api_name):
    """Map an Odds-API team name to the ESPN displayName key (exact→ci→substring)."""
    if not api_name:
        return None
    if api_name in espn_teams:
        return api_name
    low = api_name.lower()
    for name in espn_teams:
        if name.lower() == low:
            return name
    for name in espn_teams:
        if low in name.lower() or name.lower() in low:
            return name
    return None


def _build_odds_lookup(store, espn_teams):
    """
    Index a historical-odds store by (date10, espn_home, espn_away) using
    ESPN-normalized team names so it can be joined to ESPN schedule games.

    Additionally emits an id-keyed entry ("id", date10, home_code, away_code)
    whenever both canonical SFBB team codes resolve — from the codes surfaced by
    the SQL warehouse, else player_id_map.team_code_for_name. This lets the join
    prefer stable codes over the lossy exact→ci→substring name match (fixing
    franchise renames like Indians→Guardians). Fail-open: a code miss leaves only
    the name key, reproducing today's behavior.
    """
    try:
        import player_id_map
    except Exception:
        player_id_map = None
    lookup = {}
    unmatched = 0
    for entry in store.get("games", {}).values():
        date10 = (entry.get("commence_time") or "")[:10]
        if player_id_map is not None:
            try:
                hc = entry.get("home_code") or player_id_map.team_code_for_name(
                    entry.get("home_team"))
                ac = entry.get("away_code") or player_id_map.team_code_for_name(
                    entry.get("away_team"))
                if hc and ac:
                    lookup[("id", date10, hc, ac)] = entry
            except Exception:
                pass
        eh = _match_espn_name(espn_teams, entry.get("home_team"))
        ea = _match_espn_name(espn_teams, entry.get("away_team"))
        if not eh or not ea:
            unmatched += 1
            continue
        lookup[(date10, eh, ea)] = entry
    return lookup, unmatched


def _lookup_game_odds(lookup, date10, home, away):
    """Find a stored game by date (±1 day), preferring canonical SFBB team codes
    (robust to franchise renames / name-spelling drift) and falling back to the
    ESPN team-name key. Fail-open on the code resolution."""
    try:
        import player_id_map
        hc = player_id_map.team_code_for_name(home)
        ac = player_id_map.team_code_for_name(away)
    except Exception:
        hc = ac = None
    for d in (date10, _shift_date(date10, -1), _shift_date(date10, 1)):
        if hc and ac:
            hit = lookup.get(("id", d, hc, ac))
            if hit:
                return hit
        hit = lookup.get((d, home, away))
        if hit:
            return hit
    return None


def _shift_date(date10, days):
    from datetime import date as _date, timedelta
    try:
        return (_date.fromisoformat(date10) + timedelta(days=days)).isoformat()
    except (ValueError, TypeError):
        return None


_SPORT_KEY_TO_CLI = {
    "basketball_nba": "nba", "americanfootball_nfl": "nfl",
    "baseball_mlb": "mlb", "icehockey_nhl": "nhl",
}
MARKETS = ("moneyline", "spreads", "totals")


def _empty_market_bucket():
    return {"n": 0, "model_brier": [], "market_brier": [], "correct": 0,
            "bets": 0, "profit": 0.0, "blend": [], "prior_k": []}


def _grade(bucket, model_p, market_p, outcome, price_yes, price_no, threshold):
    """Update a market bucket with one observation (yes = the modelled side)."""
    bucket["n"] += 1
    bucket["model_brier"].append((model_p - outcome) ** 2)
    bucket["market_brier"].append((market_p - outcome) ** 2)
    if (1 if model_p > 0.5 else 0) == outcome:
        bucket["correct"] += 1
    bucket["blend"].append((model_p, market_p, outcome))
    if price_yes is not None and model_p - market_p >= threshold:
        bucket["bets"] += 1
        bucket["profit"] += (american_to_decimal(price_yes) - 1) if outcome else -1
    if price_no is not None and (1 - model_p) - (1 - market_p) >= threshold:
        bucket["bets"] += 1
        bucket["profit"] += (american_to_decimal(price_no) - 1) if not outcome else -1


def _moneyline_market(entry):
    """(fair_home_prob, home_price, away_price) from the stored moneyline."""
    ml = entry.get("moneyline") or {}
    h = (ml.get(entry.get("home_team")) or [None])[0]
    a = (ml.get(entry.get("away_team")) or [None])[0]
    if not h or not a:
        return None
    fair_home, _ = devig_two_way(h["implied_prob"], a["implied_prob"])
    return fair_home, h["price"], a["price"]


def _spread_market(entry):
    """(home_spread, fair_home_cover_prob, home_price, away_price) or None."""
    sp = entry.get("spreads") or {}
    h = (sp.get(entry.get("home_team")) or [None])[0]
    a = (sp.get(entry.get("away_team")) or [None])[0]
    if not h or not a:
        return None
    fair_home, _ = devig_two_way(
        american_to_implied_prob(h["price"]), american_to_implied_prob(a["price"]))
    return h["spread"], fair_home, h["price"], a["price"]


def _total_market(entry):
    """(line, fair_over_prob, over_price, under_price) or None."""
    tot = entry.get("totals") or {}
    o = (tot.get("Over") or [None])[0]
    u = (tot.get("Under") or [None])[0]
    if not o or not u:
        return None
    fair_over, _ = devig_two_way(
        american_to_implied_prob(o["price"]), american_to_implied_prob(u["price"]))
    return o["line"], fair_over, o["price"], u["price"]


def _best_blend_weight(obs, step=0.05):
    """Return (best_w, best_brier, model_brier, market_brier) minimizing Brier."""
    if not obs:
        return None
    n = len(obs)
    best_w, best_brier = 1.0, float("inf")
    w = 0.0
    while w <= 1.0001:
        brier = sum((w * pm + (1 - w) * mk - o) ** 2 for pm, mk, o in obs) / n
        if brier < best_brier:
            best_brier, best_w = brier, w
        w += step
    model_brier = sum((pm - o) ** 2 for pm, mk, o in obs) / n
    market_brier = sum((mk - o) ** 2 for pm, mk, o in obs) / n
    return round(best_w, 2), best_brier, model_brier, market_brier


def _best_market_prior_k(obs, k_values=(0, 2, 5, 10, 15, 20, 30, 50, 75, 100,
                                        150, 200),
                         threshold=0.05):
    """Sweep the market-as-prior shrinkage k (P1.1a) on prop observations.

    obs = [(model_p, market_p, outcome, n), ...] where n is the player's game
    count. For each k the model prob is shrunk toward the market prob with
    w = n/(n+k) (matching props.py) and scored by Brier. Also counts the
    threshold-clearing bets at k=0 (pure model) vs the best k, so the
    thin-sample false-positive collapse — 1.1a's acceptance criterion — is
    visible. Returns (best_k, best_brier, model_brier, bets_at_k0, bets_at_best)
    or None on empty input."""
    if not obs:
        return None
    n_obs = len(obs)

    def _blend(k):
        out = []
        for pm, mk, o, n in obs:
            w = n / (n + k) if (n + k) else 1.0
            out.append((w * pm + (1 - w) * mk, mk, o))
        return out

    def _bets(blended):
        c = 0
        for p, mk, o in blended:
            if p - mk >= threshold:
                c += 1
            if (1 - p) - (1 - mk) >= threshold:
                c += 1
        return c

    model_brier = sum((pm - o) ** 2 for pm, _, o, _ in obs) / n_obs
    bets_k0 = _bets([(pm, mk, o) for pm, mk, o, _ in obs])
    best_k, best_brier, best_bets = 0, float("inf"), bets_k0
    for k in k_values:
        blended = _blend(k)
        brier = sum((p - o) ** 2 for p, _, o in blended) / n_obs
        if brier < best_brier:
            best_brier, best_k, best_bets = brier, k, _bets(blended)
    return best_k, best_brier, model_brier, bets_k0, best_bets


def _best_shrink(obs, step=0.05):
    """Find the probability-shrink s in [0,1] minimizing Brier on the model's
    own probabilities. obs = [(model_p, market_p, outcome), ...].
    Returns (best_s, best_brier, raw_brier) or None."""
    if not obs:
        return None
    n = len(obs)
    raw_brier = sum((pm - o) ** 2 for pm, _, o in obs) / n
    best_s, best_brier = 1.0, float("inf")
    s = 0.0
    while s <= 1.0001:
        brier = sum((0.5 + s * (pm - 0.5) - o) ** 2 for pm, _, o in obs) / n
        if brier < best_brier:
            best_brier, best_s = brier, s
        s += step
    return round(best_s, 2), best_brier, raw_brier


def _load_odds_store(sport_key, store_label, source):
    """Return ``(store, source_used)`` for the team-market backtest.

    ``source``: 'auto' (default) uses the SQL odds warehouse when it's enabled,
    no --store-label was given, and it holds team-market games — else the local
    historical_odds JSON; 'warehouse' forces the warehouse; 'store' forces the
    local JSON. The warehouse store is assembled to the EXACT historical_odds
    shape so the downstream grading path is identical."""
    def _local():
        return hist_store.load_store(sport_key, store_label), "store"

    if source == "store":
        return _local()
    if source in ("auto", "warehouse"):
        try:
            import warehouse
            wh = warehouse.load_team_market_store(sport_key)
        except Exception:
            wh = None
        if source == "warehouse":
            return (wh or {"sport_key": sport_key, "games": {}}), "warehouse"
        # auto: warehouse only when it has games and no explicit local label.
        if wh and wh.get("games") and not store_label:
            return wh, "warehouse"
        return _local()
    return _local()


def _market_log_supplement(sport_key, graded_keys, date_from=None, date_to=None):
    """Extend the model-side holdout with resolved team-market forecasts the
    warehouse did NOT already grade, read from the durable market_prediction_log.

    Model-side only: a one-sided picked-side (raw_prob, outcome) row folds
    identically into the raw/shrunk Brier as its home/over-side form (the shrink
    transform and Brier are invariant under the side-flip (p,o)->(1-p,1-o)), so
    it validly pools into _best_shrink — but it can't be devigged into a two-way
    market prob, so it never touches market_brier or the blend weight.

    ``date_from``/``date_to`` (inclusive game_date bounds) scope the log rows to
    the same window the warehouse/store graded, so the pooled holdout and the
    persisted shrink describe one coherent population (a --season/--limit scoped
    run doesn't silently fold in all-time forward-log rows). When both are None
    (unscoped run) no date filter is applied.

    Returns {market: {'obs': [(raw_prob, outcome)],
                      'roi': [(is_value, price, outcome)]}} over MARKETS.
    Best-effort: empty buckets on any error (e.g. the SQL table isn't created)."""
    out = {m: {"obs": [], "roi": []} for m in MARKETS}
    try:
        import recalibration
        rows = recalibration._read_market_log(
            where={"sport_key": sport_key, "resolved": True})
    except Exception:
        return out
    bt_to_market = {"moneyline": "moneyline", "spread": "spreads",
                    "total": "totals"}
    seen = set()
    scoped = date_from is not None or date_to is not None
    for r in rows:
        try:
            if r.get("sport_key") != sport_key or not r.get("resolved"):
                continue
            if scoped:
                gd = (r.get("game_date") or "")[:10]
                if not gd:
                    continue  # can't place it in the window → out of scope
                if date_from is not None and gd < date_from:
                    continue
                if date_to is not None and gd > date_to:
                    continue
            market = bt_to_market.get(r.get("bet_type"))
            if market is None:
                continue
            outcome = r.get("outcome")
            if outcome not in (0, 1):
                continue  # push / ungraded — drop (no clean Brier target)
            ev_id = r.get("event_id")
            gkey = hist_store.game_key(
                r.get("commence_time"), r.get("home_team"), r.get("away_team"))
            toks = (gkey, ev_id) if ev_id else (gkey,)
            # Warehouse graded it (under either token) → warehouse wins (it also
            # yields market_brier); skip so the event isn't double-counted.
            if any((t, market) in graded_keys for t in toks):
                continue
            canon = ev_id or gkey            # one row per (event, market)
            if (canon, market) in seen:
                continue
            seen.add((canon, market))
            # Prefer raw_prob (pre-blend) so it matches --prob-shrink 1.0
            # pure-model grading; fall back to the blended model_prob.
            p = r.get("raw_prob")
            if p is None:
                p = r.get("model_prob")
            try:
                p = float(p)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= p <= 1.0):
                continue
            out[market]["obs"].append((p, int(outcome)))
            price = r.get("price")
            if price is not None:
                try:
                    out[market]["roi"].append(
                        (bool(r.get("is_value")), int(price), int(outcome)))
                except (TypeError, ValueError):
                    pass
        except Exception:
            continue
    return out


def _print_log_supplement_roi(supplement):
    """Print the model-side ROI from the prediction-log supplement (model
    is_value picks staked at their bet-time price), in a block labeled SEPARATELY
    from the warehouse edge-vs-close ROI so the two ROI notions are never summed.
    Not written to calibration."""
    if not any(supplement[m]["roi"] for m in MARKETS):
        return
    print("\n[prediction-log supplement] model-side ROI — is_value picks at the "
          "bet-time price (model-side holdout only; NOT written to calibration):")
    hdr = f"  {'market':<10} {'obs':>5} {'bets':>5} {'ROI%':>8} {'P/L(u)':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for market in MARKETS:
        obs = supplement[market]["obs"]
        bets, profit = 0, 0.0
        for is_value, price, outcome in supplement[market]["roi"]:
            if not is_value:
                continue
            bets += 1
            profit += (american_to_decimal(price) - 1) if outcome else -1
        roi = (100.0 * profit / bets) if bets else float("nan")
        print(f"  {market:<10} {len(obs):>5} {bets:>5} {roi:>8.2f} {profit:>8.2f}")


MIN_SHRINK_N = 200  # min graded obs before a fitted shrink factor is persisted


def _write_shrink_calibration(sport_key, results, extra_obs=None,
                              min_shrink_n=MIN_SHRINK_N):
    """Fit and persist the Brier-optimal probability shrink per team market
    (from the 'live' variant) to calibration/<sport>.json.

    ``extra_obs`` (optional): {market: [(raw_prob, outcome)]} model-side rows from
    the prediction-log supplement, folded into the shrink fit and the published
    holdout Brier (see _market_log_supplement). market_brier and the blend weight
    stay over warehouse-only rows (log rows have no market prob).

    ``min_shrink_n`` guards the LIVE shrink factors: a factor is persisted only
    when the market has >= min_shrink_n graded obs, so a thin warehouse sample
    (early on, or a narrow --season/--limit run) can't clobber an established fit
    with noise (e.g. a 45-game sample driving moneyline shrink to 0.0). The
    holdout Brier is published for EVERY graded market regardless — it's
    informational (fills the app column) and never changes live behavior."""
    from datetime import datetime, timezone
    variant = "live" if "live" in results else next(iter(results), None)
    if not variant:
        print("  [write-calibration] No variant to write.")
        return
    extra_obs = extra_obs or {}
    shrink = {}
    holdout = {}
    withheld = []
    for market in MARKETS:
        blend = results[variant][market]["blend"]
        extra = [(p, None, o) for (p, o) in extra_obs.get(market, [])]
        combined = blend + extra
        res = _best_shrink(combined)
        if not res:
            continue
        best_s, best_brier, raw_brier = res
        # Publish the scored holdout for EVERY graded market (so the app shows a
        # real Brier per market), regardless of whether shrink is applied below.
        holdout[market] = {
            "brier": round(best_brier, 4),
            "raw_brier": round(raw_brier, 4),
            "n": len(combined),
            "n_warehouse": len(blend),
            "n_log": len(extra),
        }
        # Persist shrink only when it improves calibration AND the sample is big
        # enough to trust (else keep the existing factor untouched).
        if best_s < 1.0 and best_brier < raw_brier - 1e-9:
            if len(combined) >= min_shrink_n:
                shrink[market] = round(best_s, 2)
            else:
                withheld.append((market, round(best_s, 2), len(combined)))
    if withheld:
        print("  [write-calibration] shrink withheld (thin sample, "
              f"need n>={min_shrink_n}; existing factor kept): "
              + ", ".join(f"{m} s={s} n={n}" for m, s, n in withheld))
    if not shrink and not holdout:
        print("  [write-calibration] No market graded; nothing written.")
        return
    save_prob_shrink(sport_key, shrink, holdout=holdout, meta={
        "source": "odds backtest --engine live",
        "fit_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    print(f"\n  [write-calibration] Wrote prob_shrink to "
          f"calibration/{sport_key}.json: shrink={shrink}, holdout={holdout}")


def _inflate_samples(samples, weights, k):
    """Re-spread samples around their weighted mean by factor k (k>1 widens the
    distribution, fixing the variance compression from averaging two teams'
    series). Mean is preserved, so point projections don't move."""
    if not samples or abs(k - 1.0) < 1e-9:
        return samples
    mean = _weighted_mean(samples, weights) if weights else (sum(samples) / len(samples))
    return [mean + k * (s - mean) for s in samples]


def _live_stats(prior_games):
    """Minimal stats dict accepted by analyze_spreads_value / analyze_totals_value.
    Only 'recent_games' is used when a team has matching games (always true here);
    the 'recent'/'season' fallbacks are present to avoid KeyErrors."""
    return {
        "recent_games": prior_games,
        "recent": {"avg_scored": 0.0, "avg_allowed": 0.0, "win_pct": 0.0},
        "season": {"win_pct": 0.0},
    }


def _live_spread_total_probs(entry, home_prior, away_prior, threshold_pct, sport_key,
                             matchup_features=None):
    """Run the ACTUAL live analyzers and return the PURE-model (pre-blend)
    probabilities: (home_win_prob, (home_spread, P_home_cover),
    (total_line, P_over)). This makes the backtest grade exactly what
    production computes — including the MLB starter adjustment when
    ``matchup_features`` is supplied."""
    stats_h, stats_a = _live_stats(home_prior), _live_stats(away_prior)
    home_win = home_cover = total_over = None
    for c in analyze_moneyline_value(entry, stats_h, stats_a, threshold_pct, sport_key,
                                     matchup_features=matchup_features):
        if c["home_away"] == "HOME":
            home_win = c["model_prob"] / 100.0
    for c in analyze_spreads_value(entry, stats_h, stats_a, threshold_pct, sport_key,
                                   matchup_features=matchup_features):
        if c["home_away"] == "HOME":
            home_cover = (c["spread"], c["model_cover_rate"] / 100.0)
    tot = analyze_totals_value(entry, stats_h, stats_a, threshold_pct, sport_key,
                               matchup_features=matchup_features)
    if tot:
        total_over = (tot[0]["line"], tot[0]["model_over_hit_rate"] / 100.0)
    return home_win, home_cover, total_over


# Cache the MLB starter matchup-feature builder + per-season team index so the
# odds backtest grades the same starter-adjusted model production runs, without
# rebuilding the team index for every game.
_MLB_TEAM_INDEX = {}


def _mlb_matchup_features(home, away, date, sport_key):
    """Build MLB starter/opponent matchup features for a historical game the
    same way app.py does at run time. Returns None for non-MLB or when starters
    can't be resolved (degrades to the team-only model, matching production)."""
    if sport_key != "baseball_mlb" or not (home and away and date):
        return None
    try:
        import mlb_starters
        season = int(date[:4])
        idx = _MLB_TEAM_INDEX.get(season)
        if idx is None:
            idx = mlb_starters.get_team_index(season)
            _MLB_TEAM_INDEX[season] = idx
        return mlb_starters.build_matchup_features(home, away, date, season,
                                                   team_index=idx)
    except Exception:
        return None


def _nfl_matchup_features(home, away, date, sport_key):
    """Build NFL EPA matchup features for a historical game, leakage-safe: the
    EPA ratings use only plays from games strictly BEFORE `date`, matching what
    production would have known. Returns None for non-NFL or unresolved teams
    (degrades to the team-only model)."""
    if sport_key != "americanfootball_nfl" or not (home and away and date):
        return None
    try:
        import nfl_epa
        season = nfl_epa.season_for_date(date)
        ratings = nfl_epa.team_epa(season, as_of_date=date)
        return nfl_epa.build_matchup_features(home, away, date, season,
                                              team_ratings=ratings)
    except Exception:
        return None


def _matchup_features_for(home, away, date, sport_key):
    """Dispatch to the per-sport historical matchup-feature builder used by the
    odds backtest's live engine, so it grades the same feature-enhanced model
    production runs."""
    if sport_key == "baseball_mlb":
        return _mlb_matchup_features(home, away, date, sport_key)
    if sport_key == "americanfootball_nfl":
        return _nfl_matchup_features(home, away, date, sport_key)
    return None


def _shrink_prob(p, s):
    """Pull a probability toward 0.5 by factor s (s=1 unchanged, s=0 -> 0.5).
    Fixes overconfidence: a model 'p' becomes 0.5 + s*(p-0.5)."""
    return 0.5 + s * (p - 0.5)


def _parse_seasons(spec):
    """Parse a --seasons spec into a sorted list of ints. Accepts a comma list
    ('2023,2024,2025'), an inclusive range ('2023-2025'), or a single year."""
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(x) for x in spec.split(",") if x.strip()})


def run_odds_backtest(sport_key, espn_sport, espn_league, limit, window, variants,
                      min_sample=5, season_year=None, threshold_pct=5.0,
                      write_calibration=False, store_label="", variance_inflate=1.0,
                      engine="live", prob_shrink=1.0, source="auto",
                      supplement_log=True, min_shrink_n=MIN_SHRINK_N):
    """
    Grade the model's moneyline / spread / total value flags against stored
    historical closing lines: realized ROI, model-vs-market Brier, and the
    optimal model⇄market blend weight per market. Requires a prior
    `backfill_historical_odds.py` run.

    With write_calibration=True, the best blend weight per market is written to
    calibration/<sport>.json so the live analyzers blend the model toward the
    de-vigged market line automatically.

    NOTE: this measures ROI at the closing price and whether the model adds
    information over the closing line. True closing-line value (CLV) — the gain
    between your bet price and the close — needs a second, earlier snapshot per
    game and is not computed here.
    """
    variants = {name: _resolve_params(p, sport_key) for name, p in variants.items()}
    threshold = threshold_pct / 100.0

    store, source_used = _load_odds_store(sport_key, store_label, source)
    print(f"\n[odds source: {source_used}]")
    if not store.get("games"):
        if source_used == "warehouse":
            print(f"\nNo warehoused team-market lines for {sport_key} yet. The "
                  "Azure odds warehouse fills automatically as you run analyses "
                  "(every live event-odds fetch captures the closing lines).")
            print("Force the local backfill store instead with --source store.")
        else:
            cli = _SPORT_KEY_TO_CLI.get(sport_key, sport_key)
            lbl = f" --label {store_label}" if store_label else ""
            print(f"\nNo historical odds stored for {sport_key}"
                  f"{f' (label={store_label})' if store_label else ''}.")
            print(f"Run:  python backfill_historical_odds.py --sport {cli} "
                  f"--days 60 --max-credits 5000{lbl}")
        return
    if store_label and source_used == "store":
        print(f"\n[store-label: {store_label}] grading ROI at the "
              f"{store.get('snapshot_time','labeled')} price, not the close.")

    # P3b/P4/P6: MLB team schedules come from the StatsAPI warehouse; NBA/NFL stay on
    # ESPN. The odds-store join below prefers SFBB team CODES, so the canonical
    # warehouse names join fine. (ESPN was fully removed for MLB in P4.)
    use_warehouse = espn_sport == "baseball"
    # season_year may be a single year (int/None) or an iterable of years. When
    # several years are given we pool each season's schedule so the fit can span
    # multiple seasons (e.g. NFL, whose ~200 games/season are too thin alone).
    if isinstance(season_year, (list, tuple, set)):
        seasons_list = list(season_year)
    else:
        seasons_list = [season_year]

    print(f"\n=== Loading {sport_key} team list ===")
    if use_warehouse:
        print("=== team-market inputs: StatsAPI warehouse (ESPN bypassed) ===")
        espn_teams, schedules = _warehouse_team_schedules(seasons_list)
    else:
        espn_teams = get_all_teams(espn_sport, espn_league)
    lookup, unmatched = _build_odds_lookup(store, espn_teams)
    print(f"Stored games: {len(store['games'])} "
          f"(bookmaker: {store.get('bookmaker','?')}); "
          f"name-unmatched: {unmatched}")

    print(f"\n=== Fetching schedules (cached) ===")
    if not use_warehouse:
        schedules = {}
        for sy in seasons_list:
            sched = build_schedules(espn_sport, espn_league, espn_teams,
                                    season_year=sy)
            for tid, games in sched.items():
                schedules.setdefault(tid, []).extend(games)
    all_games = all_completed_games(schedules)
    if limit and limit < len(all_games):
        all_games = all_games[-limit:]

    if engine == "live":
        print("\n[engine: live] grading the exact production analyzers "
              "(analyze_spreads_value / analyze_totals_value); variants ignored.")
        variants = {"live": next(iter(variants.values()))}
    else:
        print(f"\n[engine: convolution] variance-inflate={variance_inflate} "
              "(diagnostic model; not what production runs).")

    results = {name: {m: _empty_market_bucket() for m in MARKETS}
               for name in variants}
    # (event_key, market) pairs graded from the warehouse/store — the log
    # supplement skips these so a warehouse-graded event is never double-counted.
    graded_keys = set()
    # Date span of matched games — bounds the log supplement to the same window
    # this run graded (so a --season/--limit scoped fit stays coherent).
    graded_lo = graded_hi = None

    matched = 0
    for game in all_games:
        date = game.get("date")
        home, away = game.get("home_team"), game.get("away_team")
        if not (date and home and away):
            continue
        entry = _lookup_game_odds(lookup, date[:10], home, away)
        if not entry:
            continue
        ml = _moneyline_market(entry)
        sp = _spread_market(entry)
        tot = _total_market(entry)
        if not (ml or sp or tot):
            continue

        home_prior = prior_games_for(home, schedules, espn_teams, date, window)
        away_prior = prior_games_for(away, schedules, espn_teams, date, window)
        if len(home_prior) < min_sample or len(away_prior) < min_sample:
            continue
        annotate_opponent_strength(home_prior, home, espn_teams)
        annotate_opponent_strength(away_prior, away, espn_teams)

        actual_margin = game["home_score"] - game["away_score"]
        actual_total = game.get("total_score") or (game["home_score"] + game["away_score"])
        home_won = 1 if actual_margin > 0 else 0
        matched += 1
        d10 = date[:10]
        if graded_lo is None or d10 < graded_lo:
            graded_lo = d10
        if graded_hi is None or d10 > graded_hi:
            graded_hi = d10

        # ── LIVE engine: grade the real production probabilities ──
        if engine == "live":
            r = results["live"]
            # Record BOTH the event_id (when present) and the game_key so the
            # log supplement dedups regardless of which the store carried:
            # game_key always matches across sources for the same game, and
            # event_id disambiguates same-day doubleheaders.
            ev_id = entry.get("event_id")
            gkey = hist_store.game_key(
                entry.get("commence_time"), entry.get("home_team"),
                entry.get("away_team"))
            ekeys = (gkey, ev_id) if ev_id else (gkey,)
            matchup_features = _matchup_features_for(home, away, date[:10], sport_key)
            mwin, mhc, mov = _live_spread_total_probs(
                entry, home_prior, away_prior, threshold_pct, sport_key,
                matchup_features=matchup_features)
            if ml and mwin is not None:
                fair_home, price_home, price_away = ml
                _grade(r["moneyline"], _shrink_prob(mwin, prob_shrink),
                       fair_home, home_won, price_home, price_away, threshold)
                graded_keys.update((t, "moneyline") for t in ekeys)
            if sp and mhc is not None:
                home_spread, fair_cover, price_h, price_a = sp
                model_spread, model_cover = mhc
                if (abs(model_spread - home_spread) < 1e-9
                        and abs(actual_margin + home_spread) > 1e-9):
                    home_covers = 1 if (actual_margin + home_spread) > 0 else 0
                    _grade(r["spreads"], _shrink_prob(model_cover, prob_shrink),
                           fair_cover, home_covers, price_h, price_a, threshold)
                    graded_keys.update((t, "spreads") for t in ekeys)
            if tot and mov is not None:
                line, fair_over, price_o, price_u = tot
                model_line, model_over = mov
                if (abs(model_line - line) < 1e-9
                        and abs(actual_total - line) > 1e-9):
                    over_hit = 1 if actual_total > line else 0
                    _grade(r["totals"], _shrink_prob(model_over, prob_shrink),
                           fair_over, over_hit, price_o, price_u, threshold)
                    graded_keys.update((t, "totals") for t in ekeys)
            continue

        for variant_name, params in variants.items():
            proj = project_matchup(home, away, home_prior, away_prior, params)
            if not proj:
                continue
            r = results[variant_name]
            samples_w = proj.get("sample_weights") or []
            margin_s = _inflate_samples(proj.get("margin_samples") or [],
                                        samples_w, variance_inflate)
            total_s = _inflate_samples(proj.get("total_samples") or [],
                                       samples_w, variance_inflate)

            # ── Moneyline ──
            if ml:
                fair_home, price_home, price_away = ml
                _grade(r["moneyline"], proj["home_win_prob"], fair_home,
                       home_won, price_home, price_away, threshold)

            # ── Spread (home cover) ──
            if sp and margin_s:
                home_spread, fair_cover, price_h, price_a = sp
                # Push: refund — skip grading this market for this game.
                if abs(actual_margin + home_spread) > 1e-9:
                    model_cover = _weighted_rate(
                        margin_s, samples_w,
                        lambda m, hs=home_spread: m > -hs)
                    home_covers = 1 if (actual_margin + home_spread) > 0 else 0
                    _grade(r["spreads"], model_cover, fair_cover,
                           home_covers, price_h, price_a, threshold)

            # ── Total (over) ──
            if tot and total_s:
                line, fair_over, price_o, price_u = tot
                if abs(actual_total - line) > 1e-9:
                    model_over = _weighted_rate(
                        total_s, samples_w,
                        lambda t, ln=line: t > ln)
                    over_hit = 1 if actual_total > line else 0
                    _grade(r["totals"], model_over, fair_over,
                           over_hit, price_o, price_u, threshold)

    print(f"\nMatched {matched} games to stored closing lines "
          f"(threshold {threshold_pct:.1f}%).")
    _print_odds_results(results)

    # Model-side holdout supplement from the durable prediction log (live only),
    # scoped to the same game-date window this run graded.
    supplement = None
    if engine == "live" and supplement_log:
        supplement = _market_log_supplement(sport_key, graded_keys,
                                            date_from=graded_lo, date_to=graded_hi)
        _print_log_supplement_roi(supplement)

    if write_calibration:
        if engine == "live":
            if abs(prob_shrink - 1.0) > 1e-9:
                print("  [write-calibration] Re-run with --prob-shrink 1.0 to fit "
                      "shrink on raw model probabilities; skipping write.")
            else:
                extra_obs = ({m: supplement[m]["obs"] for m in MARKETS}
                             if supplement else None)
                _write_shrink_calibration(sport_key, results, extra_obs=extra_obs,
                                          min_shrink_n=min_shrink_n)
        else:
            _write_blend_calibration(sport_key, results)


_RELIABILITY_EDGES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001)


def _reliability_rows(blend_obs, edges=_RELIABILITY_EDGES):
    """From [(model_p, market_p, outcome), ...] build calibration rows for the
    model's CHOSEN side: fold each obs to confidence = P(side the model leans),
    then bin. Returns [(lo, hi, n, pred_mean, actual_rate), ...].

    pred_mean ≈ actual_rate  =>  well-calibrated at that confidence level."""
    folded = []  # (confidence, won)
    for t in blend_obs:
        p, o = t[0], t[-1]
        if abs(p - 0.5) < 1e-9:
            continue
        if p > 0.5:
            folded.append((p, o))
        else:
            folded.append((1.0 - p, 1 - o))
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        bucket = [(c, w) for c, w in folded if lo <= c < hi]
        if not bucket:
            continue
        n = len(bucket)
        pred = sum(c for c, _ in bucket) / n
        actual = sum(w for _, w in bucket) / n
        rows.append((lo, min(hi, 1.0), n, pred, actual))
    return rows


def _print_reliability(title, named_obs):
    """named_obs: list of (label, blend_obs). Prints a confidence→accuracy
    calibration table so you can see if e.g. 75-80% model picks win ~75-80%."""
    print(f"\n{title}")
    print("  (model's chosen side: a well-calibrated pred% should match actual%; "
          "gap>0 = model underconfident, gap<0 = overconfident)")
    hdr = f"  {'pick':<14} {'conf bin':<11} {'N':>5} {'pred%':>7} {'actual%':>8} {'gap':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, obs in named_obs:
        rows = _reliability_rows(obs)
        if not rows:
            print(f"  {label:<14} (no picks)")
            continue
        first = True
        for lo, hi, n, pred, actual in rows:
            tag = label if first else ""
            first = False
            gap = actual - pred
            print(f"  {tag:<14} {lo*100:>2.0f}-{hi*100:<6.0f} {n:>5} "
                  f"{pred*100:>6.1f} {actual*100:>7.1f} {gap*100:>+6.1f}")


def _print_odds_results(results):
    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    for market in MARKETS:
        hdr = (f"[{market}]  {'variant':<14} {'N':>5} {'mdlBrier':>9} "
               f"{'mktBrier':>9} {'dir%':>6} {'bets':>5} {'ROI%':>8} {'P/L(u)':>8}")
        print("\n" + hdr)
        print("-" * len(hdr))
        for name, mr in results.items():
            r = mr[market]
            n = r["n"]
            pad = " " * (len(f"[{market}]  "))
            if not n:
                print(f"{pad}{name:<14} {0:>5}  (no matched games)")
                continue
            acc = 100.0 * r["correct"] / n
            bets = r["bets"]
            roi = (100.0 * r["profit"] / bets) if bets else float("nan")
            print(f"{pad}{name:<14} {n:>5} {_mean(r['model_brier']):>9.4f} "
                  f"{_mean(r['market_brier']):>9.4f} {acc:>6.1f} {bets:>5} "
                  f"{roi:>8.2f} {r['profit']:>8.2f}")

    print("\nOptimal model⇄market blend (w = model weight, minimizing Brier):")
    print(f"  {'market':<10} {'variant':<14} {'best w':>7} {'blendBrier':>11} "
          f"{'vs model':>9} {'vs market':>10}")
    for market in MARKETS:
        for name, mr in results.items():
            res = _best_blend_weight(mr[market]["blend"])
            if not res:
                continue
            best_w, best_brier, model_brier, market_brier = res
            print(f"  {market:<10} {name:<14} {best_w:>7.2f} {best_brier:>11.4f} "
                  f"{model_brier - best_brier:>+9.4f} {market_brier - best_brier:>+10.4f}")
    print("\n  (Lower Brier = better. 'dir%' = directional accuracy of the model "
          "side. 'vs model'/'vs market'")
    print("   = how much the blend beats each alone.) A best w < 1.0 means "
          "blending toward the market")
    print("   closing line improves accuracy. Use --write-calibration to save "
          "these weights for live use.\n")

    variant = "all" if "all" in results else next(iter(results), None)
    if variant:
        named = [(market, results[variant][market]["blend"]) for market in MARKETS]
        _print_reliability(
            f"Model calibration by confidence  (variant '{variant}')", named)


def _write_blend_calibration(sport_key, results):
    """Write the best blend weight per market (from the chosen variant) to
    calibration/<sport>.json so the live analyzers consume it."""
    from datetime import datetime, timezone
    # Prefer the production-like 'all' variant; else the first available.
    variant = "all" if "all" in results else next(iter(results), None)
    if not variant:
        print("  [write-calibration] No variants to write.")
        return
    blend = {}
    for market in MARKETS:
        res = _best_blend_weight(results[variant][market]["blend"])
        if not res:
            continue
        best_w, best_brier, model_brier, market_brier = res
        # Only persist a weight when blending actually helps over pure model.
        if best_brier < model_brier - 1e-9:
            blend[market] = {
                "w": best_w,
                "n": results[variant][market]["n"],
                "blend_brier": round(best_brier, 5),
                "model_brier": round(model_brier, 5),
                "market_brier": round(market_brier, 5),
            }
    if not blend:
        print("  [write-calibration] No market beat the pure model; nothing written.")
        return
    save_market_blend(sport_key, blend, meta={
        "variant": variant,
        "fit_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    print(f"\n  [write-calibration] Wrote blend weights (variant '{variant}') "
          f"to calibration/{sport_key}.json:")
    for market, cfg in blend.items():
        print(f"    {market:<10} w={cfg['w']:.2f}  (n={cfg['n']}, "
              f"blendBrier={cfg['blend_brier']})")


def _player_stat_series(espn_sport, espn_league, name, prop_key):
    """
    Return a player's dated per-game stat values for a prop as a sorted list of
    (game_date_iso, value). Empty if the player can't be resolved or the source
    lacks dated per-game data (e.g. ESPN MLB pitcher splits have no game dates).

    For MLB (P3c/P6) the per-game log comes from the StatsAPI warehouse
    (mlb_warehouse.get_calib_gamelog, by the name resolved to a role-verified MLBAM id
    via mlb_starters.resolve_mlbam_id), current-season scoped. NBA/NFL use the ESPN
    path. (ESPN was fully removed for MLB in P4.)
    """
    if espn_sport == "baseball":
        import mlb_warehouse
        import mlb_starters
        # prop_key role-gates the resolution, so a batter-prop name can't bind a
        # same-name pitcher (and vice-versa); role also picks the fact table.
        resolved = mlb_starters.resolve_mlbam_id(
            name, mlb_warehouse._current_season(), prop_key=prop_key)
        if not resolved:
            return []
        role = "pitcher" if str(prop_key).startswith("pitcher_") else "batter"
        gamelog = mlb_warehouse.get_calib_gamelog(str(resolved[0]), role) or []
    else:
        aid = cached_athlete_id(espn_sport, espn_league, name)
        if not aid:
            return []
        gamelog = cached_gamelog(espn_sport, espn_league, aid, player_name=name) or []
        if not gamelog and espn_sport == "baseball" and prop_key in (
                "pitcher_outs", "pitcher_strikeouts", "pitcher_earned_runs"):
            # Real StatsAPI per-game log (dated) when the name resolves; otherwise the
            # synthesized ESPN splits, whose dateless rows are dropped by the
            # (date, value) filter below.
            import mlb_starters
            gamelog = mlb_starters._pitcher_gamelog_or_synth(
                espn_league, aid, name, None) or []
    # Never resolve a pitcher prop from a batter's gamelog (or vice-versa): the
    # "K"/"SO" strikeout labels collide across roles (see _role_matches_gamelog).
    if not _role_matches_gamelog(prop_key, gamelog):
        return []
    label = _stat_label_for(prop_key, gamelog)
    if not label:
        return []
    out = []
    for g in gamelog:
        if g.get("completed") is False:
            continue                # in-progress/partial game -> not a final value
        d = g.get("game_date")
        val = g.get(label)
        if not d or val is None:
            continue
        if prop_key == "pitcher_outs":
            val = ip_to_outs(val)   # IP notation -> outs
        out.append((d, float(val)))
    out.sort(key=lambda x: x[0])
    return out


def _props_p_over(prop_cfg, proj, line, vals, wts, emp_over):
    """Model P(stat > line): production calibration if available, else Gaussian."""
    if prop_cfg:
        p = apply_calibration_with_warmup(
            prop_cfg, proj, line, current_season_games=len(vals),
            empirical_over=emp_over)
        if p is not None:
            return max(0.0, min(1.0, p))
    sigma = _weighted_std(vals, wts, proj)
    if sigma and sigma > 0:
        return _norm_cdf((proj - line) / sigma)
    return emp_over


def _write_market_prior_calibration(sport_key, results):
    """Persist the Brier-optimal market-as-prior shrinkage k (P1.1a) per prop
    into calibration/<sport>.json as "market_prior_k".

    Only writes a prop whose best k > 0 AND whose blended Brier beats the pure
    model (k=0) — so activating the market prior never ships a regression. Reads
    the existing per-prop cfg and adds the knob so method/half_life/etc. are
    preserved. NOTE: k is chosen by single-split argmin (like the blend/shrink
    writers); the nested-CV confirmation gate is roadmap §2.1."""
    from datetime import datetime, timezone
    existing = load_calibration(sport_key) or {}
    to_write = {}
    chosen = {}
    for prop, r in results.items():
        res = _best_market_prior_k(r.get("prior_k") or [])
        if not res:
            continue
        best_k, best_brier, model_brier, _bets_k0, _bets_best = res
        if best_k > 0 and best_brier < model_brier - 1e-9:
            cfg = dict(existing.get(prop) or {})
            cfg["market_prior_k"] = best_k
            to_write[prop] = cfg
            chosen[prop] = best_k
    if not to_write:
        print("  [write-calibration] No prop improved under a market prior; "
              "nothing written.")
        return
    save_calibration(sport_key, to_write, meta={
        "market_prior_refit": {
            "source": "backtest --mode props-odds --write-calibration",
            "k_by_prop": chosen,
            "fit_timestamp": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"),
        },
    })
    print(f"  [write-calibration] Wrote market_prior_k for: {chosen}")


def run_props_odds_backtest(sport, espn_sport, espn_league, sport_key, props,
                            min_prior=5, half_life=None, threshold_pct=5.0,
                            store_label="", write_calibration=False):
    """
    Grade the model's player-prop value flags against stored historical closing
    lines (from backfill_historical_odds.py --props ...). For each captured
    book line we recompute the model's P(over) from the player's prior games,
    compare it to the de-vigged closing line, and measure ROI + model-vs-market
    Brier + the optimal model⇄market blend, per prop market.
    """
    threshold = threshold_pct / 100.0
    # P3c/P4/P6: MLB player gamelogs come from the StatsAPI warehouse; NBA/NFL stay on
    # ESPN. (ESPN was fully removed for MLB in P4.)
    use_warehouse = espn_sport == "baseball"
    if use_warehouse:
        print("=== props-odds player logs: StatsAPI warehouse (ESPN bypassed) ===")
    store = hist_store.load_store(sport_key, store_label)
    games = store.get("games", {})
    if not games:
        cli = _SPORT_KEY_TO_CLI.get(sport_key, sport_key)
        lbl = f" --label {store_label}" if store_label else ""
        print(f"\nNo historical odds stored for {sport_key}"
              f"{f' (label={store_label})' if store_label else ''}. Run "
              f"backfill_historical_odds.py --sport {cli} --props {','.join(props)}{lbl} ...")
        return
    if store_label:
        print(f"\n[store-label: {store_label}] grading ROI at the "
              f"{store.get('snapshot_time','labeled')} price, not the close.")

    calibration = load_calibration(sport_key)
    # Production applies a final Platt recalibration after residual calibration
    # (props.py). The k-sweep must be tuned on that SAME post-Platt prob, or k is
    # inflated by overconfidence Platt already removes. (Lazy import avoids any
    # import-order coupling; recalibration is otherwise unused here.)
    from recalibration import apply_platt, load_recalibration
    recal = load_recalibration(sport_key) or {}
    hl = half_life or _half_life_for(sport_key)
    results = {prop: _empty_market_bucket() for prop in props}
    series_cache = {}
    no_actual = {prop: 0 for prop in props}
    no_series = {prop: 0 for prop in props}

    def series(player, prop):
        k = (player, prop)
        if k not in series_cache:
            series_cache[k] = _player_stat_series(
                espn_sport, espn_league, player, prop)
        return series_cache[k]

    print(f"\n=== Props odds backtest: {sport_key} {props} ===")
    print(f"Stored games: {len(games)} (bookmaker: {store.get('bookmaker','?')})")

    for entry in games.values():
        gdate = entry.get("commence_time")
        if not gdate:
            continue
        d10 = gdate[:10]
        eprops = entry.get("props") or {}
        for prop in props:
            market = eprops.get(prop) or {}
            for player, info in market.items():
                line = info.get("line")
                over_imp = info.get("over_implied")
                under_imp = info.get("under_implied")
                over_price = info.get("over_price")
                under_price = info.get("under_price")
                if line is None or over_imp is None or under_imp is None:
                    continue
                ser = series(player, prop)
                if not ser:
                    no_series[prop] += 1
                    continue
                actual = None
                prior = []
                for dt, val in ser:
                    if dt[:10] == d10:
                        actual = val
                    elif dt < gdate:
                        prior.append(val)
                if actual is None or len(prior) < min_prior:
                    no_actual[prop] += 1
                    continue
                if abs(actual - line) < 1e-9:
                    continue  # push — refund
                wts = _recency_weights(len(prior), hl)
                proj = _weighted_mean(prior, wts)
                emp_over = _weighted_rate(prior, wts, lambda v, ln=line: v > ln)
                p_model = _props_p_over(calibration.get(prop), proj, line,
                                        prior, wts, emp_over)
                fair_over, _ = devig_two_way(over_imp, under_imp)
                outcome = 1 if actual > line else 0
                _grade(results[prop], p_model, fair_over, outcome,
                       over_price, under_price, threshold)
                # For the market-as-prior (P1.1a) k-sweep: record the PRODUCTION
                # final prob (post-Platt, matching props.py) so k is tuned on the
                # probability the app actually ships, plus the market fair prob,
                # the outcome, and the sample size n. (_grade/blend above keep the
                # pre-Platt prob to preserve existing backtest semantics.)
                rc = recal.get(prop) or {}
                p_final = p_model
                if rc.get("a") is not None:
                    adj = apply_platt(p_model, rc.get("a"), rc.get("b"))
                    if adj is not None:
                        p_final = max(0.0, min(1.0, adj))
                results[prop]["prior_k"].append(
                    (p_final, fair_over, outcome, len(prior)))

    # Diagnostics on coverage
    print("\nCoverage (why lines were dropped):")
    for prop in props:
        print(f"  {prop:<18} graded={results[prop]['n']:>5}  "
              f"no_dated_series={no_series[prop]:>5}  "
              f"no_actual/min_prior={no_actual[prop]:>5}")

    _print_props_odds_results(results, threshold_pct)

    if write_calibration:
        print("\n[write-calibration] Persisting market-as-prior k (P1.1a)...")
        _write_market_prior_calibration(sport_key, results)


def _print_props_odds_results(results, threshold_pct):
    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    hdr = (f"{'prop':<18} {'N':>5} {'mdlBrier':>9} {'mktBrier':>9} {'dir%':>6} "
           f"{'bets':>5} {'ROI%':>8} {'P/L(u)':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for prop, r in results.items():
        n = r["n"]
        if not n:
            print(f"{prop:<18} {0:>5}  (no gradeable lines)")
            continue
        acc = 100.0 * r["correct"] / n
        bets = r["bets"]
        roi = (100.0 * r["profit"] / bets) if bets else float("nan")
        print(f"{prop:<18} {n:>5} {_mean(r['model_brier']):>9.4f} "
              f"{_mean(r['market_brier']):>9.4f} {acc:>6.1f} {bets:>5} "
              f"{roi:>8.2f} {r['profit']:>8.2f}")

    print("\nOptimal model⇄market blend (w = model weight, minimizing Brier):")
    print(f"  {'prop':<18} {'best w':>7} {'blendBrier':>11} {'vs model':>9} {'vs market':>10}")
    for prop, r in results.items():
        res = _best_blend_weight(r["blend"])
        if not res:
            continue
        best_w, best_brier, model_brier, market_brier = res
        print(f"  {prop:<18} {best_w:>7.2f} {best_brier:>11.4f} "
              f"{model_brier - best_brier:>+9.4f} {market_brier - best_brier:>+10.4f}")

    print("\nOptimal market-as-prior shrinkage (P1.1a, w = n/(n+k) toward the "
          "market):")
    print(f"  {'prop':<18} {'best k':>6} {'k*Brier':>9} {'vs model':>9} "
          f"{'bets@k0':>8} {'bets@k*':>8}")
    for prop, r in results.items():
        res = _best_market_prior_k(r["prior_k"])
        if not res:
            continue
        best_k, best_brier, model_brier, bets_k0, bets_best = res
        print(f"  {prop:<18} {best_k:>6} {best_brier:>9.4f} "
              f"{model_brier - best_brier:>+9.4f} {bets_k0:>8} {bets_best:>8}")
    print("  best k>0 with k*Brier < model Brier = thin-sample edges the market "
          "prior should collapse (bets@k* < bets@k0).")

    print("\n  ROI = profit per 1u bet on flags where model edge over the de-vigged "
          "line ≥ threshold.")
    print("  Positive ROI with model Brier ≤ market Brier = a real prop edge.\n")

    _print_reliability(
        "Model calibration by confidence",
        [(prop, r["blend"]) for prop, r in results.items()])


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
            shrink_k=0.0, rest_adj=0.0, def_window=None,
            park_strength=0.0, rest_strength=0.0):
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
        "park_strength": park_strength,      # P1.2 ballpark road-context delta (MLB hits/ER)
        "rest_strength": rest_strength,      # §2.6 rest/days-off candidate feature (prop_features)
        # NB: no weather knob (P1.3). Weather (props._weather_factor_mult) is a
        # LIVE-ONLY signal — there's no historical per-game weather to reconstruct,
        # so a backtest variant would be a no-op. It ships gated on CLV instead.
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
    # ── Park-factor isolation (P1.2): identical MLB baseline (no decay),
    #    park adjustment off vs half vs full. Only batter_hits /
    #    pitcher_earned_runs move; other props are identical across the three.
    "park_off":     _preset(half_life=None),
    "park_half":    _preset(half_life=None, park_strength=0.5),
    "park_full":    _preset(half_life=None, park_strength=1.0),
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
    Cross-product for player-props sweep mode (P2.1b expanded, §2.6 rest axis).

    Sweeps knobs/features that each have a RUNTIME counterpart in
    props.analyze_player_props_value, so a knob the confirmation gate enables
    behaves live EXACTLY as it was validated. NBA-only preset knobs
    (use_minutes / pace_adj / rest_adj) are deliberately excluded — they have no
    MLB runtime, so selecting one would be a silent no-op (the trap P2.1 fixes):
      • half_life             (recency decay)
      • opp_defense_strength  (weight-side opponent-defense reweighting)   [P2.1b]
      • def_adj               (output-side opponent-defense scaling)
      • shrink_k              (Bayesian shrinkage toward the season mean)  [P2.1b]
      • venue                 (venue-match reweighting)
      • rest_strength         (§2.6 rest/days-off candidate feature, prop_features)

    The all-off cell 'none/opp0.0/defadj0.0/shrink0/ven0.0/rest0.0' is the
    baseline the P2.1 variant gate measures every candidate against
    (refit_calibration), so the sweep can never ship worse than plain recency
    (baseline = the floor). Grid size: 4×3×3×4×2×2 = 576 variants — bounded for
    the offline refit; the fetch happens once and each variant is a pure
    re-projection pass. The current shipped selections (hl∈{None,15},
    ven∈{0.0,0.25}, all other knobs 0) all lie inside this grid, so neither
    §2.1b nor §2.6 can regress a prop by dropping its winner.

    §2.6 rest axis note: rest is a GLOBAL {0.0, 1.0} axis, but prop_features
    restricts it to {pitcher_outs, pitcher_strikeouts, batter_hits} — for every
    OTHER prop rest_multiplier returns 1.0, so its rest1.0 cell is a byte-exact
    duplicate of its rest0.0 cell. `rest` is the INNERMOST loop so that, for any
    given knob combo, the rest0.0 variant is inserted immediately before its
    rest1.0 twin; _best_per_prop breaks ties with a strict `<` (first-seen wins),
    so an excluded prop can never persist rest_strength=1.0, and rest is adopted
    only where it strictly beats the gate.
    """
    half_lifes = [None, 5, 10, 15]
    opp_defenses = [0.0, 0.5, 1.0]
    def_adjs = [0.0, 0.5, 1.0]
    shrink_ks = [0, 5, 10, 15]
    venue_strengths = [0.0, 0.25]
    rest_strengths = [0.0, 1.0]   # §2.6 candidate feature; 0.0 first (tie-break)

    variants = {}
    for hl in half_lifes:
        for opp in opp_defenses:
            for da in def_adjs:
                for sk in shrink_ks:
                    for vs in venue_strengths:
                        for rs in rest_strengths:
                            hl_label = "none" if hl is None else f"hl{hl}"
                            name = (f"{hl_label}/opp{opp}/defadj{da}/"
                                    f"shrink{sk}/ven{vs}/rest{rs}")
                            variants[name] = _preset(
                                half_life=hl, opp_defense_strength=opp,
                                def_adj=da, shrink_k=sk, venue_strength=vs,
                                rest_strength=rs)
    return variants


# ────────────────────────────────────────────────────────────────
#  Player-props backtest
# ────────────────────────────────────────────────────────────────

# Manual fallback player lists per sport, used when --players is not provided
# AND no data-driven pool applies. Calibration refits build usage-representative
# pools instead (refit_calibration._mlb_player_pool / _nba_player_pool); these
# hand-picked names are a survivorship-biased convenience for quick ad-hoc
# backtest.py runs, NOT the calibration pool.
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
    "mlb": ["batter_hits", "batter_strikeouts", "pitcher_strikeouts",
            "pitcher_outs", "pitcher_earned_runs"],
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
    "pitcher_earned_runs": "ER",
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


def _iter_pool_players(players):
    """Normalize each player-pool entry to (mlb_id, role, name). The MLB pool is
    enriched to (mlb_id, role, name) tuples (refit_calibration._mlb_player_pool) so the
    warehouse path binds each player by his authoritative MLBAM id + role; ESPN / NBA /
    NFL pools and a --players override are plain name strings (mlb_id/role None)."""
    for entry in players:
        if isinstance(entry, (tuple, list)):
            vals = list(entry) + [None, None, None]
            yield vals[0], vals[1], vals[2]
        else:
            yield None, None, entry


def fetch_player_data(espn_sport, espn_league, players, season_year=None):
    """Resolve each player → (athlete_id, gamelog). Returns {name: gamelog_list}.

    For MLB (P3/P4/P6) each player's per-game log comes from the StatsAPI warehouse
    (mlb_warehouse.get_calib_gamelog by the pool's authoritative MLBAM id + role) — no
    name→ESPN-id round trip, no ESPN gamelog fetch. A bare-name --players override is
    resolved to its MLBAM id via the game-context-free resolver. NBA/NFL use the ESPN
    path. (ESPN was fully removed for MLB in P4.)"""
    data = {}
    for mlb_id, role, name in _iter_pool_players(players):
        if not name:
            continue
        if espn_sport == "baseball":
            import mlb_warehouse
            rid, rrole = mlb_id, role
            if not rid:
                # A bare-name --players override under the flag: resolve the
                # authoritative MLBAM id + role (get_calib_gamelog needs the role to
                # pick the batter vs pitcher fact table).
                import mlb_starters
                resolved = mlb_starters.resolve_mlbam_id(
                    name, season_year or mlb_warehouse._current_season())
                if resolved:
                    rid = resolved[0]
                    rrole = "pitcher" if resolved[1] else "batter"
            gamelog = (mlb_warehouse.get_calib_gamelog(
                str(rid), rrole, season=season_year) if rid else [])
            if not gamelog:
                print(f"  [skip] {name}: no warehouse gamelog")
                continue
            gamelog.sort(key=lambda g: g.get("game_date") or "", reverse=True)
            # The pool dedups by MLBAM id and can carry two DISTINCT players who share
            # a fullName (e.g. Will Smith the catcher + the pitcher). data is keyed by
            # name for display + the pitcher_team_name lookup, so disambiguate a
            # collision with the id rather than letting the second overwrite (and
            # silently drop) the first. Role is derived from each gamelog downstream,
            # so a suffixed key is harmless.
            key = name if name not in data else f"{name} ({rid})"
            data[key] = gamelog
            print(f"  [ok]   {key}: {len(gamelog)} games (warehouse)")
            continue
        aid = cached_athlete_id(espn_sport, espn_league, name)
        if not aid:
            print(f"  [skip] {name}: athlete not found")
            continue
        gamelog = cached_gamelog(espn_sport, espn_league, aid,
                                 season_year=season_year, player_name=name)
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


def _prop_role(prop_key):
    """'pitching' / 'hitting' for MLB pitcher_*/batter_* props, else None (a
    prop with no batter/pitcher role, e.g. NBA/NFL — no gate applies)."""
    if prop_key.startswith("pitcher_"):
        return "pitching"
    if prop_key.startswith("batter_"):
        return "hitting"
    return None


def _gamelog_is_pitcher(gamelog):
    """True when a gamelog belongs to a pitcher — the only MLB log that carries
    innings pitched. Same discriminator gamelog_store uses to tag rows."""
    return any("IP" in g for g in gamelog)


def _role_matches_gamelog(prop_key, gamelog):
    """Guard against cross-role stat-label collisions in the props sweep.

    The sweep applies EVERY prop to EVERY player's gamelog, and pitcher/batter
    strikeouts share the ESPN labels "K"/"SO" (pitchers log "K", batters "SO"),
    so ``_stat_label_for`` matches a batter's log for ``pitcher_strikeouts`` (via
    the "SO" fallback) and a pitcher's for ``batter_strikeouts`` (via "K"). Left
    ungated, a batter's strikeout games leak into the pitcher_strikeouts
    calibration pool (and pitchers' stats into the batter props). A pitcher prop
    must resolve only against a pitcher's gamelog, and vice-versa. Non-MLB props
    (role None) always match — there is no role concept."""
    role = _prop_role(prop_key)
    if role is None:
        return True
    return (role == "pitching") == _gamelog_is_pitcher(gamelog)


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


def _mlb_warehouse_defense_lookup(season_year=None):
    """Warehouse analog of _team_defense_lookup for the MLB sweep (P3/P6): returns
    (avg_lookup, series_lookup, league_avg) keyed on the CANONICAL StatsAPI team name,
    which is EXACTLY the ``opponent`` name get_calib_gamelog stamps on each player row
    (both via mlb_warehouse._team_name_map) — so _resolve_opp_pts_allowed / _asof match
    by construction, with no ESPN spelling gap to silently no-op the defense weighting.
    Built from mlb_game scores (mlb_warehouse._team_final_games); the FULL-season series
    is returned here and _resolve_opp_pa_asof does the strict-before-date leakage cut."""
    import mlb_warehouse
    names = mlb_warehouse._team_name_map()          # {mlbam_team_id: canonical_name}
    lookup, series = {}, {}
    for tid, tname in names.items():
        rows = []
        for g in mlb_warehouse._team_final_games(tid, season=season_year):
            if g["home_team"] == tname:
                allowed = g["away_score"]
            elif g["away_team"] == tname:
                allowed = g["home_score"]
            else:
                continue
            if allowed is not None:
                rows.append((g.get("date"), allowed))
        if rows:
            rows.sort(key=lambda r: r[0] or "", reverse=True)
            vals = [a for _, a in rows]
            lookup[tname] = sum(vals) / len(vals)
            series[tname] = rows
    league_avg = (sum(lookup.values()) / len(lookup)) if lookup else None
    return lookup, series, league_avg


def _resolve_opp_pa_asof(opp_name, test_date, team_series, window=None):
    """
    Return avg pts allowed by `opp_name` using only their games STRICTLY BEFORE
    `test_date` (leakage-safe). This is the backtest analogue of the runtime
    model, which sees only season-to-date opponent defense.

    window=None  → all prior games this season (season-to-date, matches runtime)
    window=N (>0) → only the trailing N games before test_date

    Returns None when no prior games exist (caller should then skip the
    adjustment rather than fall back to a full-season average, which would leak).
    """
    if not opp_name or not team_series:
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
    if window:
        prior = prior[:window]
    return sum(prior) / len(prior)


# Canonical tolerant team-defense lookup now lives in pricing_common so the
# runtime pricer (props.py) and this backtest agree on which matchups get the
# defense adjustment. Kept under the historical name for existing importers
# (book_line_calibration).
_resolve_opp_pts_allowed = _resolve_team_defense


def run_player_props_backtest(sport, espn_sport, espn_league, sport_key,
                              players, props, games_per_player, min_sample,
                              variants, sweep=False, season_year=None,
                              safe_mode=False, cushion_sweep=False,
                              safe_target=0.80, quantile_mode=False,
                              calibrate=False, cross_season="strict"):
    variants = {name: _resolve_params(p, sport_key) for name, p in variants.items()}
    # P3/P4/P6: MLB sweep inputs (player gamelogs + team apparatus) come from the
    # StatsAPI warehouse; NBA/NFL stay on ESPN. (ESPN fully removed for MLB in P4.)
    use_warehouse = espn_sport == "baseball"
    if use_warehouse:
        print("=== MLB sweep inputs: StatsAPI warehouse (ESPN bypassed) ===")
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

    # ── Reliability filter inputs (streak-based: see prop_filter.py) ──
    # Filtering is performed separately for every historical prediction below.
    # Applying it once to the complete season would let future games extend an
    # earlier streak or reveal a later layoff, changing what the model appeared
    # to know on the original prediction date.
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
                if use_warehouse:
                    # Warehouse rows carry MLBAM team_ids; the reliability filter reads
                    # only the schedule's game dates (_extract_schedule_dates), which
                    # _team_final_games supplies as 'date'.
                    import mlb_warehouse
                    team_schedules_for_filter[tid] = mlb_warehouse._team_final_games(
                        tid, season=season_year)
                else:
                    team_schedules_for_filter[tid] = cached_schedule(
                        espn_sport, espn_league, tid, season_year=season_year)
            except Exception:
                team_schedules_for_filter[tid] = []
    print(f"=== Reliability filter: as-of-date mode "
          f"(min_streak={sport_min_streak}) ===")

    # Build team-defense lookup if any variant uses defense weighting OR
    # the output-side defense adjustment.
    team_defense, team_defense_series, league_avg_def = {}, {}, None
    needs_defense = any(
        p.get("opp_defense_strength", 0.0) > 0 or p.get("def_adj", 0.0) > 0
        for p in variants.values()
    )
    if needs_defense:
        print("\n=== Fetching team schedules for defense lookup ===")
        if use_warehouse:
            team_defense, team_defense_series, league_avg_def = (
                _mlb_warehouse_defense_lookup(season_year=season_year))
        else:
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

    # Build team-id -> display-name map if any variant uses park factors
    # (P1.2). Needed to resolve the player's own home park from their team_id,
    # mirroring the production id_to_name lookup.
    team_id_to_name = {}
    pitcher_team_name = {}
    needs_park = any(
        (p.get("park_strength", 0.0) or 0.0) > 0 for p in variants.values())
    if needs_park:
        if use_warehouse:
            # MLBAM team-id → canonical name; warehouse pitcher rows DO carry the
            # pitcher's MLBAM team_id, so resolve his park from a row (no ESPN
            # search_athlete). (Dormant in the standard MLB grid — park_strength is
            # not a sweep axis — but kept ESPN-free for when a park variant is used.)
            import mlb_warehouse
            team_id_to_name = {str(k): v
                               for k, v in mlb_warehouse._team_name_map().items()}
            print(f"Built team-name map for park factors "
                  f"({len(team_id_to_name)} teams).")
            for _nm, _gl in player_data.items():
                if not _gamelog_is_pitcher(_gl):
                    continue
                _ptid = next((str(g.get("team_id")) for g in _gl
                              if g.get("team_id")), None)
                if _ptid and _ptid in team_id_to_name:
                    pitcher_team_name[_nm] = team_id_to_name[_ptid]
            if pitcher_team_name:
                print(f"Resolved park teams for {len(pitcher_team_name)} pitchers.")
        else:
            _park_teams = get_all_teams(espn_sport, espn_league)
            team_id_to_name = {
                str(info["id"]): nm for nm, info in _park_teams.items()
                if info.get("id")
            }
            print(f"Built team-name map for park factors ({len(team_id_to_name)} teams).")
            # Pitchers' real StatsAPI logs carry no ESPN team_id (StatsAPI ids are
            # MLBAM), so their home starts would drop out of the park baseline and
            # a home upcoming start would get no park adjustment. Resolve each
            # pitcher's team from the athlete record — mirroring production
            # props.py, which reads the pitcher's park from athlete.team_id, not
            # from per-game rows.
            for _nm, _gl in player_data.items():
                if not _gamelog_is_pitcher(_gl):
                    continue
                try:
                    _ath = search_athlete(espn_sport, espn_league, _nm)
                except Exception:
                    _ath = None
                _ptid = _ath.get("team_id") if _ath else None
                if _ptid and str(_ptid) in team_id_to_name:
                    pitcher_team_name[_nm] = team_id_to_name[str(_ptid)]
            if pitcher_team_name:
                print(f"Resolved park teams for {len(pitcher_team_name)} pitchers.")

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
    reliability_skips = defaultdict(int)

    for name, gamelog in player_data.items():
        is_pitcher_log = _gamelog_is_pitcher(gamelog)
        test_slice = gamelog[:games_per_player]
        for prop_key in props:
            # Role gate: a pitcher prop must only resolve against a pitcher's
            # gamelog (and a batter prop a batter's). Without this, the shared
            # "K"/"SO" strikeout labels leak batters into pitcher_strikeouts (and
            # vice-versa), inflating and corrupting the calibration pool.
            role = _prop_role(prop_key)
            if role is not None and (role == "pitching") != is_pitcher_log:
                continue
            stat_label = _stat_label_for(prop_key, gamelog)
            if not stat_label:
                continue

            for i, test_game in enumerate(test_slice):
                # An in-progress game (real pitcher logs mark today's live start
                # completed=False) is a partial line, not a final box score —
                # grading it would bias the pool. Skip it as a test game; it is
                # never in a prior_games slice (it's the newest row).
                if test_game.get("completed") is False:
                    continue
                actual = test_game.get(stat_label)
                if actual is None:
                    continue
                if prop_key == "pitcher_outs":
                    actual = ip_to_outs(actual)   # IP notation -> outs
                prior_games = gamelog[i + 1:]
                test_date = test_game.get("game_date")

                # Strict-season policy: drop prior games from earlier seasons
                # so we never project a player using stale (different team /
                # role / coach) data. `all` keeps the old cross-season pool.
                if cross_season == "strict":
                    prior_games = _filter_to_current_season(
                        prior_games, test_date, sport_key)

                # Rebuild eligibility using only information available before
                # this test game. In particular, the test game's minutes and
                # any later layoff/streak continuation must not affect whether
                # this prediction is graded.
                tid = test_game.get("team_id") or next(
                    (g.get("team_id") for g in prior_games if g.get("team_id")),
                    None,
                )
                player_team_name = (team_id_to_name.get(str(tid))
                                    if tid else None)
                if player_team_name is None:
                    # Pitcher logs carry no team_id; fall back to the athlete-
                    # record team so home-game parks stay in the baseline.
                    player_team_name = pitcher_team_name.get(name)
                schedule = (team_schedules_for_filter.get(str(tid))
                            if tid else None)
                filt = filter_player_gamelog(
                    prior_games,
                    schedule,
                    sport_key,
                    min_streak=sport_min_streak,
                    as_of_date=test_date,
                )
                if filt["skip_prediction"]:
                    reliability_skips[filt["skip_reason"] or "unknown"] += 1
                    skipped += 1
                    continue
                prior_games = filt["eligible_games"]

                # Drop prior games whose value for THIS prop's stat is None so it can't
                # poison prior_values / sum() (a legacy warehouse batter row can carry a
                # NULL TB/RBI predating the a68f4e6 capture; get_calib_gamelog emits it
                # as a present-but-None key, which g.get(label, 0.0) does NOT default).
                # No-op for always-populated labels (H/SO/K/ER/IP) and the ESPN path;
                # keeps prior_values index-aligned with the home/away/opponent arrays.
                prior_games = [g for g in prior_games
                               if g.get(stat_label) is not None]

                if len(prior_games) < min_sample:
                    skipped += 1
                    continue

                prior_values = [g.get(stat_label, 0.0) for g in prior_games]
                if prop_key == "pitcher_outs":
                    prior_values = [ip_to_outs(v) for v in prior_values]
                prior_minutes = [g.get("MIN", 0.0) for g in prior_games]
                prior_home_aways = [g.get("is_home") for g in prior_games]
                prior_opponents = [g.get("opponent") for g in prior_games]
                upcoming_is_home = test_game.get("is_home")
                upcoming_opp = test_game.get("opponent")
                upcoming_date = test_date

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
                    if output_def_s > 0 and team_defense_series and league_avg_def:
                        # Leakage-safe opponent defense: use only the opponent's
                        # games strictly before this game — season-to-date when
                        # def_window is None (matching the runtime model), else
                        # the trailing def_window games. Fitting on the
                        # full-season aggregate would leak future results and
                        # miscalibrate live probabilities (the residual
                        # distribution would be fit against projections the
                        # production model never produces). When no prior games
                        # exist yet, skip the adjustment rather than falling back
                        # to the (leaky) full-season average.
                        opp_pa = _resolve_opp_pa_asof(
                            upcoming_opp, upcoming_date,
                            team_defense_series, def_window)
                        if opp_pa:
                            projected *= 1.0 + output_def_s * (opp_pa / league_avg_def - 1.0)

                    # ── Rest-days adjustment ──
                    # rest_adj is the size of the B2B penalty (e.g., 0.05 = −5%).
                    # Only apply on B2B (days_rest == 1); leave normal/long rest alone.
                    rest_adj = params.get("rest_adj", 0.0) or 0.0
                    if rest_adj > 0 and days_rest is not None and days_rest <= 1:
                        projected *= (1.0 - rest_adj)

                    # ── Park-factor road-context delta (P1.2) ──
                    # Mirror production props._park_factor_mult: each past game's
                    # park = the player's own park when home, else the opponent's;
                    # the upcoming park is the home team's. Only batter_hits /
                    # pitcher_earned_runs move (park_factors.PROP_PARK_KIND).
                    park_s = params.get("park_strength", 0.0) or 0.0
                    if park_s > 0:
                        past_parks = []
                        for ph, opp in zip(prior_home_aways, prior_opponents):
                            if ph is True:
                                past_parks.append(player_team_name)
                            elif ph is False:
                                past_parks.append(opp)
                            else:
                                past_parks.append(None)
                        upcoming_park = (player_team_name if upcoming_is_home
                                         else upcoming_opp)
                        park_mult, _ = _park_factor_mult(
                            prop_key, past_parks, weights, upcoming_park, park_s)
                        projected *= park_mult

                    # ── §2.6 candidate-feature multiplier (rest/days-off, …) ──
                    # ONE source shared with the runtime + real-line diagnostic
                    # (prop_features). Scales the projection here (moves methods
                    # B/C/E) and — in the calib_obs block below — shifts the
                    # empirical line by its inverse (moves method A), exactly like
                    # production combined_mult / effective_line. A hard no-op (1.0)
                    # for props the feature doesn't apply to and for rest 0, so the
                    # all-off grid stays byte-identical to production. Uses only the
                    # per-game game_date strictly before the a-priori upcoming_date
                    # (no leakage).
                    feat_strengths = prop_features.strengths_from_params(params)
                    feat_mult = 1.0
                    if feat_strengths:
                        fm = prop_features.projection_multiplier(
                            prop_key, feat_strengths,
                            [g.get("game_date") for g in prior_games],
                            upcoming_date)
                        if fm and fm != 1.0:
                            feat_mult = fm
                            projected *= feat_mult

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
                            # §2.6: the empirical (method-A) over-rate reads the
                            # feature-shifted line (line / feat_mult), while the
                            # stored line stays the raw synthetic_line so the
                            # OUTCOME + methods B/C/E use it as the fixed target —
                            # the exact split production uses (effective_line for
                            # A, real line for B/C/E). feat_mult==1.0 → identical
                            # to production.
                            line_eff = (synthetic_line / feat_mult
                                        if feat_mult != 1.0 else synthetic_line)
                            empirical_over = _weighted_rate(
                                prior_values, weights, lambda v: v > line_eff)
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
    print(f"Skipped {skipped} game-prop observations (eligibility, history, or data)")
    if reliability_skips:
        print("As-of reliability skips: " + ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(reliability_skips.items())))

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
    if len(obs) < (40 if holdout else 20):
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

    return _score_calibration_methods(fit_obs, score_obs, k_values)


def _score_calibration_methods(fit_obs, score_obs, k_values):
    """
    Fit calibration params on `fit_obs` and score them on `score_obs`.

    Returns a list of {method, k, brier, hit} for methods A (empirical),
    B (pooled Gaussian), C (pooled ECDF) and B*/C* (per-player shrinkage) at
    each k. Both inputs use the calib_obs schema
    (name, projected, line, actual, empirical_over, date). Splitting fit from
    score lets callers supply arbitrary chronological folds (e.g. the
    confirmation folds used by the calibration refit) without re-deriving the
    method math. Returns [] when either set has no usable rows.
    """
    if not fit_obs or not score_obs:
        return []

    # Pool stats from fit_obs
    all_resid = [actual - proj for _, proj, _, actual, _, *_ in fit_obs]
    if not all_resid:
        return []
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
    Parse variant names like 'hl10/opp0.5/defadj1.0/shrink5/ven0.25' into a dict
    of knob values. Recognizes prefixes: hl, opp, defadj, def, shrink, ven, pace.
    """
    parts = {}
    # Order matters — match longer prefixes first ("defadj" before "def")
    prefixes = ("defadj", "shrink", "pace", "opp", "hl", "def", "ven")
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
        "opp": "Defense weight strength (opp_defense)",
        "def": "Defense weight strength",
        "defadj": "Defense output adj. strength",
        "shrink": "Bayesian shrinkage k",
        "pace": "Pace adj. strength",
        "ven": "Venue strength",
    }
    # Only display knobs that actually appear in the parsed variants
    knobs_present = {k for _, parts, _ in parsed for k in parts}
    for knob in ("hl", "opp", "defadj", "def", "shrink", "pace", "ven"):
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
    for knob in ("hl", "opp", "defadj", "def", "shrink", "pace", "ven"):
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
    p.add_argument("--mode", choices=["matchup", "props", "odds", "props-odds"],
                   default="matchup",
                   help="matchup = team-level projections; props = player-prop "
                        "projections; odds = grade team markets vs stored closing "
                        "lines; props-odds = grade player props vs stored closing "
                        "lines (ROI + model⇄market blend)")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="(odds mode) Min edge %% over the de-vigged market to place a bet.")
    p.add_argument("--write-calibration", action="store_true",
                   help="(odds mode) Save the best model⇄market blend weight per "
                        "market to calibration/<sport>.json for live use.")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="nba")
    p.add_argument("--season", type=int, default=None,
                   help="ESPN season year (e.g., 2025 = 2024-25 NBA season). Default: current.")
    p.add_argument("--seasons", default=None,
                   help="(odds mode) Multiple ESPN season years to POOL for one fit: "
                        "comma list '2023,2024,2025' or range '2023-2025'. Overrides "
                        "--season. Use when a single season is too thin to fit shrink.")
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
    p.add_argument("--store-label", default="",
                   help="(odds/props-odds) Grade against a labeled historical "
                        "store, e.g. 'morning' -> baseball_mlb__morning.json. "
                        "Default '' uses the closing store. Use this to measure "
                        "ROI at the before-noon price you actually bet.")
    p.add_argument("--engine", choices=["live", "convolution"], default="live",
                   help="(odds mode) 'live' (default) grades the exact production "
                        "analyzers (analyze_spreads_value/analyze_totals_value) so "
                        "calibration matches reality. 'convolution' uses the older "
                        "diagnostic model that understated variance (pair with "
                        "--variance-inflate to reconcile it to live).")
    p.add_argument("--variance-inflate", type=float, default=1.0,
                   help="(odds mode, convolution engine) Widen the diagnostic "
                        "model's outcome distribution by this factor (1.0 = off). "
                        "k≈2.0 reproduces the live engine's variance.")
    p.add_argument("--prob-shrink", type=float, default=1.0,
                   help="(odds mode, live engine) Pull model spread/total "
                        "probabilities toward 0.5 by this factor (1.0 = off, "
                        "0.5 = halve the edge). Fixes overconfidence; sweep and "
                        "watch the calibration table to find the value that "
                        "flattens it.")
    p.add_argument("--cross-season", choices=["strict", "all"], default="strict",
                   help="(props mode) 'strict' (default) keeps only current-season "
                        "prior games per test observation; 'all' uses the full "
                        "ESPN gamelog regardless of season boundary.")
    p.add_argument("--source", choices=["auto", "warehouse", "store"],
                   default="auto",
                   help="(odds mode) Closing-line source: 'auto' (default) uses "
                        "the Azure odds warehouse when it's configured and has "
                        "team-market games, else the local historical_odds JSON; "
                        "'warehouse' forces the warehouse; 'store' forces the "
                        "local JSON. Under 'auto', a --store-label forces the "
                        "local JSON (the warehouse has no label concept).")
    p.add_argument("--supplement-log", dest="supplement_log",
                   action="store_true", default=True,
                   help="(odds mode, live engine) Fold resolved market_prediction_"
                        "log rows into the model-side holdout (default on).")
    p.add_argument("--no-supplement-log", dest="supplement_log",
                   action="store_false",
                   help="Disable the prediction-log holdout supplement.")
    p.add_argument("--min-shrink-n", type=int, default=MIN_SHRINK_N,
                   help="(odds mode) Min graded obs before a fitted team-market "
                        f"shrink factor is persisted (default {MIN_SHRINK_N}). "
                        "Below this, only the informational holdout Brier is "
                        "written; the existing shrink factor is kept so a thin "
                        "sample can't clobber a good fit. The Holdout Brier "
                        "column still fills.")
    args = p.parse_args()

    # Target the Azure SQL warehouse/logs when the SQL_* secrets are configured
    # (outside Streamlit they aren't in the env yet). Guarded; a no-op when SQL
    # isn't configured so --source auto/store still works offline.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

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

    if args.mode == "props-odds":
        props = ([x.strip() for x in args.props.split(",") if x.strip()]
                 if args.props else DEFAULT_PROPS.get(args.sport, []))
        if not props:
            print(f"No props for sport={args.sport}. Use --props.")
            sys.exit(1)
        run_props_odds_backtest(args.sport, espn_sport, espn_league, sport_key,
                                props=props, min_prior=args.min_sample,
                                threshold_pct=args.threshold,
                                store_label=args.store_label,
                                write_calibration=args.write_calibration)
    elif args.mode == "odds":
        odds_seasons = _parse_seasons(args.seasons) if args.seasons else args.season
        run_odds_backtest(sport_key, espn_sport, espn_league,
                          limit=args.limit, window=args.window, variants=variants,
                          min_sample=args.min_sample, season_year=odds_seasons,
                          threshold_pct=args.threshold,
                          write_calibration=args.write_calibration,
                          store_label=args.store_label,
                          variance_inflate=args.variance_inflate,
                          engine=args.engine, prob_shrink=args.prob_shrink,
                          source=args.source, supplement_log=args.supplement_log,
                          min_shrink_n=args.min_shrink_n)
    elif args.mode == "matchup":
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
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
