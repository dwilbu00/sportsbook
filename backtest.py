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
import analysis
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
from stats import negbin_at_least, fit_negbin_params  # §2.2 method "E"
from pricing_common import (_resolve_team_defense, kelly_stake,
                            kelly_stake_uncertain, prob_interval_low,
                            _expected_roi)
import historical_odds as hist_store
import prop_features  # §2.6 candidate-feature registry (rest/days-off, …)
import park_factors     # PROP_PARK_KIND — weather/park apply to hits/ER only
import weather_factors  # air_density / density_factor for the weather-density sweep
from calibration_loader import (
    save_market_blend,
    save_prob_shrink,
    save_expected_runs_challenger_shares,
    save_calibration,
    load_calibration,
    apply_calibration_with_warmup,
    set_candidate_mode,
    set_serving_candidate,
    has_candidate,
    active_write_label,
    existing_candidate_notice,
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
            # #2b: include game_pk so a doubleheader's two results (same date+teams)
            # are NOT deduped down to one. NULL game_pk (non-warehouse schedules)
            # -> 4th element None for every row -> identical dedup to the pre-#2b key.
            key = (g.get("date"), g.get("home_team"), g.get("away_team"),
                   g.get("game_pk"))
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


def _et_date10(ts):
    """US-Eastern calendar date (YYYY-MM-DD) for a commence/timestamp, falling back
    to the raw prefix. The odds<->schedule join keys on the ET PLAY date so
    consecutive-day same-matchup games (which can share a UTC date) stay distinct —
    the same normalization game_key uses. Both sides must use this or the collapse
    persists. A bare date (no time component) is returned unshifted — there is no
    time-of-day to convert, and treating it as UTC-midnight would wrongly roll it
    back a day."""
    s = ts or ""
    if "T" not in s:                 # bare 'YYYY-MM-DD' (or empty) → no conversion
        return s[:10]
    try:
        from pricing_common import et_local_date
        return et_local_date(ts) or s[:10]
    except Exception:
        return s[:10]


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
        date10 = _et_date10(entry.get("commence_time"))
        # #2b: a positive StatsAPI game_pk is a globally-unique, DH-exact join key
        # (no date/teams needed) — emit it so the results side can prefer it. NULL
        # game_pk (legacy) emits nothing here -> the id/name keys below are unchanged.
        gpk = entry.get("game_pk")
        if gpk is not None:
            lookup[("pk", gpk)] = entry
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


def _lookup_game_odds(lookup, date10, home, away, game_pk=None):
    """Find a stored game, preferring an exact StatsAPI game_pk (#2b: DH-exact, so a
    doubleheader's two games never cross-match), then canonical SFBB team codes
    (robust to franchise renames / name-spelling drift), then the ESPN team-name key
    over a ±1-day window. Fail-open on the code resolution. NULL game_pk skips the
    pk key -> byte-identical to the pre-#2b lookup."""
    if game_pk is not None:
        hit = lookup.get(("pk", game_pk))
        if hit:
            return hit
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
            "bets": 0, "profit": 0.0, "blend": [], "prior_k": [],
            # (selected-side de-vigged MARKET prob, decimal_odds, won) per PLACED bet
            # — lets us bucket realized ROI by MARKET CONFIDENCE (are the model's edges
            # real only in the uncertain/near-pickem zone, or across all prices?).
            "bets_detail": [],
            # (date, served_prob, fair_over, outcome, over_price, under_price, n, cv)
            # per OBS — the SERVED (line-conditional Platt) prob + the bettor's prior
            # game-count n + recency-weighted CV (per-player outcome volatility), for
            # the fit==serve EV-gate sweep, Kelly-staking, and the ROI-by-sample-size
            # / ROI-by-variance diagnostics (props only; team uses its own lens).
            "ev_obs": [],
            # (date, p_model[pre-Platt], line, outcome, over_price, under_price) per OBS
            # — feeds the --walk-forward sim (deep seed + online current-season Platt).
            "wf_obs": []}


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
        # selected side = yes: its de-vigged market prob is market_p.
        bucket["bets_detail"].append(
            (market_p, american_to_decimal(price_yes), bool(outcome)))
    if price_no is not None and (1 - model_p) - (1 - market_p) >= threshold:
        bucket["bets"] += 1
        bucket["profit"] += (american_to_decimal(price_no) - 1) if not outcome else -1
        # selected side = no: its de-vigged market prob is 1 - market_p.
        bucket["bets_detail"].append(
            (1.0 - market_p, american_to_decimal(price_no), not outcome))


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


def _load_odds_store(sport_key, store_label, source, snapshot="close"):
    """Return ``(store, source_used)`` for the team-market backtest.

    ``source``: 'auto' (default) uses the SQL odds warehouse when it's enabled,
    no --store-label was given, and it holds team-market games — else the local
    historical_odds JSON; 'warehouse' forces the warehouse; 'store' forces the
    local JSON. The warehouse store is assembled to the EXACT historical_odds
    shape so the downstream grading path is identical. ``snapshot`` ('close' |
    'early') selects the warehoused snapshot set (ignored for the local JSON)."""
    def _local():
        return hist_store.load_store(sport_key, store_label), "store"

    if source == "store":
        return _local()
    if source in ("auto", "warehouse"):
        try:
            import warehouse
            wh = warehouse.load_team_market_store(sport_key, snapshot=snapshot)
        except Exception:
            wh = None
        if source == "warehouse":
            return (wh or {"sport_key": sport_key, "games": {}}), "warehouse"
        # auto: warehouse only when it has games and no explicit local label.
        if wh and wh.get("games") and not store_label:
            return wh, "warehouse"
        return _local()
    return _local()


def _load_prop_store(sport_key, store_label, source, snapshot="close"):
    """Return ``(store, source_used)`` for the PLAYER-PROPS odds backtest — the prop
    sibling of _load_odds_store. 'warehouse' assembles the historical_odds shape
    from the SQL warehouse (warehouse.load_prop_market_store) at the chosen
    ``snapshot`` ('close' | 'early'); 'store' forces the local historical_odds JSON;
    'auto' prefers the warehouse when it has games and no --store-label was given.
    Note the local JSON stores have historically held TEAM markets only (no props),
    so the warehouse is the real props source."""
    def _local():
        return hist_store.load_store(sport_key, store_label), "store"

    if source == "store":
        return _local()
    if source in ("auto", "warehouse"):
        try:
            import warehouse
            wh = warehouse.load_prop_market_store(sport_key, snapshot=snapshot)
        except Exception:
            wh = None
        if source == "warehouse":
            return (wh or {"sport_key": sport_key, "games": {}}), "warehouse"
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
                              min_shrink_n=MIN_SHRINK_N, skip_markets=()):
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
        return {}
    extra_obs = extra_obs or {}
    shrink = {}
    holdout = {}
    withheld = []
    for market in MARKETS:
        if market in skip_markets:
            continue
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
        return {}
    save_prob_shrink(sport_key, shrink, holdout=holdout, meta={
        "source": "odds backtest --engine live",
        "fit_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    print(f"\n  [write-calibration] Wrote prob_shrink to "
          f"calibration/{active_write_label(sport_key)}: shrink={shrink}, "
          f"holdout={holdout}")
    # Return the persisted per-market factors so a chained blend fit (--fit-blend)
    # can pull the model prob through the SAME shrink before fitting the blend
    # weight — matching the serve order (_apply_shrink then blend).
    return shrink


def _inflate_samples(samples, weights, k):
    """Re-spread samples around their weighted mean by factor k (k>1 widens the
    distribution, fixing the variance compression from averaging two teams'
    series). Mean is preserved, so point projections don't move."""
    if not samples or abs(k - 1.0) < 1e-9:
        return samples
    mean = _weighted_mean(samples, weights) if weights else (sum(samples) / len(samples))
    return [mean + k * (s - mean) for s in samples]


def _asof_season_runs(team_name, schedules, espn_teams, before_date):
    """As-of cumulative (runs_scored, runs_allowed, win_pct) for a team over all its
    games this season STRICTLY BEFORE ``before_date``, summed from the already-loaded
    schedules (zero extra queries). Feeds the Pythagorean season block so the odds
    backtest grades production's run-differential blend (DEFAULT_PYTHAG_WEIGHT) AND
    the #29 residual (actual win% − pythag), both otherwise inert in the harness
    (_live_stats carried no runs/record). Returns None when the team has no prior
    games. MLB-only by construction — the caller gates on the warehouse path."""
    info = espn_teams.get(team_name)
    if not info:
        for name, i in espn_teams.items():
            if name.lower() == team_name.lower():
                info = i
                break
    if not info:
        return None
    rs = ra = wins = games = 0
    for g in schedules.get(info["id"], []):
        if (g.get("date") or "") >= before_date:
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if g.get("home_team") == team_name:
            rs += hs or 0
            ra += as_ or 0
            games += 1
            wins += 1 if (hs is not None and as_ is not None and hs > as_) else 0
        elif g.get("away_team") == team_name:
            rs += as_ or 0
            ra += hs or 0
            games += 1
            wins += 1 if (hs is not None and as_ is not None and as_ > hs) else 0
    if not games:
        return None
    return (rs, ra, wins / games)


def _live_stats(prior_games, season_runs=None):
    """Minimal stats dict accepted by analyze_spreads_value / analyze_totals_value.
    Only 'recent_games' is used when a team has matching games (always true here);
    the 'recent'/'season' fallbacks are present to avoid KeyErrors. ``season_runs``
    = (runs_scored, runs_allowed[, win_pct]) as-of, populated for MLB so the
    Pythagorean blend + #29 residual (analyze_moneyline_value) are actually graded —
    production feeds these from /standings; the backtest sums them from the
    schedule."""
    season = {"win_pct": 0.0}
    if season_runs:
        season["runs_scored"] = season_runs[0]
        season["runs_allowed"] = season_runs[1]
        if len(season_runs) > 2:
            season["win_pct"] = season_runs[2]
    return {
        "recent_games": prior_games,
        "recent": {"avg_scored": 0.0, "avg_allowed": 0.0, "win_pct": 0.0},
        "season": season,
    }


def _live_spread_total_probs(entry, home_prior, away_prior, threshold_pct, sport_key,
                             matchup_features=None, home_season_runs=None,
                             away_season_runs=None, serve_mode=False):
    """Run the ACTUAL live analyzers and return probabilities:
    (home_win_prob, (home_spread, P_home_cover), (total_line, P_over)).

    serve_mode=False (default): PURE-model (pre-shrink/pre-blend) fields — what the
    shrink FIT needs and what the raw model + a single --prob-shrink grades.
    serve_mode=True: the SERVED fields (per-market prob_shrink + model<->market blend
    already applied inside the analyzers) — so a holdout grades EXACTLY what production
    would serve under the (staged) calibration. Includes the MLB starter adjustment
    (matchup_features) + Pythagorean blend (home/away_season_runs).

    Also returns ``spread_comp``: the RAW spread ensemble components for the HOME side
    when the expected-runs projection fired (recency cover pre-shrink, additive/expected
    cover, and the two point margins) — so the challenger ensemble share can be fit
    offline from clean inputs (see _write_shares_calibration). None otherwise."""
    stats_h = _live_stats(home_prior, home_season_runs)
    stats_a = _live_stats(away_prior, away_season_runs)
    ml_key = "blended_prob" if serve_mode else "model_prob"
    sp_key = "cover_rate" if serve_mode else "model_cover_rate"
    tot_key = "over_hit_rate" if serve_mode else "model_over_hit_rate"
    home_win = home_cover = total_over = home_pythag = spread_comp = None
    for c in analyze_moneyline_value(entry, stats_h, stats_a, threshold_pct, sport_key,
                                     matchup_features=matchup_features):
        if c["home_away"] == "HOME":
            home_win = c[ml_key] / 100.0
            # Raw home Pythagorean win prob (computed regardless of blend weight) —
            # captured so an A(recency) x B(pythag) sweep can recombine them offline.
            if c.get("pythag_win_pct") is not None:
                home_pythag = c["pythag_win_pct"] / 100.0
    for c in analyze_spreads_value(entry, stats_h, stats_a, threshold_pct, sport_key,
                                   matchup_features=matchup_features):
        if c["home_away"] == "HOME":
            home_cover = (c["spread"], c[sp_key] / 100.0)
            # Present only when the expected-runs ensemble fired (MLB, data complete).
            if c.get("current_model_cover_rate") is not None \
                    and c.get("expected_runs_cover_rate") is not None:
                spread_comp = {
                    "current_cover": c["current_model_cover_rate"] / 100.0,
                    "expected_cover": c["expected_runs_cover_rate"] / 100.0,
                    "current_margin": c.get("current_pred_game_margin"),
                    "expected_margin": (c["expected_home_runs"] - c["expected_away_runs"]
                                        if c.get("expected_home_runs") is not None
                                        and c.get("expected_away_runs") is not None
                                        else None),
                }
    tot = analyze_totals_value(entry, stats_h, stats_a, threshold_pct, sport_key,
                               matchup_features=matchup_features)
    if tot:
        total_over = (tot[0]["line"], tot[0][tot_key] / 100.0)
    return home_win, home_cover, total_over, home_pythag, spread_comp


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
        # as_of_date=date makes the starter line stats leakage-safe (byDateRange up
        # to date-1) — the backtest must not see the pitcher's full-season line.
        return mlb_starters.build_matchup_features(home, away, date, season,
                                                   team_index=idx, as_of_date=date)
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


def _prewarm_matchup_features(pw_games, schedules, sport_key, use_warehouse,
                             compute_fn=None):
    """Build {(d10, home, away): features} for the live-engine grading loop, reusing
    the feature_store cache so repeat backtests skip the slow prewarm.

    Per-season cache version = count:max_date of the FULL warehouse schedule
    (all_completed_games) → stable across --limit / season range (a partial run can't
    poison it), and a season whose warehouse changed (2026 filling, a re-ingest) bumps
    the version → automatic rebuild. None feature values (unresolved starters or a
    transient build error) are NEVER cached, so they can't persistently degrade a
    later run. Bypass with ODI_NO_FEATURE_CACHE=1; hard-reset via feature_store.clear().
    fit==serve: caches the EXACT _matchup_features_for output grading would compute.
    ``compute_fn`` is a test seam (defaults to _matchup_features_for)."""
    import feature_store
    from collections import defaultdict
    compute = compute_fn or _matchup_features_for
    prewarm = {}
    pw = [g for g in pw_games
          if g.get("date") and g.get("home_team") and g.get("away_team")]
    if not pw:
        return prewarm
    use_cache = bool(use_warehouse and not os.environ.get("ODI_NO_FEATURE_CACHE"))
    cache, ver, dirty = {}, {}, set()
    if use_cache:
        sg = defaultdict(list)
        for g in all_completed_games(schedules):
            if g.get("date"):
                sg[int(str(g["date"])[:4])].append(g)
        for s, gs in sg.items():
            ver[s] = feature_store.season_version(gs)
            cache[s] = feature_store.load(sport_key, s, ver[s]) or {}

    def _one(g):
        d10 = _et_date10(g["date"])
        return ((d10, g["home_team"], g["away_team"]),
                compute(g["home_team"], g["away_team"], d10, sport_key))

    todo, hits = [], 0
    for g in pw:
        d10 = _et_date10(g["date"])
        key = (d10, g["home_team"], g["away_team"])
        s = int(d10[:4]) if d10[:4].isdigit() else None
        if use_cache and s is not None and key in cache.get(s, {}):
            prewarm[key] = cache[s][key]
            hits += 1
        else:
            todo.append((g, s))
    if todo:
        print(f"  [prewarm] matchup features: {hits} cached, {len(todo)} to build "
              f"(thread pool) ...")
        with ThreadPoolExecutor(max_workers=16) as pool:
            futs = {pool.submit(_one, g): s for g, s in todo}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    k, v = fut.result()
                    prewarm[k] = v
                    if use_cache and s is not None and v is not None:
                        cache.setdefault(s, {})[k] = v
                        dirty.add(s)
                except Exception:
                    pass
        for s in dirty:
            feature_store.save(sport_key, s, ver[s], cache[s])
    else:
        print(f"  [prewarm] matchup features: all {hits} from cache.")
    return prewarm


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


def run_odds_backtest(*args, **kwargs):
    """Thin wrapper: scope the pitcher_asof series memo cache to the WHOLE backtest
    pass (offline only) so the additive live-engine path fetches each (entity, season)
    as-of series ONCE instead of re-reading it per game (the RP series is otherwise
    re-read for every one of a team's games). The cache is OFF everywhere else, so
    live serving is byte-identical. See pitcher_asof.series_cache()."""
    import pitcher_asof
    with pitcher_asof.series_cache():
        return _run_odds_backtest_impl(*args, **kwargs)


def _run_odds_backtest_impl(
        sport_key, espn_sport, espn_league, limit, window, variants,
        min_sample=5, season_year=None, threshold_pct=5.0,
        write_calibration=False, store_label="", variance_inflate=1.0,
        engine="live", prob_shrink=1.0, source="auto", snapshot="close",
        supplement_log=True, min_shrink_n=MIN_SHRINK_N,
        collect_obs=None, collect_dated=None, collect_lineup=None,
        collect_components=None, serve_mode=False, fit_blend=False,
        fit_shares=False, collect_bets=None):
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

    if serve_mode:
        # Serve-mode grades the ALREADY-served probs (per-market prob_shrink + blend
        # applied inside the analyzers). It is a holdout-validation pass, mutually
        # exclusive with FITTING: --write-calibration and raw-obs collection both
        # need the pre-shrink/pre-blend model prob.
        if write_calibration or collect_obs is not None:
            raise ValueError(
                "serve_mode cannot fit calibration: it grades the served "
                "(shrunk+blended) probs, but --write-calibration / raw-obs "
                "collection need the raw model prob. Fit on a raw pass, then "
                "validate the staged file on a separate serve-mode pass.")
        if engine != "live":
            raise ValueError("serve_mode requires --engine live (it grades the "
                             "production analyzers' served output).")

    store, source_used = _load_odds_store(sport_key, store_label, source, snapshot)
    print(f"\n[odds source: {source_used} ({snapshot})]")
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

    # MLB additive live-path speed: bulk-prewarm the pitcher_asof SP/RP series into the
    # already-active series_cache (2 queries) so grading's per-pitcher load_sp_series /
    # load_rp_series hit memory instead of ~1-2k remote round-trips. Byte-identical
    # results (same cache the per-entity loaders would fill) — purely a speedup.
    if use_warehouse:
        try:
            import pitcher_asof
            ns, nr = pitcher_asof.prewarm_series_cache(seasons_list)
            if ns or nr:
                print(f"[prewarm] pitcher_asof series cache: {ns} SP + {nr} RP "
                      f"entities (bulk).")
        except Exception:
            pass

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

    # PERF: matchup features are I/O-bound (StatsAPI probables + as-of starter
    # lines) and are the backtest's dominant cost when built per-game inside the
    # serial grading loop. Pre-warm them across a thread pool up front — the
    # sub-fetches share the atomic disk cache, so the grading loop then reads from
    # `prewarm_features` instead of blocking on network. Only the live engine
    # consumes matchup features. (Reuses the build_schedules ThreadPool pattern.)
    prewarm_features = {}
    if engine == "live":
        prewarm_features = _prewarm_matchup_features(
            all_games, schedules, sport_key, use_warehouse)

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
    # --fit-shares: accumulate RAW spread ensemble components across this SAME live
    # write pass so the challenger share + spreads shrink/blend can be fit offline in
    # serve order (see _write_shares_calibration). None unless we're going to fit.
    collect_shares = ({"spreads": []}
                      if (write_calibration and engine == "live" and fit_shares)
                      else None)
    # Date span of matched games — bounds the log supplement to the same window
    # this run graded (so a --season/--limit scoped fit stays coherent).
    graded_lo = graded_hi = None

    matched = 0
    # Diagnostic: how many graded games read matchup features straight from the
    # in-memory prewarm dict vs. fell back to a SERIAL _matchup_features_for
    # recompute (a key miss). A non-trivial miss count is the one thing that makes
    # the live grading phase genuinely slower than the prewarm implies.
    _pw_hits = _pw_miss = 0
    for game in all_games:
        date = game.get("date")
        home, away = game.get("home_team"), game.get("away_team")
        if not (date and home and away):
            continue
        # ET play date for the odds join / feature keys (game["date"] is a UTC
        # timestamp); the full-ISO `date` is kept for prior_games_for's chronology.
        date_et = _et_date10(date)
        entry = _lookup_game_odds(lookup, date_et, home, away,
                                  game_pk=game.get("game_pk"))   # #2b: DH-exact join
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
        d10 = date_et
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
            _mk = (date_et, home, away)
            if _mk in prewarm_features:
                matchup_features = prewarm_features[_mk]
                _pw_hits += 1
            else:
                matchup_features = _matchup_features_for(home, away, date_et,
                                                         sport_key)
                _pw_miss += 1
            # As-of season run-differential for the Pythagorean blend (MLB only;
            # DEFAULT_PYTHAG_WEIGHT). Zero extra queries — summed from the already-
            # loaded schedules through the day before this game (leakage-safe).
            home_sr = (_asof_season_runs(home, schedules, espn_teams, date)
                       if use_warehouse else None)
            away_sr = (_asof_season_runs(away, schedules, espn_teams, date)
                       if use_warehouse else None)
            mwin, mhc, mov, mpyth, mscomp = _live_spread_total_probs(
                entry, home_prior, away_prior, threshold_pct, sport_key,
                matchup_features=matchup_features,
                home_season_runs=home_sr, away_season_runs=away_sr,
                serve_mode=serve_mode)
            # serve-mode: analyzers already applied prob_shrink + blend, so grade the
            # value as-is; raw-mode: apply the single sweepable prob_shrink scalar.
            _gp = (lambda x: x) if serve_mode else (lambda x: _shrink_prob(x, prob_shrink))
            if ml and mwin is not None:
                fair_home, price_home, price_away = ml
                _grade(r["moneyline"], _gp(mwin),
                       fair_home, home_won, price_home, price_away, threshold)
                graded_keys.update((t, "moneyline") for t in ekeys)
                if collect_obs is not None:
                    # RAW model prob (pre-shrink) so the shrink can be swept offline.
                    collect_obs["moneyline"].append(
                        (mwin, fair_home, home_won, price_home, price_away))
                if collect_components is not None:
                    # (recency prob [=mwin when run at pythag=0], raw pythag prob,
                    # market, outcome, prices) for the A(recency) x B(pythag) sweep.
                    collect_components["moneyline"].append(
                        (mwin, mpyth, fair_home, home_won, price_home, price_away))
                if collect_dated is not None:
                    collect_dated["moneyline"].append(
                        (date_et, mwin, fair_home, home_won))
                if collect_bets is not None:
                    # Per-game row for the chronological bankroll sim: RAW home prob
                    # + BOTH prices + fair, so the sim selects the VALUE side (shrunk
                    # prob vs fair) exactly like _team_gate_tally — the bet may be the
                    # dog or a faded favorite, NOT necessarily the favored side.
                    # (date, raw_home_p, fair_home, price_home, price_away, home_won,
                    # n_eff=min prior-games = the interval's effective sample).
                    collect_bets["moneyline"].append((
                        date_et, mwin, fair_home, price_home, price_away, home_won,
                        min(len(home_prior), len(away_prior))))
                # #3 v2 (DIAGNOSTIC): bottom-up lineup-runs P(home win), graded
                # head-to-head vs the recency model (mwin) + market (fair_home).
                if collect_lineup is not None:
                    _lr = analysis.lineup_runs_win_prob(
                        home, away, date_et, int(date_et[:4]), matchup_features)
                    if _lr is not None:
                        collect_lineup["moneyline"].append(
                            (date_et, _lr[0], mwin, fair_home, home_won))
            if sp and mhc is not None:
                home_spread, fair_cover, price_h, price_a = sp
                model_spread, model_cover = mhc
                if (abs(model_spread - home_spread) < 1e-9
                        and abs(actual_margin + home_spread) > 1e-9):
                    home_covers = 1 if (actual_margin + home_spread) > 0 else 0
                    _grade(r["spreads"], _gp(model_cover),
                           fair_cover, home_covers, price_h, price_a, threshold)
                    graded_keys.update((t, "spreads") for t in ekeys)
                    if collect_obs is not None:
                        collect_obs["spreads"].append(
                            (model_cover, fair_cover, home_covers, price_h, price_a))
                    if collect_dated is not None:
                        collect_dated["spreads"].append(
                            (date_et, model_cover, fair_cover, home_covers))
                    if collect_bets is not None:
                        # Portfolio row (home-cover side = the served composite cover;
                        # the sim gates edge>=10% for spreads, no extra shrink).
                        collect_bets["spreads"].append((
                            date_et, model_cover, fair_cover, price_h, price_a,
                            home_covers, min(len(home_prior), len(away_prior))))
                    # RAW spread ensemble components (recency cover pre-shrink +
                    # additive/expected cover + the two point margins + de-vigged
                    # market cover + outcome) so the challenger share, spreads shrink
                    # and spreads blend can be fit offline in serve order. Only when
                    # the expected-runs ensemble fired (mscomp present).
                    if collect_shares is not None and mscomp is not None:
                        collect_shares["spreads"].append((
                            mscomp["current_cover"], mscomp["expected_cover"],
                            fair_cover, home_covers,
                            mscomp["current_margin"], mscomp["expected_margin"],
                            actual_margin))
            if tot and mov is not None:
                line, fair_over, price_o, price_u = tot
                model_line, model_over = mov
                if (abs(model_line - line) < 1e-9
                        and abs(actual_total - line) > 1e-9):
                    over_hit = 1 if actual_total > line else 0
                    _grade(r["totals"], _gp(model_over),
                           fair_over, over_hit, price_o, price_u, threshold)
                    graded_keys.update((t, "totals") for t in ekeys)
                    if collect_obs is not None:
                        collect_obs["totals"].append(
                            (model_over, fair_over, over_hit, price_o, price_u))
                    if collect_dated is not None:
                        collect_dated["totals"].append(
                            (date_et, model_over, fair_over, over_hit))
                    if collect_bets is not None:
                        # Portfolio row (over side = served over-rate; totals abstained
                        # by default policy, collected so the option can be toggled).
                        collect_bets["totals"].append((
                            date_et, model_over, fair_over, price_o, price_u,
                            over_hit, min(len(home_prior), len(away_prior))))
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
    if engine == "live":
        print(f"[grade] matchup features: {_pw_hits} from prewarm, "
              f"{_pw_miss} recomputed serially"
              + ("  <-- SERIAL FALLBACK (key miss) is slowing grading"
                 if _pw_miss else ""))
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
                # --fit-shares OWNS the spreads market end-to-end (shrink+share+blend
                # from raw components), so the generic shrink/blend fitters must skip
                # it to avoid double-counting the analyzer's composite cover.
                _skip = {"spreads"} if fit_shares else set()
                fitted_shrink = _write_shrink_calibration(
                    sport_key, results, extra_obs=extra_obs,
                    min_shrink_n=min_shrink_n, skip_markets=_skip)
                # --fit-blend: also fit + persist the model⇄market blend on the LIVE
                # model, ON TOP of the shrink just fitted (serve order = shrink then
                # blend). One raw pass yields both corrections, correctly sequenced.
                if fit_blend:
                    _write_blend_calibration(
                        sport_key, results, shrink_map=(fitted_shrink or {}),
                        min_n=min_shrink_n, skip_markets=_skip)
                # --fit-shares: fit the MLB spreads ensemble stack (challenger
                # spread_share + spreads shrink + spreads blend) from the raw
                # components captured this pass.
                if fit_shares:
                    _write_shares_calibration(
                        sport_key, (collect_shares or {}).get("spreads", []),
                        min_n=min_shrink_n)
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
        # ROI by market-confidence band (un-shrunk headline gate) — the team analog
        # of the props test: is the edge in the uncertain middle vs efficient extremes?
        _print_confidence_bands(results[variant])


def _write_blend_calibration(sport_key, results, shrink_map=None, min_n=0,
                             skip_markets=()):
    """Write the best model⇄market blend weight per market (from the chosen
    variant) to calibration/<sport>.json so the live analyzers consume it.

    ``shrink_map`` (optional, live path): {market: shrink_factor}. When supplied,
    the model prob in each blend obs is first pulled through that shrink before the
    blend weight is fit — matching the serve order (_apply_shrink THEN blend), so
    the fitted w is optimal ON TOP OF the shrink that's also in the file. Markets
    absent from the map are treated as shrink=1.0 (no shrink). When None, the blend
    is fit on the raw model prob (the convolution-engine path, unchanged).

    ``min_n`` guards each weight: below this graded-obs count the weight is withheld
    so a thin sample can't clobber an established blend (0 = no guard)."""
    from datetime import datetime, timezone
    # Prefer the production-like 'all' variant; else the first available.
    variant = "all" if "all" in results else next(iter(results), None)
    if not variant:
        print("  [write-calibration] No variants to write.")
        return
    blend = {}
    withheld = []
    for market in MARKETS:
        if market in skip_markets:
            continue
        obs = results[variant][market]["blend"]
        if shrink_map is not None:
            s = shrink_map.get(market, 1.0)
            obs = [(_shrink_prob(pm, s), mk, o) for pm, mk, o in obs]
        res = _best_blend_weight(obs)
        if not res:
            continue
        best_w, best_brier, model_brier, market_brier = res
        # Only persist a weight when blending actually helps over the (shrunk)
        # model AND the sample is big enough to trust.
        if best_brier < model_brier - 1e-9 and best_w < 1.0:
            if len(obs) >= min_n:
                blend[market] = {
                    "w": best_w,
                    "n": results[variant][market]["n"],
                    "blend_brier": round(best_brier, 5),
                    "model_brier": round(model_brier, 5),
                    "market_brier": round(market_brier, 5),
                    "on_shrunk": shrink_map is not None,
                }
            else:
                withheld.append((market, best_w, len(obs)))
    if withheld:
        print(f"  [write-calibration] blend withheld (thin sample, need n>={min_n}; "
              "existing weight kept): "
              + ", ".join(f"{m} w={w} n={n}" for m, w, n in withheld))
    if not blend:
        print("  [write-calibration] No market beat the pure model; nothing written.")
        return
    save_market_blend(sport_key, blend, meta={
        "variant": variant,
        "on_shrunk_probs": shrink_map is not None,
        "fit_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    print(f"\n  [write-calibration] Wrote blend weights (variant '{variant}'"
          f"{', on shrunk probs' if shrink_map is not None else ''}) "
          f"to calibration/{active_write_label(sport_key)}:")
    for market, cfg in blend.items():
        print(f"    {market:<10} w={cfg['w']:.2f}  (n={cfg['n']}, "
              f"blendBrier={cfg['blend_brier']})")


def _sweep_min(objective, step=0.05):
    """Return (best_x, best_val) minimizing objective(x) over x in [0,1]."""
    best_x, best_val = 0.0, float("inf")
    x = 0.0
    while x <= 1.0001:
        v = objective(x)
        if v < best_val:
            best_val, best_x = v, x
        x += step
    return round(best_x, 2), best_val


def _write_shares_calibration(sport_key, share_obs, min_n=MIN_SHRINK_N):
    """Fit + persist the FULL MLB spreads ensemble stack for the ADDITIVE model,
    from RAW components captured in the same live --write-calibration pass, in the
    exact serve order (_apply_shrink -> challenger spread_share blend -> market
    blend). This OWNS the spreads market (the generic shrink/blend fitters skip it
    under --fit-shares) because the analyzer's composite model_cover_rate already
    bakes shrink + share in, so a generic fit on it would double-count.

    share_obs: [(current_cover, expected_cover, market_cover|None, covered,
                 current_margin|None, expected_margin|None, actual_margin|None)]
      current_cover  = recency P(home covers), PRE-shrink
      expected_cover = additive/expected-runs P(home covers) (NegBin)
      covered        = 1 if the home spread actually covered

    Writes: prob_shrink[spreads]=s*, expected_runs_challenger share
    {home_minus_1_5: sigma*, margin: m*}, and market_blend[spreads]=w*.
    A single n<min_n gate withholds the WHOLE fit (keeps s*, sigma*, w* mutually
    consistent). Returns True when it wrote, False otherwise."""
    from datetime import datetime, timezone
    obs = list(share_obs or [])
    n = len(obs)
    # max(1, min_n): a min_n of 0 must still block the empty case (the sweeps below
    # divide by n / nm), so we never fit on zero obs.
    if n < max(1, min_n):
        print(f"  [write-calibration] spreads shares withheld (thin sample n={n}, "
              f"need n>={max(1, min_n)}; challenger block kept).")
        return False

    covered = [o[3] for o in obs]
    cc = [o[0] for o in obs]
    ec = [o[1] for o in obs]

    # 1) spreads shrink s* on the RAW recency cover (clean; not the composite).
    sres = _best_shrink([(o[0], None, o[3]) for o in obs])
    best_s, s_brier, s_raw = sres if sres else (1.0, None, None)
    served_s = best_s if (best_s < 1.0 and s_brier is not None
                          and s_brier < s_raw - 1e-9) else 1.0

    # 2) spread_share sigma* on shrink(current_cover, served_s) blended -> expected.
    shrunk_cc = [_shrink_prob(c, served_s) for c in cc]

    def _share_brier(sig):
        return sum((sc + sig * (e - sc) - o) ** 2
                   for sc, e, o in zip(shrunk_cc, ec, covered)) / n
    sigma, sigma_brier = _sweep_min(_share_brier)
    model_cover = [sc + sigma * (e - sc) for sc, e in zip(shrunk_cc, ec)]

    # 3) spreads market blend w* on the sigma-blended cover -> de-vigged market.
    mkt = [(mc, o) for mc, o in ((o[2], o[3]) for o in obs) if mc is not None]
    mc_model = [model_cover[i] for i, o in enumerate(obs) if o[2] is not None]
    best_w, w_brier, w_model_brier = 1.0, None, None
    if len(mkt) >= max(1, min_n):
        nm = len(mkt)

        def _blend_brier(w):
            return sum((w * mcm + (1 - w) * mk[0] - mk[1]) ** 2
                       for mcm, mk in zip(mc_model, mkt)) / nm
        best_w, w_brier = _sweep_min(_blend_brier)
        w_model_brier = sum((mcm - mk[1]) ** 2
                            for mcm, mk in zip(mc_model, mkt)) / nm

    # 4) margin_share m* — DISPLAY ONLY (pred_game_margin); fit to the actual margin.
    marg = [(o[4], o[5], o[6]) for o in obs
            if o[4] is not None and o[5] is not None and o[6] is not None]
    if marg:
        def _margin_sse(m):
            return sum((cm + m * (em - cm) - am) ** 2 for cm, em, am in marg)
        margin_share, _ = _sweep_min(_margin_sse)
    else:
        margin_share = sigma  # no margin data -> mirror the cover share

    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # ── persist ──
    # Persist served_s ALWAYS (even 1.0) so the served spreads shrink is exactly the
    # one sigma* was fit on — a candidate seeded from live could otherwise carry an
    # inherited spreads shrink and break fit==serve. s=1.0 is a serve no-op
    # (_apply_shrink returns p unchanged), so this only pins the invariant.
    save_prob_shrink(sport_key, {"spreads": round(served_s, 2)},
                     holdout={"spreads": {
                         "brier": round(s_brier if served_s < 1.0 else s_raw, 4),
                         "raw_brier": round(s_raw, 4), "n": n}},
                     meta={"source": "odds backtest --fit-shares",
                           "fit_timestamp": ts})
    save_expected_runs_challenger_shares(
        sport_key,
        {"home_minus_1_5": round(sigma, 2), "margin": round(margin_share, 2)},
        meta={"source": "odds backtest --fit-shares", "n": n,
              "shrink": round(served_s, 2), "share_brier": round(sigma_brier, 5),
              "fit_timestamp": ts})
    blend_helped = (w_brier is not None and best_w < 1.0
                    and w_brier < w_model_brier - 1e-9)
    if blend_helped:
        save_market_blend(sport_key, {"spreads": {
            "w": best_w, "n": len(mkt), "blend_brier": round(w_brier, 5),
            "model_brier": round(w_model_brier, 5), "on_shrunk": True,
            "on_shares": True}}, meta={"source": "odds backtest --fit-shares",
                                       "fit_timestamp": ts})
    else:
        # Pin the NO-OP blend (w=1.0) whenever the fit declines a market pull —
        # mirrors the unconditional served_s pin above. Otherwise a candidate
        # seeded from live could carry an INHERITED spreads blend that sigma* was
        # never fit against, breaking fit==serve (adversarial-review finding 1).
        # w=1.0 => serve applies no market pull (blend_w<1.0 gate stays closed).
        save_market_blend(sport_key, {"spreads": {
            "w": 1.0, "n": len(mkt), "on_shrunk": True, "on_shares": True}},
            meta={"source": "odds backtest --fit-shares", "fit_timestamp": ts})

    print(f"\n  [write-calibration] Wrote SPREADS ensemble stack (n={n}) to "
          f"calibration/{active_write_label(sport_key)}:")
    print(f"    shrink   s={served_s:.2f}"
          + (f"  (brier {s_brier:.4f} vs raw {s_raw:.4f})" if served_s < 1.0
             else "  (no shrink helped)"))
    print(f"    share    spread_share={sigma:.2f}  margin_share={margin_share:.2f}"
          f"  (brier {sigma_brier:.4f})")
    if blend_helped:
        print(f"    blend    w={best_w:.2f}  (brier {w_brier:.4f} vs "
              f"model {w_model_brier:.4f}, n={len(mkt)})")
    else:
        print("    blend    w=1.00 pinned (market blend did not beat the ensemble)")
    return True


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


def _props_xba_blend(proj, xstats_strength, rid, game_d10, prior_ab, wts,
                     xba_index):
    """Blend the recency projection toward the batter's LEAKAGE-SAFE as-of xBA ×
    recent AB/game — the fit==serve mirror of book_line_calibration.project_and_
    empirical (batter_hits, method C+xBA). Uses a per-game as-of estimate (< this
    game's date) so a past-dated obs never sees future contact data. Fails OPEN to
    the plain projection (unknown id / no as-of xBA / no AB / any error)."""
    s = xstats_strength or 0.0
    if not (s > 0 and xba_index is not None and rid):
        return proj
    try:
        xba = xba_index.asof_mean(str(rid), game_d10)   # min_bbe gated inside
        ab_valid = [(ab, w) for ab, w in zip(prior_ab, wts) if ab and w > 0]
        if xba is None or not ab_valid:
            return proj
        ab_pg = sum(ab * w for ab, w in ab_valid) / sum(w for _, w in ab_valid)
        if ab_pg > 0:
            s = max(0.0, min(1.0, s))
            return (1.0 - s) * proj + s * (xba * ab_pg)
    except Exception:
        pass
    return proj


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


def run_prop_lag_backtest(sport, espn_sport, espn_league, sport_key,
                          props=("batter_hits", "batter_total_bases"),
                          stale_thr=0.03, min_prior=5):
    """SLOW-PROP-LAG coherence probe. When the GAME total moves early->close (the game
    state shifts) but a batter's prop did NOT follow (|Δprop| small = STALE), bet the
    prop toward the total's move — at the CLOSE price. Within-player DELTAS control for
    batter quality (movements, not levels). Realizable at close (both snapshots seen).
    Reports ROI bucketed by game-move size |Δtotal| for STALE props vs a MOVED-prop
    control: edge = stale props +ROI at big |Δtotal| while moved props aren't."""
    import warehouse as _wh
    import mlb_starters
    import mlb_warehouse
    pe = _wh.load_prop_market_store(sport_key, snapshot="early_4h").get("games", {})
    pc = _wh.load_prop_market_store(sport_key, snapshot="close").get("games", {})
    te = _wh.load_team_market_store(sport_key, snapshot="early_4h").get("games", {})
    tc = _wh.load_team_market_store(sport_key, snapshot="close").get("games", {})
    print("\n=== SLOW-PROP-LAG: game-total move (early->close) vs stale batter props ===")
    if not (pe and pc and te and tc):
        print("  need early+close for BOTH props and team markets.")
        return

    def _tot(entry):
        t = _total_market(entry)
        return t[0] if t else None
    dtotal = {}
    for gk in set(te) & set(tc):
        et, ct = _tot(te[gk]), _tot(tc[gk])
        if et is not None and ct is not None:
            dtotal[gk] = ct - et

    yrs = sorted({str(g.get("commence_time"))[:4] for g in pc.values()
                  if g.get("commence_time")})
    gl_index = {}
    for y in yrs:
        try:
            for pid, logs in (mlb_warehouse.get_calib_gamelogs_bulk(
                    "batter", int(y)) or {}).items():
                gl_index.setdefault(pid, []).extend(logs)
        except Exception:
            pass
    _rid = {}

    def _actual(player, prop, d10):
        if player not in _rid:
            r = mlb_starters.resolve_mlbam_id(
                player, mlb_warehouse._current_season(), prop_key=prop)
            _rid[player] = str(r[0]) if r else ""
        rid = _rid[player]
        gl = gl_index.get(rid) or []
        if not (rid and gl):
            return None
        label = _stat_label_for(prop, gl)
        if not label:
            return None
        for g in gl:
            gd = g.get("game_date")
            if (gd and gd[:10] == d10 and g.get(label) is not None
                    and g.get("completed") is not False):
                return float(g[label])
        return None

    obs = []   # (Δtotal, Δprop, over_dec, under_dec, over_outcome)
    for gk, gce in pc.items():
        if gk not in pe or gk not in dtotal:
            continue
        dtot = dtotal[gk]
        d10 = str(gce.get("commence_time"))[:10]
        em = pe[gk].get("props") or {}
        for prop in props:
            cm = (gce.get("props") or {}).get(prop) or {}
            pem = em.get(prop) or {}
            for player, ci in cm.items():
                ei = pem.get(player)
                if not ei:
                    continue
                if (ci.get("over_implied") is None or ci.get("under_implied") is None
                        or ei.get("over_implied") is None
                        or ei.get("under_implied") is None
                        or ci.get("line") is None or ci.get("over_price") is None
                        or ci.get("under_price") is None):
                    continue
                co = devig_two_way(ci["over_implied"], ci["under_implied"])[0]
                eo = devig_two_way(ei["over_implied"], ei["under_implied"])[0]
                actual = _actual(player, prop, d10)
                if actual is None or abs(actual - ci["line"]) < 1e-9:
                    continue
                obs.append((dtot, co - eo,
                            american_to_decimal(ci["over_price"]),
                            american_to_decimal(ci["under_price"]),
                            1 if actual > ci["line"] else 0))

    if len(obs) < 200:
        print(f"  only {len(obs)} paired early/close prop obs — too few.")
        return
    print(f"  {len(obs)} paired obs; STALE = |Δprop|<={stale_thr}. Bet toward the total "
          "move (total up -> over) at the close price.")
    print(f"  {'|Δtotal|>=':>9} {'group':>6} {'bets':>6} {'win%':>6} {'ROI%':>8} "
          f"{'P/L(u)':>9}")
    for thr in (0.5, 1.0, 1.5):
        for grp, cond in (("stale", lambda dp: abs(dp) <= stale_thr),
                          ("moved", lambda dp: abs(dp) > stale_thr)):
            sel = []
            for dtot, dprop, do, du, o in obs:
                if abs(dtot) < thr or dtot == 0 or not cond(dprop):
                    continue
                sel.append((do, o == 1) if dtot > 0 else (du, o == 0))
            if not sel:
                print(f"  {thr:>9.1f} {grp:>6} {0:>6}")
                continue
            n = len(sel)
            wins = sum(1 for _, w in sel if w)
            pl = sum((dec - 1.0) if w else -1.0 for dec, w in sel)
            print(f"  {thr:>9.1f} {grp:>6} {n:>6} {100.0 * wins / n:>6.1f} "
                  f"{100.0 * pl / n:>8.2f} {pl:>9.2f}")
    print("  edge = STALE props +ROI at big |Δtotal| while MOVED props aren't "
          "(the lag is real + exploitable). Both flat/negative = no lag edge.")


def run_coherence_backtest(sport_key, espn_sport, espn_league, season_year=None,
                           store_label="", source="auto", snapshot="close",
                           limit=100000, train_frac=0.6):
    """Cross-market COHERENCE probe — team triad (ML / run-line / total).

    Tests whether the book's OWN run-line disagrees with what its moneyline + total
    imply, and whether that outlier is exploitable — WITHOUT out-predicting DK. We
    only use the book's other two markets as the 'truth':
      1. de-vig each game -> favorite win prob (ML), total line, favorite RL-cover.
      2. learn the book's TYPICAL favorite RL-cover as f(fav_win, total) on a TRAIN
         split (empirical baseline — the book's own coherent relationship; no scoring
         model, so a deviation is a genuine self-contradiction not our modeling error).
      3. on the OOS TEST split, bet the games whose RL deviates most from that norm
         (r>0 book overprices fav cover -> bet the +1.5 dog; r<0 -> bet the -1.5 fav),
         graded vs the actual margin at the RAW RL price. ROI by |residual| bucket.
    +ROI concentrating at larger |residual| = internal contradictions are exploitable;
    flat/near-zero = DK prices the triad coherently (expected — it's the tight one)."""
    store, source_used = _load_odds_store(sport_key, store_label, source, snapshot)
    print(f"\n[odds source: {source_used} ({snapshot})]  "
          f"=== COHERENCE: team triad (ML / run-line / total) ===")
    if not store.get("games"):
        print("  No team-market store for this sport/source.")
        return
    use_warehouse = espn_sport == "baseball"
    seasons_list = (list(season_year)
                    if isinstance(season_year, (list, tuple, set)) else [season_year])
    if use_warehouse:
        espn_teams, schedules = _warehouse_team_schedules(seasons_list)
    else:
        espn_teams = get_all_teams(espn_sport, espn_league)
        schedules = {}
        for sy in seasons_list:
            for tid, gms in build_schedules(
                    espn_sport, espn_league, espn_teams, season_year=sy).items():
                schedules.setdefault(tid, []).extend(gms)
    lookup, _unm = _build_odds_lookup(store, espn_teams)
    all_games = all_completed_games(schedules)
    if limit and limit < len(all_games):
        all_games = all_games[-limit:]

    # (date10, fav_win_prob, total, book_fav_cover, fav_price, dog_price, fav_covered)
    obs = []
    for game in all_games:
        date, home, away = (game.get("date"), game.get("home_team"),
                            game.get("away_team"))
        if not (date and home and away):
            continue
        entry = _lookup_game_odds(lookup, _et_date10(date), home, away,
                                  game_pk=game.get("game_pk"))
        if not entry:
            continue
        ml, sp, tot = (_moneyline_market(entry), _spread_market(entry),
                       _total_market(entry))
        if not (ml and sp and tot):
            continue
        p_home = ml[0]
        h_spread, fair_home_cover, sp_h_price, sp_a_price = sp
        total_line = tot[0]
        if h_spread is None or abs(abs(h_spread) - 1.5) > 1e-6:
            continue  # only the standard ±1.5 run line
        margin = game["home_score"] - game["away_score"]
        if h_spread < 0:                      # home is the -1.5 favorite
            fav_win, c_fav = p_home, fair_home_cover
            fav_price, dog_price = sp_h_price, sp_a_price
            fav_covered = 1 if margin >= 2 else 0
        else:                                 # away is the -1.5 favorite
            fav_win, c_fav = 1.0 - p_home, 1.0 - fair_home_cover
            fav_price, dog_price = sp_a_price, sp_h_price
            fav_covered = 1 if margin <= -2 else 0
        if None in (fav_win, c_fav, total_line, fav_price, dog_price):
            continue
        obs.append((_et_date10(date), fav_win, total_line, c_fav,
                    fav_price, dog_price, fav_covered))

    if len(obs) < 200:
        print(f"  only {len(obs)} usable triads — too few to test.")
        return
    obs.sort(key=lambda x: x[0])
    dates = sorted({o[0] for o in obs})
    cut = dates[int(len(dates) * train_frac)]
    train = [o for o in obs if o[0] < cut]
    test = [o for o in obs if o[0] >= cut]

    def _wbin(p):
        return round(min(max(p, 0.0), 1.0) * 20) / 20.0   # 5-point win-prob bin
    def _tbin(t):
        return round(t * 2) / 2.0                          # 0.5-run total bin

    from collections import defaultdict
    agg = defaultdict(lambda: [0.0, 0])
    for _d, fw, T, c, _fp, _dp, _cov in train:
        k = (_wbin(fw), _tbin(T))
        agg[k][0] += c
        agg[k][1] += 1
    base = {k: s / n for k, (s, n) in agg.items() if n >= 20}
    gbase = (sum(o[3] for o in train) / len(train)) if train else 0.5

    graded = []   # (|resid|, decimal_odds, won)
    for _d, fw, T, c, fp, dp, cov in test:
        chat = base.get((_wbin(fw), _tbin(T)), gbase)
        r = c - chat
        if r >= 0:            # book overprices fav cover -> bet the +1.5 dog
            graded.append((abs(r), american_to_decimal(dp), cov == 0))
        else:                 # book underprices fav cover -> bet the -1.5 fav
            graded.append((abs(r), american_to_decimal(fp), cov == 1))

    print(f"  triads: {len(obs)} usable ({len(train)} train / {len(test)} test OOS, "
          f"cut {cut}); baseline = book fav-cover ~ f(fav_win, total).")
    print(f"  {'|resid|>=':>9} {'bets':>7} {'win%':>6} {'ROI%':>8} {'P/L(u)':>9}")
    for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        sel = [(dec, won) for r, dec, won in graded if r >= thr]
        if not sel:
            print(f"  {thr:>9.2f} {0:>7}")
            continue
        n = len(sel)
        wins = sum(1 for _, w in sel if w)
        pl = sum((dec - 1.0) if w else -1.0 for dec, w in sel)
        print(f"  {thr:>9.2f} {n:>7} {100.0 * wins / n:>6.1f} "
              f"{100.0 * pl / n:>8.2f} {pl:>9.2f}")
    print("  +ROI growing with |resid| = the book's ML/RL/total contradictions are "
          "exploitable; flat/negative = DK prices the triad coherently (as expected).")


def run_props_odds_backtest(sport, espn_sport, espn_league, sport_key, props,
                            min_prior=5, half_life=None, threshold_pct=5.0,
                            store_label="", write_calibration=False,
                            source="auto", snapshot="close", seasons=None,
                            xstats_override=None, walk_forward=False):
    """
    Grade the model's player-prop value flags against stored historical closing
    lines (from backfill_historical_odds.py --props ...). For each captured
    book line we recompute the model's P(over) from the player's prior games,
    compare it to the de-vigged closing line, and measure ROI + model-vs-market
    Brier + the optimal model⇄market blend, per prop market.

    fit==serve for MLB (P-fixes): (a) player logs come from a MULTI-SEASON warehouse
    index (was current-season-only get_calib_gamelog → could only grade the live
    season); (b) batter_hits applies the SAME leakage-safe xBA blend the promoted
    C+xBA calibration was fit under (_props_xba_blend). ``seasons`` (list of ints)
    filters the graded games to those years (per-season durability read); None =
    all seasons pooled.
    """
    threshold = threshold_pct / 100.0
    # P3c/P4/P6: MLB player gamelogs come from the StatsAPI warehouse; NBA/NFL stay on
    # ESPN. (ESPN was fully removed for MLB in P4.)
    use_warehouse = espn_sport == "baseball"
    if use_warehouse:
        print("=== props-odds player logs: StatsAPI warehouse (ESPN bypassed) ===")
    store, source_used = _load_prop_store(sport_key, store_label, source, snapshot)
    print(f"\n[odds source: {source_used} ({snapshot})]")
    games = store.get("games", {})
    if not games:
        cli = _SPORT_KEY_TO_CLI.get(sport_key, sport_key)
        lbl = f" --label {store_label}" if store_label else ""
        if source_used == "warehouse":
            print(f"\nNo warehoused player-prop lines for {sport_key} at "
                  f"snapshot={snapshot}. Backfill props with "
                  f"backfill_historical_odds.py --sport {cli} --props ...")
        else:
            print(f"\nNo historical odds stored for {sport_key}"
                  f"{f' (label={store_label})' if store_label else ''}. Run "
                  f"backfill_historical_odds.py --sport {cli} "
                  f"--props {','.join(props)}{lbl} ...")
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
    import props as _props_recal   # for the LIVE line-conditional Platt resolver
    recal = load_recalibration(sport_key) or {}
    hl = half_life or _half_life_for(sport_key)
    results = {prop: _empty_market_bucket() for prop in props}
    series_cache = {}
    no_actual = {prop: 0 for prop in props}
    no_series = {prop: 0 for prop in props}
    season_set = {str(s) for s in seasons} if seasons else None

    # ── MLB: season-aware multi-season gamelog index (fixes the old current-season-
    # only priors) — one bulk query per (role, season), every player from memory.
    # Spans the obs years + the season before the earliest (cross-season warmup).
    # Honors the --seasons filter so a per-season run loads only that year's data
    # (its priors still reach back a season via the gamelog index's -1 span).
    _obs_years = sorted({int(str(e.get("commence_time"))[:4])
                         for e in games.values() if e.get("commence_time")
                         and (season_set is None
                              or str(e.get("commence_time"))[:4] in season_set)})
    _gl_index = {}   # role -> {mlb_id(str): [gamelog dict, ...]} (all seasons merged)

    def _gl_for_role(role):
        if role not in _gl_index:
            merged = {}
            try:
                import mlb_warehouse as _mw
                if _obs_years:
                    for _s in range(_obs_years[0] - 1, _obs_years[-1] + 1):
                        for pid, logs in (
                                _mw.get_calib_gamelogs_bulk(role, _s) or {}).items():
                            merged.setdefault(pid, []).extend(logs)
            except Exception:
                merged = {}
            _gl_index[role] = merged
        return _gl_index[role]

    # ── Leakage-safe as-of xBA index for the batter_hits xBA blend (fit==serve with
    # the promoted C+xBA fit; mirrors the real-line refit). Built once over the obs
    # years; None => no blend. Only built when a served prop actually carries xstats.
    # Effective xBA strength per prop: --xstats-strength overrides the calib value
    # (only for props that ALREADY carry xstats, i.e. batter_hits) so we can A/B the
    # xBA blend ON vs OFF against real odds without touching the calibration file.
    def _eff_xs(p):
        cx = (calibration.get(p) or {}).get("xstats_strength")
        return xstats_override if (xstats_override is not None and cx) else cx

    xba_index = None
    if espn_sport == "baseball" and _obs_years and any(
            (_eff_xs(p) or 0) > 0 for p in props):
        try:
            import savant_history as _sh
            import backtest_props as _bp
            _raw = []
            for _y in _obs_years:
                try:
                    _raw.extend(_sh.load_days(f"{_y}-03-01", f"{_y}-11-30"))
                except Exception:
                    pass
            if _raw:
                xba_index = _bp.build_batter_xba_index(_raw)
                print(f"  [xstats] built leakage-safe xBA index from {len(_raw):,} "
                      f"pitches ({', '.join(str(y) for y in _obs_years)})")
        except Exception:
            xba_index = None

    def series(player, prop):
        # NBA/NFL (ESPN) path — unchanged, current-season gamelog. (date, val).
        k = (player, prop)
        if k not in series_cache:
            series_cache[k] = _player_stat_series(
                espn_sport, espn_league, player, prop)
        return series_cache[k]

    _mlb_ser_cache = {}

    def _mlb_series(player, prop):
        """(rid, [(date, val, ab), ...]) from the multi-season index — season-aware
        + carries AB per game for the xBA blend. rid is the resolved MLBAM id."""
        k = (player, prop)
        if k in _mlb_ser_cache:
            return _mlb_ser_cache[k]
        import mlb_starters
        import mlb_warehouse
        resolved = mlb_starters.resolve_mlbam_id(
            player, mlb_warehouse._current_season(), prop_key=prop)
        if not resolved:
            _mlb_ser_cache[k] = (None, [])
            return _mlb_ser_cache[k]
        rid = str(resolved[0])
        role = "pitcher" if str(prop).startswith("pitcher_") else "batter"
        gl = _gl_for_role(role).get(rid) or []
        if not (_role_matches_gamelog(prop, gl) and gl):
            _mlb_ser_cache[k] = (rid, [])
            return _mlb_ser_cache[k]
        label = _stat_label_for(prop, gl)
        if not label:
            _mlb_ser_cache[k] = (rid, [])
            return _mlb_ser_cache[k]
        rows = []
        for g in gl:
            if g.get("completed") is False:
                continue
            d, v = g.get("game_date"), g.get(label)
            if not d or v is None:
                continue
            if prop == "pitcher_outs":
                v = ip_to_outs(v)
            rows.append((d, float(v), g.get("AB")))
        rows.sort(key=lambda x: x[0])
        _mlb_ser_cache[k] = (rid, rows)
        return _mlb_ser_cache[k]

    print(f"\n=== Props odds backtest: {sport_key} {props} ===")
    print(f"Stored games: {len(games)} (bookmaker: {store.get('bookmaker','?')})")

    for entry in games.values():
        gdate = entry.get("commence_time")
        if not gdate:
            continue
        d10 = gdate[:10]
        if season_set is not None and d10[:4] not in season_set:
            continue
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
                if espn_sport == "baseball":
                    rid, ser = _mlb_series(player, prop)   # season-aware + AB
                else:
                    rid = None
                    ser = [(d, v, None) for d, v in series(player, prop)]
                if not ser:
                    no_series[prop] += 1
                    continue
                actual = None
                prior = []
                prior_ab = []
                for dt, val, ab in ser:
                    if dt[:10] == d10:
                        actual = val
                    elif dt < gdate:
                        prior.append(val)
                        prior_ab.append(ab)
                if actual is None or len(prior) < min_prior:
                    no_actual[prop] += 1
                    continue
                if abs(actual - line) < 1e-9:
                    continue  # push — refund
                wts = _recency_weights(len(prior), hl)
                raw_mean = _weighted_mean(prior, wts)
                # Leakage-safe per-player OUTCOME volatility: recency-weighted CV
                # (sigma/mean) of the prior game-to-game stat — a 'how consistent is
                # this player' certainty signal, distinct from sample-size (n)
                # uncertainty. Computed on the RAW mean (pre-xBA-blend) so it reflects
                # realized game-to-game spread, not the projection.
                _sw = sum(wts) or 1.0
                _var = sum(w * (v - raw_mean) ** 2 for v, w in zip(prior, wts)) / _sw
                cv = (_var ** 0.5 / raw_mean) if raw_mean > 1e-9 else 0.0
                # fit==serve: apply the promoted batter_hits xBA blend (leakage-safe
                # as-of). No-op for any prop without a served xstats_strength.
                proj = _props_xba_blend(
                    raw_mean, _eff_xs(prop), rid, d10, prior_ab, wts, xba_index)
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
                # Served prob = the LIVE line-conditional Platt recal (fit==serve),
                # so the served-EV + Kelly diagnostics below grade exactly what the app
                # ships — not the raw pre-Platt prob. (The headline table above keeps
                # the pre-Platt prob for backtest continuity.)
                rcfg = _props_recal._resolve_recal_cfg(
                    recal, prop, line, calibration.get(prop))
                p_final = p_model
                if rcfg and rcfg.get("a") is not None:
                    adj = apply_platt(p_model, rcfg.get("a"), rcfg.get("b"))
                    if adj is not None:
                        p_final = max(0.0, min(1.0, adj))
                results[prop]["prior_k"].append(
                    (p_final, fair_over, outcome, len(prior)))
                results[prop]["ev_obs"].append(
                    (d10, p_final, fair_over, outcome, over_price, under_price,
                     len(prior), cv))
                results[prop]["wf_obs"].append(
                    (d10, p_model, line, outcome, over_price, under_price))

    # Diagnostics on coverage
    print("\nCoverage (why lines were dropped):")
    for prop in props:
        print(f"  {prop:<18} graded={results[prop]['n']:>5}  "
              f"no_dated_series={no_series[prop]:>5}  "
              f"no_actual/min_prior={no_actual[prop]:>5}")

    _print_props_odds_results(results, threshold_pct)

    if walk_forward:
        _simulate_walk_forward(results, recal, calibration)

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

    _print_confidence_bands(results)
    _print_props_served_ev_staking(results)
    _print_props_ev_by_sample_size(results)
    _print_props_ev_by_variance(results)
    _print_props_cv_ev_matrix(results)
    _simulate_daily_topn(results)


_CONF_BANDS = [
    ("dog <40%", 0.0, 0.40),
    ("40-47.5%", 0.40, 0.475),
    ("pickem 47.5-52.5%", 0.475, 0.525),
    ("52.5-60%", 0.525, 0.60),
    ("fav 60-70%", 0.60, 0.70),
    ("heavy >=70%", 0.70, 1.01),
]


def _print_confidence_bands(results):
    """Realized win%/ROI of PLACED bets bucketed by the SELECTED side's de-vigged
    MARKET probability (= market confidence). Tests the hypothesis that the model's
    edge, if any, lives in the UNCERTAIN middle (near-pickem, where the book is
    laziest) and NOT on efficient extremes (heavy favorites) or the longshot trap.
    Positive ROI concentrated in the pickem/mid bands = a price-zone to gate on."""
    print("\n=== ROI by MARKET-confidence band (selected side's de-vigged market prob) ===")
    print("  hypothesis: edge concentrates near PICKEM (book least sure); extremes are "
          "efficient (favorites) or public traps (longshots).")
    for prop, r in results.items():
        detail = r.get("bets_detail") or []
        if not detail:
            continue
        print(f"\n  {prop}:")
        print(f"  {'band':<20} {'bets':>6} {'win%':>6} {'ROI%':>8} {'P/L(u)':>9}")
        for label, lo, hi in _CONF_BANDS:
            band = [(dec, won) for mp, dec, won in detail if lo <= mp < hi]
            if not band:
                continue
            n = len(band)
            wins = sum(1 for _, won in band if won)
            pl = sum((dec - 1.0) if won else -1.0 for dec, won in band)
            print(f"  {label:<20} {n:>6} {100.0 * wins / n:>6.1f} "
                  f"{100.0 * pl / n:>8.2f} {pl:>9.2f}")


_EV_GATES = (0.0, 0.02, 0.05, 0.08, 0.12)


def _print_props_served_ev_staking(results):
    """fit==serve diagnostics on the SERVED (line-conditional Platt) prob — what the
    app actually ships (not the raw pre-Platt prob the headline table grades):
      (1) served Brier vs market Brier (calibrated accuracy),
      (2) an EV-gate sweep — side = max-EV, profit at the RAW price (vig included),
      (3) flat vs fractional-Kelly staking on the EV>=5% bets (does Kelly help or
          just amplify drawdown? — teams showed flat wins).
    A positive ROI band with served Brier <= market = a real, capturable edge."""
    print("\n=== SERVED-PROB (Platt) EV-gate sweep + staking [fit==serve] ===")
    print("  graded on the LIVE served prob (line-conditional Platt); side = max-EV;")
    print("  profit at the RAW price. This is what the app would actually bet.")
    for prop, r in results.items():
        obs = r.get("ev_obs") or []
        if not obs:
            continue
        n = len(obs)
        sb = sum((p - o) ** 2 for _, p, _, o, *_ in obs) / n
        mb = sum((f - o) ** 2 for _, _, f, o, *_ in obs) / n
        picks = []   # (date, decimal_odds, won, ev) for the max-EV side
        for _d, p, _f, o, op, up, *_ in obs:
            if op is None or up is None:
                continue
            do, du = american_to_decimal(op), american_to_decimal(up)
            ev_o, ev_u = p * do - 1.0, (1.0 - p) * du - 1.0
            if ev_o >= ev_u:
                picks.append((_d, do, o == 1, ev_o))
            else:
                picks.append((_d, du, o == 0, ev_u))
        flag = "served BEATS market" if sb < mb else "market still ahead"
        print(f"\n  {prop}: served Brier {sb:.4f} vs market {mb:.4f}  "
              f"({flag}; n={n})")
        print(f"  {'EV gate':>7} {'bets':>7} {'win%':>6} {'ROI%':>8} {'P/L(u)':>9}")
        for g in _EV_GATES:
            sel = [x for x in picks if x[3] >= g]
            if not sel:
                print(f"  {g * 100:>5.0f}%  {0:>7}")
                continue
            m = len(sel)
            wins = sum(1 for _, _, w, _ in sel if w)
            pl = sum((dec - 1.0) if w else -1.0 for _, dec, w, _ in sel)
            print(f"  {g * 100:>5.0f}%  {m:>7} {100.0 * wins / m:>6.1f} "
                  f"{100.0 * pl / m:>8.2f} {pl:>9.2f}")


_N_BANDS = [
    ("<10 gm", 0, 10),
    ("10-19", 10, 20),
    ("20-29", 20, 30),
    ("30-44", 30, 45),
    ("45+", 45, 10 ** 9),
]
# Cumulative min-n abstention gate: ROI if you only bet players with >= N prior games.
_N_GATES = (0, 10, 15, 20, 25, 30)


def _print_props_ev_by_sample_size(results, ev_gate=0.0):
    """Realized win%/ROI of the served (Platt) +EV picks bucketed by the BETTOR's
    prior game-count n — the axis we'd never sliced before.

    Tests the abstention idea: are the losses concentrated in LOW-n players (whose
    projection is dominated by the league prior via shrinkage_k, so it barely
    deviates from the book) — in which case a hard min-n gate trims them — or is
    ROI uniformly negative across n (the model just has no edge at any sample size)?
    Also sweeps a CUMULATIVE min-n gate: ROI if you require >= N prior games, i.e.
    'don't bet low-n batters' made explicit. Picks = the max-EV side at ev_gate,
    matching what the app would actually place."""
    print("\n=== ROI by BETTOR sample-size (prior game-count n) [served +EV picks] ===")
    print("  hypothesis: low-n projections shrink to league (~the market line) and are")
    print("  the losers; a min-n abstention gate should trim them. Uniform ROI across n")
    print("  = no edge at any sample size (shrink vs. abstain is moot).")
    for prop, r in results.items():
        picks = []   # (n, decimal_odds, won) for the +EV max-EV side of each obs
        for _d, p, _f, o, op, up, pn, *_ in (r.get("ev_obs") or []):
            if op is None or up is None:
                continue
            do, du = american_to_decimal(op), american_to_decimal(up)
            ev_o, ev_u = p * do - 1.0, (1.0 - p) * du - 1.0
            if ev_o >= ev_u:
                if ev_o >= ev_gate:
                    picks.append((pn, do, o == 1))
            elif ev_u >= ev_gate:
                picks.append((pn, du, o == 0))
        if not picks:
            continue
        print(f"\n  {prop} (EV gate {ev_gate * 100:.0f}%, n={len(picks)}):")
        print(f"  {'n band':<10} {'bets':>6} {'win%':>6} {'ROI%':>8} {'P/L(u)':>9}")
        for label, lo, hi in _N_BANDS:
            band = [(dec, won) for pn, dec, won in picks if lo <= pn < hi]
            if not band:
                continue
            m = len(band)
            wins = sum(1 for _, won in band if won)
            pl = sum((dec - 1.0) if won else -1.0 for dec, won in band)
            print(f"  {label:<10} {m:>6} {100.0 * wins / m:>6.1f} "
                  f"{100.0 * pl / m:>8.2f} {pl:>9.2f}")
        print(f"  {'cum min-n':<10} {'bets':>6} {'win%':>6} {'ROI%':>8} {'P/L(u)':>9}")
        for gate in _N_GATES:
            band = [(dec, won) for pn, dec, won in picks if pn >= gate]
            if not band:
                continue
            m = len(band)
            wins = sum(1 for _, won in band if won)
            pl = sum((dec - 1.0) if won else -1.0 for dec, won in band)
            print(f"  {'>=' + str(gate):<10} {m:>6} {100.0 * wins / m:>6.1f} "
                  f"{100.0 * pl / m:>8.2f} {pl:>9.2f}")


_CV_BANDS = [
    ("steady <0.6", 0.0, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.0),
    ("1.0-1.3", 1.0, 1.3),
    ("volatile 1.3+", 1.3, 10 ** 9),
]
# Cumulative max-CV gate: ROI if you only bet players calmer than a CV ceiling.
_CV_GATES = (10 ** 9, 1.3, 1.0, 0.8, 0.6)


def _print_props_ev_by_variance(results, ev_gate=0.0):
    """Realized win%/ROI of the served (Platt) +EV picks bucketed by the player's
    OUTCOME volatility — recency-weighted CV (sigma/mean) of his prior games.

    This is the OTHER uncertainty axis (distinct from sample-size n): even with a
    reliable mean estimate, a boom/bust hitter's game outcome is dominated by
    variance, so a mean-based edge is least trustworthy there. Tests whether ROI
    concentrates on STEADY (low-CV) players — in which case a max-CV abstention gate
    trims the volatile ones — vs. uniform loss (the book already prices volatility,
    so variance carries no selection signal). Also sweeps a cumulative max-CV gate:
    ROI if you only bet players below a CV ceiling. Picks = the max-EV side at
    ev_gate, matching what the app would place."""
    print("\n=== ROI by PLAYER volatility (recency-weighted CV) [served +EV picks] ===")
    print("  hypothesis: mean-based edges are most trustworthy on STEADY (low-CV)")
    print("  players; volatile players are variance-dominated noise. Uniform ROI across")
    print("  CV = the book already prices volatility (no selection signal here).")
    for prop, r in results.items():
        picks = []   # (cv, decimal_odds, won) for the +EV max-EV side of each obs
        for _d, p, _f, o, op, up, _pn, pcv in (r.get("ev_obs") or []):
            if op is None or up is None:
                continue
            do, du = american_to_decimal(op), american_to_decimal(up)
            ev_o, ev_u = p * do - 1.0, (1.0 - p) * du - 1.0
            if ev_o >= ev_u:
                if ev_o >= ev_gate:
                    picks.append((pcv, do, o == 1))
            elif ev_u >= ev_gate:
                picks.append((pcv, du, o == 0))
        if not picks:
            continue
        print(f"\n  {prop} (EV gate {ev_gate * 100:.0f}%, n={len(picks)}):")
        print(f"  {'CV band':<14} {'bets':>6} {'win%':>6} {'ROI%':>8} {'P/L(u)':>9}")
        for label, lo, hi in _CV_BANDS:
            band = [(dec, won) for pcv, dec, won in picks if lo <= pcv < hi]
            if not band:
                continue
            m = len(band)
            wins = sum(1 for _, won in band if won)
            pl = sum((dec - 1.0) if won else -1.0 for dec, won in band)
            print(f"  {label:<14} {m:>6} {100.0 * wins / m:>6.1f} "
                  f"{100.0 * pl / m:>8.2f} {pl:>9.2f}")
        print(f"  {'cum max-CV':<14} {'bets':>6} {'win%':>6} {'ROI%':>8} {'P/L(u)':>9}")
        for gate in _CV_GATES:
            band = [(dec, won) for pcv, dec, won in picks if pcv <= gate]
            if not band:
                continue
            m = len(band)
            wins = sum(1 for _, won in band if won)
            pl = sum((dec - 1.0) if won else -1.0 for dec, won in band)
            lbl = "all" if gate >= 10 ** 8 else f"<={gate:g}"
            print(f"  {lbl:<14} {m:>6} {100.0 * wins / m:>6.1f} "
                  f"{100.0 * pl / m:>8.2f} {pl:>9.2f}")


# Deployable-gate lens for the validated earned_runs high-CV variance edge: a
# CUMULATIVE CV-FLOOR (bet only pitchers with recency-weighted CV >= floor — the
# edge is in HIGH CV, so this is a min-gate, not the max-gate _CV_GATES above)
# crossed with the live EV floor. The validated grid is at EV>=0; production also
# applies value_gate.ev_floor (0.04), so this confirms whether CV>=1.3 survives
# being stacked with the 4% EV gate BEFORE the serving gate is wired in.
_CV_FLOORS = (0.0, 1.0, 1.3)
_CVEV_GATES = (0.0, 0.02, 0.04, 0.05, 0.08)


def _print_props_cv_ev_matrix(results):
    """CV-floor x EV-floor ROI matrix on the served (+EV, max-EV side) picks — the
    deployable-gate lens. Rows require the pitcher's recency-weighted CV >= floor
    (edge = HIGH CV); columns require the selected side's EV >= gate (production's
    value_gate.ev_floor = 0.04). Each cell = ROI% / n. '-' = no bets in that cell.
    The decision cell for the earned_runs edge is [CV>=1.3, EV>=4%]: if it stays
    solidly +ROI with usable volume, the CV>=1.3 gate survives the live EV floor.

    NOTE: gates on EV only (value_gate's edge_floor=0.01 rarely binds); side and EV
    are computed exactly as _print_props_served_ev_staking, so this is fit==serve."""
    print("\n=== CV-floor x EV-floor ROI matrix [served +EV picks; deployable-gate lens] ===")
    print("  rows require CV >= floor (edge = HIGH CV); cols require side EV >= gate")
    print("  (production value_gate.ev_floor = 0.04). cell = ROI%/n. '-' = no bets.")
    for prop, r in results.items():
        picks = []   # (cv, ev, decimal_odds, won) for the max-EV side of each obs
        for _d, p, _f, o, op, up, _pn, pcv in (r.get("ev_obs") or []):
            if op is None or up is None:
                continue
            do, du = american_to_decimal(op), american_to_decimal(up)
            ev_o, ev_u = p * do - 1.0, (1.0 - p) * du - 1.0
            if ev_o >= ev_u:
                picks.append((pcv, ev_o, do, o == 1))
            else:
                picks.append((pcv, ev_u, du, o == 0))
        if not picks:
            continue
        print(f"\n  {prop}:")
        print("  " + f"{'CV floor':<9}"
              + "".join(f"{'EV>=' + f'{g * 100:.0f}%':>14}" for g in _CVEV_GATES))
        for cvf in _CV_FLOORS:
            row = "  " + f"{'>=' + f'{cvf:g}':<9}"
            for g in _CVEV_GATES:
                sel = [(dec, won) for pcv, ev, dec, won in picks
                       if pcv >= cvf and ev >= g]
                if not sel:
                    row += f"{'-':>14}"
                    continue
                m = len(sel)
                pl = sum((dec - 1.0) if won else -1.0 for dec, won in sel)
                row += f"{f'{100.0 * pl / m:+.1f}%/{m}':>14}"
            print(row)


def _conf_bucket(p):
    """5-point confidence bucket of the SELECTED side's prob on [0.5, 1.0)."""
    return round(min(max(p, 0.5), 0.9989) * 20) / 20.0


def _simulate_daily_topn(results, n_per_day=10, train_frac=0.6,
                         kelly_fracs=(0.125, 0.25, 0.5), kelly_cap=0.10):
    """REALISTIC daily portfolio sim — mirrors how the app is actually used
    (bet_selector.select_top_bets): each day, take the top-N +EV props across ALL
    markets and stake them on a chronological bankroll (flat + fractional Kelly).

    Two selection/sizing modes, both simulated ONLY on the OOS test split:
      RAW   — rank + size on the served (Platt) prob (what the app does today).
      CALIB — rank + size on an OOS per-(prop, confidence-bucket) recalibrated prob:
              fit each bucket's empirical win-rate on the TRAIN split, apply on TEST.
              This is Doug's 'adjust by confidence bucket' — overconfident buckets get
              their prob (=> EV => Kelly stake) marked down; well-calibrated ones keep
              it. Buckets with too little train data are skipped (conservative).
    Kelly fraction per bet = frac * (EV / net_odds), capped at kelly_cap of bankroll.
    ⚠ Bets are sized independently (no same-day correlation haircut) — a real slate
    of 10 correlated legs is riskier than this shows; treat Kelly growth as optimistic."""
    from collections import defaultdict
    pool = []   # (date10, prop, sel_prob, decimal_odds, won, served_ev)
    for prop, r in results.items():
        for d, p, _f, o, op, up, *_ in (r.get("ev_obs") or []):
            if not d or op is None or up is None:
                continue
            do, du = american_to_decimal(op), american_to_decimal(up)
            ev_o, ev_u = p * do - 1.0, (1.0 - p) * du - 1.0
            if ev_o >= ev_u:
                pool.append((d[:10], prop, p, do, o == 1, ev_o))
            else:
                pool.append((d[:10], prop, 1.0 - p, du, o == 0, ev_u))
    if not pool:
        return
    dates = sorted({x[0] for x in pool})
    if len(dates) < 20:
        print("\n=== REALISTIC daily top-N portfolio: too few dates to split. ===")
        return
    cut = dates[int(len(dates) * train_frac)]
    test_days = [d for d in dates if d >= cut]

    # OOS calibration map: per-(prop, bucket) empirical win-rate on TRAIN (n>=50).
    agg = defaultdict(lambda: [0, 0])
    for d, prop, sp, _dec, won, _ev in pool:
        if d >= cut:
            continue
        k = (prop, _conf_bucket(sp))
        agg[k][0] += 1 if won else 0
        agg[k][1] += 1
    cmap = {k: w / nn for k, (w, nn) in agg.items() if nn >= 50}

    print(f"\n=== REALISTIC daily top-{n_per_day} EV portfolio "
          f"[OOS test {cut}..{dates[-1]}, {len(test_days)} days, start 100u] ===")
    print("  each day: bet the top-N +EV props across ALL markets; chronological "
          "bankroll.")
    print("  RAW = served prob; CALIB = OOS per-(prop,confidence-bucket) recalibrated "
          "prob (the bucket-adjust). Kelly capped at "
          f"{int(kelly_cap * 100)}%/bet; legs sized independently (optimistic).")

    test = [x for x in pool if x[0] >= cut]
    for mode in ("RAW", "CALIB"):
        byday = defaultdict(list)   # date -> [(ev_used, dec, won, prob_used)]
        for d, prop, sp, dec, won, sev in test:
            if mode == "RAW":
                pu, evu = sp, sev
            else:
                cp = cmap.get((prop, _conf_bucket(sp)))
                if cp is None:
                    continue
                pu, evu = cp, cp * dec - 1.0
            byday[d].append((evu, dec, won, pu))
        selected = []
        for d in sorted(byday):
            top = sorted([x for x in byday[d] if x[0] > 0], reverse=True)[:n_per_day]
            selected.extend(top)
        if not selected:
            print(f"\n  [{mode}] no +EV bets selected.")
            continue
        n = len(selected)
        wins = sum(1 for _, _, w, _ in selected if w)
        pl_flat = sum((dec - 1.0) if w else -1.0 for _, dec, w, _ in selected)
        print(f"\n  [{mode}] bets={n} ({n / max(1, len(test_days)):.1f}/day)  "
              f"win%={100.0 * wins / n:.1f}  flat-1u ROI={100.0 * pl_flat / n:+.2f}%  "
              f"P/L={pl_flat:+.1f}u")
        for kf in kelly_fracs:
            bank, peak, maxdd = 100.0, 100.0, 0.0
            for evu, dec, won, _pu in selected:
                b = dec - 1.0
                frac = 0.0 if b <= 0 else min(kelly_cap, max(0.0, kf * (evu / b)))
                stake = frac * bank
                bank += (stake * b) if won else -stake
                peak = max(peak, bank)
                if peak > 0:
                    maxdd = max(maxdd, (peak - bank) / peak)
            growth = (bank / 100.0 - 1.0) * 100.0
            print(f"    {kf:>5}-Kelly  final {bank:9.1f}u  growth {growth:+9.1f}%  "
                  f"maxDD {maxdd * 100:5.1f}%")


def _topn_flat(rows, n_per_day):
    """rows: [(date, prob, decimal_odds, won)]. Each day take the top-N +EV, flat-1u.
    Returns (bets, win%, ROI%)."""
    from collections import defaultdict
    byday = defaultdict(list)
    for d, p, dec, won in rows:
        ev = p * dec - 1.0
        if ev > 0:
            byday[d].append((ev, dec, won))
    sel = []
    for d in byday:
        sel.extend(sorted(byday[d], reverse=True)[:n_per_day])
    if not sel:
        return 0, 0.0, 0.0
    m = len(sel)
    wins = sum(1 for _, _, w in sel if w)
    pl = sum((dec - 1.0) if w else -1.0 for _, dec, w in sel)
    return m, 100.0 * wins / m, 100.0 * pl / m


def _simulate_walk_forward(results, recal, calibration, seed_trust=1.0,
                           min_loop=50, refit_days=7, max_fit_obs=8000,
                           n_per_day=10):
    """WALK-FORWARD test of the deep-seed + online-CURRENT-SEASON-Platt architecture
    (Doug's proposal): step through obs chronologically; per (prop, line-bucket) refit a
    loop Platt on the current-season resolved obs SO FAR (reset each season) and blend it
    with the committed 4-season SEED via w = n_loop/(n_loop + trust*n_seed) — cold-start
    leans on the seed (prior), fills in -> current takes over. Compares STATIC (seed only)
    vs WALK-FORWARD served Brier + daily top-N ROI. Leakage-safe: a day's obs join the
    accumulator only AFTER that day is graded. Answers: does within-season adaptation help
    beyond the static pooled fit? (Opt-in via --walk-forward; it re-fits Platt repeatedly.)"""
    import props as _pm
    from recalibration import fit_platt, apply_platt
    from collections import defaultdict

    def _bkey(prop, line):
        cfg = calibration.get(prop) or {}
        if cfg.get("line_methods"):
            try:
                return (prop, _pm._resolve_line_bucket(cfg, line)[0])
            except Exception:
                return (prop, None)
        return (prop, None)

    def _seed(prop, line):
        cfg = _pm._resolve_recal_cfg(recal, prop, line, calibration.get(prop))
        if cfg and cfg.get("a") is not None:
            return cfg.get("a"), cfg.get("b"), (cfg.get("n_fit") or 500)
        return None, None, 500

    byday = defaultdict(list)
    for prop, r in results.items():
        for d, pm, line, o, op, up in (r.get("wf_obs") or []):
            if not d or pm is None or op is None or up is None:
                continue
            byday[d[:10]].append((prop, line, pm, o,
                                  american_to_decimal(op), american_to_decimal(up)))
    if sum(len(v) for v in byday.values()) < 500:
        print("\n=== WALK-FORWARD (online current-season Platt): too few obs. ===")
        return
    days = sorted(byday)

    acc = defaultdict(list)      # (prop,bkey) -> [(p_model, outcome)] current season
    loop_fit, last_fit = {}, {}
    cur_season = None
    st_rows, wf_rows = [], []
    sB = defaultdict(lambda: [0.0, 0])
    wB = defaultdict(lambda: [0.0, 0])
    mB = defaultdict(lambda: [0.0, 0])

    for di, day in enumerate(days):
        if day[:4] != cur_season:
            acc.clear(); loop_fit.clear(); last_fit.clear(); cur_season = day[:4]
        for prop, line, pm, o, do, du in byday[day]:
            k = _bkey(prop, line)
            a_s, b_s, n_seed = _seed(prop, line)
            ps = pm if a_s is None else apply_platt(pm, a_s, b_s)
            ps = min(max(ps if ps is not None else pm, 0.0), 1.0)
            lf = loop_fit.get(k)
            pw = ps
            if lf:
                pl = apply_platt(pm, lf[0], lf[1])
                if pl is not None:
                    w = lf[2] / (lf[2] + seed_trust * n_seed)
                    pw = (1.0 - w) * ps + w * min(max(pl, 0.0), 1.0)
            fair_o, _ = devig_two_way(1.0 / do, 1.0 / du)
            sB[prop][0] += (ps - o) ** 2; sB[prop][1] += 1
            wB[prop][0] += (pw - o) ** 2; wB[prop][1] += 1
            mB[prop][0] += (fair_o - o) ** 2; mB[prop][1] += 1
            for probv, rows in ((ps, st_rows), (pw, wf_rows)):
                if probv * do - 1.0 >= (1.0 - probv) * du - 1.0:
                    rows.append((day, probv, do, o == 1))
                else:
                    rows.append((day, 1.0 - probv, du, o == 0))
        for k, obs in acc.items():
            if len(obs) >= min_loop and di - last_fit.get(k, -10 ** 9) >= refit_days:
                fit = fit_platt([p for p, _ in obs[-max_fit_obs:]],
                                [oo for _, oo in obs[-max_fit_obs:]], max_iter=50)
                if fit is not None:
                    loop_fit[k] = (fit[0], fit[1], len(obs))
                    last_fit[k] = di
        for prop, line, pm, o, _do, _du in byday[day]:
            acc[_bkey(prop, line)].append((pm, o))

    print(f"\n=== WALK-FORWARD: deep seed + online CURRENT-SEASON Platt "
          f"(trust={seed_trust}, refit/{refit_days}d, reset each season) ===")
    print("  STATIC = seed only; WF = seed + online current-season blend. "
          "WF Brier<static & WF ROI>static => within-season adaptation helps.")
    print(f"  {'prop':<20} {'staticBrier':>11} {'wfBrier':>9} {'mktBrier':>9}")
    for prop in results:
        if sB[prop][1] == 0:
            continue
        print(f"  {prop:<20} {sB[prop][0] / sB[prop][1]:>11.4f} "
              f"{wB[prop][0] / wB[prop][1]:>9.4f} {mB[prop][0] / mB[prop][1]:>9.4f}")
    sm, sw, sr = _topn_flat(st_rows, n_per_day)
    wm, ww, wr = _topn_flat(wf_rows, n_per_day)
    print(f"\n  daily top-{n_per_day} flat ROI:  STATIC {sr:+.2f}% ({sm} bets, "
          f"{sw:.1f}% win)   WALK-FWD {wr:+.2f}% ({wm} bets, {ww:.1f}% win)")


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
            park_strength=0.0, rest_strength=0.0, recent_n="__calib__",
            weather_density_coef=0.0, weather_wind_coef=0.0, weather_strength=0.0):
    return {
        "half_life": half_life,
        # recent_n: newest-N window BEFORE decay. "__calib__" (default) = resolve to the
        # prop's LOCKED live window in run_player_props_backtest (fit==serve); an
        # explicit int/None (recency sweep) overrides (None=full).
        "recent_n": recent_n,
        # Weather-density sweep axes (Phase B; 0 strength = off = byte-identical):
        "weather_density_coef": weather_density_coef,
        "weather_wind_coef": weather_wind_coef,
        "weather_strength": weather_strength,
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


def _build_focused_props_grid():
    """A FOCUSED subset of _build_props_sweep_grid for FAST refits (opt-in).

    Keeps FULL resolution on the axes that actually win for props — half_life ×
    venue × shrink_k (32-cell cross) — and probes the axes that historically never
    clear the P2.1 variant-confirmation gate for props (opp_defense, def_adj, rest)
    only ONCE each against baseline. ~37 variants vs the full grid's 576, so ~15×
    less CPU (crucially the per-variant NegBin method-E fit) and ~15× less peak RAM.

    Built by SELECTING a subset of the full grid by name, so every variant's preset
    and label is byte-identical to the full grid's — selections are directly
    comparable and _is_baseline_variant / _parse_variant_name / _build_prop_cfg all
    behave unchanged. The full grid stays the DEFAULT; this only runs under an
    explicit --focused-grid, so a refit that needs to explore knob INTERACTIONS
    (which this drops) can always use the full sweep. See _build_props_sweep_grid."""
    full = _build_props_sweep_grid()
    keep = set()
    # Full cross of the prop-relevant axes (dead axes held off).
    for hl in ("none", "hl5", "hl10", "hl15"):
        for vs in ("ven0.0", "ven0.25"):
            for sk in ("shrink0", "shrink5", "shrink10", "shrink15"):
                keep.add(f"{hl}/opp0.0/defadj0.0/{sk}/{vs}/rest0.0")
    # Single-axis probes for the team/feature axes that don't win props.
    keep.update({
        "none/opp0.5/defadj0.0/shrink0/ven0.0/rest0.0",
        "none/opp1.0/defadj0.0/shrink0/ven0.0/rest0.0",
        "none/opp0.0/defadj0.5/shrink0/ven0.0/rest0.0",
        "none/opp0.0/defadj1.0/shrink0/ven0.0/rest0.0",
        "none/opp0.0/defadj0.0/shrink0/ven0.0/rest1.0",
    })
    return {k: full[k] for k in keep if k in full}


def _build_recency_sweep_grid(recent_ns=None, half_lives=None):
    """Joint recent_n × half_life grid for --recency-sweep (props, STEP 1).

    Validates the two never-swept recency defaults TOGETHER: recent_n (the history
    lookback window; live MLB=20, app.py) and half_life (recency decay; live props
    decay is OFF / None). Every OTHER knob is held at baseline/off so the two
    recency axes are ISOLATED — do NOT multiply this with the 576-cell
    _build_props_sweep_grid. The incumbent cell is n20/none (recent_n=20 +
    decay off); half_life None = equal weighting, recent_n None = full history.
    Grade per-prop on OOS Brier (run with --calibrate); MAE/Hit% are diagnostics."""
    if recent_ns is None:
        recent_ns = [10, 15, 20, 25, 30, 40]
    if half_lives is None:
        half_lives = [None, 5, 7, 10, 15]
    variants = {}
    for rn in recent_ns:
        for hl in half_lives:
            rn_label = "full" if rn is None else f"n{rn}"
            hl_label = "none" if hl is None else f"hl{hl}"
            variants[f"{rn_label}/{hl_label}"] = _preset(half_life=hl, recent_n=rn)
    return variants


def _build_weather_sweep_grid(density_coefs=None, wind_coefs=None, strengths=None):
    """Joint density_coef × wind_coef × strength grid for --weather-sweep (props).

    FITS the moist-air weather model (weather_factors.density_factor) rather than
    hand-setting it: density_coef scales (baseline − air_density(temp,humidity,
    pressure)), wind_coef scales out-to-CF wind, strength is the overall fraction.
    Only batter_hits / pitcher_earned_runs move (park_factors.PROP_PARK_KIND); every
    other prop is byte-identical across cells. Incumbent = the single 'wx_off' cell
    (strength 0 → neutral regardless of coefs, so it is not looped — avoids degenerate
    duplicates). Grade per-prop on OOS Brier (run with --calibrate)."""
    if density_coefs is None:
        density_coefs = [0.5, 1.0, 1.5, 2.0]      # density dev ~±0.05-0.10 kg/m³ → ±5-20%
    if wind_coefs is None:
        wind_coefs = [0.0, 0.003, 0.006]          # brackets the shipped _WEATHER_COEF
    if strengths is None:
        strengths = [0.5, 1.0]
    variants = {"wx_off": _preset(half_life=None)}   # incumbent baseline (weather off)
    for st in strengths:
        for dc in density_coefs:
            for wc in wind_coefs:
                variants[f"dc{dc}/wc{wc}/s{st}"] = _preset(
                    half_life=None, weather_density_coef=dc,
                    weather_wind_coef=wc, weather_strength=st)
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

    For MLB (P3/P4/P6) each player's per-game log comes from the StatsAPI warehouse.
    The whole season's logs are pulled ONCE per role (mlb_warehouse.get_calib_gamelogs_
    bulk) and every player is served from that in-memory index by his authoritative
    MLBAM id — no per-player round trip (thousands on a deep multi-season sweep), no
    name→ESPN-id hop, no ESPN gamelog fetch. A bare-name --players override is resolved
    to its MLBAM id via the game-context-free resolver. NBA/NFL use the ESPN path.
    (ESPN was fully removed for MLB in P4.)"""
    data = {}
    _bulk_by_role = {}

    def _bulk_for(role):
        # Lazily pull the WHOLE season's gamelogs for a role once (one query), then
        # serve every player from memory. Normalize to the two fact tables exactly as
        # get_calib_gamelog does (role == "pitcher" → pitcher, else batter).
        norm = "pitcher" if role == "pitcher" else "batter"
        if norm not in _bulk_by_role:
            try:
                import mlb_warehouse as _mw
                _bulk_by_role[norm] = (
                    _mw.get_calib_gamelogs_bulk(norm, season_year) or {})
            except Exception:
                _bulk_by_role[norm] = {}
        return _bulk_by_role[norm]

    for mlb_id, role, name in _iter_pool_players(players):
        if not name:
            continue
        if espn_sport == "baseball":
            import mlb_warehouse
            rid, rrole = mlb_id, role
            if not rid:
                # A bare-name --players override under the flag: resolve the
                # authoritative MLBAM id + role (the bulk index is keyed by role to
                # pick the batter vs pitcher fact table).
                import mlb_starters
                resolved = mlb_starters.resolve_mlbam_id(
                    name, season_year or mlb_warehouse._current_season())
                if resolved:
                    rid = resolved[0]
                    rrole = "pitcher" if resolved[1] else "batter"
            gamelog = _bulk_for(rrole).get(str(rid)) if rid else None
            if not gamelog:
                print(f"  [skip] {name}: no warehouse gamelog")
                continue
            # Own copy: the bulk index shares one list per id, and we sort in place.
            gamelog = list(gamelog)
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
    # STEP-2: resolve each prop's LOCKED live window so the refit fits methods at the
    # SAME window production serves (fit==serve). A variant carrying the "__calib__"
    # sentinel (the _preset default → normal refit / weather sweep) resolves to the
    # prop's calibration recent_n (null=full=None; absent=per-sport default); the
    # recency sweep passes an explicit recent_n and is untouched.
    import props as _props_mod
    _locked_props = (load_calibration(sport_key) or {}).get("props", {})
    _default_locked_rn = _props_mod._player_prop_recent_n(sport_key)

    def _locked_recent_n(pk):
        cfg = _locked_props.get(pk)
        if cfg and "recent_n" in cfg:
            return cfg["recent_n"]        # None=full, int=window
        return _default_locked_rn
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

    # ── Weather-density map (Phase B) — {(home_team_id, date): (temp,humidity,
    # pressure,wind_out,dome)}, built ONCE. The projection looks up each test game's
    # HOME-park weather and applies weather_factors.density_factor at the swept coefs.
    # Only when a variant turns weather on AND we're warehouse-native (MLB); inert
    # otherwise. Weather is a pre-outcome game condition → no leakage.
    weather_density_map, weather_name_to_id = {}, {}
    needs_weather = any((p.get("weather_strength", 0.0) or 0.0) > 0
                        for p in variants.values())
    if needs_weather and use_warehouse:
        import mlb_warehouse
        _wx_seasons = ({int(season_year)} if season_year else
                       {int(g["game_date"][:4]) for gl in player_data.values()
                        for g in gl if g.get("game_date")})
        weather_density_map = mlb_warehouse.game_weather_density_map(sorted(_wx_seasons))
        # opponent arrives as a NAME in the gamelog; invert id→name to key the map.
        weather_name_to_id = {v: str(k) for k, v
                              in mlb_warehouse._team_name_map().items()}
        print(f"Built weather-density map ({len(weather_density_map)} game-weather rows).")

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
                    # recent_n: cap the recency window to the newest N prior games
                    # BEFORE decay, mirroring live serving (espn_client gamelog[:n]
                    # then _recency_weights). prior_* arrays are most-recent-first and
                    # every _weighted_* primitive zips values against weights, so a
                    # length-N weight vector selects exactly the newest N games.
                    # recent_n=None -> full history (byte-identical to pre-recency).
                    # NB: only knobs that pair with `weights` honor the window;
                    # shrink_k / use_minutes read full prior_values, so the recency
                    # grid keeps those OFF (isolated axes). See _build_recency_sweep_grid.
                    _rn = params.get("recent_n")
                    if _rn == "__calib__":       # normal refit -> the prop's LOCKED window
                        _rn = _locked_recent_n(prop_key)
                    _nw = min(_rn, len(prior_values)) if _rn else len(prior_values)
                    base_w = _recency_weights(_nw, hl)
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

                    # ── Weather-density multiplier (Phase B) ──
                    # HOME park = player's team when home, else the upcoming opponent;
                    # look up that (home_team_id, date) weather and scale by the moist-air
                    # density model at the swept coefs. Like live's combined_mult, wx_mult
                    # is folded into BOTH `projected` (methods B/C/E, here) AND the method-A
                    # effective line (calib_obs block below) so method-A props (batter_hits)
                    # see weather too. Only batter_hits / pitcher_earned_runs
                    # (PROP_PARK_KIND); dome / no data -> 1.0. NB: keyed on the gamelog's
                    # game_date[:10] (UTC), so a late game whose UTC date leads its official
                    # play date simply misses -> 1.0 (coverage caveat, never a wrong day).
                    wx_mult = 1.0
                    wx_strength = params.get("weather_strength", 0.0) or 0.0
                    if (wx_strength > 0 and weather_density_map
                            and prop_key in park_factors.PROP_PARK_KIND):
                        _home_id = (str(tid) if upcoming_is_home
                                    else weather_name_to_id.get(upcoming_opp))
                        _wx = (weather_density_map.get((_home_id, str(test_date)[:10]))
                               if _home_id and test_date else None)
                        if _wx:
                            _t, _h, _p, _wo, _dome = _wx
                            wx_mult = weather_factors.density_factor(
                                _t, _h, _p, _wo,
                                params.get("weather_density_coef", 0.0),
                                params.get("weather_wind_coef", 0.0),
                                wx_strength, dome=_dome)
                            projected *= wx_mult

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
                            # Combined line shift = feat_mult × weather (matches live
                            # combined_mult → effective_line), so method A sees weather
                            # too, not just projected. Both are 1.0 when off → identical.
                            _line_mult = feat_mult * wx_mult
                            line_eff = (synthetic_line / _line_mult
                                        if _line_mult != 1.0 else synthetic_line)
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


def _evaluate_calibration_methods(obs, k_values, holdout=False,
                                  negbin_eligible=False):
    """
    Evaluate calibration methods for one (variant, prop) observation list.
    Returns a list of dicts: {method, k, brier, hit}.
    Methods: A (empirical), B (pooled Gaussian), C (pooled ECDF),
             B*@k, C*@k (per-player with shrinkage) for each k, and
             E (§2.2 Negative Binomial count model) when `negbin_eligible`.

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

    return _score_calibration_methods(fit_obs, score_obs, k_values,
                                      negbin_eligible=negbin_eligible)


def _score_calibration_methods(fit_obs, score_obs, k_values,
                               negbin_eligible=False):
    """
    Fit calibration params on `fit_obs` and score them on `score_obs`.

    Returns a list of {method, k, brier, hit} for methods A (empirical),
    B (pooled Gaussian), C (pooled ECDF) and B*/C* (per-player shrinkage) at
    each k. When `negbin_eligible` (a count prop the caller whitelisted via
    props.PROP_NEGBIN_ELIGIBLE) also scores E (§2.2 Negative Binomial): the
    same train-fit mean_scale + dispersion the real-line selector uses
    (book_line_calibration._score_abc_real), so a count prop with NO stored book
    lines (e.g. batter_total_bases) can still select the count model from the
    synthetic sweep. Both inputs use the calib_obs schema
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

    # E: §2.2 Negative Binomial count model — train-fit (mean_scale + dispersion),
    # scored at each score row's line, ONLY for a whitelisted count prop. Mirrors
    # book_line_calibration._score_abc_real's E branch verbatim (shared fit via
    # stats.fit_negbin_params) so the synthetic sweep and the real-line selector
    # can't drift. mean_scale/dispersion are line-invariant distributional params
    # (like B/C residuals), so a fit at the synthetic season-avg line serves
    # correctly at real book lines. fail-open: an unusable fit just omits E.
    if negbin_eligible:
        nb = fit_negbin_params([(proj, actual)
                                for _, proj, _, actual, _, *_ in fit_obs])
        if nb is not None:
            mean_scale, disp = nb
            pE = []
            for _, proj, line, _, _ in rows:
                mean = max(1e-9, mean_scale * proj)
                pE.append(negbin_at_least(int(line) + 1, mean, disp))
            results.append({"method": "E", "k": None,
                            "brier": _brier(pE, outcomes),
                            "hit": _hit_rate(pE, outcomes)})

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

    # Collect all rows. Show method E (NegBin) for count props so the sweep
    # DISPLAY matches what _best_per_prop actually selects (both E-eligible) — else
    # the tables silently omit E and read as "A wins" when E was never scored here.
    from props import PROP_NEGBIN_ELIGIBLE
    by_prop_rows = {prop: [] for prop in props}
    for vname, by_prop in results.items():
        for prop_key in props:
            obs = by_prop[prop_key].get("calib_obs") or []
            evals = _evaluate_calibration_methods(
                obs, k_values, holdout=holdout,
                negbin_eligible=(prop_key in PROP_NEGBIN_ELIGIBLE))
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


def _team_gate_tally(obs, shrink, ev_floor, edge_floor, legacy_threshold):
    """Flat-1u ROI over one team market's RAW obs under one (shrink, gate) combo.

    ``obs`` = [(raw_model_p, market_p, outcome, price_yes, price_no)] (yes = the
    modelled home/over side). Shrink pulls the raw prob toward 0.5 (p' = 0.5 +
    shrink*(p-0.5)); the higher-edge side is backed. Gate is ROI-primary
    (EV >= ev_floor AND edge >= edge_floor) when ev_floor is not None, else legacy
    (edge >= legacy_threshold AND EV > 0). Returns {n, pnl, roi, hit}."""
    pnl = won = 0.0
    n = 0
    for raw_p, mkt_p, outcome, price_yes, price_no in obs:
        if mkt_p is None:
            continue
        p = 0.5 + shrink * (raw_p - 0.5)
        if p >= mkt_p:
            side_prob, price, edge, win = p, price_yes, p - mkt_p, (outcome == 1)
        else:
            side_prob, price, edge, win = (1.0 - p, price_no, mkt_p - p,
                                           (outcome == 0))
        if price is None:
            continue
        dec = american_to_decimal(price)
        er = side_prob * dec - 1.0
        ok = ((er >= ev_floor and edge >= edge_floor) if ev_floor is not None
              else (edge >= legacy_threshold and er > 0))
        if not ok:
            continue
        pnl += (dec - 1.0) if win else -1.0
        won += 1.0 if win else 0.0
        n += 1
    return {"n": n, "pnl": pnl,
            "roi": (pnl / n) if n else None, "hit": (won / n) if n else None}


def _calibration_bakeoff(dated):
    """OOS Platt-vs-scalar-shrink bake-off per market (READ-ONLY, #2).

    dated[market] = [(date, raw_model_p, market_p, outcome), ...]. Fits BOTH a
    scalar shrink and a Platt curve on a chronological TRAIN split (first 80% by
    date) and scores each on the untouched HOLDOUT (last 20%) — the fair test,
    since Platt has 2 params vs the shrink's 1 and an in-sample comparison would
    flatter Platt. Also runs recalibration.fit_platt_chronological (its strict
    2-fold expanding champion gate: calibrated must beat RAW on Brier AND log-loss
    in every later fold) to say whether a Platt fit is ship-worthy. Nothing is
    written — this only answers 'does a learned curve beat the flat scalar?'."""
    from recalibration import fit_platt, apply_platt, fit_platt_chronological
    print("\n=== CALIBRATION BAKE-OFF: Platt vs scalar-shrink (OOS holdout Brier) ===")
    print("  lower = better; TRAIN=first 80% by date, HOLDOUT=last 20%. 'platt-gate'"
          " = passes the strict beat-raw-OOS champion gate (ship-worthy).")
    print("  {:<10}{:>7}{:>9}{:>9}{:>9}{:>9}   {}".format(
        "market", "n_hold", "raw", "shrink", "platt", "market", "platt-gate"))
    for market in MARKETS:
        rows = sorted((r for r in dated.get(market, []) if r[0]),
                      key=lambda r: r[0])
        if len(rows) < 300:
            print(f"  {market:<10} (thin: {len(rows)} obs)")
            continue
        cut_date = rows[int(len(rows) * 0.8)][0]
        train = [r for r in rows if r[0] < cut_date]
        hold = [r for r in rows if r[0] >= cut_date]
        if len(train) < 100 or len(hold) < 50:
            print(f"  {market:<10} (thin split)")
            continue
        fit = fit_platt([r[1] for r in train], [r[3] for r in train])
        sres = _best_shrink([(r[1], r[2], r[3]) for r in train])
        s = sres[0] if sres else 1.0

        def _brier(fn):
            return sum((fn(r) - r[3]) ** 2 for r in hold) / len(hold)
        raw_b = _brier(lambda r: r[1])
        mkt_b = _brier(lambda r: r[2])
        shr_b = _brier(lambda r: 0.5 + s * (r[1] - 0.5))
        # fit_platt returns None when the optimal slope a<=0.2 (the raw prob has
        # ~no calibratable signal → Platt would squash to the base rate) — report
        # that as "degen", not a silent nan.
        plt_str = ("{:.4f}".format(_brier(lambda r: apply_platt(r[1], fit[0], fit[1])))
                   if fit else "  degen")
        gate = fit_platt_chronological([(r[0], r[1], r[3]) for r in rows])
        print("  {:<10}{:>7}{:>9.4f}{:>9.4f}{:>9}{:>9.4f}   {} (shrink s={})".format(
            market, len(hold), raw_b, shr_b, plt_str, mkt_b,
            "PASS" if gate else "fail", s))
    print("  platt 'degen' / gate 'fail' = a learned curve is degenerate or can't")
    print("  beat raw OOS → the scalar shrink is the calibration ceiling here.")
    print("  (Diagnostic only — nothing written.)")


def _lineup_runs_diag(lineup):
    """#3 v2 (READ-ONLY): grade the bottom-up lineup-runs P(home win) head-to-head
    vs the recency model + the market on the SAME games (Brier; raw + best-shrunk).

    lineup["moneyline"] = [(date, lineup_p, recency_p, market_p, outcome), ...].
    Answers whether a runs-FIRST lineup estimator (today's 9, as-of PA-weighted OPS
    → expected_runs_from_factors vs the opposing starter → Poisson margin) beats the
    recency margin model, and how close it gets to the close, BEFORE any live
    wiring. The additive lineup nudge was refuted (b14b9d6); this is the replace-
    the-offense-term rebuild. Nothing written."""
    rows = [r for r in lineup.get("moneyline", []) if r[0]]
    print("\n=== #3 v2 LINEUP-RUNS model vs recency vs market (moneyline Brier) ===")
    if len(rows) < 100:
        print(f"  (thin: {len(rows)} games with a resolvable lineup-runs prob — "
              "need as-of lineups + both starters; expected sparse for old seasons)")
        return
    n = len(rows)

    def _brier(idx, s=1.0):
        return sum((0.5 + s * (r[idx] - 0.5) - r[4]) ** 2 for r in rows) / n

    def _best_shrunk(idx):
        best_s, best_b = 1.0, None
        for s in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10):
            b = _brier(idx, s)
            if best_b is None or b < best_b:
                best_s, best_b = s, b
        return best_s, best_b
    print(f"  games graded: {n}   home-win base rate = {sum(r[4] for r in rows)/n:.3f}")
    print("  (same games for all three rows — apples-to-apples; lower Brier better)")
    print("  {:<12}{:>11}{:>13}{:>9}".format(
        "model", "raw Brier", "best-shrunk", "@shrink"))
    for label, idx in (("lineup", 1), ("recency", 2), ("market", 3)):
        raw = _brier(idx)
        bs, bb = _best_shrunk(idx)
        print("  {:<12}{:>11.4f}{:>13.4f}{:>9}".format(label, raw, bb, f"{bs:.2f}"))
    print("  lineup < recency => the runs-first estimator adds info the margin model")
    print("  lacks; lineup -> market => CLV potential. (Diagnostic only; nothing written.)")


def _prob_metrics_panel(obs):
    """#11: per-market probability metrics that match our creed better than raw
    Brier — Brier-SKILL vs the market (>0 = we beat the close), ECE, and the
    calibration SLOPE (<1 = overconfident, the disease we keep naming). obs =
    {market: [(raw_model_p, market_p, outcome, price_yes, price_no)]}."""
    import prob_metrics as pm
    print("\n=== PROBABILITY METRICS: model vs market (mining #11) ===")
    print("  BSS>0 = beat the close on Brier; ECE lower=better; calibSlope<1 = "
          "OVERCONFIDENT.")
    print("  {:<10}{:>7}{:>9}{:>8}{:>12}".format(
        "market", "n", "BSS", "ECE", "calibSlope"))
    for market in MARKETS:
        rows = [(o[0], o[1], o[2]) for o in obs.get(market, [])
                if o[1] is not None]
        if len(rows) < 50:
            print(f"  {market:<10} (thin: {len(rows)})")
            continue
        probs = [r[0] for r in rows]
        refs = [r[1] for r in rows]
        ys = [r[2] for r in rows]
        bss = pm.brier_skill_score(probs, ys, refs)
        e = pm.ece(probs, ys)
        cs = pm.calibration_slope(probs, ys)
        print("  {:<10}{:>7}{:>9}{:>8}{:>12}".format(
            market, len(rows),
            f"{bss*100:+.2f}%" if bss is not None else "-",
            f"{e:.3f}" if e is not None else "-",
            f"{cs['slope']:.2f}" if cs else "-"))
    print("  (BSS<0 across markets = we don't beat the close on accuracy — expected "
          "at the variance floor; watch calibSlope for overconfidence.)")


def diagnose_team_gate(sport_key, espn_sport, espn_league, season_year=None,
                       limit=100000, store_label="", source="auto",
                       snapshot="close"):
    """Team-market GATE + SHRINK lens (NO WRITE): grade ML/spread/total over the
    warehoused closing-line holdout, collect each obs's RAW (pre-shrink) model
    prob + market prob + outcome + prices, then sweep probability-shrink x
    recommendation-gate and report realized flat-1u ROI + volume per combo.

    Answers whether the live moneyline shrink (0.25) + edge-5% gate double-
    suppress ML, and which (shrink, gate) maximizes realized team-market ROI — the
    team-market analog of the props --gate-diag. Consensus/closing-priced; INFORMS,
    never auto-writes."""
    obs = {m: [] for m in MARKETS}
    dated = {m: [] for m in MARKETS}   # (date, raw_p, market_p, outcome) for #2 bake-off
    lineup = {m: [] for m in MARKETS}  # #3 v2 lineup-runs head-to-head (moneyline)
    # Grade once (RAW obs collected via the hook); prob_shrink=1.0 so the printed
    # Brier table is the unshrunk baseline — the sweep applies shrink itself.
    run_odds_backtest(sport_key, espn_sport, espn_league, limit=limit, window=10,
                      variants={"live": VARIANT_PRESETS.get("all", {})},
                      season_year=season_year, threshold_pct=5.0,
                      write_calibration=False, store_label=store_label,
                      engine="live", prob_shrink=1.0, source=source,
                      snapshot=snapshot,
                      supplement_log=False, collect_obs=obs, collect_dated=dated,
                      collect_lineup=lineup)

    SHRINKS = (1.0, 0.5, 0.25, 0.15, 0.10)
    # Expanded toward RARER, higher-edge gates — the disqualification lens: at
    # edge>=5% the model bets ~80% of games (overconfidence fabricates edges), so
    # sweep up to edge>=15% / EV>=12% to find where fewer, larger disagreements
    # actually turn profitable. '%bet' below = share of graded games recommended.
    GATES = [
        ("edge>=5% & EV>0 (legacy)", None, 0.0, 0.05),
        ("edge>=7% & EV>0", None, 0.0, 0.07),
        ("edge>=10% & EV>0", None, 0.0, 0.10),
        ("edge>=15% & EV>0", None, 0.0, 0.15),
        ("EV>=5% & edge>=1%", 0.05, 0.01, 0.0),
        ("EV>=8% & edge>=2%", 0.08, 0.02, 0.0),
        ("EV>=12% & edge>=3%", 0.12, 0.03, 0.0),
    ]
    print("\n=== TEAM-MARKET gate x shrink lens (realized flat-1u ROI) ===")
    print("  shrink pulls the model prob toward 0.5 (1.0 = none, 0.25 = live ML).")
    print("  Closing-priced; RELATIVE ranking is the signal.")
    print("  !! MONEYLINE is the CLEAN row: it collects the PRE-shrink model_prob, so")
    print("    the shrink axis maps 1:1 to the live shrink. SPREADS is CONFOUNDED —")
    print("    model_cover_rate already bakes in the live spread shrink + the")
    print("    expected_runs_challenger blend, so the shrink axis DOUBLE-counts;")
    print("    treat the spread rows as directional-only, not a shrink recommendation.")
    for market in MARKETS:
        mobs = obs[market]
        n_obs = len(mobs) or 1
        print(f"\n  {market.upper()} ({len(mobs)} graded obs):")
        print("    {:<24}{:>7}{:>8}{:>7}{:>9}{:>10}".format(
            "gate", "shrink", "n_bets", "%bet", "ROI%", "P/L(u)"))
        for label, ev_floor, edge_floor, legacy in GATES:
            for s in SHRINKS:
                t = _team_gate_tally(mobs, s, ev_floor, edge_floor, legacy)
                roi = f"{t['roi'] * 100:+.1f}" if t["roi"] is not None else "-"
                print("    {:<24}{:>7}{:>8}{:>6.0f}%{:>9}{:>+10.2f}".format(
                    label, f"{s:.2f}", t["n"], 100 * t["n"] / n_obs, roi,
                    t["pnl"]))
    print("\n  (Diagnostic only — nothing written.)")
    # #11: richer probability metrics (beat-market + calibration, not raw Brier).
    _prob_metrics_panel(obs)
    # #2: does a learned Platt curve beat the flat scalar shrink OOS?
    _calibration_bakeoff(dated)
    # #3 v2: does the bottom-up lineup-runs model beat the recency model?
    _lineup_runs_diag(lineup)


def _bankroll_sim(bets, shrink=0.25, edge_gate=0.05, method="kelly",
                  z=1.0, frac=0.5, cap=0.05, b0=100.0):
    """Chronological bankroll simulation over per-game rows
    (date, raw_home_p, fair_home, price_home, price_away, home_won, n_eff).

    Selects the VALUE side exactly like _team_gate_tally: shrink the home prob, back
    home when shrunk>=fair_home else back away (the value side — may be the dog),
    then the value gate (edge >= ``edge_gate`` AND +EV at the price), then stakes:
      'flat'   -> constant 1u (1% of b0), non-compounding (the incumbent baseline)
      'kelly'  -> fractional-Kelly of the CURRENT bankroll (compounding)
      'ukelly' -> uncertainty-Kelly: size off prob_low = prob_interval_low(sp,n_eff,z)
    Returns {n_bets, growth_pct, max_dd_pct, sharpe}. Drawdown is tracked on the
    running equity; Sharpe is mean/stdev of per-bet return-on-stake."""
    import statistics
    rows = sorted((b for b in bets if b and b[1] is not None and b[2] is not None),
                  key=lambda r: r[0])
    bankroll = peak = float(b0)
    flat_unit = float(b0) * 0.01
    max_dd = 0.0
    rets = []
    for _date, raw_home, fair_home, price_home, price_away, home_won, n_eff in rows:
        p = _shrink_prob(raw_home, shrink)
        if p >= fair_home:
            sp, price, edge, won = p, price_home, p - fair_home, (home_won == 1)
        else:
            sp, price, edge, won = (1.0 - p, price_away, fair_home - p,
                                    (home_won == 0))
        if price is None or edge < edge_gate:
            continue
        er = _expected_roi(sp, price)
        if er is None or er <= 0.0:
            continue                          # -EV at the price -> not a value bet
        if method == "flat":
            stake = flat_unit
        elif method == "kelly":
            stake = kelly_stake(sp, price, bankroll, frac, cap)
        else:  # ukelly
            stake = kelly_stake_uncertain(
                sp, prob_interval_low(sp, n_eff, z), price, bankroll, frac, cap)
        if stake <= 0.0:
            continue                          # abstained (uncertainty interval)
        dec = american_to_decimal(price)
        br_before = bankroll
        profit = stake * (dec - 1.0) if won else -stake
        bankroll += profit
        # Return on BANKROLL (not on stake) so Sharpe reflects the SIZING risk:
        # flat's tiny constant fractions -> low vol; Kelly's bankroll-proportional
        # stakes -> higher vol. (return-on-stake is stake-invariant = useless here.)
        rets.append(profit / br_before if br_before > 0 else 0.0)
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    growth = (bankroll / float(b0) - 1.0) * 100.0
    sharpe = (statistics.mean(rets) / statistics.pstdev(rets)
              if len(rets) > 1 and statistics.pstdev(rets) > 0 else float("nan"))
    return {"n_bets": len(rets), "growth_pct": growth,
            "max_dd_pct": max_dd * 100.0, "sharpe": sharpe}


def sizing_sweep(sport_key, espn_sport, espn_league, season_year=None,
                 limit=100000, store_label="", source="auto",
                 shrink=0.25, edge_gate=0.05):
    """MONEYLINE bankroll sim (Batch B1): flat-1u vs fractional-Kelly vs
    uncertainty-Kelly (size off the low bound of the win-prob interval + abstain
    when it spans break-even) over the warehoused closing-line holdout, at the
    ROI-optimal moneyline config (shrink 0.25, no blend, edge>=5%). Shows whether
    uncertainty-aware sizing improves RISK-ADJUSTED returns (growth vs max
    drawdown) — the thing flat-1u ROI can't reveal. Diagnostic only; nothing
    written."""
    _warn_small_limit(limit)
    bets = {m: [] for m in MARKETS}
    run_odds_backtest(sport_key, espn_sport, espn_league, limit=limit, window=10,
                      variants={"live": VARIANT_PRESETS.get("all", {})},
                      season_year=season_year, threshold_pct=5.0,
                      write_calibration=False, store_label=store_label,
                      engine="live", prob_shrink=1.0, source=source,
                      supplement_log=False, collect_bets=bets)
    ml = bets["moneyline"]
    print(f"\n=== MONEYLINE bankroll sim (Batch B1) — shrink {shrink}, "
          f"edge>={edge_gate*100:.0f}%, cap 5%/leg ===")
    print(f"  {len(ml)} graded moneyline games; VALUE side; chronological; b0=100u. "
          "flat=1u/bet (incumbent).")
    print("  {:<16}{:>7}{:>10}{:>9}{:>8}".format(
        "sizing", "bets", "growth%", "maxDD%", "Sharpe"))
    print("  " + "-" * 48)

    def _row(label, method, z=1.0, frac=0.5):
        r = _bankroll_sim(ml, shrink=shrink, edge_gate=edge_gate, method=method,
                          z=z, frac=frac)
        sh = f"{r['sharpe']:.3f}" if r["sharpe"] == r["sharpe"] else "-"
        print("  {:<16}{:>7}{:>+10.1f}{:>9.1f}{:>8}".format(
            label, r["n_bets"], r["growth_pct"], r["max_dd_pct"], sh))

    _row("flat-1u", "flat")
    _row("eighth-Kelly", "kelly", frac=0.125)
    _row("quarter-Kelly", "kelly", frac=0.25)
    _row("half-Kelly", "kelly", frac=0.5)
    for z in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        _row(f"unc-Kelly z={z}", "ukelly", z=z, frac=0.5)
    print("\n  (Kelly fractions on the SAME bets show the growth/drawdown trade; "
          "unc-Kelly (half base) sizes off the win-prob interval low bound + "
          "abstains when it spans")
    print("   break-even. z=0.0 == half-Kelly (sanity); rising z = more "
          "conservative / more abstains. Watch growth vs maxDD. Diagnostic only.)")


# Real-world betting policy per market for the top-N portfolio test: which markets
# are eligible, the shrink applied to the collected prob (moneyline is RAW -> shrink
# 0.25; spreads/totals are the SERVED composite -> 1.0 = none), and the edge gate.
# Moneyline = the replicated edge; spreads = high-conviction only (edge>=10%); totals
# abstained (loses at volume). See the Batch-B triage.
_PORTFOLIO_POLICY = {
    "moneyline": {"enabled": True,  "shrink": 0.25, "edge_gate": 0.05},
    "spreads":   {"enabled": True,  "shrink": 1.0,  "edge_gate": 0.10},
    "totals":    {"enabled": False, "shrink": 1.0,  "edge_gate": 0.05},
}


def _portfolio_sim(bets_by_market, top_n, policy=None, b0=100.0):
    """Real-world portfolio sim: each day, keep only the BEST ``top_n`` value bets
    (ranked by EV) across the ELIGIBLE markets, flat-1u, chronological.

    bets_by_market[market] = [(date, home_side_prob, fair, price_home, price_away,
    home_won, n_eff)] (home side = home / home-cover / over). Per market: apply the
    policy shrink, pick the value side (shrunk prob vs fair), gate on edge AND +EV.
    top_n=None = no cap (all value bets — the current all-in ROI). Returns
    {n, roi, win, growth_pct, max_dd_pct, avg_per_day}."""
    from collections import defaultdict
    policy = policy or _PORTFOLIO_POLICY
    per_day = defaultdict(list)     # date -> [(ev, price, won)]
    for market, rows in (bets_by_market or {}).items():
        pol = policy.get(market)
        if not pol or not pol.get("enabled"):
            continue
        s, gate = pol["shrink"], pol["edge_gate"]
        for row in rows:
            try:
                date, hp, fair, ph, pa, hw, _neff = row
            except (ValueError, TypeError):
                continue
            if hp is None or fair is None:
                continue
            p = _shrink_prob(hp, s)
            if p >= fair:
                sp, price, edge, won = p, ph, p - fair, (hw == 1)
            else:
                sp, price, edge, won = 1.0 - p, pa, fair - p, (hw == 0)
            if price is None or edge < gate:
                continue
            ev = _expected_roi(sp, price)
            if ev is None or ev <= 0.0:
                continue
            per_day[date].append((ev, price, won))
    bankroll = peak = float(b0)
    unit = float(b0) * 0.01
    max_dd = pnl_u = 0.0
    n = won_n = 0
    for date in sorted(per_day):
        day = sorted(per_day[date], key=lambda x: x[0], reverse=True)  # best EV first
        if top_n is not None:
            day = day[:top_n]
        for ev, price, won in day:
            dec = american_to_decimal(price)
            profit = unit * (dec - 1.0) if won else -unit
            bankroll += profit
            pnl_u += (dec - 1.0) if won else -1.0
            n += 1
            won_n += 1 if won else 0
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
    return {"n": n, "roi": (pnl_u / n * 100.0) if n else None,
            "win": (won_n / n * 100.0) if n else None,
            "growth_pct": (bankroll / float(b0) - 1.0) * 100.0,
            "max_dd_pct": max_dd * 100.0,
            "avg_per_day": (n / len(per_day)) if per_day else 0.0}


def top_n_sweep(sport_key, espn_sport, espn_league, season_year=None,
                limit=100000, store_label="", source="auto"):
    """TOP-N/DAY portfolio test (Batch B1): the REAL-WORLD ROI — you only place the
    best ~N wagers/day, not every value bet. Each day, rank the eligible value bets
    (moneyline shrink 0.25 edge>=5%; spreads high-conviction edge>=10%; totals off)
    by EV and keep the top N, flat-1u, chronological. Swept N=5/10/15/all, for the
    full policy AND moneyline-only, so you can see whether tightening to your real
    bet count helps and whether high-conviction spreads add to the portfolio.
    Diagnostic only; nothing written."""
    _warn_small_limit(limit)
    bets = {m: [] for m in MARKETS}
    run_odds_backtest(sport_key, espn_sport, espn_league, limit=limit, window=10,
                      variants={"live": VARIANT_PRESETS.get("all", {})},
                      season_year=season_year, threshold_pct=5.0,
                      write_calibration=False, store_label=store_label,
                      engine="live", prob_shrink=1.0, source=source,
                      supplement_log=False, collect_bets=bets)
    ml_only = {"moneyline": dict(_PORTFOLIO_POLICY["moneyline"]),
               "spreads": {"enabled": False}, "totals": {"enabled": False}}
    print("\n=== TOP-N/DAY portfolio (Batch B1) — best N value bets/day, EV-ranked, "
          "flat-1u ===")
    print("  ML shrink 0.25 edge>=5% + SPREADS edge>=10% (high-conviction); totals "
          "off. b0=100u.")
    for label, pol in (("ML + hi-conv spreads", _PORTFOLIO_POLICY),
                       ("moneyline only", ml_only)):
        print(f"\n  [{label}]")
        print("    {:>5}{:>7}{:>9}{:>8}{:>10}{:>9}{:>9}".format(
            "topN", "bets", "/day", "ROI%", "growth%", "maxDD%", "win%"))
        for N in (5, 10, 15, None):
            r = _portfolio_sim(bets, N, pol)
            roi = f"{r['roi']:+.1f}" if r["roi"] is not None else "-"
            win = f"{r['win']:.1f}" if r["win"] is not None else "-"
            print("    {:>5}{:>7}{:>9.1f}{:>8}{:>+10.1f}{:>9.1f}{:>9}".format(
                "all" if N is None else N, r["n"], r["avg_per_day"], roi,
                r["growth_pct"], r["max_dd_pct"], win))
    print("\n  ('/day' = avg bets placed per day; if it's below the cap, N doesn't "
          "bind. Compare N=all (every value bet) vs N=10 (what you'd really place):")
    print("   if ROI RISES as N tightens, your best bets are your good bets. "
          "Diagnostic only — nothing written.)")


def _warn_small_limit(limit):
    """These ROI sweeps need the full window; the global --limit default (200) caps
    to the last 200 games → ~100 bets/cell → noisy 'best cell' that jumps corners.
    Warn loudly so a capped run is never mistaken for signal."""
    if limit and limit < 1000:
        bar = "!" * 74
        print(f"\n{bar}\n!! --limit={limit}: this sweep is CAPPED to the last {limit} "
              f"games -> tiny per-cell\n   samples and NOISY ROI (the 'best cell' will "
              f"jump around, and the gates will\n   disagree). Re-run with  --limit "
              f"100000  for the full window before trusting it.\n{bar}")


def unleash_sweep(sport_key, espn_sport, espn_league, season_year=None,
                  limit=100000, store_label="", source="auto"):
    """Team-market UNLEASH sweep (NO WRITE): re-grade the LIVE pipeline with each
    market-anchoring knob RELEASED one at a time, and report realized flat-1u
    ROI-at-gate vs the live baseline. Resolves the anchoring-audit team-market
    shortlist: prob_shrink {moneyline, spreads, totals} + DEFAULT_PYTHAG_WEIGHT.

    Why this is more than --team-gate-sweep: ML and totals collect the PRE-shrink
    prob, so their shrink is swept offline (clean, no re-grade). But the SPREADS
    cover prob is baked (post-shrink + expected-runs challenger), and the 0.35
    Pythagorean blend is baked into the ML prob, so those two must be true
    override RE-GRADES — spreads via the prob_shrink cache (spreads->1.0, the
    challenger share held fixed), pythag via analysis.DEFAULT_PYTHAG_WEIGHT->0.0.
    This runs 3 live re-grades (baseline / pythag-off / spreads-unshrunk); the
    first pays the matchup-feature network cost, the next two hit the disk cache.

    Judge on ROI-at-gate; pass a --season / --seasons TEST window for an OOS read.
    'more extreme + rare gate' wins here ONLY if ROI rises at a LOW-%bet gate.
    Nothing is written — this is the evidence feed for the disqualification step."""
    _warn_small_limit(limit)
    import pricing_common

    live_shrink = {m: pricing_common._shrink_factor(sport_key, m) for m in MARKETS}
    live_pythag = analysis.DEFAULT_PYTHAG_WEIGHT
    _SENT = object()

    def _run(pythag_weight, shrink_cache, tag):
        obs = {m: [] for m in MARKETS}
        saved_pythag = analysis.DEFAULT_PYTHAG_WEIGHT
        saved_cache = pricing_common._PROB_SHRINK_CACHE.get(sport_key, _SENT)
        print(f"\n########## unleash variant: {tag} "
              f"(pythag={pythag_weight}, shrink={shrink_cache}) ##########")
        try:
            analysis.DEFAULT_PYTHAG_WEIGHT = pythag_weight
            pricing_common._PROB_SHRINK_CACHE[sport_key] = dict(shrink_cache)
            run_odds_backtest(
                sport_key, espn_sport, espn_league, limit=limit, window=10,
                variants={"live": VARIANT_PRESETS.get("all", {})},
                season_year=season_year, threshold_pct=5.0,
                write_calibration=False, store_label=store_label,
                engine="live", prob_shrink=1.0, source=source,
                supplement_log=False, collect_obs=obs)
        finally:
            analysis.DEFAULT_PYTHAG_WEIGHT = saved_pythag
            if saved_cache is _SENT:
                pricing_common._PROB_SHRINK_CACHE.pop(sport_key, None)
            else:
                pricing_common._PROB_SHRINK_CACHE[sport_key] = saved_cache
        return obs

    base = _run(live_pythag, live_shrink, "baseline (live)")
    pyth = _run(0.0, live_shrink, "pythag OFF")
    spun = _run(live_pythag, {**live_shrink, "spreads": 1.0}, "spreads UNSHRUNK")

    GATES = [
        ("edge>=5% (legacy)", None, 0.0, 0.05),
        ("edge>=10%",         None, 0.0, 0.10),
        ("edge>=15%",         None, 0.0, 0.15),
        ("EV>=8% & edge>=2%", 0.08, 0.02, 0.0),
        ("EV>=12% & edge>=3%", 0.12, 0.03, 0.0),
    ]

    def _cmp(title, a_lbl, a_obs, a_shr, b_lbl, b_obs, b_shr):
        na = len(a_obs) or 1
        nb = len(b_obs) or 1
        print(f"\n=== {title} ===")
        print("  {:<20}|{:>18}|{:>18}".format("gate", a_lbl, b_lbl))
        print("  {:<20}|{:>6}{:>5}{:>7}|{:>6}{:>5}{:>7}".format(
            "", "n", "%bet", "ROI%", "n", "%bet", "ROI%"))
        for label, evf, edf, leg in GATES:
            ta = _team_gate_tally(a_obs, a_shr, evf, edf, leg)
            tb = _team_gate_tally(b_obs, b_shr, evf, edf, leg)
            ra = f"{ta['roi'] * 100:+.1f}" if ta["roi"] is not None else "-"
            rb = f"{tb['roi'] * 100:+.1f}" if tb["roi"] is not None else "-"
            print("  {:<20}|{:>6}{:>4.0f}%{:>7}|{:>6}{:>4.0f}%{:>7}".format(
                label, ta["n"], 100 * ta["n"] / na, ra,
                tb["n"], 100 * tb["n"] / nb, rb))

    print("\n\n############ UNLEASH SWEEP — ROI-at-gate: LIVE vs UNLEASHED ############")
    print("  Closing-priced flat-1u ROI on the same games. A HIGHER ROI at a RARE")
    print("  (low %bet) gate = the leash was suppressing harvestable edge. A higher")
    print("  ROI only at the broad edge>=5% gate is usually just more variance.")
    print("  Judge OOS (pass a --season/--seasons test window), never on Brier.")

    _cmp("MONEYLINE — prob_shrink",
         f"LIVE s={live_shrink['moneyline']:.2f}", base["moneyline"],
         live_shrink["moneyline"], "RAW s=1.00", base["moneyline"], 1.0)
    _cmp(f"MONEYLINE — Pythagorean (ML shrink held at {live_shrink['moneyline']:.2f})",
         f"LIVE w={live_pythag:.2f}", base["moneyline"], live_shrink["moneyline"],
         "w=0.00", pyth["moneyline"], live_shrink["moneyline"])
    _cmp("MONEYLINE — BOTH released (pythag 0 + raw prob)",
         "LIVE", base["moneyline"], live_shrink["moneyline"],
         "pyth0+raw", pyth["moneyline"], 1.0)
    _cmp("TOTALS — prob_shrink",
         f"LIVE s={live_shrink['totals']:.2f}", base["totals"],
         live_shrink["totals"], "RAW s=1.00", base["totals"], 1.0)
    _cmp("SPREADS — prob_shrink (expected-runs challenger held fixed)",
         f"LIVE s={live_shrink['spreads']:.2f}", base["spreads"], 1.0,
         "RAW s=1.00", spun["spreads"], 1.0)
    print("\n  (Diagnostic only — nothing written. SPREADS rows are a true re-grade;")
    print("   ML/TOTALS shrink is the offline sweep on the pre-shrink obs.)")


def _regrade_ml_obs(sport_key, espn_sport, espn_league, season_year, limit,
                    store_label, source, pythag_weight):
    """Re-grade the live ML analyzer with DEFAULT_PYTHAG_WEIGHT overridden; return the
    collected PRE-shrink moneyline obs (the shrink axis is then swept OFFLINE). The
    global is restored in a finally. Shared by --pythag-sweep and --combo-sweep so
    the whole pythag x shrink grid costs only one re-grade per pythag weight."""
    obs = {m: [] for m in MARKETS}
    saved = analysis.DEFAULT_PYTHAG_WEIGHT
    print(f"\n########## re-grade: pythag w={pythag_weight:.2f} ##########")
    try:
        analysis.DEFAULT_PYTHAG_WEIGHT = pythag_weight
        run_odds_backtest(
            sport_key, espn_sport, espn_league, limit=limit, window=10,
            variants={"live": VARIANT_PRESETS.get("all", {})},
            season_year=season_year, threshold_pct=5.0,
            write_calibration=False, store_label=store_label,
            engine="live", prob_shrink=1.0, source=source,
            supplement_log=False, collect_obs=obs)
    finally:
        analysis.DEFAULT_PYTHAG_WEIGHT = saved
    return obs["moneyline"]


def _regrade_ml_components(sport_key, espn_sport, espn_league, season_year, limit,
                          store_label, source, tag=""):
    """Re-grade the live ML analyzer at pythag=0 and return the collected COMPONENT
    obs [(recency_p, pythag_p, market_p, outcome, price_yes, price_no)] — pure recency
    + raw per-game pythag prob, for offline A(recency) x B(pythag) recombination.
    Restores the global. Shared by --ab-sweep and --oos-ab."""
    comp = {m: [] for m in MARKETS}
    saved = analysis.DEFAULT_PYTHAG_WEIGHT
    print(f"\n########## re-grade components (pythag=0){(' ' + tag) if tag else ''} "
          f"##########")
    try:
        analysis.DEFAULT_PYTHAG_WEIGHT = 0.0
        run_odds_backtest(
            sport_key, espn_sport, espn_league, limit=limit, window=10,
            variants={"live": VARIANT_PRESETS.get("all", {})},
            season_year=season_year, threshold_pct=5.0,
            write_calibration=False, store_label=store_label,
            engine="live", prob_shrink=1.0, source=source,
            supplement_log=False, collect_components=comp)
    finally:
        analysis.DEFAULT_PYTHAG_WEIGHT = saved
    return comp["moneyline"]


def pythag_sweep(sport_key, espn_sport, espn_league, season_year=None,
                 limit=100000, store_label="", source="auto", weights=None):
    """Sweep DEFAULT_PYTHAG_WEIGHT x prob_shrink for MONEYLINE (NO WRITE).

    The Pythagorean blend is baked into the collected ML prob, so each weight is a
    live RE-GRADE; the ML shrink is applied OFFLINE on the collected pre-shrink obs,
    so every re-grade yields BOTH shrink columns (live + off) for free. Finds the
    weight that maximizes realized ROI-at-gate, and whether that optimum shifts when
    the shrink is on vs off. MONEYLINE only — pythag doesn't touch spreads/totals.
    Requires the harness season-runs feed (else pythag is inert). ``weights`` = a
    custom grid (for a refined sweep); defaults to a coarse 0..1 grid. Judge OOS via
    --season/--seasons; nothing written."""
    _warn_small_limit(limit)
    import pricing_common
    live_ml = pricing_common._shrink_factor(sport_key, "moneyline")
    WEIGHTS = weights if weights else [0.0, 0.15, 0.25, 0.35, 0.50, 0.70, 1.0]
    SHRINKS = [(f"SHRINK {live_ml:.2f} (live)", live_ml), ("SHRINK 1.00 (off)", 1.0)]
    GATES = [
        ("edge>=5%", None, 0.0, 0.05),
        ("edge>=10%", None, 0.0, 0.10),
        ("EV>=8%&edge>=2%", 0.08, 0.02, 0.0),
        ("EV>=12%&edge>=3%", 0.12, 0.03, 0.0),
    ]

    results = {w: _regrade_ml_obs(sport_key, espn_sport, espn_league, season_year,
                                  limit, store_label, source, w) for w in WEIGHTS}

    print("\n\n############ PYTHAGOREAN WEIGHT x SHRINK — MONEYLINE ROI-at-gate ############")
    print("  Cell = flat-1u ROI%% (n bets). pythag baked per weight (re-grade); shrink")
    print("  applied offline. Best weight = highest ROI at a tight EV gate; watch")
    print("  whether the optimum shifts between the two shrink tables. Judge OOS.")
    for slabel, s in SHRINKS:
        print(f"\n=== {slabel} ===")
        print("  {:<7}".format("pythag")
              + "".join("{:>17}".format(g[0]) for g in GATES))
        for w in WEIGHTS:
            obs = results[w]
            cells = []
            for _label, evf, edf, leg in GATES:
                t = _team_gate_tally(obs, s, evf, edf, leg)
                roi = f"{t['roi'] * 100:+.1f}" if t["roi"] is not None else "-"
                cells.append(f"{roi}%({t['n']})")
            print("  {:<7}".format(f"{w:.2f}")
                  + "".join("{:>17}".format(c) for c in cells))
    print("\n  (Diagnostic only — nothing written. MONEYLINE only.)")


def pythag_shrink_combo(sport_key, espn_sport, espn_league, season_year=None,
                        limit=100000, store_label="", source="auto",
                        weights=None, shrinks=None):
    """MONEYLINE Pythagorean-weight x prob-shrink COMBO grid (NO WRITE): find the
    optimal marriage of the two ML mean-reversions.

    One re-grade per pythag weight (baked into the ML prob); the shrink axis is swept
    OFFLINE on the pre-shrink obs, so the FULL 2D grid costs only N re-grades. Prints
    a pythag(rows) x shrink(cols) ROI grid per tight EV gate and flags the max cell
    (the best combo). A diagonal ridge = SUBSTITUTES (more pythag compensates for
    less shrink); a single peak = COMPLEMENTS. MONEYLINE only. Judge OOS; no write."""
    _warn_small_limit(limit)
    PW = weights if weights else [0.0, 0.20, 0.35, 0.50, 0.70, 1.0]
    SH = shrinks if shrinks else [0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0]
    GATES = [("EV>=8% & edge>=2%", 0.08, 0.02, 0.0),
             ("EV>=12% & edge>=3%", 0.12, 0.03, 0.0)]
    obs_by_w = {w: _regrade_ml_obs(sport_key, espn_sport, espn_league, season_year,
                                   limit, store_label, source, w) for w in PW}

    print("\n\n############ PYTHAG x SHRINK COMBO — MONEYLINE ROI-at-gate ############")
    print("  Rows = pythag weight, cols = prob_shrink. Cell = flat-1u ROI%%. The MAX")
    print("  cell is the best marriage; a diagonal ridge (high-pythag/low-shrink ==")
    print("  low-pythag/high-shrink) means they are SUBSTITUTES. Judge OOS, not Brier.")
    for glabel, evf, edf, leg in GATES:
        print(f"\n=== {glabel} ===")
        print("  {:<8}".format("py\\shr")
              + "".join("{:>8}".format(f"{s:.2f}") for s in SH))
        best = None
        for w in PW:
            obs = obs_by_w[w]
            cells = []
            for s in SH:
                t = _team_gate_tally(obs, s, evf, edf, leg)
                roi = t["roi"]
                cells.append(f"{roi * 100:+.1f}" if roi is not None else "-")
                if roi is not None and (best is None or roi > best[0]):
                    best = (roi, w, s, t["n"])
            print("  {:<8}".format(f"{w:.2f}")
                  + "".join("{:>8}".format(c) for c in cells))
        if best:
            print(f"  -> best: pythag={best[1]:.2f} shrink={best[2]:.2f} "
                  f"ROI={best[0] * 100:+.1f}% (n={best[3]})")
    print("\n  (Diagnostic only — nothing written. MONEYLINE only; cell bet-counts")
    print("   vary across the grid — see --pythag-sweep for per-cell n.)")


def _ab_gate_tally(obs, a, b, ev_floor, edge_floor, legacy_threshold):
    """Flat-1u ROI over ML component obs under INDEPENDENT recency/pythag weights:
    final = 0.5 + a*(recency-0.5) + b*(pythag-0.5), clamped to [0,1]. obs =
    [(recency_p, pythag_p, market_p, outcome, price_yes, price_no)]; a None pythag_p
    (no season runs yet) is treated as neutral 0.5 so its term drops — matching the
    production skip. Backs the higher-edge side; gate is ROI-primary or legacy."""
    pnl = won = 0.0
    n = 0
    for rec, pyt, mkt, outcome, price_yes, price_no in obs:
        if mkt is None or rec is None:
            continue
        q = pyt if pyt is not None else 0.5
        p = 0.5 + a * (rec - 0.5) + b * (q - 0.5)
        p = max(0.0, min(1.0, p))
        if p >= mkt:
            side_prob, price, edge, win = p, price_yes, p - mkt, (outcome == 1)
        else:
            side_prob, price, edge, win = (1.0 - p, price_no, mkt - p,
                                           (outcome == 0))
        if price is None:
            continue
        dec = american_to_decimal(price)
        er = side_prob * dec - 1.0
        ok = ((er >= ev_floor and edge >= edge_floor) if ev_floor is not None
              else (edge >= legacy_threshold and er > 0))
        if not ok:
            continue
        pnl += (dec - 1.0) if win else -1.0
        won += 1.0 if win else 0.0
        n += 1
    return {"n": n, "pnl": pnl,
            "roi": (pnl / n) if n else None, "hit": (won / n) if n else None}


def _ab_returns(obs, a, b, ev_floor, edge_floor, legacy_threshold):
    """Per-bet profit-per-unit list under an (A,B) blend + gate — feeds the
    overfit-brake deflated_roi (which needs the raw return series, not aggregates)."""
    out = []
    for rec, pyt, mkt, outcome, price_yes, price_no in obs:
        if mkt is None or rec is None:
            continue
        q = pyt if pyt is not None else 0.5
        p = max(0.0, min(1.0, 0.5 + a * (rec - 0.5) + b * (q - 0.5)))
        if p >= mkt:
            side_prob, price, edge, win = p, price_yes, p - mkt, (outcome == 1)
        else:
            side_prob, price, edge, win = 1.0 - p, price_no, mkt - p, (outcome == 0)
        if price is None:
            continue
        dec = american_to_decimal(price)
        er = side_prob * dec - 1.0
        ok = ((er >= ev_floor and edge >= edge_floor) if ev_floor is not None
              else (edge >= legacy_threshold and er > 0))
        if ok:
            out.append((dec - 1.0) if win else -1.0)
    return out


def _pbo_over_ab_grid(obs, a_weights, b_weights, gate, n_blocks=10):
    """Build the CSCV perf matrix (chronological block x A*B config, cell = block
    P/L) over the A x B grid at one gate, and return overfit_stats.pbo_cscv(...).

    ``obs`` is appended in date order by the backtest, so contiguous index-slices
    are chronological blocks — no date field needed. Returns None if too thin."""
    import overfit_stats
    _, evf, edf, leg = gate
    n = len(obs)
    if n < n_blocks * 4:
        return None
    size = n // n_blocks
    blocks = [obs[i * size:(i + 1) * size] for i in range(n_blocks - 1)]
    blocks.append(obs[(n_blocks - 1) * size:])   # remainder into the last block
    cells = [(a, b) for a in a_weights for b in b_weights]
    matrix = [[_ab_gate_tally(blk, a, b, evf, edf, leg)["pnl"] for (a, b) in cells]
              for blk in blocks]
    return overfit_stats.pbo_cscv(matrix)


def _append_experiment(record):
    """Append-only experiment registry (mining idea #2): one JSON line per selection
    run so every sweep/OOS decision is auditable. Best-effort; never raises."""
    try:
        import json
        from datetime import datetime, timezone
        record = dict(record)
        record["ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with open("experiment_registry.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


def ab_sweep(sport_key, espn_sport, espn_league, season_year=None, limit=100000,
             store_label="", source="auto", a_weights=None, b_weights=None):
    """MONEYLINE independent recency-weight (A) x pythag-weight (B) ROI grid (NO WRITE).

    final = 0.5 + A*(recency-0.5) + B*(pythag-0.5). This is the (pythag w, shrink s)
    combo REPARAMETRIZED onto independent axes (the current order gives A=s(1-w),
    B=s*w, which couples them; here A and B move freely). Costs ONE re-grade: pythag
    is forced to 0 so the collected ML prob IS the pure recency prob, and the raw
    per-game pythag prob is captured alongside it — the whole A x B grid is then
    offline. Reads directly as 'trust recent margins this much, trust season run-
    differential that much'. Judge OOS; nothing written."""
    _warn_small_limit(limit)
    A_W = a_weights if a_weights else [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]
    B_W = b_weights if b_weights else [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80]
    GATES = [("EV>=8% & edge>=2%", 0.08, 0.02, 0.0),
             ("EV>=12% & edge>=3%", 0.12, 0.03, 0.0)]

    obs = _regrade_ml_components(sport_key, espn_sport, espn_league, season_year,
                                 limit, store_label, source)
    n_pyth = sum(1 for r in obs if r[1] is not None)
    print(f"\n  captured {len(obs)} ML obs ({n_pyth} with a pythag prob).")

    print("\n\n############ A (recency) x B (pythag) — MONEYLINE ROI-at-gate ############")
    print("  final = 0.5 + A*(recency-0.5) + B*(pythag-0.5). Rows = A (recency trust),")
    print("  cols = B (pythag trust) — INDEPENDENT axes. Cell = flat-1u ROI%%; best")
    print("  cell flagged. A rising-B ridge = run-differential carries the signal.")
    print("  Judge OOS, not Brier.")
    for glabel, evf, edf, leg in GATES:
        print(f"\n=== {glabel} ===")
        print("  {:<7}".format("A\\B")
              + "".join("{:>8}".format(f"{b:.2f}") for b in B_W))
        best = None
        for a in A_W:
            cells = []
            for b in B_W:
                t = _ab_gate_tally(obs, a, b, evf, edf, leg)
                roi = t["roi"]
                cells.append(f"{roi * 100:+.1f}" if roi is not None else "-")
                if roi is not None and (best is None or roi > best[0]):
                    best = (roi, a, b, t["n"])
            print("  {:<7}".format(f"{a:.2f}")
                  + "".join("{:>8}".format(c) for c in cells))
        if best:
            print(f"  -> best: A(recency)={best[1]:.2f} B(pythag)={best[2]:.2f} "
                  f"ROI={best[0] * 100:+.1f}% (n={best[3]})")
            # Overfit brake (idea #2): is this best-of-N in-sample cell more than
            # search noise? deflated_prob << the ROI suggests => yes, it's noise.
            import overfit_stats
            _dfl = overfit_stats.deflated_roi(
                _ab_returns(obs, best[1], best[2], evf, edf, leg), len(A_W) * len(B_W))
            if _dfl:
                print(f"     haircut (best of {len(A_W)*len(B_W)}): deflated P(edge "
                      f"real)={_dfl['deflated_prob']*100:.0f}% "
                      f"[{'CREDIBLE' if _dfl['credible'] else 'LIKELY NOISE'}] — "
                      f"confirm with --oos-ab before trusting.")
    print("\n  (Diagnostic only — nothing written. MONEYLINE only. In-sample: the")
    print("   'best cell' is a best-of-many pick — see the haircut, then --oos-ab.)")


def oos_ab(sport_key, espn_sport, espn_league, train_seasons, test_seasons,
           limit=100000, store_label="", source="auto", a_weights=None,
           b_weights=None):
    """OUT-OF-SAMPLE MONEYLINE validation (NO WRITE): pick the A(recency) x B(pythag)
    blend on a TRAIN window, then measure it — plus a GATE-tightening ladder — on a
    disjoint TEST window it never saw. This is the honest antidote to the in-sample
    argmax: the sweeps optimize over ~100 cells on one window (winner's curse), so
    their peak overstates real ROI. Here the config is FIXED from train and the test
    number is untouched. Answers 'does the pythag/shrink finding hold?' AND 'do we
    need tighter gates?' (the ladder) in one pass. Judge the TEST column; the
    train-vs-test gap at the selected cell is the overfit tax."""
    _warn_small_limit(limit)
    A_W = a_weights if a_weights else [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    B_W = b_weights if b_weights else [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]
    # Gate used to PICK the best (A,B) on train (fixed, sensible; not itself swept):
    SELECT = ("EV>=12% & edge>=3%", 0.12, 0.03, 0.0)
    # Gate-tightening ladder measured at the fixed (A,B) — EV gates AND pure-edge
    # gates (edge gates need a big prob disagreement regardless of price, so they
    # don't preferentially pass longshots the way EV gates do).
    LADDER = [
        ("edge>=5%", None, 0.0, 0.05), ("edge>=7%", None, 0.0, 0.07),
        ("edge>=10%", None, 0.0, 0.10), ("edge>=12%", None, 0.0, 0.12),
        ("EV>=8% & edge>=2%", 0.08, 0.02, 0.0),
        ("EV>=12% & edge>=3%", 0.12, 0.03, 0.0),
        ("EV>=16% & edge>=4%", 0.16, 0.04, 0.0),
        ("EV>=20% & edge>=5%", 0.20, 0.05, 0.0),
        ("EV>=25% & edge>=6%", 0.25, 0.06, 0.0),
    ]
    train = _regrade_ml_components(sport_key, espn_sport, espn_league, train_seasons,
                                   limit, store_label, source, tag="TRAIN")
    test = _regrade_ml_components(sport_key, espn_sport, espn_league, test_seasons,
                                  limit, store_label, source, tag="TEST")

    # Select best (A,B) on TRAIN at the SELECT gate.
    best = None
    for a in A_W:
        for b in B_W:
            t = _ab_gate_tally(train, a, b, SELECT[1], SELECT[2], SELECT[3])
            if t["roi"] is not None and (best is None or t["roi"] > best[0]):
                best = (t["roi"], a, b, t["n"])
    # Test-optimal at the SELECT gate (the ceiling test COULD hit) — for the gap.
    test_ceil = None
    for a in A_W:
        for b in B_W:
            t = _ab_gate_tally(test, a, b, SELECT[1], SELECT[2], SELECT[3])
            if t["roi"] is not None and (test_ceil is None or t["roi"] > test_ceil[0]):
                test_ceil = (t["roi"], a, b, t["n"])

    print("\n\n############ OOS MONEYLINE — train-selected A x B, tested out-of-sample "
          "############")
    print(f"  TRAIN seasons={train_seasons}  ({len(train)} ML obs)")
    print(f"  TEST  seasons={test_seasons}  ({len(test)} ML obs)")
    if not best:
        print("  (train produced no qualifying cell — widen the window/grid)")
        return
    A, B = best[1], best[2]
    tr_sel = _ab_gate_tally(train, A, B, SELECT[1], SELECT[2], SELECT[3])
    te_sel = _ab_gate_tally(test, A, B, SELECT[1], SELECT[2], SELECT[3])
    print(f"\n  Selected on TRAIN @ {SELECT[0]}:  A(recency)={A:.2f}  B(pythag)={B:.2f}")
    print(f"    train ROI {tr_sel['roi']*100:+.1f}% (n={tr_sel['n']})   "
          f"->  TEST ROI {te_sel['roi']*100:+.1f}% (n={te_sel['n']})"
          if te_sel['roi'] is not None else "    (thin test)")
    if test_ceil:
        print(f"    overfit tax: test-OPTIMAL cell would be A={test_ceil[1]:.2f} "
              f"B={test_ceil[2]:.2f} @ {test_ceil[0]*100:+.1f}% -- the gap vs our "
              f"{te_sel['roi']*100:+.1f}% is what in-sample tuning can't capture.")

    # OVERFIT BRAKES (mining idea #2): was the TRAIN selection (best of N=|grid|
    # cells) more than noise, and does the selection procedure itself overfit?
    import overfit_stats
    ncfg = len(A_W) * len(B_W)
    dfl = overfit_stats.deflated_roi(
        _ab_returns(train, A, B, SELECT[1], SELECT[2], SELECT[3]), ncfg)
    pbo = _pbo_over_ab_grid(train, A_W, B_W, SELECT)
    print(f"\n  OVERFIT BRAKES (train selection was best of {ncfg} A x B cells):")
    if dfl:
        print(f"    haircut: train t-stat {dfl['t_stat']:.2f} vs multiple-testing bar "
              f"{dfl['noise_bar']:.2f} -> deflated P(edge real) "
              f"{dfl['deflated_prob']*100:.0f}%  "
              f"[{'CREDIBLE' if dfl['credible'] else 'LIKELY NOISE'}]")
    if pbo:
        print(f"    PBO (prob the A x B selection overfits, CSCV over "
              f"{pbo['n_splits']} splits x {pbo['n_blocks']} blocks): "
              f"{pbo['pbo']*100:.0f}%  [{'OK' if pbo['pbo'] < 0.5 else 'HIGH'}]")
    print("    (Low deflated-prob / high PBO => the train pick is search noise; the "
          "TEST column is the truth.)")
    _append_experiment({
        "tool": "oos_ab", "sport": sport_key,
        "train_seasons": str(train_seasons), "test_seasons": str(test_seasons),
        "select_gate": SELECT[0], "n_configs": ncfg,
        "selected_A": A, "selected_B": B,
        "train_roi": tr_sel["roi"], "test_roi": te_sel["roi"],
        "deflated_prob": (dfl or {}).get("deflated_prob"),
        "pbo": (pbo or {}).get("pbo"),
    })

    print(f"\n  GATE LADDER at the FIXED train-selected (A={A:.2f}, B={B:.2f}) — does "
          "tightening help OOS?")
    n_tr = len(train) or 1
    n_te = len(test) or 1
    print("  {:<20}{:>20}{:>20}".format("gate", "TRAIN n/%bet/ROI", "TEST n/%bet/ROI"))
    for label, evf, edf, leg in LADDER:
        tt = _ab_gate_tally(train, A, B, evf, edf, leg)
        te = _ab_gate_tally(test, A, B, evf, edf, leg)
        tr_s = (f"{tt['n']}/{100*tt['n']/n_tr:.0f}%/{tt['roi']*100:+.1f}"
                if tt['roi'] is not None else f"{tt['n']}/-/-")
        te_s = (f"{te['n']}/{100*te['n']/n_te:.0f}%/{te['roi']*100:+.1f}"
                if te['roi'] is not None else f"{te['n']}/-/-")
        print("  {:<20}{:>20}{:>20}".format(label, tr_s, te_s))
    print("\n  Read the TEST column: if ROI CLIMBS as the gate tightens, tighter gates")
    print("  help (fewer, better bets); if it flattens/falls into tiny n, they don't.")
    print("  (Diagnostic only — nothing written. MONEYLINE only.)")


def spread_dispersion_sweep(sport_key, espn_sport, espn_league, season_year=None,
                            limit=100000, store_label="", source="auto",
                            dispersions=None):
    """SPREADS overdispersion sweep (mining idea #5, NO WRITE): re-grade the run-
    line challenger across NegBin dispersion d (variance = mean + d*mean**2; d=0 =
    today's Poisson, byte-identical), and report per-d the spread calibration slope
    (does widening the run distribution fix over-confidence? slope -> 1) + Brier-
    skill-vs-market + ROI-at-gate. INERT until you set analysis.DEFAULT_SPREAD_
    DISPERSION live; judge here first. One re-grade per d (challenger is baked)."""
    _warn_small_limit(limit)
    import prob_metrics
    DS = dispersions if dispersions else [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    GATES = [("edge>=5%", None, 0.0, 0.05), ("edge>=10%", None, 0.0, 0.10),
             ("EV>=8%&edge>=2%", 0.08, 0.02, 0.0),
             ("EV>=12%&edge>=3%", 0.12, 0.03, 0.0)]
    saved = analysis.DEFAULT_SPREAD_DISPERSION
    results = {}
    try:
        for d in DS:
            analysis.DEFAULT_SPREAD_DISPERSION = d
            obs = {m: [] for m in MARKETS}
            print(f"\n########## re-grade: spread dispersion d={d:.2f} ##########")
            run_odds_backtest(
                sport_key, espn_sport, espn_league, limit=limit, window=10,
                variants={"live": VARIANT_PRESETS.get("all", {})},
                season_year=season_year, threshold_pct=5.0,
                write_calibration=False, store_label=store_label,
                engine="live", prob_shrink=1.0, source=source,
                supplement_log=False, collect_obs=obs)
            results[d] = obs["spreads"]
    finally:
        analysis.DEFAULT_SPREAD_DISPERSION = saved

    print("\n\n############ SPREADS overdispersion sweep (mining #5) ############")
    print("  d=0 is today's Poisson. calibSlope<1 = overconfident (widening d should")
    print("  raise it toward 1); BSS>0 = beat the close; ROI at closing prices.")
    print("  {:<6}{:>7}{:>11}{:>8}".format("d", "n", "calibSlope", "BSS")
          + "".join("{:>16}".format(g[0]) for g in GATES))
    for d in DS:
        obs = results[d]
        rows = [(o[0], o[1], o[2]) for o in obs if o[1] is not None]
        if len(rows) < 50:
            print(f"  {d:<6.2f}(thin {len(rows)})")
            continue
        probs = [r[0] for r in rows]
        refs = [r[1] for r in rows]
        ys = [r[2] for r in rows]
        cs = prob_metrics.calibration_slope(probs, ys)
        bss = prob_metrics.brier_skill_score(probs, ys, refs)
        cells = []
        for _, evf, edf, leg in GATES:
            t = _team_gate_tally(obs, 1.0, evf, edf, leg)
            cells.append(f"{t['roi']*100:+.1f}%({t['n']})" if t["roi"] is not None
                         else "-")
        print("  {:<6.2f}{:>7}{:>11}{:>8}".format(
            d, len(rows),
            f"{cs['slope']:.2f}" if cs else "-",
            f"{bss*100:+.1f}%" if bss is not None else "-")
            + "".join("{:>16}".format(c) for c in cells))
    print("\n  (Diagnostic only — nothing written. SPREADS only. If a d>0 lifts the")
    print("   calib slope toward 1 AND holds ROI, backtest-confirm then set live.)")


def pythag_residual_sweep(sport_key, espn_sport, espn_league, season_year=None,
                          limit=100000, store_label="", source="auto", weights=None):
    """MONEYLINE Pythagorean-RESIDUAL contrarian-weight sweep (mining idea #29, NO
    WRITE). Fades over-performers (actual win% > pythag win%). Re-grade per weight
    (residual baked into the ML prob, on top of the live pythag 0.35), then report
    per weight: calibration slope + Brier-skill-vs-market + ROI-at-gate at the LIVE
    ML shrink. INERT until analysis.DEFAULT_PYTHAG_RESIDUAL_WEIGHT is set live —
    judge here first. Unlike the shrink family, this is a directional inefficiency
    bet, so watch ROI (not just calibration)."""
    _warn_small_limit(limit)
    import prob_metrics
    import pricing_common
    W = weights if weights else [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
    live_shrink = pricing_common._shrink_factor(sport_key, "moneyline")
    GATES = [("edge>=5%", None, 0.0, 0.05), ("edge>=10%", None, 0.0, 0.10),
             ("EV>=8%&edge>=2%", 0.08, 0.02, 0.0),
             ("EV>=12%&edge>=3%", 0.12, 0.03, 0.0)]
    saved = analysis.DEFAULT_PYTHAG_RESIDUAL_WEIGHT
    results = {}
    try:
        for w in W:
            analysis.DEFAULT_PYTHAG_RESIDUAL_WEIGHT = w
            obs = {m: [] for m in MARKETS}
            print(f"\n########## re-grade: pythag-residual w={w:.2f} ##########")
            run_odds_backtest(
                sport_key, espn_sport, espn_league, limit=limit, window=10,
                variants={"live": VARIANT_PRESETS.get("all", {})},
                season_year=season_year, threshold_pct=5.0,
                write_calibration=False, store_label=store_label,
                engine="live", prob_shrink=1.0, source=source,
                supplement_log=False, collect_obs=obs)
            results[w] = obs["moneyline"]
    finally:
        analysis.DEFAULT_PYTHAG_RESIDUAL_WEIGHT = saved

    print("\n\n############ MONEYLINE pythag-residual sweep (mining #29) ############")
    print(f"  Fade over-performers. ROI at the live ML shrink ({live_shrink:.2f}); "
          "calibSlope<1=overconfident; BSS>0=beat close.")
    print("  {:<6}{:>7}{:>11}{:>8}".format("w", "n", "calibSlope", "BSS")
          + "".join("{:>16}".format(g[0]) for g in GATES))
    for w in W:
        obs = results[w]
        rows = [(o[0], o[1], o[2]) for o in obs if o[1] is not None]
        if len(rows) < 50:
            print(f"  {w:<6.2f}(thin {len(rows)})")
            continue
        probs = [r[0] for r in rows]
        refs = [r[1] for r in rows]
        ys = [r[2] for r in rows]
        cs = prob_metrics.calibration_slope(probs, ys)
        bss = prob_metrics.brier_skill_score(probs, ys, refs)
        cells = []
        for _, evf, edf, leg in GATES:
            t = _team_gate_tally(obs, live_shrink, evf, edf, leg)
            cells.append(f"{t['roi']*100:+.1f}%({t['n']})" if t["roi"] is not None
                         else "-")
        print("  {:<6.2f}{:>7}{:>11}{:>8}".format(
            w, len(rows),
            f"{cs['slope']:.2f}" if cs else "-",
            f"{bss*100:+.1f}%" if bss is not None else "-")
            + "".join("{:>16}".format(c) for c in cells))
    print("\n  (Diagnostic only — nothing written. MONEYLINE only. w=0 = today. A w>0")
    print("   that LIFTS ROI at a gate is a real edge — confirm OOS via train/test.)")


def main():
    p = argparse.ArgumentParser(description="Backtest the sportsbook projection model")
    p.add_argument("--mode",
                   choices=["matchup", "props", "odds", "props-odds", "coherence",
                            "prop-lag"],
                   default="matchup",
                   help="matchup = team-level projections; props = player-prop "
                        "projections; odds = grade team markets vs stored closing "
                        "lines; props-odds = grade player props vs stored closing "
                        "lines (ROI + model⇄market blend)")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="(odds mode) Min edge %% over the de-vigged market to place a bet.")
    p.add_argument("--write-calibration", action="store_true",
                   help="(odds mode) Save the best model⇄market blend weight per "
                        "market to calibration/<sport>.json for live use. Stages a "
                        "candidate by default (promote via refit_calibration.py "
                        "--promote); pass --live to write the live file directly.")
    p.add_argument("--live", action="store_true",
                   help="Write calibration straight to the LIVE file, skipping the "
                        "candidate staging that --write-calibration uses by default.")
    p.add_argument("--team-gate-sweep", action="store_true",
                   help="(odds mode) Team-market gate x shrink lens: grade ML/"
                        "spread/total on the closing-line holdout, then sweep "
                        "probability-shrink x recommendation-gate and report "
                        "realized flat-1u ROI per combo (finds whether the live "
                        "shrink+edge gate over-suppress moneyline). No write.")
    p.add_argument("--topn-sweep", action="store_true",
                   help="(odds mode, Batch B1) TOP-N/DAY portfolio test: the real-"
                        "world ROI — each day keep only the best N value bets "
                        "(EV-ranked, moneyline edge>=5%% + spreads high-conviction "
                        "edge>=10%%, totals off), flat-1u, swept N=5/10/15/all for the "
                        "full policy AND moneyline-only. Shows if tightening to your "
                        "real ~10 bets/day helps. No write.")
    p.add_argument("--sizing-sweep", action="store_true",
                   help="(odds mode, Batch B1) MONEYLINE bankroll sim: flat-1u vs "
                        "fractional-Kelly vs uncertainty-Kelly (size off the win-prob "
                        "interval's low bound + abstain when it spans break-even) at "
                        "the ROI-optimal config (shrink 0.25, edge>=5%%). Reports "
                        "growth / max-drawdown / Sharpe so risk-adjusted sizing is "
                        "visible (flat-1u ROI can't show it). No write.")
    p.add_argument("--unleash-sweep", action="store_true",
                   help="(odds mode) UNLEASH sweep: re-grade the LIVE pipeline with "
                        "each market-anchoring knob released one at a time "
                        "(prob_shrink ML/spreads/totals + Pythagorean), and print "
                        "ROI-at-gate LIVE vs UNLEASHED. Spreads/pythag are true "
                        "override re-grades. Judge OOS (--season/--seasons). No write.")
    p.add_argument("--pythag-sweep", action="store_true",
                   help="(odds mode) Sweep DEFAULT_PYTHAG_WEIGHT x prob_shrink for "
                        "MONEYLINE: re-grade per pythag weight, tally ROI-at-gate at "
                        "shrink on AND off. Finds the best pythag weight + whether it "
                        "shifts with shrink. Judge OOS (--season/--seasons). No write.")
    p.add_argument("--combo-sweep", action="store_true",
                   help="(odds mode) MONEYLINE pythag-weight x prob-shrink 2D ROI-at-"
                        "gate grid (the 'optimal marriage'). One re-grade per pythag "
                        "weight; shrink swept offline. Flags the best cell. No write.")
    p.add_argument("--pythag-weights", default=None,
                   help="(--pythag-sweep/--combo-sweep) comma list of pythag weights, "
                        "e.g. 0.35,0.4,0.45,0.5,0.55,0.6. Default: coarse 0..1 grid.")
    p.add_argument("--combo-shrinks", default=None,
                   help="(--combo-sweep) comma list of prob_shrink values for the grid "
                        "columns, e.g. 0.1,0.2,0.3,0.4. Default: 0.10..1.0 grid.")
    p.add_argument("--ab-sweep", action="store_true",
                   help="(odds mode) MONEYLINE independent recency-weight A x pythag-"
                        "weight B ROI grid: final = 0.5 + A*(recency-0.5) + "
                        "B*(pythag-0.5). One re-grade (pythag forced to 0); A x B swept "
                        "offline. The decoupled view of --combo-sweep. No write.")
    p.add_argument("--a-weights", default=None,
                   help="(--ab-sweep/--oos-ab) comma list of A (recency) weights.")
    p.add_argument("--b-weights", default=None,
                   help="(--ab-sweep/--oos-ab) comma list of B (pythag) weights.")
    p.add_argument("--oos-ab", action="store_true",
                   help="(odds mode) OUT-OF-SAMPLE ML validation: pick the A x B blend "
                        "on --train-seasons, then measure it + a gate-tightening ladder "
                        "on --test-seasons (never seen). The honest antidote to the "
                        "in-sample sweep argmax. No write.")
    p.add_argument("--train-seasons", default=None,
                   help="(--oos-ab) seasons to SELECT the A x B blend on, e.g. 2023,2024.")
    p.add_argument("--test-seasons", default=None,
                   help="(--oos-ab) disjoint seasons to TEST on, e.g. 2025,2026.")
    p.add_argument("--spread-dispersion-sweep", action="store_true",
                   help="(odds mode) SPREADS NegBin overdispersion sweep: re-grade the "
                        "run-line challenger across dispersion d (0=Poisson), report "
                        "calibration slope + BSS + ROI-at-gate per d. INERT until set "
                        "live. No write.")
    p.add_argument("--dispersions", default=None,
                   help="(--spread-dispersion-sweep) comma list, e.g. 0,0.05,0.1,0.2.")
    p.add_argument("--pythag-residual-sweep", action="store_true",
                   help="(odds mode) MONEYLINE Pythagorean-residual contrarian sweep "
                        "(fade over-performers): re-grade per weight, report calib "
                        "slope + BSS + ROI-at-gate. INERT until set live. No write.")
    p.add_argument("--residual-weights", default=None,
                   help="(--pythag-residual-sweep) comma list, e.g. 0,0.1,0.2,0.3,0.5.")
    p.add_argument("--recency-sweep", action="store_true",
                   help="(--mode props) STEP-1 joint recent_n x half_life sweep to "
                        "validate the never-swept lookback window + decay defaults. "
                        "Isolated axes (all other knobs off); incumbent cell n20/none. "
                        "Run with --calibrate to score per-prop OOS Brier. No write.")
    p.add_argument("--recent-ns", default=None,
                   help="(--recency-sweep) comma list of lookback windows, e.g. "
                        "10,15,20,25,30,40 ('full'=entire history).")
    p.add_argument("--half-lives", default=None,
                   help="(--recency-sweep) comma list of decay half-lives, e.g. "
                        "none,5,7,10,15 ('none'=decay off / equal weight).")
    p.add_argument("--weather-sweep", action="store_true",
                   help="(--mode props) FIT the moist-air weather model: a "
                        "density_coef x wind_coef x strength sweep for batter_hits + "
                        "pitcher_earned_runs. Incumbent cell wx_off. Run with "
                        "--calibrate for per-prop OOS Brier. No write.")
    p.add_argument("--density-coefs", default=None,
                   help="(--weather-sweep) comma list, e.g. 0.5,1.0,1.5,2.0.")
    p.add_argument("--wind-coefs", default=None,
                   help="(--weather-sweep) comma list, e.g. 0,0.003,0.006.")
    p.add_argument("--strengths", default=None,
                   help="(--weather-sweep) comma list, e.g. 0.5,1.0.")
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
    p.add_argument("--fit-shares", action="store_true",
                   help="(odds mode, live engine, MLB, with --write-calibration) Fit "
                        "the SPREADS ensemble stack end-to-end from raw components — "
                        "challenger spread_share (recency<->additive cover), plus the "
                        "spreads prob_shrink and market blend — in serve order. Owns "
                        "the spreads market: the generic shrink/blend fitters skip it "
                        "(their composite cover would double-count). Requires the "
                        "additive expected-runs projection to fire (data complete).")
    p.add_argument("--fit-blend", action="store_true",
                   help="(odds mode, live engine, with --write-calibration) After "
                        "fitting prob_shrink on the raw model prob, ALSO fit + persist "
                        "the model<->market blend weight per market — fit on the "
                        "shrunk prob (serve order: shrink then blend), so one raw pass "
                        "produces both corrections. Without this, only prob_shrink is "
                        "written (current default).")
    p.add_argument("--serve-mode", action="store_true",
                   help="(odds mode, live engine) Grade the SERVED probabilities — "
                        "per-market prob_shrink + model<->market blend applied INSIDE "
                        "the analyzers — instead of the raw model prob. This is the "
                        "holdout-VALIDATION pass: it grades exactly what production "
                        "would serve under the live (or --staging candidate) "
                        "calibration. Mutually exclusive with --write-calibration "
                        "(fitting needs the raw prob).")
    p.add_argument("--cross-season", choices=["strict", "all"], default="strict",
                   help="(props mode) 'strict' (default) keeps only current-season "
                        "prior games per test observation; 'all' uses the full "
                        "ESPN gamelog regardless of season boundary.")
    p.add_argument("--source", choices=["auto", "warehouse", "store"],
                   default="auto",
                   help="(odds / props-odds mode) Odds source: 'auto' (default) uses "
                        "the Azure odds warehouse when it's configured and has "
                        "games, else the local historical_odds JSON; "
                        "'warehouse' forces the warehouse; 'store' forces the "
                        "local JSON. Under 'auto', a --store-label forces the "
                        "local JSON (the warehouse has no label concept). NOTE the "
                        "local JSON has historically held TEAM markets only, so "
                        "props-odds needs --source warehouse.")
    p.add_argument("--snapshot",
                   choices=["close", "closing", "early_12h", "early_4h"],
                   default="close",
                   help="(odds / props-odds mode, --source warehouse) Which "
                        "warehoused snapshot to grade: 'close' (default, nearest-"
                        "pre-commence = CLV reference) or an exact precise-backfill "
                        "window by source — 'closing', 'early_4h', or 'early_12h' "
                        "(team only; props have no 12h snapshot).")
    p.add_argument("--xstats-strength", type=float, default=None,
                   help="(props-odds mode) Override the xBA blend weight for xstats "
                        "props (batter_hits) — 0 turns xBA OFF, 0.75 = the shipped "
                        "value. Lets you A/B whether the xBA blend helps or HURTS vs "
                        "real odds. Default None = use the calibration file's value.")
    p.add_argument("--walk-forward", action="store_true",
                   help="(props-odds mode) Simulate the deep-seed + online CURRENT-"
                        "SEASON Platt architecture: per (prop,bucket) refit a loop Platt "
                        "on current-season obs so far (reset each season), blend with the "
                        "4-season seed, grade forward. Compares STATIC vs WALK-FORWARD "
                        "Brier + daily top-N ROI (tests whether within-season adaptation "
                        "helps). Slower — re-fits Platt repeatedly.")
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
    p.add_argument("--lineup-weight", type=float, default=None,
                   help="(odds mode, EXPERIMENT) Override analysis.DEFAULT_LINEUP_"
                        "WEIGHT for this run — the runs-of-margin shift per unit of"
                        " lineup-offense edge (today's 9 batters' as-of OPS). Use to"
                        " sweep whether the lineup input helps, e.g. --lineup-weight"
                        " 3. Default None keeps the inert 0.0.")
    args = p.parse_args()

    if getattr(args, "serve_mode", False):
        if args.write_calibration:
            p.error("--serve-mode grades the served (shrunk+blended) probs and "
                    "cannot also --write-calibration (fitting needs the raw model "
                    "prob). Fit on a raw pass, then validate with --serve-mode.")
        if args.engine != "live":
            p.error("--serve-mode requires --engine live.")
    if getattr(args, "fit_shares", False) or getattr(args, "fit_blend", False):
        if not args.write_calibration:
            p.error("--fit-shares / --fit-blend only apply with --write-calibration.")
        if args.engine != "live":
            p.error("--fit-shares / --fit-blend require --engine live.")

    # EXPERIMENT hook: activate the lineup-offense margin shift for THIS backtest
    # run only (live stays inert until fit + wired). Set before any analyzer runs.
    if args.lineup_weight is not None:
        import analysis
        analysis.DEFAULT_LINEUP_WEIGHT = args.lineup_weight
        print(f"[experiment] analysis.DEFAULT_LINEUP_WEIGHT = {args.lineup_weight}")

    # Default-safe: calibration writes stage a candidate (never the live file the
    # app serves) unless --live is passed. Promotion is a separate, explicit step
    # (refit_calibration.py --promote). Setting the mode is harmless when no write
    # occurs (it only affects the save_* helpers).
    staging = not args.live
    set_candidate_mode(staging)
    # Also grade the STAGED team-market calibration (additive / prob_shrink / blend /
    # challenger) when staging — so a candidate can be validated on a holdout without
    # promoting. --live reads the live file. Separate from write-staging so the refit +
    # live app (which never enable this) keep serving live mid-staging.
    set_serving_candidate(staging)
    if staging and args.write_calibration:
        _notice = existing_candidate_notice(SPORT_MAP[args.sport][2])
        if _notice:
            print(_notice)

    # Target the Azure SQL warehouse/logs when the SQL_* secrets are configured
    # (outside Streamlit they aren't in the env yet). Guarded; a no-op when SQL
    # isn't configured so --source auto/store still works offline.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]

    if getattr(args, "recency_sweep", False):
        if args.mode != "props":
            print("--recency-sweep is only valid with --mode props.")
            sys.exit(1)

        def _parse_grid(raw, none_words):
            if not raw:
                return None
            out = []
            for tok in raw.split(","):
                t = tok.strip()
                if not t:
                    continue
                out.append(None if t.lower() in none_words else int(t))
            return out or None

        rns = _parse_grid(args.recent_ns, {"full", "none", "all"})
        hls = _parse_grid(args.half_lives, {"none", "null", "off"})
        variants = _build_recency_sweep_grid(rns, hls)
        args.sweep = True   # reuse the sweep tabulator + code path
        args.calibrate = True   # STEP-1 selects on OOS Brier, not just MAE
        variant_names = list(variants.keys())
        print(f"\n{'#'*60}")
        print(f"#  RECENCY SWEEP: {len(variants)} recent_n x half_life combos "
              f"(--calibrate forced for per-prop Brier)")
    elif getattr(args, "weather_sweep", False):
        if args.mode != "props":
            print("--weather-sweep is only valid with --mode props.")
            sys.exit(1)

        def _floats(raw):
            return ([float(x) for x in raw.split(",") if x.strip()]
                    if raw else None)

        variants = _build_weather_sweep_grid(
            _floats(args.density_coefs), _floats(args.wind_coefs),
            _floats(args.strengths))
        args.sweep = True        # reuse the sweep tabulator + code path
        args.calibrate = True    # weather is selected on per-prop OOS Brier
        variant_names = list(variants.keys())
        print(f"\n{'#'*60}")
        print(f"#  WEATHER SWEEP: {len(variants)} density x wind x strength combos "
              f"(--calibrate forced for per-prop Brier)")
    elif args.sweep:
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
                                write_calibration=args.write_calibration,
                                source=args.source, snapshot=args.snapshot,
                                seasons=(_parse_seasons(args.seasons)
                                         if args.seasons else None),
                                xstats_override=args.xstats_strength,
                                walk_forward=args.walk_forward)
    elif args.mode == "odds":
        odds_seasons = _parse_seasons(args.seasons) if args.seasons else args.season
        if args.team_gate_sweep:
            diagnose_team_gate(sport_key, espn_sport, espn_league,
                               season_year=odds_seasons, limit=args.limit,
                               store_label=args.store_label, source=args.source,
                               snapshot=args.snapshot)
        elif args.sizing_sweep:
            sizing_sweep(sport_key, espn_sport, espn_league,
                         season_year=odds_seasons, limit=args.limit,
                         store_label=args.store_label, source=args.source)
        elif args.topn_sweep:
            top_n_sweep(sport_key, espn_sport, espn_league,
                        season_year=odds_seasons, limit=args.limit,
                        store_label=args.store_label, source=args.source)
        elif args.unleash_sweep:
            unleash_sweep(sport_key, espn_sport, espn_league,
                          season_year=odds_seasons, limit=args.limit,
                          store_label=args.store_label, source=args.source)
        elif args.pythag_sweep:
            _pw = ([float(x) for x in args.pythag_weights.split(",")]
                   if args.pythag_weights else None)
            pythag_sweep(sport_key, espn_sport, espn_league,
                         season_year=odds_seasons, limit=args.limit,
                         store_label=args.store_label, source=args.source,
                         weights=_pw)
        elif args.combo_sweep:
            _pw = ([float(x) for x in args.pythag_weights.split(",")]
                   if args.pythag_weights else None)
            _sh = ([float(x) for x in args.combo_shrinks.split(",")]
                   if args.combo_shrinks else None)
            pythag_shrink_combo(sport_key, espn_sport, espn_league,
                                season_year=odds_seasons, limit=args.limit,
                                store_label=args.store_label, source=args.source,
                                weights=_pw, shrinks=_sh)
        elif args.ab_sweep:
            _aw = ([float(x) for x in args.a_weights.split(",")]
                   if args.a_weights else None)
            _bw = ([float(x) for x in args.b_weights.split(",")]
                   if args.b_weights else None)
            ab_sweep(sport_key, espn_sport, espn_league,
                     season_year=odds_seasons, limit=args.limit,
                     store_label=args.store_label, source=args.source,
                     a_weights=_aw, b_weights=_bw)
        elif args.oos_ab:
            _aw = ([float(x) for x in args.a_weights.split(",")]
                   if args.a_weights else None)
            _bw = ([float(x) for x in args.b_weights.split(",")]
                   if args.b_weights else None)
            _tr = _parse_seasons(args.train_seasons) if args.train_seasons else None
            _te = _parse_seasons(args.test_seasons) if args.test_seasons else odds_seasons
            if not _tr:
                print("--oos-ab requires --train-seasons (e.g. 2023,2024) and "
                      "--test-seasons (e.g. 2025,2026).")
            else:
                oos_ab(sport_key, espn_sport, espn_league, _tr, _te,
                       limit=args.limit, store_label=args.store_label,
                       source=args.source, a_weights=_aw, b_weights=_bw)
        elif args.spread_dispersion_sweep:
            _ds = ([float(x) for x in args.dispersions.split(",")]
                   if args.dispersions else None)
            spread_dispersion_sweep(sport_key, espn_sport, espn_league,
                                    season_year=odds_seasons, limit=args.limit,
                                    store_label=args.store_label, source=args.source,
                                    dispersions=_ds)
        elif args.pythag_residual_sweep:
            _rw = ([float(x) for x in args.residual_weights.split(",")]
                   if args.residual_weights else None)
            pythag_residual_sweep(sport_key, espn_sport, espn_league,
                                  season_year=odds_seasons, limit=args.limit,
                                  store_label=args.store_label, source=args.source,
                                  weights=_rw)
        else:
            run_odds_backtest(sport_key, espn_sport, espn_league,
                              limit=args.limit, window=args.window, variants=variants,
                              min_sample=args.min_sample, season_year=odds_seasons,
                              threshold_pct=args.threshold,
                              write_calibration=args.write_calibration,
                              store_label=args.store_label,
                              variance_inflate=args.variance_inflate,
                              engine=args.engine, prob_shrink=args.prob_shrink,
                              source=args.source, snapshot=args.snapshot,
                              supplement_log=args.supplement_log,
                              min_shrink_n=args.min_shrink_n,
                              serve_mode=args.serve_mode, fit_blend=args.fit_blend,
                              fit_shares=args.fit_shares)
    elif args.mode == "coherence":
        run_coherence_backtest(
            sport_key, espn_sport, espn_league,
            season_year=(_parse_seasons(args.seasons) if args.seasons
                         else args.season),
            store_label=args.store_label, source=args.source,
            snapshot=args.snapshot, limit=args.limit)
    elif args.mode == "prop-lag":
        _props = ([x.strip() for x in args.props.split(",") if x.strip()]
                  if args.props else ("batter_hits", "batter_total_bases"))
        run_prop_lag_backtest(args.sport, espn_sport, espn_league, sport_key,
                              props=_props, min_prior=args.min_sample)
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
        elif ((getattr(args, "recency_sweep", False)
               or getattr(args, "weather_sweep", False)) and args.sport == "mlb"):
            # STEP-1 decision-grade: the hand-picked DEFAULT_STARTERS is stable
            # superstars, which are SURVIVORSHIP-BIASED toward longer windows (a
            # stable-talent player always benefits from more history). Use the
            # usage-representative refit pool (rookies / role-changers / part-timers
            # included) so the recent_n/half_life pick generalizes to who we
            # actually bet. Pass --players to override.
            try:
                from refit_calibration import _mlb_player_pool
                players = [name for (_pid, _role, name)
                          in _mlb_player_pool(args.season)]
                print(f"#  (recency-sweep) representative MLB pool: "
                      f"{len(players)} players (not the star DEFAULT_STARTERS)")
            except Exception as e:
                print(f"  [warn] representative pool unavailable ({e}); "
                      f"falling back to DEFAULT_STARTERS.")
                players = DEFAULT_STARTERS.get(args.sport, [])
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

    # Point the user at the promotion step whenever a calibration write staged a
    # candidate (or tell them a --live run went straight to the live file).
    if args.write_calibration and not staging:
        print(f"\n⇢ Wrote LIVE calibration/{sport_key}.json directly (--live).")
    elif args.write_calibration and staging and has_candidate(sport_key):
        print(f"\n⇢ Staged to calibration/{sport_key}.candidate.json — the live "
              f"file is UNTOUCHED. Promote: python refit_calibration.py "
              f"--sport {args.sport} --promote (review with --diff).")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
