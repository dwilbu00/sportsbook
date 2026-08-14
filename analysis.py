"""
Analysis engine for comparing sportsbook odds against historical data.
Identifies value bets where book implied probability < historical probability.
"""

import math

import mlb_starters
from odds_client import (
    american_to_implied_prob,
    devig_two_way,
)


# Odds/EV math, calibration glue, and the shared venue/defense multipliers now
# live in pricing_common.py; imported + re-exported here so the modules that
# import them from analysis (backtest, book_line_calibration, tests) keep working
# and so the split's props/parlay modules share one implementation.
from pricing_common import (  # noqa: E402
    _MARKET_BLEND_CACHE,
    _PROB_SHRINK_CACHE,
    _STARTER_ADJ_CACHE,
    _decimal_to_american,
    _consensus_price_for_line,
    _expected_roi,
    _prop_is_value,
    _devig_fair,
    _starter_adjustment,
    _shrink_factor,
    _apply_shrink,
    _blend_weight,
    VENUE_MATCH_WEIGHTS,
    DEFAULT_VENUE_WEIGHTS,
    _venue_match_multiplier,
    _opponent_defense_multiplier,
)


from calibration_loader import load_expected_runs_challenger


# _MARKET_BLEND_CACHE / _PROB_SHRINK_CACHE / _STARTER_ADJ_CACHE now live in
# pricing_common.py (imported + re-exported above).
_EXPECTED_RUNS_CACHE = {}


def _apply_starter_logit(p, edge, weight):
    """Shift probability p in logit space by weight*edge, bounded to [.02,.98].
    edge>0 favors the team; weight is the (calibratable) logit multiplier."""
    if not weight or edge is None or p <= 0 or p >= 1:
        return p
    lg = math.log(p / (1 - p)) + weight * edge
    return max(0.02, min(0.98, 1.0 / (1.0 + math.exp(-lg))))


# ── Pythagorean team strength (ACTIVE moneyline blend) ────────────────────────
# Expected win% from run differential via the canonical, None-safe 1.83-exponent
# helper mlb_starters.pythagorean_win_probability (RS^e/(RS^e+RA^e)) — a better
# season strength estimate than raw W-L%. Runs come from the warehouse /standings
# season block (runs_scored/runs_allowed); the ESPN block has none, so this is a
# warehouse-enabled signal (None when runs absent → skipped).
#
# Weight 0.35 chosen from the 2023-2026 backtest (backtest_starters --test-final,
# per-season holdouts): the raw run-differential Pythagorean beat the current
# win%-based moneyline model in 2024 (0.2460 vs 0.2490) and 2026 (0.2471 vs
# 0.2492) — both years the current model was near coin-flip — and lagged only in
# 2025 (0.2486 vs 0.2461), when the current model was unusually strong. A modest
# blend (pull toward, not replace) captures the hedge-when-weak upside while
# keeping the worst-year downside small (~0.0016 Brier). Spreads/run-line already
# run on the validated expected_runs_challenger; totals stay on the current model.
DEFAULT_PYTHAG_WEIGHT = 0.35


# ──────────────────────────────────────────────────────────────────────────────
#  Pure-Python statistics helpers now live in stats.py (imported + re-exported
#  here for backward compatibility with the modules that import them from
#  analysis: backtest, book_line_calibration, backtest_starters).
# ──────────────────────────────────────────────────────────────────────────────
from stats import (  # noqa: E402  (kept near original location, not top-of-file)
    _SQRT2,
    _norm_cdf,
    _norm_ppf,
    _normal_inv_cdf,
    RECENCY_HALF_LIFE,
    DEFAULT_HALF_LIFE,
    _half_life_for,
    _recency_weights,
    _weighted_mean,
    _weighted_rate,
    _weighted_quantile,
    _weighted_std,
)


def _predict_margin(game_odds, home_team_stats, away_team_stats, sport_key,
                    matchup_features=None):
    """Shared home-perspective baseline game-margin distribution.

    Moneyline and spread analyzers both start from this Normal distribution.
    A validated market-specific challenger may be blended into the MLB spread
    result later without changing this baseline or the moneyline model.

    Returns (pred_margin, pred_std, home_stats, away_stats) or None when either
    team lacks usable recent games.
    """
    half_life = _half_life_for(sport_key)
    home_team = game_odds["home_team"]
    away_team = game_odds["away_team"]

    def _team_margin_stats(team_name, stats, is_home):
        recent_games = stats.get("recent_games", [])
        margins = []
        past_is_home_list = []
        for game in recent_games:
            if game["home_team"] == team_name:
                margin = game["home_score"] - game["away_score"]
                past_is_home_list.append(True)
            elif game["away_team"] == team_name:
                margin = game["away_score"] - game["home_score"]
                past_is_home_list.append(False)
            else:
                continue
            margins.append(margin)
        if not margins:
            return None
        base_weights = _recency_weights(len(margins), half_life)
        weights = [
            bw * _venue_match_multiplier(past_h, is_home, sport_key)
            for bw, past_h in zip(base_weights, past_is_home_list)
        ]
        mean = _weighted_mean(margins, weights)
        std = _weighted_std(margins, weights, mean=mean)
        return {"margins": margins, "weights": weights, "mean": mean, "std": std}

    home_stats = _team_margin_stats(home_team, home_team_stats, True)
    away_stats = _team_margin_stats(away_team, away_team_stats, False)
    if not home_stats or not away_stats:
        return None

    pred_margin = home_stats["mean"] - away_stats["mean"]
    # Phase 1: shift the predicted margin by the (innings-weighted, two-sided)
    # starter edge. Positive edge = home's effective run-prevention is better, so
    # the home margin rises. This single shift feeds BOTH ML and spreads.
    if matchup_features and matchup_features.get("starter_edge") is not None:
        pred_margin += (_starter_adjustment(sport_key, "spreads")
                        * matchup_features["starter_edge"])
    # Floor each team's std at 1.0 to avoid degenerate certainty on thin samples.
    home_var = max(home_stats["std"], 1.0) ** 2
    away_var = max(away_stats["std"], 1.0) ** 2
    pred_std = math.sqrt(home_var + away_var)
    return pred_margin, pred_std, home_stats, away_stats


def _mlb_expected_runs_projection(sport_key, matchup_features):
    """Return the enabled spread-only expected-runs projection, when complete."""
    if sport_key != "baseball_mlb" or not matchup_features:
        return None
    factors = matchup_features.get("expected_runs") or {}
    if not factors.get("complete"):
        return None

    if sport_key not in _EXPECTED_RUNS_CACHE:
        try:
            _EXPECTED_RUNS_CACHE[sport_key] = (
                load_expected_runs_challenger(sport_key) or {})
        except Exception:
            _EXPECTED_RUNS_CACHE[sport_key] = {}
    config = _EXPECTED_RUNS_CACHE[sport_key]
    live_markets = config.get("live_markets") or {}
    final_validation = config.get("final_2025_validation") or {}
    model = final_validation.get("model") or {}
    shares = final_validation.get("ensemble_challenger_share") or {}
    if not config.get("enabled") or not live_markets.get("spreads"):
        return None

    try:
        offense_weight = float(model["offense_weight"])
        pitching_weight = float(model["pitching_weight"])
        home_base_runs = float(model["home_base_runs"])
        away_base_runs = float(model["away_base_runs"])
        home_offense = float(factors["home_offense_factor"])
        away_offense = float(factors["away_offense_factor"])
        home_staff = float(factors["home_staff_suppression"])
        away_staff = float(factors["away_staff_suppression"])
        spread_share = float(shares["home_minus_1_5"])
        margin_share = float(shares["margin"])
    except (KeyError, TypeError, ValueError):
        return None
    if (min(home_base_runs, away_base_runs, home_offense, away_offense,
            home_staff, away_staff) <= 0
            or offense_weight < 0 or pitching_weight < 0
            or not 0.0 <= spread_share <= 1.0
            or not 0.0 <= margin_share <= 1.0):
        return None

    home_runs = mlb_starters.expected_runs_from_factors(
        home_base_runs, home_offense, away_staff,
        offense_weight, pitching_weight)
    away_runs = mlb_starters.expected_runs_from_factors(
        away_base_runs, away_offense, home_staff,
        offense_weight, pitching_weight)
    if home_runs is None or away_runs is None:
        return None
    return {
        "home_runs": home_runs,
        "away_runs": away_runs,
        "margin": home_runs - away_runs,
        "spread_share": spread_share,
        "margin_share": margin_share,
    }


def analyze_moneyline_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0, sport_key=None, matchup_features=None):
    """
    Compare moneyline implied probabilities against historical win rates.

    Parameters:
        game_odds (dict): Parsed game odds from odds_client.parse_game_odds()
        home_team_stats (dict): Home team stats with 'season' and 'recent' keys
        away_team_stats (dict): Away team stats with 'season' and 'recent' keys
        threshold_pct (float): Minimum edge (percentage points) to flag as value

    Returns:
        list: Value candidates with edge details
    """
    candidates = []
    threshold = threshold_pct / 100.0
    half_life = _half_life_for(sport_key)

    home_team = game_odds["home_team"]
    away_team = game_odds["away_team"]

    # Consensus implied probability per team (used for de-vigging the market
    # when blending the model toward the closing line).
    avg_implied_by_team = {}
    for tn in (home_team, away_team):
        lst = game_odds["moneyline"].get(tn, [])
        if lst:
            avg_implied_by_team[tn] = sum(o["implied_prob"] for o in lst) / len(lst)
    blend_w = _blend_weight(sport_key, "moneyline")

    # ── Shared baseline margin model ────────────────────────────────────────
    # ML and spreads begin with ONE predicted-margin distribution:
    #   P(home win) = P(margin > 0) = Φ(pred_margin / pred_std)
    # The starter/opponent edge already shifts the margin inside
    # _predict_margin(), so no separate ML starter logit is needed. Validated
    # spread-only overlays are applied later and do not alter this ML result.
    margin = _predict_margin(game_odds, home_team_stats, away_team_stats,
                             sport_key, matchup_features)
    model_win_by_team = {}
    if margin is not None:
        pred_margin, pred_std, _, _ = margin
        home_win = _norm_cdf(pred_margin / pred_std)
        model_win_by_team[home_team] = home_win
        model_win_by_team[away_team] = 1.0 - home_win

    for team_name, stats in [(home_team, home_team_stats), (away_team, away_team_stats)]:
        ml_odds = game_odds["moneyline"].get(team_name, [])
        if not ml_odds:
            continue

        # Average implied probability across all books
        avg_implied = sum(o["implied_prob"] for o in ml_odds) / len(ml_odds)

        # Informational win-rate context (recency-weighted). Still surfaced in
        # the UI, but no longer the source of the model probability.
        season_wp = stats["season"]["win_pct"]
        flat_recent_wp = stats["recent"]["win_pct"]
        # Pythagorean strength from the warehouse season block's runs (None on the
        # ESPN path, which carries no runs). Exposed always; blended only when
        # weighted (inert by default — see DEFAULT_PYTHAG_WEIGHT).
        pythag_wp = mlb_starters.pythagorean_win_probability(
            stats["season"].get("runs_scored"),
            stats["season"].get("runs_allowed"))
        upcoming_is_home = (team_name == home_team)

        recent_games = stats.get("recent_games", [])
        wins_for_team = []
        past_is_home_list = []
        for g in recent_games:
            if g["home_team"] == team_name:
                wins_for_team.append(1 if g["home_score"] > g["away_score"] else 0)
                past_is_home_list.append(True)
            elif g["away_team"] == team_name:
                wins_for_team.append(1 if g["away_score"] > g["home_score"] else 0)
                past_is_home_list.append(False)

        if wins_for_team:
            base_weights = _recency_weights(len(wins_for_team), half_life)
            weights = [
                bw * _venue_match_multiplier(past_h, upcoming_is_home, sport_key)
                for bw, past_h in zip(base_weights, past_is_home_list)
            ]
            recent_wp = _weighted_mean(wins_for_team, weights)
        else:
            recent_wp = flat_recent_wp

        # Model win probability from the shared margin distribution. Fall back to
        # the recency-weighted win% blend (+ starter logit) only when the margin
        # model is unavailable because a team lacks usable recent games.
        if team_name in model_win_by_team:
            model_prob = model_win_by_team[team_name]
        else:
            model_prob = (0.4 * season_wp) + (0.6 * recent_wp)
            if matchup_features and matchup_features.get("starter_edge") is not None:
                sign = 1.0 if team_name == home_team else -1.0
                model_prob = _apply_starter_logit(
                    model_prob, sign * matchup_features["starter_edge"],
                    _starter_adjustment(sport_key, "moneyline"))

        # Pythagorean blend (WIRING ONLY — inert while DEFAULT_PYTHAG_WEIGHT==0.0).
        # When weighted, pull the model win prob toward the run-differential
        # estimate; skipped when runs are absent (ESPN path).
        if DEFAULT_PYTHAG_WEIGHT > 0 and pythag_wp is not None:
            model_prob = ((1.0 - DEFAULT_PYTHAG_WEIGHT) * model_prob
                          + DEFAULT_PYTHAG_WEIGHT * pythag_wp)

        # Calibrated overconfidence correction (no-op until an ML shrink is fit
        # from backfilled h2h history), then optional model⇄market blend toward
        # the de-vigged closing line (blend_w=1.0 → pure model).
        shrunk = _apply_shrink(model_prob, sport_key, "moneyline")
        final_prob = shrunk
        opp = away_team if team_name == home_team else home_team
        if (blend_w < 1.0 and team_name in avg_implied_by_team
                and opp in avg_implied_by_team):
            fair_team, _ = devig_two_way(avg_implied_by_team[team_name],
                                         avg_implied_by_team[opp])
            final_prob = blend_w * shrunk + (1.0 - blend_w) * fair_team

        # Edge is measured against the DE-VIGGED fair probability so it is
        # comparable across markets (and with props); value additionally
        # requires positive EV at the executable price (see _prop_is_value).
        # `opp` is defined above for the model⇄market blend.
        fair_implied = _devig_fair(
            avg_implied_by_team.get(team_name), avg_implied_by_team.get(opp))
        if fair_implied is None:
            fair_implied = avg_implied
        edge = final_prob - fair_implied

        best_offer = min(ml_odds, key=lambda o: o["implied_prob"])
        best_book_prob = best_offer["implied_prob"]
        best_edge = final_prob - best_book_prob
        expected_roi = _expected_roi(final_prob, best_offer["price"])

        result = {
            "type": "moneyline",
            "team": team_name,
            "opponent": away_team if team_name == home_team else home_team,
            "home_away": "HOME" if team_name == home_team else "AWAY",
            "book_implied_prob": round(avg_implied * 100, 2),
            "season_win_pct": round(season_wp * 100, 2),
            "recent_win_pct": round(recent_wp * 100, 2),
            "pythag_win_pct": (round(pythag_wp * 100, 2)
                               if pythag_wp is not None else None),
            # model_prob = pure shared-margin win prob (graded by the backtest);
            # hist_prob mirrors it for backward-compatible UI display.
            "model_prob": round(model_prob * 100, 2),
            "hist_prob": round(model_prob * 100, 2),
            "blended_prob": round(final_prob * 100, 2),
            "edge_pct": round(edge * 100, 2),
            "best_edge_pct": round(best_edge * 100, 2),
            "best_book_implied_prob": round(best_book_prob * 100, 2),
            "best_book": best_offer["book"],
            "best_price": best_offer["price"],
            "expected_roi_pct": (round(expected_roi * 100, 2)
                                  if expected_roi is not None else None),
            "is_value": _prop_is_value(edge, threshold, expected_roi),
        }
        candidates.append(result)

    return candidates


def analyze_totals_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0, sport_key=None, matchup_features=None):
    """
    Compare over/under lines against historical scoring averages.

    Parameters:
        game_odds (dict): Parsed game odds
        home_team_stats (dict): Home team stats
        away_team_stats (dict): Away team stats
        threshold_pct (float): Minimum edge to flag

    Returns:
        list: Value candidates for over/under
    """
    candidates = []
    threshold = threshold_pct / 100.0
    half_life = _half_life_for(sport_key)

    over_odds = game_odds["totals"].get("Over", [])
    under_odds = game_odds["totals"].get("Under", [])

    if not over_odds:
        return candidates

    # Get the consensus line (most common total)
    lines = [o["line"] for o in over_odds]
    consensus_line = max(set(lines), key=lines.count) if lines else 0

    home_team = game_odds["home_team"]
    away_team = game_odds["away_team"]

    # Recency-, opponent-strength-, and venue-match-weighted scoring averages,
    # computed per-team from recent_games. Falls back to the flat precomputed
    # averages if no recent_games are present.
    def _weighted_team_scoring(team_name, stats, upcoming_is_home):
        # opp_strength removed (verified zero impact via backtest); recency + venue only.
        recent_games = stats.get("recent_games", [])
        scored, allowed, past_is_home = [], [], []
        for g in recent_games:
            if g["home_team"] == team_name:
                scored.append(g["home_score"])
                allowed.append(g["away_score"])
                past_is_home.append(True)
            elif g["away_team"] == team_name:
                scored.append(g["away_score"])
                allowed.append(g["home_score"])
                past_is_home.append(False)
        if not scored:
            return stats["recent"]["avg_scored"], stats["recent"]["avg_allowed"]
        base_weights = _recency_weights(len(scored), half_life)
        weights = [
            bw * _venue_match_multiplier(past_h, upcoming_is_home, sport_key)
            for bw, past_h in zip(base_weights, past_is_home)
        ]
        return _weighted_mean(scored, weights), _weighted_mean(allowed, weights)

    home_avg_scored, home_avg_allowed = _weighted_team_scoring(home_team, home_team_stats, True)
    away_avg_scored, away_avg_allowed = _weighted_team_scoring(away_team, away_team_stats, False)

    # Projected total: average of (home_scored + away_scored) and (home_allowed + away_allowed)
    projected_from_offense = home_avg_scored + away_avg_scored
    projected_from_defense = home_avg_allowed + away_avg_allowed
    projected_total = (projected_from_offense + projected_from_defense) / 2

    # Phase 1: shift the projected total by combined starter quality. Better
    # probable starters (run_suppression > 1 via xERA) suppress scoring, so a
    # strong pair pulls the total down (and vice versa). run_scale = runs
    # suppressed per unit of combined excess (calibratable prior).
    starter_total_shift = 0.0
    if matchup_features:
        excess = 0.0            # combined starter run-suppression excess
        bullpen_excess = 0.0    # combined bullpen run-suppression excess (Phase 3)
        for side in ("home", "away"):
            sd = matchup_features.get(side)
            if not sd:
                continue
            if sd.get("starter"):
                excess += (sd["starter"].get("run_suppression", 1.0) - 1.0)
            if sd.get("bullpen"):
                bullpen_excess += (sd["bullpen"].get("bullpen_suppression", 1.0) - 1.0)
        run_scale = _starter_adjustment(sport_key, "run_scale")
        bullpen_w = _starter_adjustment(sport_key, "bullpen")
        # Better arms (suppression > 1) pull the projected total DOWN.
        starter_total_shift = -(run_scale * excess) - (bullpen_w * bullpen_excess)
        projected_total += starter_total_shift

    # Build the historical total-score spread around the projection. The same
    # projected_total used for display is the mean of the probability model;
    # this prevents the displayed projection and value probability from moving
    # in opposite directions through separate starter adjustments.
    over_weight = 0.0
    total_weight = 0.0
    historical_totals = []
    historical_total_weights = []
    for team_name, stats, upcoming_is_home in [
        (home_team, home_team_stats, True),
        (away_team, away_team_stats, False),
    ]:
        games = stats.get("recent_games", [])
        if not games:
            continue
        base_weights = _recency_weights(len(games), half_life)
        for g, bw in zip(games, base_weights):
            if g["home_team"] == team_name:
                past_h = True
            elif g["away_team"] == team_name:
                past_h = False
            else:
                past_h = None
            w = bw * _venue_match_multiplier(past_h, upcoming_is_home, sport_key)
            total_weight += w
            historical_totals.append(g["total_score"])
            historical_total_weights.append(w)
            if g["total_score"] > consensus_line:
                over_weight += w

    empirical_over_rate = (over_weight / total_weight) if total_weight > 0 else 0.5
    total_std = _weighted_std(historical_totals, historical_total_weights)
    if total_std > 0:
        z = (consensus_line - projected_total) / total_std
        model_over_hit_rate = 1.0 - _norm_cdf(z)
    else:
        model_over_hit_rate = empirical_over_rate

    diff = projected_total - consensus_line

    # Prices for the consensus line, picked from each side's odds list.
    over_price = _consensus_price_for_line(over_odds, consensus_line, "line")
    under_price = _consensus_price_for_line(under_odds, consensus_line, "line")

    # Blend the model over-rate toward the de-vigged market over probability
    # using the offline-fitted weight (w=1.0 → pure model = original).
    # Calibrated overconfidence correction: pull the model probability toward
    # 0.5 before any market blend (no-op when no shrink is configured).
    over_hit_rate = _apply_shrink(model_over_hit_rate, sport_key, "totals")
    blend_w = _blend_weight(sport_key, "totals")
    if blend_w < 1.0 and over_price is not None and under_price is not None:
        fair_over, _ = devig_two_way(american_to_implied_prob(over_price),
                                     american_to_implied_prob(under_price))
        over_hit_rate = blend_w * over_hit_rate + (1.0 - blend_w) * fair_over

    # Edge vs the DE-VIGGED fair probability (comparable across markets); fall
    # back to raw implied when a side has no price, or 0.5 when both are missing.
    raw_over_implied = (american_to_implied_prob(over_price)
                        if over_price is not None else None)
    raw_under_implied = (american_to_implied_prob(under_price)
                         if under_price is not None else None)
    over_implied = _devig_fair(raw_over_implied, raw_under_implied)
    if over_implied is None:
        over_implied = 0.50
    under_implied = _devig_fair(raw_under_implied, raw_over_implied)
    if under_implied is None:
        under_implied = 0.50
    over_edge = over_hit_rate - over_implied
    under_hit_rate = 1.0 - over_hit_rate
    under_edge = under_hit_rate - under_implied
    over_roi = (_expected_roi(over_hit_rate, over_price)
                if over_price is not None else None)
    under_roi = (_expected_roi(under_hit_rate, under_price)
                 if under_price is not None else None)

    candidates.append({
        "type": "total_over",
        "matchup": f"{game_odds['away_team']} @ {game_odds['home_team']}",
        "line": consensus_line,
        "projected_total": round(projected_total, 2),
        "diff_from_line": round(diff, 2),
        "model_over_hit_rate": round(model_over_hit_rate * 100, 2),
        "over_hit_rate": round(over_hit_rate * 100, 2),
        "over_implied": round(over_implied * 100, 2),
        "under_implied": round(under_implied * 100, 2),
        "over_edge_pct": round(over_edge * 100, 2),
        "under_edge_pct": round(under_edge * 100, 2),
        "over_expected_roi_pct": (round(over_roi * 100, 2)
                                   if over_roi is not None else None),
        "under_expected_roi_pct": (round(under_roi * 100, 2)
                                    if under_roi is not None else None),
        "home_avg_scored": round(home_avg_scored, 2),
        "away_avg_scored": round(away_avg_scored, 2),
        "is_over_value": diff > 0 and _prop_is_value(over_edge, threshold, over_roi),
        "is_under_value": diff < 0 and _prop_is_value(under_edge, threshold, under_roi),
        "over_price": over_price,
        "under_price": under_price,
        # When a side has no consensus price, its over_implied/over_edge_pct are
        # a 0.50-devig placeholder, not a real market number. The value gate
        # already blocks surfacing (over_roi/under_roi is None without a price),
        # so these flags are display metadata: the UI can mark the side unpriced
        # instead of showing the placeholder-derived edge.
        "over_price_missing": over_price is None,
        "under_price_missing": under_price is None,
    })

    return candidates


def analyze_spreads_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0, sport_key=None, matchup_features=None):
    """
    Compare spread lines against historical scoring margins, using a joint
    baseline distribution of the predicted game margin (home perspective).

    For each team, the model estimates the weighted mean and weighted std of
    that team's recent margins. The game's actual margin is then approximated
    as Normal(home_mean − away_mean, sqrt(home_var + away_var)) under
    independence. For MLB games with complete probable-starter and handedness
    inputs, the chronologically validated expected-runs model is blended into
    spread probability and displayed margin only. Moneylines remain unchanged.
    Home and away probabilities remain complementary for opposing half-run lines.

    Parameters:
        game_odds (dict): Parsed game odds from odds_client.parse_game_odds()
        home_team_stats (dict): Home team stats with 'recent' and 'recent_games' keys
        away_team_stats (dict): Away team stats with 'recent' and 'recent_games' keys
        threshold_pct (float): Minimum edge to flag as value

    Returns:
        list: Value candidates for spread bets (at most one will be is_value=True per game)
    """
    threshold = threshold_pct / 100.0

    home_team = game_odds["home_team"]
    away_team = game_odds["away_team"]

    # ── Shared baseline predicted-margin distribution ───────────────────────
    margin = _predict_margin(game_odds, home_team_stats, away_team_stats,
                             sport_key, matchup_features)
    # If we lack usable recent games for either side we cannot compute a
    # meaningful cover probability, so skip the matchup entirely (better than
    # recommending both halves of a bet on half the information).
    if margin is None:
        return []
    current_pred_margin, pred_std, home_stats, away_stats = margin
    expected_runs = _mlb_expected_runs_projection(
        sport_key, matchup_features)
    pred_margin = current_pred_margin
    if expected_runs:
        margin_share = expected_runs["margin_share"]
        pred_margin += margin_share * (
            expected_runs["margin"] - pred_margin)

    # ── Consensus spread per team ───────────────────────────────────────────
    def _consensus_spread(team):
        spread_odds = game_odds["spreads"].get(team, [])
        if not spread_odds:
            return None
        spreads = [o["spread"] for o in spread_odds]
        return max(set(spreads), key=spreads.count)

    home_spread = _consensus_spread(home_team)
    away_spread = _consensus_spread(away_team)

    # ── Build candidate per team using the joint cover probability ──────────
    candidates = []
    games_sampled = min(len(home_stats["margins"]), len(away_stats["margins"]))

    home_price = _consensus_price_for_line(
        game_odds["spreads"].get(home_team, []), home_spread, "spread") \
        if home_spread is not None else None
    away_price = _consensus_price_for_line(
        game_odds["spreads"].get(away_team, []), away_spread, "spread") \
        if away_spread is not None else None

    # De-vigged market P(home covers), used to blend the model toward the line.
    blend_w = _blend_weight(sport_key, "spreads")
    market_home_cover = None
    if blend_w < 1.0 and home_price is not None and away_price is not None:
        market_home_cover, _ = devig_two_way(
            american_to_implied_prob(home_price),
            american_to_implied_prob(away_price))

    # De-vigged fair cover probabilities for the EDGE baseline (independent of
    # the model⇄market blend above): edge is measured against the fair line,
    # not the vig-inflated raw implied prob. Falls back to raw per-side implied
    # inside _add_candidate when a side's price is missing.
    fair_home_cover = fair_away_cover = None
    if home_price is not None and away_price is not None:
        fair_home_cover, fair_away_cover = devig_two_way(
            american_to_implied_prob(home_price),
            american_to_implied_prob(away_price))

    def _cover_probabilities(current_cover, expected_cover, market_cover):
        # The final holdout fitted the challenger share against the already-
        # shrunk current spread model. Do not shrink the resulting ensemble a
        # second time. With no complete expected-runs projection this follows
        # the original path exactly.
        if expected_cover is None:
            model_cover = current_cover
            adjusted = _apply_shrink(current_cover, sport_key, "spreads")
        else:
            current_adjusted = _apply_shrink(
                current_cover, sport_key, "spreads")
            share = expected_runs["spread_share"]
            model_cover = current_adjusted + share * (
                expected_cover - current_adjusted)
            adjusted = model_cover
        if market_cover is None:
            return model_cover, adjusted
        return (model_cover,
                blend_w * adjusted + (1.0 - blend_w) * market_cover)

    def _add_candidate(team_name, opponent, is_home, spread, model_cover,
                       cover_prob, team_avg_margin, price,
                       current_cover, expected_cover, fair_implied):
        raw_implied = (american_to_implied_prob(price)
                       if price is not None else 0.50)
        # Edge vs the de-vigged fair cover prob; fall back to raw implied when
        # the market is one-sided. Value additionally requires +EV at the price.
        if fair_implied is None:
            fair_implied = raw_implied
        edge = cover_prob - fair_implied
        roi = _expected_roi(cover_prob, price) if price is not None else None
        candidates.append({
            "type": "spread",
            "team": team_name,
            "opponent": opponent,
            "home_away": "HOME" if is_home else "AWAY",
            "spread": spread,
            "avg_margin": round(team_avg_margin, 2),
            "model_cover_rate": round(model_cover * 100, 2),
            "cover_rate": round(cover_prob * 100, 2),
            "implied_prob": round(fair_implied * 100, 2),
            "games_sampled": games_sampled,
            "edge_pct": round(edge * 100, 2),
            "expected_roi_pct": (round(roi * 100, 2)
                                  if roi is not None else None),
            "is_value": _prop_is_value(edge, threshold, roi),
            "pred_game_margin": round(pred_margin, 2),
            "pred_game_std": round(pred_std, 2),
            "price": price,
            # When price is missing, implied_prob/edge_pct are a 0.50 placeholder
            # (not a market number). The value gate already blocks surfacing
            # (roi is None without a price); this flag is display metadata.
            "price_missing": price is None,
            "model_source": ("expected_runs_ensemble" if expected_runs
                             else "current_margin_model"),
        })
        if expected_runs:
            candidates[-1].update({
                "current_model_cover_rate": round(current_cover * 100, 2),
                "expected_runs_cover_rate": round(expected_cover * 100, 2),
                "current_pred_game_margin": round(current_pred_margin, 2),
                "expected_home_runs": round(expected_runs["home_runs"], 2),
                "expected_away_runs": round(expected_runs["away_runs"], 2),
            })

    if home_spread is not None:
        # Home covers iff actual_margin + home_spread > 0  ⇔  margin > -home_spread.
        # P(margin > -home_spread) = Φ((pred_margin + home_spread) / pred_std)
        current_home_cover = _norm_cdf(
            (current_pred_margin + home_spread) / pred_std)
        expected_home_cover = (
            mlb_starters.poisson_margin_probability(
                expected_runs["home_runs"], expected_runs["away_runs"],
                home_spread)
            if expected_runs else None)
        model_home_cover, home_cover_prob = _cover_probabilities(
            current_home_cover, expected_home_cover, market_home_cover)
        _add_candidate(home_team, away_team, True, home_spread,
                       model_home_cover, home_cover_prob,
                       home_stats["mean"], home_price,
                       current_home_cover, expected_home_cover,
                       fair_home_cover)
    if away_spread is not None:
        # Away covers iff -actual_margin + away_spread > 0  ⇔  margin < away_spread.
        # P(margin < away_spread) = Φ((away_spread - pred_margin) / pred_std)
        current_away_cover = _norm_cdf(
            (away_spread - current_pred_margin) / pred_std)
        expected_away_cover = (
            mlb_starters.poisson_margin_probability(
                expected_runs["away_runs"], expected_runs["home_runs"],
                away_spread)
            if expected_runs else None)
        market_away_cover = (1.0 - market_home_cover
                             if market_home_cover is not None else None)
        model_away_cover, away_cover_prob = _cover_probabilities(
            current_away_cover, expected_away_cover, market_away_cover)
        _add_candidate(away_team, home_team, False, away_spread,
                       model_away_cover, away_cover_prob,
                       away_stats["mean"], away_price,
                       current_away_cover, expected_away_cover,
                       fair_away_cover)

    return candidates


def format_moneyline_report(candidates):
    """Format moneyline value candidates into a readable report."""
    value_bets = [c for c in candidates if c["is_value"]]
    non_value = [c for c in candidates if not c["is_value"]]

    lines = []
    if value_bets:
        lines.append("=" * 80)
        lines.append("  VALUE BETS FOUND (Historical Prob > Book Implied Prob + Threshold)")
        lines.append("=" * 80)
        for c in sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True):
            lines.append("")
            lines.append(f"  >>> {c['team']} ({c['home_away']}) vs {c['opponent']}")
            lines.append(f"      Book Implied:   {c['book_implied_prob']}%")
            lines.append(f"      Season Win%:    {c['season_win_pct']}%")
            lines.append(f"      Recent Win%:    {c['recent_win_pct']}% (last N games)")
            lines.append(f"      Blended Hist:   {c['hist_prob']}%")
            lines.append(f"      EDGE:           +{c['edge_pct']}% (best: +{c['best_edge_pct']}% at {c['best_book']})")
            lines.append(f"      Best Price:     {c['best_price']:+d}")
    else:
        lines.append("\n  No moneyline value bets found above threshold.")

    if non_value:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Other Matchups (no edge above threshold)")
        lines.append("-" * 80)
        for c in non_value:
            edge_str = f"+{c['edge_pct']}%" if c['edge_pct'] > 0 else f"{c['edge_pct']}%"
            lines.append(f"  {c['team']:30s} | Implied: {c['book_implied_prob']:5.2f}% | Hist: {c['hist_prob']:5.2f}% | Edge: {edge_str}")

    return "\n".join(lines)


def format_spreads_report(candidates):
    """Format spread value candidates into a readable report."""
    value_bets = [c for c in candidates if c["is_value"]]
    non_value = [c for c in candidates if not c["is_value"]]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  SPREAD ANALYSIS (Historical Cover Rate vs Implied ~50%)")
    lines.append("=" * 80)

    if value_bets:
        for c in sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True):
            lines.append("")
            lines.append(f"  >>> {c['team']} ({c['home_away']}) vs {c['opponent']}")
            lines.append(f"      Spread:         {c['spread']:+.2f}")
            lines.append(f"      Avg Margin:     {c['avg_margin']:+.2f} (last {c['games_sampled']} games)")
            lines.append(f"      Cover Rate:     {c['cover_rate']}%")
            lines.append(f"      EDGE:           +{c['edge_pct']}% over implied 50%")
    else:
        lines.append("\n  No spread value bets found above threshold.")

    if non_value:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Other Spreads (no edge above threshold)")
        lines.append("-" * 80)
        for c in non_value:
            edge_str = f"+{c['edge_pct']}%" if c['edge_pct'] > 0 else f"{c['edge_pct']}%"
            lines.append(f"  {c['team']:30s} | Spread: {c['spread']:+5.2f} | Cover: {c['cover_rate']:5.2f}% | Edge: {edge_str}")

    return "\n".join(lines)


def format_totals_report(candidates):
    """Format totals value candidates into a readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  OVER/UNDER ANALYSIS")
    lines.append("=" * 80)

    for c in candidates:
        flag = ""
        if c["is_over_value"]:
            flag = " <<< OVER VALUE"
        elif c["is_under_value"]:
            flag = " <<< UNDER VALUE"

        lines.append(f"")
        lines.append(f"  {c['matchup']}")
        lines.append(f"      Line:              {c['line']}")
        lines.append(f"      Projected Total:   {c['projected_total']} (diff: {c['diff_from_line']:+.2f}){flag}")
        lines.append(f"      Over Hit Rate:     {c['over_hit_rate']}% (recent games)")
        lines.append(f"      Avg Scored (Home): {c['home_avg_scored']}  |  (Away): {c['away_avg_scored']}")

    return "\n".join(lines)


def make_bet_checklist_entry(candidate, bet_type, side=None):
    """Build a price-free DraftKings instruction from one value candidate."""
    def _line(value, signed=False):
        try:
            value = float(value)
            return f"{value:+g}" if signed else f"{value:g}"
        except (TypeError, ValueError):
            return str(value)

    if bet_type in ("moneyline", "spread"):
        team = candidate["team"]
        opponent = candidate["opponent"]
        matchup = (
            f"{opponent} @ {team}"
            if candidate.get("home_away") == "HOME"
            else f"{team} @ {opponent}"
        )
    else:
        matchup = candidate["matchup"]

    if bet_type == "moneyline":
        type_label = "Moneyline"
        bet = f"{candidate['team']} moneyline"
        team = candidate["team"]
        identity = (candidate.get("event_id") or matchup, bet_type, team)
    elif bet_type == "spread":
        type_label = "Spread"
        spread = _line(candidate["spread"], signed=True)
        bet = f"{candidate['team']} {spread}"
        team = candidate["team"]
        identity = (
            candidate.get("event_id") or matchup,
            bet_type,
            team,
            spread,
        )
    elif bet_type == "total":
        type_label = "Game total"
        side = (side or "").upper()
        line = _line(candidate["line"])
        bet = f"{side} {line}"
        team = "Both teams"
        identity = (
            candidate.get("event_id") or matchup,
            bet_type,
            side,
            line,
        )
    elif bet_type == "player_prop":
        type_label = "Player prop"
        direction = candidate.get("direction", "OVER").upper()
        if candidate.get("safe_mode"):
            threshold = _line(candidate["safe_threshold"])
            bet = (
                f"{candidate['player']} — {candidate['prop_label']} "
                f"{threshold}+"
            )
            bet_line = threshold
        else:
            bet_line = _line(candidate["line"])
            bet = (
                f"{candidate['player']} — {candidate['prop_label']} "
                f"{direction} {bet_line}"
            )
        team = candidate.get("team") or "Team unavailable"
        identity = (
            candidate.get("event_id") or matchup,
            bet_type,
            candidate["player"],
            candidate.get("prop"),
            direction,
            bet_line,
        )
    else:
        raise ValueError(f"Unsupported checklist bet type: {bet_type}")

    selection_key = "bet_selection:" + "::".join(
        str(value) for value in identity)
    return {
        "selection_key": selection_key,
        "type": type_label,
        "bet": bet,
        "matchup": matchup,
        "team": team,
    }


# ── Parlay / SGP pricing moved to parlay.py (re-exported for compatibility) ────
from parlay import (  # noqa: E402
    _parlay_value_joint,
    _cholesky,
    _make_psd_cholesky,
    _box_muller_pairs,
    _gaussian_copula_joint_prob,
    _normalize_legs,
    _has_hard_conflict,
    _pair_correlation,
    _build_corr_matrix,
    _copula_joint_hit_prob,
    _correlation_penalty,
    _same_team_prop_count,
    _score_parlay,
    generate_parlays,
)


# ── Player-prop pricing moved to props.py (re-exported for compatibility) ──────
from props import (  # noqa: E402
    analyze_player_props_value,
    format_props_report,
    _log5_rate,
    _mlb_prop_matchup_mult,
    _lineup_exposure_mult,
    _mlb_lineup_exposure_mult,
    _player_prop_defense_strength,
    _player_prop_venue_strength,
    _player_prop_output_defense_strength,
    _player_prop_shrinkage_k,
    _player_prop_park_strength,
    _park_factor_mult,
    _player_prop_half_life,
    _MLB_LEAGUE,
    _LINEUP_ADJ_CACHE,
)
