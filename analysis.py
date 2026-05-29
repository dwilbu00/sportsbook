"""
Analysis engine for comparing sportsbook odds against historical data.
Identifies value bets where book implied probability < historical probability.
"""

import heapq
import math
import random

from odds_client import PROP_LABELS, american_to_decimal


def _decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds (rounded to nearest integer)."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return 0
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def _consensus_price_for_line(items, line, line_key):
    """Pick a representative price for entries matching `line` (median)."""
    if not items or line is None:
        return None
    matching = [o["price"] for o in items if o.get(line_key) == line]
    if not matching:
        return None
    sorted_prices = sorted(matching)
    return sorted_prices[len(sorted_prices) // 2]
from calibration_loader import (
    load_calibration,
    apply_calibration_with_warmup,
    count_current_season_games,
)
from prop_filter import filter_player_gamelog
from recalibration import (
    load_recalibration,
    apply_platt,
    log_prediction,
    maybe_auto_refit,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Pure-Python statistics helpers (no numpy/scipy dependency)
# ──────────────────────────────────────────────────────────────────────────────
_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _norm_ppf(p):
    """
    Inverse standard normal CDF (Acklam's rational approximation).
    Accurate to ~1e-9 across the full (0, 1) interval.
    """
    if p <= 0.0:
        return -8.0  # effectively -inf for our use
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1
        return num / den
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1
    return num / den


def _cholesky(m):
    """
    Cholesky decomposition. Returns lower-triangular L such that L @ L^T = m.
    Raises ValueError if m is not positive-definite.
    """
    n = len(m)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = m[i][i] - s
                if v <= 0:
                    raise ValueError("matrix is not positive-definite")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (m[i][j] - s) / L[j][j]
    return L


def _make_psd_cholesky(R):
    """
    Return Cholesky factor of R, shrinking off-diagonals toward 0 if needed
    so the result is positive-definite. Always succeeds (identity fallback).
    """
    n = len(R)
    for shrink in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        m = [[R[i][j] if i == j else shrink * R[i][j] for j in range(n)]
             for i in range(n)]
        try:
            return _cholesky(m)
        except ValueError:
            continue
    return _cholesky([[1.0 if i == j else 0.0 for j in range(n)]
                      for i in range(n)])


def _box_muller_pairs(rng, n):
    """Generate `n` standard-normal samples using Box-Muller. Returns a list."""
    out = []
    while len(out) < n:
        u1 = rng.random()
        if u1 <= 0.0:
            continue
        u2 = rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out.append(r * math.cos(theta))
        if len(out) < n:
            out.append(r * math.sin(theta))
    return out


def _gaussian_copula_joint_prob(probs, corr_matrix, n_samples=5000, seed=42):
    """
    Monte-Carlo estimate of P(all events occur) under a Gaussian copula with
    given Bernoulli marginals `probs` and correlation matrix `corr_matrix`.

    Independence (corr_matrix = I) recovers the product ∏ p_i in expectation.
    """
    n = len(probs)
    if n == 0:
        return 1.0
    if n == 1:
        return max(0.0, min(1.0, probs[0]))
    if any(p <= 0.0 for p in probs):
        return 0.0
    if all(p >= 1.0 for p in probs):
        return 1.0

    thresholds = [_norm_ppf(p) for p in probs]
    L = _make_psd_cholesky(corr_matrix)
    rng = random.Random(seed)

    hits = 0
    for _ in range(n_samples):
        zi = _box_muller_pairs(rng, n)
        # z = L @ zi  (only need lower-triangular sums)
        ok = True
        for i in range(n):
            zi_i = 0.0
            Li = L[i]
            for k in range(i + 1):
                zi_i += Li[k] * zi[k]
            if zi_i > thresholds[i]:
                ok = False
                break
        if ok:
            hits += 1
    return hits / n_samples


# Per-sport exponential-decay half-life (in games).
# A game played `half_life` games ago contributes half the weight of the most
# recent game. Tuned to reflect typical week-to-week volatility per sport.
RECENCY_HALF_LIFE = {
    "basketball_nba": 10,
    "americanfootball_nfl": 4,
    "baseball_mlb": 7,   # tuned via backtest on 2024-25 MLB season (Winner % +2.3pp vs hl=18)
    "icehockey_nhl": 10,
}
DEFAULT_HALF_LIFE = 10


def _half_life_for(sport_key):
    """Return the recency half-life (in games) for a given sport key."""
    if sport_key is None:
        return DEFAULT_HALF_LIFE
    return RECENCY_HALF_LIFE.get(sport_key, DEFAULT_HALF_LIFE)


def _recency_weights(n, half_life):
    """
    Build a list of exponential-decay weights, ordered most-recent first
    (index 0 = newest game). A game `half_life` games ago receives weight 0.5.

    Parameters:
        n (int): Number of games to weight.
        half_life (float): Half-life in games. None or <= 0 disables decay.

    Returns:
        list[float]: Weights of length n.
    """
    if n <= 0:
        return []
    if not half_life or half_life <= 0:
        return [1.0] * n
    decay = math.log(2) / half_life
    return [math.exp(-decay * i) for i in range(n)]


def _weighted_mean(values, weights):
    """Weighted mean. Falls back to unweighted mean if weights sum to zero."""
    if not values:
        return 0.0
    total_w = sum(weights)
    if total_w == 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _weighted_rate(values, weights, predicate):
    """Weighted fraction (0..1) of values for which predicate(v) is True."""
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(w for v, w in zip(values, weights) if predicate(v)) / total_w


def _weighted_quantile(values, weights, q):
    """
    Weighted empirical quantile. Returns the smallest value v such that the
    cumulative weight ≤ v is ≥ q · total_weight. q=0 → min, q=1 → max.
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


def _weighted_std(values, weights, mean=None):
    """
    Weighted standard deviation. Falls back to unweighted std if weights
    sum to zero. Returns 0.0 for samples of size < 2.
    """
    if not values or len(values) < 2:
        return 0.0
    total_w = sum(weights)
    if total_w <= 0:
        m = sum(values) / len(values) if mean is None else mean
        var = sum((v - m) ** 2 for v in values) / len(values)
        return math.sqrt(var)
    if mean is None:
        mean = sum(v * w for v, w in zip(values, weights)) / total_w
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total_w
    return math.sqrt(var)


def _normal_inv_cdf(p):
    """
    Inverse standard-normal CDF (probit). Acklam's rational approximation,
    accurate to ~1e-9 across the full domain. Used by safe-mode to translate
    a target hit-rate into a z-score for parametric lower-bound thresholds.
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    # Coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def _opponent_strength_multiplier(opp_win_pct):
    """
    Map an opponent's win percentage (0..1) to a "meaningfulness" multiplier
    applied on top of the recency weight. A game against a 0.500 opponent
    receives multiplier 1.0; the range is bounded to [0.5, 1.5].

    Intuition: results vs strong opponents are more informative; results vs
    weak opponents are less so. This adjusts every recent game's contribution
    accordingly, in addition to its recency weight.
    """
    if opp_win_pct is None:
        return 1.0
    clamped = max(0.0, min(1.0, opp_win_pct))
    return 0.5 + clamped


# Per-sport venue-match weights (match, mismatch). A past game whose venue
# (home/away) matches the upcoming game gets the first multiplier; mismatched
# venue gets the second. Larger spread = stronger home-court signal.
VENUE_MATCH_WEIGHTS = {
    "basketball_nba": (1.25, 0.85),
    "americanfootball_nfl": (1.20, 0.85),
    "icehockey_nhl": (1.10, 0.95),
    "baseball_mlb": (1.40, 0.60),  # tuned via backtest on 2024-25 MLB season
}
DEFAULT_VENUE_WEIGHTS = (1.15, 0.90)


def _venue_match_multiplier(past_is_home, upcoming_is_home, sport_key):
    """
    Return a multiplier that up-weights past games played at the same venue
    type (home vs road) as the upcoming game. If either side's venue is
    unknown, return 1.0 (no adjustment).
    """
    if past_is_home is None or upcoming_is_home is None:
        return 1.0
    match_w, mismatch_w = VENUE_MATCH_WEIGHTS.get(sport_key, DEFAULT_VENUE_WEIGHTS)
    return match_w if past_is_home == upcoming_is_home else mismatch_w


# Per-sport strength of the opponent-defense multiplier for player props.
# 0.0 disables it. Tuned per backtest:
#   - NBA: 0.0 (no measurable effect; adds noise — backtest sweep showed
#     MAE delta < 0.001 and Hit% slightly worse with defense weighting)
PLAYER_PROP_DEFENSE_STRENGTH = {
    "basketball_nba": 0.0,
}
DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH = 1.0


# Per-sport strength of the venue-match multiplier for player props (separate
# from team-level VENUE_MATCH_WEIGHTS because individual players' stat lines
# are less venue-sensitive than team-level scoring). 0.0 disables it.
# Tuned per backtest sweep on 18 NBA starters × 60 games:
#   - NBA: 0.0 — combined sweep showed ven=0.25 worsens MAE by ~0.006 and
#     leaves hit-rate essentially flat (52.82% → 52.87%).
PLAYER_PROP_VENUE_STRENGTH = {
    "basketball_nba": 0.0,
}
DEFAULT_PLAYER_PROP_VENUE_STRENGTH = None  # None = inherit from VENUE_MATCH_WEIGHTS


# Per-sport strength of the OUTPUT-side opponent-defense adjustment for
# player props. Unlike PLAYER_PROP_DEFENSE_STRENGTH (which down/up-weights
# *prior* games against tough defenses), this multiplier scales the final
# projection up/down based on TONIGHT's opponent's defense.
#   projection *= 1 + strength * (opp_pts_allowed / league_avg − 1)
# Tuned per backtest sweep on 18 NBA starters × 60 games:
#   - NBA: 1.0 — best single-feature gain: MAE 3.774 → 3.751, Hit% +2.79pp,
#     bias drops +0.086 → +0.060.
PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH = {
    "basketball_nba": 1.0,
}
DEFAULT_PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH = 0.0  # off by default for unknown sports


# Bayesian shrinkage of the recency-weighted projection toward the unweighted
# (season-long) prior mean. `k` is in pseudo-observations:
#   projection = (eff_n * weighted_mean + k * unweighted_mean) / (eff_n + k)
# Regularizes the projection so small-sample / volatile players aren't over-fit
# to their most recent few games.
# Tuned per backtest on 18 NBA starters × 60 games:
#   - NBA: 10 — Combined MAE 3.751 → 3.742 (−0.009), monotonic improvement
#     k ∈ {3,5,10}. Negligible cost. Hit% essentially unchanged.
PLAYER_PROP_SHRINKAGE_K = {
    "basketball_nba": 10,
}
DEFAULT_PLAYER_PROP_SHRINKAGE_K = 0  # off by default for unknown sports


# Per-sport override for the recency half-life *for player props specifically*.
# When set, overrides the team-level RECENCY_HALF_LIFE for the player-prop
# projection. When None, inherits from RECENCY_HALF_LIFE.
# Tuned per backtest on 18 NBA starters × 60 games:
#   - NBA: 7 — Gives the lowest total safe-mode cushion@80% (11.58 vs 11.62
#     at hl=10) at a negligible MAE cost (+0.008, ~0.2%). Prioritized for
#     safe-mode usage. Team-level matchup analysis still uses hl=10.
PLAYER_PROP_HALF_LIFE = {
    "basketball_nba": 7,
}
DEFAULT_PLAYER_PROP_HALF_LIFE = None  # None = inherit from RECENCY_HALF_LIFE


def _player_prop_defense_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH
    return PLAYER_PROP_DEFENSE_STRENGTH.get(sport_key, DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH)


def _player_prop_venue_strength(sport_key):
    """
    Per-sport override for the venue-match multiplier *as applied to player
    props*. Returns None when the team-level VENUE_MATCH_WEIGHTS should be
    used (the historical default behavior).
    """
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_VENUE_STRENGTH
    return PLAYER_PROP_VENUE_STRENGTH.get(sport_key, DEFAULT_PLAYER_PROP_VENUE_STRENGTH)


def _player_prop_output_defense_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH
    return PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH.get(
        sport_key, DEFAULT_PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH)


def _player_prop_shrinkage_k(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_SHRINKAGE_K
    return PLAYER_PROP_SHRINKAGE_K.get(sport_key, DEFAULT_PLAYER_PROP_SHRINKAGE_K)


def _player_prop_half_life(sport_key):
    """
    Per-sport override for the recency half-life applied to player props.
    Falls back to the team-level RECENCY_HALF_LIFE when not overridden.
    """
    if sport_key is None:
        return _half_life_for(None)
    override = PLAYER_PROP_HALF_LIFE.get(sport_key, DEFAULT_PLAYER_PROP_HALF_LIFE)
    return override if override is not None else _half_life_for(sport_key)


def _opponent_defense_multiplier(opp_pts_allowed, league_avg_pts_allowed, strength=1.0):
    """
    Player-prop opponent-defense multiplier. A game against a defense that
    allows fewer points (tougher D) is up-weighted; vs a soft D it is
    down-weighted. The bounded ratio's distance from 1.0 is scaled by `strength`
    (0.0 = disabled, 1.0 = full effect bounded to [0.5, 1.5]).
    """
    if (not opp_pts_allowed or not league_avg_pts_allowed
            or league_avg_pts_allowed <= 0 or strength <= 0):
        return 1.0
    ratio = league_avg_pts_allowed / opp_pts_allowed
    bounded = max(0.5, min(1.5, ratio))
    return 1.0 + strength * (bounded - 1.0)


def analyze_moneyline_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0, sport_key=None):
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

    for team_name, stats in [(home_team, home_team_stats), (away_team, away_team_stats)]:
        ml_odds = game_odds["moneyline"].get(team_name, [])
        if not ml_odds:
            continue

        # Average implied probability across all books
        avg_implied = sum(o["implied_prob"] for o in ml_odds) / len(ml_odds)

        # Historical probability: weighted blend of season and recent form.
        # Replace the flat recent win% with an exponentially-weighted win rate
        # built from recent_games (newest first) so streaks/slumps matter more.
        season_wp = stats["season"]["win_pct"]
        flat_recent_wp = stats["recent"]["win_pct"]

        # Whether this team is playing the upcoming game at home.
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
            # opp_strength was removed: backtest on 1,153 NBA + 2,350 MLB games
            # showed it contributes zero signal at team level (balanced schedules
            # mean opponent_win_pct averages out across the season).
            base_weights = _recency_weights(len(wins_for_team), half_life)
            weights = [
                bw * _venue_match_multiplier(past_h, upcoming_is_home, sport_key)
                for bw, past_h in zip(base_weights, past_is_home_list)
            ]
            recent_wp = _weighted_mean(wins_for_team, weights)
        else:
            recent_wp = flat_recent_wp

        # Weight recent form more heavily (60/40)
        hist_prob = (0.4 * season_wp) + (0.6 * recent_wp)

        edge = hist_prob - avg_implied

        best_odds = max(ml_odds, key=lambda o: o["implied_prob"] if o["price"] > 0 else -o["price"])
        worst_book_prob = min(o["implied_prob"] for o in ml_odds)
        best_edge = hist_prob - worst_book_prob

        result = {
            "type": "moneyline",
            "team": team_name,
            "opponent": away_team if team_name == home_team else home_team,
            "home_away": "HOME" if team_name == home_team else "AWAY",
            "book_implied_prob": round(avg_implied * 100, 2),
            "season_win_pct": round(season_wp * 100, 2),
            "recent_win_pct": round(recent_wp * 100, 2),
            "hist_prob": round(hist_prob * 100, 2),
            "edge_pct": round(edge * 100, 2),
            "best_edge_pct": round(best_edge * 100, 2),
            "best_book": min(ml_odds, key=lambda o: o["implied_prob"])["book"],
            "best_price": min(ml_odds, key=lambda o: o["implied_prob"])["price"],
            "is_value": edge >= threshold,
        }
        candidates.append(result)

    return candidates


def analyze_totals_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0, sport_key=None):
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

    # Determine over/under probability from historical games (recency-,
    # opponent-strength-, and venue-match-weighted across both teams).
    over_weight = 0.0
    total_weight = 0.0
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
            if g["total_score"] > consensus_line:
                over_weight += w

    over_hit_rate = (over_weight / total_weight) if total_weight > 0 else 0.5

    diff = projected_total - consensus_line

    # Prices for the consensus line, picked from each side's odds list.
    over_price = _consensus_price_for_line(over_odds, consensus_line, "line")
    under_price = _consensus_price_for_line(under_odds, consensus_line, "line")

    candidates.append({
        "type": "total_over",
        "matchup": f"{game_odds['away_team']} @ {game_odds['home_team']}",
        "line": consensus_line,
        "projected_total": round(projected_total, 2),
        "diff_from_line": round(diff, 2),
        "over_hit_rate": round(over_hit_rate * 100, 2),
        "home_avg_scored": round(home_avg_scored, 2),
        "away_avg_scored": round(away_avg_scored, 2),
        "is_over_value": diff > 0 and over_hit_rate > 0.5 + threshold,
        "is_under_value": diff < 0 and (1 - over_hit_rate) > 0.5 + threshold,
        "over_price": over_price,
        "under_price": under_price,
    })

    return candidates


def analyze_spreads_value(game_odds, home_team_stats, away_team_stats, threshold_pct=5.0, sport_key=None):
    """
    Compare spread lines against historical scoring margins, using a JOINT
    distribution of the predicted game margin (home perspective).

    For each team, the model estimates the weighted mean and weighted std of
    that team's recent margins. The game's actual margin is then approximated
    as Normal(home_mean − away_mean, sqrt(home_var + away_var)) under
    independence. Cover probabilities are derived from this joint distribution,
    so home_cover_prob + away_cover_prob ≈ 1 (zero-sum, vig aside) — only one
    side can be a value bet in a given matchup.

    Parameters:
        game_odds (dict): Parsed game odds from odds_client.parse_game_odds()
        home_team_stats (dict): Home team stats with 'recent' and 'recent_games' keys
        away_team_stats (dict): Away team stats with 'recent' and 'recent_games' keys
        threshold_pct (float): Minimum edge to flag as value

    Returns:
        list: Value candidates for spread bets (at most one will be is_value=True per game)
    """
    threshold = threshold_pct / 100.0
    half_life = _half_life_for(sport_key)

    home_team = game_odds["home_team"]
    away_team = game_odds["away_team"]

    # ── Per-team weighted margin distributions ──────────────────────────────
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

    # ── Consensus spread per team ───────────────────────────────────────────
    def _consensus_spread(team):
        spread_odds = game_odds["spreads"].get(team, [])
        if not spread_odds:
            return None
        spreads = [o["spread"] for o in spread_odds]
        return max(set(spreads), key=spreads.count)

    home_spread = _consensus_spread(home_team)
    away_spread = _consensus_spread(away_team)

    # ── Joint margin distribution (home perspective) ────────────────────────
    # If we have stats for both teams, build the joint estimate. Otherwise we
    # cannot compute a meaningful cover probability for either side and skip
    # the matchup entirely (better than recommending both halves of a bet on
    # half the information).
    if not home_stats or not away_stats:
        return []

    pred_margin = home_stats["mean"] - away_stats["mean"]
    # Floor each team's std at 1.0 point to avoid degenerate certainty when a
    # team has very few recent games or unusually flat margins.
    home_var = max(home_stats["std"], 1.0) ** 2
    away_var = max(away_stats["std"], 1.0) ** 2
    pred_std = math.sqrt(home_var + away_var)

    # ── Build candidate per team using the joint cover probability ──────────
    candidates = []
    games_sampled = min(len(home_stats["margins"]), len(away_stats["margins"]))

    def _add_candidate(team_name, opponent, is_home, spread, cover_prob, team_avg_margin, price):
        edge = cover_prob - 0.50
        candidates.append({
            "type": "spread",
            "team": team_name,
            "opponent": opponent,
            "home_away": "HOME" if is_home else "AWAY",
            "spread": spread,
            "avg_margin": round(team_avg_margin, 2),
            "cover_rate": round(cover_prob * 100, 2),
            "games_sampled": games_sampled,
            "edge_pct": round(edge * 100, 2),
            "is_value": edge >= threshold,
            "pred_game_margin": round(pred_margin, 2),
            "pred_game_std": round(pred_std, 2),
            "price": price,
        })

    if home_spread is not None:
        # Home covers iff actual_margin + home_spread > 0  ⇔  margin > -home_spread.
        # P(margin > -home_spread) = Φ((pred_margin + home_spread) / pred_std)
        home_cover_prob = _norm_cdf((pred_margin + home_spread) / pred_std)
        home_price = _consensus_price_for_line(
            game_odds["spreads"].get(home_team, []), home_spread, "spread")
        _add_candidate(home_team, away_team, True, home_spread,
                       home_cover_prob, home_stats["mean"], home_price)
    if away_spread is not None:
        # Away covers iff -actual_margin + away_spread > 0  ⇔  margin < away_spread.
        # P(margin < away_spread) = Φ((away_spread - pred_margin) / pred_std)
        away_cover_prob = _norm_cdf((away_spread - pred_margin) / pred_std)
        away_price = _consensus_price_for_line(
            game_odds["spreads"].get(away_team, []), away_spread, "spread")
        _add_candidate(away_team, home_team, False, away_spread,
                       away_cover_prob, away_stats["mean"], away_price)

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


def analyze_player_props_value(prop_data, player_histories, threshold_pct=5.0,
                               sport_key=None, team_defense=None, espn_teams=None,
                               safe_mode=False, safe_target=0.95,
                               team_schedules=None):
    """
    Compare player prop lines against historical stat values from ESPN.

    Parameters:
        prop_data (dict): Parsed player props from odds_client.parse_player_props()
        player_histories (dict): {player_name: {prop_key: stat_history_dict}}
        threshold_pct (float): Minimum edge to flag as value
        sport_key (str): Sport key for recency half-life selection
        team_defense (dict): Optional {team_display_name: avg_points_allowed}
            lookup. When provided, each historical game's weight is multiplied
            by an opponent-defense factor.
        espn_teams (dict): Optional {display_name: team_info} lookup. When
            provided, each player's upcoming home/away status is resolved
            (via team_id from their gamelog) and a venue-match multiplier is
            applied per past game.

    Returns:
        list: Value candidates for player props
    """
    candidates = []
    threshold = threshold_pct / 100.0
    # Persistent per-(sport, prop) calibration overrides the in-code defaults
    # when available; absent file → falls back to defaults below.
    calibration = load_calibration(sport_key) if sport_key else {}

    # Self-updating Platt recalibration: on first call per process per sport,
    # resolve any past-game predictions to outcomes and refit Platt params
    # if enough new data has accumulated. Cheap if nothing to do.
    if sport_key:
        maybe_auto_refit(sport_key)
    recalibration = load_recalibration(sport_key) if sport_key else {}

    # Pull commence_time once so we can log this game's predictions for
    # later outcome resolution.
    commence_iso = prop_data.get("commence_time")
    log_game_date = commence_iso[:10] if commence_iso else None

    def _knob(prop_key, name, default):
        cfg = calibration.get(prop_key) if calibration else None
        if cfg and name in cfg and cfg[name] is not None:
            return cfg[name]
        return default

    default_half_life = _player_prop_half_life(sport_key)
    default_def_strength = _player_prop_defense_strength(sport_key)
    default_venue_override = _player_prop_venue_strength(sport_key)
    default_output_def = _player_prop_output_defense_strength(sport_key)
    default_shrinkage_k = _player_prop_shrinkage_k(sport_key)

    # Per-prop max strength across defaults + any calibrated overrides — used
    # only to decide whether league-average defense needs to be computed.
    def _any_prop_uses(name, default_val):
        if calibration:
            for cfg in calibration.values():
                if (cfg.get(name) or 0) > 0:
                    return True
        return (default_val or 0) > 0

    league_avg_def = None
    if team_defense and (_any_prop_uses("opp_defense_strength", default_def_strength)
                         or _any_prop_uses("output_def_strength", default_output_def)):
        vals = [v for v in team_defense.values() if v]
        if vals:
            league_avg_def = sum(vals) / len(vals)

    # Reverse lookup: team_id -> display_name (for venue resolution).
    id_to_name = {}
    if espn_teams:
        id_to_name = {str(info.get("id")): name for name, info in espn_teams.items() if info.get("id")}

    home_team_name = prop_data["home_team"]
    away_team_name = prop_data["away_team"]
    matchup = f"{away_team_name} @ {home_team_name}"

    for prop_key, players in prop_data.get("props", {}).items():
        # Resolve per-prop knobs: calibration overrides the in-code defaults
        # where present, else fall back to the per-sport defaults.
        half_life = _knob(prop_key, "half_life", default_half_life)
        defense_strength = _knob(prop_key, "opp_defense_strength", default_def_strength)
        venue_strength_override = _knob(prop_key, "venue_strength", default_venue_override)
        output_def_strength = _knob(prop_key, "output_def_strength", default_output_def)
        shrinkage_k = _knob(prop_key, "shrinkage_k", default_shrinkage_k)
        prop_calib_cfg = calibration.get(prop_key) if calibration else None

        for player_name, odds_info in players.items():
            line = odds_info["line"]
            over_implied = odds_info["over_implied"]
            under_implied = odds_info["under_implied"]
            over_price = odds_info["over_price"]
            under_price = odds_info["under_price"]

            history = player_histories.get(player_name, {}).get(prop_key)
            if not history or not history.get("found") or not history.get("values"):
                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "prop": prop_key,
                    "prop_label": PROP_LABELS.get(prop_key, prop_key),
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "over_implied": round(over_implied * 100, 2),
                    "under_implied": round(under_implied * 100, 2),
                    "avg_stat": None,
                    "over_rate": None,
                    "games_sampled": 0,
                    "edge_pct": 0,
                    "direction": None,
                    "is_value": False,
                    "no_history": True,
                })
                continue

            # values are ordered most-recent first (see espn_client.get_player_stat_history)
            values = history["values"]
            opponents = history.get("opponents") or [None] * len(values)
            past_home_aways = history.get("home_aways") or [None] * len(values)
            minutes = history.get("minutes") or [None] * len(values)
            game_dates = history.get("game_dates") or [None] * len(values)
            player_team_id = history.get("team_id")

            # ── Reliability filter ──
            # Drop low-minutes games AND the 1-game-pre + 1-game-post window
            # around any layoff (≥3 missed team games for NBA/MLB, ≥2 for NFL).
            # Also flags the player as "currently fragile" → skip prediction
            # when their last actual game was excluded (still injured /
            # ramping up) or their last game had limited minutes.
            synthetic = [
                {"game_date": gd, "MIN": m, "_value": v, "_opp": o, "_ha": ha}
                for v, o, ha, m, gd in zip(
                    values, opponents, past_home_aways, minutes, game_dates)
            ]
            team_schedule = None
            if team_schedules and player_team_id:
                team_schedule = team_schedules.get(str(player_team_id))
            filt = filter_player_gamelog(
                synthetic, team_schedule, sport_key, half_life=half_life)

            if filt["skip_prediction"]:
                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "prop": prop_key,
                    "prop_label": PROP_LABELS.get(prop_key, prop_key),
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "over_implied": round(over_implied * 100, 2),
                    "under_implied": round(under_implied * 100, 2),
                    "avg_stat": None,
                    "over_rate": None,
                    "games_sampled": 0,
                    "edge_pct": 0,
                    "direction": None,
                    "is_value": False,
                    "no_history": False,
                    "skip_reason": filt["skip_reason"],
                })
                continue

            eligible = filt["eligible_games"]
            values = [g["_value"] for g in eligible]
            opponents = [g["_opp"] for g in eligible]
            past_home_aways = [g["_ha"] for g in eligible]

            # Resolve the player's upcoming home/away by matching their team_id
            # to the home/away team names of the upcoming game.
            upcoming_is_home = None
            if player_team_id and id_to_name:
                player_team_name = id_to_name.get(str(player_team_id))
                if player_team_name == home_team_name:
                    upcoming_is_home = True
                elif player_team_name == away_team_name:
                    upcoming_is_home = False

            base_weights = _recency_weights(len(values), half_life)
            weights = []
            for bw, opp, past_h in zip(base_weights, opponents, past_home_aways):
                w = bw
                if team_defense and league_avg_def and defense_strength > 0:
                    w *= _opponent_defense_multiplier(
                        team_defense.get(opp), league_avg_def, defense_strength)
                # Venue multiplier: a per-sport PLAYER_PROP_VENUE_STRENGTH of
                # 0.0 disables it; None (default) inherits the team-level
                # VENUE_MATCH_WEIGHTS via _venue_match_multiplier.
                if venue_strength_override != 0.0:
                    w *= _venue_match_multiplier(past_h, upcoming_is_home, sport_key)
                weights.append(w)

            # ── Output-side opponent-defense adjustment ──
            # Scales the projection up/down based on TONIGHT's opponent's
            # defense (independent from per-prior-game weighting above).
            # Backtest finding: this is the single best feature gain for NBA
            # player props (MAE −0.6%, hit-rate +2.79pp).
            output_def_mult = 1.0
            if (output_def_strength > 0 and team_defense and league_avg_def
                    and upcoming_is_home is not None):
                opp_name = away_team_name if upcoming_is_home else home_team_name
                opp_pa = team_defense.get(opp_name)
                if opp_pa:
                    output_def_mult = 1.0 + output_def_strength * (
                        opp_pa / league_avg_def - 1.0)

            # ── Bayesian shrinkage toward unweighted prior mean ──
            # Regularizes the recency-weighted estimate by `k` pseudo-obs.
            base_proj = _weighted_mean(values, weights)
            if shrinkage_k > 0 and values:
                unweighted_mean = sum(values) / len(values)
                eff_n = sum(weights) if weights else 0.0
                if eff_n + shrinkage_k > 0:
                    base_proj = ((eff_n * base_proj) + (shrinkage_k * unweighted_mean)) / (eff_n + shrinkage_k)
            avg_stat = base_proj * output_def_mult
            # When the projection is scaled, the over-rate calc shifts the
            # comparison line by the inverse so historical frequencies are
            # interpreted in the projection's adjusted frame.
            effective_line = line / output_def_mult if output_def_mult else line
            empirical_over = _weighted_rate(values, weights, lambda v: v > effective_line)
            over_rate = empirical_over

            # ── Residual calibration with warmup blending ──
            # If a calibration file exists for this (sport, prop), replace the
            # raw empirical over-rate with a Brier-better calibrated probability.
            # Early-season players (few current-season games) blend with the
            # prior-season warmup distribution.
            calibration_meta = None
            if prop_calib_cfg and prop_calib_cfg.get("method"):
                game_dates = history.get("game_dates") or []
                curr_games = count_current_season_games(game_dates, sport_key)
                p_cal = apply_calibration_with_warmup(
                    prop_calib_cfg, avg_stat, line, curr_games,
                    empirical_over=empirical_over,
                )
                if p_cal is not None:
                    over_rate = max(0.0, min(1.0, p_cal))
                    warmup_games = prop_calib_cfg.get("warmup_games", 10) or 1
                    blend_w = min(curr_games / float(warmup_games), 1.0)
                    calibration_meta = {
                        "method": prop_calib_cfg.get("method"),
                        "curr_games": curr_games,
                        "warmup_games": warmup_games,
                        "blend_weight": round(blend_w, 3),
                        "empirical_over": round(empirical_over * 100, 2),
                    }

            # ── Safe mode (OVER-only, integer alt-line) ──
            if safe_mode:
                # values / weights already had DNPs filtered above.
                #
                # Parametric lower bound:
                #   threshold = round_down(projected_mean − z · weighted_std)
                # where z = Phi⁻¹(safe_target). For safe_target=0.95, z≈1.645.
                #
                # Why parametric instead of pure empirical quantile?
                # With ~10 games of history, the 5th-percentile empirical
                # quantile collapses to the sample minimum (a single bad
                # game's weight ≥ 5% of total). The parametric Normal bound
                # uses ALL recent games to estimate location + spread, which
                # gives a stable threshold instead of "Wemby 4+ points"
                # whenever a 4-pt foul-out game exists in his last 10.
                import math as _math
                if not values:
                    continue
                proj_mean = avg_stat  # already shrunk + def-adjusted
                wstd = _weighted_std(values, weights, mean=base_proj)
                # Scale std by the same output-defense factor so the spread
                # is in the same frame as the projection.
                wstd_adj = wstd * (output_def_mult if output_def_mult else 1.0)
                z = _normal_inv_cdf(safe_target)
                alt_q = proj_mean - z * wstd_adj

                # "Points {N}+" means the player needs actual ≥ N to win.
                # Floor (not ceil) for OVER thresholds: alt_q=8.7 → 8+,
                # because 8 is the largest integer the model expects them
                # to clear with ≥ safe_target probability.
                safe_threshold = max(1, int(_math.floor(alt_q)))

                # Empirical hit-rate at the chosen integer threshold (sanity).
                p_at_safe = _weighted_rate(values, weights,
                                           lambda v, t=safe_threshold: v >= t)

                # Empirical refinement: the parametric Normal floor can be
                # overly conservative when actual game-to-game spread is
                # tighter than wstd suggests. Bump the threshold up while
                # the empirical hit rate is still ≥ safe_target so the
                # suggested alt line is as tight as the player's history
                # actually supports.
                while True:
                    next_t = safe_threshold + 1
                    p_next = _weighted_rate(values, weights,
                                            lambda v, t=next_t: v >= t)
                    if p_next >= safe_target:
                        safe_threshold = next_t
                        p_at_safe = p_next
                    else:
                        break

                # Tight sanity guard: drop when historical hit rate at the
                # suggested threshold is more than 5pp below safe_target.
                # Was 15pp tolerance — measured to admit false positives
                # (NBA assists @ 95% claimed but actually hit 76%; MLB
                # batter_hits @ 85% claimed but actually hit 69%). The
                # 5pp band keeps out-of-sample hit rate within ~5pp of
                # the user-visible safe_target.
                if p_at_safe < (safe_target - 0.05):
                    continue

                # Floor-collapse guard: when safe_threshold is 1 (or 0), the
                # bet collapses to "did the player do the thing at all?" —
                # a binary-ish outcome whose true probability is just the
                # player's intrinsic rate of doing the thing once. The
                # parametric Normal quantile underestimates variance for
                # low-mean integer distributions and overstates how "safe"
                # this is. Measured: forward hit rate for "1+" suggestions
                # at high safe_target averages 65-80%, not 95%. Refuse to
                # market these as high-confidence picks above safe_target=80%.
                if safe_threshold <= 1 and safe_target > 0.80:
                    continue

                # Realism guard: if the safe threshold sits absurdly far
                # below the book line, two things are wrong:
                #   1. the player's game-to-game variance is so high that
                #      our 95% floor is unreliable (not actually "safe"),
                #   2. no sportsbook offers alt OVER lines that far below
                #      the main line, so the bet can't be placed anyway.
                # Require safe_threshold to be at least 50% of the book
                # line. (Wemby with line=27.5 → must be ≥14 to surface.)
                SAFE_MIN_RATIO = 0.5
                if line > 0 and safe_threshold < line * SAFE_MIN_RATIO:
                    continue

                # Our model's confidence at the standard book line.
                model_hit_at_line = _weighted_rate(values, weights,
                                                   lambda v: v > line)

                # Gap from book line to our safe threshold. Larger positive
                # gap = book line is below safe floor (bet straight OVER).
                # Negative = user must hunt for an alt OVER line ≤ (safe_threshold − 1).
                line_gap = safe_threshold - line
                bettable_at_standard_line = line < safe_threshold

                # Confidence delta between our safe suggestion and the book line.
                # Positive = our suggestion is safer than the standard line.
                model_delta = p_at_safe - model_hit_at_line

                # Edge in safe mode = how much MORE likely our safe
                # suggestion hits than the standard book line.
                # Always ≥ 0 (because safe_threshold ≤ line by construction
                # whenever p_at_safe ≥ safe_target). Used as edge_pct so the
                # parlay builder treats safe-mode legs as positive-edge.
                edge = model_delta

                # Filter trash: drop bets where even the standard book line
                # is below 50/50 model confidence AND the safe threshold sits
                # at the floor of the distribution (suggesting low-volume
                # player with no realistic upside). Keeps suggestions where
                # either the standard line is decent OR the safe threshold is
                # meaningfully above the floor.
                is_value = (safe_threshold >= 1
                            and (model_hit_at_line >= 0.50 or safe_threshold > 1))

                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "prop": prop_key,
                    "prop_label": PROP_LABELS.get(prop_key, prop_key),
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "over_implied": round(over_implied * 100, 2),
                    "under_implied": round(under_implied * 100, 2),
                    "avg_stat": round(avg_stat, 2),
                    "over_rate": round(over_rate * 100, 2),
                    "games_sampled": len(values),
                    "edge_pct": round(edge * 100, 2),
                    "direction": "OVER",
                    "best_price": over_price,
                    "is_value": is_value,
                    "no_history": False,
                    # ── Safe-mode-specific fields ──
                    "safe_mode": True,
                    "safe_target": safe_target,
                    "safe_threshold": safe_threshold,        # display as "{N}+"
                    "safe_alt_q": round(alt_q, 2),           # raw quantile (continuous)
                    "model_hit_at_safe": round(p_at_safe * 100, 2),     # prob at suggested
                    "model_hit_at_line": round(model_hit_at_line * 100, 2),  # prob at book line
                    "model_delta": round(model_delta * 100, 2),         # safe − book line
                    "line_gap": round(line_gap, 2),
                    "bettable_at_standard_line": bettable_at_standard_line,
                    "_values": list(values),
                    "_weights": list(weights),
                })
                continue

            # ── Platt recalibration (self-updating) ──
            # Stretch/shrink the over-rate so its calibration matches reality.
            # Fit from a JSONL log of past predictions + ESPN outcomes; auto-
            # refit gated by maybe_auto_refit() above.
            raw_over_rate = over_rate
            recal_cfg = recalibration.get(prop_key) if recalibration else None
            recal_meta = None
            if recal_cfg and recal_cfg.get("a") is not None:
                p_recal = apply_platt(over_rate,
                                      recal_cfg.get("a"),
                                      recal_cfg.get("b"))
                if p_recal is not None:
                    over_rate = max(0.0, min(1.0, p_recal))
                    recal_meta = {
                        "a": recal_cfg.get("a"),
                        "b": recal_cfg.get("b"),
                        "n_fit": recal_cfg.get("n_fit"),
                        "raw_prob": round(raw_over_rate * 100, 2),
                    }

            # Compare historical over rate vs book implied over probability
            over_edge = over_rate - over_implied
            under_rate = 1 - over_rate
            under_edge = under_rate - under_implied

            if over_edge > under_edge:
                direction = "OVER"
                edge = over_edge
                best_price = over_price
            else:
                direction = "UNDER"
                edge = under_edge
                best_price = under_price

            # Log the published probability so future refits learn from it.
            # We log the *raw* (pre-Platt) probability — that's what Platt
            # was fit against and what subsequent refits should map.
            if log_game_date and sport_key:
                log_prediction(
                    sport_key=sport_key,
                    prop_key=prop_key,
                    player=player_name,
                    game_date=log_game_date,
                    line=line,
                    raw_prob=raw_over_rate,
                    projected=avg_stat,
                    direction=direction,
                )

            candidates.append({
                "type": "player_prop",
                "matchup": matchup,
                "player": player_name,
                "prop": prop_key,
                "prop_label": PROP_LABELS.get(prop_key, prop_key),
                "line": line,
                "over_price": over_price,
                "under_price": under_price,
                "over_implied": round(over_implied * 100, 2),
                "under_implied": round(under_implied * 100, 2),
                "avg_stat": round(avg_stat, 2),
                "over_rate": round(over_rate * 100, 2),
                "games_sampled": len(values),
                "edge_pct": round(edge * 100, 2),
                "direction": direction,
                "best_price": best_price,
                "is_value": edge >= threshold,
                "no_history": False,
                "calibration": calibration_meta,
                "recalibration": recal_meta,
                "_values": list(values),
                "_weights": list(weights),
            })

    return candidates


def format_props_report(candidates):
    """Format player props value candidates into a readable report."""
    value_bets = [c for c in candidates if c["is_value"]]
    no_history = [c for c in candidates if c.get("no_history")]
    non_value = [c for c in candidates if not c["is_value"] and not c.get("no_history")]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  PLAYER PROPS ANALYSIS")
    lines.append("=" * 80)

    if value_bets:
        lines.append("")
        lines.append("  VALUE PROPS FOUND:")
        for c in sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True):
            lines.append("")
            lines.append(f"  >>> {c['player']} — {c['prop_label']} {c['direction']} {c['line']}")
            lines.append(f"      Matchup:        {c['matchup']}")
            lines.append(f"      Line:           {c['line']}  |  Over: {c['over_price']:+d}  |  Under: {c['under_price']:+d}")
            lines.append(f"      Avg Stat:       {c['avg_stat']} (last {c['games_sampled']} games)")
            lines.append(f"      Over Rate:      {c['over_rate']}% historical")
            lines.append(f"      Book Implied:   Over {c['over_implied']}% / Under {c['under_implied']}%")
            lines.append(f"      EDGE:           +{c['edge_pct']}% on {c['direction']} ({c['best_price']:+d})")
    else:
        lines.append("\n  No player prop value bets found above threshold.")

    if non_value:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Other Props (no edge above threshold)")
        lines.append("-" * 80)
        for c in non_value:
            edge_str = f"+{c['edge_pct']}%" if c['edge_pct'] > 0 else f"{c['edge_pct']}%"
            dir_str = c['direction'] or "?"
            lines.append(f"  {c['player']:25s} | {c['prop_label']:18s} | Line: {c['line']:5} | Avg: {c['avg_stat']:5} | {dir_str}: {edge_str}")

    if no_history:
        lines.append("")
        lines.append(f"  ({len(no_history)} prop(s) skipped — no ESPN history found)")

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


def _normalize_legs(all_ml, all_spreads, all_totals, all_props):
    """
    Convert all analysis results into a uniform leg format for parlay building.
    Only include legs with positive edge.
    
    Each leg dict has:
        game_key: str (e.g., "Team A @ Team B" or matchup)
        team: str or None
        bet_type: str (moneyline, spread, total_over, total_under, player_prop_over, player_prop_under)
        label: str (human readable description)
        player: str or None
        prop_key: str or None (e.g., "player_points")
        edge_pct: float
        odds_price: int or None (American odds)
        hist_prob: float (0-1, historical probability)
        implied_prob: float (0-1, book implied probability)
    """
    legs = []
    
    for c in all_ml:
        if c["edge_pct"] <= 0:
            continue
        game_key = f"{c['opponent']} @ {c['team']}" if c["home_away"] == "HOME" else f"{c['team']} @ {c['opponent']}"
        legs.append({
            "game_key": game_key,
            "team": c["team"],
            "bet_type": "moneyline",
            "label": f"{c['team']} ML ({c['home_away']})",
            "player": None,
            "prop_key": None,
            "edge_pct": c["edge_pct"],
            "odds_price": c.get("best_price"),
            "hist_prob": c["hist_prob"] / 100.0,
            "implied_prob": c["book_implied_prob"] / 100.0,
        })
    
    for c in all_spreads:
        if c["edge_pct"] <= 0 or c.get("games_sampled", 0) < 5:
            continue
        game_key = f"{c['opponent']} @ {c['team']}" if c["home_away"] == "HOME" else f"{c['team']} @ {c['opponent']}"
        legs.append({
            "game_key": game_key,
            "team": c["team"],
            "bet_type": "spread",
            "label": f"{c['team']} {c['spread']:+.2f}",
            "player": None,
            "prop_key": None,
            "edge_pct": c["edge_pct"],
            "odds_price": c.get("price"),
            "hist_prob": c["cover_rate"] / 100.0,
            "implied_prob": 0.50,
        })
    
    for c in all_totals:
        if c.get("is_over_value"):
            legs.append({
                "game_key": c["matchup"],
                "team": None,
                "bet_type": "total_over",
                "label": f"OVER {c['line']} ({c['matchup']})",
                "player": None,
                "prop_key": None,
                "edge_pct": c["over_hit_rate"] - 50.0,
                "odds_price": c.get("over_price"),
                "hist_prob": c["over_hit_rate"] / 100.0,
                "implied_prob": 0.50,
            })
        if c.get("is_under_value"):
            legs.append({
                "game_key": c["matchup"],
                "team": None,
                "bet_type": "total_under",
                "label": f"UNDER {c['line']} ({c['matchup']})",
                "player": None,
                "prop_key": None,
                "edge_pct": (100.0 - c["over_hit_rate"]) - 50.0,
                "odds_price": c.get("under_price"),
                "hist_prob": (100.0 - c["over_hit_rate"]) / 100.0,
                "implied_prob": 0.50,
            })
    
    for c in all_props:
        if c.get("no_history") or c["edge_pct"] <= 0 or c.get("games_sampled", 0) < 5:
            continue
        direction = c.get("direction", "OVER")
        bt = f"player_prop_{direction.lower()}"
        price = c.get("best_price", c.get("over_price") if direction == "OVER" else c.get("under_price"))

        if c.get("safe_mode"):
            # Safe-mode legs: bet is "{prop} {N}+" and the hist prob is the
            # model probability AT our safe threshold (not the book line).
            label = f"{c['player']} {c['prop_label']} {c['safe_threshold']}+"
            hp = (c.get("model_hit_at_safe", 0.0) or 0.0) / 100.0
            ip = (c["over_implied"] / 100.0) if c.get("over_implied") is not None else 0.5
        elif direction == "OVER":
            label = f"{c['player']} {c['prop_label']} {direction} {c['line']}"
            hp = (c["over_rate"] / 100.0) if c.get("over_rate") is not None else 0.5
            ip = (c["over_implied"] / 100.0) if c.get("over_implied") is not None else 0.5
        else:
            label = f"{c['player']} {c['prop_label']} {direction} {c['line']}"
            hp = (1.0 - c["over_rate"] / 100.0) if c.get("over_rate") is not None else 0.5
            ip = (c["under_implied"] / 100.0) if c.get("under_implied") is not None else 0.5

        leg = {
            "game_key": c["matchup"],
            "team": None,
            "bet_type": bt,
            "label": label,
            "player": c["player"],
            "prop_key": c.get("prop"),
            "edge_pct": c["edge_pct"],
            "odds_price": price,
            "hist_prob": hp,
            "implied_prob": ip,
        }
        if c.get("safe_mode"):
            # Extra fields used by the "value parlays in safe mode" ranker / UI.
            leg["safe_mode"] = True
            leg["safe_threshold"] = c.get("safe_threshold")
            leg["book_line"] = c.get("line")
            leg["line_gap"] = c.get("line_gap", 0.0)
            leg["model_hit_at_safe"] = c.get("model_hit_at_safe")
            leg["model_hit_at_line"] = c.get("model_hit_at_line")
            # If an alt-line price was fetched for the suggested safe
            # threshold, prefer it over the book-line price for parlay payout
            # calculation. The book-line price was over_price for the standard
            # line, not the threshold we're actually betting.
            if c.get("safe_alt_price") is not None:
                leg["odds_price"] = c["safe_alt_price"]
                leg["safe_alt_line"] = c.get("safe_alt_line")
        legs.append(leg)
    
    return legs


def _has_hard_conflict(leg_a, leg_b):
    """
    Check if two legs have a hard conflict (mutually exclusive or contradictory).
    These combos should NEVER appear in a parlay together.
    """
    same_game = leg_a["game_key"] == leg_b["game_key"]
    
    if not same_game:
        return False
    
    ta = leg_a["bet_type"]
    tb = leg_b["bet_type"]
    
    # Opposite moneylines in same game
    if ta == "moneyline" and tb == "moneyline":
        return leg_a["team"] != leg_b["team"]  # different teams = conflict
    
    # Opposite spreads in same game
    if ta == "spread" and tb == "spread":
        return leg_a["team"] != leg_b["team"]
    
    # Over + Under on same game total
    if {ta, tb} == {"total_over", "total_under"}:
        return True
    
    # Same player, same prop, opposite direction
    if "player_prop" in ta and "player_prop" in tb:
        if (leg_a["player"] == leg_b["player"] 
            and leg_a["prop_key"] == leg_b["prop_key"]
            and ta != tb):
            return True
    
    return False


def _pair_correlation(leg_a, leg_b, sport_key):
    """
    Estimate the rank correlation ρ ∈ [-0.5, 0.5] between two leg outcomes
    (each treated as a Bernoulli "did it hit").  Feeds the Gaussian copula
    that computes joint parlay hit probability.

    Rules mirror the heuristic synergies/conflicts in `_correlation_penalty`
    but in calibrated correlation units rather than arbitrary scores.
    """
    same_game = leg_a["game_key"] == leg_b["game_key"]
    ta = leg_a["bet_type"]
    tb = leg_b["bet_type"]

    # Cross-game legs: treat as effectively independent.
    if not same_game:
        return 0.0

    if sport_key == "basketball_nba":
        if "player_prop_over" in ta and "player_prop_over" in tb:
            return -0.20  # shared possession / usage cap
        if (ta == "total_under" and "player_prop_over" in tb and
                leg_b.get("prop_key") == "player_points"):
            return -0.40
        if (tb == "total_under" and "player_prop_over" in ta and
                leg_a.get("prop_key") == "player_points"):
            return -0.40
        if (ta == "total_over" and "player_prop_over" in tb and
                leg_b.get("prop_key") == "player_points"):
            return 0.30
        if (tb == "total_over" and "player_prop_over" in ta and
                leg_a.get("prop_key") == "player_points"):
            return 0.30
        if ta == "moneyline" and "player_prop_over" in tb:
            return 0.15
        if tb == "moneyline" and "player_prop_over" in ta:
            return 0.15

    elif sport_key == "americanfootball_nfl":
        if (ta == "moneyline" and "player_prop_over" in tb and
                leg_b.get("prop_key") == "player_pass_yds"):
            return 0.30
        if (tb == "moneyline" and "player_prop_over" in ta and
                leg_a.get("prop_key") == "player_pass_yds"):
            return 0.30
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_rush_yds"
                and tb == "total_under"):
            return -0.25
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_rush_yds"
                and ta == "total_under"):
            return -0.25
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_pass_yds"
                and tb == "total_over"):
            return 0.25
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_pass_yds"
                and ta == "total_over"):
            return 0.25
        if "player_prop_over" in ta and "player_prop_over" in tb:
            return -0.10

    elif sport_key == "baseball_mlb":
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts"
                and tb == "total_under"):
            return 0.35
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts"
                and ta == "total_under"):
            return 0.35
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts"
                and tb == "total_over"):
            return -0.30
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts"
                and ta == "total_over"):
            return -0.30
        if (ta == "moneyline" and "player_prop_over" in tb and
                leg_b.get("prop_key") == "batter_hits"):
            return 0.25
        if (tb == "moneyline" and "player_prop_over" in ta and
                leg_a.get("prop_key") == "batter_hits"):
            return 0.25

    # Generic same-game legs: small positive (shared game conditions).
    return 0.05


def _build_corr_matrix(legs, sport_key):
    """Symmetric correlation matrix with 1.0 on the diagonal."""
    n = len(legs)
    R = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            rho = _pair_correlation(legs[i], legs[j], sport_key)
            R[i][j] = rho
            R[j][i] = rho
    return R


def _copula_joint_hit_prob(legs, sport_key, n_samples=5000, seed=42):
    """Joint P(all legs hit) under a Gaussian copula keyed to `sport_key`."""
    probs = [max(0.0, min(1.0, leg["hist_prob"])) for leg in legs]
    R = _build_corr_matrix(legs, sport_key)
    return _gaussian_copula_joint_prob(probs, R, n_samples=n_samples, seed=seed)


def _correlation_penalty(leg_a, leg_b, sport_key):
    """
    Return a correlation penalty (negative = bad combo, positive = good synergy).
    Used as a cheap heuristic to score parlay candidates during enumeration;
    the final winner is re-ranked using the exact Gaussian copula joint prob.

    Returns a float:
        negative values = legs work against each other
        0 = neutral
        positive values = legs complement each other (positively correlated)
    """
    same_game = leg_a["game_key"] == leg_b["game_key"]
    ta = leg_a["bet_type"]
    tb = leg_b["bet_type"]
    
    # Cross-game parlays are preferred (less priced in by books)
    if not same_game:
        # Small bonus for cross-game diversification
        return 0.5
    
    # ── Same-game correlation rules ──
    
    # NBA-specific
    if sport_key == "basketball_nba":
        # Two player prop overs from same team = negative (usage cap)
        if "player_prop_over" in ta and "player_prop_over" in tb:
            # Same team check: if both players are in the same matchup, 
            # they might be on same team. We can't tell for sure from matchup alone,
            # but penalize same-game multi-prop overs
            return -2.0
        
        # Game total under + player points over = negative
        if (ta == "total_under" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "player_points"):
            return -3.0
        if (tb == "total_under" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "player_points"):
            return -3.0
        
        # Game total over + player points over = positive
        if (ta == "total_over" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "player_points"):
            return 1.5
        if (tb == "total_over" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "player_points"):
            return 1.5
        
        # Team ML + player prop over for same team = positive
        if ta == "moneyline" and "player_prop_over" in tb:
            return 1.0
        if tb == "moneyline" and "player_prop_over" in ta:
            return 1.0
    
    # NFL-specific
    elif sport_key == "americanfootball_nfl":
        # QB passing yards over + team ML = strong positive
        if (ta == "moneyline" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "player_pass_yds"):
            return 2.0
        if (tb == "moneyline" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "player_pass_yds"):
            return 2.0
        
        # RB rushing yards over + game under = negative
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_rush_yds" 
            and tb == "total_under"):
            return -2.0
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_rush_yds" 
            and ta == "total_under"):
            return -2.0
        
        # QB passing yards over + game over = positive
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_pass_yds" 
            and tb == "total_over"):
            return 1.5
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_pass_yds" 
            and ta == "total_over"):
            return 1.5
        
        # Multiple player prop overs same game = slight negative (usage)
        if "player_prop_over" in ta and "player_prop_over" in tb:
            return -1.0
    
    # MLB-specific
    elif sport_key == "baseball_mlb":
        # Pitcher K's over + game under = positive (dominant pitching)
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts" 
            and tb == "total_under"):
            return 2.0
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts" 
            and ta == "total_under"):
            return 2.0
        
        # Pitcher K's over + game over = negative
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts" 
            and tb == "total_over"):
            return -2.0
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts" 
            and ta == "total_over"):
            return -2.0
        
        # Batter hits over + team ML = positive
        if (ta == "moneyline" and "player_prop_over" in tb and 
            leg_b.get("prop_key") == "batter_hits"):
            return 1.5
        if (tb == "moneyline" and "player_prop_over" in ta and 
            leg_a.get("prop_key") == "batter_hits"):
            return 1.5
    
    # Default same-game slight penalty (less diversification)
    return -0.5


def _same_team_prop_count(legs):
    """Count how many player prop overs are from the same game (proxy for same team)."""
    game_prop_counts = {}
    for leg in legs:
        if "player_prop_over" in leg["bet_type"]:
            gk = leg["game_key"]
            game_prop_counts[gk] = game_prop_counts.get(gk, 0) + 1
    return max(game_prop_counts.values()) if game_prop_counts else 0


def _score_parlay(legs, sport_key, mode="value"):
    """
    Score a parlay combination. Higher is better.
    
    Modes:
        value: Prioritizes edge (higher edge = better)
        safe: Prioritizes probability of hitting (higher hist_prob = better)
    """
    # Pairwise correlation scoring
    correlation_score = 0.0
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            correlation_score += _correlation_penalty(legs[i], legs[j], sport_key)
    
    # Penalty for too many same-game player prop overs (usage cap)
    same_team_count = _same_team_prop_count(legs)
    usage_penalty = 0
    if same_team_count > 2:
        usage_penalty = -5.0 * (same_team_count - 2)
    
    # Combined probabilities
    combined_hist = 1.0
    combined_implied = 1.0
    for leg in legs:
        combined_hist *= leg["hist_prob"]
        combined_implied *= leg["implied_prob"]
    
    if mode == "safe":
        # Prioritize highest combined probability of hitting
        # Scale hist_prob heavily so it dominates the score
        prob_score = combined_hist * 1000
        # Still consider edge but weighted much less
        total_edge = sum(leg["edge_pct"] for leg in legs) * 0.1
        return prob_score + total_edge + correlation_score + usage_penalty
    elif mode == "safe_value":
        # "Value parlays" built from safe-mode candidates.
        # Rank by aggressiveness: prefer legs whose safe threshold sits AT or
        # ABOVE the book line (line_gap ≥ 0), with hit probability as a
        # tiebreaker. Non-safe legs (ML/spread/totals) fall back to edge_pct.
        gap_sum = 0.0
        nonsafe_edge_sum = 0.0
        for leg in legs:
            if leg.get("safe_mode"):
                gap_sum += leg.get("line_gap", 0.0)
            else:
                nonsafe_edge_sum += leg["edge_pct"]
        prob_score = combined_hist * 100
        return (gap_sum * 10) + nonsafe_edge_sum + prob_score + correlation_score + usage_penalty
    else:
        # Prioritize edge value
        total_edge = sum(leg["edge_pct"] for leg in legs)
        parlay_edge = (combined_hist - combined_implied) * 100
        return total_edge + correlation_score + usage_penalty + parlay_edge


def generate_parlays(all_ml, all_spreads, all_totals, all_props, sport_key, mode="value"):
    """
    Generate the top recommended 3, 4, and 5 leg parlays.
    
    Parameters:
        all_ml: Moneyline analysis results
        all_spreads: Spread analysis results
        all_totals: Totals analysis results
        all_props: Player prop analysis results
        sport_key: Sport key (e.g., 'basketball_nba')
        mode: 'value' (prioritize edge) or 'safe' (prioritize hit probability)
    
    Returns:
        dict: {3: parlay_dict, 4: parlay_dict, 5: parlay_dict}
    """
    from itertools import combinations
    
    legs = _normalize_legs(all_ml, all_spreads, all_totals, all_props)
    
    if len(legs) < 3:
        return {}

    # If the user asked for "value" parlays but the underlying analysis was run
    # in safe mode (props carry safe_mode=True), edge_pct is `model_delta`
    # (uplift of safe threshold vs book line) and `implied_prob` is at the book
    # line — these can't be used like a normal value edge. Switch to a
    # dedicated "safe_value" ranker that prefers safe legs whose threshold is
    # closest to (or above) the book line.
    has_safe_legs = any(leg.get("safe_mode") for leg in legs)
    effective_mode = mode
    if mode == "value" and has_safe_legs:
        effective_mode = "safe_value"

    # Sort and take top candidates to limit combinatorics
    if effective_mode == "safe":
        legs.sort(key=lambda x: x["hist_prob"], reverse=True)
    elif effective_mode == "safe_value":
        # Primary: line_gap (safe legs only; 0 for non-safe legs as a neutral
        # floor); Secondary: hist_prob. Both descending.
        legs.sort(
            key=lambda x: (x.get("line_gap", 0.0) if x.get("safe_mode") else 0.0,
                           x["hist_prob"]),
            reverse=True,
        )
    else:
        legs.sort(key=lambda x: x["edge_pct"], reverse=True)
    candidates = legs[:25]  # Cap at 25 to keep combos manageable
    
    results = {}

    # Re-rank the top-K heuristic candidates per size with the exact Gaussian
    # copula joint hit probability. Cap K low enough to keep MC cost bounded.
    RERANK_TOP_K = 30
    MC_SAMPLES = 5000

    for size in [3, 4, 5]:
        if len(candidates) < size:
            continue

        # ── Stage 1: heuristic enumeration → keep top-K candidates ──
        top_heap = []  # (heuristic_score, tiebreak, combo_list)
        tiebreak = 0
        for combo in combinations(candidates, size):
            combo_list = list(combo)

            has_conflict = False
            for i in range(len(combo_list)):
                for j in range(i + 1, len(combo_list)):
                    if _has_hard_conflict(combo_list[i], combo_list[j]):
                        has_conflict = True
                        break
                if has_conflict:
                    break
            if has_conflict:
                continue

            h_score = _score_parlay(combo_list, sport_key, effective_mode)
            tiebreak += 1
            if len(top_heap) < RERANK_TOP_K:
                heapq.heappush(top_heap, (h_score, tiebreak, combo_list))
            elif h_score > top_heap[0][0]:
                heapq.heapreplace(top_heap, (h_score, tiebreak, combo_list))

        if not top_heap:
            continue

        # ── Stage 2: re-rank survivors with the Gaussian copula joint prob ──
        best_parlay = None
        best_score = float("-inf")
        best_joint = 0.0
        best_combined_implied = 0.0
        best_combined_edge = 0.0

        for _, _, combo_list in top_heap:
            combined_implied = 1.0
            combined_edge = 0.0
            for leg in combo_list:
                combined_implied *= leg["implied_prob"]
                combined_edge += leg["edge_pct"]

            joint = _copula_joint_hit_prob(
                combo_list, sport_key,
                n_samples=MC_SAMPLES,
                seed=42 + size,
            )

            if effective_mode == "safe":
                score = joint * 1000 + combined_edge * 0.1
            elif effective_mode == "safe_value":
                gap_sum = sum(
                    leg.get("line_gap", 0.0)
                    for leg in combo_list
                    if leg.get("safe_mode")
                )
                nonsafe_edge_sum = sum(
                    leg["edge_pct"]
                    for leg in combo_list
                    if not leg.get("safe_mode")
                )
                score = (gap_sum * 10) + nonsafe_edge_sum + joint * 100
            else:
                score = combined_edge + (joint - combined_implied) * 100

            if score > best_score:
                best_score = score
                best_parlay = combo_list
                best_joint = joint
                best_combined_implied = combined_implied
                best_combined_edge = combined_edge

        if best_parlay:
            combined_hist_indep = 1.0
            for leg in best_parlay:
                combined_hist_indep *= leg["hist_prob"]

            # When the parlay contains ANY safe-mode leg, `combined_edge`
            # (sum of per-leg edge_pct, which is model_delta for safe legs)
            # and `parlay_edge_pct` (joint-vs-combined-implied where the
            # implied side is the BOOK LINE, not the safe threshold) compare
            # quantities at different lines — apples to oranges. Suppress them
            # only in that case. Regular analysis + Safe Parlays button still
            # produces meaningful values here.
            has_safe = any(leg.get("safe_mode") for leg in best_parlay)
            if has_safe:
                combined_edge_out = None
                parlay_edge_out = None
            else:
                combined_edge_out = round(best_combined_edge, 2)
                parlay_edge_out = round((best_joint - best_combined_implied) * 100, 2)

            # Gap stats for the safe_value display (avg/total line gap across
            # safe legs only).
            safe_gaps = [leg.get("line_gap", 0.0)
                         for leg in best_parlay if leg.get("safe_mode")]
            total_line_gap = round(sum(safe_gaps), 2) if safe_gaps else None
            avg_line_gap = (round(sum(safe_gaps) / len(safe_gaps), 2)
                            if safe_gaps else None)

            # Parlay payout: product of each leg's decimal odds. Legs missing a
            # price (rare — primarily older totals/spreads from cached data)
            # are treated as -110 (decimal 1.909), which is the standard US
            # spread/total price. `payout_uses_default` flags when this fallback
            # was applied so the UI can warn.
            decimal_product = 1.0
            payout_uses_default = False
            for leg in best_parlay:
                price = leg.get("odds_price")
                if price is None:
                    decimal_product *= american_to_decimal(-110)
                    payout_uses_default = True
                else:
                    decimal_product *= american_to_decimal(price)
            parlay_decimal = round(decimal_product, 3)
            parlay_american = _decimal_to_american(decimal_product)
            payout_per_10 = round((decimal_product - 1.0) * 10, 2)

            # Same-game parlay: 2+ legs in the same matchup. DK (and other
            # books) apply proprietary correlation adjustments to SGPs so the
            # actual book payout will differ from the naive multiplied price.
            game_keys = [leg.get("game_key") for leg in best_parlay]
            has_sgp = len(game_keys) != len(set(game_keys))

            results[size] = {
                "legs": best_parlay,
                "score": best_score,
                "mode": effective_mode,
                "combined_edge": combined_edge_out,
                "combined_hist_prob": round(best_joint * 100, 2),
                "combined_hist_prob_indep": round(combined_hist_indep * 100, 2),
                "combined_implied_prob": round(best_combined_implied * 100, 2),
                "parlay_edge_pct": parlay_edge_out,
                "correlation_adjustment_pct": round(
                    (best_joint - combined_hist_indep) * 100, 2
                ),
                "total_line_gap": total_line_gap,
                "avg_line_gap": avg_line_gap,
                # ── Book payout (computed client-side from leg prices) ──
                "parlay_decimal_odds": parlay_decimal,
                "parlay_american_odds": parlay_american,
                "payout_per_10": payout_per_10,
                "payout_uses_default_price": payout_uses_default,
                "has_sgp": has_sgp,
            }

    return results
