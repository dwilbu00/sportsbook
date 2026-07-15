"""
Analysis engine for comparing sportsbook odds against historical data.
Identifies value bets where book implied probability < historical probability.
"""

import heapq
import math
import random

import mlb_starters
from odds_client import (
    PROP_LABELS,
    american_to_decimal,
    american_to_implied_prob,
    devig_two_way,
)


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


def _expected_roi(probability, american_price):
    """Expected profit per dollar staked, or None when no price is available."""
    if american_price is None:
        return None
    return probability * american_to_decimal(american_price) - 1.0
from calibration_loader import (
    load_calibration,
    load_market_blend,
    load_prob_shrink,
    load_starter_adjustment,
    load_expected_runs_challenger,
    load_lineup_adjustment,
    apply_calibration_with_warmup,
    count_current_season_games,
)
from prop_filter import filter_player_gamelog
from recalibration import (
    load_recalibration,
    apply_platt,
    log_prediction,
    log_prediction_rows,
    maybe_auto_refit,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Model⇄market blend weights (fitted offline by backtest.py --mode odds)
# ──────────────────────────────────────────────────────────────────────────────
_MARKET_BLEND_CACHE = {}
_PROB_SHRINK_CACHE = {}
_STARTER_ADJ_CACHE = {}
_LINEUP_ADJ_CACHE = {}
_EXPECTED_RUNS_CACHE = {}

# MLB league baselines used as log5-style denominators for the props matchup
# multiplier. Priors (not fitted); refresh occasionally.
_MLB_LEAGUE = {
    "k_pct": 0.222,
    "ba": 0.243,
    "ops": 0.711,
}


def _log5_rate(player_rate, opponent_rate, league_rate):
    """Combine two binary-event rates relative to their league environment."""
    if player_rate is None or opponent_rate is None or league_rate is None:
        return None
    clamp = lambda value: max(0.001, min(0.999, float(value)))
    player_rate, opponent_rate, league_rate = map(
        clamp, (player_rate, opponent_rate, league_rate))
    odds = ((player_rate / (1.0 - player_rate))
            * (opponent_rate / (1.0 - opponent_rate))
            / (league_rate / (1.0 - league_rate)))
    return odds / (1.0 + odds)


def _mlb_prop_matchup_mult(prop_key, upcoming_is_home, matchup_features, weight,
                           player_context=None):
    """
    Bounded projection multiplier for an MLB player prop based on the
    starter/opponent matchup (Phase 2). 1.0 = no change.

    Pitcher props scale by the OPPOSING lineup's quality vs the starter's hand;
    batter props scale by the OPPOSING starter's quality. When recent batter
    exposure is available, hits and strikeouts use a true log5 rate for the
    projected starter's workload and a neutral rate for the remaining bullpen
    workload. `weight` (0..1) is the calibratable fraction of the raw ratio to
    apply. Bullpen prop rates stay neutral until an as-of history can validate
    them without leakage.
    """
    if not matchup_features or upcoming_is_home is None or not weight:
        return 1.0
    side = "home" if upcoming_is_home else "away"
    opp_side = "away" if upcoming_is_home else "home"
    raw = 1.0

    if prop_key in ("pitcher_strikeouts", "pitcher_outs", "pitcher_earned_runs"):
        sd = matchup_features.get(side) or {}
        opp = sd.get("opp_offense_vs_hand")
        if not opp:
            return 1.0
        if prop_key == "pitcher_strikeouts" and opp.get("k_pct"):
            raw = opp["k_pct"] / _MLB_LEAGUE["k_pct"]           # whiff-prone lineup → more Ks
        elif opp.get("ops"):
            r = opp["ops"] / _MLB_LEAGUE["ops"]
            # Better opposing offense → more earned runs, fewer outs recorded.
            raw = r if prop_key == "pitcher_earned_runs" else (2.0 - r)
    elif prop_key in ("batter_hits", "batter_strikeouts"):
        opp_sd = matchup_features.get(opp_side) or {}
        stp = opp_sd.get("starter")
        if not stp:
            return 1.0
        context = player_context or {}
        base_projection = context.get("base_projection")
        exposure = context.get("expected_exposure")
        if base_projection and exposure:
            batter_rate = base_projection / exposure
            avg_ip = stp.get("avg_ip")
            starter_share = max(0.10, min(0.80, (avg_ip or 5.5) / 9.0))
            if prop_key == "batter_strikeouts":
                league_rate = _MLB_LEAGUE["k_pct"]
                starter_k_pct = stp.get("k_pct")
                if stp.get("bf") is not None and stp["bf"] < 50:
                    starter_k_pct = None
                starter_rate = _log5_rate(
                    batter_rate, starter_k_pct, league_rate)
                bullpen_rate = _log5_rate(
                    batter_rate, league_rate, league_rate)
            else:
                league_rate = _MLB_LEAGUE["ba"]
                starter_rate = _log5_rate(
                    batter_rate, stp.get("xba"), league_rate)
                bullpen_rate = _log5_rate(
                    batter_rate, league_rate, league_rate)
            if starter_rate is not None and bullpen_rate is not None:
                matchup_rate = (starter_share * starter_rate
                                + (1.0 - starter_share) * bullpen_rate)
                raw = exposure * matchup_rate / base_projection
        elif prop_key == "batter_hits" and stp.get("xba"):
            raw = stp["xba"] / _MLB_LEAGUE["ba"]
        elif (prop_key == "batter_strikeouts" and stp.get("k_pct")
              and (stp.get("bf") is None or stp["bf"] >= 50)):
            raw = stp["k_pct"] / _MLB_LEAGUE["k_pct"]

    mult = 1.0 + weight * (raw - 1.0)
    return max(0.7, min(1.4, mult))


def _lineup_exposure_mult(expected_exposure, batting_order, weight,
                          slot_expected_exposure):
    """Blend recent opportunity with a batting-slot expectation."""
    if not expected_exposure or not batting_order or not weight:
        return 1.0
    try:
        slot_exposure = float(slot_expected_exposure[str(int(batting_order))])
        expected_exposure = float(expected_exposure)
        weight = float(weight)
    except (KeyError, TypeError, ValueError):
        return 1.0
    if expected_exposure <= 0 or slot_exposure <= 0:
        return 1.0
    adjusted_exposure = (expected_exposure
                         + weight * (slot_exposure - expected_exposure))
    return max(0.8, min(1.2, adjusted_exposure / expected_exposure))


def _mlb_lineup_exposure_mult(prop_key, player_context):
    """Return the validated batting-order multiplier, failing closed to 1.0."""
    if prop_key != "batter_hits" or not player_context:
        return 1.0
    sport_key = "baseball_mlb"
    if sport_key not in _LINEUP_ADJ_CACHE:
        try:
            _LINEUP_ADJ_CACHE[sport_key] = load_lineup_adjustment(sport_key) or {}
        except Exception:
            _LINEUP_ADJ_CACHE[sport_key] = {}
    cfg = _LINEUP_ADJ_CACHE[sport_key]
    if not cfg or cfg.get("enabled") is False:
        return 1.0
    weights = cfg.get("props") or {}
    slot_exposure = ((cfg.get("slot_expected_exposure") or {})
                     .get(prop_key) or {})
    weight = weights.get(prop_key)
    if not isinstance(weight, (int, float)):
        return 1.0
    return _lineup_exposure_mult(
        player_context.get("expected_exposure"),
        player_context.get("batting_order"),
        weight,
        slot_exposure,
    )


def _starter_adjustment(sport_key, key, prop_key=None):
    """Return the calibrated (or default-prior) starter-adjustment weight for
    `key` in ('moneyline','spreads','run_scale','bullpen','props').
    `props` may be either a legacy shared number or a per-prop mapping.

    Missing or malformed calibration always fails closed to 0.0. An unavailable
    fit must never silently turn an unvalidated matchup prior on in production.
    """
    if not sport_key:
        return 0.0
    if sport_key not in _STARTER_ADJ_CACHE:
        try:
            _STARTER_ADJ_CACHE[sport_key] = load_starter_adjustment(sport_key) or {}
        except Exception:
            _STARTER_ADJ_CACHE[sport_key] = {}
    cfg = _STARTER_ADJ_CACHE[sport_key]
    if not cfg or cfg.get("enabled") is False:
        return 0.0
    val = cfg.get(key)
    if key == "props" and isinstance(val, dict):
        val = val.get(prop_key)
    return val if isinstance(val, (int, float)) else 0.0


def _apply_starter_logit(p, edge, weight):
    """Shift probability p in logit space by weight*edge, bounded to [.02,.98].
    edge>0 favors the team; weight is the (calibratable) logit multiplier."""
    if not weight or edge is None or p <= 0 or p >= 1:
        return p
    lg = math.log(p / (1 - p)) + weight * edge
    return max(0.02, min(0.98, 1.0 / (1.0 + math.exp(-lg))))


def _shrink_factor(sport_key, market):
    """Return the probability-shrink factor s in (0,1] for a team `market`
    ('spreads'/'totals'). p' = 0.5 + s*(p-0.5) corrects model overconfidence.
    Defaults to 1.0 (no shrink) when none is calibrated."""
    if not sport_key:
        return 1.0
    if sport_key not in _PROB_SHRINK_CACHE:
        try:
            _PROB_SHRINK_CACHE[sport_key] = load_prob_shrink(sport_key) or {}
        except Exception:
            _PROB_SHRINK_CACHE[sport_key] = {}
    s = _PROB_SHRINK_CACHE[sport_key].get(market)
    return s if isinstance(s, (int, float)) and 0.0 <= s <= 1.0 else 1.0


def _apply_shrink(p, sport_key, market):
    """Pull probability p toward 0.5 by the calibrated shrink factor."""
    s = _shrink_factor(sport_key, market)
    return 0.5 + s * (p - 0.5) if s != 1.0 else p


def _blend_weight(sport_key, market):
    """
    Return the model weight w in [0,1] for blending the model probability with
    the de-vigged market probability for `market` ('moneyline'/'spreads'/
    'totals'). Defaults to 1.0 (pure model = original behavior) when no
    calibrated weight exists.
    """
    if not sport_key:
        return 1.0
    if sport_key not in _MARKET_BLEND_CACHE:
        try:
            _MARKET_BLEND_CACHE[sport_key] = load_market_blend(sport_key) or {}
        except Exception:
            _MARKET_BLEND_CACHE[sport_key] = {}
    cfg = _MARKET_BLEND_CACHE[sport_key].get(market)
    if not cfg:
        return 1.0
    w = cfg.get("w")
    return w if isinstance(w, (int, float)) and 0.0 <= w <= 1.0 else 1.0


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
# projection. Zero disables exponential decay; None inherits from
# RECENCY_HALF_LIFE.
# MLB chronological 2024 holdout (20-game live window): no decay improved
# batter hits, batter strikeouts, and pitcher earned runs. Pitcher strikeouts
# and outs retain their separately calibrated half_lives from baseball_mlb.json.
# Tuned per backtest on 18 NBA starters × 60 games:
#   - NBA: 7 — Gives the lowest total safe-mode cushion@80% (11.58 vs 11.62
#     at hl=10) at a negligible MAE cost (+0.008, ~0.2%). Prioritized for
#     safe-mode usage. Team-level matchup analysis still uses hl=10.
PLAYER_PROP_HALF_LIFE = {
    "basketball_nba": 7,
    "baseball_mlb": 0,
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

        edge = final_prob - avg_implied

        best_offer = min(ml_odds, key=lambda o: o["implied_prob"])
        best_book_prob = best_offer["implied_prob"]
        best_edge = final_prob - best_book_prob

        result = {
            "type": "moneyline",
            "team": team_name,
            "opponent": away_team if team_name == home_team else home_team,
            "home_away": "HOME" if team_name == home_team else "AWAY",
            "book_implied_prob": round(avg_implied * 100, 2),
            "season_win_pct": round(season_wp * 100, 2),
            "recent_win_pct": round(recent_wp * 100, 2),
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
            "expected_roi_pct": round(_expected_roi(final_prob, best_offer["price"]) * 100, 2),
            "is_value": edge >= threshold,
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

    over_implied = (american_to_implied_prob(over_price)
                    if over_price is not None else 0.50)
    under_implied = (american_to_implied_prob(under_price)
                     if under_price is not None else 0.50)
    over_edge = over_hit_rate - over_implied
    under_hit_rate = 1.0 - over_hit_rate
    under_edge = under_hit_rate - under_implied

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
        "over_expected_roi_pct": (round(_expected_roi(over_hit_rate, over_price) * 100, 2)
                                   if over_price is not None else None),
        "under_expected_roi_pct": (round(_expected_roi(under_hit_rate, under_price) * 100, 2)
                                    if under_price is not None else None),
        "home_avg_scored": round(home_avg_scored, 2),
        "away_avg_scored": round(away_avg_scored, 2),
        "is_over_value": diff > 0 and over_edge >= threshold,
        "is_under_value": diff < 0 and under_edge >= threshold,
        "over_price": over_price,
        "under_price": under_price,
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
                       current_cover, expected_cover):
        implied_prob = (american_to_implied_prob(price)
                        if price is not None else 0.50)
        edge = cover_prob - implied_prob
        candidates.append({
            "type": "spread",
            "team": team_name,
            "opponent": opponent,
            "home_away": "HOME" if is_home else "AWAY",
            "spread": spread,
            "avg_margin": round(team_avg_margin, 2),
            "model_cover_rate": round(model_cover * 100, 2),
            "cover_rate": round(cover_prob * 100, 2),
            "implied_prob": round(implied_prob * 100, 2),
            "games_sampled": games_sampled,
            "edge_pct": round(edge * 100, 2),
            "expected_roi_pct": (round(_expected_roi(cover_prob, price) * 100, 2)
                                  if price is not None else None),
            "is_value": edge >= threshold,
            "pred_game_margin": round(pred_margin, 2),
            "pred_game_std": round(pred_std, 2),
            "price": price,
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
                       current_home_cover, expected_home_cover)
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
                       current_away_cover, expected_away_cover)

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
                               team_schedules=None, matchup_features=None):
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
    prediction_rows = []
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
        if cfg and name in cfg:
            value = cfg[name]
            # In calibration JSON, half_life=null explicitly means equal
            # weighting. Other null knobs continue to mean "use the default."
            if value is not None or name == "half_life":
                return value
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
            over_book = odds_info.get("over_book")
            under_book = odds_info.get("under_book")

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
            plate_appearances = (history.get("plate_appearances")
                                 or [None] * len(values))
            at_bats = history.get("at_bats") or [None] * len(values)
            player_team_id = history.get("team_id")

            # ── Reliability filter ──
            # Drop low-minutes games AND the 1-game-pre + 1-game-post window
            # around any layoff (≥3 missed team games for NBA/MLB, ≥2 for NFL).
            # Also flags the player as "currently fragile" → skip prediction
            # when their last actual game was excluded (still injured /
            # ramping up) or their last game had limited minutes.
            synthetic = [
                {"game_date": gd, "MIN": m, "_value": v, "_opp": o,
                 "_ha": ha, "_pa": pa, "_ab": ab}
                for v, o, ha, m, gd, pa, ab in zip(
                    values, opponents, past_home_aways, minutes, game_dates,
                    plate_appearances, at_bats)
            ]
            team_schedule = None
            if team_schedules and player_team_id:
                team_schedule = team_schedules.get(str(player_team_id))
            # Half-life historically also set the healthy-game streak floor.
            # Removing MLB decay should change weighting only, not make a
            # previously-untrusted player eligible three games sooner. Preserve
            # the former MLB hl=7 threshold (8 games) for no-decay props;
            # calibrated pitcher half-lives continue to set their own floors.
            reliability_min_streak = None
            if sport_key == "baseball_mlb" and not half_life:
                reliability_min_streak = 8
            filt = filter_player_gamelog(
                synthetic, team_schedule, sport_key, half_life=half_life,
                min_streak=reliability_min_streak)

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
            plate_appearances = [g.get("_pa") for g in eligible]
            at_bats = [g.get("_ab") for g in eligible]

            # Resolve the player's upcoming home/away by matching their team_id
            # to the home/away team names of the upcoming game.
            upcoming_is_home = None
            player_team_name = None
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
            # ── MLB starter/opponent matchup multiplier (Phase 2) ──
            # Scales the projection by the log5-style matchup (opposing lineup
            # for pitcher props; opposing starter for batter props). No-op for
            # other sports or when features/weight are absent.
            matchup_mult = 1.0
            lineup_mult = 1.0
            if sport_key == "baseball_mlb":
                player_context = None
                exposures = (plate_appearances if prop_key == "batter_strikeouts"
                             else at_bats if prop_key == "batter_hits" else None)
                if exposures:
                    valid = [(value, weight) for value, weight in zip(exposures, weights)
                             if isinstance(value, (int, float)) and value > 0]
                    if valid:
                        expected_exposure = _weighted_mean(
                            [value for value, _ in valid],
                            [weight for _, weight in valid],
                        )
                        if expected_exposure > 0 and base_proj > 0:
                            player_context = {
                                "base_projection": base_proj,
                                "expected_exposure": expected_exposure,
                            }
                batting_order = history.get("batting_order")
                if batting_order and player_context:
                    player_context["batting_order"] = batting_order
                if matchup_features:
                    matchup_mult = _mlb_prop_matchup_mult(
                        prop_key, upcoming_is_home, matchup_features,
                        _starter_adjustment(sport_key, "props", prop_key),
                        player_context=player_context)
                lineup_mult = _mlb_lineup_exposure_mult(
                    prop_key, player_context)
            combined_mult = output_def_mult * matchup_mult * lineup_mult

            avg_stat = base_proj * combined_mult
            # When the projection is scaled, the over-rate calc shifts the
            # comparison line by the inverse so historical frequencies are
            # interpreted in the projection's adjusted frame.
            effective_line = line / combined_mult if combined_mult else line
            empirical_over = _weighted_rate(values, weights, lambda v: v > effective_line)
            over_rate = empirical_over

            # ── Residual calibration with warmup blending ──
            # If a calibration file exists for this (sport, prop), replace the
            # raw empirical over-rate with a Brier-better calibrated probability.
            # Early-season players (few current-season games) blend with the
            # prior-season warmup distribution.
            calibration_meta = None
            calibration_game_dates = history.get("game_dates") or []
            curr_games = count_current_season_games(calibration_game_dates, sport_key)
            if prop_calib_cfg and prop_calib_cfg.get("method"):
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

            # ── Platt recalibration (self-updating) ──
            # Apply the same final calibration layer before either standard or
            # Safe Mode branches. Safe Mode previously exited before this step,
            # so its displayed confidence did not have parity with standard props.
            raw_over_rate = over_rate
            recal_cfg = recalibration.get(prop_key) if recalibration else None

            def _apply_final_recalibration(probability):
                if not recal_cfg or recal_cfg.get("a") is None:
                    return probability
                adjusted = apply_platt(
                    probability,
                    recal_cfg.get("a"),
                    recal_cfg.get("b"),
                )
                if adjusted is None:
                    return probability
                return max(0.0, min(1.0, adjusted))

            over_rate = _apply_final_recalibration(over_rate)
            recal_meta = None
            if over_rate != raw_over_rate:
                recal_meta = {
                    "a": recal_cfg.get("a"),
                    "b": recal_cfg.get("b"),
                    "n_fit": recal_cfg.get("n_fit"),
                    "raw_prob": round(raw_over_rate * 100, 2),
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
                # Scale std by the same projection factor (output-defense ×
                # MLB matchup) so the spread is in the projection's frame.
                wstd_adj = wstd * (combined_mult if combined_mult else 1.0)
                z = _normal_inv_cdf(safe_target)
                alt_q = proj_mean - z * wstd_adj

                # "Points {N}+" means the player needs actual ≥ N to win.
                # Floor (not ceil) for OVER thresholds: alt_q=8.7 → 8+,
                # because 8 is the largest integer the model expects them
                # to clear with ≥ safe_target probability.
                safe_threshold = max(1, int(_math.floor(alt_q)))

                def _probability_at_threshold(threshold_value):
                    """Return (historical, final model) P(actual >= threshold)."""
                    historical = _weighted_rate(
                        values, weights,
                        lambda v, t=threshold_value: v >= t,
                    )
                    threshold_line = threshold_value - 0.5
                    effective_threshold_line = (
                        threshold_line / combined_mult if combined_mult else threshold_line
                    )
                    empirical_adjusted = _weighted_rate(
                        values, weights,
                        lambda v, t=effective_threshold_line: v > t,
                    )
                    raw_probability = empirical_adjusted
                    if prop_calib_cfg and prop_calib_cfg.get("method"):
                        calibrated = apply_calibration_with_warmup(
                            prop_calib_cfg,
                            avg_stat,
                            threshold_line,
                            curr_games,
                            empirical_over=empirical_adjusted,
                        )
                        if calibrated is not None:
                            raw_probability = max(0.0, min(1.0, calibrated))
                    return historical, _apply_final_recalibration(raw_probability)

                historical_at_safe, p_at_safe = _probability_at_threshold(safe_threshold)

                # Tighten or relax the parametric starting threshold using the
                # same residual + warmup + Platt probability stack that standard
                # props use. This keeps the displayed target tied to the actual
                # production probability rather than a separate empirical-only
                # calculation.
                while p_at_safe < safe_target and safe_threshold > 1:
                    safe_threshold -= 1
                    historical_at_safe, p_at_safe = _probability_at_threshold(safe_threshold)
                # Cap tightening at the largest adjusted outcome in the sample.
                # Production Platt slopes are positive, but this also prevents a
                # malformed future recalibration from making the loop unbounded.
                max_safe_threshold = max(
                    1,
                    int(_math.ceil(max(values) * (combined_mult or 1.0))),
                )
                while safe_threshold < max_safe_threshold:
                    next_t = safe_threshold + 1
                    historical_next, p_next = _probability_at_threshold(next_t)
                    if p_next >= safe_target:
                        safe_threshold = next_t
                        p_at_safe = p_next
                        historical_at_safe = historical_next
                    else:
                        break

                if p_at_safe < safe_target:
                    continue

                # Tight sanity guard: drop when historical hit rate at the
                # suggested threshold is more than 5pp below safe_target.
                # Was 15pp tolerance — measured to admit false positives
                # (NBA assists @ 95% claimed but actually hit 76%; MLB
                # batter_hits @ 85% claimed but actually hit 69%). The
                # 5pp band keeps out-of-sample hit rate within ~5pp of
                # the user-visible safe_target.
                if historical_at_safe < (safe_target - 0.05):
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

                # Our model's final calibrated confidence at the standard line.
                model_hit_at_line = over_rate

                # Gap from book line to our safe threshold. Larger positive
                # gap = book line is below safe floor (bet straight OVER).
                # Negative = user must hunt for an alt OVER line ≤ (safe_threshold − 1).
                line_gap = safe_threshold - line
                bettable_at_standard_line = line < safe_threshold

                # Confidence delta between our safe suggestion and the book line.
                # Positive = our suggestion is safer than the standard line.
                model_delta = p_at_safe - model_hit_at_line

                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "team": player_team_name,
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
                    # Price/edge/EV are intentionally pending until the exact
                    # alternate line is fetched. Comparing this probability to
                    # the standard book-line price would mix different bets.
                    "edge_pct": 0.0,
                    "expected_roi_pct": None,
                    "direction": "OVER",
                    "best_price": over_price,
                    "is_value": False,
                    "value_pending": True,
                    "no_history": False,
                    # ── Safe-mode-specific fields ──
                    "safe_mode": True,
                    "safe_target": safe_target,
                    "safe_threshold": safe_threshold,        # display as "{N}+"
                    "safe_alt_q": round(alt_q, 2),           # raw quantile (continuous)
                    "model_hit_at_safe": round(p_at_safe * 100, 2),     # prob at suggested
                    "historical_hit_at_safe": round(historical_at_safe * 100, 2),
                    "model_hit_at_line": round(model_hit_at_line * 100, 2),  # prob at book line
                    "model_delta": round(model_delta * 100, 2),         # safe − book line
                    "line_gap": round(line_gap, 2),
                    "bettable_at_standard_line": bettable_at_standard_line,
                    "calibration": calibration_meta,
                    "recalibration": recal_meta,
                    "_values": list(values),
                    "_weights": list(weights),
                })
                continue

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

            expected_roi = _expected_roi(
                over_rate if direction == "OVER" else under_rate,
                best_price,
            )

            # Log the published probability so future refits learn from it.
            # We log the *raw* (pre-Platt) probability — that's what Platt
            # was fit against and what subsequent refits should map.
            if log_game_date and sport_key:
                prediction_row = log_prediction(
                    sport_key=sport_key,
                    event_id=prop_data.get("game_id"),
                    commence_time=commence_iso,
                    prop_key=prop_key,
                    player=player_name,
                    game_date=log_game_date,
                    line=line,
                    raw_prob=raw_over_rate,
                    final_prob=over_rate,
                    projected=avg_stat,
                    direction=direction,
                    price=best_price,
                    book=over_book if direction == "OVER" else under_book,
                    is_value=edge >= threshold,
                    write=False,
                )
                if prediction_row:
                    prediction_rows.append(prediction_row)

            candidates.append({
                "type": "player_prop",
                "matchup": matchup,
                "player": player_name,
                "team": player_team_name,
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
                "expected_roi_pct": (round(expected_roi * 100, 2)
                                      if expected_roi is not None else None),
                "is_value": edge >= threshold,
                "no_history": False,
                "calibration": calibration_meta,
                "recalibration": recal_meta,
                "_values": list(values),
                "_weights": list(weights),
            })

    log_prediction_rows(prediction_rows)
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


def _normalize_legs(all_ml, all_spreads, all_totals, all_props):
    """
    Convert all analysis results into a uniform leg format for parlay building.
    Only include legs that passed their analyzer's value recommendation filter.
    
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
        if not c.get("is_value") or c["edge_pct"] <= 0:
            continue
        game_key = f"{c['opponent']} @ {c['team']}" if c["home_away"] == "HOME" else f"{c['team']} @ {c['opponent']}"
        legs.append({
            "game_key": game_key,
            "team": c["team"],
            "bet_type": "moneyline",
            "label": f"{c['team']} ML ({c['home_away']})",
            "player": None,
            "prop_key": None,
            "edge_pct": c.get("best_edge_pct", c["edge_pct"]),
            "odds_price": c.get("best_price"),
            "hist_prob": c.get("blended_prob", c["hist_prob"]) / 100.0,
            "implied_prob": c.get("best_book_implied_prob", c["book_implied_prob"]) / 100.0,
        })
    
    for c in all_spreads:
        # Safe mode rewrites is_value to enforce its confidence threshold.
        if (not c.get("is_value") or c["edge_pct"] <= 0
                or c.get("games_sampled", 0) < 5):
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
            "implied_prob": c.get("implied_prob", 50.0) / 100.0,
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
                "edge_pct": c.get("over_edge_pct", c["over_hit_rate"] - 50.0),
                "odds_price": c.get("over_price"),
                "hist_prob": c["over_hit_rate"] / 100.0,
                "implied_prob": c.get("over_implied", 50.0) / 100.0,
            })
        if c.get("is_under_value"):
            legs.append({
                "game_key": c["matchup"],
                "team": None,
                "bet_type": "total_under",
                "label": f"UNDER {c['line']} ({c['matchup']})",
                "player": None,
                "prop_key": None,
                "edge_pct": c.get("under_edge_pct", (100.0 - c["over_hit_rate"]) - 50.0),
                "odds_price": c.get("under_price"),
                "hist_prob": (100.0 - c["over_hit_rate"]) / 100.0,
                "implied_prob": c.get("under_implied", 50.0) / 100.0,
            })
    
    for c in all_props:
        if (not c.get("is_value") or c.get("no_history")
                or c["edge_pct"] <= 0 or c.get("games_sampled", 0) < 5):
            continue
        direction = c.get("direction", "OVER")
        bt = f"player_prop_{direction.lower()}"
        price = c.get("best_price", c.get("over_price") if direction == "OVER" else c.get("under_price"))

        if c.get("safe_mode"):
            # Safe-mode legs: bet is "{prop} {N}+" and the hist prob is the
            # model probability AT our safe threshold (not the book line).
            label = f"{c['player']} {c['prop_label']} {c['safe_threshold']}+"
            hp = (c.get("model_hit_at_safe", 0.0) or 0.0) / 100.0
            ip = (c.get("safe_alt_implied") or 0.0) / 100.0
            price = c.get("safe_alt_price")
            if price is None or ip <= 0:
                continue
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
            "team": c.get("team"),
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
                leg["expected_roi_pct"] = c.get("expected_roi_pct")
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
        if ("player_prop_over" in ta and "player_prop_over" in tb
                and leg_a.get("team") and leg_a.get("team") == leg_b.get("team")):
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
        if (ta == "moneyline" and "player_prop_over" in tb
                and leg_a.get("team") == leg_b.get("team")):
            return 0.15
        if (tb == "moneyline" and "player_prop_over" in ta
                and leg_b.get("team") == leg_a.get("team")):
            return 0.15

    elif sport_key == "americanfootball_nfl":
        if (ta == "moneyline" and "player_prop_over" in tb and
                leg_a.get("team") == leg_b.get("team") and
                leg_b.get("prop_key") == "player_pass_yds"):
            return 0.30
        if (tb == "moneyline" and "player_prop_over" in ta and
                leg_b.get("team") == leg_a.get("team") and
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
        if ("player_prop_over" in ta and "player_prop_over" in tb
                and leg_a.get("team") and leg_a.get("team") == leg_b.get("team")):
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
                leg_a.get("team") == leg_b.get("team") and
                leg_b.get("prop_key") == "batter_hits"):
            return 0.25
        if (tb == "moneyline" and "player_prop_over" in ta and
                leg_b.get("team") == leg_a.get("team") and
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
    Cheap enumeration score derived from the same correlation used by the
    Gaussian copula. Keeping one rule source prevents candidate selection and
    final probability ranking from assigning opposite signs to the same pair.
    """
    if leg_a["game_key"] != leg_b["game_key"]:
        return 0.5
    return 5.0 * _pair_correlation(leg_a, leg_b, sport_key)


def _same_team_prop_count(legs):
    """Count player-prop overs sharing an identified team in the same game."""
    team_prop_counts = {}
    for leg in legs:
        if "player_prop_over" in leg["bet_type"] and leg.get("team"):
            key = (leg["game_key"], leg["team"])
            team_prop_counts[key] = team_prop_counts.get(key, 0) + 1
    return max(team_prop_counts.values()) if team_prop_counts else 0


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
        # Every alt leg has an exact fetched price, so maximize estimated return
        # rather than payout alone. A long price is not value unless the modeled
        # probability is high enough to compensate for it.
        payout_product = 1.0
        for leg in legs:
            price = leg.get("odds_price")
            payout_product *= american_to_decimal(price) if price is not None else 1.91
        expected_roi = combined_hist * payout_product - 1.0
        return (expected_roi * 100) + correlation_score + usage_penalty
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

    # If the underlying analysis was run in safe mode, use a dedicated value
    # ranker. Safe legs are price-verified before they reach this function, so
    # it can maximize expected return at the exact alternate-line prices.
    has_safe_legs = any(leg.get("safe_mode") for leg in legs)
    effective_mode = mode
    if mode == "value" and has_safe_legs:
        effective_mode = "safe_value"

    # Sort and take top candidates to limit combinatorics
    if effective_mode == "safe":
        legs.sort(key=lambda x: x["hist_prob"], reverse=True)
    elif effective_mode == "safe_value":
        # Prefer legs with the strongest single-bet expected return at their
        # exact fetched alt-line price.
        legs.sort(
            key=lambda x: (
                x["hist_prob"] * american_to_decimal(x["odds_price"]) - 1.0
                if x.get("odds_price") is not None else float("-inf")
            ),
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
                # Re-rank by correlation-adjusted expected return at the exact
                # fetched price for every leg.
                payout_product = 1.0
                for leg in combo_list:
                    price = leg.get("odds_price")
                    payout_product *= american_to_decimal(price) if price is not None else 1.91
                score = (joint * payout_product - 1.0) * 100
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

            # Safe-mode legs now carry implied probability from the exact alt
            # price, so these comparisons are at the same line for every leg.
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
            expected_roi_pct = round((best_joint * decimal_product - 1.0) * 100, 2)

            # A Value Parlay must still be value after the copula correlation
            # adjustment and the actual leg prices are combined.
            if effective_mode in ("value", "safe_value") and expected_roi_pct <= 0:
                continue

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
                "expected_roi_pct": expected_roi_pct,
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
