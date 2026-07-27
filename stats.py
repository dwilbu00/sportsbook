"""Pure-Python statistics helpers (no numpy/scipy dependency).

Shared low-level math used across the pricing engine (analysis / props / parlay)
and the offline backtest / calibration harnesses. This is a leaf module — it
imports only the standard library, so anything may import it without risking a
circular import. ``analysis`` re-exports these names for backward compatibility.
"""

import math

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


def hits_at_least(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p) — pure-stdlib binomial survival.

    Used by the distributional batter_hits model: with n = expected at-bats and
    p = per-at-bat hit probability, a line of (k - 0.5) hits maps to
    hits_at_least(k, n, p); the line-0.5 case P(>=1 hit) = 1 - (1 - p)^n.

    Sums the lower tail via the ratio recurrence
    C(n,i) p^i q^(n-i) = prev * (n - i + 1) / i * (p / q), so no factorials or
    lgamma are needed and it stays exact for the small n (~3-6 at-bats) here.
    ``n`` is coerced to a non-negative integer (the caller rounds expected AB)."""
    k = int(k)
    if k <= 0:
        return 1.0
    n = int(round(n))
    if n <= 0 or p <= 0.0:
        return 0.0
    if k > n:                # can't get k successes in n trials — even if p == 1
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p
    term = q ** n          # i = 0 term: (1 - p)^n = P(X = 0)
    cdf = term
    ratio = p / q
    for i in range(1, k):  # accumulate P(X <= k-1)
        term *= (n - i + 1) / i * ratio
        cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


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
