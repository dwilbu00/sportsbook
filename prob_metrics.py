"""Richer probability + betting metrics (mined from the baseball-predictions clone,
idea #11). Our creed is 'Brier != ROI' and 'beat the market, don't be overconfident'
— but we mostly report raw Brier + ROI. These measure the things we actually care
about, all as pure functions over (prob, outcome) obs or a returns series:

  brier_skill_score  — Brier vs a REFERENCE (the market); >0 = we beat the market.
  ece                — expected calibration error (binned |confidence - accuracy|).
  calibration_slope  — logistic slope of outcome ~ logit(prob); 1 = calibrated,
                       <1 = OVERCONFIDENT (our named disease), >1 = underconfident.
  equity_stats       — Sharpe + max drawdown of a flat-stake P&L series.
  tier_monotonicity  — does realized ROI rise with model confidence (it should)?

No numpy; pure Python (matches the rest of deploy). See [[baseballpredictions-mining]].
"""
from __future__ import annotations

import math


def brier(probs, outcomes):
    """Mean Brier score of ``probs`` (P of the modeled side) vs 0/1 ``outcomes``."""
    n = len(probs)
    if n == 0:
        return None
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / n


def brier_skill_score(probs, outcomes, ref_probs):
    """1 - Brier(model)/Brier(reference). >0 = model beats the reference (market)
    on Brier; 0 = tie; <0 = worse. This is our 'do we actually beat the close'
    metric, not raw Brier which is dominated by the sport's variance floor."""
    b_m = brier(probs, outcomes)
    b_r = brier(ref_probs, outcomes)
    if b_m is None or not b_r:
        return None
    return 1.0 - (b_m / b_r)


def ece(probs, outcomes, bins=10):
    """Expected Calibration Error: sample-weighted mean over ``bins`` confidence
    buckets of |mean_predicted - mean_actual|. 0 = perfectly calibrated."""
    n = len(probs)
    if n == 0:
        return None
    buckets = [[] for _ in range(bins)]
    for p, y in zip(probs, outcomes):
        idx = min(int(p * bins), bins - 1)
        buckets[idx].append((p, y))
    total = 0.0
    for b in buckets:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        total += (len(b) / n) * abs(conf - acc)
    return total


def _logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def calibration_slope(probs, outcomes, iters=50):
    """Fit outcome ~ sigmoid(intercept + slope*logit(prob)) by IRLS/Newton and
    return {slope, intercept}. slope==1 & intercept==0 => calibrated; slope<1 =>
    OVERCONFIDENT (predictions too extreme for the realized rate); slope>1 =>
    underconfident. None if degenerate/too thin."""
    n = len(probs)
    if n < 10:
        return None
    xs = [_logit(p) for p in probs]
    if len(set(outcomes)) < 2:
        return None
    b0, b1 = 0.0, 1.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, outcomes):
            z = b0 + b1 * x
            z = max(-700.0, min(700.0, z))
            mu = 1.0 / (1.0 + math.exp(-z))
            w = mu * (1.0 - mu)
            r = y - mu
            g0 += r
            g1 += r * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (-h01 * g0 + h00 * g1) / det
        b0 += d0
        b1 += d1
        if abs(d0) + abs(d1) < 1e-9:
            break
    return {"slope": b1, "intercept": b0}


def equity_stats(returns):
    """Sharpe (mean/sd) + max drawdown of a FLAT-stake per-bet return series
    (profit-per-unit, e.g. +0.91 / -1.0). Drawdown is on the cumulative P&L in
    units. None if too thin."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(var)
    sharpe = (mean / sd) if sd > 0 else 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {"n": n, "mean": mean, "sharpe": sharpe,
            "final_units": equity, "max_drawdown_units": max_dd}


def tier_monotonicity(tier_rois):
    """Given ROI per ascending confidence tier (list, low->high confidence, None
    allowed for empty tiers), return {monotonic, spearman_like, top_minus_bottom}.
    A well-behaved model earns MORE ROI at higher confidence; a negative/zero
    trend is a red flag the confidence isn't real."""
    vals = [(i, r) for i, r in enumerate(tier_rois) if r is not None]
    if len(vals) < 2:
        return None
    # fraction of adjacent-pair steps that go up (crude monotonicity)
    ups = sum(1 for (a, b) in zip(vals, vals[1:]) if b[1] >= a[1])
    monotonic = ups == (len(vals) - 1)
    # rank correlation of tier-index vs ROI (concordant - discordant)/pairs
    conc = disc = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            d = (vals[j][1] - vals[i][1])
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    pairs = conc + disc
    return {
        "monotonic": monotonic,
        "rank_corr": ((conc - disc) / pairs) if pairs else 0.0,
        "top_minus_bottom": vals[-1][1] - vals[0][1],
    }
