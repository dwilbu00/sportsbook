"""Backtest-overfit brakes (mined from the baseball-predictions clone, idea #2).

Two statistical guards for our ROI/gate/pythag/shrink SWEEPS, which pick the best
of many configs on one window and are therefore prone to selection optimism (the
#1 named cause of our forward > backtest gap):

  1. multiple-testing haircut / deflated ROI — when you keep the best of N swept
     configs, its apparent edge is inflated. Compare the winner's t-statistic to
     the EXPECTED MAX t of N pure-noise trials; the winner is only credible if it
     clears that bar. (A t-framed approximation of the Deflated Sharpe Ratio,
     Bailey & Lopez de Prado.)

  2. PBO — Probability of Backtest Overfit via Combinatorially-Symmetric Cross-
     Validation (CSCV, Bailey et al.): across all balanced train/test block
     splits, how often does the in-sample-best config land below the OOS median?
     High PBO => the selection procedure itself overfits, independent of any one
     config.

Pure functions, no I/O. See [[baseballpredictions-mining]], [[team-market-audit]].
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

_ND = NormalDist()
_EULER = 0.5772156649015329


def expected_max_z(n_trials):
    """Expected maximum of ``n_trials`` i.i.d. N(0,1) draws — the t-stat bar a
    best-of-N winner must clear to be more than noise. Uses the standard
    extreme-value approximation E[max] ~= (1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(N*e)).
    Returns 0.0 for N<=1 (no multiple-testing penalty)."""
    n = int(n_trials)
    if n <= 1:
        return 0.0
    inv = _ND.inv_cdf
    return ((1.0 - _EULER) * inv(1.0 - 1.0 / n)
            + _EULER * inv(1.0 - 1.0 / (n * math.e)))


def deflated_roi(returns, n_trials):
    """Multiple-testing haircut for the BEST of ``n_trials`` swept configs.

    ``returns`` = per-bet profit-per-unit at the winning config (e.g. +0.91 on a
    -110 win, -1.0 on a loss). Returns a dict:
      mean            realized flat-1u ROI
      t_stat          mean / (sd/sqrt(n)) — the naive 'edge is real' t-statistic
      noise_bar       expected_max_z(n_trials) — the t a best-of-N winner must beat
      deflated_prob   ~P(true edge > 0 | best-of-N) = Phi(t_stat - noise_bar)
      credible        deflated_prob >= 0.95 (clears the haircut at ~95%)
    Returns None if fewer than 2 bets. deflated_prob well below the naive one-test
    p-value is the warning: the winner is largely a product of the search."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    sd = math.sqrt(var)
    t_stat = (mean / (sd / math.sqrt(n))) if sd > 0 else 0.0
    noise_bar = expected_max_z(n_trials)
    deflated_prob = _ND.cdf(t_stat - noise_bar)
    return {
        "n": n, "mean": mean, "t_stat": t_stat, "noise_bar": noise_bar,
        "n_trials": int(n_trials), "deflated_prob": deflated_prob,
        "credible": deflated_prob >= 0.95,
    }


def pbo_cscv(perf_matrix):
    """Probability of Backtest Overfit via CSCV (Bailey et al.).

    ``perf_matrix``: rows = time BLOCKS, cols = configs; cell = that config's
    performance in that block (ROI, P/L, or Sharpe — any 'higher is better'
    metric). For every balanced split of the S blocks into IS/OOS halves, pick the
    IS-best config and record its OOS rank; PBO = fraction of splits where the
    IS-best config lands BELOW the OOS median (logit < 0). High PBO (>~0.5) => the
    selection procedure overfits regardless of which config it picked. Needs an
    even S >= 4 and >= 2 configs. Returns dict or None if too small."""
    S = len(perf_matrix)
    if S < 4 or S % 2 != 0:
        return None
    ncfg = len(perf_matrix[0])
    if ncfg < 2 or any(len(row) != ncfg for row in perf_matrix):
        return None
    blocks = list(range(S))
    logits = []
    for train in itertools.combinations(blocks, S // 2):
        train_set = set(train)
        test = [b for b in blocks if b not in train_set]
        is_perf = [sum(perf_matrix[b][c] for b in train) for c in range(ncfg)]
        oos_perf = [sum(perf_matrix[b][c] for b in test) for c in range(ncfg)]
        best = max(range(ncfg), key=lambda c: is_perf[c])
        # fractional OOS rank of the IS-best config (share of configs it beats)
        worse = sum(1 for c in range(ncfg) if oos_perf[best] > oos_perf[c])
        rank = worse / (ncfg - 1) if ncfg > 1 else 0.5
        rank = min(max(rank, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(rank / (1.0 - rank)))
    pbo = sum(1 for lg in logits if lg < 0) / len(logits)
    return {"pbo": pbo, "n_splits": len(logits), "n_configs": ncfg, "n_blocks": S}
