"""Cross-market COHERENCE: does DraftKings contradict ITSELF across its own team
markets? If so, one of its prices is provably wrong — and we can bet the outlier
WITHOUT out-predicting the market (the thing we've shown we can't do).

The premise, now on solid ground: the sharpness test proved DK is efficient on
moneyline AND totals (Brier ties Pinnacle). So DK's ML + total are accurate. If we
translate those two known-good prices into an IMPLIED run-line — via a joint
run-distribution — and DK's ACTUAL run-line disagrees, then DK's run-line is
mispriced (not us being smarter than the market). We bet DK's run-line where it's
+EV against the ML+total-implied fair.

Method (all from one snapshot's DK prices):
  1. devig ML  -> P_home_win ;  devig total -> P(total > line).
  2. solve independent per-team run means (mean_h, mean_a) so the joint reproduces
     BOTH (P_home_win and P_total_over) under the SAME NegBin/Poisson score model the
     rest of the system uses (mlb_starters._run_pmf).
  3. read the model's implied run-line cover P(home + spread > away) at DK's spread.
  4. EV of DK's run-line at DK's price, using that implied cover as truth.

Pure module (reuses mlb_starters._run_pmf; no warehouse/IO). The backtest feeds DK
prices + final scores and grades whether betting the incoherent side profits.
"""
from mlb_starters import _run_pmf

_MAX_RUNS = 25


def _pmfs(mean_h, mean_a, dispersion, max_runs):
    return (_run_pmf(mean_h, dispersion, max_runs),
            _run_pmf(mean_a, dispersion, max_runs))


def p_home_win(mean_h, mean_a, dispersion=0.0, max_runs=_MAX_RUNS):
    """P(home wins) = P(H>A) with the tie mass redistributed (MLB has no ties —
    extra innings resolve them), matching how the moneyline actually settles."""
    hp, ap = _pmfs(mean_h, mean_a, dispersion, max_runs)
    win = sum(h * a for hi, h in enumerate(hp) for ai, a in enumerate(ap) if hi > ai)
    lose = sum(h * a for hi, h in enumerate(hp) for ai, a in enumerate(ap) if hi < ai)
    tot = win + lose
    return win / tot if tot > 0 else 0.5


def p_total_over(mean_h, mean_a, total_line, dispersion=0.0, max_runs=_MAX_RUNS):
    """P(home + away runs > total_line)."""
    hp, ap = _pmfs(mean_h, mean_a, dispersion, max_runs)
    return sum(h * a for hi, h in enumerate(hp) for ai, a in enumerate(ap)
               if hi + ai > total_line)


def p_home_cover(mean_h, mean_a, home_spread, dispersion=0.0, max_runs=_MAX_RUNS):
    """P(home + home_spread > away) — the run-line cover for the home side at
    ``home_spread`` (e.g. -1.5 when home is the run-line favorite)."""
    hp, ap = _pmfs(mean_h, mean_a, dispersion, max_runs)
    return sum(h * a for hi, h in enumerate(hp) for ai, a in enumerate(ap)
               if hi + home_spread > ai)


def solve_run_means(ml_home_fair, total_line, total_over_fair, dispersion=0.0,
                    max_runs=_MAX_RUNS, outer_iters=12):
    """Solve (mean_h, mean_a) so the joint reproduces BOTH the ML win prob and the
    total over prob. Coordinate bisection: given a total mean T, bisect the home
    share f to hit P_home_win; given f, bisect T to hit P_total_over; repeat (the
    two constraints are near-separable — f controls who wins, T controls the total —
    so it converges in a handful of rounds). Returns (mean_h, mean_a) or None."""
    if not (0.0 < ml_home_fair < 1.0) or not (0.0 < total_over_fair < 1.0):
        return None
    if not total_line or total_line <= 0:
        return None
    T = float(total_line)
    f = 0.5
    for _ in range(outer_iters):
        flo, fhi = 0.02, 0.98
        for _ in range(26):                       # P_home_win increases in f
            fm = 0.5 * (flo + fhi)
            if p_home_win(fm * T, (1 - fm) * T, dispersion, max_runs) < ml_home_fair:
                flo = fm
            else:
                fhi = fm
        f = 0.5 * (flo + fhi)
        tlo, thi = 3.0, 22.0
        for _ in range(26):                       # P_total_over increases in T
            tm = 0.5 * (tlo + thi)
            if p_total_over(f * tm, (1 - f) * tm, total_line, dispersion, max_runs) < total_over_fair:
                tlo = tm
            else:
                thi = tm
        T = 0.5 * (tlo + thi)
    return f * T, (1 - f) * T


def implied_home_cover(ml_home_fair, total_line, total_over_fair, home_spread,
                       dispersion=0.0, max_runs=_MAX_RUNS):
    """The run-line home-cover probability IMPLIED by DK's ML + total (the coherent
    fair for the run-line). None if the means can't be solved."""
    means = solve_run_means(ml_home_fair, total_line, total_over_fair,
                            dispersion, max_runs)
    if means is None:
        return None
    return p_home_cover(means[0], means[1], home_spread, dispersion, max_runs)
