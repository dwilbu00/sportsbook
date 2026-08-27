"""R2 backtest grading + aggregation (pure).

Given a bet's realized outcome, grade it and compute profit; then aggregate a set
of graded bets into ROI / hit-rate / significance, sliced the way an R2 verdict
needs it (by EV bucket, by same-line vs projected, and — critically — per SEASON,
because our discipline is that an edge must replicate each season out-of-sample,
not merely pool positive). Pure: no warehouse, no I/O. The backtest data layer
feeds it graded rows; this decides whether the edge is real.
"""
import math
from collections import namedtuple

from odds_client import american_to_decimal

Summary = namedtuple(
    "Summary", "n decided wins pushes roi hit_rate t_stat mean_ev total_profit")


def grade_over_under(actual, line, side):
    """'win' | 'loss' | 'push' for an OVER/UNDER bet given the realized count.

    A half-integer line never pushes; an integer line pushes when actual == line
    (matching the model's OVER <=> X > line convention). None on missing inputs."""
    if actual is None or line is None or not side:
        return None
    try:
        actual = float(actual)
        line = float(line)
    except (TypeError, ValueError):
        return None
    s = side.upper()
    if abs(actual - line) < 1e-9:
        return "push"
    over_wins = actual > line
    if s == "OVER":
        return "win" if over_wins else "loss"
    if s == "UNDER":
        return "loss" if over_wins else "win"
    return None


def grade_moneyline(bet_team, winner):
    """'win' | 'loss' for a moneyline bet. None on missing inputs (MLB has no
    regulation ties, so no push)."""
    if not bet_team or not winner:
        return None
    return "win" if bet_team == winner else "loss"


def profit(american_price, result):
    """Profit per $1 staked: win -> decimal-1, loss -> -1, push -> 0. None if the
    result isn't a graded outcome or the price is unusable."""
    if result == "push":
        return 0.0
    if result not in ("win", "loss"):
        return None
    try:
        dec = american_to_decimal(int(american_price))
    except (TypeError, ValueError):
        return None
    return (dec - 1.0) if result == "win" else -1.0


def summarize(rows):
    """Aggregate graded bets into a Summary.

    ``rows`` = iterable of dicts with 'result' ('win'|'loss'|'push'), 'profit'
    (per-$1, push=0.0) and optional 'ev' (the modeled edge, for mean_ev). The
    t-stat is on mean profit vs 0 (H0: no edge) using the per-bet profit variance —
    a bet-level significance, so a big-ROI/tiny-n bucket doesn't read as real."""
    rows = [r for r in rows if r.get("profit") is not None]
    n = len(rows)
    if n == 0:
        return Summary(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    profits = [r["profit"] for r in rows]
    wins = sum(1 for r in rows if r.get("result") == "win")
    pushes = sum(1 for r in rows if r.get("result") == "push")
    decided = n - pushes
    total = sum(profits)
    roi = total / n
    hit = wins / decided if decided else 0.0
    if n > 1:
        var = sum((p - roi) ** 2 for p in profits) / (n - 1)
        se = math.sqrt(var / n)
        t = roi / se if se > 0 else 0.0
    else:
        t = 0.0
    mean_ev = sum(r.get("ev", 0.0) or 0.0 for r in rows) / n
    return Summary(n, decided, wins, pushes, roi, hit, t, mean_ev, total)


def by_key(rows, keyfn):
    """Group ``rows`` and summarize each group -> {key: Summary}, sorted by key.
    ``keyfn(row)`` yields the slice key (season, EV bucket, market, ...)."""
    groups = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


# EV buckets for slicing the R2 signal (edge magnitude at DK's price).
_EV_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20]


def ev_bucket(ev):
    """Label an EV into a coarse bucket for the ROI-by-edge table."""
    if ev is None:
        return "n/a"
    if ev < 0:
        return "<0"
    lo = 0.0
    for hi in _EV_EDGES[1:]:
        if ev < hi:
            return f"{lo:.0%}-{hi:.0%}"
        lo = hi
    return f">={_EV_EDGES[-1]:.0%}"


def replicates_per_season(rows, seasonfn, min_n=30):
    """True iff EVERY season with >= ``min_n`` bets has positive ROI — the
    per-season OOS replication gate (an edge that only pools positive fails).
    Returns (ok, {season: Summary}). Seasons under min_n are ignored for the gate
    but still reported."""
    per = by_key(rows, seasonfn)
    judged = {s: sm for s, sm in per.items() if sm.n >= min_n}
    ok = bool(judged) and all(sm.roi > 0 for sm in judged.values())
    return ok, per
