"""R2 sharp-reference pricing: turn a sharp book's (Pinnacle's) posted prices into
a no-vig "fair" probability for ANY line — including a line the sharp book did not
post but DraftKings did.

WHY (the R2 thesis): the edge we're hunting is market STRUCTURE, not prediction —
DK's price/line being stale relative to the sharp consensus. To measure that we
need the sharp book's fair probability *at DK's exact line*. Two cases:

  1. SAME LINE  — DK and Pinnacle both post, e.g., batter_hits 0.5. Just devig
     Pinnacle's two-sided price (Clarke power method) -> fair P(Over). No model.

  2. DIFFERENT LINE — Pinnacle posts hits 1.5, DK posts 0.5 (Doug's bullseye).
     The probabilities aren't comparable directly. So we back out the count
     distribution implied by Pinnacle's fair prob(s) and read off its survival at
     DK's line. We invert the SAME NegBin/Poisson survival (stats.negbin_at_least)
     the model itself prices props with, so the projection is self-consistent with
     the rest of the system rather than a second, divergent count model.

This module is PURE (no warehouse, no I/O): the backtest layer feeds it offers and
consumes the fair probs. Continuity convention matches the model: a line L means
OVER <=> X >= int(L)+1 (a half-integer line has no push; an integer line excludes
the push mass P(X=L)).
"""
from collections import namedtuple

from odds_client import american_to_implied_prob, devig_two_way
from stats import negbin_at_least

# Result of pricing a sharp fair probability at a target line.
#   prob       : fair P(Over) at the target line (0..1), or None if unpriceable
#   projected  : True if we had to project across lines (target line not posted)
#   distance   : |target_point - nearest posted point| (0.0 when same-line)
#   mean       : the inferred count mean (None for same-line / moneyline)
#   dispersion : the NegBin dispersion used (0.0 = Poisson)
#   n_lines    : how many two-sided sharp lines informed the fit
SharpFair = namedtuple("SharpFair",
                       "prob projected distance mean dispersion n_lines")

# Dispersion grid for the multi-line self-consistency fit (0.0 = Poisson). Kept
# modest: props counts are low, and the fit only needs to pick the shape under
# which the posted lines agree on one mean.
_DISPERSION_GRID = [round(0.05 * i, 3) for i in range(0, 31)]   # 0.00 .. 1.50


def _implied(price):
    """American price -> vigged implied prob, or None on a missing/invalid price."""
    if price is None or isinstance(price, bool):
        return None
    try:
        return american_to_implied_prob(int(price))
    except (TypeError, ValueError):
        return None


def fair_two_way(over_price, under_price):
    """No-vig fair (P_over, P_under) from a two-sided market's American prices, via
    Clarke's power method. For moneyline pass (home_price, away_price). Returns
    (None, None) unless BOTH sides are present (the hold can't be removed one-sided)."""
    io, iu = _implied(over_price), _implied(under_price)
    if io is None or iu is None:
        return None, None
    return devig_two_way(io, iu)


def _line_threshold(point):
    """OVER <=> X >= k for a line ``point`` (int(point)+1), the model's convention."""
    return int(point) + 1


def _survival_points(offers):
    """[(point, k, fair_over)] for each offer that has BOTH sides, deviged.

    ``offers`` = iterable of dicts with ``point``, ``over_price``, ``under_price``
    (one book's two-sided lines on one prop/total). Lines missing a side or a point
    are skipped. fair_over is P(X >= k) under the sharp's no-vig price."""
    pts = []
    for o in offers or []:
        point = o.get("point")
        if point is None:
            continue
        fair_over, _ = fair_two_way(o.get("over_price"), o.get("under_price"))
        if fair_over is None:
            continue
        pts.append((float(point), _line_threshold(point), fair_over))
    return pts


def solve_count_mean(k, survival, dispersion=0.0, tol=1e-10, max_iter=100):
    """Invert the survival function: find the count mean m with
    ``negbin_at_least(k, m, dispersion) == survival``.

    negbin_at_least is continuous and strictly increasing in the mean for a fixed
    threshold k>=1, so a bracket-and-bisect is exact and monotone-safe. Returns 0.0
    for a non-positive target, and an upper cap for a target >= 1 (an unbounded mean
    would be needed). ``k`` is the integer OVER threshold from _line_threshold."""
    k = int(k)
    if k <= 0:
        return 0.0
    if survival is None or survival <= 0.0:
        return 0.0
    if survival >= 1.0:
        return float("inf")
    lo, hi = 0.0, 1.0
    # Grow the upper bracket until it exceeds the target survival (or we cap out).
    for _ in range(100):
        if negbin_at_least(k, hi, dispersion) >= survival:
            break
        lo = hi
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = negbin_at_least(k, mid, dispersion)
        if abs(s - survival) < tol:
            return mid
        if s < survival:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_count_shape(points, default_dispersion=0.0):
    """Infer (mean, dispersion) from deviged sharp survival points.

    ``points`` = [(point, k, fair_over)] (from _survival_points).
      * 0 points  -> None (nothing to fit).
      * 1 point   -> solve the mean at ``default_dispersion`` (can't identify shape
                     from a single line; the caller's default governs projection
                     curvature — 0.0 = Poisson).
      * 2+ points -> pick the dispersion under which the per-line inferred means
                     agree best (minimize their spread), then return their average.
                     The right shape is the one that reconciles all posted lines to
                     ONE distribution; ties prefer the smaller (more parsimonious)
                     dispersion. This is what lets two sharp lines (e.g. 0.5 and 1.5)
                     jointly pin the curve we then read at DK's line."""
    if not points:
        return None
    if len(points) == 1:
        _pt, k, s = points[0]
        return solve_count_mean(k, s, default_dispersion), default_dispersion

    best = None   # (spread, dispersion, mean)
    for disp in _DISPERSION_GRID:
        means = [solve_count_mean(k, s, disp) for _pt, k, s in points]
        means = [m for m in means if m != float("inf")]
        if len(means) < 2:
            continue
        mbar = sum(means) / len(means)
        # Coefficient-of-variation spread: scale-free so a big-mean total and a
        # small-mean prop are judged on the same footing.
        spread = (sum((m - mbar) ** 2 for m in means) / len(means)) ** 0.5
        spread = spread / mbar if mbar > 0 else spread
        if best is None or spread < best[0] - 1e-9:
            best = (spread, disp, mbar)
    if best is None:
        _pt, k, s = points[0]
        return solve_count_mean(k, s, default_dispersion), default_dispersion
    return best[2], best[1]


def fair_prob_at_line(offers, target_point, default_dispersion=0.0):
    """Sharp fair P(Over) at ``target_point`` from a book's posted ``offers``.

    Same-line shortcut: if the sharp posted the exact target line, return its
    deviged fair prob directly (no model, distance 0). Otherwise back out the count
    shape from the posted lines and evaluate its survival at the target — the
    cross-line projection. Returns a SharpFair; prob is None if nothing is posted."""
    pts = _survival_points(offers)
    if not pts:
        return SharpFair(None, False, None, None, None, 0)
    tp = float(target_point)
    for point, _k, fair_over in pts:
        if abs(point - tp) < 1e-9:                     # exact line -> pure devig
            return SharpFair(fair_over, False, 0.0, None, None, len(pts))
    shape = fit_count_shape(pts, default_dispersion)
    nearest = min(abs(point - tp) for point, _k, _s in pts)
    if shape is None:
        return SharpFair(None, True, nearest, None, None, len(pts))
    mean, disp = shape
    if mean == float("inf"):
        return SharpFair(1.0, True, nearest, mean, disp, len(pts))
    prob = negbin_at_least(_line_threshold(tp), mean, disp)
    return SharpFair(prob, True, nearest, mean, disp, len(pts))
