"""R2 edge computation: measure DraftKings' price against the sharp (Pinnacle)
fair probability at DK's own line.

The R2 thesis is that the harvestable edge is DK being STALE/mispriced relative to
the sharp consensus — not our model out-predicting the market. So we treat the
sharp book's no-vig fair probability (via r2_sharp) as the truth estimate and ask:
at DK's posted price, is the bet +EV under that truth?

  EV per $1 staked at DK  =  sharp_fair_prob * dk_decimal_odds - 1

A positive EV means DK is offering a price richer than the sharp fair — the R2
signal. We compute it per bettable DK leg (prop OVER/UNDER at DK's line, or a
moneyline side), carrying the projection metadata (was the sharp line the same as
DK's, or did we project across a line gap — Doug's Pinnacle-1.5-vs-DK-0.5 case) so
the backtest can bucket by it.

Pure module (no warehouse/IO): the backtest feeds DK + Pinnacle rows and consumes
LegEdge records; live selection could reuse the same primitives later.
"""
from collections import namedtuple

from odds_client import american_to_decimal, american_to_implied_prob
from r2_sharp import fair_prob_at_line, fair_two_way

# One bettable DK leg scored against the sharp fair.
#   side       : 'OVER' | 'UNDER' | 'ML'
#   point      : DK's line (None for moneyline)
#   dk_price   : DK American price we'd bet at
#   dk_implied : DK's vigged implied prob (what DK's price says)
#   sharp_fair : sharp no-vig fair prob for THIS side at DK's line
#   ev         : expected profit per $1 at dk_price under sharp_fair (the R2 signal)
#   prob_edge  : sharp_fair - dk_implied (how much cheaper DK is, in prob terms)
#   projected  : True if the sharp fair was projected across a line gap
#   distance   : |DK point - nearest sharp posted line| (0.0 = same line)
#   n_lines    : # of two-sided sharp lines that informed the fair prob
LegEdge = namedtuple("LegEdge",
                     "side point dk_price dk_implied sharp_fair ev prob_edge "
                     "projected distance n_lines")


def _american(price):
    if price is None or isinstance(price, bool):
        return None
    try:
        return int(price)
    except (TypeError, ValueError):
        return None


def _leg(side, point, dk_price, sharp_fair, projected, distance, n_lines):
    """Build a LegEdge for one side given its sharp fair prob and DK price."""
    p = _american(dk_price)
    if p is None or sharp_fair is None:
        return None
    dk_implied = american_to_implied_prob(p)
    ev = sharp_fair * american_to_decimal(p) - 1.0
    return LegEdge(side, point, p, dk_implied, sharp_fair, ev,
                   sharp_fair - dk_implied, projected, distance, n_lines)


def prop_leg_edges(dk_point, dk_over_price, dk_under_price,
                   pinnacle_offers, default_dispersion=0.0):
    """LegEdges for DK's OVER/UNDER on a prop at ``dk_point``, scored against the
    sharp fair at that same point.

    ``pinnacle_offers`` = the sharp book's two-sided lines on this prop
    ([{point, over_price, under_price}, ...]); may be at a DIFFERENT point than DK
    (projected). Returns [] if the sharp offers can't be priced at DK's line."""
    sf = fair_prob_at_line(pinnacle_offers, dk_point, default_dispersion)
    if sf.prob is None:
        return []
    over_fair = sf.prob
    under_fair = 1.0 - sf.prob      # same line -> the two sides partition the mass
    legs = []
    over = _leg("OVER", dk_point, dk_over_price, over_fair,
                sf.projected, sf.distance, sf.n_lines)
    under = _leg("UNDER", dk_point, dk_under_price, under_fair,
                 sf.projected, sf.distance, sf.n_lines)
    if over:
        legs.append(over)
    if under:
        legs.append(under)
    return legs


def moneyline_edge(dk_price, pin_price, pin_other_price):
    """LegEdge for a DK moneyline side, scored against Pinnacle's deviged fair for
    that side. ``pin_price`` = Pinnacle's price on the SAME side as ``dk_price``;
    ``pin_other_price`` = Pinnacle's price on the opponent (needed to remove vig).
    No line/projection (moneyline has no point). None if unpriceable."""
    fair, _ = fair_two_way(pin_price, pin_other_price)
    return _leg("ML", None, dk_price, fair, False, 0.0, 1)


def best_positive_leg(legs):
    """The single highest-EV leg among ``legs`` if it is +EV, else None. A book
    rarely offers +EV on both sides, so this picks the bet worth taking."""
    pos = [lg for lg in legs if lg is not None and lg.ev > 0.0]
    return max(pos, key=lambda lg: lg.ev) if pos else None
