"""Shared odds/EV math, calibration glue, and cross-concern projection
multipliers for the pricing engine.

This is the layer between the leaves (stats / calibration_loader / odds_client)
and the three pricing modules that all build on it: analysis (margin / ML /
totals / spreads), props, and parlay. Keeping these shared helpers here avoids
an ``analysis <-> props`` import cycle (the margin path and the props path both
use ``_starter_adjustment`` and the venue/defense multipliers). ``analysis``
re-exports every name below for backward compatibility with its importers.
"""

from calibration_loader import (
    load_market_blend,
    load_prob_shrink,
    load_starter_adjustment,
)
from odds_client import american_to_decimal, devig_two_way


# Calibration caches (populated lazily). Tests mutate these in place through the
# analysis namespace, which re-exports these exact objects — so they must be
# module-level dicts here, never re-bound.
_MARKET_BLEND_CACHE = {}
_PROB_SHRINK_CACHE = {}
_STARTER_ADJ_CACHE = {}


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


def profit(american_price, stake, won):
    """Realized profit on a settled bet at ``american_price`` for ``stake`` units.

    ``won`` is True (win), False (loss), or None (push). A win returns
    ``(decimal_odds - 1) * stake``; a loss returns ``-stake``; a push returns
    0.0 (stake returned). Mirrors the priced-ROI formula used for the forward
    prediction log, but scaled by an actual dollar stake for the bets ledger."""
    try:
        stake = float(stake)
    except (TypeError, ValueError):
        return 0.0
    if won is None:
        return 0.0
    if not won:
        return -stake
    if american_price is None:
        return 0.0
    return (american_to_decimal(american_price) - 1.0) * stake


def _prop_is_value(edge, threshold, expected_roi):
    """Decide whether a player prop qualifies as a value bet.

    Requires BOTH: the fair-market edge clears `threshold`, AND the bet is
    positive-EV at the price actually bettable. The edge is measured against the
    de-vigged consensus probability, but the stake is placed at the vigged best
    executable price, so the single-side vig can exceed the edge threshold — an
    edge-only gate can flag a -EV bet as value (e.g. Over -300 at +5.0% edge is
    -1.03% ROI). Requiring `expected_roi > 0` closes that gap and rejects legs
    with no executable price (`expected_roi is None`).
    """
    return (
        edge >= threshold
        and expected_roi is not None
        and expected_roi > 0
    )


def _devig_fair(side_implied, other_implied):
    """De-vigged fair probability for one side of a two-way market.

    Returns the raw implied prob when the other side is unavailable (the hold
    cannot be removed from a one-sided market), or None when this side has no
    price. Used so every market measures edge against the same fair baseline
    that player props already use, instead of the vig-inflated raw implied prob.
    """
    if side_implied is None:
        return None
    if other_implied is None:
        return side_implied
    fair, _ = devig_two_way(side_implied, other_implied)
    return fair


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
