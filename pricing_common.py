"""Shared odds/EV math, calibration glue, and cross-concern projection
multipliers for the pricing engine.

This is the layer between the leaves (stats / calibration_loader / odds_client)
and the three pricing modules that all build on it: analysis (margin / ML /
totals / spreads), props, and parlay. Keeping these shared helpers here avoids
an ``analysis <-> props`` import cycle (the margin path and the props path both
use ``_starter_adjustment`` and the venue/defense multipliers). ``analysis``
re-exports every name below for backward compatibility with its importers.
"""

from datetime import datetime, timezone

from calibration_loader import (
    load_market_blend,
    load_prob_shrink,
    load_starter_adjustment,
    load_value_gate,
)
from odds_client import american_to_decimal, devig_two_way


def et_local_date(commence_iso):
    """US-Eastern calendar date (YYYY-MM-DD) for an ISO commence timestamp.

    The Odds API ``commence_time`` is UTC; a late US game (first pitch after
    ~8pm ET) has a UTC date one day AHEAD of its official/local game date. The
    bet ledger and prediction log key grading and display off the local date, so
    derive it in ``America/New_York`` — the convention ``app.py`` already uses
    for MLB matchup features. Falls back to the raw UTC date when the timestamp
    is unparseable or tz data is unavailable; never raises."""
    if not commence_iso:
        return None
    raw = str(commence_iso)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return raw[:10] or None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return dt.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return dt.date().isoformat()


# Calibration caches (populated lazily). Tests mutate these in place through the
# analysis namespace, which re-exports these exact objects — so they must be
# module-level dicts here, never re-bound.
_MARKET_BLEND_CACHE = {}
_PROB_SHRINK_CACHE = {}
_STARTER_ADJ_CACHE = {}
_VALUE_GATE_CACHE = {}


def _market_suppressed(sport_key, market):
    """True if `market` is in the calibration value_gate ``suppress`` list — a
    suppressed market (player prop OR team market) must NEVER be flagged as a
    value bet or enter bet selection. Team markets ('moneyline'/'spreads'/
    'totals') share the same suppress list as props; the names don't collide.
    Fails OPEN (not suppressed) on any load error so a config miss can't silently
    blank the whole card."""
    if not sport_key or not market:
        return False
    if sport_key not in _VALUE_GATE_CACHE:
        try:
            _VALUE_GATE_CACHE[sport_key] = load_value_gate(sport_key) or {}
        except Exception:
            _VALUE_GATE_CACHE[sport_key] = {}
    return market in (_VALUE_GATE_CACHE[sport_key].get("suppress") or [])


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


def kelly_fraction(prob, american_price, fraction=0.5, cap=0.05):
    """Fractional-Kelly stake as a fraction of bankroll, in [0, cap].

    The Kelly optimum for a Bernoulli bet at net decimal odds ``b`` is
    ``f* = (p*b - (1-p)) / b``, which is exactly ``_expected_roi(p, price) / b``
    (the expected profit per dollar staked, divided by the net odds). Scaled by
    ``fraction`` (0.5 = half-Kelly) and clamped to ``cap`` (a hard fraction of
    bankroll). Returns 0.0 — never a negative or a raise — for a missing price,
    a non-positive-EV bet (the edge is already gone to the vig, mirroring the
    ``_prop_is_value`` EV gate), or any degenerate input, so a caller can size a
    whole slate without guarding each leg."""
    try:
        if prob is None or american_price is None:
            return 0.0
        er = _expected_roi(prob, american_price)
        if er is None or er <= 0.0:
            return 0.0
        b = american_to_decimal(american_price) - 1.0
        if b <= 0.0:
            return 0.0
        f = fraction * (er / b)
        if f <= 0.0:
            return 0.0
        return min(f, cap)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def kelly_stake(prob, american_price, bankroll, fraction=0.5, cap=0.05):
    """Dollar stake = ``bankroll * kelly_fraction(...)``, rounded to cents.

    Fail-open to 0.0 on any bad input or a non-positive bankroll, and never
    negative, so it always satisfies the ledger ``stake >= 0`` constraint."""
    try:
        bankroll = float(bankroll)
    except (TypeError, ValueError):
        return 0.0
    if bankroll <= 0.0:
        return 0.0
    f = kelly_fraction(prob, american_price, fraction, cap)
    return round(bankroll * f, 2)


def prob_interval_low(prob, n_eff, z=1.0):
    """Conservative LOWER bound of a win-probability estimate, via a normal (Wald)
    interval sized by the effective sample count behind it:

        prob_low = prob - z * sqrt(prob*(1-prob)/n_eff)     (clamped to [0, prob])

    Larger ``n_eff`` -> tighter interval -> prob_low near prob; a thin sample (or a
    mid-range prob) -> wider interval -> more conservative. ``z`` is the interval
    half-width in SDs (1.0 ~ 1 SD). Fail-OPEN to the point estimate (returns
    ``prob`` unchanged) when n_eff is missing/<=0 or prob is degenerate, so a caller
    can apply it unconditionally and only ever get a <= prob result."""
    import math
    try:
        p = float(prob)
        n = float(n_eff)
    except (TypeError, ValueError):
        return prob
    if not (0.0 < p < 1.0) or n <= 0.0:
        return prob
    low = p - float(z) * math.sqrt(p * (1.0 - p) / n)
    return low if low > 0.0 else 0.0


def kelly_fraction_uncertain(prob, prob_low, american_price, fraction=0.5, cap=0.05):
    """Uncertainty-aware fractional Kelly: size off the LOWER bound of the win-prob
    interval (``prob_low``) rather than the point estimate.

    This attacks selection optimism directly (our named #1 forward>backtest cause):
      * a shaky edge (wide interval -> low prob_low) is staked SMALL;
      * an edge whose interval spans break-even (prob_low below the price's implied
        prob) is ABSTAINED -- the existing kelly_fraction EV<=0 guard returns 0.0;
      * since prob_low <= prob, it NEVER exceeds the point-estimate Kelly.
    ``prob_low is None`` -> falls back to the point estimate (byte-identical to
    kelly_fraction), so the whole path is opt-in per leg."""
    if prob_low is None:
        return kelly_fraction(prob, american_price, fraction, cap)
    return kelly_fraction(prob_low, american_price, fraction, cap)


def kelly_stake_uncertain(prob, prob_low, american_price, bankroll,
                          fraction=0.5, cap=0.05):
    """Dollar stake off the uncertainty-aware fractional Kelly (see
    ``kelly_fraction_uncertain``). Fail-open to 0.0 exactly like ``kelly_stake``."""
    try:
        bankroll = float(bankroll)
    except (TypeError, ValueError):
        return 0.0
    if bankroll <= 0.0:
        return 0.0
    f = kelly_fraction_uncertain(prob, prob_low, american_price, fraction, cap)
    return round(bankroll * f, 2)


def scale_to_slate_cap(stakes, bankroll, cap_fraction):
    """Proportionally scale a list of dollar stakes so their sum does not exceed
    ``cap_fraction * bankroll`` (a slate-total exposure cap).

    Returns a NEW list rounded to cents. A no-op (each stake merely rounded) when
    the sum is already within the cap, or when any input is degenerate — so the
    caller can apply it unconditionally. Never raises."""
    try:
        stakes = [float(s or 0.0) for s in stakes]
    except (TypeError, ValueError):
        return list(stakes)
    try:
        cap = float(bankroll) * float(cap_fraction)
    except (TypeError, ValueError):
        return [round(s, 2) for s in stakes]
    total = sum(stakes)
    if cap <= 0.0 or total <= 0.0 or total <= cap:
        return [round(s, 2) for s in stakes]
    factor = cap / total
    return [round(s * factor, 2) for s in stakes]


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


def _venue_match_multiplier(past_is_home, upcoming_is_home, sport_key,
                            strength=None):
    """
    Return a multiplier that up-weights past games played at the same venue
    type (home vs road) as the upcoming game. If either side's venue is
    unknown, return 1.0 (no adjustment).

    ``strength`` (P2.1 parity fix): when a numeric half-spread is supplied, use
    ``(1+strength, 1-strength)`` — IDENTICAL to backtest.venue_mult, so a
    backtest-selected venue_strength behaves the same live as it was validated.
    When None (default / team-level callers) fall back to the fixed per-sport
    VENUE_MATCH_WEIGHTS (legacy behavior).
    """
    if past_is_home is None or upcoming_is_home is None:
        return 1.0
    if strength is not None:
        match_w, mismatch_w = 1.0 + strength, 1.0 - strength
    else:
        match_w, mismatch_w = VENUE_MATCH_WEIGHTS.get(sport_key, DEFAULT_VENUE_WEIGHTS)
    return match_w if past_is_home == upcoming_is_home else mismatch_w


def _resolve_team_defense(opp_name, team_defense):
    """Tolerant {team_name: value} lookup — exact, then partial substring match.

    The opponent name on a gamelog/upcoming game (from the odds/ESPN feed) does
    not always match the ``team_defense`` key verbatim (e.g. "Yankees" vs "New
    York Yankees", "LA Angels" vs "Los Angeles Angels"). A plain ``dict.get``
    fails open — the defense adjustment silently becomes 1.0 — so the runtime
    quietly drops a feature the backtest validated. This mirrors the backtest's
    ``_resolve_opp_pts_allowed`` so runtime pricing and the sweep agree on which
    matchups actually get the adjustment. Returns None when nothing matches
    (caller then applies no adjustment)."""
    if not opp_name or not team_defense:
        return None
    if opp_name in team_defense:
        return team_defense[opp_name]
    lo = opp_name.lower()
    for k, v in team_defense.items():
        kl = k.lower()
        if lo in kl or kl in lo or kl.split()[-1] == lo or lo.split()[-1] == kl.split()[-1]:
            return v
    return None


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
