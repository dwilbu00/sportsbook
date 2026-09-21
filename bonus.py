"""bonus.py — promo/bonus EV engine (the actual +EV mechanism).

We can't beat the book on straights, but a profit-BOOST flips high-probability −EV bets to
+EV — because the boost pays on the PAYOUT, and it's biggest on parlays. The catch: the
boosted EV is only real if the leg probabilities are genuinely calibrated (over-confident P
=> fake +EV). So this engine consumes OUR calibrated P (trustworthy props only) + the book
odds + a bonus spec, and returns honest boosted EV, qualification, and Kelly-safe sizing.

Bonus schema (Doug's parameters):
  bet_type       : 'single' | 'parlay' | 'sgp' | 'any' | 'any_parlay' | 'sgp_sgpx'
  boost_pct      : profit boost as a fraction (0.30 = +30% on winnings)
  min_odds_leg   : each leg's American odds must be >= this (not shorter than)
  min_odds_overall: combined American odds must be >= this
  min_legs       : minimum number of legs (parlay boosts often require >=2 or >=3)
  max_wager      : $ cap
  min_wager      : $ floor
  markets        : MARKET SCOPE — the market/prop keys this bonus applies to (e.g.
                   ("batter_home_runs",) or ("anytime_td",) or ("team",)). Empty = all
                   eligible markets for the sport. Drives the bonus-scoped analysis +
                   which legs the optimizer builds.

EV per $1 with a profit boost b on a bet of true prob P and decimal odds D:
  win  -> +(D-1)*(1+b);  lose -> -1
  EV   =  P*(D-1)*(1+b) - (1-P)
Kelly fraction on the boosted payout: f* = P - (1-P)/((D-1)*(1+b)).
"""
from dataclasses import dataclass


@dataclass
class Bonus:
    bet_type: str                        # 'single' | 'parlay' | 'sgp' | 'any'
    boost_pct: float                     # profit boost fraction (0.30 = +30% on winnings)
    min_odds_leg: float = -100000.0      # American; each leg must be >= this (default = no floor)
    min_odds_overall: float = -100000.0  # American; combined must be >= this
    min_legs: int = 1                    # parlay boosts often require >=2 or >=3
    max_wager: float = 1e9
    min_wager: float = 0.0
    book: str = "draftkings"             # DK bonuses use DK odds; FD bonuses use FD odds
    label: str = ""                      # human tag for reporting (blank -> display_name)
    sport: str = "americanfootball_nfl"  # which sport's slate this boost applies to
    markets: tuple = ()                  # market scope (see module docstring); () = all


# ── Human-readable auto-naming (book · sport · type · boost · scope) ────────────
_BOOK_TAGS = {"draftkings": "DK", "fanduel": "FD", "pinnacle": "Pin"}
_SPORT_TAGS = {"baseball_mlb": "MLB", "americanfootball_nfl": "NFL",
               "basketball_nba": "NBA", "icehockey_nhl": "NHL"}
_MARKET_TAGS = {
    "team": "team", "batter_home_runs": "HR", "anytime_td": "TD",
    "player_anytime_td": "TD", "batter_hits": "hits", "batter_total_bases": "TB",
    "batter_rbis": "RBI", "batter_strikeouts": "bat-K", "pitcher_strikeouts": "P-K",
    "pitcher_outs": "outs", "pitcher_earned_runs": "ER", "player_receptions": "rec",
    "player_rush_attempts": "rush-att", "player_pass_attempts": "pass-att",
    "player_pass_yds": "pass-yds", "player_rush_yds": "rush-yds",
    "player_reception_yds": "rec-yds", "player_pass_tds": "pass-TD",
}


def _market_scope_tag(markets):
    """Short human tag for a bonus's market scope, or '' when it spans all markets."""
    mkts = [m for m in (markets or ()) if m]
    if not mkts:
        return ""
    tags = [_MARKET_TAGS.get(m, m.replace("player_", "").replace("batter_", "")
                             .replace("pitcher_", "")) for m in mkts]
    return "+".join(tags[:3]) + ("+…" if len(tags) > 3 else "")


def display_name(bonus):
    """Auto-generated readable name: 'DK · MLB · SGP 50% · HR'. Used in the manager UI
    and as the ticket label fallback when the freeform `label` is blank."""
    book = _BOOK_TAGS.get(bonus.book, (bonus.book or "?")[:3].upper())
    sport = _SPORT_TAGS.get(bonus.sport, (bonus.sport or "?").upper())
    typ = (bonus.bet_type or "?").upper().replace("_", "-")
    boost = f"{bonus.boost_pct * 100:.0f}%"
    scope = _market_scope_tag(getattr(bonus, "markets", ()))
    parts = [book, sport, f"{typ} {boost}"]
    if scope:
        parts.append(scope)
    return " · ".join(parts)


def is_valid_american(a):
    """American odds are valid only at |a| >= 100 (a <= -100 or a >= +100). 0, ±1..99,
    NaN and infinities are invalid — rejecting them stops a zero/garbage price from
    crashing the pricer with a ZeroDivisionError or producing nonsense EV. [F15]"""
    try:
        a = float(a)
    except (TypeError, ValueError):
        return False
    if a != a or a in (float("inf"), float("-inf")):
        return False
    return abs(a) >= 100.0


def american_to_dec(a):
    a = float(a)
    if not is_valid_american(a):
        raise ValueError(f"invalid American odds {a!r}: must be <= -100 or >= +100")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def dec_to_american(d):
    d = float(d)
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def combined_decimal(legs_american):
    prod = 1.0
    for a in legs_american:
        prod *= american_to_dec(a)
    return prod


def boosted_ev_per_dollar(P, dec_odds, boost):
    """EV per $1 staked with a profit boost `boost` (fraction)."""
    return P * (dec_odds - 1.0) * (1.0 + boost) - (1.0 - P)


def kelly_fraction(P, dec_odds, boost):
    """Full-Kelly fraction of bankroll for the boosted payout (0 if -EV)."""
    b = (dec_odds - 1.0) * (1.0 + boost)          # net odds actually received
    if b <= 0:
        return 0.0
    f = P - (1.0 - P) / b
    return max(0.0, f)


def _min_dec(min_odds):
    """Decimal floor for a promo min-odds constraint; an invalid/absent value means NO
    constraint (floor 1.0) rather than a crash on a malformed stored promo. [F15]"""
    try:
        return american_to_dec(min_odds)
    except (ValueError, TypeError):
        return 1.0


def _leg_ok(a, bonus):
    try:
        return american_to_dec(a) >= _min_dec(bonus.min_odds_leg) - 1e-9
    except (ValueError, TypeError):
        return False                      # a leg with invalid odds can't qualify


def evaluate(legs, bonus, joint_prob=None, bankroll=1000.0, kelly_frac=0.25):
    """legs = [(P, american_odds), ...]. joint_prob overrides the independent product
    (pass the correlation-adjusted joint for an SGP). Returns the full EV/sizing verdict."""
    n = len(legs)
    kind = "single" if n == 1 else "parlay"
    # A leg with invalid American odds can't be priced -> not a qualifying ticket
    # (rather than crashing the whole page). [F15]
    try:
        dec = combined_decimal([a for _p, a in legs])
    except (ValueError, TypeError):
        return {"n_legs": n, "kind": kind, "joint_P": None, "combined_dec": None,
                "combined_american": None, "boosted_ev_pct": 0.0, "qualifies": False,
                "legs_ok": False, "overall_ok": False, "type_ok": False,
                "legs_count_ok": False, "kelly_stake": 0.0}
    P = joint_prob if joint_prob is not None else _prod(p for p, _a in legs)
    ev = boosted_ev_per_dollar(P, dec, bonus.boost_pct)
    # constraints
    legs_ok = all(_leg_ok(a, bonus) for _p, a in legs)
    overall_ok = dec >= _min_dec(bonus.min_odds_overall) - 1e-9
    # Ticket-type gate (leg-count only; the OPTIMIZER enforces cross-game vs same-game
    # composition). bet_type vocabulary:
    #   single       -> exactly 1 leg
    #   any          -> anything (singles + parlays + SGPs)
    #   any_parlay   -> any multi-leg (excludes singles) — cross-game OR same-game
    #   parlay       -> cross-game multi-leg
    #   sgp          -> same-game multi-leg
    #   sgp_sgpx     -> SGP or SGPx (same-game, or multiple SGPs combined across games)
    if bonus.bet_type == "any":
        type_ok = True
    elif bonus.bet_type == "single":
        type_ok = n == 1
    else:                                   # parlay | sgp | sgp_sgpx | any_parlay
        type_ok = n >= 2
    legs_count_ok = n >= max(1, bonus.min_legs)
    qualifies = legs_ok and overall_ok and type_ok and legs_count_ok
    # sizing (fractional Kelly, clamped to wager bounds when +EV & qualifying)
    f = kelly_fraction(P, dec, bonus.boost_pct) * kelly_frac
    stake = 0.0
    if qualifies and ev > 0:
        # The promo min/max are ticket CONSTRAINTS, not permission to stake beyond the
        # bankroll. Coerce an unavailable/negative/non-finite bankroll to 0, and only
        # size a ticket the bankroll can AFFORD: if it can't cover the promo minimum
        # there is no eligible sized stake (0) rather than a phantom min-wager bet
        # against money the app has no evidence exists. [F11]
        try:
            bk = float(bankroll)
        except (TypeError, ValueError):
            bk = 0.0
        if bk != bk or bk in (float("inf"), float("-inf")) or bk < 0:
            bk = 0.0
        if bk >= bonus.min_wager:
            stake = min(bonus.max_wager, bk, max(bonus.min_wager, f * bk))
    return {"n_legs": n, "kind": kind, "joint_P": P, "combined_dec": dec,
            "combined_american": dec_to_american(dec),
            "boosted_ev_pct": ev * 100.0, "qualifies": qualifies,
            "legs_ok": legs_ok, "overall_ok": overall_ok, "type_ok": type_ok,
            "legs_count_ok": legs_count_ok, "kelly_stake": stake}


def _prod(it):
    p = 1.0
    for x in it:
        p *= x
    return p


# Doug's live scenarios (2026-09). book is set per-run (DK bonus => DK odds, FD bonus => FD odds).
LIVE_BONUSES = [
    Bonus(bet_type="any", boost_pct=0.25, min_odds_leg=-200, max_wager=10.0,
          label="25% any-bet, max $10, min leg -200"),
    Bonus(bet_type="sgp", boost_pct=0.30, min_odds_leg=-250, min_legs=3,
          label="30% SGP, min leg -250, >=3 legs"),
    Bonus(bet_type="parlay", boost_pct=0.50, min_odds_leg=-300, min_legs=2, max_wager=10.0,
          label="50% parlay/SGP, max $10, min leg -300"),
]


if __name__ == "__main__":
    # sanity checks — the numbers cited in discussion
    b30 = Bonus(bet_type="single", boost_pct=0.30)
    r = evaluate([(0.50, -110)], b30)
    print(f"coin-flip -110 + 30% boost: EV = {r['boosted_ev_pct']:+.1f}%  (expect ~+9.1%)")
    p3 = Bonus(bet_type="parlay", boost_pct=0.30, min_odds_overall=100)
    r = evaluate([(0.52, -110)] * 3, p3)
    print(f"3-leg 52% parlay -110 + 30% boost: EV = {r['boosted_ev_pct']:+.1f}%, "
          f"combined={r['combined_american']:+.0f}, qualifies={r['qualifies']}  (expect ~+23%)")
    r0 = evaluate([(0.52, -110)] * 3, Bonus(bet_type="parlay", boost_pct=0.0))
    print(f"same parlay NO boost: EV = {r0['boosted_ev_pct']:+.1f}%  (expect ~-2%)")
    # min_legs gate: the 30% SGP bonus (>=3 legs) must REJECT a 2-leg ticket
    sgp = LIVE_BONUSES[1]
    r2 = evaluate([(0.70, -180)] * 2, sgp)
    r3 = evaluate([(0.70, -180)] * 3, sgp)
    print(f"30% SGP 2-leg qualifies={r2['qualifies']} (expect False, <3 legs); "
          f"3-leg qualifies={r3['qualifies']} EV={r3['boosted_ev_pct']:+.1f}%")
