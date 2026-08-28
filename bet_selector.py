"""Auto-pick the top-N rule-compliant value bets for the DraftKings checklist.

Pure, Streamlit-free, and testable. `select_top_bets` ranks the pool of value
bets (by EV / edge / win-prob / balanced), then greedily fills a slate best-first,
skipping any bet that violates a standalone filter or conflicts with a bet already
chosen. Greedy is correct here: these are *independent single bets* (total EV is
additive), so only feasibility couples them — unlike a parlay whose legs multiply
into one joint payout. We therefore reuse the parlay generator's anti-correlation
sources (`parlay._has_hard_conflict`, `parlay._pair_correlation`) but NOT its
copula / enumeration.

Rule layers applied while filling:
  L1  structural hard conflicts (all sports) — parlay._has_hard_conflict
  L2  anti-correlation (all sports) — parlay._pair_correlation(...) <= -0.20
  L3  MLB-only slate rules, keyed on event_id (unique per game — doubleheader safe):
      (a) Rule of 3: <= 3 batter_hits OVER per team (team=None exempt)
      (b) no pitcher_* UNDER + opposing batter_hits UNDER in the same game
      (c) no game-total OVER + player-prop UNDER in the same game
      (d) batter_hits OVER excluded only when the confirmed batting order slot > 4
          (unconfirmed / None fails OPEN — the bet is allowed)

The pool entry shape is exactly what `app._iter_wager_candidates` yields:
`(selection_key, bet_type, side, candidate)` where bet_type ∈
{moneyline, spread, total, player_prop} and side ∈ {OVER, UNDER, None} (props carry
their direction in candidate["direction"], not in `side`).
"""

import parlay

_NEG_INF = float("-inf")

# Anti-correlation cut: block a pair whose rank correlation is at or below this.
_CORR_BLOCK = -0.20

# Max batter_hits OVER bets from a single team (the user's "Rule of 3").
_MAX_TEAM_HITS_OVERS = 3


# ── metric resolution (per bet_type / side) ────────────────────────────────

def _ev(bet_type, side, c):
    if bet_type == "total":
        return c.get("over_expected_roi_pct") if side == "OVER" \
            else c.get("under_expected_roi_pct")
    return c.get("expected_roi_pct")


def _edge(bet_type, side, c):
    if bet_type == "total":
        return c.get("over_edge_pct") if side == "OVER" \
            else c.get("under_edge_pct")
    return c.get("edge_pct")


def _prob(bet_type, side, c):
    """Model probability that the bet hits, as a percent (0–100)."""
    if bet_type == "moneyline":
        return c.get("blended_prob")
    if bet_type in ("spread", "runline_coherence"):
        return c.get("cover_rate")
    if bet_type == "total":
        ohr = c.get("over_hit_rate")
        if ohr is None:
            return None
        return ohr if side == "OVER" else 100.0 - ohr
    # player_prop — over_rate is P(OVER); direction lives on the candidate.
    over_rate = c.get("over_rate")
    if over_rate is None:
        return None
    return over_rate if c.get("direction") == "OVER" else 100.0 - over_rate


_METRIC_FN = {"ev": _ev, "edge": _edge, "prob": _prob}


def _num(x):
    """Coerce a possibly-None metric to a sortable float (None sinks to the end)."""
    return x if x is not None else _NEG_INF


def _order(pool, metric):
    """Return the pool sorted best-first for a simple (non-balanced) metric.

    Ties break by edge (desc) then selection_key (asc) so the order is fully
    deterministic — no None-vs-float comparison can raise.
    """
    fn = _METRIC_FN.get(metric, _ev)

    def key(entry):
        sel_key, bet_type, side, cand = entry
        return (
            -_num(fn(bet_type, side, cand)),
            -_num(_edge(bet_type, side, cand)),
            sel_key,
        )

    return sorted(pool, key=key)


def _rank(pool, metric):
    """Order the whole pool best-first for the chosen ranking metric.

    "balanced" blends the EV and win-probability orderings with a Borda count
    (sum of the two 0-based positions; lower is better) so a bet needs to score
    well on *both* to rise — without cross-pool unit normalization.
    """
    if metric == "balanced":
        ev_pos = {e[0]: i for i, e in enumerate(_order(pool, "ev"))}
        prob_pos = {e[0]: i for i, e in enumerate(_order(pool, "prob"))}
        return sorted(
            pool,
            key=lambda e: (ev_pos.get(e[0], 0) + prob_pos.get(e[0], 0), e[0]),
        )
    return _order(pool, metric)


# ── leg adapter for the parlay rule functions ──────────────────────────────

def _leg(bet_type, side, cand):
    """Minimal leg dict holding only the five fields parlay._has_hard_conflict /
    _pair_correlation read. NOT parlay._normalize_legs (which re-applies stricter
    gates, drops the selection_key, and splits a total into two legs)."""
    if bet_type == "total":
        mapped = "total_over" if side == "OVER" else "total_under"
    elif bet_type == "player_prop":
        mapped = ("player_prop_over" if cand.get("direction") == "OVER"
                  else "player_prop_under")
    elif bet_type == "runline_coherence":
        # A coherence run-line IS a run-line: treat it as a spread for structural /
        # anti-correlation conflict checks so it can't co-select with the model's own
        # spread (or opposite side) on the same game.
        mapped = "spread"
    else:
        mapped = bet_type  # moneyline / spread, unchanged
    return {
        "game_key": cand.get("event_id"),  # unique per game — doubleheader safe
        "team": cand.get("team"),
        "bet_type": mapped,
        "player": cand.get("player"),
        "prop_key": cand.get("prop"),
    }


def correlation_haircut(stakes, legs, sport_key):
    """R6 (honest Kelly): shrink each leg's stake by its POSITIVE-correlation load
    with the rest of the slate so positively-correlated legs (same game / same
    pitcher / shared game conditions) aren't overbet when sized independently.

        stake_i *= 1 / (1 + sum_{j!=i} max(rho_ij, 0))

    with rho from parlay._pair_correlation (0.0 cross-game). Fully independent legs
    are unchanged; a cluster of n positively-correlated legs collapses toward one
    leg's total exposure (n perfectly-correlated → 1/n each). Sizing correlated legs
    off a frozen bankroll without this overbets the cluster (an all-+EV slate can
    still lose). ``legs`` are _leg()-shape dicts aligned 1:1 with ``stakes``; a None
    leg or falsy stake passes through untouched. Returns a NEW list. Never raises."""
    n = len(stakes)
    if n <= 1:
        return list(stakes)
    out = []
    for i in range(n):
        s = stakes[i]
        leg_i = legs[i] if i < len(legs) else None
        if not s or leg_i is None:
            out.append(s)
            continue
        load = 0.0
        for j in range(n):
            leg_j = legs[j] if j < len(legs) else None
            if j == i or leg_j is None:
                continue
            try:
                rho = parlay._pair_correlation(leg_i, leg_j, sport_key)
            except Exception:
                rho = 0.0
            if rho > 0:
                load += rho
        out.append(s / (1.0 + load))
    return out


# ── MLB slate rules (L3) ───────────────────────────────────────────────────

def _is_batter_hits_over(rec):
    c = rec["cand"]
    return (rec["bet_type"] == "player_prop"
            and c.get("prop") == "batter_hits"
            and c.get("direction") == "OVER")


def _same_game(a, b):
    ea, eb = a["cand"].get("event_id"), b["cand"].get("event_id")
    return ea is not None and ea == eb


def _mlb_pitcher_vs_hitter_under(a, b):
    """Rule (b): a pitcher_* UNDER and a batter_hits UNDER on *opposing* teams in
    the same game are a negative-correlation trap. Opposing = same event_id, both
    teams resolved and different. Directional pair — call both orderings."""
    ca, cb = a["cand"], b["cand"]
    a_pitcher_under = (a["bet_type"] == "player_prop"
                       and ca.get("direction") == "UNDER"
                       and str(ca.get("prop") or "").startswith("pitcher_"))
    b_hits_under = (b["bet_type"] == "player_prop"
                    and cb.get("direction") == "UNDER"
                    and cb.get("prop") == "batter_hits")
    if not (a_pitcher_under and b_hits_under):
        return False
    if not _same_game(a, b):
        return False
    ta, tb = ca.get("team"), cb.get("team")
    if ta is None or tb is None:
        return False
    return ta != tb


def _mlb_over_under_mix(a, b):
    """Rule (c): never co-select a game-total OVER with a player-prop UNDER in the
    same game. Directional pair — call both orderings."""
    cb = b["cand"]
    a_total_over = a["bet_type"] == "total" and a["side"] == "OVER"
    b_prop_under = (b["bet_type"] == "player_prop"
                    and cb.get("direction") == "UNDER")
    if not (a_total_over and b_prop_under):
        return False
    return _same_game(a, b)


# ── standalone + pairwise gating ───────────────────────────────────────────

def _passes_standalone(sport_key, rec):
    """Rule (d): drop a batter_hits OVER only when the batting order is *confirmed*
    outside the top four. None (lineup not posted) fails OPEN → allowed."""
    if sport_key == "baseball_mlb" and _is_batter_hits_over(rec):
        slot = rec["cand"].get("batting_order")
        if slot is not None and slot > 4:
            return False
    return True


def _pair_conflict(sport_key, a, b):
    """True if two chosen bets may not coexist on the slate."""
    la, lb = a["leg"], b["leg"]
    # L0 — one run-line per game. A coherence run-line (_leg maps it to "spread") and
    # the model's OWN run-line/spread are the same market on the same game; betting
    # both doubles the position. _has_hard_conflict only blocks OPPOSITE teams, so a
    # same-side duplicate (model HOME -1.5 + coherence HOME -1.5) would slip through.
    if la.get("bet_type") == "spread" and lb.get("bet_type") == "spread":
        ga, gb = la.get("game_key"), lb.get("game_key")
        if ga is not None and ga == gb:
            return True
    # L1 — structural hard conflicts (cross-game returns False internally).
    if parlay._has_hard_conflict(la, lb):
        return True
    # L2 — anti-correlation (cross-game returns 0.0 internally).
    if parlay._pair_correlation(la, lb, sport_key) <= _CORR_BLOCK:
        return True
    # L3 — MLB-only slate rules.
    if sport_key == "baseball_mlb":
        if _mlb_pitcher_vs_hitter_under(a, b) or _mlb_pitcher_vs_hitter_under(b, a):
            return True
        if _mlb_over_under_mix(a, b) or _mlb_over_under_mix(b, a):
            return True
    return False


def _team_hits_over_cap_hit(sport_key, rec, chosen):
    """Rule (a): would adding this batter_hits OVER exceed 3 from its team?
    team=None candidates are exempt (unresolved team can't be capped)."""
    if sport_key != "baseball_mlb" or not _is_batter_hits_over(rec):
        return False
    team = rec["cand"].get("team")
    if team is None:
        return False
    count = sum(1 for r in chosen
                if _is_batter_hits_over(r) and r["cand"].get("team") == team)
    return count >= _MAX_TEAM_HITS_OVERS


# ── public entry point ─────────────────────────────────────────────────────

def select_top_bets(pool, sport_key, n, metric="ev"):
    """Pick up to `n` selection_keys: the top-ranked value bets that satisfy the
    standalone filter and conflict with none of the bets already chosen.

    pool    : list of (selection_key, bet_type, side, candidate)
    sport_key: e.g. "baseball_mlb" (gates the MLB-only rules; other sports use L1+L2)
    n       : max bets to return (>= 1)
    metric  : "ev" (default) | "edge" | "prob" | "balanced"
    Returns a list of selection_keys, best-first, honoring all rule layers.
    """
    if n is None or n < 1 or not pool:
        return []

    chosen = []          # list of records (see below)
    chosen_keys = []
    for sel_key, bet_type, side, cand in _rank(pool, metric):
        if len(chosen_keys) >= n:
            break
        rec = {
            "sel_key": sel_key,
            "bet_type": bet_type,
            "side": side,
            "cand": cand,
            "leg": _leg(bet_type, side, cand),
        }
        if not _passes_standalone(sport_key, rec):
            continue
        if _team_hits_over_cap_hit(sport_key, rec, chosen):
            continue
        if any(_pair_conflict(sport_key, rec, other) for other in chosen):
            continue
        chosen.append(rec)
        chosen_keys.append(sel_key)
    return chosen_keys
