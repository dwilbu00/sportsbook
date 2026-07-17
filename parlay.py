"""Parlay / same-game-parlay (SGP) pricing.

The Gaussian-copula joint-probability machinery plus the leg-normalization,
conflict/correlation scoring, and top-N parlay search. Split out of analysis.py
(P3). A clean sibling of analysis (margin) and props: it consumes the candidate
dicts those analyzers produce (passed in as arguments) and never imports them,
so it only depends downward on stats / pricing_common / odds_client.
"""

import heapq
import math
import random

from odds_client import american_to_decimal
from pricing_common import _decimal_to_american
from stats import _norm_ppf


def _parlay_value_joint(best_joint, independent_joint, has_sgp):
    """Return the joint hit probability to use for a parlay's value/EV gate.

    Cross-game legs are independent to the book, so the naive product-of-prices
    payout is real and we credit the copula joint probability (`best_joint`).
    For a same-game parlay the book prices leg correlation *out of the payout*;
    crediting the copula's correlation benefit (best_joint > independent_joint)
    against that naive payout double-counts correlation and can flag a -EV SGP as
    value. Pricing the SGP against the independent joint cancels the correlation
    term the book removes, so a positively-correlated SGP cannot claim an edge
    the book will not pay.
    """
    return independent_joint if has_sgp else best_joint


def _cholesky(m):
    """
    Cholesky decomposition. Returns lower-triangular L such that L @ L^T = m.
    Raises ValueError if m is not positive-definite.
    """
    n = len(m)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = m[i][i] - s
                if v <= 0:
                    raise ValueError("matrix is not positive-definite")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (m[i][j] - s) / L[j][j]
    return L


def _make_psd_cholesky(R):
    """
    Return ``(L, applied_shrink)``: the Cholesky factor of R after shrinking its
    off-diagonals toward 0 by the largest factor in the ladder that still yields
    a positive-definite matrix, together with that shrink factor. ``1.0`` means R
    was used as-is; ``< 1.0`` means some correlation had to be discarded to make
    R PSD; ``0.0`` means it collapsed all the way to the identity (independence).
    Always succeeds. Returning ``applied_shrink`` lets callers surface how much
    correlation the copula silently had to drop.
    """
    n = len(R)
    for shrink in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        m = [[R[i][j] if i == j else shrink * R[i][j] for j in range(n)]
             for i in range(n)]
        try:
            return _cholesky(m), shrink
        except ValueError:
            continue
    return (_cholesky([[1.0 if i == j else 0.0 for j in range(n)]
                       for i in range(n)]), 0.0)


def _box_muller_pairs(rng, n):
    """Generate `n` standard-normal samples using Box-Muller. Returns a list."""
    out = []
    while len(out) < n:
        u1 = rng.random()
        if u1 <= 0.0:
            continue
        u2 = rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out.append(r * math.cos(theta))
        if len(out) < n:
            out.append(r * math.sin(theta))
    return out


def _gaussian_copula_joint_prob(probs, corr_matrix, n_samples=5000, seed=42,
                                return_shrink=False):
    """
    Monte-Carlo estimate of P(all events occur) under a Gaussian copula with
    given Bernoulli marginals `probs` and correlation matrix `corr_matrix`.

    Independence (corr_matrix = I) recovers the product ∏ p_i in expectation.

    With ``return_shrink=True`` returns ``(prob, applied_shrink)``, where
    ``applied_shrink`` is the shrink factor `_make_psd_cholesky` had to apply to
    keep the matrix positive-definite (1.0 = none, < 1.0 = correlation partially
    discarded, 0.0 = collapsed to independence). Default returns just ``prob``.
    """
    def _ret(prob, shrink):
        return (prob, shrink) if return_shrink else prob

    n = len(probs)
    if n == 0:
        return _ret(1.0, 1.0)
    if n == 1:
        return _ret(max(0.0, min(1.0, probs[0])), 1.0)
    if any(p <= 0.0 for p in probs):
        return _ret(0.0, 1.0)
    if all(p >= 1.0 for p in probs):
        return _ret(1.0, 1.0)

    thresholds = [_norm_ppf(p) for p in probs]
    L, applied_shrink = _make_psd_cholesky(corr_matrix)
    rng = random.Random(seed)

    hits = 0
    for _ in range(n_samples):
        zi = _box_muller_pairs(rng, n)
        # z = L @ zi  (only need lower-triangular sums)
        ok = True
        for i in range(n):
            zi_i = 0.0
            Li = L[i]
            for k in range(i + 1):
                zi_i += Li[k] * zi[k]
            if zi_i > thresholds[i]:
                ok = False
                break
        if ok:
            hits += 1
    return _ret(hits / n_samples, applied_shrink)


def _normalize_legs(all_ml, all_spreads, all_totals, all_props):
    """
    Convert all analysis results into a uniform leg format for parlay building.
    Only include legs that passed their analyzer's value recommendation filter.
    
    Each leg dict has:
        game_key: str (e.g., "Team A @ Team B" or matchup)
        team: str or None
        bet_type: str (moneyline, spread, total_over, total_under, player_prop_over, player_prop_under)
        label: str (human readable description)
        player: str or None
        prop_key: str or None (e.g., "player_points")
        edge_pct: float
        odds_price: int or None (American odds)
        hist_prob: float (0-1, historical probability)
        implied_prob: float (0-1, book implied probability)
    """
    legs = []
    
    for c in all_ml:
        if not c.get("is_value") or c["edge_pct"] <= 0:
            continue
        game_key = f"{c['opponent']} @ {c['team']}" if c["home_away"] == "HOME" else f"{c['team']} @ {c['opponent']}"
        legs.append({
            "game_key": game_key,
            "team": c["team"],
            "bet_type": "moneyline",
            "label": f"{c['team']} ML ({c['home_away']})",
            "player": None,
            "prop_key": None,
            "edge_pct": c.get("best_edge_pct", c["edge_pct"]),
            "odds_price": c.get("best_price"),
            "hist_prob": c.get("blended_prob", c["hist_prob"]) / 100.0,
            "implied_prob": c.get("best_book_implied_prob", c["book_implied_prob"]) / 100.0,
        })
    
    for c in all_spreads:
        # Safe mode rewrites is_value to enforce its confidence threshold.
        if (not c.get("is_value") or c["edge_pct"] <= 0
                or c.get("games_sampled", 0) < 5):
            continue
        game_key = f"{c['opponent']} @ {c['team']}" if c["home_away"] == "HOME" else f"{c['team']} @ {c['opponent']}"
        legs.append({
            "game_key": game_key,
            "team": c["team"],
            "bet_type": "spread",
            "label": f"{c['team']} {c['spread']:+.2f}",
            "player": None,
            "prop_key": None,
            "edge_pct": c["edge_pct"],
            "odds_price": c.get("price"),
            "hist_prob": c["cover_rate"] / 100.0,
            "implied_prob": c.get("implied_prob", 50.0) / 100.0,
        })
    
    for c in all_totals:
        if c.get("is_over_value"):
            legs.append({
                "game_key": c["matchup"],
                "team": None,
                "bet_type": "total_over",
                "label": f"OVER {c['line']} ({c['matchup']})",
                "player": None,
                "prop_key": None,
                "edge_pct": c.get("over_edge_pct", c["over_hit_rate"] - 50.0),
                "odds_price": c.get("over_price"),
                "hist_prob": c["over_hit_rate"] / 100.0,
                "implied_prob": c.get("over_implied", 50.0) / 100.0,
            })
        if c.get("is_under_value"):
            legs.append({
                "game_key": c["matchup"],
                "team": None,
                "bet_type": "total_under",
                "label": f"UNDER {c['line']} ({c['matchup']})",
                "player": None,
                "prop_key": None,
                "edge_pct": c.get("under_edge_pct", (100.0 - c["over_hit_rate"]) - 50.0),
                "odds_price": c.get("under_price"),
                "hist_prob": (100.0 - c["over_hit_rate"]) / 100.0,
                "implied_prob": c.get("under_implied", 50.0) / 100.0,
            })
    
    for c in all_props:
        if (not c.get("is_value") or c.get("no_history")
                or c["edge_pct"] <= 0 or c.get("games_sampled", 0) < 5):
            continue
        direction = c.get("direction", "OVER")
        bt = f"player_prop_{direction.lower()}"
        price = c.get("best_price", c.get("over_price") if direction == "OVER" else c.get("under_price"))

        if c.get("safe_mode"):
            # Safe-mode legs: bet is "{prop} {N}+" and the hist prob is the
            # model probability AT our safe threshold (not the book line).
            label = f"{c['player']} {c['prop_label']} {c['safe_threshold']}+"
            hp = (c.get("model_hit_at_safe", 0.0) or 0.0) / 100.0
            ip = (c.get("safe_alt_implied") or 0.0) / 100.0
            price = c.get("safe_alt_price")
            if price is None or ip <= 0:
                continue
        elif direction == "OVER":
            label = f"{c['player']} {c['prop_label']} {direction} {c['line']}"
            hp = (c["over_rate"] / 100.0) if c.get("over_rate") is not None else 0.5
            ip = (c["over_implied"] / 100.0) if c.get("over_implied") is not None else 0.5
        else:
            label = f"{c['player']} {c['prop_label']} {direction} {c['line']}"
            hp = (1.0 - c["over_rate"] / 100.0) if c.get("over_rate") is not None else 0.5
            ip = (c["under_implied"] / 100.0) if c.get("under_implied") is not None else 0.5

        leg = {
            "game_key": c["matchup"],
            "team": c.get("team"),
            "bet_type": bt,
            "label": label,
            "player": c["player"],
            "prop_key": c.get("prop"),
            "edge_pct": c["edge_pct"],
            "odds_price": price,
            "hist_prob": hp,
            "implied_prob": ip,
        }
        if c.get("safe_mode"):
            # Extra fields used by the "value parlays in safe mode" ranker / UI.
            leg["safe_mode"] = True
            leg["safe_threshold"] = c.get("safe_threshold")
            leg["book_line"] = c.get("line")
            leg["line_gap"] = c.get("line_gap", 0.0)
            leg["model_hit_at_safe"] = c.get("model_hit_at_safe")
            leg["model_hit_at_line"] = c.get("model_hit_at_line")
            # If an alt-line price was fetched for the suggested safe
            # threshold, prefer it over the book-line price for parlay payout
            # calculation. The book-line price was over_price for the standard
            # line, not the threshold we're actually betting.
            if c.get("safe_alt_price") is not None:
                leg["odds_price"] = c["safe_alt_price"]
                leg["safe_alt_line"] = c.get("safe_alt_line")
                leg["expected_roi_pct"] = c.get("expected_roi_pct")
        legs.append(leg)
    
    return legs


def _has_hard_conflict(leg_a, leg_b):
    """
    Check if two legs have a hard conflict (mutually exclusive or contradictory).
    These combos should NEVER appear in a parlay together.
    """
    same_game = leg_a["game_key"] == leg_b["game_key"]
    
    if not same_game:
        return False
    
    ta = leg_a["bet_type"]
    tb = leg_b["bet_type"]
    
    # Opposite moneylines in same game
    if ta == "moneyline" and tb == "moneyline":
        return leg_a["team"] != leg_b["team"]  # different teams = conflict
    
    # Opposite spreads in same game
    if ta == "spread" and tb == "spread":
        return leg_a["team"] != leg_b["team"]
    
    # Over + Under on same game total
    if {ta, tb} == {"total_over", "total_under"}:
        return True
    
    # Same player, same prop, opposite direction
    if "player_prop" in ta and "player_prop" in tb:
        if (leg_a["player"] == leg_b["player"] 
            and leg_a["prop_key"] == leg_b["prop_key"]
            and ta != tb):
            return True
    
    return False


def _pair_correlation(leg_a, leg_b, sport_key):
    """
    Estimate the rank correlation ρ ∈ [-0.5, 0.5] between two leg outcomes
    (each treated as a Bernoulli "did it hit").  Feeds the Gaussian copula
    that computes joint parlay hit probability.

    Rules mirror the heuristic synergies/conflicts in `_correlation_penalty`
    but in calibrated correlation units rather than arbitrary scores.
    """
    same_game = leg_a["game_key"] == leg_b["game_key"]
    ta = leg_a["bet_type"]
    tb = leg_b["bet_type"]

    # Cross-game legs: treat as effectively independent.
    if not same_game:
        return 0.0

    if sport_key == "basketball_nba":
        if ("player_prop_over" in ta and "player_prop_over" in tb
                and leg_a.get("team") and leg_a.get("team") == leg_b.get("team")):
            return -0.20  # shared possession / usage cap
        if (ta == "total_under" and "player_prop_over" in tb and
                leg_b.get("prop_key") == "player_points"):
            return -0.40
        if (tb == "total_under" and "player_prop_over" in ta and
                leg_a.get("prop_key") == "player_points"):
            return -0.40
        if (ta == "total_over" and "player_prop_over" in tb and
                leg_b.get("prop_key") == "player_points"):
            return 0.30
        if (tb == "total_over" and "player_prop_over" in ta and
                leg_a.get("prop_key") == "player_points"):
            return 0.30
        if (ta == "moneyline" and "player_prop_over" in tb
                and leg_a.get("team") == leg_b.get("team")):
            return 0.15
        if (tb == "moneyline" and "player_prop_over" in ta
                and leg_b.get("team") == leg_a.get("team")):
            return 0.15

    elif sport_key == "americanfootball_nfl":
        if (ta == "moneyline" and "player_prop_over" in tb and
                leg_a.get("team") == leg_b.get("team") and
                leg_b.get("prop_key") == "player_pass_yds"):
            return 0.30
        if (tb == "moneyline" and "player_prop_over" in ta and
                leg_b.get("team") == leg_a.get("team") and
                leg_a.get("prop_key") == "player_pass_yds"):
            return 0.30
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_rush_yds"
                and tb == "total_under"):
            return -0.25
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_rush_yds"
                and ta == "total_under"):
            return -0.25
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "player_pass_yds"
                and tb == "total_over"):
            return 0.25
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "player_pass_yds"
                and ta == "total_over"):
            return 0.25
        if ("player_prop_over" in ta and "player_prop_over" in tb
                and leg_a.get("team") and leg_a.get("team") == leg_b.get("team")):
            return -0.10

    elif sport_key == "baseball_mlb":
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts"
                and tb == "total_under"):
            return 0.35
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts"
                and ta == "total_under"):
            return 0.35
        if ("player_prop_over" in ta and leg_a.get("prop_key") == "pitcher_strikeouts"
                and tb == "total_over"):
            return -0.30
        if ("player_prop_over" in tb and leg_b.get("prop_key") == "pitcher_strikeouts"
                and ta == "total_over"):
            return -0.30
        if (ta == "moneyline" and "player_prop_over" in tb and
                leg_a.get("team") == leg_b.get("team") and
                leg_b.get("prop_key") == "batter_hits"):
            return 0.25
        if (tb == "moneyline" and "player_prop_over" in ta and
                leg_b.get("team") == leg_a.get("team") and
                leg_a.get("prop_key") == "batter_hits"):
            return 0.25

    # Generic same-game legs: small positive (shared game conditions).
    return 0.05


def _build_corr_matrix(legs, sport_key):
    """Symmetric correlation matrix with 1.0 on the diagonal."""
    n = len(legs)
    R = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            rho = _pair_correlation(legs[i], legs[j], sport_key)
            R[i][j] = rho
            R[j][i] = rho
    return R


def _copula_joint_hit_prob(legs, sport_key, n_samples=5000, seed=42,
                           return_shrink=False):
    """Joint P(all legs hit) under a Gaussian copula keyed to `sport_key`.

    With ``return_shrink=True`` returns ``(prob, applied_shrink)`` so the caller
    can surface how much leg correlation the copula had to discard to keep the
    matrix positive-definite (see `_make_psd_cholesky`)."""
    probs = [max(0.0, min(1.0, leg["hist_prob"])) for leg in legs]
    R = _build_corr_matrix(legs, sport_key)
    return _gaussian_copula_joint_prob(probs, R, n_samples=n_samples, seed=seed,
                                       return_shrink=return_shrink)


def _correlation_penalty(leg_a, leg_b, sport_key):
    """
    Cheap enumeration score derived from the same correlation used by the
    Gaussian copula. Keeping one rule source prevents candidate selection and
    final probability ranking from assigning opposite signs to the same pair.
    """
    if leg_a["game_key"] != leg_b["game_key"]:
        return 0.5
    return 5.0 * _pair_correlation(leg_a, leg_b, sport_key)


def _same_team_prop_count(legs):
    """Count player-prop overs sharing an identified team in the same game."""
    team_prop_counts = {}
    for leg in legs:
        if "player_prop_over" in leg["bet_type"] and leg.get("team"):
            key = (leg["game_key"], leg["team"])
            team_prop_counts[key] = team_prop_counts.get(key, 0) + 1
    return max(team_prop_counts.values()) if team_prop_counts else 0


def _score_parlay(legs, sport_key, mode="value"):
    """
    Score a parlay combination. Higher is better.
    
    Modes:
        value: Prioritizes edge (higher edge = better)
        safe: Prioritizes probability of hitting (higher hist_prob = better)
    """
    # Pairwise correlation scoring
    correlation_score = 0.0
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            correlation_score += _correlation_penalty(legs[i], legs[j], sport_key)
    
    # Penalty for too many same-game player prop overs (usage cap)
    same_team_count = _same_team_prop_count(legs)
    usage_penalty = 0
    if same_team_count > 2:
        usage_penalty = -5.0 * (same_team_count - 2)
    
    # Combined probabilities
    combined_hist = 1.0
    combined_implied = 1.0
    for leg in legs:
        combined_hist *= leg["hist_prob"]
        combined_implied *= leg["implied_prob"]
    
    if mode == "safe":
        # Prioritize highest combined probability of hitting
        # Scale hist_prob heavily so it dominates the score
        prob_score = combined_hist * 1000
        # Still consider edge but weighted much less
        total_edge = sum(leg["edge_pct"] for leg in legs) * 0.1
        return prob_score + total_edge + correlation_score + usage_penalty
    elif mode == "safe_value":
        # "Value parlays" built from safe-mode candidates.
        # Every alt leg has an exact fetched price, so maximize estimated return
        # rather than payout alone. A long price is not value unless the modeled
        # probability is high enough to compensate for it.
        payout_product = 1.0
        for leg in legs:
            price = leg.get("odds_price")
            payout_product *= american_to_decimal(price) if price is not None else 1.91
        expected_roi = combined_hist * payout_product - 1.0
        return (expected_roi * 100) + correlation_score + usage_penalty
    else:
        # Prioritize edge value
        total_edge = sum(leg["edge_pct"] for leg in legs)
        parlay_edge = (combined_hist - combined_implied) * 100
        return total_edge + correlation_score + usage_penalty + parlay_edge


def generate_parlays(all_ml, all_spreads, all_totals, all_props, sport_key, mode="value"):
    """
    Generate the top recommended 3, 4, and 5 leg parlays.
    
    Parameters:
        all_ml: Moneyline analysis results
        all_spreads: Spread analysis results
        all_totals: Totals analysis results
        all_props: Player prop analysis results
        sport_key: Sport key (e.g., 'basketball_nba')
        mode: 'value' (prioritize edge) or 'safe' (prioritize hit probability)
    
    Returns:
        dict: {3: parlay_dict, 4: parlay_dict, 5: parlay_dict}
    """
    from itertools import combinations
    
    legs = _normalize_legs(all_ml, all_spreads, all_totals, all_props)
    
    if len(legs) < 3:
        return {}

    # If the underlying analysis was run in safe mode, use a dedicated value
    # ranker. Safe legs are price-verified before they reach this function, so
    # it can maximize expected return at the exact alternate-line prices.
    has_safe_legs = any(leg.get("safe_mode") for leg in legs)
    effective_mode = mode
    if mode == "value" and has_safe_legs:
        effective_mode = "safe_value"

    # Sort and take top candidates to limit combinatorics
    if effective_mode == "safe":
        legs.sort(key=lambda x: x["hist_prob"], reverse=True)
    elif effective_mode == "safe_value":
        # Prefer legs with the strongest single-bet expected return at their
        # exact fetched alt-line price.
        legs.sort(
            key=lambda x: (
                x["hist_prob"] * american_to_decimal(x["odds_price"]) - 1.0
                if x.get("odds_price") is not None else float("-inf")
            ),
            reverse=True,
        )
    else:
        legs.sort(key=lambda x: x["edge_pct"], reverse=True)
    candidates = legs[:25]  # Cap at 25 to keep combos manageable
    
    results = {}

    # Re-rank the top-K heuristic candidates per size with the exact Gaussian
    # copula joint hit probability. Cap K low enough to keep MC cost bounded.
    RERANK_TOP_K = 30
    MC_SAMPLES = 5000

    for size in [3, 4, 5]:
        if len(candidates) < size:
            continue

        # ── Stage 1: heuristic enumeration → keep top-K candidates ──
        top_heap = []  # (heuristic_score, tiebreak, combo_list)
        tiebreak = 0
        for combo in combinations(candidates, size):
            combo_list = list(combo)

            has_conflict = False
            for i in range(len(combo_list)):
                for j in range(i + 1, len(combo_list)):
                    if _has_hard_conflict(combo_list[i], combo_list[j]):
                        has_conflict = True
                        break
                if has_conflict:
                    break
            if has_conflict:
                continue

            # Value / safe_value parlays are gated on EV computed from the real
            # leg prices; never admit a leg with no executable price, whose
            # payout would otherwise fall back to a fabricated -110 / 1.91 and
            # could clear the EV gate on a price the book never offered. 'safe'
            # mode ranks on hit probability, so a priceless leg there is only
            # informational (and still flagged via payout_uses_default_price).
            if effective_mode in ("value", "safe_value") and any(
                    leg.get("odds_price") is None for leg in combo_list):
                continue

            h_score = _score_parlay(combo_list, sport_key, effective_mode)
            tiebreak += 1
            if len(top_heap) < RERANK_TOP_K:
                heapq.heappush(top_heap, (h_score, tiebreak, combo_list))
            elif h_score > top_heap[0][0]:
                heapq.heapreplace(top_heap, (h_score, tiebreak, combo_list))

        if not top_heap:
            continue

        # ── Stage 2: re-rank survivors with the Gaussian copula joint prob ──
        best_parlay = None
        best_score = float("-inf")
        best_joint = 0.0
        best_combined_implied = 0.0
        best_combined_edge = 0.0
        best_corr_shrink = 1.0

        for _, _, combo_list in top_heap:
            combined_implied = 1.0
            combined_edge = 0.0
            for leg in combo_list:
                combined_implied *= leg["implied_prob"]
                combined_edge += leg["edge_pct"]

            joint, corr_shrink = _copula_joint_hit_prob(
                combo_list, sport_key,
                n_samples=MC_SAMPLES,
                seed=42 + size,
                return_shrink=True,
            )

            if effective_mode == "safe":
                score = joint * 1000 + combined_edge * 0.1
            elif effective_mode == "safe_value":
                # Re-rank by correlation-adjusted expected return at the exact
                # fetched price for every leg.
                payout_product = 1.0
                for leg in combo_list:
                    price = leg.get("odds_price")
                    payout_product *= american_to_decimal(price) if price is not None else 1.91
                score = (joint * payout_product - 1.0) * 100
            else:
                score = combined_edge + (joint - combined_implied) * 100

            if score > best_score:
                best_score = score
                best_parlay = combo_list
                best_joint = joint
                best_combined_implied = combined_implied
                best_combined_edge = combined_edge
                best_corr_shrink = corr_shrink

        if best_parlay:
            combined_hist_indep = 1.0
            for leg in best_parlay:
                combined_hist_indep *= leg["hist_prob"]

            # Safe-mode legs now carry implied probability from the exact alt
            # price, so these comparisons are at the same line for every leg.
            combined_edge_out = round(best_combined_edge, 2)
            parlay_edge_out = round((best_joint - best_combined_implied) * 100, 2)

            # Gap stats for the safe_value display (avg/total line gap across
            # safe legs only).
            safe_gaps = [leg.get("line_gap", 0.0)
                         for leg in best_parlay if leg.get("safe_mode")]
            total_line_gap = round(sum(safe_gaps), 2) if safe_gaps else None
            avg_line_gap = (round(sum(safe_gaps) / len(safe_gaps), 2)
                            if safe_gaps else None)

            # Parlay payout: product of each leg's decimal odds. Legs missing a
            # price (rare — primarily older totals/spreads from cached data)
            # are treated as -110 (decimal 1.909), which is the standard US
            # spread/total price. `payout_uses_default` flags when this fallback
            # was applied so the UI can warn.
            decimal_product = 1.0
            payout_uses_default = False
            for leg in best_parlay:
                price = leg.get("odds_price")
                if price is None:
                    decimal_product *= american_to_decimal(-110)
                    payout_uses_default = True
                else:
                    decimal_product *= american_to_decimal(price)
            parlay_decimal = round(decimal_product, 3)
            parlay_american = _decimal_to_american(decimal_product)
            payout_per_10 = round((decimal_product - 1.0) * 10, 2)

            # Same-game parlay: 2+ legs in the same matchup. DK (and other
            # books) apply proprietary correlation adjustments to SGPs, so the
            # book's real SGP payout is materially lower than `decimal_product`
            # (the product of *independent* leg prices).
            game_keys = [leg.get("game_key") for leg in best_parlay]
            has_sgp = len(game_keys) != len(set(game_keys))

            # Expected ROI. For a cross-game parlay the legs are independent to
            # the book, so the naive product payout is real and we credit the
            # copula joint probability (`best_joint`). For a same-game parlay
            # the book prices the leg correlation *out of the payout*; crediting
            # the copula's correlation benefit (best_joint > independent joint)
            # against that naive payout double-counts correlation and can flag a
            # -EV SGP as value. We neutralize it by pricing the SGP against the
            # INDEPENDENT joint probability — this cancels the correlation term
            # the book removes (best_joint * D_sgp ≈ combined_hist_indep *
            # decimal_product), so a positively-correlated SGP can no longer
            # claim an edge the book will not pay.
            payout_joint = _parlay_value_joint(
                best_joint, combined_hist_indep, has_sgp)
            expected_roi_pct = round((payout_joint * decimal_product - 1.0) * 100, 2)
            # Naive figure (copula joint × independent payout) kept for display
            # transparency; not used to gate value.
            expected_roi_pct_naive = round(
                (best_joint * decimal_product - 1.0) * 100, 2)

            # A Value Parlay must still be value after correlation and the
            # actual leg prices are combined (SGP payout neutralized above).
            if effective_mode in ("value", "safe_value") and expected_roi_pct <= 0:
                continue

            results[size] = {
                "legs": best_parlay,
                "score": best_score,
                "mode": effective_mode,
                "combined_edge": combined_edge_out,
                "combined_hist_prob": round(best_joint * 100, 2),
                "combined_hist_prob_indep": round(combined_hist_indep * 100, 2),
                "combined_implied_prob": round(best_combined_implied * 100, 2),
                "parlay_edge_pct": parlay_edge_out,
                "expected_roi_pct": expected_roi_pct,
                "correlation_adjustment_pct": round(
                    (best_joint - combined_hist_indep) * 100, 2
                ),
                "total_line_gap": total_line_gap,
                "avg_line_gap": avg_line_gap,
                # ── Book payout (computed client-side from leg prices) ──
                "parlay_decimal_odds": parlay_decimal,
                "parlay_american_odds": parlay_american,
                "payout_per_10": payout_per_10,
                "payout_uses_default_price": payout_uses_default,
                "has_sgp": has_sgp,
                "expected_roi_pct_naive": expected_roi_pct_naive,
                # Correlation-shrink transparency: how much leg correlation the
                # Gaussian copula had to discard to keep its matrix PSD. 1.0 =
                # none; < 1.0 = partially discarded; near 0.0 = collapsed to
                # independence. Previously this degradation was silent and looked
                # identical to a legitimately uncorrelated parlay.
                "correlation_shrink": round(best_corr_shrink, 3),
                "correlation_degraded": best_corr_shrink < 1.0,
            }

    return results
