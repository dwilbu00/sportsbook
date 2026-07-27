"""
Book-line calibration analysis.

Joins cached sportsbook odds snapshots (`cache/*.json`) to ESPN player
gamelogs, finds the actual stat the player produced in each game, and
evaluates three probabilistic forecasters at the real book line:

    A) empirical    — weighted fraction of prior games > line (current method)
    B) resid_normal — bias-correct projection, then Φ((projected+μ_r − L)/σ_r)
    C) resid_ecdf   — bias-correct projection, then 1 − F_r(L − (projected+μ_r))

Residual stats (μ_r, σ_r, F_r) are fit from a chronological HOLDOUT split
to avoid leakage: the earliest 50% of observations build calibration, the
later 50% are scored.

Usage:
    python book_line_calibration.py --sport nba
    python book_line_calibration.py --sport mlb --variants recency,all
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

from analysis import (
    _norm_cdf, _half_life_for, _recency_weights, _weighted_rate,
    _weighted_std, _normal_inv_cdf,
)
from backtest import (
    cached_gamelog, cached_athlete_id,
    SPORT_MAP, VARIANT_PRESETS, _empirical_cdf, _brier, _logloss, _hit_rate,
    _resolve_params, opp_defense_mult, venue_mult,
    _team_defense_lookup, _resolve_opp_pts_allowed,
    _per_player_stats, _shrunk,
)
from espn_client import PROP_STAT_MAP, ip_to_outs


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ODDS_CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")


PROPS_BY_SPORT = {
    "nba":  ["player_points", "player_rebounds", "player_assists"],
    "mlb":  ["pitcher_strikeouts", "batter_hits"],
    "nfl":  ["player_pass_yds", "player_rush_yds"],
}


# ──────────────────────────────────────────────────────────────────────────────
#  Step 1: harvest book lines from cached odds snapshots
# ──────────────────────────────────────────────────────────────────────────────

def _is_prop_market(market_key):
    return market_key.startswith(("player_", "pitcher_", "batter_"))


def harvest_book_lines(sport_key, target_props):
    """
    Walk cache/*.json and return a list of dicts:
      {sport_key, commence_time, game_date, home_team, away_team,
       player, prop_key, line, over_price, under_price}
    Picks the consensus (median) line across books per (player, prop, game).
    """
    raw = []  # (game_date, sport, home, away, player, prop, side, line, price)
    for path in sorted(glob.glob(os.path.join(ODDS_CACHE_DIR, "*.json"))):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        data = doc.get("data", doc) if isinstance(doc, dict) else doc
        if not isinstance(data, dict):
            continue
        if data.get("sport_key") != sport_key:
            continue
        ct = data.get("commence_time")
        if not ct:
            continue
        date = ct[:10]
        home = data.get("home_team")
        away = data.get("away_team")
        for book in data.get("bookmakers", []) or []:
            for mkt in book.get("markets", []) or []:
                mk = mkt.get("key", "")
                if mk not in target_props:
                    continue
                for o in mkt.get("outcomes", []) or []:
                    player = o.get("description")
                    side = o.get("name")  # "Over" / "Under"
                    point = o.get("point")
                    price = o.get("price")
                    if not (player and side and point is not None):
                        continue
                    raw.append((date, sport_key, home, away,
                                player, mk, side, point, price))

    # Consolidate to one row per (player, prop, game): use the consensus line
    # (median across books) and the best price per side.
    by_key = defaultdict(lambda: {"lines": [], "over_prices": [], "under_prices": [],
                                  "home": None, "away": None})
    for date, sport, home, away, player, mk, side, point, price in raw:
        key = (date, sport, player, mk)
        cell = by_key[key]
        cell["home"] = home
        cell["away"] = away
        cell["lines"].append(point)
        if side.lower() == "over" and price is not None:
            cell["over_prices"].append(price)
        elif side.lower() == "under" and price is not None:
            cell["under_prices"].append(price)

    out = []
    for (date, sport, player, mk), cell in by_key.items():
        if not cell["lines"]:
            continue
        # Consensus line = mode/median of book lines (most books agree)
        lines_sorted = sorted(cell["lines"])
        consensus = lines_sorted[len(lines_sorted) // 2]
        out.append({
            "sport_key": sport,
            "game_date": date,
            "home_team": cell["home"],
            "away_team": cell["away"],
            "player": player,
            "prop_key": mk,
            "line": consensus,
            "over_price": max(cell["over_prices"]) if cell["over_prices"] else None,
            "under_price": max(cell["under_prices"]) if cell["under_prices"] else None,
        })
    out.sort(key=lambda r: (r["game_date"], r["player"], r["prop_key"]))
    return out


def harvest_book_lines_from_store(sport_key, target_props, label=""):
    """
    Harvest (player, prop, game) book lines from the DURABLE historical_odds
    store (backfill_historical_odds.py output) instead of the ephemeral
    cache/*.json HTTP cache. Returns the same shape as harvest_book_lines():
      {sport_key, game_date, home_team, away_team, player, prop_key, line,
       over_price, under_price}

    The store already consolidates each (player, prop, game) to one consensus
    line + best executable price per side (see odds_client.parse_player_props),
    so no per-book median is computed here.
    """
    import historical_odds as store_mod
    store = store_mod.load_store(sport_key, label)
    target = set(target_props or [])
    out = []
    for entry in (store.get("games") or {}).values():
        date = (entry.get("commence_time") or "")[:10]
        if not date:
            continue
        home = entry.get("home_team")
        away = entry.get("away_team")
        for prop_key, by_player in (entry.get("props") or {}).items():
            if prop_key not in target:
                continue
            for player, info in (by_player or {}).items():
                line = info.get("line")
                if line is None:
                    continue
                out.append({
                    "sport_key": sport_key,
                    "game_date": date,
                    "home_team": home,
                    "away_team": away,
                    "player": player,
                    "prop_key": prop_key,
                    "line": line,
                    "over_price": info.get("over_price"),
                    "under_price": info.get("under_price"),
                })
    out.sort(key=lambda r: (r["game_date"], r["player"], r["prop_key"]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Step 2: join each book line to the player's actual stat in that game
# ──────────────────────────────────────────────────────────────────────────────

def _stat_label_for(prop_key, gamelog):
    for label in PROP_STAT_MAP.get(prop_key, []):
        if any(label in g for g in gamelog):
            return label
    return None


def join_book_lines_to_actuals(book_lines, espn_sport, espn_league):
    """
    For each book line, resolve the player's athlete_id, pull their gamelog,
    locate the game on `game_date`, and attach `actual` + `prior_games`.
    Returns a list of enriched dicts (skipping unjoinable rows).
    """
    # Group by player so we only fetch each gamelog once
    by_player = defaultdict(list)
    for row in book_lines:
        by_player[row["player"]].append(row)

    enriched = []
    skipped_no_player = 0
    skipped_no_game = 0

    for player, rows in by_player.items():
        aid = cached_athlete_id(espn_sport, espn_league, player)
        if not aid:
            skipped_no_player += len(rows)
            continue
        gamelog = cached_gamelog(espn_sport, espn_league, aid)
        if not gamelog:
            skipped_no_player += len(rows)
            continue
        gamelog.sort(key=lambda g: g.get("game_date") or "", reverse=True)

        # Build a date → game-index lookup
        date_idx = {}
        for i, g in enumerate(gamelog):
            d = (g.get("game_date") or "")[:10]
            if d and d not in date_idx:
                date_idx[d] = i

        for row in rows:
            stat_label = _stat_label_for(row["prop_key"], gamelog)
            if not stat_label:
                continue
            # ESPN game_date is UTC-ish; the book commence_time is also UTC,
            # so date-only match is usually accurate. Also try ±1 day in case
            # of timezone slippage.
            d = row["game_date"]
            idx = date_idx.get(d)
            if idx is None:
                from datetime import date as _date, timedelta
                try:
                    d0 = _date.fromisoformat(d)
                    for delta in (-1, 1):
                        alt = (d0 + timedelta(days=delta)).isoformat()
                        if alt in date_idx:
                            idx = date_idx[alt]
                            break
                except ValueError:
                    pass
            if idx is None:
                skipped_no_game += 1
                continue

            test_game = gamelog[idx]
            actual = test_game.get(stat_label)
            if actual is None:
                skipped_no_game += 1
                continue
            if stat_label == "IP":       # pitcher_outs: IP notation -> outs
                actual = ip_to_outs(actual)
            min_played = test_game.get("MIN", 0.0) or 0.0
            if min_played and min_played < 10.0:
                skipped_no_game += 1
                continue

            prior_games = gamelog[idx + 1:]
            if len(prior_games) < 10:
                skipped_no_game += 1
                continue

            enriched.append({
                **row,
                "stat_label": stat_label,
                "actual": float(actual),
                "test_game": test_game,
                "prior_games": prior_games,
            })

    print(f"  joined {len(enriched)} (player, prop, game) observations; "
          f"skipped {skipped_no_player} (no player resolved), "
          f"{skipped_no_game} (no game/stat match).")
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
#  Step 3: produce projected stat + empirical_over for each observation
# ──────────────────────────────────────────────────────────────────────────────

def project_and_empirical(obs, params, sport_key,
                          team_defense=None, league_avg_def=None,
                          xstats_strength=0.0, xba_index=None):
    """
    Mirrors backtest.run_player_props_backtest's per-obs projection logic,
    but takes the line from the book instead of synthetic.

    When ``xstats_strength > 0`` and a leakage-safe ``xba_index`` (an
    ``AsOfIndex`` keyed by batter MLBAM id, from
    ``backtest_props.build_batter_xba_index``) is supplied, the reconstructed
    projection is blended toward the batter's OWN as-of xBA × recent AB/game —
    exactly the P2.4a blend props.py applies live — so the re-fit residual
    distribution matches production. Uses a per-GAME as-of estimate (strictly
    before the obs's game_date), never the current-as-of SQL table. Fails open
    (unknown id / <MIN_BBE ABs / no AB data → unblended).

    Returns (projected, empirical_over) or (None, None) if data is too thin.
    """
    prior_games = obs["prior_games"]
    stat_label = obs["stat_label"]
    line = obs["line"]
    test_game = obs["test_game"]

    prior_values = [g.get(stat_label, 0.0) for g in prior_games]
    if stat_label == "IP":               # pitcher_outs: IP notation -> outs
        prior_values = [ip_to_outs(v) for v in prior_values]
    prior_minutes = [g.get("MIN", 0.0) for g in prior_games]
    prior_home_aways = [g.get("is_home") for g in prior_games]
    prior_opponents = [g.get("opponent") for g in prior_games]

    if any(prior_minutes):
        MIN_FLOOR = 10.0
        kept = [(v, m, ha, opp) for v, m, ha, opp in zip(
                    prior_values, prior_minutes,
                    prior_home_aways, prior_opponents)
                if (m or 0) >= MIN_FLOOR]
        if kept:
            prior_values = [v for v, _, _, _ in kept]
            prior_minutes = [m for _, m, _, _ in kept]
            prior_home_aways = [ha for _, _, ha, _ in kept]
            prior_opponents = [opp for _, _, _, opp in kept]

    if not prior_values:
        return None, None

    upcoming_is_home = test_game.get("is_home")

    hl = params.get("half_life")
    base_w = _recency_weights(len(prior_values), hl)
    venue_s = params.get("venue_strength", 0.0)
    def_s = params.get("opp_defense_strength", 0.0)

    weights = []
    for bw, ph, opp in zip(base_w, prior_home_aways, prior_opponents):
        w = bw * venue_mult(ph, upcoming_is_home, venue_s)
        if def_s > 0 and team_defense:
            opp_pa = _resolve_opp_pts_allowed(opp, team_defense)
            w *= opp_defense_mult(opp_pa, league_avg_def, def_s)
        weights.append(w)

    if sum(weights) <= 0:
        return None, None

    if params.get("use_minutes"):
        rates = [v / m for v, m in zip(prior_values, prior_minutes) if m and m > 0]
        rate_weights = [w for w, m in zip(weights, prior_minutes) if m and m > 0]
        if rates and sum(rate_weights) > 0:
            per_min_rate = sum(v * w for v, w in zip(rates, rate_weights)) / sum(rate_weights)
            proj_min = sum(m * w for m, w in zip(prior_minutes, weights)) / sum(weights)
            projected = per_min_rate * proj_min
        else:
            projected = sum(v * w for v, w in zip(prior_values, weights)) / sum(weights)
    else:
        projected = sum(v * w for v, w in zip(prior_values, weights)) / sum(weights)

    # ── Statcast xBA blend (P2.4a, leakage-safe as-of) ──
    # Shrink the projection toward the batter's own as-of xBA × recent AB/game,
    # mirroring props._xstats_blend, so the re-fit residuals match production.
    # Only fires when a strength + as-of index are supplied (batter_hits). Uses a
    # per-game as-of estimate (< this obs's game_date); fails open. Note A
    # (empirical_over) is unaffected — it doesn't read `projected` — so this
    # changes only the B/C-method residual basis.
    if (xstats_strength and xstats_strength > 0 and xba_index is not None
            and len(prior_games) == len(weights)):
        try:
            import mlb_starters
            game_date = obs.get("game_date")
            player = obs.get("player")
            season = int(str(game_date)[:4]) if game_date else None
            pid_info = (mlb_starters.find_player_id(player, season)
                        if (player and season) else None)
            if pid_info and pid_info[0] and not pid_info[1]:   # batter only
                xba = xba_index.asof_mean(str(pid_info[0]), game_date)  # min_bbe=40
                ab_valid = [(g.get("AB"), w) for g, w in zip(prior_games, weights)
                            if g.get("AB") and w > 0]
                if xba is not None and ab_valid:
                    ab_pg = (sum(ab * w for ab, w in ab_valid)
                             / sum(w for _, w in ab_valid))
                    if ab_pg > 0:
                        wgt = max(0.0, min(1.0, xstats_strength))
                        projected = (1.0 - wgt) * projected + wgt * (xba * ab_pg)
        except Exception:
            pass  # fail open — never let the xBA lookup break the refit

    # Empirical over-probability AT THE BOOK LINE
    empirical_over = _weighted_rate(
        prior_values, weights, lambda v: v > line)

    return projected, empirical_over


def project_distributional(obs, params, sport_key, team_defense=None,
                           league_avg_def=None, xba_index=None,
                           quality_index=None, xstats_strength=0.0,
                           hardhit_coef=None, barrel_coef=None,
                           xba_window=None, xba_min_count=None,
                           home_ab_delta=0.0):
    """Distributional P(over) for a batter_hits obs at its REAL book line
    (§2.4b-2 method "D"), or None if not applicable / too thin.

    Mirrors project_and_empirical's recency×venue×opp_defense weighting, derives
    the weighted per-AB hit rate + expected AB from the prior games, looks up the
    batter's leakage-safe as-of xBA / contact-quality (per-game indices, strictly
    before obs["game_date"]), and returns ``props._dist_p_over`` — the SAME
    composite the runtime uses, so the two can't drift. Offline eval applies NO
    output (park/weather/matchup) multipliers — rate_mult = exposure_mult = 1,
    the same limitation the C-method real-line fit has. Fails open."""
    if obs.get("prop_key") != "batter_hits":
        return None
    prior_games = obs["prior_games"]
    if not prior_games:
        return None
    line = obs["line"]
    upcoming_is_home = obs["test_game"].get("is_home")
    prior_hits = [g.get("H") for g in prior_games]
    prior_ab = [g.get("AB") for g in prior_games]
    prior_home_aways = [g.get("is_home") for g in prior_games]
    prior_opponents = [g.get("opponent") for g in prior_games]

    base_w = _recency_weights(len(prior_games), params.get("half_life"))
    venue_s = params.get("venue_strength", 0.0)
    def_s = params.get("opp_defense_strength", 0.0)
    weights = []
    for bw, ph, opp in zip(base_w, prior_home_aways, prior_opponents):
        w = bw * venue_mult(ph, upcoming_is_home, venue_s)
        if def_s > 0 and team_defense:
            opp_pa = _resolve_opp_pts_allowed(opp, team_defense)
            w *= opp_defense_mult(opp_pa, league_avg_def, def_s)
        weights.append(w)

    hits_w = ab_w = w_valid = 0.0
    for h, ab, w in zip(prior_hits, prior_ab, weights):
        if ab is None or ab <= 0 or w <= 0 or h is None:
            continue
        hits_w += w * h
        ab_w += w * ab
        w_valid += w
    if ab_w <= 0 or w_valid <= 0:
        return None
    r_emp = hits_w / ab_w
    expected_ab = ab_w / w_valid
    if expected_ab <= 0:
        return None
    # Experiment: home batters get slightly fewer plate appearances (the home
    # team skips the bottom 9th when leading). ``home_ab_delta`` nudges the
    # binomial n for a home game; 0.0 = off. Floored so n stays positive.
    if home_ab_delta and obs.get("test_game", {}).get("is_home"):
        expected_ab = max(0.5, expected_ab + home_ab_delta)

    xba = hh = brl = None
    try:
        import mlb_starters
        game_date = obs.get("game_date")
        player = obs.get("player")
        season = int(str(game_date)[:4]) if game_date else None
        pid_info = (mlb_starters.find_player_id(player, season)
                    if (player and season) else None)
        if pid_info and pid_info[0] and not pid_info[1]:   # batter only
            pid = str(pid_info[0])
            if xba_index is not None:
                if xba_window:            # rolling last-N-BBE xBA vs season-to-date
                    xba = xba_index.asof_window_mean(
                        pid, game_date, xba_window, xba_min_count or 1)
                else:
                    xba = xba_index.asof_mean(pid, game_date)
            if quality_index is not None:
                q = quality_index.asof(pid, game_date)
                if q:
                    hh = q.get("hard_hit_pct")
                    brl = q.get("barrel_pct")
    except Exception:
        xba = hh = brl = None   # fail open

    import props
    p_over, _ = props._dist_p_over(
        r_emp, expected_ab, xba, hh, brl, 1.0, 1.0, line,
        xstats_strength, hardhit_coef, barrel_coef)
    return p_over


# ──────────────────────────────────────────────────────────────────────────────
#  Step 4: forecaster comparison with chronological holdout
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_calibration(per_prop_obs, prop_key, label, shrinkage_k=15):
    """
    per_prop_obs: list of dicts with keys:
      player, projected, line, actual, empirical_over, game_date

    Returns metrics for five forecasters: A (empirical), B (pooled Gaussian),
    C (pooled ECDF), B* (per-player Gaussian + shrinkage), C* (per-player
    ECDF + shrinkage).
    """
    rows = [r for r in per_prop_obs if r["actual"] != r["line"]]
    if len(rows) < 20:
        return None

    rows.sort(key=lambda r: r["game_date"])
    split = len(rows) // 2
    train = rows[:split]
    test = rows[split:]

    # ── Pool stats (train only) ──
    train_resid = [r["actual"] - r["projected"] for r in train]
    mu_pool = sum(train_resid) / len(train_resid)
    var_pool = sum((r - mu_pool) ** 2 for r in train_resid) / len(train_resid)
    sigma_pool = math.sqrt(var_pool) if var_pool > 0 else 1e-6
    sorted_pool = sorted(train_resid)

    # ── Per-player stats (train only) ──
    player_resid = {}
    for r in train:
        player_resid.setdefault(r["player"], []).append(r["actual"] - r["projected"])
    player_stats = _per_player_stats(player_resid)
    player_sorted = {p: sorted(rs) for p, rs in player_resid.items()}

    pA, pB, pC, pBs, pCs, outcomes = [], [], [], [], [], []
    for r in test:
        o = 1 if r["actual"] > r["line"] else 0
        outcomes.append(o)
        pA.append(max(0.0, min(1.0, r["empirical_over"])))

        # B (pooled Gaussian)
        corrected_pool = r["projected"] + mu_pool
        z = (corrected_pool - r["line"]) / sigma_pool if sigma_pool > 0 else 0.0
        pB.append(_norm_cdf(z))

        # C (pooled ECDF)
        pC.append(1.0 - _empirical_cdf(sorted_pool, r["line"] - corrected_pool))

        # B* per-player Gaussian + shrinkage
        if r["player"] in player_stats:
            n_p, mu_p, sigma_p = player_stats[r["player"]]
            mu_s = _shrunk(mu_p, mu_pool, n_p, shrinkage_k)
            sigma_s = _shrunk(sigma_p, sigma_pool, n_p, shrinkage_k)
        else:
            mu_s, sigma_s = mu_pool, sigma_pool
        corrected_s = r["projected"] + mu_s
        z_s = (corrected_s - r["line"]) / sigma_s if sigma_s > 0 else 0.0
        pBs.append(_norm_cdf(z_s))

        # C* per-player ECDF blended with pool by λ
        if r["player"] in player_sorted:
            n_p = len(player_sorted[r["player"]])
            lam = n_p / (n_p + shrinkage_k)
            f_player = _empirical_cdf(player_sorted[r["player"]],
                                      r["line"] - corrected_s)
            f_pool = _empirical_cdf(sorted_pool, r["line"] - corrected_s)
            f_blend = lam * f_player + (1 - lam) * f_pool
        else:
            f_blend = _empirical_cdf(sorted_pool, r["line"] - corrected_s)
        pCs.append(1.0 - f_blend)

    return {
        "n_test": len(test),
        "n_train": len(train),
        "mu_r": mu_pool,
        "sigma_r": sigma_pool,
        "n_players": len(player_stats),
        "brier": (_brier(pA, outcomes), _brier(pB, outcomes), _brier(pBs, outcomes),
                  _brier(pC, outcomes), _brier(pCs, outcomes)),
        "hit":   (_hit_rate(pA, outcomes), _hit_rate(pB, outcomes), _hit_rate(pBs, outcomes),
                  _hit_rate(pC, outcomes), _hit_rate(pCs, outcomes)),
        "probs": {"A": pA, "B": pB, "B*": pBs, "C": pC, "C*": pCs},
        "outcomes": outcomes,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Step 5: re-select the deployed calibration METHOD at REAL book lines
#          (roadmap 0.3 — the synthetic-vs-real-line fix)
#
#  refit_calibration.py chooses the A/B/C method by Brier at a SYNTHETIC
#  season-average line, but props.py applies it at the REAL book line. Residual
#  mu/sigma/ecdf are line-invariant, so the only thing that must be re-decided at
#  real lines is the A-vs-B/C method choice (method A's empirical over-rate is
#  already recomputed at the real line at runtime). These helpers mirror the gate
#  in refit_calibration._best_per_prop / _confirms_over_baseline but score every
#  method at the real book line carried on each row.
# ──────────────────────────────────────────────────────────────────────────────

def build_real_line_obs(enriched, params, sport_key, prop_key,
                        team_defense=None, league_avg_def=None,
                        xstats_strength=0.0, xba_index=None):
    """Project every joined observation for one prop at its REAL book line.

    Returns a list of row dicts {player, projected, line, actual,
    empirical_over, game_date}, reusing project_and_empirical (which evaluates
    the empirical over-rate at obs["line"], the book line). ``xstats_strength`` +
    ``xba_index`` opt into the P2.4a xBA projection blend (batter_hits).
    """
    rows = []
    for obs in enriched:
        if obs["prop_key"] != prop_key:
            continue
        projected, emp = project_and_empirical(
            obs, params, sport_key, team_defense, league_avg_def,
            xstats_strength=xstats_strength, xba_index=xba_index)
        if projected is None:
            continue
        rows.append({
            "player": obs["player"],
            "projected": projected,
            "line": obs["line"],
            "actual": obs["actual"],
            "empirical_over": emp,
            "game_date": obs["game_date"],
        })
    return rows


def _score_abc_real(train, test):
    """Fit pooled residuals on `train`, score methods A/B/C (and D when present)
    on `test` at each row's REAL book line. Returns ({method: brier}, (mu, sigma,
    sorted)).

    A = empirical over-rate passthrough; B = pooled Gaussian residual;
    C = pooled residual ECDF. D (distributional, §2.4b-2) is scored only when
    every test row carries a precomputed leakage-safe ``p_dist`` — D needs no
    train fit (its prob is a closed form on each obs's own as-of stats), so a
    row's ``p_dist`` is split-independent. Same math as evaluate_calibration,
    factored so the single split and the confirmation folds share one impl.
    """
    resid = [r["actual"] - r["projected"] for r in train]
    mu = sum(resid) / len(resid)
    var = sum((x - mu) ** 2 for x in resid) / len(resid)
    sigma = math.sqrt(var) if var > 0 else 1e-6
    srt = sorted(resid)
    pA, pB, pC, pD, out = [], [], [], [], []
    has_d = bool(test) and all(r.get("p_dist") is not None for r in test)
    for r in test:
        out.append(1 if r["actual"] > r["line"] else 0)
        pA.append(max(0.0, min(1.0, r["empirical_over"])))
        corrected = r["projected"] + mu
        z = (corrected - r["line"]) / sigma if sigma > 0 else 0.0
        pB.append(_norm_cdf(z))
        pC.append(1.0 - _empirical_cdf(srt, r["line"] - corrected))
        if has_d:
            pD.append(r["p_dist"])
    scores = {"A": _brier(pA, out), "B": _brier(pB, out), "C": _brier(pC, out)}
    if has_d:
        scores["D"] = _brier(pD, out)
    return scores, (mu, sigma, srt)


def _real_line_folds(rows, min_set_n=20):
    """Two expanding-train chronological folds over real-line rows (dict shape).

    Mirrors refit_calibration._chronological_folds: cut at 60%/80% of the
    date-sorted rows, each fold with strictly earlier train than test and
    >= min_set_n in both sets. Returns [] when the data can't form two folds.
    """
    rows = sorted(rows, key=lambda r: r["game_date"])
    n = len(rows)
    if n < 3 * min_set_n:
        return []
    cut1 = rows[int(n * 0.6)]["game_date"]
    cut2 = rows[int(n * 0.8)]["game_date"]
    if not cut1 or not cut2 or cut1 == cut2:
        return []
    folds = [
        ([r for r in rows if r["game_date"] < cut1],
         [r for r in rows if cut1 <= r["game_date"] < cut2]),
        ([r for r in rows if r["game_date"] < cut2],
         [r for r in rows if r["game_date"] >= cut2]),
    ]
    for fit_rows, score_rows in folds:
        if len(fit_rows) < min_set_n or len(score_rows) < min_set_n:
            return []
    return folds


def select_method_at_real_lines(rows, shrinkage_k=15):
    """Choose the deployed A/B/C method at REAL book lines, gated like refit.

    Method A (empirical) is the safe baseline and always eligible. A non-empirical
    method (B/C) is chosen only if it beats A on the single chronological holdout
    by >= MIN_CALIB_BRIER_GAIN AND beats A in BOTH expanding confirmation folds
    (defeats winner's-curse). Residuals for the winner are fit on ALL usable obs
    (the deployed distribution), matching refit_calibration._fit_residuals.

    Returns None if fewer than 20 usable (actual != line) observations, else:
      {method, fit_brier, baseline_brier, cv_brier, confirmed,
       residual_mu, residual_sigma, residual_ecdf, n_obs}
    """
    from refit_calibration import MIN_CALIB_BRIER_GAIN  # single source of truth

    usable = [r for r in rows if r["actual"] != r["line"]]
    if len(usable) < 20:
        return None
    usable.sort(key=lambda r: r["game_date"])

    # Single chronological holdout: baseline (A) + candidate (B/C) Briers.
    split = len(usable) // 2
    single, _ = _score_abc_real(usable[:split], usable[split:])
    baseline = single["A"]

    # Two-fold out-of-sample confirmation for non-empirical methods.
    folds = _real_line_folds(usable)
    fold_scores = [_score_abc_real(tr, te)[0] for tr, te in folds] if folds else []

    def _confirms(method):
        if not fold_scores:
            return False
        return all(fs.get("A") is not None and fs.get(method) is not None
                   and fs[method] < fs["A"] for fs in fold_scores)

    # D (distributional) is a candidate only when the rows carry a leakage-safe
    # p_dist (the per-bucket line-conditional path supplies it); the pooled A/B/C
    # callers pass rows without it, so their behavior is unchanged.
    candidate_methods = ["B", "C"]
    if any(r.get("p_dist") is not None for r in usable):
        candidate_methods.append("D")
    best_method, best_brier = "A", baseline
    for method in candidate_methods:
        cand = single.get(method)
        if cand is None or baseline is None:
            continue
        if baseline - cand < MIN_CALIB_BRIER_GAIN:
            continue
        if not _confirms(method):
            continue
        if best_brier is None or cand < best_brier:
            best_method, best_brier = method, cand

    # Deployed residual distribution: fit on ALL usable obs.
    resid = [r["actual"] - r["projected"] for r in usable]
    mu = sum(resid) / len(resid)
    var = sum((x - mu) ** 2 for x in resid) / len(resid)
    sigma = math.sqrt(var) if var > 0 else 0.0

    cv_brier = None
    if fold_scores:
        vals = [fs.get(best_method) for fs in fold_scores]
        if all(v is not None for v in vals):
            cv_brier = round(sum(vals) / len(vals), 4)

    return {
        "method": best_method,
        "fit_brier": round(best_brier, 4) if best_brier is not None else None,
        "baseline_brier": round(baseline, 4) if baseline is not None else None,
        "cv_brier": cv_brier,
        "confirmed": best_method != "A",
        "residual_mu": mu,
        "residual_sigma": sigma,
        "residual_ecdf": sorted(resid),
        "n_obs": len(usable),
        # Per-method Brier on the single holdout — lets a per-bucket caller
        # compare its winner against the POOLED method on the same split.
        "single_split": single,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Confidence-threshold "precision vs coverage" report
# ──────────────────────────────────────────────────────────────────────────────

def confidence_threshold_table(probs, outcomes, thresholds=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)):
    """
    For each threshold T, simulate "only bet when max(p, 1-p) >= T".
    Picked side is OVER if p >= 0.5 else UNDER. Hit when picked side matches
    the actual outcome (outcomes[i] == 1 means actual > line).
    Returns list of (T, n_actionable, n_hits, hit_rate, pct_of_total).
    """
    total = len(outcomes)
    rows = []
    for T in thresholds:
        n = 0
        hits = 0
        for p, o in zip(probs, outcomes):
            conf = max(p, 1.0 - p)
            if conf < T:
                continue
            n += 1
            pick = 1 if p >= 0.5 else 0
            if pick == o:
                hits += 1
        rate = (hits / n) if n else 0.0
        cov = (n / total) if total else 0.0
        rows.append((T, n, hits, rate, cov))
    return rows


def print_confidence_report(per_variant_prop_results):
    """
    per_variant_prop_results: dict (variant_name, prop_key) -> evaluate_calibration result
    """
    print("\n=== Confidence-threshold hit rates (per forecaster) ===")
    print("'If we only bet when model confidence (max(p,1-p)) >= T, what % of the time")
    print(" did the chosen side win?'  Holdout-only (no leakage).\n")

    forecasters = ["A", "B", "B*", "C", "C*"]
    header = (f"{'Variant':<10} {'Prop':<22} {'Fcst':<5} "
              f"{'T':>5} {'N':>4} {'Hits':>5} {'Hit%':>6} {'Cov%':>6}")
    print(header)
    print("-" * len(header))

    for (vname, prop_key), res in per_variant_prop_results.items():
        outcomes = res["outcomes"]
        for fcst in forecasters:
            probs = res["probs"][fcst]
            rows = confidence_threshold_table(probs, outcomes)
            for T, n, hits, rate, cov in rows:
                if n == 0:
                    continue
                print(f"{vname:<10} {prop_key:<22} {fcst:<5} "
                      f"{T:>5.2f} {n:>4d} {hits:>5d} "
                      f"{rate * 100:>5.1f}% {cov * 100:>5.1f}%")
            print()


def simulate_safe_mode(obs, params, sport_key, safe_target,
                       team_defense=None, league_avg_def=None):
    """
    Replicate analysis.py's safe-mode pipeline for one (player, prop, game)
    observation and report what the production code would have done.

    Returns dict:
      {
        'eligible':       bool,   # passed both production guards
        'filter_reason':  str|None,
        'safe_threshold': int|None,
        'p_at_safe':      float|None,    # model-claimed hit prob
        'hit':            bool|None,     # actual >= safe_threshold
        'line':           float,         # book line (reference)
      }
    """
    prior_games = obs["prior_games"]
    stat_label = obs["stat_label"]
    line = obs["line"]
    test_game = obs["test_game"]
    actual = obs["actual"]

    prior_values = [g.get(stat_label, 0.0) for g in prior_games]
    if stat_label == "IP":               # pitcher_outs: IP notation -> outs
        prior_values = [ip_to_outs(v) for v in prior_values]
    prior_minutes = [g.get("MIN", 0.0) for g in prior_games]
    prior_home_aways = [g.get("is_home") for g in prior_games]
    prior_opponents = [g.get("opponent") for g in prior_games]

    # Same DNP-floor filter as production safe-mode prep
    if any(prior_minutes):
        kept = [(v, m, ha, opp) for v, m, ha, opp in zip(
                    prior_values, prior_minutes,
                    prior_home_aways, prior_opponents)
                if (m or 0) >= 10.0]
        if kept:
            prior_values = [v for v, _, _, _ in kept]
            prior_minutes = [m for _, m, _, _ in kept]
            prior_home_aways = [ha for _, _, ha, _ in kept]
            prior_opponents = [opp for _, _, _, opp in kept]

    if not prior_values:
        return {"eligible": False, "filter_reason": "no_prior", "safe_threshold": None,
                "p_at_safe": None, "hit": None, "line": line}

    upcoming_is_home = test_game.get("is_home")

    hl = params.get("half_life")
    base_w = _recency_weights(len(prior_values), hl)
    venue_s = params.get("venue_strength", 0.0)
    def_s = params.get("opp_defense_strength", 0.0)
    output_def_s = params.get("output_def_strength", 0.0)
    shrinkage_k = params.get("shrinkage_k", 0) or 0

    weights = []
    for bw, ph, opp in zip(base_w, prior_home_aways, prior_opponents):
        w = bw * venue_mult(ph, upcoming_is_home, venue_s)
        if def_s > 0 and team_defense:
            opp_pa = _resolve_opp_pts_allowed(opp, team_defense)
            w *= opp_defense_mult(opp_pa, league_avg_def, def_s)
        weights.append(w)

    if sum(weights) <= 0:
        return {"eligible": False, "filter_reason": "zero_weights", "safe_threshold": None,
                "p_at_safe": None, "hit": None, "line": line}

    # Output-defense multiplier (uses test_game's opponent)
    output_def_mult = 1.0
    opp_name = test_game.get("opponent")
    if output_def_s > 0 and team_defense and league_avg_def and opp_name:
        opp_pa = team_defense.get(opp_name)
        if opp_pa:
            output_def_mult = 1.0 + output_def_s * (
                opp_pa / league_avg_def - 1.0)

    # Bayesian shrinkage toward unweighted mean
    base_proj = sum(v * w for v, w in zip(prior_values, weights)) / sum(weights)
    if shrinkage_k > 0:
        unweighted = sum(prior_values) / len(prior_values)
        eff_n = sum(weights)
        if eff_n + shrinkage_k > 0:
            base_proj = ((eff_n * base_proj) + (shrinkage_k * unweighted)) / (eff_n + shrinkage_k)
    avg_stat = base_proj * output_def_mult

    # Parametric quantile floor (mirrors analysis.py exactly)
    wstd = _weighted_std(prior_values, weights, mean=base_proj)
    wstd_adj = wstd * (output_def_mult if output_def_mult else 1.0)
    z = _normal_inv_cdf(safe_target)
    alt_q = avg_stat - z * wstd_adj
    safe_threshold = max(1, int(math.floor(alt_q)))

    # Production sanity guard (mirrors analysis.py): drop if historical
    # hit rate at the suggested threshold is more than 5pp below target.
    p_at_safe = _weighted_rate(prior_values, weights,
                               lambda v, t=safe_threshold: v >= t)
    if p_at_safe < (safe_target - 0.05):
        return {"eligible": False, "filter_reason": "p_at_safe<target-0.05",
                "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
                "hit": None, "line": line}

    # Floor-collapse guard (mirrors analysis.py): reject "1+" type bets
    # for safe_target > 0.80 — parametric Normal under-estimates variance
    # for low-mean integer distributions and the forward hit rate is
    # systematically below the claimed safe_target.
    if safe_threshold <= 1 and safe_target > 0.80:
        return {"eligible": False, "filter_reason": "floor_collapse",
                "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
                "hit": None, "line": line}

    SAFE_MIN_RATIO = 0.5
    if line > 0 and safe_threshold < line * SAFE_MIN_RATIO:
        return {"eligible": False, "filter_reason": "threshold<50%_of_book",
                "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
                "hit": None, "line": line}

    hit = float(actual) >= safe_threshold
    return {"eligible": True, "filter_reason": None,
            "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
            "hit": hit, "line": line}


def print_safe_mode_report(enriched, params, sport_key,
                           team_defense=None, league_avg_def=None,
                           targets=(0.85, 0.90, 0.95)):
    """
    For each safe_target, simulate production safe-mode on every observation,
    then bucket eligible suggestions per prop and report actual hit rate.
    """
    print("\n=== Safe-mode actual hit rates (per prop, per safe_target) ===")
    print("'If we'd bet the SAFE alt-line production would have suggested,")
    print(" how often did the player actually clear that line?'\n")

    by_prop = defaultdict(list)
    for obs in enriched:
        by_prop[obs["prop_key"]].append(obs)

    header = (f"{'Prop':<22} {'Target':>7} {'Eligible':>8} {'Hits':>5} "
              f"{'Hit%':>6} {'Filtered':>9} "
              f"{'Avg gap':>8} {'Median th':>10}")
    print(header)
    print("-" * len(header))
    for prop_key in sorted(by_prop.keys()):
        rows = by_prop[prop_key]
        for tgt in targets:
            results = [simulate_safe_mode(o, params, sport_key, tgt,
                                          team_defense, league_avg_def)
                       for o in rows]
            eligible = [r for r in results if r["eligible"]]
            n_elig = len(eligible)
            n_filt = len(results) - n_elig
            if n_elig == 0:
                print(f"{prop_key:<22} {tgt*100:>6.0f}% {n_elig:>8d} "
                      f"{'—':>5} {'—':>6} {n_filt:>9d} {'—':>8} {'—':>10}")
                continue
            hits = sum(1 for r in eligible if r["hit"])
            rate = hits / n_elig
            gaps = [r["line"] - r["safe_threshold"] for r in eligible]
            avg_gap = sum(gaps) / len(gaps)
            ths = sorted(r["safe_threshold"] for r in eligible)
            med_th = ths[len(ths) // 2]
            print(f"{prop_key:<22} {tgt*100:>6.0f}% {n_elig:>8d} "
                  f"{hits:>5d} {rate * 100:>5.1f}% {n_filt:>9d} "
                  f"{avg_gap:>+8.2f} {med_th:>10d}")
        print()

    print("Eligible  = passed both production guards (p_at_safe ≥ target-0.05 AND")
    print("            safe_threshold ≥ 50% of book line)")
    print("Avg gap   = book_line − safe_threshold (positive = suggestion is below the line)")
    print("Median th = median suggested alt-line across eligible obs")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="nba")
    p.add_argument("--variants", default="recency,all",
                   help="Comma-separated subset of " + ",".join(VARIANT_PRESETS.keys()))
    p.add_argument("--confidence-report", action="store_true",
                   help="Also print precision-vs-coverage table: hit-rate when "
                        "model confidence is >= threshold (0.55, 0.60, ... 0.90).")
    p.add_argument("--safe-mode-report", action="store_true",
                   help="Also simulate production safe-mode (alt-line) suggestions "
                        "and report actual hit rates per safe_target.")
    p.add_argument("--safe-targets", default="0.85,0.90,0.95",
                   help="Comma-separated safe_target values to evaluate.")
    args = p.parse_args()

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]
    target_props = PROPS_BY_SPORT.get(args.sport, [])
    if not target_props:
        print(f"No default prop list for {args.sport}; edit PROPS_BY_SPORT.")
        sys.exit(1)

    print(f"\n=== Harvesting cached book lines for {sport_key} "
          f"({', '.join(target_props)}) ===")
    book_lines = harvest_book_lines(sport_key, target_props)
    print(f"  found {len(book_lines)} unique (player, prop, game) book lines")
    if not book_lines:
        print("Nothing to evaluate. Run the live tool first to populate cache.")
        sys.exit(0)

    by_prop = defaultdict(int)
    for r in book_lines:
        by_prop[r["prop_key"]] += 1
    for k, v in sorted(by_prop.items()):
        print(f"    {k}: {v}")

    print(f"\n=== Joining to ESPN gamelogs (using {espn_sport}/{espn_league}) ===")
    enriched = join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("No observations could be joined to actuals.")
        sys.exit(0)

    # Optional team-defense lookup if any variant uses it
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = {n: _resolve_params(VARIANT_PRESETS[n], sport_key) for n in variant_names}
    team_defense, league_avg_def = {}, None
    if any(v.get("opp_defense_strength", 0.0) > 0 for v in variants.values()):
        print("\n=== Building team-defense lookup for one variant ===")
        team_defense, _, league_avg_def = _team_defense_lookup(espn_sport, espn_league)

    print("\n=== Evaluating forecasters at REAL book lines ===")
    print()
    header = (f"{'Variant':<10} {'Prop':<22} {'N_tr':>4} {'N_te':>4} {'#pl':>4}  "
              f"{'μ_r':>7} {'σ_r':>6}  "
              f"{'BrA':>6} {'BrB':>6} {'BrB*':>6} {'BrC':>6} {'BrC*':>6}  "
              f"{'HitA':>5} {'HitB':>5} {'HitB*':>5} {'HitC':>5} {'HitC*':>5}")
    print(header); print("-" * len(header))

    per_variant_prop_results = {}
    for vname, params in variants.items():
        per_prop_obs = defaultdict(list)
        for obs in enriched:
            projected, emp = project_and_empirical(obs, params, sport_key,
                                                   team_defense, league_avg_def)
            if projected is None:
                continue
            per_prop_obs[obs["prop_key"]].append({
                "player": obs["player"],
                "projected": projected,
                "line": obs["line"],
                "actual": obs["actual"],
                "empirical_over": emp,
                "game_date": obs["game_date"],
            })

        for prop_key in sorted(per_prop_obs.keys()):
            res = evaluate_calibration(per_prop_obs[prop_key], prop_key, prop_key)
            if not res:
                continue
            per_variant_prop_results[(vname, prop_key)] = res
            brA, brB, brBs, brC, brCs = res["brier"]
            hA, hB, hBs, hC, hCs = res["hit"]
            print(f"{vname:<10} {prop_key:<22} {res['n_train']:>4} {res['n_test']:>4} "
                  f"{res['n_players']:>4}  "
                  f"{res['mu_r']:>+7.3f} {res['sigma_r']:>6.3f}  "
                  f"{brA:>6.4f} {brB:>6.4f} {brBs:>6.4f} {brC:>6.4f} {brCs:>6.4f}  "
                  f"{hA:>4.1f}% {hB:>4.1f}% {hBs:>4.1f}% {hC:>4.1f}% {hCs:>4.1f}%")

    print()
    print("A  = empirical (weighted prior-game fraction > line) — current production method")
    print("B  = pooled Gaussian residual model")
    print("C  = pooled empirical residual CDF")
    print("B* = per-player Gaussian residual model, shrunk toward pool by λ=n/(n+15)")
    print("C* = per-player empirical residual CDF, mixture-blended with pool by same λ")
    print()
    print("Calibration fit on chronologically earliest half; scoring on later half (no leakage).")

    if args.confidence_report and per_variant_prop_results:
        print_confidence_report(per_variant_prop_results)

    if args.safe_mode_report:
        try:
            targets = tuple(float(t.strip()) for t in args.safe_targets.split(",") if t.strip())
        except ValueError:
            targets = (0.85, 0.90, 0.95)
        # Use the "all" variant params (the production default-ish settings)
        sm_params = _resolve_params(VARIANT_PRESETS["all"], sport_key)
        sm_team_def = team_defense if team_defense else {}
        sm_league_avg = league_avg_def
        if sm_params.get("output_def_strength", 0.0) > 0 and not sm_team_def:
            print("\n=== Building team-defense lookup for safe-mode sim ===")
            sm_team_def, _, sm_league_avg = _team_defense_lookup(espn_sport, espn_league)
        print_safe_mode_report(enriched, sm_params, sport_key,
                               sm_team_def, sm_league_avg, targets=targets)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
