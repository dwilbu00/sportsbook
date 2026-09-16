"""mlb_sgp_correlation.py — same-game leg correlation for MLB SGP joint probabilities.

The MLB analog of nfl_sgp_correlation. Same latent-stat Gaussian copula (X_i = the player's
stat level; OVER iff X_i > t_i where P = market de-vig fair-over), so an SGP's true joint P
isn't the independent product. Correlation depends on the pair's RELATIONSHIP:

  team_batters       two batters, SAME team    — co-vary with team offense/runs (+)
  opp_batters        two batters, both teams    — share the game's run environment (weak +)
  pitcher_vs_opp_bat pitcher + OPPOSING batter  — pitcher dominates -> opp batters suppressed (-)
  pitcher_own_bat    pitcher + own-team batter  — different innings, ~independent
  opp_pitchers       both starters              — anti-correlated via who's winning (rare)

rho per category = tetrachoric fit to the observed co-over rate (reuses nfl_sgp_correlation's
bivariate-normal math). Fit on DISCOVERY seasons, validate on HELD-OUT: does the copula predict
realized same-game co-hit rates better than independence? Substrate = book_line_calibration real
lines + warehouse actuals (same as the threshold backtest), grouped by game_pk. Diagnostic, no
spend. Doug runs it; the frozen rho feeds the MLB bonus optimizer's SGP joint P.
"""
import argparse
import json
import math
import os
from collections import defaultdict
from itertools import combinations

import numpy as np

import book_line_calibration as blc
import nfl_sgp_correlation as ncorr        # reuse the copula primitives (sport-agnostic)
from odds_client import american_to_implied_prob, devig_two_way

FROZEN_PATH = "calibration/mlb_sgp_correlations.json"
BATTER = "batter"
PITCHER = "pitcher"


def _fam(prop):
    return PITCHER if prop.startswith("pitcher_") else BATTER


def category(a, b):
    """Relationship category for two same-game legs (by family + team)."""
    fa, fb = a["fam"], b["fam"]
    same_team = a["team_id"] is not None and a["team_id"] == b["team_id"]
    if fa == BATTER and fb == BATTER:
        return "team_batters" if same_team else "opp_batters"
    if fa == PITCHER and fb == PITCHER:
        return "opp_pitchers"
    # one pitcher, one batter
    return "pitcher_own_bat" if same_team else "pitcher_vs_opp_bat"


def collect(seasons, snapshot):
    """{game_pk: [leg,...]} for joinable MLB real-line bets in `seasons`. leg carries fam,
    team_id, fair_over, t_over, over_ind."""
    props = ["batter_hits", "batter_total_bases", "batter_rbis",
             "pitcher_strikeouts", "pitcher_earned_runs", "pitcher_outs"]
    book_lines, _a, _b = blc.harvest_real_line_book_lines("baseball_mlb", props, snapshot=snapshot)
    print(f"  harvested {len(book_lines)} lines; joining…")
    enriched = blc.join_book_lines_to_actuals(book_lines, "baseball", "mlb")
    games = defaultdict(list)
    for r in enriched:
        season = str(r.get("game_date") or "")[:4]
        if season not in seasons:
            continue
        line, actual = r.get("line"), r.get("actual")
        if line is None or actual is None or abs(actual - line) < 1e-9:
            continue
        op, up = r.get("over_price"), r.get("under_price")
        if op is None or up is None:
            continue
        try:
            fair = devig_two_way(american_to_implied_prob(op), american_to_implied_prob(up))[0]
        except Exception:
            continue
        gpk = r.get("game_pk") or (r.get("player_mlb_id"), r.get("game_date"))
        tg = r.get("test_game") or {}
        games[gpk].append({
            "fam": _fam(r["prop_key"]), "prop": r["prop_key"],
            "team_id": tg.get("team_id"), "player": r.get("player"),
            "fair_over": fair, "t_over": ncorr._ppf(1.0 - fair),
            "over_ind": 1 if actual > line else 0})
    return games


def fit_correlations(games):
    """Tetrachoric rho per category from co-over rates (least-squares over a rho grid)."""
    pairs = defaultdict(list)
    for legs in games.values():
        for a, b in combinations(legs, 2):
            pairs[category(a, b)].append((a["t_over"], b["t_over"], a["over_ind"] * b["over_ind"]))
    grid = np.linspace(-0.6, 0.9, 61)
    fitted = {}
    for cat, rows in pairs.items():
        if len(rows) < 40:
            fitted[cat] = (0.0, len(rows), None)
            continue
        obs = np.mean([r[2] for r in rows])
        best, berr = 0.0, 1e9
        for rho in grid:
            pred = np.mean([ncorr._bvn_upper(ta, tb, rho) for ta, tb, _ in rows])
            if (pred - obs) ** 2 < berr:
                berr, best = (pred - obs) ** 2, rho
        indep = np.mean([(1 - ncorr._Phi(ta)) * (1 - ncorr._Phi(tb)) for ta, tb, _ in rows])
        fitted[cat] = (round(float(best), 3), len(rows), (obs, indep))
    return fitted


def _corr_matrix(legs, rho):
    n = len(legs)
    R = np.eye(n)
    for i, j in combinations(range(n), 2):
        r = ncorr._rho_val(rho.get(category(legs[i], legs[j])))
        R[i, j] = R[j, i] = r
    w, V = np.linalg.eigh(R)
    R = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def joint_prob(legs, rho, sides=None, rng=None, draws=4000):
    rng = rng or np.random.default_rng(0)
    if sides is None:
        sides = [lg["fair_over"] >= 0.5 for lg in legs]
    L = np.linalg.cholesky(_corr_matrix(legs, rho))
    Z = (L @ rng.standard_normal((len(legs), draws))).T
    t = np.array([lg["t_over"] for lg in legs])
    win = np.where(np.array(sides), Z > t, Z < t)
    return float(np.all(win, axis=1).mean())


def freeze(fitted, path=FROZEN_PATH):
    out = {c: round(ncorr._rho_val(v), 3) for c, v in fitted.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rho_by_category": out, "note": "MLB tetrachoric fit"}, f, indent=2)
    return out


def validate(games, rho, K=3):
    """Same-game K-leg FAVORITE tickets: realized co-hit vs independence vs copula."""
    rng = np.random.default_rng(7)
    rows = []
    for legs in games.values():
        favs = sorted(legs, key=lambda l: -max(l["fair_over"], 1 - l["fair_over"]))[:8]
        for combo in combinations(favs, K):
            sides = [l["fair_over"] >= 0.5 for l in combo]
            p_ind = 1.0
            for l in combo:
                p_ind *= (l["fair_over"] if l["fair_over"] >= 0.5 else 1 - l["fair_over"])
            p_cop = joint_prob(list(combo), rho, sides, rng)
            won = int(all((l["over_ind"] == 1) == s for l, s in zip(combo, sides)))
            rows.append((p_ind, p_cop, won))
    if not rows:
        return None
    n = len(rows)

    def ll(i):
        return -sum(r[2] * math.log(max(1e-6, r[i])) + (1 - r[2]) * math.log(max(1e-6, 1 - r[i]))
                    for r in rows) / n
    return {"n": n, "real": sum(r[2] for r in rows) / n * 100,
            "ind": sum(r[0] for r in rows) / n * 100, "cop": sum(r[1] for r in rows) / n * 100,
            "ll_ind": ll(0), "ll_cop": ll(1)}


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default="2023,2024,2025")
    ap.add_argument("--val", default="2026")
    ap.add_argument("--legs", type=int, default=3)
    ap.add_argument("--snapshot", default="closing")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    fit_seasons = set(s.strip() for s in args.fit.split(",") if s.strip())
    val_seasons = set(s.strip() for s in args.val.split(",") if s.strip())

    print("=" * 92)
    print(f"  MLB SGP CORRELATION — same-game copula (fit {sorted(fit_seasons)} -> val {sorted(val_seasons)})")
    print("=" * 92)
    gfit = collect(fit_seasons, args.snapshot)
    rho = fit_correlations(gfit)
    print(f"  fit games={len(gfit)}   correlations by relationship:")
    notes = {"team_batters": "2 batters same team (runs)", "opp_batters": "2 batters both teams",
             "pitcher_vs_opp_bat": "pitcher vs opposing batter", "pitcher_own_bat": "pitcher + own batter",
             "opp_pitchers": "both starters"}
    for cat in ("team_batters", "opp_batters", "pitcher_vs_opp_bat", "pitcher_own_bat", "opp_pitchers"):
        if cat not in rho:
            continue
        r, npairs, ctx = rho[cat]
        if ctx:
            print(f"    {cat:<20} rho={r:>+6.3f}  n={npairs:>6}  coOver={ctx[0]*100:>5.1f}% "
                  f"indep={ctx[1]*100:>5.1f}%  {notes.get(cat,'')}")
        else:
            print(f"    {cat:<20} thin (n={npairs})")
    if args.freeze:
        print(f"\n  FROZEN -> {freeze(rho)}")
    gval = collect(val_seasons, args.snapshot)
    v = validate(gval, rho, K=args.legs)
    print(f"\n  VALIDATION {sorted(val_seasons)} {args.legs}-leg same-game favorite tickets:")
    if v:
        print(f"    tickets={v['n']:,}  realized={v['real']:.2f}%  independence={v['ind']:.2f}%  "
              f"copula={v['cop']:.2f}%")
        print(f"    log-loss: indep={v['ll_ind']:.4f}  copula={v['ll_cop']:.4f}  "
              f"({'COPULA better' if v['ll_cop'] < v['ll_ind'] else 'independence better'})")
    print("=" * 92)


if __name__ == "__main__":
    main()
