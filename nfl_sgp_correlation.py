"""nfl_sgp_correlation.py — same-game leg correlation for honest SGP joint probabilities.

For a same-game parlay the independent product Pi(p_i) is WRONG: same-game legs share game
script / total / pace, so they are correlated. Two OVERS in a shootout co-hit more than
independence says (joint > product); an over paired with an under co-hits less. The bonus
engine needs the TRUE joint P (the book bakes its own correlation haircut into the SGP price;
our edge is knowing the real joint better than the generic haircut).

Model (honest + estimable from actuals, no SGP price data needed):
  * Each leg has a latent STANDARD-NORMAL "stat level" X_i; the OVER hits iff X_i > t_i where
    P(X_i > t_i) = fair_over (market de-vigged — the calibrated marginal we proved to use).
    Modeling X on the STAT (not the win indicator) keeps the sign right regardless of bet side.
  * Correlation rho(X_i, X_j) depends on the pair's RELATIONSHIP category (same player, same-team
    QB<->pass-catcher, same-team rush<->pass = game-script, two same-team catchers, opposing, ...).
  * rho per category = tetrachoric fit: the rho whose bivariate-normal upper-orthant
    P(X>t_i, X>t_j; rho) best matches the observed co-over rate across all category pairs.
  * Joint P(ticket) = Monte-Carlo over correlated normals with the ticket's category-built R,
    each leg's win mapped to X>t (over bet) or X<t (under bet).

ESTIMATE on 2023-2024, VALIDATE on 2025: does the copula predict realized same-game 3-leg
co-hit rates better (calibration + log-loss) than independence? Reads the local ladder extract
(DK closing) + actuals. Diagnostic — no spend. Exposes joint_prob() for the bonus optimizer.

RESULT (fit 2023-24, validate 2025). Correlations among TRUSTWORTHY VOLUME props are REAL but
sign-MIXED: QB volume<->his pass-catcher +0.175; RB committee -0.55; pass-vs-rush (script)
-0.375; opposing volume -0.275 (possession is ~conserved — unlike yardage/TDs these trade off);
two same-team catchers ~0. On aggregate 3-leg favorite tickets these ~cancel: realized co-hit
15.96% vs independent product 14.77% (copula 15.01%), and independence WINS overall log-loss —
so independence is an adequate, slightly-CONSERVATIVE base joint (the ~1pp surplus is the small
favorite-bias also seen in the backtest, noise-level, don't lean on it). The copula's robust,
ACTIONABLE value is COMPOSITION SELECTION: positively-correlated same-side stacks co-hit far
more (pos-stack 19.3% vs weak/opposing 14.5%; copula wins log-loss ONLY there) — independence
UNDER-states them ~4pp, so QB+his-catcher SGPs genuinely hit more than the naive product (a real
SGP edge the book's generic haircut likely under-credits). ⇒ OPTIMIZER RULE: independence is the
safe base joint; PREFER +corr same-side stacks (QB + his WR/TE), AVOID same-team script-conflict
legs (two RBs, pass+rush) on the same side; use joint_prob() (copula) for EV when a correlated
pair is present. Precise copula pricing beyond that doesn't beat independence on these props.
"""
import argparse
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
try:
    from scipy.special import erf as _verf     # vectorized
except Exception:
    _verf = np.vectorize(math.erf)

import nfl_schedule
import nfl_data
import nfl_props_scan as scan
import nfl_props_accuracy as acc
import nfl_ladder_clv as clv
from odds_client import american_to_implied_prob, devig_two_way

TRUSTWORTHY = {"player_receptions": 6.9, "player_rush_attempts": 8.5, "player_pass_attempts": 23.4}
PROP_ABBR = {"player_receptions": "rec", "player_rush_attempts": "rush", "player_pass_attempts": "pass"}
TOP_N = 8          # top favorite legs per game to enumerate for validation tickets
MC = 4000         # Monte-Carlo draws per ticket joint


# ── standard normal helpers (dependency-free) ──
def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ppf(p):
    """Acklam's inverse normal CDF."""
    p = min(1 - 1e-12, max(1e-12, p))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)   # precomputed once (perf)
_SQRT2 = math.sqrt(2.0)


def _bvn_upper(ta, tb, rho):
    """P(X > ta, X > tb) for standard bivariate normal with correlation rho.
    Vectorized Gauss-Legendre of phi(x)*Phibar((tb - rho x)/sqrt(1-rho^2)) over x in [ta, 8]."""
    if rho <= -0.999:
        return max(0.0, _Phi(-ta) + _Phi(-tb) - 1.0)
    if rho >= 0.999:
        return _Phi(-max(ta, tb))
    lo, hi = ta, 8.0
    if lo >= hi:
        return 0.0
    s = math.sqrt(1.0 - rho * rho)
    x = 0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * _GL_W
    phi = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    phibar = 0.5 * (1.0 - _verf((tb - rho * x) / (s * _SQRT2)))   # 1 - Phi(z)
    return float(max(0.0, min(1.0, np.sum(w * phi * phibar))))


# ── leg collection ──
def _team_map(seasons):
    """(norm_name, season, week) -> (team, opponent, position)."""
    pw = nfl_data.player_week([str(s) for s in seasons])
    m = {}
    if pw is None:
        return m
    namecol = "player_display_name" if "player_display_name" in pw.columns else "player_name"
    for r in pw.to_dict("records"):
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        nm = scan._norm(r.get(namecol))
        if nm:
            m[(nm, str(r.get("season")), wk)] = (r.get("team"), r.get("opponent_team"),
                                                 r.get("position"))
    return m


def collect_game_legs(train_seasons, test_seasons):
    """{(season, gid): [leg,...]} for DK closing, trustworthy high-opp props only.
    leg: prop, player, team, opp, fair_over, t_over(=ppf(1-fair_over)), over_ind, fav_over."""
    tmap = _team_map(test_seasons)
    idx = nfl_schedule.game_index([str(s) for s in test_seasons])
    games = defaultdict(list)
    for prop, tmin in TRUSTWORTHY.items():
        cfg = acc.PROPS[prop]
        hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
        acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
        train = acc._obs_from_series(acc._series(train_seasons), cfg)
        if len(train) < 200:
            continue
        comp = acc.fit_components(train, cfg)
        feat = {(o["name"], o["season"], o["week"]): o
                for o in acc._obs_from_series(acc._series(test_seasons), cfg)}
        lines, meta = clv._load_local(prop, test_seasons)
        for (eid, player), bybook in lines.items():
            d = bybook.get("draftkings", {}).get("closing")
            if not d:
                continue
            over_q, under_q = d.get("OVER"), d.get("UNDER")
            if not (over_q and under_q):
                continue
            m = meta[(eid, player)]
            gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
            if gid is None:
                continue
            ps = gid.split("_"); week = int(ps[1])
            o = feat.get((scan._norm(player), ps[0], week))
            if o is None or o.get("exp_vol", o.get("mean_base", 0.0)) < tmin:
                continue
            line = over_q[0]
            if line is None or abs(o["actual"] - line) < 1e-9:
                continue
            try:
                fair = devig_two_way(american_to_implied_prob(over_q[1]),
                                     american_to_implied_prob(under_q[1]))[0]
            except Exception:
                continue
            tm, opp, pos = tmap.get((scan._norm(player), ps[0], week), (None, None, None))
            games[(ps[0], gid)].append({
                "prop": prop, "player": scan._norm(player), "team": tm, "opp": opp, "pos": pos,
                "fair_over": fair, "t_over": _ppf(1.0 - fair),
                "over_ind": 1 if o["actual"] > line else 0,
                "fav_over": fair >= 0.5})
    return games


# ── relationship category ──
def category(a, b):
    if a["player"] == b["player"]:
        return "same_player"
    same_team = a["team"] is not None and a["team"] == b["team"]
    props = {PROP_ABBR[a["prop"]], PROP_ABBR[b["prop"]]}
    if same_team:
        if props == {"pass", "rec"}:
            return "team_qb_rec"           # QB pass volume <-> teammate targets (+)
        if props == {"pass", "rush"}:
            return "team_pass_rush"        # pass vs rush volume = game script (-)
        if props == {"rec"}:
            return "team_rec_rec"          # two pass-catchers, same team (target share)
        if props == {"rush"}:
            return "team_rush_rush"        # RB committee (-)
        return "team_other"
    # opposing (both teams of the same game) — share the game total/pace (+)
    return "opposing"


def fit_correlations(games):
    """Tetrachoric rho per category from co-over rates (least-squares over a rho grid)."""
    pairs = defaultdict(list)   # cat -> [(ta, tb, both_over)]
    for legs in games.values():
        for a, b in combinations(legs, 2):
            pairs[category(a, b)].append((a["t_over"], b["t_over"],
                                          a["over_ind"] * b["over_ind"]))
    grid = np.linspace(-0.6, 0.9, 61)
    fitted = {}
    for cat, rows in pairs.items():
        if len(rows) < 40:
            fitted[cat] = (0.0, len(rows), None)
            continue
        obs = np.mean([r[2] for r in rows])
        best, berr = 0.0, 1e9
        for rho in grid:
            pred = np.mean([_bvn_upper(ta, tb, rho) for ta, tb, _ in rows])
            err = (pred - obs) ** 2
            if err < berr:
                berr, best = err, rho
        # indep baseline co-over for context (product of marginals)
        indep = np.mean([(1 - _Phi(ta)) * (1 - _Phi(tb)) for ta, tb, _ in rows])
        fitted[cat] = (round(float(best), 3), len(rows), (obs, indep))
    return fitted


# ── joint probability of a ticket via the copula ──
def _corr_matrix(legs, rho_by_cat, default=0.0):
    n = len(legs)
    R = np.eye(n)
    for i, j in combinations(range(n), 2):
        r = rho_by_cat.get(category(legs[i], legs[j]), (default,))[0]
        R[i, j] = R[j, i] = r
    # nearest PD (clip eigenvalues)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def joint_prob(legs, rho_by_cat, sides=None, rng=None, draws=MC):
    """P(all legs win) for a same-game ticket. sides[i] True=bet over, False=under
    (default: favorite side). legs carry t_over. Monte-Carlo over correlated normals."""
    rng = rng or np.random.default_rng(0)
    if sides is None:
        sides = [lg["fav_over"] for lg in legs]
    R = _corr_matrix(legs, rho_by_cat)
    L = np.linalg.cholesky(R)
    Z = (L @ rng.standard_normal((len(legs), draws))).T   # draws x n
    t = np.array([lg["t_over"] for lg in legs])
    win = np.where(np.array(sides), Z > t, Z < t)          # over: X>t ; under: X<t
    return float(np.all(win, axis=1).mean())


def _ticket_stratum(combo, rho_by_cat):
    """Classify a ticket by its strongest pairwise correlation (signed): where independence
    is most wrong. pos-stack => co-hits MORE than product (edge); neg-stack => LESS (trap)."""
    rhos = [rho_by_cat.get(category(a, b), (0.0,))[0] for a, b in combinations(combo, 2)]
    hi, lo = max(rhos), min(rhos)
    if hi >= 0.10 and hi >= -lo:
        return "pos-stack (+corr pair, e.g. QB+catcher)"
    if lo <= -0.30:
        return "neg-stack (-corr pair, e.g. RB committee / script)"
    return "weak (~independent, mostly opposing)"


def _agg_rows(rows):
    n = len(rows)
    if n == 0:
        return None
    real = sum(r[2] for r in rows) / n
    def ll(i):
        return -sum(r[2]*math.log(max(1e-6, r[i])) + (1-r[2])*math.log(max(1e-6, 1-r[i]))
                    for r in rows) / n
    return {"n": n, "real": real*100, "pred_ind": sum(r[0] for r in rows)/n*100,
            "pred_cop": sum(r[1] for r in rows)/n*100, "ll_ind": ll(0), "ll_cop": ll(1)}


def validate(games_test, rho_by_cat, K=3):
    """Same-game K-leg FAVORITE tickets on the test season: realized co-hit vs predicted
    under independence and copula, OVERALL and stratified by correlation content."""
    rng = np.random.default_rng(7)
    rows, by_stratum = [], defaultdict(list)
    for (season, gid), legs in games_test.items():
        favs = sorted(legs, key=lambda l: -max(l["fair_over"], 1 - l["fair_over"]))[:TOP_N]
        for combo in combinations(favs, K):
            sides = [lg["fav_over"] for lg in combo]
            p_ind = 1.0
            for lg in combo:
                p_ind *= (lg["fair_over"] if lg["fav_over"] else 1 - lg["fair_over"])
            p_cop = joint_prob(list(combo), rho_by_cat, sides, rng)
            won = int(all((lg["over_ind"] == 1) == s for lg, s in zip(combo, sides)))
            rows.append((p_ind, p_cop, won))
            by_stratum[_ticket_stratum(combo, rho_by_cat)].append((p_ind, p_cop, won))
    if not rows:
        return None
    return {"overall": _agg_rows(rows),
            "strata": {k: _agg_rows(v) for k, v in by_stratum.items()}}


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022")
    ap.add_argument("--fit", default="2023,2024", help="seasons to fit correlations on")
    ap.add_argument("--val", default="2025", help="held-out season to validate on")
    ap.add_argument("--legs", type=int, default=3)
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    fit_seasons = [s.strip() for s in args.fit.split(",") if s.strip()]
    val_seasons = [s.strip() for s in args.val.split(",") if s.strip()]

    print("=" * 96)
    print(f"  NFL SGP CORRELATION — same-game leg copula (fit {fit_seasons} -> validate {val_seasons})")
    print("  Latent stat X per leg; OVER iff X>t. rho by relationship, tetrachoric from co-over rate.")
    print("=" * 96)
    games_fit = collect_game_legs(train_seasons, fit_seasons)
    rho = fit_correlations(games_fit)
    print(f"  fit games={len(games_fit)}   estimated correlations by relationship:")
    print(f"    {'category':<16} {'rho':>7} {'nPairs':>7}  {'coOver%':>8} {'indep%':>7}  read")
    order = ["same_player", "team_qb_rec", "team_rec_rec", "team_rush_rush",
             "team_pass_rush", "team_other", "opposing"]
    notes = {"same_player": "same player, 2 props", "team_qb_rec": "QB vol <-> his catcher",
             "team_rec_rec": "2 catchers same team", "team_rush_rush": "RB committee",
             "team_pass_rush": "pass vs rush (script)", "team_other": "same team misc",
             "opposing": "both teams (game total)"}
    for cat in order:
        if cat not in rho:
            continue
        r, npairs, ctx = rho[cat]
        if ctx:
            print(f"    {cat:<16} {r:>+7.3f} {npairs:>7}  {ctx[0]*100:>7.1f}% {ctx[1]*100:>6.1f}%  {notes[cat]}")
        else:
            print(f"    {cat:<16} {'thin':>7} {npairs:>7}   (n<40 -> rho=0)   {notes[cat]}")

    print(f"\n  VALIDATION on {val_seasons}: same-game {args.legs}-leg favorite tickets, "
          "realized vs predicted")
    games_val = collect_game_legs(train_seasons, val_seasons)
    v = validate(games_val, rho, K=args.legs)
    if v is None:
        print("    (no same-game tickets)")
    else:
        def _line(label, a):
            gi, gc = a['pred_ind'] - a['real'], a['pred_cop'] - a['real']
            win = "copula" if a['ll_cop'] < a['ll_ind'] else "indep "
            print(f"    {label:<44} n={a['n']:>5}  real={a['real']:>5.2f}%  "
                  f"indep={a['pred_ind']:>5.2f}%({gi:+.2f})  cop={a['pred_cop']:>5.2f}%({gc:+.2f})  "
                  f"LL:{win}")
        _line("OVERALL", v["overall"])
        print("    -- by correlation content (where independence is most wrong) --")
        for k in ("pos-stack (+corr pair, e.g. QB+catcher)",
                  "neg-stack (-corr pair, e.g. RB committee / script)",
                  "weak (~independent, mostly opposing)"):
            if k in v["strata"] and v["strata"][k]:
                _line(k, v["strata"][k])
    print("\n  READ: if same-side same-game legs are POSITIVELY correlated, realized co-hit > the")
    print("  independent product => SGPs hit MORE than the naive product; the copula should track")
    print("  realized and cut log-loss. That true joint P (vs the book's generic SGP haircut) is")
    print("  what the bonus optimizer needs. rho by category is exposed via joint_prob().")
    print("=" * 96)


if __name__ == "__main__":
    main()
