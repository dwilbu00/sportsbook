"""nfl_props_accuracy.py — actuals-only distributional accuracy on the FULL history.

We don't need book lines to measure whether the MODEL is accurate — we have the actual
result for every player-game back to 2012. This trains each prop's process components on
a disjoint historical era (2012-2022, ~195k player-games) and scores them on held-out
data, so there is zero leakage (Doug's train/test design):

  TRAIN  2012-2022 : fit league-level components (NegBin dispersion φ, the aDOT→conversion
                     curve, league rates, the variance-vs-volume model). Frozen after this.
  TEST   2023-2025 : (A) ACTUALS-ONLY on every player-game — a proper distributional score
                     (log-score) + a line-grid Brier (the over/under call swept across many
                     candidate lines, so it measures skill, not just median calibration);
                     (B) MARKET reference on the games that carry a real DK line — our Brier
                     at the true line vs the de-vigged market Brier (distance to the sharp
                     close).

Grading against our OWN predicted number would be ~50% by construction (it only tests
whether our point sits at the median); scoring the whole predicted DISTRIBUTION against
the actual — via log-score and a Brier swept over a grid of lines — is the real skill test.

Baselines on the same corpus: hit-rate (empirical over-rate) and, for counts, Poisson —
so we re-confirm the opportunity/over-dispersion wins on 10 seasons, not just 3.

Per-player features always use only strictly-earlier weeks in the same season (as live).
Components are fit ONLY on the train era. Diagnostic only — writes nothing.
"""
import argparse
import math
from collections import defaultdict

import nfl_data
import nfl_schedule
import nfl_props_scan as scan
from odds_client import american_to_implied_prob, devig_two_way

HALF_LIFE = 4
MIN_PRIOR = 3

# family: count | count_conv (receptions) | yards ;  var: negbin | volnorm
PROPS = {
    "player_receptions":      dict(fam="count_conv", stat="receptions",     vol="targets",     qual="adot",  var="negbin"),
    "player_rush_yds":        dict(fam="yards",      stat="rushing_yards",  vol="carries",     qual=None,    var="volnorm"),
    "player_reception_yds":   dict(fam="yards",      stat="receiving_yards", vol="receptions", qual="adot",  var="volnorm"),
    "player_pass_yds":        dict(fam="yards",      stat="passing_yards",  vol="attempts",    qual=None,    var="volnorm"),
    "player_rush_attempts":   dict(fam="count",      stat="carries",        var="negbin"),
    "player_pass_attempts":   dict(fam="count",      stat="attempts",       var="negbin"),
    "player_pass_completions": dict(fam="count",     stat="completions",    var="negbin"),
    "player_pass_tds":        dict(fam="count",      stat="passing_tds",    var="negbin"),
}

# line grids for the swept-Brier skill test (plausible book-line ranges per prop)
GRIDS = {
    "player_receptions":      [i + 0.5 for i in range(0, 11)],
    "player_rush_yds":        list(range(10, 141, 10)),
    "player_reception_yds":   list(range(10, 141, 10)),
    "player_pass_yds":        list(range(150, 351, 25)),
    "player_rush_attempts":   [i + 0.5 for i in range(2, 26)],
    "player_pass_attempts":   [i + 0.5 for i in range(15, 46)],
    "player_pass_completions": [i + 0.5 for i in range(10, 36)],
    "player_pass_tds":        [i + 0.5 for i in range(0, 5)],
}
CONV_BOUNDS = {"count_conv": (0.20, 0.95), "yards": (1.5, 25.0)}
SHRINK_K = 12.0
SD_FLOOR = 6.0

# Winners from the 2012-2020 fit / 2021-2022 validation sweep (HALF_LIFE, MIN_PRIOR, K).
# HALF_LIFE 999 == flat (no recency decay). Applied with --use-swept for the frozen
# 2023-2025 holdout confirmation.
SWEPT = {
    "player_receptions":      (6, 3, 20),
    "player_rush_yds":        (3, 4, 40),
    "player_reception_yds":   (8, 4, 40),
    "player_pass_yds":        (2, 4, 40),
    "player_rush_attempts":   (4, 4, 12),
    "player_pass_attempts":   (2, 4, 12),
    "player_pass_completions": (2, 4, 12),
    "player_pass_tds":        (999, 4, 12),
}


def _recency_weights(n):
    return [0.5 ** (i / HALF_LIFE) for i in range(n)] if n else []


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _series(seasons):
    pw = nfl_data.player_week([str(s) for s in seasons])
    if pw is None:
        return {}
    cols = set(pw.columns)
    namecol = "player_display_name" if "player_display_name" in cols else "player_name"
    keep = ["targets", "receptions", "receiving_air_yards", "receiving_yards",
            "carries", "rushing_yards", "attempts", "completions", "passing_yards",
            "passing_tds"]
    keep = [c for c in keep if c in cols]
    out = defaultdict(list)
    for r in pw.to_dict("records"):
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        nm = scan._norm(r.get(namecol))
        if not nm:
            continue
        rec = {c: _num(r.get(c)) for c in keep}
        rec["week"] = wk
        out[(nm, str(r.get("season")))].append(rec)
    for k in out:
        out[k].sort(key=lambda x: -x["week"])
    return out


def _obs_from_series(series, cfg):
    """Every player-game with >= MIN_PRIOR prior weeks → feature dict + actual.
    Actuals-only; no book line. Keyed also by (name,season,week) for later line-match."""
    stat, vol, qual = cfg["stat"], cfg.get("vol"), cfg.get("qual")
    obs = []
    for (nm, season), rows in series.items():
        for i, g in enumerate(rows):
            wk = g["week"]
            actual = g.get(stat)
            if actual is None:
                continue
            prior = [r for r in rows if r["week"] < wk]
            if len(prior) < MIN_PRIOR:
                continue
            w = _recency_weights(len(prior))
            wsum = sum(w) or 1.0

            def wsum_of(col):
                return sum(wi * (r.get(col) or 0.0) for r, wi in zip(prior, w))

            o = {"name": nm, "season": season, "week": wk, "actual": float(actual)}
            if cfg["fam"] == "count":
                o["mean_base"] = wsum_of(stat) / wsum
                if o["mean_base"] <= 0:
                    continue
            else:                                   # count_conv | yards
                sv = wsum_of(vol)
                if sv <= 0:
                    continue
                o["exp_vol"] = sv / wsum
                o["conv_own"] = wsum_of(stat) / sv
                o["vw"] = sv                         # weight for conv fits/shrink
                if qual == "adot":
                    st = wsum_of("targets")
                    o["adot"] = (wsum_of("receiving_air_yards") / st) if st > 0 else None
                else:
                    o["adot"] = None
            obs.append(o)
    return obs


# ── NegBin / Normal distribution helpers ──
def _negbin_logpmf(k, mean, phi):
    mean = max(1e-9, mean)
    if phi <= 0:
        return -mean + k * math.log(mean) - math.lgamma(k + 1)
    size = 1.0 / phi
    p = size / (size + mean)
    return (math.lgamma(k + size) - math.lgamma(size) - math.lgamma(k + 1)
            + size * math.log(p) + k * math.log1p(-p))


def _negbin_sf(kth, mean, phi):
    """P(X >= kth) via lower-tail recurrence (kth integer)."""
    if kth <= 0:
        return 1.0
    mean = max(1e-9, mean)
    if phi <= 0:
        term = math.exp(-mean); cdf = term
        for s in range(1, kth):
            term *= mean / s; cdf += term
        return max(0.0, min(1.0, 1.0 - cdf))
    size = 1.0 / phi
    p = size / (size + mean); q = 1.0 - p
    term = p ** size; cdf = term
    for s in range(1, kth):
        term *= (s - 1.0 + size) / s * q; cdf += term
    return max(0.0, min(1.0, 1.0 - cdf))


_LOG2PI = math.log(2 * math.pi)


def _normal_logpdf(x, mean, sd):
    sd = max(SD_FLOOR, sd)
    return -0.5 * _LOG2PI - math.log(sd) - (x - mean) ** 2 / (2 * sd * sd)


def _normal_sf(x, mean, sd):
    sd = max(SD_FLOOR, sd)
    return 1.0 - 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2))))


def _solve(A, b):
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _fit_negbin_phi(pairs, cap=2.0):
    pts = [(m, a) for m, a in pairs if m and m > 0]
    if len(pts) < 2:
        return 0.0
    sm = sum(m for m, _ in pts); sm2 = sum(m * m for m, _ in pts)
    ssq = sum((a - m) ** 2 for m, a in pts)
    if sm2 <= 0 or (ssq - sm) / sm2 <= 0:
        return 0.0

    def nll(phi):
        return -sum(_negbin_logpmf(int(round(a)), m, phi) for m, a in pts)

    best_phi, best = 0.0, nll(0.0)
    for i in range(1, 41):
        phi = cap * i / 40
        v = nll(phi)
        if v < best:
            best, best_phi = v, phi
    return best_phi


def _mean_of(o, cfg, comp):
    if cfg["fam"] == "count":
        return o["mean_base"]
    conv = o["conv_own"]
    lo, hi = CONV_BOUNDS[cfg["fam"]]
    if cfg.get("qual") == "adot" and comp.get("curve") and o.get("adot") is not None:
        a, b = comp["curve"]
        prior = max(lo, min(hi, a + b * o["adot"]))
    else:
        prior = comp["league_conv"]
    wt = o["vw"] / (o["vw"] + SHRINK_K)
    conv_adj = max(lo, min(hi, wt * conv + (1 - wt) * prior))
    return o["exp_vol"] * conv_adj


def fit_components(train, cfg):
    comp = {}
    # count_conv / yards need league_conv (+ curve) fit BEFORE any mean is computed.
    if cfg["fam"] != "count":
        lo, hi = CONV_BOUNDS[cfg["fam"]]
        sv = sum(o["vw"] for o in train)
        comp["league_conv"] = (sum(o["conv_own"] * o["vw"] for o in train) / sv
                               if sv else (lo + hi) / 2)
        if cfg.get("qual") == "adot":
            pts = [(o["adot"], o["conv_own"], o["vw"]) for o in train
                   if o.get("adot") is not None and o["vw"] > 0]
            if len(pts) >= 50:
                sw = sum(w for *_, w in pts)
                mx = sum(w * x for x, _, w in pts) / sw
                my = sum(w * y for _, y, w in pts) / sw
                sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
                sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
                comp["curve"] = (my - (sxy / sxx) * mx, sxy / sxx) if sxx > 0 else None
    if cfg["var"] == "negbin":
        comp["phi"] = _fit_negbin_phi(
            [(_mean_of(o, cfg, comp), o["actual"]) for o in train])
    else:                                            # volnorm: Var ~ a+b*vol+c*vol^2
        X, y = [], []
        for o in train:
            m = _mean_of(o, cfg, comp)
            X.append([1.0, o["exp_vol"], o["exp_vol"] ** 2])
            y.append((o["actual"] - m) ** 2)
        A = [[0.0] * 3 for _ in range(3)]; bvec = [0.0] * 3
        for xi, yi in zip(X, y):
            for i in range(3):
                bvec[i] += xi[i] * yi
                for j in range(3):
                    A[i][j] += xi[i] * xi[j]
        comp["varcoef"] = _solve(A, bvec) or None
        resid = math.sqrt(sum(y) / len(y)) if y else SD_FLOOR
        comp["sd_const"] = max(SD_FLOOR, resid)
    return comp


def _sd_of(o, cfg, comp):
    vc = comp.get("varcoef")
    if not vc:
        return comp.get("sd_const", SD_FLOOR)
    a, b, c = vc
    v = a + b * o["exp_vol"] + c * o["exp_vol"] ** 2
    return max(SD_FLOOR, math.sqrt(v)) if v > 0 else comp.get("sd_const", SD_FLOOR)


def p_over(o, line, cfg, comp):
    m = _mean_of(o, cfg, comp)
    if cfg["var"] == "negbin":
        return _negbin_sf(int(line) + 1, m, comp["phi"])
    return max(0.0, min(1.0, _normal_sf(line, m, _sd_of(o, cfg, comp))))


def logscore(o, cfg, comp):
    m = _mean_of(o, cfg, comp)
    if cfg["var"] == "negbin":
        return _negbin_logpmf(int(round(o["actual"])), m, comp["phi"])
    return _normal_logpdf(o["actual"], m, _sd_of(o, cfg, comp))


def evaluate(prop, cfg, train_seasons, test_seasons, snapshot, book):
    tr_series = _series(train_seasons)
    te_series = _series(test_seasons)
    train = _obs_from_series(tr_series, cfg)
    test = _obs_from_series(te_series, cfg)
    if len(train) < 200 or len(test) < 100:
        print(f"  {prop}: thin (train {len(train)}, test {len(test)}) — skip")
        return
    comp = fit_components(train, cfg)

    # ── (A) actuals-only: line-grid Brier (skill) + mean log-score ──
    grid = GRIDS[prop]
    n_cells = 0
    sse_model = 0.0
    sse_base = 0.0                       # base rate (constant over-rate per line) baseline
    # per-line base rate computed on TRAIN (no leakage)
    base_rate = {L: sum(1 for o in train if o["actual"] > L) / len(train) for L in grid}
    ls = 0.0
    for o in test:
        ls += logscore(o, cfg, comp)
        for L in grid:
            y = 1 if o["actual"] > L else 0
            p = p_over(o, L, cfg, comp)
            sse_model += (p - y) ** 2
            sse_base += (base_rate[L] - y) ** 2
            n_cells += 1
    brier_model = sse_model / n_cells
    brier_base = sse_base / n_cells
    print(f"  {prop:<24} train={len(train):>6} test={len(test):>6}  "
          f"| grid-Brier model={brier_model:.4f} base={brier_base:.4f} "
          f"(Δ{brier_base - brier_model:+.4f})  | mean logscore={ls / len(test):+.3f}"
          + (f"  φ={comp['phi']:.2f}" if cfg["var"] == "negbin" else ""))

    # ── (B) market reference on games with a real DK line ──
    feat = {(o["name"], o["season"], o["week"]): o for o in test}
    props = scan._load_props(test_seasons, snapshot, book)
    idx = nfl_schedule.game_index([str(s) for s in test_seasons])
    n = 0; sse_m = 0.0; sse_k = 0.0
    for (eid, player, pk), d in props.items():
        if pk != prop:
            continue
        gid, _ = nfl_schedule.resolve_event(d["home"], d["away"], d["commence"], index=idx)
        if gid is None:
            continue
        ps = gid.split("_"); season, week = ps[0], int(ps[1])
        o = feat.get((scan._norm(player), season, week))
        if o is None:
            continue
        line = d.get("OVER", d.get("UNDER", (None, None)))[0]
        over, under = d.get("OVER"), d.get("UNDER")
        if line is None or abs(o["actual"] - line) < 1e-9:
            continue
        y = 1 if o["actual"] > line else 0
        sse_k += (p_over(o, line, cfg, comp) - y) ** 2
        if over and under:
            try:
                mk = devig_two_way(american_to_implied_prob(over[1]),
                                   american_to_implied_prob(under[1]))[0]
                sse_m += (mk - y) ** 2
                n += 1
            except Exception:
                pass
    if n:
        print(f"  {'':<24} at REAL DK lines (n={n}):  model Brier={sse_k / n:.4f}   "
              f"market Brier={sse_m / n:.4f}   gap={sse_k / n - sse_m / n:+.4f}")


def _mean_logscore(obs, cfg, comp):
    return sum(logscore(o, cfg, comp) for o in obs) / len(obs) if obs else -9e9


def sweep_prop(prop, cfg, fit_seasons, val_seasons):
    """Tune HALF_LIFE / MIN_PRIOR / SHRINK_K on a train-era validation split (fit on
    fit_seasons, validate on val_seasons) by mean log-score — the 2023-2026 holdout is
    never touched. Reports best combo vs the defaults."""
    global HALF_LIFE, MIN_PRIOR, SHRINK_K
    fit_ser = _series(fit_seasons)
    val_ser = _series(val_seasons)
    HL_GRID = [2, 3, 4, 6, 8, 999]
    MP_GRID = [2, 3, 4]
    K_GRID = [6, 12, 20, 40] if cfg["fam"] != "count" else [12]
    results = []
    for hl in HL_GRID:
        for mp in MP_GRID:
            HALF_LIFE, MIN_PRIOR = hl, mp
            tr = _obs_from_series(fit_ser, cfg)
            va = _obs_from_series(val_ser, cfg)
            if len(tr) < 200 or len(va) < 100:
                continue
            for k in K_GRID:
                SHRINK_K = k
                comp = fit_components(tr, cfg)
                results.append(((hl, mp, k), _mean_logscore(va, cfg, comp)))
    if not results:
        print(f"  {prop:<24} (too thin to sweep)")
        return
    results.sort(key=lambda x: -x[1])
    best, best_ls = results[0]
    dflt = next((ls for combo, ls in results if combo == (4, 3, 12)), None)
    HALF_LIFE, MIN_PRIOR, SHRINK_K = 4, 3, 12
    hl_s = "flat" if best[0] == 999 else best[0]
    print(f"  {prop:<24} best HL={hl_s} MIN_PRIOR={best[1]} K={best[2]}  "
          f"val logscore={best_ls:+.4f}"
          + (f"   default(4,3,12)={dflt:+.4f}  Δ={best_ls - dflt:+.4f}"
             if dflt is not None else ""))


def sweep_modern(prop, cfg, comp_seasons, modern_seasons):
    """Nested CV INSIDE the odds era: components frozen on comp_seasons (2012-2022); for
    each combo score each modern season; then for each held-out modern season, SELECT the
    combo on the other modern seasons and report its score on the held-out one (rotate).
    Uses the relevant modern window for selection while never reporting on selected data."""
    global HALF_LIFE, MIN_PRIOR, SHRINK_K
    comp_ser = _series(comp_seasons)
    mod_ser = {s: _series([s]) for s in modern_seasons}
    HL_GRID = [2, 3, 4, 6, 8, 999]
    MP_GRID = [2, 3, 4]
    K_GRID = [6, 12, 20, 40] if cfg["fam"] != "count" else [12]
    scores = {}                                   # combo -> {season: logscore}
    for hl in HL_GRID:
        for mp in MP_GRID:
            HALF_LIFE, MIN_PRIOR = hl, mp
            ctr = _obs_from_series(comp_ser, cfg)
            if len(ctr) < 200:
                continue
            mod_obs = {s: _obs_from_series(mod_ser[s], cfg) for s in modern_seasons}
            if any(len(mod_obs[s]) < 60 for s in modern_seasons):
                continue
            for k in K_GRID:
                SHRINK_K = k
                comp = fit_components(ctr, cfg)
                scores[(hl, mp, k)] = {s: _mean_logscore(mod_obs[s], cfg, comp)
                                       for s in modern_seasons}
    if not scores:
        print(f"  {prop:<24} (too thin)")
        return
    dflt = (4, 3, 12)
    fold_lines = []
    picks = []
    for test_s in modern_seasons:
        sel = [s for s in modern_seasons if s != test_s]
        best = max(scores, key=lambda c: sum(scores[c][x] for x in sel) / len(sel))
        picks.append(best)
        d = scores.get(dflt, {}).get(test_s)
        fold_lines.append((test_s, best, scores[best][test_s], d))
    swept_mean = sum(l[2] for l in fold_lines) / len(fold_lines)
    dflt_mean = (sum(l[3] for l in fold_lines) / len(fold_lines)
                 if all(l[3] is not None for l in fold_lines) else None)
    from collections import Counter
    modal = Counter(picks).most_common(1)[0][0]
    hl_s = "flat" if modal[0] == 999 else modal[0]
    tag = (f"   default={dflt_mean:+.4f}  Δ={swept_mean - dflt_mean:+.4f}"
           if dflt_mean is not None else "")
    print(f"  {prop:<24} nested-CV swept logscore={swept_mean:+.4f}{tag}   "
          f"modal pick HL={hl_s} MP={modal[1]} K={modal[2]}")
    for ts, bc, sc, d in fold_lines:
        hh = "flat" if bc[0] == 999 else bc[0]
        print(f"        held-out {ts}: pick HL={hh} MP={bc[1]} K={bc[2]}  "
              f"logscore={sc:+.4f}" + (f" (default {d:+.4f})" if d is not None else ""))
    HALF_LIFE, MIN_PRIOR, SHRINK_K = 4, 3, 12


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022")
    ap.add_argument("--test", default="2023,2024,2025")
    ap.add_argument("--snapshot", default="closing")
    ap.add_argument("--book", default="draftkings")
    ap.add_argument("--prop", default="all")
    ap.add_argument("--sweep", action="store_true",
                    help="tune HL/MIN_PRIOR/K on a train-era split (fit<=val), holdout untouched")
    ap.add_argument("--use-swept", dest="use_swept", action="store_true",
                    help="apply the frozen per-prop SWEPT hyperparameters for the holdout test")
    ap.add_argument("--sweep-modern", dest="sweep_modern", action="store_true",
                    help="nested CV inside the odds era: select HP on modern seasons, test rotated")
    ap.add_argument("--fit", default="2012,2013,2014,2015,2016,2017,2018,2019,2020")
    ap.add_argument("--val", default="2021,2022")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    props = list(PROPS) if args.prop == "all" else [args.prop]
    if getattr(args, "use_swept", False):
        global HALF_LIFE, MIN_PRIOR, SHRINK_K
    if getattr(args, "sweep_modern", False):
        print("=" * 104)
        print(f"  NESTED-CV SWEEP INSIDE THE ODDS ERA — components frozen on "
              f"{train_seasons[0]}-{train_seasons[-1]}; select HP on 2 modern seasons, "
              f"test on the 3rd (rotate). objective = held-out log-score")
        print("=" * 104)
        for prop in props:
            sweep_modern(prop, PROPS[prop], train_seasons, test_seasons)
        print("=" * 104)
        return
    if args.sweep:
        fit_seasons = [s.strip() for s in args.fit.split(",") if s.strip()]
        val_seasons = [s.strip() for s in args.val.split(",") if s.strip()]
        print("=" * 104)
        print(f"  HYPERPARAMETER SWEEP — fit {fit_seasons[0]}-{fit_seasons[-1]} / "
              f"validate {val_seasons} (2023-2026 holdout untouched); objective = val log-score")
        print("=" * 104)
        for prop in props:
            sweep_prop(prop, PROPS[prop], fit_seasons, val_seasons)
        print("=" * 104)
        return
    print("=" * 104)
    print(f"  NFL PROPS ACCURACY — train {train_seasons[0]}-{train_seasons[-1]} "
          f"(actuals) → test {test_seasons} | grid-Brier (skill, swept lines) + logscore")
    print("  Δ>0 vs base ⇒ model beats the naive per-line base rate. gap = distance to sharp DK.")
    print("=" * 104)
    for prop in props:
        if getattr(args, "use_swept", False) and prop in SWEPT:
            HALF_LIFE, MIN_PRIOR, SHRINK_K = SWEPT[prop]
        evaluate(prop, PROPS[prop], train_seasons, test_seasons, args.snapshot, args.book)
    print("=" * 104)


if __name__ == "__main__":
    main()
