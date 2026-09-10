"""nfl_receiving_yds_process.py — PROCESS-first receiving-yards model.

The richest carry-over: receiving_yds is COMPOUND yardage (like rushing_yds) AND the
aDOT quality layer applies directly to the conversion — a deeper average target yields
more yards PER RECEPTION, not just a lower catch rate. So this market combines both
innovations proven on the earlier props:

  * the variance-vs-volume fix (rushing_yds) — a Normal whose sd GROWS with expected
    receptions (Var ≈ a + b·E[rec] + c·E[rec]²) instead of a constant pooled sd; and
  * the aDOT-driven conversion (receptions) — yards-per-reception modeled from target
    DEPTH, so the projection leans on what's underneath the stat, not the stat itself.

receiving_yds = receptions × yards-per-reception (YPR). Nested models (all →
P(receiving_yds > line) via a Normal with the model's mean/sd):
  M0 incumbent A   : empirical over-rate at the line over prior weeks (baseline).
  M1 naive Gauss   : mean = exp_rec × own YPR, CONSTANT pooled sd.
  M2 vol-var Gauss : same mean, variance GROWS with expected receptions — the FIX.
  M3 +YPR shrink   : YPR shrunk toward league YPR (robust cold-start / self-shrink).
  M4 +aDOT YPR     : YPR shrunk toward the aDOT-IMPLIED league YPR (depth drives yards
                     per reception) — the fundamentals conversion carried from receptions.
  market           : de-vigged two-way (the sharp reference).

Same DEFAULTS as the other props (HALF_LIFE/MIN_PRIOR/shrink K) — template generalization
is tested on identical knobs BEFORE any sweep. Leakage-safe: features use only strictly
earlier weeks; variance coeffs, league YPR and the aDOT→YPR curve fit on the train fold,
scored on the held-out half (2-fold). Diagnostic only. Reuses nfl_props_scan loaders.
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
import nfl_data
import nfl_props_scan as scan
from stats import _norm_cdf
from odds_client import (american_to_decimal, american_to_implied_prob,
                         devig_two_way)

SPORT = "americanfootball_nfl"
PROP = "player_reception_yds"

HALF_LIFE = 4
MIN_PRIOR = 3
YPR_BOUNDS = (4.0, 20.0)      # yards per reception (RB ~7, WR ~12-15, deep threat higher)
YPR_SHRINK_K = 12.0           # pseudo-receptions pulling own YPR toward the prior
ADOT_BOUNDS = (0.5, 20.0)
SD_FLOOR = 8.0                # yards


def _recency_weights(n, half_life):
    if n <= 0:
        return []
    if not half_life or half_life <= 0:
        return [1.0] * n
    return [0.5 ** (i / half_life) for i in range(n)]


def _player_series(seasons):
    """{(norm_name, season): [(week, targets, receptions, rec_yds, air_yards), ...]
    most-recent-first}."""
    pw = nfl_data.player_week([str(s) for s in seasons])
    if pw is None:
        return {}
    cols = set(pw.columns)
    namecol = "player_display_name" if "player_display_name" in cols else "player_name"

    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    series = defaultdict(list)
    for r in pw.to_dict("records"):
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        nm = scan._norm(r.get(namecol))
        if not nm:
            continue
        series[(nm, str(r.get("season")))].append(
            (wk, num(r.get("targets")), num(r.get("receptions")),
             num(r.get("receiving_yards")), num(r.get("receiving_air_yards"))))
    for k in series:
        series[k].sort(key=lambda x: -x[0])
    return series


def build_corpus(seasons, snapshot="closing", book="draftkings"):
    props = scan._load_props(seasons, snapshot, book)
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    pw_stat = scan._player_week_index(seasons)
    snaps = scan._snap_presence([str(s) for s in seasons])
    series = _player_series(seasons)

    obs = []
    n_props = n_nogame = n_void = n_played0 = n_thin = n_norec = 0
    for (eid, player, pk), d in props.items():
        if pk != PROP:
            continue
        n_props += 1
        gid, _ = nfl_schedule.resolve_event(d["home"], d["away"], d["commence"],
                                            index=idx)
        if gid is None:
            n_nogame += 1
            continue
        parts = gid.split("_")
        season, week = parts[0], int(parts[1])
        nm = scan._norm(player)
        actual = (pw_stat.get((nm, season, str(week))) or {}).get("receiving_yards")
        if actual is None:
            if snaps.get((nm, season, str(week))):
                actual = 0.0
                n_played0 += 1
            else:
                n_void += 1
                continue
        prior = [row for row in series.get((nm, season), []) if row[0] < week]
        if len(prior) < MIN_PRIOR:
            n_thin += 1
            continue
        w = _recency_weights(len(prior), HALF_LIFE)
        wsum = sum(w) or 1.0
        tw = sum(wi * (r[1] or 0.0) for r, wi in zip(prior, w))       # targets
        rcw = sum(wi * (r[2] or 0.0) for r, wi in zip(prior, w))      # receptions
        yw = sum(wi * (r[3] or 0.0) for r, wi in zip(prior, w))       # rec yards
        ayw = sum(wi * (r[4] or 0.0) for r, wi in zip(prior, w))      # air yards
        exp_rec = rcw / wsum
        if exp_rec <= 0 or rcw <= 0:
            n_norec += 1
            continue
        ypr = yw / rcw
        ypr = max(YPR_BOUNDS[0], min(YPR_BOUNDS[1], ypr))
        adot = (ayw / tw) if tw > 0 else None
        if adot is not None:
            adot = max(ADOT_BOUNDS[0], min(ADOT_BOUNDS[1], adot))
        ry_hist = [r[3] for r in prior if r[3] is not None]
        line = d.get("OVER", d.get("UNDER", (None, None)))[0]
        if line is None:
            continue
        emp_over = (sum(1 for ry in ry_hist if ry > line) / len(ry_hist)
                    if ry_hist else 0.5)
        over, under = d.get("OVER"), d.get("UNDER")
        obs.append({
            "season": season, "week": week, "player": player, "line": line,
            "over_price": over[1] if over else None,
            "under_price": under[1] if under else None,
            "actual": float(actual),
            "exp_rec": exp_rec, "ypr": ypr, "adot": adot, "rcw": rcw,
            "emp_over": emp_over, "n_prior": len(prior),
        })
    meta = {"n_props": n_props, "n_obs": len(obs), "n_nogame": n_nogame,
            "n_void": n_void, "n_played0": n_played0, "n_thin": n_thin,
            "n_norec": n_norec,
            "seasons": sorted({g.split('_')[0] for g in idx})}
    return obs, meta


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


def _fit_league_ypr(train):
    rc = sum(o["rcw"] for o in train)
    if rc <= 0:
        return sum(o["ypr"] for o in train) / len(train) if train else 11.0
    return sum(o["ypr"] * o["rcw"] for o in train) / rc


def _fit_adot_ypr_curve(train):
    """Reception-weighted OLS  YPR ~ a + b*aDOT  on train (deeper target → higher YPR)."""
    pts = [(o["adot"], o["ypr"], o["rcw"]) for o in train
           if o["adot"] is not None and o["rcw"] > 0]
    if len(pts) < 50:
        return None
    sw = sum(w for _, _, w in pts)
    mx = sum(w * x for x, _, w in pts) / sw
    my = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
    if sxx <= 0:
        return None
    b = sxy / sxx
    return (my - b * mx, b)


def _ypr_shrink(o, prior_ypr):
    wt = o["rcw"] / (o["rcw"] + YPR_SHRINK_K)
    y = wt * o["ypr"] + (1 - wt) * prior_ypr
    return max(YPR_BOUNDS[0], min(YPR_BOUNDS[1], y))


def _mean_for(o, model, league_ypr, curve):
    if model in ("M1", "M2"):
        return o["exp_rec"] * o["ypr"]
    if model == "M3":
        return o["exp_rec"] * _ypr_shrink(o, league_ypr)
    if model == "M4":
        if curve is not None and o["adot"] is not None:
            a, b = curve
            prior = max(YPR_BOUNDS[0], min(YPR_BOUNDS[1], a + b * o["adot"]))
        else:
            prior = league_ypr
        return o["exp_rec"] * _ypr_shrink(o, prior)
    return None


def _fit_variance(train, model, league_ypr, curve):
    resid2, recs = [], []
    for o in train:
        m = _mean_for(o, model, league_ypr, curve)
        resid2.append((o["actual"] - m) ** 2)
        recs.append(o["exp_rec"])
    if model == "M1":
        return ("const", max(SD_FLOOR, math.sqrt(sum(resid2) / len(resid2))))
    X = [[1.0, c, c * c] for c in recs]
    A = [[0.0] * 3 for _ in range(3)]
    bvec = [0.0] * 3
    for xi, yi in zip(X, resid2):
        for i in range(3):
            bvec[i] += xi[i] * yi
            for j in range(3):
                A[i][j] += xi[i] * xi[j]
    coefs = _solve(A, bvec)
    if coefs is None:
        return ("const", max(SD_FLOOR, math.sqrt(sum(resid2) / len(resid2))))
    return ("quad", tuple(coefs))


def _sd(o, var):
    kind, p = var
    if kind == "const":
        return p
    a, b, c = p
    v = a + b * o["exp_rec"] + c * o["exp_rec"] ** 2
    return max(SD_FLOOR, math.sqrt(v)) if v > 0 else SD_FLOOR


def _predict(o, model, var, league_ypr, curve):
    if model == "M0":
        return max(0.0, min(1.0, o["emp_over"]))
    mean = _mean_for(o, model, league_ypr, curve)
    return max(0.0, min(1.0, 1.0 - _norm_cdf((o["line"] - mean) / _sd(o, var))))


def _brier(sub, fn):
    return sum((fn(o) - o["y"]) ** 2 for o in sub) / len(sub) if sub else None


MODELS = ["M0", "M1", "M2", "M3", "M4"]
LABELS = {"M0": "incumbent A", "M1": "naive Gauss", "M2": "vol-var Gauss",
          "M3": "+YPR shrink", "M4": "+aDOT YPR"}


def _fit_fold(train):
    league_ypr = _fit_league_ypr(train)
    curve = _fit_adot_ypr_curve(train)
    var = {m: _fit_variance(train, m, league_ypr, curve)
           for m in ("M1", "M2", "M3", "M4")}
    return league_ypr, curve, var


def probe(seasons, snapshot="closing", book="draftkings"):
    obs, meta = build_corpus(seasons, snapshot, book)
    print("=" * 100)
    print(f"  NFL RECEIVING YARDS process model — nested-layer PROBE (DK {snapshot}, {meta['seasons']})")
    print(f"  built {meta['n_obs']:,} obs  (played0={meta['n_played0']}, "
          f"void={meta['n_void']}, no-game={meta['n_nogame']}, thin<{MIN_PRIOR}="
          f"{meta['n_thin']}, no-rec={meta['n_norec']})")
    print("=" * 100)
    rows = []
    for o in obs:
        if abs(o["actual"] - o["line"]) < 1e-9:
            continue
        o["y"] = 1 if o["actual"] > o["line"] else 0
        if o["over_price"] is not None and o["under_price"] is not None:
            try:
                o["mkt_over"] = devig_two_way(
                    american_to_implied_prob(o["over_price"]),
                    american_to_implied_prob(o["under_price"]))[0]
            except Exception:
                o["mkt_over"] = None
        else:
            o["mkt_over"] = None
        rows.append(o)
    if len(rows) < 200:
        print(f"  Only {len(rows)} usable obs (<200) — too thin.")
        return
    rows.sort(key=lambda r: (r["season"], r["week"]))
    split = len(rows) // 2
    folds = [("fold A (train early → test late)", rows[:split], rows[split:]),
             ("fold B (train late → test early)", rows[split:], rows[:split])]

    print("\n  1) PREDICTION — OOS Brier (lower better), fit variance + YPR curve on train:")
    agg = {m: [0.0, 0] for m in MODELS}
    mkt_agg = [0.0, 0]
    for name, tr, te in folds:
        league_ypr, curve, var = _fit_fold(tr)
        cstr = (f"YPR≈{curve[0]:.1f}{curve[1]:+.2f}·aDOT" if curve else "curve=NA")
        print(f"     {name}:  league_YPR={league_ypr:.2f}   {cstr}   M1 sd={var['M1'][1]:.1f}")
        line = "        "
        for m in MODELS:
            b = _brier(te, lambda o, mm=m, vv=var, ly=league_ypr, cc=curve: _predict(
                o, mm, vv.get(mm), ly, cc))
            agg[m][0] += b * len(te)
            agg[m][1] += len(te)
            line += f"{LABELS[m]}={b:.4f}  "
        mk = [o for o in te if o["mkt_over"] is not None]
        mb = _brier(mk, lambda o: o["mkt_over"])
        if mb is not None:
            mkt_agg[0] += mb * len(mk)
            mkt_agg[1] += len(mk)
        line += f"market={mb:.4f}" if mb is not None else "market=NA"
        print(line)
    print("\n     pooled 2-fold OOS Brier:")
    base = agg["M0"][0] / agg["M0"][1]
    pooled = {m: agg[m][0] / agg[m][1] for m in MODELS}
    for m in MODELS:
        tag = "" if m == "M0" else f"   Δ vs A = {base - pooled[m]:+.4f}"
        print(f"        {LABELS[m]:<16} {pooled[m]:.4f}{tag}")
    if mkt_agg[1]:
        print(f"        {'market':<16} {mkt_agg[0] / mkt_agg[1]:.4f}   (sharp reference)")
    best = min(("M1", "M2", "M3", "M4"), key=lambda m: pooled[m])

    print("\n  2) PREDICTION per season (fit on the other seasons):")
    for s in meta["seasons"]:
        te = [o for o in rows if o["season"] == s]
        tr = [o for o in rows if o["season"] != s]
        if len(te) < 50 or len(tr) < 200:
            continue
        league_ypr, curve, var = _fit_fold(tr)
        parts = []
        for m in MODELS:
            b = _brier(te, lambda o, mm=m, vv=var, ly=league_ypr, cc=curve: _predict(
                o, mm, vv.get(mm), ly, cc))
            parts.append(f"{m}={b:.4f}")
        mk = [o for o in te if o["mkt_over"] is not None]
        mb = _brier(mk, lambda o: o["mkt_over"])
        parts.append(f"mkt={mb:.4f}" if mb is not None else "mkt=NA")
        print(f"        {s}: n={len(te):>5}  " + "  ".join(parts))

    league_ypr, curve, var = _fit_fold(rows[:split])
    priced = [o for o in rows[split:] if o["mkt_over"] is not None]
    print(f"\n  3) EDGE context — best model ({LABELS[best]}) vs market on the held-out "
          f"half ({len(priced)} priced).")

    def _pb(o):
        return _predict(o, best, var[best], league_ypr, curve)

    def _edge_report(sub, label):
        if not sub:
            print(f"     {label:<20} (empty)")
            return
        roi, realized, implied = [], [], []
        for o in sub:
            p = _pb(o)
            if p >= o["mkt_over"]:
                win = o["y"] == 1; px = o["over_price"]
                realized.append(o["y"]); implied.append(o["mkt_over"])
            else:
                win = o["y"] == 0; px = o["under_price"]
                realized.append(1 - o["y"]); implied.append(1 - o["mkt_over"])
            d = american_to_decimal(px)
            roi.append((d - 1.0) if win else -1.0)
        n = len(roi)
        edge = sum(realized) / n - sum(implied) / n
        print(f"     {label:<20} n={n:>5}  win={sum(1 for r in roi if r>0)/n*100:4.1f}%  "
              f"ROI={sum(roi)/n*100:+6.2f}%  realized−implied={edge*100:+5.2f}%")

    for lo, hi, lbl in [(0.0, 0.03, "|edge| 0-3%"), (0.03, 0.06, "|edge| 3-6%"),
                        (0.06, 0.10, "|edge| 6-10%"), (0.10, 1.0, "|edge| >=10%")]:
        _edge_report([o for o in priced if lo <= abs(_pb(o) - o["mkt_over"]) < hi], lbl)
    print("=" * 100)
    print("  READ: M1→M2 variance-vs-volume fix; M2→M3 YPR self-shrink; M3→M4 aDOT drives")
    print("  yards-per-reception (depth = the 'underneath' for YARDAGE, not just catch).")
    print("  Any Mi below incumbent A ⇒ ships. Market Brier is the accuracy ceiling.")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h"])
    ap.add_argument("--book", default="draftkings")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    probe(seasons, snapshot=args.snapshot, book=args.book)


if __name__ == "__main__":
    main()
