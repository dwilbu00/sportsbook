"""nfl_rushing_yds_process.py — PROCESS-first rushing-yards model (template carry-over).

Carries the receptions template (`nfl_receptions_process.py`) to a YARDAGE market.
rushing_yds is COMPOUND: carries (an overdispersed count, exactly like targets) ×
yards-per-carry (its own distribution with a fat right tail from breakaway runs). So
the "count-variance fix" that was the giant lever for receptions takes a different
FORM here: the naive model uses a CONSTANT standard deviation; the correct model lets
the variance GROW WITH VOLUME —

    Var(total) ≈ E[carries]·Var(ypc)  +  Var(carries)·ypc²
                 └ linear in E[carries] ┘   └ quadratic in E[carries] ┘

which we fit empirically as Var ≈ a + b·E[carries] + c·E[carries]² on train (the
per-game data has no per-carry split, but the quadratic-in-volume shape captures both
terms). Testing whether that beats the constant-sd Gaussian is the headline
generalization test — the direct analog of receptions' NegBin win.

Nested models scored (all → P(rush_yds > line) via a Normal with the model's mean/sd):
  M0 incumbent A   : empirical over-rate at the line over prior weeks (baseline).
  M1 naive Gauss   : mean = raw_carries × ypc, CONSTANT pooled sd — the naive analog.
  M2 vol-var Gauss : same mean, variance GROWS with volume (a+b·c+c·c²) — the FIX.
  M3 decomp vol    : M2 with carries = carry_share × team carries (role × pace).
  M4 +ypc shrink   : M3 with ypc shrunk toward league ypc (robust cold-start — the
                     self-shrink "M5b" conversion carried over from receptions).
  market           : de-vigged two-way (the sharp reference).

Layer-2 "quality" for rushing (box count / game script / O-line) is NOT in the
player-week layer, so the conversion robustness here is the ypc self-shrink rather
than a fundamentals model — box count/game script noted as a future quality layer.

Same DEFAULTS as receptions (HALF_LIFE, MIN_PRIOR, shrink K) — we test whether the
template generalizes on identical knobs BEFORE any hyperparameter sweep (a real sweep
needs nested CV; tuning on the eval folds would leak). Leakage-safe: features use only
strictly-earlier weeks; variance coeffs + league ypc fit on the train fold, scored on
the held-out half (2-fold). Diagnostic only — writes nothing. Reuses nfl_props_scan
loaders (quote-time + participation-safe grading).
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
PROP = "player_rush_yds"

HALF_LIFE = 4              # weeks — shared default with receptions
MIN_PRIOR = 3
YPC_BOUNDS = (1.5, 8.0)    # sane per-carry yards
YPC_SHRINK_K = 12.0        # pseudo-carries pulling own ypc toward league (self-shrink)
SD_FLOOR = 8.0             # yards — never let modeled sd collapse


def _recency_weights(n, half_life):
    if n <= 0:
        return []
    if not half_life or half_life <= 0:
        return [1.0] * n
    return [0.5 ** (i / half_life) for i in range(n)]


def _player_series(seasons):
    """{(norm_name, season): [(week, carries, rush_yds, team), ...] most-recent-first}
    and {(team, season, week): team_total_carries}."""
    pw = nfl_data.player_week([str(s) for s in seasons])
    if pw is None:
        return {}, {}
    cols = set(pw.columns)
    namecol = "player_display_name" if "player_display_name" in cols else "player_name"

    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    series = defaultdict(list)
    team_car = defaultdict(float)
    for r in pw.to_dict("records"):
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        season = str(r.get("season"))
        team = r.get("team")
        car = num(r.get("carries"))
        if team and car:
            team_car[(team, season, str(wk))] += car
        nm = scan._norm(r.get(namecol))
        if not nm:
            continue
        series[(nm, season)].append(
            (wk, car, num(r.get("rushing_yards")), team))
    for k in series:
        series[k].sort(key=lambda x: -x[0])
    return series, team_car


def build_corpus(seasons, snapshot="closing", book="draftkings"):
    props = scan._load_props(seasons, snapshot, book)
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    pw_stat = scan._player_week_index(seasons)
    snaps = scan._snap_presence([str(s) for s in seasons])
    series, team_car = _player_series(seasons)

    obs = []
    n_props = n_nogame = n_void = n_played0 = n_thin = n_nocarries = 0
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
        actual = (pw_stat.get((nm, season, str(week))) or {}).get("rushing_yards")
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
        cw = sum(wi * (r[1] or 0.0) for r, wi in zip(prior, w))       # carries
        yw = sum(wi * (r[2] or 0.0) for r, wi in zip(prior, w))       # yards
        exp_car_raw = cw / wsum
        if exp_car_raw <= 0:
            n_nocarries += 1
            continue
        # decomp volume: recency carry_share × recency team carries
        team = prior[0][3]
        shares = []
        for r, wi in zip(prior, w):
            tc = team_car.get((team, season, str(r[0])))
            if tc and r[1] is not None and tc > 0:
                shares.append((wi, r[1] / tc))
        exp_share = (sum(wi * s for wi, s in shares) / sum(wi for wi, _ in shares)
                     if shares else None)
        team_prior = [(wk, team_car.get((team, season, str(wk)), 0.0))
                      for wk in range(1, week) if (team, season, str(wk)) in team_car]
        if team_prior and exp_share is not None:
            team_prior.sort(key=lambda x: -x[0])
            tw2 = _recency_weights(len(team_prior), HALF_LIFE)
            exp_team_car = (sum(wi * v for (_wk, v), wi in zip(team_prior, tw2))
                            / (sum(tw2) or 1.0))
            exp_car_decomp = exp_share * exp_team_car
        else:
            exp_car_decomp = exp_car_raw
        ypc = yw / cw if cw > 0 else 0.0
        ypc = max(YPC_BOUNDS[0], min(YPC_BOUNDS[1], ypc))
        ry_hist = [r[2] for r in prior if r[2] is not None]
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
            "exp_car_raw": exp_car_raw, "exp_car_decomp": exp_car_decomp,
            "ypc": ypc, "cw": cw, "emp_over": emp_over, "n_prior": len(prior),
        })
    meta = {"n_props": n_props, "n_obs": len(obs), "n_nogame": n_nogame,
            "n_void": n_void, "n_played0": n_played0, "n_thin": n_thin,
            "n_nocarries": n_nocarries,
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


def _fit_league_ypc(train):
    cw = sum(o["cw"] for o in train)
    if cw <= 0:
        return sum(o["ypc"] for o in train) / len(train) if train else 4.2
    return sum(o["ypc"] * o["cw"] for o in train) / cw     # carry-weighted


def _shrunk_ypc(o, league_ypc):
    wt = o["cw"] / (o["cw"] + YPC_SHRINK_K)
    y = wt * o["ypc"] + (1 - wt) * league_ypc
    return max(YPC_BOUNDS[0], min(YPC_BOUNDS[1], y))


def _mean_for(o, model, league_ypc):
    if model == "M1":
        return o["exp_car_raw"] * o["ypc"]
    if model == "M2":
        return o["exp_car_raw"] * o["ypc"]
    if model == "M3":
        return o["exp_car_decomp"] * o["ypc"]
    if model == "M4":
        return o["exp_car_decomp"] * _shrunk_ypc(o, league_ypc)
    return None


def _car_for(o, model):
    return o["exp_car_decomp"] if model in ("M3", "M4") else o["exp_car_raw"]


def _fit_variance(train, model, league_ypc):
    """M1 → constant sd (pooled residual std). M2/M3/M4 → variance grows with volume:
    OLS  resid² ~ a + b·E[carries] + c·E[carries]²  on train (clamped ≥ SD_FLOOR²)."""
    resid2, cars = [], []
    for o in train:
        m = _mean_for(o, model, league_ypc)
        resid2.append((o["actual"] - m) ** 2)
        cars.append(_car_for(o, model))
    if model == "M1":
        return ("const", max(SD_FLOOR, math.sqrt(sum(resid2) / len(resid2))))
    X = [[1.0, c, c * c] for c in cars]
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


def _sd(o, model, var):
    kind, p = var
    if kind == "const":
        return p
    a, b, c = p
    car = _car_for(o, model)
    v = a + b * car + c * car * car
    return max(SD_FLOOR, math.sqrt(v)) if v > 0 else SD_FLOOR


def _predict(o, model, var, league_ypc):
    if model == "M0":
        return max(0.0, min(1.0, o["emp_over"]))
    mean = _mean_for(o, model, league_ypc)
    sd = _sd(o, model, var)
    return max(0.0, min(1.0, 1.0 - _norm_cdf((o["line"] - mean) / sd)))


def _brier(sub, fn):
    return sum((fn(o) - o["y"]) ** 2 for o in sub) / len(sub) if sub else None


MODELS = ["M0", "M1", "M2", "M3", "M4"]
LABELS = {"M0": "incumbent A", "M1": "naive Gauss", "M2": "vol-var Gauss",
          "M3": "decomp vol", "M4": "+ypc shrink"}


def _fit_fold(train):
    league_ypc = _fit_league_ypc(train)
    var = {m: _fit_variance(train, m, league_ypc) for m in ("M1", "M2", "M3", "M4")}
    return league_ypc, var


def probe(seasons, snapshot="closing", book="draftkings"):
    obs, meta = build_corpus(seasons, snapshot, book)
    print("=" * 100)
    print(f"  NFL RUSHING YARDS process model — nested-layer PROBE (DK {snapshot}, {meta['seasons']})")
    print(f"  built {meta['n_obs']:,} obs  (played0={meta['n_played0']}, "
          f"void={meta['n_void']}, no-game={meta['n_nogame']}, thin<{MIN_PRIOR}="
          f"{meta['n_thin']}, no-carries={meta['n_nocarries']})")
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

    print("\n  1) PREDICTION — OOS Brier (lower better), fit variance + league ypc on train:")
    agg = {m: [0.0, 0] for m in MODELS}
    mkt_agg = [0.0, 0]
    for name, tr, te in folds:
        league_ypc, var = _fit_fold(tr)
        m2 = var["M2"][1]
        vstr = (f"Var≈{m2[0]:.0f}{m2[1]:+.1f}·car{m2[2]:+.2f}·car²"
                if var["M2"][0] == "quad" else "Var=const")
        print(f"     {name}:  league_ypc={league_ypc:.2f}   M2 {vstr}   "
              f"M1 sd={var['M1'][1]:.1f}")
        line = "        "
        for m in MODELS:
            b = _brier(te, lambda o, mm=m, vv=var, ly=league_ypc: _predict(
                o, mm, vv.get(mm), ly))
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
        league_ypc, var = _fit_fold(tr)
        parts = []
        for m in MODELS:
            b = _brier(te, lambda o, mm=m, vv=var, ly=league_ypc: _predict(
                o, mm, vv.get(mm), ly))
            parts.append(f"{m}={b:.4f}")
        mk = [o for o in te if o["mkt_over"] is not None]
        mb = _brier(mk, lambda o: o["mkt_over"])
        parts.append(f"mkt={mb:.4f}" if mb is not None else "mkt=NA")
        print(f"        {s}: n={len(te):>5}  " + "  ".join(parts))

    # ── 3) EDGE context (NOT a gate) ──
    league_ypc, var = _fit_fold(rows[:split])
    priced = [o for o in rows[split:] if o["mkt_over"] is not None]
    print(f"\n  3) EDGE context — best model ({LABELS[best]}) vs market on the held-out "
          f"half ({len(priced)} priced).")

    def _pb(o):
        return _predict(o, best, var[best], league_ypc)

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
    print("  READ: M1→M2 = the variance-vs-volume fix (the count-fix analog); if M2 beats")
    print("  the constant-sd M1 and closes the gap to market, the template GENERALIZES to")
    print("  yardage. M3 = volume decomp, M4 = ypc self-shrink (robust cold-start).")


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
