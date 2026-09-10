"""nfl_receptions_process.py — PROCESS-first receptions model (the video's 3-layer thesis).

The first receptions POC (`nfl_props_probe.py`) was a WEAK test: it modeled
targets × catch-rate with a FIXED-n binomial and used raw recency targets as the
feature. Two documented flaws:
  * the fixed-n binomial mis-models receptions' dominant TARGET-COUNT variance
    (NFL targets swing 4<->12/game) → variance too tight → over-confident;
  * it skipped LAYER 2 (quality/context): a screen target and a go route are not
    the same opportunity, but a single catch-rate lumps them.

This rebuild separates the three layers the way the process thesis prescribes and,
critically, tests them as NESTED models so we can see exactly which layer earns its
keep (KEEP-IF-BETTER on OOS prediction — per the governing principle, we do NOT
require beating the close; edge is realized live by the soft-line logger):

  L1 VOLUME    : expected targets. Two projections compared —
                 raw   = recency-weighted player targets;
                 decomp= recency-weighted target_share × recency-weighted team targets
                         (role × pace — each more stable than their product).
  L2 QUALITY   : aDOT (receiving_air_yards / targets). A deep target converts lower
                 than a screen; catch rate is shrunk toward a league catch(aDOT) curve
                 fit on train, so depth informs conversion instead of raw hit-rate.
  L3 CONVERSION: receptions | targets. The COUNT-VARIANCE FIX — receptions is a
                 thinned target count, so it inherits target overdispersion; we model
                 it as a NegBin (variance = mean + phi*mean^2, phi fit on train)
                 instead of a fixed-n binomial. (Poisson thinning of a NegBin target
                 count is NegBin, so this is the right family, not a hack.)

Nested models scored (all → P(receptions >= k), k=int(line)+1):
  M0 incumbent A   : empirical over-rate at the line over prior weeks (baseline).
  M1 raw binomial  : the OLD probe model (fixed-n binomial, raw targets) — reference.
  M2 raw NegBin    : mean = raw_targets × catch, NegBin phi — isolates the COUNT FIX.
  M3 decomp NegBin : mean = (share × team) × catch, NegBin phi — isolates L1 decomp.
  M4 +depth NegBin : M3 with catch shrunk toward league catch(aDOT) — isolates L2.
  market           : de-vigged two-way (the sharp reference).

Leakage-safe: every feature uses only strictly-earlier weeks in the same season; the
NegBin phi and the league catch(aDOT) curve are fit on the chronological train half
and scored on the held-out half (then swapped for a 2-fold confirm). Diagnostic only —
writes nothing. Reuses nfl_props_scan loaders (quote-time + participation-safe grading).
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
import nfl_data
import nfl_props_scan as scan
from stats import hits_at_least, negbin_at_least, fit_negbin_dispersion
from odds_client import (american_to_decimal, american_to_implied_prob,
                         devig_two_way)

SPORT = "americanfootball_nfl"
PROP = "player_receptions"

HALF_LIFE = 4              # weeks — recency weight on prior-week features
MIN_PRIOR = 3             # need >= this many prior-week rows to project
CATCH_BOUNDS = (0.20, 0.95)
CATCH_SHRINK_K = 12.0      # pseudo-targets pulling own catch toward the aDOT prior
ADOT_BOUNDS = (0.5, 20.0)


def _recency_weights(n, half_life):
    if n <= 0:
        return []
    if not half_life or half_life <= 0:
        return [1.0] * n
    return [0.5 ** (i / half_life) for i in range(n)]   # index 0 = most recent


def _player_series(seasons):
    """{(norm_name, season): [(week, targets, receptions, air_yards, target_share,
    team), ...] most-recent-first}, {(team, season, week): team_total_targets}, and
    {(team, season, week): attempt-weighted team passing CPOE} (QB accuracy context)."""
    pw = nfl_data.player_week([str(s) for s in seasons])
    if pw is None:
        return {}, {}, {}
    cols = set(pw.columns)
    namecol = "player_display_name" if "player_display_name" in cols else "player_name"

    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    series = defaultdict(list)
    team_tgts = defaultdict(float)      # (team, season, week) -> sum targets
    cpoe_num = defaultdict(float)       # (team, season, week) -> sum(cpoe*attempts)
    cpoe_den = defaultdict(float)       # (team, season, week) -> sum attempts
    for r in pw.to_dict("records"):
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        season = str(r.get("season"))
        team = r.get("team")
        tg = num(r.get("targets"))
        if team and tg:
            team_tgts[(team, season, str(wk))] += tg
        att = num(r.get("attempts"))
        cpoe = num(r.get("passing_cpoe"))
        if team and att and att > 0 and cpoe is not None:
            cpoe_num[(team, season, str(wk))] += cpoe * att
            cpoe_den[(team, season, str(wk))] += att
        nm = scan._norm(r.get(namecol))
        if not nm:
            continue
        series[(nm, season)].append(
            (wk, tg, num(r.get("receptions")), num(r.get("receiving_air_yards")),
             num(r.get("target_share")), team))
    for k in series:
        series[k].sort(key=lambda x: -x[0])
    team_cpoe = {k: cpoe_num[k] / cpoe_den[k] for k in cpoe_den if cpoe_den[k] > 0}
    return series, team_tgts, team_cpoe


def build_corpus(seasons, snapshot="closing", book="draftkings"):
    props = scan._load_props(seasons, snapshot, book)
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    pw_stat = scan._player_week_index(seasons)
    snaps = scan._snap_presence([str(s) for s in seasons])
    series, team_tgts, team_cpoe = _player_series(seasons)

    obs = []
    n_props = n_nogame = n_void = n_played0 = n_thin = n_notargets = 0
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
        actual = (pw_stat.get((nm, season, str(week))) or {}).get("receptions")
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
        # ── L1 raw: recency targets ──
        tw = sum(wi * (r[1] or 0.0) for r, wi in zip(prior, w))
        rw = sum(wi * (r[2] or 0.0) for r, wi in zip(prior, w))
        exp_tg_raw = tw / wsum
        if exp_tg_raw <= 0:
            n_notargets += 1
            continue
        # ── L1 decomp: recency target_share × recency team targets ──
        ts_num = sum(wi * r[4] for r, wi in zip(prior, w) if r[4] is not None)
        ts_den = sum(wi for r, wi in zip(prior, w) if r[4] is not None)
        exp_ts = (ts_num / ts_den) if ts_den else None
        team = prior[0][5]
        team_prior = [(wk, team_tgts.get((team, season, str(wk)), 0.0))
                      for wk in range(1, week) if (team, season, str(wk)) in team_tgts]
        if team_prior and exp_ts is not None:
            team_prior.sort(key=lambda x: -x[0])
            tw2 = _recency_weights(len(team_prior), HALF_LIFE)
            exp_team_tg = (sum(wi * v for (_wk, v), wi in zip(team_prior, tw2))
                           / (sum(tw2) or 1.0))
            exp_tg_decomp = exp_ts * exp_team_tg
        else:
            exp_tg_decomp = exp_tg_raw     # fall back to raw when share/team missing
        # ── L2: aDOT (air yards per target) ──
        ay = sum(wi * (r[3] or 0.0) for r, wi in zip(prior, w))
        adot = (ay / tw) if tw > 0 else None
        if adot is not None:
            adot = max(ADOT_BOUNDS[0], min(ADOT_BOUNDS[1], adot))
        # ── L2: QB accuracy context (recency-weighted team CPOE over prior weeks) ──
        cpoe_prior = [(wk, team_cpoe[(team, season, str(wk))])
                      for wk in range(1, week) if (team, season, str(wk)) in team_cpoe]
        if cpoe_prior:
            cpoe_prior.sort(key=lambda x: -x[0])
            cw = _recency_weights(len(cpoe_prior), HALF_LIFE)
            exp_cpoe = sum(wi * v for (_wk, v), wi in zip(cpoe_prior, cw)) / (sum(cw) or 1.0)
        else:
            exp_cpoe = None
        # ── L3: own catch rate ──
        catch = rw / tw if tw > 0 else 0.0
        catch = max(CATCH_BOUNDS[0], min(CATCH_BOUNDS[1], catch))
        rec_hist = [r[2] for r in prior if r[2] is not None]
        line = d.get("OVER", d.get("UNDER", (None, None)))[0]
        if line is None:
            continue
        emp_over = (sum(1 for rc in rec_hist if rc > line) / len(rec_hist)
                    if rec_hist else 0.5)
        over, under = d.get("OVER"), d.get("UNDER")
        obs.append({
            "season": season, "week": week, "player": player, "line": line,
            "over_price": over[1] if over else None,
            "under_price": under[1] if under else None,
            "actual": float(actual),
            "exp_tg_raw": exp_tg_raw, "exp_tg_decomp": exp_tg_decomp,
            "catch": catch, "adot": adot, "cpoe": exp_cpoe, "tw": tw,
            "emp_over": emp_over, "n_prior": len(prior),
        })
    meta = {"n_props": n_props, "n_obs": len(obs), "n_nogame": n_nogame,
            "n_void": n_void, "n_played0": n_played0, "n_thin": n_thin,
            "n_notargets": n_notargets,
            "seasons": sorted({g.split('_')[0] for g in idx})}
    return obs, meta


def _adjacent_binom(k, m, catch):
    lo = int(m)
    w_hi = m - lo
    surv = lambda nn: hits_at_least(k, nn, catch) if nn >= 1 else 0.0
    return (1.0 - w_hi) * surv(lo) + w_hi * surv(lo + 1)


def _fit_adot_catch_curve(train):
    """OLS catch_own ~ a + b*aDOT on train (target-weighted). Returns (a, b) or None.
    Captures the league tendency for deeper average depth to convert lower."""
    pts = [(o["adot"], o["catch"], o["tw"]) for o in train
           if o["adot"] is not None and o["tw"] > 0]
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
    a = my - b * mx
    return (a, b)


def _depth_catch(o, curve):
    """Shrink own catch toward the aDOT-implied league catch (more targets → trust own)."""
    if curve is None or o["adot"] is None:
        return o["catch"]
    a, b = curve
    prior_catch = max(CATCH_BOUNDS[0], min(CATCH_BOUNDS[1], a + b * o["adot"]))
    wt = o["tw"] / (o["tw"] + CATCH_SHRINK_K)
    c = wt * o["catch"] + (1 - wt) * prior_catch
    return max(CATCH_BOUNDS[0], min(CATCH_BOUNDS[1], c))


def _solve(A, b):
    """Gaussian elimination with partial pivot for a small dense system A x = b."""
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


def _fit_catch_model_mv(train):
    """Target-weighted OLS  catch_own ~ a + b_adot*aDOT + b_cpoe*teamCPOE  on train.
    Models the conversion rate from FUNDAMENTALS (depth + QB accuracy) instead of the
    player's own catch history — the 'something underneath' the stat. Returns
    {'coefs':(a,b_adot,b_cpoe), 'cpoe_mean':float} or None (falls back to the aDOT curve)."""
    pts = [(o["adot"], o["cpoe"], o["catch"], o["tw"]) for o in train
           if o["adot"] is not None and o["cpoe"] is not None and o["tw"] > 0]
    if len(pts) < 100:
        return None
    sw = sum(w for *_, w in pts)
    cpoe_mean = sum(w * cp for _ad, cp, _y, w in pts) / sw
    # weighted normal equations for design [1, aDOT, cpoe]
    X = [[1.0, ad, cp] for ad, cp, _y, _w in pts]
    ys = [y for *_, y, _w in pts]
    ws = [w for *_, w in pts]
    A = [[0.0] * 3 for _ in range(3)]
    bvec = [0.0] * 3
    for xi, yi, wi in zip(X, ys, ws):
        for i in range(3):
            bvec[i] += wi * xi[i] * yi
            for j in range(3):
                A[i][j] += wi * xi[i] * xi[j]
    coefs = _solve(A, bvec)
    if coefs is None:
        return None
    return {"coefs": tuple(coefs), "cpoe_mean": cpoe_mean}


def _fund_catch(o, mv, curve):
    """Predicted catch rate purely from fundamentals (aDOT + QB CPOE). Substitutes the
    train-mean CPOE when a game's QB context is missing; falls back to the aDOT curve
    (then own catch) when the mv model is unavailable."""
    if mv is None or o["adot"] is None:
        if curve is not None and o["adot"] is not None:
            a, b = curve
            return max(CATCH_BOUNDS[0], min(CATCH_BOUNDS[1], a + b * o["adot"]))
        return o["catch"]
    a, b_ad, b_cp = mv["coefs"]
    cp = o["cpoe"] if o["cpoe"] is not None else mv["cpoe_mean"]
    pred = a + b_ad * o["adot"] + b_cp * cp
    return max(CATCH_BOUNDS[0], min(CATCH_BOUNDS[1], pred))


def _blend_fund_catch(o, mv, curve):
    """M5b: shrink the player's own catch toward the multivariate fundamentals prior."""
    fund = _fund_catch(o, mv, curve)
    wt = o["tw"] / (o["tw"] + CATCH_SHRINK_K)
    c = wt * o["catch"] + (1 - wt) * fund
    return max(CATCH_BOUNDS[0], min(CATCH_BOUNDS[1], c))


def _k(line):
    return int(line) + 1


def _predict(o, model, phi, curve, mv=None):
    line = o["line"]
    k = _k(line)
    if model == "M0":
        return max(0.0, min(1.0, o["emp_over"]))
    if model == "M1":                       # fixed-n binomial, raw targets (old probe)
        return max(0.0, min(1.0, _adjacent_binom(k, o["exp_tg_raw"], o["catch"])))
    mean = _mean_for(o, model, curve, mv)
    return negbin_at_least(k, mean, phi)


def _mean_for(o, model, curve, mv=None):
    if model == "M2":                       # NegBin, raw-target mean
        return o["exp_tg_raw"] * o["catch"]
    if model == "M3":                       # NegBin, decomposed volume
        return o["exp_tg_decomp"] * o["catch"]
    if model == "M4":                       # decomposed volume + aDOT depth catch
        return o["exp_tg_decomp"] * _depth_catch(o, curve)
    if model == "M5f":                      # catch PURELY from fundamentals (de-anchored)
        return o["exp_tg_decomp"] * _fund_catch(o, mv, curve)
    if model == "M5b":                      # fundamentals prior + light own shrink
        return o["exp_tg_decomp"] * _blend_fund_catch(o, mv, curve)
    return None


def _brier(sub, fn):
    return sum((fn(o) - o["y"]) ** 2 for o in sub) / len(sub) if sub else None


def _fit_fold(train):
    """Fit the aDOT curve + multivariate fundamentals catch model, then the NegBin phi
    per model on the train fold."""
    curve = _fit_adot_catch_curve(train)
    mv = _fit_catch_model_mv(train)
    phis = {}
    for model in ("M2", "M3", "M4", "M5f", "M5b"):
        pairs = [(_mean_for(o, model, curve, mv), o["actual"]) for o in train]
        phis[model] = fit_negbin_dispersion(pairs, cap=2.0)
    return curve, mv, phis


MODELS = ["M0", "M1", "M2", "M3", "M4", "M5f", "M5b"]
LABELS = {"M0": "incumbent A", "M1": "raw binomial", "M2": "raw NegBin",
          "M3": "decomp NegBin", "M4": "+depth NegBin", "M5f": "fund catch (pure)",
          "M5b": "fund catch (blend)"}


def probe(seasons, snapshot="closing", book="draftkings"):
    obs, meta = build_corpus(seasons, snapshot, book)
    print("=" * 100)
    print(f"  NFL RECEPTIONS process model — nested-layer PROBE (DK {snapshot}, {meta['seasons']})")
    print(f"  built {meta['n_obs']:,} obs  (played0={meta['n_played0']}, "
          f"void={meta['n_void']}, no-game={meta['n_nogame']}, thin<{MIN_PRIOR}="
          f"{meta['n_thin']}, no-tgts={meta['n_notargets']})")
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

    # ── 1) PREDICTION: nested models, 2-fold OOS Brier ──
    print("\n  1) PREDICTION — OOS Brier (lower better), fit phi + depth curve on train:")
    agg = {m: [0.0, 0] for m in MODELS}      # sum sq err, n  (pooled over folds' tests)
    mkt_agg = [0.0, 0]
    for name, tr, te in folds:
        curve, mv, phis = _fit_fold(tr)
        cstr = (f"catch(aDOT)={curve[0]:.3f}{curve[1]:+.4f}·aDOT" if curve else "curve=NA")
        mvstr = (f"catch~{mv['coefs'][0]:.3f}{mv['coefs'][1]:+.4f}·aDOT"
                 f"{mv['coefs'][2]:+.4f}·CPOE" if mv else "mv=NA")
        print(f"     {name}:  {cstr}   {mvstr}")
        line = "        "
        for m in MODELS:
            b = _brier(te, lambda o, mm=m, cc=curve, mvv=mv, pp=phis: _predict(
                o, mm, pp.get(mm, 0.0), cc, mvv))
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
        print(f"        {LABELS[m]:<18} {pooled[m]:.4f}{tag}")
    if mkt_agg[1]:
        print(f"        {'market':<18} {mkt_agg[0] / mkt_agg[1]:.4f}   (sharp reference)")
    best = min(("M2", "M3", "M4", "M5f", "M5b"), key=lambda m: pooled[m])

    # ── 2) per-season Brier (all obs, fit on the OTHER seasons) ──
    print("\n  2) PREDICTION per season (fit phi+curve on the other seasons):")
    for s in meta["seasons"]:
        te = [o for o in rows if o["season"] == s]
        tr = [o for o in rows if o["season"] != s]
        if len(te) < 50 or len(tr) < 200:
            continue
        curve, mv, phis = _fit_fold(tr)
        parts = []
        for m in MODELS:
            b = _brier(te, lambda o, mm=m, cc=curve, mvv=mv, pp=phis: _predict(
                o, mm, pp.get(mm, 0.0), cc, mvv))
            parts.append(f"{m}={b:.4f}")
        mk = [o for o in te if o["mkt_over"] is not None]
        mb = _brier(mk, lambda o: o["mkt_over"])
        parts.append(f"mkt={mb:.4f}" if mb is not None else "mkt=NA")
        print(f"        {s}: n={len(te):>5}  " + "  ".join(parts))

    # ── 3) EDGE context (NOT a gate — informs the live logger's detection band) ──
    curve, mv, phis = _fit_fold(rows[:split])
    priced = [o for o in rows[split:] if o["mkt_over"] is not None]
    print(f"\n  3) EDGE context — best model ({LABELS[best]}) vs market on the held-out "
          f"half ({len(priced)} priced).")
    print("     NOT a ship gate (edge is realized live); shows how often an accurate")
    print("     model would disagree with the close and whether those disagreements win.")

    def _predict_best(o):
        return _predict(o, best, phis[best], curve, mv)

    def _edge_report(sub, label):
        if not sub:
            print(f"     {label:<20} (empty)")
            return
        roi, realized, implied = [], [], []
        for o in sub:
            p = _predict_best(o)
            if p >= o["mkt_over"]:
                win = o["y"] == 1
                px = o["over_price"]
                realized.append(o["y"]); implied.append(o["mkt_over"])
            else:
                win = o["y"] == 0
                px = o["under_price"]
                realized.append(1 - o["y"]); implied.append(1 - o["mkt_over"])
            d = american_to_decimal(px)
            roi.append((d - 1.0) if win else -1.0)
        n = len(roi)
        edge = sum(realized) / n - sum(implied) / n
        print(f"     {label:<20} n={n:>5}  win={sum(1 for r in roi if r>0)/n*100:4.1f}%  "
              f"ROI={sum(roi)/n*100:+6.2f}%  realized−implied={edge*100:+5.2f}%")

    for lo, hi, lbl in [(0.0, 0.03, "|edge| 0-3%"), (0.03, 0.06, "|edge| 3-6%"),
                        (0.06, 0.10, "|edge| 6-10%"), (0.10, 1.0, "|edge| >=10%")]:
        _edge_report([o for o in priced
                      if lo <= abs(_predict_best(o) - o["mkt_over"]) < hi], lbl)
    print("=" * 100)
    print("  READ: any Mi below incumbent A ⇒ that layer improves the model → SHIP it")
    print("  (governing principle: better prediction ships regardless of the close).")
    print("  Layers: M1→M2 count-variance fix; M2→M3 volume decomp; M3→M4 aDOT depth;")
    print("  M4→M5f/M5b conversion from FUNDAMENTALS (aDOT+QB CPOE). M5f≈M4 ⇒ we can")
    print("  DE-ANCHOR from the player's own catch history with no loss (the video's")
    print("  'something underneath' the stat). Market Brier is the accuracy ceiling.")


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
