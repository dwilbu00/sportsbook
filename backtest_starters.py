"""
Fit the MLB `starter_adjustment` weights from HISTORICAL outcomes — no odds,
no waiting for live data. Uses:

  * StatsAPI  → games, final scores, probable/starting pitchers  (outcomes)
  * Statcast  → leakage-safe as-of xwOBAcon per starter          (features)

Objective is MODEL ACCURACY vs reality (not the market), so it costs zero Odds
API credits:
  * moneyline weight → 1-D logistic fit of home_win ~ starter_edge
  * run_scale        → 1-D OLS fit of total_runs ~ combined starter excess

v1 fits the Phase-1 team-market weights (moneyline, run_scale). Props/bullpen
weight fitting (needs batter xwOBA-vs-hand + prop box outcomes) is a planned v2
on the same cached data.

Usage:
    python backtest_starters.py --season 2024 --fetch      # slow one-time pull
    python backtest_starters.py --season 2024              # fit (days cached)
    python backtest_starters.py --season 2024 --save       # fit + write weights

Caveats (documented, not hidden):
  * Uses the *probable* starter from the schedule (≈ actual; late scratches
    are rare but unmodeled).
  * As-of pitcher quality uses xwOBAcon (contact), a slightly different measure
    than the live season xERA index; both are Savant x-stats and correlated.
  * The isolated logistic coefficient can overlap with signal team form already
    captures, so treat the fitted moneyline weight as a mild upper bound and
    keep it bounded.
"""

import argparse
import json
import math
import os

import mlb_starters
import savant_history as sh
from calibration_loader import load_starter_adjustment, save_starter_adjustment

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _season_bounds(season):
    # MLB regular season ≈ late March → early October. Generous bounds.
    return f"{season}-03-20", f"{season}-10-05"


def get_season_games(season, start=None, end=None, verbose=True):
    """Fetch Final games with starters + scores, day-by-day (cached per season)."""
    s0, s1 = _season_bounds(season)
    start, end = start or s0, end or s1
    # v2 adds the home-perspective run margin (for spread/run-line fitting).
    path = os.path.join(CACHE_DIR, f"season_games_v2_{season}_{start}_{end}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    from datetime import date, timedelta
    games = []
    day = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while day <= last:
        ds = day.isoformat()
        try:
            data = mlb_starters._get("schedule", {
                "sportId": 1, "date": ds, "hydrate": "probablePitcher,linescore"})
        except Exception:
            data = {"dates": []}
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("detailedState") != "Final":
                    continue
                h, a = g["teams"]["home"], g["teams"]["away"]
                hs, as_ = h.get("score"), a.get("score")
                h_sp = (h.get("probablePitcher") or {}).get("id")
                a_sp = (a.get("probablePitcher") or {}).get("id")
                if hs is None or as_ is None or not h_sp or not a_sp:
                    continue
                games.append({
                    "date": ds,
                    "home_sp": h_sp, "away_sp": a_sp,
                    "home_win": 1 if hs > as_ else 0,
                    "total_runs": hs + as_,
                    "margin": hs - as_,  # home perspective (for spreads)
                })
        day += timedelta(days=1)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(games, f)
    if verbose:
        print(f"season {season}: {len(games)} Final games with starters")
    return games


def _bullpen_features(rows, season):
    """Leakage-safe as-of BULLPEN quality per team.

    A team's bullpen = every pitch it threw that day EXCEPT its own starter's.
    We identify the pitching team + starter for each Statcast row via the cached
    schedule (batting_team → that day's opponent + opposing probable starter),
    then index the non-starter (relief) batted balls by pitching team with
    prefix sums (AsOfIndex) for O(log n) as-of means.

    Returns (bullpen_index, league_bullpen_xwoba, game_team_map) where
    game_team_map keys (date, home_sp, away_sp) → (home_abbr, away_abbr) so a
    season_games row can be matched to its two teams.
    """
    from backtest_props import season_schedule, AsOfIndex

    sched = season_schedule(season)
    bat2pitch = {}   # (date, batting_team) -> (pitching_team, pitching_starter)
    game_teams = {}  # (date, home_sp, away_sp) -> (home_abbr, away_abbr)
    for date, gs in sched.items():
        for g in gs:
            ha, aa = g.get("home_abbr"), g.get("away_abbr")
            if not ha or not aa:
                continue
            bat2pitch[(date, ha)] = (aa, str(g.get("away_sp")))
            bat2pitch[(date, aa)] = (ha, str(g.get("home_sp")))
            game_teams[(date, str(g.get("home_sp")), str(g.get("away_sp")))] = (ha, aa)

    idx = AsOfIndex()
    bp_vals = []
    for r in rows:
        x = r.get("xwoba")
        if x is None:
            continue
        pt = bat2pitch.get((r.get("game_date"), r.get("batting_team")))
        if not pt:
            continue
        pitching_team, starter = pt
        if str(r.get("pitcher")) == starter:
            continue  # starter's pitch — not the bullpen
        idx.add(pitching_team, r["game_date"], x)
        bp_vals.append(x)
    league_bp = sum(bp_vals) / len(bp_vals) if bp_vals else 0.0
    return idx, league_bp, game_teams


def _ip_index(season):
    """As-of average innings/start per starter, from cached pitcher gameLogs.

    Values are per-game OUTS (IP*3); the caller divides by 3. Leakage-safe:
    asof_mean only averages starts strictly before the game date.
    """
    from backtest_props import gamelog, AsOfIndex, starter_ids

    idx = AsOfIndex()
    for pid in starter_ids([season]):
        for sp in gamelog(pid, season, "pitching"):
            outs = sp.get("stat", {}).get("outs")
            date = sp.get("date")
            if outs is not None and date:
                try:
                    idx.add(str(pid), date, float(outs))
                except (TypeError, ValueError):
                    pass
    return idx


def build_dataset(games, season):
    """Attach leakage-safe as-of starter + bullpen features to each game.

    Returns rows with starter_edge, combined_excess, bullpen_excess (may be
    None when a team lacks enough as-of relief data), home_win, total_runs.
    Games missing an as-of estimate for either starter are skipped.
    """
    s0, s1 = _season_bounds(season)
    rows = sh.load_days(s0, s1)
    if not rows:
        raise RuntimeError("No Statcast days cached — run with --fetch first.")

    # League baseline xwOBAcon (same measure as the per-pitcher estimate) so the
    # run_suppression ratio is well-centered.
    league_vals = [r["xwoba"] for r in rows if r["xwoba"] is not None]
    league_xwoba = sum(league_vals) / len(league_vals)

    bp_idx, league_bp, game_teams = _bullpen_features(rows, season)

    # As-of starter index (prefix sums) — identical result to
    # sh.asof_pitcher_xwoba but O(log n) instead of a full scan per game.
    from backtest_props import build_pitcher_index
    sp_idx = build_pitcher_index(rows)
    ip_idx = _ip_index(season)  # as-of avg innings/start per starter

    out = []
    for g in games:
        hx = sp_idx.asof_mean(str(g["home_sp"]), g["date"])
        ax = sp_idx.asof_mean(str(g["away_sp"]), g["date"])
        if hx is None or ax is None:
            continue
        # run_suppression: lower xwOBA allowed = better pitcher = >1 (bounded).
        h_rs = max(0.5, min(2.0, league_xwoba / hx))
        a_rs = max(0.5, min(2.0, league_xwoba / ax))

        # Bullpen suppression per team (same league-relative form as starters).
        h_bs = a_bs = None
        bullpen_excess = None
        gt = game_teams.get((g["date"], str(g["home_sp"]), str(g["away_sp"])))
        if gt and league_bp:
            ha, aa = gt
            h_bp = bp_idx.asof_mean(ha, g["date"])
            a_bp = bp_idx.asof_mean(aa, g["date"])
            if h_bp and a_bp:
                h_bs = max(0.5, min(2.0, league_bp / h_bp))
                a_bs = max(0.5, min(2.0, league_bp / a_bp))
                bullpen_excess = (h_bs - 1.0) + (a_bs - 1.0)

        # As-of average innings/start (min 3 prior starts), None if unknown.
        h_outs = ip_idx.asof_mean(str(g["home_sp"]), g["date"], min_bbe=3)
        a_outs = ip_idx.asof_mean(str(g["away_sp"]), g["date"], min_bbe=3)
        h_ip = h_outs / 3.0 if h_outs is not None else None
        a_ip = a_outs / 3.0 if a_outs is not None else None

        out.append({
            "starter_edge": math.tanh(h_rs - a_rs),
            "combined_excess": (h_rs - 1.0) + (a_rs - 1.0),
            "bullpen_excess": bullpen_excess,
            "home_win": g["home_win"],
            "total_runs": g["total_runs"],
            # extra raw fields for the innings-weighting A/B (fit() ignores them)
            "h_sp_sup": h_rs, "a_sp_sup": a_rs,
            "h_bp_sup": h_bs, "a_bp_sup": a_bs,
            "h_ip": h_ip, "a_ip": a_ip,
            "margin": g.get("margin"),
        })
    return out, league_xwoba


# ── pure-python fitters ────────────────────────────────────────────────────
def fit_ols_1d(xs, ys):
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    var = sum((x - xbar) ** 2 for x in xs)
    if var == 0:
        return 0.0, ybar
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / var
    return slope, ybar - slope * xbar


def _sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def fit_logistic_1d(xs, ys, lr=0.1, iters=5000):
    """Fit P(y=1)=sigmoid(a + b*x) via gradient descent (x standardized)."""
    n = len(xs)
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / n) ** 0.5 or 1.0
    z = [(x - mu) / sd for x in xs]
    a, b = 0.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for zi, yi in zip(z, ys):
            err = _sigmoid(a + b * zi) - yi
            ga += err
            gb += err * zi
        a -= lr * ga / n
        b -= lr * gb / n
    # De-standardize coefficient back to raw-x scale.
    return b / sd, a - b * mu / sd


def brier(preds, ys):
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)


def rmse(preds, ys):
    return (sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)) ** 0.5


def build_pooled_dataset(seasons):
    """Pool leakage-safe features across multiple seasons.

    Each season is normalized against its OWN league xwOBAcon baseline (so
    era-to-era run-environment differences don't distort the league-relative
    run_suppression feature), then the resulting rows are concatenated.
    """
    pooled = []
    per_season = {}
    for s in seasons:
        games = get_season_games(s)
        data, lg = build_dataset(games, s)
        per_season[s] = {"games": len(data), "league_xwoba": lg}
        pooled.extend(data)
    return pooled, per_season


def _eff_edge(d):
    """Innings-weighted effective run-prevention edge (home − away), tanh-bounded.

    Mirrors mlb_starters.build_matchup_features._eff exactly: a starter's
    quality counts in proportion to how deep he goes, bullpen covers the rest;
    falls back to starter-quality-only when bullpen or innings are missing.
    """
    def eff(sp, bp, ip):
        if bp and ip:
            w = max(0.30, min(0.85, ip / 9.0))
            return w * sp + (1.0 - w) * bp
        return sp
    return math.tanh(eff(d["h_sp_sup"], d.get("h_bp_sup"), d.get("h_ip"))
                     - eff(d["a_sp_sup"], d.get("a_bp_sup"), d.get("a_ip")))


def fit(seasons, do_save=False):
    if isinstance(seasons, int):
        seasons = [seasons]
    seasons_str = ",".join(str(s) for s in seasons)
    data, per_season = build_pooled_dataset(seasons)
    league_xwoba = (sum(v["league_xwoba"] for v in per_season.values())
                    / len(per_season)) if per_season else 0.0
    for s, v in per_season.items():
        print(f"  season {s}: {v['games']} gradeable games "
              f"(league xwOBAcon {v['league_xwoba']:.3f})")
    if len(data) < 100:
        print(f"Only {len(data)} usable games total — need more cached Statcast days.")
        return
    # Innings-weighted effective-suppression edge — mirrors the live
    # build_matchup_features blend exactly (falls back to starter-quality-only
    # when a game lacks as-of bullpen or innings data).
    edge = [_eff_edge(d) for d in data]
    win = [d["home_win"] for d in data]
    exc = [d["combined_excess"] for d in data]
    tot = [d["total_runs"] for d in data]

    # Moneyline: home_win ~ eff_edge (logistic).
    b_ml, a_ml = fit_logistic_1d(edge, win)
    base_ml = sum(win) / len(win)
    base_brier = brier([base_ml] * len(win), win)
    feat_brier = brier([_sigmoid(a_ml + b_ml * e) for e in edge], win)

    # Spreads: margin ~ eff_edge (OLS). Slope = runs of margin per unit edge,
    # applied at runtime as pred_margin += spreads_w * edge.
    mar_data = [d for d in data if d.get("margin") is not None]
    spreads_w = 0.0
    sp_base_rmse = sp_feat_rmse = None
    if len(mar_data) >= 100:
        m_edge = [_eff_edge(d) for d in mar_data]
        mar = [d["margin"] for d in mar_data]
        sp_slope, sp_int = fit_ols_1d(m_edge, mar)
        # Clamp loose: eff_edge is tanh-compressed (p99 ≈ 0.36), so even slope 8
        # shifts margin < 3 runs on realistic games; guards only pathologies.
        spreads_w = max(0.0, min(8.0, sp_slope))
        sp_base_rmse = rmse([sum(mar) / len(mar)] * len(mar), mar)
        sp_feat_rmse = rmse([sp_int + sp_slope * e for e in m_edge], mar)

    # Totals: total_runs ~ combined_excess (OLS). Better arms → fewer runs.
    slope, intercept = fit_ols_1d(exc, tot)
    run_scale = -slope
    base_rmse = rmse([sum(tot) / len(tot)] * len(tot), tot)
    feat_rmse = rmse([intercept + slope * x for x in exc], tot)

    # Loosened from 1.5: the innings-weighted edge is tanh-compressed, so a
    # larger logit coef still yields a modest shift on realistic games (p99
    # edge 0.36 → 0.7 logits at coef 1.9).
    ml_weight = max(0.0, min(2.5, b_ml))
    run_scale_c = max(0.0, min(3.0, run_scale))

    # ── Bullpen: marginal fit on the totals RESIDUAL, holding run_scale fixed ──
    # The model applies bullpen additively on top of the starter shift, so we
    # fit the bullpen coefficient against what the starter-only model leaves on
    # the table (keeps the already-validated run_scale untouched).
    bp_data = [d for d in data if d.get("bullpen_excess") is not None]
    bullpen_w = 0.0
    bp_base_rmse = bp_feat_rmse = None
    if len(bp_data) >= 100:
        bexc = [d["bullpen_excess"] for d in bp_data]
        bp_tot = [d["total_runs"] for d in bp_data]
        starter_pred = [intercept + slope * d["combined_excess"] for d in bp_data]
        resid = [t - p for t, p in zip(bp_tot, starter_pred)]
        bp_slope, _ = fit_ols_1d(bexc, resid)
        bullpen_w = max(0.0, min(3.0, -bp_slope))
        bp_base_rmse = rmse(starter_pred, bp_tot)
        bp_feat_rmse = rmse([p - bullpen_w * e for p, e in zip(starter_pred, bexc)],
                            bp_tot)

    # ── Totals logit weight: does the combined run-shift predict over/under? ──
    # Leakage-safe reference = the pooled mean total (no betting line needed).
    # The fitted coefficient is logits-per-run, exactly what _apply_starter_logit
    # multiplies the (run-unit) starter_total_shift by at runtime.
    mean_tot = sum(tot) / len(tot)
    shift = [-(run_scale_c * d["combined_excess"])
             - (bullpen_w * (d.get("bullpen_excess") or 0.0)) for d in data]
    over = [1 if d["total_runs"] > mean_tot else 0 for d in data]
    b_tot, a_tot = fit_logistic_1d(shift, over)
    totals_w = max(0.0, min(3.0, b_tot))
    base_tot_brier = brier([sum(over) / len(over)] * len(over), over)
    feat_tot_brier = brier([_sigmoid(a_tot + b_tot * s) for s in shift], over)

    print(f"\n=== starter_adjustment fit — {seasons_str} ({len(data)} games) ===")
    print(f"league xwOBAcon baseline: {league_xwoba:.3f}")
    print(f"moneyline weight (logit coef on innings-wtd edge): {b_ml:.3f} "
          f"-> clamped {ml_weight:.3f}")
    print(f"  Brier: baseline {base_brier:.4f} -> with-starters {feat_brier:.4f} "
          f"({'BETTER' if feat_brier < base_brier else 'no gain'})")
    if sp_base_rmse is not None:
        print(f"spreads weight (runs of margin / unit edge): {spreads_w:.3f} "
              f"({len(mar_data)} games)")
        print(f"  Margin RMSE: baseline {sp_base_rmse:.3f} -> with-starters "
              f"{sp_feat_rmse:.3f} "
              f"({'BETTER' if sp_feat_rmse < sp_base_rmse else 'no gain'})")
    print(f"run_scale (runs suppressed / unit excess): {run_scale:.3f} "
          f"-> clamped {run_scale_c:.3f}")
    print(f"  Totals RMSE: baseline {base_rmse:.3f} -> with-starters {feat_rmse:.3f} "
          f"({'BETTER' if feat_rmse < base_rmse else 'no gain'})")
    if bp_base_rmse is not None:
        print(f"bullpen weight (runs / unit bullpen excess): {bullpen_w:.3f} "
              f"({len(bp_data)} games w/ as-of relief data)")
        print(f"  Totals RMSE (on residual): starters {bp_base_rmse:.3f} -> "
              f"+bullpen {bp_feat_rmse:.3f} "
              f"({'BETTER' if bp_feat_rmse < bp_base_rmse else 'no gain'})")
    else:
        print("bullpen weight: not enough as-of relief data to fit.")
    print(f"totals weight (logit per run of combined shift): {b_tot:.3f} "
          f"-> clamped {totals_w:.3f}")
    print(f"  Over/under Brier (vs pooled mean {mean_tot:.2f}): "
          f"baseline {base_tot_brier:.4f} -> with-shift {feat_tot_brier:.4f} "
          f"({'BETTER' if feat_tot_brier < base_tot_brier else 'no gain'})")

    if do_save:
        cur = load_starter_adjustment("baseball_mlb") or {}
        cur.update({"moneyline": round(ml_weight, 3),
                    "spreads": round(spreads_w, 3),
                    "run_scale": round(run_scale_c, 3),
                    "bullpen": round(bullpen_w, 3),
                    "totals": round(totals_w, 3)})
        cur["_note"] = (f"moneyline/spreads/run_scale/bullpen/totals fit from "
                        f"{seasons_str} ({len(data)} games, "
                        f"{len(bp_data)} w/ bullpen); moneyline & spreads use the "
                        f"innings-weighted edge; props/bvp still 0.")
        save_starter_adjustment("baseball_mlb", cur,
                                meta={"source": f"backtest_starters:{seasons_str}",
                                      "fit": True, "n_games": len(data)})
        print("saved fitted weights "
              "(moneyline, spreads, run_scale, bullpen, totals).")


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else 0.0


def test_innings_weighting(seasons):
    """A/B: does innings-weighting the starter (and letting the bullpen cover
    the rest) beat the current starter-quality-only signal for ML, spreads,
    and totals? Leakage-safe, no save — reports metrics only.

    Effective run-prevention per team:
        w = clamp(avg_innings/9, 0.30, 0.85)          # share the starter covers
        eff = w * starter_suppression + (1-w) * bullpen_suppression
    vs the current baseline (starter_suppression alone).
    """
    seasons_str = ",".join(str(s) for s in seasons)
    data, _ = build_pooled_dataset(seasons)
    rows = [d for d in data
            if d.get("h_bp_sup") and d.get("a_bp_sup")
            and d.get("h_ip") and d.get("a_ip") and d.get("margin") is not None]
    if len(rows) < 200:
        print(f"Only {len(rows)} rows with bullpen+IP+margin — need more cache.")
        return

    def w_ip(ip):
        return max(0.30, min(0.85, ip / 9.0))

    base_edge, eff_edge = [], []
    base_exc, eff_exc = [], []
    for d in rows:
        wh, wa = w_ip(d["h_ip"]), w_ip(d["a_ip"])
        eff_h = wh * d["h_sp_sup"] + (1 - wh) * d["h_bp_sup"]
        eff_a = wa * d["a_sp_sup"] + (1 - wa) * d["a_bp_sup"]
        base_edge.append(math.tanh(d["h_sp_sup"] - d["a_sp_sup"]))
        eff_edge.append(math.tanh(eff_h - eff_a))
        base_exc.append((d["h_sp_sup"] - 1) + (d["a_sp_sup"] - 1))
        eff_exc.append((eff_h - 1) + (eff_a - 1))
    win = [d["home_win"] for d in rows]
    mar = [d["margin"] for d in rows]
    tot = [d["total_runs"] for d in rows]
    bexc = [(d["h_bp_sup"] - 1) + (d["a_bp_sup"] - 1) for d in rows]

    print(f"\n=== innings-weighting A/B — {seasons_str} ({len(rows)} games) ===")
    print(f"avg innings/start: home {sum(d['h_ip'] for d in rows)/len(rows):.2f} "
          f"(range shows spread matters at the extremes)")

    # ── ML: home_win ~ edge (logistic, Brier) ──
    def brier_logit(x, y):
        b, a = fit_logistic_1d(x, y)
        return brier([_sigmoid(a + b * xi) for xi in x], y)
    ml_base = brier_logit(base_edge, win)
    ml_eff = brier_logit(eff_edge, win)
    print(f"\nMONEYLINE  Brier: starter-only {ml_base:.4f} -> innings-wtd {ml_eff:.4f} "
          f"({'BETTER' if ml_eff < ml_base else 'no gain'})")

    # ── Spreads: margin ~ edge (OLS RMSE + correlation with margin) ──
    def ols_rmse(x, y):
        s, i = fit_ols_1d(x, y)
        return rmse([i + s * xi for xi in x], y)
    sp_base, sp_eff = ols_rmse(base_edge, mar), ols_rmse(eff_edge, mar)
    print(f"SPREADS    margin RMSE: starter-only {sp_base:.3f} -> innings-wtd {sp_eff:.3f} "
          f"({'BETTER' if sp_eff < sp_base else 'no gain'})")
    print(f"           corr w/ margin: starter-only {_pearson(base_edge, mar):+.3f} "
          f"-> innings-wtd {_pearson(eff_edge, mar):+.3f}")

    # ── Totals: starter-only vs starter+bullpen additive vs innings-weighted ──
    s0, i0 = fit_ols_1d(base_exc, tot)
    pred_s = [i0 + s0 * x for x in base_exc]
    tot_starter = rmse(pred_s, tot)
    resid = [t - p for t, p in zip(tot, pred_s)]
    sb, _ = fit_ols_1d(bexc, resid)
    tot_add = rmse([p + sb * b for p, b in zip(pred_s, bexc)], tot)
    tot_eff = ols_rmse(eff_exc, tot)
    print(f"TOTALS     RMSE: starter-only {tot_starter:.3f} | "
          f"starter+bullpen(current) {tot_add:.3f} | innings-wtd {tot_eff:.3f}")
    best = min(tot_starter, tot_add, tot_eff)
    tag = ("innings-wtd" if best == tot_eff else
           "starter+bullpen" if best == tot_add else "starter-only")
    print(f"           best = {tag}")


def _parse_seasons(spec):
    """Parse '2024', '2021,2022', or '2021-2024' into a sorted list of ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True,
                    help="single '2024', list '2021,2022', or range '2021-2024'")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch+cache each season's Statcast days first (slow)")
    ap.add_argument("--save", action="store_true", help="write fitted weights")
    ap.add_argument("--test-ip", action="store_true",
                    help="A/B test innings-weighted starter/bullpen blend "
                         "(ML, spreads, totals); reports metrics, no save")
    args = ap.parse_args()

    seasons = _parse_seasons(args.season)
    if args.fetch:
        for s in seasons:
            s0, s1 = _season_bounds(s)
            print(f"fetching Statcast {s0}..{s1} (one-time, slow)...")
            sh.fetch_range(s0, s1)
            get_season_games(s)
    if args.test_ip:
        test_innings_weighting(seasons)
    else:
        fit(seasons, do_save=args.save)
