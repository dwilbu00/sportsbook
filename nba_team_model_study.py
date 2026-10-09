"""NBA team-model study (READ-ONLY; 0 DTU via the mirrors; no Odds API credits).

Step 3 of the pace/team-model plan. Decides whether the pace-adjusted EFFICIENCY model
(nba_ratings) replaces the generic production model for NBA moneyline/spread/total, and with
which knobs. Every number is walk-forward: a game is predicted only from games before its date.

Predictors per game (home perspective):
  generic   PRODUCTION model: analysis._predict_margin (real code) on each team's
            recent_games exactly as served live (current + prior season, every completed game,
            newest first); total = the analyzer's weighted scored/allowed average.
  eff       nba_ratings.project_matchup (possessions x ORtg x opp DRtg) + home-court edge.
  eff+rest  eff + a back-to-back margin term.
  market    DraftKings CLOSE: margin = -home spread, total = closing total, ML = de-vigged.
HCA, the totals offset, the rest term and the outcome sigmas are fit ONLY on seasons before the
one being scored (expanding window). Knobs are tuned on --tune seasons and reported on --test
seasons, so the choice isn't fit on the games it's judged on.

Reports: (1) knob sweep on the tune seasons; (2) generic vs eff vs eff+rest vs market on games
with a DK close (margin/total MAE+RMSE, ML Brier/log-loss, home-win accuracy); (3) sigma
calibration — the model's sigma is its ONLY input to the line-value spread/total bets, so
z = (actual - market line) / sigma should have std ~1.0 and ~68% inside +-1; (4) incremental
information over the market (best blend weight of model into the de-vigged ML price).

Run (owner's machine, after the nba_team_game backfill + --export-mirror):
  $env:PYTHONIOENCODING="utf-8"
  python nba_team_model_study.py
  # options: --tune 2022,2023,2024  --test 2025,2026  --seasons-from 2021
"""
import argparse
import math

import nba_team_model as TM
import nba_warehouse as nw

SPORT_KEY = "basketball_nba"
_PHI_EPS = 1e-6


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _parse_seasons(spec):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return sorted(set(out))


# ─────────────────────────────────────────────────────────────── games + odds join
def _team_name_index(seasons):
    """{normalized full name: team_id} + {nickname: {team_ids}} from the stats.nba.com team
    box (dep-free) — maps odds-feed names ("Los Angeles Clippers") to team ids."""
    import db_store
    import nba_stats_source as src
    full, nick = {}, {}
    for s in seasons:
        df = src.load_team_box(int(s))
        if df is None or getattr(df, "empty", True):
            continue
        for r in df[["team_id", "team_city", "team_name"]].drop_duplicates().to_dict("records"):
            tid = int(r["team_id"])
            name = f"{r['team_city']} {r['team_name']}"
            full[db_store.normalize_name(name)] = tid
            nick.setdefault(db_store.normalize_name(str(r["team_name"])).split()[-1],
                            set()).add(tid)
    return full, nick


def _resolve(name, full, nick):
    import db_store
    nn = db_store.normalize_name(name or "")
    if nn in full:
        return full[nn]
    ids = nick.get(nn.split()[-1]) if nn else None
    return next(iter(ids)) if ids and len(ids) == 1 else None


def _market_index(full, nick):
    """{(home_id, away_id, date): {margin, total, p_home}} from the DK closing team store."""
    import datetime
    import warehouse
    try:
        from pricing_common import et_local_date
    except Exception:
        def et_local_date(c):
            return str(c)[:10] if c else None
    store = warehouse.load_team_market_store(SPORT_KEY, snapshot="closing")
    out = {}
    for e in (store.get("games") or {}).values():
        hid = _resolve(e.get("home_team"), full, nick)
        aid = _resolve(e.get("away_team"), full, nick)
        d = et_local_date(e.get("commence_time"))
        if hid is None or aid is None or not d:
            continue
        m = {}
        sp = (e.get("spreads") or {}).get(e.get("home_team")) or []
        if sp and sp[0].get("spread") is not None:
            m["margin"] = -float(sp[0]["spread"])
        tot = (e.get("totals") or {}).get("Over") or []
        if tot and tot[0].get("line") is not None:
            m["total"] = float(tot[0]["line"])
        ml = e.get("moneyline") or {}
        hp, ap = ml.get(e.get("home_team")) or [], ml.get(e.get("away_team")) or []
        if hp and ap and hp[0].get("implied_prob") and ap[0].get("implied_prob"):
            ih, ia = float(hp[0]["implied_prob"]), float(ap[0]["implied_prob"])
            m["p_home"] = ih / (ih + ia)
        if m:
            base = datetime.date.fromisoformat(d)
            for k in (0, -1, 1):                          # exact, then +-1 (UTC/ET slippage)
                key = (hid, aid, (base + datetime.timedelta(days=k)).isoformat())
                out.setdefault(key, m)
    return out


# ──────────────────────────────────────────────────────── generic (production) model
def _generic_predictions(games):
    """{game_id: (margin, pred_std, total, total_std)} from the PRODUCTION analyzer logic,
    fed each team's recent_games as served live (current + prior season, newest first)."""
    import analysis
    hl = analysis._half_life_for(SPORT_KEY)
    hist = {}                                             # team_id -> [game dicts] ascending
    out = {}
    by_date = {}
    for g in games:
        by_date.setdefault(g["date"], []).append(g)
    for date in sorted(by_date):
        for g in by_date[date]:
            s = g["season"]
            stats = {}
            for tid in (g["home"], g["away"]):
                rg = [x for x in reversed(hist.get(tid, [])) if x["season"] in (s, s - 1)]
                stats[tid] = {"recent_games": rg, "recent": {"avg_scored": 0, "avg_allowed": 0}}
            pm = analysis._predict_margin({"home_team": g["home"], "away_team": g["away"]},
                                          stats[g["home"]], stats[g["away"]], SPORT_KEY)
            if pm is None:
                continue
            margin, pstd = pm[0], pm[1]
            tot, tstd = _generic_total(analysis, hl, g, stats)
            out[g["game_id"]] = (margin, pstd, tot, tstd)
        for g in by_date[date]:                           # absorb after the whole date
            row = {"home_team": g["home"], "away_team": g["away"], "home_score": g["hp"],
                   "away_score": g["ap"], "total_score": g["hp"] + g["ap"],
                   "season": g["season"]}
            hist.setdefault(g["home"], []).append(row)
            hist.setdefault(g["away"], []).append(row)
    return out


def _generic_total(analysis, hl, g, stats):
    """Replica of analysis.analyze_totals' projection: average of (both teams' weighted
    points scored) and (both teams' weighted points allowed), recency x venue weights; sigma =
    weighted std of both teams' past game totals (same helpers as production)."""
    parts, tots, wts = [], [], []
    for tid, is_home in ((g["home"], True), (g["away"], False)):
        rg = stats[tid]["recent_games"]
        scored, allowed, past_h = [], [], []
        for x in rg:
            own_home = x["home_team"] == tid
            scored.append(x["home_score"] if own_home else x["away_score"])
            allowed.append(x["away_score"] if own_home else x["home_score"])
            past_h.append(own_home)
        bw = analysis._recency_weights(len(scored), hl)
        w = [b * analysis._venue_match_multiplier(p, is_home, SPORT_KEY)
             for b, p in zip(bw, past_h)]
        parts.append((analysis._weighted_mean(scored, w), analysis._weighted_mean(allowed, w)))
        tots += [x["total_score"] for x in rg]
        wts += w
    proj = ((parts[0][0] + parts[1][0]) + (parts[0][1] + parts[1][1])) / 2.0
    return proj, analysis._weighted_std(tots, wts)


# ────────────────────────────────────────────────────────────────────── metrics
def _metrics(games, preds, ids=None):
    """MAE/RMSE for margin + total, ML Brier / log-loss / accuracy (P = Phi(margin/sigma))."""
    am, at, bri, ll, acc, n = [], [], [], [], [], 0
    for g in games:
        if ids is not None and g["game_id"] not in ids:
            continue
        p = preds.get(g["game_id"])
        if not p:
            continue
        margin, sm, total, _st = p
        actual_m, actual_t = g["hp"] - g["ap"], g["hp"] + g["ap"]
        am.append(actual_m - margin)
        if total is not None:
            at.append(actual_t - total)
        ph = min(max(_phi(margin / sm) if sm else 0.5, _PHI_EPS), 1 - _PHI_EPS)
        y = 1.0 if actual_m > 0 else 0.0
        bri.append((ph - y) ** 2)
        ll.append(-(y * math.log(ph) + (1 - y) * math.log(1 - ph)))
        acc.append(1.0 if (ph > 0.5) == (y == 1.0) else 0.0)
        n += 1
    if not n:
        return None

    def mae(x):
        return sum(abs(v) for v in x) / len(x) if x else float("nan")

    def rmse(x):
        return math.sqrt(sum(v * v for v in x) / len(x)) if x else float("nan")
    return {"n": n, "m_mae": mae(am), "m_rmse": rmse(am), "t_mae": mae(at), "t_rmse": rmse(at),
            "brier": sum(bri) / n, "logloss": sum(ll) / n, "acc": sum(acc) / n}


def _market_preds(games, mkt):
    """Market as a predictor: margin = -spread, total = close, P(home) = de-vigged ML (sigma
    backed out so Phi(margin/sigma) reproduces it is unnecessary — metrics read p directly)."""
    out = {}
    for g in games:
        m = mkt.get((g["home"], g["away"], g["date"]))
        if m and "margin" in m and "total" in m and "p_home" in m:
            out[g["game_id"]] = m
    return out


def _market_metrics(games, mpreds):
    am, at, bri, ll, acc = [], [], [], [], []
    for g in games:
        m = mpreds.get(g["game_id"])
        if not m:
            continue
        actual_m, actual_t = g["hp"] - g["ap"], g["hp"] + g["ap"]
        am.append(actual_m - m["margin"])
        at.append(actual_t - m["total"])
        ph = min(max(m["p_home"], _PHI_EPS), 1 - _PHI_EPS)
        y = 1.0 if actual_m > 0 else 0.0
        bri.append((ph - y) ** 2)
        ll.append(-(y * math.log(ph) + (1 - y) * math.log(1 - ph)))
        acc.append(1.0 if (ph > 0.5) == (y == 1.0) else 0.0)
    n = len(am)
    if not n:
        return None
    return {"n": n, "m_mae": sum(map(abs, am)) / n, "m_rmse": math.sqrt(sum(v * v for v in am) / n),
            "t_mae": sum(map(abs, at)) / n, "t_rmse": math.sqrt(sum(v * v for v in at) / n),
            "brier": sum(bri) / n, "logloss": sum(ll) / n, "acc": sum(acc) / n}


def _sigma_calibration(games, preds, mpreds, kind):
    """z = (actual - market line) / model sigma. Well-sized sigma -> std(z) ~ 1, ~68% |z|<1."""
    zs = []
    for g in games:
        p, m = preds.get(g["game_id"]), mpreds.get(g["game_id"])
        if not p or not m:
            continue
        if kind == "margin":
            sig, line, actual = p[1], m["margin"], g["hp"] - g["ap"]
        else:
            sig, line, actual = p[3], m["total"], g["hp"] + g["ap"]
        if sig:
            zs.append((actual - line) / sig)
    if not zs:
        return None
    sd = math.sqrt(sum(z * z for z in zs) / len(zs))
    return {"n": len(zs), "std_z": sd, "within_1": sum(abs(z) < 1 for z in zs) / len(zs)}


def _blend_weight(games, preds, mpreds):
    """Best w in p = (1-w)*market + w*model by Brier (grid). w>0 = model adds information."""
    pairs = []
    for g in games:
        p, m = preds.get(g["game_id"]), mpreds.get(g["game_id"])
        if p and m and p[1]:
            pairs.append((m["p_home"], _phi(p[0] / p[1]), 1.0 if g["hp"] > g["ap"] else 0.0))
    if not pairs:
        return None
    best = None
    for w in [i / 20 for i in range(0, 21)]:
        b = sum(((1 - w) * pm + w * pd - y) ** 2 for pm, pd, y in pairs) / len(pairs)
        if best is None or b < best[1]:
            best = (w, b)
    base = sum((pm - y) ** 2 for pm, _pd, y in pairs) / len(pairs)
    return {"n": len(pairs), "best_w": best[0], "brier_at_best": best[1], "brier_market": base}


# ─────────────────────────────────────────────────────────────────────── report
def _row(label, m):
    if not m:
        return f"  {label:<22} (no games)"
    return (f"  {label:<22} {m['n']:>6,} {m['m_mae']:>7.2f} {m['m_rmse']:>7.2f} "
            f"{m['t_mae']:>7.2f} {m['t_rmse']:>7.2f} {m['brier']:>7.4f} {m['logloss']:>7.4f} "
            f"{m['acc']:>6.3f}")


_HDR = (f"  {'model':<22} {'n':>6} {'mMAE':>7} {'mRMSE':>7} {'tMAE':>7} {'tRMSE':>7} "
        f"{'Brier':>7} {'logL':>7} {'acc':>6}")


def run(seasons_from, tune, test):
    seasons = list(range(seasons_from, max(test + tune) + 1))
    rows = nw.team_games(seasons)
    if not rows:
        where = ("the mirror has no nba_team_game files and Azure returned no rows"
                 if nw.enabled() else
                 "no mirror files and SQL isn't configured (no .streamlit/secrets.toml?)")
        raise SystemExit(f"No team games for {seasons}: {where}. Run "
                         "`python nba_warehouse.py --seasons 2021-2026 --team-games-only`, "
                         "then `--export-mirror`.")
    games = TM.games_from_rows(rows)
    print("=" * 96)
    print("NBA TEAM-MODEL STUDY  (walk-forward; offsets fit on prior seasons only)")
    print("=" * 96)
    print(f"  seasons loaded {seasons}  games {len(games):,}  tune {tune}  test {test}")

    # (1) knob sweep on the tune seasons (no odds needed)
    tune_ids = {g["game_id"] for g in games if g["season"] in tune}
    print(f"\n(1) EFFICIENCY KNOB SWEEP on tune seasons {tune} "
          f"(score = margin RMSE + total RMSE; lower is better)")
    print(f"  {'half_life':>9} {'opp_adj':>7} {'regress':>7} {'prior':>5}   "
          f"{'mRMSE':>7} {'tRMSE':>7} {'Brier':>7}")
    scored = []
    for hl, oa, rg, pg in TM.SWEEP_GRID:
        params = {"half_life": hl, "opp_adjust": oa, "regress": rg, "prior_games": pg}
        preds, _f = TM.walk_forward_predictions(games, rows, params, tune, use_rest=True)
        m = _metrics(games, preds, tune_ids)
        if m:
            scored.append((m["m_rmse"] + m["t_rmse"], params, m))
    scored.sort(key=lambda x: x[0])
    for _s, p, m in scored[:10]:
        hl = "inf" if p["half_life"] >= 1e8 else f"{p['half_life']:.0f}"
        print(f"  {hl:>9} {str(p['opp_adjust']):>7} {p['regress']:>7} {p['prior_games']:>5.0f}"
              f"   {m['m_rmse']:>7.3f} {m['t_rmse']:>7.3f} {m['brier']:>7.4f}")
    best = scored[0][1]
    print(f"  -> chosen on tune seasons: {best}")

    # (2) head-to-head on test seasons with a DK close
    full, nick = _team_name_index(seasons)
    mkt = _market_index(full, nick)
    test_games = [g for g in games if g["season"] in test]
    mpreds = _market_preds(test_games, mkt)
    ids = set(mpreds)
    print(f"\n(2) HEAD-TO-HEAD on test seasons {test}, games with a DK close: {len(ids):,} "
          f"of {len(test_games):,}")
    gen = _generic_predictions(games)
    eff, fits = TM.walk_forward_predictions(games, rows, best, test, use_rest=False)
    effr, _ = TM.walk_forward_predictions(games, rows, best, test, use_rest=True)
    print(_HDR)
    print(_row("generic (production)", _metrics(test_games, gen, ids)))
    print(_row("efficiency", _metrics(test_games, eff, ids)))
    print(_row("efficiency + rest", _metrics(test_games, effr, ids)))
    print(_row("market (DK close)", _market_metrics(test_games, mpreds)))
    for s in test:
        f = fits.get(s)
        if f:
            print(f"  fit for {s} (from prior seasons): HCA {f['hca']:+.2f}  total offset "
                  f"{f['tot_off']:+.2f}  b2b {f['beta_rest']:+.2f}  sigma margin "
                  f"{f['sig_m']:.2f}  sigma total {f['sig_t']:.2f}")

    # by season, regular season only
    print("\n  by season (regular season, games with a DK close):")
    print(_HDR)
    for s in test:
        sid = {g["game_id"] for g in test_games
               if g["season"] == s and g["season_type"] == "regular-season"} & ids
        print(f"  [{s}]")
        print(_row("  generic", _metrics(test_games, gen, sid)))
        print(_row("  efficiency + rest", _metrics(test_games, effr, sid)))
        print(_row("  market", _market_metrics([g for g in test_games if g["game_id"] in sid],
                                               mpreds)))

    # (3) sigma calibration (the model's only input to line-value spreads/totals)
    print("\n(3) SIGMA CALIBRATION vs the DK line (target std(z) ~1.00, within+-1 ~0.68)")
    for label, preds in (("generic (production)", gen), ("efficiency + rest", effr)):
        for kind in ("margin", "total"):
            c = _sigma_calibration(test_games, preds, mpreds, kind)
            if c:
                print(f"  {label:<22} {kind:<6} n {c['n']:>5,}  std(z) {c['std_z']:.3f}  "
                      f"within+-1 {c['within_1']:.3f}")

    # (4) incremental information over the market
    print("\n(4) INCREMENTAL INFO: best blend weight of the model into the de-vigged DK ML "
          "(w=0 -> market already has it)")
    for label, preds in (("generic (production)", gen), ("efficiency + rest", effr)):
        b = _blend_weight(test_games, preds, mpreds)
        if b:
            print(f"  {label:<22} n {b['n']:>5,}  best w {b['best_w']:.2f}  Brier "
                  f"{b['brier_at_best']:.5f} vs market {b['brier_market']:.5f}")
    print()


def main():
    ap = argparse.ArgumentParser(description="NBA team-model study (read-only).")
    ap.add_argument("--seasons-from", type=int, default=2021,
                    help="first season loaded (warm-up for ratings + offset fits)")
    ap.add_argument("--tune", default="2022,2023,2024", help="seasons the knobs are chosen on")
    ap.add_argument("--test", default="2025,2026", help="seasons reported head-to-head")
    args = ap.parse_args()
    import db_store
    db_store.promote_secrets_from_toml()          # Azure fallback for un-mirrored seasons
    run(args.seasons_from, _parse_seasons(args.tune), _parse_seasons(args.test))


if __name__ == "__main__":
    main()
