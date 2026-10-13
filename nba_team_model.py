"""nba_team_model.py — the NBA team-market model: pace-adjusted efficiency + back-to-back.

Validated in nba_team_model_study.py (walk-forward, 2025-26 test, 2,607 games with a DK close):
beats the generic production model on every metric (ML Brier 0.2072 vs 0.2129, margin MAE 11.25
vs 11.46, total MAE 14.83 vs 15.12) with a correctly-sized outcome sigma (std(z) 0.99 vs the DK
line, where the generic margin sigma was ~49% too wide).

    possessions = pace_h * pace_a / lg_pace                     (nba_ratings, as-of the game)
    margin      = possessions * (ORtg_h*DRtg_a - ORtg_a*DRtg_h) / lg_rtg / 100
                  + HCA + beta_rest * (home_b2b - away_b2b)
    total       = possessions * (ORtg_h*DRtg_a + ORtg_a*DRtg_h) / lg_rtg / 100 + total_offset
    P(home)     = Phi(margin / sigma_margin)

Fit (``python nba_team_model.py --fit``) picks the ratings knobs by walk-forward score and fits
HCA / total offset / rest / sigmas, then (``--save``) stores the block in
calibration/basketball_nba.json ``starter_adjustment['nba_team_model']`` — candidate-staged
unless ``--live``; review with ``refit_calibration.py --sport nba --diff``, then ``--promote``.
Serving: build_matchup_features() -> analysis._predict_margin / analyze_totals use the model's
margin, total and sigmas (fit == serve: the odds backtest calls the same builder with a
walk-forward ratings index). Inert until the block exists.
"""
from __future__ import annotations

import itertools
import math
import time

import nba_ratings as R

SPORT_KEY = "basketball_nba"
MODEL_KEY = "nba_team_model"
_REQUIRED = ("params", "hca", "tot_off", "beta_rest", "sigma_margin", "sigma_total")

# Ratings-knob grid the fit sweeps (half_life x opp_adjust x regress x prior_games).
SWEEP_GRID = tuple(itertools.product((10.0, 20.0, 40.0, 1e9), (True, False), (0.25, 0.5),
                                     (5.0, 10.0, 20.0)))
MIN_FIT_GAMES = 200


# ───────────────────────────────────────────────────────────── shared (study + fit)
def games_from_rows(rows):
    """[{game_id, season, season_type, date, home, away, hp, ap}] ascending, from
    nba_warehouse.team_games rows (games with both team rows + possessions only)."""
    out = []
    for date, season, gid, a, b in R._pair_games(rows):
        h, aw = (a, b) if a.get("home_away") == "home" else (b, a)
        if h.get("home_away") != "home":
            continue
        out.append({"game_id": gid, "season": season, "season_type": a.get("season_type"),
                    "date": date, "home": h["team_id"], "away": aw["team_id"],
                    "hp": float(h["points"]), "ap": float(aw["points"])})
    return out


def b2b_diff(h, a):
    """+1 home on a back-to-back, -1 away on one, 0 neither/both."""
    return (1 if h.get("rest_days") == 1 else 0) - (1 if a.get("rest_days") == 1 else 0)


def raw_projections(games, pre):
    """{game_id: (margin_no_hca, total_no_offset, b2b_diff)} from a pregame-ratings map."""
    out = {}
    for g in games:
        h, a = pre.get((g["game_id"], g["home"])), pre.get((g["game_id"], g["away"]))
        if not h or not a:
            continue
        pj = R.project_matchup(h, a, hca=0.0)
        out[g["game_id"]] = (pj["margin"], pj["total"], b2b_diff(h, a))
    return out


def fit_offsets(games, raw, seasons):
    """HCA (mean margin residual), total offset (mean total residual), rest coefficient (OLS
    of the remaining margin residual on b2b_diff) and outcome sigmas, from games in
    ``seasons``. None below MIN_FIT_GAMES."""
    res_m, res_t, b2b = [], [], []
    for g in games:
        if g["season"] not in seasons or g["game_id"] not in raw:
            continue
        m, t, b = raw[g["game_id"]]
        res_m.append((g["hp"] - g["ap"]) - m)
        res_t.append((g["hp"] + g["ap"]) - t)
        b2b.append(b)
    if len(res_m) < MIN_FIT_GAMES:
        return None
    hca = sum(res_m) / len(res_m)
    tot_off = sum(res_t) / len(res_t)
    resid = [r - hca for r in res_m]
    sxx = sum(b * b for b in b2b)
    beta = (sum(r * b for r, b in zip(resid, b2b)) / sxx) if sxx else 0.0
    sig_m = math.sqrt(sum((r - beta * b) ** 2 for r, b in zip(resid, b2b)) / len(resid))
    sig_t = math.sqrt(sum((r - tot_off) ** 2 for r in res_t) / len(res_t))
    return {"hca": hca, "tot_off": tot_off, "beta_rest": beta, "sig_m": sig_m,
            "sig_t": sig_t, "n": len(res_m)}


def predict(h, a, m, use_rest=True):
    """(margin, sigma_margin, total, sigma_total) from two pre-game ratings + a fitted
    model block (keys hca/tot_off/beta_rest/sigma_margin/sigma_total)."""
    pj = R.project_matchup(h, a, hca=float(m["hca"]))
    margin = pj["margin"] + (float(m["beta_rest"]) * b2b_diff(h, a) if use_rest else 0.0)
    return (margin, float(m["sigma_margin"]), pj["total"] + float(m["tot_off"]),
            float(m["sigma_total"]))


def walk_forward_predictions(games, rows, params, eval_seasons, use_rest=True):
    """{game_id: (margin, sigma_m, total, sigma_t)} where each season's offsets are fit only
    on EARLIER seasons (expanding window). Returns (preds, fits_by_season)."""
    pre = R.pregame_ratings(rows, params)
    raw = raw_projections(games, pre)
    seasons = sorted({g["season"] for g in games})
    fits = {s: fit_offsets(games, raw, [x for x in seasons if x < s]) for s in eval_seasons}
    out = {}
    for g in games:
        f = fits.get(g["season"])
        if not f or g["game_id"] not in raw:
            continue
        m, t, b = raw[g["game_id"]]
        out[g["game_id"]] = (m + f["hca"] + (f["beta_rest"] * b if use_rest else 0.0),
                             f["sig_m"], t + f["tot_off"], f["sig_t"])
    return out, fits


def score(games, preds, ids):
    """(margin RMSE + total RMSE, margin RMSE, total RMSE, n) over ``ids``."""
    em, et = [], []
    for g in games:
        p = preds.get(g["game_id"])
        if g["game_id"] in ids and p:
            em.append((g["hp"] - g["ap"]) - p[0])
            et.append((g["hp"] + g["ap"]) - p[2])
    if not em:
        return None
    rm = math.sqrt(sum(e * e for e in em) / len(em))
    rt = math.sqrt(sum(e * e for e in et) / len(et))
    return rm + rt, rm, rt, len(em)


# ──────────────────────────────────────────────────────────────────────── serving
_MODEL_CACHE = {}                                     # (serving path, mtime_ns) -> block


def model():
    """The fitted model block from calibration, or None (model inert until fit + promoted).
    Memoized on the serving file's (path, mtime): the calibration JSON is ~10 MB, and the prop
    context / backtests ask for the knobs per observation (an uncached read is ~100 ms)."""
    import os
    try:
        import calibration_loader
        path = calibration_loader._serving_path(SPORT_KEY)
        key = (path, os.stat(path).st_mtime_ns) if os.path.exists(path) else (path, None)
    except Exception:
        key = None
    if key is not None and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from calibration_loader import load_starter_adjustment
        m = (load_starter_adjustment(SPORT_KEY) or {}).get(MODEL_KEY)
    except Exception:
        return None
    if not isinstance(m, dict) or not all(k in m for k in _REQUIRED):
        m = None
    if key is not None:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = m
    return m


_LIVE_CACHE = {}                                      # (date, season, params) -> (ts, ratings)
_LIVE_TTL = 1800


def live_ratings(date, season, params):
    """{team_id: rating} as of ``date`` from nba_team_game (current + 2 prior seasons, so the
    season-start prior is built from real data). Cached per (date, season, params)."""
    import nba_warehouse as nw
    key = (str(date)[:10], int(season), tuple(sorted((params or {}).items())))
    hit = _LIVE_CACHE.get(key)
    if hit and (time.time() - hit[0]) < _LIVE_TTL:
        return hit[1]
    rows = nw.team_games([int(season) - 2, int(season) - 1, int(season)])
    ratings = R.ratings_as_of(rows, str(date)[:10], int(season), params=params) if rows else {}
    _LIVE_CACHE[key] = (time.time(), ratings)
    return ratings


def pregame_index(rows, params):
    """{(date, team_id): pre-game rating} — the walk-forward lookup the odds backtest feeds to
    build_matchup_features (one walk instead of one per game)."""
    pre = R.pregame_ratings(rows, params)
    out = {}
    for date, _season, gid, a, b in R._pair_games(rows):
        for r in (a, b):
            rt = pre.get((gid, r["team_id"]))
            if rt is not None:
                out[(date, r["team_id"])] = rt
    return out


def build_matchup_features(home, away, game_date, pregame=None, m=None):
    """Model features for one game (odds-feed team names, ET game date), or None when the
    model isn't fit, a team can't be resolved, or ratings are missing. ``pregame`` (backtest)
    = pregame_index output; live builds ratings as of ``game_date``. Never raises."""
    try:
        m = m or model()
        if not m:
            return None
        import nba_stats_source as src
        import nba_warehouse as nw
        hid, aid = nw.resolve_team_id(home), nw.resolve_team_id(away)
        if hid is None or aid is None:
            return None
        d = str(game_date)[:10]
        if pregame is not None:
            h, a = pregame.get((d, hid)), pregame.get((d, aid))
        else:
            season = src.end_year_for_date(d)
            ratings = live_ratings(d, season, m["params"])
            h, a = ratings.get(hid), ratings.get(aid)
        if not h or not a:
            return None
        margin, sm, total, st = predict(h, a, m)
        return {
            "nba_pred_margin": margin, "nba_pred_std": sm,
            "nba_pred_total": total, "nba_total_std": st,
            "nba_possessions": h["pace"] * a["pace"] / h["lg_pace"],
            "nba_home": {k: h.get(k) for k in ("pace", "ortg", "drtg", "n", "rest_days")},
            "nba_away": {k: a.get(k) for k in ("pace", "ortg", "drtg", "n", "rest_days")},
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────── fit/save
def fit(seasons, save=False, verbose=True):
    """Choose ratings knobs by walk-forward score over every season after the first (warm-up),
    then fit offsets/sigmas on those seasons. ``save`` stages the block (candidate unless the
    caller disabled candidate mode). Returns the model block."""
    import nba_warehouse as nw
    seasons = sorted(int(s) for s in seasons)
    rows = nw.team_games(seasons)
    games = games_from_rows(rows)
    eval_seasons = seasons[1:]
    ids = {g["game_id"] for g in games if g["season"] in eval_seasons}
    if not ids:
        raise SystemExit("No team games for the fit seasons — backfill + export the mirror.")
    best = None
    for hl, oa, rg, pg in SWEEP_GRID:
        params = dict(R.DEFAULT_PARAMS, half_life=hl, opp_adjust=oa, regress=rg,
                      prior_games=pg)
        preds, _f = walk_forward_predictions(games, rows, params, eval_seasons)
        sc = score(games, preds, ids)
        if sc and (best is None or sc[0] < best[0][0]):
            best = (sc, params)
    (sc, rm, rt, n), params = best
    raw = raw_projections(games, R.pregame_ratings(rows, params))
    f = fit_offsets(games, raw, eval_seasons)
    block = {"params": params, "hca": round(f["hca"], 4), "tot_off": round(f["tot_off"], 4),
             "beta_rest": round(f["beta_rest"], 4), "sigma_margin": round(f["sig_m"], 4),
             "sigma_total": round(f["sig_t"], 4),
             "fit_seasons": f"{eval_seasons[0]}-{eval_seasons[-1]}", "n_games": f["n"],
             "walk_forward": {"margin_rmse": round(rm, 4), "total_rmse": round(rt, 4),
                              "n": n}}
    if verbose:
        print(f"=== NBA team model fit — warm-up {seasons[0]}, fit {block['fit_seasons']} ===")
        print(f"  knobs (walk-forward best of {len(SWEEP_GRID)}): {params}")
        print(f"  walk-forward margin RMSE {rm:.3f}  total RMSE {rt:.3f}  (n={n})")
        print(f"  HCA {block['hca']:+.2f}  total offset {block['tot_off']:+.2f}  "
              f"back-to-back {block['beta_rest']:+.2f}  sigma margin "
              f"{block['sigma_margin']:.2f}  sigma total {block['sigma_total']:.2f}")
    if save:
        from calibration_loader import (active_write_label, load_starter_adjustment,
                                        save_starter_adjustment)
        cur = dict(load_starter_adjustment(SPORT_KEY) or {})
        cur[MODEL_KEY] = block
        save_starter_adjustment(SPORT_KEY, cur, meta={
            MODEL_KEY: f"nba_team_model.py --fit --seasons {seasons[0]}-{seasons[-1]}"})
        if verbose:
            print(f"  [save] wrote starter_adjustment['{MODEL_KEY}'] -> "
                  f"{active_write_label(SPORT_KEY)}")
    return block


def main():
    import argparse
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Fit / stage the NBA team-market model.")
    ap.add_argument("--fit", action="store_true", help="fit the model (prints the block)")
    ap.add_argument("--seasons", default="2021-2026",
                    help="season-ENDING years; the first is the ratings warm-up")
    ap.add_argument("--save", action="store_true",
                    help="stage the block into calibration/basketball_nba.candidate.json")
    ap.add_argument("--live", action="store_true",
                    help="with --save: write the LIVE calibration file instead of staging")
    args = ap.parse_args()
    if not args.fit:
        ap.error("nothing to do (pass --fit)")
    spec = args.seasons
    if "-" in spec:
        a, b = spec.split("-", 1)
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = [int(x) for x in spec.split(",") if x.strip()]
    import db_store
    db_store.promote_secrets_from_toml()
    if args.save and not args.live:
        from calibration_loader import set_candidate_mode
        set_candidate_mode(True)
    fit(seasons, save=args.save)
    if args.save and not args.live:
        print("  review: python refit_calibration.py --sport nba --diff ; then --promote")


if __name__ == "__main__":
    main()
