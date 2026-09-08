"""nfl_model.py — the dedicated NFL game-margin model (validated, leakage-safe).

Replaces the noisy recency-weighted-raw-margin base for NFL with the model we
actually validated OOS:
    margin ~= HFA + w_epa·net_epa_edge + w_qb·qb_edge
                  + w_off·off_inj_edge + w_def·def_inj_edge + w_rest·rest_edge
All terms are leakage-safe as-of the game date. Fit by OLS on the seasons where
every feature is available (injuries/snaps: 2023+); saved into the calibration
``starter_adjustment['nfl_margin_model']`` block. build_matchup_features calls
predict() and hands analysis._predict_margin a ready pred_margin + pred_std, so
the live model is exactly the validated one (no recency-margin base).

Findings that justify this (see memory): HFA was missing from the origin-fit
research model (+~2 pts, -0.11 to -0.16 RMSE); opponent adjustment was tested and
REJECTED (worse OOS even at 10 seasons); box-score EPA tops out ~13.2 RMSE vs the
closing line's 12.67 (the residual is the market's irreducible edge).
"""
import math

import nfl_epa
import nfl_qb_asof
import nfl_injury_impact as inj
import nfl_data
from calibration_loader import load_starter_adjustment, save_starter_adjustment

SPORT_KEY = "americanfootball_nfl"
DEFAULT_SIGMA = 13.2
# feature order for the saved weight vector
FEATURE_KEYS = ("hfa", "w_epa", "w_qb", "w_off", "w_def", "w_rest")


# ── as-of feature edges (home-perspective; + = home better) ────────────────────
def _rest_edge(season, gid):
    c = nfl_data.game_context([str(season)]).get(gid) if gid else None
    if not c:
        return 0.0
    re = c.get("rest_edge_home")
    return float(re) if re is not None else 0.0


def feature_edges(season, date, home_abbr, away_abbr, team_ratings=None,
                  starter_ids=None, gid=None, week=None):
    """(net_epa_edge, qb_edge, off_inj_edge, def_inj_edge, rest_edge), as-of.
    ``starter_ids`` {abbr: gsis} → game-actual starters (backtest); else projected
    (live). ``team_ratings`` avoids recomputing EPA."""
    if team_ratings is None:
        team_ratings = nfl_epa.team_epa(season, as_of_date=date)
    h, a = team_ratings.get(home_abbr), team_ratings.get(away_abbr)
    net_edge = (h["net_epa"] - a["net_epa"]) if h and a else 0.0

    qr, _lg = nfl_qb_asof.qb_ratings(season, date)
    if starter_ids:
        qh = nfl_qb_asof.qb_edge_delta(season, date, home_abbr, starter_ids.get(home_abbr), qr)
        qa = nfl_qb_asof.qb_edge_delta(season, date, away_abbr, starter_ids.get(away_abbr), qr)
    else:
        qh = nfl_qb_asof.projected_qb_delta(season, date, home_abbr, qr)
        qa = nfl_qb_asof.projected_qb_delta(season, date, away_abbr, qr)
    qb_edge = qh - qa

    off_inj = def_inj = 0.0
    if week is None:
        week = nfl_qb_asof._week_for(season, home_abbr, date)
    if week is not None:
        try:
            off_inj, _ol, def_inj = inj.injury_edges(season, int(week), home_abbr, away_abbr)
        except Exception:
            off_inj = def_inj = 0.0

    return net_edge, qb_edge, off_inj, def_inj, _rest_edge(season, gid)


# ── live prediction ────────────────────────────────────────────────────────────
def weights():
    """Saved model weights dict, or None if not fit yet."""
    m = (load_starter_adjustment(SPORT_KEY) or {}).get("nfl_margin_model")
    if not m or not all(k in m for k in FEATURE_KEYS):
        return None
    return m


def predict(home_abbr, away_abbr, date, season, team_ratings=None,
            starter_ids=None, gid=None, week=None, w=None):
    """(pred_margin, pred_std) for one game, or None if the model isn't fit.
    Home-perspective margin. All features leakage-safe as-of ``date``."""
    if w is None:
        w = weights()
    if not w:
        return None
    net, qb, off_i, def_i, rest = feature_edges(
        season, date, home_abbr, away_abbr, team_ratings, starter_ids, gid, week)
    margin = (w["hfa"] + w["w_epa"] * net + w["w_qb"] * qb +
              w["w_off"] * off_i + w["w_def"] * def_i + w["w_rest"] * rest)
    return margin, float(w.get("sigma", DEFAULT_SIGMA))


# ── fit + save ─────────────────────────────────────────────────────────────────
def _solve(X, y):
    from nfl_injury_edge import _solve as s
    return s(X, y)


def build_dataset(seasons):
    """[(hfa=1, net, qb, off, def, rest, margin, season, gid)] with game-actual
    starters (the fit basis)."""
    from nfl_injury_edge import build_rows
    rows = build_rows(seasons)             # (net, qb, off, def, margin, season, gid)
    out = []
    for (net, qb, off_i, def_i, margin, season, gid) in rows:
        rest = _rest_edge(season, gid)
        out.append((1.0, net, qb, off_i, def_i, rest, margin, season, gid))
    return out


def _oos_rmse(rows):
    from nfl_injury_edge import _rmse
    seasons = sorted({r[7] for r in rows})
    tot = 0.0
    for test in seasons:
        tr = [r for r in rows if r[7] != test]
        te = [r for r in rows if r[7] == test]
        if not tr or not te:
            continue
        b = _solve([list(r[:6]) for r in tr], [r[6] for r in tr])
        tot += _rmse([sum(bi * xi for bi, xi in zip(b, r[:6])) for r in te],
                     [r[6] for r in te]) * len(te)
    return tot / len(rows)


def fit(seasons, do_save=False):
    from nfl_injury_edge import _rmse
    rows = build_dataset(seasons)
    if not rows:
        print("no games"); return
    X = [list(r[:6]) for r in rows]
    y = [r[6] for r in rows]
    b = _solve(X, y)
    resid = [yi - sum(bi * xi for bi, xi in zip(b, xr)) for xr, yi in zip(X, y)]
    sigma = (sum(e * e for e in resid) / len(resid)) ** 0.5
    oos = _oos_rmse(rows)
    model = dict(zip(FEATURE_KEYS, [round(x, 4) for x in b]))
    model["sigma"] = round(sigma, 3)
    ss = f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])
    print(f"=== NFL margin model — {ss} ({len(rows)} games) ===")
    print(f"  {model}")
    print(f"  in-sample RMSE={ (sum(e*e for e in resid)/len(resid))**0.5:.3f}  "
          f"OOS RMSE={oos:.3f}")
    if do_save:
        cur = load_starter_adjustment(SPORT_KEY) or {}
        cur["nfl_margin_model"] = model
        save_starter_adjustment(SPORT_KEY, cur, meta={
            "nfl_margin_model": f"nfl_model.py --seasons {ss}", "oos_rmse": round(oos, 3)})
        print(f"  [save] wrote starter_adjustment['nfl_margin_model'] "
              f"(candidate-staged unless --live)")
    return model


def main():
    import argparse
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    from calibration_loader import (set_candidate_mode, has_candidate,
                                    existing_candidate_notice)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--live", action="store_true", help="write live (skip candidate staging)")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    staging = not args.live
    set_candidate_mode(staging)
    if staging and args.save:
        n = existing_candidate_notice(SPORT_KEY)
        if n:
            print(n)
    fit(seasons, do_save=args.save)
    if args.save and staging and has_candidate(SPORT_KEY):
        print("  ⇢ staged to candidate; promote: python refit_calibration.py --sport nfl --promote")


if __name__ == "__main__":
    main()
