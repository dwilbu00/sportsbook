"""
Fit the NFL `starter_adjustment` margin weight from HISTORICAL outcomes — no
odds needed. The NFL analog to backtest_starters.py (which fits MLB).

The live model (analysis._predict_margin) computes a game margin and then adds
    weight * starter_edge
where, for NFL, ``starter_edge`` is the home-minus-away net-EPA/play difference
built by nfl_epa.build_matchup_features(). This script fits ``weight`` (points of
margin per unit EPA edge) by OLS of actual game margin on the leakage-safe
as-of-date EPA edge, pooled across seasons, and writes it to
calibration/americanfootball_nfl.json under starter_adjustment['spreads'].

Because moneyline and spreads share _predict_margin, this single weight drives
BOTH team markets. prob_shrink (fit separately from odds history) then corrects
any residual over/under-confidence, so a mildly aggressive weight is safe.

Usage:
    python3 backtest_nfl_epa.py --seasons 2023-2025            # report only
    python3 backtest_nfl_epa.py --seasons 2023-2025 --save     # write weight
"""

import argparse

import nfl_epa
import nfl_qb_asof
from calibration_loader import (
    load_starter_adjustment, save_starter_adjustment, set_candidate_mode,
    has_candidate, active_write_label, existing_candidate_notice,
)

SPORT_KEY = "americanfootball_nfl"


def _parse_seasons(spec):
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(x) for x in spec.split(",") if x.strip()})


def build_dataset(seasons):
    """Return [(edge, margin)] over all graded games in `seasons`, using only
    EPA computed from games strictly before each game's date (leakage-safe)."""
    data = []
    for season in seasons:
        plays = nfl_epa.load_plays(season)
        games = {}
        for p in plays:
            if "home_score" in p:
                games[p["game_id"]] = (
                    p["game_date"], p["home_team"], p["away_team"],
                    p["home_score"] - p["away_score"])
        graded = 0
        for gid, (d, h, a, margin) in sorted(games.items(), key=lambda kv: kv[1][0]):
            ratings = nfl_epa.team_epa(season, as_of_date=d)
            rh, ra = ratings.get(h), ratings.get(a)
            # Require some current-season sample on both sides so week-1 games
            # (pure prior-season carryover) don't dominate the early fit.
            if not rh or not ra:
                continue
            if rh["off_plays"] <= 0 or ra["off_plays"] <= 0:
                continue
            data.append((rh["net_epa"] - ra["net_epa"], float(margin)))
            graded += 1
        print(f"  {season}: {graded} games graded")
    return data


def build_dataset_qb(seasons):
    """Like build_dataset but each row is (base_edge, qb_diff, margin, season),
    where qb_diff = (home starter-vs-baseline QB delta) - (away delta). All
    leakage-safe (EPA + QB ratings as-of the game date; starter identity is
    public pre-game). Used by the OOS QB-value comparison."""
    data = []
    for season in seasons:
        plays = nfl_epa.load_plays(season)
        starters = nfl_qb_asof.game_starters(season)
        games = {}
        for p in plays:
            if "home_score" in p:
                games[p["game_id"]] = (p["game_date"], p["home_team"],
                                       p["away_team"],
                                       p["home_score"] - p["away_score"])
        graded = 0
        for gid, (d, h, a, margin) in sorted(games.items(), key=lambda kv: kv[1][0]):
            ratings = nfl_epa.team_epa(season, as_of_date=d)
            rh, ra = ratings.get(h), ratings.get(a)
            if not rh or not ra or rh["off_plays"] <= 0 or ra["off_plays"] <= 0:
                continue
            base_edge = rh["net_epa"] - ra["net_epa"]
            qb_r, _lg = nfl_qb_asof.qb_ratings(season, d)
            gs = starters.get(gid, {})
            dh = nfl_qb_asof.qb_edge_delta(season, d, h, gs.get(h), qb_r)
            da = nfl_qb_asof.qb_edge_delta(season, d, a, gs.get(a), qb_r)
            data.append((base_edge, dh - da, float(margin), season))
            graded += 1
        print(f"  {season}: {graded} games graded (QB dataset)")
    return data


def _fit1(xs, ys):
    """OLS slope through origin for y = w*x."""
    sxx = sum(x * x for x in xs)
    return (sum(x * y for x, y in zip(xs, ys)) / sxx) if sxx else 0.0


def _fit2(x1s, x2s, ys):
    """OLS (no intercept) for y = a*x1 + b*x2 via the 2x2 normal equations."""
    s11 = sum(x * x for x in x1s)
    s22 = sum(x * x for x in x2s)
    s12 = sum(p * q for p, q in zip(x1s, x2s))
    s1y = sum(x * y for x, y in zip(x1s, ys))
    s2y = sum(x * y for x, y in zip(x2s, ys))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return _fit1(x1s, ys), 0.0
    a = (s1y * s22 - s2y * s12) / det
    b = (s11 * s2y - s12 * s1y) / det
    return a, b


def _rmse(pred, ys):
    n = len(ys)
    return (sum((p - y) ** 2 for p, y in zip(pred, ys)) / n) ** 0.5 if n else 0.0


def oos_qb_value(seasons):
    """Leave-one-season-out: for each held-out season, fit on the OTHERS and
    measure held-out margin RMSE for (a) base EPA edge only and (b) EPA edge +
    QB delta. If QB info helps, the QB model's OOS RMSE is LOWER. This is the
    honest go/no-go before building the full odds beat-the-close backtest."""
    data = build_dataset_qb(seasons)
    if len(seasons) < 2:
        print("Need >=2 seasons for leave-one-out OOS."); return
    print(f"\n=== OOS QB-value (leave-one-season-out, {len(data)} games) ===")
    print("  held  n     base_RMSE   +QB_RMSE    delta     w_epa   w_qb   n_qb!=0")
    tot_b = tot_q = 0.0
    ntot = 0
    for test in seasons:
        tr = [d for d in data if d[3] != test]
        te = [d for d in data if d[3] == test]
        if not tr or not te:
            continue
        # base model: fit w on train base_edge, apply to test
        wb = _fit1([d[0] for d in tr], [d[2] for d in tr])
        pb = [wb * d[0] for d in te]
        rb = _rmse(pb, [d[2] for d in te])
        # qb model: fit (w_epa, w_qb) on train, apply to test
        wa, wq = _fit2([d[0] for d in tr], [d[1] for d in tr], [d[2] for d in tr])
        pq = [wa * d[0] + wq * d[1] for d in te]
        rq = _rmse(pq, [d[2] for d in te])
        nqb = sum(1 for d in te if abs(d[1]) > 1e-9)
        print(f"  {test}  {len(te):<4}  {rb:8.4f}   {rq:8.4f}   {rq-rb:+7.4f}   "
              f"{wa:6.2f}  {wq:6.2f}  {nqb}")
        tot_b += rb * len(te); tot_q += rq * len(te); ntot += len(te)
    if ntot:
        print(f"  ---- weighted OOS RMSE: base={tot_b/ntot:.4f}  "
              f"+QB={tot_q/ntot:.4f}  delta={ (tot_q-tot_b)/ntot:+.4f} "
              f"({'QB HELPS' if tot_q < tot_b else 'QB does NOT help'}) ----")
        # restrict to the games that actually HAVE a QB change — where the signal lives
        te_all = data
        chg = [d for d in te_all if abs(d[1]) > 1e-9]
        print(f"\n  On the {len(chg)} QB-CHANGE games only (pooled, in-sample weights):")
        if chg:
            wa, wq = _fit2([d[0] for d in te_all], [d[1] for d in te_all],
                           [d[2] for d in te_all])
            wb = _fit1([d[0] for d in te_all], [d[2] for d in te_all])
            rb = _rmse([wb * d[0] for d in chg], [d[2] for d in chg])
            rq = _rmse([wa * d[0] + wq * d[1] for d in chg], [d[2] for d in chg])
            print(f"    base_RMSE={rb:.4f}  +QB_RMSE={rq:.4f}  delta={rq-rb:+.4f} "
                  f"({'QB HELPS' if rq < rb else 'QB does NOT help'})")


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def fit_ols_through_origin(xs, ys):
    """Least-squares slope for y = w*x (no intercept: a zero EPA edge should
    predict a zero margin shift on top of the base model)."""
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    return sxy / sxx if sxx else 0.0


def fit(seasons, do_save=False):
    data = build_dataset(seasons)
    if not data:
        print("No games; aborting.")
        return
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    w = fit_ols_through_origin(xs, ys)
    corr = _pearson(xs, ys)
    seasons_str = f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])
    print(f"\n=== NFL EPA margin fit — {seasons_str} ({len(data)} games) ===")
    print(f"  corr(edge, margin) = {corr:+.4f}")
    print(f"  OLS margin weight (points per unit net-EPA edge) = {w:.3f}")
    # A predicted-margin RMSE sanity check: baseline (predict 0) vs edge model.
    base_rmse = (sum(y * y for y in ys) / len(ys)) ** 0.5
    mdl_rmse = (sum((y - w * x) ** 2 for x, y in zip(xs, ys)) / len(ys)) ** 0.5
    print(f"  margin RMSE: baseline={base_rmse:.3f}  edge-model={mdl_rmse:.3f}")

    if do_save:
        cur = load_starter_adjustment(SPORT_KEY) or {}
        cur["enabled"] = True
        cur["spreads"] = round(w, 3)
        # moneyline shares the margin model, so it needs no separate weight; keep
        # any existing value for transparency but it is unused by _predict_margin.
        save_starter_adjustment(SPORT_KEY, cur, meta={
            "source": f"backtest_nfl_epa.py --seasons {seasons_str}",
            "corr": round(corr, 4),
            "games": len(data),
        })
        print(f"\n  [save] wrote starter_adjustment['spreads']={round(w,3)} "
              f"to calibration/{active_write_label(SPORT_KEY)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="e.g. 2023-2025 or 2023,2024,2025")
    ap.add_argument("--save", action="store_true", help="write fitted weight "
                    "(stages a candidate; promote via refit_calibration.py "
                    "--sport nfl --promote)")
    ap.add_argument("--live", action="store_true",
                    help="with --save, write the LIVE calibration file directly "
                         "(skip candidate staging)")
    ap.add_argument("--oos-qb", action="store_true",
                    help="leave-one-season-out OOS test of whether the starting-QB "
                         "adjustment improves margin prediction (go/no-go, no save)")
    args = ap.parse_args()
    if args.oos_qb:
        oos_qb_value(_parse_seasons(args.seasons))
        return
    staging = not args.live
    set_candidate_mode(staging)
    if staging and args.save:
        _n = existing_candidate_notice(SPORT_KEY)
        if _n:
            print(_n)
    fit(_parse_seasons(args.seasons), do_save=args.save)
    if args.save and staging and has_candidate(SPORT_KEY):
        print(f"\n⇢ Staged to calibration/{SPORT_KEY}.candidate.json — live file "
              f"UNTOUCHED. Promote: python refit_calibration.py --sport nfl "
              f"--promote (review with --diff).")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
