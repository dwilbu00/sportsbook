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
from calibration_loader import load_starter_adjustment, save_starter_adjustment

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
              f"to calibration/{SPORT_KEY}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True,
                    help="e.g. 2023-2025 or 2023,2024,2025")
    ap.add_argument("--save", action="store_true", help="write fitted weight")
    args = ap.parse_args()
    fit(_parse_seasons(args.seasons), do_save=args.save)


if __name__ == "__main__":
    main()
