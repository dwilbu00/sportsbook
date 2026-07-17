"""
Sweep per-sport min_streak thresholds and report Brier + sample size per prop.

Goal: figure out whether the default `min_streak` (production threshold for
"consecutive valid games required before predictions resume") is calibrated
correctly given that the low-minutes and pre/post-layoff filters already
catch most of the "player isn't right tonight" cases.

For each threshold we run a sweep over the full props grid (so the variant
pick is honest to that threshold), then report the best out-of-sample
holdout (variant × method) per prop.

Usage:
    cd SPORTSBOOK_ODDS && python3 eval_min_streak.py --sport nba
    cd SPORTSBOOK_ODDS && python3 eval_min_streak.py --sport mlb --thresholds 3,4,5,6,7,8
    cd SPORTSBOOK_ODDS && python3 eval_min_streak.py --sport nfl --thresholds 2,3,4,5,6
"""
import argparse

from backtest import (
    SPORT_MAP, DEFAULT_STARTERS, DEFAULT_PROPS,
    SPORT_DEFAULT_MIN_STREAK, _build_props_sweep_grid,
    run_player_props_backtest,
)
from refit_calibration import _best_per_prop


DEFAULT_THRESHOLDS = {
    "nba": [3, 4, 5, 6, 7, 8, 10, 12],
    "mlb": [3, 4, 5, 6, 7, 8, 10],
    "nfl": [2, 3, 4, 5, 6, 8],
}


def main():
    p = argparse.ArgumentParser(description="Evaluate optimal min_streak per sport")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="nba")
    p.add_argument("--thresholds", default=None,
                   help="Comma-separated min_streak values to test")
    p.add_argument("--games-per-player", type=int, default=80)
    args = p.parse_args()

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]
    players = DEFAULT_STARTERS[args.sport]
    props = DEFAULT_PROPS[args.sport]
    variants = _build_props_sweep_grid()

    if args.thresholds:
        thresholds = [int(t) for t in args.thresholds.split(",")]
    else:
        thresholds = DEFAULT_THRESHOLDS.get(args.sport, [3, 5, 7, 10])

    summary = {}
    for thresh in thresholds:
        SPORT_DEFAULT_MIN_STREAK[sport_key] = thresh
        print(f"\n{'#'*72}")
        print(f"#  {args.sport.upper()}: EVALUATING min_streak = {thresh}")
        print(f"{'#'*72}", flush=True)
        results = run_player_props_backtest(
            args.sport, espn_sport, espn_league, sport_key,
            players=players, props=props,
            games_per_player=args.games_per_player, min_sample=5,
            variants=variants, sweep=True,
            safe_mode=True, calibrate=True,
            cross_season="strict",
        )
        if not results:
            print(f"  No results at threshold {thresh}; skipping.", flush=True)
            continue
        winners = _best_per_prop(results, props)
        first_v = next(iter(results))
        n_obs = {prop: len(results[first_v][prop].get("calib_obs") or [])
                 for prop in props}
        summary[thresh] = {"winners": winners, "n_obs": n_obs}

    print("\n" + "=" * 100)
    print(f"{args.sport.upper()} min_streak comparison "
          f"({len(players)} players × {args.games_per_player} games, "
          f"holdout-Brier per prop)")
    print("=" * 100)
    header = (f"{'thresh':>6} {'prop':<6} {'N_obs':>6} {'best variant':<26} "
              f"{'method':>7} {'Brier':>7} {'Hit %':>7}")
    print(header)
    print("-" * len(header))
    for thresh in sorted(summary):
        for prop in props:
            short = (prop.replace("player_", "").replace("batter_", "")
                     .replace("pitcher_", "")[:5].upper())
            w = summary[thresh]["winners"].get(prop)
            n = summary[thresh]["n_obs"].get(prop, 0)
            if not w:
                print(f"{thresh:>6} {short:<6} {n:>6} {'(no winner)':<26}")
                continue
            print(f"{thresh:>6} {short:<6} {n:>6} {w['variant']:<26} "
                  f"{w['method']:>7} {w['brier']:>7.4f} {w['hit']:>6.2f}%")
        print()


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
