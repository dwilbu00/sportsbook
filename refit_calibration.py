"""
Refit persistent player-prop calibration files.

For each (sport, prop):
  1. Run the existing backtest sweep with calibration capture on the CURRENT
     season's data.
  2. Pick the best (variant × calibration method) per prop based on
     out-of-sample (chronological holdout) Brier.
  3. From the chosen variant's full calib_obs, fit pooled residual stats
     (mu, sigma, ECDF) — these are the runtime distributions analysis.py uses.
  4. Repeat against the PRIOR season's data to produce the warmup block.
  5. Write SPORTSBOOK_ODDS/calibration/<sport_key>.json.

Usage:
    cd SPORTSBOOK_ODDS && python3 refit_calibration.py --sport nba
    python3 refit_calibration.py --sport mlb --warmup-games 10
"""
import argparse
import math
import sys
from datetime import datetime

from backtest import (
    SPORT_MAP, DEFAULT_STARTERS, DEFAULT_PROPS, VARIANT_PRESETS,
    _build_props_sweep_grid, _evaluate_calibration_methods,
    run_player_props_backtest,
)
from calibration_loader import save_calibration


def _mlb_player_pool(season, max_batters=40, max_pitchers=30):
    """Resolve a broad, data-driven MLB calibration pool from cached seasons."""
    if not season:
        season = datetime.utcnow().year
    try:
        import mlb_starters
        from backtest_props import frequent_batter_ids, starter_ids

        player_ids = (frequent_batter_ids([season], max_batters)
                      + starter_ids([season])[:max_pitchers])
        names = []
        for start in range(0, len(player_ids), 50):
            chunk = player_ids[start:start + 50]
            data = mlb_starters._get(
                "people", {"personIds": ",".join(map(str, chunk))})
            names.extend(
                person.get("fullName") for person in data.get("people", [])
                if person.get("fullName")
            )
        return list(dict.fromkeys(names))
    except Exception as exc:
        print(f"  [warn] broad MLB player pool unavailable: {exc}")
        return []


def _parse_variant_name(name):
    """
    Parse a sweep variant key like 'hl15/defadj1.0/ven0.25' into a dict of
    {half_life, opp_defense_strength, output_def_strength, venue_strength}.
    Returns None if the format isn't recognized.

    NOTE: The sweep grid uses `def_adj` (output-side defense). The
    `opp_defense_strength` (weight-side) is always 0 in the sweep grid.
    """
    parts = name.split("/")
    if len(parts) != 3:
        return None
    hl_part, da_part, ven_part = parts
    try:
        # _build_props_sweep_grid emits "none" or "hl<N>" for the half-life.
        if hl_part == "none":
            hl = None
        elif hl_part.startswith("hl"):
            hl = int(hl_part[2:])
        else:
            return None
        da = float(da_part[len("defadj"):])
        ven = float(ven_part[len("ven"):])
    except (ValueError, IndexError):
        return None
    return {
        "half_life": hl,
        "opp_defense_strength": 0.0,
        "output_def_strength": da,
        "venue_strength": ven,
    }


def _fit_residuals(obs):
    """Pool residuals (actual - projected) over all observations."""
    residuals = [actual - proj for _, proj, _, actual, _, *_ in obs]
    if not residuals:
        return None
    mu = sum(residuals) / len(residuals)
    var = sum((r - mu) ** 2 for r in residuals) / len(residuals)
    sigma = math.sqrt(var) if var > 0 else 0.0
    return {
        "residual_mu": mu,
        "residual_sigma": sigma,
        "residual_ecdf": sorted(residuals),
        "n_obs": len(residuals),
    }


def _best_per_prop(results, props, k_values=(0,)):
    """
    For each prop, evaluate every (variant × method) on a chronological holdout
    and return the winner by lowest Brier. Returns:
        {prop_key: {"variant": str, "method": str, "brier": float, "hit": float}}
    """
    winners = {}
    for prop_key in props:
        best = None
        for vname, by_prop in results.items():
            obs = by_prop[prop_key].get("calib_obs") or []
            if not obs:
                continue
            evals = _evaluate_calibration_methods(obs, k_values, holdout=True)
            for e in evals:
                if e["brier"] is None or e["k"] not in (None, 0):
                    continue
                # Only persist non-shrinkage methods (A, B, C) — per-player
                # shrinkage variants (B*, C*) overfit out-of-sample per the
                # NBA holdout sweep.
                if e["method"] not in ("A", "B", "C"):
                    continue
                if best is None or e["brier"] < best["brier"]:
                    best = {
                        "variant": vname,
                        "method": e["method"],
                        "brier": e["brier"],
                        "hit": e["hit"],
                    }
        if best:
            winners[prop_key] = best
    return winners


def _build_prop_cfg(winner, results, prop_key, shrinkage_k_default):
    """Combine winner's variant params + fitted residuals into a JSON entry."""
    vname = winner["variant"]
    parsed = _parse_variant_name(vname) or {}
    obs = results[vname][prop_key].get("calib_obs") or []
    fit = _fit_residuals(obs) or {}
    cfg = {
        "method": winner["method"],
        "half_life": parsed.get("half_life"),
        "venue_strength": parsed.get("venue_strength"),
        "opp_defense_strength": parsed.get("opp_defense_strength", 0.0),
        "output_def_strength": parsed.get("output_def_strength", 0.0),
        "shrinkage_k": shrinkage_k_default,
        "variant_label": vname,
        "fit_brier": round(winner["brier"], 4),
        "fit_hit_pct": round(winner["hit"], 2) if winner["hit"] is not None else None,
    }
    cfg.update(fit)
    return cfg


def refit_sport(sport, season=None, prior_season=None, players=None, props=None,
                games_per_player=80, warmup_games=10, shrinkage_k_default=0,
                mlb_max_batters=40, mlb_max_pitchers=30):
    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    if sport == "mlb" and season is None:
        season = datetime.utcnow().year
    if players is None and sport == "mlb":
        players = _mlb_player_pool(
            season, max_batters=mlb_max_batters,
            max_pitchers=mlb_max_pitchers)
        if not players:
            print("No data-driven MLB player pool was available; aborting rather "
                  "than fitting all MLB props from the small static fallback.")
            sys.exit(1)
    players = players or DEFAULT_STARTERS.get(sport)
    props = props or DEFAULT_PROPS.get(sport)
    if not players or not props:
        print(f"No default players/props for {sport}; please pass --players/--props.")
        sys.exit(1)

    variants = _build_props_sweep_grid()

    print(f"\n=== Fitting CURRENT-season calibration for {sport_key} ===")
    curr_results = run_player_props_backtest(
        sport, espn_sport, espn_league, sport_key,
        players=players, props=props,
        games_per_player=games_per_player,
        min_sample=5, variants=variants, sweep=True,
        season_year=season, safe_mode=True,
        cushion_sweep=False, safe_target=0.80,
        quantile_mode=False, calibrate=True,
        cross_season="strict",
    )
    if not curr_results:
        print("Current-season run produced no results; aborting.")
        sys.exit(2)

    curr_winners = _best_per_prop(curr_results, props)

    warmup_results = None
    warmup_winners = {}
    if prior_season is not None:
        print(f"\n=== Fitting WARMUP (prior season={prior_season}) calibration ===")
        warmup_results = run_player_props_backtest(
            sport, espn_sport, espn_league, sport_key,
            players=players, props=props,
            games_per_player=games_per_player,
            min_sample=5, variants=variants, sweep=True,
            season_year=prior_season, safe_mode=True,
            cushion_sweep=False, safe_target=0.80,
            quantile_mode=False, calibrate=True,
            cross_season="all",  # within a single prior season this is fine
        )
        if warmup_results:
            warmup_winners = _best_per_prop(warmup_results, props)

    # Build final cfg
    props_cfg = {}
    for prop_key in props:
        winner = curr_winners.get(prop_key)
        if not winner:
            print(f"  [skip] {prop_key}: no calibration winner")
            continue
        cfg = _build_prop_cfg(winner, curr_results, prop_key, shrinkage_k_default)
        cfg["warmup_games"] = warmup_games
        cfg["fit_season"] = season

        warm_winner = warmup_winners.get(prop_key)
        if warm_winner and warmup_results:
            warm_obs = warmup_results[warm_winner["variant"]][prop_key].get("calib_obs") or []
            warm_fit = _fit_residuals(warm_obs) or {}
            cfg["warmup"] = {
                "method": warm_winner["method"],
                "variant_label": warm_winner["variant"],
                "fit_brier": round(warm_winner["brier"], 4),
                "fit_season": prior_season,
                **warm_fit,
            }
        print(f"  [{prop_key}] method={cfg['method']} hl={cfg['half_life']} "
              f"defadj={cfg['output_def_strength']} ven={cfg['venue_strength']} "
              f"brier={cfg.get('fit_brier')} n={cfg.get('n_obs')}")
        props_cfg[prop_key] = cfg

    meta = {
        "current_season": season,
        "warmup_season": prior_season,
        "games_per_player": games_per_player,
        "warmup_games": warmup_games,
        "n_players": len(players),
    }
    save_calibration(sport_key, props_cfg, meta=meta)
    print(f"\n✓ Wrote calibration/{sport_key}.json "
          f"({len(props_cfg)} props)")


def main():
    p = argparse.ArgumentParser(description="Refit persistent calibration files")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), required=True)
    p.add_argument("--season", type=int, default=None,
                   help="Current season year (ESPN convention). Default: current.")
    p.add_argument("--prior-season", type=int, default=None,
                   help="Prior season year for warmup. Recommended.")
    p.add_argument("--players", default=None,
                   help="Comma-separated player names. Default: built-in starters.")
    p.add_argument("--props", default=None,
                   help="Comma-separated prop keys. Default: per-sport defaults.")
    p.add_argument("--games-per-player", type=int, default=80)
    p.add_argument("--warmup-games", type=int, default=10,
                   help="Player current-season games count at which warmup blend = 0.")
    p.add_argument("--shrinkage-k", type=int, default=0,
                   help="Bayesian shrinkage k written into the calibration file "
                        "(applied at runtime by analysis.py).")
    p.add_argument("--mlb-max-batters", type=int, default=40,
                   help="Data-driven MLB batter pool size when --players is omitted.")
    p.add_argument("--mlb-max-pitchers", type=int, default=30,
                   help="Data-driven MLB pitcher pool size when --players is omitted.")
    args = p.parse_args()

    players = [n.strip() for n in args.players.split(",")] if args.players else None
    props = [pk.strip() for pk in args.props.split(",")] if args.props else None

    refit_sport(args.sport, season=args.season, prior_season=args.prior_season,
                players=players, props=props,
                games_per_player=args.games_per_player,
                warmup_games=args.warmup_games,
                shrinkage_k_default=args.shrinkage_k,
                mlb_max_batters=args.mlb_max_batters,
                mlb_max_pitchers=args.mlb_max_pitchers)


if __name__ == "__main__":
    main()
