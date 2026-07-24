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
    _score_calibration_methods, _team_defense_lookup,
    run_player_props_backtest,
)
from calibration_loader import load_calibration, save_calibration
from espn_cache import seed_athlete_id
from espn_client import list_season_athletes

# A non-empirical method (B pooled-Gaussian / C pooled-ECDF) must beat the
# empirical baseline (method A) by at least this much holdout Brier AND confirm
# out-of-sample in two expanding chronological folds before it can be selected.
# Without this gate, argmin-Brier over ~250 (variant × method) candidates on a
# single split is winner's-curse selection: a method no better than empirical
# gets shipped and advertises an optimistic fit_brier. See P1.4.
MIN_CALIB_BRIER_GAIN = 0.002


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


def _nba_player_pool(season, max_players=150, min_games=15):
    """Resolve a broad, usage-representative NBA calibration pool.

    Replaces the hand-picked `DEFAULT_STARTERS["nba"]` (18 superstars), which is
    survivorship-biased twice over: the names were chosen with hindsight, and the
    list contains no role players / DNP games — so the residual fit sees only
    elite, high-floor production and the P2j low-participation filter is never
    exercised (the star pool has no 0-minute games to drop).

    Selection = the top `max_players` NBA athletes by TOTAL minutes among those
    with >= `min_games` games in `season`. This is usage-based rather than
    stardom-based — the same trade-off the MLB pool (`_mlb_player_pool`, top-N by
    batted-ball volume) already accepts. It is still mildly survivorship-flavored
    (a season-ending injury lowers minutes), but that matches the population
    books actually offer props on; `min_games` drops 10-day / deep-bench noise
    that isn't bet on while keeping rotation players who have real DNPs.

    The authoritative ESPN ids from the listing are pinned into the id cache
    (`seed_athlete_id`) so the broad name list resolves to the RIGHT players
    instead of `search_athlete`'s first name match. Returns a de-duplicated list
    of names, or [] on failure (refit_sport then aborts rather than fitting from
    the 18-star fallback).
    """
    if not season:
        season = datetime.utcnow().year
    try:
        athletes = list_season_athletes("basketball", "nba", season)
    except Exception as exc:
        print(f"  [warn] NBA player pool unavailable: {exc}")
        return []

    eligible = [a for a in athletes
                if (a.get("games") or 0) >= min_games
                and (a.get("minutes") or 0) > 0]
    eligible.sort(key=lambda a: a.get("minutes") or 0.0, reverse=True)

    names, seen = [], set()
    for a in eligible[:max_players]:
        name = a.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            seed_athlete_id("basketball", "nba", name, a.get("id"))
        except OSError:
            pass  # best effort; cached_athlete_id falls back to name search
        names.append(name)
    return names


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


def _chronological_folds(obs, min_set_n=20):
    """Two expanding-train chronological folds over calib_obs (date at index 5).

    Returns [(fit_obs, score_obs), (fit_obs, score_obs)] with strictly earlier
    train than test in each fold, or [] when there is not enough dated data to
    form two disjoint later test windows. Mirrors the expanding-window
    confirmation used by the MLB prop and starter gates.
    """
    rows = sorted(obs, key=lambda o: o[5] if len(o) > 5 else "")
    n = len(rows)
    if n < 3 * min_set_n:
        return []
    cut1 = rows[int(n * 0.6)][5]
    cut2 = rows[int(n * 0.8)][5]
    if not cut1 or not cut2 or cut1 == cut2:
        return []
    folds = [
        ([o for o in rows if o[5] < cut1],
         [o for o in rows if cut1 <= o[5] < cut2]),
        ([o for o in rows if o[5] < cut2],
         [o for o in rows if o[5] >= cut2]),
    ]
    for fit_obs, score_obs in folds:
        if len(fit_obs) < min_set_n or len(score_obs) < min_set_n:
            return []
    return folds


def _method_brier_by_fold(obs, method):
    """Out-of-sample Brier of `method` in each confirmation fold (A/B/C only)."""
    briers = []
    for fit_obs, score_obs in _chronological_folds(obs):
        evals = _score_calibration_methods(fit_obs, score_obs, ())
        by_method = {e["method"]: e for e in evals
                     if e["k"] in (None, 0) and e["brier"] is not None}
        briers.append(by_method)
    return briers


def _confirms_over_baseline(obs, method):
    """A non-empirical method must beat method A out-of-sample in BOTH folds.

    This is the guard that defeats winner's-curse selection: noise that makes a
    method win the single holdout split will not also beat empirical in two
    independent later folds. Returns False when there isn't enough data to
    confirm (so the safe empirical baseline is used instead).
    """
    folds = _method_brier_by_fold(obs, method)
    if not folds:
        return False
    for by_method in folds:
        a = by_method.get("A")
        m = by_method.get(method)
        if not a or not m or m["brier"] >= a["brier"]:
            return False
    return True


def _cv_brier(obs, method):
    """Mean out-of-sample Brier of `method` across the confirmation folds.

    Persisted as a less-biased estimate of the DEPLOYED calibration's quality
    than the single-split holdout Brier (which is the argmax-selected winner's
    optimistic number). Returns None when folds are unavailable.
    """
    folds = _method_brier_by_fold(obs, method)
    briers = [by_method[method]["brier"]
              for by_method in folds if method in by_method]
    if not briers or len(briers) < len(folds):
        return None
    return round(sum(briers) / len(briers), 4)


def _is_baseline_variant(vname):
    """True for the all-knobs-off sweep cell (plain recency mean, no def/venue).
    It is the floor the P2.1 variant gate measures candidates against, and is
    always eligible WITHOUT the gate (so a winner always exists)."""
    p = _parse_variant_name(vname)
    return (bool(p) and not p.get("half_life") and not p.get("output_def_strength")
            and not p.get("venue_strength") and not p.get("opp_defense_strength"))


def _baseline_variant_obs(results, prop_key):
    """calib_obs for the baseline (all-off) variant, for the variant gate."""
    for vname, by_prop in results.items():
        if _is_baseline_variant(vname):
            return by_prop.get(prop_key, {}).get("calib_obs") or []
    return []


def _variant_confirms(cand_obs, base_obs, cand_method):
    """P2.1: a non-baseline VARIANT (with its selected method) must beat the
    BASELINE variant (method A — knobs off, empirical) out-of-sample in BOTH
    chronological folds. Clones _confirms_over_baseline onto the knob axis: noise
    that wins the single 50/50 split will not also beat the do-nothing baseline in
    two independent later folds. Returns False when data is too thin to confirm
    (→ the safe baseline is used instead)."""
    if not base_obs:
        return False
    cand_folds = _chronological_folds(cand_obs)
    base_folds = _chronological_folds(base_obs)
    if not cand_folds or not base_folds or len(cand_folds) != len(base_folds):
        return False
    for (cf, cs), (bf, bs) in zip(cand_folds, base_folds):
        ce = {e["method"]: e for e in _score_calibration_methods(cf, cs, ())
              if e["k"] in (None, 0) and e["brier"] is not None}
        be = {e["method"]: e for e in _score_calibration_methods(bf, bs, ())
              if e["k"] in (None, 0) and e["brier"] is not None}
        c = ce.get(cand_method)
        b = be.get("A")
        if not c or not b or c["brier"] >= b["brier"]:
            return False
    return True


def _best_per_prop(results, props, k_values=(0,)):
    """
    For each prop, evaluate every (variant × method) on a chronological holdout
    and return the winner by lowest Brier. Method A (empirical) is the safe
    baseline and is always eligible; a fancier method (B/C) is eligible only if
    it beats A on the holdout by >= MIN_CALIB_BRIER_GAIN AND confirms
    out-of-sample in two expanding chronological folds (see P1.4). Returns:
        {prop_key: {"variant", "method", "brier", "hit",
                    "baseline_brier", "cv_brier", "confirmed"}}
    """
    winners = {}
    for prop_key in props:
        base_obs = _baseline_variant_obs(results, prop_key)
        best = None
        for vname, by_prop in results.items():
            obs = by_prop[prop_key].get("calib_obs") or []
            if not obs:
                continue
            is_baseline = _is_baseline_variant(vname)
            evals = _evaluate_calibration_methods(obs, k_values, holdout=True)
            by_method = {}
            for e in evals:
                if e["brier"] is None or e["k"] not in (None, 0):
                    continue
                # Only persist non-shrinkage methods (A, B, C) — per-player
                # shrinkage variants (B*, C*) overfit out-of-sample per the
                # NBA holdout sweep.
                if e["method"] in ("A", "B", "C"):
                    by_method[e["method"]] = e
            baseline = by_method.get("A")
            for method, e in by_method.items():
                if method != "A":
                    # Fancier methods must clear the baseline margin AND confirm.
                    if (baseline is None
                            or baseline["brier"] - e["brier"] < MIN_CALIB_BRIER_GAIN):
                        continue
                    if not _confirms_over_baseline(obs, method):
                        continue
                # P2.1 variant gate: a non-baseline knob combo must ALSO beat the
                # baseline variant out-of-sample in both folds — else it's likely a
                # single-split winner's-curse and we keep the baseline (the floor).
                # Only active when the sweep actually contains the baseline cell
                # (always true in the real grid; skipped in narrow unit fixtures).
                if (base_obs and not is_baseline
                        and not _variant_confirms(obs, base_obs, method)):
                    continue
                if best is None or e["brier"] < best["brier"]:
                    best = {
                        "variant": vname,
                        "method": method,
                        "brier": e["brier"],
                        "hit": e["hit"],
                        "baseline_brier": (round(baseline["brier"], 4)
                                           if baseline else None),
                        "cv_brier": _cv_brier(obs, method),
                        "confirmed": method != "A",
                        "variant_confirmed": not is_baseline,
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
        # P1.4 provenance: the empirical-baseline Brier this method was measured
        # against, the cross-validated (two-fold) Brier of the DEPLOYED method,
        # and whether a non-empirical method cleared the confirmation gate.
        "baseline_brier": winner.get("baseline_brier"),
        "cv_brier": winner.get("cv_brier"),
        "confirmed": winner.get("confirmed", False),
        # P2.1: whether the winning knob combo cleared the variant confirmation
        # gate (beat the baseline variant in both folds). False = baseline (floor).
        "variant_confirmed": winner.get("variant_confirmed", False),
    }
    cfg.update(fit)
    return cfg


def refit_sport(sport, season=None, prior_season=None, players=None, props=None,
                games_per_player=80, warmup_games=10, shrinkage_k_default=0,
                mlb_max_batters=40, mlb_max_pitchers=30,
                nba_max_players=150, nba_min_games=15):
    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    if sport in ("mlb", "nba") and season is None:
        season = datetime.utcnow().year
    if players is None and sport == "mlb":
        players = _mlb_player_pool(
            season, max_batters=mlb_max_batters,
            max_pitchers=mlb_max_pitchers)
        if not players:
            print("No data-driven MLB player pool was available; aborting rather "
                  "than fitting all MLB props from the small static fallback.")
            sys.exit(1)
    if players is None and sport == "nba":
        players = _nba_player_pool(
            season, max_players=nba_max_players, min_games=nba_min_games)
        if not players:
            print("No data-driven NBA player pool was available; aborting rather "
                  "than fitting NBA props from the 18-star fallback.")
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


def refit_sport_real_lines(sport, store_label="", warmup_games=10,
                           shrinkage_k_default=0, xstats_strength=0.0,
                           dry_run=False):
    """Re-select each prop's calibration METHOD at REAL book lines (roadmap 0.3).

    The synthetic-line sweep (`refit_sport`) chooses each prop's A/B/C method by
    Brier at a season-average line, but `props.py` applies it at the real book
    line. Residual mu/sigma/ecdf are line-invariant, so only the A-vs-B/C choice
    needs re-deciding at real lines. This reads the real book lines held in the
    durable historical_odds store, re-selects the method per prop using the same
    confirmation gate as `_best_per_prop` (via
    book_line_calibration.select_method_at_real_lines), and MERGES the result
    into calibration/<sport>.json.

    The projection variant (half_life/venue/def) is held fixed at the shipped
    choice; only `method` and its (line-invariant) residual distribution are
    refit. Props with too little real-line data are SKIPPED (their synthetic fit
    is preserved). The warmup block is carried forward unchanged — there is no
    prior-season real-line data, and warmup only affects players with fewer than
    `warmup_games` current-season games.

    OFFLINE + FREE: reads the store + free ESPN gamelogs; no Odds API credits.
    """
    import book_line_calibration as blc

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key)
    if not existing:
        print(f"No existing calibration/{sport_key}.json props to refit; run the "
              f"synthetic sweep (refit_sport) first.")
        return
    target_props = list(existing.keys())

    print(f"\n=== Re-selecting calibration method at REAL book lines for "
          f"{sport_key} ===")
    book_lines = blc.harvest_book_lines_from_store(
        sport_key, target_props, store_label)
    print(f"  {len(book_lines)} store book lines across {len(target_props)} "
          f"calibrated props")
    if not book_lines:
        print("  No real book lines in the store; nothing to refit.")
        return
    enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("  No observations joined to actuals; nothing to refit.")
        return

    # Weight-side opponent-defense lookup only if some prop's variant uses it.
    team_defense, league_avg_def = {}, None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in existing):
        print("  Building team-defense lookup (a variant uses opp_defense)...")
        team_defense, _, league_avg_def = _team_defense_lookup(
            espn_sport, espn_league)

    # ── P2.4a: leakage-safe as-of xBA index for the projection blend ──
    # Built once from the raw Statcast day cache spanning the obs seasons. Only
    # applies to props in props.PROP_XSTATS_KIND (batter_hits). Uses a per-game
    # as-of index (NOT the current-as-of SQL table) so a past-dated obs never
    # sees future data. xstats_strength=0 → byte-identical to the prior behavior.
    from props import PROP_XSTATS_KIND
    xba_index = None
    if xstats_strength and xstats_strength > 0:
        import savant_history as sh
        import backtest_props
        years = sorted({str(o["game_date"])[:4] for o in enriched
                        if o.get("game_date")})
        raw = []
        for y in years:
            try:
                raw.extend(sh.load_days(f"{y}-03-01", f"{y}-11-30"))
            except Exception:
                pass
        if raw:
            xba_index = backtest_props.build_batter_xba_index(raw)
            print(f"  [xstats] as-of xBA index built from {len(raw)} pitch rows "
                  f"over {years} (strength={xstats_strength})")
        else:
            print("  [xstats] no Statcast days cached for the obs seasons — "
                  "xBA blend inactive; refit falls back to the plain projection.")

    changed = {}
    for prop_key in target_props:
        cfg = existing.get(prop_key) or {}
        params = {
            "half_life": cfg.get("half_life"),
            "venue_strength": cfg.get("venue_strength", 0.0),
            "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
            "use_minutes": cfg.get("use_minutes", False),
        }
        prop_xstats = (xstats_strength if (xba_index is not None
                       and prop_key in PROP_XSTATS_KIND) else 0.0)
        rows = blc.build_real_line_obs(
            enriched, params, sport_key, prop_key, team_defense, league_avg_def,
            xstats_strength=prop_xstats, xba_index=xba_index)
        sel = blc.select_method_at_real_lines(rows)
        if sel is None:
            print(f"  [skip] {prop_key}: {len(rows)} real-line obs "
                  f"(<20 usable) — keeping synthetic fit "
                  f"(method={cfg.get('method')})")
            continue

        old_method = cfg.get("method")
        # Normally only a genuine method FLIP is written (a same-method re-fit
        # would just churn the residuals onto a smaller real-line basis for no
        # runtime change). BUT when the PROJECTION BASIS changed (P2.4a xBA blend
        # applied to this prop), the residuals MUST be re-fit even if the method
        # stays the same — the "no runtime change" premise is false.
        projection_changed = prop_xstats > 0
        if sel["method"] == old_method and not projection_changed:
            print(f"  [keep] {prop_key}: real-line eval confirms method "
                  f"{old_method} (real-line brier={sel['fit_brier']}, "
                  f"baseline(A)={sel['baseline_brier']}, n={sel['n_obs']})")
            continue

        # Preserve variant params, shrinkage_k, variant_label, warmup, etc.;
        # overwrite only the method + its line-invariant residual distribution.
        new_cfg = dict(cfg)
        new_cfg.update({
            "method": sel["method"],
            "residual_mu": sel["residual_mu"],
            "residual_sigma": sel["residual_sigma"],
            "residual_ecdf": sel["residual_ecdf"],
            "n_obs": sel["n_obs"],
            "fit_brier": sel["fit_brier"],
            "baseline_brier": sel["baseline_brier"],
            "cv_brier": sel["cv_brier"],
            "confirmed": sel["confirmed"],
        })
        new_cfg.setdefault("warmup_games", warmup_games)
        new_cfg.setdefault("shrinkage_k", cfg.get("shrinkage_k",
                                                  shrinkage_k_default))
        new_cfg["real_line_fit"] = {
            "fit_at_real_lines": True,
            "n_obs": sel["n_obs"],
            "source": "historical_odds store",
            "store_label": store_label or "default",
            "xstats_strength": prop_xstats,
        }
        # Persist the xBA blend weight into the prop cfg so props._knob activates
        # it in production at exactly the weight its residuals were re-fit under.
        if prop_xstats > 0:
            new_cfg["xstats_strength"] = prop_xstats
        changed[prop_key] = new_cfg
        note = (f"{old_method}→{sel['method']} FLIP"
                if sel["method"] != old_method
                else (f"re-fit @xstats={prop_xstats}" if projection_changed
                      else "unchanged"))
        print(f"  [{prop_key}] method {note}  brier={sel['fit_brier']} "
              f"baseline(A)={sel['baseline_brier']} cv={sel['cv_brier']} "
              f"n={sel['n_obs']}")

    if not changed:
        print("\nNo props had enough real-line data to re-select. "
              "Nothing written.")
        return
    if dry_run:
        print(f"\n[dry-run] {len(changed)} prop(s) would be written "
              f"({sorted(changed.keys())}); nothing saved.")
        return

    save_calibration(
        sport_key, changed,
        meta={"real_line_refit": {
            "props": sorted(changed.keys()),
            "store_label": store_label or "default",
        }},
        merge_props=True)
    print(f"\n✓ Updated calibration/{sport_key}.json "
          f"({len(changed)} prop(s) re-selected at real book lines; "
          f"other props/blocks preserved)")


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
    p.add_argument("--nba-max-players", type=int, default=150,
                   help="Data-driven NBA pool size (top-N by minutes) when "
                        "--players is omitted.")
    p.add_argument("--nba-min-games", type=int, default=15,
                   help="Minimum games played for an NBA player to enter the "
                        "data-driven pool.")
    p.add_argument("--real-lines", action="store_true",
                   help="Re-select each prop's calibration method at REAL book "
                        "lines from the durable historical_odds store (roadmap "
                        "0.3), instead of the synthetic-line sweep. OFFLINE + "
                        "free; merges into the existing calibration file.")
    p.add_argument("--store-label", default="",
                   help="historical_odds store label to read real book lines "
                        "from with --real-lines (default: the unlabeled store).")
    p.add_argument("--xstats-strength", type=float, default=0.0,
                   help="P2.4a: with --real-lines, re-fit batter_hits residuals "
                        "under the Statcast xBA projection blend at this weight "
                        "(leakage-safe as-of), and persist the weight into the "
                        "prop cfg. 0 = no blend (default).")
    p.add_argument("--dry-run", action="store_true",
                   help="With --real-lines, compute + print per-prop Brier but "
                        "write nothing (use to compare --xstats-strength values).")
    args = p.parse_args()

    if args.real_lines:
        refit_sport_real_lines(args.sport, store_label=args.store_label,
                               warmup_games=args.warmup_games,
                               shrinkage_k_default=args.shrinkage_k,
                               xstats_strength=args.xstats_strength,
                               dry_run=args.dry_run)
        return

    players = [n.strip() for n in args.players.split(",")] if args.players else None
    props = [pk.strip() for pk in args.props.split(",")] if args.props else None

    refit_sport(args.sport, season=args.season, prior_season=args.prior_season,
                players=players, props=props,
                games_per_player=args.games_per_player,
                warmup_games=args.warmup_games,
                shrinkage_k_default=args.shrinkage_k,
                mlb_max_batters=args.mlb_max_batters,
                mlb_max_pitchers=args.mlb_max_pitchers,
                nba_max_players=args.nba_max_players,
                nba_min_games=args.nba_min_games)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
