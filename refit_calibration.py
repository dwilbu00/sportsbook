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
from datetime import datetime, timezone

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

# ── Data-gated line-conditional method selection (§2.4b-2 follow-up) ──
# The best calibration method is line-dependent (diagnostic: C wins at line 0.5,
# D dist:+xBA at >=1.5). refit_sport_real_lines picks the method PER LINE BUCKET
# for these props, but a bucket adopts its own method only when it has enough obs
# AND clears the confirmation gate AND beats the pooled method on that bucket —
# else it inherits the pooled method. Ships inert until a bucket earns it.
LINE_CONDITIONAL_PROPS = {"batter_hits"}
LINE_BUCKETS = [0.5, None]          # ascending max_line; None = open-ended top
# Per-bucket floor before a bucket can flip. Comfortably above the 2-fold
# confirmation gate's own minimum (~60 obs to form the two expanding folds) for a
# more robust confirmation, but below the whole-prop obs count so a higher-line
# bucket can qualify as it accrues. The 2-fold gate is still the winner's-curse
# safeguard: a bucket only flips if its winner beats empirical in BOTH folds AND
# beats the pooled method on the bucket by >= MIN_CALIB_BRIER_GAIN.
MIN_BUCKET_OBS = 100
LINE_COND_XSTATS_STRENGTH = 0.5    # xBA weight used when scoring method D


def _lc_bucket_counts(enriched, prop_key):
    """{max_line_cap: n_obs} for a prop's line buckets (LINE_BUCKETS). A line
    falls in the first bucket whose cap >= it (cap None = the open-ended top)."""
    counts = {}
    prev = None
    for cap in LINE_BUCKETS:
        n = 0
        for o in enriched:
            if not isinstance(o, dict) or o.get("prop_key") != prop_key:
                continue
            ln = o.get("line")
            if ln is None:
                continue
            if (prev is None or ln > prev) and (cap is None or ln <= cap):
                n += 1
        counts[cap] = n
        prev = cap
    return counts


def _lc_bucket_ready(enriched, target_props):
    """Cheap data gate: True iff some line-conditional prop has a NON-primary line
    bucket with >= MIN_BUCKET_OBS observations — i.e. line-conditional selection
    could actually flip a bucket. Lets us skip the expensive raw-Statcast load on
    a plain --real-lines run until a higher-line bucket has earned a look."""
    for pk in (LINE_CONDITIONAL_PROPS & set(target_props)):
        counts = _lc_bucket_counts(enriched, pk)
        for i, cap in enumerate(LINE_BUCKETS):
            if i == 0:              # the primary bucket alone never triggers a flip
                continue
            if counts.get(cap, 0) >= MIN_BUCKET_OBS:
                return True
    return False


def _select_line_methods(prop_key, enriched, params, sport_key, team_defense,
                         league_avg_def, pooled_method, xba_index, quality_index):
    """Per-line-bucket method selection for a line-conditional prop, or None.

    Builds real-line rows carrying a leakage-safe distributional ``p_dist``,
    buckets them by line (``LINE_BUCKETS``), and for each bucket runs the same
    gated selection (``select_method_at_real_lines``, now D-aware). A bucket
    adopts its OWN method only when: n >= MIN_BUCKET_OBS, the gate-confirmed
    winner differs from the pooled method, AND it beats the pooled method on the
    bucket's single holdout by >= MIN_CALIB_BRIER_GAIN. Otherwise the bucket
    inherits the pooled method (no residuals stored). Returns a ``line_methods``
    list only when at least one bucket adopts its own method, else None (inert)."""
    import book_line_calibration as blc
    from props import _DIST_HARDHIT_COEF, _DIST_BARREL_COEF

    rows = []
    for obs in enriched:
        if not isinstance(obs, dict) or obs.get("prop_key") != prop_key:
            continue
        projected, emp = blc.project_and_empirical(
            obs, params, sport_key, team_defense, league_avg_def)
        if projected is None or emp is None:
            continue
        p_dist = blc.project_distributional(
            obs, params, sport_key, team_defense, league_avg_def,
            xba_index=xba_index, quality_index=quality_index,
            xstats_strength=LINE_COND_XSTATS_STRENGTH)
        if p_dist is None:
            continue
        rows.append({
            "player": obs["player"], "projected": projected, "line": obs["line"],
            "actual": obs["actual"], "empirical_over": emp,
            "game_date": obs["game_date"], "p_dist": p_dist,
        })
    if not rows:
        return None

    line_methods, adopted_any, prev_cap = [], False, None
    for cap in LINE_BUCKETS:
        if cap is None:
            bucket = [r for r in rows if prev_cap is None or r["line"] > prev_cap]
        else:
            bucket = [r for r in rows
                      if (prev_cap is None or r["line"] > prev_cap)
                      and r["line"] <= cap]
        entry = {"max_line": cap, "method": pooled_method}   # default: inherit
        if len(bucket) >= MIN_BUCKET_OBS:
            sel_b = blc.select_method_at_real_lines(bucket)
            single = (sel_b or {}).get("single_split") or {}
            b_best = single.get((sel_b or {}).get("method"))
            b_pooled = single.get(pooled_method)
            if (sel_b and sel_b["confirmed"]
                    and sel_b["method"] != pooled_method
                    and b_best is not None and b_pooled is not None
                    and b_pooled - b_best >= MIN_CALIB_BRIER_GAIN):
                entry = {
                    "max_line": cap, "method": sel_b["method"],
                    "n_obs": sel_b["n_obs"], "fit_brier": sel_b["fit_brier"],
                    "baseline_brier": sel_b["baseline_brier"],
                    "cv_brier": sel_b["cv_brier"], "confirmed": True,
                }
                if sel_b["method"] == "D":
                    entry.update({"xstats_strength": LINE_COND_XSTATS_STRENGTH,
                                  "dist_hardhit_coef": _DIST_HARDHIT_COEF,
                                  "dist_barrel_coef": _DIST_BARREL_COEF})
                else:
                    entry.update({"residual_mu": sel_b["residual_mu"],
                                  "residual_sigma": sel_b["residual_sigma"],
                                  "residual_ecdf": sel_b["residual_ecdf"]})
                adopted_any = True
        line_methods.append(entry)
        prev_cap = cap
    return line_methods if adopted_any else None


def _mlb_player_pool(season, max_batters=40, max_pitchers=30):
    """Resolve a broad, data-driven MLB calibration pool from cached seasons."""
    if not season:
        season = datetime.now(timezone.utc).year
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
        season = datetime.now(timezone.utc).year
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
    Parse a sweep variant key into
    {half_life, opp_defense_strength, output_def_strength, shrink_k,
     venue_strength}. Returns None if the format isn't recognized.

    Two formats are accepted so legacy committed labels stay parseable:
      • legacy 3-part 'hl15/defadj1.0/ven0.25'
          — the pre-P2.1b grid; opp_defense_strength defaults to 0.0 and shrink_k
            to None (== "unspecified" → _build_prop_cfg falls back to the CLI
            --shrinkage-k, since the label never swept it).
      • P2.1b 5-part  'hl15/opp0.5/defadj1.0/shrink5/ven0.25'
          — adds the weight-side opponent-defense and Bayesian-shrinkage knobs
            (both have a props.py runtime, so an enabled knob behaves live as
            validated). shrink_k is an explicit float here — a swept 0 is honored.
    """
    parts = name.split("/")
    if len(parts) not in (3, 5):
        return None
    try:
        # _build_props_sweep_grid emits "none" or "hl<N>" for the half-life.
        hl_part = parts[0]
        if hl_part == "none":
            hl = None
        elif hl_part.startswith("hl"):
            hl = int(hl_part[2:])
        else:
            return None
        if len(parts) == 3:
            da_part, ven_part = parts[1], parts[2]
            opp, shrink = 0.0, None
        else:  # 5-part P2.1b
            opp_part, da_part, shrink_part, ven_part = parts[1:5]
            if not (opp_part.startswith("opp")
                    and shrink_part.startswith("shrink")):
                return None
            opp = float(opp_part[len("opp"):])
            shrink = float(shrink_part[len("shrink"):])
        if not (da_part.startswith("defadj") and ven_part.startswith("ven")):
            return None
        da = float(da_part[len("defadj"):])
        ven = float(ven_part[len("ven"):])
    except (ValueError, IndexError):
        return None
    return {
        "half_life": hl,
        "opp_defense_strength": opp,
        "output_def_strength": da,
        "shrink_k": shrink,
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
            and not p.get("venue_strength") and not p.get("opp_defense_strength")
            and not p.get("shrink_k"))


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
    # P2.1b: shrinkage is now a swept knob (parsed from the winning label). The
    # CLI --shrinkage-k is only a fallback for legacy 3-part labels that carry no
    # shrink token — a candidate that won with shrink=0 keeps 0 (the gate chose
    # it), it is NOT overridden by the CLI default.
    parsed_shrink = parsed.get("shrink_k")
    shrinkage_k = int(parsed_shrink if parsed_shrink is not None
                      else shrinkage_k_default)
    cfg = {
        "method": winner["method"],
        "half_life": parsed.get("half_life"),
        "venue_strength": parsed.get("venue_strength"),
        "opp_defense_strength": parsed.get("opp_defense_strength", 0.0),
        "output_def_strength": parsed.get("output_def_strength", 0.0),
        "shrinkage_k": shrinkage_k,
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
        season = datetime.now(timezone.utc).year
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
        vconf = "confirmed" if winner.get("variant_confirmed") else "baseline"
        print(f"  [{prop_key}] method={cfg['method']} hl={cfg['half_life']} "
              f"opp={cfg['opp_defense_strength']} defadj={cfg['output_def_strength']} "
              f"shrink={cfg['shrinkage_k']} ven={cfg['venue_strength']} "
              f"[{vconf}] brier={cfg.get('fit_brier')} n={cfg.get('n_obs')}")
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
    book_lines, n_primary, n_pred = blc.harvest_real_line_book_lines(
        sport_key, target_props, store_label)
    # Primary source label mirrors harvest's choice (Azure warehouse when SQL is
    # on and no --store-label is forced, else the local backfill store).
    primary_src = "backfill store"
    try:
        import db_store
        if db_store.enabled() and not store_label:
            primary_src = "Azure odds warehouse"
    except Exception:
        pass
    print(f"  Harvested {len(book_lines):,} real book lines for "
          f"{len(target_props)} calibrated prop(s) "
          f"({n_primary:,} from the {primary_src} + {n_pred:,} from the "
          f"prediction log)")
    if not book_lines:
        print("  No real book lines (store or prediction log); nothing to refit.")
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
    # Build the as-of indices to score method D per bucket ONLY when a
    # line-conditional prop has a deep-enough non-primary bucket (cheap data
    # gate) — so a plain --real-lines run doesn't pay the raw-Statcast load until
    # a higher-line bucket has earned a look. Independent of --xstats-strength.
    need_lc = _lc_bucket_ready(enriched, target_props)
    # Surface per-bucket obs counts so it's visible whether line-conditional
    # selection can engage (and, if not, how far the higher-line bucket has to go).
    for _pk in (LINE_CONDITIONAL_PROPS & set(target_props)):
        _counts = _lc_bucket_counts(enriched, _pk)
        _pretty = ", ".join(
            (f"line<={cap}" if cap is not None else "higher lines") + f": {_counts[cap]}"
            for cap in LINE_BUCKETS)
        print(f"  [line-cond] {_pk}: obs by line bucket ({_pretty}) — per-line-bucket "
              f"method selection {'ON' if need_lc else 'OFF'} "
              f"(needs >={MIN_BUCKET_OBS} obs above the primary line)")
    xba_index = None
    lc_quality_index = None
    if (xstats_strength and xstats_strength > 0) or need_lc:
        import savant_history as sh
        import backtest_props
        years = sorted({str(o["game_date"])[:4] for o in enriched
                        if isinstance(o, dict) and o.get("game_date")})
        raw = []
        for y in years:
            try:
                raw.extend(sh.load_days(f"{y}-03-01", f"{y}-11-30"))
            except Exception:
                pass
        if raw:
            xba_index = backtest_props.build_batter_xba_index(raw)
            if need_lc:
                lc_quality_index = backtest_props.build_batter_quality_index(raw)
            print(f"  [xstats] built leakage-safe Statcast index from "
                  f"{len(raw):,} pitches ({', '.join(years)}) for method-D "
                  f"scoring (xBA blend weight={xstats_strength}, "
                  f"per-line-bucket={'on' if need_lc else 'off'})")
        else:
            print("  [xstats] no Statcast days cached for the obs seasons — "
                  "xBA blend + line-conditional D inactive; plain projection.")

    changed = {}
    _change_notes = {}   # prop_key -> short human-readable what-changed note
    kept = []       # method confirmed at real lines; nothing rewritten
    skipped = []    # too few real-line obs; synthetic fit preserved
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
            skipped.append(prop_key)
            print(f"  [skip]   {prop_key}: only {len(rows)} real-line obs "
                  f"(need >=20) — keeping the synthetic-line fit "
                  f"(method {cfg.get('method')})")
            continue

        old_method = cfg.get("method")
        # Data-gated line-conditional selection (batter_hits): may adopt a
        # different method per line bucket. Computed against the POOLED method so
        # a bucket only flips when it genuinely beats pooling on that bucket.
        line_methods = None
        if prop_key in LINE_CONDITIONAL_PROPS and need_lc:
            line_methods = _select_line_methods(
                prop_key, enriched, params, sport_key, team_defense,
                league_avg_def, sel["method"], xba_index, lc_quality_index)

        # Normally only a genuine method FLIP is written (a same-method re-fit
        # would just churn the residuals onto a smaller real-line basis for no
        # runtime change). BUT when the PROJECTION BASIS changed (P2.4a xBA blend
        # applied to this prop) OR a line bucket adopts its own method, we must
        # write.
        projection_changed = prop_xstats > 0
        pooled_flip = sel["method"] != old_method or projection_changed
        if not pooled_flip and not line_methods:
            note = f"real-line eval confirms method {old_method}"
            if "line_methods" not in cfg:
                kept.append(prop_key)
                print(f"  [keep]   {prop_key}: method {old_method} confirmed at "
                      f"real lines (brier {sel['fit_brier']} vs baseline-A "
                      f"{sel['baseline_brier']}, n={sel['n_obs']})")
                continue
            # A previously-written line_methods no longer qualifies → drop it.
            new_cfg = dict(cfg)
            new_cfg.pop("line_methods", None)
            changed[prop_key] = new_cfg
            _change_notes[prop_key] = (f"removed stale per-line-bucket override; "
                                       f"method {old_method} confirmed")
            print(f"  [update] {prop_key}: removed stale per-line-bucket override; "
                  f"{note}")
            continue

        # Preserve variant params, shrinkage_k, variant_label, warmup, etc.;
        # overwrite only the method + its line-invariant residual distribution.
        new_cfg = dict(cfg)
        if pooled_flip:
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
            # Persist the xBA blend weight so props._knob activates it in
            # production at exactly the weight its residuals were re-fit under.
            if prop_xstats > 0:
                new_cfg["xstats_strength"] = prop_xstats
        # Attach (or refresh / drop) the per-line-bucket method map.
        if line_methods:
            new_cfg["line_methods"] = line_methods
        else:
            new_cfg.pop("line_methods", None)
        changed[prop_key] = new_cfg
        note = (f"method {old_method}->{sel['method']} FLIP"
                if sel["method"] != old_method
                else (f"same method, re-fit on real lines (xBA blend "
                      f"{prop_xstats})" if projection_changed
                      else "pooled method unchanged"))
        if line_methods:
            adopted = [f"line<={b['max_line']}:{b['method']}" if b["max_line"]
                       is not None else f"higher:{b['method']}"
                       for b in line_methods if b.get("confirmed")]
            note += f" + per-line-bucket [{', '.join(adopted)}]"
        _change_notes[prop_key] = note
        print(f"  [update] {prop_key}: {note}  (brier {sel['fit_brier']} vs "
              f"baseline-A {sel['baseline_brier']}, cv {sel['cv_brier']}, "
              f"n={sel['n_obs']})")

    # ── At-a-glance recap ──
    total = len(target_props)
    print(f"\n  Summary ({total} prop{'s' if total != 1 else ''} evaluated at real "
          f"book lines): {len(changed)} updated, {len(kept)} confirmed unchanged, "
          f"{len(skipped)} skipped for too little real-line data.")
    for pk in sorted(changed):
        print(f"    updated  {pk}: {_change_notes.get(pk, 're-selected')}")
    if skipped:
        print(f"    skipped: {', '.join(skipped)}")

    if dry_run:
        if changed:
            print(f"\n[dry-run] {len(changed)} prop(s) would be written "
                  f"({sorted(changed.keys())}); nothing saved.")
        else:
            print("\n[dry-run] no props would be re-selected; nothing saved.")
        return

    # The refit evaluated the accumulated resolved data, so flag those
    # prediction-log rows as consumed — this resets the app's "time to refit"
    # banner. Done even when no method changed (the data WAS used), never on a
    # dry-run. Best-effort: a flag-write failure must not fail the refit.
    try:
        import recalibration
        flagged = recalibration.mark_predictions_refit(sport_key)
        if flagged:
            print(f"  Marked {flagged:,} resolved prediction(s) as used by this "
                  f"refit (resets the app's 'time to refit' banner).")
    except Exception:
        pass

    if not changed:
        print("\nNo props had enough real-line data to re-select. "
              "Nothing written (calibration unchanged).")
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


def diagnose_distributional(sport, store_label="", xstats_strength=0.5):
    """§2.4b-2 diagnostic (NO WRITE): score the distributional batter_hits model
    against the shipped method C on the SAME real-line chronological holdout.

    Compares method A (empirical over-rate), method C (pooled residual ECDF — the
    incumbent at real lines), and three distributional variants (empirical-rate
    only; +xBA; +xBA+contact-quality). Reports out-of-sample Brier per variant and
    whether the best distributional variant beats C by >= MIN_CALIB_BRIER_GAIN —
    the go/no-go for wiring the auto-flip (the ship path then confirms under the
    full 2-fold gate + re-seeds Platt). Leakage-safe per-game as-of xBA / quality
    indices. OFFLINE + FREE (store + free ESPN gamelogs + cached raw Statcast
    days). Batter_hits only; writes nothing."""
    import book_line_calibration as blc

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key)
    cfg = (existing or {}).get("batter_hits")
    if not cfg:
        print("No batter_hits calibration to compare against; run refit first.")
        return
    print(f"\n=== §2.4b-2 distributional diagnostic: {sport_key} batter_hits ===")
    book_lines, n_store, n_pred = blc.harvest_real_line_book_lines(
        sport_key, ["batter_hits"], store_label)
    print(f"  {len(book_lines)} book lines ({n_store} backfill store + {n_pred} "
          f"prediction log)")
    if not book_lines:
        print("  No real book lines (store or prediction log); nothing to diagnose.")
        return
    enriched = [o for o in blc.join_book_lines_to_actuals(
        book_lines, espn_sport, espn_league)
        if o.get("prop_key") == "batter_hits"]
    if not enriched:
        print("  No batter_hits observations joined to actuals.")
        return

    # Weight-side opp-defense lookup only if the shipped variant uses it.
    team_defense, league_avg_def = {}, None
    if (cfg.get("opp_defense_strength") or 0.0) > 0:
        team_defense, _, league_avg_def = _team_defense_lookup(
            espn_sport, espn_league)

    # Leakage-safe as-of xBA + contact-quality indices from the raw pitch cache.
    import savant_history as sh
    import backtest_props
    years = sorted({str(o["game_date"])[:4] for o in enriched if o.get("game_date")})
    raw = []
    for y in years:
        try:
            raw.extend(sh.load_days(f"{y}-03-01", f"{y}-11-30"))
        except Exception:
            pass
    if not raw:
        print(f"  [warn] no raw Statcast days cached for {years} — the xBA / "
              f"quality variants will fall back to the empirical rate.")
    xba_index = backtest_props.build_batter_xba_index(raw) if raw else None
    quality_index = backtest_props.build_batter_quality_index(raw) if raw else None

    params = {
        "half_life": cfg.get("half_life"),
        "venue_strength": cfg.get("venue_strength", 0.0),
        "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
        "use_minutes": False,
    }

    import math
    S = xstats_strength
    D_VARIANTS = [
        ("D dist: empirical", {}),
        (f"D dist: +xBA (s={S})", {"xba_index": xba_index}),
    ]

    rows = []
    for obs in enriched:
        projected, emp = blc.project_and_empirical(
            obs, params, sport_key, team_defense, league_avg_def)
        if projected is None or emp is None:
            continue
        base = blc.project_distributional(
            obs, params, sport_key, team_defense, league_avg_def,
            xstats_strength=0.0)
        if base is None:             # no usable AB -> exclude from all variants
            continue
        pv = {}
        for name, kw in D_VARIANTS:
            strength = 0.0 if "empirical" in name else S
            p = blc.project_distributional(
                obs, params, sport_key, team_defense, league_avg_def,
                xstats_strength=strength, **kw)
            pv[name] = p if p is not None else base   # fall back to empirical
        rows.append({
            "obs": obs, "game_date": obs["game_date"], "line": obs["line"],
            "actual": obs["actual"], "projected": projected,
            "empirical_over": emp, "pv": pv,
        })

    # Drop pushes; chronological holdout (mirror evaluate_calibration's split).
    rows = [r for r in rows if r["actual"] != r["line"]]
    if len(rows) < 40:
        print(f"  Only {len(rows)} usable obs (<40) — too thin to judge.")
        return
    rows.sort(key=lambda r: r["game_date"])
    split = len(rows) // 2
    train, test = rows[:split], rows[split:]

    # Pooled residual fit on TRAIN — methods B (Gaussian) and C (ECDF) share it.
    resid = sorted(r["actual"] - r["projected"] for r in train)
    mu = sum(resid) / len(resid)
    var = sum((x - mu) ** 2 for x in resid) / len(resid)
    sigma = math.sqrt(var) if var > 0 else 1e-6

    for r in test:                        # attach outcome + every method's P(over)
        r["o"] = 1 if r["actual"] > r["line"] else 0
        corrected = r["projected"] + mu
        r["m"] = {
            "A empirical": max(0.0, min(1.0, r["empirical_over"])),
            "B pooled Gaussian": blc._norm_cdf((corrected - r["line"]) / sigma),
            "C residual ECDF (shipped)":
                1.0 - blc._empirical_cdf(resid, r["line"] - corrected),
        }
        r["m"].update(r["pv"])

    method_names = (["A empirical", "B pooled Gaussian",
                     "C residual ECDF (shipped)"] + [n for n, _ in D_VARIANTS])

    def _brier(subset, name):
        if not subset:
            return None
        return sum((row["m"][name] - row["o"]) ** 2 for row in subset) / len(subset)

    b05 = [r for r in test if abs(r["line"] - 0.5) < 1e-9]
    buckets = [("all", test), ("line 0.5", b05),
               ("line >=1.5", [r for r in test if r["line"] >= 1.5])]

    # ── 1) Method × line-bucket Brier ──
    print(f"  n_train={len(train)} n_test={len(test)}  "
          f"(out-of-sample Brier by line bucket — lower is better)")
    header = "    {:<32}".format("method")
    for bname, bsub in buckets:
        header += "{:>15}".format(f"{bname}(n={len(bsub)})")
    print(header)
    for name in method_names:
        row_str = "    {:<32}".format(name)
        for _, bsub in buckets:
            br = _brier(bsub, name)
            row_str += "{:>15}".format(f"{br:.4f}" if br is not None else "-")
        print(row_str)
    print()
    for bname, bsub in buckets:
        if not bsub:
            continue
        c_br = _brier(bsub, "C residual ECDF (shipped)")
        best = min(method_names, key=lambda n: _brier(bsub, n))
        best_br = _brier(bsub, best)
        tag = ("C already best" if best.startswith("C")
               else f"{best} beats C by {c_br - best_br:+.4f}")
        print(f"  {bname:<11} n={len(bsub):<4} best: {best} ({best_br:.4f}); "
              f"C={c_br:.4f} -> {tag}")

    # ── 2) Home-team AB-reduction sweep (D dist: empirical, -delta AB on home) ──
    print("\n  Home-team AB-reduction sweep (D dist: empirical; -delta AB on "
          "home games; lower Brier = better):")
    print("    {:<10}{:>14}{:>16}".format("home_dAB", "Brier(all)",
                                          "Brier(line0.5)"))
    for d in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        se_all = se_05 = 0.0
        for r in test:
            p = blc.project_distributional(
                r["obs"], params, sport_key, team_defense, league_avg_def,
                xstats_strength=0.0, home_ab_delta=-d)
            if p is None:
                p = r["m"]["D dist: empirical"]
            e = (p - r["o"]) ** 2
            se_all += e
            if abs(r["line"] - 0.5) < 1e-9:
                se_05 += e
        br_all = se_all / len(test)
        br_05 = se_05 / len(b05) if b05 else None
        print("    {:<10}{:>14}{:>16}".format(
            f"-{d}", f"{br_all:.4f}", f"{br_05:.4f}" if br_05 is not None else "-"))

    # ── 3) Direction split at line 0.5 (the "exclude under 0.5" question) ──
    # Brier is over/under symmetric, so it can't rank a direction. What answers
    # "are under-0.5 picks worth taking" is the realized WIN RATE of the model's
    # predicted side: predict OVER when P(over)>=0.5 (wins if >=1 hit), else UNDER
    # (wins if 0 hits). Below the ~52.4% breakeven at -110, that side loses money.
    if b05:
        base_rate = sum(r["o"] for r in b05) / len(b05)
        print(f"\n  Direction split at line 0.5 (n={len(b05)}, base rate P(>=1 "
              f"hit)={base_rate:.1%}; breakeven ~52.4% @ -110):")
        print("    {:<30}{:>20}{:>20}".format(
            "method", "OVER pick n/win%", "UNDER pick n/win%"))
        for name in ["C residual ECDF (shipped)", f"D dist: +xBA (s={S})"]:
            over = [r for r in b05 if r["m"][name] >= 0.5]
            under = [r for r in b05 if r["m"][name] < 0.5]
            ow = (sum(1 for r in over if r["o"] == 1) / len(over)) if over else None
            uw = (sum(1 for r in under if r["o"] == 0) / len(under)) if under else None
            print("    {:<30}{:>20}{:>20}".format(
                name,
                f"{len(over)}/{ow:.1%}" if ow is not None else f"{len(over)}/-",
                f"{len(under)}/{uw:.1%}" if uw is not None else f"{len(under)}/-"))

    print("\n  (Diagnostic only — nothing written.)")


# ── Conditional-calibration ("reliability by prediction stratum") report ──
# Answers, market-free: "when the model says 60%, does it happen 60%?" and "does
# the model systematically over/under-project in a given line/projection band?"
# — then overlays edge/ROI vs the harvested consensus prices. NO WRITE.
_CC_LINE_BANDS = ["0.5", "1.5", "2.5+"]
_CC_PROJ_BANDS = ["proj<0.75", "0.75-1.25", "1.25-1.75", "proj>=1.75"]


def _cc_num_or_none(x):
    """Coerce a book price to float, or None if missing / non-numeric."""
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _cc_wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion (pure stdlib).
    k = successes (overs), n = trials. Returns (lo, hi, phat)."""
    if n <= 0:
        return (0.0, 1.0, 0.0)
    import math
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half), phat)


def _cc_line_band(line):
    if abs(line - 0.5) < 1e-9:
        return "0.5"
    if abs(line - 1.5) < 1e-9:
        return "1.5"
    return "2.5+"


def _cc_proj_band(proj):
    if proj < 0.75:
        return "proj<0.75"
    if proj < 1.25:
        return "0.75-1.25"
    if proj < 1.75:
        return "1.25-1.75"
    return "proj>=1.75"


def _cc_stratum_table(rowset, pkey, key_name, band_fn, band_order, min_cell_n):
    """Print one stratified summary table: per band -> N, predicted vs realized
    mean (+gap = pred-real, so >0 means the model over-projects), the market edge
    overlay (mean model P(over) - devigged consensus, +coverage), and realized
    ROI/hit-rate for a DIRECTION-AWARE unit-stake strategy that backs the +edge
    side each row (over if p_model > devigged mkt_over, else under). Consensus
    prices (incl. DK @ wt 1), NOT the DK-executable close."""
    print("     {:<11}{:>5}{:>8}{:>8}{:>8}{:>9}{:>11}{:>16}{:>7}".format(
        "band", "N", "pred", "real", "gap", "edge", "cov", "ROI+-(nbets)", "hit"))
    for band in band_order:
        cell = [r for r in rowset if band_fn(r[key_name]) == band]
        n = len(cell)
        if n == 0:
            continue
        pred = sum(r["projected"] for r in cell) / n
        real = sum(r["actual"] for r in cell) / n
        gap = pred - real
        priced = [r for r in cell if r["mkt_over"] is not None]
        edge = (sum(r[pkey] - r["mkt_over"] for r in priced) / len(priced)
                if priced else None)
        # Back the +edge side: over if p_model > devigged mkt_over, else under.
        # Skip exact-parity rows (zero edge, no side). Needs both prices.
        bets = [r for r in cell if r["mkt_over"] is not None
                and r["over_dec"] is not None and r["under_dec"] is not None
                and abs(r[pkey] - r["mkt_over"]) > 1e-9]
        if bets:
            pnl = won = 0.0
            for r in bets:
                if r[pkey] > r["mkt_over"]:          # +edge on the OVER
                    win = (r["o"] == 1)
                    pnl += (r["over_dec"] - 1.0) if win else -1.0
                else:                                # +edge on the UNDER
                    win = (r["o"] == 0)
                    pnl += (r["under_dec"] - 1.0) if win else -1.0
                won += 1.0 if win else 0.0
            roi = pnl / len(bets)
            hit = won / len(bets)
        else:
            roi = hit = None
        flag = " [THIN]" if n < min_cell_n else ""
        edge_s = f"{edge:+.3f}" if edge is not None else "-"
        roi_s = f"{roi * 100:+.1f}%({len(bets)})" if roi is not None else "-"
        hit_s = f"{hit * 100:.1f}%" if hit is not None else "-"
        print("     {:<11}{:>5}{:>8.3f}{:>8.3f}{:>+8.3f}{:>9}{:>11}{:>16}{:>7}{}"
              .format(band, n, pred, real, gap, edge_s, f"{len(priced)}/{n}",
                      roi_s, hit_s, flag))


def _cc_reliability(label, sub, pkey, min_cell_n):
    """Reliability sub-table: bin P(over) into deciles, show empirical over-freq
    and its Wilson 95% CI per bin. Deciles fragment the cell, so flag n<25."""
    print(f"     reliability — {label} (n={len(sub)})")
    print("       {:<12}{:>5}{:>8}{:>20}{:>8}".format(
        "p(over)bin", "n", "emp", "95% CI (Wilson)", "flag"))
    for b in range(10):
        binrows = [r for r in sub
                   if min(9, max(0, int(r[pkey] * 10 + 1e-9))) == b]
        n = len(binrows)
        if n == 0:
            continue
        k = sum(r["o"] for r in binrows)
        lo, hi, phat = _cc_wilson_ci(k, n)
        flag = "[THIN]" if n < 25 else ""
        print("       {:<12}{:>5}{:>8.3f}{:>20}{:>8}".format(
            f"{b / 10:.1f}-{(b + 1) / 10:.1f}", n, phat,
            f"[{lo:.3f}, {hi:.3f}]", flag))


def _cc_report_lens(title, rowset, pkey, min_cell_n):
    print(f"\n  ── {title} ──")
    print("   by LINE band:")
    _cc_stratum_table(rowset, pkey, "line", _cc_line_band,
                      _CC_LINE_BANDS, min_cell_n)
    print('   by PROJECTION band (model projected hit count):')
    _cc_stratum_table(rowset, pkey, "projected", _cc_proj_band,
                      _CC_PROJ_BANDS, min_cell_n)
    print("   reliability (empirical over-freq per P(over) decile; Wilson 95% CI):")
    _cc_reliability("all", rowset, pkey, min_cell_n)
    for lb in _CC_LINE_BANDS:
        sub = [r for r in rowset if _cc_line_band(r["line"]) == lb]
        if sub:
            _cc_reliability(f"line {lb}", sub, pkey, min_cell_n)


def _cc_load_scored_rows(sport, store_label=""):
    """Shared chronological loader for --reliability and --recalibrate.

    Harvests real book lines, joins to actuals, builds one leaner obs per row
    (point projection + method-A empirical over-rate, both leakage-safe / as-of;
    no distributional/xBA machinery, so no AB-gate row loss -> maximal N), drops
    pushes, sorts chronologically, and stamps the binary outcome.

    Prints the harvest/join diagnostics. Returns (sport_key, cfg, rows) where each
    row carries game_date / line / actual / projected / empirical_over / mkt_over /
    over_dec / under_dec / o. Returns rows=None (after printing the reason) on any
    hard miss; cfg=None if there is no batter_hits calibration."""
    import book_line_calibration as blc
    from odds_client import (american_to_implied_prob, devig_two_way,
                             american_to_decimal)

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key)
    cfg = (existing or {}).get("batter_hits")
    if not cfg:
        print("No batter_hits calibration to compare against; run refit first.")
        return sport_key, None, None
    book_lines, n_store, n_pred = blc.harvest_real_line_book_lines(
        sport_key, ["batter_hits"], store_label)
    print(f"  {len(book_lines)} book lines ({n_store} backfill store + {n_pred} "
          f"prediction log)")
    if not book_lines:
        print("  No real book lines (store or prediction log); nothing to report.")
        return sport_key, cfg, None
    enriched = [o for o in blc.join_book_lines_to_actuals(
        book_lines, espn_sport, espn_league)
        if o.get("prop_key") == "batter_hits"]
    if not enriched:
        print("  No batter_hits observations joined to actuals.")
        return sport_key, cfg, None

    # Weight-side opp-defense lookup only if the shipped variant uses it.
    team_defense, league_avg_def = {}, None
    if (cfg.get("opp_defense_strength") or 0.0) > 0:
        team_defense, _, league_avg_def = _team_defense_lookup(
            espn_sport, espn_league)
    params = {
        "half_life": cfg.get("half_life"),
        "venue_strength": cfg.get("venue_strength", 0.0),
        "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
        "use_minutes": False,
    }

    rows = []
    for obs in enriched:
        projected, emp = blc.project_and_empirical(
            obs, params, sport_key, team_defense, league_avg_def)
        if projected is None or emp is None:
            continue
        op = _cc_num_or_none(obs.get("over_price"))
        up = _cc_num_or_none(obs.get("under_price"))
        mkt_over = over_dec = under_dec = None
        if op is not None and up is not None:
            mkt_over = devig_two_way(american_to_implied_prob(op),
                                     american_to_implied_prob(up))[0]
            over_dec = american_to_decimal(op)
            under_dec = american_to_decimal(up)
        rows.append({
            "game_date": obs["game_date"], "line": obs["line"],
            "actual": obs["actual"], "projected": projected,
            "empirical_over": max(0.0, min(1.0, emp)),
            "mkt_over": mkt_over, "over_dec": over_dec, "under_dec": under_dec,
        })

    rows = [r for r in rows if r["actual"] != r["line"]]   # drop pushes
    rows.sort(key=lambda r: r["game_date"])
    for r in rows:
        r["o"] = 1 if r["actual"] > r["line"] else 0
    return sport_key, cfg, rows


def diagnose_conditional_calibration(sport, store_label="", min_cell_n=50):
    """Conditional-calibration ("reliability by prediction stratum") report for
    MLB batter_hits (NO WRITE). For each LINE band (0.5 / 1.5 / 2.5+) and each
    model PROJECTED-count band, report N, predicted vs realized mean (the
    over/under-projection gap), a reliability sub-table (empirical over-frequency
    per P(over) decile with a Wilson 95% CI), and a market EDGE + realized-ROI
    overlay priced at the harvested CONSENSUS book prices (NOT DK-executable).

    Two lenses:
      - Method C (residual-ECDF, shipped) on the chronological TEST half — the
        honest out-of-sample reliability of the production probability.
      - Method A (recency-weighted empirical over-rate) on the FULL sample —
        leakage-safe (as-of prior games only), higher N, the base the model wraps.

    Leakage-safe: chronological split, residual pool fit on TRAIN only. OFFLINE +
    free (durable store / prediction log + free ESPN gamelogs). Writes nothing."""
    import book_line_calibration as blc

    print(f"\n=== Conditional calibration: {SPORT_MAP[sport][2]} batter_hits ===")
    sport_key, cfg, rows = _cc_load_scored_rows(sport, store_label)
    if not cfg or rows is None:
        return
    if len(rows) < 40:
        print(f"  Only {len(rows)} usable obs (<40) — too thin to report.")
        return
    split = len(rows) // 2
    train, test = rows[:split], rows[split:]

    # Method C (shipped): residual pool fit on TRAIN, applied OOS to TEST — mirror
    # diagnose_distributional's construction (mu shift + residual ECDF tail).
    resid = sorted(r["actual"] - r["projected"] for r in train)
    mu = sum(resid) / len(resid)
    for r in test:
        corrected = r["projected"] + mu
        r["p_C"] = 1.0 - blc._empirical_cdf(resid, r["line"] - corrected)
    for r in rows:                       # method A on the full sample (o preset)
        r["p_A"] = r["empirical_over"]

    n_priced = sum(1 for r in rows if r["mkt_over"] is not None)
    print(f"  n_total={len(rows)}  n_train={len(train)}  n_test={len(test)}  "
          f"min_cell_n={min_cell_n}  price_coverage={n_priced}/{len(rows)} "
          f"({100.0 * n_priced / len(rows):.1f}%)  train_residual_mu={mu:+.3f}")
    print("  edge/ROI priced at harvested CONSENSUS book prices (incl. DK @ wt 1),"
          " NOT the DK-executable close. Cells with n<min_cell_n flagged [THIN].")
    print("  ROI+- = unit-stake, backs the +edge side each row (over if p>mkt_over,"
          " else under); hit = win rate of that side.")
    print("  gap = pred_mean - real_mean  (>0 => model over-projects this stratum).")

    _cc_report_lens("Method C — residual ECDF (shipped), OUT-OF-SAMPLE test half",
                    test, "p_C", min_cell_n)
    _cc_report_lens("Method A — recency-weighted empirical over-rate, FULL sample "
                    "(leakage-safe, as-of)", rows, "p_A", min_cell_n)
    print("\n  (Diagnostic only — nothing written.)")


# ── Recalibration: fit a post-hoc map that fixes the over-dispersion the ──
# reliability report surfaced (low P too low, high P too high). Two candidate
# maps, fit on TRAIN, all metrics on held-out TEST, NO WRITE:
#   Platt   — q = sigmoid(a*logit(p) + b). a<1 shrinks p toward the base rate
#             (the exact correction for over-dispersion); 2 params, low variance.
#   Isotonic— nonparametric monotone (pool-adjacent-violators); higher variance,
#             catches non-sigmoidal shape. Winner chosen by OOS log-loss.
def _rc_clamp01(p, eps=1e-6):
    return max(eps, min(1.0 - eps, p))


def _rc_logit(p):
    import math
    p = _rc_clamp01(p)
    return math.log(p / (1.0 - p))


def _rc_sigmoid(z):
    import math
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _rc_fit_platt(ps, os, iters=100):
    """Platt scaling: fit q = sigmoid(a*logit(p) + b) by Newton's method on the
    Bernoulli NLL (1-feature logistic regression, feature = logit(p)). Returns
    (a, b). a<1 => probabilities shrunk toward the base rate (over-dispersion fix),
    a>1 => sharpened. Ridge-stabilised; converges in a handful of steps."""
    xs = [_rc_logit(p) for p in ps]
    a, b = 1.0, 0.0
    for _ in range(iters):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, o in zip(xs, os):
            q = _rc_sigmoid(a * x + b)
            d = q - o
            w = q * (1.0 - q)
            g_a += d * x
            g_b += d
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w
        h_aa += 1e-6
        h_bb += 1e-6                     # ridge for a well-conditioned Hessian
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (g_a * h_bb - g_b * h_ab) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a -= da
        b -= db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return a, b


def _rc_apply_platt(p, a, b):
    return _rc_sigmoid(a * _rc_logit(p) + b)


def _rc_fit_isotonic(ps, os):
    """Isotonic regression via pool-adjacent-violators. Returns (kx, ky), both
    non-decreasing, for monotone linear-interpolation prediction."""
    pairs = sorted(zip(ps, os), key=lambda t: t[0])
    blocks = []                          # each: [sum_y, count, right_x]
    for x, o in pairs:
        blocks.append([float(o), 1, x])
        while (len(blocks) > 1 and
               blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]):
            sy = blocks[-1][0] + blocks[-2][0]
            c = blocks[-1][1] + blocks[-2][1]
            rx = blocks[-1][2]
            blocks.pop()
            blocks[-1] = [sy, c, rx]
    kx = [rx for _, _, rx in blocks]
    ky = [sy / c for sy, c, _ in blocks]
    return kx, ky


def _rc_apply_isotonic(p, knots):
    import bisect
    kx, ky = knots
    if not kx:
        return p
    if p <= kx[0]:
        return ky[0]
    if p >= kx[-1]:
        return ky[-1]
    j = bisect.bisect_right(kx, p)       # kx[j-1] <= p < kx[j]
    i = j - 1
    x0, x1 = kx[i], kx[i + 1] if i + 1 < len(kx) else kx[i]
    y0, y1 = ky[i], ky[i + 1] if i + 1 < len(ky) else ky[i]
    if x1 <= x0:
        return y1
    t = (p - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _rc_metrics(rows, pget):
    """OOS scoring for a probability accessor pget(row): Brier, log-loss, and
    expected calibration error (ECE) over deciles. Lower is better for all."""
    import math
    n = len(rows)
    if n == 0:
        return {"n": 0, "brier": None, "logloss": None, "ece": None}
    brier = sum((pget(r) - r["o"]) ** 2 for r in rows) / n
    ll = 0.0
    for r in rows:
        p = _rc_clamp01(pget(r))
        ll -= math.log(p) if r["o"] == 1 else math.log(1.0 - p)
    ll /= n
    ece = 0.0
    for b in range(10):
        binrows = [r for r in rows
                   if min(9, max(0, int(pget(r) * 10 + 1e-9))) == b]
        if not binrows:
            continue
        mp = sum(pget(r) for r in binrows) / len(binrows)
        mo = sum(r["o"] for r in binrows) / len(binrows)
        ece += (len(binrows) / n) * abs(mp - mo)
    return {"n": n, "brier": brier, "logloss": ll, "ece": ece}


def _rc_edge_summary(rows, pget):
    """Edge-vs-consensus distribution + realized +edge-side ROI for accessor pget,
    over the priced test rows. Shows how far the fake big edges collapse."""
    priced = [r for r in rows if r["mkt_over"] is not None]
    if not priced:
        return None
    edges = [pget(r) - r["mkt_over"] for r in priced]
    absmean = sum(abs(e) for e in edges) / len(edges)
    out = {"n": len(priced), "absmean": absmean,
           "c05": sum(1 for e in edges if abs(e) > 0.05),
           "c10": sum(1 for e in edges if abs(e) > 0.10),
           "c20": sum(1 for e in edges if abs(e) > 0.20),
           "max": max(abs(e) for e in edges)}
    bets = [r for r in priced if r["over_dec"] is not None
            and r["under_dec"] is not None
            and abs(pget(r) - r["mkt_over"]) > 1e-9]
    if bets:
        pnl = won = 0.0
        for r in bets:
            if pget(r) > r["mkt_over"]:
                win = (r["o"] == 1)
                pnl += (r["over_dec"] - 1.0) if win else -1.0
            else:
                win = (r["o"] == 0)
                pnl += (r["under_dec"] - 1.0) if win else -1.0
            won += 1.0 if win else 0.0
        out["roi"] = pnl / len(bets)
        out["hit"] = won / len(bets)
        out["nbets"] = len(bets)
    else:
        out["roi"] = out["hit"] = None
        out["nbets"] = 0
    return out


def _rc_method_for_line(line, line_methods, default_method):
    """Reproduce the runtime per-line-bucket method pick: the finite bucket with
    the smallest max_line >= line, else the catch-all (max_line == null)."""
    if not line_methods:
        return default_method
    finite = sorted((b for b in line_methods if b.get("max_line") is not None),
                    key=lambda b: b["max_line"])
    for b in finite:
        if line <= b["max_line"] + 1e-9:
            return b.get("method", default_method)
    for b in line_methods:
        if b.get("max_line") is None:
            return b.get("method", default_method)
    return default_method


def _rc_run_bucket(name, method, brows, blc, min_cell_n):
    """Fit + OOS-evaluate a recal map on one shipped line-bucket. Method A raw p
    is the as-of empirical over-rate (2-way chronological split); method C raw p
    is the residual-ECDF tail fit on an OLDER pool (3-way split). brows is already
    chronological. Prints before/after; writes nothing."""
    n = len(brows)
    print(f"\n  ── {name}: shipped method {method}, n={n} ──")
    if method == "A":
        if n < 120:
            print(f"     too thin (n={n} < 120) to split + fit; skipped.")
            return
        split = n // 2
        train, test = brows[:split], brows[split:]
        for r in train + test:
            r["p_raw"] = r["empirical_over"]
        print(f"     split: train={len(train)}  test={len(test)} (2-way, "
              f"as-of empirical over-rate)")
    elif method == "C":
        if n < 180:
            print(f"     too thin (n={n} < 180) for a 3-way pool/train/test "
                  f"split; skipped.")
            return
        t1, t2 = n // 3, 2 * n // 3
        pool, train, test = brows[:t1], brows[t1:t2], brows[t2:]
        resid = sorted(r["actual"] - r["projected"] for r in pool)
        mu = sum(resid) / len(resid)
        for r in train + test:
            r["p_raw"] = 1.0 - blc._empirical_cdf(
                resid, r["line"] - (r["projected"] + mu))
        print(f"     split: pool={len(pool)} (residual ECDF, mu={mu:+.3f})  "
              f"train={len(train)}  test={len(test)} (3-way)")
    else:
        print(f"     method {method} unsupported for recalibration; skipped.")
        return

    ps = [r["p_raw"] for r in train]
    osv = [r["o"] for r in train]
    a, b = _rc_fit_platt(ps, osv)
    knots = _rc_fit_isotonic(ps, osv)
    base = sum(osv) / len(osv)
    if a <= 0.0:
        shrink = "INVERTS ordering — overfit, disqualified"
    elif a < 1.0:
        shrink = "shrinks toward base"
    else:
        shrink = "sharpens"
    print(f"     Platt: a={a:.3f} ({shrink}), b={b:+.3f}; train over-rate={base:.3f}")

    def _raw(r):
        return r["p_raw"]

    def _pl(r):
        return _rc_apply_platt(r["p_raw"], a, b)

    def _is(r):
        return _rc_apply_isotonic(r["p_raw"], knots)

    m_raw, m_pl, m_is = (_rc_metrics(test, _raw), _rc_metrics(test, _pl),
                         _rc_metrics(test, _is))
    print("     OOS test metrics (lower = better):")
    print("       {:<10}{:>9}{:>10}{:>8}".format("map", "brier", "logloss", "ece"))
    for lbl, m in [("raw", m_raw), ("platt", m_pl), ("isotonic", m_is)]:
        print("       {:<10}{:>9.4f}{:>10.4f}{:>8.4f}".format(
            lbl, m["brier"], m["logloss"], m["ece"]))
    # Candidate set INCLUDES raw so a map that loses OOS is never displayed as
    # "AFTER" (also blunts winner's-curse: a genuinely-null map can't win by
    # noise). A Platt fit with a<=0 is disqualified — a monotone-DECREASING map
    # inverts the probability ordering, which is always noise-fitting on a thin
    # slice, never a legitimate recalibration.
    cands = [("raw", m_raw, _raw), ("isotonic", m_is, _is)]
    if a > 0.0:
        cands.append(("platt", m_pl, _pl))
    win_lbl, win_m, win_get = min(cands, key=lambda t: t[1]["logloss"])
    if win_lbl == "raw":
        print(f"     winner (OOS log-loss): raw — no map beats raw "
              f"(logloss {m_raw['logloss']:.4f}, ECE {m_raw['ece']:.4f}); "
              f"leave this bucket as-is")
        return
    print(f"     winner (OOS log-loss): {win_lbl} — improves raw "
          f"({m_raw['logloss']:.4f} -> {win_m['logloss']:.4f}, "
          f"ECE {m_raw['ece']:.4f} -> {win_m['ece']:.4f})")

    for r in test:
        r["p_before"] = _raw(r)
        r["p_after"] = win_get(r)
    print("     reliability BEFORE (raw):")
    _cc_reliability(f"{name} raw", test, "p_before", min_cell_n)
    print(f"     reliability AFTER ({win_lbl}):")
    _cc_reliability(f"{name} {win_lbl}", test, "p_after", min_cell_n)

    e_b, e_a = _rc_edge_summary(test, _raw), _rc_edge_summary(test, win_get)
    if e_b and e_a:
        print("     edge vs consensus on priced test rows (fake-edge collapse):")
        print("       {:<10}{:>8}{:>8}{:>9}{:>9}{:>8}{:>14}".format(
            "map", "mean|e|", "|e|>5%", "|e|>10%", "|e|>20%", "max|e|",
            "ROI+-(n)"))
        for lbl, e in [("raw", e_b), (win_lbl, e_a)]:
            roi_s = (f"{e['roi'] * 100:+.1f}%({e['nbets']})"
                     if e["roi"] is not None else "-")
            print("       {:<10}{:>8.3f}{:>8}{:>9}{:>9}{:>8.3f}{:>14}".format(
                lbl, e["absmean"], e["c05"], e["c10"], e["c20"], e["max"],
                roi_s))


def diagnose_recalibration(sport, store_label="", min_cell_n=50):
    """Fit a post-hoc recalibration map (Platt shrinkage + isotonic) on the
    SHIPPED per-line-bucket batter_hits probability, evaluate it OUT-OF-SAMPLE,
    and show the reliability curve flatten and the fake edges collapse (NO WRITE).

    The raw probability reconstructs exactly what production emits per line bucket
    (method A: as-of empirical over-rate; method C: residual-ECDF tail on an older
    pool). Each bucket is split chronologically, the map is fit on TRAIN only, and
    every metric (Brier / log-loss / ECE / reliability / edge) is measured on the
    held-out TEST slice. Leakage-safe. OFFLINE + free. Writes nothing."""
    import book_line_calibration as blc

    print(f"\n=== Recalibration (Platt shrinkage + isotonic): "
          f"{SPORT_MAP[sport][2]} batter_hits ===")
    sport_key, cfg, rows = _cc_load_scored_rows(sport, store_label)
    if not cfg or rows is None:
        return
    if len(rows) < 120:
        print(f"  Only {len(rows)} usable obs (<120) — too thin to recalibrate.")
        return
    line_methods = cfg.get("line_methods")
    default_method = cfg.get("method", "A")
    shipped = ([(bk.get("max_line"), bk.get("method")) for bk in line_methods]
               if line_methods else default_method)
    print(f"  n_total={len(rows)}  shipped line_methods={shipped}")
    print("  Raw p reconstructs the SHIPPED estimator per line bucket, OOS. Recal "
          "map fit on TRAIN; Brier/log-loss/ECE/reliability/edge on held-out TEST.")
    print("  edge/ROI at CONSENSUS prices (incl. DK @ wt 1), NOT DK-executable.")

    groups = {}
    for r in rows:
        m = _rc_method_for_line(r["line"], line_methods, default_method)
        groups.setdefault(m, []).append(r)      # subset preserves chronology
    for m in sorted(groups, key=lambda k: (k != "A", k)):  # dominant A first
        brows = groups[m]
        lines = sorted(set(r["line"] for r in brows))
        band = ("line 0.5" if lines == [0.5]
                else f"lines {min(lines):g}-{max(lines):g}")
        _rc_run_bucket(band, m, brows, blc, min_cell_n)

    print("\n  (Diagnostic only — nothing written. To deploy: store the winning "
          "map per line-bucket and apply g(p) after calibrate_prob.)")


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
    p.add_argument("--dist-diag", action="store_true",
                   help="§2.4b-2: score the distributional batter_hits model vs "
                        "method C on the real-line holdout (no write).")
    p.add_argument("--dist-xstats-strength", type=float, default=0.5,
                   help="xBA blend weight for the --dist-diag xBA variants "
                        "(default 0.5).")
    p.add_argument("--reliability", action="store_true",
                   help="Conditional-calibration report for batter_hits: "
                        "reliability (are 60-percent predictions right 60 percent "
                        "of the time?), realized vs predicted mean, and a market "
                        "edge/ROI overlay, stratified by line and projected-count "
                        "band (no write).")
    p.add_argument("--min-cell-n", type=int, default=50,
                   help="Minimum obs per stratum before a --reliability cell is "
                        "trusted; smaller cells are still printed and flagged "
                        "[THIN].")
    p.add_argument("--recalibrate", action="store_true",
                   help="Fit a post-hoc recalibration map (Platt shrinkage + "
                        "isotonic) on the SHIPPED per-line-bucket batter_hits "
                        "probability and show, OUT-OF-SAMPLE, the reliability curve "
                        "flatten and the fake edges collapse (no write).")
    args = p.parse_args()

    # Target the SQL backend when the SQL_* secrets are configured (mirrors the
    # app's boot promotion + forward_tracker; outside Streamlit these aren't in
    # the env yet). Without this the offline refit's mark_predictions_refit
    # (banner reset) would write the LOCAL log instead of prod SQL. Falls back to
    # Blob/local when SQL isn't configured.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

    if args.dist_diag:
        diagnose_distributional(args.sport, store_label=args.store_label,
                                xstats_strength=args.dist_xstats_strength)
        return

    if args.reliability:
        diagnose_conditional_calibration(args.sport, store_label=args.store_label,
                                         min_cell_n=args.min_cell_n)
        return

    if args.recalibrate:
        diagnose_recalibration(args.sport, store_label=args.store_label,
                               min_cell_n=args.min_cell_n)
        return

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
