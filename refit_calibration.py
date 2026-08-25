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
    _build_props_sweep_grid, _build_focused_props_grid,
    _evaluate_calibration_methods,
    _score_calibration_methods, _team_defense_lookup,
    _mlb_warehouse_defense_lookup,
    run_player_props_backtest,
)
from calibration_loader import (
    load_calibration, save_calibration, set_candidate_mode, active_write_label,
    has_candidate, promote_calibration, discard_candidate, diff_calibration,
    candidate_path, load_calibration_for_refit, existing_candidate_notice,
)
from espn_cache import seed_athlete_id
from espn_client import list_season_athletes


def _defense_lookup(espn_sport, espn_league, season_year=None):
    """Opponent-defense lookup for the calibration/diagnostic paths.

    MLB routes to the warehouse (mlb_game scores, canonical StatsAPI names that
    match the warehouse-joined obs) so no calibration path touches ESPN — the
    live-analog of the sweep's own defense source. Other sports keep the ESPN
    schedule lookup. Same (avg_lookup, series_lookup, league_avg) return shape.
    """
    if espn_sport == "baseball":
        return _mlb_warehouse_defense_lookup(season_year=season_year)
    return _team_defense_lookup(espn_sport, espn_league, season_year=season_year)


def _defense_by_season(espn_sport, espn_league, enriched):
    """Per-season pooled opponent-defense map {year-str: (avg_lookup, league_avg)}
    for the obs seasons, so each obs re-weights against its OWN season only (no
    cross-season/future leak). Mirrors the season-scoped synthetic sweep. {} if no
    dated obs."""
    seasons = sorted({(o.get("game_date") or "")[:4] for o in enriched
                      if isinstance(o, dict) and o.get("game_date")})
    out = {}
    for yr in seasons:
        try:
            avg, _, lg = _defense_lookup(espn_sport, espn_league, season_year=int(yr))
            out[yr] = (avg, lg)
        except Exception:
            out[yr] = ({}, None)
    return out


# A non-empirical method (B pooled-Gaussian / C pooled-ECDF) must beat the
# empirical baseline (method A) by at least this much holdout Brier AND confirm
# out-of-sample in two expanding chronological folds before it can be selected.
# Without this gate, argmin-Brier over ~250 (variant × method) candidates on a
# single split is winner's-curse selection: a method no better than empirical
# gets shipped and advertises an optimistic fit_brier. See P1.4.
MIN_CALIB_BRIER_GAIN = 0.002

# Max Statcast days a single --real-lines refit will auto-fetch to fill the durable
# SQL store (savant_history.ensure_days) before building the xBA index. Bounds a
# first run (before the one-time bulk backfill) to the RECENT gap instead of a slow
# multi-season Savant pull; a larger gap is left for `savant_history.py --ensure`.
STATCAST_GAPFILL_CAP = 45

# ── Incumbent protection at real book lines (anti-churn on thin samples) ──
# The 2-fold real-line confirmation gate is unreliable below a few hundred obs:
# a thin prop's fold verdict flips run-to-run as the chronological split boundary
# shifts (observed live: pitcher_strikeouts read B->A one day and B-crushes-A the
# next, both at n~300). Re-selecting on that noise churns the shipped method for
# no real gain. So a real-line result may OVERRIDE the shipped method only when
# the sample is deep enough to trust the gate (see _incumbent_protected). Deep
# props (batter_hits, n~3.7k) are unaffected; thin props (pitcher_*, n~200-330
# today) hold their incumbent until they earn the override. Conservative floor —
# comfortably above every thin pitcher prop and far below batter_hits; revisit as
# the warehouse accrues. This is a gate PARAMETER, so a --real-lines write that
# changes it needs the same explicit approval as any other calibration write.
MIN_REAL_LINE_OVERRIDE_OBS = 500

# ── ROI tiebreaker within the Brier noise band (§2.6-adjacent) ──
# When >=2 calibration methods land within MIN_CALIB_BRIER_GAIN of each other on
# the single holdout (i.e. Brier can't tell them apart — including a confirmed
# method within the band of empirical A), break the tie by realized flat-1u ROI
# through the LIVE edge+EV recommendation gate (refit_calibration._roi_sim_method).
# Brier stays primary and every anti-winner's-curse guard is intact: only methods
# that already pass the 2-fold OOS confirmation are ROI-eligible, and an override
# fires only when the ROI winner clears a bet floor AND beats the Brier leader's
# ROI by a real margin (no flip-flopping on ROI noise). Prices are de-vigged
# CONSENSUS (best-of-book), so only the RELATIVE method ranking is trusted, never
# the absolute ROI. Unpriced obs -> the tiebreak no-ops and Brier decides.
ROI_TIEBREAK_MIN_BETS = 15         # min simulated value-bets on the test half for a method's ROI to count
ROI_TIEBREAK_MIN_ROI_GAIN = 0.02   # ROI winner must beat the Brier leader's ROI by >= this (flat-unit ROI)
ROI_TIEBREAK_THRESHOLD = 0.05      # edge threshold for the sim (matches diagnose_roi's default / live gate)

# ── Data-gated line-conditional method selection (§2.4b-2 follow-up) ──
# The best calibration method can be line-dependent (which method wins in which
# line bucket is data-driven and regime-dependent — e.g. D dist:+xBA has won the
# line<=0.5 bucket on 2024-2025). refit_sport_real_lines picks the method PER
# LINE BUCKET for these props, but a bucket adopts its own method only when it
# has enough obs AND clears the confirmation gate AND beats the pooled method on
# that bucket — else it inherits the pooled method. Ships inert until a bucket
# earns it.
LINE_CONDITIONAL_PROPS = {"batter_hits"}
LINE_BUCKETS = [0.5, 1.5, None]     # ascending max_line; None = open-ended top.
# Buckets: <=0.5 ("gets one"), (0.5,1.5] ("2+"), >1.5. The 1.5 split was added so
# RBI/TB (and any prop) can adopt a distinct method at the common 1.5 line instead
# of lumping it with 2.5/3.5 in one top bucket. Global, but the per-bucket adopt
# guard (n>=MIN_BUCKET_OBS + gate-confirmed winner beats pooled by
# MIN_CALIB_BRIER_GAIN) means a prop only gets the finer split where it actually
# helps -- an unhelpful bucket just inherits the pooled method (inert).
# Per-bucket floor before a bucket can flip. Comfortably above the 2-fold
# confirmation gate's own minimum (~60 obs to form the two expanding folds) for a
# more robust confirmation, but below the whole-prop obs count so a higher-line
# bucket can qualify as it accrues. The 2-fold gate is still the winner's-curse
# safeguard: a bucket only flips if its winner beats empirical in BOTH folds AND
# beats the pooled method on the bucket by >= MIN_CALIB_BRIER_GAIN.
MIN_BUCKET_OBS = 100
LINE_COND_XSTATS_STRENGTH = 0.5    # default xBA weight for method-D bucket
                                   # scoring/serving; used only when the caller
                                   # passes no --xstats-strength (>0 overrides it,
                                   # so the pooled + D-bucket xBA weights agree)


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


def _line_bucket_key(line):
    """Book line -> LINE_BUCKETS key: 'le_<cap:g>' for the first finite cap >= line,
    else 'top' (open-ended). Pure; matches props._resolve_line_bucket's canonical
    keys exactly. Returns None when line is unusable (None / non-numeric).
    Diagnostic-only helper (used by the per-line-bucket --feature-diag lens); the
    WRITE-path selectors keep their own inline cap-walks. The invariant that all
    three agree is pinned by test_feature_diag.test_keys_match_props_resolver."""
    try:
        ln = float(line)
    except (TypeError, ValueError):
        return None
    for cap in LINE_BUCKETS:            # ascending caps; None = open-ended top
        if cap is None:
            return "top"
        if ln <= cap:
            return f"le_{cap:g}"        # 0.5 -> 'le_0.5'
    return "top"


def _partition_rows_by_bucket(rows):
    """Group real-line obs rows (dicts carrying 'line') by LINE_BUCKETS key, in
    canonical bucket order (finite caps ascending, then 'top'). Rows with an
    unusable line are dropped; a key is present only when it has >= 1 row. Pure — does
    not mutate rows. Diagnostic-only (per-bucket --feature-diag lens)."""
    grouped = {}
    for r in rows:
        key = _line_bucket_key(r.get("line"))
        if key is not None:
            grouped.setdefault(key, []).append(r)
    ordered = {}
    for cap in LINE_BUCKETS:
        key = "top" if cap is None else f"le_{cap:g}"
        if key in grouped:
            ordered[key] = grouped[key]
    return ordered


def _select_line_methods(prop_key, enriched, params, sport_key, team_defense,
                         league_avg_def, pooled_method, xba_index, quality_index,
                         roi_tiebreak=True, defense_by_season=None,
                         xstats_strength=LINE_COND_XSTATS_STRENGTH):
    """Per-line-bucket method selection for a line-conditional prop, or None.

    Builds real-line rows carrying a leakage-safe distributional ``p_dist``,
    buckets them by line (``LINE_BUCKETS``), and for each bucket runs the same
    gated selection (``select_method_at_real_lines``, now D-aware). A bucket
    adopts its OWN method only when: n >= MIN_BUCKET_OBS, the gate-confirmed
    winner differs from the pooled method, AND it beats the pooled method on the
    bucket's single holdout by >= MIN_CALIB_BRIER_GAIN. Otherwise the bucket
    inherits the pooled method (no residuals stored). Returns a ``line_methods``
    list only when at least one bucket adopts its own method, else None (inert).

    ``xstats_strength`` is the xBA blend weight used both to SCORE and to SERVE
    method D in a bucket. The caller threads the operator's per-prop
    --xstats-strength here so the D bucket agrees with the pooled path (the D
    bucket carries the MAJORITY of a prop's volume — a mismatch would silently
    serve a different xBA weight than the pooled method was fit under). Falls back
    to ``LINE_COND_XSTATS_STRENGTH`` when the caller passes no strength."""
    import book_line_calibration as blc
    from props import _DIST_HARDHIT_COEF, _DIST_BARREL_COEF, PROP_NEGBIN_ELIGIBLE

    rows = []
    for obs in enriched:
        if not isinstance(obs, dict) or obs.get("prop_key") != prop_key:
            continue
        projected, emp = blc.project_and_empirical(
            obs, params, sport_key, team_defense, league_avg_def,
            defense_by_season=defense_by_season)
        if projected is None or emp is None:
            continue
        p_dist = blc.project_distributional(
            obs, params, sport_key, team_defense, league_avg_def,
            xba_index=xba_index, quality_index=quality_index,
            xstats_strength=xstats_strength,
            defense_by_season=defense_by_season)
        if p_dist is None:
            continue
        rows.append({
            "player": obs["player"], "projected": projected, "line": obs["line"],
            "actual": obs["actual"], "empirical_over": emp,
            "game_date": obs["game_date"], "p_dist": p_dist,
            # Book prices for the ROI tiebreaker (None when unpriced); harmless to
            # existing consumers, used only inside select_method_at_real_lines.
            "over_price": obs.get("over_price"),
            "under_price": obs.get("under_price"),
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
            # Thread negbin_eligible so E is scored per-bucket: without it a
            # pooled method of "E" is unscorable here (single.get("E") is None
            # -> the adopt guard below fails for EVERY bucket -> line_methods
            # returns None -> the merge silently DROPS a live override). With it,
            # a bucket can also legitimately adopt E on its own.
            sel_b = blc.select_method_at_real_lines(
                bucket, negbin_eligible=(prop_key in PROP_NEGBIN_ELIGIBLE),
                roi_tiebreak=roi_tiebreak)
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
                    entry.update({"xstats_strength": xstats_strength,
                                  "dist_hardhit_coef": _DIST_HARDHIT_COEF,
                                  "dist_barrel_coef": _DIST_BARREL_COEF})
                elif sel_b["method"] == "E":
                    entry.update({"mean_scale": sel_b.get("mean_scale"),
                                  "dispersion": sel_b.get("dispersion")})
                else:
                    entry.update({"residual_mu": sel_b["residual_mu"],
                                  "residual_sigma": sel_b["residual_sigma"],
                                  "residual_ecdf": sel_b["residual_ecdf"]})
                adopted_any = True
        line_methods.append(entry)
        prev_cap = cap
    return line_methods if adopted_any else None


def _incumbent_protected(sel, old_method, min_override_obs=MIN_REAL_LINE_OVERRIDE_OBS):
    """Return a reason string when a proposed real-line method FLIP away from the
    shipped ``old_method`` should be SUPPRESSED (keep the incumbent), else None.

    Two anti-churn guards (see MIN_REAL_LINE_OVERRIDE_OBS):
      • thin sample — ``sel['n_obs'] < min_override_obs``: the 2-fold gate is too
        noisy at this depth to trust an override.
      • not-better-than-incumbent — the newly-selected method scores WORSE than
        the incumbent on the current single chronological split. The selector can
        land on the safe baseline A over a synthetic-sweep B that beats A pooled
        but loses a confirmation fold; flipping B->A there would drop a method
        that is actually stronger on the split for no gain.

    Never fires (returns None) when no flip is proposed, when there is no
    incumbent to protect (``old_method`` falsy — a brand-new prop is fit by the
    synthetic sweep first), or when ``sel`` is falsy. Pure/side-effect-free so it
    is unit-testable in isolation; the caller reverts ``sel['method']`` on a hit."""
    if not sel or not old_method or sel.get("method") == old_method:
        return None
    n_obs = sel.get("n_obs") or 0
    if n_obs < min_override_obs:
        return (f"real-line n={n_obs} < {min_override_obs} — too thin to override "
                f"incumbent {old_method}")
    ss = sel.get("single_split") or {}
    inc_b, pick_b = ss.get(old_method), ss.get(sel.get("method"))
    if inc_b is not None and pick_b is not None and pick_b > inc_b + 1e-9:
        return (f"selected {sel.get('method')} (single-split brier {pick_b:.4f}) "
                f"not better than incumbent {old_method} ({inc_b:.4f})")
    return None


def _mlb_player_pool(season, max_batters=40, max_pitchers=30):
    """Resolve a broad, data-driven MLB calibration pool from cached seasons.

    Returns a list of (mlb_id, role, name) tuples — role ∈ {'batter','pitcher'}, from
    which of frequent_batter_ids / starter_ids the id came — so the P3/P6 warehouse
    sweep binds each player by his AUTHORITATIVE MLBAM id + role (get_calib_gamelog)
    with no lossy name round-trip. The name is retained for display + the ESPN sweep
    path (which resolves name→ESPN id); deduped by id (an id is a player's identity,
    so a shared fullName keeps both, unlike the old name-dedup)."""
    if not season:
        season = datetime.now(timezone.utc).year
    try:
        import mlb_starters
        from backtest_props import frequent_batter_ids, starter_ids

        batter_ids = frequent_batter_ids([season], max_batters)
        pitcher_ids = starter_ids([season])[:max_pitchers]
        role_by_id = {}
        for pid in batter_ids:
            role_by_id.setdefault(str(pid), "batter")
        for pid in pitcher_ids:
            role_by_id.setdefault(str(pid), "pitcher")   # two-way rarity: first wins
        player_ids = list(batter_ids) + list(pitcher_ids)
        name_by_id = {}
        for start in range(0, len(player_ids), 50):
            chunk = player_ids[start:start + 50]
            data = mlb_starters._get(
                "people", {"personIds": ",".join(map(str, chunk))})
            for person in data.get("people", []):
                pid, full = person.get("id"), person.get("fullName")
                if pid and full:
                    name_by_id[str(pid)] = full
        pool, seen = [], set()
        for pid in player_ids:
            spid = str(pid)
            name = name_by_id.get(spid)
            if not name or spid in seen:
                continue
            seen.add(spid)
            pool.append((spid, role_by_id.get(spid, "batter"), name))
        return pool
    except Exception as exc:
        print(f"  [warn] broad MLB player pool unavailable: {exc}")
        return []


def _mlb_pool_union(seasons, max_batters=40, max_pitchers=30):
    """Union the per-season MLB pools across `seasons`, deduped by (mlb_id, role).

    A multi-season refit pools each season's top-N by volume; the UNION is what
    lets a player active in only one of the pooled seasons still contribute his
    games (and naturally widens the thin pitcher pool — the whole reason we pool).
    Order is season-then-rank so the newest season's leaders sort first."""
    pool, seen = [], set()
    for sy in seasons:
        for entry in _mlb_player_pool(sy, max_batters=max_batters,
                                      max_pitchers=max_pitchers):
            key = (entry[0], entry[1])       # (mlb_id, role) — identity
            if key in seen:
                continue
            seen.add(key)
            pool.append(entry)
    return pool


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


def _nba_pool_union(seasons, max_players=150, min_games=15):
    """Union the per-season NBA pools across `seasons`, deduped by name."""
    pool, seen = [], set()
    for sy in seasons:
        for name in _nba_player_pool(sy, max_players=max_players,
                                     min_games=min_games):
            if name in seen:
                continue
            seen.add(name)
            pool.append(name)
    return pool


def _parse_variant_name(name):
    """
    Parse a sweep variant key into
    {half_life, opp_defense_strength, output_def_strength, shrink_k,
     venue_strength, rest_strength}. Returns None if the format isn't recognized.

    Three formats are accepted so legacy committed labels stay parseable:
      • legacy 3-part 'hl15/defadj1.0/ven0.25'
          — the pre-P2.1b grid; opp_defense_strength defaults to 0.0 and shrink_k
            to None (== "unspecified" → _build_prop_cfg falls back to the CLI
            --shrinkage-k, since the label never swept it).
      • P2.1b 5-part  'hl15/opp0.5/defadj1.0/shrink5/ven0.25'
          — adds the weight-side opponent-defense and Bayesian-shrinkage knobs
            (both have a props.py runtime, so an enabled knob behaves live as
            validated). shrink_k is an explicit float here — a swept 0 is honored.
      • §2.6 6-part   'hl15/opp0.5/defadj1.0/shrink5/ven0.25/rest1.0'
          — appends a candidate-FEATURE axis (rest/days-off, prop_features). The
            token is optional; 3/5-part labels get rest_strength 0.0 (off).
    """
    parts = name.split("/")
    if len(parts) not in (3, 5, 6):
        return None
    rest_s = 0.0
    try:
        # Optional trailing /rest<r> feature token (§2.6); trim then reuse the
        # 3/5-part parse below unchanged.
        if len(parts) == 6:
            rest_part = parts[5]
            if not rest_part.startswith("rest"):
                return None
            rest_s = float(rest_part[len("rest"):])
            parts = parts[:5]
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
        "rest_strength": rest_s,
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


def _method_brier_by_fold(obs, method, negbin_eligible=False):
    """Out-of-sample Brier of `method` in each confirmation fold (A/B/C, and E
    when `negbin_eligible`)."""
    briers = []
    for fit_obs, score_obs in _chronological_folds(obs):
        evals = _score_calibration_methods(fit_obs, score_obs, (),
                                           negbin_eligible=negbin_eligible)
        by_method = {e["method"]: e for e in evals
                     if e["k"] in (None, 0) and e["brier"] is not None}
        briers.append(by_method)
    return briers


def _confirms_over_baseline(obs, method, negbin_eligible=False):
    """A non-empirical method must beat method A out-of-sample in BOTH folds.

    This is the guard that defeats winner's-curse selection: noise that makes a
    method win the single holdout split will not also beat empirical in two
    independent later folds. Returns False when there isn't enough data to
    confirm (so the safe empirical baseline is used instead).
    """
    folds = _method_brier_by_fold(obs, method, negbin_eligible=negbin_eligible)
    if not folds:
        return False
    for by_method in folds:
        a = by_method.get("A")
        m = by_method.get(method)
        if not a or not m or m["brier"] >= a["brier"]:
            return False
    return True


def _cv_brier(obs, method, negbin_eligible=False):
    """Mean out-of-sample Brier of `method` across the confirmation folds.

    Persisted as a less-biased estimate of the DEPLOYED calibration's quality
    than the single-split holdout Brier (which is the argmax-selected winner's
    optimistic number). Returns None when folds are unavailable.
    """
    folds = _method_brier_by_fold(obs, method, negbin_eligible=negbin_eligible)
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
            and not p.get("shrink_k") and not p.get("rest_strength"))


def _baseline_variant_obs(results, prop_key):
    """calib_obs for the baseline (all-off) variant, for the variant gate."""
    for vname, by_prop in results.items():
        if _is_baseline_variant(vname):
            return by_prop.get(prop_key, {}).get("calib_obs") or []
    return []


def _variant_confirms(cand_obs, base_obs, cand_method, negbin_eligible=False):
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
        ce = {e["method"]: e for e in _score_calibration_methods(
                  cf, cs, (), negbin_eligible=negbin_eligible)
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
    from props import PROP_NEGBIN_ELIGIBLE
    winners = {}
    for prop_key in props:
        # §2.2: a whitelisted count prop admits method E (NegBin) as a candidate in
        # the SYNTHETIC sweep too (not just the real-line path) — the fix that lets
        # a count prop with NO stored book lines (e.g. batter_total_bases) select
        # the count model. Non-count props are unaffected (E never scored).
        negbin_eligible = prop_key in PROP_NEGBIN_ELIGIBLE
        base_obs = _baseline_variant_obs(results, prop_key)
        best = None
        for vname, by_prop in results.items():
            obs = by_prop[prop_key].get("calib_obs") or []
            if not obs:
                continue
            is_baseline = _is_baseline_variant(vname)
            evals = _evaluate_calibration_methods(
                obs, k_values, holdout=True, negbin_eligible=negbin_eligible)
            by_method = {}
            for e in evals:
                if e["brier"] is None or e["k"] not in (None, 0):
                    continue
                # Persist non-shrinkage methods A, B, C (+ E for count props) —
                # per-player shrinkage variants (B*, C*) overfit out-of-sample per
                # the NBA holdout sweep.
                if e["method"] in ("A", "B", "C", "E"):
                    by_method[e["method"]] = e
            baseline = by_method.get("A")
            for method, e in by_method.items():
                if method != "A":
                    # Fancier methods must clear the baseline margin AND confirm.
                    if (baseline is None
                            or baseline["brier"] - e["brier"] < MIN_CALIB_BRIER_GAIN):
                        continue
                    if not _confirms_over_baseline(
                            obs, method, negbin_eligible=negbin_eligible):
                        continue
                # P2.1 variant gate: a non-baseline knob combo must ALSO beat the
                # baseline variant out-of-sample in both folds — else it's likely a
                # single-split winner's-curse and we keep the baseline (the floor).
                # Only active when the sweep actually contains the baseline cell
                # (always true in the real grid; skipped in narrow unit fixtures).
                if (base_obs and not is_baseline
                        and not _variant_confirms(
                            obs, base_obs, method,
                            negbin_eligible=negbin_eligible)):
                    continue
                if best is None or e["brier"] < best["brier"]:
                    best = {
                        "variant": vname,
                        "method": method,
                        "brier": e["brier"],
                        "hit": e["hit"],
                        "baseline_brier": (round(baseline["brier"], 4)
                                           if baseline else None),
                        "cv_brier": _cv_brier(
                            obs, method, negbin_eligible=negbin_eligible),
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
    # §2.2: method E ships the NegBin (mean_scale, dispersion) props.py serves
    # (_negbin_over_rate), fit on the winning variant's obs via the SAME shared
    # fitter the real-line path uses. Residual mu/sigma/ecdf stay in the cfg
    # (harmless — E serving ignores them; keeps n_obs/provenance uniform).
    if winner["method"] == "E":
        import stats
        nb = stats.fit_negbin_params(
            [(o[1], o[3]) for o in obs])         # (projected, actual)
        if nb is not None:
            fit = dict(fit)
            fit["mean_scale"], fit["dispersion"] = nb
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
        # §2.6 candidate-feature axis (0.0 = off; props.py reads this knob).
        "rest_strength": parsed.get("rest_strength", 0.0),
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
        # Provenance of the shipped numbers (display-only; no reader consumes it).
        # Every prop starts life fit at the SYNTHETIC season-average line here; the
        # real-line pass (below) promotes this to "real_line" only when it flips a
        # prop on genuine book-line obs. Keeps the app from showing stale synthetic
        # pitcher Briers as if they were measured on real lines.
        "fit_basis": "synthetic_sweep",
    }
    cfg.update(fit)
    return cfg


def _merge_props_results(acc, new):
    """Pool two ``run_player_props_backtest`` results dicts (same variant×prop
    grid) into ``acc`` so a multi-season refit fits on the COMBINED calib_obs.

    Sums the tally counters and concatenates ``calib_obs`` / ``errors``. Each
    season's obs were projected strictly WITHIN that season (the backtest runs
    per-season with ``cross_season='strict'``), so pooling adds sample without
    introducing cross-season projection leakage — every residual still came from
    a same-season prior slice. Winner selection (``_best_per_prop``) and the
    residual fit (``_build_prop_cfg``) then run once over the pooled obs."""
    for vname, by_prop in new.items():
        acc_v = acc.setdefault(vname, {})
        for prop_key, cell in by_prop.items():
            a = acc_v.get(prop_key)
            if a is None:
                acc_v[prop_key] = cell
                continue
            a["errors"].extend(cell.get("errors") or [])
            a["n"] += cell.get("n", 0)
            a["hits"] += cell.get("hits", 0)
            a["decisive"] += cell.get("decisive", 0)
            if a.get("calib_obs") is not None and cell.get("calib_obs"):
                a["calib_obs"].extend(cell["calib_obs"])
            for off, tally in (cell.get("safe") or {}).items():
                dst = a.setdefault("safe", {}).setdefault(
                    off, {"hits": 0, "n": 0})
                dst["hits"] += tally.get("hits", 0)
                dst["n"] += tally.get("n", 0)
            for q, tally in (cell.get("quantile") or {}).items():
                dst = a.setdefault("quantile", {}).setdefault(
                    q, {"hits": 0, "n": 0, "cushions": []})
                dst["hits"] += tally.get("hits", 0)
                dst["n"] += tally.get("n", 0)
                dst["cushions"].extend(tally.get("cushions") or [])
    return acc


def refit_sport(sport, season=None, prior_season=None, players=None, props=None,
                games_per_player=80, warmup_games=10, shrinkage_k_default=0,
                mlb_max_batters=40, mlb_max_pitchers=30,
                nba_max_players=150, nba_min_games=15, seasons=None,
                focused_grid=False):
    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    # Resolve the set of seasons to POOL for the main fit. `--seasons` pools
    # several seasons' residuals into ONE fit (triples the thin pitcher-prop
    # sample + cures the earned_runs method C↔A instability); `--season` alone
    # keeps the single-season fit (byte-identical to the pre-pooling path).
    if seasons:
        fit_seasons = sorted({int(s) for s in seasons})
    elif season is not None:
        fit_seasons = [int(season)]
    elif sport in ("mlb", "nba"):
        fit_seasons = [datetime.now(timezone.utc).year]
    else:
        fit_seasons = [None]
    # The "current" season drives cfg["fit_season"], the warmup boundary, and
    # meta — it is the newest of the pooled set.
    season = max((s for s in fit_seasons if s is not None), default=None)

    if players is None and sport == "mlb":
        players = _mlb_pool_union(
            fit_seasons, max_batters=mlb_max_batters,
            max_pitchers=mlb_max_pitchers)
        if not players:
            print("No data-driven MLB player pool was available; aborting rather "
                  "than fitting all MLB props from the small static fallback.")
            sys.exit(1)
    if players is None and sport == "nba":
        players = _nba_pool_union(
            fit_seasons, max_players=nba_max_players, min_games=nba_min_games)
        if not players:
            print("No data-driven NBA player pool was available; aborting rather "
                  "than fitting NBA props from the 18-star fallback.")
            sys.exit(1)
    players = players or DEFAULT_STARTERS.get(sport)
    props = props or DEFAULT_PROPS.get(sport)
    if not players or not props:
        print(f"No default players/props for {sport}; please pass --players/--props.")
        sys.exit(1)

    variants = (_build_focused_props_grid() if focused_grid
                else _build_props_sweep_grid())
    if focused_grid:
        print(f"[focused-grid] {len(variants)} variants (vs full 576) — full "
              f"resolution on half_life×venue×shrink, dead axes probed once. "
              f"~15× faster/lighter; drops knob-interaction cells.")

    pooled = len(fit_seasons) > 1
    _season_lbl = (", ".join(str(s) for s in fit_seasons) if pooled
                   else str(fit_seasons[0]))
    print(f"\n=== Fitting {'POOLED ' if pooled else 'CURRENT-season '}"
          f"calibration for {sport_key} (seasons: {_season_lbl}, "
          f"{len(players)} players) ===")
    # Run the sweep once per season and MERGE the per-season results so method
    # selection + the residual fit see the COMBINED calib_obs. Each season runs
    # strictly within-season (cross_season='strict'), so pooling never crosses a
    # season boundary inside any one projection — it only widens the sample.
    curr_results = None
    for sy in fit_seasons:
        if pooled:
            print(f"\n--- pool season {sy} ---")
        season_res = run_player_props_backtest(
            sport, espn_sport, espn_league, sport_key,
            players=players, props=props,
            games_per_player=games_per_player,
            min_sample=5, variants=variants, sweep=True,
            season_year=sy, safe_mode=True,
            cushion_sweep=False, safe_target=0.80,
            quantile_mode=False, calibrate=True,
            cross_season="strict",
        )
        if not season_res:
            print(f"  [WARN] pool season {sy} produced no results; excluding it.")
            continue
        curr_results = (season_res if curr_results is None
                        else _merge_props_results(curr_results, season_res))
    if not curr_results:
        print("Calibration run produced no results; aborting.")
        sys.exit(2)

    curr_winners = _best_per_prop(curr_results, props)

    warmup_results = None
    warmup_winners = {}
    if prior_season is not None and prior_season in fit_seasons:
        # A season can't be both a pooled main-fit season AND the warmup prior:
        # its games are already in curr_results, so a separate warmup block would
        # double-count them (and blend the pool against itself). Skip it loudly.
        print(f"\n[note] prior season {prior_season} is already in the pooled fit "
              f"({_season_lbl}); skipping the redundant warmup block.")
    elif prior_season is not None:
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
        else:
            # A requested warmup that yields nothing would otherwise ship a calibration
            # with NO warmup block (degrading players with < warmup_games current-season
            # games) behind a single buried "Aborting." line. Make it loud + auditable —
            # for MLB the likeliest cause is the warehouse lacking prior-season facts.
            print(f"  [WARN] WARMUP (prior season {prior_season}) produced NO results "
                  f"— shipping calibration WITHOUT a warmup block."
                  + (" The MLB warehouse may lack that season's facts; backfill it."
                     if sport == "mlb" else ""))

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
        # The full pooled season set (a single-season fit lists just [season]);
        # provenance for how much sample the shipped residuals were fit on.
        "fit_seasons": fit_seasons,
        "warmup_season": prior_season,
        "games_per_player": games_per_player,
        "warmup_games": warmup_games,
        "n_players": len(players),
        # False when a warmup was requested but yielded no fit (e.g. the warehouse
        # lacks prior-season facts) — an at-a-glance signal that the shipped file has
        # no warmup block, since the omission is otherwise silent.
        "warmup_present": bool(warmup_winners),
    }
    save_calibration(sport_key, props_cfg, meta=meta)
    print(f"\n✓ Wrote calibration/{active_write_label(sport_key)} "
          f"({len(props_cfg)} props)")


def refit_sport_real_lines(sport, store_label="", warmup_games=10,
                           shrinkage_k_default=0, xstats_strength=0.0,
                           dry_run=False, roi_tiebreak=True,
                           min_override_obs=MIN_REAL_LINE_OVERRIDE_OBS,
                           seasons=None):
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
    # Staging-aware: when a candidate is staged (e.g. from a just-run sweep), the
    # real-line pass re-selects methods against THAT candidate's methods/residuals
    # so the two steps compose — mirroring how the pre-staging flow chained
    # through the live file. Falls back to the live props otherwise.
    existing = load_calibration_for_refit(sport_key)
    if not existing:
        print(f"No existing calibration/{active_write_label(sport_key)} props to "
              f"refit; run the synthetic sweep (refit_sport) first.")
        return
    target_props = list(existing.keys())

    print(f"\n=== Re-selecting calibration method at REAL book lines for "
          f"{sport_key} ===")
    book_lines, n_primary, n_pred = blc.harvest_real_line_book_lines(
        sport_key, target_props, store_label)
    # Season scope (e.g. exclude the 2023 pitch-clock-transition regime): keep only
    # book lines whose US-Eastern game_date year is in `seasons`. Applied BEFORE the
    # actuals join so a dropped season never enters the method/residual fit.
    if seasons:
        _yset = {str(s) for s in seasons}
        book_lines = [r for r in book_lines
                      if str(r.get("game_date") or "")[:4] in _yset]
        print(f"  [seasons] real-line fit scoped to {sorted(_yset)} — "
              f"{len(book_lines):,} book lines after the season filter")
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
    # PER SEASON (leakage guard): the obs span multiple seasons, so build one
    # pooled lookup per season and let the project fns self-select each obs's own
    # season — a single all-season lookup would re-weight a 2024 obs against 2025/
    # 2026 (future) defense. Mirrors the synthetic sweep, which is season-scoped.
    defense_by_season = None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in existing):
        _seasons = sorted({(o["game_date"] or "")[:4] for o in enriched
                           if isinstance(o, dict) and o.get("game_date")})
        print(f"  Building per-season team-defense lookups (a variant uses "
              f"opp_defense; seasons: {', '.join(_seasons) or 'none'})...")
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

    # ── P2.4a: leakage-safe as-of xBA index for the projection blend ──
    # Built once from the raw Statcast day cache spanning the obs seasons. Only
    # applies to props in props.PROP_XSTATS_KIND (batter_hits). Uses a per-game
    # as-of index (NOT the current-as-of SQL table) so a past-dated obs never
    # sees future data. xstats_strength=0 → byte-identical to the prior behavior.
    from props import PROP_XSTATS_KIND, PROP_NEGBIN_ELIGIBLE
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
        # Statcast SQL gap-fill (incremental): ensure the obs-season days are ingested
        # to the durable store BEFORE loading them, so the xBA index + method D are
        # available on ANY box (no machine-local cache). Capped so a first run before
        # the one-time bulk backfill fills only the RECENT gap + warns to run the bulk,
        # rather than blocking on a multi-season Savant pull.
        if years:
            _s, _e = f"{years[0]}-03-01", f"{years[-1]}-11-30"
            try:
                _n_new, _n_missing = sh.ensure_days(
                    _s, _e, cap=STATCAST_GAPFILL_CAP, verbose=True)
                if _n_missing > _n_new:
                    print(f"  [statcast] {_n_missing - _n_new} day(s) still missing "
                          f"in {_s}..{_e} (cap={STATCAST_GAPFILL_CAP}) — run "
                          f"`python savant_history.py --ensure --start {_s} "
                          f"--end {_e}` for the one-time bulk backfill.")
            except Exception as _exc:
                print(f"  [statcast] gap-fill skipped: {type(_exc).__name__}: {_exc}")
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
            # §2.6: forward an adopted feature knob so the real-line method
            # re-selection scores the SAME (feature-adjusted) projection the
            # synthetic sweep chose. 0.0 (default) → byte-identical no-op.
            "rest_strength": cfg.get("rest_strength", 0.0),
        }
        prop_xstats = (xstats_strength if (xba_index is not None
                       and prop_key in PROP_XSTATS_KIND) else 0.0)
        rows = blc.build_real_line_obs(
            enriched, params, sport_key, prop_key,
            xstats_strength=prop_xstats, xba_index=xba_index,
            defense_by_season=defense_by_season)
        sel = blc.select_method_at_real_lines(
            rows, negbin_eligible=(prop_key in PROP_NEGBIN_ELIGIBLE),
            roi_tiebreak=roi_tiebreak)
        if sel is None:
            skipped.append(prop_key)
            print(f"  [skip]   {prop_key}: only {len(rows)} real-line obs "
                  f"(need >=20) — keeping the synthetic-line fit "
                  f"(method {cfg.get('method')})")
            continue

        old_method = cfg.get("method")

        # ── Incumbent protection: suppress a thin / not-better real-line flip ──
        # (see _incumbent_protected / MIN_REAL_LINE_OVERRIDE_OBS). Revert sel to
        # the incumbent BEFORE the per-line-bucket pass so buckets inherit and
        # compare against the method we will actually keep, and so pooled_flip
        # below evaluates to False. Only the shipped pooled method is protected;
        # a projection-basis re-fit (xBA blend) and per-bucket overrides are
        # unaffected. Deep props (batter_hits) never trip this.
        protect_note = _incumbent_protected(sel, old_method,
                                            min_override_obs=min_override_obs)
        if protect_note:
            sel["method"] = old_method

        # Data-gated line-conditional selection (batter_hits): may adopt a
        # different method per line bucket. Computed against the POOLED method so
        # a bucket only flips when it genuinely beats pooling on that bucket.
        line_methods = None
        if prop_key in LINE_CONDITIONAL_PROPS and need_lc:
            # Thread the operator's per-prop xBA weight so the D bucket serves the
            # SAME strength as the pooled path (prop_xstats>0 → --xstats-strength;
            # 0 → keep the LINE_COND_XSTATS_STRENGTH default for a bare re-fit).
            lc_xstats = prop_xstats if prop_xstats > 0 else LINE_COND_XSTATS_STRENGTH
            line_methods = _select_line_methods(
                prop_key, enriched, params, sport_key, None,
                None, sel["method"], xba_index, lc_quality_index,
                roi_tiebreak=roi_tiebreak, defense_by_season=defense_by_season,
                xstats_strength=lc_xstats)

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
                if protect_note:
                    print(f"  [keep]   {prop_key}: incumbent {old_method} "
                          f"PROTECTED — {protect_note} (selector proposed a flip; "
                          f"suppressed)")
                else:
                    print(f"  [keep]   {prop_key}: method {old_method} confirmed "
                          f"at real lines (brier {sel['fit_brier']} vs baseline-A "
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
            # §2.2 method "E" (NegBin) ships two extra count-model params instead
            # of consuming the residual distribution above; dispatched at the props
            # seam, not via calibrate_prob/warmup. Persist them when E won.
            if sel["method"] == "E":
                new_cfg["mean_scale"] = sel.get("mean_scale")
                new_cfg["dispersion"] = sel.get("dispersion")
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
            # Provenance flips synthetic -> real: these method/residual numbers were
            # just measured on genuine book-line obs, not the synthetic sweep.
            new_cfg["fit_basis"] = "real_line"
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
        rt = sel.get("roi_tiebreak")
        if rt and rt.get("applied"):
            w, lead = rt["winner"], rt["brier_leader"]
            rw, rl = rt["rois"].get(w), rt["rois"].get(lead)
            note += (f" [ROI tiebreak: {w} over Brier-pick {lead} within noise "
                     f"band {rt.get('tie_set')} — roi {rw:+.3f} vs {rl:+.3f}, "
                     f"n_bets {rt['n_bets'].get(w)}]")
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
    print(f"\n✓ Updated calibration/{active_write_label(sport_key)} "
          f"({len(changed)} prop(s) re-selected at real book lines; "
          f"other props/blocks preserved)")


def diagnose_distributional(sport, store_label="", xstats_strength=0.5,
                            seasons=None):
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
    if seasons:
        _yset = {str(s) for s in seasons}
        book_lines = [r for r in book_lines
                      if str(r.get("game_date") or "")[:4] in _yset]
        print(f"  [seasons] dist-diag scoped to {sorted(_yset)}")
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

    # Weight-side opp-defense lookup only if the shipped variant uses it. PER SEASON
    # (leakage guard): each obs re-weights against its OWN season's pooled defense.
    defense_by_season = None
    if (cfg.get("opp_defense_strength") or 0.0) > 0:
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

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
            obs, params, sport_key, defense_by_season=defense_by_season)
        if projected is None or emp is None:
            continue
        base = blc.project_distributional(
            obs, params, sport_key, xstats_strength=0.0,
            defense_by_season=defense_by_season)
        if base is None:             # no usable AB -> exclude from all variants
            continue
        pv = {}
        for name, kw in D_VARIANTS:
            strength = 0.0 if "empirical" in name else S
            p = blc.project_distributional(
                obs, params, sport_key, xstats_strength=strength,
                defense_by_season=defense_by_season, **kw)
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
                r["obs"], params, sport_key, xstats_strength=0.0,
                home_ab_delta=-d, defense_by_season=defense_by_season)
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


def diagnose_negbin(sport, store_label=""):
    """§2.2 diagnostic (NO WRITE): for each Negative-Binomial-eligible count prop
    that is already calibrated, score method "E" against A/B/C on the SAME real-line
    chronological holdout the ship path uses, and report whether E beats the
    incumbent by >= MIN_CALIB_BRIER_GAIN and whether it would clear the full 2-fold
    confirmation gate.

    This is the roadmap's mandated "benchmark NegBin against the incumbent
    (empirical-C / Gaussian-B) before adopting" — go/no-go for a `--real-lines`
    flip. It reuses book_line_calibration.select_method_at_real_lines(...,
    negbin_eligible=True) verbatim, so the diagnostic and the ship path share ONE
    scoring/gate impl (no drift). OFFLINE + FREE (store + free ESPN gamelogs);
    writes nothing."""
    import book_line_calibration as blc
    from props import PROP_NEGBIN_ELIGIBLE

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key) or {}
    props_to_check = [pk for pk in existing if pk in PROP_NEGBIN_ELIGIBLE]
    if not props_to_check:
        print(f"No NegBin-eligible calibrated props in calibration/{sport_key}.json "
              f"(eligible: {sorted(PROP_NEGBIN_ELIGIBLE)}); run refit first.")
        return

    print(f"\n=== §2.2 Negative-Binomial (method E) diagnostic: {sport_key} ===")
    print(f"  Eligible + calibrated: {', '.join(sorted(props_to_check))}")
    book_lines, n_primary, n_pred = blc.harvest_real_line_book_lines(
        sport_key, props_to_check, store_label)
    print(f"  {len(book_lines):,} real book lines "
          f"({n_primary:,} store + {n_pred:,} prediction log)")
    if not book_lines:
        print("  No real book lines (store or prediction log); nothing to diagnose.")
        return
    enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("  No observations joined to actuals; nothing to diagnose.")
        return

    # Weight-side opponent-defense lookup only if some eligible prop's variant uses
    # it (mirrors refit_sport_real_lines' gating so the projection basis matches).
    # PER SEASON (leakage guard): each obs re-weights against its OWN season.
    defense_by_season = None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in props_to_check):
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

    from props import PROP_XSTATS_KIND
    for prop_key in sorted(props_to_check):
        cfg = existing[prop_key]
        incumbent = cfg.get("method")
        # xBA caveat: E's mean is avg_stat, so this diag MUST score A/B/C/E on the
        # plain projection basis (xstats_strength=0.0) for a like-for-like compare.
        # But if the SHIPPED prop blends xBA (xstats_strength > 0 and xstats-kind),
        # the incumbent's live Brier is measured on a DIFFERENT (xBA) basis than the
        # A/B/C numbers here -> the gap below understates the incumbent. Warn so the
        # reader knows this diag is directional only and --real-lines is the decider.
        ship_xstats = cfg.get("xstats_strength") or 0.0
        xba_shipped = ship_xstats > 0 and prop_key in PROP_XSTATS_KIND
        params = {
            "half_life": cfg.get("half_life"),
            "venue_strength": cfg.get("venue_strength", 0.0),
            "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
            "use_minutes": cfg.get("use_minutes", False),
        }
        # Plain projection basis (no xBA blend) — E's mean is avg_stat, so this is
        # the like-for-like basis for A/B/C/E (the xba_shipped caveat below flags
        # when the incumbent's live basis differs).
        rows = blc.build_real_line_obs(
            enriched, params, sport_key, prop_key,
            xstats_strength=0.0, xba_index=None,
            defense_by_season=defense_by_season)
        sel = blc.select_method_at_real_lines(rows, negbin_eligible=True,
                                              roi_tiebreak=False)
        n_usable = len([r for r in rows if r["actual"] != r["line"]])
        if sel is None:
            print(f"\n  {prop_key}: only {n_usable} usable real-line obs (need "
                  f">=20) — can't score E.")
            continue
        ss = sel.get("single_split", {})
        folds = blc._real_line_folds(
            sorted([r for r in rows if r["actual"] != r["line"]],
                   key=lambda r: r["game_date"]))
        print(f"\n  {prop_key} (incumbent={incumbent}, n_usable={n_usable}, "
              f"folds={len(folds)}):")
        if xba_shipped:
            print(f"    ⚠ CAVEAT: shipped {prop_key} blends xBA "
                  f"(xstats_strength={ship_xstats:.2f}); this diag scores A/B/C/E "
                  f"on PLAIN projections, so the incumbent's Brier here is NOT its "
                  f"live basis — treat the E-vs-{incumbent} gap as directional and "
                  f"let --real-lines decide.")
        print("    holdout Brier — " + "  ".join(
            f"{m}={ss[m]:.4f}" for m in ("A", "B", "C", "D", "E") if m in ss))
        e_br = ss.get("E")
        inc_br = ss.get(incumbent)
        if e_br is not None and inc_br is not None:
            gain = inc_br - e_br
            verdict = (f"E beats incumbent {incumbent} by {gain:+.4f} "
                       f"(>= {MIN_CALIB_BRIER_GAIN} threshold: "
                       f"{'YES' if gain >= MIN_CALIB_BRIER_GAIN else 'no'})")
        else:
            verdict = "E or incumbent Brier unavailable on this split"
        gate = ("E CONFIRMED under the full 2-fold gate — would flip"
                if sel["method"] == "E"
                else f"gate keeps method {sel['method']} (E not confirmed)")
        ms, disp = sel.get("mean_scale"), sel.get("dispersion")
        params_str = (f"mean_scale={ms:.3f}, dispersion={disp:.4f}"
                      if ms is not None and disp is not None else "n/a")
        print(f"    {verdict}")
        print(f"    fitted E params (all usable obs): {params_str}")
        print(f"    {gate}")

    print("\n  (Diagnostic only — nothing written. Run --real-lines to apply the "
          "gate for real.)")


def diagnose_center(sport, prop_filter=None, store_label=""):
    """Mean-vs-median central-tendency diagnostic (NO WRITE).

    A reader suggested that for count stats (which can't go negative) the mean is a
    positively-biased center and the MEDIAN might predict better. This scores every
    calibrated prop's incumbent method on the SAME real-line chronological holdout
    the ship path uses, once with the production recency-weighted MEAN center and
    once with a recency-weighted MEDIAN center (same weights, only the operator
    changes — book_line_calibration._center_estimate), and reports the per-method
    Brier under each and whether MEDIAN beats MEAN on the incumbent by
    >= MIN_CALIB_BRIER_GAIN.

    Reuses book_line_calibration.select_method_at_real_lines verbatim (roi_tiebreak
    off) so this shares ONE scoring/gate impl with the ship path. OFFLINE + FREE
    (store + free ESPN gamelogs); writes nothing.

    Self-check: method A never reads ``projected`` (it returns the empirical
    over-rate at the book line), so A's Brier is IDENTICAL under both centers — a
    built-in proof the harness is isolating the center. Only methods B/C/E (which
    center a distribution on the projection) can move. So a mean->median swap is a
    literal no-op for a prop shipped on method A (e.g. pitcher_earned_runs,
    pitcher_outs) — it only bites the projection-centered methods."""
    import book_line_calibration as blc
    from props import PROP_NEGBIN_ELIGIBLE, PROP_XSTATS_KIND

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key) or {}
    props_to_check = sorted(existing)
    if prop_filter:
        props_to_check = [pk for pk in props_to_check if pk in prop_filter]
    if not props_to_check:
        print(f"No calibrated props to check in calibration/{sport_key}.json"
              + (f" matching {sorted(prop_filter)}" if prop_filter else "")
              + "; run refit first.")
        return

    print(f"\n=== Mean-vs-median center diagnostic: {sport_key} ===")
    print(f"  Checking: {', '.join(props_to_check)}")
    book_lines, n_primary, n_pred = blc.harvest_real_line_book_lines(
        sport_key, props_to_check, store_label)
    print(f"  {len(book_lines):,} real book lines "
          f"({n_primary:,} store + {n_pred:,} prediction log)")
    if not book_lines:
        print("  No real book lines (store or prediction log); nothing to diagnose.")
        return
    enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("  No observations joined to actuals; nothing to diagnose.")
        return

    # PER SEASON (leakage guard): each obs re-weights against its OWN season.
    defense_by_season = None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in props_to_check):
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

    for prop_key in props_to_check:
        cfg = existing[prop_key]
        incumbent = cfg.get("method")
        elig = prop_key in PROP_NEGBIN_ELIGIBLE
        # Plain projection basis (no xBA blend) so mean vs median differ only in the
        # operator. Warn when the shipped prop blends xBA — then the live basis
        # differs from what we score here (directional only).
        ship_xstats = cfg.get("xstats_strength") or 0.0
        xba_shipped = ship_xstats > 0 and prop_key in PROP_XSTATS_KIND
        base_params = {
            "half_life": cfg.get("half_life"),
            "venue_strength": cfg.get("venue_strength", 0.0),
            "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
            "use_minutes": cfg.get("use_minutes", False),
        }

        sels = {}
        for center in ("mean", "median"):
            params = dict(base_params, center=center)
            rows = blc.build_real_line_obs(
                enriched, params, sport_key, prop_key,
                xstats_strength=0.0, xba_index=None,
                defense_by_season=defense_by_season)
            sels[center] = (rows, blc.select_method_at_real_lines(
                rows, negbin_eligible=elig, roi_tiebreak=False))

        rows_mean, sel_mean = sels["mean"]
        rows_med, sel_med = sels["median"]
        n_usable = len([r for r in rows_mean if r["actual"] != r["line"]])
        if sel_mean is None or sel_med is None:
            print(f"\n  {prop_key}: only {n_usable} usable real-line obs "
                  f"(need >=20) — can't score.")
            continue

        ss_mean = sel_mean.get("single_split", {})
        ss_med = sel_med.get("single_split", {})
        print(f"\n  {prop_key} (incumbent={incumbent}, n_usable={n_usable}):")
        if xba_shipped:
            print(f"    ⚠ CAVEAT: shipped {prop_key} blends xBA "
                  f"(xstats_strength={ship_xstats:.2f}); scored here on PLAIN "
                  f"projections, so treat as directional.")
        methods = [m for m in ("A", "B", "C", "D", "E")
                   if m in ss_mean or m in ss_med]
        print("    holdout Brier by method:")
        print("      center  " + "  ".join(f"{m:>8}" for m in methods))
        for center, ss in (("mean", ss_mean), ("median", ss_med)):
            cells = "  ".join(
                (f"{ss[m]:8.4f}" if m in ss else f"{'—':>8}") for m in methods)
            print(f"      {center:<6}  {cells}")

        # Incumbent-method comparison (the live-relevant number).
        mean_br = ss_mean.get(incumbent)
        med_br = ss_med.get(incumbent)
        if mean_br is not None and med_br is not None:
            gain = mean_br - med_br  # positive => median better
            passes = gain >= MIN_CALIB_BRIER_GAIN
            print(f"    incumbent {incumbent}: mean={mean_br:.4f} "
                  f"median={med_br:.4f}  (median gain {gain:+.4f}; "
                  f">= {MIN_CALIB_BRIER_GAIN} gate: {'YES' if passes else 'no'})")
            if incumbent == "A":
                print("    note: method A ignores the projected center — a "
                      "mean/median swap is a no-op here by construction.")
        else:
            print(f"    incumbent {incumbent} Brier unavailable on this split.")

        # Did the GATE's chosen method change under median?
        if sel_mean.get("method") != sel_med.get("method"):
            print(f"    gate pick: mean-center={sel_mean.get('method')} -> "
                  f"median-center={sel_med.get('method')} (selection would change)")
        else:
            print(f"    gate pick unchanged ({sel_mean.get('method')}) under "
                  f"either center.")

    print("\n  (Diagnostic only — nothing written. This adds a `center` param to "
          "the offline harness; the live prediction path is untouched until we "
          "wire it into props.py.)")


# ── ROI diagnostic: profitability lens on calibration-method selection ──
# Method selection is Brier-only; a method can narrowly FAIL the Brier gate yet
# lift betting ROI ("throwing away a beneficial result"). --roi-diag replays each
# method (A/B/C/D/E) through the LIVE edge+EV recommendation gate at best-of-book
# prices and reports realized flat-1u ROI ALONGSIDE Brier, so both numbers are
# visible before a method is chosen. Read-only; informs, never automates. NO WRITE.
def _roi_sim_method(rows, prob_of, threshold):
    """Replay the live edge+EV recommendation gate for ONE calibration method over
    the priced rows and tally flat-1-unit ROI.

    Mirrors props.analyze_player_props_value's decision exactly: the edge is
    measured against the de-vigged consensus (``mkt_over``), the model backs the
    higher-edge side, and the bet is taken only when ``_prop_is_value(edge,
    threshold, expected_roi)`` holds — the edge clears ``threshold`` AND the bet is
    +EV at the price. Those two legs make this the RECOMMENDATION gate, not just
    "back any +edge side" like _cc_stratum_table. With de-vigged consensus
    ``under_implied == 1 - mkt_over``, so picking the higher-edge side reduces to
    ``sign(p - mkt_over)``. Payoff is the codebase-universal flat unit
    (win = decimal-1, loss = -1). Unpriced rows are skipped. ``prob_of(row)`` is the
    method's P(over) for that row.

    Returns {n_bets, pnl, roi, hit, avg_edge} (roi/hit/avg_edge None at n_bets=0)."""
    from pricing_common import _expected_roi, _prop_is_value
    pnl = won = sum_edge = 0.0
    n_bets = 0
    for r in rows:
        if (r["mkt_over"] is None or r["over_dec"] is None
                or r["under_dec"] is None):
            continue
        p = prob_of(r)
        over_edge = p - r["mkt_over"]
        if over_edge > 0.0:                       # back the OVER (higher-edge side)
            side_prob, price, dec, edge, over = (
                p, r["over_price"], r["over_dec"], over_edge, True)
        else:                                     # back the UNDER (ties -> under)
            side_prob, price, dec, edge, over = (
                1.0 - p, r["under_price"], r["under_dec"], -over_edge, False)
        expected_roi = _expected_roi(side_prob, price)
        if not _prop_is_value(edge, threshold, expected_roi):
            continue
        win = (r["o"] == 1) if over else (r["o"] == 0)
        pnl += (dec - 1.0) if win else -1.0
        won += 1.0 if win else 0.0
        sum_edge += edge
        n_bets += 1
    return {
        "n_bets": n_bets, "pnl": pnl,
        "roi": (pnl / n_bets) if n_bets else None,
        "hit": (won / n_bets) if n_bets else None,
        "avg_edge": (sum_edge / n_bets) if n_bets else None,
    }


def _roi_build_rows(enriched, params, sport_key, prop_key,
                    team_defense=None, league_avg_def=None,
                    defense_by_season=None):
    """Per-prop rows for the ROI sim: the leakage-safe as-of point projection +
    method-A empirical over-rate (blc.project_and_empirical), BOTH american book
    prices, and the de-vigged consensus ``mkt_over`` + decimal payouts. A prop-
    generic clone of _cc_load_scored_rows' row loop that also keeps the raw ``obs``
    (method D reads it) and neither drops pushes nor sorts (the caller does that per
    prop). Unpriced rows are kept with ``mkt_over=None`` so price coverage is
    honest; the sim skips them."""
    import book_line_calibration as blc
    from odds_client import (american_to_implied_prob, devig_two_way,
                             american_to_decimal)
    rows = []
    for obs in enriched:
        if obs.get("prop_key") != prop_key:
            continue
        projected, emp = blc.project_and_empirical(
            obs, params, sport_key, team_defense, league_avg_def,
            defense_by_season=defense_by_season)
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
            "obs": obs, "game_date": obs["game_date"], "line": obs["line"],
            "actual": obs["actual"], "projected": projected,
            "empirical_over": max(0.0, min(1.0, emp)),
            "over_price": op, "under_price": up,
            "mkt_over": mkt_over, "over_dec": over_dec, "under_dec": under_dec,
        })
    return rows


def diagnose_roi(sport, store_label="", threshold_pct=5.0, xstats_strength=0.0):
    """Profitability lens (NO WRITE): for each calibrated prop, replay methods
    A/B/C/D/E through the LIVE edge+EV recommendation gate at BEST-OF-BOOK consensus
    prices and report realized flat-1u ROI ALONGSIDE holdout Brier, so a method that
    narrowly FAILS the Brier gate but lifts ROI becomes visible.

    Same population + chronological 50/50 split the Brier gate uses (params fit on
    TRAIN; P(over) + betting simulated on the TEST half), so ROI is comparable to
    Brier method-for-method. Prices come from the harvested real lines — the odds
    warehouse never stores DraftKings, so this is best-of-book / de-vigged
    consensus: OPTIMISTIC vs the DK price the user actually bets and NOT DK-specific;
    the price is common across methods per obs, so the RELATIVE ranking is sound.
    INFORMS the method choice, never automates it. OFFLINE + FREE; writes nothing."""
    import book_line_calibration as blc
    from props import PROP_NEGBIN_ELIGIBLE, PROP_XSTATS_KIND

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key) or {}
    props_to_check = sorted(existing.keys())
    if not props_to_check:
        print(f"No calibrated props in calibration/{sport_key}.json; run refit first.")
        return

    print(f"\n=== ROI diagnostic (profitability lens): {sport_key} ===")
    print("  ⚠ ROI is priced at BEST-OF-BOOK / de-vigged consensus from the")
    print("    harvested real lines (the warehouse never stores DraftKings). This is")
    print("    OPTIMISTIC vs the DK price you actually bet and is NOT a DK claim; the")
    print("    price is common across methods per obs, so the RELATIVE method")
    print("    ranking holds but the ABSOLUTE ROI does not transfer to DK.")
    print("  ⚠ Method params (B/C residuals, E negbin) are fit on the TRAIN half")
    print("    only; the DEPLOYED calibration fits on ALL usable obs, so these")
    print("    out-of-sample numbers won't equal shipped params (the honest basis).")
    print("  ⚠ A/B/C/E score PLAIN projections (no xBA unless --roi-xstats-strength")
    print("    >0); D applies no park/weather/matchup multipliers. No gate; no write.")

    book_lines, n_primary, n_pred = blc.harvest_real_line_book_lines(
        sport_key, props_to_check, store_label)
    print(f"  {len(book_lines):,} real book lines "
          f"({n_primary:,} store + {n_pred:,} prediction log)")
    if not book_lines:
        print("  No real book lines (store or prediction log); nothing to diagnose.")
        return
    enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("  No observations joined to actuals; nothing to diagnose.")
        return

    # Weight-side opp-defense lookup only if some prop's variant uses it (mirror
    # diagnose_negbin so the projection basis matches the shipped fit). PER SEASON
    # (leakage guard): each obs re-weights against its OWN season.
    defense_by_season = None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in props_to_check):
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

    # Leakage-safe as-of xBA index for method D, only if requested (mirror
    # diagnose_distributional). Default 0.0 keeps the diag free (no Statcast pull).
    xba_index = None
    if xstats_strength > 0:
        import savant_history as sh
        import backtest_props
        years = sorted({str(o["game_date"])[:4]
                        for o in enriched if o.get("game_date")})
        raw = []
        for y in years:
            try:
                raw.extend(sh.load_days(f"{y}-03-01", f"{y}-11-30"))
            except Exception:
                pass
        xba_index = backtest_props.build_batter_xba_index(raw) if raw else None
        if xba_index is None:
            print(f"  [warn] no raw Statcast cached for {years}; method D falls "
                  f"back to plain projections.")

    threshold = threshold_pct / 100.0
    for prop_key in props_to_check:
        cfg = existing[prop_key]
        incumbent = cfg.get("method")
        params = {
            "half_life": cfg.get("half_life"),
            "venue_strength": cfg.get("venue_strength", 0.0),
            "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
            "use_minutes": cfg.get("use_minutes", False),
        }
        rows = _roi_build_rows(enriched, params, sport_key, prop_key,
                               defense_by_season=defense_by_season)
        rows = [r for r in rows if r["actual"] != r["line"]]     # drop pushes
        if len(rows) < 40:
            print(f"\n  {prop_key}: only {len(rows)} usable obs (<40) — too thin.")
            continue
        rows.sort(key=lambda r: r["game_date"])
        split = len(rows) // 2
        train, test = rows[:split], rows[split:]

        # Pooled residual fit on TRAIN (methods B/C); NegBin fit on TRAIN (method E).
        resid = sorted(r["actual"] - r["projected"] for r in train)
        mu = sum(resid) / len(resid)
        var = sum((x - mu) ** 2 for x in resid) / len(resid)
        sigma = math.sqrt(var) if var > 0 else 1e-6
        nb = (blc._fit_negbin_real(train)
              if prop_key in PROP_NEGBIN_ELIGIBLE else None)

        # Per-obs A/B/C/D/E P(over) on TEST — same math as _score_abc_real, exposed
        # per row (clone of diagnose_distributional's r["m"] block, extended w/ D/E).
        for r in test:
            r["o"] = 1 if r["actual"] > r["line"] else 0
            corrected = r["projected"] + mu
            m = {
                "A": max(0.0, min(1.0, r["empirical_over"])),
                "B": blc._norm_cdf((corrected - r["line"]) / sigma),
                "C": 1.0 - blc._empirical_cdf(resid, r["line"] - corrected),
            }
            if prop_key == "batter_hits":
                p_d = blc.project_distributional(
                    r["obs"], params, sport_key,
                    xstats_strength=xstats_strength, xba_index=xba_index,
                    defense_by_season=defense_by_season)
                if p_d is not None:
                    m["D"] = p_d
            if nb is not None:
                ms, disp = nb
                mean = max(1e-9, ms * r["projected"])
                m["E"] = blc.negbin_at_least(int(r["line"]) + 1, mean, disp)
            r["m"] = m

        n_priced = sum(1 for r in test if r["mkt_over"] is not None)
        print(f"\n  {prop_key} (incumbent={incumbent}, n_usable={len(rows)}, "
              f"n_test={len(test)}, priced={n_priced}/{len(test)}):")
        ship_xstats = cfg.get("xstats_strength") or 0.0
        if ship_xstats > 0 and prop_key in PROP_XSTATS_KIND:
            print(f"    ⚠ shipped {prop_key} blends xBA (s={ship_xstats:.2f}); "
                  f"A/B/C/E here run on PLAIN projections — Brier is directional.")
        print("    {:<7}{:>7}{:>9}{:>8}{:>9}{:>8}{:>10}".format(
            "method", "n", "Brier", "n_bets", "ROI%", "hit%", "avg_edge"))
        method_order = [k for k in ("A", "B", "C", "D", "E")
                        if any(k in r["m"] for r in test)]
        for k in method_order:
            scored = [r for r in test if k in r["m"]]
            br = blc._brier([r["m"][k] for r in scored],
                            [r["o"] for r in scored])
            sim = _roi_sim_method(scored, lambda r, _k=k: r["m"][_k], threshold)
            tag = "  <- incumbent" if k == incumbent else ""
            br_s = f"{br:.4f}" if br is not None else "-"
            roi_s = f"{sim['roi'] * 100:+.1f}" if sim["roi"] is not None else "-"
            hit_s = f"{sim['hit'] * 100:.1f}" if sim["hit"] is not None else "-"
            edge_s = (f"{sim['avg_edge']:+.3f}"
                      if sim["avg_edge"] is not None else "-")
            print("    {:<7}{:>7}{:>9}{:>8}{:>9}{:>8}{:>10}{}".format(
                k, len(scored), br_s, sim["n_bets"], roi_s, hit_s, edge_s, tag))

    print("\n  (Diagnostic only — nothing written.)")


def _consensus_prop_stats(test, threshold):
    """Cross-method AGREEMENT lens on the held-out TEST rows (see diagnose_consensus).

    Each PRICED row already carries r["m"] (per-method P(over)), r["o"] (outcome),
    and r["mkt_over"] (de-vigged consensus). For each model method A/B/C/D/E present,
    the VALUE side is OVER iff P(over) > mkt_over; ``edge_agree`` is True when every
    method points to the SAME value side. The consensus prob is the mean of the
    present methods. Then — as the live edge+EV gate would actually bet (via
    _roi_sim_method at ``threshold``, backing the consensus side) — realized hit-rate
    + flat-1u ROI are split by all-agree vs split, and by A/B/C dispersion tercile.
    Returns a dict (None if no priced rows). ROI is de-vigged CONSENSUS (optimistic
    vs DK) so only the RELATIVE agree-vs-split comparison is trustworthy."""
    methods = [k for k in ("A", "B", "C", "D", "E")
               if any(k in r["m"] for r in test)]
    priced = [r for r in test if r.get("mkt_over") is not None]
    if not priced:
        return None
    for r in priced:
        present = [k for k in methods if k in r["m"]]
        r["_cons_p"] = sum(r["m"][k] for k in present) / len(present)
        abc = [r["m"][k] for k in ("A", "B", "C") if k in r["m"]]
        r["_abc_disp"] = (max(abc) - min(abc)) if len(abc) >= 2 else 0.0
        # value side per method (over iff P(over) > market); agree = all one side.
        r["_edge_agree"] = len({r["m"][k] > r["mkt_over"] for k in present}) == 1
        # absolute-direction agreement among A/B/C (p>0.5) — confirms A/B/C rarely
        # split on SIGN (only near the coin-flip line), unlike on magnitude.
        r["_abc_sign_agree"] = len(
            {r["m"][k] > 0.5 for k in ("A", "B", "C") if k in r["m"]}) == 1

    def _sim(rows):
        return _roi_sim_method(rows, lambda r: r["_cons_p"], threshold)

    out = {
        "methods": methods, "n_test": len(test), "n_priced": len(priced),
        "edge_agree_pct": 100.0 * sum(r["_edge_agree"] for r in priced) / len(priced),
        "abc_sign_agree_pct":
            100.0 * sum(r["_abc_sign_agree"] for r in priced) / len(priced),
        "all": _sim(priced),
        "agree": _sim([r for r in priced if r["_edge_agree"]]),
        "split": _sim([r for r in priced if not r["_edge_agree"]]),
    }
    srt = sorted(priced, key=lambda r: r["_abc_disp"])
    if len(srt) >= 6:
        t = len(srt) // 3
        out["disp_tight"] = _sim(srt[:t])
        out["disp_wide"] = _sim(srt[-t:])
    return out


def diagnose_consensus(sport, store_label="", xstats_strength=0.5, threshold_pct=5.0):
    """Cross-method AGREEMENT lens (NO WRITE): does agreement among the calibration
    methods (A/B/C/D/E) on the VALUE side predict a better bet? For each prop, on the
    held-out test half at the live edge+EV gate, report flat-1u ROI + hit-rate when
    ALL methods agree on the value side vs when they SPLIT, plus the A/B/C dispersion
    split. Evidence for whether a consensus/agreement bet-selection layer would add a
    real edge BEFORE building one. OFFLINE + FREE; writes nothing.

    Same population + chronological 50/50 split + prep as diagnose_roi (params fit on
    TRAIN; probs + betting on TEST). Prices are de-vigged CONSENSUS (optimistic vs DK,
    common across methods) so the RELATIVE agree-vs-split comparison is sound but the
    ABSOLUTE ROI is not a DK claim. Method D needs xstats_strength>0 (default 0.5)."""
    import book_line_calibration as blc
    from props import PROP_NEGBIN_ELIGIBLE

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key) or {}
    props_to_check = sorted(existing.keys())
    if not props_to_check:
        print(f"No calibrated props in calibration/{sport_key}.json; run refit first.")
        return

    print(f"\n=== Consensus/agreement diagnostic: {sport_key} ===")
    print("  Q: when the methods AGREE on the value side, is the bet better?")
    print("     (evidence for a consensus/agreement bet-selection layer)")
    print("  ⚠ ROI at de-vigged CONSENSUS prices (optimistic vs DK; RELATIVE only);")
    print("    method params fit on TRAIN only (out-of-sample, not shipped params).")

    book_lines, n_primary, n_pred = blc.harvest_real_line_book_lines(
        sport_key, props_to_check, store_label)
    print(f"  {len(book_lines):,} real book lines "
          f"({n_primary:,} store + {n_pred:,} prediction log)")
    if not book_lines:
        print("  No real book lines; nothing to diagnose.")
        return
    enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("  No observations joined to actuals; nothing to diagnose.")
        return

    # PER SEASON (leakage guard): each obs re-weights against its OWN season.
    defense_by_season = None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in props_to_check):
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

    xba_index = None
    if xstats_strength > 0:
        import savant_history as sh
        import backtest_props
        years = sorted({str(o["game_date"])[:4]
                        for o in enriched if o.get("game_date")})
        raw = []
        for y in years:
            try:
                raw.extend(sh.load_days(f"{y}-03-01", f"{y}-11-30"))
            except Exception:
                pass
        xba_index = backtest_props.build_batter_xba_index(raw) if raw else None
        if xba_index is None:
            print(f"  [warn] no raw Statcast cached for {years}; method D excluded.")

    threshold = threshold_pct / 100.0

    def _fmt(sim):
        if not sim or sim.get("roi") is None:
            return f"n_bets={(sim or {}).get('n_bets', 0):>4}  ROI=    -    hit=   - "
        return (f"n_bets={sim['n_bets']:>4}  ROI={sim['roi'] * 100:+6.1f}%  "
                f"hit={sim['hit'] * 100:5.1f}%")

    for prop_key in props_to_check:
        cfg = existing[prop_key]
        params = {
            "half_life": cfg.get("half_life"),
            "venue_strength": cfg.get("venue_strength", 0.0),
            "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
            "use_minutes": cfg.get("use_minutes", False),
        }
        rows = _roi_build_rows(enriched, params, sport_key, prop_key,
                               defense_by_season=defense_by_season)
        rows = [r for r in rows if r["actual"] != r["line"]]
        if len(rows) < 40:
            print(f"\n  {prop_key}: only {len(rows)} usable obs (<40) — too thin.")
            continue
        rows.sort(key=lambda r: r["game_date"])
        split = len(rows) // 2
        train, test = rows[:split], rows[split:]

        resid = sorted(r["actual"] - r["projected"] for r in train)
        mu = sum(resid) / len(resid)
        var = sum((x - mu) ** 2 for x in resid) / len(resid)
        sigma = math.sqrt(var) if var > 0 else 1e-6
        nb = (blc._fit_negbin_real(train)
              if prop_key in PROP_NEGBIN_ELIGIBLE else None)
        for r in test:
            r["o"] = 1 if r["actual"] > r["line"] else 0
            corrected = r["projected"] + mu
            m = {
                "A": max(0.0, min(1.0, r["empirical_over"])),
                "B": blc._norm_cdf((corrected - r["line"]) / sigma),
                "C": 1.0 - blc._empirical_cdf(resid, r["line"] - corrected),
            }
            if prop_key == "batter_hits" and xba_index is not None:
                p_d = blc.project_distributional(
                    r["obs"], params, sport_key,
                    xstats_strength=xstats_strength, xba_index=xba_index,
                    defense_by_season=defense_by_season)
                if p_d is not None:
                    m["D"] = p_d
            if nb is not None:
                ms, disp = nb
                mean = max(1e-9, ms * r["projected"])
                m["E"] = blc.negbin_at_least(int(r["line"]) + 1, mean, disp)
            r["m"] = m

        st = _consensus_prop_stats(test, threshold)
        if st is None:
            print(f"\n  {prop_key}: no priced test rows.")
            continue
        print(f"\n  {prop_key} (methods={'/'.join(st['methods'])}, "
              f"n_test={st['n_test']}, priced={st['n_priced']}):")
        print(f"    A/B/C agree on SIGN {st['abc_sign_agree_pct']:.0f}%   |   "
              f"all methods agree on VALUE SIDE {st['edge_agree_pct']:.0f}%")
        print(f"    bet ALL:     {_fmt(st['all'])}")
        print(f"    bet AGREE:   {_fmt(st['agree'])}")
        print(f"    bet SPLIT:   {_fmt(st['split'])}")
        if "disp_tight" in st:
            print(f"    A/B/C tight: {_fmt(st['disp_tight'])}")
            print(f"    A/B/C wide:  {_fmt(st['disp_wide'])}")

    print("\n  (Diagnostic only — nothing written. If AGREE ROI/hit clearly beats "
          "SPLIT, an agreement bet-filter has teeth.)")


def _roi_sim_gate(rows, ev_floor, edge_floor):
    """Slate ROI sim under a parameterized recommendation gate.

    Each row already carries ``r['p_ship']`` (the SHIPPED method's P(over)).
    Backs the higher-edge side (vs de-vigged ``mkt_over``) and takes the bet only
    when ``edge >= edge_floor AND expected_roi >= ev_floor`` — a generalization of
    _prop_is_value that lets us A/B the edge-floor gate against an EV-primary gate.
    Flat 1u payoff (win = dec-1, loss = -1); unpriced rows skipped.
    Returns {n_bets, pnl, roi, hit}."""
    from pricing_common import _expected_roi
    pnl = won = 0.0
    n_bets = 0
    for r in rows:
        if (r["mkt_over"] is None or r["over_dec"] is None
                or r["under_dec"] is None):
            continue
        p = r["p_ship"]
        over_edge = p - r["mkt_over"]
        if over_edge > 0.0:
            side_prob, price, dec, edge, over = (
                p, r["over_price"], r["over_dec"], over_edge, True)
        else:
            side_prob, price, dec, edge, over = (
                1.0 - p, r["under_price"], r["under_dec"], -over_edge, False)
        er = _expected_roi(side_prob, price)
        if er is None or edge < edge_floor or er < ev_floor:
            continue
        win = (r["o"] == 1) if over else (r["o"] == 0)
        pnl += (dec - 1.0) if win else -1.0
        won += 1.0 if win else 0.0
        n_bets += 1
    return {"n_bets": n_bets, "pnl": pnl,
            "roi": (pnl / n_bets) if n_bets else None,
            "hit": (won / n_bets) if n_bets else None}


def diagnose_gate(sport, store_label=""):
    """Value-GATE lens (NO WRITE): replay each prop's SHIPPED method through a set
    of recommendation GATES over the real-line holdout and report aggregate flat-1u
    ROI + volume per gate, so the current edge-floor gate can be A/B'd against
    EV-primary gates (the "what makes a bet a suggestion?" question).

    Same population + chronological 50/50 split + shipped-method params as
    diagnose_roi (params fit on TRAIN; gate simulated on TEST). Consensus-priced
    (OPTIMISTIC vs DK) but the price is common across gates, so the RELATIVE gate
    ranking is sound. batter_hits' shipped xBA blend is approximated by the plain
    projection here (consistent across gates). INFORMS the gate choice; writes
    nothing."""
    import book_line_calibration as blc
    from props import PROP_NEGBIN_ELIGIBLE

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key) or {}
    props_to_check = sorted(existing.keys())
    if not props_to_check:
        print(f"No calibrated props in calibration/{sport_key}.json; run refit first.")
        return

    print(f"\n=== VALUE-GATE lens (A/B recommendation gates): {sport_key} ===")
    print("  ⚠ Consensus-priced (optimistic vs DK) + shipped-method params on a")
    print("    chronological TEST half; the price is common across gates so the")
    print("    RELATIVE gate ranking is sound. batter_hits xBA approx by plain proj.")

    book_lines, _n_store, _n_pred = blc.harvest_real_line_book_lines(
        sport_key, props_to_check, store_label)
    if not book_lines:
        print("  No real book lines; nothing to diagnose.")
        return
    enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("  No observations joined to actuals; nothing to diagnose.")
        return

    # PER SEASON (leakage guard): each obs re-weights against its OWN season.
    defense_by_season = None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in props_to_check):
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

    slate = []            # all props' TEST rows carrying r['p_ship']
    per_prop_rows = {}     # prop_key -> its TEST rows (for the per-prop breakdown)
    for prop_key in props_to_check:
        cfg = existing[prop_key]
        incumbent = cfg.get("method")
        params = {
            "half_life": cfg.get("half_life"),
            "venue_strength": cfg.get("venue_strength", 0.0),
            "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
            "use_minutes": cfg.get("use_minutes", False),
        }
        rows = _roi_build_rows(enriched, params, sport_key, prop_key,
                               defense_by_season=defense_by_season)
        rows = [r for r in rows if r["actual"] != r["line"]]
        if len(rows) < 40:
            continue
        rows.sort(key=lambda r: r["game_date"])
        split = len(rows) // 2
        train, test = rows[:split], rows[split:]
        resid = sorted(r["actual"] - r["projected"] for r in train)
        mu = sum(resid) / len(resid)
        var = sum((x - mu) ** 2 for x in resid) / len(resid)
        sigma = math.sqrt(var) if var > 0 else 1e-6
        nb = (blc._fit_negbin_real(train)
              if prop_key in PROP_NEGBIN_ELIGIBLE else None)

        kept = []
        for r in test:
            r["o"] = 1 if r["actual"] > r["line"] else 0
            corrected = r["projected"] + mu
            if incumbent == "A":
                p = max(0.0, min(1.0, r["empirical_over"]))
            elif incumbent == "B":
                p = blc._norm_cdf((corrected - r["line"]) / sigma)
            elif incumbent == "C":
                p = 1.0 - blc._empirical_cdf(resid, r["line"] - corrected)
            elif incumbent == "E" and nb is not None:
                ms, disp = nb
                p = blc.negbin_at_least(int(r["line"]) + 1,
                                        max(1e-9, ms * r["projected"]), disp)
            else:
                continue     # D/other shipped methods not simulated here
            if r["mkt_over"] is None:
                continue
            r["p_ship"] = p
            kept.append(r)
        if kept:
            per_prop_rows[prop_key] = kept
            slate.extend(kept)

    if not slate:
        print("  No priced shipped-method test rows; nothing to compare.")
        return

    GATES = [
        ("current  edge>=5% & EV>0", 1e-9, 0.05),
        ("flat     edge>=3% & EV>0", 1e-9, 0.03),
        ("ROI-1    EV>=2% & edge>=1%", 0.02, 0.01),
        ("ROI-2    EV>=3% & edge>=1%", 0.03, 0.01),
        ("ROI-3    EV>=4% & edge>=1%", 0.04, 0.01),
        ("ROI-4    EV>=5% & edge>=1%", 0.05, 0.01),
    ]
    print(f"\n  SLATE-WIDE (all shipped methods, {len(slate)} priced test rows):")
    print("    {:<28}{:>8}{:>9}{:>8}{:>10}".format(
        "gate", "n_bets", "ROI%", "hit%", "P/L(u)"))
    for label, ev_floor, edge_floor in GATES:
        s = _roi_sim_gate(slate, ev_floor, edge_floor)
        roi_s = f"{s['roi'] * 100:+.1f}" if s["roi"] is not None else "-"
        hit_s = f"{s['hit'] * 100:.1f}" if s["hit"] is not None else "-"
        print("    {:<28}{:>8}{:>9}{:>8}{:>+10.2f}".format(
            label, s["n_bets"], roi_s, hit_s, s["pnl"]))

    print("\n  PER-PROP (n_bets / ROI% under each gate):")
    hdr = "    {:<22}" + "{:>15}" * len(GATES)
    print(hdr.format("prop", *[g[0].split()[0] for g in GATES]))
    for prop_key in sorted(per_prop_rows):
        cells = []
        for _label, ev_floor, edge_floor in GATES:
            s = _roi_sim_gate(per_prop_rows[prop_key], ev_floor, edge_floor)
            roi_s = f"{s['roi'] * 100:+.0f}" if s["roi"] is not None else "-"
            cells.append(f"{s['n_bets']}/{roi_s}")
        print(hdr.format(prop_key, *cells))
    print("\n  (Diagnostic only — nothing written.)")


# ── §2.6 candidate-feature evaluation harness (NO WRITE) ──
# Generalizes the confirmation-gate philosophy across a curated FEATURE set
# (prop_features.FEATURE_REGISTRY): for each calibrated prop and each candidate
# feature that applies, score the prop's methods on the SAME real-line holdout
# the ship path uses, at each feature strength, and report whether turning the
# feature on clears the gate. Consensus ROI of the gate-selected method is a
# co-signal beside Brier. Reuses blc.select_method_at_real_lines verbatim; the
# feature is injected via params['features'] and threaded into the projection by
# prop_features.strengths_from_params, so strength 0 == production bit-for-bit.
def _roi_by_method(enriched, params, sport_key, prop_key, elig,
                   team_defense, league_avg_def, threshold,
                   defense_by_season=None):
    """Consensus-priced ROI per method (A/B/C/E) for one prop under ``params``.

    Mirrors diagnose_roi's fit + per-row-P(over) block (chronological 50/50
    split; residual + NegBin fit on TRAIN; P(over) on TEST) but returns
    {method: roi_sim_dict} so the caller can pull the ROI of whatever method the
    Brier gate selects. De-vigged best-of-book consensus (NOT DK); relative
    ranking only. Method D is omitted (needs the Statcast xBA index; this diag
    runs plain + free). Returns {} when too thin to price (<40 usable)."""
    import book_line_calibration as blc
    rows = _roi_build_rows(enriched, params, sport_key, prop_key,
                           team_defense, league_avg_def,
                           defense_by_season=defense_by_season)
    rows = [r for r in rows if r["actual"] != r["line"]]     # drop pushes
    if len(rows) < 40:
        return {}
    rows.sort(key=lambda r: r["game_date"])
    split = len(rows) // 2
    train, test = rows[:split], rows[split:]
    resid = sorted(r["actual"] - r["projected"] for r in train)
    mu = sum(resid) / len(resid)
    var = sum((x - mu) ** 2 for x in resid) / len(resid)
    sigma = math.sqrt(var) if var > 0 else 1e-6
    nb = blc._fit_negbin_real(train) if elig else None
    for r in test:
        r["o"] = 1 if r["actual"] > r["line"] else 0
        corrected = r["projected"] + mu
        m = {
            "A": max(0.0, min(1.0, r["empirical_over"])),
            "B": blc._norm_cdf((corrected - r["line"]) / sigma),
            "C": 1.0 - blc._empirical_cdf(resid, r["line"] - corrected),
        }
        if nb is not None:
            ms, disp = nb
            mean = max(1e-9, ms * r["projected"])
            m["E"] = blc.negbin_at_least(int(r["line"]) + 1, mean, disp)
        r["m"] = m
    out = {}
    for k in ("A", "B", "C", "E"):
        scored = [r for r in test if k in r["m"]]
        if scored:
            out[k] = _roi_sim_method(
                scored, lambda r, _k=k: r["m"][_k], threshold)
    return out


def diagnose_features(sport, feature=None, prop_filter=None, store_label="",
                      strengths_override=None):
    """Candidate-feature evaluation harness (roadmap §2.6, NO WRITE).

    For each calibrated prop and each registered candidate feature that applies
    to it, score the prop's calibration methods on the SAME real-line
    chronological holdout the ship path uses, once per feature STRENGTH, and
    report whether turning the feature on clears the confirmation gate (the
    incumbent's single-split Brier improves by >= MIN_CALIB_BRIER_GAIN, and does
    the 2-fold-gated method selection change). Consensus-priced ROI of the
    gate-selected method is printed ALONGSIDE Brier as a co-signal — Brier
    decides; true DK CLV is blocked on data accrual.

    Reuses blc.build_real_line_obs + blc.select_method_at_real_lines verbatim
    (roi_tiebreak off) so this shares ONE scoring/gate impl with the ship path.
    The feature is injected via params['features'] = {name: strength}; strength 0
    reproduces production bit-for-bit (a built-in self-check). OFFLINE + FREE.

    ``feature`` restricts to one registered feature; ``prop_filter`` to props."""
    import book_line_calibration as blc
    import prop_features
    from props import PROP_NEGBIN_ELIGIBLE

    espn_sport, espn_league, sport_key = SPORT_MAP[sport]
    existing = load_calibration(sport_key) or {}
    props_to_check = sorted(existing)
    if prop_filter:
        props_to_check = [pk for pk in props_to_check if pk in prop_filter]

    feats = [f for f in prop_features.FEATURE_REGISTRY
             if feature is None or f["name"] == feature]
    if feature is not None and not feats:
        print(f"Unknown feature '{feature}'. Registered: "
              f"{', '.join(f['name'] for f in prop_features.FEATURE_REGISTRY)}")
        return
    props_to_check = [pk for pk in props_to_check
                      if any(prop_features.feature_applies(f["name"], pk)
                             for f in feats)]
    if not props_to_check:
        print(f"No calibrated props in calibration/{sport_key}.json that a "
              f"registered feature applies to"
              + (f" (feature={feature})" if feature else "")
              + "; run refit first.")
        return

    print(f"\n=== Candidate-feature diagnostic (§2.6): {sport_key} ===")
    print(f"  Features: {', '.join(f['name'] for f in feats)}")
    print(f"  Props:    {', '.join(props_to_check)}")
    book_lines, n_primary, n_pred = blc.harvest_real_line_book_lines(
        sport_key, props_to_check, store_label)
    print(f"  {len(book_lines):,} real book lines "
          f"({n_primary:,} store + {n_pred:,} prediction log)")
    if not book_lines:
        print("  No real book lines (store or prediction log); nothing to diagnose.")
        return
    enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("  No observations joined to actuals; nothing to diagnose.")
        return

    # PER SEASON (leakage guard): each obs re-weights against its OWN season.
    defense_by_season = None
    if any((existing[pk].get("opp_defense_strength") or 0.0) > 0
           for pk in props_to_check):
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)

    threshold = 0.05   # edge threshold for the ROI sim (matches diagnose_roi)
    for prop_key in props_to_check:
        cfg = existing[prop_key]
        incumbent = cfg.get("method")
        elig = prop_key in PROP_NEGBIN_ELIGIBLE
        base_params = {
            "half_life": cfg.get("half_life"),
            "venue_strength": cfg.get("venue_strength", 0.0),
            "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
            "use_minutes": cfg.get("use_minutes", False),
        }
        for f in feats:
            if not prop_features.feature_applies(f["name"], prop_key):
                continue
            # A --feature-strengths override sweeps a finer grid (0.0 is always
            # the off baseline, prepended + de-duped) so a small optimum (e.g. 0.2)
            # isn't missed by the coarse registry default (0.0, 0.5, 1.0).
            strengths = (([0.0] + [s for s in sorted(set(strengths_override))
                                   if s > 0])
                         if strengths_override else list(f["strengths"]))
            off = strengths[0]
            sels, rois, rows_by_s = {}, {}, {}
            for s in strengths:
                params = dict(base_params, features={f["name"]: s})
                rows = blc.build_real_line_obs(
                    enriched, params, sport_key, prop_key,
                    xstats_strength=0.0, xba_index=None,
                    defense_by_season=defense_by_season)
                rows_by_s[s] = rows        # stash for the per-line-bucket pass below
                sels[s] = blc.select_method_at_real_lines(
                    rows, negbin_eligible=elig, roi_tiebreak=False)
                rois[s] = _roi_by_method(enriched, params, sport_key, prop_key,
                                         elig, None, None, threshold,
                                         defense_by_season=defense_by_season)
            # n_usable is strength-invariant (pushes = actual==raw line), so all
            # strengths return None together when too thin.
            if sels[off] is None:
                print(f"\n  {prop_key} [{f['name']}]: too few usable real-line "
                      f"obs (need >=20) — can't score.")
                continue

            ss_off = sels[off].get("single_split", {})
            methods = [m for m in ("A", "B", "C", "D", "E")
                       if any(m in (sels[s].get("single_split", {}) or {})
                              for s in strengths)]
            print(f"\n  {prop_key} [{f['name']}] (incumbent={incumbent}):")
            print("    holdout Brier by method:")
            print(f"    {'strength':>8}  " + "  ".join(f"{m:>8}" for m in methods))
            for s in strengths:
                ss = sels[s].get("single_split", {})
                cells = "  ".join(
                    (f"{ss[m]:8.4f}" if m in ss else f"{'—':>8}") for m in methods)
                print(f"    {s:>8.2f}  {cells}")

            inc_off = ss_off.get(incumbent)
            for s in strengths:
                if s == off:
                    continue
                inc_s = sels[s].get("single_split", {}).get(incumbent)
                if inc_off is not None and inc_s is not None:
                    gain = inc_off - inc_s
                    passes = gain >= MIN_CALIB_BRIER_GAIN
                    print(f"    incumbent {incumbent} @ strength {s:.2f}: "
                          f"off={inc_off:.4f} on={inc_s:.4f} (gain {gain:+.4f}; "
                          f">= {MIN_CALIB_BRIER_GAIN} gate: "
                          f"{'YES' if passes else 'no'})")
                if sels[s].get("method") != sels[off].get("method"):
                    print(f"      gate pick: off={sels[off].get('method')} -> "
                          f"strength{s:.2f}={sels[s].get('method')} "
                          f"(selection would change)")

            print("    consensus-ROI of the gate-selected method "
                  "(de-vigged best-of-book; relative only):")
            print(f"    {'strength':>8}{'method':>8}{'n_bets':>8}{'ROI%':>9}")
            for s in strengths:
                mth = sels[s].get("method")
                sim = rois[s].get(mth)
                if sim and sim.get("roi") is not None:
                    print(f"    {s:>8.2f}{mth:>8}{sim['n_bets']:>8}"
                          f"{sim['roi'] * 100:>8.1f}%")
                else:
                    print(f"    {s:>8.2f}{mth:>8}{'—':>8}{'—':>9}")

            # ── Per-line-bucket pass (DIAGNOSTIC-ONLY; every pooled table above is
            #    unchanged). Calibration METHODS get a per-line-bucket selector but
            #    FEATURES are gated only on the POOLED holdout. ~87.5% of batter_hits
            #    obs sit at line <= 0.5 where P(>=1 hit) is saturated (~binary), so a
            #    feature that only helps at >= 1.5 can't move the pooled number. Here
            #    we re-score the incumbent method WITHIN each line bucket to surface
            #    signal the pooled gate hides. Bucket membership by raw book line is
            #    strength-invariant (a feature moves projection/p_dist, never the
            #    line), so we just partition each strength's already-built rows and
            #    score each bucket with the SAME select_method_at_real_lines the ship
            #    path uses. Print-only: cannot change any shipped decision.
            parts = {s: _partition_rows_by_bucket(rows_by_s[s]) for s in strengths}
            if parts.get(off):
                print("    per-line-bucket gain (diagnostic; pooled gate above may "
                      "saturate at line<=0.5):")
            for bkey, off_bucket in parts.get(off, {}).items():
                if len(off_bucket) < MIN_BUCKET_OBS:
                    print(f"      bucket {bkey}: n={len(off_bucket)} < "
                          f"{MIN_BUCKET_OBS} floor - skipped (thin).")
                    continue
                sel_b = {s: blc.select_method_at_real_lines(
                             parts[s].get(bkey, []), negbin_eligible=elig,
                             roi_tiebreak=False)
                         for s in strengths}
                if sel_b[off] is None:
                    print(f"      bucket {bkey}: n={len(off_bucket)} but "
                          f"usable<20 - can't score.")
                    continue
                # Faithful per-bucket incumbent: if the shipped cfg has a line_methods
                # entry for this bucket use THAT bucket's method, else the pooled
                # incumbent (batter_hits line_methods is null today -> pooled method).
                b_inc = _rc_method_for_line(
                    off_bucket[0]["line"], cfg.get("line_methods"), incumbent)
                inc_off_b = (sel_b[off].get("single_split", {}) or {}).get(b_inc)
                n_use = sel_b[off].get("n_obs")
                print(f"      bucket {bkey} (incumbent={b_inc}, usable n={n_use}):")
                print(f"        {'strength':>8}{'off':>9}{'on':>9}{'gain':>9}  gate")
                for s in strengths:
                    if s == off:
                        continue
                    inc_s_b = ((sel_b[s].get("single_split", {}) or {}).get(b_inc)
                               if sel_b[s] else None)
                    if inc_off_b is None or inc_s_b is None:
                        print(f"        {s:>8.2f}{'-':>9}{'-':>9}{'-':>9}  "
                              f"n/a (incumbent {b_inc} unscored in bucket)")
                        continue
                    gain_b = inc_off_b - inc_s_b
                    passes_b = gain_b >= MIN_CALIB_BRIER_GAIN
                    print(f"        {s:>8.2f}{inc_off_b:>9.4f}{inc_s_b:>9.4f}"
                          f"{gain_b:>+9.4f}  {'YES' if passes_b else 'no'}")
                    if (sel_b[s] and sel_b[off]
                            and sel_b[s].get("method") != sel_b[off].get("method")):
                        print(f"          gate pick: off={sel_b[off].get('method')} "
                              f"-> strength{s:.2f}={sel_b[s].get('method')} "
                              f"(selection would change)")

    print(f"\n  (Diagnostic only — nothing written. Strengths are injected into "
          f"the offline projection via prop_features; strength 0 == production. A "
          f"feature auto-adopts only if a prop clears the same 2-fold + "
          f"{MIN_CALIB_BRIER_GAIN} gate on the next --refit.)")


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

    # Weight-side opp-defense lookup only if the shipped variant uses it. PER SEASON
    # (leakage guard): each obs re-weights against its OWN season's pooled defense.
    defense_by_season = None
    if (cfg.get("opp_defense_strength") or 0.0) > 0:
        defense_by_season = _defense_by_season(espn_sport, espn_league, enriched)
    params = {
        "half_life": cfg.get("half_life"),
        "venue_strength": cfg.get("venue_strength", 0.0),
        "opp_defense_strength": cfg.get("opp_defense_strength", 0.0),
        "use_minutes": False,
    }

    # Method-D reconstruction: when a batter_hits line-bucket ships as D, build the
    # as-of xBA index once + stamp each obs's distributional prob (p_dist) so
    # _rc_run_bucket can recalibrate D like A/C. None (off) when no D bucket ships.
    _lm = cfg.get("line_methods") or []
    _d_buckets = [b for b in _lm if b.get("method") == "D"]
    _default_m = cfg.get("method", "A")
    _uses_d = bool(_d_buckets) or _default_m == "D"
    xba_index = _diag_build_xba_index(enriched) if _uses_d else None
    d_xstats = (_d_buckets[0].get("xstats_strength", 0.75) if _d_buckets
                else cfg.get("xstats_strength", 0.75))

    rows = []
    for obs in enriched:
        projected, emp = blc.project_and_empirical(
            obs, params, sport_key, defense_by_season=defense_by_season)
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
        rowd = {
            "game_date": obs["game_date"], "line": obs["line"],
            "actual": obs["actual"], "projected": projected,
            "empirical_over": max(0.0, min(1.0, emp)),
            "mkt_over": mkt_over, "over_dec": over_dec, "under_dec": under_dec,
        }
        if xba_index is not None and _rc_method_for_line(
                obs["line"], _lm, _default_m) == "D":
            try:
                rowd["p_dist"] = blc.project_distributional(
                    obs, params, sport_key, xba_index=xba_index,
                    xstats_strength=d_xstats, defense_by_season=defense_by_season)
            except Exception:
                rowd["p_dist"] = None
        rows.append(rowd)

    rows = [r for r in rows if r["actual"] != r["line"]]   # drop pushes
    rows.sort(key=lambda r: r["game_date"])
    for r in rows:
        r["o"] = 1 if r["actual"] > r["line"] else 0
    return sport_key, cfg, rows


def _diag_build_xba_index(enriched):
    """Leakage-safe as-of xBA index for reconstructing method-D probs in
    --recalibrate (mirrors the --real-lines build). Fail-open -> None when no
    Statcast days are cached for the obs seasons (D rows then drop out)."""
    try:
        import savant_history as sh
        import backtest_props
    except Exception:
        return None
    years = sorted({str(o["game_date"])[:4] for o in enriched
                    if isinstance(o, dict) and o.get("game_date")})
    raw = []
    for y in years:
        try:
            raw.extend(sh.load_days(f"{y}-03-01", f"{y}-11-30"))
        except Exception:
            pass
    if not raw:
        print("  [xstats] no Statcast days cached — method-D buckets can't be "
              "reconstructed; they show as skipped.")
        return None
    try:
        idx = backtest_props.build_batter_xba_index(raw)
        print(f"  [xstats] built leakage-safe xBA index from {len(raw):,} pitches "
              f"({', '.join(years)}) to reconstruct method-D probs.")
        return idx
    except Exception:
        return None


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


_RC_ISOTONIC_MIN_TRAIN = 1000   # min train rows before isotonic is a WINNER
#                                 candidate; below it a non-parametric step map is
#                                 noise-fit (sklearn rule of thumb ~1000) -> Platt.


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
    elif method == "D":
        drows = [r for r in brows if r.get("p_dist") is not None]
        if len(drows) < 120:
            print(f"     too thin (n={len(drows)} rows with a reconstructed "
                  f"distributional prob < 120); skipped — D needs as-of AB + a "
                  f"cached xBA index (run the Statcast bulk backfill if 0).")
            return
        split = len(drows) // 2
        train, test = drows[:split], drows[split:]
        for r in train + test:
            r["p_raw"] = r["p_dist"]
        print(f"     split: train={len(train)}  test={len(test)} (2-way, method-D "
              f"distributional prob w/ as-of xBA; {n - len(drows)} row(s) dropped "
              f"for missing AB/xBA)")
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
    # R4b(i): isotonic is a WINNER candidate only with enough train rows — below
    # _RC_ISOTONIC_MIN_TRAIN a step map is noise-fit, so fall back to Platt/raw.
    cands = [("raw", m_raw, _raw)]
    if len(train) >= _RC_ISOTONIC_MIN_TRAIN:
        cands.append(("isotonic", m_is, _is))
    else:
        print(f"     isotonic held out as a winner (train={len(train)} < "
              f"{_RC_ISOTONIC_MIN_TRAIN}: too thin to trust a step map)")
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

    # R4b(ii) market-BSS gate: beating RAW makes a map calibration-honest, but it is
    # only SELECTION-worthy if it ALSO beats the de-vigged MARKET's Brier on the
    # priced rows (BSS > 0). A map that beats raw but not the market improves our
    # numbers without adding anything the market doesn't already price -- it must
    # NOT be allowed to change bet selection (the no-edge trap).
    priced = [r for r in test if r.get("mkt_over") is not None]
    if priced:
        mkt_brier = sum((r["mkt_over"] - r["o"]) ** 2 for r in priced) / len(priced)
        win_brier = sum((win_get(r) - r["o"]) ** 2 for r in priced) / len(priced)
        bss = (1.0 - win_brier / mkt_brier) if mkt_brier > 0 else None
        verdict = ("SELECTION-worthy (beats the market)" if win_brier < mkt_brier
                   else "calibration-only -- does NOT beat the market; must not "
                        "alter bet selection")
        print("     vs de-vigged MARKET (priced n={}): map brier {:.4f} vs market "
              "{:.4f}{} -> {}".format(
                  len(priced), win_brier, mkt_brier,
                  ", BSS {:+.4f}".format(bss) if bss is not None else "", verdict))

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


def _print_calibration_diff(sport):
    """Human-readable candidate-vs-live diff for --diff."""
    _, _, sport_key = SPORT_MAP[sport]
    d = diff_calibration(sport_key)
    if not d.get("has_candidate"):
        print(f"No staged candidate for {sport_key} "
              f"({candidate_path(sport_key)} does not exist).")
        return
    print(f"\n=== candidate vs live: {sport_key} ===")
    print(f"  candidate fit: {d.get('candidate_ts')}")
    print(f"  live fit:      {d.get('live_ts') or '(no live file)'}")
    props = d["props"]
    if props["added"]:
        print(f"  + props added ({len(props['added'])}): "
              f"{', '.join(props['added'])}")
    if props["removed"]:
        print(f"  - props removed ({len(props['removed'])}): "
              f"{', '.join(props['removed'])}")
    method_changes = [c for c in props["changed"]
                      if c["live_method"] != c["candidate_method"]]
    nobs_changes = [c for c in props["changed"]
                    if c["live_method"] == c["candidate_method"]]
    if method_changes:
        print(f"  ~ METHOD changes ({len(method_changes)}) — review these:")
        for c in method_changes:
            print(f"      {c['prop']}: {c['live_method']} -> "
                  f"{c['candidate_method']} "
                  f"(n {c['live_nobs']} -> {c['candidate_nobs']})")
    if nobs_changes:
        print(f"  ~ re-fit, method unchanged ({len(nobs_changes)}): "
              f"{', '.join(c['prop'] for c in nobs_changes)}")
    blocks = d["blocks"]
    for label, keys in (("added", blocks["added"]),
                        ("removed", blocks["removed"]),
                        ("changed", blocks["changed"])):
        if keys:
            print(f"  block {label}: {', '.join(keys)}")
    if not (props["added"] or props["removed"] or props["changed"]
            or blocks["added"] or blocks["removed"] or blocks["changed"]):
        print("  (candidate is identical to live)")
    print(f"\n  Promote: python refit_calibration.py --sport {sport} --promote")
    print(f"  Discard: python refit_calibration.py --sport {sport} --discard")


def _report_staging(sport, staging, wrote=True):
    """Post-refit pointer telling the user where the fit landed."""
    _, _, sport_key = SPORT_MAP[sport]
    if not staging:
        print(f"\n⇢ Wrote LIVE calibration/{sport_key}.json directly (--live).")
        return
    if not (wrote and has_candidate(sport_key)):
        return  # dry-run or nothing written
    print(f"\n⇢ Staged to calibration/{sport_key}.candidate.json — the live "
          f"file the app serves is UNTOUCHED.")
    print(f"    Review:  python refit_calibration.py --sport {sport} --diff")
    print(f"    Promote: python refit_calibration.py --sport {sport} --promote")
    print(f"    Discard: python refit_calibration.py --sport {sport} --discard")


def main():
    p = argparse.ArgumentParser(description="Refit persistent calibration files")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), required=True)
    p.add_argument("--season", type=int, default=None,
                   help="Current season year (ESPN convention). Default: current.")
    p.add_argument("--seasons", default=None,
                   help="Comma-separated seasons to POOL into one fit "
                        "(e.g. 2024,2025,2026). Widens the player pool (union) "
                        "and triples the thin pitcher-prop sample. Overrides "
                        "--season for the synthetic base fit; each season is "
                        "still projected strictly within-season.")
    p.add_argument("--prior-season", type=int, default=None,
                   help="Prior season year for warmup. Recommended. Ignored if "
                        "it is already one of --seasons (already pooled).")
    p.add_argument("--focused-grid", action="store_true",
                   help="Use the FOCUSED ~37-variant sweep grid instead of the full "
                        "576 (full resolution on half_life×venue×shrink, dead axes "
                        "probed once). ~15× faster + lighter — recommended for "
                        "iterating on pooled multi-season refits; drops knob-"
                        "interaction cells, so use the full grid for a final run.")
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
    p.add_argument("--no-roi-tiebreak", action="store_true",
                   help="With --real-lines, DISABLE the ROI tiebreaker (methods "
                        "within the Brier noise band fall back to A / lowest "
                        "Brier). Default: ROI breaks ties. Use to A/B a run "
                        "against the pure-Brier selection.")
    p.add_argument("--min-override-obs", type=int,
                   default=MIN_REAL_LINE_OVERRIDE_OBS,
                   help="With --real-lines, the real-line obs floor below which a "
                        "prop keeps its incumbent method (anti-noise churn guard; "
                        f"default {MIN_REAL_LINE_OVERRIDE_OBS}). Lower it to adopt "
                        "a confirmed-better method on a prop just under the floor "
                        "(e.g. a losing-ROI incumbent backed by --roi-diag).")
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
    p.add_argument("--negbin-diag", action="store_true",
                   help="§2.2: score the Negative-Binomial count model (method E) "
                        "vs A/B/C on the real-line holdout for each eligible count "
                        "prop, and report whether E would clear the ship gate (no "
                        "write).")
    p.add_argument("--center-diag", action="store_true",
                   help="Mean-vs-median: re-score each calibrated prop's incumbent "
                        "method on the real-line holdout with a recency-weighted "
                        "MEDIAN center vs the production MEAN center, and report "
                        "whether median beats mean by >= the ship gate (no write).")
    p.add_argument("--center-prop", default=None,
                   help="Restrict --center-diag to these prop_key(s), "
                        "comma-separated (e.g. pitcher_strikeouts).")
    p.add_argument("--feature-diag", action="store_true",
                   help="§2.6 feature-eval harness: for each calibrated prop and "
                        "each registered candidate feature (prop_features, e.g. "
                        "rest/days-off), re-score the prop's methods on the "
                        "real-line holdout at each feature strength and report "
                        "whether it clears the ship gate, with consensus ROI "
                        "alongside Brier (no write).")
    p.add_argument("--feature", default=None,
                   help="Restrict --feature-diag to one registered feature "
                        "(e.g. rest).")
    p.add_argument("--feature-prop", default=None,
                   help="Restrict --feature-diag to these prop_key(s), "
                        "comma-separated (e.g. pitcher_outs,pitcher_strikeouts).")
    p.add_argument("--feature-strengths", default=None,
                   help="Override the --feature-diag strength grid with a finer "
                        "comma list (e.g. '0.1,0.2,0.3,0.4,0.5,0.75,1.0'); 0.0 is "
                        "always prepended as the off baseline. Finds a small "
                        "optimum the coarse registry default (0.0,0.5,1.0) misses.")
    p.add_argument("--roi-diag", action="store_true",
                   help="Profitability lens: for each calibrated prop, replay the "
                        "live edge+EV recommendation gate at BEST-OF-BOOK consensus "
                        "prices and report flat-1u ROI per method (A/B/C/D/E) "
                        "alongside holdout Brier, so a method that narrowly fails "
                        "the Brier gate but lifts ROI is visible (no write).")
    p.add_argument("--roi-threshold-pct", type=float, default=5.0,
                   help="Edge threshold (percent) the --roi-diag gate requires, "
                        "matching props.analyze_player_props_value (default 5.0).")
    p.add_argument("--gate-diag", action="store_true",
                   help="Value-GATE lens: replay each prop's SHIPPED method through "
                        "the current edge-floor gate vs EV-primary gates over the "
                        "real-line holdout, and report aggregate + per-prop ROI/"
                        "volume per gate (answers 'what should make a bet a "
                        "suggestion?'). No write.")
    p.add_argument("--roi-xstats-strength", type=float, default=0.0,
                   help="xBA blend weight for method D under --roi-diag. >0 builds "
                        "the leakage-safe as-of xBA index (needs cached raw "
                        "Statcast); 0 (default) scores D on plain projections and "
                        "keeps the diagnostic free.")
    p.add_argument("--consensus-diag", action="store_true",
                   help="Cross-method AGREEMENT lens: does agreement among methods "
                        "A/B/C/D/E on the value side predict a better bet? Reports "
                        "ROI/hit when methods AGREE vs SPLIT (+ A/B/C dispersion) "
                        "over the real-line holdout — evidence for a consensus bet-"
                        "selection layer. Uses --roi-threshold-pct. No write.")
    p.add_argument("--consensus-xstats-strength", type=float, default=0.5,
                   help="xBA blend weight for method D under --consensus-diag "
                        "(default 0.5 so D is in the mix; needs Statcast in SQL).")
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
    # ── candidate-file staging (default-safe calibration writes) ──
    # A refit writes to calibration/<sport>.candidate.json, NEVER the live file
    # the app serves — so an accidental/experimental run can't clobber a carefully
    # tuned live calibration. Review with --diff, then --promote to make it live
    # (the old live is archived first) or --discard to throw it away.
    p.add_argument("--live", action="store_true",
                   help="Write straight to the LIVE calibration file, skipping "
                        "candidate staging (advanced; the default stages a "
                        "candidate you review then --promote).")
    p.add_argument("--promote", action="store_true",
                   help="Promote the staged candidate to live (archives the "
                        "current live file first), then exit. No refit is run.")
    p.add_argument("--diff", action="store_true",
                   help="Show how the staged candidate differs from the live "
                        "calibration (methods, n_obs, added/removed props and "
                        "blocks), then exit.")
    p.add_argument("--discard", action="store_true",
                   help="Delete the staged candidate without promoting, then exit.")
    args = p.parse_args()

    # Candidate-file management (--promote/--diff/--discard) are pure local-file
    # operations: handle + exit before any SQL/backend setup so they always work.
    if args.promote or args.diff or args.discard:
        _, _, sport_key = SPORT_MAP[args.sport]
        if args.diff:
            _print_calibration_diff(args.sport)
        elif args.discard:
            removed = discard_candidate(sport_key)
            print(f"{'✓ Discarded' if removed else 'No'} staged candidate for "
                  f"{sport_key}.")
        else:  # --promote
            try:
                archived = promote_calibration(sport_key)
            except FileNotFoundError as e:
                p.error(str(e))
            print(f"✓ Promoted candidate → calibration/{sport_key}.json (live).")
            if archived:
                import os as _os
                print(f"  Previous live archived to {_os.path.relpath(archived)}")
        return

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
    else:
        # SQL-off hardening (WS1 Layer C): a refit against a signalled-but-off SQL
        # deployment would silently train/write nothing (degenerate warehouse
        # reads + local-disk writes wiped on restart). Abort loudly. --store-label
        # is an explicit local backfill read, so it is exempt from the read abort;
        # Layer A still guards any writes it makes (mark_predictions_refit).
        if (not args.store_label and db_store.require_sql()
                and not db_store.enabled()):
            p.error("SQL backend not reachable but a SQL deployment is configured "
                    "(SPORTSBOOK_REQUIRE_SQL or SQL_* secrets present); aborting so "
                    "the refit does not silently train/write nothing. Fix SQL_* "
                    "secrets or pass --store-label for an intentional local run.")

    if args.dist_diag:
        diagnose_distributional(args.sport, store_label=args.store_label,
                                xstats_strength=args.dist_xstats_strength,
                                seasons=([int(s.strip()) for s in
                                          args.seasons.split(",") if s.strip()]
                                         if args.seasons else None))
        return

    if args.negbin_diag:
        diagnose_negbin(args.sport, store_label=args.store_label)
        return

    if args.center_diag:
        prop_filter = ([p.strip() for p in args.center_prop.split(",")]
                       if args.center_prop else None)
        diagnose_center(args.sport, prop_filter=prop_filter,
                        store_label=args.store_label)
        return

    if args.feature_diag:
        prop_filter = ([p.strip() for p in args.feature_prop.split(",")]
                       if args.feature_prop else None)
        strengths_override = ([float(s) for s in args.feature_strengths.split(",")]
                              if args.feature_strengths else None)
        diagnose_features(args.sport, feature=args.feature,
                          prop_filter=prop_filter, store_label=args.store_label,
                          strengths_override=strengths_override)
        return

    if args.roi_diag:
        diagnose_roi(args.sport, store_label=args.store_label,
                     threshold_pct=args.roi_threshold_pct,
                     xstats_strength=args.roi_xstats_strength)
        return

    if args.consensus_diag:
        diagnose_consensus(args.sport, store_label=args.store_label,
                           xstats_strength=args.consensus_xstats_strength,
                           threshold_pct=args.roi_threshold_pct)
        return

    if args.gate_diag:
        diagnose_gate(args.sport, store_label=args.store_label)
        return

    if args.reliability:
        diagnose_conditional_calibration(args.sport, store_label=args.store_label,
                                         min_cell_n=args.min_cell_n)
        return

    if args.recalibrate:
        diagnose_recalibration(args.sport, store_label=args.store_label,
                               min_cell_n=args.min_cell_n)
        return

    # Default-safe: a refit stages a candidate; --live writes the live file.
    staging = not args.live
    set_candidate_mode(staging)
    if staging:
        _, _, _sk = SPORT_MAP[args.sport]
        _notice = existing_candidate_notice(_sk)
        if _notice:
            print(_notice)

    if args.real_lines:
        refit_sport_real_lines(args.sport, store_label=args.store_label,
                               warmup_games=args.warmup_games,
                               shrinkage_k_default=args.shrinkage_k,
                               xstats_strength=args.xstats_strength,
                               dry_run=args.dry_run,
                               roi_tiebreak=not args.no_roi_tiebreak,
                               min_override_obs=args.min_override_obs,
                               seasons=([int(s.strip()) for s in
                                         args.seasons.split(",") if s.strip()]
                                        if args.seasons else None))
        _report_staging(args.sport, staging, wrote=not args.dry_run)
        return

    players = [n.strip() for n in args.players.split(",")] if args.players else None
    props = [pk.strip() for pk in args.props.split(",")] if args.props else None
    seasons = ([int(s.strip()) for s in args.seasons.split(",") if s.strip()]
               if args.seasons else None)

    refit_sport(args.sport, season=args.season, prior_season=args.prior_season,
                players=players, props=props,
                games_per_player=args.games_per_player,
                warmup_games=args.warmup_games,
                shrinkage_k_default=args.shrinkage_k,
                mlb_max_batters=args.mlb_max_batters,
                mlb_max_pitchers=args.mlb_max_pitchers,
                nba_max_players=args.nba_max_players,
                nba_min_games=args.nba_min_games,
                seasons=seasons, focused_grid=args.focused_grid)
    _report_staging(args.sport, staging)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
