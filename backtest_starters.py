"""
Fit the MLB `starter_adjustment` weights from HISTORICAL outcomes — no odds,
no waiting for live data. Uses:

  * StatsAPI  → games, final scores, probable/starting pitchers  (outcomes)
  * Statcast  → leakage-safe as-of xwOBAcon per starter          (features)

Objective is MODEL ACCURACY vs reality (not the market), so it costs zero Odds
API credits:
  * moneyline weight → 1-D logistic fit of home_win ~ starter_edge
  * run_scale        → 1-D OLS fit of total_runs ~ combined starter excess

v1 fits the Phase-1 team-market weights (moneyline, run_scale). Props/bullpen
weight fitting (needs batter xwOBA-vs-hand + prop box outcomes) is a planned v2
on the same cached data.

Usage:
    python backtest_starters.py --season 2024 --fetch      # slow one-time pull
    python backtest_starters.py --season 2024              # fit (days cached)
    python backtest_starters.py --season 2024 --save       # fit + write weights
    # Expected-runs challenger plus park/workload/NB holdout validations:
    python backtest_starters.py --season 2024 --test-runs
    # Add a prior full season to the pre-2024 holdout fit:
    python backtest_starters.py --season 2023,2024 --test-runs
    # Predeclared candidates against an untouched final season:
    python backtest_starters.py --season 2023-2025 --test-final

Caveats (documented, not hidden):
  * Uses the *probable* starter from the schedule (≈ actual; late scratches
    are rare but unmodeled).
  * As-of pitcher quality uses xwOBAcon (contact), a slightly different measure
    than the live season xERA index; both are Savant x-stats and correlated.
  * Historical park factors use the home-team abbreviation as a venue proxy;
    rare neutral-site games are not identified in the cached schedule shape.
  * The isolated logistic coefficient can overlap with signal team form already
    captures, so treat the fitted moneyline weight as a mild upper bound and
    keep it bounded.
"""

import argparse
from collections import defaultdict
from datetime import date, timedelta
import json
import math
import os

import additive_runs as ar
import mlb_starters
import savant_history as sh
import xera_lite
from calibration_loader import (
    load_starter_adjustment, save_starter_adjustment, set_candidate_mode,
    has_candidate, existing_candidate_notice,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _season_bounds(season):
    # MLB regular season ≈ late March → early October. Generous bounds.
    return f"{season}-03-20", f"{season}-10-05"


def get_season_games(season, start=None, end=None, verbose=True):
    """Fetch Final games with starters + scores, day-by-day (cached per season)."""
    s0, s1 = _season_bounds(season)
    start, end = start or s0, end or s1
    # v2 adds the home-perspective run margin (for spread/run-line fitting).
    path = os.path.join(CACHE_DIR, f"season_games_v2_{season}_{start}_{end}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    from datetime import date, timedelta
    games = []
    day = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while day <= last:
        ds = day.isoformat()
        try:
            data = mlb_starters._get("schedule", {
                "sportId": 1, "date": ds, "hydrate": "probablePitcher,linescore"})
        except Exception:
            data = {"dates": []}
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("detailedState") != "Final":
                    continue
                h, a = g["teams"]["home"], g["teams"]["away"]
                hs, as_ = h.get("score"), a.get("score")
                h_sp = (h.get("probablePitcher") or {}).get("id")
                a_sp = (a.get("probablePitcher") or {}).get("id")
                if hs is None or as_ is None or not h_sp or not a_sp:
                    continue
                games.append({
                    "date": ds,
                    "home_sp": h_sp, "away_sp": a_sp,
                    "home_win": 1 if hs > as_ else 0,
                    "total_runs": hs + as_,
                    "margin": hs - as_,  # home perspective (for spreads)
                })
        day += timedelta(days=1)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(games, f)
    if verbose:
        print(f"season {season}: {len(games)} Final games with starters")
    return games


def _season_venue_index(season):
    """Return actual MLB venue IDs keyed by historical matchup identity."""
    path = os.path.join(CACHE_DIR, f"season_venues_{season}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            cached = json.load(f)
    else:
        from backtest_props import _team_maps

        start, end = _season_bounds(season)
        id_abbr, _ = _team_maps(season)
        try:
            data = mlb_starters._get("schedule", {
                "sportId": 1,
                "startDate": start,
                "endDate": end,
                "hydrate": "probablePitcher,venue",
            }, timeout=120)
        except Exception:
            data = {"dates": []}
        cached = []
        for day in data.get("dates", []):
            for game in day.get("games", []):
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                venue_id = (game.get("venue") or {}).get("id")
                home_abbr = id_abbr.get((home.get("team") or {}).get("id"))
                away_abbr = id_abbr.get((away.get("team") or {}).get("id"))
                if not venue_id or not home_abbr or not away_abbr:
                    continue
                cached.append({
                    "date": day.get("date"),
                    "home_team": home_abbr,
                    "away_team": away_abbr,
                    "home_sp": (home.get("probablePitcher") or {}).get("id"),
                    "away_sp": (away.get("probablePitcher") or {}).get("id"),
                    "venue_id": str(venue_id),
                })
        if cached:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w") as f:
                json.dump(cached, f)

    exact = {}
    matchup = {}
    for row in cached:
        exact[(row["date"], str(row.get("home_sp")),
               str(row.get("away_sp")))] = row["venue_id"]
        matchup[(row["date"], row["home_team"],
                 row["away_team"])] = row["venue_id"]
    return exact, matchup


def _enrich_games(games, season, include_venues=False):
    """Attach team identities and separate run outcomes to cached games."""
    from backtest_props import season_schedule

    identities = {}
    for date, scheduled in season_schedule(season).items():
        for game in scheduled:
            key = (date, str(game.get("home_sp")), str(game.get("away_sp")))
            identities[key] = (game.get("home_abbr"), game.get("away_abbr"))

    venue_exact, venue_matchup = (
        _season_venue_index(season) if include_venues else ({}, {}))
    enriched = []
    for game in games:
        key = (game["date"], str(game["home_sp"]), str(game["away_sp"]))
        teams = identities.get(key)
        if not teams or not all(teams):
            continue
        home_runs = (game["total_runs"] + game["margin"]) / 2.0
        away_runs = (game["total_runs"] - game["margin"]) / 2.0
        enriched.append(dict(
            game,
            season=season,
            home_team=teams[0],
            away_team=teams[1],
            venue_id=(venue_exact.get(key) or venue_matchup.get(
                (game["date"], teams[0], teams[1]))),
            home_runs=home_runs,
            away_runs=away_runs,
        ))
    return enriched


def _bullpen_features(rows, season):
    """Leakage-safe as-of BULLPEN quality per team.

    A team's bullpen = every pitch it threw that day EXCEPT its own starter's.
    We identify the pitching team + starter for each Statcast row via the cached
    schedule (batting_team → that day's opponent + opposing probable starter),
    then index the non-starter (relief) batted balls by pitching team with
    prefix sums (AsOfIndex) for O(log n) as-of means.

    Returns (bullpen_index, league_bullpen_xwoba, game_team_map) where
    game_team_map keys (date, home_sp, away_sp) → (home_abbr, away_abbr) so a
    season_games row can be matched to its two teams.
    """
    from backtest_props import season_schedule, AsOfIndex

    sched = season_schedule(season)
    bat2pitch = {}   # (date, batting_team) -> (pitching_team, pitching_starter)
    game_teams = {}  # (date, home_sp, away_sp) -> (home_abbr, away_abbr)
    for date, gs in sched.items():
        for g in gs:
            ha, aa = g.get("home_abbr"), g.get("away_abbr")
            if not ha or not aa:
                continue
            bat2pitch[(date, ha)] = (aa, str(g.get("away_sp")))
            bat2pitch[(date, aa)] = (ha, str(g.get("home_sp")))
            game_teams[(date, str(g.get("home_sp")), str(g.get("away_sp")))] = (ha, aa)

    idx = AsOfIndex()
    bp_vals = []
    for r in rows:
        x = r.get("xwoba")
        if x is None:
            continue
        pt = bat2pitch.get((r.get("game_date"), r.get("batting_team")))
        if not pt:
            continue
        pitching_team, starter = pt
        if str(r.get("pitcher")) == starter:
            continue  # starter's pitch — not the bullpen
        idx.add(pitching_team, r["game_date"], x)
        bp_vals.append(x)
    league_bp = sum(bp_vals) / len(bp_vals) if bp_vals else 0.0
    return idx, league_bp, game_teams


def _bullpen_workload_features(rows, season):
    """Return each team's pregame, prior-three-day relief workload.

    Statcast contains one row per pitch, so this uses actual relief pitches
    rather than games played or a schedule proxy. Yesterday's pitches count
    most, then decay over the preceding two days. Every probable starter listed
    for a team/date is excluded, which keeps doubleheaders from misclassifying
    one of that day's starters as a reliever.

    Values are raw weighted pitch totals. The model centers them using only its
    pre-holdout training rows, avoiding leakage from the holdout workload
    distribution.
    """
    from backtest_props import season_schedule

    schedule = season_schedule(season)
    batting_opponent = {}
    starters = defaultdict(set)
    team_dates = set()
    for game_date, games in schedule.items():
        for game in games:
            home = game.get("home_abbr")
            away = game.get("away_abbr")
            if not home or not away:
                continue
            batting_opponent[(game_date, home)] = away
            batting_opponent[(game_date, away)] = home
            team_dates.add((game_date, home))
            team_dates.add((game_date, away))
            if game.get("home_sp"):
                starters[(game_date, home)].add(str(game["home_sp"]))
            if game.get("away_sp"):
                starters[(game_date, away)].add(str(game["away_sp"]))

    relief_pitches = defaultdict(int)
    for row in rows:
        game_date = row.get("game_date")
        pitching_team = batting_opponent.get(
            (game_date, row.get("batting_team")))
        pitcher = str(row.get("pitcher"))
        if (not pitching_team or not row.get("pitcher")
                or pitcher in starters[(game_date, pitching_team)]):
            continue
        relief_pitches[(game_date, pitching_team)] += 1

    raw_workload = {}
    day_weights = (1.0, 0.6, 0.3)
    for game_date, team in team_dates:
        current = date.fromisoformat(game_date)
        raw_workload[(game_date, team)] = sum(
            weight * relief_pitches[
                ((current - timedelta(days=days_ago)).isoformat(), team)
            ]
            for days_ago, weight in enumerate(day_weights, start=1)
        )
    return raw_workload


def _ip_index(season):
    """As-of average innings/start per starter, from cached pitcher gameLogs.

    Values are per-game OUTS (IP*3); the caller divides by 3. Leakage-safe:
    asof_mean only averages starts strictly before the game date.
    """
    from backtest_props import gamelog, AsOfIndex, starter_ids

    idx = AsOfIndex()
    for pid in starter_ids([season]):
        for sp in gamelog(pid, season, "pitching"):
            outs = sp.get("stat", {}).get("outs")
            date = sp.get("date")
            if outs is not None and date:
                try:
                    idx.add(str(pid), date, float(outs))
                except (TypeError, ValueError):
                    pass
    return idx


def _pitcher_xwoba_index(rows):
    """Build the starter xwOBAcon index used by the team-market backtest."""
    from backtest_props import AsOfIndex

    index = AsOfIndex()
    for row in rows:
        if (row.get("pitcher") and row.get("game_date")
                and row.get("xwoba") is not None):
            index.add(row["pitcher"], row["game_date"], row["xwoba"])
    return index


def build_dataset(games, season):
    """Attach leakage-safe as-of starter + bullpen features to each game.

    Returns rows with starter_edge, combined_excess, bullpen_excess (may be
    None when a team lacks enough as-of relief data), home_win, total_runs.
    Games missing an as-of estimate for either starter are skipped.
    """
    s0, s1 = _season_bounds(season)
    rows = sh.load_days(s0, s1)
    if not rows:
        raise RuntimeError("No Statcast days cached — run with --fetch first.")

    # League baseline xwOBAcon (same measure as the per-pitcher estimate) so the
    # run_suppression ratio is well-centered.
    league_vals = [r["xwoba"] for r in rows if r["xwoba"] is not None]
    league_xwoba = sum(league_vals) / len(league_vals)

    bp_idx, league_bp, game_teams = _bullpen_features(rows, season)
    bp_workload = _bullpen_workload_features(rows, season)

    # As-of starter index (prefix sums) — identical result to
    # sh.asof_pitcher_xwoba but O(log n) instead of a full scan per game.
    sp_idx = _pitcher_xwoba_index(rows)
    ip_idx = _ip_index(season)  # as-of avg innings/start per starter

    # Two-sided edge: as-of offense a lineup PRODUCES vs a pitcher's hand, so a
    # starter's effective run-prevention is degraded by the offense he faces.
    off_idx = _offense_index(rows)
    sp_hand = {}
    for r in rows:
        p, ph = str(r.get("pitcher")), r.get("p_throws")
        if p and ph and p not in sp_hand:
            sp_hand[p] = ph

    out = []
    for g in games:
        hx = sp_idx.asof_mean(str(g["home_sp"]), g["date"])
        ax = sp_idx.asof_mean(str(g["away_sp"]), g["date"])
        if hx is None or ax is None:
            continue
        # run_suppression: lower xwOBA allowed = better pitcher = >1 (bounded).
        h_rs = max(0.5, min(2.0, league_xwoba / hx))
        a_rs = max(0.5, min(2.0, league_xwoba / ax))

        # Bullpen suppression per team (same league-relative form as starters).
        h_bs = a_bs = None
        bullpen_excess = None
        gt = game_teams.get((g["date"], str(g["home_sp"]), str(g["away_sp"])))
        ha, aa = gt if gt else (None, None)   # resolved team abbrs (None if unmatched)
        h_bp_workload = a_bp_workload = None
        if gt and league_bp:
            ha, aa = gt
            h_bp = bp_idx.asof_mean(ha, g["date"])
            a_bp = bp_idx.asof_mean(aa, g["date"])
            if h_bp and a_bp:
                h_bs = max(0.5, min(2.0, league_bp / h_bp))
                a_bs = max(0.5, min(2.0, league_bp / a_bp))
                bullpen_excess = (h_bs - 1.0) + (a_bs - 1.0)
            h_bp_workload = bp_workload.get((g["date"], ha))
            a_bp_workload = bp_workload.get((g["date"], aa))

        # As-of average innings/start (min 3 prior starts), None if unknown.
        h_outs = ip_idx.asof_mean(str(g["home_sp"]), g["date"], min_bbe=3)
        a_outs = ip_idx.asof_mean(str(g["away_sp"]), g["date"], min_bbe=3)
        h_ip = h_outs / 3.0 if h_outs is not None else None
        a_ip = a_outs / 3.0 if a_outs is not None else None

        # Offense faced by each staff (home staff faces the away lineup vs the
        # home starter's hand, and vice versa). Factor >1 = tougher-than-league
        # offense. Defaults to 1.0 (neutral) when as-of data is missing.
        h_off = a_off = 1.0
        if gt:
            ha, aa = gt
            hh, ah = sp_hand.get(str(g["home_sp"])), sp_hand.get(str(g["away_sp"]))
            if hh:
                ov = off_idx.asof_mean(f"{aa}|{hh}", g["date"])
                if ov:
                    h_off = max(0.5, min(2.0, ov / league_xwoba))
            if ah:
                ov = off_idx.asof_mean(f"{ha}|{ah}", g["date"])
                if ov:
                    a_off = max(0.5, min(2.0, ov / league_xwoba))

        out.append({
            "season": season,
            "date": g["date"],
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "home_abbr": ha, "away_abbr": aa,   # resolved abbrs for the RP-bullpen join
            "venue_id": g.get("venue_id"),
            "home_sp": g["home_sp"],
            "away_sp": g["away_sp"],
            "home_runs": g.get("home_runs"),
            "away_runs": g.get("away_runs"),
            "starter_edge": math.tanh(h_rs - a_rs),
            "combined_excess": (h_rs - 1.0) + (a_rs - 1.0),
            "bullpen_excess": bullpen_excess,
            "home_win": g["home_win"],
            "total_runs": g["total_runs"],
            # extra raw fields for the innings-weighting A/B (fit() ignores them)
            "h_sp_sup": h_rs, "a_sp_sup": a_rs,
            "h_bp_sup": h_bs, "a_bp_sup": a_bs,
            "h_bp_workload": h_bp_workload,
            "a_bp_workload": a_bp_workload,
            "h_ip": h_ip, "a_ip": a_ip,
            "h_off_faced": h_off, "a_off_faced": a_off,
            "margin": g.get("margin"),
        })
    return out, league_xwoba


# ── pure-python fitters ────────────────────────────────────────────────────
def fit_ols_1d(xs, ys):
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    var = sum((x - xbar) ** 2 for x in xs)
    if var == 0:
        return 0.0, ybar
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / var
    return slope, ybar - slope * xbar


def _sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def fit_logistic_1d(xs, ys, lr=0.1, iters=5000):
    """Fit P(y=1)=sigmoid(a + b*x) via gradient descent (x standardized)."""
    n = len(xs)
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / n) ** 0.5 or 1.0
    z = [(x - mu) / sd for x in xs]
    a, b = 0.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        for zi, yi in zip(z, ys):
            err = _sigmoid(a + b * zi) - yi
            ga += err
            gb += err * zi
        a -= lr * ga / n
        b -= lr * gb / n
    # De-standardize coefficient back to raw-x scale.
    return b / sd, a - b * mu / sd


def brier(preds, ys):
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)


def rmse(preds, ys):
    return (sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)) ** 0.5


def build_pooled_dataset(seasons, include_venues=False):
    """Pool leakage-safe features across multiple seasons.

    Each season is normalized against its OWN league xwOBAcon baseline (so
    era-to-era run-environment differences don't distort the league-relative
    run_suppression feature), then the resulting rows are concatenated.
    """
    pooled = []
    per_season = {}
    for s in seasons:
        games = _enrich_games(
            get_season_games(s), s, include_venues=include_venues)
        data, lg = build_dataset(games, s)
        per_season[s] = {"games": len(data), "league_xwoba": lg}
        pooled.extend(data)
    return pooled, per_season


def _staff_suppression(row, side):
    """Return the innings-weighted starter/bullpen suppression for one side."""
    starter = row[f"{side}_sp_sup"]
    bullpen = row.get(f"{side}_bp_sup")
    innings = row.get(f"{side}_ip")
    if bullpen and innings:
        weight = max(0.30, min(0.85, innings / 9.0))
        return weight * starter + (1.0 - weight) * bullpen
    return starter


def _eff_edge(d):
    """Innings-weighted, two-sided effective run-prevention edge (home − away),
    tanh-bounded.

    Mirrors mlb_starters.build_matchup_features._eff exactly: a starter's
    quality counts in proportion to how deep he goes, the bullpen covers the
    rest, and the blend is then divided by the offense factor of the lineup the
    staff faces (pitcher-vs-lineup, both directions). Falls back to
    starter-quality-only when bullpen/innings are missing, and to a neutral 1.0
    offense factor when lineup data is missing.
    """
    return math.tanh(
        _staff_suppression(d, "h") / (d.get("h_off_faced") or 1.0)
        - _staff_suppression(d, "a") / (d.get("a_off_faced") or 1.0))


def fit(seasons, do_save=False):
    if isinstance(seasons, int):
        seasons = [seasons]
    seasons_str = ",".join(str(s) for s in seasons)
    data, per_season = build_pooled_dataset(seasons)
    league_xwoba = (sum(v["league_xwoba"] for v in per_season.values())
                    / len(per_season)) if per_season else 0.0
    for s, v in per_season.items():
        print(f"  season {s}: {v['games']} gradeable games "
              f"(league xwOBAcon {v['league_xwoba']:.3f})")
    if len(data) < 100:
        print(f"Only {len(data)} usable games total — need more cached Statcast days.")
        return
    # Innings-weighted effective-suppression edge — mirrors the live
    # build_matchup_features blend exactly (falls back to starter-quality-only
    # when a game lacks as-of bullpen or innings data).
    edge = [_eff_edge(d) for d in data]
    win = [d["home_win"] for d in data]
    exc = [d["combined_excess"] for d in data]
    tot = [d["total_runs"] for d in data]

    # Moneyline: home_win ~ eff_edge (logistic).
    b_ml, a_ml = fit_logistic_1d(edge, win)
    base_ml = sum(win) / len(win)
    base_brier = brier([base_ml] * len(win), win)
    feat_brier = brier([_sigmoid(a_ml + b_ml * e) for e in edge], win)

    # Spreads: margin ~ eff_edge (OLS). Slope = runs of margin per unit edge,
    # applied at runtime as pred_margin += spreads_w * edge.
    mar_data = [d for d in data if d.get("margin") is not None]
    spreads_w = 0.0
    sp_base_rmse = sp_feat_rmse = None
    if len(mar_data) >= 100:
        m_edge = [_eff_edge(d) for d in mar_data]
        mar = [d["margin"] for d in mar_data]
        sp_slope, sp_int = fit_ols_1d(m_edge, mar)
        # Clamp loose: eff_edge is tanh-compressed (p99 ≈ 0.36), so even slope 8
        # shifts margin < 3 runs on realistic games; guards only pathologies.
        spreads_w = max(0.0, min(8.0, sp_slope))
        sp_base_rmse = rmse([sum(mar) / len(mar)] * len(mar), mar)
        sp_feat_rmse = rmse([sp_int + sp_slope * e for e in m_edge], mar)

    # Totals: total_runs ~ combined_excess (OLS). Better arms → fewer runs.
    slope, intercept = fit_ols_1d(exc, tot)
    run_scale = -slope
    base_rmse = rmse([sum(tot) / len(tot)] * len(tot), tot)
    feat_rmse = rmse([intercept + slope * x for x in exc], tot)

    # Loosened from 1.5: the innings-weighted edge is tanh-compressed, so a
    # larger logit coef still yields a modest shift on realistic games (p99
    # edge 0.36 → 0.7 logits at coef 1.9).
    ml_weight = max(0.0, min(2.5, b_ml))
    run_scale_c = max(0.0, min(3.0, run_scale))

    # ── Bullpen: marginal fit on the totals RESIDUAL, holding run_scale fixed ──
    # The model applies bullpen additively on top of the starter shift, so we
    # fit the bullpen coefficient against what the starter-only model leaves on
    # the table (keeps the already-validated run_scale untouched).
    bp_data = [d for d in data if d.get("bullpen_excess") is not None]
    bullpen_w = 0.0
    bp_base_rmse = bp_feat_rmse = None
    if len(bp_data) >= 100:
        bexc = [d["bullpen_excess"] for d in bp_data]
        bp_tot = [d["total_runs"] for d in bp_data]
        starter_pred = [intercept + slope * d["combined_excess"] for d in bp_data]
        resid = [t - p for t, p in zip(bp_tot, starter_pred)]
        bp_slope, _ = fit_ols_1d(bexc, resid)
        bullpen_w = max(0.0, min(3.0, -bp_slope))
        bp_base_rmse = rmse(starter_pred, bp_tot)
        bp_feat_rmse = rmse([p - bullpen_w * e for p, e in zip(starter_pred, bexc)],
                            bp_tot)

    print(f"\n=== starter_adjustment fit — {seasons_str} ({len(data)} games) ===")
    print(f"league xwOBAcon baseline: {league_xwoba:.3f}")
    print(f"moneyline weight (logit coef on innings-wtd edge): {b_ml:.3f} "
          f"-> clamped {ml_weight:.3f}")
    print(f"  Brier: baseline {base_brier:.4f} -> with-starters {feat_brier:.4f} "
          f"({'BETTER' if feat_brier < base_brier else 'no gain'})")
    if sp_base_rmse is not None:
        print(f"spreads weight (runs of margin / unit edge): {spreads_w:.3f} "
              f"({len(mar_data)} games)")
        print(f"  Margin RMSE: baseline {sp_base_rmse:.3f} -> with-starters "
              f"{sp_feat_rmse:.3f} "
              f"({'BETTER' if sp_feat_rmse < sp_base_rmse else 'no gain'})")
    print(f"run_scale (runs suppressed / unit excess): {run_scale:.3f} "
          f"-> clamped {run_scale_c:.3f}")
    print(f"  Totals RMSE: baseline {base_rmse:.3f} -> with-starters {feat_rmse:.3f} "
          f"({'BETTER' if feat_rmse < base_rmse else 'no gain'})")
    if bp_base_rmse is not None:
        print(f"bullpen weight (runs / unit bullpen excess): {bullpen_w:.3f} "
              f"({len(bp_data)} games w/ as-of relief data)")
        print(f"  Totals RMSE (on residual): starters {bp_base_rmse:.3f} -> "
              f"+bullpen {bp_feat_rmse:.3f} "
              f"({'BETTER' if bp_feat_rmse < bp_base_rmse else 'no gain'})")
    else:
        print("bullpen weight: not enough as-of relief data to fit.")
    if do_save:
        cur = load_starter_adjustment("baseball_mlb") or {}
        cur.update({"moneyline": round(ml_weight, 3),
                    "spreads": round(spreads_w, 3),
                    "run_scale": round(run_scale_c, 3),
                    "bullpen": round(bullpen_w, 3)})
        # The run_scale/bullpen shift is already applied to projected_total,
        # which is now the probability distribution's mean. A second totals
        # logit coefficient would double-count the same starter signal.
        cur.pop("totals", None)
        cur["_note"] = (f"moneyline/spreads/run_scale/bullpen fit from "
                        f"{seasons_str} ({len(data)} games, "
                        f"{len(bp_data)} w/ bullpen); moneyline & spreads use the "
                        f"innings-weighted edge; independently fitted prop "
                        f"weights and bvp setting preserved.")
        save_starter_adjustment("baseball_mlb", cur,
                                meta={"team_markets": {
                                    "source": f"backtest_starters:{seasons_str}",
                                    "fit": True,
                                    "n_games": len(data),
                                }})
        print("saved fitted weights "
              "(moneyline, spreads, run_scale, bullpen).")


EXPECTED_RUN_WEIGHT_GRID = tuple(i / 4.0 for i in range(7))
MIN_HOLDOUT_PROBABILITY_GAIN = 0.001
MIN_HOLDOUT_NLL_GAIN = 0.001
MIN_HOLDOUT_RMSE_GAIN = 0.005


def _expected_run_multipliers(row, offense_weight, pitching_weight):
    """Return home/away multiplicative run factors for one historical game."""
    # a_off_faced is the HOME lineup facing the away starter's hand; the
    # inverse naming comes from indexing offense from the staff's perspective.
    home_offense = row.get("a_off_faced") or 1.0
    away_offense = row.get("h_off_faced") or 1.0
    home_staff = _staff_suppression(row, "h")
    away_staff = _staff_suppression(row, "a")
    return (
        home_offense ** offense_weight / away_staff ** pitching_weight,
        away_offense ** offense_weight / home_staff ** pitching_weight,
    )


def fit_expected_run_model(rows):
    """Fit offense/pitching powers before the chronological holdout.

    The grid minimizes Poisson negative log likelihood. Separate home and away
    baselines absorb the run environment and ordinary home-field advantage.
    """
    usable = [row for row in rows
              if row.get("home_runs") is not None
              and row.get("away_runs") is not None]
    if not usable:
        return None
    best = None
    for offense_weight in EXPECTED_RUN_WEIGHT_GRID:
        for pitching_weight in EXPECTED_RUN_WEIGHT_GRID:
            multipliers = [
                _expected_run_multipliers(
                    row, offense_weight, pitching_weight)
                for row in usable
            ]
            home_base = (sum(row["home_runs"] for row in usable)
                         / sum(pair[0] for pair in multipliers))
            away_base = (sum(row["away_runs"] for row in usable)
                         / sum(pair[1] for pair in multipliers))
            loss = 0.0
            for row, (home_mult, away_mult) in zip(usable, multipliers):
                home_expected = max(0.5, min(12.0, home_base * home_mult))
                away_expected = max(0.5, min(12.0, away_base * away_mult))
                loss += home_expected - row["home_runs"] * math.log(home_expected)
                loss += away_expected - row["away_runs"] * math.log(away_expected)
            candidate = {
                "offense_weight": offense_weight,
                "pitching_weight": pitching_weight,
                "home_base_runs": home_base,
                "away_base_runs": away_base,
                "poisson_nll": loss / (2.0 * len(usable)),
                "n_train": len(usable),
            }
            if best is None or candidate["poisson_nll"] < best["poisson_nll"]:
                best = candidate
    return best


def project_expected_runs(row, model):
    """Project home and away runs from one fitted challenger model."""
    home_mult, away_mult = _expected_run_multipliers(
        row, model["offense_weight"], model["pitching_weight"])
    home_runs = mlb_starters.expected_runs_from_factors(
        model["home_base_runs"], home_mult, 1.0)
    away_runs = mlb_starters.expected_runs_from_factors(
        model["away_base_runs"], away_mult, 1.0)

    park_factors = model.get("park_factors") or {}
    park_strength = model.get("park_strength", 0.0)
    park_key = row.get("venue_id") or row.get("home_team")
    raw_park = park_factors.get(park_key, 1.0)
    park_multiplier = 1.0 + park_strength * (raw_park - 1.0)

    fatigue_weight = model.get("fatigue_weight", 0.0)
    # The home offense faces the away bullpen and vice versa.
    workload_center = model.get("workload_center")
    if workload_center:
        away_fatigue = max(-1.0, min(
            2.0, (row.get("a_bp_workload") or 0.0) / workload_center - 1.0))
        home_fatigue = max(-1.0, min(
            2.0, (row.get("h_bp_workload") or 0.0) / workload_center - 1.0))
    else:
        away_fatigue = home_fatigue = 0.0
    home_runs *= park_multiplier * math.exp(fatigue_weight * away_fatigue)
    away_runs *= park_multiplier * math.exp(fatigue_weight * home_fatigue)
    return (max(0.5, min(12.0, home_runs)),
            max(0.5, min(12.0, away_runs)))


def _poisson_score_nll(actual, expected):
    expected = max(1e-9, expected)
    return expected - actual * math.log(expected) + math.lgamma(actual + 1.0)


def _negative_binomial_score_nll(actual, expected, dispersion):
    """Negative log likelihood under variance=mean+dispersion*mean^2."""
    if dispersion <= 0:
        return _poisson_score_nll(actual, expected)
    size = 1.0 / dispersion
    success_probability = size / (size + expected)
    log_probability = (
        math.lgamma(actual + size)
        - math.lgamma(size)
        - math.lgamma(actual + 1.0)
        + size * math.log(success_probability)
        + actual * math.log1p(-success_probability)
    )
    return -log_probability


def _score_nll(rows, model, distribution="poisson", dispersion=0.0):
    losses = []
    for row in rows:
        home_expected, away_expected = project_expected_runs(row, model)
        for actual, expected in (
                (row["home_runs"], home_expected),
                (row["away_runs"], away_expected)):
            if distribution == "negative_binomial":
                losses.append(_negative_binomial_score_nll(
                    actual, expected, dispersion))
            else:
                losses.append(_poisson_score_nll(actual, expected))
    return sum(losses) / len(losses)


def _raw_park_factors(rows, model):
    """Fit home-venue run multipliers from pre-holdout score residuals."""
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    for row in rows:
        home_expected, away_expected = project_expected_runs(row, model)
        park_key = row.get("venue_id") or row["home_team"]
        bucket = totals[park_key]
        bucket[0] += row["home_runs"] + row["away_runs"]
        bucket[1] += home_expected + away_expected
        bucket[2] += 1
    return {
        team: max(0.75, min(1.25, actual / expected))
        for team, (actual, expected, games) in totals.items()
        if games >= 20 and expected > 0
    }


def fit_expected_run_extensions(rows, base_model, use_park=False,
                                use_fatigue=False):
    """Fit optional park and three-day bullpen-workload terms on train only."""
    park_factors = _raw_park_factors(rows, base_model) if use_park else {}
    park_grid = [step / 10.0 for step in range(11)] if use_park else [0.0]
    workloads = [
        row.get(f"{side}_bp_workload")
        for row in rows
        for side in ("h", "a")
        if row.get(f"{side}_bp_workload") is not None
    ] if use_fatigue else []
    workload_center = (
        sum(workloads) / len(workloads) if workloads else None
    )
    fatigue_grid = (
        [step / 20.0 for step in range(-5, 11)]
        if workload_center else [0.0]
    )
    best = None
    for park_strength in park_grid:
        for fatigue_weight in fatigue_grid:
            candidate = dict(base_model)
            candidate.update({
                "park_factors": park_factors,
                "park_strength": park_strength,
                "fatigue_weight": fatigue_weight,
                "workload_center": workload_center,
            })
            loss = _score_nll(rows, candidate)
            if best is None or loss < best[0]:
                best = (loss, candidate)
    best[1]["train_poisson_nll"] = best[0]
    return best[1]


def fit_negative_binomial_dispersion(rows, model):
    """Fit score overdispersion on pre-holdout games only."""
    best = None
    for step in range(51):
        dispersion = step / 100.0
        loss = _score_nll(
            rows, model, distribution="negative_binomial",
            dispersion=dispersion)
        if best is None or loss < best[1]:
            best = (dispersion, loss)
    return best


def _fit_probability_shrink(probabilities, outcomes):
    """Fit p' = .5 + s*(p-.5) on training observations only."""
    best = None
    for step in range(21):
        shrink = step / 20.0
        predictions = [0.5 + shrink * (prob - 0.5)
                       for prob in probabilities]
        score = brier(predictions, outcomes)
        if best is None or score < best[1]:
            best = (shrink, score)
    return best[0]


def _probability_metrics(probabilities, outcomes):
    clipped = [max(1e-9, min(1.0 - 1e-9, prob))
               for prob in probabilities]
    return {
        "brier": brier(clipped, outcomes),
        "log_loss": -sum(
            outcome * math.log(prob) + (1 - outcome) * math.log(1 - prob)
            for prob, outcome in zip(clipped, outcomes)
        ) / len(outcomes),
        "accuracy": sum(
            (prob >= 0.5) == bool(outcome)
            for prob, outcome in zip(clipped, outcomes)
        ) / len(outcomes),
    }


def _margin_metrics(predictions, outcomes):
    return {
        "mae": sum(abs(pred - actual)
                   for pred, actual in zip(predictions, outcomes)) / len(outcomes),
        "rmse": rmse(predictions, outcomes),
    }


def _challenger_variant_metrics(train, holdout, model,
                                distribution="poisson", dispersion=0.0):
    """Fit probability shrink on train and grade one challenger on holdout."""
    train_ml = []
    train_spread = []
    train_win = []
    train_cover = []
    probability_function = (
        mlb_starters.negative_binomial_margin_probability
        if distribution == "negative_binomial"
        else mlb_starters.poisson_margin_probability
    )
    for row in train:
        home_runs, away_runs = project_expected_runs(row, model)
        train_ml.append(mlb_starters.pythagorean_win_probability(
            home_runs, away_runs))
        if distribution == "negative_binomial":
            spread_probability = probability_function(
                home_runs, away_runs, -1.5, dispersion)
        else:
            spread_probability = probability_function(
                home_runs, away_runs, -1.5)
        train_spread.append(spread_probability)
        train_win.append(row["home_win"])
        train_cover.append(1 if row["margin"] > 1.5 else 0)
    ml_shrink = _fit_probability_shrink(train_ml, train_win)
    spread_shrink = _fit_probability_shrink(train_spread, train_cover)

    holdout_ml = []
    holdout_spread = []
    expected_margin = []
    expected_total = []
    actual_win = []
    actual_cover = []
    actual_margin = []
    actual_total = []
    for row in holdout:
        home_runs, away_runs = project_expected_runs(row, model)
        ml_probability = mlb_starters.pythagorean_win_probability(
            home_runs, away_runs)
        if distribution == "negative_binomial":
            spread_probability = probability_function(
                home_runs, away_runs, -1.5, dispersion)
        else:
            spread_probability = probability_function(
                home_runs, away_runs, -1.5)
        holdout_ml.append(0.5 + ml_shrink * (ml_probability - 0.5))
        holdout_spread.append(
            0.5 + spread_shrink * (spread_probability - 0.5))
        expected_margin.append(home_runs - away_runs)
        expected_total.append(home_runs + away_runs)
        actual_win.append(row["home_win"])
        actual_cover.append(1 if row["margin"] > 1.5 else 0)
        actual_margin.append(row["margin"])
        actual_total.append(row["total_runs"])
    return {
        "ml_shrink": ml_shrink,
        "spread_shrink": spread_shrink,
        "ml": _probability_metrics(holdout_ml, actual_win),
        "spread": _probability_metrics(holdout_spread, actual_cover),
        "margin": _margin_metrics(expected_margin, actual_margin),
        "total_rmse": rmse(expected_total, actual_total),
        "score_nll": _score_nll(
            holdout, model, distribution=distribution,
            dispersion=dispersion),
    }


def _game_key(row):
    return (row["season"], row["date"],
            str(row["home_sp"]), str(row["away_sp"]))


def _current_margin_predictions(seasons, rows):
    """Reproduce the current live MLB margin engine without future games."""
    from analysis import _norm_cdf, _predict_margin

    row_by_key = {_game_key(row): row for row in rows}
    predictions = {}
    for season in seasons:
        games = _enrich_games(get_season_games(season), season)
        by_date = defaultdict(list)
        for game in games:
            by_date[game["date"]].append(game)
        history = defaultdict(list)
        for date in sorted(by_date):
            # Grade every same-day game before adding any same-day result. This
            # keeps doubleheaders and split games from leaking into each other.
            for game in by_date[date]:
                row = row_by_key.get(_game_key(game))
                if not row:
                    continue
                home_prior = list(reversed(history[game["home_team"]][-20:]))
                away_prior = list(reversed(history[game["away_team"]][-20:]))
                if not home_prior or not away_prior:
                    continue
                margin = _predict_margin(
                    {"home_team": game["home_team"],
                     "away_team": game["away_team"]},
                    {"recent_games": home_prior},
                    {"recent_games": away_prior},
                    "baseball_mlb",
                    {"starter_edge": _eff_edge(row)},
                )
                if margin is None:
                    continue
                predicted_margin, predicted_std, _, _ = margin
                predictions[_game_key(row)] = {
                    "margin": predicted_margin,
                    "std": predicted_std,
                    "win_probability": _norm_cdf(
                        predicted_margin / predicted_std),
                    "minus_1_5_probability": _norm_cdf(
                        (predicted_margin - 1.5) / predicted_std),
                }
            for game in by_date[date]:
                record = {
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "home_score": game["home_runs"],
                    "away_score": game["away_runs"],
                }
                history[game["home_team"]].append(record)
                history[game["away_team"]].append(record)
    return predictions


def _fit_blend_weight(current_values, challenger_values, outcomes):
    """Fit challenger share on training MSE; 0=current and 1=challenger."""
    best = None
    for step in range(21):
        weight = step / 20.0
        predictions = [
            current + weight * (challenger - current)
            for current, challenger in zip(current_values, challenger_values)
        ]
        loss = sum((prediction - outcome) ** 2
                   for prediction, outcome in zip(predictions, outcomes))
        loss /= len(outcomes)
        if best is None or loss < best[1]:
            best = (weight, loss)
    return best[0]


def _final_candidate_series(rows, current, model):
    """Build aligned current and raw expected-runs predictions."""
    from analysis import _apply_shrink

    series = defaultdict(list)
    for row in rows:
        baseline = current.get(_game_key(row))
        if not baseline:
            continue
        home_runs, away_runs = project_expected_runs(row, model)
        series["rows"].append(row)
        series["current_ml"].append(_apply_shrink(
            baseline["win_probability"], "baseball_mlb", "moneyline"))
        series["challenger_ml"].append(
            mlb_starters.pythagorean_win_probability(home_runs, away_runs))
        series["current_spread"].append(_apply_shrink(
            baseline["minus_1_5_probability"],
            "baseball_mlb", "spreads"))
        series["challenger_spread"].append(
            mlb_starters.poisson_margin_probability(
                home_runs, away_runs, -1.5))
        series["current_margin"].append(baseline["margin"])
        series["challenger_margin"].append(home_runs - away_runs)
        series["challenger_total"].append(home_runs + away_runs)
        series["win"].append(row["home_win"])
        series["cover"].append(1 if row["margin"] > 1.5 else 0)
        series["margin"].append(row["margin"])
        series["total"].append(row["total_runs"])
    return series


def _blend_values(current_values, challenger_values, weight):
    return [
        current + weight * (challenger - current)
        for current, challenger in zip(current_values, challenger_values)
    ]


def test_final_expected_run_candidates(seasons, holdout_start=None):
    """Test predeclared expected-runs candidates on an untouched final season.

    Candidates are deliberately fixed before inspecting the holdout:
      * raw (unshrunk) Pythagorean moneyline probability,
      * current/expected-runs blends fitted only on the training window, and
      * strength-1 venue factors estimated only from completed prior seasons.
    """
    latest_season = max(seasons)
    prior_seasons = [season for season in seasons if season < latest_season]
    if not prior_seasons:
        print("Final validation needs at least one completed prior season.")
        return None
    holdout_start = holdout_start or f"{latest_season}-07-01"
    rows, _ = build_pooled_dataset(seasons, include_venues=True)
    train = [row for row in rows
             if row["season"] < latest_season or row["date"] < holdout_start]
    holdout = [row for row in rows
               if row["season"] == latest_season
               and row["date"] >= holdout_start]
    prior_rows = [row for row in train if row["season"] < latest_season]
    venue_coverage = sum(row.get("venue_id") is not None for row in prior_rows)
    if (len(train) < 500 or len(holdout) < 200
            or venue_coverage < 0.9 * len(prior_rows)):
        print("Insufficient train, holdout, or actual-venue coverage for final test.")
        return None

    model = fit_expected_run_model(train)
    current = _current_margin_predictions(seasons, rows)
    train_series = _final_candidate_series(train, current, model)
    holdout_series = _final_candidate_series(holdout, current, model)
    if len(holdout_series["rows"]) < 200:
        print("Not enough final-holdout games with current-model history.")
        return None

    blend_weights = {
        "moneyline": _fit_blend_weight(
            train_series["current_ml"], train_series["challenger_ml"],
            train_series["win"]),
        "spread": _fit_blend_weight(
            train_series["current_spread"],
            train_series["challenger_spread"], train_series["cover"]),
        "margin": _fit_blend_weight(
            train_series["current_margin"],
            train_series["challenger_margin"], train_series["margin"]),
    }
    ensemble_ml = _blend_values(
        holdout_series["current_ml"], holdout_series["challenger_ml"],
        blend_weights["moneyline"])
    ensemble_spread = _blend_values(
        holdout_series["current_spread"],
        holdout_series["challenger_spread"], blend_weights["spread"])
    ensemble_margin = _blend_values(
        holdout_series["current_margin"],
        holdout_series["challenger_margin"], blend_weights["margin"])

    # The park candidate uses only completed prior-season outcomes. Strength
    # was fixed at 1.0 before opening the final holdout; nothing is tuned
    # against the final season's results.
    park_model = dict(model)
    park_model.update({
        "park_factors": _raw_park_factors(prior_rows, model),
        "park_strength": 1.0,
    })
    park_series = _final_candidate_series(holdout, current, park_model)
    park_coverage = sum(
        row.get("venue_id") in park_model["park_factors"]
        for row in park_series["rows"])

    current_ml_metrics = _probability_metrics(
        holdout_series["current_ml"], holdout_series["win"])
    raw_ml_metrics = _probability_metrics(
        holdout_series["challenger_ml"], holdout_series["win"])
    ensemble_ml_metrics = _probability_metrics(
        ensemble_ml, holdout_series["win"])
    current_spread_metrics = _probability_metrics(
        holdout_series["current_spread"], holdout_series["cover"])
    raw_spread_metrics = _probability_metrics(
        holdout_series["challenger_spread"], holdout_series["cover"])
    ensemble_spread_metrics = _probability_metrics(
        ensemble_spread, holdout_series["cover"])
    park_spread_metrics = _probability_metrics(
        park_series["challenger_spread"], park_series["cover"])
    current_margin_metrics = _margin_metrics(
        holdout_series["current_margin"], holdout_series["margin"])
    raw_margin_metrics = _margin_metrics(
        holdout_series["challenger_margin"], holdout_series["margin"])
    ensemble_margin_metrics = _margin_metrics(
        ensemble_margin, holdout_series["margin"])
    park_margin_metrics = _margin_metrics(
        park_series["challenger_margin"], park_series["margin"])
    raw_total_rmse = rmse(
        holdout_series["challenger_total"], holdout_series["total"])
    park_total_rmse = rmse(
        park_series["challenger_total"], park_series["total"])
    raw_score_nll = _score_nll(holdout_series["rows"], model)
    park_score_nll = _score_nll(park_series["rows"], park_model)

    raw_ml_passed = (
        current_ml_metrics["brier"] - raw_ml_metrics["brier"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
        and current_ml_metrics["log_loss"] - raw_ml_metrics["log_loss"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
    )
    ensemble_ml_passed = (
        current_ml_metrics["brier"] - ensemble_ml_metrics["brier"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
        and current_ml_metrics["log_loss"] - ensemble_ml_metrics["log_loss"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
    )
    raw_spread_passed = (
        current_spread_metrics["brier"] - raw_spread_metrics["brier"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
        and current_spread_metrics["log_loss"]
        - raw_spread_metrics["log_loss"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
    )
    ensemble_spread_passed = (
        current_spread_metrics["brier"] - ensemble_spread_metrics["brier"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
        and current_spread_metrics["log_loss"]
        - ensemble_spread_metrics["log_loss"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
    )
    raw_margin_passed = (
        current_margin_metrics["rmse"] - raw_margin_metrics["rmse"]
        >= MIN_HOLDOUT_RMSE_GAIN
    )
    ensemble_margin_passed = (
        current_margin_metrics["rmse"] - ensemble_margin_metrics["rmse"]
        >= MIN_HOLDOUT_RMSE_GAIN
    )
    park_passed = (
        park_coverage >= 0.9 * len(park_series["rows"])
        and raw_score_nll - park_score_nll >= MIN_HOLDOUT_NLL_GAIN
        and raw_total_rmse - park_total_rmse >= MIN_HOLDOUT_RMSE_GAIN
        and park_spread_metrics["brier"] <= raw_spread_metrics["brier"]
    )

    print(f"\n=== FINAL expected-runs candidates — holdout {holdout_start}+ ===")
    print(f"prior seasons={','.join(map(str, prior_seasons))}; "
          f"train={len(train_series['rows'])}; "
          f"holdout={len(holdout_series['rows'])}")
    print("expected-runs fit: offense={offense_weight:.2f}, "
          "pitching={pitching_weight:.2f}, home_base={home_base_runs:.3f}, "
          "away_base={away_base_runs:.3f}".format(**model))
    print("ensemble challenger shares fitted on train: "
          f"ML={blend_weights['moneyline']:.2f}, "
          f"home -1.5={blend_weights['spread']:.2f}, "
          f"margin={blend_weights['margin']:.2f}")
    print("\nMONEYLINE")
    for label, metrics in (
            ("current", current_ml_metrics),
            ("raw Pyth", raw_ml_metrics),
            ("ensemble", ensemble_ml_metrics)):
        print(f"  {label:<10} Brier={metrics['brier']:.4f} "
              f"logloss={metrics['log_loss']:.4f} "
              f"accuracy={metrics['accuracy']:.2%}")
    print("\nMARGIN / HOME -1.5")
    for label, margin_metrics, spread_metrics in (
            ("current", current_margin_metrics, current_spread_metrics),
            ("raw runs", raw_margin_metrics, raw_spread_metrics),
            ("ensemble", ensemble_margin_metrics, ensemble_spread_metrics),
            ("prior park", park_margin_metrics, park_spread_metrics)):
        print(f"  {label:<10} margin RMSE={margin_metrics['rmse']:.3f} "
              f"-1.5 Brier={spread_metrics['brier']:.4f}")
    print("\nPRIOR-SEASON ACTUAL-VENUE PARK FACTORS")
    print(f"  holdout coverage={park_coverage}/{len(park_series['rows'])}; "
          f"score NLL {raw_score_nll:.4f} -> {park_score_nll:.4f}; "
          f"total RMSE {raw_total_rmse:.3f} -> {park_total_rmse:.3f}")
    print("\nGATES")
    print(f"  raw Pythagorean ML: {'PASS' if raw_ml_passed else 'FAIL'}")
    print(f"  current/Pythagorean ML ensemble: "
          f"{'PASS' if ensemble_ml_passed else 'FAIL'}")
    print(f"  raw expected-runs margin / run line: "
          f"{'PASS' if raw_margin_passed and raw_spread_passed else 'FAIL'}")
    print(f"  current/expected-runs margin / run-line ensemble: "
          f"{'PASS' if ensemble_margin_passed and ensemble_spread_passed else 'FAIL'}")
    print(f"  prior-season park factors: {'PASS' if park_passed else 'FAIL'}")
    print("  No live settings changed.")
    return {
        "holdout_start": holdout_start,
        "train_n": len(train_series["rows"]),
        "holdout_n": len(holdout_series["rows"]),
        "model": model,
        "blend_weights": blend_weights,
        "current_ml": current_ml_metrics,
        "raw_ml": raw_ml_metrics,
        "ensemble_ml": ensemble_ml_metrics,
        "current_spread": current_spread_metrics,
        "raw_spread": raw_spread_metrics,
        "ensemble_spread": ensemble_spread_metrics,
        "park_spread": park_spread_metrics,
        "current_margin": current_margin_metrics,
        "raw_margin": raw_margin_metrics,
        "ensemble_margin": ensemble_margin_metrics,
        "park_margin": park_margin_metrics,
        "raw_score_nll": raw_score_nll,
        "park_score_nll": park_score_nll,
        "raw_total_rmse": raw_total_rmse,
        "park_total_rmse": park_total_rmse,
        "park_coverage": park_coverage,
        "raw_ml_passed": raw_ml_passed,
        "ensemble_ml_passed": ensemble_ml_passed,
        "raw_spread_passed": raw_spread_passed,
        "ensemble_spread_passed": ensemble_spread_passed,
        "raw_margin_passed": raw_margin_passed,
        "ensemble_margin_passed": ensemble_margin_passed,
        "park_passed": park_passed,
    }


def test_expected_runs_challenger(seasons, holdout_start=None):
    """Chronologically compare Pythagorean expected runs with production."""
    from analysis import _apply_shrink

    latest_season = max(seasons)
    holdout_start = holdout_start or f"{latest_season}-07-01"
    rows, _ = build_pooled_dataset(seasons)
    train = [row for row in rows
             if row["season"] < latest_season or row["date"] < holdout_start]
    holdout = [row for row in rows
               if row["season"] == latest_season
               and row["date"] >= holdout_start]
    if len(train) < 200 or len(holdout) < 200:
        print(f"Need at least 200 train and holdout games; found "
              f"{len(train)} and {len(holdout)}.")
        return None

    model = fit_expected_run_model(train)

    train_ml_prob = []
    train_spread_prob = []
    train_win = []
    train_cover = []
    for row in train:
        home_runs, away_runs = project_expected_runs(row, model)
        train_ml_prob.append(mlb_starters.pythagorean_win_probability(
            home_runs, away_runs))
        train_spread_prob.append(mlb_starters.poisson_margin_probability(
            home_runs, away_runs, -1.5))
        train_win.append(row["home_win"])
        train_cover.append(1 if row["margin"] > 1.5 else 0)
    ml_shrink = _fit_probability_shrink(train_ml_prob, train_win)
    spread_shrink = _fit_probability_shrink(
        train_spread_prob, train_cover)

    current = _current_margin_predictions(seasons, rows)
    comparison = [row for row in holdout if _game_key(row) in current]
    if len(comparison) < 200:
        print(f"Only {len(comparison)} holdout games have current-model history.")
        return None
    actual_win = [row["home_win"] for row in comparison]
    actual_margin = [row["margin"] for row in comparison]
    actual_cover = [1 if row["margin"] > 1.5 else 0 for row in comparison]

    null_win_probability = sum(train_win) / len(train_win)
    null_cover_probability = sum(train_cover) / len(train_cover)
    null_margin_prediction = sum(row["margin"] for row in train) / len(train)
    null_ml_metrics = _probability_metrics(
        [null_win_probability] * len(comparison), actual_win)
    null_spread_metrics = _probability_metrics(
        [null_cover_probability] * len(comparison), actual_cover)
    null_margin_metrics = _margin_metrics(
        [null_margin_prediction] * len(comparison), actual_margin)

    current_ml = []
    current_spread = []
    current_margins = []
    challenger_ml_raw = []
    challenger_ml = []
    challenger_spread = []
    challenger_margins = []
    for row in comparison:
        baseline = current[_game_key(row)]
        current_ml.append(_apply_shrink(
            baseline["win_probability"], "baseball_mlb", "moneyline"))
        current_spread.append(_apply_shrink(
            baseline["minus_1_5_probability"],
            "baseball_mlb", "spreads"))
        current_margins.append(baseline["margin"])

        home_runs, away_runs = project_expected_runs(row, model)
        raw_ml = mlb_starters.pythagorean_win_probability(
            home_runs, away_runs)
        raw_spread = mlb_starters.poisson_margin_probability(
            home_runs, away_runs, -1.5)
        challenger_ml_raw.append(raw_ml)
        challenger_ml.append(0.5 + ml_shrink * (raw_ml - 0.5))
        challenger_spread.append(
            0.5 + spread_shrink * (raw_spread - 0.5))
        challenger_margins.append(home_runs - away_runs)

    current_ml_metrics = _probability_metrics(current_ml, actual_win)
    challenger_raw_metrics = _probability_metrics(
        challenger_ml_raw, actual_win)
    challenger_ml_metrics = _probability_metrics(challenger_ml, actual_win)
    current_spread_metrics = _probability_metrics(
        current_spread, actual_cover)
    challenger_spread_metrics = _probability_metrics(
        challenger_spread, actual_cover)
    current_margin_metrics = _margin_metrics(current_margins, actual_margin)
    challenger_margin_metrics = _margin_metrics(
        challenger_margins, actual_margin)

    # Validation extensions. Every parameter below is fitted on `train`; only
    # the untouched chronological comparison window is used for these scores.
    base_variant = _challenger_variant_metrics(train, comparison, model)
    park_model = fit_expected_run_extensions(
        train, model, use_park=True)
    fatigue_model = fit_expected_run_extensions(
        train, model, use_fatigue=True)
    combined_model = fit_expected_run_extensions(
        train, model, use_park=True, use_fatigue=True)
    park_variant = _challenger_variant_metrics(
        train, comparison, park_model)
    fatigue_variant = _challenger_variant_metrics(
        train, comparison, fatigue_model)
    combined_variant = _challenger_variant_metrics(
        train, comparison, combined_model)
    dispersion, train_nb_nll = fit_negative_binomial_dispersion(train, model)
    nb_variant = _challenger_variant_metrics(
        train, comparison, model,
        distribution="negative_binomial", dispersion=dispersion)

    park_passed = (
        park_model["park_strength"] > 0
        and base_variant["score_nll"] - park_variant["score_nll"]
        >= MIN_HOLDOUT_NLL_GAIN
        and base_variant["total_rmse"] - park_variant["total_rmse"]
        >= MIN_HOLDOUT_RMSE_GAIN
        and park_variant["spread"]["brier"]
        <= base_variant["spread"]["brier"]
    )
    fatigue_passed = (
        fatigue_model["fatigue_weight"] > 0
        and base_variant["score_nll"] - fatigue_variant["score_nll"]
        >= MIN_HOLDOUT_NLL_GAIN
        and (
            base_variant["ml"]["brier"]
            - fatigue_variant["ml"]["brier"]
            >= MIN_HOLDOUT_PROBABILITY_GAIN
            or base_variant["spread"]["brier"]
            - fatigue_variant["spread"]["brier"]
            >= MIN_HOLDOUT_PROBABILITY_GAIN
        )
    )
    nb_passed = (
        dispersion > 0
        and base_variant["spread"]["brier"]
        - nb_variant["spread"]["brier"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
        and base_variant["spread"]["log_loss"]
        - nb_variant["spread"]["log_loss"]
        >= MIN_HOLDOUT_PROBABILITY_GAIN
    )

    passed = (
        challenger_ml_metrics["brier"]
        < min(current_ml_metrics["brier"], null_ml_metrics["brier"])
        and challenger_spread_metrics["brier"]
        < min(current_spread_metrics["brier"], null_spread_metrics["brier"])
        and challenger_margin_metrics["rmse"]
        < min(current_margin_metrics["rmse"], null_margin_metrics["rmse"])
    )
    print(f"\n=== expected-runs challenger — holdout {holdout_start}+ ===")
    print(f"train={len(train)} holdout={len(comparison)}; "
          f"Pythagorean exponent={mlb_starters.PYTHAGOREAN_EXPONENT}")
    print("fit: offense_weight={offense_weight:.2f}, "
          "pitching_weight={pitching_weight:.2f}, home_base={home_base_runs:.3f}, "
          "away_base={away_base_runs:.3f}".format(**model))
    print(f"probability shrink fitted on train: ML={ml_shrink:.2f}, "
          f"home -1.5={spread_shrink:.2f}")
    print("\nMONEYLINE (lower is better)")
    print(f"  base rate:  Brier={null_ml_metrics['brier']:.4f} "
          f"logloss={null_ml_metrics['log_loss']:.4f} "
          f"accuracy={null_ml_metrics['accuracy']:.2%}")
    print(f"  current:    Brier={current_ml_metrics['brier']:.4f} "
          f"logloss={current_ml_metrics['log_loss']:.4f} "
          f"accuracy={current_ml_metrics['accuracy']:.2%}")
    print(f"  Pyth raw:   Brier={challenger_raw_metrics['brier']:.4f} "
          f"logloss={challenger_raw_metrics['log_loss']:.4f}")
    print(f"  challenger: Brier={challenger_ml_metrics['brier']:.4f} "
          f"logloss={challenger_ml_metrics['log_loss']:.4f} "
          f"accuracy={challenger_ml_metrics['accuracy']:.2%}")
    print("\nMARGIN / RUN LINE")
    print(f"  base-rate margin:  MAE={null_margin_metrics['mae']:.3f} "
          f"RMSE={null_margin_metrics['rmse']:.3f}")
    print(f"  current margin:    MAE={current_margin_metrics['mae']:.3f} "
          f"RMSE={current_margin_metrics['rmse']:.3f}")
    print(f"  challenger margin: MAE={challenger_margin_metrics['mae']:.3f} "
          f"RMSE={challenger_margin_metrics['rmse']:.3f}")
    print(f"  current home -1.5 Brier:    "
          f"{current_spread_metrics['brier']:.4f}")
    print(f"  base-rate home -1.5 Brier:  "
          f"{null_spread_metrics['brier']:.4f}")
    print(f"  challenger home -1.5 Brier: "
          f"{challenger_spread_metrics['brier']:.4f}")
    print("\nOUT-OF-SAMPLE FEATURE / DISTRIBUTION VALIDATIONS")
    print("  variant          score NLL  total RMSE  margin RMSE  "
          "ML Brier  -1.5 Brier")

    def print_variant(label, metrics):
        print(f"  {label:<16} {metrics['score_nll']:.4f}     "
              f"{metrics['total_rmse']:.3f}       "
              f"{metrics['margin']['rmse']:.3f}       "
              f"{metrics['ml']['brier']:.4f}    "
              f"{metrics['spread']['brier']:.4f}")

    print_variant("base Poisson", base_variant)
    print_variant("+ park", park_variant)
    print_variant("+ BP workload", fatigue_variant)
    print_variant("+ park + BP", combined_variant)
    print_variant("negative binom", nb_variant)
    park_values = list(park_model["park_factors"].values())
    park_range = (
        f"{min(park_values):.3f}..{max(park_values):.3f}"
        if park_values else "n/a"
    )
    workload_coverage = sum(
        row.get("h_bp_workload") is not None
        and row.get("a_bp_workload") is not None
        for row in comparison
    )
    print(f"  park: train-only raw range={park_range}, "
          f"selected strength={park_model['park_strength']:.2f} — "
          f"{'PASS' if park_passed else 'NO INCREMENTAL PASS'}")
    print(f"  bullpen workload: {workload_coverage}/{len(comparison)} holdout games, "
          f"selected coefficient={fatigue_model['fatigue_weight']:+.2f} — "
          f"{'PASS' if fatigue_passed else 'NO INCREMENTAL PASS'}")
    print(f"  negative binomial: train dispersion={dispersion:.2f}, "
          f"train score NLL={train_nb_nll:.4f}, holdout -1.5 logloss "
          f"{base_variant['spread']['log_loss']:.4f} -> "
          f"{nb_variant['spread']['log_loss']:.4f} — "
          f"{'PASS' if nb_passed else 'NO RUN-LINE PASS'}")
    print("\nDECISION: " + (
        "BASE PASS for this holdout; extension gates are reported separately. "
        "No live settings changed."
        if passed else
        "BASE FAIL for this holdout; keep off. No live settings changed."
    ))
    return {
        "model": model,
        "train_n": len(train),
        "holdout_n": len(comparison),
        "ml_shrink": ml_shrink,
        "spread_shrink": spread_shrink,
        "null_ml": null_ml_metrics,
        "current_ml": current_ml_metrics,
        "challenger_ml": challenger_ml_metrics,
        "null_margin": null_margin_metrics,
        "current_margin": current_margin_metrics,
        "challenger_margin": challenger_margin_metrics,
        "null_spread": null_spread_metrics,
        "current_spread": current_spread_metrics,
        "challenger_spread": challenger_spread_metrics,
        "extensions": {
            "base": base_variant,
            "park": {
                "strength": park_model["park_strength"],
                "metrics": park_variant,
                "passed": park_passed,
            },
            "bullpen_workload": {
                "coefficient": fatigue_model["fatigue_weight"],
                "coverage": workload_coverage,
                "metrics": fatigue_variant,
                "passed": fatigue_passed,
            },
            "combined": combined_variant,
            "negative_binomial": {
                "dispersion": dispersion,
                "metrics": nb_variant,
                "passed": nb_passed,
            },
        },
        "all_extensions_passed": (
            park_passed and fatigue_passed and nb_passed),
        "passed": passed,
    }


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else 0.0


def test_innings_weighting(seasons):
    """A/B: does innings-weighting the starter (and letting the bullpen cover
    the rest) beat the current starter-quality-only signal for ML, spreads,
    and totals? Leakage-safe, no save — reports metrics only.

    Effective run-prevention per team:
        w = clamp(avg_innings/9, 0.30, 0.85)          # share the starter covers
        eff = w * starter_suppression + (1-w) * bullpen_suppression
    vs the current baseline (starter_suppression alone).
    """
    seasons_str = ",".join(str(s) for s in seasons)
    data, _ = build_pooled_dataset(seasons)
    rows = [d for d in data
            if d.get("h_bp_sup") and d.get("a_bp_sup")
            and d.get("h_ip") and d.get("a_ip") and d.get("margin") is not None]
    if len(rows) < 200:
        print(f"Only {len(rows)} rows with bullpen+IP+margin — need more cache.")
        return

    def w_ip(ip):
        return max(0.30, min(0.85, ip / 9.0))

    base_edge, eff_edge = [], []
    base_exc, eff_exc = [], []
    for d in rows:
        wh, wa = w_ip(d["h_ip"]), w_ip(d["a_ip"])
        eff_h = wh * d["h_sp_sup"] + (1 - wh) * d["h_bp_sup"]
        eff_a = wa * d["a_sp_sup"] + (1 - wa) * d["a_bp_sup"]
        base_edge.append(math.tanh(d["h_sp_sup"] - d["a_sp_sup"]))
        eff_edge.append(math.tanh(eff_h - eff_a))
        base_exc.append((d["h_sp_sup"] - 1) + (d["a_sp_sup"] - 1))
        eff_exc.append((eff_h - 1) + (eff_a - 1))
    win = [d["home_win"] for d in rows]
    mar = [d["margin"] for d in rows]
    tot = [d["total_runs"] for d in rows]
    bexc = [(d["h_bp_sup"] - 1) + (d["a_bp_sup"] - 1) for d in rows]

    print(f"\n=== innings-weighting A/B — {seasons_str} ({len(rows)} games) ===")
    print(f"avg innings/start: home {sum(d['h_ip'] for d in rows)/len(rows):.2f} "
          f"(range shows spread matters at the extremes)")

    # ── ML: home_win ~ edge (logistic, Brier) ──
    def brier_logit(x, y):
        b, a = fit_logistic_1d(x, y)
        return brier([_sigmoid(a + b * xi) for xi in x], y)
    ml_base = brier_logit(base_edge, win)
    ml_eff = brier_logit(eff_edge, win)
    print(f"\nMONEYLINE  Brier: starter-only {ml_base:.4f} -> innings-wtd {ml_eff:.4f} "
          f"({'BETTER' if ml_eff < ml_base else 'no gain'})")

    # ── Spreads: margin ~ edge (OLS RMSE + correlation with margin) ──
    def ols_rmse(x, y):
        s, i = fit_ols_1d(x, y)
        return rmse([i + s * xi for xi in x], y)
    sp_base, sp_eff = ols_rmse(base_edge, mar), ols_rmse(eff_edge, mar)
    print(f"SPREADS    margin RMSE: starter-only {sp_base:.3f} -> innings-wtd {sp_eff:.3f} "
          f"({'BETTER' if sp_eff < sp_base else 'no gain'})")
    print(f"           corr w/ margin: starter-only {_pearson(base_edge, mar):+.3f} "
          f"-> innings-wtd {_pearson(eff_edge, mar):+.3f}")

    # ── Totals: starter-only vs starter+bullpen additive vs innings-weighted ──
    s0, i0 = fit_ols_1d(base_exc, tot)
    pred_s = [i0 + s0 * x for x in base_exc]
    tot_starter = rmse(pred_s, tot)
    resid = [t - p for t, p in zip(tot, pred_s)]
    sb, _ = fit_ols_1d(bexc, resid)
    tot_add = rmse([p + sb * b for p, b in zip(pred_s, bexc)], tot)
    tot_eff = ols_rmse(eff_exc, tot)
    print(f"TOTALS     RMSE: starter-only {tot_starter:.3f} | "
          f"starter+bullpen(current) {tot_add:.3f} | innings-wtd {tot_eff:.3f}")
    best = min(tot_starter, tot_add, tot_eff)
    tag = ("innings-wtd" if best == tot_eff else
           "starter+bullpen" if best == tot_add else "starter-only")
    print(f"           best = {tag}")


def _parse_seasons(spec):
    """Parse '2024', '2021,2022', or '2021-2024' into a sorted list of ints."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def _offense_index(rows):
    """As-of index of the xwOBAcon a team's hitters PRODUCED, keyed by
    'team|opposingPitcherHand', for the two-sided (pitcher-vs-lineup) edge."""
    from backtest_props import AsOfIndex
    idx = AsOfIndex()
    for r in rows:
        x = r.get("xwoba")
        if x is None:
            continue
        bt, ph = r.get("batting_team"), r.get("p_throws")
        if not bt or not ph:
            continue
        idx.add(f"{bt}|{ph}", r["game_date"], x)
    return idx


def test_two_sided(seasons):
    """A/B: does folding the OPPOSING lineup's as-of xwOBAcon-vs-hand into the
    starter edge (pitcher-vs-lineup, both directions) beat the current
    pitcher-only edge at predicting the game margin? Leakage-safe, no save.

    Two-sided effective prevention per team = starter_suppression divided by the
    offense factor of the lineup that starter faces (offense factor >1 = the
    opposing hitters are better than league vs that hand, so prevention drops).
    """
    base_edges, two_edges, margins = [], [], []
    for s in seasons:
        games = get_season_games(s)
        s0, s1 = _season_bounds(s)
        rows = sh.load_days(s0, s1)
        if not rows:
            continue
        vals = [r["xwoba"] for r in rows if r["xwoba"] is not None]
        league = sum(vals) / len(vals)
        sp_idx = _pitcher_xwoba_index(rows)
        off_idx = _offense_index(rows)
        _, _, game_teams = _bullpen_features(rows, s)
        hands = {}
        for r in rows:
            p, ph = str(r.get("pitcher")), r.get("p_throws")
            if p and ph and p not in hands:
                hands[p] = ph
        for g in games:
            if g.get("margin") is None:
                continue
            hsp, asp = str(g["home_sp"]), str(g["away_sp"])
            hx = sp_idx.asof_mean(hsp, g["date"])
            ax = sp_idx.asof_mean(asp, g["date"])
            if hx is None or ax is None:
                continue
            gt = game_teams.get((g["date"], hsp, asp))
            hh, ah = hands.get(hsp), hands.get(asp)
            if not gt or not hh or not ah:
                continue
            home_abbr, away_abbr = gt
            off_away = off_idx.asof_mean(f"{away_abbr}|{hh}", g["date"])
            off_home = off_idx.asof_mean(f"{home_abbr}|{ah}", g["date"])
            if off_away is None or off_home is None:
                continue
            h_rs = max(0.5, min(2.0, league / hx))
            a_rs = max(0.5, min(2.0, league / ax))
            off_a = off_away / league   # away lineup offense factor
            off_h = off_home / league   # home lineup offense factor
            base_edges.append(math.tanh(h_rs - a_rs))
            two_edges.append(math.tanh((h_rs / off_a) - (a_rs / off_h)))
            margins.append(g["margin"])
    n = len(margins)
    print(f"\n=== two-sided (pitcher-vs-lineup) edge A/B — {','.join(map(str, seasons))} "
          f"({n} games w/ offense data) ===")
    if n < 200:
        print("Not enough games with as-of offense data to judge.")
        return
    cb, ct = _pearson(base_edges, margins), _pearson(two_edges, margins)
    print(f"corr w/ margin:  pitcher-only {cb:+.4f}  ->  two-sided {ct:+.4f}"
          f"   (delta {ct - cb:+.4f})")
    print("Higher |corr| = the edge tracks real margins better. A meaningful "
          "gain justifies wiring the lineup side into fit + live; a flat/negative "
          "delta means keep the pitcher-only edge (like the props matchup).")


# ──────────────────────────────────────────────────────────────────────────────
# Additive (xERA-lite) bake-off: additive Savant runs model vs the multiplicative
# incumbent, graded by IDENTICAL code on a chronological holdout. (Tier A #1b)
# ──────────────────────────────────────────────────────────────────────────────

_ADDITIVE_FEATURES = ("xwobacon", "k9")   # v1 (owner-endorsed), available now

# Feature families for the bake-off, in increasing richness. The bake-off runs each
# so the MARGINAL value of adding the contact-quality bundle, then the walk-rate
# (FIP) bundle, then CSW, is visible (the owner-endorsed marginal-then-joint read).
#   v1      — xwOBAcon + K/9 (warehouse + statcast, populated now).
#   contact — + barrel% + whiff% (statcast rates, ALSO populated now: no schema wait).
#   fip     — + K% + BB% (need the #1c-a BB/BF unlock + re-backfill; NULL until then,
#             so this family is auto-SKIPPED with a note on pre-unlock data).
#   csw     — contact + CSW% (called-strikes+whiffs / pitches): a compact swing-and-miss
#             / pitcher-dominance summary already computed + stored in pitcher_asof_daily
#             (Batch A #25 "activate inert CSW"). Built on CONTACT, not fip, on purpose:
#             fip's K%/BB% are NULL until the #1c-a BB/BF re-backfill, and feat_from_row
#             drops a row on ANY null key — so a fip-based csw would auto-skip until that
#             backfill. contact is populated now, so csw grades on CURRENT data. Its
#             MARGINAL over contact is what the bake-off measures; CSW correlates with
#             whiff% so a near-zero marginal is the expected null. INERT until a
#             csw-bearing additive config is fit + promoted (candidate staging) — live
#             pricing reads the fitted config's feature_keys, never these lists, so this
#             addition is byte-identical.
#   gb      — contact + GB% (ground-ball rate): the SIERA ingredient that IS populated
#             now (Batch A SIERA). Grades on current data; isolates the ground-ball
#             marginal (grounders suppress XBH/HR) before the walk-rate unlock.
#   siera   — fip + GB% = the full SIERA skill set (K%, BB%, GB% + contact) fed linearly
#             into xERA-lite (true nonlinear SIERA interactions deferred). Like fip it
#             needs k_pct/bb_pct, so it AUTO-SKIPS until the #1c-a BB/BF re-backfill +
#             a pitcher_asof rebuild; then it grades.
_ADDITIVE_FEATURE_SETS = {
    "v1": ("xwobacon", "k9"),
    "contact": ("xwobacon", "k9", "barrel_pct", "whiff_pct"),
    "fip": ("xwobacon", "k9", "barrel_pct", "whiff_pct", "k_pct", "bb_pct"),
    "csw": ("xwobacon", "k9", "barrel_pct", "whiff_pct", "csw_pct"),
    "gb": ("xwobacon", "k9", "barrel_pct", "whiff_pct", "gb_pct"),
    "siera": ("xwobacon", "k9", "barrel_pct", "whiff_pct", "k_pct", "bb_pct",
              "gb_pct"),
}

# Every feature any family may request — the series loader pulls all of these so a
# family can be selected without re-querying. MUST stay equal (as a set) to
# pitcher_asof._SERIES_FEATURES or the live single-entity series omits a column a
# family needs -> fit != serve (guarded by test_additive_bakeoff).
_ALL_ASOF_FEATURES = ("xwobacon", "k9", "barrel_pct", "whiff_pct", "hard_hit_pct",
                      "gb_pct", "k_pct", "bb_pct", "csw_pct")


def _variant_metrics_projfn(train, holdout, project_fn,
                            distribution="poisson", dispersion=0.0):
    """Grade a model given ONLY a project_fn(row) -> (home_runs, away_runs), so the
    additive and multiplicative projectors are scored by identical code (fair
    comparison). Mirrors _challenger_variant_metrics but projector-agnostic and with
    the Poisson/NegBin NLL computed inline (no _score_nll(model) dependency)."""
    prob_fn = (mlb_starters.negative_binomial_margin_probability
               if distribution == "negative_binomial"
               else mlb_starters.poisson_margin_probability)

    def _spread_p(hr, ar):
        return (prob_fn(hr, ar, -1.5, dispersion)
                if distribution == "negative_binomial" else prob_fn(hr, ar, -1.5))

    tr_ml, tr_spread, tr_win, tr_cover = [], [], [], []
    for row in train:
        hr, ar = project_fn(row)
        tr_ml.append(mlb_starters.pythagorean_win_probability(hr, ar))
        tr_spread.append(_spread_p(hr, ar))
        tr_win.append(row["home_win"])
        tr_cover.append(1 if row["margin"] > 1.5 else 0)
    ml_shrink = _fit_probability_shrink(tr_ml, tr_win)
    spread_shrink = _fit_probability_shrink(tr_spread, tr_cover)

    ho_ml, ho_spread, exp_margin, exp_total = [], [], [], []
    a_win, a_cover, a_margin, a_total, nll = [], [], [], [], []
    for row in holdout:
        hr, ar = project_fn(row)
        ho_ml.append(0.5 + ml_shrink * (
            mlb_starters.pythagorean_win_probability(hr, ar) - 0.5))
        ho_spread.append(0.5 + spread_shrink * (_spread_p(hr, ar) - 0.5))
        exp_margin.append(hr - ar)
        exp_total.append(hr + ar)
        a_win.append(row["home_win"])
        a_cover.append(1 if row["margin"] > 1.5 else 0)
        a_margin.append(row["margin"])
        a_total.append(row["total_runs"])
        for actual, expected in ((row["home_runs"], hr), (row["away_runs"], ar)):
            nll.append(_negative_binomial_score_nll(actual, expected, dispersion)
                       if distribution == "negative_binomial"
                       else _poisson_score_nll(actual, expected))
    return {
        "ml": _probability_metrics(ho_ml, a_win),
        "spread": _probability_metrics(ho_spread, a_cover),
        "margin": _margin_metrics(exp_margin, a_margin),
        "total_rmse": rmse(exp_total, a_total),
        "score_nll": sum(nll) / len(nll) if nll else float("nan"),
    }


def _sp_feats(asof, sp_id, date, feature_keys):
    """(feature dict, n_bbe) from a FLAT {(entity,date): row} as-of dict."""
    r = asof.get((str(sp_id), str(date)[:10]))
    if not r:
        return None, None
    return {k: r.get(k) for k in feature_keys}, r.get("n_bbe")


def _dict_feat_getter(asof, feature_keys):
    """A cumulative feature getter over a flat {(entity,date): row} dict."""
    return lambda pid, date: _sp_feats(asof, pid, date, feature_keys)


# Extracted to additive_runs.py so the OFFLINE bake-off + the LIVE path (#1d) share
# ONE implementation (fit == serve by construction). These aliases keep the bake-off's
# internal call sites + tests unchanged.
_exp_ip = ar.exp_ip


def _additive_training_rows(rows, feat_getter, feature_keys):
    """Per starter-game training rows: features -> label = the OPPONENT's actual
    runs that game (runs the starter's team allowed). feat_getter(pid, date) ->
    (feats, n). Rows missing any feature are dropped."""
    train = []
    for r in rows:
        if r.get("home_runs") is None or r.get("away_runs") is None:
            continue
        hf, _ = feat_getter(r["home_sp"], r["date"])
        af, _ = feat_getter(r["away_sp"], r["date"])
        if hf and all(hf.get(k) is not None for k in feature_keys):
            train.append({**{k: hf[k] for k in feature_keys},
                          "label": r["away_runs"]})   # home SP -> allowed away_runs
        if af and all(af.get(k) is not None for k in feature_keys):
            train.append({**{k: af[k] for k in feature_keys},
                          "label": r["home_runs"]})   # away SP -> allowed home_runs
    return train


_make_additive_projector = ar.make_additive_projector   # extracted (#1d shared spine)
_make_run_env_fn = ar.make_run_env_fn                   # Batch A park/weather run_env


# ── windowing / prior-blend as-of feature getters (recency vs cumulative) ─────

def _load_pitcher_asof_series(seasons):
    """{entity_id: [rows sorted by as_of_date]} spanning `seasons` + the season
    BEFORE the earliest (so the current+prior blend has a prior). Each row carries
    as_of_date, season_bucket, n_bbe, ip, and every feature in _ALL_ASOF_FEATURES
    (v1 + contact + FIP). Enables windowed (difference two rows) + blended
    (prior-season final) as-of features across feature families. {} on error."""
    try:
        import db_store
        import pitcher_asof
        from sqlalchemy import select
        t = pitcher_asof.pitcher_asof_daily
        want = {int(s) for s in seasons}
        want |= {min(want) - 1}                       # prior season for the blend
        cols = [t.c.entity_id, t.c.as_of_date, t.c.season_bucket, t.c.n_bbe, t.c.ip]
        cols += [t.c[k] for k in _ALL_ASOF_FEATURES]
        by_pid = {}
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(*cols).where((t.c.season_bucket.in_(sorted(want)))
                                    & (t.c.role == "SP"))).mappings().all()
        for r in rows:
            rec = {"as_of_date": str(r["as_of_date"])[:10],
                   "season_bucket": r["season_bucket"],
                   "n_bbe": r["n_bbe"], "ip": r["ip"]}
            rec.update({k: r[k] for k in _ALL_ASOF_FEATURES})
            by_pid.setdefault(str(r["entity_id"]), []).append(rec)
        for lst in by_pid.values():
            lst.sort(key=lambda r: r["as_of_date"])
        return by_pid
    except Exception:
        return {}


def _load_bullpen_asof_series(seasons):
    """({team_id: [rows sorted by as_of_date]}, league_rp_era) for role='RP' team
    bullpen aggregates — the GS==0 relief as-of curve materialized by
    pitcher_asof.build_season (#1c-b). Each row: as_of_date, era. league_rp_era is
    the innings-weighted league mean, used to make the per-team term league-relative
    (so the earned-vs-total scale cancels). ({}, None) on error / empty."""
    try:
        import db_store
        import pitcher_asof
        from sqlalchemy import select
        t = pitcher_asof.pitcher_asof_daily
        want = {int(s) for s in seasons}
        want |= {min(want) - 1}
        by_tid = {}
        num = den = 0.0
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(t.c.entity_id, t.c.as_of_date, t.c.era, t.c.ip,
                       t.c.season_bucket)
                .where((t.c.season_bucket.in_(sorted(want)))
                       & (t.c.role == "RP"))).all()
        for eid, d, era, ip, sb in rows:
            if era is None:
                continue
            # Carry ip + season_bucket per-row (cumulative relief IP strictly-before) so
            # make_bp_getter can difference ip WITHIN a season for the trailing-workload
            # fatigue term (Batch A #13); cumulative ip resets per season.
            by_tid.setdefault(str(eid), []).append(
                {"as_of_date": str(d)[:10], "era": float(era),
                 "ip": (float(ip) if ip is not None else None),
                 "season_bucket": sb})
            if ip:                                    # innings-weighted league mean
                num += float(era) * float(ip)
                den += float(ip)
        for lst in by_tid.values():
            lst.sort(key=lambda r: r["as_of_date"])
        league_rp_era = (num / den) if den else None
        return by_tid, league_rp_era
    except Exception:
        return {}, None


_make_bp_getter = ar.make_bp_getter   # extracted (#1d); 2nd arg is now a resolve_id
                                      # CALLABLE — the bake-off call site passes .get


_feat_from_row = ar.feat_from_row       # extracted (#1d shared spine)
_window_diff = ar.window_diff
_make_feat_getter = ar.make_feat_getter


def _bakeoff_row(label, m):
    return (f"  {label:22} margin_rmse {m['margin']['rmse']:.3f}  "
            f"ML_brier {m['ml']['brier']:.4f}  "
            f"spread_brier {m['spread']['brier']:.4f}  "
            f"total_rmse {m['total_rmse']:.3f}  "
            f"pois_nll {m['score_nll']:.4f}")


def _abbr_to_team_id(seasons):
    """{team_abbr: MLBAM team_id} across `seasons`, inverting backtest_props._team_maps
    (id->abbr). Bridges the abbr-keyed game rows to the team_id-keyed RP series. {} on
    error (bp_getter then falls back to the flat league bullpen)."""
    out = {}
    try:
        from backtest_props import _team_maps
        for s in seasons:
            id_abbr, _ = _team_maps(s)
            for tid, abbr in id_abbr.items():
                if abbr and tid is not None:
                    out[abbr] = str(tid)
    except Exception:
        return {}
    return out


def test_additive_expected_runs(seasons, holdout_start=None,
                                feature_sets=None,
                                window_modes=("cumulative", "blend", "window"),
                                rp_bullpen=False, bullpen_fatigue_weight=0.0,
                                park_weight=0.0):
    """Bake-off: the multiplicative incumbent vs the additive Savant xERA-lite runs
    model across FEATURE FAMILIES (v1/contact/fip/csw/gb/siera) x as-of WINDOW modes
    (cumulative / prior-season blend / trailing window), fit on a chronological train
    split and graded by identical code on the holdout. rp_bullpen=True swaps the flat
    league bullpen term for the team's GS-based as-of RP aggregate (#1c-b, league-
    relative). bullpen_fatigue_weight>0 layers the trailing-workload fatigue term on
    that RP aggregate (Batch A #13; needs rp_bullpen). Reads pitcher_asof_daily.
    Families needing NULL columns (fip before the #1c-a re-backfill) auto-skip with a
    note."""
    feature_sets = feature_sets or _ADDITIVE_FEATURE_SETS
    all_rows = []
    for s in seasons:
        games = get_season_games(s)
        rows, _lg = build_dataset(games, s)
        # get_season_games stores margin + total_runs (not per-team runs); derive
        # home_runs/away_runs (margin = home-away, total = home+away).
        for r in rows:
            if (r.get("home_runs") is None and r.get("margin") is not None
                    and r.get("total_runs") is not None):
                r["home_runs"] = (r["total_runs"] + r["margin"]) / 2.0
                r["away_runs"] = (r["total_runs"] - r["margin"]) / 2.0
        all_rows.extend(rows)
    if not all_rows:
        print("No dataset rows (need Statcast days cached / warehouse games).")
        return None
    series = _load_pitcher_asof_series(seasons)
    if not series:
        print("!! pitcher_asof_daily is EMPTY — run `python pitcher_asof.py "
              "--build` first. Aborting additive bake-off.")
        return None
    # Batch A PARK run_env: attach the ACTUAL venue_id per game (mlb_game
    # (official_date, home_team_id) -> venue_id, so neutral-site games key their real
    # park) and build the venue-keyed run_env_fn. INERT at park_weight=0 (run_env_fn
    # stays None -> projector byte-identical).
    run_env_fn = None
    if park_weight:
        import mlb_warehouse
        venue_idx = mlb_warehouse.game_venue_index(seasons)
        abbr_id = _abbr_to_team_id(seasons)
        for r in all_rows:
            tid = abbr_id.get(r.get("home_abbr"))
            r["venue_id"] = (venue_idx.get((str(r.get("date"))[:10], tid))
                             if tid else None)
        park_runs = mlb_warehouse.venue_park_runs_map()
        run_env_fn = _make_run_env_fn(
            park_runs_of=lambda row: park_runs.get(row.get("venue_id"), 1.0),
            park_weight=park_weight)
        n_res = sum(1 for r in all_rows if r.get("venue_id"))
        print(f"  [park] run_env park_weight={park_weight}: {n_res}/{len(all_rows)} "
              f"games venue-resolved, {len(park_runs)} venues in mlb_venue.")

    all_rows.sort(key=lambda r: r["date"])
    if holdout_start is None:
        cut = int(len(all_rows) * 0.7)
        train, holdout = all_rows[:cut], all_rows[cut:]
    else:
        train = [r for r in all_rows if r["date"] < holdout_start]
        holdout = [r for r in all_rows if r["date"] >= holdout_start]
    if not train or not holdout:
        print("Empty train or holdout split.")
        return None

    # Optional GS-based RP bullpen term (league-relative so the earned/total scale
    # cancels). None -> flat league bullpen (v1 behavior).
    if bullpen_fatigue_weight and not rp_bullpen:
        print("  !! --bullpen-fatigue-weight has NO effect without --rp-bullpen "
              "(the flat league bullpen has no per-team as-of workload). Ignoring.")
    bp_getter = None
    if rp_bullpen:
        bp_series, league_rp_era = _load_bullpen_asof_series(seasons)
        if not bp_series or not league_rp_era:
            print("  !! --rp-bullpen: no role='RP' rows in pitcher_asof_daily "
                  "(needs the #1c-a GS re-backfill + rebuild). Using league bullpen.")
        else:
            abbr_to_id = _abbr_to_team_id(seasons)
            # league_bp is the additive fit's league_rate9; resolved per-family below.
            bp_getter = ("pending", bp_series, abbr_to_id, league_rp_era)

    mult_model = fit_expected_run_model(train)
    if mult_model is None:
        print("Could not fit the multiplicative incumbent.")
        return None
    results = {"multiplicative": _variant_metrics_projfn(
        train, holdout, lambda r: project_expected_runs(r, mult_model))}

    labels = []                                  # preserve family/mode print order
    for fs_name, feature_keys in feature_sets.items():
        for mode in window_modes:
            getter = _make_feat_getter(series, mode, feature_keys)
            xm = xera_lite.fit(
                _additive_training_rows(train, getter, feature_keys),
                list(feature_keys))
            if xm is None:
                print(f"  additive[{fs_name}/{mode}]: no fit (feature columns NULL "
                      f"until re-backfill, or too few rows).")
                continue
            league_bp = xm.get("league_rate9", 4.3)
            bpg = None
            if bp_getter:                        # bind league_bp now that it's known
                _, bp_series, abbr_to_id, league_rp_era = bp_getter
                bpg = _make_bp_getter(bp_series, abbr_to_id.get,   # resolve_id callable
                                      league_rp_era, league_bp,
                                      fatigue_weight=bullpen_fatigue_weight)
            key = f"additive_{fs_name}_{mode}"
            results[key] = _variant_metrics_projfn(
                train, holdout,
                _make_additive_projector(getter, xm, league_bp, feature_keys, bpg,
                                         run_env_fn=run_env_fn))
            labels.append((f"B additive[{fs_name}/{mode}]", key))

    bp_tag = "team-RP" if bp_getter else "league-avg"
    if bp_getter and bullpen_fatigue_weight:
        bp_tag += f"+fatigue(w={bullpen_fatigue_weight})"
    print(f"\n=== Additive (xERA-lite) bake-off — train {len(train)} / holdout "
          f"{len(holdout)} — bullpen {bp_tag} ===")
    print(_bakeoff_row("A multiplicative", results["multiplicative"]))
    for label, key in labels:
        print(_bakeoff_row(label, results[key]))
    metric_of = {
        "margin_rmse": lambda m: m["margin"]["rmse"],
        "ML_brier": lambda m: m["ml"]["brier"],
        "spread_brier": lambda m: m["spread"]["brier"],
        "total_rmse": lambda m: m["total_rmse"],
        "pois_nll": lambda m: m["score_nll"]}
    best = {mk: min(results, key=lambda lbl: fn(results[lbl]))
            for mk, fn in metric_of.items()}
    print(f"  best per metric (lower=better): {best}")
    if not rp_bullpen:
        print("  NOTE: bullpen = flat league-avg. Re-run with --rp-bullpen for the "
              "GS-based team RP term (needs the #1c-a re-backfill).")
    return results


def save_additive_model(seasons, feature_keys=("xwobacon", "k9"),
                        mode="blend", blend_k=200.0, n_starts=10,
                        bullpen_fatigue_weight=0.0, park_weight=0.0,
                        weather_weight=0.0):
    """Fit the additive expected-runs model on the FULL span and STAGE it as the
    calibration candidate block `expected_runs_additive` (Tier A #1d, commit 5).
    Default = the bake-off winner: v1 features (xwOBAcon+K9) + prior-season BLEND +
    team-RP bullpen (league-relative). Uses the SAME helpers the bake-off + live path
    use, so the persisted model reproduces live. CANDIDATE-ONLY (forces
    set_candidate_mode(True)); never writes the live file — owner promotes via
    refit_calibration.py --promote. Returns the staged block, or None."""
    import calibration_loader as _cl
    all_rows = []
    for s in seasons:
        games = get_season_games(s)
        rows, _lg = build_dataset(games, s)
        for r in rows:
            if (r.get("home_runs") is None and r.get("margin") is not None
                    and r.get("total_runs") is not None):
                r["home_runs"] = (r["total_runs"] + r["margin"]) / 2.0
                r["away_runs"] = (r["total_runs"] - r["margin"]) / 2.0
        all_rows.extend(rows)
    series = _load_pitcher_asof_series(seasons)
    if not all_rows or not series:
        print("  [additive-save] no rows / empty pitcher_asof_daily — aborting.")
        return None
    getter = _make_feat_getter(series, mode, feature_keys, n_starts=n_starts,
                               blend_k=blend_k)
    xm = xera_lite.fit(_additive_training_rows(all_rows, getter, feature_keys),
                       list(feature_keys))
    if xm is None:
        print("  [additive-save] xera_lite fit failed (too few rows / NULL features).")
        return None
    _bp_series, league_rp_era = _load_bullpen_asof_series(seasons)
    block = {
        "enabled": True,
        "feature_keys": list(feature_keys),
        "model": xm,
        "blend": {"mode": mode, "blend_k": blend_k, "n_starts": n_starts},
        "bullpen": {"league_rp_era": league_rp_era,
                    "league_bp": xm.get("league_rate9"),
                    # Batch A #13; 0.0 -> inert (live make_bp_getter byte-identical).
                    "fatigue_weight": bullpen_fatigue_weight},
        # Batch A run_env (park/weather); 0.0 weights -> inert (byte-identical live).
        "run_env": {"park_weight": park_weight, "weather_weight": weather_weight},
    }
    _cl.set_candidate_mode(True)                    # candidate-ONLY, never live
    _cl.save_expected_runs_additive(
        "baseball_mlb", block,
        meta={"seasons": list(seasons), "n_train": len(all_rows)})
    print(f"  [additive-save] STAGED expected_runs_additive candidate — "
          f"features={list(feature_keys)} mode={mode} n={xm.get('n')} "
          f"league_rate9={xm.get('league_rate9')} league_rp_era={league_rp_era}. "
          f"Review: refit_calibration.py --sport mlb --diff ; promote: --promote. "
          f"Activate live: set ODI_MLB_ADDITIVE_RUNS=1.")
    return block


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    # Promote SQL_* secrets so load_days / pitcher_asof read the Azure warehouse
    # outside Streamlit (mirrors warehouse.py / refit_calibration mains). Guarded.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True,
                    help="single '2024', list '2021,2022', or range '2021-2024'")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch+cache each season's Statcast days first (slow)")
    ap.add_argument("--save", action="store_true", help="write fitted weights "
                    "(stages a candidate; promote via refit_calibration.py "
                    "--sport mlb --promote)")
    ap.add_argument("--live", action="store_true",
                    help="with --save, write the LIVE calibration file directly "
                         "(skip candidate staging)")
    ap.add_argument("--test-ip", action="store_true",
                    help="A/B test innings-weighted starter/bullpen blend "
                         "(ML, spreads, totals); reports metrics, no save")
    ap.add_argument("--test-2sided", action="store_true",
                    help="A/B test the two-sided (pitcher-vs-lineup) edge vs "
                         "the pitcher-only edge; reports metrics, no save")
    ap.add_argument("--test-runs", action="store_true",
                    help="chronologically test expected runs + Pythagorean 1.83 "
                         "against the current MLB moneyline/spread engine")
    ap.add_argument("--test-final", action="store_true",
                    help="test raw Pythagorean, current/Pythagorean ensemble, "
                         "and prior-season actual-venue park factors against "
                         "the latest season's untouched chronological holdout")
    ap.add_argument("--additive-bakeoff", action="store_true",
                    help="Tier A #1b: additive Savant xERA-lite runs model vs the "
                         "multiplicative incumbent on a chronological holdout "
                         "(reads pitcher_asof_daily; no save).")
    ap.add_argument("--holdout-start", default=None,
                    help="YYYY-MM-DD chronological holdout cutoff for "
                         "--additive-bakeoff (default: last 30%% of games).")
    ap.add_argument("--window-modes", default="cumulative,blend,window",
                    help="comma list of as-of feature modes for --additive-bakeoff "
                         "(cumulative|blend|window).")
    ap.add_argument("--feature-sets", default=None,
                    help="comma list of additive feature families for "
                         "--additive-bakeoff (v1|contact|fip|csw|gb|siera). "
                         "Default: all (fip/siera auto-skip pre BB/BF re-backfill).")
    ap.add_argument("--rp-bullpen", action="store_true",
                    help="with --additive-bakeoff, use the GS-based team RP as-of "
                         "bullpen term (league-relative) instead of the flat league "
                         "average (needs the #1c-a re-backfill to populate GS).")
    ap.add_argument("--bullpen-fatigue-weight", type=float, default=0.0,
                    help="with --additive-bakeoff --rp-bullpen, A/B the trailing-"
                         "workload bullpen fatigue term (Batch A #13) at this weight "
                         "(0 = off/current behavior; try e.g. 0.3).")
    ap.add_argument("--park-weight", type=float, default=0.0,
                    help="with --additive-bakeoff, A/B the venue park run_env term "
                         "(Batch A) at this weight (0 = off; 1 = full mlb_venue park "
                         "runs factor). Needs `--build-venues` populated first.")
    ap.add_argument("--additive-save", action="store_true",
                    help="Tier A #1d: fit the additive expected-runs model (v1/blend/"
                         "team-RP) on --season and STAGE it as the calibration "
                         "candidate expected_runs_additive (never live). Promote via "
                         "refit_calibration.py --promote; activate with "
                         "ODI_MLB_ADDITIVE_RUNS=1.")
    args = ap.parse_args()

    # Default-safe: --save stages a candidate unless --live is given.
    staging = not args.live
    set_candidate_mode(staging)
    if staging and args.save:
        _n = existing_candidate_notice("baseball_mlb")
        if _n:
            print(_n)

    seasons = _parse_seasons(args.season)
    if args.fetch:
        for s in seasons:
            s0, s1 = _season_bounds(s)
            print(f"fetching Statcast {s0}..{s1} (one-time, slow)...")
            sh.fetch_range(s0, s1)
            get_season_games(s)
    if args.test_ip:
        test_innings_weighting(seasons)
    elif args.test_2sided:
        test_two_sided(seasons)
    elif args.test_final:
        test_final_expected_run_candidates(seasons)
    elif args.additive_bakeoff:
        _modes = tuple(m.strip() for m in args.window_modes.split(",") if m.strip())
        _fs = None
        if args.feature_sets:
            _fs = {n.strip(): _ADDITIVE_FEATURE_SETS[n.strip()]
                   for n in args.feature_sets.split(",")
                   if n.strip() in _ADDITIVE_FEATURE_SETS}
        test_additive_expected_runs(seasons, holdout_start=args.holdout_start,
                                    feature_sets=_fs, window_modes=_modes,
                                    rp_bullpen=args.rp_bullpen,
                                    bullpen_fatigue_weight=args.bullpen_fatigue_weight,
                                    park_weight=args.park_weight)
    elif args.additive_save:
        save_additive_model(seasons,
                            bullpen_fatigue_weight=args.bullpen_fatigue_weight)
    elif args.test_runs:
        test_expected_runs_challenger(seasons)
    else:
        fit(seasons, do_save=args.save)
        if args.save and staging and has_candidate("baseball_mlb"):
            print("\n⇢ Staged to calibration/baseball_mlb.candidate.json — live "
                  "file UNTOUCHED. Promote: python refit_calibration.py "
                  "--sport mlb --promote (review with --diff).")
