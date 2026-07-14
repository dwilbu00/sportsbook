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
    python backtest_starters.py --season 2024 --test-runs  # expected-runs challenger

Caveats (documented, not hidden):
  * Uses the *probable* starter from the schedule (≈ actual; late scratches
    are rare but unmodeled).
  * As-of pitcher quality uses xwOBAcon (contact), a slightly different measure
    than the live season xERA index; both are Savant x-stats and correlated.
  * The isolated logistic coefficient can overlap with signal team form already
    captures, so treat the fitted moneyline weight as a mild upper bound and
    keep it bounded.
"""

import argparse
from collections import defaultdict
import json
import math
import os

import mlb_starters
import savant_history as sh
from calibration_loader import load_starter_adjustment, save_starter_adjustment

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


def _enrich_games(games, season):
    """Attach team identities and separate run outcomes to cached games."""
    from backtest_props import season_schedule

    identities = {}
    for date, scheduled in season_schedule(season).items():
        for game in scheduled:
            key = (date, str(game.get("home_sp")), str(game.get("away_sp")))
            identities[key] = (game.get("home_abbr"), game.get("away_abbr"))

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
        if gt and league_bp:
            ha, aa = gt
            h_bp = bp_idx.asof_mean(ha, g["date"])
            a_bp = bp_idx.asof_mean(aa, g["date"])
            if h_bp and a_bp:
                h_bs = max(0.5, min(2.0, league_bp / h_bp))
                a_bs = max(0.5, min(2.0, league_bp / a_bp))
                bullpen_excess = (h_bs - 1.0) + (a_bs - 1.0)

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


def build_pooled_dataset(seasons):
    """Pool leakage-safe features across multiple seasons.

    Each season is normalized against its OWN league xwOBAcon baseline (so
    era-to-era run-environment differences don't distort the league-relative
    run_suppression feature), then the resulting rows are concatenated.
    """
    pooled = []
    per_season = {}
    for s in seasons:
        games = _enrich_games(get_season_games(s), s)
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
    return home_runs, away_runs


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
    print("\nDECISION: " + (
        "PASS challenger gate; validate additional seasons before live use."
        if passed else
        "KEEP OFF; challenger did not beat current ML, margin, and spread gates."
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True,
                    help="single '2024', list '2021,2022', or range '2021-2024'")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch+cache each season's Statcast days first (slow)")
    ap.add_argument("--save", action="store_true", help="write fitted weights")
    ap.add_argument("--test-ip", action="store_true",
                    help="A/B test innings-weighted starter/bullpen blend "
                         "(ML, spreads, totals); reports metrics, no save")
    ap.add_argument("--test-2sided", action="store_true",
                    help="A/B test the two-sided (pitcher-vs-lineup) edge vs "
                         "the pitcher-only edge; reports metrics, no save")
    ap.add_argument("--test-runs", action="store_true",
                    help="chronologically test expected runs + Pythagorean 1.83 "
                         "against the current MLB moneyline/spread engine")
    args = ap.parse_args()

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
    elif args.test_runs:
        test_expected_runs_challenger(seasons)
    else:
        fit(seasons, do_save=args.save)
