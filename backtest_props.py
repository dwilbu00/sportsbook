"""
Fit / validate the MLB player-prop matchup weight (`starter_adjustment.props`)
from HISTORICAL outcomes — leakage-safe, no Odds API credits.

This is the v2 companion to backtest_starters.py. Where that tool fits the
TEAM-market starter weights (moneyline, run_scale), this one answers the prop
question the conflict-audit deferred:

    "Does the opposing starter / opposing lineup matchup signal actually improve
     player-prop projections, and if so, what `props` weight should we use?"

Data sources (both free):
  * StatsAPI player gameLog  → per-game ACTUALS (hits, K, outs, earned runs)
  * Statcast day cache       → leakage-safe as-of x-stats for the matchup

The harness mirrors the LIVE projection + matchup path exactly so the fitted
weight transfers to runtime:
  * base projection  = recency-weighted mean of the player's PRIOR games
    (same _recency_weights / _weighted_mean analysis.py uses, per-prop
    half_life read from the calibration file)
  * matchup multiplier = analysis._mlb_prop_matchup_mult(...) fed leakage-safe
    as-of features, so the exact same clamp/shape/weight semantics apply.

For each candidate weight w we recompute adjusted_proj = base_proj * mult(w)
and score projection accuracy (MAE / RMSE of actual − adjusted_proj). Weight
selection uses the earlier observations and must still improve MAE on a later
chronological holdout before a nonzero `props` weight can be saved.

LEAKAGE-SAFE MATCHUP COVERAGE (what we can build from Statcast contact data):
  * batter_hits          → opposing STARTER as-of xwOBAcon  (EXACT runtime
                           mirror: runtime also uses the starter's xwoba)
  * pitcher_earned_runs  → opposing LINEUP as-of xwOBAcon vs hand, mapped to
    pitcher_outs           an OPS-equivalent ratio (runtime uses season OPS;
                           this is the leakage-safe contact-quality proxy)
  * pitcher_strikeouts   → NOT fittable here. Runtime scales Ks by the lineup's
                           K%, which Statcast contact xwOBA does not carry.
                           Reported as "no as-of signal" rather than guessed.

Usage:
    python backtest_props.py --season 2024                    # fit, no save
    python backtest_props.py --season 2021-2024               # pooled fit
    python backtest_props.py --season 2021-2024 --save        # + write weight
    python backtest_props.py --season 2024 --max-batters 60   # smaller/faster

Caveats (documented, not hidden):
  * Uses the schedule's probable starter (≈ actual; late scratches unmodeled).
  * A single shared `props` weight is fit (matches the current runtime knob).
    Per-prop diagnostics are printed so you can see which props carry the signal.
  * Enabling props>0 for a Gaussian-residual prop (method "B", e.g.
    pitcher_strikeouts) also requires re-fitting that prop's residual_* block,
    since those stats were fit at mult=1. Method "A" props (batter_hits) read
    the empirical over-rate at the shifted line and need no residual re-fit.
"""

import argparse
import bisect
import json
import os
from collections import Counter, defaultdict

import analysis
import mlb_starters
import savant_history as sh
from backtest_starters import _parse_seasons, _season_bounds, get_season_games
from calibration_loader import (
    load_starter_adjustment,
    save_starter_adjustment,
    _load_blob,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# prop_key -> (statsapi group, gameLog stat field)
PROP_SPEC = {
    "batter_hits": ("hitting", "hits"),
    "pitcher_strikeouts": ("pitching", "strikeOuts"),
    "pitcher_outs": ("pitching", "outs"),
    "pitcher_earned_runs": ("pitching", "earnedRuns"),
}
PITCHER_PROPS = {"pitcher_strikeouts", "pitcher_outs", "pitcher_earned_runs"}

DEFAULT_PROPS = ["batter_hits", "pitcher_outs", "pitcher_earned_runs",
                 "pitcher_strikeouts"]
WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_PRIOR = 5          # prior games before an observation is usable
GAMELOG_CACHE_AGE = 90 * 24 * 3600  # historical logs never change

# StatsAPI abbreviation -> Savant abbreviation (only mismatch is the Athletics).
_ABBR_ALIAS = {"OAK": "ATH"}


# ── team maps ──────────────────────────────────────────────────────────────
_TEAM_CACHE = {}


def _team_maps(season):
    """Return (id->savant_abbr, name->id) for a season, cached."""
    if season in _TEAM_CACHE:
        return _TEAM_CACHE[season]
    data = mlb_starters._get("teams", {"sportId": 1, "season": season})
    id_abbr, name_id = {}, {}
    for t in data.get("teams", []):
        ab = t.get("abbreviation")
        ab = _ABBR_ALIAS.get(ab, ab)
        id_abbr[t["id"]] = ab
        name_id[t.get("name")] = t["id"]
    _TEAM_CACHE[season] = (id_abbr, name_id)
    return _TEAM_CACHE[season]


# ── schedule: date -> games with team ids/abbrevs + probable starters ────────
def season_schedule(season):
    """
    {date: [{home_id, away_id, home_abbr, away_abbr, home_sp, away_sp,
             home_sp_hand, away_sp_hand}]} for the whole season, file-cached.
    Probable-starter handedness comes from the schedule hydrate when present.
    """
    path = os.path.join(CACHE_DIR, f"prop_schedule_{season}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    from datetime import date, timedelta
    s0, s1 = _season_bounds(season)
    id_abbr, _ = _team_maps(season)
    out = {}
    day = date.fromisoformat(s0)
    last = date.fromisoformat(s1)
    while day <= last:
        ds = day.isoformat()
        try:
            data = mlb_starters._get("schedule", {
                "sportId": 1, "date": ds,
                "hydrate": "probablePitcher(note)"})
        except Exception:
            data = {"dates": []}
        games = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                h, a = g["teams"]["home"], g["teams"]["away"]
                hp = h.get("probablePitcher") or {}
                ap = a.get("probablePitcher") or {}
                games.append({
                    "home_id": h["team"]["id"], "away_id": a["team"]["id"],
                    "home_abbr": id_abbr.get(h["team"]["id"]),
                    "away_abbr": id_abbr.get(a["team"]["id"]),
                    "home_sp": hp.get("id"), "away_sp": ap.get("id"),
                })
        if games:
            out[ds] = games
        day += timedelta(days=1)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def _pitcher_hands(rows):
    """Map str(pitcher_id) -> 'L'/'R' from Statcast rows (handedness is static)."""
    hands = {}
    for r in rows:
        pid, h = r.get("pitcher"), r.get("p_throws")
        if pid and h and pid not in hands:
            hands[pid] = h
    return hands


# ── as-of indices ────────────────────────────────────────────────────────────
# A naive as-of estimate rescans the whole season per observation (billions of
# ops). Instead pre-bucket batted balls by key, sorted by date, and keep prefix
# sums so an as-of mean is a single bisect + subtraction (O(log n)).
class AsOfIndex:
    def __init__(self):
        self._buckets = defaultdict(list)   # key -> list[(date, xwoba)]
        self._built = {}                    # key -> (dates, prefix_sum)

    def add(self, key, date, xwoba):
        self._buckets[key].append((date, xwoba))

    def _prep(self, key):
        if key in self._built:
            return self._built[key]
        rows = sorted(self._buckets.get(key, []))
        dates = [d for d, _ in rows]
        pref = [0.0]
        for _, x in rows:
            pref.append(pref[-1] + x)
        self._built[key] = (dates, pref)
        return self._built[key]

    def asof_mean(self, key, as_of, min_bbe=sh.MIN_BBE):
        """Mean xwOBAcon for `key` over batted balls with date < as_of."""
        dates, pref = self._prep(key)
        i = bisect.bisect_left(dates, as_of)   # # of balls strictly before as_of
        if i < min_bbe:
            return None
        return pref[i] / i


def build_pitcher_index(rows):
    """xwOBAcon a pitcher ALLOWED, keyed by str(pitcher_id)."""
    idx = AsOfIndex()
    for r in rows:
        x = r.get("xwoba")
        if x is not None and r.get("pitcher") and r.get("game_date"):
            idx.add(r["pitcher"], r["game_date"], x)
    return idx


def build_team_hand_index(rows):
    """xwOBAcon a team PRODUCED vs a pitcher hand, keyed by (team, p_throws)."""
    idx = AsOfIndex()
    for r in rows:
        x = r.get("xwoba")
        if (x is not None and r.get("batting_team") and r.get("p_throws")
                and r.get("game_date")):
            idx.add((r["batting_team"], r["p_throws"]), r["game_date"], x)
    return idx


# ── player game logs ─────────────────────────────────────────────────────────
def gamelog(player_id, season, group):
    """Per-game splits for a player/season (cached long — history is frozen)."""
    name = f"proplog_{player_id}_{season}_{group}"
    cached = mlb_starters._read_cache(name, max_age=GAMELOG_CACHE_AGE)
    if cached is not None:
        return cached
    try:
        data = mlb_starters._get(
            f"people/{player_id}/stats",
            {"stats": "gameLog", "group": group, "season": season})
        splits = data.get("stats", [{}])[0].get("splits", [])
    except Exception:
        splits = []
    mlb_starters._write_cache(name, splits)
    return splits


def prefetch(seasons, top_n, workers=12):
    """
    Warm every schedule + player gameLog cache the fit needs, in parallel.
    Separated from fit() so the slow one-time StatsAPI pull can be run on its
    own (mirrors backtest_starters.py --fetch). Resume-safe: cached logs skip.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Schedules (one file per season) — needed to resolve opposing starters.
    for s in seasons:
        season_schedule(s)
        print(f"  schedule {s} cached")

    # Collect (player_id, season, group) for every log the fit will read.
    jobs = set()
    for s in seasons:
        for pid in starter_ids([s]):
            jobs.add((pid, s, "pitching"))
        for bid in frequent_batter_ids([s], top_n):
            jobs.add((bid, s, "hitting"))

    todo = [j for j in jobs
            if mlb_starters._read_cache(
                f"proplog_{j[0]}_{j[1]}_{j[2]}", max_age=GAMELOG_CACHE_AGE) is None]
    print(f"gameLogs: {len(jobs)} needed, {len(jobs) - len(todo)} cached, "
          f"fetching {len(todo)} ...")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(gamelog, pid, s, grp): (pid, s, grp)
                for (pid, s, grp) in todo}
        for f in as_completed(futs):
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  fetched {done}/{len(todo)}")
    print("prefetch complete.")


def _half_life(prop_key):
    """Per-prop recency half-life from the calibration file (fallback to sport)."""
    try:
        blob = _load_blob("baseball_mlb")
        cfg = (blob.get("props") or {}).get(prop_key) or {}
        if "half_life" in cfg:
            return cfg["half_life"]
    except Exception:
        pass
    return analysis._half_life_for("baseball_mlb")


# ── player universes ─────────────────────────────────────────────────────────
def starter_ids(seasons):
    """All probable-starter ids seen across the requested seasons."""
    ids = set()
    for s in seasons:
        for g in get_season_games(s, verbose=False):
            ids.add(g["home_sp"])
            ids.add(g["away_sp"])
    return sorted(i for i in ids if i)


def frequent_batter_ids(seasons, top_n):
    """The top_n batter ids by batted-ball volume in the Statcast cache."""
    counts = Counter()
    for s in seasons:
        s0, s1 = _season_bounds(s)
        for r in sh.load_days(s0, s1):
            b = r.get("batter")
            if b and r.get("xwoba") is not None:
                counts[b] += 1
    return [b for b, _ in counts.most_common(top_n)]


# ── observation builders ─────────────────────────────────────────────────────
def _stat_val(split, field):
    try:
        return float(split["stat"].get(field))
    except (KeyError, TypeError, ValueError):
        return None


def build_pitcher_obs(prop_key, seasons, league_by_season, hands_by_season,
                      team_index_by_season, verbose=True):
    """
    Observations for a pitcher prop. matchup feature = opposing lineup's as-of
    xwOBAcon vs the pitcher's hand, mapped to an OPS-equivalent ratio so
    analysis._mlb_prop_matchup_mult's OPS branch fires exactly as at runtime.
    (pitcher_strikeouts has no contact-based as-of signal → skipped there.)
    """
    _, field = PROP_SPEC[prop_key]
    obs = []
    for season in seasons:
        league = league_by_season[season]
        hands = hands_by_season[season]
        team_idx = team_index_by_season[season]
        id_abbr, name_id = _team_maps(season)
        for pid in starter_ids([season]):
            splits = gamelog(pid, season, "pitching")
            splits = sorted(splits, key=lambda s: s.get("date", ""))
            vals = []           # actuals, most-recent-first as we go
            for sp in splits:
                actual = _stat_val(sp, field)
                date = sp.get("date")
                if actual is None or not date:
                    continue
                if len(vals) >= MIN_PRIOR:
                    base = analysis._weighted_mean(
                        vals, analysis._recency_weights(len(vals),
                                                        _half_life(prop_key)))
                    is_home = bool(sp.get("isHome"))
                    opp_id = name_id.get((sp.get("opponent") or {}).get("name"))
                    opp_abbr = id_abbr.get(opp_id)
                    hand = hands.get(str(pid))
                    feat = None
                    if base > 0 and opp_abbr and hand:
                        opp_x = team_idx.asof_mean((opp_abbr, hand), date)
                        if opp_x is not None:
                            # OPS-equivalent ratio: opponent contact vs league.
                            ratio = opp_x / league if league else 1.0
                            side = "home" if is_home else "away"
                            feat = {side: {"opp_offense_vs_hand": {
                                "ops": analysis._MLB_LEAGUE["ops"] * ratio,
                                "k_pct": None}}}
                    if feat is not None:
                        obs.append((base, actual, is_home, feat, date))
                # prepend newest for most-recent-first ordering
                vals.insert(0, actual)
        if verbose:
            print(f"  {prop_key} {season}: pooled {len(obs)} obs so far")
    return obs


def build_batter_obs(prop_key, seasons, pitcher_index_by_season, top_n,
                     verbose=True):
    """
    Observations for a batter prop. matchup feature = opposing STARTER's as-of
    xwOBAcon (exact runtime mirror — runtime scales batter_hits by the starter's
    xwoba). Opposing starter is resolved via the schedule for that date/team.
    """
    _, field = PROP_SPEC[prop_key]
    obs = []
    for season in seasons:
        pitch_idx = pitcher_index_by_season[season]
        sched = season_schedule(season)
        _, name_id = _team_maps(season)
        for bid in frequent_batter_ids([season], top_n):
            splits = gamelog(bid, season, "hitting")
            splits = sorted(splits, key=lambda s: s.get("date", ""))
            vals = []
            for sp in splits:
                actual = _stat_val(sp, field)
                date = sp.get("date")
                if actual is None or not date:
                    continue
                if len(vals) >= MIN_PRIOR:
                    base = analysis._weighted_mean(
                        vals, analysis._recency_weights(len(vals),
                                                        _half_life(prop_key)))
                    is_home = bool(sp.get("isHome"))
                    team_id = name_id.get((sp.get("team") or {}).get("name"))
                    opp_sp = _opposing_starter(sched.get(date, []), team_id,
                                               is_home)
                    feat = None
                    if base > 0 and opp_sp:
                        sx = pitch_idx.asof_mean(str(opp_sp), date)
                        if sx is not None:
                            opp_side = "away" if is_home else "home"
                            feat = {opp_side: {"starter": {"xwoba": sx}}}
                    if feat is not None:
                        obs.append((base, actual, is_home, feat, date))
                vals.insert(0, actual)
        if verbose:
            print(f"  {prop_key} {season}: pooled {len(obs)} obs so far")
    return obs


def _opposing_starter(games, team_id, is_home):
    """Find the opposing probable starter for team_id in this date's games."""
    if not team_id:
        return None
    for g in games:
        if is_home and g["home_id"] == team_id:
            return g.get("away_sp")
        if not is_home and g["away_id"] == team_id:
            return g.get("home_sp")
    return None


# ── scoring ──────────────────────────────────────────────────────────────────
def _score(obs, prop_key, weight):
    """MAE/RMSE of (actual − adjusted_proj) for a given props weight."""
    if not obs:
        return None, None
    se = ae = 0.0
    for base, actual, is_home, feat, *_ in obs:
        mult = analysis._mlb_prop_matchup_mult(prop_key, is_home, feat, weight)
        adj = base * mult
        d = actual - adj
        ae += abs(d)
        se += d * d
    n = len(obs)
    return ae / n, (se / n) ** 0.5


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def _signal_corr(obs, prop_key):
    """Correlation of the raw matchup signal (mult at w=1, minus 1) with the
    projection residual (actual − base). Confirms the signal points the right
    way before we trust the weight."""
    raws, resids = [], []
    for base, actual, is_home, feat, *_ in obs:
        m1 = analysis._mlb_prop_matchup_mult(prop_key, is_home, feat, 1.0)
        raws.append(m1 - 1.0)
        resids.append(actual - base)
    return _pearson(raws, resids)


def fit(seasons, props, top_n=120, do_save=False):
    seasons_str = ",".join(str(s) for s in seasons)
    # Multiple seasons: fit on all earlier seasons and hold out the latest one.
    # One season: use a July 1 boundary, matching the regular-season first/second
    # half split used for the committed 2024 validation.
    holdout_start = (f"{max(seasons)}-01-01" if len(seasons) > 1
                     else f"{seasons[0]}-07-01")

    # Preload per-season Statcast rows once, then build as-of indices (prefix
    # sums) so each matchup lookup is O(log n) instead of a full-season scan.
    league_by_season, hands_by_season = {}, {}
    pitcher_index_by_season, team_index_by_season = {}, {}
    for s in seasons:
        s0, s1 = _season_bounds(s)
        rows = sh.load_days(s0, s1)
        if not rows:
            raise RuntimeError(
                f"No Statcast days cached for {s} — run backtest_starters.py "
                f"--season {s} --fetch first.")
        lv = [r["xwoba"] for r in rows if r["xwoba"] is not None]
        league_by_season[s] = sum(lv) / len(lv)
        hands_by_season[s] = _pitcher_hands(rows)
        pitcher_index_by_season[s] = build_pitcher_index(rows)
        team_index_by_season[s] = build_team_hand_index(rows)

    per_prop = {}
    for prop_key in props:
        if prop_key not in PROP_SPEC:
            print(f"  [skip] unknown prop {prop_key}")
            continue
        print(f"\nbuilding observations for {prop_key} ...")
        if prop_key in PITCHER_PROPS:
            if prop_key == "pitcher_strikeouts":
                print("  [skip] pitcher_strikeouts: no as-of K% signal in the "
                      "Statcast contact cache — needs a K-rate history harness. "
                      "Runtime mult stays 1.0; weight not fit from this prop.")
                continue
            obs = build_pitcher_obs(prop_key, seasons, league_by_season,
                                    hands_by_season, team_index_by_season)
        else:
            obs = build_batter_obs(prop_key, seasons,
                                   pitcher_index_by_season, top_n)
        train_obs = [o for o in obs if len(o) > 4 and o[4] < holdout_start]
        holdout_obs = [o for o in obs if len(o) > 4 and o[4] >= holdout_start]
        if len(train_obs) < 100 or len(holdout_obs) < 100:
            print(f"  [skip] {prop_key}: train={len(train_obs)}, "
                  f"holdout={len(holdout_obs)} (need 100 each).")
            continue
        corr = _signal_corr(train_obs, prop_key)
        train_scores = {w: _score(train_obs, prop_key, w) for w in WEIGHT_GRID}
        holdout_scores = {w: _score(holdout_obs, prop_key, w) for w in WEIGHT_GRID}
        per_prop[prop_key] = {
            "train_n": len(train_obs),
            "holdout_n": len(holdout_obs),
            "corr": corr,
            "train_scores": train_scores,
            "holdout_scores": holdout_scores,
        }

    if not per_prop:
        print("\nNo props produced a usable fit.")
        return

    # ── report ──
    print(f"\n=== props matchup chronological fit — {seasons_str} ===")
    print(f"holdout starts {holdout_start}")
    print(f"{'prop':<22}{'fit':>7}{'test':>7}{'sig_corr':>10}  FIT MAE by weight "
          f"(0 / .25 / .5 / .75 / 1)")
    pooled_train = defaultdict(lambda: [0.0, 0])
    pooled_holdout = defaultdict(lambda: [0.0, 0])
    for prop_key, r in per_prop.items():
        maes = [r["train_scores"][w][0] for w in WEIGHT_GRID]
        base_mae = maes[0]
        best_w = min(WEIGHT_GRID, key=lambda w: r["train_scores"][w][0])
        mae_str = " / ".join(f"{m:.3f}" for m in maes)
        fit_gain = ((base_mae - r["train_scores"][best_w][0]) / base_mae * 100
                    if base_mae else 0)
        holdout_base = r["holdout_scores"][0.0][0]
        holdout_selected = r["holdout_scores"][best_w][0]
        holdout_gain = ((holdout_base - holdout_selected) / holdout_base * 100
                        if holdout_base else 0)
        print(f"{prop_key:<22}{r['train_n']:>7}{r['holdout_n']:>7}"
              f"{r['corr']:>+10.3f}  {mae_str}")
        print(f"{'':46}  selected w={best_w}; fit MAE {fit_gain:+.2f}%; "
              f"holdout {holdout_base:.3f} → {holdout_selected:.3f} "
              f"({holdout_gain:+.2f}%)")
        for w in WEIGHT_GRID:
            pooled_train[w][0] += r["train_scores"][w][0] * r["train_n"]
            pooled_train[w][1] += r["train_n"]
            pooled_holdout[w][0] += r["holdout_scores"][w][0] * r["holdout_n"]
            pooled_holdout[w][1] += r["holdout_n"]

    pooled_fit = {w: pooled_train[w][0] / pooled_train[w][1] for w in WEIGHT_GRID}
    pooled_test = {w: pooled_holdout[w][0] / pooled_holdout[w][1] for w in WEIGHT_GRID}
    chosen = min(WEIGHT_GRID, key=lambda w: pooled_fit[w])
    fit_gain = ((pooled_fit[0.0] - pooled_fit[chosen]) / pooled_fit[0.0] * 100
                if pooled_fit[0.0] else 0)
    holdout_gain = ((pooled_test[0.0] - pooled_test[chosen]) / pooled_test[0.0] * 100
                    if pooled_test[0.0] else 0)
    all_props_improve = all(
        r["holdout_scores"][chosen][0] < r["holdout_scores"][0.0][0]
        for r in per_prop.values()
    ) if chosen != 0.0 else False
    print("\npooled FIT MAE: " + " / ".join(f"{pooled_fit[w]:.3f}" for w in WEIGHT_GRID))
    print("pooled HOLDOUT MAE: " + " / ".join(f"{pooled_test[w]:.3f}" for w in WEIGHT_GRID))
    print(f"fit-selected props weight = {chosen}; fit gain={fit_gain:+.2f}%; "
          f"holdout gain={holdout_gain:+.2f}%; "
          f"every prop improved={all_props_improve}")
    if (chosen == 0.0 or fit_gain <= 0 or holdout_gain <= 0
            or not all_props_improve):
        print("nonzero shared weight did not improve fit, pooled holdout, and "
              "every prop holdout → keeping 0.0")
        chosen = 0.0

    if do_save:
        cur = load_starter_adjustment("baseball_mlb") or {}
        cur["props"] = round(float(chosen), 3)
        methodB = [p for p in per_prop if _prop_method(p) == "B"]
        note = (f"props weight {chosen} fit from {seasons_str}; chronological "
                f"holdout starts {holdout_start} (fit gain {fit_gain:+.2f}%, "
                f"holdout gain {holdout_gain:+.2f}%); ")
        if methodB:
            note += (f"WARNING: re-fit residual_* for {','.join(methodB)} "
                     f"(method B) before trusting — fit at mult=1.")
        else:
            note += "method-A props read empirical over-rate at shifted line (safe)."
        cur["_note"] = note
        save_starter_adjustment("baseball_mlb", cur,
                                meta={"source": f"backtest_props:{seasons_str}",
                                      "fit": True,
                                      "holdout_start": holdout_start})
        print(f"\nsaved starter_adjustment.props = {chosen}")
        if methodB:
            print("NOTE:", note)


def _prop_method(prop_key):
    try:
        blob = _load_blob("baseball_mlb")
        return ((blob.get("props") or {}).get(prop_key) or {}).get("method")
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True,
                    help="single '2024', list '2021,2022', or range '2021-2024'")
    ap.add_argument("--props", default=",".join(DEFAULT_PROPS),
                    help="comma-separated prop keys")
    ap.add_argument("--max-batters", type=int, default=120,
                    help="how many high-volume batters to include")
    ap.add_argument("--fetch", action="store_true",
                    help="pre-cache all schedules + gameLogs (slow, one-time)")
    ap.add_argument("--save", action="store_true",
                    help="write the fitted props weight to calibration")
    args = ap.parse_args()

    seasons = _parse_seasons(args.season)
    props = [p.strip() for p in args.props.split(",") if p.strip()]
    if args.fetch:
        prefetch(seasons, args.max_batters)
    fit(seasons, props, top_n=args.max_batters, do_save=args.save)
