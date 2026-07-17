"""
Fit / validate per-prop MLB matchup weights (`starter_adjustment.props.<prop>`)
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
chronological holdout and two expanding-window validation folds before a
nonzero `props` weight can be saved.

LEAKAGE-SAFE MATCHUP COVERAGE (what we can build from Statcast contact data):
  * batter_hits          → opposing STARTER as-of xBA, combined at the AB level;
                           retained at weight 0 unless a hit-specific holdout wins
  * batter_strikeouts    → batter recent K/PA combined with opposing STARTER's
                           as-of K/BF through the same log5 runtime function
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
  * Each prop gets its own weight. A weak hit signal can no longer disable a
    useful strikeout signal, or vice versa.
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
    load_lineup_adjustment,
    load_starter_adjustment,
    save_lineup_adjustment,
    save_starter_adjustment,
    _load_blob,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# prop_key -> (statsapi group, gameLog stat field)
PROP_SPEC = {
    "batter_hits": ("hitting", "hits"),
    "batter_strikeouts": ("hitting", "strikeOuts"),
    "pitcher_strikeouts": ("pitching", "strikeOuts"),
    "pitcher_outs": ("pitching", "outs"),
    "pitcher_earned_runs": ("pitching", "earnedRuns"),
}
PITCHER_PROPS = {"pitcher_strikeouts", "pitcher_outs", "pitcher_earned_runs"}

DEFAULT_PROPS = ["batter_hits", "batter_strikeouts", "pitcher_outs",
                 "pitcher_earned_runs", "pitcher_strikeouts"]
WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_PRIOR = 5          # prior games before an observation is usable
LIVE_RECENT_N = 20     # app.py MLB recent_n_default; keep harness in parity
GAMELOG_CACHE_AGE = 90 * 24 * 3600  # historical logs never change

# Average at-bats by announced batting slot among the 120 highest-volume 2024
# batters. These describe opportunity, not the prop outcome being predicted.
LINEUP_SLOT_EXPECTED_AB = {
    "1": 4.0602, "2": 3.9427, "3": 3.8170,
    "4": 3.7922, "5": 3.7030, "6": 3.6185,
    "7": 3.4869, "8": 3.4076, "9": 3.4635,
}

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


def season_lineup_slots(season):
    """Return {date: {player_id: batting_slot}} from announced lineups."""
    path = os.path.join(CACHE_DIR, f"prop_lineups_{season}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    s0, s1 = _season_bounds(season)
    try:
        data = mlb_starters._get("schedule", {
            "sportId": 1,
            "startDate": s0,
            "endDate": s1,
            "hydrate": "lineups",
        }, timeout=120)
    except Exception:
        data = {"dates": []}
    out = {}
    for day in data.get("dates", []):
        slots = {}
        for game in day.get("games", []):
            for player in mlb_starters._lineup_players(game).values():
                player_id = player.get("player_id")
                if player_id:
                    slots[str(player_id)] = player["batting_order"]
        if slots:
            out[day.get("date")] = slots
    if out:
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


class AsOfPitcherKIndex:
    """Leakage-safe pitcher K/BF and average innings/start prefix index."""

    def __init__(self):
        self._buckets = defaultdict(list)
        self._built = {}

    def add(self, pitcher_id, date, strikeouts, batters_faced, innings):
        self._buckets[str(pitcher_id)].append(
            (date, strikeouts, batters_faced, innings))

    def _prep(self, pitcher_id):
        key = str(pitcher_id)
        if key in self._built:
            return self._built[key]
        rows = sorted(self._buckets.get(key, []))
        dates = []
        pref_k = [0.0]
        pref_bf = [0.0]
        pref_ip = [0.0]
        for date, strikeouts, batters_faced, innings in rows:
            dates.append(date)
            pref_k.append(pref_k[-1] + strikeouts)
            pref_bf.append(pref_bf[-1] + batters_faced)
            pref_ip.append(pref_ip[-1] + innings)
        self._built[key] = (dates, pref_k, pref_bf, pref_ip)
        return self._built[key]

    def asof(self, pitcher_id, as_of, min_bf=50):
        dates, pref_k, pref_bf, pref_ip = self._prep(pitcher_id)
        i = bisect.bisect_left(dates, as_of)
        if i < 1 or pref_bf[i] < min_bf:
            return None
        return {
            "k_pct": pref_k[i] / pref_bf[i],
            "bf": pref_bf[i],
            "avg_ip": pref_ip[i] / i,
        }


def build_pitcher_xba_index(rows):
    """Expected BA a pitcher allowed, available in Statcast cache schema v3."""
    idx = AsOfIndex()
    for row in rows:
        xba = row.get("xba")
        if xba is not None and row.get("pitcher") and row.get("game_date"):
            idx.add(row["pitcher"], row["game_date"], xba)
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


def build_pitcher_k_index(season):
    """Build as-of starter K rates from historical StatsAPI game logs."""
    idx = AsOfPitcherKIndex()
    for pitcher_id in starter_ids([season]):
        for split in gamelog(pitcher_id, season, "pitching"):
            date = split.get("date")
            strikeouts = _stat_val(split, "strikeOuts")
            batters_faced = _stat_val(split, "battersFaced")
            innings = mlb_starters._parse_ip(
                (split.get("stat") or {}).get("inningsPitched"))
            if (date and strikeouts is not None and batters_faced
                    and innings is not None):
                idx.add(pitcher_id, date, strikeouts, batters_faced, innings)
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
    return analysis._player_prop_half_life("baseball_mlb")


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
                    recent_vals = vals[:LIVE_RECENT_N]
                    base = analysis._weighted_mean(
                        recent_vals,
                        analysis._recency_weights(len(recent_vals),
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


def build_batter_obs(prop_key, seasons, pitcher_xba_index_by_season,
                     pitcher_k_index_by_season,
                     top_n, verbose=True):
    """
    Observations for a batter prop. matchup feature = opposing STARTER's as-of
    xBA (hits) or K/BF (strikeouts). Opposing starter is resolved via the
    schedule for that date/team.
    """
    _, field = PROP_SPEC[prop_key]
    obs = []
    for season in seasons:
        pitcher_xba_idx = pitcher_xba_index_by_season[season]
        pitcher_k_idx = pitcher_k_index_by_season[season]
        sched = season_schedule(season)
        lineup_slots = (season_lineup_slots(season)
                        if prop_key == "batter_hits" else {})
        _, name_id = _team_maps(season)
        for bid in frequent_batter_ids([season], top_n):
            splits = gamelog(bid, season, "hitting")
            splits = sorted(splits, key=lambda s: s.get("date", ""))
            vals = []
            plate_appearances = []
            at_bats = []
            for sp in splits:
                actual = _stat_val(sp, field)
                date = sp.get("date")
                if actual is None or not date:
                    continue
                if len(vals) >= MIN_PRIOR:
                    recent_vals = vals[:LIVE_RECENT_N]
                    recent_pa = plate_appearances[:LIVE_RECENT_N]
                    recent_ab = at_bats[:LIVE_RECENT_N]
                    recent_weights = analysis._recency_weights(
                        len(recent_vals), _half_life(prop_key))
                    base = analysis._weighted_mean(
                        recent_vals, recent_weights)
                    is_home = bool(sp.get("isHome"))
                    team_id = name_id.get((sp.get("team") or {}).get("name"))
                    opp_sp = _opposing_starter(sched.get(date, []), team_id,
                                               is_home)
                    feat = None
                    player_context = None
                    if base > 0 and opp_sp:
                        if prop_key == "batter_strikeouts":
                            starter = pitcher_k_idx.asof(str(opp_sp), date)
                        else:
                            sxba = pitcher_xba_idx.asof_mean(str(opp_sp), date)
                            workload = pitcher_k_idx.asof(str(opp_sp), date)
                            starter = ({
                                "xba": sxba,
                                "avg_ip": ((workload or {}).get("avg_ip")
                                           or 5.5),
                            } if sxba is not None else None)
                        if starter is not None:
                            opp_side = "away" if is_home else "home"
                            feat = {opp_side: {"starter": starter}}
                            if prop_key in ("batter_hits", "batter_strikeouts"):
                                recent_exposure = (recent_ab if prop_key == "batter_hits"
                                                   else recent_pa)
                                valid_pa = [
                                    (pa, weight) for pa, weight in zip(
                                        recent_exposure, recent_weights)
                                    if pa is not None and pa > 0
                                ]
                                if valid_pa:
                                    expected_pa = analysis._weighted_mean(
                                        [pa for pa, _ in valid_pa],
                                        [weight for _, weight in valid_pa])
                                    player_context = {
                                        "base_projection": base,
                                        "expected_exposure": expected_pa,
                                    }
                                    batting_order = (lineup_slots.get(date) or {}).get(
                                        str(bid))
                                    if batting_order:
                                        player_context["batting_order"] = batting_order
                    if feat is not None:
                        obs.append((base, actual, is_home, feat, date,
                                    player_context))
                vals.insert(0, actual)
                plate_appearances.insert(
                    0, _stat_val(sp, "plateAppearances"))
                at_bats.insert(0, _stat_val(sp, "atBats"))
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
    for row in obs:
        base, actual, is_home, feat = row[:4]
        player_context = row[5] if len(row) > 5 else None
        mult = analysis._mlb_prop_matchup_mult(
            prop_key, is_home, feat, weight,
            player_context=player_context)
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
    for row in obs:
        base, actual, is_home, feat = row[:4]
        player_context = row[5] if len(row) > 5 else None
        m1 = analysis._mlb_prop_matchup_mult(
            prop_key, is_home, feat, 1.0,
            player_context=player_context)
        raws.append(m1 - 1.0)
        resids.append(actual - base)
    return _pearson(raws, resids)


def _rolling_splits(obs, min_fold_n=100):
    """Two expanding-train chronological folds, keeping dates intact."""
    rows = sorted(obs, key=lambda row: row[4])
    if len(rows) < 3 * min_fold_n:
        return []
    cut1 = rows[int(len(rows) * 0.6)][4]
    cut2 = rows[int(len(rows) * 0.8)][4]
    if cut1 == cut2:
        return []
    folds = [
        ([row for row in rows if row[4] < cut1],
         [row for row in rows if cut1 <= row[4] < cut2]),
        ([row for row in rows if row[4] < cut2],
         [row for row in rows if row[4] >= cut2]),
    ]
    if any(len(train) < min_fold_n or len(test) < min_fold_n
           for train, test in folds):
        return []
    return folds


def _rolling_weight_validation(obs, prop_key, candidate_weight):
    """Validate a fixed candidate while reporting each expanding fit's choice."""
    results = []
    for train, test in _rolling_splits(obs):
        train_scores = {w: _score(train, prop_key, w) for w in WEIGHT_GRID}
        fit_weight = min(
            WEIGHT_GRID, key=lambda weight: train_scores[weight][0])
        baseline_mae = _score(test, prop_key, 0.0)[0]
        candidate_mae = _score(test, prop_key, candidate_weight)[0]
        gain = ((baseline_mae - candidate_mae) / baseline_mae * 100
                if baseline_mae else 0.0)
        results.append({
            "start": test[0][4],
            "n": len(test),
            "fit_weight": fit_weight,
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
            "gain_pct": gain,
        })
    return results


def _score_lineup(obs, weight):
    """MAE for batter-hit projections with a candidate lineup-order weight."""
    if not obs:
        return None
    absolute_error = 0.0
    for row in obs:
        base, actual = row[:2]
        context = row[5]
        mult = analysis._lineup_exposure_mult(
            context.get("expected_exposure"),
            context.get("batting_order"),
            weight,
            LINEUP_SLOT_EXPECTED_AB,
        )
        absolute_error += abs(actual - base * mult)
    return absolute_error / len(obs)


def _rolling_lineup_validation(obs, candidate_weight):
    """Run the same two expanding chronological folds for lineup exposure."""
    results = []
    for train, test in _rolling_splits(obs):
        train_scores = {weight: _score_lineup(train, weight)
                        for weight in WEIGHT_GRID}
        fit_weight = min(WEIGHT_GRID, key=train_scores.get)
        baseline_mae = _score_lineup(test, 0.0)
        candidate_mae = _score_lineup(test, candidate_weight)
        gain = ((baseline_mae - candidate_mae) / baseline_mae * 100
                if baseline_mae else 0.0)
        results.append({
            "start": test[0][4],
            "n": len(test),
            "fit_weight": fit_weight,
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
            "gain_pct": gain,
        })
    return results


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
    pitcher_xba_index_by_season = {}
    pitcher_k_index_by_season = {}
    team_index_by_season = {}
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
        pitcher_xba_index_by_season[s] = build_pitcher_xba_index(rows)
        pitcher_k_index_by_season[s] = build_pitcher_k_index(s)
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
                                   pitcher_xba_index_by_season,
                                   pitcher_k_index_by_season, top_n)
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
            "observations": obs,
        }

    if not per_prop:
        print("\nNo props produced a usable fit.")
        return

    # Batting-order opportunity is a separate signal from starter quality.
    # Fit it only for hits: strikeout exposure did not pass forward validation.
    lineup_result = None
    hits_result = per_prop.get("batter_hits")
    if hits_result:
        lineup_obs = [
            row for row in hits_result["observations"]
            if len(row) > 5 and row[5]
            and row[5].get("batting_order")
            and row[5].get("expected_exposure")
        ]
        lineup_train = [row for row in lineup_obs if row[4] < holdout_start]
        lineup_holdout = [row for row in lineup_obs if row[4] >= holdout_start]
        if len(lineup_train) >= 100 and len(lineup_holdout) >= 100:
            train_scores = {
                weight: _score_lineup(lineup_train, weight)
                for weight in WEIGHT_GRID
            }
            holdout_scores = {
                weight: _score_lineup(lineup_holdout, weight)
                for weight in WEIGHT_GRID
            }
            best_weight = min(WEIGHT_GRID, key=train_scores.get)
            baseline_mae = holdout_scores[0.0]
            candidate_mae = holdout_scores[best_weight]
            holdout_gain = ((baseline_mae - candidate_mae) / baseline_mae * 100
                            if baseline_mae else 0.0)
            rolling = _rolling_lineup_validation(lineup_obs, best_weight)
            rolling_passed = (len(rolling) == 2
                              and all(fold["gain_pct"] > 0 for fold in rolling))
            accepted_weight = (best_weight if best_weight != 0.0
                               and holdout_gain > 0 and rolling_passed else 0.0)
            lineup_result = {
                "fit_n": len(lineup_train),
                "holdout_n": len(lineup_holdout),
                "candidate_weight": best_weight,
                "selected_weight": accepted_weight,
                "baseline_mae": round(baseline_mae, 6),
                "candidate_mae": round(candidate_mae, 6),
                "selected_mae": round(holdout_scores[accepted_weight], 6),
                "holdout_gain_pct": round(holdout_gain, 6),
                "rolling_folds": rolling,
                "decision": (
                    "enabled: passed holdout and both rolling folds"
                    if accepted_weight else
                    "disabled: candidate failed at least one validation gate"
                ),
            }

    # ── report ──
    print(f"\n=== props matchup chronological fit — {seasons_str} ===")
    print(f"holdout starts {holdout_start}")
    print(f"{'prop':<22}{'fit':>7}{'test':>7}{'sig_corr':>10}  FIT MAE by weight "
          f"(0 / .25 / .5 / .75 / 1)")
    pooled_train = defaultdict(lambda: [0.0, 0])
    pooled_holdout = defaultdict(lambda: [0.0, 0])
    selected_weights = {}
    validation_results = {}
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
        rolling = _rolling_weight_validation(
            r["observations"], prop_key, best_w)
        rolling_passed = (len(rolling) == 2
                          and all(fold["gain_pct"] > 0 for fold in rolling))
        accepted_w = (best_w if best_w != 0.0 and fit_gain > 0
                      and holdout_gain > 0 and r["corr"] > 0
                      and rolling_passed else 0.0)
        selected_weights[prop_key] = accepted_w
        validation_results[prop_key] = {
            "fit_n": r["train_n"],
            "holdout_n": r["holdout_n"],
            "signal_correlation": round(r["corr"], 6),
            "candidate_weight": best_w,
            "selected_weight": accepted_w,
            "baseline_mae": round(holdout_base, 6),
            "candidate_mae": round(holdout_selected, 6),
            "selected_mae": round(
                r["holdout_scores"][accepted_w][0], 6),
            "rolling_folds": rolling,
            "decision": ("enabled: passed holdout and both rolling folds"
                         if accepted_w else
                         "disabled: candidate failed at least one validation gate"),
        }
        print(f"{prop_key:<22}{r['train_n']:>7}{r['holdout_n']:>7}"
              f"{r['corr']:>+10.3f}  {mae_str}")
        print(f"{'':46}  fit w={best_w}; accepted w={accepted_w}; "
              f"fit MAE {fit_gain:+.2f}%; "
              f"holdout {holdout_base:.3f} → {holdout_selected:.3f} "
              f"({holdout_gain:+.2f}%)")
        if rolling:
            print(f"{'':46}  rolling folds: " + "; ".join(
                f"{fold['start']} n={fold['n']} train-w={fold['fit_weight']} "
                f"candidate {fold['gain_pct']:+.2f}%"
                for fold in rolling))
        else:
            print(f"{'':46}  rolling folds unavailable (weight rejected)")
        for w in WEIGHT_GRID:
            pooled_train[w][0] += r["train_scores"][w][0] * r["train_n"]
            pooled_train[w][1] += r["train_n"]
            pooled_holdout[w][0] += r["holdout_scores"][w][0] * r["holdout_n"]
            pooled_holdout[w][1] += r["holdout_n"]

    pooled_fit = {w: pooled_train[w][0] / pooled_train[w][1] for w in WEIGHT_GRID}
    pooled_test = {w: pooled_holdout[w][0] / pooled_holdout[w][1] for w in WEIGHT_GRID}
    print("\npooled FIT MAE: " + " / ".join(f"{pooled_fit[w]:.3f}" for w in WEIGHT_GRID))
    print("pooled HOLDOUT MAE: " + " / ".join(f"{pooled_test[w]:.3f}" for w in WEIGHT_GRID))
    print("accepted per-prop weights: " + ", ".join(
        f"{prop}={weight}" for prop, weight in selected_weights.items()))
    if lineup_result:
        print("\n=== announced-lineup exposure fit — batter_hits ===")
        print(f"fit n={lineup_result['fit_n']}; "
              f"holdout n={lineup_result['holdout_n']}; "
              f"candidate w={lineup_result['candidate_weight']}; "
              f"accepted w={lineup_result['selected_weight']}")
        print(f"holdout {lineup_result['baseline_mae']:.6f} → "
              f"{lineup_result['candidate_mae']:.6f} "
              f"({lineup_result['holdout_gain_pct']:+.3f}%)")
        print("rolling folds: " + "; ".join(
            f"{fold['start']} n={fold['n']} "
            f"candidate {fold['gain_pct']:+.3f}%"
            for fold in lineup_result["rolling_folds"]))

    if do_save:
        cur = load_starter_adjustment("baseball_mlb") or {}
        existing_prop_weights = cur.get("props")
        if not isinstance(existing_prop_weights, dict):
            existing_prop_weights = {}
        prop_weights = {
            prop: existing_prop_weights.get(prop, 0.0)
            for prop in PROP_SPEC
        }
        prop_weights.update({
            prop: round(float(weight), 3)
            for prop, weight in selected_weights.items()
        })
        cur["props"] = prop_weights
        methodB = [p for p, weight in selected_weights.items()
                   if weight and _prop_method(p) == "B"]
        note = (f"per-prop weights fit from {seasons_str}; chronological "
                f"holdout starts {holdout_start}; live recent window "
                f"={LIVE_RECENT_N}; two rolling validation folds required; "
                f"selected {prop_weights}. ")
        if methodB:
            note += (f"WARNING: re-fit residual_* for {','.join(methodB)} "
                     f"(method B) before trusting — fit at mult=1.")
        else:
            note += "method-A props read empirical over-rate at shifted line (safe)."
        cur["_note"] = note
        existing_results = (((_load_blob("baseball_mlb").get("meta") or {})
                             .get("starter_adjustment") or {})
                            .get("props") or {}).get("results") or {}
        merged_results = dict(existing_results)
        merged_results.update(validation_results)
        save_starter_adjustment("baseball_mlb", cur,
                                meta={"props": {
                                    "source": f"backtest_props:{seasons_str}",
                                    "fit": True,
                                    "holdout_start": holdout_start,
                                    "holdout_window": (
                                        f"{holdout_start} through season end"),
                                    "metric": "MAE",
                                    "recent_n": LIVE_RECENT_N,
                                    "rolling_validation_folds": 2,
                                    "results": merged_results,
                                }})
        print(f"\nsaved starter_adjustment.props = {prop_weights}")
        if methodB:
            print("NOTE:", note)
        if lineup_result:
            lineup_cfg = load_lineup_adjustment("baseball_mlb") or {}
            lineup_cfg.update({
                "enabled": True,
                "props": {
                    "batter_hits": lineup_result["selected_weight"],
                    "batter_strikeouts": 0.0,
                },
                "slot_expected_exposure": {
                    "batter_hits": LINEUP_SLOT_EXPECTED_AB,
                },
                "_note": (
                    "Announced batting order adjusts expected at-bats for "
                    "batter hits only; strikeout exposure failed forward "
                    "validation and remains disabled."
                ),
            })
            save_lineup_adjustment(
                "baseball_mlb",
                lineup_cfg,
                meta={
                    "source": f"backtest_props:{seasons_str}",
                    "fit": True,
                    "holdout_start": holdout_start,
                    "holdout_window": f"{holdout_start} through season end",
                    "metric": "MAE",
                    "recent_n": LIVE_RECENT_N,
                    "slot_exposure_source": (
                        "Average at-bats by batting slot among the 120 "
                        "highest-volume 2024 batters"
                    ),
                    "results": {"batter_hits": lineup_result},
                },
            )
            print("saved lineup_adjustment.props = "
                  f"{lineup_cfg['props']}")
    return selected_weights


def _prop_method(prop_key):
    try:
        blob = _load_blob("baseball_mlb")
        return ((blob.get("props") or {}).get(prop_key) or {}).get("method")
    except Exception:
        return None


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
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
