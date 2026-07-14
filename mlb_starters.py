"""
Client for the MLB Stats API (https://statsapi.mlb.com) — free, no API key.

Phase 1 MLB enhancement. Supplies the starter-aware features the team/prop
models were previously blind to:

  * probable starting pitchers per game,
  * starter handedness + season quality (ERA, K%, BB%),
  * opposing lineup's offensive quality split by pitcher handedness (vs LHP/RHP).

We hit the JSON endpoints directly with ``requests`` (already a dependency)
instead of adding ``pybaseball``, keeping this consistent with odds_client.py
and espn_client.py. Results are file-cached like the other clients.

IMPORTANT — nothing here decides how strongly these features move a prediction.
This module only *produces* normalized features. The blend/weight is a separate,
empirically-fit knob (see calibration) so the strength is fit from graded
outcomes rather than guessed.
"""

import csv
import io
import json
import math
import os
import random
import time
import unicodedata

import requests


BASE_URL = "https://statsapi.mlb.com/api/v1"
# Baseball Savant (Statcast) — source of the "expected" (x) stats that strip
# out luck/sequencing. Only reachable with a browser-like User-Agent.
SAVANT_BASE = "https://baseballsavant.mlb.com"
SAVANT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsbookValueFinder/1.0)"}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
# Probable starters and daily splits move during the day, so keep this short.
CACHE_MAX_AGE = 3600  # 1 hour

# League-average baselines used purely as *priors* for normalization / log5-style
# matchup blending. These are stable year-to-year but drift slowly; they are NOT
# fitted parameters and NOT expected values for any test. Refresh occasionally.
LEAGUE_AVG = {
    "era": 4.10,     # runs allowed per 9
    "k_pct": 0.222,  # strikeouts / batters faced
    "bb_pct": 0.082, # walks / batters faced
    "ops": 0.711,    # team OPS baseline
}
PYTHAGOREAN_EXPONENT = 1.83


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name):
    import hashlib
    safe = hashlib.md5(name.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"mlb_{safe}.json")


def _read_cache(name, max_age=CACHE_MAX_AGE):
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            blob = json.load(f)
        if time.time() - blob.get("cached_at", 0) < max_age:
            return blob.get("data")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _write_cache(name, data):
    _ensure_cache_dir()
    with open(_cache_path(name), "w") as f:
        json.dump({"cached_at": time.time(), "data": data}, f)


def _get(path, params=None, max_retries=4, backoff_base=1.5, timeout=30):
    """GET the Stats API with retry+backoff on 429/5xx (see odds_client)."""
    url = f"{BASE_URL}/{path.lstrip('/')}"
    resp = None
    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params or {}, timeout=timeout)
        retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
        if retryable and attempt < max_retries:
            delay = backoff_base ** attempt + random.uniform(0, 0.5)
            time.sleep(delay)
            continue
        break
    resp.raise_for_status()
    return resp.json()


def _get_savant_csv(path, params=None, max_retries=4, backoff_base=1.5):
    """
    Fetch a Baseball Savant leaderboard as CSV and return a list of dict rows.

    Savant rejects the default urllib/library User-Agent (403), so we send a
    browser-like UA. Parsed with the stdlib csv module — no pandas needed.
    """
    url = f"{SAVANT_BASE}/{path.lstrip('/')}"
    resp = None
    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params or {}, headers=SAVANT_HEADERS, timeout=30)
        retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
        if retryable and attempt < max_retries:
            time.sleep(backoff_base ** attempt + random.uniform(0, 0.5))
            continue
        break
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def get_pitcher_expected_stats(season, min_bip=40):
    """
    Baseball Savant expected (x) stats for all pitchers with >= ``min_bip``
    balls in play, keyed by str(player_id) (same MLBAM id as the Stats API).

    Returns {player_id: {'xera','xwoba','xba'}}. These are the luck-stripped
    "true" stats that set Savant apart from ESPN/traditional lines.
    """
    # v2 adds xBA to the cached row shape; do not let a pre-xBA daily cache
    # silently disable the hit-specific matchup model.
    cache = f"savant_xstats_pitcher_v2_{season}_{min_bip}"
    cached = _read_cache(cache, max_age=24 * 3600)  # refresh daily
    if cached is not None:
        return cached
    rows = _get_savant_csv("leaderboard/expected_statistics", {
        "type": "pitcher", "year": season, "position": "", "team": "",
        "filterType": "bip", "min": min_bip, "csv": "true",
    })
    out = {}
    for r in rows:
        pid = r.get("player_id")
        if not pid:
            continue
        out[str(pid)] = {
            "xera": _to_float(r.get("xera")),
            "xwoba": _to_float(r.get("est_woba")),
            "xba": _to_float(r.get("est_ba")),
        }
    _write_cache(cache, out)
    return out


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ip(v):
    """Parse StatsAPI inningsPitched ('150.1' = 150 + 1/3) into float innings."""
    if v is None:
        return None
    try:
        whole, _, frac = str(v).partition(".")
        return int(whole) + {"": 0, "0": 0, "1": 1, "2": 2}.get(frac, 0) / 3.0
    except (TypeError, ValueError):
        return None


def _norm(name):
    """Normalize a team name for matching across data sources."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(c for c in n.lower() if c.isalnum() or c.isspace()).strip()


def _safe_div(a, b):
    try:
        a = float(a)
        b = float(b)
        return a / b if b else None
    except (TypeError, ValueError):
        return None


def expected_runs_from_factors(base_runs, offense_factor,
                               staff_suppression, offense_weight=1.0,
                               pitching_weight=1.0):
    """Convert league-relative offense and run prevention into expected runs.

    ``offense_factor`` and ``staff_suppression`` are centered on 1.0. A better
    offense raises the expectation; better opposing run prevention lowers it.
    The weights are fitted chronologically by ``backtest_starters.py`` rather
    than selected in the live application.
    """
    try:
        base_runs = float(base_runs)
        offense_factor = float(offense_factor)
        staff_suppression = float(staff_suppression)
        offense_weight = float(offense_weight)
        pitching_weight = float(pitching_weight)
    except (TypeError, ValueError):
        return None
    if (base_runs <= 0 or offense_factor <= 0 or staff_suppression <= 0
            or offense_weight < 0 or pitching_weight < 0):
        return None
    expected = (base_runs * offense_factor ** offense_weight
                / staff_suppression ** pitching_weight)
    # Protect the downstream score distribution from a pathological upstream
    # feed while retaining a much wider range than normal MLB expectations.
    return max(0.5, min(12.0, expected))


def pythagorean_win_probability(runs_scored, runs_allowed,
                                exponent=PYTHAGOREAN_EXPONENT):
    """Return Bill James's modern-baseball Pythagorean win probability."""
    try:
        runs_scored = float(runs_scored)
        runs_allowed = float(runs_allowed)
        exponent = float(exponent)
    except (TypeError, ValueError):
        return None
    if runs_scored <= 0 or runs_allowed <= 0 or exponent <= 0:
        return None
    scored_power = runs_scored ** exponent
    return scored_power / (scored_power + runs_allowed ** exponent)


def poisson_margin_probability(home_runs, away_runs, home_spread,
                               max_runs=30):
    """Return P(home score + spread > away score) from expected runs.

    This is intended for MLB half-run spreads, which cannot push. Any tiny
    probability above ``max_runs`` is folded into that terminal score bucket.
    """
    try:
        home_runs = float(home_runs)
        away_runs = float(away_runs)
        home_spread = float(home_spread)
        max_runs = int(max_runs)
    except (TypeError, ValueError):
        return None
    if home_runs <= 0 or away_runs <= 0 or max_runs < 1:
        return None

    def probabilities(expected):
        values = [math.exp(-expected)]
        for score in range(1, max_runs + 1):
            values.append(values[-1] * expected / score)
        values[-1] += max(0.0, 1.0 - sum(values))
        return values

    home_prob = probabilities(home_runs)
    away_prob = probabilities(away_runs)
    return sum(
        hp * ap
        for home_score, hp in enumerate(home_prob)
        for away_score, ap in enumerate(away_prob)
        if home_score + home_spread > away_score
    )


def negative_binomial_margin_probability(home_runs, away_runs, home_spread,
                                         dispersion, max_runs=30):
    """Return a run-line probability with overdispersed team run totals.

    ``dispersion`` uses ``variance = mean + dispersion * mean**2``. A value of
    zero is exactly the independent-Poisson model, while positive values allow
    the heavier score tails seen in baseball. The dispersion is fitted only on
    pre-holdout games by ``backtest_starters.py``.
    """
    try:
        home_runs = float(home_runs)
        away_runs = float(away_runs)
        home_spread = float(home_spread)
        dispersion = float(dispersion)
        max_runs = int(max_runs)
    except (TypeError, ValueError):
        return None
    if (home_runs <= 0 or away_runs <= 0 or dispersion < 0
            or max_runs < 1):
        return None
    if dispersion == 0:
        return poisson_margin_probability(
            home_runs, away_runs, home_spread, max_runs)

    def probabilities(expected):
        size = 1.0 / dispersion
        success_probability = size / (size + expected)
        failure_probability = 1.0 - success_probability
        values = [success_probability ** size]
        for score in range(1, max_runs + 1):
            values.append(
                values[-1]
                * (score - 1.0 + size) / score
                * failure_probability
            )
        values[-1] += max(0.0, 1.0 - sum(values))
        return values

    home_prob = probabilities(home_runs)
    away_prob = probabilities(away_runs)
    return sum(
        hp * ap
        for home_score, hp in enumerate(home_prob)
        for away_score, ap in enumerate(away_prob)
        if home_score + home_spread > away_score
    )


def get_team_index(season):
    """Return {normalized_name: {'id', 'name', 'abbr'}} for all 30 MLB teams."""
    cache = f"team_index_{season}"
    cached = _read_cache(cache, max_age=7 * 24 * 3600)  # teams change yearly
    if cached is not None:
        return cached
    data = _get("teams", {"sportId": 1, "season": season})
    index = {}
    for t in data.get("teams", []):
        if t.get("sport", {}).get("id") != 1:
            continue
        index[_norm(t["name"])] = {
            "id": t["id"],
            "name": t["name"],
            "abbr": t.get("abbreviation"),
        }
    _write_cache(cache, index)
    return index


def _match_team_id(team_name, team_index):
    """Match an odds/ESPN team name to a Stats API team id (tolerant)."""
    key = _norm(team_name)
    if key in team_index:
        return team_index[key]
    # Substring fallback (handles "Oakland Athletics" vs "Athletics", etc.)
    for norm_name, info in team_index.items():
        if norm_name in key or key in norm_name:
            return info
        # last word (nickname) match, e.g. "... Athletics"
        if norm_name.split() and norm_name.split()[-1] == key.split()[-1:] and key.split():
            return info
    return None


def get_probable_starters(date):
    """
    Probable starting pitchers for every game on ``date`` (YYYY-MM-DD).

    Returns {normalized_team_name: {'pitcher_id', 'name', 'team_id'}}.
    Teams without an announced starter are simply omitted.
    """
    cache = f"probables_{date}"
    cached = _read_cache(cache)
    if cached is not None:
        return cached
    data = _get("schedule", {
        "sportId": 1, "date": date, "hydrate": "probablePitcher",
    })
    out = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            for side in ("home", "away"):
                team = g.get("teams", {}).get(side, {})
                pp = team.get("probablePitcher") or {}
                tname = team.get("team", {}).get("name")
                if tname and pp.get("id"):
                    out[_norm(tname)] = {
                        "pitcher_id": pp["id"],
                        "name": pp.get("fullName"),
                        "team_id": team.get("team", {}).get("id"),
                    }
    _write_cache(cache, out)
    return out


def get_confirmed_lineup(home_team, away_team, date):
    """Return announced batting orders for one game, or an empty context.

    The Stats API exposes each side's players in batting-order sequence through
    ``hydrate=lineups``. Results use a short cache because orders are commonly
    posted or changed shortly before first pitch.
    """
    empty = {
        "home_confirmed": False,
        "away_confirmed": False,
        "players": {},
    }
    cache = f"lineups_{date}"
    data = _read_cache(cache, max_age=5 * 60)
    if data is None:
        data = _get("schedule", {
            "sportId": 1, "date": date, "hydrate": "lineups",
        })
        _write_cache(cache, data)

    target_home = _norm(home_team)
    target_away = _norm(away_team)
    for day in data.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            game_home = _norm(
                ((teams.get("home") or {}).get("team") or {}).get("name"))
            game_away = _norm(
                ((teams.get("away") or {}).get("team") or {}).get("name"))
            if not (_names_match(target_home, game_home)
                    and _names_match(target_away, game_away)):
                continue

            result = dict(empty)
            result["players"] = _lineup_players(game)
            for side in ("home", "away"):
                result[f"{side}_confirmed"] = sum(
                    player["side"] == side
                    for player in result["players"].values()) == 9
            return result
    return empty


def _lineup_players(game):
    """Extract normalized player records from one hydrated schedule game."""
    lineups = game.get("lineups") or {}
    result = {}
    for side in ("home", "away"):
        players = (lineups.get(f"{side}Players") or [])[:9]
        for slot, player in enumerate(players, 1):
            name = player.get("fullName")
            if not name:
                continue
            result[_norm(name)] = {
                "player_id": player.get("id"),
                "name": name,
                "side": side,
                "batting_order": slot,
            }
    return result


def lineup_player_context(lineup, player_name):
    """Return a confirmed player's lineup record, or None when not announced."""
    if not lineup or not player_name:
        return None
    player = (lineup.get("players") or {}).get(_norm(player_name))
    if not player or not lineup.get(f"{player.get('side')}_confirmed"):
        return None
    return dict(player)


def _names_match(left, right):
    """Tolerant normalized team-name comparison."""
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    return left.split()[-1] == right.split()[-1]


def get_pitcher_quality(pitcher_id, season):
    """
    Season pitching quality + handedness for a starter.

    Returns {'name','throws','era','k_pct','bb_pct','ip','bf',
             'run_suppression'} where run_suppression >1 means better than a
    league-average pitcher (LEAGUE_AVG['era'] / era), clamped to a sane range.
    """
    cache = f"pitcher_{pitcher_id}_{season}"
    cached = _read_cache(cache)
    if cached is not None:
        return cached
    data = _get(f"people/{pitcher_id}", {
        "hydrate": f"stats(group=pitching,type=season,season={season})",
    })
    people = data.get("people", [])
    if not people:
        return None
    p = people[0]
    throws = p.get("pitchHand", {}).get("code")
    stat = {}
    for grp in p.get("stats", []) or []:
        splits = grp.get("splits", [])
        if splits:
            stat = splits[0].get("stat", {})
            break
    era = _safe_div(stat.get("earnedRuns"), None)  # placeholder, prefer 'era'
    try:
        era = float(stat.get("era")) if stat.get("era") not in (None, "-.--") else None
    except (TypeError, ValueError):
        era = None
    bf = stat.get("battersFaced")
    # Average innings per start (workload/durability). Lets the model weight the
    # starter's quality by how much of the game he actually covers, with the
    # bullpen covering the rest (see build_matchup_features).
    gs = _to_float(stat.get("gamesStarted"))
    ip_innings = _parse_ip(stat.get("inningsPitched"))
    avg_ip = (ip_innings / gs) if (gs and ip_innings is not None) else None
    out = {
        "name": p.get("fullName"),
        "throws": throws,
        "era": era,
        "ip": stat.get("inningsPitched"),
        "avg_ip": avg_ip,
        "bf": bf,
        "k_pct": _safe_div(stat.get("strikeOuts"), bf),
        "bb_pct": _safe_div(stat.get("baseOnBalls"), bf),
        "xera": None,
        "xwoba": None,
        "xba": None,
    }

    # Prefer Savant expected stats (luck-stripped "true" quality) when the
    # pitcher is in the leaderboard; fall back to traditional ERA otherwise.
    try:
        xstats = get_pitcher_expected_stats(season).get(str(pitcher_id))
    except requests.RequestException:
        xstats = None  # Savant unreachable — degrade to ERA-based quality.
    if xstats:
        out["xera"] = xstats.get("xera")
        out["xwoba"] = xstats.get("xwoba")
        out["xba"] = xstats.get("xba")

    # Run-suppression index vs league average (higher = better pitcher).
    # Uses xERA when available (removes BABIP/sequencing luck), else ERA.
    basis = out["xera"] if out["xera"] else era
    if basis and basis > 0:
        out["run_suppression"] = max(0.5, min(2.0, LEAGUE_AVG["era"] / basis))
        out["run_suppression_basis"] = "xera" if out["xera"] else "era"
    else:
        out["run_suppression"] = 1.0
        out["run_suppression_basis"] = "none"
    _write_cache(cache, out)
    return out


def get_team_offense_splits(team_id, season):
    """
    Team offensive quality split by opposing pitcher handedness.

    Returns {'vL': {'ops','k_pct'}, 'vR': {'ops','k_pct'}} (either may be None).
    """
    cache = f"team_splits_{team_id}_{season}"
    cached = _read_cache(cache)
    if cached is not None:
        return cached
    data = _get(f"teams/{team_id}/stats", {
        "season": season, "group": "hitting",
        "stats": "statSplits", "sitCodes": "vl,vr",
    })
    out = {"vL": None, "vR": None}
    for grp in data.get("stats", []):
        for sp in grp.get("splits", []):
            code = sp.get("split", {}).get("code")
            s = sp.get("stat", {})
            try:
                ops = float(s.get("ops")) if s.get("ops") is not None else None
            except (TypeError, ValueError):
                ops = None
            entry = {"ops": ops, "k_pct": _safe_div(s.get("strikeOuts"),
                                                     s.get("plateAppearances"))}
            if code == "vl":
                out["vL"] = entry
            elif code == "vr":
                out["vR"] = entry
    _write_cache(cache, out)
    return out


def get_team_bullpen_quality(team_id, season):
    """
    Bullpen (reliever) run-suppression for a team (Phase 3).

    Returns {'rp_era', 'bullpen_suppression'} where bullpen_suppression > 1
    means a better-than-league bullpen (LEAGUE_AVG['era'] / rp_era), clamped.
    Relievers finish games, so a weak pen inflates totals late.
    """
    cache = f"bullpen_{team_id}_{season}"
    cached = _read_cache(cache, max_age=24 * 3600)
    if cached is not None:
        return cached
    data = _get(f"teams/{team_id}/stats", {
        "season": season, "group": "pitching",
        "stats": "statSplits", "sitCodes": "rp",
    })
    rp_era = None
    for grp in data.get("stats", []):
        for sp in grp.get("splits", []):
            if sp.get("split", {}).get("code") == "rp":
                rp_era = _to_float(sp.get("stat", {}).get("era"))
    out = {"rp_era": rp_era}
    if rp_era and rp_era > 0:
        out["bullpen_suppression"] = max(0.5, min(2.0, LEAGUE_AVG["era"] / rp_era))
    else:
        out["bullpen_suppression"] = 1.0
    _write_cache(cache, out)
    return out


def get_bvp(batter_id, pitcher_id, season=None):
    """
    Batter-vs-pitcher head-to-head history (Phase 4 — weak prior only).

    Returns {'pa','hits','avg','ops'} or None. WARNING: BvP samples are tiny
    and noisy; this has near-zero predictive value beyond the two players'
    overall talent and must only ever be used as a heavily-shrunk prior. It is
    NOT wired into the live projection path by default (its calibration weight
    defaults to 0) precisely for this reason.
    """
    params = {"stats": "vsPlayer", "group": "hitting", "opposingPlayerId": pitcher_id}
    if season:
        params["season"] = season
    data = _get(f"people/{batter_id}", {
        "hydrate": f"stats({','.join(f'{k}={v}' for k, v in params.items())})",
    })
    people = data.get("people", [])
    if not people:
        return None
    for grp in people[0].get("stats", []) or []:
        for sp in grp.get("splits", []):
            s = sp.get("stat", {})
            pa = s.get("plateAppearances")
            if pa:
                return {
                    "pa": pa,
                    "hits": s.get("hits"),
                    "avg": _to_float(s.get("avg")),
                    "ops": _to_float(s.get("ops")),
                }
    return None


def build_matchup_features(home_team, away_team, date, season, team_index=None):
    """
    Assemble Phase 1 starter/opponent features for one game.

    For each side returns the probable starter's quality and the *opposing*
    lineup's offense vs that starter's handedness, plus a single normalized
    ``starter_edge`` in roughly [-1, 1] (home minus away run-suppression,
    scaled). Returns None for any piece that can't be resolved so callers can
    degrade gracefully to the existing team-only model.

    NOTE: this returns raw normalized features only. How much they should move
    a prediction is a calibratable weight fit elsewhere, not decided here.
    """
    if team_index is None:
        team_index = get_team_index(season)

    probables = get_probable_starters(date)
    result = {"home": None, "away": None, "starter_edge": None}

    sides = {"home": home_team, "away": away_team}
    quality = {}
    for side, tname in sides.items():
        pinfo = probables.get(_norm(tname))
        if not pinfo:
            continue
        q = get_pitcher_quality(pinfo["pitcher_id"], season)
        if not q:
            continue
        quality[side] = q
        # Opposing lineup offense vs this starter's hand.
        opp_name = away_team if side == "home" else home_team
        opp = _match_team_id(opp_name, team_index)
        opp_split = None
        if opp:
            splits = get_team_offense_splits(opp["id"], season)
            hand_key = "vL" if q.get("throws") == "L" else "vR"
            opp_split = splits.get(hand_key)
        result[side] = {
            "starter": q,
            "opp_offense_vs_hand": opp_split,
        }

    # Phase 3: bullpen quality per team (independent of starter availability).
    for side, tname in sides.items():
        own = _match_team_id(tname, team_index)
        if not own:
            continue
        try:
            pen = get_team_bullpen_quality(own["id"], season)
        except requests.RequestException:
            pen = None
        if pen:
            if result[side] is None:
                result[side] = {"starter": None, "opp_offense_vs_hand": None}
            result[side]["bullpen"] = pen

    if "home" in quality and "away" in quality:
        # Innings-weighted effective run-prevention: a starter's quality counts
        # in proportion to how deep he goes; the bullpen covers the rest. Falls
        # back to starter-quality-only when avg_ip or bullpen are unavailable
        # (so the signal degrades gracefully, matching the backtest fit).
        def _off_factor(side):
            # Offense the side's staff faces = opposing lineup's OPS vs this
            # starter's hand, relative to league. >1 = tougher offense. Neutral
            # 1.0 when the split is unavailable. Mirrors the fit's offense factor
            # (which uses savant xwOBAcon); both are ~1.0-centered multipliers.
            split = (result.get(side) or {}).get("opp_offense_vs_hand")
            if not split or not split.get("ops"):
                return 1.0
            return max(0.5, min(2.0, split["ops"] / LEAGUE_AVG["ops"]))

        def _eff(side):
            q = quality[side]
            sp = q.get("run_suppression", 1.0)
            pen = (result.get(side) or {}).get("bullpen")
            avg_ip = q.get("avg_ip")
            if pen and avg_ip:
                w = max(0.30, min(0.85, avg_ip / 9.0))
                base = w * sp + (1.0 - w) * pen.get("bullpen_suppression", 1.0)
            else:
                base = sp
            # Two-sided: degrade run-prevention by the offense faced.
            return base / _off_factor(side)
        # Positive => home better than away. tanh keeps it bounded; magnitude of
        # effect is applied by the (calibratable) weight in the analyzer.
        result["starter_edge"] = _tanh(_eff("home") - _eff("away"))

    return result


def _tanh(x):
    import math
    return math.tanh(x)


if __name__ == "__main__":
    # Smoke test against the live API.
    import datetime
    day = os.environ.get("MLB_TEST_DATE", "2025-07-05")
    season = int(day[:4])
    idx = get_team_index(season)
    print(f"teams indexed: {len(idx)}")
    probs = get_probable_starters(day)
    print(f"probable starters on {day}: {len(probs)} teams")
    sample = list(probs.items())[:2]
    for tname, info in sample:
        q = get_pitcher_quality(info["pitcher_id"], season)
        print(f"  {info['name']} ({tname}): throws={q['throws']} "
              f"era={q['era']} xera={q['xera']} xwoba={q['xwoba']} "
              f"k%={q['k_pct'] and round(q['k_pct'],3)} "
              f"rs={q['run_suppression']} (basis={q['run_suppression_basis']})")
