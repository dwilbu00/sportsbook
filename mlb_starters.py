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
from datetime import date as _date, datetime, timedelta, timezone
import io
import json
import math
import os
import random
import sys
import time
import unicodedata

import requests


def _warn(msg):
    """Surface a non-fatal data problem to stderr (fail-visible, not fail-silent).

    Mirrors ``espn_client._warn``. Used where a silent fallback would otherwise
    hide a real defect — e.g. a Savant/StatsAPI team-key mismatch quietly
    disabling the expected-runs ensemble challenger for the affected teams.
    """
    print(f"[mlb_starters] {msg}", file=sys.stderr)


# Baseball Savant returns team abbreviations (in the grouped-by-team CSV's
# ``player_name`` column) that occasionally diverge from the MLB Stats API
# ``abbreviation`` the consumer looks up by. Known divergence: the Athletics
# rebrand — Savant emits ``ATH`` while the Stats API (2024) still returns
# ``OAK``. The mapping is Savant-key -> StatsAPI-abbr; _canonical_team_key
# applies it only when the target actually exists in the season's team index
# (so it can't mis-remap a season where StatsAPI itself uses ``ATH``), and the
# coverage validation in get_expected_runs_team_factors loudly flags any NEW
# divergence that this table doesn't yet cover.
_SAVANT_TO_STATSAPI_ABBR = {"ATH": "OAK"}


def _canonical_team_key(savant_key, statsapi_abbrs):
    """Map a Savant team key into the StatsAPI-abbreviation namespace.

    Self-correcting across seasons: an alias is only applied when the mapped
    abbreviation is present in ``statsapi_abbrs`` for this season AND the raw
    key is not, so it fixes the current divergence without breaking a future
    season where the two sources happen to agree. Unresolved keys are returned
    unchanged so the caller's validation can flag them.
    """
    if savant_key in statsapi_abbrs:
        return savant_key
    alias = _SAVANT_TO_STATSAPI_ABBR.get(savant_key)
    if alias and alias in statsapi_abbrs:
        return alias
    return savant_key


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


def get_expected_runs_team_factors(season, as_of, min_pa=40):
    """Return leakage-safe live team inputs for the expected-runs model.

    The model was validated on league-relative Savant expected-wOBA averages,
    split by opposing pitcher hand for offenses and restricted to relievers for
    bullpens. Savant's aggregate search endpoint produces those same averages
    in small team-level responses, avoiding a runtime pitch-level download.
    Only games before ``as_of`` are included.
    """
    cutoff = _date.fromisoformat(as_of) - timedelta(days=1)
    if cutoff.year < int(season):
        return None

    # v2: team keys are now normalized into the StatsAPI-abbreviation namespace
    # (see below); a stale v1 cache holds raw Savant keys, so bump to invalidate.
    cache = f"savant_expected_runs_teams_v2_{season}_{cutoff.isoformat()}_{min_pa}"
    cached = _read_cache(cache, max_age=24 * 3600)
    if cached is not None:
        return cached

    common = {
        "all": "true",
        "game_date_gt": f"{season}-01-01",
        "game_date_lt": cutoff.isoformat(),
        "group_by": "team",
        "min_pitches": 0,
        "min_results": 0,
        "min_pas": 0,
        "hfGT": "R|",
    }

    def _team_xwoba(extra):
        rows = _get_savant_csv("statcast_search/csv", dict(common, **extra))
        parsed = {}
        for row in rows:
            team = row.get("player_name")
            xwoba = _to_float(row.get("xwoba"))
            pa = _to_float(row.get("pa"))
            if team and xwoba and pa and pa >= min_pa:
                parsed[team] = {"xwoba": xwoba, "pa": pa}
        return parsed

    offense_vs_hand = {
        hand: _team_xwoba({
            "player_type": "batter",
            "pitcher_throws": hand,
        })
        for hand in ("L", "R")
    }
    bullpens = _team_xwoba({
        "player_type": "pitcher",
        "position": "RP",
    })

    # Normalize Savant's team keys into the StatsAPI-abbreviation namespace the
    # consumer (_expected_offense / _expected_staff) looks up by. Without this a
    # divergent key (e.g. Savant 'ATH' vs StatsAPI 'OAK') silently yields no
    # match -> expected_runs.complete=False -> the ensemble challenger is dropped
    # for that team with zero visibility. A team-index hiccup just skips
    # normalization (raw keys still match ~29/30) rather than disabling the model.
    try:
        team_index = get_team_index(season)
    except (OSError, ValueError, requests.RequestException) as exc:
        _warn(f"team index unavailable for season {season}: "
              f"{type(exc).__name__}: {exc}; skipping Savant team-key normalization")
        team_index = None
    statsapi_abbrs = {info.get("abbr") for info in (team_index or {}).values()
                      if info.get("abbr")}
    if statsapi_abbrs:
        def _norm_keys(d):
            return {_canonical_team_key(k, statsapi_abbrs): v
                    for k, v in d.items()}
        offense_vs_hand = {hand: _norm_keys(rows)
                           for hand, rows in offense_vs_hand.items()}
        bullpens = _norm_keys(bullpens)
        # Fail-visible coverage check: an unmapped Savant key (likely a new team
        # rename not yet in _SAVANT_TO_STATSAPI_ABBR) or a StatsAPI team with no
        # Savant coverage means the challenger is disabled for those teams. Warn
        # loudly instead of failing silently as before.
        offense_keys = {k for rows in offense_vs_hand.values() for k in rows}
        unmapped = sorted(offense_keys - statsapi_abbrs)
        missing = sorted(statsapi_abbrs - offense_keys)
        if unmapped or missing:
            _warn(f"expected-runs team-key coverage gap for season {season}: "
                  f"Savant keys outside the StatsAPI namespace={unmapped}; "
                  f"StatsAPI teams with no Savant offense data={missing}. The "
                  f"ensemble challenger is disabled for these teams — extend "
                  f"_SAVANT_TO_STATSAPI_ABBR if 'unmapped' is a rename.")

    offense_rows = [
        row
        for hand_rows in offense_vs_hand.values()
        for row in hand_rows.values()
    ]
    if not offense_rows or not bullpens:
        return None

    league_xwoba = (
        sum(row["xwoba"] * row["pa"] for row in offense_rows)
        / sum(row["pa"] for row in offense_rows)
    )
    bullpen_rows = list(bullpens.values())
    league_bullpen_xwoba = (
        sum(row["xwoba"] * row["pa"] for row in bullpen_rows)
        / sum(row["pa"] for row in bullpen_rows)
    )
    out = {
        "league_xwoba": league_xwoba,
        "league_bullpen_xwoba": league_bullpen_xwoba,
        "offense_vs_hand": {
            hand: {team: row["xwoba"] for team, row in rows.items()}
            for hand, rows in offense_vs_hand.items()
        },
        "bullpen_xwoba": {
            team: row["xwoba"] for team, row in bullpens.items()
        },
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


def player_start_status(prop_key, player_name, home_team, away_team,
                        confirmed_lineup, probable_starters, season=None):
    """Tri-state pre-game availability for a prop's player.

    Returns one of:
      "out"     -- high-confidence NOT playing this game in the prop's role. The
                   bet is dead; the caller suppresses the recommendation AND skips
                   the calibration log (a label that would never resolve).
      "in"      -- confirmed present (in the posted lineup / announced probable).
      "unknown" -- not yet determinable -> the caller FAILS OPEN (current behavior).

    Batter props gate on the confirmed batting lineup; pitcher props gate on the
    announced probable starter. The gate only ever ACTS on a confident "out";
    missing data, non-MLB, or any error all degrade to "unknown". A positive
    ``season`` lets the batter/pitcher "out" arms confirm identity via
    ``find_player_id`` before ruling a player out, guarding against name-spelling
    drift between the odds feed and the Stats API; pass ``season=None`` to use the
    name-only logic (unit tests).
    """
    try:
        if not player_name:
            return "unknown"
        key = _norm(player_name)

        # --- Pitcher prop: announced probable starters (available all day). ---
        if str(prop_key or "").startswith("pitcher_"):
            probs = probable_starters or {}
            sides = [probs.get(_norm(home_team)), probs.get(_norm(away_team))]
            announced = [s for s in sides if s and s.get("pitcher_id")]
            if not announced:
                return "unknown"
            pid = None
            if season is not None:
                resolved = find_player_id(player_name, season)
                if resolved:
                    pid = resolved[0]
            for s in announced:
                if _norm(s.get("name")) == key or (
                        pid is not None and str(s.get("pitcher_id")) == str(pid)):
                    return "in"
            # Not an announced starter. Only OUT when BOTH sides are announced
            # AND we positively resolved his id (guards a false out on a name
            # the probable feed spells differently); a TBD side stays unknowable.
            if len(announced) == 2 and pid is not None:
                return "out"
            return "unknown"

        # --- Batter prop: confirmed batting lineup. ---
        lineup = confirmed_lineup or {}
        players = lineup.get("players") or {}
        rec = players.get(key)
        if rec:
            side = rec.get("side")
            return "in" if lineup.get(f"{side}_confirmed") else "unknown"
        # Absent by name. This can only become "out" once BOTH 9-man lineups are
        # posted (he's in neither); a single posted side can't rule him out.
        if not (lineup.get("home_confirmed") and lineup.get("away_confirmed")):
            return "unknown"
        # Both lineups posted -> confirm his identity by id before ruling OUT
        # (the odds feed may spell him differently than the Stats API).
        if season is not None:
            resolved = find_player_id(player_name, season)
            if resolved:
                pid = resolved[0]
                for p in players.values():
                    if str(p.get("player_id")) == str(pid):
                        side = p.get("side")
                        return "in" if lineup.get(f"{side}_confirmed") else "unknown"
        return "out"
    except Exception:
        return "unknown"


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

        def _staff_factor(side):
            q = quality[side]
            sp = q.get("run_suppression", 1.0)
            pen = (result.get(side) or {}).get("bullpen")
            avg_ip = q.get("avg_ip")
            if pen and avg_ip:
                w = max(0.30, min(0.85, avg_ip / 9.0))
                return (w * sp
                        + (1.0 - w) * pen.get("bullpen_suppression", 1.0))
            return sp

        def _eff(side):
            # Two-sided: degrade run-prevention by the offense faced.
            return _staff_factor(side) / _off_factor(side)
        # Positive => home better than away. tanh keeps it bounded; magnitude of
        # effect is applied by the (calibratable) weight in the analyzer.
        result["starter_edge"] = _tanh(_eff("home") - _eff("away"))

        # Build the spread-only challenger from the same Savant expected-wOBA
        # scale used in its historical fit. Keep this separate from _eff so the
        # existing starter adjustment, moneyline, and totals paths are unchanged.
        try:
            expected_inputs = get_expected_runs_team_factors(season, date)
        except (OSError, ValueError, requests.RequestException):
            expected_inputs = None

        home_info = _match_team_id(home_team, team_index)
        away_info = _match_team_id(away_team, team_index)
        home_abbr = home_info.get("abbr") if home_info else None
        away_abbr = away_info.get("abbr") if away_info else None

        def _expected_offense(abbr, opposing_hand):
            if not expected_inputs or not abbr or opposing_hand not in ("L", "R"):
                return None
            xwoba = ((expected_inputs.get("offense_vs_hand") or {})
                     .get(opposing_hand, {}).get(abbr))
            league = expected_inputs.get("league_xwoba")
            if not xwoba or not league:
                return None
            return max(0.5, min(2.0, xwoba / league))

        def _expected_staff(side, abbr):
            if not expected_inputs or not abbr:
                return None
            league = expected_inputs.get("league_xwoba")
            starter_xwoba = quality[side].get("xwoba")
            if not league or not starter_xwoba:
                return None
            starter = max(0.5, min(2.0, league / starter_xwoba))
            bullpen_xwoba = ((expected_inputs.get("bullpen_xwoba") or {})
                              .get(abbr))
            bullpen_league = expected_inputs.get("league_bullpen_xwoba")
            avg_ip = quality[side].get("avg_ip")
            if bullpen_xwoba and bullpen_league and avg_ip:
                weight = max(0.30, min(0.85, avg_ip / 9.0))
                bullpen = max(
                    0.5, min(2.0, bullpen_league / bullpen_xwoba))
                return weight * starter + (1.0 - weight) * bullpen
            return starter

        home_offense = _expected_offense(
            home_abbr, quality["away"].get("throws"))
        away_offense = _expected_offense(
            away_abbr, quality["home"].get("throws"))
        home_staff = _expected_staff("home", home_abbr)
        away_staff = _expected_staff("away", away_abbr)
        result["expected_runs"] = {
            "complete": all(value is not None for value in (
                home_offense, away_offense, home_staff, away_staff)),
            "home_offense_factor": home_offense,
            "away_offense_factor": away_offense,
            "home_staff_suppression": home_staff,
            "away_staff_suppression": away_staff,
        }

    return result


def _tanh(x):
    import math
    return math.tanh(x)


# ──────────────────────────────────────────────────────────────────────────────
# Hard-ID outcome resolution (gamePk) — used by recalibration.resolve_pending_outcomes
# ──────────────────────────────────────────────────────────────────────────────
# The prediction log carries an Odds API event id (not a gamePk) plus the game's
# commence_time. To grade a bet against the *right* game — including the correct
# leg of a doubleheader — we resolve the player -> MLBAM id -> season gameLog ->
# gamePk, and join gamePk to that date's schedule to pick the game whose start is
# nearest commence_time. This also grades pitcher props, which ESPN's synthesized
# (dateless) pitcher gamelogs cannot.

def _parse_utc(value):
    """Parse an ISO timestamp as tz-aware UTC (coercing naive -> UTC), or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _candidate_dates(game_date):
    """[game_date, game_date-1, game_date+1] for UTC/local slippage tolerance."""
    day = str(game_date)[:10]
    out = [day]
    try:
        base = _date.fromisoformat(day)
    except (TypeError, ValueError):
        return out
    for delta in (-1, 1):
        out.append((base + timedelta(days=delta)).isoformat())
    return out


# Read-time freshness for a schedule that still contains a non-final game. Once
# every game on a date is Final the schedule is immutable and cached a full day;
# while any game is scheduled/live we only trust the cache briefly so the
# ``status`` used by the outcome-resolution final gate reflects reality.
_SCHEDULE_LIVE_TTL = 900        # 15 min while any game is not Final
_SCHEDULE_FINAL_TTL = 24 * 3600  # 1 day once all games are Final


# statsapi can report abstractGameState "Final" for a game that never truly
# completed: a postponed/suspended/cancelled game may surface as "Final" with a
# 0-0 or partial box score, which would wrongly grade bets (a rained-out game's
# over bets settling as WIN off a bogus line). A rain-SHORTENED but OFFICIAL game
# reads detailedState "Completed Early" and MUST still grade, so exclude only the
# genuine non-completion states.
_NON_FINAL_DETAILED = ("postpon", "suspend", "cancel")


def _is_genuine_final(info):
    """True when a schedule-index entry is a real, gradable completion: abstract
    state "Final" AND a detailedState that isn't postponed/suspended/cancelled.
    A missing detailedState (older cached index, pre-upgrade) trusts
    abstractGameState so stale caches keep resolving."""
    info = info or {}
    if info.get("status") != "Final":
        return False
    detailed = str(info.get("detailedState") or "").lower()
    return not any(bad in detailed for bad in _NON_FINAL_DETAILED)


def _all_final(index):
    """True when every game in a schedule index is a genuine completion (schedule
    is static). A postponed/suspended game reads as not-final so its date keeps
    refreshing on the short TTL and picks up the eventual makeup."""
    return bool(index) and all(
        _is_genuine_final(info) for info in index.values())


def get_schedule_index(date):
    """{str(gamePk): {gameDate, gameNumber, doubleHeader, home, away, status,
    detailedState}} for one calendar date (YYYY-MM-DD). gamePk is the hard game
    id; gameDate is the UTC start used to disambiguate doubleheaders against a
    forecast's commence_time; ``status`` is the statsapi abstractGameState
    ('Final', 'Live', 'Preview') and ``detailedState`` the finer status
    ('Postponed', 'Suspended', 'Completed Early', …) — together they gate outcome
    resolution to genuine completions via ``_is_genuine_final`` (a postponed game
    can report abstractGameState 'Final').

    Adaptive cache: a date whose games are all Final is static and cached for a
    day; a date with any non-final game refreshes every ~15 min so live status
    doesn't go stale (a wrong 'Final' would let a live game be graded)."""
    cache = f"schedule_index_{date}"
    # Trust the long cache only when it says every game is Final; otherwise fall
    # back to the short freshness window (and refetch when it too has expired).
    cached = _read_cache(cache, max_age=_SCHEDULE_FINAL_TTL)
    if cached is not None and _all_final(cached):
        return cached
    fresh = _read_cache(cache, max_age=_SCHEDULE_LIVE_TTL)
    if fresh is not None:
        return fresh
    data = _get("schedule", {"sportId": 1, "date": date})
    out = {}
    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            pk = g.get("gamePk")
            if pk is None:
                continue
            teams = g.get("teams", {}) or {}
            out[str(pk)] = {
                "gameDate": g.get("gameDate"),
                "gameNumber": g.get("gameNumber"),
                "doubleHeader": g.get("doubleHeader"),
                "home": ((teams.get("home") or {}).get("team") or {}).get("name"),
                "away": ((teams.get("away") or {}).get("team") or {}).get("name"),
                "status": (g.get("status") or {}).get("abstractGameState"),
                "detailedState": (g.get("status") or {}).get("detailedState"),
            }
    _write_cache(cache, out)
    return out


_PLAYER_INDEX_CACHE = {}  # season -> {norm_name: [(mlbam_id, is_pitcher), ...]}


def _player_index(season):
    """{normalized_full_name: [(id, is_pitcher), ...]} for a season's players."""
    cached = _PLAYER_INDEX_CACHE.get(season)
    if cached is not None:
        return cached
    disk = _read_cache(f"players_index_{season}", max_age=7 * 24 * 3600)
    if disk is not None:
        index = {k: [tuple(v) for v in vs] for k, vs in disk.items()}
        _PLAYER_INDEX_CACHE[season] = index
        return index
    data = _get("sports/1/players", {"season": season})
    index = {}
    for p in data.get("people", []) or []:
        pid = p.get("id")
        full = p.get("fullName")
        if not pid or not full:
            continue
        pos = p.get("primaryPosition") or {}
        is_pitcher = (pos.get("abbreviation") == "P"
                      or pos.get("type") == "Pitcher"
                      or str(pos.get("code")) == "1")
        index.setdefault(_norm(full), []).append((pid, is_pitcher))
    _write_cache(f"players_index_{season}",
                 {k: [[pid, isp] for pid, isp in vs] for k, vs in index.items()})
    _PLAYER_INDEX_CACHE[season] = index
    return index


_PITCHER_BY_ID_CACHE = {}  # season -> {str(mlbam_id): is_pitcher}


def _is_pitcher_index(season):
    """{str(mlbam_id): is_pitcher} for a season, inverted from _player_index (the
    same cached statsapi payload — no extra fetch)."""
    cached = _PITCHER_BY_ID_CACHE.get(season)
    if cached is not None:
        return cached
    idx = {}
    for matches in _player_index(season).values():
        for pid, is_pitcher in matches:
            idx[str(pid)] = is_pitcher
    _PITCHER_BY_ID_CACHE[season] = idx
    return idx


def _player_id_map():
    """Lazily import the SFBB id-map module, or None if unavailable (missing
    SQLAlchemy / import error). Guarded like the other optional SQL backends so the
    pricing core keeps working without it."""
    try:
        import player_id_map
        return player_id_map
    except Exception:                          # pragma: no cover - import guard
        return None


def _resolve_is_pitcher(mid, season, row):
    """is_pitcher for an MLBAM id. The statsapi season roster is authoritative (so
    two-way players like Ohtani keep their statsapi position); the SFBB map's ALLPOS
    is used only when the id isn't in the roster (e.g. a mid-season callup absent
    from the cached payload)."""
    idx = _is_pitcher_index(season)
    if str(mid) in idx:
        return idx[str(mid)]
    allpos = ((row or {}).get("allpos") or "").upper().replace(",", "/")
    return "P" in [p.strip() for p in allpos.split("/")]


def find_player_id(name, season):
    """(mlbam_id, is_pitcher) for a UNIQUE exact full-name match, else None.

    Resolves via the SFBB player id-map FIRST — it disambiguates namesakes the
    statsapi unique-exact match drops (preferring the single active player) and
    folds accents — then falls back to the statsapi season roster's unique-exact
    name match. is_pitcher stays statsapi-authoritative. The forecast row carries
    no team, so a STILL-ambiguous name (two active namesakes) is skipped rather
    than risk binding a prop to the wrong player and poisoning the fit."""
    pim = _player_id_map()
    if pim is not None:
        mid = pim.mlb_id_for_name(name)
        if mid:
            return (mid, _resolve_is_pitcher(mid, season, pim.get_row(name)))
    matches = _player_index(season).get(_norm(name))
    if not matches or len(matches) != 1:
        return None
    return matches[0]


def _player_gamelog_splits(pid, group, season, max_age=CACHE_MAX_AGE):
    """Season gameLog splits for a player + stat group ('hitting'/'pitching').

    ``max_age`` overrides the cache freshness: the outcome-resolution path passes
    0 so it always reads a FRESH gamelog. Otherwise a partial stat cached during
    a live game (default 1h TTL) could be read moments after the game goes final
    and grade a bet off an incomplete line."""
    cache = f"gamelog_{pid}_{group}_{season}"
    cached = _read_cache(cache, max_age=max_age)
    if cached is not None:
        return cached
    data = _get(f"people/{pid}/stats",
                {"stats": "gameLog", "group": group, "season": season})
    splits = []
    for grp in data.get("stats", []) or []:
        splits.extend(grp.get("splits", []) or [])
    _write_cache(cache, splits)
    return splits


def _f(v):
    """Float-coerce a StatsAPI stat value; None on missing/garbage."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_pitcher_gamelog(player_name, season=None):
    """TRUE per-game pitcher log from MLB StatsAPI, shaped like
    ``espn_client.get_athlete_gamelog`` output (newest-first).

    Replaces the synthesized-from-season-splits ``espn_client.get_pitcher_stats``
    for the three fetched pitcher props: each row carries a real ``game_date`` and
    real per-game variance, so recency weighting, as-of leakage slicing, and
    real-line calibration all work for pitchers exactly as they do for batters.

    Returns ``[]`` (so callers fall back to the synthesized source) when the name
    is not a UNIQUE exact match, resolves to a non-pitcher, or StatsAPI yields no
    usable games — never a wrong-player bind.

    Row keys mirror what the pipeline reads (``PROP_STAT_MAP`` / ``_PITCHER_STATS``):
      * ``IP`` — innings pitched as a base-3 FLOAT (6.1 = 6 IP + 1 out), the same
        shape ``get_pitcher_stats`` emits and ``espn_client.ip_to_outs`` expects.
        StatsAPI returns ``inningsPitched`` as a STRING ("6.1"); we float-coerce it
        WITHOUT decimalizing (``float("6.1") == 6.1``, NOT ``_parse_ip`` which would
        yield 6.33). ``ip_to_outs`` then reads the thirds correctly.
      * ``K`` / ``ER`` — strikeouts / earned runs (floats).
    Meta keys: ``game_date`` (ISO date), ``opponent`` (full club name, matches ESPN
    ``team_defense`` via the tolerant matcher), ``is_home``, ``completed``. ``H`` /
    ``BB`` are carried for callers/tests but dropped by the SQL store (no columns).
    ``team_id`` is intentionally omitted — StatsAPI ids are MLBAM, not ESPN, so a
    stored id would mis-key ESPN park/team lookups (synth had none either).
    """
    season = season or datetime.now(timezone.utc).year
    found = find_player_id(player_name, season)
    if not found:
        return []
    pid, is_pitcher = found
    if not is_pitcher:                       # never bind a pitcher prop to a batter
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = []
    for sp in _player_gamelog_splits(pid, "pitching", season):
        stat = sp.get("stat") or {}
        ip_raw = stat.get("inningsPitched")  # StatsAPI base-3 STRING ("6.1")
        if ip_raw is None:                   # no appearance -> not a game row
            continue
        ip = _f(ip_raw)                      # -> base-3 FLOAT (6.1), NOT decimalized
        if ip is None:                       # unparseable IP -> skip (not a game)
            continue
        d = sp.get("date")
        is_home = sp.get("isHome")
        rows.append({
            "IP": ip,
            "K": _f(stat.get("strikeOuts")),
            "ER": _f(stat.get("earnedRuns")),
            "H": _f(stat.get("hits")),
            "BB": _f(stat.get("baseOnBalls")),
            "game_date": d,
            "opponent": (sp.get("opponent") or {}).get("name"),
            "is_home": bool(is_home) if is_home is not None else None,
            # A same-day line can be an in-progress partial; mark it not-complete so
            # the clobber guard (_completed_count) never counts it as a final game.
            "completed": bool(d) and d[:10] != today,
            "_gamePk": (sp.get("game") or {}).get("gamePk"),  # local sort tiebreak
        })

    # Newest-first to match the SQL store's id-order invariant and the
    # ``prior_games = gamelog[idx+1:]`` as-of slice; gamePk breaks doubleheader ties
    # deterministically.
    rows.sort(key=lambda r: (r.get("game_date") or "", r.get("_gamePk") or 0),
              reverse=True)
    for r in rows:
        r.pop("_gamePk", None)
    return rows


def _pitcher_gamelog_or_synth(league, athlete_id, player_name, season=None):
    """Fail-open pitcher gamelog: TRUE StatsAPI log -> synthesized ESPN splits -> [].

    When ``player_name`` is falsy the StatsAPI path is skipped entirely and this is
    byte-identical to the ``espn_client.get_pitcher_stats(league, athlete_id,
    season)`` call it replaces at each fallback chokepoint — so callers that don't
    thread a name see no behavior change. ``espn_client`` is imported lazily to
    keep this leaf module import-cycle free.

    The StatsAPI call is guarded: if the real log raises (network error / HTTP
    5xx-4xx), we fall through to the synth splits rather than propagating — the
    same fail-open contract ``espn_client.get_pitcher_stats`` provided before.
    """
    if player_name:
        try:
            real = get_pitcher_gamelog(player_name, season)
        except Exception:
            real = None                       # StatsAPI down -> fall through to synth
        if real:
            return real
    try:
        import espn_client
        return espn_client.get_pitcher_stats(league, athlete_id,
                                             season=season) or []
    except Exception:
        return []


# Sentinel: the player's game was located but is not yet Final. Distinct from
# None (which means "couldn't resolve" and permits the ESPN fallback), it tells
# the caller to keep the bet PENDING and never grade off a live/partial stat.
GAME_NOT_FINAL = object()


def resolve_player_game_stat(name, commence_time, game_date, group, stat_key,
                             season):
    """Resolve one player's actual stat for a specific game via the gamePk hard
    ID, disambiguating doubleheaders by commence_time. Returns a float; None when
    the player/game/stat can't be resolved unambiguously (caller falls back to
    the ESPN path); or ``GAME_NOT_FINAL`` when the bet's game exists but is still
    in progress (caller keeps it pending). Position group must match (a pitching
    prop only binds to a pitcher) to avoid same-name cross-position mismatches."""
    found = find_player_id(name, season)
    if not found:
        return None
    pid, is_pitcher = found
    if group == "pitching" and not is_pitcher:
        return None
    if group == "hitting" and is_pitcher:
        return None

    # Fresh read (max_age=0): a gamelog cached during live play holds a partial
    # line, so never trust the cache when we may be about to grade a bet.
    by_pk = {}
    for sp in _player_gamelog_splits(pid, group, season, max_age=0):
        pk = (sp.get("game") or {}).get("gamePk") or sp.get("gamePk")
        if pk is None:
            continue
        stat = sp.get("stat") or {}
        if stat_key not in stat:
            continue
        try:
            by_pk[str(pk)] = float(stat[stat_key])
        except (TypeError, ValueError):
            continue
    if not by_pk:
        return None

    target = _parse_utc(commence_time)
    # Gather the player's games across the forecast date AND adjacent days. The
    # stored game_date is the UTC date of first pitch (props logs commence[:10]),
    # which for late US games is one day AHEAD of the schedule's official/local
    # date — so the true game can live under game_date-1. Collect every candidate
    # and choose by nearest scheduled start, never first-date-wins (which would
    # bind an everyday hitter to the following day's game).
    # One entry per physical gamePk. A postponed game keeps its ORIGINAL gamePk
    # when it's made up, so the same pk surfaces under both its original date
    # (detailedState 'Postponed') and its makeup date (genuine Final). Dedup by pk
    # but PREFER the genuine-final occurrence — otherwise the earlier 'Postponed'
    # entry wins the dedup and a made-up game strands forever (never grades, and
    # never voids either, since is_confirmed_dnp sees the pk still in the schedule).
    by_pk_cand = {}  # gamePk -> (scheduled_start_dt_or_None, info)
    for d in _candidate_dates(game_date):
        for pk, info in get_schedule_index(d).items():
            if pk not in by_pk:
                continue
            existing = by_pk_cand.get(pk)
            if existing is None or (not _is_genuine_final(existing[1])
                                    and _is_genuine_final(info)):
                by_pk_cand[pk] = (_parse_utc(info.get("gameDate")), info)
    if not by_pk_cand:
        return None
    candidates = [(pk, gdt, info) for pk, (gdt, info) in by_pk_cand.items()]

    # Identify WHICH physical game the bet is on (nearest scheduled start to
    # commence_time), THEN gate on that game's status. Picking the game first is
    # what stops a doubleheader's already-final leg from grading a bet on the
    # still-live leg.
    if len(candidates) == 1:
        pk, _, info = candidates[0]
    else:
        # Multiple nearby games (doubleheader or date slippage): without a
        # commence_time we can't choose safely.
        if target is None:
            return None
        best = None  # (delta_seconds, gamePk, info)
        for cand_pk, gdt, cand_info in candidates:
            if gdt is None:
                continue
            delta = abs((gdt - target).total_seconds())
            if best is None or delta < best[0]:
                best = (delta, cand_pk, cand_info)
        if best is None:
            return None
        _, pk, info = best

    # A suspended game carries a PARTIAL box score with the player's gamePk (so it
    # reaches here) yet reports abstractGameState "Final"; _is_genuine_final also
    # rejects postponed/cancelled. Keep the bet pending rather than grade a bogus
    # line — DK voids these, and the stale-DNP sweep clears a permanent no-show.
    if not _is_genuine_final(info):
        return GAME_NOT_FINAL
    return by_pk[pk]


def is_confirmed_dnp(name, commence_time, game_date, group, season,
                     max_age=3600):
    """True when the player HAS a season game log but no game on the forecast
    date — a scratch/DNP, so a prop logged for that game is permanently
    unresolvable. Returns False when we can't be sure — unknown/ambiguous player,
    a position-group mismatch, an EMPTY log (possible data outage, keep retrying),
    or a matching game DOES exist (not a DNP). Uses a cached gamelog read by
    default since the resolver typically fetched it fresh in the same pass."""
    found = find_player_id(name, season)
    if not found:
        return False
    pid, is_pitcher = found
    if (group == "pitching" and not is_pitcher) or (group == "hitting"
                                                    and is_pitcher):
        return False
    pks = set()
    for sp in _player_gamelog_splits(pid, group, season, max_age=max_age):
        pk = (sp.get("game") or {}).get("gamePk") or sp.get("gamePk")
        if pk is not None:
            pks.add(str(pk))
    if not pks:
        return False   # no log at all -> possible data outage, keep retrying
    for d in _candidate_dates(game_date):
        for pk in get_schedule_index(d):
            if pk in pks:
                return False   # a matching game exists -> not a DNP
    return True   # has games this season, but none on the forecast date -> DNP


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
