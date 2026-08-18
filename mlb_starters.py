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
import threading
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
# Immutable historical data (as-of/past-dated pulls, e.g. a starter's line through
# a bygone date) never changes — cache it ~forever so backtests don't re-fetch the
# same ~18k byDateRange responses on every (or every hour of a) run.
PERMANENT_CACHE_AGE = 3650 * 24 * 3600  # ~10 years
_COVERAGE_WARNED = set()  # (season, unmapped, missing) → expected-runs gap warned once

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
# Expected plate appearances per lineup slot (9-inning game): the top of the order
# bats ~20% more than the bottom, so lineup-offense is a PA-weighted mean of the 9
# batters' OPS, not a flat mean (leadoff → 9-hole).
_SLOT_PA_WEIGHTS = (4.65, 4.54, 4.43, 4.32, 4.21, 4.10, 3.99, 3.88, 3.77)

# ── §3.1 GameContext constants ──────────────────────────────────────────────
# One shared, cached, leakage-safe run/hits environment per game (see
# build_game_context). These are conservative, hand-set constants (the small caps
# + the confirmation gate's 2-fold check are the backstops), noted fit-elsewhere
# like LEAGUE_AVG. NONE of the object's run_pmf/total/team_hits fields feed the
# confirmation gate — only gc_factor (a per-batter MEAN multiplier) does, and it
# depends solely on the leakage-safe xwOBA offense/bullpen factors below.
GC_EPS = 0.55            # runs->hits (and run-index->gc) elasticity (hits < runs vol)
GC_STARTER_SHARE = 0.62  # opposing-staff PA share owned by the STARTER, which is
                         #   pinned to neutral 1.0 here (matchup_mult owns the
                         #   per-batter starter log5 — we must not double-count it)
GC_RUN_CAP = 0.10        # hard bound on the raw gc_factor: [1-CAP, 1+CAP]
LEAGUE_RUNS_PER_GAME = 4.30   # object mu_runs base only (does NOT feed gc_factor)
LEAGUE_HITS_PER_GAME = 8.30   # object team_hits_mean base only (does NOT feed gc_factor)

_GC_CACHE = {}                # (home_abbr, away_abbr, date) -> (built_at, context)
_GC_LIVE_TTL = 20 * 60        # incomplete context: re-build this often
_GC_FINAL_TTL = 24 * 3600     # complete context: trust a day (inputs are 24h-cached)


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
    # Atomic write (temp in the same dir + os.replace) so a concurrent reader --
    # or a second writer racing the same file from a Phase-2 pool worker -- never
    # sees a half-written/truncated JSON blob. os.replace is atomic on the same
    # filesystem on both POSIX and Windows.
    _ensure_cache_dir()
    path = _cache_path(name)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"cached_at": time.time(), "data": data}, f)
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


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


def _mlb_warehouse_offense_enabled():
    """ODI_MLB_WAREHOUSE_OFFENSE: derive the expected-runs challenger's team OFFENSE
    factors from the statcast_pitch warehouse instead of the live Savant HTTP endpoint.
    OFF (default, unset) = byte-identical Savant path. Mirrors the other ODI_MLB_* env
    gates; promoted from st.secrets at boot in app.py."""
    return os.environ.get("ODI_MLB_WAREHOUSE_OFFENSE", "").strip().lower() in (
        "1", "true", "on", "yes")


def _finalize_offense(raw, statsapi_abbrs):
    """{"L"/"R": {team: {"xwoba", "pa"}}} (already min_pa-filtered) -> the offense
    factors dict (Savant->StatsAPI-abbr normalized, PA-weighted league_xwoba, empty
    bullpen), or None if no team clears the bar. SHARED by the live warehouse path and
    the offline precompute so both emit byte-identical output."""
    d = {"L": raw.get("L") or {}, "R": raw.get("R") or {}}
    if not (d["L"] or d["R"]):
        return None
    if statsapi_abbrs:
        d = {hand: {_canonical_team_key(k, statsapi_abbrs): v for k, v in hd.items()}
             for hand, hd in d.items()}
    offense_rows = [v for hd in d.values() for v in hd.values()]
    total_pa = sum(r["pa"] for r in offense_rows)
    if not total_pa:
        return None
    league_xwoba = sum(r["xwoba"] * r["pa"] for r in offense_rows) / total_pa
    return {
        "league_xwoba": league_xwoba,
        "league_bullpen_xwoba": None,      # v1: warehouse bullpen not derived yet
        "offense_vs_hand": {hand: {t: r["xwoba"] for t, r in hd.items()}
                            for hand, hd in d.items()},
        "bullpen_xwoba": {},               # empty -> _expected_staff uses starter-only
    }


def _warn_offense_coverage(season, result, statsapi_abbrs):
    """Fail-VISIBLE coverage warning (mirrors the Savant path): a warehouse team key
    outside the StatsAPI namespace silently disables the challenger for that team.
    Deduped per unique gap with a ("wh", ...) signature."""
    offense_keys = {k for hd in result["offense_vs_hand"].values() for k in hd}
    unmapped = sorted(offense_keys - statsapi_abbrs)
    missing = sorted(statsapi_abbrs - offense_keys)
    _sig = ("wh", int(season), tuple(unmapped), tuple(missing))
    if (unmapped or missing) and _sig not in _COVERAGE_WARNED:
        _COVERAGE_WARNED.add(_sig)
        _warn(f"warehouse expected-runs team-key coverage gap for season {season}: "
              f"warehouse keys outside the StatsAPI namespace={unmapped}; "
              f"StatsAPI teams with no warehouse offense data={missing}. The "
              f"ensemble challenger is disabled for these teams — extend "
              f"_SAVANT_TO_STATSAPI_ABBR if 'unmapped' is a rename.")


def precompute_offense_cache(seasons, min_pa=40, chunk_days=10, verbose=True):
    """One-pass warehouse team-offense precompute for FAST backtests. Chunk-reads
    statcast_pitch in small, timeout-safe date windows (a plain SELECT, NOT a growing
    per-cutoff GROUP BY), maintains a running per-(team, hand) cumulative in memory, and
    writes the wh_expected_runs_teams_v1_* cache file for EVERY cutoff date — so an odds
    backtest with ODI_MLB_WAREHOUSE_OFFENSE=1 reads all team offense from disk with ZERO
    per-cutoff SQL. Replaces the ~500 growing GROUP BY calls that time out on a remote /
    throttled Azure SQL. Uses _finalize_offense so each snapshot is byte-identical to
    what _warehouse_team_factors would compute live. Idempotent (overwrites). Returns
    the number of cache files written. Run ONCE before the backtest."""
    try:
        import savant_history as sh
        import db_store
        from sqlalchemy import select
    except Exception:
        return 0
    if not sh.enabled():
        if verbose:
            print("  [offense-precompute] SQL not configured — nothing to do.")
        return 0
    sp = sh.statcast_pitch
    written = 0
    for _season in seasons:
        season = int(_season)
        s0, s1 = _date(season, 1, 1), _date(season, 12, 31)
        try:
            team_index = get_team_index(season)
            statsapi_abbrs = {info.get("abbr") for info in (team_index or {}).values()
                              if info.get("abbr")}
        except Exception:
            statsapi_abbrs = set()
        # 1) chunk-read the whole season into {date -> [(team, hand, xwoba)]}.
        date_rows, chunks, failed = {}, 0, 0
        d = s0
        while d <= s1:
            hi = min(d + timedelta(days=chunk_days), s1 + timedelta(days=1))
            stmt = select(sp.c.game_date, sp.c.batting_team, sp.c.p_throws,
                          sp.c.xwoba).where(
                (sp.c.xwoba.isnot(None)) & (sp.c.batting_team.isnot(None))
                & (sp.c.p_throws.in_(("L", "R")))
                & (sp.c.game_date >= d.isoformat())
                & (sp.c.game_date < hi.isoformat()))
            rows = None
            for _attempt in range(4):
                try:
                    with db_store.get_engine().connect() as conn:
                        rows = conn.execute(stmt).all()
                    break
                except Exception:
                    rows = None
                    time.sleep(1.0)
            chunks += 1
            if rows is None:
                failed += 1
                if verbose:
                    print(f"  [offense-precompute] {season}: chunk {d}..{hi} FAILED "
                          f"after retries (cache will be incomplete).")
            for gd, team, hand, xw in (rows or []):
                date_rows.setdefault(str(gd)[:10], []).append(
                    (str(team), hand, float(xw)))
            d = hi
        # 2) walk every calendar date, accumulate, snapshot -> cache[cutoff=date].
        cum, cur, season_written = {}, s0, 0
        while cur <= s1:
            ds = cur.isoformat()
            for team, hand, xw in date_rows.get(ds, []):
                acc = cum.get((team, hand))
                if acc is None:
                    cum[(team, hand)] = [xw, 1]
                else:
                    acc[0] += xw
                    acc[1] += 1
            raw = {"L": {}, "R": {}}
            for (team, hand), (ssum, n) in cum.items():
                if hand in ("L", "R") and n >= min_pa:
                    raw[hand][team] = {"xwoba": ssum / n, "pa": n}
            out = _finalize_offense(raw, statsapi_abbrs)
            if out is not None:
                _write_cache(f"wh_expected_runs_teams_v1_{season}_{ds}_{min_pa}", out)
                season_written += 1
            cur += timedelta(days=1)
        written += season_written
        if verbose:
            print(f"  [offense-precompute] {season}: {season_written} cache files "
                  f"({chunks} chunk reads, {failed} failed).")
    if verbose:
        print(f"  [offense-precompute] DONE — {written} cache files written. Run the "
              f"backtest now (ODI_MLB_WAREHOUSE_OFFENSE=1) — offense reads from disk.")
    return written


def _warehouse_team_factors(season, as_of, min_pa=40):
    """Warehouse-native team OFFENSE inputs for the expected-runs challenger, derived
    from statcast_pitch (leakage-safe: game_date in [season-01-01, as_of-1d]) instead of
    the live Savant statcast_search/csv endpoint. Validated to reproduce Savant team
    xwOBA to ~0.002, with NO network — so a backtest on a Savant-unreachable box still
    fires the challenger + additive model. Same dict shape as get_expected_runs_team_
    factors, but bullpen_xwoba is EMPTY (v1): _expected_staff falls back to starter-only
    when a team's bullpen xwOBA is absent. Returns the dict, or None (empty / error /
    SQL off) so the caller can fall back to Savant."""
    try:
        import savant_history as sh
        import db_store
        from sqlalchemy import select, func
    except Exception:
        return None
    if not sh.enabled():
        return None
    try:
        cutoff = _date.fromisoformat(str(as_of)[:10]) - timedelta(days=1)
    except (TypeError, ValueError):
        return None
    if cutoff.year < int(season):
        return None
    season_start = f"{int(season)}-01-01"
    sp = sh.statcast_pitch
    stmt = (select(sp.c.batting_team, sp.c.p_throws,
                   func.avg(sp.c.xwoba), func.count(sp.c.xwoba)).where(
                (sp.c.xwoba.isnot(None))
                & (sp.c.batting_team.isnot(None))
                & (sp.c.p_throws.in_(("L", "R")))
                & (sp.c.game_date >= season_start)
                & (sp.c.game_date <= cutoff.isoformat()))
            .group_by(sp.c.batting_team, sp.c.p_throws))
    rows = None
    for _attempt in range(3):                 # tolerate a transient Azure SQL timeout
        try:
            with db_store.get_engine().connect() as conn:
                rows = conn.execute(stmt).all()
            break
        except Exception:
            if _attempt < 2:
                time.sleep(0.5)
    if not rows:                              # error (all retries) OR no data yet -> None
        return None
    raw = {"L": {}, "R": {}}
    for team, hand, avg, n in (rows or []):
        if team and hand in ("L", "R") and avg is not None and n and n >= min_pa:
            raw[hand][str(team)] = {"xwoba": float(avg), "pa": int(n)}
    try:
        team_index = get_team_index(season)
        statsapi_abbrs = {info.get("abbr") for info in (team_index or {}).values()
                          if info.get("abbr")}
    except (OSError, ValueError, requests.RequestException):
        statsapi_abbrs = set()
    result = _finalize_offense(raw, statsapi_abbrs)   # shared with the precompute
    if result is not None and statsapi_abbrs:
        _warn_offense_coverage(season, result, statsapi_abbrs)
    return result


def get_expected_runs_team_factors(season, as_of, min_pa=40):
    """Return leakage-safe live team inputs for the expected-runs model.

    The model was validated on league-relative Savant expected-wOBA averages,
    split by opposing pitcher hand for offenses and restricted to relievers for
    bullpens. Savant's aggregate search endpoint produces those same averages
    in small team-level responses, avoiding a runtime pitch-level download.
    Only games before ``as_of`` are included.

    When ODI_MLB_WAREHOUSE_OFFENSE is set, the OFFENSE factors are derived from the
    statcast_pitch warehouse (no network; works on a Savant-unreachable box) and only
    fall through to Savant when the warehouse is thin/unavailable. OFF = the original
    Savant-only path, byte-identical.
    """
    cutoff = _date.fromisoformat(as_of) - timedelta(days=1)
    if cutoff.year < int(season):
        return None

    if _mlb_warehouse_offense_enabled():
        # Warehouse-ONLY when the flag is on: the whole point is offline / no-Savant, so
        # an empty result (early season — no team past min_pa yet) returns None (the
        # challenger is correctly OFF, no offense data exists anywhere yet) rather than
        # falling back to the (often unreachable) Savant endpoint, which would emit a
        # misleading "Savant ... all teams missing" coverage warning. Flag OFF still runs
        # the Savant path below, byte-identical. Distinct cache key (numbers differ).
        wcache = f"wh_expected_runs_teams_v1_{season}_{cutoff.isoformat()}_{min_pa}"
        wcached = _read_cache(wcache, max_age=24 * 3600)
        if wcached is not None:
            return wcached
        wf = _warehouse_team_factors(season, as_of, min_pa)
        if wf is not None:
            _write_cache(wcache, wf)
        return wf

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
        # Warn ONCE per unique gap (season+keys): get_expected_runs_team_factors is
        # called per game-date, and the parallel backtest pre-warm would otherwise
        # print this thousands of times.
        _sig = (int(season), tuple(unmapped), tuple(missing))
        if (unmapped or missing) and _sig not in _COVERAGE_WARNED:
            _COVERAGE_WARNED.add(_sig)
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


def expected_runs_additive(starter_rate9, bullpen_rate9, exp_ip,
                           offense_factor=1.0, run_env=1.0, full_game_ip=9.0):
    """Additive innings x rate projection of a team's expected runs — the Savant
    "xERA-lite" formulation (the challenger to expected_runs_from_factors).

    expected = [starter_rate9 * (exp_ip/9) + bullpen_rate9 * ((9 - exp_ip)/9)]
               * offense_factor * run_env

    The OPPOSING starter's expected runs/9 for his expected innings + the OPPOSING
    bullpen's runs/9 for the rest, scaled by the batting team's league-relative
    offense and the park/weather run environment. Unlike the multiplicative
    power-of-ratio model, ``*_rate9`` are already on the runs/9 scale (fitted by
    backtest_starters to ACTUAL total runs allowed per 9), so this is a direct
    run-scale sum and NEVER divides two different-scale xwOBAs — which is what
    dissolves the fit<->serve<->grade scale trap: xwOBAcon enters only as an input
    FEATURE to the fitted rate9 map, not as a ratio numerator. ``offense_factor`` /
    ``run_env`` are centered on 1.0 and applied ONCE (rate9 already assumes a
    neutral park + league-average opponent). Returns expected runs clamped to
    0.5..12.0, or None on bad input."""
    try:
        starter_rate9 = float(starter_rate9)
        bullpen_rate9 = float(bullpen_rate9)
        exp_ip = float(exp_ip)
        offense_factor = float(offense_factor)
        run_env = float(run_env)
        full_game_ip = float(full_game_ip)
    except (TypeError, ValueError):
        return None
    if (starter_rate9 <= 0 or bullpen_rate9 <= 0 or full_game_ip <= 0
            or offense_factor <= 0 or run_env <= 0):
        return None
    exp_ip = max(0.0, min(full_game_ip, exp_ip))       # bound the starter's share
    starter_share = exp_ip / full_game_ip
    base = starter_rate9 * starter_share + bullpen_rate9 * (1.0 - starter_share)
    expected = base * offense_factor * run_env
    return max(0.5, min(12.0, expected))


def _mlb_additive_runs_enabled():
    """ODI_MLB_ADDITIVE_RUNS gate for the live additive expected-runs model (Tier A
    #1d, SPREADS). OFF (unset) = byte-identical multiplicative path. Mirrors the
    espn_client ODI_MLB_* env→bool idiom; promoted from st.secrets at boot in app.py."""
    return os.environ.get("ODI_MLB_ADDITIVE_RUNS", "").strip().lower() in (
        "1", "true", "on", "yes")


def _mlb_additive_totals_enabled():
    """ODI_MLB_ADDITIVE_TOTALS gate (Tier B): use the additive expected TOTAL runs as
    the totals projection (runs-first) instead of the recency+starter-shift mean. OFF
    (unset) = byte-identical current totals model. Separate flag from spreads so each
    market's additive is an independent, evidence-gated A/B."""
    return os.environ.get("ODI_MLB_ADDITIVE_TOTALS", "").strip().lower() in (
        "1", "true", "on", "yes")


def _mlb_additive_ml_enabled():
    """ODI_MLB_ADDITIVE_ML gate (Tier B): derive the moneyline model win prob from the
    additive expected runs (runs-first, symmetric Poisson margin at 0) instead of the
    recency margin Φ. OFF (unset) = byte-identical current moneyline model. Separate
    flag from spreads/totals so each market's additive is an independent, evidence-
    gated A/B."""
    return os.environ.get("ODI_MLB_ADDITIVE_ML", "").strip().lower() in (
        "1", "true", "on", "yes")


def _any_additive_enabled():
    """True when ANY additive-market flag (spreads/totals/ML) is on — i.e. when
    live_additive_runs may be called and build_matchup_features must surface the keys
    it needs (sp_ids, team_ids/names, game_date, avg_ip). Gating on only spreads+totals
    silently starved the ML-only path (live_additive_runs saw no ids -> None -> ML fell
    back), so ML additive was inert even with its flag on; this closes that gap."""
    return (_mlb_additive_runs_enabled() or _mlb_additive_totals_enabled()
            or _mlb_additive_ml_enabled())


def live_additive_runs(sport_key, factors):
    """(home_runs, away_runs) from the ADDITIVE expected-runs model (Tier A #1d), or
    None to fall through to the multiplicative path. The live twin of the bake-off:
    reuses the SHARED additive_runs helpers on the SAME as-of rows (pitcher_asof_daily,
    on-demand-warmed for today), so it reproduces the validated bake-off number
    (fit == serve). Returns None on: non-MLB, config disabled/missing, any surfaced
    input absent, or any error → the caller keeps the multiplicative projection.

    ``factors`` is matchup_features['expected_runs'] with the #1d surfaced keys
    (home/away_sp_id, home/away_team_id, game_date, home/away_avg_ip) added ONLY when
    the flag is on. CROSSED orientation matches additive_runs.make_additive_projector:
    home_runs faces the AWAY starter + AWAY bullpen + HOME-lineup offense."""
    if sport_key != "baseball_mlb" or not factors:
        return None
    try:
        import additive_runs as ar
        import pitcher_asof
        from calibration_loader import load_expected_runs_additive
        cfg = load_expected_runs_additive(sport_key)
        if not cfg or not cfg.get("enabled"):
            return None
        hsp, asp = factors.get("home_sp_id"), factors.get("away_sp_id")
        htid, atid = factors.get("home_team_id"), factors.get("away_team_id")
        gd = factors.get("game_date")
        if not (hsp and asp and htid and atid and gd):
            return None
        model = cfg.get("model") or {}
        feature_keys = tuple(cfg.get("feature_keys")
                             or model.get("feature_keys") or ())
        league_bp = model.get("league_rate9")          # ONE source (stress fix #3):
        if not feature_keys or not model.get("coef") or league_bp is None:
            return None                                # feeds BOTH projector + bp scale
        season = int(str(gd)[:4])
        blend = cfg.get("blend") or {}
        bullpen = cfg.get("bullpen") or {}
        # Load the SP as-of series first (cached per-process during a backtest). Only
        # warm via get_or_fill when the EXACT game-date row is missing: in a fully-built
        # store (the backtest replaying history) it already exists, so this skips ~2
        # redundant SQL round-trips PER GAME and gives the live-engine backtest the
        # bake-off's bulk-load speed. Live (today's row not yet materialized) still
        # lazily fills then reloads — byte-identical result either way. RP has no
        # on-demand fill (strictly-before covers today), matching the bake-off.
        gd10 = str(gd)[:10]
        sp_series = {str(hsp): pitcher_asof.load_sp_series(hsp, season),
                     str(asp): pitcher_asof.load_sp_series(asp, season)}
        for pid in (str(hsp), str(asp)):
            if not any(r.get("as_of_date") == gd10 for r in sp_series.get(pid, [])):
                pitcher_asof.get_or_fill(pid, gd, "SP")     # genuine miss -> lazy fill
                sp_series[pid] = pitcher_asof.load_sp_series(pid, season)
        feat_getter = ar.make_feat_getter(
            sp_series, blend.get("mode", "blend"), feature_keys,
            n_starts=int(blend.get("n_starts", 10)),
            blend_k=float(blend.get("blend_k", 200.0)))
        bp_getter = None
        league_rp_era = bullpen.get("league_rp_era")
        if league_rp_era:
            rp_series = {str(htid): pitcher_asof.load_rp_series(htid, season),
                         str(atid): pitcher_asof.load_rp_series(atid, season)}
            # Bullpen fatigue (Batch A #13): sourced from the fitted config's bullpen
            # block; absent -> 0.0 -> the pre-fatigue league-relative term (byte-
            # identical). load_rp_series now carries the cumulative ip the getter needs.
            bp_getter = ar.make_bp_getter(
                rp_series, str, league_rp_era, league_bp,
                fatigue_weight=float(bullpen.get("fatigue_weight") or 0.0))
        projector = ar.make_additive_projector(
            feat_getter, model, league_bp, feature_keys, bp_getter)
        # CROSSED mapping (see make_additive_projector): home_runs uses the away
        # starter + away bullpen + a_off_faced=home-lineup offense; away_runs mirrors.
        row = {"date": str(gd)[:10], "home_sp": hsp, "away_sp": asp,
               "home_abbr": str(htid), "away_abbr": str(atid),
               "a_ip": factors.get("away_avg_ip"), "h_ip": factors.get("home_avg_ip"),
               "a_off_faced": factors.get("home_offense_factor") or 1.0,
               "h_off_faced": factors.get("away_offense_factor") or 1.0}
        pair = projector(row)
        if not pair or pair[0] is None or pair[1] is None:
            return None
        return pair
    except Exception:
        return None


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


def _run_pmf(mu, dispersion=0.0, max_runs=30):
    """Discrete score PMF over 0..``max_runs`` for a team's expected runs ``mu``.

    ``dispersion == 0`` -> independent Poisson; ``dispersion > 0`` -> negative
    binomial with variance = mean + dispersion*mean**2 (the overdispersed
    baseball score tails). The terminal bucket absorbs any probability past
    ``max_runs``. Extracted VERBATIM from the inner ``probabilities`` of
    poisson_margin_probability / negative_binomial_margin_probability (identical
    arithmetic and evaluation order, so their spread output is byte-identical) so
    the per-team run joint those models build — and then discarded after
    collapsing to a spread scalar — is now retained and reusable (roadmap §3.1).
    """
    mu = float(mu)
    max_runs = int(max_runs)
    if dispersion and dispersion > 0:
        size = 1.0 / dispersion
        success_probability = size / (size + mu)
        failure_probability = 1.0 - success_probability
        values = [success_probability ** size]
        for score in range(1, max_runs + 1):
            values.append(
                values[-1]
                * (score - 1.0 + size) / score
                * failure_probability
            )
    else:
        values = [math.exp(-mu)]
        for score in range(1, max_runs + 1):
            values.append(values[-1] * mu / score)
    values[-1] += max(0.0, 1.0 - sum(values))
    return values


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
        return _run_pmf(expected, dispersion=0.0, max_runs=max_runs)

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
        return _run_pmf(expected, dispersion=dispersion, max_runs=max_runs)

    home_prob = probabilities(home_runs)
    away_prob = probabilities(away_runs)
    return sum(
        hp * ap
        for home_score, hp in enumerate(home_prob)
        for away_score, ap in enumerate(away_prob)
        if home_score + home_spread > away_score
    )


# ── §3.1 GameContext ─────────────────────────────────────────────────────────
def _gc_own_off(expected_inputs, abbr):
    """Hand-averaged own-offense factor (>1 = better than league), or None.

    Averages the team's league-relative xwOBA vs LHP and vs RHP so no
    probable-starter hand is needed (the offline gate has none). Mirrors the
    per-hand factor in build_matchup_features._expected_offense."""
    if not expected_inputs or not abbr:
        return None
    league = expected_inputs.get("league_xwoba")
    ovh = expected_inputs.get("offense_vs_hand") or {}
    vals = []
    for hand in ("L", "R"):
        x = (ovh.get(hand) or {}).get(abbr)
        if x and league:
            vals.append(max(0.5, min(2.0, x / league)))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _gc_opp_bull_supp(expected_inputs, opp_abbr):
    """Opposing-bullpen run suppression (>1 = better opposing pen -> fewer hits),
    or None. Mirrors the bullpen term in build_matchup_features._expected_staff."""
    if not expected_inputs or not opp_abbr:
        return None
    league = expected_inputs.get("league_bullpen_xwoba")
    bx = (expected_inputs.get("bullpen_xwoba") or {}).get(opp_abbr)
    if not league or not bx:
        return None
    return max(0.5, min(2.0, league / bx))


def _gc_factor_from_terms(own_off, opp_bull_supp, form="full"):
    """Combine the leakage-safe terms into the bounded per-batter gc_factor.

    The opposing STARTER is pinned to neutral 1.0 — it is owned by matchup_mult
    (the SHIPPED opp0.5 batter_hits log5), so re-pricing it here would double
    count. ``form`` selects the ablation the free --feature-diag read compares
    (all measured MARGINAL over the SHIPPED opp_defense, which is already in the
    gate's baseline params, so no special handling is needed):
      full -> own_off / eff_supp   (own offense AND opposing bullpen)
      own  -> own_off              (own offense only; orthogonal to opp_defense)
      opp  -> 1 / eff_supp         (opposing-bullpen residual only)
    Returns 1.0 (no-op) whenever a required term is missing (fail-open)."""
    if form == "own":
        if own_off is None:
            return 1.0
        run_index = own_off
    elif form == "opp":
        if opp_bull_supp is None:
            return 1.0
        eff_supp = GC_STARTER_SHARE + (1.0 - GC_STARTER_SHARE) * opp_bull_supp
        run_index = 1.0 / eff_supp
    else:  # full
        if own_off is None or opp_bull_supp is None:
            return 1.0
        eff_supp = GC_STARTER_SHARE + (1.0 - GC_STARTER_SHARE) * opp_bull_supp
        run_index = own_off / eff_supp
    if not run_index or run_index <= 0:
        return 1.0
    val = run_index ** GC_EPS
    return max(1.0 - GC_RUN_CAP, min(1.0 + GC_RUN_CAP, val))


def _convolve_pmf(a, b):
    """Discrete convolution of two score PMFs -> the sum's PMF."""
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if not ai:
            continue
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _pmf_moments(pmf):
    """(mean, std) of a PMF indexed by outcome value 0, 1, 2, ..."""
    mean = sum(k * p for k, p in enumerate(pmf))
    var = sum((k * k) * p for k, p in enumerate(pmf)) - mean * mean
    return mean, (math.sqrt(var) if var > 0 else 0.0)


def build_game_context(home_team, away_team, date, season, team_index=None):
    """Shared, cached, leakage-safe per-game run/hits environment (roadmap §3.1).

    Returns a dict::

        {complete, mu_runs:{home,away}, run_pmf:{home:[..30],away:[..30]},
         total_pmf:[..60], total_mean, total_std, team_hits_mean:{home,away},
         gc_factor:{home:{full,own,opp}, away:{full,own,opp}}}

    ``gc_factor`` is the per-batter MEAN multiplier the confirmation gate can see
    (per side, per ablation form). Everything else (run/total PMFs, team-hit
    means — the run joint the spread model used to discard, now retained) is the
    reusable foundation for the deferred totals-unification + parlay consumer and
    does NOT feed the gate.

    LEAKAGE-SAFE: run inputs come ONLY from get_expected_runs_team_factors
    (as-of, cutoff = date-1d) — never the season-to-date get_team_offense_splits
    / get_team_bullpen_quality / get_pitcher_quality. Needs only two team abbrs +
    a date (no probable-starter, no lineup fetch), so it is computable over
    historical backtest dates. Incomplete inputs / unmatched team -> complete
    False and gc_factor 1.0 (a fail-open no-op preserving strength-0 byte parity).

    Memoized per (home_abbr, away_abbr, date) with the completeness-OR-TTL guard
    game_results._SCORE_CACHE uses: an incomplete build is retried soon, a
    complete one is trusted for a day (the inputs are themselves 24h-cached)."""
    if team_index is None:
        try:
            team_index = get_team_index(season)
        except (OSError, ValueError, requests.RequestException):
            team_index = None
    home_info = _match_team_id(home_team, team_index) if team_index else None
    away_info = _match_team_id(away_team, team_index) if team_index else None
    home_abbr = home_info.get("abbr") if home_info else None
    away_abbr = away_info.get("abbr") if away_info else None

    key = (home_abbr, away_abbr, str(date))
    now = time.time()
    cached = _GC_CACHE.get(key)
    if cached is not None:
        built_at, prev = cached
        ttl = _GC_FINAL_TTL if prev.get("complete") else _GC_LIVE_TTL
        if now - built_at < ttl:
            return prev

    expected_inputs = None
    if home_abbr and away_abbr:  # no point fetching if we'll fail open anyway
        try:
            expected_inputs = get_expected_runs_team_factors(season, str(date))
        except (OSError, ValueError, requests.RequestException):
            expected_inputs = None

    neutral = {"full": 1.0, "own": 1.0, "opp": 1.0}
    ctx = {
        "complete": False,
        "mu_runs": {"home": None, "away": None},
        "run_pmf": {"home": None, "away": None},
        "total_pmf": None, "total_mean": None, "total_std": None,
        "team_hits_mean": {"home": None, "away": None},
        "gc_factor": {"home": dict(neutral), "away": dict(neutral)},
    }

    if expected_inputs and home_abbr and away_abbr:
        # Per side: this team's own offense vs the OPPONENT's bullpen.
        terms = {
            "home": (_gc_own_off(expected_inputs, home_abbr),
                     _gc_opp_bull_supp(expected_inputs, away_abbr)),
            "away": (_gc_own_off(expected_inputs, away_abbr),
                     _gc_opp_bull_supp(expected_inputs, home_abbr)),
        }
        ctx["gc_factor"] = {
            side: {form: _gc_factor_from_terms(own, opp, form)
                   for form in ("full", "own", "opp")}
            for side, (own, opp) in terms.items()
        }
        # Object foundation (NOT gate-facing): expected team runs with the starter
        # pinned neutral (no probable starter offline), each side's run PMF, the
        # convolved game-total distribution, and a runs->hits mean map.
        mu = {}
        for side, (own, opp) in terms.items():
            off = own if own is not None else 1.0
            supp = (GC_STARTER_SHARE
                    + (1.0 - GC_STARTER_SHARE) * (opp if opp is not None else 1.0))
            er = expected_runs_from_factors(LEAGUE_RUNS_PER_GAME, off, supp)
            mu[side] = er if er is not None else LEAGUE_RUNS_PER_GAME
        ctx["mu_runs"] = mu
        ctx["run_pmf"] = {s: _run_pmf(mu[s], 0.0, 30) for s in ("home", "away")}
        total = _convolve_pmf(ctx["run_pmf"]["home"], ctx["run_pmf"]["away"])
        ctx["total_pmf"] = total
        ctx["total_mean"], ctx["total_std"] = _pmf_moments(total)
        ctx["team_hits_mean"] = {
            s: LEAGUE_HITS_PER_GAME * (mu[s] / LEAGUE_RUNS_PER_GAME) ** GC_EPS
            for s in ("home", "away")}
        ctx["complete"] = all(
            terms[s][0] is not None and terms[s][1] is not None
            for s in ("home", "away"))

    _GC_CACHE[key] = (now, ctx)
    return ctx


def gamecontext_factor(own_team, opp_team, date, season, form="full",
                       team_index=None):
    """Own-side per-batter gc_factor for ``own_team`` facing ``opp_team`` — the
    Phase-3 runtime/sweep entry point. gc_factor depends only on own offense +
    opposing bullpen (not home/away), so this needs no game orientation. Same
    leakage-safe terms and bounds as build_game_context; 1.0 fail-open on a
    miss (unmatched team / missing as-of inputs)."""
    if team_index is None:
        try:
            team_index = get_team_index(season)
        except (OSError, ValueError, requests.RequestException):
            team_index = None
    if not team_index:
        return 1.0
    own_info = _match_team_id(own_team, team_index)
    opp_info = _match_team_id(opp_team, team_index)
    own_abbr = own_info.get("abbr") if own_info else None
    opp_abbr = opp_info.get("abbr") if opp_info else None
    if not own_abbr or not opp_abbr:
        return 1.0
    try:
        ei = get_expected_runs_team_factors(season, str(date))
    except (OSError, ValueError, requests.RequestException):
        ei = None
    return _gc_factor_from_terms(_gc_own_off(ei, own_abbr),
                                 _gc_opp_bull_supp(ei, opp_abbr), form)


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
            if not resolved:
                # Could not positively identify him -- a spelling we can't bridge, or a
                # transient StatsAPI/index outage (resolve_mlbam_id swallows infra
                # errors to None). That is NOT a confident "out": rule out only on a
                # POSITIVE id that is positively absent, mirroring the pitcher arm
                # (which requires pid is not None) and this function's documented
                # "any uncertainty -> unknown" fail-open. A blanket "out" here would
                # demote a valid bet AND drop its calibration label during an outage.
                return "unknown"
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


def _asof_pitcher_quality_from_warehouse(pitcher_id, as_of_date):
    """As-of starter quality from the warehouse mlb_pitcher_game facts (ZERO
    network, O(1 query)/season) — the backtest-fast, leakage-safe source for
    get_pitcher_quality's as-of mode. Same dict shape as the StatsAPI path with the
    Savant xERA fields None (as-of always skips them). Returns None when the
    warehouse has no in-season games for him yet, so the caller can fall back to
    the byDateRange StatsAPI path."""
    try:
        import mlb_warehouse
        st = mlb_warehouse.asof_pitcher_stats(pitcher_id, as_of_date)
        if not st:
            return None
        throws = mlb_warehouse.pitcher_throws(pitcher_id)
    except Exception:
        return None
    era = st.get("era")
    out = {"name": None, "throws": throws, "era": era, "ip": None,
           "avg_ip": st.get("avg_ip"), "bf": None, "k_pct": None,
           "bb_pct": None, "xera": None, "xwoba": None, "xba": None}
    if era and era > 0:
        out["run_suppression"] = max(0.5, min(2.0, LEAGUE_AVG["era"] / era))
        out["run_suppression_basis"] = "warehouse_era"
    else:
        out["run_suppression"] = 1.0
        out["run_suppression_basis"] = "none"
    return out


def get_pitcher_quality(pitcher_id, season, as_of_date=None):
    """
    Pitching quality + handedness for a starter.

    Returns {'name','throws','era','k_pct','bb_pct','ip','bf',
             'run_suppression'} where run_suppression >1 means better than a
    league-average pitcher (LEAGUE_AVG['era'] / era), clamped to a sane range.

    ``as_of_date`` (YYYY-MM-DD): LEAKAGE-SAFE mode for backtests — the line stats
    are pulled via ``type=byDateRange`` up to as_of_date-1 (never the full season),
    and the Savant xERA is SKIPPED (only a full-season aggregate exists, which
    would leak the future), so run_suppression falls back to the as-of ERA. Live
    callers leave it None → the original season+xERA behavior is byte-identical.
    (The teams-endpoint splits/bullpen can't be date-bounded — StatsAPI ignores
    the range on statSplits — so the opposing-offense factor stays season-based;
    that residual leak is small + from a stable stat.)
    """
    if as_of_date:
        cutoff = (_date.fromisoformat(as_of_date[:10]) - timedelta(days=1)).isoformat()
        cache = f"pitcher_{pitcher_id}_{season}_asof_{cutoff}"
        # A bygone as-of line is immutable → cache ~forever (kills the backtest's
        # per-(pitcher,date) re-fetch storm).
        max_age = PERMANENT_CACHE_AGE
    else:
        cache = f"pitcher_{pitcher_id}_{season}"
        max_age = CACHE_MAX_AGE
    cached = _read_cache(cache, max_age=max_age)
    if cached is not None:
        return cached
    # As-of (backtest): prefer the warehouse facts — zero network, scales O(1) per
    # season as the data grows. Fall back to the StatsAPI byDateRange below only
    # when the warehouse has no in-season games for him yet.
    if as_of_date:
        wh = _asof_pitcher_quality_from_warehouse(pitcher_id, as_of_date)
        if wh is not None:
            _write_cache(cache, wh)
            return wh
    if as_of_date:
        hydrate = (f"stats(group=pitching,type=byDateRange,"
                   f"startDate={season}-03-01,endDate={cutoff},season={season})")
    else:
        hydrate = f"stats(group=pitching,type=season,season={season})"
    data = _get(f"people/{pitcher_id}", {"hydrate": hydrate})
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
    # SKIP in as-of mode: the Savant leaderboard is season-aggregate only, so
    # using it for a past-dated backtest game would leak the future.
    xstats = None
    if not as_of_date:
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


def build_matchup_features(home_team, away_team, date, season, team_index=None,
                           as_of_date=None):
    """
    Assemble Phase 1 starter/opponent features for one game.

    ``as_of_date`` (backtests): bound the starter's line stats to before that date
    (leakage-safe; see get_pitcher_quality). Live callers leave it None so the
    season+xERA path is unchanged.

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
    pitcher_ids = {}                     # side -> MLBAM starter id (#1d live surfacing)
    for side, tname in sides.items():
        pinfo = probables.get(_norm(tname))
        if not pinfo:
            continue
        q = get_pitcher_quality(pinfo["pitcher_id"], season, as_of_date=as_of_date)
        if not q:
            continue
        quality[side] = q
        pitcher_ids[side] = pinfo["pitcher_id"]
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
            if league and starter_xwoba:
                starter = max(0.5, min(2.0, league / starter_xwoba))
            else:
                # As-of (backtest) grading omits Savant xwOBA to stay leakage-safe,
                # which used to drop the whole expected-runs challenger. Fall back to
                # the ERA-based run_suppression (already computed as-of; same 1.0-
                # centered run-prevention scale) so the challenger is GRADED
                # historically. Live keeps xwOBA (present), so production pricing is
                # byte-unchanged; a live pitcher missing Savant xwOBA now degrades
                # gracefully instead of silently disabling the challenger.
                starter = quality[side].get("run_suppression")
                if not starter:
                    return None
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
        # #1d/Tier B: surface the keys the live additive model needs to hit
        # pitcher_asof_daily — ONLY when an additive flag (spreads/totals/ML) is on, so
        # with all OFF the expected_runs dict is byte-identical.
        if _any_additive_enabled():
            result["expected_runs"].update({
                "home_sp_id": pitcher_ids.get("home"),
                "away_sp_id": pitcher_ids.get("away"),
                "home_team_id": home_info.get("id") if home_info else None,
                "away_team_id": away_info.get("id") if away_info else None,
                "game_date": str(as_of_date or date)[:10],
                "home_avg_ip": quality["home"].get("avg_ip"),
                "away_avg_ip": quality["away"].get("avg_ip"),
            })

    # Lineup-offense edge (today's actual 9, not the team-season blob). Warehouse
    # as-of, zero network; None for live games (no batter facts yet) so it's
    # inert live until a confirmed-lineup path is wired. Weighted (inert by
    # default) in analysis._predict_margin.
    result["lineup_edge"] = lineup_offense_edge(
        home_team, away_team, as_of_date or date, team_index, season)

    return result


def lineup_offense_edge(home_team, away_team, date, team_index, season,
                        shrink_pa=100):
    """Home-minus-away lineup offense edge, in OPS units, from the warehouse
    (as-of, ZERO network, O(queries)/season via cached indexes).

    Each starter's as-of OPS is PA-SHRUNK toward league (a thin sample — a 2-PA
    callup, or anyone early-season — is pulled toward the mean so it can't skew the
    lineup), then combined per side as a SLOT-PA-WEIGHTED mean (the top of the order
    bats more; see _SLOT_PA_WEIGHTS). The edge is the home-minus-away weighted OPS,
    clamped to +/-0.3 for safety; a fitted runs-of-margin weight (DEFAULT_LINEUP_
    WEIGHT) turns it into a margin shift. Backtest uses the box-score starters
    (top-9 by PA, list order proxies the batting order); None for live games (no
    batter facts yet) or on any miss."""
    try:
        import mlb_warehouse
        home = _match_team_id(home_team, team_index)
        away = _match_team_id(away_team, team_index)
        if not home or not away:
            return None
        game_pk = mlb_warehouse._game_pk_index(season).get(
            (str(date)[:10], str(home["id"]), str(away["id"])))
        if not game_pk:
            return None
        lineups = mlb_warehouse._game_lineup_index(season).get(game_pk) or {}
        h = _lineup_side_ops(lineups, home["id"], date, shrink_pa)
        a = _lineup_side_ops(lineups, away["id"], date, shrink_pa)
        if h is None or a is None:
            return None
        return max(-0.3, min(0.3, h - a))
    except Exception:
        return None


def _lineup_side_ops(lineups, team_id, date, shrink_pa=100):
    """Slot-PA-weighted, PA-shrunk mean as-of OPS for one side's starters, or None.

    Each batter's as-of OPS is shrunk toward league by sample size (a thin callup
    can't skew it), then combined weighted by the batting slot's expected PA (the
    top of the order bats more). ``lineups`` = mlb_warehouse._game_lineup_index
    entry; list order proxies the batting order (backtest) / is the announced order
    (live v2)."""
    import mlb_warehouse
    lg = LEAGUE_AVG["ops"]
    num = den = 0.0
    for i, a in enumerate(lineups.get(str(team_id)) or []):
        o = mlb_warehouse.asof_batter_ops(a, date)
        if not o:
            continue
        pa = o.get("pa") or 0.0
        shrunk = (pa * o["ops"] + shrink_pa * lg) / (pa + shrink_pa)
        w = _SLOT_PA_WEIGHTS[i] if i < len(_SLOT_PA_WEIGHTS) else _SLOT_PA_WEIGHTS[-1]
        num += w * shrunk
        den += w
    return (num / den) if den else None


def lineup_offense_factors(home_team, away_team, date, team_index, season,
                           shrink_pa=100):
    """Per-team lineup offense FACTOR (1.0-centered: lineup OPS / league OPS) for a
    game, from the warehouse (as-of, zero network). Returns {'home': f, 'away': f}
    or None on any miss. Unlike the refuted additive edge, this is the multiplier
    that feeds a bottom-up expected-RUNS projection (expected_runs_from_factors),
    so the lineup actually drives predicted runs — see analysis.lineup_expected_runs."""
    try:
        import mlb_warehouse
        home = _match_team_id(home_team, team_index)
        away = _match_team_id(away_team, team_index)
        if not home or not away:
            return None
        game_pk = mlb_warehouse._game_pk_index(season).get(
            (str(date)[:10], str(home["id"]), str(away["id"])))
        if not game_pk:
            return None
        lineups = mlb_warehouse._game_lineup_index(season).get(game_pk) or {}
        h = _lineup_side_ops(lineups, home["id"], date, shrink_pa)
        a = _lineup_side_ops(lineups, away["id"], date, shrink_pa)
        if h is None or a is None:
            return None
        lg = LEAGUE_AVG["ops"]
        return {"home": h / lg, "away": a / lg}
    except Exception:
        return None


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


_PLAYER_INDEX_CACHE = {}       # season -> {norm_name: [(mlbam_id, is_pitcher), ...]}
_PLAYER_INDEX_LOADED_AT = {}   # season -> time.monotonic() of the last real load
_PLAYER_INDEX_TTL = 6 * 3600   # in-memory freshness: a long-lived process re-reads
                               # the (7-day) disk cache instead of pinning a season
                               # snapshot for its whole lifetime, so a mid-season
                               # callup isn't permanently shadowed by a namesake.


def _player_index(season):
    """{normalized_full_name: [(id, is_pitcher), ...]} for a season's players.

    The in-memory copy carries a bounded TTL so a long-lived process doesn't hold a
    stale season snapshot indefinitely (which would let a same-name incumbent shadow
    a debuting callup absent from it). On expiry it re-reads the disk cache /
    refetches; if that (re)load fails it keeps serving the stale in-memory copy
    rather than going dark. Test-primed caches (no load timestamp) are treated as
    pinned/fresh."""
    cached = _PLAYER_INDEX_CACHE.get(season)
    if cached is not None:
        loaded = _PLAYER_INDEX_LOADED_AT.get(season)
        if loaded is None or (time.monotonic() - loaded) < _PLAYER_INDEX_TTL:
            return cached
    try:
        disk = _read_cache(f"players_index_{season}", max_age=7 * 24 * 3600)
        if disk is not None:
            index = {k: [tuple(v) for v in vs] for k, vs in disk.items()}
        else:
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
                         {k: [[pid, isp] for pid, isp in vs]
                          for k, vs in index.items()})
    except Exception:
        if cached is not None:
            return cached          # (re)load failed -> keep the stale copy, not dark
        raise                      # cold + failed -> propagate (caller fail-opens)
    _PLAYER_INDEX_CACHE[season] = index
    _PLAYER_INDEX_LOADED_AT[season] = time.monotonic()
    _PITCHER_BY_ID_CACHE.pop(season, None)     # derived; refresh alongside a reload
    return index


def warm_player_index(season):
    """Pre-load the season player index (idempotent). Call ONCE on the main thread
    before fanning MLB work out to a pool so the workers hit the populated
    module/disk cache instead of racing N concurrent full-roster fetches + cache
    writes (the no-network-racing-file-cache invariant). Never raises."""
    try:
        _player_index(season)
    except Exception:
        pass


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


def _role_known(mid, season, row):
    """(is_pitcher, known) for an MLBAM id. The statsapi season roster is
    authoritative (so two-way players like Ohtani keep their statsapi position); the
    SFBB row's ALLPOS is used only when the id isn't in the roster (e.g. a mid-season
    callup absent from the cached payload). ``known`` is False when NEITHER source
    establishes the role -- a role-verifying caller must not trust the defaulted
    is_pitcher, because an unverifiable role is exactly the drift risk (a batter prop
    could otherwise bind an unroster'd pitcher id whose role can't be contradicted)."""
    idx = _is_pitcher_index(season)
    if str(mid) in idx:
        return idx[str(mid)], True
    allpos = ((row or {}).get("allpos") or "").upper().replace(",", "/")
    positions = [p.strip() for p in allpos.split("/") if p.strip()]
    if positions:
        return ("P" in positions), True
    return False, False


def _resolve_is_pitcher(mid, season, row):
    """is_pitcher for an MLBAM id (see _role_known); defaults False when unknown."""
    return _role_known(mid, season, row)[0]


# Generational suffix tokens the odds feed sometimes DROPS ("Bobby Witt") while the
# StatsAPI fullName KEEPS them ("Bobby Witt Jr."). Stripped only as an on-miss
# fallback so a name that already resolves is never broadened. Mirrors
# player_id_map._NAME_SUFFIXES (kept local so the resolver has no import cycle).
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def _norm_suffix_stripped(name):
    """_norm(name) with any trailing generational-suffix tokens removed, or ""."""
    toks = _norm(name).split()
    while toks and toks[-1] in _NAME_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def _iter_team_names(teams):
    """Yield the individual team strings from a hint that may be a single str or an
    iterable (e.g. the odds event's (home, away))."""
    if not teams:
        return
    if isinstance(teams, str):
        yield teams
        return
    try:
        for t in teams:
            if t:
                yield t
    except TypeError:
        yield teams


def _game_context_id(name, role, teams, confirmed_lineup, probable_starters):
    """MLBAM id for ``name`` from TODAY'S posted game, role-partitioned, or None.

    ``role`` is "P" (pitcher prop → announced probables, scoped to the matchup
    ``teams``), "B" (batter prop → posted lineup), or None (unknown → accept a hit
    only when exactly ONE of lineup/probables matches, so a same-spelling cross-role
    namesake is left for the roster tier). Exact _norm match then a suffix-stripped
    fallback (odds "Luis Garcia Jr." vs statsapi "Luis García Jr."). The ids come
    from the ACTUAL players in today's game, so this is authoritative, trade-aware
    (today's team) and namesake-safe (role + two-team scope)."""
    key = _norm(name)
    skey = _norm_suffix_stripped(name)

    def _from_lineup():
        players = (confirmed_lineup or {}).get("players") or {}
        rec = players.get(key)
        if rec is None and skey:
            for k, r in players.items():
                if _norm_suffix_stripped(k) == skey:
                    rec = r
                    break
        return (rec or {}).get("player_id")

    def _from_probables():
        probs = probable_starters or {}
        want = [_norm(t) for t in _iter_team_names(teams)]
        # probs is keyed by _norm(StatsAPI team name) but ``teams`` are the raw ODDS
        # names, which diverge (odds "Athletics" vs statsapi "Oakland Athletics").
        # Match tolerantly (like get_confirmed_lineup) so the authoritative pitcher
        # game-context tier doesn't silently no-op for divergent-name teams; a strict
        # probs.get would drop them and fall through to the drift-prone SFBB tier.
        if want:
            cands = [s for k, s in probs.items()
                     if s and any(_names_match(k, w) for w in want)]
        else:
            cands = list(probs.values())
        for s in cands:
            nm = (s or {}).get("name")
            if nm and (_norm(nm) == key
                       or (skey and _norm_suffix_stripped(nm) == skey)):
                return s.get("pitcher_id")
        return None

    if role == "P":
        return _from_probables()
    if role == "B":
        return _from_lineup()
    bid, pid = _from_lineup(), _from_probables()
    if bid and pid and str(bid) != str(pid):
        return None                        # ambiguous same-name cross-role → defer
    return bid or pid


def resolve_mlbam_id(name, season, prop_key=None, teams=None,
                     confirmed_lineup=None, probable_starters=None):
    """(mlbam_id, is_pitcher) for a name resolved GAME-CONTEXT-FIRST, else None.

    The single StatsAPI-native identity resolver for MLB — resolution order (each
    tier yields an MLBAM id; ``is_pitcher`` stays statsapi-authoritative):

      1. TODAY'S POSTED GAME (``_game_context_id``) — role-partitioned by ``prop_key``:
         a batter prop matches the confirmed lineup's ``player_id``, a pitcher prop the
         announced probable's ``pitcher_id`` (scoped to the matchup ``teams``).
         Authoritative, trade-aware, namesake-safe.
      2. StatsAPI SEASON-ROSTER unique-exact (``_player_index``) — trade-safe for a
         uniquely-named player before the lineup posts. Because the statsapi name KEEPS
         the generational suffix, "Luis García Jr." is a DISTINCT roster key from
         "Luis García", so a suffixed namesake resolves correctly here without context.
         This tier is teams-BLIND, so it is deferred to tier 3 when the team-hinted
         SFBB map names a DIFFERENT same-role player (the ~weekly index can miss a
         recent add while a namesake is its sole entry).
      3. SFBB (``mlb_id_for_name``, team-hinted) — reordered BELOW statsapi and never
         trusted blind: when the prop's role is known, an id whose role CONTRADICTS the
         prop (the Luis Garcia Jr. pitcher-id-for-a-batter-prop drift) — or whose role
         cannot be VERIFIED at all (id absent from the index + no SFBB position) — is
         rejected. This is the ONLY tier that can drift, and it is now last-resort.

    ``prop_key=None`` (the find_player_id contract) resolves role-agnostically and lets
    the caller role-gate the tuple. Returns None rather than risk a wrong bind; never
    raises."""
    try:
        if not name:
            return None
        role = None
        if prop_key is not None:
            role = "P" if str(prop_key).startswith("pitcher_") else "B"

        # 1. Today's posted game (authoritative, trade-aware, namesake-safe).
        gc = _game_context_id(name, role, teams, confirmed_lineup, probable_starters)
        if gc:
            # A probable starter is definitionally a pitcher; a lineup batter's
            # is_pitcher stays statsapi-authoritative (two-way players like Ohtani).
            isp = True if role == "P" else _resolve_is_pitcher(gc, season, None)
            return (gc, isp)

        # Team-aware SFBB candidate, resolved ONCE: it both cross-checks the
        # teams-blind statsapi tier below and is the last-resort fallback. Its role
        # is "known" only when the season index or the SFBB row can establish it.
        sfbb_id, sfbb_isp, sfbb_known = None, False, False
        pim = _player_id_map()
        if pim is not None:
            sid = pim.mlb_id_for_name(name, teams=teams)
            if sid:
                sfbb_id = sid
                sfbb_isp, sfbb_known = _role_known(sid, season, pim.get_row(name))

        # 2. StatsAPI season-roster unique-exact (trade-safe, pre-lineup). Accept the
        #    teams-blind index hit UNLESS a team-aware SFBB lookup names a DIFFERENT
        #    SAME-ROLE player under a matchup hint: the ~weekly index can miss a
        #    recently-added player while a namesake is its sole entry, so defer that
        #    conflict to the team-aware SFBB tier. (A cross-role SFBB disagreement is
        #    the drift the index is meant to beat, so it does NOT trigger a deferral.)
        matches = _player_index(season).get(_norm(name))
        if matches and len(matches) == 1:
            mid, isp = matches[0]
            if role is None or (role == "P") == bool(isp):
                conflict = (role is not None and bool(teams) and sfbb_id
                            and str(sfbb_id) != str(mid)
                            and sfbb_known and (role == "P") == bool(sfbb_isp))
                if not conflict:
                    return (mid, isp)

        # 3. SFBB fallback, role-verified (never trusted blind): reject an id whose
        #    role CONTRADICTS the prop, AND -- because an unverifiable role is exactly
        #    the drift risk -- reject a role-known prop whose SFBB id role can't be
        #    established at all (id absent from the index + no SFBB position).
        if sfbb_id:
            if role is not None and (not sfbb_known
                                     or (role == "P") != bool(sfbb_isp)):
                return None
            return (sfbb_id, sfbb_isp)
        return None
    except Exception:                       # pragma: no cover - fail-open guard
        return None


def find_player_id(name, season, teams=None, lineup=None, probables=None):
    """(mlbam_id, is_pitcher) for a resolvable name, else None — GAME-CONTEXT-FIRST.

    Thin, stable, test-patched entry point that delegates to ``resolve_mlbam_id``
    (role-agnostic here — callers role-gate the returned tuple). ``lineup`` /
    ``probables`` (today's posted lineup / announced probables) let the resolver bind
    off the actual game before falling to the statsapi season roster, then to a
    role-checked SFBB fallback. The two-tuple contract and the never-wrong-bind
    refusal (None on an ambiguous name) are unchanged; the ONLY change from the prior
    SFBB-first order is that statsapi identity now wins over the drift-prone SFBB
    cross-map."""
    return resolve_mlbam_id(name, season, prop_key=None, teams=teams,
                            confirmed_lineup=lineup, probable_starters=probables)


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


# Sentinel: resolve_player_game_stat returns this (NOT None) for a game that is not a
# genuine final yet (live/partial/postponed/suspended). Distinct from None ("couldn't
# resolve"): it tells the caller to keep the bet PENDING and never grade off a
# live/partial stat.
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
