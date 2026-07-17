"""
NFL EPA (Expected Points Added) feature layer — the NFL analog to the MLB
Savant x-stat layer (mlb_starters.py / savant_history.py).

EPA per play is the closest thing NFL has to Baseball Savant's "expected" stats:
it measures the value of each play independent of final-score noise, so team
offensive/defensive EPA/play is far more predictive of future margins than raw
points or yards. A quick check on 2024 (season-aggregate) put net-EPA-diff vs
game-margin correlation at ~0.61, versus ~coin-flip for the recency-margin proxy.

Data source: nflverse play-by-play CSVs published on GitHub releases
(github.com/nflverse/nflverse-data). We fetch the gzipped CSV per season with
`requests` and parse with the stdlib `csv` module — NO pandas / nfl_data_py
runtime dependency, mirroring how savant_history.py pulls Savant CSVs.

Two aggregation modes share one code path:
  * LIVE  (as_of_date=None): current-season-to-date EPA, shrunk toward the prior
    season when the current sample is thin (early weeks).
  * BACKTEST (as_of_date set): identical, but only plays from games STRICTLY
    BEFORE the target date are counted, so a past game is never graded with
    information from itself or later games (leakage-safe).

Play-by-play for a completed season never changes, so season caches are
permanent once downloaded.
"""

import csv
import gzip
import io
import json
import os
from datetime import datetime, timezone

import requests

PBP_BASE = ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
            "play_by_play_{season}.csv.gz")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsbookValueFinder/1.0)"}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cache", "nflverse_pbp")
# Precomputed per-season team ratings shipped in the repo so the LIVE app can
# skip the ~19MB/season pbp download on cold start (see export_ratings /
# live_ratings). Committed like the calibration files.
RATINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "calibration")
# Bump when the trimmed-play shape changes so stale caches are re-parsed.
SCHEMA_VERSION = 1

# Current-season plays needed before we trust it fully; below this we shrink the
# rating toward the prior-season aggregate (≈ one play-heavy game ≈ 130 plays,
# so ~400 ≈ 3 games of stabilization).
STABILIZE_PLAYS = 400
# League-average EPA/play is ~0 by construction, so a missing team defaults to 0.
LEAGUE_AVG_EPA = 0.0

# nflverse team abbreviation -> full name used by ESPN / the odds API.
TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
NAME_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_NAME.items()}


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _raw_path(season):
    return os.path.join(CACHE_DIR, f"play_by_play_{season}.csv.gz")


def _trimmed_path(season):
    return os.path.join(CACHE_DIR, f"plays_v{SCHEMA_VERSION}_{season}.json")


def fetch_pbp(season, force=False):
    """Download (and permanently cache) the raw season pbp CSV.gz."""
    _ensure_dir()
    path = _raw_path(season)
    if not force and os.path.exists(path):
        return path
    url = PBP_BASE.format(season=season)
    r = requests.get(url, headers=HEADERS, timeout=180)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


_PLAYS_CACHE = {}


def load_plays(season, force=False):
    """
    Return a list of trimmed regular-season pass/rush plays for `season`:
        {game_id, game_date, home_team, away_team, posteam, defteam,
         epa, home_score, away_score}
    Scores are the game's FINAL totals (constant per game_id), used for the
    totals baseline; epa is per play. Cached in memory and as compact JSON.
    """
    if not force and season in _PLAYS_CACHE:
        return _PLAYS_CACHE[season]
    tpath = _trimmed_path(season)
    if not force and os.path.exists(tpath):
        try:
            with open(tpath, "r") as f:
                plays = json.load(f)
            _PLAYS_CACHE[season] = plays
            return plays
        except (json.JSONDecodeError, OSError):
            pass

    raw = fetch_pbp(season, force=force)
    plays = []
    # Track each game's final score (last non-empty total_*_score seen).
    final_score = {}
    with gzip.open(raw, "rt") as f:
        for r in csv.DictReader(f):
            if r.get("season_type") != "REG":
                continue
            gid = r.get("game_id")
            hs, as_ = r.get("total_home_score"), r.get("total_away_score")
            if gid and hs not in (None, "", "NA") and as_ not in (None, "", "NA"):
                try:
                    final_score[gid] = (float(hs), float(as_))
                except ValueError:
                    pass
            pt, dt, epa = r.get("posteam"), r.get("defteam"), r.get("epa")
            if not pt or not dt or epa in (None, "", "NA"):
                continue
            if r.get("pass") != "1" and r.get("rush") != "1":
                continue
            try:
                e = float(epa)
            except ValueError:
                continue
            plays.append({
                "game_id": gid,
                "game_date": r.get("game_date"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "posteam": pt,
                "defteam": dt,
                "epa": e,
            })
    # Attach final scores (used by the totals baseline) now that they're known.
    for p in plays:
        sc = final_score.get(p["game_id"])
        if sc:
            p["home_score"], p["away_score"] = sc

    _ensure_dir()
    try:
        with open(tpath, "w") as f:
            json.dump(plays, f)
    except OSError:
        pass
    _PLAYS_CACHE[season] = plays
    return plays


def season_for_date(date):
    """nflverse season year for a YYYY-MM-DD date. The NFL season Y spans
    Sep Y → Feb Y+1, so Jan/Feb games belong to the previous calendar year's
    season (month >= 8 ⇒ current year, else prior year)."""
    y, m = int(date[:4]), int(date[5:7])
    return y if m >= 8 else y - 1


def _aggregate(plays, as_of_date=None):
    """Mean offensive and defensive EPA/play per team abbreviation from `plays`,
    counting only games strictly before `as_of_date` (YYYY-MM-DD) when set."""
    off_sum, off_n = {}, {}
    def_sum, def_n = {}, {}
    for p in plays:
        if as_of_date and (p["game_date"] or "") >= as_of_date:
            continue
        pt, dt, e = p["posteam"], p["defteam"], p["epa"]
        off_sum[pt] = off_sum.get(pt, 0.0) + e
        off_n[pt] = off_n.get(pt, 0) + 1
        def_sum[dt] = def_sum.get(dt, 0.0) + e
        def_n[dt] = def_n.get(dt, 0) + 1
    teams = set(off_n) | set(def_n)
    out = {}
    for t in teams:
        on, dn = off_n.get(t, 0), def_n.get(t, 0)
        out[t] = {
            "off_epa": off_sum.get(t, 0.0) / on if on else LEAGUE_AVG_EPA,
            "def_epa": def_sum.get(t, 0.0) / dn if dn else LEAGUE_AVG_EPA,
            "off_plays": on,
            "def_plays": dn,
        }
    return out


# Cache aggregates so repeated build_matchup_features calls in one run are cheap.
_AGG_CACHE = {}


def team_epa(season, as_of_date=None, prior_shrink=True):
    """
    Team EPA ratings for `season` as of `as_of_date` (None = full season / live).

    Each team maps to {off_epa, def_epa, net_epa, off_plays, def_plays}. When
    `prior_shrink` is on and a team's current sample is below STABILIZE_PLAYS,
    its off/def EPA is blended toward the prior season's full-season value
    (weight = current_plays / STABILIZE_PLAYS), which stabilizes early-week and
    playoff-cutover ratings.
    """
    key = (season, as_of_date, prior_shrink)
    if key in _AGG_CACHE:
        return _AGG_CACHE[key]

    cur = _aggregate(load_plays(season), as_of_date)
    prior = None
    if prior_shrink:
        try:
            prior = _aggregate(load_plays(season - 1))  # full prior season
        except Exception:
            prior = None

    out = {}
    for t, c in cur.items():
        off, deff = c["off_epa"], c["def_epa"]
        if prior_shrink and prior and t in prior:
            pn = min(c["off_plays"], c["def_plays"])
            w = min(1.0, pn / STABILIZE_PLAYS)
            off = w * off + (1.0 - w) * prior[t]["off_epa"]
            deff = w * deff + (1.0 - w) * prior[t]["def_epa"]
        out[t] = {
            "off_epa": off,
            "def_epa": deff,
            "net_epa": off - deff,
            "off_plays": c["off_plays"],
            "def_plays": c["def_plays"],
        }
    # Include prior-only teams (no current plays yet, e.g. week 1) at prior value.
    if prior_shrink and prior:
        for t, pv in prior.items():
            if t not in out:
                out[t] = {"off_epa": pv["off_epa"], "def_epa": pv["def_epa"],
                          "net_epa": pv["off_epa"] - pv["def_epa"],
                          "off_plays": 0, "def_plays": 0}
    _AGG_CACHE[key] = out
    return out


def _abbr(team_name):
    """Resolve an ESPN/odds full team name to its nflverse abbreviation."""
    if team_name in NAME_TO_ABBR:
        return NAME_TO_ABBR[team_name]
    # Fallback: match on last word (nickname), e.g. "Commanders".
    nick = team_name.rsplit(" ", 1)[-1].lower()
    for name, ab in NAME_TO_ABBR.items():
        if name.rsplit(" ", 1)[-1].lower() == nick:
            return ab
    return None


def build_matchup_features(home_team, away_team, date, season,
                           team_ratings=None):
    """
    Assemble NFL EPA matchup features for one game.

    Returns a dict with:
      * ``starter_edge`` — the home-minus-away NET EPA/play difference (raw, not
        squashed; naturally bounded to ≈ ±0.5). This is the SAME generic
        margin-edge hook the MLB path uses, so it feeds analyze_moneyline_value
        and analyze_spreads_value via _predict_margin with the calibratable
        NFL 'spreads' starter_adjustment weight — no analysis.py change needed.
      * ``home`` / ``away`` — {off_epa, def_epa, net_epa} for transparency.

    Returns starter_edge=None (graceful degrade to the team-only model) when
    either team can't be resolved.

    ``date`` (YYYY-MM-DD) makes the ratings leakage-safe: only games before it
    are counted. Pass ``team_ratings`` from team_epa(...) to avoid recomputing.
    """
    result = {"home": None, "away": None, "starter_edge": None}
    ha, aa = _abbr(home_team), _abbr(away_team)
    if not ha or not aa:
        return result

    if team_ratings is None:
        team_ratings = team_epa(season, as_of_date=date)

    h, a = team_ratings.get(ha), team_ratings.get(aa)
    if not h or not a:
        return result

    result["home"] = {"off_epa": h["off_epa"], "def_epa": h["def_epa"],
                      "net_epa": h["net_epa"]}
    result["away"] = {"off_epa": a["off_epa"], "def_epa": a["def_epa"],
                      "net_epa": a["net_epa"]}
    result["starter_edge"] = h["net_epa"] - a["net_epa"]
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Precomputed live ratings (avoid the runtime pbp download on the deployed app)
# ──────────────────────────────────────────────────────────────────────────────
def _ratings_path(season):
    return os.path.join(RATINGS_DIR, f"nfl_epa_ratings_{season}.json")


def export_ratings(season):
    """Compute season-to-date team EPA ratings (with prior-season shrink) from
    the pbp feed and write a compact JSON the deployed app reads via
    live_ratings(). Run this locally to refresh, then commit — like calibration.
    """
    ratings = team_epa(season)  # full season-to-date, prior-shrunk
    trimmed = {t: {"off_epa": round(v["off_epa"], 5),
                   "def_epa": round(v["def_epa"], 5),
                   "net_epa": round(v["net_epa"], 5)}
               for t, v in ratings.items()}
    blob = {"season": season,
            "generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ratings": trimmed}
    os.makedirs(RATINGS_DIR, exist_ok=True)
    with open(_ratings_path(season), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    return _ratings_path(season)


_LIVE_RATINGS_CACHE = {}


def live_ratings(season):
    """Team EPA ratings for LIVE use. Prefers the committed precomputed file
    (no download); falls back to fetching + computing from pbp if the file is
    missing. Returns {abbr: {off_epa, def_epa, net_epa}} or None on failure."""
    if season in _LIVE_RATINGS_CACHE:
        return _LIVE_RATINGS_CACHE[season]
    path = _ratings_path(season)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)["ratings"]
            _LIVE_RATINGS_CACHE[season] = r
            return r
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    try:
        r = team_epa(season)
    except Exception:
        return None
    _LIVE_RATINGS_CACHE[season] = r
    return r


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    import argparse
    ap = argparse.ArgumentParser(description="NFL EPA ratings tools")
    ap.add_argument("--export-ratings", type=int, metavar="SEASON",
                    help="Compute + write committed live ratings for SEASON.")
    args = ap.parse_args()
    if args.export_ratings:
        p = export_ratings(args.export_ratings)
        print(f"Wrote {p}")
    else:
        ap.print_help()
