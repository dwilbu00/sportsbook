"""nfl_schedule.py — the canonical NFL game SPINE (nflverse), and the integrity-
checked bridge from sportsbook odds events to that spine.

WHY THIS EXISTS
---------------
The MLB backtests bred fake edges partly because three loosely-associated
sources (Odds-API event_id, StatsAPI game_pk, Savant) were bridged by fragile
date/name matching. NFL gets ONE canonical key instead: nflverse's `game_id`
(``{season}_{week:02d}_{AWAY}_{HOME}``, e.g. ``2024_01_BAL_KC``, Super Bowl
``2024_22_KC_PHI``). The SAME game_id is the primary key across nflverse's
schedules (this file), play-by-play EPA (``nfl_epa.py``), and player-week stats,
so those layers join with zero fuzzy matching. This module:

  1. fetches + caches the single-file ``games.csv`` (all seasons) dep-free
     (requests + stdlib csv), exactly like ``nfl_epa`` fetches pbp;
  2. exposes the schedule/result records + a scores index (grading spine);
  3. resolves an odds event (full team names + commence_time) to a ``game_id``
     with FAIL-CLOSED integrity checks (never a silent mis-join).

Preseason CANNOT leak: nflverse ``games.csv`` contains only
``game_type ∈ {REG, WC, DIV, CON, SB}`` — no PRE rows — so a preseason odds
event simply finds no match and is quarantined. We KEEP all playoffs + the
Super Bowl (WC/DIV/CON/SB) and EXCLUDE only preseason, exactly as intended.

Data: https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
(CC-BY-4.0). Completed seasons are immutable; the current season fills in within
~a day of each slate, so the cache carries a short TTL.
"""
import csv
import io
import os
import time
from datetime import datetime, timezone

import requests

import nfl_epa   # reuse NAME_TO_ABBR, _abbr, season_for_date (single source of truth)

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsbookValueFinder/1.0)"}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cache", "nflverse_games")
_CACHE_PATH = os.path.join(CACHE_DIR, "games.csv")
# Re-fetch when the cached file is older than this (the file mutates as the live
# season fills; completed seasons within it never change). ~12h keeps in-season
# results ≤ a day stale without re-downloading every call.
_CACHE_TTL_SECONDS = 12 * 3600

# The game types we KEEP (playoffs + Super Bowl included; preseason absent from
# the feed entirely, so this set is belt-and-suspenders).
KEEP_GAME_TYPES = frozenset({"REG", "WC", "DIV", "CON", "SB"})

# Columns we retain from games.csv (the feed has ~40; these are what we use).
_FIELDS = ("game_id", "season", "game_type", "week", "gameday", "gametime",
           "home_team", "away_team", "home_score", "away_score", "location",
           "result", "total", "espn")


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_games(force=False):
    """Download + cache the single-file games.csv. Re-fetches when stale (TTL) or
    ``force``; otherwise returns the cached path. Never re-raises a stale-cache
    read into the caller — a network failure with a cache present keeps the cache."""
    _ensure_dir()
    fresh = (os.path.exists(_CACHE_PATH)
             and (time.time() - os.path.getmtime(_CACHE_PATH)) < _CACHE_TTL_SECONDS)
    if fresh and not force:
        return _CACHE_PATH
    try:
        r = requests.get(GAMES_URL, headers=HEADERS, timeout=120)
        r.raise_for_status()
        tmp = f"{_CACHE_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(r.text)
        os.replace(tmp, _CACHE_PATH)
    except Exception:
        if os.path.exists(_CACHE_PATH):
            return _CACHE_PATH          # degrade to the (possibly stale) cache
        raise
    return _CACHE_PATH


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# The canonical mirror parquet (written offline by nfl_ingest.py via nflreadpy),
# LFS-shared alongside the MLB mirror. Runtime reads THIS dep-free; the raw-CSV
# path below is only a bootstrap fallback (before the first ingest, or on a box
# without the mirror) so the join is never blocked and stays unit-testable.
def _mirror_game_file():
    try:
        import warehouse_mirror as _wm
        mdir = _wm.MIRROR_DIR
    except Exception:
        mdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "warehouse_mirror_data")
    return os.path.join(mdir, "nfl_game__americanfootball_nfl.parquet")


def _load_from_mirror(want):
    """Return records from the nfl_game mirror parquet, or None if absent/stub."""
    path = _mirror_game_file()
    try:
        import warehouse_mirror as _wm
        if not _wm._is_real_parquet(path):   # LFS pointer stub -> treat as absent
            return None
    except Exception:
        if not os.path.exists(path):
            return None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except Exception:
        return None
    out = []
    for r in df.to_dict("records"):
        if r.get("game_type") not in KEEP_GAME_TYPES:
            continue
        if want is not None and str(r.get("season")) not in want:
            continue
        rec = {k: r.get(k) for k in _FIELDS}
        rec["season"] = str(rec["season"])
        rec["home_score"] = _to_int(rec["home_score"])
        rec["away_score"] = _to_int(rec["away_score"])
        out.append(rec)
    return out


def _load_from_csv(want, force=False):
    """Bootstrap fallback: parse the raw nflverse games.csv directly."""
    path = fetch_games(force=force)
    out = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("game_type") not in KEEP_GAME_TYPES:
                continue
            if want is not None and r.get("season") not in want:
                continue
            rec = {k: r.get(k) for k in _FIELDS}
            rec["home_score"] = _to_int(rec["home_score"])
            rec["away_score"] = _to_int(rec["away_score"])
            out.append(rec)
    return out


_GAMES_CACHE = {}   # frozenset(seasons) or None -> [rec, ...]


def load_games(seasons=None, force=False, prefer_mirror=True):
    """Return schedule/result records (list of dicts with ``_FIELDS``) for the
    requested seasons (all if None), filtered to KEEP_GAME_TYPES. ``home_team`` /
    ``away_team`` are nflverse ABBREVIATIONS. ``home_score``/``away_score`` are
    ints (None for a not-yet-played game). Reads the mirror parquet first (the
    canonical artifact from nfl_ingest); falls back to the raw CSV. Cached."""
    key = frozenset(str(s) for s in seasons) if seasons else None
    if not force and key in _GAMES_CACHE:
        return _GAMES_CACHE[key]
    out = _load_from_mirror(key) if prefer_mirror else None
    if out is None:
        out = _load_from_csv(key, force=force)
    _GAMES_CACHE[key] = out
    return out


def game_index(seasons=None, force=False):
    """``{game_id: rec}`` for the requested seasons."""
    return {r["game_id"]: r for r in load_games(seasons, force=force)}


def team_scores_index(seasons=None, force=False):
    """``{game_id: (home_score, away_score)}`` for COMPLETED games — the NFL
    grading spine (analog of r2_data.build_team_scores_index for MLB)."""
    out = {}
    for r in load_games(seasons, force=force):
        if r["home_score"] is not None and r["away_score"] is not None:
            out[r["game_id"]] = (float(r["home_score"]), float(r["away_score"]))
    return out


# ── odds → game_id resolution (fail-closed, integrity-checked) ────────────────

def _commence_date(commence_time):
    """UTC date (date obj) from an ISO commence timestamp, or None."""
    if not commence_time:
        return None
    try:
        s = str(commence_time).replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc).date()
    except (TypeError, ValueError):
        return None


def _gameday_date(gameday):
    try:
        return datetime.strptime(str(gameday)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def resolve_event(home_name, away_name, commence_time, index=None,
                  max_day_gap=1):
    """Resolve one odds event to a nflverse ``game_id``. Returns
    ``(game_id, reason)``: game_id is None on any failure and ``reason`` is a
    short machine tag (never a silent mis-join):

      ok                       matched exactly one game
      unresolved_home/away     a team name didn't map to an abbreviation
      no_commence              missing/unparseable commence_time
      no_match                 no {season,home,away} game (likely PRESEASON)
      ambiguous               >1 candidate and the date tiebreak couldn't pick one
      date_out_of_range        nearest candidate is > max_day_gap days from commence

    ``index`` = game_index(...) (built once by the batch caller)."""
    ha = nfl_epa._abbr(home_name)
    if ha is None:
        return None, "unresolved_home"
    aa = nfl_epa._abbr(away_name)
    if aa is None:
        return None, "unresolved_away"
    cdate = _commence_date(commence_time)
    if cdate is None:
        return None, "no_commence"
    season = str(nfl_epa.season_for_date(cdate.isoformat()))
    if index is None:
        index = game_index([season])
    cands = [g for g in index.values()
             if g["season"] == season and g["home_team"] == ha
             and g["away_team"] == aa]
    if not cands:
        return None, "no_match"          # preseason or an unseen matchup
    # Disambiguate a regular-vs-playoff rematch (same home team) by kickoff date
    # nearest the odds commence; enforce a ±max_day_gap sanity so a wrong-season
    # or mis-typed event can't bind to a far-off game.
    scored = []
    for g in cands:
        gd = _gameday_date(g["gameday"])
        gap = abs((gd - cdate).days) if gd else 10 ** 6
        scored.append((gap, g))
    scored.sort(key=lambda x: x[0])
    best_gap, best = scored[0]
    if best_gap > max_day_gap:
        return None, "date_out_of_range"
    if len(scored) > 1 and scored[1][0] == best_gap:
        return None, "ambiguous"
    return best["game_id"], "ok"


def resolve_events(events, seasons=None):
    """Batch-resolve odds events → game_ids with an integrity report.

    ``events`` = iterable of dicts each carrying ``home``, ``away``,
    ``commence_time`` (and optionally ``event_id``). Returns
    ``(mapping, report)`` where mapping = ``{event_id_or_index: game_id}`` for
    the OK rows and report is a dict of counts + quarantined examples. Enforces
    the integrity invariants: 1:1 event→game (a game_id claimed by >1 event is
    flagged), team-name completeness, no preseason (no_match), date sanity."""
    idx = game_index(seasons)
    mapping, reasons, quarantine = {}, {}, []
    game_to_events = {}
    for i, e in enumerate(events):
        eid = e.get("event_id", i)
        gid, reason = resolve_event(e.get("home"), e.get("away"),
                                    e.get("commence_time"), index=idx)
        reasons[reason] = reasons.get(reason, 0) + 1
        if gid is None:
            if len(quarantine) < 25:
                quarantine.append({"event_id": eid, "home": e.get("home"),
                                   "away": e.get("away"),
                                   "commence_time": e.get("commence_time"),
                                   "reason": reason})
            continue
        mapping[eid] = gid
        game_to_events.setdefault(gid, []).append(eid)
    # 1:1 integrity: a single nflverse game claimed by multiple distinct odds
    # events is a red flag (duplicate capture or a bad match).
    collisions = {g: evs for g, evs in game_to_events.items() if len(evs) > 1}
    report = {
        "n_events": len(reasons) and sum(reasons.values()),
        "n_matched": len(mapping),
        "n_games_matched": len(game_to_events),
        "reasons": reasons,
        "collisions": {g: evs for g, evs in list(collisions.items())[:25]},
        "n_collisions": len(collisions),
        "quarantine_sample": quarantine,
    }
    return mapping, report


if __name__ == "__main__":
    import argparse
    from cli_encoding import configure_stdio
    configure_stdio()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    ap.add_argument("--force", action="store_true", help="force re-fetch games.csv")
    ap.add_argument("--audit", action="store_true",
                    help="resolve the captured NFL odds events -> game_id and "
                         "report join integrity (needs the odds mirror/warehouse).")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    recs = load_games(seasons, force=args.force)
    by_season = {}
    for r in recs:
        by_season.setdefault(r["season"], {"n": 0, "final": 0, "types": {}})
        d = by_season[r["season"]]
        d["n"] += 1
        d["types"][r["game_type"]] = d["types"].get(r["game_type"], 0) + 1
        if r["home_score"] is not None:
            d["final"] += 1
    print(f"  nflverse games.csv — spine loaded ({sum(d['n'] for d in by_season.values())} "
          f"games, {len(seasons)} seasons):")
    for s in sorted(by_season):
        d = by_season[s]
        print(f"    {s}: {d['n']} games ({d['final']} final)  {d['types']}")

    if args.audit:
        # Enumerate distinct captured NFL odds events from the mirror (0 DTU).
        import warehouse_mirror as wm
        seen = {}
        for book in ("draftkings", "fanduel", "pinnacle"):
            for s in seasons:
                try:
                    rows = wm.team_market_lines("americanfootball_nfl",
                                                date_from=f"{s}-01-01",
                                                date_to=f"{s}-12-31", bookmaker=book)
                except Exception:
                    rows = []
                for r in rows:
                    eid = r.get("event_id")
                    if eid and eid not in seen:
                        seen[eid] = {"event_id": eid, "home": r.get("home"),
                                     "away": r.get("away"),
                                     "commence_time": r.get("commence_time")}
        events = list(seen.values())
        mapping, report = resolve_events(events, seasons=seasons)
        print(f"\n  === ODDS→game_id JOIN AUDIT ({len(events)} distinct NFL odds events) ===")
        print(f"    matched: {report['n_matched']}/{report['n_events']} "
              f"({report['n_games_matched']} distinct games)")
        print(f"    reasons: {report['reasons']}")
        print(f"    1:1 collisions (game claimed by >1 event): {report['n_collisions']}")
        for ex in report["quarantine_sample"][:12]:
            print(f"      [drop:{ex['reason']}] {ex['away']} @ {ex['home']} "
                  f"{ex['commence_time']}")
        # Coverage: matched events vs games that actually have odds should be ~1:1.
        print("    (want: reasons dominated by 'ok'; no_match ≈ preseason events; "
              "0 collisions; unresolved_* = 0)")
