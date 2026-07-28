"""Team-market final scores and grading for the actual-bets ledger.

Grades moneyline / spread / total wagers on real games. Final scores are
fetched PER DATE (never a whole season): MLB via the free statsapi schedule,
NBA/NFL/NHL via ESPN's public scoreboard. Team matching mirrors the backtest
infra (``_team_key`` normalization) and disambiguates same-day games by the
start time nearest the wager's ``commence_time`` (doubleheaders / date slippage).

Everything fails closed: a missing score or an outage returns None so grading
simply stays pending and is retried later — it never raises into the app.
"""
import os
import time
from datetime import date as _date, datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# sport_key -> (espn_sport, espn_league). MLB is handled separately (statsapi).
_ESPN_MAP = {
    "basketball_nba": ("basketball", "nba"),
    "americanfootball_nfl": ("football", "nfl"),
    "icehockey_nhl": ("hockey", "nhl"),
}

# In-process memo so one grading pass reuses a date's scoreboard fetch. Value is
# (fetched_at_epoch, [game dicts]): past dates are immutable (memoized for the
# process lifetime), but a recent date may still be finalizing, so its memo
# expires after a short TTL — otherwise a game that goes final mid-session stays
# masked by the stale (final-only) slate that dropped it, stranding its bet.
_SCORE_CACHE = {}
_RECENT_MEMO_TTL = 20 * 60  # seconds


def _team_key(name):
    """Normalize a team name for matching (mirrors backtest_market_consensus)."""
    normalized = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    aliases = {
        "oaklandathletics": "athletics",
        "theathletics": "athletics",
        "laclippers": "clippers",
        "losangelesclippers": "clippers",
    }
    return aliases.get(normalized, normalized)


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


# ──────────────────────────────────────────────────────────────────────────────
# Grading
# ──────────────────────────────────────────────────────────────────────────────

def side_for_team(team, home_team, away_team):
    """'home'/'away' for the side ``team`` is on, or None if it matches neither.

    Lets the grader resolve which side a moneyline/spread bet is on from the
    authoritative team names rather than a stored 'side' (which is set at submit
    from home_away and would grade the wrong team if it were ever stale/flipped).
    Uses the same normalization as score matching."""
    tk = _team_key(team)
    if not tk:
        return None
    if tk == _team_key(home_team):
        return "home"
    if tk == _team_key(away_team):
        return "away"
    return None


def grade_team_bet(bet_type, side, point, home_score, away_score):
    """Grade one team-market bet given final scores.

    ``bet_type``: 'moneyline' | 'spread' | 'total'.
    ``side``: 'home'/'away' for moneyline & spread; 'over'/'under' for total.
    ``point``: the bettor-side spread (signed, for spread) or the total line
               (for total); ignored for moneyline.
    Returns 'won' | 'lost' | 'push', or None when inputs are invalid.
    The math mirrors backtest_market_consensus._grade_side."""
    try:
        home_score = float(home_score)
        away_score = float(away_score)
    except (TypeError, ValueError):
        return None
    bt = (bet_type or "").lower()
    sd = (side or "").lower()
    if bt == "moneyline":
        value = (home_score - away_score) if sd == "home" else (away_score - home_score)
    elif bt == "spread":
        if point is None:
            return None
        margin = ((home_score - away_score) if sd == "home"
                  else (away_score - home_score))
        value = margin + float(point)
    elif bt == "total":
        if point is None:
            return None
        value = home_score + away_score - float(point)
        if sd == "under":
            value = -value
    else:
        return None
    if value > 1e-9:
        return "won"
    if value < -1e-9:
        return "lost"
    return "push"


# ──────────────────────────────────────────────────────────────────────────────
# Per-date final-score fetch
# ──────────────────────────────────────────────────────────────────────────────

def _mlb_scores_for_date(game_date):
    """List of FINAL MLB games on ``game_date`` (YYYY-MM-DD) with scores.

    Uses the free statsapi schedule with a linescore hydrate as a run fallback.
    Cached on disk (results are immutable once final)."""
    try:
        import mlb_starters
    except Exception:
        return []
    cache = f"final_scores_mlb_{game_date}"
    # Past dates are immutable; a current/future date may still have games in
    # progress (the final-only list omits them), so trust its disk cache only
    # briefly — else a stale slate that dropped the not-yet-final game strands the
    # bet (or, via the ±1-day fallback, risks matching the wrong night of a series).
    today = datetime.now(timezone.utc).date().isoformat()
    max_age = 6 * 3600 if str(game_date)[:10] < today else 20 * 60
    cached = mlb_starters._read_cache(cache, max_age=max_age)
    if cached is not None:
        return cached
    try:
        data = mlb_starters._get(
            "schedule", {"sportId": 1, "date": game_date, "hydrate": "linescore"})
    except Exception:
        return []
    games = []
    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            status = g.get("status", {}) or {}
            if status.get("abstractGameState") != "Final":
                continue
            # A postponed/suspended/cancelled game can report abstractGameState
            # "Final" with a 0-0 or partial score; excluding those keeps a
            # rained-out game's team bets pending (DK voids them) instead of
            # settling off a bogus box score. A rain-shortened OFFICIAL game reads
            # detailedState "Completed Early" and is intentionally NOT excluded.
            detailed = str(status.get("detailedState") or "").lower()
            if any(b in detailed for b in mlb_starters._NON_FINAL_DETAILED):
                continue
            teams = g.get("teams", {}) or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            hs = home.get("score")
            as_ = away.get("score")
            if not isinstance(hs, (int, float)):
                hs = (((g.get("linescore") or {}).get("teams", {}) or {}).get(
                    "home", {}) or {}).get("runs")
            if not isinstance(as_, (int, float)):
                as_ = (((g.get("linescore") or {}).get("teams", {}) or {}).get(
                    "away", {}) or {}).get("runs")
            if not isinstance(hs, (int, float)) or not isinstance(as_, (int, float)):
                continue
            games.append({
                "home_team": (home.get("team") or {}).get("name"),
                "away_team": (away.get("team") or {}).get("name"),
                "home_score": float(hs),
                "away_score": float(as_),
                "commence_time": g.get("gameDate"),
            })
    mlb_starters._write_cache(cache, games)
    return games


def _espn_scores_for_date(espn_sport, espn_league, game_date):
    """List of completed ESPN games on ``game_date`` (YYYY-MM-DD) with scores."""
    import requests
    yyyymmdd = str(game_date)[:10].replace("-", "")
    url = (f"https://site.api.espn.com/apis/site/v2/sports/"
           f"{espn_sport}/{espn_league}/scoreboard")
    try:
        resp = requests.get(url, params={"dates": yyyymmdd}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    games = []
    for event in data.get("events", []) or []:
        comp = (event.get("competitions") or [{}])[0]
        status = ((comp.get("status") or {}).get("type") or {})
        if not status.get("completed"):
            continue
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hs = float(home.get("score"))
            as_ = float(away.get("score"))
        except (TypeError, ValueError):
            continue
        games.append({
            "home_team": (home.get("team") or {}).get("displayName"),
            "away_team": (away.get("team") or {}).get("displayName"),
            "home_score": hs,
            "away_score": as_,
            "commence_time": event.get("date"),
        })
    return games


def _scores_for_date(sport_key, game_date):
    key = (sport_key, str(game_date)[:10])
    today = datetime.now(timezone.utc).date().isoformat()
    entry = _SCORE_CACHE.get(key)
    if entry is not None:
        fetched_at, out = entry
        # Past dates are immutable → reuse forever; a recent date may still be
        # finalizing, so re-fetch after the short memo TTL.
        if key[1] < today or (time.time() - fetched_at) < _RECENT_MEMO_TTL:
            return out
    if sport_key == "baseball_mlb":
        out = _mlb_scores_for_date(key[1])
    else:
        pair = _ESPN_MAP.get(sport_key)
        out = _espn_scores_for_date(pair[0], pair[1], key[1]) if pair else []
    _SCORE_CACHE[key] = (time.time(), out)
    return out


def final_score(sport_key, game_date, home_team, away_team, commence_time=None):
    """(home_score, away_score) for a completed game, or None.

    Scans ``game_date`` ±1 day (UTC/local slippage), matches by normalized team
    keys, and disambiguates same-day games by the start nearest commence_time."""
    if not sport_key or (sport_key != "baseball_mlb"
                         and sport_key not in _ESPN_MAP):
        return None
    hk, ak = _team_key(home_team), _team_key(away_team)
    if not hk or not ak:
        return None
    day0 = str(game_date)[:10]
    candidates = []          # (source_date, game_dict)
    seen = set()
    for d in _candidate_dates(game_date):
        for g in _scores_for_date(sport_key, d):
            if _team_key(g.get("home_team")) != hk:
                continue
            if _team_key(g.get("away_team")) != ak:
                continue
            marker = (g.get("commence_time"), g.get("home_score"),
                      g.get("away_score"))
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append((d, g))
    if not candidates:
        return None
    target = _parse_utc(commence_time)
    if target is not None:
        def _delta(item):
            gc = _parse_utc(item[1].get("commence_time"))
            return (abs((gc - target).total_seconds())
                    if gc is not None else float("inf"))
        candidates.sort(key=_delta)
        # The same matchup can appear on adjacent days (a series), or the exact-day
        # game can be missing from a stale final-only slate. Grading the nearest
        # match blindly would settle a bet on the WRONG night. If even the closest
        # game starts >20h from this wager's first pitch, it's a different game →
        # return None (stay pending, retry once fresh scores land). Legitimate
        # UTC/local slippage is only a few hours, so the real game is never rejected.
        if _delta(candidates[0]) > 20 * 3600:
            return None
    else:
        # No commence to disambiguate: trust only an exact game_date match, and
        # bail if the matchup is still ambiguous across the ±1-day window.
        exact = [c for c in candidates if c[0] == day0]
        if exact:
            candidates = exact
        if len(candidates) > 1:
            return None
    g = candidates[0][1]
    return (g["home_score"], g["away_score"])
