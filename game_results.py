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
from datetime import date as _date, datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# sport_key -> (espn_sport, espn_league). MLB is handled separately (statsapi).
_ESPN_MAP = {
    "basketball_nba": ("basketball", "nba"),
    "americanfootball_nfl": ("football", "nfl"),
    "icehockey_nhl": ("hockey", "nhl"),
}

# In-process memo so one grading pass reuses a date's scoreboard fetch.
_SCORE_CACHE = {}  # (sport_key, game_date) -> [game dicts]


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
    cached = mlb_starters._read_cache(cache, max_age=6 * 3600)
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
    if key in _SCORE_CACHE:
        return _SCORE_CACHE[key]
    if sport_key == "baseball_mlb":
        out = _mlb_scores_for_date(key[1])
    else:
        pair = _ESPN_MAP.get(sport_key)
        out = _espn_scores_for_date(pair[0], pair[1], key[1]) if pair else []
    _SCORE_CACHE[key] = out
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
    candidates = []
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
            candidates.append(g)
    if not candidates:
        return None
    if len(candidates) > 1:
        target = _parse_utc(commence_time)
        if target is not None:
            candidates.sort(key=lambda g: abs(
                ((_parse_utc(g.get("commence_time")) or target)
                 - target).total_seconds()))
    g = candidates[0]
    return (g["home_score"], g["away_score"])
