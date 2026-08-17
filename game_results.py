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
# (fetched_at_epoch, [game dicts], complete): a slate is memoized for the process
# lifetime only once COMPLETE — every game on the date has reached a terminal
# state, so the slate can no longer change. While any game is still scheduled or
# live the memo expires after a short TTL, so a game that finishes mid-session is
# never masked forever by the earlier (partial) slate that dropped it.
#
# COMPLETENESS — not the calendar date — is what makes a slate immutable. Keying
# immutability off "date < today (UTC)" was wrong: an MLB game_date is US-Eastern,
# so the moment UTC passed midnight (8pm ET) that evening's still-in-progress
# slate already looked "past" and its partial scores were cached forever, leaving
# every late game's bet/forecast pending until the process restarted.
_SCORE_CACHE = {}
_RECENT_MEMO_TTL = 20 * 60      # incomplete slate: re-fetch this often (in-process)
_SLATE_LIVE_TTL = 20 * 60       # incomplete slate on disk: trust only briefly
_SLATE_FINAL_TTL = 24 * 3600    # complete slate on disk: immutable → trust a day


def _team_key(name):
    """Normalize a team name for matching. For MLB, canonicalize through the SFBB
    team map (abbreviation / nickname / ESPN-code aware) so divergent feed spellings
    collapse to one stable 3-letter code; any miss (non-MLB name, SQL off, map
    unavailable) falls back to the alnum-lower normalization + curated aliases used
    across the whole grading/backtest infra. This is the single shared team key —
    backtest_market_consensus imports it."""
    try:
        import player_id_map
        code = player_id_map.team_code_for_name(name)
        if code:
            return code
    except Exception:            # fail open — map unavailable / not an MLB name
        pass
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


# Sentinel: the bet's game is POSITIVELY identified (by game_pk) but not yet final,
# so the caller must stay pending WITHOUT falling back to the name+date path (which
# could grade off a near-same-commence doubleheader sibling). Distinct from None
# (uncertain → fall back) and from a (status, score) result.
GRADE_PENDING = object()
_GRADE_PK_STALE_HOURS = 6   # a still-'live' warehouse row this long past commence is
                            # stale (ingest lag/miss) → stop waiting, fall back


def _grade_pk_stale(commence_time, hours=_GRADE_PK_STALE_HOURS):
    """True when ``commence_time`` (ISO ts) + ``hours`` is in the past. False when
    commence is missing/unparseable/naive — a stamped game_pk's commence is normally
    tz-aware & parseable (find_game_pk_by_commence required it), so the rare
    unparseable case stays pending (mis-grade-safe) rather than falling back."""
    if not commence_time:
        return False
    try:
        from datetime import datetime, timezone, timedelta
        ts = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            return False
        return datetime.now(timezone.utc) >= ts + timedelta(hours=hours)
    except (ValueError, TypeError):
        return False


def grade_team_bet_by_game_pk(sport_key, game_pk, bet_type, side, team, point):
    """DH-exact fast path (Tier A #2): grade a team bet off the warehouse game
    identified by ``game_pk`` instead of the name+date slate. Returns:
      (status, 'hs-as')  graded from the warehouse final score;
      GRADE_PENDING      game positively identified but not yet final — stay pending
                         (caller must NOT fall back to name+date);
      None               anything uncertain → caller falls back to name+date
                         (byte-identical to today).
    MLB-only (hard sport gate). ML/spread orientation is derived from the bet team's
    MLBAM id vs the warehouse game's home/away ids — never a stored 'side' label.
    Every uncertainty (non-MLB, no pk, warehouse off, missing row, terminal game,
    team unmappable, unknown bet_type, any exception) → None."""
    if sport_key != "baseball_mlb" or not game_pk:
        return None
    try:
        import mlb_warehouse
        fg = mlb_warehouse.final_game_by_pk(game_pk)
        if not fg:
            return None
        state = fg["state"]
        if state == "terminal":
            return None                    # postponed/suspended → name+date (makeup)
        if state == "live":
            # Positively-identified but not final → stay pending so a DH sibling's
            # already-final score can't grade this game via name+date. Staleness
            # guard so a lagging/missed ingest can't strand the bet forever: once
            # commence + N hours has passed and the row STILL isn't final, fall back
            # to the (fresh-StatsAPI) name+date path. Safe: a stamped pk is always
            # commence-disambiguable, so name+date can't cross-grade a simultaneous
            # DH sibling.
            if _grade_pk_stale(fg.get("commence_time")):
                return None
            return GRADE_PENDING
        # state == 'final'
        if (bet_type or "").lower() == "total":
            graded_side = side             # over/under is orientation-invariant
        else:
            tid = mlb_warehouse.team_id_for_name_tolerant(team)
            if tid is not None and str(tid) == str(fg.get("home_team_id")):
                graded_side = "home"
            elif tid is not None and str(tid) == str(fg.get("away_team_id")):
                graded_side = "away"
            else:
                return None                # can't confirm orientation → fall back
        status = grade_team_bet(bet_type, graded_side, point,
                                fg["home_score"], fg["away_score"])
        if status is None:
            return None
        return (status, f"{fg['home_score']:g}-{fg['away_score']:g}")
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-date final-score fetch
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap_slate(cached):
    """(games, complete) from a cached slate payload.

    New caches store ``{"games": [...], "complete": bool}``; a legacy bare-list
    cache has unknown completeness, so treat it as incomplete — it then rides the
    short freshness window and self-heals into the new format on the next fetch."""
    if isinstance(cached, dict):
        return list(cached.get("games") or []), bool(cached.get("complete"))
    if isinstance(cached, list):
        return cached, False
    return [], False


def _mlb_slate_for_date(game_date):
    """(games, complete) for one MLB date (YYYY-MM-DD).

    ``games`` is the list of genuine-final games with scores; ``complete`` is True
    only when EVERY scheduled game that day has reached a terminal state, so the
    slate can no longer change and is safe to cache indefinitely. A still-live or
    postponed/suspended game keeps ``complete`` False, so the date keeps
    refreshing on the short window until it truly settles — this is what a bare
    "date < today (UTC)" check got wrong for late US (Eastern) games.

    Uses the free statsapi schedule with a linescore hydrate as a run fallback."""
    try:
        import mlb_starters
    except Exception:
        return [], False
    cache = f"final_scores_mlb_{game_date}"
    # A complete slate is immutable → trust the day-long cache; an incomplete one
    # (some games still live) is trusted only for the short live window so the
    # not-yet-final games are picked up as they end.
    cached = mlb_starters._read_cache(cache, max_age=_SLATE_FINAL_TTL)
    if cached is not None:
        games, complete = _unwrap_slate(cached)
        if complete:
            return games, True
    fresh = mlb_starters._read_cache(cache, max_age=_SLATE_LIVE_TTL)
    if fresh is not None:
        return _unwrap_slate(fresh)
    try:
        data = mlb_starters._get(
            "schedule", {"sportId": 1, "date": game_date, "hydrate": "linescore"})
    except Exception:
        return [], False
    games = []
    total = 0
    all_final = True
    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            total += 1
            status = g.get("status", {}) or {}
            # A postponed/suspended/cancelled game can report abstractGameState
            # "Final" with a 0-0 or partial score; excluding those keeps a
            # rained-out game's team bets pending (DK voids them) instead of
            # settling off a bogus box score. A rain-shortened OFFICIAL game reads
            # detailedState "Completed Early" and is intentionally NOT excluded.
            detailed = str(status.get("detailedState") or "").lower()
            postponed = any(b in detailed for b in mlb_starters._NON_FINAL_DETAILED)
            if status.get("abstractGameState") != "Final" or postponed:
                # Still live/scheduled (or postponed): the slate can still change,
                # so it is not immutable yet. A postponed game never becomes a
                # gradable score on THIS date, so its bet simply stays pending.
                all_final = False
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
                all_final = False  # final but no score yet → still settling
                continue
            games.append({
                "home_team": (home.get("team") or {}).get("name"),
                "away_team": (away.get("team") or {}).get("name"),
                "home_score": float(hs),
                "away_score": float(as_),
                "commence_time": g.get("gameDate"),
            })
    complete = total > 0 and all_final
    mlb_starters._write_cache(cache, {"games": games, "complete": complete})
    return games, complete


def _mlb_scores_for_date(game_date):
    """List of genuine-final MLB games on ``game_date`` (back-compat shim)."""
    return _mlb_slate_for_date(game_date)[0]


def _espn_slate_for_date(espn_sport, espn_league, game_date):
    """(games, complete) for one ESPN date (YYYY-MM-DD).

    ``complete`` is True only when there is at least one event and EVERY event on
    the scoreboard is completed, so the slate is immutable; a live or postponed
    event keeps it False so the date keeps refreshing until it truly settles."""
    import requests
    yyyymmdd = str(game_date)[:10].replace("-", "")
    url = (f"https://site.api.espn.com/apis/site/v2/sports/"
           f"{espn_sport}/{espn_league}/scoreboard")
    try:
        resp = requests.get(url, params={"dates": yyyymmdd}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [], False
    games = []
    total = 0
    all_complete = True
    for event in data.get("events", []) or []:
        total += 1
        comp = (event.get("competitions") or [{}])[0]
        status = ((comp.get("status") or {}).get("type") or {})
        if not status.get("completed"):
            all_complete = False
            continue
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            all_complete = False
            continue
        try:
            hs = float(home.get("score"))
            as_ = float(away.get("score"))
        except (TypeError, ValueError):
            all_complete = False
            continue
        games.append({
            "home_team": (home.get("team") or {}).get("displayName"),
            "away_team": (away.get("team") or {}).get("displayName"),
            "home_score": hs,
            "away_score": as_,
            "commence_time": event.get("date"),
        })
    return games, (total > 0 and all_complete)


def _espn_scores_for_date(espn_sport, espn_league, game_date):
    """List of completed ESPN games on ``game_date`` (back-compat shim)."""
    return _espn_slate_for_date(espn_sport, espn_league, game_date)[0]


def _scores_for_date(sport_key, game_date):
    key = (sport_key, str(game_date)[:10])
    entry = _SCORE_CACHE.get(key)
    if entry is not None:
        fetched_at, out, complete = entry
        # Reuse for the process lifetime only once the slate is COMPLETE (every
        # game final → immutable); while any game is still live, re-fetch after
        # the short memo TTL so a game that finishes mid-session isn't masked by
        # the earlier partial slate. (Completeness, not the calendar date: a late
        # Eastern game's UTC date rolls over while it is still being played.)
        if complete or (time.time() - fetched_at) < _RECENT_MEMO_TTL:
            return out
    if sport_key == "baseball_mlb":
        out, complete = _mlb_slate_for_date(key[1])
    else:
        pair = _ESPN_MAP.get(sport_key)
        out, complete = (_espn_slate_for_date(pair[0], pair[1], key[1])
                         if pair else ([], False))
    _SCORE_CACHE[key] = (time.time(), out, complete)
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
