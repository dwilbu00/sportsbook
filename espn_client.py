"""
Client for the unofficial ESPN API.
Fetches team records, recent results, and scoring averages.
No authentication required.
"""

import os
import sys
import requests
from datetime import datetime, timedelta

# Optional SQL backend (Phase C durable gamelog store). Guarded so a missing
# SQLAlchemy install just leaves SQL disabled and the direct-fetch path is used.
# gamelog_store is imported LAZILY inside get_player_stat_history to avoid an
# import cycle (gamelog_store imports espn_client).
try:
    import db_store
except Exception:  # pragma: no cover - import guard
    db_store = None


SITE_API = "https://site.api.espn.com/apis/site/v2/sports"
STANDINGS_API = "https://site.api.espn.com/apis/v2/sports"
CORE_API = "https://sports.core.api.espn.com/v2/sports"

# Common timeout for all ESPN requests
TIMEOUT = 15


def _warn(message):
    """Surface an otherwise-swallowed ESPN failure to the server log.

    These calls fail closed (return None/[]) so the app keeps working, but a
    silent failure is indistinguishable from a player genuinely having no data
    and can quietly degrade projections. Emitting to stderr makes an outage or
    a systematic parse break visible to operators without crashing analysis."""
    print(f"[espn_client] {message}", file=sys.stderr)


def get_all_teams(sport, league):
    """
    Fetch all teams for a sport/league.

    Returns:
        dict: Mapping of team displayName -> team info dict
    """
    url = f"{SITE_API}/{sport}/{league}/teams"
    params = {"limit": 100}
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    teams = {}
    for entry in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = entry.get("team", {})
        display_name = team.get("displayName", "")
        record_items = team.get("record", {}).get("items", [])

        overall_record = ""
        wins = 0
        losses = 0
        for rec in record_items:
            if rec.get("type") == "total" or rec.get("description") == "Overall Record":
                overall_record = rec.get("summary", "")
                stats = {s["name"]: s["value"] for s in rec.get("stats", [])}
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                break

        # Fallback: parse summary string if stats not available
        if not wins and not losses and overall_record:
            parts = overall_record.split("-")
            if len(parts) >= 2:
                try:
                    wins = int(parts[0])
                    losses = int(parts[1])
                except ValueError:
                    pass

        teams[display_name] = {
            "id": team.get("id"),
            "abbreviation": team.get("abbreviation", ""),
            "display_name": display_name,
            "short_name": team.get("shortDisplayName", ""),
            "record": overall_record,
            "wins": wins,
            "losses": losses,
            "win_pct": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
        }

    # ESPN's league-wide teams endpoint no longer includes records. Merge the
    # official regular-season records from the standings endpoint, which
    # returns every team in one additional request. Retain the values parsed
    # above as a fallback if standings are temporarily unavailable.
    standings_url = f"{STANDINGS_API}/{sport}/{league}/standings"
    try:
        standings_resp = requests.get(standings_url, timeout=TIMEOUT)
        standings_resp.raise_for_status()
        standings_data = standings_resp.json()
    except (requests.RequestException, ValueError):
        return teams

    teams_by_id = {info["id"]: info for info in teams.values() if info.get("id")}
    groups = [standings_data]
    while groups:
        group = groups.pop()
        groups.extend(group.get("children", []))
        entries = group.get("standings", {}).get("entries", [])
        for entry in entries:
            team_id = entry.get("team", {}).get("id")
            team_info = teams_by_id.get(team_id)
            if not team_info:
                continue

            stats = {
                stat.get("name"): stat.get("value")
                for stat in entry.get("stats", [])
                if stat.get("name")
            }
            try:
                wins = int(stats.get("wins", 0))
                losses = int(stats.get("losses", 0))
                ties = int(stats.get("ties", 0))
            except (TypeError, ValueError):
                continue

            games = wins + losses + ties
            if games <= 0:
                continue

            win_pct = stats.get("winPercent")
            if not isinstance(win_pct, (int, float)):
                win_pct = (wins + 0.5 * ties) / games

            team_info["record"] = (
                f"{wins}-{losses}-{ties}" if ties else f"{wins}-{losses}"
            )
            team_info["wins"] = wins
            team_info["losses"] = losses
            team_info["win_pct"] = win_pct

    return teams


def get_team_schedule(sport, league, team_id, season_year=None):
    """
    Fetch a team's schedule/results for a season.

    Fetches BOTH the regular season (seasontype=2) and postseason
    (seasontype=3) and merges the results. Without explicit seasontype,
    ESPN's endpoint defaults to the league's *current* season type — e.g.,
    during the NBA playoffs it returns only postseason games, leaving
    non-playoff teams with empty schedules. Querying both types explicitly
    fixes that.

    Returns:
        list: List of completed game result dicts (most recent first, deduped)
    """
    url = f"{SITE_API}/{sport}/{league}/teams/{team_id}/schedule"
    games = []
    seen = set()
    for seasontype in (2, 3):
        params = {"seasontype": seasontype}
        if season_year:
            params["season"] = season_year
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        for event in data.get("events", []):
            status_type = event.get("competitions", [{}])[0].get("status", {}).get("type", {})
            if status_type.get("completed", False) is not True:
                continue

            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            if len(competitors) < 2:
                continue

            home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

            home_score = int(home_comp.get("score", {}).get("value", home_comp.get("score", 0)))
            away_score = int(away_comp.get("score", {}).get("value", away_comp.get("score", 0)))

            date = event.get("date", "")
            home_name = home_comp.get("team", {}).get("displayName", "")
            away_name = away_comp.get("team", {}).get("displayName", "")
            key = (date, home_name, away_name)
            if key in seen:
                continue
            seen.add(key)

            games.append({
                "date": date,
                "home_team": home_name,
                "away_team": away_name,
                "home_score": home_score,
                "away_score": away_score,
                "total_score": home_score + away_score,
            })

    # Sort by date descending (most recent first)
    games.sort(key=lambda g: g["date"], reverse=True)
    return games


def get_team_pace_factor(sport, league, team_id, season_year=None, seasontype=2):
    """
    Fetch a team's pace factor (possessions per game proxy) from the
    ESPN core stats API. Returns None when unavailable.

    Endpoint: /v2/sports/{sport}/leagues/{league}/seasons/{year}/types/{type}/teams/{id}/statistics

    Season-year inference (when not provided): for sports whose season spans
    two calendar years (NBA/NHL Oct→Jun), ESPN labels the season by its end
    year. We try the most likely candidate first, then fall back to the prior
    year (e.g., during the offseason, before the new season begins).
    """
    if season_year is not None:
        candidates = [season_year]
    else:
        now = datetime.now()
        if now.month >= 10:
            # Oct-Dec: new season has begun, end year is next calendar year
            candidates = [now.year + 1, now.year]
        else:
            # Jan-Sep: either tail of last season or offseason. Try current
            # year first (season ending this calendar year), then prior.
            candidates = [now.year, now.year - 1]

    for year in candidates:
        url = (f"{CORE_API}/{sport}/leagues/{league}/seasons/{year}"
               f"/types/{seasontype}/teams/{team_id}/statistics")
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        splits = data.get("splits", {})
        if not isinstance(splits, dict):
            continue
        for cat in splits.get("categories", []):
            for stat in cat.get("stats", []):
                if stat.get("name") == "paceFactor":
                    val = stat.get("value")
                    try:
                        if val is not None:
                            return float(val)
                    except (TypeError, ValueError):
                        pass
    return None


def get_team_record_and_stats(sport, league, team_id):
    """
    Fetch detailed team record including home/away splits.

    Returns:
        dict: Record details with overall, home, and away records
    """
    url = f"{SITE_API}/{sport}/{league}/teams/{team_id}"
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    team_data = data.get("team", {})
    records = {}

    for rec in team_data.get("record", {}).get("items", []):
        rec_type = rec.get("type", rec.get("description", "unknown"))
        stats = {s["name"]: s["value"] for s in rec.get("stats", [])}
        records[rec_type] = {
            "summary": rec.get("summary", ""),
            "wins": int(stats.get("wins", 0)),
            "losses": int(stats.get("losses", 0)),
            "win_pct": stats.get("winPercent", 0.0),
        }

    return records


def compute_recent_form(games, team_name, n=10):
    """
    Compute recent form stats from the last N completed games for a team.

    Parameters:
        games (list): Game results from get_team_schedule()
        team_name (str): Team display name
        n (int): Number of recent games to consider

    Returns:
        dict: Recent form stats
    """
    recent = games[:n]
    if not recent:
        return {"games": 0, "wins": 0, "losses": 0, "win_pct": 0.0,
                "avg_scored": 0.0, "avg_allowed": 0.0, "avg_total": 0.0}

    wins = 0
    losses = 0
    points_scored = []
    points_allowed = []

    for game in recent:
        if game["home_team"] == team_name:
            scored = game["home_score"]
            allowed = game["away_score"]
        elif game["away_team"] == team_name:
            scored = game["away_score"]
            allowed = game["home_score"]
        else:
            continue

        if scored > allowed:
            wins += 1
        else:
            losses += 1

        points_scored.append(scored)
        points_allowed.append(allowed)

    total_games = wins + losses
    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "win_pct": wins / total_games if total_games > 0 else 0.0,
        "avg_scored": sum(points_scored) / len(points_scored) if points_scored else 0.0,
        "avg_allowed": sum(points_allowed) / len(points_allowed) if points_allowed else 0.0,
        "avg_total": sum(s + a for s, a in zip(points_scored, points_allowed)) / len(points_scored) if points_scored else 0.0,
    }


def compute_team_defense(games, team_name):
    """
    Compute average points allowed per game for a team from their schedule.

    Returns:
        float: Avg points allowed, or None if no completed games for the team.
    """
    allowed = []
    for g in games:
        if g.get("home_team") == team_name:
            allowed.append(g.get("away_score", 0))
        elif g.get("away_team") == team_name:
            allowed.append(g.get("home_score", 0))
    if not allowed:
        return None
    return sum(allowed) / len(allowed)


def build_team_defense_lookup(schedule_results, teams_dict):
    """
    Build a {team_display_name: avg_points_allowed} lookup from previously
    fetched schedule data.

    Parameters:
        schedule_results (dict): {team_id: list_of_games} as produced by
            get_team_schedule (one entry per team you've already fetched).
        teams_dict (dict): ESPN teams keyed by displayName (id → name reverse).

    Returns:
        dict: {display_name: avg_points_allowed_per_game}
    """
    id_to_name = {info["id"]: name for name, info in teams_dict.items()}
    lookup = {}
    for team_id, games in schedule_results.items():
        team_name = id_to_name.get(team_id)
        if not team_name or not games:
            continue
        avg_allowed = compute_team_defense(games, team_name)
        if avg_allowed is not None:
            lookup[team_name] = avg_allowed
    return lookup


def annotate_opponent_strength(games, team_name, teams_dict):
    """
    Augment each game dict in-place with an 'opponent_win_pct' field
    looked up from the ESPN teams dict. Falls back to 0.5 (average) when
    the opponent can't be matched.

    Parameters:
        games (list): Game result dicts from get_team_schedule().
        team_name (str): The team's own display name (to identify opponent).
        teams_dict (dict): ESPN teams keyed by displayName.

    Returns:
        list: The same games list, with 'opponent_win_pct' added per entry.
    """
    for g in games:
        if g.get("home_team") == team_name:
            opp = g.get("away_team")
        elif g.get("away_team") == team_name:
            opp = g.get("home_team")
        else:
            opp = None
        opp_info = find_team(teams_dict, opp) if opp else None
        g["opponent_win_pct"] = opp_info.get("win_pct", 0.5) if opp_info else 0.5
    return games


def find_team(teams_dict, search_name):
    """
    Find a team in the ESPN teams dict by matching against the odds API team name.
    Tries exact match first, then substring matching.

    Parameters:
        teams_dict (dict): ESPN teams keyed by displayName
        search_name (str): Team name from the odds API

    Returns:
        dict or None: Team info dict
    """
    # Exact match
    if search_name in teams_dict:
        return teams_dict[search_name]

    # Normalized match (case-insensitive)
    search_lower = search_name.lower()
    for name, info in teams_dict.items():
        if name.lower() == search_lower:
            return info

    # Substring match (e.g., "Los Angeles Lakers" contains "Lakers")
    for name, info in teams_dict.items():
        if search_lower in name.lower() or name.lower() in search_lower:
            return info

    return None


def list_season_athletes(sport, league, season_year, seasontype=2,
                         limit=250, max_pages=6):
    """
    Return season-long per-athlete stats for a league (free, no key).

    Hits ESPN's public `statistics/byathlete` endpoint and returns a list of
        {"id", "name", "position", "games", "minutes", "avg_minutes"}
    for every athlete carrying a `general` stat block, across up to `max_pages`
    pages. Used to build a usage-representative calibration pool (top-N by
    minutes) instead of a hand-picked star list.

    Column values are mapped to their names via the response's top-level
    `categories[].names` (zipped positionally with each athlete's category
    `values`), so a column reorder won't silently mis-read a stat.

    Fails closed as a WHOLE: the byathlete feed is NOT minutes-ordered, so a
    page-1-only subset omits high-minute rotation players stranded on later
    pages. If any page errors mid-pagination — or `max_pages` truncates before
    the last page is reached — the function warns and returns [] (discarding the
    partial sample) so `_nba_player_pool` -> `refit_sport` aborts loudly rather
    than silently fitting a ranking-biased subset. Only a clean traversal to the
    last page (or an empty page) returns data. `max_pages` is a runaway safety
    cap, not a sampling knob; the default comfortably covers a full league.
    """
    url = (f"https://site.web.api.espn.com/apis/common/v3/sports/"
           f"{sport}/{league}/statistics/byathlete")
    base_params = {
        "region": "us", "lang": "en", "contentorigin": "espn",
        "isqualified": "false", "seasontype": seasontype, "limit": limit,
    }
    if season_year:
        base_params["season"] = season_year

    out = []
    complete = False
    page = 1
    while page <= max_pages:
        try:
            resp = requests.get(url, params=dict(base_params, page=page),
                                timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            _warn(f"list_season_athletes failed for {sport}/{league} "
                  f"season={season_year} page={page}: "
                  f"{type(exc).__name__}: {exc}")
            break  # incomplete: `complete` stays False -> return [] below

        # Column names are defined once at the top level, per category.
        columns = {cat.get("name"): (cat.get("names") or [])
                   for cat in data.get("categories", []) if cat.get("name")}

        athletes = data.get("athletes", [])
        for entry in athletes:
            parsed = _parse_byathlete_entry(entry, columns)
            if parsed:
                out.append(parsed)

        # Completeness is decided by page traversal, NOT by comparing the count
        # to `pagination.count` — legitimate parse skips (rows without an id)
        # make `len(out) < count` on a perfectly complete fetch.
        if not athletes:
            complete = True
            break
        total_pages = (data.get("pagination") or {}).get("pages")
        if total_pages and page >= total_pages:
            complete = True
            break
        page += 1

    if not complete:
        _warn(f"list_season_athletes incomplete for {sport}/{league} "
              f"season={season_year}: stopped at page {page} with {len(out)} "
              f"athletes collected — returning [] (fail closed)")
        return []
    return out


def _parse_byathlete_entry(entry, columns):
    """
    Extract id/name/position/games/minutes from one `byathlete` row.

    `columns` maps a category name (e.g. "general") to its list of stat column
    names, taken from the response's top-level `categories`. Returns None when
    the row has no usable athlete id/name.
    """
    athlete = entry.get("athlete") or {}
    aid = athlete.get("id")
    name = athlete.get("displayName") or athlete.get("fullName")
    if not aid or not name:
        return None

    def stat(category, stat_name):
        names = columns.get(category) or []
        if stat_name not in names:
            return None
        idx = names.index(stat_name)
        for cat in entry.get("categories", []):
            if cat.get("name") == category:
                values = cat.get("values") or []
                if idx < len(values):
                    try:
                        return float(values[idx])
                    except (TypeError, ValueError):
                        return None
        return None

    position = (athlete.get("position") or {}).get("abbreviation")
    return {
        "id": str(aid),
        "name": name,
        "position": position,
        "games": stat("general", "gamesPlayed"),
        "minutes": stat("general", "minutes"),
        "avg_minutes": stat("general", "avgMinutes"),
    }


def search_athlete(sport, league, name, team_ids=None):
    """
    Search for an athlete by name.
    Tries the site API first (works for NBA/NFL), falls back to web search API (works for MLB).

    Parameters:
        sport (str): ESPN sport (e.g., 'basketball')
        league (str): ESPN league (e.g., 'nba')
        name (str): Player name to search for
        team_ids (iterable, optional): ESPN team ids for the game this player is
            in (typically the two teams of the matchup). When provided, a
            candidate whose team matches one of these is preferred over ESPN's
            raw first result — disambiguating same-name players (e.g. two
            "Will Smith"s) so we don't project the wrong athlete's stats onto a
            bet. Falls back to the first result when no candidate matches, so
            behavior is never worse than the previous first-match-wins logic.

    Returns:
        dict or None: {'id': str, 'name': str, 'team_id': str|None} or None
        if not found
    """
    wanted_teams = {str(t) for t in team_ids if t} if team_ids else set()

    def _result(athlete):
        team = athlete.get("team") or {}
        team_id = team.get("id") if isinstance(team, dict) else None
        if team_id is None:
            for relationship in athlete.get("teamRelationships") or []:
                if relationship.get("type") != "team":
                    continue
                team_id = (relationship.get("core") or {}).get("id")
                if team_id is not None:
                    break
        return {
            "id": str(athlete.get("id", "")),
            "name": athlete.get(
                "displayName", athlete.get("fullName", name)),
            "team_id": str(team_id) if team_id is not None else None,
        }

    def _pick(raw_candidates):
        """Map raw ESPN entries to results and pick the best one.

        Prefers a candidate on one of ``wanted_teams``; otherwise the first
        (ESPN's own ranking), matching the historical behavior.
        """
        candidates = [_result(c) for c in raw_candidates
                      if isinstance(c, dict)]
        if not candidates:
            return None
        if wanted_teams:
            for c in candidates:
                if c.get("team_id") and c["team_id"] in wanted_teams:
                    return c
        return candidates[0]

    # Try site API first (works for NBA, NFL)
    try:
        url = f"{SITE_API}/{sport}/{league}/athletes"
        params = {"search": name, "limit": 5}
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            picked = _pick(data.get("athletes") or data.get("items") or [])
            if picked:
                return picked
    except Exception as exc:
        # Keep the broad catch here so a site-API hiccup still falls through to
        # the web-search fallback below, but no longer swallow it silently.
        _warn(f"search_athlete site API failed for "
              f"{sport}/{league} {name!r}: {type(exc).__name__}: {exc}")

    # Fallback: web search API (works for MLB and all sports)
    try:
        url = "https://site.web.api.espn.com/apis/common/v3/search"
        params = {"query": name, "limit": 5, "type": "player"}
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            # The web search is cross-sport (a name can hit MLB/NHL/CFB/etc.), so
            # restrict to the requested sport/league and apply the team hint only
            # within that set. If nothing matches, fall back to the first player
            # result exactly as before — do NOT team-prefer across a cross-sport
            # list (team ids overlap numerically between sports, so a coincidental
            # match could otherwise surface a wrong-sport athlete).
            matched = [it for it in items
                       if it.get("sport") == sport and it.get("league") == league]
            if matched:
                picked = _pick(matched)
            elif items:
                picked = _result(items[0])
            else:
                picked = None
            if picked:
                return picked
    except Exception as exc:
        _warn(f"search_athlete web search failed for "
              f"{sport}/{league} {name!r}: {type(exc).__name__}: {exc}")

    return None


def _game_completed(ev):
    """Best-effort 'is this gamelog game final?' from an ESPN event dict.

    Returns True/False when ESPN gives a clear signal, else None (unknown). A
    final game carries a ``gameResult`` (W/L/D) or a completed status flag; a
    live game has neither — its running score is deliberately NOT treated as
    completion (that would grade a partial line). The outcome resolver combines
    an unknown (None) result with the game's date to stay fail-safe (past-dated
    games are final; a same-day game with no completion signal stays pending)."""
    if not isinstance(ev, dict):
        return None
    result = ev.get("gameResult") or ev.get("result")
    if isinstance(result, str) and result.strip():
        return True
    status = ev.get("status")
    if isinstance(status, dict):
        stype = status.get("type")
        if isinstance(stype, dict) and "completed" in stype:
            return bool(stype.get("completed"))
        if "completed" in status:
            return bool(status.get("completed"))
    return None


def get_athlete_gamelog(sport, league, athlete_id, season_year=None):
    """
    Fetch game log (game-by-game stats) for an athlete.
    Handles two ESPN response formats:
      - NBA/NFL: seasonTypes -> categories -> events with stats
      - MLB batters: top-level labels + events dict keyed by event ID

    Parameters:
        sport (str): ESPN sport
        league (str): ESPN league
        athlete_id (str): ESPN athlete ID
        season_year (int|None): If provided, request ESPN's gamelog for that
            season year (e.g., 2024 for the 2024 MLB season, or for NBA the
            season ending in 2024). When None, ESPN returns the current
            season by default.

    Returns:
        list: List of dicts with stat values per game, e.g.:
              [{'PTS': 25, 'AST': 5, 'REB': 10, ...}, ...]
              Returns empty list on failure.
    """
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/{sport}/{league}/athletes/{athlete_id}/gamelog"
    params = {}
    if season_year:
        params["season"] = season_year
    try:
        resp = requests.get(url, params=params or None, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        # Network fault or malformed JSON. Fail closed (callers treat [] as
        # "no games"), but surface it: a swallowed outage is otherwise
        # indistinguishable from a player who genuinely has no gamelog.
        _warn(f"get_athlete_gamelog failed for {sport}/{league} "
              f"athlete {athlete_id}: {type(exc).__name__}: {exc}")
        return []

    games = []

    # Top-level labels (used by MLB and sometimes shared across categories)
    top_labels = data.get("labels", [])

    # Top-level events dict carries opponent metadata keyed by event id.
    top_events = data.get("events", {})
    if not isinstance(top_events, dict):
        top_events = {}

    def _resolve_event(ev_or_id):
        return ev_or_id if isinstance(ev_or_id, dict) else top_events.get(str(ev_or_id), {})

    def _opponent_name(ev_or_id):
        ev = _resolve_event(ev_or_id)
        opp = ev.get("opponent") or {}
        return opp.get("displayName") or opp.get("name") or None

    def _is_home(ev_or_id):
        """atVs == 'vs' means the player's team was at home; '@' means away."""
        ev = _resolve_event(ev_or_id)
        atvs = ev.get("atVs")
        if atvs is None:
            return None
        return atvs.strip().lower() == "vs"

    def _team_id(ev_or_id):
        ev = _resolve_event(ev_or_id)
        team = ev.get("team") or {}
        tid = team.get("id")
        return str(tid) if tid is not None else None

    def _game_date(ev_or_id):
        ev = _resolve_event(ev_or_id)
        return ev.get("gameDate") or ev.get("date") or None

    # Format 1: seasonTypes -> categories -> events
    # NBA/NFL categories have their own labels.
    # MLB batter categories may NOT have labels — use top-level labels instead.
    season_types = data.get("seasonTypes", [])
    for st in season_types:
        categories = st.get("categories", [])
        for cat in categories:
            labels = cat.get("labels", []) or top_labels
            if not labels:
                continue
            events = cat.get("events", [])
            for event in events:
                stats_list = event.get("stats", [])
                if len(stats_list) != len(labels):
                    continue
                game_stats = _parse_stat_row(labels, stats_list)
                eid_key = event.get("eventId") or event.get("id")
                game_stats["opponent"] = _opponent_name(eid_key)
                game_stats["is_home"] = _is_home(eid_key)
                game_stats["team_id"] = _team_id(eid_key)
                game_stats["game_date"] = _game_date(eid_key)
                game_stats["completed"] = _game_completed(_resolve_event(eid_key))
                games.append(game_stats)

    if games:
        return games

    # Format 2: MLB top-level events dict (keyed by event ID)
    if top_labels and top_events:
        for event_id, event in top_events.items():
            stats_list = event.get("stats", [])
            if len(stats_list) == len(top_labels):
                game_stats = _parse_stat_row(top_labels, stats_list)
                game_stats["opponent"] = _opponent_name(event)
                game_stats["is_home"] = _is_home(event)
                game_stats["team_id"] = _team_id(event)
                game_stats["game_date"] = _game_date(event)
                game_stats["completed"] = _game_completed(event)
                games.append(game_stats)

    return games


def ip_to_outs(ip):
    """Convert baseball innings-pitched notation to a raw out count.

    IP is recorded so the fractional digit counts OUTS, not tenths: X.1 = X
    innings + 1 out, X.2 = X + 2 outs (X.0 = X exactly). So 6.1 IP = 19 outs.
    ``pitcher_outs`` is the ONLY prop whose ESPN stat label ('IP') needs this
    conversion; every read of that label for pitcher_outs must pass through here
    before any arithmetic (average / compare / grade) or the thirds get treated
    as decimals. Returns None for None; assumes valid .0/.1/.2 notation."""
    if ip is None:
        return None
    whole = int(ip)
    frac = round((ip - whole) * 10)
    return whole * 3 + frac


def outs_to_ip(outs):
    """Inverse of ip_to_outs: an out count -> IP notation (X.0/.1/.2).

    Used to synthesize per-game IP estimates (get_pitcher_stats) that downstream
    code converts back via ip_to_outs — so the average must be taken in OUT space
    and re-encoded here, never divided as a decimal. Rounds a fractional out
    count to the nearest whole out."""
    if outs is None:
        return None
    total = int(round(outs))
    whole = total // 3
    rem = total - whole * 3
    return whole + rem / 10.0


def get_pitcher_stats(league, athlete_id, season=None):
    """
    Fetch pitcher stats from ESPN splits endpoint.
    The gamelog endpoint doesn't support MLB pitchers, so we use splits
    to get per-opponent game data which approximates game-by-game stats.

    Parameters:
        league (str): ESPN league (e.g., 'mlb')
        athlete_id (str): ESPN athlete ID
        season (int): Season year (default: current)

    Returns:
        list: List of dicts with per-game-approximated pitching stats,
              each containing keys like 'K', 'IP', 'H', 'ER', 'BB', 'GP'
    """
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/baseball/{league}/athletes/{athlete_id}/splits"
    params = {}
    if season:
        params["season"] = season

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    labels = data.get("labels", [])
    if not labels:
        return []

    # Get overall season totals to compute per-start averages
    games = []
    overall_stats = None

    for sc in data.get("splitCategories", []):
        cat_name = sc.get("displayName", "")

        if cat_name == "Overall":
            for sp in sc.get("splits", []):
                stats = sp.get("stats", [])
                if len(labels) == len(stats):
                    overall_stats = _parse_stat_row(labels, stats)

        # Per-opponent splits give us the closest to game-by-game data
        if cat_name == "Opponent":
            for sp in sc.get("splits", []):
                stats = sp.get("stats", [])
                if len(labels) == len(stats):
                    row = _parse_stat_row(labels, stats)
                    gp = row.get("GP", 1)
                    # Expand per-opponent aggregates into per-game estimates
                    for _ in range(max(1, int(gp))):
                        game_row = {}
                        for key, val in row.items():
                            if key in ("ERA", "WHIP", "W%", "OBA"):
                                game_row[key] = val  # rates stay as-is
                            elif key == "GP":
                                game_row[key] = 1
                            elif key == "IP":
                                # IP is base-3 notation: average in OUT space,
                                # then re-encode so the downstream ip_to_outs
                                # converter reads it correctly (dividing the
                                # notation as a decimal scrambles the outs).
                                game_row[key] = (outs_to_ip(ip_to_outs(val) / gp)
                                                 if gp > 0 else val)
                            else:
                                game_row[key] = round(val / gp, 1) if gp > 0 else val
                        games.append(game_row)

    # If no per-opponent data, create per-start averages from overall
    if not games and overall_stats:
        gs = overall_stats.get("GS", overall_stats.get("GP", 1))
        if gs < 1:
            gs = 1
        for _ in range(int(gs)):
            game_row = {}
            for key, val in overall_stats.items():
                if key in ("ERA", "WHIP", "W%", "OBA"):
                    game_row[key] = val
                elif key in ("GP", "GS"):
                    game_row[key] = 1
                elif key == "IP":
                    # Average IP in OUT space, then re-encode (see the per-
                    # opponent branch above).
                    game_row[key] = outs_to_ip(ip_to_outs(val) / gs)
                else:
                    game_row[key] = round(val / gs, 1)
            games.append(game_row)

    return games


def _parse_stat_row(labels, stats_list):
    """
    Parse a row of stat values into a dict, converting strings to floats.

    Handles ESPN's made-attempted format (e.g., "11-22" for FG made-attempted)
    by extracting the made count (the left-hand number). Empty/DNP-like
    markers ("", "-", "--") become 0.0.
    """
    game_stats = {}
    for label, val in zip(labels, stats_list):
        try:
            if isinstance(val, (int, float)):
                game_stats[label] = float(val)
            elif val in ("", "-", "--"):
                game_stats[label] = 0.0
            elif isinstance(val, str) and "-" in val and not val.startswith("-"):
                # Made-attempted format like "11-22" → use the made count.
                # Only the first hyphen is treated as a separator.
                made_part = val.split("-", 1)[0].strip()
                game_stats[label] = float(made_part) if made_part else 0.0
            else:
                game_stats[label] = float(val)
        except (ValueError, TypeError):
            game_stats[label] = 0.0
    return game_stats


# Maps Odds API prop market key -> list of possible ESPN gamelog stat labels
# Multiple labels listed as fallbacks since ESPN labels vary by sport/season
PROP_STAT_MAP = {
    "player_points": ["PTS"],
    "player_assists": ["AST"],
    "player_rebounds": ["REB"],
    "player_anytime_td": ["TD"],
    "player_rush_yds": ["RUSH YDS", "YDS"],
    "player_pass_yds": ["PASS YDS", "YDS"],
    "batter_hits": ["H"],
    "pitcher_strikeouts": ["K", "SO"],
    "pitcher_outs": ["IP"],  # innings pitched * 3 = outs
    "batter_strikeouts": ["K", "SO"],
    "pitcher_earned_runs": ["ER"],
    # TB/RBI resolve against the warehouse calib-gamelog dicts (get_calib_gamelog
    # emits 'TB'/'RBI' columns). The live ESPN gamelog path serves neither in prod
    # (the SQL cache gamelog_store._BATTER_STATS drops RBI, and ESPN has no TB label
    # at all — get_player_stat_history reads a single label, can't derive TB) → these
    # props are warehouse-served; ESPN is a no_history fall-open for them.
    "batter_total_bases": ["TB"],
    "batter_rbis": ["RBI"],
}


# batter_total_bases / batter_rbis are WAREHOUSE-ONLY on EVERY MLB read path: the
# warehouse (get_player_history / resolve_actual / get_calib_gamelog) is their only
# legitimate source. ESPN must NEVER serve them — TB has no ESPN label at all, and a
# raw ESPN batter row's 'RBI' is UNCALIBRATED (the SQL cache's reduced shape drops it,
# but the slow-path / SQL-off raw row still carries it). Binding it would leak an
# uncalibrated value into a projection, a graded bet, or the calibration corpus. All
# three ESPN read paths guard on this set: get_player_stat_history (history, below),
# recalibration.resolve_one_prop (grading fallback), and
# book_line_calibration.join_book_lines_to_actuals (calibration ESPN fall-open).
WAREHOUSE_ONLY_PROPS = frozenset({"batter_total_bases", "batter_rbis"})


# P4 model-input cutover flag. When ON (and SQL enabled), MLB player histories are
# served from the StatsAPI warehouse facts (mlb_warehouse.get_player_history) with
# ESPN as the fail-open fallback; OFF (default) keeps the pure ESPN path. Env-gated
# so the cutover is flipped deliberately after --player-input parity is clean.
_MLB_WAREHOUSE_HIST_ENV = "ODI_MLB_WAREHOUSE_HIST"


def _mlb_warehouse_hist_enabled():
    return os.environ.get(_MLB_WAREHOUSE_HIST_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


def _mlb_warehouse_history(sport, player_name, prop_key, n, teams=None,
                          confirmed_lineup=None, probable_starters=None):
    """Warehouse-first MLB player history (the P4 model-input flip): return the
    get_player_stat_history contract dict from the StatsAPI facts, or None so the
    caller falls open to the ESPN path. Gated on: the env flag, sport=='baseball',
    SQL enabled, a fact-servable prop, and a GAME-CONTEXT-FIRST name→MLBAM resolution
    (mlb_starters.resolve_mlbam_id): today's posted lineup / announced probables for
    the matchup ``teams`` (authoritative + trade-aware), then the statsapi season
    roster, then a role-verified SFBB fallback — so a namesake like "Max Muncy" /
    "Luis Garcia Jr." binds to the id that actually appears in this game rather than
    the drift-prone SFBB cross-map; a still-ambiguous / unknown name stays None.
    History is fetched BY that MLBAM id (constant across trades). Scoped to the
    CURRENT season to match the ESPN/gamelog_store baseline (current-season-only), so
    the recent-N window isn't padded with prior-season games early in the year. Never
    raises."""
    if sport != "baseball" or not _mlb_warehouse_hist_enabled():
        return None
    try:
        if db_store is None or not db_store.enabled():
            return None
        import mlb_warehouse
        if mlb_warehouse._ACTUAL_STAT_SPEC.get(prop_key) is None:
            return None                       # HR/TB/RBI etc. → ESPN
        import mlb_starters
        resolved = mlb_starters.resolve_mlbam_id(
            player_name, mlb_warehouse._current_season(), prop_key=prop_key,
            teams=teams, confirmed_lineup=confirmed_lineup,
            probable_starters=probable_starters)
        mlb_id = resolved[0] if resolved else None
        if not mlb_id:
            return None                       # unknown / still-ambiguous → ESPN
        return mlb_warehouse.get_player_history(
            mlb_id, prop_key, n=n, season=mlb_warehouse._current_season(),
            player_name=player_name)
    except Exception:
        return None


# P4 team-market cutover flag (independent of the player-history flag). When ON
# (+ SQL enabled), MLB team-market inputs — season block, recent form, recent_games,
# team defense — are served from the StatsAPI warehouse with ESPN as the fail-open
# fallback; OFF (default) keeps the pure ESPN build.
_MLB_WAREHOUSE_TEAM_ENV = "ODI_MLB_WAREHOUSE_TEAM"


def _mlb_warehouse_team_enabled():
    return os.environ.get(_MLB_WAREHOUSE_TEAM_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


def mlb_warehouse_gate_status():
    """The MLB→StatsAPI warehouse gate states (read from os.environ, which the app
    boot-promotes from st.secrets) + whether SQL is on — for an at-a-glance operator
    indicator so a flag flip's effect is VERIFIABLE (predictions don't record which
    source served them). Keys mirror the gate helpers in espn_client / props /
    book_line_calibration."""
    def _on(k):
        return os.environ.get(k, "").strip().lower() in ("1", "true", "on", "yes")
    return {
        "history": _on(_MLB_WAREHOUSE_HIST_ENV),        # ODI_MLB_WAREHOUSE_HIST
        "team": _on(_MLB_WAREHOUSE_TEAM_ENV),           # ODI_MLB_WAREHOUSE_TEAM
        "calib": _on("ODI_MLB_WAREHOUSE_CALIB"),        # book_line_calibration
        "enforce_identity": _on("ODI_MLB_ENFORCE_IDENTITY"),   # props
        "sql": bool(db_store is not None and db_store.enabled()),
    }


def mlb_warehouse_team_stats(sport, team_name, recent_n=10):
    """Warehouse-first team-market stats (the P4 team-market flip): the
    {season, recent, recent_games} dict the team analyzers consume, sourced from the
    StatsAPI warehouse, or None to fall open to the ESPN build. MLB only; env-gated
    (ODI_MLB_WAREHOUSE_TEAM); current-season scoped (matches the ESPN schedule).

    ``recent_games`` carry the queried team's ODDS-FEED name (``team_name``), not the
    canonical mlb_team.name, because compute_recent_form + the analyzers
    (_predict_margin / analyze_moneyline / analyze_totals) identify the team by EXACT
    string match against the odds name — a canonical/odds spelling gap (e.g.
    Athletics) would otherwise silently zero the form. Never raises."""
    if sport != "baseball" or not _mlb_warehouse_team_enabled():
        return None
    try:
        if db_store is None or not db_store.enabled():
            return None
        import mlb_warehouse
        canonical = mlb_warehouse.team_name_canonical(team_name)
        if not canonical:
            return None
        season = mlb_warehouse.get_team_standings(team_name)
        games = mlb_warehouse.get_team_games(
            team_name, season=mlb_warehouse._current_season(), limit=recent_n)
        if not season or not games:
            return None
        # Rekey the queried team's own name to the odds spelling (opponent names are
        # left canonical — the analyzers never match on them). get_team_games returns
        # fresh dicts, so mutation is safe.
        for g in games:
            if g.get("home_team") == canonical:
                g["home_team"] = team_name
            elif g.get("away_team") == canonical:
                g["away_team"] = team_name
        recent = compute_recent_form(games, team_name, n=recent_n)
        return {"season": season, "recent": recent, "recent_games": games}
    except Exception:
        return None


def mlb_warehouse_team_defense(sport):
    """Warehouse team-defense lookup (P4 team-market flip): {team display name: avg
    runs allowed} from /standings, or None to fall open to build_team_defense_lookup.
    MLB only; env-gated; fail-open. Consumers match team names tolerantly
    (_resolve_team_defense), so the canonical mlb_team.name keys are fine."""
    if sport != "baseball" or not _mlb_warehouse_team_enabled():
        return None
    try:
        if db_store is None or not db_store.enabled():
            return None
        import mlb_warehouse
        return mlb_warehouse.get_team_defense() or None
    except Exception:
        return None


def get_player_stat_history(sport, league, player_name, prop_key, n=20,
                            team_ids=None, allow_warehouse=True, teams=None,
                            confirmed_lineup=None, probable_starters=None):
    """
    Look up a player on ESPN and return their recent stat values for a given prop.

    Parameters:
        sport (str): ESPN sport
        league (str): ESPN league
        player_name (str): Player name from the odds API
        prop_key (str): Odds API prop market key (e.g., 'player_points')
        n (int): Number of recent games to return
        team_ids (iterable, optional): ESPN team ids of the matchup, forwarded to
            search_athlete to disambiguate same-name players (see that function).
        confirmed_lineup, probable_starters (optional): MLB-ONLY today's-game context
            (mlb_starters.get_confirmed_lineup / get_probable_starters). When present
            they make the warehouse-history id resolution GAME-CONTEXT-FIRST
            (trade-aware, namesake-safe); ignored for every non-baseball sport (the
            warehouse branch early-returns), so those callers stay byte-identical.

    Returns:
        dict: {
            'player': str,
            'athlete_id': str or None,
            'stat_label': str,
            'values': list of floats (most recent first, up to n),
            'found': bool,
        }
    """
    result = {
        "player": player_name,
        "athlete_id": None,
        "stat_label": "",
        "values": [],
        "opponents": [],
        "home_aways": [],
        "game_dates": [],
        "plate_appearances": [],
        "at_bats": [],
        "team_id": None,
        "found": False,
        # Which data path served this history — stamped onto prediction_log.source so
        # a warehouse-gate flip's effect is auditable per prediction. This ESPN path
        # is "espn"; the warehouse branch below returns get_player_history's dict,
        # which carries "warehouse".
        "source": "espn",
    }

    # P4 model-input cutover (MLB only, env-gated, fail-open): serve the recent-game
    # history straight from the StatsAPI warehouse facts when enabled + resolvable,
    # else fall through to the ESPN path below unchanged. allow_warehouse=False forces
    # the ESPN path (the parity harness passes it so it always diffs the TRUE ESPN
    # side even when the flip flag is on).
    if allow_warehouse:
        wh = _mlb_warehouse_history(sport, player_name, prop_key, n, teams=teams,
                                    confirmed_lineup=confirmed_lineup,
                                    probable_starters=probable_starters)
        if wh is not None:
            return wh

    # batter_total_bases / batter_rbis are WAREHOUSE-ONLY (fact-served above via
    # _mlb_warehouse_history when ODI_MLB_WAREHOUSE_HIST is on). The live ESPN gamelog
    # must NOT serve them: TB is not an ESPN label, and while the SQL-cache reduced
    # shape (gamelog_store._BATTER_STATS) drops RBI, the gamelog_store SLOW path
    # (cache miss/stale) returns the RAW ESPN row which still carries 'RBI' — matching
    # it here would leak an UNCALIBRATED live over-rate the design keeps inert until the
    # warehouse is populated + flipped on (findings verify: nondeterministic across
    # cache TTL / thread race). PROP_STAT_MAP keeps the TB/RBI labels for the backtest/
    # calibration reshape, which reads get_calib_gamelog directly, not this function.
    if sport == "baseball" and prop_key in WAREHOUSE_ONLY_PROPS:
        return result

    # Durable SQL path (Phase C): swap ONLY the two source lookups so the exact
    # extraction/return below is reused (identical result-dict shape). Gated on a
    # sport that has a fact table; other sports (e.g. NHL) keep the direct path
    # unchanged. gamelog_store.get_gamelog handles the MLB pitcher fallback.
    use_sql = (db_store is not None and db_store.enabled()
               and sport in ("baseball", "basketball", "football"))
    if use_sql:
        import gamelog_store
        athlete = gamelog_store.get_athlete_id(sport, league, player_name,
                                               team_ids=team_ids)
        if not athlete or not athlete["id"]:
            return result
        result["athlete_id"] = athlete["id"]
        result["team_id"] = athlete.get("team_id")
        gamelog = gamelog_store.get_gamelog(sport, league, athlete["id"],
                                            player_name=player_name)
        if not gamelog:
            return result
    else:
        athlete = search_athlete(sport, league, player_name, team_ids=team_ids)
        if not athlete or not athlete["id"]:
            return result

        result["athlete_id"] = athlete["id"]
        result["team_id"] = athlete.get("team_id")

        gamelog = get_athlete_gamelog(sport, league, athlete["id"])

        # For MLB pitchers, the gamelog endpoint returns empty. Prefer the TRUE
        # StatsAPI per-game log (real variance + game_date); fall back to the
        # synthesized ESPN splits when the name can't be resolved.
        if not gamelog and sport == "baseball":
            import mlb_starters
            gamelog = mlb_starters._pitcher_gamelog_or_synth(
                league, athlete["id"], player_name, None)

        if not gamelog:
            return result

    # Find the matching stat label
    stat_labels = PROP_STAT_MAP.get(prop_key, [])
    matched_label = None
    for label in stat_labels:
        if any(label in game for game in gamelog):
            matched_label = label
            break

    if not matched_label:
        return result

    result["stat_label"] = matched_label

    # Extract values + per-game opponent / home-away. Pitcher_outs special case (IP * 3).
    values = []
    opponents = []
    home_aways = []
    minutes = []  # raw MIN per game (used by safe-mode to filter DNPs)
    game_dates = []  # ISO timestamp per game (used for current-season counting)
    plate_appearances = []  # MLB batter exposure for PA-level matchup models
    at_bats = []
    # Prefer the player's current team from search. Some ESPN game-log rows
    # omit team metadata, and the game log can also reflect a former team
    # immediately after a trade. Fall back to the newest row that has an ID.
    team_id = result["team_id"]
    for game in gamelog[:n]:
        val = game.get(matched_label, 0.0)
        if prop_key == "pitcher_outs":
            # IP (e.g. 6.1 = 6 innings + 1 out) -> outs; see ip_to_outs.
            val = ip_to_outs(val)
        values.append(val)
        opponents.append(game.get("opponent"))
        home_aways.append(game.get("is_home"))
        minutes.append(game.get("MIN", 0.0) or 0.0)
        game_dates.append(game.get("game_date"))
        ab = game.get("AB")
        pa = game.get("PA")
        if pa is None and ab is not None:
            pa = ((ab or 0.0) + (game.get("BB") or 0.0)
                  + (game.get("HBP") or 0.0) + (game.get("SF") or 0.0)
                  + (game.get("SH") or 0.0))
        plate_appearances.append(pa)
        at_bats.append(ab)
        # Capture the player's team id from the most recent game that has it.
        if team_id is None and game.get("team_id"):
            team_id = game.get("team_id")

    result["values"] = values
    result["opponents"] = opponents
    result["home_aways"] = home_aways
    result["minutes"] = minutes
    result["game_dates"] = game_dates
    result["plate_appearances"] = plate_appearances
    result["at_bats"] = at_bats
    result["team_id"] = team_id
    result["found"] = len(values) > 0
    return result
