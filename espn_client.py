"""
Client for the unofficial ESPN API.
Fetches team records, recent results, and scoring averages.
No authentication required.
"""

import requests
from datetime import datetime, timedelta


SITE_API = "https://site.api.espn.com/apis/site/v2/sports"
CORE_API = "https://sports.core.api.espn.com/v2/sports"

# Common timeout for all ESPN requests
TIMEOUT = 15


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

    return teams


def get_team_schedule(sport, league, team_id, season_year=None):
    """
    Fetch a team's schedule/results for a season.

    Returns:
        list: List of game result dicts (most recent first)
    """
    url = f"{SITE_API}/{sport}/{league}/teams/{team_id}/schedule"
    params = {}
    if season_year:
        params["season"] = season_year

    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    games = []
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

        games.append({
            "date": event.get("date", ""),
            "home_team": home_comp.get("team", {}).get("displayName", ""),
            "away_team": away_comp.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "total_score": home_score + away_score,
        })

    # Sort by date descending (most recent first)
    games.sort(key=lambda g: g["date"], reverse=True)
    return games


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


def search_athlete(sport, league, name):
    """
    Search for an athlete by name.
    Tries the site API first (works for NBA/NFL), falls back to web search API (works for MLB).

    Parameters:
        sport (str): ESPN sport (e.g., 'basketball')
        league (str): ESPN league (e.g., 'nba')
        name (str): Player name to search for

    Returns:
        dict or None: {'id': str, 'name': str} or None if not found
    """
    # Try site API first (works for NBA, NFL)
    try:
        url = f"{SITE_API}/{sport}/{league}/athletes"
        params = {"search": name, "limit": 5}
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            athletes = data.get("athletes", [])
            if athletes and isinstance(athletes[0], dict):
                athlete = athletes[0]
                return {"id": str(athlete.get("id", "")), "name": athlete.get("displayName", athlete.get("fullName", name))}
            items = data.get("items", [])
            if items:
                athlete = items[0]
                return {"id": str(athlete.get("id", "")), "name": athlete.get("displayName", athlete.get("fullName", name))}
    except Exception:
        pass

    # Fallback: web search API (works for MLB and all sports)
    try:
        url = "https://site.web.api.espn.com/apis/common/v3/search"
        params = {"query": name, "limit": 5, "type": "player"}
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            for item in items:
                if item.get("sport") == sport and item.get("league") == league:
                    return {"id": str(item.get("id", "")), "name": item.get("displayName", name)}
            # If no sport/league match, take first player result
            if items:
                return {"id": str(items[0].get("id", "")), "name": items[0].get("displayName", name)}
    except Exception:
        pass

    return None


def get_athlete_gamelog(sport, league, athlete_id):
    """
    Fetch game log (game-by-game stats) for an athlete.
    Handles two ESPN response formats:
      - NBA/NFL: seasonTypes -> categories -> events with stats
      - MLB batters: top-level labels + events dict keyed by event ID

    Parameters:
        sport (str): ESPN sport
        league (str): ESPN league
        athlete_id (str): ESPN athlete ID

    Returns:
        list: List of dicts with stat values per game, e.g.:
              [{'PTS': 25, 'AST': 5, 'REB': 10, ...}, ...]
              Returns empty list on failure.
    """
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/{sport}/{league}/athletes/{athlete_id}/gamelog"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    games = []

    # Top-level labels (used by MLB and sometimes shared across categories)
    top_labels = data.get("labels", [])

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
                games.append(game_stats)

    if games:
        return games

    # Format 2: MLB top-level events dict (keyed by event ID)
    events = data.get("events", {})
    if top_labels and isinstance(events, dict):
        for event_id, event in events.items():
            stats_list = event.get("stats", [])
            if len(stats_list) == len(top_labels):
                game_stats = _parse_stat_row(top_labels, stats_list)
                games.append(game_stats)

    return games


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
                else:
                    game_row[key] = round(val / gs, 1)
            games.append(game_row)

    return games


def _parse_stat_row(labels, stats_list):
    """Parse a row of stat values into a dict, converting strings to floats."""
    game_stats = {}
    for label, val in zip(labels, stats_list):
        try:
            if isinstance(val, (int, float)):
                game_stats[label] = float(val)
            elif val in ("", "-", "--"):
                game_stats[label] = 0.0
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
}


def get_player_stat_history(sport, league, player_name, prop_key, n=20):
    """
    Look up a player on ESPN and return their recent stat values for a given prop.

    Parameters:
        sport (str): ESPN sport
        league (str): ESPN league
        player_name (str): Player name from the odds API
        prop_key (str): Odds API prop market key (e.g., 'player_points')
        n (int): Number of recent games to return

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
        "found": False,
    }

    athlete = search_athlete(sport, league, player_name)
    if not athlete or not athlete["id"]:
        return result

    result["athlete_id"] = athlete["id"]

    gamelog = get_athlete_gamelog(sport, league, athlete["id"])

    # For MLB pitchers, the gamelog endpoint returns empty.
    # Fall back to the splits-based pitcher stats.
    is_pitcher_prop = prop_key in ("pitcher_strikeouts", "pitcher_outs")
    if not gamelog and sport == "baseball":
        gamelog = get_pitcher_stats(league, athlete["id"])
        # Also try for batter props if gamelog was empty (unlikely but safe)

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

    # Extract values, handle pitcher_outs special case (IP * 3)
    values = []
    for game in gamelog[:n]:
        val = game.get(matched_label, 0.0)
        if prop_key == "pitcher_outs":
            # Convert innings pitched to outs (IP like 6.1 means 6 innings + 1 out = 19 outs)
            # ESPN stores IP as decimal where .1 = 1/3, .2 = 2/3
            whole = int(val)
            frac = round((val - whole) * 10)
            val = whole * 3 + frac
        values.append(val)

    result["values"] = values
    result["found"] = len(values) > 0
    return result
