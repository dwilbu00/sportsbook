"""
Sportsbook Value Finder
=======================
Compares sportsbook odds (The Odds API) against historical stats (ESPN)
to identify value betting candidates where books underestimate probability.

Usage:
    python main.py                          # Full interactive mode
    python main.py --sport nba              # Skip sport selection
    python main.py --sport mlb --markets h2h,spreads
    python main.py --sport nfl --all        # Fetch all games (no selection)
    python main.py --sport nba --props      # Include player props analysis

Credit costs (The Odds API):
    - Listing events:       FREE (0 credits)
    - Fetching odds:        1 credit per market per region
    - Player props:         1 credit per prop market per game
"""

import argparse
import json
import os
import sys
from datetime import datetime

from odds_client import (
    get_upcoming_events,
    get_upcoming_odds,
    get_event_odds,
    parse_game_odds,
    parse_player_props,
    consensus_odds,
    PLAYER_PROPS_BY_SPORT,
    PROP_LABELS,
)
from espn_client import (
    get_all_teams,
    get_team_schedule,
    compute_recent_form,
    find_team,
    annotate_opponent_strength,
    get_player_stat_history,
    mlb_warehouse_team_stats,
)
from analysis import (
    analyze_moneyline_value,
    analyze_spreads_value,
    analyze_totals_value,
    analyze_player_props_value,
    format_moneyline_report,
    format_spreads_report,
    format_totals_report,
    format_props_report,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

SPORT_CHOICES = {
    "1": ("basketball_nba", "NBA"),
    "2": ("baseball_mlb", "MLB"),
    "3": ("americanfootball_nfl", "NFL"),
    "nba": ("basketball_nba", "NBA"),
    "mlb": ("baseball_mlb", "MLB"),
    "nfl": ("americanfootball_nfl", "NFL"),
}

MARKET_LABELS = {
    "h2h": "Moneyline",
    "spreads": "Spreads",
    "totals": "Over/Under",
}


def load_config():
    """Load configuration from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def select_sport():
    """Interactive sport selection menu."""
    print("\n  Select a sport:")
    print("    1) NBA")
    print("    2) MLB")
    print("    3) NFL")
    print()
    choice = input("  Enter choice (1-3): ").strip().lower()

    if choice in SPORT_CHOICES:
        return SPORT_CHOICES[choice]

    print(f"  Invalid choice: {choice}")
    sys.exit(1)


def select_markets(sport_key):
    """Interactive market selection to control credit usage."""
    print("\n  Select markets to analyze (each costs 1 credit per game):")
    print("    1) Moneyline only         (1 credit/game)")
    print("    2) Spreads only           (1 credit/game)")
    print("    3) Over/Under only        (1 credit/game)")
    print("    4) Moneyline + Spreads    (2 credits/game)")
    print("    5) Moneyline + Over/Under (2 credits/game)")
    print("    6) Spreads + Over/Under   (2 credits/game)")
    print("    7) All three              (3 credits/game)")
    print()

    choice = input("  Enter choice (1-7): ").strip()

    market_map = {
        "1": "h2h",
        "2": "spreads",
        "3": "totals",
        "4": "h2h,spreads",
        "5": "h2h,totals",
        "6": "spreads,totals",
        "7": "h2h,spreads,totals",
    }

    if choice in market_map:
        markets = market_map[choice]
        labels = ", ".join(MARKET_LABELS[m] for m in markets.split(","))
        print(f"  Selected: {labels}")
        return markets

    print(f"  Invalid choice: {choice}")
    sys.exit(1)


def select_props(sport_key):
    """Ask whether to include player props and which ones."""
    available = PLAYER_PROPS_BY_SPORT.get(sport_key, [])
    if not available:
        return []

    print(f"\n  Player props available for this sport:")
    for i, prop in enumerate(available, 1):
        print(f"    {i}) {PROP_LABELS.get(prop, prop)}")
    print(f"    a) All of the above")
    print(f"    n) None (skip props)")

    prop_count = len(available)
    choice = input(f"\n  Select props (e.g., '1,3', 'a', or 'n'): ").strip().lower()

    if choice == "n" or choice == "":
        return []
    if choice == "a":
        print(f"  Selected all {prop_count} prop markets (+{prop_count} credits/game)")
        return available

    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = [available[i] for i in indices if 0 <= i < len(available)]
        if selected:
            labels = ", ".join(PROP_LABELS.get(p, p) for p in selected)
            print(f"  Selected: {labels} (+{len(selected)} credits/game)")
        return selected
    except (ValueError, IndexError):
        print("  Invalid input, skipping props.")
        return []


def select_games(events, api_key, sport_key, markets, bookmakers, prop_markets=None):
    """
    Show upcoming games and let the user pick which to analyze.
    Uses the FREE /events endpoint first, then fetches odds only for selected games.
    """
    if not events:
        print("  No upcoming events found.")
        return [], []

    print(f"\n  Upcoming games ({len(events)} found):")
    print(f"  {'#':>3}  {'Time':20s}  {'Matchup'}")
    print(f"  {'-'*3}  {'-'*20}  {'-'*40}")

    for i, event in enumerate(events, 1):
        commence = event.get("commence_time", "")
        try:
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            time_str = dt.strftime("%b %d  %I:%M %p")
        except (ValueError, AttributeError):
            time_str = commence[:19] if commence else "TBD"

        home = event.get("home_team", "?")
        away = event.get("away_team", "?")
        print(f"  {i:3d}  {time_str:20s}  {away} @ {home}")

    market_count = len(markets.split(","))
    prop_count = len(prop_markets) if prop_markets else 0
    total_per_game = market_count + prop_count
    print(f"\n  Credit cost: {total_per_game} credit(s) per game ({market_count} markets + {prop_count} props)")
    print(f"  Options: Enter game numbers (e.g., '1,3,5'), 'all', or 'q' to quit")
    choice = input("\n  Select games: ").strip().lower()

    if choice == "q":
        sys.exit(0)

    if choice == "all":
        selected_indices = list(range(len(events)))
    else:
        try:
            selected_indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected_indices = [i for i in selected_indices if 0 <= i < len(events)]
        except ValueError:
            print("  Invalid input.")
            sys.exit(1)

    if not selected_indices:
        print("  No games selected.")
        return [], []

    total_cost = len(selected_indices) * total_per_game
    print(f"\n  Fetching odds for {len(selected_indices)} game(s)... (estimated cost: {total_cost} credits)")

    # Fetch game odds (h2h/spreads/totals)
    games_with_odds = []
    for idx in selected_indices:
        event = events[idx]
        event_id = event["id"]
        try:
            game_data = get_event_odds(api_key, sport_key, event_id,
                                       markets=markets, bookmakers=bookmakers)
            games_with_odds.append(game_data)
        except Exception as e:
            print(f"    WARNING: Failed to fetch odds for event {event_id}: {e}")

    # Fetch player prop odds (separate calls since props use event odds endpoint)
    props_with_odds = []
    if prop_markets:
        prop_markets_str = ",".join(prop_markets)
        for idx in selected_indices:
            event = events[idx]
            event_id = event["id"]
            try:
                prop_data = get_event_odds(api_key, sport_key, event_id,
                                           markets=prop_markets_str, bookmakers=bookmakers)
                props_with_odds.append(prop_data)
            except Exception as e:
                print(f"    WARNING: Failed to fetch props for event {event_id}: {e}")

    return games_with_odds, props_with_odds


def run_analysis(sport_key, config, markets, prop_markets=None, fetch_all=False):
    """Main analysis pipeline."""
    sport_config = config["sports"][sport_key]
    api_key = config["odds_api_key"]
    threshold = config.get("value_threshold_pct", 5.0)
    recent_n = config.get("recent_games_window", 10)
    bookmakers = config.get("bookmakers", []) or None

    espn_sport = sport_config["espn_sport"]
    espn_league = sport_config["espn_league"]
    odds_sport = sport_config["odds_api_sport"]
    market_list = markets.split(",")

    if fetch_all:
        print(f"\n  Fetching all odds for {odds_sport} ({markets})...")
        try:
            raw_games = get_upcoming_odds(api_key, odds_sport, markets=markets,
                                          bookmakers=bookmakers)
        except Exception as e:
            print(f"  ERROR fetching odds: {e}")
            sys.exit(1)
        raw_props = []
        # For --all with props, fetch props for each game
        if prop_markets:
            prop_markets_str = ",".join(prop_markets)
            events = get_upcoming_events(api_key, odds_sport)
            for event in events:
                try:
                    prop_data = get_event_odds(api_key, odds_sport, event["id"],
                                               markets=prop_markets_str, bookmakers=bookmakers)
                    raw_props.append(prop_data)
                except Exception:
                    pass
    else:
        print(f"\n  Listing upcoming {odds_sport} events (FREE)...")
        try:
            events = get_upcoming_events(api_key, odds_sport)
        except Exception as e:
            print(f"  ERROR fetching events: {e}")
            sys.exit(1)

        raw_games, raw_props = select_games(events, api_key, odds_sport, markets,
                                            bookmakers, prop_markets)

    if not raw_games and not raw_props:
        print("  No games to analyze.")
        return

    # Team data. MLB (P6): team resolution is warehouse-only (no ESPN get_all_teams);
    # NBA/NFL/NHL still fetch from ESPN.
    if sport_key == "baseball_mlb":
        espn_teams = {}
        print("\n  MLB team resolution: StatsAPI warehouse (no ESPN).")
    else:
        print(f"\n  Fetching {espn_league.upper()} team data from ESPN...")
        try:
            espn_teams = get_all_teams(espn_sport, espn_league)
        except Exception as e:
            print(f"  ERROR fetching ESPN data: {e}")
            sys.exit(1)
        print(f"  Loaded {len(espn_teams)} teams.")

    # Analyze team-level markets (h2h, spreads, totals)
    all_ml_candidates = []
    all_spread_candidates = []
    all_totals_candidates = []
    # Collect per-team avg points allowed for the player-prop opponent-defense lookup.
    team_defense = {}

    if raw_games:
        print(f"\n  Analyzing {len(raw_games)} game(s) for team markets...")
        for i, raw_game in enumerate(raw_games, 1):
            game_odds = parse_game_odds(raw_game)
            home = game_odds["home_team"]
            away = game_odds["away_team"]

            print(f"  [{i}/{len(raw_games)}] {away} @ {home}")

            home_espn = _resolve_team_dim(sport_key, home, espn_teams)
            away_espn = _resolve_team_dim(sport_key, away, espn_teams)

            if not home_espn:
                print(f"    WARNING: Could not resolve '{home}'. Skipping.")
                continue
            if not away_espn:
                print(f"    WARNING: Could not resolve '{away}'. Skipping.")
                continue

            print(f"    {home_espn['display_name']} ({home_espn['record']}) vs "
                  f"{away_espn['display_name']} ({away_espn['record']})")

            # P4 team-market flip: prefer the StatsAPI warehouse (MLB-only,
            # env-gated); fall open to the ESPN build. Require both sides.
            wh_home = mlb_warehouse_team_stats(espn_sport, home, recent_n)
            wh_away = mlb_warehouse_team_stats(espn_sport, away, recent_n)
            if wh_home and wh_away:
                home_stats, away_stats = wh_home, wh_away
            else:
                home_stats = build_team_stats(home_espn, espn_sport, espn_league,
                                              recent_n, espn_teams, sport_key)
                away_stats = build_team_stats(away_espn, espn_sport, espn_league,
                                              recent_n, espn_teams, sport_key)

            # Cache per-team avg_allowed for opponent-defense weighting on props.
            team_defense[home_espn["display_name"]] = home_stats["recent"]["avg_allowed"]
            team_defense[away_espn["display_name"]] = away_stats["recent"]["avg_allowed"]

            if "h2h" in market_list:
                ml_candidates = analyze_moneyline_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key)
                all_ml_candidates.extend(ml_candidates)

            if "spreads" in market_list:
                spread_candidates = analyze_spreads_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key)
                all_spread_candidates.extend(spread_candidates)

            if "totals" in market_list:
                totals_candidates = analyze_totals_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key)
                all_totals_candidates.extend(totals_candidates)

    # Analyze player props
    all_props_candidates = []

    if raw_props:
        print(f"\n  Analyzing player props for {len(raw_props)} game(s)...")
        print(f"  (Fetching ESPN player histories — this may take a moment)\n")

        for i, raw_prop in enumerate(raw_props, 1):
            prop_data = parse_player_props(raw_prop)
            print(f"  [{i}/{len(raw_props)}] {prop_data['away_team']} @ {prop_data['home_team']}")

            # ESPN team ids disambiguate same-name players for the ESPN history path
            # (search_athlete). MLB (P6) serves history from the warehouse by
            # globally-unique NAME, not team ids, so pass no hint for baseball.
            if sport_key == "baseball_mlb":
                event_team_ids = []
            else:
                event_teams = [find_team(espn_teams, tn)
                               for tn in (prop_data.get("home_team"),
                                          prop_data.get("away_team")) if tn]
                event_team_ids = [str(t["id"]) for t in event_teams
                                  if t and t.get("id")]

            # Collect all unique players and their prop keys
            player_histories = {}
            for prop_key, players in prop_data.get("props", {}).items():
                for player_name in players:
                    if player_name not in player_histories:
                        player_histories[player_name] = {}
                    if prop_key not in player_histories[player_name]:
                        print(f"    Looking up {player_name} ({PROP_LABELS.get(prop_key, prop_key)})...")
                        history = get_player_stat_history(
                            espn_sport, espn_league, player_name, prop_key,
                            n=recent_n, team_ids=event_team_ids,
                            # MLB: matchup teams narrow a namesake to its MLBAM id.
                            teams=[prop_data.get("home_team"),
                                   prop_data.get("away_team")],
                        )
                        player_histories[player_name][prop_key] = history
                        if history["found"]:
                            avg = sum(history["values"]) / len(history["values"])
                            print(f"      Found {len(history['values'])} games, avg: {avg:.1f}")
                        else:
                            print(f"      No ESPN data found")

            props_candidates = analyze_player_props_value(prop_data, player_histories, threshold,
                                                          sport_key=sport_key,
                                                          team_defense=team_defense,
                                                          espn_teams=espn_teams)
            all_props_candidates.extend(props_candidates)

    # Print reports
    print("\n")
    if all_ml_candidates:
        print(format_moneyline_report(all_ml_candidates))
    if all_spread_candidates:
        print(format_spreads_report(all_spread_candidates))
    if all_totals_candidates:
        print(format_totals_report(all_totals_candidates))
    if all_props_candidates:
        print(format_props_report(all_props_candidates))
    print()


def _resolve_team_dim(sport_key, name, espn_teams):
    """Team dict for a matchup team. MLB (P6 teardown): resolve off the StatsAPI
    warehouse (warehouse_find_team) so team markets no longer need the ESPN teams
    dict; ESPN find_team is the transition fallback (no MLB game silently drops)
    and the sole path for NBA/NFL/NHL."""
    if sport_key == "baseball_mlb":
        try:
            import mlb_warehouse
            wh = mlb_warehouse.warehouse_find_team(name)
            if wh:
                return wh
        except Exception:
            pass
    return find_team(espn_teams, name)


def build_team_stats(team_info, espn_sport, espn_league, recent_n, espn_teams=None,
                     sport_key=None):
    """Build a stats dict for a team by fetching their schedule and computing form.
    MLB (P6): the schedule comes from the StatsAPI warehouse (get_team_games), no
    ESPN. This is only the fallback when mlb_warehouse_team_stats misses."""
    team_id = team_info["id"]
    display_name = team_info["display_name"]

    try:
        if sport_key == "baseball_mlb":
            import mlb_warehouse
            games = mlb_warehouse.get_team_games(display_name) or []
        else:
            games = get_team_schedule(espn_sport, espn_league, team_id)
    except Exception as e:
        print(f"    WARNING: Could not fetch schedule for {display_name}: {e}")
        games = []

    recent = compute_recent_form(games, display_name, n=recent_n)
    recent_games = games[:recent_n]

    # Augment recent games with opponent win% for opponent-strength weighting.
    if espn_teams:
        annotate_opponent_strength(recent_games, display_name, espn_teams)

    return {
        "season": {
            "record": team_info["record"],
            "wins": team_info["wins"],
            "losses": team_info["losses"],
            "win_pct": team_info["win_pct"],
        },
        "recent": recent,
        "recent_games": recent_games,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sportsbook Value Finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Credit costs (The Odds API):
  Listing events:       FREE
  h2h only:             1 credit/game
  h2h,spreads:          2 credits/game
  h2h,spreads,totals:   3 credits/game
  Player props:         +1 credit per prop market per game

Examples:
  python main.py                              # Full interactive mode
  python main.py --sport nba                  # NBA, interactive game/market selection
  python main.py --sport nba --markets h2h    # NBA moneyline, pick games
  python main.py --sport nba --props          # Include player props
  python main.py --sport mlb --all            # MLB, all games (skip selection)
        """,
    )
    parser.add_argument("--sport", type=str, choices=["nba", "mlb", "nfl"],
                        help="Sport to analyze")
    parser.add_argument("--markets", type=str, default=None,
                        help="Comma-separated markets: h2h,spreads,totals (default: interactive)")
    parser.add_argument("--props", action="store_true",
                        help="Include player props analysis (additional credits per game)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Value edge threshold in percentage points (default: from config)")
    parser.add_argument("--recent", type=int, default=None,
                        help="Number of recent games for form analysis (default: from config)")
    parser.add_argument("--all", action="store_true", dest="fetch_all",
                        help="Fetch all games at once (skip game selection)")
    args = parser.parse_args()

    config = load_config()

    if config["odds_api_key"] == "YOUR_API_KEY_HERE":
        print("\n  ERROR: Please set your Odds API key in config.json")
        print(f"  File: {CONFIG_PATH}")
        sys.exit(1)

    if args.threshold is not None:
        config["value_threshold_pct"] = args.threshold
    if args.recent is not None:
        config["recent_games_window"] = args.recent

    # Sport selection
    if args.sport:
        sport_key, sport_label = SPORT_CHOICES[args.sport]
    else:
        sport_key, sport_label = select_sport()

    # Market selection
    if args.markets:
        markets = args.markets
    elif args.fetch_all:
        markets = "h2h,spreads,totals"
    else:
        markets = select_markets(sport_key)

    # Player props selection
    if args.props:
        prop_markets = PLAYER_PROPS_BY_SPORT.get(sport_key, [])
        if prop_markets:
            labels = ", ".join(PROP_LABELS.get(p, p) for p in prop_markets)
            print(f"  Props: {labels} (+{len(prop_markets)} credits/game)")
    elif not args.fetch_all and not args.markets:
        prop_markets = select_props(sport_key)
    else:
        prop_markets = []

    market_labels = ", ".join(MARKET_LABELS.get(m, m) for m in markets.split(","))
    prop_label_str = ""
    if prop_markets:
        prop_label_str = f"\n  Props: {', '.join(PROP_LABELS.get(p, p) for p in prop_markets)}"

    print(f"\n{'=' * 80}")
    print(f"  SPORTSBOOK VALUE FINDER — {sport_label}")
    print(f"  Markets: {market_labels}{prop_label_str}")
    print(f"  Threshold: {config['value_threshold_pct']}% edge  |  Recent window: {config['recent_games_window']} games")
    print(f"{'=' * 80}")

    run_analysis(sport_key, config, markets, prop_markets=prop_markets, fetch_all=args.fetch_all)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
