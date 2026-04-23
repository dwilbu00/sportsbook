"""
Sportsbook Value Finder — Streamlit UI
=======================================
Launch with:  streamlit run app.py
"""

import streamlit as st
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Add script dir to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from odds_client import (
    get_upcoming_events,
    get_event_odds,
    parse_game_odds,
    parse_player_props,
    get_remaining_credits,
    is_event_cached,
    PLAYER_PROPS_BY_SPORT,
    PROP_LABELS,
)
from espn_client import (
    get_all_teams,
    get_team_schedule,
    compute_recent_form,
    find_team,
    get_player_stat_history,
)
from analysis import (
    analyze_moneyline_value,
    analyze_spreads_value,
    analyze_totals_value,
    analyze_player_props_value,
    generate_parlays,
)

SPORTS = {
    "NBA": {
        "key": "basketball_nba",
        "espn_sport": "basketball",
        "espn_league": "nba",
    },
    "MLB": {
        "key": "baseball_mlb",
        "espn_sport": "baseball",
        "espn_league": "mlb",
    },
    "NFL": {
        "key": "americanfootball_nfl",
        "espn_sport": "football",
        "espn_league": "nfl",
    },
}

MARKET_OPTIONS = {
    "Moneyline": "h2h",
    "Spreads": "spreads",
    "Over/Under": "totals",
}


def get_api_key():
    """Get API key from Streamlit secrets."""
    try:
        return st.secrets["ODDS_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_data(ttl=3600)
def fetch_events(api_key, sport_key):
    return get_upcoming_events(api_key, sport_key)


@st.cache_data(ttl=3600)
def fetch_espn_teams(espn_sport, espn_league):
    return get_all_teams(espn_sport, espn_league)


@st.cache_data(ttl=3600)
def fetch_event_odds_cached(api_key, sport_key, event_id, markets, bookmakers_str):
    bookmakers = bookmakers_str.split(",") if bookmakers_str else None
    return get_event_odds(api_key, sport_key, event_id, markets=markets, bookmakers=bookmakers)


@st.cache_data(ttl=3600)
def fetch_team_schedule_cached(espn_sport, espn_league, team_id):
    return get_team_schedule(espn_sport, espn_league, team_id)


@st.cache_data(ttl=3600)
def fetch_player_history_cached(espn_sport, espn_league, player_name, prop_key, n):
    return get_player_stat_history(espn_sport, espn_league, player_name, prop_key, n)


def build_team_stats(team_info, espn_sport, espn_league, recent_n):
    team_id = team_info["id"]
    display_name = team_info["display_name"]
    try:
        games = fetch_team_schedule_cached(espn_sport, espn_league, team_id)
    except Exception:
        games = []
    recent = compute_recent_form(games, display_name, n=recent_n)
    return {
        "season": {
            "record": team_info["record"],
            "wins": team_info["wins"],
            "losses": team_info["losses"],
            "win_pct": team_info["win_pct"],
        },
        "recent": recent,
        "recent_games": games[:recent_n],
    }


def format_time(commence_time):
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        eastern = dt.astimezone(ZoneInfo("America/New_York"))
        return eastern.strftime("%b %d  %I:%M %p ET")
    except (ValueError, AttributeError):
        return commence_time[:19] if commence_time else "TBD"


def render_value_badge(edge_pct):
    if edge_pct >= 10:
        st.markdown(f":green[**+{edge_pct}%** 🔥]")
    elif edge_pct >= 5:
        st.markdown(f":green[**+{edge_pct}%** ✅]")
    elif edge_pct > 0:
        st.markdown(f":orange[+{edge_pct}%]")
    else:
        st.markdown(f":red[{edge_pct}%]")


# ──────────────────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Sportsbook Value Finder", page_icon="🎯", layout="wide")

# ──────────────────────────────────────────────────────────
#  API Key from secrets
# ──────────────────────────────────────────────────────────
api_key = get_api_key()
if not api_key:
    st.title("🎯 Sportsbook Value Finder")
    st.error("API key not configured. Add `ODDS_API_KEY` to your Streamlit secrets.")
    st.stop()

st.title("🎯 Sportsbook Value Finder")
st.caption("Compare book odds vs historical stats to find value")

# ──────────────────────────────────────────────────────────
#  Sidebar — Configuration
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Reset game selection when sport changes
    def on_sport_change():
        st.session_state["selected_games"] = []
        st.session_state["markets"] = []
        st.session_state["props"] = []
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("parlay_results", None)
        st.session_state.pop("parlay_mode", None)
        st.session_state.pop("result_filter", None)

    sport_label = st.selectbox("Sport", list(SPORTS.keys()), key="sport", on_change=on_sport_change)
    sport = SPORTS[sport_label]

    st.subheader("Markets")
    selected_markets = st.multiselect(
        "Select markets",
        list(MARKET_OPTIONS.keys()),
        default=["Moneyline"],
        key="markets",
    )
    market_keys = [MARKET_OPTIONS[m] for m in selected_markets]
    markets_str = ",".join(market_keys)

    st.subheader("Player Props")
    available_props = PLAYER_PROPS_BY_SPORT.get(sport["key"], [])
    prop_labels = {PROP_LABELS[p]: p for p in available_props}
    selected_prop_labels = st.multiselect(
        "Select props",
        list(prop_labels.keys()),
        key="props",
    )
    selected_props = [prop_labels[l] for l in selected_prop_labels]

    st.subheader("Analysis Settings")
    threshold = st.slider("Value threshold (%)", 1.0, 20.0, 5.0, 0.5, key="threshold")
    recent_n = st.slider("Recent games window", 5, 30, 10, key="recent_n")

    total_per_game = len(market_keys) + len(selected_props)
    remaining = get_remaining_credits()
    credit_info = f"**{total_per_game} credit(s)** per game (max)"
    if remaining is not None:
        credit_info += f"  |  **{remaining}** credits remaining"
    st.info(credit_info)

    st.divider()
    st.caption(f"API Key: ...{api_key[-6:]}")

# ──────────────────────────────────────────────────────────
#  Main — Game Selection
# ──────────────────────────────────────────────────────────
st.subheader(f"📅 Upcoming {sport_label} Games")

with st.spinner("Loading events (free)..."):
    try:
        events = fetch_events(api_key, sport["key"])
    except Exception as e:
        st.error(f"Failed to fetch events: {e}")
        st.stop()

if not events:
    st.warning("No upcoming games found.")
    st.stop()

game_options = {}
for event in events:
    home = event.get("home_team", "?")
    away = event.get("away_team", "?")
    time_str = format_time(event.get("commence_time", ""))
    label = f"{time_str}  —  {away} @ {home}"
    game_options[label] = event

selected_game_labels = st.multiselect(
    "Select games to analyze",
    list(game_options.keys()),
    help=f"Each game costs {total_per_game} credit(s)",
    key="selected_games",
)

if not selected_game_labels:
    st.info("👆 Select one or more games above to begin analysis")
    st.stop()

# Calculate actual credit cost (accounting for cached data)
bookmakers_param = None
actual_cost = 0
for gl in selected_game_labels:
    ev = game_options[gl]
    eid = ev["id"]
    if markets_str and not is_event_cached(sport["key"], eid, markets=markets_str, bookmakers=bookmakers_param):
        actual_cost += len(market_keys)
    if selected_props:
        prop_markets_str = ",".join(selected_props)
        if not is_event_cached(sport["key"], eid, markets=prop_markets_str, bookmakers=bookmakers_param):
            actual_cost += len(selected_props)

total_max_cost = len(selected_game_labels) * total_per_game
remaining = get_remaining_credits()
credit_parts = []
if actual_cost < total_max_cost:
    credit_parts.append(f"Estimated cost: **{actual_cost}** (cached saves {total_max_cost - actual_cost})")
else:
    credit_parts.append(f"Estimated cost: **{actual_cost}**")
if remaining is not None:
    credit_parts.append(f"Credits remaining: **{remaining}**")
st.caption("  |  ".join(credit_parts))

# ──────────────────────────────────────────────────────────
#  Run Analysis
# ──────────────────────────────────────────────────────────
col_analyze, col_parlay, col_safe = st.columns(3)
with col_analyze:
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col_parlay:
    parlay_clicked = st.button("🎰 Value Parlays", use_container_width=True)
with col_safe:
    safe_clicked = st.button("🛡️ Safe Parlays", use_container_width=True)

# ── Generate Parlays ──
if (parlay_clicked or safe_clicked) and "analysis_results" in st.session_state:
    ar = st.session_state["analysis_results"]
    # Only include bet types currently selected
    p_ml = ar["all_ml"] if "h2h" in market_keys else []
    p_spreads = ar["all_spreads"] if "spreads" in market_keys else []
    p_totals = ar["all_totals"] if "totals" in market_keys else []
    p_props = ar["all_props"] if selected_props else []
    mode = "safe" if safe_clicked else "value"
    parlays = generate_parlays(p_ml, p_spreads, p_totals, p_props, ar["sport_key"], mode)
    st.session_state["parlay_results"] = parlays
    st.session_state["parlay_mode"] = mode

# ── Display Parlay Results (at top) ──
if "parlay_results" in st.session_state:
    parlays = st.session_state["parlay_results"]
    mode = st.session_state.get("parlay_mode", "value")
    mode_label = "🛡️ Safe" if mode == "safe" else "🎰 Value"
    st.divider()
    st.subheader(f"{mode_label} Parlays")
    if mode == "safe":
        st.caption("Prioritizing highest probability of hitting with positive edge")
    else:
        st.caption("Prioritizing best edge value with sport-specific correlation analysis")

    if parlays:
        for size in [3, 4, 5]:
            if size not in parlays:
                continue
            p = parlays[size]
            with st.expander(
                f"{'⭐' * size}  Best {size}-Leg Parlay  —  "
                + (f"Hit Prob: {p['combined_hist_prob']}%" if mode == "safe"
                   else f"Combined Edge: +{p['parlay_edge_pct']}%"),
                expanded=(size == 3),
            ):
                for i, leg in enumerate(p["legs"], 1):
                    prob_pct = min(round(leg["hist_prob"] * 100, 2), 99.99)
                    if mode == "safe":
                        st.markdown(
                            f"**Leg {i}:** {leg['label']}  —  "
                            f"Prob: {prob_pct}%  |  Edge: +{leg['edge_pct']}%"
                            + (f"  ({leg['odds_price']:+d})" if leg.get('odds_price') else "")
                        )
                    else:
                        edge_icon = "🔥" if leg["edge_pct"] >= 10 else "✅" if leg["edge_pct"] >= 5 else "📊"
                        st.markdown(
                            f"**Leg {i}:** {edge_icon} {leg['label']}  —  "
                            f"Edge: +{leg['edge_pct']}%"
                            + (f"  ({leg['odds_price']:+d})" if leg.get('odds_price') else "")
                        )

                st.divider()
                cols = st.columns(4)
                cols[0].metric("Legs", size)
                cols[1].metric("Hit Probability", f"{p['combined_hist_prob']}%")
                cols[2].metric("Sum of Edges", f"+{p['combined_edge']}%")
                cols[3].metric("Parlay Edge", f"+{p['parlay_edge_pct']}%")
    else:
        st.info("Not enough positive-edge bets to build parlays. Try selecting more games or markets.")

if analyze_clicked:
    progress = st.progress(0, text="Starting analysis...")

    # Fetch ESPN teams
    progress.progress(5, text="Loading ESPN team data...")
    try:
        espn_teams = fetch_espn_teams(sport["espn_sport"], sport["espn_league"])
    except Exception as e:
        st.error(f"Failed to fetch ESPN data: {e}")
        st.stop()

    selected_events = [game_options[l] for l in selected_game_labels]
    total_games = len(selected_events)

    all_ml = []
    all_spreads = []
    all_totals = []
    all_props = []
    warnings = []

    bookmakers_str = ""

    progress.progress(10, text="Getting game odds...")

    # ── Phase 1: Fire off ALL API calls in parallel ──
    with ThreadPoolExecutor(max_workers=20) as pool:
        # Submit odds fetches for every game
        odds_futures = {}
        prop_odds_futures = {}
        team_schedule_futures = {}

        for event in selected_events:
            eid = event["id"]
            home = event["home_team"]
            away = event["away_team"]

            # Game odds (moneyline/spreads/totals)
            if markets_str:
                bookmakers = bookmakers_str.split(",") if bookmakers_str else None
                odds_futures[eid] = pool.submit(
                    get_event_odds, api_key, sport["key"], eid, markets=markets_str, bookmakers=bookmakers
                )

            # Player prop odds
            if selected_props:
                prop_markets_str = ",".join(selected_props)
                bookmakers = bookmakers_str.split(",") if bookmakers_str else None
                prop_odds_futures[eid] = pool.submit(
                    get_event_odds, api_key, sport["key"], eid, markets=prop_markets_str, bookmakers=bookmakers
                )

            # Team schedules (for both teams)
            home_espn = find_team(espn_teams, home)
            away_espn = find_team(espn_teams, away)
            if home_espn and home_espn["id"] not in team_schedule_futures:
                team_schedule_futures[home_espn["id"]] = pool.submit(
                    get_team_schedule, sport["espn_sport"], sport["espn_league"], home_espn["id"]
                )
            if away_espn and away_espn["id"] not in team_schedule_futures:
                team_schedule_futures[away_espn["id"]] = pool.submit(
                    get_team_schedule, sport["espn_sport"], sport["espn_league"], away_espn["id"]
                )

        progress.progress(25, text="Loading team schedules...")

        # Collect results as they complete
        odds_results = {}
        for eid, fut in odds_futures.items():
            try:
                odds_results[eid] = fut.result()
            except Exception as e:
                warnings.append(f"Failed to fetch odds for event {eid}: {e}")

        prop_odds_results = {}
        for eid, fut in prop_odds_futures.items():
            try:
                prop_odds_results[eid] = fut.result()
            except Exception as e:
                warnings.append(f"Failed to fetch prop odds for event {eid}: {e}")

        schedule_results = {}
        for tid, fut in team_schedule_futures.items():
            try:
                schedule_results[tid] = fut.result()
            except Exception:
                schedule_results[tid] = []

        progress.progress(50, text="Getting player props...")

        # ── Phase 2: Parse prop data and fire off ALL player history lookups ──
        prop_history_futures = {}  # (player_name, prop_key) -> future
        parsed_props = {}  # eid -> parsed prop data

        for eid, raw_data in prop_odds_results.items():
            parsed = parse_player_props(raw_data)
            parsed_props[eid] = parsed
            for prop_key, players in parsed.get("props", {}).items():
                for player_name in players:
                    key = (player_name, prop_key)
                    if key not in prop_history_futures:
                        prop_history_futures[key] = pool.submit(
                            get_player_stat_history,
                            sport["espn_sport"], sport["espn_league"],
                            player_name, prop_key, recent_n,
                        )

        # Collect all player histories
        prop_history_results = {}
        for key, fut in prop_history_futures.items():
            try:
                prop_history_results[key] = fut.result()
            except Exception:
                prop_history_results[key] = {"player": key[0], "found": False, "values": []}

    progress.progress(80, text="Analyzing odds against statistics history...")

    # ── Phase 3: Run analysis (CPU-only, fast) ──
    for event in selected_events:
        eid = event["id"]
        home = event["home_team"]
        away = event["away_team"]

        # Team market analysis
        if eid in odds_results:
            game_odds = parse_game_odds(odds_results[eid])
            home_espn = find_team(espn_teams, home)
            away_espn = find_team(espn_teams, away)

            if home_espn and away_espn:
                home_games = schedule_results.get(home_espn["id"], [])
                away_games = schedule_results.get(away_espn["id"], [])

                home_stats = {
                    "season": {
                        "record": home_espn["record"],
                        "wins": home_espn["wins"],
                        "losses": home_espn["losses"],
                        "win_pct": home_espn["win_pct"],
                    },
                    "recent": compute_recent_form(home_games, home_espn["display_name"], n=recent_n),
                    "recent_games": home_games[:recent_n],
                }
                away_stats = {
                    "season": {
                        "record": away_espn["record"],
                        "wins": away_espn["wins"],
                        "losses": away_espn["losses"],
                        "win_pct": away_espn["win_pct"],
                    },
                    "recent": compute_recent_form(away_games, away_espn["display_name"], n=recent_n),
                    "recent_games": away_games[:recent_n],
                }

                if "h2h" in market_keys:
                    all_ml.extend(analyze_moneyline_value(game_odds, home_stats, away_stats, threshold))
                if "spreads" in market_keys:
                    all_spreads.extend(analyze_spreads_value(game_odds, home_stats, away_stats, threshold))
                if "totals" in market_keys:
                    all_totals.extend(analyze_totals_value(game_odds, home_stats, away_stats, threshold))

        # Player props analysis
        if eid in parsed_props:
            prop_data = parsed_props[eid]
            player_histories = {}
            for prop_key, players in prop_data.get("props", {}).items():
                for player_name in players:
                    if player_name not in player_histories:
                        player_histories[player_name] = {}
                    player_histories[player_name][prop_key] = prop_history_results.get(
                        (player_name, prop_key),
                        {"player": player_name, "found": False, "values": []},
                    )
            all_props.extend(analyze_player_props_value(prop_data, player_histories, threshold))

    # Show any warnings that occurred during parallel fetches
    for w in warnings:
        st.warning(w)

    progress.progress(100, text="Analysis complete!")

    # Store results in session state
    st.session_state["analysis_results"] = {
        "all_ml": all_ml,
        "all_spreads": all_spreads,
        "all_totals": all_totals,
        "all_props": all_props,
        "sport_key": sport["key"],
        "total_games": total_games,
        "total_cost": actual_cost,
    }
    # Clear any previous parlay results
    st.session_state.pop("parlay_results", None)

# ──────────────────────────────────────────────────────────
#  Display Results (from session state, persists across reruns)
# ──────────────────────────────────────────────────────────
if "analysis_results" in st.session_state:
    ar = st.session_state["analysis_results"]
    all_ml = ar["all_ml"]
    all_spreads = [c for c in ar["all_spreads"] if c.get("games_sampled", 0) >= 5]
    all_totals = ar["all_totals"]
    all_props = [c for c in ar["all_props"] if c.get("no_history") or c.get("games_sampled", 0) >= 5]

    st.divider()

    # Search filter
    search_filter = st.text_input("🔎 Filter results by team or player", key="result_filter",
                                  placeholder="e.g., Lakers, LeBron James...").strip().lower()

    if search_filter:
        all_ml = [c for c in all_ml if search_filter in c.get("team", "").lower()
                  or search_filter in c.get("opponent", "").lower()]
        all_spreads = [c for c in all_spreads if search_filter in c.get("team", "").lower()
                       or search_filter in c.get("opponent", "").lower()]
        all_totals = [c for c in all_totals if search_filter in c.get("matchup", "").lower()]
        all_props = [c for c in all_props if search_filter in c.get("player", "").lower()
                     or search_filter in c.get("matchup", "").lower()]

    # Count value bets
    value_count = (
        sum(1 for c in all_ml if c["is_value"])
        + sum(1 for c in all_spreads if c["is_value"])
        + sum(1 for c in all_totals if c.get("is_over_value") or c.get("is_under_value"))
        + sum(1 for c in all_props if c["is_value"])
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Games Analyzed", ar["total_games"])
    col2.metric("Value Bets Found", value_count)
    col3.metric("Credits Used", ar["total_cost"])

    # Moneyline results
    if all_ml:
        st.subheader("💰 Moneyline Analysis")
        value_ml = [c for c in all_ml if c["is_value"]]
        other_ml = [c for c in all_ml if not c["is_value"]]

        if value_ml:
            st.success(f"**{len(value_ml)} value bet(s) found!**")
            for c in sorted(value_ml, key=lambda x: x["edge_pct"], reverse=True):
                with st.expander(f"🔥 {c['team']} ({c['home_away']}) vs {c['opponent']}  —  Edge: +{c['edge_pct']}%", expanded=True):
                    cols = st.columns(4)
                    cols[0].metric("Book Implied", f"{c['book_implied_prob']}%")
                    cols[1].metric("Season Win%", f"{c['season_win_pct']}%")
                    cols[2].metric("Recent Win%", f"{c['recent_win_pct']}%")
                    cols[3].metric("Edge", f"+{c['edge_pct']}%", delta=f"{c['best_price']:+d} at {c['best_book']}")

        if other_ml:
            with st.expander(f"Other matchups ({len(other_ml)})"):
                rows = []
                for c in other_ml:
                    rows.append({
                        "Team": c["team"],
                        "Home/Away": c["home_away"],
                        "Book Implied": f"{c['book_implied_prob']}%",
                        "Historical": f"{c['hist_prob']}%",
                        "Edge": f"{c['edge_pct']:+.2f}%",
                    })
                st.table(rows)

    # Spreads results
    if all_spreads:
        st.subheader("📊 Spread Analysis")
        value_sp = [c for c in all_spreads if c["is_value"]]
        other_sp = [c for c in all_spreads if not c["is_value"]]

        if value_sp:
            st.success(f"**{len(value_sp)} spread value bet(s) found!**")
            for c in sorted(value_sp, key=lambda x: x["edge_pct"], reverse=True):
                with st.expander(f"🔥 {c['team']} {c['spread']:+.2f} ({c['home_away']})  —  Edge: +{c['edge_pct']}%", expanded=True):
                    cols = st.columns(4)
                    cols[0].metric("Spread", f"{c['spread']:+.2f}")
                    cols[1].metric("Avg Margin", f"{c['avg_margin']:+.2f}")
                    cols[2].metric("Cover Rate", f"{c['cover_rate']}%")
                    cols[3].metric("Edge", f"+{c['edge_pct']}%")

        if other_sp:
            with st.expander(f"Other spreads ({len(other_sp)})"):
                rows = []
                for c in other_sp:
                    rows.append({
                        "Team": c["team"],
                        "Spread": f"{c['spread']:+.2f}",
                        "Cover Rate": f"{c['cover_rate']}%",
                        "Edge": f"{c['edge_pct']:+.2f}%",
                    })
                st.table(rows)

    # Totals results
    if all_totals:
        st.subheader("📈 Over/Under Analysis")
        for c in all_totals:
            flag = ""
            if c.get("is_over_value"):
                flag = " 🔥 OVER VALUE"
            elif c.get("is_under_value"):
                flag = " 🔥 UNDER VALUE"

            with st.expander(f"{c['matchup']}  —  Line: {c['line']}{flag}"):
                cols = st.columns(4)
                cols[0].metric("Line", c["line"])
                cols[1].metric("Projected Total", c["projected_total"])
                cols[2].metric("Diff from Line", f"{c['diff_from_line']:+.2f}")
                cols[3].metric("Over Hit Rate", f"{c['over_hit_rate']}%")

    # Player Props results
    if all_props:
        st.subheader("🏀 Player Props Analysis")
        value_props = [c for c in all_props if c["is_value"]]
        no_hist = [c for c in all_props if c.get("no_history")]
        other_props = [c for c in all_props if not c["is_value"] and not c.get("no_history")]

        if value_props:
            st.success(f"**{len(value_props)} prop value bet(s) found!**")
            for c in sorted(value_props, key=lambda x: x["edge_pct"], reverse=True):
                with st.expander(f"🔥 {c['player']} — {c['prop_label']} {c['direction']} {c['line']}  —  Edge: +{c['edge_pct']}%", expanded=True):
                    cols = st.columns(5)
                    cols[0].metric("Line", c["line"])
                    cols[1].metric("Avg Stat", c["avg_stat"])
                    cols[2].metric("Over Rate", f"{c['over_rate']}%")
                    cols[3].metric("Direction", c["direction"])
                    cols[4].metric("Edge", f"+{c['edge_pct']}%", delta=f"{c['best_price']:+d}")
                    st.caption(f"Matchup: {c['matchup']}  |  Over: {c['over_price']:+d} ({c['over_implied']}%)  |  Under: {c['under_price']:+d} ({c['under_implied']}%)  |  {c['games_sampled']} games sampled")

        if other_props:
            with st.expander(f"Other props ({len(other_props)})"):
                rows = []
                for c in other_props:
                    rows.append({
                        "Player": c["player"],
                        "Prop": c["prop_label"],
                        "Line": c["line"],
                        "Avg": c["avg_stat"],
                        "Direction": c["direction"],
                        "Edge": f"{c['edge_pct']:+.2f}%",
                    })
                st.table(rows)

        if no_hist:
            st.caption(f"ℹ️ {len(no_hist)} prop(s) skipped — no ESPN history found")

    if not any([all_ml, all_spreads, all_totals, all_props]) and not search_filter:
        st.info("No results. Select markets or props and click Analyze.")
