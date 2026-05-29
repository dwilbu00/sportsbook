"""
Sportsbook Value Finder — Streamlit UI
=======================================
Launch with:  streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Add script dir to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Auto-shutdown when browser disconnects ──
SHUTDOWN_TIMEOUT = 30  # seconds after last session disconnects

if "shutdown_watchdog_started" not in st.session_state:
    st.session_state.shutdown_watchdog_started = True

    def _shutdown_watchdog():
        """Monitor active sessions and exit when all browsers disconnect."""
        from streamlit.runtime import get_instance
        idle_since = None
        while True:
            time.sleep(5)
            runtime = get_instance()
            if runtime is None:
                continue
            active = runtime._session_mgr.list_active_sessions()
            if not active:
                if idle_since is None:
                    idle_since = time.time()
                elif time.time() - idle_since >= SHUTDOWN_TIMEOUT:
                    os._exit(0)
            else:
                idle_since = None

    threading.Thread(target=_shutdown_watchdog, daemon=True).start()

from odds_client import (
    get_upcoming_events,
    get_event_odds,
    parse_game_odds,
    parse_player_props,
    parse_alt_player_props,
    parse_alt_team_lines,
    get_remaining_credits,
    is_event_cached,
    american_to_decimal,
    PLAYER_PROPS_BY_SPORT,
    PLAYER_PROP_ALTS_BY_SPORT,
    TEAM_ALT_MARKETS,
    PROP_LABELS,
)


def fetch_and_attach_alt_lines(ar, api_key, sport_key, bookmakers_str):
    """Pull alternate-line prices for the events that produced value bets
    in `ar` (the cached analysis_results dict) and mutate `ar` in place so
    safe-mode props get real alt prices, ladders are attached for display,
    and safe-mode props without an actual DK alt line are filtered out.

    Returns (alt_credits_used, warnings, removed_no_alt).
    """
    all_props = ar["all_props"]
    all_spreads = ar["all_spreads"]
    all_totals = ar["all_totals"]

    needed_alts = {}
    prop_alt_map = PLAYER_PROP_ALTS_BY_SPORT.get(sport_key, {})
    for c in all_props:
        # DK only offers OVER alt lines for player props, so skip UNDER value
        # bets (safe-mode props are always implicitly OVER thresholds).
        if not c.get("event_id"):
            continue
        if c.get("safe_mode"):
            pass  # always include — safe-mode needs alt prices to be bettable
        elif c.get("is_value") and c.get("direction") == "OVER":
            pass
        else:
            continue
        alt_key = prop_alt_map.get(c.get("prop"))
        if alt_key:
            needed_alts.setdefault(c["event_id"], set()).add(alt_key)
    for c in all_spreads:
        if c.get("event_id") and c.get("is_value"):
            needed_alts.setdefault(c["event_id"], set()).add(TEAM_ALT_MARKETS["spreads"])
    for c in all_totals:
        if c.get("event_id") and (c.get("is_over_value") or c.get("is_under_value")):
            needed_alts.setdefault(c["event_id"], set()).add(TEAM_ALT_MARKETS["totals"])

    alt_data_by_event = {}
    alt_credits_used = 0
    warnings = []
    if not needed_alts:
        return alt_credits_used, warnings, 0

    bookmakers = bookmakers_str.split(",") if bookmakers_str else None
    with ThreadPoolExecutor(max_workers=10) as pool:
        alt_futures = {}
        for eid, alt_market_set in needed_alts.items():
            alt_markets_str = ",".join(sorted(alt_market_set))
            alt_credits_used += len(alt_market_set)
            alt_futures[eid] = pool.submit(
                get_event_odds, api_key, sport_key, eid,
                markets=alt_markets_str, bookmakers=bookmakers,
            )
        for eid, fut in alt_futures.items():
            try:
                raw = fut.result()
                alt_data_by_event[eid] = {
                    "props": parse_alt_player_props(raw),
                    "team": parse_alt_team_lines(raw),
                }
            except Exception as e:
                warnings.append(f"Failed to fetch alt lines for event {eid}: {e}")

    # Attach alt prices to safe-mode prop candidates.
    for c in all_props:
        if not c.get("safe_mode") or not c.get("event_id"):
            continue
        event_alts = alt_data_by_event.get(c["event_id"], {}).get("props", {})
        ladder = event_alts.get((c["player"], c["prop"]), [])
        target_line = c["safe_threshold"] - 0.5
        match = next((e for e in ladder if e["line"] == target_line), None)
        if match and match.get("over_price") is not None:
            c["safe_alt_price"] = match["over_price"]
            c["safe_alt_line"] = match["line"]
        if ladder:
            c["alt_ladder"] = ladder

    for c in all_spreads:
        if c.get("is_value") and c.get("event_id") in alt_data_by_event:
            c["alt_ladder"] = alt_data_by_event[c["event_id"]]["team"]["spreads"].get(c["team"], [])
    for c in all_totals:
        if (c.get("is_over_value") or c.get("is_under_value")) and c.get("event_id") in alt_data_by_event:
            team_alts = alt_data_by_event[c["event_id"]]["team"]["totals"]
            c["alt_ladder_over"] = team_alts.get("Over", [])
            c["alt_ladder_under"] = team_alts.get("Under", [])
    for c in all_props:
        if c.get("is_value") and not c.get("safe_mode") and c.get("event_id"):
            event_alts = alt_data_by_event.get(c["event_id"], {}).get("props", {})
            ladder = event_alts.get((c["player"], c["prop"]), [])
            if ladder:
                c["alt_ladder"] = ladder

    # Filter out safe-mode props without an actual DK alt line.
    removed_no_alt = 0
    filtered_props = []
    for c in all_props:
        if c.get("safe_mode") and c.get("safe_alt_price") is None:
            removed_no_alt += 1
            continue
        filtered_props.append(c)
    ar["all_props"] = filtered_props
    ar["alt_data"] = alt_data_by_event
    ar["alt_credits"] = ar.get("alt_credits", 0) + alt_credits_used
    ar["total_cost"] = ar.get("total_cost", 0) + alt_credits_used
    ar["alts_applied"] = True
    return alt_credits_used, warnings, removed_no_alt


def _prop_prob_fn(candidate, direction="over"):
    """Return a callable (line) → percent hit probability for a player-prop
    candidate, using its stored historical values + recency weights. Returns
    None if the candidate lacks stored values."""
    values = candidate.get("_values")
    weights = candidate.get("_weights")
    if not values or not weights:
        return None
    total_w = sum(weights)
    if total_w == 0:
        return None

    def _fn(line):
        if direction == "over":
            hit = sum(w for v, w in zip(values, weights) if v > line)
        else:
            hit = sum(w for v, w in zip(values, weights) if v < line)
        return 100.0 * hit / total_w

    return _fn


def _dk_payout_strs(american_price, stake=10):
    """Return (value_str, delta_str) for a single-bet DK Payout metric.
    Value is the total payout in the box (e.g. '$11.82'); delta below the
    box is the stake context ('on $10 (+118)')."""
    if american_price is None:
        return "n/a", ""
    dec = american_to_decimal(american_price)
    total_payout = dec * stake
    return f"${total_payout:.2f}", f"on ${stake:.0f} ({american_price:+d})"


def _render_alt_ladder(ladder, direction="over", title="Alt lines (DK)", around_line=None, n_around=3, line_style="decimal", prob_fn=None):
    """Render a compact alt-line ladder near a target line. If `direction` is
    'over', show OVER price column; if 'under', show UNDER; if 'both', show both.
    `line_style`: 'decimal' (e.g. 24.5), 'spread' (signed, e.g. -3.5), or
    'threshold' (player-prop N+ form, e.g. 2.5 → '3+').
    `prob_fn`: optional callable (line) → percentage (0–100) that adds a
    'Prob @ Line' column between Price and Payout (single-direction tables only)."""
    import math
    if not ladder:
        return
    # Filter to lines near the target (±n_around steps) if a target is given.
    if around_line is not None:
        sorted_l = sorted(ladder, key=lambda e: abs(e["line"] - around_line))
        ladder = sorted(sorted_l[: 2 * n_around + 1], key=lambda e: e["line"])
    rows = []
    for entry in ladder:
        if line_style == "spread":
            line_str = f"{entry['line']:+g}"
        elif line_style == "threshold":
            line_str = f"{int(math.ceil(entry['line']))}+"
        else:
            line_str = entry["line"]
        row = {"Line": line_str}

        def _payout(price):
            if price is None:
                return "—"
            return f"${american_to_decimal(price) * 10:.2f}"

        if direction in ("over", "both"):
            op = entry.get("over_price")
            if op is None and "price" in entry:  # spreads/totals use flat 'price'
                op = entry["price"]
            if direction == "both":
                row["OVER"] = f"{op:+d}" if op is not None else "—"
                row["OVER Payout"] = _payout(op)
            else:
                row["Price"] = f"{op:+d}" if op is not None else "—"
                if prob_fn is not None:
                    p = prob_fn(entry["line"])
                    row["Prob @ Line"] = f"{p:.1f}%" if p is not None else "—"
                row["Payout on $10"] = _payout(op)
        if direction in ("under", "both"):
            up = entry.get("under_price")
            if direction == "both":
                row["UNDER"] = f"{up:+d}" if up is not None else "—"
                row["UNDER Payout"] = _payout(up)
            else:
                row["Price"] = f"{up:+d}" if up is not None else "—"
                if prob_fn is not None:
                    p = prob_fn(entry["line"])
                    row["Prob @ Line"] = f"{p:.1f}%" if p is not None else "—"
                row["Payout on $10"] = _payout(up)
        rows.append(row)
    st.caption(f"**{title}**")
    st.dataframe(rows, hide_index=True, width='content')
from espn_client import (
    get_all_teams,
    get_team_schedule,
    compute_recent_form,
    find_team,
    annotate_opponent_strength,
    build_team_defense_lookup,
    get_player_stat_history,
)
from analysis import (
    analyze_moneyline_value,
    analyze_spreads_value,
    analyze_totals_value,
    analyze_player_props_value,
    generate_parlays,
)

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

SPORTS = {
    "NBA": {
        "key": "basketball_nba",
        "espn_sport": "basketball",
        "espn_league": "nba",
        "recent_n_default": 10,
    },
    "MLB": {
        "key": "baseball_mlb",
        "espn_sport": "baseball",
        "espn_league": "mlb",
        "recent_n_default": 20,
    },
    "NFL": {
        "key": "americanfootball_nfl",
        "espn_sport": "football",
        "espn_league": "nfl",
        "recent_n_default": 8,
    },
}

MARKET_OPTIONS = {
    "Moneyline": "h2h",
    "Spreads": "spreads",
    "Over/Under": "totals",
}


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_api_key(key):
    """Save the API key to config.json."""
    config = load_config()
    config["odds_api_key"] = key
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)


def needs_setup(config):
    """Check if the API key needs to be configured."""
    key = config.get("odds_api_key", "")
    return not key or key == "YOUR_API_KEY_HERE"


def show_setup_wizard():
    """Display a first-run setup wizard to help the user get an API key."""
    st.markdown("## 👋 Welcome to Sportsbook Value Finder!")
    st.markdown(
        "This app compares sportsbook odds against historical stats "
        "to find value betting opportunities. To get started, you'll "
        "need a **free API key** from The Odds API."
    )

    st.divider()

    st.markdown("### Get your free API key in 3 steps:")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Step 1")
        st.markdown("Click the link below to open The Odds API signup page")
        st.link_button(
            "🔑 Get Free API Key",
            "https://the-odds-api.com/#get-access",
            width='stretch',
        )
    with col2:
        st.markdown("#### Step 2")
        st.markdown(
            'Choose the **Starter (FREE)** plan — 500 credits/month. '
            "Enter your email and you'll receive your API key instantly."
        )
    with col3:
        st.markdown("#### Step 3")
        st.markdown(
            "Check your email for the API key, then paste it below and click Save."
        )

    st.divider()

    st.markdown("### Enter your API key")
    new_key = st.text_input(
        "API Key",
        placeholder="Paste your API key here...",
        help="The key will be saved locally in config.json. It never leaves your computer.",
    )

    if st.button("💾 Save API Key", type="primary", width='stretch'):
        if new_key and len(new_key) > 10 and new_key != "YOUR_API_KEY_HERE":
            save_api_key(new_key)
            st.success("API key saved! The app will now reload...")
            st.balloons()
            st.rerun()
        else:
            st.error("Please enter a valid API key.")

    st.divider()
    with st.expander("ℹ️ About credits & costs"):
        st.markdown(
            """
            - The **free plan** gives you **500 credits/month** (resets on the 1st)
            - Listing upcoming games: **FREE** (0 credits)
            - Each market (moneyline, spreads, totals): **1 credit per call**
            - Each player prop market: **1 credit per game**
            - This app **caches results for 10 minutes** to avoid wasting credits
            - Example: analyzing 3 games with moneyline + spreads = **6 credits**
            """
        )


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
def fetch_player_history_cached(espn_sport, espn_league, player_name, prop_key, n,
                                cache_version="v2-minutes"):
    """
    Cached player history. `cache_version` is a no-op arg whose value is
    bumped whenever the upstream return shape changes (forces cache miss).
    Note: must NOT start with underscore — Streamlit excludes _-prefixed
    args from the cache key.
    """
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
#  First-run setup check
# ──────────────────────────────────────────────────────────
config = load_config()

if needs_setup(config):
    st.title("🎯 Sportsbook Value Finder")
    show_setup_wizard()
    st.stop()

api_key = config["odds_api_key"]

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
    # Recent-games window is now hardcoded per sport (half-life weighting
    # inside makes the upper bound effectively self-tuning).
    recent_n = sport.get("recent_n_default", 10)
    safe_mode = st.toggle(
        "🛡️ Safe mode (player props)",
        value=False,
        key="safe_mode",
        help=(
            "OVER-only player props. Uses per-player weighted-quantile alt "
            "lines (whole numbers, e.g. 'Points 8+') derived from each "
            "player's recent-game distribution at the chosen confidence."
        ),
    )
    if safe_mode:
        safe_target = st.slider(
            "Safe-mode confidence",
            0.70, 0.99, 0.95, 0.01,
            key="safe_target",
            help="Target hit-rate at our suggested alt threshold.",
        )
    else:
        safe_target = 0.95

    fetch_alt_lines = st.toggle(
        "📋 Fetch alt-line prices",
        value=False,
        key="fetch_alt_lines",
        help=(
            "After the main analysis, pull alternate-line prices only for "
            "events that produced a value bet. Lets the UI show real DK "
            "prices at suggested alt lines and lets safe-mode parlay payouts "
            "use the actual alt-line price. Costs ~1 extra credit per "
            "(value-bet event × alt market) — typically a small add-on. "
            "Toggling this ON while viewing an analysis pulls alts for the "
            "current results without re-running the full analysis."
        ),
    )

    total_per_game = len(market_keys) + len(selected_props)
    remaining = get_remaining_credits()
    credit_info = f"**{total_per_game} credit(s)** per game (max)"
    if remaining is not None:
        credit_info += f"  |  **{remaining}** credits remaining"
    st.info(credit_info)

    st.divider()
    with st.expander("🔑 API Key"):
        st.caption(f"Key: ...{api_key[-6:]}")
        new_key = st.text_input("Change API Key", type="password", label_visibility="collapsed",
                                placeholder="Enter new key...")
        if st.button("Update Key"):
            if new_key and len(new_key) > 10:
                save_api_key(new_key)
                st.success("Key updated!")
                st.rerun()
            else:
                st.error("Invalid key")

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
bookmakers_list = config.get("bookmakers", [])
bookmakers_param = bookmakers_list if bookmakers_list else None
bookmakers_str = ",".join(bookmakers_list) if bookmakers_list else ""
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
    analyze_clicked = st.button("🔍 Analyze", type="primary", width='stretch')
with col_parlay:
    parlay_clicked = st.button("🎰 Value Parlays", width='stretch')
with col_safe:
    safe_clicked = st.button("🛡️ Safe Parlays", width='stretch')

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
    # The analysis layer may upgrade mode="value" to "safe_value" when the
    # underlying props are safe-mode candidates. Use the per-parlay `mode`
    # field if present.
    sample = next((parlays[s] for s in [3, 4, 5] if s in parlays), None)
    effective_mode = (sample or {}).get("mode", mode)
    # `has_safe_legs` tells us the underlying analysis was run in safe mode.
    # When False, both Safe and Value parlay buttons share the same display
    # style ("regular") with consistent labels and metrics.
    has_safe_legs = any(
        leg.get("safe_mode") for leg in (sample or {}).get("legs", [])
    )

    # Determine display style independent of button:
    #   - "sa_safe":  safe-mode analysis + Safe Parlays
    #   - "sa_value": safe-mode analysis + Value Parlays (auto-upgraded to safe_value)
    #   - "regular":  regular analysis (either button)
    if has_safe_legs and effective_mode == "safe":
        display = "sa_safe"
    elif effective_mode == "safe_value":
        display = "sa_value"
    else:
        display = "regular"

    if display == "sa_safe":
        mode_label = "🛡️ Safe"
        caption_text = "Prioritizing highest probability of hitting with positive edge"
    elif display == "sa_value":
        mode_label = "🎯 Aggressive Safe"
        caption_text = "Safe-mode legs ranked by how close their suggested threshold sits to the book line (or above it) — more aggressive than Safe Parlays, still grounded in the model's high-confidence thresholds."
    elif effective_mode == "safe":  # regular analysis + Safe button
        mode_label = "🛡️ Safe"
        caption_text = "Prioritizing highest joint probability of hitting across positive-edge legs"
    else:  # regular analysis + Value button
        mode_label = "🎰 Value"
        caption_text = "Prioritizing best parlay edge over the book with sport-specific correlation analysis"

    st.divider()
    st.subheader(f"{mode_label} Parlays")
    st.caption(caption_text)

    if parlays:
        for size in [3, 4, 5]:
            if size not in parlays:
                continue
            p = parlays[size]
            # Expander headline
            if display == "sa_safe":
                headline = f"Hit Prob: {p['combined_hist_prob']}%"
            elif display == "sa_value":
                gap = p.get("avg_line_gap")
                gap_str = f"{gap:+.2f}" if gap is not None else "n/a"
                headline = f"Hit Prob: {p['combined_hist_prob']}%  |  Avg gap to book line: {gap_str}"
            else:  # regular
                pe = p.get("parlay_edge_pct")
                pe_str = f"{pe:+.2f}%" if pe is not None else "n/a"
                headline = (f"Hit Prob: {p['combined_hist_prob']}%  |  "
                            f"Parlay Edge: {pe_str}")

            with st.expander(
                f"{'⭐' * size}  Best {size}-Leg Parlay  —  {headline}",
                expanded=(size == 3),
            ):
                for i, leg in enumerate(p["legs"], 1):
                    prob_pct = min(round(leg["hist_prob"] * 100, 2), 99.99)
                    price_str = (f"  ({leg['odds_price']:+d})"
                                 if leg.get("odds_price") else "")

                    if display == "sa_safe":
                        st.markdown(
                            f"**Leg {i}:** {leg['label']}  —  "
                            f"Prob: {prob_pct}%  |  Δ vs book line: +{leg['edge_pct']}%"
                            + price_str
                        )
                    elif display == "sa_value":
                        if leg.get("safe_mode"):
                            gap = leg.get("line_gap", 0.0)
                            gap_icon = "🚀" if gap >= 0 else "📍"
                            st.markdown(
                                f"**Leg {i}:** {gap_icon} {leg['label']}  —  "
                                f"Prob: {prob_pct}%  |  "
                                f"Book line: {leg.get('book_line')}  |  "
                                f"Suggested: {leg.get('safe_threshold')}+  |  "
                                f"Gap: {gap:+.2f}"
                                + price_str
                            )
                        else:
                            # Non-safe leg (ML / spread / total)
                            st.markdown(
                                f"**Leg {i}:** {leg['label']}  —  "
                                f"Prob: {prob_pct}%  |  Edge: +{leg['edge_pct']}%"
                                + price_str
                            )
                    else:  # regular
                        st.markdown(
                            f"**Leg {i}:** {leg['label']}  —  "
                            f"Prob: {prob_pct}%  |  Edge: +{leg['edge_pct']}%"
                            + price_str
                        )

                st.divider()
                # ── Book payout metrics (shown for every display style) ──
                american_str = (f"{p['parlay_american_odds']:+d}"
                                if p.get("parlay_american_odds") is not None else "n/a")
                payout_help = (
                    "Estimated DK payout from multiplying each leg's decimal odds. "
                    "Standard cross-game parlays should match DK's slip exactly. "
                    + ("⚠️ Same-game parlay: DK applies a correlation discount, so the actual slip will pay less than this. "
                       if p.get("has_sgp") else "")
                    + ("Spread/total legs without a captured price were assumed -110."
                       if p.get("payout_uses_default_price") else "")
                ).strip()

                if display == "sa_safe":
                    cols = st.columns(4)
                    cols[0].metric("Legs", size)
                    cols[1].metric(
                        "Hit Probability",
                        f"{p['combined_hist_prob']}%",
                        help="Joint probability all legs hit at their suggested safe thresholds, adjusted for sport-specific correlations between legs.",
                    )
                    cols[2].metric(
                        "Hit Prob (no correlation)",
                        f"{p['combined_hist_prob_indep']}%",
                        help="Naive product of each leg's probability, assuming legs are independent. Compare against Hit Probability to see how much correlation between legs helps or hurts.",
                    )
                    cols[3].metric(
                        "DK Payout",
                        american_str,
                        delta=f"$10 pays ${p.get('payout_per_10', 0) + 10:.2f}",
                        delta_color="off",
                        help=payout_help,
                    )
                elif display == "sa_value":
                    cols = st.columns(5)
                    cols[0].metric("Legs", size)
                    cols[1].metric(
                        "Hit Probability",
                        f"{p['combined_hist_prob']}%",
                        help="Joint probability all legs hit at their suggested thresholds, adjusted for sport-specific correlations.",
                    )
                    avg_gap = p.get("avg_line_gap")
                    cols[2].metric(
                        "Avg Gap to Book Line",
                        f"{avg_gap:+.2f}" if avg_gap is not None else "n/a",
                        help="Average of (suggested threshold − book line) across the safe-mode legs. 0 = right at the book line; positive = the model expects a result above the book line; negative = a safer threshold below the book line.",
                    )
                    total_gap = p.get("total_line_gap")
                    cols[3].metric(
                        "Total Gap",
                        f"{total_gap:+.2f}" if total_gap is not None else "n/a",
                        help="Sum of per-leg gaps to the book line. Higher = more aggressive parlay.",
                    )
                    cols[4].metric(
                        "DK Payout",
                        american_str,
                        delta=f"$10 pays ${p.get('payout_per_10', 0) + 10:.2f}",
                        delta_color="off",
                        help=payout_help,
                    )
                else:  # regular analysis (either button)
                    cols = st.columns(5)
                    cols[0].metric("Legs", size)
                    cols[1].metric(
                        "Hit Probability",
                        f"{p['combined_hist_prob']}%",
                        help="Joint probability all legs hit, adjusted for sport-specific correlations between legs.",
                    )
                    cols[2].metric(
                        "Hit Prob (no correlation)",
                        f"{p['combined_hist_prob_indep']}%",
                        help="Naive product of each leg's probability, assuming legs are independent. Compare against Hit Probability to see how much correlation between legs helps or hurts.",
                    )
                    pe = p.get("parlay_edge_pct")
                    cols[3].metric(
                        "Parlay Edge",
                        f"{pe:+.2f}%" if pe is not None else "n/a",
                        help="Joint hit probability minus the book's combined implied probability — the model's expected edge over the book on the full parlay.",
                    )
                    cols[4].metric(
                        "DK Payout",
                        american_str,
                        delta=f"$10 pays ${p.get('payout_per_10', 0) + 10:.2f}",
                        delta_color="off",
                        help=payout_help,
                    )

                if p.get("has_sgp"):
                    st.caption(
                        "⚠️ Same-game parlay — DK applies a correlation discount; "
                        "the actual slip will pay less than the listed DK Payout."
                    )
                if p.get("payout_uses_default_price"):
                    st.caption(
                        "ℹ️ One or more spread/total legs had no captured price; "
                        "payout assumes -110 for those legs."
                    )
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

        # Build a per-team avg-points-allowed lookup so the player-prop analyzer
        # can apply an opponent-defense weighting to historical games.
        team_defense = build_team_defense_lookup(schedule_results, espn_teams)

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

                home_recent = home_games[:recent_n]
                away_recent = away_games[:recent_n]
                # Annotate each recent game with opponent_win_pct for
                # opponent-strength weighting in the analyzers.
                annotate_opponent_strength(home_recent, home_espn["display_name"], espn_teams)
                annotate_opponent_strength(away_recent, away_espn["display_name"], espn_teams)

                home_stats = {
                    "season": {
                        "record": home_espn["record"],
                        "wins": home_espn["wins"],
                        "losses": home_espn["losses"],
                        "win_pct": home_espn["win_pct"],
                    },
                    "recent": compute_recent_form(home_games, home_espn["display_name"], n=recent_n),
                    "recent_games": home_recent,
                }
                away_stats = {
                    "season": {
                        "record": away_espn["record"],
                        "wins": away_espn["wins"],
                        "losses": away_espn["losses"],
                        "win_pct": away_espn["win_pct"],
                    },
                    "recent": compute_recent_form(away_games, away_espn["display_name"], n=recent_n),
                    "recent_games": away_recent,
                }

                def _tag_event(cands):
                    for c in cands:
                        c["event_id"] = eid
                    return cands

                if "h2h" in market_keys:
                    all_ml.extend(_tag_event(analyze_moneyline_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"])))
                if "spreads" in market_keys:
                    all_spreads.extend(_tag_event(analyze_spreads_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"])))
                if "totals" in market_keys:
                    all_totals.extend(_tag_event(analyze_totals_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"])))

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
            new_props = analyze_player_props_value(prop_data, player_histories, threshold,
                                                  sport_key=sport["key"],
                                                  team_defense=team_defense,
                                                  espn_teams=espn_teams,
                                                  safe_mode=safe_mode,
                                                  safe_target=safe_target,
                                                  team_schedules=schedule_results)
            for c in new_props:
                c["event_id"] = eid
            all_props.extend(new_props)

    # Show any warnings that occurred during parallel fetches
    for w in warnings:
        st.warning(w)

    progress.progress(95, text="Analysis complete!")

    # Store results in session state (alt fetch happens below, possibly
    # post-hoc when the user toggles "Fetch alt-line prices" ON).
    st.session_state["analysis_results"] = {
        "all_ml": all_ml,
        "all_spreads": all_spreads,
        "all_totals": all_totals,
        "all_props": all_props,
        "sport_key": sport["key"],
        "total_games": total_games,
        "total_cost": actual_cost,
        "alt_credits": 0,
        "alt_data": {},
        "alts_applied": False,
    }
    # Clear any previous parlay results
    st.session_state.pop("parlay_results", None)

# ── Conditional alt-line fetch ──
# Always fetch when safe mode is on (safe-mode props need real alt prices and
# are filtered if no DK alt exists). Otherwise only fetch when the toggle is
# ON. The toggle separately controls whether ladders are *displayed*.
if "analysis_results" in st.session_state:
    ar = st.session_state["analysis_results"]
    need_alts = fetch_alt_lines or any(c.get("safe_mode") for c in ar.get("all_props", []))
    if need_alts and not ar.get("alts_applied"):
        with st.spinner("Pulling alt-line prices for value-bet events..."):
            alt_credits, alt_warnings, removed_no_alt = fetch_and_attach_alt_lines(
                ar, api_key, ar["sport_key"], bookmakers_str
            )
        for w in alt_warnings:
            st.warning(w)
        if removed_no_alt:
            st.warning(
                f"Hid {removed_no_alt} safe-mode prop(s) with no DK alt line at the suggested threshold."
            )
        # Clear cached parlays so they re-build using new alt prices.
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
                    cols = st.columns(5)
                    cols[0].metric("Book Implied", f"{c['book_implied_prob']}%")
                    cols[1].metric("Season Win%", f"{c['season_win_pct']}%")
                    cols[2].metric("Recent Win%", f"{c['recent_win_pct']}%")
                    cols[3].metric("Edge", f"+{c['edge_pct']}%", delta=f"{c['best_price']:+d} at {c['best_book']}")
                    p_val, p_delta = _dk_payout_strs(c.get("best_price"))
                    cols[4].metric("DK Payout", p_val, delta=p_delta, delta_color="off",
                                   help="American odds and profit on a $10 bet at DraftKings.")

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
                    cols = st.columns(5)
                    cols[0].metric("Spread", f"{c['spread']:+.2f}")
                    cols[1].metric("Avg Margin", f"{c['avg_margin']:+.2f}")
                    cols[2].metric("Cover Rate", f"{c['cover_rate']}%")
                    cols[3].metric("Edge", f"+{c['edge_pct']}%")
                    p_val, p_delta = _dk_payout_strs(c.get("price"))
                    cols[4].metric("DK Payout", p_val, delta=p_delta, delta_color="off",
                                   help="American odds and profit on a $10 bet at DraftKings.")
                    if fetch_alt_lines and c.get("alt_ladder"):
                        _render_alt_ladder(c["alt_ladder"], direction="over",
                                           title=f"Alt spreads for {c['team']}",
                                           around_line=c["spread"],
                                           line_style="spread")

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
                # Show the payout for whichever side is flagged value (default
                # to OVER if neither, so the metric is informative).
                if c.get("is_under_value"):
                    payout_price = c.get("under_price")
                    payout_label = "DK Payout (UNDER)"
                else:
                    payout_price = c.get("over_price")
                    payout_label = "DK Payout (OVER)"
                p_val, p_delta = _dk_payout_strs(payout_price)
                cols = st.columns(5)
                cols[0].metric("Line", c["line"])
                cols[1].metric("Projected Total", c["projected_total"])
                cols[2].metric("Diff from Line", f"{c['diff_from_line']:+.2f}")
                cols[3].metric("Over Hit Rate", f"{c['over_hit_rate']}%")
                cols[4].metric(payout_label, p_val, delta=p_delta, delta_color="off",
                               help="American odds and profit on a $10 bet at DraftKings.")
                # Merge OVER and UNDER alt ladders by line so the table shows
                # both prices side-by-side.
                if fetch_alt_lines:
                    over_l = c.get("alt_ladder_over") or []
                    under_l = c.get("alt_ladder_under") or []
                    if over_l or under_l:
                        merged = {}
                        for e in over_l:
                            merged.setdefault(e["line"], {})["over_price"] = e.get("price")
                        for e in under_l:
                            merged.setdefault(e["line"], {})["under_price"] = e.get("price")
                        combined = [{"line": ln, **vals} for ln, vals in sorted(merged.items())]
                        _render_alt_ladder(combined, direction="both",
                                           title="Alt totals",
                                           around_line=c["line"])

    # Player Props results
    if all_props:
        is_safe = any(c.get("safe_mode") for c in all_props)
        header = "🛡️ Player Props Analysis (Safe Mode)" if is_safe else "🏀 Player Props Analysis"
        st.subheader(header)
        value_props = [c for c in all_props if c["is_value"]]
        no_hist = [c for c in all_props if c.get("no_history")]
        other_props = [c for c in all_props if not c["is_value"] and not c.get("no_history")]

        def _safe_label(c):
            """Display bet as 'Points {N}+' instead of 'OVER 9.5' in safe mode."""
            return f"{c['prop_label']} {c['safe_threshold']}+"

        if value_props:
            st.success(f"**{len(value_props)} prop value bet(s) found!**")
            # In safe mode rank by line_gap (book-line cushion); else by edge%.
            sorted_props = sorted(value_props,
                                  key=lambda x: x.get("line_gap", x["edge_pct"]),
                                  reverse=True)
            for c in sorted_props:
                if c.get("safe_mode"):
                    gap = c["line_gap"]
                    if gap >= 0:
                        tag = "✅ bet standard line"
                        alt_advice = f"OVER {c['line']} is already safe (gap +{gap})"
                    else:
                        # Suggest the highest standard alt below safe threshold
                        alt_line = c["safe_threshold"] - 0.5
                        tag = "↘ alt line needed"
                        alt_advice = f"find an OVER ≤ {alt_line} (gap {gap})"
                    title = (f"🛡️ {c['player']} — {_safe_label(c)}  "
                             f"[{tag}]  book line: {c['line']}")
                    with st.expander(title, expanded=(gap >= 0)):
                        cols = st.columns(6)
                        cols[0].metric("Suggested", f"{c['prop_label']} {c['safe_threshold']}+")
                        cols[1].metric(
                            "Prob @ Suggested",
                            f"{c['model_hit_at_safe']}%",
                        )
                        cols[2].metric(
                            "Prob @ Book Line",
                            f"{c['model_hit_at_line']}%",
                        )
                        cols[3].metric(
                            "Δ (safe − book)",
                            f"{c['model_delta']:+.2f}%",
                        )
                        cols[4].metric("Avg Stat", c["avg_stat"])
                        # Prefer real alt-line price when we fetched alts; fall
                        # back to the book-line price with a caveat.
                        if c.get("safe_alt_price") is not None:
                            p_val, p_delta = _dk_payout_strs(c["safe_alt_price"])
                            payout_label = f"DK Payout (OVER {c['safe_alt_line']})"
                            payout_help = f"Actual DraftKings price for OVER {c['safe_alt_line']} (≡ {c['prop_label']} {c['safe_threshold']}+)."
                        else:
                            p_val, p_delta = _dk_payout_strs(c.get("over_price"))
                            payout_label = "DK Payout (book line)"
                            payout_help = "Payout for the OVER at the standard book line. Alt-line fetch is disabled or no alt was offered at the suggested threshold — DK's actual alt price will differ."
                        cols[5].metric(payout_label, p_val, delta=p_delta,
                                       delta_color="off", help=payout_help)
                        st.caption(
                            f"Matchup: {c['matchup']}"
                            f"  |  Predicted line: {c['avg_stat']}"
                            f"  |  Line gap: {c['line_gap']:+.2f}"
                        )
                        if fetch_alt_lines and c.get("alt_ladder"):
                            _render_alt_ladder(c["alt_ladder"], direction="over",
                                               title=f"Alt lines for {c['player']} ({c['prop_label']})",
                                               around_line=c["safe_threshold"] - 0.5,
                                               line_style="threshold",
                                               prob_fn=_prop_prob_fn(c, "over"))
                else:
                    hit_prob = c["over_rate"] if c["direction"] == "OVER" else round(100.0 - c["over_rate"], 2)
                    bet_price = (c.get("over_price") if c["direction"] == "OVER"
                                 else c.get("under_price"))
                    with st.expander(f"🔥 {c['player']} — {c['prop_label']} {c['direction']} {c['line']}  —  Edge: +{c['edge_pct']}%", expanded=True):
                        cols = st.columns(6)
                        cols[0].metric("Line", c["line"])
                        cols[1].metric("Avg Stat", c["avg_stat"])
                        cols[2].metric(f"{c['direction']} Hit Prob", f"{hit_prob}%")
                        cols[3].metric("Direction", c["direction"])
                        cols[4].metric("Edge", f"+{c['edge_pct']}%", delta=f"{c['best_price']:+d}")
                        p_val, p_delta = _dk_payout_strs(bet_price)
                        cols[5].metric(
                            f"DK Payout ({c['direction']})", p_val, delta=p_delta, delta_color="off",
                            help=f"Payout for the {c['direction']} at the book line on DraftKings.",
                        )
                        line_gap = c["avg_stat"] - c["line"]
                        st.caption(
                            f"Matchup: {c['matchup']}"
                            f"  |  Book line: {c['line']}"
                            f"  |  Predicted line: {c['avg_stat']}"
                            f"  |  Line gap: {line_gap:+.2f}"
                        )
                        # DK only offers OVER alt lines for player props.
                        if fetch_alt_lines and c["direction"] == "OVER" and c.get("alt_ladder"):
                            _render_alt_ladder(c["alt_ladder"], direction="over",
                                               title=f"Alt lines for {c['player']} ({c['prop_label']}) — OVER",
                                               around_line=c["line"],
                                               line_style="threshold",
                                               prob_fn=_prop_prob_fn(c, "over"))

        if other_props:
            with st.expander(f"Other props ({len(other_props)})"):
                rows = []
                for c in other_props:
                    if c.get("safe_mode"):
                        rows.append({
                            "Player": c["player"],
                            "Suggested": f"{c['prop_label']} {c['safe_threshold']}+",
                            "Prob @ Suggested": f"{c['model_hit_at_safe']}%",
                            "Book Line": c["line"],
                            "Prob @ Book": f"{c['model_hit_at_line']}%",
                            "Δ": f"{c['model_delta']:+.2f}%",
                        })
                    else:
                        hit_prob = c["over_rate"] if c["direction"] == "OVER" else round(100.0 - c["over_rate"], 2)
                        rows.append({
                            "Player": c["player"],
                            "Prop": c["prop_label"],
                            "Line": c["line"],
                            "Avg": c["avg_stat"],
                            "Direction": c["direction"],
                            "Hit Prob": f"{hit_prob}%",
                            "Edge": f"{c['edge_pct']:+.2f}%",
                        })
                st.table(rows)

        if no_hist:
            st.caption(f"ℹ️ {len(no_hist)} prop(s) skipped — no ESPN history found")

    if not any([all_ml, all_spreads, all_totals, all_props]) and not search_filter:
        st.info("No results. Select markets or props and click Analyze.")
