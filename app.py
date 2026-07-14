"""
Sportsbook Value Finder — Streamlit UI
=======================================
Launch with:  streamlit run app.py
"""

import streamlit as st
import json
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
    parse_alt_player_props,
    parse_alt_team_lines,
    get_remaining_credits,
    is_event_cached,
    american_to_decimal,
    american_to_implied_prob,
    PLAYER_PROPS_BY_SPORT,
    PLAYER_PROP_ALTS_BY_SPORT,
    TEAM_ALT_MARKETS,
    PROP_LABELS,
)


def fetch_and_attach_alt_lines(ar, api_key, sport_key, bookmakers_str):
    """Pull alternate-line prices for the events that produced value bets
    in `ar` (the cached analysis_results dict) and mutate `ar` in place so
    safe-mode props get real alt prices, ladders are attached for display,
    and safe-mode props without an actual DK alt line or enough price-adjusted
    edge are filtered out.

    Returns (alt_credits_used, warnings, removed_no_alt, removed_no_value).
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
        # A safe prop whose market has no Odds API alternate-market mapping
        # (currently including pitcher outs) cannot be price-verified, so do
        # not surface it as a value recommendation.
        removed_no_alt = sum(1 for c in all_props if c.get("safe_mode"))
        ar["all_props"] = [c for c in all_props if not c.get("safe_mode")]
        ar["alts_applied"] = True
        return alt_credits_used, warnings, removed_no_alt, 0

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
            model_prob = c.get("model_hit_at_safe", 0.0) / 100.0
            implied_prob = american_to_implied_prob(match["over_price"])
            edge = model_prob - implied_prob
            expected_roi = model_prob * american_to_decimal(match["over_price"]) - 1.0
            c["safe_alt_implied"] = round(implied_prob * 100, 2)
            c["safe_alt_edge_pct"] = round(edge * 100, 2)
            c["edge_pct"] = round(edge * 100, 2)
            c["expected_roi_pct"] = round(expected_roi * 100, 2)
            c["best_price"] = match["over_price"]
            c["value_pending"] = False
            c["is_value"] = edge >= (ar.get("threshold_pct", 5.0) / 100.0)
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

    # Filter out safe-mode props without an actual DK alt line or without the
    # same minimum model-vs-price edge required by standard recommendations.
    removed_no_alt = 0
    removed_no_value = 0
    filtered_props = []
    for c in all_props:
        if c.get("safe_mode") and c.get("safe_alt_price") is None:
            removed_no_alt += 1
            continue
        if c.get("safe_mode") and not c.get("is_value"):
            removed_no_value += 1
            continue
        filtered_props.append(c)
    ar["all_props"] = filtered_props
    ar["alt_data"] = alt_data_by_event
    ar["alt_credits"] = ar.get("alt_credits", 0) + alt_credits_used
    ar["total_cost"] = ar.get("total_cost", 0) + alt_credits_used
    ar["alts_applied"] = True
    return alt_credits_used, warnings, removed_no_alt, removed_no_value


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
    `prob_fn`: optional callable (line) → weighted historical percentage (0–100)
    that adds a history column (single-direction tables only)."""
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
                    row["Historical @ Line"] = f"{p:.1f}%" if p is not None else "—"
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
                    row["Historical @ Line"] = f"{p:.1f}%" if p is not None else "—"
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


def render_model_guide():
    """Render model definitions and committed chronological-backtest results."""
    st.title("📘 Model Guide & Performance")
    st.caption(
        "How recommendations are calculated, what each number means, and the "
        "validation evidence currently committed with the production model."
    )

    st.warning(
        "Backtests estimate historical performance; they do not guarantee future "
        "results. Sportsbook prices, injuries, lineups, and player roles can change "
        "after an analysis is run."
    )

    overview_tab, performance_tab, safe_tab, mlb_tab = st.tabs([
        "How predictions work",
        "Backtest performance",
        "Safe Mode",
        "MLB xStats",
    ])

    with overview_tab:
        st.subheader("The four numbers to compare on every recommendation")
        st.markdown(
            """
            - **Model probability** — the calibrated probability that this exact bet wins.
            - **Book implied probability** — the break-even probability encoded by the offered American price, including vig.
            - **Edge** — `model probability − book implied probability`, measured in percentage points.
            - **Expected ROI** — `model probability × decimal odds − 1`; the modeled average profit or loss per dollar staked.

            A positive edge is necessary for a value recommendation. The standard
            recommendation threshold is **+5 percentage points**. A high hit probability
            alone is not value if the price is too expensive, and a high projected average
            does not by itself determine the chance of clearing a line.
            """
        )
        st.info(
            "Projected average is the center of the modeled stat distribution—not a "
            "guaranteed result and not the same thing as hit probability. A 1.2-hit "
            "average can still produce a meaningful probability of 2+ hits because the "
            "full discrete game-to-game distribution determines the bet."
        )
        st.markdown(
            """
            **Team markets** use recency- and venue-weighted scoring/margin histories,
            coherent Normal margin distributions, probability shrinkage, and sport-specific
            matchup features. **Player props** add reliability filtering, residual-distribution
            calibration, warmup blending, and—when enough resolved live predictions exist—
            Platt recalibration. **Parlays** estimate a joint probability with a Gaussian
            copula rather than blindly multiplying independent probabilities.
            """
        )

    calibration_blobs = {}
    sport_names = {
        "baseball_mlb": "MLB",
        "basketball_nba": "NBA",
        "americanfootball_nfl": "NFL",
    }
    for sport_key, sport_name in sport_names.items():
        path = os.path.join(SCRIPT_DIR, "calibration", f"{sport_key}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                calibration_blobs[sport_key] = json.load(f)
        except (OSError, json.JSONDecodeError):
            calibration_blobs[sport_key] = {}

    with performance_tab:
        st.subheader("Player-prop chronological calibration scores")
        st.markdown(
            "Each prop's probability method and tuning variant were selected by fitting "
            "the earlier 50% of observations and scoring the later 50%. Accuracy is the "
            "percentage of later-period outcomes for which the model put the correct side above "
            "50%. Brier score evaluates the probability itself (lower is better; 0.25 is "
            "the uninformative score for a balanced 50/50 forecast). Because that later "
            "period also selected the production variant, these are model-selection "
            "scores—not an untouched final deployment audit."
        )
        prop_rows = []
        for sport_key, sport_name in sport_names.items():
            blob = calibration_blobs.get(sport_key, {})
            fit_date = (blob.get("fit_timestamp") or "")[:10] or "Not recorded"
            for prop_key, cfg in (blob.get("props") or {}).items():
                prop_rows.append({
                    "Sport": sport_name,
                    "Prediction": PROP_LABELS.get(prop_key, prop_key.replace("_", " ").title()),
                    "Chronological accuracy": (
                        f"{cfg['fit_hit_pct']:.2f}%"
                        if cfg.get("fit_hit_pct") is not None else "Not exported"
                    ),
                    "Chronological Brier": (
                        f"{cfg['fit_brier']:.4f}"
                        if cfg.get("fit_brier") is not None else "Not exported"
                    ),
                    "Probability method": cfg.get("method", "—"),
                    "Full fit observations": cfg.get("n_obs", "—"),
                    "Calibration date": fit_date,
                })
        if prop_rows:
            st.dataframe(prop_rows, hide_index=True, width="stretch")
        else:
            st.info("No committed player-prop calibration summaries were found.")

        st.subheader("Forward prediction tracking")
        from recalibration import prediction_performance_summary
        forward = prediction_performance_summary()
        if forward["total"]:
            hit_rate = forward.get("direction_hit_rate")
            probability_brier = forward.get("probability_brier")
            realized_roi = forward.get("realized_roi")
            forward_cols = st.columns(5)
            forward_cols[0].metric("Logged predictions", forward["total"])
            forward_cols[1].metric("Resolved", forward["resolved"])
            forward_cols[2].metric(
                "Model-side hit rate",
                f"{hit_rate * 100:.1f}%" if hit_rate is not None else "Not enough data",
            )
            forward_cols[3].metric(
                "Probability Brier",
                f"{probability_brier:.4f}"
                if probability_brier is not None else "Not enough data",
            )
            forward_cols[4].metric(
                "Model-side ROI",
                f"{realized_roi * 100:+.1f}%"
                if realized_roi is not None else "Awaiting priced results",
            )
            forward_rows = []
            for row in forward["by_prop"]:
                row_hit_rate = row.get("direction_hit_rate")
                row_brier = row.get("probability_brier")
                row_roi = row.get("realized_roi")
                forward_rows.append({
                    "Sport": sport_names.get(row["sport_key"], row["sport_key"]),
                    "Prediction": PROP_LABELS.get(
                        row["prop_key"],
                        (row["prop_key"] or "Unknown").replace("_", " ").title(),
                    ),
                    "Logged": row["total"],
                    "Resolved": row["resolved"],
                    "Pending": row["pending"],
                    "Pushes": row["pushes"],
                    "Model-side hit rate": (
                        f"{row_hit_rate * 100:.1f}%"
                        if row_hit_rate is not None else "—"
                    ),
                    "Probability Brier": (
                        f"{row_brier:.4f}" if row_brier is not None else "—"
                    ),
                    "Model-side ROI": (
                        f"{row_roi * 100:+.1f}%" if row_roi is not None else "—"
                    ),
                    "Priced results": row["priced_resolved"],
                })
            st.dataframe(forward_rows, hide_index=True, width="stretch")
            st.caption(
                "Forward rows are real app predictions, deduplicated by "
                "sport/player/date/prop/line. New rows score the final published "
                "probability; legacy rows fall back to their pre-Platt raw "
                "probability. ROI is shown only for newly logged rows that "
                "retain the offered "
                "price; older rows remain usable for hit-rate and Brier scoring. "
                "Closing-line value is not available because closing prices are "
                "not captured."
            )
        else:
            st.info(
                "No forward predictions have been logged yet. Predictions will "
                "appear here after player-prop analyses are run and past games "
                "are resolved."
            )

        st.subheader("Team-market validation status")
        team_rows = []
        market_labels = {
            "moneyline": "Moneyline",
            "spreads": "Spread",
            "totals": "Total",
        }
        for sport_key, sport_name in sport_names.items():
            blob = calibration_blobs.get(sport_key, {})
            shrink = blob.get("prob_shrink") or {}
            source = ((blob.get("meta") or {}).get("prob_shrink") or {}).get("source")
            for market_key, market_label in market_labels.items():
                shrink_value = shrink.get(market_key)
                team_rows.append({
                    "Sport": sport_name,
                    "Prediction": market_label,
                    "Backtest calibration": source or "No committed odds-fit metadata",
                    "Probability shrink": (
                        f"{shrink_value:.3f}"
                        if isinstance(shrink_value, (int, float)) else "Not fitted"
                    ),
                    "Published holdout accuracy": "Not exported",
                })
        st.dataframe(team_rows, hide_index=True, width="stretch")
        st.caption(
            "The team-market calibration files currently persist fitted shrink weights but "
            "not the complete scored holdout report. The app says ‘Not exported’ rather "
            "than presenting an unsupported accuracy percentage."
        )

    with safe_tab:
        st.subheader("Safe Mode uses confidence and price—neither one overrides the other")
        st.markdown(
            """
            1. Filter unreliable or post-layoff player histories.
            2. Build the same adjusted projection used by standard player props.
            3. Apply the same residual, early-season warmup, and Platt probability calibration.
            4. Find the highest integer threshold whose calibrated model probability reaches the selected confidence target.
            5. Require the weighted historical hit rate to be within 5 percentage points of that target.
            6. Fetch the exact DraftKings alternate-line price.
            7. Recommend the bet only when `model probability − actual alt-price implied probability ≥ 5 percentage points`.

            High-confidence `1+` floors are rejected above an 80% target, and thresholds
            that collapse below half the standard book line are rejected. Value Parlays
            rank price-verified legs by correlation-adjusted expected return; Safe Parlays
            prioritize joint hit probability among those same positive-edge legs.
            """
        )

    with mlb_tab:
        st.subheader("MLB expected-stat and matchup features")
        st.markdown(
            "xERA/xwOBA, starter quality, handedness splits, lineup quality, and bullpen "
            "suppression are active inputs for MLB team markets. Batter props use "
            "starter xBA or K rate at the starter's projected workload, with the "
            "remaining bullpen exposure held neutral until it can be validated as-of. "
            "When a complete batting order is announced, batter-hit projections also "
            "blend recent at-bats with the expected at-bats for that lineup slot. The "
            "same batting-order adjustment is disabled for batter strikeouts because "
            "it did not improve forward validation."
        )
        mlb_meta = calibration_blobs.get("baseball_mlb", {}).get("meta") or {}
        prop_fit = ((mlb_meta.get("starter_adjustment") or {}).get("props") or {})
        holdout = (prop_fit if prop_fit.get("results")
                   else mlb_meta.get("prop_matchup_holdout") or {})
        holdout_rows = []
        for prop_key, result in (holdout.get("results") or {}).items():
            if result.get("status"):
                holdout_rows.append({
                    "Prediction": PROP_LABELS.get(prop_key, prop_key),
                    "Fit observations": "—",
                    "Holdout observations": "—",
                    "Selected weight": result.get("selected_weight", 0.0),
                    "Holdout MAE": "Not tested",
                    "Decision": result["status"],
                })
                continue
            baseline = result.get("baseline_mae")
            selected = result.get("selected_mae")
            holdout_rows.append({
                "Prediction": PROP_LABELS.get(prop_key, prop_key),
                "Fit observations": result.get("fit_n", "—"),
                "Holdout observations": result.get("holdout_n", "—"),
                "Selected weight": result.get("selected_weight", 0.0),
                "Holdout MAE": (
                    f"{baseline:.5f} → {selected:.5f}"
                    if baseline is not None and selected is not None else "Not exported"
                ),
                "Decision": result.get("decision") or (
                    "Enabled" if result.get("selected_weight", 0.0) > 0
                    else "Disabled"
                ),
            })
        if holdout_rows:
            st.dataframe(holdout_rows, hide_index=True, width="stretch")
            st.caption(
                f"Source: {holdout.get('source', 'committed calibration metadata')}; "
                f"holdout: {holdout.get('holdout_window', 'not recorded')}."
            )
        st.info(
            "Current result: **batter strikeouts use a 0.5 starter-matchup weight** "
            "after improving the main holdout and both rolling folds. Batter hits remain "
            "at 0.0 because their xBA candidate regressed in one rolling fold. Pitcher "
            "strikeouts, outs, and earned runs remain at 0.0."
        )
        lineup_fit = mlb_meta.get("lineup_adjustment") or {}
        lineup_hits = (lineup_fit.get("results") or {}).get("batter_hits") or {}
        if lineup_hits:
            rolling_gains = ", ".join(
                f"{fold.get('gain_pct', 0):+.3f}%"
                for fold in lineup_hits.get("rolling_folds", []))
            st.caption(
                "Announced-lineup batter-hits validation: "
                f"n={lineup_hits.get('holdout_n', '—')}; holdout MAE "
                f"{lineup_hits.get('baseline_mae', 0):.6f} → "
                f"{lineup_hits.get('selected_mae', 0):.6f}; "
                f"rolling-fold gains {rolling_gains or 'not exported'}."
            )

        recency = ((calibration_blobs.get("baseball_mlb", {}).get("meta") or {})
                   .get("prop_recency_holdout") or {})
        recency_rows = []
        for prop_key, result in (recency.get("results") or {}).items():
            decay_mae = result.get("decay_mae")
            no_decay_mae = result.get("no_decay_mae")
            recency_rows.append({
                "Prediction": PROP_LABELS.get(prop_key, prop_key),
                "Holdout observations": result.get("holdout_n", "—"),
                "Decay MAE": f"{decay_mae:.5f}" if decay_mae is not None else "—",
                "No-decay MAE": (
                    f"{no_decay_mae:.5f}" if no_decay_mae is not None else "—"
                ),
                "Production choice": result.get("selected", "—"),
            })
        if recency_rows:
            st.subheader("MLB player-prop recency validation")
            st.dataframe(recency_rows, hide_index=True, width="stretch")
            st.caption(
                "MLB props use equal weights inside the live 20-game window by default. "
                "Pitcher strikeouts and pitcher outs retain their calibrated decay because "
                "removing it made the later-period error worse. MLB team-market recency is "
                "calibrated separately."
            )

        rolling = ((calibration_blobs.get("baseball_mlb", {}).get("meta") or {})
                   .get("rolling_player_form_holdout") or {})
        rolling_rows = []
        for prop_key, result in (rolling.get("results") or {}).items():
            baseline = result.get("baseline_mae")
            candidate = result.get("candidate_mae")
            rolling_rows.append({
                "Prediction": PROP_LABELS.get(prop_key, prop_key),
                "Rolling signal": result.get("signal", "—"),
                "Holdout observations": result.get("holdout_n", "—"),
                "Candidate weight": result.get("candidate_weight", 0.0),
                "Holdout MAE": (
                    f"{baseline:.5f} → {candidate:.5f}"
                    if baseline is not None and candidate is not None else "—"
                ),
                "Production weight": result.get("selected_weight", 0.0),
            })
        if rolling_rows:
            st.subheader("Savant rolling player-form validation")
            st.dataframe(rolling_rows, hide_index=True, width="stretch")
            st.caption(
                "The exact Baseball Savant Rolling Selection service was evaluated as-of "
                "each game: 50-PA rolling xBA for hitters and 100-PA rolling K%/xwOBA for "
                "pitchers. A signal is enabled only if the weight selected on the earlier "
                "period also improves the later holdout. None currently pass that gate."
            )


# ──────────────────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Sportsbook Value Finder", page_icon="🎯", layout="wide")

# ──────────────────────────────────────────────────────────
#  First-run setup check
# ──────────────────────────────────────────────────────────
config = load_config()

with st.sidebar:
    app_page = st.radio(
        "Navigate",
        ["🎯 Value Finder", "📘 Model Guide & Performance"],
        key="app_page",
    )

if app_page == "📘 Model Guide & Performance":
    render_model_guide()
    st.stop()

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
    if "markets" not in st.session_state:
        st.session_state["markets"] = ["Moneyline"]
    selected_markets = st.multiselect(
        "Select markets",
        list(MARKET_OPTIONS.keys()),
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
    threshold = 5.0  # default value edge threshold for non-safe-mode analysis
    # Recent-games window is hardcoded per sport. Per-prop calibration decides
    # whether games inside the window are equally or exponentially weighted.
    recent_n = sport.get("recent_n_default", 10)
    safe_mode = st.toggle(
        "🎯 Alt lines (player props)",
        value=False,
        key="safe_mode",
        help=(
            "OVER-only player props. Uses calibrated per-player lower bounds "
            "(whole numbers, e.g. 'Points 8+') derived from each player's "
            "weighted recent-game distribution at the chosen confidence. "
            "The exact alt price is fetched automatically; a pick is shown only "
            "when it also clears the normal value-edge threshold at that price."
        ),
    )
    if safe_mode:
        safe_target_pct = st.slider(
            "Alt-lines confidence",
            70, 99, 80, 1,
            key="safe_target_pct",
            format="%d%%",
            help="Target hit-rate at our suggested alt threshold.",
        )
        safe_target = safe_target_pct / 100.0
    else:
        safe_target = 0.95

    fetch_alt_lines = st.toggle(
        "📋 Fetch alt-line prices",
        value=False,
        key="fetch_alt_lines",
        help=(
            "After the main analysis, pull alternate-line prices only for "
            "events that produced a value bet. Lets the UI show real DK "
            "prices at suggested alt lines and lets alt-line parlay payouts "
            "use the actual alt-line price. Costs ~1 extra credit per "
            "(value-bet event × alt market) — typically a small add-on. "
            "Toggling this ON while viewing an analysis pulls alts for the "
            "current results without re-running the full analysis."
            " Safe Mode always fetches its suggested alt prices regardless of "
            "this display toggle."
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

# Clear stale parlays as soon as a new Analyze is requested so they don't
# render below this point before the analysis block runs.
if analyze_clicked:
    st.session_state.pop("parlay_results", None)
    st.session_state.pop("parlay_mode", None)

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
        mode_label = "🎯 Alt Lines"
        caption_text = "Prioritizing highest probability of hitting with positive edge"
    elif display == "sa_value":
        mode_label = "⚡ Aggressive Alt Lines"
        caption_text = "Confidence-qualified, price-verified alt-line legs ranked by correlation-adjusted expected return."
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
            # DK payout (total return on $10) shown in headline + metric.
            total_payout = (p.get("payout_per_10", 0) or 0) + 10
            payout_label_str = f"${total_payout:.2f}"

            # Expander headline
            pe = p.get("parlay_edge_pct")
            pe_str = f"{pe:+.2f}%" if pe is not None else "n/a"
            headline = (f"Hit Prob: {p['combined_hist_prob']}%  |  "
                        f"Parlay Edge: {pe_str}  |  "
                        f"Expected ROI: {p.get('expected_roi_pct', 0):+.2f}%  |  "
                        f"DK Payout: {payout_label_str}")

            with st.expander(
                f"{'⭐' * size}  Best {size}-Leg Parlay  —  {headline}",
                expanded=(size == 3),
            ):
                for i, leg in enumerate(p["legs"], 1):
                    prob_pct = min(round(leg["hist_prob"] * 100, 2), 99.99)
                    price_str = (f"  ({leg['odds_price']:+d})"
                                 if leg.get("odds_price") else "")
                    leg_ev = (leg["hist_prob"] * american_to_decimal(leg["odds_price"]) - 1.0) * 100 \
                        if leg.get("odds_price") is not None else None
                    leg_ev_str = f"  |  Expected ROI: {leg_ev:+.2f}%" if leg_ev is not None else ""

                    if display == "sa_safe":
                        st.markdown(
                            f"**Leg {i}:** {leg['label']}  —  "
                            f"Prob: {prob_pct}%  |  Edge: {leg['edge_pct']:+.2f}%"
                            + leg_ev_str
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
                                f"Edge: {leg['edge_pct']:+.2f}%"
                                + leg_ev_str
                                + price_str
                            )
                        else:
                            # Non-safe leg (ML / spread / total)
                            st.markdown(
                                f"**Leg {i}:** {leg['label']}  —  "
                                f"Prob: {prob_pct}%  |  Edge: {leg['edge_pct']:+.2f}%"
                                + leg_ev_str
                                + price_str
                            )
                    else:  # regular
                        st.markdown(
                            f"**Leg {i}:** {leg['label']}  —  "
                            f"Prob: {prob_pct}%  |  Edge: {leg['edge_pct']:+.2f}%"
                            + leg_ev_str
                            + price_str
                        )

                st.divider()
                # ── Book payout metrics (shown for every display style) ──
                payout_help = (
                    "Estimated DK payout from multiplying each leg's decimal odds. "
                    "Standard cross-game parlays should match DK's slip exactly. "
                    + ("⚠️ Same-game parlay: DK applies a correlation discount, so the actual slip will pay less than this. "
                       if p.get("has_sgp") else "")
                    + ("Spread/total legs without a captured price were assumed -110."
                       if p.get("payout_uses_default_price") else "")
                ).strip()

                # Same DK Payout format as value bets: $X.XX in box, "on $10 (+XXX)" below.
                payout_delta = (f"on $10 ({p['parlay_american_odds']:+d})"
                                if p.get("parlay_american_odds") is not None else "on $10")

                if display == "sa_safe":
                    cols = st.columns(7)
                    cols[0].metric("Legs", size)
                    cols[1].metric(
                        "Hit Probability",
                        f"{p['combined_hist_prob']}%",
                        help="Joint probability all legs hit at their suggested safe thresholds, adjusted for sport-specific correlations between legs.",
                    )
                    cols[2].metric("Book Implied", f"{p['combined_implied_prob']}%")
                    cols[3].metric("Parlay Edge", f"{p['parlay_edge_pct']:+.2f}%")
                    cols[4].metric("Expected ROI", f"{p['expected_roi_pct']:+.2f}%")
                    cols[5].metric(
                        "Hit Prob (no correlation)",
                        f"{p['combined_hist_prob_indep']}%",
                        help="Naive product of each leg's probability, assuming legs are independent. Compare against Hit Probability to see how much correlation between legs helps or hurts.",
                    )
                    cols[6].metric(
                        "DK Payout",
                        payout_label_str,
                        delta=payout_delta,
                        delta_color="off",
                        help=payout_help,
                    )
                elif display == "sa_value":
                    cols = st.columns(6)
                    cols[0].metric("Legs", size)
                    cols[1].metric(
                        "Hit Probability",
                        f"{p['combined_hist_prob']}%",
                        help="Joint probability all legs hit at their suggested thresholds, adjusted for sport-specific correlations.",
                    )
                    cols[2].metric("Book Implied", f"{p['combined_implied_prob']}%")
                    cols[3].metric("Parlay Edge", f"{p['parlay_edge_pct']:+.2f}%")
                    cols[4].metric("Expected ROI", f"{p['expected_roi_pct']:+.2f}%")
                    cols[5].metric(
                        "DK Payout",
                        payout_label_str,
                        delta=payout_delta,
                        delta_color="off",
                        help=payout_help,
                    )
                else:  # regular analysis (either button)
                    cols = st.columns(7)
                    cols[0].metric("Legs", size)
                    cols[1].metric(
                        "Hit Probability",
                        f"{p['combined_hist_prob']}%",
                        help="Joint probability all legs hit, adjusted for sport-specific correlations between legs.",
                    )
                    cols[2].metric("Book Implied", f"{p['combined_implied_prob']}%")
                    cols[3].metric(
                        "Hit Prob (no correlation)",
                        f"{p['combined_hist_prob_indep']}%",
                        help="Naive product of each leg's probability, assuming legs are independent. Compare against Hit Probability to see how much correlation between legs helps or hurts.",
                    )
                    cols[4].metric(
                        "Parlay Edge",
                        f"{pe:+.2f}%" if pe is not None else "n/a",
                        help="Joint hit probability minus the book's combined implied probability — the model's expected edge over the book on the full parlay.",
                    )
                    cols[5].metric("Expected ROI", f"{p['expected_roi_pct']:+.2f}%")
                    cols[6].metric(
                        "DK Payout",
                        payout_label_str,
                        delta=payout_delta,
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

        # Phase 1–3 (MLB): probable-starter / opponent / bullpen matchup
        # features, built once per game and shared by team markets AND props.
        # Degrades to None (existing model) if anything is unavailable.
        matchup_features = None
        confirmed_lineup = None
        if sport["key"] == "baseball_mlb":
            import mlb_starters
            game_date = event.get("commence_time", "")[:10]
            try:
                commence = datetime.fromisoformat(
                    event["commence_time"].replace("Z", "+00:00"))
                game_date = commence.astimezone(
                    ZoneInfo("America/New_York")).date().isoformat()
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            try:
                matchup_features = mlb_starters.build_matchup_features(
                    home, away, game_date, int(game_date[:4]))
            except Exception as e:
                warnings.append(f"Starter features unavailable for {away} @ {home}: {e}")
            try:
                confirmed_lineup = mlb_starters.get_confirmed_lineup(
                    home, away, game_date)
            except Exception as e:
                warnings.append(f"Lineup data unavailable for {away} @ {home}: {e}")
        elif sport["key"] == "americanfootball_nfl":
            # NFL EPA edge (net EPA/play, season-to-date) — feeds ML + spreads
            # via the same generic starter_edge margin hook. Degrades to None.
            try:
                import nfl_epa
                game_date = event.get("commence_time", "")[:10]
                season = nfl_epa.season_for_date(game_date)
                # Prefer the committed precomputed ratings (no runtime download).
                ratings = nfl_epa.live_ratings(season)
                matchup_features = nfl_epa.build_matchup_features(
                    home, away, game_date, season, team_ratings=ratings)
            except Exception as e:
                warnings.append(f"EPA features unavailable for {away} @ {home}: {e}")

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
                    all_ml.extend(_tag_event(analyze_moneyline_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"], matchup_features=matchup_features)))
                if "spreads" in market_keys:
                    all_spreads.extend(_tag_event(analyze_spreads_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"], matchup_features=matchup_features)))
                if "totals" in market_keys:
                    all_totals.extend(_tag_event(analyze_totals_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"], matchup_features=matchup_features)))

        # Player props analysis
        if eid in parsed_props:
            prop_data = parsed_props[eid]
            player_histories = {}
            for prop_key, players in prop_data.get("props", {}).items():
                for player_name in players:
                    if player_name not in player_histories:
                        player_histories[player_name] = {}
                    history = dict(prop_history_results.get(
                        (player_name, prop_key),
                        {"player": player_name, "found": False, "values": []},
                    ))
                    if confirmed_lineup:
                        lineup_context = mlb_starters.lineup_player_context(
                            confirmed_lineup, player_name)
                        if lineup_context:
                            history["batting_order"] = lineup_context["batting_order"]
                    player_histories[player_name][prop_key] = history
            new_props = analyze_player_props_value(prop_data, player_histories, threshold,
                                                  sport_key=sport["key"],
                                                  team_defense=team_defense,
                                                  espn_teams=espn_teams,
                                                  safe_mode=safe_mode,
                                                  safe_target=safe_target,
                                                  team_schedules=schedule_results,
                                                  matchup_features=matchup_features)
            for c in new_props:
                c["event_id"] = eid
            all_props.extend(new_props)

    # ── Safe-mode confidence filter for spreads / totals ──
    # Apply the alt-lines confidence (rounded to nearest 10%) to team-level
    # bets so only spreads/totals whose model hit-probability clears the
    # threshold are flagged as value.
    if safe_mode:
        # Round the model's hit probability to the nearest whole percent
        # (79.96→80) and compare against the slider directly.
        def _round1(p):
            return int(p + 0.5)
        target_pct = safe_target * 100
        for c in all_spreads:
            c["is_value"] = (c.get("is_value", False)
                             and _round1(c.get("cover_rate", 0)) >= target_pct)
        for c in all_totals:
            ohr = c.get("over_hit_rate", 0)
            c["is_over_value"] = (c.get("is_over_value", False)
                                  and _round1(ohr) >= target_pct)
            c["is_under_value"] = (c.get("is_under_value", False)
                                   and _round1(100 - ohr) >= target_pct)

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
        "threshold_pct": threshold,
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
            alt_credits, alt_warnings, removed_no_alt, removed_no_value = fetch_and_attach_alt_lines(
                ar, api_key, ar["sport_key"], bookmakers_str
            )
        for w in alt_warnings:
            st.warning(w)
        if removed_no_alt:
            st.warning(
                f"Hid {removed_no_alt} alt-line prop(s) with no DK alt at the suggested threshold."
            )
        if removed_no_value:
            st.info(
                f"Hid {removed_no_value} alt-line prop(s) whose model edge was below "
                f"the {ar.get('threshold_pct', 5.0):g}% minimum at the actual DK price."
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
                with st.expander(f"🔥 {c['team']} ({c['home_away']}) vs {c['opponent']}  —  Best edge: +{c['best_edge_pct']}%", expanded=True):
                    cols = st.columns(7)
                    cols[0].metric("Model Probability", f"{c['blended_prob']}%")
                    cols[1].metric("Book Implied", f"{c['best_book_implied_prob']}%")
                    cols[2].metric("Edge", f"{c['best_edge_pct']:+.2f}%",
                                   delta=f"{c['best_price']:+d} at {c['best_book']}")
                    cols[3].metric("Expected ROI", f"{c['expected_roi_pct']:+.2f}%")
                    cols[4].metric("Season Win%", f"{c['season_win_pct']}%")
                    cols[5].metric("Recent Win%", f"{c['recent_win_pct']}%")
                    p_val, p_delta = _dk_payout_strs(c.get("best_price"))
                    cols[6].metric("DK Payout", p_val, delta=p_delta, delta_color="off",
                                   help="American odds and profit on a $10 bet at DraftKings.")

        if other_ml:
            with st.expander(f"Other matchups ({len(other_ml)})"):
                rows = []
                for c in other_ml:
                    rows.append({
                        "Team": c["team"],
                        "Home/Away": c["home_away"],
                        "Book Implied": f"{c['book_implied_prob']}%",
                        "Model Probability": f"{c.get('blended_prob', c['hist_prob'])}%",
                        "Edge": f"{c['edge_pct']:+.2f}%",
                        "Expected ROI": f"{c.get('expected_roi_pct', 0):+.2f}%",
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
                    cols = st.columns(7)
                    cols[0].metric("Spread", f"{c['spread']:+.2f}")
                    cols[1].metric("Model Probability", f"{c['cover_rate']}%")
                    cols[2].metric("Book Implied", f"{c['implied_prob']}%")
                    cols[3].metric("Edge", f"{c['edge_pct']:+.2f}%")
                    cols[4].metric("Expected ROI", f"{c['expected_roi_pct']:+.2f}%")
                    cols[5].metric("Projected Margin", f"{c['pred_game_margin']:+.2f}")
                    p_val, p_delta = _dk_payout_strs(c.get("price"))
                    cols[6].metric("DK Payout", p_val, delta=p_delta, delta_color="off",
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
                        "Model Probability": f"{c['cover_rate']}%",
                        "Book Implied": f"{c.get('implied_prob', 50.0)}%",
                        "Edge": f"{c['edge_pct']:+.2f}%",
                        "Expected ROI": (f"{c['expected_roi_pct']:+.2f}%"
                                         if c.get("expected_roi_pct") is not None else "n/a"),
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
                    side = "UNDER"
                    payout_price = c.get("under_price")
                    payout_label = "DK Payout (UNDER)"
                    model_probability = 100.0 - c["over_hit_rate"]
                    implied_probability = c.get("under_implied", 50.0)
                    edge_pct = c.get("under_edge_pct", model_probability - implied_probability)
                    expected_roi = c.get("under_expected_roi_pct")
                else:
                    side = "OVER"
                    payout_price = c.get("over_price")
                    payout_label = "DK Payout (OVER)"
                    model_probability = c["over_hit_rate"]
                    implied_probability = c.get("over_implied", 50.0)
                    edge_pct = c.get("over_edge_pct", model_probability - implied_probability)
                    expected_roi = c.get("over_expected_roi_pct")
                p_val, p_delta = _dk_payout_strs(payout_price)
                cols = st.columns(7)
                cols[0].metric("Line", c["line"])
                cols[1].metric("Projected Total", c["projected_total"])
                cols[2].metric(f"Model Probability ({side})", f"{model_probability:.2f}%")
                cols[3].metric("Book Implied", f"{implied_probability:.2f}%")
                cols[4].metric("Edge", f"{edge_pct:+.2f}%")
                cols[5].metric("Expected ROI", (f"{expected_roi:+.2f}%"
                                                 if expected_roi is not None else "n/a"))
                cols[6].metric(payout_label, p_val, delta=p_delta, delta_color="off",
                               help="American odds and profit on a $10 bet at DraftKings.")
                st.caption(f"Projected total minus book line: {c['diff_from_line']:+.2f}")
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
        header = "🎯 Player Props Analysis (Alt Lines)" if is_safe else "🏀 Player Props Analysis"
        st.subheader(header)
        value_props = [c for c in all_props if c["is_value"]]
        no_hist = [c for c in all_props if c.get("no_history")]
        other_props = [c for c in all_props if not c["is_value"] and not c.get("no_history")]

        def _safe_label(c):
            """Display bet as 'Points {N}+' instead of 'OVER 9.5' in safe mode."""
            return f"{c['prop_label']} {c['safe_threshold']}+"

        if value_props:
            st.success(f"**{len(value_props)} prop value bet(s) found!**")

            # Safe-mode bets pass both the confidence and actual-price value
            # filters. Rank them by expected return, not by long payout alone.
            def _safe_sort_key(c):
                if c.get("safe_mode") and c.get("safe_alt_price") is not None:
                    dec = american_to_decimal(c["safe_alt_price"])
                    c["alt_payout"] = dec
                    return c.get("expected_roi_pct", float("-inf"))
                return c.get("expected_roi_pct", c["edge_pct"])

            sorted_props = sorted(value_props, key=_safe_sort_key, reverse=True)
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
                    payout_str = ""
                    if c.get("alt_payout") is not None:
                        payout_str = f"  Pays: ${c['alt_payout']*10:.2f}  |"
                    title = (f"🎯 {c['player']} — {_safe_label(c)}  "
                             f"[{tag}]  Edge: {c['edge_pct']:+.2f}%  |{payout_str}  book line: {c['line']}")
                    with st.expander(title, expanded=(gap >= 0)):
                        cols = st.columns(8)
                        cols[0].metric("Suggested", f"{c['prop_label']} {c['safe_threshold']}+")
                        cols[1].metric(
                            "Model Probability",
                            f"{c['model_hit_at_safe']}%",
                        )
                        cols[2].metric(
                            "Book Implied",
                            f"{c['safe_alt_implied']}%",
                        )
                        cols[3].metric(
                            "Edge",
                            f"{c['edge_pct']:+.2f}%",
                        )
                        cols[4].metric("Expected ROI", f"{c['expected_roi_pct']:+.2f}%")
                        cols[5].metric("Projected Average", c["avg_stat"])
                        cols[6].metric("Weighted History", f"{c['historical_hit_at_safe']}%")
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
                        cols[7].metric(payout_label, p_val, delta=p_delta,
                                       delta_color="off", help=payout_help)
                        st.caption(
                            f"Matchup: {c['matchup']}"
                            f"  |  Projected average: {c['avg_stat']}"
                            f"  |  Model probability at book line: {c['model_hit_at_line']}%"
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
                    implied_prob = (c["over_implied"] if c["direction"] == "OVER"
                                    else c["under_implied"])
                    bet_price = (c.get("over_price") if c["direction"] == "OVER"
                                 else c.get("under_price"))
                    with st.expander(f"🔥 {c['player']} — {c['prop_label']} {c['direction']} {c['line']}  —  Edge: +{c['edge_pct']}%", expanded=True):
                        cols = st.columns(8)
                        cols[0].metric("Line", c["line"])
                        cols[1].metric("Projected Average", c["avg_stat"])
                        cols[2].metric("Model Probability", f"{hit_prob}%")
                        cols[3].metric("Book Implied", f"{implied_prob}%")
                        cols[4].metric("Edge", f"{c['edge_pct']:+.2f}%")
                        cols[5].metric("Expected ROI", f"{c['expected_roi_pct']:+.2f}%")
                        cols[6].metric("Direction", c["direction"])
                        p_val, p_delta = _dk_payout_strs(bet_price)
                        cols[7].metric(
                            f"DK Payout ({c['direction']})", p_val, delta=p_delta, delta_color="off",
                            help=f"Payout for the {c['direction']} at the book line on DraftKings.",
                        )
                        line_gap = c["avg_stat"] - c["line"]
                        st.caption(
                            f"Matchup: {c['matchup']}"
                            f"  |  Book line: {c['line']}"
                            f"  |  Projected average: {c['avg_stat']}"
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
                            "Book Implied": f"{c.get('safe_alt_implied', 0)}%",
                            "Edge": f"{c['edge_pct']:+.2f}%",
                            "Expected ROI": (f"{c['expected_roi_pct']:+.2f}%"
                                             if c.get("expected_roi_pct") is not None else "n/a"),
                        })
                    else:
                        hit_prob = c["over_rate"] if c["direction"] == "OVER" else round(100.0 - c["over_rate"], 2)
                        rows.append({
                            "Player": c["player"],
                            "Prop": c["prop_label"],
                            "Line": c["line"],
                            "Projected Average": c["avg_stat"],
                            "Direction": c["direction"],
                            "Model Probability": f"{hit_prob}%",
                            "Book Implied": (f"{c['over_implied']}%" if c["direction"] == "OVER"
                                             else f"{c['under_implied']}%"),
                            "Edge": f"{c['edge_pct']:+.2f}%",
                            "Expected ROI": (f"{c['expected_roi_pct']:+.2f}%"
                                             if c.get("expected_roi_pct") is not None else "n/a"),
                        })
                st.table(rows)

        if no_hist:
            st.caption(f"ℹ️ {len(no_hist)} prop(s) skipped — no ESPN history found")

    if not any([all_ml, all_spreads, all_totals, all_props]) and not search_filter:
        st.info("No results. Select markets or props and click Analyze.")
