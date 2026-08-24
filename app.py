"""
Sportsbook Value Finder — Streamlit UI
=======================================
Launch with:  streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pricing_common
import weather_factors

# Add script dir to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
# Promote the Azure SQL connection secrets into the environment so the storage
# layer (db_store, imported lazily by recalibration) can build its engine. When
# these are unset the app keeps using the local store unchanged.
# SPORTSBOOK_REQUIRE_SQL is promoted too so the SQL-off boot guard (below) and
# db_store.require_sql() can key on the operator's explicit prod opt-in.
try:
    for _sql_key in ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD",
                     "SPORTSBOOK_REQUIRE_SQL"):
        _sql_val = st.secrets.get(_sql_key)
        if _sql_val:
            os.environ.setdefault(_sql_key, str(_sql_val))
except Exception:
    pass

# P4/P5 MLB→StatsAPI warehouse feature gates. On Streamlit Cloud there are no
# settable OS env vars — only st.secrets — and the gate helpers read os.environ, so
# promote the gate flags here (boot) exactly like the SQL secrets above. Absent →
# unset → the gate stays OFF (its default). Set e.g. ODI_MLB_WAREHOUSE_HIST = "1" in
# the app's Secrets to flip a gate; str() handles a TOML boolean (true → "True").
try:
    for _gate_key in ("ODI_MLB_WAREHOUSE_HIST", "ODI_MLB_WAREHOUSE_TEAM",
                      "ODI_MLB_WAREHOUSE_CALIB", "ODI_MLB_ENFORCE_IDENTITY",
                      "ODI_MLB_ADDITIVE_RUNS", "ODI_MLB_WAREHOUSE_OFFENSE",
                      "ODI_MLB_ADDITIVE_TOTALS", "ODI_MLB_ADDITIVE_ML"):
        _gate_val = st.secrets.get(_gate_key)
        if _gate_val is not None:
            os.environ.setdefault(_gate_key, str(_gate_val))
except Exception:
    pass

# SQL-off hardening (WS1 Layer C): if a SQL deployment is signalled but the SQL
# backend is not reachable, halt at boot instead of silently writing bets and
# predictions to the ephemeral local disk (wiped on Streamlit Cloud restart).
# Keyed on require_sql() → a local dev app with no SQL_* secrets is unaffected.
# Only the predicate is wrapped; st.stop() runs outside so its halt propagates.
try:
    import db_store as _db_boot
    _sql_misconfigured = _db_boot.require_sql() and not _db_boot.enabled()
except Exception:
    _sql_misconfigured = False
if _sql_misconfigured:
    st.error(
        "Durable SQL backend is not reachable, but a SQL deployment is "
        "configured (SPORTSBOOK_REQUIRE_SQL or SQL_* secrets present). "
        "The app is halted to avoid silently writing bets and predictions "
        "to ephemeral local disk that is wiped on restart. Fix the SQL_* "
        "secrets, or set SPORTSBOOK_REQUIRE_SQL=0 for an intentional local run.")
    st.stop()

from odds_client import (
    get_upcoming_events,
    get_event_odds,
    parse_game_odds,
    parse_player_props,
    parse_alt_player_props,
    parse_alt_team_lines,
    build_market_comparisons,
    get_remaining_credits,
    reset_remaining_credits,
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

    # value_gate suppression applies in EVERY mode: a suppressed market must never
    # be recommended, and the safe-mode alt path below re-decides is_value outside
    # the standard props gate, so re-apply the suppress list here.
    from calibration_loader import load_value_gate
    _gate_suppress = set(load_value_gate(sport_key).get("suppress") or [])

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
            c["is_value"] = (edge >= (ar.get("threshold_pct", 5.0) / 100.0)
                             and c["prop"] not in _gate_suppress)
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


def _market_offer_text(comparison, peer=False, market_key="h2h", side=None):
    prefix = "peer_median_" if peer else "primary_"
    price = comparison.get(f"{prefix}price")
    line = comparison.get(f"{prefix}line")
    price_text = f"{price:+g}" if price is not None else "n/a"
    if market_key == "spreads" and line is not None:
        return f"{line:+g} · {price_text}"
    if market_key == "totals" and line is not None:
        return f"{side.upper()} {line:g} · {price_text}"
    return price_text


def _market_comparison_status(comparison):
    line_advantage = comparison.get("line_advantage")
    price_advantage = comparison.get("price_advantage", 0.0)
    if line_advantage is None:
        if price_advantage > 1e-9:
            return "Better DK payout"
        if price_advantage < -1e-9:
            return "Better peer payout"
        return "In line with market"
    if line_advantage > 1e-9:
        if comparison.get("dominates_peer_offer"):
            return "Best of both"
        return "Better line, higher cost"
    if line_advantage < -1e-9:
        if price_advantage > 1e-9:
            return "Worse line, better payout"
        return "Peer median has better line"
    if price_advantage > 1e-9:
        return "Same line, better DK payout"
    if price_advantage < -1e-9:
        return "Same line, better peer payout"
    return "In line with market"


def _market_comparison_summary(
        comparison, market_key="h2h", side=None):
    if not comparison:
        return "—"
    primary = _market_offer_text(
        comparison, market_key=market_key, side=side)
    peer = _market_offer_text(
        comparison, peer=True, market_key=market_key, side=side)
    return (
        f"{_market_comparison_status(comparison)} · "
        f"DK {primary} vs peer {peer}"
    )


def _render_market_comparison(
        comparison, sport_key, market_key="h2h", side=None):
    if not comparison:
        return
    st.markdown("**Market Comparison**")
    cols = st.columns(3)
    cols[0].metric(
        "DraftKings offer",
        _market_offer_text(
            comparison, market_key=market_key, side=side),
    )
    cols[1].metric(
        f"Peer median ({comparison['peer_count']} books)",
        _market_offer_text(
            comparison, peer=True, market_key=market_key, side=side),
    )
    cols[2].metric(
        "Offer context", _market_comparison_status(comparison))
    if (sport_key == "americanfootball_nfl"
            and market_key == "spreads"
            and comparison.get("key_numbers")):
        keys = " and ".join(
            f"{key:g}" for key in comparison["key_numbers"])
        better_side = (
            "DraftKings' better point"
            if comparison.get("line_advantage", 0.0) > 0
            else "The peer median's better point"
        )
        st.caption(f"🏈 {better_side} reaches or crosses NFL key number {keys}.")


def _clear_bet_selections(rendered_keys=None):
    # Reset every bet-selection checkbox. Two Streamlit facts pull in opposite
    # directions:
    #   1. Popping a widget key does NOT reliably uncheck a checkbox that
    #      re-renders on the same run — Streamlit restores it from its retained
    #      widget value, so the box comes back ticked and the bet reappears. A
    #      keyed widget is reset reliably only by WRITING its value False before
    #      it instantiates.
    #   2. But writing False promotes the key to a durable session_state entry
    #      that escapes Streamlit's GC of unrendered keys, so the bet_selection:*
    #      namespace would grow unbounded across sport switches / re-analyses.
    # Resolve both: set the keys that WILL re-render this run (`rendered_keys` —
    # the current slate's selection keys) to False, and pop everything else.
    # Always called from a callback / pre-render point, so no bet_selection
    # widget is instantiated yet this run and both writes are safe. With no
    # rendered_keys (e.g. sport change / new analyze that drops analysis_results)
    # nothing re-renders, so the plain pop-all is correct.
    rendered = set(rendered_keys or ())
    for key in list(st.session_state):
        if not str(key).startswith("bet_selection:"):
            continue
        if key in rendered:
            st.session_state[key] = False
        else:
            st.session_state.pop(key, None)


def _value_bet_checklist_entries(
        all_ml, all_spreads, all_totals, all_props):
    entries = []
    entries.extend(
        make_bet_checklist_entry(candidate, "moneyline")
        for candidate in all_ml if candidate.get("is_value")
    )
    entries.extend(
        make_bet_checklist_entry(candidate, "spread")
        for candidate in all_spreads if candidate.get("is_value")
    )
    for candidate in all_totals:
        if candidate.get("is_over_value"):
            entries.append(make_bet_checklist_entry(
                candidate, "total", side="OVER"))
        if candidate.get("is_under_value"):
            entries.append(make_bet_checklist_entry(
                candidate, "total", side="UNDER"))
    entries.extend(
        make_bet_checklist_entry(candidate, "player_prop")
        for candidate in all_props
        if candidate.get("is_value") and not candidate.get("no_history")
    )
    type_order = {
        "Moneyline": 0,
        "Spread": 1,
        "Game total": 2,
        "Player prop": 3,
    }
    return sorted(entries, key=lambda entry: (
        entry["matchup"], type_order[entry["type"]], entry["bet"]
    ))


def _select_bet_checkbox(candidate, bet_type, side=None):
    entry = make_bet_checklist_entry(candidate, bet_type, side=side)
    st.checkbox(
        "Add to DraftKings bet list",
        key=entry["selection_key"],
        help=(
            "Adds only the bet instruction and matchup/team context to the "
            "consolidated list."
        ),
    )


def _iter_wager_candidates(ar):
    """Yield (selection_key, bet_type, side, candidate) for every value bet in
    `ar`, matching the checklist's selection_key so a checked box maps back to
    the full candidate (with model price/prob) needed to build a wager row."""
    for c in ar.get("all_ml", []):
        if c.get("is_value"):
            key = make_bet_checklist_entry(c, "moneyline")["selection_key"]
            yield key, "moneyline", None, c
    for c in ar.get("all_spreads", []):
        if c.get("is_value"):
            key = make_bet_checklist_entry(c, "spread")["selection_key"]
            yield key, "spread", None, c
    for c in ar.get("all_totals", []):
        if c.get("is_over_value"):
            key = make_bet_checklist_entry(c, "total", side="OVER")["selection_key"]
            yield key, "total", "OVER", c
        if c.get("is_under_value"):
            key = make_bet_checklist_entry(c, "total", side="UNDER")["selection_key"]
            yield key, "total", "UNDER", c
    for c in ar.get("all_props", []):
        if c.get("is_value") and not c.get("no_history"):
            key = make_bet_checklist_entry(c, "player_prop")["selection_key"]
            yield key, "player_prop", None, c


# Fractional-Kelly bet-sizing defaults. The three limiters are editable in the
# submit form and PERSIST across sessions (via bankroll.save/load_kelly_settings);
# the bankroll Kelly scales against is the durable ledger balance
# (bankroll.current_balance()), not a knob. Kelly is very sensitive to probability
# miscalibration, so the shipped default is HALF-Kelly with a per-bet cap and a
# slate-total exposure cap; those caps — not the fraction alone — are the real
# safety margin.
_KELLY_DEFAULTS = {
    "kelly_fraction": 0.5,        # 0-1 multiplier on the full-Kelly fraction
    "kelly_cap_pct": 5.0,         # per-bet hard cap, % of bankroll
    "kelly_slate_cap_pct": 25.0,  # slate-total exposure cap, % of bankroll
}


def _save_kelly_settings():
    """Persist the three Kelly limiters durably (best-effort) so they survive a
    new session, not just a page switch. Runs as an on_change callback after the
    edited value has committed to session_state."""
    try:
        import bankroll as bankroll_mod
        bankroll_mod.save_kelly_settings(
            _kelly_float("kelly_fraction"),
            _kelly_float("kelly_cap_pct"),
            _kelly_float("kelly_slate_cap_pct"))
    except Exception:
        pass


def _kelly_float(key):
    """Read a Kelly sizing knob from session_state, coerced to float.

    Falls back to the shipped default when the key is absent (first render, before
    the form widget below is created) or unparseable. Preserves an explicit 0.0
    (which ``x or default`` would wrongly discard)."""
    v = st.session_state.get(key, _KELLY_DEFAULTS[key])
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(_KELLY_DEFAULTS[key])


def _selected_kelly_rows(ar, valid_keys=None):
    """Build fractional-Kelly-sized ledger rows for the currently-checked value
    bets, plus a {selection_key: stake} map.

    Shared by the submit callback AND the pre-submit preview so the stake shown is
    exactly the stake recorded — both size each leg via ``wagers.build_wager_row``
    in Kelly mode (off the shrunk model prob at the DK price), then apply the
    slate-total exposure cap across the batch. ``valid_keys`` restricts submission
    to the currently-rendered checklist (drops phantom ticks); see
    ``_submit_selected_picks``."""
    import wagers
    import bankroll as bankroll_mod
    bankroll = bankroll_mod.current_balance()  # durable ledger balance, not a widget
    fraction = _kelly_float("kelly_fraction")
    cap_pct = _kelly_float("kelly_cap_pct")
    slate_pct = _kelly_float("kelly_slate_cap_pct")
    events = ar.get("events", {})
    sport_key = ar.get("sport_key")
    placed_at = datetime.now(timezone.utc).isoformat()
    built = []  # (selection_key, row)
    seq = 0
    for sel_key, bet_type, side, candidate in _iter_wager_candidates(ar):
        if not st.session_state.get(sel_key, False):
            continue
        if valid_keys is not None and sel_key not in valid_keys:
            continue
        meta = dict(events.get(candidate.get("event_id"), {}))
        meta.update({
            "sport_key": sport_key,
            "event_id": candidate.get("event_id"),
            "stake": 0.0,  # flat fallback; Kelly overwrites when the leg is sizable
            "placed_at": placed_at,
            "seq": seq,
            "kelly": True,
            "bankroll": bankroll,
            "kelly_fraction": fraction,
            "kelly_cap": cap_pct / 100.0,
        })
        try:
            row = wagers.build_wager_row(bet_type, side, candidate, meta)
        except Exception:
            row = None
        if row:
            built.append((sel_key, row))
            seq += 1
    # Slate-total exposure cap: scale the whole batch down proportionally so the
    # sum of stakes never exceeds slate_pct% of bankroll.
    scaled = pricing_common.scale_to_slate_cap(
        [r.get("stake") for _k, r in built], bankroll, slate_pct / 100.0)
    for (_sel_key, row), stake in zip(built, scaled):
        row["stake"] = stake
    rows = [row for _k, row in built]
    stake_by_key = {sel_key: row.get("stake") for sel_key, row in built}
    return rows, stake_by_key


def _submit_selected_picks(ar, valid_keys=None):
    """Record the checked value bets to the actual-bets ledger at fractional-Kelly
    stakes (sized off the shrunk model prob at the DK price; see
    ``_selected_kelly_rows``). Runs as a (form-)button callback (before widgets
    re-instantiate), so clearing the checkbox keys afterward is safe.

    ``valid_keys`` is the set of selection keys in the CURRENTLY rendered
    checklist; when given, only those are submitted. A checkbox key can linger
    True in session_state after its bet leaves the list (search filter,
    games_sampled drop, or a re-run producing a different slate), and those
    phantom ticks would otherwise be written to the ledger even though the user
    can no longer see or untick them."""
    import wagers
    rows, _stake_by_key = _selected_kelly_rows(ar, valid_keys)
    if not rows:
        st.session_state["_submit_picks_msg"] = (
            "warning", "No selected bets to submit.")
        return
    # Surface storage failures instead of silently reporting "Submitted 0", and
    # keep the selections checked so the user can retry without re-picking.
    try:
        added = wagers.submit_wagers(rows)
    except Exception as exc:
        st.session_state["_submit_picks_msg"] = (
            "error",
            f"Couldn't save your {len(rows)} pick(s) to the ledger: {exc}. "
            "Your selections were kept — try again.",
        )
        return
    if not added:
        st.session_state["_submit_picks_msg"] = (
            "error",
            f"Storage reported 0 of {len(rows)} pick(s) written. Your "
            "selections were kept — try again.",
        )
        return
    # Uncheck the just-submitted boxes reliably: the same slate re-renders this
    # run, so pass its keys to be written False (not merely popped).
    _clear_bet_selections(valid_keys)
    st.session_state["_submit_picks_msg"] = (
        "success",
        f"Submitted {added} pick(s) to your bet ledger — track ROI on "
        "the 🧾 My Bets page.",
    )


# Auto-pick ranking-metric labels (UI) → bet_selector metric keys.
_AUTO_PICK_METRICS = {
    "Expected value (EV)": "ev",
    "Edge %": "edge",
    "Win probability": "prob",
    "Balanced (EV + win prob)": "balanced",
}


def _auto_pick_top_bets(ar):
    """Fill the checklist with the top-N rule-compliant value bets (EV-ranked by
    default). Runs as a button callback — before the checkbox widgets re-instantiate
    — so clearing then setting bet_selection:* keys is safe. REPLACES the current
    selection (not auto-submitted): the user still adds/removes bets and sets the
    stake before hitting Submit."""
    import bet_selector
    try:
        n = int(st.session_state.get("auto_pick_count", 5) or 5)
    except (TypeError, ValueError):
        n = 5
    n = max(1, n)
    metric = _AUTO_PICK_METRICS.get(
        st.session_state.get("auto_pick_metric"), "ev")
    sport_key = ar.get("sport_key")
    # Build the pool from the same source the Submit path uses, but drop spread /
    # player-prop candidates with < 5 games sampled so every auto-pick has a real
    # rendered checkbox (mirrors the results filter at the render site below).
    pool = []
    for sel_key, bet_type, side, cand in _iter_wager_candidates(ar):
        if (bet_type in ("spread", "player_prop")
                and (cand.get("games_sampled") or 0) < 5):
            continue
        pool.append((sel_key, bet_type, side, cand))
    chosen = bet_selector.select_top_bets(pool, sport_key, n, metric=metric)
    # Reset every on-screen box to False (reliable uncheck of the prior group),
    # then tick the chosen ones — order matters so chosen end up True.
    _clear_bet_selections({key for key, *_ in _iter_wager_candidates(ar)})
    for key in chosen:
        st.session_state[key] = True
    if chosen:
        st.session_state["_submit_picks_msg"] = (
            "success",
            f"Replaced your selections with the top {len(chosen)} value bet(s) "
            "— adjust below, then Submit.",
        )
    else:
        st.session_state["_submit_picks_msg"] = (
            "warning", "No value bets available to auto-pick.")


def _render_selected_bet_checklist(entries, ar):
    msg = st.session_state.pop("_submit_picks_msg", None)
    selected = [
        entry for entry in entries
        if st.session_state.get(entry["selection_key"], False)
    ]
    with st.container(border=True):
        st.subheader("🧾 Selected DraftKings Bets")
        if msg:
            getattr(st, msg[0], st.info)(msg[1])
        # Auto-pick row — rendered ABOVE the "nothing selected yet" guard so it's
        # available before any box is ticked (the whole section only renders when
        # value bets exist). Selects the top-N EV-ranked, rule-compliant value
        # bets into the checklist; not auto-submitted.
        auto_n = int(st.session_state.get("auto_pick_count", 5) or 5)
        count_col, metric_col, pick_col = st.columns([1, 1.6, 1.4])
        with count_col:
            st.number_input(
                "How many", min_value=1, max_value=25, value=5, step=1,
                key="auto_pick_count",
                help="Number of top value bets to auto-select.",
            )
        with metric_col:
            st.selectbox(
                "Rank by", list(_AUTO_PICK_METRICS.keys()),
                key="auto_pick_metric",
                help="Metric used to rank value bets before applying the parlay "
                     "anti-correlation rules and MLB slate rules.",
            )
        with pick_col:
            st.button(
                f"🎯 Auto-pick top {auto_n}", key="auto_pick_btn",
                width="stretch",
                help="Replace your current selection with the top-ranked, "
                     "rule-compliant value bets (not auto-submitted).",
                on_click=_auto_pick_top_bets, args=(ar,))
        if not selected:
            st.caption(
                "Select “Add to DraftKings bet list” on any value bet to build "
                "a simple checklist here."
            )
            return
        # Only the bets in the CURRENT checklist are submittable — the phantom-
        # tick guard in _submit_selected_picks drops any stale/filtered-out ticks.
        valid_keys = {entry["selection_key"] for entry in entries}
        # Preview the fractional-Kelly stakes for the checked bets: exactly the
        # rows the Submit callback will persist (same sizing + slate-total cap),
        # so the size is visible before betting.
        _rows, stake_by_key = _selected_kelly_rows(ar, valid_keys)
        import bankroll as bankroll_mod
        bankroll = bankroll_mod.current_balance()  # durable ledger balance

        def _fmt_stake(s):
            return f"${s:,.2f}" if isinstance(s, (int, float)) else "—"

        st.caption(f"{len(selected)} selected bet(s)")
        st.dataframe(
            [{
                "Bet type": entry["type"],
                "Bet to place": entry["bet"],
                "Matchup": entry["matchup"],
                "Team": entry["team"],
                "Stake $": _fmt_stake(stake_by_key.get(entry["selection_key"])),
            } for entry in selected],
            hide_index=True,
            width="stretch",
        )
        total_stake = sum(
            v for v in stake_by_key.values() if isinstance(v, (int, float)))
        if bankroll > 0:
            st.caption(
                f"Total staked ${total_stake:,.2f} of ${bankroll:,.2f} bankroll "
                f"({total_stake / bankroll * 100:.0f}% exposure) · fractional "
                "Kelly, capped per-bet and per-slate."
            )
        else:
            st.caption(
                "Set your bankroll on the 🧾 My Bets page to size these bets "
                "(stakes stay $0.00 until the bankroll ledger is above $0)."
            )
        # Submit the checked bets to the actual-bets ledger at the previewed
        # fractional-Kelly stakes, executed at the DK price at submit (tracks REAL
        # ROI). The sizing inputs live OUTSIDE a form so an edit commits to
        # session_state immediately and the preview above updates on the next
        # rerun — a form would defer the write and desync the preview from what is
        # recorded. All buttons use on_click callbacks (fire before widgets
        # re-instantiate), so clearing the checkbox keys stays safe.
        bank_col, submit_col = st.columns([1, 1])
        with bank_col:
            st.metric("Bankroll", f"${bankroll:,.2f}")
            st.caption("Durable ledger balance — manage it on 🧾 My Bets.")
        with submit_col:
            st.button(
                "✅ Submit Picks", key="submit_picks_btn", width="stretch",
                help="Record the checked bets to your actual-bets ledger at the "
                     "previewed Kelly stakes.",
                on_click=_submit_selected_picks, args=(ar, valid_keys))
        with st.expander("⚙️ Sizing settings (Kelly)"):
            st.number_input(
                "Kelly fraction", min_value=0.0, max_value=1.0,
                value=_KELLY_DEFAULTS["kelly_fraction"], step=0.05,
                format="%.2f", key="kelly_fraction", on_change=_save_kelly_settings,
                help="Multiplier on the full-Kelly bet fraction. 0.5 = half-Kelly "
                     "(recommended); lower is more conservative.",
            )
            st.number_input(
                "Per-bet cap (% of bankroll)", min_value=0.0, max_value=100.0,
                value=_KELLY_DEFAULTS["kelly_cap_pct"], step=0.5,
                format="%.1f", key="kelly_cap_pct", on_change=_save_kelly_settings,
                help="Hard ceiling on any single bet, as a percent of bankroll.",
            )
            st.number_input(
                "Slate-total cap (% of bankroll)", min_value=0.0, max_value=100.0,
                value=_KELLY_DEFAULTS["kelly_slate_cap_pct"], step=1.0,
                format="%.1f", key="kelly_slate_cap_pct",
                on_change=_save_kelly_settings,
                help="Ceiling on total exposure across all submitted bets; the "
                     "batch is scaled down proportionally if it would exceed this.",
            )
            st.caption("Sizing settings are saved across sessions.")
        st.button("Clear selected bets", key="clear_selected_bets",
                  width="stretch", on_click=_clear_bet_selections,
                  args=(valid_keys,))


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
    mlb_warehouse_team_stats,
    mlb_warehouse_team_defense,
    mlb_warehouse_gate_status,
)
from props import prop_fetch_limit
from analysis import (
    analyze_moneyline_value,
    analyze_spreads_value,
    analyze_totals_value,
    analyze_player_props_value,
    generate_parlays,
    make_bet_checklist_entry,
)

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
# config.json holds the API key and is git-ignored (see .gitignore). Only the
# placeholder template is committed, so a fresh clone falls back to it and lands
# on the setup screen instead of crashing.
CONFIG_EXAMPLE_PATH = os.path.join(SCRIPT_DIR, "config.json.example")

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


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_calibration_blobs(sport_keys):
    """Committed calibration/<sport>.json files (change only on redeploy → a new
    process clears the cache). Cached so the Model Guide doesn't re-read + parse
    them on every rerun."""
    blobs = {}
    for sport_key in sport_keys:
        path = os.path.join(SCRIPT_DIR, "calibration", f"{sport_key}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                blobs[sport_key] = json.load(f)
        except (OSError, json.JSONDecodeError):
            blobs[sport_key] = {}
    return blobs


@st.cache_data(ttl=60, show_spinner=False)
def _cached_prediction_summary():
    """Forward prediction-log summary (a full log read). Short TTL so the Model
    Guide reflects newly-graded outcomes without a blob GET on every rerun."""
    from recalibration import prediction_performance_summary
    return prediction_performance_summary()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_market_prediction_summary():
    """Team-market forward-log summary (moneyline/spread/total). Short TTL so the
    Model Guide reflects newly-graded outcomes without a read on every rerun."""
    from recalibration import market_prediction_performance_summary
    return market_prediction_performance_summary()


@st.cache_data(ttl=60, show_spinner=False)
def _cached_pickrules_summaries():
    """Per-sport pick-rules ROI lens: the recommended SLATE (bet_selector rules
    re-derived over each day's logged value picks) vs the raw is_value POOL,
    graded on real outcomes (flat 1u/pick). Reads both forward logs once and
    computes every sport with resolved value picks. Short TTL so newly-graded
    outcomes surface without a read on every rerun."""
    import pickrules_roi
    import recalibration
    preds = recalibration.read_prediction_log()
    markets = recalibration.read_market_prediction_log()
    out = {}
    for sport_key in recalibration.SPORT_ESPN_MAP:
        result = pickrules_roi.slate_vs_pool(preds, markets, sport_key)
        if result["combined"]["pool"]["total"]:
            out[sport_key] = result
    return out


@st.cache_data(show_spinner=False)
def load_config():
    # Cached: config.json + secrets + env are stable within a session, so this
    # avoids a file read + JSON parse on every rerun. save_api_key() clears it.
    # Neither file is guaranteed in a hosted deploy (config.json is git-ignored;
    # config.json.example may be absent) — the key comes from st.secrets there, so
    # degrade to an empty config + setup screen instead of crashing at import.
    path = (CONFIG_PATH if os.path.exists(CONFIG_PATH)
            else CONFIG_EXAMPLE_PATH if os.path.exists(CONFIG_EXAMPLE_PATH)
            else None)
    config = {}
    if path:
        try:
            with open(path, "r") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            config = {}
    # Prefer the deployment secret / env var over the local file so the key
    # never has to live on disk in a hosted environment.
    try:
        secret_api_key = st.secrets.get("ODDS_API_KEY")
        if secret_api_key:
            config["odds_api_key"] = str(secret_api_key)
    except Exception:
        pass
    if not config.get("odds_api_key"):
        config["odds_api_key"] = os.environ.get("ODDS_API_KEY", "")
    return config


def save_api_key(key):
    """Save the API key to the local, git-ignored config.json.

    config.json is git-ignored, so the key stays on this machine. In a hosted
    deployment (e.g. Streamlit Cloud) this local file is ephemeral — set
    ODDS_API_KEY in Streamlit secrets there instead.
    """
    if os.path.exists(CONFIG_PATH):
        config = load_config()
    elif os.path.exists(CONFIG_EXAMPLE_PATH):
        with open(CONFIG_EXAMPLE_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {}
    config["odds_api_key"] = key
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    load_config.clear()  # invalidate the cached config so the new key is read


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
        help=("Saved to a local, git-ignored config.json on this machine. "
              "For a hosted deployment, set ODDS_API_KEY in Streamlit secrets "
              "instead — the local file is not durable there."),
    )

    if st.button("💾 Save API Key", type="primary", width='stretch'):
        if new_key and len(new_key) > 10 and new_key != "YOUR_API_KEY_HERE":
            save_api_key(new_key)
            reset_remaining_credits()  # start the new key with an unknown (ungated) balance
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
            - This app **caches results for 60 minutes (1 hour)** to avoid wasting credits
            - Example: analyzing 3 games with moneyline + spreads = **6 credits**
            """
        )


@st.cache_data(ttl=3600)
def fetch_events(api_key, sport_key):
    return get_upcoming_events(api_key, sport_key)


@st.cache_data(ttl=3600)
def fetch_espn_teams(espn_sport, espn_league):
    return get_all_teams(espn_sport, espn_league)


def _resolve_team_dim(sport_key, name, espn_teams):
    """Team dict {id, display_name, record, wins, losses, win_pct, ...} for a
    matchup team. MLB (P6 teardown): resolve off the StatsAPI warehouse
    (warehouse_find_team) so team markets + the schedule/venue bridge no longer need
    the ESPN teams dict; ESPN find_team is the transition fallback (no MLB game is
    silently dropped if the warehouse can't resolve a team) and the sole path for
    NBA/NFL/NHL. warehouse_find_team returns id = the MLBAM team_id, which matches
    the warehouse player history's team_id — so a warehouse-keyed schedule resolves
    in props with no change there."""
    if sport_key == "baseball_mlb":
        try:
            import mlb_warehouse
            wh = mlb_warehouse.warehouse_find_team(name)
            if wh:
                return wh
        except Exception:
            pass
    return find_team(espn_teams, name)


def _fetch_team_schedule(sport_key, team, espn_sport, espn_league):
    """Recent team games for the layoff/recent-form bridge. MLB: from the StatsAPI
    warehouse (get_team_games, by display name — no ESPN); other sports: ESPN
    get_team_schedule. Returns a list of per-game dicts (each with a 'date')."""
    if sport_key == "baseball_mlb":
        try:
            import mlb_warehouse
            return mlb_warehouse.get_team_games(team["display_name"]) or []
        except Exception:
            return []
    return get_team_schedule(espn_sport, espn_league, team["id"])


def format_time(commence_time):
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        eastern = dt.astimezone(ZoneInfo("America/New_York"))
        return eastern.strftime("%b %d  %I:%M %p ET")
    except (ValueError, AttributeError):
        return commence_time[:19] if commence_time else "TBD"


def _event_not_started(event, now=None):
    """True when an event's start time is unknown or still in the future.

    Used to prune already-started/finished games from a restored (stale) events
    slate so the empty-refetch fallback can't re-offer a game that has begun.
    Unknown/unparseable start times are kept rather than over-filtered.
    """
    commence = event.get("commence_time")
    if not commence:
        return True
    try:
        start = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        if now is None:
            now = datetime.now(timezone.utc)
        return start > now
    except (ValueError, AttributeError, TypeError):
        return True


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
            - **Price break-even probability** — the win rate required to break even at the offered American price, including vig.
            - **Edge** — `model probability − price break-even probability`, measured in percentage points.
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

    sport_names = {
        "baseball_mlb": "MLB",
        "basketball_nba": "NBA",
        "americanfootball_nfl": "NFL",
    }
    calibration_blobs = _cached_calibration_blobs(tuple(sport_names))

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
                    # Provenance of the Brier/accuracy numbers above: props re-fit
                    # on genuine book lines vs. still carrying the offline synthetic
                    # season-average sweep. The fallback labels pre-fit_basis
                    # entries correctly (real_line_fit present => real book lines).
                    "Fit basis": (
                        "Real book lines"
                        if cfg.get("fit_basis") == "real_line"
                        or cfg.get("real_line_fit")
                        else "Synthetic sweep"
                    ),
                    "Full fit observations": cfg.get("n_obs", "—"),
                    "Calibration date": fit_date,
                })
        if prop_rows:
            st.dataframe(prop_rows, hide_index=True, width="stretch")
        else:
            st.info("No committed player-prop calibration summaries were found.")

        st.subheader("Forward prediction tracking")
        from recalibration import prediction_log_storage
        forward = _cached_prediction_summary()
        storage_backend = prediction_log_storage()
        if storage_backend == "Local cache":
            st.warning(
                "Forward data is using local container storage, which can reset "
                "when Streamlit Cloud restarts or redeploys. Configure the Azure "
                "SQL secrets (SQL_SERVER/DATABASE/USER/PASSWORD) for shared "
                "durable storage."
            )
        else:
            st.caption(f"Prediction log storage: {storage_backend} (shared and durable).")

        # MLB→StatsAPI warehouse gates. Predictions don't record which source
        # served them, so this is the only at-a-glance way to confirm a flag flip
        # actually took effect on the running app (default OFF = ESPN path).
        try:
            _wh = mlb_warehouse_gate_status()
            _on = [name for name, key in (
                ("player-history", "history"),
                ("team-markets", "team"),
                ("calibration", "calib"),
                ("identity-enforce", "enforce_identity"),
            ) if _wh.get(key)]
            if _wh.get("sql") and _on:
                st.caption("MLB warehouse (StatsAPI) active: " + ", ".join(_on) + ".")
            elif _wh.get("sql"):
                st.caption("MLB warehouse gates all OFF — MLB data served from ESPN.")
            else:
                st.caption("MLB warehouse gates require SQL (not enabled) — serving from ESPN.")
        except Exception:
            pass

        # Manual full-drain of the prediction backlog. The automatic loop stays
        # small (80/cycle, hourly) so the app stays responsive; this grades
        # everything on demand when you fall behind. It blocks while ESPN/statsapi
        # box scores are fetched (up to a minute or two for a large backlog; uses
        # no odds credits), so it's opt-in. Today's unfinished games stay pending.
        if st.button(
            "⚖️ Resolve all pending predictions now",
            help="Grade every past-game forecast against final box scores. The "
                 "automatic loop only resolves 80/hour to stay fast — use this to "
                 "catch up a backlog. May take a minute; no odds credits used.",
        ):
            from recalibration import maintain_sport
            resolved_total = 0
            market_total = 0
            with st.status("Resolving pending predictions…", expanded=True) as _drain:
                for _sk, _sname in sport_names.items():
                    try:
                        _res = maintain_sport(_sk, max_resolve=5000)
                    except Exception as _exc:  # best-effort; report + continue
                        _drain.write(f"⚠️ {_sname}: {type(_exc).__name__}")
                        continue
                    _n = _res.get("newly_resolved", 0)
                    _nm = _res.get("newly_resolved_markets", 0)
                    resolved_total += _n
                    market_total += _nm
                    if _n or _nm:
                        _parts = []
                        if _n:
                            _parts.append(f"{_n} prop")
                        if _nm:
                            _parts.append(f"{_nm} team-market")
                        _drain.write(f"✓ {_sname}: resolved {', '.join(_parts)}")
                _drain.update(
                    label=(f"Resolved {resolved_total} prop + {market_total} "
                           "team-market prediction(s)."),
                    state="complete")
            _cached_prediction_summary.clear()
            _cached_market_prediction_summary.clear()
            _cached_pickrules_summaries.clear()
            st.rerun()

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
                "price; older rows remain usable for hit-rate and Brier scoring."
            )
        else:
            st.info(
                "No forward predictions have been logged yet. Predictions will "
                "appear here after player-prop analyses are run and past games "
                "are resolved."
            )

        st.subheader("Team-market forward tracking")
        market_forward = _cached_market_prediction_summary()
        market_type_labels = {
            "moneyline": "Moneyline",
            "spread": "Spread",
            "total": "Total",
        }
        if market_forward["total"]:
            mf_cols = st.columns(4)
            mf_cols[0].metric("Logged predictions", market_forward["total"])
            mf_cols[1].metric("Resolved", market_forward["resolved"])
            _mhr = market_forward.get("hit_rate")
            mf_cols[2].metric(
                "Pick hit rate",
                f"{_mhr * 100:.1f}%" if _mhr is not None else "Not enough data",
            )
            _mroi = market_forward.get("roi")
            mf_cols[3].metric(
                "Pick ROI",
                f"{_mroi * 100:+.1f}%"
                if _mroi is not None else "Awaiting priced results",
            )
            market_forward_rows = []
            for row in market_forward["by_market"]:
                _rhr = row.get("hit_rate")
                _rbrier = row.get("brier")
                _rroi = row.get("roi")
                market_forward_rows.append({
                    "Sport": sport_names.get(row["sport_key"], row["sport_key"]),
                    "Market": market_type_labels.get(
                        row["bet_type"],
                        (row["bet_type"] or "Unknown").title()),
                    "Logged": row["total"],
                    "Resolved": row["resolved"],
                    "Pending": row["pending"],
                    "Pushes": row["pushes"],
                    "Pick hit rate": (
                        f"{_rhr * 100:.1f}%" if _rhr is not None else "—"),
                    "Brier": (f"{_rbrier:.4f}" if _rbrier is not None else "—"),
                    "Pick ROI": (
                        f"{_rroi * 100:+.1f}%" if _rroi is not None else "—"),
                    "Priced results": row["priced_resolved"],
                })
            st.dataframe(market_forward_rows, hide_index=True, width="stretch")
            st.caption(
                "The model's favored side per game and market (moneyline/spread/"
                "total), logged on every analysis and graded against final "
                "scores — independent of whether you placed the bet. Brier scores "
                "the picked side's probability; ROI stakes each pick at its "
                "logged price."
            )
        else:
            st.info(
                "No team-market predictions have been logged yet. They appear "
                "here after moneyline/spread/total analyses are run and past "
                "games are resolved."
            )

        st.subheader("Pick-rules ROI (recommended slate vs value pool)")
        pickrules = _cached_pickrules_summaries()
        if pickrules:
            pr_rows = []
            for _sk, _res in pickrules.items():
                for _label, _summ in (("Value pool", _res["combined"]["pool"]),
                                      ("Rule slate", _res["combined"]["slate"])):
                    _hr, _roi = _summ.get("hit_rate"), _summ.get("roi")
                    pr_rows.append({
                        "Sport": sport_names.get(_sk, _sk),
                        "Set": _label,
                        "Picks": _summ["total"],
                        "Resolved": _summ["resolved"],
                        "Priced": _summ["priced_resolved"],
                        "Hit rate": f"{_hr * 100:.1f}%" if _hr is not None else "—",
                        "ROI": f"{_roi * 100:+.1f}%" if _roi is not None else "—",
                    })
            st.dataframe(pr_rows, hide_index=True, width="stretch")
            for _sk, _res in pickrules.items():
                _d = _res["delta"]["roi"]
                _sroi = _res["combined"]["slate"].get("roi")
                _proi = _res["combined"]["pool"].get("roi")
                if _d is not None:
                    st.caption(
                        f"**{sport_names.get(_sk, _sk)}**: rule slate ROI "
                        f"{_sroi * 100:+.1f}% vs pool {_proi * 100:+.1f}% "
                        f"(Δ {_d * 100:+.1f}pp); the pick rules dropped "
                        f"{_res['n_dropped']} pick(s).")
            if any(_res["rules_skipped"] for _res in pickrules.values()):
                st.caption(
                    "⚠️ Team-based rules (Rule-of-3, opposing-team, batting-order) "
                    "aren't fully replayable on rows logged before team/batting-"
                    "order capture; they replay in full as new predictions accrue."
                )
            st.caption(
                "Grades the RECOMMENDED SLATE — the pick rules (is_value, "
                "under-0.5 suppression, Rule-of-3, batting-order, L1/L2/L3, ER/K) "
                "re-derived over each ET game-date's logged value picks — against "
                "the raw is_value POOL, on real outcomes at the logged price "
                "(flat 1 unit/pick). The delta is what the pick rules bought. "
                "Distinct from the backtest's edge-vs-close ROI and the actual-bet "
                "ledger; neither set grades both sides of a line."
            )
        else:
            st.info(
                "No resolved value picks yet to grade the pick rules against. "
                "This fills in as player-prop / team-market analyses are run and "
                "past games are resolved."
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
            meta = blob.get("meta") or {}
            source = (meta.get("prob_shrink") or {}).get("source")
            holdout = meta.get("prob_shrink_holdout") or {}
            for market_key, market_label in market_labels.items():
                shrink_value = shrink.get(market_key)
                hold = holdout.get(market_key) or {}
                cal_brier = hold.get("brier")
                raw_brier = hold.get("raw_brier")
                n_hold = hold.get("n")
                if isinstance(cal_brier, (int, float)):
                    holdout_display = f"Brier {cal_brier:.4f}"
                    if isinstance(raw_brier, (int, float)):
                        holdout_display += f" (raw {raw_brier:.4f})"
                    if n_hold:
                        holdout_display += f", n={n_hold}"
                else:
                    holdout_display = "Not exported"
                team_rows.append({
                    "Sport": sport_name,
                    "Prediction": market_label,
                    "Backtest calibration": source or "No committed odds-fit metadata",
                    "Probability shrink": (
                        f"{shrink_value:.3f}"
                        if isinstance(shrink_value, (int, float)) else "Not fitted"
                    ),
                    "Holdout Brier": holdout_display,
                })
        st.dataframe(team_rows, hide_index=True, width="stretch")
        st.caption(
            "Holdout Brier is the scored backtest calibration error on held-out "
            "games (lower is better; raw = before probability shrink). Markets "
            "not yet backtested with `odds backtest --engine live` show ‘Not "
            "exported’. This is the offline calibration report — live model-side "
            "accuracy is in Team-market forward tracking above."
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
            7. Recommend the bet only when `model probability − actual alt-price break-even probability ≥ 5 percentage points`.

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
        mlb_blob = calibration_blobs.get("baseball_mlb", {})
        mlb_meta = mlb_blob.get("meta") or {}
        expected_runs = mlb_blob.get("expected_runs_challenger") or {}
        if expected_runs:
            expected_holdout = expected_runs.get("holdout") or {}
            st.subheader("Expected-runs / Pythagorean challenger")
            st.dataframe([
                {
                    "Gate": "Moneyline Brier",
                    **(expected_holdout.get("moneyline_brier") or {}),
                },
                {
                    "Gate": "Moneyline accuracy",
                    **(expected_holdout.get("moneyline_accuracy") or {}),
                },
                {
                    "Gate": "Margin RMSE",
                    **(expected_holdout.get("margin_rmse") or {}),
                },
                {
                    "Gate": "Home -1.5 Brier",
                    **(expected_holdout.get("home_minus_1_5_brier") or {}),
                },
            ], hide_index=True, width="stretch")
            validation_summary = expected_runs.get("validation_summary") or []
            if validation_summary:
                st.caption("Chronological base-model validation by fit history")
                st.dataframe(
                    validation_summary, hide_index=True, width="stretch")
            extension_validation = (
                expected_runs.get("extension_validation") or [])
            if extension_validation:
                st.caption("Incremental feature and score-distribution gates")
                st.dataframe(
                    extension_validation, hide_index=True, width="stretch")
            final_validation = (
                expected_runs.get("final_2025_validation") or {})
            final_candidates = final_validation.get("candidate_summary") or []
            if final_candidates:
                st.caption("Untouched 2025 final holdout")
                st.dataframe(
                    final_candidates, hide_index=True, width="stretch")
            if (expected_runs.get("enabled")
                    and (expected_runs.get("live_markets") or {}).get("spreads")):
                st.success(
                    "The expected-runs ensemble is active for **MLB spreads "
                    "and displayed margin only** when its live inputs are "
                    "complete. " + expected_runs.get("decision", ""))
            else:
                st.warning(
                    "The 1.83 Pythagorean challenger remains **disabled**. "
                    + expected_runs.get(
                        "decision", "More chronological validation is required.")
                )
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
# How long a grading pass stays "fresh" before My Bets auto-re-grades on open.
# Replaces a permanent per-session "already graded" boolean, which never re-ran
# for games that settled AFTER the app started — leaving the user to click
# Refresh. A short TTL means opening My Bets picks up newly-final games itself.
_GRADE_STALE_SECONDS = 300


def _apply_bankroll_adjustment():
    """Record a manual bankroll adjustment from the typed target (button
    callback, fires before widgets re-instantiate). Writes a single transaction
    for the SIGNED difference target - current, so the ledger balance becomes
    exactly what the user entered. Best-effort; leaves a flash for the rerun."""
    import bankroll
    try:
        target = float(st.session_state.get("bankroll_adjust_target"))
    except (TypeError, ValueError):
        return
    delta = bankroll.record_adjustment(target)
    new_balance = bankroll.current_balance()
    # Re-sync the input to the new balance so it reads correctly next render.
    st.session_state["bankroll_adjust_target"] = new_balance
    if abs(delta) < 0.005:
        st.session_state["_bankroll_flash"] = (
            "info", f"Bankroll already at ${new_balance:,.2f} — no change.")
    else:
        verb = "Deposit" if delta > 0 else "Withdrawal"
        st.session_state["_bankroll_flash"] = (
            "success",
            f"{verb} of ${abs(delta):,.2f} recorded — bankroll now "
            f"${new_balance:,.2f}.")


def _render_bankroll_section():
    """💰 Bankroll ledger — realized bet P/L plus manual adjustments. The balance
    is the SUM of every transaction (never stored), so it can't drift when a bet
    is re-graded; fractional-Kelly stakes on the Value Finder size against it."""
    import bankroll
    bsummary = bankroll.summary()
    with st.container(border=True):
        st.subheader("💰 Bankroll")
        bflash = st.session_state.pop("_bankroll_flash", None)
        if bflash:
            getattr(st, bflash[0], st.info)(bflash[1])
        bcols = st.columns(3)
        bcols[0].metric("Current bankroll", f"${bsummary['balance']:,.2f}")
        bcols[1].metric("Realized bet P/L", f"${bsummary['bets_total']:+,.2f}")
        bcols[2].metric("Manual adjustments",
                        f"${bsummary['adjustments_total']:+,.2f}")
        st.caption(
            "Balance = realized P/L from settled bets + your manual adjustments. "
            "Fractional-Kelly stakes on the 🎯 Value Finder size against it."
        )
        with st.expander("Adjust bankroll (deposit / withdrawal / correction)"):
            st.caption(
                "Enter your **real** current bankroll. The app writes a single "
                "adjustment for the difference so the ledger balance becomes "
                "exactly what you enter — e.g. ledger $700, you withdraw $200 and "
                "enter $500 → it records −$200; add it back and enter $700 → +$200."
            )
            # Prefill the user's real bankroll with the current ledger balance,
            # but never below the input's floor: before a starting bankroll is
            # set, settled losses can leave the derived balance negative, and a
            # value under min_value crashes the widget.
            st.number_input(
                "Current bankroll ($)", min_value=0.0,
                value=max(0.0, float(bsummary["balance"])), step=0.01,
                format="%.2f", key="bankroll_adjust_target",
            )
            st.button(
                "Update bankroll", key="bankroll_adjust_btn",
                on_click=_apply_bankroll_adjustment,
                help="Record the difference between this value and the current "
                     "ledger balance as an adjustment transaction.")
        txns = bsummary["txns"]
        if txns:
            with st.expander(f"Bankroll history ({len(txns)} transaction(s))"):
                # Running balance is cumulative oldest→newest; display newest first.
                running = 0.0
                with_balance = []
                for t in reversed(txns):            # oldest first
                    running = round(running + bankroll.txn_amount(t), 2)
                    with_balance.append((t, running))
                with_balance.reverse()               # newest first for the table
                st.dataframe([
                    {
                        "When": (t.get("created_at") or "")[:19].replace("T", " "),
                        "Type": (t.get("txn_type") or "").title(),
                        "Amount": f"${bankroll.txn_amount(t):+,.2f}",
                        "Balance": f"${bal:,.2f}",
                        "Note": t.get("note") or "",
                    } for t, bal in with_balance
                ], hide_index=True, width="stretch")


def render_my_bets():
    """Actual-bets ledger — realized ROI on the picks the user really placed."""
    import wagers
    st.title("🧾 My Bets — Actual ROI")
    # A save (edit/delete) reruns to refresh the tables; carry the confirmation
    # across that rerun so it isn't lost.
    flash = st.session_state.pop("_wagers_flash", None)
    if flash:
        st.success(flash)
    st.caption(
        "Realized return on the bets you actually submitted (fractional-Kelly "
        "stake off your bankroll, executed at the DK price at submit). Bets "
        "auto-grade from free box scores and their P/L maintains the bankroll "
        "ledger below; closing-line value fills in as the odds warehouse "
        "accumulates."
    )

    if wagers.storage_backend() == "Local cache":
        st.warning(
            "Your bet ledger is stored locally and resets when a hosted app "
            "(e.g. Streamlit Cloud) restarts or redeploys. Set the Azure SQL "
            "secrets (SQL_SERVER/DATABASE/USER/PASSWORD) for durable storage."
        )

    refresh = st.button("🔄 Refresh results",
                        help="Grade any newly settled bets now.")
    try:
        graded_at = st.session_state.get("_wagers_graded_at", 0.0)
        stale = (time.time() - graded_at) > _GRADE_STALE_SECONDS
        if refresh or stale:
            with st.spinner("Grading settled bets..."):
                graded = wagers.resolve_pending_wagers()
            st.session_state["_wagers_graded_at"] = time.time()
            if graded:
                st.success(f"Graded {graded} newly settled bet(s).")
    except Exception:
        pass

    rows, read_error = wagers.read_wagers_with_status()
    if read_error is not None:
        # Distinguish an unreachable durable store from an empty ledger so a
        # transient outage never reads as "you have no bets".
        st.warning(
            "⚠️ Couldn't reach your bet ledger right now "
            f"({type(read_error).__name__}). Your bets are safe — this is a read "
            "timeout, not data loss. On Azure SQL serverless this usually means "
            "the database is **resuming from auto-pause** (wait a few seconds and "
            "click **🔄 Refresh results**), or the **monthly free compute is used "
            "up** (it resets on the 1st of the month)."
        )
        return

    # Keep the bankroll ledger in step with the CURRENT settled-wager P/L on every
    # open/rerun (covers grade, re-grade, edit, and delete via one idempotent
    # sweep; passes the rows already read to avoid a second fetch).
    try:
        import bankroll
        bankroll.reconcile_bet_txns(rows)
    except Exception:
        pass

    # 💰 Bankroll — the money header of the money page, rendered BEFORE the
    # "no bets yet" guard so a new user can set their starting bankroll before
    # placing a single bet (fractional-Kelly on the Value Finder sizes against it).
    _render_bankroll_section()

    if not rows:
        st.info(
            "No submitted picks yet. On the 🎯 Value Finder page, check "
            "“Add to DraftKings bet list” on the bets you place, then click "
            "**Submit Picks**."
        )
        return

    # CLV is read straight from the durable ledger; the app never fills it. The
    # old render-time warehouse fill (attach_clv/persist_clv) was retired — it
    # compared DK executed prices against a best-of-book / de-vigged CONSENSUS
    # close the warehouse can't tie to DraftKings, and it hit the warehouse (a
    # manifest + snapshot GET per un-priced row) on every rerun. CLV now comes
    # exclusively from the DK closing-line backfill CLI (backfill_dk_clv.py),
    # which reads DK's price at the EXACT line you bet.

    sport_labels = {info["key"]: name for name, info in SPORTS.items()}

    def _bet_label(r):
        bt = r.get("bet_type")
        if bt == "player_prop":
            return (f"{r.get('player')} — {r.get('prop_label')} "
                    f"{r.get('direction')} {r.get('line')}")
        if bt == "total":
            return f"{r.get('direction')} {r.get('line')}"
        if bt == "spread":
            point = r.get("point")
            return (f"{r.get('team')} {point:+g}" if isinstance(point, (int, float))
                    else f"{r.get('team')} spread")
        return f"{r.get('team')} ML"

    summary = wagers.summarize_wagers(rows)
    cols = st.columns(5)
    cols[0].metric("Bets placed", summary["total"])
    cols[1].metric("Settled", summary["resolved"])
    roi = summary.get("roi")
    cols[2].metric("Realized ROI",
                   f"{roi * 100:+.1f}%" if roi is not None else "—")
    cols[3].metric("Realized P/L", f"${summary['realized_profit']:+.2f}")
    hit = summary.get("hit_rate")
    cols[4].metric("Hit rate",
                   f"{hit * 100:.1f}%" if hit is not None else "—")
    st.caption(
        f"Record: {summary['won']}–{summary['lost']}–{summary['push']} "
        f"(W–L–P)  ·  Total staked: ${summary['total_staked']:.2f}  ·  "
        f"Pending: {summary['pending']} (${summary['pending_stake']:.2f})"
    )

    if summary["by_bet_type"]:
        st.subheader("By bet type")
        st.dataframe([
            {
                "Bet type": b.get("label") or (b["bet_type"] or "—").title(),
                "Pending": b["pending"],
                "Settled": b["resolved"],
                "W–L–P": f"{b['won']}–{b['lost']}–{b['push']}",
                "Staked": f"${b['total_staked']:.2f}",
                "P/L": f"${b['realized_profit']:+.2f}",
                "ROI": (f"{b['roi'] * 100:+.1f}%" if b["roi"] is not None else "—"),
            } for b in summary["by_bet_type"]
        ], hide_index=True, width="stretch")

    settled = [r for r in rows if r.get("status") in ("won", "lost", "push")]
    pending = [r for r in rows if r.get("status") == "pending"]

    # Editor keys carry a nonce that bumps after each Save/Apply so the tables
    # rebuild fresh (otherwise ticked Delete/Re-grade boxes persist across the
    # post-save rerun).
    editor_nonce = st.session_state.get("_wagers_editor_nonce", 0)

    if pending:
        st.subheader("Pending bets")
        st.caption(
            "Correct **Price**, **Line**, or **Stake** if a number changed "
            "between running the analysis and placing the bet, or tick "
            "**Delete** to remove an accidental bet. Then click **Save "
            "changes**."
        )
        pending_df = pd.DataFrame([
            {
                "wager_id": r.get("wager_id"),
                "Delete": False,
                # ET, not raw UTC: a bet placed after ~8pm ET has a UTC date one
                # day ahead, which would display tomorrow's date on tonight's bet.
                "Placed": pricing_common.et_local_date(r.get("placed_at")) or "",
                "Sport": sport_labels.get(r.get("sport_key"), r.get("sport_key")),
                "Bet": _bet_label(r),
                "Matchup": r.get("matchup"),
                "Game date": r.get("game_date"),
                "Price": r.get("executed_price"),
                "Line": r.get("line"),
                "Stake": _safe_float(r.get("stake")),
            } for r in sorted(pending, key=lambda r: r.get("game_date") or "")
        ]).set_index("wager_id")
        # A form batches all edits: cell changes and Delete ticks do NOT rerun
        # the app until "Save changes" is pressed, so editing no longer dims/
        # reruns the page on every keystroke.
        with st.form(f"pending_form_{editor_nonce}", clear_on_submit=False):
            edited_pending = st.data_editor(
                pending_df,
                hide_index=True,
                width="stretch",
                key=f"pending_editor_{editor_nonce}",
                disabled=["Placed", "Sport", "Bet", "Matchup", "Game date"],
                column_config={
                    "Delete": st.column_config.CheckboxColumn(
                        "Delete", default=False, help="Remove this bet on Save"),
                    "Price": st.column_config.NumberColumn(
                        "Price", help="American odds you actually got", step=1,
                        format="%d"),
                    "Line": st.column_config.NumberColumn(
                        "Line", help="The line you placed at", step=0.5,
                        format="%.1f"),
                    "Stake": st.column_config.NumberColumn(
                        "Stake", help="Dollar stake", min_value=0.0, step=0.01,
                        format="$%.2f"),
                },
            )
            submit_pending = st.form_submit_button("💾 Save changes")
        if submit_pending:
            _apply_wager_edits(pending_df, edited_pending, editable=True)

    if settled:
        st.subheader("Settled bets")
        st.caption(
            "Settled bets can't be edited. Tick **Re-grade** to reset a bet to "
            "pending and re-grade it on refresh (fixes bets graded while the "
            "game was still live), or tick **Delete** to remove one. Then click "
            "**Apply**."
        )
        settled_df = pd.DataFrame([
            {
                "wager_id": r.get("wager_id"),
                "Delete": False,
                "Re-grade": False,
                # ET, not raw UTC (see the pending table): an evening-ET placement
                # otherwise shows the next calendar day in the settled ledger.
                "Placed": pricing_common.et_local_date(r.get("placed_at")) or "",
                "Sport": sport_labels.get(r.get("sport_key"), r.get("sport_key")),
                "Bet": _bet_label(r),
                "Matchup": r.get("matchup"),
                "Stake": f"${_safe_float(r.get('stake')):.2f}",
                "Price": r.get("executed_price"),
                "Result": (r.get("status") or "").upper(),
                "P/L": f"${_safe_float(r.get('profit')):+.2f}",
                "CLV": (f"{r.get('clv_pct'):+.1f}%"
                        if r.get("clv_pct") is not None else "—"),
            } for r in sorted(settled, key=lambda r: r.get("placed_at") or "",
                              reverse=True)
        ]).set_index("wager_id")
        with st.form(f"settled_form_{editor_nonce}", clear_on_submit=False):
            edited_settled = st.data_editor(
                settled_df,
                hide_index=True,
                width="stretch",
                key=f"settled_editor_{editor_nonce}",
                disabled=["Placed", "Sport", "Bet", "Matchup", "Stake", "Price",
                          "Result", "P/L", "CLV"],
                column_config={
                    "Delete": st.column_config.CheckboxColumn(
                        "Delete", default=False, help="Remove this bet on Apply"),
                    "Re-grade": st.column_config.CheckboxColumn(
                        "Re-grade", default=False,
                        help="Reset to pending and re-grade on the next refresh."),
                },
            )
            submit_settled = st.form_submit_button("💾 Apply")
        if submit_settled:
            _apply_wager_edits(settled_df, edited_settled, regradable=True)


def _wager_ids(df):
    """Wager-id index values from a bets dataframe (drops any blank add-rows)."""
    return {i for i in df.index.tolist() if isinstance(i, str) and i}


def _coerce_int(value):
    try:
        if value is None or pd.isna(value):
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_wager_edits(original_df, edited_df, editable=False, regradable=False):
    """Read an edited bets table and persist the requested changes.

    Rows with the **Delete** box ticked are removed by wager_id. When
    ``editable``, changed Price/Line/Stake cells (pending only) go through
    wagers.update_wagers. When ``regradable``, rows with the **Re-grade** box
    ticked are reset to pending via wagers.regrade_wagers so the next refresh
    re-grades them. Surfaces storage failures instead of silently dropping
    them."""
    import wagers
    ids = _wager_ids(edited_df)

    def _checked(wid, col):
        return col in edited_df.columns and bool(edited_df.loc[wid].get(col))

    deleted = sorted(wid for wid in ids if _checked(wid, "Delete"))
    deleted_set = set(deleted)
    survivors = [wid for wid in ids if wid not in deleted_set]

    edits = {}
    if editable:
        field_map = (("Price", "executed_price", _coerce_int),
                     ("Line", "line", _coerce_float),
                     ("Stake", "stake", _coerce_float))
        for wid in survivors:
            before = original_df.loc[wid]
            after = edited_df.loc[wid]
            patch = {}
            for col, field, coerce in field_map:
                new = coerce(after.get(col))
                if new is not None and new != coerce(before.get(col)):
                    patch[field] = new
            if patch:
                edits[wid] = patch

    regrades = []
    if regradable:
        regrades = [wid for wid in survivors if _checked(wid, "Re-grade")]

    if not deleted and not edits and not regrades:
        st.info("No changes to save.")
        return
    try:
        n_del = wagers.delete_wagers(deleted) if deleted else 0
        n_edit = wagers.update_wagers(edits) if edits else 0
        n_regrade = wagers.regrade_wagers(regrades) if regrades else 0
    except Exception as exc:
        st.error(f"Could not save changes ({type(exc).__name__}). "
                 "Your ledger is unchanged — please try again.")
        return
    if not n_del and not n_edit and not n_regrade:
        st.info("No changes to save.")
        return
    parts = ([f"{n_edit} edited"] if n_edit else []) + \
            ([f"{n_regrade} reset to pending"] if n_regrade else []) + \
            ([f"{n_del} deleted"] if n_del else [])
    st.session_state["_wagers_flash"] = "Saved: " + ", ".join(parts) + "."
    # A re-grade must trigger a fresh grading pass on the rerun below.
    if n_regrade:
        st.session_state["_wagers_graded_at"] = 0.0
    # Bump the editor nonce so both tables rebuild with fresh keys — clears the
    # ticked Delete/Re-grade boxes and reflects the new ledger — then rerun so
    # the summary metrics and tables recompute.
    st.session_state["_wagers_editor_nonce"] = \
        st.session_state.get("_wagers_editor_nonce", 0) + 1
    st.rerun()


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


st.set_page_config(page_title="Sportsbook Value Finder", page_icon="🎯", layout="wide")

# ──────────────────────────────────────────────────────────
#  First-run setup check
# ──────────────────────────────────────────────────────────
config = load_config()

# Preserve Value Finder widget state across page switches. Streamlit clears a
# widget's key from session_state when that widget is NOT rendered on a run —
# which happens whenever the Model Guide or My Bets page is showing. Without this,
# returning to Value Finder resets the Sport selectbox to its default, which fires
# on_sport_change and wipes the whole analysis (results + game selection + bet
# ticks). Re-touching these keys here (before any of them is created) keeps them,
# so navigating away and back preserves the analysis. Genuine user sport changes
# still fire on_sport_change normally. `selected_games` is deliberately excluded:
# its options are the live slate (games drop off), and re-injecting stale labels
# risks an options-mismatch — the analysis itself survives via analysis_results,
# which is a plain key Streamlit never garbage-collects.
for _persist_key in list(st.session_state.keys()):
    if (_persist_key in ("sport", "markets", "props", "result_filter",
                         "kelly_fraction", "kelly_cap_pct",
                         "kelly_slate_cap_pct", "auto_pick_count",
                         "auto_pick_metric")
            or str(_persist_key).startswith("bet_selection:")):
        st.session_state[_persist_key] = st.session_state[_persist_key]

# Load the durable Kelly limiters once per session (they persist across sessions
# in the app_settings KV store, not just across page switches). setdefault runs
# BEFORE the submit-form widgets are created, so each number_input adopts the
# persisted value instead of the shipped default. Best-effort.
if not st.session_state.get("_kelly_settings_loaded"):
    st.session_state["_kelly_settings_loaded"] = True
    try:
        import bankroll as _bankroll
        for _k, _v in _bankroll.load_kelly_settings().items():
            st.session_state.setdefault(_k, _v)
    except Exception:
        pass

# Prefetch the My Bets ledger once per session, BEFORE the page router, so the
# 🧾 My Bets page opens ready (it then just reads the already-graded durable
# ledger; CLV is filled offline by backfill_dk_clv.py, never at render time).
# Runs above the st.stop() router + setup gate so it fires on whichever page
# loads first, and needs no odds API key (grading uses free box scores). Results
# persist to the blob, so this is a one-time cost per session. A separate
# sentinel from the grade timestamp so the post-save regrade reset doesn't
# re-trigger this block; stamping _wagers_graded_at=now lets My Bets skip its own
# pass until the grade goes stale (_GRADE_STALE_SECONDS).
if not st.session_state.get("_wagers_prefetched"):
    st.session_state["_wagers_prefetched"] = True
    try:
        import wagers as _wagers
        with st.status("Updating your bet ledger…", expanded=False) as _status:
            _graded = _wagers.resolve_pending_wagers()
            st.session_state["_wagers_graded_at"] = time.time()
            # Sync realized bet P/L into the bankroll ledger (idempotent).
            try:
                import bankroll as _bankroll
                _bankroll.reconcile_bet_txns()
            except Exception:
                pass
            _status.update(
                label=(f"Updated: {_graded} bet(s) graded." if _graded
                       else "Bet ledger up to date."),
                state="complete")
    except Exception:
        # Best-effort: My Bets lazily grades on open as the backstop.
        pass

# "Time to refit" nudge: once per session, count RESOLVED MLB predictions not yet
# consumed by an offline calibration refit (refit_performed=0, now trackable since
# SQL rows are stable). When enough have accrued, surface a banner pointing at the
# offline refit command. Cheap COUNT, cached per session; best-effort (never
# breaks a page). The count resets after `refit_calibration ... --real-lines`
# flags those rows.
if "_pending_refit_count" not in st.session_state:
    try:
        import recalibration as _recal
        st.session_state["_pending_refit_count"] = _recal.count_pending_refit(
            "baseball_mlb")
        st.session_state["_pending_refit_threshold"] = \
            _recal.MIN_NEW_FOR_OFFLINE_REFIT
    except Exception:
        st.session_state["_pending_refit_count"] = 0
        st.session_state["_pending_refit_threshold"] = None
_pending_refit = st.session_state.get("_pending_refit_count", 0)
_refit_threshold = st.session_state.get("_pending_refit_threshold")
if _refit_threshold and _pending_refit >= _refit_threshold:
    st.info(
        f"**{_pending_refit} new resolved predictions** have accumulated since "
        f"the last MLB calibration refit — enough to re-tune the model. When "
        f"convenient, run the offline refit (free; uses ESPN + the durable "
        f"store):\n\n"
        f"```\npython refit_calibration.py --sport mlb --real-lines\n```",
        icon="📊")

with st.sidebar:
    app_page = st.radio(
        "Navigate",
        ["🎯 Value Finder", "📘 Model Guide & Performance", "🧾 My Bets"],
        key="app_page",
    )

if app_page == "📘 Model Guide & Performance":
    render_model_guide()
    st.stop()

if app_page == "🧾 My Bets":
    render_my_bets()
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

    # Durability notice (P1.8): the forward-tracking prediction log is only
    # durable when Azure SQL is configured. Without it the log lives in
    # ephemeral container storage that a hosted deploy wipes on restart —
    # silently resetting resolved outcomes and recalibration. Surface it during
    # normal use, not only on the Model Guide page.
    try:
        from recalibration import prediction_log_storage as _pls
        if _pls() == "Local cache":
            st.warning(
                "Forward-tracking data is stored locally and resets when a "
                "hosted app (e.g. Streamlit Cloud) restarts or redeploys. Set "
                "the Azure SQL secrets (SQL_SERVER/DATABASE/USER/PASSWORD) for "
                "durable storage."
            )
    except Exception:
        pass

    # Reset game selection when sport changes
    def on_sport_change():
        st.session_state["selected_games"] = []
        st.session_state["markets"] = []
        st.session_state["props"] = []
        _clear_bet_selections()
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("parlay_results", None)
        st.session_state.pop("parlay_mode", None)
        st.session_state.pop("result_filter", None)
        st.session_state.pop("_last_events", None)

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
    # Recent-games window for TEAM-market form is hardcoded per sport. Per-prop
    # calibration decides whether games inside the window are equally or
    # exponentially weighted.
    recent_n = sport.get("recent_n_default", 10)
    # Player-prop history FETCH: a full-season superset (props.py slices per prop to
    # its own recent_n; STEP-1 sweep found hits/K want the full season). Kept
    # SEPARATE from the team-market recent_n above so a longer prop window never
    # changes team-stat aggregation.
    prop_fetch_n = prop_fetch_limit(sport["key"])
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
                save_api_key(new_key)          # writes config.json + load_config.clear()
                reset_remaining_credits()      # lift the exhausted-credit gate for the new key
                fetch_events.clear()           # drop events cached under the old key
                st.success("Key updated!")
                st.rerun()
            else:
                st.error("Invalid key")

# ──────────────────────────────────────────────────────────
#  Main — Game Selection
# ──────────────────────────────────────────────────────────
st.subheader(f"📅 Upcoming {sport_label} Games")

with st.spinner("Loading events (free)..."):
    events, events_error = [], None
    try:
        events = fetch_events(api_key, sport["key"])
    except Exception as e:
        events_error = e

# A background rerun (notably the forward-capture timer, which only arms after an
# analysis has logged a prediction) re-runs this block on its own. If the live
# "upcoming" list has momentarily gone empty — a just-analyzed game drops off the
# API's list once it starts — or the fetch fails transiently, the st.stop() gates
# here (and the "no games selected" gate below) would collapse the page and
# discard a completed, on-screen analysis. Guard against that:
#   * stash the last non-empty slate per sport, timestamped;
#   * on an empty/failed refetch, fall back to a *fresh* stash minus games that
#     have already started (never re-offer a begun game or a finished slate);
#   * never st.stop() while an analysis is on screen — show the notice inline and
#     let the stored results keep rendering.
_STALE_EVENTS_MAX_AGE = 20 * 60  # seconds: long enough to ride out a rerun storm
                                 # near tip-off, short enough not to surface an
                                 # already-finished slate as "upcoming".
_now = datetime.now(timezone.utc)
# Stash the raw non-empty slate per sport (timestamped) for the fallback below.
if events:
    st.session_state["_last_events"] = {
        "sport": sport["key"], "events": events, "ts": _now}

_using_stale_slate = False
if not events:
    _stash = st.session_state.get("_last_events")
    _fresh = (_stash and _stash.get("sport") == sport["key"] and _stash.get("ts")
              and (_now - _stash["ts"]).total_seconds() < _STALE_EVENTS_MAX_AGE)
    if _fresh:
        events = _stash.get("events", [])
        _using_stale_slate = True

# Never offer a game that has already started/finished, regardless of source —
# a live fetch (the Odds API events endpoint includes in-play games), the 1-hour
# fetch cache (which can span a tip-off), or the stale-slate fallback. Pricing a
# begun game against pre-game history is wrong and pollutes forward-tracking.
events = [e for e in events if _event_not_started(e, _now)]

_has_results = "analysis_results" in st.session_state
if _using_stale_slate and events:
    st.caption(
        "⚠️ Showing the last known games — the live schedule is momentarily "
        "unavailable; start times may be slightly stale."
    )

if not events:
    # Only hard-stop on a genuine cold start. If a completed analysis is already
    # on screen, keep it — a transient empty/failed refetch must not wipe it, and
    # the misleading "no games" banner must not sit above the preserved results.
    if not _has_results:
        if events_error is not None:
            st.error(f"Failed to fetch events: {events_error}")
        else:
            st.warning("No upcoming games found.")
        st.stop()
    st.caption(
        "ℹ️ The live game list is momentarily unavailable — showing your most "
        "recent analysis below."
    )

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
    # Don't prompt-and-halt over a completed analysis: a background rerun can
    # empty the selection when a started game drops off the slate. Only show the
    # "select games" prompt (and stop) on a genuine cold start; otherwise keep
    # the stored results rendering below without a contradictory banner.
    if "analysis_results" not in st.session_state:
        st.info("👆 Select one or more games above to begin analysis")
        st.stop()

# Calculate actual credit cost (accounting for cached data)
bookmakers_list = config.get("bookmakers", [])
# bookmakers_str (DraftKings) still scopes the safe-mode ALT-line fetch; standard
# props + team markets now fetch all U.S. books for line-shopping (P1.1b).
bookmakers_str = ",".join(bookmakers_list) if bookmakers_list else ""
actual_cost = 0
for gl in selected_game_labels:
    ev = game_options[gl]
    eid = ev["id"]
    # Team markets fetch all U.S. books in one same-cost request so the UI can
    # show peer context. Only DraftKings is passed into the model below.
    if markets_str and not is_event_cached(
            sport["key"], eid, markets=markets_str, bookmakers=None):
        actual_cost += len(market_keys)
    if selected_props:
        prop_markets_str = ",".join(selected_props)
        # Props now fetch all U.S. books (bookmakers=None) for line-shopping;
        # keep the cache-hit check aligned or credits are mis-estimated / the
        # cache key won't match the fetch below (P1.1b).
        if not is_event_cached(sport["key"], eid, markets=prop_markets_str, bookmakers=None):
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
if analyze_clicked and selected_game_labels:
    # Defer the selection reset to the results block (after the new slate's
    # entries are known, before the checkboxes render) so any re-rendered boxes
    # are written False — reliable even on a same-slate re-analyze, where pop()
    # alone would let Streamlit restore the old ticks.
    st.session_state["_reset_bet_selections"] = True
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
                    cols[2].metric("Price Break-even", f"{p['combined_implied_prob']}%")
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
                    cols[2].metric("Price Break-even", f"{p['combined_implied_prob']}%")
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
                    cols[2].metric("Price Break-even", f"{p['combined_implied_prob']}%")
                    cols[3].metric(
                        "Hit Prob (no correlation)",
                        f"{p['combined_hist_prob_indep']}%",
                        help="Naive product of each leg's probability, assuming legs are independent. Compare against Hit Probability to see how much correlation between legs helps or hurts.",
                    )
                    cols[4].metric(
                        "Parlay Edge",
                        f"{pe:+.2f}%" if pe is not None else "n/a",
                        help="Joint hit probability minus the parlay price's break-even probability — the model's expected edge over the book on the full parlay.",
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

if analyze_clicked and selected_game_labels:
    # Pre-flight credit-budget guard (P1.7): don't silently spend into (or past)
    # a reserve floor. `remaining` is None on a cold process (no live call has
    # reported the balance yet), so we can only guard once it's known. Nothing
    # has been spent at this point — the paid fan-out is below.
    _reserve = int(config.get("credit_reserve", 0) or 0)
    if (remaining is not None and actual_cost > 0
            and actual_cost > remaining - _reserve):
        st.error(
            f"⛔ This analysis needs about {actual_cost} credit(s) but only "
            f"{remaining} remain"
            + (f" (holding {_reserve} in reserve)" if _reserve else "")
            + ". Nothing was spent. Deselect some games or markets, wait for "
            "your monthly reset, or lower `credit_reserve` in config.json."
        )
        st.stop()

    progress = st.progress(0, text="Starting analysis...")

    # Team data. MLB (P6): team resolution is warehouse-only (parity-verified 30/30,
    # full-slate 9/9), so skip the ESPN get_all_teams fetch entirely — the last
    # ungated ESPN team call for MLB. NBA/NFL/NHL still fetch from ESPN.
    progress.progress(5, text="Loading team data...")
    if sport["key"] == "baseball_mlb":
        espn_teams = {}
    else:
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
        weather_futures = {}

        for event in selected_events:
            eid = event["id"]
            home = event["home_team"]
            away = event["away_team"]

            # Game odds (moneyline/spreads/totals)
            if markets_str:
                odds_futures[eid] = pool.submit(
                    get_event_odds, api_key, sport["key"], eid,
                    markets=markets_str, bookmakers=None
                )

            # Player prop odds — fetch ALL U.S. books (same credit cost as a
            # single-book request) so props can line-shop the best price and
            # de-vig a real multi-book consensus (P1.1b). DraftKings is carved
            # out inside parse_player_props for staking/display.
            if selected_props:
                prop_markets_str = ",".join(selected_props)
                prop_odds_futures[eid] = pool.submit(
                    get_event_odds, api_key, sport["key"], eid, markets=prop_markets_str, bookmakers=None
                )

            # Team schedules (for both teams). MLB (P6): resolve off the warehouse
            # (transition fallback to ESPN) and source the schedule from the
            # warehouse — no ESPN get_team_schedule fan-out for baseball. Keyed by
            # the resolved team id (MLBAM for the warehouse path).
            home_espn = _resolve_team_dim(sport["key"], home, espn_teams)
            away_espn = _resolve_team_dim(sport["key"], away, espn_teams)
            for team in (home_espn, away_espn):
                if team and team["id"] not in team_schedule_futures:
                    team_schedule_futures[team["id"]] = pool.submit(
                        _fetch_team_schedule, sport["key"], team,
                        sport["espn_sport"], sport["espn_league"])

            # Pre-game weather forecast (MLB only — park geo is MLB). Feeds the
            # weather/wind projection nudge on batter_hits / pitcher_earned_runs;
            # fails open to neutral inside get_game_weather (never raises).
            if sport["key"] == "baseball_mlb":
                weather_futures[eid] = pool.submit(
                    weather_factors.get_game_weather,
                    home, event.get("commence_time"))

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

        # Stale-odds warning (P1.5): get_event_odds serves bounded-age cached
        # odds when credits are exhausted / rate-limited, flagged `_stale_cache`.
        # Surface it so edges aren't trusted against outdated lines.
        stale_events = {
            eid for eid, data in
            list(odds_results.items()) + list(prop_odds_results.items())
            if isinstance(data, dict) and data.get("_stale_cache")
        }
        if stale_events:
            warnings.append(
                f"⚠️ Odds API credits appear exhausted or rate-limited — "
                f"{len(stale_events)} event(s) were priced from STALE cached "
                "odds (up to a few hours old). Verify current prices before "
                "betting; edges for those games may be against lines that have "
                "since moved."
            )

        schedule_results = {}
        for tid, fut in team_schedule_futures.items():
            try:
                schedule_results[tid] = fut.result()
            except Exception:
                schedule_results[tid] = []

        # Per-event weather (MLB); get_game_weather already swallows errors, so a
        # miss just yields a neutral dict the analyzer treats as no adjustment.
        event_weather = {}
        for eid, fut in weather_futures.items():
            try:
                event_weather[eid] = fut.result()
            except Exception:
                event_weather[eid] = None

        # Build a per-team avg-points-allowed lookup so the player-prop analyzer
        # can apply an opponent-defense weighting to historical games. P4 flip:
        # prefer the StatsAPI /standings-derived defense when enabled (MLB-only,
        # fail-open to the ESPN schedule scan).
        team_defense = (mlb_warehouse_team_defense(sport["espn_sport"])
                        or build_team_defense_lookup(schedule_results, espn_teams))

        progress.progress(50, text="Getting player props...")

        # ── Phase 2: Parse prop data and fire off ALL player history lookups ──
        prop_history_futures = {}  # history key -> future
        parsed_props = {}  # eid -> parsed prop data
        events_by_id = {e["id"]: e for e in selected_events}

        # P6/F3: pre-warm the StatsAPI season player index on the MAIN thread so the
        # Phase-2 pool workers (resolve_mlbam_id -> _player_index) hit the populated
        # module/disk cache instead of each racing a full-roster fetch + cache write
        # (the no-network-racing-file-cache invariant).
        if sport["key"] == "baseball_mlb":
            try:
                import mlb_starters
                import mlb_warehouse
                mlb_starters.warm_player_index(mlb_warehouse._current_season())
            except Exception:
                pass

        for eid, raw_data in prop_odds_results.items():
            parsed = parse_player_props(raw_data)
            parsed_props[eid] = parsed
            # ESPN team ids for THIS matchup disambiguate same-name players so
            # the correct athlete's history is fetched (see search_athlete
            # team_ids). NON-MLB sports dedup the future GLOBALLY by (player, prop):
            # the first event to reference a name wins its hint — a namesake in a
            # different game on the same slate is a documented residual. MLB keys the
            # future per-event (below), so that residual does not apply to baseball.
            # ESPN team ids disambiguate same-name players for the ESPN history
            # path (search_athlete). MLB (P6) serves history from the warehouse by
            # globally-unique NAME, not team ids — and a warehouse MLBAM team id
            # would mis-hint ESPN search_athlete — so pass no hint for baseball.
            mlb_lineup = None
            mlb_probables = None
            if sport["key"] == "baseball_mlb":
                event_team_ids = []
                # P6 game-context-first identity: today's posted lineup + announced
                # probables give the AUTHORITATIVE, trade-aware, namesake-safe
                # name→MLBAM id for the two teams playing (mlb_starters.resolve_mlbam_id
                # inside the warehouse-history path). Fetched here so it is in scope at
                # history-resolution time (Phase 3's per-event fetch is too late); both
                # are date-global cached, so Phase 3's re-fetch is a cache hit.
                import mlb_starters
                _ev = events_by_id.get(eid)
                _gd = (_ev or {}).get("commence_time", "")[:10]
                try:
                    _c = datetime.fromisoformat(
                        _ev["commence_time"].replace("Z", "+00:00"))
                    _gd = _c.astimezone(
                        ZoneInfo("America/New_York")).date().isoformat()
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
                try:
                    mlb_lineup = mlb_starters.get_confirmed_lineup(
                        parsed.get("home_team"), parsed.get("away_team"), _gd)
                except Exception:
                    mlb_lineup = None
                try:
                    mlb_probables = mlb_starters.get_probable_starters(_gd)
                except Exception:
                    mlb_probables = None
            else:
                event_teams = [find_team(espn_teams, tn)
                               for tn in (parsed.get("home_team"),
                                          parsed.get("away_team")) if tn]
                event_team_ids = [str(t["id"]) for t in event_teams
                                  if t and t.get("id")]
            for prop_key, players in parsed.get("props", {}).items():
                for player_name in players:
                    # MLB serves history from the warehouse by resolved MLBAM id and a
                    # player appears in exactly one game/day, so keying the future
                    # per-event loses no dedup while preventing a same-name batter in
                    # ANOTHER game on this slate from reusing this game's lineup-
                    # resolved id (F4). Other sports keep the global (player, prop) key.
                    key = ((eid, player_name, prop_key)
                           if sport["key"] == "baseball_mlb"
                           else (player_name, prop_key))
                    if key not in prop_history_futures:
                        prop_history_futures[key] = pool.submit(
                            get_player_stat_history,
                            sport["espn_sport"], sport["espn_league"],
                            player_name, prop_key, prop_fetch_n,
                            team_ids=event_team_ids,
                            # MLB: the matchup teams narrow a namesake to its MLBAM
                            # id in the warehouse (Max Muncy / Luis Garcia Jr.) so it
                            # resolves off the warehouse instead of falling to ESPN.
                            teams=[parsed.get("home_team"),
                                   parsed.get("away_team")],
                            # MLB game-context-first identity (None for other sports).
                            confirmed_lineup=mlb_lineup,
                            probable_starters=mlb_probables,
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
        probable_starters = None
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
            # §2.5A: announced probable starters (available days ahead) gate
            # pitcher props all day; confirmed_lineup gates batter props once
            # posted. Both feed mlb_starters.player_start_status below.
            try:
                probable_starters = mlb_starters.get_probable_starters(game_date)
            except Exception as e:
                warnings.append(f"Probable starters unavailable for {away} @ {home}: {e}")
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
            raw_game_odds = odds_results[eid]
            market_comparisons = build_market_comparisons(raw_game_odds)
            # Team-market analysis runs on a DraftKings-ONLY view of the odds, so
            # every executable price the analyzers surface (moneyline best_price,
            # spread/total consensus price) is DK's price — the team-market
            # equivalent of the props' P1.1b "stake at DK" rule (props instead
            # parse all books and carry dk_over/under_price separately). This is
            # what makes a submitted team bet's executed_price DraftKings, hence
            # its CLV a true DK-vs-DK comparison against the DK close backfilled
            # by backfill_dk_clv.py. Do NOT widen this to all books without giving
            # the team path its own dk_price field, or team CLV silently reverts
            # to a mixed DK-close-vs-best-of-book comparison.
            draftkings_game_odds = dict(raw_game_odds)
            draftkings_game_odds["bookmakers"] = [
                book for book in raw_game_odds.get("bookmakers", [])
                if book.get("key") == "draftkings"
            ]
            game_odds = parse_game_odds(draftkings_game_odds)
            home_espn = _resolve_team_dim(sport["key"], home, espn_teams)
            away_espn = _resolve_team_dim(sport["key"], away, espn_teams)

            if home_espn and away_espn:
                # P4 team-market flip: prefer the StatsAPI warehouse for both teams
                # (MLB-only, env-gated). Require BOTH so the two sides are never
                # mixed warehouse/ESPN; any miss → the ESPN build for both.
                wh_home = mlb_warehouse_team_stats(sport["espn_sport"], home, recent_n)
                wh_away = mlb_warehouse_team_stats(sport["espn_sport"], away, recent_n)
                if wh_home and wh_away:
                    home_stats, away_stats = wh_home, wh_away
                else:
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

                def _tag_event(cands, market_key):
                    for c in cands:
                        c["event_id"] = eid
                        if market_key == "totals":
                            c["market_comparisons"] = (
                                market_comparisons[market_key])
                        else:
                            comparison = market_comparisons[market_key].get(
                                c.get("team"))
                            if comparison:
                                c["market_comparison"] = comparison
                    return cands

                if "h2h" in market_keys:
                    all_ml.extend(_tag_event(analyze_moneyline_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"], matchup_features=matchup_features), "h2h"))
                if "spreads" in market_keys:
                    all_spreads.extend(_tag_event(analyze_spreads_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"], matchup_features=matchup_features), "spreads"))
                if "totals" in market_keys:
                    all_totals.extend(_tag_event(analyze_totals_value(game_odds, home_stats, away_stats, threshold, sport_key=sport["key"], matchup_features=matchup_features), "totals"))

        # Player props analysis
        if eid in parsed_props:
            prop_data = parsed_props[eid]
            player_histories = {}
            for prop_key, players in prop_data.get("props", {}).items():
                for player_name in players:
                    if player_name not in player_histories:
                        player_histories[player_name] = {}
                    # Mirror the Phase-2 key: MLB futures are per-event (F4).
                    _hk = ((eid, player_name, prop_key)
                           if sport["key"] == "baseball_mlb"
                           else (player_name, prop_key))
                    history = dict(prop_history_results.get(
                        _hk,
                        {"player": player_name, "found": False, "values": []},
                    ))
                    if confirmed_lineup:
                        lineup_context = mlb_starters.lineup_player_context(
                            confirmed_lineup, player_name)
                        if lineup_context:
                            history["batting_order"] = lineup_context["batting_order"]
                    # §2.5A tri-state availability. Batter props gate on the
                    # confirmed lineup; pitcher props on the announced probable.
                    # Fails open to "unknown" — props.py only acts on "out".
                    if sport["key"] == "baseball_mlb":
                        history["lineup_status"] = mlb_starters.player_start_status(
                            prop_key, player_name, home, away,
                            confirmed_lineup or {}, probable_starters or {},
                            int(game_date[:4]))
                    player_histories[player_name][prop_key] = history
            new_props = analyze_player_props_value(prop_data, player_histories, threshold,
                                                  sport_key=sport["key"],
                                                  team_defense=team_defense,
                                                  espn_teams=espn_teams,
                                                  safe_mode=safe_mode,
                                                  safe_target=safe_target,
                                                  team_schedules=schedule_results,
                                                  matchup_features=matchup_features,
                                                  weather=event_weather.get(eid),
                                                  # Commit C: game-context identity
                                                  # stamp (role-partitioned).
                                                  confirmed_lineup=confirmed_lineup,
                                                  probable_starters=probable_starters)
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
        # Per-event metadata so "Submit Picks" can build ledger rows (commence,
        # game_date, home/away) without a live re-fetch. game_date is the
        # US-Eastern local date (a late US game's UTC date is one day ahead).
        "events": {
            e["id"]: {
                "commence_time": e.get("commence_time"),
                "game_date": pricing_common.et_local_date(e.get("commence_time")),
                "home_team": e.get("home_team"),
                "away_team": e.get("away_team"),
            }
            for e in selected_events
        },
    }
    # Forward-track the model's team-market picks (moneyline/spread/total),
    # mirroring per-prop forward logging. Best-effort: no-ops gracefully until the
    # market_prediction_log table exists, and never breaks analysis.
    try:
        import recalibration as _recal_mkt
        _ar_mkt = st.session_state["analysis_results"]
        _recal_mkt.log_market_prediction_rows(
            _recal_mkt.build_market_prediction_rows(_ar_mkt, _ar_mkt["sport_key"]))
    except Exception:
        pass
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

    # Fold the fetched odds snapshots into the durable warehouse (roadmap 0.4).
    # Runs after both fetch waves; a no-op when nothing new was captured.
    try:
        import warehouse
        warehouse.flush()
    except Exception:
        pass

# ──────────────────────────────────────────────────────────
#  Display Results (from session state, persists across reruns)
# ──────────────────────────────────────────────────────────
if "analysis_results" in st.session_state:
    ar = st.session_state["analysis_results"]
    all_ml = ar["all_ml"]
    all_spreads = [c for c in ar["all_spreads"] if c.get("games_sampled", 0) >= 5]
    all_totals = ar["all_totals"]
    all_props = [c for c in ar["all_props"] if c.get("no_history") or c.get("games_sampled", 0) >= 5]
    checklist_entries = _value_bet_checklist_entries(
        all_ml, all_spreads, all_totals, all_props)
    # A fresh Analyze deferred its selection reset to here — now that the new
    # slate's entries are known and before the checkboxes render below, write
    # their keys False (reliable uncheck) and pop any leftover from a prior slate.
    if st.session_state.pop("_reset_bet_selections", False):
        _clear_bet_selections(
            {entry["selection_key"] for entry in checklist_entries})

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
    if checklist_entries:
        _render_selected_bet_checklist(checklist_entries, ar)

    # Moneyline results
    if all_ml:
        st.subheader("💰 Moneyline Analysis")
        if any(c.get("market_comparison") for c in all_ml):
            st.caption(
                "Market Comparison is line-shopping context only and does not "
                "change model probability, edge, or ranking."
            )
        value_ml = [c for c in all_ml if c["is_value"]]
        other_ml = [c for c in all_ml if not c["is_value"]]

        if value_ml:
            st.success(f"**{len(value_ml)} value bet(s) found!**")
            for c in sorted(value_ml, key=lambda x: x["edge_pct"], reverse=True):
                with st.expander(f"🔥 {c['team']} ({c['home_away']}) vs {c['opponent']}  —  Best edge: +{c['best_edge_pct']}%", expanded=True):
                    _select_bet_checkbox(c, "moneyline")
                    cols = st.columns(7)
                    cols[0].metric("Model Probability", f"{c['blended_prob']}%")
                    cols[1].metric("Price Break-even", f"{c['best_book_implied_prob']}%")
                    cols[2].metric("Edge", f"{c['best_edge_pct']:+.2f}%",
                                   delta=f"{c['best_price']:+d} at {c['best_book']}")
                    cols[3].metric("Expected ROI", f"{c['expected_roi_pct']:+.2f}%")
                    cols[4].metric("Season Win%", f"{c['season_win_pct']}%")
                    cols[5].metric("Recent Win%", f"{c['recent_win_pct']}%")
                    p_val, p_delta = _dk_payout_strs(c.get("best_price"))
                    cols[6].metric("DK Payout", p_val, delta=p_delta, delta_color="off",
                                   help="American odds and profit on a $10 bet at DraftKings.")
                    _render_market_comparison(
                        c.get("market_comparison"), ar["sport_key"])

        if other_ml:
            with st.expander(f"Other matchups ({len(other_ml)})"):
                rows = []
                for c in other_ml:
                    rows.append({
                        "Team": c["team"],
                        "Home/Away": c["home_away"],
                        "Price Break-even": f"{c['book_implied_prob']}%",
                        "Model Probability": f"{c.get('blended_prob', c['hist_prob'])}%",
                        "Edge": f"{c['edge_pct']:+.2f}%",
                        "Expected ROI": f"{c.get('expected_roi_pct', 0):+.2f}%",
                        "Market Comparison": _market_comparison_summary(
                            c.get("market_comparison")),
                    })
                st.table(rows)

    # Spreads results
    if all_spreads:
        st.subheader("📊 Spread Analysis")
        if any(c.get("market_comparison") for c in all_spreads):
            st.caption(
                "Market Comparison shows line and price together. It is "
                "informational and does not change model edge or ranking."
            )
        value_sp = [c for c in all_spreads if c["is_value"]]
        other_sp = [c for c in all_spreads if not c["is_value"]]

        if value_sp:
            st.success(f"**{len(value_sp)} spread value bet(s) found!**")
            for c in sorted(value_sp, key=lambda x: x["edge_pct"], reverse=True):
                with st.expander(f"🔥 {c['team']} {c['spread']:+.2f} ({c['home_away']})  —  Edge: +{c['edge_pct']}%", expanded=True):
                    _select_bet_checkbox(c, "spread")
                    cols = st.columns(7)
                    cols[0].metric("Spread", f"{c['spread']:+.2f}")
                    cols[1].metric("Model Probability", f"{c['cover_rate']}%")
                    cols[2].metric("Price Break-even", f"{c['implied_prob']}%")
                    cols[3].metric("Edge", f"{c['edge_pct']:+.2f}%")
                    cols[4].metric("Expected ROI", f"{c['expected_roi_pct']:+.2f}%")
                    cols[5].metric("Projected Margin", f"{c['pred_game_margin']:+.2f}")
                    p_val, p_delta = _dk_payout_strs(c.get("price"))
                    cols[6].metric("DK Payout", p_val, delta=p_delta, delta_color="off",
                                   help="American odds and profit on a $10 bet at DraftKings.")
                    _render_market_comparison(
                        c.get("market_comparison"), ar["sport_key"],
                        market_key="spreads")
                    if c.get("model_source") == "expected_runs_ensemble":
                        st.caption(
                            "Expected-runs spread ensemble active · projected "
                            f"score {c['expected_away_runs']:.2f}–"
                            f"{c['expected_home_runs']:.2f} (away–home)."
                        )
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
                        "Price Break-even": f"{c.get('implied_prob', 50.0)}%",
                        "Edge": f"{c['edge_pct']:+.2f}%",
                        "Expected ROI": (f"{c['expected_roi_pct']:+.2f}%"
                                         if c.get("expected_roi_pct") is not None else "n/a"),
                        "Market Comparison": _market_comparison_summary(
                            c.get("market_comparison"), market_key="spreads"),
                    })
                st.table(rows)

    # Totals results
    if all_totals:
        st.subheader("📈 Over/Under Analysis")
        if any(c.get("market_comparisons") for c in all_totals):
            st.caption(
                "Market Comparison shows the displayed side's line and price "
                "together. It is informational and does not change model edge "
                "or ranking."
            )
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
                if c.get("is_over_value") or c.get("is_under_value"):
                    _select_bet_checkbox(c, "total", side=side)
                p_val, p_delta = _dk_payout_strs(payout_price)
                cols = st.columns(7)
                cols[0].metric("Line", c["line"])
                cols[1].metric("Projected Total", c["projected_total"])
                cols[2].metric(f"Model Probability ({side})", f"{model_probability:.2f}%")
                cols[3].metric("Price Break-even", f"{implied_probability:.2f}%")
                cols[4].metric("Edge", f"{edge_pct:+.2f}%")
                cols[5].metric("Expected ROI", (f"{expected_roi:+.2f}%"
                                                 if expected_roi is not None else "n/a"))
                cols[6].metric(payout_label, p_val, delta=p_delta, delta_color="off",
                               help="American odds and profit on a $10 bet at DraftKings.")
                st.caption(f"Projected total minus book line: {c['diff_from_line']:+.2f}")
                _render_market_comparison(
                    (c.get("market_comparisons") or {}).get(side.title()),
                    ar["sport_key"], market_key="totals", side=side)
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

        def _lineup_badge(c):
            """§2.5A confirmed/OUT badge; '' when unknown (lineup not yet posted)."""
            status = c.get("lineup_status")
            if status == "in":
                return "✓ confirmed lineup"
            if status == "out":
                return "⚠ OUT — not in posted lineup"
            return ""

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
                        _select_bet_checkbox(c, "player_prop")
                        cols = st.columns(8)
                        cols[0].metric("Suggested", f"{c['prop_label']} {c['safe_threshold']}+")
                        cols[1].metric(
                            "Model Probability",
                            f"{c['model_hit_at_safe']}%",
                        )
                        cols[2].metric(
                            "Price Break-even",
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
                            p_val, p_delta = _dk_payout_strs(c.get("dk_over_price"))
                            payout_label = "DK Payout (book line)"
                            payout_help = "Payout for the OVER at the standard book line on DraftKings. Alt-line fetch is disabled or no alt was offered at the suggested threshold — DK's actual alt price will differ."
                        cols[7].metric(payout_label, p_val, delta=p_delta,
                                       delta_color="off", help=payout_help)
                        badge = _lineup_badge(c)
                        st.caption(
                            f"Matchup: {c['matchup']}"
                            f"  |  Projected average: {c['avg_stat']}"
                            f"  |  Model probability at book line: {c['model_hit_at_line']}%"
                            f"  |  Line gap: {c['line_gap']:+.2f}"
                            + (f"  |  {badge}" if badge else "")
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
                    # DraftKings price is what the user bets and is displayed;
                    # multi-book data only feeds the consensus break-even/edge
                    # (P1.1b, DK-only). Value bets always have a DK price.
                    dk_bet_price = c.get("dk_price")
                    with st.expander(f"🔥 {c['player']} — {c['prop_label']} {c['direction']} {c['line']}  —  Edge: +{c['edge_pct']}%", expanded=True):
                        _select_bet_checkbox(c, "player_prop")
                        cols = st.columns(8)
                        cols[0].metric("Line", c["line"])
                        cols[1].metric("Projected Average", c["avg_stat"])
                        cols[2].metric("Model Probability", f"{hit_prob}%")
                        cols[3].metric("Price Break-even", f"{implied_prob}%")
                        cols[4].metric("Edge", f"{c['edge_pct']:+.2f}%")
                        cols[5].metric("Expected ROI", f"{c['expected_roi_pct']:+.2f}%")
                        cols[6].metric("Direction", c["direction"])
                        p_val, p_delta = _dk_payout_strs(dk_bet_price)
                        cols[7].metric(
                            f"DK Payout ({c['direction']})", p_val, delta=p_delta, delta_color="off",
                            help=(f"DraftKings price for the {c['direction']} at the book "
                                  "line. Edge is measured vs the de-vigged consensus of "
                                  "all U.S. books; Expected ROI is at this DK price."),
                        )
                        line_gap = c["avg_stat"] - c["line"]
                        badge = _lineup_badge(c)
                        st.caption(
                            f"Matchup: {c['matchup']}"
                            f"  |  Book line: {c['line']}"
                            f"  |  Projected average: {c['avg_stat']}"
                            f"  |  Line gap: {line_gap:+.2f}"
                            + (f"  |  {badge}" if badge else "")
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
                            "Price Break-even": f"{c.get('safe_alt_implied', 0)}%",
                            "Edge": f"{c['edge_pct']:+.2f}%",
                            "Expected ROI": (f"{c['expected_roi_pct']:+.2f}%"
                                             if c.get("expected_roi_pct") is not None else "n/a"),
                            "Lineup": _lineup_badge(c) or "—",
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
                            "Price Break-even": (f"{c['over_implied']}%" if c["direction"] == "OVER"
                                                 else f"{c['under_implied']}%"),
                            "Edge": f"{c['edge_pct']:+.2f}%",
                            "Expected ROI": (f"{c['expected_roi_pct']:+.2f}%"
                                             if c.get("expected_roi_pct") is not None else "n/a"),
                            "Lineup": _lineup_badge(c) or "—",
                        })
                st.table(rows)

        if no_hist:
            st.caption(f"ℹ️ {len(no_hist)} prop(s) skipped — no ESPN history found")

    if not any([all_ml, all_spreads, all_totals, all_props]) and not search_filter:
        st.info("No results. Select markets or props and click Analyze.")
