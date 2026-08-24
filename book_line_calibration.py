"""
Book-line calibration analysis.

Joins cached sportsbook odds snapshots (`cache/*.json`) to ESPN player
gamelogs, finds the actual stat the player produced in each game, and
evaluates three probabilistic forecasters at the real book line:

    A) empirical    — weighted fraction of prior games > line (current method)
    B) resid_normal — bias-correct projection, then Φ((projected+μ_r − L)/σ_r)
    C) resid_ecdf   — bias-correct projection, then 1 − F_r(L − (projected+μ_r))

Residual stats (μ_r, σ_r, F_r) are fit from a chronological HOLDOUT split
to avoid leakage: the earliest 50% of observations build calibration, the
later 50% are scored.

Usage:
    python book_line_calibration.py --sport nba
    python book_line_calibration.py --sport mlb --variants recency,all
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

from analysis import (
    _norm_cdf, _half_life_for, _recency_weights, _weighted_rate,
    _weighted_std, _normal_inv_cdf, _weighted_quantile,
)
from stats import (  # §2.2 method "E"
    negbin_at_least, fit_negbin_dispersion, fit_negbin_params)
from backtest import (
    cached_gamelog, cached_athlete_id,
    SPORT_MAP, VARIANT_PRESETS, _empirical_cdf, _brier, _logloss, _hit_rate,
    _resolve_params, opp_defense_mult, venue_mult,
    _team_defense_lookup, _resolve_opp_pts_allowed,
    _per_player_stats, _shrunk, _role_matches_gamelog,
)
from espn_client import PROP_STAT_MAP, WAREHOUSE_ONLY_PROPS, ip_to_outs
from pricing_common import et_local_date  # UTC at rest, ET on read
import prop_features  # §2.6 candidate-feature registry (rest/days-off, ...)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ODDS_CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")


PROPS_BY_SPORT = {
    "nba":  ["player_points", "player_rebounds", "player_assists"],
    "mlb":  ["pitcher_strikeouts", "batter_hits"],
    "nfl":  ["player_pass_yds", "player_rush_yds"],
}


# ──────────────────────────────────────────────────────────────────────────────
#  Step 1: harvest book lines from cached odds snapshots
# ──────────────────────────────────────────────────────────────────────────────

def _is_prop_market(market_key):
    return market_key.startswith(("player_", "pitcher_", "batter_"))


def harvest_book_lines(sport_key, target_props):
    """
    Walk cache/*.json and return a list of dicts:
      {sport_key, commence_time, game_date, home_team, away_team,
       player, prop_key, line, over_price, under_price}
    Picks the consensus (median) line across books per (player, prop, game).
    """
    raw = []  # (game_date, sport, home, away, player, prop, side, line, price)
    for path in sorted(glob.glob(os.path.join(ODDS_CACHE_DIR, "*.json"))):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        data = doc.get("data", doc) if isinstance(doc, dict) else doc
        if not isinstance(data, dict):
            continue
        if data.get("sport_key") != sport_key:
            continue
        ct = data.get("commence_time")
        if not ct:
            continue
        date = ct[:10]
        home = data.get("home_team")
        away = data.get("away_team")
        for book in data.get("bookmakers", []) or []:
            for mkt in book.get("markets", []) or []:
                mk = mkt.get("key", "")
                if mk not in target_props:
                    continue
                for o in mkt.get("outcomes", []) or []:
                    player = o.get("description")
                    side = o.get("name")  # "Over" / "Under"
                    point = o.get("point")
                    price = o.get("price")
                    if not (player and side and point is not None):
                        continue
                    raw.append((date, sport_key, home, away,
                                player, mk, side, point, price))

    # Consolidate to one row per (player, prop, game): use the consensus line
    # (median across books) and the best price per side.
    by_key = defaultdict(lambda: {"lines": [], "over_prices": [], "under_prices": [],
                                  "home": None, "away": None})
    for date, sport, home, away, player, mk, side, point, price in raw:
        key = (date, sport, player, mk)
        cell = by_key[key]
        cell["home"] = home
        cell["away"] = away
        cell["lines"].append(point)
        if side.lower() == "over" and price is not None:
            cell["over_prices"].append(price)
        elif side.lower() == "under" and price is not None:
            cell["under_prices"].append(price)

    out = []
    for (date, sport, player, mk), cell in by_key.items():
        if not cell["lines"]:
            continue
        # Consensus line = mode/median of book lines (most books agree)
        lines_sorted = sorted(cell["lines"])
        consensus = lines_sorted[len(lines_sorted) // 2]
        out.append({
            "sport_key": sport,
            "game_date": date,
            "home_team": cell["home"],
            "away_team": cell["away"],
            "player": player,
            "prop_key": mk,
            "line": consensus,
            "over_price": max(cell["over_prices"]) if cell["over_prices"] else None,
            "under_price": max(cell["under_prices"]) if cell["under_prices"] else None,
        })
    out.sort(key=lambda r: (r["game_date"], r["player"], r["prop_key"]))
    return out


def harvest_book_lines_from_store(sport_key, target_props, label=""):
    """
    Harvest (player, prop, game) book lines from the DURABLE historical_odds
    store (backfill_historical_odds.py output) instead of the ephemeral
    cache/*.json HTTP cache. Returns the same shape as harvest_book_lines():
      {sport_key, game_date, home_team, away_team, player, prop_key, line,
       over_price, under_price}

    The store already consolidates each (player, prop, game) to one consensus
    line + best executable price per side (see odds_client.parse_player_props),
    so no per-book median is computed here.
    """
    import historical_odds as store_mod
    store = store_mod.load_store(sport_key, label)
    target = set(target_props or [])
    out = []
    for entry in (store.get("games") or {}).values():
        commence = entry.get("commence_time")
        date = et_local_date(commence)   # UTC at rest, ET on read
        if not date:
            continue
        home = entry.get("home_team")
        away = entry.get("away_team")
        for prop_key, by_player in (entry.get("props") or {}).items():
            if prop_key not in target:
                continue
            for player, info in (by_player or {}).items():
                line = info.get("line")
                if line is None:
                    continue
                out.append({
                    "sport_key": sport_key,
                    "game_date": date,
                    "commence_time": commence,
                    "event_id": entry.get("event_id"),   # local store: usually None
                    "home_team": home,
                    "away_team": away,
                    "player": player,
                    "prop_key": prop_key,
                    "line": line,
                    "over_price": info.get("over_price"),
                    "under_price": info.get("under_price"),
                })
    out.sort(key=lambda r: (r["game_date"], r["player"], r["prop_key"]))
    return out


def harvest_book_lines_from_prediction_log(sport_key, target_props):
    """Harvest (player, prop, game) book lines from the app's own RESOLVED
    prediction log — the lines it logged at analysis time. Unlike the
    historical_odds backfill store (Odds-API credits, offline-only), this GROWS
    for free with live analysis. Same shape as harvest_book_lines_from_store;
    home/away/prices are absent (the calibration pipeline doesn't use them, and
    join_book_lines_to_actuals re-derives the actual from the ESPN gamelog).
    Resolved rows only — a completed game; a DNP-void (resolved, no game) simply
    finds no game in the join and is skipped. Best-effort → [] on any error."""
    try:
        import recalibration
        rows = recalibration._read_log(
            where={"sport_key": sport_key, "resolved": True})
    except Exception:
        return []
    target = set(target_props or [])
    out = []
    for r in rows:
        if not r.get("resolved"):          # belt-and-suspenders (Blob path ignores where)
            continue
        pk = r.get("prop_key")
        if pk not in target:
            continue
        player = r.get("player")
        line = r.get("line")
        commence = r.get("commence_time")
        # UTC at rest, ET on read: derive the calendar date in Eastern, falling
        # back to the stored game_date only when commence is absent.
        gd = et_local_date(commence) or (r.get("game_date") or "")[:10]
        if not player or line is None or not gd:
            continue
        out.append({
            "sport_key": sport_key, "game_date": gd,
            "commence_time": commence,
            "event_id": r.get("event_id"),   # present on the pred-log row
            "home_team": None, "away_team": None,
            "player": player, "player_mlb_id": r.get("player_mlb_id"),
            "prop_key": pk, "line": line,
            "over_price": None, "under_price": None,
        })
    return out


def _round_line(row):
    try:
        return round(float(row.get("line")), 1)
    except (TypeError, ValueError):
        return row.get("line")


def _book_line_key(row):
    """Dedup identity for a book line across sources: (player, prop, date, line)."""
    return (row.get("player"), row.get("prop_key"), row.get("game_date"),
            _round_line(row))


def _book_line_id_key(row):
    """Id-based dedup dimension: (player_mlb_id, prop, date, line), or None when
    the row carries no canonical id. Collapses spelling/accent variants of one
    player that the name key would keep as distinct rows. Because both harvest
    sources carry the same odds-feed name, the name key still catches the case
    where only one source has been id-enriched — so adding this key only ever
    removes true duplicates, never inflates obs."""
    mlb_id = row.get("player_mlb_id")
    if not mlb_id:
        return None
    return (mlb_id, row.get("prop_key"), row.get("game_date"), _round_line(row))


def harvest_real_line_book_lines(sport_key, target_props, label=""):
    """Union of real book lines for calibration, deduped by (player, prop,
    ET-date, line), preferring the primary (richer: prices/teams) on a collision.

    Primary source = the Azure ``odds_line`` warehouse (durable, grows with every
    live analysis) when SQL is enabled and no ``label`` is forced; else the local
    historical_odds JSON store (offline fallback). The RESOLVED prediction log is
    unioned as a BACKSTOP — it covers analyses that ran on cached/credit-exhausted
    odds (no fresh fetch → no warehouse capture). True same-day doubleheaders are
    dropped from both sources by event_id (ambiguous box-score attribution).
    Returns (book_lines, n_primary, n_pred)."""
    target = set(target_props or [])
    try:
        import warehouse
    except Exception:
        warehouse = None
    use_warehouse = False
    if warehouse is not None and not label:
        try:
            import db_store
            use_warehouse = db_store.enabled()
        except Exception:
            use_warehouse = False
    if use_warehouse:
        # Pass the target props so the warehouse read filters in SQL (a single-prop
        # diagnostic then transfers ~1/7 the rows). The Python filter stays as a
        # belt-and-suspenders on the assembled output.
        primary = [r for r in warehouse.load_prop_lines(
                       sport_key, prop_keys=(list(target) or None))
                   if r.get("prop_key") in target]
    else:
        primary = harvest_book_lines_from_store(sport_key, target_props, label)
    pred_lines = harvest_book_lines_from_prediction_log(sport_key, target_props)

    # Drop true same-day doubleheaders (detected from the team-carrying primary
    # rows; game_date is ET so consecutive-day series games are NOT flagged).
    dh_events = warehouse.doubleheader_event_ids(primary) if warehouse else set()

    seen, seen_ids, out, dropped_dh = set(), set(), [], 0
    for src in (primary, pred_lines):   # primary first → preferred on collision
        for r in src:
            eid = r.get("event_id")
            if eid and eid in dh_events:
                dropped_dh += 1
                continue
            k = _book_line_key(r)
            idk = _book_line_id_key(r)
            if k in seen or (idk is not None and idk in seen_ids):
                continue
            seen.add(k)
            if idk is not None:
                seen_ids.add(idk)
            out.append(r)
        if src is primary:
            n_primary = len(out)
    out.sort(key=lambda r: (r["game_date"], r["player"], r["prop_key"]))
    if dropped_dh:
        print(f"  Dropped {dropped_dh} doubleheader line(s) "
              f"(ambiguous box-score attribution).")
    return out, n_primary, len(out) - n_primary


# ──────────────────────────────────────────────────────────────────────────────
#  Step 2: join each book line to the player's actual stat in that game
# ──────────────────────────────────────────────────────────────────────────────

def _stat_label_for(prop_key, gamelog):
    for label in PROP_STAT_MAP.get(prop_key, []):
        if any(label in g for g in gamelog):
            return label
    return None


# Short current-season gamelog TTL for the calibration join. The default
# cached_gamelog TTL is 30 days, so recent games (the last week+) are absent from
# a "fresh" cache — which silently drops every book line for a game newer than the
# cache. That's fatal for props whose ONLY data is the live prediction log /
# warehouse (recent games), e.g. pitcher_outs / pitcher_earned_runs, which
# otherwise join to zero obs. 6h guarantees a morning refit picks up the prior
# night's finalized games (games finalize overnight; gamelog_store keeps stored
# rows on a failed refetch, so a short TTL is safe). Past seasons are unaffected
# (their bucket TTL stays immutably long).
CALIB_GAMELOG_TTL_HOURS = 6

# P4 calibration cutover (default OFF): grade real-line fits off the StatsAPI
# warehouse per-game facts instead of the ESPN cached_gamelog. Ships GATED OFF while
# the warehouse backfill runs; fail-open to ESPN. Scope is the real-line join only —
# the synthetic-sweep backtest engine stays on ESPN (its park/reliability/defense
# apparatus is ESPN-team-id-keyed). ⚠ run a per-game-dict opponent-NAME parity check
# before flipping (opp_defense + park key on the opponent name).
_MLB_WAREHOUSE_CALIB_ENV = "ODI_MLB_WAREHOUSE_CALIB"


def _warehouse_calib_enabled():
    return os.environ.get(_MLB_WAREHOUSE_CALIB_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


def _calib_role(rows):
    """batter vs pitcher for the warehouse log fetch — the majority role across a
    player's book lines (ties → batter). _role_matches_gamelog drops any cross-role
    rows downstream, mirroring the ESPN one-log-per-player behavior."""
    pitcher = sum(1 for r in rows
                  if str(r.get("prop_key") or "").startswith("pitcher_"))
    return "pitcher" if pitcher > (len(rows) - pitcher) else "batter"


def _match_rows_to_gamelog(gamelog, rows, from_warehouse, espn_sport, enriched,
                           counters):
    """Grade one player-SEASON gamelog (ESPN cached_gamelog shape) against its
    book-line ``rows``, appending matched obs to ``enriched`` and accumulating
    ``counters['no_game']``. Extracted so the MLB warehouse path can grade each
    obs against ITS OWN season's gamelog while the non-baseball path keeps its
    single-gamelog behavior."""
    gamelog = sorted(gamelog, key=lambda g: g.get("game_date") or "", reverse=True)

    # Build a date → game-index lookup. A date carrying MORE THAN ONE gamelog
    # entry is a doubleheader: first-wins would mis-bind a book line to the wrong
    # game's box score, so track those dates and skip them in the match below
    # (belt-and-suspenders; true doubleheaders are already dropped upstream in
    # harvest_real_line_book_lines).
    date_idx = {}
    date_counts = {}
    for i, g in enumerate(gamelog):
        if g.get("completed") is False:
            continue             # in-progress/partial game: not a gradeable box score
        d = (g.get("game_date") or "")[:10]
        if not d:
            continue
        date_counts[d] = date_counts.get(d, 0) + 1
        if d not in date_idx:
            date_idx[d] = i
    dup_dates = {d for d, c in date_counts.items() if c > 1}

    for row in rows:
        # batter_total_bases / batter_rbis are WAREHOUSE-ONLY: an ESPN gamelog's
        # 'RBI' is uncalibrated and it has no 'TB', so NEVER fit these off an
        # ESPN-sourced gamelog. (from_warehouse is False only on a non-baseball /
        # ESPN log; the MLB path is always warehouse post-P6.)
        if (not from_warehouse and espn_sport == "baseball"
                and row["prop_key"] in WAREHOUSE_ONLY_PROPS):
            counters["no_game"] += 1
            continue
        # Never grade a pitcher prop off a batter's gamelog (or vice-versa): the
        # "K"/"SO" strikeout labels collide across MLB roles. A role mismatch
        # drops just this row; non-MLB props (role None) always pass.
        if not _role_matches_gamelog(row["prop_key"], gamelog):
            counters["no_game"] += 1
            continue
        stat_label = _stat_label_for(row["prop_key"], gamelog)
        if not stat_label:
            continue
        # game_date is UTC-ish on both sides, so date-only match is usually
        # accurate. Also try ±1 day for timezone slippage (stays within the
        # season's gamelog — MLB seasons don't cross a calendar-year boundary).
        d = row["game_date"]
        matched_d = d if d in date_idx else None
        if matched_d is None:
            from datetime import date as _date, timedelta
            try:
                d0 = _date.fromisoformat(d)
                for delta in (-1, 1):
                    alt = (d0 + timedelta(days=delta)).isoformat()
                    if alt in date_idx:
                        matched_d = alt
                        break
            except ValueError:
                pass
        if matched_d is None:
            counters["no_game"] += 1
            continue
        if matched_d in dup_dates:       # doubleheader → can't attribute cleanly
            counters["no_game"] += 1
            continue

        idx = date_idx[matched_d]
        test_game = gamelog[idx]
        actual = test_game.get(stat_label)
        if actual is None:
            counters["no_game"] += 1
            continue
        if stat_label == "IP":           # pitcher_outs: IP notation -> outs
            actual = ip_to_outs(actual)
        min_played = test_game.get("MIN", 0.0) or 0.0
        if min_played and min_played < 10.0:
            counters["no_game"] += 1
            continue

        # Strictly-earlier games in THIS season (gamelog is most-recent-first, so
        # idx+1: are the prior games) → the as-of feature window. No future/other-
        # season leakage: only the obs's own season, only games before it.
        prior_games = gamelog[idx + 1:]
        if len(prior_games) < 10:
            counters["no_game"] += 1
            continue

        enriched.append({
            **row,
            "stat_label": stat_label,
            "actual": float(actual),
            "test_game": test_game,
            "prior_games": prior_games,
        })


def join_book_lines_to_actuals(book_lines, espn_sport, espn_league):
    """
    For each book line, resolve the player, pull their gamelog, locate the game on
    `game_date`, and attach `actual` + `prior_games`. Returns enriched dicts
    (skipping unjoinable rows).

    MLB (warehouse) path grades each obs against ITS OWN season's gamelog
    (get_calib_gamelog per season = game_date year), so book lines from ALL
    backfilled seasons — not just the current one — enter the fit. The join was
    previously current-season-only, silently dropping every prior-season real
    book line. Leakage-safe: each obs uses only its own season's actuals +
    within-season prior_games. NBA/NFL keep the ESPN name→aid→single-gamelog path.

    P6 MLB cutover — WAREHOUSE-ONLY (no ESPN for baseball): a valid MLBAM id is
    sufficient; a warehouse miss / missing MLBAM id drops the row (never falls
    open to ESPN).
    """
    # Group by canonical id (falling back to name) so spelling variants of one
    # player pool together.
    by_player = defaultdict(list)
    for row in book_lines:
        by_player[row.get("player_mlb_id") or row["player"]].append(row)

    enriched = []
    counters = {"no_game": 0}
    skipped_no_player = 0

    for _, rows in by_player.items():
        player = rows[0].get("player")
        mlb_id = rows[0].get("player_mlb_id")
        if espn_sport == "baseball":
            if not mlb_id:
                skipped_no_player += len(rows)
                continue
            role = _calib_role(rows)
            # Grade each obs against its OWN season's gamelog (season = game_date
            # year; MLB season == calendar year). One fetch per (player, season).
            rows_by_season = defaultdict(list)
            for r in rows:
                rows_by_season[(r.get("game_date") or "")[:4]].append(r)
            for yr, srows in rows_by_season.items():
                try:
                    season = int(yr)
                except (TypeError, ValueError):
                    skipped_no_player += len(srows)
                    continue
                gamelog = None
                try:
                    import mlb_warehouse
                    gamelog = mlb_warehouse.get_calib_gamelog(
                        mlb_id, role, season=season) or None
                except Exception:
                    gamelog = None
                if not gamelog:
                    skipped_no_player += len(srows)
                    continue
                _match_rows_to_gamelog(gamelog, srows, True, espn_sport,
                                       enriched, counters)
        else:
            aid = cached_athlete_id(espn_sport, espn_league, player)
            if not aid:
                skipped_no_player += len(rows)
                continue
            gamelog = cached_gamelog(espn_sport, espn_league, aid,
                                     ttl_hours=CALIB_GAMELOG_TTL_HOURS,
                                     player_name=player)
            if not gamelog:
                skipped_no_player += len(rows)
                continue
            _match_rows_to_gamelog(gamelog, rows, False, espn_sport,
                                   enriched, counters)

    print(f"  Matched {len(enriched):,} book lines to actual results "
          f"(one per player-prop-game); dropped {skipped_no_player:,} "
          f"(player not found) and {counters['no_game']:,} (no matching game/stat).")
    _attach_gamecontext(enriched, espn_sport)
    _attach_platoon(enriched, espn_sport)
    return enriched


def _attach_gamecontext(enriched, espn_sport):
    """Attach each obs's own-side GameContext gc_factor forms (roadmap §3.1).

    Groups obs by (home_team, away_team, game_date), builds the shared, cached,
    leakage-safe GameContext once per game (mlb_starters.build_game_context —
    run inputs come ONLY from the as-of get_expected_runs_team_factors, never a
    season-to-date source), and writes ``obs["gc_factor"] = {full, own, opp}``
    for the batter's OWN side (test_game.is_home). MLB only.

    Fail-open, never raises: rows whose home_team/away_team is None (prediction-
    log-sourced, no game frame) or whose context is incomplete are left WITHOUT
    gc_factor, so the gamecontext feature no-ops to 1.0 and the obs stays in the
    sample at baseline (not dropped). A GameContext failure must never break the
    refit. Runs once per join (diagnose_features reuses the enriched list across
    all feature strengths), so the cost is one as-of fetch per distinct game-date,
    memoized in mlb_starters._GC_CACHE."""
    if espn_sport != "baseball":
        return enriched
    try:
        import mlb_starters
    except Exception:
        return enriched
    team_index_by_season = {}
    for obs in enriched:
        home = obs.get("home_team")
        away = obs.get("away_team")
        gd = obs.get("game_date")
        if not home or not away or not gd:
            continue                      # fail-open: no game frame -> gc absent
        try:
            season = int(str(gd)[:4])
        except (TypeError, ValueError):
            continue
        if season not in team_index_by_season:
            try:
                team_index_by_season[season] = mlb_starters.get_team_index(season)
            except Exception:
                team_index_by_season[season] = None
        try:
            gc = mlb_starters.build_game_context(
                home, away, gd, season,
                team_index=team_index_by_season[season])
        except Exception:
            continue
        if not gc or not gc.get("complete"):
            continue
        side = "home" if obs.get("test_game", {}).get("is_home") else "away"
        obs["gc_factor"] = gc["gc_factor"][side]
    return enriched


# ── platoon attach (roadmap §2.6, batter vs opposing-starter hand) ────────────
# The raw Statcast cache (savant_history) stores EVERY pitch with
# pitcher/p_throws/batting_team/batter, so the opposing starter's HAND is
# recoverable entirely offline — no schedule fetch, no cross-source team mapping
# (cheaper than gamecontext's per-date network call). Pre-clamp bound + sample
# thresholds live here (the attach side); prop_features.PLATOON_FEAT_CAP is the
# downstream strength-scaled re-clamp (mirrors GC_RUN_CAP / GC_FEAT_CAP split).
PLATOON_RUN_CAP = 0.15        # pre-clamp on the raw vs-hand / all-hands ratio
PLATOON_MIN_BBE_BASE = 40     # all-hands baseline leg (== the xBA-blend gate)
PLATOON_MIN_BBE_VS = 25       # vs-hand leg — lower for LHP coverage (~28% of BBE)

_PLATOON_CACHE = {}           # frozenset(years) -> built tuple | False (failed)


def _build_platoon_indices(raw, bp):
    """Build the four leakage-safe platoon lookups from raw Statcast PITCH rows.

    Returns (base_idx, vs_idx, starter_hand, batter_team) or None when empty:
      * base_idx  — AsOfIndex keyed by batter -> as-of ALL-hands xwOBAcon
      * vs_idx    — AsOfIndex keyed by (batter, p_throws) -> as-of vs-hand xwOBAcon
      * starter_hand[(game_date, batting_team)] — p_throws of the max-pitch-count
        opposing pitcher that date (a robust starter proxy: starters throw ~4x a
        reliever's pitches)
      * batter_team[(batter, game_date)] — the batter's own batting_team that date
        (self-derived so keys match Savant's namespace, no ESPN abbr mapping)."""
    if not raw:
        return None
    base_idx = bp.AsOfIndex()
    vs_idx = bp.AsOfIndex()
    hands = bp._pitcher_hands(raw)
    pitch_counts = defaultdict(lambda: defaultdict(int))  # (gd,team) -> {pid: n}
    batter_team = {}
    for r in raw:
        b = r.get("batter")
        gd = r.get("game_date")
        bt = r.get("batting_team")
        # xwOBAcon legs: batted balls only (xwoba populated on BBE), leakage-safe
        # via AsOfIndex's strict date<as_of bisect.
        x = r.get("xwoba")
        if b and gd and x is not None:
            base_idx.add(b, gd, x)
            pth = r.get("p_throws")
            if pth:
                vs_idx.add((b, pth), gd, x)
        # team + opposing-starter HAND: pre-game facts, over ALL pitches.
        if b and gd and bt:
            batter_team[(b, gd)] = bt          # consistent within a game
        pit = r.get("pitcher")
        if gd and bt and pit:
            pitch_counts[(gd, bt)][pit] += 1
    starter_hand = {}
    for key, counts in pitch_counts.items():
        top_pid = max(counts, key=counts.get)  # most pitches vs this team = starter
        h = hands.get(top_pid)
        if h:
            starter_hand[key] = h
    return base_idx, vs_idx, starter_hand, batter_team


def _attach_platoon(enriched, espn_sport):
    """Attach each batter_hits obs's leakage-safe platoon factor (roadmap §2.6).

    For each batter_hits obs, resolves the opposing starter's hand (max-pitch
    pitcher vs the batter's team that date), then writes ``obs["platoon_factor"]``
    = the batter's as-of vs-hand xwOBAcon / all-hands baseline (pre-clamped to
    +/-PLATOON_RUN_CAP). Both quality legs use only batted balls STRICTLY before
    the obs's game_date (AsOfIndex bisect_left), so no outcome from the graded
    game enters the factor — only the opposing hand + the batter's team, both
    pre-game facts, are read from the graded date.

    Fail-open, never raises: any missing piece (thin vs-hand sample, unknown
    opposing hand, unresolved player id, non-MLB) leaves the obs WITHOUT a
    platoon_factor, so the platoon feature no-ops to 1.0 and the obs stays in the
    sample at baseline (not dropped). Runs once per join (diagnose_features reuses
    the enriched list across strengths); the built indices are memoized in
    _PLATOON_CACHE by season-set, and the whole attach short-circuits off the
    batter_hits path so unrelated joins pay nothing."""
    if espn_sport != "baseball":
        return enriched
    if not any(o.get("prop_key") == "batter_hits" for o in enriched):
        return enriched                     # perf: only batter_hits consumes it
    try:
        import savant_history as sh
        import backtest_props as bp
        import mlb_starters
    except Exception:
        return enriched

    years = sorted({str(o.get("game_date"))[:4] for o in enriched
                    if o.get("game_date")})
    if not years:
        return enriched
    cache_key = frozenset(years)
    built = _PLATOON_CACHE.get(cache_key)
    if built is None:
        try:
            raw = []
            for y in years:
                raw.extend(sh.load_days(f"{y}-03-01", f"{y}-11-30"))
            built = _build_platoon_indices(raw, bp)
        except Exception:
            built = False                   # remember the failure; fail-open
        _PLATOON_CACHE[cache_key] = built
    if not built:
        return enriched
    base_idx, vs_idx, starter_hand, batter_team = built

    for obs in enriched:
        if obs.get("prop_key") != "batter_hits":
            continue
        try:
            gd = obs.get("game_date")
            player = obs.get("player")
            if not gd or not player:
                continue
            season = int(str(gd)[:4])
            pid_info = mlb_starters.find_player_id(player, season)
            if not pid_info or not pid_info[0] or pid_info[1]:   # batter only
                continue
            pid = str(pid_info[0])
            team = batter_team.get((pid, gd))
            if not team:
                continue
            opp_hand = starter_hand.get((gd, team))
            if not opp_hand:
                continue
            vs = vs_idx.asof_mean((pid, opp_hand), gd, min_bbe=PLATOON_MIN_BBE_VS)
            base = base_idx.asof_mean(pid, gd, min_bbe=PLATOON_MIN_BBE_BASE)
            if vs is None or base is None or base <= 0:
                continue
            ratio = vs / base
            obs["platoon_factor"] = max(
                1.0 - PLATOON_RUN_CAP, min(1.0 + PLATOON_RUN_CAP, ratio))
        except Exception:
            continue                        # fail-open per obs
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
#  Step 3: produce projected stat + empirical_over for each observation
# ──────────────────────────────────────────────────────────────────────────────

def _center_estimate(values, weights, center="mean"):
    """Central-tendency of prior-game values under the harness's recency weights.

    ``center="mean"`` (default) reproduces production exactly — the recency-weighted
    mean (EWMA). ``center="median"`` returns the recency-weighted 0.5-quantile
    (same weights, only the operator changes): robust to blow-up games and — the
    hypothesis under test — a less positively-biased center for right-skewed count
    stats. Falls back to the weighted mean if the quantile is undefined. Weights are
    guaranteed to sum > 0 by the caller (project_and_empirical returns early on
    sum(weights) <= 0), so the mean branch matches the prior inline formula bit-for-bit."""
    if center == "median":
        m = _weighted_quantile(values, weights, 0.5)
        if m is not None:
            return m
    tw = sum(weights)
    if tw <= 0:
        return sum(values) / len(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / tw


def _obs_season_defense(obs, team_defense, league_avg_def, defense_by_season):
    """Per-season pooled opponent-defense for one obs. When ``defense_by_season``
    (a {season-year-str: (team_defense, league_avg)} map) is supplied — the
    multi-season real-line refit path — return the obs's OWN season's pooled
    lookup so a prior-season obs never re-weights against another season's
    (including FUTURE) defense. This mirrors the synthetic sweep, which builds a
    per-season pooled lookup because it runs one season per backtest call. Falls
    back to the single passed lookup (non-baseball / single-season callers), which
    is byte-identical to the prior behavior."""
    if defense_by_season:
        return defense_by_season.get((obs.get("game_date") or "")[:4], ({}, None))
    return team_defense, league_avg_def


def project_and_empirical(obs, params, sport_key,
                          team_defense=None, league_avg_def=None,
                          xstats_strength=0.0, xba_index=None,
                          defense_by_season=None):
    """
    Mirrors backtest.run_player_props_backtest's per-obs projection logic,
    but takes the line from the book instead of synthetic.

    When ``xstats_strength > 0`` and a leakage-safe ``xba_index`` (an
    ``AsOfIndex`` keyed by batter MLBAM id, from
    ``backtest_props.build_batter_xba_index``) is supplied, the reconstructed
    projection is blended toward the batter's OWN as-of xBA × recent AB/game —
    exactly the P2.4a blend props.py applies live — so the re-fit residual
    distribution matches production. Uses a per-GAME as-of estimate (strictly
    before the obs's game_date), never the current-as-of SQL table. Fails open
    (unknown id / <MIN_BBE ABs / no AB data → unblended).

    Returns (projected, empirical_over) or (None, None) if data is too thin.
    """
    prior_games = obs["prior_games"]
    stat_label = obs["stat_label"]
    line = obs["line"]
    test_game = obs["test_game"]

    prior_values = [g.get(stat_label, 0.0) for g in prior_games]
    if stat_label == "IP":               # pitcher_outs: IP notation -> outs
        prior_values = [ip_to_outs(v) for v in prior_values]
    prior_minutes = [g.get("MIN", 0.0) for g in prior_games]
    prior_home_aways = [g.get("is_home") for g in prior_games]
    prior_opponents = [g.get("opponent") for g in prior_games]

    if any(prior_minutes):
        MIN_FLOOR = 10.0
        kept = [(v, m, ha, opp) for v, m, ha, opp in zip(
                    prior_values, prior_minutes,
                    prior_home_aways, prior_opponents)
                if (m or 0) >= MIN_FLOOR]
        if kept:
            prior_values = [v for v, _, _, _ in kept]
            prior_minutes = [m for _, m, _, _ in kept]
            prior_home_aways = [ha for _, _, ha, _ in kept]
            prior_opponents = [opp for _, _, _, opp in kept]

    if not prior_values:
        return None, None

    upcoming_is_home = test_game.get("is_home")

    hl = params.get("half_life")
    base_w = _recency_weights(len(prior_values), hl)
    venue_s = params.get("venue_strength", 0.0)
    def_s = params.get("opp_defense_strength", 0.0)
    # Per-season pooled defense (leakage guard): a multi-season refit passes
    # defense_by_season so this obs re-weights only against its OWN season.
    team_defense, league_avg_def = _obs_season_defense(
        obs, team_defense, league_avg_def, defense_by_season)

    weights = []
    for bw, ph, opp in zip(base_w, prior_home_aways, prior_opponents):
        w = bw * venue_mult(ph, upcoming_is_home, venue_s)
        if def_s > 0 and team_defense:
            opp_pa = _resolve_opp_pts_allowed(opp, team_defense)
            w *= opp_defense_mult(opp_pa, league_avg_def, def_s)
        weights.append(w)

    if sum(weights) <= 0:
        return None, None

    # Central-tendency operator (P2.x mean-vs-median experiment): "mean" (default)
    # reproduces production; "median" swaps to the recency-weighted 0.5-quantile.
    # The per-minute RATE path (NBA use_minutes) stays a mean — a rate is already a
    # ratio and its median is a different construct; the knob targets count props.
    center = params.get("center", "mean")
    if params.get("use_minutes"):
        rates = [v / m for v, m in zip(prior_values, prior_minutes) if m and m > 0]
        rate_weights = [w for w, m in zip(weights, prior_minutes) if m and m > 0]
        if rates and sum(rate_weights) > 0:
            per_min_rate = sum(v * w for v, w in zip(rates, rate_weights)) / sum(rate_weights)
            proj_min = sum(m * w for m, w in zip(prior_minutes, weights)) / sum(weights)
            projected = per_min_rate * proj_min
        else:
            projected = _center_estimate(prior_values, weights, center)
    else:
        projected = _center_estimate(prior_values, weights, center)

    # ── Statcast xBA blend (P2.4a, leakage-safe as-of) ──
    # Shrink the projection toward the batter's own as-of xBA × recent AB/game,
    # mirroring props._xstats_blend, so the re-fit residuals match production.
    # Only fires when a strength + as-of index are supplied (batter_hits). Uses a
    # per-game as-of estimate (< this obs's game_date); fails open. Note A
    # (empirical_over) is unaffected — it doesn't read `projected` — so this
    # changes only the B/C-method residual basis.
    if (xstats_strength and xstats_strength > 0 and xba_index is not None
            and len(prior_games) == len(weights)):
        try:
            import mlb_starters
            game_date = obs.get("game_date")
            player = obs.get("player")
            season = int(str(game_date)[:4]) if game_date else None
            pid_info = (mlb_starters.find_player_id(player, season)
                        if (player and season) else None)
            if pid_info and pid_info[0] and not pid_info[1]:   # batter only
                xba = xba_index.asof_mean(str(pid_info[0]), game_date)  # min_bbe=40
                ab_valid = [(g.get("AB"), w) for g, w in zip(prior_games, weights)
                            if g.get("AB") and w > 0]
                if xba is not None and ab_valid:
                    ab_pg = (sum(ab * w for ab, w in ab_valid)
                             / sum(w for _, w in ab_valid))
                    if ab_pg > 0:
                        wgt = max(0.0, min(1.0, xstats_strength))
                        projected = (1.0 - wgt) * projected + wgt * (xba * ab_pg)
        except Exception:
            pass  # fail open — never let the xBA lookup break the refit

    # ── Candidate-feature multiplier (roadmap §2.6) ──
    # Bounded projection multiplier (e.g. rest/days-off) from prop_features — the
    # ONE source shared with the runtime + synthetic sweep. Empty / all-off ->
    # 1.0, so the projection AND the line stay byte-identical to production (the
    # self-check the diagnostic relies on). Mirrors props.py combined_mult: scale
    # the projection AND divide the line, so methods A-E all move (unlike the
    # `center` operator, which only touched the mean).
    feat_strengths = prop_features.strengths_from_params(params)
    feat_mult = 1.0
    if feat_strengths:
        fm = prop_features.projection_multiplier(
            obs.get("prop_key"), feat_strengths,
            [g.get("game_date") for g in prior_games], obs.get("game_date"),
            gamecontext_factors=obs.get("gc_factor"),
            platoon_factor=obs.get("platoon_factor"))
        if fm:
            feat_mult = fm

    # ── Park-factor road-context delta (P1.2), mirroring production combined_mult ──
    # Reconstruct production's park_mult offline so the re-fit residual basis (and
    # the shifted comparison line) match the LIVE projection, which already scales
    # batter_hits / pitcher_earned_runs by park via combined_mult -> avg_stat /
    # effective_line (props.py:1213-1258). Uses props._park_factor_mult -- the SAME
    # function the runtime calls -- with the identical past-park reconstruction:
    # each past game's park is the player's own team's park when he was home, else
    # the opponent's park; the upcoming park is always the home team's. Strictly
    # leakage-safe (static park table, no future data). park_factor_strength is not
    # a swept knob and no MLB prop overrides it, so this resolves to the sport
    # default (1.0 for MLB) -- matching production's _knob fallback. Fails open to
    # 1.0 for park-neutral props, prediction-log rows (home_team None), unknown
    # parks, or any error -- never breaks the refit.
    park_mult = 1.0
    try:
        import props
        park_strength = params.get("park_factor_strength")
        if park_strength is None:
            park_strength = props._player_prop_park_strength(sport_key)
        if park_strength and park_strength > 0:
            home_team = obs.get("home_team")
            player_team_name = (
                home_team if upcoming_is_home is True
                else obs.get("away_team") if upcoming_is_home is False
                else None)
            past_parks = []
            for ph, opp in zip(prior_home_aways, prior_opponents):
                if ph is True:
                    past_parks.append(player_team_name)
                elif ph is False:
                    past_parks.append(opp)
                else:
                    past_parks.append(None)
            park_mult, _pm = props._park_factor_mult(
                obs.get("prop_key"), past_parks, weights, home_team,
                park_strength)
    except Exception:
        park_mult = 1.0   # fail open -- never let park reconstruction break the refit

    # Fold the feature + park multipliers into the projection AND the comparison
    # line (mirrors props.py combined_mult: scale the projection, divide the line).
    # Both == 1.0 -> projection AND line stay byte-identical to production (the
    # diagnostic self-check), so methods A-E are unchanged wherever no factor fires.
    combined_mult = feat_mult * park_mult
    line_eff = line
    if combined_mult and combined_mult != 1.0:
        projected *= combined_mult
        line_eff = line / combined_mult

    # Empirical over-probability AT THE (feature + park-shifted) BOOK LINE
    empirical_over = _weighted_rate(
        prior_values, weights, lambda v: v > line_eff)

    return projected, empirical_over


def project_distributional(obs, params, sport_key, team_defense=None,
                           league_avg_def=None, xba_index=None,
                           quality_index=None, xstats_strength=0.0,
                           hardhit_coef=None, barrel_coef=None,
                           xba_window=None, xba_min_count=None,
                           home_ab_delta=0.0, defense_by_season=None):
    """Distributional P(over) for a batter_hits obs at its REAL book line
    (§2.4b-2 method "D"), or None if not applicable / too thin.

    Mirrors project_and_empirical's recency×venue×opp_defense weighting, derives
    the weighted per-AB hit rate + expected AB from the prior games, looks up the
    batter's leakage-safe as-of xBA / contact-quality (per-game indices, strictly
    before obs["game_date"]), and returns ``props._dist_p_over`` — the SAME
    composite the runtime uses, so the two can't drift. Offline eval applies NO
    output (park/weather/matchup) multipliers — rate_mult = exposure_mult = 1,
    the same limitation the C-method real-line fit has. Fails open."""
    if obs.get("prop_key") != "batter_hits":
        return None
    prior_games = obs["prior_games"]
    if not prior_games:
        return None
    line = obs["line"]
    upcoming_is_home = obs["test_game"].get("is_home")
    prior_hits = [g.get("H") for g in prior_games]
    prior_ab = [g.get("AB") for g in prior_games]
    prior_home_aways = [g.get("is_home") for g in prior_games]
    prior_opponents = [g.get("opponent") for g in prior_games]

    base_w = _recency_weights(len(prior_games), params.get("half_life"))
    venue_s = params.get("venue_strength", 0.0)
    def_s = params.get("opp_defense_strength", 0.0)
    team_defense, league_avg_def = _obs_season_defense(
        obs, team_defense, league_avg_def, defense_by_season)
    weights = []
    for bw, ph, opp in zip(base_w, prior_home_aways, prior_opponents):
        w = bw * venue_mult(ph, upcoming_is_home, venue_s)
        if def_s > 0 and team_defense:
            opp_pa = _resolve_opp_pts_allowed(opp, team_defense)
            w *= opp_defense_mult(opp_pa, league_avg_def, def_s)
        weights.append(w)

    hits_w = ab_w = w_valid = 0.0
    for h, ab, w in zip(prior_hits, prior_ab, weights):
        if ab is None or ab <= 0 or w <= 0 or h is None:
            continue
        hits_w += w * h
        ab_w += w * ab
        w_valid += w
    if ab_w <= 0 or w_valid <= 0:
        return None
    r_emp = hits_w / ab_w
    expected_ab = ab_w / w_valid
    if expected_ab <= 0:
        return None
    # Experiment: home batters get slightly fewer plate appearances (the home
    # team skips the bottom 9th when leading). ``home_ab_delta`` nudges the
    # binomial n for a home game; 0.0 = off. Floored so n stays positive.
    if home_ab_delta and obs.get("test_game", {}).get("is_home"):
        expected_ab = max(0.5, expected_ab + home_ab_delta)

    xba = hh = brl = None
    try:
        import mlb_starters
        game_date = obs.get("game_date")
        player = obs.get("player")
        season = int(str(game_date)[:4]) if game_date else None
        pid_info = (mlb_starters.find_player_id(player, season)
                    if (player and season) else None)
        if pid_info and pid_info[0] and not pid_info[1]:   # batter only
            pid = str(pid_info[0])
            if xba_index is not None:
                if xba_window:            # rolling last-N-BBE xBA vs season-to-date
                    xba = xba_index.asof_window_mean(
                        pid, game_date, xba_window, xba_min_count or 1)
                else:
                    xba = xba_index.asof_mean(pid, game_date)
            if quality_index is not None:
                q = quality_index.asof(pid, game_date)
                if q:
                    hh = q.get("hard_hit_pct")
                    brl = q.get("barrel_pct")
    except Exception:
        xba = hh = brl = None   # fail open

    import props
    p_over, _ = props._dist_p_over(
        r_emp, expected_ab, xba, hh, brl, 1.0, 1.0, line,
        xstats_strength, hardhit_coef, barrel_coef)
    return p_over


# ──────────────────────────────────────────────────────────────────────────────
#  Step 4: forecaster comparison with chronological holdout
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_calibration(per_prop_obs, prop_key, label, shrinkage_k=15):
    """
    per_prop_obs: list of dicts with keys:
      player, projected, line, actual, empirical_over, game_date

    Returns metrics for five forecasters: A (empirical), B (pooled Gaussian),
    C (pooled ECDF), B* (per-player Gaussian + shrinkage), C* (per-player
    ECDF + shrinkage).
    """
    rows = [r for r in per_prop_obs if r["actual"] != r["line"]]
    if len(rows) < 20:
        return None

    rows.sort(key=lambda r: r["game_date"])
    split = len(rows) // 2
    train = rows[:split]
    test = rows[split:]

    # ── Pool stats (train only) ──
    train_resid = [r["actual"] - r["projected"] for r in train]
    mu_pool = sum(train_resid) / len(train_resid)
    var_pool = sum((r - mu_pool) ** 2 for r in train_resid) / len(train_resid)
    sigma_pool = math.sqrt(var_pool) if var_pool > 0 else 1e-6
    sorted_pool = sorted(train_resid)

    # ── Per-player stats (train only) ──
    player_resid = {}
    for r in train:
        player_resid.setdefault(r["player"], []).append(r["actual"] - r["projected"])
    player_stats = _per_player_stats(player_resid)
    player_sorted = {p: sorted(rs) for p, rs in player_resid.items()}

    pA, pB, pC, pBs, pCs, outcomes = [], [], [], [], [], []
    for r in test:
        o = 1 if r["actual"] > r["line"] else 0
        outcomes.append(o)
        pA.append(max(0.0, min(1.0, r["empirical_over"])))

        # B (pooled Gaussian)
        corrected_pool = r["projected"] + mu_pool
        z = (corrected_pool - r["line"]) / sigma_pool if sigma_pool > 0 else 0.0
        pB.append(_norm_cdf(z))

        # C (pooled ECDF)
        pC.append(1.0 - _empirical_cdf(sorted_pool, r["line"] - corrected_pool))

        # B* per-player Gaussian + shrinkage
        if r["player"] in player_stats:
            n_p, mu_p, sigma_p = player_stats[r["player"]]
            mu_s = _shrunk(mu_p, mu_pool, n_p, shrinkage_k)
            sigma_s = _shrunk(sigma_p, sigma_pool, n_p, shrinkage_k)
        else:
            mu_s, sigma_s = mu_pool, sigma_pool
        corrected_s = r["projected"] + mu_s
        z_s = (corrected_s - r["line"]) / sigma_s if sigma_s > 0 else 0.0
        pBs.append(_norm_cdf(z_s))

        # C* per-player ECDF blended with pool by λ
        if r["player"] in player_sorted:
            n_p = len(player_sorted[r["player"]])
            lam = n_p / (n_p + shrinkage_k)
            f_player = _empirical_cdf(player_sorted[r["player"]],
                                      r["line"] - corrected_s)
            f_pool = _empirical_cdf(sorted_pool, r["line"] - corrected_s)
            f_blend = lam * f_player + (1 - lam) * f_pool
        else:
            f_blend = _empirical_cdf(sorted_pool, r["line"] - corrected_s)
        pCs.append(1.0 - f_blend)

    return {
        "n_test": len(test),
        "n_train": len(train),
        "mu_r": mu_pool,
        "sigma_r": sigma_pool,
        "n_players": len(player_stats),
        "brier": (_brier(pA, outcomes), _brier(pB, outcomes), _brier(pBs, outcomes),
                  _brier(pC, outcomes), _brier(pCs, outcomes)),
        "hit":   (_hit_rate(pA, outcomes), _hit_rate(pB, outcomes), _hit_rate(pBs, outcomes),
                  _hit_rate(pC, outcomes), _hit_rate(pCs, outcomes)),
        "probs": {"A": pA, "B": pB, "B*": pBs, "C": pC, "C*": pCs},
        "outcomes": outcomes,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Step 5: re-select the deployed calibration METHOD at REAL book lines
#          (roadmap 0.3 — the synthetic-vs-real-line fix)
#
#  refit_calibration.py chooses the A/B/C method by Brier at a SYNTHETIC
#  season-average line, but props.py applies it at the REAL book line. Residual
#  mu/sigma/ecdf are line-invariant, so the only thing that must be re-decided at
#  real lines is the A-vs-B/C method choice (method A's empirical over-rate is
#  already recomputed at the real line at runtime). These helpers mirror the gate
#  in refit_calibration._best_per_prop / _confirms_over_baseline but score every
#  method at the real book line carried on each row.
# ──────────────────────────────────────────────────────────────────────────────

def build_real_line_obs(enriched, params, sport_key, prop_key,
                        team_defense=None, league_avg_def=None,
                        xstats_strength=0.0, xba_index=None,
                        defense_by_season=None):
    """Project every joined observation for one prop at its REAL book line.

    Returns a list of row dicts {player, projected, line, actual,
    empirical_over, game_date}, reusing project_and_empirical (which evaluates
    the empirical over-rate at obs["line"], the book line). ``xstats_strength`` +
    ``xba_index`` opt into the P2.4a xBA projection blend (batter_hits).
    """
    rows = []
    for obs in enriched:
        if obs["prop_key"] != prop_key:
            continue
        projected, emp = project_and_empirical(
            obs, params, sport_key, team_defense, league_avg_def,
            xstats_strength=xstats_strength, xba_index=xba_index,
            defense_by_season=defense_by_season)
        if projected is None:
            continue
        rows.append({
            "player": obs["player"],
            "projected": projected,
            "line": obs["line"],
            "actual": obs["actual"],
            "empirical_over": emp,
            "game_date": obs["game_date"],
            # Book prices (de-vigged consensus) for the ROI tiebreaker; None when
            # the obs is prediction-log-sourced. select_method_at_real_lines uses
            # them only inside the noise band and no-ops when absent.
            "over_price": obs.get("over_price"),
            "under_price": obs.get("under_price"),
        })
    return rows


def _fit_negbin_real(rows):
    """Fit the §2.2 NegBin method-"E" params on a set of real-line obs:
    ``(mean_scale, dispersion)`` or None when unusable.

    ``mean_scale = sum(actual) / sum(projected)`` clamped to [0.5, 2.0] — the MLE
    mean-scale for a proportional count mean and the multiplicative analog of
    method B's additive ``residual_mu`` (multiplicative keeps the mean > 0 at low
    counts, where an additive shift could go negative and break the NegBin
    variance=mean+phi*mean^2 link). ``dispersion`` is fit by
    ``stats.fit_negbin_dispersion`` on the SCALED means. Leakage-safe: the caller
    fits on the train split (or all usable obs for the deployed params) and scores
    on the held-out split, exactly like the B/C residual fit.

    Thin adapter over the shared ``stats.fit_negbin_params`` (one fit impl for both
    the real-line selector and the synthetic sweep — see that docstring)."""
    return fit_negbin_params([(r.get("projected"), r.get("actual")) for r in rows])


def _score_abc_real(train, test, negbin_eligible=False):
    """Fit pooled residuals on `train`, score methods A/B/C (and D/E when
    applicable) on `test` at each row's REAL book line. Returns ({method: brier},
    (mu, sigma, sorted), {method: [p per test row]}, [0/1 outcome per test row]).

    The per-row ``probs`` + ``out`` are the same P(over) vectors used to compute
    the Briers, exposed so the caller can replay them through the ROI tiebreaker
    (single source of truth for the prob math — no duplication in the ROI path).

    A = empirical over-rate passthrough; B = pooled Gaussian residual;
    C = pooled residual ECDF. D (distributional, §2.4b-2) is scored only when
    every test row carries a precomputed leakage-safe ``p_dist`` — D needs no
    train fit (its prob is a closed form on each obs's own as-of stats), so a
    row's ``p_dist`` is split-independent. E (§2.2 Negative Binomial) IS train-fit
    (mean_scale + dispersion on `train`, like B/C), scored only when
    ``negbin_eligible`` — a count prop whitelisted by the caller. Same math as
    evaluate_calibration, factored so the single split and the confirmation folds
    share one impl."""
    resid = [r["actual"] - r["projected"] for r in train]
    mu = sum(resid) / len(resid)
    var = sum((x - mu) ** 2 for x in resid) / len(resid)
    sigma = math.sqrt(var) if var > 0 else 1e-6
    srt = sorted(resid)
    nb = _fit_negbin_real(train) if negbin_eligible else None
    pA, pB, pC, pD, pE, out = [], [], [], [], [], []
    has_d = bool(test) and all(r.get("p_dist") is not None for r in test)
    for r in test:
        out.append(1 if r["actual"] > r["line"] else 0)
        pA.append(max(0.0, min(1.0, r["empirical_over"])))
        corrected = r["projected"] + mu
        z = (corrected - r["line"]) / sigma if sigma > 0 else 0.0
        pB.append(_norm_cdf(z))
        pC.append(1.0 - _empirical_cdf(srt, r["line"] - corrected))
        if has_d:
            pD.append(r["p_dist"])
        if nb is not None:
            mean_scale, disp = nb
            mean = max(1e-9, mean_scale * r["projected"])
            pE.append(negbin_at_least(int(r["line"]) + 1, mean, disp))
    scores = {"A": _brier(pA, out), "B": _brier(pB, out), "C": _brier(pC, out)}
    probs = {"A": pA, "B": pB, "C": pC}
    if has_d:
        scores["D"] = _brier(pD, out)
        probs["D"] = pD
    if nb is not None:
        scores["E"] = _brier(pE, out)
        probs["E"] = pE
    return scores, (mu, sigma, srt), probs, out


def _real_line_folds(rows, min_set_n=20):
    """Two expanding-train chronological folds over real-line rows (dict shape).

    Mirrors refit_calibration._chronological_folds: cut at 60%/80% of the
    date-sorted rows, each fold with strictly earlier train than test and
    >= min_set_n in both sets. Returns [] when the data can't form two folds.
    """
    rows = sorted(rows, key=lambda r: r["game_date"])
    n = len(rows)
    if n < 3 * min_set_n:
        return []
    cut1 = rows[int(n * 0.6)]["game_date"]
    cut2 = rows[int(n * 0.8)]["game_date"]
    if not cut1 or not cut2 or cut1 == cut2:
        return []
    folds = [
        ([r for r in rows if r["game_date"] < cut1],
         [r for r in rows if cut1 <= r["game_date"] < cut2]),
        ([r for r in rows if r["game_date"] < cut2],
         [r for r in rows if r["game_date"] >= cut2]),
    ]
    for fit_rows, score_rows in folds:
        if len(fit_rows) < min_set_n or len(score_rows) < min_set_n:
            return []
    return folds


def _roi_tiebreak(test, single_probs, single_out, tie_methods, brier_leader,
                  threshold, min_bets, min_roi_gain):
    """Break a Brier near-tie between calibration methods by realized flat-1u ROI.

    Replays each tied method's P(over) vector — already computed on the SAME
    holdout that produced the Brier scores (leakage-consistent) — through the LIVE
    edge+EV recommendation gate (refit_calibration._roi_sim_method) at de-vigged
    CONSENSUS prices. An override is recommended ONLY when the ROI winner differs
    from ``brier_leader``, both clear ``min_bets`` simulated value bets, it beats
    the leader's ROI by >= ``min_roi_gain``, AND the winner is itself projected
    profitable (roi > 0). The last guard means a Brier-tie is never broken toward
    a money-losing method: when no tied method is profitable there is nothing to
    select FOR, so calibration (Brier) keeps the pick. Consensus prices mean only
    the RELATIVE ranking is trusted, never the absolute ROI.

    Returns a record dict {ran, winner, applied, rois, n_bets} for the caller to
    act on and log, or None when there are no usable prices (tiebreak inert)."""
    from refit_calibration import _roi_sim_method
    from odds_client import (american_to_implied_prob, devig_two_way,
                             american_to_decimal)

    # Price each test row once (shared across methods); carry the row index so a
    # method's precomputed P(over) is looked up per row inside the sim.
    priced = []
    for i, r in enumerate(test):
        op, up = r.get("over_price"), r.get("under_price")
        if op is None or up is None:
            continue
        priced.append({
            "_i": i,
            "o": single_out[i],
            "mkt_over": devig_two_way(american_to_implied_prob(op),
                                      american_to_implied_prob(up))[0],
            "over_price": op, "under_price": up,
            "over_dec": american_to_decimal(op),
            "under_dec": american_to_decimal(up),
        })
    if not priced:
        return None

    sims = {}
    for m in tie_methods:
        pv = single_probs.get(m)
        if pv is None:
            continue
        sims[m] = _roi_sim_method(priced, lambda r, _pv=pv: _pv[r["_i"]], threshold)

    record = {
        "ran": True, "brier_leader": brier_leader, "winner": brier_leader,
        "applied": False,
        "rois": {m: (sims[m]["roi"] if m in sims else None) for m in tie_methods},
        "n_bets": {m: (sims[m]["n_bets"] if m in sims else 0) for m in tie_methods},
    }
    leader = sims.get(brier_leader)
    if not leader or leader["roi"] is None or leader["n_bets"] < min_bets:
        return record  # leader itself untradeable on this split -> keep Brier pick

    best_m, best_roi = brier_leader, leader["roi"]
    for m, s in sims.items():
        if m == brier_leader or s["roi"] is None or s["n_bets"] < min_bets:
            continue
        if s["roi"] > best_roi:
            best_m, best_roi = m, s["roi"]
    if (best_m != brier_leader and best_roi - leader["roi"] >= min_roi_gain
            and best_roi > 0.0):
        record["winner"], record["applied"] = best_m, True
    return record


def select_method_at_real_lines(rows, shrinkage_k=15, negbin_eligible=False,
                                roi_tiebreak=True):
    """Choose the deployed A/B/C method at REAL book lines, gated like refit.

    Method A (empirical) is the safe baseline and always eligible. A non-empirical
    method (B/C, and E when ``negbin_eligible``) is chosen only if it beats A on
    the single chronological holdout by >= MIN_CALIB_BRIER_GAIN AND beats A in BOTH
    expanding confirmation folds (defeats winner's-curse). Residuals for the winner
    are fit on ALL usable obs (the deployed distribution), matching
    refit_calibration._fit_residuals; when E wins, its ``mean_scale``/``dispersion``
    are likewise fit on ALL usable obs and returned.

    ``negbin_eligible`` admits the §2.2 Negative-Binomial count method "E" as a
    candidate (a count prop whitelisted by the caller via
    props.PROP_NEGBIN_ELIGIBLE). Off by default so existing callers are unchanged.

    ``roi_tiebreak`` (default True) breaks a Brier near-tie (methods within
    MIN_CALIB_BRIER_GAIN of each other on the single holdout, all confirmed
    out-of-sample) by realized ROI through the live edge+EV gate — see
    _roi_tiebreak. An ROI override is honored only if the winner ALSO beats the
    Brier leader by the ROI margin in BOTH expanding folds (a cross-fold guard
    mirroring the Brier gate's _confirms). Pass False for a pure-Brier view
    (e.g. diagnose_negbin).

    Returns None if fewer than 20 usable (actual != line) observations, else:
      {method, fit_brier, baseline_brier, cv_brier, confirmed,
       residual_mu, residual_sigma, residual_ecdf, n_obs, single_split,
       roi_tiebreak[, mean_scale, dispersion]}
    """
    from refit_calibration import (  # single source of truth
        MIN_CALIB_BRIER_GAIN, ROI_TIEBREAK_MIN_BETS, ROI_TIEBREAK_MIN_ROI_GAIN,
        ROI_TIEBREAK_THRESHOLD)

    usable = [r for r in rows if r["actual"] != r["line"]]
    if len(usable) < 20:
        return None
    usable.sort(key=lambda r: r["game_date"])

    # Single chronological holdout: baseline (A) + candidate (B/C/E) Briers, plus
    # the per-method P(over) vectors + outcomes (reused by the ROI tiebreak).
    split = len(usable) // 2
    single, _, single_probs, single_out = _score_abc_real(
        usable[:split], usable[split:], negbin_eligible)
    baseline = single["A"]

    # Two-fold out-of-sample confirmation for non-empirical methods. Keep the FULL
    # per-fold tuple (scores, _, probs, out); fold_scores stays value-identical for
    # _confirms/cv_brier, and the retained probs/out feed the ROI cross-fold guard
    # below (same number of _score_abc_real calls as before — zero added compute).
    folds = _real_line_folds(usable)
    fold_full = ([_score_abc_real(tr, te, negbin_eligible)
                  for tr, te in folds] if folds else [])
    fold_scores = [ff[0] for ff in fold_full]

    def _confirms(method):
        if not fold_scores:
            return False
        return all(fs.get("A") is not None and fs.get(method) is not None
                   and fs[method] < fs["A"] for fs in fold_scores)

    # D (distributional) is a candidate only when the rows carry a leakage-safe
    # p_dist (the per-bucket line-conditional path supplies it); the pooled A/B/C
    # callers pass rows without it, so their behavior is unchanged. E (NegBin) is a
    # candidate only when the caller whitelists this count prop.
    candidate_methods = ["B", "C"]
    if any(r.get("p_dist") is not None for r in usable):
        candidate_methods.append("D")
    if negbin_eligible:
        candidate_methods.append("E")
    best_method, best_brier = "A", baseline
    for method in candidate_methods:
        cand = single.get(method)
        if cand is None or baseline is None:
            continue
        if baseline - cand < MIN_CALIB_BRIER_GAIN:
            continue
        if not _confirms(method):
            continue
        if best_brier is None or cand < best_brier:
            best_method, best_brier = method, cand

    # ROI tiebreak within the Brier noise band. Eligible = A + every candidate that
    # PASSED 2-fold confirmation (never an unconfirmed method). When >=2 of those
    # land within MIN_CALIB_BRIER_GAIN of the best eligible Brier, realized ROI
    # arbitrates — this can rescue a confirmed method that beat A in both folds but
    # missed the single-split margin (best_method is provably always in tie_set).
    roi_record = None
    if roi_tiebreak:
        eligible = ["A"] + [m for m in candidate_methods
                            if single.get(m) is not None and _confirms(m)]
        b_min = min(single[m] for m in eligible)
        tie_set = [m for m in eligible if single[m] - b_min < MIN_CALIB_BRIER_GAIN]
        if len(tie_set) >= 2:
            rec = _roi_tiebreak(
                usable[split:], single_probs, single_out, tie_set, best_method,
                ROI_TIEBREAK_THRESHOLD, ROI_TIEBREAK_MIN_BETS,
                ROI_TIEBREAK_MIN_ROI_GAIN)
            if rec:
                rec["tie_set"] = tie_set
                roi_record = rec
                if rec["applied"]:
                    # Cross-fold ROI guard — the ROI analogue of _confirms: a
                    # single-split ROI override is honored only if the SAME winner
                    # also beats the Brier leader by the ROI margin in BOTH expanding
                    # folds (defeats the winner's-curse the Brier gate already guards
                    # against; consensus n_bets can be as low as 15, so a lone-split
                    # ROI edge is exactly the noise to reconfirm). Else keep the
                    # Brier pick. best_method here is still the pre-ROI Brier leader.
                    winner = rec["winner"]
                    fold_recs = [
                        _roi_tiebreak(te, fp, fo, tie_set, best_method,
                                      ROI_TIEBREAK_THRESHOLD, ROI_TIEBREAK_MIN_BETS,
                                      ROI_TIEBREAK_MIN_ROI_GAIN)
                        for (_tr, te), (_fs, _f2, fp, fo) in zip(folds, fold_full)]
                    rec["fold_confirmed"] = bool(fold_full) and all(
                        fr and fr.get("applied") and fr.get("winner") == winner
                        for fr in fold_recs)
                    rec["fold_recs"] = fold_recs
                    if rec["fold_confirmed"]:
                        best_method = winner
                        best_brier = single.get(best_method, best_brier)
                    else:
                        rec["applied"] = False

    # Deployed residual distribution: fit on ALL usable obs.
    resid = [r["actual"] - r["projected"] for r in usable]
    mu = sum(resid) / len(resid)
    var = sum((x - mu) ** 2 for x in resid) / len(resid)
    sigma = math.sqrt(var) if var > 0 else 0.0

    cv_brier = None
    if fold_scores:
        vals = [fs.get(best_method) for fs in fold_scores]
        if all(v is not None for v in vals):
            cv_brier = round(sum(vals) / len(vals), 4)

    result = {
        "method": best_method,
        "fit_brier": round(best_brier, 4) if best_brier is not None else None,
        "baseline_brier": round(baseline, 4) if baseline is not None else None,
        "cv_brier": cv_brier,
        "confirmed": best_method != "A",
        "residual_mu": mu,
        "residual_sigma": sigma,
        "residual_ecdf": sorted(resid),
        "n_obs": len(usable),
        # Per-method Brier on the single holdout — lets a per-bucket caller
        # compare its winner against the POOLED method on the same split.
        "single_split": single,
        # ROI-tiebreak audit trail (None when it didn't run): {ran, winner,
        # applied, rois, n_bets, tie_set[, fold_confirmed, fold_recs]}. When a
        # single-split override is proposed, fold_confirmed records whether it
        # survived the 2-fold ROI guard (applied is forced False if not).
        # Diagnostic only — not persisted to JSON.
        "roi_tiebreak": roi_record,
    }
    # When E wins, its deployed params are fit on ALL usable obs (mirrors the
    # residual block above). Always fit when eligible so the diagnostic can report
    # them even for a prop that doesn't flip.
    if negbin_eligible:
        nb = _fit_negbin_real(usable)
        if nb is not None:
            result["mean_scale"], result["dispersion"] = nb
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Confidence-threshold "precision vs coverage" report
# ──────────────────────────────────────────────────────────────────────────────

def confidence_threshold_table(probs, outcomes, thresholds=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)):
    """
    For each threshold T, simulate "only bet when max(p, 1-p) >= T".
    Picked side is OVER if p >= 0.5 else UNDER. Hit when picked side matches
    the actual outcome (outcomes[i] == 1 means actual > line).
    Returns list of (T, n_actionable, n_hits, hit_rate, pct_of_total).
    """
    total = len(outcomes)
    rows = []
    for T in thresholds:
        n = 0
        hits = 0
        for p, o in zip(probs, outcomes):
            conf = max(p, 1.0 - p)
            if conf < T:
                continue
            n += 1
            pick = 1 if p >= 0.5 else 0
            if pick == o:
                hits += 1
        rate = (hits / n) if n else 0.0
        cov = (n / total) if total else 0.0
        rows.append((T, n, hits, rate, cov))
    return rows


def print_confidence_report(per_variant_prop_results):
    """
    per_variant_prop_results: dict (variant_name, prop_key) -> evaluate_calibration result
    """
    print("\n=== Confidence-threshold hit rates (per forecaster) ===")
    print("'If we only bet when model confidence (max(p,1-p)) >= T, what % of the time")
    print(" did the chosen side win?'  Holdout-only (no leakage).\n")

    forecasters = ["A", "B", "B*", "C", "C*"]
    header = (f"{'Variant':<10} {'Prop':<22} {'Fcst':<5} "
              f"{'T':>5} {'N':>4} {'Hits':>5} {'Hit%':>6} {'Cov%':>6}")
    print(header)
    print("-" * len(header))

    for (vname, prop_key), res in per_variant_prop_results.items():
        outcomes = res["outcomes"]
        for fcst in forecasters:
            probs = res["probs"][fcst]
            rows = confidence_threshold_table(probs, outcomes)
            for T, n, hits, rate, cov in rows:
                if n == 0:
                    continue
                print(f"{vname:<10} {prop_key:<22} {fcst:<5} "
                      f"{T:>5.2f} {n:>4d} {hits:>5d} "
                      f"{rate * 100:>5.1f}% {cov * 100:>5.1f}%")
            print()


def simulate_safe_mode(obs, params, sport_key, safe_target,
                       team_defense=None, league_avg_def=None):
    """
    Replicate analysis.py's safe-mode pipeline for one (player, prop, game)
    observation and report what the production code would have done.

    Returns dict:
      {
        'eligible':       bool,   # passed both production guards
        'filter_reason':  str|None,
        'safe_threshold': int|None,
        'p_at_safe':      float|None,    # model-claimed hit prob
        'hit':            bool|None,     # actual >= safe_threshold
        'line':           float,         # book line (reference)
      }
    """
    prior_games = obs["prior_games"]
    stat_label = obs["stat_label"]
    line = obs["line"]
    test_game = obs["test_game"]
    actual = obs["actual"]

    prior_values = [g.get(stat_label, 0.0) for g in prior_games]
    if stat_label == "IP":               # pitcher_outs: IP notation -> outs
        prior_values = [ip_to_outs(v) for v in prior_values]
    prior_minutes = [g.get("MIN", 0.0) for g in prior_games]
    prior_home_aways = [g.get("is_home") for g in prior_games]
    prior_opponents = [g.get("opponent") for g in prior_games]

    # Same DNP-floor filter as production safe-mode prep
    if any(prior_minutes):
        kept = [(v, m, ha, opp) for v, m, ha, opp in zip(
                    prior_values, prior_minutes,
                    prior_home_aways, prior_opponents)
                if (m or 0) >= 10.0]
        if kept:
            prior_values = [v for v, _, _, _ in kept]
            prior_minutes = [m for _, m, _, _ in kept]
            prior_home_aways = [ha for _, _, ha, _ in kept]
            prior_opponents = [opp for _, _, _, opp in kept]

    if not prior_values:
        return {"eligible": False, "filter_reason": "no_prior", "safe_threshold": None,
                "p_at_safe": None, "hit": None, "line": line}

    upcoming_is_home = test_game.get("is_home")

    hl = params.get("half_life")
    base_w = _recency_weights(len(prior_values), hl)
    venue_s = params.get("venue_strength", 0.0)
    def_s = params.get("opp_defense_strength", 0.0)
    output_def_s = params.get("output_def_strength", 0.0)
    shrinkage_k = params.get("shrinkage_k", 0) or 0

    weights = []
    for bw, ph, opp in zip(base_w, prior_home_aways, prior_opponents):
        w = bw * venue_mult(ph, upcoming_is_home, venue_s)
        if def_s > 0 and team_defense:
            opp_pa = _resolve_opp_pts_allowed(opp, team_defense)
            w *= opp_defense_mult(opp_pa, league_avg_def, def_s)
        weights.append(w)

    if sum(weights) <= 0:
        return {"eligible": False, "filter_reason": "zero_weights", "safe_threshold": None,
                "p_at_safe": None, "hit": None, "line": line}

    # Output-defense multiplier (uses test_game's opponent)
    output_def_mult = 1.0
    opp_name = test_game.get("opponent")
    if output_def_s > 0 and team_defense and league_avg_def and opp_name:
        opp_pa = team_defense.get(opp_name)
        if opp_pa:
            output_def_mult = 1.0 + output_def_s * (
                opp_pa / league_avg_def - 1.0)

    # Bayesian shrinkage toward unweighted mean
    base_proj = sum(v * w for v, w in zip(prior_values, weights)) / sum(weights)
    if shrinkage_k > 0:
        unweighted = sum(prior_values) / len(prior_values)
        eff_n = sum(weights)
        if eff_n + shrinkage_k > 0:
            base_proj = ((eff_n * base_proj) + (shrinkage_k * unweighted)) / (eff_n + shrinkage_k)
    avg_stat = base_proj * output_def_mult

    # Parametric quantile floor (mirrors analysis.py exactly)
    wstd = _weighted_std(prior_values, weights, mean=base_proj)
    wstd_adj = wstd * (output_def_mult if output_def_mult else 1.0)
    z = _normal_inv_cdf(safe_target)
    alt_q = avg_stat - z * wstd_adj
    safe_threshold = max(1, int(math.floor(alt_q)))

    # Production sanity guard (mirrors analysis.py): drop if historical
    # hit rate at the suggested threshold is more than 5pp below target.
    p_at_safe = _weighted_rate(prior_values, weights,
                               lambda v, t=safe_threshold: v >= t)
    if p_at_safe < (safe_target - 0.05):
        return {"eligible": False, "filter_reason": "p_at_safe<target-0.05",
                "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
                "hit": None, "line": line}

    # Floor-collapse guard (mirrors analysis.py): reject "1+" type bets
    # for safe_target > 0.80 — parametric Normal under-estimates variance
    # for low-mean integer distributions and the forward hit rate is
    # systematically below the claimed safe_target.
    if safe_threshold <= 1 and safe_target > 0.80:
        return {"eligible": False, "filter_reason": "floor_collapse",
                "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
                "hit": None, "line": line}

    SAFE_MIN_RATIO = 0.5
    if line > 0 and safe_threshold < line * SAFE_MIN_RATIO:
        return {"eligible": False, "filter_reason": "threshold<50%_of_book",
                "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
                "hit": None, "line": line}

    hit = float(actual) >= safe_threshold
    return {"eligible": True, "filter_reason": None,
            "safe_threshold": safe_threshold, "p_at_safe": p_at_safe,
            "hit": hit, "line": line}


def print_safe_mode_report(enriched, params, sport_key,
                           team_defense=None, league_avg_def=None,
                           targets=(0.85, 0.90, 0.95)):
    """
    For each safe_target, simulate production safe-mode on every observation,
    then bucket eligible suggestions per prop and report actual hit rate.
    """
    print("\n=== Safe-mode actual hit rates (per prop, per safe_target) ===")
    print("'If we'd bet the SAFE alt-line production would have suggested,")
    print(" how often did the player actually clear that line?'\n")

    by_prop = defaultdict(list)
    for obs in enriched:
        by_prop[obs["prop_key"]].append(obs)

    header = (f"{'Prop':<22} {'Target':>7} {'Eligible':>8} {'Hits':>5} "
              f"{'Hit%':>6} {'Filtered':>9} "
              f"{'Avg gap':>8} {'Median th':>10}")
    print(header)
    print("-" * len(header))
    for prop_key in sorted(by_prop.keys()):
        rows = by_prop[prop_key]
        for tgt in targets:
            results = [simulate_safe_mode(o, params, sport_key, tgt,
                                          team_defense, league_avg_def)
                       for o in rows]
            eligible = [r for r in results if r["eligible"]]
            n_elig = len(eligible)
            n_filt = len(results) - n_elig
            if n_elig == 0:
                print(f"{prop_key:<22} {tgt*100:>6.0f}% {n_elig:>8d} "
                      f"{'—':>5} {'—':>6} {n_filt:>9d} {'—':>8} {'—':>10}")
                continue
            hits = sum(1 for r in eligible if r["hit"])
            rate = hits / n_elig
            gaps = [r["line"] - r["safe_threshold"] for r in eligible]
            avg_gap = sum(gaps) / len(gaps)
            ths = sorted(r["safe_threshold"] for r in eligible)
            med_th = ths[len(ths) // 2]
            print(f"{prop_key:<22} {tgt*100:>6.0f}% {n_elig:>8d} "
                  f"{hits:>5d} {rate * 100:>5.1f}% {n_filt:>9d} "
                  f"{avg_gap:>+8.2f} {med_th:>10d}")
        print()

    print("Eligible  = passed both production guards (p_at_safe ≥ target-0.05 AND")
    print("            safe_threshold ≥ 50% of book line)")
    print("Avg gap   = book_line − safe_threshold (positive = suggestion is below the line)")
    print("Median th = median suggested alt-line across eligible obs")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="nba")
    p.add_argument("--variants", default="recency,all",
                   help="Comma-separated subset of " + ",".join(VARIANT_PRESETS.keys()))
    p.add_argument("--confidence-report", action="store_true",
                   help="Also print precision-vs-coverage table: hit-rate when "
                        "model confidence is >= threshold (0.55, 0.60, ... 0.90).")
    p.add_argument("--safe-mode-report", action="store_true",
                   help="Also simulate production safe-mode (alt-line) suggestions "
                        "and report actual hit rates per safe_target.")
    p.add_argument("--safe-targets", default="0.85,0.90,0.95",
                   help="Comma-separated safe_target values to evaluate.")
    args = p.parse_args()

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]
    target_props = PROPS_BY_SPORT.get(args.sport, [])
    if not target_props:
        print(f"No default prop list for {args.sport}; edit PROPS_BY_SPORT.")
        sys.exit(1)

    print(f"\n=== Harvesting cached book lines for {sport_key} "
          f"({', '.join(target_props)}) ===")
    book_lines = harvest_book_lines(sport_key, target_props)
    print(f"  found {len(book_lines)} unique (player, prop, game) book lines")
    if not book_lines:
        print("Nothing to evaluate. Run the live tool first to populate cache.")
        sys.exit(0)

    by_prop = defaultdict(int)
    for r in book_lines:
        by_prop[r["prop_key"]] += 1
    for k, v in sorted(by_prop.items()):
        print(f"    {k}: {v}")

    print(f"\n=== Joining to ESPN gamelogs (using {espn_sport}/{espn_league}) ===")
    enriched = join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        print("No observations could be joined to actuals.")
        sys.exit(0)

    # Optional team-defense lookup if any variant uses it
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = {n: _resolve_params(VARIANT_PRESETS[n], sport_key) for n in variant_names}
    team_defense, league_avg_def = {}, None
    if any(v.get("opp_defense_strength", 0.0) > 0 for v in variants.values()):
        print("\n=== Building team-defense lookup for one variant ===")
        team_defense, _, league_avg_def = _team_defense_lookup(espn_sport, espn_league)

    print("\n=== Evaluating forecasters at REAL book lines ===")
    print()
    header = (f"{'Variant':<10} {'Prop':<22} {'N_tr':>4} {'N_te':>4} {'#pl':>4}  "
              f"{'μ_r':>7} {'σ_r':>6}  "
              f"{'BrA':>6} {'BrB':>6} {'BrB*':>6} {'BrC':>6} {'BrC*':>6}  "
              f"{'HitA':>5} {'HitB':>5} {'HitB*':>5} {'HitC':>5} {'HitC*':>5}")
    print(header); print("-" * len(header))

    per_variant_prop_results = {}
    for vname, params in variants.items():
        per_prop_obs = defaultdict(list)
        for obs in enriched:
            projected, emp = project_and_empirical(obs, params, sport_key,
                                                   team_defense, league_avg_def)
            if projected is None:
                continue
            per_prop_obs[obs["prop_key"]].append({
                "player": obs["player"],
                "projected": projected,
                "line": obs["line"],
                "actual": obs["actual"],
                "empirical_over": emp,
                "game_date": obs["game_date"],
            })

        for prop_key in sorted(per_prop_obs.keys()):
            res = evaluate_calibration(per_prop_obs[prop_key], prop_key, prop_key)
            if not res:
                continue
            per_variant_prop_results[(vname, prop_key)] = res
            brA, brB, brBs, brC, brCs = res["brier"]
            hA, hB, hBs, hC, hCs = res["hit"]
            print(f"{vname:<10} {prop_key:<22} {res['n_train']:>4} {res['n_test']:>4} "
                  f"{res['n_players']:>4}  "
                  f"{res['mu_r']:>+7.3f} {res['sigma_r']:>6.3f}  "
                  f"{brA:>6.4f} {brB:>6.4f} {brBs:>6.4f} {brC:>6.4f} {brCs:>6.4f}  "
                  f"{hA:>4.1f}% {hB:>4.1f}% {hBs:>4.1f}% {hC:>4.1f}% {hCs:>4.1f}%")

    print()
    print("A  = empirical (weighted prior-game fraction > line) — current production method")
    print("B  = pooled Gaussian residual model")
    print("C  = pooled empirical residual CDF")
    print("B* = per-player Gaussian residual model, shrunk toward pool by λ=n/(n+15)")
    print("C* = per-player empirical residual CDF, mixture-blended with pool by same λ")
    print()
    print("Calibration fit on chronologically earliest half; scoring on later half (no leakage).")

    if args.confidence_report and per_variant_prop_results:
        print_confidence_report(per_variant_prop_results)

    if args.safe_mode_report:
        try:
            targets = tuple(float(t.strip()) for t in args.safe_targets.split(",") if t.strip())
        except ValueError:
            targets = (0.85, 0.90, 0.95)
        # Use the "all" variant params (the production default-ish settings)
        sm_params = _resolve_params(VARIANT_PRESETS["all"], sport_key)
        sm_team_def = team_defense if team_defense else {}
        sm_league_avg = league_avg_def
        if sm_params.get("output_def_strength", 0.0) > 0 and not sm_team_def:
            print("\n=== Building team-defense lookup for safe-mode sim ===")
            sm_team_def, _, sm_league_avg = _team_defense_lookup(espn_sport, espn_league)
        print_safe_mode_report(enriched, sm_params, sport_key,
                               sm_team_def, sm_league_avg, targets=targets)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
