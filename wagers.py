"""Actual-bets ledger — the wagers the user really placed and their realized ROI.

The forward prediction log tracks how the *model* would have done at a flat unit;
this ledger tracks how the *user* actually did on the bets they placed money on.
The "Submit Picks" button turns the selected-bets checklist into ledger rows
(flat unit stake, executed at the model's best price at submit), which are then
auto-graded and rolled up into stake-weighted realized ROI. Closing-line value
(CLV) is filled offline by the DraftKings backfill CLI (backfill_dk_clv.py),
which reads DK's price at the exact line each bet was placed on.

Storage is a single NDJSON sibling blob (``wagers.jsonl``) written through the
generalized read-modify-write store in ``recalibration`` — no new secret. Rows
key on a unique ``wager_id`` (placed_at + sequence), NOT the model's forecast
identity, so multiple real bets on the same line and team-market bet types both
have a home the prediction log can't give them.

Every public entry point is best-effort and never raises into the app.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import game_results
import pricing_common
import recalibration

WAGERS_FILE = "wagers.jsonl"
_SETTLED = ("won", "lost", "push", "void")

# Minimum plausible game duration per sport. A wager is not even considered for
# grading until now >= commence_time + this buffer, so a game that is still in
# progress is never fetched and graded off a partial line. This is only a cheap
# pre-filter; the resolvers' own final-status gates are the real guarantee, so
# the buffer is deliberately short and a long/extra-innings game that runs past
# it simply stays pending (its resolver returns "not final" until it ends).
_MIN_GAME_DURATION = {
    "baseball_mlb": timedelta(hours=3),
    "basketball_nba": timedelta(hours=2, minutes=30),
    "americanfootball_nfl": timedelta(hours=3, minutes=30),
    "icehockey_nhl": timedelta(hours=3),
}
_DEFAULT_MIN_GAME_DURATION = timedelta(hours=3)


def storage_backend():
    """Human-readable active ledger backend (mirrors the prediction log)."""
    return recalibration.prediction_log_storage()


# ──────────────────────────────────────────────────────────────────────────────
# Building rows from analysis candidates
# ──────────────────────────────────────────────────────────────────────────────

def _pct(value):
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def _matchup(candidate):
    team = candidate.get("team")
    opponent = candidate.get("opponent")
    if (candidate.get("home_away") or "").upper() == "HOME":
        return f"{opponent} @ {team}"
    return f"{team} @ {opponent}"


def _blank_row(bet_type, meta):
    commence = meta.get("commence_time")
    return {
        "wager_id": f"{meta.get('placed_at')}#{meta.get('seq', 0)}",
        "placed_at": meta.get("placed_at"),
        "sport_key": meta.get("sport_key"),
        "bet_type": bet_type,
        "event_id": meta.get("event_id"),
        "commence_time": commence,
        # US-Eastern local date, NOT the raw UTC date: a late US game's UTC
        # first-pitch date is one day ahead, which mis-buckets grading + display.
        "game_date": meta.get("game_date") or pricing_common.et_local_date(commence),
        "home_team": meta.get("home_team"),
        "away_team": meta.get("away_team"),
        "matchup": None,
        "team": None,
        "opponent": None,
        "home_away": None,
        "point": None,
        "player": None,
        "prop_key": None,
        "prop_label": None,
        "direction": None,
        "line": None,
        "side": None,
        "executed_price": None,
        "stake": float(meta.get("stake") or 0.0),
        "book": None,
        "model_prob": None,
        "model_edge": None,
        "model_price": None,
        "status": "pending",
        "actual": None,
        "profit": None,
        "resolved_at": None,
        "close_line": None,
        "close_price": None,
        "clv_pct": None,
    }


def _enrich_ids(row):
    """Best-effort: stamp SFBB canonical team codes (+ the player's MLBAM id for
    props) onto a wager row for id-based joins. MLB-only — team_code_for_name
    returns None for other sports, and player_mlb_id is resolved only for baseball
    (a non-MLB name could otherwise collide with an MLB player). Fail-open, never
    raises; mutates ``row`` in place."""
    if not (row.get("sport_key") or "").startswith("baseball"):
        return row
    try:
        import player_id_map
    except Exception:                       # pragma: no cover - import guard
        return row
    try:
        tc = player_id_map.team_code_for_name
        row["home_code"] = tc(row.get("home_team")) if row.get("home_team") else None
        row["away_code"] = tc(row.get("away_team")) if row.get("away_team") else None
        row["team_code"] = tc(row.get("team")) if row.get("team") else None
        row["opponent_code"] = tc(row.get("opponent")) if row.get("opponent") else None
        if row.get("player"):
            # P3: resolve player → (MLBAM id, game_pk) fail-closed with the game's
            # BOTH teams as the namesake-tie hint (stronger than the old single-team
            # SFBB call). An unresolved player keeps NULL ids (shadow signal).
            # Commit C: pass prop_key so the game-context resolver role-partitions
            # this stamp. This is a single-row submit (no batch), so the resolver
            # lazily loads the season roster index on first use — no warm needed here.
            import entity_resolver
            ident = entity_resolver.resolve(
                row.get("player"), row.get("sport_key"),
                row.get("home_team"), row.get("away_team"),
                game_date=row.get("game_date"), commence=row.get("commence_time"),
                prop_key=row.get("prop_key"))
            row["player_mlb_id"] = ident.get("mlb_player_id")
            row["game_pk"] = ident.get("game_pk")
    except Exception:                       # pragma: no cover - never break submit
        pass
    return row


def build_wager_row(bet_type, side, candidate, meta):
    """Build one ledger row from an analysis candidate. Returns None on failure.

    Flat unit stake (meta['stake']); executed price = the model's best price at
    submit for the chosen side. The event metadata the app supplies
    (commence/game_date/home/away) lets grading run with no live re-fetch, and is
    enriched best-effort with SFBB canonical team codes / the player's MLBAM id."""
    try:
        row = _blank_row(bet_type, meta)
        if not row["event_id"]:
            row["event_id"] = candidate.get("event_id")

        if bet_type == "moneyline":
            home_away = candidate.get("home_away")
            price = candidate.get("best_price")
            row.update({
                "team": candidate.get("team"),
                "opponent": candidate.get("opponent"),
                "home_away": home_away,
                "matchup": _matchup(candidate),
                "side": "home" if (home_away or "").upper() == "HOME" else "away",
                "executed_price": price, "model_price": price,
                "book": candidate.get("best_book"),
                "model_prob": _pct(candidate.get("blended_prob")),
                "model_edge": candidate.get("best_edge_pct"),
            })
        elif bet_type == "spread":
            home_away = candidate.get("home_away")
            price = candidate.get("price")
            spread = candidate.get("spread")
            row.update({
                "team": candidate.get("team"),
                "opponent": candidate.get("opponent"),
                "home_away": home_away,
                "matchup": _matchup(candidate),
                "side": "home" if (home_away or "").upper() == "HOME" else "away",
                "point": spread, "line": spread,
                "executed_price": price, "model_price": price,
                "model_prob": _pct(candidate.get("cover_rate")),
                "model_edge": candidate.get("edge_pct"),
            })
        elif bet_type == "total":
            sd = (side or "OVER").upper()
            line = candidate.get("line")
            price = (candidate.get("over_price") if sd == "OVER"
                     else candidate.get("under_price"))
            over_hit = candidate.get("over_hit_rate")
            model_prob = (_pct(over_hit) if sd == "OVER"
                          else (None if over_hit is None else _pct(100.0 - over_hit)))
            edge = (candidate.get("over_edge_pct") if sd == "OVER"
                    else candidate.get("under_edge_pct"))
            row.update({
                "matchup": candidate.get("matchup"),
                "team": "Both teams",
                "side": sd.lower(), "direction": sd,
                "point": line, "line": line,
                "executed_price": price, "model_price": price,
                "model_prob": model_prob, "model_edge": edge,
            })
        elif bet_type == "player_prop":
            direction = (candidate.get("direction") or "OVER").upper()
            if candidate.get("safe_mode"):
                line = candidate.get("safe_alt_line")
                price = candidate.get("safe_alt_price")
                model_prob = _pct(candidate.get("model_hit_at_safe"))
                direction = "OVER"
            else:
                line = candidate.get("line")
                # Stake at the DraftKings price the user actually bets (P1.1b).
                # over/under_price is the best-across-books price used only for
                # the value/EV decision; fall back to it when DK is absent.
                best_side = (candidate.get("over_price") if direction == "OVER"
                             else candidate.get("under_price"))
                dk_side = (candidate.get("dk_over_price") if direction == "OVER"
                           else candidate.get("dk_under_price"))
                price = dk_side if dk_side is not None else best_side
                over_rate = candidate.get("over_rate")
                model_prob = (_pct(over_rate) if direction == "OVER"
                              else (None if over_rate is None
                                    else _pct(100.0 - over_rate)))
            row.update({
                "matchup": candidate.get("matchup"),
                "team": candidate.get("team"),
                "player": candidate.get("player"),
                "prop_key": candidate.get("prop"),
                "prop_label": candidate.get("prop_label"),
                "direction": direction,
                "line": line, "point": line,
                "executed_price": price, "model_price": price,
                "model_prob": model_prob,
                "model_edge": candidate.get("edge_pct"),
            })
        else:
            return None
        # Fractional-Kelly stake (P-Kelly): size from the already-shrunk model
        # probability (row['model_prob'] is a 0-1 fraction via _pct) at the DK
        # executed price (row['executed_price']). Sizing here keeps the stake
        # perfectly consistent with what grading/ROI read off the same row.
        # Fail-open: leave the flat _blank_row stake (meta['stake']) when Kelly
        # is off or the leg is not sizable (no DK price / non-positive EV).
        if (meta.get("kelly")
                and row.get("model_prob") is not None
                and row.get("executed_price") is not None):
            row["stake"] = pricing_common.kelly_stake(
                row["model_prob"], row["executed_price"],
                meta.get("bankroll"), meta.get("kelly_fraction", 0.5),
                meta.get("kelly_cap", 0.05))
        return _enrich_ids(row)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Store I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_wagers_with_status():
    """(rows, error): like read_wagers but surfaces a backend failure.

    Lets the UI tell an UNREACHABLE durable store (Azure SQL serverless resuming
    from auto-pause, or its monthly free-tier compute exhausted → a connection
    timeout) apart from a genuinely EMPTY ledger — both otherwise look like [].
    ``error`` is None on success (``rows`` may still legitimately be [])."""
    try:
        rows, _ = recalibration._read_ndjson_blob(WAGERS_FILE, use_cache=True)
        return rows, None
    except Exception as exc:
        return [], exc


def read_wagers(where=None):
    """All ledger rows (best-effort snapshot), or only those matching ``where``
    (an equality/IN {col: value} map, SQL path only) so a reconciliation caller
    pulls just the subset it needs instead of the whole ledger.

    The unfiltered read uses the short-TTL cache (the ledger is read on every My
    Bets rerun, and every writer invalidates the cache). A filtered read is
    uncached — its subset is not the cached full snapshot."""
    if where is None:
        return read_wagers_with_status()[0]
    try:
        rows, _ = recalibration._read_ndjson_blob(WAGERS_FILE, where=where)
        return rows
    except Exception:
        return []


def submit_wagers(rows):
    """Append new wager rows, de-duplicating by wager_id. Returns count added.

    Raises on a storage failure rather than swallowing it — a Submit is a
    user-initiated action, so the caller must be able to tell the user it failed
    (and keep their selections) instead of silently reporting zero."""
    if not rows:
        return 0

    def upsert(existing):
        have = {r.get("wager_id") for r in existing}
        added = 0
        for row in rows:
            if row.get("wager_id") in have:
                continue
            existing.append(row)
            have.add(row.get("wager_id"))
            added += 1
        return added

    return recalibration.mutate_ndjson_log(WAGERS_FILE, upsert)


# Fields the user may correct on a PENDING bet when the number changed between
# running the analysis and actually placing the bet. Editing the line also syncs
# ``point`` (see update_wagers) so grading stays consistent across bet types.
_EDITABLE_FIELDS = ("executed_price", "line", "stake")


def delete_wagers(wager_ids):
    """Remove ledger rows by wager_id. Returns the count removed.

    Raises on a storage failure (like submit_wagers, this is user-initiated) so
    the caller can report it instead of silently reporting zero."""
    ids = {wid for wid in (wager_ids or []) if wid}
    if not ids:
        return 0

    def prune(existing):
        before = len(existing)
        existing[:] = [r for r in existing if r.get("wager_id") not in ids]
        return before - len(existing)

    # SQL path: pull only the targeted rows so the surgical diff emits DELETEs for
    # exactly them (the whole-ledger read is wasted egress here). The Blob path
    # ignores ``where`` and prunes the full list — same result.
    return recalibration.mutate_ndjson_log(
        WAGERS_FILE, prune, where={"wager_id": list(ids)})


def update_wagers(edits):
    """Apply field corrections to PENDING rows. Returns the count changed.

    ``edits`` maps wager_id -> {field: value} over ``_EDITABLE_FIELDS``. Settled
    rows are never touched — their grade and profit are already realized. Editing
    the line syncs both ``line`` (prop grading) and ``point`` (team-market
    grading) so a corrected line grades every bet type consistently. Raises on a
    storage failure so the caller can surface it."""
    clean = {wid: patch for wid, patch in (edits or {}).items() if wid and patch}
    if not clean:
        return 0

    def apply(existing):
        changed = 0
        for row in existing:
            wid = row.get("wager_id")
            if wid not in clean or row.get("status") != "pending":
                continue
            touched = False
            for field in _EDITABLE_FIELDS:
                patch = clean[wid]
                if field not in patch:
                    continue
                value = patch[field]
                if row.get(field) == value:
                    continue
                row[field] = value
                if field == "line":
                    row["point"] = value  # keep the graded point in sync
                touched = True
            if touched:
                changed += 1
        return changed

    # SQL path: read only the edited rows (the mutator still self-filters to
    # pending). Blob path ignores ``where``.
    return recalibration.mutate_ndjson_log(
        WAGERS_FILE, apply, where={"wager_id": list(clean)})


def regrade_wagers(wager_ids):
    """Reset settled rows to pending so the next resolve pass re-grades them.

    Clears the realized fields (status/actual/profit/resolved_at) on the given
    SETTLED rows while leaving stake/price/line intact. Use to correct bets that
    were graded under the old buggy live-game logic (a still-live game marked as
    a loss). Already-pending rows are ignored. Returns the count reset. Raises on
    a storage failure so the caller can surface it."""
    ids = {wid for wid in (wager_ids or []) if wid}
    if not ids:
        return 0

    def reset(existing):
        changed = 0
        for row in existing:
            if row.get("wager_id") in ids and row.get("status") in _SETTLED:
                row.update({"status": "pending", "actual": None,
                            "profit": None, "resolved_at": None})
                changed += 1
        return changed

    # SQL path: read only the targeted rows (mutator still self-filters to
    # settled). Blob path ignores ``where``.
    return recalibration.mutate_ndjson_log(
        WAGERS_FILE, reset, where={"wager_id": list(ids)})


# ──────────────────────────────────────────────────────────────────────────────
# Grading
# ──────────────────────────────────────────────────────────────────────────────

def _grade_wager(row):
    """(status, actual) for a settled wager, or None if not yet gradable."""
    bet_type = row.get("bet_type")
    commence = row.get("commence_time")
    game_date = (row.get("game_date") or "")[:10]

    if bet_type == "player_prop":
        # P4: pass the P3-stamped (game_pk, mlb_player_id) so grading can read the
        # actual straight from the warehouse facts (0 network) when available.
        actual = recalibration.resolve_one_prop(
            row.get("sport_key"), row.get("player"), row.get("prop_key"),
            row.get("line"), game_date, commence,
            game_pk=row.get("game_pk"), mlb_player_id=row.get("player_mlb_id"))
        if actual is None:
            return None
        try:
            line = float(row.get("line"))
        except (TypeError, ValueError):
            return None
        direction = (row.get("direction") or "OVER").upper()
        if actual == line:
            status = "push"
        elif direction == "UNDER":
            status = "won" if actual < line else "lost"
        else:
            status = "won" if actual > line else "lost"
        return status, actual

    # Team markets (moneyline / spread / total).
    # Fast path (Tier A #2, MLB only): when the wager carries a game_pk, grade off
    # the EXACT warehouse game (doubleheader-safe) instead of name+date. A positive
    # 'still live' verdict returns None-from-GRADE_PENDING and stays pending — we do
    # NOT fall back, because name+date could grade off a near-same-commence DH
    # sibling. Anything uncertain returns None and falls through to the unchanged
    # name+date path below (byte-identical to pre-#2 when game_pk is absent / non-MLB).
    game_pk = row.get("game_pk")
    if row.get("sport_key") == "baseball_mlb" and game_pk:
        fast = game_results.grade_team_bet_by_game_pk(
            row.get("sport_key"), game_pk, bet_type, row.get("side"),
            row.get("team"), row.get("point"))
        if fast is game_results.GRADE_PENDING:
            return None
        if fast is not None:
            return fast

    score = game_results.final_score(
        row.get("sport_key"), game_date, row.get("home_team"),
        row.get("away_team"), commence)
    if score is None:
        return None
    home_score, away_score = score
    side = row.get("side")
    if bet_type in ("moneyline", "spread"):
        # Grade by the bet's TEAM identity, not the stored side. `side` is set at
        # submit from home_away; if that were ever stale/flipped it would settle
        # the bet on the wrong team (a Yankees ML grading off the Phillies' win).
        # final_score already matched the game on these exact home/away names, so
        # the bet's team maps unambiguously to a side here; fall back to the stored
        # side only if the team can't be matched (name drift → don't guess wrong).
        resolved = game_results.side_for_team(
            row.get("team"), row.get("home_team"), row.get("away_team"))
        if resolved is not None:
            side = resolved
    status = game_results.grade_team_bet(
        bet_type, side, row.get("point"), home_score, away_score)
    if status is None:
        return None
    return status, f"{home_score:g}-{away_score:g}"


def _dnp_void_update(row, now):
    """A void update for a player prop whose player is a confirmed, stale DNP, or
    None if the row isn't one.

    Mirrors the prediction resolver's stale-DNP sweep: a player who was listed but
    never took the field leaves no game log to intersect, so ``resolve_one_prop``
    returns None forever. Once the game is at least ``STALE_DNP_HOURS`` old the
    scratch is permanent, so we void the bet (stake refunded, ROI-neutral) instead
    of leaving it pending every tick. The age gate lives in ``_is_stale_dnp``, so a
    same-day data lag is never voided. Never raises."""
    if row.get("bet_type") != "player_prop":
        return None
    game_date = (row.get("game_date") or "")[:10]
    try:
        stale = recalibration._is_stale_dnp(
            row.get("sport_key"), row.get("prop_key"), row.get("player"),
            game_date, row.get("commence_time"))
    except Exception:
        return None
    if not stale:
        return None
    update = {
        "status": "void",
        "actual": None,
        "profit": 0.0,  # stake refunded — a void is ROI-neutral
        "resolved_at": now.isoformat(),
    }
    healed = pricing_common.et_local_date(row.get("commence_time"))
    if healed:
        update["game_date"] = healed
    return update


def _maybe_finished(row, now):
    """True when the wager's game has plausibly ended (commence + sport buffer).

    Prefers ``commence_time`` (timezone-correct — sidesteps the UTC-vs-local
    date trap that let a live evening game look 'past' and stranded a finished
    late game as 'pending'). Falls back to the legacy UTC game_date < today check
    only when commence is missing/unparseable. The resolver's own final-status
    gate is what actually prevents grading a still-live game; this is a cheap
    pre-filter that also avoids fetching clearly-unfinished games."""
    commence = game_results._parse_utc(row.get("commence_time"))
    if commence is not None:
        buffer = _MIN_GAME_DURATION.get(row.get("sport_key"),
                                        _DEFAULT_MIN_GAME_DURATION)
        return now >= commence + buffer
    game_date = (row.get("game_date") or "")[:10]
    return bool(game_date) and game_date < now.date().isoformat()


def resolve_pending_wagers(max_to_resolve=200, now=None):
    """Grade finished pending wagers and record status/actual/profit.

    Returns the number newly graded. Best-effort; never raises."""
    now = now or datetime.now(timezone.utc)
    # Pull only PENDING rows out of the DB — settled bets are the bulk and are
    # skipped anyway. (Blob/local path ignores the filter and self-filters below.)
    pending_only = {"status": "pending"}
    rows = read_wagers(where=pending_only)
    if not rows:
        return 0
    updates = {}
    count = 0
    for row in rows:
        if count >= max_to_resolve:
            break
        if row.get("status") != "pending":
            continue
        if not _maybe_finished(row, now):
            continue  # game hasn't plausibly finished yet
        graded = _grade_wager(row)
        if graded is None:
            # Not gradable. If it's a confirmed, stale DNP (player never played),
            # it will never become gradable — void it so it stops stranding as
            # pending. Otherwise leave it pending to retry next tick.
            void = _dnp_void_update(row, now)
            if void is not None:
                updates[row.get("wager_id")] = void
                count += 1
            continue
        status, actual = graded
        won = None if status == "push" else (status == "won")
        realized = pricing_common.profit(
            row.get("executed_price"), row.get("stake"), won)
        update = {
            "status": status,
            "actual": actual,
            "profit": realized,
            "resolved_at": now.isoformat(),
        }
        # Heal a legacy row stored with the raw UTC date so it displays the
        # correct US-local game date now that it has settled.
        healed = pricing_common.et_local_date(row.get("commence_time"))
        if healed:
            update["game_date"] = healed
        updates[row.get("wager_id")] = update
        count += 1

    if not updates:
        return 0

    def apply(current):
        changed = 0
        for row in current:
            update = updates.get(row.get("wager_id"))
            if update and row.get("status") == "pending":
                row.update(update)
                changed += 1
        return changed

    try:
        return recalibration.mutate_ndjson_log(WAGERS_FILE, apply,
                                                where=pending_only)
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Closing-line value (DraftKings, via backfill_dk_clv.py)
#
# CLV is filled EXCLUSIVELY by the on-demand DraftKings closing-line backfill
# (backfill_dk_clv.py) — DK-vs-DK at the exact bet line, for both player props
# AND team markets (moneyline/spread/total). The old render-time warehouse fill
# (attach_clv/persist_clv) was retired: the warehouse stores best-of-book /
# de-vigged CONSENSUS (never DraftKings) and matched lines fuzzily, so a
# DK-only bettor's CLV was optimistically biased and could compare mismatched
# lines. This module now only holds the shared writer/reset helpers; the render
# path just READS whatever CLV the backfill has written.
# ──────────────────────────────────────────────────────────────────────────────

def _executed_implied(price):
    try:
        from odds_client import american_to_implied_prob
        return american_to_implied_prob(int(price))
    except (TypeError, ValueError, ImportError):
        return None


def _commence_passed(row, now):
    """True once first pitch/kickoff has passed (the closing line is now final)."""
    commence = game_results._parse_utc(row.get("commence_time"))
    return commence is not None and now >= commence


def apply_clv_updates(filled):
    """Write close_price/close_line/clv_pct for the given wagers durably.

    ``filled`` maps wager_id -> {close_price, close_line, clv_pct}. Only rows
    whose close_price IS NULL are updated, so re-runs are idempotent (the SQL
    read is restricted to them; the Blob path ignores ``where`` and the mutator
    self-filters). This is the sole CLV writer — the DK closing-line backfill
    (backfill_dk_clv.py, props + team markets) calls it. Returns the count
    persisted. Best-effort; never raises."""
    if not filled:
        return 0

    def apply(current):
        changed = 0
        for row in current:
            update = filled.get(row.get("wager_id"))
            if update and row.get("close_price") is None:
                row.update(update)
                changed += 1
        return changed

    try:
        return recalibration.mutate_ndjson_log(
            WAGERS_FILE, apply, where={"close_price": None})
    except Exception:
        return 0


def reset_clv(wager_ids=None):
    """Clear close_price/close_line/clv_pct so the DK backfill recomputes them.

    Use after a closing-line correction, or to drop stale warehouse-era CLV so
    backfill_dk_clv.py can refill it DK-vs-DK: apply_clv_updates only fills rows
    whose close_price IS NULL, so already-filled (stale) values must be cleared
    first. Limited to ``wager_ids`` when given, else (``wager_ids is None``) every
    row that has a CLV value. An explicitly EMPTY selection clears nothing — never
    everything — so a caller that computed "the rows to reset" and got none can't
    accidentally wipe the ledger. Returns the count cleared. Raises on a storage
    failure so callers can surface it."""
    ids = None
    if wager_ids is not None:
        ids = {wid for wid in wager_ids if wid}
        if not ids:
            return 0

    def clear(rows):
        changed = 0
        for row in rows:
            if ids is not None and row.get("wager_id") not in ids:
                continue
            if row.get("close_price") is None and row.get("clv_pct") is None:
                continue
            row["close_price"] = None
            row["close_line"] = None
            row["clv_pct"] = None
            changed += 1
        return changed

    return recalibration.mutate_ndjson_log(WAGERS_FILE, clear)


# ──────────────────────────────────────────────────────────────────────────────
# Summary (stake-weighted realized ROI)
# ──────────────────────────────────────────────────────────────────────────────

def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _implied(price):
    """American odds -> vigged implied probability (our side). None if unusable."""
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price == 0:
        return None
    return (-price) / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def _metrics(group):
    # A VOID (scratch/DNP refund) is RESOLVED but ROI-neutral: the stake is
    # returned, so it carries no won/lost/push, no staked amount, and no realized
    # P/L — but it must NOT sit in the pending bucket (a voided bet is settled, and
    # leaving it as pending would misreport it as still-open forever).
    graded = [r for r in group if r.get("status") in ("won", "lost", "push")]
    voided = sum(1 for r in group if r.get("status") == "void")
    staked = sum(_num(r.get("stake")) for r in graded)
    realized = sum(_num(r.get("profit")) for r in graded)
    won = sum(1 for r in graded if r.get("status") == "won")
    lost = sum(1 for r in graded if r.get("status") == "lost")
    push = sum(1 for r in graded if r.get("status") == "push")
    clvs = [_num(r.get("clv_pct")) for r in group if r.get("clv_pct") is not None]
    decided = won + lost
    resolved = len(graded) + voided
    hit_rate = (won / decided) if decided else None
    # P&L ATTRIBUTION (mining idea #6): decompose realized edge into MODEL SKILL vs
    # LINE/TIMING value (CLV). We bet DK-only, so there is no cross-book line-
    # shopping and no decision-vs-fill slippage (executed price IS the analyzed
    # price) — the two live levers are (a) did our SIDE win more than the close
    # priced it (skill vs the efficient close), and (b) did we bet a BETTER NUMBER
    # than the close (CLV). Over won/lost (push excluded):
    decided_rows = [r for r in graded if r.get("status") in ("won", "lost")]
    mps = [_num(r.get("model_prob")) for r in decided_rows
           if r.get("model_prob") is not None]
    avg_model_prob = (sum(mps) / len(mps)) if mps else None
    closes = [_implied(r.get("close_price")) for r in decided_rows]
    closes = [c for c in closes if c is not None]
    avg_close_implied = (sum(closes) / len(closes)) if closes else None
    return {
        "total": len(group),
        "resolved": resolved,
        "pending": len(group) - resolved,
        "total_staked": staked,
        "realized_profit": realized,
        "roi": (realized / staked) if staked else None,
        "won": won,
        "lost": lost,
        "push": push,
        "void": voided,
        "hit_rate": hit_rate,
        "avg_clv_pct": (sum(clvs) / len(clvs)) if clvs else None,
        # attribution:
        "avg_model_prob": avg_model_prob,
        "avg_close_implied": avg_close_implied,
        # model overconfidence on this bucket (model said this %, we hit this %):
        "model_calib_gap": ((avg_model_prob - hit_rate)
                            if (avg_model_prob is not None and hit_rate is not None)
                            else None),
        # did our side beat the CLOSE (skill/luck vs the efficient line):
        "skill_vs_close": ((hit_rate - avg_close_implied)
                           if (avg_close_implied is not None and hit_rate is not None)
                           else None),
        "n_decided": decided,
    }


def _bet_type_group_key(row):
    """Group key for the by-bet-type split. Player props split by market
    (prop_key) so ROI is visible per prop type; other markets group by bet_type.
    Returns (bet_type, sub) where sub is the prop_key for props else ''."""
    bt = row.get("bet_type")
    if bt == "player_prop":
        return ("player_prop", row.get("prop_key") or "")
    return (bt, "")


def _bet_type_group_label(key, group):
    """Human label for a by-bet-type group. Player props read
    'Player Prop — <Prop Label>' (prefers the stored prop_label, falls back to a
    prettified prop_key); other markets use the title-cased bet type."""
    bt, sub = key
    if bt == "player_prop":
        for r in group:
            if r.get("prop_label"):
                return f"Player Prop — {r['prop_label']}"
        pretty = (sub or "prop").replace("_", " ").title()
        return f"Player Prop — {pretty}"
    return (bt or "—").title()


def summarize_wagers(rows):
    """Stake-weighted realized ROI summary with by-sport / by-bet-type splits.

    The by-bet-type split reports each player-prop MARKET separately (batter
    hits, pitcher strikeouts, …) rather than one pooled 'Player Prop' row, so the
    user sees ROI per market; team markets stay pooled by bet type."""
    summary = _metrics(rows)
    summary["pending_stake"] = sum(
        _num(r.get("stake")) for r in rows if r.get("status") == "pending")
    by_sport = defaultdict(list)
    by_type = defaultdict(list)
    for row in rows:
        by_sport[row.get("sport_key")].append(row)
        by_type[_bet_type_group_key(row)].append(row)
    summary["by_sport"] = [
        {"sport_key": key, **_metrics(group)}
        for key, group in sorted(by_sport.items(), key=lambda kv: str(kv[0]))
    ]
    summary["by_bet_type"] = [
        {"bet_type": key[0], "prop_key": key[1] or None,
         "label": _bet_type_group_label(key, group), **_metrics(group)}
        for key, group in sorted(
            by_type.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))
    ]
    return summary


def _attribution_verdict(m):
    """One-line diagnosis of WHERE a bucket's realized ROI comes from / leaks to
    (mining idea #6): model overconfidence, bad numbers (CLV), or variance."""
    n = m.get("n_decided") or 0
    if n < 20:
        return "thin (n<20) — not diagnosable"
    roi, clv = m.get("roi"), m.get("avg_clv_pct")
    gap, skill = m.get("model_calib_gap"), m.get("skill_vs_close")
    tags = []
    if clv is not None:
        tags.append("CLV+" if clv > 0.5 else ("CLV-" if clv < -0.5 else "CLV~0"))
    if gap is not None and gap > 0.05:
        tags.append("model OVERCONFIDENT")
    if skill is not None and skill > 0.01:
        tags.append("beats close")
    if roi is not None and roi < -0.005:
        if clv is not None and clv < -0.5:
            tags.append("=> bad NUMBERS (timing/price leak, not just model)")
        elif gap is not None and gap > 0.05:
            tags.append("=> model over-recommends here; shrink/tighten this bucket")
        else:
            tags.append("=> prices ok — likely variance/small-sample")
    elif roi is not None and roi > 0.005:
        tags.append("=> healthy")
    return " ".join(tags) or "flat"


def print_attribution(rows=None):
    """Standalone P&L ATTRIBUTION report over SETTLED wagers, per market: realized
    ROI split into model-skill (hit vs close-implied), model calibration gap
    (model_prob vs hit), and CLV (executed vs close). Answers 'is a losing bucket a
    MODEL problem or a PRICING/timing problem?' — the actionable read behind our
    open neg-ROI findings. Read-only."""
    if rows is None:
        rows = read_wagers()
    summary = summarize_wagers(rows)
    print("\n=== P&L ATTRIBUTION (settled wagers, per market) ===")
    print("  ROI = realized flat/stake-wtd; CLV% = executed-vs-close (>0 = beat the "
          "close);")
    print("  calib_gap = model_prob − hit (>0 = overconfident); skill = hit − "
          "close_implied.")
    print("  {:<26}{:>6}{:>8}{:>8}{:>9}{:>9}{:>8}".format(
        "market", "n", "ROI%", "hit%", "calibGap", "CLV%", "skill%"))
    groups = sorted(summary.get("by_bet_type", []),
                    key=lambda g: -(g.get("n_decided") or 0))
    for g in groups:
        n = g.get("n_decided") or 0
        if n == 0:
            continue

        def _p(v, scale=100.0, dec=1):
            return f"{v*scale:+.{dec}f}" if v is not None else "-"
        print("  {:<26}{:>6}{:>8}{:>8}{:>9}{:>9}{:>8}".format(
            (g.get("label") or g.get("bet_type") or "?")[:26], n,
            _p(g.get("roi")), _p(g.get("hit_rate")),
            _p(g.get("model_calib_gap")),
            (f"{g.get('avg_clv_pct'):+.1f}" if g.get("avg_clv_pct") is not None
             else "-"),
            _p(g.get("skill_vs_close"))))
        print(f"      -> {_attribution_verdict(g)}")
    print("\n  (Read-only. DK-only, so no cross-book line-shopping / decision-vs-fill")
    print("   slippage component; the two live levers are model-skill and CLV.)")


if __name__ == "__main__":
    # Offline read of the prod ledger needs SQL secrets promoted first.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass
    print_attribution()
