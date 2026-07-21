"""Actual-bets ledger — the wagers the user really placed and their realized ROI.

The forward prediction log tracks how the *model* would have done at a flat unit;
this ledger tracks how the *user* actually did on the bets they placed money on.
The "Submit Picks" button turns the selected-bets checklist into ledger rows
(flat unit stake, executed at the model's best price at submit), which are then
auto-graded and rolled up into stake-weighted realized ROI. Closing lines from
the odds warehouse add CLV once they accumulate.

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


def build_wager_row(bet_type, side, candidate, meta):
    """Build one ledger row from an analysis candidate. Returns None on failure.

    Flat unit stake (meta['stake']); executed price = the model's best price at
    submit for the chosen side. Pure function — the app supplies event metadata
    (commence/game_date/home/away) so grading needs no live re-fetch."""
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
        return row
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Store I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_wagers():
    """All ledger rows (best-effort snapshot).

    Uses the short-TTL blob read cache: the ledger is read on every My Bets
    rerun, and every writer invalidates the cache, so this avoids a full-ledger
    GET+parse per rerun without risking staleness after an edit."""
    try:
        rows, _ = recalibration._read_ndjson_blob(WAGERS_FILE, use_cache=True)
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

    return recalibration.mutate_ndjson_log(WAGERS_FILE, prune)


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

    return recalibration.mutate_ndjson_log(WAGERS_FILE, apply)


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

    return recalibration.mutate_ndjson_log(WAGERS_FILE, reset)


# ──────────────────────────────────────────────────────────────────────────────
# Grading
# ──────────────────────────────────────────────────────────────────────────────

def _grade_wager(row):
    """(status, actual) for a settled wager, or None if not yet gradable."""
    bet_type = row.get("bet_type")
    commence = row.get("commence_time")
    game_date = (row.get("game_date") or "")[:10]

    if bet_type == "player_prop":
        actual = recalibration.resolve_one_prop(
            row.get("sport_key"), row.get("player"), row.get("prop_key"),
            row.get("line"), game_date, commence)
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
    score = game_results.final_score(
        row.get("sport_key"), game_date, row.get("home_team"),
        row.get("away_team"), commence)
    if score is None:
        return None
    home_score, away_score = score
    status = game_results.grade_team_bet(
        bet_type, row.get("side"), row.get("point"), home_score, away_score)
    if status is None:
        return None
    return status, f"{home_score:g}-{away_score:g}"


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
    rows = read_wagers()
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
        return recalibration.mutate_ndjson_log(WAGERS_FILE, apply)
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Closing-line value (from the warehouse)
# ──────────────────────────────────────────────────────────────────────────────

def _executed_implied(price):
    try:
        from odds_client import american_to_implied_prob
        return american_to_implied_prob(int(price))
    except (TypeError, ValueError, ImportError):
        return None


def attach_clv(rows):
    """Fill close_line/close_price/clv_pct from warehoused closing lines.

    Positive CLV = the executed price implied a lower probability (better odds)
    than the closing line. Mutates and returns ``rows``. Best-effort."""
    try:
        import warehouse
    except Exception:
        return rows
    for row in rows:
        if row.get("close_price") is not None:
            continue
        try:
            bet_type = row.get("bet_type")
            commence = row.get("commence_time")
            # The warehouse partitions snapshots by the UTC commence date
            # (commence[:10]); the row's game_date is now US-local, so a late
            # game would miss its snapshot folder. Query by the UTC date, only
            # falling back to game_date when commence is absent.
            wh_date = (commence or "")[:10] or (row.get("game_date") or "")[:10]
            common = dict(sport=row.get("sport_key"), game_date=wh_date,
                          event_id=row.get("event_id"),
                          commence_time=commence)
            if bet_type == "player_prop":
                close = warehouse.closing_line_for(
                    bet_type="player_prop", player=row.get("player"),
                    prop_key=row.get("prop_key"), direction=row.get("direction"),
                    point=row.get("line"), **common)
            elif bet_type == "total":
                close = warehouse.closing_line_for(
                    bet_type="total", selection=row.get("side"),
                    point=row.get("point"), **common)
            else:  # moneyline / spread — selection is the team
                close = warehouse.closing_line_for(
                    bet_type=bet_type, selection=row.get("team"),
                    point=row.get("point"), **common)
            if not close or close.get("implied_prob") is None:
                continue
            row["close_price"] = close.get("price")
            row["close_line"] = row.get("point")
            executed = _executed_implied(row.get("executed_price"))
            if executed is not None:
                row["clv_pct"] = round(
                    (close["implied_prob"] - executed) * 100.0, 2)
        except Exception:
            continue
    return rows


def _commence_passed(row, now):
    """True once first pitch/kickoff has passed (the closing line is now final)."""
    commence = game_results._parse_utc(row.get("commence_time"))
    return commence is not None and now >= commence


def persist_clv(rows=None, now=None):
    """Compute closing-line value for started games and write it durably.

    CLV is only meaningful once the game has started — before then the warehouse
    has no post-commence "closing" snapshot and the latest line is still moving —
    so this only fills rows whose commence_time has passed and that don't already
    have a close_price. The filled close_price/close_line/clv_pct are written
    back through the ETag-safe ledger store so later renders (and restarts) read
    them cheaply instead of re-hitting the warehouse every time. Returns the
    count newly persisted. Best-effort; never raises."""
    now = now or datetime.now(timezone.utc)
    rows = read_wagers() if rows is None else rows
    if not rows:
        return 0
    candidates = [r for r in rows
                  if r.get("close_price") is None and _commence_passed(r, now)]
    if not candidates:
        return 0
    attach_clv(candidates)  # fills close_price/close_line/clv_pct in memory
    filled = {
        r["wager_id"]: {k: r.get(k)
                        for k in ("close_price", "close_line", "clv_pct")}
        for r in candidates
        if r.get("wager_id") and r.get("close_price") is not None
    }
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
        return recalibration.mutate_ndjson_log(WAGERS_FILE, apply)
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Summary (stake-weighted realized ROI)
# ──────────────────────────────────────────────────────────────────────────────

def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metrics(group):
    settled = [r for r in group if r.get("status") in ("won", "lost", "push")]
    staked = sum(_num(r.get("stake")) for r in settled)
    realized = sum(_num(r.get("profit")) for r in settled)
    won = sum(1 for r in settled if r.get("status") == "won")
    lost = sum(1 for r in settled if r.get("status") == "lost")
    push = sum(1 for r in settled if r.get("status") == "push")
    clvs = [_num(r.get("clv_pct")) for r in group if r.get("clv_pct") is not None]
    decided = won + lost
    return {
        "total": len(group),
        "resolved": len(settled),
        "pending": len(group) - len(settled),
        "total_staked": staked,
        "realized_profit": realized,
        "roi": (realized / staked) if staked else None,
        "won": won,
        "lost": lost,
        "push": push,
        "hit_rate": (won / decided) if decided else None,
        "avg_clv_pct": (sum(clvs) / len(clvs)) if clvs else None,
    }


def summarize_wagers(rows):
    """Stake-weighted realized ROI summary with by-sport / by-bet-type splits."""
    summary = _metrics(rows)
    summary["pending_stake"] = sum(
        _num(r.get("stake")) for r in rows if r.get("status") == "pending")
    by_sport = defaultdict(list)
    by_type = defaultdict(list)
    for row in rows:
        by_sport[row.get("sport_key")].append(row)
        by_type[row.get("bet_type")].append(row)
    summary["by_sport"] = [
        {"sport_key": key, **_metrics(group)}
        for key, group in sorted(by_sport.items(), key=lambda kv: str(kv[0]))
    ]
    summary["by_bet_type"] = [
        {"bet_type": key, **_metrics(group)}
        for key, group in sorted(by_type.items(), key=lambda kv: str(kv[0]))
    ]
    return summary
