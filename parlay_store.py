"""parlay_store.py — persistence + grading for the dedicated parlay tracker.

Parlays are irreplaceable wagers → durable in AZURE SQL (the system of record) via the same store
path the wagers ledger uses (recalibration.mutate_ndjson_log → db_store → Azure SQL; a local
NDJSON file is used ONLY in no-SQL dev/test, never in prod). Two tables:
`parlays` (ticket) + `parlay_legs` (legs). Captures our joint P, boosted EV, and correlation at
placement so the LIVE record validates the parlay+SGP thesis (which can't be backtested — no
historical SGP prices). Per-leg grading reuses the wager grader's prop resolver
(recalibration.resolve_one_prop) → "which leg broke it" analytics.
"""
import uuid
from datetime import datetime, timezone

import recalibration
import bonus as bonuslib

PARLAYS_FILE = "parlays.jsonl"
LEGS_FILE = "parlay_legs.jsonl"


def _now():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return "pl_" + uuid.uuid4().hex[:12]


def save_parlay(ticket, legs):
    """Upsert one ticket + its legs. `ticket` = parlays-column dict (parlay_id optional →
    generated); `legs` = list of parlay_legs-column dicts (leg_index/parlay_id filled here).
    Returns the parlay_id."""
    pid = ticket.get("parlay_id") or new_id()
    trow = {**ticket, "parlay_id": pid}
    trow.setdefault("placed_at", _now())
    trow.setdefault("status", "pending")
    trow.setdefault("sport_key", "americanfootball_nfl")
    trow["n_legs"] = len(legs)

    norm = [{**lg, "parlay_id": pid, "leg_index": i,
             "sport_key": lg.get("sport_key") or trow["sport_key"]}
            for i, lg in enumerate(legs)]

    def up_ticket(rows):
        for r in rows:
            if r.get("parlay_id") == pid:
                r.update(trow)
                return 1
        rows.append(trow)
        return 1

    def up_legs(rows):
        # Replace this ticket's COMPLETE leg set: drop all of pid's existing legs
        # (so shrinking 3->2 legs leaves no ghost index-2 leg) then insert the new
        # ones. Self-filters, so it is correct on the local path (mutator sees all
        # rows) and the SQL path (rows scoped to pid). [F04]
        kept = [r for r in rows if r.get("parlay_id") != pid]
        kept.extend(norm)
        rows[:] = kept
        return len(norm)

    # Legs FIRST, ticket LAST. The ticket is what load_parlays surfaces, so writing
    # it last makes it the commit point: a failed leg write raises before any ticket
    # exists (no orphan ticket-with-no-legs), and a surfaced ticket always carries
    # its legs. True cross-table atomicity (one transaction spanning both tables +
    # promo consumption) is the deferred T04 unit-of-work — the local NDJSON fallback
    # cannot do multi-file atomicity. [F04]
    recalibration.mutate_ndjson_log(LEGS_FILE, up_legs, where={"parlay_id": pid})
    recalibration.mutate_ndjson_log(PARLAYS_FILE, up_ticket, where={"parlay_id": pid})
    return pid


def _cols(spec_name):
    try:
        import db_store
        return [n for n, _ in getattr(db_store, spec_name)]
    except Exception:
        return []


def load_parlays():
    """All tickets (most recent first), each with a sorted ``legs`` list attached. Missing
    columns are filled with None so the local-blob and SQL backends return identical shapes."""
    try:
        tickets, _ = recalibration._read_ndjson_blob(PARLAYS_FILE, use_cache=False)
        legs, _ = recalibration._read_ndjson_blob(LEGS_FILE, use_cache=False)
    except Exception:
        return []
    tcols, lcols = _cols("_PARLAY_SPEC"), _cols("_PARLAY_LEG_SPEC")

    def _fill(row, cols):
        return {**{c: None for c in cols}, **row}

    by_pid = {}
    for lg in (legs or []):
        by_pid.setdefault(lg.get("parlay_id"), []).append(_fill(lg, lcols))
    out = []
    for t in (tickets or []):
        t = _fill(t, tcols)
        t["legs"] = sorted(by_pid.get(t.get("parlay_id"), []),
                           key=lambda x: x.get("leg_index") or 0)
        out.append(t)
    out.sort(key=lambda t: t.get("placed_at") or "", reverse=True)
    return out


def _grade_leg(lg):
    """(actual, result) for one leg, or (None, None) if not yet resolvable.
    result: 1 win, 0 loss, None push."""
    actual = recalibration.resolve_one_prop(
        lg.get("sport_key"), lg.get("player"), lg.get("prop_key"), lg.get("line"),
        (lg.get("game_date") or "")[:10], lg.get("commence_time"))
    if actual is None:
        return None, None
    try:
        line = float(lg.get("line"))
    except (TypeError, ValueError):
        return None, None
    side = (lg.get("side") or "OVER").upper()
    if actual == line:
        return actual, None                       # push
    if side == "UNDER":
        return actual, (1 if actual < line else 0)
    return actual, (1 if actual > line else 0)


def _settle_fields(ticket):
    """Compute (status, payout, profit) for a ticket whose legs carry results.
    won = all legs win; lost = any leg lost; a push leg → 'void' (book recomputes an SGP,
    so leave the realized $ for manual review); else pending. Boosted payout."""
    legs = ticket.get("legs", [])
    results = [lg.get("result") for lg in legs]
    actuals_known = all((lg.get("result") is not None or lg.get("actual") is not None)
                        for lg in legs) and legs
    if any(r == 0 for r in results):
        stake = float(ticket.get("stake") or 0.0)
        return "lost", 0.0, -stake
    if not actuals_known:
        return "pending", None, None
    if any(r is None and lg.get("actual") is not None      # a push leg among resolved
           for r, lg in zip(results, legs)):
        return "void", None, None
    # all legs won
    stake = float(ticket.get("stake") or 0.0)
    boost = float(ticket.get("boost_pct") or 0.0)
    dec = bonuslib.american_to_dec(ticket.get("combined_american") or 100)
    profit = stake * (dec - 1.0) * (1.0 + boost)
    return "won", stake + profit, profit


def settle_manual(parlay_id, status):
    """Manually settle a ticket (won|lost|void|push|pending) — for pushes/overrides the
    auto-grader can't resolve. Recomputes payout/profit (boosted). Returns 1 on write."""
    tickets = [t for t in load_parlays() if t.get("parlay_id") == parlay_id]
    if not tickets:
        return 0
    t = tickets[0]
    stake = float(t.get("stake") or 0.0)
    boost = float(t.get("boost_pct") or 0.0)
    if status == "won":
        dec = bonuslib.american_to_dec(t.get("combined_american") or 100)
        profit = stake * (dec - 1.0) * (1.0 + boost)
        payout = stake + profit
    elif status == "lost":
        profit, payout = -stake, 0.0
    elif status == "pending":
        profit, payout = None, None
    else:                                              # void | push
        profit, payout = 0.0, stake

    def up(rows):
        for r in rows:
            if r.get("parlay_id") == parlay_id:
                r.update({"status": status, "profit": profit, "payout": payout,
                          "settled_at": (None if status == "pending" else _now())})
                return 1
        return 0

    return recalibration.mutate_ndjson_log(PARLAYS_FILE, up, where={"parlay_id": parlay_id})


def update_stake(parlay_id, stake):
    """Edit a ticket's wager. If already settled won/lost, recompute payout/profit (boosted)
    from the new stake. Returns 1 on write."""
    tickets = [t for t in load_parlays() if t.get("parlay_id") == parlay_id]
    if not tickets:
        return 0
    t = tickets[0]
    stake = float(stake)
    upd = {"stake": stake}
    if t.get("status") == "won":
        dec = bonuslib.american_to_dec(t.get("combined_american") or 100)
        profit = stake * (dec - 1.0) * (1.0 + float(t.get("boost_pct") or 0.0))
        upd["profit"], upd["payout"] = profit, stake + profit
    elif t.get("status") == "lost":
        upd["profit"], upd["payout"] = -stake, 0.0
    elif t.get("status") in ("void", "push"):
        # A fully void/push ticket refunds the stake in full — recompute the refund
        # from the NEW stake (was left at the old refund). [F06] Matches
        # settle_manual's void handling (profit 0, payout = stake).
        upd["profit"], upd["payout"] = 0.0, stake

    def up(rows):
        for r in rows:
            if r.get("parlay_id") == parlay_id:
                r.update(upd)
                return 1
        return 0

    return recalibration.mutate_ndjson_log(PARLAYS_FILE, up, where={"parlay_id": parlay_id})


def delete_parlay(parlay_id):
    """Remove a ticket and its legs (e.g. a mistaken log). Returns 1 if the ticket was found."""
    def drop(rows):
        keep = [r for r in rows if r.get("parlay_id") != parlay_id]
        removed = len(rows) - len(keep)
        rows[:] = keep
        return removed

    recalibration.mutate_ndjson_log(LEGS_FILE, drop, where={"parlay_id": parlay_id})
    return recalibration.mutate_ndjson_log(PARLAYS_FILE, drop, where={"parlay_id": parlay_id})


def grade_parlays():
    """Grade every pending parlay: resolve unresolved legs, settle the ticket. Returns the
    number of tickets newly settled. Best-effort; per-ticket failures are skipped."""
    settled = 0
    for t in load_parlays():
        if t.get("status") != "pending":
            continue
        pid = t.get("parlay_id")
        changed_legs = []
        for lg in t["legs"]:
            if lg.get("result") is None and lg.get("actual") is None:
                actual, result = _grade_leg(lg)
                if actual is not None:
                    lg = {**lg, "actual": str(actual), "result": result}
                    changed_legs.append(lg)
        # persist any newly-graded legs
        if changed_legs:
            def up_legs(rows, _cl=changed_legs):
                by = {(r.get("parlay_id"), r.get("leg_index")): r for r in rows}
                for lg in _cl:
                    r = by.get((pid, lg["leg_index"]))
                    if r is not None:
                        r.update({"actual": lg["actual"], "result": lg["result"]})
                return len(_cl)
            recalibration.mutate_ndjson_log(LEGS_FILE, up_legs, where={"parlay_id": pid})
            # refresh the in-memory legs for settlement
            merged = {lg["leg_index"]: lg for lg in changed_legs}
            t["legs"] = [merged.get(lg.get("leg_index"), lg) for lg in t["legs"]]
        status, payout, profit = _settle_fields(t)
        if status != "pending":
            def up_ticket(rows, _st=status, _po=payout, _pf=profit):
                for r in rows:
                    if r.get("parlay_id") == pid:
                        r.update({"status": _st, "payout": _po, "profit": _pf,
                                  "settled_at": _now()})
                        return 1
                return 0
            recalibration.mutate_ndjson_log(PARLAYS_FILE, up_ticket, where={"parlay_id": pid})
            settled += 1
    return settled
