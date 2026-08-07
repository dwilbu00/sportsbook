"""Durable bankroll ledger + a small key/value settings store.

The wagers ledger (wagers.py) records every real bet and its realized profit;
this module turns that stream of profits into a running bankroll the app can
size Kelly stakes against, and lets the user reconcile it to their real-world
balance with a single number.

Two things live here, both on top of the same NDJSON-store machinery the wagers
ledger uses (a SQLAlchemy table behind ``recalibration.mutate_ndjson_log`` /
``_read_ndjson_blob`` -- no new secret, SQL in prod with a Blob/local fallback):

1. The bankroll ledger (``bankroll_ledger.jsonl``). One signed transaction per
   row; the current bankroll is the SUM of every ``amount``. The balance is
   never stored, only derived, so a re-graded or deleted wager can never leave a
   stale running total behind. Two kinds of transaction:
     * ``bet``        -- one per settled wager, amount = its realized profit,
                         txn_id = ``bet:<wager_id>``. These are maintained by an
                         idempotent reconcile sweep (``reconcile_bet_txns``), NOT
                         by hooking wagers.py, so the ledger self-heals across
                         settle / re-grade / delete with no coupling.
     * ``adjustment`` -- a manual deposit / withdrawal / correction. The user
                         types their real current bankroll; we write ONE txn for
                         the signed difference ``target - current`` so the derived
                         balance becomes exactly the target. Example: ledger says
                         $700, the user withdraws $200 and enters $500 -> write
                         -$200; re-deposits the next day and enters $700 -> +$200.

2. Durable Kelly sizing settings (``app_settings.jsonl``, a generic KV store) so
   the fraction / per-bet cap / slate-total cap persist across sessions, not just
   across page switches within one session.

Every public entry point is best-effort and never raises into the app.
"""
from datetime import datetime, timezone

import recalibration

BANKROLL_FILE = "bankroll_ledger.jsonl"
SETTINGS_FILE = "app_settings.jsonl"

# Statuses a wager must reach before it contributes a realized-profit txn. Mirrors
# wagers._SETTLED (won/lost pay profit; push/void are 0.0). Kept local so this
# module never imports wagers at import time (avoids a cycle; the reconcile sweep
# imports it lazily).
_SETTLED = ("won", "lost", "push", "void")

# The Kelly sizing knobs persisted across sessions (the bankroll itself is the
# ledger balance, not a stored setting). Values are stored as strings in the KV
# table and parsed back to float on read.
_KELLY_SETTING_KEYS = ("kelly_fraction", "kelly_cap_pct", "kelly_slate_cap_pct")
_KELLY_SETTING_DEFAULTS = {
    "kelly_fraction": 0.5,
    "kelly_cap_pct": 5.0,
    "kelly_slate_cap_pct": 25.0,
}


def storage_backend():
    """Human-readable active backend (mirrors the wagers / prediction log)."""
    return recalibration.prediction_log_storage()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _amount(row):
    """Signed dollar amount of a txn row, 0.0 when missing/unparseable."""
    try:
        return round(float(row.get("amount") or 0.0), 2)
    except (TypeError, ValueError):
        return 0.0


def txn_amount(row):
    """Public alias for a txn's normalized signed amount (for the history view)."""
    return _amount(row)


# ------------------------------------------------------------------------------
# Reading the ledger
# ------------------------------------------------------------------------------

def read_ledger(use_cache=True):
    """All bankroll transactions (best-effort snapshot); [] on any error."""
    try:
        rows, _ = recalibration._read_ndjson_blob(BANKROLL_FILE, use_cache=use_cache)
        return rows or []
    except Exception:
        return []


def read_ledger_with_status():
    """(rows, error): like read_ledger but surfaces a backend failure so the UI
    can tell an unreachable durable store from a genuinely empty ledger (both
    otherwise look like []). ``error`` is None on success."""
    try:
        rows, _ = recalibration._read_ndjson_blob(BANKROLL_FILE, use_cache=True)
        return rows or [], None
    except Exception as exc:
        return [], exc


def current_balance(rows=None):
    """Derived bankroll = SUM(amount) over all txns; 0.0 on empty/error."""
    if rows is None:
        rows = read_ledger()
    return round(sum(_amount(r) for r in rows), 2)


def summary(rows=None):
    """Roll-up for the UI: balance and the split between realized bet P/L and
    manual adjustments, plus the raw txns (newest first) for a history view."""
    if rows is None:
        rows = read_ledger()
    bets_total = round(
        sum(_amount(r) for r in rows if r.get("txn_type") == "bet"), 2)
    adj_total = round(
        sum(_amount(r) for r in rows if r.get("txn_type") == "adjustment"), 2)
    ordered = sorted(rows, key=lambda r: (r.get("created_at") or ""), reverse=True)
    return {
        "balance": round(bets_total + adj_total, 2),
        "bets_total": bets_total,
        "adjustments_total": adj_total,
        "n_txns": len(rows),
        "txns": ordered,
    }


# ------------------------------------------------------------------------------
# Manual adjustments (deposit / withdrawal / correction by target value)
# ------------------------------------------------------------------------------

def _adjust_note(current, target, delta):
    verb = "deposited" if delta > 0 else "withdrew"
    return "Balance set to $%.2f (%s $%.2f)" % (target, verb, abs(delta))


def record_adjustment(target, note=None):
    """Write ONE adjustment txn so the derived balance becomes ``target``.

    The amount written is the signed difference ``target - current_balance``,
    computed atomically inside the mutator off the authoritative rows (never a
    cached read). A target within half a cent of the current balance is a no-op.
    Returns the signed delta actually written (0.0 when nothing was written)."""
    try:
        target = round(float(target), 2)
    except (TypeError, ValueError):
        return 0.0

    written = {"delta": 0.0}

    def add(rows):
        current = round(sum(_amount(r) for r in rows), 2)
        delta = round(target - current, 2)
        if abs(delta) < 0.005:
            return 0
        ts = _now_iso()
        # Unique txn_id even if two adjustments land in the same ISO instant.
        have = {r.get("txn_id") for r in rows}
        n = sum(1 for t in have if str(t or "").startswith("adj:"))
        txn_id = "adj:%s#%d" % (ts, n)
        while txn_id in have:
            n += 1
            txn_id = "adj:%s#%d" % (ts, n)
        rows.append({
            "txn_id": txn_id,
            "txn_type": "adjustment",
            "amount": delta,
            "wager_id": None,
            "note": note or _adjust_note(current, target, delta),
            "created_at": ts,
        })
        written["delta"] = delta
        return 1

    try:
        recalibration.mutate_ndjson_log(BANKROLL_FILE, add)
    except Exception:
        return 0.0
    return written["delta"]


# ------------------------------------------------------------------------------
# Reconciling realized bet P/L into the ledger (idempotent sweep)
# ------------------------------------------------------------------------------

def _bet_note(status):
    return str(status or "")


def reconcile_bet_txns(wager_rows=None):
    """Sync ``bet:<wager_id>`` txns to the CURRENT settled-wager P/L.

    For every settled wager there should be exactly one ``bet`` txn whose amount
    equals its realized profit; every ``bet`` txn for a wager that is no longer
    settled (re-graded back to pending) or no longer exists (deleted) is removed.
    Adjustment txns are never touched. Idempotent: a second call with no wager
    change writes nothing and returns 0.

    ``wager_rows`` may be passed by a caller that already read the ledger (My
    Bets) to avoid a second read; otherwise the wagers ledger is read lazily.
    Best-effort -- returns the change count, 0 on any error."""
    if wager_rows is None:
        try:
            import wagers
            wager_rows = wagers.read_wagers()
        except Exception:
            return 0

    desired = {}   # txn_id -> {amount, wager_id, status, resolved_at}
    for w in (wager_rows or []):
        wid = w.get("wager_id")
        if not wid or w.get("status") not in _SETTLED:
            continue
        try:
            amt = round(float(w.get("profit") or 0.0), 2)
        except (TypeError, ValueError):
            amt = 0.0
        desired["bet:%s" % wid] = {
            "amount": amt,
            "wager_id": wid,
            "status": w.get("status"),
            "resolved_at": w.get("resolved_at"),
        }

    def sync(rows):
        changed = 0
        by_txn = {r.get("txn_id"): r for r in rows
                  if str(r.get("txn_id") or "").startswith("bet:")}
        # Upsert a bet txn for each settled wager.
        for txn_id, d in desired.items():
            note = _bet_note(d["status"])
            existing = by_txn.get(txn_id)
            if existing is None:
                rows.append({
                    "txn_id": txn_id,
                    "txn_type": "bet",
                    "amount": d["amount"],
                    "wager_id": d["wager_id"],
                    "note": note,
                    "created_at": d.get("resolved_at") or _now_iso(),
                })
                changed += 1
            elif (_amount(existing) != d["amount"]
                    or existing.get("note") != note
                    or existing.get("txn_type") != "bet"
                    or existing.get("wager_id") != d["wager_id"]):
                existing["amount"] = d["amount"]
                existing["note"] = note
                existing["txn_type"] = "bet"
                existing["wager_id"] = d["wager_id"]
                changed += 1
        # Drop bet txns whose wager is no longer settled (or gone).
        stale = [t for t in by_txn if t not in desired]
        if stale:
            drop = set(stale)
            rows[:] = [r for r in rows if r.get("txn_id") not in drop]
            changed += len(stale)
        return changed

    try:
        return recalibration.mutate_ndjson_log(BANKROLL_FILE, sync) or 0
    except Exception:
        return 0


# ------------------------------------------------------------------------------
# Durable Kelly sizing settings (generic KV store)
# ------------------------------------------------------------------------------

def load_kelly_settings():
    """The persisted Kelly knobs as floats, falling back to the shipped defaults
    for any key not stored yet. Never raises."""
    out = dict(_KELLY_SETTING_DEFAULTS)
    try:
        rows, _ = recalibration._read_ndjson_blob(SETTINGS_FILE, use_cache=True)
    except Exception:
        return out
    for r in (rows or []):
        key = r.get("setting_key")
        if key not in _KELLY_SETTING_DEFAULTS:
            continue
        try:
            out[key] = float(r.get("setting_value"))
        except (TypeError, ValueError):
            continue
    return out


def save_kelly_settings(fraction, cap_pct, slate_cap_pct):
    """Upsert the three Kelly knobs into the KV store. Best-effort; returns the
    number of settings written/updated (0 on no-op or error)."""
    values = {
        "kelly_fraction": fraction,
        "kelly_cap_pct": cap_pct,
        "kelly_slate_cap_pct": slate_cap_pct,
    }
    clean = {}
    for key, value in values.items():
        try:
            clean[key] = "%s" % float(value)
        except (TypeError, ValueError):
            continue
    if not clean:
        return 0

    def upsert(rows):
        ts = _now_iso()
        by_key = {r.get("setting_key"): r for r in rows}
        changed = 0
        for key, value in clean.items():
            existing = by_key.get(key)
            if existing is None:
                rows.append({
                    "setting_key": key,
                    "setting_value": value,
                    "updated_at": ts,
                })
                changed += 1
            elif existing.get("setting_value") != value:
                existing["setting_value"] = value
                existing["updated_at"] = ts
                changed += 1
        return changed

    try:
        return recalibration.mutate_ndjson_log(SETTINGS_FILE, upsert) or 0
    except Exception:
        return 0
