"""
Self-updating Platt recalibration for player-prop probabilities.

What it does
------------
1. Logs every published over-probability the analyzer produces to a JSONL
   prediction log (`cache/predictions/prediction_log.jsonl`).
2. Resolves outcomes for past-dated log entries against cached ESPN gamelogs
   (idempotent: marks entries with `actual`, `outcome`, `resolved=True`).
3. Fits a per-(sport, prop) Platt sigmoid mapping raw model probability to a
   recalibrated probability, using only resolved entries as training data.
4. Saves the fit to `calibration/recalibration_<sport>.json`.
5. On app launch, auto-refits when enough new resolved entries exist since the
   last fit (and not more often than every `MIN_REFIT_INTERVAL_HOURS`).

Platt math
----------
Recalibrated probability p_cal = sigmoid(a * logit(p_raw) + b)
where (a, b) are fit by Newton-Raphson on cross-entropy loss over
holdout (raw_prob, outcome) pairs.

Schema for recalibration_<sport>.json
-------------------------------------
{
  "sport_key": "basketball_nba",
  "fit_timestamp": "2026-05-29T12:34:56Z",
  "props": {
    "player_points": {"a": 0.91, "b": -0.18, "n_fit": 412},
    ...
  }
}
"""
import copy
import json
import math
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
PRED_DIR = os.path.join(CACHE_DIR, "predictions")
LOG_PATH = os.path.join(PRED_DIR, "prediction_log.jsonl")
CALIB_DIR = os.path.join(SCRIPT_DIR, "calibration")

MIN_FIT_SAMPLES = 50          # below this, skip Platt fit for a prop
MIN_VALIDATION_SAMPLES = 20   # later chronological observations held out
MIN_NEW_FOR_REFIT = 25        # need this many new resolved obs to bother refitting
MIN_REFIT_INTERVAL_HOURS = 12 # don't re-resolve+refit more than this often
MAX_RESOLVE_PER_LAUNCH = 80   # cap ESPN calls per auto-refit cycle
# A self-learned (loop) fit may override a *seeded* prop only after clearing this
# obs floor and beating the seed out-of-sample (the "wait longer" gate). Well
# above the base fit gate (MIN_FIT_SAMPLES + 2*MIN_VALIDATION_SAMPLES = 90).
MIN_OBS_FOR_OVERRIDE = 300
# Prior strength (× the seed's n_fit) in the seed↔loop shrinkage blend weight
# w = n_loop / (n_loop + RECAL_SEED_TRUST*n_seed). Higher = slower takeover, so a
# single loop never moves the applied map drastically off the book-line seed.
RECAL_SEED_TRUST = 1.0
# New RESOLVED predictions (refit_performed=0) that make the app suggest an
# OFFLINE calibration refit. Higher than the online Platt gate (that's a cheap
# 2-param nudge; this triggers a full offline method re-selection).
MIN_NEW_FOR_OFFLINE_REFIT = 200
# A prediction whose game is at least this old AND whose player never played it
# (a confirmed scratch/DNP) is permanently unresolvable → voided so it stops
# re-attempting and clears out of forward-tracking's pending list. A played
# game's box score posts within hours, so 24h is ample for the common case. The
# one edge it doesn't preserve — a postponement replayed as a next-day makeup —
# is a non-issue: DK voids that bet, and one throwaway makeup calibration label
# is negligible.
STALE_DNP_HOURS = 24
AUTO_MAINTENANCE_INTERVAL_SECONDS = 3600
RECAL_LOAD_TTL_SECONDS = 300  # in-memory reuse before re-checking the recal store

_lock = threading.Lock()
_last_auto_maintenance = {}  # sport_key -> attempt timestamp

# Short-TTL in-memory cache for read-only NDJSON reads (e.g. the wagers
# ledger read on every My Bets rerun). Only the SQL path is cached; local
# disk reads are already cheap. Every writer goes through mutate_ndjson_log,
# which reads FRESH (use_cache=False) and pops this cache after a successful
# write, so cached reads never mask a just-persisted change within a session.
_NDJSON_CACHE = {}            # filename -> (rows, version, fetched_at)
_NDJSON_CACHE_TTL = 30        # seconds


# ── SQL backend (Azure SQL) dispatch ──
# When db_store is importable AND its SQL_* secret is configured, the durable
# stores here (prediction log, wagers ledger, recalibration params) route to SQL
# instead of local disk. A missing SQLAlchemy install or unset secret leaves
# _sql() False → the local-disk path is used unchanged. The row-list mutators are
# reused verbatim; db_store runs them inside a transaction and replaces the
# store's rows (its CHECK/UNIQUE constraints reject bad data).
try:
    import db_store as _db
except Exception:  # pragma: no cover - SQLAlchemy absent
    _db = None

# Structured DB-failure telemetry (WS1b). Guarded so a partial deploy without the
# module degrades to a silent no-op rather than breaking the store.
try:
    import ops_telemetry as _ops
except Exception:  # pragma: no cover
    class _ops:  # noqa: N801 - tiny no-op stand-in
        @staticmethod
        def ops_event(*a, **k):
            pass

_PRED_TABLE = "prediction_log"


def _sql():
    return _db is not None and _db.enabled()


# F1 guard: a durable write must fail LOUDLY when SQL is off but a SQL deployment
# is configured, instead of silently degrading to ephemeral local disk. Env-only
# so it is inert in every hermetic test and no-SQL dev run. The _db-is-None branch
# (SQLAlchemy absent but SQL_* configured = a prod misconfig we must still catch)
# reimplements require_sql() from the environment, since _db.require_sql() is
# unreachable then.
_REQUIRE_SQL_ENV = "SPORTSBOOK_REQUIRE_SQL"
_SQL_SECRET_KEYS = ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD")


def _require_sql():
    if _db is not None:
        return _db.require_sql()
    flag = os.environ.get(_REQUIRE_SQL_ENV, "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    return any(os.environ.get(k, "").strip() for k in _SQL_SECRET_KEYS)


def _ensure_durable(op):
    """Raise if a durable ``op`` would hit ephemeral local disk in a prod context.
    No-op unless SQL is off AND a SQL deployment is signalled — so it never fires
    in tests, no-SQL dev, or healthy prod."""
    if _sql() or not _require_sql():
        return
    raise RuntimeError(
        f"Refusing to {op}: the SQL backend is not enabled but a SQL deployment "
        f"is configured (SPORTSBOOK_REQUIRE_SQL or SQL_* secrets present). Writing "
        f"to ephemeral local disk would silently lose data. Fix the SQL_* secrets "
        f"/ the db_store import, or set SPORTSBOOK_REQUIRE_SQL=0 for intentional "
        f"local use.")


def _table_for(filename):
    """Map an NDJSON store filename to its SQL table (e.g. 'wagers.jsonl'
    → 'wagers')."""
    return os.path.splitext(filename)[0]


# ──────────────────────────────────────────────────────────────────────────────
# Path / file helpers
# ──────────────────────────────────────────────────────────────────────────────

def recalibration_path(sport_key):
    return os.path.join(CALIB_DIR, f"recalibration_{sport_key}.json")


def _ensure_dirs():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)


def prediction_log_storage():
    """Human-readable active prediction-log backend."""
    return "Azure SQL" if _sql() else "Local cache"


def _parse_log_text(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _serialize_log(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


class _LogConflict(Exception):
    pass


@contextmanager
def _local_file_lock(path):
    """Hold an inter-process lock while replacing a local NDJSON file.

    Generalizes the prediction-log lock to any sibling NDJSON store (e.g.
    wagers.jsonl) so concurrent local writers serialize on the same file."""
    _ensure_dirs()
    lock_path = path + ".lock"
    with open(lock_path, "a+b") as lock_file:
        if os.name == "nt":
            import msvcrt
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _local_log_lock():
    """Hold an inter-process lock while replacing the local prediction log."""
    with _local_file_lock(LOG_PATH):
        yield


def _read_log_snapshot(where=None):
    """Return (rows, version) from SQL or local disk.

    ``where`` (SQL path only) is an equality/IN {col: value} filter; the local
    path ignores it (single-file NDJSON has no partial read) and the caller
    self-filters in Python."""
    if _sql():
        return _db.read_rows(_PRED_TABLE, where=where), None
    if not os.path.exists(LOG_PATH):
        return [], None
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return _parse_log_text(f.read()), None


def _write_log_snapshot(rows, version=None):
    """Write a complete log snapshot atomically (local disk).

    ``version`` is accepted for signature stability but ignored on the local path."""
    content = _serialize_log(rows)
    _ensure_dirs()
    tmp = LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, LOG_PATH)


def mutate_prediction_log(mutator, max_retries=5, where=None):
    """Atomically mutate the prediction log; returns the mutator's result.

    ``where`` (SQL path only) restricts the rows read into the mutator to a subset
    (see db_store.mutate) — for update-only passes like grading unresolved rows.
    The local path ignores it (reads all); the mutator must self-filter."""
    _ensure_durable("write the prediction log")
    if _sql():
        # Serialize with the module lock, mirroring the local path: SQL's
        # read->mutate->replace is a read-modify-write, so concurrent threads in
        # the (single-replica) Streamlit process must not interleave and lose an
        # update. db_store.mutate's transaction is the atomicity guarantee; this
        # lock is the in-process concurrency guard.
        with _lock:
            return _db.mutate(_PRED_TABLE, mutator, where=where)
    with _lock:
        with _local_log_lock():
            rows, version = _read_log_snapshot()
            result = mutator(rows)
            if result:
                _write_log_snapshot(rows, version)
            return result


def count_pending_refit(sport_key=None):
    """Count RESOLVED prediction-log rows not yet consumed by an offline refit
    (refit_performed falsy). This is the app's "enough new labeled data to refit"
    signal — cheap SQL COUNT on the SQL path, in-memory count on the local
    path. Best-effort: returns 0 on any error (a banner must never break a page)."""
    where = {"resolved": True, "refit_performed": False}
    if sport_key:
        where["sport_key"] = sport_key
    try:
        if _sql():
            return _db.count_rows(_PRED_TABLE, where=where)
        rows = _read_log()
        return sum(1 for r in rows
                   if r.get("resolved") and not r.get("refit_performed")
                   and (not sport_key or r.get("sport_key") == sport_key))
    except Exception:
        return 0


def mark_predictions_refit(sport_key):
    """Flag a sport's RESOLVED prediction-log rows as consumed by an offline refit
    (refit_performed -> True), resetting count_pending_refit. Call after a
    completed offline calibration refit. Returns the number flagged; best-effort.

    Surgical on the SQL path (WHERE resolved=1 AND refit_performed=0); the mutator
    self-filters on resolved so the where-ignoring local path is also correct."""
    where = {"sport_key": sport_key, "resolved": True, "refit_performed": False}

    def apply(rows):
        changed = 0
        for r in rows:
            if not r.get("resolved") or r.get("refit_performed"):
                continue
            if not _sql() and sport_key and r.get("sport_key") != sport_key:
                continue    # local path ignores `where` -> self-filter by sport
            r["refit_performed"] = True
            changed += 1
        return changed

    try:
        return mutate_prediction_log(apply, where=where)
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Generalized NDJSON store (sibling files: wagers.jsonl, etc.)
# ──────────────────────────────────────────────────────────────────────────────
# The prediction log's read-modify-write is reusable for any small NDJSON store.
# These helpers parameterize it by filename (SQL table on the SQL path, sibling
# file on the local path), mirroring the prediction-log path exactly. The
# dedicated prediction-log functions above are left untouched.

def _ndjson_local_path(filename):
    return os.path.join(PRED_DIR, filename)


def _read_ndjson_blob(filename, use_cache=False, where=None):
    """Return (rows, version) for an NDJSON store, from SQL or local disk.

    ``use_cache`` (read-only callers only) serves the read from a short-TTL
    in-memory cache to avoid a full read+parse on every rerun. Writers
    (mutate_ndjson_log) MUST leave it False so the read-modify-write always sees
    the authoritative store.

    ``where`` (SQL path only) is an equality/IN {col: value} filter that pulls only
    matching rows — for reconciliation callers that need just the ungraded subset.
    A filtered read never uses or populates the full-read cache (its result is a
    subset), and the local path ignores it (a single NDJSON file has no
    partial read; those callers self-filter in Python)."""
    cacheable = use_cache and where is None
    if _sql():
        if cacheable:
            entry = _NDJSON_CACHE.get(filename)
            if entry and (time.time() - entry[2]) < _NDJSON_CACHE_TTL:
                return copy.deepcopy(entry[0]), entry[1]
        rows = _db.read_rows(_table_for(filename), where=where)
        if cacheable:
            _NDJSON_CACHE[filename] = (copy.deepcopy(rows), None, time.time())
        return rows, None
    path = _ndjson_local_path(filename)
    if not os.path.exists(path):
        return [], None
    with open(path, "r", encoding="utf-8") as f:
        return _parse_log_text(f.read()), None


def _write_ndjson_blob(filename, rows, version=None):
    """Write a complete NDJSON snapshot atomically (local disk).

    ``version`` is accepted for signature stability but ignored on the local path."""
    content = _serialize_log(rows)
    path = _ndjson_local_path(filename)
    _ensure_dirs()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def mutate_ndjson_log(filename, mutator, max_retries=5, where=None):
    """Atomically mutate an arbitrary NDJSON store; returns the mutator's result.

    Generalizes mutate_prediction_log to any sibling NDJSON file. The mutator
    receives the row list and mutates it in place; a falsy return skips the
    write. Local writers hold an inter-process file lock.

    ``where`` (SQL path only) restricts the rows read into the mutator to a subset
    (see db_store.mutate) — for update-only reconciliation passes. The local
    path ignores it and reads all rows, so the mutator must be correct on the full
    set (it self-filters)."""
    _ensure_durable(f"write {filename}")
    if _sql():
        with _lock:  # in-process serialization of read->mutate->replace
            result = _db.mutate(_table_for(filename), mutator, where=where)
        if result:
            _NDJSON_CACHE.pop(filename, None)
        return result
    with _lock:
        with _local_file_lock(_ndjson_local_path(filename)):
            rows, version = _read_ndjson_blob(filename)
            result = mutator(rows)
            if result:
                _write_ndjson_blob(filename, rows, version)
                _NDJSON_CACHE.pop(filename, None)
            return result


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def log_prediction(sport_key, prop_key, player, game_date, line, raw_prob,
                   projected=None, direction=None, price=None, book=None,
                   final_prob=None, event_id=None, commence_time=None,
                   is_value=None, team=None, batting_order=None,
                   mlb_player_id=None, game_pk=None, ids_resolved=False,
                   source=None, write=True):
    """Build and optionally append one prediction row. Best-effort, never raises.

    ``team`` and ``batting_order`` are the rule inputs the pick-rules ROI lens
    (pickrules_roi.py) re-derives the recommended slate from — they are not used
    by calibration. Both are optional; older rows logged without them leave the
    lens's team-based rules (Rule-of-3, opposing-team L3) reported as skipped.

    ``mlb_player_id``/``game_pk``/``ids_resolved`` (P3): when the caller resolved
    the player at the odds boundary via entity_resolver (``ids_resolved=True``),
    the resolved MLBAM id (may be None on a fail-closed miss) + game_pk are stamped
    verbatim and the weaker single-team SFBB fallback in _enrich_prediction_ids is
    SUPPRESSED — a miss must stay NULL (the P3 shadow signal), not be re-guessed.
    """
    if not sport_key or not prop_key or not player or game_date is None:
        return
    try:
        raw_prob = float(raw_prob)
        line = float(line)
        projected = float(projected) if projected is not None else None
        final_prob = float(final_prob) if final_prob is not None else None
        price = int(price) if price is not None else None
    except (TypeError, ValueError):
        return
    # Auxiliary lens input: a malformed batting_order must not drop the whole
    # forecast (which calibration needs) — coerce to int-or-None best-effort.
    try:
        batting_order = int(batting_order) if batting_order is not None else None
    except (TypeError, ValueError):
        batting_order = None
    if not (0.0 <= raw_prob <= 1.0):
        return
    if final_prob is not None and not (0.0 <= final_prob <= 1.0):
        final_prob = None
    _ensure_dirs()
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "sport_key": sport_key,
        "event_id": event_id,
        "commence_time": commence_time,
        "prop_key": prop_key,
        "player": player,
        "game_date": str(game_date)[:10],  # YYYY-MM-DD
        "line": line,
        "raw_prob": raw_prob,
        "final_prob": final_prob,
        "projected": projected,
        "direction": direction,
        "price": price,
        "book": book,
        "team": team or None,
        "batting_order": batting_order,
        "is_value": bool(is_value) if is_value is not None else None,
        "resolved": False,
        "actual": None,
        "outcome": None,    # 1=over_won, 0=under_won, None=push/unresolved
    }
    row["game_pk"] = game_pk
    row["source"] = source       # data path that served the model-input history
    if ids_resolved:
        row["player_mlb_id"] = mlb_player_id     # resolver-authoritative (may be None)
    _enrich_prediction_ids(row, trust_ids=ids_resolved)
    if write:
        log_prediction_rows([row])
    return row


def _enrich_prediction_ids(row, trust_ids=False):
    """Best-effort: stamp the SFBB MLBAM id + canonical team code onto a prediction
    row, then compute its hybrid player_key. MLB-only (an id-space that exists only
    for baseball; other sports keep player_mlb_id/team_code NULL and player_key
    falls back to name:<norm>). Fail-open — a missing map / SQL-off / unknown or
    ambiguous name leaves the id columns NULL. Never raises (log_prediction is
    best-effort); mutates ``row`` in place and returns it.

    ``trust_ids`` (P3): when True the caller already resolved player_mlb_id at the
    odds boundary (two-team-hinted, fail-closed) — keep it VERBATIM and skip the
    weaker single-team fallback here (which could bind where the stronger resolver
    refused). team_code is still stamped; player_key is computed after either way."""
    row.setdefault("player_mlb_id", None)
    row.setdefault("team_code", None)
    try:
        if (row.get("sport_key") or "").startswith("baseball"):
            import player_id_map
            if not trust_ids:
                row["player_mlb_id"] = player_id_map.mlb_id_for_name(
                    row.get("player"), teams=row.get("team"))
            if row.get("team"):
                row["team_code"] = player_id_map.team_code_for_name(row.get("team"))
    except Exception:                       # pragma: no cover - never break logging
        pass
    try:
        import db_store
        row["player_key"] = db_store.player_key(row)
    except Exception:                       # pragma: no cover
        row["player_key"] = None
    return row


def log_prediction_rows(new_rows):
    """Append prediction rows, de-duplicating by forecast identity.

    Re-viewing the same slate would otherwise append a fresh row for every
    (sport, event, prop, player, line) on each analysis, growing the log
    without bound and rewriting the whole log every append. Instead we keep a
    single row per forecast identity: the newest forecast supersedes stale
    unresolved duplicates, while any graded outcome on an older duplicate is
    folded forward so scoring survives. A forecast whose outcome is already
    resolved is never overwritten by a later re-log.

    Returns the number of log rows added or changed.
    """
    if not new_rows:
        return 0
    try:
        def upsert(rows):
            incoming = {}
            for row in new_rows:
                incoming[prediction_identity(row)] = row  # last in batch wins
            incoming_idents = set(incoming)
            existing = defaultdict(list)
            for row in rows:
                ident = prediction_identity(row)
                if ident in incoming_idents:
                    existing[ident].append(row)
            kept = [row for row in rows
                    if prediction_identity(row) not in incoming_idents]
            changes = 0
            for ident, row in incoming.items():
                group = existing.get(ident, [])
                collapsed = _collapse_identity_rows(group + [row])
                kept.append(collapsed)
                # Re-logging a single already-resolved forecast changes nothing.
                no_op = (len(group) == 1 and group[0].get("resolved")
                         and collapsed == group[0])
                if not no_op:
                    changes += 1
            rows[:] = kept
            return changes
        return mutate_prediction_log(upsert)
    except Exception:
        return 0


def compact_prediction_log():
    """Collapse duplicate forecasts (same identity) into a single row each.

    Bounds historical growth of the prediction log. A no-op (returns 0, writes
    nothing) once the log already holds one row per forecast identity."""
    def compact(rows):
        order = []
        grouped = defaultdict(list)
        for row in rows:
            ident = prediction_identity(row)
            if ident not in grouped:
                order.append(ident)
            grouped[ident].append(row)
        merged = [_collapse_identity_rows(grouped[ident]) for ident in order]
        removed = len(rows) - len(merged)
        if removed <= 0:
            return 0
        rows[:] = merged
        return removed
    try:
        return mutate_prediction_log(compact)
    except Exception:
        return 0


def _read_log(where=None):
    try:
        rows, _ = _read_log_snapshot(where)
        return rows
    except Exception:
        return []


def read_prediction_log():
    """Public read-only snapshot for maintenance tools."""
    return _read_log()


def prediction_identity(row):
    """Stable forecast identity, keyed on the hybrid player_key (mlb:<id> else
    name:<norm>) so two spellings of one player collapse to a single forecast.

    Must stay in lockstep with db_store._prediction_identity — both compute the
    key via db_store.player_key so the mutator's dedup and the surgical-diff
    layer agree. On the local path (SQLAlchemy absent, ``_db`` is None) we
    fall back to the raw player name: those stores were never re-keyed and the
    id columns don't exist there (the id-based identity is a SQL-only invariant).
    Legacy fallback for pre-event-ID rows preserved via event_ref."""
    event_ref = row.get("event_id") or row.get("game_date")
    player = _db.player_key(row) if _db is not None else row.get("player")
    return (
        row.get("sport_key"), event_ref, row.get("prop_key"),
        player, row.get("line"),
    )


def prediction_row_key(row):
    """Identity for updating one physical log row, including its timestamp."""
    return (row.get("ts"),) + prediction_identity(row)


def _merge_outcome_fields(dest, src):
    """Fold resolved-outcome fields from `src` into `dest` when `dest` lacks
    them, so de-duplicating never drops a graded result."""
    if not dest.get("resolved") and src.get("resolved"):
        for key in ("resolved", "actual", "outcome", "resolved_at"):
            if key in src:
                dest[key] = src[key]


def _collapse_identity_rows(group):
    """Merge duplicate rows that share a prediction identity into one row.

    Prefers the most recent *resolved* forecast (else the most recent forecast)
    as the base, then folds in outcome fields captured on any sibling. Returns
    the sole row unchanged when there is nothing to merge."""
    if len(group) == 1:
        return group[0]

    def _recency(row):
        return row.get("resolved_at") or row.get("ts") or ""

    resolved = [row for row in group if row.get("resolved")]
    base = dict(max(resolved or group, key=_recency))
    for row in group:
        if row is not base:
            _merge_outcome_fields(base, row)
    return base


def summarize_prediction_rows(rows, sport_key=None):
    """Summarize deduplicated forward predictions without making API calls."""
    unique = {}
    for row in rows:
        if sport_key and row.get("sport_key") != sport_key:
            continue
        identity = prediction_identity(row)
        current = unique.get(identity)
        if current and current.get("resolved") and not row.get("resolved"):
            continue
        unique[identity] = row
    predictions = list(unique.values())

    def metrics(group):
        resolved = [row for row in group if row.get("resolved")]
        graded = [row for row in resolved if row.get("outcome") in (0, 1)]
        probabilities = []
        outcomes = []
        direction_hits = []
        realized_returns = []
        for row in graded:
            try:
                logged_probability = row.get("final_prob")
                if logged_probability is None:
                    logged_probability = row.get("raw_prob")
                probability = float(logged_probability)
            except (TypeError, ValueError):
                probability = None
            if probability is not None and 0.0 <= probability <= 1.0:
                probabilities.append(probability)
                outcomes.append(int(row["outcome"]))
            direction = (row.get("direction") or "").upper()
            if direction == "OVER":
                direction_hits.append(row["outcome"] == 1)
            elif direction == "UNDER":
                direction_hits.append(row["outcome"] == 0)
            else:
                continue
            try:
                price = int(row.get("price"))
            except (TypeError, ValueError):
                continue
            if price == 0:
                continue
            won = direction_hits[-1]
            profit = (price / 100.0 if price > 0 else 100.0 / -price)
            realized_returns.append(profit if won else -1.0)
        # A resolved push returns the stake. Include it in priced ROI even
        # though it is excluded from binary probability scoring.
        for row in resolved:
            if row.get("outcome") is not None:
                continue
            if (row.get("direction") or "").upper() not in ("OVER", "UNDER"):
                continue
            try:
                price = int(row.get("price"))
            except (TypeError, ValueError):
                continue
            if price:
                realized_returns.append(0.0)
        brier = (sum((p - y) ** 2 for p, y in zip(probabilities, outcomes))
                 / len(outcomes) if outcomes else None)
        return {
            "total": len(group),
            "resolved": len(resolved),
            "pending": len(group) - len(resolved),
            "pushes": len(resolved) - len(graded),
            "graded": len(graded),
            "direction_hit_rate": (
                sum(direction_hits) / len(direction_hits)
                if direction_hits else None
            ),
            "probability_brier": brier,
            "priced_resolved": len(realized_returns),
            "realized_roi": (
                sum(realized_returns) / len(realized_returns)
                if realized_returns else None
            ),
        }

    summary = metrics(predictions)
    groups = defaultdict(list)
    for row in predictions:
        groups[(row.get("sport_key"), row.get("prop_key"))].append(row)
    summary["by_prop"] = [
        {
            "sport_key": key[0],
            "prop_key": key[1],
            **metrics(group),
        }
        for key, group in sorted(
            groups.items(),
            key=lambda item: tuple(str(value or "") for value in item[0]),
        )
    ]
    summary["last_prediction_ts"] = max(
        (row.get("ts") or "" for row in predictions), default="")
    return summary


def prediction_performance_summary(sport_key=None):
    """Return current forward-log status and resolved forecast performance."""
    return summarize_prediction_rows(_read_log(), sport_key=sport_key)


# ──────────────────────────────────────────────────────────────────────────────
# Team-market forward tracking (moneyline / spread / total)
# ──────────────────────────────────────────────────────────────────────────────
# Sibling of the player-prop prediction log: logs the MODEL's pick per
# (game, market) so team markets get the same forward-tracked accuracy (hit rate,
# Brier, ROI) props already have, and a durable model-side record survives a
# deleted wager. Kept a SEPARATE store (its own table/file) so the prop
# calibration/refit pipeline that reads prediction_log stays uncontaminated. One
# row = the favored side (ML/spread) or over/under lean (total) with its
# probability, price, and value flag. Resolution reuses the shared team graders
# (game_results.final_score / side_for_team / grade_team_bet), exactly as
# wagers._grade_wager.

MARKET_PREDICTION_LOG_FILE = "market_prediction_log.jsonl"


def market_prediction_identity(row):
    """Stable per-(game, market) identity, with legacy pre-event-ID fallback."""
    event_ref = row.get("event_id") or row.get("game_date")
    return (row.get("sport_key"), event_ref, row.get("bet_type"))


def market_prediction_row_key(row):
    """Identity for updating one physical row, including its timestamp."""
    return (row.get("ts"),) + market_prediction_identity(row)


def mutate_market_prediction_log(mutator, max_retries=5, where=None):
    """Atomically mutate the team-market prediction log (SQL or local)."""
    return mutate_ndjson_log(MARKET_PREDICTION_LOG_FILE, mutator,
                             max_retries=max_retries, where=where)


def _read_market_log(where=None):
    try:
        rows, _ = _read_ndjson_blob(MARKET_PREDICTION_LOG_FILE, where=where)
        return rows
    except Exception:
        return []


def read_market_prediction_log():
    """Public read-only snapshot for maintenance tools."""
    return _read_market_log()


def _mkt_num(value, default=float("-inf")):
    """Numeric key for picking the favored side; unusable values sort last."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mkt_prob01(pct):
    """A 0-100 percentage → a 0-1 probability, clamped; None if unusable."""
    try:
        p = float(pct) / 100.0
    except (TypeError, ValueError):
        return None
    return min(max(p, 0.0), 1.0)


def _mkt_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mkt_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mkt_bool(value):
    return bool(value) if value is not None else None


def _mkt_matchup(candidate, event_meta):
    if candidate.get("matchup"):
        return candidate["matchup"]
    meta = event_meta or {}
    home, away = meta.get("home_team"), meta.get("away_team")
    return f"{away} @ {home}" if home and away else None


def build_market_prediction_rows(ar, sport_key):
    """Map an analysis result to one forecast row per (event, market): the model's
    favored side (ML/spread) or over/under lean (total). Pure — no I/O, never
    raises (returns [] on any malformed input so the caller logs best-effort)."""
    if not ar or not sport_key:
        return []
    try:
        events = ar.get("events") or {}
        ts = datetime.now(timezone.utc).isoformat()
        rows = []

        def _common(event_id):
            meta = events.get(event_id) or {}
            game_date = meta.get("game_date")
            return {
                "ts": ts,
                "sport_key": sport_key,
                "event_id": event_id,
                "commence_time": meta.get("commence_time"),
                "game_date": str(game_date)[:10] if game_date else None,
                "home_team": meta.get("home_team"),
                "away_team": meta.get("away_team"),
                "resolved": False,
                "actual": None,
                "outcome": None,
                "resolved_at": None,
            }

        # Moneyline: the higher-blended_prob side per event.
        ml_by_event = defaultdict(list)
        for c in ar.get("all_ml") or []:
            ml_by_event[c.get("event_id")].append(c)
        for event_id, cands in ml_by_event.items():
            if not event_id:
                continue
            pick = max(cands, key=lambda c: _mkt_num(c.get("blended_prob")))
            model_prob = _mkt_prob01(pick.get("blended_prob"))
            if model_prob is None:
                continue
            home_away = (pick.get("home_away") or "").upper()
            row = _common(event_id)
            row.update({
                "bet_type": "moneyline",
                "team": pick.get("team"),
                "opponent": pick.get("opponent"),
                "home_away": home_away,
                "side": "home" if home_away == "HOME" else "away",
                "matchup": _mkt_matchup(pick, events.get(event_id)),
                "book": pick.get("best_book"),
                "point": None,
                "model_prob": model_prob,
                "raw_prob": _mkt_prob01(pick.get("model_prob")),
                "price": _mkt_int(pick.get("best_price")),
                "is_value": _mkt_bool(pick.get("is_value")),
            })
            rows.append(row)

        # Spread: the higher-cover_rate side per event.
        sp_by_event = defaultdict(list)
        for c in ar.get("all_spreads") or []:
            sp_by_event[c.get("event_id")].append(c)
        for event_id, cands in sp_by_event.items():
            if not event_id:
                continue
            pick = max(cands, key=lambda c: _mkt_num(c.get("cover_rate")))
            model_prob = _mkt_prob01(pick.get("cover_rate"))
            point = _mkt_float(pick.get("spread"))
            if model_prob is None or point is None:
                continue
            home_away = (pick.get("home_away") or "").upper()
            row = _common(event_id)
            row.update({
                "bet_type": "spread",
                "team": pick.get("team"),
                "opponent": pick.get("opponent"),
                "home_away": home_away,
                "side": "home" if home_away == "HOME" else "away",
                "matchup": _mkt_matchup(pick, events.get(event_id)),
                "book": None,
                "point": point,
                "model_prob": model_prob,
                "raw_prob": _mkt_prob01(pick.get("model_cover_rate")),
                "price": _mkt_int(pick.get("price")),
                "is_value": _mkt_bool(pick.get("is_value")),
            })
            rows.append(row)

        # Total: over if over_hit_rate >= 50 else under.
        for c in ar.get("all_totals") or []:
            event_id = c.get("event_id")
            if not event_id:
                continue
            over_hit = c.get("over_hit_rate")
            point = _mkt_float(c.get("line"))
            if over_hit is None or point is None:
                continue
            over = float(over_hit) >= 50.0
            model_over = c.get("model_over_hit_rate")
            raw_pct = (model_over if over
                       else (None if model_over is None else 100.0 - model_over))
            row = _common(event_id)
            row.update({
                "bet_type": "total",
                "team": None,
                "opponent": None,
                "home_away": None,
                "side": "over" if over else "under",
                "matchup": _mkt_matchup(c, events.get(event_id)),
                "book": None,
                "point": point,
                "model_prob": _mkt_prob01(over_hit if over else 100.0 - over_hit),
                "raw_prob": _mkt_prob01(raw_pct),
                "price": _mkt_int(c.get("over_price") if over
                                  else c.get("under_price")),
                "is_value": _mkt_bool(c.get("is_over_value") if over
                                      else c.get("is_under_value")),
            })
            rows.append(row)

        for row in rows:
            _enrich_market_ids(row)
        return rows
    except Exception:
        return []


def _enrich_market_ids(row):
    """Best-effort: stamp SFBB canonical team codes onto a team-market forecast row
    (home/away + picked team/opponent) for id-based joins. MLB-only — the SFBB team
    map covers baseball and returns None elsewhere. Fail-open, never raises; mutates
    ``row`` in place."""
    if not (row.get("sport_key") or "").startswith("baseball"):
        return row
    try:
        import player_id_map
        tc = player_id_map.team_code_for_name
        row["home_code"] = tc(row.get("home_team")) if row.get("home_team") else None
        row["away_code"] = tc(row.get("away_team")) if row.get("away_team") else None
        row["team_code"] = tc(row.get("team")) if row.get("team") else None
        row["opponent_code"] = tc(row.get("opponent")) if row.get("opponent") else None
    except Exception:                       # pragma: no cover - never break logging
        pass
    return row


def log_market_prediction_rows(new_rows):
    """Upsert team-market forecast rows by (sport, event, bet_type) identity.

    The newest forecast supersedes a stale UNRESOLVED row (a re-analysis, even a
    side flip or line move); a row already resolved is never overwritten so its
    graded outcome survives. Best-effort: a missing SQL table (before the DDL is
    run) or any backend error is swallowed so analysis is never broken. Returns
    the number of rows added or changed."""
    if not new_rows:
        return 0
    try:
        def upsert(rows):
            incoming = {}
            for row in new_rows:
                incoming[market_prediction_identity(row)] = row  # last wins
            resolved_idents = {market_prediction_identity(r)
                               for r in rows if r.get("resolved")}
            # Keep other markets + any already-resolved rows; drop the stale
            # UNRESOLVED rows we're re-logging (replaced below).
            kept = [r for r in rows
                    if market_prediction_identity(r) not in incoming
                    or market_prediction_identity(r) in resolved_idents]
            changes = 0
            for ident, row in incoming.items():
                if ident in resolved_idents:
                    continue  # graded — never overwrite
                kept.append(row)
                changes += 1
            rows[:] = kept
            return changes
        return mutate_market_prediction_log(upsert)
    except Exception:
        return 0


def resolve_pending_market_outcomes(sport_key, max_to_resolve=MAX_RESOLVE_PER_LAUNCH):
    """Grade past-dated unresolved team-market forecasts against final scores.

    Mirrors resolve_pending_outcomes but team-shaped: uses the shared team graders
    (game_results.final_score / side_for_team / grade_team_bet), exactly as
    wagers._grade_wager. A still-live/unavailable game stays pending (retried once
    scores land). Returns the number newly resolved. Best-effort; never raises."""
    try:
        import game_results
    except Exception:
        return 0
    unresolved = {"sport_key": sport_key, "resolved": False}
    rows = _read_market_log(where=unresolved)
    if not rows:
        return 0

    today = datetime.now(timezone.utc).date().isoformat()
    resolved_updates = {}
    resolved_count = 0
    for r in rows:
        if resolved_count >= max_to_resolve:
            break
        if r.get("sport_key") != sport_key or r.get("resolved"):
            continue
        game_date = (r.get("game_date") or "")[:10]
        if not game_date or game_date >= today:
            continue  # game hasn't happened yet
        score = game_results.final_score(
            sport_key, game_date, r.get("home_team"), r.get("away_team"),
            r.get("commence_time"))
        if score is None:
            continue  # stay pending; retry once final scores land
        home_score, away_score = score
        bet_type = r.get("bet_type")
        side = r.get("side")
        if bet_type in ("moneyline", "spread"):
            # Grade by TEAM identity, not the stored side (final_score already
            # matched the game on these home/away names) — mirrors _grade_wager.
            resolved_side = game_results.side_for_team(
                r.get("team"), r.get("home_team"), r.get("away_team"))
            if resolved_side is not None:
                side = resolved_side
        status = game_results.grade_team_bet(
            bet_type, side, r.get("point"), home_score, away_score)
        if status is None:
            continue
        outcome = 1 if status == "won" else (0 if status == "lost" else None)
        resolved_updates[market_prediction_row_key(r)] = {
            "actual": f"{home_score:g}-{away_score:g}",
            "outcome": outcome,
            "resolved": True,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
        resolved_count += 1

    if not resolved_updates:
        return 0

    def apply_resolutions(current_rows):
        changed = 0
        for row in current_rows:
            update = resolved_updates.get(market_prediction_row_key(row))
            if update and not row.get("resolved"):
                row.update(update)
                changed += 1
        return changed

    try:
        mutate_market_prediction_log(apply_resolutions, where=unresolved)
    except Exception:
        return 0
    return resolved_count


def summarize_market_prediction_rows(rows, sport_key=None):
    """Summarize deduplicated team-market forecasts without making API calls.

    model_prob is the PICKED side's probability and outcome==1 means that pick
    won, so the Brier is (model_prob - outcome)^2 directly. ROI stakes the pick
    at its logged price (win: payout; loss: -1; push: 0)."""
    unique = {}
    for row in rows:
        if sport_key and row.get("sport_key") != sport_key:
            continue
        identity = market_prediction_identity(row)
        current = unique.get(identity)
        if current and current.get("resolved") and not row.get("resolved"):
            continue
        unique[identity] = row
    predictions = list(unique.values())

    def metrics(group):
        resolved = [r for r in group if r.get("resolved")]
        graded = [r for r in resolved if r.get("outcome") in (0, 1)]
        probabilities, outcomes, realized_returns = [], [], []
        hits = 0
        for r in graded:
            outcome = int(r["outcome"])
            if outcome == 1:
                hits += 1
            try:
                prob = float(r.get("model_prob"))
            except (TypeError, ValueError):
                prob = None
            if prob is not None and 0.0 <= prob <= 1.0:
                probabilities.append(prob)
                outcomes.append(outcome)
            try:
                price = int(r.get("price"))
            except (TypeError, ValueError):
                price = None
            if price:
                profit = (price / 100.0 if price > 0 else 100.0 / -price)
                realized_returns.append(profit if outcome == 1 else -1.0)
        # A resolved push returns the stake — include in priced ROI.
        for r in resolved:
            if r.get("outcome") is not None:
                continue
            try:
                price = int(r.get("price"))
            except (TypeError, ValueError):
                continue
            if price:
                realized_returns.append(0.0)
        brier = (sum((p - y) ** 2 for p, y in zip(probabilities, outcomes))
                 / len(outcomes) if outcomes else None)
        return {
            "total": len(group),
            "resolved": len(resolved),
            "pending": len(group) - len(resolved),
            "pushes": len(resolved) - len(graded),
            "graded": len(graded),
            "hit_rate": (hits / len(graded)) if graded else None,
            "brier": brier,
            "priced_resolved": len(realized_returns),
            "roi": (sum(realized_returns) / len(realized_returns)
                    if realized_returns else None),
        }

    summary = metrics(predictions)
    groups = defaultdict(list)
    for row in predictions:
        groups[(row.get("sport_key"), row.get("bet_type"))].append(row)
    summary["by_market"] = [
        {"sport_key": key[0], "bet_type": key[1], **metrics(group)}
        for key, group in sorted(
            groups.items(),
            key=lambda item: tuple(str(v or "") for v in item[0]))
    ]
    summary["last_prediction_ts"] = max(
        (row.get("ts") or "" for row in predictions), default="")
    return summary


def market_prediction_performance_summary(sport_key=None):
    """Return current team-market forward-log status and resolved performance."""
    return summarize_market_prediction_rows(_read_market_log(), sport_key=sport_key)


# ──────────────────────────────────────────────────────────────────────────────
# Outcome resolution against ESPN gamelogs
# ──────────────────────────────────────────────────────────────────────────────

# Sport → (espn_sport, espn_league)
SPORT_ESPN_MAP = {
    "basketball_nba":       ("basketball", "nba"),
    "baseball_mlb":         ("baseball",   "mlb"),
    "americanfootball_nfl": ("football",   "nfl"),
    "icehockey_nhl":        ("hockey",     "nhl"),
}


def _stat_label(prop_key, gamelog):
    """Resolve stat label for a prop, sniffed from a sample game."""
    try:
        from espn_client import PROP_STAT_MAP
    except Exception:
        return None
    for label in PROP_STAT_MAP.get(prop_key, []):
        if any(label in g for g in gamelog):
            return label
    return None


def _parse_dt(value):
    """Parse an ISO timestamp as tz-aware UTC (coercing naive -> UTC), or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _pick_candidate(candidates, commence):
    """Pick the gamelog index best matching a forecast's commence_time.

    `candidates` is a list of (full_datetime_str, idx) that share a calendar
    date. Disambiguates same-day games (doubleheaders) by nearest start time;
    falls back to the first candidate when commence is absent or the gamelog
    timestamps are unparseable / date-only (so single-game and legacy rows keep
    resolving exactly as before).
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][1]
    target = _parse_dt(commence)
    if target is not None:
        best_idx = best_delta = None
        for full, idx in candidates:
            dt = _parse_dt(full)
            if dt is None:
                continue
            delta = abs((dt - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta, best_idx = delta, idx
        if best_idx is not None:
            return best_idx
    return candidates[0][1]


def _today_et():
    """Current calendar date in US-Eastern (YYYY-MM-DD)."""
    now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return now.date().isoformat()


def _espn_row_final(row):
    """True unless an ESPN gamelog row could still be in progress.

    A row explicitly flagged ``completed`` is final. Otherwise a row dated today
    (US-Eastern) or later may be live, so treat it as not-final and keep the bet
    pending rather than grade a partial line; a past-dated game is always final.
    A row with no date preserves the legacy assume-final behavior. This is the
    backstop for the narrow case where the statsapi hard-ID path can't resolve a
    live MLB game and grading falls through to the ESPN gamelog."""
    if row.get("completed"):
        return True
    row_date = str(row.get("game_date") or "")[:10]
    if not row_date:
        return True
    return row_date < _today_et()


# prop_key -> (statsapi group, statsapi gameLog stat key). Unmapped props fall
# through to the ESPN gamelog path.
_MLB_STAT_SPEC = {
    "batter_hits": ("hitting", "hits"),
    "batter_home_runs": ("hitting", "homeRuns"),
    "batter_total_bases": ("hitting", "totalBases"),
    "batter_rbis": ("hitting", "rbi"),
    "batter_strikeouts": ("hitting", "strikeOuts"),
    "pitcher_strikeouts": ("pitching", "strikeOuts"),
    "pitcher_outs": ("pitching", "outs"),
    "pitcher_earned_runs": ("pitching", "earnedRuns"),
}


def _mlb_stat_spec(prop_key):
    return _MLB_STAT_SPEC.get(prop_key)


def _resolve_mlb_actual(sport_key, prop_key, player, game_date, commence):
    """Resolve a player's actual stat for the forecast game via the MLB statsapi
    hard-ID (gamePk) path, disambiguating doubleheaders by commence_time. Returns
    the stat value; None to fall back to the ESPN path; or the
    ``mlb_starters.GAME_NOT_FINAL`` sentinel when the bet's game is still live
    (caller keeps it pending). Never raises, so a statsapi outage degrades to
    ESPN and never blocks other sports."""
    if sport_key != "baseball_mlb":
        return None
    spec = _mlb_stat_spec(prop_key)
    if not spec:
        return None
    group, stat_key = spec
    try:
        season = int(str(game_date)[:4])
    except (TypeError, ValueError):
        return None
    try:
        import mlb_starters
        return mlb_starters.resolve_player_game_stat(
            player, commence, game_date, group, stat_key, season)
    except Exception:
        return None


def _is_stale_dnp(sport_key, prop_key, player, game_date, commence):
    """True when an unresolved MLB prop is a confirmed scratch/DNP whose game is
    at least STALE_DNP_HOURS old — permanently unresolvable, so it's safe to void
    (clears it from pending + stops re-attempting every tick). Gated on age so a
    same-day data lag isn't voided; gated on is_confirmed_dnp so a genuine data
    outage (missing log) keeps retrying. Never raises."""
    if sport_key != "baseball_mlb":
        return False
    spec = _mlb_stat_spec(prop_key)
    if not spec:
        return False
    try:
        import mlb_starters
        commence_dt = mlb_starters._parse_utc(commence)
        if commence_dt is None:
            return False
        age_hours = ((datetime.now(timezone.utc) - commence_dt).total_seconds()
                     / 3600.0)
        if age_hours < STALE_DNP_HOURS:
            return False
        season = int(str(game_date)[:4])
        return mlb_starters.is_confirmed_dnp(
            player, commence, game_date, spec[0], season)
    except Exception:
        return False


def _load_player_gamelog(espn_sport, espn_league, player):
    """(gamelog, {date: [(full_datetime, idx), ...]}) from ESPN, or (None, {}).

    Keeps EVERY game per date (a first-wins index would drop the 2nd game of a
    doubleheader). Never raises; a missing player/gamelog degrades to (None, {}).
    """
    try:
        from espn_cache import cached_athlete_id, cached_gamelog
        aid = cached_athlete_id(espn_sport, espn_league, player)
        if not aid:
            return None, {}
        # Outcome maintenance needs yesterday's result; the cache helper's
        # 30-day historical default is too stale for forward tracking.
        gamelog = cached_gamelog(espn_sport, espn_league, aid, ttl_hours=6)
        if not gamelog:
            return None, {}
    except Exception:
        return None, {}
    by_date = defaultdict(list)
    for i, g in enumerate(gamelog):
        full = g.get("game_date") or ""
        d = full[:10]
        if d:
            by_date[d].append((full, i))
    return gamelog, by_date


def resolve_one_prop(sport_key, player, prop_key, line, game_date, commence,
                     game_pk=None, mlb_player_id=None, _use_warehouse=True):
    """Resolve one player's actual stat for a single forecast game.

    P4 unified path: when the caller has the P3-stamped ``(game_pk, mlb_player_id)``,
    the actual is resolved FROM THE WAREHOUSE via mlb_warehouse.resolve_actual —
    which refreshes a still-correctable game from a fresh boxscore (so the DB is the
    single source of truth) and reads a settled game frozen (0 network). A miss
    (game not in the warehouse, pre-P3 row, an unsupported prop, or a player DNP)
    falls through to the live per-player path below. ``_use_warehouse=False`` skips
    it (the sweep passes this for its explicit live fallback, having already tried
    the warehouse).

    MLB rows otherwise resolve via the statsapi hard-ID (gamePk) path (disambiguating
    doubleheaders and grading pitcher props ESPN cannot), falling back to cached
    ESPN gamelogs with a ±1-day tolerance for UTC/local date slippage. Returns
    the actual stat as a float, or None when it can't be resolved. `line` is part
    of the forecast context callers pass but is not used to derive the value.
    Never raises. Shared by the prediction-log resolver and the wagers resolver.
    """
    if _use_warehouse and sport_key == "baseball_mlb" and game_pk and mlb_player_id:
        try:
            import mlb_warehouse
            v = mlb_warehouse.resolve_actual(mlb_player_id, game_pk, prop_key)
            if v is not None:
                return v
        except Exception:                   # pragma: no cover - never break grading
            pass
    pair = SPORT_ESPN_MAP.get(sport_key)
    if not pair:
        return None
    espn_sport, espn_league = pair
    game_date = str(game_date)[:10]
    # Hard-ID path first (MLB statsapi gamePk); None -> ESPN fallback.
    actual = _resolve_mlb_actual(sport_key, prop_key, player, game_date, commence)
    # A live game located by statsapi returns the GAME_NOT_FINAL sentinel: keep
    # the bet pending and DO NOT fall through to the un-gated ESPN partial line.
    try:
        import mlb_starters
        if actual is mlb_starters.GAME_NOT_FINAL:
            return None
    except Exception:
        pass
    if actual is None:
        # P6 grading cutover — MLB grades from the WAREHOUSE + statsapi ONLY:
        # grading already tried the warehouse (resolve_actual) then the statsapi
        # hard-ID path (_resolve_mlb_actual, which resolves by NAME — no game_pk
        # needed). Both missed, so DO NOT fall to the ESPN gamelog for baseball —
        # stay pending (return None). _ENFORCE_IDENTITY keeps new unpinnable rows
        # from ever being logged, so the only rows this leaves pending are legacy
        # id-less prospects ESPN couldn't reliably grade anyway (aged out
        # separately). NBA/NFL/NHL keep the ESPN gamelog path below unchanged.
        if sport_key == "baseball_mlb":
            return None
        gamelog, by_date = _load_player_gamelog(espn_sport, espn_league, player)
        if not gamelog:
            return None
        # Never grade a pitcher prop off a batter's gamelog (or vice-versa): the
        # "K"/"SO" strikeout labels collide across MLB roles, so _stat_label
        # would bind the wrong role's log (mirrors backtest._role_matches_gamelog
        # and the book-line calibration guard). Non-MLB props (role None) always
        # pass. Lazy import — backtest imports recalibration, so a top-level
        # import would be circular. Fail-open: if the guard can't load, grade as
        # before (defensive hardening, not a load-bearing dependency).
        try:
            from backtest import _role_matches_gamelog
            if not _role_matches_gamelog(prop_key, gamelog):
                return None
        except Exception:
            pass
        stat_label = _stat_label(prop_key, gamelog)
        if not stat_label:
            return None
        idx = _pick_candidate(by_date.get(game_date), commence)
        if idx is None:
            # ±1 day fallback for timezone slippage: the bet's real game is
            # labeled game_date but ESPN filed it a calendar day off (UTC/local).
            # Genuine slippage keeps the SAME start time, so require the matched
            # row within ~20h of commence — otherwise this is a DIFFERENT game on
            # the adjacent day (e.g. a postponed game's bet matching the prior
            # night's game), which must stay pending, never grade.
            target = _parse_dt(commence)
            for delta in (-1, 1):
                try:
                    alt = (datetime.fromisoformat(game_date)
                           + timedelta(days=delta)).date().isoformat()
                except Exception:
                    continue
                cand = _pick_candidate(by_date.get(alt), commence)
                if cand is None:
                    continue
                if target is not None:
                    row_dt = _parse_dt((gamelog[cand] or {}).get("game_date"))
                    if row_dt is not None and abs(
                            (row_dt - target).total_seconds()) > 20 * 3600:
                        continue
                idx = cand
                break
        if idx is None:
            return None
        # Completion gate: a same-day-or-later ESPN row may be a live game with a
        # partial line. Grade only confirmed-final games; stay pending otherwise.
        if not _espn_row_final(gamelog[idx]):
            return None
        actual = gamelog[idx].get(stat_label)
        # (The MLB-only pitcher_outs IP->outs conversion lived here; MLB no longer
        # reaches this ESPN branch — see the baseball_mlb return above — so it's gone.
        # This branch is now NBA/NFL/NHL-only, whose stats need no such conversion.)
    if actual is None:
        return None
    try:
        return float(actual)
    except (TypeError, ValueError):
        return None


def resolve_pending_outcomes(sport_key, max_to_resolve=MAX_RESOLVE_PER_LAUNCH):
    """
    Walk the log, find unresolved entries for this sport whose game_date is in
    the past, and fill in actual + outcome. MLB rows resolve via the statsapi
    hard-ID (gamePk) path first (disambiguating doubleheaders and grading pitcher
    props ESPN cannot), falling back to cached ESPN gamelogs. Caps per-launch
    resolution by `max_to_resolve`. Returns the number of newly resolved entries.
    """
    pair = SPORT_ESPN_MAP.get(sport_key)
    if not pair:
        return 0
    # Pull only this sport's UNRESOLVED rows out of the DB (settled rows are the
    # bulk and are discarded anyway); the game_date<today refinement stays in
    # Python below. The local path ignores the filter and self-filters.
    unresolved = {"sport_key": sport_key, "resolved": False}
    rows = _read_log(where=unresolved)
    if not rows:
        return 0

    today = datetime.now(timezone.utc).date().isoformat()
    # group unresolved rows for this sport by player
    by_player = defaultdict(list)
    for r in rows:
        if r.get("sport_key") != sport_key:
            continue
        if r.get("resolved"):
            continue
        if (r.get("game_date") or "") >= today:
            continue  # game hasn't happened yet
        by_player[r["player"]].append(r)

    if not by_player:
        return 0

    resolved_count = 0     # genuine resolutions (a real outcome) — the return value
    void_count = 0         # stale scratch/DNP rows retired (no outcome)
    network_attempts = 0   # live fetches only — the P4 cap counts these, NOT the
                           # warehouse fast path (a free DB read of already-ingested
                           # game facts), so a fully-ingested slate resolves unbounded
    resolved_updates = {}
    # The cap bounds statsapi/ESPN work per pass (a DNP backlog must not spin
    # resolve_one_prop/is_confirmed_dnp unbounded); warehouse hits are exempt.
    for player, p_rows in by_player.items():
        for r in p_rows:
            commence = r.get("commence_time")
            gpk = r.get("game_pk")
            pid = r.get("player_mlb_id")
            # Warehouse path: resolve from the DB (resolve_actual refreshes a
            # recent/active game from a fresh boxscore, reads a settled game frozen
            # with ZERO network). Not counted against the cap — historical reads are
            # free and active refreshes are boxscore-cache-bounded + few (~2 days of
            # games). The cap bounds only the LIVE per-player fallback below.
            actual = None
            if sport_key == "baseball_mlb" and gpk and pid:
                try:
                    import mlb_warehouse
                    actual = mlb_warehouse.resolve_actual(pid, gpk, r["prop_key"])
                except Exception:
                    actual = None
            if actual is None:
                # Live per-player fallback (game not in the warehouse, unsupported
                # prop, or DNP), bounded by the network cap. Skip (leave pending)
                # once spent — but keep scanning, so warehouse rows still get done.
                if network_attempts >= max_to_resolve:
                    continue
                network_attempts += 1
                actual = resolve_one_prop(
                    sport_key, player, r["prop_key"], r.get("line"),
                    r["game_date"], commence, game_pk=gpk, mlb_player_id=pid,
                    _use_warehouse=False)
                if actual is None:
                    # A confirmed scratch/DNP whose game is well past is permanently
                    # unresolvable → void it (resolved, no outcome) so it leaves
                    # pending and stops re-attempting. A still-live game (resolver
                    # returns the sentinel, not None) or a data outage is NOT voided.
                    if _is_stale_dnp(sport_key, r["prop_key"], player,
                                     r["game_date"], commence):
                        resolved_updates[prediction_row_key(r)] = {
                            "actual": None,
                            "outcome": None,
                            "resolved": True,
                            "resolved_at": datetime.now(timezone.utc).isoformat(),
                        }
                        void_count += 1
                    continue
            line = float(r["line"])
            if actual == line:
                outcome = None  # push
            else:
                outcome = 1 if actual > line else 0
            resolved_updates[prediction_row_key(r)] = {
                "actual": actual,
                "outcome": outcome,
                "resolved": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            resolved_count += 1

    if not resolved_updates:
        return 0

    def apply_resolutions(current_rows):
        changed = 0
        for row in current_rows:
            update = resolved_updates.get(prediction_row_key(row))
            if update and not row.get("resolved"):
                row.update(update)
                changed += 1
        return changed

    try:
        mutate_prediction_log(apply_resolutions, where=unresolved)
    except Exception:
        return 0
    if void_count:
        print(f"  [resolve] voided {void_count} stale unresolvable "
              f"(scratch/DNP) prediction(s) for {sport_key}")
    # Return only GENUINE resolutions — voids carry no label and must not trip
    # the Platt refit gate (maintain_sport keys on this count).
    return resolved_count


# ──────────────────────────────────────────────────────────────────────────────
# Platt scaling
# ──────────────────────────────────────────────────────────────────────────────

_EPS = 1e-6


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p):
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def fit_platt(raw_probs, outcomes, max_iter=100, tol=1e-7):
    """
    Fit Platt sigmoid: p_cal = sigmoid(a * logit(p_raw) + b).
    Returns (a, b) or None if not fittable.

    Uses Newton-Raphson on cross-entropy loss with mild L2 regularization
    on (a-1, b) to keep parameters from blowing up on small samples.
    """
    pairs = [(rp, o) for rp, o in zip(raw_probs, outcomes)
             if rp is not None and o is not None and o in (0, 1)]
    if len(pairs) < MIN_FIT_SAMPLES:
        return None

    xs = [_logit(rp) for rp, _ in pairs]
    ys = [o for _, o in pairs]
    n = len(xs)

    # Smoothed targets per Platt's recipe (avoids 0/1 boundary issues)
    n_pos = sum(ys)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    targets = [hi if y == 1 else lo for y in ys]

    a, b = 1.0, 0.0
    lam = 1.0 / n  # tiny L2 (~ ridge) toward (1, 0)

    for _ in range(max_iter):
        # gradient + hessian
        g_a = g_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for x, t in zip(xs, targets):
            z = a * x + b
            p = _sigmoid(z)
            err = p - t
            g_a += err * x
            g_b += err
            w = p * (1.0 - p)
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w
        # regularization toward (1, 0)
        g_a += lam * (a - 1.0)
        g_b += lam * b
        h_aa += lam
        h_bb += lam

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        # solve 2x2 system H * delta = g
        d_a = (h_bb * g_a - h_ab * g_b) / det
        d_b = (-h_ab * g_a + h_aa * g_b) / det

        a_new = a - d_a
        b_new = b - d_b

        if abs(d_a) < tol and abs(d_b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    # Sanity: clamp wild fits
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if a <= 0.2:
        # a≈0 means "raw prob has ~no signal vs outcomes" — applying it would
        # squash every prediction to a constant ~base-rate. Treat as no fit
        # and let the raw probability pass through unchanged. (User keeps
        # whatever edge the underlying model had; doesn't get artificially
        # zeroed by Platt.)
        return None
    if abs(a) > 10 or abs(b) > 10:
        return None
    return (a, b)


def apply_platt(raw_prob, a, b):
    """Apply Platt sigmoid; returns recalibrated probability."""
    if raw_prob is None:
        return None
    if a is None or b is None:
        return raw_prob
    return _sigmoid(a * _logit(raw_prob) + b)


def _probability_scores(probabilities, outcomes):
    """Return (Brier, log loss) for binary probability forecasts."""
    if not probabilities:
        return None, None
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)
    log_loss = -sum(
        y * math.log(min(max(p, _EPS), 1.0 - _EPS))
        + (1 - y) * math.log(min(max(1.0 - p, _EPS), 1.0 - _EPS))
        for p, y in zip(probabilities, outcomes)
    ) / len(outcomes)
    return brier, log_loss


def fit_platt_chronological(records, incumbent=None):
    """
    Validate Platt scaling on two expanding-window chronological folds before
    fitting the final parameters on all observations. `records` contains
    (date_or_timestamp, raw_probability, outcome).

    Returns a parameter/metric dict, or None when the calibrated probabilities
    do not improve both Brier score and log loss in every untouched later fold.
    Rows sharing a date always stay in the same side of a boundary.

    `incumbent` is an optional (a, b) seed map for this key (the committed
    book-line prior). When given, this is a *champion gate*: the loop fit must
    additionally clear MIN_OBS_FOR_OVERRIDE observations and beat the seed map
    (not just raw) on both metrics in every fold, or it does not override the
    seed. `incumbent=None` (offline seeding, loop-only props) keeps the original
    beat-raw behavior unchanged.
    """
    rows = sorted(
        (str(date), float(raw), int(outcome))
        for date, raw, outcome in records
        if raw is not None and outcome in (0, 1)
    )
    if len(rows) < MIN_FIT_SAMPLES + 2 * MIN_VALIDATION_SAMPLES:
        return None
    if incumbent is not None and len(rows) < MIN_OBS_FOR_OVERRIDE:
        return None

    cut1 = rows[int(len(rows) * 0.6)][0]
    cut2 = rows[int(len(rows) * 0.8)][0]
    if cut1 == cut2:
        return None
    folds = [
        ([row for row in rows if row[0] < cut1],
         [row for row in rows if cut1 <= row[0] < cut2]),
        ([row for row in rows if row[0] < cut2],
         [row for row in rows if row[0] >= cut2]),
    ]
    validation_folds = []
    score_totals = [0.0, 0.0, 0.0, 0.0]
    validation_n = 0
    for train, holdout in folds:
        if (len(train) < MIN_FIT_SAMPLES
                or len(holdout) < MIN_VALIDATION_SAMPLES):
            return None
        fit = fit_platt([row[1] for row in train], [row[2] for row in train])
        if fit is None:
            return None
        a, b = fit
        holdout_raw = [row[1] for row in holdout]
        holdout_y = [row[2] for row in holdout]
        holdout_cal = [apply_platt(raw, a, b) for raw in holdout_raw]
        raw_brier, raw_log_loss = _probability_scores(holdout_raw, holdout_y)
        cal_brier, cal_log_loss = _probability_scores(holdout_cal, holdout_y)
        if cal_brier >= raw_brier or cal_log_loss >= raw_log_loss:
            return None
        if incumbent is not None:
            # Champion gate: the loop fit must beat the *seed* map on this
            # untouched later window, not merely raw, to earn an override.
            a_s, b_s = incumbent
            seed_cal = [apply_platt(raw, a_s, b_s) for raw in holdout_raw]
            seed_brier, seed_log_loss = _probability_scores(seed_cal, holdout_y)
            if cal_brier >= seed_brier or cal_log_loss >= seed_log_loss:
                return None
        n_holdout = len(holdout)
        for i, score in enumerate((raw_brier, cal_brier,
                                   raw_log_loss, cal_log_loss)):
            score_totals[i] += score * n_holdout
        validation_n += n_holdout
        validation_folds.append({
            "holdout_start": holdout[0][0],
            "n_validation": n_holdout,
            "raw_brier": raw_brier,
            "calibrated_brier": cal_brier,
            "raw_log_loss": raw_log_loss,
            "calibrated_log_loss": cal_log_loss,
        })

    raw_brier, cal_brier, raw_log_loss, cal_log_loss = (
        total / validation_n for total in score_totals)

    final_fit = fit_platt([row[1] for row in rows], [row[2] for row in rows])
    if final_fit is None:
        return None
    final_a, final_b = final_fit
    # The deployed (a, b) are refit on *all* observations for the strongest
    # estimate. The holdout_* metrics are the cross-validated score of the
    # fitting *procedure* over two expanding folds — not of these exact deployed
    # parameters, which by construction have seen every row and so cannot be
    # scored on unseen data. Both folds had to beat the raw probabilities on
    # their untouched later window to reach here, so the procedure is validated
    # even though the final parameters themselves are not directly held out.
    return {
        "a": final_a,
        "b": final_b,
        "n_fit": len(rows),
        "n_validation": validation_n,
        "n_validation_folds": len(validation_folds),
        "holdout_start": validation_folds[0]["holdout_start"],
        "holdout_raw_brier": raw_brier,
        "holdout_calibrated_brier": cal_brier,
        "holdout_raw_log_loss": raw_log_loss,
        "holdout_calibrated_log_loss": cal_log_loss,
        "holdout_metric_scope": "fold_cross_validation",
        "deploy_fit_scope": "all_observations",
        "validation_folds": validation_folds,
        "validated": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence + load
# ──────────────────────────────────────────────────────────────────────────────

def save_recalibration(sport_key, per_prop_params, meta=None, to_blob=True):
    """Persist a Platt fit. Offline seeding (`to_blob=False`) writes the local
    git-committed file (the seed/prior that ships to Cloud). A runtime SQL refit
    (`to_blob=True` and `_sql()`) persists to Azure SQL *only* and deliberately
    leaves the local seed untouched, so it stays a pristine prior for the per-key
    merge/champion-gate in `_load_recal_cached`/`refit_sport`. Pure-local dev
    (no SQL) still writes the local file."""
    _ensure_dirs()
    blob = {
        "sport_key": sport_key,
        "fit_timestamp": datetime.now(timezone.utc).isoformat(),
        "props": per_prop_params,
    }
    if meta:
        blob["meta"] = meta
    # A runtime refit (to_blob=True) is a durable write: refuse to let it land on
    # ephemeral local disk when a SQL deployment is signalled but SQL is off. The
    # offline seed path (to_blob=False) is *meant* to write the committed git file,
    # so it stays unguarded.
    if to_blob:
        _ensure_durable("save recalibration params")
    # Local write (atomic swap). A runtime SQL refit skips this so it cannot
    # clobber the committed seed (which the merge/gate read back as the prior).
    if not (to_blob and _sql()):
        tmp = recalibration_path(sport_key) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
        os.replace(tmp, recalibration_path(sport_key))
    # SQL is the durable overlay when configured. Best-effort: a transient SQL
    # failure leaves the local seed intact and the next refit retries; swallowing
    # keeps the free loop (maybe_auto_refit) crash-proof.
    if to_blob and _sql():
        try:
            _db.save_recal(sport_key, blob)
        except Exception as e:
            _ops.ops_event("database_failure", op="save_recal",
                           sport=sport_key, error=type(e).__name__)
    _LOAD_CACHE.pop(sport_key, None)


_LOAD_CACHE = {}  # sport_key -> {fetched_at, etag, fit_ts, props}


def _parse_recal_blob(blob):
    """(fit_ts_epoch_or_None, validated_props) from a raw recalibration blob.
    Fully defensive: any malformed shape yields (None, {}) rather than raising
    into the (unwrapped) free-loop load path. Legacy/unvalidated fits are kept on
    disk for provenance but never applied."""
    if not isinstance(blob, dict):
        return None, {}
    fit_ts = None
    ts_raw = blob.get("fit_timestamp")
    if ts_raw:
        try:
            fit_ts = datetime.fromisoformat(
                str(ts_raw).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            fit_ts = None
    props = {}
    raw_props = blob.get("props")
    if isinstance(raw_props, dict):
        for prop_key, cfg in raw_props.items():
            if isinstance(cfg, dict) and cfg.get("validated") is True:
                props[prop_key] = cfg
    return fit_ts, props


def _read_local_recal(sport_key):
    """(fit_ts_epoch_or_None, validated_props) from the local git-committed
    file, or (None, {}). This is the bootstrap the SQL overlay merges onto."""
    path = recalibration_path(sport_key)
    if not os.path.exists(path):
        return None, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return None, {}
    return _parse_recal_blob(blob)


def _blend_recal(seed_cfg, loop_cfg):
    """Precision-weighted shrinkage of a self-learned (loop) fit toward the
    committed seed prior. Platt is affine in logit space, so averaging (a, b) is
    exactly averaging the calibrated logits — a principled blend, not a hack. The
    loop's weight grows with its evidence:
        w = n_loop / (n_loop + RECAL_SEED_TRUST * n_seed)
    Returns a cfg identical to `loop_cfg` (keeps validated + holdout provenance)
    but with the blended a/b and a blend audit trail. Degrades to the loop fit if
    the weights are unusable."""
    n_seed = seed_cfg.get("n_fit") or 0
    n_loop = loop_cfg.get("n_fit") or 0
    if n_seed <= 0:
        n_seed = MIN_FIT_SAMPLES  # anchor a count-less seed rather than ignore it
    denom = n_loop + RECAL_SEED_TRUST * n_seed
    if denom <= 0:
        return loop_cfg
    w = n_loop / denom
    a = round(w * loop_cfg["a"] + (1.0 - w) * seed_cfg["a"], 5)
    b = round(w * loop_cfg["b"] + (1.0 - w) * seed_cfg["b"], 5)
    return {
        **loop_cfg,
        "a": a,
        "b": b,
        "blend_weight": round(w, 4),
        "blend_seed": {
            "a": seed_cfg["a"], "b": seed_cfg["b"], "n_fit": seed_cfg.get("n_fit")},
    }


def _load_recal_cached(sport_key):
    """
    Return (fit_ts_epoch_or_None, validated_props).

    Reads the SQL overlay when configured — with a short TTL — and merges it onto
    the local git-committed seed so a fresh/empty overlay never hides the shipped
    seed. Never raises; degrades to the last good cache, then the local file, then
    {}. Shared by load_recalibration (applied on every analyze) and the
    maintain_sport refit gate, so a maintenance tick plus the following analyze
    issue a single SQL read.
    """
    if _sql():
        # SQL overlay of the git-committed baseline:
        # a sport with no (validated) SQL fit falls back to the shipped seed.
        now = time.time()
        cached = _LOAD_CACHE.get(sport_key)
        if cached and (now - cached["fetched_at"]) < RECAL_LOAD_TTL_SECONDS:
            return cached["fit_ts"], cached["props"]
        try:
            cfg = _db.load_recal(sport_key)
        except Exception:
            if cached:
                cached["fetched_at"] = now
                return cached["fit_ts"], cached["props"]
            return _read_local_recal(sport_key)
        sql_fit_ts, sql_props = _parse_recal_blob(cfg) if cfg else (None, {})
        seed_fit_ts, seed_props = _read_local_recal(sport_key)
        if sql_props:
            # Per-key overlay: the seed is the prior for every key it holds, and a
            # self-learned SQL fit for a key blends toward it (shrinkage). Seed-only
            # keys survive untouched — one prop's first SQL fit no longer erases the
            # rest of the committed seed (the old all-or-nothing fallback bug).
            merged = dict(seed_props)
            for key, loop_cfg in sql_props.items():
                seed_cfg = seed_props.get(key)
                merged[key] = _blend_recal(seed_cfg, loop_cfg) if seed_cfg else loop_cfg
            # fit_ts keys on the PRE-merge sql_props: `merged` is always non-empty
            # once a seed exists, so keying on it would mask "no SQL fit yet" from
            # the maintain_sport gate (last_fit_ts=None) and refit every tick.
            fit_ts, props = sql_fit_ts, merged
        else:
            fit_ts, props = seed_fit_ts, seed_props
        _LOAD_CACHE[sport_key] = {
            "fetched_at": now, "etag": None, "fit_ts": fit_ts, "props": props}
        return fit_ts, props
    # Local-only: mtime-keyed cache (unchanged semantics).
    path = recalibration_path(sport_key)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _LOAD_CACHE.pop(sport_key, None)
        return None, {}
    cached = _LOAD_CACHE.get(sport_key)
    if cached and cached.get("fetched_at") == mtime:
        return cached["fit_ts"], cached["props"]
    fit_ts, props = _read_local_recal(sport_key)
    _LOAD_CACHE[sport_key] = {
        "fetched_at": mtime, "etag": None, "fit_ts": fit_ts, "props": props}
    return fit_ts, props


def load_recalibration(sport_key):
    """Return {prop_key: {"a": ..., "b": ..., "n_fit": ...}} of validated fits,
    or {}. SQL-overlaid (merged onto the local baseline) when configured."""
    _, props = _load_recal_cached(sport_key)
    return props


# ──────────────────────────────────────────────────────────────────────────────
# Refit (resolve + fit + save) and gated auto-refit
# ──────────────────────────────────────────────────────────────────────────────

def refit_sport(sport_key, resolve_first=True, max_resolve=MAX_RESOLVE_PER_LAUNCH,
                newly_resolved=None):
    """Resolve pending outcomes, then refit Platt for every prop with enough
    resolved entries. Returns dict {fit_key: (a, b, n_fit)} where fit_key is the
    bare prop_key, or "<prop>@<bucket>" for props with a line_methods cfg."""
    if resolve_first:
        newly_resolved = resolve_pending_outcomes(sport_key, max_to_resolve=max_resolve)
    elif newly_resolved is None:
        newly_resolved = 0

    # Per-line-bucket recal: fit each line_methods bucket separately so a prop
    # whose line regimes have different base rates (e.g. batter_hits 0.5 vs 1.5)
    # is not miscalibrated by a single pooled map. Lazily imported to avoid the
    # props <-> recalibration import cycle (same pattern as the seed path).
    from props import _composite_recal_key
    from calibration_loader import load_calibration
    cal = load_calibration(sport_key) or {}

    # The committed seed is the prior for the champion gate: a loop fit overrides
    # a seeded key only if it beats that seed out-of-sample (see refit loop). In
    # prod the local file stays pristine (save_recalibration skips it for SQL
    # refits), so this is always the book-line seed, never the loop's own output.
    _, seed_props = _read_local_recal(sport_key)

    def _line_methods_for(prop_key):
        return (cal.get(prop_key) or {}).get("line_methods")

    rows = _read_log()
    # Repeated app launches can log the same published player/game/line more
    # than once. Keep the latest version of each forecast so frequently viewed
    # games do not receive accidental extra weight in calibration.
    unique_rows = {}
    for r in rows:
        if r.get("sport_key") != sport_key:
            continue
        if not r.get("resolved"):
            continue
        o = r.get("outcome")
        if o not in (0, 1):
            continue
        identity = prediction_identity(r)
        unique_rows[identity] = r

    by_prop_records = defaultdict(list)
    for r in unique_rows.values():
        o = r["outcome"]
        order_key = r.get("game_date") or r.get("ts") or ""
        rec_key = _composite_recal_key(
            r["prop_key"], r.get("line"), _line_methods_for(r["prop_key"]))
        if rec_key is None:
            continue
        by_prop_records[rec_key].append(
            (order_key, float(r["raw_prob"]), int(o)))

    fits = {}
    per_prop_params = {}
    for fit_key, records in by_prop_records.items():
        seed_cfg = seed_props.get(fit_key)
        incumbent = (seed_cfg["a"], seed_cfg["b"]) if seed_cfg else None
        result = fit_platt_chronological(records, incumbent=incumbent)
        if result is None:
            continue
        a, b = result["a"], result["b"]
        fits[fit_key] = (a, b, result["n_fit"])
        per_prop_params[fit_key] = {
            "a": round(a, 5),
            "b": round(b, 5),
            "n_fit": result["n_fit"],
            "n_validation": result["n_validation"],
            "n_validation_folds": result["n_validation_folds"],
            "holdout_start": result["holdout_start"],
            "holdout_raw_brier": round(result["holdout_raw_brier"], 6),
            "holdout_calibrated_brier": round(
                result["holdout_calibrated_brier"], 6),
            "holdout_raw_log_loss": round(result["holdout_raw_log_loss"], 6),
            "holdout_calibrated_log_loss": round(
                result["holdout_calibrated_log_loss"], 6),
            "holdout_metric_scope": result.get(
                "holdout_metric_scope", "fold_cross_validation"),
            "deploy_fit_scope": result.get(
                "deploy_fit_scope", "all_observations"),
            "validation_folds": result["validation_folds"],
            "validated": True,
        }

    if per_prop_params:
        save_recalibration(sport_key, per_prop_params,
                           meta={"newly_resolved_this_run": newly_resolved})
        _LOAD_CACHE.pop(sport_key, None)
    return fits


def _count_resolved_since(sport_key, since_ts):
    """How many resolved entries exist (used to gate refit)."""
    rows = _read_log()
    cutoff = since_ts or 0
    n = 0
    for r in rows:
        if r.get("sport_key") != sport_key:
            continue
        if not r.get("resolved"):
            continue
        try:
            timestamp = r.get("resolved_at") or r.get("ts") or ""
            ts = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = 0
        if ts > cutoff:
            n += 1
    return n


_last_statcast_derived_build = 0.0     # module gate for the heavier daily rebuild

# Durable completeness watermark for the raw statcast_pitch corpus: the last date
# through which _statcast_maintenance has confirmed [.., date] fully ingested. Lets
# the hourly loop SELF-HEAL a multi-day idle gap (the trailing recent-window alone
# would skip the middle days) instead of needing an offline `savant_history --ensure`.
# Stored in the shared app_settings KV store (alongside the Kelly knobs).
_APP_SETTINGS_FILE = "app_settings.jsonl"
_STATCAST_WATERMARK_KEY = "statcast_last_ensured"


def _read_statcast_watermark():
    """The date (datetime.date) through which statcast_pitch is known-complete, or
    None. Fail-open (None) on any error so maintenance always still runs."""
    try:
        rows, _ = _read_ndjson_blob(_APP_SETTINGS_FILE, use_cache=False)
    except Exception:
        return None
    for r in (rows or []):
        if r.get("setting_key") == _STATCAST_WATERMARK_KEY:
            try:
                return datetime.fromisoformat(
                    str(r.get("setting_value"))[:10]).date()
            except (TypeError, ValueError):
                return None
    return None


def _write_statcast_watermark(day):
    """Upsert the completeness watermark to ``day`` (a date or ISO string), preserving
    every other setting. Best-effort; never raises."""
    val = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]

    def _upsert(rows):
        ts = datetime.now(timezone.utc).isoformat()
        for r in rows:
            if r.get("setting_key") == _STATCAST_WATERMARK_KEY:
                if r.get("setting_value") == val:
                    return 0
                r["setting_value"] = val
                r["updated_at"] = ts
                return 1
        rows.append({"setting_key": _STATCAST_WATERMARK_KEY,
                     "setting_value": val, "updated_at": ts})
        return 1

    try:
        return mutate_ndjson_log(_APP_SETTINGS_FILE, _upsert) or 0
    except Exception:
        return 0


def _statcast_maintenance(recent_days=4, ensure_cap=12, derived_interval_h=20):
    """Bounded, CRON-FREE statcast freshness for the app's hourly loop (owner has no
    scheduler). Two steps, both MLB Savant-network + fail-open so neither blocks
    maintenance:

      (1) EVERY call: ensure the recent raw statcast_pitch days are ingested (cheap +
          idempotent via the day manifest — only genuinely-missing recent days are
          fetched, capped). This keeps the raw corpus current so pitcher_asof
          .get_or_fill computes FRESH as-of pitcher rows on demand (pitcher_asof
          needs no rebuild — it self-fills lazily on read). SELF-HEALING: a durable
          watermark (statcast_last_ensured) widens the lookback to cover an idle gap
          since the last confirmed-complete run, so a burst-usage owner who skips N
          days still gets the middle days filled with no offline `--ensure`. A gap
          bigger than ensure_cap drains newest-first over successive hourly calls (the
          watermark advances only once the whole range is confirmed complete), floored
          at the current season start (a deeper hole is a one-time offline prime).
      (2) DAILY-gated: rebuild the derived per-batter statcast_asof snapshot the LIVE
          prop path reads (get_batter_xba) — heavier (season aggregate), so at most
          every derived_interval_h hours per process.

    Historical seasons are a one-time offline prime (savant_history --ensure +
    statcast_asof --build + pitcher_asof --build); this only keeps the CURRENT season
    live-fresh."""
    try:
        import savant_history as sh
    except Exception:
        return
    if not sh.enabled():
        return
    import datetime as _dt
    today = _dt.date.today()
    base_lo = today - _dt.timedelta(days=recent_days)
    lo = base_lo
    wm = _read_statcast_watermark()
    if wm is not None and wm < base_lo:
        # Idle since `wm`: the trailing window alone would skip [wm+1, base_lo). Widen
        # back to heal the gap, floored at this season's start and never NARROWER than
        # the trailing window (the min guard covers the Jan year-boundary case).
        season_start = _dt.date(today.year, 1, 1)
        lo = min(base_lo, max(wm + _dt.timedelta(days=1), season_start))
        gap = (base_lo - lo).days
        if gap > 0:
            print(f"  [statcast] self-heal: idle since {wm}; ensuring {lo}..{today} "
                  f"({gap}d beyond the {recent_days}d window, cap {ensure_cap}/call)")
    complete = False
    try:
        result = sh.ensure_days(lo.isoformat(), today.isoformat(),
                                cap=ensure_cap, verbose=False)
        try:
            n_fetched, n_missing = result
        except (TypeError, ValueError):
            n_fetched = n_missing = 0        # non-tuple return (e.g. a test mock)
        complete = n_fetched >= n_missing    # nothing left behind by the cap/failures
    except Exception:                        # network / vendor hiccup — never block
        complete = False
    if complete:
        # [lo, today] is fully ingested (and everything older was complete at the prior
        # watermark) -> advance. On a partially-drained big gap, leave it so the next
        # call retries the still-missing older days.
        _write_statcast_watermark(today)
    global _last_statcast_derived_build
    now = time.time()
    if now - _last_statcast_derived_build >= derived_interval_h * 3600:
        _last_statcast_derived_build = now
        try:
            import statcast_asof
            statcast_asof.build(today.year, fetch=False, verbose=False)
        except Exception:
            pass


def maintain_sport(sport_key, max_resolve=MAX_RESOLVE_PER_LAUNCH):
    """Resolve pending rows and refit only when the existing gates allow it.

    ``max_resolve`` caps successful resolutions this pass. The live-app hot path
    keeps the default (80) to stay responsive + ESPN-polite; an offline drain
    (forward_tracker --resolve --max-resolve N) can pass a high value to clear a
    backlog in one run."""
    # P4: keep the StatsAPI warehouse current BEFORE resolving — pull recent finals'
    # facts + flip their statuses to Final (so grading takes the warehouse path) and
    # pre-load upcoming schedule (so new predictions get a game_pk). MLB-only,
    # fail-open: a warehouse hiccup must never block resolution/refit. Rate-bounded
    # by the hourly maybe_auto_refit gate + idempotent/cached ingestion.
    if sport_key == "baseball_mlb":
        try:
            import mlb_warehouse
            mlb_warehouse.ingest_maintenance()
        except Exception:               # pragma: no cover - never block maintenance
            pass
        # Cron-free statcast freshness: keep the raw Savant corpus + derived
        # snapshots current so pitcher_asof (via get_or_fill) and live batter-prop
        # xBA don't go stale. Bounded + fail-open (see _statcast_maintenance).
        try:
            _statcast_maintenance()
        except Exception:               # pragma: no cover - never block maintenance
            pass
    newly_resolved = resolve_pending_outcomes(sport_key, max_to_resolve=max_resolve)
    # Team-market forecasts resolve alongside props but are kept OUT of the
    # newly_resolved count (that gates the prop Platt refit; team markets have no
    # Platt layer). Best-effort — never blocks prop maintenance.
    try:
        newly_resolved_markets = resolve_pending_market_outcomes(
            sport_key, max_to_resolve=max_resolve)
    except Exception:
        newly_resolved_markets = 0
    # Gate on the last fit's timestamp from the durable store (SQL-backed when
    # configured, else the local file's mtime). Using os.path.getmtime alone
    # would refit on every Cloud restart, since the ephemeral FS has no file.
    last_fit_ts, _ = _load_recal_cached(sport_key)
    do_refit = last_fit_ts is None
    if not do_refit:
        age_hours = (time.time() - last_fit_ts) / 3600.0
        if age_hours >= MIN_REFIT_INTERVAL_HOURS:
            do_refit = (_count_resolved_since(sport_key, last_fit_ts)
                        >= MIN_NEW_FOR_REFIT)
    fits = {}
    if do_refit:
        fits = refit_sport(
            sport_key, resolve_first=False, newly_resolved=newly_resolved)
    # Opportunistically bound log growth from repeated same-slate logging.
    # Writes only when duplicates actually exist, so this is free on a clean log.
    compact_prediction_log()
    return {"newly_resolved": newly_resolved,
            "newly_resolved_markets": newly_resolved_markets,
            "refit": bool(fits)}


def maybe_auto_refit(sport_key):
    """
    Called by analysis.py on first prop analysis per (process, sport).
    Runs bounded outcome maintenance at most hourly per process. Refit remains
    gated by calibration age and the number of newly resolved observations.
    Best-effort; never raises.
    """
    now = time.time()
    last_attempt = _last_auto_maintenance.get(sport_key, 0.0)
    if now - last_attempt < AUTO_MAINTENANCE_INTERVAL_SECONDS:
        return
    _last_auto_maintenance[sport_key] = now

    try:
        maintain_sport(sport_key)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap from existing book-line cache
# ──────────────────────────────────────────────────────────────────────────────

def seed_from_book_line_cache(sport, espn_sport, espn_league, sport_key, target_props):
    """
    Use existing odds-cache snapshots + ESPN gamelogs to fit Platt with the
    *current production* analysis pipeline (no need for new predictions to
    accumulate first). Calls book_line_calibration's machinery to produce
    (raw_prob, outcome) pairs, then fits Platt per prop.

    Returns dict {prop_key: (a, b, n_fit)} actually fit & saved.
    """
    from book_line_calibration import (
        harvest_book_lines, harvest_real_line_book_lines,
        join_book_lines_to_actuals, project_and_empirical, _team_defense_lookup,
    )
    from calibration_loader import (
        load_calibration as _load_cal,
        apply_calibration_with_warmup, count_current_season_games,
    )
    # Imported lazily (function scope) to avoid the props <-> recalibration
    # module import cycle; by call time both modules are fully loaded.
    from props import (
        _player_prop_half_life, _player_prop_defense_strength,
        _player_prop_venue_strength, _method_cfg_for_line,
        _composite_recal_key,
    )

    # Union the DURABLE historical_odds backfill store with the app's own
    # RESOLVED prediction log (deduped) so the seed dataset — and its line
    # buckets — keep growing for free with live usage, not just paid backfill
    # credits. Fall back to the ephemeral HTTP odds cache when the union is empty.
    book_lines, n_store, n_pred = harvest_real_line_book_lines(
        sport_key, target_props)
    if book_lines:
        print(f"[seed] {len(book_lines)} real-line obs "
              f"({n_store} store + {n_pred} prediction-log) for {sport_key}")
    else:
        book_lines = harvest_book_lines(sport_key, target_props)
    if not book_lines:
        return {}
    enriched = join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        return {}

    cal = _load_cal(sport_key) or {}

    # Resolve per-prop projection knobs EXACTLY as the production analyzer does
    # (props.py::analyze_player_props_value _knob resolution), so the seed fits
    # Platt on the SAME raw-probability distribution production emits. A single
    # global preset (e.g. VARIANT_PRESETS["all"], venue_strength=0.25 for all)
    # would fit Platt on the WRONG distribution: empirical_over depends on the
    # per-prop half_life and venue weighting, which vary by prop in the shipped
    # calibration (e.g. MLB half_life null/7/15, venue 0.0/0.25).
    default_half_life = _player_prop_half_life(sport_key)
    default_def_strength = _player_prop_defense_strength(sport_key)
    default_venue = _player_prop_venue_strength(sport_key)

    def _knob(cfg, name, default):
        if cfg and name in cfg:
            value = cfg[name]
            # half_life=null explicitly means equal weighting; other null knobs
            # mean "use the default".
            if value is not None or name == "half_life":
                return value
        return default

    def _params_for(prop_key):
        cfg = cal.get(prop_key) or {}
        return {
            "half_life": _knob(cfg, "half_life", default_half_life),
            # None venue/defense defaults -> 0.0 (project_and_empirical needs a
            # scalar strength; MLB props set both explicitly so this is exact).
            "venue_strength": _knob(cfg, "venue_strength", default_venue) or 0.0,
            "opp_defense_strength": (
                _knob(cfg, "opp_defense_strength", default_def_strength) or 0.0),
            "use_minutes": False,
        }

    # Team-defense lookup only if some target prop actually uses input-side
    # opponent-defense weighting (MLB props ship at 0.0 -> skipped).
    need_defense = any(
        (_params_for(pk)["opp_defense_strength"] or 0.0) > 0
        for pk in (target_props or [])
    )
    team_defense = league_avg_def = None
    if need_defense:
        team_defense, _, league_avg_def = _team_defense_lookup(
            espn_sport, espn_league)

    by_prop_records = defaultdict(list)

    for obs in enriched:
        params = _params_for(obs["prop_key"])
        projected, emp = project_and_empirical(obs, params, sport_key,
                                                team_defense, league_avg_def)
        if projected is None or emp is None:
            continue
        prop_cfg = cal.get(obs["prop_key"]) or {}
        # Resolve the per-LINE method/cfg (line_methods bucket) so the seed fits
        # Platt on the SAME raw distribution runtime emits for that line: a prop
        # whose ≥1.5 bucket adopted method B must be seeded with B's raw there,
        # not the pooled method's. A cfg without line_methods → the pooled method
        # (unchanged behavior).
        method, method_cfg = _method_cfg_for_line(prop_cfg, obs["line"])
        raw = emp
        if method:
            # Match runtime warmup blending: count only the player's prior games
            # inside the observation's *current season*, not every historical
            # game — otherwise the current-season fit is over-weighted and the
            # seeded raw probabilities diverge from what production emits.
            prior_dates = [g.get("game_date")
                           for g in (obs.get("prior_games") or [])]
            try:
                obs_now = datetime.fromisoformat(str(obs.get("game_date"))[:10])
            except (TypeError, ValueError):
                obs_now = None
            curr_games = count_current_season_games(
                prior_dates, sport_key, now=obs_now)
            p_cal = apply_calibration_with_warmup(
                method_cfg, projected, obs["line"], curr_games,
                empirical_over=emp,
            )
            if p_cal is not None:
                raw = max(0.0, min(1.0, p_cal))
        actual = obs["actual"]
        line = obs["line"]
        if actual == line:
            continue
        y = 1 if actual > line else 0
        # Per-line-bucket key: a prop with line_methods is fit per bucket under
        # "<prop>@<bucket>" (matching the runtime apply); a prop without it keeps
        # the bare prop_key (pooled, unchanged). A line that resolves to no bucket
        # -> None -> skip (symmetric with the apply side).
        rec_key = _composite_recal_key(
            obs["prop_key"], line, prop_cfg.get("line_methods"))
        if rec_key is None:
            continue
        by_prop_records[rec_key].append(
            (obs.get("game_date") or "", raw, y))

    fits = {}
    per_prop_params = {}
    for fit_key, records in by_prop_records.items():
        result = fit_platt_chronological(records)
        if result is None:
            continue
        a, b = result["a"], result["b"]
        fits[fit_key] = (a, b, result["n_fit"])
        per_prop_params[fit_key] = {
            "a": round(a, 5),
            "b": round(b, 5),
            "n_fit": result["n_fit"],
            "n_validation": result["n_validation"],
            "n_validation_folds": result["n_validation_folds"],
            "holdout_start": result["holdout_start"],
            "holdout_raw_brier": round(result["holdout_raw_brier"], 6),
            "holdout_calibrated_brier": round(
                result["holdout_calibrated_brier"], 6),
            "holdout_raw_log_loss": round(result["holdout_raw_log_loss"], 6),
            "holdout_calibrated_log_loss": round(
                result["holdout_calibrated_log_loss"], 6),
            "holdout_metric_scope": result.get(
                "holdout_metric_scope", "fold_cross_validation"),
            "deploy_fit_scope": result.get(
                "deploy_fit_scope", "all_observations"),
            "validation_folds": result["validation_folds"],
            "validated": True,
            "source": "book_line_cache_seed",
        }

    if per_prop_params:
        # Offline bootstrap: write the local file only (committed to git and
        # shipped as the Cloud baseline). The production SQL store is populated by
        # the online refit loop, not by local seed runs.
        save_recalibration(sport_key, per_prop_params,
                           meta={"source": "book_line_cache_seed"},
                           to_blob=False)
        _LOAD_CACHE.pop(sport_key, None)
    return fits


# ──────────────────────────────────────────────────────────────────────────────
# CLI: seed / refit / inspect
# ──────────────────────────────────────────────────────────────────────────────

def _main_cli():
    import argparse
    from backtest import SPORT_MAP
    from book_line_calibration import PROPS_BY_SPORT

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", action="store_true",
                   help="Bootstrap Platt fit from existing book-line cache "
                        "(no waiting for new predictions to accumulate).")
    p.add_argument("--refit", action="store_true",
                   help="Resolve pending outcomes and refit Platt from the "
                        "prediction log.")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="nba")
    p.add_argument("--show", action="store_true",
                   help="Print current recalibration params for the sport.")
    args = p.parse_args()

    # Target the durable SQL backend when the SQL_* secrets are
    # configured (mirrors the app's boot promotion + refit_calibration.main;
    # outside Streamlit these aren't in the env yet). Without this, --seed/--refit
    # read the odds_line warehouse + prediction log via _sql()/db_store.enabled(),
    # both False -> fall back to the empty local JSON store -> "Nothing fit."
    # --seed still writes local-only (save_recalibration to_blob=False); --refit
    # updates the durable overlay, as intended. Falls back to local when unset.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]
    target_props = PROPS_BY_SPORT.get(args.sport, [])

    if args.seed:
        print(f"Seeding Platt fit for {sport_key} from cached book lines...")
        fits = seed_from_book_line_cache(args.sport, espn_sport, espn_league,
                                         sport_key, target_props)
        if not fits:
            print("  Nothing fit — not enough cached lines + outcomes.")
        else:
            for prop_key, (a, b, n) in sorted(fits.items()):
                print(f"  {prop_key:<22} a={a:+.4f} b={b:+.4f}  n={n}")
            print(f"  Saved: {recalibration_path(sport_key)}")

    if args.refit:
        print(f"Refitting from prediction log for {sport_key}...")
        fits = refit_sport(sport_key)
        if not fits:
            print("  No prop had enough resolved entries to fit.")
        else:
            for prop_key, (a, b, n) in sorted(fits.items()):
                print(f"  {prop_key:<22} a={a:+.4f} b={b:+.4f}  n={n}")
            print(f"  Saved: {recalibration_path(sport_key)}")

    if args.show or not (args.seed or args.refit):
        params = load_recalibration(sport_key)
        if not params:
            print(f"(no recalibration_{sport_key}.json yet)")
        else:
            print(f"Current Platt params for {sport_key}:")
            for prop_key, cfg in sorted(params.items()):
                print(f"  {prop_key:<22} a={cfg.get('a'):+.4f} "
                      f"b={cfg.get('b'):+.4f}  n_fit={cfg.get('n_fit')}")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    _main_cli()
