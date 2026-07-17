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
REMOTE_LOG_URL_ENV = "PREDICTION_LOG_BLOB_URL"

MIN_FIT_SAMPLES = 50          # below this, skip Platt fit for a prop
MIN_VALIDATION_SAMPLES = 20   # later chronological observations held out
MIN_NEW_FOR_REFIT = 25        # need this many new resolved obs to bother refitting
MIN_REFIT_INTERVAL_HOURS = 12 # don't re-resolve+refit more than this often
MAX_RESOLVE_PER_LAUNCH = 80   # cap ESPN calls per auto-refit cycle
AUTO_MAINTENANCE_INTERVAL_SECONDS = 3600

_lock = threading.Lock()
_last_auto_maintenance = {}  # sport_key -> attempt timestamp


# ──────────────────────────────────────────────────────────────────────────────
# Path / file helpers
# ──────────────────────────────────────────────────────────────────────────────

def recalibration_path(sport_key):
    return os.path.join(CALIB_DIR, f"recalibration_{sport_key}.json")


def _ensure_dirs():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)


def _prediction_log_blob_url():
    """Return an optional Azure Blob SAS URL for durable shared log storage."""
    url = os.environ.get(REMOTE_LOG_URL_ENV, "").strip()
    if url:
        return url
    secrets_path = os.path.join(SCRIPT_DIR, ".streamlit", "secrets.toml")
    try:
        import tomllib
        with open(secrets_path, "rb") as f:
            value = tomllib.load(f).get(REMOTE_LOG_URL_ENV)
        return str(value).strip() if value else ""
    except (ImportError, OSError, TypeError, ValueError):
        return ""


def prediction_log_storage():
    """Human-readable active prediction-log backend."""
    return "Azure Blob" if _prediction_log_blob_url() else "Local cache"


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
def _local_log_lock():
    """Hold an inter-process lock while replacing the local prediction log."""
    _ensure_dirs()
    lock_path = LOG_PATH + ".lock"
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


def _read_log_snapshot():
    """Return (rows, version) from local disk or the configured Azure blob."""
    blob_url = _prediction_log_blob_url()
    if blob_url:
        import requests
        response = requests.get(blob_url, timeout=30)
        if response.status_code == 404:
            return [], None
        response.raise_for_status()
        return _parse_log_text(response.text), response.headers.get("ETag")
    if not os.path.exists(LOG_PATH):
        return [], None
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return _parse_log_text(f.read()), None


def _write_log_snapshot(rows, version=None):
    """Conditionally write a complete log snapshot."""
    content = _serialize_log(rows)
    blob_url = _prediction_log_blob_url()
    if blob_url:
        import requests
        headers = {
            "Content-Type": "application/x-ndjson",
            "x-ms-blob-type": "BlockBlob",
            "x-ms-version": "2023-11-03",
        }
        headers["If-Match" if version else "If-None-Match"] = version or "*"
        response = requests.put(
            blob_url, data=content.encode("utf-8"), headers=headers, timeout=30)
        if response.status_code in (409, 412):
            raise _LogConflict()
        response.raise_for_status()
        return
    _ensure_dirs()
    tmp = LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, LOG_PATH)


def mutate_prediction_log(mutator, max_retries=5):
    """Atomically mutate the prediction log; returns the mutator's result."""
    if not _prediction_log_blob_url():
        with _lock:
            with _local_log_lock():
                rows, version = _read_log_snapshot()
                result = mutator(rows)
                if result:
                    _write_log_snapshot(rows, version)
                return result
    for _ in range(max_retries):
        rows, version = _read_log_snapshot()
        result = mutator(rows)
        if not result:
            return result
        try:
            _write_log_snapshot(rows, version)
            return result
        except _LogConflict:
            continue
    raise RuntimeError("Prediction log changed repeatedly; update was not saved")


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def log_prediction(sport_key, prop_key, player, game_date, line, raw_prob,
                   projected=None, direction=None, price=None, book=None,
                   final_prob=None, event_id=None, commence_time=None,
                   is_value=None, write=True):
    """Build and optionally append one prediction row. Best-effort, never raises."""
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
        "is_value": bool(is_value) if is_value is not None else None,
        "resolved": False,
        "actual": None,
        "outcome": None,    # 1=over_won, 0=under_won, None=push/unresolved
    }
    if write:
        log_prediction_rows([row])
    return row


def log_prediction_rows(new_rows):
    """Append prediction rows, de-duplicating by forecast identity.

    Re-viewing the same slate would otherwise append a fresh row for every
    (sport, event, prop, player, line) on each analysis, growing the log
    without bound and rewriting the whole blob every append. Instead we keep a
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


def _read_log():
    try:
        rows, _ = _read_log_snapshot()
        return rows
    except Exception:
        return []


def read_prediction_log():
    """Public read-only snapshot for maintenance tools."""
    return _read_log()


def prediction_identity(row):
    """Stable forecast identity, with legacy fallback for pre-event-ID rows."""
    event_ref = row.get("event_id") or row.get("game_date")
    return (
        row.get("sport_key"), event_ref, row.get("prop_key"),
        row.get("player"), row.get("line"),
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


def resolve_pending_outcomes(sport_key, max_to_resolve=MAX_RESOLVE_PER_LAUNCH):
    """
    Walk the log, find unresolved entries for this sport whose game_date is
    in the past, and fill in actual + outcome from cached ESPN gamelogs.
    Caps per-launch resolution by `max_to_resolve` to bound ESPN cost.
    Returns the number of newly resolved entries.
    """
    pair = SPORT_ESPN_MAP.get(sport_key)
    if not pair:
        return 0
    try:
        from espn_cache import cached_athlete_id, cached_gamelog
    except Exception:
        return 0

    espn_sport, espn_league = pair
    rows = _read_log()
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

    resolved_count = 0
    resolved_updates = {}
    for player, p_rows in by_player.items():
        if resolved_count >= max_to_resolve:
            break
        try:
            aid = cached_athlete_id(espn_sport, espn_league, player)
            if not aid:
                continue
            # Outcome maintenance needs yesterday's result; the cache helper's
            # 30-day historical default is too stale for forward tracking.
            gamelog = cached_gamelog(
                espn_sport, espn_league, aid, ttl_hours=6)
            if not gamelog:
                continue
        except Exception:
            continue

        date_idx = {}
        for i, g in enumerate(gamelog):
            d = (g.get("game_date") or "")[:10]
            if d and d not in date_idx:
                date_idx[d] = i

        for r in p_rows:
            stat_label = _stat_label(r["prop_key"], gamelog)
            if not stat_label:
                continue
            d = r["game_date"]
            idx = date_idx.get(d)
            if idx is None:
                # ±1 day fallback for timezone slippage
                for delta in (-1, 1):
                    try:
                        dt = datetime.fromisoformat(d) + timedelta(days=delta)
                        alt = dt.date().isoformat()
                    except Exception:
                        continue
                    if alt in date_idx:
                        idx = date_idx[alt]
                        break
            if idx is None:
                continue
            actual = gamelog[idx].get(stat_label)
            if actual is None:
                continue
            try:
                actual = float(actual)
            except (TypeError, ValueError):
                continue
            line = float(r["line"])
            if actual == line:
                outcome = None  # push
            else:
                outcome = 1 if actual > line else 0
            resolved_at = datetime.now(timezone.utc).isoformat()
            resolved_updates[prediction_row_key(r)] = {
                "actual": actual,
                "outcome": outcome,
                "resolved": True,
                "resolved_at": resolved_at,
            }
            resolved_count += 1
            if resolved_count >= max_to_resolve:
                break

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
        return mutate_prediction_log(apply_resolutions)
    except Exception:
        return 0


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


def fit_platt_chronological(records):
    """
    Validate Platt scaling on two expanding-window chronological folds before
    fitting the final parameters on all observations. `records` contains
    (date_or_timestamp, raw_probability, outcome).

    Returns a parameter/metric dict, or None when the calibrated probabilities
    do not improve both Brier score and log loss in every untouched later fold.
    Rows sharing a date always stay in the same side of a boundary.
    """
    rows = sorted(
        (str(date), float(raw), int(outcome))
        for date, raw, outcome in records
        if raw is not None and outcome in (0, 1)
    )
    if len(rows) < MIN_FIT_SAMPLES + 2 * MIN_VALIDATION_SAMPLES:
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

def save_recalibration(sport_key, per_prop_params, meta=None):
    _ensure_dirs()
    blob = {
        "sport_key": sport_key,
        "fit_timestamp": datetime.now(timezone.utc).isoformat(),
        "props": per_prop_params,
    }
    if meta:
        blob["meta"] = meta
    tmp = recalibration_path(sport_key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    os.replace(tmp, recalibration_path(sport_key))


_LOAD_CACHE = {}  # sport_key -> (mtime, blob)


def load_recalibration(sport_key):
    """Return {prop_key: {"a": ..., "b": ..., "n_fit": ...}} or {}."""
    path = recalibration_path(sport_key)
    if not os.path.exists(path):
        return {}
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    cached = _LOAD_CACHE.get(sport_key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return {}
    # Legacy fits were trained and evaluated on the same rows. Keep them on
    # disk for provenance, but do not apply them until a chronological holdout
    # has demonstrated that recalibration improves unseen probabilities.
    props = {
        prop_key: cfg
        for prop_key, cfg in (blob.get("props", {}) or {}).items()
        if cfg.get("validated") is True
    }
    _LOAD_CACHE[sport_key] = (mtime, props)
    return props


# ──────────────────────────────────────────────────────────────────────────────
# Refit (resolve + fit + save) and gated auto-refit
# ──────────────────────────────────────────────────────────────────────────────

def refit_sport(sport_key, resolve_first=True, max_resolve=MAX_RESOLVE_PER_LAUNCH,
                newly_resolved=None):
    """Resolve pending outcomes, then refit Platt for every prop with enough
    resolved entries. Returns dict {prop_key: (a, b, n_fit)}."""
    if resolve_first:
        newly_resolved = resolve_pending_outcomes(sport_key, max_to_resolve=max_resolve)
    elif newly_resolved is None:
        newly_resolved = 0

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
        by_prop_records[r["prop_key"]].append(
            (order_key, float(r["raw_prob"]), int(o)))

    fits = {}
    per_prop_params = {}
    for prop_key, records in by_prop_records.items():
        result = fit_platt_chronological(records)
        if result is None:
            continue
        a, b = result["a"], result["b"]
        fits[prop_key] = (a, b, result["n_fit"])
        per_prop_params[prop_key] = {
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


def maintain_sport(sport_key):
    """Resolve pending rows and refit only when the existing gates allow it."""
    newly_resolved = resolve_pending_outcomes(sport_key)
    path = recalibration_path(sport_key)
    do_refit = not os.path.exists(path)
    last_fit_ts = 0.0
    if not do_refit:
        try:
            last_fit_ts = os.path.getmtime(path)
        except OSError:
            last_fit_ts = 0.0
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
    return {"newly_resolved": newly_resolved, "refit": bool(fits)}


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
        harvest_book_lines, join_book_lines_to_actuals,
        project_and_empirical, _team_defense_lookup, VARIANT_PRESETS,
    )
    from backtest import _resolve_params
    from calibration_loader import (
        load_calibration as _load_cal,
        apply_calibration_with_warmup, count_current_season_games,
    )

    book_lines = harvest_book_lines(sport_key, target_props)
    if not book_lines:
        return {}
    enriched = join_book_lines_to_actuals(book_lines, espn_sport, espn_league)
    if not enriched:
        return {}

    # Use the "all" preset (matches what production resolves to in most cases)
    params = _resolve_params(VARIANT_PRESETS["all"], sport_key)
    team_defense = league_avg_def = None
    if params.get("opp_defense_strength", 0.0) > 0:
        team_defense, _, league_avg_def = _team_defense_lookup(espn_sport, espn_league)

    cal = _load_cal(sport_key) or {}

    by_prop_records = defaultdict(list)

    for obs in enriched:
        projected, emp = project_and_empirical(obs, params, sport_key,
                                                team_defense, league_avg_def)
        if projected is None or emp is None:
            continue
        prop_cfg = cal.get(obs["prop_key"]) or {}
        raw = emp
        if prop_cfg.get("method"):
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
                prop_cfg, projected, obs["line"], curr_games,
                empirical_over=emp,
            )
            if p_cal is not None:
                raw = max(0.0, min(1.0, p_cal))
        actual = obs["actual"]
        line = obs["line"]
        if actual == line:
            continue
        y = 1 if actual > line else 0
        by_prop_records[obs["prop_key"]].append(
            (obs.get("game_date") or "", raw, y))

    fits = {}
    per_prop_params = {}
    for prop_key, records in by_prop_records.items():
        result = fit_platt_chronological(records)
        if result is None:
            continue
        a, b = result["a"], result["b"]
        fits[prop_key] = (a, b, result["n_fit"])
        per_prop_params[prop_key] = {
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
        save_recalibration(sport_key, per_prop_params,
                           meta={"source": "book_line_cache_seed"})
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
