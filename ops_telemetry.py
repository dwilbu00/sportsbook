"""Lightweight structured operational telemetry (audit P20, WS1 subset).

A single-user Streamlit app doesn't need an event bus; it needs its silent
fail-open / fail-closed handlers to become *observable*. ``ops_event(kind,
**fields)`` logs one structured line and bumps a per-run counter, WITHOUT changing
control flow — callers keep swallowing the exception exactly as before, so no
betting math and no fail-open behavior changes.

WS1 wires only the ``database_failure`` kind: the durable-store errors that used to
vanish into ``except Exception: pass``. This complements the WS1 SQL-off guard —
the guard fails loud on a *misconfigured* backend, this surfaces *transient* DB
errors on a correctly-configured one. WS14 later adds the remaining kinds
(``identity_failure``, ``model_fallback``, ``api_failure``, ``cache_stale``, …)
and a Streamlit Diagnostics expander that renders ``counters()``.

Stdlib-only and dependency-free on purpose (imported by db_store/recalibration/
warehouse) so it can never introduce an import cycle. The logger adds no handlers,
so it inherits the app/root logging config; with none, Python's last-resort
handler still surfaces WARNING+ on stderr.
"""
import logging
import threading
from collections import Counter

logger = logging.getLogger("sportsbook.ops")
# Library convention: attach a NullHandler so events stay silent by default
# (no last-resort stderr spam in tests / an unconfigured process) yet still
# propagate to the app/root logging config when the operator sets one up.
logger.addHandler(logging.NullHandler())

_counters = Counter()
_lock = threading.Lock()


def ops_event(kind, level=logging.WARNING, **fields):
    """Record one operational event: bump its per-run counter and log a single
    structured line. Never raises — telemetry must not break a caller's
    (deliberately fail-open) path."""
    try:
        with _lock:
            _counters[kind] += 1
        if fields:
            detail = " ".join(f"{k}={v!r}" for k, v in sorted(fields.items()))
            logger.log(level, "ops_event kind=%s %s", kind, detail)
        else:
            logger.log(level, "ops_event kind=%s", kind)
    except Exception:  # pragma: no cover - telemetry is best-effort
        pass


def counters():
    """Snapshot of per-run event counts (for the WS14 Diagnostics expander)."""
    with _lock:
        return dict(_counters)


def count(kind):
    """Per-run count for one event kind (0 if never seen)."""
    with _lock:
        return _counters.get(kind, 0)


def reset_counters():
    """Clear all per-run counters (tests, or an explicit per-run reset)."""
    with _lock:
        _counters.clear()
