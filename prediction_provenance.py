"""prediction_provenance.py — F28 P1 provenance stamp for prediction rows.

Builds the immutable context recorded on each NEW prediction (recalibration.
log_prediction) so a stored prediction can be attributed and later replayed:

  * event_season / event_week  — from the EVENT's commence_time, NOT wall-clock
    (fixes the NFL week=99 serve-clock gap for the recorded cohort);
  * as_of                      — the serve/quote cutoff timestamp;
  * calibration_hash           — calibration_loader.serving_fingerprint(sport);
  * model_schema               — feature_store.FEATURE_SCHEMA_VERSION + a content hash
                                 of the frozen model artifact(s);
  * method                     — the serving method actually used (+ fallback);
  * reference_book_policy       — how the de-vig reference was formed;
  * context_fingerprint        — md5 of the version inputs above (EXCLUDING the volatile
                                 as_of), so it changes iff a model/data/quote version
                                 changes — a stable id for attribution/partitioning.

All best-effort: any field that can't be computed is None and nothing raises, so a
logging call never breaks on provenance. Legacy rows (no context) stay valid as
unknown provenance.
"""
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

# Frozen model artifacts whose CONTENT is hashed into model_schema (stable across
# machines, unlike mtime). Add files here as more sports freeze artifacts.
_ARTIFACT_FILES = ("calibration/nfl_prop_models.json",)
_ARTIFACT_HASH = None
# Fields that define the serving VERSION (drive context_fingerprint); as_of excluded.
_FP_FIELDS = ("event_season", "event_week", "calibration_hash", "model_schema",
              "method", "reference_book_policy")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _base_dir():
    return os.path.dirname(os.path.abspath(__file__))


def artifact_hash():
    """Short stable content hash of the frozen model artifact(s); '' if none readable.
    Cached in-process (artifacts change only on an offline refit + redeploy)."""
    global _ARTIFACT_HASH
    if _ARTIFACT_HASH is not None:
        return _ARTIFACT_HASH
    h = hashlib.md5()
    found = False
    for rel in _ARTIFACT_FILES:
        try:
            with open(os.path.join(_base_dir(), rel), "rb") as f:
                data = f.read()
            # Normalize line endings (CRLF/CR -> LF) so the SAME committed text
            # artifact hashes identically on Windows (git checks out CRLF) and the
            # Cloud's Linux (LF). Without this, replay on a Windows dev box reports
            # every server-stamped row as artifacts_changed on a pure EOL diff.
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            h.update(rel.encode("utf-8"))
            h.update(data)
            found = True
        except Exception:
            continue
    _ARTIFACT_HASH = h.hexdigest()[:12] if found else ""
    return _ARTIFACT_HASH


def _nfl_season_of(dt):
    # NFL season Y runs Sep Y → Feb Y+1, so a Jan/Feb game belongs to season Y-1.
    return dt.year if dt.month >= 8 else dt.year - 1


def event_season_week(sport_key, commence_time):
    """(season, week) from the EVENT's commence_time (not wall-clock). NFL maps the
    date to the schedule's (season, week); other sports return (calendar year, None)."""
    try:
        dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
    except Exception:
        return None, None
    if sport_key != "americanfootball_nfl":
        return dt.year, None
    season = _nfl_season_of(dt)
    try:
        import nfl_schedule
        # The schedule's gameday is US-Eastern local; an evening kickoff (SNF/MNF/TNF,
        # ~8pm ET) rolls to the NEXT calendar day in UTC, so commence_time's UTC date
        # can be one day AHEAD of gameday. Match the UTC date OR the day before — games
        # on two adjacent days are always the same NFL week, so this never picks a
        # wrong week, and it fixes every primetime game (was silently week=None).
        cands = {dt.date().isoformat(),
                 (dt.date() - timedelta(days=1)).isoformat()}
        for r in nfl_schedule.load_games([season]):
            if str(r.get("gameday"))[:10] in cands and r.get("week"):
                return season, int(r["week"])
    except Exception:
        pass
    return season, None


def model_schema(sport_key):
    try:
        import feature_store
        fsv = feature_store.FEATURE_SCHEMA_VERSION
    except Exception:
        fsv = "?"
    ah = artifact_hash()
    return f"fs{fsv}:art{ah}" if ah else f"fs{fsv}"


def build_context(sport_key, commence_time, method=None,
                  reference_book_policy=None, as_of=None):
    """Immutable provenance context dict for one prediction. Best-effort; never raises."""
    try:
        season, week = event_season_week(sport_key, commence_time)
    except Exception:
        season, week = None, None
    try:
        import calibration_loader
        cal = calibration_loader.serving_fingerprint(sport_key)
    except Exception:
        cal = None
    ctx = {
        "event_season": season,
        "event_week": week,
        "as_of": as_of or _now_iso(),
        "calibration_hash": cal,
        "model_schema": model_schema(sport_key),
        "method": (str(method) if method else None),
        "reference_book_policy": (str(reference_book_policy)[:64]
                                  if reference_book_policy else None),
    }
    payload = json.dumps({k: ctx.get(k) for k in _FP_FIELDS},
                         sort_keys=True, default=str)
    ctx["context_fingerprint"] = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    return ctx
