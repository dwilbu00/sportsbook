"""prediction_replay.py — F28 P2 replay harness.

Recompute a stored prediction's RAW model probability from its recorded provenance
(event-derived season/week + the artifacts named by its hashes), with NO wall-clock
reads — the test of whether a historical recommendation is exactly reproducible.

Design:
  * Artifact PARITY first: if the serving calibration/model version changed since the
    row was logged (calibration_hash / model_schema differ from the current build), we
    do NOT claim a replay — status 'artifacts_changed'. This is the honest signal that
    a probability shift is a version change, not drift.
  * Recompute the RAW prob (row['raw_prob'] = the model p_over, pre-recalibration) from
    the recorded event_season/event_week — so replay never depends on the wall clock.
    (final_prob adds the online recal layer, which is not a clean standalone; the raw
    prob is the deterministic model output and the right parity target.)

Scope P2: NFL model-served rows (nfl_prop_serving.project is pure given season/week/
line/artifact). Other sports/methods return 'unsupported' until their as-of recompute
is wired. All best-effort; never raises.
"""
_TOL = 1e-6


def _replay_nfl_raw(row):
    """Recompute the NFL model's raw p_over for a stored row using its EVENT season/week
    (not wall-clock). None if unavailable."""
    try:
        season = int(row.get("event_season"))
        week = int(row.get("event_week"))
    except (TypeError, ValueError):
        return None            # need an event-derived week to replay deterministically
    try:
        import nfl_prop_serving
        import nfl_props_scan
        r = nfl_prop_serving.project(
            nfl_props_scan._norm(row.get("player")), season, week,
            row.get("prop_key"), row.get("line"))
        return None if not r else r.get("p_over")
    except Exception:
        return None


def replay(row, tol=_TOL):
    """Replay one stored prediction row. Returns a dict:
      {status, stored, recomputed, matches, reason?}
    status: 'ok' | 'no_provenance' | 'artifacts_changed' | 'unsupported'."""
    if not row.get("context_fingerprint"):
        return {"status": "no_provenance"}
    # Artifact parity: refuse to claim a replay across a serving-version change.
    try:
        import prediction_provenance
        cur = prediction_provenance.build_context(
            row.get("sport_key"), row.get("commence_time"),
            method=row.get("method"),
            reference_book_policy=row.get("reference_book_policy"))
    except Exception:
        cur = {}
    if (cur.get("calibration_hash") != row.get("calibration_hash")
            or cur.get("model_schema") != row.get("model_schema")):
        return {"status": "artifacts_changed",
                "stored_fingerprint": row.get("context_fingerprint"),
                "current_fingerprint": cur.get("context_fingerprint")}

    sport, method = row.get("sport_key"), row.get("method")
    if sport == "americanfootball_nfl" and method == "nfl_model":
        recomputed = _replay_nfl_raw(row)
    else:
        return {"status": "unsupported",
                "reason": f"replay for {sport}/{method} not wired (P2 = NFL model)"}
    if recomputed is None:
        return {"status": "unsupported", "reason": "recompute unavailable"}

    stored = row.get("raw_prob")
    matches = stored is not None and abs(recomputed - float(stored)) <= tol
    return {"status": "ok", "stored": stored, "recomputed": recomputed,
            "matches": matches}


def replay_many(rows, tol=_TOL):
    """Replay a batch of rows; return a status tally plus a few divergence examples.
    'matched'/'mismatched' split the 'ok' rows by whether recompute == stored."""
    summary = {"total": 0, "ok": 0, "matched": 0, "mismatched": 0,
               "artifacts_changed": 0, "no_provenance": 0, "unsupported": 0,
               "mismatches": []}
    for row in rows or []:
        out = replay(row, tol=tol)
        st = out.get("status")
        summary["total"] += 1
        if st == "ok":
            summary["ok"] += 1
            key = "matched" if out.get("matches") else "mismatched"
            summary[key] += 1
            if key == "mismatched" and len(summary["mismatches"]) < 20:
                summary["mismatches"].append({
                    "player": row.get("player"), "prop_key": row.get("prop_key"),
                    "line": row.get("line"), "event_season": row.get("event_season"),
                    "event_week": row.get("event_week"),
                    "stored": out.get("stored"), "recomputed": out.get("recomputed")})
        elif st in summary:
            summary[st] += 1
    return summary


def replay_log(sport_key=None, limit=None, tol=_TOL):
    """Replay stored prediction rows from the live log (READ-ONLY). Optional sport
    filter + cap to the N most-recent rows. Returns the replay_many summary."""
    try:
        import recalibration
        rows = recalibration.read_prediction_log() or []
    except Exception as e:                                   # pragma: no cover
        return {"error": f"could not read prediction log: {e}"}
    if sport_key:
        rows = [r for r in rows if r.get("sport_key") == sport_key]
    rows = sorted(rows, key=lambda r: str(r.get("ts") or ""), reverse=True)
    if limit:
        rows = rows[:int(limit)]
    return replay_many(rows, tol=tol)


if __name__ == "__main__":                                  # pragma: no cover
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(
        description="F28 P2 replay harness — verify stored predictions reproduce "
                    "from their recorded provenance (read-only).")
    ap.add_argument("--sport", default=None,
                    help="filter to one sport_key (e.g. americanfootball_nfl)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap to the N most-recent rows")
    ap.add_argument("--tol", type=float, default=_TOL)
    a = ap.parse_args()
    print(_json.dumps(replay_log(sport_key=a.sport, limit=a.limit, tol=a.tol),
                      indent=2, default=str))
