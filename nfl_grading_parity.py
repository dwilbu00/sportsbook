"""nfl_grading_parity.py — the confidence gate for the nflverse NFL grading cutover.

Re-resolves every already-graded NFL player-prop row via the NEW nflverse path
(recalibration._resolve_nfl_actual) and diffs it against the stored actual (graded by
the OLD ESPN gamelog path). The swap is only safe to flip on when MISMATCH == 0.

Reads the live prediction_log (promotes SQL secrets, exactly like prediction_replay.py)
and fetches nflverse live (NO Odds-API credits — ESPN/nflverse box scores are free).

Run (owner box):
    PYTHONIOENCODING=utf-8 python nfl_grading_parity.py
    PYTHONIOENCODING=utf-8 python nfl_grading_parity.py --verbose      # list every row

Look for:  MATCH == eligible, MISMATCH == 0.  A nonzero MISMATCH prints each offending
row (player / prop / date / stored vs nflverse) and exits 1 — do NOT flip until it's 0.
"""
import argparse
import sys

import recalibration as rc

_TOL = 1e-6            # stats are integers/exact; only floating dust should differ


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _promote_secrets():
    try:
        import db_store
        db_store.promote_secrets_from_toml()
        return db_store.enabled()
    except Exception:
        return False


def run(verbose=False, limit=None):
    on_sql = _promote_secrets()
    print(f"prediction_log storage: {rc.prediction_log_storage()}")
    if not on_sql:
        print("  WARNING: SQL backend NOT configured — reading the local/empty store. "
              "Set SQL_* secrets to diff the real corpus.")

    rows = rc.read_prediction_log()
    eligible = [
        r for r in rows
        if r.get("sport_key") == "americanfootball_nfl"
        and r.get("resolved")
        and r.get("prop_key") in rc._NFLVERSE_PROP_COL
        and _to_float(r.get("actual")) is not None
    ]
    if limit:
        eligible = eligible[:limit]

    match = mismatch = pending = 0
    mism_rows, pend_rows = [], []
    for r in eligible:
        prop_key = r.get("prop_key")
        player = r.get("player")
        game_date = r.get("game_date")
        commence = r.get("commence") or r.get("commence_time")
        stored = _to_float(r.get("actual"))
        nflv = rc._resolve_nfl_actual(prop_key, player, game_date, commence)
        if nflv is None:
            pending += 1
            pend_rows.append((player, prop_key, game_date, stored))
            if verbose:
                print(f"  PENDING  {player:<24} {prop_key:<22} {str(game_date)[:10]}  "
                      f"stored={stored:g}  nflverse=<no row>")
            continue
        if abs(nflv - stored) <= _TOL:
            match += 1
            if verbose:
                print(f"  MATCH    {player:<24} {prop_key:<22} {str(game_date)[:10]}  "
                      f"{stored:g}")
        else:
            mismatch += 1
            mism_rows.append((player, prop_key, game_date, stored, nflv))
            print(f"  MISMATCH {player:<24} {prop_key:<22} {str(game_date)[:10]}  "
                  f"stored={stored:g}  nflverse={nflv:g}")

    print("\nNFL grading parity — nflverse vs stored (ESPN) actuals")
    print(f"  eligible resolved NFL prop rows : {len(eligible)}")
    print(f"  MATCH                           : {match}")
    print(f"  MISMATCH  (must be 0 to flip)   : {mismatch}")
    print(f"  nflverse PENDING (no row yet)   : {pending}")
    if pending and not verbose:
        print("  (re-run with --verbose to list the pending rows; common causes: a very "
              "old game, a DNP, or nflverse not having posted that week yet.)")
    if mismatch == 0:
        print("\n  RESULT: clean — safe to flip (_NFL_GRADE_FROM_NFLVERSE_DEFAULT = True "
              "or ODI_NFL_GRADE_NFLVERSE=1).")
    else:
        print(f"\n  RESULT: {mismatch} mismatch(es) — DO NOT flip; investigate above.")
    return mismatch


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="list every row, not just mismatches")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (debug)")
    a = ap.parse_args()
    sys.exit(1 if run(verbose=a.verbose, limit=a.limit) else 0)
