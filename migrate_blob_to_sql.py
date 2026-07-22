"""One-time backfill: copy durable state from Azure Blob into Azure SQL.

Run ONCE on the warm machine AFTER:
  1. creating the SQL tables manually (sql/schema.sql) + grants (sql/grants.sql),
  2. putting SQL_SERVER/SQL_DATABASE/SQL_USER/SQL_PASSWORD and the existing
     PREDICTION_LOG_BLOB_URL in .streamlit/secrets.toml.

Two phases, deliberately ordered so the SQL dispatch is OFF while reading Blob:
  * Phase 1 reads the Blob stores (prediction log, wagers, recal params). This
    requires the SQL_* env vars to be UNSET (so recalibration's SQL dispatch stays
    off and the reads hit Blob). The blob SAS URL is read from secrets.toml.
  * Phase 2 promotes the SQL_* secrets into the environment and writes to SQL.

Idempotent: prediction rows upsert by forecast identity, wagers upsert by
wager_id, recal is replace-per-sport. Copy-not-move — the Blob is left intact, so
removing the SQL secret rolls the app back to Blob with no data loss.

Scope: prediction log + wagers ledger + recalibration params. The odds warehouse
starts fresh in SQL (Phase B of the migration).

Usage:
    python migrate_blob_to_sql.py --dry-run   # report Blob counts only
    python migrate_blob_to_sql.py             # migrate
"""

import argparse
import os

import db_store
import recalibration
import wagers

SPORTS = ("baseball_mlb", "basketball_nba", "americanfootball_nfl")
WAGERS_FILE = "wagers.jsonl"


def _valid_prediction(row):
    """A prediction row must carry the NOT-NULL / identity columns."""
    return bool(row.get("sport_key") and row.get("prop_key")
                and row.get("player") and row.get("line") is not None)


def _migrate_predictions(pred):
    """Upsert prediction rows into SQL, LOUDLY (a failure aborts the migration).

    Routes through mutate_prediction_log (which, on the SQL path, propagates a
    constraint/DB error) rather than log_prediction_rows (which swallows). Skips
    rows missing required columns so one legacy row can't abort the whole insert,
    and dedupes by forecast identity (idempotent re-run)."""
    valid = [row for row in pred if _valid_prediction(row)]
    skipped = len(pred) - len(valid)
    if skipped:
        print(f"  (skipped {skipped} prediction row(s) missing required fields)")
    if not valid:
        return 0

    def upsert(rows):
        by_ident = {recalibration.prediction_identity(r): r for r in rows}
        for row in valid:
            by_ident[recalibration.prediction_identity(row)] = row  # blob wins
        rows[:] = list(by_ident.values())
        return len(rows)

    return recalibration.mutate_prediction_log(upsert)


def _read_blob_state():
    """(prediction_rows, wager_rows, {sport: recal_cfg}) read from Blob."""
    pred = recalibration.read_prediction_log()
    wager_rows, _ = recalibration._read_ndjson_blob(WAGERS_FILE)
    recal = {}
    for sport in SPORTS:
        status, blob, _ = recalibration._read_json_blob(
            f"recalibration_{sport}.json")
        if status == "ok" and isinstance(blob, dict) and blob.get("props"):
            recal[sport] = blob
    return pred, wager_rows, recal


def main():
    parser = argparse.ArgumentParser(description="Backfill Azure Blob → Azure SQL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be migrated; write nothing.")
    args = parser.parse_args()

    # ── Phase 1: read Blob (SQL dispatch must be OFF) ──
    if db_store.enabled():
        raise SystemExit(
            "SQL_* env vars are set — unset them for the Blob read phase "
            "(this script promotes them itself before writing).")
    if not recalibration._prediction_log_blob_url():
        raise SystemExit(
            "No PREDICTION_LOG_BLOB_URL configured — nothing to migrate from.")
    pred, wager_rows, recal = _read_blob_state()
    print(f"Blob source: {len(pred)} prediction rows, {len(wager_rows)} wagers, "
          f"recal for {sorted(recal) or 'none'}")
    if args.dry_run:
        print("(dry run — no writes)")
        return

    # ── Phase 2: enable SQL and write ──
    if not db_store.promote_secrets_from_toml():
        raise SystemExit("SQL_* secrets not configured; cannot write to SQL.")
    if not recalibration._sql():
        raise SystemExit("SQL backend did not enable after promoting secrets.")

    # submit_wagers and save_recal already raise on failure (loud); the
    # prediction log is routed through the loud path by _migrate_predictions.
    n_pred = _migrate_predictions(pred) if pred else 0
    n_wag = wagers.submit_wagers(wager_rows) if wager_rows else 0
    n_recal = 0
    for sport, cfg in recal.items():
        db_store.save_recal(sport, cfg)  # SQL-only (never touches the git file)
        n_recal += 1

    print(f"Migrated to Azure SQL: {n_pred} prediction rows now present, "
          f"{n_wag} wagers added, {n_recal} recal sport(s).")
    print("Blob left intact (copy-not-move). Set the SQL_* secrets in the app to "
          "cut over; remove them to roll back.")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
