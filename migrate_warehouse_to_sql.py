"""One-time backfill: copy the accumulated Azure Blob odds warehouse into SQL,
then recompute closing-line value (CLV) on the ledger.

``warehouse.py --seed-from-store`` only seeds the small local historical_odds
file. The real accumulated closing-line history is the Blob warehouse the live
app captured (``warehouse/{sport}/{date}/{event}/{kind}/*.json`` + per-day
manifests) before the Phase B cutover — this migrates that into the normalized
``odds_snapshot``/``odds_line`` SQL tables.

Because the SAS has no ``list`` right, snapshots are enumerable only via the
per-(sport, date) manifests, so this scans the dates that actually matter for
CLV: every (sport, UTC-commence-date) present in the wagers ledger and the
prediction log (both read from SQL). Pass ``--dates`` to add more.

The Blob reads go through warehouse's blob-only helpers (``_read_json`` /
``read_snapshot``), which ignore the SQL flag, so Blob-read + SQL-write run
together. Writes are write-once (idempotent — safe to re-run). Finally it clears
and recomputes CLV so the ledger reflects the corrected (nearest-at-or-before)
closing snapshot.

Prereqs: SQL_* + PREDICTION_LOG_BLOB_URL in .streamlit/secrets.toml; the odds
tables created (sql/schema.sql); pymssql installed.

Usage:
    python migrate_warehouse_to_sql.py --dry-run
    python migrate_warehouse_to_sql.py
    python migrate_warehouse_to_sql.py --dates 2026-07-20,2026-07-21
"""

import argparse

import db_store
import recalibration
import wagers
import warehouse


def _commence_date(row):
    return ((row.get("commence_time") or "")[:10]
            or (row.get("game_date") or "")[:10])


def _relevant_pairs(extra_dates):
    """(sport, UTC-date) pairs from the ledger + prediction log (SQL-backed)."""
    pairs = set()
    for row in wagers.read_wagers():
        sport, date = row.get("sport_key"), _commence_date(row)
        if sport and date:
            pairs.add((sport, date))
    for row in recalibration.read_prediction_log():
        sport, date = row.get("sport_key"), _commence_date(row)
        if sport and date:
            pairs.add((sport, date))
    for sport, date in extra_dates:
        pairs.add((sport, date))
    return pairs


def _migrate(pairs, dry_run):
    read = written = 0
    for sport, date in sorted(pairs):
        status, manifest, _ = warehouse._read_json(
            warehouse.manifest_name(sport, date))
        if status != "ok" or not isinstance(manifest, dict):
            continue
        for entry in manifest.get("snapshots", []):
            env = warehouse.read_snapshot(entry.get("name"))  # blob-only read
            if not env or not isinstance(env.get("payload"), dict):
                continue
            read += 1
            if dry_run:
                continue
            kind = env.get("kind")
            meta = {
                "sport": env.get("sport") or sport,
                "game_date": (env.get("commence_time") or "")[:10] or date,
                "event_id": env.get("event_id"),
                "kind": kind,
                "snapshot_hour": warehouse._hour_bucket(env.get("captured_at")),
                "captured_at": env.get("captured_at"),
                "commence_time": env.get("commence_time"),
                "home": env.get("home"), "away": env.get("away"),
                "regions": env.get("regions"), "markets": env.get("markets"),
                "bookmakers": warehouse._books_str(env.get("bookmakers")),
            }
            if not meta["event_id"]:
                continue
            lines = warehouse._enumerate_lines(
                env["payload"], env.get("format"), kind)
            meta, lines = warehouse._enrich_ids(meta.get("sport"), meta, lines)
            if db_store.capture_odds_snapshot(meta, lines):
                written += 1
    return read, written


def main():
    parser = argparse.ArgumentParser(
        description="Backfill the Azure Blob odds warehouse into SQL + recompute CLV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan + count only; write nothing.")
    parser.add_argument("--dates", default="",
                        help="Extra sport:date or date pairs, comma-separated "
                             "(date-only assumes baseball_mlb).")
    args = parser.parse_args()

    if not db_store.promote_secrets_from_toml():
        raise SystemExit("SQL_* secrets not configured; cannot write to SQL.")
    if not warehouse._blob_base():
        raise SystemExit("PREDICTION_LOG_BLOB_URL not configured; nothing to read.")

    extra = set()
    for token in (d.strip() for d in args.dates.split(",") if d.strip()):
        if ":" in token:
            sport, _, date = token.partition(":")
            extra.add((sport, date))
        else:
            extra.add(("baseball_mlb", token))

    pairs = _relevant_pairs(extra)
    print(f"Scanning {len(pairs)} (sport, date) pair(s) from the Blob warehouse...")
    read, written = _migrate(pairs, args.dry_run)
    print(f"  {read} snapshot(s) read; {written} written to SQL"
          f"{' (dry run)' if args.dry_run else ''}.")
    if args.dry_run:
        return

    # Recompute CLV against the corrected closing snapshot (the _order bugfix).
    # Only TEAM markets (moneyline/spread/total) derive from the warehouse, so
    # reset only those. Player-prop CLV now comes from DraftKings via
    # backfill_dk_clv.py and persist_clv can't recompute it — a blanket
    # reset_clv() would wipe real DK-backfilled prop CLV and leave it blank.
    team_ids = [r.get("wager_id") for r in wagers.read_wagers()
                if r.get("bet_type") != "player_prop" and r.get("wager_id")]
    reset = wagers.reset_clv(wager_ids=team_ids)
    recomputed = wagers.persist_clv()
    print(f"CLV: cleared {reset} stale team-market row(s); recomputed "
          f"{recomputed} (player-prop CLV left to backfill_dk_clv.py).")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
