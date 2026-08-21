"""Archive-then-reset the app's forward-tracking + bet ledger for a clean-slate
relaunch (offline, Azure-SQL only).

Non-destructive by construction: every target table is dumped to ONE timestamped
JSON archive AND the archive is re-read + row-count-verified BEFORE any DELETE,
inside a single transaction — so a verify failure rolls back and deletes nothing,
and a completed reset is fully reversible from the archive file. An
``app_data_epoch`` marker is written to app_settings recording the reset
(timestamp, archive filename, per-table counts).

Target tables (owner-confirmed 2026-08-21):
    prediction_log         props forward-tracking
    market_prediction_log  team-market forward-tracking
    wagers                 the bet ledger (My Bets)
    bankroll_ledger        cleared -> derived balance becomes 0 (re-seed fresh)

PRESERVES everything else — calibration fits (calibration/*.json,
recalibration_*), resolved facts (mlb_game/gamelogs/statcast), the odds corpus
(odds_snapshot/odds_line), and all OTHER app_settings keys (Kelly knobs, etc.).

⚠ Run with the app STOPPED (relaunch context) so nothing writes between the
archive snapshot and the delete.

    python archive_app_data.py                 # dry-run: show row counts
    python archive_app_data.py --apply --yes   # archive -> verify -> delete -> mark
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Table-object names on db_store, in delete order (all independent — no FKs
# between them — but listed most-derived-first for readability).
_TABLES = ["prediction_log", "market_prediction_log", "wagers", "bankroll_ledger"]

_EPOCH_KEY = "app_data_epoch"


def _archive_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "warehouse", "archive")


def _engine():
    import db_store
    if not db_store.promote_secrets_from_toml() and not db_store.enabled():
        print("Azure SQL is not configured (SQL_* secrets). This tool is SQL-only.")
        sys.exit(2)
    return db_store.get_engine()


def _fsync_dir(path):
    """Best-effort directory fsync so the new archive file's directory entry is
    durable before the destructive commit. No-op where unsupported (e.g. Windows,
    which can't fsync a directory handle)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def _mark_epoch(conn, db_store, ts, archive_name, counts):
    """Upsert the app_data_epoch marker (leaves all other app_settings keys)."""
    from sqlalchemy import select, insert, update
    s = db_store.app_settings
    val = json.dumps({"reset_at": ts, "archive": archive_name, "counts": counts})
    exists = conn.execute(
        select(s.c.id).where(s.c.setting_key == _EPOCH_KEY)).first()
    if exists:
        conn.execute(update(s).where(s.c.setting_key == _EPOCH_KEY)
                     .values(setting_value=val, updated_at=ts))
    else:
        conn.execute(insert(s).values(setting_key=_EPOCH_KEY,
                                      setting_value=val, updated_at=ts))


def archive_and_reset(apply, yes, tables=None, archive_dir=None, now=None):
    """Archive + clear the target tables. Dry-run unless apply; apply also needs
    yes. ``now`` (a UTC datetime) is injectable for deterministic tests."""
    import db_store
    from sqlalchemy import select, func, delete
    eng = _engine()
    names = tables or _TABLES
    tbl = {n: getattr(db_store, n) for n in names}

    with eng.connect() as c:
        counts = {n: c.execute(select(func.count()).select_from(t)).scalar()
                  for n, t in tbl.items()}
    print("\n=== archive + reset app data ===")
    for n in names:
        print(f"  {n:24} {counts[n]:>8} rows")
    total = sum(counts.values())
    if total == 0:
        print("  all target tables already empty -- nothing to do.")
        return
    if not apply:
        print("\n  dry-run only. Re-run with --apply --yes to archive + reset.")
        return
    if not yes:
        print("\n  --apply requires --yes (destructive). Aborting.")
        sys.exit(3)

    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    arch_dir = archive_dir or _archive_dir()
    os.makedirs(arch_dir, exist_ok=True)
    path = os.path.join(arch_dir, f"app_data_epoch_{ts}.json")

    try:
        with eng.begin() as c:
            # Archive-select + delete in ONE transaction (consistent read) and
            # bound the delete by the archived max id (below) so that, even if the
            # app were still writing, deleted ⊆ archived — never an un-backed-up row.
            data = {n: [dict(r._mapping) for r in c.execute(select(t)).all()]
                    for n, t in tbl.items()}
            arch_counts = {n: len(data[n]) for n in names}
            payload = {"archived_at": ts, "counts": arch_counts, "tables": data}
            # Write + fsync the archive to DISK (not just the OS page cache) before
            # the durable remote DELETE, so a crash between the two can't lose the
            # backup. allow_nan=False fails closed on NaN/Inf (invalid JSON) rather
            # than writing an unrestorable archive.
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            _fsync_dir(arch_dir)
            # Re-read from disk and verify BOTH row counts AND full content round-
            # trip faithfully before deleting anything. Any mismatch raises -> the
            # transaction rolls back -> nothing deleted.
            with open(path, "r", encoding="utf-8") as f:
                reread = json.load(f)
            for n in names:
                if len(reread["tables"].get(n, [])) != arch_counts[n]:
                    raise RuntimeError(
                        f"archive count mismatch for {n} "
                        f"({len(reread['tables'].get(n, []))} != {arch_counts[n]})")
            if reread["tables"] != json.loads(json.dumps(data, default=str,
                                                         allow_nan=False)):
                raise RuntimeError("archive content did not round-trip faithfully")
            # Delete only rows we archived: id <= the archived max id per table.
            # IDENTITY ids are monotonic, so a row inserted after the snapshot has a
            # larger id and survives; id<=max also avoids the SQL Server ~2100-param
            # IN() limit on large tables (e.g. prediction_log).
            for n, t in tbl.items():
                if not data[n]:
                    continue
                max_id = max(r["id"] for r in data[n])
                c.execute(delete(t).where(t.c.id <= max_id))
            _mark_epoch(c, db_store, ts, os.path.basename(path), arch_counts)
    except Exception as e:
        print(f"  ABORT ({type(e).__name__}): {e}. "
              f"Transaction rolled back — nothing was deleted.")
        sys.exit(4)

    print(f"  archived {total} rows -> {path}")
    print(f"  ok: cleared {total} rows; bankroll balance now 0; epoch marker "
          f"'{_EPOCH_KEY}' written. Reversible from the archive above.")


def main():
    p = argparse.ArgumentParser(
        description="Archive-then-reset the app's forward-tracking + bet ledger.")
    p.add_argument("--tables", default=None,
                   help="Comma-separated subset of "
                        f"{','.join(_TABLES)} (default: all).")
    p.add_argument("--apply", action="store_true", help="Write (default is dry-run).")
    p.add_argument("--yes", action="store_true",
                   help="Required with --apply (destructive; archive-first).")
    args = p.parse_args()
    tables = None
    if args.tables:
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]
        bad = [t for t in tables if t not in _TABLES]
        if bad:
            p.error(f"unknown table(s): {', '.join(bad)}. Choose from {_TABLES}.")
    archive_and_reset(args.apply, args.yes, tables=tables)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
