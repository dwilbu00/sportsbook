"""
Phase 3 of the precise-odds pipeline: BACK UP → PURGE → RELOAD the Azure odds
warehouse (odds_snapshot + odds_line) from the precise-backfill parquet corpus.

WHY
---
The precise backfill (backfill_precise.py) captures game-relative snapshots at rigid
windows (−12h / −4h / close) for exactly DK/FD/Pinnacle. We are REPLACING the
warehouse's odds with this corrected corpus so every reader sees uniform snapshot
timing and only the books we actually use — no legacy mixed-timing live captures or
40-book noise diluting the rigid windows.

SAFETY (this is a DESTRUCTIVE prod operation)
  * --backup  : dump the FULL odds_snapshot + odds_line (ALL sports) to parquet FIRST.
                Free, read-only, and the restore point for everything below. NBA/NFL
                are included so their odds can be re-derived / translated later.
  * --purge   : empty odds_line + odds_snapshot (all sports — Doug is repopulating
                NBA/NFL too). Refuses to run unless a backup exists (or --force).
  * --load    : reload MLB from the precise parquet via warehouse.capture_event_odds
                (reuses the exact best-price/consensus extraction + kind derivation,
                source='backfill_precise') so existing readers work unchanged.
  * --verify  : parity-check reloaded closing lines against the parquet.

Nothing is destructive without an explicit flag; --backup and --verify never write to
the warehouse. Run order: --backup → (--purge) → --load → --verify.
"""
import argparse
import os
from datetime import datetime, timezone

BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "odds_backfill", "warehouse_backup")

_TABLES = ("odds_snapshot", "odds_line")


def _engine():
    import db_store
    db_store.promote_secrets_from_toml()
    return db_store.get_engine()


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def snapshot_counts(eng):
    """Row counts to scope the purge: odds_snapshot per sport + odds_line total."""
    from sqlalchemy import text
    out = {"by_sport": {}, "snapshot_total": 0, "line_total": 0}
    with eng.connect() as c:
        for sport, n in c.execute(text(
                "SELECT sport, COUNT(*) FROM odds_snapshot GROUP BY sport")).all():
            out["by_sport"][sport] = int(n)
            out["snapshot_total"] += int(n)
        out["line_total"] = int(
            c.execute(text("SELECT COUNT(*) FROM odds_line")).scalar() or 0)
    return out


def backup(eng, chunksize=200_000):
    """Dump both tables (all sports) to parquet under BACKUP_DIR/<stamp>/. odds_line
    is streamed in chunks (it is the large table) via a pyarrow ParquetWriter."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    stamp = _stamp()
    dest = os.path.join(BACKUP_DIR, stamp)
    os.makedirs(dest, exist_ok=True)
    counts = snapshot_counts(eng)
    print(f"=== BACKUP → {dest} ===")
    print(f"  odds_snapshot: {counts['snapshot_total']} rows "
          f"({', '.join(f'{k}={v}' for k, v in sorted(counts['by_sport'].items()))})")
    print(f"  odds_line:     {counts['line_total']} rows")

    # odds_snapshot (small) — whole-table read.
    snap = pd.read_sql("SELECT * FROM odds_snapshot", eng)
    snap_path = os.path.join(dest, "odds_snapshot.parquet")
    snap.to_parquet(snap_path, index=False)
    print(f"  wrote {snap_path} ({len(snap)} rows)")

    # odds_line (large) — chunked stream to a single parquet file.
    line_path = os.path.join(dest, "odds_line.parquet")
    writer = None
    written = 0
    for chunk in pd.read_sql("SELECT * FROM odds_line", eng, chunksize=chunksize):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(line_path, table.schema)
        writer.write_table(table)
        written += len(chunk)
        print(f"    odds_line … {written}/{counts['line_total']} rows")
    if writer is not None:
        writer.close()
    else:                                   # empty table → still emit a valid file
        pd.DataFrame().to_parquet(line_path, index=False)
    print(f"  wrote {line_path} ({written} rows)")

    # Manifest (also the sentinel --purge checks for).
    import json
    with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"stamp": stamp, "counts": counts,
                   "snapshot_rows": len(snap), "line_rows": written}, f, indent=2)
    # Pointer to the latest backup.
    with open(os.path.join(BACKUP_DIR, "LATEST.txt"), "w", encoding="utf-8") as f:
        f.write(stamp + "\n")
    print(f"=== BACKUP done. {len(snap)} snapshots + {written} lines → {dest} ===")
    return dest


def _latest_backup():
    p = os.path.join(BACKUP_DIR, "LATEST.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        stamp = f.read().strip()
    d = os.path.join(BACKUP_DIR, stamp)
    return d if os.path.isdir(d) else None


def main():
    p = argparse.ArgumentParser(description="Phase 3: back up / purge / reload the "
                                            "Azure odds warehouse from precise parquet.")
    p.add_argument("--backup", action="store_true",
                   help="Dump odds_snapshot + odds_line (all sports) to parquet. "
                        "Free, read-only. Do this FIRST.")
    p.add_argument("--counts", action="store_true",
                   help="Print current warehouse odds row counts and exit (free).")
    p.add_argument("--chunksize", type=int, default=200_000,
                   help="odds_line read chunk size for the backup stream.")
    args = p.parse_args()

    eng = _engine()
    if args.counts:
        c = snapshot_counts(eng)
        print(f"odds_snapshot: {c['snapshot_total']} "
              f"({', '.join(f'{k}={v}' for k, v in sorted(c['by_sport'].items()))})")
        print(f"odds_line:     {c['line_total']}")
        return
    if args.backup:
        backup(eng, chunksize=args.chunksize)
        return
    p.print_help()


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
