"""Warehouse odds-provenance maintenance (offline, Azure-SQL only).

Two operations, BOTH dry-run by default — writing requires --apply (and the
destructive prune also requires --yes):

  --retag       One-time backfill of odds_snapshot.source for rows where it is
                NULL (new captures already stamp it at write time):
                    kind = 'seed'          -> 'seed'
                    event_id LIKE 'sbr-%'  -> 'sbr'
                    otherwise              -> 'live'
                Non-destructive UPDATE. Run once after the schema ALTER.

  --prune-seed  DELETE the bulk season-seed snapshots (the ~18h-pre-pitch api
                seed) for a scope so only the intended early/close lines remain.
                ARCHIVES the deleted snapshots + their odds_line rows to a
                timestamped JSON first (reversible), THEN deletes. Scope with
                --sport (default baseball_mlb) and --years (default 2024,2025).
                ⚠ Run this ONLY AFTER the replacement early+close lines are
                backfilled and verified, so there is never a coverage gap.

Examples:
    python odds_provenance.py --retag                 # dry-run: show counts
    python odds_provenance.py --retag --apply         # write source for NULL rows
    python odds_provenance.py --prune-seed --sport mlb --years 2024,2025          # dry-run
    python odds_provenance.py --prune-seed --sport mlb --years 2024,2025 --apply --yes
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

_SPORT_KEY = {"mlb": "baseball_mlb", "nfl": "americanfootball_nfl",
              "nba": "basketball_nba", "nhl": "icehockey_nhl"}

# Legacy plain-'backfill' rows are split into early/close by the capture CLOCK
# TIME, not a duration-before-commence: the backfill's EARLY snapshot is always
# taken at the fixed --early-time (default 11:00Z) and no game's CLOSE is ever
# captured near 11:00Z (all sports tip >=15:00Z; NFL London 13:30Z), so this is
# exact regardless of commence — it avoids the duration-threshold ambiguity (a
# stale close captured hours early vs a rare early-day game's early snapshot).
# Must match backfill_historical_odds.DEFAULT_EARLY_TIME (11:00Z). New captures
# are tagged role-explicitly at write, so this only reclassifies pre-refinement rows.
EARLY_TIME_HOUR = 11         # UTC hour of the fixed early snapshot
EARLY_WINDOW_HOURS = 1.5     # +/- window around it (covers at-or-before snapshots)


def _capture_role(captured):
    """'early' if captured_at is near the fixed early-time clock hour, 'close' if
    clearly not, or None if unparseable (left as plain 'backfill')."""
    try:
        from datetime import datetime as _dt
        cap = _dt.fromisoformat(str(captured).replace("Z", "+00:00"))
        hod = cap.hour + cap.minute / 60.0
        return "early" if abs(hod - EARLY_TIME_HOUR) <= EARLY_WINDOW_HOURS else "close"
    except Exception:
        return None


def _chunks(seq, n=1000):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _engine():
    import db_store
    if not db_store.promote_secrets_from_toml() and not db_store.enabled():
        print("Azure SQL is not configured (SQL_* secrets). This tool is SQL-only.")
        sys.exit(2)
    return db_store.get_engine()


def _retag(apply):
    """(1) Backfill source for NULL-source rows (seed/sbr/live) and (2) split any
    legacy plain-'backfill' rows into backfill_early/backfill_close by capture
    timing. Idempotent. Prints a plan; writes only with --apply."""
    import db_store
    from sqlalchemy import select, func, update
    t = db_store.odds_snapshot
    eng = _engine()
    with eng.connect() as c:
        null_total = c.execute(select(func.count()).select_from(t)
                               .where(t.c.source.is_(None))).scalar()
        seed = c.execute(select(func.count()).select_from(t)
                         .where(t.c.source.is_(None) & (t.c.kind == "seed"))).scalar()
        sbr = c.execute(select(func.count()).select_from(t)
                        .where(t.c.source.is_(None) & (t.c.kind != "seed")
                               & t.c.event_id.like("sbr-%"))).scalar()
        live = null_total - seed - sbr
        # Legacy rows tagged plain 'backfill' (pre role-explicit refinement) — split
        # by the capture CLOCK time (early snapshot ~= the fixed early-time; close
        # ~= commence). Unparseable stays plain 'backfill' (never mislabeled).
        bf = c.execute(select(t.c.id, t.c.captured_at)
                       .where(t.c.source == "backfill")).all()
    close_ids, early_ids = [], []
    for rid, cap in bf:
        role = _capture_role(cap)
        if role is None:
            continue
        (early_ids if role == "early" else close_ids).append(rid)
    print(f"\n=== retag ===")
    print(f"  NULL source ({null_total}) -> seed={seed}, sbr={sbr}, live={live}")
    print(f"  plain 'backfill' ({len(bf)}) -> close={len(close_ids)}, "
          f"early={len(early_ids)}, unchanged={len(bf) - len(close_ids) - len(early_ids)}")
    if not apply:
        print("\n  dry-run only. Re-run with --apply to write.")
        return
    with eng.begin() as c:
        # NULL rows: tag seed + sbr first, remainder -> live.
        c.execute(update(t).where(t.c.source.is_(None) & (t.c.kind == "seed"))
                  .values(source="seed"))
        c.execute(update(t).where(t.c.source.is_(None) & t.c.event_id.like("sbr-%"))
                  .values(source="sbr"))
        c.execute(update(t).where(t.c.source.is_(None)).values(source="live"))
        # Split legacy 'backfill' by role (batched for the SQL Server IN() limit).
        for chunk in _chunks(close_ids):
            c.execute(update(t).where(t.c.id.in_(chunk)).values(source="backfill_close"))
        for chunk in _chunks(early_ids):
            c.execute(update(t).where(t.c.id.in_(chunk)).values(source="backfill_early"))
    print(f"  ✓ tagged NULL rows + reclassified {len(close_ids) + len(early_ids)} "
          f"'backfill' rows (close={len(close_ids)}, early={len(early_ids)}).")


def _prune_seed(sport_key, years, apply, yes):
    """Archive + delete kind='seed' snapshots for a sport. ``years`` is a list of
    year prefixes to scope, or None/empty = ALL years' seed for the sport."""
    import db_store
    from sqlalchemy import select, func, or_, delete
    t, ln = db_store.odds_snapshot, db_store.odds_line
    eng = _engine()
    scope = (t.c.sport == sport_key) & (t.c.kind == "seed")
    if years:
        scope = scope & or_(*[t.c.game_date.like(f"{y}%") for y in years])
    with eng.connect() as c:
        snap_ids = [r[0] for r in c.execute(select(t.c.id).where(scope)).all()]
        n_snap = len(snap_ids)
        n_line = (c.execute(select(func.count()).select_from(ln)
                            .where(ln.c.snapshot_id.in_(snap_ids))).scalar()
                  if snap_ids else 0)
    print(f"\n=== prune-seed: {sport_key} kind='seed' "
          f"years={','.join(years) if years else 'ALL'} ===")
    print(f"  snapshots to delete: {n_snap}   odds_line rows to delete: {n_line}")
    if n_snap == 0:
        print("  nothing matches — done.")
        return
    if not apply:
        print("\n  dry-run only. Re-run with --apply --yes to archive + delete.")
        return
    if not yes:
        print("\n  --apply requires --yes (destructive). Aborting.")
        sys.exit(3)
    # Archive first (reversible): full snapshot + line rows to a timestamped file.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "warehouse", "archive")
    os.makedirs(arch_dir, exist_ok=True)
    path = os.path.join(arch_dir, f"pruned_seed_{sport_key}_{ts}.json")
    with eng.connect() as c:
        snaps = [dict(r._mapping) for r in c.execute(select(t).where(scope)).all()]
        lines = [dict(r._mapping) for r in c.execute(
            select(ln).where(ln.c.snapshot_id.in_(snap_ids))).all()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sport": sport_key, "years": years, "archived_at": ts,
                   "snapshots": snaps, "lines": lines}, f, indent=2, default=str)
    print(f"  archived {len(snaps)} snapshots + {len(lines)} lines -> {path}")
    # Delete lines first (explicit; not relying on FK cascade for portability),
    # then the snapshots.
    with eng.begin() as c:
        c.execute(delete(ln).where(ln.c.snapshot_id.in_(snap_ids)))
        c.execute(delete(t).where(scope))
    print(f"  ✓ deleted {n_snap} seed snapshots + {n_line} lines. "
          f"(reversible from the archive above.)")


def main():
    p = argparse.ArgumentParser(description="Warehouse odds-provenance maintenance")
    p.add_argument("--retag", action="store_true",
                   help="Backfill source for NULL-source rows (seed/sbr/live).")
    p.add_argument("--prune-seed", action="store_true",
                   help="Archive + delete kind='seed' snapshots for a scope.")
    p.add_argument("--sport", default="mlb", choices=list(_SPORT_KEY.keys()))
    p.add_argument("--years", default="2024,2025",
                   help="Comma-separated seasons for --prune-seed (default 2024,2025), "
                        "or 'all' to prune every year's seed for the sport.")
    p.add_argument("--apply", action="store_true", help="Write (default is dry-run).")
    p.add_argument("--yes", action="store_true",
                   help="Required with --apply for the destructive --prune-seed.")
    args = p.parse_args()
    if not (args.retag or args.prune_seed):
        p.error("choose --retag and/or --prune-seed")
    if args.retag:
        _retag(args.apply)
    if args.prune_seed:
        years = (None if args.years.strip().lower() == "all"
                 else [y.strip() for y in args.years.split(",") if y.strip()])
        _prune_seed(_SPORT_KEY[args.sport], years, args.apply, args.yes)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
