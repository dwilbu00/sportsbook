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

# Snapshot role → the `source` value stored on odds_snapshot. `source` becomes the
# window selector (Doug 2026-09-01): a reader picks a window with WHERE source=...
ROLE_TO_SOURCE = {"early_12h": "early_12h", "early_4h": "early_4h", "close": "closing"}

# Per-kind market strings recorded on the snapshot meta (drives _kind_for_markets
# identity + is human-auditable). Full and F5 come from the SAME cached team payload.
_KIND_MARKETS = {
    "team": "h2h,spreads,totals",
    "first_five": "h2h_1st_5_innings,spreads_1st_5_innings,totals_1st_5_innings",
}


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

    # odds_line (large) — chunked stream to ONE parquet with a PINNED schema. Don't
    # infer the schema from chunk-1: odds_line has nullable string cols (player,
    # prop_key, direction, …) that are all-NULL for early (team-only) rows → pyarrow
    # infers `null` type, then a later chunk with real strings infers `string` and
    # ParquetWriter.write_table rejects the schema mismatch, aborting the backup
    # mid-stream. A fixed schema (from_pandas maps by NAME + casts null→declared) makes
    # every chunk conform. ORDER BY id makes chunking deterministic.
    line_schema = pa.schema([
        ("id", pa.int64()), ("snapshot_id", pa.int64()),
        ("bet_type", pa.string()), ("selection", pa.string()),
        ("point", pa.float64()), ("player", pa.string()),
        ("prop_key", pa.string()), ("direction", pa.string()),
        ("price", pa.int64()), ("implied_prob", pa.float64()),
        ("player_mlb_id", pa.string()), ("team_code", pa.string()),
        ("game_pk", pa.int64()), ("bookmaker", pa.string()),
        ("region", pa.string()),
    ])
    line_path = os.path.join(dest, "odds_line.parquet")
    writer = pq.ParquetWriter(line_path, line_schema)
    written = 0
    try:
        for chunk in pd.read_sql("SELECT * FROM odds_line ORDER BY id", eng,
                                 chunksize=chunksize):
            table = pa.Table.from_pandas(chunk, schema=line_schema,
                                         preserve_index=False)
            writer.write_table(table)
            written += len(chunk)
            print(f"    odds_line … {written}/{counts['line_total']} rows")
    finally:
        writer.close()
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
    """Return the latest backup dir ONLY if it holds a COMPLETE restore set (manifest
    + both parquet files) — not just an empty stamp dir — so the purge gate actually
    enforces a restorable backup."""
    p = os.path.join(BACKUP_DIR, "LATEST.txt")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        stamp = f.read().strip()
    d = os.path.join(BACKUP_DIR, stamp)
    required = ("manifest.json", "odds_snapshot.parquet", "odds_line.parquet")
    if os.path.isdir(d) and all(os.path.isfile(os.path.join(d, r)) for r in required):
        return d
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PURGE — empty odds_line + odds_snapshot (all sports). DESTRUCTIVE.
# ──────────────────────────────────────────────────────────────────────────────
def purge(eng, apply=False, force=False):
    """Empty odds_line then odds_snapshot (ALL sports — Doug is repopulating NBA/NFL
    too). Refuses to run without a completed --backup unless --force. On SQL Server,
    odds_snapshot can't be TRUNCATEd while odds_line's FK references it, so we
    truncate the (already-emptied) child, drop the FK, truncate the parent, and
    re-add the FK. On SQLite (local/tests) falls back to DELETE."""
    from sqlalchemy import text
    if not apply:
        c = snapshot_counts(eng)
        print(f"  [dry-run] would TRUNCATE odds_line ({c['line_total']} rows) + "
              f"odds_snapshot ({c['snapshot_total']} rows, ALL sports). "
              f"Re-run with --apply --yes.")
        return
    if not force and not _latest_backup():
        raise SystemExit("REFUSING to purge: no backup found. Run --backup first "
                         "(or --force to override).")
    dialect = eng.dialect.name
    print(f"  PURGE (dialect={dialect}) — emptying odds_line + odds_snapshot…")
    if dialect == "mssql":
        # DELETE, not TRUNCATE: TRUNCATE (and ALTER for the FK dance) needs DDL/ALTER
        # permission the app SQL login lacks (db_datawriter has only DELETE). Delete the
        # child (odds_line) in COMMITTED batches so the transaction log stays bounded on
        # the ~3.7M-row table and the Azure round-trip never times out, then the small
        # parent (odds_snapshot). Child-first respects the FK — no constraint drop.
        # Non-atomic but idempotent: re-running continues until both are empty (and the
        # --backup restore point exists regardless).
        deleted = 0
        while True:
            with eng.begin() as c:
                r = c.execute(text("DELETE TOP (100000) FROM odds_line"))
            n = r.rowcount or 0
            deleted += n
            if n:
                print(f"    … deleted {deleted:,} odds_line rows")
            if n == 0:
                break
        with eng.begin() as c:
            c.execute(text("DELETE FROM odds_snapshot"))
    else:  # sqlite / others: no TRUNCATE, but child-first DELETE works
        with eng.begin() as c:
            c.execute(text("DELETE FROM odds_line"))
            c.execute(text("DELETE FROM odds_snapshot"))
    after = snapshot_counts(eng)
    print(f"  PURGE done. Now: odds_snapshot={after['snapshot_total']}, "
          f"odds_line={after['line_total']}.")
    if after["snapshot_total"] or after["line_total"]:
        raise SystemExit(
            f"PURGE INCOMPLETE: odds_snapshot={after['snapshot_total']}, "
            f"odds_line={after['line_total']} still present — aborting before --load.")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD — reload MLB from the precise cache (per-book lines, source=role, per kind)
# ──────────────────────────────────────────────────────────────────────────────
def _kinds_for_group(group):
    return ["team", "first_five"] if group == "team" else ["props"]


def load(eng, sport, seasons, tier, books, date_from=None, date_to=None,
         apply=False, limit=0, progress_every=500, workers=12):
    """Reload one sport's odds from the precise cache. Drives off backfill_precise's
    PLAN (role explicit per offset), reads each cached payload (0 credits), emits
    PER-BOOK lines via ingest_multibook_cache._per_book_lines (byte-identical to live
    capture; F5 via the key-shim for MLB), and writes one snapshot per (event, kind,
    role) with source=ROLE_TO_SOURCE[role] through db_store.capture_odds_snapshot
    (write-once). Dry-run by default (counts only). Cached-only reads → spends nothing."""
    import backfill_precise as bp
    import ingest_multibook_cache as im
    import warehouse as wh
    import db_store
    from collections import Counter
    db_store.promote_secrets_from_toml()

    games, id_by_pk, _ = bp.enumerate_for_sport(
        sport, None, seasons=seasons, date_from=date_from, date_to=date_to,
        allow_api=False, verbose=True)
    # CRITICAL: for MLB the warehouse event-id resolution reads odds_snapshot, which
    # --purge TRUNCATEd before this runs → id_by_pk would be ~empty and the reload
    # would silently write nothing. Recover game_pk→event_id from the compiled parquet
    # (it persists BOTH), so the reload is self-contained on cache+parquet and immune
    # to the purge (also 0 credits, order-independent). Live-resolved ids win; parquet
    # fills the rest.
    import glob
    import pandas as _pd
    pmap = {}
    for _p in glob.glob(os.path.join(bp.PARQUET_DIR,
                                     f"{bp.SPORT_TAG}_precise_*.parquet")):
        _df = _pd.read_parquet(_p, columns=["game_pk", "event_id"]).drop_duplicates()
        for _pk, _eid in _df.itertuples(index=False):
            if _eid:
                pmap[_pk] = _eid
    id_by_pk = {**pmap, **id_by_pk}
    resolved = sum(1 for g in games if id_by_pk.get(g["game_pk"]))
    print(f"  event_id coverage: {resolved}/{len(games)} games mapped "
          f"({len(pmap)} from parquet).")
    if apply and games and resolved < 0.5 * len(games):
        raise SystemExit(
            f"ABORT: only {resolved}/{len(games)} games have an event_id — refusing "
            f"to write a near-empty reload. Run Phase-2 --compile (parquet) first.")
    specs = bp._group_specs(tier, books)
    gpk_cache, id_cache = {}, {}
    snaps = Counter()          # (kind, source) -> snapshots
    src_lines = Counter()      # source -> lines
    books_ct = Counter()
    written = skipped = errors = pending = empty = 0
    n = 0

    mode = "APPLY (writing)" if apply else "DRY-RUN (no writes)"
    print(f"\n=== LOAD precise → warehouse [{mode}] sport={bp.SPORT_KEY} "
          f"workers={workers} ===")

    # Build the work list: one cached payload per (game, group, offset); a worker emits
    # 1-2 kind-snapshots from it. Parallelised because the cost is one transaction +
    # commit round-trip per snapshot over Azure latency — serial that is ~55k×latency
    # for MLB. Threads overlap the network waits (each capture uses its own pooled conn).
    tasks = []
    for g in games:
        eid = id_by_pk.get(g["game_pk"])
        if not eid:
            continue
        for group, markets, offsets, regions in specs:
            if group == "props" and g["official_date"] < bp.PROPS_MIN_DATE:
                continue
            markets_csv = ",".join(markets)
            for hours_before, role in offsets:
                ts = bp._offset_ts(g["commence"], hours_before)
                tasks.append((g, eid, group, markets_csv, regions, ts, role))
    if limit:
        tasks = tasks[:limit]

    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    st = {"n": 0, "written": 0, "skipped": 0, "errors": 0, "pending": 0, "empty": 0}

    def _work(task):
        g, eid, group, markets_csv, regions, ts, role = task
        if not bp.is_historical_event_cached(bp.SPORT_KEY, eid, ts, regions=regions,
                                             markets=markets_csv, bookmakers=books):
            with lock:
                st["pending"] += 1
            return
        data, snap = bp.get_historical_event_odds(
            api_key=None, sport=bp.SPORT_KEY, event_id=eid, date=ts,
            regions=regions, markets=markets_csv, bookmakers=books)
        if not data:
            with lock:
                st["empty"] += 1
            return
        source = ROLE_TO_SOURCE[role]
        snapshot_hour = wh._hour_bucket(ts)   # rigid per-role hour (uq safe)
        commence = data.get("commence_time") or g["commence"]
        for kind in _kinds_for_group(group):
            lines = im._per_book_lines(data, kind)
            if not lines:
                continue
            meta = {
                "sport": bp.SPORT_KEY, "game_date": commence[:10],
                "event_id": eid, "kind": kind,
                "snapshot_hour": snapshot_hour, "captured_at": snap,
                "commence_time": commence,
                "home": data.get("home_team"), "away": data.get("away_team"),
                "regions": regions, "markets": _KIND_MARKETS.get(kind, markets_csv),
                "bookmakers": ",".join(books), "source": source,
            }
            # enrich caches are shared; GIL-safe dict ops, benign double-compute.
            meta, lines = im._enrich_lines_fast(
                bp.SPORT_KEY, meta, lines, gpk_cache, id_cache)
            ok = None
            if apply:
                try:
                    ok = db_store.capture_odds_snapshot(meta, lines)
                except Exception as exc:
                    with lock:
                        st["errors"] += 1
                        if st["errors"] <= 5:
                            print(f"  [err] {eid} {kind}/{source}: "
                                  f"{type(exc).__name__} ({exc})")
                    continue
            with lock:
                snaps[(kind, source)] += 1
                src_lines[source] += len(lines)
                for ln in lines:
                    books_ct[ln.get("bookmaker")] += 1
                if apply:
                    st["written"] += 1 if ok else 0
                    st["skipped"] += 0 if ok else 1
                st["n"] += 1
                nn = st["n"]
            if progress_every and nn % progress_every == 0:
                print(f"  …{nn:,} snapshots ({'written ' + format(st['written'], ',') if apply else 'dry-run'})")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        list(ex.map(_work, tasks))
    pending, empty = st["pending"], st["empty"]
    written, skipped, errors, n = st["written"], st["skipped"], st["errors"], st["n"]

    print(f"\n  snapshots {'written' if apply else 'planned'}: {n:,}")
    for (kind, source) in sorted(snaps):
        print(f"    {kind:11} source={source:9} : {snaps[(kind, source)]:,}")
    print(f"  lines by source: "
          f"{', '.join(f'{s}={src_lines[s]:,}' for s in sorted(src_lines))}")
    print(f"  books: {', '.join(f'{b}={books_ct[b]:,}' for b in sorted(books_ct))}")
    print(f"  pending(uncached)={pending:,}  empty={empty:,}"
          + (f"  written={written:,} skipped(dup)={skipped:,} errors={errors:,}"
             if apply else ""))
    if apply and skipped:
        print(f"  [warn] {skipped:,} write-once COLLISIONS (dup uq) — a window may "
              f"have been dropped; investigate if nonzero (expected 0).")
    if not apply:
        print("  [dry-run] nothing written. Re-run with --apply --yes.")


# ──────────────────────────────────────────────────────────────────────────────
# VERIFY — parity-check the reloaded warehouse against the precise parquet
# ──────────────────────────────────────────────────────────────────────────────
def verify(eng, sport, books, sample=2000):
    """Post-load parity check for one sport. (1) Warehouse composition: snapshot counts
    by (kind, source) + book. (2) Price parity: closing team moneyline + totals matched
    to the parquet on (event_id, book, selection, point). Reports match/mismatch/missing
    so a reload can be trusted before we rely on it. Read-only."""
    import pandas as pd
    import glob
    from sqlalchemy import text
    import backfill_precise as bp
    bp.configure_sport(sport)
    sport_key, tag = bp.SPORT_KEY, bp.SPORT_TAG
    print(f"\n=== VERIFY (read-only) sport={sport_key} ===")
    with eng.connect() as c:
        print("  Warehouse composition (kind, source):")
        for kind, src, n in c.execute(text(
                "SELECT kind, source, COUNT(*) FROM odds_snapshot "
                "WHERE sport=:sp GROUP BY kind, source ORDER BY kind, source"),
                {"sp": sport_key}).all():
            print(f"    {kind:11} {str(src):9} : {n:,}")
        print("  odds_line by bookmaker (source=closing):")
        for bk, n in c.execute(text(
                "SELECT l.bookmaker, COUNT(*) FROM odds_line l "
                "JOIN odds_snapshot s ON l.snapshot_id=s.id "
                "WHERE s.sport=:sp AND s.source='closing' "
                "GROUP BY l.bookmaker ORDER BY l.bookmaker"), {"sp": sport_key}).all():
            print(f"    {str(bk):12} : {n:,}")

    # Price parity on closing team moneyline + totals (glob this sport's team parquet).
    _MK = {"h2h": "moneyline", "totals": "total"}
    par = {}   # (event_id, book, bet_type, selection, point) -> price
    pdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "odds_backfill", "parquet")
    for p in glob.glob(os.path.join(pdir, f"{tag}_precise_team_*.parquet")):
        df = pd.read_parquet(p)
        df = df[(df["role"] == "close") & (df["market"].isin(_MK))]
        for r in df.itertuples(index=False):
            pt = None if pd.isna(r.point) else float(r.point)
            par[(r.event_id, r.book, _MK[r.market], r.outcome, pt)] = int(r.price)
    if not par:
        print(f"  [parity] no {tag} closing team parquet — run --compile first.")
        return
    match = mism = miss = 0
    seen = set()
    examples = []
    with eng.connect() as c:
        rows = c.execute(text(
            "SELECT s.event_id, l.bookmaker, l.bet_type, l.selection, l.point, l.price "
            "FROM odds_line l JOIN odds_snapshot s ON l.snapshot_id=s.id "
            "WHERE s.sport=:sp AND s.source='closing' AND s.kind='team' "
            "AND l.bet_type IN ('moneyline','total')"), {"sp": sport_key}).all()
    for eid, bk, bt, sel, pt, price in rows:
        key = (eid, bk, bt, sel, None if pt is None else float(pt))
        if key not in par:
            miss += 1
            continue
        seen.add(key)
        if int(price) == par[key]:
            match += 1
        else:
            mism += 1
            if len(examples) < 8:
                examples.append(f"{eid[:8]} {bk} {bt} {sel} pt={pt}: "
                                f"wh={price} vs parquet={par[key]}")
    tot = match + mism + miss
    # Coverage the OTHER direction: parquet closing lines missing from the warehouse
    # (an empty/incomplete reload or a write-once drop would show up here).
    missing_from_wh = sum(1 for k in par if k not in seen)
    # Purge-completeness: after a reload the ONLY snapshots for this sport must be the
    # three role sources. Any legacy source surviving = the purge was skipped/incomplete
    # (verify's parity is scoped to source='closing', so it would otherwise MISS a
    # warehouse still carrying the mixed-timing/40-book legacy corpus the reload exists
    # to eliminate).
    with eng.connect() as c:
        legacy = c.execute(text(
            "SELECT COUNT(*) FROM odds_snapshot WHERE sport=:sp AND "
            "(source IS NULL OR source NOT IN ('early_12h','early_4h','closing'))"),
            {"sp": sport_key}).scalar() or 0
    print(f"\n  PRICE PARITY (closing team moneyline+total): {match:,} match, "
          f"{mism:,} mismatch, {miss:,} warehouse-rows-not-in-parquet "
          f"(of {tot:,} warehouse rows); {missing_from_wh:,} parquet lines "
          f"MISSING from warehouse (of {len(par):,}); {legacy:,} legacy/non-role "
          f"snapshots surviving.")
    for e in examples:
        print(f"    MISMATCH {e}")
    # OK requires: warehouse non-empty, zero price mismatches, near-complete
    # parquet→warehouse coverage (tiny gap tolerated for empty/lag games), and ZERO
    # legacy snapshots (purge actually happened).
    cover_ok = missing_from_wh <= 0.02 * max(len(par), 1)
    ok = (tot > 0 and mism == 0 and cover_ok and legacy == 0)
    if ok:
        print("  === VERIFY OK ===")
    else:
        why = []
        if tot == 0:
            why.append("warehouse EMPTY (no closing team rows)")
        if mism:
            why.append(f"{mism} price mismatches")
        if not cover_ok:
            why.append(f"{missing_from_wh}/{len(par)} parquet lines missing from wh")
        if legacy:
            why.append(f"{legacy} legacy/non-role snapshots survive (purge incomplete)")
        print(f"  === VERIFY FAILED: {'; '.join(why)} ===")


def main():
    p = argparse.ArgumentParser(description="Phase 3: back up / purge / reload the "
                                            "Azure odds warehouse from precise parquet.")
    p.add_argument("--backup", action="store_true",
                   help="Dump odds_snapshot + odds_line (all sports) to parquet. "
                        "Free, read-only. Do this FIRST.")
    p.add_argument("--counts", action="store_true",
                   help="Print current warehouse odds row counts and exit (free).")
    p.add_argument("--purge", action="store_true",
                   help="DESTRUCTIVE: empty odds_line + odds_snapshot (all sports). "
                        "Dry-run unless --apply --yes; refuses without a backup.")
    p.add_argument("--load", action="store_true",
                   help="Reload MLB from the precise cache (per-book, source=role). "
                        "Dry-run unless --apply --yes. Spends nothing (cached reads).")
    p.add_argument("--verify", action="store_true",
                   help="Read-only parity check: warehouse composition + closing "
                        "team price parity vs the parquet. Run after --load --apply.")
    p.add_argument("--apply", action="store_true",
                   help="Actually write/destroy (with --purge/--load). Needs --yes.")
    p.add_argument("--yes", action="store_true", help="Double-confirm with --apply.")
    p.add_argument("--force", action="store_true",
                   help="Allow --purge without a backup present (not recommended).")
    p.add_argument("--sport", choices=["mlb", "nba", "nfl"], default="mlb",
                   help="Sport to --load/--verify (default mlb). --backup/--purge are "
                        "all-sports.")
    p.add_argument("--seasons", default="2024,2025,2026",
                   help="MLB reload seasons (default 2024,2025,2026).")
    p.add_argument("--date-from", default=None,
                   help="NBA/NFL reload: enumerate from this date (default props floor).")
    p.add_argument("--date-to", default=None,
                   help="NBA/NFL reload: enumerate through this date (default today).")
    p.add_argument("--tier", choices=["team", "props", "all"], default="all")
    p.add_argument("--books", default="draftkings,fanduel,pinnacle")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap snapshots processed (0=all); for a fast dry-run sample.")
    p.add_argument("--workers", type=int, default=12,
                   help="Concurrent write workers for --load (default 12). Overlaps the "
                        "per-snapshot Azure round-trip; each uses its own pooled conn.")
    p.add_argument("--chunksize", type=int, default=200_000,
                   help="odds_line read chunk size for the backup stream.")
    args = p.parse_args()

    apply = bool(args.apply and args.yes)
    if (args.apply and not args.yes):
        print("--apply requires --yes (double-confirm). Nothing done.")
        return
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    books = [b.strip() for b in args.books.split(",") if b.strip()]

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
    if args.purge:
        purge(eng, apply=apply, force=args.force)
        return
    if args.load:
        load(eng, args.sport, seasons, args.tier, books,
             date_from=args.date_from, date_to=args.date_to,
             apply=apply, limit=args.limit, workers=args.workers)
        return
    if args.verify:
        verify(eng, args.sport, books)
        return
    p.print_help()


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
