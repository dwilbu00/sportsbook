"""Warehouse odds-provenance maintenance (offline, Azure-SQL only).

Two operations, BOTH dry-run by default — writing requires --apply (and the
destructive prune also requires --yes):

  --retag       Tag odds_snapshot.source for rows where it is NULL (rows written
                before the source column existed; new captures stamp it at write).
                Classification, in order:
                    kind = 'seed'             -> 'seed'
                    event_id LIKE 'sbr-%'     -> 'sbr'
                    game_date >= --live-since -> 'live'
                    game_date <  --live-since -> 'backfill_close'
                WITHOUT --live-since the ambiguous (non-seed/non-sbr) rows are
                LEFT NULL -- never guessed as 'live'. The pre-column population
                MIXES genuine live captures with historical backfill/reload
                closes, so "everything not seed/sbr = live" mislabels the whole
                historical corpus. --live-since is the season boundary: the
                earliest game_date the app captured odds in REAL TIME. Scope with
                --sport / --years (retag defaults to ALL sports + ALL years).
                Prints a sport x year x kind composition breakdown before writing.

  --prune-seed  DELETE the bulk season-seed snapshots (the ~18h-pre-pitch api
                seed) for a scope so only the intended early+close lines remain.
                ARCHIVES the deleted snapshots + their odds_line rows to a
                timestamped JSON first (reversible), THEN deletes. Scope with
                --sport (required) and --years (default 2024,2025).
                Run this ONLY AFTER the replacement early+close lines are
                backfilled and verified, so there is never a coverage gap.

Examples:
    python odds_provenance.py --retag --sport mlb                       # dry-run
    python odds_provenance.py --retag --sport mlb --live-since 2026-01-01 --apply
    python odds_provenance.py --prune-seed --sport mlb --years all --apply --yes
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

_SPORT_KEY = {"mlb": "baseball_mlb", "nfl": "americanfootball_nfl",
              "nba": "basketball_nba", "nhl": "icehockey_nhl"}

def _chunks(seq, n=1000):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _engine():
    import db_store
    if not db_store.promote_secrets_from_toml() and not db_store.enabled():
        print("Azure SQL is not configured (SQL_* secrets). This tool is SQL-only.")
        sys.exit(2)
    return db_store.get_engine()


def _classify(kind, event_id, game_date, live_since, all_backfill=False):
    """Provenance for one NULL-source row, or None if it must be left untagged.
    Positive rules first (seed by kind, sbr by id prefix); the live-vs-backfill
    split is a game_date season gate, NEVER classification-by-elimination — the
    pre-column population mixes real-time live captures with historical backfill
    closes, so 'everything not seed/sbr = live' would mislabel the whole corpus.
    ``all_backfill`` = the sport was never live-captured, so every non-seed/non-sbr
    row is backfill_close (used for other sports with no real-time capture)."""
    if kind == "seed":
        return "seed"
    if (event_id or "").startswith("sbr-"):
        return "sbr"
    if all_backfill:
        return "backfill_close"
    if live_since is None:
        return None  # ambiguous without a boundary → leave NULL, never guess live
    return "live" if (game_date or "") >= live_since else "backfill_close"


def _retag(apply, sport_key=None, years=None, live_since=None, all_backfill=False):
    """Tag source for NULL-source odds_snapshot rows, scoped by sport/years, via
    _classify. Prints a sport×year×kind→target composition breakdown so the
    affected population is visible before any write. Idempotent; writes only with
    --apply. Non-seed/non-sbr rows are left NULL unless --live-since (date gate) or
    --all-backfill (no live capture -> all backfill_close) is given."""
    import db_store
    from sqlalchemy import select, or_, update
    t = db_store.odds_snapshot
    eng = _engine()

    scope = t.c.source.is_(None)
    if sport_key:
        scope = scope & (t.c.sport == sport_key)
    if years:
        scope = scope & or_(*[t.c.game_date.like(f"{y}%") for y in years])

    with eng.connect() as c:
        rows = c.execute(select(t.c.id, t.c.sport, t.c.game_date, t.c.kind,
                                t.c.event_id).where(scope)).all()

    buckets = {"seed": [], "sbr": [], "live": [], "backfill_close": [], None: []}
    comp = {}
    for rid, sport, gd, kind, eid in rows:
        target = _classify(kind, eid, gd, live_since, all_backfill)
        buckets[target].append(rid)
        key = (sport, (gd or "")[:4], kind, target or "(left NULL)")
        comp[key] = comp.get(key, 0) + 1

    scope_desc = (f"sport={sport_key or 'ALL'}  "
                  f"years={','.join(years) if years else 'ALL'}  "
                  + ("mode=all-backfill" if all_backfill
                     else f"live_since={live_since or '(unset)'}"))
    print(f"\n=== retag ({scope_desc}) ===")
    print(f"  NULL-source rows in scope: {len(rows)}")
    for key in sorted(comp):
        sport, yr, kind, target = key
        print(f"    {sport:18} {yr or '????'}  {kind:6} -> {target:16} {comp[key]:>7}")
    n_amb = len(buckets[None])
    if n_amb:
        print(f"  ! {n_amb} non-seed/non-sbr row(s) AMBIGUOUS (no --live-since) -> "
              f"left NULL. Pass --live-since YYYY-MM-DD to tag them "
              f"live (game_date >=) / backfill_close (<).")
    if not apply:
        print("\n  dry-run only. Re-run with --apply to write.")
        return
    with eng.begin() as c:
        for target in ("seed", "sbr", "live", "backfill_close"):
            for chunk in _chunks(buckets[target]):
                c.execute(update(t).where(t.c.id.in_(chunk)).values(source=target))
    tagged = sum(len(buckets[k]) for k in ("seed", "sbr", "live", "backfill_close"))
    print(f"  ok: tagged {tagged} row(s); left {n_amb} ambiguous row(s) NULL.")


def _prune(sport_key, years, apply, yes, kind=None, source=None, ids=None):
    """Archive + delete odds_snapshot rows, then delete. Two scoping modes:
      - ``ids`` (a list of snapshot ids): delete EXACTLY those rows — surgical,
        for removing specific corrupt/date-broken snapshots; sport/kind/source/
        years are ignored (the ids are the scope).
      - otherwise: filter by ``kind`` and/or ``source`` (at least one REQUIRED —
        never prune a whole sport unscoped) within ``sport_key`` + ``years``
        (list of year prefixes, or None = all years).
    Archives the full snapshot + line rows to a timestamped JSON first (reversible),
    THEN deletes. Covers seed cleanup (kind='seed'), the 2026 live prune
    (source='live'), untagged cruft (source='null' -> source IS NULL), and
    surgical id-deletes."""
    import db_store
    from sqlalchemy import select, func, or_, delete
    t, ln = db_store.odds_snapshot, db_store.odds_line
    eng = _engine()
    if ids:
        scope = t.c.id.in_(ids)
        desc = f"ids={','.join(str(i) for i in ids)}"
        tag = "ids"
    else:
        if not (kind or source):
            print("  refusing to prune without a --kind or --source filter (safety).")
            sys.exit(3)
        scope = (t.c.sport == sport_key)
        if kind:
            scope = scope & (t.c.kind == kind)
        if source:
            # 'null'/'none' targets the untagged (source IS NULL) legacy rows.
            if str(source).strip().lower() in ("null", "none"):
                scope = scope & t.c.source.is_(None)
            else:
                scope = scope & (t.c.source == source)
        if years:
            scope = scope & or_(*[t.c.game_date.like(f"{y}%") for y in years])
        filt = " ".join(x for x in [f"kind='{kind}'" if kind else "",
                                    f"source='{source}'" if source else ""] if x)
        desc = (f"{sport_key} {filt} "
                f"years={','.join(years) if years else 'ALL'}")
        tag = "_".join(x for x in [kind or "", source or ""] if x) or "all"
    # Subquery (not a materialized id list) so the line count/archive/delete never
    # hit SQL Server's ~2100-param IN() limit on a large scope (e.g. 3k+ seed rows).
    snap_id_q = select(t.c.id).where(scope)
    with eng.connect() as c:
        n_snap = c.execute(select(func.count()).select_from(t).where(scope)).scalar()
        n_line = c.execute(select(func.count()).select_from(ln)
                           .where(ln.c.snapshot_id.in_(snap_id_q))).scalar()
    print(f"\n=== prune: {desc} ===")
    print(f"  snapshots to delete: {n_snap}   odds_line rows to delete: {n_line}")
    if n_snap == 0:
        print("  nothing matches -- done.")
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
    path = os.path.join(arch_dir, f"pruned_{tag}_{sport_key or 'byid'}_{ts}.json")
    with eng.connect() as c:
        snaps = [dict(r._mapping) for r in c.execute(select(t).where(scope)).all()]
        lines = [dict(r._mapping) for r in c.execute(
            select(ln).where(ln.c.snapshot_id.in_(snap_id_q))).all()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"sport": sport_key, "years": years, "kind": kind,
                   "source": source, "ids": ids, "archived_at": ts,
                   "snapshots": snaps, "lines": lines}, f, indent=2, default=str)
    print(f"  archived {len(snaps)} snapshots + {len(lines)} lines -> {path}")
    # Delete lines first (explicit; not relying on FK cascade for portability),
    # then the snapshots.
    with eng.begin() as c:
        c.execute(delete(ln).where(ln.c.snapshot_id.in_(snap_id_q)))
        c.execute(delete(t).where(scope))
    print(f"  ok: deleted {n_snap} snapshots + {n_line} lines. "
          f"(reversible from the archive above.)")


def _parse_years(years_arg, default_all):
    """Comma-list / 'all' / unset -> a list of year prefixes or None (= all years).
    ``default_all`` picks what an unset --years means: True (retag) = all years;
    False (prune) = the safe 2024,2025 default."""
    if years_arg is None:
        return None if default_all else ["2024", "2025"]
    if years_arg.strip().lower() == "all":
        return None
    return [y.strip() for y in years_arg.split(",") if y.strip()]


def _resolve_sport(sport_arg):
    """Map --sport to a warehouse sport_key: a known alias (mlb->baseball_mlb),
    'all'/None -> None (no filter), or an ARBITRARY raw sport_key passed through so
    odds from sports beyond the 4 aliases (e.g. soccer_epl) can be targeted."""
    if sport_arg in (None, "all"):
        return None
    return _SPORT_KEY.get(sport_arg, sport_arg)


def main():
    p = argparse.ArgumentParser(description="Warehouse odds-provenance maintenance")
    p.add_argument("--retag", action="store_true",
                   help="Tag source for NULL-source rows (seed/sbr/live/backfill_close).")
    p.add_argument("--prune-seed", action="store_true",
                   help="Archive + delete kind='seed' snapshots for a scope "
                        "(sugar for --prune --kind seed).")
    p.add_argument("--prune", action="store_true",
                   help="Archive + delete snapshots filtered by --kind and/or --source "
                        "(e.g. --prune --source live --years 2026 = drop the 2026 "
                        "pre-relaunch live odds), or by explicit --ids. A filter or "
                        "--ids is required.")
    p.add_argument("--ids", default=None,
                   help="Comma-separated odds_snapshot.id list for a SURGICAL --prune "
                        "of exactly those rows (e.g. corrupt/date-broken snapshots). "
                        "Ignores --sport/--kind/--source/--years.")
    p.add_argument("--kind", default=None,
                   help="odds_snapshot.kind filter for --prune (e.g. seed, team, props).")
    p.add_argument("--source", default=None,
                   help="odds_snapshot.source filter for --prune (e.g. live, backfill_close; "
                        "'null'/'none' = untagged legacy rows where source IS NULL).")
    p.add_argument("--sport", default=None,
                   help="Scope: an alias (mlb/nfl/nba/nhl), 'all', or an arbitrary raw "
                        "sport_key (e.g. soccer_epl) for odds beyond the 4 aliases. "
                        "--retag: omit or 'all' = every sport. --prune/--prune-seed: "
                        "REQUIRED (a specific sport, not 'all').")
    p.add_argument("--years", default=None,
                   help="Comma-separated seasons, or 'all'. --retag: omit = ALL years. "
                        "--prune/--prune-seed: omit = 2024,2025 (safe default).")
    p.add_argument("--live-since", default=None, metavar="YYYY-MM-DD",
                   help="Live-vs-backfill boundary for --retag: NULL non-seed/non-sbr "
                        "rows with game_date >= this -> 'live', earlier -> "
                        "'backfill_close'. Without it those rows are LEFT NULL (never "
                        "guessed as live). This is the season the app began real-time "
                        "capture (e.g. 2026-01-01 for MLB).")
    p.add_argument("--all-backfill", action="store_true",
                   help="--retag: the scope had NO real-time live capture, so tag every "
                        "non-seed/non-sbr row 'backfill_close' (for other sports never "
                        "captured live). Mutually exclusive with --live-since.")
    p.add_argument("--apply", action="store_true", help="Write (default is dry-run).")
    p.add_argument("--yes", action="store_true",
                   help="Required with --apply for the destructive --prune/--prune-seed.")
    args = p.parse_args()
    if not (args.retag or args.prune_seed or args.prune):
        p.error("choose --retag, --prune-seed, and/or --prune")
    if args.live_since and args.all_backfill:
        p.error("--live-since and --all-backfill are mutually exclusive "
                "(one is a date gate, the other says 'no live capture at all').")
    if args.retag:
        _retag(args.apply, sport_key=_resolve_sport(args.sport),
               years=_parse_years(args.years, default_all=True),
               live_since=args.live_since, all_backfill=args.all_backfill)
    if args.ids and not args.prune:
        p.error("--ids is only valid with --prune (surgical id-delete).")
    if args.prune and args.ids:
        # Surgical id-delete: ids are the scope; --sport/--kind/--source/--years ignored.
        try:
            ids = [int(x) for x in args.ids.split(",") if x.strip()]
        except ValueError:
            p.error("--ids must be a comma-separated list of integer snapshot ids.")
        if not ids:
            p.error("--ids was empty.")
        _prune(None, None, args.apply, args.yes, ids=ids)
    elif args.prune_seed or args.prune:
        if args.sport in (None, "all"):
            p.error("--prune/--prune-seed requires a specific --sport "
                    "(an alias or raw sport_key), not 'all'.")
        # --prune-seed = kind='seed'; --prune uses the explicit --kind/--source.
        kind = "seed" if args.prune_seed else args.kind
        source = None if args.prune_seed else args.source
        _prune(_resolve_sport(args.sport),
               _parse_years(args.years, default_all=False),
               args.apply, args.yes, kind=kind, source=source)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
