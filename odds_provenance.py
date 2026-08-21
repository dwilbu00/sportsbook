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


def _classify(kind, event_id, game_date, live_since):
    """Provenance for one NULL-source row, or None if it must be left untagged.
    Positive rules first (seed by kind, sbr by id prefix); the live-vs-backfill
    split is a game_date season gate, NEVER classification-by-elimination — the
    pre-column population mixes real-time live captures with historical backfill
    closes, so 'everything not seed/sbr = live' would mislabel the whole corpus."""
    if kind == "seed":
        return "seed"
    if (event_id or "").startswith("sbr-"):
        return "sbr"
    if live_since is None:
        return None  # ambiguous without a boundary → leave NULL, never guess live
    return "live" if (game_date or "") >= live_since else "backfill_close"


def _retag(apply, sport_key=None, years=None, live_since=None):
    """Tag source for NULL-source odds_snapshot rows, scoped by sport/years, via
    _classify. Prints a sport×year×kind→target composition breakdown so the
    affected population is visible before any write. Idempotent; writes only with
    --apply. Non-seed/non-sbr rows are left NULL unless --live-since is given."""
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
        target = _classify(kind, eid, gd, live_since)
        buckets[target].append(rid)
        key = (sport, (gd or "")[:4], kind, target or "(left NULL)")
        comp[key] = comp.get(key, 0) + 1

    scope_desc = (f"sport={sport_key or 'ALL'}  "
                  f"years={','.join(years) if years else 'ALL'}  "
                  f"live_since={live_since or '(unset)'}")
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


def _parse_years(years_arg, default_all):
    """Comma-list / 'all' / unset -> a list of year prefixes or None (= all years).
    ``default_all`` picks what an unset --years means: True (retag) = all years;
    False (prune) = the safe 2024,2025 default."""
    if years_arg is None:
        return None if default_all else ["2024", "2025"]
    if years_arg.strip().lower() == "all":
        return None
    return [y.strip() for y in years_arg.split(",") if y.strip()]


def main():
    p = argparse.ArgumentParser(description="Warehouse odds-provenance maintenance")
    p.add_argument("--retag", action="store_true",
                   help="Tag source for NULL-source rows (seed/sbr/live/backfill_close).")
    p.add_argument("--prune-seed", action="store_true",
                   help="Archive + delete kind='seed' snapshots for a scope.")
    p.add_argument("--sport", default=None,
                   choices=list(_SPORT_KEY.keys()) + ["all"],
                   help="Scope. --retag: omit or 'all' = every sport. "
                        "--prune-seed: REQUIRED (a specific sport, not 'all').")
    p.add_argument("--years", default=None,
                   help="Comma-separated seasons, or 'all'. --retag: omit = ALL years. "
                        "--prune-seed: omit = 2024,2025 (safe default).")
    p.add_argument("--live-since", default=None, metavar="YYYY-MM-DD",
                   help="Live-vs-backfill boundary for --retag: NULL non-seed/non-sbr "
                        "rows with game_date >= this -> 'live', earlier -> "
                        "'backfill_close'. Without it those rows are LEFT NULL (never "
                        "guessed as live). This is the season the app began real-time "
                        "capture (e.g. 2026-01-01 for MLB).")
    p.add_argument("--apply", action="store_true", help="Write (default is dry-run).")
    p.add_argument("--yes", action="store_true",
                   help="Required with --apply for the destructive --prune-seed.")
    args = p.parse_args()
    if not (args.retag or args.prune_seed):
        p.error("choose --retag and/or --prune-seed")
    if args.retag:
        sport_key = None if args.sport in (None, "all") else _SPORT_KEY[args.sport]
        _retag(args.apply, sport_key=sport_key,
               years=_parse_years(args.years, default_all=True),
               live_since=args.live_since)
    if args.prune_seed:
        if args.sport in (None, "all"):
            p.error("--prune-seed requires a specific --sport (mlb/nfl/nba/nhl), not 'all'.")
        _prune_seed(_SPORT_KEY[args.sport],
                    _parse_years(args.years, default_all=False),
                    args.apply, args.yes)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
