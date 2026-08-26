"""
Purge one season's MLB data from the Azure warehouse — DRY-RUN FIRST.

WHY: 2023 is the pitch-clock-transition regime, excluded from calibration, and
statcast_pitch is the biggest table — dropping 2023 reclaims a lot of rows.

SAFETY MODEL
------------
- DRY-RUN by default: prints per-table 2023 row counts in FK-safe delete order.
  Nothing is deleted without BOTH --apply AND --yes.
- FK-safe order: odds_snapshot first (cascades odds_line ON DELETE CASCADE);
  mlb_batter_game/mlb_pitcher_game BEFORE mlb_game (RESTRICT FK -> a bare
  mlb_game delete would fail while child fact rows exist).
- BULK tables (default: PURGE) are free to re-ingest:
    statcast_pitch/statcast_day   -> savant_history.py --ensure --season 2023
    mlb_game/_batter_game/_pitcher_game -> mlb_warehouse.py --ingest-range 2023-03-01 2023-11-30
    mlb_batter_gamelog/_pitcher_gamelog -> gamelog_store (ESPN, best-effort)
    odds_snapshot/odds_line       -> 2023 was SBR-sourced + already excluded; low value
- SPECIAL tables (default: KEEP; opt-in flags to purge) — tiny and/or not free
  to restore and/or a live consumer:
    pitcher_asof_daily + statcast_player_asof  (--purge-asof)
        The additive expected-runs model reads season-1 (2023) as the prior-season
        blend for EARLY-2024 starts. Deleting silently shifts early-2024 additive
        features to the league prior (no crash). Tiny table -> keep by default.
    weather_game        (--purge-weather)   only re-ingestible via the PAID
        Visual Crossing API; also a variance-signal candidate for the upset study.
    mlb_team_standings  (--purge-standings)  point-in-time as-of snapshots are
        not cleanly re-derivable.

USAGE
-----
    python purge_season.py --season 2023                 # dry-run (counts only)
    python purge_season.py --season 2023 --apply --yes   # execute the default (bulk) purge
    python purge_season.py --season 2023 --apply --yes --purge-asof --purge-weather --purge-standings
"""
import argparse
import sys


# (table, where-clause, category, note). Order is the FK-safe DELETE order.
# category: "bulk" (purge by default) | "asof" | "weather" | "standings" (keep by default).
def _specs(season):
    y = int(season)
    lo, hi = f"{y}-01-01", f"{y}-12-31"
    return [
        # odds — DELETE THE CHILD LINES EXPLICITLY FIRST (via the parent's indexed
        # id set), THEN the snapshots. We do NOT lean on ON DELETE CASCADE here: a
        # single DELETE TOP(50000) on odds_snapshot cascades to millions of odds_line
        # rows in ONE transaction and blows Azure's 60s pymssql query timeout
        # (DB-Lib 20003 "connection timed out"). Child-first keeps every batch a
        # plain indexed delete (ix_odds_line_snapshot / ix_odds_snapshot_event) that
        # finishes well under the timeout. The subquery re-resolves the 2023 snapshot
        # ids each batch — cheap on the index — and by the time we reach the snapshot
        # delete its children are already gone, so the cascade is a no-op.
        ("odds_line",
         f"snapshot_id IN (SELECT id FROM odds_snapshot WHERE sport = 'baseball_mlb' "
         f"AND game_date >= '{lo}' AND game_date <= '{hi}')",
         "bulk", "child lines first (explicit — avoids the cascade timeout)"),
        ("odds_snapshot", f"sport = 'baseball_mlb' AND game_date >= '{lo}' AND game_date <= '{hi}'",
         "bulk", "snapshot parents (children already deleted → cascade no-op)"),
        # game facts BEFORE mlb_game (RESTRICT FK)
        ("mlb_batter_game", f"season_bucket = {y}", "bulk", "child of mlb_game"),
        ("mlb_pitcher_game", f"season_bucket = {y}", "bulk", "child of mlb_game"),
        ("mlb_game", f"season = {y}", "bulk", "spine — delete after facts"),
        # statcast corpus
        ("statcast_pitch", f"game_date >= '{lo}' AND game_date <= '{hi}'", "bulk", "the whale (~700k/season)"),
        ("statcast_day", f"game_date LIKE '{y}-%'", "bulk", "ingest manifest"),
        # ESPN gamelogs
        ("mlb_batter_gamelog", f"season_bucket = {y}", "bulk", ""),
        ("mlb_pitcher_gamelog", f"season_bucket = {y}", "bulk", ""),
        # SPECIAL (keep by default)
        ("pitcher_asof_daily", f"season_bucket = {y}", "asof",
         "additive prior-season blend for early-{}".format(y + 1)),
        ("statcast_player_asof", f"season_bucket = {y}", "asof", "derived as-of rates"),
        ("weather_game", f"weather_date >= '{lo}' AND weather_date <= '{hi}'", "weather",
         "PAID to re-ingest"),
        ("mlb_team_standings", f"season = {y}", "standings", "as-of snapshots not cleanly restorable"),
    ]


def main():
    p = argparse.ArgumentParser(description="Purge one MLB season from the warehouse (dry-run first).")
    p.add_argument("--season", type=int, default=2023)
    p.add_argument("--apply", action="store_true", help="Execute deletes (default = dry-run counts only).")
    p.add_argument("--yes", action="store_true", help="Required with --apply (double-confirm).")
    p.add_argument("--purge-asof", action="store_true",
                   help="ALSO purge pitcher_asof_daily + statcast_player_asof (additive prior-season blend).")
    p.add_argument("--purge-weather", action="store_true",
                   help="ALSO purge weather_game (only PAID-restorable).")
    p.add_argument("--purge-standings", action="store_true",
                   help="ALSO purge mlb_team_standings (as-of snapshots not cleanly restorable).")
    p.add_argument("--batch", type=int, default=20000,
                   help="Rows per DELETE TOP() chunk. Each chunk must finish inside "
                        "Azure's ~60s pymssql query timeout; lower it (e.g. 5000) if a "
                        "table still times out. Default 20000.")
    args = p.parse_args()

    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    import db_store
    db_store.promote_secrets_from_toml()
    if not db_store.enabled():
        print("SQL not configured (db_store.enabled() is False). Aborting.")
        sys.exit(1)
    from sqlalchemy import text
    engine = db_store.get_engine()

    keep = {"asof": not args.purge_asof, "weather": not args.purge_weather,
            "standings": not args.purge_standings}
    specs = _specs(args.season)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n{'='*66}\n  Purge MLB season {args.season}  [{mode}]\n{'='*66}")
    print(f"  {'table':<22} {'2023 rows':>12}  action")
    print(f"  {'-'*22} {'-'*12}  {'-'*22}")

    to_delete, kept, total_del = [], [], 0
    with engine.connect() as conn:
        for table, where, cat, note in specs:
            try:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {where}")).scalar() or 0
            except Exception as exc:
                print(f"  {table:<22} {'ERR':>12}  count failed: {type(exc).__name__} ({exc})")
                continue
            will_keep = cat != "bulk" and keep.get(cat, False)
            action = ("KEEP  (" + note + ")" if will_keep
                      else ("PURGE " + (f"({note})" if note else "")).rstrip())
            print(f"  {table:<22} {n:>12,}  {action}")
            if will_keep:
                kept.append((table, n))
            else:
                to_delete.append((table, where, n))
                total_del += n

    print(f"  {'-'*22} {'-'*12}")
    print(f"  {'TO DELETE':<22} {total_del:>12,}  across {len(to_delete)} table(s)")
    if kept:
        print(f"  KEPT (add flags to purge): "
              + ", ".join(f"{t}({n:,})" for t, n in kept))

    if not args.apply:
        print("\n  DRY-RUN — nothing deleted. Re-run with --apply --yes to execute.")
        print("  Re-ingest 2023 later if needed: savant_history.py --ensure --season 2023 ;"
              " mlb_warehouse.py --ingest-range 2023-03-01 2023-11-30")
        return
    if not args.yes:
        print("\n  --apply requires --yes (double-confirm). Nothing deleted.")
        sys.exit(1)

    print(f"\n  Executing {len(to_delete)} delete(s) in FK-safe order...")
    # Chunked deletes: a single huge DELETE bloats the Azure SQL transaction log +
    # risks lock escalation, and — critically — must finish inside pymssql's ~60s
    # query timeout (a 50k-snapshot cascade to odds_line exceeded it: DB-Lib 20003
    # "connection timed out"). DELETE TOP(BATCH) in a loop keeps each transaction
    # small + re-runnable (a resumed run just deletes what's left); odds_line is
    # deleted child-first (see _specs) so no batch carries a cascade. Lower --batch
    # if any single table still times out.
    BATCH = args.batch
    done = 0
    for table, where, n in to_delete:
        try:
            deleted = 0
            while True:
                with engine.begin() as conn:
                    res = conn.execute(
                        text(f"DELETE TOP ({BATCH}) FROM {table} WHERE {where}"))
                got = getattr(res, "rowcount", 0) or 0
                deleted += got
                if got < BATCH:      # last chunk (0 on an exact multiple → clean stop)
                    break
            done += 1
            print(f"  [{done}/{len(to_delete)}] {table:<22} deleted {deleted:,}")
        except Exception as exc:
            print(f"  [FAIL] {table}: {type(exc).__name__} ({exc}) — stopping to avoid a partial-order violation.")
            sys.exit(1)
    print(f"\n  Done. Purged season {args.season} from {done} table(s).")


if __name__ == "__main__":
    main()
