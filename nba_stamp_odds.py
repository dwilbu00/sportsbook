"""nba_stamp_odds.py — P5b: pre-compute the NBA odds->actuals join by stamping stats.nba.com
``nba_person_id`` + ``nba_game_id`` onto the durable ``odds_line`` rows.

Materializes the line->actuals link that ``book_line_calibration._nba_grade_book_lines`` derives
today from (normalized name + EXACT ET game_date). Stamping it once makes the join namesake- and
date-proof and lets any consumer join by id instead of re-deriving the name+date match every run.

Resolution (``nba_warehouse.resolve_prop_ids``): name_norm (``db_store.normalize_name``) + the
line's US-Eastern commence date -> EXACTLY one warehouse player-game, else left unstamped.
UNAMBIGUOUS-ONLY — a namesake collision or a no-match is never guessed (the name+date fallback
still handles it). Because the stamp points at ACTUALS, a line for a not-yet-played game resolves
to nothing; re-run this after each warehouse (re-)backfill to stamp newly-completed slates — it is
both the historical backfill AND the periodic forward-stamp (there is no capture-time hook: at
capture the game has no ``nba_player_game`` row yet).

Policy (mirrors restamp_legacy_ids): DRY-RUN BY DEFAULT (``--apply`` to write); FILL nulls +
OVERWRITE a drifted stamp, NEVER null a good stamp on a no-resolve; id-keyed UPDATE grouped by
value, chunked transactions under a lock with an OperationalError retry; idempotent. Writes only
the two stamp columns — never touches odds data (non-destructive on the durable corpus).

Run (owner, Azure; free DB I/O, no Odds API credits):
  $env:PYTHONIOENCODING="utf-8"
  python nba_stamp_odds.py            # dry-run preview (no writes)
  python nba_stamp_odds.py --apply    # write the stamp
"""
import argparse
import threading
import time
from collections import defaultdict

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError

import db_store
import nba_warehouse

_LOCK = threading.Lock()
_SPORT = "basketball_nba"


def _et_date(commence_time, game_date):
    """US-Eastern calendar date of the line's commence (what the real-line join matches), with
    the stored snapshot game_date as a fallback."""
    try:
        from pricing_common import et_local_date
        d = et_local_date(commence_time)
        if d:
            return str(d)[:10]
    except Exception:
        pass
    return str(game_date)[:10] if game_date else None


def _season_of(et_date):
    try:
        import nba_stats_source
        return nba_stats_source.end_year_for_date(et_date)
    except Exception:
        try:
            return int(str(et_date)[:4])
        except (TypeError, ValueError):
            return None


def _prop_rows(conn):
    """NBA player-prop odds_line rows + the parent snapshot's commence/date context."""
    ol, s = db_store.odds_line, db_store.odds_snapshot
    stmt = (select(ol.c.id, ol.c.player, ol.c.nba_person_id, ol.c.nba_game_id,
                   s.c.commence_time, s.c.game_date)
            .select_from(ol.join(s, ol.c.snapshot_id == s.c.id))
            .where((s.c.sport == _SPORT)
                   & (ol.c.bet_type == "player_prop")
                   & (ol.c.player.isnot(None))))
    for rid, player, pid, gid, commence, gdate in conn.execute(stmt):
        yield {"id": rid, "player": player, "nba_person_id": pid,
               "nba_game_id": gid, "commence_time": commence, "game_date": gdate}


def _plan(conn):
    """Return (groups, stats, diffs). groups = {(person_id, game_id): [odds_line.id]} to UPDATE;
    stats = per-outcome counts; diffs = a sample of planned/observed changes."""
    rows = list(_prop_rows(conn))
    seasons = sorted({_season_of(_et_date(r["commence_time"], r["game_date"]))
                      for r in rows} - {None})
    index = nba_warehouse.prop_id_index(seasons) if seasons else {}

    cache = {}                                           # (name_norm-input, et_date) -> (pid, gid)

    def _resolve(player, et_date):
        key = (player or "", et_date or "")
        if key not in cache:
            cache[key] = nba_warehouse.resolve_prop_ids(index, player, et_date)
        return cache[key]

    groups = defaultdict(list)
    stats = defaultdict(int)
    diffs = []
    for r in rows:
        stats["scanned"] += 1
        et = _et_date(r["commence_time"], r["game_date"])
        pid, gid = _resolve(r["player"], et)
        if pid is None or gid is None:
            stats["unresolved"] += 1                     # no match / namesake-ambiguous / future
            continue
        cur = (r["nba_person_id"], r["nba_game_id"])
        want = (int(pid), str(gid))
        if (cur[0] is not None and int(cur[0]) == want[0]
                and cur[1] is not None and str(cur[1]) == want[1]):
            stats["already_correct"] += 1
            continue
        drift = cur[0] is not None or cur[1] is not None
        stats["overwrite_drift" if drift else "fill"] += 1
        groups[want].append(r["id"])
        if len(diffs) < 25:
            diffs.append((r["id"], r["player"], et, cur, want,
                          "overwrite" if drift else "fill"))
    stats["to_update_rows"] = sum(len(v) for v in groups.values())
    stats["to_update_games"] = len(groups)
    stats["seasons"] = seasons
    return groups, stats, diffs


def _apply(eng, groups, chunk=1000, per_txn=400, verbose=True):
    """id-keyed UPDATE of (nba_person_id, nba_game_id), grouped by value (one UPDATE per distinct
    (pid, gid), chunked at ``chunk`` ids). Commits ``per_txn`` UPDATE statements per transaction
    — bounds the SQL log AND the transaction/round-trip count on the 20-DTU tier (vs one txn per
    group) — under _LOCK with a bounded OperationalError retry. Single-threaded (no concurrency →
    no DTU saturation). Idempotent, so an interrupted run resumes cleanly on re-run."""
    ol = db_store.odds_line
    stmts = []                                            # [(pid, gid, [ids])] — one UPDATE each
    for (pid, gid), ids in groups.items():
        for i in range(0, len(ids), chunk):
            stmts.append((pid, gid, ids[i:i + chunk]))
    total_rows = sum(len(ids) for _, _, ids in stmts)

    def _flush(batch):
        for attempt in range(3):
            try:
                with _LOCK:
                    with eng.begin() as conn:
                        for pid, gid, bids in batch:
                            conn.execute(update(ol).where(ol.c.id.in_(bids))
                                         .values(nba_person_id=pid, nba_game_id=gid))
                return
            except OperationalError:
                if attempt == 2:
                    raise
                time.sleep(1 + 2 * attempt)

    done = 0
    for i in range(0, len(stmts), per_txn):
        batch = stmts[i:i + per_txn]
        _flush(batch)
        done += sum(len(b[2]) for b in batch)
        if verbose:
            print(f"    stamped {done:,}/{total_rows:,} rows "
                  f"({100.0 * done / total_rows:.0f}%)", flush=True)


def stamp(dry_run=True):
    """Plan (and unless dry_run, apply) the NBA odds-line id stamp. Returns a summary dict."""
    summary = {"dry_run": dry_run, "skipped": not db_store.enabled()}
    if not db_store.enabled():
        return summary
    eng = db_store.get_engine()
    with eng.connect() as conn:
        groups, stats, diffs = _plan(conn)
    if not dry_run and groups:
        _apply(eng, groups)
    summary["stats"] = dict(stats)
    summary["diffs"] = diffs
    return summary


def _report(summary):
    print("\n=== P5b: NBA odds_line id-stamp ===")
    if summary.get("skipped"):
        print("  SKIPPED — SQL not configured (SQL_* secrets).")
        return
    st = summary.get("stats", {})
    mode = "DRY-RUN (no writes)" if summary.get("dry_run") else "APPLIED"
    print(f"  mode: {mode}   seasons: {st.get('seasons')}")
    print(f"  scanned NBA prop lines : {st.get('scanned', 0):,}")
    print(f"  already correct        : {st.get('already_correct', 0):,}")
    print(f"  fill (was NULL)        : {st.get('fill', 0):,}")
    print(f"  overwrite (drift)      : {st.get('overwrite_drift', 0):,}")
    print(f"  unresolved (skip)      : {st.get('unresolved', 0):,}  "
          f"(no match / namesake-ambiguous / game not yet played)")
    print(f"  -> rows to update      : {st.get('to_update_rows', 0):,}  "
          f"across {st.get('to_update_games', 0):,} distinct (person, game)")
    diffs = summary.get("diffs") or []
    if diffs:
        print(f"\n  sample (first {len(diffs)}):  id  player  et-date  cur -> new  [action]")
        for rid, player, et, cur, want, action in diffs:
            print(f"    [{action}] odds_line#{rid}  {player}  {et}  "
                  f"{cur} -> {want}")
    if summary.get("dry_run") and st.get("to_update_rows"):
        print("\n  Re-run with --apply to write the stamp.")


def _main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="P5b: stamp nba_person_id/nba_game_id onto NBA player-prop odds_line rows.")
    ap.add_argument("--apply", action="store_true",
                    help="Write the stamp (default: dry-run preview, no writes).")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not db_store.enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to stamp.")
    _report(stamp(dry_run=not args.apply))


if __name__ == "__main__":
    _main()
