"""One-time backfill: populate the SFBB id/code columns on the durable SQL tables
and collapse prediction-log spelling collisions — the data step of Phase 4 of the
player/team cross-map migration (see sql/schema.sql, player_id_map.py).

Run ONCE on the warm machine, in this ORDER (the ordering is the crux):
  1. sql/schema.sql  — additive columns exist, unique still on the raw player name.
  2. python player_id_map.py --refresh   — populate player_id_map / team_id_map.
  3. python backfill_player_ids.py       — THIS script (enrich + merge collisions).
  4. sql/schema.sql (re-run)             — the guarded Phase-4 block then flips
     player_key NOT NULL and swaps uq_prediction_identity → …_v2. It is a NO-OP
     until every prediction_log row has a non-NULL player_key, which this backfill
     guarantees.

Why direct id-keyed SQL (not db_store.mutate): this backfill CHANGES the hybrid
player_key on existing rows. db_store.mutate diffs before/after by that very key and
builds its UPDATE/DELETE WHERE from it, so a key that changes (or a stored
player_key still NULL) would mis-target — delete nothing, insert a duplicate. Keying
every write on the immutable PRIMARY KEY ``id`` makes the backfill correct no matter
whether the identity-code flip has shipped yet, and keeps it a pure UPDATE/DELETE.

Fail-open + idempotent: a map miss / non-MLB row leaves the id columns NULL and
player_key = "name:<norm>"; re-running is a no-op once everything is populated and
merged. Copy-not-destructive except the deliberate collision merges (which fold
outcome fields forward via recalibration._collapse_identity_rows — no graded result
is ever dropped). TAKE A SQL BACKUP BEFORE RUNNING (documented in the runbook).

Usage:
    python backfill_player_ids.py --dry-run   # report what would change; no writes
    python backfill_player_ids.py             # enrich + merge (prediction/wagers/market)
    python backfill_player_ids.py --odds      # ALSO backfill odds_snapshot/odds_line
"""

import argparse
from collections import defaultdict

from sqlalchemy import delete, select, update

import db_store
import player_id_map
import recalibration


def _is_baseball(sport):
    return (sport or "").startswith("baseball")


def _mlb_id(name):
    try:
        return player_id_map.mlb_id_for_name(name)
    except Exception:
        return None


def _team_code(name):
    if not name:
        return None
    try:
        return player_id_map.team_code_for_name(name)
    except Exception:
        return None


# ── prediction_log: enrich ids + collapse hybrid-key collisions ──

def _backfill_predictions(conn, dry_run):
    t = db_store.prediction_log
    rows = [dict(m._mapping) for m in conn.execute(select(t)).all()]
    if not rows:
        return 0, 0

    # 1. Enrich every row's id/code/player_key columns in place (MLB-gated).
    for r in rows:
        if _is_baseball(r.get("sport_key")):
            r["player_mlb_id"] = _mlb_id(r.get("player"))
            r["team_code"] = _team_code(r.get("team"))
        else:
            r.setdefault("player_mlb_id", None)
            r.setdefault("team_code", None)
        r["player_key"] = db_store.player_key(r)

    # 2. Group by the NEW hybrid identity and collapse collisions (two spellings
    #    that map to one MLBID become one row; outcome fields fold forward).
    grouped = defaultdict(list)
    order = []
    for r in rows:
        event_key = r.get("event_id") or r.get("game_date") or ""
        ident = (r.get("sport_key"), event_key, r.get("prop_key"),
                 r.get("player_key"), db_store._f(r.get("line")))
        if ident not in grouped:
            order.append(ident)
        grouped[ident].append(r)

    # Plan first (winner + folded values per group, loser ids), then apply
    # DELETEs BEFORE UPDATEs: a winner whose player_key changes to match a sibling
    # would otherwise transiently violate uq_prediction_identity_v2 mid-transaction
    # (SQL Server checks UNIQUE per-statement — not deferrable).
    planned, deletes = [], []
    for ident in order:
        group = grouped[ident]
        winner = recalibration._collapse_identity_rows(group)
        # _collapse_identity_rows returns a COPY when it merges (>1 row), so an
        # ``is`` check would flag the winner itself as a loser. The copy carries
        # the base row's immutable ``id``, so key the keep/delete split on that.
        winner_id = winner["id"]
        for r in group:
            if r["id"] != winner_id:
                deletes.append(r["id"])
        planned.append((winner_id, {
            "player_mlb_id": db_store._s(winner.get("player_mlb_id")),
            "team_code": db_store._s(winner.get("team_code")),
            "player_key": db_store._s(winner.get("player_key")),
            "resolved": db_store._bexact(winner.get("resolved")),
            "actual": db_store._f(winner.get("actual")),
            "outcome": db_store._i(winner.get("outcome")),
            "resolved_at": db_store._s(winner.get("resolved_at")),
        }))

    if not dry_run:
        if deletes:
            conn.execute(delete(t).where(t.c.id.in_(deletes)))
        for row_id, values in planned:
            conn.execute(update(t).where(t.c.id == row_id).values(**values))
    return len(planned), len(deletes)


# ── wagers / market_prediction_log: pure id/code enrichment (identity unchanged) ──

def _backfill_team_codes(conn, table, dry_run, player_col=None):
    t = table
    rows = [dict(m._mapping) for m in conn.execute(select(t)).all()]
    changed = 0
    for r in rows:
        if not _is_baseball(r.get("sport_key")):
            continue
        values = {
            "home_code": db_store._s(_team_code(r.get("home_team"))),
            "away_code": db_store._s(_team_code(r.get("away_team"))),
            "team_code": db_store._s(_team_code(r.get("team"))),
            "opponent_code": db_store._s(_team_code(r.get("opponent"))),
        }
        if player_col and r.get(player_col):
            values["player_mlb_id"] = db_store._s(_mlb_id(r.get(player_col)))
        if not dry_run:
            conn.execute(update(t).where(t.c.id == r["id"]).values(**values))
        changed += 1
    return changed


# ── odds_snapshot / odds_line: optional (potentially high-volume) enrichment ──

def _backfill_odds(conn, dry_run):
    snap = db_store.odds_snapshot
    line = db_store.odds_line
    snaps = [dict(m._mapping) for m in conn.execute(select(snap)).all()]
    sport_by_id, baseball_ids = {}, set()
    snap_changed = 0
    for r in snaps:
        sport_by_id[r["id"]] = r.get("sport")
        if not _is_baseball(r.get("sport")):
            continue
        baseball_ids.add(r["id"])
        values = {"home_code": db_store._s(_team_code(r.get("home"))),
                  "away_code": db_store._s(_team_code(r.get("away")))}
        if not dry_run:
            conn.execute(update(snap).where(snap.c.id == r["id"]).values(**values))
        snap_changed += 1

    lines = [dict(m._mapping) for m in conn.execute(select(line)).all()]
    line_changed = 0
    for r in lines:
        if r.get("snapshot_id") not in baseball_ids:
            continue
        if (r.get("bet_type") or "") == "player_prop":
            values = {"player_mlb_id": db_store._s(_mlb_id(r.get("player")))}
        else:
            values = {"team_code": db_store._s(_team_code(r.get("selection")))}
        if not dry_run:
            conn.execute(update(line).where(line.c.id == r["id"]).values(**values))
        line_changed += 1
    return snap_changed, line_changed


def main():
    parser = argparse.ArgumentParser(
        description="Backfill SFBB id/code columns + collapse prediction-log "
                    "hybrid-key collisions (Phase 4 data step).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change; write nothing.")
    parser.add_argument("--odds", action="store_true",
                        help="ALSO backfill odds_snapshot/odds_line (high volume).")
    args = parser.parse_args()

    if not db_store.promote_secrets_from_toml():
        raise SystemExit("SQL_* secrets not configured; cannot reach SQL.")
    if not db_store.enabled():
        raise SystemExit("SQL backend did not enable after promoting secrets.")
    if not player_id_map.enabled():
        raise SystemExit("player_id_map SQL not enabled — run player_id_map.py "
                         "--refresh first.")

    tag = " (dry run — no writes)" if args.dry_run else ""
    print(f"Backfilling SFBB ids/codes into Azure SQL{tag}")

    # Warm the id/team indexes (and settle any TTL refetch) BEFORE opening the
    # write transaction: keeps a possible network fetch off the critical section
    # so the backfill never holds row locks while waiting on SFBB, and keeps all
    # the map's SQL reads out of the one write transaction.
    player_id_map._player_idx()
    player_id_map._team_idx()

    engine = db_store.get_engine()
    with engine.begin() as conn:
        pred_upd, pred_del = _backfill_predictions(conn, args.dry_run)
        wag = _backfill_team_codes(conn, db_store.wagers, args.dry_run,
                                   player_col="player")
        mkt = _backfill_team_codes(conn, db_store.market_prediction_log,
                                   args.dry_run)
        print(f"  prediction_log: {pred_upd} row(s) enriched, "
              f"{pred_del} collision duplicate(s) merged away")
        print(f"  wagers:         {wag} baseball row(s) enriched")
        print(f"  market_pred:    {mkt} baseball row(s) enriched")
        if args.odds:
            snap, line = _backfill_odds(conn, args.dry_run)
            print(f"  odds_snapshot:  {snap} baseball row(s) enriched")
            print(f"  odds_line:      {line} baseball row(s) enriched")
        else:
            print("  odds_snapshot/odds_line: SKIPPED (pass --odds to include)")

    if args.dry_run:
        print("(dry run — no writes; transaction not committed with any changes)")
        return
    print("Done. Now re-run sql/schema.sql to flip player_key NOT NULL and swap "
          "uq_prediction_identity → uq_prediction_identity_v2.")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
