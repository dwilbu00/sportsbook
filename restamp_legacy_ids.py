"""restamp_legacy_ids.py — Commit C Phase 4: dry-run-first legacy identity re-stamp.

Re-derives the MLBAM identity STAMP (player_mlb_id + player_key) on the durable SQL
corpus — prediction_log, wagers, and optionally odds_line — through the flipped
game-context/role-verified resolver (mlb_starters.resolve_mlbam_id, the SAME id-core
the live stamp now runs), correcting the drift the pre-flip SFBB-only stamp left in
the corpus (e.g. a batter prop bound to a namesake PITCHER id — the Luis Garcia Jr.
case). Mirrors mlb_warehouse.backfill_legacy_game_pk: DRY-RUN BY DEFAULT, --apply to
write; id-keyed UPDATE under a write lock inside one transaction with an
OperationalError retry; MLB-only; idempotent; per-table report + per-row old->new
diff sample.

Policy (owner-chosen 2026-08-15):
  * OVERWRITE-drifted + FILL-gains, NEVER-null-a-good-id — overwrite an existing
    non-null id only when the resolver confidently returns a DIFFERENT id, fill a
    NULL id, and leave the stored id untouched when the resolver returns None.
  * BOTH-TEAMS hint — prediction_log stores only the player's OWN team, so recover
    home/away from the row's real event_id via odds_snapshot / market_prediction_log
    (id-INDEPENDENT). A row with only a game_date event_key (no event_id) falls back to
    the own team rather than risk a wrong-game guess. game_pk is NOT a teams source —
    it was P5-pinned off the OLD id, so it would mis-hint the very drift rows being
    corrected. wagers/odds_line carry both teams directly / via the snapshot.
  * game_pk — when a row's id CHANGES off a non-null old id, NULL its (now-stale,
    P5-pinned-off-the-old-id) game_pk so `mlb_warehouse.py --backfill-game-pk
    --apply` re-pins it for the corrected player.
  * prediction_log COLLISION-MERGE — a gain (name:<norm> -> mlb:<id>) can collide
    with an already-correctly-stamped row on uq_prediction_identity_v2; collapse the
    group via recalibration._collapse_identity_rows (folding graded outcomes forward
    so nothing is lost), DELETE losers BEFORE UPDATE-ing the winner (SQL Server
    checks UNIQUE per-statement, non-deferrable).

SAFETY: warms the season roster index and ABORTS if any corpus season's index is
cold/empty — so a StatsAPI outage can't mass-restamp everything through the
drift-prone SFBB tier. SQL-only; there is NO in-code snapshot — TAKE A SQL BACKUP
before --apply. Writes are keyed on the immutable PK id (never the mutable
player_key). Run shadow_stamp.py first and eyeball its changed/loss sets.
"""

import argparse
import threading
import time
from collections import defaultdict

from sqlalchemy import delete, select, update
from sqlalchemy.exc import OperationalError

import db_store
import recalibration

_LOCK = threading.Lock()


def _season(game_date, commence=None):
    for v in (game_date, commence):
        if v:
            try:
                return int(str(v)[:4])
            except (TypeError, ValueError):
                pass
    return None


def _resolve_id_uncached(name, season, prop_key, teams):
    """The gate-ON id-core (the exact call shadow_stamp._new_id + the live stamp use
    for a no-context event): the resolved MLBAM id as a str, or None. Never raises."""
    if not name or not season:
        return None
    try:
        import mlb_starters
        found = mlb_starters.resolve_mlbam_id(
            name, season, prop_key=prop_key, teams=teams,
            confirmed_lineup=None, probable_starters=None)
    except Exception:
        return None
    return str(found[0]) if found else None


def _resolve_id(name, season, prop_key, teams, cache):
    """Cached _resolve_id_uncached — dedups the resolver across the (often many)
    rows that share a (name, role, teams, season)."""
    role = (None if not prop_key
            else ("P" if str(prop_key).startswith("pitcher_") else "B"))
    key = (name, role, tuple(teams) if teams else (), season)
    if key not in cache:
        cache[key] = _resolve_id_uncached(name, season, prop_key, teams)
    return cache[key]


def _moved(old_id, eff_id):
    """True when the effective id differs from the stored id (both str-normalized)."""
    return (str(eff_id) if eff_id is not None else None) != (
        str(old_id) if old_id is not None else None)


def _team_hint_maps(conn):
    """Both-teams recovery lookups for a prediction_log row (own-team-only), keyed on
    the UNAMBIGUOUS, id-INDEPENDENT odds event_id: by event_id [odds snapshot] and by
    (sport_key, event_key) [market log]. game_pk is deliberately NOT a source — it was
    P5-pinned FROM the (possibly wrong) OLD player id, so for the very drift rows this
    re-stamp corrects it would encode the wrong game's teams and mis-hint the resolve."""
    by_event_key, by_event_id = {}, {}
    m = db_store.market_prediction_log
    for sk, ek, home, away in conn.execute(
            select(m.c.sport_key, m.c.event_key, m.c.home_team, m.c.away_team)
            .where(m.c.sport_key == "baseball_mlb")):
        if ek and home and away:
            by_event_key.setdefault((sk, str(ek)), (home, away))
    s = db_store.odds_snapshot
    for eid, home, away in conn.execute(
            select(s.c.event_id, s.c.home, s.c.away)
            .where(s.c.sport == "baseball_mlb")):
        if eid and home and away:
            by_event_id.setdefault(str(eid), (home, away))
    return by_event_key, by_event_id


def _pred_teams(row, maps):
    """Two-team hint for a prediction_log row, derived ONLY from a real (unambiguous)
    event_id via the odds snapshot / market log; falls back to the player's own team.
    A row whose event_key is a game_date (event_id NULL) is NOT looked up by that date
    (multiple games share one date → wrong-game teams), because a correct single
    own-team hint beats a possibly-wrong two-team one."""
    by_event_key, by_event_id = maps
    eid = row.get("event_id")
    if eid:
        t = (by_event_id.get(str(eid))
             or by_event_key.get((row.get("sport_key"), str(eid))))
        if t:
            return [t[0], t[1]]
    own = row.get("team")
    return [own] if own else None


def _plan_predictions(conn, maps, cache, diffs):
    """Return (updates, deletes) for prediction_log. updates: [{rid, **values}];
    deletes: [rid] (collision losers). Delta-minimal: only changed/merged rows."""
    t = db_store.prediction_log
    cols = ("id", "ts", "sport_key", "event_id", "event_key", "commence_time",
            "prop_key", "player", "game_date", "resolved_at", "line", "actual",
            "outcome", "resolved", "team", "player_mlb_id", "game_pk")
    rows = [dict(zip(cols, r)) for r in conn.execute(
        select(*[t.c[c] for c in cols]).where(t.c.sport_key == "baseball_mlb"))]

    # Re-derive id + new player_key + game_pk policy on each row (in memory).
    for r in rows:
        old_id = r.get("player_mlb_id")
        season = _season(r.get("game_date"), r.get("commence_time"))
        new_id = _resolve_id(r.get("player"), season, r.get("prop_key"),
                             _pred_teams(r, maps), cache)
        eff = new_id if new_id is not None else old_id     # never-null-on-None
        r["_old_id"] = old_id
        r["_moved"] = _moved(old_id, eff)
        r["player_mlb_id"] = eff
        r["player_key"] = db_store.player_key(r)
        if r["_moved"] and old_id is not None:             # stale P5 pin -> re-pin
            r["game_pk"] = None

    # Group by the post-restamp identity == uq_prediction_identity_v2.
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("sport_key"), r.get("event_key") or "", r.get("prop_key"),
                r.get("player_key"), db_store._f(r.get("line")))].append(r)

    updates, deletes = [], []
    for group in groups.values():
        winner = recalibration._collapse_identity_rows(group)
        winner_id = winner["id"]
        merged = len(group) > 1
        for g in group:
            if g["id"] != winner_id:
                deletes.append(g["id"])
                diffs.append(("prediction_log", g["id"], g.get("player"),
                              g.get("prop_key"), g.get("_old_id"), None,
                              "merge-delete"))
        if not (merged or winner.get("_moved")):
            continue
        values = {"player_mlb_id": db_store._s(winner.get("player_mlb_id")),
                  "player_key": db_store._s(winner.get("player_key")),
                  "game_pk": db_store._i(winner.get("game_pk"))}
        if merged:   # fold the collapsed outcome forward onto the winner
            values.update({
                "resolved": db_store._bexact(winner.get("resolved")),
                "actual": db_store._f(winner.get("actual")),
                "outcome": db_store._i(winner.get("outcome")),
                "resolved_at": db_store._s(winner.get("resolved_at"))})
        updates.append({"rid": winner_id, **values})
        diffs.append(("prediction_log", winner_id, winner.get("player"),
                      winner.get("prop_key"), winner.get("_old_id"),
                      winner.get("player_mlb_id"),
                      "merge-winner" if merged else "restamp"))
    return updates, deletes


def _plan_flat(conn, table, cache, diffs, table_name, teams_of):
    """Re-stamp plan for a flat (no player_key / no collision) table — wagers,
    odds_line. ``teams_of(row)`` -> (teams_list, season). Uniform {player_mlb_id,
    game_pk} value dicts. Only MOVED player-prop rows are updated."""
    updates = []
    for r in _flat_rows(conn, table, table_name):
        teams, season = teams_of(r)
        old_id = r.get("player_mlb_id")
        new_id = _resolve_id(r.get("player"), season, r.get("prop_key"), teams, cache)
        eff = new_id if new_id is not None else old_id
        if not _moved(old_id, eff):
            continue
        updates.append({
            "rid": r["id"],
            "player_mlb_id": db_store._s(eff),
            # null a stale game_pk only when the id moved OFF a non-null old id.
            "game_pk": None if old_id is not None else db_store._i(r.get("game_pk"))})
        diffs.append((table_name, r["id"], r.get("player"), r.get("prop_key"),
                      old_id, eff, "restamp"))
    return updates


def _flat_rows(conn, table, table_name):
    """Read baseball player-prop rows + the context each flat table needs."""
    if table_name == "wagers":
        c = table.c
        for r in conn.execute(select(
                c.id, c.player, c.prop_key, c.player_mlb_id, c.game_pk,
                c.home_team, c.away_team, c.game_date, c.commence_time)
                .where((c.sport_key == "baseball_mlb") & (c.player.isnot(None)))):
            yield dict(zip(("id", "player", "prop_key", "player_mlb_id", "game_pk",
                            "home_team", "away_team", "game_date", "commence_time"),
                           r))
    else:  # odds_line — context (teams/date/sport) lives on the parent snapshot
        ol, s = table.c, db_store.odds_snapshot.c
        snaps = {sid: (home, away, gd, ct) for sid, home, away, gd, ct
                 in conn.execute(select(s.id, s.home, s.away, s.game_date,
                                        s.commence_time)
                                 .where(s.sport == "baseball_mlb"))}
        for r in conn.execute(select(
                ol.id, ol.snapshot_id, ol.player, ol.prop_key, ol.player_mlb_id,
                ol.game_pk).where((ol.bet_type == "player_prop")
                                  & (ol.player.isnot(None)))):
            snap = snaps.get(r[1])
            if not snap:
                continue   # not a baseball snapshot
            home, away, gd, ct = snap
            yield {"id": r[0], "player": r[2], "prop_key": r[3],
                   "player_mlb_id": r[4], "game_pk": r[5],
                   "home_team": home, "away_team": away,
                   "game_date": gd, "commence_time": ct}


def _teams_of_flat(row):
    teams = [x for x in (row.get("home_team"), row.get("away_team")) if x] or None
    return teams, _season(row.get("game_date"), row.get("commence_time"))


def _apply(eng, table, updates, deletes):
    """One transaction: DELETE losers BEFORE per-row id-keyed UPDATEs (heterogeneous
    value sets), under _LOCK with a bounded OperationalError retry."""
    if not updates and not deletes:
        return
    for attempt in range(3):
        try:
            with _LOCK:
                with eng.begin() as conn:
                    if deletes:
                        conn.execute(delete(table).where(table.c.id.in_(deletes)))
                    for u in updates:
                        vals = {k: v for k, v in u.items() if k != "rid"}
                        conn.execute(update(table)
                                     .where(table.c.id == u["rid"]).values(**vals))
            return
        except OperationalError:
            if attempt == 2:
                raise
            time.sleep(1 + 2 * attempt)


def _corpus_seasons(eng):
    seasons = set()
    with eng.connect() as conn:
        for (gd,) in conn.execute(
                select(db_store.prediction_log.c.game_date)
                .where(db_store.prediction_log.c.sport_key == "baseball_mlb")
                .distinct()):
            if gd:
                try:
                    seasons.add(int(str(gd)[:4]))
                except (TypeError, ValueError):
                    pass
    return sorted(seasons)


def restamp(dry_run=True, do_odds=False, samples=25):
    """Plan (and, unless dry_run, apply) the legacy identity re-stamp. Returns a
    JSON-able summary dict. NEVER raises into a write (fail-open per table)."""
    summary = {"dry_run": dry_run, "skipped": not db_store.enabled()}
    if not db_store.enabled():
        return summary

    eng = db_store.get_engine()

    # SAFETY — warm the season roster indexes + ABORT if any is cold/empty, so a
    # StatsAPI outage can't push the whole batch onto the drift-prone SFBB tier.
    import mlb_starters
    seasons = _corpus_seasons(eng)
    cold = []
    for s in seasons:
        try:
            mlb_starters.warm_player_index(s)
            if not mlb_starters._player_index(s):
                cold.append(s)
        except Exception:
            cold.append(s)
    if cold:
        summary["aborted"] = (f"cold/empty StatsAPI roster index for season(s) "
                              f"{cold} — refusing to mass-restamp via SFBB-only; "
                              f"retry when StatsAPI is healthy")
        return summary
    summary["seasons"] = seasons

    cache, diffs = {}, []
    with eng.connect() as conn:
        maps = _team_hint_maps(conn)
    with eng.connect() as conn:
        pred_upd, pred_del = _plan_predictions(conn, maps, cache, diffs)
    with eng.connect() as conn:
        wag_upd = _plan_flat(conn, db_store.wagers, cache, diffs, "wagers",
                             _teams_of_flat)
    odds_upd = []
    if do_odds:
        with eng.connect() as conn:
            odds_upd = _plan_flat(conn, db_store.odds_line, cache, diffs,
                                  "odds_line", _teams_of_flat)

    if not dry_run:
        _apply(eng, db_store.prediction_log, pred_upd, pred_del)
        _apply(eng, db_store.wagers, wag_upd, [])
        if do_odds:
            _apply(eng, db_store.odds_line, odds_upd, [])

    summary["prediction_log"] = {"updated": len(pred_upd),
                                 "merged_deleted": len(pred_del)}
    summary["wagers"] = {"updated": len(wag_upd)}
    if do_odds:
        summary["odds_line"] = {"updated": len(odds_upd)}
    summary["total_changes"] = (len(pred_upd) + len(pred_del) + len(wag_upd)
                                + len(odds_upd))
    summary["diffs_sample"] = diffs[:samples]
    return summary


def _report(summary):
    import json
    print("\n=== Commit C legacy identity re-stamp ===")
    if summary.get("skipped"):
        print("  SKIPPED — SQL not configured.")
        return
    if summary.get("aborted"):
        print(f"  ABORTED — {summary['aborted']}")
        return
    mode = "DRY-RUN (no writes)" if summary.get("dry_run") else "APPLIED"
    print(f"  mode: {mode}   seasons: {summary.get('seasons')}")
    for tbl in ("prediction_log", "wagers", "odds_line"):
        if tbl in summary:
            print(f"  {tbl}: {json.dumps(summary[tbl])}")
    print(f"  total changes: {summary.get('total_changes', 0)}")
    diffs = summary.get("diffs_sample") or []
    if diffs:
        print(f"\n  old -> new (first {len(diffs)}):")
        for tbl, rid, player, pk, old, new, action in diffs:
            print(f"    [{action}] {tbl}#{rid} {player} [{pk}]: {old} -> {new}")
    if not summary.get("dry_run") and summary.get("total_changes"):
        print("\n  NOTE: id changes nulled some stale game_pk values — now run "
              "`python mlb_warehouse.py --backfill-game-pk --apply` to re-pin them.")


def _main():
    from cli_encoding import configure_stdio
    configure_stdio()
    ap = argparse.ArgumentParser(
        description="Commit C Phase 4: dry-run-first legacy MLB identity re-stamp.")
    ap.add_argument("--apply", action="store_true",
                    help="Write the re-stamp (default: dry-run preview, no writes).")
    ap.add_argument("--odds", action="store_true",
                    help="Also re-stamp odds_line (the calibration/CLV corpus).")
    ap.add_argument("--samples", type=int, default=25,
                    help="How many old->new diff rows to print (default 25).")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    if not db_store.enabled():
        raise SystemExit("SQL is not configured (SQL_* secrets) — nothing to re-stamp.")
    _report(restamp(dry_run=not args.apply, do_odds=args.odds, samples=args.samples))


if __name__ == "__main__":
    _main()
