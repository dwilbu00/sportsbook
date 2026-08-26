"""
Ingest the raw multibook odds cache into the PER-BOOK warehouse (DRY-RUN FIRST).

Reads the cache written by pull_multibook_odds.py and writes one odds_line row
per (bookmaker, market, point) into the warehouse, so a single snapshot holds
every book's price (incl. Pinnacle, the R2 sharp reference). Creditless — no Odds
API calls; it only reads the on-disk cache + the warehouse.

PARITY BY CONSTRUCTION: each book is fed as a SINGLE-BOOK payload through the
exact parser the live/backfill capture uses (odds_client.parse_game_odds /
parse_player_props + warehouse._emit_team_lines / _emit_prop_lines). Those
functions collapse "best-across-books"; with one book in, that degenerates to
that book — so a DraftKings row here is byte-identical to the old DK-only
capture. game_pk + player_mlb_id are stamped via warehouse._enrich_ids (the same
DH-safe resolver the backfill used).

PREREQUISITES (run in order):
  1. odds_line has bookmaker/region columns (sql/schema.sql ALTER — Phase 1 DDL).
  2. The OLD DraftKings-only MLB odds are DELETED (else write-once uq collides and
     the new snapshot is silently dropped). Delete + ingest back-to-back.

WRITE MODEL: capture_odds_snapshot is write-once (uq = sport,game_date,event_id,
kind,snapshot_hour). One snapshot per (game, kind, snapshot_hour) carries all
books' lines. Re-running skips already-written snapshots (resumable). source =
multibook_close / multibook_open (by proximity of the snapshot ts to commence).

USAGE
    python ingest_multibook_cache.py                       # DRY-RUN coverage (all)
    python ingest_multibook_cache.py --limit 200           # fast dry-run sample
    python ingest_multibook_cache.py --apply --yes         # write to the warehouse
    python ingest_multibook_cache.py --cache-dir cache --sport baseball_mlb --apply --yes
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

# props are keyed by these market prefixes in the Odds API payload
_PROP_PREFIXES = ("batter_", "pitcher_", "player_")
# The pull ran two passes: CLOSE (snapshot ts = each game's commence) and EARLY
# (snapshot ts = a fixed morning time, ~12:00Z per the cache audit). BUT the
# historical featured endpoint returns the WHOLE SLATE at each ts — so a close-pass
# fetch at game A's first pitch also contains the later games at their as-of-then
# (intraday) prices. We keep only each game's TRUE close (ts within a window of its
# OWN commence) and its morning open (ts at the fixed early hour); the intraday
# pre-close snapshots from other games' close-pass files are dropped (the raw cache
# retains them for a future line-movement ingest if we ever want it).
_DEFAULT_EARLY_HOUR = 12          # UTC hour of the early pull (audit: 12:00Z cluster)
_DEFAULT_CLOSE_WINDOW_MIN = 30    # |snapshot ts - commence| <= this => that game's close


def _iter_cache_games(cache_dir, sport):
    """Yield (game_dict, snapshot_ts, path) for the target sport, handling the
    double-nested historical wrapper {cached_at, data:{data, timestamp}} and the
    single-nested live wrapper {cached_at, data}."""
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                outer = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        payload = outer.get("data") if isinstance(outer, dict) else None
        ts, body = None, payload
        if isinstance(payload, dict) and "timestamp" in payload and "data" in payload:
            ts = payload.get("timestamp")
            body = payload.get("data")
        if isinstance(body, list):
            games = body
        elif isinstance(body, dict) and "bookmakers" in body:
            games = [body]
        else:
            games = []
        for g in games:
            if isinstance(g, dict) and g.get("sport_key") == sport:
                yield g, ts, path


def _game_market_keys(game):
    keys = set()
    for bk in game.get("bookmakers") or []:
        for m in bk.get("markets") or []:
            if m.get("key"):
                keys.add(m["key"])
    return keys


def _per_book_lines(game, kind):
    """One line dict per (bookmaker, market, outcome/point), tagged with the book.

    Feeds each book as a single-book payload through the SAME parser the warehouse
    capture uses, so a DK row is byte-identical to the legacy DK-only capture."""
    import warehouse as wh
    from odds_client import parse_game_odds, parse_player_props

    out = []
    for bk in game.get("bookmakers") or []:
        book_key = bk.get("key")
        if not book_key:
            continue
        one_book_game = {**game, "bookmakers": [bk]}
        book_lines = []
        try:
            if kind == "props":
                wh._emit_prop_lines(parse_player_props(one_book_game), book_lines)
            else:
                wh._emit_team_lines(parse_game_odds(one_book_game), book_lines)
        except Exception:
            continue
        for ln in book_lines:
            ln["bookmaker"] = book_key
            # region is not derivable per-book from the payload (regions was a
            # request param, not tagged per book) -> left NULL; backfill later
            # from a book->region map only if R2 needs it.
            ln["region"] = None
        out.extend(book_lines)
    return out


def _classify_snapshot(snapshot_ts, commence, early_hour, close_window_min):
    """'multibook_close' when the snapshot ts is within close_window_min of first
    pitch (this game's true close), 'multibook_open' when the ts is at the fixed
    early-pull hour (the morning line), else None to SKIP — an intraday pre-close
    snapshot that landed in another game's whole-slate close-pass file."""
    import warehouse as wh
    sdt = wh._parse_utc(snapshot_ts)
    cdt = wh._parse_utc(commence)
    if not sdt or not cdt:
        return None
    if abs((cdt - sdt).total_seconds()) <= close_window_min * 60:
        return "multibook_close"
    if sdt.hour == early_hour:
        return "multibook_open"
    return None


def _enrich_lines_fast(sport, meta, lines, gpk_cache, id_cache):
    """Fast id/game_pk stamping for a snapshot's lines (MLB only), the bulk-ingest
    counterpart to warehouse._enrich_ids.

    The heavy per-snapshot cost in _enrich_ids is (a) find_game_pk_by_commence per
    snapshot and (b) the game-context entity_resolver PER PROP PLAYER. But every
    line of one event (team + props, close + open) shares ONE game, and the odds
    feed's player ids come from the same SFBB cross-map the rest of the system uses.
    So: resolve game_pk ONCE per event (memoized in gpk_cache, DH-safe via commence),
    memoize player_mlb_id per (name, teams) in id_cache — a player recurs ~26x in one
    snapshot (books x over/under) and again in the close+open snapshots — and take
    ids straight from the in-memory SFBB map. Fail-open (a miss leaves the field
    None -> name-based join, exactly as before)."""
    try:
        if not (sport or "").startswith("baseball"):
            return meta, lines
        import player_id_map
        import mlb_warehouse
        home, away = meta.get("home"), meta.get("away")
        meta["home_code"] = player_id_map.team_code_for_name(home)
        meta["away_code"] = player_id_map.team_code_for_name(away)
        eid = meta.get("event_id")
        if eid in gpk_cache:
            gpk = gpk_cache[eid]
        else:
            hid = mlb_warehouse.team_id_for_name_tolerant(home) if home else None
            aid = mlb_warehouse.team_id_for_name_tolerant(away) if away else None
            commence = meta.get("commence_time")
            gpk = (mlb_warehouse.find_game_pk_by_commence(hid, aid, commence)
                   if hid and aid and commence else None)
            gpk_cache[eid] = gpk
        teams = (home, away)
        for ln in lines:
            if (ln.get("bet_type") or "") == "player_prop":
                nm = ln.get("player")
                ck = (nm, home, away)
                if ck not in id_cache:
                    id_cache[ck] = player_id_map.mlb_id_for_name(nm, teams=teams)
                ln["player_mlb_id"] = id_cache[ck]
            else:
                ln["team_code"] = player_id_map.team_code_for_name(ln.get("selection"))
            ln["game_pk"] = gpk
    except Exception:
        pass
    return meta, lines


def main():
    p = argparse.ArgumentParser(description="Ingest the multibook cache into the per-book warehouse (dry-run first).")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--sport", default="baseball_mlb")
    p.add_argument("--apply", action="store_true", help="Write to the warehouse (default = dry-run).")
    p.add_argument("--yes", action="store_true", help="Required with --apply.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N snapshots (0 = all; use for a fast dry-run sample).")
    p.add_argument("--no-enrich", action="store_true",
                   help="Skip player_mlb_id/game_pk resolution — a FAST structural "
                        "preview (books, source split, line counts, dates) without "
                        "the id/game_pk lookups. Dry-run only.")
    p.add_argument("--full-enrich", action="store_true",
                   help="Use the thorough warehouse._enrich_ids (game-context entity "
                        "resolver) instead of the fast SFBB path — slower; only if the "
                        "fast player_mlb_id coverage comes back materially lower.")
    p.add_argument("--early-hour", type=int, default=_DEFAULT_EARLY_HOUR,
                   help=f"UTC hour of the early/open pull (default {_DEFAULT_EARLY_HOUR}).")
    p.add_argument("--close-window-min", type=int, default=_DEFAULT_CLOSE_WINDOW_MIN,
                   help=f"|snapshot ts - commence| <= this many minutes counts as that "
                        f"game's close (default {_DEFAULT_CLOSE_WINDOW_MIN}).")
    p.add_argument("--progress-every", type=int, default=500)
    args = p.parse_args()

    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    import warehouse as wh
    import db_store
    db_store.promote_secrets_from_toml()
    if args.no_enrich and args.apply:
        print("--no-enrich is a preview-only mode (writes need the ids/game_pk). "
              "Drop --no-enrich for --apply.")
        sys.exit(1)
    if args.apply:
        if not args.yes:
            print("--apply requires --yes (double-confirm). Nothing written.")
            sys.exit(1)
        if not db_store.enabled():
            print("SQL not configured (db_store.enabled() is False). Aborting.")
            sys.exit(1)

    mode = "APPLY (writing)" if args.apply else "DRY-RUN (no writes)"
    print(f"\n{'='*70}\n  Multibook ingest — {args.cache_dir} -> warehouse  [{mode}]\n{'='*70}")

    snaps_by_kind = Counter()
    source_ct = Counter()
    books_ct = Counter()
    lines_total = 0
    gpk_lines = 0
    prop_lines = 0
    prop_gpk = 0
    prop_mlbid = 0
    intraday_skipped = 0
    open_hours = Counter()    # ts-hour of OPEN snapshots (confidence: expect ~all early_hour)
    close_delta = Counter()   # |ts - commence| bucket of CLOSE snapshots (expect ~all 0-5m)
    written = skipped = errors = 0
    dates = []
    seen = set()          # (event_id, kind, snapshot_hour) — de-dupe within this run
    gpk_cache = {}        # event_id -> game_pk (resolve once per game, DH-safe)
    id_cache = {}         # (name, home, away) -> player_mlb_id (recurs ~26x/snapshot)
    n = 0

    for game, ts, _path in _iter_cache_games(args.cache_dir, args.sport):
        commence = game.get("commence_time")
        if not commence:
            continue
        mkeys = _game_market_keys(game)
        if not mkeys:
            continue
        kind = wh._kind_for_markets(",".join(sorted(mkeys)))
        if kind == "alt":
            continue                                     # alternates weren't pulled
        # Classify BEFORE the expensive parse/enrich so intraday snapshots (a game
        # seen pre-close in another game's whole-slate close-pass file) are dropped
        # cheaply — we keep only each game's true close + its morning open.
        source = _classify_snapshot(ts, commence, args.early_hour, args.close_window_min)
        if source is None:
            intraday_skipped += 1
            continue
        snapshot_hour = wh._hour_bucket(ts or commence)
        key = (game.get("id"), kind, snapshot_hour)
        if key in seen:
            continue
        seen.add(key)

        lines = _per_book_lines(game, kind)
        if not lines:
            continue
        meta = {
            "sport": args.sport, "game_date": commence[:10],
            "event_id": game.get("id"), "kind": kind,
            "snapshot_hour": snapshot_hour, "captured_at": ts,
            "commence_time": commence,
            "home": game.get("home_team"), "away": game.get("away_team"),
            "regions": "us,eu", "markets": ",".join(sorted(mkeys)),
            "bookmakers": "multibook", "source": source,
        }
        # Stamp player_mlb_id + DH-safe game_pk + team codes. Fast path (default):
        # game_pk once per event (cached) + in-memory SFBB ids. --full-enrich uses
        # the thorough game-context resolver. --no-enrich skips it (preview only).
        if args.no_enrich:
            pass
        elif args.full_enrich:
            meta, lines = wh._enrich_ids(args.sport, meta, lines)
        else:
            meta, lines = _enrich_lines_fast(args.sport, meta, lines, gpk_cache, id_cache)

        snaps_by_kind[kind] += 1
        source_ct[source] += 1
        dates.append(meta["game_date"])
        # Confidence diagnostics for the close/open call (see the report).
        _sdt, _cdt = wh._parse_utc(ts), wh._parse_utc(commence)
        if source == "multibook_open" and _sdt:
            open_hours[_sdt.hour] += 1
        elif source == "multibook_close" and _sdt and _cdt:
            _dm = abs((_cdt - _sdt).total_seconds()) / 60
            close_delta["0-5m" if _dm <= 5 else "5-15m" if _dm <= 15
                        else "15-30m" if _dm <= 30 else ">30m"] += 1
        for ln in lines:
            lines_total += 1
            books_ct[ln.get("bookmaker")] += 1
            is_prop = (ln.get("bet_type") == "player_prop")
            if is_prop:
                prop_lines += 1
                if ln.get("player_mlb_id"):
                    prop_mlbid += 1
            if ln.get("game_pk"):
                gpk_lines += 1
                if is_prop:
                    prop_gpk += 1

        if args.apply:
            try:
                ok = db_store.capture_odds_snapshot(meta, lines)
                written += 1 if ok else 0
                skipped += 0 if ok else 1
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"  [err] {meta['event_id']} {kind}: {type(exc).__name__} ({exc})")

        n += 1
        if args.progress_every and n % args.progress_every == 0:
            print(f"  ...{n:,} snapshots processed "
                  f"({'written ' + format(written, ',') if args.apply else 'dry-run'})")
        if args.limit and n >= args.limit:
            print(f"  [limit] stopping at {args.limit} snapshots.")
            break

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n  snapshots: {n:,}   (team={snaps_by_kind.get('team', 0):,}, "
          f"props={snaps_by_kind.get('props', 0):,})")
    print(f"  source split: multibook_close={source_ct.get('multibook_close', 0):,}  "
          f"multibook_open={source_ct.get('multibook_open', 0):,}")
    print(f"  intraday snapshots skipped (kept in cache): {intraday_skipped:,}")
    # Confidence check on the close/open call:
    if open_hours:
        print(f"  OPEN ts-hour (confidence: expect ~all {args.early_hour:02d}Z): "
              f"{dict(sorted(open_hours.items()))}")
    if close_delta:
        order = ["0-5m", "5-15m", "15-30m", ">30m"]
        print(f"  CLOSE |ts-commence| (confidence: expect ~all 0-5m): "
              f"{{{', '.join(f'{b}: {close_delta[b]}' for b in order if close_delta[b])}}}")
    if dates:
        yr = Counter(d[:4] for d in dates)
        print(f"  snapshot date range: {min(dates)} .. {max(dates)}   by year: {dict(sorted(yr.items()))}")
    print(f"  odds_line rows: {lines_total:,}   (prop lines: {prop_lines:,})")
    if args.no_enrich:
        print(f"  game_pk / mlb_id coverage: (skipped — --no-enrich; drop it for a real % on a small --limit)")
    elif lines_total:
        print(f"  game_pk coverage: {gpk_lines:,}/{lines_total:,} lines "
              f"({100*gpk_lines/lines_total:.1f}%)"
              + (f"   props: {prop_gpk:,}/{prop_lines:,} "
                 f"({100*prop_gpk/prop_lines:.1f}%)" if prop_lines else ""))
        if prop_lines:
            print(f"  player_mlb_id coverage (props): {prop_mlbid:,}/{prop_lines:,} "
                  f"({100*prop_mlbid/prop_lines:.1f}%)"
                  + ("" if args.full_enrich else "  [fast SFBB path — use --full-enrich if low]"))
    print(f"\n  Bookmakers written (occurrences):")
    for k, v in books_ct.most_common():
        tag = "  <-- SHARP REF" if k == "pinnacle" else ""
        print(f"    {str(k):<16} {v:>10,}{tag}")

    if args.apply:
        print(f"\n  WROTE {written:,} snapshots  (skipped {skipped:,} already-present / "
              f"{errors:,} errors).")
    else:
        print(f"\n  DRY-RUN — nothing written. Re-run with --apply --yes to ingest.")
        print(f"  (Delete the old DK-only MLB odds FIRST, or write-once collisions "
              f"will drop every new snapshot.)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
