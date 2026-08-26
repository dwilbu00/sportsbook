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
# snapshot taken >= this many hours before first pitch => the EARLY (open) pull;
# nearer than that => the CLOSE pull. The early pull used a fixed morning time
# (~12:00Z), the close pull used per-game commence, so a 3h margin separates them
# for every realistic MLB start time.
_OPEN_MIN_HOURS_BEFORE = 3.0


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


def _source_for(snapshot_ts, commence):
    """multibook_open (early) when the snapshot predates first pitch by >= the
    margin, else multibook_close."""
    import warehouse as wh
    sdt = wh._parse_utc(snapshot_ts)
    cdt = wh._parse_utc(commence)
    if sdt and cdt and (cdt - sdt).total_seconds() >= _OPEN_MIN_HOURS_BEFORE * 3600:
        return "multibook_open"
    return "multibook_close"


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
                        "the slow per-snapshot entity/game_pk lookups. Dry-run only.")
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
    written = skipped = errors = 0
    dates = []
    seen = set()          # (event_id, kind, snapshot_hour) — de-dupe within this run
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
        snapshot_hour = wh._hour_bucket(ts or commence)
        key = (game.get("id"), kind, snapshot_hour)
        if key in seen:
            continue
        seen.add(key)

        lines = _per_book_lines(game, kind)
        if not lines:
            continue
        source = _source_for(ts, commence)
        meta = {
            "sport": args.sport, "game_date": commence[:10],
            "event_id": game.get("id"), "kind": kind,
            "snapshot_hour": snapshot_hour, "captured_at": ts,
            "commence_time": commence,
            "home": game.get("home_team"), "away": game.get("away_team"),
            "regions": "us,eu", "markets": ",".join(sorted(mkeys)),
            "bookmakers": "multibook", "source": source,
        }
        # Stamp player_mlb_id + DH-safe game_pk + team codes (same resolver the
        # backfill used). Runs in dry-run too so the coverage % is real — unless
        # --no-enrich (fast structural preview: skip the slow per-snapshot lookups).
        if not args.no_enrich:
            meta, lines = wh._enrich_ids(args.sport, meta, lines)

        snaps_by_kind[kind] += 1
        source_ct[source] += 1
        dates.append(meta["game_date"])
        for ln in lines:
            lines_total += 1
            books_ct[ln.get("bookmaker")] += 1
            is_prop = (ln.get("bet_type") == "player_prop")
            if is_prop:
                prop_lines += 1
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
    if dates:
        yr = Counter(d[:4] for d in dates)
        print(f"  snapshot date range: {min(dates)} .. {max(dates)}   by year: {dict(sorted(yr.items()))}")
    print(f"  odds_line rows: {lines_total:,}   (prop lines: {prop_lines:,})")
    if args.no_enrich:
        print(f"  game_pk coverage: (skipped — --no-enrich; drop it for a real % on a small --limit)")
    elif lines_total:
        print(f"  game_pk coverage: {gpk_lines:,}/{lines_total:,} lines "
              f"({100*gpk_lines/lines_total:.1f}%)"
              + (f"   props: {prop_gpk:,}/{prop_lines:,} "
                 f"({100*prop_gpk/prop_lines:.1f}%)" if prop_lines else ""))
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
