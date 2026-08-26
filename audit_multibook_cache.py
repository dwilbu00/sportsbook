"""
Audit the raw odds cache to confirm a multibook pull actually captured what we
think — Pinnacle + the soft US books + MLB props, across 2024-2026, for BOTH the
closing and early snapshots. READ-ONLY: opens each cache file, tallies coverage,
writes nothing. Run this on the machine that ran pull_multibook_odds.py BEFORE
building/running the per-book warehouse ingester.

WHY: the R2 sharp-staleness edge needs per-book prices incl. a sharp reference
(Pinnacle). If the pull silently landed DK-only (or missed props / a season /
the early snapshot), the whole migration is moot — catch it here, cheaply.

Cache format (from odds_client._write_cache / _read_cache_permanent):
  file        = {"cached_at": <epoch>, "data": <payload>}
  historical  -> payload = {"data": <games>, "timestamp": <snapshot ts>}   (double-nested)
  live        -> payload = <games>                                          (single-nested)
  <games>     = list of game dicts (featured/team) OR one game dict (event/props)
  game        = {id, sport_key, commence_time, home_team, away_team, bookmakers:[...]}
  bookmaker   = {key, title, markets:[{key, outcomes:[...]}]}
  outcome     = team: {name[, point], price} | prop: {name:Over/Under, description:<player>, point, price}

USAGE
    python audit_multibook_cache.py                       # ./cache, sport=baseball_mlb
    python audit_multibook_cache.py --cache-dir cache --sport baseball_mlb
    python audit_multibook_cache.py --sample              # also dump one MLB game's books/markets
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def _iter_games(outer):
    """Yield (game_dict, snapshot_ts) from one loaded cache file, handling the
    double-nested historical wrapper and the single-nested live wrapper."""
    payload = outer.get("data") if isinstance(outer, dict) else None
    ts = None
    body = payload
    if isinstance(payload, dict) and "timestamp" in payload and "data" in payload:
        ts = payload.get("timestamp")           # historical: {data, timestamp}
        body = payload.get("data")
    if isinstance(body, list):
        games = body
    elif isinstance(body, dict) and "bookmakers" in body:
        games = [body]
    else:
        games = []
    for g in games:
        if isinstance(g, dict):
            yield g, ts


def _is_prop_market(key):
    return key.startswith(("batter_", "pitcher_", "player_"))


def main():
    p = argparse.ArgumentParser(description="Audit the raw multibook odds cache (read-only).")
    p.add_argument("--cache-dir", default="cache", help="Cache directory (default ./cache).")
    p.add_argument("--sport", default="baseball_mlb", help="Sport key to focus the MLB-style report on.")
    p.add_argument("--sample", action="store_true", help="Dump one focus-sport game's books/markets as an example.")
    args = p.parse_args()

    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.json")))
    if not files:
        print(f"No *.json files in {args.cache_dir!r}. Wrong directory?")
        return

    books_all = Counter()                 # bookmaker key -> occurrences (all sports)
    books_focus = Counter()               # bookmaker key -> occurrences (focus sport only)
    sport_events = defaultdict(set)       # sport_key -> {event_id}
    focus_events = set()                  # focus-sport event ids
    focus_team_events = set()             # focus events carrying a team market
    focus_prop_events = set()             # focus events carrying a prop market
    focus_prop_markets = Counter()        # prop market key -> occurrences (focus)
    focus_books_by_event = defaultdict(set)   # event_id -> {book keys} (focus, for the DK+Pinnacle join)
    focus_dates = []                      # commence_time date strings (focus)
    focus_snap_hours = Counter()          # snapshot-ts hour bucket (focus) -> close/early split
    parse_errors = 0
    sample_shown = False

    for i, path in enumerate(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                outer = json.load(f)
        except (json.JSONDecodeError, OSError):
            parse_errors += 1
            continue
        for g, ts in _iter_games(outer):
            sk = g.get("sport_key") or "?"
            eid = g.get("id") or ""
            sport_events[sk].add(eid)
            bookmakers = g.get("bookmakers") or []
            is_focus = (sk == args.sport)
            if is_focus:
                focus_events.add(eid)
                if g.get("commence_time"):
                    focus_dates.append(str(g["commence_time"])[:10])
                if ts:
                    hh = str(ts)[11:13]           # UTC hour of the snapshot ts
                    focus_snap_hours[hh] += 1
            for bk in bookmakers:
                key = bk.get("key") or "?"
                books_all[key] += 1
                markets = bk.get("markets") or []
                if is_focus:
                    books_focus[key] += 1
                    focus_books_by_event[eid].add(key)
                    for m in markets:
                        mk = m.get("key") or ""
                        if _is_prop_market(mk):
                            focus_prop_markets[mk] += 1
                            focus_prop_events.add(eid)
                        elif mk in ("h2h", "spreads", "totals"):
                            focus_team_events.add(eid)
            if args.sample and is_focus and not sample_shown and bookmakers:
                sample_shown = True
                print("\n--- SAMPLE focus-sport game ---")
                print(f"  event {eid}  {g.get('away_team')} @ {g.get('home_team')}  "
                      f"commence={g.get('commence_time')}  snapshot_ts={ts}")
                for bk in bookmakers[:6]:
                    mks = ", ".join(sorted({m.get('key') for m in (bk.get('markets') or [])}))
                    print(f"    book={bk.get('key'):<12} markets: {mks}")
                print("--- end sample ---\n")

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  Multibook cache audit — {args.cache_dir}\n{'='*70}")
    print(f"  files scanned: {len(files):,}  ({parse_errors} unparseable)")

    print(f"\n  Bookmakers (ALL sports) — occurrences:")
    for k, n in books_all.most_common():
        print(f"    {k:<16} {n:>10,}")

    print(f"\n  Games by sport (distinct event ids):")
    for sk, ev in sorted(sport_events.items(), key=lambda x: -len(x[1])):
        print(f"    {sk:<22} {len(ev):>8,}")

    print(f"\n  --- focus sport: {args.sport} ---")
    print(f"  distinct events:            {len(focus_events):,}")
    print(f"  events with a team market:  {len(focus_team_events):,}")
    print(f"  events with a prop market:  {len(focus_prop_events):,}")
    if focus_dates:
        print(f"  commence date range:        {min(focus_dates)} .. {max(focus_dates)}")
        yr = Counter(d[:4] for d in focus_dates)
        print(f"  by year (game-appearances): {dict(sorted(yr.items()))}")

    print(f"\n  Bookmakers ({args.sport} only) — occurrences:")
    if not books_focus:
        print("    (none — the focus sport is not in this cache)")
    for k, n in books_focus.most_common():
        tag = "  <-- SHARP REF" if k == "pinnacle" else ""
        print(f"    {k:<16} {n:>10,}{tag}")

    # The R2-critical join: how many focus events carry BOTH DraftKings and Pinnacle
    both = sum(1 for ev, bset in focus_books_by_event.items()
               if "draftkings" in bset and "pinnacle" in bset)
    dk_only = sum(1 for bset in focus_books_by_event.values()
                  if "draftkings" in bset and "pinnacle" not in bset)
    print(f"\n  R2 join readiness ({args.sport}):")
    print(f"    events with BOTH draftkings + pinnacle: {both:,}")
    print(f"    events with draftkings but NO pinnacle: {dk_only:,}")

    if focus_prop_markets:
        print(f"\n  Prop markets present ({args.sport}):")
        for k, n in focus_prop_markets.most_common():
            print(f"    {k:<24} {n:>10,}")
    else:
        print(f"\n  Prop markets present ({args.sport}): NONE")

    if focus_snap_hours:
        print(f"\n  Snapshot-ts UTC hour histogram ({args.sport}) "
              f"— early≈13:00Z, close≈evening game times:")
        for hh, n in sorted(focus_snap_hours.items()):
            tag = "  (early?)" if hh == "13" else ""
            print(f"    {hh}:00Z  {n:>8,}{tag}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    ok = books_focus.get("pinnacle", 0) > 0 and len(focus_events) > 0
    if not books_focus:
        print("  VERDICT: focus sport absent from this cache — wrong machine / cache-dir?")
    elif books_focus.get("pinnacle", 0) == 0:
        print("  VERDICT: NO PINNACLE for the focus sport — this looks DK-only (the R2\n"
              "           sharp reference is missing). Do NOT migrate against this cache.")
    elif both == 0:
        print("  VERDICT: Pinnacle present but NO event has both DK + Pinnacle — check the\n"
              "           snapshot alignment before building the R2 join.")
    else:
        print(f"  VERDICT: looks good — {both:,} events carry both DK + Pinnacle. Multibook\n"
              f"           capture confirmed; safe to build the per-book ingester.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
