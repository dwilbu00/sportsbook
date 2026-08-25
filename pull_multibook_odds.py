"""
Pull ALL-BOOKS historical odds into the local raw cache — parallelized.

WHY THIS EXISTS
---------------
Our warehouse currently holds DraftKings-only historical odds (the backfill was
run with bookmakers=[draftkings]). To test the real edge thesis — that DK is a
soft/slow book we beat by referencing a SHARP book (Pinnacle) and catching DK
staleness (edge = DK_odds / sharp_vig_free_odds - 1) — we need the per-book
prices of BOTH the sharp reference AND the soft books, historically.

The Odds API returns EVERY book in a region in one call and charges per REGION,
not per book. Pinnacle (the sharp anchor) sits in the `eu` region; DK / FanDuel /
BetMGM etc. are `us`. So we pull `regions=us,eu` to capture the sharp line + all
US soft books in one shot. Cost = 10 x markets x 2 regions per call.

WHAT IT DOES
------------
Calls the odds_client HISTORICAL endpoints with bookmakers=None (all books) for
every game in a season, at the chosen snapshot (close = per-game tip-off).
odds_client caches every raw response PERMANENTLY under a per-request key that
includes the regions + (empty) bookmaker set — a DIFFERENT key from the old
DK-only pull, so nothing collides and re-runs of THIS pull are free. It does NOT
write the local labeled store or the warehouse — the raw cache is the landing
zone; per-book warehouse ingestion is a separate (creditless) build.

    ⚠ BACK UP the cache/ folder after this runs. Those JSONs are the
      irreplaceable multi-book payloads; once your monthly credits revert you
      cannot re-fetch them.

USAGE
-----
    # See the plan + exact credit cost, spend nothing:
    python pull_multibook_odds.py --sport mlb --season 2025 --dry-run

    # Real pull, hard-capped at 19000 credits, 8 parallel workers:
    python pull_multibook_odds.py --sport mlb --season 2025 --max-credits 19000

    # Team markets only (cheapest), a date range:
    python pull_multibook_odds.py --sport mlb --start 2025-04-01 --end 2025-04-30 \
        --props "" --max-credits 3000

Idempotent: cached snapshots re-read for 0 credits, so a re-run only fetches new
data. Safe to Ctrl-C and resume.
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the PROVEN discovery + snapshot-timing + config helpers from the backfill
# so this script can't drift from the tested behavior.
from backfill_historical_odds import (
    load_config,
    collect_completed_games,
    _find_sampled,
    _snap_ts_for_date,
    SPORT_MAP,
    PROPS_MIN_DATE,
    DEFAULT_EARLY_TIME,
    BACKFILL_PROPS_BY_SPORT,
)
from odds_client import (
    get_historical_odds,
    get_historical_events,
    get_historical_event_odds,
    is_historical_odds_cached,
    is_historical_events_cached,
    is_historical_event_cached,
    get_remaining_credits,
)

try:
    from cli_encoding import configure_stdio
    configure_stdio()
except Exception:
    pass


def _count(csv):
    return len([x for x in (csv or "").split(",") if x.strip()])


def main():
    p = argparse.ArgumentParser(
        description="Pull ALL-BOOKS (incl. Pinnacle sharp reference) historical "
                    "odds into the local raw cache, parallelized.")
    p.add_argument("--sport", required=True, choices=sorted(SPORT_MAP))
    p.add_argument("--season", type=int, default=None,
                   help="Season year (e.g. 2025). Omit to use --start/--end over "
                        "all discoverable games.")
    p.add_argument("--start", default=None, help="ISO date lower bound (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="ISO date upper bound (YYYY-MM-DD).")
    p.add_argument("--markets", default="h2h,spreads,totals",
                   help="Featured/team markets (comma-sep). '' to skip team.")
    p.add_argument("--props", default=None,
                   help="Prop markets (comma-sep). Default: the sport's backfill "
                        "prop set. '' to skip props.")
    p.add_argument("--regions", default="us,eu",
                   help="Odds API regions. Default 'us,eu' to include Pinnacle "
                        "(eu = the SHARP reference) alongside US soft books. Cost "
                        "scales with region COUNT.")
    p.add_argument("--snapshot", choices=["close", "early"], default="close",
                   help="close = per-game tip-off (true closing line, the "
                        "CLV-relevant snapshot); early = a fixed morning line.")
    p.add_argument("--featured-cadence", choices=["commence", "daily"],
                   default="commence",
                   help="TEAM (featured) snapshot cadence. commence = one per game "
                        "tip-off (true close; ~10x more calls). daily = one per DATE "
                        "at the day's first tip (near-close; ~10x cheaper). Props are "
                        "always per-game regardless.")
    p.add_argument("--early-time", default=DEFAULT_EARLY_TIME,
                   help=f"UTC HH:MM for --snapshot early (default {DEFAULT_EARLY_TIME}).")
    p.add_argument("--max-credits", type=int, default=19000,
                   help="Hard cap on credits this run may spend. Default 19000.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel fetch workers. Default 8. Each worker is one "
                        "in-flight API call; keep modest to respect rate limits.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan + estimated cost without calling the API.")
    args = p.parse_args()

    espn_sport, espn_league, sport_key = SPORT_MAP[args.sport]
    n_regions = _count(args.regions)
    props = (args.props if args.props is not None
             else ",".join(BACKFILL_PROPS_BY_SPORT.get(sport_key, [])))
    n_feat = _count(args.markets)
    n_prop = _count(props)
    feat_cost = 10 * n_feat * n_regions
    prop_cost = 10 * n_prop * n_regions
    cfg = load_config()
    api_key = cfg["odds_api_key"]

    print(f"\n=== Multi-book raw pull: {sport_key} ===")
    print(f"  Regions: {args.regions}  ({n_regions} region(s); 'eu' = Pinnacle sharp ref)")
    print(f"  Team markets: {args.markets or '(none)'}  ({feat_cost} cr/snapshot)")
    print(f"  Prop markets: {props or '(none)'}  ({prop_cost} cr/game)")
    print(f"  Snapshot: {args.snapshot}   Budget cap: {args.max_credits} cr   "
          f"Workers: {args.workers}")

    # ── Discover games ───────────────────────────────────────────────────────
    print("\n=== Discovering games (ESPN schedules) ===")
    games = collect_completed_games(espn_sport, espn_league, args.season)
    for g in games:
        # Snapshot timestamp per game: close = its own tip-off; early = fixed AM.
        d10 = (g.get("date") or "")[:10]
        g["_snap_ts"] = (g["date"] if args.snapshot == "close"
                         else _snap_ts_for_date(d10, args.early_time))
    if args.start or args.end:
        lo, hi = (args.start or "0000-00-00"), (args.end or "9999-99-99")
        games = [g for g in games if lo <= (g.get("date") or "")[:10] <= hi]
    games = [g for g in games if g.get("date") and g.get("home_team") and g.get("away_team")]
    games.sort(key=lambda g: g.get("date") or "", reverse=True)  # freshest first
    if not games:
        print("  No games found. Nothing to pull.")
        return
    print(f"  {len(games)} games in scope.")

    # ── Plan featured snapshots (dedup by snapshot ts) ───────────────────────
    feat_ts = {}   # snap_ts -> [games covered by that snapshot]
    if n_feat:
        if args.featured_cadence == "daily":
            by_date = {}
            for g in games:
                by_date.setdefault((g.get("date") or "")[:10], []).append(g)
            for d10, gs in by_date.items():
                ts = min(g["_snap_ts"] for g in gs)  # day's first tip (near-close)
                feat_ts[ts] = gs
        else:  # commence — one snapshot per game tip-off (true close)
            for g in games:
                feat_ts.setdefault(g["_snap_ts"], []).append(g)
    feat_todo = [ts for ts in feat_ts
                 if not is_historical_odds_cached(
                     sport_key, ts, regions=args.regions,
                     markets=args.markets, bookmakers=None)]

    # ── Plan prop games (gated by the props floor) ───────────────────────────
    prop_games = []
    prop_floor_skipped = 0
    if n_prop:
        for g in games:
            if (g.get("date") or "")[:10] < PROPS_MIN_DATE:
                prop_floor_skipped += 1
                continue
            prop_games.append(g)

    feat_credits = len(feat_todo) * feat_cost
    # We can't know prop cached-state until we have event IDs (harvested from
    # featured), so estimate props on the full (floor-passed) set; the per-game
    # cache check at fetch time makes already-pulled games free.
    prop_credits_est = len(prop_games) * prop_cost

    print(f"\n  Featured snapshots: {len(feat_ts)} total, {len(feat_todo)} uncached "
          f"(~{feat_credits} cr)")
    if prop_floor_skipped:
        print(f"  [props-floor] {prop_floor_skipped} game(s) before {PROPS_MIN_DATE} "
              f"skipped (Odds API has no historical props there).")
    print(f"  Prop games: {len(prop_games)}  (<= ~{prop_credits_est} cr; "
          f"cached games are free)")

    # ── Budget trim (featured kept; props fill remaining budget) ─────────────
    budget_for_props = args.max_credits - feat_credits
    max_prop_games = (budget_for_props // prop_cost) if prop_cost else 0
    if prop_cost and len(prop_games) > max_prop_games:
        print(f"  Budget allows {max(max_prop_games, 0)} prop games — trimming "
              f"{len(prop_games) - max(max_prop_games, 0)} oldest to fit "
              f"{args.max_credits} cr.")
        prop_games = prop_games[:max(max_prop_games, 0)]
    planned = feat_credits + len(prop_games) * prop_cost
    bal = get_remaining_credits()
    print(f"\n  THIS RUN (worst case): {len(feat_todo)} featured + {len(prop_games)} "
          f"prop games ≈ {planned} cr."
          + (f"  Account balance: {bal}." if bal is not None else ""))

    if args.dry_run:
        print("\n  [dry-run] No API calls made. Re-run without --dry-run to fetch.\n")
        return
    if planned <= 0:
        print("\n  Nothing to fetch (all cached or empty plan). Done.\n")
        return
    if feat_credits > args.max_credits:
        print("\n  [abort] Featured alone exceeds the budget cap. Raise "
              "--max-credits or narrow the scope.\n")
        return

    lock = threading.Lock()
    spent = {"cr": 0}
    event_ids = {}   # (d10, home, away) -> event_id, harvested from featured

    def _gkey(g):
        return ((g.get("date") or "")[:10], g.get("home_team"), g.get("away_team"))

    # ── Phase 1: FEATURED (parallel) — also harvests event IDs for props ─────
    def _fetch_featured(ts):
        try:
            slate, _snap = get_historical_odds(
                api_key, sport_key, date=ts, regions=args.regions,
                markets=args.markets, bookmakers=None)
            return ts, (slate or [])
        except Exception as e:
            return ts, e

    if feat_todo:
        print(f"\n=== Phase 1: featured — {len(feat_todo)} snapshots "
              f"({args.workers} workers) ===")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_fetch_featured, ts): ts for ts in feat_todo}
            for fut in as_completed(futs):
                ts, res = fut.result()
                done += 1
                if isinstance(res, Exception):
                    print(f"  [warn] featured {ts}: {res}")
                    continue
                with lock:
                    spent["cr"] += feat_cost
                # Harvest event IDs by matching the slate to our scoped games.
                date_games = feat_ts.get(ts, [])
                for api_game in res:
                    g = _find_sampled(api_game.get("home_team"),
                                      api_game.get("away_team"), date_games)
                    if g and api_game.get("id"):
                        event_ids[_gkey(g)] = api_game["id"]
                if done % 25 == 0 or done == len(feat_todo):
                    print(f"  [featured {done}/{len(feat_todo)}] ~{spent['cr']} cr "
                          f"(bal {get_remaining_credits()}), {len(event_ids)} ids")

    # Also harvest IDs already cached from a prior featured pull (free), so props
    # can proceed even when featured was fully cached this run.
    if n_prop and n_feat:
        for ts, gs in feat_ts.items():
            if any(_gkey(g) not in event_ids for g in gs) and is_historical_odds_cached(
                    sport_key, ts, regions=args.regions, markets=args.markets,
                    bookmakers=None):
                try:
                    slate, _ = get_historical_odds(
                        api_key, sport_key, date=ts, regions=args.regions,
                        markets=args.markets, bookmakers=None)   # cached → free
                    for api_game in (slate or []):
                        g = _find_sampled(api_game.get("home_team"),
                                          api_game.get("away_team"), gs)
                        if g and api_game.get("id"):
                            event_ids.setdefault(_gkey(g), api_game["id"])
                except Exception:
                    pass

    # ── Phase 1b: event-ID lookups when featured is skipped (props-only) ─────
    if n_prop and not n_feat:
        id_dates = sorted({g["_snap_ts"] for g in prop_games})
        print(f"\n=== Phase 1b: event-ID lookups — {len(id_dates)} dates ===")
        for ts in id_dates:
            if not is_historical_events_cached(sport_key, ts):
                with lock:
                    if spent["cr"] + 1 > args.max_credits:
                        print("  [stop] budget reached during ID lookup.")
                        break
                    spent["cr"] += 1
            try:
                events, _ = get_historical_events(api_key, sport_key, date=ts)
            except Exception as e:
                print(f"  [warn] events {ts}: {e}")
                continue
            date_games = [g for g in prop_games if g["_snap_ts"] == ts]
            for ev in events or []:
                g = _find_sampled(ev.get("home_team"), ev.get("away_team"), date_games)
                if g and ev.get("id"):
                    event_ids[_gkey(g)] = ev["id"]

    # ── Phase 2: PROPS (parallel, per game) ──────────────────────────────────
    def _fetch_props(g):
        eid = event_ids.get(_gkey(g))
        if not eid:
            return g, "no-event-id"
        ts = g["_snap_ts"]
        cached = is_historical_event_cached(
            sport_key, eid, ts, regions=args.regions, markets=props, bookmakers=None)
        try:
            get_historical_event_odds(
                api_key, sport_key, eid, date=ts, regions=args.regions,
                markets=props, bookmakers=None)
            return g, (0 if cached else prop_cost)
        except Exception as e:
            return g, e

    if n_prop and prop_games:
        print(f"\n=== Phase 2: props — {len(prop_games)} games "
              f"({args.workers} workers) ===")
        done = fetched = no_id = 0
        # Only submit games whose remaining cost fits the cap (cached = free).
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {}
            for g in prop_games:
                # Pre-check budget on the assumption of a fresh fetch; cached
                # games cost 0 and are always allowed.
                cached = event_ids.get(_gkey(g)) and is_historical_event_cached(
                    sport_key, event_ids[_gkey(g)], g["_snap_ts"],
                    regions=args.regions, markets=props, bookmakers=None)
                if not cached and spent["cr"] + prop_cost > args.max_credits:
                    continue  # would breach cap — skip (resume later)
                if not cached:
                    spent["cr"] += prop_cost   # reserve upfront (single-threaded here)
                futs[pool.submit(_fetch_props, g)] = g
            for fut in as_completed(futs):
                g, res = fut.result()
                done += 1
                if res == "no-event-id":
                    no_id += 1
                elif isinstance(res, Exception):
                    print(f"  [warn] props {_gkey(g)}: {res}")
                else:
                    fetched += 1
                if done % 50 == 0 or done == len(futs):
                    print(f"  [props {done}/{len(futs)}] ~{spent['cr']} cr "
                          f"(bal {get_remaining_credits()}), {fetched} fetched, "
                          f"{no_id} without an event id")

    print(f"\n=== Done. Spent ~{spent['cr']} cr this run "
          f"(account balance: {get_remaining_credits()}). ===")
    print("  Raw multi-book payloads are cached in ./cache/ (permanent).")
    print("  ⚠ BACK UP the cache/ folder — those JSONs cannot be re-fetched "
          "after your monthly credits revert.\n")


if __name__ == "__main__":
    sys.exit(main())
