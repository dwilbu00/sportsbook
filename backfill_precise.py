"""
PRECISE historical-odds backfill — Phase 1 (FETCH → cache only).

WHY THIS EXISTS
---------------
The old backfill_historical_odds.py samples the FEATURED slate endpoint at a flat
wall-clock time per date (~9am ET). That is why we could only ever find sparse
morning snapshots and never a clean EARLY+CLOSE pair per game: it was a
capture-method limitation, not a data-availability one.

This tool captures GAME-RELATIVE snapshots via the PER-EVENT historical endpoint
(`get_historical_event_odds`), which:
  * returns the ACTUAL snapshot timestamp at/before the requested moment,
  * exposes PERIOD markets (first-5-innings h2h/spreads/totals) the bulk endpoint
    cannot,
  * is PERMANENTLY cached by odds_client → idempotent, resumable, free re-runs.

For each regular-season final game (2024/2025/2026) it fetches, per market group:
  * team+F5 markets at offsets  −12h, −4h, close(−10min)  @ regions us,eu
  * props markets      at offsets        −4h, close(−10min) @ region  us
and stores ONLY four books: DraftKings, FanDuel, bet365 (bet books) + Pinnacle
(the sole reference). Everything else is dropped as noise.

THREE-PHASE RAW-FIRST PIPELINE (protects the one-time credit window)
  Phase 1 (this file): FETCH → cache/hist_event_odds/  (spends credits; the ONLY
                       phase racing the Sep-21 credit expiry).
  Phase 2 (later, FREE): compile cached raw → deploy/odds_backfill/parquet/.
  Phase 3 (later, FREE): load parquet → Azure warehouse.
A parse/schema bug re-runs Phase 2/3 for $0 — only the raw fetch is irreplaceable.

SAFETY
  * --dry-run prices the whole plan and spends NOTHING.
  * --probe fires ONE tiny real call to read the true per-call credit cost from
    the response header (and to settle the regions-vs-bookmakers billing question)
    before committing to the full spend.
  * --max-credits is a HARD cap; the loop stops before exceeding it.
  * Newest-first ordering, so an early stop keeps the freshest data.
  * Cached calls re-read for 0 credits, so re-runs never double-charge.
  * Nothing is ever overwritten (odds_client cache is write-once/permanent).

Examples
--------
    # Price the full plan, spend nothing:
    python backfill_precise.py --seasons 2024,2025,2026 --tier all --dry-run

    # Read the TRUE per-call cost from the API header before the big spend:
    python backfill_precise.py --seasons 2026 --tier team --probe

    # Fire (after dry-run + probe + review + explicit go), capped:
    python backfill_precise.py --seasons 2026 --tier all --max-credits 500000
"""
import argparse

from odds_client import (
    get_historical_events,
    get_historical_event_odds,
    is_historical_event_cached,
    is_historical_events_cached,
    get_remaining_credits,
    _normalize_snapshot_date,
)
from backfill_historical_odds import load_config, _names_match

SPORT_KEY = "baseball_mlb"

# The four books we keep: three bet books + Pinnacle (reference-only).
DEFAULT_BOOKS = ["draftkings", "fanduel", "bet365", "pinnacle"]

# Region needed per book (Odds API region grouping). bet365 US = us; Pinnacle = eu.
_BOOK_REGION = {
    "draftkings": "us", "fanduel": "us", "bet365": "us", "pinnacle": "eu",
}

# Market groups. Team+F5 uses the per-event PERIOD markets (bulk endpoint can't).
TEAM_MARKETS = [
    "h2h", "spreads", "totals",
    "h2h_1st_5_innings", "spreads_1st_5_innings", "totals_1st_5_innings",
]
PROP_MARKETS = [
    "batter_hits", "batter_total_bases", "batter_rbis", "batter_strikeouts",
    "pitcher_strikeouts", "pitcher_earned_runs", "pitcher_outs",
]

# Game-relative snapshot offsets (hours before commence) → role tag.
# team+F5: three snapshots; props post same-day so skip −12h.
TEAM_OFFSETS = [(12.0, "early_12h"), (4.0, "early_4h"), (10.0 / 60.0, "close")]
PROP_OFFSETS = [(4.0, "early_4h"), (10.0 / 60.0, "close")]

CREDIT_PER_MARKET = 10  # Odds API: cost = 10 × markets × regions (probe verifies).

# Props have no historical data before this date (all sports); our seasons are all
# after it, but keep the guard so a stray earlier date can't burn empty calls.
PROPS_MIN_DATE = "2023-05-03"


# ──────────────────────────────────────────────────────────────────────────────
# Enumeration — regular-season final games from the warehouse
# ──────────────────────────────────────────────────────────────────────────────
def enumerate_games(seasons):
    """Regular-season FINAL games for `seasons` from mlb_game (joined to mlb_team
    for canonical home/away names). Returns dicts:
        {game_pk, commence (full ISO UTC), official_date (YYYY-MM-DD),
         home, away, season}
    ordered NEWEST-FIRST. Fail-open → [] if the warehouse is unavailable."""
    import mlb_warehouse as wh
    import db_store
    from sqlalchemy import select, or_
    if not wh.enabled():
        return []
    g = wh.mlb_game
    home = wh.mlb_team.alias("home_t")
    away = wh.mlb_team.alias("away_t")
    joined = (g.join(home, g.c.home_team_id == home.c.team_id, isouter=True)
              .join(away, g.c.away_team_id == away.c.team_id, isouter=True))
    stmt = (select(g.c.game_pk, g.c.game_date, g.c.official_date, g.c.season,
                   home.c.name.label("home_name"), away.c.name.label("away_name"))
            .select_from(joined)
            .where(g.c.season.in_([int(s) for s in seasons]))
            .where(g.c.game_type == "R")
            .where(g.c.status == "Final")
            .where(g.c.home_score.isnot(None))
            .where(g.c.away_score.isnot(None))
            .order_by(g.c.game_date.desc(), g.c.game_pk.desc()))
    out = []
    with db_store.get_engine().connect() as conn:
        for r in conn.execute(stmt).fetchall():
            m = r._mapping
            commence = m["game_date"]
            if not commence or not m["home_name"] or not m["away_name"]:
                continue
            out.append({
                "game_pk": m["game_pk"],
                "commence": commence,
                "official_date": (m["official_date"] or commence[:10]),
                "home": m["home_name"],
                "away": m["away_name"],
                "season": m["season"],
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Event-ID mapping — warehouse reuse first (free), then historical-events harvest
# ──────────────────────────────────────────────────────────────────────────────
def _warehouse_event_ids():
    """{(official_date, home_norm, away_norm) -> event_id} from odds_snapshot rows
    already in the warehouse (free, no API). Fail-open → {}."""
    try:
        import db_store
        from sqlalchemy import select
        db_store.promote_secrets_from_toml()
        t = db_store.odds_snapshot
        eng = db_store.get_engine()
        norm = db_store.normalize_name
    except Exception:
        return {}
    idx = {}
    try:
        with eng.connect() as c:
            q = (select(t.c.game_date, t.c.event_id, t.c.home, t.c.away)
                 .where(t.c.sport == SPORT_KEY).distinct())
            for gd, eid, h, a in c.execute(q).all():
                if not eid:
                    continue
                idx[((gd or "")[:10], norm(h or ""), norm(a or ""))] = eid
    except Exception:
        return {}
    return idx


def _harvest_ts(official_date):
    """Timestamp to list a date's events at: noon UTC (~8am ET) — the full slate is
    posted and even day games (first pitch ~17:00Z+) are still upcoming, so the
    historical-events listing returns every game for the date."""
    return f"{official_date}T12:00:00Z"


def resolve_event_ids(api_key, games, allow_api):
    """Map each game to its Odds-API event_id.

    Order of resolution (cheapest first):
      1. Warehouse odds_snapshot mapping (free).
      2. Cached historical-events listing (free).
      3. Live historical-events call, 1 credit/date — ONLY when allow_api=True.

    Returns (id_by_pk, harvest_credits). In dry-run (allow_api=False) uncached
    dates are NOT called; their harvest cost is counted so the estimate is honest
    and those games are left unmapped (planned at full, cache-unknown cost)."""
    wh_idx = _warehouse_event_ids()
    id_by_pk = {}
    dates = sorted({g["official_date"] for g in games})
    # Group games by date for listing-based matching.
    by_date = {}
    for g in games:
        by_date.setdefault(g["official_date"], []).append(g)

    import db_store
    try:
        norm = db_store.normalize_name
    except Exception:
        norm = lambda s: (s or "").lower().strip()

    # Pass 1: warehouse reuse.
    unresolved_dates = set()
    for d, gs in by_date.items():
        for g in gs:
            eid = wh_idx.get((d, norm(g["home"]), norm(g["away"])))
            if eid:
                id_by_pk[g["game_pk"]] = eid
            else:
                unresolved_dates.add(d)

    # Pass 2/3: historical-events listing per still-unresolved date.
    harvest_credits = 0
    for d in sorted(unresolved_dates):
        ts = _harvest_ts(d)
        cached = is_historical_events_cached(SPORT_KEY, ts)
        if not cached and not allow_api:
            harvest_credits += 1          # would-be spend, not made in dry-run
            continue
        if not cached:
            harvest_credits += 1
        try:
            events, _ = get_historical_events(api_key, SPORT_KEY, date=ts)
        except Exception:
            continue
        gs = by_date.get(d, [])
        for ev in events or []:
            for g in gs:
                if g["game_pk"] in id_by_pk:
                    continue
                if (_names_match(g["home"], ev.get("home_team"))
                        and _names_match(g["away"], ev.get("away_team"))):
                    id_by_pk[g["game_pk"]] = ev.get("id")
                    break
    return id_by_pk, harvest_credits


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot timestamps + planning
# ──────────────────────────────────────────────────────────────────────────────
def _offset_ts(commence, hours_before):
    """commence (full ISO UTC) minus `hours_before` → normalized snapshot ts."""
    from datetime import datetime, timedelta, timezone
    d = _normalize_snapshot_date(commence)
    dt = datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    dt -= timedelta(hours=hours_before)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _regions_for(books, market_group):
    """Region string for a market group given the selected books. Props are us-only
    (Pinnacle has no MLB props → never pay eu for props)."""
    if market_group == "props":
        return "us"
    regions = []
    for b in books:
        r = _BOOK_REGION.get(b)
        if r and r not in regions:
            regions.append(r)
    # Deterministic order: us before eu.
    return ",".join(sorted(regions))


def _group_specs(tier, books):
    """List of (group, markets_list, offsets, regions) to fetch for a tier."""
    specs = []
    if tier in ("team", "all"):
        specs.append(("team", TEAM_MARKETS, TEAM_OFFSETS,
                      _regions_for(books, "team")))
    if tier in ("props", "all"):
        specs.append(("props", PROP_MARKETS, PROP_OFFSETS,
                      _regions_for(books, "props")))
    return specs


def plan_and_run(api_key, games, id_by_pk, tier, books, max_credits,
                 dry_run, harvest_credits):
    """Price (and, unless dry_run, execute) the per-event fetch plan. Cache-aware:
    an already-cached call is free. Returns nothing; prints a full report."""
    specs = _group_specs(tier, books)
    n_markets = {g: len(m) for g, m, _o, _r in specs}
    n_regions = {g: len(r.split(",")) for g, m, _o, r in specs}
    cost_per_call = {g: CREDIT_PER_MARKET * n_markets[g] * n_regions[g]
                     for g, m, _o, r in specs}

    print(f"\n=== PRECISE BACKFILL {SPORT_KEY} — Phase 1 (fetch → cache) ===")
    print(f"  Books: {', '.join(books)}")
    for g, m, offs, r in specs:
        print(f"  [{g}] markets={len(m)} regions={r} offsets="
              f"{[o[1] for o in offs]}  → {cost_per_call[g]} credits/call")
    print(f"  Games enumerated: {len(games)}   with event_id: {len(id_by_pk)}")
    if harvest_credits:
        print(f"  Event-ID harvest (uncached dates): ~{harvest_credits} credits")

    # Build the task list (newest-first — games already sorted DESC).
    tasks = []            # (game, group, markets_csv, regions, ts, role)
    est_credits = harvest_credits
    cached_calls = new_calls = unmapped_calls = 0
    for g in games:
        eid = id_by_pk.get(g["game_pk"])
        for group, markets, offsets, regions in specs:
            if group == "props" and g["official_date"] < PROPS_MIN_DATE:
                continue
            markets_csv = ",".join(markets)
            for hours_before, role in offsets:
                ts = _offset_ts(g["commence"], hours_before)
                if not eid:
                    # Unmapped game: count as would-be new spend (upper bound).
                    unmapped_calls += 1
                    est_credits += cost_per_call[group]
                    continue
                if is_historical_event_cached(SPORT_KEY, eid, ts, regions=regions,
                                              markets=markets_csv, bookmakers=books):
                    cached_calls += 1
                    continue
                new_calls += 1
                est_credits += cost_per_call[group]
                tasks.append((g, group, markets_csv, regions, ts, role, eid))

    print(f"\n  Calls: {new_calls} new, {cached_calls} cached(free), "
          f"{unmapped_calls} unmapped(no event_id)")
    print(f"  ESTIMATED credits this run: ~{est_credits}  "
          f"(cap: {max_credits})")

    if dry_run:
        print("\n  [dry-run] No API calls made. Re-run without --dry-run to fetch.")
        return
    if not tasks:
        print("\n  Nothing new to fetch (all cached or unmapped). Done.")
        return

    # ── Real fetch loop (cache-only; no warehouse/parquet in Phase 1) ──
    spent = 0
    fetched = empty = 0
    try:
        for i, (g, group, markets_csv, regions, ts, role, eid) in enumerate(tasks, 1):
            this_cost = cost_per_call[group]
            if spent + this_cost > max_credits:
                print(f"  [stop] Budget cap {max_credits} reached "
                      f"(spent ~{spent}).")
                break
            # Re-check cache (a prior offset/group may have primed it mid-run).
            if is_historical_event_cached(SPORT_KEY, eid, ts, regions=regions,
                                          markets=markets_csv, bookmakers=books):
                continue
            try:
                data, snap_ts = get_historical_event_odds(
                    api_key, SPORT_KEY, eid, date=ts, regions=regions,
                    markets=markets_csv, bookmakers=books)
            except Exception as e:
                print(f"  [warn] {g['official_date']} {g['away']}@{g['home']} "
                      f"{group}/{role}: {e}; skipping.")
                continue
            spent += this_cost
            if data is None:
                empty += 1
            else:
                fetched += 1
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] spent ~{spent}, fetched {fetched}, "
                      f"empty {empty} (remaining: {get_remaining_credits()})")
    except KeyboardInterrupt:
        print("\n  [interrupt] Cache is already persisted per-call; safe to resume.")

    print(f"\n=== Phase 1 done. Spent ~{spent} credits. "
          f"Snapshots fetched: {fetched}, empty/expired: {empty}. ===")
    print(f"  Account credits remaining: {get_remaining_credits()}")


# ──────────────────────────────────────────────────────────────────────────────
# Probe — one tiny real call to read TRUE per-call cost + billing model
# ──────────────────────────────────────────────────────────────────────────────
def run_probe(api_key, games, id_by_pk, books):
    """Fire a handful of tiny real calls on ONE event to read the actual credit
    cost from the response header and settle the regions-vs-bookmakers billing
    question BEFORE the full spend. Spends ~a few hundred credits total."""
    target = None
    for g in games:
        if id_by_pk.get(g["game_pk"]):
            target = g
            break
    if not target:
        print("  [probe] No mapped event to probe. Run without --dry-run to "
              "harvest at least one event_id first.")
        return
    eid = id_by_pk[target["game_pk"]]
    ts = _offset_ts(target["commence"], 4.0)
    team_csv = ",".join(TEAM_MARKETS)
    print(f"\n=== PROBE — event {eid} ({target['away']}@{target['home']} "
          f"{target['official_date']}) @ {ts} ===")
    variants = [
        ("team us-only  +books", "us", team_csv, books),
        ("team us,eu    +books", "us,eu", team_csv, books),
        ("team us,eu    no-books", "us,eu", team_csv, None),
    ]
    for label, regions, markets_csv, bks in variants:
        before = get_remaining_credits()
        try:
            data, snap = get_historical_event_odds(
                api_key, SPORT_KEY, eid, date=ts, regions=regions,
                markets=markets_csv, bookmakers=bks)
        except Exception as e:
            print(f"  [{label}] error: {e}")
            continue
        after = get_remaining_credits()
        delta = (before - after) if (before is not None and after is not None) else "?"
        n_books = len(data.get("bookmakers", [])) if data else 0
        print(f"  [{label}] regions={regions} → header-cost≈{delta} credits, "
              f"snapshot={snap}, books_returned={n_books}")
    print("  (Compare the three: us-only vs us,eu shows the region multiplier; "
          "+books vs no-books shows whether the bookmakers filter changes billing.)")


def main():
    p = argparse.ArgumentParser(
        description="Precise game-relative historical-odds backfill (Phase 1).")
    p.add_argument("--seasons", default="2024,2025,2026",
                   help="Comma-separated seasons (default 2024,2025,2026).")
    p.add_argument("--tier", choices=["team", "props", "all"], default="all")
    p.add_argument("--books", default=",".join(DEFAULT_BOOKS),
                   help="Comma-separated books to store (default the 4).")
    p.add_argument("--max-credits", type=int, default=5000,
                   help="Hard cap on credits this run may spend (default 5000).")
    p.add_argument("--limit-games", type=int, default=0,
                   help="Cap enumerated games (0 = all); for testing/probe scoping.")
    p.add_argument("--dry-run", action="store_true",
                   help="Price the plan and spend nothing.")
    p.add_argument("--probe", action="store_true",
                   help="Fire a few tiny real calls to read the true per-call cost "
                        "and billing model before the full spend.")
    args = p.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    books = [b.strip() for b in args.books.split(",") if b.strip()]
    cfg = load_config()
    api_key = cfg["odds_api_key"]

    # Warehouse reads (enumeration + event-id reuse) need the Azure SQL_* secrets in
    # the env; outside Streamlit they aren't promoted yet. Free/read-only.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

    games = enumerate_games(seasons)
    if not games:
        print("No enumerated games (warehouse unavailable or empty). "
              "Ensure SQL secrets are set and mlb_game is populated.")
        return
    if args.limit_games:
        games = games[:args.limit_games]
    print(f"Enumerated {len(games)} regular-season final games "
          f"({', '.join(seasons)}), newest-first.")

    # Resolve event_ids. Dry-run resolves only via free sources (warehouse + cache);
    # probe/real runs may make the 1-credit/date harvest calls.
    id_by_pk, harvest_credits = resolve_event_ids(
        api_key, games, allow_api=(not args.dry_run))

    if args.probe:
        run_probe(api_key, games, id_by_pk, books)
        return

    plan_and_run(api_key, games, id_by_pk, args.tier, books,
                 args.max_credits, args.dry_run, harvest_credits)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
