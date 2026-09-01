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
import os

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

# Books we keep: the bet books + Pinnacle (reference-only). bet365 was dropped
# 2026-09-01 — the historical per-event endpoint carries no bet365 MLB depth (probe
# 2026-08-30 showed it absent from the full 33-book slate), so paying to request it
# would just add an always-empty book. DK/FD are the executable bet books here;
# Pinnacle is the sole reference.
DEFAULT_BOOKS = ["draftkings", "fanduel", "pinnacle"]

# Region needed per book (Odds API region grouping). DK/FD = us; Pinnacle = eu.
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
    from sqlalchemy import select, func
    if not wh.enabled():
        return []
    # Terminal-state denylist: a postponed/suspended/cancelled game can surface as
    # abstractGameState 'Final' with a PARTIAL (0-0) linescore that passes the
    # score-not-null test, so gate on detailed_state too — same guard the rest of the
    # codebase uses (mlb_starters._is_genuine_final / mlb_warehouse.final_game_by_pk).
    # coalesce('') keeps NULL detailed_state rows enumerated (trust abstract state),
    # matching that convention. Without this we'd spend ~500 cr/game fetching odds for
    # ungradable games and pollute the cache.
    try:
        from mlb_starters import _NON_FINAL_DETAILED
    except Exception:
        _NON_FINAL_DETAILED = ("postpon", "suspend", "cancel")
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
            .where(g.c.away_score.isnot(None)))
    for bad in _NON_FINAL_DETAILED:
        stmt = stmt.where(
            ~func.lower(func.coalesce(g.c.detailed_state, "")).like(f"%{bad}%"))
    stmt = stmt.order_by(g.c.game_date.desc(), g.c.game_pk.desc())
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
def _parse_ts(ts):
    """ISO ts → aware UTC datetime, or None. Uses the same normalizer the fetch path
    uses so warehouse/listing commence strings compare consistently."""
    from datetime import datetime, timezone
    try:
        d = _normalize_snapshot_date(ts)
        return datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _nearest(cands, commence, used):
    """From (commence_time, event_id) candidates, pick the UNUSED event whose
    commence is nearest `commence` (falls back to the first unused when times are
    unparseable). Returns event_id or None. Disambiguates same-day same-matchup
    (doubleheader) games so each game_pk binds to ITS OWN event."""
    gc = _parse_ts(commence)
    best = None
    for ct, eid in cands:
        if not eid or eid in used:
            continue
        ec = _parse_ts(ct)
        dist = abs((ec - gc).total_seconds()) if (gc and ec) else float("inf")
        if best is None or dist < best[0]:
            best = (dist, eid)
    return best[1] if best else None


def _warehouse_event_ids():
    """{(official_date, home_norm, away_norm) -> [(commence_time, event_id), ...]}
    from odds_snapshot rows already in the warehouse (free, no API). MULTI-valued so
    a split doubleheader (two distinct event_ids sharing date+matchup) is not
    collapsed to one id. Fail-open → {}."""
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
            q = (select(t.c.game_date, t.c.event_id, t.c.home, t.c.away,
                        t.c.commence_time)
                 .where(t.c.sport == SPORT_KEY).distinct())
            for gd, eid, h, a, ct in c.execute(q).all():
                if not eid:
                    continue
                key = ((gd or "")[:10], norm(h or ""), norm(a or ""))
                bucket = idx.setdefault(key, [])
                if not any(e == eid for _c, e in bucket):   # dedupe by event_id
                    bucket.append((ct, eid))
    except Exception:
        return {}
    return idx


def _harvest_ts(official_date):
    """Timestamp to list a date's events at: noon UTC (~8am ET) — the full slate is
    posted and even day games (first pitch ~17:00Z+) are still upcoming, so the
    historical-events listing returns every game for the date."""
    return f"{official_date}T12:00:00Z"


def resolve_event_ids(api_key, games, allow_api, max_credits=None):
    """Map each game to its Odds-API event_id.

    Order of resolution (cheapest first):
      1. Warehouse odds_snapshot mapping (free).
      2. Cached historical-events listing (free).
      3. Live historical-events call, 1 credit/date — ONLY when allow_api=True,
         and STOPPED once harvest spend would reach `max_credits` (the hard cap
         bounds harvest too, not just the fetch loop).

    Doubleheaders: same-day same-matchup games are disambiguated by NEAREST commence
    time (one event per game_pk), in BOTH the warehouse-reuse and listing passes.

    Returns (id_by_pk, harvest_credits). In dry-run (allow_api=False) uncached dates
    are NOT called; their harvest cost is counted so the estimate is honest and those
    games are left unmapped (planned at full, cache-unknown cost)."""
    wh_idx = _warehouse_event_ids()
    id_by_pk = {}
    by_date = {}
    for g in games:
        by_date.setdefault(g["official_date"], []).append(g)

    import db_store
    try:
        norm = db_store.normalize_name
    except Exception:
        norm = lambda s: (s or "").lower().strip()

    # Pass 1: warehouse reuse — cluster by (date, matchup) so a DH is matched
    # one-to-one by nearest commence instead of collapsing to a single event_id.
    unresolved_dates = set()
    for d, gs in by_date.items():
        clusters = {}
        for g in gs:
            clusters.setdefault((d, norm(g["home"]), norm(g["away"])), []).append(g)
        for key, cluster in clusters.items():
            cands = list(wh_idx.get(key) or [])
            used = set()
            for g in sorted(cluster, key=lambda x: x["commence"]):
                eid = _nearest(cands, g["commence"], used)
                if eid:
                    id_by_pk[g["game_pk"]] = eid
                    used.add(eid)
            # Any game in this matchup still unmapped (no candidate, or fewer
            # warehouse events than DH games) → let the harvest pass fill it.
            if any(g["game_pk"] not in id_by_pk for g in cluster):
                unresolved_dates.add(d)

    # Pass 2/3: historical-events listing per still-unresolved date, cap-bounded.
    harvest_credits = 0
    for d in sorted(unresolved_dates):
        ts = _harvest_ts(d)
        cached = is_historical_events_cached(SPORT_KEY, ts)
        if not cached:
            if not allow_api:
                harvest_credits += 1      # would-be spend, not made in dry-run
                continue
            if max_credits is not None and harvest_credits >= max_credits:
                continue                  # hard cap reached — stop harvesting
            harvest_credits += 1
        try:
            events, _ = get_historical_events(api_key, SPORT_KEY, date=ts)
        except Exception as e:
            print(f"  [warn] events harvest failed {d}: {e}")
            continue
        gs = by_date.get(d, [])
        # Events already bound to this date's games (from warehouse) can't be reused.
        used = {id_by_pk[g["game_pk"]] for g in gs if g["game_pk"] in id_by_pk}
        # Name-matching candidates from the listing, carrying commence for DH tiebreak.
        for g in sorted((x for x in gs if x["game_pk"] not in id_by_pk),
                        key=lambda x: x["commence"]):
            cands = [(ev.get("commence_time"), ev.get("id")) for ev in (events or [])
                     if _names_match(g["home"], ev.get("home_team"))
                     and _names_match(g["away"], ev.get("away_team"))]
            eid = _nearest(cands, g["commence"], used)
            if eid:
                id_by_pk[g["game_pk"]] = eid
                used.add(eid)
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
    # Deterministic (sorted) order → "eu,us" for team; the exact string is part of
    # the odds_client cache key, so it must be computed identically everywhere.
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


_UNMAPPED_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "odds_backfill", "unmapped_games.csv")


def _write_unmapped(unmapped_games, dry_run):
    """Print + persist the games with no resolved event_id (audit trail)."""
    tag = "expected in dry-run" if dry_run else "INVESTIGATE name-match/harvest"
    print(f"\n  UNMAPPED ({len(unmapped_games)}) — no event_id ({tag}):")
    for g in unmapped_games[:20]:
        print(f"    {g['official_date']}  {g['away']} @ {g['home']}  "
              f"(pk={g['game_pk']})")
    if len(unmapped_games) > 20:
        print(f"    … +{len(unmapped_games) - 20} more (full list in the CSV)")
    try:
        os.makedirs(os.path.dirname(_UNMAPPED_CSV), exist_ok=True)
        with open(_UNMAPPED_CSV, "w", encoding="utf-8") as f:
            f.write("official_date,away,home,game_pk,commence\n")
            for g in unmapped_games:
                f.write(f"{g['official_date']},{g['away']},{g['home']},"
                        f"{g['game_pk']},{g['commence']}\n")
        print(f"    → wrote {_UNMAPPED_CSV}")
    except OSError as e:
        print(f"    [warn] could not write unmapped manifest: {e}")


def plan_and_run(api_key, games, id_by_pk, tier, books, max_credits,
                 dry_run, harvest_credits, workers=12):
    """Price (and, unless dry_run, execute) the per-event fetch plan. Cache-aware:
    an already-cached call is free. Returns nothing; prints a full report."""
    specs = _group_specs(tier, books)
    n_markets = {g: len(m) for g, m, _o, _r in specs}
    n_regions = {g: len(r.split(",")) for g, m, _o, r in specs}
    # BILLING (probe-verified 2026-09-01): The Odds API bills a per-event call with a
    # `bookmakers=` filter as ONE region, regardless of how many regions those books
    # span — so a team call at regions=us,eu +books costs 10×markets×1 (=60), NOT
    # 10×markets×2 (=120), and STILL returns Pinnacle (eu) alongside DK/FD (us) in a
    # single call. We always pass `books`, so billing is 1 region; we keep the full
    # regions STRING on the call itself (needed to actually return the eu book).
    cost_per_call = {g: CREDIT_PER_MARKET * n_markets[g]
                     * (1 if books else n_regions[g])
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
    tasks = []            # (game, group, markets_csv, regions, ts, role, eid)
    est_credits = harvest_credits
    cached_calls = new_calls = unmapped_calls = 0
    unmapped_games = []
    for g in games:
        eid = id_by_pk.get(g["game_pk"])
        if not eid:
            unmapped_games.append(g)   # once per game, for the audit manifest
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

    # Audit trail: surface + persist the games we could NOT map to an event_id, so a
    # name-match miss or harvest failure is fixable/re-runnable (free) before the
    # credit window closes — never a silent, unrecoverable gap. (In dry-run, unmapped
    # is expected/large because uncached dates aren't harvested — informational only.)
    if unmapped_games:
        _write_unmapped(unmapped_games, dry_run)

    if dry_run:
        print("\n  [dry-run] No API calls made. Re-run without --dry-run to fetch.")
        return
    if not tasks:
        print("\n  Nothing new to fetch (all cached or unmapped). Done.")
        return

    # ── Budget-slice BEFORE dispatch ──────────────────────────────────────────
    # Each task in `tasks` is a distinct cache key = exactly one genuine paid call
    # (cached calls were already excluded at plan time). So we can pick the prefix
    # that fits --max-credits deterministically, newest-first, and enforce the cap
    # WITHOUT racing a shared counter across worker threads. A worker that finds a
    # call cached (primed since planning) simply spends less — the cap still holds.
    # Seed `committed` with harvest spend already made this run so the fetch prefix
    # + harvest together never exceed --max-credits (the cap bounds the WHOLE run).
    budgeted, committed = [], harvest_credits
    for t in tasks:
        c = cost_per_call[t[1]]
        if committed + c > max_credits:
            break
        committed += c
        budgeted.append(t)
    if len(budgeted) < len(tasks):
        print(f"  [budget] cap {max_credits} allows {len(budgeted)}/{len(tasks)} "
              f"calls (~{committed} cr) this run; re-run to continue "
              f"(already-fetched calls re-read free).")

    # ── Concurrent fetch loop (cache-only; no warehouse/parquet in Phase 1) ──
    # Network-bound per-event calls → threads. odds_client writes each response to
    # its own cache file (distinct key per task), so concurrent writes never collide.
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()
    ctr = {"spent": 0, "fetched": 0, "empty": 0, "done": 0, "err": 0}
    total = len(budgeted)

    def _fetch(task):
        g, group, markets_csv, regions, ts, role, eid = task
        # Re-check cache (another task/run may have primed this exact key): free skip.
        if is_historical_event_cached(SPORT_KEY, eid, ts, regions=regions,
                                      markets=markets_csv, bookmakers=books):
            with lock:
                ctr["done"] += 1
            return
        try:
            data, _snap = get_historical_event_odds(
                api_key, SPORT_KEY, eid, date=ts, regions=regions,
                markets=markets_csv, bookmakers=books)
        except Exception as e:
            with lock:
                ctr["err"] += 1
                ctr["done"] += 1
            return ("err", g, group, role, str(e))
        with lock:
            ctr["spent"] += cost_per_call[group]
            if data is None:
                ctr["empty"] += 1
            else:
                ctr["fetched"] += 1
            ctr["done"] += 1
        return None

    n_workers = max(1, int(workers))
    print(f"  Dispatching {total} calls across {n_workers} workers…")
    ex = ThreadPoolExecutor(max_workers=n_workers)
    try:
        futures = [ex.submit(_fetch, t) for t in budgeted]
        for fut in as_completed(futures):
            res = fut.result()
            if res and res[0] == "err":
                _, g, group, role, msg = res
                print(f"  [warn] {g['official_date']} {g['away']}@{g['home']} "
                      f"{group}/{role}: {msg}")
            with lock:
                done = ctr["done"]
                spent = ctr["spent"]
                fetched = ctr["fetched"]
                empty = ctr["empty"]
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] spent ~{spent}, fetched {fetched}, "
                      f"empty {empty} (remaining: {get_remaining_credits()})")
    except KeyboardInterrupt:
        print("\n  [interrupt] Cancelling pending calls; cache is persisted "
              "per-call, so re-running resumes for free.")
        ex.shutdown(wait=False, cancel_futures=True)
    else:
        ex.shutdown(wait=True)

    print(f"\n=== Phase 1 done. Spent ~{ctr['spent']} credits. "
          f"Snapshots fetched: {ctr['fetched']}, empty/expired: {ctr['empty']}, "
          f"errors: {ctr['err']}. ===")
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
    # The prod-shaped variant derives regions the SAME way plan_and_run does
    # (_regions_for → sorted "eu,us"), so it seeds the exact cache key the real run
    # checks — that event's early_4h team snapshot then re-reads free instead of
    # being re-fetched. The us-only / no-books variants are diagnostic (region
    # multiplier + bookmakers-billing) and intentionally not reused.
    prod_regions = _regions_for(books, "team")
    variants = [
        ("team us-only    +books", "us", team_csv, books),
        ("team eu-only    +books", "eu", team_csv, books),
        (f"team {prod_regions} +books (prod)", prod_regions, team_csv, books),
        (f"team {prod_regions} no-books", prod_regions, team_csv, None),
    ]
    want = set(books)
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
        keys = sorted({b.get("key") for b in (data.get("bookmakers", []) if data else [])})
        # For the +books variants, call out whether every requested book came back —
        # a silently-dropped Pinnacle (our sole reference) would gut the corpus.
        miss = ""
        if bks is not None:
            missing = sorted(want - set(keys))
            got = [k for k in keys if k in want]
            miss = f"  requested={got or '[]'}  MISSING={missing or 'none'}"
        print(f"  [{label}] regions={regions} → header-cost≈{delta} credits, "
              f"snapshot={snap}, books_returned={len(keys)}{miss}")
        if bks is None:
            print(f"      all books present @ this snapshot: {keys}")
    print("\n  KEY QUESTIONS: (1) us-only vs us,eu (+books) → does the bookmakers "
          "filter bill us,eu as ONE region (60) or two (120)? (2) Does the cheap "
          "us,eu +books call actually RETURN pinnacle, or must we make a separate "
          "eu call? (3) Is any MISSING book just absent at this snapshot vs never "
          "returned in its region?")


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
    p.add_argument("--workers", type=int, default=12,
                   help="Concurrent API workers for the fetch loop (default 12). "
                        "Network-bound calls parallelize well; the --max-credits "
                        "cap is enforced by budget-slicing BEFORE dispatch.")
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
        api_key, games, allow_api=(not args.dry_run),
        max_credits=args.max_credits)

    if args.probe:
        run_probe(api_key, games, id_by_pk, books)
        return

    plan_and_run(api_key, games, id_by_pk, args.tier, books,
                 args.max_credits, args.dry_run, harvest_credits,
                 workers=args.workers)


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
