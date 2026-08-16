#!/usr/bin/env python3
"""Gap-targeted top-up of MLB team-market CLOSING lines from The Odds API.

Warehouse-driven (no ESPN): finds every completed game in the StatsAPI warehouse
(``mlb_game`` facts — regular + postseason, status Final) that LACKS a team-market
odds snapshot, then fetches ONE historical daily snapshot per missing DATE from
The Odds API and archives the whole slate to the warehouse via
``warehouse.capture_event_odds`` (write-once dedups games already present).

Fetching per-DATE (not per-game) is what keeps it affordable: one historical
"featured" snapshot returns every game on the slate at cost ``10 x markets x
regions`` = 30 credits/date (h2h+spreads+totals, 1 region), versus ~30 credits
per GAME for true per-tip-off closing (~2500 missing games ≈ 75k credits —
infeasible). The daily snapshot is taken at the date's FIRST tip-off, so every
game's line is a game-day near-closing line (the same tradeoff the prior team
backfill accepted).

Coverage is decided by a MULTISET match on (official_date ±1 day, canonical team
code-set), so doubleheaders need two snapshots to count as covered and UTC-vs-ET
date boundaries don't cause false gaps. DraftKings-only.

Dry-run by default (game-level gap report + exact credit cost). ``--apply``
fetches and writes, hard-capped by ``--max-credits`` and gated behind an explicit
``--yes`` (the caller must have already confirmed the spend).
"""

import argparse
import os
from collections import Counter, defaultdict
from datetime import date as _date, datetime, timedelta, timezone

SPORT_KEY = "baseball_mlb"
BOOK_KEY = "draftkings"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"
FEATURED_COST = 30                      # 10 * 3 markets * 1 region
GAME_TYPES = frozenset({"R", "F", "D", "L", "W"})   # regular + postseason
FINAL_STATES = frozenset({"Final", "Completed Early", "Game Over"})


# ──────────────────────────────────────────────────────────────────────────────
# Coverage diff (pure-ish: takes already-fetched rows) — unit-tested offline
# ──────────────────────────────────────────────────────────────────────────────

def _shift(date10, days):
    return (_date.fromisoformat(date10) + timedelta(days=days)).isoformat()


def compute_gap(games, snapshots):
    """Diff authoritative games against existing team-snapshot coverage.

    ``games``: list of {official_date, commence, codeset(frozenset of 2 codes),
        game_pk} for completed regular/postseason games (codes already resolved;
        a game with an unresolved code carries codeset=None).
    ``snapshots``: list of {date10, codeset} for existing team snapshots.

    A game is COVERED when an existing snapshot with the SAME code-set falls on
    its official_date ±1 day; matching is a MULTISET (each snapshot covers at most
    one game, so a doubleheader needs two). Returns
    (missing_games, missing_dates, unresolved) where missing_dates maps
    official_date -> count of missing games that date."""
    # available snapshots indexed by codeset -> Counter(date10)
    avail = defaultdict(Counter)
    for s in snapshots:
        cs = s.get("codeset")
        d = s.get("date10")
        if cs and d:
            avail[cs][d] += 1

    missing_games, unresolved = [], []
    missing_dates = Counter()
    # Process in a stable order so multiset consumption is deterministic.
    for g in sorted(games, key=lambda x: (x.get("official_date") or "",
                                          x.get("game_pk") or 0)):
        cs = g.get("codeset")
        od = g.get("official_date")
        if not cs or not od:
            unresolved.append(g)
            continue
        # Try to consume a snapshot on od, od-1, od+1 (nearest day first).
        consumed = False
        for d in (od, _shift(od, -1), _shift(od, 1)):
            if avail.get(cs, {}).get(d, 0) > 0:
                avail[cs][d] -= 1
                consumed = True
                break
        if not consumed:
            missing_games.append(g)
            missing_dates[od] += 1
    return missing_games, dict(missing_dates), unresolved


def snapshot_ts_for_date(commences):
    """The date's first tip-off (min commence) as an ISO-Z timestamp, so the daily
    historical snapshot is a game-day near-closing line. Falls back to 17:00Z (≈
    1pm ET) when no commence is known."""
    good = sorted(c for c in commences if c)
    if good:
        return good[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Warehouse reads
# ──────────────────────────────────────────────────────────────────────────────

def _team_code_map(conn):
    import mlb_warehouse as mw
    import player_id_map as pim
    from sqlalchemy import select
    tm = mw.mlb_team
    out = {}
    for tid, name, ab in conn.execute(
            select(tm.c.team_id, tm.c.name, tm.c.abbreviation)).all():
        out[str(tid)] = pim.team_code_for_abbr(ab) or pim.team_code_for_name(name)
    return out


def load_games(conn, date_from, date_to):
    """Completed regular/postseason games in [date_from, date_to] as gap-diff
    dicts (codeset resolved via the team map)."""
    import mlb_warehouse as mw
    from sqlalchemy import select
    g = mw.mlb_game
    codes = _team_code_map(conn)
    rows = conn.execute(
        select(g.c.game_pk, g.c.game_date, g.c.official_date, g.c.game_type,
               g.c.status, g.c.home_team_id, g.c.away_team_id)
        .where((g.c.official_date >= date_from)
               & (g.c.official_date <= date_to))).all()
    out = []
    for pk, commence, od, gt, status, hid, aid in rows:
        if gt not in GAME_TYPES:
            continue
        if status not in FINAL_STATES:
            continue
        hc, ac = codes.get(str(hid)), codes.get(str(aid))
        cs = frozenset({hc, ac}) if (hc and ac and hc != ac) else None
        out.append({"game_pk": pk, "commence": commence, "official_date": od,
                    "game_type": gt, "codeset": cs,
                    "home_code": hc, "away_code": ac})
    return out


def load_snapshots(conn, date_from, date_to):
    """Existing team snapshots in [date_from-1, date_to+1] as gap-diff dicts."""
    from sqlalchemy import select
    import db_store
    t = db_store.odds_snapshot
    lo, hi = _shift(date_from, -1), _shift(date_to, 1)
    rows = conn.execute(
        select(t.c.game_date, t.c.home_code, t.c.away_code, t.c.event_id)
        .where((t.c.sport == SPORT_KEY) & (t.c.kind == "team")
               & (t.c.game_date >= lo) & (t.c.game_date <= hi))).all()
    out = []
    for gd, hc, ac, eid in rows:
        cs = frozenset({hc, ac}) if (hc and ac and hc != ac) else None
        out.append({"date10": (gd or "")[:10], "codeset": cs, "event_id": eid})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Report + apply
# ──────────────────────────────────────────────────────────────────────────────

def _month(d):
    return (d or "")[:7]


def _report(games, missing_games, missing_dates, unresolved):
    print(f"\n  completed R+postseason games in window: {len(games)}")
    print(f"  unresolved team code (skipped): {len(unresolved)}")
    covered = len(games) - len(missing_games) - len(unresolved)
    print(f"  already covered by a team snapshot: {covered}")
    print(f"  MISSING games: {len(missing_games)}  across "
          f"{len(missing_dates)} dates")
    # by month
    gm = Counter(_month(g["official_date"]) for g in missing_games)
    dm = Counter(_month(d) for d in missing_dates)
    print("\n  missing by month (games / dates):")
    for m in sorted(set(gm) | set(dm)):
        print(f"    {m}:  {gm.get(m,0):4d} games / {dm.get(m,0):3d} dates")
    cost = len(missing_dates) * FEATURED_COST
    print(f"\n  daily-cadence cost = {len(missing_dates)} dates x "
          f"{FEATURED_COST} cr = ~{cost} credits.")
    return cost


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="date_from", default="2023-04-01",
                    help="earliest official_date (default 2023-04-01)")
    ap.add_argument("--to", dest="date_to", default=None,
                    help="latest official_date (default: today UTC)")
    ap.add_argument("--max-credits", type=int, default=8000,
                    help="hard cap on credits this run may spend (default 8000)")
    ap.add_argument("--max-dates", type=int, default=None,
                    help="cap number of dates fetched (smoke/testing)")
    ap.add_argument("--apply", action="store_true",
                    help="fetch + write (default: dry-run)")
    ap.add_argument("--yes", action="store_true",
                    help="required with --apply: confirms the credit spend")
    args = ap.parse_args()

    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception as e:
        print(f"  (secret promotion failed: {type(e).__name__})")
    import db_store
    if not db_store.enabled():
        print("  SQL warehouse not enabled — aborting.")
        return 2

    date_to = args.date_to or datetime.now(timezone.utc).date().isoformat()
    print(f"MLB team-market Odds-API top-up  ({'APPLY' if args.apply else 'DRY-RUN'})")
    print(f"  window: {args.date_from} .. {date_to}   book: {BOOK_KEY}   "
          f"markets: {MARKETS}")

    with db_store.get_engine().connect() as conn:
        games = load_games(conn, args.date_from, date_to)
        snaps = load_snapshots(conn, args.date_from, date_to)
        # commence times per missing date for the snapshot timestamp
        commence_by_date = defaultdict(list)
        for g in games:
            commence_by_date[g["official_date"]].append(g.get("commence"))

    missing_games, missing_dates, unresolved = compute_gap(games, snaps)
    cost = _report(games, missing_games, missing_dates, unresolved)

    if not args.apply:
        print("\n  dry-run only — re-run with --apply --yes to fetch "
              "(spends credits).")
        return 0
    if not args.yes:
        print("\n  --apply requires --yes (confirms the credit spend). Aborting.")
        return 2
    if cost > args.max_credits:
        print(f"\n  planned {cost} cr exceeds --max-credits {args.max_credits}. "
              f"Raise the cap or narrow the window. Aborting.")
        return 2

    # ── fetch loop ──────────────────────────────────────────────────────────
    import json as _json
    from odds_client import get_historical_odds, get_remaining_credits
    import warehouse
    cfg = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "config.json")))
    api_key = cfg["odds_api_key"]

    dates = sorted(missing_dates.keys())
    if args.max_dates is not None:
        dates = dates[:args.max_dates]
    print(f"\n  fetching {len(dates)} daily snapshots (~{len(dates)*FEATURED_COST} "
          f"cr cap {args.max_credits}) ...")
    spent, captured, empty = 0, 0, 0
    for i, d in enumerate(dates, 1):
        if spent + FEATURED_COST > args.max_credits:
            print(f"  [stop] budget cap reached at {spent} cr.")
            break
        ts = snapshot_ts_for_date(commence_by_date.get(d, [])) or f"{d}T17:00:00Z"
        try:
            slate, snap_ts = get_historical_odds(
                api_key, SPORT_KEY, date=ts, regions=REGIONS,
                markets=MARKETS, bookmakers=[BOOK_KEY])
        except Exception as e:
            print(f"  [warn] {d}: fetch failed ({type(e).__name__}); skipping.")
            continue
        spent += FEATURED_COST
        n = 0
        for api_game in (slate or []):
            eid = api_game.get("id")
            if not eid:
                continue
            warehouse.capture_event_odds(
                SPORT_KEY, eid, REGIONS, MARKETS, [BOOK_KEY], api_game,
                captured_at=snap_ts)
            n += 1
        captured += n
        if n == 0:
            empty += 1
        if i % 20 == 0 or i == len(dates):
            print(f"    [{i}/{len(dates)}] spent ~{spent} cr, {captured} games "
                  f"captured (remaining: {get_remaining_credits()})")
    print(f"\n  done — spent ~{spent} cr, captured {captured} games across "
          f"{len(dates)} dates ({empty} dates returned no slate).")
    print(f"  account credits remaining: {get_remaining_credits()}")
    return 0


if __name__ == "__main__":
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    raise SystemExit(main())
