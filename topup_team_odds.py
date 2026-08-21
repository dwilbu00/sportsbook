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

    ``games``: list of {official_date (ET play date), codeset (frozenset of 2
        distinct codes, else None), game_pk} for completed reg/postseason games.
    ``snapshots``: list of {date10 (ET date of the snapshot's commence), codeset}
        for existing team snapshots THAT CARRY A TEAM LINE (empty snapshots are
        excluded upstream in load_snapshots).

    Coverage is EXACT-membership on (ET date, code-set) — no ±1-day window and no
    per-row consumption. Both sides are ET calendar dates (official_date from
    StatsAPI; et_local_date(commence) for snapshots), so a UTC/ET boundary can't
    make a game look uncovered, and — critically — a game can no longer be marked
    covered by a *same-matchup neighbor's* snapshot a day away (the old ±1 greedy
    multiset bug). A doubleheader (two games, same date+code-set) is covered by a
    single snapshot: that matches how the backtest's code-key join collapses a DH
    to one entry (and calibration drops DHs), so requiring two would only force
    pointless re-fetches. Returns (missing_games, missing_dates, unresolved) where
    missing_dates maps official_date -> count of missing games that date."""
    covered = set()
    for s in snapshots:
        cs, d = s.get("codeset"), s.get("date10")
        if cs and d:
            covered.add((d, cs))

    missing_games, unresolved = [], []
    missing_dates = Counter()
    for g in sorted(games, key=lambda x: (x.get("official_date") or "",
                                          x.get("game_pk") or 0)):
        cs = g.get("codeset")
        od = g.get("official_date")
        if not cs or not od:
            unresolved.append(g)
            continue
        if (od, cs) in covered:
            continue
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
    """Existing team snapshots that CARRY A TEAM LINE, in [date_from-1, date_to+1],
    as gap-diff dicts keyed by the ET date of their commence.

    Joins odds_snapshot -> odds_line (bet_type in moneyline/spread/total) so a
    lineless snapshot (DK absent, or a too-early live capture) does NOT read as
    coverage — the top-up can then upgrade it to a real closing line. date10 is
    the ET calendar date of commence_time (matches mlb_game.official_date)."""
    from sqlalchemy import select
    import db_store
    from pricing_common import et_local_date
    t, ln = db_store.odds_snapshot, db_store.odds_line
    lo, hi = _shift(date_from, -1), _shift(date_to, 1)
    joined = t.join(ln, ln.c.snapshot_id == t.c.id)
    rows = conn.execute(
        select(t.c.commence_time, t.c.home_code, t.c.away_code)
        .select_from(joined)
        .where((t.c.sport == SPORT_KEY) & (t.c.kind == "team")
               & (t.c.game_date >= lo) & (t.c.game_date <= hi)
               & ln.c.bet_type.in_(("moneyline", "spread", "total")))
        .distinct()).all()
    out = []
    for commence, hc, ac in rows:
        cs = frozenset({hc, ac}) if (hc and ac and hc != ac) else None
        d = et_local_date(commence) or (commence or "")[:10]
        out.append({"date10": d, "codeset": cs})
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
    ap.add_argument("--from", dest="date_from", default="2023-03-01",
                    help="earliest official_date (default 2023-03-01, before any "
                         "season opener so late-March games aren't skipped)")
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
    dates = sorted(missing_dates.keys())
    if args.max_dates is not None:
        dates = dates[:args.max_dates]
    # Abort on the EFFECTIVE cost (after --max-dates), not the full-window cost, so
    # a small smoke run on a large window isn't refused. The in-loop guard is the
    # true ceiling regardless.
    effective_cost = len(dates) * FEATURED_COST
    if effective_cost > args.max_credits:
        print(f"\n  planned {effective_cost} cr exceeds --max-credits "
              f"{args.max_credits}. Raise the cap or narrow the window. Aborting.")
        return 2

    # ── fetch loop ──────────────────────────────────────────────────────────
    import json as _json
    from odds_client import get_historical_odds, get_remaining_credits
    import warehouse
    cfg = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "config.json")))
    api_key = cfg["odds_api_key"]
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
                captured_at=snap_ts, source="backfill")
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
