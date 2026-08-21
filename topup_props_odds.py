#!/usr/bin/env python3
"""Gap-targeted, budget-capped reconciler for MLB PLAYER-PROP closing lines.

Player props are the expensive market (the Odds API charges per-GAME:
10 x prop-markets x regions), so a full month can't fit one 20k credit budget.
This tool amortizes the backlog: it finds completed games that LACK prop coverage
for a chosen market set, works NEWEST-FIRST (so the corpus deepens backward while
the most-relevant recent seasons are covered first), and fetches only up to a
credit cap — so you run it once a month with `--max-credits <your budget>` and it
chips away at the backlog forever, using the off-season budget too.

Per game it requests ONLY that game's MISSING markets (write-once already dedups,
but not requesting present markets also saves credits). Event IDs are discovered
once per date via the cheap historical-events endpoint (1 credit/date). Each
game's props are snapshotted at the game's own commence (true per-game closing).

Warehouse-driven (mlb_game facts, no ESPN), ET-exact coverage match (same as
topup_team_odds), DraftKings-only, write-once/idempotent. Dry-run by default
(backlog + per-market gap + credit projection); --apply requires --yes.

STATUS (2026-08-15): dry-run verified; NOT yet spend-hardened. Before any --apply,
add offline gap-diff tests + the 4-lens adversarial review the team tools got, and
FINALIZE the --markets set from the backtest + refit_calibration results (some
props may not be worth offering, so don't pay to backfill them). Parked pending
that refit. DEFAULT_MARKETS below is a placeholder (5 calibration-backed core;
skips batter_total_bases / batter_rbis, which have no calibration fit yet).
"""

import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import topup_team_odds as tt   # reuse load_games / _shift / snapshot_ts_for_date

SPORT_KEY = "baseball_mlb"
BOOK_KEY = "draftkings"
REGIONS = "us"
PROPS_MIN_DATE = "2023-05-03"          # Odds API historical props start here
CREDITS_PER_MARKET = 10                # 10 x markets x regions, regions=1
EVENTS_LOOKUP_COST = 1                 # historical-events endpoint, per date

# Default market set: the 5 calibration-backed core props. batter_total_bases /
# batter_rbis are omitted — no calibration fit yet, so warehousing them across
# seasons is low value until seeded. Override with --markets.
DEFAULT_MARKETS = ("batter_hits", "pitcher_strikeouts", "pitcher_outs",
                   "pitcher_earned_runs", "batter_strikeouts")


# ──────────────────────────────────────────────────────────────────────────────
# Coverage diff (offline-testable)
# ──────────────────────────────────────────────────────────────────────────────

def compute_prop_gap(games, coverage, markets, newest_first=True):
    """Per game, which requested prop markets are missing.

    ``games``: topup_team_odds.load_games dicts (official_date, codeset, commence).
    ``coverage``: {(ET date, codeset): set(prop_key)} of existing prop coverage.
    ``markets``: requested prop_key list.
    Returns (needed, unresolved) where needed is a list of
    {game_pk, official_date, codeset, commence, missing:[markets]} ordered by date
    (newest-first by default), and unresolved is games whose codeset didn't resolve.
    A game with all requested markets present is skipped entirely."""
    want = list(dict.fromkeys(markets))
    needed, unresolved = [], []
    for g in games:
        cs, od = g.get("codeset"), g.get("official_date")
        if not cs or not od:
            unresolved.append(g)
            continue
        have = coverage.get((od, cs), set())
        missing = [m for m in want if m not in have]
        if missing:
            needed.append({"game_pk": g.get("game_pk"), "official_date": od,
                           "codeset": cs, "commence": g.get("commence"),
                           "missing": missing})
    needed.sort(key=lambda x: (x["official_date"], x["game_pk"] or 0),
                reverse=newest_first)
    return needed, unresolved


def plan_cost(needed):
    """Credits to fetch the whole ``needed`` list: per-game 10 x missing-markets,
    plus 1 credit per distinct date for the events lookup."""
    game_cr = sum(CREDITS_PER_MARKET * len(n["missing"]) for n in needed)
    dates = {n["official_date"] for n in needed}
    return game_cr + EVENTS_LOOKUP_COST * len(dates)


# ──────────────────────────────────────────────────────────────────────────────
# Warehouse read
# ──────────────────────────────────────────────────────────────────────────────

def load_prop_coverage(conn, date_from, date_to):
    """{(ET date, codeset): set(prop_key)} of existing player-prop coverage."""
    from sqlalchemy import select
    from pricing_common import et_local_date
    import db_store
    t, ln = db_store.odds_snapshot, db_store.odds_line
    lo, hi = tt._shift(date_from, -1), tt._shift(date_to, 1)
    joined = t.join(ln, ln.c.snapshot_id == t.c.id)
    rows = conn.execute(
        select(t.c.commence_time, t.c.home_code, t.c.away_code, ln.c.prop_key)
        .select_from(joined)
        .where((t.c.sport == SPORT_KEY) & (ln.c.bet_type == "player_prop")
               & (t.c.game_date >= lo) & (t.c.game_date <= hi))
        .distinct()).all()
    cov = defaultdict(set)
    for commence, hc, ac, prop_key in rows:
        if not (hc and ac and hc != ac and prop_key):
            continue
        d = et_local_date(commence) or (commence or "")[:10]
        cov[(d, frozenset({hc, ac}))].add(prop_key)
    return cov


# ──────────────────────────────────────────────────────────────────────────────
# Report + apply
# ──────────────────────────────────────────────────────────────────────────────

def _report(games, needed, unresolved, markets):
    covered_games = len(games) - len(needed) - len(unresolved)
    print(f"\n  prop-eligible completed games in window: {len(games)}")
    print(f"  unresolved team code (skipped): {len(unresolved)}")
    print(f"  fully covered for {list(markets)}: {covered_games}")
    print(f"  games needing props: {len(needed)}")
    gm = Counter(n["official_date"][:7] for n in needed)
    print("\n  needing-props by month (newest first):")
    for m in sorted(gm, reverse=True)[:18]:
        print(f"    {m}:  {gm[m]:4d} games")
    if len(gm) > 18:
        print(f"    ... (+{len(gm)-18} older months)")
    mk = Counter()
    for n in needed:
        for m in n["missing"]:
            mk[m] += 1
    print("\n  missing games per market:")
    for m in markets:
        print(f"    {m:24s} {mk.get(m,0):5d}")
    cost = plan_cost(needed)
    print(f"\n  full-backlog cost = ~{cost} credits "
          f"({len({n['official_date'] for n in needed})} dates x 1 events + "
          f"{sum(len(n['missing']) for n in needed)} market-fetches x 10).")
    return cost


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="date_from", default=PROPS_MIN_DATE,
                    help=f"earliest official_date (default {PROPS_MIN_DATE}, the "
                         "Odds API historical-props start)")
    ap.add_argument("--to", dest="date_to", default=None,
                    help="latest official_date (default: today UTC)")
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS),
                    help="comma list of prop market keys (default: 5 core)")
    ap.add_argument("--max-credits", type=int, default=18000,
                    help="hard cap on credits this run may spend (default 18000; "
                         "set to your monthly budget minus live-ops headroom)")
    ap.add_argument("--oldest-first", action="store_true",
                    help="process oldest games first (default: newest-first, so "
                         "the corpus deepens backward from the most relevant data)")
    ap.add_argument("--apply", action="store_true",
                    help="fetch + write (default: dry-run)")
    ap.add_argument("--yes", action="store_true",
                    help="required with --apply: confirms the credit spend")
    args = ap.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
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
    print(f"MLB player-prop top-up  ({'APPLY' if args.apply else 'DRY-RUN'})")
    print(f"  window: {args.date_from} .. {date_to}   book: {BOOK_KEY}")
    print(f"  markets ({len(markets)}): {','.join(markets)}")
    print(f"  order: {'oldest-first' if args.oldest_first else 'newest-first'}   "
          f"max-credits: {args.max_credits}")

    with db_store.get_engine().connect() as conn:
        games = tt.load_games(conn, args.date_from, date_to)
        coverage = load_prop_coverage(conn, args.date_from, date_to)
        commence_by_date = defaultdict(list)
        for g in games:
            commence_by_date[g["official_date"]].append(g.get("commence"))

    needed, unresolved = compute_prop_gap(games, coverage, markets,
                                          newest_first=not args.oldest_first)
    _report(games, needed, unresolved, markets)

    # How far this run's budget reaches (newest-first prefix that fits).
    fit, spent_est, dates_seen = [], 0, set()
    for n in needed:
        inc = CREDITS_PER_MARKET * len(n["missing"])
        if n["official_date"] not in dates_seen:
            inc += EVENTS_LOOKUP_COST
        if spent_est + inc > args.max_credits:
            break
        spent_est += inc
        dates_seen.add(n["official_date"])
        fit.append(n)
    print(f"\n  this run (cap {args.max_credits}): would fetch {len(fit)} of "
          f"{len(needed)} games (~{spent_est} cr), covering "
          f"{len(dates_seen)} dates.")
    if fit:
        print(f"  reaches back to {min(n['official_date'] for n in fit)} "
              f"(from {max(n['official_date'] for n in fit)}).")

    if not args.apply:
        print("\n  dry-run only — re-run with --apply --yes to fetch.")
        return 0
    if not args.yes:
        print("\n  --apply requires --yes (confirms the credit spend). Aborting.")
        return 2

    # ── fetch loop ──────────────────────────────────────────────────────────
    import json as _json
    from odds_client import (get_historical_events, get_historical_event_odds,
                             get_remaining_credits)
    import player_id_map
    import warehouse
    cfg = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "config.json")))
    api_key = cfg["odds_api_key"]

    events_by_date = {}          # date -> {codeset: event_id}  (1 cr per date)
    spent, captured_games, captured_markets = 0, 0, 0
    for n in needed:
        d, cs = n["official_date"], n["codeset"]
        game_cost = CREDITS_PER_MARKET * len(n["missing"])
        # events-lookup cost only for a date not yet resolved this run
        lookup_cost = 0 if d in events_by_date else EVENTS_LOOKUP_COST
        if spent + lookup_cost + game_cost > args.max_credits:
            print(f"  [stop] budget cap reached at {spent} cr.")
            break
        # Resolve the date's event IDs once (at the date's first tip-off).
        if d not in events_by_date:
            ts = tt.snapshot_ts_for_date(commence_by_date.get(d, [])) or f"{d}T17:00:00Z"
            try:
                events, _ = get_historical_events(api_key, SPORT_KEY, date=ts)
            except Exception as e:
                print(f"  [warn] events {d}: {type(e).__name__}; skipping date.")
                events_by_date[d] = {}
                continue
            spent += EVENTS_LOOKUP_COST
            idx = {}
            for ev in events or []:
                hc = player_id_map.team_code_for_name(ev.get("home_team"))
                ac = player_id_map.team_code_for_name(ev.get("away_team"))
                if hc and ac and hc != ac:
                    idx[frozenset({hc, ac})] = ev.get("id")
            events_by_date[d] = idx
        eid = events_by_date[d].get(cs)
        if not eid:
            continue   # no matching event that date (skip; not charged for game)
        try:
            data, snap_ts = get_historical_event_odds(
                api_key, SPORT_KEY, eid, date=n.get("commence") or f"{d}T23:00:00Z",
                regions=REGIONS, markets=",".join(n["missing"]),
                bookmakers=[BOOK_KEY])
        except Exception as e:
            print(f"  [warn] event-odds {eid}: {type(e).__name__}; skipping.")
            continue
        spent += game_cost
        if data is None:
            continue   # event expired at that timestamp
        warehouse.capture_event_odds(SPORT_KEY, eid, REGIONS,
                                     ",".join(n["missing"]), [BOOK_KEY], data,
                                     captured_at=snap_ts, source="backfill")
        captured_games += 1
        captured_markets += len(n["missing"])
        if captured_games % 50 == 0:
            print(f"    {captured_games} games, {captured_markets} market-fetches, "
                  f"~{spent} cr (remaining: {get_remaining_credits()})")
    print(f"\n  done — spent ~{spent} cr, captured {captured_games} games "
          f"({captured_markets} market-fetches).")
    print(f"  account credits remaining: {get_remaining_credits()}")
    return 0


if __name__ == "__main__":
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    raise SystemExit(main())
