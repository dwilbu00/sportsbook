"""
Backfill DraftKings closing-line value (CLV) into the wagers ledger, on demand,
with a hard credit budget so it can never blow your monthly Odds-API quota.

Why this exists
---------------
CLV in My Bets was wrong for a DraftKings-only bettor in two ways:
  1. It compared the user's DK executed price against a best-of-book / de-vigged
     CONSENSUS close from the odds warehouse (which never stores DraftKings) —
     optimistically biased.
  2. The warehouse prop lookup ignores the LINE, and the fill hardcoded
     close_line = the bet's line, so a DK line move produced a bogus cross-line
     CLV number.

This utility fetches DraftKings' historical closing snapshot for each of your
started prop bets straight from The Odds API historical event-odds endpoint
(bookmakers=draftkings, date=commence_time -> nearest snapshot at-or-before the
close), reads DK's price at the EXACT line you bet, and writes a true DK-vs-DK,
same-line CLV.

When DK's standard close does NOT carry the exact line you bet, the row is left
UNFILLED (blank CLV), not stamped with a mismatched line: the settled table
shows only clv_pct (never close_line), so recording a different line would be
invisible AND would permanently strand the row (close_price set -> never
retried). Leaving it unfilled keeps it eligible for a future alternate-line pass
and costs nothing to retry (permanent snapshot cache). This covers genuine DK
line moves and alternate-line / safe-mode ("N+") bets, whose exact line lives in
the '_alternate' market this tool does not yet fetch (see Out of scope).

Only player-prop wagers are handled here; team markets keep their warehouse CLV
path. Historical responses are cached permanently by odds_client, so re-running
only ever pays for genuinely new games. attach_clv no longer fills props from
the warehouse, so a prop's CLV appears only after this backfill runs.

  IMPORTANT after first deploy: props settled before this change still carry the
  old (biased/cross-line) warehouse CLV. Run once with --refresh to clear and
  recompute them from DraftKings.

Cost: 10 x prop-markets x 1 region PER GAME (one call covers all your bets on
that game). Scoped to your actual bets this is small; --dry-run prints the exact
NEW (uncached) cost first, and --max-credits caps the spend.

Examples
--------
    # See the plan + exact cost, spend nothing:
    python backfill_dk_clv.py --sport mlb --dry-run

    # Real run, tightly budgeted:
    python backfill_dk_clv.py --sport mlb --max-credits 200

    # Recompute after a correction, or clear stale pre-deploy prop CLV (clears
    # existing prop CLV first; re-fetch is free thanks to the permanent cache):
    python backfill_dk_clv.py --sport mlb --refresh
"""
import argparse
import json
import os
from datetime import datetime, timezone

import requests

import wagers
from db_store import normalize_name
from odds_client import (
    get_historical_event_odds,
    is_historical_event_cached,
    get_upcoming_events,
    dk_prop_lines,
    american_to_implied_prob,
    get_remaining_credits,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

# short flag -> full Odds-API sport key (matches backfill_historical_odds.py)
SPORT_MAP = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
}

REGIONS = "us"  # DraftKings lives in the US region


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lines_equal(a, b):
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) < 1e-6


def _same_sport(stored, full):
    """A wager's stored sport_key matches the run's sport — accept the full Odds-
    API key ('baseball_mlb') or the bare league suffix ('mlb')."""
    s = (stored or "").lower()
    return s == full or s == full.split("_")[-1]


def _implied(price):
    """American price -> implied probability (0-1), or None. Vig included on both
    sides, so DK close vs DK executed compares like for like."""
    try:
        return american_to_implied_prob(int(price))
    except (TypeError, ValueError):
        return None


def _is_prop_candidate(row, sport_full, now):
    """A started prop wager of this sport that still needs a DK close."""
    return (row.get("bet_type") == "player_prop"
            # Re-check in Python: the Blob/local read ignores a SQL ``where``
            # filter and returns every row, so without this an already-filled
            # prop would be reconsidered (and re-fetched) on that backend.
            and row.get("close_price") is None
            and _same_sport(row.get("sport_key"), sport_full)
            and wagers._commence_passed(row, now))


def refresh_reset(sport_full):
    """Clear existing prop CLV for this sport so rows left blank by an earlier
    line-move (or filled by the old warehouse path pre-deploy) recompute; the
    re-fetch is free via the permanent snapshot cache. Best-effort."""
    ids = [r.get("wager_id") for r in wagers.read_wagers()
           if r.get("bet_type") == "player_prop"
           and _same_sport(r.get("sport_key"), sport_full)
           and r.get("wager_id")]
    if not ids:
        return
    try:
        cleared = wagers.reset_clv(wager_ids=ids)
        print(f"  [refresh] cleared CLV on {cleared} prop wager(s).")
    except Exception as exc:
        print(f"  [warn] --refresh reset failed: {exc}")


def group_by_event(candidates, all_rows, sport_full):
    """event_id -> {commence, markets:set(prop_key), rows:[candidate rows]}.

    ``markets`` is the FULL set of prop markets across EVERY prop wager on the
    event (filled or not), not just the unfilled candidates. Requesting the same
    market set every run keeps the permanent-cache key stable, so once an event
    is fetched, re-runs are free — a set that shrank as sibling props filled
    would hash to a new key and re-bill the API on every run.

    Wagers missing event_id / commence_time / prop_key can't be fetched and are
    dropped from ``rows`` (counted as skipped)."""
    full_markets = {}
    for r in all_rows:
        if r.get("bet_type") != "player_prop":
            continue
        if not _same_sport(r.get("sport_key"), sport_full):
            continue
        eid, prop = r.get("event_id"), r.get("prop_key")
        if eid and prop:
            full_markets.setdefault(eid, set()).add(prop)

    groups = {}
    skipped = 0
    for r in candidates:
        eid = r.get("event_id")
        commence = r.get("commence_time")
        prop = r.get("prop_key")
        if not eid or not commence or not prop:
            skipped += 1
            continue
        g = groups.setdefault(eid, {"commence": commence,
                                    "markets": set(), "rows": []})
        g["markets"] |= full_markets.get(eid, {prop})
        g["rows"].append(r)
    return groups, skipped


def dk_close_for_wager(dk_offers, row):
    """Resolve DraftKings' closing price/line for one prop wager.

    ``dk_offers`` = dk_prop_lines(data, prop_key) for this wager's prop. Returns
    ``(close_price, close_line, clv_pct)`` only when DK posted the EXACT line the
    user bet at the close (a true DK-vs-DK, same-line comparison). Returns None
    otherwise — DK didn't post this player/side, OR DK's standard close carried a
    different line (a genuine move, or an alternate-line/safe-mode bet whose line
    lives in the '_alternate' market). Unfilled rows keep clv blank and are
    retried free next run (permanent cache); nothing misleading is written.
    """
    norm = normalize_name(row.get("player"))
    direction = (row.get("direction") or "OVER").upper()
    side_key = "over_price" if direction == "OVER" else "under_price"
    bet_line = row.get("line")

    mine = [o for o in dk_offers if normalize_name(o.get("player")) == norm]
    priced = [o for o in mine if o.get(side_key) is not None]
    if not priced:
        return None  # DK didn't post this player/side at the close

    exact = next((o for o in priced if _lines_equal(o.get("line"), bet_line)),
                 None)
    if exact is None:
        return None  # line moved / alternate line — leave unfilled (no bogus row)

    close_price = exact[side_key]
    close_line = exact["line"]
    close_imp = _implied(close_price)
    executed = wagers._executed_implied(row.get("executed_price"))
    clv = (round((close_imp - executed) * 100.0, 2)
           if close_imp is not None and executed is not None else None)
    return close_price, close_line, clv


def main():
    p = argparse.ArgumentParser(
        description="Backfill DraftKings closing-line CLV into the wagers ledger "
                    "(budget-guarded).")
    p.add_argument("--sport", choices=list(SPORT_MAP.keys()), default="mlb")
    p.add_argument("--max-credits", type=int, default=2000,
                   help="Hard cap on credits this run may spend. Default 2000.")
    p.add_argument("--reserve", type=int, default=0,
                   help="Stop if remaining account credits would drop below this.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan and estimated cost without calling the API.")
    p.add_argument("--refresh", action="store_true",
                   help="Clear existing prop CLV for this sport first, then "
                        "recompute (re-fetch is free via the permanent cache).")
    args = p.parse_args()

    # Target the SQL backend when the SQL_* secrets are configured (mirrors the
    # app's boot promotion; outside Streamlit these aren't in the env yet).
    # Without this the ledger read/write would hit the LOCAL store instead of
    # prod SQL. Surface which backend we actually landed on so a misconfigured
    # run (missing secrets.toml) can't silently write to the wrong store.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
        if db_store.enabled():
            print("  Backend: Azure SQL (prod).")
        else:
            print("  [warn] SQL not configured -- reading/writing the LOCAL/Blob "
                  "store, NOT prod SQL. Set SQL_* in .streamlit/secrets.toml to "
                  "target production.")
    except Exception as exc:
        print(f"  [warn] SQL secret promotion failed ({exc}); using local store.")

    sport_full = SPORT_MAP[args.sport]
    cfg = load_config()
    api_key = cfg["odds_api_key"]
    now = datetime.now(timezone.utc)

    print(f"\n=== Backfill DraftKings CLV: {sport_full} player props ===")
    print(f"  Region(s): {REGIONS}   Budget cap: {args.max_credits} credits")

    if args.refresh:
        refresh_reset(sport_full)

    # One ledger read (surfaces an outage instead of masking it as "empty").
    all_rows, err = wagers.read_wagers_with_status()
    if err is not None:
        print(f"  [error] Could not read the wagers ledger: {err}")
        print("  The durable store may be temporarily unreachable; try again.")
        raise SystemExit(2)

    candidates = [r for r in all_rows if _is_prop_candidate(r, sport_full, now)]
    if not candidates:
        print("  No started prop wagers need a DK close. Nothing to do.")
        return
    groups, skipped = group_by_event(candidates, all_rows, sport_full)
    if skipped:
        print(f"  [note] {skipped} wager(s) missing event_id/commence/prop_key "
              f"— cannot fetch; skipped.")
    if not groups:
        print("  No fetchable events. Done.")
        return

    per_event_cost = {eid: 10 * len(g["markets"]) * 1 for eid, g in groups.items()}
    total_cost = sum(per_event_cost.values())
    n_wagers = sum(len(g["rows"]) for g in groups.values())
    # Freshest games first: that is how a budget-trimmed real run spends.
    order = sorted(groups, key=lambda e: groups[e]["commence"], reverse=True)

    def _cached(eid):
        return is_historical_event_cached(
            sport_full, eid, groups[eid]["commence"], regions=REGIONS,
            markets=",".join(sorted(groups[eid]["markets"])),
            bookmakers=["draftkings"])

    new_cost = sum(per_event_cost[eid] for eid in order if not _cached(eid))
    print(f"  {len(groups)} event(s), {n_wagers} prop wager(s) needing CLV.")
    print(f"  Full cost if nothing were cached: ~{total_cost} credits "
          f"(10 x prop-markets/game).")
    print(f"  Estimated NEW spend (uncached events only): ~{new_cost} credits.")

    if args.dry_run:
        for eid in order:
            g = groups[eid]
            tag = " [cached/free]" if _cached(eid) else ""
            print(f"    - {g['commence']}  {eid}  "
                  f"markets={','.join(sorted(g['markets']))}  "
                  f"~{per_event_cost[eid]} cr{tag}  ({len(g['rows'])} bet(s))")
        print("  [dry-run] No API calls made. Re-run without --dry-run to fetch.")
        return

    # If a reserve floor is set but we don't yet know the account balance, make
    # one FREE call so the reserve check binds from the very first event (the
    # balance is otherwise unknown until the first paid fetch's headers arrive).
    if args.reserve > 0 and get_remaining_credits() is None:
        try:
            get_upcoming_events(api_key, sport_full)
        except Exception:
            pass  # best-effort; reserve then binds from the 2nd event onward

    spent = 0

    def _budget_ok(cost):
        if spent + cost > args.max_credits:
            return False
        remaining = get_remaining_credits()
        if remaining is not None and remaining - cost < args.reserve:
            return False
        return True

    filled = {}
    n_exact = n_unmatched = 0
    events_done = written = 0
    try:
        for eid in order:
            g = groups[eid]
            markets = ",".join(sorted(g["markets"]))
            cached = _cached(eid)
            # Cache hits (and cached 404s) cost 0 — don't let them trip the cap.
            est_cost = 0 if cached else per_event_cost[eid]
            if not _budget_ok(est_cost):
                print("  [stop] Budget/reserve reached; remaining events skipped.")
                break
            try:
                data, _snap = get_historical_event_odds(
                    api_key, sport_full, eid, date=g["commence"],
                    regions=REGIONS, markets=markets, bookmakers=["draftkings"])
            except requests.exceptions.RequestException as e:
                print(f"  [warn] {eid}: request failed ({e}); skipping.")
                continue
            except ValueError as e:  # malformed JSON body
                print(f"  [warn] {eid}: bad response ({e}); skipping.")
                continue
            events_done += 1
            if data is None:
                # 404 / expired snapshot — no credits charged.
                print(f"  [note] {eid}: no snapshot (expired); skipping.")
                continue
            # Bill only a genuine paid fetch; cache hits and 404s are free.
            if not cached:
                spent += per_event_cost[eid]
            need = {row.get("prop_key") for row in g["rows"]}
            dk_by_prop = {prop: dk_prop_lines(data, prop) for prop in need}
            for row in g["rows"]:
                res = dk_close_for_wager(dk_by_prop.get(row.get("prop_key"), []),
                                         row)
                if res is None:
                    n_unmatched += 1
                    continue
                close_price, close_line, clv = res
                filled[row["wager_id"]] = {
                    "close_price": close_price,
                    "close_line": close_line,
                    "clv_pct": clv,
                }
                n_exact += 1
    except KeyboardInterrupt:
        print("\n  [interrupt] Writing progress before exit...")
    finally:
        # Always persist what we computed, even on an unexpected error, so a
        # mid-run failure never discards already-fetched CLV.
        written = wagers.apply_clv_updates(filled)

    print(f"\n=== Done. Spent ~{spent} credits across {events_done} event(s). ===")
    print(f"  Same-line CLV filled: {n_exact}   "
          f"No exact DK line at close (left blank): {n_unmatched}")
    print(f"  Durably wrote {written} wager(s).")
    print(f"  Account credits remaining: {get_remaining_credits()}")


if __name__ == "__main__":
    from cli_encoding import configure_stdio
    configure_stdio()
    main()
