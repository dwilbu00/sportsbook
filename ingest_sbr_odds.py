#!/usr/bin/env python3
"""Ingest the SBR historical team-market odds dataset into the Azure warehouse.

Source: ``cache/mlb_odds_dataset.json`` (Sports Book Review scrape, 2021-2025),
DraftKings-only (the only book we bet), scoped by default to 2023-forward — the
years the StatsAPI warehouse can grade against — and to regular + postseason game
types.

Each game's DraftKings ``currentLine`` (moneyline / pointspread / totals) is
transformed into the Odds-API v4 event-odds payload shape and written via
``warehouse.capture_event_odds`` — the SAME writer the paid Odds-API historical
backfill uses — so the rows are indistinguishable from API-sourced ones and read
back through ``warehouse.load_team_market_store`` / ``team_market_lines`` unchanged
(the join prefers the canonical SFBB team codes that capture's ``_enrich_ids``
stamps, so franchise-name drift like the A's rebrand is handled).

One snapshot per game with ``captured_at = commence`` (the game's start), so it is
the CLOSING snapshot by construction (``_closing_sort_key`` picks nearest
at-or-before commence). Writes are write-once / idempotent (``uq_odds_snapshot``);
re-runs skip rows already present. Synthetic event ids are prefixed ``sbr-`` so
they never collide with the API's hash ids for the same real game (the backtest
dedups by team-code key regardless).

Dry-run by default (reports exactly what WOULD be written); ``--apply`` performs
the writes. No network, no credits: reads local JSON, writes the DB.
"""

import argparse
import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "cache", "mlb_odds_dataset.json")

SPORT_KEY = "baseball_mlb"
BOOK_KEY = "draftkings"
BOOK_TITLE = "DraftKings"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"        # → _kind_for_markets == "team"
EVENT_PREFIX = "sbr"

# Regular season + the four postseason rounds (wild-card F, division D, league L,
# world series W). Spring training (S), All-Star (A) and Unknown are dropped by
# default: they have no warehouse game facts to grade against, so their lines
# would sit unused. Override with --game-types.
DEFAULT_GAME_TYPES = frozenset({"R", "F", "D", "L", "W"})


# ──────────────────────────────────────────────────────────────────────────────
# Pure transform (no SQL, no network) — unit-tested in test_ingest_sbr_odds.py
# ──────────────────────────────────────────────────────────────────────────────

def _clean_team(name):
    """SBR's 2025 A's rebrand doubles the field ("Athletics Athletics"); collapse
    it to the canonical odds-feed short "Athletics". team_code_for_name resolves
    every A's spelling to ATH regardless, but keep the stored name clean."""
    if name == "Athletics Athletics":
        return "Athletics"
    return name


def _num(v):
    """A finite American-odds / point number, or None (rejects bool/str/null)."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _dk_current(offers):
    """The DraftKings ``currentLine`` (closing) dict from a market's offer list,
    or None when DK is absent / has no current line."""
    for o in offers or []:
        if o.get("sportsbook") == BOOK_KEY:
            return (o.get("currentLine") or {}) or None
    return None


def build_event_id(date10, away_short, home_short):
    """Stable synthetic id for a game: ``sbr-<UTC date>-<away>-<home>``."""
    return f"{EVENT_PREFIX}-{date10}-{away_short}-{home_short}"


def build_payload(event_id, gv, ml, ps, tt):
    """Odds-API v4 event-odds payload (DK-only) from the SBR DK currentLine dicts.

    Returns (payload, markets_present) — a list like ["moneyline","total"] — or
    (None, []) when no market carries a complete, valid two-sided price."""
    home = _clean_team(gv["homeTeam"]["fullName"])
    away = _clean_team(gv["awayTeam"]["fullName"])
    dk_markets, present = [], []

    if ml:
        ho, ao = _num(ml.get("homeOdds")), _num(ml.get("awayOdds"))
        if ho is not None and ao is not None:
            dk_markets.append({"key": "h2h", "outcomes": [
                {"name": home, "price": ho}, {"name": away, "price": ao}]})
            present.append("moneyline")

    if ps:
        ho, ao = _num(ps.get("homeOdds")), _num(ps.get("awayOdds"))
        hs, as_ = _num(ps.get("homeSpread")), _num(ps.get("awaySpread"))
        if None not in (ho, ao, hs, as_):
            dk_markets.append({"key": "spreads", "outcomes": [
                {"name": home, "price": ho, "point": hs},
                {"name": away, "price": ao, "point": as_}]})
            present.append("spread")

    if tt:
        oo, uo = _num(tt.get("overOdds")), _num(tt.get("underOdds"))
        total = _num(tt.get("total"))
        if None not in (oo, uo, total):
            dk_markets.append({"key": "totals", "outcomes": [
                {"name": "Over", "price": oo, "point": total},
                {"name": "Under", "price": uo, "point": total}]})
            present.append("total")

    if not dk_markets:
        return None, []
    payload = {
        "id": event_id,
        "commence_time": gv.get("startDate"),
        "home_team": home,
        "away_team": away,
        "bookmakers": [{"key": BOOK_KEY, "title": BOOK_TITLE,
                        "markets": dk_markets}],
    }
    return payload, present


def scan(data, min_year=2023, max_year=None, allowed_types=DEFAULT_GAME_TYPES):
    """Full pass over the dataset → (candidates list, skips Counter). Skip reasons:
    pre_year, wrong_type, not_final, no_dk_line, no_startdate."""
    skips = Counter()
    cands = []
    seen_ids = set()
    for date10 in sorted(data.keys()):
        for g in data.get(date10) or []:
            gv = (g or {}).get("gameView") or {}
            od = (g or {}).get("odds") or {}
            start = gv.get("startDate") or ""
            year = start[:4]
            if not year.isdigit():
                skips["no_startdate"] += 1
                continue
            yr = int(year)
            if yr < min_year or (max_year is not None and yr > max_year):
                skips["pre_year"] += 1
                continue
            gt = gv.get("gameType")
            if allowed_types is not None and gt not in allowed_types:
                skips["wrong_type"] += 1
                continue
            if not str(gv.get("gameStatusText") or "").startswith("Final"):
                skips["not_final"] += 1
                continue
            ml = _dk_current(od.get("moneyline"))
            ps = _dk_current(od.get("pointspread"))
            tt = _dk_current(od.get("totals"))
            aw = ((gv.get("awayTeam") or {}).get("shortName")) or "AWY"
            hm = ((gv.get("homeTeam") or {}).get("shortName")) or "HOM"
            base = build_event_id(start[:10], aw, hm)
            event_id, n = base, 1
            while event_id in seen_ids:
                n += 1
                event_id = f"{base}-g{n}"
            payload, present = build_payload(event_id, gv, ml, ps, tt)
            if payload is None:
                skips["no_dk_line"] += 1
                continue
            seen_ids.add(event_id)
            cands.append({"event_id": event_id, "commence": start,
                          "payload": payload, "present": present,
                          "game_type": gt, "year": yr})
    return cands, skips


# ──────────────────────────────────────────────────────────────────────────────
# Warehouse I/O
# ──────────────────────────────────────────────────────────────────────────────

def existing_event_ids():
    """The set of ``sbr-*`` team event ids already in the warehouse (for skip +
    reporting). Best-effort: {} on any error / SQL off."""
    try:
        import db_store
        from sqlalchemy import select
        t = db_store.odds_snapshot
        with db_store.get_engine().connect() as conn:
            rows = conn.execute(
                select(t.c.event_id).where(
                    (t.c.sport == SPORT_KEY)
                    & (t.c.event_id.like(EVENT_PREFIX + "-%"))
                ).distinct()
            ).all()
        return {r[0] for r in rows}
    except Exception as e:
        print(f"  (could not read existing sbr snapshots: {type(e).__name__})")
        return set()


def _report(cands, skips, existing):
    by_year = Counter(c["year"] for c in cands)
    by_type = Counter(c["game_type"] for c in cands)
    mkt = Counter()
    for c in cands:
        for m in c["present"]:
            mkt[m] += 1
    already = sum(1 for c in cands if c["event_id"] in existing)
    new = len(cands) - already

    print(f"\n  candidates (DK team-market games): {len(cands)}")
    print("    by year: " + ", ".join(f"{y}:{by_year[y]}" for y in sorted(by_year)))
    print("    by game_type: "
          + ", ".join(f"{t}:{by_type[t]}" for t in sorted(by_type, key=str)))
    print(f"    market coverage: moneyline {mkt['moneyline']}, "
          f"spread {mkt['spread']}, total {mkt['total']}")
    print("  skipped: " + ", ".join(f"{k}:{v}" for k, v in sorted(skips.items())))
    print(f"  already in warehouse (sbr-): {len(existing)}  "
          f"(of these candidates: {already})")
    print(f"  NEW to write: {new}")
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="path to mlb_odds_dataset.json")
    ap.add_argument("--min-year", type=int, default=2023,
                    help="earliest game year to ingest (default 2023)")
    ap.add_argument("--max-year", type=int, default=None,
                    help="latest game year to ingest (default: no upper bound)")
    ap.add_argument("--game-types", default=None,
                    help="comma list of gameType codes to include, or 'all' "
                         "(default: R,F,D,L,W = regular + postseason)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of NEW writes (smoke test)")
    ap.add_argument("--apply", action="store_true",
                    help="perform the warehouse writes (default: dry-run only)")
    args = ap.parse_args()

    if (args.game_types or "").lower() == "all":
        allowed = None
    elif args.game_types:
        allowed = frozenset(t.strip().upper() for t in args.game_types.split(",")
                            if t.strip())
    else:
        allowed = DEFAULT_GAME_TYPES

    # Promote SQL secrets so the warehouse is live AND team_code_for_name resolves
    # (both back onto the same Azure config the app uses).
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception as e:
        print(f"  (secret promotion failed: {type(e).__name__})")

    import warehouse
    sql_on = False
    try:
        import db_store as _dbs
        sql_on = _dbs.enabled()
    except Exception:
        pass

    print(f"SBR team-market ingest  ({'APPLY' if args.apply else 'DRY-RUN'})")
    print(f"  dataset: {args.dataset}")
    print(f"  scope: years >= {args.min_year}"
          + (f" and <= {args.max_year}" if args.max_year else "")
          + f", game_types = {'ALL' if allowed is None else ','.join(sorted(allowed))}")
    print(f"  warehouse backend: {warehouse.storage_backend()} "
          f"(SQL {'ON' if sql_on else 'OFF'})")

    if not os.path.exists(args.dataset):
        print(f"  dataset not found: {args.dataset}")
        return 2
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    cands, skips = scan(data, min_year=args.min_year, max_year=args.max_year,
                        allowed_types=allowed)
    existing = existing_event_ids() if sql_on else set()
    new = _report(cands, skips, existing)

    if not args.apply:
        print("\n  dry-run only — re-run with --apply to write.")
        return 0

    if not sql_on:
        print("\n  SQL warehouse not enabled — cannot --apply. Aborting.")
        return 2

    to_write = [c for c in cands if c["event_id"] not in existing]
    if args.limit is not None:
        to_write = to_write[:args.limit]
    print(f"\n  writing {len(to_write)} snapshots via capture_event_odds ...")
    written = 0
    for i, c in enumerate(to_write, 1):
        warehouse.capture_event_odds(
            SPORT_KEY, c["event_id"], REGIONS, MARKETS, [BOOK_KEY],
            c["payload"], captured_at=c["commence"])
        written += 1
        if i % 500 == 0:
            print(f"    {i}/{len(to_write)} ...")
    print(f"  done — {written} capture calls issued "
          f"(write-once; duplicates are silently skipped).")
    # Confirm what actually landed.
    after = existing_event_ids()
    print(f"  sbr- snapshots in warehouse now: {len(after)} "
          f"(was {len(existing)}; delta {len(after) - len(existing)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
