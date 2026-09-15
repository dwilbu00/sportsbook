"""nfl_alt_capture.py — capture historical DK/FD ALTERNATE-line odds for the skewed props.

Phase 1 showed rush_yds/recv_yds are strongly right-skewed; if DK/FD price their alternate
ladders lazily (symmetric / flat margin off the main line) they'd misprice the fat over-tail.
This grabs the CLOSING alternate ladders (player_rush_yds_alternate, player_reception_yds_
alternate) for DK+FD across past games so we can backtest each offered alternate (line, odds)
vs the actual outcome.

Cost: 10 x #markets x 1 region = 20 credits/call (one call per game). Cached calls = 0.
Stores to a LOCAL research parquet (nfl_alt_data/alt__{season}.parquet) — re-fetchable until
the credit window closes; persist to Azure only if the edge proves real. --dry-run / --max-credits
guard the spend. Runs on Doug's machine (where the parquets live).
"""
import argparse
import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import warehouse_mirror as wm
from odds_client import get_historical_event_odds, get_remaining_credits

SPORT = "americanfootball_nfl"
ALT_MARKETS = ["player_rush_yds_alternate", "player_reception_yds_alternate"]
MARKETS = ",".join(ALT_MARKETS)
BOOKS = ["draftkings", "fanduel"]
STORE = "nfl_alt_data"
WORKERS = 8
COST_PER_CALL = 10 * len(ALT_MARKETS) * 1          # 20
CLOSE_MIN = 10                                      # snapshot at commence - 10 min


def _load_key():
    for mod in ("app", "config"):
        try:
            m = __import__(mod)
            if hasattr(m, "load_config"):
                k = m.load_config().get("odds_api_key")
                if k:
                    return k
        except Exception:
            pass
    return os.environ.get("ODDS_API_KEY", "")


def enumerate_games(season):
    games = {}
    rows = wm.player_prop_lines(SPORT, date_from=f"{season}-01-01",
                                date_to=f"{season}-12-31", bookmaker="draftkings") or []
    for r in rows:
        eid = r.get("event_id")
        if eid and eid not in games and r.get("commence_time"):
            games[eid] = {"event_id": eid, "commence": r.get("commence_time"),
                          "home": r.get("home"), "away": r.get("away"), "season": season}
    return list(games.values())


def _fetch(key, game):
    c0 = dt.datetime.fromisoformat(game["commence"].replace("Z", "+00:00"))
    date = (c0 - dt.timedelta(minutes=CLOSE_MIN)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        data, snap = get_historical_event_odds(key, SPORT, game["event_id"], date,
                                               regions="us", markets=MARKETS, bookmakers=BOOKS)
    except Exception:
        return []
    if not data:
        return []
    out = []
    for bk in data.get("bookmakers", []):
        if bk.get("key") not in BOOKS:
            continue
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                out.append({"event_id": game["event_id"], "season": game["season"],
                            "commence": game["commence"], "home": game["home"],
                            "away": game["away"], "snapshot_ts": snap,
                            "book": bk["key"], "market": mk.get("key"),
                            "player": oc.get("description"), "side": oc.get("name"),
                            "point": oc.get("point"), "price": oc.get("price")})
    return out


def run(seasons, max_credits, dry_run):
    games = []
    for s in seasons:
        games += enumerate_games(s)
    ceiling = len(games) * COST_PER_CALL
    print("=" * 90)
    print(f"  NFL ALT CAPTURE — {len(games)} games × closing × {ALT_MARKETS} (DK+FD)")
    print(f"  ~{COST_PER_CALL}/call  ceiling ≈ {ceiling:,} credits  cap={max_credits:,}")
    print("=" * 90)
    if dry_run:
        print("  (dry-run — no spend)")
        return
    key = _load_key()
    if not key:
        print("  NO API KEY. Aborting.")
        return
    os.makedirs(STORE, exist_ok=True)
    spent_est = 0
    for s in seasons:
        path = os.path.join(STORE, f"alt__{s}.parquet")
        if os.path.exists(path):
            print(f"  {s}: already captured ({path}) — skipping")
            continue
        sg = [g for g in games if g["season"] == s]
        if spent_est + len(sg) * COST_PER_CALL > max_credits:
            print(f"  [cap] stopping before {s} (est {spent_est:,} + {len(sg)*COST_PER_CALL:,} > {max_credits:,})")
            break
        rows = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(_fetch, key, g) for g in sg]
            for f in as_completed(futs):
                rows.extend(f.result())
        if rows:
            pd.DataFrame(rows).to_parquet(path, index=False)
        spent_est += len(sg) * COST_PER_CALL
        rem = get_remaining_credits()
        print(f"  {s}: {len(sg)} games, {len(rows):,} alt rows → {path}  "
              f"(est spent ≤ {spent_est:,}, remaining {rem if rem is not None else '?'})")
    print("=" * 90)
    rem = get_remaining_credits()
    print(f"  DONE. est spent ≤ {spent_est:,}; remaining {rem if rem is not None else '?'}.")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--max-credits", type=int, default=60000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    run(seasons, args.max_credits, args.dry_run)


if __name__ == "__main__":
    main()
