"""nfl_ladder_backfill.py — dense 6h opener→close ladder for NFL team + prop markets.

The opener probe proved NFL props post ~4-5 days out, and the −4h→close CLV test was
negative because −4h is already near the market consensus. To test the OPENER thesis
(the early line ≈ the book's own model, before sharp money corrects it → model-vs-model,
a fair fight) we capture a dense time ladder from the opener down to the close, for all
three books, and later measure how our model's edge/CLV decays as kickoff approaches.

CAPTURE (one historical call per snapshot = 10 × #markets × 1 region):
  * books   : draftkings, fanduel (EXECUTABLE) + pinnacle (SHARP REFERENCE, analysis-only
              per the standing rule — never sized off / recommended; used only to gauge how
              soft a DK/FD number is vs the sharp consensus).
  * PROPS   : 8 player-prop markets at EVERY rung (80 credits) — props are where the edge
              lives, so they get the full dense ladder.
  * TEAM    : h2h/spreads/totals only at the OPENER rung (the earliest offset) — team
              markets have less edge and we already hold their late snapshots (early_12h,
              early_4h, close) in the mirror, so teams only need the opener. (+30 credits
              at that one rung.)
  * offsets : 6h ladder from −114h → −6h, plus close (−10min), ordered NEAREST-KICKOFF
              FIRST so a --max-credits stop only drops the earliest dead-zone snaps (where
              props barely exist).
  * regions : 'us' — the probe confirmed all three books (incl. pinnacle) return here at
              single-region cost (no eu doubling).

SAFETY: get_historical_event_odds caches permanently, so re-runs re-read for 0 credits
(fully resumable). --max-credits is a hard cap checked between offsets. --dry-run prints
the plan and spends nothing. Writes per-offset parquet to nfl_ladder_data/.
"""
import argparse
import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import warehouse_mirror as wm
from odds_client import get_historical_event_odds, get_remaining_credits

SPORT = "americanfootball_nfl"
TEAM_MARKETS = ["h2h", "spreads", "totals"]
PROP_MARKETS = ["player_pass_yds", "player_pass_tds", "player_pass_attempts",
                "player_pass_completions", "player_rush_yds", "player_rush_attempts",
                "player_receptions", "player_reception_yds"]
PROP_ONLY = ",".join(PROP_MARKETS)
TEAM_PLUS = ",".join(TEAM_MARKETS + PROP_MARKETS)
BOOKS = ["draftkings", "fanduel", "pinnacle"]        # pinnacle = analysis-only reference
STORE_DIR = "nfl_ladder_data"
WORKERS = 10
COST_PROPS = 10 * len(PROP_MARKETS) * 1                       # 80
COST_TEAMPLUS = 10 * (len(TEAM_MARKETS) + len(PROP_MARKETS))  # 110

# 6h ladder + close, NEAREST-KICKOFF FIRST.
OFFSETS_H = [10.0 / 60.0] + list(range(6, 115, 6))   # close, −6h, −12h, … −114h
TEAM_OPENER_H = 114                                  # pull team markets only at this rung


def _markets_for(offset_h):
    """Team markets ride along only at the opener rung; every other rung is props-only."""
    return TEAM_PLUS if int(round(offset_h)) == TEAM_OPENER_H else PROP_ONLY


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


def enumerate_games(seasons):
    """Distinct (event_id, commence, home, away, season) for NFL games in the mirror."""
    games = {}
    for y in seasons:
        rows = wm.player_prop_lines(SPORT, date_from=f"{y}-01-01",
                                    date_to=f"{y}-12-31", bookmaker="draftkings") or []
        for r in rows:
            eid = r.get("event_id")
            if eid and eid not in games:
                games[eid] = {"event_id": eid, "commence": r.get("commence_time"),
                              "home": r.get("home"), "away": r.get("away"), "season": y}
    return list(games.values())


def _rows_from_snapshot(data, snap_ts, game, offset_h):
    """Flatten bookmakers→markets→outcomes into research rows."""
    if not data:
        return []
    out = []
    for bk in data.get("bookmakers", []):
        book = bk.get("key")
        if book not in BOOKS:
            continue
        for mk in bk.get("markets", []):
            m = mk.get("key")
            is_prop = m.startswith("player_")
            for oc in mk.get("outcomes", []):
                out.append({
                    "event_id": game["event_id"], "season": game["season"],
                    "commence": game["commence"], "home": game["home"],
                    "away": game["away"], "offset_h": round(offset_h, 3),
                    "snapshot_ts": snap_ts, "book": book, "market": m,
                    "player": oc.get("description") if is_prop else None,
                    "name": oc.get("name"), "point": oc.get("point"),
                    "price": oc.get("price"),
                })
    return out


def _fetch(key, game, offset_h, markets):
    c0 = dt.datetime.fromisoformat(game["commence"].replace("Z", "+00:00"))
    date = (c0 - dt.timedelta(hours=offset_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        data, snap_ts = get_historical_event_odds(
            key, SPORT, game["event_id"], date, regions="us",
            markets=markets, bookmakers=BOOKS)
    except Exception:
        return []
    return _rows_from_snapshot(data, snap_ts, game, offset_h)


def run(seasons, max_credits, dry_run):
    games = enumerate_games(seasons)
    n_calls = len(games) * len(OFFSETS_H)
    ceiling = len(games) * ((len(OFFSETS_H) - 1) * COST_PROPS + COST_TEAMPLUS)
    print("=" * 96)
    print(f"  NFL LADDER BACKFILL — {len(games)} games × {len(OFFSETS_H)} offsets "
          f"= {n_calls:,} calls")
    print(f"  books={BOOKS}  props={COST_PROPS}/call every rung, "
          f"team+props={COST_TEAMPLUS} only at −{TEAM_OPENER_H}h opener")
    print(f"  ceiling ≈ {ceiling:,} credits   cap={max_credits:,}")
    print(f"  offsets (nearest-first): {[round(o,1) for o in OFFSETS_H]}")
    print("=" * 96)
    if dry_run:
        print("  (dry-run — no credits spent)")
        return

    key = _load_key()
    if not key:
        print("  NO API KEY (app.load_config / ODDS_API_KEY). Aborting — nothing spent.")
        return
    os.makedirs(STORE_DIR, exist_ok=True)
    start_rem = get_remaining_credits()
    print(f"  start credits: {start_rem}")

    for offset in OFFSETS_H:
        markets = _markets_for(offset)
        est = len(games) * (COST_TEAMPLUS if markets == TEAM_PLUS else COST_PROPS)
        spent = (start_rem - get_remaining_credits()) if start_rem else 0
        if spent + est > max_credits:
            print(f"  [cap] stopping before −{offset:.1f}h "
                  f"(spent {spent:,}, next offset would exceed {max_credits:,})")
            break
        tag = "close" if offset < 1 else f"{int(offset):03d}h"
        path = os.path.join(STORE_DIR, f"ladder__{tag}.parquet")
        if os.path.exists(path):
            print(f"  −{offset:>6.1f}h  already stored ({path}) — skipping")
            continue
        rows = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(_fetch, key, g, offset, markets) for g in games]
            for f in as_completed(futs):
                rows.extend(f.result())
        if rows:
            pd.DataFrame(rows).to_parquet(path, index=False)
        rem = get_remaining_credits()
        print(f"  −{offset:>6.1f}h  rows={len(rows):>7,}  → {path}   "
              f"(remaining {rem:,}, spent {start_rem-rem:,})")
    print("=" * 96)
    print(f"  DONE. total spent ≈ {start_rem - get_remaining_credits():,} credits.")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    ap.add_argument("--max-credits", type=int, default=1_450_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    run(seasons, args.max_credits, args.dry_run)


if __name__ == "__main__":
    main()
