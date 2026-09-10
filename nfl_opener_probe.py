"""nfl_opener_probe.py — how early does The Odds API have historical NFL player props?

The −4h→close CLV test was negative, but −4h is NOT the opener (props sharpen earlier in
the week). To test the opener thesis we need opener snapshots — which we've never captured.
Before a full historical backfill, this PROBE fires a small sweep on a handful of games at
increasingly-early offsets and reports, for each, the ACTUAL snapshot timestamp the API
returns and how many prop markets/players are present. That tells us how early NFL props
actually exist historically (the 'props post same-day' note in backfill_precise was for
MLB — NFL is untested).

Cost: 10 × #markets × #regions per call = ~80 credits/call. Default sweep = a few games ×
several offsets → a few thousand credits (trivial against the pre-Sep-21 1M+ balance).
Cached calls re-read for 0 credits, so re-runs are free. Spends credits → run only on go.
"""
import argparse
import datetime as dt

from odds_client import get_historical_event_odds

SPORT = "americanfootball_nfl"
NFL_PROPS = ("player_pass_yds,player_pass_tds,player_pass_attempts,"
             "player_pass_completions,player_rush_yds,player_rush_attempts,"
             "player_receptions,player_reception_yds")
BOOKS = ["draftkings", "fanduel"]

# (event_id, commence_time) — sampled 2024 NFL games (Sun + Mon night).
GAMES = [
    ("a207d1faa78a276de49cdebb7ef4fc0e", "2024-09-30T23:30:00Z"),  # MIA-TEN (Mon)
    ("ce925c8cb892e1806399345e7885828d", "2024-09-30T00:15:00Z"),  # BAL-BUF (Sun night)
    ("3f180e883b429700dd32ff2133ac37ef", "2024-09-29T20:25:00Z"),  # LV-CLE  (Sun)
    ("b839e959b4e28cdd96341dbcf99d4a42", "2024-09-29T20:25:00Z"),  # LAC-KC  (Sun)
]
OFFSETS_H = [240, 168, 120, 96, 72, 48, 24, 12, 6, 4]   # hours before commence


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
    import os
    return os.environ.get("ODDS_API_KEY", "")


def _prop_summary(data):
    """(#distinct prop markets, #distinct players) present in the snapshot, per any book."""
    if not data:
        return 0, 0
    markets, players = set(), set()
    for bk in data.get("bookmakers", []):
        if bk.get("key") not in BOOKS:
            continue
        for mk in bk.get("markets", []):
            markets.add(mk.get("key"))
            for oc in mk.get("outcomes", []):
                if oc.get("description"):
                    players.add(oc.get("description"))
    return len(markets), len(players)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the calls + est. credits, spend nothing")
    args = ap.parse_args()

    n_calls = len(GAMES) * len(OFFSETS_H)
    est = n_calls * 10 * len(NFL_PROPS.split(",")) * 1  # 10 x markets x regions(us=1)
    print("=" * 96)
    print(f"  NFL OPENER PROBE — {len(GAMES)} games × {len(OFFSETS_H)} offsets = "
          f"{n_calls} calls, est ≤ {est:,} credits (cached calls = 0)")
    print("=" * 96)
    if args.dry_run:
        for eid, commence in GAMES:
            c0 = dt.datetime.fromisoformat(commence.replace("Z", "+00:00"))
            for h in OFFSETS_H:
                ts = (c0 - dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
                print(f"    would fetch {eid[:8]}  commence {commence}  −{h}h → date {ts}")
        print("  (dry-run — no credits spent)")
        return

    key = _load_key()
    if not key:
        print("  NO API KEY found (app.load_config / ODDS_API_KEY). Aborting — nothing spent.")
        return

    for eid, commence in GAMES:
        c0 = dt.datetime.fromisoformat(commence.replace("Z", "+00:00"))
        print(f"\n  {eid[:8]}  commence {commence}")
        print(f"    {'offset':>7}  {'requested date':>21}  {'returned snapshot ts':>21}  markets players")
        for h in OFFSETS_H:
            date = (c0 - dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                data, snap_ts = get_historical_event_odds(
                    key, SPORT, eid, date, regions="us", markets=NFL_PROPS,
                    bookmakers=BOOKS)
            except Exception as e:
                print(f"    −{h:>4}h  {date:>21}  ERROR {type(e).__name__}: {e}")
                continue
            nm, npl = _prop_summary(data)
            print(f"    −{h:>4}h  {date:>21}  {str(snap_ts):>21}  {nm:>7} {npl:>7}")
    print("\n" + "=" * 96)
    print("  READ: the earliest offset with markets>0 ≈ when NFL props open. If props")
    print("  appear days out, an opener backfill is viable → then price the full sweep.")


if __name__ == "__main__":
    main()
