"""odds_links_probe.py — confirm The Odds API returns DK/FD betslip links + source ids.

The V4 API supports includeLinks=true (bookmaker links to events/markets/BETSLIPS "if
available") and includeSids=true (source ids to construct our own links). These are FREE
enrichment flags -- cost is still markets x regions -- so this probe spends only the base
request credits and shows EXACTLY which link/sid fields DK/FanDuel populate, per level.

>>> COST: h2h only = ~1 credit (1 market x 1 region). With --props: +~1 credit (one event's
    player_pass_yds). Run it yourself so you control the spend.

Usage (your machine, ODDS_API_KEY in env or .streamlit/secrets.toml):
  python odds_links_probe.py
  python odds_links_probe.py --props
  python odds_links_probe.py --api-key <KEY> --sport americanfootball_nfl
"""
import argparse
import os
import re

import requests

BASE = "https://api.the-odds-api.com/v4"
BOOKS = ("draftkings", "fanduel")


def _load_key(cli_key):
    if cli_key:
        return cli_key
    for env in ("ODDS_API_KEY", "THE_ODDS_API_KEY"):
        if os.environ.get(env):
            return os.environ[env]
    # fall back to .streamlit/secrets.toml
    path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(path):
        try:
            import tomllib
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if data.get("ODDS_API_KEY"):
                return data["ODDS_API_KEY"]
        except Exception:
            txt = open(path, encoding="utf-8").read()
            m = re.search(r'ODDS_API_KEY\s*=\s*"([^"]+)"', txt)
            if m:
                return m.group(1)
    return None


def _get(url, params):
    r = requests.get(url, params=params, timeout=30)
    used = r.headers.get("x-requests-used")
    rem = r.headers.get("x-requests-remaining")
    print(f"  HTTP {r.status_code}  credits: used={used} remaining={rem}")
    r.raise_for_status()
    return r.json()


def _dump_links(events, limit_events=3):
    found = {"bookmaker": 0, "market": 0, "outcome": 0}
    for ev in events[:limit_events]:
        print(f"\nEVENT {ev.get('home_team')} vs {ev.get('away_team')}  "
              f"(id={ev.get('id')})  link={ev.get('link')} sid={ev.get('sid')}")
        for bk in ev.get("bookmakers", []):
            if bk.get("key") not in BOOKS:
                continue
            if bk.get("link") or bk.get("sid"):
                found["bookmaker"] += 1
            print(f"  [{bk.get('key')}] link={bk.get('link')}  sid={bk.get('sid')}")
            for mk in bk.get("markets", []):
                if mk.get("link") or mk.get("sid"):
                    found["market"] += 1
                print(f"    market={mk.get('key')}  link={mk.get('link')}  sid={mk.get('sid')}")
                for oc in mk.get("outcomes", []):
                    if oc.get("link") or oc.get("sid"):
                        found["outcome"] += 1
                    print(f"      outcome={oc.get('name')} {oc.get('description') or ''} "
                          f"pt={oc.get('point')} price={oc.get('price')}\n"
                          f"        link={oc.get('link')}\n        sid={oc.get('sid')}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--sport", default="americanfootball_nfl")
    ap.add_argument("--regions", default="us")
    ap.add_argument("--props", action="store_true",
                    help="also probe one event's player_pass_yds (+~1 credit)")
    a = ap.parse_args()

    key = _load_key(a.api_key)
    if not key:
        print("No API key. Pass --api-key, set ODDS_API_KEY, or add it to "
              ".streamlit/secrets.toml.")
        return

    common = {"apiKey": key, "regions": a.regions, "oddsFormat": "american",
              "bookmakers": ",".join(BOOKS),
              "includeLinks": "true", "includeSids": "true"}

    print(f"=== TEAM (h2h) — {a.sport} ===")
    events = _get(f"{BASE}/sports/{a.sport}/odds", {**common, "markets": "h2h"})
    print(f"events returned: {len(events)}")
    team_found = _dump_links(events)
    print(f"\nTEAM link/sid presence -> {team_found}")

    if a.props and events:
        eid = events[0]["id"]
        print(f"\n=== PROPS (player_pass_yds) — event {eid} ===")
        try:
            ev = _get(f"{BASE}/sports/{a.sport}/events/{eid}/odds",
                      {**common, "markets": "player_pass_yds"})
            prop_found = _dump_links([ev])
            print(f"\nPROPS link/sid presence -> {prop_found}")
        except Exception as exc:
            print(f"  props probe failed: {type(exc).__name__}: {str(exc)[:160]}")

    print("\nWhat to look for: an OUTCOME-level `link` on draftkings/fanduel is a ready-made "
          "betslip link (tap-test one on your phone). `sid` present lets us build our own. "
          "Report the outcome link/sid presence for TEAM and (if run) PROPS.")


if __name__ == "__main__":
    main()
