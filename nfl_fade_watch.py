"""nfl_fade_watch.py — forward-track the (possible) NFL prop over-bias fade signal.

Context: across 2023-2025 the only recurring flicker was fading the OVER (bet UNDER at the
higher/shaded of DK/FD, esp. in the 10-20% line-disagreement band). BUT it's snapshot-
INCONSISTENT (opener flashed 2025; closing flashed 2023/24; early_4h nothing) — the
fingerprint of NOISE, not a durable edge. So this is a NEUTRAL ARBITER, not an edge we
believe: run it as 2026 games complete; if the fade cuts hold up (+real-impl AND +ROI at
fair prices), the bias is real/strengthening and becomes actionable; if they revert, it was
noise and we're done.

Uses the CLOSING snapshot — sustainable (ongoing 2026 games get closing DK+FD captured by
the normal warehouse pipeline; NO opener-capture cron needed, which Streamlit can't host).
Bets DK/FD only (executable). Reads the warehouse (scoped, one season). Diagnostic — writes
nothing, spends no API credits.
"""
import argparse
from collections import defaultdict

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way

SPORT = "americanfootball_nfl"
BOOKS = ["draftkings", "fanduel"]


def _fair_over(d):
    o, u = d.get("OVER"), d.get("UNDER")
    if not (o and u):
        return None
    try:
        return devig_two_way(american_to_implied_prob(o[1]),
                             american_to_implied_prob(u[1]))[0]
    except Exception:
        return None


def run(season, snapshot):
    # Read the LEAN mirror (props_scan loader), NOT the ladder-bloated odds_line — fast +
    # sustainable (ongoing 2026 closing/early_4h land in the mirror via the normal pipeline).
    lines = defaultdict(lambda: defaultdict(dict))     # (eid,player,prop)->book->{OVER/UNDER}
    meta = {}
    for book in BOOKS:
        props = scan._load_props([str(season)], snapshot, book)   # {(eid,player,pk):{...}}
        for (eid, player, pk), d in props.items():
            for side in ("OVER", "UNDER"):
                if side in d:
                    lines[(eid, player, pk)][book][side] = d[side]
            meta.setdefault((eid, player, pk), {"commence": d.get("commence"),
                                                "home": d.get("home"), "away": d.get("away")})
    idx = nfl_schedule.game_index([str(season)])
    actuals = scan._player_week_index([str(season)])

    # accumulate by week: [n, wins, impl, roi] for two cuts
    fade = defaultdict(lambda: [0, 0.0, 0.0, 0.0])       # all divergent, UNDER@higher
    band = defaultdict(lambda: [0, 0.0, 0.0, 0.0])       # 10-20% gap subset
    for (eid, player, pk), bybook in lines.items():
        if pk not in acc.PROPS:
            continue
        stat = acc.PROPS[pk]["stat"]
        dk, fd = bybook.get("draftkings", {}), bybook.get("fanduel", {})
        if not dk or not fd:
            continue
        m = meta[(eid, player, pk)]
        gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
        if gid is None:
            continue
        ps = gid.split("_"); week = int(ps[1])
        av = (actuals.get((scan._norm(player), ps[0], str(week))) or {}).get(stat)
        if av is None:
            continue
        dl = dk.get("OVER", dk.get("UNDER", (None,)))[0]
        fl = fd.get("OVER", fd.get("UNDER", (None,)))[0]
        if dl is None or fl is None or abs(dl - fl) < 1e-9:
            continue
        high = dk if dl > fl else fd
        pt, px = high.get("UNDER", (None, None))
        if pt is None or px is None or abs(av - pt) < 1e-9:
            continue
        won = av < pt
        frov = _fair_over(high); impl = (1 - frov) if frov is not None else 0.5
        roi = (american_to_decimal(px) - 1.0) if won else -1.0
        gap = abs(dl - fl) / ((dl + fl) / 2.0)
        for acc_d in (fade,):
            r = acc_d[week]; r[0] += 1; r[1] += won; r[2] += impl; r[3] += roi
        if 0.10 <= gap < 0.20:
            r = band[week]; r[0] += 1; r[1] += won; r[2] += impl; r[3] += roi

    def summarize(d, label):
        tot = [0, 0.0, 0.0, 0.0]
        for wk in sorted(d):
            for i in range(4):
                tot[i] += d[wk][i]
        n = tot[0]
        if n == 0:
            print(f"    {label:<28} (no completed games yet)")
            return
        ri = (tot[1] - tot[2]) / n * 100
        roi = tot[3] / n * 100
        verdict = "✅ holding" if (ri > 0 and roi > 0) else ("~ +number/-price" if ri > 0 else "❌ reverted")
        print(f"    {label:<28} n={n:>4}  win={tot[1]/n*100:4.1f}%  real-impl={ri:+5.2f}%  "
              f"ROI={roi:+6.2f}%  {verdict}")

    print("=" * 90)
    print(f"  NFL FADE WATCH — {season} @ {snapshot}  (fade the higher/shaded DK/FD line)")
    print("  Signal was snapshot-inconsistent 2023-25 (likely noise); this is the arbiter.")
    print("=" * 90)
    summarize(fade, "fade-higher-line (all)")
    summarize(band, "fade + 10-20% gap band")
    print("  READ: both +real-impl AND +ROI, sustained over weeks ⇒ real; reverts ⇒ noise.")
    print("=" * 90)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2026")
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h"])
    args = ap.parse_args()
    run(args.season, args.snapshot)


if __name__ == "__main__":
    main()
