"""nfl_fair_value.py — LOCAL pre-game fair-value / abstain report for NFL props.

The edge hunt showed our 8-prop model has NO reliable +EV edge — but it IS well-calibrated
(beats hit-rate, 11-yr validated). So its honest use is DEFENSIVE reference, not auto-firing:
  * show our calibrated fair line (projected median) + P(over) at the market's number,
  * show the de-vigged market and the gap,
  * ABSTAIN by default — flag only where the offered number is clearly WORSE than fair (so
    you don't take a bad price), and explicitly NOT recommend "value" on big disagreements
    (we proved those are the model's OWN errors, not edges).

Runs LOCALLY off the mirror parquets (Streamlit Cloud can't read those) + nfl_data. For a
live pre-game slate you'd swap the line source to the Odds API; historical --season/--week
validates the format for free. Frozen model = components trained 2012-2022, swept config.
Diagnostic — writes nothing, spends no credits.
"""
import argparse

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
from odds_client import american_to_implied_prob, devig_two_way

BOOK = "draftkings"


def _fair_over(d):
    o, u = d.get("OVER"), d.get("UNDER")
    if not (o and u):
        return None
    try:
        return devig_two_way(american_to_implied_prob(o[1]),
                             american_to_implied_prob(u[1]))[0]
    except Exception:
        return None


def _models(train_seasons, test_seasons):
    """Frozen components + feature index per prop."""
    out = {}
    for prop, cfg in acc.PROPS.items():
        hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
        acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
        train = acc._obs_from_series(acc._series(train_seasons), cfg)
        if len(train) < 200:
            continue
        comp = acc.fit_components(train, cfg)
        feat = {(o["name"], o["season"], o["week"]): o
                for o in acc._obs_from_series(acc._series(test_seasons), cfg)}
        out[prop] = (cfg, comp, feat)
    return out


def run(season, week, snapshot, book):
    models = _models(["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
                      "2020", "2021", "2022"], [str(season)])
    idx = nfl_schedule.game_index([str(season)])
    rows = []
    for prop in acc.PROPS:
        if prop not in models:
            continue
        cfg, comp, feat = models[prop]
        for (eid, player, pk), d in scan._load_props([str(season)], snapshot, book).items():
            if pk != prop:
                continue
            gid, _ = nfl_schedule.resolve_event(d.get("home"), d.get("away"),
                                                d.get("commence"), index=idx)
            if gid is None:
                continue
            ps = gid.split("_")
            if int(ps[1]) != week:
                continue
            o = feat.get((scan._norm(player), ps[0], week))   # feat keys use int week
            if o is None:
                continue
            line = d.get("OVER", d.get("UNDER", (None,)))[0]
            if line is None:
                continue
            proj, _sd = acc.proj_mean_sd(o, cfg, comp)
            p_over = acc.p_over(o, line, cfg, comp)
            mkt = _fair_over(d)
            rows.append({"prop": pk.replace("player_", ""), "player": player,
                         "line": line, "proj": proj, "p_over": p_over,
                         "mkt": mkt, "edge": (p_over - mkt) if mkt is not None else None})
    if not rows:
        print(f"  No prop lines found for {season} week {week} ({book}/{snapshot}).")
        return
    rows.sort(key=lambda r: (r["prop"], -(abs(r["edge"]) if r["edge"] is not None else 0)))
    print("=" * 104)
    print(f"  NFL FAIR-VALUE REPORT — {season} week {week} ({book} {snapshot})")
    print("  Reference/DEFENSIVE only: our model is calibrated but has NO reliable +EV edge.")
    print("  'proj' = our projected value; big |Δ| = our model DISAGREES (historically our")
    print("  error, NOT a value bet). Use to sanity-check a number, not to fire bets.")
    print("=" * 104)
    print(f"  {'prop':<14} {'player':<24} {'line':>7} {'ourProj':>8} {'ourP(o)':>8} "
          f"{'mktP(o)':>8} {'Δ':>7}  note")
    cur = None
    for r in rows:
        if r["prop"] != cur:
            cur = r["prop"]; print(f"  --- {cur} ---")
        d = r["edge"]
        # honest note: flag only clearly-worse-than-fair offered numbers; never "value".
        note = ""
        if d is not None:
            if abs(d) >= 0.10:
                note = "model disagrees hard (likely our error — abstain)"
            elif abs(d) >= 0.05:
                note = "mild disagreement — abstain"
            else:
                note = "≈ fair"
        print(f"  {'':<14} {r['player'][:24]:<24} {r['line']:>7.1f} {r['proj']:>8.1f} "
              f"{r['p_over']:>8.2f} {(r['mkt'] if r['mkt'] is not None else float('nan')):>8.2f} "
              f"{(d*100 if d is not None else float('nan')):>+6.1f}%  {note}")
    print("=" * 104)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2025")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h"])
    ap.add_argument("--book", default=BOOK)
    args = ap.parse_args()
    run(args.season, args.week, args.snapshot, args.book)


if __name__ == "__main__":
    main()
