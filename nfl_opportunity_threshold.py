"""nfl_opportunity_threshold.py — find the OPTIMAL abstain threshold per prop.

For the defensive fair-value tool we want to SPEAK only where our model is trustworthy and
ABSTAIN below that. "Trustworthy" = our Brier is within TOL of the (efficient) market Brier.
Optimal threshold = the LOWEST expected-opportunity T (max coverage) such that, for bets with
opp >= T, gap = ourBrier - mktBrier <= TOL.

Chosen on DISCOVERY (2023-2024), then VALIDATED on held-out 2025 (does the gap stay within
tolerance at that T, and what coverage remains?) — so the threshold isn't cherry-picked to the
test data. If no T gets the gap under TOL, the prop is never trustworthy -> abstain entirely.
Reads the local ladder extract (closing DK) + actuals. Diagnostic — no spend.
"""
import argparse
from collections import defaultdict

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
import nfl_ladder_clv as clv
from odds_client import american_to_implied_prob, devig_two_way

OPP_PROPS = ["player_receptions", "player_reception_yds", "player_rush_yds",
             "player_pass_yds", "player_rush_attempts", "player_pass_attempts",
             "player_pass_completions"]
UNIT = {"player_receptions": "tgt", "player_reception_yds": "rec", "player_rush_yds": "car",
        "player_pass_yds": "att", "player_rush_attempts": "car", "player_pass_attempts": "att",
        "player_pass_completions": "att"}


def _opp(o):
    return o.get("exp_vol", o.get("mean_base", 0.0))


def _fair_over(d):
    ov, un = d.get("OVER"), d.get("UNDER")
    if not (ov and un):
        return None
    try:
        return devig_two_way(american_to_implied_prob(ov[1]),
                             american_to_implied_prob(un[1]))[0]
    except Exception:
        return None


def collect(prop, cfg, train_seasons, test_seasons):
    hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
    acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
    train = acc._obs_from_series(acc._series(train_seasons), cfg)
    if len(train) < 200:
        return []
    comp = acc.fit_components(train, cfg)
    feat = {(o["name"], o["season"], o["week"]): o
            for o in acc._obs_from_series(acc._series(test_seasons), cfg)}
    idx = nfl_schedule.game_index([str(s) for s in test_seasons])
    lines, meta = clv._load_local(prop, test_seasons)
    recs = []
    for (eid, player), bybook in lines.items():
        d = bybook.get("draftkings", {}).get("closing")
        if not d:
            continue
        m = meta[(eid, player)]
        gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
        if gid is None:
            continue
        ps = gid.split("_"); week = int(ps[1])
        o = feat.get((scan._norm(player), ps[0], week))
        if o is None:
            continue
        line = d.get("OVER", d.get("UNDER", (None,)))[0]
        if line is None or abs(o["actual"] - line) < 1e-9:
            continue
        fair = _fair_over(d)
        if fair is None:
            continue
        y = 1 if o["actual"] > line else 0
        p = acc.p_over(o, line, cfg, comp)
        recs.append({"season": ps[0], "opp": _opp(o),
                     "oe": (p - y) ** 2, "me": (fair - y) ** 2})
    return recs


def _gap_cov(recs, T):
    sub = [r for r in recs if r["opp"] >= T]
    n = len(sub)
    if n == 0:
        return None, 0, 0.0
    gap = sum(r["oe"] - r["me"] for r in sub) / n
    return gap, n, n / len(recs)


def run(prop, cfg, train_seasons, test_seasons, tol):
    recs = collect(prop, cfg, train_seasons, test_seasons)
    if len(recs) < 400:
        print(f"\n  {prop.replace('player_','')}: thin ({len(recs)}) — skip")
        return
    disc = [r for r in recs if r["season"] in ("2023", "2024")]
    val = [r for r in recs if r["season"] == "2025"]
    opps = sorted(r["opp"] for r in disc)
    grid = [opps[int(q * (len(opps) - 1))] for q in [i / 20 for i in range(0, 20)]]
    grid = sorted(set(round(g, 1) for g in grid))
    # optimal = lowest T on discovery with gap <= tol
    best = None
    for T in grid:
        gap, n, cov = _gap_cov(disc, T)
        if gap is not None and n >= 150 and gap <= tol:
            best = T
            break
    print(f"\n  {prop.replace('player_','')}  ({UNIT[prop]})")
    print(f"    {'T':>6} {'disc cov':>9} {'disc gap':>9}   {'val cov':>8} {'val gap':>9}")
    for T in grid[::2]:
        dg, dn, dc = _gap_cov(disc, T)
        vg, vn, vc = _gap_cov(val, T)
        mark = " <= OPTIMAL" if best is not None and abs(T - best) < 1e-9 else ""
        if dg is None:
            continue
        print(f"    {T:>6.1f} {dc*100:>7.0f}% {dg:>+9.4f}   "
              f"{(vc*100 if vg is not None else 0):>6.0f}% {(vg if vg is not None else float('nan')):>+9.4f}{mark}")
    if best is None:
        print(f"    => NO threshold gets gap <= {tol:.3f} -> model never trustworthy here; ABSTAIN entirely.")
    else:
        vg, vn, vc = _gap_cov(val, best)
        ok = vg is not None and vg <= tol * 1.5
        print(f"    => OPTIMAL T = {best:.1f} {UNIT[prop]}  (discovery gap<= {tol:.3f}); "
              f"HELD-OUT 2025: gap={vg:+.4f}, coverage={vc*100:.0f}%  "
              f"{'CONFIRMED' if ok else 'FAILS validation (gap re-widens)'}")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022")
    ap.add_argument("--test", default="2023,2024,2025")
    ap.add_argument("--tol", type=float, default=0.010,
                    help="max acceptable ourBrier-mktBrier gap (trustworthy tolerance)")
    ap.add_argument("--prop", default="all")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    props = OPP_PROPS if args.prop == "all" else [args.prop]
    print("=" * 92)
    print(f"  NFL OPPORTUNITY THRESHOLD — optimal abstain cutoff (tol={args.tol}), {test_seasons}")
    print("  optimal = lowest opp T where discovery(2023-24) gap<=tol; validated on 2025.")
    print("=" * 92)
    for prop in props:
        run(prop, acc.PROPS[prop], train_seasons, test_seasons, args.tol)
    print("=" * 92)


if __name__ == "__main__":
    main()
