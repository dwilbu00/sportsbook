"""nfl_opportunity_filter.py — does filtering to HIGH-opportunity players help? (Doug's idea)

We built opportunity projections (expected targets/carries/attempts). Hypothesis: low-
opportunity players are where the model goes haywire (role-stale) AND where outcomes are
noisiest, so we should ABSTAIN below some opportunity threshold. This buckets every prop bet
(at the closing standard line) by EXPECTED OPPORTUNITY and reports, per bucket per season:
  * accuracy: our Brier vs the market Brier (does the gap to the market shrink for high-opp?)
  * edge: our-side real-impl and ROI (does any edge concentrate in high-opp?)

If high-opp tightens toward the market AND/OR shows +real-impl every season, the opportunity
filter is the key (both for a cleaner defensive tool and for any bettable subset). Reads the
local ladder extract (closing DK) + actuals. Diagnostic — no spend.
"""
import argparse
from collections import defaultdict

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
import nfl_ladder_clv as clv
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way

# props with a meaningful opportunity projection (volume-driven)
OPP_PROPS = ["player_receptions", "player_reception_yds", "player_rush_yds",
             "player_pass_yds", "player_rush_attempts", "player_pass_attempts",
             "player_pass_completions"]
OPP_LABEL = {"player_receptions": "exp targets", "player_reception_yds": "exp rec",
             "player_rush_yds": "exp carries", "player_pass_yds": "exp att",
             "player_rush_attempts": "exp carries", "player_pass_attempts": "exp att",
             "player_pass_completions": "exp att"}


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


def run(prop, cfg, train_seasons, test_seasons):
    hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
    acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
    train = acc._obs_from_series(acc._series(train_seasons), cfg)
    if len(train) < 200:
        return
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
        p = acc.p_over(o, line, cfg, comp)
        y = 1 if o["actual"] > line else 0
        over = p >= fair
        won = (y == 1) if over else (y == 0)
        px = (d.get("OVER") if over else d.get("UNDER"))[1]
        recs.append({"season": ps[0], "opp": _opp(o), "y": y, "p": p, "fair": fair,
                     "over": over, "won": won,
                     "roi": (american_to_decimal(px) - 1.0) if won else -1.0})
    if len(recs) < 300:
        return
    # opportunity tertiles
    opps = sorted(r["opp"] for r in recs)
    q1, q2 = opps[len(opps) // 3], opps[2 * len(opps) // 3]
    def bucket(v):
        return "LOW" if v < q1 else ("MID" if v < q2 else "HIGH")
    print(f"\n  {prop.replace('player_','')}  ({OPP_LABEL[prop]}; tertiles <{q1:.1f} / <{q2:.1f} / +)")
    for bl in ("LOW", "MID", "HIGH"):
        sub = [r for r in recs if bucket(r["opp"]) == bl]
        if len(sub) < 60:
            continue
        n = len(sub)
        our_b = sum((r["p"] - r["y"]) ** 2 for r in sub) / n
        mkt_b = sum((r["fair"] - r["y"]) ** 2 for r in sub) / n
        ri = (sum(r["won"] for r in sub) / n
              - sum((r["fair"] if r["over"] else 1 - r["fair"]) for r in sub) / n) * 100
        roi = sum(r["roi"] for r in sub) / n * 100
        # per-season real-impl consistency
        cells = []
        for s in ("2023", "2024", "2025"):
            ss = [r for r in sub if r["season"] == s]
            if len(ss) >= 30:
                ris = (sum(r["won"] for r in ss) / len(ss)
                       - sum((r["fair"] if r["over"] else 1 - r["fair"]) for r in ss) / len(ss)) * 100
                cells.append(f"{s}:{ris:+.1f}%")
        print(f"    {bl:<5} n={n:>5}  ourBrier={our_b:.4f} mktBrier={mkt_b:.4f} "
              f"gap={our_b-mkt_b:+.4f}  real-impl={ri:+.2f}%  ROI={roi:+.1f}%  [{' '.join(cells)}]")


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
    ap.add_argument("--prop", default="all")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    props = OPP_PROPS if args.prop == "all" else [args.prop]
    print("=" * 100)
    print(f"  NFL OPPORTUNITY FILTER — accuracy + edge by expected-opportunity tertile, {test_seasons}")
    print("  Q: does HIGH-opportunity tighten the gap to market AND/OR show +real-impl every season?")
    print("=" * 100)
    for prop in props:
        run(prop, acc.PROPS[prop], train_seasons, test_seasons)
    print("=" * 100)


if __name__ == "__main__":
    main()
