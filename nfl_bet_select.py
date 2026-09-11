"""nfl_bet_select.py — selective betting / learn-to-abstain analysis.

Reframe (Doug): the product is NOT "bet the whole envelope for ROI" — it's a high-precision
recommender that ABSTAINS aggressively (0-5 bets/game). So the question isn't "do our
disagreements beat the market on average" (they don't) but "is there a SUBSET where we're
reliably right, identifiable up-front, that we bet while abstaining on everything else?"

Two parts:
  1. DISCRIMINATIVE — for our model-side bets, compare WON vs LOST on a-priori confidence
     features, to see what (if anything) separates right from wrong: model confidence
     (line distance in projected SDs = z_our), market disagreement (edge), data sufficiency
     (n_prior games), sharp (Pinnacle) confluence, direction, prop.
  2. SELECTION — test simple 1-2 feature rules and report real-impl + ROI PER SEASON and at
     FAIR prices. A rule counts ONLY if it is positive every season (2023/24/25), profitable
     at fair prices (not the juice band), and leaves a sensibly-sized set. Guards against the
     winner's-curse the pooled-envelope view hides.

Bets are placed at a chosen rung (default the OPENER = earliest available DK line, the
softest). DK/FD executable; Pinnacle sharp reference. Reads the local ladder extract
(nfl_ladder_clv writes it); Azure stays source of truth. Diagnostic — writes nothing.
"""
import argparse
from collections import defaultdict

import nfl_schedule
import nfl_props_accuracy as acc
import nfl_props_scan as scan
import nfl_ladder_clv as clv
from odds_client import american_to_decimal


def collect(prop, cfg, train_seasons, test_seasons, rung_pref):
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
    bets = []
    for (eid, player), bybook in lines.items():
        m = meta[(eid, player)]
        gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
        if gid is None:
            continue
        ps = gid.split("_"); season, week = ps[0], int(ps[1])
        o = feat.get((scan._norm(player), season, week))
        if o is None:
            continue
        actual = o["actual"]
        dk = bybook.get("draftkings", {})
        pinn = bybook.get("pinnacle", {})
        src = (rung_pref if rung_pref in dk
               else next((s for s, _h in clv.RUNGS if s in dk), None))
        if src is None:
            continue
        d = dk[src]
        line = d.get("OVER", d.get("UNDER", (None, None)))[0]
        if line is None or abs(actual - line) < 1e-9:
            continue
        fair = clv._fair_over(d)
        if fair is None:
            continue
        p = acc.p_over(o, line, cfg, comp)
        over_side = p >= fair
        mean, sd = acc.proj_mean_sd(o, cfg, comp)
        z_our = ((mean - line) if over_side else (line - mean)) / (sd if sd else 1.0)
        won = (actual > line) if over_side else (actual < line)
        px = (d.get("OVER") if over_side else d.get("UNDER"))[1]
        pf = clv._fair_over(pinn.get(src)) if pinn.get(src) else None
        sharp = (pf > fair) == over_side if pf is not None else None
        bets.append({"season": season, "prop": prop, "over": over_side,
                     "edge": abs(p - fair), "z": z_our, "conf": abs(p - 0.5),
                     "n_prior": o.get("n_prior", 0), "cv": o.get("cv", 0.0),
                     "sharp": sharp,
                     "won": 1 if won else 0, "implied": fair if over_side else 1 - fair,
                     "roi": (american_to_decimal(px) - 1.0) if won else -1.0,
                     "price": px})
    return bets


def _stat(bets):
    n = len(bets)
    if not n:
        return None
    return {"n": n, "win": sum(b["won"] for b in bets) / n * 100,
            "ri": (sum(b["won"] for b in bets) - sum(b["implied"] for b in bets)) / n * 100,
            "roi": sum(b["roi"] for b in bets) / n * 100}


def discriminative(bets):
    won = [b for b in bets if b["won"]]
    lost = [b for b in bets if not b["won"]]
    print(f"\n  DISCRIMINATIVE — won ({len(won)}) vs lost ({len(lost)}) feature means:")
    print(f"    {'feature':<16} {'won':>10} {'lost':>10}  {'gap':>8}")
    for f, lbl in [("z", "z_our (SDs)"), ("edge", "edge"), ("conf", "conf |p-.5|"),
                   ("n_prior", "n_prior"), ("cv", "cv (boom/bust)")]:
        wv = sum(b[f] for b in won) / len(won) if won else 0
        lv = sum(b[f] for b in lost) / len(lost) if lost else 0
        print(f"    {lbl:<16} {wv:>10.3f} {lv:>10.3f}  {wv-lv:>+8.3f}")
    for lbl, sub in [("over", [b for b in bets if b["over"]]),
                     ("under", [b for b in bets if not b["over"]]),
                     ("sharp-confirms", [b for b in bets if b["sharp"] is True])]:
        s = _stat(sub)
        if s:
            print(f"    win% {lbl:<14} {s['win']:.1f}%  (n={s['n']})")


def _fair(bets):
    return [b for b in bets if -110 <= b["price"] < 100]   # fair-price band only


def rule_report(bets, name, pred):
    sub = [b for b in bets if pred(b)]
    if len(sub) < 40:
        print(f"    {name:<34} n={len(sub):>5}  (thin)")
        return
    line = f"    {name:<34} n={len(sub):>5}  "
    ok = True
    for s in ("2023", "2024", "2025"):
        st = _stat([b for b in sub if b["season"] == s])
        if not st or st["n"] < 15:
            line += f"{s}:na  "; ok = False
        else:
            line += f"{s}:{st['ri']:+.1f}%/{st['roi']:+.1f}%  "
            if st["ri"] <= 0:
                ok = False
    fair = _stat(_fair(sub))
    line += f"| FAIR roi={fair['roi']:+.1f}%(n={fair['n']})" if fair else "| FAIR:na"
    line += "  <<SURVIVES" if (ok and fair and fair["roi"] > 0) else ""
    print(line)


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
    ap.add_argument("--rung", default="opener",
                    help="'opener' (earliest DK line) or a source tag e.g. early_4h")
    ap.add_argument("--prop", default="all")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    props = list(acc.PROPS) if args.prop == "all" else [args.prop]
    rung_pref = None if args.rung == "opener" else args.rung

    allb = []
    for prop in props:
        allb.extend(collect(prop, acc.PROPS[prop], train_seasons, test_seasons, rung_pref))
    print("=" * 100)
    print(f"  BET SELECTION — {len(allb)} bets, rung={args.rung}, test {test_seasons}")
    print("  per-season shows real-impl/ROI; SURVIVES = +real-impl every season AND +ROI at fair prices")
    print("=" * 100)
    discriminative(allb)
    print("\n  SELECTION RULES (survive-every-season-at-fair-prices):")
    rule_report(allb, "ALL (baseline)", lambda b: True)
    for zt in (0.5, 1.0, 1.5):
        rule_report(allb, f"z_our >= {zt}", lambda b, z=zt: b["z"] >= z)
    for et in (0.05, 0.10):
        rule_report(allb, f"edge >= {et}", lambda b, e=et: b["edge"] >= e)
    for npr in (6, 8, 10):
        rule_report(allb, f"n_prior >= {npr}", lambda b, n=npr: b["n_prior"] >= n)
    for cvt in (0.35, 0.50):
        rule_report(allb, f"cv <= {cvt} (steady players)", lambda b, c=cvt: 0 < b["cv"] <= c)
    rule_report(allb, "sharp confirms", lambda b: b["sharp"] is True)
    rule_report(allb, "z>=1.0 & sharp confirms",
                lambda b: b["z"] >= 1.0 and b["sharp"] is True)
    rule_report(allb, "z>=1.0 & n_prior>=8",
                lambda b: b["z"] >= 1.0 and b["n_prior"] >= 8)
    rule_report(allb, "z>=1.0 & edge in [.05,.12]",
                lambda b: b["z"] >= 1.0 and 0.05 <= b["edge"] < 0.12)
    print("=" * 100)


if __name__ == "__main__":
    main()
