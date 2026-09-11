"""nfl_book_shop.py — DK vs FD (the two books Doug actually bets): do they disagree, does
one lag, and is shopping the better LINE/price exploitable?

Doug bets DraftKings AND FanDuel, so the actionable question isn't "are we off the sharp
consensus" — it's "do DK and FD disagree with EACH OTHER (line, not just odds), can I take
the better side at whichever book, and does one book lag the other (a stale number to grab)?"
The 6h ladder can't see sub-hour transient moves, but a DK≠FD gap at one snapshot is the
closest proxy for book-level staleness we can measure.

Three parts (DK vs FD; Pinnacle secondary — MLB showed it's no sharper than DK):
  A. DIVERGENCE (model-free): how often do DK and FD post different LINES, by how much;
     and when the line matches, how much do the PRICES differ (pure odds-shop value).
  B. LEAD/LAG: when they differ at rung t, does the lagging book move toward the other by
     the next (later) rung — and which book leads? (who to trust / who's stale.)
  C. EXPLOIT: on our model's side, does taking the BEST-of-both (best line, then best price)
     beat betting a single book — real-impl / ROI per season, at fair prices.

Reads the local ladder extract (nfl_ladder_clv --extract). Diagnostic — writes nothing.
"""
import argparse
from collections import defaultdict

import nfl_schedule
import nfl_props_accuracy as acc
import nfl_props_scan as scan
import nfl_ladder_clv as clv
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way

RUNG_ORDER = [s for s, _h in clv.RUNGS]           # earliest → latest


def _line_px(d, side):
    v = d.get(side)
    return (v[0], v[1]) if v else (None, None)


def divergence_and_lag(props, seasons):
    """A + B: model-free line/price divergence and lead-lag, pooled over all props."""
    same = diff = 0
    absdiff = []
    price_gap = []                                 # prob-terms price gap when lines match
    lag = {"dk_leads": 0, "fd_leads": 0, "both_move": 0, "n": 0}
    for prop in props:
        lines, meta = clv._load_local(prop, seasons)
        for (eid, player), bybook in lines.items():
            dk = bybook.get("draftkings", {})
            fd = bybook.get("fanduel", {})
            # divergence per rung
            for src in RUNG_ORDER:
                if src not in dk or src not in fd:
                    continue
                dl = _line_px(dk[src], "OVER")[0] or _line_px(dk[src], "UNDER")[0]
                fl = _line_px(fd[src], "OVER")[0] or _line_px(fd[src], "UNDER")[0]
                if dl is None or fl is None:
                    continue
                if abs(dl - fl) < 1e-9:
                    same += 1
                    do = _line_px(dk[src], "OVER")[1]; fo = _line_px(fd[src], "OVER")[1]
                    if do is not None and fo is not None:
                        try:
                            price_gap.append(abs(american_to_implied_prob(do)
                                                 - american_to_implied_prob(fo)))
                        except Exception:
                            pass
                else:
                    diff += 1
                    absdiff.append(abs(dl - fl))
            # lead-lag: consecutive rungs where a gap CLOSES
            present = [s for s in RUNG_ORDER if s in dk and s in fd]
            for a, b in zip(present, present[1:]):     # a earlier, b later
                da = _line_px(dk[a], "OVER")[0] or _line_px(dk[a], "UNDER")[0]
                fa = _line_px(fd[a], "OVER")[0] or _line_px(fd[a], "UNDER")[0]
                db = _line_px(dk[b], "OVER")[0] or _line_px(dk[b], "UNDER")[0]
                fb = _line_px(fd[b], "OVER")[0] or _line_px(fd[b], "UNDER")[0]
                if None in (da, fa, db, fb) or abs(da - fa) < 1e-9:
                    continue
                lag["n"] += 1
                dk_moved = abs(db - da) > 1e-9
                fd_moved = abs(fb - fa) > 1e-9
                # did the later gap shrink, and who moved toward whom?
                if abs(db - fb) < abs(da - fa):        # converged
                    if fd_moved and not dk_moved:
                        lag["dk_leads"] += 1           # FD moved to DK → DK led
                    elif dk_moved and not fd_moved:
                        lag["fd_leads"] += 1
                    else:
                        lag["both_move"] += 1
    tot = same + diff
    print(f"\n  A. DK vs FD LINE divergence (n={tot:,} matched snapshots):")
    if tot:
        print(f"     same line: {same/tot*100:.1f}%   different line: {diff/tot*100:.1f}%")
        if absdiff:
            absdiff.sort()
            ge = lambda t: sum(1 for x in absdiff if x >= t) / tot * 100
            print(f"     when different: mean |Δline|={sum(absdiff)/len(absdiff):.2f}, "
                  f"median={absdiff[len(absdiff)//2]:.2f}; "
                  f"|Δ|≥0.5 in {ge(0.5):.1f}% of all, ≥1.0 in {ge(1.0):.1f}%")
        if price_gap:
            print(f"     when SAME line: mean price gap={sum(price_gap)/len(price_gap)*100:.2f}%"
                  f" implied (pure odds-shop value)")
    if lag["n"]:
        n = lag["n"]
        print(f"\n  B. LEAD/LAG (n={n:,} divergent pairs that then converged-or-not):")
        print(f"     DK leads (FD catches up): {lag['dk_leads']/n*100:.1f}%   "
              f"FD leads: {lag['fd_leads']/n*100:.1f}%   both move: {lag['both_move']/n*100:.1f}%")


def _best_side(bybook_src, over):
    """Best executable (line, price) for a side across DK+FD at one rung."""
    cands = []
    for book in ("draftkings", "fanduel"):
        d = bybook_src.get(book)
        if not d:
            continue
        pt, px = _line_px(d, "OVER" if over else "UNDER")
        if pt is not None and px is not None:
            cands.append((pt, px, book))
    if not cands:
        return None
    # over: lowest line then best (max) price; under: highest line then max price
    cands.sort(key=lambda c: (c[0] if over else -c[0], -c[1]))
    return cands[0]


def exploit(props, train_seasons, test_seasons, rung_pref):
    """C: our-side best-of-both vs single book, per season."""
    def blank():
        return {"dk": [], "fd": [], "best": []}
    acc_by = defaultdict(blank)                     # season -> lists of (won, implied, roi)
    for prop in props:
        cfg = acc.PROPS[prop]
        hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
        acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
        train = acc._obs_from_series(acc._series(train_seasons), cfg)
        if len(train) < 200:
            continue
        comp = acc.fit_components(train, cfg)
        feat = {(o["name"], o["season"], o["week"]): o
                for o in acc._obs_from_series(acc._series(test_seasons), cfg)}
        idx = nfl_schedule.game_index([str(s) for s in test_seasons])
        lines, meta = clv._load_local(prop, test_seasons)
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
            src = next((s for s in RUNG_ORDER if s in bybook.get("draftkings", {})
                        or s in bybook.get("fanduel", {})), None) if rung_pref == "opener" \
                else rung_pref
            if src is None:
                continue
            at = {b: bybook[b][src] for b in ("draftkings", "fanduel")
                  if b in bybook and src in bybook[b]}
            if not at:
                continue
            # model side off DK if present else FD (devig that book)
            ref = at.get("draftkings") or at.get("fanduel")
            fair = clv._fair_over(ref)
            if fair is None:
                continue
            line0 = _line_px(ref, "OVER")[0] or _line_px(ref, "UNDER")[0]
            p = acc.p_over(o, line0, cfg, comp)
            over = p >= fair
            for tag, sel in (("dk", at.get("draftkings")), ("fd", at.get("fanduel")),
                             ("best", None)):
                if tag == "best":
                    b = _best_side({k2: at[k2] for k2 in at}, over)
                    if not b:
                        continue
                    pt, px, _bk = b
                else:
                    if not sel:
                        continue
                    pt, px = _line_px(sel, "OVER" if over else "UNDER")
                    if pt is None:
                        continue
                if abs(actual - pt) < 1e-9:
                    continue
                won = (actual > pt) if over else (actual < pt)
                # implied at that book's own devig for real-impl reference
                fr = clv._fair_over(sel) if tag != "best" else fair
                impl = (fr if over else 1 - fr) if fr is not None else fair
                acc_by[season][tag].append((1 if won else 0,
                                            impl, (american_to_decimal(px) - 1.0) if won else -1.0))
    print(f"\n  C. EXPLOIT — our-side, best-of-both vs single book (rung={rung_pref}):")
    print(f"     {'season':<8} {'DK roi':>9} {'FD roi':>9} {'BEST roi':>9}  {'BEST real-impl':>15} {'n':>6}")
    for s in sorted(acc_by):
        row = acc_by[s]
        def roi(lst):
            return sum(r[2] for r in lst) / len(lst) * 100 if lst else float("nan")
        def ri(lst):
            return (sum(r[0] for r in lst) - sum(r[1] for r in lst)) / len(lst) * 100 if lst else float("nan")
        print(f"     {s:<8} {roi(row['dk']):>+8.2f}% {roi(row['fd']):>+8.2f}% "
              f"{roi(row['best']):>+8.2f}%  {ri(row['best']):>+14.2f}% {len(row['best']):>6}")


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
    ap.add_argument("--rung", default="opener")
    ap.add_argument("--prop", default="all")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    props = list(acc.PROPS) if args.prop == "all" else [args.prop]
    print("=" * 100)
    print(f"  DK vs FD BOOK SHOP — test {test_seasons}, rung={args.rung}")
    print("=" * 100)
    divergence_and_lag(props, test_seasons)
    exploit(props, train_seasons, test_seasons, args.rung)
    print("=" * 100)


if __name__ == "__main__":
    main()
