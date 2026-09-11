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


def _leg_pnl(win, push, price):
    if push:
        return 0.0
    return (american_to_decimal(price) - 1.0) if win else -1.0


def middle(props, seasons, rung_pref):
    """Bet OVER@low-line + UNDER@high-line when DK/FD diverge. NOT a hedge: net -vig when the
    result lands outside the gap, big win inside. Measures combined ROI, gap-hit rate, and
    flags 'free/dead' middles (both legs plus-money => a miss breaks even)."""
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    actuals = scan._player_week_index(seasons)
    print(f"\n  MIDDLE test (over@low + under@high on DK/FD divergence, rung={rung_pref}):")
    print(f"    {'prop':<22} {'n':>6} {'gap-hit':>8} {'ROI/pair':>9} {'free-mid':>8} {'freeROI':>8}")
    tot_n = tot_pnl = 0.0
    for prop in props:
        stat = acc.PROPS[prop]["stat"]
        lines, meta = clv._load_local(prop, seasons)
        n = hits = free = 0
        pnl = free_pnl = 0.0
        for (eid, player), bybook in lines.items():
            dk = bybook.get("draftkings", {})
            fd = bybook.get("fanduel", {})
            src = (next((s for s in RUNG_ORDER if s in dk or s in fd), None)
                   if rung_pref == "opener" else rung_pref)
            if src is None or src not in dk or src not in fd:
                continue
            m = meta[(eid, player)]
            gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
            if gid is None:
                continue
            ps = gid.split("_")
            av = (actuals.get((scan._norm(player), ps[0], str(int(ps[1])))) or {}).get(stat)
            if av is None:
                continue
            dl = _line_px(dk[src], "OVER")[0] or _line_px(dk[src], "UNDER")[0]
            fl = _line_px(fd[src], "OVER")[0] or _line_px(fd[src], "UNDER")[0]
            if dl is None or fl is None or abs(dl - fl) < 1e-9:
                continue
            low, high = (dk[src], fd[src]) if dl < fl else (fd[src], dk[src])
            lo_line, lo_over = _line_px(low, "OVER")
            hi_line, hi_under = _line_px(high, "UNDER")
            if lo_over is None or hi_under is None:
                continue
            over_win = av > lo_line; over_push = abs(av - lo_line) < 1e-9
            under_win = av < hi_line; under_push = abs(av - hi_line) < 1e-9
            pair = _leg_pnl(over_win, over_push, lo_over) + _leg_pnl(under_win, under_push, hi_under)
            n += 1
            pnl += pair
            if over_win and under_win:
                hits += 1
            if lo_over > 0 and hi_under > 0:            # both plus-money = free/dead middle
                free += 1
                free_pnl += pair
        if n >= 30:
            print(f"    {prop.replace('player_',''):<22} {n:>6} {hits/n*100:>7.1f}% "
                  f"{pnl/n/2*100:>+8.2f}% {free:>8} "
                  f"{(free_pnl/free/2*100 if free else float('nan')):>+7.2f}%")
            tot_n += n; tot_pnl += pnl
    if tot_n:
        print(f"    {'POOLED':<22} {int(tot_n):>6} {'':>8} {tot_pnl/tot_n/2*100:>+8.2f}%")


def fade_laggard(props, seasons, rung_pref):
    """When DK≠FD, treat one book as leader and bet the OTHER (laggard)'s stale side:
    laggard_line < leader_line ⇒ laggard's OVER is too easy ⇒ bet OVER at laggard.
    Model-free. Primary: leader=FD (FD leads per the lag data); control: leader=DK."""
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    actuals = scan._player_week_index(seasons)
    print(f"\n  FADE-THE-LAGGARD (bet stale book's favorable side, rung={rung_pref}):")
    for leader, laggard in (("fanduel", "draftkings"), ("draftkings", "fanduel")):
        by = defaultdict(lambda: [0, 0.0, 0.0, 0.0])   # season -> [n, wins, impl, roi]
        fair_pl = [0, 0.0]
        for prop in props:
            stat = acc.PROPS[prop]["stat"]
            lines, meta = clv._load_local(prop, seasons)
            for (eid, player), bybook in lines.items():
                lead = bybook.get(leader, {}); lag = bybook.get(laggard, {})
                src = (next((s for s in RUNG_ORDER if s in lead and s in lag), None)
                       if rung_pref == "opener" else rung_pref)
                if src is None or src not in lead or src not in lag:
                    continue
                m = meta[(eid, player)]
                gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
                if gid is None:
                    continue
                ps = gid.split("_")
                av = (actuals.get((scan._norm(player), ps[0], str(int(ps[1])))) or {}).get(stat)
                if av is None:
                    continue
                ll = _line_px(lag[src], "OVER")[0] or _line_px(lag[src], "UNDER")[0]
                el = _line_px(lead[src], "OVER")[0] or _line_px(lead[src], "UNDER")[0]
                if ll is None or el is None or abs(ll - el) < 1e-9:
                    continue
                over = ll < el                            # laggard line low ⇒ bet its OVER
                pt, px = _line_px(lag[src], "OVER" if over else "UNDER")
                if pt is None or abs(av - pt) < 1e-9:
                    continue
                won = (av > pt) if over else (av < pt)
                fair = clv._fair_over(lag[src])
                impl = (fair if over else 1 - fair) if fair is not None else 0.5
                roi = (american_to_decimal(px) - 1.0) if won else -1.0
                r = by[ps[0]]
                r[0] += 1; r[1] += 1 if won else 0; r[2] += impl; r[3] += roi
                if -110 <= px < 100:
                    fair_pl[0] += 1; fair_pl[1] += roi
        print(f"    leader={leader}, bet {laggard} stale side:")
        for s in sorted(by):
            n, w, im, roi = by[s]
            if n < 30:
                continue
            print(f"      {s}: n={n:>5}  real-impl={(w-im)/n*100:+5.2f}%  ROI={roi/n*100:+6.2f}%")
        if fair_pl[0]:
            print(f"      FAIR-price ROI={fair_pl[1]/fair_pl[0]*100:+.2f}% (n={fair_pl[0]})")


def best_ev(props, train_seasons, test_seasons, rung_pref, diverge_only=False):
    """Doug's exact rule: for the model's side, pick the DK/FD offer that MAXIMIZES model EV
    (P(side@that line) × decimal_odds − 1), at each offer's own line+price. Then bucket by
    that best EV (the 'required-price' reframe): does betting only EV≥X select winners?"""
    buckets = [(-9, 0.0, "any"), (0.0, 0.03, "EV 0-3%"), (0.03, 0.06, "EV 3-6%"),
               (0.06, 9, "EV≥6%")]
    agg = {lbl: defaultdict(lambda: [0, 0.0, 0.0]) for *_a, lbl in buckets}  # lbl->season->[n,wins,roi]
    fairagg = {lbl: [0, 0.0] for *_a, lbl in buckets}
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
            src = (next((s for s in RUNG_ORDER if s in bybook.get("draftkings", {})
                         or s in bybook.get("fanduel", {})), None)
                   if rung_pref == "opener" else rung_pref)
            at = {b: bybook[b][src] for b in ("draftkings", "fanduel")
                  if b in bybook and src in bybook.get(b, {})} if src else {}
            if not at:
                continue
            if diverge_only:                          # abstain when books agree
                if len(at) < 2:
                    continue
                dl = _line_px(at["draftkings"], "OVER")[0] or _line_px(at["draftkings"], "UNDER")[0]
                fl = _line_px(at["fanduel"], "OVER")[0] or _line_px(at["fanduel"], "UNDER")[0]
                if dl is None or fl is None or abs(dl - fl) < 1e-9:
                    continue
            ref = at.get("draftkings") or at.get("fanduel")
            fair = clv._fair_over(ref)
            if fair is None:
                continue
            l0 = _line_px(ref, "OVER")[0] or _line_px(ref, "UNDER")[0]
            over = acc.p_over(o, l0, cfg, comp) >= fair
            # best-EV offer for our side across the books
            best = None
            for d in at.values():
                pt, px = _line_px(d, "OVER" if over else "UNDER")
                if pt is None:
                    continue
                pw = acc.p_over(o, pt, cfg, comp)
                psd = pw if over else 1 - pw
                ev = psd * american_to_decimal(px) - 1.0
                if best is None or ev > best[0]:
                    best = (ev, pt, px)
            if best is None or abs(actual - best[1]) < 1e-9:
                continue
            ev, pt, px = best
            won = (actual > pt) if over else (actual < pt)
            roi = (american_to_decimal(px) - 1.0) if won else -1.0
            for lo, hi, lbl in buckets:
                if lo <= ev < hi:
                    r = agg[lbl][season]
                    r[0] += 1; r[1] += 1 if won else 0; r[2] += roi
                    if -110 <= px < 100:
                        fairagg[lbl][0] += 1; fairagg[lbl][1] += roi
    tag = "DIVERGENT-ONLY (abstain when books agree)" if diverge_only else "ALL cases"
    print(f"\n  BEST-EV OFFER [{tag}] (max-EV DK/FD offer for model's side, rung={rung_pref}):")
    print("    (EV bucket = the 'required-price' reframe — only bet if best EV clears the bar)")
    for *_a, lbl in buckets:
        row = agg[lbl]
        cells = []
        for s in sorted(row):
            n, w, roi = row[s]
            if n >= 30:
                cells.append(f"{s}:{roi/n*100:+.1f}%(n={n})")
        fp = fairagg[lbl]
        fair = f"FAIR {fp[1]/fp[0]*100:+.1f}%(n={fp[0]})" if fp[0] else "FAIR:na"
        print(f"    {lbl:<9} " + "  ".join(cells) + f"  | {fair}")


def doug_rule(props, seasons, rung_pref):
    """Doug's model-free rule: when DK/FD lines differ, the higher-line book 'knows' the total
    is high → bet OVER at the LOWER-line book. Tests both that AND its mirror (UNDER at the
    higher-line book = trust the lower book) to see which book is actually the informed one.
    real-impl = did the side win MORE than the chosen book's own price implied (mispriced?)."""
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    actuals = scan._player_week_index(seasons)
    print(f"\n  DOUG'S RULE (divergent lines, model-free; rung={rung_pref}):")
    for label, side in (("OVER @ lower-line book (trust higher book)", "over_low"),
                        ("UNDER @ higher-line book (trust lower book)", "under_high")):
        by = defaultdict(lambda: [0, 0.0, 0.0, 0.0])   # season->[n,wins,impl,roi]
        fair = [0, 0.0]
        for prop in props:
            stat = acc.PROPS[prop]["stat"]
            lines, meta = clv._load_local(prop, seasons)
            for (eid, player), bybook in lines.items():
                dk = bybook.get("draftkings", {}); fd = bybook.get("fanduel", {})
                src = (next((s for s in RUNG_ORDER if s in dk and s in fd), None)
                       if rung_pref == "opener" else rung_pref)
                if src is None or src not in dk or src not in fd:
                    continue
                m = meta[(eid, player)]
                gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
                if gid is None:
                    continue
                ps = gid.split("_")
                av = (actuals.get((scan._norm(player), ps[0], str(int(ps[1])))) or {}).get(stat)
                if av is None:
                    continue
                dl = _line_px(dk[src], "OVER")[0] or _line_px(dk[src], "UNDER")[0]
                fl = _line_px(fd[src], "OVER")[0] or _line_px(fd[src], "UNDER")[0]
                if dl is None or fl is None or abs(dl - fl) < 1e-9:
                    continue
                low, high = (dk[src], fd[src]) if dl < fl else (fd[src], dk[src])
                if side == "over_low":
                    pt, px = _line_px(low, "OVER"); over = True
                    fr = clv._fair_over(low)
                else:
                    pt, px = _line_px(high, "UNDER"); over = False
                    frov = clv._fair_over(high); fr = (1 - frov) if frov is not None else None
                if pt is None or px is None or abs(av - pt) < 1e-9:
                    continue
                won = (av > pt) if over else (av < pt)
                roi = (american_to_decimal(px) - 1.0) if won else -1.0
                r = by[ps[0]]
                r[0] += 1; r[1] += 1 if won else 0
                r[2] += fr if fr is not None else 0.5; r[3] += roi
                if -110 <= px < 100:
                    fair[0] += 1; fair[1] += roi
        print(f"    {label}:")
        for s in sorted(by):
            n, w, im, roi = by[s]
            if n >= 30:
                print(f"      {s}: n={n:>5}  win={w/n*100:4.1f}%  real-impl={(w-im)/n*100:+5.2f}%  "
                      f"ROI={roi/n*100:+6.2f}%")
        if fair[0]:
            print(f"      FAIR-price ROI={fair[1]/fair[0]*100:+.2f}% (n={fair[0]})")


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
    doug_rule(props, test_seasons, args.rung)
    print("=" * 100)


if __name__ == "__main__":
    main()
