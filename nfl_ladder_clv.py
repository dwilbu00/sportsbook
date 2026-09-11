"""nfl_ladder_clv.py — edge & CLV vs time-to-kickoff across the opener→close ladder.

Tests Doug's thesis directly: the EARLIER you bet, the more it's model-vs-model (the line ≈
the book's own opener, before sharp money) — beatable; the LATER, the more it's
model-vs-market (consensus) — hopeless. So our frozen model's edge should be FAT at the
opener rungs and DECAY toward close.

For each of the 8 props we take the FROZEN model (components trained on 2012-2022, the
nested-CV `SWEPT` config — never sees the odds era) and, at every ladder rung
(ladder_114h … ladder_006h, then the existing early_4h and closing), measure on the
EXECUTABLE books (DK/FD):

  real−impl : realized(our side) − de-vigged implied(our side) at that rung — price-robust
              edge; POSITIVE and shrinking toward close ⇒ the thesis holds.
  ROI       : at that rung's price for the side our model prefers.
  CLV→close : de-vig(our side)@close − de-vig(our side)@rung — did the line move our way?
  vs sharp  : how far the DK/FD number sits from Pinnacle (the analysis-only sharp
              reference) at that rung, and whether our side agrees with that gap.

Reads the warehouse mirror (all sources at once). Bets are DK/FD only; Pinnacle is the
sharp reference (never sized/recommended). Diagnostic — writes nothing, spends nothing.
"""
import argparse
from collections import defaultdict

import nfl_schedule
import warehouse_mirror as wm
import nfl_props_scan as scan
import nfl_props_accuracy as acc
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way

SPORT = "americanfootball_nfl"
EXEC_BOOKS = ["draftkings", "fanduel"]
SHARP_BOOK = "pinnacle"
# rung label → nominal hours-to-kickoff (for the decay axis), earliest→latest
RUNGS = ([(f"ladder_{h:03d}h", float(h)) for h in range(114, 5, -6)]
         + [("early_4h", 4.0), ("closing", 0.17)])


def _load_all(prop, seasons, books):
    """{(event_id, player): {book: {source: {'OVER':(pt,px),'UNDER':(pt,px)}}}} + game meta."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    meta = {}
    for book in books:
        for y in seasons:
            rows = wm.player_prop_lines(SPORT, date_from=f"{y}-01-01",
                                        date_to=f"{y}-12-31", bookmaker=book) or []
            for r in rows:
                if r.get("prop_key") != prop:
                    continue
                src = r.get("source")
                direction = str(r.get("direction") or "").upper()
                pt, px = r.get("point"), r.get("price")
                if direction not in ("OVER", "UNDER") or pt is None or px is None:
                    continue
                key = (r.get("event_id"), r.get("player"))
                out[key][book][src][direction] = (pt, px)
                meta.setdefault(key, {"commence": r.get("commence_time"),
                                      "home": r.get("home"), "away": r.get("away")})
    return out, meta


def _fair_over(d):
    o, u = d.get("OVER"), d.get("UNDER")
    if not (o and u):
        return None
    try:
        return devig_two_way(american_to_implied_prob(o[1]),
                             american_to_implied_prob(u[1]))[0]
    except Exception:
        return None


def run(prop, cfg, train_seasons, test_seasons):
    hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
    acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
    train = acc._obs_from_series(acc._series(train_seasons), cfg)
    if len(train) < 200:
        print(f"  {prop:<24} (train thin)")
        return None
    comp = acc.fit_components(train, cfg)
    feat = {(o["name"], o["season"], o["week"]): o
            for o in acc._obs_from_series(acc._series(test_seasons), cfg)}
    idx = nfl_schedule.game_index([str(s) for s in test_seasons])

    lines, meta = _load_all(prop, test_seasons, EXEC_BOOKS + [SHARP_BOOK])
    # per-rung accumulators
    agg = {src: {"n": 0, "real": 0.0, "impl": 0.0, "roi": 0.0, "clv": 0.0, "nclv": 0,
                 "vsharp": 0.0, "nsharp": 0} for src, _ in RUNGS}

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
        # closing fair per exec book (for CLV reference), and sharp closing
        for book in EXEC_BOOKS:
            src_map = bybook.get(book, {})
            close = src_map.get("closing")
            close_fair = _fair_over(close) if close else None
            for src, _h in RUNGS:
                d = src_map.get(src)
                if not d:
                    continue
                line = d.get("OVER", d.get("UNDER", (None, None)))[0]
                if line is None or abs(actual - line) < 1e-9:
                    continue
                fair = _fair_over(d)
                if fair is None:
                    continue
                p = acc.p_over(o, line, cfg, comp)
                over_side = p >= fair
                win = (actual > line) if over_side else (actual < line)
                px = (d.get("OVER") if over_side else d.get("UNDER"))[1]
                a = agg[src]
                a["n"] += 1
                a["real"] += 1 if win else 0
                a["impl"] += fair if over_side else (1 - fair)
                a["roi"] += (american_to_decimal(px) - 1.0) if win else -1.0
                if close_fair is not None:
                    our_close = close_fair if over_side else 1 - close_fair
                    our_rung = fair if over_side else 1 - fair
                    a["clv"] += our_close - our_rung
                    a["nclv"] += 1
                # sharp gap: pinnacle fair at same rung
                sd = bybook.get(SHARP_BOOK, {}).get(src)
                sf = _fair_over(sd) if sd else None
                if sf is not None:
                    our_book = fair if over_side else 1 - fair
                    our_sharp = sf if over_side else 1 - sf
                    a["vsharp"] += our_sharp - our_book   # +: sharp likes our side more
                    a["nsharp"] += 1
    return agg


def _print(prop, agg):
    if not agg:
        return
    print(f"\n  {prop}")
    print(f"    {'rung':<12} {'n':>6} {'real-impl':>10} {'ROI':>8} {'CLV→close':>10} {'vs sharp':>9}")
    for src, _h in RUNGS:
        a = agg[src]
        if a["n"] < 30:
            continue
        ri = (a["real"] - a["impl"]) / a["n"] * 100
        roi = a["roi"] / a["n"] * 100
        clv = (a["clv"] / a["nclv"] * 100) if a["nclv"] else float("nan")
        vs = (a["vsharp"] / a["nsharp"] * 100) if a["nsharp"] else float("nan")
        print(f"    {src:<12} {a['n']:>6} {ri:>+9.2f}% {roi:>+7.2f}% {clv:>+9.2f}% {vs:>+8.2f}%")


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
    props = list(acc.PROPS) if args.prop == "all" else [args.prop]
    print("=" * 92)
    print(f"  NFL LADDER CLV — frozen model (train {train_seasons[0]}-{train_seasons[-1]}) "
          f"edge/CLV by rung, DK/FD executable, test {test_seasons}")
    print("  Thesis: real-impl POSITIVE & shrinking opener→close ⇒ early is beatable.")
    print("=" * 92)
    pooled = {src: {"n": 0, "real": 0.0, "impl": 0.0, "roi": 0.0, "clv": 0.0,
                    "nclv": 0, "vsharp": 0.0, "nsharp": 0} for src, _ in RUNGS}
    for prop in props:
        agg = run(prop, acc.PROPS[prop], train_seasons, test_seasons)
        if not agg:
            continue
        _print(prop, agg)
        for src, _h in RUNGS:
            for kk in pooled[src]:
                pooled[src][kk] += agg[src][kk]
    print("\n" + "=" * 92)
    _print("POOLED (all props)", pooled)
    print("=" * 92)


if __name__ == "__main__":
    main()
