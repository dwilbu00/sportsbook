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
import os
os.environ.setdefault("SQL_TIMEOUT", "900")   # 20-DTU tier: big analytical reads need >60s
from collections import defaultdict

import pandas as pd
from sqlalchemy import text, bindparam

import db_store
import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way

SPORT = "americanfootball_nfl"
EXEC_BOOKS = ["draftkings", "fanduel"]
SHARP_BOOK = "pinnacle"
BOOKS = EXEC_BOOKS + [SHARP_BOOK]
STORE = "nfl_ladder_data"
# rung label → nominal hours-to-kickoff (for the decay axis), earliest→latest
RUNGS = ([(f"ladder_{h:03d}h", float(h)) for h in range(114, 5, -6)]
         + [("early_4h", 4.0), ("closing", 0.17)])


# index-friendly: sport + game_date range hits the uq_odds_snapshot prefix (sport,game_date,
# ...), kind='props' skips team snapshots, then the snapshot→line join uses ix_odds_line_snapshot.
_EXTRACT_Q = text(
    "SELECT s.event_id, l.player, s.source, l.bookmaker, l.direction, "
    "       l.point, l.price, s.commence_time, s.home, s.away, l.prop_key "
    "FROM odds_snapshot s JOIN odds_line l ON l.snapshot_id = s.id "
    "WHERE s.sport = :sp AND s.kind = 'props' "
    "  AND s.game_date >= :d0 AND s.game_date <= :d1 "
    "  AND l.bookmaker IN :books "
    "  AND (s.source LIKE 'ladder_%' OR s.source IN ('early_4h', 'closing'))"
).bindparams(bindparam("books", expanding=True))


def _store_path(season):
    return os.path.join(STORE, f"ladder_lines__{season}.parquet")


def extract(seasons):
    """ONE-TIME targeted pull Azure→local parquet (per season): all ladder + early_4h/closing
    prop lines for DK/FD/Pinnacle. Azure stays the durable source; this is a re-creatable
    work cache so the analysis reads locally instead of hammering the 20-DTU tier per prop."""
    db_store.promote_secrets_from_toml()
    if not db_store.enabled():
        print("  Azure SQL not configured — cannot extract. Aborting.")
        return
    os.makedirs(STORE, exist_ok=True)
    eng = db_store.get_engine()
    for s in seasons:
        path = _store_path(s)
        if os.path.exists(path):
            print(f"  {s}: already extracted ({path}) — skipping")
            continue
        print(f"  {s}: querying Azure (this is the slow one-time read)...", flush=True)
        rows = []
        with eng.connect() as c:
            res = c.execute(_EXTRACT_Q, {"sp": SPORT, "d0": f"{s}-01-01",
                                         "d1": f"{s}-12-31", "books": BOOKS})
            for r in res:
                rows.append({"event_id": r[0], "player": r[1], "source": r[2],
                             "book": r[3], "direction": str(r[4]).upper(),
                             "point": r[5], "price": r[6], "commence": r[7],
                             "home": r[8], "away": r[9], "prop_key": r[10]})
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"  {s}: wrote {len(rows):,} lines → {path}")


def _load_local(prop, seasons):
    """{(event_id, player): {book: {source: {'OVER':(pt,px),'UNDER':(pt,px)}}}}, meta —
    read the extracted local parquet(s) and filter to this prop."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    meta = {}
    for s in seasons:
        path = _store_path(s)
        if not os.path.exists(path):
            print(f"    [warn] no extract for {s} ({path}) — run --extract first")
            continue
        df = pd.read_parquet(path)
        df = df[df["prop_key"] == prop]
        for r in df.itertuples(index=False):
            if r.direction not in ("OVER", "UNDER") or r.point is None or r.price is None:
                continue
            out[(r.event_id, r.player)][r.book][r.source][r.direction] = (r.point, r.price)
            meta.setdefault((r.event_id, r.player),
                            {"commence": r.commence, "home": r.home, "away": r.away})
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

    lines, meta = _load_local(prop, test_seasons)
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
    ap.add_argument("--extract", action="store_true",
                    help="one-time targeted pull Azure→local parquet, then exit")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    props = list(acc.PROPS) if args.prop == "all" else [args.prop]
    if args.extract:
        print("=" * 92)
        print(f"  EXTRACT ladder lines Azure→local parquet, seasons {test_seasons}")
        print("=" * 92)
        extract(test_seasons)
        return
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
