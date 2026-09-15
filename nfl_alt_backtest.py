"""nfl_alt_backtest.py — do DK/FD misprice ALTERNATE lines on the right-skewed props?

Grades every captured alternate (line, odds) vs the actual outcome, bucketed by SIDE and by
OFFSET from the standard closing line (signed % of the standard line). The skew hypothesis:
OVER bets at POSITIVE offset (above the line, in the fat right tail) are +EV if the book
prices the ladder off a too-symmetric model. Guardrail (per our scars): a bucket counts only
if +ROI EVERY season at the real book prices — else it's noise/winner's-curse.

Reads the local alt capture (nfl_alt_data) + the standard closing line (nfl_ladder_data) +
actuals. Diagnostic — no spend.
"""
import argparse
from collections import defaultdict

import pandas as pd

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
from odds_client import american_to_decimal

ALT_TO_STD = {"player_rush_yds_alternate": "player_rush_yds",
              "player_reception_yds_alternate": "player_reception_yds"}
# offset buckets: (alt_point - std_line)/std_line
OVER_BUCKETS = [(-0.60, -0.20), (-0.20, -0.10), (-0.10, 0.0),
                (0.0, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 9.0)]
UNDER_BUCKETS = [(-9.0, -0.30), (-0.30, -0.15), (-0.15, 0.0),
                 (0.0, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 9.0)]


def _std_lines(seasons):
    """{(event_id, player, std_prop): std_closing_DK_line}."""
    out = {}
    for s in seasons:
        try:
            df = pd.read_parquet(f"nfl_ladder_data/ladder_lines__{s}.parquet")
        except Exception:
            continue
        df = df[(df.source == "closing") & (df.book == "draftkings")
                & (df.prop_key.isin(ALT_TO_STD.values()))]
        for r in df.itertuples(index=False):
            k = (r.event_id, r.player, r.prop_key)
            if k not in out and r.point is not None:
                out[k] = float(r.point)
    return out


def run(seasons, book_filter):
    std = _std_lines(seasons)
    actuals = scan._player_week_index([str(s) for s in seasons])
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    # bets[(market, book, side, bucket)][season] = [n, wins, roi]
    bets = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    n_rows = n_used = 0
    for s in seasons:
        try:
            df = pd.read_parquet(f"nfl_alt_data/alt__{s}.parquet")
        except Exception:
            print(f"  [warn] missing nfl_alt_data/alt__{s}.parquet")
            continue
        for r in df.itertuples(index=False):
            n_rows += 1
            if book_filter and r.book != book_filter:
                continue
            std_prop = ALT_TO_STD.get(r.market)
            if not std_prop or r.point is None or r.price is None or not r.side:
                continue
            sl = std.get((r.event_id, r.player, std_prop))
            if sl is None or sl <= 0:
                continue
            gid, _ = nfl_schedule.resolve_event(r.home, r.away, r.commence, index=idx)
            if gid is None:
                continue
            ps = gid.split("_")
            stat = acc.PROPS[std_prop]["stat"]
            av = (actuals.get((scan._norm(r.player), ps[0], str(int(ps[1])))) or {}).get(stat)
            if av is None or abs(av - r.point) < 1e-9:
                continue
            over = str(r.side).lower().startswith("o")
            offset = (r.point - sl) / sl
            buckets = OVER_BUCKETS if over else UNDER_BUCKETS
            b = next((bb for bb in buckets if bb[0] <= offset < bb[1]), None)
            if b is None:
                continue
            won = (av > r.point) if over else (av < r.point)
            roi = (american_to_decimal(r.price) - 1.0) if won else -1.0
            rec = bets[(r.market, r.book, "OVER" if over else "UNDER", b)][ps[0]]
            rec[0] += 1; rec[1] += 1 if won else 0; rec[2] += roi
            n_used += 1
    print("=" * 100)
    print(f"  NFL ALT-LINE BACKTEST — {seasons}  ({n_used:,}/{n_rows:,} alt bets graded)")
    print("  +ROI EVERY season = candidate; OVER at positive offset = the fat-tail bet.")
    print("=" * 100)
    for market in sorted(ALT_TO_STD):
        for book in (["draftkings", "fanduel"] if not book_filter else [book_filter]):
            for side, bl in (("OVER", OVER_BUCKETS), ("UNDER", UNDER_BUCKETS)):
                hdr = False
                for b in bl:
                    per = bets[(market, book, side, b)]
                    tot_n = sum(per[s][0] for s in per)
                    if tot_n < 60:
                        continue
                    if not hdr:
                        print(f"\n  {market.replace('player_','').replace('_alternate','')} "
                              f"[{book}] {side}:"); hdr = True
                    cells = []
                    ok = True
                    for s in ("2023", "2024", "2025"):
                        n, w, roi = per.get(s, [0, 0, 0])
                        if n >= 20:
                            cells.append(f"{s}:{roi/n*100:+.0f}%(n={n})")
                            if roi <= 0:
                                ok = False
                        else:
                            cells.append(f"{s}:-"); ok = False
                    alln = sum(per[s][0] for s in per)
                    allroi = sum(per[s][2] for s in per) / alln * 100
                    lbl = f"off[{int(b[0]*100):+d},{int(b[1]*100):+d}%)" if b[1] < 9 else f"off≥{int(b[0]*100):+d}%"
                    tag = "  <<+EV all seasons" if ok else ""
                    print(f"    {lbl:<16} pooled ROI={allroi:+6.1f}% (n={alln:>4})  "
                          + "  ".join(cells) + tag)
    print("=" * 100)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--book", default="", help="draftkings|fanduel|'' for both")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    run(seasons, args.book or None)


if __name__ == "__main__":
    main()
