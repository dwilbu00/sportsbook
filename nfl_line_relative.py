"""nfl_line_relative.py — characterize the outcome distribution RELATIVE to the book line.

Doug's "harness, don't beat" idea, done with the right statistics. The standard line is
efficient, so we treat it as a good anchor and measure how ACTUALS fall around it:

  * TRUE bias = median(actual) - line  (NOT "how far under when it goes under" — that's just
    left-tail dispersion and would overstate the bias ~10x). Small negative = mild overest.
  * SHAPE = skew of (actual - line). Prop outcomes (esp. yardage) are right-skewed (boom
    games = fat right tail); if the book prices ALTERNATE lines off a too-symmetric model,
    specific alternates could be mispriced -> the only place a real edge could hide here.
  * ALTERNATE-LINE HIT CURVE = for alternate over lines below the standard (and unders
    above), the EMPIRICAL P(hit) and the FAIR (break-even) American odds it implies. Compare
    to what a book actually charges for that alternate: if the book's price is BETTER than
    the fair break-even, that alternate is +EV. (Confirming that needs alternate-odds data —
    a separate fetch; this is the free half that says whether it's even plausible.)

Reads the local ladder extract (closing DK lines) + actuals. Diagnostic — no spend.
"""
import argparse
import math
from collections import defaultdict

import pandas as pd

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc

STORE = "nfl_ladder_data"
YARDAGE = {"player_pass_yds", "player_rush_yds", "player_reception_yds"}
# alternate offsets: fraction of the line for yardage, absolute counts otherwise
FRAC_OFFSETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ABS_OFFSETS = [1, 2, 3, 4]


def _fair_american(p):
    if p <= 0.0 or p >= 1.0:
        return None
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def _pct(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals); i = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
    return s[i]


def _skew(vals):
    n = len(vals)
    if n < 3:
        return float("nan")
    m = sum(vals) / n
    sd = (sum((v - m) ** 2 for v in vals) / n) ** 0.5
    if sd == 0:
        return float("nan")
    return sum(((v - m) / sd) ** 3 for v in vals) / n


def load(seasons):
    """{prop: [(line, actual)]} at the CLOSING DK line, matched to actuals."""
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    actuals = scan._player_week_index([str(s) for s in seasons])
    out = defaultdict(list)
    for s in seasons:
        path = f"{STORE}/ladder_lines__{s}.parquet"
        try:
            df = pd.read_parquet(path)
        except Exception:
            print(f"  [warn] missing {path} — run nfl_ladder_clv.py --extract --test {s}")
            continue
        df = df[(df["source"] == "closing") & (df["book"] == "draftkings")]
        seen = set()
        for r in df.itertuples(index=False):
            key = (r.event_id, r.player, r.prop_key)
            if key in seen or r.point is None:
                continue
            seen.add(key)
            gid, _ = nfl_schedule.resolve_event(r.home, r.away, r.commence, index=idx)
            if gid is None:
                continue
            ps = gid.split("_")
            stat = acc.PROPS.get(r.prop_key, {}).get("stat")
            if not stat:
                continue
            av = (actuals.get((scan._norm(r.player), ps[0], str(int(ps[1])))) or {}).get(stat)
            if av is None:
                continue
            out[r.prop_key].append((float(r.point), float(av)))
    return out


def report(prop, pairs):
    n = len(pairs)
    if n < 100:
        print(f"\n  {prop}: n={n} (thin) — skip")
        return
    diffs = [a - l for l, a in pairs]
    over_rate = sum(1 for d in diffs if d > 0) / n
    med = _pct(diffs, 0.5)
    mean = sum(diffs) / n
    sk = _skew(diffs)
    print(f"\n  {prop}  (n={n:,})")
    print(f"    over-rate={over_rate*100:.1f}%   median(actual-line)={med:+.1f}   "
          f"mean={mean:+.1f}   skew={sk:+.2f}"
          + ("  [book median-overestimates]" if med < -0.5 else
             "  [book median-underestimates]" if med > 0.5 else "  [~unbiased median]"))
    # alternate OVER lines (easier overs, below the standard line): empirical P(hit) + fair odds
    yard = prop in YARDAGE
    offs = FRAC_OFFSETS if yard else ABS_OFFSETS
    print(f"    alternate OVER (easier line) — empirical hit% and FAIR (break-even) odds:")
    for off in offs:
        # alt line = each obs's own line reduced (frac of line, or absolute)
        hits = 0
        for l, a in pairs:
            alt = l * (1 - off) if yard else l - off
            if a > alt:
                hits += 1
        p = hits / n
        fa = _fair_american(p)
        lbl = f"line-{int(off*100)}%" if yard else f"line-{off}"
        print(f"      {lbl:<10} P(over)={p*100:5.1f}%   fair odds={fa:+.0f}"
              if fa is not None else f"      {lbl:<10} P(over)={p*100:5.1f}%")
    print(f"    alternate UNDER (easier line) — empirical hit% and FAIR odds:")
    for off in offs:
        hits = 0
        for l, a in pairs:
            alt = l * (1 + off) if yard else l + off
            if a < alt:
                hits += 1
        p = hits / n
        fa = _fair_american(p)
        lbl = f"line+{int(off*100)}%" if yard else f"line+{off}"
        print(f"      {lbl:<10} P(under)={p*100:5.1f}%   fair odds={fa:+.0f}"
              if fa is not None else f"      {lbl:<10} P(under)={p*100:5.1f}%")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--prop", default="all")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    data = load(seasons)
    props = list(acc.PROPS) if args.prop == "all" else [args.prop]
    print("=" * 92)
    print(f"  NFL LINE-RELATIVE OUTCOME DISTRIBUTION — closing DK, {seasons}")
    print("  READ: a book alternate is +EV only if its OFFERED odds beat the FAIR odds here.")
    print("  Right-skew + fat over-tail = where alternate mispricing could hide.")
    print("=" * 92)
    for prop in props:
        if prop in data:
            report(prop, data[prop])
    print("=" * 92)


if __name__ == "__main__":
    main()
