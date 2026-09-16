"""mlb_opportunity_threshold.py — find the OPPORTUNITY cutoff where MLB prop legs are
market-trustworthy (the MLB analog of nfl_opportunity_threshold; Doug runs it).

The MLB bonus optimizer will field legs ONLY from props above an opportunity gate (expected PA
for batters, expected OUTS for pitchers). This validates, per prop, the LOWEST opportunity T at
which the de-vigged MARKET price (the calibrated leg probability the bonus uses) is well-
calibrated against realized outcomes — so above T a leg's market P can be trusted, below T it's
noisy/participation-risky and we abstain. Chosen on DISCOVERY seasons, confirmed on HELD-OUT.

Substrate (all durable, no spend): historical real-line prop lines via
book_line_calibration.harvest_real_line_book_lines + join_book_lines_to_actuals (warehouse
actuals + strictly-earlier in-season prior_games), opportunity via mlb_opportunity_serving
(computed from prior_games — leakage-safe). Market P = devig_two_way(over,under).
"""
import argparse
from collections import defaultdict

import book_line_calibration as blc
import mlb_opportunity_serving as opp
from odds_client import american_to_implied_prob, devig_two_way

MLB_PROPS = ["batter_hits", "batter_total_bases", "batter_strikeouts", "batter_rbis",
             "pitcher_strikeouts", "pitcher_earned_runs", "pitcher_outs"]
UNIT = {p: ("PA" if p.startswith("batter_") else "outs") for p in MLB_PROPS}


def _fair_over(over_price, under_price):
    if over_price is None or under_price is None:
        return None
    try:
        return devig_two_way(american_to_implied_prob(over_price),
                             american_to_implied_prob(under_price))[0]
    except Exception:
        return None


def collect(props, snapshot):
    """{prop: [record]} where record = {season, opp, y, fair} for each joinable real-line MLB
    prop bet (pushes dropped)."""
    book_lines, _n_primary, _n_pred = blc.harvest_real_line_book_lines(
        "baseball_mlb", props, snapshot=snapshot)
    print(f"  harvested {len(book_lines)} real book line(s); joining to actuals…")
    enriched = blc.join_book_lines_to_actuals(book_lines, "baseball", "mlb")
    out = defaultdict(list)
    for r in enriched:
        prop = r.get("prop_key")
        if prop not in props:
            continue
        line, actual = r.get("line"), r.get("actual")
        if line is None or actual is None or abs(actual - line) < 1e-9:
            continue                                   # push / missing
        fair = _fair_over(r.get("over_price"), r.get("under_price"))
        if fair is None:
            continue
        o = opp.opportunity_from_rows(r.get("prior_games"), prop)
        if o is None:
            continue
        season = str(r.get("game_date") or "")[:4]
        out[prop].append({"season": season, "opp": o,
                          "y": 1 if actual > line else 0, "fair": fair})
    return out


def _cal(sub):
    """(n, market Brier, calibration gap |mean fair - mean y|, realized over%, implied%)."""
    n = len(sub)
    if not n:
        return None
    my = sum(r["y"] for r in sub) / n
    mf = sum(r["fair"] for r in sub) / n
    brier = sum((r["fair"] - r["y"]) ** 2 for r in sub) / n
    return n, brier, abs(mf - my), my * 100, mf * 100


def run(props, discovery, val, tol, snapshot):
    data = collect(props, snapshot)
    print("=" * 96)
    print(f"  MLB OPPORTUNITY THRESHOLD — market-calibration by opportunity, tol={tol}")
    print(f"  discovery={discovery}  held-out={val}   (gate = lowest opp T with discovery cal-gap<=tol)")
    print("=" * 96)
    for prop in props:
        recs = data.get(prop, [])
        disc = [r for r in recs if r["season"] in discovery]
        held = [r for r in recs if r["season"] in val]
        if len(disc) < 300:
            print(f"\n  {prop:<22} thin ({len(disc)} discovery) — skip")
            continue
        opps = sorted(r["opp"] for r in disc)
        grid = sorted(set(round(opps[int(q * (len(opps) - 1))], 1)
                          for q in [i / 20 for i in range(20)]))
        best = None
        for T in grid:
            c = _cal([r for r in disc if r["opp"] >= T])
            if c and c[0] >= 150 and c[2] <= tol:
                best = T
                break
        print(f"\n  {prop}  ({UNIT[prop]})   discovery n={len(disc)}  held-out n={len(held)}")
        print(f"    {'T':>6} {'disc n':>7} {'calGap':>7} {'over%':>6} {'impl%':>6}   "
              f"{'held n':>6} {'calGap':>7}")
        for T in grid[::2]:
            dc = _cal([r for r in disc if r["opp"] >= T])
            hc = _cal([r for r in held if r["opp"] >= T])
            if not dc:
                continue
            mark = " <= GATE" if best is not None and abs(T - best) < 1e-9 else ""
            print(f"    {T:>6.1f} {dc[0]:>7} {dc[2]:>+7.3f} {dc[3]:>5.0f}% {dc[4]:>5.0f}%   "
                  f"{(hc[0] if hc else 0):>6} {(hc[2] if hc else float('nan')):>+7.3f}{mark}")
        if best is None:
            print(f"    => NO opp T gets cal-gap <= {tol} on discovery → market not reliably "
                  "calibrated for this prop; abstain / needs a tighter leg universe.")
        else:
            hc = _cal([r for r in held if r["opp"] >= best])
            ok = hc and hc[2] <= tol * 1.5
            print(f"    => GATE T = {best:.1f} {UNIT[prop]}; HELD-OUT cal-gap="
                  f"{(hc[2] if hc else float('nan')):+.3f} "
                  f"{'CONFIRMED' if ok else 'FAILS (re-widens)'}")
    print("=" * 96)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discovery", default="2023,2024,2025")
    ap.add_argument("--val", default="2026")
    ap.add_argument("--tol", type=float, default=0.03,
                    help="max acceptable market calibration gap (|mean implied - mean realized|)")
    ap.add_argument("--prop", default="all")
    ap.add_argument("--snapshot", default="closing")
    args = ap.parse_args()
    props = MLB_PROPS if args.prop == "all" else [args.prop]
    discovery = tuple(s.strip() for s in args.discovery.split(",") if s.strip())
    val = tuple(s.strip() for s in args.val.split(",") if s.strip())
    run(props, discovery, val, args.tol, args.snapshot)


if __name__ == "__main__":
    main()
