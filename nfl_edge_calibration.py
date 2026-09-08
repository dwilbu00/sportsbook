"""nfl_edge_calibration.py — the REQUIRED-EDGE curve for real-time hunting.

Not "what method made money in hindsight" but: given our model vs the book vs
actuals, WHAT COMPUTED EDGE must the app see in the moment before a bet is truly
+EV? We bucket every bet by the app's model edge (EV%) computed at that snapshot's
OWN price, grade it at that same price against the real result, and plot realized
ROI per edge bucket. The bucket where ROI crosses 0 (and stays +) = the runtime
fire-threshold X*. At runtime we ignore average-ROI and fire only when the live
computed edge >= X*, hunting transient soft numbers the hours-apart snapshots miss.

What this proves: the NECESSARY condition — is our edge estimate INFORMATIVE
(ROI rises with computed edge)? If the curve slopes up and crosses 0, X* is real
and the model just needs to be as sharp as possible. If it's flat-negative or
slopes DOWN (bigger edge = we're just more wrong), the runtime strategy can't work.
It does NOT prove transients are exploitable — that needs live CLV logging (a
transient dip from dumb-money flooding is a gift; one from news we lack makes us
the sucker). Backtest = necessary; live logger = sufficient.

Model = the full EPA+QB+injury margin model, OOS (leave-one-season-out), converted
to cover/win probabilities via Normal(pred_margin, sigma_oos). Edge = model EV at
the book price. Graded at DK, all three snapshot windows.
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
from nfl_injury_edge import build_rows, _solve, _rmse, MODELS
from nfl_market_scan import (_load_closing_odds, _spread_outcome, _total_outcome,
                             _ml_outcome, _profit)
from odds_client import american_to_decimal
from r2_sharp import fair_two_way

# EV buckets (fraction). A bet is assigned to the highest bucket its EV clears.
EV_EDGES = [0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 1.0]
FEATURE = MODELS["base+QB+inj"]


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bucket(ev):
    for i in range(len(EV_EDGES) - 1):
        if EV_EDGES[i] <= ev < EV_EDGES[i + 1]:
            return f"{EV_EDGES[i]*100:.0f}-{EV_EDGES[i+1]*100:.0f}%"
    return None


def _oos_fit(rows, test):
    """Full-model weights + residual sigma fit on all seasons except `test`."""
    tr = [r for r in rows if r[5] != test]
    b = _solve([FEATURE(r) for r in tr], [r[4] for r in tr])
    resid = [r[4] - sum(bi * xi for bi, xi in zip(b, FEATURE(r))) for r in tr]
    sigma = (sum(e * e for e in resid) / len(resid)) ** 0.5 if resid else 13.5
    return b, sigma


def run(seasons, book="draftkings", snapshots=("early_12h", "early_4h", "closing")):
    rows = build_rows(seasons)
    by_gid = {r[6]: r for r in rows}
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    scores = nfl_schedule.team_scores_index([str(s) for s in seasons])

    # OOS model prediction + sigma per game
    pred, sig = {}, {}
    for test in seasons:
        b, sigma = _oos_fit(rows, test)
        for r in rows:
            if r[5] == test:
                pred[r[6]] = sum(bi * xi for bi, xi in zip(b, FEATURE(r)))
                sig[r[6]] = sigma

    # cells[(snapshot, market, bucket)][season] -> [profit]
    cells = defaultdict(lambda: defaultdict(list))

    for snap in snapshots:
        odds = _load_closing_odds([str(s) for s in seasons], book, snap)
        for eid, o in odds.items():
            gid, _ = nfl_schedule.resolve_event(o["home"], o["away"],
                                                o["commence_time"], index=idx)
            if gid is None or gid not in by_gid or gid not in scores:
                continue
            hs, as_ = scores[gid]
            season = by_gid[gid][5]
            pm, sg = pred.get(gid), sig.get(gid)
            if pm is None:
                continue

            def consider(market, prob_home, sides):
                """sides = [(is_home_side, point, price, outcome_fn)]. Bet the
                single +EV side with the highest model EV; bucket by that EV."""
                best = None
                for is_home, pt, px, outfn in sides:
                    if px is None:
                        continue
                    p = prob_home if is_home else (1.0 - prob_home)
                    dec = american_to_decimal(px)
                    if not dec:
                        continue
                    ev = p * dec - 1.0
                    if ev <= 0:
                        continue
                    if best is None or ev > best[0]:
                        best = (ev, px, outfn)
                if best:
                    bk = _bucket(best[0])
                    if bk:
                        cells[(snap, market, bk)][season].append(
                            _profit(best[1], best[2]()))

            # ── spread: cover prob for home = P(margin + home_pt > 0) ──
            sph, spa = o["spread"].get("home"), o["spread"].get("away")
            if sph and spa:
                p_home_cover = _norm_cdf((pm + sph[0]) / sg)
                consider("spread", p_home_cover, [
                    (True, sph[0], sph[1], lambda: _spread_outcome(True, sph[0], hs, as_)),
                    (False, spa[0], spa[1], lambda: _spread_outcome(False, spa[0], hs, as_)),
                ])
            # ── moneyline: win prob for home = P(margin > 0) ──
            mlh, mla = o["ml"].get("home"), o["ml"].get("away")
            if mlh is not None and mla is not None:
                p_home_win = _norm_cdf(pm / sg)
                consider("moneyline", p_home_win, [
                    (True, None, mlh, lambda: _ml_outcome(True, hs, as_)),
                    (False, None, mla, lambda: _ml_outcome(False, hs, as_)),
                ])

    _report(cells, seasons)


def _stats(pl):
    n = len(pl)
    if not n:
        return None
    mean = sum(pl) / n
    se = (math.sqrt(sum((p - mean) ** 2 for p in pl) / (n - 1) / n)) if n > 1 else 0.0
    wins = sum(1 for p in pl if p > 0)
    return {"n": n, "roi": mean, "hit": wins / n, "t": (mean / se if se else 0.0)}


def _report(cells, seasons):
    print("=" * 100)
    print("  NFL REQUIRED-EDGE calibration — realized ROI by app-computed edge (EV%) bucket")
    print("  full EPA+QB+injury model, OOS; graded at each snapshot's OWN DK price.")
    print("  X* = the edge bucket where ROI turns reliably positive = the runtime fire-threshold.")
    print("=" * 100)
    snaps = sorted({k[0] for k in cells}, key=lambda s: {"early_12h": 0, "early_4h": 1, "closing": 2}.get(s, 9))
    markets = ("spread", "moneyline")
    buckets = [f"{EV_EDGES[i]*100:.0f}-{EV_EDGES[i+1]*100:.0f}%" for i in range(len(EV_EDGES) - 1)]
    for snap in snaps:
        for market in markets:
            keys = [(snap, market, b) for b in buckets if (snap, market, b) in cells]
            if not keys:
                continue
            print(f"\n── {snap} / {market} ──")
            print(f"    {'edge bucket':<12}{'n':>5}{'ROI':>9}{'hit':>7}{'t':>6}   per-season")
            for (s, m, b) in keys:
                smap = cells[(s, m, b)]
                allp = [p for v in smap.values() for p in v]
                st = _stats(allp)
                if not st:
                    continue
                seas = " ".join(f"{ss}:{(sum(v)/len(v)*100 if v else 0):+.0f}%({len(v)})"
                                for ss, v in sorted(smap.items()))
                print(f"    {b:<12}{st['n']:>5}{st['roi']*100:+8.1f}%{st['hit']*100:6.1f}%"
                      f"{st['t']:+6.2f}   [{seas}]")
    print("\n" + "=" * 100)
    print("  READ: within a snapshot/market, does ROI RISE with the edge bucket and cross 0?")
    print("  If yes -> that crossing is X*, and a bigger computed edge is trustworthy (fire in")
    print("  real time above it). If ROI stays negative or FALLS as edge rises, our edge estimate")
    print("  is not informative enough — the runtime-hunting premise fails for this market.")
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
    ap.add_argument("--book", default="draftkings")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    run(seasons, book=args.book)


if __name__ == "__main__":
    main()
