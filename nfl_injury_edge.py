"""nfl_injury_edge.py — does adding INJURY features (incl. defense) improve the
model out-of-sample, over base-EPA + QB?

Go/no-go, same honest harness as the QB test: leave-one-season-out OOS margin
RMSE for nested models, then (if it helps) beat-the-close on the injury-affected
slice. Features per game (home perspective, all leakage-safe as-of):
    base_edge  = net_epa_home - net_epa_away
    qb_diff    = QB starter-vs-baseline delta (home - away)
    off_inj    = away offensive-skill EPA-out - home  (production missing)
    def_inj    = away defensive-regulars-out - home   (count)
Models compared OOS: [base], [base+QB], [base+QB+injuries].
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
import nfl_epa
import nfl_qb_asof
import nfl_injury_impact as inj
from nfl_market_scan import _load_closing_odds, _spread_outcome, _profit


def _solve(X, y):
    """OLS coefficients (no intercept) for design matrix X (list of rows) via
    normal equations A b = c with Gaussian elimination. Returns [b0..bk-1]."""
    k = len(X[0]) if X else 0
    A = [[0.0] * k for _ in range(k)]
    c = [0.0] * k
    for row, yi in zip(X, y):
        for i in range(k):
            c[i] += row[i] * yi
            for j in range(k):
                A[i][j] += row[i] * row[j]
    # Gaussian elimination with partial pivoting
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        if abs(A[p][i]) < 1e-12:
            return [0.0] * k
        A[i], A[p] = A[p], A[i]
        c[i], c[p] = c[p], c[i]
        for r in range(k):
            if r == i:
                continue
            f = A[r][i] / A[i][i]
            for j in range(i, k):
                A[r][j] -= f * A[i][j]
            c[r] -= f * c[i]
    return [c[i] / A[i][i] for i in range(k)]


def _rmse(pred, y):
    n = len(y)
    return (sum((p - t) ** 2 for p, t in zip(pred, y)) / n) ** 0.5 if n else 0.0


def build_rows(seasons):
    """[(base, qb, off_inj, def_inj, margin, season, game_id)].

    QB identity uses the PREGAME-KNOWN projected starter (recent-game starter +
    injury/depth), NOT the majority passer in the game being predicted — the latter
    is determined by in-game injuries/benchings, i.e. leakage. [review 2026-09-09]
    """
    rows = []
    for season in seasons:
        plays = nfl_epa.load_plays(season)
        games = {}
        for p in plays:
            if "home_score" in p:
                games[p["game_id"]] = (p["game_date"], p["home_team"],
                                       p["away_team"], p["home_score"] - p["away_score"])
        for gid, (d, h, a, margin) in sorted(games.items(), key=lambda kv: kv[1][0]):
            r = nfl_epa.team_epa(season, as_of_date=d)
            rh, ra = r.get(h), r.get(a)
            if not rh or not ra or rh["off_plays"] <= 0 or ra["off_plays"] <= 0:
                continue
            qb_r, _ = nfl_qb_asof.qb_ratings(season, d)
            qb = (nfl_qb_asof.projected_qb_delta(season, d, h, qb_r) -
                  nfl_qb_asof.projected_qb_delta(season, d, a, qb_r))
            week = int(gid.split("_")[1])
            off_e, _ol, def_e = inj.injury_edges(season, week, h, a)
            rows.append((rh["net_epa"] - ra["net_epa"], qb, off_e, def_e,
                         float(margin), season, gid))
    return rows


MODELS = {
    "base":         lambda r: [r[0]],
    "base+QB":      lambda r: [r[0], r[1]],
    "base+QB+inj":  lambda r: [r[0], r[1], r[2], r[3]],
}


def oos(seasons):
    rows = build_rows(seasons)
    print(f"=== INJURY OOS margin-RMSE (leave-one-season-out, {len(rows)} games) ===")
    n_off = sum(1 for r in rows if abs(r[2]) > 1e-9)
    n_def = sum(1 for r in rows if abs(r[3]) > 1e-9)
    print(f"  games with an offensive-injury edge: {n_off}   defensive-injury edge: {n_def}")
    print(f"  {'model':<14}{'OOS RMSE':>10}   coefficients (fit on all, for scale)")
    tot = {}
    for name, feat in MODELS.items():
        t = 0.0
        for test in seasons:
            tr = [r for r in rows if r[5] != test]
            te = [r for r in rows if r[5] == test]
            if not tr or not te:
                continue
            b = _solve([feat(r) for r in tr], [r[4] for r in tr])
            pred = [sum(bi * xi for bi, xi in zip(b, feat(r))) for r in te]
            t += _rmse(pred, [r[4] for r in te]) * len(te)
        tot[name] = t / len(rows)
        ball = _solve([feat(r) for r in rows], [r[4] for r in rows])
        print(f"  {name:<14}{tot[name]:>10.4f}   {[round(x,3) for x in ball]}")
    print(f"\n  lift base->+QB      : {tot['base+QB']-tot['base']:+.4f}")
    print(f"  lift +QB->+injuries : {tot['base+QB+inj']-tot['base+QB']:+.4f} "
          f"({'INJURIES HELP' if tot['base+QB+inj'] < tot['base+QB'] else 'no help'})")
    # injury-affected games only
    aff = [r for r in rows if abs(r[2]) > 1e-9 or abs(r[3]) > 1e-9]
    if aff:
        print(f"\n  On the {len(aff)} injury-affected games only:")
        for name in ("base+QB", "base+QB+inj"):
            feat = MODELS[name]
            t = 0.0; n = 0
            for test in seasons:
                tr = [r for r in rows if r[5] != test]
                te = [r for r in aff if r[5] == test]
                if not tr or not te:
                    continue
                b = _solve([feat(r) for r in tr], [r[4] for r in tr])
                pred = [sum(bi * xi for bi, xi in zip(b, feat(r))) for r in te]
                t += _rmse(pred, [r[4] for r in te]) * len(te); n += len(te)
            print(f"    {name:<14}{t/n:>10.4f}")
    return rows


def _summ(rows):
    allp = [p for v in rows.values() for p in v]
    n = len(allp)
    if not n:
        return None
    mean = sum(allp) / n
    se = (math.sqrt(sum((p - mean) ** 2 for p in allp) / (n - 1) / n)
          if n > 1 else 0.0)
    wins = sum(1 for p in allp if p > 0)
    seas = " ".join(f"{s}:{(sum(v)/len(v)*100 if v else 0):+.0f}%({len(v)})"
                    for s, v in sorted(rows.items()))
    return {"n": n, "roi": mean, "hit": wins / n, "t": (mean / se if se else 0.0),
            "seas": seas}


def beat_close(seasons, snapshot="closing", book="draftkings", thresholds=(2.0, 3.0)):
    """OOS ATS beat-the-close: bet the side the FULL model favors vs the closing
    spread, on injury-affected games, graded at DK closing price. Compares base /
    base+QB / full so any injury contribution is isolated."""
    rows = build_rows(seasons)
    by_gid = {r[6]: r for r in rows}
    odds = _load_closing_odds([str(s) for s in seasons], book, snapshot)
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    scores = nfl_schedule.team_scores_index([str(s) for s in seasons])
    mkt = {}   # gid -> (sph, spa, hs, as)
    for eid, o in odds.items():
        gid, _ = nfl_schedule.resolve_event(o["home"], o["away"], o["commence_time"], index=idx)
        if gid is None or gid not in by_gid or gid not in scores:
            continue
        if o["spread"].get("home") and o["spread"].get("away"):
            mkt[gid] = (o["spread"]["home"], o["spread"]["away"], *scores[gid])

    # OOS predictions per model
    preds = {name: {} for name in MODELS}
    for test in seasons:
        tr = [r for r in rows if r[5] != test]
        for name, feat in MODELS.items():
            b = _solve([feat(r) for r in tr], [r[4] for r in tr])
            for r in rows:
                if r[5] == test:
                    preds[name][r[6]] = sum(bi * xi for bi, xi in zip(b, feat(r)))

    def bets(pool_gids, name, thr):
        out = defaultdict(list)
        for gid in pool_gids:
            if gid not in mkt:
                continue
            sph, spa, hs, as_ = mkt[gid]
            pred = preds[name].get(gid)
            if pred is None:
                continue
            disagree = pred - (-sph[0])
            if abs(disagree) < thr:
                continue
            pt, px = (sph if disagree > 0 else spa)
            out[by_gid[gid][5]].append(_profit(px, _spread_outcome(disagree > 0, pt, hs, as_)))
        return out

    aff = [r[6] for r in rows if abs(r[2]) > 1e-9 or abs(r[3]) > 1e-9]
    defaff = [r[6] for r in rows if abs(r[3]) > 1e-9]
    print("\n" + "=" * 92)
    print(f"  INJURY beat-the-close (OOS, DK {snapshot}) — ATS at closing price")
    print("=" * 92)
    for label, pool in (("injury-affected games", aff), ("DEFENSIVE-injury games", defaff)):
        print(f"\n  ── {label} ({len(pool)}) ──")
        print(f"    {'thr':<5}{'model':<14}{'n':>5}  {'ROI':>8}  {'hit':>6}  {'t':>6}   per-season")
        for thr in thresholds:
            for name in ("base", "base+QB", "base+QB+inj"):
                st = _summ(bets(pool, name, thr))
                if st:
                    print(f"    {thr:<5}{name:<14}{st['n']:>5}  {st['roi']*100:+7.2f}%  "
                          f"{st['hit']*100:5.1f}%  {st['t']:+5.2f}   [{st['seas']}]")
    print("\n" + "=" * 92)
    print("  READ: edge = +ROI replicating per season, full model beating base+QB AND the vig,")
    print("  concentrated in the injury slice. Else the market prices injuries too -> ABSTAIN.")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--beat-close", action="store_true",
                    help="also run the OOS ATS beat-the-close test")
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h", "early_12h"])
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    oos(seasons)
    if args.beat_close:
        beat_close(seasons, snapshot=args.snapshot)


if __name__ == "__main__":
    main()
