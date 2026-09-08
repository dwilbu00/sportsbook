"""nfl_qb_edge.py — does the QB-aware EPA model BEAT THE CLOSE?

The OOS margin test (backtest_nfl_epa --oos-qb) showed the starting-QB adjustment
lowers out-of-sample margin RMSE in every fold. But the market knows the starter
too, so a prediction gain only becomes MONEY if the market MISPRICES QB changes.
This grades that directly: fit the model out-of-sample (leave-one-season-out),
compare its predicted margin to the CLOSING spread, bet the side it favors when
the disagreement clears a threshold, and grade ATS at the real DK closing price.

The decisive slice is the QB-CHANGE games (starter != team's baseline QB): there
the QB model's margin differs from a pure team-EPA model, so if the market under-
adjusts to backups/injuries, the edge shows up there and NOT in the full slate.

Honest bar (MLB + props scars): OOS weights (never fit on the graded season),
real closing prices (vig in), per-season replication, t-stats. Diagnostic only —
nothing written. Bets graded at DraftKings; completed games only.
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
import nfl_epa
import nfl_qb_asof
from nfl_market_scan import _load_closing_odds, _spread_outcome, _ml_outcome, _profit
from backtest_nfl_epa import _fit1, _fit2
from r2_sharp import fair_two_way
from odds_client import american_to_decimal

SPORT = "americanfootball_nfl"
NFL_MARGIN_SIGMA = 13.5   # ~game-margin RMSE; maps model margin -> win prob


def build_games(seasons):
    """{game_id: {season,date,home,away,margin,base_edge,qb_diff}} — leakage-safe
    (EPA + QB ratings as-of the game date; starter identity is public pre-game)."""
    out = {}
    for season in seasons:
        plays = nfl_epa.load_plays(season)
        starters = nfl_qb_asof.game_starters(season)
        games = {}
        for p in plays:
            if "home_score" in p:
                games[p["game_id"]] = (p["game_date"], p["home_team"],
                                       p["away_team"],
                                       p["home_score"] - p["away_score"])
        for gid, (d, h, a, margin) in sorted(games.items(), key=lambda kv: kv[1][0]):
            ratings = nfl_epa.team_epa(season, as_of_date=d)
            rh, ra = ratings.get(h), ratings.get(a)
            if not rh or not ra or rh["off_plays"] <= 0 or ra["off_plays"] <= 0:
                continue
            qb_r, _lg = nfl_qb_asof.qb_ratings(season, d)
            gs = starters.get(gid, {})
            dh = nfl_qb_asof.qb_edge_delta(season, d, h, gs.get(h), qb_r)
            da = nfl_qb_asof.qb_edge_delta(season, d, a, gs.get(a), qb_r)
            out[gid] = {"season": season, "date": d, "home": h, "away": a,
                        "margin": float(margin), "base_edge": rh["net_epa"] - ra["net_epa"],
                        "qb_diff": dh - da}
    return out


def _oos_weights(games, test_season):
    """Fit (w_epa alone) and (w_epa,w_qb) on all seasons EXCEPT test_season."""
    tr = [g for g in games.values() if g["season"] != test_season]
    wb = _fit1([g["base_edge"] for g in tr], [g["margin"] for g in tr])
    wa, wq = _fit2([g["base_edge"] for g in tr], [g["qb_diff"] for g in tr],
                   [g["margin"] for g in tr])
    return wb, (wa, wq)


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def grade(seasons, snapshot="closing", book="draftkings", thresholds=(1.0, 2.0, 3.0)):
    games = build_games(seasons)
    odds = _load_closing_odds([str(s) for s in seasons], book, snapshot)
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    scores = nfl_schedule.team_scores_index([str(s) for s in seasons])

    # attach closing spread + ml to each game via the spine join
    joined = 0
    for eid, o in odds.items():
        gid, _ = nfl_schedule.resolve_event(o["home"], o["away"],
                                            o["commence_time"], index=idx)
        if gid is None or gid not in games:
            continue
        sc = scores.get(gid)
        if sc is None:
            continue
        g = games[gid]
        g["hs"], g["as"] = sc
        g["sph"] = o["spread"].get("home")   # (point, price) home line
        g["spa"] = o["spread"].get("away")
        g["mlh"], g["mla"] = o["ml"].get("home"), o["ml"].get("away")
        joined += 1

    gl = [g for g in games.values() if "hs" in g and g.get("sph") and g.get("spa")]
    print("=" * 96)
    print(f"  NFL QB-model BEAT-THE-CLOSE (OOS leave-one-season-out, DK {snapshot}, "
          f"{sorted({g['season'] for g in gl})})")
    print(f"  games with EPA+odds+score = {len(gl)}   (of {len(games)} modeled)")
    print("  ATS: bet the side the model favors when |model_margin - market_margin| >= thr; "
          "graded at DK closing price.")
    print("=" * 96)

    # precompute OOS predictions per game
    for test in seasons:
        wb, (wa, wq) = _oos_weights(games, test)
        for g in gl:
            if g["season"] != test:
                continue
            g["pred_base"] = wb * g["base_edge"]
            g["pred_qb"] = wa * g["base_edge"] + wq * g["qb_diff"]

    def _ats_bets(pool, pred_key, thr):
        """Return season->[profit] for ATS bets off pred_key at threshold thr."""
        rows = defaultdict(list)
        for g in pool:
            pred = g.get(pred_key)
            if pred is None:
                continue
            # market margin from the HOME spread point: home line -3 => market
            # expects home to win by 3 => market_margin_home = -point.
            mkt_margin = -g["sph"][0]
            disagree = pred - mkt_margin
            if abs(disagree) < thr:
                continue
            if disagree > 0:        # model likes home more than the line -> bet home
                pt, px = g["sph"]
                out = _spread_outcome(True, pt, g["hs"], g["as"])
            else:                   # bet away
                pt, px = g["spa"]
                out = _spread_outcome(False, pt, g["hs"], g["as"])
            rows[g["season"]].append(_profit(px, out))
        return rows

    def _summ(rows):
        allp = [p for v in rows.values() for p in v]
        n = len(allp)
        if not n:
            return None
        mean = sum(allp) / n
        if n > 1:
            var = sum((p - mean) ** 2 for p in allp) / (n - 1)
            se = math.sqrt(var / n) if var > 0 else 0.0
        else:
            se = 0.0
        wins = sum(1 for p in allp if p > 0)
        seas = " ".join(f"{s}:{(sum(v)/len(v)*100 if v else 0):+.0f}%({len(v)})"
                        for s, v in sorted(rows.items()))
        return {"n": n, "roi": mean, "hit": wins / n,
                "t": (mean / se if se else 0.0), "seas": seas}

    chg = [g for g in gl if abs(g["qb_diff"]) > 1e-9]
    print(f"\n  QB-change games in slate: {len(chg)} of {len(gl)}")
    for label, pool in (("ALL games", gl), ("QB-CHANGE games only", chg)):
        print(f"\n  ── {label} ──")
        print(f"    {'thr':<5}{'model':<7}{'n':>5}  {'ROI':>8}  {'hit':>6}  {'t':>6}   per-season")
        for thr in thresholds:
            for mk, mlabel in (("pred_base", "base"), ("pred_qb", "+QB")):
                st = _summ(_ats_bets(pool, mk, thr))
                if not st:
                    print(f"    {thr:<5}{mlabel:<7}{'0':>5}   (no bets)")
                    continue
                print(f"    {thr:<5}{mlabel:<7}{st['n']:>5}  {st['roi']*100:+7.2f}%  "
                      f"{st['hit']*100:5.1f}%  {st['t']:+5.2f}   [{st['seas']}]")

    # ── moneyline on QB-change games (model prob vs devigged close) ──
    print(f"\n  ── MONEYLINE, QB-CHANGE games (model prob vs devigged close) ──")
    print(f"    {'edge>=':<7}{'n':>5}  {'ROI':>8}  {'hit':>6}  {'t':>6}   per-season")
    for edge_thr in (0.03, 0.05, 0.08):
        rows = defaultdict(list)
        for g in chg:
            if g.get("mlh") is None or g.get("mla") is None or g.get("pred_qb") is None:
                continue
            fh, fa = fair_two_way(g["mlh"], g["mla"])
            if fh is None:
                continue
            p_home = _norm_cdf(g["pred_qb"] / NFL_MARGIN_SIGMA)
            # bet the side where model prob exceeds devigged fair prob by edge_thr
            if p_home - fh >= edge_thr:
                out = _ml_outcome(True, g["hs"], g["as"]); px = g["mlh"]
            elif (1 - p_home) - fa >= edge_thr:
                out = _ml_outcome(False, g["hs"], g["as"]); px = g["mla"]
            else:
                continue
            rows[g["season"]].append(_profit(px, out))
        st = _summ(rows)
        if not st:
            print(f"    {edge_thr:<7}{'0':>5}   (no bets)")
        else:
            print(f"    {edge_thr:<7}{st['n']:>5}  {st['roi']*100:+7.2f}%  "
                  f"{st['hit']*100:5.1f}%  {st['t']:+5.2f}   [{st['seas']}]")

    print("\n" + "=" * 96)
    print("  READ: an edge = +ROI that REPLICATES per season, concentrated in the QB-change")
    print("  slice, with the +QB model beating both the base model and the vig. Otherwise the")
    print("  market already prices the QB and the honest call is ABSTAIN.")
    print("=" * 96)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h", "early_12h"])
    ap.add_argument("--book", default="draftkings")
    ap.add_argument("--thresholds", default="1,2,3",
                    help="ATS disagreement thresholds in points")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    thr = tuple(float(x) for x in args.thresholds.split(",") if x.strip())
    grade(seasons, snapshot=args.snapshot, book=args.book, thresholds=thr)


if __name__ == "__main__":
    main()
