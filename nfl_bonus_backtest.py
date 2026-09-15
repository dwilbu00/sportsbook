"""nfl_bonus_backtest.py — does a profit BOOST actually turn a profit on REAL outcomes?

The thesis: we can't beat the book on straights, but a promo boost pays on the PAYOUT, so it
flips high-probability legs to +EV — IF our leg probabilities are genuinely calibrated. This
backtest is the proof-or-refutation on real historical outcomes:

  * Legs come ONLY from the trustworthy props (count props above their validated opportunity
    thresholds — the same abstain rule as the fair-value tool). Yardage/pass_tds excluded:
    over-confident P => fake +EV. This is where all the calibration work pays off.
  * Each leg = our calibrated P on the FAVORITE side, that side's American price at the bonus's
    OWN book (DK bonus -> DK odds, FD bonus -> FD odds), and whether it actually won.
  * We build CROSS-GAME parlays (distinct games => legs independent => the joint = product of
    marginals is honest; SGP correlation is a separate model, not assumed here).
  * For each bonus we bet every qualifying +EV ticket flat $1 and grade vs actuals.

The money question per bonus/size/season:
    realized ROI  vs  predicted boosted EV        (do they match? => calibration holds)
    realized hit-rate  vs  mean joint_P           (are the parlays hitting as often as our P says?)
If realized ROI is +, consistent across seasons, and tracks the prediction => the mechanism is
real. Reads the local ladder extract (closing lines) + actuals. Diagnostic — no spend.

RESULT (2023-25, DK+FD): the boost is a REAL +EV edge — but ONLY with MARKET de-vigged P.
Our MODEL P is over-confident in the high-P tail we bet (calls 80% -> wins 61%; ECE 0.063),
so --prob model inflates predEV to fiction and the 25% 3-leg drifts negative. With --prob
market (ECE 0.008-0.016, calibrated) predEV is honest and POSITIVE every season: 50% parlay
~+20% predEV / ~+30% realized (worst season +18%), 25% any ~+5%. So the leg probability must
be the book's de-vigged price; the MODEL's only job here is the opportunity/trustworthiness
ABSTAIN (field legs only where the market number is meaningful). Default --prob=market.
"""
import argparse
from collections import defaultdict
from itertools import combinations

import nfl_schedule
import nfl_props_scan as scan
import nfl_props_accuracy as acc
import nfl_ladder_clv as clv
import bonus as bonuslib
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way

# Only the props validated as market-trustworthy (nfl_opportunity_threshold.py), with their
# opportunity abstain cutoffs. Legs below the cutoff, or from any other prop, are NOT eligible.
TRUSTWORTHY = {
    "player_receptions": 6.9,       # expected targets
    "player_rush_attempts": 8.5,    # expected carries
    "player_pass_attempts": 23.4,   # expected attempts
}
TOP_N = 14          # cap qualifying legs per (book,season,week) before enumerating combos
SEASONS = ("2023", "2024", "2025")


def _opp(o):
    return o.get("exp_vol", o.get("mean_base", 0.0))


def collect_legs(train_seasons, test_seasons, books):
    """{(book, season, week): [leg, ...]} where a leg is the FAVORITE side of a trustworthy,
    high-opportunity count prop: our calibrated P, that side's price at that book, won/lost."""
    pool = defaultdict(list)
    idx = nfl_schedule.game_index([str(s) for s in test_seasons])
    for prop, tmin in TRUSTWORTHY.items():
        cfg = acc.PROPS[prop]
        hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
        acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
        train = acc._obs_from_series(acc._series(train_seasons), cfg)
        if len(train) < 200:
            continue
        comp = acc.fit_components(train, cfg)
        feat = {(o["name"], o["season"], o["week"]): o
                for o in acc._obs_from_series(acc._series(test_seasons), cfg)}
        lines, meta = clv._load_local(prop, test_seasons)
        for (eid, player), bybook in lines.items():
            m = meta[(eid, player)]
            gid, _ = nfl_schedule.resolve_event(m["home"], m["away"], m["commence"], index=idx)
            if gid is None:
                continue
            ps = gid.split("_")
            week = int(ps[1])
            o = feat.get((scan._norm(player), ps[0], week))
            if o is None or _opp(o) < tmin:
                continue
            for book in books:
                d = bybook.get(book, {}).get("closing")
                if not d:
                    continue
                over_q, under_q = d.get("OVER"), d.get("UNDER")
                if not (over_q and under_q):
                    continue
                line = over_q[0]
                if line is None or abs(o["actual"] - line) < 1e-9:
                    continue
                p_o = acc.p_over(o, line, cfg, comp)
                try:
                    fair_o = devig_two_way(american_to_implied_prob(over_q[1]),
                                           american_to_implied_prob(under_q[1]))[0]
                except Exception:
                    continue
                over = p_o >= 0.5                      # bet the favorite side (our model's pick)
                price = (over_q if over else under_q)[1]
                won = (o["actual"] > line) if over else (o["actual"] < line)
                pool[(book, ps[0], week)].append({
                    "gid": gid,
                    "P_model": p_o if over else (1.0 - p_o),
                    "P_mkt": fair_o if over else (1.0 - fair_o),
                    "odds": price, "won": bool(won),
                    "prop": prop.replace("player_", ""), "player": player, "line": line})
    return pool


def _sizes_for(bonus):
    """Ticket sizes to test for a bonus, honoring bet_type + min_legs."""
    lo = max(1, bonus.min_legs)
    if bonus.bet_type == "single":
        return [1]
    if bonus.bet_type == "any":
        return sorted(set([max(1, lo)] + [1, 2, 3]))
    return [s for s in (2, 3) if s >= lo] or [lo]   # parlay/sgp


def backtest_bonus(pool, bonus, sizes, pkey):
    """Bet every qualifying +EV ticket flat $1; return per-size aggregates + per-season.
    pkey selects the leg probability fed to the engine ('P_model' or 'P_mkt')."""
    book = bonus.book
    out = {}
    for K in sizes:
        rows = []                                       # one per bet ticket
        for (bk, season, week), legs in pool.items():
            if bk != book:
                continue
            elig = [lg for lg in legs
                    if american_to_decimal(lg["odds"]) >= bonuslib.american_to_dec(bonus.min_odds_leg) - 1e-9]
            elig.sort(key=lambda x: -x[pkey])
            elig = elig[:TOP_N]
            for combo in combinations(elig, K):
                if len({lg["gid"] for lg in combo}) != K:   # cross-game only
                    continue
                r = bonuslib.evaluate([(lg[pkey], lg["odds"]) for lg in combo], bonus)
                if not (r["qualifies"] and r["boosted_ev_pct"] > 0):
                    continue
                won_all = all(lg["won"] for lg in combo)
                profit = (r["combined_dec"] - 1.0) * (1.0 + bonus.boost_pct) if won_all else -1.0
                rows.append({"season": season, "pred_ev": r["boosted_ev_pct"] / 100.0,
                             "joint_P": r["joint_P"], "won": won_all, "profit": profit})
        out[K] = rows
    return out


def leg_reliability(pool, book, pkey):
    """Reliability of the chosen favorite-side probability on ALL eligible legs (unselected):
    binned predicted-P vs realized win-rate. Distinguishes base miscalibration from the
    +EV optimizer's-curse selection effect seen in the ticket results."""
    legs = [lg for (bk, _s, _w), v in pool.items() if bk == book for lg in v]
    bins = defaultdict(lambda: [0.0, 0, 0])   # [sum_P, wins, n]
    for lg in legs:
        b = min(9, int(lg[pkey] * 10))
        bins[b][0] += lg[pkey]; bins[b][1] += int(lg["won"]); bins[b][2] += 1
    print(f"\n  [{book}] leg reliability, prob={pkey} (favorite side, ALL eligible legs)")
    print(f"    {'predP':>10} {'n':>6} {'meanPred%':>10} {'realWin%':>9}  gap")
    ece = 0.0
    for b in sorted(bins):
        sp, w, n = bins[b]
        if n < 25:
            continue
        mp, rw = sp / n * 100, w / n * 100
        ece += abs(mp - rw) / 100 * n
        print(f"    {b/10:.1f}-{b/10+0.1:.1f}   {n:>6} {mp:>9.1f}% {rw:>8.1f}%  {mp-rw:>+6.1f}")
    print(f"    ECE={ece/max(1,len(legs)):.4f}  (low => base P calibrated; high tail gap => "
          "over-confident where we bet)")


def _agg(rows):
    n = len(rows)
    if n == 0:
        return None
    return {"n": n,
            "roi": sum(r["profit"] for r in rows) / n * 100,
            "pred_ev": sum(r["pred_ev"] for r in rows) / n * 100,
            "hit": sum(r["won"] for r in rows) / n * 100,
            "exp_hit": sum(r["joint_P"] for r in rows) / n * 100}


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
    ap.add_argument("--book", default="both", choices=["draftkings", "fanduel", "both"],
                    help="which book's odds to run the bonuses against")
    ap.add_argument("--prob", default="market", choices=["model", "market"],
                    help="leg probability: our model, or the book's de-vigged (calibrated) price")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    books = ["draftkings", "fanduel"] if args.book == "both" else [args.book]
    pkey = "P_model" if args.prob == "model" else "P_mkt"

    print("=" * 100)
    print(f"  NFL BONUS BACKTEST — realized ROI of boosted CROSS-GAME parlays, {test_seasons}")
    print("  Legs: trustworthy count props above opportunity cutoff, favorite side, calibrated P.")
    print(f"  Cross-game only (independent). Flat $1/ticket. Leg probability = {args.prob.upper()}.")
    print("=" * 100)
    pool = collect_legs(train_seasons, test_seasons, books)
    tot = sum(len(v) for v in pool.values())
    print(f"  eligible legs: {tot:,} across {len({(k[0],k[1],k[2]) for k in pool})} book-week slates")
    for book in books:
        leg_reliability(pool, book, pkey)

    for base in bonuslib.LIVE_BONUSES:
        for book in books:
            bonus = bonuslib.Bonus(**{**base.__dict__, "book": book})
            sizes = _sizes_for(bonus)
            res = backtest_bonus(pool, bonus, sizes, pkey)
            print(f"\n  [{book}] {base.label}")
            print(f"    {'size':>4} {'bets':>6} {'hit%':>7} {'expHit%':>8} "
                  f"{'ROI%':>8} {'predEV%':>8}   per-season ROI%")
            for K in sizes:
                a = _agg(res[K])
                if a is None:
                    print(f"    {K:>4} {'--':>6}   (no qualifying +EV tickets)")
                    continue
                cells = []
                for s in test_seasons:
                    sa = _agg([r for r in res[K] if r["season"] == s])
                    cells.append(f"{s}:{sa['roi']:+.0f}%(n{sa['n']})" if sa else f"{s}:--")
                kind = "single" if K == 1 else f"{K}-leg"
                print(f"    {kind:>4} {a['n']:>6} {a['hit']:>6.1f}% {a['exp_hit']:>7.1f}% "
                      f"{a['roi']:>+7.1f}% {a['pred_ev']:>+7.1f}%   {'  '.join(cells)}")
    print("\n  READ: ROI% ~= predEV% and hit% ~= expHit% => our calibration holds inside the boost,")
    print("  and +ROI that repeats every season => the bonus is a REAL edge (not one-year noise).")
    print("  Overlapping tickets share legs => treat per-season n as tickets, not independent trials.")
    print("=" * 100)


if __name__ == "__main__":
    main()
