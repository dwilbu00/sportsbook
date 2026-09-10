"""nfl_props_clv.py — does the EARLY prop line beat the CLOSE? (the timing-edge test)

The soft-line thesis: books post props early (soft), the market sharpens by kickoff. If
our (now-accurate) model can spot which early numbers are stale, we'd bet the early price
and the line would move TOWARD us by close — positive CLV, the gold-standard predictor of
long-run profit — WITHOUT needing to out-predict the close itself.

We can test this for FREE: the mirror already has two snapshots per prop for 2023-2025,
`early_4h` (~4h pre-kickoff) and `closing`. For each prop where both exist:
  * project P(over) with the FROZEN model (components trained on 2012-2022, swept config);
  * pick the side our model likes at the EARLY line;
  * measure whether the line/price moved our way by close (CLV) and whether the side won.

Reported per prop + pooled, bucketed by model edge:
  CLV-pts   : line movement in our favor (points; the number we locked vs the close).
  beat%     : share of model-picked early bets where the line moved our way (CLV-pts>0).
  probCLV   : on the SAME-line subset, de-vigged prob move toward our side (clean price CLV).
  mkt→model : does the line move toward the model's side? (avg CLV-pts by model-liked side).
  ROIearly  : outcome ROI at the EARLY price; real−impl = realized minus early-implied.

Positive CLV (esp. in the higher-edge buckets, replicated) ⇒ a real capturable timing
edge ⇒ a cheap scheduled logger is justified. Flat/negative ⇒ the early line is already
efficient; drop the logger. Diagnostic only — writes nothing, spends no API credits.
"""
import argparse

import nfl_schedule
import nfl_props_scan as scan
from odds_client import american_to_decimal, american_to_implied_prob, devig_two_way
import nfl_props_accuracy as acc


def _fair_over(over, under):
    if not (over and under):
        return None
    try:
        return devig_two_way(american_to_implied_prob(over[1]),
                             american_to_implied_prob(under[1]))[0]
    except Exception:
        return None


def run(prop, cfg, train_seasons, test_seasons, book):
    # frozen model: components on 2012-2022 with this prop's swept config
    hl, mp, k = acc.SWEPT.get(prop, (4, 3, 12))
    acc.HALF_LIFE, acc.MIN_PRIOR, acc.SHRINK_K = hl, mp, k
    train = acc._obs_from_series(acc._series(train_seasons), cfg)
    if len(train) < 200:
        print(f"  {prop:<24} (train thin)")
        return
    comp = acc.fit_components(train, cfg)
    feat = {(o["name"], o["season"], o["week"]): o
            for o in acc._obs_from_series(acc._series(test_seasons), cfg)}

    early = scan._load_props(test_seasons, "early_4h", book)
    close = scan._load_props(test_seasons, "closing", book)
    idx = nfl_schedule.game_index([str(s) for s in test_seasons])

    rows = []
    for key, de in early.items():
        if key[2] != prop:
            continue
        dc = close.get(key)
        if not dc:
            continue
        gid, _ = nfl_schedule.resolve_event(de["home"], de["away"], de["commence"], index=idx)
        if gid is None:
            continue
        ps = gid.split("_"); season, week = ps[0], int(ps[1])
        o = feat.get((scan._norm(key[1]), season, week))
        if o is None:
            continue
        le = de.get("OVER", de.get("UNDER", (None, None)))[0]
        lc = dc.get("OVER", dc.get("UNDER", (None, None)))[0]
        if le is None or lc is None:
            continue
        fe = _fair_over(de.get("OVER"), de.get("UNDER"))
        fc = _fair_over(dc.get("OVER"), dc.get("UNDER"))
        if fe is None or fc is None:
            continue
        if abs(o["actual"] - le) < 1e-9:
            continue
        p_e = acc.p_over(o, le, cfg, comp)
        over_side = p_e >= fe                          # model likes OVER at the early line
        # CLV in points (favorable line movement for our side)
        clv_pts = (lc - le) if over_side else (le - lc)
        # clean price CLV on same-line subset
        prob_clv = None
        if abs(lc - le) < 1e-9:
            our_e = fe if over_side else 1 - fe
            our_c = fc if over_side else 1 - fc
            prob_clv = our_c - our_e
        # outcome at the early price
        win = (o["actual"] > le) if over_side else (o["actual"] < le)
        px = (de.get("OVER") if over_side else de.get("UNDER"))[1]
        roi = (american_to_decimal(px) - 1.0) if win else -1.0
        realized = 1 if win else 0
        implied = (fe if over_side else 1 - fe)
        rows.append({"season": season, "edge": abs(p_e - fe), "over": over_side,
                     "clv_pts": clv_pts, "prob_clv": prob_clv, "roi": roi,
                     "realized": realized, "implied": implied,
                     "line_move_signed": lc - le, "model_over": p_e >= fe})
    if len(rows) < 100:
        print(f"  {prop:<24} (only {len(rows)} matched early+close obs)")
        return

    def rep(sub, label):
        n = len(sub)
        if not n:
            print(f"      {label:<16} (empty)")
            return
        clv = sum(r["clv_pts"] for r in sub) / n
        beat = sum(1 for r in sub if r["clv_pts"] > 1e-9) / n * 100
        pcl = [r["prob_clv"] for r in sub if r["prob_clv"] is not None]
        pcl_m = (sum(pcl) / len(pcl) * 100) if pcl else float("nan")
        roi = sum(r["roi"] for r in sub) / n * 100
        ri = (sum(r["realized"] for r in sub) - sum(r["implied"] for r in sub)) / n * 100
        print(f"      {label:<16} n={n:>5}  CLV-pts={clv:+.3f}  beat%={beat:4.1f}  "
              f"probCLV={pcl_m:+.2f}%  ROIearly={roi:+6.2f}%  real-impl={ri:+5.2f}%")

    print(f"  {prop}  ({len(rows)} matched early+close obs)")
    rep(rows, "ALL")
    for lo, hi, lbl in [(0.0, 0.03, "|edge| 0-3%"), (0.03, 0.06, "|edge| 3-6%"),
                        (0.06, 1.0, "|edge| >=6%")]:
        rep([r for r in rows if lo <= r["edge"] < hi], lbl)
    # does the line move toward the model's side?
    ov = [r["line_move_signed"] for r in rows if r["model_over"]]
    un = [r["line_move_signed"] for r in rows if not r["model_over"]]
    if ov and un:
        print(f"      mkt→model: model-OVER avg line move={sum(ov)/len(ov):+.3f}  "
              f"model-UNDER avg line move={sum(un)/len(un):+.3f}  "
              f"(over>0 & under<0 ⇒ line follows model)")


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
    ap.add_argument("--book", default="draftkings")
    ap.add_argument("--prop", default="all")
    args = ap.parse_args()
    train_seasons = [s.strip() for s in args.train.split(",") if s.strip()]
    test_seasons = [s.strip() for s in args.test.split(",") if s.strip()]
    props = list(acc.PROPS) if args.prop == "all" else [args.prop]
    print("=" * 104)
    print(f"  NFL PROPS CLV — early_4h vs closing ({args.book}, test {test_seasons}); "
          f"frozen model trained {train_seasons[0]}-{train_seasons[-1]}")
    print("  Positive CLV in higher-edge buckets, replicated ⇒ capturable timing edge.")
    print("=" * 104)
    for prop in props:
        run(prop, acc.PROPS[prop], train_seasons, test_seasons, args.book)
    print("=" * 104)


if __name__ == "__main__":
    main()
