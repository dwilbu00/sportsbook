"""nba_opportunity_threshold.py — is the de-vigged NBA prop price a trustworthy bonus leg, and
above what playing-time level? (NBA analog of mlb/nfl_opportunity_threshold; owner runs it.)

The bonus engine prices every leg at the BOOK's de-vigged probability, so a leg is only as good
as that price's calibration — and the bonus bets FAVORITES (the high-P tail), where an average
calibration check can hide a miscalibrated tail (the NFL lesson). For each book (DK, FD) and
prop this reports, on real historical lines graded against warehouse actuals:

  (a) over-side calibration          |mean implied over - realized over| (the MLB test)
  (b) favorite-side reliability      binned by the favorite's de-vigged P, + ECE
  (c) (b) by EXPECTED MINUTES        recency-weighted mean of the player's prior minutes
  (d) the minutes GATE               lowest expected-minutes T whose favorite-side calibration
                                     gap <= tol with n >= 150 on DISCOVERY seasons, then
                                     confirmed on HELD-OUT (gap <= 1.5 x tol)

Every PLAYED test game counts (a bet settles on the box score once the player checks in; only a
DNP voids) — unlike projection fitting, which drops <10-minute games. Opportunity comes from
strictly-earlier same-season games (leakage-safe). Read-only; 0 DTU via the mirrors; no credits.

Run (owner):
  $env:PYTHONIOENCODING="utf-8"
  python nba_opportunity_threshold.py                       # DK + FD, all props
  python nba_opportunity_threshold.py --book draftkings --prop player_points
"""
import argparse
import math
from collections import defaultdict

import book_line_calibration as blc
from odds_client import american_to_implied_prob, devig_two_way

NBA_PROPS = ["player_points", "player_rebounds", "player_assists", "player_threes",
             "player_steals", "player_blocks", "player_turnovers", "player_points_rebounds",
             "player_points_assists", "player_points_rebounds_assists"]
MIN_HALF_LIFE = 5.0                                     # games, for expected minutes
P_BANDS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))
MIN_BUCKETS = ((0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 99))
MIN_N = 150


def _fair_over(over_price, under_price):
    if over_price is None or under_price is None:
        return None
    try:
        return devig_two_way(american_to_implied_prob(over_price),
                             american_to_implied_prob(under_price))[0]
    except Exception:
        return None


def expected_minutes(prior_games, half_life=MIN_HALF_LIFE):
    """Recency-weighted mean of prior played minutes (newest-first rows), or None."""
    mins = [float(g["MIN"]) for g in (prior_games or []) if g.get("MIN") is not None
            and float(g["MIN"]) > 0]
    if not mins:
        return None
    w = [math.exp(-math.log(2) / half_life * i) for i in range(len(mins))]
    return sum(m * x for m, x in zip(mins, w)) / sum(w)


def _season(game_date):
    import nba_stats_source as src
    return src.end_year_for_date(str(game_date)[:10])


def collect(book, props, snapshot):
    """[record] for each joinable real NBA prop line at ``book`` (pushes + DNPs dropped)."""
    lines, _np, _nl = blc.harvest_real_line_book_lines("basketball_nba", props,
                                                       snapshot=snapshot, bookmaker=book)
    print(f"  [{book}] harvested {len(lines):,} real lines; grading (every played game)…")
    enriched = blc._nba_grade_book_lines(lines, min_test_minutes=0.0) or []
    out = []
    for r in enriched:
        prop = r.get("prop_key")
        if prop not in props:
            continue
        line, actual = r.get("line"), r.get("actual")
        if line is None or actual is None or abs(actual - line) < 1e-9:
            continue                                    # push / missing
        fair = _fair_over(r.get("over_price"), r.get("under_price"))
        em = expected_minutes(r.get("prior_games"))
        if fair is None or em is None:
            continue
        y = 1 if actual > line else 0
        fav_over = fair >= 0.5
        out.append({"season": _season(r.get("game_date")), "prop": prop, "opp": em,
                    "fair": fair, "y": y,
                    "fav_p": fair if fav_over else 1.0 - fair,
                    "fav_won": y if fav_over else 1 - y})
    return out


def _over_cal(sub):
    n = len(sub)
    if not n:
        return None
    return n, abs(sum(r["fair"] for r in sub) / n - sum(r["y"] for r in sub) / n)


def _fav_cal(sub):
    """(n, mean fav P, fav win rate, |gap|, ECE over P_BANDS) for favorite-side legs."""
    n = len(sub)
    if not n:
        return None
    mp = sum(r["fav_p"] for r in sub) / n
    mw = sum(r["fav_won"] for r in sub) / n
    ece = 0.0
    for lo, hi in P_BANDS:
        b = [r for r in sub if lo <= r["fav_p"] < hi]
        if b:
            ece += len(b) / n * abs(sum(r["fav_p"] for r in b) / len(b)
                                    - sum(r["fav_won"] for r in b) / len(b))
    return n, mp, mw, abs(mp - mw), ece


def _gate(disc, held, tol):
    """Lowest expected-minutes T (discovery quantile grid) with favorite gap <= tol, n >= MIN_N;
    returns (T, held-out (n, gap)) or (None, None)."""
    if len(disc) < 2 * MIN_N:
        return None, None
    opps = sorted(r["opp"] for r in disc)
    grid = sorted(set(round(opps[int(q * (len(opps) - 1))], 1) for q in [i / 20 for i in range(20)]))
    for T in grid:
        c = _fav_cal([r for r in disc if r["opp"] >= T])
        if c and c[0] >= MIN_N and c[3] <= tol:
            h = _fav_cal([r for r in held if r["opp"] >= T])
            return T, ((h[0], h[3]) if h else None)
    return None, None


def report(book, recs, props, discovery, held_out, tol):
    print("=" * 104)
    print(f"  NBA OPPORTUNITY / LEG-TRUST — {book.upper()}  (discovery {discovery}, held-out "
          f"{held_out}, tol {tol})")
    print("=" * 104)
    print(f"  {'prop':<32} {'n':>7} {'overGap':>7} | {'fav n':>7} {'favP':>6} {'won':>6} "
          f"{'gap':>6} {'ECE':>6} | {'GATE min':>8} {'held n':>7} {'held gap':>8}")
    by_prop = defaultdict(list)
    for r in recs:
        by_prop[r["prop"]].append(r)
    for prop in props + ["ALL"]:
        sub = recs if prop == "ALL" else by_prop.get(prop, [])
        disc = [r for r in sub if r["season"] in discovery]
        held = [r for r in sub if r["season"] in held_out]
        oc, fc = _over_cal(sub), _fav_cal(sub)
        if not oc:
            continue
        T, h = _gate(disc, held, tol)
        gate = (f"{T:>8.1f} {h[0]:>7,} {h[1]:>8.3f}" + ("  ok" if h[1] <= 1.5 * tol else "  FAIL")
                if (T is not None and h) else f"{'none':>8} {'':>7} {'':>8}")
        print(f"  {prop:<32} {oc[0]:>7,} {oc[1]:>7.3f} | {fc[0]:>7,} {fc[1]:>6.3f} "
              f"{fc[2]:>6.3f} {fc[3]:>6.3f} {fc[4]:>6.3f} | {gate}")
    # favorite-side reliability by P band and by expected minutes (all props pooled)
    print("\n  favorite-side reliability, all props (predicted -> realized):")
    print(f"  {'P band':<10} {'n':>7} {'pred':>6} {'won':>6} {'gap':>7}")
    for lo, hi in P_BANDS:
        b = [r for r in recs if lo <= r["fav_p"] < hi]
        if b:
            c = _fav_cal(b)
            print(f"  {f'{lo:.1f}-{min(hi, 1):.1f}':<10} {c[0]:>7,} {c[1]:>6.3f} {c[2]:>6.3f} "
                  f"{c[2] - c[1]:>+7.3f}")
    print(f"\n  by EXPECTED MINUTES (all props): {'n':>7} {'favP':>6} {'won':>6} {'gap':>7} "
          f"{'ECE':>6} {'overGap':>8}")
    for lo, hi in MIN_BUCKETS:
        b = [r for r in recs if lo <= r["opp"] < hi]
        if b:
            c, o = _fav_cal(b), _over_cal(b)
            print(f"  {f'{lo}-{hi} min':<32} {c[0]:>7,} {c[1]:>6.3f} {c[2]:>6.3f} "
                  f"{c[2] - c[1]:>+7.3f} {c[4]:>6.3f} {o[1]:>8.3f}")
    print()


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
    ap.add_argument("--book", default="both", choices=("both", "draftkings", "fanduel"))
    ap.add_argument("--prop", default="all")
    ap.add_argument("--discovery", default="2023,2024,2025")
    ap.add_argument("--val", default="2026")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="max favorite-side calibration gap |mean P - win rate| for the gate")
    ap.add_argument("--snapshot", default="closing")
    args = ap.parse_args()
    props = NBA_PROPS if args.prop == "all" else [args.prop]
    discovery = {int(s) for s in args.discovery.split(",") if s.strip()}
    held = {int(s) for s in args.val.split(",") if s.strip()}
    books = ("draftkings", "fanduel") if args.book == "both" else (args.book,)
    for book in books:
        report(book, collect(book, props, args.snapshot), props, discovery, held, args.tol)


if __name__ == "__main__":
    main()
