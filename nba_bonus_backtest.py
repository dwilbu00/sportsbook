"""nba_bonus_backtest.py — NBA bonus-engine validation on real historical DK/FD prop lines
(READ-ONLY; 0 DTU via the mirrors; no credits). Owner runs it.

nba_opportunity_threshold established that the de-vigged NBA prop price is calibrated at every
minutes level and from opening night, so a leg's P = the book's de-vig (the engine's rule). This
answers the two remaining questions:

 (A) SAME-GAME CO-HIT: the live SGP builder prices a same-game stack as the PRODUCT of leg Ps
     (independence). For favorite pairs in the same game this compares the realized both-hit
     rate with that product, by relationship (same player nested/other, teammates same/other
     stat, opponents same/other stat) and by side combination (OO / UU / OU). ratio < 1 means
     independence OVERSTATES the stack (a fake +EV) -> avoid that composition; > 1 means the
     product is conservative -> prefer it.
 (B) CROSS-GAME BOOSTED ROI: the NFL backtest's sport-agnostic core (backtest_bonus) on NBA
     legs grouped by slate date: every qualifying +EV cross-game ticket for each LIVE_BONUSES
     promo, flat $1, per book. (Historical SGP combined prices don't exist, so SGP ROI can't be
     backtested — (A) is the SGP evidence, as for NFL.)

Run (owner):
  $env:PYTHONIOENCODING="utf-8"
  python nba_bonus_backtest.py                 # both books
"""
import argparse
import dataclasses
from collections import defaultdict
from itertools import combinations

import bonus as bonuslib
import book_line_calibration as blc
import nfl_bonus_backtest as core
from odds_client import american_to_implied_prob, devig_two_way

NBA_PROPS = ["player_points", "player_rebounds", "player_assists", "player_threes",
             "player_steals", "player_blocks", "player_turnovers", "player_points_rebounds",
             "player_points_assists", "player_points_rebounds_assists"]


def _cols(prop):
    import nba_stats_source as src
    c = src.PROP_STAT_COLS.get(prop)
    return set(c if isinstance(c, tuple) else (c,)) if c else set()


def collect(book, snapshot="closing", calibrated=False):
    """One favorite-side leg per (event, player, prop) at ``book``: every played game counts,
    pushes dropped. Each leg carries the event, slate date, season, teams and the favorite's
    de-vigged P, realized result and the book's own price for that side."""
    import nba_stats_source as src
    lines, _a, _b = blc.harvest_real_line_book_lines("basketball_nba", NBA_PROPS,
                                                     snapshot=snapshot, bookmaker=book)
    rows = blc._nba_grade_book_lines(lines, min_test_minutes=0.0, min_priors=0) or []
    maps = None
    if calibrated:                                  # per-book over-bias correction maps
        import book_calibration as bc
        maps = bc.load_maps("basketball_nba")
    legs = {}
    for r in rows:
        line, actual = r.get("line"), r.get("actual")
        op, up = r.get("over_price"), r.get("under_price")
        if line is None or actual is None or op is None or up is None:
            continue
        if abs(actual - line) < 1e-9:
            continue
        try:
            fair = devig_two_way(american_to_implied_prob(op), american_to_implied_prob(up))[0]
        except Exception:
            continue
        if maps is not None:
            import book_calibration as bc
            fair = bc.apply("basketball_nba", r.get("prop_key"), fair, maps=maps, book=book)
        tg = r.get("test_game") or {}
        ev = r.get("event_id") or (r.get("game_date"), r.get("home_team"), r.get("away_team"))
        over_won = 1 if actual > line else 0
        fav_over = fair >= 0.5
        key = (ev, r.get("player"), r.get("prop_key"))
        leg = {"event": ev, "date": str(r.get("game_date"))[:10],
               "season": src.end_year_for_date(str(r.get("game_date"))[:10]),
               "player": r.get("player"), "prop": r.get("prop_key"),
               "team": tg.get("team_id"), "opp": tg.get("opponent"),
               "side": "O" if fav_over else "U",
               "P": fair if fav_over else 1.0 - fair,
               "won": over_won if fav_over else 1 - over_won,
               "odds": op if fav_over else up}
        prev = legs.get(key)
        if prev is None or abs(leg["P"] - 0.5) < abs(prev["P"] - 0.5):
            legs[key] = leg                               # keep the main (closest-to-even) line
    return list(legs.values())


def _relationship(a, b):
    if a["player"] == b["player"]:
        ca, cb = _cols(a["prop"]), _cols(b["prop"])
        return "same player, nested" if (ca <= cb or cb <= ca) else "same player, other"
    if a["team"] is not None and a["team"] == b["team"]:
        return "teammates, same stat" if a["prop"] == b["prop"] else "teammates, other stat"
    return "opponents, same stat" if a["prop"] == b["prop"] else "opponents, other stat"


def co_hit(legs):
    """{(relationship, side combo): [n, sum(P1*P2), both_won]} over every same-game pair."""
    by_ev = defaultdict(list)
    for lg in legs:
        by_ev[lg["event"]].append(lg)
    acc = defaultdict(lambda: [0, 0.0, 0])
    for ev_legs in by_ev.values():
        for a, b in combinations(ev_legs, 2):
            sides = "".join(sorted(a["side"] + b["side"]))
            k = (_relationship(a, b), {"OO": "OO", "UU": "UU"}.get(sides, "OU"))
            c = acc[k]
            c[0] += 1
            c[1] += a["P"] * b["P"]
            c[2] += a["won"] * b["won"]
    return acc


def report_co_hit(book, acc):
    print(f"\n  (A) [{book}] SAME-GAME CO-HIT vs independence (favorite pairs)")
    print(f"    {'relationship':<24} {'sides':>5} {'pairs':>9} {'indep%':>7} {'real%':>7} "
          f"{'ratio':>6}")
    rels = ("same player, nested", "same player, other", "teammates, same stat",
            "teammates, other stat", "opponents, same stat", "opponents, other stat")
    for rel in rels:
        for sides in ("OO", "UU", "OU"):
            n, sp, w = acc.get((rel, sides), (0, 0.0, 0))
            if n < 200:
                continue
            ind, real = sp / n, w / n
            flag = "  <- independence OVERSTATES" if real / ind < 0.97 else (
                "  <- conservative (prefer)" if real / ind > 1.03 else "")
            print(f"    {rel:<24} {sides:>5} {n:>9,} {ind * 100:>6.2f}% {real * 100:>6.2f}% "
                  f"{real / ind:>6.3f}{flag}")


def cross_game(book, legs, seasons):
    pool = defaultdict(list)
    for lg in legs:
        if lg["season"] in seasons:
            pool[(book, lg["season"], lg["date"])].append(
                {"gid": lg["event"], "P_mkt": lg["P"], "odds": lg["odds"], "won": lg["won"]})
    print(f"\n  (B) [{book}] CROSS-GAME BOOSTED ROI (market-devig legs, flat $1, every qualifying "
          f"+EV ticket; seasons {sorted(seasons)})")
    print(f"    {'bonus':<42} {'K':>2} {'tickets':>8} {'predEV%':>8} {'ROI%':>7} {'hit%':>6} "
          f"{'expHit%':>8}")
    for b in bonuslib.LIVE_BONUSES:
        bb = dataclasses.replace(b, book=book)
        res = core.backtest_bonus(pool, bb, core._sizes_for(bb), "P_mkt")
        for k, rows in res.items():
            a = core._agg(rows)
            if a:
                print(f"    {b.label[:42]:<42} {k:>2} {a['n']:>8,} {a['pred_ev']:>8.2f} "
                      f"{a['roi']:>7.2f} {a['hit']:>6.2f} {a['exp_hit']:>8.2f}")
                by_s = defaultdict(list)
                for r in rows:
                    by_s[r["season"]].append(r)
                print("        by season: " + "  ".join(
                    f"{s}: {core._agg(v)['roi']:+.1f}% (n {len(v):,})"
                    for s, v in sorted(by_s.items())))


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
    ap.add_argument("--seasons", default="2024,2025,2026", help="seasons for (B)")
    ap.add_argument("--calibrated", action="store_true",
                    help="apply calibration/book_calibration_basketball_nba.json (the per-book "
                         "over-bias maps) to each leg's de-vig P before picking the favorite")
    args = ap.parse_args()
    seasons = {int(s) for s in args.seasons.split(",") if s.strip()}
    books = ("draftkings", "fanduel") if args.book == "both" else (args.book,)
    print("=" * 100)
    print("NBA BONUS BACKTEST — same-game co-hit (SGP composition) + cross-game boosted ROI")
    print("=" * 100)
    print(f"  leg P: {'book de-vig + per-book over-bias map' if args.calibrated else 'raw book de-vig'}")
    for book in books:
        legs = collect(book, calibrated=args.calibrated)
        print(f"\n  [{book}] favorite legs: {len(legs):,}")
        report_co_hit(book, co_hit(legs))
        cross_game(book, legs, seasons)
    print()


if __name__ == "__main__":
    main()
