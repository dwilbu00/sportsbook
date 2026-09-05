"""nfl_market_scan.py — FLAT-SIDE market-structure scan for NFL.

Grades flat-betting each side of every game — no model involved — off the clean
nflverse ``game_id`` spine, to surface recreational-money biases (favorite-longshot,
home lean, over lean, dog-cover). This is the FIRST NFL look because MLB's ONLY
clean-data survivor was a market-structure cell (inverted public over-bias on
batter-K), not the model.

Sides graded:
  moneyline : favorite / dog / home / away
  spread    : favorite-cover / dog-cover / home-cover / away-cover
  total     : over / under

HONEST METHODOLOGY (the MLB scars, encoded):
  * Devig with r2_sharp.fair_two_way (Clarke) — NEVER band on raw vigged implied
    prob (that artifact killed the MLB coherence edge).
  * Bucket by devigged-fair FAVORITE strength (ML/spread) or total line (totals),
    AND by season — per-season replication is the honesty gate.
  * Report t-stats; flag CANDIDATE cells = n>=100, +ROI pooled, +ROI every season.
  * Prices = the book we actually bet (DraftKings). Completed games only.
  * Default snapshot = closing (the efficient reference); --snapshot early_4h to
    check the execution window.

Nothing is written — diagnostic only.
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
from odds_client import american_to_decimal
from r2_sharp import fair_two_way

SPORT = "americanfootball_nfl"

# Devigged-fair FAVORITE win-prob bands (ML/spread lenses).
_FAV_BANDS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
              (0.70, 0.80), (0.80, 1.01)]
# Total-line bands (totals lens).
_TOTAL_BANDS = [(0, 41), (41, 44), (44, 47), (47, 50), (50, 200)]


def _band(bands, v):
    if v is None:
        return None
    for lo, hi in bands:
        if lo <= v < hi:
            return f"{lo:g}-{hi:g}"
    return None


def _profit(price, outcome):
    """Flat 1u profit: outcome in {'win','loss','push'}."""
    if outcome == "push":
        return 0.0
    if outcome == "win":
        d = american_to_decimal(price)
        return (d - 1.0) if d else 0.0
    return -1.0


def _load_closing_odds(seasons, book, snapshot):
    """{event_id: game dict} from the odds mirror for one book+window. Each dict:
    home, away, commence_time, ml{home,away price}, spread{home,away (point,price)},
    total{over,under (point,price)}."""
    import warehouse_mirror as wm
    games = {}
    for s in seasons:
        try:
            rows = wm.team_market_lines(SPORT, date_from=f"{s}-01-01",
                                        date_to=f"{s}-12-31", bookmaker=book)
        except Exception:
            rows = []
        for r in rows:
            if r.get("source") != snapshot:
                continue
            eid = r.get("event_id")
            if not eid:
                continue
            g = games.setdefault(eid, {
                "home": r.get("home"), "away": r.get("away"),
                "commence_time": r.get("commence_time"),
                "ml": {}, "spread": {}, "total": {}})
            bt, sel = r.get("bet_type"), r.get("selection")
            pt, px = r.get("point"), r.get("price")
            if px is None:
                continue
            if bt == "moneyline":
                side = "home" if sel == g["home"] else ("away" if sel == g["away"] else None)
                if side:
                    g["ml"][side] = px
            elif bt == "spread":
                side = "home" if sel == g["home"] else ("away" if sel == g["away"] else None)
                if side and pt is not None:
                    g["spread"][side] = (pt, px)
            elif bt == "total":
                side = str(sel).lower()   # 'over' / 'under'
                if side in ("over", "under") and pt is not None:
                    g["total"][side] = (pt, px)
    return games


def _ml_outcome(bet_home, home_score, away_score):
    if home_score == away_score:
        return "push"   # ties are rare but possible in NFL
    home_won = home_score > away_score
    return "win" if (home_won == bet_home) else "loss"


def _spread_outcome(bet_home, point, home_score, away_score):
    """Did the bet side cover? point is that side's spread (home line if bet_home)."""
    margin = (home_score - away_score) if bet_home else (away_score - home_score)
    edge = margin + point
    if abs(edge) < 1e-9:
        return "push"
    return "win" if edge > 0 else "loss"


def _total_outcome(bet_over, point, home_score, away_score):
    tot = home_score + away_score
    if abs(tot - point) < 1e-9:
        return "push"
    return "win" if ((tot > point) == bet_over) else "loss"


def scan(seasons, snapshot="closing", book="draftkings"):
    odds = _load_closing_odds(seasons, book, snapshot)
    idx = nfl_schedule.game_index(seasons)
    scores = nfl_schedule.team_scores_index(seasons)

    # rows[(lens, side, band)][season] -> list of profits
    rows = defaultdict(lambda: defaultdict(list))
    n_games = n_graded = n_nojoin = n_noscore = 0

    for eid, g in odds.items():
        n_games += 1
        gid, reason = nfl_schedule.resolve_event(g["home"], g["away"],
                                                 g["commence_time"], index=idx)
        if gid is None:
            n_nojoin += 1
            continue
        sc = scores.get(gid)
        if sc is None:
            n_noscore += 1          # not yet played (e.g. current-season future)
            continue
        hs, as_ = sc
        season = gid.split("_")[0]
        n_graded += 1

        # Devigged fair favorite prob (for banding ML + spread lenses).
        fav_band = None
        mlh, mla = g["ml"].get("home"), g["ml"].get("away")
        if mlh is not None and mla is not None:
            fh, fa = fair_two_way(mlh, mla)
            if fh is not None:
                fav_prob = max(fh, fa)
                fav_is_home = fh >= fa
                fav_band = _band(_FAV_BANDS, fav_prob)

        # ── MONEYLINE: favorite / dog / home / away ──
        if mlh is not None and mla is not None and fav_band:
            for side, bet_home, price in (
                    ("home", True, mlh), ("away", False, mla)):
                out = _ml_outcome(bet_home, hs, as_)
                rows[("ml", side, fav_band)][season].append(_profit(price, out))
            fav_price = mlh if fav_is_home else mla
            dog_price = mla if fav_is_home else mlh
            rows[("ml", "favorite", fav_band)][season].append(
                _profit(fav_price, _ml_outcome(fav_is_home, hs, as_)))
            rows[("ml", "dog", fav_band)][season].append(
                _profit(dog_price, _ml_outcome(not fav_is_home, hs, as_)))

        # ── SPREAD: favorite-cover / dog-cover / home / away ──
        sph, spa = g["spread"].get("home"), g["spread"].get("away")
        if sph and spa and fav_band:
            for side, bet_home, (pt, px) in (
                    ("home", True, sph), ("away", False, spa)):
                out = _spread_outcome(bet_home, pt, hs, as_)
                rows[("spread", side, fav_band)][season].append(_profit(px, out))
            # favorite = negative point side
            fav_side_home = sph[0] < 0
            fpt, fpx = sph if fav_side_home else spa
            dpt, dpx = spa if fav_side_home else sph
            rows[("spread", "favorite", fav_band)][season].append(
                _profit(fpx, _spread_outcome(fav_side_home, fpt, hs, as_)))
            rows[("spread", "dog", fav_band)][season].append(
                _profit(dpx, _spread_outcome(not fav_side_home, dpt, hs, as_)))

        # ── TOTAL: over / under (banded by total line) ──
        ov, un = g["total"].get("over"), g["total"].get("under")
        if ov and un:
            tband = _band(_TOTAL_BANDS, ov[0])
            if tband:
                rows[("total", "over", tband)][season].append(
                    _profit(ov[1], _total_outcome(True, ov[0], hs, as_)))
                rows[("total", "under", tband)][season].append(
                    _profit(un[1], _total_outcome(False, un[0], hs, as_)))

    return rows, {"n_games": n_games, "n_graded": n_graded,
                  "n_nojoin": n_nojoin, "n_noscore": n_noscore,
                  "seasons": sorted({gid.split('_')[0] for gid in scores})}


def _stats(profits):
    n = len(profits)
    if not n:
        return None
    mean = sum(profits) / n
    if n > 1:
        var = sum((p - mean) ** 2 for p in profits) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
    else:
        se = 0.0
    t = (mean / se) if se > 0 else 0.0
    wins = sum(1 for p in profits if p > 0)
    return {"n": n, "roi": mean, "hit": wins / n, "t": t, "pl": sum(profits)}


def report(rows, meta, min_n=100, min_season_n=30):
    print("=" * 78)
    print(f"  NFL FLAT-SIDE MARKET-STRUCTURE SCAN  (DraftKings, {meta['seasons']})")
    print(f"  games with odds={meta['n_games']}  graded={meta['n_graded']}  "
          f"no-join={meta['n_nojoin']}  no-score/future={meta['n_noscore']}")
    print("  ROI = flat-1u; bands = DEVIGGED-fair favorite prob (ML/spread) or total "
          "line; per-season replication is the honesty gate.")
    print("=" * 78)

    # Pool per (lens, side) across bands, + per (lens, side, band). Aggregate seasons.
    def _agg(season_map):
        allp = [p for lst in season_map.values() for p in lst]
        return allp, {s: lst for s, lst in season_map.items()}

    candidates = []
    for lens in ("ml", "spread", "total"):
        keys = sorted(k for k in rows if k[0] == lens)
        if not keys:
            continue
        print(f"\n── {lens.upper()} ──────────────────────────────────────────────")
        # pooled-per-side (across bands)
        side_pool = defaultdict(lambda: defaultdict(list))
        for (l, side, band) in keys:
            for s, lst in rows[(l, side, band)].items():
                side_pool[side][s].extend(lst)
        print("  POOLED per side (all bands):")
        for side in sorted(side_pool):
            allp, smap = _agg(side_pool[side])
            st = _stats(allp)
            if not st:
                continue
            seas = " ".join(f"{s}:{(_stats(v) or {}).get('roi',0)*100:+.1f}%"
                            for s, v in sorted(smap.items()))
            print(f"    {side:<10} n={st['n']:>5} ROI={st['roi']*100:+6.2f}% "
                  f"hit={st['hit']*100:4.1f}% t={st['t']:+5.2f}   [{seas}]")
        # by band
        print("  by favorite band / total line (n>=%d):" % min_n)
        for (l, side, band) in keys:
            allp, smap = _agg(rows[(l, side, band)])
            st = _stats(allp)
            if not st or st["n"] < min_n:
                continue
            season_sts = {s: _stats(v) for s, v in smap.items()}
            judged = [ss for ss in season_sts.values() if ss and ss["n"] >= min_season_n]
            all_pos = judged and all(ss["roi"] > 0 for ss in judged)
            flag = " <<< CANDIDATE" if (st["roi"] > 0 and all_pos and len(judged) >= 2) else ""
            if st["roi"] > 0 or flag:
                print(f"    {side:<9} band={band:<9} n={st['n']:>5} "
                      f"ROI={st['roi']*100:+6.2f}% hit={st['hit']*100:4.1f}% "
                      f"t={st['t']:+5.2f}{flag}")
            if flag:
                candidates.append((lens, side, band, st))

    print("\n" + "=" * 78)
    if candidates:
        print("  === CANDIDATE cells (n>=%d, +ROI pooled, +ROI every judged season) ===" % min_n)
        for lens, side, band, st in sorted(candidates, key=lambda x: -x[3]["roi"]):
            print(f"    {lens} {side} band={band}  n={st['n']} "
                  f"ROI={st['roi']*100:+.2f}% t={st['t']:+.2f}")
    else:
        print("  === NO candidate cells cleared (n>=%d, +ROI pooled, +ROI every season) ===" % min_n)
    print("  (Diagnostic only — nothing written. Same honesty bar as MLB: assume "
          "artifact until it replicates.)")
    print("=" * 78)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--snapshot", choices=["closing", "early_4h", "early_12h"],
                    default="closing")
    ap.add_argument("--book", default="draftkings")
    ap.add_argument("--min-n", type=int, default=100)
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    rows, meta = scan(seasons, snapshot=args.snapshot, book=args.book)
    report(rows, meta, min_n=args.min_n)


if __name__ == "__main__":
    main()
