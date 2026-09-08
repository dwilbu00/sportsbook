"""nfl_situational_scan.py — CONDITIONAL market-structure scan for NFL.

The flat-side scan (nfl_market_scan) pooled across situations and found nothing.
NFL's documented recreational/structural soft spots are CONDITIONAL — this grades
flat ATS/total bets restricted to specific situations, off the clean data model
(game_context from the spine + weather from pbp + DK closing spreads/totals +
results). Same honest bar as MLB: devigged prices, per-season replication gate,
t-stats, candidate = n>=100 + +ROI pooled + +ROI every judged season.

Angles tested (each = a documented NFL bias hypothesis + the side it implies):
  rest      off-bye team ATS; net-rest-edge (>=3 days) team ATS; fade short-week team
  division  UNDER; underdog ATS (familiarity → closer, lower-scoring)
  primetime UNDER; home ATS
  home_dog  home underdog ATS (public backs road favorites)
  weather   wind>=15mph UNDER; temp<=32F UNDER; dome OVER

Diagnostic only — nothing written. Bets graded at DraftKings; completed games only.
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
import nfl_data
from nfl_market_scan import (_load_closing_odds, _spread_outcome, _total_outcome,
                             _profit, _ml_outcome)
from r2_sharp import fair_two_way

SPORT = "americanfootball_nfl"


def _weather_by_game(seasons):
    """{game_id: {temp,wind,roof}} from pbp (one value/game). Absent if no pbp."""
    pb = nfl_data.pbp([str(s) for s in seasons])
    if pb is None:
        return {}
    out = {}
    cols = [c for c in ("temp", "wind", "roof") if c in pb.columns]
    for gid, sub in pb.groupby("game_id"):
        rec = {}
        for c in cols:
            v = sub[c].dropna()
            rec[c] = v.iloc[0] if len(v) else None
        out[gid] = rec
    return out


def _ats(side, g, sp):
    """(profit, outcome) for betting the home/away SIDE ATS at its DK spread, or None.
    `sp` is keyed by 'home'/'away' (see _load_closing_odds)."""
    if not sp or side not in sp:
        return None
    pt, px = sp[side]
    out = _spread_outcome(side == "home", pt, g["_hs"], g["_as"])
    return _profit(px, out), out


def _total(side, g, tot):
    if not tot or side not in tot:
        return None
    pt, px = tot[side]
    out = _total_outcome(side == "over", pt, g["_hs"], g["_as"])
    return _profit(px, out), out


def scan(seasons, snapshot="closing", book="draftkings"):
    odds = _load_closing_odds(seasons, book, snapshot)   # {eid: {home,away,commence,ml,spread,total}}
    scores = nfl_schedule.team_scores_index([str(s) for s in seasons])
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    ctx = nfl_data.game_context(seasons)
    wx = _weather_by_game(seasons)

    # rows[angle][season] -> [profit,...]
    rows = defaultdict(lambda: defaultdict(list))

    def add(angle, season, graded):
        if graded is not None:
            rows[angle][season].append(graded[0])

    n_graded = 0
    for eid, o in odds.items():
        gid, _ = nfl_schedule.resolve_event(o["home"], o["away"], o["commence_time"], index=idx)
        if gid is None or gid not in scores:
            continue
        c = ctx.get(gid)
        if not c:
            continue
        hs, as_ = scores[gid]
        season = gid.split("_")[0]
        g = dict(o); g["_hs"], g["_as"] = hs, as_
        sp, tot, ml = o["spread"], o["total"], o["ml"]
        home, away = o["home"], o["away"]
        n_graded += 1

        # ── rest ── (sp is keyed by 'home'/'away')
        if c.get("off_bye_home"):
            add("rest: off-bye team ATS", season, _ats("home", g, sp))
        if c.get("off_bye_away"):
            add("rest: off-bye team ATS", season, _ats("away", g, sp))
        re = c.get("rest_edge_home")
        if re is not None and re >= 3:
            add("rest: net-rest-edge (>=3d) team ATS", season, _ats("home", g, sp))
        elif re is not None and re <= -3:
            add("rest: net-rest-edge (>=3d) team ATS", season, _ats("away", g, sp))
        if c.get("short_week_home"):
            add("rest: fade short-week team ATS", season, _ats("away", g, sp))
        if c.get("short_week_away"):
            add("rest: fade short-week team ATS", season, _ats("home", g, sp))

        # ── division ──
        if c.get("is_division"):
            add("division: UNDER", season, _total("under", g, tot))
            if ml.get("home") is not None and ml.get("away") is not None:
                fh, fa = fair_two_way(ml["home"], ml["away"])
                if fh is not None and fh != fa:
                    dog_side = "home" if fh < fa else "away"
                    add("division: underdog ATS", season, _ats(dog_side, g, sp))

        # ── primetime ──
        if c.get("is_primetime"):
            add("primetime: UNDER", season, _total("under", g, tot))
            add("primetime: home ATS", season, _ats("home", g, sp))

        # ── home dog ──
        if ml.get("home") is not None and ml.get("away") is not None:
            fh, fa = fair_two_way(ml["home"], ml["away"])
            if fh is not None and fh < fa:   # home is the underdog (lower fair prob)
                add("home_dog: home ATS", season, _ats("home", g, sp))

        # ── weather ──
        w = wx.get(gid) or {}
        roof = str(w.get("roof") or "").lower()
        wind, temp = w.get("wind"), w.get("temp")
        outdoor = roof in ("outdoors", "open", "")
        if outdoor and wind is not None and wind >= 15:
            add("weather: wind>=15 UNDER", season, _total("under", g, tot))
        if outdoor and temp is not None and temp <= 32:
            add("weather: temp<=32 UNDER", season, _total("under", g, tot))
        if roof in ("dome", "closed"):
            add("weather: dome/closed OVER", season, _total("over", g, tot))

    return rows, {"n_graded": n_graded, "seasons": sorted({gid.split('_')[0] for gid in scores})}


def _stats(pl):
    n = len(pl)
    if not n:
        return None
    mean = sum(pl) / n
    if n > 1:
        var = sum((p - mean) ** 2 for p in pl) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
    else:
        se = 0.0
    wins = sum(1 for p in pl if p > 0)
    return {"n": n, "roi": mean, "hit": wins / n, "t": (mean / se if se else 0.0), "pl": sum(pl)}


def report(rows, meta, min_n=80, min_season_n=20):
    print("=" * 82)
    print(f"  NFL SITUATIONAL market-structure scan (DK closing, {meta['seasons']}, "
          f"{meta['n_graded']} games)")
    print("  flat ROI on the situational side; devigged where a side is chosen; per-season "
          "replication = honesty gate; candidate = n>=%d + +ROI pooled + every season +." % min_n)
    print("=" * 82)
    candidates = []
    for angle in sorted(rows):
        smap = rows[angle]
        allp = [p for lst in smap.values() for p in lst]
        st = _stats(allp)
        if not st or st["n"] < min_n:
            continue
        season_sts = {s: _stats(v) for s, v in smap.items()}
        judged = [ss for ss in season_sts.values() if ss and ss["n"] >= min_season_n]
        all_pos = judged and all(ss["roi"] > 0 for ss in judged)
        flag = st["roi"] > 0 and all_pos and len(judged) >= 2
        seas = " ".join(f"{s}:{(_stats(v) or {}).get('roi',0)*100:+.0f}%" for s, v in sorted(smap.items()))
        mark = " <<< CANDIDATE" if flag else ""
        print(f"  {angle:<38} n={st['n']:>4} ROI={st['roi']*100:+6.2f}% "
              f"hit={st['hit']*100:4.1f}% t={st['t']:+5.2f}  [{seas}]{mark}")
        if flag:
            candidates.append((angle, st))
    print("=" * 82)
    if candidates:
        print("  === CANDIDATES (n>=%d, +ROI pooled, +ROI every judged season) ===" % min_n)
        for angle, st in sorted(candidates, key=lambda x: -x[1]["roi"]):
            print(f"    {angle}  n={st['n']} ROI={st['roi']*100:+.2f}% t={st['t']:+.2f}")
    else:
        print("  === NO situational candidate cleared the bar ===")
    print("  (Diagnostic only. Same skepticism as MLB — thin/small-n/no-mechanism = noise.)")


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
    ap.add_argument("--min-n", type=int, default=80)
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    rows, meta = scan(seasons, snapshot=args.snapshot)
    report(rows, meta, min_n=args.min_n)


if __name__ == "__main__":
    main()
