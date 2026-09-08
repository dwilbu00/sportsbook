"""nfl_totals.py — an NFL TOTAL-POINTS model (the market we never modeled).

Everything so far predicts MARGIN (net EPA). Total points is a separate market,
driven by different signal — offensive AND defensive EPA *levels* (not net), pace
(plays/game), and weather (wind/cold/dome) — and it's where the recreational
OVER-bias lives, so it may be softer than spreads. This models expected total
points and tests it OOS + beat-the-close, same honest harness as the margin work.

Features per game (as-of, leakage-safe):
  epa_sum   = (off_epa[home]-def_epa[away]) + (off_epa[away]-def_epa[home])
              — how much both offenses should move the ball vs these defenses.
  pace      = combined plays/game (as-of) — more snaps => more scoring chances.
  wind      = max(wind-10, 0) outdoors (wind suppresses passing/kicking).
  cold      = max(32-temp, 0) outdoors.  dome = 1 if roof closed/dome.
Fit: actual_total ~ intercept + b*epa_sum + c*pace + wind + cold + dome (OLS, OOS).
"""
import argparse
import math
from collections import defaultdict

import nfl_epa
import nfl_schedule
import nfl_data
from nfl_injury_edge import _solve, _rmse
from nfl_market_scan import _load_closing_odds, _total_outcome, _profit

LEAGUE_PACE = 64.0     # ~offensive plays per team-game (centering)


def _weather(season):
    """{game_id: (wind, temp, roof)} from the pbp mirror."""
    pb = nfl_data.pbp([str(season)])
    out = {}
    if pb is not None and {"game_id", "wind", "temp", "roof"} <= set(pb.columns):
        for gid, sub in pb.groupby("game_id"):
            def first(col):
                v = sub[col].dropna()
                return v.iloc[0] if len(v) else None
            out[gid] = (first("wind"), first("temp"), first("roof"))
    return out


def _games_played_asof(season):
    """{(team, game_date): games_played_before} — for pace = plays / games."""
    # not needed separately; pace derived from off_plays/games via team_epa off_plays
    return None


def _pace_asof(season, as_of_date, team_ratings):
    """combined plays/game proxy: off_plays / games_played, per team, as-of."""
    # count each team's games strictly before as_of from the spine
    games = nfl_schedule.load_games([str(season)])
    gp = defaultdict(int)
    for g in games:
        d = str(g.get("gameday"))[:10]
        if d and d < as_of_date:
            gp[g.get("home_team")] += 1
            gp[g.get("away_team")] += 1
    def pace(t):
        r = team_ratings.get(t)
        n = gp.get(t, 0)
        return (r["off_plays"] / n) if (r and n > 0) else LEAGUE_PACE
    return pace


def _wx_terms(wx, gid):
    wind, temp, roof = wx.get(gid, (None, None, None))
    r = str(roof or "").lower()
    dome = 1.0 if r in ("dome", "closed") else 0.0
    outdoor = r in ("outdoors", "open", "")
    w = max(float(wind) - 10.0, 0.0) if (outdoor and wind is not None and wind == wind) else 0.0
    c = max(32.0 - float(temp), 0.0) if (outdoor and temp is not None and temp == temp) else 0.0
    return w, c, dome


def build_rows(seasons):
    """[(1, epa_sum, pace_c, wind, cold, dome, actual_total, season, gid)]."""
    rows = []
    for s in seasons:
        plays = nfl_epa.load_plays(s)
        wx = _weather(s)
        finals = {}
        for p in plays:
            if "home_score" in p:
                finals[p["game_id"]] = (p["game_date"], p["home_team"], p["away_team"],
                                        p["home_score"] + p["away_score"])
        for gid, (d, h, a, total) in sorted(finals.items(), key=lambda kv: kv[1][0]):
            R = nfl_epa.team_epa(s, as_of_date=d)
            rh, ra = R.get(h), R.get(a)
            if not rh or not ra or rh["off_plays"] <= 0 or ra["off_plays"] <= 0:
                continue
            epa_sum = ((rh["off_epa"] - ra["def_epa"]) + (ra["off_epa"] - rh["def_epa"]))
            pace = _pace_asof(s, d, R)
            pace_c = (pace(h) + pace(a)) - 2 * LEAGUE_PACE
            wind, cold, dome = _wx_terms(wx, gid)
            rows.append((1.0, epa_sum, pace_c, wind, cold, dome, float(total), s, gid))
    return rows


def oos(seasons):
    rows = build_rows(seasons)
    print(f"=== NFL TOTALS model — OOS ({len(rows)} games, leave-one-season-out) ===")
    feats = {
        "mean-only":        lambda r: [1.0],
        "+EPA":             lambda r: [1.0, r[1]],
        "+EPA+pace":        lambda r: [1.0, r[1], r[2]],
        "+EPA+pace+weather":lambda r: [1.0, r[1], r[2], r[3], r[4], r[5]],
    }
    ss = sorted({r[7] for r in rows})
    for name, fn in feats.items():
        tot = 0.0
        for test in ss:
            tr = [r for r in rows if r[7] != test]; te = [r for r in rows if r[7] == test]
            if not tr or not te: continue
            b = _solve([fn(r) for r in tr], [r[6] for r in tr])
            tot += _rmse([sum(bi*xi for bi,xi in zip(b,fn(r))) for r in te],
                         [r[6] for r in te]) * len(te)
        ball = _solve([fn(r) for r in rows], [r[6] for r in rows])
        print(f"  {name:<20} OOS_RMSE={tot/len(rows):.3f}  coef={[round(x,3) for x in ball]}")
    return rows


def beat_close(seasons, book="draftkings", snapshot="closing", thresholds=(2.0, 3.0, 4.0)):
    rows = build_rows(seasons)
    by_gid = {r[8]: r for r in rows}
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    scores = nfl_schedule.team_scores_index([str(s) for s in seasons])
    odds = _load_closing_odds([str(s) for s in seasons], book, snapshot)
    FEAT = lambda r: [1.0, r[1], r[2], r[3], r[4], r[5]]
    ss = sorted({r[7] for r in rows})
    pred = {}
    for test in ss:
        tr = [r for r in rows if r[7] != test]
        b = _solve([FEAT(r) for r in tr], [r[6] for r in tr])
        for r in rows:
            if r[7] == test:
                pred[r[8]] = sum(bi*xi for bi,xi in zip(b, FEAT(r)))
    mkt = {}
    for eid, o in odds.items():
        gid, _ = nfl_schedule.resolve_event(o["home"], o["away"], o["commence_time"], index=idx)
        if gid in by_gid and o["total"].get("over") and o["total"].get("under") and gid in scores:
            mkt[gid] = (o["total"]["over"], o["total"]["under"], *scores[gid])
    print(f"\n=== TOTALS beat-the-close (OOS, DK {snapshot}) ===")
    print(f"  model total vs closing total; bet over/under when |diff| >= thr, grade at close")
    print(f"    {'thr':<5}{'n':>5}{'ROI':>9}{'hit':>7}{'t':>7}   per-season")
    for thr in thresholds:
        rowsb = defaultdict(list)
        for gid, m in mkt.items():
            if gid not in pred: continue
            ov, un, hs, as_ = m
            line = ov[0]
            diff = pred[gid] - line
            if abs(diff) < thr: continue
            if diff > 0:   # model higher -> OVER
                out = _total_outcome(True, ov[0], hs, as_); px = ov[1]
            else:
                out = _total_outcome(False, un[0], hs, as_); px = un[1]
            rowsb[by_gid[gid][7]].append(_profit(px, out))
        allp = [p for v in rowsb.values() for p in v]
        if not allp:
            print(f"    {thr:<5}{'0':>5}  (no bets)"); continue
        n=len(allp); roi=sum(allp)/n
        se=(sum((p-roi)**2 for p in allp)/(n-1)/n)**.5 if n>1 else 0
        hit=sum(1 for p in allp if p>0)/n
        seas=" ".join(f"{s}:{(sum(v)/len(v)*100 if v else 0):+.0f}%({len(v)})" for s,v in sorted(rowsb.items()))
        print(f"    {thr:<5}{n:>5}{roi*100:+8.1f}%{hit*100:6.1f}%{(roi/se if se else 0):+7.2f}   [{seas}]")


def main():
    try:
        from cli_encoding import configure_stdio; configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--beat-close", action="store_true")
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    oos(seasons)
    if args.beat_close:
        beat_close(seasons)


if __name__ == "__main__":
    main()
