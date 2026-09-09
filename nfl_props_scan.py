"""nfl_props_scan.py — flat-side PROP market-structure scan for NFL.

Props are the biggest recreational NFL market and the one place MLB left a survivor
(batter-K UNDER — an inverted public over-bias). This grades flat-betting each side
(OVER / UNDER) of every player prop at the DK closing price vs the REAL result
(from the player_week layer), per market + per season, to find systematically
over-bet sides that clear the vig on the inverse.

Honest bar (MLB scars): actual price ROI (vig included); per-season replication
gate; t-stats; candidate = n>=150 + +ROI pooled + +ROI every judged season. A −ROI
side means ABSTAIN, not invert — UNLESS the loss exceeds the round-trip vig, then the
opposite side is the +EV bet (the batter-K-under lesson).

Join: prop `player` name + the game's (season,week) [via the spine] → player_week
actual, matched on a normalized name. DNPs (no player_week row) are ungradable → skip.
Diagnostic only. DraftKings; completed games only.
"""
import argparse
import math
import re
from collections import defaultdict

import nfl_schedule
import nfl_data
from odds_client import american_to_decimal

SPORT = "americanfootball_nfl"

# prop_key -> player_week actual column
PROP_MAP = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_pass_attempts": "attempts",
    "player_pass_completions": "completions",
    "player_pass_interceptions": "passing_interceptions",
    "player_rush_yds": "rushing_yards",
    "player_rush_attempts": "carries",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}

# per-market line-bucket width (for the line-conditional cut — MLB's edge hid at a
# specific line, not pooled)
_BUCKET_W = {
    "player_pass_yds": 25, "player_rush_yds": 15, "player_reception_yds": 15,
    "player_pass_attempts": 5, "player_pass_completions": 4, "player_rush_attempts": 3,
    "player_receptions": 1, "player_pass_tds": 1, "player_pass_interceptions": 1,
}


def _line_bucket(pk, point):
    if point is None:
        return None
    w = _BUCKET_W.get(pk, 10)
    lo = int(point // w) * w
    return f"{lo}-{lo + w}"


_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _norm(name):
    if not name:
        return None
    s = str(name).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _player_week_index(seasons):
    """{(norm_name, season, week): {stat_col: value}} from player_week."""
    pw = nfl_data.player_week([str(s) for s in seasons])
    if pw is None:
        return {}
    namecol = "player_display_name" if "player_display_name" in pw.columns else "player_name"
    idx = {}
    stat_cols = [c for c in set(PROP_MAP.values()) if c in pw.columns]
    for r in pw.to_dict("records"):
        key = (_norm(r.get(namecol)), str(r.get("season")), str(r.get("week")))
        idx[key] = {c: r.get(c) for c in stat_cols}
    return idx


def _load_props(seasons, snapshot, book):
    """{(event_id, player, prop_key): {'OVER':(pt,px), 'UNDER':(pt,px), 'commence':..,
    'home':.., 'away':..}} from the props mirror."""
    import warehouse_mirror as wm
    out = {}
    for s in seasons:
        try:
            rows = wm.player_prop_lines(SPORT, date_from=f"{s}-01-01",
                                        date_to=f"{s}-12-31", bookmaker=book)
        except Exception:
            rows = []
        from nfl_market_scan import _after_kickoff
        for r in rows:
            if r.get("source") != snapshot:
                continue
            if _after_kickoff(r.get("captured_at"), r.get("commence_time")):
                continue                  # post-kickoff quote — not pregame info [review 2026-09-09]
            pk = r.get("prop_key")
            if pk not in PROP_MAP:
                continue
            key = (r.get("event_id"), r.get("player"), pk)
            d = out.setdefault(key, {"commence": r.get("commence_time"),
                                     "home": r.get("home"), "away": r.get("away")})
            direction = str(r.get("direction") or "").upper()
            pt, px = r.get("point"), r.get("price")
            if direction in ("OVER", "UNDER") and pt is not None and px is not None:
                d[direction] = (pt, px)
    return out


def _grade(actual, point, over):
    if actual is None or point is None:
        return None
    if abs(actual - point) < 1e-9:
        return "push"
    hit = (actual > point) if over else (actual < point)
    return "win" if hit else "loss"


def _profit(price, outcome):
    if outcome == "push":
        return 0.0
    if outcome == "win":
        d = american_to_decimal(price)
        return (d - 1.0) if d else 0.0
    return -1.0


def _snap_presence(seasons):
    """{(norm_name, season, week): True} for players with ANY recorded snap that week
    (from snap_counts). Distinguishes a player who PLAYED but has no player_week stat
    row (active-zero → a real 0 for the stat) from a true nonparticipant (VOID).
    [review 2026-09-09]"""
    out = {}
    for s in seasons:
        try:
            sc = nfl_data.snap_counts([str(s)])
        except Exception:
            sc = None
        if sc is None or not {"player", "week"} <= set(sc.columns):
            continue
        for r in sc.to_dict("records"):
            snaps = (r.get("offense_snaps") or 0) or (r.get("defense_snaps") or 0) \
                or (r.get("st_snaps") or 0)
            if not snaps:
                continue
            try:
                wk = str(int(r.get("week")))
            except (TypeError, ValueError):
                continue
            out[(_norm(r.get("player")), str(s), wk)] = True
    return out


def scan(seasons, snapshot="closing", book="draftkings", drill=None):
    """drill = (prop_key, side, bucket) → also collect per-bet detail for that cell:
    list of (season, week, player, point, price, actual, outcome, profit)."""
    props = _load_props(seasons, snapshot, book)
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    pw = _player_week_index(seasons)
    snaps = _snap_presence([str(s) for s in seasons])

    rows = defaultdict(lambda: defaultdict(list))       # (prop_key,side)->season->[profit]
    rows_line = defaultdict(lambda: defaultdict(list))  # (prop_key,side,bucket)->season->[profit]
    drill_rows = []
    n_props = n_graded = n_nogame = n_void = n_played0 = 0
    for (eid, player, pk), d in props.items():
        n_props += 1
        gid, _ = nfl_schedule.resolve_event(d["home"], d["away"], d["commence"], index=idx)
        if gid is None:
            n_nogame += 1
            continue
        parts = gid.split("_")
        season, week = parts[0], str(int(parts[1]))
        stat = PROP_MAP[pk]
        actual = (pw.get((_norm(player), season, week)) or {}).get(stat)
        if actual is None:
            # No player_week stat row. Use snap_counts to decide: if the player took
            # snaps, they PLAYED and simply recorded nothing → a genuine 0 for this
            # stat (grade it). If no snaps, they're a nonparticipant → VOID (skip; do
            # NOT invent an automatic OVER-loss/UNDER-win — DK voids non-participants).
            if snaps.get((_norm(player), season, week)):
                actual = 0.0
                n_played0 += 1
            else:
                n_void += 1
                continue
        n_graded += 1
        for side, over in (("OVER", True), ("UNDER", False)):
            if side not in d:
                continue
            pt, px = d[side]
            out = _grade(actual, pt, over)
            if out is not None:
                profit = _profit(px, out)
                rows[(pk, side)][season].append(profit)
                b = _line_bucket(pk, pt)
                if b is not None:
                    rows_line[(pk, side, b)][season].append(profit)
                if drill and (pk, side, b) == drill:
                    drill_rows.append((season, week, player, pt, px, actual, out, profit))
    return rows, rows_line, {"n_props": n_props, "n_graded": n_graded,
                            "n_nogame": n_nogame, "n_void": n_void,
                            "n_played0": n_played0,
                            "seasons": sorted({g.split('_')[0] for g in idx}),
                            "drill_rows": drill_rows}


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
    return {"n": n, "roi": mean, "hit": wins / n, "t": (mean / se if se else 0.0)}


def report(rows, meta, min_n=150, min_season_n=40):
    print("=" * 88)
    print(f"  NFL PROPS flat-side scan (DK closing, {meta['seasons']})  "
          f"graded={meta['n_graded']} (incl {meta.get('n_played0',0)} played-zero)  "
          f"no-join={meta['n_nogame']}  VOID non-participants={meta.get('n_void',0)}")
    print("  flat ROI at DK price (vig in); per-season replication = honesty gate; "
          "candidate = n>=%d + +ROI pooled + every season +." % min_n)
    print("=" * 88)
    candidates = []
    for pk in sorted(PROP_MAP):
        for side in ("OVER", "UNDER"):
            smap = rows.get((pk, side))
            if not smap:
                continue
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
            print(f"  {pk:<26} {side:<5} n={st['n']:>5} ROI={st['roi']*100:+6.2f}% "
                  f"hit={st['hit']*100:4.1f}% t={st['t']:+5.2f}  [{seas}]{mark}")
            if flag:
                candidates.append((pk, side, st))
    print("=" * 88)
    if candidates:
        print("  === CANDIDATES (n>=%d, +ROI pooled, +ROI every judged season) ===" % min_n)
        for pk, side, st in sorted(candidates, key=lambda x: -x[2]["roi"]):
            print(f"    {pk} {side}  n={st['n']} ROI={st['roi']*100:+.2f}% t={st['t']:+.2f}")
    else:
        print("  === NO prop side cleared the bar ===")
    print("  (a −ROI side = ABSTAIN unless loss > round-trip vig, then bet the inverse.)")


def report_by_line(rows_line, meta, min_bucket_n=100, min_season_n=30):
    print("\n" + "=" * 88)
    print("  LINE-CONDITIONAL cut (per market × side × line bucket — where MLB's edge hid)")
    print(f"  cells n>={min_bucket_n}; candidate = +ROI pooled + +ROI every judged season")
    print("=" * 88)
    candidates = []
    by_prop = defaultdict(list)
    for (pk, side, b), smap in rows_line.items():
        allp = [p for lst in smap.values() for p in lst]
        st = _stats(allp)
        if not st or st["n"] < min_bucket_n:
            continue
        season_sts = {s: _stats(v) for s, v in smap.items()}
        judged = [ss for ss in season_sts.values() if ss and ss["n"] >= min_season_n]
        all_pos = judged and all(ss["roi"] > 0 for ss in judged)
        flag = st["roi"] > 0 and all_pos and len(judged) >= 2
        by_prop[pk].append((side, b, st, season_sts, flag))
        if flag:
            candidates.append((pk, side, b, st))
    for pk in sorted(by_prop):
        # show only cells that are +ROI or flagged (the interesting ones)
        cells = [c for c in by_prop[pk] if c[2]["roi"] > 0 or c[4]]
        if not cells:
            continue
        print(f"  {pk}:")
        for side, b, st, sss, flag in sorted(cells, key=lambda x: -x[2]["roi"]):
            seas = " ".join(f"{s}:{(v or {}).get('roi',0)*100:+.0f}%" for s, v in sorted(sss.items()))
            print(f"    {side:<5} line {b:<8} n={st['n']:>4} ROI={st['roi']*100:+6.2f}% "
                  f"hit={st['hit']*100:4.1f}% t={st['t']:+5.2f}  [{seas}]"
                  + (" <<< CANDIDATE" if flag else ""))
    print("=" * 88)
    if candidates:
        print("  === LINE-CONDITIONAL CANDIDATES ===")
        for pk, side, b, st in sorted(candidates, key=lambda x: -x[3]["roi"]):
            print(f"    {pk} {side} line {b}  n={st['n']} ROI={st['roi']*100:+.2f}% t={st['t']:+.2f}")
    else:
        print("  === NO line-conditional cell cleared the bar ===")


def _american_of(px):
    """price back to american int for display (odds are stored american already)."""
    try:
        return int(round(px))
    except (TypeError, ValueError):
        return None


def report_drill(drill, drill_rows):
    pk, side, b = drill
    print("\n" + "=" * 88)
    print(f"  DRILL: {pk} {side} line-bucket {b}  (is this actually bettable +EV, or a juiced trap?)")
    print("=" * 88)
    if not drill_rows:
        print("  (no bets in this cell)")
        return
    # price-band buckets on american odds
    bands = [(-100000, -300, "<= -300 (heavy juice)"),
             (-300, -200, "-300..-200"),
             (-200, -150, "-200..-150"),
             (-150, -110, "-150..-110"),
             (-110, 100, "-110..+100"),
             (100, 100000, ">= +100 (plus money)")]
    from collections import defaultdict as _dd
    band_pl = _dd(list)
    exact_pt = _dd(list)
    for (s, wk, player, pt, px, actual, out, profit) in drill_rows:
        am = _american_of(px)
        exact_pt[pt].append(profit)
        for lo, hi, lbl in bands:
            if am is not None and lo <= am < hi:
                band_pl[lbl].append((profit, out))
                break
    print(f"  n={len(drill_rows)} bets.  Exact line-points present: "
          + ", ".join(f"{pt}({len(v)})" for pt, v in sorted(exact_pt.items())))
    print("  --- ROI by CLOSING-PRICE band (the execution reality) ---")
    for lo, hi, lbl in bands:
        pls = band_pl.get(lbl)
        if not pls:
            continue
        prof = [p for p, _ in pls]
        n = len(prof); roi = sum(prof) / n
        wins = sum(1 for _, o in pls if o == "win")
        print(f"    {lbl:<24} n={n:>4} ROI={roi*100:+6.2f}% hit={wins/n*100:4.1f}%")
    # a few sample bets
    print("  --- sample bets (season wk player line price actual → outcome) ---")
    for r in sorted(drill_rows)[:12]:
        s, wk, player, pt, px, actual, out, profit = r
        print(f"    {s} w{wk:<2} {str(player)[:22]:<22} {pt} @ {_american_of(px):+5d}  "
              f"actual={actual} → {out}")
    print("  (grading note: players who took snaps but have no player_week stat row are")
    print("   now graded as a real 0 for the stat; true non-participants are VOID — so this")
    print("   cell no longer has a scratch-survivorship bias. [review 2026-09-09])")
    print("=" * 88)
    print("  READ: if the +ROI concentrates in plus-money / light-juice bands with real n,")
    print("        it's a bettable pattern; if it lives only in heavy-juice cells or a")
    print("        handful of outliers, it's a mirage / limit-gated.")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h"])
    ap.add_argument("--min-n", type=int, default=150)
    ap.add_argument("--by-line", action="store_true",
                    help="also show the line-conditional cut (per market × side × line bucket)")
    ap.add_argument("--min-bucket-n", type=int, default=100)
    ap.add_argument("--drill", default=None,
                    help="deep-dive one cell: 'prop_key:SIDE:bucket' e.g. 'player_receptions:OVER:0-1'")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    drill = None
    if args.drill:
        p = args.drill.split(":")
        if len(p) == 3:
            drill = (p[0], p[1].upper(), p[2])
    rows, rows_line, meta = scan(seasons, snapshot=args.snapshot, drill=drill)
    report(rows, meta, min_n=args.min_n)
    if args.by_line:
        report_by_line(rows_line, meta, min_bucket_n=args.min_bucket_n)
    if drill:
        report_drill(drill, meta["drill_rows"])


if __name__ == "__main__":
    main()
