"""Market-only CLV early-vs-close study across the precise-backfill windows.

STEP 1 of the post-reload re-test sequence. NO model, NO calibration, NO game
outcomes — this measures ONLY how the market's own line behaves between the rigid
precise-backfill windows (early_12h / early_4h / closing). It answers the transaction
question we need settled BEFORE recalibrating: for each market, is the CLOSE
meaningfully tighter/sharper (=> bet late, the early price is stale/wide) or are the
early prices already competitive (=> room to transact early and hold CLV)?

Why market-only first: line movement is deterministic per event (no win/loss variance),
so a few thousand paired events give a confident read, whereas realized-ROI timing
studies need far more bets to clear noise. We characterize the market here, then let
the recalibrated model decide direction in later steps.

Reads through the warehouse `snapshot` selector, which filters on the `odds_snapshot.source`
label — so early_12h / early_4h / closing are distinguished by their WINDOW ROLE, not a
captured_at heuristic. With the parquet mirror present this is 0-DTU / fully offline.

Metrics, per sport x market x window-pair (early->close):
  - N        paired events present in BOTH the early window and closing
  - vigE/vigC mean overround (raw two-way implied sum - 1) at early vs close; a tighter
             close vig means the close is the sharp/liquid price
  - |dp|     mean & median ABSOLUTE move in the de-vigged fair prob of the reference
             side (home ML / home cover / over) early->close = how much timing matters
  - dp       mean SIGNED move (directional drift: favorite steam, over/under bias)
  - |dpts|   mean absolute move in the LINE (points) for spreads/totals

Team markets: moneyline, spread (run/point line), total — early_12h AND early_4h vs
closing. Props: over/under fair prob — early_4h vs closing only (props have no 12h).

Usage (run on the fast box; reads the LFS-materialized mirror):
  python clv_window_study.py --sport mlb
  python clv_window_study.py --sport all
"""
import argparse
import statistics as stats

import warehouse as _wh
import warehouse_mirror as _mir
from odds_client import devig_two_way, american_to_implied_prob

SPORT_KEYS = {
    "mlb": "baseball_mlb",
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
}
# Team-market windows to compare against closing. Props omit early_12h (never captured).
TEAM_PAIRS = (("early_12h", "12h->close"), ("early_4h", "4h->close"))
PROP_PAIRS = (("early_4h", "4h->close"),)


# ── reference-side fair prob + vig per market (de-vig two-way) ─────────────────

def _ml(entry):
    """moneyline -> (fair_home_prob, vig, line=None) or None."""
    ml = entry.get("moneyline") or {}
    h = (ml.get(entry.get("home_team")) or [None])[0]
    a = (ml.get(entry.get("away_team")) or [None])[0]
    if not h or not a:
        return None
    ih, ia = h.get("implied_prob"), a.get("implied_prob")
    if ih is None or ia is None:
        return None
    fair_home, _ = devig_two_way(ih, ia)
    return fair_home, (ih + ia) - 1.0, None


def _spread(entry):
    """spread -> (fair_home_cover_prob, vig, home_spread_line) or None."""
    sp = entry.get("spreads") or {}
    h = (sp.get(entry.get("home_team")) or [None])[0]
    a = (sp.get(entry.get("away_team")) or [None])[0]
    if not h or not a or h.get("price") is None or a.get("price") is None:
        return None
    ih = american_to_implied_prob(h["price"])
    ia = american_to_implied_prob(a["price"])
    fair, _ = devig_two_way(ih, ia)
    return fair, (ih + ia) - 1.0, h.get("spread")


def _total(entry):
    """total -> (fair_over_prob, vig, line) or None."""
    tot = entry.get("totals") or {}
    o = (tot.get("Over") or [None])[0]
    u = (tot.get("Under") or [None])[0]
    if not o or not u or o.get("price") is None or u.get("price") is None:
        return None
    io = american_to_implied_prob(o["price"])
    iu = american_to_implied_prob(u["price"])
    fair, _ = devig_two_way(io, iu)
    return fair, (io + iu) - 1.0, o.get("line")


TEAM_MARKETS = (("moneyline", _ml), ("spread", _spread), ("total", _total))


def _prop_ref(info):
    """prop over/under row -> (fair_over_prob, vig, line) or None."""
    io, iu = info.get("over_implied"), info.get("under_implied")
    if io is None or iu is None:
        return None
    fair, _ = devig_two_way(io, iu)
    return fair, (io + iu) - 1.0, info.get("line")


# ── formatting ────────────────────────────────────────────────────────────────

def _mean(xs):
    return stats.fmean(xs) if xs else float("nan")


def _fmt_row(label, pair, n, vigE, vigC, dabs, dmed, dsig, dpts):
    pts = f"{dpts * 100:8.2f}" if dpts == dpts else "     n/a"  # NaN check
    print(f"  {label:<10} {pair:<10} {n:>6} {vigE * 100:>7.2f} {vigC * 100:>7.2f} "
          f"{dabs * 100:>8.2f} {dmed * 100:>8.2f} {dsig * 100:>+8.2f} {pts}")


# ── study ─────────────────────────────────────────────────────────────────────

def _dates_for(sport_key, file_fn, seasons):
    """Concrete game_date list for the mirror-covered seasons (canonical DK file).
    Passing an explicit `dates` list keeps the loaders on the single-call mirror path
    (0 DTU) instead of the unscoped 2019->now year-loop that falls back to Azure for
    seasons with no mirror file. Skips seasons whose mirror file is absent."""
    dates = set()
    for s in seasons:
        df = _mir._read(file_fn(sport_key, "draftkings", str(s)))
        if df is None or "game_date" not in getattr(df, "columns", []):
            continue
        dates.update(str(d) for d in df["game_date"].dropna().unique())
    return sorted(dates)


def study_team(sport_key, seasons):
    dates = _dates_for(sport_key, _mir._team_file, seasons)
    if not dates:
        print("  (no team mirror files for requested seasons — run git lfs pull?)")
        return
    close = _wh.load_team_market_store(
        sport_key, dates=dates, snapshot="closing").get("games", {})
    if not close:
        print("  (no closing team store — mirror missing or window unpopulated)")
        return
    print("\n  TEAM markets  (fair prob = de-vigged reference side: home ML / home "
          "cover / over)")
    print(f"  {'market':<10} {'window':<10} {'N':>6} {'vigE%':>7} {'vigC%':>7} "
          f"{'|dp|%':>8} {'med%':>8} {'dp%':>8} {'|dLine|':>8}")
    for early_src, pair in TEAM_PAIRS:
        early = _wh.load_team_market_store(
            sport_key, dates=dates, snapshot=early_src).get("games", {})
        if not early:
            continue
        for name, fn in TEAM_MARKETS:
            dabs, dsig, dpts, vE, vC = [], [], [], [], []
            for gk in set(early) & set(close):
                e, c = fn(early[gk]), fn(close[gk])
                if not e or not c:
                    continue
                dabs.append(abs(c[0] - e[0]))
                dsig.append(c[0] - e[0])
                vE.append(e[1])
                vC.append(c[1])
                if e[2] is not None and c[2] is not None:
                    dpts.append(abs(c[2] - e[2]))
            if not dabs:
                continue
            _fmt_row(name, pair, len(dabs), _mean(vE), _mean(vC), _mean(dabs),
                     stats.median(dabs), _mean(dsig),
                     _mean(dpts) / 100 if dpts else float("nan"))


def study_props(sport_key, seasons):
    dates = _dates_for(sport_key, _mir._prop_file, seasons)
    if not dates:
        print("\n  (no prop mirror files for requested seasons)")
        return
    close = _wh.load_prop_market_store(
        sport_key, dates=dates, snapshot="closing").get("games", {})
    if not close:
        print("\n  (no closing prop store)")
        return
    prop_keys = sorted({p for g in close.values() for p in (g.get("props") or {})})
    if not prop_keys:
        print("\n  (closing prop store has no markets)")
        return
    print("\n  PROPS  (fair over prob; early_4h -> close only)")
    print(f"  {'prop':<24} {'N':>6} {'vigE%':>7} {'vigC%':>7} {'|dp|%':>8} "
          f"{'med%':>8} {'dp%':>8} {'|dLine|':>8}")
    for early_src, _pair in PROP_PAIRS:
        early = _wh.load_prop_market_store(
            sport_key, dates=dates, snapshot=early_src).get("games", {})
        if not early:
            print("  (no early_4h prop store)")
            return
        for prop in prop_keys:
            dabs, dsig, dpts, vE, vC = [], [], [], [], []
            for gk in set(early) & set(close):
                cm = (close[gk].get("props") or {}).get(prop) or {}
                em = (early[gk].get("props") or {}).get(prop) or {}
                for player, ci in cm.items():
                    ei = em.get(player)
                    if not ei:
                        continue
                    e, c = _prop_ref(ei), _prop_ref(ci)
                    if not e or not c:
                        continue
                    dabs.append(abs(c[0] - e[0]))
                    dsig.append(c[0] - e[0])
                    vE.append(e[1])
                    vC.append(c[1])
                    if e[2] is not None and c[2] is not None:
                        dpts.append(abs(c[2] - e[2]))
            if len(dabs) < 50:
                continue
            pts = f"{_mean(dpts):8.2f}" if dpts else "     n/a"
            print(f"  {prop:<24} {len(dabs):>6} {_mean(vE) * 100:>7.2f} "
                  f"{_mean(vC) * 100:>7.2f} {_mean(dabs) * 100:>8.2f} "
                  f"{stats.median(dabs) * 100:>8.2f} {_mean(dsig) * 100:>+8.2f} {pts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", default="mlb",
                    choices=["mlb", "nba", "nfl", "all"])
    ap.add_argument("--seasons", default="2023,2024,2025,2026",
                    help="comma-separated seasons to include (mirror-covered only)")
    ap.add_argument("--props", action="store_true",
                    help="include the props section (slower; team-only by default)")
    args = ap.parse_args()
    sports = ["mlb", "nba", "nfl"] if args.sport == "all" else [args.sport]
    seasons = [x.strip() for x in args.seasons.split(",") if x.strip()]
    print(f"reading from: {_mir.source_label()}  (mirror enabled={_mir.enabled()})  "
          f"seasons={','.join(seasons)}")
    for s in sports:
        key = SPORT_KEYS[s]
        print(f"\n=== CLV early-vs-close (MARKET-ONLY) : {key} ===")
        study_team(key, seasons)
        if args.props:
            study_props(key, seasons)
    print("\nRead: vigC << vigE  => close is the sharp/tight price (bet late). "
          "|dp| large => timing matters a lot. dp (signed) far from 0 => a directional "
          "drift the model can exploit by transacting on the correct side of it.")


if __name__ == "__main__":
    main()
