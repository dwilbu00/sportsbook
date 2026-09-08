"""nfl_anchoring.py — the Week-1 preseason-anchoring FADE (NFL edge lead).

Thesis (Patterson/Shank/Fodor, "Anchoring Bias in the NFL Gambling Market", SSRN
2025, N=5,088, 2003-2023): bettors anchor to PRE-SEASON Super Bowl futures odds all
season, over-betting the preseason favorite. That over-betting is INVERSELY related
to WEEK-1 ATS profitability → the preseason-anchored favorite UNDER-covers in Week 1.
The profit relationship is WEEK-1-ONLY (the bias persists all season but the line
adjusts). So the actionable rule is narrow and once-a-year:

    In WEEK 1, for each game, FADE the team with the BETTER (lower) preseason Super
    Bowl futures price — bet its opponent ATS at the closing/early_4h number.

The edge concentrates at the EXTREMES (paper's quintiles): strongest when the two
teams' preseason SB odds differ a LOT. `--min-gap` (in implied-prob points of the
devigged SB-title prob difference) restricts to the high-conviction games.

⚠ Small-sample honesty: Week 1 = ~16 games/yr, so our 2023-2025 corpus is ~48 games
— TOO THIN to prove. Lean on the paper's 20-year prior; use the backtest as a sanity
check + forward-track. Bets at DraftKings/FanDuel only.

Inputs:
  * preseason SB futures odds per team per season → `calibration/nfl_preseason_sb_odds.json`
    shape {"2026": {"KC": 550, "BUF": 650, ...}, "2023": {...}} (American odds; lower =
    bigger favorite). Fill from any book's preseason SB-winner market (the paper used
    Pro-Football-Reference). Team keys = nflverse abbreviations (nfl_epa.TEAM_ABBR_TO_NAME).
  * game spine + results: nfl_schedule; closing/early_4h spreads: the odds mirror (DK).

Diagnostic/advisory — never auto-bets.
"""
import argparse
import json
import os

import nfl_schedule
from odds_client import american_to_implied_prob

SPORT = "americanfootball_nfl"
SB_ODDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration", "nfl_preseason_sb_odds.json")


SB_WINNER_SPORT = "americanfootball_nfl_super_bowl_winner"


def fetch_sb_odds(season, book="draftkings", path=SB_ODDS_PATH):
    """Pull the current Super Bowl-winner futures (outrights) for one book from the
    Odds API and MERGE into the SB-odds file under ``season``. Small spend (~1 credit:
    1 market × 1 region). Pre-season now = the anchoring-relevant odds. Returns the
    {abbr: american} dict written."""
    import backfill_historical_odds as _cfg
    import odds_client
    import nfl_epa
    api_key = _cfg.load_config()["odds_api_key"]
    data = odds_client.get_upcoming_odds(api_key, SB_WINNER_SPORT, regions="us",
                                         markets="outrights", bookmakers=[book])
    odds = {}
    for event in (data or []):
        for bm in event.get("bookmakers", []):
            if bm.get("key") != book:
                continue
            for mk in bm.get("markets", []):
                if mk.get("key") != "outrights":
                    continue
                for oc in mk.get("outcomes", []):
                    ab = nfl_epa._abbr(oc.get("name"))
                    price = oc.get("price")
                    if ab and price is not None:
                        odds[ab] = int(price)
    if not odds:
        print("  ⚠ no outrights parsed — check the response / book key.")
        return {}
    allsb = load_sb_odds(path)
    allsb[str(season)] = odds
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(allsb, f, indent=2, sort_keys=True)
    print(f"  wrote {len(odds)}/32 teams for {season} → {os.path.basename(path)} "
          f"(book={book}).")
    missing = set(nfl_epa.TEAM_ABBR_TO_NAME) - set(odds)
    if missing:
        print(f"  ⚠ missing teams (no SB price returned): {sorted(missing)}")
    return odds


def load_sb_odds(path=SB_ODDS_PATH):
    """{season(str): {team_abbr: american_odds(int)}} or {} if absent."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _title_prob(american):
    """Raw implied SB-title prob from American odds (not devigged across the field —
    the 32-way book overround is large, but we only need the RELATIVE ordering + a
    gap, and raw implied preserves ordering monotonically)."""
    try:
        return american_to_implied_prob(int(american))
    except (TypeError, ValueError):
        return None


def anchored_favorite(home, away, sb_season):
    """Return (fade_team, keep_team, gap) where fade_team = the preseason-anchored
    favorite (higher SB-title prob = lower/ better odds) to bet AGAINST, keep_team =
    its opponent (the side we BET ATS), gap = |title-prob difference|. None if either
    team lacks an SB price."""
    ph, pa = _title_prob(sb_season.get(home)), _title_prob(sb_season.get(away))
    if ph is None or pa is None:
        return None
    if ph == pa:
        return None
    fade, keep = (home, away) if ph > pa else (away, home)
    return fade, keep, abs(ph - pa)


# ── live: this week's fade plays ──────────────────────────────────────────────

def week_plays(season, week, min_gap=0.0, snapshot="closing", book="draftkings"):
    """Advisory plays for (season, week): the ATS side to BET (fade the anchored fav).
    Attaches the game's spread for the KEEP side from the odds mirror when available."""
    sb = load_sb_odds().get(str(season), {})
    if not sb:
        return None
    games = [g for g in nfl_schedule.load_games([str(season)])
             if str(g["week"]) == str(week)]
    # spreads from the odds mirror (keyed by event via resolve; here we join by teams)
    spreads = _spreads_by_matchup(season, snapshot, book)
    out = []
    for g in sorted(games, key=lambda r: str(r["gameday"])):
        res = anchored_favorite(g["home_team"], g["away_team"], sb)
        if res is None:
            continue
        fade, keep, gap = res
        if gap < min_gap:
            continue
        sp = spreads.get((g["home_team"], g["away_team"]))
        keep_line = None
        if sp:
            keep_line = sp.get(keep)   # the spread (point, price) for the side we bet
        out.append({"gameday": g["gameday"], "away": g["away_team"],
                    "home": g["home_team"], "bet_ats": keep, "fade": fade,
                    "gap_prob": gap, "keep_line": keep_line})
    return out


def _spreads_by_matchup(season, snapshot, book):
    """{(home,away): {team: (point, price)}} for a season's spreads at a window."""
    try:
        import warehouse_mirror as wm
        rows = wm.team_market_lines(SPORT, date_from=f"{season}-01-01",
                                    date_to=f"{season}-12-31", bookmaker=book)
    except Exception:
        rows = []
    out = {}
    for r in rows:
        if r.get("source") != snapshot or r.get("bet_type") != "spread":
            continue
        pt, px = r.get("point"), r.get("price")
        if pt is None or px is None:
            continue
        key = (r.get("home"), r.get("away"))
        sel = r.get("selection")
        # map full-name selection -> abbr via nfl_epa
        import nfl_epa
        ab = nfl_epa._abbr(sel)
        if ab:
            out.setdefault(key, {})[ab] = (pt, px)
    # re-key matchups by abbr for the join in week_plays
    import nfl_epa
    rekey = {}
    for (h, a), d in out.items():
        hh, aa = nfl_epa._abbr(h), nfl_epa._abbr(a)
        if hh and aa:
            rekey[(hh, aa)] = d
    return rekey


# ── backtest: Week-1 fade ATS on completed seasons (the prior sanity check) ────

def _cover(bet_home, point, home_score, away_score):
    margin = (home_score - away_score) if bet_home else (away_score - home_score)
    edge = margin + point
    if abs(edge) < 1e-9:
        return "push"
    return "win" if edge > 0 else "loss"


def backtest(seasons, week=1, min_gap=0.0, snapshot="closing", book="draftkings"):
    from odds_client import american_to_decimal
    sb_all = load_sb_odds()
    scores = nfl_schedule.team_scores_index([str(s) for s in seasons])
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    rows = []
    for season in seasons:
        sb = sb_all.get(str(season), {})
        if not sb:
            continue
        spreads = _spreads_by_matchup(season, snapshot, book)
        for g in idx.values():
            if g["season"] != str(season) or str(g["week"]) != str(week):
                continue
            res = anchored_favorite(g["home_team"], g["away_team"], sb)
            if res is None:
                continue
            fade, keep, gap = res
            if gap < min_gap:
                continue
            sc = scores.get(g["game_id"])
            sp = spreads.get((g["home_team"], g["away_team"]))
            if sc is None or not sp or keep not in sp:
                continue
            pt, px = sp[keep]
            hs, as_ = sc
            out = _cover(keep == g["home_team"], pt, hs, as_)
            profit = 0.0 if out == "push" else ((american_to_decimal(px) - 1.0)
                                                if out == "win" else -1.0)
            rows.append({"season": str(season), "keep": keep, "fade": fade,
                         "gap": gap, "result": out, "profit": profit})
    return rows


def _summ(rows):
    n = len(rows)
    if not n:
        return None
    import math
    pl = [r["profit"] for r in rows]
    mean = sum(pl) / n
    wins = sum(1 for r in rows if r["result"] == "win")
    if n > 1:
        var = sum((p - mean) ** 2 for p in pl) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
    else:
        se = 0.0
    return {"n": n, "roi": mean, "hit": wins / n, "t": (mean / se if se else 0.0),
            "pl": sum(pl)}


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2026")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="min |devigged-ish SB title-prob difference| to flag (0=all). "
                         "Edge concentrates at extremes → try 0.05-0.10.")
    ap.add_argument("--snapshot", default="closing",
                    choices=["closing", "early_4h", "early_12h"])
    ap.add_argument("--backtest", action="store_true",
                    help="grade the Week-1 fade ATS on 2023-2025 (needs historical SB odds)")
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--fetch-sb-odds", action="store_true",
                    help="pull current Super Bowl futures (outrights, DK) from the Odds "
                         "API into the SB-odds file for --season (~1 credit spend)")
    ap.add_argument("--book", default="draftkings")
    args = ap.parse_args()

    if args.fetch_sb_odds:
        fetch_sb_odds(args.season, book=args.book)
        # fall through to print this week's plays with the freshly-written odds

    sb = load_sb_odds()
    if not sb:
        print(f"⚠ No preseason SB odds file at {SB_ODDS_PATH}.")
        print('  Fill it: {"2026": {"KC": 550, "BUF": 650, ...}, "2023": {...}} '
              "(American odds, nflverse abbreviations). Then re-run.")
        return

    if args.backtest:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
        rows = backtest(seasons, week=args.week, min_gap=args.min_gap,
                        snapshot=args.snapshot)
        print(f"=== Week-{args.week} preseason-anchoring FADE backtest "
              f"({args.snapshot}, min_gap={args.min_gap}) ===")
        alls = _summ(rows)
        if not alls:
            print("  no gradable games (missing SB odds / spreads / results).")
            return
        print(f"  POOLED n={alls['n']} ROI={alls['roi']*100:+.2f}% "
              f"hit={alls['hit']*100:.1f}% t={alls['t']:+.2f} P/L={alls['pl']:+.2f}u")
        for s in sorted({r["season"] for r in rows}):
            ss = _summ([r for r in rows if r["season"] == s])
            print(f"    {s}: n={ss['n']} ROI={ss['roi']*100:+.2f}% hit={ss['hit']*100:.1f}%")
        print("  ⚠ ~16 games/season — thin; the paper's 2003-2023 N=5,088 is the prior.")
        return

    plays = week_plays(args.season, args.week, min_gap=args.min_gap,
                       snapshot=args.snapshot)
    print(f"=== NFL Week {args.week} {args.season} — preseason-anchoring FADE plays "
          f"(bet the ATS side; min_gap={args.min_gap}) ===")
    if not plays:
        print("  (no plays — missing SB odds for this season, or no games pass the gap.)")
        return
    for p in plays:
        line = p["keep_line"]
        line_s = (f"{line[0]:+g} ({line[1]:+d})" if line else "line: check DK/FD")
        print(f"  {p['gameday']}  {p['away']} @ {p['home']}  → BET {p['bet_ats']} "
              f"{line_s}  [fade {p['fade']}, gap {p['gap_prob']*100:.1f}pp]")
    print("  Bet at DraftKings/FanDuel. Small stakes (Week-1-only, thin-sample edge).")


if __name__ == "__main__":
    main()
