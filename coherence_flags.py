"""Daily coherence run-line FLAGS — the live/forward-tracking counterpart to
coherence_backtest.py.

The backtest validated a small, robust (but marginal, t~1.66) DK-internal edge: DK's
run-line drifts from what its OWN moneyline + total imply, and betting the +EV side
nets ~4-6% OOS. The honest way to confirm a t~1.66 edge is FORWARD — so this prints
today's flagged run-line plays (DK price + edge) to bet small and track.

Per game: devig DK's ML + total, solve run means (coherence.py), read the implied
run-line cover, subtract the historical calibration offset (the Poisson-shape bias,
fit on completed seasons — no leakage for a forward bet), and flag whichever DK
run-line side is +EV past the floor.

DK-only + DK-internal: needs no other book. Two ways to get today's DK odds:
  • --live (default OFF): one self-contained Odds-API call fetches DK's current
    ML+spread+total for the whole slate (~3 credits = 3 markets x 1 region, flat
    regardless of slate size). No app dependency — just run it.
  • warehouse (default): reads the latest pre-commence DK snapshot already captured
    (credit-free), so run it after the app's live analysis has stored today's odds.

USAGE
    python coherence_flags.py --live               # today, MLB, self-contained (~3 credits)
    python coherence_flags.py                       # today, MLB, from warehouse (free)
    python coherence_flags.py --date 2026-08-27
    python coherence_flags.py --live --ev-floor 0.05 --haircut 0.03
"""
import argparse
import datetime

import coherence
from odds_client import american_to_decimal
from r2_sharp import fair_two_way

DEFAULT_OFFSET_SEASONS = ["2024", "2025", "2026"]


def compute_offset(sport, seasons, dispersion=0.0):
    """Calibration offset = mean (implied - DK RL fair) over completed-season triads
    (the stable Poisson-shape bias). Fit on history, applied forward — no leakage."""
    import r2_data
    triads_by_season, _ = r2_data.load_team_triad(sport, seasons)
    vals = []
    for triads in triads_by_season.values():
        for t in triads:
            mlf, _ = fair_two_way(t.ml_home, t.ml_away)
            ovf, _ = fair_two_way(t.total_over, t.total_under)
            rlf, _ = fair_two_way(t.rl_home, t.rl_away)
            if None in (mlf, ovf, rlf):
                continue
            impl = coherence.implied_home_cover(mlf, t.total_line, ovf,
                                                t.rl_home_point, dispersion)
            if impl is not None:
                vals.append(impl - rlf)
    return (sum(vals) / len(vals)) if vals else 0.0, len(vals)


def flag_games(triads, offset, dispersion=0.0, haircut=0.02, ev_floor=0.03):
    """PURE: flag the +EV DK run-line side per game. Returns flag dicts sorted by EV
    (descending). ``offset`` is the calibration constant from compute_offset."""
    flags = []
    for t in triads:
        ml_home_fair, _ = fair_two_way(t.ml_home, t.ml_away)
        over_fair, _ = fair_two_way(t.total_over, t.total_under)
        if ml_home_fair is None or over_fair is None:
            continue
        implied = coherence.implied_home_cover(
            ml_home_fair, t.total_line, over_fair, t.rl_home_point, dispersion)
        if implied is None:
            continue
        implied = min(1.0 - 1e-6, max(1e-6, implied - offset))   # calibrated fair
        sides = [("home", t.home, t.rl_home_point, t.rl_home, implied),
                 ("away", t.away, -t.rl_home_point, t.rl_away, 1.0 - implied)]
        for side, team, point, price, fair in sides:
            try:
                ev = fair * american_to_decimal(int(price)) * (1.0 - haircut) - 1.0
            except (TypeError, ValueError):
                continue
            if ev >= ev_floor:
                flags.append({
                    "event_id": t.event_id, "game_date": t.game_date,
                    "away": t.away, "home": t.home, "side": side, "team": team,
                    "point": point, "dk_price": int(price), "ev": ev,
                    "coherent_fair": fair,
                })
    return sorted(flags, key=lambda f: f["ev"], reverse=True)


def load_triads_for_date(sport, date):
    """Latest pre-commence DK team triads (ML+RL+total) for one game_date, from the
    warehouse (whatever the live capture has stored). Credit-free."""
    import db_store
    import r2_data
    db_store.promote_secrets_from_toml()
    rows = db_store.team_market_lines(sport, dates=[date], bookmaker="draftkings")
    rows = [dict(r, book="draftkings") for r in rows if r.get("kind") == "team"]
    return r2_data.select_team_triad(rows)


def triads_from_upcoming(games, book_key="draftkings"):
    """PURE: parse the Odds-API upcoming-odds JSON (list of game dicts) into TeamTriad
    records for one book. Each game needs that book's complete ML+spread+total; games
    missing any leg (or the book) are dropped and counted. No snapshot timing — these
    ARE the current live prices."""
    from collections import Counter

    import r2_data
    triads, stats = [], Counter()
    for g in games or []:
        stats["events"] += 1
        home, away = g.get("home_team"), g.get("away_team")
        book = next((b for b in g.get("bookmakers", [])
                     if b.get("key") == book_key), None)
        if not home or not away or book is None:
            stats["events_dropped_no_book"] += 1
            continue
        ml, rl, tot = {}, {}, {}
        for m in book.get("markets", []):
            key = m.get("key")
            for o in m.get("outcomes", []):
                name, price, point = o.get("name"), o.get("price"), o.get("point")
                if key == "h2h":
                    ml[name] = price
                elif key == "spreads":
                    rl[name] = (point, price)
                elif key == "totals":
                    tot[name] = (point, price)
        ml_home, ml_away = ml.get(home), ml.get(away)
        rl_home, rl_away = rl.get(home), rl.get(away)
        over, under = tot.get("Over"), tot.get("Under")
        if None in (ml_home, ml_away, rl_home, rl_away, over, under):
            stats["events_dropped_incomplete_triad"] += 1
            continue
        commence = g.get("commence_time")
        triads.append(r2_data.TeamTriad(
            event_id=g.get("id"), game_date=(commence or "")[:10],
            commence_time=commence, snapshot_id=None, game_pk=None,
            home=home, away=away, ml_home=ml_home, ml_away=ml_away,
            rl_home_point=rl_home[0], rl_home=rl_home[1], rl_away=rl_away[1],
            total_line=over[0], total_over=over[1], total_under=under[1]))
        stats["triads_built"] += 1
    return triads, stats


def load_triads_live(sport, book_key="draftkings"):
    """Self-contained: one Odds-API call fetches the whole slate's current DK
    ML+spread+total (~3 credits = 3 markets x 1 region) and parses it into triads.
    No warehouse/app dependency."""
    import odds_client
    from backfill_historical_odds import load_config
    api_key = load_config()["odds_api_key"]
    games = odds_client.get_upcoming_odds(
        api_key, sport, regions="us", markets="h2h,spreads,totals",
        bookmakers=[book_key])
    return triads_from_upcoming(games, book_key)


def main():
    ap = argparse.ArgumentParser(description="Daily coherence run-line flags (DK-internal).")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--live", action="store_true",
                    help="Self-contained: fetch DK's current odds via the Odds API "
                         "(~3 credits) instead of reading the warehouse.")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD (default: today). Ignored with --live.")
    ap.add_argument("--offset-seasons", default=",".join(DEFAULT_OFFSET_SEASONS),
                    help="Completed seasons to fit the calibration offset on.")
    ap.add_argument("--dispersion", type=float, default=0.0)
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--ev-floor", type=float, default=0.03)
    args = ap.parse_args()
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    date = args.date or datetime.date.today().isoformat()
    offset_seasons = [s.strip() for s in args.offset_seasons.split(",") if s.strip()]
    offset, n_off = compute_offset(args.sport, offset_seasons, args.dispersion)
    if args.live:
        triads, stats = load_triads_live(args.sport)
        src = "live (Odds API, current DK prices)"
    else:
        triads, stats = load_triads_for_date(args.sport, date)
        src = f"warehouse {date}"

    print(f"\n  Coherence run-line flags — {args.sport} — {src}")
    print(f"  offset {offset:+.4f} (from {n_off:,} historical triads)  "
          f"ev_floor {args.ev_floor:.0%}  haircut {args.haircut:.0%}")
    print(f"  games with a complete DK triad: {len(triads)}  "
          f"(dropped incomplete: {stats.get('events_dropped_incomplete_triad', 0)})")
    if not triads:
        msg = ("  No DK triads returned by the Odds API — check the slate has games "
               "and DK has posted lines." if args.live else
               "  No DK triads for this date — run the app's live analysis first to "
               "capture today's odds, or check the date has games.")
        print(msg)
        return

    flags = flag_games(triads, offset, args.dispersion, args.haircut, args.ev_floor)
    if not flags:
        print("  No +EV run-line flags today.")
        return
    print(f"\n  {len(flags)} flag(s) — bet SMALL + track (edge is marginal, t~1.66):")
    for f in flags:
        sign = "+" if f["point"] > 0 else ""
        px = f["dk_price"]
        print(f"    {f['away']} @ {f['home']:<24}  {f['side'].upper():<5} "
              f"{f['team']} {sign}{f['point']} @ {px:+d}   "
              f"EV {f['ev']:+.1%}  (coherent fair {f['coherent_fair']:.3f})")
    print()


if __name__ == "__main__":
    main()
