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
from odds_client import american_to_decimal, american_to_implied_prob
from r2_sharp import fair_two_way

DEFAULT_OFFSET_SEASONS = ["2024", "2025", "2026"]

# App-integration defaults (mirror the CLI): the validated coherence run-line was
# fit Poisson (dispersion 0), and forward-flagged at a 3% EV floor / 2% vig haircut.
DEFAULT_DISPERSION = 0.0
DEFAULT_HAIRCUT = 0.02
DEFAULT_EV_FLOOR = 0.03

# SHARPENING (2026-08-28): the coherence run-line edge is a pure UNDERDOG +1.5 signal
# (coherence_backtest flags 0 favorite -1.5 bets), and its ROI concentrates at
# MODERATE favorites — flat below ~60% ML implied, ~+11% at 60-70%, NEGATIVE on heavy
# favorites (>=70%). Two independent methods agree (coherence_backtest + scenario_
# backtest dog_runline, the latter ~+20% replicating all 3 seasons at 65-70%). So the
# live flag is sharpened to (a) the dog +1.5 side only and (b) this favorite band.
# The band is data-selected on 2024-2026, so it's a FORWARD-TRACK hypothesis — widen/
# relax via the params if the in-app forward log says otherwise.
DEFAULT_FAV_MIN = 0.60
DEFAULT_FAV_MAX = 0.70


def _fav_band_ok(ml_home_fair, fav_min, fav_max):
    """True if the game's FAVORITE devigged win prob is in [fav_min, fav_max) — the
    moderate-favorite band where the dog +1.5 coherence edge concentrates."""
    fav_imp = max(ml_home_fair, 1.0 - ml_home_fair)
    return fav_min <= fav_imp < fav_max


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


def flag_games(triads, offset, dispersion=0.0, haircut=0.02, ev_floor=0.03,
               fav_min=DEFAULT_FAV_MIN, fav_max=DEFAULT_FAV_MAX):
    """PURE: flag the +EV DK run-line side per game. Returns flag dicts sorted by EV
    (descending). ``offset`` is the calibration constant from compute_offset.

    SHARPENED: only the underdog +1.5 side, only when the favorite is in the moderate
    [fav_min, fav_max) band (see the module note). Pass fav_min=0.0, fav_max=1.0 to
    disable the band and recover the raw (all-favorites, dog-only) behavior."""
    flags = []
    for t in triads:
        ml_home_fair, _ = fair_two_way(t.ml_home, t.ml_away)
        over_fair, _ = fair_two_way(t.total_over, t.total_under)
        if ml_home_fair is None or over_fair is None:
            continue
        if not _fav_band_ok(ml_home_fair, fav_min, fav_max):
            continue
        implied = coherence.implied_home_cover(
            ml_home_fair, t.total_line, over_fair, t.rl_home_point, dispersion)
        if implied is None:
            continue
        implied = min(1.0 - 1e-6, max(1e-6, implied - offset))   # calibrated fair
        sides = [("home", t.home, t.rl_home_point, t.rl_home, implied),
                 ("away", t.away, -t.rl_home_point, t.rl_away, 1.0 - implied)]
        for side, team, point, price, fair in sides:
            if point <= 0:          # coherence edge is the UNDERDOG +1.5 side only
                continue
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


def _first_price(entries):
    """First usable american price from a parse_game_odds entry list (DK-only view)."""
    for e in entries or []:
        p = e.get("price")
        if p is not None:
            return p
    return None


def _run_line_entry(entries):
    """The MAIN run-line (|point|==1.5) as (signed_point, price) from a team's
    parse_game_odds spreads list; falls back to the entry nearest ±1.5. (None, None)
    if the team has no usable spread."""
    best = None
    for e in entries or []:
        sp, pr = e.get("spread"), e.get("price")
        if sp is None or pr is None:
            continue
        try:
            sp = float(sp)
        except (TypeError, ValueError):
            continue
        if abs(abs(sp) - 1.5) < 1e-9:
            return sp, pr
        d = abs(abs(sp) - 1.5)
        if best is None or d < best[0]:
            best = (d, sp, pr)
    return (best[1], best[2]) if best else (None, None)


def run_line_candidates(game_odds, offset, dispersion=DEFAULT_DISPERSION,
                        haircut=DEFAULT_HAIRCUT, ev_floor=DEFAULT_EV_FLOOR,
                        fav_min=DEFAULT_FAV_MIN, fav_max=DEFAULT_FAV_MAX):
    """PURE: from ONE game's parsed DK odds (odds_client.parse_game_odds shape:
    moneyline/spreads/totals dicts + home_team/away_team), emit the +EV coherence
    run-line side(s) as spread-shaped candidate dicts tagged type='runline_coherence'.

    Coherence = solve run means from DK's own ML + total, read the implied run-line
    cover, subtract the historical Poisson-shape ``offset`` (calibration), and flag
    whichever DK run-line side clears ``ev_floor`` at a vig ``haircut``. Returns a
    list (0-1 side is +EV in practice; both are checked). The candidate mirrors the
    analyze_spreads_value key set so it rides the existing pool/checklist/wager rails;
    ``event_id`` is stamped by the caller. Never raises — returns [] on any gap."""
    try:
        home = game_odds.get("home_team")
        away = game_odds.get("away_team")
        ml = game_odds.get("moneyline") or {}
        sp = game_odds.get("spreads") or {}
        tot = game_odds.get("totals") or {}
        ml_home, ml_away = _first_price(ml.get(home)), _first_price(ml.get(away))
        point_h, price_h = _run_line_entry(sp.get(home))
        point_a, price_a = _run_line_entry(sp.get(away))
        over_entries, under_entries = tot.get("Over"), tot.get("Under")
        over_price = _first_price(over_entries)
        under_price = _first_price(under_entries)
        total_line = next((e.get("line") for e in (over_entries or [])
                           if e.get("line") is not None), None)
        if None in (home, away, ml_home, ml_away, point_h, price_h, point_a,
                    price_a, over_price, under_price, total_line):
            return []
        # The two run-line points must be the complementary main line (±1.5).
        if abs(float(point_h) + float(point_a)) > 1e-6:
            return []

        ml_home_fair, _ = fair_two_way(ml_home, ml_away)
        over_fair, _ = fair_two_way(over_price, under_price)
        if ml_home_fair is None or over_fair is None:
            return []
        # SHARPENED: only surface the edge at moderate favorites (see module note).
        if not _fav_band_ok(ml_home_fair, fav_min, fav_max):
            return []
        implied = coherence.implied_home_cover(
            ml_home_fair, float(total_line), over_fair, float(point_h), dispersion)
        if implied is None:
            return []
        implied = min(1.0 - 1e-6, max(1e-6, implied - offset))   # calibrated fair
        rl_home_fair, rl_away_fair = fair_two_way(price_h, price_a)  # devigged market

        out = []
        sides = [("HOME", home, away, float(point_h), price_h, implied, rl_home_fair),
                 ("AWAY", away, home, float(point_a), price_a, 1.0 - implied,
                  rl_away_fair)]
        for home_away, team, opp, point, price, coh_fair, mkt_fair in sides:
            if point <= 0:          # coherence edge is the UNDERDOG +1.5 side only
                continue
            try:
                dec = american_to_decimal(int(price))
            except (TypeError, ValueError):
                continue
            ev = coh_fair * dec * (1.0 - haircut) - 1.0
            if ev < ev_floor:
                continue
            baseline = mkt_fair if mkt_fair is not None else \
                american_to_implied_prob(int(price))
            out.append({
                "type": "runline_coherence",
                "team": team, "opponent": opp, "home_away": home_away,
                "spread": point,
                "cover_rate": round(coh_fair * 100, 2),
                "model_cover_rate": round(coh_fair * 100, 2),
                "implied_prob": round(baseline * 100, 2),
                "edge_pct": round((coh_fair - baseline) * 100, 2),
                "expected_roi_pct": round(ev * 100, 2),
                "is_value": True,
                "price": int(price), "price_missing": False,
                "games_sampled": None,
                "pred_game_margin": None, "pred_game_std": None,
                "model_source": "coherence",
                "matchup": f"{away} @ {home}",
                "coherent_fair": round(coh_fair, 4),
            })
        return out
    except Exception:
        return []


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
    ap.add_argument("--fav-min", type=float, default=DEFAULT_FAV_MIN,
                    help="Min favorite ML-implied prob to flag (moderate-fav band).")
    ap.add_argument("--fav-max", type=float, default=DEFAULT_FAV_MAX,
                    help="Max favorite ML-implied prob to flag. Pass --fav-min 0 "
                         "--fav-max 1 to disable the band (all favorites).")
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

    flags = flag_games(triads, offset, args.dispersion, args.haircut, args.ev_floor,
                       fav_min=args.fav_min, fav_max=args.fav_max)
    print(f"  sharpened to dog +1.5 at favorites in "
          f"[{args.fav_min:.0%}, {args.fav_max:.0%}) ML-implied")
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
