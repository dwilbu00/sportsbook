"""Daily UNDER / F5-shape FLAGS — forward-tracking the DK-vs-Pinnacle-F5 total edge.

scenario_backtest.py --scenario f5_decomp validated (2024-26, in-sample) a game-SHAPE
mispricing: DK's full-game total MINUS Pinnacle's FIRST-5 total = DK's implied runs in
innings 6-9 (normal ~4.0-4.6). When DK prices in a LOT of late scoring (a big gap), the
game reliably stays UNDER at DK's full total; when it prices in little (small gap), it
tends OVER. The relationship is monotone, replicates 3/3 seasons, holds WITHIN a fixed
total line (not line-height), and DK's full total matches Pinnacle's exactly (no line
offset). Realized: UNDER at gap 4.5-5.0 = +21% (t=4.05), 4.0-4.5 = +12%; OVER at gap
<3.5 = +9.6% (thin, non-replicating). Same ~10% pooled as the flat-under bias, but the
gap picks WHERE it's strongest.

⚠ FORWARD-UNCONFIRMED + extraordinary magnitude + measured on 12-24h lines. This tool
exists to bet SMALL and grow a live record before trusting it. DK is the bet book;
Pinnacle F5 is the analysis reference (never bet).

USAGE
    python under_flags.py --live            # today's slate via Odds API (DK totals + Pin F5)
    python under_flags.py                    # today, from the warehouse (credit-free)
    python under_flags.py --date 2026-09-01
"""
import argparse
import datetime

# Flag zones from f5_decomp (bt = backtest ROI, in-sample 2024-26):
_UNDER_STRONG = (4.5, 5.0)     # +21% t=4.05, 3/3
_UNDER_LEAN = (4.0, 4.5)       # +12% t=3.18, 3/3
_OVER_SPEC_MAX = 3.5           # gap < 3.5 -> OVER +9.6% (thin n=63, NOT replicating)


def classify_gap(gap):
    """(side, strength) for a DK_full − Pin_F5 gap, or (None, None) to skip. Zones are
    the validated f5_decomp buckets; the weak middle (3.5-4.0) and tail (>=5.0) skip."""
    if gap is None:
        return None, None
    if _UNDER_STRONG[0] <= gap < _UNDER_STRONG[1]:
        return "UNDER", "STRONG (+21% bt, t=4.0)"
    if _UNDER_LEAN[0] <= gap < _UNDER_LEAN[1]:
        return "UNDER", "lean (+12% bt)"
    if gap < _OVER_SPEC_MAX:
        return "OVER", "speculative (+9.6% bt, thin/unreplicated)"
    return None, None


def _book_market(game, book_key, market_key):
    for b in game.get("bookmakers", []) or []:
        if b.get("key") == book_key:
            for m in b.get("markets", []) or []:
                if m.get("key") == market_key:
                    return m
    return None


def _total_from_market(m):
    """(point, over_price, under_price) from a totals market, or None if incomplete."""
    if not m:
        return None
    over = under = point = None
    for o in m.get("outcomes", []) or []:
        if o.get("name") == "Over":
            over, point = o.get("price"), o.get("point")
        elif o.get("name") == "Under":
            under = o.get("price")
    if point is None or over is None or under is None:
        return None
    return (point, over, under)


def flag_from_pairs(pairs):
    """PURE: pairs = list of dicts with event_id/home/away/dk (point,over,under)/pin_f5
    (point,over,under). Emit the +EV under/over flag per game (DK price on DK's full
    total). Sorted UNDER-strong first, then by gap distance from the normal ~4.6."""
    flags = []
    for p in pairs:
        dk, pf = p.get("dk"), p.get("pin_f5")
        if not dk or not pf:
            continue
        gap = dk[0] - pf[0]
        side, strength = classify_gap(gap)
        if side is None:
            continue
        price = dk[2] if side == "UNDER" else dk[1]
        flags.append({
            "event_id": p.get("event_id"), "home": p.get("home"), "away": p.get("away"),
            "commence_time": p.get("commence_time"),
            "dk_total": dk[0], "pin_f5_total": pf[0], "gap": round(gap, 2),
            "side": side, "strength": strength, "dk_price": price,
        })
    order = {"STRONG (+21% bt, t=4.0)": 0, "lean (+12% bt)": 1,
             "speculative (+9.6% bt, thin/unreplicated)": 2}
    return sorted(flags, key=lambda f: (order.get(f["strength"], 9), -f["gap"]))


def pairs_from_upcoming(games):
    """PURE: Odds-API upcoming JSON -> [{event_id, home, away, commence_time, dk, pin_f5}].
    dk = DraftKings FULL-game total; pin_f5 = Pinnacle first-5 total. Games missing either
    are dropped."""
    out = []
    for g in games or []:
        dk = _total_from_market(_book_market(g, "draftkings", "totals"))
        pf = _total_from_market(_book_market(g, "pinnacle", "totals_1st_5_innings"))
        if dk is None or pf is None:
            continue
        out.append({"event_id": g.get("id"), "home": g.get("home_team"),
                    "away": g.get("away_team"), "commence_time": g.get("commence_time"),
                    "dk": dk, "pin_f5": pf})
    return out


def load_live(sport):
    """One Odds-API call for DK full totals + Pinnacle first-5 totals for the slate."""
    import odds_client
    from backfill_historical_odds import load_config
    api_key = load_config()["odds_api_key"]
    games = odds_client.get_upcoming_odds(
        api_key, sport, regions="us", markets="totals,totals_1st_5_innings",
        bookmakers=["draftkings", "pinnacle"])
    return pairs_from_upcoming(games)


def _close_total(rows):
    """Per event, the latest-captured<=commence (point, over, under) from total rows."""
    import r2_data
    by_event = {}
    for r in rows:
        if r.get("bet_type") != "total":
            continue
        eid = r.get("event_id")
        cap = r2_data._parse_ts(r.get("captured_at"))
        com = r2_data._parse_ts(r.get("commence_time"))
        if cap is None or com is None or cap > com:
            continue
        cur = by_event.get(eid)
        if cur is None or cap > cur[0]:
            by_event[eid] = (cap, {})
        slot = by_event[eid][1]
        if r.get("selection") == "Over":
            slot["point"], slot["over"] = r.get("point"), r.get("price")
        elif r.get("selection") == "Under":
            slot["under"] = r.get("price")
    out = {}
    for eid, (_, s) in by_event.items():
        if s.get("point") is not None and s.get("over") is not None and s.get("under") is not None:
            out[eid] = (s["point"], s["over"], s["under"])
    return out


def load_warehouse(sport, date):
    """Credit-free: latest captured DK full totals + Pinnacle F5 totals for one date."""
    import db_store
    db_store.promote_secrets_from_toml()
    dk_rows = [r for r in db_store.team_market_lines(sport, dates=[date], bookmaker="draftkings")
               if r.get("kind") == "team"]
    pin_rows = [r for r in db_store.team_market_lines(sport, dates=[date], bookmaker="pinnacle")
                if r.get("kind") == "first_five"]
    dk_close, pin_close = _close_total(dk_rows), _close_total(pin_rows)
    meta = {}
    for r in dk_rows:
        meta.setdefault(r.get("event_id"),
                        {"home": r.get("home"), "away": r.get("away"),
                         "commence_time": r.get("commence_time")})
    pairs = []
    for eid, dk in dk_close.items():
        pf = pin_close.get(eid)
        if pf is None:
            continue
        m = meta.get(eid, {})
        pairs.append({"event_id": eid, "home": m.get("home"), "away": m.get("away"),
                      "commence_time": m.get("commence_time"), "dk": dk, "pin_f5": pf})
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Daily UNDER / F5-shape flags (DK vs Pinnacle F5).")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--live", action="store_true",
                    help="Fetch DK totals + Pinnacle F5 via the Odds API (~2 credits) "
                         "instead of reading the warehouse.")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default today). Ignored with --live.")
    args = ap.parse_args()
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    date = args.date or datetime.date.today().isoformat()
    if args.live:
        pairs = load_live(args.sport)
        src = "live (Odds API: DK totals + Pinnacle F5)"
    else:
        pairs = load_warehouse(args.sport, date)
        src = f"warehouse {date}"

    print(f"\n  UNDER / F5-shape flags — {args.sport} — {src}")
    print(f"  gap = DK full total − Pinnacle F5 total (normal ~4.0-4.6); "
          f"games with both totals: {len(pairs)}")
    flags = flag_from_pairs(pairs)
    if not flags:
        print("  No flags today (no game in the UNDER 4.0-5.0 or OVER <3.5 gap zones)." if pairs
              else "  No games with both a DK full total and a Pinnacle F5 total.")
        return
    print(f"\n  {len(flags)} flag(s) — bet SMALL + track (forward-UNCONFIRMED, extraordinary):")
    for f in flags:
        print(f"    {f['away']} @ {f['home']:<24}  {f['side']:<5} "
              f"(DK {f['dk_total']} / PinF5 {f['pin_f5_total']}, gap {f['gap']:+.1f}) "
              f"@ {f['dk_price']:+d}   {f['strength']}")
    print("\n  Bet the UNDER/OVER at DraftKings' FULL-game total. Pinnacle = reference only.")
    print()


if __name__ == "__main__":
    main()
