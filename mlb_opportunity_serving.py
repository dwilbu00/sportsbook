"""mlb_opportunity_serving.py — MLB opportunity projection for the bonus eligibility gate.

The MLB analog of nfl_opportunity_serving: recency-weighted EXPECTED VOLUME as-of a game, the
input the bonus gate thresholds on. Batter props gate on expected PLATE APPEARANCES; pitcher
props gate on expected OUTS (IP*3). Read straight from the StatsAPI warehouse facts
(mlb_warehouse.get_player_history — most-recent-first, as-of aware), so it works for both live
serving (as_of_date=today) and the threshold backtest (as_of_date=game date).

Thresholds themselves are set by the backtest (mlb_opportunity_threshold.py, Brier-gap-to-market,
like the NFL gate) — not hard-coded here. This module only produces the number.
"""
BATTER_PROPS = ("batter_hits", "batter_total_bases", "batter_strikeouts", "batter_rbis")
PITCHER_PROPS = ("pitcher_strikeouts", "pitcher_earned_runs", "pitcher_outs")

# SWEPT-style recency per prop family (half_life, min_prior). Pitchers start ~weekly so a
# shorter window/looser min; batters play ~daily. Tune in the threshold backtest.
_RECENCY = {"batter": (5, 4), "pitcher": (3, 3)}


def _family(prop_key):
    return "batter" if prop_key.startswith("batter_") else "pitcher"


def _ip_to_outs(ip):
    try:
        from espn_client import ip_to_outs
        return ip_to_outs(ip)
    except Exception:
        return None


def opportunity_from_rows(prior_rows, prop_key):
    """Recency-weighted expected volume from a list of prior warehouse gamelog ROW DICTS
    (most-recent-first) — the backtest path (book_line_calibration attaches `prior_games`).
    Same formula as expected_opportunity: batters -> PA (AB+BB+HBP+SF+SH), pitchers -> OUTS
    (ip_to_outs(IP)). None if too few priors."""
    fam = _family(prop_key)
    hl, min_prior = _RECENCY[fam]
    rows = list(prior_rows or [])
    if len(rows) < min_prior:
        return None

    def _vol(r):
        if fam == "batter":
            return sum(float(r.get(k) or 0.0) for k in ("AB", "BB", "HBP", "SF", "SH"))
        return float(_ip_to_outs(r.get("IP")) or 0.0)

    vals = [_vol(r) for r in rows]
    w = [0.5 ** (i / hl) for i in range(len(vals))]
    return sum(wi * v for v, wi in zip(vals, w)) / (sum(w) or 1.0)


def expected_opportunity(mlb_player_id, prop_key, as_of_date=None, season=None):
    """Recency-weighted expected volume for the gate: PA for batters, OUTS for pitchers.
    None if the player has < min_prior prior games or the warehouse can't serve."""
    try:
        import mlb_warehouse
    except Exception:
        return None
    fam = _family(prop_key)
    hl, min_prior = _RECENCY[fam]
    # Pitchers: read the OUTS history (pitcher_outs values = outs) as the volume proxy for every
    # pitcher prop. Batters: any batter prop's history carries plate_appearances.
    read_prop = "pitcher_outs" if fam == "pitcher" else prop_key
    h = mlb_warehouse.get_player_history(mlb_player_id, read_prop, n=20,
                                         as_of_date=as_of_date, season=season)
    if not h:
        return None
    vol = h.get("values") if fam == "pitcher" else h.get("plate_appearances")
    vol = [float(v) for v in (vol or []) if v is not None]
    if len(vol) < min_prior:
        return None
    w = [0.5 ** (i / hl) for i in range(len(vol))]
    return sum(wi * v for v, wi in zip(vol, w)) / (sum(w) or 1.0)


def _selftest(season):
    """Spot-check expected PA/outs for a few resolvable players from this season's warehouse."""
    import mlb_warehouse
    try:
        import mlb_starters
    except Exception:
        mlb_starters = None
    names = [("Aaron Judge", "batter_hits"), ("Shohei Ohtani", "batter_total_bases"),
             ("Tarik Skubal", "pitcher_strikeouts"), ("Zack Wheeler", "pitcher_outs")]
    print(f"mlb_opportunity_serving self-test — season {season}")
    for name, prop in names:
        mlb_id = None
        if mlb_starters is not None:
            try:
                mlb_id = mlb_starters.resolve_mlbam_id(name, season, prop_key=prop)
                if isinstance(mlb_id, (tuple, list)):   # returns (mlbam_id, flag)
                    mlb_id = mlb_id[0]
            except Exception:
                mlb_id = None
        if not mlb_id:
            print(f"  {name:18} {prop:20} — could not resolve MLBAM id")
            continue
        ev = expected_opportunity(mlb_id, prop, season=season)
        unit = "PA" if prop.startswith("batter_") else "outs"
        print(f"  {name:18} {prop:20} id={mlb_id} exp={('%.1f %s' % (ev, unit)) if ev else 'None'}")


if __name__ == "__main__":
    import argparse
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026")
    a = ap.parse_args()
    _selftest(a.season)
