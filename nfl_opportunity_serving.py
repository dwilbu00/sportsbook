"""nfl_opportunity_serving.py — live eligibility gate for the bonus optimizer.

The bonus optimizer only fields legs from TRUSTWORTHY props above an OPPORTUNITY threshold
(validated in nfl_opportunity_threshold.py: only high-volume roles are market-trustworthy). Offline
that gate reads nfl_data.player_week (mirror parquet). The live app can't (Streamlit Cloud has only
LFS stubs), so this module serves the SAME gate from the authoritative public source — nflverse —
fetched DIRECTLY as parquet with pandas (NO nflreadpy/polars: the runtime rule in nfl_ingest.py).

Self-healing: `_load(season)` fetches the nflverse release parquet and caches it with a TTL; a
re-entry after the TTL (or a new week) refetches. player_week for a public season is a REPLACEABLE
dataset (not the never-destroy class), so it is cached, not persisted to Azure.

Gate = recency-weighted expected VOLUME (the same stat + SWEPT half-life the backtest used):
  receptions -> targets (HL 4, min-prior 3);  rush_attempts -> carries (HL 4, min 4);
  pass_attempts -> attempts (HL 2, min 4).  expected_volume >= TRUSTWORTHY[prop] -> eligible.
Prior-season spillover fills the recency window early in a season (role carries ~across the break).
"""
import time

import pandas as pd

import nfl_props_scan as scan
import nfl_props_accuracy as acc

TRUSTWORTHY = {"player_receptions": 6.9, "player_rush_attempts": 8.5, "player_pass_attempts": 23.4}
VOL_STAT = {"player_receptions": "targets", "player_rush_attempts": "carries",
            "player_pass_attempts": "attempts"}
NFLVERSE_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                "stats_player/stats_player_week_{season}.parquet")
_KEEP = ["player_display_name", "season", "week", "team", "position",
         "targets", "carries", "attempts", "receptions"]
CACHE_TTL = 6 * 3600         # 6h — a season's weekly data changes at most once/week
_CACHE = {}                  # season(int) -> (fetch_ts, DataFrame|None)


def _load(season, ttl=CACHE_TTL):
    """Fetch one nflverse season's player-week parquet (cached, TTL). Self-healing:
    re-fetches once the cache is older than ttl. Returns a normalized DataFrame or None."""
    s = int(season)
    hit = _CACHE.get(s)
    if hit is not None and (time.time() - hit[0]) < ttl:
        return hit[1]
    try:
        df = pd.read_parquet(NFLVERSE_URL.format(season=s))
        cols = [c for c in _KEEP if c in df.columns]
        df = df[cols].copy()
        df["player_norm"] = df["player_display_name"].map(scan._norm)
        for c in ("targets", "carries", "attempts", "receptions"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            else:
                df[c] = 0.0
        _CACHE[s] = (time.time(), df)
        return df
    except Exception:
        return hit[1] if hit is not None else None   # serve stale on transient fetch failure


def _prior_games(player_norm, season, week, min_prior):
    """That player's prior games, most-recent first. SAME-SEASON prior weeks only when there are
    already >= min_prior of them (byte-for-byte the offline/backtest gate); otherwise spill into
    the prior season so early-season windows still get coverage instead of abstaining."""
    s = int(season)
    cur = _load(s)
    same = []
    if cur is not None:
        sub = cur[(cur["player_norm"] == player_norm) & (cur["week"] < int(week))]
        same = sub.sort_values("week", ascending=False).to_dict("records")
    if len(same) >= min_prior:
        return same                                 # matches offline exactly (same-season only)
    prev = _load(s - 1)                              # early-season fallback only
    if prev is not None:
        sub = prev[prev["player_norm"] == player_norm]
        same += sub.sort_values("week", ascending=False).to_dict("records")
    return same


def expected_volume(player_norm, season, week, prop):
    """Recency-weighted expected volume for the gate — identical stat + half-life to the
    backtest. None if the player has fewer than MIN_PRIOR prior games (gate abstains)."""
    if prop not in VOL_STAT:
        return None
    hl, min_prior, _k = acc.SWEPT.get(prop, (4, 3, 12))
    col = VOL_STAT[prop]
    prior = _prior_games(player_norm, season, week, min_prior)
    if len(prior) < min_prior:
        return None
    w = [0.5 ** (i / hl) for i in range(len(prior))]
    wsum = sum(w) or 1.0
    return sum(wi * float(r.get(col) or 0.0) for r, wi in zip(prior, w)) / wsum


def passes_gate(player_norm, season, week, prop):
    """True iff prop is trustworthy AND the player's expected volume clears its threshold."""
    tmin = TRUSTWORTHY.get(prop)
    if tmin is None:
        return False
    ev = expected_volume(player_norm, season, week, prop)
    return ev is not None and ev >= tmin


def _selftest(season, week):
    print(f"nfl_opportunity_serving self-test — {season} wk{week}")
    for prop, tmin in TRUSTWORTHY.items():
        df = _load(season)
        if df is None:
            print(f"  {prop}: nflverse load FAILED"); continue
        names = df[df["week"] < int(week)]["player_norm"].unique()
        vals = [(n, expected_volume(n, season, week, prop)) for n in names]
        elig = [(n, v) for n, v in vals if v is not None and v >= tmin]
        elig.sort(key=lambda x: -x[1])
        print(f"  {prop:<22} thr {tmin:>5} -> {len(elig):>3} eligible; "
              f"top: {', '.join(f'{n}={v:.1f}' for n, v in elig[:3])}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2024")
    ap.add_argument("--week", type=int, default=8)
    a = ap.parse_args()
    _selftest(a.season, a.week)
