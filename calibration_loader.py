"""
Persistent player-prop calibration storage and runtime lookup.

A calibration file lives at:
    SPORTSBOOK_ODDS/calibration/<sport_key>.json

It captures per-prop slow-moving config + a fitted residual distribution from
the current season, plus an embedded "warmup" block fitted on the prior
season's data. At runtime, early-season player projections blend the
current-season residual distribution with the warmup distribution based on
how many games the player has accumulated in the current season:

    w = min(player_current_season_games / warmup_games, 1.0)
    p_over = w * p_current + (1 - w) * p_warmup

JSON schema per prop entry:
{
  "method": "B",                       # A=empirical, B=pooled Gaussian, C=pooled ECDF
  "half_life": 15,
  "venue_strength": 0.25,
  "opp_defense_strength": 0.0,
  "output_def_strength": 1.0,
  "shrinkage_k": 10,
  "residual_mu": 0.05,
  "residual_sigma": 8.21,
  "residual_ecdf": [-25.0, -22.1, ...],  # sorted residuals from fit set
  "n_obs": 612,
  "warmup_games": 10,
  "warmup": {
      "method": "B",
      "residual_mu": ...,
      "residual_sigma": ...,
      "residual_ecdf": [...],
      "n_obs": ...
  }
}
"""
import json
import math
import os
from datetime import datetime

# Match backtest.py: months when each sport's season starts.
SPORT_SEASON_START_MONTH = {
    "basketball_nba": 10,
    "americanfootball_nfl": 9,
    "baseball_mlb": 3,
    "icehockey_nhl": 10,
}

CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "calibration")


def calibration_path(sport_key):
    return os.path.join(CALIBRATION_DIR, f"{sport_key}.json")


def load_calibration(sport_key):
    """
    Load calibration for a sport. Returns a dict of {prop_key: prop_cfg} or
    an empty dict if no calibration file exists for this sport.
    Failures are silent — analyzers fall back to defaults.
    """
    path = calibration_path(sport_key)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return {}
    return blob.get("props", {})


def _load_blob(sport_key):
    """Load the raw calibration blob (all top-level keys), or {} if missing."""
    path = calibration_path(sport_key)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_prob_shrink(sport_key):
    """
    Load per-market probability-shrink factors for team markets, e.g.:
        {"spreads": 0.25, "totals": 0.0}
    A factor s pulls the model probability toward 0.5: p' = 0.5 + s*(p-0.5),
    correcting overconfidence. Returns {} (→ analyzers use 1.0 = no shrink) if
    none configured. Failures are silent.
    """
    if not sport_key:
        return {}
    return _load_blob(sport_key).get("prob_shrink", {})


def save_calibration(sport_key, props_cfg, meta=None):
    """Persist a calibration blob; creates calibration/ if needed."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = {
        "sport_key": sport_key,
        "fit_timestamp": datetime.utcnow().isoformat() + "Z",
        "props": props_cfg,
    }
    if meta:
        blob["meta"] = meta
    with open(calibration_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


# ─── season-aware date helpers ────────────────────────────────────────────

def _season_start_iso(now=None, sport_key=None):
    """
    First day of the current season for `sport_key`. NBA/NHL: Oct 1 of the
    last year the season started. MLB: Mar 1. NFL: Sep 1. Default: Jan 1.
    """
    now = now or datetime.utcnow()
    start_month = SPORT_SEASON_START_MONTH.get(sport_key)
    if not start_month:
        return f"{now.year}-01-01"
    start_year = now.year if now.month >= start_month else now.year - 1
    return f"{start_year}-{start_month:02d}-01"


def count_current_season_games(game_dates, sport_key, now=None):
    """How many of the player's recent games fall inside the current season."""
    if not game_dates:
        return 0
    cutoff = _season_start_iso(now=now, sport_key=sport_key)
    return sum(1 for d in game_dates if d and d >= cutoff)


# ─── core calibration math ────────────────────────────────────────────────

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _empirical_cdf(sorted_vals, x):
    """Fraction of sorted_vals <= x. Returns 0.5 on empty input."""
    if not sorted_vals:
        return 0.5
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(sorted_vals)


def calibrate_prob(method, projection, line, residual_mu, residual_sigma,
                   residual_ecdf, empirical_over=None):
    """
    Convert (projection, line) → P(actual > line) using the chosen method.

      A → empirical_over passthrough (no residual correction).
      B → pooled Gaussian: actual ≈ projection + N(mu, sigma²).
      C → pooled ECDF on residuals.
    """
    if method == "A":
        if empirical_over is None:
            return 0.5
        return max(0.0, min(1.0, empirical_over))

    corrected = projection + (residual_mu or 0.0)

    if method == "B":
        sigma = residual_sigma or 1e-6
        if sigma <= 0:
            sigma = 1e-6
        z = (corrected - line) / sigma
        return _norm_cdf(z)

    if method == "C":
        ecdf = residual_ecdf or []
        # P(actual > line) = P(residual > line - corrected) = 1 - F(line - corrected)
        return 1.0 - _empirical_cdf(ecdf, line - corrected)

    # Unknown method → degrade to empirical or 0.5
    return max(0.0, min(1.0, empirical_over)) if empirical_over is not None else 0.5


def apply_calibration_with_warmup(prop_cfg, projection, line, current_season_games,
                                  empirical_over=None):
    """
    Apply the prop's calibration to (projection, line), blending the
    current-season fit with the warmup fit by the player's current-season
    game count.

        w = min(current_season_games / warmup_games, 1.0)
        p = w * p_current + (1 - w) * p_warmup

    Returns the calibrated probability of OVER, or None when no usable
    calibration is available for this prop.
    """
    if not prop_cfg:
        return None

    method = prop_cfg.get("method")
    if not method:
        return None

    p_curr = calibrate_prob(
        method, projection, line,
        prop_cfg.get("residual_mu"),
        prop_cfg.get("residual_sigma"),
        prop_cfg.get("residual_ecdf"),
        empirical_over=empirical_over,
    )

    warmup = prop_cfg.get("warmup")
    warmup_games = prop_cfg.get("warmup_games", 10)
    if not warmup or warmup_games <= 0:
        return p_curr

    p_warm = calibrate_prob(
        warmup.get("method", method), projection, line,
        warmup.get("residual_mu"),
        warmup.get("residual_sigma"),
        warmup.get("residual_ecdf"),
        empirical_over=empirical_over,
    )

    w = min(max(current_season_games, 0) / float(warmup_games), 1.0)
    return w * p_curr + (1.0 - w) * p_warm
