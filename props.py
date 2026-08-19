"""Player-prop pricing.

analyze_player_props_value (the projection + calibration + value gate for player
props) and its formatting/report helper, plus the MLB matchup/lineup multipliers
and the per-sport prop tuning knobs. Split out of analysis.py (P3). A clean
sibling of analysis (margin) and parlay: it depends only downward on stats /
pricing_common / odds_client / calibration_loader / recalibration / prop_filter.
analysis re-exports these names for backward compatibility.
"""

import os

from calibration_loader import (
    apply_calibration_with_warmup,
    count_current_season_games,
    load_calibration,
    load_lineup_adjustment,
    load_value_gate,
)
from odds_client import PROP_LABELS
from park_factors import PROP_PARK_KIND, park_factor
from pricing_common import (
    _expected_roi,
    _opponent_defense_multiplier,
    _prop_is_value,
    _resolve_team_defense,
    _starter_adjustment,
    _venue_match_multiplier,
    et_local_date,
)
from prop_filter import filter_player_gamelog
import prop_features  # §2.6 candidate-feature registry (rest/days-off, …)
from recalibration import (
    apply_platt,
    load_recalibration,
    log_prediction,
    log_prediction_rows,
    maybe_auto_refit,
)
from stats import (
    _half_life_for,
    _normal_inv_cdf,
    _recency_weights,
    _weighted_mean,
    _weighted_rate,
    _weighted_std,
    hits_at_least,
    negbin_at_least,
)


# P4 fail-closed identity enforcement (default OFF). When on, an MLB player the
# entity resolver can't UNIQUELY pin (unknown / namesake-ambiguous) gets NO
# prediction — betting a mis-identified player is worse than skipping. OFF = the P3
# shadow posture (the prediction still logs with a NULL id). Env-gated so the blast
# radius (`prediction_log WHERE sport_key='baseball_mlb' AND player_mlb_id IS NULL`)
# is understood + resolvable SFBB gaps fixed before flipping.
_MLB_ENFORCE_IDENTITY_ENV = "ODI_MLB_ENFORCE_IDENTITY"

# Circuit breaker: enforcement drops UNPINNED players, but a SYSTEMIC failure (a SQL
# blip, or the SFBB id-map being rebuilt) makes the resolver return resolved=False
# for EVERYONE — which would silently suppress the whole slate. Above this unpinned
# fraction, treat it as systemic and fail OPEN for the run. Normal rate is ~2-3%.
_ENFORCE_FAILOPEN_FRACTION = 0.5


def _identity_enforced():
    return os.environ.get(_MLB_ENFORCE_IDENTITY_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


# MLB league baselines used as log5-style denominators for the props matchup
# multiplier. Priors (not fitted); refresh occasionally.
_MLB_LEAGUE = {
    "k_pct": 0.222,
    "ba": 0.243,
    "ops": 0.711,
}

# Lazily-populated lineup-adjustment cache. Mutated in place by tests through the
# analysis namespace, which re-exports this exact object.
_LINEUP_ADJ_CACHE = {}


def _output_defense_multiplier(opp_pa, league_avg_def, strength):
    """Output-side opponent-defense multiplier for a player-prop projection.

    Scales the projection by tonight's opponent's points-allowed relative to the
    league average (softer defense -> higher projection). Bounded to [0.5, 1.5]
    so a sparse/garbage `opp_pa` or a calibrated `strength` > 1 can't drive the
    projection to an extreme — or negative — value. Returns 1.0 when the inputs
    are missing or the feature is disabled (strength <= 0). With matchup_mult in
    [0.7, 1.4] and lineup_mult in [0.8, 1.2], the combined multiplier is then
    bounded to ~[0.28, 2.52] (> 0), so effective_line = line / combined_mult
    stays finite.
    """
    if not opp_pa or not league_avg_def or league_avg_def <= 0 or strength <= 0:
        return 1.0
    mult = 1.0 + strength * (opp_pa / league_avg_def - 1.0)
    return max(0.5, min(1.5, mult))


def _log5_rate(player_rate, opponent_rate, league_rate):
    """Combine two binary-event rates relative to their league environment."""
    if player_rate is None or opponent_rate is None or league_rate is None:
        return None
    clamp = lambda value: max(0.001, min(0.999, float(value)))
    player_rate, opponent_rate, league_rate = map(
        clamp, (player_rate, opponent_rate, league_rate))
    odds = ((player_rate / (1.0 - player_rate))
            * (opponent_rate / (1.0 - opponent_rate))
            / (league_rate / (1.0 - league_rate)))
    return odds / (1.0 + odds)


def _mlb_prop_matchup_mult(prop_key, upcoming_is_home, matchup_features, weight,
                           player_context=None):
    """
    Bounded projection multiplier for an MLB player prop based on the
    starter/opponent matchup (Phase 2). 1.0 = no change.

    Pitcher props scale by the OPPOSING lineup's quality vs the starter's hand;
    batter props scale by the OPPOSING starter's quality. When recent batter
    exposure is available, hits and strikeouts use a true log5 rate for the
    projected starter's workload and a neutral rate for the remaining bullpen
    workload. `weight` (0..1) is the calibratable fraction of the raw ratio to
    apply. Bullpen prop rates stay neutral until an as-of history can validate
    them without leakage.
    """
    if not matchup_features or upcoming_is_home is None or not weight:
        return 1.0
    side = "home" if upcoming_is_home else "away"
    opp_side = "away" if upcoming_is_home else "home"
    raw = 1.0

    if prop_key in ("pitcher_strikeouts", "pitcher_outs", "pitcher_earned_runs"):
        sd = matchup_features.get(side) or {}
        opp = sd.get("opp_offense_vs_hand")
        if not opp:
            return 1.0
        if prop_key == "pitcher_strikeouts" and opp.get("k_pct"):
            raw = opp["k_pct"] / _MLB_LEAGUE["k_pct"]           # whiff-prone lineup → more Ks
        elif opp.get("ops"):
            r = opp["ops"] / _MLB_LEAGUE["ops"]
            # Better opposing offense → more earned runs, fewer outs recorded.
            raw = r if prop_key == "pitcher_earned_runs" else (2.0 - r)
    elif prop_key in ("batter_hits", "batter_strikeouts"):
        opp_sd = matchup_features.get(opp_side) or {}
        stp = opp_sd.get("starter")
        if not stp:
            return 1.0
        context = player_context or {}
        base_projection = context.get("base_projection")
        exposure = context.get("expected_exposure")
        if base_projection and exposure:
            batter_rate = base_projection / exposure
            avg_ip = stp.get("avg_ip")
            starter_share = max(0.10, min(0.80, (avg_ip or 5.5) / 9.0))
            if prop_key == "batter_strikeouts":
                league_rate = _MLB_LEAGUE["k_pct"]
                starter_k_pct = stp.get("k_pct")
                if stp.get("bf") is not None and stp["bf"] < 50:
                    starter_k_pct = None
                starter_rate = _log5_rate(
                    batter_rate, starter_k_pct, league_rate)
                bullpen_rate = _log5_rate(
                    batter_rate, league_rate, league_rate)
            else:
                league_rate = _MLB_LEAGUE["ba"]
                starter_rate = _log5_rate(
                    batter_rate, stp.get("xba"), league_rate)
                bullpen_rate = _log5_rate(
                    batter_rate, league_rate, league_rate)
            if starter_rate is not None and bullpen_rate is not None:
                matchup_rate = (starter_share * starter_rate
                                + (1.0 - starter_share) * bullpen_rate)
                raw = exposure * matchup_rate / base_projection
        elif prop_key == "batter_hits" and stp.get("xba"):
            raw = stp["xba"] / _MLB_LEAGUE["ba"]
        elif (prop_key == "batter_strikeouts" and stp.get("k_pct")
              and (stp.get("bf") is None or stp["bf"] >= 50)):
            raw = stp["k_pct"] / _MLB_LEAGUE["k_pct"]

    mult = 1.0 + weight * (raw - 1.0)
    return max(0.7, min(1.4, mult))


def _lineup_exposure_mult(expected_exposure, batting_order, weight,
                          slot_expected_exposure):
    """Blend recent opportunity with a batting-slot expectation."""
    if not expected_exposure or not batting_order or not weight:
        return 1.0
    try:
        slot_exposure = float(slot_expected_exposure[str(int(batting_order))])
        expected_exposure = float(expected_exposure)
        weight = float(weight)
    except (KeyError, TypeError, ValueError):
        return 1.0
    if expected_exposure <= 0 or slot_exposure <= 0:
        return 1.0
    adjusted_exposure = (expected_exposure
                         + weight * (slot_exposure - expected_exposure))
    return max(0.8, min(1.2, adjusted_exposure / expected_exposure))


def _mlb_lineup_exposure_mult(prop_key, player_context):
    """Return the validated batting-order multiplier, failing closed to 1.0."""
    if prop_key != "batter_hits" or not player_context:
        return 1.0
    sport_key = "baseball_mlb"
    if sport_key not in _LINEUP_ADJ_CACHE:
        try:
            _LINEUP_ADJ_CACHE[sport_key] = load_lineup_adjustment(sport_key) or {}
        except Exception:
            _LINEUP_ADJ_CACHE[sport_key] = {}
    cfg = _LINEUP_ADJ_CACHE[sport_key]
    if not cfg or cfg.get("enabled") is False:
        return 1.0
    weights = cfg.get("props") or {}
    slot_exposure = ((cfg.get("slot_expected_exposure") or {})
                     .get(prop_key) or {})
    weight = weights.get(prop_key)
    if not isinstance(weight, (int, float)):
        return 1.0
    return _lineup_exposure_mult(
        player_context.get("expected_exposure"),
        player_context.get("batting_order"),
        weight,
        slot_exposure,
    )


# Per-sport strength of the opponent-defense multiplier for player props.
# 0.0 disables it. Tuned per backtest:
#   - NBA: 0.0 (no measurable effect; adds noise — backtest sweep showed
#     MAE delta < 0.001 and Hit% slightly worse with defense weighting)
PLAYER_PROP_DEFENSE_STRENGTH = {
    "basketball_nba": 0.0,
}
DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH = 1.0


# Per-sport strength of the venue-match multiplier for player props (separate
# from team-level VENUE_MATCH_WEIGHTS because individual players' stat lines
# are less venue-sensitive than team-level scoring). 0.0 disables it.
# Tuned per backtest sweep on 18 NBA starters × 60 games:
#   - NBA: 0.0 — combined sweep showed ven=0.25 worsens MAE by ~0.006 and
#     leaves hit-rate essentially flat (52.82% → 52.87%).
PLAYER_PROP_VENUE_STRENGTH = {
    "basketball_nba": 0.0,
}
DEFAULT_PLAYER_PROP_VENUE_STRENGTH = None  # None = inherit from VENUE_MATCH_WEIGHTS


# Per-sport strength of the OUTPUT-side opponent-defense adjustment for
# player props. Unlike PLAYER_PROP_DEFENSE_STRENGTH (which down/up-weights
# *prior* games against tough defenses), this multiplier scales the final
# projection up/down based on TONIGHT's opponent's defense.
#   projection *= 1 + strength * (opp_pts_allowed / league_avg − 1)
# Tuned per backtest sweep on 18 NBA starters × 60 games:
#   - NBA: 1.0 — best single-feature gain: MAE 3.774 → 3.751, Hit% +2.79pp,
#     bias drops +0.086 → +0.060.
PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH = {
    "basketball_nba": 1.0,
}
DEFAULT_PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH = 0.0  # off by default for unknown sports


# Per-sport toggle for confirmed-lineup / probable-starter gating of player props
# (§2.5A). When a player is high-confidence NOT playing in the prop's role (a
# benched batter once BOTH lineups are posted, or a pitcher who isn't the
# announced starter) the prop is demoted from recommendations AND skipped from the
# calibration log — a dead bet whose label would never resolve. Tri-state and
# FAIL-OPEN: only a confident "out" acts; "in"/"unknown" leave behavior unchanged.
# MLB-only — lineup/probable data comes from mlb_starters; other sports fall
# through untouched (no lineup_status is ever set for them).
PLAYER_PROP_LINEUP_GATING = {
    "baseball_mlb": True,
}
DEFAULT_PLAYER_PROP_LINEUP_GATING = False


# Bayesian shrinkage of the recency-weighted projection toward the unweighted
# (season-long) prior mean. `k` is in pseudo-observations:
#   projection = (eff_n * weighted_mean + k * unweighted_mean) / (eff_n + k)
# Regularizes the projection so small-sample / volatile players aren't over-fit
# to their most recent few games.
# Tuned per backtest on 18 NBA starters × 60 games:
#   - NBA: 10 — Combined MAE 3.751 → 3.742 (−0.009), monotonic improvement
#     k ∈ {3,5,10}. Negligible cost. Hit% essentially unchanged.
PLAYER_PROP_SHRINKAGE_K = {
    "basketball_nba": 10,
}
DEFAULT_PLAYER_PROP_SHRINKAGE_K = 0  # off by default for unknown sports


# ── Market-as-prior shrinkage (P1.1a) ──
# Distinct from the *projection* shrinkage above. That one regularizes the mean
# stat toward the season prior; this one regularizes the final OVER *probability*
# toward the de-vigged market OVER probability (the sharp consensus prior):
#   p_shrunk = w * p_model + (1 - w) * p_market_novig,   w = n / (n + k)
# where n is the player's sampled game count. A ~15-game recency-weighted
# over-rate has SE ≈ ±13pp — larger than most flagged edges — so thin samples
# should lean on the market and only large samples / large deviations survive.
# `k` is in pseudo-observations (0 = off). Tuned per prop by the props-odds
# backtest (backtest.py --mode props-odds); persisted per-prop into
# calibration/<sport>.json as "market_prior_k".
PLAYER_PROP_MARKET_PRIOR_K = {}
DEFAULT_PLAYER_PROP_MARKET_PRIOR_K = 0  # off by default until tuned per backtest


# ── Park-factor road-context delta (P1.2) ──
# Strength of the ballpark adjustment applied to the projection. The adjustment
# scales by how the upcoming park's factor compares to the weighted-average park
# factor of the player's recent games (which already embed his home park — so a
# DELTA, never an absolute multiply). Only props in park_factors.PROP_PARK_KIND
# (batter_hits, pitcher_earned_runs) get a non-neutral effect; everything else
# is 1.0 regardless of strength. Ships ON for MLB (park factors are a known
# physical effect the market prices; aligning the projection trims false
# Coors-driven edges). 0.0 disables. Per-prop override via calibration JSON
# "park_factor_strength".
PLAYER_PROP_PARK_STRENGTH = {"baseball_mlb": 1.0}
DEFAULT_PLAYER_PROP_PARK_STRENGTH = 0.0  # off for sports with no park table
# Bounds on the final park multiplier (caps extreme parks like Coors runs).
PARK_FACTOR_BOUNDS = (0.85, 1.20)


# ── Weather / wind adjustment (P1.3) ──
# Strength of the pre-game weather nudge applied to the projection. Unlike park
# factors this is an ABSOLUTE, baseline-relative adjustment (vs 70 F / no wind):
# a player's ~15-game sample spans random weather, so its mean already reflects
# typical conditions — no road-context delta or double-count. The per-game
# forecast (temperature + out/in-to-CF wind) is fetched by the caller
# (weather_factors.get_game_weather) and passed in; only props in
# park_factors.PROP_PARK_KIND (batter_hits, pitcher_earned_runs) move. This
# CANNOT be validated on the projection backtest (no historical weather stored),
# so it ships ON at a CONSERVATIVE strength and is gated on CLV as forward data
# accrues (like the sharp-book weights). 0.0 disables; per-prop override via
# calibration JSON "weather_factor_strength".
PLAYER_PROP_WEATHER_STRENGTH = {"baseball_mlb": 0.5}
DEFAULT_PLAYER_PROP_WEATHER_STRENGTH = 0.0  # off for sports with no park geo
# Bounds on the final weather multiplier (caps extreme temp/wind forecasts).
WEATHER_FACTOR_BOUNDS = (0.88, 1.15)
WEATHER_BASELINE_TEMP_F = 70.0
# Per stat kind (park_factors.PROP_PARK_KIND): (fraction per F above baseline,
# fraction per mph of out-to-CF wind). Judgment priors from standard sabermetric
# rules of thumb — conservative, knob-tunable, refine on CLV. Runs (earned runs)
# is more weather-sensitive than a single hit, so it carries larger coefficients.
_WEATHER_COEF = {
    "hits": (0.0010, 0.0030),
    "runs": (0.0015, 0.0060),
}


# ── Statcast expected-BA blend (P2.4a) ──
# Shrinks the batter_hits projection toward the player's season-to-date Statcast
# expected BA (xBA × recent AB/game). xBA (quality-of-contact) stabilizes in far
# fewer PA than a raw ~15-game hit rate, so this trims the small-sample false
# edges the market prices sharply. Read from the durable `statcast_asof` SQL table
# (built offline from the raw pitch cache). `strength` is the LINEAR blend weight
# w∈[0,1]: base_proj = (1-w)·base + w·(xBA×AB/game). Gated on ≥ XSTATS_MIN_N
# official ABs. Ships at 0 (OFF) until validated on the backtest holdout
# (`backtest_props.py --xstats`); per-prop override via calibration JSON
# "xstats_strength".
PLAYER_PROP_XSTATS_STRENGTH = {"baseball_mlb": 0.0}
DEFAULT_PLAYER_PROP_XSTATS_STRENGTH = 0.0
# Which props blend toward which Statcast stat. Only batter_hits ← xBA for v1
# (strikeouts ← whiff/CSW is §2.4b, pending a raw re-pull).
PROP_XSTATS_KIND = {"batter_hits": "xba"}
# Minimum official ABs behind the as-of xBA before it is trusted (mirrors MIN_BBE).
XSTATS_MIN_N = 40
# ── Rest / days-off candidate feature (§2.6) ──
# Strength of the bounded rest/days-off projection multiplier from
# prop_features.rest_multiplier (the ONE source shared with the offline
# confirmation gate + synthetic sweep). Sized by a single per-prop knob the gate
# writes (calibration JSON "rest_strength"). Default 0.0 EVERYWHERE — the live
# path is byte-identical to production until the offline refit adopts rest for a
# prop, at which point live behaves EXACTLY as the gate validated (same
# prop_features math, same projection-scale + line-shift). prop_features'
# registry restricts WHICH props it can move (pitcher_outs / pitcher_strikeouts /
# batter_hits), so a stray knob on any other prop no-ops.
PLAYER_PROP_REST_STRENGTH = {}  # no sport-level default-on; gate writes per-prop
DEFAULT_PLAYER_PROP_REST_STRENGTH = 0.0
# §2.2: props eligible for the Negative-Binomial count method ("E"). Low-count
# non-negative-integer outcomes where a continuous Gaussian residual (method B)
# misfits the floor. Limited to the currently-calibrated count props — the offline
# refit only re-selects methods for props already in calibration/<sport>.json.
# Widening this needs an espn_client.PROP_STAT_MAP entry + a synthetic refit first.
PROP_NEGBIN_ELIGIBLE = {
    "pitcher_strikeouts", "pitcher_outs", "pitcher_earned_runs", "batter_hits",
    # Total bases / RBI are low-count, over-dispersed per-game integer counts —
    # the same shape NegBin (method E) models well. Eligibility only makes E a
    # CANDIDATE in the real-line selection; it ships only if it clears the gate.
    "batter_total_bases", "batter_rbis",
}


# ── Recommendation filter: suppress model UNDER picks on cheap lines ──
# §2.4b-2 direction-split diagnostic (2026-07-27): a model-predicted UNDER on a
# batter_hits 0.5 line wins only ~43% out-of-sample (n=427) — below the ~52.4%
# breakeven at -110 — because "player goes hitless" is an anti-predictive,
# market-sharp event (recent cold form reverts). Such picks are never flagged as
# value; the candidate is still returned (for display), just not recommended.
# Map prop_key -> max line to block; raise the cap to widen, or drop the entry
# (or set {} ) to disable.
SUPPRESS_UNDER_MAX_LINE = {"batter_hits": 0.5}


def _suppress_under(prop_key, line):
    """True when a model UNDER pick on (prop_key, line) should be demoted from
    recommendations (is_value forced False). See SUPPRESS_UNDER_MAX_LINE."""
    cap = SUPPRESS_UNDER_MAX_LINE.get(prop_key)
    if cap is None:
        return False
    try:
        return float(line) <= cap
    except (TypeError, ValueError):
        return False


def _lineup_gating_enabled(sport_key):
    """True when confirmed-lineup / probable-starter gating applies (§2.5A)."""
    return PLAYER_PROP_LINEUP_GATING.get(
        sport_key, DEFAULT_PLAYER_PROP_LINEUP_GATING)


def _xstats_blend(base_proj, xba, ab_per_game, strength):
    """Linear blend of the projection toward the xBA-implied per-game mean.

    Returns (blended_proj, xstats_mean). Pure — the caller supplies the looked-up
    xBA and the player's recent AB/game. strength is clamped to [0, 1]."""
    w = max(0.0, min(1.0, strength))
    xstats_mean = xba * ab_per_game
    return (1.0 - w) * base_proj + w * xstats_mean, xstats_mean


# ── §2.4b-2 distributional batter_hits model ──
# Per-AB hit probability from a contact-quality composite -> binomial P(>=k
# hits). A distribution (unlike the mean-count method C, whose residuals cluster
# by integer outcome) responds smoothly to the rate at line 0.5, so xBA /
# quality-of-contact signal actually flows into P(>=1 hit). Ships OFF: activated
# only when a prop's calibration method is "D". League-average Statcast rates are
# the quality-nudge denominators (priors — refresh occasionally; not fitted).
_MLB_LEAGUE_QUALITY = {"hard_hit_pct": 0.39, "barrel_pct": 0.075}
_DIST_HARDHIT_COEF = 0.10          # weight on relative hard-hit%
_DIST_BARREL_COEF = 0.10           # weight on relative barrel%
_DIST_QADJ_BOUNDS = (0.85, 1.20)   # bound the quality nudge
_DIST_PAB_BOUNDS = (0.01, 0.85)    # bound the final per-AB hit prob


def _dist_p_over(r_emp, expected_ab, xba, hh, brl, rate_mult, exposure_mult,
                 line, xstats_strength, hardhit_coef=None, barrel_coef=None):
    """Pure contact-quality composite → binomial P(over) for batter_hits.

    Shared by the runtime (`_distributional_over_rate`) and the offline
    diagnostic (`book_line_calibration.project_distributional`) so the two can't
    drift. All Statcast inputs are optional (None → that term drops out):
      level       = (1-s)·r_emp + s·xBA               (s = xstats_strength, xba≠None)
      quality_adj = clamp(1 + a·(hh/LG-1) + b·(brl/LG-1))
      p_AB        = clamp(level · quality_adj · rate_mult)
      n           = round(expected_ab · exposure_mult); k = int(line)+1
      P(over)     = hits_at_least(k, n, p_AB)
    Returns (p_over, meta). ``expected_ab`` must be > 0."""
    a = _DIST_HARDHIT_COEF if hardhit_coef is None else hardhit_coef
    b = _DIST_BARREL_COEF if barrel_coef is None else barrel_coef
    s = max(0.0, min(1.0, xstats_strength or 0.0)) if xba is not None else 0.0
    level = (1.0 - s) * r_emp + s * xba if s else r_emp
    q_adj = 1.0
    lg_hh = _MLB_LEAGUE_QUALITY["hard_hit_pct"]
    lg_brl = _MLB_LEAGUE_QUALITY["barrel_pct"]
    if hh is not None and lg_hh > 0:
        q_adj += a * (hh / lg_hh - 1.0)
    if brl is not None and lg_brl > 0:
        q_adj += b * (brl / lg_brl - 1.0)
    lo_q, hi_q = _DIST_QADJ_BOUNDS
    q_adj = max(lo_q, min(hi_q, q_adj))
    lo_p, hi_p = _DIST_PAB_BOUNDS
    p_ab = max(lo_p, min(hi_p, level * q_adj * (rate_mult or 1.0)))
    n = max(1, int(round(expected_ab * (exposure_mult or 1.0))))
    k = int(line) + 1                     # half-integer lines: OVER ⇔ hits >= k
    p_over = hits_at_least(k, n, p_ab)
    meta = {
        "k": k,
        "n_ab_expected": n,
        "p_ab": round(p_ab, 4),
        "r_emp": round(r_emp, 4),
        "xba": round(xba, 3) if xba is not None else None,
        "xba_weight": s,
        "hard_hit_pct": round(hh, 3) if hh is not None else None,
        "barrel_pct": round(brl, 3) if brl is not None else None,
        "q_adj": round(q_adj, 3),
        "rate_mult": round(rate_mult or 1.0, 3),
        "exposure_mult": round(exposure_mult or 1.0, 3),
    }
    return p_over, meta


def _distributional_over_rate(prop_key, line, values, at_bats, weights,
                              rate_mult, exposure_mult, player_name,
                              commence_iso, cfg, xstats_strength, teams=None):
    """P(over) for batter_hits as a binomial survival, or (None, None) to fall
    back to the empirical over-rate (§2.4b-2).

    p_AB = clamp( level · quality_adj · rate_mult ), with
      level       = (1-s)·(weighted hits/AB) + s·xBA         (s = xstats_strength)
      quality_adj = 1 + a·(hard_hit%/LG - 1) + b·(barrel%/LG - 1)   (bounded)
    n = round(expected_AB · exposure_mult), k = int(line)+1, and
    P(>=k) = stats.hits_at_least(k, n, p_AB).

    Rate multipliers (opp-defense / starter matchup / park / weather) scale the
    per-AB hit RATE; the batting-order EXPOSURE multiplier scales n (AB count) —
    a binomial needs each on the right parameter. Fails OPEN (returns None) when
    the prop isn't whitelisted or no usable AB history exists. xBA + quality
    rates come from statcast_asof (fail-open to the empirical level / no nudge).
    Built from the raw per-game hits/AB — NOT base_proj — so it never double-
    counts the §2.4a mean blend."""
    if PROP_XSTATS_KIND.get(prop_key) != "xba":   # whitelist: batter_hits only
        return None, None
    # Weighted per-AB hit rate + weighted mean AB/game from the player's games.
    hits_w = ab_w = w_valid = 0.0
    for v, ab, w in zip(values, at_bats, weights):
        if ab is None or ab <= 0 or w is None or w <= 0 or v is None:
            continue
        hits_w += w * v
        ab_w += w * ab
        w_valid += w
    if ab_w <= 0 or w_valid <= 0:
        return None, None
    r_emp = hits_w / ab_w
    expected_ab = ab_w / w_valid
    if expected_ab <= 0:
        return None, None

    xba = hh = brl = None
    n_ab = 0
    try:
        import mlb_starters
        import statcast_asof
        season = int(str(commence_iso)[:4]) if commence_iso else None
        if season:
            pid_info = mlb_starters.find_player_id(player_name, season, teams=teams)
            if pid_info and pid_info[0] and not pid_info[1]:   # batter only
                rates = statcast_asof.get_rates(pid_info[0], season, "bat")
                if rates and (rates.get("n_ab") or 0) >= XSTATS_MIN_N:
                    xba = rates.get("xba")
                    hh = rates.get("hard_hit_pct")
                    brl = rates.get("barrel_pct")
                    n_ab = rates.get("n_ab") or 0
    except Exception:
        xba = hh = brl = None    # fail open — never block a rec on Statcast

    p_over, meta = _dist_p_over(
        r_emp, expected_ab, xba, hh, brl, rate_mult, exposure_mult, line,
        xstats_strength,
        hardhit_coef=cfg.get("dist_hardhit_coef"),
        barrel_coef=cfg.get("dist_barrel_coef"))
    meta["n_ab_sample"] = n_ab
    return p_over, meta


def _negbin_over_rate(avg_stat, mean_scale, dispersion, line):
    """§2.2 method "E": P(over) for a low-count integer prop as a Negative-Binomial
    survival, or None to fall back (fail-open).

    The count mean is the adjusted projection ``avg_stat`` (= base_proj *
    combined_mult, the SAME quantity method B calibrates), scaled by the offline-fit
    multiplicative bias ``mean_scale``; ``dispersion`` (phi >= 0) is the fitted
    over-dispersion (variance = mean + phi*mean^2). A half-integer line L maps to
    P(X >= int(L)+1), the same continuity convention the binomial method D uses.
    Returns None when the projection is unusable so the caller keeps the empirical
    over-rate."""
    try:
        mean = avg_stat * (mean_scale if mean_scale else 1.0)
    except (TypeError, ValueError):
        return None
    if mean is None or mean <= 0:
        return None
    k = int(line) + 1                     # OVER <=> count >= k (matches _dist_p_over)
    return negbin_at_least(k, mean, dispersion or 0.0)


def _resolve_line_bucket(prop_calib_cfg, line):
    """Resolve (bucket_key, bucket_dict) for a specific book line — the single
    source of truth for line -> ``line_methods`` bucket matching, shared by
    ``_method_cfg_for_line`` (base calibration) and the online recalibration
    composite key (``_composite_recal_key`` / ``_resolve_recal_cfg``). Both the
    fit side and the apply side derive their bucket from THIS function so they can
    never disagree.

    First-match on ``ln <= max_line`` (``max_line`` None/absent = the open-ended
    top bucket), mirroring the original ``_method_cfg_for_line`` loop. The
    canonical key is ``"top"`` for the open bucket, else ``f"le_{cap:g}"`` (e.g.
    ``"le_0.5"``). Returns ``(None, None)`` when there is no ``line_methods``, no
    matching bucket, or ``line`` is unusable."""
    if not prop_calib_cfg:
        return None, None
    buckets = prop_calib_cfg.get("line_methods")
    if not buckets:
        return None, None
    try:
        ln = float(line)
    except (TypeError, ValueError):
        return None, None
    for b in buckets:
        cap = b.get("max_line")
        if cap is None or ln <= cap:
            key = "top" if cap is None else f"le_{cap:g}"
            return key, b
    return None, None


def _method_cfg_for_line(prop_calib_cfg, line):
    """Resolve (method, method_cfg) for a specific book line.

    Data-gated line-conditional selection: when the offline refit has written a
    ``line_methods`` list on a prop cfg (a bucket earned its own method), pick the
    first bucket whose ``max_line`` covers ``line`` (max_line None/absent = the
    open-ended top bucket) and return its method MERGED OVER the pooled cfg —
    ``{**pooled, **bucket}``. The merge is essential: an INHERITED bucket is
    written as a bare ``{max_line, method}`` (no residuals), and an adopted B/C
    bucket carries only its own residuals (no warmup), so returning the bare
    bucket alone would strip the pooled residual_ecdf / warmup and collapse
    calibration (method C → constant 0.5). Merging lets an inherited bucket reuse
    the pooled residuals + warmup, and an adopted bucket override just the fields
    it supplies (its own residuals, or the D composite params). Falls back to the
    pooled cfg when there is no ``line_methods``, no matching usable bucket, or
    ``line`` is unusable — so a cfg without line_methods behaves exactly as
    before.

    Bucket matching is delegated to ``_resolve_line_bucket`` (shared with the
    online recalibration key) so fit and apply never diverge."""
    if not prop_calib_cfg:
        return None, prop_calib_cfg
    _, bucket = _resolve_line_bucket(prop_calib_cfg, line)
    if bucket is not None and bucket.get("method"):
        return bucket.get("method"), {**prop_calib_cfg, **bucket}
    # No line_methods, no usable/matching bucket, or a matched-but-malformed
    # bucket (no method) -> fall back to the pooled cfg, exactly as before.
    return prop_calib_cfg.get("method"), prop_calib_cfg


def _composite_recal_key(prop_key, line, line_methods):
    """Key for the online Platt recalibration map (``recalibration_<sport>.json``).

    The map is a FLAT ``{key: fit}`` dict. A prop WITHOUT ``line_methods`` keeps
    its bare ``prop_key`` (unchanged behavior). A prop WITH ``line_methods`` is fit
    and applied per line bucket under ``f"{prop_key}@{bucket_key}"`` (e.g.
    ``batter_hits@le_0.5``, ``batter_hits@top``). Returns ``None`` when the prop
    has ``line_methods`` but ``line`` resolves to no bucket — the fit side then
    skips that record and the apply side applies no recal, keeping the two sides
    symmetric. ``line_methods`` is the base-calibration cfg's list; a real
    ``prop_key`` never contains ``"@"``, so composite keys never collide with a
    bare prop key."""
    if not line_methods:
        return prop_key
    bkey, _ = _resolve_line_bucket({"line_methods": line_methods}, line)
    if bkey is None:
        return None
    return f"{prop_key}@{bkey}"


def _resolve_recal_cfg(recal_map, prop_key, line, prop_calib_cfg):
    """Select the online recal fit to apply for (prop, line) from the loaded map.

    - No ``line_methods`` on the prop -> the bare ``prop_key`` fit (flat, exactly
      today's behavior; covers all NBA/NFL and most MLB props).
    - ``line_methods`` present and this line's ``prop_key@bucket`` fit exists ->
      that fit.
    - ``line_methods`` present, this bucket has no fit, but SOME ``prop_key@…``
      composite key exists (a post-migration per-bucket map) -> ``None``: the
      bucket earned no valid map, so leave the base-calibration prob untouched
      rather than borrow another bucket's shrinkage.
    - ``line_methods`` present but the map has NO composite key for this prop (a
      pre-migration flat blob still in the overlay) -> fall back to the bare
      ``prop_key`` fit, so behavior is unchanged until the first per-bucket refit
      rewrites the overlay."""
    if not recal_map:
        return None
    line_methods = (prop_calib_cfg or {}).get("line_methods")
    if line_methods:
        bkey, _ = _resolve_line_bucket(prop_calib_cfg, line)
        if bkey is not None:
            hit = recal_map.get(f"{prop_key}@{bkey}")
            if hit is not None:
                return hit
        prefix = prop_key + "@"
        if any(k.startswith(prefix) for k in recal_map):
            return None
        # Pre-migration flat map: no composite keys for this prop yet.
    return recal_map.get(prop_key)


# Per-sport override for the recency half-life *for player props specifically*.
# When set, overrides the team-level RECENCY_HALF_LIFE for the player-prop
# projection. Zero disables exponential decay; None inherits from
# RECENCY_HALF_LIFE.
# MLB chronological 2024 holdout (20-game live window): no decay improved
# batter hits, batter strikeouts, and pitcher earned runs. Pitcher strikeouts
# and outs retain their separately calibrated half_lives from baseball_mlb.json.
# Tuned per backtest on 18 NBA starters × 60 games:
#   - NBA: 7 — Gives the lowest total safe-mode cushion@80% (11.58 vs 11.62
#     at hl=10) at a negligible MAE cost (+0.008, ~0.2%). Prioritized for
#     safe-mode usage. Team-level matchup analysis still uses hl=10.
PLAYER_PROP_HALF_LIFE = {
    "basketball_nba": 7,
    "baseball_mlb": 0,
}
DEFAULT_PLAYER_PROP_HALF_LIFE = None  # None = inherit from RECENCY_HALF_LIFE

# Per-sport DEFAULT history LOOKBACK WINDOW for player-prop projections — the number
# of most-recent prior games the projection is built from, applied BEFORE the
# reliability filter (mirrors the recency sweep's newest-N slice). A per-prop
# override lives in calibration[prop]["recent_n"] (read via _knob), where
# recent_n=null means FULL season (no cap), exactly like the sweep's "full" cell.
# These per-sport defaults reproduce the pre-sweep windows (MLB 20 / NBA 10 / NFL 8)
# so a prop with no calibrated recent_n is byte-identical to the old behavior.
# STEP-1 recency sweep (2026-08-19): MLB hits/batter-K/pitcher-K want FULL season,
# pitcher_ER ~20 — set per-prop in the calibration file, not here.
PLAYER_PROP_RECENT_N = {
    "basketball_nba": 10,
    "baseball_mlb": 20,
    "americanfootball_nfl": 8,
}
# Unknown/None sport → NO window cap (all 3 real sports are configured above; this
# default only hits tests that pass a history directly, or a not-yet-configured
# sport, where truncating below a reliability streak floor would be wrong).
DEFAULT_PLAYER_PROP_RECENT_N = None

# Per-sport player-prop history FETCH cap — a superset sized to cover the largest
# window any prop could slice to (a full season), so a full-season prop has its
# games available while a shorter-window prop just slices down. Kept SEPARATE from
# the team-market recent_n so lengthening a prop's window never touches team stats.
PROP_FETCH_LIMIT = {
    "baseball_mlb": 200,        # full MLB season + doubleheader/postseason headroom
    "basketball_nba": 100,
    "americanfootball_nfl": 25,
}
DEFAULT_PROP_FETCH_LIMIT = 100


def _player_prop_defense_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH
    return PLAYER_PROP_DEFENSE_STRENGTH.get(sport_key, DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH)


def _player_prop_recent_n(sport_key):
    """Per-sport DEFAULT history window (games) for player-prop projections. A
    per-prop calibration override (recent_n; null=full season) supersedes this via
    _knob. Preserves the pre-sweep per-sport window when unset."""
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_RECENT_N
    return PLAYER_PROP_RECENT_N.get(sport_key, DEFAULT_PLAYER_PROP_RECENT_N)


def prop_fetch_limit(sport_key):
    """How many most-recent games to FETCH per player for props — a superset that
    covers the largest window any prop slices to (a full season). Separate from the
    team-market recent_n so a longer prop window never changes team-stat fetches."""
    if sport_key is None:
        return DEFAULT_PROP_FETCH_LIMIT
    return PROP_FETCH_LIMIT.get(sport_key, DEFAULT_PROP_FETCH_LIMIT)


def _player_prop_venue_strength(sport_key):
    """
    Per-sport override for the venue-match multiplier *as applied to player
    props*. Returns None when the team-level VENUE_MATCH_WEIGHTS should be
    used (the historical default behavior).
    """
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_VENUE_STRENGTH
    return PLAYER_PROP_VENUE_STRENGTH.get(sport_key, DEFAULT_PLAYER_PROP_VENUE_STRENGTH)


def _player_prop_output_defense_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH
    return PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH.get(
        sport_key, DEFAULT_PLAYER_PROP_OUTPUT_DEFENSE_STRENGTH)


def _player_prop_shrinkage_k(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_SHRINKAGE_K
    return PLAYER_PROP_SHRINKAGE_K.get(sport_key, DEFAULT_PLAYER_PROP_SHRINKAGE_K)


def _player_prop_market_prior_k(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_MARKET_PRIOR_K
    return PLAYER_PROP_MARKET_PRIOR_K.get(sport_key, DEFAULT_PLAYER_PROP_MARKET_PRIOR_K)


def _player_prop_park_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_PARK_STRENGTH
    return PLAYER_PROP_PARK_STRENGTH.get(
        sport_key, DEFAULT_PLAYER_PROP_PARK_STRENGTH)


def _player_prop_weather_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_WEATHER_STRENGTH
    return PLAYER_PROP_WEATHER_STRENGTH.get(
        sport_key, DEFAULT_PLAYER_PROP_WEATHER_STRENGTH)


def _player_prop_xstats_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_XSTATS_STRENGTH
    return PLAYER_PROP_XSTATS_STRENGTH.get(
        sport_key, DEFAULT_PLAYER_PROP_XSTATS_STRENGTH)


def _player_prop_rest_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_REST_STRENGTH
    return PLAYER_PROP_REST_STRENGTH.get(
        sport_key, DEFAULT_PLAYER_PROP_REST_STRENGTH)


def _park_factor_mult(prop_key, past_parks, past_weights, upcoming_park,
                      strength):
    """Road-context ballpark multiplier for an MLB player-prop projection.

    Scales the projection by ``park_factor(upcoming) / park_factor(baseline)``,
    where the baseline is the recency-weighted average park factor of the
    player's recent games. Because his logs already embed the parks he played
    in (≈half at home), only the DIFFERENCE between tonight's park and his
    recent park environment should move the projection — this avoids
    double-counting his home park. 1.0 = no change.

    Args:
        prop_key: odds-API prop market key (maps to a stat kind via
            park_factors.PROP_PARK_KIND; unmapped props → no effect).
        past_parks: per-recent-game home-team park names (odds-API/ESPN
            spellings), aligned with ``past_weights``; None for a game whose
            park is unknown (excluded from the baseline).
        past_weights: the recency/venue/defense weights already computed for
            those games.
        upcoming_park: the upcoming game's home-team name (the park played in).
        strength: 0..1 fraction of the raw ratio to apply.

    Returns:
        (multiplier, meta_dict or None). Fails closed to (1.0, None) when the
        prop is park-neutral, strength<=0, the upcoming park is unknown, or no
        recent game has a known park.
    """
    kind = PROP_PARK_KIND.get(prop_key)
    if not kind or not strength or strength <= 0 or not upcoming_park:
        return 1.0, None
    num = 0.0
    den = 0.0
    for name, weight in zip(past_parks or [], past_weights or []):
        if not name or weight is None or weight <= 0:
            continue
        num += park_factor(name, kind) * weight
        den += weight
    if den <= 0:
        return 1.0, None
    pf_base = num / den
    pf_up = park_factor(upcoming_park, kind)
    if pf_base <= 0:
        return 1.0, None
    raw = pf_up / pf_base
    lo, hi = PARK_FACTOR_BOUNDS
    mult = max(lo, min(hi, 1.0 + strength * (raw - 1.0)))
    if mult == 1.0:
        return 1.0, None
    return mult, {
        "kind": kind,
        "pf_up": round(pf_up, 3),
        "pf_base": round(pf_base, 3),
        "mult": round(mult, 3),
    }


def _weather_factor_mult(prop_key, weather, strength):
    """Baseline-relative weather multiplier for an MLB player-prop projection.

    Scales the projection by tonight's forecast temperature + out/in-to-CF wind
    relative to a NEUTRAL baseline (70 F, no wind). This is an ABSOLUTE adjustment,
    not a road-context delta like park factors: a player's recent sample spans
    random weather, so its mean ≈ typical conditions and only tonight's departure
    from neutral should move the projection. 1.0 = no change.

    Args:
        prop_key: odds-API prop market key (maps to a stat kind via
            park_factors.PROP_PARK_KIND; unmapped props → no effect).
        weather: per-game forecast dict from weather_factors.get_game_weather
            ({"temp_f","wind_out_mph","dome",...}); None/empty/dome → no effect.
        strength: 0..1 fraction of the raw departure to apply.

    Returns:
        (multiplier, meta_dict or None). Fails closed to (1.0, None) when the prop
        is park-neutral, strength<=0, the game is a dome, or no usable forecast.
    """
    kind = PROP_PARK_KIND.get(prop_key)
    if not kind or not strength or strength <= 0 or not weather:
        return 1.0, None
    if weather.get("dome"):
        return 1.0, None
    temp_f = weather.get("temp_f")
    wind_out = weather.get("wind_out_mph")
    if temp_f is None and wind_out is None:
        return 1.0, None
    temp_coef, wind_coef = _WEATHER_COEF[kind]
    raw = 1.0
    if temp_f is not None:
        raw += temp_coef * (temp_f - WEATHER_BASELINE_TEMP_F)
    if wind_out is not None:
        raw += wind_coef * wind_out
    lo, hi = WEATHER_FACTOR_BOUNDS
    mult = max(lo, min(hi, 1.0 + strength * (raw - 1.0)))
    if mult == 1.0:
        return 1.0, None
    return mult, {
        "kind": kind,
        "temp_f": round(temp_f, 1) if temp_f is not None else None,
        "wind_out_mph": round(wind_out, 1) if wind_out is not None else None,
        "mult": round(mult, 3),
    }


def _player_prop_half_life(sport_key):
    """
    Per-sport override for the recency half-life applied to player props.
    Falls back to the team-level RECENCY_HALF_LIFE when not overridden.
    """
    if sport_key is None:
        return _half_life_for(None)
    override = PLAYER_PROP_HALF_LIFE.get(sport_key, DEFAULT_PLAYER_PROP_HALF_LIFE)
    return override if override is not None else _half_life_for(sport_key)


# DK-alone guard: when DK is the ONLY book quoting its line (odds_client sets
# market_implied_method == 'dk_selfdevig_fallback'), over_implied degrades to DK's
# own two-way de-vig, so the edge gate (model vs fair) and the EV gate (model vs DK
# price) collapse into one comparison. Require this multiple of the normal edge
# threshold before flagging value, so a miscalibrated model can't rubber-stamp a
# DK-only line with no independent market check.
_DK_SELFDEVIG_EDGE_MULT = 2.0


def _prop_gate_is_value(edge, expected_roi, prop_key, ev_floor, edge_floor,
                        suppress, legacy_threshold, dk_alone):
    """Recommendation-gate decision for one prop side (pure + unit-testable).

    Two gate shapes, selected by whether a calibrated ``ev_floor`` is present:
      * ROI-primary (value_gate configured): profit-led — take the bet when its
        EV at the DK price >= ``ev_floor`` AND its fair-market edge >= ``edge_floor``
        (a light sanity floor that the model agrees the bet beats the fair market).
        Backtested via refit_calibration --gate-diag; beats the legacy gate on
        realized holdout ROI (esp. long-odds / generous-DK-price bets a flat edge
        floor wrongly suppressed).
      * legacy edge-threshold gate (no value_gate): edge >= ``legacy_threshold``
        AND EV > 0 (see _prop_is_value / P1.1) — unchanged for untuned sports.

    ``dk_alone`` (DK is the sole book at this line → degraded de-vig, no
    independent market check) requires MORE margin in both shapes: the legacy gate
    doubles its edge threshold, and the ROI-primary gate doubles BOTH its EV floor
    and its edge floor by _DK_SELFDEVIG_EDGE_MULT — so a miscalibrated model can't
    rubber-stamp a DK-only line off a thin EV any more than off a thin edge. A prop
    in ``suppress`` is never value (e.g. a market that loses under every
    calibration method, pending a projection fix).
    """
    if prop_key in (suppress or ()):
        return False
    if ev_floor is not None:
        ev = ev_floor
        floor = (edge_floor or 0.0)
        if dk_alone:
            ev *= _DK_SELFDEVIG_EDGE_MULT
            floor *= _DK_SELFDEVIG_EDGE_MULT
        return (expected_roi is not None
                and expected_roi >= ev
                and edge >= floor)
    eff_threshold = (legacy_threshold * _DK_SELFDEVIG_EDGE_MULT if dk_alone
                     else legacy_threshold)
    return _prop_is_value(edge, eff_threshold, expected_roi)


def analyze_player_props_value(prop_data, player_histories, threshold_pct=5.0,
                               sport_key=None, team_defense=None, espn_teams=None,
                               safe_mode=False, safe_target=0.95,
                               team_schedules=None, matchup_features=None,
                               weather=None, confirmed_lineup=None,
                               probable_starters=None):
    """
    Compare player prop lines against historical stat values from ESPN.

    Parameters:
        prop_data (dict): Parsed player props from odds_client.parse_player_props()
        player_histories (dict): {player_name: {prop_key: stat_history_dict}}
        threshold_pct (float): Minimum edge to flag as value
        sport_key (str): Sport key for recency half-life selection
        team_defense (dict): Optional {team_display_name: avg_points_allowed}
            lookup. When provided, each historical game's weight is multiplied
            by an opponent-defense factor.
        espn_teams (dict): Optional {display_name: team_info} lookup. When
            provided, each player's upcoming home/away status is resolved
            (via team_id from their gamelog) and a venue-match multiplier is
            applied per past game.

    Returns:
        list: Value candidates for player props
    """
    candidates = []
    prediction_rows = []
    threshold = threshold_pct / 100.0
    # Persistent per-(sport, prop) calibration overrides the in-code defaults
    # when available; absent file → falls back to defaults below.
    calibration = load_calibration(sport_key) if sport_key else {}
    # Recommendation gate config. When a value_gate is calibrated, the gate is
    # ROI-primary (EV >= ev_floor AND edge >= edge_floor) + a per-prop suppress
    # list; absent → legacy edge-threshold gate (edge >= threshold AND EV > 0).
    # Backtested via refit_calibration --gate-diag; promoted like any calib block.
    value_gate = load_value_gate(sport_key) if sport_key else {}
    gate_ev_floor = value_gate.get("ev_floor")
    gate_edge_floor = value_gate.get("edge_floor", 0.0) or 0.0
    gate_suppress = set(value_gate.get("suppress") or [])

    # Self-updating Platt recalibration: on first call per process per sport,
    # resolve any past-game predictions to outcomes and refit Platt params
    # if enough new data has accumulated. Cheap if nothing to do.
    if sport_key:
        maybe_auto_refit(sport_key)
    recalibration = load_recalibration(sport_key) if sport_key else {}

    # Pull commence_time once so we can log this game's predictions for
    # later outcome resolution.
    commence_iso = prop_data.get("commence_time")
    # US-Eastern local date (a late US game's UTC date is one day ahead), so the
    # outcome resolver buckets the prediction under its official game date.
    log_game_date = et_local_date(commence_iso) if commence_iso else None

    def _knob(prop_key, name, default):
        cfg = calibration.get(prop_key) if calibration else None
        if cfg and name in cfg:
            value = cfg[name]
            # In calibration JSON, half_life=null explicitly means equal weighting
            # and recent_n=null explicitly means the FULL season (no window cap) —
            # so both must pass a null through rather than fall to the default. Other
            # null knobs continue to mean "use the default."
            if value is not None or name in ("half_life", "recent_n"):
                return value
        return default

    default_half_life = _player_prop_half_life(sport_key)
    default_def_strength = _player_prop_defense_strength(sport_key)
    default_venue_override = _player_prop_venue_strength(sport_key)
    default_output_def = _player_prop_output_defense_strength(sport_key)
    default_shrinkage_k = _player_prop_shrinkage_k(sport_key)
    default_market_prior_k = _player_prop_market_prior_k(sport_key)
    default_park_strength = _player_prop_park_strength(sport_key)
    default_weather_strength = _player_prop_weather_strength(sport_key)
    default_xstats_strength = _player_prop_xstats_strength(sport_key)
    default_rest_strength = _player_prop_rest_strength(sport_key)
    default_recent_n = _player_prop_recent_n(sport_key)

    # Per-prop max strength across defaults + any calibrated overrides — used
    # only to decide whether league-average defense needs to be computed.
    def _any_prop_uses(name, default_val):
        if calibration:
            for cfg in calibration.values():
                if (cfg.get(name) or 0) > 0:
                    return True
        return (default_val or 0) > 0

    league_avg_def = None
    if team_defense and (_any_prop_uses("opp_defense_strength", default_def_strength)
                         or _any_prop_uses("output_def_strength", default_output_def)):
        vals = [v for v in team_defense.values() if v]
        if vals:
            league_avg_def = sum(vals) / len(vals)

    # Reverse lookup: team_id -> display_name (for venue resolution).
    id_to_name = {}
    if espn_teams:
        id_to_name = {str(info.get("id")): name for name, info in espn_teams.items() if info.get("id")}

    # P4 model-input flip bridge: the warehouse history dict carries the player's
    # team as an MLBAM id + display NAME (team_name), NOT an ESPN id, so the
    # ESPN-id-keyed id_to_name / team_schedules lookups miss. Map display name →
    # ESPN schedule so a warehouse-sourced player still gets layoff filtering. Only
    # consulted when the ESPN-id lookup misses (i.e. MLB warehouse dicts), so
    # NBA/NFL and the ESPN path are byte-identical.
    name_to_schedule = {}
    if espn_teams and team_schedules:
        for name, info in espn_teams.items():
            sched = team_schedules.get(str(info.get("id"))) if info.get("id") else None
            if sched:
                name_to_schedule[name] = sched

    home_team_name = prop_data["home_team"]
    away_team_name = prop_data["away_team"]
    matchup = f"{away_team_name} @ {home_team_name}"

    # P3: resolve each odds-feed player NAME → (MLBAM id, game_pk) ONCE per event,
    # fail-closed, at the odds boundary (home/away/commence all in scope here).
    # The resolved ids are stamped onto the prediction rows below. P3 posture is
    # SHADOW: an unresolved MLB player keeps NULL ids (the signal). P4 ENFORCEMENT
    # (env-gated, below) flips that to fail-CLOSED: no prediction for a player we
    # can't uniquely pin. MLB only; cached across this event's prop loops.
    _is_mlb_props = str(sport_key or "").startswith("baseball_mlb")
    _enforce_identity = _is_mlb_props and _identity_enforced()
    _ident_cache = {}

    def _resolve_ident(pname, prop_key=None):
        # Commit C: the id-core is ROLE-PARTITIONED (a pitcher-prop name binds off the
        # announced probable, a batter-prop name off the confirmed lineup), so the
        # cache is keyed by (name, role) — a same-name cross-role pair (e.g. a two-way
        # player offered as both) must not share one cached id. role is the only
        # prop_key-derived input; None (no prop_key) is its own bucket.
        if not _is_mlb_props:
            return None
        role = (None if prop_key is None
                else ("P" if str(prop_key).startswith("pitcher_") else "B"))
        ckey = (pname, role)
        if ckey not in _ident_cache:
            try:
                import entity_resolver
                _ident_cache[ckey] = entity_resolver.resolve(
                    pname, sport_key, home_team_name, away_team_name,
                    game_date=log_game_date, commence=commence_iso,
                    prop_key=prop_key, confirmed_lineup=confirmed_lineup,
                    probable_starters=probable_starters)
            except Exception:               # never break analysis on a resolver hiccup
                _ident_cache[ckey] = None
        return _ident_cache[ckey]

    def _ident_pinned(pname, prop_key=None):
        """True if the resolver UNIQUELY pinned this MLB player (resolved +
        mlb_player_id) FOR THIS PROP'S ROLE. A systemic hiccup (_resolve_ident →
        None, e.g. an import break) counts as pinned = kept (a crash must not drop a
        player)."""
        ident = _resolve_ident(pname, prop_key)
        if ident is None:
            return True
        return bool(ident.get("resolved") and ident.get("mlb_player_id"))

    # P4 enforcement + circuit breaker (see _ENFORCE_FAILOPEN_FRACTION): pre-scan the
    # slate; if an implausible fraction of MLB players is unpinned, that's a systemic
    # failure (not per-player ambiguity) → fail OPEN for this run so the whole card
    # isn't silently suppressed (the NULL-id shadow rows reappear as the signal).
    _enforce_active = _enforce_identity
    if _enforce_identity:
        # Systemic-failure pre-scan. The id-core is role-partitioned (a name can be
        # pinnable as a pitcher but not a batter) and the per-row drop is per
        # (name, prop_key), so the breaker counts that same (name, prop_key) unit.
        _units = {(p, pk) for pk, _pl in prop_data.get("props", {}).items()
                  for p in _pl}
        _unpinned = sum(1 for p, pk in _units if not _ident_pinned(p, pk))
        if _units and _unpinned / len(_units) >= _ENFORCE_FAILOPEN_FRACTION:
            _enforce_active = False
            print(f"[identity-enforce] slate-level FAIL-OPEN: {_unpinned}/"
                  f"{len(_units)} MLB prop-players unresolved "
                  f"(>= {int(_ENFORCE_FAILOPEN_FRACTION * 100)}%) — likely systemic; "
                  f"enforcement skipped this run")

    for prop_key, players in prop_data.get("props", {}).items():
        # Resolve per-prop knobs: calibration overrides the in-code defaults
        # where present, else fall back to the per-sport defaults.
        half_life = _knob(prop_key, "half_life", default_half_life)
        defense_strength = _knob(prop_key, "opp_defense_strength", default_def_strength)
        venue_strength_override = _knob(prop_key, "venue_strength", default_venue_override)
        output_def_strength = _knob(prop_key, "output_def_strength", default_output_def)
        shrinkage_k = _knob(prop_key, "shrinkage_k", default_shrinkage_k)
        market_prior_k = _knob(prop_key, "market_prior_k", default_market_prior_k)
        park_strength = _knob(prop_key, "park_factor_strength", default_park_strength)
        weather_strength = _knob(prop_key, "weather_factor_strength", default_weather_strength)
        xstats_strength = _knob(prop_key, "xstats_strength", default_xstats_strength)
        rest_strength = _knob(prop_key, "rest_strength", default_rest_strength)
        # History window: newest N prior games feeding the projection (calibration
        # recent_n=null → full season; absent → per-sport default). STEP-1 sweep.
        recent_n = _knob(prop_key, "recent_n", default_recent_n)
        prop_calib_cfg = calibration.get(prop_key) if calibration else None

        for player_name, odds_info in players.items():
            # P4 fail-closed identity enforcement (env-gated): drop an MLB player the
            # resolver can't uniquely pin — no candidate, no prediction row. Default
            # OFF → the P3 shadow posture (a NULL-id prediction still emits).
            # _enforce_active folds in the systemic-failure circuit breaker above.
            if _enforce_active and not _ident_pinned(player_name, prop_key):
                continue
            line = odds_info["line"]
            over_implied = odds_info["over_implied"]
            under_implied = odds_info["under_implied"]
            over_price = odds_info["over_price"]
            under_price = odds_info["under_price"]
            over_book = odds_info.get("over_book")
            under_book = odds_info.get("under_book")
            # DraftKings executable prices (P1.1b): edge/EV use the best price
            # above; staking/display use these. None when DK didn't post the
            # prop at the consensus line → callers fall back to over/under_price.
            dk_over_price = odds_info.get("dk_over_price")
            dk_under_price = odds_info.get("dk_under_price")
            dk_over_book = odds_info.get("dk_over_book")
            dk_under_book = odds_info.get("dk_under_book")

            history = player_histories.get(player_name, {}).get(prop_key)
            if not history or not history.get("found") or not history.get("values"):
                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "prop": prop_key,
                    "prop_label": PROP_LABELS.get(prop_key, prop_key),
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "over_implied": round(over_implied * 100, 2),
                    "under_implied": round(under_implied * 100, 2),
                    "avg_stat": None,
                    "over_rate": None,
                    "games_sampled": 0,
                    "edge_pct": 0,
                    "direction": None,
                    "is_value": False,
                    "no_history": True,
                    "lineup_status": (history or {}).get("lineup_status"),
                })
                continue

            # values are ordered most-recent first (see espn_client.get_player_stat_history)
            values = history["values"]
            opponents = history.get("opponents") or [None] * len(values)
            past_home_aways = history.get("home_aways") or [None] * len(values)
            minutes = history.get("minutes") or [None] * len(values)
            game_dates = history.get("game_dates") or [None] * len(values)
            plate_appearances = (history.get("plate_appearances")
                                 or [None] * len(values))
            at_bats = history.get("at_bats") or [None] * len(values)
            player_team_id = history.get("team_id")
            # Present only on warehouse-sourced (MLB flip) dicts — the player's team
            # DISPLAY NAME, since team_id is then MLBAM (not an ESPN id).
            player_team_name_wh = history.get("team_name")

            # ── Reliability filter ──
            # Drop low-minutes games AND the 1-game-pre + 1-game-post window
            # around any layoff (≥3 missed team games for NBA/MLB, ≥2 for NFL).
            # Also flags the player as "currently fragile" → skip prediction
            # when their last actual game was excluded (still injured /
            # ramping up) or their last game had limited minutes.
            synthetic = [
                {"game_date": gd, "MIN": m, "_value": v, "_opp": o,
                 "_ha": ha, "_pa": pa, "_ab": ab}
                for v, o, ha, m, gd, pa, ab in zip(
                    values, opponents, past_home_aways, minutes, game_dates,
                    plate_appearances, at_bats)
            ]
            # STEP-1 recency window: keep the newest `recent_n` games BEFORE the
            # reliability filter (recent_n falsy → full season, no cap). `synthetic`
            # is most-recent-first (the fetch returns newest-first), so [:recent_n]
            # is the newest N — matching the sweep and the live gamelog[:n]. Slicing
            # PRE-filter (not post) keeps an unchanged-window prop byte-identical to
            # the old behavior even when the filter later drops games.
            if recent_n:
                synthetic = synthetic[:recent_n]
            team_schedule = None
            if team_schedules and player_team_id:
                team_schedule = team_schedules.get(str(player_team_id))
            if team_schedule is None and player_team_name_wh:   # MLBAM id → bridge by name
                # Tolerant match: the warehouse team_name (StatsAPI mlb_team.name)
                # can diverge from the ESPN spelling (e.g. "Athletics" vs "Oakland
                # Athletics"), so reuse the substring/last-word matcher used for
                # opponent defense rather than an exact .get (which would silently
                # drop layoff filtering for divergent teams — the ESPN path resolved
                # this by ESPN id). NOTE: the warehouse game_date is seconds-precise
                # while the ESPN schedule is minute-precise, a ~1-game layoff-bisect
                # skew that resolves when team schedules move to the warehouse too.
                team_schedule = _resolve_team_defense(
                    player_team_name_wh, name_to_schedule)
            # Half-life historically also set the healthy-game streak floor.
            # Removing MLB decay should change weighting only, not make a
            # previously-untrusted player eligible three games sooner. Preserve
            # the former MLB hl=7 threshold (8 games) for no-decay props;
            # calibrated pitcher half-lives continue to set their own floors.
            reliability_min_streak = None
            if sport_key == "baseball_mlb" and not half_life:
                reliability_min_streak = 8
            filt = filter_player_gamelog(
                synthetic, team_schedule, sport_key, half_life=half_life,
                min_streak=reliability_min_streak)

            if filt["skip_prediction"]:
                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "prop": prop_key,
                    "prop_label": PROP_LABELS.get(prop_key, prop_key),
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "over_implied": round(over_implied * 100, 2),
                    "under_implied": round(under_implied * 100, 2),
                    "avg_stat": None,
                    "over_rate": None,
                    "games_sampled": 0,
                    "edge_pct": 0,
                    "direction": None,
                    "is_value": False,
                    "no_history": False,
                    "skip_reason": filt["skip_reason"],
                })
                continue

            eligible = filt["eligible_games"]
            values = [g["_value"] for g in eligible]
            opponents = [g["_opp"] for g in eligible]
            past_home_aways = [g["_ha"] for g in eligible]
            plate_appearances = [g.get("_pa") for g in eligible]
            at_bats = [g.get("_ab") for g in eligible]

            # Resolve the player's upcoming home/away by matching their team NAME
            # to the home/away team names of the upcoming game. Prefer the
            # warehouse-provided name (MLB flip: team_id is MLBAM, not an ESPN id);
            # else reverse-map the ESPN id (unchanged ESPN path).
            upcoming_is_home = None
            player_team_name = player_team_name_wh
            if player_team_name is None and player_team_id and id_to_name:
                player_team_name = id_to_name.get(str(player_team_id))
            if player_team_name_wh:
                # Warehouse StatsAPI name may diverge from the odds spelling →
                # tolerant match (mirrors the schedule bridge above). Build the
                # side map from truthy names only, so a blank name can't crash /
                # spuriously substring-match in _resolve_team_defense.
                ha_map = {}
                if home_team_name:
                    ha_map[home_team_name] = True
                if away_team_name:
                    ha_map[away_team_name] = False
                side = _resolve_team_defense(player_team_name_wh, ha_map)
                if side is not None:
                    upcoming_is_home = side
            elif player_team_name == home_team_name:
                upcoming_is_home = True
            elif player_team_name == away_team_name:
                upcoming_is_home = False

            base_weights = _recency_weights(len(values), half_life)
            weights = []
            for bw, opp, past_h in zip(base_weights, opponents, past_home_aways):
                w = bw
                if team_defense and league_avg_def and defense_strength > 0:
                    w *= _opponent_defense_multiplier(
                        _resolve_team_defense(opp, team_defense),
                        league_avg_def, defense_strength)
                # Venue multiplier: a per-sport PLAYER_PROP_VENUE_STRENGTH of
                # 0.0 disables it; None (default) inherits the team-level
                # VENUE_MATCH_WEIGHTS. P2.1 parity fix: a NUMERIC override applies
                # the (1±s) half-spread the backtest validated (was silently using
                # the fixed 1.40/0.60 regardless of the selected magnitude).
                if venue_strength_override != 0.0:
                    w *= _venue_match_multiplier(
                        past_h, upcoming_is_home, sport_key,
                        strength=venue_strength_override)
                weights.append(w)

            # ── Output-side opponent-defense adjustment ──
            # Scales the projection up/down based on TONIGHT's opponent's
            # defense (independent from per-prior-game weighting above).
            # Backtest finding: this is the single best feature gain for NBA
            # player props (MAE −0.6%, hit-rate +2.79pp).
            output_def_mult = 1.0
            if (output_def_strength > 0 and team_defense and league_avg_def
                    and upcoming_is_home is not None):
                opp_name = away_team_name if upcoming_is_home else home_team_name
                output_def_mult = _output_defense_multiplier(
                    _resolve_team_defense(opp_name, team_defense), league_avg_def,
                    output_def_strength)

            # ── Bayesian shrinkage toward unweighted prior mean ──
            # Regularizes the recency-weighted estimate by `k` pseudo-obs.
            base_proj = _weighted_mean(values, weights)
            if shrinkage_k > 0 and values:
                unweighted_mean = sum(values) / len(values)
                eff_n = sum(weights) if weights else 0.0
                if eff_n + shrinkage_k > 0:
                    base_proj = ((eff_n * base_proj) + (shrinkage_k * unweighted_mean)) / (eff_n + shrinkage_k)

            # ── Statcast expected-BA blend (P2.4a) ──
            # Shrink base_proj toward the batter's season-to-date xBA-implied hits
            # (xBA × recent AB/game) — quality-of-contact stabilizes faster than
            # the noisy ~15-game rate. MLB batter_hits only; reads the durable
            # statcast_asof SQL table keyed by MLBAM id. Fails OPEN (unknown id /
            # n<XSTATS_MIN_N / SQL miss → no change). The park/weather/matchup
            # multipliers below still apply on the blended base_proj.
            # (No explicit sport gate: xstats_strength defaults to 0 for any sport
            # without a PLAYER_PROP_XSTATS_STRENGTH entry, and PROP_XSTATS_KIND
            # only whitelists MLB batter_hits — so this is a no-op elsewhere.)
            xstats_meta = None
            if (xstats_strength and xstats_strength > 0
                    and PROP_XSTATS_KIND.get(prop_key) and commence_iso):
                try:
                    import mlb_starters
                    import statcast_asof
                    season = int(str(commence_iso)[:4])
                    pid_info = mlb_starters.find_player_id(
                        player_name, season,
                        teams=(home_team_name, away_team_name))
                    if pid_info and pid_info[0] and not pid_info[1]:  # batter only
                        xba, n_ab = statcast_asof.get_batter_xba(pid_info[0], season)
                        ab_valid = [(ab, w) for ab, w in zip(at_bats, weights)
                                    if ab is not None and ab > 0 and w > 0]
                        if xba is not None and n_ab >= XSTATS_MIN_N and ab_valid:
                            ab_per_game = _weighted_mean(
                                [ab for ab, _ in ab_valid],
                                [w for _, w in ab_valid])
                            if ab_per_game and ab_per_game > 0:
                                base_before = base_proj
                                base_proj, xstats_mean = _xstats_blend(
                                    base_proj, xba, ab_per_game, xstats_strength)
                                xstats_meta = {
                                    "xba": round(xba, 3),
                                    "n_ab": n_ab,
                                    "ab_per_game": round(ab_per_game, 2),
                                    "proj_xstats": round(xstats_mean, 3),
                                    "weight": xstats_strength,
                                    "base_before": round(base_before, 3),
                                    "blended": round(base_proj, 3),
                                }
                except Exception:
                    xstats_meta = None  # fail open — never block a rec on xStats

            # ── MLB starter/opponent matchup multiplier (Phase 2) ──
            # Scales the projection by the log5-style matchup (opposing lineup
            # for pitcher props; opposing starter for batter props). No-op for
            # other sports or when features/weight are absent.
            matchup_mult = 1.0
            lineup_mult = 1.0
            if sport_key == "baseball_mlb":
                player_context = None
                exposures = (plate_appearances if prop_key == "batter_strikeouts"
                             else at_bats if prop_key == "batter_hits" else None)
                if exposures:
                    valid = [(value, weight) for value, weight in zip(exposures, weights)
                             if isinstance(value, (int, float)) and value > 0]
                    if valid:
                        expected_exposure = _weighted_mean(
                            [value for value, _ in valid],
                            [weight for _, weight in valid],
                        )
                        if expected_exposure > 0 and base_proj > 0:
                            player_context = {
                                "base_projection": base_proj,
                                "expected_exposure": expected_exposure,
                            }
                batting_order = history.get("batting_order")
                if batting_order and player_context:
                    player_context["batting_order"] = batting_order
                if matchup_features:
                    matchup_mult = _mlb_prop_matchup_mult(
                        prop_key, upcoming_is_home, matchup_features,
                        _starter_adjustment(sport_key, "props", prop_key),
                        player_context=player_context)
                lineup_mult = _mlb_lineup_exposure_mult(
                    prop_key, player_context)

            # ── Park-factor road-context delta (P1.2) ──
            # Scale by how the upcoming park differs from the player's recent
            # park environment. Each past game's park is the player's own park
            # when he was home, else the opponent's park; the upcoming park is
            # always the home team's. MLB batter_hits / pitcher_earned_runs
            # only (park_factors.PROP_PARK_KIND) — 1.0 for everything else.
            park_mult = 1.0
            park_meta = None
            if park_strength and park_strength > 0:
                past_parks = []
                for _opp, _past_h in zip(opponents, past_home_aways):
                    if _past_h is True:
                        past_parks.append(player_team_name)
                    elif _past_h is False:
                        past_parks.append(_opp)
                    else:
                        past_parks.append(None)
                park_mult, park_meta = _park_factor_mult(
                    prop_key, past_parks, weights, home_team_name,
                    park_strength)

            # ── Weather / wind adjustment (P1.3) ──
            # Baseline-relative (70 F / no wind) temperature + out/in-to-CF wind
            # nudge for MLB batter_hits / pitcher_earned_runs. `weather` is the
            # per-game forecast the caller (app.py) fetched via
            # weather_factors.get_game_weather; None / dome / no forecast → 1.0.
            weather_mult, weather_meta = _weather_factor_mult(
                prop_key, weather, weather_strength)

            # ── Rest / days-off candidate feature (§2.6) ──
            # Bounded projection multiplier from prop_features — the SAME source
            # the offline confirmation gate + synthetic sweep validate against, so
            # live behaves exactly as the gate measured. Default rest_strength=0.0
            # → 1.0 (no-op), keeping the live path byte-identical to production
            # until the refit adopts rest for this prop. projection_multiplier
            # enforces the registry applies-to filter (a stray knob on a
            # non-eligible prop no-ops), and is leakage-safe: only the eligible
            # prior games' dates (strictly before commence) feed it.
            rest_mult = 1.0
            if rest_strength and rest_strength > 0:
                rest_mult = prop_features.projection_multiplier(
                    prop_key, {"rest": rest_strength},
                    [g.get("game_date") for g in eligible], commence_iso)

            combined_mult = (output_def_mult * matchup_mult
                             * lineup_mult * park_mult * weather_mult * rest_mult)

            avg_stat = base_proj * combined_mult
            # When the projection is scaled, the over-rate calc shifts the
            # comparison line by the inverse so historical frequencies are
            # interpreted in the projection's adjusted frame.
            effective_line = line / combined_mult if combined_mult else line
            empirical_over = _weighted_rate(values, weights, lambda v: v > effective_line)
            over_rate = empirical_over

            # ── Residual calibration with warmup blending ──
            # If a calibration file exists for this (sport, prop), replace the
            # raw empirical over-rate with a Brier-better calibrated probability.
            # Early-season players (few current-season games) blend with the
            # prior-season warmup distribution.
            calibration_meta = None
            dist_meta = None
            calibration_game_dates = history.get("game_dates") or []
            curr_games = count_current_season_games(calibration_game_dates, sport_key)
            # Line-conditional method selection: when the prop cfg carries a
            # data-gated `line_methods` list, resolve the method + its sub-cfg for
            # THIS line (else the pooled single method). Ships inert until the
            # offline refit writes line_methods for a bucket that earned it.
            method, method_cfg = _method_cfg_for_line(prop_calib_cfg, line)
            if method == "D":
                # §2.4b-2 distributional P(>=k hits) from a contact-quality
                # composite. Whitelisted to batter_hits; fails open to the raw
                # empirical over-rate. Rate multipliers (defense/matchup/park/
                # weather) scale the per-AB hit RATE; the batting-order exposure
                # multiplier scales the AB COUNT (n) — a binomial needs each on
                # its own parameter, so they're passed separately (not the lumped
                # combined_mult). The bucket cfg may override the xBA weight.
                rate_mult = (output_def_mult * matchup_mult
                             * park_mult * weather_mult)
                dist_strength = (method_cfg.get("xstats_strength", xstats_strength)
                                 if method_cfg else xstats_strength)
                p_dist, dist_meta = _distributional_over_rate(
                    prop_key, line, values, at_bats, weights,
                    rate_mult, lineup_mult, player_name, commence_iso,
                    method_cfg, dist_strength,
                    teams=(home_team_name, away_team_name))
                if p_dist is not None:
                    over_rate = max(0.0, min(1.0, p_dist))
                    calibration_meta = {
                        "method": "D",
                        "curr_games": curr_games,
                        "empirical_over": round(empirical_over * 100, 2),
                    }
            elif method == "E":
                # §2.2 Negative-Binomial count model. The mean is `avg_stat` (the
                # same adjusted projection method B calibrates), scaled by the
                # offline-fit `mean_scale`; `dispersion` is the fitted over-
                # dispersion. No warmup block (like D). Fails open to the empirical
                # over-rate when dispersion/mean_scale are absent or the mean is
                # unusable.
                p_nb = _negbin_over_rate(
                    avg_stat, method_cfg.get("mean_scale"),
                    method_cfg.get("dispersion"), line)
                if p_nb is not None:
                    over_rate = max(0.0, min(1.0, p_nb))
                    calibration_meta = {
                        "method": "E",
                        "curr_games": curr_games,
                        "mean_scale": method_cfg.get("mean_scale"),
                        "dispersion": method_cfg.get("dispersion"),
                        "empirical_over": round(empirical_over * 100, 2),
                    }
            elif method:
                p_cal = apply_calibration_with_warmup(
                    method_cfg, avg_stat, line, curr_games,
                    empirical_over=empirical_over,
                )
                if p_cal is not None:
                    over_rate = max(0.0, min(1.0, p_cal))
                    warmup_games = method_cfg.get("warmup_games", 10) or 1
                    blend_w = min(curr_games / float(warmup_games), 1.0)
                    calibration_meta = {
                        "method": method,
                        "curr_games": curr_games,
                        "warmup_games": warmup_games,
                        "blend_weight": round(blend_w, 3),
                        "empirical_over": round(empirical_over * 100, 2),
                    }

            # ── Platt recalibration (self-updating) ──
            # Apply the same final calibration layer before either standard or
            # Safe Mode branches. Safe Mode previously exited before this step,
            # so its displayed confidence did not have parity with standard props.
            raw_over_rate = over_rate
            # Per-line-bucket recal: a prop with line_methods gets its bucket's
            # Platt fit (or none, if that bucket earned no valid map); a prop
            # without line_methods keeps its single pooled fit (unchanged).
            recal_cfg = _resolve_recal_cfg(
                recalibration, prop_key, line, prop_calib_cfg)

            def _apply_final_recalibration(probability, cfg):
                if not cfg or cfg.get("a") is None:
                    return probability
                adjusted = apply_platt(
                    probability,
                    cfg.get("a"),
                    cfg.get("b"),
                )
                if adjusted is None:
                    return probability
                return max(0.0, min(1.0, adjusted))

            over_rate = _apply_final_recalibration(over_rate, recal_cfg)
            recal_meta = None
            if over_rate != raw_over_rate:
                recal_meta = {
                    "a": recal_cfg.get("a"),
                    "b": recal_cfg.get("b"),
                    "n_fit": recal_cfg.get("n_fit"),
                    "raw_prob": round(raw_over_rate * 100, 2),
                }

            # ── Safe mode (OVER-only, integer alt-line) ──
            if safe_mode:
                # values / weights already had DNPs filtered above.
                #
                # Parametric lower bound:
                #   threshold = round_down(projected_mean − z · weighted_std)
                # where z = Phi⁻¹(safe_target). For safe_target=0.95, z≈1.645.
                #
                # Why parametric instead of pure empirical quantile?
                # With ~10 games of history, the 5th-percentile empirical
                # quantile collapses to the sample minimum (a single bad
                # game's weight ≥ 5% of total). The parametric Normal bound
                # uses ALL recent games to estimate location + spread, which
                # gives a stable threshold instead of "Wemby 4+ points"
                # whenever a 4-pt foul-out game exists in his last 10.
                import math as _math
                if not values:
                    continue
                proj_mean = avg_stat  # already shrunk + def-adjusted
                wstd = _weighted_std(values, weights, mean=base_proj)
                # Scale std by the same projection factor (output-defense ×
                # MLB matchup) so the spread is in the projection's frame.
                wstd_adj = wstd * (combined_mult if combined_mult else 1.0)
                z = _normal_inv_cdf(safe_target)
                alt_q = proj_mean - z * wstd_adj

                # "Points {N}+" means the player needs actual ≥ N to win.
                # Floor (not ceil) for OVER thresholds: alt_q=8.7 → 8+,
                # because 8 is the largest integer the model expects them
                # to clear with ≥ safe_target probability.
                safe_threshold = max(1, int(_math.floor(alt_q)))

                def _probability_at_threshold(threshold_value):
                    """Return (historical, final model) P(actual >= threshold)."""
                    historical = _weighted_rate(
                        values, weights,
                        lambda v, t=threshold_value: v >= t,
                    )
                    threshold_line = threshold_value - 0.5
                    effective_threshold_line = (
                        threshold_line / combined_mult if combined_mult else threshold_line
                    )
                    empirical_adjusted = _weighted_rate(
                        values, weights,
                        lambda v, t=effective_threshold_line: v > t,
                    )
                    raw_probability = empirical_adjusted
                    # Resolve the line-conditional bucket by THIS threshold_line
                    # (not the book line) for parity with the standard path. A
                    # D-method bucket degrades to the empirical rate here — full
                    # distributional P(>=k) inside the safe-threshold loop is a
                    # deferred refinement (inert until line_methods ships).
                    sm_method, sm_cfg = _method_cfg_for_line(
                        prop_calib_cfg, threshold_line)
                    if sm_method and sm_method != "D":
                        calibrated = apply_calibration_with_warmup(
                            sm_cfg,
                            avg_stat,
                            threshold_line,
                            curr_games,
                            empirical_over=empirical_adjusted,
                        )
                        if calibrated is not None:
                            raw_probability = max(0.0, min(1.0, calibrated))
                    # Recalibrate by the bucket for THIS threshold_line (the alt
                    # line Safe Mode actually prices), matching the base-cfg
                    # bucket resolved just above — not the book-line bucket.
                    sm_recal_cfg = _resolve_recal_cfg(
                        recalibration, prop_key, threshold_line, prop_calib_cfg)
                    return historical, _apply_final_recalibration(
                        raw_probability, sm_recal_cfg)

                historical_at_safe, p_at_safe = _probability_at_threshold(safe_threshold)

                # Tighten or relax the parametric starting threshold using the
                # same residual + warmup + Platt probability stack that standard
                # props use. This keeps the displayed target tied to the actual
                # production probability rather than a separate empirical-only
                # calculation.
                while p_at_safe < safe_target and safe_threshold > 1:
                    safe_threshold -= 1
                    historical_at_safe, p_at_safe = _probability_at_threshold(safe_threshold)
                # Cap tightening at the largest adjusted outcome in the sample.
                # Production Platt slopes are positive, but this also prevents a
                # malformed future recalibration from making the loop unbounded.
                max_safe_threshold = max(
                    1,
                    int(_math.ceil(max(values) * (combined_mult or 1.0))),
                )
                while safe_threshold < max_safe_threshold:
                    next_t = safe_threshold + 1
                    historical_next, p_next = _probability_at_threshold(next_t)
                    if p_next >= safe_target:
                        safe_threshold = next_t
                        p_at_safe = p_next
                        historical_at_safe = historical_next
                    else:
                        break

                if p_at_safe < safe_target:
                    continue

                # Tight sanity guard: drop when historical hit rate at the
                # suggested threshold is more than 5pp below safe_target.
                # Was 15pp tolerance — measured to admit false positives
                # (NBA assists @ 95% claimed but actually hit 76%; MLB
                # batter_hits @ 85% claimed but actually hit 69%). The
                # 5pp band keeps out-of-sample hit rate within ~5pp of
                # the user-visible safe_target.
                if historical_at_safe < (safe_target - 0.05):
                    continue

                # Floor-collapse guard: when safe_threshold is 1 (or 0), the
                # bet collapses to "did the player do the thing at all?" —
                # a binary-ish outcome whose true probability is just the
                # player's intrinsic rate of doing the thing once. The
                # parametric Normal quantile underestimates variance for
                # low-mean integer distributions and overstates how "safe"
                # this is. Measured: forward hit rate for "1+" suggestions
                # at high safe_target averages 65-80%, not 95%. Refuse to
                # market these as high-confidence picks above safe_target=80%.
                if safe_threshold <= 1 and safe_target > 0.80:
                    continue

                # Realism guard: if the safe threshold sits absurdly far
                # below the book line, two things are wrong:
                #   1. the player's game-to-game variance is so high that
                #      our 95% floor is unreliable (not actually "safe"),
                #   2. no sportsbook offers alt OVER lines that far below
                #      the main line, so the bet can't be placed anyway.
                # Require safe_threshold to be at least 50% of the book
                # line. (Wemby with line=27.5 → must be ≥14 to surface.)
                SAFE_MIN_RATIO = 0.5
                if line > 0 and safe_threshold < line * SAFE_MIN_RATIO:
                    continue

                # Our model's final calibrated confidence at the standard line.
                model_hit_at_line = over_rate

                # Gap from book line to our safe threshold. Larger positive
                # gap = book line is below safe floor (bet straight OVER).
                # Negative = user must hunt for an alt OVER line ≤ (safe_threshold − 1).
                line_gap = safe_threshold - line
                bettable_at_standard_line = line < safe_threshold

                # Confidence delta between our safe suggestion and the book line.
                # Positive = our suggestion is safer than the standard line.
                model_delta = p_at_safe - model_hit_at_line

                candidates.append({
                    "type": "player_prop",
                    "matchup": matchup,
                    "player": player_name,
                    "team": player_team_name,
                    "prop": prop_key,
                    "prop_label": PROP_LABELS.get(prop_key, prop_key),
                    "line": line,
                    "over_price": over_price,
                    "under_price": under_price,
                    "dk_over_price": dk_over_price,
                    "dk_under_price": dk_under_price,
                    "over_implied": round(over_implied * 100, 2),
                    "under_implied": round(under_implied * 100, 2),
                    "avg_stat": round(avg_stat, 2),
                    "over_rate": round(over_rate * 100, 2),
                    "games_sampled": len(values),
                    # Price/edge/EV are intentionally pending until the exact
                    # alternate line is fetched. Comparing this probability to
                    # the standard book-line price would mix different bets.
                    "edge_pct": 0.0,
                    "expected_roi_pct": None,
                    "direction": "OVER",
                    "best_price": over_price,
                    "is_value": False,
                    "value_pending": True,
                    "no_history": False,
                    # ── Safe-mode-specific fields ──
                    "safe_mode": True,
                    "safe_target": safe_target,
                    "safe_threshold": safe_threshold,        # display as "{N}+"
                    "safe_alt_q": round(alt_q, 2),           # raw quantile (continuous)
                    "model_hit_at_safe": round(p_at_safe * 100, 2),     # prob at suggested
                    "historical_hit_at_safe": round(historical_at_safe * 100, 2),
                    "model_hit_at_line": round(model_hit_at_line * 100, 2),  # prob at book line
                    "model_delta": round(model_delta * 100, 2),         # safe − book line
                    "line_gap": round(line_gap, 2),
                    "bettable_at_standard_line": bettable_at_standard_line,
                    "calibration": calibration_meta,
                    "recalibration": recal_meta,
                    "park_factor": park_meta,
                    "weather": weather_meta,
                    "xstats": xstats_meta,
                    "distributional": dist_meta,
                    "_values": list(values),
                    "_weights": list(weights),
                })
                continue

            # ── Market-as-prior shrinkage (P1.1a) ──
            # Blend the final calibrated model OVER prob toward the de-vigged
            # market OVER prob with w = n/(n+k). Thin samples (small n) lean on
            # the sharp market prior; large samples keep the model. Applied to
            # the standard line only — over_implied is defined here; safe mode's
            # alt-line probabilities exited above via `continue`. raw_over_rate
            # (logged for Platt refits) was snapshotted before this, so the blend
            # shows up in final_prob only and never contaminates future fits.
            market_prior_meta = None
            if market_prior_k and market_prior_k > 0 and values:
                n_prior = len(values)
                w = n_prior / (n_prior + market_prior_k)
                pre_blend = over_rate
                over_rate = max(0.0, min(
                    1.0, w * over_rate + (1.0 - w) * over_implied))
                market_prior_meta = {
                    "k": market_prior_k,
                    "n": n_prior,
                    "w": round(w, 3),
                    "prior": round(over_implied * 100, 2),
                    "pre_blend": round(pre_blend * 100, 2),
                }

            # Compare historical over rate vs book implied over probability
            over_edge = over_rate - over_implied
            under_rate = 1 - over_rate
            under_edge = under_rate - under_implied

            if over_edge > under_edge:
                direction = "OVER"
                edge = over_edge
                dk_price = dk_over_price
                dk_book = dk_over_book
            else:
                direction = "UNDER"
                edge = under_edge
                dk_price = dk_under_price
                dk_book = dk_under_book

            # The user bets exclusively at DraftKings (2026-07-21). Multi-book
            # odds feed ONLY the de-vigged consensus (over_implied) the edge is
            # measured against — market analysis, not execution. Price, EV,
            # staking, and display all use the DK price. `best_price` is retained
            # as the executable (DK) price for downstream compatibility. When DK
            # doesn't post this prop/line, dk_price is None → expected_roi is None
            # → the bet is NOT flagged as value (never recommend an un-bettable
            # line). (P1.1b, DK-only per user.)
            best_price = dk_price
            expected_roi = _expected_roi(
                over_rate if direction == "OVER" else under_rate,
                dk_price,
            )

            dk_alone = (odds_info.get("market_implied_method")
                        == "dk_selfdevig_fallback")
            is_value = _prop_gate_is_value(
                edge, expected_roi, prop_key, gate_ev_floor, gate_edge_floor,
                gate_suppress, threshold, dk_alone)

            # §2.4b-2 direction split: demote losing UNDER picks on cheap lines
            # (batter_hits under 0.5 wins only ~43% OOS) from recommendations.
            # The candidate is still returned; it just isn't flagged as value.
            under_suppressed = False
            if is_value and direction == "UNDER" and _suppress_under(prop_key, line):
                is_value = False
                under_suppressed = True

            # §2.5A confirmed-lineup / probable-starter gate. A high-confidence
            # "out" (benched batter once both lineups are posted, or a pitcher who
            # isn't the announced starter) is a dead bet: demote it from
            # recommendations and skip the log below — the label would never
            # resolve. Tri-state and fail-open: "in"/"unknown" change nothing.
            lineup_status = history.get("lineup_status")
            lineup_out = (_lineup_gating_enabled(sport_key)
                          and lineup_status == "out")
            if lineup_out:
                is_value = False

            # Log the published probability so future refits learn from it.
            # We log the *raw* (pre-Platt) probability — that's what Platt
            # was fit against and what subsequent refits should map.
            if log_game_date and sport_key and not lineup_out:
                _ident = _resolve_ident(player_name, prop_key)
                prediction_row = log_prediction(
                    sport_key=sport_key,
                    event_id=prop_data.get("game_id"),
                    commence_time=commence_iso,
                    prop_key=prop_key,
                    player=player_name,
                    game_date=log_game_date,
                    line=line,
                    raw_prob=raw_over_rate,
                    final_prob=over_rate,
                    projected=avg_stat,
                    direction=direction,
                    price=dk_price,
                    book=dk_book,
                    is_value=is_value,
                    team=player_team_name,
                    batting_order=history.get("batting_order"),
                    mlb_player_id=(_ident.get("mlb_player_id") if _ident else None),
                    game_pk=(_ident.get("game_pk") if _ident else None),
                    ids_resolved=bool(_ident),
                    source=history.get("source"),   # "warehouse" | "espn" (audit the flip)
                    write=False,
                )
                if prediction_row:
                    prediction_rows.append(prediction_row)

            candidates.append({
                "type": "player_prop",
                "matchup": matchup,
                "player": player_name,
                "team": player_team_name,
                # Confirmed lineup slot (1-9) when posted, else None. Used by the
                # auto-pick "top X value bets" rule that focuses hitter Overs on
                # the top of the order; .get() (not the MLB-scoped local var) so
                # non-MLB sports safely resolve to None.
                "batting_order": history.get("batting_order"),
                # §2.5A tri-state availability ("in"/"out"/"unknown"/None) for the
                # UI badge. "out" props are demoted from recs above; kept in the
                # candidate list so they still surface (as non-value "other props").
                "lineup_status": lineup_status,
                "prop": prop_key,
                "prop_label": PROP_LABELS.get(prop_key, prop_key),
                "line": line,
                "over_price": over_price,
                "under_price": under_price,
                "over_implied": round(over_implied * 100, 2),
                "under_implied": round(under_implied * 100, 2),
                "avg_stat": round(avg_stat, 2),
                "over_rate": round(over_rate * 100, 2),
                "games_sampled": len(values),
                "edge_pct": round(edge * 100, 2),
                "direction": direction,
                "best_price": best_price,
                "dk_over_price": dk_over_price,
                "dk_under_price": dk_under_price,
                "dk_price": dk_price,
                "dk_book": dk_book,
                # Which line this was analyzed at: 'dk' (DK's own line) vs
                # 'consensus' (DK didn't offer the player); peer_count = independent
                # books at that line. Surfaced for the UI + audit of the DK anchor.
                "line_source": odds_info.get("line_source"),
                "peer_count": odds_info.get("peer_count"),
                "expected_roi_pct": (round(expected_roi * 100, 2)
                                      if expected_roi is not None else None),
                "is_value": is_value,
                "under_suppressed": under_suppressed,
                "no_history": False,
                "calibration": calibration_meta,
                "recalibration": recal_meta,
                "market_prior": market_prior_meta,
                "park_factor": park_meta,
                "weather": weather_meta,
                "xstats": xstats_meta,
                "distributional": dist_meta,
                "_values": list(values),
                "_weights": list(weights),
            })

    log_prediction_rows(prediction_rows)
    return candidates


def format_props_report(candidates):
    """Format player props value candidates into a readable report."""
    value_bets = [c for c in candidates if c["is_value"]]
    no_history = [c for c in candidates if c.get("no_history")]
    non_value = [c for c in candidates if not c["is_value"] and not c.get("no_history")]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  PLAYER PROPS ANALYSIS")
    lines.append("=" * 80)

    if value_bets:
        lines.append("")
        lines.append("  VALUE PROPS FOUND:")
        for c in sorted(value_bets, key=lambda x: x["edge_pct"], reverse=True):
            lines.append("")
            lines.append(f"  >>> {c['player']} — {c['prop_label']} {c['direction']} {c['line']}")
            lines.append(f"      Matchup:        {c['matchup']}")
            lines.append(f"      Line:           {c['line']}  |  Over: {c['over_price']:+d}  |  Under: {c['under_price']:+d}")
            lines.append(f"      Avg Stat:       {c['avg_stat']} (last {c['games_sampled']} games)")
            lines.append(f"      Over Rate:      {c['over_rate']}% historical")
            lines.append(f"      Book Implied:   Over {c['over_implied']}% / Under {c['under_implied']}%")
            lines.append(f"      EDGE:           +{c['edge_pct']}% on {c['direction']} ({c['best_price']:+d})")
    else:
        lines.append("\n  No player prop value bets found above threshold.")

    if non_value:
        lines.append("")
        lines.append("-" * 80)
        lines.append("  Other Props (no edge above threshold)")
        lines.append("-" * 80)
        for c in non_value:
            edge_str = f"+{c['edge_pct']}%" if c['edge_pct'] > 0 else f"{c['edge_pct']}%"
            dir_str = c['direction'] or "?"
            lines.append(f"  {c['player']:25s} | {c['prop_label']:18s} | Line: {c['line']:5} | Avg: {c['avg_stat']:5} | {dir_str}: {edge_str}")

    if no_history:
        lines.append("")
        lines.append(f"  ({len(no_history)} prop(s) skipped — no ESPN history found)")

    return "\n".join(lines)
