"""Player-prop pricing.

analyze_player_props_value (the projection + calibration + value gate for player
props) and its formatting/report helper, plus the MLB matchup/lineup multipliers
and the per-sport prop tuning knobs. Split out of analysis.py (P3). A clean
sibling of analysis (margin) and parlay: it depends only downward on stats /
pricing_common / odds_client / calibration_loader / recalibration / prop_filter.
analysis re-exports these names for backward compatibility.
"""

from calibration_loader import (
    apply_calibration_with_warmup,
    count_current_season_games,
    load_calibration,
    load_lineup_adjustment,
)
from odds_client import PROP_LABELS
from park_factors import PROP_PARK_KIND, park_factor
from pricing_common import (
    _expected_roi,
    _opponent_defense_multiplier,
    _prop_is_value,
    _starter_adjustment,
    _venue_match_multiplier,
    et_local_date,
)
from prop_filter import filter_player_gamelog
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
)


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


def _xstats_blend(base_proj, xba, ab_per_game, strength):
    """Linear blend of the projection toward the xBA-implied per-game mean.

    Returns (blended_proj, xstats_mean). Pure — the caller supplies the looked-up
    xBA and the player's recent AB/game. strength is clamped to [0, 1]."""
    w = max(0.0, min(1.0, strength))
    xstats_mean = xba * ab_per_game
    return (1.0 - w) * base_proj + w * xstats_mean, xstats_mean


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


def _player_prop_defense_strength(sport_key):
    if sport_key is None:
        return DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH
    return PLAYER_PROP_DEFENSE_STRENGTH.get(sport_key, DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH)


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


def analyze_player_props_value(prop_data, player_histories, threshold_pct=5.0,
                               sport_key=None, team_defense=None, espn_teams=None,
                               safe_mode=False, safe_target=0.95,
                               team_schedules=None, matchup_features=None,
                               weather=None):
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
            # In calibration JSON, half_life=null explicitly means equal
            # weighting. Other null knobs continue to mean "use the default."
            if value is not None or name == "half_life":
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

    home_team_name = prop_data["home_team"]
    away_team_name = prop_data["away_team"]
    matchup = f"{away_team_name} @ {home_team_name}"

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
        prop_calib_cfg = calibration.get(prop_key) if calibration else None

        for player_name, odds_info in players.items():
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
            team_schedule = None
            if team_schedules and player_team_id:
                team_schedule = team_schedules.get(str(player_team_id))
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

            # Resolve the player's upcoming home/away by matching their team_id
            # to the home/away team names of the upcoming game.
            upcoming_is_home = None
            player_team_name = None
            if player_team_id and id_to_name:
                player_team_name = id_to_name.get(str(player_team_id))
                if player_team_name == home_team_name:
                    upcoming_is_home = True
                elif player_team_name == away_team_name:
                    upcoming_is_home = False

            base_weights = _recency_weights(len(values), half_life)
            weights = []
            for bw, opp, past_h in zip(base_weights, opponents, past_home_aways):
                w = bw
                if team_defense and league_avg_def and defense_strength > 0:
                    w *= _opponent_defense_multiplier(
                        team_defense.get(opp), league_avg_def, defense_strength)
                # Venue multiplier: a per-sport PLAYER_PROP_VENUE_STRENGTH of
                # 0.0 disables it; None (default) inherits the team-level
                # VENUE_MATCH_WEIGHTS via _venue_match_multiplier.
                if venue_strength_override != 0.0:
                    w *= _venue_match_multiplier(past_h, upcoming_is_home, sport_key)
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
                    team_defense.get(opp_name), league_avg_def,
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
                    pid_info = mlb_starters.find_player_id(player_name, season)
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

            combined_mult = (output_def_mult * matchup_mult
                             * lineup_mult * park_mult * weather_mult)

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
            calibration_game_dates = history.get("game_dates") or []
            curr_games = count_current_season_games(calibration_game_dates, sport_key)
            if prop_calib_cfg and prop_calib_cfg.get("method"):
                p_cal = apply_calibration_with_warmup(
                    prop_calib_cfg, avg_stat, line, curr_games,
                    empirical_over=empirical_over,
                )
                if p_cal is not None:
                    over_rate = max(0.0, min(1.0, p_cal))
                    warmup_games = prop_calib_cfg.get("warmup_games", 10) or 1
                    blend_w = min(curr_games / float(warmup_games), 1.0)
                    calibration_meta = {
                        "method": prop_calib_cfg.get("method"),
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
            recal_cfg = recalibration.get(prop_key) if recalibration else None

            def _apply_final_recalibration(probability):
                if not recal_cfg or recal_cfg.get("a") is None:
                    return probability
                adjusted = apply_platt(
                    probability,
                    recal_cfg.get("a"),
                    recal_cfg.get("b"),
                )
                if adjusted is None:
                    return probability
                return max(0.0, min(1.0, adjusted))

            over_rate = _apply_final_recalibration(over_rate)
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
                    if prop_calib_cfg and prop_calib_cfg.get("method"):
                        calibrated = apply_calibration_with_warmup(
                            prop_calib_cfg,
                            avg_stat,
                            threshold_line,
                            curr_games,
                            empirical_over=empirical_adjusted,
                        )
                        if calibrated is not None:
                            raw_probability = max(0.0, min(1.0, calibrated))
                    return historical, _apply_final_recalibration(raw_probability)

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

            # Value requires clearing the fair-market edge AND being +EV at the
            # DraftKings price (see _prop_is_value / P1.1).
            is_value = _prop_is_value(edge, threshold, expected_roi)

            # Log the published probability so future refits learn from it.
            # We log the *raw* (pre-Platt) probability — that's what Platt
            # was fit against and what subsequent refits should map.
            if log_game_date and sport_key:
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
                "expected_roi_pct": (round(expected_roi * 100, 2)
                                      if expected_roi is not None else None),
                "is_value": is_value,
                "no_history": False,
                "calibration": calibration_meta,
                "recalibration": recal_meta,
                "market_prior": market_prior_meta,
                "park_factor": park_meta,
                "weather": weather_meta,
                "xstats": xstats_meta,
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
