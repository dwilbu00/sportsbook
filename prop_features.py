"""Candidate-feature registry for the player-prop model (roadmap §2.6).

ONE source of truth for feature multipliers, imported by all three projection
paths so they can never drift:

  1. offline real-line   — book_line_calibration.project_and_empirical
                           (drives refit_calibration.diagnose_features / --feature-diag)
  2. synthetic sweep     — backtest.run_player_props_backtest
                           (drives auto-adoption through the existing confirmation gate)
  3. runtime             — props.analyze_player_props_value
                           (default OFF until the gate writes a non-zero strength)

A "feature" is a bounded multiplier on the projection, sized by a single strength
knob so the confirmation gate (2 expanding folds + MIN_CALIB_BRIER_GAIN) can
accept / reject / size it — automating the confirmation-gate philosophy across a
feature set instead of hand-tuning. The feature also shifts the effective line
(callers divide the line by the multiplier), so it moves calibration methods
A-E alike, exactly as props.analyze_player_props_value's combined_mult does.

First tenant: REST / days-off. Fully computable offline from the per-game
``game_date`` already carried on every prior game (no new fetch, no leakage —
only dates strictly before the graded game feed it), and it has no runtime yet.

stdlib only.
"""

import math
from datetime import date


# ── rest / days-off feature ─────────────────────────────────────────────────
# Continuous, bounded tanh of "how much more/less rest than the player's OWN
# cadence" (median inter-game gap). The cadence normalization is what lets one
# form self-adapt across a ~5-day starter rotation and a ~1-day batter schedule
# with no position logic. Direction: more-rest-than-cadence -> modestly MORE
# production (correctly signed only for production-positive props; see
# FEATURE_REGISTRY["rest"]["props"]).
REST_K = 0.10        # max fractional nudge before the strength scaling / cap
REST_TAU = 2.0       # tanh saturation scale (in days of rest-vs-cadence delta)
REST_DELTA_CAP = 4.0  # clamp rest-vs-cadence delta to +/- this many days
REST_CAP = 0.08      # hard bound on the multiplier: [1-CAP, 1+CAP]
REST_MIN_PRIOR = 3   # need >= this many valid prior dates for a stable cadence


def _to_date(v):
    """Parse an ISO date/datetime string (or date) to a date; None on failure."""
    if isinstance(v, date):
        return v
    if not v:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except (ValueError, TypeError):
        return None


def _median(vals):
    """Median of an unsorted numeric list (0.0 on empty)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def rest_multiplier(prior_game_dates, graded_date, strength):
    """Bounded rest/days-off multiplier for one projection.

    ``prior_game_dates`` — ISO strings (or dates) of the player's PRIOR games.
    ``graded_date`` — the a-priori commence date of the game being projected
    (never derived from the outcome). ``strength`` in [0, 1] sizes the effect.

    Leakage-safe: only prior dates STRICTLY before ``graded_date`` are used.
    Returns 1.0 (a no-op) when strength <= 0 or there is too little history to
    establish a cadence — so strength 0 reproduces production bit-for-bit."""
    if not strength or strength <= 0:
        return 1.0
    g = _to_date(graded_date)
    if g is None:
        return 1.0
    dates = sorted({d for d in (_to_date(x) for x in prior_game_dates)
                    if d is not None and d < g})
    if len(dates) < REST_MIN_PRIOR:
        return 1.0
    days_rest = (g - dates[-1]).days
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    cadence = max(1.0, _median(gaps))
    delta = max(-REST_DELTA_CAP, min(REST_DELTA_CAP, days_rest - cadence))
    mult = 1.0 + strength * REST_K * math.tanh(delta / REST_TAU)
    return max(1.0 - REST_CAP, min(1.0 + REST_CAP, mult))


def _rest_fn(ctx, strength):
    return rest_multiplier(ctx.get("prior_game_dates") or [],
                           ctx.get("graded_date"), strength)


# ── gamecontext feature (roadmap §3.1) ───────────────────────────────────────
# A per-batter MEAN multiplier from the shared, cached, leakage-safe GameContext
# (mlb_starters.build_game_context): the batter's own team run-scoring environment
# (own offense vs the OPPOSING bullpen; the opposing STARTER is pinned neutral,
# owned by matchup_mult) mapped to a hits nudge. The raw gc_factor is already
# bounded to +/-GC_RUN_CAP and computed offline/leakage-safe upstream; here it is
# only strength-scaled and re-clamped, exactly mirroring rest's shape. Registered
# in THREE ablation forms (full / own-offense-only / opposing-bullpen-only) so a
# single free --feature-diag prints all three verdicts, each measured MARGINAL
# over the already-SHIPPED opp_defense (which the gate's base_params carry), to
# resolve empirically which term (if any) clears the gate. Whitelisted to
# batter_hits; a hard no-op everywhere else and whenever the factor is absent
# (incomplete context / None-team obs / non-MLB) -> strength-0 byte parity.
GC_FEAT_CAP = 0.08   # hard bound on the gamecontext multiplier: [1-CAP, 1+CAP]


def _gamecontext_fn(form):
    """Build the registry fn for gamecontext ablation ``form`` (full/own/opp).

    Reads the batter's own-side gc_factor forms threaded in as
    ``ctx["gamecontext_factors"]`` (a {full, own, opp} dict, or None). Returns
    1.0 (no-op) when strength<=0 or the factor is absent, so both strength-0 and
    missing context reproduce production bit-for-bit."""
    def fn(ctx, strength):
        if not strength or strength <= 0:
            return 1.0
        factors = ctx.get("gamecontext_factors")
        if not factors:
            return 1.0
        gc = factors.get(form)
        if not gc:
            return 1.0
        mult = 1.0 + strength * (gc - 1.0)
        return max(1.0 - GC_FEAT_CAP, min(1.0 + GC_FEAT_CAP, mult))
    return fn


# ── platoon feature (roadmap §2.6, batter vs opposing-starter hand) ───────────
# A per-batter multiplier from the batter's as-of vs-hand xwOBAcon RELATIVE to
# his own all-hands baseline — a RESIDUAL over overall quality, so it doesn't
# double-count the level the projection already carries. The opposing STARTER's
# HAND is the one platoon input matchup_mult leaves on the table (matchup_mult
# reads only the starter's aggregate xBA-allowed, never his hand). The raw ratio
# is already bounded to +/-PLATOON_RUN_CAP offline (book_line_calibration.
# _attach_platoon, leakage-safe as-of); here it is only strength-scaled and
# re-clamped, exactly mirroring rest/gamecontext's shape. Whitelisted to
# batter_hits; a hard no-op everywhere else and whenever the factor is absent
# (thin vs-hand sample / unknown opposing hand / non-MLB) -> strength-0 byte
# parity.
PLATOON_FEAT_CAP = 0.08   # hard bound on the platoon multiplier: [1-CAP, 1+CAP]


def _platoon_fn(ctx, strength):
    """Strength-scale + re-clamp the pre-clamped platoon ratio threaded in as
    ``ctx["platoon_factor"]`` (a scalar vs-hand / all-hands xwOBAcon ratio, or
    None). Returns 1.0 (no-op) when strength<=0 or the factor is absent, so both
    strength-0 and missing context reproduce production bit-for-bit."""
    if not strength or strength <= 0:
        return 1.0
    factor = ctx.get("platoon_factor")
    if not factor:
        return 1.0
    mult = 1.0 + strength * (factor - 1.0)
    return max(1.0 - PLATOON_FEAT_CAP, min(1.0 + PLATOON_FEAT_CAP, mult))


# ── registry ────────────────────────────────────────────────────────────────
# HIGH-FIDELITY strength grid: 0.0..1.0 in 0.1 steps. This is the DIAGNOSTIC
# domain (--feature-diag, refit_calibration.diagnose_features) — read ONLY there,
# NOT by the main props sweep grid (backtest._build_props_sweep_grid hardcodes
# rest as a global {0.0, 1.0} axis), so a fine grid here can't explode the sweep.
# Fine steps matter because a curated feature's optimum is often small (e.g. 0.2-
# 0.3) and a coarse {0.5, 1.0} grid would overshoot the cap and miss it. Winner's
# curse is still gated by the 2-fold confirmation on the SELECTED strength.
HIGH_FIDELITY_STRENGTHS = tuple(round(i * 0.1, 1) for i in range(11))  # 0.0..1.0

# Each entry is plain data. ``props`` restricts where the feature applies (None =
# all props); a feature is a hard no-op elsewhere. ``strengths`` is the sweep /
# diagnostic domain. ``runtime_knob`` is the per-prop cfg / knob key the gate
# writes and props.py reads. ``fn(ctx, strength) -> multiplier``.
FEATURE_REGISTRY = [
    {
        "name": "rest",
        "props": frozenset({"pitcher_outs", "pitcher_strikeouts", "batter_hits"}),
        "strengths": HIGH_FIDELITY_STRENGTHS,
        "runtime_knob": "rest_strength",
        "fn": _rest_fn,
    },
    {
        # own offense AND opposing bullpen (opposing STARTER pinned neutral —
        # owned by matchup_mult). The full form; kept registered but INERT (no
        # sweep-grid axis, no runtime knob set) so a bare --feature-diag can
        # re-check it for free as the warehouse grows, exactly like `rest`.
        # §3.1 first-verdict (2026-08-05, 3,547 obs): +0.0002 @strength 1.0 on
        # the incumbent E — correctly signed but far below the 0.002 gate, so it
        # ships nowhere. The two diagnostic ablations (own-only ~+0.0000, opp-only
        # ~+0.0001) resolved the judge disagreement — full is marginally best,
        # opp overlaps the shipped opp_defense — and were dropped once settled.
        "name": "gamecontext",
        "props": frozenset({"batter_hits"}),
        "strengths": HIGH_FIDELITY_STRENGTHS,
        "runtime_knob": "gamecontext_strength",
        "fn": _gamecontext_fn("full"),
    },
    {
        # batter platoon vs the OPPOSING STARTER's hand — the last untested
        # curated §2.6 tenant. matchup_mult reads only the starter's aggregate
        # xBA-allowed (never his hand), so the platoon edge is ADDITIVE here.
        # Sized as the batter's as-of vs-hand xwOBAcon RESIDUAL over his own
        # all-hands baseline (book_line_calibration._attach_platoon). Kept
        # registered but INERT (no sweep-grid axis, no runtime knob ever set) so
        # a bare --feature-diag can re-check it for free as the warehouse grows,
        # exactly like `gamecontext`. Whitelisted to batter_hits; hard no-op
        # elsewhere / when the factor is absent -> strength-0 byte parity.
        "name": "platoon",
        "props": frozenset({"batter_hits"}),
        "strengths": HIGH_FIDELITY_STRENGTHS,
        "runtime_knob": "platoon_strength",
        "fn": _platoon_fn,
    },
]

_BY_NAME = {f["name"]: f for f in FEATURE_REGISTRY}


def feature_applies(name, prop_key):
    """True when feature ``name`` is registered and applies to ``prop_key``."""
    f = _BY_NAME.get(name)
    if not f:
        return False
    return f["props"] is None or prop_key in f["props"]


def strengths_from_params(params):
    """Extract {feature_name: strength} from a params/cfg dict.

    Reads each feature's ``runtime_knob`` (the scalar the sweep persists and
    props.py reads) and merges an explicit ``params['features']`` override map
    (how the diagnostic injects a strength). Returns only non-zero strengths."""
    out = {}
    for f in FEATURE_REGISTRY:
        v = params.get(f["runtime_knob"])
        if v:
            out[f["name"]] = float(v)
    for k, v in (params.get("features") or {}).items():
        if v:
            out[k] = float(v)
        else:
            out.pop(k, None)
    return out


def projection_multiplier(prop_key, feature_strengths, prior_game_dates,
                          graded_date, gamecontext_factors=None,
                          platoon_factor=None):
    """Combined projection multiplier over all registered features that apply to
    ``prop_key``, at the strengths in ``feature_strengths`` ({name: strength}).

    ``gamecontext_factors`` — the batter's own-side {full, own, opp} gc_factor
    dict for this game (from mlb_starters.build_game_context), or None; threaded
    to the gamecontext feature fns. Absent -> those fns no-op to 1.0.

    ``platoon_factor`` — the batter's pre-clamped as-of vs-(opposing-starter-
    hand) xwOBAcon / all-hands ratio for this game (book_line_calibration.
    _attach_platoon), or None; threaded to the platoon feature fn. Absent -> it
    no-ops to 1.0.

    Returns 1.0 when nothing applies — so an all-off / empty map is a no-op."""
    if not feature_strengths:
        return 1.0
    ctx = {"prior_game_dates": prior_game_dates, "graded_date": graded_date,
           "gamecontext_factors": gamecontext_factors,
           "platoon_factor": platoon_factor}
    mult = 1.0
    for name, strength in feature_strengths.items():
        f = _BY_NAME.get(name)
        if not f or not strength:
            continue
        if f["props"] is not None and prop_key not in f["props"]:
            continue
        mult *= f["fn"](ctx, strength)
    return mult
