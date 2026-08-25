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

Count-distribution methods (D, E) do NOT use the residual block or warmup and are
NOT dispatched through calibrate_prob() below — props.py routes them directly at
the projection seam:
  * method "D" (§2.4b) — binomial contact-quality count model (batter_hits); needs
    no persisted params (its P(over) is a closed form on each obs's own as-of AB/p).
  * method "E" (§2.2) — over-dispersed Negative Binomial count model for low-count
    integer props (variance = mean + dispersion*mean^2). Persists two extra fields
    instead of the residual block:
        "method": "E",
        "mean_scale": 1.03,   # multiplicative mean bias correction, clamped [0.5, 2.0]
        "dispersion": 0.18,   # 0.0 => Poisson limit
    Runtime mean = avg_stat * mean_scale; P(over) = negbin_at_least(int(line)+1,
    mean, dispersion). Selected only when it clears the real-line confirmation gate
    (see book_line_calibration.select_method_at_real_lines).
"""
import json
import math
import os
import shutil
from datetime import datetime, timezone

from stats import _norm_cdf  # canonical shared implementation (P3 dedup)

# Match backtest.py: months when each sport's season starts.
SPORT_SEASON_START_MONTH = {
    "basketball_nba": 10,
    "americanfootball_nfl": 9,
    "baseball_mlb": 3,
    "icehockey_nhl": 10,
}

CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "calibration")
# Old live files are copied here on promotion, as timestamped rollback points.
ARCHIVE_DIR = os.path.join(CALIBRATION_DIR, "archive")


def calibration_path(sport_key):
    return os.path.join(CALIBRATION_DIR, f"{sport_key}.json")


# ─── candidate-file staging ────────────────────────────────────────────────
# A calibration refit is expensive to get right (incumbent method E/D splices,
# real-line re-selection) and easy to clobber by an accidental re-run. When
# candidate mode is active, EVERY calibration write (all save_* helpers) targets
# calibration/<sport>.candidate.json instead of the live file, and a
# block-preserving save reads back from the candidate so successive refits in one
# staging cycle accumulate into a single staged file. The live file the app
# serves from is never touched until promote_calibration() atomically swaps the
# candidate in. Default OFF, so all existing callers (the online recalibration
# loop, tests, ad-hoc save_*) write the live file exactly as before — only the
# offline refit/backtest entrypoints opt in via set_candidate_mode(True).
_CANDIDATE_MODE = False


def set_candidate_mode(on):
    """Enable/disable candidate-file staging for calibration WRITES (process-global).
    Does NOT change serving reads — serving stays LIVE even mid-staging, so a refit
    building a candidate never makes the app serve unpromoted values. To grade a STAGED
    calibration in an offline backtest, use set_serving_candidate() (separate flag)."""
    global _CANDIDATE_MODE
    _CANDIDATE_MODE = bool(on)


_SERVING_CANDIDATE = False


def set_serving_candidate(on):
    """Enable/disable candidate-aware SERVING reads (process-global) — OFFLINE backtest
    ONLY, so it can grade a STAGED team-market calibration (additive / prob_shrink /
    market_blend / challenger) on a holdout WITHOUT promoting. Kept SEPARATE from
    set_candidate_mode (writes) so the invariant 'serving reads stay live mid-staging'
    holds for the refit + the live app (which never enables this). Clears the serving
    caches so a toggle never cross-serves a stale block."""
    global _SERVING_CANDIDATE
    _SERVING_CANDIDATE = bool(on)
    _reset_serving_caches()


def serving_candidate_active():
    return _SERVING_CANDIDATE


def _reset_serving_caches():
    """Invalidate every process-level calibration serving cache (this module's additive
    cache + pricing_common's shrink/blend caches + analysis's challenger cache) so a
    candidate-mode toggle can't cross-serve a stale live/candidate block. Reaches
    already-imported modules via sys.modules (no import cycle)."""
    import sys
    _ADDITIVE_CFG_CACHE.clear()
    pc = sys.modules.get("pricing_common")
    if pc is not None:
        getattr(pc, "_PROB_SHRINK_CACHE", {}).clear()
        getattr(pc, "_MARKET_BLEND_CACHE", {}).clear()
    an = sys.modules.get("analysis")
    if an is not None:
        getattr(an, "_EXPECTED_RUNS_CACHE", {}).clear()


def candidate_mode_active():
    return _CANDIDATE_MODE


def candidate_path(sport_key):
    return os.path.join(CALIBRATION_DIR, f"{sport_key}.candidate.json")


def has_candidate(sport_key):
    """True if a staged (un-promoted) candidate exists for this sport."""
    return os.path.exists(candidate_path(sport_key))


def _write_path(sport_key):
    """The file a save writes to: the candidate when staging, else the live file."""
    return candidate_path(sport_key) if _CANDIDATE_MODE else calibration_path(sport_key)


def active_write_path(sport_key):
    """Full path a save would write to right now (candidate when staging, else live)."""
    return _write_path(sport_key)


def active_write_label(sport_key):
    """Basename of the current write target — for user-facing messages."""
    return os.path.basename(_write_path(sport_key))


def load_calibration(sport_key):
    """
    Load calibration for a sport. Returns a dict of {prop_key: prop_cfg} or
    an empty dict if no calibration file exists for this sport.
    Failures are silent — analyzers fall back to defaults.

    Candidate-aware via _serving_path (the SAME offline-only mechanism the team-market
    readers use through _serving_blob): the live app never enables serving-candidate mode,
    so _SERVING_CANDIDATE stays False and this reads the LIVE file byte-identically. An
    offline backtest that calls set_serving_candidate(True) grades a STAGED candidate's
    PROPS on a holdout without promoting. Props were previously left on the live file, so a
    staged --seasons refit couldn't be graded through the props path — this closes that gap.
    """
    path = _serving_path(sport_key)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return {}
    return blob.get("props", {})


def _read_json(path):
    """Parse a JSON file, or None if missing/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_blob(sport_key):
    """Load the raw LIVE calibration blob (all top-level keys), or {} if missing.

    Always reads the live file — every runtime/serving reader (load_calibration,
    load_market_blend, load_prob_shrink, ...) goes through here, so staging a
    candidate never changes what the app serves.
    """
    return _read_json(calibration_path(sport_key)) or {}


def _serving_path(sport_key):
    """Path a SERVE/grade-time read resolves to: the staged candidate when candidate
    mode is active AND one exists (so an OFFLINE backtest grades the staged team-market
    calibration without promoting), else the live file. The live app never enables
    candidate mode -> serving always reads live (byte-identical)."""
    if _SERVING_CANDIDATE and has_candidate(sport_key):
        return candidate_path(sport_key)
    return calibration_path(sport_key)


def _serving_blob(sport_key):
    """The calibration blob a serve/grade-time reader should use — candidate-aware via
    _serving_path. Every team-market serving reader (prob_shrink, market_blend,
    expected_runs_challenger, starter_adjustment) goes through here so a candidate-staged
    calibration can be graded on a holdout without promoting."""
    return _read_json(_serving_path(sport_key)) or {}


def _load_write_blob(sport_key):
    """The blob a save STARTS from (to preserve other props/blocks).

    In candidate mode, read the candidate if it exists so successive fits in one
    staging cycle accumulate into a single staged file; otherwise seed from the
    live file so the candidate begins as a complete copy of live plus the new
    fit. Outside candidate mode this is identical to _load_blob (the live file),
    so existing write behavior is byte-for-byte unchanged.
    """
    if _CANDIDATE_MODE:
        cand = _read_json(candidate_path(sport_key))
        if cand is not None:
            return cand
    return _load_blob(sport_key)


def load_calibration_for_refit(sport_key):
    """Props a REFIT should build its decisions on (NOT for serving).

    When staging (candidate mode) and a candidate already exists, return the
    candidate's props so a multi-step staged refit composes — e.g. a staged
    ``--real-lines`` pass re-selects each method against the methods/residuals a
    staged sweep just wrote, exactly as the pre-staging flow chained through the
    live file. Falls back to the live props (load_calibration) otherwise, so a
    first staged pass reads the live incumbents (e.g. a spliced-in method E).
    Serving/analysis code must keep using load_calibration (always live).
    """
    if _CANDIDATE_MODE:
        cand = _read_json(candidate_path(sport_key))
        if cand is not None:
            return cand.get("props", {}) or {}
    return load_calibration(sport_key)


def existing_candidate_notice(sport_key):
    """Warning text when a staged candidate already exists at the start of a new
    staged refit: the run ACCUMULATES onto it (it is not reseeded from live), so a
    forgotten candidate could carry stale blocks into a later promotion. Returns
    None when nothing is staged.
    """
    if not has_candidate(sport_key):
        return None
    blob = _read_json(candidate_path(sport_key)) or {}
    return (f"[candidate] A staged candidate already exists for {sport_key} "
            f"(fit {blob.get('fit_timestamp', '?')}); this run ACCUMULATES onto "
            f"it rather than reseeding from live. Inspect it with --diff or start "
            f"fresh with --discard.")


def save_calibration(sport_key, props_cfg, meta=None, merge_props=True):
    """Persist a calibration blob; creates calibration/ if needed.

    Starts from the existing blob so a props refit preserves every other
    top-level block (market_blend, starter_adjustment, prob_shrink, ...).
    By default merges props into any existing props dict rather than
    replacing it, so a partial refit (e.g. only pitcher_outs) does not wipe
    already-calibrated props. Meta is merged shallowly for the same reason.
    """
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob["fit_timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    existing_props = blob.get("props")
    if merge_props and isinstance(existing_props, dict):
        existing_props.update(props_cfg)
        blob["props"] = existing_props
    else:
        blob["props"] = props_cfg
    if meta:
        existing_meta = blob.get("meta")
        if isinstance(existing_meta, dict):
            existing_meta.update(meta)
            blob["meta"] = existing_meta
        else:
            blob["meta"] = meta
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def discard_candidate(sport_key):
    """Delete a staged candidate (if any). Returns True if one was removed."""
    cand = candidate_path(sport_key)
    if os.path.exists(cand):
        os.remove(cand)
        return True
    return False


def promote_calibration(sport_key):
    """Make the staged candidate the live calibration.

    Archives the current live file to calibration/archive/<sport>.<ts>.json first
    (a non-destructive rollback point), then atomically replaces the live file
    with the candidate (os.replace within one directory is atomic and consumes
    the candidate). Returns the archive path, or None if there was no prior live
    file. Raises FileNotFoundError when no candidate is staged.
    """
    cand = candidate_path(sport_key)
    if not os.path.exists(cand):
        raise FileNotFoundError(
            f"No staged candidate for {sport_key} (expected {cand}).")
    live = calibration_path(sport_key)
    archived = None
    if os.path.exists(live):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = os.path.join(ARCHIVE_DIR, f"{sport_key}.{ts}.json")
        shutil.copy2(live, archived)
    os.replace(cand, live)
    return archived


def diff_calibration(sport_key):
    """Summarize how a staged candidate differs from the live file, for review.

    Returns ``{"has_candidate": False}`` when nothing is staged, else:
        {has_candidate, live_exists, candidate_ts, live_ts,
         props: {added: [...], removed: [...],
                 changed: [{prop, live_method, candidate_method,
                            live_nobs, candidate_nobs}]},
         blocks: {added: [...], removed: [...], changed: [...]}}
    where ``blocks`` covers top-level non-props config (market_blend,
    prob_shrink, starter_adjustment, ...).
    """
    cand_blob = _read_json(candidate_path(sport_key))
    if cand_blob is None:
        return {"has_candidate": False}
    live_blob = _read_json(calibration_path(sport_key)) or {}
    cand_props = cand_blob.get("props") or {}
    live_props = live_blob.get("props") or {}

    def _field(props, k, field):
        v = props.get(k)
        return v.get(field) if isinstance(v, dict) else None

    changed = []
    for k in sorted(set(cand_props) & set(live_props)):
        cm = _field(cand_props, k, "method")
        lm = _field(live_props, k, "method")
        cn = _field(cand_props, k, "n_obs")
        ln = _field(live_props, k, "n_obs")
        if cm != lm or cn != ln:
            changed.append({"prop": k, "live_method": lm, "candidate_method": cm,
                            "live_nobs": ln, "candidate_nobs": cn})
    _skip = {"props", "sport_key", "fit_timestamp", "meta"}
    cand_blocks = set(cand_blob) - _skip
    live_blocks = set(live_blob) - _skip
    return {
        "has_candidate": True,
        "live_exists": bool(live_blob),
        "candidate_ts": cand_blob.get("fit_timestamp"),
        "live_ts": live_blob.get("fit_timestamp"),
        "props": {
            "added": sorted(set(cand_props) - set(live_props)),
            "removed": sorted(set(live_props) - set(cand_props)),
            "changed": changed,
        },
        "blocks": {
            "added": sorted(cand_blocks - live_blocks),
            "removed": sorted(live_blocks - cand_blocks),
            "changed": sorted(b for b in (cand_blocks & live_blocks)
                              if cand_blob.get(b) != live_blob.get(b)),
        },
    }


def load_market_blend(sport_key):
    """
    Load per-market model⇄market blend weights for team markets, e.g.:
        {"moneyline": {"w": 0.6, ...}, "spreads": {...}, "totals": {...}}
    Returns {} if none are configured. Failures are silent.
    """
    if not sport_key:
        return {}
    return _serving_blob(sport_key).get("market_blend", {})


def save_market_blend(sport_key, blend, meta=None):
    """
    Persist per-market blend weights into calibration/<sport>.json, preserving
    the existing 'props' calibration block.

    The per-market weights are MERGED into any existing market_blend block (like
    prob_shrink) so a partial fit — e.g. the spreads-only shares fitter writing
    just the spreads blend, or a moneyline/totals run — updates only the markets
    it fit and leaves the others intact instead of wiping them.
    """
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    existing = blob.get("market_blend")
    if isinstance(existing, dict):
        merged = dict(existing)
        merged.update(blend)
        blob["market_blend"] = merged
    else:
        blob["market_blend"] = blend
    if meta:
        blob.setdefault("meta", {})
        if isinstance(blob["meta"], dict):
            blob["meta"]["market_blend"] = meta
        else:
            blob["meta"] = {"market_blend": meta}
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def load_starter_adjustment(sport_key):
    """
    Load MLB starter/opponent adjustment weights (Phase 1), e.g.:
        {"moneyline": 0.35, "totals": 0.6, "run_scale": 1.0}
    These are logit-space weights applied to the normalized starter_edge /
    combined run-suppression features from mlb_starters.py. Returns {} when
    none configured (→ analyzers apply no adjustment). Failures are silent.

    NOTE: default weights are conservative priors; they should be re-fit from
    graded outcomes via backtest, like market_blend / prob_shrink.
    """
    if not sport_key:
        return {}
    return _serving_blob(sport_key).get("starter_adjustment", {})


def load_expected_runs_challenger(sport_key):
    """Load the validated MLB expected-runs market configuration."""
    if not sport_key:
        return {}
    return _serving_blob(sport_key).get("expected_runs_challenger", {})


def save_starter_adjustment(sport_key, adj, meta=None):
    """Persist starter-adjustment weights, preserving other calibration blocks."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    blob["starter_adjustment"] = adj
    if meta:
        blob.setdefault("meta", {})
        if isinstance(blob["meta"], dict):
            existing = blob["meta"].get("starter_adjustment")
            if isinstance(existing, dict):
                existing.update(meta)
                blob["meta"]["starter_adjustment"] = existing
            else:
                blob["meta"]["starter_adjustment"] = meta
        else:
            blob["meta"] = {"starter_adjustment": meta}
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def save_expected_runs_challenger_shares(sport_key, shares, meta=None):
    """Update ONLY the challenger ensemble blend shares
    (expected_runs_challenger.final_2025_validation.ensemble_challenger_share)
    in calibration/<sport>.json, preserving the rest of the challenger block
    (enabled / live_markets / the fallback multiplicative `model`) and every
    other calibration block. ``shares`` = {"home_minus_1_5": <spread_share>,
    "margin": <margin_share>}. Candidate-aware via _load_write_blob / _write_path."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    chal = blob.get("expected_runs_challenger")
    if not isinstance(chal, dict):
        chal = {}
    fv = chal.get("final_2025_validation")
    if not isinstance(fv, dict):
        fv = {}
    # Per-key MERGE (like save_prob_shrink / save_market_blend): update only the
    # shares we fit (home_minus_1_5 / margin) and preserve any sibling keys (e.g. a
    # moneyline challenger share) instead of clobbering the whole sub-block.
    existing_share = fv.get("ensemble_challenger_share")
    if isinstance(existing_share, dict):
        merged = dict(existing_share)
        merged.update(shares)
        fv["ensemble_challenger_share"] = merged
    else:
        fv["ensemble_challenger_share"] = shares
    chal["final_2025_validation"] = fv
    blob["expected_runs_challenger"] = chal
    if meta:
        blob.setdefault("meta", {})
        if isinstance(blob["meta"], dict):
            blob["meta"]["expected_runs_challenger_share"] = meta
        else:
            blob["meta"] = {"expected_runs_challenger_share": meta}
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


# Neutral challenger ensemble prior used by stage_team_market_reset: NOT empty,
# because the spreads expected-runs projection needs valid shares present to fire
# (analysis._mlb_expected_runs_projection reads shares["home_minus_1_5"]/["margin"]
# and returns None on KeyError). --fit-shares overwrites these, and the raw cover
# components it captures are independent of the share value, so 0.5/0.5 is safe.
_NEUTRAL_CHALLENGER_SHARE = {"home_minus_1_5": 0.5, "margin": 0.5}


def stage_team_market_reset(sport_key):
    """STAGE a candidate with the TEAM-MARKET calibration blocks reset to a clean
    slate, so a fresh team-market recalibration leaves no residue from prior builds.

    NON-DESTRUCTIVE: writes ONLY the candidate (set_candidate_mode), seeded from live;
    the live file the app serves is UNTOUCHED. Revert with refit_calibration.py
    --discard; review with --diff.

    RESET (the blocks the fresh RUN regenerates):
      - prob_shrink -> {}  (the fit repopulates every graded market; an un-endorsed
        market then serves at 1.0 = no shrink, the intended clean slate)
      - market_blend -> removed  (the fit repopulates; --fit-shares pins spreads)
      - expected_runs_challenger.final_2025_validation.ensemble_challenger_share
        -> the NEUTRAL prior (strips any stale sibling e.g. a leftover moneyline share;
        --fit-shares overwrites home_minus_1_5/margin)
      - team-market meta entries cleared
    PRESERVED: props, starter_adjustment, lineup_adjustment, value_gate, and the
    challenger enabled/live_markets/model blocks (the projection needs them to fire).
    expected_runs_additive is left alone — --additive-save fully overwrites it, so
    clearing it here would only add an ordering footgun. Returns the candidate path."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    set_candidate_mode(True)                 # write path -> candidate; seed from live
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    blob["prob_shrink"] = {}
    blob.pop("market_blend", None)
    chal = blob.get("expected_runs_challenger")
    if isinstance(chal, dict):
        fv = chal.get("final_2025_validation")
        if not isinstance(fv, dict):
            fv = {}
        fv["ensemble_challenger_share"] = dict(_NEUTRAL_CHALLENGER_SHARE)
        chal["final_2025_validation"] = fv
        blob["expected_runs_challenger"] = chal
    meta = blob.get("meta")
    if isinstance(meta, dict):
        for k in ("prob_shrink", "prob_shrink_holdout", "market_blend",
                  "expected_runs_challenger_share"):
            meta.pop(k, None)
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    return _write_path(sport_key)


_ADDITIVE_CFG_CACHE = {}   # sport_key -> ((st_mtime_ns, st_size), block)


def load_expected_runs_additive(sport_key):
    """Load the additive expected-runs model config (Tier A #1d), or {} if none.
    Read by mlb_starters.live_additive_runs on the flag-ON path; {} keeps the live
    path on the multiplicative model (byte-identical).

    CANDIDATE-AWARE READ: under set_candidate_mode(True) with a staged candidate, this
    reads <sport>.candidate.json so an OFFLINE backtest can grade a staged additive
    config (park/fatigue run_env weights) WITHOUT promoting to live. The live app never
    sets candidate mode, so SERVING always reads the live file (byte-identical). Mirrors
    the write-side _load_write_blob / load_calibration_for_refit candidate pattern.

    PROCESS-CACHED by (path, mtime_ns, size): live_additive_runs calls this once PER
    GAME and the MLB blob is large (~2 MB). The path is in the key so a live<->candidate
    switch never cross-serves, and any write (--promote / save) auto-invalidates."""
    if not sport_key:
        return {}
    path = _serving_path(sport_key)
    try:
        st = os.stat(path)
        stat_key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        stat_key = (path, None)
    cached = _ADDITIVE_CFG_CACHE.get(sport_key)
    if cached is not None and cached[0] == stat_key:
        return cached[1]
    block = (_read_json(path) or {}).get("expected_runs_additive", {})
    _ADDITIVE_CFG_CACHE[sport_key] = (stat_key, block)
    return block


def save_expected_runs_additive(sport_key, model, meta=None):
    """Persist the fitted additive expected-runs model (Tier A #1d), preserving every
    other calibration block (props, expected_runs_challenger, starter_adjustment, ...).
    Candidate-aware: under set_candidate_mode(True) this writes <sport>.candidate.json,
    NOT live — the owner promotes via refit_calibration.py --promote."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    blob["expected_runs_additive"] = model
    if meta:
        blob.setdefault("meta", {})
        if isinstance(blob["meta"], dict):
            existing = blob["meta"].get("expected_runs_additive")
            if isinstance(existing, dict):
                existing.update(meta)
                blob["meta"]["expected_runs_additive"] = existing
            else:
                blob["meta"]["expected_runs_additive"] = meta
        else:
            blob["meta"] = {"expected_runs_additive": meta}
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def load_lineup_adjustment(sport_key):
    """Load validated per-prop batting-order exposure adjustments."""
    if not sport_key:
        return {}
    return _load_blob(sport_key).get("lineup_adjustment", {})


def save_lineup_adjustment(sport_key, adjustment, meta=None):
    """Persist batting-order exposure settings without replacing other fits."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    blob["lineup_adjustment"] = adjustment
    if meta:
        blob.setdefault("meta", {})
        if isinstance(blob["meta"], dict):
            blob["meta"]["lineup_adjustment"] = meta
        else:
            blob["meta"] = {"lineup_adjustment": meta}
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def load_prob_shrink(sport_key):
    """
    Load per-market probability-shrink factors for team markets, e.g.:
        {"spreads": 0.6, "totals": 0.5}
    A factor s pulls the model probability toward 0.5: p' = 0.5 + s*(p-0.5),
    correcting overconfidence. Returns {} (→ analyzers use 1.0 = no shrink) if
    none configured. Failures are silent.
    """
    if not sport_key:
        return {}
    return _serving_blob(sport_key).get("prob_shrink", {})


def save_prob_shrink(sport_key, shrink, meta=None, holdout=None):
    """
    Persist per-market probability-shrink factors into calibration/<sport>.json,
    preserving the existing 'props' and 'market_blend' blocks.

    The per-market shrink factors are MERGED into any existing prob_shrink block
    so a partial fit (e.g. a moneyline-only run whose ESPN schedule can't grade
    the currently-stored spread/total games) updates only the markets it fit and
    leaves the others intact, instead of wiping them.

    ``holdout`` (optional): per-market scored-holdout metrics
    {market: {brier, raw_brier, n}} from the backtest. Persisted (additively,
    merged like prob_shrink) under meta.prob_shrink_holdout so the app can publish
    real team-market holdout accuracy instead of "Not exported". The backtest may
    also include ``n_warehouse``/``n_log`` provenance keys (warehouse-graded vs
    prediction-log-supplemented obs counts, n == n_warehouse + n_log); the app
    reads ``brier``/``raw_brier``/``n`` and ignores the extra keys.
    """
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    existing = blob.get("prob_shrink")
    if isinstance(existing, dict):
        merged = dict(existing)
        merged.update(shrink)
        blob["prob_shrink"] = merged
    else:
        blob["prob_shrink"] = shrink
    if not isinstance(blob.get("meta"), dict):
        blob["meta"] = {}
    if meta:
        blob["meta"]["prob_shrink"] = meta
    if holdout:
        existing_holdout = blob["meta"].get("prob_shrink_holdout")
        if isinstance(existing_holdout, dict):
            merged_holdout = dict(existing_holdout)
            merged_holdout.update(holdout)
            blob["meta"]["prob_shrink_holdout"] = merged_holdout
        else:
            blob["meta"]["prob_shrink_holdout"] = dict(holdout)
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


def load_value_gate(sport_key):
    """Load the player-prop recommendation-gate config, e.g.:
        {"ev_floor": 0.04, "edge_floor": 0.01, "suppress": ["pitcher_outs"]}
    ``ev_floor``/``edge_floor`` are fractions; a prop is flagged value only when
    its EV at the DK price >= ev_floor AND its fair-market edge >= edge_floor, and
    never when its key is in ``suppress``. Returns {} when none is configured —
    the analyzer then falls back to the legacy edge-threshold gate (edge >=
    threshold_pct AND EV > 0), so untuned sports are unaffected. Failures silent.
    """
    if not sport_key:
        return {}
    gate = _load_blob(sport_key).get("value_gate", {})
    if not isinstance(gate, dict):
        return {}
    # Coerce/validate field types so a misconfigured block fails SAFE rather than
    # silently mis-gating: a string ev_floor/edge_floor would raise at the gate
    # comparison, and a string ``suppress`` would iterate into per-character keys
    # (disabling real suppression). Drop bad numerics (→ legacy gate) and accept a
    # bare-string suppress as a single-key list.
    out = {}
    for key in ("ev_floor", "edge_floor"):
        v = gate.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    supp = gate.get("suppress")
    if isinstance(supp, str):
        out["suppress"] = [supp]
    elif isinstance(supp, (list, tuple, set)):
        out["suppress"] = [str(s) for s in supp]
    # R5 selectivity config (optional): time_bands = [[max_hours, mult], ...]
    # (ascending by max_hours); longshot_surcharge = [min_american_price, mult].
    # Malformed entries are dropped → the caller falls back to its module defaults.
    def _num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    tb = gate.get("time_bands")
    if isinstance(tb, (list, tuple)):
        bands = [(float(r[0]), float(r[1])) for r in tb
                 if isinstance(r, (list, tuple)) and len(r) == 2
                 and _num(r[0]) and _num(r[1])]
        if bands:
            out["time_bands"] = bands
    ls = gate.get("longshot_surcharge")
    if (isinstance(ls, (list, tuple)) and len(ls) == 2
            and _num(ls[0]) and _num(ls[1])):
        out["longshot_surcharge"] = (float(ls[0]), float(ls[1]))
    # Prop-conditional recency-weighted-CV floor {prop_key: min_cv}: recommend the
    # prop only when the player's outcome volatility clears the floor (validated for
    # pitcher_earned_runs — the market underprices high-variance starters). Drop
    # non-numeric entries so a misconfigured floor fails SAFE (that prop just isn't
    # CV-gated) rather than raising at the gate comparison.
    cvf = gate.get("cv_floor")
    if isinstance(cvf, dict):
        parsed = {str(k): float(v) for k, v in cvf.items() if _num(v)}
        if parsed:
            out["cv_floor"] = parsed
    return out


def save_value_gate(sport_key, gate, meta=None):
    """Persist the player-prop recommendation-gate config, preserving other
    blocks (props/prob_shrink/...). Replaces the value_gate block wholesale (it's
    small + selected as one unit, unlike the per-market prob_shrink merge)."""
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    blob = _load_write_blob(sport_key)
    blob["sport_key"] = sport_key
    blob.setdefault("props", blob.get("props", {}))
    blob["value_gate"] = gate
    if meta:
        if not isinstance(blob.get("meta"), dict):
            blob["meta"] = {}
        blob["meta"]["value_gate"] = meta
    with open(_write_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)


# ─── season-aware date helpers ────────────────────────────────────────────

def _season_start_iso(now=None, sport_key=None):
    """
    First day of the current season for `sport_key`. NBA/NHL: Oct 1 of the
    last year the season started. MLB: Mar 1. NFL: Sep 1. Default: Jan 1.
    """
    now = now or datetime.now(timezone.utc)
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
# _norm_cdf is imported from stats (canonical, shared with analysis).


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
