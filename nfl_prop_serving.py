"""nfl_prop_serving.py — live NFL prop projections from OUR calibrated models (not ESPN).

The app's Value Finder historically projected NFL props from ESPN gamelogs (ambiguous labels,
recently broken by an ESPN format change). This serves the SAME 8-prop models we built and
validated (nfl_props_accuracy) directly in the live app:

  * COMPONENTS are frozen offline (fit on 2012-2022) in calibration/nfl_prop_models.json — Cloud
    can't read the mirror to fit. Loaded once here.
  * FEATURES are built live from the nflverse player-week fetch (nfl_opportunity_serving._load),
    replicating nfl_props_accuracy._obs_from_series for one player as-of the upcoming game
    (all completed weeks are prior; same SWEPT half-life; prior-season spillover early).
  * project() returns the projected mean + SD + P(over) via the exact model math
    (nfl_props_accuracy.proj_mean_sd / p_over).

No ESPN, no mirror, no credits. Same nflverse source as the bonus optimizer's eligibility gate.
"""
import json
import os

import nfl_props_accuracy as acc
import nfl_opportunity_serving as srv

MODELS_PATH = "calibration/nfl_prop_models.json"
_MODELS = None


def _load_models():
    global _MODELS
    if _MODELS is None:
        try:
            with open(MODELS_PATH) as f:
                _MODELS = json.load(f)
        except Exception:
            _MODELS = {}
    return _MODELS


def build_obs(player_norm, season, week, prop):
    """Feature dict for one player as-of an upcoming game — mirrors
    nfl_props_accuracy._obs_from_series (recency-weighted volume/conversion/aDOT). None if the
    player has < MIN_PRIOR prior games or no volume."""
    cfg = acc.PROPS.get(prop)
    if cfg is None:
        return None
    hl, min_prior, _k = acc.SWEPT.get(prop, (4, 3, 12))
    prior = srv._prior_games(player_norm, season, week, min_prior)   # most-recent first
    if len(prior) < min_prior:
        return None
    w = [0.5 ** (i / hl) for i in range(len(prior))]
    wsum = sum(w) or 1.0

    def ws(col):
        return sum(wi * float(r.get(col) or 0.0) for r, wi in zip(prior, w))

    stat, vol, qual = cfg["stat"], cfg.get("vol"), cfg.get("qual")
    pv = [float(r.get(stat) or 0.0) for r in prior]
    pm = (sum(pv) / len(pv)) if pv else 0.0
    cv = ((sum((v - pm) ** 2 for v in pv) / len(pv)) ** 0.5 / pm
          if pm > 0 and len(pv) > 1 else 0.0)
    o = {"n_prior": len(prior), "cv": cv}
    if cfg["fam"] == "count":
        o["mean_base"] = ws(stat) / wsum
        if o["mean_base"] <= 0:
            return None
    else:
        sv = ws(vol)
        if sv <= 0:
            return None
        o["exp_vol"] = sv / wsum
        o["conv_own"] = ws(stat) / sv
        o["vw"] = sv
        if qual == "adot":
            st = ws("targets")
            o["adot"] = (ws("receiving_air_yards") / st) if st > 0 else None
        else:
            o["adot"] = None
    return o


def project(player_norm, season, week, prop, line=None):
    """Projected mean + SD (+ P(over) if a line is given) for one player/prop from our frozen
    model. Returns None when we can't project (unknown prop, thin history, no model). player_norm
    must be scan._norm(display_name)."""
    models = _load_models()
    entry = models.get(prop)
    cfg = acc.PROPS.get(prop)
    if entry is None or cfg is None:
        return None
    comp = entry["comp"]
    # _mean_of reads the module-global SHRINK_K (the conversion self-shrink strength).
    acc.SHRINK_K = entry.get("swept", [4, 3, 12])[2]
    o = build_obs(player_norm, season, week, prop)
    if o is None:
        return None
    proj, sd = acc.proj_mean_sd(o, cfg, comp)
    k = entry.get("swept", [4, 3, 12])[2]

    def p_at(L, _o=o, _cfg=cfg, _comp=comp, _k=k):
        """P(over L) from the frozen model at an arbitrary line (for alt/safe-mode lines).
        Re-pins SHRINK_K (the module global _mean_of reads) so a later call is order-safe."""
        acc.SHRINK_K = _k
        try:
            return max(0.0, min(1.0, acc.p_over(_o, float(L), _cfg, _comp)))
        except Exception:
            return None

    p_over = p_at(line) if line is not None else None
    return {"proj": proj, "sd": sd, "p_over": p_over, "p_at": p_at,
            "opportunity": o.get("exp_vol", o.get("mean_base")), "n_prior": o["n_prior"]}


def _selftest(season, week):
    import nfl_props_scan as scan
    print(f"nfl_prop_serving self-test — {season} wk{week}")
    checks = [("Josh Allen", "player_pass_yds", 32.5), ("Josh Allen", "player_pass_tds", 1.5),
              ("Josh Allen", "player_rush_yds", 35.5), ("James Cook", "player_rush_yds", 60.5),
              ("James Cook", "player_rush_attempts", 15.5),
              ("Amon-Ra St. Brown", "player_receptions", 6.5)]
    for name, prop, line in checks:
        r = project(scan._norm(name), season, week, prop, line)
        if r:
            print(f"  {name:20} {prop:22} proj={r['proj']:6.1f} sd={r['sd']:5.1f} "
                  f"P(o {line})={r['p_over']:.2f} n={r['n_prior']}")
        else:
            print(f"  {name:20} {prop:22} — no projection")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2026")
    ap.add_argument("--week", type=int, default=99)
    a = ap.parse_args()
    _selftest(a.season, a.week)
