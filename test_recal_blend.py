"""Tests for the seed-as-prior online recalibration: the shrinkage blend
(_blend_recal), the per-key merge in _load_recal_cached (SQL overlay of the
committed seed), and the champion gate in fit_platt_chronological (a loop fit
overrides a seeded key only after clearing an obs floor AND beating the seed
out-of-sample). No pytest dependency — run directly:

    python test_recal_blend.py
"""
import os
import sys
import random
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recalibration


def _cfg(a, b, n, validated=True):
    return {"a": a, "b": b, "n_fit": n, "validated": validated}


# ────────────────────────── _blend_recal (shrinkage) ──────────────────────────

def test_blend_equal_evidence_is_midpoint():
    # n_seed == n_loop, RECAL_SEED_TRUST 1.0 → w = 0.5 → exact average.
    out = recalibration._blend_recal(_cfg(0.30, 0.05, 300), _cfg(0.70, 0.10, 300))
    assert abs(out["a"] - 0.50) < 1e-9, out
    assert abs(out["b"] - 0.075) < 1e-9, out
    assert abs(out["blend_weight"] - 0.5) < 1e-9
    # provenance + inherited loop fields
    assert out["blend_seed"] == {"a": 0.30, "b": 0.05, "n_fit": 300}
    assert out["validated"] is True


def test_blend_loop_zero_obs_is_all_seed():
    out = recalibration._blend_recal(_cfg(0.30, 0.05, 300), _cfg(0.70, 0.10, 0))
    assert abs(out["a"] - 0.30) < 1e-9
    assert abs(out["b"] - 0.05) < 1e-9
    assert abs(out["blend_weight"]) < 1e-9


def test_blend_loop_dominates_with_volume():
    out = recalibration._blend_recal(_cfg(0.30, 0.05, 10), _cfg(0.70, 0.10, 100000))
    assert out["a"] > 0.699, out
    assert out["blend_weight"] > 0.999


def test_blend_missing_seed_nfit_floored():
    # Seed without n_fit is floored to MIN_FIT_SAMPLES, not ignored.
    seed = {"a": 0.30, "b": 0.05, "validated": True}          # no n_fit
    out = recalibration._blend_recal(seed, _cfg(0.70, 0.10, recalibration.MIN_FIT_SAMPLES))
    # w = n_loop / (n_loop + 1.0*MIN_FIT_SAMPLES) = 0.5 → midpoint
    assert abs(out["a"] - 0.50) < 1e-9, out
    assert out["blend_seed"]["n_fit"] is None


def test_blend_zero_denominator_returns_loop_unchanged():
    # denom = n_loop + TRUST*n_seed == 0 → return the loop cfg untouched.
    with mock.patch.object(recalibration, "RECAL_SEED_TRUST", 0.0):
        out = recalibration._blend_recal(_cfg(0.30, 0.05, 300), _cfg(0.70, 0.10, 0))
    assert out["a"] == 0.70 and out["b"] == 0.10
    assert "blend_weight" not in out


def test_blend_seed_trust_slows_takeover():
    # Higher RECAL_SEED_TRUST → smaller loop weight → stays closer to the seed.
    with mock.patch.object(recalibration, "RECAL_SEED_TRUST", 3.0):
        out = recalibration._blend_recal(_cfg(0.30, 0.0, 300), _cfg(0.70, 0.0, 300))
    # w = 300/(300+3*300) = 0.25 → a = 0.25*0.70 + 0.75*0.30 = 0.40
    assert abs(out["a"] - 0.40) < 1e-9, out


# ─────────────────── per-key merge in _load_recal_cached (SQL) ──────────────────

def _merge(sql_blob, seed):
    """Run _load_recal_cached against a faked SQL backend + seed."""
    fake_db = mock.Mock()
    fake_db.load_recal.return_value = sql_blob
    with mock.patch.object(recalibration, "_sql", return_value=True), \
         mock.patch.object(recalibration, "_db", fake_db), \
         mock.patch.object(recalibration, "_read_local_recal", return_value=seed):
        recalibration._LOAD_CACHE.pop("baseball_mlb", None)
        return recalibration._load_recal_cached("baseball_mlb")


def test_merge_blends_shared_and_preserves_seed_only():
    sql_blob = {"fit_timestamp": "2026-07-30T00:00:00+00:00",
                "props": {"pitcher_strikeouts": _cfg(0.70, 0.10, 300)}}
    seed = (1234.0, {
        "batter_hits@le_0.5": _cfg(0.40, 0.20, 900),   # seed-only
        "pitcher_strikeouts": _cfg(0.30, 0.05, 300),   # shared → blends
    })
    fit_ts, props = _merge(sql_blob, seed)
    # seed-only key survives untouched (no whole-map erasure)
    assert props["batter_hits@le_0.5"]["a"] == 0.40
    assert "blend_weight" not in props["batter_hits@le_0.5"]
    # shared key blended toward the seed (w=0.5 → a=0.5)
    assert abs(props["pitcher_strikeouts"]["a"] - 0.50) < 1e-9
    assert abs(props["pitcher_strikeouts"]["blend_weight"] - 0.5) < 1e-9
    # fit_ts keys on the SQL fit timestamp (a live fit exists)
    import datetime as dt
    expect = dt.datetime.fromisoformat("2026-07-30T00:00:00+00:00").timestamp()
    assert abs(fit_ts - expect) < 1e-6


def test_merge_loop_only_key_passthrough():
    sql_blob = {"fit_timestamp": "2026-07-30T00:00:00+00:00",
                "props": {"pitcher_strikeouts": _cfg(0.70, 0.10, 300)}}
    seed = (100.0, {"batter_hits@le_0.5": _cfg(0.40, 0.20, 900)})  # no shared key
    _, props = _merge(sql_blob, seed)
    # loop-only key applied verbatim (no seed to blend with)
    assert props["pitcher_strikeouts"]["a"] == 0.70
    assert "blend_weight" not in props["pitcher_strikeouts"]
    # seed-only key still present
    assert props["batter_hits@le_0.5"]["a"] == 0.40


def test_merge_sql_empty_returns_seed_with_seed_fit_ts():
    seed = (555.0, {"batter_hits@le_0.5": _cfg(0.40, 0.20, 900)})
    fit_ts, props = _merge(None, seed)          # load_recal → None
    # CRITICAL: seed_fit_ts, NOT None (None would make maintain_sport refit every tick)
    assert fit_ts == 555.0
    assert props["batter_hits@le_0.5"]["a"] == 0.40
    assert "blend_weight" not in props["batter_hits@le_0.5"]


def test_merge_sql_unvalidated_only_returns_seed():
    # SQL row exists but is not validated → parsed to no props → fall back to seed.
    sql_blob = {"fit_timestamp": "2026-07-30T00:00:00+00:00",
                "props": {"pitcher_strikeouts": _cfg(0.70, 0.10, 300, validated=False)}}
    seed = (555.0, {"batter_hits@le_0.5": _cfg(0.40, 0.20, 900)})
    fit_ts, props = _merge(sql_blob, seed)
    assert fit_ts == 555.0
    assert "batter_hits@le_0.5" in props
    assert "pitcher_strikeouts" not in props


# ─────────────────── champion gate in fit_platt_chronological ──────────────────

def _records(n, over=0.45, seed=0):
    """Over-dispersed synthetic (date, raw, outcome) rows: raw is over-confident
    vs the true rate, so a shrinking Platt (a<1) beats raw and passes the CV gate."""
    rng = random.Random(seed)
    base = datetime(2025, 1, 1)
    recs = []
    for i in range(n):
        raw = 0.12 + 0.76 * (i / float(max(n - 1, 1)))
        p_true = 0.5 + over * (raw - 0.5)
        y = 1 if rng.random() < p_true else 0
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        recs.append((d, raw, y))
    return recs


def test_gate_incumbent_none_is_unchanged_beat_raw():
    recs = _records(400, seed=3)
    a = recalibration.fit_platt_chronological(recs)
    b = recalibration.fit_platt_chronological(recs, incumbent=None)
    assert a is not None and b is not None
    # identical result whether the arg is omitted or explicitly None
    assert (a["a"], a["b"]) == (b["a"], b["b"])


def test_gate_obs_floor_rejects_below_min_override():
    # Enough rows to pass the base fit gate + beat raw, but below MIN_OBS_FOR_OVERRIDE.
    n = recalibration.MIN_OBS_FOR_OVERRIDE - 50
    recs = _records(n, seed=0)
    assert recalibration.fit_platt_chronological(recs) is not None       # no incumbent
    assert recalibration.fit_platt_chronological(recs, incumbent=(1.0, 0.0)) is None


def test_gate_weak_incumbent_passes():
    # Identity seed (a=1,b=0) scores exactly like raw; a loop that beats raw beats it.
    recs = _records(400, seed=3)
    assert recalibration.fit_platt_chronological(recs) is not None
    assert recalibration.fit_platt_chronological(recs, incumbent=(1.0, 0.0)) is not None


def test_gate_strong_incumbent_rejects():
    # Incumbent = the fit on ALL rows (it has seen every holdout point), so a
    # prefix-trained fold fit cannot beat it out-of-sample → override rejected,
    # even though the same records DO beat raw (test_gate_weak_incumbent_passes).
    recs = _records(400, seed=3)
    rows = sorted(recs)
    allfit = recalibration.fit_platt([r[1] for r in rows], [r[2] for r in rows])
    assert allfit is not None
    assert recalibration.fit_platt_chronological(recs, incumbent=allfit) is None


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} recal-blend/gate tests passed.")


if __name__ == "__main__":
    _run_all()
