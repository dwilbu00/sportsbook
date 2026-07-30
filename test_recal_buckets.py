"""Tests for per-line-bucket online recalibration (composite-key schema).

Covers the shared line->bucket matcher, the composite-key builder, the apply-side
resolver (flat / composite hit / composite miss / pre-migration fallback), the fit
producer (refit_sport bucketing + gate), and the blob-parse round-trip. No pytest
dependency — run directly: ``python test_recal_buckets.py``."""
import os
import sys
import random
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import props
import recalibration

# batter_hits ships this: line 0.5 -> method A (le_0.5 bucket), >0.5 -> method C (top).
LM = [{"max_line": 0.5, "method": "A"}, {"max_line": None, "method": "C"}]


# ── _resolve_line_bucket: the single source of truth for line -> bucket ──

def test_resolve_line_bucket_keys():
    cfg = {"line_methods": LM}
    assert props._resolve_line_bucket(cfg, 0.5)[0] == "le_0.5"
    assert props._resolve_line_bucket(cfg, 0.4)[0] == "le_0.5"
    assert props._resolve_line_bucket(cfg, 1.5)[0] == "top"
    assert props._resolve_line_bucket(cfg, 2.5)[0] == "top"
    # boundary is inclusive (ln <= cap)
    assert props._resolve_line_bucket(cfg, 0.50)[0] == "le_0.5"
    assert props._resolve_line_bucket(cfg, 0.51)[0] == "top"


def test_resolve_line_bucket_no_match():
    cfg = {"line_methods": LM}
    assert props._resolve_line_bucket({"method": "A"}, 0.5) == (None, None)  # no lm
    assert props._resolve_line_bucket(cfg, None) == (None, None)             # unusable
    assert props._resolve_line_bucket(cfg, "x") == (None, None)
    assert props._resolve_line_bucket(None, 0.5) == (None, None)
    assert props._resolve_line_bucket({}, 0.5) == (None, None)


def test_resolve_line_bucket_g_format():
    # f"le_{cap:g}" trims trailing zeros: 0.50 -> "le_0.5", 2.0 -> "le_2".
    lm = [{"max_line": 2.0, "method": "A"}, {"max_line": None, "method": "C"}]
    assert props._resolve_line_bucket({"line_methods": lm}, 1.0)[0] == "le_2"


# ── _method_cfg_for_line must pick the SAME bucket (drift guard) ──

def test_method_cfg_agrees_with_bucket():
    cfg = {"method": "C", "line_methods": LM}
    for ln in [0.0, 0.5, 0.51, 1.0, 1.5, 2.5, 10.0]:
        bkey, bucket = props._resolve_line_bucket(cfg, ln)
        method, mcfg = props._method_cfg_for_line(cfg, ln)
        assert method == bucket["method"], (ln, method, bucket)
        assert mcfg["method"] == bucket["method"]


def test_method_cfg_three_buckets_middle():
    lm3 = [{"max_line": 0.5, "method": "A"},
           {"max_line": 1.5, "method": "B"},
           {"max_line": None, "method": "C"}]
    cfg = {"method": "C", "line_methods": lm3}
    assert props._resolve_line_bucket(cfg, 1.0)[0] == "le_1.5"
    assert props._method_cfg_for_line(cfg, 1.0)[0] == "B"
    assert props._method_cfg_for_line(cfg, 0.5)[0] == "A"
    assert props._method_cfg_for_line(cfg, 2.0)[0] == "C"


def test_method_cfg_unchanged_without_line_methods():
    cfg = {"method": "A", "residual_ecdf": [1, 2, 3]}
    assert props._method_cfg_for_line(cfg, 0.5) == ("A", cfg)
    assert props._method_cfg_for_line(None, 0.5) == (None, None)
    assert props._method_cfg_for_line({}, 0.5) == (None, {})


def test_method_cfg_malformed_bucket_falls_back():
    # A matched bucket with no method -> fall back to the pooled cfg (unchanged).
    cfg = {"method": "A", "line_methods": [{"max_line": 0.5}]}
    assert props._method_cfg_for_line(cfg, 0.5) == ("A", cfg)


# ── _composite_recal_key: the fit/apply storage key ──

def test_composite_recal_key():
    assert props._composite_recal_key("batter_hits", 0.5, None) == "batter_hits"
    assert props._composite_recal_key("batter_hits", 0.5, LM) == "batter_hits@le_0.5"
    assert props._composite_recal_key("batter_hits", 1.5, LM) == "batter_hits@top"
    # line_methods present but line unusable -> None (fit skips; apply no-ops)
    assert props._composite_recal_key("batter_hits", None, LM) is None
    # no line_methods -> bare key regardless of a weird line
    assert props._composite_recal_key("pitcher_outs", None, None) == "pitcher_outs"


# ── _resolve_recal_cfg: the apply-side selector ──

def test_resolve_recal_flat():
    m = {"batter_hits": {"a": 0.66, "b": 0.07, "validated": True}}
    # a prop without line_methods -> bare key (exactly today's behavior)
    assert props._resolve_recal_cfg(m, "batter_hits", 0.5, {"method": "A"})["a"] == 0.66
    assert props._resolve_recal_cfg({}, "batter_hits", 0.5, {"method": "A"}) is None
    assert props._resolve_recal_cfg(None, "batter_hits", 0.5, None) is None


def test_resolve_recal_composite_hit_and_miss():
    m = {"batter_hits@le_0.5": {"a": 0.354, "b": 0.213, "validated": True}}
    cfg = {"method": "C", "line_methods": LM}
    # line 0.5 -> its bucket's fit
    assert props._resolve_recal_cfg(m, "batter_hits", 0.5, cfg)["a"] == 0.354
    # line 1.5 -> @top has no fit, but composite keys exist -> NO recal (no borrow)
    assert props._resolve_recal_cfg(m, "batter_hits", 1.5, cfg) is None


def test_resolve_recal_pre_migration_fallback():
    # Old flat blob still in the overlay; the prop has line_methods now. Until a
    # per-bucket refit rewrites the overlay, fall back to the bare fit (no regress).
    m = {"batter_hits": {"a": 0.66, "b": 0.07, "validated": True}}
    cfg = {"method": "C", "line_methods": LM}
    assert props._resolve_recal_cfg(m, "batter_hits", 0.5, cfg)["a"] == 0.66
    assert props._resolve_recal_cfg(m, "batter_hits", 1.5, cfg)["a"] == 0.66


def test_resolve_recal_mixed_flat_and_composite():
    # Both a legacy flat entry AND a new composite entry present: composite wins,
    # and the un-fit bucket does NOT borrow the legacy flat map.
    m = {"batter_hits": {"a": 0.66, "b": 0.07, "validated": True},
         "batter_hits@le_0.5": {"a": 0.354, "b": 0.213, "validated": True}}
    cfg = {"method": "C", "line_methods": LM}
    assert props._resolve_recal_cfg(m, "batter_hits", 0.5, cfg)["a"] == 0.354
    assert props._resolve_recal_cfg(m, "batter_hits", 1.5, cfg) is None


# ── Producer: refit_sport buckets by composite key and honors the gate ──

def _make_rows(prop_key, line, n, over_dispersion=0.45, seed=0):
    """Synthetic resolved log rows whose raw prob is over-dispersed vs the true
    rate, so a shrinking Platt (a<1) beats raw and passes the CV gate."""
    rng = random.Random(seed)
    base = datetime(2025, 1, 1)
    rows = []
    for i in range(n):
        raw = 0.12 + 0.76 * (i / float(max(n - 1, 1)))     # spread [0.12, 0.88]
        p_true = 0.5 + over_dispersion * (raw - 0.5)        # raw over-confident
        y = 1 if rng.random() < p_true else 0
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append({
            "sport_key": "baseball_mlb",
            "prop_key": prop_key,
            "line": line,
            "raw_prob": raw,
            "outcome": y,
            "resolved": True,
            "game_date": d,
            "ts": d + "T12:00:00+00:00",
            "event_id": f"{prop_key}-{line}-{i}",
            "player": f"P{i}",
        })
    return rows


def test_refit_buckets_by_composite_key():
    # line 0.5: plenty of rows -> passes; line 1.5: thin (<90) -> gate rejects.
    log_rows = (_make_rows("batter_hits", 0.5, 300, seed=1)
                + _make_rows("batter_hits", 1.5, 30, seed=2))
    captured = {}

    def fake_save(sport_key, params, meta=None, to_blob=True):
        captured["params"] = params

    cal = {"batter_hits": {"method": "C", "line_methods": LM}}
    with mock.patch.object(recalibration, "_read_log", return_value=log_rows), \
         mock.patch.object(recalibration, "save_recalibration", side_effect=fake_save), \
         mock.patch("calibration_loader.load_calibration", return_value=cal):
        recalibration._LOAD_CACHE.pop("baseball_mlb", None)
        fits = recalibration.refit_sport("baseball_mlb", resolve_first=False)

    params = captured.get("params", {})
    assert "batter_hits@le_0.5" in params, sorted(params.keys())
    assert "batter_hits@top" not in params      # thin bucket -> no map
    assert "batter_hits" not in params          # never the bare key for a lm prop
    assert params["batter_hits@le_0.5"]["validated"] is True
    assert 0.2 < params["batter_hits@le_0.5"]["a"] < 1.0   # shrinks over-dispersion
    assert fits["batter_hits@le_0.5"][2] == params["batter_hits@le_0.5"]["n_fit"]


def test_refit_flat_prop_unchanged():
    # A prop WITHOUT line_methods keeps its bare key (byte-identical to today).
    # Heavy over-dispersion + ample n so the CV gate passes regardless of noise.
    log_rows = _make_rows("pitcher_strikeouts", 5.5, 400, over_dispersion=0.30, seed=1)
    captured = {}

    def fake_save(sport_key, params, meta=None, to_blob=True):
        captured["params"] = params

    cal = {}  # no line_methods for this prop
    with mock.patch.object(recalibration, "_read_log", return_value=log_rows), \
         mock.patch.object(recalibration, "save_recalibration", side_effect=fake_save), \
         mock.patch("calibration_loader.load_calibration", return_value=cal):
        recalibration._LOAD_CACHE.pop("baseball_mlb", None)
        recalibration.refit_sport("baseball_mlb", resolve_first=False)

    params = captured.get("params", {})
    assert "pitcher_strikeouts" in params
    assert not any("@" in k for k in params)


# ── Blob parse: composite entries survive (they carry top-level `validated`) ──

def test_parse_blob_keeps_composite_entries():
    blob = {
        "sport_key": "baseball_mlb",
        "fit_timestamp": "2025-05-01T00:00:00+00:00",
        "props": {
            "batter_hits@le_0.5": {"a": 0.354, "b": 0.213, "validated": True},
            "batter_hits@top": {"a": 0.5, "b": 0.0, "validated": False},  # dropped
            "pitcher_outs": {"a": 0.7, "b": 0.1, "validated": True},      # flat
        },
    }
    fit_ts, out = recalibration._parse_recal_blob(blob)
    assert "batter_hits@le_0.5" in out
    assert "batter_hits@top" not in out    # unvalidated -> filtered, same as flat
    assert "pitcher_outs" in out
    assert fit_ts is not None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} recal-bucket tests passed.")


if __name__ == "__main__":
    _run_all()
