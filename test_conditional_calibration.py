"""Hermetic tests for the --reliability conditional-calibration report helpers
in refit_calibration.py. Pure stdlib; no network / no secrets / no data.

Run: python test_conditional_calibration.py   (also pytest-compatible)
"""
import io
import math
from contextlib import redirect_stdout

import refit_calibration as rc


def _row(line, projected, actual, p, mkt_over=None, over_dec=None,
         under_dec=None):
    """Build a synthetic scored row shaped like the report's rowset."""
    return {
        "line": line, "projected": projected, "actual": actual,
        "p_A": p, "p_C": p, "mkt_over": mkt_over, "over_dec": over_dec,
        "under_dec": under_dec,
        "o": 1 if actual > line else 0,
    }


def test_wilson_ci_known_values():
    lo, hi, phat = rc._cc_wilson_ci(5, 10)
    assert abs(phat - 0.5) < 1e-9
    assert abs(lo - 0.2366) < 1e-3, lo
    assert abs(hi - 0.7634) < 1e-3, hi
    # symmetric around 0.5
    assert abs((lo + hi) / 2 - 0.5) < 1e-9


def test_wilson_ci_edges():
    # n == 0 -> maximally wide, phat 0
    lo, hi, phat = rc._cc_wilson_ci(0, 0)
    assert (lo, hi, phat) == (0.0, 1.0, 0.0)
    # all failures: lo clamps at 0, hi small and < 0.2
    lo, hi, phat = rc._cc_wilson_ci(0, 20)
    assert phat == 0.0
    assert lo == 0.0
    assert 0.0 < hi < 0.2, hi
    # all successes: hi clamps at 1.0
    lo, hi, phat = rc._cc_wilson_ci(20, 20)
    assert phat == 1.0
    assert hi == 1.0
    assert 0.8 < lo < 1.0, lo


def test_line_band():
    assert rc._cc_line_band(0.5) == "0.5"
    assert rc._cc_line_band(1.5) == "1.5"
    assert rc._cc_line_band(2.5) == "2.5+"
    assert rc._cc_line_band(3.5) == "2.5+"


def test_proj_band():
    assert rc._cc_proj_band(0.50) == "proj<0.75"
    assert rc._cc_proj_band(0.74) == "proj<0.75"
    assert rc._cc_proj_band(0.75) == "0.75-1.25"
    assert rc._cc_proj_band(1.00) == "0.75-1.25"
    assert rc._cc_proj_band(1.25) == "1.25-1.75"
    assert rc._cc_proj_band(1.74) == "1.25-1.75"
    assert rc._cc_proj_band(1.75) == "proj>=1.75"
    assert rc._cc_proj_band(3.00) == "proj>=1.75"


def test_num_or_none():
    assert rc._cc_num_or_none(None) is None
    assert rc._cc_num_or_none("abc") is None
    assert rc._cc_num_or_none(-110) == -110.0
    assert rc._cc_num_or_none("-110") == -110.0
    assert rc._cc_num_or_none(105.0) == 105.0


def test_stratum_table_over_side_edge_roi_and_thin():
    # 5 rows at line 0.5: p=0.7 > mkt_over=0.6 -> all back the OVER, edge +0.100.
    # over_dec = 2.0 (even money). 3 overs win / 2 lose: pnl = 3*(+1)+2*(-1)=+1,
    # roi = 1/5 = +20.0%, hit = 3/5 = 60.0%.
    rows = [
        _row(0.5, 0.9, 1, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # win
        _row(0.5, 0.9, 2, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # win
        _row(0.5, 0.9, 1, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # win
        _row(0.5, 0.9, 0, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # loss
        _row(0.5, 0.9, 0, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # loss
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc._cc_stratum_table(rows, "p_A", "line", rc._cc_line_band,
                             rc._CC_LINE_BANDS, min_cell_n=50)
    out = buf.getvalue()
    assert "+0.100" in out, out          # edge = mean(0.7 - 0.6)
    assert "+20.0%(5)" in out, out       # ROI+-(nbets)
    assert "60.0%" in out, out           # hit rate
    assert "[THIN]" in out, out          # n=5 < min_cell_n=50
    assert "5/5" in out, out             # price coverage n_priced/n


def test_stratum_table_under_side_backed():
    # p=0.3 < mkt_over=0.6 -> model backs the UNDER. line 1.5, under_dec=2.0.
    # under wins when o==0 (actual <= line). 4 unders (actual 0 or 1) win,
    # 1 over (actual 2) loses the under bet: pnl = 4*(+1)+1*(-1)=+3, roi=+60%,
    # hit = 4/5 = 80.0%. edge = mean(0.3-0.6) = -0.300 (over-side edge, negative).
    rows = [
        _row(1.5, 1.0, 0, 0.3, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # win
        _row(1.5, 1.0, 1, 0.3, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # win
        _row(1.5, 1.0, 0, 0.3, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # win
        _row(1.5, 1.0, 1, 0.3, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # win
        _row(1.5, 1.0, 2, 0.3, mkt_over=0.6, over_dec=2.0, under_dec=2.0),  # loss
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc._cc_stratum_table(rows, "p_A", "line", rc._cc_line_band,
                             rc._CC_LINE_BANDS, min_cell_n=50)
    out = buf.getvalue()
    assert "-0.300" in out, out          # over-side edge is negative
    assert "+60.0%(5)" in out, out       # ROI from backing the under side
    assert "80.0%" in out, out           # under hit rate


def test_stratum_table_no_prices_blanks_edge():
    rows = [_row(1.5, 1.4, 2, 0.55) for _ in range(60)]  # no prices, n>=50
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc._cc_stratum_table(rows, "p_A", "line", rc._cc_line_band,
                             rc._CC_LINE_BANDS, min_cell_n=50)
    out = buf.getvalue()
    # unpriced -> edge and ROI blank "-", coverage 0/60, and not THIN (n=60)
    assert "0/60" in out, out
    assert "[THIN]" not in out, out


def test_reliability_perfect_bins():
    # Bin 0.6-0.7: 30 rows at p=0.65, exactly 60% overs -> emp 0.600.
    rows = []
    for i in range(30):
        actual = 1 if i < 18 else 0     # 18/30 = 0.60
        rows.append(_row(0.5, 0.9, actual, 0.65))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc._cc_reliability("all", rows, "p_A", min_cell_n=50)
    out = buf.getvalue()
    assert "0.6-0.7" in out, out
    assert "0.600" in out, out           # empirical over-freq
    assert "[THIN]" not in out, out      # n=30 >= 25 decile bar


def test_reliability_thin_decile_flagged():
    rows = [_row(0.5, 0.9, 1, 0.95) for _ in range(10)]  # bin 0.9-1.0, n=10<25
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc._cc_reliability("all", rows, "p_A", min_cell_n=50)
    out = buf.getvalue()
    assert "0.9-1.0" in out, out
    assert "[THIN]" in out, out


# ── recalibration helpers ────────────────────────────────────────────────

def test_logit_sigmoid_roundtrip():
    for p in (0.01, 0.25, 0.5, 0.73, 0.99):
        assert abs(rc._rc_sigmoid(rc._rc_logit(p)) - p) < 1e-9, p
    # sigmoid is symmetric and monotone
    assert rc._rc_sigmoid(0.0) == 0.5
    assert rc._rc_sigmoid(10.0) > rc._rc_sigmoid(-10.0)


def test_platt_identity_on_calibrated_data():
    # Perfectly calibrated 3-point data -> MLE is the identity (a=1, b=0).
    ps, os = [], []
    for p, k in [(0.3, 30), (0.5, 50), (0.7, 70)]:
        for i in range(100):
            ps.append(p)
            os.append(1 if i < k else 0)
    a, b = rc._rc_fit_platt(ps, os)
    assert abs(a - 1.0) < 0.08, a
    assert abs(b) < 0.08, b


def test_platt_shrinks_overdispersed_data():
    # Over-dispersed: model says 0.8 but only 60% happen; says 0.2 but 40% happen.
    # The correcting map must SHRINK toward 0.5 -> slope a < 1.
    ps, os = [], []
    for i in range(200):
        ps.append(0.8)
        os.append(1 if i < 120 else 0)   # 60%
    for i in range(200):
        ps.append(0.2)
        os.append(1 if i < 80 else 0)    # 40%
    a, b = rc._rc_fit_platt(ps, os)
    assert a < 0.9, a                     # genuine shrinkage
    # applied to an extreme prob, the calibrated value is pulled toward center
    cal = rc._rc_apply_platt(0.8, a, b)
    assert 0.5 < cal < 0.8, cal


def test_apply_platt_shrink_direction():
    # a<1, b=0 pulls both tails toward 0.5
    assert 0.5 < rc._rc_apply_platt(0.9, 0.5, 0.0) < 0.9
    assert 0.1 < rc._rc_apply_platt(0.1, 0.5, 0.0) < 0.5


def test_isotonic_monotone_and_pooling():
    # Clean step: below-median 0, above-median 1.
    kx, ky = rc._rc_fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
    assert ky == sorted(ky), ky          # non-decreasing
    assert rc._rc_apply_isotonic(0.05, (kx, ky)) == 0.0   # clip below
    assert rc._rc_apply_isotonic(0.5, (kx, ky)) == 1.0    # clip above
    # Violation in the middle pools to a flat 0.5 there.
    kx2, ky2 = rc._rc_fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    assert ky2 == sorted(ky2), ky2
    assert ky2[0] == 0.0 and ky2[-1] == 1.0
    assert any(abs(y - 0.5) < 1e-9 for y in ky2), ky2   # pooled block


def test_metrics_known_values():
    # Perfect predictions -> brier 0, ece 0, logloss ~0.
    perfect = [_row(0.5, 0.9, 1, 1.0), _row(0.5, 0.9, 0, 0.0)]
    m = rc._rc_metrics(perfect, lambda r: r["p_A"])
    assert abs(m["brier"]) < 1e-9, m
    assert abs(m["ece"]) < 1e-9, m
    assert m["logloss"] < 1e-4, m
    # Coin flips called at 0.5 -> brier 0.25, ece 0 (mp==mo in the bin).
    coin = [_row(0.5, 0.9, 1, 0.5), _row(0.5, 0.9, 0, 0.5)]
    m2 = rc._rc_metrics(coin, lambda r: r["p_A"])
    assert abs(m2["brier"] - 0.25) < 1e-9, m2
    assert abs(m2["ece"]) < 1e-9, m2


def test_edge_summary_counts_and_roi():
    # p=0.7 vs mkt 0.6 (over side, +0.10) x3 wins; p=0.3 vs mkt 0.6 (under, big
    # negative over-edge) x2 -> both count as |e|>0.05; one >0.20 (the 0.3 rows).
    rows = [
        _row(0.5, 0.9, 1, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),
        _row(0.5, 0.9, 1, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),
        _row(0.5, 0.9, 0, 0.7, mkt_over=0.6, over_dec=2.0, under_dec=2.0),
        _row(1.5, 1.0, 0, 0.3, mkt_over=0.6, over_dec=2.0, under_dec=2.0),
        _row(1.5, 1.0, 2, 0.3, mkt_over=0.6, over_dec=2.0, under_dec=2.0),
    ]
    e = rc._rc_edge_summary(rows, lambda r: r["p_A"])
    assert e["n"] == 5
    assert e["c05"] == 5                  # all |edge|>0.05 (0.10 and 0.30)
    assert e["c20"] == 2                  # only the two 0.3-vs-0.6 rows (|e|=0.30)
    assert abs(e["max"] - 0.30) < 1e-9
    assert e["nbets"] == 5


def test_method_for_line_bucket_pick():
    lm = [{"max_line": 0.5, "method": "A"}, {"max_line": None, "method": "C"}]
    assert rc._rc_method_for_line(0.5, lm, "A") == "A"
    assert rc._rc_method_for_line(1.5, lm, "A") == "C"
    assert rc._rc_method_for_line(2.5, lm, "A") == "C"
    assert rc._rc_method_for_line(1.5, None, "A") == "A"   # no buckets -> default


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} conditional-calibration tests passed.")


if __name__ == "__main__":
    _run_all()
