"""nfl_td_model.py — anytime-TD model from RED-ZONE opportunity (does new signal help?).

TDs are opportunity-driven like HRs, but the opportunity that matters is RED-ZONE / goal-line
usage — which our volume features never captured (why pass_tds failed trustworthiness). This
derives real RZ opportunity from play-by-play (2016-25) and asks the accuracy question Doug
posed: does an RZ-opportunity model predict anytime-TD meaningfully BETTER than a naive
recency-TD baseline, and is it well-calibrated?

  actual   = (rushing_tds + receiving_tds) >= 1 that game (from player_week).
  RZ model = lambda = exp_rz_carries*r_c + exp_rz_targets*r_t + exp_touches*r_long
             (rates fit on TRAIN); P(anytime) = 1 - exp(-lambda).
  baseline = recency-weighted anytime-TD frequency (no RZ signal).
Train 2016-2022, test 2023-2025 (disjoint eras). Metrics: Brier + log-loss + calibration.
No market odds needed (none exist yet); pure predictive-accuracy test. No spend.
"""
import argparse
import math
from collections import defaultdict

import nfl_data
import nfl_props_scan as scan

HALF_LIFE = 4
MIN_PRIOR = 3


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _rz_by_game(seasons):
    """{(player_id, season, week): {rz_c, rz_t, gl_c, touches}} from pbp red-zone plays,
    plus an id->name map from player_week."""
    rz = defaultdict(lambda: defaultdict(float))
    for s in seasons:
        pbp = nfl_data._read_layer("nfl_pbp", str(s))
        if pbp is None:
            continue
        cols = set(pbp.columns)
        for r in pbp.to_dict("records"):
            y = _num(r.get("yardline_100"))
            wk = r.get("week")
            if y is None or wk is None:
                continue
            try:
                wk = int(wk)
            except (TypeError, ValueError):
                continue
            pt = r.get("play_type")
            if pt == "run" and r.get("rusher_player_id"):
                k = (r["rusher_player_id"], str(s), wk)
                rz[k]["touches"] += 1
                if y <= 20:
                    rz[k]["rz_c"] += 1
                if y <= 5:
                    rz[k]["gl_c"] += 1
            elif pt == "pass" and r.get("receiver_player_id"):
                k = (r["receiver_player_id"], str(s), wk)
                rz[k]["touches"] += 1
                if y <= 20:
                    rz[k]["rz_t"] += 1
    return rz


def _series(seasons):
    """{(pid, season): [(week, rz_c, rz_t, gl_c, touches, rush_td, rec_td, name)] desc}."""
    rz = _rz_by_game(seasons)
    pw = nfl_data.player_week([str(s) for s in seasons])
    out = defaultdict(list)
    if pw is None:
        return out
    namecol = "player_display_name" if "player_display_name" in pw.columns else "player_name"
    for r in pw.to_dict("records"):
        pid = r.get("player_id")
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        s = str(r.get("season"))
        rzg = rz.get((pid, s, wk), {})
        rtd = _num(r.get("rushing_tds")) or 0.0
        retd = _num(r.get("receiving_tds")) or 0.0
        out[(pid, s)].append((wk, rzg.get("rz_c", 0.0), rzg.get("rz_t", 0.0),
                              rzg.get("gl_c", 0.0), rzg.get("touches", 0.0),
                              rtd, retd, r.get(namecol)))
    for k in out:
        out[k].sort(key=lambda x: -x[0])
    return out


def _rw(n):
    return [0.5 ** (i / HALF_LIFE) for i in range(n)]


def build(seasons):
    """as-of obs: recency RZ opportunity from prior weeks + actual anytime-TD."""
    ser = _series(seasons)
    obs = []
    for (pid, s), rows in ser.items():
        for i, g in enumerate(rows):
            wk = g[0]
            prior = [r for r in rows if r[0] < wk]
            if len(prior) < MIN_PRIOR:
                continue
            w = _rw(len(prior)); wsum = sum(w) or 1.0
            def rec(idx):
                return sum(wi * r[idx] for r, wi in zip(prior, w)) / wsum
            actual = 1 if (g[5] + g[6]) >= 1 else 0
            td_hist = [1 if (r[5] + r[6]) >= 1 else 0 for r in prior]
            obs.append({"pid": pid, "season": s, "week": wk, "actual": actual,
                        "rz_c": rec(1), "rz_t": rec(2), "gl_c": rec(3),
                        "touches": rec(4),
                        "naive": sum(wi * h for h, wi in zip(td_hist, w)) / wsum})
    return obs


def _fit_rates(train):
    """MLE-ish TD rates per RZ-opportunity channel on train (total TDs / total opp)."""
    sc = sum(o["rz_c"] for o in train); st = sum(o["rz_t"] for o in train)
    sto = sum(o["touches"] for o in train)
    # actual expected TDs proxy: use anytime rate to anchor; fit per-unit via regression-free ratios
    tot_td = sum(o["actual"] for o in train)   # anytime count (>=1), approx for rate anchoring
    # simple non-negative rates: split league TDs proportional to each channel's volume
    # r_c, r_t from RZ volume; r_long tiny from total touches
    rz_vol = sc + st
    if rz_vol <= 0:
        return 0.0, 0.0, 0.0
    # calibrate lambda so mean P(>=1) matches observed base rate
    base = tot_td / len(train)
    # start from volume shares, then scale to match base rate
    r_c = 1.0; r_t = 1.0; r_long = 0.02
    def mean_p(rc, rt, rl):
        return sum(1 - math.exp(-(o["rz_c"] * rc + o["rz_t"] * rt + o["touches"] * rl))
                   for o in train) / len(train)
    # scale r_c,r_t,r_long uniformly to hit base rate
    lo, hi = 0.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if mean_p(r_c * mid, r_t * mid, r_long * mid) < base:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    return r_c * k, r_t * k, r_long * k


def _p_rz(o, rates):
    rc, rt, rl = rates
    lam = o["rz_c"] * rc + o["rz_t"] * rt + o["touches"] * rl
    return 1 - math.exp(-lam)


def _metrics(obs, fn):
    n = len(obs)
    brier = sum((fn(o) - o["actual"]) ** 2 for o in obs) / n
    ll = -sum(o["actual"] * math.log(max(1e-6, fn(o)))
              + (1 - o["actual"]) * math.log(max(1e-6, 1 - fn(o))) for o in obs) / n
    return brier, ll


def _calib(obs, fn):
    """10-bin reliability: mean predicted vs mean actual, ECE."""
    bins = defaultdict(lambda: [0.0, 0.0, 0])
    for o in obs:
        p = fn(o); b = min(9, int(p * 10))
        bins[b][0] += p; bins[b][1] += o["actual"]; bins[b][2] += 1
    ece = sum(abs(v[0] / v[2] - v[1] / v[2]) * v[2] for v in bins.values()) / len(obs)
    return ece


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="2016,2017,2018,2019,2020,2021,2022")
    ap.add_argument("--test", default="2023,2024,2025")
    args = ap.parse_args()
    tr_s = [s.strip() for s in args.train.split(",") if s.strip()]
    te_s = [s.strip() for s in args.test.split(",") if s.strip()]
    print("=" * 88)
    print(f"  NFL ANYTIME-TD MODEL — RZ opportunity vs naive recency (train {tr_s[0]}-{tr_s[-1]} → test {te_s})")
    print("=" * 88)
    train = build(tr_s)
    test = build(te_s)
    print(f"  train obs={len(train):,}  test obs={len(test):,}  test base-rate(anytime TD)="
          f"{sum(o['actual'] for o in test)/len(test)*100:.1f}%")
    rates = _fit_rates(train)
    print(f"  fitted RZ rates: per-rz-carry={rates[0]:.3f}  per-rz-target={rates[1]:.3f}  "
          f"per-touch(long)={rates[2]:.4f}")
    for name, fn in [("naive recency-TD", lambda o: o["naive"]),
                     ("RZ-opportunity", lambda o: _p_rz(o, rates))]:
        b, ll = _metrics(test, fn)
        ece = _calib(test, fn)
        print(f"    {name:<20} Brier={b:.4f}  logloss={ll:.4f}  ECE={ece:.4f}")
    print("  READ: RZ Brier/logloss BELOW naive ⇒ red-zone signal makes TD prediction better;")
    print("  low ECE ⇒ well-calibrated. (Market-trustworthiness needs anytime-TD odds — separate.)")
    print("=" * 88)


if __name__ == "__main__":
    main()
