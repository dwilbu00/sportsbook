"""nfl_count_props_process.py — PROCESS-first model for the pure-COUNT NFL props.

rush_attempts / pass_attempts / completions / pass_tds are direct counts — the outcome
IS the count, so there is no volume×conversion decomposition (that layer earned ~nothing
on receptions/rush_yds/recv_yds anyway). The whole game here is the VARIANCE: the naive
count model is Poisson (variance = mean); real football counts are OVER-dispersed (game
script swings attempts 20↔45), so the fix is a NegBin (variance = mean + φ·mean²) with φ
fit on train. This is the same "variance done right" lever, in its purest form.

Nested models per prop (all → P(count >= k), k = int(line)+1):
  M0 incumbent A : empirical over-rate at the line over prior weeks (baseline hit-rate).
  M1 Poisson     : mean = recency-weighted count, Poisson variance (the naive analog).
  M2 NegBin      : same mean, over-dispersion φ fit on train — the variance FIX.
  market         : de-vigged two-way (the sharp reference).

Confirms whether the NegBin count-fix that was the giant lever for receptions is as
dominant for attempts/completions/TDs. Same DEFAULTS (HALF_LIFE/MIN_PRIOR) as the other
props — generalization tested on identical knobs before any sweep. Leakage-safe: features
use only strictly-earlier weeks; φ fit on the train fold, scored on the held-out half
(2-fold). Diagnostic only. Reuses nfl_props_scan loaders.
"""
import argparse
import math
from collections import defaultdict

import nfl_schedule
import nfl_data
import nfl_props_scan as scan
from stats import negbin_at_least, fit_negbin_dispersion
from odds_client import (american_to_implied_prob, devig_two_way)

SPORT = "americanfootball_nfl"
COUNT_PROPS = [
    ("player_rush_attempts", "carries"),
    ("player_pass_attempts", "attempts"),
    ("player_pass_completions", "completions"),
    ("player_pass_tds", "passing_tds"),
]

HALF_LIFE = 4
MIN_PRIOR = 3


def _recency_weights(n, half_life):
    if n <= 0:
        return []
    if not half_life or half_life <= 0:
        return [1.0] * n
    return [0.5 ** (i / half_life) for i in range(n)]


def _series(seasons, stat_col):
    """{(norm_name, season): [(week, value), ...] most-recent-first} for one stat."""
    pw = nfl_data.player_week([str(s) for s in seasons])
    if pw is None or stat_col not in pw.columns:
        return {}
    cols = set(pw.columns)
    namecol = "player_display_name" if "player_display_name" in cols else "player_name"

    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    series = defaultdict(list)
    for r in pw.to_dict("records"):
        try:
            wk = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        nm = scan._norm(r.get(namecol))
        if not nm:
            continue
        series[(nm, str(r.get("season")))].append((wk, num(r.get(stat_col))))
    for k in series:
        series[k].sort(key=lambda x: -x[0])
    return series


def build_corpus(prop_key, stat_col, seasons, snapshot, book, idx, pw_stat, snaps):
    props = scan._load_props(seasons, snapshot, book)
    series = _series(seasons, stat_col)
    obs = []
    n_props = n_thin = 0
    for (eid, player, pk), d in props.items():
        if pk != prop_key:
            continue
        n_props += 1
        gid, _ = nfl_schedule.resolve_event(d["home"], d["away"], d["commence"],
                                            index=idx)
        if gid is None:
            continue
        parts = gid.split("_")
        season, week = parts[0], int(parts[1])
        nm = scan._norm(player)
        actual = (pw_stat.get((nm, season, str(week))) or {}).get(stat_col)
        if actual is None:
            if snaps.get((nm, season, str(week))):
                actual = 0.0
            else:
                continue
        prior = [row for row in series.get((nm, season), []) if row[0] < week]
        if len(prior) < MIN_PRIOR:
            n_thin += 1
            continue
        w = _recency_weights(len(prior), HALF_LIFE)
        vw = sum(wi * (v or 0.0) for (_wk, v), wi in zip(prior, w))
        exp_ct = vw / (sum(w) or 1.0)
        hist = [v for (_wk, v) in prior if v is not None]
        line = d.get("OVER", d.get("UNDER", (None, None)))[0]
        if line is None:
            continue
        emp_over = (sum(1 for v in hist if v > line) / len(hist) if hist else 0.5)
        over, under = d.get("OVER"), d.get("UNDER")
        obs.append({
            "season": season, "line": line, "actual": float(actual),
            "over_price": over[1] if over else None,
            "under_price": under[1] if under else None,
            "exp_ct": exp_ct, "emp_over": emp_over,
        })
    return obs, {"n_props": n_props, "n_thin": n_thin, "n_obs": len(obs)}


def _predict(o, model, phi):
    k = int(o["line"]) + 1
    if model == "M0":
        return max(0.0, min(1.0, o["emp_over"]))
    if model == "M1":                       # Poisson (dispersion 0)
        return negbin_at_least(k, o["exp_ct"], 0.0)
    return negbin_at_least(k, o["exp_ct"], phi)   # M2 NegBin


def _brier(sub, fn):
    return sum((fn(o) - o["y"]) ** 2 for o in sub) / len(sub) if sub else None


def probe_prop(prop_key, stat_col, seasons, snapshot, book, idx, pw_stat, snaps):
    obs, meta = build_corpus(prop_key, stat_col, seasons, snapshot, book,
                             idx, pw_stat, snaps)
    print("-" * 100)
    print(f"  {prop_key}  ({stat_col})  —  built {meta['n_obs']:,} obs  "
          f"(props={meta['n_props']}, thin<{MIN_PRIOR}={meta['n_thin']})")
    rows = []
    for o in obs:
        if abs(o["actual"] - o["line"]) < 1e-9:
            continue
        o["y"] = 1 if o["actual"] > o["line"] else 0
        if o["over_price"] is not None and o["under_price"] is not None:
            try:
                o["mkt_over"] = devig_two_way(
                    american_to_implied_prob(o["over_price"]),
                    american_to_implied_prob(o["under_price"]))[0]
            except Exception:
                o["mkt_over"] = None
        else:
            o["mkt_over"] = None
        rows.append(o)
    if len(rows) < 200:
        print(f"    only {len(rows)} usable obs (<200) — too thin, skipping.")
        return
    rows.sort(key=lambda r: (r["season"]))
    split = len(rows) // 2
    folds = [(rows[:split], rows[split:]), (rows[split:], rows[:split])]
    agg = {m: [0.0, 0] for m in ("M0", "M1", "M2")}
    mkt_agg = [0.0, 0]
    phis = []
    for tr, te in folds:
        phi = fit_negbin_dispersion([(o["exp_ct"], o["actual"]) for o in tr], cap=2.0)
        phis.append(phi)
        for m in ("M0", "M1", "M2"):
            b = _brier(te, lambda o, mm=m, pp=phi: _predict(o, mm, pp))
            agg[m][0] += b * len(te)
            agg[m][1] += len(te)
        mk = [o for o in te if o["mkt_over"] is not None]
        mb = _brier(mk, lambda o: o["mkt_over"])
        if mb is not None:
            mkt_agg[0] += mb * len(mk)
            mkt_agg[1] += len(mk)
    pooled = {m: agg[m][0] / agg[m][1] for m in agg}
    base = pooled["M0"]
    print(f"    pooled 2-fold OOS Brier (φ={phis[0]:.2f}/{phis[1]:.2f}):  "
          f"hit-rate A={pooled['M0']:.4f}   "
          f"Poisson={pooled['M1']:.4f} (Δ{base - pooled['M1']:+.4f})   "
          f"NegBin={pooled['M2']:.4f} (Δ{base - pooled['M2']:+.4f})   "
          f"market={mkt_agg[0] / mkt_agg[1]:.4f}"
          if mkt_agg[1] else "market=NA")
    # per season
    seasons_seen = sorted({o["season"] for o in rows})
    for s in seasons_seen:
        te = [o for o in rows if o["season"] == s]
        tr = [o for o in rows if o["season"] != s]
        if len(te) < 50 or len(tr) < 200:
            continue
        phi = fit_negbin_dispersion([(o["exp_ct"], o["actual"]) for o in tr], cap=2.0)
        mk = [o for o in te if o["mkt_over"] is not None]
        print(f"      {s}: n={len(te):>5}  A={_brier(te, lambda o: o['emp_over']):.4f}  "
              f"Pois={_brier(te, lambda o: _predict(o, 'M1', 0.0)):.4f}  "
              f"NB={_brier(te, lambda o, pp=phi: _predict(o, 'M2', pp)):.4f}  "
              f"mkt={(_brier(mk, lambda o: o['mkt_over']) if mk else float('nan')):.4f}")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025")
    ap.add_argument("--snapshot", default="closing", choices=["closing", "early_4h"])
    ap.add_argument("--book", default="draftkings")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    idx = nfl_schedule.game_index([str(s) for s in seasons])
    pw_stat = scan._player_week_index(seasons)
    snaps = scan._snap_presence([str(s) for s in seasons])
    print("=" * 100)
    print(f"  NFL COUNT PROPS — NegBin vs Poisson vs hit-rate (DK {args.snapshot}, {seasons})")
    print("  READ: M1→M2 (Poisson→NegBin) is the pure variance/over-dispersion fix; any")
    print("  model below hit-rate A ships. Market Brier is the accuracy ceiling.")
    print("=" * 100)
    for prop_key, stat_col in COUNT_PROPS:
        probe_prop(prop_key, stat_col, seasons, args.snapshot, args.book,
                   idx, pw_stat, snaps)
    print("=" * 100)


if __name__ == "__main__":
    main()
