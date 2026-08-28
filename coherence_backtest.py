"""Coherence backtest: bet DK's run-line where it contradicts DK's OWN moneyline +
total, grade vs the final score. Needs no sharper book — the edge is DK's internal
inconsistency (one of its own prices must be wrong).

Flow per event (DK close snapshot): devig ML -> P_home_win, total -> P_over, RL ->
DK's own P_home_cover. Solve run means from ML+total, read the IMPLIED P_home_cover
(coherence.py), and bet DK's run-line at DK's price using that implied cover as
truth. Grade home_covered = (home_score + rl_home_point > away_score).

CRITICAL calibration guard (the analog of the props projector trap): before trusting
any residual, the report prints the MEAN incoherence (implied - DK's RL fair) across
all events. If that's ~0, the run model is coherent with DK on average, so the
per-event outliers are genuine mispricings. If it's systematically off, the model
(or its dispersion) is biased and the "edge" would be model error — so we'd tune the
dispersion to zero the mean before believing any bucket.

Cache-first like r2_backtest; reuses its haircut/gate/BH helpers. Run:
  python coherence_backtest.py --seasons 2024,2025,2026
"""
import argparse
import os
import pickle
from collections import Counter

import coherence
import r2_backtest as rbt
import r2_data
import r2_grade
from odds_client import american_to_decimal
from r2_sharp import fair_two_way

_CACHE_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "r2_backtest_cache")


def _incoh_bucket(x):
    a = abs(x)
    if a <= 0.02:
        return "<=0.02"
    if a <= 0.05:
        return "(0.02,0.05]"
    if a <= 0.10:
        return "(0.05,0.10]"
    return ">0.10"


def _fav_bucket(imp):
    """Favorite-strength bin (matches scenario_backtest.dog_runline) so we can test
    whether coherence's EV-selected edge concentrates in the SAME dog+1.5 /
    moderate-favorite regime the scenario screen surfaced."""
    for hi in (0.55, 0.60, 0.65, 0.70):
        if imp < hi:
            return f"<{hi:.0%}"
    return ">=70%"


def grade_coherence(triads_by_season, scores_idx, dispersion=0.0, haircut=0.02,
                    ev_floor=0.03, bias=0.0):
    """Score + grade the run-line coherence bet for every event. Returns (rows, cov).

    ``bias`` is the calibration offset (mean implied-minus-DK-RL over all events):
    subtracting it makes the run model coherent with DK ON AVERAGE, so the remaining
    per-event residual is a genuine DK inconsistency rather than the Poisson shape
    error (the run-line-favorite one-run-game bias). Each row carries the CALIBRATED
    `incoh` = (implied - bias) - DK RL fair."""
    rows, cov = [], Counter()
    for season, triads in triads_by_season.items():
        # bias may be a scalar (global) or {season: offset} (leave-one-season-out,
        # so each season is calibrated from the OTHER seasons = out-of-sample).
        b = bias.get(str(season), 0.0) if isinstance(bias, dict) else bias
        for t in triads:
            cov["events"] += 1
            ml_home_fair, _ = fair_two_way(t.ml_home, t.ml_away)
            over_fair, _ = fair_two_way(t.total_over, t.total_under)
            rl_home_fair, _ = fair_two_way(t.rl_home, t.rl_away)
            if None in (ml_home_fair, over_fair, rl_home_fair):
                cov["dropped_undevigable"] += 1
                continue
            implied = coherence.implied_home_cover(
                ml_home_fair, t.total_line, over_fair, t.rl_home_point, dispersion)
            if implied is None:
                cov["dropped_unsolvable"] += 1
                continue
            implied = min(1.0 - 1e-6, max(1e-6, implied - b))      # calibrated fair
            incoh = implied - rl_home_fair          # + => DK underprices home cover
            cov["priced"] += 1
            fav_imp = max(ml_home_fair, 1.0 - ml_home_fair)   # favorite strength
            # EV of each DK run-line side under the implied (coherent) cover prob.
            legs = [("home", t.rl_home, implied), ("away", t.rl_away, 1.0 - implied)]
            for side, price, fair in legs:
                try:
                    dec = american_to_decimal(int(price)) * (1.0 - haircut)
                except (TypeError, ValueError):
                    continue
                evh = fair * dec - 1.0
                if evh < ev_floor:
                    continue
                cov["selected"] += 1
                if t.game_pk is None or int(t.game_pk) not in scores_idx:
                    cov["dropped_no_score"] += 1
                    continue
                hs, as_ = scores_idx[int(t.game_pk)]
                home_covered = (hs + t.rl_home_point) > as_
                won = home_covered if side == "home" else (not home_covered)
                # Which structural side is this bet? +1.5 = underdog, -1.5 = favorite.
                bet_point = t.rl_home_point if side == "home" else -t.rl_home_point
                bet_side_type = "dog+1.5" if bet_point > 0 else "fav-1.5"
                rows.append({
                    "season": str(season), "prop_key": "run_line", "side": side,
                    "incoh": incoh, "incoh_bucket": _incoh_bucket(incoh),
                    "ev": evh, "ev_bucket": r2_grade.ev_bucket(evh),
                    "implied": implied, "dk_rl_fair": rl_home_fair,
                    "fav_imp": fav_imp, "fav_bucket": _fav_bucket(fav_imp),
                    "bet_side_type": bet_side_type,
                    "result": "win" if won else "loss",
                    "profit": (dec - 1.0) if won else -1.0,
                    "game_pk": t.game_pk,
                })
                cov["graded"] += 1
    return rows, cov


def build_coherence_report(rows, coverage, all_incoh, min_n=100, min_seasons=2,
                           min_t=2.0):
    out = []
    p = out.append
    p("=" * 74)
    p("  Coherence backtest — DK run-line vs DK's own ML+total")
    p("=" * 74)
    p("\n  Coverage:")
    for k in ("events", "priced", "selected", "graded", "dropped_undevigable",
              "dropped_unsolvable", "dropped_no_score"):
        if coverage.get(k):
            p(f"    {k:<22} {coverage[k]:>8,}")

    # CALIBRATION GUARD: mean incoherence across ALL priced events (not just bets).
    if all_incoh:
        n = len(all_incoh)
        mean = sum(all_incoh) / n
        var = sum((x - mean) ** 2 for x in all_incoh) / (n - 1) if n > 1 else 0.0
        se = (var / n) ** 0.5 if n else 0.0
        t = mean / se if se > 0 else 0.0
        p(f"\n  Model calibration (implied - DK RL fair) over {n:,} priced events:")
        p(f"    mean bias={mean:+.4f} (t={t:+.2f})  "
          f"{'OK ~coherent on average' if abs(t) < 3 else 'SYSTEMATIC BIAS -> tune dispersion'}")
        p("    (a real per-event residual is only trustworthy if this mean ~0)")

    # Verdict: pooled + per season, hardened gate.
    passed, reason, per, pooled = rbt.hardened_gate(rows, min_n, min_seasons, min_t)
    p(f"\n  Run-line coherence bet: pooled n={pooled.n:,} ROI={pooled.roi:+.2%} "
      f"hit={pooled.hit_rate:.1%} t={pooled.t_stat:.2f}")
    for s in sorted(per):
        sm = per[s]
        tag = "" if sm.decided >= min_n else "  (insufficient)"
        p(f"    {s}: n={sm.decided:,} ROI={sm.roi:+.2%} t={sm.t_stat:.2f}{tag}")
    p(f"    GATE: {'PASS' if passed else 'FAIL'} — {reason}")

    # By incoherence magnitude (does a bigger contradiction pay more?) + by side.
    p("\n  By incoherence magnitude |implied - DK RL fair|:")
    cells = r2_grade.by_key(rows, lambda r: r["incoh_bucket"])
    for b in ("<=0.02", "(0.02,0.05]", "(0.05,0.10]", ">0.10"):
        if b in cells:
            sm = cells[b]
            tag = "" if sm.decided >= min_n else "  (insufficient)"
            p(f"    {b:<12} n={sm.decided:>5,} ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}{tag}")
    p("\n  By side:")
    for side in ("home", "away"):
        sm = r2_grade.summarize([r for r in rows if r["side"] == side])
        p(f"    {side:<6} n={sm.decided:>5,} ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}")

    # CONVERGENCE CHECK vs scenario_backtest's dog+1.5 finding: is coherence's
    # EV-selected edge the SAME inefficiency (dog +1.5, moderate favorites)?
    p("\n  By structural side (dog +1.5 vs favorite -1.5):")
    for bt in ("dog+1.5", "fav-1.5"):
        sm = r2_grade.summarize([r for r in rows if r["bet_side_type"] == bt])
        p(f"    {bt:<8} n={sm.decided:>5,} ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}")
    p("\n  DOG +1.5 bets by favorite strength (scenario sweet spot = 65-70%):")
    dogs = [r for r in rows if r["bet_side_type"] == "dog+1.5"]
    cells = r2_grade.by_key(dogs, lambda r: r["fav_bucket"])
    for b in ("<55%", "<60%", "<65%", "<70%", ">=70%"):
        if b in cells:
            sm = cells[b]
            tag = "" if sm.decided >= 30 else "  (thin)"
            p(f"    {b:<8} n={sm.decided:>5,} ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}{tag}")

    # Realized ROI vs MODELED EV — a real edge is monotone (bigger modeled EV ->
    # bigger realized ROI); a flat/noisy pattern says the signal isn't structural.
    p("\n  By modeled-EV bucket (should rise with EV if the edge is real):")
    cells = r2_grade.by_key(rows, lambda r: r["ev_bucket"])
    order = ["0%-2%", "2%-5%", "5%-10%", "10%-20%", ">=20%"]
    for b in [x for x in order if x in cells] + [k for k in sorted(cells) if k not in order]:
        sm = cells[b]
        tag = "" if sm.decided >= min_n else "  (insufficient)"
        p(f"    EV {b:<8} n={sm.decided:>5,} ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}{tag}")
    p("=" * 74)
    text = "\n".join(out)
    print(text)
    return {"passed": passed, "pooled": pooled, "text": text}


def _cache_path(sport, seasons):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = f"coh_{sport}_{'-'.join(map(str, seasons))}"
    return os.path.join(_CACHE_DIR, tag.replace("/", "_") + ".pkl")


def load_or_fetch(sport, seasons, refresh=False, refresh_mirror=False):
    path = _cache_path(sport, seasons)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    import warehouse_mirror
    warehouse_mirror.autobuild(sport, seasons, refresh=refresh_mirror)
    triads_by_season, stats = r2_data.load_team_triad(sport, seasons)
    scores_idx = r2_data.build_team_scores_index(seasons)
    blob = {"triads_by_season": triads_by_season, "scores_idx": scores_idx,
            "stats": {s: dict(c) for s, c in stats.items()}}
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    return blob, path


def main():
    ap = argparse.ArgumentParser(description="Coherence backtest (DK run-line vs DK ML+total).")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--dispersion", type=float, default=0.0,
                    help="Team-run NegBin dispersion for the coherence model (0=Poisson).")
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--ev-floor", type=float, default=0.03)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-seasons", type=int, default=2)
    ap.add_argument("--min-t", type=float, default=2.0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--refresh-mirror", action="store_true",
                    help="re-sync + re-verify the parquet mirror (with ODI_BACKTEST_MIRROR)")
    ap.add_argument("--raw", action="store_true",
                    help="Skip calibration (raw model-biased result).")
    ap.add_argument("--global-bias", action="store_true",
                    help="Use one in-sample offset (leaky) instead of the default "
                         "leave-one-season-out out-of-sample calibration.")
    args = ap.parse_args()
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    blob, path = load_or_fetch(args.sport, seasons, refresh=args.refresh,
                               refresh_mirror=args.refresh_mirror)
    n = sum(len(v) for v in blob["triads_by_season"].values())
    print(f"  data: {n:,} team triads (DK ML+RL+total)  (cache: {path})")

    # Precompute the per-season incoherence distribution for the calibration guard
    # + the leave-one-season-out offsets.
    incoh_by_season = {}
    for season, triads in blob["triads_by_season"].items():
        vals = []
        for t in triads:
            mlf, _ = fair_two_way(t.ml_home, t.ml_away)
            ovf, _ = fair_two_way(t.total_over, t.total_under)
            rlf, _ = fair_two_way(t.rl_home, t.rl_away)
            if None in (mlf, ovf, rlf):
                continue
            impl = coherence.implied_home_cover(mlf, t.total_line, ovf,
                                                t.rl_home_point, args.dispersion)
            if impl is not None:
                vals.append(impl - rlf)
        incoh_by_season[str(season)] = vals
    all_incoh = [v for vals in incoh_by_season.values() for v in vals]

    # Calibration: default = LEAVE-ONE-SEASON-OUT (each season's offset from the
    # OTHER seasons = out-of-sample, no leakage). --global-bias = in-sample single
    # offset (leaky, for comparison). --raw = none.
    if args.raw:
        bias = 0.0
        print("  calibration: NONE (--raw; model-biased)")
    elif args.global_bias:
        bias = sum(all_incoh) / len(all_incoh) if all_incoh else 0.0
        print(f"  calibration: GLOBAL in-sample offset {bias:+.4f} (leaky, comparison only)")
    else:
        bias = {}
        for s in incoh_by_season:
            others = [v for ss, vals in incoh_by_season.items() if ss != s for v in vals]
            bias[s] = (sum(others) / len(others)) if others else (
                sum(incoh_by_season[s]) / len(incoh_by_season[s]) if incoh_by_season[s] else 0.0)
        print("  calibration: LEAVE-ONE-SEASON-OUT (out-of-sample) offsets = "
              + ", ".join(f"{s}:{bias[s]:+.4f}" for s in sorted(bias)))

    rows, cov = grade_coherence(blob["triads_by_season"], blob["scores_idx"],
                                dispersion=args.dispersion, haircut=args.haircut,
                                ev_floor=args.ev_floor, bias=bias)
    build_coherence_report(rows, cov, all_incoh, min_n=args.min_n,
                           min_seasons=args.min_seasons, min_t=args.min_t)
    print(f"\n  params: dispersion={args.dispersion} haircut={args.haircut} "
          f"ev_floor={args.ev_floor} min_n={args.min_n} calibrated={not args.raw}")


if __name__ == "__main__":
    main()
