"""R2 DK-vs-sharp backtest driver: does betting DraftKings where its price is +EV
against the Pinnacle no-vig fair (at DK's own line) actually profit out-of-sample?

Ties the pieces together per the red-team spec (wf_46cca99d):
  r2_data (paired DK+Pinnacle close legs + game_pk outcome index)
    -> r2_edge (EV = sharp_fair * dk_decimal - 1, same-line or projected)
    -> vig/CLV HAIRCUT on the DK price + a pre-registered EV FLOOR (not ev>0)
    -> r2_grade (game_pk-exact win/loss/push -> realized profit)
    -> hardened per-season OOS gate + full grid + Benjamini-Hochberg + coverage.

The verdict discipline (why this is more than "pooled ROI > 0"):
  * PRIMARY hypothesis = same-line (distance==0, pure devig, model-free) on the
    market with a real prior of edge (pitcher_earned_runs). Projected/cross-line
    legs are a SEPARATE lower-prior arm, never folded into the primary verdict.
  * A vig haircut + EV floor kills the winner's-curse tail (small-EV legs dominated
    by devig-rounding noise, exactly where DK limits are lowest).
  * The gate requires >=2 seasons each over a real min-n with positive ROI AND a
    pooled t>=2 on haircut profit — r2_grade.replicates_per_season alone passes on a
    single season and ignores t, so we harden it here.
  * The full market x arm x season grid prints with per-cell n+t, thin cells marked
    'insufficient', and BH across the family — so no lucky thin cell masquerades.

Fetch is cached locally (pickle) so re-running the report with a different haircut /
EV floor / gate never re-hits Azure. --refresh re-fetches.

Grading + report are pure (grade_legs/build_report take data in, no I/O) so they
unit-test on synthetic legs. Run: python r2_backtest.py --seasons 2024,2025,2026
"""
import argparse
import math
import os
import pickle
from collections import Counter, defaultdict

from odds_client import american_to_decimal
import r2_data
import r2_edge
import r2_grade

# R2-PRICEABLE props = the ones Pinnacle actually books two-sided (verified against
# the warehouse 2026-08-27): Pinnacle posts ZERO batter_hits/batter_strikeouts/
# batter_rbis, so R2 (which needs a sharp reference) structurally cannot price them —
# they were dropping to zero paired legs. Pinnacle DOES book the pitcher props +
# batter_total_bases (densely: ~2,536 events). earned_runs stays PRIMARY (our one
# validated keeper); TB is the largest-sample batter prop and 100% gradeable
# (the legacy NULL-TB caveat is stale — re-ingest filled it).
DEFAULT_PROPS = ["pitcher_earned_runs", "pitcher_strikeouts", "pitcher_outs",
                 "batter_total_bases", "batter_hits"]
PRIMARY_PROP = "pitcher_earned_runs"
_CACHE_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "r2_backtest_cache")


# ── EV / profit with a vig haircut (the pure modules take raw american) ───────

def _haircut_decimal(american, h):
    """DK decimal odds shaded down by haircut h in [0,1) for vig/limit realism."""
    return american_to_decimal(int(american)) * (1.0 - h)


def ev_haircut(sharp_fair, american, h):
    """EV per $1 at the shaded DK price under the sharp fair prob."""
    return sharp_fair * _haircut_decimal(american, h) - 1.0


def profit_haircut(american, result, h):
    """Realized profit per $1 at the shaded price: win -> shaded_decimal-1, loss ->
    -1, push -> 0, None on a non-graded result."""
    if result == "push":
        return 0.0
    if result not in ("win", "loss"):
        return None
    return (_haircut_decimal(american, h) - 1.0) if result == "win" else -1.0


# ── Grading (pure: legs + outcome index in, graded rows out) ──────────────────

def grade_legs(legs_by_season, outcome_idx, haircut, ev_floor,
               default_dispersion=0.0, resolve_fn=None):
    """Score + grade every leg. Returns (rows, coverage).

    A row is one SELECTED bet (a DK side whose haircut EV clears ev_floor) that was
    gradeable by exact (player_mlb_id, game_pk). resolve_fn(name, season, prop_key)
    -> mlb_id|None recovers a NULL player id (drop+count on None). Never coalesces a
    missing actual to 0."""
    rows, cov = [], Counter()
    for season, legs in legs_by_season.items():
        for lg in legs:
            cov["legs"] += 1
            edges = r2_edge.prop_leg_edges(
                lg.dk_point, lg.dk_over_price, lg.dk_under_price,
                lg.pinnacle_offers, default_dispersion=default_dispersion)
            if not edges:
                cov["legs_no_pinnacle_fair"] += 1
                continue
            # Resolve a NULL player id ONCE per leg (shared by its sides).
            pid = lg.player_mlb_id
            if pid is None and resolve_fn is not None:
                try:
                    pid = resolve_fn(lg.player, int(str(lg.game_date)[:4]), lg.prop_key)
                except (TypeError, ValueError):
                    pid = None
            for e in edges:
                evh = ev_haircut(e.sharp_fair, e.dk_price, haircut)
                if evh < ev_floor:
                    continue
                cov["selected"] += 1
                if pid is None:
                    cov["dropped_null_id"] += 1
                    continue
                if lg.game_pk is None:
                    cov["dropped_null_game_pk"] += 1
                    continue
                actual = r2_data.outcome_value(outcome_idx, lg.prop_key, pid, lg.game_pk)
                if actual is None:
                    cov["dropped_no_actual"] += 1
                    continue
                result = r2_grade.grade_over_under(actual, e.point, e.side)
                if result is None:
                    cov["dropped_ungradable"] += 1
                    continue
                rows.append({
                    "season": str(season), "prop_key": lg.prop_key, "side": e.side,
                    "arm": "projected" if e.projected else "same_line",
                    "ref_prop": lg.ref_prop,   # non-None -> priced via a synonym (hits<-TB)
                    "dk_point": e.point, "dk_price": e.dk_price,
                    "sharp_fair": e.sharp_fair, "ev_raw": e.ev, "ev": evh,
                    "ev_bucket": r2_grade.ev_bucket(evh), "distance": e.distance,
                    "n_lines": e.n_lines, "actual": actual, "result": result,
                    "profit": profit_haircut(e.dk_price, result, haircut),
                    "profit_raw": r2_grade.profit(e.dk_price, result),
                    "game_pk": lg.game_pk, "player_mlb_id": pid,
                    "event_id": lg.event_id,
                })
                cov["graded"] += 1
    return rows, cov


# ── Significance helpers ──────────────────────────────────────────────────────

def dist_bucket(distance):
    """Bucket the line-disagreement magnitude (|DK point - nearest Pinnacle point|)
    for the projected arm — the test of whether a BIGGER line gap (the Pinnacle-1.5-
    vs-DK-0.5 bullseye) carries more realized edge."""
    if distance is None:
        return "n/a"
    if distance <= 0.5:
        return "<=0.5"
    if distance <= 1.0:
        return "(0.5,1.0]"
    if distance <= 1.5:
        return "(1.0,1.5]"
    return ">1.5"


def _one_sided_p(t):
    """One-sided upper p-value for a t/z stat via the normal approx (erfc)."""
    return 0.5 * math.erfc(t / math.sqrt(2.0))


def benjamini_hochberg(pvals, alpha=0.05):
    """Return the set of indices passing BH control at level alpha."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    passing, thresh = set(), 0
    for rank, i in enumerate(idx, start=1):
        if pvals[i] <= alpha * rank / m:
            thresh = rank
    for rank, i in enumerate(idx, start=1):
        if rank <= thresh:
            passing.add(i)
    return passing


def hardened_gate(rows, min_n, min_seasons=2, min_t=2.0):
    """The per-season OOS gate, hardened past r2_grade.replicates_per_season (which
    passes on ONE judged season and ignores t). Requires >=min_seasons judged
    seasons (decided-n>=min_n) EACH with roi>0, AND pooled haircut-profit t>=min_t.
    Returns (passed, reason, per_season, pooled)."""
    per = r2_grade.by_key(rows, lambda r: r["season"])
    judged = {s: sm for s, sm in per.items() if sm.decided >= min_n}
    pooled = r2_grade.summarize(rows)
    if len(judged) < min_seasons:
        return False, f"only {len(judged)} season(s) with n>={min_n} (need {min_seasons})", per, pooled
    neg = [s for s, sm in judged.items() if sm.roi <= 0]
    if neg:
        return False, f"judged season(s) not positive: {','.join(sorted(neg))}", per, pooled
    if pooled.t_stat < min_t:
        return False, f"pooled t={pooled.t_stat:.2f} < {min_t}", per, pooled
    return True, "PASS", per, pooled


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(rows, coverage, min_n=100, min_seasons=2, min_t=2.0,
                 primary_prop=PRIMARY_PROP, leg_counts=None):
    """Print the coverage, the PRIMARY verdict, and the full BH-corrected grid.
    ``leg_counts`` (optional {prop_key: n_paired_legs}) prints the per-prop funnel
    so a prop that Pinnacle doesn't book (0 paired legs) is visible, not silent.
    Returns a verdict dict (for tests / programmatic use)."""
    out = []
    p = out.append
    p("=" * 74)
    p("  R2 DK-vs-sharp backtest")
    p("=" * 74)

    p("\n  Coverage:")
    for k in ("legs", "legs_no_pinnacle_fair", "selected", "graded",
              "dropped_null_id", "dropped_null_game_pk", "dropped_no_actual",
              "dropped_ungradable"):
        if coverage.get(k):
            p(f"    {k:<24} {coverage[k]:>8,}")

    # Per-prop funnel: paired legs (Pinnacle-priceable) -> selected (cleared EV
    # floor) -> graded. A prop Pinnacle doesn't book shows 0 paired legs.
    sel_by_prop, grd_by_prop = Counter(), Counter()
    for r in rows:
        grd_by_prop[r["prop_key"]] += 1
    if leg_counts is not None:
        p("\n  Per-prop funnel  [paired legs -> graded bets]:")
        for pk in sorted(set(leg_counts) | set(grd_by_prop)):
            p(f"    {pk:<24} legs={leg_counts.get(pk, 0):>6,}  graded={grd_by_prop.get(pk, 0):>5,}")

    # PRIMARY: same-line, primary prop, pooled.
    primary = [r for r in rows if r["prop_key"] == primary_prop
               and r["arm"] == "same_line"]
    passed, reason, per, pooled = hardened_gate(primary, min_n, min_seasons, min_t)
    p(f"\n  PRIMARY hypothesis: {primary_prop} same-line (distance==0, pure devig)")
    if pooled.n:
        p(f"    pooled: n={pooled.n:,} decided={pooled.decided:,} "
          f"ROI={pooled.roi:+.2%} hit={pooled.hit_rate:.1%} t={pooled.t_stat:.2f}")
        for s in sorted(per):
            sm = per[s]
            tag = "" if sm.decided >= min_n else "  (insufficient)"
            p(f"      {s}: n={sm.decided:,} ROI={sm.roi:+.2%} t={sm.t_stat:.2f}{tag}")
    else:
        p("    (no primary legs cleared the EV floor)")
    p(f"    GATE: {'PASS' if passed else 'FAIL'} — {reason}")

    # FULL GRID: (prop x arm) x season, with BH across the family of pooled cells.
    p("\n  Grid — (prop x arm) pooled + per season  [n / ROI / t]:")
    cells = r2_grade.by_key(rows, lambda r: (r["prop_key"], r["arm"]))
    keys = sorted(cells)
    pooled_cells = [(k, cells[k]) for k in keys if cells[k].decided >= min_n]
    pvals = [_one_sided_p(sm.t_stat) for _k, sm in pooled_cells]
    bh_pass = benjamini_hochberg(pvals) if pvals else set()
    bh_keys = {pooled_cells[i][0] for i in bh_pass}
    for k in keys:
        sm = cells[k]
        thin = sm.decided < min_n
        flag = "insufficient" if thin else ("BH✓" if k in bh_keys else "")
        p(f"    {k[0]:<22} {k[1]:<10} n={sm.decided:>5,} "
          f"ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}  {flag}")
        by_s = r2_grade.by_key([r for r in rows
                                if (r["prop_key"], r["arm"]) == k],
                               lambda r: r["season"])
        for s in sorted(by_s):
            ssm = by_s[s]
            st = "" if ssm.decided >= min_n else " (insufficient)"
            p(f"        {s}: n={ssm.decided:>5,} ROI={ssm.roi:+.2%} t={ssm.t_stat:+.2f}{st}")

    # Projected arm by LINE-GAP magnitude — does a bigger DK-vs-Pinnacle line
    # disagreement carry more realized edge (the bullseye)?
    proj = [r for r in rows if r["arm"] == "projected"]
    if proj:
        p("\n  Projected arm by line-gap |DK point - nearest Pinnacle point|:")
        cells = r2_grade.by_key(proj, lambda r: dist_bucket(r.get("distance")))
        for b in ("<=0.5", "(0.5,1.0]", "(1.0,1.5]", ">1.5", "n/a"):
            if b in cells:
                sm = cells[b]
                st = "" if sm.decided >= min_n else " (insufficient)"
                p(f"    gap {b:<10} n={sm.decided:>5,} ROI={sm.roi:+.2%} "
                  f"t={sm.t_stat:+.2f}{st}")

    # Synonym (cross-market) pricing note: hits priced off Pinnacle TB.
    syn = sum(1 for r in rows if r.get("ref_prop"))
    if syn:
        p(f"\n  Cross-market synonym legs (e.g. hits priced off Pinnacle TB): {syn:,}")

    p("=" * 74)
    report_text = "\n".join(out)
    print(report_text)
    return {"primary_passed": passed, "primary_reason": reason,
            "primary_pooled": pooled, "primary_per_season": per,
            "grid_bh_pass": bh_keys, "coverage": dict(coverage),
            "text": report_text}


# ── Fetch + cache ─────────────────────────────────────────────────────────────

def _cache_path(sport, seasons, prop_keys):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = f"{sport}_{'-'.join(map(str, seasons))}_{'-'.join(sorted(prop_keys))}"
    return os.path.join(_CACHE_DIR, tag.replace("/", "_") + ".pkl")


def load_or_fetch(sport, seasons, prop_keys, refresh=False):
    """Fetch paired legs + the outcome index (or load the local cache). The fetch is
    the only Azure round-trip; re-report with different haircut/floor reads cache."""
    path = _cache_path(sport, seasons, prop_keys)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    legs_by_season, fetch_stats = r2_data.load_prop_legs(sport, seasons, prop_keys)
    outcome_idx = r2_data.build_outcome_index(seasons, prop_keys)
    blob = {"legs_by_season": legs_by_season, "outcome_idx": outcome_idx,
            "fetch_stats": {s: dict(c) for s, c in fetch_stats.items()}}
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    return blob, path


def main():
    ap = argparse.ArgumentParser(description="R2 DK-vs-sharp closing-line backtest.")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--props", default=",".join(DEFAULT_PROPS))
    ap.add_argument("--haircut", type=float, default=0.02,
                    help="Vig/CLV haircut on the DK price (decimal shade). Default 0.02.")
    ap.add_argument("--ev-floor", type=float, default=0.03,
                    help="Pre-registered minimum haircut-EV to bet. Default 0.03.")
    ap.add_argument("--dispersion", type=float, default=0.0,
                    help="Fixed a-priori NegBin dispersion for cross-line projection.")
    ap.add_argument("--min-n", type=int, default=100,
                    help="Min decided bets/season to judge that season (gate + grid).")
    ap.add_argument("--min-seasons", type=int, default=2)
    ap.add_argument("--min-t", type=float, default=2.0)
    ap.add_argument("--refresh", action="store_true", help="Re-fetch (ignore cache).")
    args = ap.parse_args()

    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    prop_keys = [p.strip() for p in args.props.split(",") if p.strip()]

    blob, path = load_or_fetch(args.sport, seasons, prop_keys, refresh=args.refresh)
    print(f"  data: {sum(len(v) for v in blob['legs_by_season'].values()):,} paired "
          f"legs across {len(seasons)} seasons  (cache: {path})")
    # Per-prop paired-leg counts (which props Pinnacle actually references) + the
    # select_prop_legs drop funnel.
    leg_counts = Counter()
    for legs in blob["legs_by_season"].values():
        for lg in legs:
            leg_counts[lg.prop_key] += 1
    drop = Counter()
    for st in blob.get("fetch_stats", {}).values():
        drop.update(st)
    if drop:
        print("  select drops: " + "  ".join(
            f"{k}={drop[k]:,}" for k in sorted(drop) if k.startswith(("events_dropped", "leg_dropped"))))

    try:
        import mlb_starters
        resolve_fn = mlb_starters.resolve_mlbam_id
    except Exception:
        resolve_fn = None

    rows, cov = grade_legs(blob["legs_by_season"], blob["outcome_idx"],
                           haircut=args.haircut, ev_floor=args.ev_floor,
                           default_dispersion=args.dispersion, resolve_fn=resolve_fn)
    build_report(rows, cov, min_n=args.min_n, min_seasons=args.min_seasons,
                 min_t=args.min_t, leg_counts=leg_counts)
    print(f"\n  params: haircut={args.haircut} ev_floor={args.ev_floor} "
          f"min_n={args.min_n} min_seasons={args.min_seasons} min_t={args.min_t}")


if __name__ == "__main__":
    main()
