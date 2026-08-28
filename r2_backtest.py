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
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_cache")


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


# ── Side-split diagnostic (is DK directionally mispriced? the "fade" test) ────

def side_split_rows(legs_by_season, outcome_idx, default_dispersion=0.0,
                    haircut=0.0, resolve_fn=None):
    """UNCONDITIONAL realized ROI of flat-betting each side of every leg — NO EV
    floor, NO fair-selection (so no winner's curse), and the fair/projector is not
    used to decide anything (we only tag same_line vs projected). This isolates
    DK's own directional calibration: if DK OVERs systematically lose while UNDERs
    win (replicating across seasons), that's a real fade edge. Returns graded rows
    (both sides of each gradeable leg)."""
    rows = []
    for season, legs in legs_by_season.items():
        for lg in legs:
            edges = r2_edge.prop_leg_edges(
                lg.dk_point, lg.dk_over_price, lg.dk_under_price,
                lg.pinnacle_offers, default_dispersion=default_dispersion)
            if not edges:
                continue
            pid = lg.player_mlb_id
            if pid is None and resolve_fn is not None:
                try:
                    pid = resolve_fn(lg.player, int(str(lg.game_date)[:4]), lg.prop_key)
                except (TypeError, ValueError):
                    pid = None
            if pid is None or lg.game_pk is None:
                continue
            actual = r2_data.outcome_value(outcome_idx, lg.prop_key, pid, lg.game_pk)
            if actual is None:
                continue
            for e in edges:
                result = r2_grade.grade_over_under(actual, e.point, e.side)
                if result is None:
                    continue
                rows.append({
                    "season": str(season), "prop_key": lg.prop_key, "side": e.side,
                    "arm": "projected" if e.projected else "same_line",
                    "result": result,
                    "profit": profit_haircut(e.dk_price, result, haircut),
                })
    return rows


def build_side_split_report(rows, min_n=100):
    """Per (prop, side) realized ROI — same-line (clean) then all legs (power) —
    with a per-prop fade verdict. Raw DK prices (haircut 0): an efficient two-sided
    market shows BOTH sides ~-half-hold; a fade = one side systematically positive /
    much-less-negative than the other, replicating each season."""
    out = []
    p = out.append
    p("=" * 74)
    p("  R2 side-split — is DK directionally mispriced? (raw prices, no EV floor)")
    p("=" * 74)

    def _section(title, subset):
        p(f"\n  {title}:")
        props = sorted({r["prop_key"] for r in subset})
        for prop in props:
            pr = [r for r in subset if r["prop_key"] == prop]
            line = []
            per_side = {}
            for side in ("OVER", "UNDER"):
                sm = r2_grade.summarize([r for r in pr if r["side"] == side])
                per_side[side] = sm
                line.append(f"{side} n={sm.decided:,} ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}")
            p(f"    {prop:<22} " + "   ".join(line))
            # per-season, both sides, to check replication of any gap
            seasons = sorted({r["season"] for r in pr})
            for s in seasons:
                cells = []
                for side in ("OVER", "UNDER"):
                    sm = r2_grade.summarize(
                        [r for r in pr if r["side"] == side and r["season"] == s])
                    tag = "" if sm.decided >= min_n else "*"
                    cells.append(f"{side} {sm.roi:+.1%}(n{sm.decided}{tag})")
                p(f"        {s}: " + "  ".join(cells))
            # fade verdict: one side positive every judged season, other negative
            verdict = _fade_verdict(pr, min_n)
            if verdict:
                p(f"      -> {verdict}")

    _section("SAME-LINE only (model-free, cleanest)",
             [r for r in rows if r["arm"] == "same_line"])
    _section("ALL legs (same-line + projected; higher power, DK-calibration test)",
             rows)
    p("\n  (* = season n below min_n; ROI is realized at DK's raw price, hold not removed)")
    p("=" * 74)
    text = "\n".join(out)
    print(text)
    return {"text": text}


def _fade_verdict(prop_rows, min_n):
    """A per-prop fade call: does ONE side clear ROI>0 in EVERY judged season while
    the other is negative? Returns a short string or None."""
    seasons = sorted({r["season"] for r in prop_rows})
    judged = {}
    for side in ("OVER", "UNDER"):
        per = {}
        for s in seasons:
            sm = r2_grade.summarize([r for r in prop_rows
                                     if r["side"] == side and r["season"] == s])
            if sm.decided >= min_n:
                per[s] = sm.roi
        judged[side] = per
    for win, lose in (("UNDER", "OVER"), ("OVER", "UNDER")):
        w, l = judged[win], judged[lose]
        if len(w) >= 2 and all(v > 0 for v in w.values()) and \
           l and all(v < 0 for v in l.values()):
            return (f"FADE signal: {win} positive every judged season, "
                    f"{lose} negative — possible DK {lose.lower()} bias")
    return None


# ── Sharpness test: is Pinnacle actually sharper than DK? (R2's premise) ──────

def sharpness_rows(legs_by_season, outcome_idx, default_dispersion=0.0, resolve_fn=None):
    """Model-FREE head-to-head: on SAME-LINE legs (both books post the same point,
    so their devigged probs price the identical over/under event), record DK's fair
    P(over), Pinnacle's fair P(over), and the realized over (0/1). Lets us score each
    book's closing prices against outcomes (Brier/log-loss) — the direct test of
    whether Pinnacle is sharper, with NO model and NO EV selection."""
    from r2_sharp import fair_two_way, fair_prob_at_line
    rows = []
    for season, legs in legs_by_season.items():
        for lg in legs:
            sf = fair_prob_at_line(lg.pinnacle_offers, lg.dk_point, default_dispersion)
            if sf.prob is None or sf.projected:      # same-line only (apples-to-apples)
                continue
            dk_fair, _ = fair_two_way(lg.dk_over_price, lg.dk_under_price)
            if dk_fair is None:
                continue
            pid = lg.player_mlb_id
            if pid is None and resolve_fn is not None:
                try:
                    pid = resolve_fn(lg.player, int(str(lg.game_date)[:4]), lg.prop_key)
                except (TypeError, ValueError):
                    pid = None
            if pid is None or lg.game_pk is None:
                continue
            actual = r2_data.outcome_value(outcome_idx, lg.prop_key, pid, lg.game_pk)
            if actual is None or actual == lg.dk_point:   # None=DNP; ==point=push
                continue
            rows.append({"season": str(season), "prop_key": lg.prop_key,
                         "dk_fair": dk_fair, "pin_fair": sf.prob,
                         "over": 1.0 if actual > lg.dk_point else 0.0})
    return rows


def _brier_logloss(rows, prob_key):
    """(mean Brier, mean log-loss) of a book's fair probs vs the realized over."""
    n = len(rows)
    if not n:
        return None, None
    brier = sum((r[prob_key] - r["over"]) ** 2 for r in rows) / n
    ll = 0.0
    for r in rows:
        p = min(max(r[prob_key], 1e-12), 1.0 - 1e-12)
        ll += -(r["over"] * math.log(p) + (1 - r["over"]) * math.log(1 - p))
    return brier, ll / n


def _paired_brier_t(rows):
    """Paired t on per-leg (DK_brier - Pin_brier): mean>0 & t>2 => Pinnacle sharper
    (lower Brier). Returns (mean_diff, t)."""
    d = [(r["dk_fair"] - r["over"]) ** 2 - (r["pin_fair"] - r["over"]) ** 2 for r in rows]
    n = len(d)
    if n < 2:
        return (0.0, 0.0)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return mean, (mean / se if se > 0 else 0.0)


def build_sharpness_report(rows, min_n=100):
    """Per-prop + pooled Brier/log-loss for DK vs Pinnacle, with a paired t on the
    Brier difference. The definitive 'is Pinnacle sharper?' read."""
    out = []
    p = out.append
    p("=" * 74)
    p("  R2 sharpness — DK vs Pinnacle closing-line accuracy (same-line, model-free)")
    p("=" * 74)

    def _line(label, subset):
        db, dll = _brier_logloss(subset, "dk_fair")
        pb, pll = _brier_logloss(subset, "pin_fair")
        if db is None:
            p(f"  {label:<24} (no same-line legs)")
            return
        md, t = _paired_brier_t(subset)
        if md > 0 and t >= 2:
            verdict = f"PINNACLE sharper (t={t:+.2f})"
        elif md < 0 and t <= -2:
            verdict = f"DK sharper (t={t:+.2f})"
        else:
            verdict = f"tie (t={t:+.2f})"
        tag = "" if len(subset) >= min_n else "  *thin"
        p(f"  {label:<24} n={len(subset):>5,}  Brier DK={db:.4f} Pin={pb:.4f}  "
          f"logloss DK={dll:.4f} Pin={pll:.4f}  -> {verdict}{tag}")

    _line("ALL same-line", rows)
    p("")
    for prop in sorted({r["prop_key"] for r in rows}):
        pr = [r for r in rows if r["prop_key"] == prop]
        _line(prop, pr)
        for s in sorted({r["season"] for r in pr}):
            _line(f"  {prop[:16]} {s}", [r for r in pr if r["season"] == s])
    p("\n  Lower Brier/log-loss = sharper. Paired t>+2 => Pinnacle significantly "
      "sharper; this VALIDATES (or refutes) R2's use of Pinnacle as the yardstick.")
    p("=" * 74)
    text = "\n".join(out)
    print(text)
    return {"text": text}


def team_sharpness_rows(legs_by_season, finals_idx, label="moneyline_team"):
    """Rows (in build_sharpness_report's shape) for the MONEYLINE sharpness gate:
    DK's vs Pinnacle's devigged fair P(home win) vs the realized home win, from
    paired same-snapshot moneyline legs. `over` = home won. Works for full-game
    (kind=team) or F5 (kind=first_five) legs identically."""
    from r2_sharp import fair_two_way
    rows = []
    for season, legs in legs_by_season.items():
        for lg in legs:
            if lg.game_pk is None:
                continue
            hw = finals_idx.get(int(lg.game_pk))
            if hw is None:
                continue
            dk_h, _ = fair_two_way(lg.dk_home, lg.dk_away)
            pin_h, _ = fair_two_way(lg.pin_home, lg.pin_away)
            if dk_h is None or pin_h is None:
                continue
            rows.append({"season": str(season), "prop_key": label,
                         "dk_fair": dk_h, "pin_fair": pin_h, "over": hw})
    return rows


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


def load_or_fetch(sport, seasons, prop_keys, refresh=False, refresh_mirror=False):
    """Fetch paired legs + the outcome index (or load the local cache). The fetch is
    the only Azure round-trip; re-report with different haircut/floor reads cache."""
    path = _cache_path(sport, seasons, prop_keys)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    import warehouse_mirror
    warehouse_mirror.autobuild(sport, seasons, refresh=refresh_mirror)
    print(f"  cold cache build — reading from {warehouse_mirror.source_label()}")
    legs_by_season, fetch_stats = r2_data.load_prop_legs(sport, seasons, prop_keys)
    outcome_idx = r2_data.build_outcome_index(seasons, prop_keys)
    blob = {"legs_by_season": legs_by_season, "outcome_idx": outcome_idx,
            "fetch_stats": {s: dict(c) for s, c in fetch_stats.items()}}
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    return blob, path


def _ml_cache_path(sport, seasons, kind):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = f"ml_{kind}_{sport}_{'-'.join(map(str, seasons))}"
    return os.path.join(_CACHE_DIR, tag.replace("/", "_") + ".pkl")


def load_or_fetch_ml(sport, seasons, kind, refresh=False, refresh_mirror=False):
    """Fetch + pair DK/Pinnacle moneyline legs (kind='team' or 'first_five') + the
    finals index (or load cache). Feeds the sharpness GATE."""
    path = _ml_cache_path(sport, seasons, kind)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    import warehouse_mirror
    warehouse_mirror.autobuild(sport, seasons, refresh=refresh_mirror)
    print(f"  cold cache build — reading from {warehouse_mirror.source_label()}")
    legs_by_season, stats = r2_data.load_team_ml_legs(sport, seasons, kind=kind)
    finals_idx = r2_data.build_team_finals_index(seasons)
    blob = {"legs_by_season": legs_by_season, "finals_idx": finals_idx,
            "stats": {s: dict(c) for s, c in stats.items()}}
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
    ap.add_argument("--refresh-mirror", action="store_true",
                    help="re-sync + re-verify the parquet mirror (on by default; ODI_BACKTEST_MIRROR=0 disables)")
    ap.add_argument("--sharpness", action="store_true",
                    help="Diagnostic: is Pinnacle sharper than DK? (model-free "
                         "closing-line Brier/log-loss on same-line legs). Cache-only.")
    ap.add_argument("--side-split", action="store_true",
                    help="Diagnostic: realized ROI by side (fade test) — is DK "
                         "directionally mispriced? Unconditional, cache-only.")
    ap.add_argument("--ml-sharpness", action="store_true",
                    help="GATE: is Pinnacle sharper than DK on MONEYLINE? (Brier/"
                         "log-loss, model-free). --ml-kind selects team vs first_five.")
    ap.add_argument("--ml-kind", default="team", choices=("team", "first_five"),
                    help="Moneyline snapshot kind for --ml-sharpness (default team).")
    args = ap.parse_args()

    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    prop_keys = [p.strip() for p in args.props.split(",") if p.strip()]

    # GATE: moneyline sharpness (team or F5) — its own data/cache path.
    if args.ml_sharpness:
        mlblob, mlpath = load_or_fetch_ml(args.sport, seasons, args.ml_kind,
                                          refresh=args.refresh,
                                          refresh_mirror=args.refresh_mirror)
        n = sum(len(v) for v in mlblob["legs_by_season"].values())
        print(f"  data: {n:,} paired {args.ml_kind} moneyline legs "
              f"(DK+Pinnacle)  (cache: {mlpath})")
        build_sharpness_report(
            team_sharpness_rows(mlblob["legs_by_season"], mlblob["finals_idx"],
                                label=f"moneyline_{args.ml_kind}"),
            min_n=args.min_n)
        return

    blob, path = load_or_fetch(args.sport, seasons, prop_keys, refresh=args.refresh,
                               refresh_mirror=args.refresh_mirror)
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

    # Diagnostics (cache-only; answer premise/direction questions, not the verdict).
    if args.sharpness:
        build_sharpness_report(
            sharpness_rows(blob["legs_by_season"], blob["outcome_idx"],
                           default_dispersion=args.dispersion, resolve_fn=resolve_fn),
            min_n=args.min_n)
    if args.side_split:
        build_side_split_report(
            side_split_rows(blob["legs_by_season"], blob["outcome_idx"],
                            default_dispersion=args.dispersion, haircut=0.0,
                            resolve_fn=resolve_fn),
            min_n=args.min_n)
    if args.sharpness or args.side_split:
        return

    rows, cov = grade_legs(blob["legs_by_season"], blob["outcome_idx"],
                           haircut=args.haircut, ev_floor=args.ev_floor,
                           default_dispersion=args.dispersion, resolve_fn=resolve_fn)
    build_report(rows, cov, min_n=args.min_n, min_seasons=args.min_seasons,
                 min_t=args.min_t, leg_counts=leg_counts)
    print(f"\n  params: haircut={args.haircut} ev_floor={args.ev_floor} "
          f"min_n={args.min_n} min_seasons={args.min_seasons} min_t={args.min_t}")


if __name__ == "__main__":
    main()
