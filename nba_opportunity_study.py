"""NBA opportunity->conversion HEAD-TO-HEAD (READ-ONLY, no writes, 0 DTU via the mirror).

Step 2 of the opportunity question. Step 1 (nba_minutes_study.py) showed MINUTES carry a
real recency signal (wmean_hl3 wins) but are only moderately predictable. Minutes MAE alone
can't tell us whether DECOMPOSING a prop into minutes x per-minute-rate actually beats
predicting the counting stat directly. This does the real test.

Per prop, leakage-safe walk-forward (strictly-prior played games, within season), it compares
three predictors of the ACTUAL stat:

  single@S     stat_hat = wmean_hl{S}(prior stat)                 -- current method / sweep winner
  single@L     stat_hat = wmean_hl{L}(prior stat)                 -- CONTROL: longer single-stage window
  two_stage    stat_hat = minutes_hat x pooled_rate
                 minutes_hat = wmean_hl{M}(prior minutes)         -- reactive (role changes)
                 pooled_rate = Sum(w_L . stat) / Sum(w_L . minutes)  -- stable skill, long window

The control matters: if single@L ~= two_stage, any gain was just "longer window," NOT the
decomposition. two_stage only earns its keep if it beats BOTH singles -- and it should win
most clearly in the ROLE-CHANGE segment (games where the player's minutes recently shifted),
which is reported separately.

Defaults: S=3 (the sweep winner), L=10, M=3 (the minutes-study winner).

Run (owner's machine, fresh stats.nba.com mirror):
  $env:PYTHONIOENCODING="utf-8"
  python nba_opportunity_study.py --seasons 2022,2023,2024,2025,2026
  # options: --stat-hl 3  --long-hl 10  --min-hl 3  --min-history 8  --shift-thresh 6
"""
import argparse
import math

import nba_warehouse
from nba_minutes_study import _parse_seasons, _role, ROLE_BANDS, _valid_min


# prop -> component canonical box columns (combos sum their parts)
PROPS = [
    ("points", ("points",)),
    ("rebounds", ("rebounds",)),
    ("assists", ("assists",)),
    ("threes", ("three_point_field_goals_made",)),
    ("steals", ("steals",)),
    ("blocks", ("blocks",)),
    ("turnovers", ("turnovers",)),
    ("pts+reb", ("points", "rebounds")),
    ("pts+ast", ("points", "assists")),
    ("pra", ("points", "rebounds", "assists")),
]
METHODS = ("single@S", "single@L", "two_stage")


def _weights(n, hl):
    """Recency weights for n ascending games; newest (index n-1) gets games_back=0."""
    return [0.5 ** (((n - 1) - j) / hl) for j in range(n)]


def _stat_val(g, cols):
    """Sum of component columns for one game; None if any component is missing/NaN."""
    tot = 0.0
    for c in cols:
        v = g.get(c)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if v != v:                                        # NaN
            return None
        tot += v
    return tot


def _predict(prior_mins, prior_stats, stat_hl, long_hl, min_hl):
    """prior_* ascending (oldest..newest). Returns {method: stat_hat}."""
    n = len(prior_mins)
    w_s = _weights(n, stat_hl)
    w_l = _weights(n, long_hl)
    w_m = _weights(n, min_hl)

    single_s = sum(w * x for w, x in zip(w_s, prior_stats)) / sum(w_s)
    single_l = sum(w * x for w, x in zip(w_l, prior_stats)) / sum(w_l)

    min_hat = sum(w * x for w, x in zip(w_m, prior_mins)) / sum(w_m)
    den = sum(w * x for w, x in zip(w_l, prior_mins))     # pooled rate = Sw.stat / Sw.min
    rate = (sum(w * x for w, x in zip(w_l, prior_stats)) / den) if den > 0 else 0.0
    two = min_hat * rate
    return {"single@S": single_s, "single@L": single_l, "two_stage": two}


class _Acc:
    __slots__ = ("abs_sum", "sq_sum", "n")

    def __init__(self):
        self.abs_sum = self.sq_sum = 0.0
        self.n = 0

    def add(self, pred, actual):
        if pred != pred:
            return
        e = pred - actual
        self.abs_sum += abs(e)
        self.sq_sum += e * e
        self.n += 1

    def mae(self):
        return self.abs_sum / self.n if self.n else float("nan")

    def rmse(self):
        return math.sqrt(self.sq_sum / self.n) if self.n else float("nan")


def _new_cell():
    return {m: _Acc() for m in METHODS}


def run(seasons, stat_hl, long_hl, min_hl, min_history, shift_thresh):
    corpus = nba_warehouse.calib_gamelogs_bulk(seasons)
    if not corpus:
        raise SystemExit(
            "Empty corpus. No mirror parquet for these seasons and DB not enabled.\n"
            "Run `python nba_warehouse.py --seasons <...> --export-mirror` first, or set "
            "secrets.toml.")

    pooled = {p: _new_cell() for p, _ in PROPS}
    by_role = {r: {p: _new_cell() for p, _ in PROPS} for r, _, _ in ROLE_BANDS}
    # role-change segments (points, rebounds, assists) — the decomposition's thesis case
    seg_props = ("points", "rebounds", "assists")
    changed = {p: _new_cell() for p in seg_props}
    steady = {p: _new_cell() for p in seg_props}
    n_scored = 0

    for name, games in corpus.items():
        per_season = {}
        for g in games:
            s = g.get("season")
            if s in seasons:
                per_season.setdefault(s, []).append(g)
        for s, gl in per_season.items():
            gl.sort(key=lambda x: x.get("game_date") or "")
            p_mins, p_games = [], []                       # prior played minutes + game dicts
            for g in gl:
                m = _valid_min(g)
                if m is None:
                    continue
                if len(p_mins) >= min_history:
                    role = _role(sum(p_mins) / len(p_mins))
                    # role-change flag: last-3 vs prior-3 minutes mean
                    seg = None
                    if len(p_mins) >= 6:
                        recent = sum(p_mins[-3:]) / 3.0
                        before = sum(p_mins[-6:-3]) / 3.0
                        seg = "changed" if abs(recent - before) >= shift_thresh else "steady"
                    for prop, cols in PROPS:
                        actual = _stat_val(g, cols)
                        if actual is None:
                            continue
                        pstats = [sv for sv in (_stat_val(pg, cols) for pg in p_games)
                                  if sv is not None]
                        if len(pstats) != len(p_mins):     # stat gap -> skip (keep arrays aligned)
                            continue
                        preds = _predict(p_mins, pstats, stat_hl, long_hl, min_hl)
                        for mth, val in preds.items():
                            pooled[prop][mth].add(val, actual)
                            by_role[role][prop][mth].add(val, actual)
                            if seg and prop in seg_props:
                                (changed if seg == "changed" else steady)[prop][mth].add(
                                    val, actual)
                    n_scored += 1
                p_mins.append(m)
                p_games.append(g)

    _report(seasons, stat_hl, long_hl, min_hl, min_history, shift_thresh,
            pooled, by_role, changed, steady, seg_props, n_scored)


def _cmp_table(title, cell_by_prop, props):
    print(f"\n{title}")
    hdr = (f"  {'prop':<9} {'single@S':>9} {'single@L':>9} {'two_stage':>10} "
           f"{'bestMAE':>9} {'2s vs best-single':>18} {'n':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for prop in props:
        cell = cell_by_prop[prop]
        maes = {m: cell[m].mae() for m in METHODS}
        if not cell["single@S"].n:
            continue
        best_single = min(maes["single@S"], maes["single@L"])
        two = maes["two_stage"]
        best = min(maes, key=lambda m: maes[m])
        if best_single > 1e-9:
            delta = 100.0 * (two - best_single) / best_single   # negative = two_stage better
            delta_str = f"{delta:>+16.2f}%"
        else:
            delta_str = f"{'n/a':>17}"
        flag = "  <= two_stage WINS" if two < best_single - 1e-9 else ""
        print(f"  {prop:<9} {maes['single@S']:>9.3f} {maes['single@L']:>9.3f} "
              f"{two:>10.3f} {best:>9} {delta_str}{flag}  {cell['single@S'].n:>6,}")


def _report(seasons, stat_hl, long_hl, min_hl, min_history, shift_thresh,
            pooled, by_role, changed, steady, seg_props, n_scored):
    print("=" * 86)
    print("NBA OPPORTUNITY->CONVERSION HEAD-TO-HEAD  (actual-stat MAE; walk-forward, "
          "within-season)")
    print("=" * 86)
    print(f"  seasons={seasons}  stat_hl(S)={stat_hl}  long_hl(L)={long_hl}  "
          f"min_hl(M)={min_hl}  min_history={min_history}  shift_thresh={shift_thresh}m")
    print(f"  player-games scored={n_scored:,}")
    print("  single@S = direct wmean_hl{S} (current) | single@L = direct wmean_hl{L} (control) "
          "| two_stage = min@M x rate@L")
    print("  '2s vs best-single' = two_stage MAE vs the BETTER of the two singles "
          "(negative % = decomposition wins).")

    all_props = [p for p, _ in PROPS]
    _cmp_table("POOLED (all roles):", pooled, all_props)

    for r, lo, hi in ROLE_BANDS:
        band = f"{lo:.0f}-{hi:.0f}m" if hi < 99 else f">={lo:.0f}m"
        _cmp_table(f"BY ROLE  [{r} ({band})]:", by_role[r], all_props)

    print("\n" + "=" * 86)
    print("ROLE-CHANGE SEGMENT  (thesis case: minutes recently shifted >= "
          f"{shift_thresh}m vs steady)")
    print("=" * 86)
    _cmp_table("  minutes CHANGED recently:", changed, list(seg_props))
    _cmp_table("  minutes STEADY:", steady, list(seg_props))
    print()


def main():
    ap = argparse.ArgumentParser(
        description="NBA opportunity->conversion head-to-head vs single-stage (read-only).")
    ap.add_argument("--seasons", default="2022,2023,2024,2025,2026")
    ap.add_argument("--stat-hl", type=int, default=3, help="single-stage half-life S (sweep winner).")
    ap.add_argument("--long-hl", type=int, default=10, help="long window L (control single + rate).")
    ap.add_argument("--min-hl", type=int, default=3, help="minutes half-life M (reactive).")
    ap.add_argument("--min-history", type=int, default=8,
                    help="min prior played games before scoring (>=6 needed for the shift flag).")
    ap.add_argument("--shift-thresh", type=float, default=6.0,
                    help="|last3 - prior3| minutes to tag a game as a role change.")
    args = ap.parse_args()
    run(_parse_seasons(args.seasons), args.stat_hl, args.long_hl, args.min_hl,
        args.min_history, args.shift_thresh)


if __name__ == "__main__":
    main()
