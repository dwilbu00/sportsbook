"""NBA prop pace + defense study (READ-ONLY; 0 DTU via the mirrors; no Odds API credits).

Step 5a of the pace plan: BEFORE wiring pace/defense into the sweep, the real-line selector and
live pricing, measure whether they make NBA prop projections more accurate — and in the right
direction. Walk-forward: each game is projected only from the player's earlier games that
season, and every context number (pace, what a defense allows) is the as-of PRE-GAME value from
nba_ratings (shipped team-model knobs).

Projection (mirrors production): recency-weighted mean (the prop's shipped half-life) with the
shipped Bayesian shrinkage toward the plain mean. Context adjustment = normalize-then-reapply on
each past value before averaging:

    adjusted_i = value_i x (poss_hat_t / poss_i)^s_pace x (def_t / def_i)^s_def

  poss_i     ACTUAL possessions of past game i (team box; removes realized pace/OT noise)
  poss_hat_t PROJECTED possessions tonight = pace_team x pace_opp / lg_pace (pre-game ratings)
  def_x      the opponent's ALLOWED per 100 possessions for the prop's stat(s) / league, pre-game

Reports per prop: projection MAE / RMSE vs the actual stat for base and each strength combo, and
the SIGN CHECK — slope of the base residual (actual - base) on the adjustment (adjusted - base).
Slope ~1: the adjustment moves the projection the right way by the right amount; ~0: noise;
< 0: wrong sign (the NFL defense lesson).

Run (owner's machine, after the nba_team_game backfill + --export-mirror):
  $env:PYTHONIOENCODING="utf-8"
  python nba_prop_context_study.py --seasons 2023,2024,2025,2026
"""
import argparse
import json
import math
import os

import nba_ratings as R
import nba_warehouse as nw

SPORT_KEY = "basketball_nba"
STRENGTHS = ((0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.0, 1.0), (0.0, 0.5), (1.0, 1.0),
             (1.0, 0.5), (0.5, 0.5))
MIN_PRIORS = 8
MIN_TEST_MINUTES = 10.0


def _parse_seasons(spec):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return sorted(set(out))


def _shipped_props():
    """{prop_key: (cols tuple, half_life, shrinkage_k)} from the live calibration file."""
    import nba_stats_source as src
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration",
                        f"{SPORT_KEY}.json")
    with open(path, encoding="utf-8") as f:
        props = json.load(f).get("props", {})
    out = {}
    for pk, cfg in props.items():
        cols = src.PROP_STAT_COLS.get(pk)
        if cols is None:
            continue
        cols = cols if isinstance(cols, tuple) else (cols,)
        out[pk] = (cols, float(cfg.get("half_life") or 0) or None,
                   float(cfg.get("shrinkage_k") or 0))
    return out


def _ratings_params():
    try:
        import nba_team_model
        m = nba_team_model.model()
        if m:
            return m["params"]
    except Exception:
        pass
    return dict(R.DEFAULT_PARAMS)


def _context_index(seasons, params):
    """{(game_id, team_id): {poss, poss_hat, def_rel{stat}}} — def_rel is the OPPONENT's
    pre-game allowed/100 relative to the league, per canonical stat."""
    rows = nw.team_games([min(seasons) - 2] + list(range(min(seasons) - 1, max(seasons) + 1)))
    pre = R.pregame_ratings(rows, params)
    out = {}
    for _d, _s, gid, a, b in R._pair_games(rows):
        for r, o in ((a, b), (b, a)):
            mine, opp = pre.get((gid, r["team_id"])), pre.get((gid, o["team_id"]))
            if not mine or not opp:
                continue
            rel = {}
            for st in R.ALLOWED_STATS:
                al, lg = opp["allowed100"].get(st), opp["lg_allowed100"].get(st)
                rel[st] = (al, lg)
            out[(gid, r["team_id"])] = {
                "poss": float(r["possessions"]),
                "poss_hat": mine["pace"] * opp["pace"] / mine["lg_pace"],
                "allowed": rel}
    return out


def _def_rel(ctx, cols):
    """Opponent's allowed/100 for the prop's columns (summed for combos) / league."""
    al = sum((ctx["allowed"].get(c) or (None, None))[0] or 0.0 for c in cols)
    lg = sum((ctx["allowed"].get(c) or (None, None))[1] or 0.0 for c in cols)
    return (al / lg) if al > 0 and lg > 0 else None


def _coeffs(n, half_life, k):
    """Linear projection coefficients for n newest-first values: recency weights + Bayesian
    shrinkage toward the plain mean, i.e. projection = sum(c_i * value_i) (production math)."""
    import numpy as np
    w = (np.exp(-math.log(2) / half_life * np.arange(n)) if half_life else np.ones(n))
    tw = w.sum()
    return (w + (k / n)) / (tw + k) if k > 0 else w / tw


def run(seasons):
    import numpy as np
    props = _shipped_props()
    params = _ratings_params()
    ctx = _context_index(seasons, params)
    bulk = nw.calib_gamelogs_bulk(seasons)
    if not bulk or not ctx:
        raise SystemExit("No data — need the nba_player_game + nba_team_game mirrors/backfill.")
    pks = list(props)
    combos = list(STRENGTHS)
    acc = {pk: {s: [0.0, 0.0, 0] for s in combos} for pk in pks}
    sign = {pk: {"pace": [], "def": []} for pk in pks}
    big = {pk: [] for pk in pks}                       # (|p1d1 - base|, |e_base|, |e_p1d1|)
    coef_cache = {}
    for _name, games in bulk.items():
        by_season = {}
        for g in games:
            if g.get("season") in seasons:
                by_season.setdefault(g["season"], []).append(g)
        for _s, gl in by_season.items():
            gl.sort(key=lambda g: g.get("game_date") or "")
            poss_hist = []                                  # asc
            val_hist = {pk: [] for pk in pks}
            def_hist = {pk: [] for pk in pks}               # opponent def_rel (None -> nan)
            for g in gl:
                played = g.get("played") and (g.get("MIN") or 0) > 0
                c = ctx.get((str(g.get("game_pk")), g.get("team_id")))
                if (played and (g.get("MIN") or 0) >= MIN_TEST_MINUTES and c
                        and len(poss_hist) >= MIN_PRIORS):
                    n = len(poss_hist)
                    pf = c["poss_hat"] / np.array(poss_hist[::-1])
                    for pk in pks:
                        cols, hl, k = props[pk]
                        key = (n, hl, k)
                        co = coef_cache.get(key)
                        if co is None:
                            co = coef_cache[key] = _coeffs(n, hl, k)
                        v = np.array(val_hist[pk][::-1])
                        d_t = _def_rel(c, cols)
                        di = np.array(def_hist[pk][::-1], dtype=float)
                        df = np.where(np.isnan(di) | (d_t is None), 1.0,
                                      (d_t or 1.0) / np.where(np.isnan(di), 1.0, di))
                        actual = sum(float(g.get(cc) or 0.0) for cc in cols)
                        base = float(co @ v)
                        for sp, sd in combos:
                            pr = float(co @ (v * pf ** sp * df ** sd)) if (sp or sd) else base
                            e = actual - pr
                            a = acc[pk][(sp, sd)]
                            a[0] += abs(e)
                            a[1] += e * e
                            a[2] += 1
                            if (sp, sd) == (1.0, 0.0):
                                sign[pk]["pace"].append((pr - base, actual - base))
                            elif (sp, sd) == (0.0, 1.0):
                                sign[pk]["def"].append((pr - base, actual - base))
                            if (sp, sd) == (1.0, 1.0):
                                big[pk].append((abs(pr - base), abs(actual - base),
                                                abs(actual - pr)))
                if played and c:
                    poss_hist.append(c["poss"])
                    for pk in pks:
                        cols = props[pk][0]
                        val_hist[pk].append(sum(float(g.get(cc) or 0.0) for cc in cols))
                        d = _def_rel(c, cols)
                        def_hist[pk].append(float("nan") if d is None else d)
    _report(seasons, params, props, acc, sign)
    _report_big(big)


def _report_big(big):
    """Where the adjustment is LARGE (top decile of |p1/d1 - base|) — the extreme-context
    games (fast vs slow, elite D) where a context-blind projection is most off."""
    print("  LARGE-ADJUSTMENT games (top 10% by |pace+defense adjustment|, full strength):")
    print(f"  {'prop':<32} {'n':>6} {'avg |adj|':>9} {'baseMAE':>8} {'adjMAE':>8} {'change':>8}")
    for pk, rows in big.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r[0], reverse=True)
        top = rows[:max(1, len(rows) // 10)]
        b = sum(r[1] for r in top) / len(top)
        a = sum(r[2] for r in top) / len(top)
        print(f"  {pk:<32} {len(top):>6,} {sum(r[0] for r in top) / len(top):>9.3f} "
              f"{b:>8.3f} {a:>8.3f} {100.0 * (a - b) / b:>+7.2f}%")
    print()


def _slope(pairs):
    sxx = sum(x * x for x, _ in pairs)
    return (sum(x * y for x, y in pairs) / sxx) if sxx > 0 else float("nan")


def _report(seasons, params, props, acc, sign):
    print("=" * 100)
    print("NBA PROP PACE + DEFENSE STUDY  (walk-forward, within-season; pre-game context)")
    print("=" * 100)
    print(f"  seasons {seasons}  ratings params {params}")
    print("  cells = % change in projection MAE vs base (negative = better). Sign slope: "
          "~1 right size, ~0 noise, <0 wrong sign.\n")
    combos = [s for s in STRENGTHS if s != (0.0, 0.0)]
    hdr = f"  {'prop':<32} {'n':>7} {'baseMAE':>8} " + " ".join(
        f"{f'p{sp:g}/d{sd:g}':>9}" for sp, sd in combos) + f" {'slopeP':>7} {'slopeD':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pk in props:
        b = acc[pk][(0.0, 0.0)]
        if not b[2]:
            continue
        base_mae = b[0] / b[2]
        cells = []
        for s in combos:
            a = acc[pk][s]
            cells.append(f"{100.0 * (a[0] / a[2] - base_mae) / base_mae:>+8.2f}%")
        print(f"  {pk:<32} {b[2]:>7,} {base_mae:>8.3f} " + " ".join(cells)
              + f" {_slope(sign[pk]['pace']):>7.2f} {_slope(sign[pk]['def']):>7.2f}")
    print("\n  RMSE % change (same layout):")
    for pk in props:
        b = acc[pk][(0.0, 0.0)]
        if not b[2]:
            continue
        base = math.sqrt(b[1] / b[2])
        cells = []
        for s in combos:
            a = acc[pk][s]
            cells.append(f"{100.0 * (math.sqrt(a[1] / a[2]) - base) / base:>+8.2f}%")
        print(f"  {pk:<32} {b[2]:>7,} {base:>8.3f} " + " ".join(cells))
    print()


def main():
    ap = argparse.ArgumentParser(description="NBA prop pace + defense study (read-only).")
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    args = ap.parse_args()
    import db_store
    db_store.promote_secrets_from_toml()
    run(_parse_seasons(args.seasons))


if __name__ == "__main__":
    main()
