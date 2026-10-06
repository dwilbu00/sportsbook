"""NBA minutes-prediction accuracy probe (READ-ONLY, no writes, 0 DTU via the mirror).

Step 1 of the opportunity->conversion question for NBA: BEFORE building a two-stage
minutes x per-minute-rate model, measure how well we can predict a player's MINUTES
for the next game from his recent history. If minutes aren't predictable, the whole
decomposition's upside is capped; if they are, it justifies the build.

Framing (important):
  * We predict minutes CONDITIONAL ON PLAYING. A player's availability (healthy scratch /
    injury / rest) can't be read off a box score, so DNPs are excluded from BOTH the history
    window and the scored target. This isolates the "how many minutes given he's in the
    rotation" signal, which is the opportunity leg the conversion model would multiply by.
    The DNP rate is reported separately so you can see how much availability uncertainty
    sits OUTSIDE this model.
  * Leakage-safe walk-forward: each game is predicted from that player's STRICTLY-PRIOR
    played games WITHIN THE SAME SEASON (no cross-season bleed, no future games), matching
    how the live model would see only season-to-date history.

Predictors compared (all over prior played games, within season):
  last1        - most recent played game's minutes
  mean_std     - season-to-date mean (unweighted)
  mean_last5   - trailing 5 played games
  mean_last10  - trailing 10 played games
  wmean_hlN    - recency-weighted mean, half-life N games (w = 0.5 ** (games_back / N))

Run (owner's machine, where the fresh stats.nba.com mirror lives):
  $env:PYTHONIOENCODING="utf-8"
  python nba_minutes_study.py --seasons 2022,2023,2024,2025,2026
  # options: --min-history 5   --half-lives 3,5,10   --role-split (default on)
"""
import argparse
import math

import nba_warehouse


# ── role buckets by a player's full-season mean PLAYED minutes (segmentation only,
#    never a prediction input) — starters are easy, deep bench is noise-dominated ──
ROLE_BANDS = [("Starter", 30.0, 99.0), ("Rotation", 20.0, 30.0),
              ("Bench", 10.0, 20.0), ("Deep", 0.0, 10.0)]


def _role(mean_min):
    for name, lo, hi in ROLE_BANDS:
        if lo <= mean_min < hi:
            return name
    return "Deep"


def _parse_seasons(spec):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def _valid_min(g):
    """Played, with a real positive minutes value."""
    if not g.get("played"):
        return None
    m = g.get("MIN")
    try:
        m = float(m)
    except (TypeError, ValueError):
        return None
    if m != m or m <= 0:                                   # NaN or 0 -> treat as not-played
        return None
    return m


def _wmean(vals, hl):
    """Recency-weighted mean. vals ascending (oldest..newest); newest gets games_back=0."""
    n = len(vals)
    num = den = 0.0
    for j, v in enumerate(vals):
        back = (n - 1) - j
        w = 0.5 ** (back / hl)
        num += w * v
        den += w
    return num / den if den else float("nan")


def _predictors(prior, half_lives):
    """prior = ascending list of prior played minutes. Returns {name: prediction}."""
    preds = {
        "last1": prior[-1],
        "mean_std": sum(prior) / len(prior),
        "mean_last5": sum(prior[-5:]) / len(prior[-5:]),
        "mean_last10": sum(prior[-10:]) / len(prior[-10:]),
    }
    for hl in half_lives:
        preds[f"wmean_hl{hl}"] = _wmean(prior, hl)
    return preds


class _Acc:
    """Streaming error accumulator for one predictor (optionally within one bucket)."""
    __slots__ = ("abs_sum", "sq_sum", "signed_sum", "n", "le4", "le6", "abs_list")

    def __init__(self):
        self.abs_sum = self.sq_sum = self.signed_sum = 0.0
        self.n = 0
        self.le4 = self.le6 = 0
        self.abs_list = []

    def add(self, pred, actual):
        if pred != pred:                                  # NaN guard
            return
        err = pred - actual
        ae = abs(err)
        self.abs_sum += ae
        self.sq_sum += err * err
        self.signed_sum += err
        self.n += 1
        if ae <= 4.0:
            self.le4 += 1
        if ae <= 6.0:
            self.le6 += 1
        self.abs_list.append(ae)

    def row(self):
        if not self.n:
            return None
        self.abs_list.sort()
        med = self.abs_list[self.n // 2]
        return {"mae": self.abs_sum / self.n, "rmse": math.sqrt(self.sq_sum / self.n),
                "medae": med, "bias": self.signed_sum / self.n,
                "le4": 100.0 * self.le4 / self.n, "le6": 100.0 * self.le6 / self.n,
                "n": self.n}


def run(seasons, min_history, half_lives):
    corpus = nba_warehouse.calib_gamelogs_bulk(seasons)
    if not corpus:
        raise SystemExit(
            "Empty corpus. No mirror parquet found for these seasons and DB not enabled.\n"
            "Run `python nba_warehouse.py --seasons <...> --export-mirror` first, or set "
            "secrets.toml so calib_gamelogs_bulk can read Azure.")

    pred_names = ["last1", "mean_std", "mean_last5", "mean_last10"] + \
                 [f"wmean_hl{hl}" for hl in half_lives]
    pooled = {p: _Acc() for p in pred_names}
    by_role = {r: {p: _Acc() for p in pred_names} for r, _, _ in ROLE_BANDS}
    by_season = {s: {p: _Acc() for p in pred_names} for s in seasons}

    n_players = n_scored = n_games_total = n_dnp = 0
    for name, games in corpus.items():
        # group this player's games by season, ascending by date
        per_season = {}
        for g in games:
            s = g.get("season")
            if s in by_season:
                per_season.setdefault(s, []).append(g)
        touched = False
        for s, gl in per_season.items():
            gl.sort(key=lambda x: x.get("game_date") or "")
            played = []                                    # ascending played minutes so far
            for g in gl:
                n_games_total += 1
                m = _valid_min(g)
                if m is None:
                    n_dnp += 1
                    continue
                if len(played) >= min_history:             # enough history -> score this game
                    preds = _predictors(played, half_lives)
                    role = _role(sum(played) / len(played))  # season-to-date role (leak-safe)
                    for p, val in preds.items():
                        pooled[p].add(val, m)
                        by_role[role][p].add(val, m)
                        by_season[s][p].add(val, m)
                    n_scored += 1
                    touched = True
                played.append(m)
        if touched:
            n_players += 1

    _report(seasons, min_history, half_lives, pred_names, pooled, by_role, by_season,
            n_players, n_scored, n_games_total, n_dnp)


def _fmt_table(title, pred_names, accs, extra_label=None, extra_vals=None):
    print(f"\n{title}")
    hdr = f"  {'predictor':<14} {'MAE':>6} {'RMSE':>6} {'medAE':>6} " \
          f"{'bias':>7} {'<=4m%':>7} {'<=6m%':>7} {'n':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in pred_names:
        r = accs[p].row()
        if not r:
            continue
        print(f"  {p:<14} {r['mae']:>6.2f} {r['rmse']:>6.2f} {r['medae']:>6.1f} "
              f"{r['bias']:>+7.2f} {r['le4']:>6.1f}% {r['le6']:>6.1f}% {r['n']:>8,}")


def _report(seasons, min_history, half_lives, pred_names, pooled, by_role, by_season,
            n_players, n_scored, n_games_total, n_dnp):
    print("=" * 74)
    print("NBA MINUTES-PREDICTION ACCURACY  (conditional on playing; walk-forward, "
          "within-season)")
    print("=" * 74)
    print(f"  seasons={seasons}  min_history={min_history}  half_lives={half_lives}")
    dnp_pct = 100.0 * n_dnp / n_games_total if n_games_total else 0.0
    print(f"  players scored={n_players:,}  games scored={n_scored:,}  "
          f"(skipped first {min_history} played/player-season + DNPs)")
    print(f"  DNP rate (availability uncertainty, OUTSIDE this model) = "
          f"{n_dnp:,}/{n_games_total:,} = {dnp_pct:.1f}%")
    print("  MAE/RMSE/medAE/bias in MINUTES; <=4m%/<=6m% = share of games within that "
          "abs error.")

    _fmt_table("POOLED (all roles, all seasons):", pred_names, pooled)

    # pick the headline predictor = lowest pooled MAE, for the per-role / per-season cuts
    best = min((p for p in pred_names if pooled[p].row()),
               key=lambda p: pooled[p].row()["mae"])
    print(f"\n  -> lowest-MAE predictor: {best}")

    print("\nBY ROLE  (season-to-date mean minutes defines the band):")
    for r, lo, hi in ROLE_BANDS:
        band = f"{lo:.0f}-{hi:.0f}m" if hi < 99 else f">={lo:.0f}m"
        _fmt_table(f"  [{r} ({band})]", pred_names, by_role[r])

    print(f"\nBY SEASON  (headline predictor = {best}):")
    hdr = f"  {'season':<8} {'MAE':>6} {'RMSE':>6} {'medAE':>6} {'bias':>7} " \
          f"{'<=4m%':>7} {'<=6m%':>7} {'n':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in seasons:
        r = by_season[s][best].row()
        if not r:
            continue
        print(f"  {s:<8} {r['mae']:>6.2f} {r['rmse']:>6.2f} {r['medae']:>6.1f} "
              f"{r['bias']:>+7.2f} {r['le4']:>6.1f}% {r['le6']:>6.1f}% {r['n']:>8,}")
    print()


def main():
    ap = argparse.ArgumentParser(description="NBA minutes-prediction accuracy probe (read-only).")
    ap.add_argument("--seasons", default="2022,2023,2024,2025,2026",
                    help="season-ENDING years; range (2022-2026) or comma list.")
    ap.add_argument("--min-history", type=int, default=5,
                    help="Min prior played games (within season) before a game is scored.")
    ap.add_argument("--half-lives", default="3,5,10",
                    help="Comma list of recency half-lives (games) to try.")
    args = ap.parse_args()
    seasons = _parse_seasons(args.seasons)
    half_lives = [int(x) for x in args.half_lives.split(",") if x.strip()]
    run(seasons, args.min_history, half_lives)


if __name__ == "__main__":
    main()
