"""F5 (first-5-innings) moneyline edge: is DK (and FanDuel) soft/stale on F5 vs
sharp Pinnacle? The best-prior remaining hunt — full-game markets tied (DK==Pinnacle
sharpness), but on F5 DK books moneyline-ONLY (thin attention) while Pinnacle books
it seriously (its 0.0 "spread" == its F5 ML). Grades on actual first-5 scores
(mlb_game.home_score_f5/away_score_f5).

Two stages:
  1. SHARPNESS GATE (the precondition): Brier/log-loss of DK's F5 ML fair vs
     Pinnacle's (its deviged 0.0-spread) against the realized F5 result. If Pinnacle
     is significantly sharper, its fair is a valid yardstick and the edge test is
     meaningful; if it ties, F5 is efficient too (like full-game) and the edge test
     will null.
  2. EDGE TEST: bet DK's and FanDuel's F5 ML where it's +EV vs the Pinnacle-derived
     fair; grade win/loss/push (F5 tie = push) at a vig haircut; hardened per-season
     OOS gate + by-EV-bucket + by-book.

Pinnacle's F5 ML is a DIRECT deviged price (no run-distribution translation), so
there's no model-shape bias to calibrate (unlike the coherence run-line). Reuses
r2_sharp devig, r2_grade, and r2_backtest's sharpness/gate helpers. Cache-first.
Run: python f5_backtest.py --seasons 2024,2025,2026
"""
import argparse
import os
import pickle
from collections import Counter

import r2_backtest as rbt
import r2_data
import r2_grade
from odds_client import american_to_decimal
from r2_sharp import fair_two_way

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_cache")


def grade_f5_ml(side, h5, a5):
    """'win' | 'loss' | 'push' for a 2-way F5 moneyline bet (tie after 5 = push)."""
    if h5 == a5:
        return "push"
    home_win = h5 > a5
    if side == "home":
        return "win" if home_win else "loss"
    return "loss" if home_win else "win"


def sharpness_rows(legs_by_season, scores_idx):
    """DK vs Pinnacle F5-ML closing accuracy: per non-tie game, DK's + Pinnacle's
    deviged fair P(home win) and the realized home win (ties excluded — a 2-way ML
    push isn't a prediction target). Feeds r2_backtest.build_sharpness_report."""
    rows = []
    for season, legs in legs_by_season.items():
        for lg in legs:
            if lg.game_pk is None or int(lg.game_pk) not in scores_idx:
                continue
            h5, a5 = scores_idx[int(lg.game_pk)]
            if h5 == a5:
                continue
            dk_fair, _ = fair_two_way(lg.dk_home, lg.dk_away)
            pin_fair, _ = fair_two_way(lg.pin_home, lg.pin_away)
            if dk_fair is None or pin_fair is None:
                continue
            rows.append({"season": str(season), "prop_key": "f5_moneyline",
                         "dk_fair": dk_fair, "pin_fair": pin_fair,
                         "over": 1.0 if h5 > a5 else 0.0})
    return rows


def grade_edge(legs_by_season, scores_idx, haircut=0.02, ev_floor=0.03):
    """Bet DK's and FanDuel's F5 ML where +EV vs the Pinnacle-derived fair; grade
    win/loss/push on the F5 score. Returns (rows, coverage)."""
    rows, cov = [], Counter()
    for season, legs in legs_by_season.items():
        for lg in legs:
            cov["events"] += 1
            pin_home, _ = fair_two_way(lg.pin_home, lg.pin_away)
            if pin_home is None:
                cov["dropped_no_pin_fair"] += 1
                continue
            if lg.game_pk is None or int(lg.game_pk) not in scores_idx:
                cov["dropped_no_score"] += 1
                continue
            h5, a5 = scores_idx[int(lg.game_pk)]
            books = [("dk", lg.dk_home, lg.dk_away)]
            if lg.fd_home is not None and lg.fd_away is not None:
                books.append(("fd", lg.fd_home, lg.fd_away))
            for book, p_home, p_away in books:
                for side, price, fair in (("home", p_home, pin_home),
                                          ("away", p_away, 1.0 - pin_home)):
                    try:
                        dec = american_to_decimal(int(price)) * (1.0 - haircut)
                    except (TypeError, ValueError):
                        continue
                    ev = fair * dec - 1.0
                    if ev < ev_floor:
                        continue
                    cov[f"selected_{book}"] += 1
                    result = grade_f5_ml(side, h5, a5)
                    rows.append({
                        "season": str(season), "book": book, "side": side,
                        "ev": ev, "ev_bucket": r2_grade.ev_bucket(ev),
                        "result": result,
                        "profit": 0.0 if result == "push" else (
                            dec - 1.0 if result == "win" else -1.0),
                        "game_pk": lg.game_pk,
                    })
                    cov["graded"] += 1
    return rows, cov


def build_edge_report(rows, cov, min_n=100, min_seasons=2, min_t=2.0):
    out = []
    p = out.append
    p("=" * 74)
    p("  F5 moneyline EDGE — bet DK/FD F5 ML vs Pinnacle-derived fair")
    p("=" * 74)
    p("\n  Coverage:")
    for k in ("events", "graded", "selected_dk", "selected_fd",
              "dropped_no_pin_fair", "dropped_no_score"):
        if cov.get(k):
            p(f"    {k:<22} {cov[k]:>8,}")
    for book in ("dk", "fd"):
        br = [r for r in rows if r["book"] == book]
        if not br:
            continue
        passed, reason, per, pooled = rbt.hardened_gate(br, min_n, min_seasons, min_t)
        p(f"\n  {book.upper()} F5 ML: pooled n={pooled.n:,} ROI={pooled.roi:+.2%} "
          f"hit={pooled.hit_rate:.1%} t={pooled.t_stat:.2f}")
        for s in sorted(per):
            sm = per[s]
            tag = "" if sm.decided >= min_n else "  (insufficient)"
            p(f"    {s}: n={sm.decided:,} ROI={sm.roi:+.2%} t={sm.t_stat:.2f}{tag}")
        p(f"    GATE: {'PASS' if passed else 'FAIL'} — {reason}")
        cells = r2_grade.by_key(br, lambda r: r["ev_bucket"])
        order = ["0%-2%", "2%-5%", "5%-10%", "10%-20%", ">=20%"]
        p(f"    by EV bucket:")
        for b in [x for x in order if x in cells]:
            sm = cells[b]
            tag = "" if sm.decided >= min_n else "  (insufficient)"
            p(f"      EV {b:<8} n={sm.decided:>5,} ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}{tag}")
    p("=" * 74)
    text = "\n".join(out)
    print(text)
    return {"text": text}


def _cache_path(sport, seasons):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = f"f5_{sport}_{'-'.join(map(str, seasons))}"
    return os.path.join(_CACHE_DIR, tag.replace("/", "_") + ".pkl")


def load_or_fetch(sport, seasons, refresh=False, refresh_mirror=False):
    path = _cache_path(sport, seasons)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    import warehouse_mirror
    warehouse_mirror.autobuild(sport, seasons, refresh=refresh_mirror)
    print(f"  cold cache build — reading from {warehouse_mirror.source_label()}")
    legs_by_season, stats = r2_data.load_f5_ml_legs(sport, seasons)
    scores_idx = r2_data.build_f5_scores_index(seasons)
    blob = {"legs_by_season": legs_by_season, "scores_idx": scores_idx,
            "stats": {s: dict(c) for s, c in stats.items()}}
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    return blob, path


def main():
    ap = argparse.ArgumentParser(description="F5 moneyline sharpness gate + edge test.")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--ev-floor", type=float, default=0.03)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--min-seasons", type=int, default=2)
    ap.add_argument("--min-t", type=float, default=2.0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--refresh-mirror", action="store_true",
                    help="re-sync + re-verify the parquet mirror (on by default; ODI_BACKTEST_MIRROR=0 disables)")
    args = ap.parse_args()
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    blob, path = load_or_fetch(args.sport, seasons, refresh=args.refresh,
                               refresh_mirror=args.refresh_mirror)
    n = sum(len(v) for v in blob["legs_by_season"].values())
    print(f"  data: {n:,} F5 ML legs (DK+Pinnacle paired)  (cache: {path})")

    print("\n  STAGE 1 — sharpness gate (is Pinnacle sharper than DK on F5?):")
    rbt.build_sharpness_report(
        sharpness_rows(blob["legs_by_season"], blob["scores_idx"]), min_n=args.min_n)

    print("\n  STAGE 2 — edge test (meaningful only if Pinnacle is sharper above):")
    rows, cov = grade_edge(blob["legs_by_season"], blob["scores_idx"],
                           haircut=args.haircut, ev_floor=args.ev_floor)
    build_edge_report(rows, cov, min_n=args.min_n, min_seasons=args.min_seasons,
                      min_t=args.min_t)
    print(f"\n  params: haircut={args.haircut} ev_floor={args.ev_floor} "
          f"min_n={args.min_n}")


if __name__ == "__main__":
    main()
