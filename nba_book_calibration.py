"""nba_book_calibration.py — correct the systematic OVER bias in de-vigged NBA prop prices,
per book and prop, for the bonus engine's leg probabilities (owner runs it; offline, free).

Finding (nba_bonus_backtest side split, every played game, 2023-26): the even-split de-vig
OVERSTATES overs and UNDERSTATES unders — favorite overs win ~1.5 pts below their de-vig P
(DK -0.016, FD -0.015), favorite unders ~0.7-1.4 pts above, in every prop at both books. The
pooled favorite-side check looked perfect only because the two sides cancel.

Method: a Platt map on the de-vigged OVER probability (logit p' = a·logit p + b; the intercept
carries the side shift), fit per (book, prop) on DISCOVERY seasons and scored on HELD-OUT
against the raw de-vig — Brier, log-loss, and the favorite gaps split by side. A map is adopted
only if it beats raw on held-out log-loss; the adopted map is then refit on ALL seasons and
(--save) staged into calibration/book_calibration_basketball_nba.json as "<prop>@<book>", which
book_calibration.apply(..., book=) serves to the bonus optimizer.

Run:
  $env:PYTHONIOENCODING="utf-8"
  python nba_book_calibration.py             # report only
  python nba_book_calibration.py --save      # stage the adopted maps (review git diff)
"""
import argparse
import math

import book_calibration as bc
import nba_bonus_backtest as nbb
import recalibration as rc

SPORT_KEY = "basketball_nba"


def _rows(book):
    """[(season, prop, fair_over, y_over)] from the favorite-leg collection (every played
    game, one main line per event/player/prop, pushes dropped)."""
    out = []
    for lg in nbb.collect(book):
        fo = lg["P"] if lg["side"] == "O" else 1.0 - lg["P"]
        yo = lg["won"] if lg["side"] == "O" else 1 - lg["won"]
        out.append((lg["season"], lg["prop"], fo, yo))
    return out


def _metrics(pairs):
    """(n, Brier, log-loss, over-fav gap, under-fav gap) for [(p_over, y_over)]."""
    n = len(pairs)
    if not n:
        return None
    eps = 1e-6
    br = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(min(max(p, eps), 1 - eps))
              + (1 - y) * math.log(1 - min(max(p, eps), 1 - eps)) for p, y in pairs) / n
    of = [(p, y) for p, y in pairs if p >= 0.5]
    uf = [(1 - p, 1 - y) for p, y in pairs if p < 0.5]
    og = (sum(y for _, y in of) - sum(p for p, _ in of)) / len(of) if of else float("nan")
    ug = (sum(y for _, y in uf) - sum(p for p, _ in uf)) / len(uf) if uf else float("nan")
    return n, br, ll, og, ug


def run(books, discovery, held_out, save):
    print("=" * 108)
    print("NBA BOOK CALIBRATION — Platt on the de-vigged OVER probability, per (book, prop)")
    print(f"  fit on {sorted(discovery)}, scored on held-out {sorted(held_out)}; adopt iff held-out "
          "log-loss beats raw; adopted map refit on all seasons")
    print("=" * 108)
    staged = []
    for book in books:
        rows = _rows(book)
        print(f"\n  [{book}] {len(rows):,} lines")
        print(f"  {'prop':<32} {'held n':>7} {'Brier raw':>9} {'Brier cal':>9} {'logL raw':>9} "
              f"{'logL cal':>9} | {'overFav gap raw->cal':>21} {'underFav gap raw->cal':>22} "
              f"{'adopt':>6}")
        for prop in nbb.NBA_PROPS + ["ALL"]:
            sub = [r for r in rows if prop == "ALL" or r[1] == prop]
            disc = [(p, y) for s, _pk, p, y in sub if s in discovery]
            held = [(p, y) for s, _pk, p, y in sub if s in held_out]
            if len(disc) < 500 or len(held) < 200:
                continue
            fit = rc.fit_platt([p for p, _ in disc], [y for _, y in disc])
            if not fit:
                continue
            a, b = fit
            cal = [(rc.apply_platt(p, a, b), y) for p, y in held]
            mr, mc = _metrics(held), _metrics(cal)
            adopt = mc[2] < mr[2]
            print(f"  {prop:<32} {mr[0]:>7,} {mr[1]:>9.5f} {mc[1]:>9.5f} {mr[2]:>9.5f} "
                  f"{mc[2]:>9.5f} | {mr[3]:>+9.3f} -> {mc[3]:>+8.3f} {mr[4]:>+10.3f} -> "
                  f"{mc[4]:>+8.3f} {('YES' if adopt else 'no'):>6}")
            if adopt and prop != "ALL":
                allp = [(p, y) for _s, _pk, p, y in sub]
                final = rc.fit_platt([p for p, _ in allp], [y for _, y in allp])
                if final:
                    staged.append((f"{prop}@{book}", {"kind": "platt",
                                                       "a": round(final[0], 6),
                                                       "b": round(final[1], 6),
                                                       "n": len(allp)}))
    print()
    if save:
        for key, entry in staged:
            path = bc.save_map(SPORT_KEY, key, entry)
        print(f"  [save] staged {len(staged)} map(s) -> calibration/book_calibration_"
              f"{SPORT_KEY}.json (review `git diff`, then commit)" if staged
              else "  [save] nothing adopted — no file written")
    else:
        print(f"  {len(staged)} map(s) would be staged (re-run with --save).")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="both", choices=("both", "draftkings", "fanduel"))
    ap.add_argument("--discovery", default="2023,2024,2025")
    ap.add_argument("--val", default="2026")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    books = ("draftkings", "fanduel") if args.book == "both" else (args.book,)
    run(books, {int(s) for s in args.discovery.split(",") if s.strip()},
        {int(s) for s in args.val.split(",") if s.strip()}, args.save)


if __name__ == "__main__":
    main()
