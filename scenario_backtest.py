"""Rule-based scenario backtests — Doug's "curiosity itch" what-ifs.

Not an edge hunt with a hardened gate; a quick, honest realized-ROI look at
simple betting RULES over the clean warehouse (2024-2026, DK prices, DK-only —
the books Doug actually bets). Every scenario reports POOLED and PER-SEASON ROI +
hit-rate + t-stat, because the recurring lesson is that a pooled edge that doesn't
replicate each season is variance, not signal (see the moneyline/team-market
nulls). Prices are DK closes; grading is off StatsAPI outcomes (final scores,
boxscore hits, pitcher ER). No Pinnacle pairing — these are single-book DK rules.

Scenarios:
  under_hits    Bet UNDER 1.5 on every DK batter_hits line. (Is the rec over-bias
                on hits exploitable straight-up on the under?)
  home_runline  Bet the HOME team +1.5 run-line whenever home is the +1.5 dog.
  fav_combo     Per game, bet the FAVORITE's ML AND the underdog's +1.5 run-line.
                Reported two ways: as a 2-leg PARLAY (wins iff the fav wins by
                exactly 1) and as TWO INDEPENDENT STRAIGHTS (a partial hedge).
  er_ml         Per game, bet the ML of the team whose starting pitcher has the
                LOWER earned_runs line (market says he'll allow fewer runs).

Run (on the faster warehouse machine):
  python scenario_backtest.py --scenario all --seasons 2024,2025,2026
  python scenario_backtest.py --scenario under_hits --refresh
"""
import argparse
import os
import pickle
from collections import Counter, defaultdict

import r2_data
import r2_grade
from odds_client import (american_to_decimal, american_to_implied_prob)

_CACHE_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "r2_backtest_cache")


# ── grading helpers ─────────────────────────────────────────────────────────

def _grade_runline(is_home, point, hs, as_):
    """'win'|'loss' for a run-line bet on the home (is_home=True) or away team at a
    signed ``point`` (e.g. +1.5). Half-lines never push. Cover iff
    team_margin + point > 0."""
    margin = (hs - as_) if is_home else (as_ - hs)
    return "win" if (margin + point) > 0 else "loss"


def _winner_home(hs, as_):
    return hs > as_          # MLB has no regulation ties


# ── close-snapshot pickers ───────────────────────────────────────────────────

def _close_prop_offers(rows):
    """Per (event_id, player_mlb_id), pick the close snapshot (latest captured
    <= commence) and collect that snapshot's offers.

    Returns {(event_id, player_mlb_id): {"game_pk", "game_date", "offers"}} where
    offers = {(round(point,1), "OVER"|"UNDER"): price}."""
    by_pg = defaultdict(list)
    for r in rows:
        pid = r.get("player_mlb_id")
        if pid is None:
            continue
        by_pg[(r.get("event_id"), pid)].append(r)
    out = {}
    for key, pr in by_pg.items():
        snaps = {}
        for r in pr:
            cap = r2_data._parse_ts(r.get("captured_at"))
            com = r2_data._parse_ts(r.get("commence_time"))
            if cap is None or com is None or cap > com:
                continue
            s = snaps.setdefault(r.get("snapshot_id"), {"cap": cap, "rows": []})
            s["rows"].append(r)
            s["cap"] = cap
        if not snaps:
            continue
        _sid, close = max(snaps.items(), key=lambda kv: kv[1]["cap"])
        meta = close["rows"][0]
        offers = {}
        for r in close["rows"]:
            pt, dr, px = r.get("point"), r.get("direction"), r.get("price")
            if pt is None or dr is None or px is None:
                continue
            offers[(round(float(pt), 1), str(dr).upper())] = px
        if offers:
            out[key] = {"game_pk": meta.get("game_pk"),
                        "game_date": meta.get("game_date"), "offers": offers}
    return out


def _central_line(offers):
    """The 'main' O/U line among a pitcher's close offers = the point whose OVER
    implied prob is closest to 0.5 (alt ladders skew away). Returns (point, over_px,
    under_px) or None."""
    best, best_gap = None, 1e9
    points = {p for (p, _d) in offers}
    for p in points:
        ov = offers.get((p, "OVER"))
        un = offers.get((p, "UNDER"))
        if ov is None:
            continue
        try:
            gap = abs(american_to_implied_prob(int(ov)) - 0.5)
        except (TypeError, ValueError):
            continue
        if gap < best_gap:
            best, best_gap = (p, ov, un), gap
    return best


# ── scenario: UNDER 1.5 batter hits ─────────────────────────────────────────

def scenario_under_hits(blob):
    rows, cov = [], Counter()
    idx = blob["hits_outcome_idx"]
    for season, prop_rows in blob["hits_rows_by_season"].items():
        for key, rec in _close_prop_offers(prop_rows).items():
            cov["player_games"] += 1
            price = rec["offers"].get((1.5, "UNDER"))
            if price is None:
                cov["no_1.5_under_line"] += 1
                continue
            gpk = rec["game_pk"]
            pid = key[1]
            actual = r2_data.outcome_value(idx, "batter_hits", pid, gpk)
            if actual is None:
                cov["no_actual"] += 1
                continue
            result = r2_grade.grade_over_under(actual, 1.5, "UNDER")
            p = r2_grade.profit(price, result)
            if p is None:
                cov["ungradable"] += 1
                continue
            cov["graded"] += 1
            rows.append({"season": str(season), "result": result, "profit": p,
                         "price": price})
    return rows, cov


# ── scenario: HOME +1.5 run-line ─────────────────────────────────────────────

def scenario_home_runline(blob):
    rows, cov = [], Counter()
    scores = blob["team_scores"]
    for season, triads in blob["triads_by_season"].items():
        for t in triads:
            cov["games"] += 1
            # Only when the HOME team is the +1.5 dog (home run-line point == +1.5).
            if abs((t.rl_home_point or 0.0) - 1.5) > 0.01:
                cov["home_not_+1.5_dog"] += 1
                continue
            gpk = t.game_pk
            if gpk is None or int(gpk) not in scores:
                cov["no_score"] += 1
                continue
            hs, as_ = scores[int(gpk)]
            result = _grade_runline(True, 1.5, hs, as_)
            p = r2_grade.profit(t.rl_home, result)
            if p is None:
                cov["ungradable"] += 1
                continue
            cov["graded"] += 1
            rows.append({"season": str(season), "result": result, "profit": p,
                         "price": t.rl_home})
    return rows, cov


# ── scenario: UNDERDOG +1.5 run-line (the isolated lead) ─────────────────────

def _fav_bucket(imp):
    """Favorite-strength bucket by its devigged-ish ML implied prob."""
    for hi in (0.55, 0.60, 0.65, 0.70):
        if imp < hi:
            return f"<{hi:.0%}"
    return ">=70%"


def scenario_dog_runline(blob):
    """Bet the UNDERDOG's +1.5 run-line in every game with a clear ML favorite
    (home OR away dog). Isolates the +4.2%/replicating signal that fell out of
    fav_combo's straights, tagged by side + favorite strength so we can see where
    the edge concentrates (the favorite-longshot bias should be strongest on heavy
    favorites)."""
    rows, cov = [], Counter()
    scores = blob["team_scores"]
    for season, triads in blob["triads_by_season"].items():
        for t in triads:
            cov["games"] += 1
            try:
                imp_home = american_to_implied_prob(int(t.ml_home))
                imp_away = american_to_implied_prob(int(t.ml_away))
            except (TypeError, ValueError):
                cov["bad_ml"] += 1
                continue
            if imp_home == imp_away:
                cov["pickem"] += 1
                continue
            fav_is_home = imp_home > imp_away
            fav_imp = max(imp_home, imp_away)
            if fav_is_home:                       # bet AWAY dog +1.5
                dog_is_home, dog_point, dog_price = False, -(t.rl_home_point or 0.0), t.rl_away
            else:                                 # bet HOME dog +1.5
                dog_is_home, dog_point, dog_price = True, (t.rl_home_point or 0.0), t.rl_home
            if abs(dog_point - 1.5) > 0.01:
                cov["dog_not_+1.5"] += 1
                continue
            gpk = t.game_pk
            if gpk is None or int(gpk) not in scores:
                cov["no_score"] += 1
                continue
            hs, as_ = scores[int(gpk)]
            result = _grade_runline(dog_is_home, 1.5, hs, as_)
            p = r2_grade.profit(dog_price, result)
            if p is None:
                cov["ungradable"] += 1
                continue
            cov["graded"] += 1
            rows.append({"season": str(season), "result": result, "profit": p,
                         "price": dog_price,
                         "side": "home_dog" if dog_is_home else "away_dog",
                         "fav_bucket": _fav_bucket(fav_imp)})
    return rows, cov


# ── scenario: FAV ML + DOG +1.5 (parlay AND two straights) ───────────────────

def scenario_fav_combo(blob):
    parlay, straights, cov = [], [], Counter()
    scores = blob["team_scores"]
    for season, triads in blob["triads_by_season"].items():
        for t in triads:
            cov["games"] += 1
            try:
                imp_home = american_to_implied_prob(int(t.ml_home))
                imp_away = american_to_implied_prob(int(t.ml_away))
            except (TypeError, ValueError):
                cov["bad_ml"] += 1
                continue
            if imp_home == imp_away:
                cov["pickem_skipped"] += 1
                continue
            fav_is_home = imp_home > imp_away
            # The favorite's ML price + the underdog's +1.5 run-line price.
            if fav_is_home:
                fav_ml = t.ml_home
                dog_rl, dog_point = t.rl_away, -(t.rl_home_point or 0.0)
            else:
                fav_ml = t.ml_away
                dog_rl, dog_point = t.rl_home, (t.rl_home_point or 0.0)
            if abs(dog_point - 1.5) > 0.01:      # underdog isn't the standard +1.5
                cov["dog_not_+1.5"] += 1
                continue
            gpk = t.game_pk
            if gpk is None or int(gpk) not in scores:
                cov["no_score"] += 1
                continue
            hs, as_ = scores[int(gpk)]
            fav_won = (_winner_home(hs, as_) == fav_is_home)
            # underdog covers +1.5 iff it loses by <=1 (or wins)
            dog_is_home = not fav_is_home
            dog_covers = _grade_runline(dog_is_home, 1.5, hs, as_) == "win"
            res_a = "win" if fav_won else "loss"
            res_b = "win" if dog_covers else "loss"
            prof_a = r2_grade.profit(fav_ml, res_a)
            prof_b = r2_grade.profit(dog_rl, res_b)
            if prof_a is None or prof_b is None:
                cov["ungradable"] += 1
                continue
            cov["graded_games"] += 1
            # (a) PARLAY — both legs must win (== fav wins by exactly 1 run).
            try:
                dec_a = american_to_decimal(int(fav_ml))
                dec_b = american_to_decimal(int(dog_rl))
            except (TypeError, ValueError):
                dec_a = dec_b = None
            if dec_a and dec_b:
                if fav_won and dog_covers:
                    parlay.append({"season": str(season), "result": "win",
                                   "profit": dec_a * dec_b - 1.0})
                else:
                    parlay.append({"season": str(season), "result": "loss",
                                   "profit": -1.0})
            # (b) TWO STRAIGHTS — each leg its own bet (partial hedge).
            straights.append({"season": str(season), "leg": "fav_ml",
                              "result": res_a, "profit": prof_a})
            straights.append({"season": str(season), "leg": "dog_+1.5",
                              "result": res_b, "profit": prof_b})
    return parlay, straights, cov


# ── scenario: ML of the lower earned-runs-line starter ───────────────────────

def scenario_er_ml(blob):
    rows, cov = [], Counter()
    scores = blob["team_scores"]
    pt_team = blob["pitcher_team"]          # {(athlete_id_str, game_pk_int): (team_id, GS)}
    gt = blob["game_teams"]                 # {game_pk_int: (home_team_id, away_team_id)}
    # Index each game's triad (DK ML + names) by game_pk.
    triad_by_pk = {}
    for triads in blob["triads_by_season"].values():
        for t in triads:
            if t.game_pk is not None:
                triad_by_pk[int(t.game_pk)] = t
    for season, prop_rows in blob["er_rows_by_season"].items():
        offers = _close_prop_offers(prop_rows)
        # group ER lines by game_pk -> {home_team_id/away_team_id: line}
        by_game = defaultdict(dict)          # gpk -> {"home": line, "away": line}
        for (eid, pid), rec in offers.items():
            gpk = rec["game_pk"]
            if gpk is None:
                continue
            gpk = int(gpk)
            central = _central_line(rec["offers"])
            if central is None:
                continue
            line = central[0]
            starter = pt_team.get((str(pid), gpk))
            if starter is None or not starter[1]:   # not found or GS != 1 (not a starter)
                continue
            team_id, _gs = starter
            teams = gt.get(gpk)
            if teams is None:
                continue
            home_tid, away_tid = teams
            if str(team_id) == str(home_tid):
                by_game[gpk]["home"] = line
            elif str(team_id) == str(away_tid):
                by_game[gpk]["away"] = line
        for gpk, lines in by_game.items():
            cov["games_with_er"] += 1
            if "home" not in lines or "away" not in lines:
                cov["missing_a_starter_line"] += 1
                continue
            if lines["home"] == lines["away"]:
                cov["tie_line_skipped"] += 1
                continue
            t = triad_by_pk.get(gpk)
            if t is None:
                cov["no_ml_triad"] += 1
                continue
            if gpk not in scores:
                cov["no_score"] += 1
                continue
            hs, as_ = scores[gpk]
            bet_home = lines["home"] < lines["away"]     # lower ER line = bet that ML
            price = t.ml_home if bet_home else t.ml_away
            won = (_winner_home(hs, as_) == bet_home)
            result = "win" if won else "loss"
            p = r2_grade.profit(price, result)
            if p is None:
                cov["ungradable"] += 1
                continue
            cov["graded"] += 1
            rows.append({"season": str(season), "result": result, "profit": p,
                         "price": price})
    return rows, cov


# ── reporting ────────────────────────────────────────────────────────────────

def _report(title, rows, cov=None, cov_keys=()):
    out = []
    p = out.append
    p("=" * 70)
    p(f"  {title}")
    p("=" * 70)
    if cov is not None:
        p("  coverage:")
        for k in cov_keys or sorted(cov):
            if cov.get(k):
                p(f"    {k:<24} {cov[k]:>8,}")
    pooled = r2_grade.summarize(rows)
    p(f"\n  POOLED: n={pooled.n:,} ROI={pooled.roi:+.2%} "
      f"hit={pooled.hit_rate:.1%} t={pooled.t_stat:+.2f} "
      f"profit={pooled.total_profit:+.1f}u")
    per = r2_grade.by_key(rows, lambda r: r["season"])
    for s in sorted(per):
        sm = per[s]
        p(f"    {s}: n={sm.n:,} ROI={sm.roi:+.2%} hit={sm.hit_rate:.1%} "
          f"t={sm.t_stat:+.2f}")
    ok = bool(per) and all(sm.roi > 0 for sm in per.values() if sm.n >= 30)
    judged = [s for s, sm in per.items() if sm.n >= 30]
    if judged:
        p(f"  per-season replication (n>=30): "
          f"{'PASS — positive every season' if ok else 'FAIL — not every season +'}")
    p("")
    print("\n".join(out))
    return pooled


def _print_slice(rows, keyfn, label):
    """Print a by-key breakdown (side, favorite bucket, ...) — ROI/hit/t per cell."""
    cells = r2_grade.by_key(rows, keyfn)
    print(f"  by {label}:")
    for k in sorted(cells, key=str):
        sm = cells[k]
        tag = "" if sm.n >= 30 else "  (thin)"
        print(f"    {str(k):<12} n={sm.n:>5,} ROI={sm.roi:+.2%} "
              f"hit={sm.hit_rate:.1%} t={sm.t_stat:+.2f}{tag}")
    print("")


# ── data load (cache-first, shared across scenarios) ─────────────────────────

def _pitcher_team_index(seasons):
    """{(athlete_id_str, game_pk_int): (team_id_str, GS_float)} from mlb_pitcher_game."""
    import mlb_warehouse as wh
    import db_store
    from sqlalchemy import select as _select
    t = wh.mlb_pitcher_game
    idx = {}
    with db_store.get_engine().connect() as conn:
        rows = conn.execute(
            _select(t.c.athlete_id, t.c.game_pk, t.c.team_id, t.c.GS)).fetchall()
    for aid, gpk, tid, gs in rows:
        if gpk is None:
            continue
        idx[(str(aid), int(gpk))] = (str(tid), gs)
    return idx


def _game_teams_index():
    """{game_pk_int: (home_team_id_str, away_team_id_str)} from mlb_game."""
    import mlb_warehouse as wh
    import db_store
    from sqlalchemy import select as _select
    g = wh.mlb_game
    idx = {}
    with db_store.get_engine().connect() as conn:
        rows = conn.execute(
            _select(g.c.game_pk, g.c.home_team_id, g.c.away_team_id)).fetchall()
    for gpk, h, a in rows:
        idx[int(gpk)] = (str(h), str(a))
    return idx


def _prop_rows_by_season(sport, seasons, prop_key):
    import db_store
    by_season = {}
    for s in seasons:
        by_season[s] = db_store.player_prop_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31",
            prop_keys=[prop_key], bookmaker="draftkings")
    return by_season


def _cache_path(sport, seasons):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = f"scenario_{sport}_{'-'.join(map(str, seasons))}"
    return os.path.join(_CACHE_DIR, tag.replace("/", "_") + ".pkl")


def load_or_fetch(sport, seasons, refresh=False):
    path = _cache_path(sport, seasons)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    import db_store
    db_store.promote_secrets_from_toml()
    triads_by_season, _ = r2_data.load_team_triad(sport, seasons)
    blob = {
        "triads_by_season": triads_by_season,
        "hits_rows_by_season": _prop_rows_by_season(sport, seasons, "batter_hits"),
        "er_rows_by_season": _prop_rows_by_season(sport, seasons, "pitcher_earned_runs"),
        "team_scores": r2_data.build_team_scores_index(seasons),
        "hits_outcome_idx": r2_data.build_outcome_index(seasons, ["batter_hits"]),
        "pitcher_team": _pitcher_team_index(seasons),
        "game_teams": _game_teams_index(),
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    return blob, path


def main():
    ap = argparse.ArgumentParser(description="Rule-based scenario backtests (DK, 2024-26).")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--scenario", default="all",
                    choices=["all", "under_hits", "home_runline", "dog_runline",
                             "fav_combo", "er_ml"])
    ap.add_argument("--refresh", action="store_true",
                    help="re-read the warehouse (else use the pickle cache)")
    args = ap.parse_args()
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    blob, path = load_or_fetch(args.sport, seasons, refresh=args.refresh)
    ng = sum(len(v) for v in blob["triads_by_season"].values())
    print(f"  data: {ng:,} DK team triads + hits/ER prop lines  (cache: {path})\n")

    want = args.scenario
    if want in ("all", "under_hits"):
        rows, cov = scenario_under_hits(blob)
        _report("UNDER 1.5 batter hits (every DK line)", rows, cov,
                ("player_games", "graded", "no_1.5_under_line", "no_actual"))
    if want in ("all", "home_runline"):
        rows, cov = scenario_home_runline(blob)
        _report("HOME +1.5 run-line (home is the +1.5 dog)", rows, cov,
                ("games", "graded", "home_not_+1.5_dog", "no_score"))
    if want in ("all", "dog_runline"):
        rows, cov = scenario_dog_runline(blob)
        _report("UNDERDOG +1.5 run-line (clear ML favorite; home or away dog)", rows,
                cov, ("games", "graded", "dog_not_+1.5", "no_score", "pickem"))
        _print_slice(rows, lambda r: r["side"], "side")
        _print_slice(rows, lambda r: r["fav_bucket"], "favorite strength (ML implied)")
        # Does the hot 65-70% bucket replicate per season, or is it one lucky year?
        # (an isolated spike that lives in a single season = variance, not edge.)
        _print_slice(rows, lambda r: (r["fav_bucket"], r["season"]),
                     "favorite strength x season")
    if want in ("all", "fav_combo"):
        parlay, straights, cov = scenario_fav_combo(blob)
        print(f"  fav_combo coverage: graded_games={cov.get('graded_games',0):,} "
              f"pickem={cov.get('pickem_skipped',0):,} "
              f"dog_not_+1.5={cov.get('dog_not_+1.5',0):,} "
              f"no_score={cov.get('no_score',0):,}\n")
        _report("FAV ML + DOG +1.5 — as a 2-LEG PARLAY (fav wins by exactly 1)",
                parlay)
        _report("FAV ML + DOG +1.5 — as TWO STRAIGHTS (all legs pooled)", straights)
        for leg in ("fav_ml", "dog_+1.5"):
            _report(f"   straights breakdown: {leg}",
                    [r for r in straights if r["leg"] == leg])
    if want in ("all", "er_ml"):
        rows, cov = scenario_er_ml(blob)
        _report("ML of the LOWER earned-runs-line starter", rows, cov,
                ("games_with_er", "graded", "missing_a_starter_line",
                 "tie_line_skipped", "no_score"))


if __name__ == "__main__":
    main()
