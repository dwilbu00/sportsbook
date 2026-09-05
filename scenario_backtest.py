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
from r2_sharp import fair_two_way   # devigged fair prob — MUST match coherence_flags' live band

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_cache")


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


def _fav_bucket_fine(imp):
    """2.5%-wide favorite-strength bins to check whether the 65-70% signal is a
    smooth PLATEAU (trust) or a single-bin SPIKE (knife-edge = suspect)."""
    edges = [0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75]
    if imp < edges[0]:
        return "a <60.0%"
    lo = edges[0]
    for i, hi in enumerate(edges[1:], 1):
        if imp < hi:
            return f"{chr(ord('b') + i - 1)} {lo:.1%}-{hi:.1%}"
        lo = hi
    return "z >=75.0%"


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
            # Band/bucket on the DEVIGGED fair prob (Clarke power method) — the EXACT
            # basis coherence_flags._fav_band_ok uses live. Raw american-implied (with
            # vig) shifts the [fav_min,fav_max) boundary vs the live gate, so boundary
            # games near 0.60/0.70 classify oppositely and the backtest measures a
            # different population than we actually bet.
            fair_home, fair_away = fair_two_way(t.ml_home, t.ml_away)
            if fair_home is None or fair_away is None:
                cov["bad_ml"] += 1
                continue
            if fair_home == fair_away:
                cov["pickem"] += 1
                continue
            fav_is_home = fair_home > fair_away
            fav_imp = max(fair_home, fair_away)
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
                         "price": dog_price, "fav_imp": fav_imp,
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


# ── scenario: SP volatility (ER-CV) → team-market mispricing ─────────────────

def _asof_sp_cv(games_sorted, game_date, half_life):
    """Recency-weighted ER-CV of a starter's PRIOR in-season starts, leakage-safe
    (strictly before game_date), via props._recency_weighted_cv — the EXACT function
    the earned_runs high-CV (cv_floor) edge was validated on. games_sorted =
    [(official_date, outs, er, ...), ...] ascending. Returns None if <5 priors / mean~0."""
    from props import _recency_weighted_cv
    gd = (game_date or "")[:10]
    prior = [g[2] for g in games_sorted if (g[0] or "") < gd]   # ER, oldest→newest
    prior.reverse()                                             # newest-first (weights convention)
    return _recency_weighted_cv(prior, half_life)


def _cv_bucket(cv):
    """SP ER-CV bucket; the >=1.3 cut is the validated earned_runs cv_floor threshold."""
    if cv is None:
        return None
    if cv < 1.0:
        return "a <1.0"
    if cv < 1.3:
        return "b 1.0-1.3"
    return "c >=1.3 (high)"


def scenario_team_variance(blob, half_life=5.0):
    """Does STARTER volatility (recency-weighted ER-CV — the validated cv_floor signal)
    predict DK mispricing TEAM markets? Fat-right-tailed volatile SPs plausibly make the
    team total OVER + the underdog ML/+1.5 underpriced. Bets: total OVER (conditioned on
    the game's MAX SP CV — either starter volatile → blow-up risk), and dog ML + dog +1.5
    (conditioned on the FAVORITE's SP CV — a volatile favorite can implode). Rows tagged
    by CV bucket so we see whether +ROI concentrates in the high-CV (>=1.3) cell."""
    rows, cov = [], Counter()
    scores = blob["team_scores"]
    pt_team = blob["pitcher_team"]        # {(aid_str, gpk_int): (team_id, GS)}
    gt = blob["game_teams"]               # {gpk: (home_tid, away_tid)}
    pidx = blob["pitcher_idx"]            # {season_str: {aid_str: [(date,outs,er,...)]}}
    # starters per game_pk (GS==1), mapped to home/away
    starters = {}
    for (aid, gpk), (tid, gs) in pt_team.items():
        if not gs:
            continue
        teams = gt.get(gpk)
        if not teams:
            continue
        if str(tid) == str(teams[0]):
            starters.setdefault(gpk, {})["home"] = aid
        elif str(tid) == str(teams[1]):
            starters.setdefault(gpk, {})["away"] = aid
    for season, triads in blob["triads_by_season"].items():
        idx = pidx.get(str(season)) or {}
        for t in triads:
            cov["games"] += 1
            gpk = t.game_pk
            if gpk is None or int(gpk) not in scores:
                cov["no_score"] += 1
                continue
            gpk = int(gpk)
            hs, as_ = scores[gpk]
            st = starters.get(gpk) or {}
            home_cv = _asof_sp_cv(idx.get(str(st.get("home"))) or [], t.game_date, half_life)
            away_cv = _asof_sp_cv(idx.get(str(st.get("away"))) or [], t.game_date, half_life)
            try:
                imp_home = american_to_implied_prob(int(t.ml_home))
                imp_away = american_to_implied_prob(int(t.ml_away))
            except (TypeError, ValueError):
                imp_home = imp_away = None
            fav_is_home = (imp_home is not None and imp_away is not None
                           and imp_home > imp_away)
            # (1) TOTAL OVER — conditioned on the game's MAX SP CV
            cvs = [c for c in (home_cv, away_cv) if c is not None]
            if cvs and t.total_line is not None and t.total_over is not None:
                gv = max(cvs)
                res = r2_grade.grade_over_under(hs + as_, t.total_line, "OVER")
                p = r2_grade.profit(t.total_over, res)
                if p is not None:
                    rows.append({"season": str(season), "bet": "total_over", "cv": gv,
                                 "cv_bucket": _cv_bucket(gv), "result": res, "profit": p})
                    cov["total_over_graded"] += 1
            else:
                cov["skip_no_cv"] += 1
            # (2)+(3) DOG ML + DOG +1.5 — conditioned on the FAVORITE's SP CV
            if imp_home is not None and imp_away is not None and imp_home != imp_away:
                fav_cv = home_cv if fav_is_home else away_cv
                if fav_cv is None:
                    cov["skip_no_fav_cv"] += 1
                    continue
                dog_is_home = not fav_is_home
                b = _cv_bucket(fav_cv)
                dog_ml = t.ml_home if dog_is_home else t.ml_away
                res_ml = "win" if (_winner_home(hs, as_) == dog_is_home) else "loss"
                p_ml = r2_grade.profit(dog_ml, res_ml)
                if p_ml is not None:
                    rows.append({"season": str(season), "bet": "dog_ml", "cv": fav_cv,
                                 "cv_bucket": b, "result": res_ml, "profit": p_ml})
                    cov["dog_ml_graded"] += 1
                dog_point = (t.rl_home_point or 0.0) if dog_is_home else -(t.rl_home_point or 0.0)
                dog_rl = t.rl_home if dog_is_home else t.rl_away
                if abs(dog_point - 1.5) < 0.01:
                    res_rl = _grade_runline(dog_is_home, 1.5, hs, as_)
                    p_rl = r2_grade.profit(dog_rl, res_rl)
                    if p_rl is not None:
                        rows.append({"season": str(season), "bet": "dog_rl", "cv": fav_cv,
                                     "cv_bucket": b, "result": res_rl, "profit": p_rl})
                        cov["dog_rl_graded"] += 1
    return rows, cov


def scenario_coherence_stable_sp(blob, half_life=5.0, fav_min=0.60, fav_max=0.70):
    """SHARPEN the coherence dog+1.5 edge with a STABLE-favorite-SP filter. The live
    coherence edge is dog +1.5 at MODERATE favorites (~[fav_min,fav_max) ML-implied).
    The team_variance byproduct found dog+1.5 concentrates in LOW-CV (stable) favorite
    games (+6.5%/t=3.18) and dies on high-CV (volatile) favorites. Test that DIRECTLY
    on the coherence band: bet dog+1.5 everywhere, tag by (in-band?, favorite-SP ER-CV
    bucket), so a fav-SP-CV_MAX gate's lift on the live flag is measurable. Per-season
    replication of the IN-BAND STABLE cell is the honesty gate. fav_cv via the EXACT
    _asof_sp_cv the earned_runs cv_floor edge was validated on (leakage-safe, priors
    strictly before game_date)."""
    rows, cov = [], Counter()
    scores = blob["team_scores"]
    pt_team = blob["pitcher_team"]        # {(aid_str, gpk_int): (team_id, GS)}
    gt = blob["game_teams"]               # {gpk: (home_tid, away_tid)}
    pidx = blob["pitcher_idx"]            # {season_str: {aid_str: [(date,outs,er,...)]}}
    starters = {}                         # gpk -> {"home"/"away": aid} (GS==1)
    for (aid, gpk), (tid, gs) in pt_team.items():
        if not gs:
            continue
        teams = gt.get(gpk)
        if not teams:
            continue
        if str(tid) == str(teams[0]):
            starters.setdefault(gpk, {})["home"] = aid
        elif str(tid) == str(teams[1]):
            starters.setdefault(gpk, {})["away"] = aid
    for season, triads in blob["triads_by_season"].items():
        idx = pidx.get(str(season)) or {}
        for t in triads:
            cov["games"] += 1
            # Band/bucket on the DEVIGGED fair prob (Clarke power method) — the EXACT
            # basis coherence_flags._fav_band_ok uses live. Raw american-implied (with
            # vig) shifts the [fav_min,fav_max) boundary vs the live gate, so boundary
            # games near 0.60/0.70 classify oppositely and the backtest measures a
            # different population than we actually bet.
            fair_home, fair_away = fair_two_way(t.ml_home, t.ml_away)
            if fair_home is None or fair_away is None:
                cov["bad_ml"] += 1
                continue
            if fair_home == fair_away:
                cov["pickem"] += 1
                continue
            fav_is_home = fair_home > fair_away
            fav_imp = max(fair_home, fair_away)
            dog_is_home = not fav_is_home
            dog_point = (t.rl_home_point or 0.0) if dog_is_home else -(t.rl_home_point or 0.0)
            if abs(dog_point - 1.5) > 0.01:
                cov["dog_not_+1.5"] += 1
                continue
            gpk = t.game_pk
            if gpk is None or int(gpk) not in scores:
                cov["no_score"] += 1
                continue
            gpk = int(gpk)
            hs, as_ = scores[gpk]
            st = starters.get(gpk) or {}
            fav_aid = st.get("home") if fav_is_home else st.get("away")
            fav_cv = _asof_sp_cv(idx.get(str(fav_aid)) or [], t.game_date, half_life)
            if fav_cv is None:
                cov["no_fav_cv"] += 1
                continue
            dog_rl = t.rl_home if dog_is_home else t.rl_away
            res = _grade_runline(dog_is_home, 1.5, hs, as_)
            p = r2_grade.profit(dog_rl, res)
            if p is None:
                cov["ungradable"] += 1
                continue
            cov["graded"] += 1
            in_band = (fav_min <= fav_imp < fav_max)
            rows.append({"season": str(season), "result": res, "profit": p,
                         "fav_imp": fav_imp, "in_band": in_band,
                         "band": "in_band" if in_band else "out_band",
                         "cv": fav_cv, "cv_bucket": _cv_bucket(fav_cv),
                         "side": "home_dog" if dog_is_home else "away_dog"})
    return rows, cov


def _report_coherence_stable_sp(rows, cov, fav_min, fav_max, half_life):
    band = [r for r in rows if r["in_band"]]
    print("=" * 74)
    print(f"  COHERENCE dog+1.5 × STABLE-favorite-SP  "
          f"(band=[{fav_min:.2f},{fav_max:.2f}) devigged-fair ML, cv_half_life={half_life})")
    print(f"  games={cov.get('games',0):,}  graded={cov.get('graded',0):,}  "
          f"in-band={len(band):,}  (no_fav_cv={cov.get('no_fav_cv',0):,}, "
          f"dog_not_+1.5={cov.get('dog_not_+1.5',0):,})")
    print("=" * 74)
    # 1) THE MONEY TABLE — in-band dog+1.5 (the coherence bet) by favorite-SP CV.
    _report("IN-BAND dog+1.5 (the coherence bet), all favorite-SP CV pooled", band)
    _print_slice(band, lambda r: r["cv_bucket"], "favorite SP ER-CV bucket (IN-BAND)")
    # 2) HONESTY GATE — does the stable (<1.0) in-band cell replicate per season?
    stable = [r for r in band if r["cv_bucket"] == "a <1.0"]
    if stable:
        _print_slice(stable, lambda r: r["season"], "STABLE (<1.0) IN-BAND × season")
    # 3) INTERACTION — is the CV effect band-SPECIFIC or does it hold out-of-band too?
    _print_slice(rows, lambda r: (r["band"], r["cv_bucket"]), "band × CV bucket")
    print("  READ: want the IN-BAND '<1.0' cell +ROI & replicating, and clearly better")
    print("  than IN-BAND '>=1.3'. If so, add a favorite-SP-CV_MAX gate to coherence_flags.")
    print("=" * 74)


# ── scenario: per-(prop × line × side) realized ROI stratification ───────────

# The 7 gradable DK props (mirrors mlb_warehouse._ACTUAL_STAT_SPEC).
_ALL_PROPS = ("batter_hits", "batter_strikeouts", "batter_total_bases", "batter_rbis",
              "pitcher_strikeouts", "pitcher_earned_runs", "pitcher_outs")


def scenario_prop_roi(sport, seasons):
    """Realized ROI of betting EVERY DK prop line (raw DK price, vig in), bucketed by
    (prop, line, side), to surface bettable strata the prop-aggregate hides — the same
    'find the +ROI stratum' move that produced cv_floor + dog+1.5, on the prop/line/side
    axis, extending the replicating DK recreational OVER-bias. Ungated screen; the
    per-season replication of a cell is the honesty gate. Loads its own data (mirror-
    backed via r2_data)."""
    idx = r2_data.build_outcome_index(seasons, list(_ALL_PROPS))
    rows, cov = [], Counter()
    for prop_key in _ALL_PROPS:
        for s in seasons:
            prop_rows = r2_data._read_player_prop_lines(
                sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31",
                prop_keys=[prop_key], bookmaker="draftkings")
            for (eid, pid), rec in _close_prop_offers(prop_rows).items():
                gpk = rec["game_pk"]
                actual = r2_data.outcome_value(idx, prop_key, pid, gpk)
                if actual is None:
                    cov[f"{prop_key}:no_actual"] += 1
                    continue
                for (line, direction), price in rec["offers"].items():
                    res = r2_grade.grade_over_under(actual, line, direction)
                    p = r2_grade.profit(price, res)
                    if p is None:
                        continue
                    rows.append({"season": str(s), "prop": prop_key, "line": line,
                                 "side": str(direction).upper(), "result": res,
                                 "profit": p})
                    cov[f"{prop_key}:graded"] += 1
    return rows, cov


# ── scenario: line-timing feasibility probe (is there an early→close path?) ──

def scenario_line_timing(sport, seasons):
    """STEP 1 for the line-timing / CLV-decay study: does the warehouse hold a real
    early→close price PATH per event, or only single (close) captures? A CLV-by-lead-
    time study is only buildable offline if events have BOTH an early and a near-close
    snapshot; otherwise it needs Odds-API historical credits. Reports, per market kind
    (team / props, DK), snapshots-per-event + the lead-time (hours-to-commence)
    distribution. Mirror-backed; no Azure."""
    import statistics

    def _lead_hours(cap, com):
        c1, c2 = r2_data._parse_ts(cap), r2_data._parse_ts(com)
        if c1 is None or c2 is None:
            return None
        return (c2 - c1).total_seconds() / 3600.0

    print("=" * 74)
    print("  LINE-TIMING feasibility — snapshots-per-event + lead-time coverage (DK)")
    print("=" * 74)
    for kind_name, reader in (("team ", r2_data._read_team_market_lines),
                              ("props", r2_data._read_player_prop_lines)):
        snaps_by_event = defaultdict(set)
        leads_by_event = defaultdict(list)
        for s in seasons:
            rows = reader(sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31",
                          bookmaker="draftkings")
            for r in rows:
                eid = r.get("event_id")
                if eid is None:
                    continue
                snaps_by_event[eid].add(r.get("captured_at"))
                lh = _lead_hours(r.get("captured_at"), r.get("commence_time"))
                if lh is not None:
                    leads_by_event[eid].append(lh)
        n_ev = max(len(snaps_by_event), 1)
        multi = sum(1 for c in snaps_by_event.values() if len(c) >= 2)
        # a usable PATH = an event with a snapshot >6h out AND one <1h out
        path = sum(1 for leads in leads_by_event.values()
                   if leads and max(leads) > 6 and min(leads) < 1)
        all_leads = [l for leads in leads_by_event.values() for l in leads]
        print(f"\n  {kind_name} (DK): events={len(snaps_by_event):,}  "
              f">=2 snapshots={multi:,} ({100*multi/n_ev:.0f}%)  "
              f"early+close path (>6h & <1h)={path:,} ({100*path/n_ev:.0f}%)")
        if all_leads:
            q = statistics.quantiles(all_leads, n=4) if len(all_leads) > 3 else [0, 0, 0]
            print(f"    lead-hours over {len(all_leads):,} snapshots: "
                  f"min={min(all_leads):.1f}  p25={q[0]:.1f}  med={q[1]:.1f}  "
                  f"p75={q[2]:.1f}  max={max(all_leads):.1f}")
    print("\n  READ: a real CLV-by-lead-time study needs the 'early+close path' % to be")
    print("  meaningful. If ~0, our captures are single/close-only -> the study needs")
    print("  Odds-API historical credits (a spend), not just the warehouse.")
    print("=" * 74)


# ── line-timing STUDY: realized ROI at EARLY vs CLOSE, per team side-category ──

def _snapshot_sides(rows):
    """One snapshot's team rows -> the per-side prices/lines + home/away/game_pk."""
    meta = rows[0]
    home, away = meta.get("home"), meta.get("away")
    ml, rl, tot = {}, {}, {}
    for r in rows:
        bt, sel = r.get("bet_type"), r.get("selection")
        if bt == "moneyline":
            ml[sel] = r.get("price")
        elif bt == "spread":
            rl[sel] = (r.get("point"), r.get("price"))
        elif bt == "total":
            tot[sel] = (r.get("point"), r.get("price"))
    rlh, rla = rl.get(home), rl.get(away)
    ov, un = tot.get("Over"), tot.get("Under")
    return {"home": home, "away": away, "game_pk": meta.get("game_pk"),
            "ml_home": ml.get(home), "ml_away": ml.get(away),
            "rl_pt": rlh[0] if rlh else None,
            "rl_home": rlh[1] if rlh else None, "rl_away": rla[1] if rla else None,
            "total_line": ov[0] if ov else None,
            "over": ov[1] if ov else None, "under": un[1] if un else None}


def _event_snapshots(rows):
    """Per event -> [(lead_hours, sides), ...] for all PRE-commence snapshots, EARLIEST
    (largest lead) first; in-play (lead<0) dropped."""
    by_event = defaultdict(list)
    for r in rows:
        by_event[r.get("event_id")].append(r)
    out = {}
    for eid, ev in by_event.items():
        by_cap, lead_of = defaultdict(list), {}
        for r in ev:
            cap = r2_data._parse_ts(r.get("captured_at"))
            com = r2_data._parse_ts(r.get("commence_time"))
            if cap is None or com is None:
                continue
            lead = (com - cap).total_seconds() / 3600.0
            if lead < 0:
                continue
            k = r.get("captured_at")
            by_cap[k].append(r)
            lead_of[k] = lead
        snaps = sorted(((lead_of[k], _snapshot_sides(rs)) for k, rs in by_cap.items()),
                       key=lambda x: -x[0])
        if snaps:
            out[eid] = snaps
    return out


def _mkt_ok(cat, s):
    """Does this snapshot carry the market for a category (both sides / the ±1.5 RL)?"""
    if cat in ("fav_ml", "dog_ml"):
        return s["ml_home"] is not None and s["ml_away"] is not None
    if cat == "over":
        return s["total_line"] is not None and s["over"] is not None
    if cat == "under":
        return s["total_line"] is not None and s["under"] is not None
    return (s["rl_pt"] is not None and abs(abs(s["rl_pt"]) - 1.5) < 0.01
            and s["rl_home"] is not None and s["rl_away"] is not None)


_LT_CATS = ("fav_ml", "dog_ml", "over", "under", "rl_dog_+1.5", "rl_fav_-1.5")


def scenario_line_timing_study(sport, seasons, min_gap_h=6.0):
    """Per team event, for EACH side-category independently, pick CLOSE = the LATEST
    pre-commence snapshot that carries that market, and EARLY = the earliest snapshot
    carrying it at least ``min_gap_h`` hours before that close. Grade the category at
    both (own line+price; line moves are part of the timing value). Only events where
    BOTH exist for a category are counted — a close capture that dropped spreads/totals
    just lowers that market's paired-n instead of faking a 0% ROI. Favorite is fixed by
    the close moneyline. Returns rows tagged {season, category, timing}."""
    scores = r2_data.build_team_scores_index(seasons)
    rows, cov = [], Counter()

    def cat_pr(sd, cat, fav_home, winner_home, hs, as_):
        """(price, result) for a category off one snapshot, or (None, None) if N/A."""
        if cat == "fav_ml":
            return (sd["ml_home"] if fav_home else sd["ml_away"],
                    "win" if winner_home == fav_home else "loss")
        if cat == "dog_ml":
            return (sd["ml_away"] if fav_home else sd["ml_home"],
                    "win" if winner_home != fav_home else "loss")
        if cat == "over":
            return sd["over"], r2_grade.grade_over_under(hs + as_, sd["total_line"], "OVER")
        if cat == "under":
            return sd["under"], r2_grade.grade_over_under(hs + as_, sd["total_line"], "UNDER")
        dog_home = not fav_home
        if cat == "rl_dog_+1.5":
            if abs((sd["rl_pt"] if dog_home else -sd["rl_pt"]) - 1.5) > 0.01:
                return None, None
            return (sd["rl_home"] if dog_home else sd["rl_away"],
                    _grade_runline(dog_home, 1.5, hs, as_))
        if abs((sd["rl_pt"] if fav_home else -sd["rl_pt"]) + 1.5) > 0.01:
            return None, None
        return (sd["rl_home"] if fav_home else sd["rl_away"],
                _grade_runline(fav_home, -1.5, hs, as_))

    for s in seasons:
        team_rows = r2_data._read_team_market_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31", bookmaker="draftkings")
        team_rows = [r for r in team_rows if r.get("kind") == "team"]  # drop F5
        for eid, snaps in _event_snapshots(team_rows).items():
            cov["events"] += 1
            close_ml = next((sd for ld, sd in reversed(snaps)
                             if sd["ml_home"] is not None and sd["ml_away"] is not None), None)
            gpk = snaps[0][1].get("game_pk")
            if close_ml is None or gpk is None or int(gpk) not in scores:
                cov["no_close_ml_or_score"] += 1
                continue
            try:
                fav_home = (american_to_implied_prob(int(close_ml["ml_home"]))
                            > american_to_implied_prob(int(close_ml["ml_away"])))
            except (TypeError, ValueError):
                continue
            hs, as_ = scores[int(gpk)]
            winner_home = hs > as_
            for cat in _LT_CATS:
                close = next(((ld, sd) for ld, sd in reversed(snaps) if _mkt_ok(cat, sd)), None)
                if close is None:
                    continue
                c_lead, c_sd = close
                early = next(((ld, sd) for ld, sd in snaps
                              if _mkt_ok(cat, sd) and ld >= c_lead + min_gap_h), None)
                if early is None:
                    cov[f"no_early:{cat}"] += 1
                    continue
                e_sd = early[1]
                ep, er = cat_pr(e_sd, cat, fav_home, winner_home, hs, as_)
                cp, cr = cat_pr(c_sd, cat, fav_home, winner_home, hs, as_)
                pe = r2_grade.profit(ep, er) if er else None
                pc = r2_grade.profit(cp, cr) if cr else None
                if pe is None or pc is None:
                    continue
                cov[f"pairs:{cat}"] += 1
                rows.append({"season": str(s), "category": cat, "timing": "early",
                             "result": er, "profit": pe})
                rows.append({"season": str(s), "category": cat, "timing": "close",
                             "result": cr, "profit": pc})
    return rows, cov


def _report_line_timing(rows, cov):
    print("=" * 74)
    print("  LINE TIMING — realized ROI at EARLY vs CLOSE, by team side-category")
    print(f"  events={cov.get('events',0):,}   n = PAIRED events (both an early >=6h out")
    print("  AND a close price for that same market); ROIs are on that identical set.")
    print("=" * 74)
    print(f"  {'category':<14} {'n':>6}  {'ROI@early':>10} {'ROI@close':>10} "
          f"{'ΔROI(e-c)':>10}  per-season Δ")
    for cat in _LT_CATS:
        er = [r for r in rows if r["category"] == cat and r["timing"] == "early"]
        cr = [r for r in rows if r["category"] == cat and r["timing"] == "close"]
        e, c = r2_grade.summarize(er), r2_grade.summarize(cr)
        d = e.roi - c.roi
        signs = []
        for s in sorted({r["season"] for r in er}):
            es = r2_grade.summarize([r for r in er if r["season"] == s])
            cs = r2_grade.summarize([r for r in cr if r["season"] == s])
            if es.n >= 30:
                signs.append("+" if (es.roi - cs.roi) > 0 else "-")
        thin = "" if e.n >= 100 else "  (thin)"
        print(f"  {cat:<14} {e.n:>6,}  {e.roi:>+9.2%} {c.roi:>+9.2%} {d:>+9.2%}  "
              f"{''.join(signs) or 'n/a'}{thin}")
    print("\n  ΔROI>0 => betting EARLY beat CLOSE for that market (bet early); <0 => wait.")
    print("  n=0 or thin => close capture lacks that market (RL/totals dropped near game")
    print("  time) -> not answerable offline for that market, NOT a real 0% ROI.")
    print("=" * 74)


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


def _report_prop_roi(rows, cov, min_n=100):
    """Per (prop, line, side) cell: pooled ROI/hit/t + per-season replication. Ends
    with the CANDIDATE cells (n>=min_n, +ROI pooled, +ROI every judged season) — the
    actionable strata to consider gating to."""
    print("=" * 74)
    print("  PROP ROI by (prop, line, side) — DK raw price, every line, UNGATED")
    print("=" * 74)
    candidates = []
    for prop in _ALL_PROPS:
        pr = [r for r in rows if r["prop"] == prop]
        if not pr:
            continue
        print(f"\n  {prop}  (n={len(pr):,}):")
        cells = r2_grade.by_key(pr, lambda r: (r["side"], r["line"]))
        for key in sorted(cells, key=lambda k: (k[0], k[1])):
            side, line = key
            sm = cells[key]
            per = r2_grade.by_key([r for r in pr if r["side"] == side and r["line"] == line],
                                  lambda r: r["season"])
            judged = [s for s, x in per.items() if x.n >= 30]
            repl = bool(judged) and all(per[s].roi > 0 for s in judged)
            thin = "" if sm.n >= min_n else "  (thin)"
            hit = f"{sm.hit_rate:.1%}" if sm.decided else "n/a"
            star = "  <== +ROI, replicates" if (sm.n >= min_n and sm.roi > 0 and repl) else ""
            print(f"    {side:<5} {line:>5}  n={sm.n:>6,} ROI={sm.roi:+.2%} "
                  f"hit={hit} t={sm.t_stat:+.2f}{thin}{star}")
            if sm.n >= min_n and sm.roi > 0 and repl:
                candidates.append((prop, side, line, sm))
    print(f"\n  === CANDIDATE cells (n>={min_n}, +ROI pooled, +ROI every judged season) ===")
    if not candidates:
        print("    (none — no stratum beats vig with per-season replication)")
    for prop, side, line, sm in sorted(candidates, key=lambda c: -c[3].roi):
        print(f"    {prop:<20} {side:<5} {line:>5}  n={sm.n:>6,} "
              f"ROI={sm.roi:+.2%} t={sm.t_stat:+.2f}")
    print("=" * 74)


# ── data load (cache-first, shared across scenarios) ─────────────────────────

def _pitcher_team_index(seasons):
    """{(athlete_id_str, game_pk_int): (team_id_str, GS_float)} from mlb_pitcher_game."""
    _m = r2_data._mirror()
    if _m is not None:
        mi = _m.pitcher_team_index(seasons=seasons)
        if mi is not None:
            return mi
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
    _m = r2_data._mirror()
    if _m is not None:
        mi = _m.game_teams_index()
        if mi is not None:
            return mi
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
    # Routes through r2_data's mirror-aware reader (parquet if enabled, else Azure).
    by_season = {}
    for s in seasons:
        by_season[s] = r2_data._read_player_prop_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31",
            prop_keys=[prop_key], bookmaker="draftkings")
    return by_season


def _pitcher_idx_by_season(seasons):
    """{season_str: {athlete_id_str: [(official_date, outs, er, ...) asc]}} — the as-of
    pitcher ER series for leakage-safe SP volatility. Mirror (parquet) if enabled, else
    mlb_warehouse's per-season pitcher game index."""
    _m = r2_data._mirror()
    out = {}
    for s in seasons:
        idx = _m.pitcher_game_index(int(s)) if _m is not None else None
        if idx is None:
            import mlb_warehouse as wh
            idx = wh._pitcher_game_index(int(s)) or {}
        out[str(s)] = {str(aid): games for aid, games in idx.items()}
    return out


def _cache_path(sport, seasons):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    # v2 adds pitcher_idx (SP volatility) — bump so an old cache doesn't KeyError.
    tag = f"scenario_v2_{sport}_{'-'.join(map(str, seasons))}"
    return os.path.join(_CACHE_DIR, tag.replace("/", "_") + ".pkl")


def load_or_fetch(sport, seasons, refresh=False, refresh_mirror=False):
    path = _cache_path(sport, seasons)
    if not refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path
    import warehouse_mirror
    warehouse_mirror.autobuild(sport, seasons, refresh=refresh_mirror)
    print(f"  cold cache build — reading from {warehouse_mirror.source_label()}")
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
        "pitcher_idx": _pitcher_idx_by_season(seasons),
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    return blob, path


# ── scenario: UPSET DATAMINE — what schedule/venue/umpire factors move ROI? ──

def _load_game_meta(seasons):
    """Direct warehouse read of per-game schedule/venue/umpire meta (regular-season,
    final), keyed by game_pk. Read-only + mirror-independent (this datamine is
    exploratory + run infrequently, so a live Azure read is fine)."""
    import db_store
    import mlb_warehouse as wh
    from sqlalchemy import select as _sel
    db_store.promote_secrets_from_toml()
    g = wh.mlb_game
    eng = db_store.get_engine()
    out = {}
    for s in seasons:
        with eng.connect() as conn:
            rows = conn.execute(_sel(
                g.c.game_pk, g.c.official_date, g.c.venue_id, g.c.double_header,
                g.c.game_number, g.c.hp_umpire_name, g.c.home_team_id,
                g.c.away_team_id, g.c.home_score, g.c.away_score
            ).where((g.c.season == int(s)) & (g.c.game_type == "R")
                    & (g.c.home_score.isnot(None)))).fetchall()
        for r in rows:
            out[int(r._mapping["game_pk"])] = dict(r._mapping)
    return out


def _rest_bucket(r):
    if r is None:
        return "unk"
    if r <= 0:
        return "0 (DH/same-day)"
    if r == 1:
        return "1 (daily)"
    if r == 2:
        return "2 (1 off-day)"
    return "3+ (rested)"


def _team_schedule_features(meta):
    """Per (game_pk, team_id): days rest since the team's previous game, whether it
    traveled (venue changed from its last game), and whether it's a doubleheader game.
    Built from the team's own game sequence within the regular-season meta."""
    import datetime as _dt
    by_team = defaultdict(list)      # tid -> [(official_date, game_number, gpk, venue_id)]
    for gpk, m in meta.items():
        for tid in (m["home_team_id"], m["away_team_id"]):
            by_team[str(tid)].append(
                (m["official_date"], m["game_number"] or 1, gpk, m["venue_id"]))
    feat = defaultdict(dict)
    for tid, games in by_team.items():
        games.sort(key=lambda x: (x[0] or "", x[1]))
        prev_date, prev_venue = None, None
        for od, gn, gpk, ven in games:
            rest = None
            if prev_date and od:
                try:
                    rest = (_dt.date.fromisoformat(od)
                            - _dt.date.fromisoformat(prev_date)).days
                except ValueError:
                    rest = None
            traveled = (prev_venue is not None and ven is not None and ven != prev_venue)
            dh = (meta[gpk]["double_header"] not in (None, "N")) or (gn and gn > 1)
            feat[gpk][tid] = {"rest": rest, "traveled": bool(traveled), "dh": bool(dh)}
            prev_date, prev_venue = od, ven
    return feat


def scenario_upset_datamine(sport, seasons):
    """DATAMINE: which schedule / venue / umpire factors move the realized ROI of the
    base team bets? For each game, tag it with candidate factors (home-plate umpire,
    doubleheader, favorite/underdog days-rest, favorite/underdog travel) and record the
    realized ROI of each base bet (dog ML, dog +1.5, total OVER, total UNDER). The
    reporter stratifies ROI by factor bucket with per-season replication — a +ROI cell
    that replicates is a candidate edge (or a bet-disqualification signal, if -ROI).
    Read-only; loads its own odds (r2_data) + game meta (warehouse). Actionable output,
    not descriptive: every cell is a bet you could actually place."""
    import r2_data
    triads_by_season, _ = r2_data.load_team_triad(sport, seasons)
    meta = _load_game_meta(seasons)
    feat = _team_schedule_features(meta)
    rows, cov = [], Counter()
    for season, triads in triads_by_season.items():
        for t in triads:
            gpk = t.game_pk
            if gpk is None or int(gpk) not in meta:
                cov["no_meta"] += 1
                continue
            gpk = int(gpk)
            m = meta[gpk]
            hs, as_ = m["home_score"], m["away_score"]
            if hs is None or as_ is None:
                cov["no_score"] += 1
                continue
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
            dog_is_home = not fav_is_home
            winner_home = hs > as_
            fav_tid = str(m["home_team_id"] if fav_is_home else m["away_team_id"])
            dog_tid = str(m["away_team_id"] if fav_is_home else m["home_team_id"])
            ff = feat.get(gpk, {}).get(fav_tid, {})
            df = feat.get(gpk, {}).get(dog_tid, {})
            tags = {
                "ump": m["hp_umpire_name"] or "unknown",
                "dh": "DH" if (ff.get("dh") or df.get("dh")) else "single",
                "fav_rest": _rest_bucket(ff.get("rest")),
                "dog_rest": _rest_bucket(df.get("rest")),
                "fav_travel": "fav_traveled" if ff.get("traveled") else "fav_settled",
                "dog_travel": "dog_traveled" if df.get("traveled") else "dog_settled",
            }
            base = {}
            dog_ml = t.ml_home if dog_is_home else t.ml_away
            base["dog_ml"] = (dog_ml, "win" if winner_home == dog_is_home else "loss")
            dog_point = ((t.rl_home_point or 0.0) if dog_is_home
                         else -(t.rl_home_point or 0.0))
            if abs(dog_point - 1.5) < 0.01:
                dog_rl = t.rl_home if dog_is_home else t.rl_away
                base["dog_rl"] = (dog_rl, _grade_runline(dog_is_home, 1.5, hs, as_))
            if t.total_line is not None:
                base["total_over"] = (t.total_over,
                                      r2_grade.grade_over_under(hs + as_, t.total_line, "OVER"))
                base["total_under"] = (t.total_under,
                                       r2_grade.grade_over_under(hs + as_, t.total_line, "UNDER"))
            emitted = False
            for bet, (price, res) in base.items():
                p = r2_grade.profit(price, res) if res else None
                if p is None:
                    continue
                rows.append({"season": str(season), "bet": bet, "result": res,
                             "profit": p, **tags})
                emitted = True
            if emitted:
                cov["graded_games"] += 1
    return rows, cov


def _repl_slice(rows, keyfn, label, min_n, top=None):
    """Print ROI by factor bucket, gated to n>=min_n, with a per-season sign string and
    a ★ on cells that are +ROI AND positive every season (>=2 seasons) — the honesty
    gate. ``top`` shows only the best+worst N cells (for high-cardinality factors)."""
    seasons = sorted({r["season"] for r in rows})
    cells = r2_grade.by_key(rows, keyfn)
    items = []
    for k, sm in cells.items():
        if sm.n < min_n:
            continue
        signs = []
        for s in seasons:
            ss = r2_grade.summarize([r for r in rows if keyfn(r) == k and r["season"] == s])
            if ss.n >= max(20, min_n // 3):
                signs.append("+" if ss.roi > 0 else "-")
        items.append((k, sm, "".join(signs) or "n/a"))
    items.sort(key=lambda x: x[1].roi, reverse=True)
    show = items
    if top and len(items) > 2 * top:
        show = items[:top] + items[-top:]
    print(f"  by {label} (n>={min_n:,}):")
    if not show:
        print("    (no cells meet n)\n")
        return
    for k, sm, signs in show:
        # ★ = +ROI AND positive every season (>=2) AND pooled |t|>=1.5 (magnitude gate,
        # not just sign — sign-consistency alone flags tiny t~0.5 cells as false leads).
        star = (" ★" if (sm.roi > 0 and set(signs) == {"+"} and len(signs) >= 2
                         and sm.t_stat >= 1.5) else "")
        print(f"    {str(k)[:30]:<30} n={sm.n:>5,} ROI={sm.roi:+.2%} "
              f"t={sm.t_stat:+.2f} [{signs}]{star}")
    print("")


def _report_upset_datamine(rows, cov, min_n=60):
    print("=" * 78)
    print("  UPSET DATAMINE — realized ROI of base bets, stratified by market-blind factor")
    print(f"  graded_games={cov.get('graded_games',0):,}  (no_meta={cov.get('no_meta',0):,} "
          f"no_score={cov.get('no_score',0):,} pickem={cov.get('pickem',0):,})")
    print("  ★ = +ROI AND positive every season (>=2) at n>=min — a candidate signal.")
    print("=" * 78)
    # Which factors to cross with which base bets (the plausible story per factor).
    plan = [
        ("dog_ml",      [("dh", "doubleheader"), ("dog_rest", "underdog days-rest"),
                         ("fav_rest", "favorite days-rest"), ("dog_travel", "underdog travel"),
                         ("fav_travel", "favorite travel")]),
        ("dog_rl",      [("dh", "doubleheader"), ("dog_rest", "underdog days-rest"),
                         ("fav_rest", "favorite days-rest")]),
        ("total_over",  [("ump", "home-plate umpire"), ("dh", "doubleheader")]),
        ("total_under", [("ump", "home-plate umpire"), ("dh", "doubleheader")]),
    ]
    for bet, factors in plan:
        br = [r for r in rows if r["bet"] == bet]
        pooled = r2_grade.summarize(br)
        print(f"\n  ── {bet.upper()}  (baseline: n={pooled.n:,} ROI={pooled.roi:+.2%} "
              f"t={pooled.t_stat:+.2f}) " + "─" * 20)
        for fkey, flabel in factors:
            top = 5 if fkey == "ump" else None      # umpire is high-cardinality
            _repl_slice(br, lambda r, _k=fkey: r[_k], flabel, min_n, top=top)
    print("=" * 78)
    print("  READ: a ★ cell (or a clearly -ROI replicating cell) is the deliverable — a")
    print("  factor the market misprices in a bettable (or disqualifiable) direction.")
    print("  Everything else = the market is efficient on that factor. Verify a ★ with a")
    print("  finer cut before trusting it (high-cardinality umpire cells overfit easily).")
    print("=" * 78)


def _late_gap_bucket(g):
    """DK full-game total MINUS Pinnacle first-5 total = DK's implied innings-6-9 runs
    (normal ~4.6). Low = DK implies few late runs (full total maybe too low → over)."""
    if g is None:
        return "unk"
    if g < 3.5:
        return "a <3.5 (few late)"
    if g < 4.0:
        return "b 3.5-4.0"
    if g < 4.5:
        return "c 4.0-4.5"
    if g < 5.0:
        return "d 4.5-5.0"
    return "e >=5.0 (many late)"


def scenario_f5_decomp(sport, seasons, lead_lo=12.0, lead_hi=24.0):
    """Doug's period decomposition: DK full-game total MINUS Pinnacle FIRST-5 total = DK's
    implied runs in innings 6-9. Bet at DK's full total, grade BOTH over & under, and
    bucket ROI by that implied late-innings gap. Hypothesis: anomalously LOW gap (DK
    implies few late runs → full total too low) → OVER wins; high gap → UNDER. Also
    reports the DK-full vs Pin-FULL gap (integrity: ~0 => DK not line-offset). Everything
    kind-filtered (no full/F5 contamination) — the fix for the under_dkpin mixing bug."""
    import r2_data
    meta = _load_game_meta(seasons)
    rows, cov, integ = [], Counter(), []
    for s in seasons:
        dk = r2_data._read_team_market_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31", bookmaker="draftkings")
        pin = r2_data._read_team_market_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31", bookmaker="pinnacle")
        dk_full = _event_snapshots([r for r in dk if r.get("kind") == "team"])
        pin_f5 = _event_snapshots([r for r in pin if r.get("kind") == "first_five"])
        pin_full = _event_snapshots([r for r in pin if r.get("kind") == "team"])
        for eid, snaps in dk_full.items():
            dpick = next(((ld, sd) for ld, sd in reversed(snaps)
                          if _mkt_ok("over", sd) and lead_lo <= ld < lead_hi), None)
            if dpick is None:
                cov["no_dk_total"] += 1
                continue
            dsd = dpick[1]
            gpk = dsd.get("game_pk")
            if gpk is None or int(gpk) not in meta:
                cov["no_meta"] += 1
                continue
            m = meta[int(gpk)]
            hs, as_ = m["home_score"], m["away_score"]
            if hs is None or as_ is None:
                cov["no_score"] += 1
                continue
            pu = pin_full.get(eid)
            pup = (next(((ld, sd) for ld, sd in reversed(pu) if _mkt_ok("over", sd)), None)
                   if pu else None)
            if pup is not None:
                integ.append(dsd["total_line"] - pup[1]["total_line"])
            pf = pin_f5.get(eid)
            pfp = (next(((ld, sd) for ld, sd in reversed(pf) if _mkt_ok("over", sd)), None)
                   if pf else None)
            if pfp is None:
                cov["no_pin_f5"] += 1
                continue
            gap = dsd["total_line"] - pfp[1]["total_line"]
            b = _late_gap_bucket(gap)
            for bet, price, side in (("over", dsd["over"], "OVER"),
                                     ("under", dsd["under"], "UNDER")):
                res = r2_grade.grade_over_under(hs + as_, dsd["total_line"], side)
                p = r2_grade.profit(price, res) if res else None
                if p is None:
                    continue
                rows.append({"season": str(s), "bet": bet, "result": res, "profit": p,
                             "gap": gap, "gap_bucket": b,
                             "line_bucket": _total_line_bucket(dsd["total_line"])})
            cov["graded"] += 1
    return rows, cov, integ


def _gap_x_line_under(rows, min_n=30):
    """Tautology check: UNDER ROI by gap WITHIN each fixed total-line bucket. If ROI
    rises with the gap down a line COLUMN, the gap is real shape info (not line-height)."""
    gaps = ["a <3.5 (few late)", "b 3.5-4.0", "c 4.0-4.5", "d 4.5-5.0",
            "e >=5.0 (many late)"]
    lines = ["a <=7.5", "b 8-8.5", "c 9-9.5", "d >=10"]
    under = [r for r in rows if r["bet"] == "under"]
    print("  TAUTOLOGY CHECK — UNDER ROI by gap WITHIN each line bucket (n>=%d):" % min_n)
    for g in gaps:
        parts = []
        for ln in lines:
            sm = r2_grade.summarize([r for r in under
                                     if r["gap_bucket"] == g and r["line_bucket"] == ln])
            if sm.n >= min_n:
                parts.append(f"{ln[2:]}={sm.roi:+.1%}(n={sm.n})")
        if parts:
            print(f"    gap {g[2:]:<16} | " + "  ".join(parts))
    print("  Read DOWN a line column: ROI rising with gap => real shape edge; flat => the")
    print("  gap was just proxying the line.\n")


def _report_f5_decomp(rows, cov, integ):
    import statistics
    print("=" * 78)
    print("  F5 DECOMPOSITION — DK full total − Pinnacle FIRST-5 total = DK's implied")
    print("  innings-6-9 runs. Over/under ROI (bet at DK's full total) by that gap.")
    print(f"  graded={cov.get('graded',0):,}  (no_pin_f5={cov.get('no_pin_f5',0):,} "
          f"no_dk_total={cov.get('no_dk_total',0):,})")
    if integ:
        print(f"  INTEGRITY DK_full − Pin_FULL: mean {sum(integ)/len(integ):+.3f}  "
              f"median {statistics.median(integ):+.2f}  n={len(integ):,}  (want ~0)")
    over_gaps = [r["gap"] for r in rows if r["bet"] == "over"]
    if over_gaps:
        print(f"  implied late gap (DK_full − Pin_F5): mean {sum(over_gaps)/len(over_gaps):+.2f}"
              f"  median {statistics.median(over_gaps):+.2f}")
    print("=" * 78)
    for bet in ("over", "under"):
        br = [r for r in rows if r["bet"] == bet]
        pooled = r2_grade.summarize(br)
        print(f"\n  ── {bet.upper()} @ DK full total  (pooled n={pooled.n:,} "
              f"ROI={pooled.roi:+.2%} t={pooled.t_stat:+.2f}) " + "─" * 12)
        _repl_slice(br, lambda r: r["gap_bucket"], "implied late-innings gap", 50)
    print()
    _gap_x_line_under(rows)
    print("  READ (Doug's hypothesis): if DK misprices game SHAPE, OVER ROI should be")
    print("  highest in the LOW-gap buckets (DK implied few late runs → full total too")
    print("  low) and UNDER highest in the HIGH-gap buckets. Flat across gaps => DK's")
    print("  full-vs-F5 split is coherent (no shape edge).")
    print("=" * 78)


def _diff_bucket(d):
    if d is None:
        return "unk"
    if d >= 1.0:
        return "a DK >=+1.0"
    if d >= 0.5:
        return "b DK +0.5"
    if d > -0.5:
        return "c equal (0)"
    if d > -1.0:
        return "d DK -0.5"
    return "e DK <=-1.0"


def scenario_under_dkpin(sport, seasons, lead_lo=12.0, lead_hi=24.0):
    """DK-vs-PINNACLE total integrity check for the flat-under signal. Per event, compare
    our captured DK total (12-24h window = the bet line) to Pinnacle's total (an
    INDEPENDENT capture — Pinnacle is NOT sharper, so this is a line-level / capture
    check, not beat-the-sharp). Reports the (DK−Pin) gap distribution + the DK-under ROI
    bucketed by that gap. mean gap ~0 & flat ROI => our capture is fine + it's a both-book
    regime miss (real, likely decays); DK systematically higher & unders concentrate in
    DK>Pin => DK over-shades totals (line-level edge); wild gaps => capture artifact."""
    import r2_data
    meta = _load_game_meta(seasons)
    rows, cov = [], Counter()
    for s in seasons:
        dk = r2_data._read_team_market_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31", bookmaker="draftkings")
        pin = r2_data._read_team_market_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31", bookmaker="pinnacle")
        dk = [r for r in dk if r.get("kind") == "team"]      # drop F5 (contamination)
        pin = [r for r in pin if r.get("kind") == "team"]
        dk_snaps, pin_snaps = _event_snapshots(dk), _event_snapshots(pin)
        for eid, snaps in dk_snaps.items():
            dpick = next(((ld, sd) for ld, sd in reversed(snaps)
                          if _mkt_ok("over", sd) and lead_lo <= ld < lead_hi), None)
            if dpick is None:
                cov["no_dk_total"] += 1
                continue
            dsd = dpick[1]
            psnaps = pin_snaps.get(eid)
            ppick = (next(((ld, sd) for ld, sd in reversed(psnaps) if _mkt_ok("over", sd)),
                          None) if psnaps else None)
            if ppick is None:
                cov["no_pin_total"] += 1
                continue
            psd = ppick[1]
            gpk = dsd.get("game_pk")
            if gpk is None or int(gpk) not in meta:
                cov["no_meta"] += 1
                continue
            m = meta[int(gpk)]
            hs, as_ = m["home_score"], m["away_score"]
            if hs is None or as_ is None:
                cov["no_score"] += 1
                continue
            diff = dsd["total_line"] - psd["total_line"]
            res = r2_grade.grade_over_under(hs + as_, dsd["total_line"], "UNDER")
            p = r2_grade.profit(dsd["under"], res) if res else None
            if p is None:
                cov["ungradable"] += 1
                continue
            rows.append({"season": str(s), "result": res, "profit": p, "diff": diff,
                         "diff_bucket": _diff_bucket(diff),
                         "dk_line": dsd["total_line"], "pin_line": psd["total_line"]})
            cov["matched"] += 1
    return rows, cov


def _report_under_dkpin(rows, cov):
    import statistics
    print("=" * 78)
    print("  DK-vs-PINNACLE TOTAL — integrity check for the flat-under signal (Pinnacle")
    print("  = independent capture, NOT a sharper reference). Is DK's total offset?")
    print(f"  matched={cov.get('matched',0):,}  (no_dk_total={cov.get('no_dk_total',0):,} "
          f"no_pin_total={cov.get('no_pin_total',0):,})")
    print("=" * 78)
    diffs = [r["diff"] for r in rows]
    if diffs:
        pos = sum(1 for d in diffs if d > 0) / len(diffs)
        eq = sum(1 for d in diffs if d == 0) / len(diffs)
        neg = sum(1 for d in diffs if d < 0) / len(diffs)
        print(f"  DK−Pin total gap:  mean {sum(diffs)/len(diffs):+.3f}  "
              f"median {statistics.median(diffs):+.2f}  |  "
              f"DK>Pin {pos:.0%}   equal {eq:.0%}   DK<Pin {neg:.0%}")
    pooled = r2_grade.summarize(rows)
    print(f"  DK-under (matched set): n={pooled.n:,} ROI={pooled.roi:+.2%} "
          f"t={pooled.t_stat:+.2f}\n")
    _repl_slice(rows, lambda r: r["diff_bucket"], "DK-minus-Pinnacle total gap", 50)
    print("  READ: mean~0 & flat ROI across gap buckets => capture is fine, edge is a")
    print("  BOTH-BOOK regime miss (real in-sample, likely decays; not a capture bug).")
    print("  DK systematically higher & under ROI concentrated in 'DK >=+0.5' => DK")
    print("  over-shades totals (a real line-level under edge). Wild gaps => capture bug.")
    print("=" * 78)


def _lead_bucket(h):
    if h is None:
        return "unk"
    if h < 1:
        return "a <1h"
    if h < 3:
        return "b 1-3h"
    if h < 6:
        return "c 3-6h"
    if h < 12:
        return "d 6-12h"
    if h < 24:
        return "e 12-24h"
    return "f >=24h"


_LEAD_ORDER = ["a <1h", "b 1-3h", "c 3-6h", "d 6-12h", "e 12-24h", "f >=24h", "unk"]


def scenario_under_timing(sport, seasons):
    """TIMING-CONTROL the flat-UNDER signal from upset_datamine. For each event, take the
    CLOSE total snapshot (latest pre-commence capture that carries a total) AND its lead
    hours, grade over/under at THAT line vs the final, and bucket ROI by lead time. Reads
    RAW snapshots (not the triad) so the per-event capture lead is known. READ: if the
    under edge SHRINKS toward the <1h bucket (vanishes near-close), it's a stale-high-line
    ARTIFACT — our captured total is earlier/higher than the true close; if it HOLDS at
    <1h, it's a real closing-line edge worth a paid true-close confirmation."""
    import r2_data
    scores = r2_data.build_team_scores_index(seasons)
    rows, cov = [], Counter()
    for s in seasons:
        team_rows = r2_data._read_team_market_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31", bookmaker="draftkings")
        team_rows = [r for r in team_rows if r.get("kind") == "team"]  # drop F5
        for eid, snaps in _event_snapshots(team_rows).items():
            cov["events"] += 1
            close = next(((ld, sd) for ld, sd in reversed(snaps) if _mkt_ok("over", sd)),
                         None)
            if close is None:
                cov["no_total"] += 1
                continue
            lead, sd = close
            gpk = sd.get("game_pk")
            if gpk is None or int(gpk) not in scores:
                cov["no_score"] += 1
                continue
            hs, as_ = scores[int(gpk)]
            lb = _lead_bucket(lead)
            for bet, price, side in (("total_over", sd["over"], "OVER"),
                                     ("total_under", sd["under"], "UNDER")):
                res = r2_grade.grade_over_under(hs + as_, sd["total_line"], side)
                p = r2_grade.profit(price, res) if res else None
                if p is None:
                    continue
                rows.append({"season": str(s), "bet": bet, "result": res, "profit": p,
                             "lead_bucket": lb, "lead": lead})
            cov["graded"] += 1
    return rows, cov


def _total_line_bucket(x):
    if x is None:
        return "unk"
    if x <= 7.5:
        return "a <=7.5"
    if x <= 8.5:
        return "b 8-8.5"
    if x <= 9.5:
        return "c 9-9.5"
    return "d >=10"


def _load_weather(seasons):
    """{(venue_id, official_date): temp_f} from weather_game (first-pitch-hour Visual
    Crossing temps). Empty/partial if the weather backfill hasn't run for a season."""
    import db_store
    import mlb_warehouse as wh
    from sqlalchemy import or_ as _or
    from sqlalchemy import select as _sel
    db_store.promote_secrets_from_toml()
    w = wh.weather_game
    out = {}
    with db_store.get_engine().connect() as conn:
        rows = conn.execute(_sel(w.c.venue_id, w.c.weather_date, w.c.temp_f).where(
            _or(*[w.c.weather_date.like(f"{s}-%") for s in seasons]))).fetchall()
    for r in rows:
        m = r._mapping
        if m["temp_f"] is not None:
            out[(str(m["venue_id"]), m["weather_date"])] = m["temp_f"]
    return out


def _temp_bucket(t):
    if t is None:
        return "z unk"
    if t < 55:
        return "a <55F cold"
    if t < 65:
        return "b 55-65 cool"
    if t < 75:
        return "c 65-75 mild"
    if t < 85:
        return "d 75-85 warm"
    return "e >=85 hot"


def scenario_under_stress(sport, seasons, lead_lo=12.0, lead_hi=24.0):
    """FREE offline stress-test of the flat-UNDER signal, on the bettable 12-24h window
    (the strongest OBSERVABLE bucket from under_timing). Breaks under ROI down by total-
    line value, favorite strength, month, and season (magnitude + t) to judge whether the
    +9.85% is BROAD + stable (a real market over-lean, bettable) or CONCENTRATED in a
    suspicious subset (artifact/overfit). Uses the close total snapshot within
    [lead_lo, lead_hi); scores + month from the warehouse game meta."""
    import r2_data
    meta = _load_game_meta(seasons)
    weather = _load_weather(seasons)
    rows, cov = [], Counter()
    for s in seasons:
        team_rows = r2_data._read_team_market_lines(
            sport, date_from=f"{s}-01-01", date_to=f"{s}-12-31", bookmaker="draftkings")
        team_rows = [r for r in team_rows if r.get("kind") == "team"]  # drop F5
        for eid, snaps in _event_snapshots(team_rows).items():
            pick = next(((ld, sd) for ld, sd in reversed(snaps)
                         if _mkt_ok("over", sd) and lead_lo <= ld < lead_hi), None)
            if pick is None:
                cov["no_total_in_window"] += 1
                continue
            lead, sd = pick
            gpk = sd.get("game_pk")
            if gpk is None or int(gpk) not in meta:
                cov["no_meta"] += 1
                continue
            m = meta[int(gpk)]
            hs, as_ = m["home_score"], m["away_score"]
            if hs is None or as_ is None:
                cov["no_score"] += 1
                continue
            res = r2_grade.grade_over_under(hs + as_, sd["total_line"], "UNDER")
            p = r2_grade.profit(sd["under"], res) if res else None
            if p is None:
                cov["ungradable"] += 1
                continue
            try:
                imp_h = american_to_implied_prob(int(sd["ml_home"]))
                imp_a = american_to_implied_prob(int(sd["ml_away"]))
                fav = _fav_bucket(max(imp_h, imp_a))
            except (TypeError, ValueError):
                fav = "unk"
            temp = weather.get((str(m["venue_id"]), m["official_date"]))
            rows.append({"season": str(s), "result": res, "profit": p,
                         "line_bucket": _total_line_bucket(sd["total_line"]),
                         "month": (m["official_date"] or "")[5:7] or "unk",
                         "fav_bucket": fav, "temp_bucket": _temp_bucket(temp)})
            cov["graded"] += 1
            if temp is not None:
                cov["with_temp"] += 1
    return rows, cov


def _report_under_stress(rows, cov, min_n=80):
    pooled = r2_grade.summarize(rows)
    print("=" * 78)
    print("  UNDER STRESS-TEST — flat-UNDER ROI on the 12-24h window, broken down to test")
    print("  breadth vs concentration. Broad+ across cells => real over-lean; one cell")
    print("  carrying it => suspect (artifact/overfit).")
    print(f"  POOLED: n={pooled.n:,} ROI={pooled.roi:+.2%} hit={pooled.hit_rate:.1%} "
          f"t={pooled.t_stat:+.2f}   (temp coverage={cov.get('with_temp',0):,}/{pooled.n:,})")
    print("=" * 78)
    # Mechanism test FIRST: is it literally cold-weather run-environment?
    _repl_slice(rows, lambda r: r["temp_bucket"], "game-time temperature (MECHANISM)", min_n)
    _repl_slice(rows, lambda r: r["line_bucket"], "total-line value", min_n)
    _repl_slice(rows, lambda r: r["fav_bucket"], "favorite strength", min_n)
    _repl_slice(rows, lambda r: r["month"], "month", min_n)
    _repl_slice(rows, lambda r: r["season"], "season", 30)
    print("  READ: want +ROI in MOST line/fav/month cells AND every season with similar")
    print("  magnitude. If it's one line-bucket or one month, it's a narrow effect (or")
    print("  overfit), not a broad over-lean — treat with more suspicion before any spend.")
    print("=" * 78)


def _report_under_timing(rows, cov):
    print("=" * 78)
    print("  UNDER-TIMING CONTROL — over/under ROI by how close-to-first-pitch the total")
    print("  snapshot was captured. Under edge fading toward '<1h' = STALE-LINE ARTIFACT.")
    print(f"  events={cov.get('events',0):,} graded={cov.get('graded',0):,} "
          f"(no_total={cov.get('no_total',0):,} no_score={cov.get('no_score',0):,})")
    print("=" * 78)
    seasons = sorted({r["season"] for r in rows})
    for bet in ("total_under", "total_over"):
        br = [r for r in rows if r["bet"] == bet]
        pooled = r2_grade.summarize(br)
        print(f"\n  ── {bet.upper()}  (pooled: n={pooled.n:,} ROI={pooled.roi:+.2%} "
              f"t={pooled.t_stat:+.2f}) " + "─" * 18)
        cells = r2_grade.by_key(br, lambda r: r["lead_bucket"])
        print(f"    {'lead bucket':<12} {'n':>6} {'ROI':>9} {'t':>7}  per-season")
        for b in _LEAD_ORDER:
            sm = cells.get(b)
            if not sm or sm.n < 30:
                continue
            signs = []
            for s in seasons:
                ss = r2_grade.summarize(
                    [r for r in br if r["lead_bucket"] == b and r["season"] == s])
                if ss.n >= 20:
                    signs.append("+" if ss.roi > 0 else "-")
            print(f"    {b:<12} {sm.n:>6,} {sm.roi:>+8.2%} {sm.t_stat:>+7.2f}  "
                  f"[{''.join(signs) or 'n/a'}]")
    print("\n  READ: compare the '<1h' (near-close) row to the longer-lead rows. Under ROI")
    print("  high at long leads but ~0 at <1h => the +5% was our stale-high line, NOT a")
    print("  real edge. Under ROI still clearly + at <1h => real; worth a true-close pull.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Rule-based scenario backtests (DK, 2024-26).")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--scenario", default="all",
                    choices=["all", "under_hits", "home_runline", "dog_runline",
                             "fav_combo", "er_ml", "team_variance", "prop_roi",
                             "line_timing", "coherence_stable_sp", "upset_datamine",
                             "under_timing", "under_stress", "under_dkpin",
                             "f5_decomp"])
    ap.add_argument("--datamine-min-n", type=int, default=60,
                    help="Min bets for an upset_datamine factor cell to be shown.")
    ap.add_argument("--coh-fav-min", type=float, default=0.60,
                    help="Coherence band lower ML-implied bound (coherence_stable_sp).")
    ap.add_argument("--coh-fav-max", type=float, default=0.70,
                    help="Coherence band upper ML-implied bound (coherence_stable_sp).")
    ap.add_argument("--prop-roi-min-n", type=int, default=100,
                    help="Min pooled bets for a (prop,line,side) cell to be a candidate.")
    ap.add_argument("--cv-half-life", type=float, default=5.0,
                    help="Half-life (games) for the SP ER-CV in team_variance; 5 "
                         "matches the live earned_runs cv_floor signal. 0 = equal weight.")
    ap.add_argument("--refresh", action="store_true",
                    help="re-read the warehouse (else use the pickle cache)")
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
        # Knife-edge vs plateau: finer 2.5% bins around the 65-70% sweet spot.
        _print_slice(rows, lambda r: _fav_bucket_fine(r["fav_imp"]),
                     "favorite strength (fine 2.5% bins)")
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
    if want in ("all", "team_variance"):
        rows, cov = scenario_team_variance(blob, half_life=args.cv_half_life)
        print(f"  team_variance coverage: games={cov.get('games',0):,} "
              f"total_over={cov.get('total_over_graded',0):,} "
              f"dog_ml={cov.get('dog_ml_graded',0):,} dog_rl={cov.get('dog_rl_graded',0):,} "
              f"(skip_no_cv={cov.get('skip_no_cv',0):,} "
              f"skip_no_fav_cv={cov.get('skip_no_fav_cv',0):,})  "
              f"cv_half_life={args.cv_half_life}\n")
        for bet, label in (("total_over", "TEAM TOTAL OVER"),
                           ("dog_ml", "UNDERDOG ML"),
                           ("dog_rl", "UNDERDOG +1.5")):
            br = [r for r in rows if r["bet"] == bet]
            cond = "max SP ER-CV" if bet == "total_over" else "favorite's SP ER-CV"
            _report(f"SP volatility → {label}  (conditioned on {cond})", br)
            _print_slice(br, lambda r: r["cv_bucket"], "SP ER-CV bucket")
            hi = [r for r in br if r["cv_bucket"] == "c >=1.3 (high)"]
            if hi:
                _print_slice(hi, lambda r: r["season"], "high-CV (>=1.3) x season")
    if want in ("all", "coherence_stable_sp"):
        rows, cov = scenario_coherence_stable_sp(
            blob, half_life=args.cv_half_life,
            fav_min=args.coh_fav_min, fav_max=args.coh_fav_max)
        _report_coherence_stable_sp(rows, cov, args.coh_fav_min, args.coh_fav_max,
                                    args.cv_half_life)
    if want in ("all", "prop_roi"):
        rows, cov = scenario_prop_roi(args.sport, seasons)
        _report_prop_roi(rows, cov, min_n=args.prop_roi_min_n)
    if want == "line_timing":          # explicit only, not in "all"
        scenario_line_timing(args.sport, seasons)                    # coverage probe
        rows, cov = scenario_line_timing_study(args.sport, seasons)  # early-vs-close study
        _report_line_timing(rows, cov)
    if want == "upset_datamine":       # explicit only, not in "all"
        rows, cov = scenario_upset_datamine(args.sport, seasons)
        _report_upset_datamine(rows, cov, min_n=args.datamine_min_n)
    if want == "under_timing":         # explicit only, not in "all"
        rows, cov = scenario_under_timing(args.sport, seasons)
        _report_under_timing(rows, cov)
    if want == "under_stress":         # explicit only, not in "all"
        rows, cov = scenario_under_stress(args.sport, seasons)
        _report_under_stress(rows, cov)
    if want == "under_dkpin":          # explicit only, not in "all"
        rows, cov = scenario_under_dkpin(args.sport, seasons)
        _report_under_dkpin(rows, cov)
    if want == "f5_decomp":            # explicit only, not in "all"
        rows, cov, integ = scenario_f5_decomp(args.sport, seasons)
        _report_f5_decomp(rows, cov, integ)


if __name__ == "__main__":
    main()
