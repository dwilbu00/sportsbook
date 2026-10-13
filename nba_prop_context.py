"""nba_prop_context.py — NBA player-prop PACE + DEFENSE context adjustment (one implementation).

Shared by the offline sweep (backtest.run_player_props_backtest), the real-line method selector
(book_line_calibration.project_and_empirical) and live pricing (props.py), so fit == serve.
Validated in nba_prop_context_study.py (2023-26, 82,451 player-games/prop): every prop improves
at every strength, the sign-check slopes are positive and ~1 (right direction, right size), and
in the top-decile adjustment games MAE improves 1.1-1.8%.

Normalize-then-reapply on each past value before the projection's averaging:

    adjusted_i = value_i x (poss_hat_t / poss_i)^s_pace x (def_t / def_i)^s_def

  poss_i      ACTUAL possessions of past game i (team box)
  poss_hat_t  PROJECTED possessions for the game being priced = pace_team * pace_opp / lg_pace
  def_x       the opponent's ALLOWED per 100 possessions for the prop's stat(s) / league (summed
              for combo props), every rating PRE-GAME (nba_ratings, team-model knobs)

Contexts are keyed (game_date, team_id): a team plays once a day, and every pipeline has both.
Strengths live per prop in calibration as ``nba_pace_strength`` / ``nba_def_strength`` (0 = off;
an off prop is byte-identical to the unadjusted projection).
"""
from __future__ import annotations

import time

import nba_ratings as R

PACE_KEY = "nba_pace_strength"
DEF_KEY = "nba_def_strength"


def ratings_params():
    """The shipped team-model ratings knobs (one source of truth), else nba_ratings defaults."""
    try:
        import nba_team_model
        m = nba_team_model.model()
        if m:
            return dict(m["params"])
    except Exception:
        pass
    return dict(R.DEFAULT_PARAMS)


def stat_cols(prop_key):
    """Canonical stat column(s) for an NBA prop (tuple), or None if unmapped."""
    try:
        import nba_stats_source as src
    except Exception:
        return None
    cols = src.PROP_STAT_COLS.get(prop_key)
    if cols is None:
        return None
    return cols if isinstance(cols, tuple) else (cols,)


def _ctx(mine, opp, poss=None):
    """One game context from the two teams' PRE-GAME ratings (``poss`` = actual, if played)."""
    return {"poss": poss,
            "poss_hat": mine["pace"] * opp["pace"] / mine["lg_pace"],
            "allowed": {st: (opp["allowed100"].get(st), opp["lg_allowed100"].get(st))
                        for st in R.ALLOWED_STATS}}


def build_index(rows, params=None):
    """{(game_date, team_id): context} for every paired team-game in ``rows``
    (nba_warehouse.team_games), from a walk-forward nba_ratings pass — each context uses only
    games strictly before its date."""
    params = params or ratings_params()
    pre = R.pregame_ratings(rows, params)
    out = {}
    for date, _season, gid, a, b in R._pair_games(rows):
        for r, o in ((a, b), (b, a)):
            mine, opp = pre.get((gid, r["team_id"])), pre.get((gid, o["team_id"]))
            if mine and opp:
                out[(date, int(r["team_id"]))] = _ctx(mine, opp, float(r["possessions"]))
    return out


_INDEX_CACHE = {}                                       # (seasons, params) -> index


def index_for_seasons(seasons, params=None):
    """Cached build_index over ``seasons`` plus two warm-up seasons (the same window the team
    model serves), for the offline pipelines. {} when no team games are available."""
    import nba_warehouse as nw
    params = params or ratings_params()
    ss = sorted({int(s) for s in seasons if s is not None})
    if not ss:
        return {}
    key = (tuple(ss), tuple(sorted(params.items())))
    if key not in _INDEX_CACHE:
        rows = nw.team_games(list(range(ss[0] - 2, ss[-1] + 1)))
        _INDEX_CACHE[key] = build_index(rows, params) if rows else {}
    return _INDEX_CACHE[key]


def def_rel(ctx, cols):
    """The opponent's allowed/100 for ``cols`` (summed) relative to the league, or None."""
    if not ctx:
        return None
    al = lg = 0.0
    for c in cols:
        a, l_ = ctx["allowed"].get(c) or (None, None)
        if a is None or l_ is None:
            return None
        al += a
        lg += l_
    return (al / lg) if al > 0 and lg > 0 else None


def adjust_values(values, past_ctxs, up_ctx, cols, s_pace, s_def):
    """Context-adjusted copies of ``values`` (index-aligned with ``past_ctxs``). A missing
    context on either side leaves that factor at 1.0 (fail open); both strengths 0 returns the
    input list unchanged."""
    if (not s_pace and not s_def) or not up_ctx or not cols:
        return values
    d_t = def_rel(up_ctx, cols) if s_def else None
    out = []
    for v, c in zip(values, past_ctxs):
        f = 1.0
        if c:
            if s_pace and c.get("poss") and up_ctx.get("poss_hat"):
                f *= (up_ctx["poss_hat"] / c["poss"]) ** s_pace
            if s_def and d_t:
                d_i = def_rel(c, cols)
                if d_i:
                    f *= (d_t / d_i) ** s_def
        out.append(v * f if v is not None else v)
    return out


def adjust_games(values, games, test_date, team_id, index, cols, s_pace, s_def,
                 team_key="team_id", date_key="game_date"):
    """adjust_values for the offline pipelines: past contexts looked up from each game dict's
    (date, team) and the priced game's context = the index entry for (test_date, team_id)."""
    if (not s_pace and not s_def) or not index or team_id is None:
        return values
    up = index.get((str(test_date)[:10], int(team_id)))
    if not up:
        return values
    past = [index.get((str(g.get(date_key) or "")[:10], int(g[team_key])))
            if g.get(team_key) is not None else None for g in games]
    return adjust_values(values, past, up, cols, s_pace, s_def)


# ───────────────────────────────────────────────────────────────────────── live
_LIVE = {}                                             # (date, params) -> (ts, index, ratings)
_LIVE_TTL = 1800


def live_context(date):
    """(index of past-game contexts, {team_id: rating as of ``date``}) for live pricing — one
    nba_ratings walk over the current + two prior seasons, cached per date."""
    import nba_stats_source as src
    import nba_warehouse as nw
    params = ratings_params()
    d = str(date)[:10]
    key = (d, tuple(sorted(params.items())))
    hit = _LIVE.get(key)
    if hit and (time.time() - hit[0]) < _LIVE_TTL:
        return hit[1], hit[2]
    season = src.end_year_for_date(d)
    rows = [r for r in nw.team_games([season - 2, season - 1, season])
            if str(r.get("game_date") or "")[:10] < d]
    index = build_index(rows, params) if rows else {}
    ratings = R.ratings_as_of(rows, d, season, params=params) if rows else {}
    _LIVE[key] = (time.time(), index, ratings)
    return index, ratings


def live_adjust(values, game_dates, team_ids, player_team_id, home_team, away_team, date,
                prop_key, s_pace, s_def):
    """Live-pricing adjustment: resolve the upcoming opponent from the odds-feed team names
    (the event team that isn't the player's), build tonight's context from as-of ratings, and
    adjust each past value by its own game's context. Fails open (returns ``values``)."""
    if not s_pace and not s_def:
        return values
    try:
        import nba_warehouse as nw
        cols = stat_cols(prop_key)
        if not cols or player_team_id is None:
            return values
        hid, aid = nw.resolve_team_id(home_team), nw.resolve_team_id(away_team)
        tid = int(player_team_id)
        opp = aid if tid == hid else hid if tid == aid else None
        if opp is None:
            return values
        index, ratings = live_context(date)
        mine, theirs = ratings.get(tid), ratings.get(opp)
        if not mine or not theirs:
            return values
        up = _ctx(mine, theirs)
        past = [index.get((str(d or "")[:10], int(t))) if (d and t is not None) else None
                for d, t in zip(game_dates, team_ids)]
        return adjust_values(values, past, up, cols, s_pace, s_def)
    except Exception:
        return values
