"""nba_ratings.py — as-of NBA team pace / efficiency ratings (one source of truth).

Shared by the NBA team-market model, the rebuilt player-prop pace/defense, and every backtest,
so fit == serve. Input = ``nba_warehouse.team_games`` rows (one per team per game, with
possessions). Output = each team's PRE-GAME ratings, computed only from games strictly before
that date.

Method (all per team, within a season):
  * Exponentially decayed sums by games played (``half_life`` games), so recent form counts
    more: pace = 48·Σposs/Σminutes, ORtg = 100·Σpts/Σposs, DRtg = 100·Σpts_allowed/Σposs.
  * Season start: last season's final (opponent-adjusted) rating, regressed ``regress`` of the
    way to the league mean, entered as ``prior_games`` pseudo-games that decay away as real
    games arrive. No prior season -> league mean (pure shrinkage).
  * Opponent adjustment (``opp_adjust``): ORtg is divided by the average relative DRtg of the
    defenses faced (and DRtg by the offenses faced, pace by the opponents' pace), using each
    opponent's own PRE-GAME rating at the time of the game — a single leakage-safe pass.
  * League averages: season-to-date cumulative, blended with last season's league values as
    ``league_prior_games`` pseudo team-games early on.
  * Same-date games are SIMULTANEOUS: every game on a date is rated before any of them updates
    the state, so a 7pm game never informs a 10pm game.
  * Per-stat ALLOWED per 100 possessions (``ALLOWED_STATS``) for the prop defense factor, and
    rest days since the team's previous game.

The derived possessions run ~1.6% below the official stats.nba.com count (a near-constant
offset, corr 0.995); every consumer uses ratings RELATIVE to the league average, where it cancels.
"""
from __future__ import annotations

import datetime

DEFAULT_PARAMS = {
    "half_life": 20.0,          # games: weight of a game halves every N later games
    "prior_games": 10.0,        # pseudo-games of last season's (regressed) rating at season start
    "regress": 0.33,            # share of last season's deviation from league mean regressed away
    "opp_adjust": True,         # schedule-strength adjustment of pace / ORtg / DRtg
    "league_prior_games": 60.0, # pseudo team-games of last season's league averages
}

# Opponent stats tracked as "allowed per 100 possessions" (canonical warehouse names).
ALLOWED_STATS = ("points", "three_point_field_goals_made", "three_point_field_goals_attempted",
                 "field_goals_made", "field_goals_attempted", "free_throws_made",
                 "free_throws_attempted", "offensive_rebounds", "defensive_rebounds",
                 "rebounds", "assists", "steals", "blocks", "turnovers")

_FALLBACK_LEAGUE = {"pace": 99.0, "rtg": 112.0}   # only when no data at all precedes a game


def _params(params):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return p


def _days_between(a, b):
    try:
        return (datetime.date.fromisoformat(str(b)[:10])
                - datetime.date.fromisoformat(str(a)[:10])).days
    except (TypeError, ValueError):
        return None


class _TeamState:
    """Decayed per-team, per-season sums."""
    __slots__ = ("w", "poss", "mins", "pts", "opp_pts", "allowed", "sos_d", "sos_o",
                 "sos_p", "n")

    def __init__(self):
        self.w = self.poss = self.mins = self.pts = self.opp_pts = 0.0
        self.sos_d = self.sos_o = self.sos_p = 0.0
        self.allowed = {s: 0.0 for s in ALLOWED_STATS}
        self.n = 0                                          # real games this season


class _League:
    """Cumulative per-season league sums (team-game grain)."""
    __slots__ = ("poss", "mins", "pts", "allowed", "n")

    def __init__(self):
        self.poss = self.mins = self.pts = 0.0
        self.allowed = {s: 0.0 for s in ALLOWED_STATS}
        self.n = 0


def _pair_games(rows):
    """[(game_date, season, game_id, rowA, rowB)] for games with exactly two team rows and
    usable possessions/minutes, sorted by (date, game_id)."""
    by = {}
    for r in rows:
        gid = r.get("game_id")
        if gid is not None:
            by.setdefault(str(gid), []).append(r)
    out = []
    for gid, rs in by.items():
        if len(rs) != 2:
            continue
        a, b = rs
        if not a.get("possessions") or not a.get("game_minutes") or a.get("game_date") is None:
            continue
        out.append((str(a["game_date"])[:10], int(a["season"]), gid, a, b))
    out.sort(key=lambda g: (g[0], g[2]))
    return out


class RatingsWalk:
    """Walk-forward state machine. ``feed(date, games)`` rates then absorbs one date's games."""

    def __init__(self, params=None):
        self.p = _params(params)
        self.decay = 0.5 ** (1.0 / float(self.p["half_life"]))
        self.state = {}             # (season, team_id) -> _TeamState
        self.league = {}            # season -> _League
        self.final = {}             # (season, team_id) -> last pregame-style rating dict
        self.league_final = {}      # season -> league dict at season end
        self.last_date = {}         # team_id -> last game date (any season)

    # ── league ──
    def league_as_of(self, season):
        """{pace, rtg, allowed100{}} season-to-date, blended with last season's league."""
        cur = self.league.get(season) or _League()
        prev = self.league_final.get(season - 1)
        k = float(self.p["league_prior_games"]) if prev else 0.0
        if prev:
            pp = prev["pace"] * k                           # pseudo possessions over k*48 min
            pm = 48.0 * k
            pts = pp * prev["rtg"] / 100.0
            allowed = {s: pp * prev["allowed100"][s] / 100.0 for s in ALLOWED_STATS}
        else:
            pp = pm = pts = 0.0
            allowed = {s: 0.0 for s in ALLOWED_STATS}
        poss, mins = cur.poss + pp, cur.mins + pm
        if poss <= 0 or mins <= 0:
            return {"pace": _FALLBACK_LEAGUE["pace"], "rtg": _FALLBACK_LEAGUE["rtg"],
                    "allowed100": {s: None for s in ALLOWED_STATS}, "n": 0}
        return {"pace": 48.0 * poss / mins, "rtg": 100.0 * (cur.pts + pts) / poss,
                "allowed100": {s: 100.0 * (cur.allowed[s] + allowed[s]) / poss
                               for s in ALLOWED_STATS},
                "n": cur.n}

    # ── team ──
    def _init_team(self, season, team_id):
        """Seed a team's season state with last season's regressed rating as pseudo-games."""
        st = _TeamState()
        lg = self.league_as_of(season)
        prev = self.final.get((season - 1, team_id))
        lg_prev = self.league_final.get(season - 1)
        k = float(self.p["prior_games"])
        r = float(self.p["regress"])
        if prev and lg_prev:
            # regress last season's RELATIVE ratings toward 1.0, then express vs this league
            pace = lg["pace"] * (1 + (1 - r) * (prev["pace"] / lg_prev["pace"] - 1))
            ortg = lg["rtg"] * (1 + (1 - r) * (prev["ortg"] / lg_prev["rtg"] - 1))
            drtg = lg["rtg"] * (1 + (1 - r) * (prev["drtg"] / lg_prev["rtg"] - 1))
            allowed = {}
            for s in ALLOWED_STATS:
                a, la, lc = prev["allowed100"].get(s), lg_prev["allowed100"].get(s), \
                    lg["allowed100"].get(s)
                allowed[s] = (lc * (1 + (1 - r) * (a / la - 1))
                              if a is not None and la and lc is not None else lc)
        else:
            pace, ortg, drtg = lg["pace"], lg["rtg"], lg["rtg"]
            allowed = dict(lg["allowed100"])
        if k > 0:
            st.w = k
            st.poss = k * pace
            st.mins = k * 48.0
            st.pts = st.poss * ortg / 100.0
            st.opp_pts = st.poss * drtg / 100.0
            st.allowed = {s: (st.poss * allowed[s] / 100.0 if allowed[s] is not None else 0.0)
                          for s in ALLOWED_STATS}
            st.sos_d = st.sos_o = st.sos_p = k            # prior is already neutral (rel 1.0)
        return st

    def _team(self, season, team_id):
        key = (season, team_id)
        if key not in self.state:
            self.state[key] = self._init_team(season, team_id)
        return self.state[key]

    def rating(self, season, team_id, date=None):
        """PRE-GAME rating dict for a team as of the current walk position."""
        st = self._team(season, team_id)
        lg = self.league_as_of(season)
        if st.poss <= 0 or st.mins <= 0:
            pace_raw, ortg_raw, drtg_raw = lg["pace"], lg["rtg"], lg["rtg"]
        else:
            pace_raw = 48.0 * st.poss / st.mins
            ortg_raw = 100.0 * st.pts / st.poss
            drtg_raw = 100.0 * st.opp_pts / st.poss
        if self.p["opp_adjust"] and st.w > 0:
            sd, so, sp = st.sos_d / st.w, st.sos_o / st.w, st.sos_p / st.w
            ortg = ortg_raw / sd if sd > 0 else ortg_raw    # faced good D (sd<1) -> scale up
            drtg = drtg_raw / so if so > 0 else drtg_raw    # faced good O (so>1) -> scale down
            pace = pace_raw / sp if sp > 0 else pace_raw
        else:
            pace, ortg, drtg = pace_raw, ortg_raw, drtg_raw
        allowed = {s: (100.0 * st.allowed[s] / st.poss if st.poss > 0 else None)
                   for s in ALLOWED_STATS}
        last = self.last_date.get(team_id)
        rest = _days_between(last, date) if (last and date) else None
        return {"season": season, "team_id": team_id, "n": st.n,
                "pace": pace, "ortg": ortg, "drtg": drtg,
                "pace_raw": pace_raw, "ortg_raw": ortg_raw, "drtg_raw": drtg_raw,
                "lg_pace": lg["pace"], "lg_rtg": lg["rtg"],
                "allowed100": allowed, "lg_allowed100": dict(lg["allowed100"]),
                "rest_days": rest}

    # ── walk ──
    def feed(self, date, games):
        """Rate every game on ``date`` (pre-game, simultaneous), then absorb them.
        ``games`` = [(season, game_id, rowA, rowB)]. Returns {(game_id, team_id): rating}."""
        out = {}
        pre = {}
        for season, gid, a, b in games:
            for r in (a, b):
                tid = r["team_id"]
                pre[(gid, tid)] = self.rating(season, tid, date)
                out[(gid, tid)] = pre[(gid, tid)]
        for season, gid, a, b in games:
            for r, o in ((a, b), (b, a)):
                self._absorb(season, r, o, pre[(gid, o["team_id"])])
        for season, gid, a, b in games:
            for r in (a, b):
                self.last_date[r["team_id"]] = date
                self.final[(season, r["team_id"])] = self.rating(season, r["team_id"], date)
            self.league_final[season] = self.league_as_of(season)
        return out

    def _absorb(self, season, r, o, opp_pre):
        st = self._team(season, r["team_id"])
        d = self.decay
        poss, mins = float(r["possessions"]), float(r["game_minutes"])
        lg_rtg, lg_pace = opp_pre["lg_rtg"], opp_pre["lg_pace"]
        st.w = st.w * d + 1.0
        st.poss = st.poss * d + poss
        st.mins = st.mins * d + mins
        st.pts = st.pts * d + float(r.get("points") or 0.0)
        st.opp_pts = st.opp_pts * d + float(o.get("points") or 0.0)
        for s in ALLOWED_STATS:
            st.allowed[s] = st.allowed[s] * d + float(o.get(s) or 0.0)
        # opponent's PRE-GAME relative ratings (its strength at the time of this game)
        st.sos_d = st.sos_d * d + (opp_pre["drtg"] / lg_rtg if lg_rtg else 1.0)
        st.sos_o = st.sos_o * d + (opp_pre["ortg"] / lg_rtg if lg_rtg else 1.0)
        st.sos_p = st.sos_p * d + (opp_pre["pace"] / lg_pace if lg_pace else 1.0)
        st.n += 1
        lg = self.league.setdefault(season, _League())
        lg.poss += poss
        lg.mins += mins
        lg.pts += float(r.get("points") or 0.0)
        for s in ALLOWED_STATS:
            lg.allowed[s] += float(o.get(s) or 0.0)
        lg.n += 1


def pregame_ratings(rows, params=None):
    """{(game_id, team_id): pre-game rating} for every paired game in ``rows`` — the
    walk-forward input for backtests and the team-model study."""
    walk = RatingsWalk(params)
    out = {}
    by_date = {}
    for date, season, gid, a, b in _pair_games(rows):
        by_date.setdefault(date, []).append((season, gid, a, b))
    for date in sorted(by_date):
        out.update(walk.feed(date, by_date[date]))
    return out


def ratings_as_of(rows, date, season, team_ids=None, params=None):
    """{team_id: rating} as of ``date`` (only games strictly before it) for ``season`` — the
    live/serving entry. ``team_ids`` limits the output (default: every team seen)."""
    walk = RatingsWalk(params)
    by_date = {}
    for d, s, gid, a, b in _pair_games(rows):
        if d < str(date)[:10]:
            by_date.setdefault(d, []).append((s, gid, a, b))
    seen = set()
    for d in sorted(by_date):
        walk.feed(d, by_date[d])
        for s, gid, a, b in by_date[d]:
            seen.update((a["team_id"], b["team_id"]))
    ids = team_ids if team_ids is not None else sorted(seen)
    return {t: walk.rating(int(season), t, str(date)[:10]) for t in ids}


def project_matchup(home, away, hca=0.0):
    """Efficiency-model projection from two pre-game ratings (same season):
    possessions = pace_h·pace_a / lg_pace; each side's points = possessions × its ORtg × the
    other's DRtg / lg_rtg / 100; ``hca`` points split ± half to home/away.
    Returns {possessions, home_pts, away_pts, margin, total}."""
    lg_pace, lg_rtg = home["lg_pace"], home["lg_rtg"]
    poss = home["pace"] * away["pace"] / lg_pace
    h = poss * (home["ortg"] * away["drtg"] / lg_rtg) / 100.0 + hca / 2.0
    a = poss * (away["ortg"] * home["drtg"] / lg_rtg) / 100.0 - hca / 2.0
    return {"possessions": poss, "home_pts": h, "away_pts": a,
            "margin": h - a, "total": h + a}
