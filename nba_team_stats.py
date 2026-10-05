"""nba_team_stats.py — NBA team-market stats from the stats.nba.com warehouse spine (no ESPN).

Produces the exact ``{season, recent, recent_games}`` dict the team analyzers consume
(analyze_moneyline / analyze_spreads / analyze_totals), from the nba_game spine +nba_team dim
(nba_warehouse, stats.nba.com). The NBA analog of nfl_team_stats / mlb_warehouse_team_stats, so
the live app needs no ESPN for NBA team recent-form / win% / scoring.

``recent_games`` carry the QUERIED team's odds-feed name on its own side (the analyzers identify
a team by exact string match), opponents given their warehouse full name. None on any miss
(disabled / team not resolved / empty) so the caller falls back to the ESPN build.
"""
import db_store


def _norm(name):
    return db_store.normalize_name(name)


def _recent_form(recent_games, team_name, n):
    """win%/avg scored/allowed over the last ``n`` games — mirrors nfl_team_stats._recent_form
    (and espn_client.compute_recent_form), reimplemented to keep this an ESPN-free module."""
    recent = recent_games[:n]
    wins = losses = 0
    scored, allowed = [], []
    for g in recent:
        if g["home_team"] == team_name:
            s, a = g["home_score"], g["away_score"]
        elif g["away_team"] == team_name:
            s, a = g["away_score"], g["home_score"]
        else:
            continue
        if s > a:
            wins += 1
        else:
            losses += 1
        scored.append(s)
        allowed.append(a)
    tot = wins + losses
    return {
        "games": tot, "wins": wins, "losses": losses,
        "win_pct": wins / tot if tot else 0.0,
        "avg_scored": sum(scored) / len(scored) if scored else 0.0,
        "avg_allowed": sum(allowed) / len(allowed) if allowed else 0.0,
        "avg_total": (sum(s + a for s, a in zip(scored, allowed)) / len(scored)
                      if scored else 0.0),
    }


def team_stats(team_name, recent_n=10, seasons=None):
    """``{season, recent, recent_games}`` for one NBA team (odds-feed full name), from the
    stats.nba.com warehouse spine, or None if the team/season can't be resolved. Never raises.

    ``season`` = current-season record + win% (runs_scored/allowed None — not baseball);
    ``recent`` = form over the last ``recent_n`` completed games; ``recent_games`` = those games
    most-recent-first, the queried team's side keyed to ``team_name``."""
    try:
        import nba_warehouse as nw
        from sqlalchemy import or_, select
        if not nw.enabled():
            return None
        want = _norm(team_name)
        if not want:
            return None
        g, t = nw.nba_game, nw.nba_team
        with db_store.get_engine().connect() as conn:
            trow = conn.execute(select(t.c.team_id)
                                .where(t.c.name_norm == want)).fetchone()
            if trow is None:                             # nickname (last token) fallback
                tok = want.split()[-1]
                rows = conn.execute(select(t.c.team_id)
                                    .where(t.c.name_norm.like("% " + tok))).fetchall()
                if len(rows) != 1:
                    return None
                trow = rows[0]
            tid = trow[0]
            id2name = {r[0]: r[1] for r in conn.execute(select(t.c.team_id, t.c.name))}
            if seasons is None:
                cur = nw._current_season()
                seasons = [cur, cur - 1]                 # +prior so early-season has form
            stmt = (select(g.c.season, g.c.game_date, g.c.home_team_id, g.c.away_team_id,
                           g.c.home_score, g.c.away_score)
                    .where(or_(g.c.home_team_id == tid, g.c.away_team_id == tid),
                           g.c.season.in_([int(s) for s in seasons]),
                           g.c.home_score.isnot(None), g.c.away_score.isnot(None))
                    .order_by(g.c.game_date.desc()))
            games = [dict(r._mapping) for r in conn.execute(stmt)]
        if not games:
            return None
        recent_games = []
        for gm in games:
            own_home = gm["home_team_id"] == tid
            hs, as_ = float(gm["home_score"]), float(gm["away_score"])
            recent_games.append({
                "home_team": team_name if own_home else id2name.get(gm["home_team_id"]),
                "away_team": id2name.get(gm["away_team_id"]) if own_home else team_name,
                "home_score": hs, "away_score": as_, "total_score": hs + as_,
                "date": str(gm["game_date"])[:10], "season": gm["season"]})
        recent = _recent_form(recent_games, team_name, recent_n)
        cur_season = max(gm["season"] for gm in games)
        wins = losses = 0
        for gm in games:
            if gm["season"] != cur_season:
                continue
            own_home = gm["home_team_id"] == tid
            scored = gm["home_score"] if own_home else gm["away_score"]
            allowed = gm["away_score"] if own_home else gm["home_score"]
            if scored > allowed:
                wins += 1
            else:
                losses += 1
        tot = wins + losses
        season = {"record": f"{wins}-{losses}", "wins": wins, "losses": losses,
                  "win_pct": wins / tot if tot else 0.0,
                  "runs_scored": None, "runs_allowed": None}
        return {"season": season, "recent": recent, "recent_games": recent_games}
    except Exception:
        return None
