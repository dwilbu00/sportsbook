"""nba_team_stats.py — NBA team-market stats from the sportsdataverse schedule spine (no ESPN).

Produces the exact ``{season, recent, recent_games}`` dict the team analyzers consume
(analyze_moneyline / analyze_spreads / analyze_totals), sourced from
``nba_source.load_schedule`` (sportsdataverse ESPN-family parquet — dep-free, cloud-safe, no
credits). The analog of ``nfl_team_stats`` / ``espn_client.mlb_warehouse_team_stats`` for NBA,
so the live app no longer needs the ESPN API for NBA team recent-form / win% / scoring.

``recent_games`` carry the QUERIED team's odds-feed name on its own side (the analyzers
identify a team by exact string match), with opponents given their sportsdataverse full name.
"""
import nba_source


def _norm(name):
    return nba_source._norm(name)


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


def _completed_games(df):
    """Completed rows (both scores present, status complete) as plain dicts."""
    out = []
    for r in df.to_dict("records"):
        if not bool(r.get("status_type_completed")):
            continue
        hs, as_ = r.get("home_score"), r.get("away_score")
        try:
            hs, as_ = float(hs), float(as_)
        except (TypeError, ValueError):
            continue
        if hs != hs or as_ != as_:                       # NaN
            continue
        out.append({
            "home": r.get("home_display_name"), "away": r.get("away_display_name"),
            "home_norm": _norm(r.get("home_display_name")),
            "away_norm": _norm(r.get("away_display_name")),
            "home_score": hs, "away_score": as_,
            "date": str(r.get("game_date"))[:10],
        })
    return out


def team_stats(team_name, recent_n=10, seasons=None):
    """``{season, recent, recent_games}`` for one NBA team (odds-feed full name), from the
    sportsdataverse schedule spine, or None if the team/season can't be resolved. Never raises.

    ``season`` = current-season record + win% (runs_scored/allowed are None — not baseball);
    ``recent`` = form over the last ``recent_n`` completed games; ``recent_games`` = those
    games most-recent-first, the queried team's side keyed to ``team_name``."""
    try:
        want = _norm(team_name)
        if not want:
            return None
        if seasons is None:
            cur = nba_source.current_end_year()
            seasons = [cur, cur - 1]                      # +prior so early-season has form
        games = []
        for s in seasons:
            df = nba_source.load_schedule(int(s))
            if df is None or getattr(df, "empty", True):
                continue
            for g in _completed_games(df):
                if want in (g["home_norm"], g["away_norm"]):
                    g["season"] = int(s)
                    games.append(g)
        if not games:
            return None
        games.sort(key=lambda g: g["date"], reverse=True)   # most recent first
        recent_games = []
        for g in games:
            own_home = g["home_norm"] == want
            recent_games.append({
                "home_team": team_name if own_home else g["home"],
                "away_team": g["away"] if own_home else team_name,
                "home_score": g["home_score"], "away_score": g["away_score"],
                "total_score": g["home_score"] + g["away_score"], "date": g["date"],
            })
        recent = _recent_form(recent_games, team_name, recent_n)
        # Season record/win% over the CURRENT (most recent) season's completed games.
        cur_season = max(g["season"] for g in games)
        wins = losses = 0
        for g in games:
            if g["season"] != cur_season:
                continue
            own_home = g["home_norm"] == want
            scored = g["home_score"] if own_home else g["away_score"]
            allowed = g["away_score"] if own_home else g["home_score"]
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
