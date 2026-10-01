"""nfl_team_stats.py — NFL team-market stats from the nflverse games spine (no ESPN).

Produces the exact ``{season, recent, recent_games}`` dict the team analyzers consume
(analyze_moneyline / analyze_spreads / analyze_totals), sourced from
``nfl_schedule.load_games`` (nflverse games.csv / mirror parquet — dep-free, cloud-safe,
no credits). The analog of ``espn_client.mlb_warehouse_team_stats`` for NFL, so the live
app no longer needs ESPN for NFL team recent-form / win% / scoring.

``recent_games`` carry the QUERIED team's odds-feed name on its own side (the analyzers
identify a team by exact string match), with opponents given their nflverse full name.
The NFL model signal (margin/EPA) already comes from nfl_epa; this only replaces the
recency/record inputs that previously came from ESPN.
"""
from datetime import datetime, timezone

import nfl_epa
import nfl_schedule


def _current_season():
    try:
        return int(nfl_epa.season_for_date(
            datetime.now(timezone.utc).date().isoformat()))
    except Exception:
        d = datetime.now(timezone.utc).date()
        return d.year if d.month >= 8 else d.year - 1


def _abbr_to_name():
    """nflverse abbreviation -> a canonical full team name (first wins)."""
    out = {}
    for name, ab in nfl_epa.NAME_TO_ABBR.items():
        out.setdefault(ab, name)
    return out


def _recent_form(recent_games, team_name, n):
    """win%/avg scored/allowed over the last ``n`` games — byte-identical to
    espn_client.compute_recent_form, reimplemented to keep this an ESPN-free module."""
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
    """``{season, recent, recent_games}`` for one NFL team (odds-feed full name), from the
    nflverse games spine, or None if the team/season can't be resolved. Never raises.

    ``season`` = current-season record + win% (runs_scored/allowed are None — NFL);
    ``recent`` = form over the last ``recent_n`` completed games; ``recent_games`` = those
    games most-recent-first, the queried team's side keyed to ``team_name``."""
    try:
        ab = nfl_epa._abbr(team_name)
        if not ab:
            return None
        if seasons is None:
            cur = _current_season()
            seasons = [str(cur), str(cur - 1)]   # +prior so early-season weeks have form
        seasons = [str(s) for s in seasons]
        a2n = _abbr_to_name()
        rows = [g for g in nfl_schedule.load_games(seasons)
                if g.get("home_score") is not None and g.get("away_score") is not None
                and ab in (g.get("home_team"), g.get("away_team"))]
        if not rows:
            return None
        rows.sort(key=lambda g: str(g.get("gameday")), reverse=True)   # most recent first
        recent_games = []
        for g in rows:
            h_ab, a_ab = g["home_team"], g["away_team"]
            hs, as_ = float(g["home_score"]), float(g["away_score"])
            recent_games.append({
                "home_team": team_name if h_ab == ab else a2n.get(h_ab, h_ab),
                "away_team": team_name if a_ab == ab else a2n.get(a_ab, a_ab),
                "home_score": hs, "away_score": as_, "total_score": hs + as_,
                "date": str(g.get("gameday")),
            })
        recent = _recent_form(recent_games, team_name, recent_n)
        # Season record/win% over the CURRENT season's completed games.
        cur_season = seasons[0]
        wins = losses = 0
        for g in rows:
            if str(g.get("season")) != cur_season:
                continue
            own_home = g["home_team"] == ab
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
