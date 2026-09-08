"""nfl_data.py — dep-free readers for the NFL mirror parquets + derived game-context.

Runtime/backtest side of the NFL data model. Reads the per-season mirror parquets
written offline by ``nfl_ingest`` (nflreadpy), mirror-first and LFS-stub-safe (a
pointer stub on Streamlit → treated as absent, callers fall back). NO nflreadpy /
polars import here. All layers key to the ``game_id`` spine (and ``player_id``/gsis
for player layers), so joins are exact.

Also derives GAME-CONTEXT (rest/bye/short-week/division/primetime/home) straight
from the schedule spine — zero extra data, leakage-free, historically the most
robust NFL angles.

Leakage note: the raw layers are per-play / per-player-game. AS-OF aggregation
(only rows strictly before a target game_date) is the CONSUMER's job — see
``nfl_epa`` for the pattern; do NOT build season-to-date features that peek.
"""
import os

_SPORT = "americanfootball_nfl"


def _mirror_dir():
    try:
        import warehouse_mirror as _wm
        return _wm.MIRROR_DIR
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "warehouse_mirror_data")


def _is_real(path):
    try:
        import warehouse_mirror as _wm
        return _wm._is_real_parquet(path)
    except Exception:
        return os.path.exists(path)


_CACHE = {}


def _read_layer(layer, season):
    """Return a pandas DataFrame for one per-season layer parquet, or None if the
    file is absent / an LFS stub. Cached in-process."""
    key = (layer, str(season))
    if key in _CACHE:
        return _CACHE[key]
    path = os.path.join(_mirror_dir(), f"{layer}__{_SPORT}__{season}.parquet")
    df = None
    if _is_real(path):
        try:
            import pandas as pd
            df = pd.read_parquet(path)
        except Exception:
            df = None
    _CACHE[key] = df
    return df


def _concat(layer, seasons):
    import pandas as pd
    frames = [d for d in (_read_layer(layer, s) for s in seasons) if d is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


# ── layer readers (each returns a DataFrame or None) ──────────────────────────
def pbp(seasons):               return _concat("nfl_pbp", seasons)
def player_week(seasons):       return _concat("nfl_player_week", seasons)
def rosters(seasons):           return _concat("nfl_roster", seasons)
def depth_charts(seasons):      return _concat("nfl_depth", seasons)
def injuries(seasons):          return _concat("nfl_injury", seasons)
def team_week(seasons):         return _concat("nfl_team_week", seasons)
def ngs(seasons, stat_type="passing"):
    return _concat(f"nfl_ngs_{stat_type}", seasons)


# ── NFL divisions (static; for the division-game context flag) ────────────────
DIVISIONS = {
    "BUF": "AFCE", "MIA": "AFCE", "NE": "AFCE", "NYJ": "AFCE",
    "BAL": "AFCN", "CIN": "AFCN", "CLE": "AFCN", "PIT": "AFCN",
    "HOU": "AFCS", "IND": "AFCS", "JAX": "AFCS", "TEN": "AFCS",
    "DEN": "AFCW", "KC": "AFCW", "LV": "AFCW", "LAC": "AFCW",
    "DAL": "NFCE", "NYG": "NFCE", "PHI": "NFCE", "WAS": "NFCE",
    "CHI": "NFCN", "DET": "NFCN", "GB": "NFCN", "MIN": "NFCN",
    "ATL": "NFCS", "CAR": "NFCS", "NO": "NFCS", "TB": "NFCS",
    "ARI": "NFCW", "LA": "NFCW", "SF": "NFCW", "SEA": "NFCW",
}


def _gameday_dt(gameday):
    from datetime import datetime
    try:
        return datetime.strptime(str(gameday)[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _is_primetime(gameday, gametime):
    """Thu/Sun-night/Mon night ≈ primetime. Uses weekday + kickoff hour (ET)."""
    dt = _gameday_dt(gameday)
    if dt is None:
        return None
    wd = dt.weekday()   # Mon=0 .. Sun=6
    try:
        hr = int(str(gametime)[:2]) if gametime else None
    except (TypeError, ValueError):
        hr = None
    if wd in (0, 3):            # Monday or Thursday game = primetime
        return True
    if wd == 6 and hr is not None and hr >= 19:   # Sunday night
        return True
    return False


def game_context(seasons):
    """Per-game context derived from the spine alone (leakage-free): rest days for
    each team (days since its previous game), short-week / off-bye flags, division
    flag, primetime, home/away. Returns {game_id: {...}}.

    Rest = days since that team's previous scheduled game (any game_type), computed
    chronologically within the loaded seasons — so week-1 rest is undefined (None).
    """
    import nfl_schedule
    games = []
    for s in seasons:
        games.extend(nfl_schedule.load_games([str(s)]))
    games.sort(key=lambda g: (str(g.get("gameday")), str(g.get("gametime") or "")))
    last_played = {}   # team -> datetime of its previous game
    out = {}
    for g in games:
        dt = _gameday_dt(g.get("gameday"))
        h, a = g.get("home_team"), g.get("away_team")
        gid = g.get("game_id")
        rest_h = (dt - last_played[h]).days if (dt and h in last_played) else None
        rest_a = (dt - last_played[a]).days if (dt and a in last_played) else None
        out[gid] = {
            "game_id": gid, "season": g.get("season"), "week": g.get("week"),
            "home_team": h, "away_team": a, "gameday": g.get("gameday"),
            "rest_home": rest_h, "rest_away": rest_a,
            "short_week_home": (rest_h is not None and rest_h <= 5),
            "short_week_away": (rest_a is not None and rest_a <= 5),
            "off_bye_home": (rest_h is not None and rest_h >= 13),
            "off_bye_away": (rest_a is not None and rest_a >= 13),
            "rest_edge_home": ((rest_h - rest_a) if (rest_h is not None and rest_a is not None) else None),
            "is_division": (DIVISIONS.get(h) is not None and DIVISIONS.get(h) == DIVISIONS.get(a)),
            "is_primetime": _is_primetime(g.get("gameday"), g.get("gametime")),
        }
        if dt is not None:
            last_played[h] = dt
            last_played[a] = dt
    return out


if __name__ == "__main__":
    import argparse
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    ap.add_argument("--layers", action="store_true", help="report which layers are present")
    ap.add_argument("--context", action="store_true", help="summarize derived game-context")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    if args.layers or not args.context:
        print(f"=== NFL mirror layers present ({seasons}) ===")
        for layer in ("nfl_pbp", "nfl_player_week", "nfl_roster", "nfl_depth",
                      "nfl_injury", "nfl_team_week", "nfl_ngs_passing",
                      "nfl_ngs_rushing", "nfl_ngs_receiving"):
            per = []
            for s in seasons:
                d = _read_layer(layer, s)
                per.append(f"{s}:{len(d) if d is not None else '—'}")
            print(f"  {layer:20} {'  '.join(per)}")
    if args.context:
        ctx = game_context(seasons)
        n = len(ctx)
        div = sum(1 for c in ctx.values() if c["is_division"])
        pt = sum(1 for c in ctx.values() if c["is_primetime"])
        sw = sum(1 for c in ctx.values() if c["short_week_home"] or c["short_week_away"])
        bye = sum(1 for c in ctx.values() if c["off_bye_home"] or c["off_bye_away"])
        print(f"=== game-context ({n} games) === division={div} primetime={pt} "
              f"short-week={sw} off-bye={bye}")
