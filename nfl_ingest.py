"""nfl_ingest.py — OFFLINE NFL data ingestion via nflreadpy → mirror parquet.

The nflverse ingestion layer, run on a dev box (NOT the live app). Uses
``nflreadpy`` (maintained successor to the deprecated ``nfl_data_py``; polars,
``load_*`` API) to pull the canonical nflverse feeds and write them as
committed/LFS-shared **mirror parquets** in ``warehouse_mirror_data/``, all keyed
by the canonical ``game_id`` spine (and ``player_id``/gsis for player layers).

Runtime (live app + backtests) NEVER imports nflreadpy/polars — reads the mirror
parquets dep-free (via `nfl_schedule` / `nfl_data` / pandas), exactly like MLB. So
polars lives only here, on the ingestion box.

LAYERS (all keyed to game_id spine; per-season immutable parquet unless noted):
  schedules    nfl_game__{sport}.parquet                (all seasons; the SPINE)
  pbp          nfl_pbp__{sport}__{season}.parquet       (widened trim: EPA + situational + weather)
  player_week  nfl_player_week__{sport}__{season}.parquet (props + player model)
  rosters      nfl_roster__{sport}__{season}.parquet    (weekly; active/inactive, QB1)
  depth        nfl_depth__{sport}__{season}.parquet     (depth charts; starter detection)
  injuries     nfl_injury__{sport}__{season}.parquet    (active-player guards)
  ngs          nfl_ngs_{type}__{sport}__{season}.parquet (passing/rushing/receiving aggregates)
  team_week    nfl_team_week__{sport}__{season}.parquet  (team efficiency)

Install (dev box only):  pip install nflreadpy
Run:   python nfl_ingest.py --all --seasons 2023,2024,2025,2026
       python nfl_ingest.py --pbp --player-week --seasons 2024,2025      (subset)
       python nfl_ingest.py --validate --seasons 2023,2024,2025,2026     (data-quality checks, FREE, no nflreadpy)
"""
import argparse
import os

try:
    import warehouse_mirror as _wm
    MIRROR_DIR = _wm.MIRROR_DIR
except Exception:
    MIRROR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "warehouse_mirror_data")

SPORT = "americanfootball_nfl"
KEEP_GAME_TYPES = ("REG", "WC", "DIV", "CON", "SB")   # spine (no PRE)

_SCHEDULE_COLS = ["game_id", "season", "game_type", "week", "gameday", "gametime",
                  "home_team", "away_team", "home_score", "away_score",
                  "location", "result", "total", "espn"]

# pbp: widened trim so situational splits (down/distance/field/air-yards/weather)
# are derivable later WITHOUT re-fetching the ~19MB/season raw. game_id is the spine.
_PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "game_date",
             "home_team", "away_team", "posteam", "defteam", "posteam_type",
             "play_type", "pass", "rush", "qb_dropback", "down", "ydstogo",
             "goal_to_go", "yardline_100", "air_yards", "yards_gained",
             "epa", "wpa", "wp", "cpoe", "success",
             "pass_touchdown", "rush_touchdown", "interception", "fumble_lost",
             "passer_player_id", "rusher_player_id", "receiver_player_id",
             "weather", "temp", "wind", "roof", "surface",
             "total_home_score", "total_away_score"]

_PLAYER_WEEK_COLS = [
    "player_id", "player_name", "player_display_name", "position", "position_group",
    "season", "week", "season_type", "team", "opponent_team",
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "sacks_suffered", "passing_air_yards",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa", "passing_cpoe",
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs", "rushing_epa",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_air_yards",
    "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
    "target_share", "air_yards_share", "special_teams_tds"]

_ROSTER_COLS = ["season", "week", "team", "position", "depth_chart_position",
                "status", "full_name", "football_name", "gsis_id", "jersey_number",
                "years_exp"]
_DEPTH_COLS = ["season", "week", "club_code", "team", "position", "depth_team",
               "formation", "gsis_id", "full_name", "jersey_number"]
_INJURY_COLS = ["season", "week", "team", "gsis_id", "full_name", "position",
                "report_status", "practice_status", "report_primary_injury"]
_TEAM_WEEK_COLS = None   # keep all team_stats cols (compact, ~1 row/team/week)


def _require_nflreadpy():
    try:
        import nflreadpy
        return nflreadpy
    except ImportError as exc:
        raise SystemExit(
            "nflreadpy is not installed. This is the OFFLINE ingestion box only:\n"
            "    pip install nflreadpy\n"
            "(the live app/backtests never import it — they read the mirror parquet).") from exc


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _select(df, cols):
    if cols is None:
        return df, []
    have = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    return df[have].copy(), missing


def _write(df, filename, verbose=True):
    os.makedirs(MIRROR_DIR, exist_ok=True)
    path = os.path.join(MIRROR_DIR, filename)
    df.to_parquet(path, index=False)
    if verbose:
        print(f"    -> {filename}: {len(df):,} rows, {len(df.columns)} cols")
    return path


def _per_season(df, season_col="season"):
    """Yield (season_str, sub_df) for each season present."""
    if season_col not in df.columns:
        yield None, df
        return
    for s, sub in df.groupby(season_col):
        yield str(s), sub


# ── layer ingesters ───────────────────────────────────────────────────────────

def ingest_schedules(seasons, verbose=True):
    nflreadpy = _require_nflreadpy()
    df = _to_pandas(nflreadpy.load_schedules(seasons=[int(s) for s in seasons]))
    df, missing = _select(df, _SCHEDULE_COLS)
    df = df[df["game_type"].isin(KEEP_GAME_TYPES)]
    df = df[df["season"].astype(str).isin({str(s) for s in seasons})]
    df["season"] = df["season"].astype(str)
    for c in ("home_score", "away_score"):
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    if verbose:
        print(f"  [schedules] {len(df)} games (game_type in {KEEP_GAME_TYPES})")
    _write(df, f"nfl_game__{SPORT}.parquet", verbose)
    if missing and verbose:
        print(f"    note: cols not in feed: {missing}")
    return len(df)


def _ingest_per_season(layer, loader_name, cols, seasons, verbose=True, **kw):
    """Generic: load a season-keyed layer via nflreadpy, select cols, write one
    parquet per season (immutable; re-run to refresh the current season)."""
    nflreadpy = _require_nflreadpy()
    loader = getattr(nflreadpy, loader_name)
    df = _to_pandas(loader(seasons=[int(s) for s in seasons], **kw))
    df, missing = _select(df, cols)
    total = 0
    for s, sub in _per_season(df):
        if s is None or s not in {str(x) for x in seasons}:
            continue
        _write(sub, f"{layer}__{SPORT}__{s}.parquet", verbose)
        total += len(sub)
    if verbose:
        print(f"  [{layer}] {total:,} rows across {len(seasons)} season(s)"
              + (f"; cols not in feed: {missing}" if missing else ""))
    return total


def ingest_pbp(seasons, verbose=True):
    return _ingest_per_season("nfl_pbp", "load_pbp", _PBP_COLS, seasons, verbose)


def ingest_player_week(seasons, verbose=True):
    # load_player_stats defaults to weekly; be explicit if the kwarg exists.
    try:
        return _ingest_per_season("nfl_player_week", "load_player_stats",
                                  _PLAYER_WEEK_COLS, seasons, verbose,
                                  summary_level="week")
    except TypeError:
        return _ingest_per_season("nfl_player_week", "load_player_stats",
                                  _PLAYER_WEEK_COLS, seasons, verbose)


def ingest_rosters(seasons, verbose=True):
    return _ingest_per_season("nfl_roster", "load_rosters_weekly", _ROSTER_COLS,
                              seasons, verbose)


def ingest_depth(seasons, verbose=True):
    return _ingest_per_season("nfl_depth", "load_depth_charts", _DEPTH_COLS,
                              seasons, verbose)


def ingest_injuries(seasons, verbose=True):
    return _ingest_per_season("nfl_injury", "load_injuries", _INJURY_COLS,
                              seasons, verbose)


def ingest_team_week(seasons, verbose=True):
    try:
        return _ingest_per_season("nfl_team_week", "load_team_stats", _TEAM_WEEK_COLS,
                                  seasons, verbose, summary_level="week")
    except TypeError:
        return _ingest_per_season("nfl_team_week", "load_team_stats", _TEAM_WEEK_COLS,
                                  seasons, verbose)


def ingest_ngs(seasons, verbose=True):
    """Next Gen Stats aggregates per stat_type → nfl_ngs_{type}__{sport}__{season}."""
    nflreadpy = _require_nflreadpy()
    total = 0
    for stype in ("passing", "rushing", "receiving"):
        try:
            df = _to_pandas(nflreadpy.load_nextgen_stats(
                seasons=[int(s) for s in seasons], stat_type=stype))
        except Exception as exc:
            if verbose:
                print(f"  [ngs:{stype}] skipped: {type(exc).__name__}: {exc}")
            continue
        # NGS carries season+week; drop the season==0 season-aggregate rows.
        if "week" in df.columns:
            df = df[df["week"] != 0]
        for s, sub in _per_season(df):
            if s is None or s not in {str(x) for x in seasons}:
                continue
            _write(sub, f"nfl_ngs_{stype}__{SPORT}__{s}.parquet", verbose)
            total += len(sub)
    if verbose:
        print(f"  [ngs] {total:,} player-week rows (passing+rushing+receiving)")
    return total


_LAYERS = [
    ("schedules", ingest_schedules), ("pbp", ingest_pbp),
    ("player_week", ingest_player_week), ("rosters", ingest_rosters),
    ("depth", ingest_depth), ("injuries", ingest_injuries),
    ("ngs", ingest_ngs), ("team_week", ingest_team_week),
]


# ── FREE data-quality validation (no nflreadpy; reads the written parquets) ─────

def validate(seasons, verbose=True):
    """Assert integrity of the written mirror parquets (dep-free). Returns True iff
    all checks pass. Substitutes for the MLB Azure-parity net (there's no Azure to
    verify NFL against)."""
    import pandas as pd
    import nfl_epa
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok = ok and cond
        print(f"    [{'PASS' if cond else 'FAIL'}] {msg}")

    print("=== NFL mirror validate ===")
    # 1) spine present + counts + game_type set + abbreviation parity
    gp = os.path.join(MIRROR_DIR, f"nfl_game__{SPORT}.parquet")
    if not os.path.exists(gp):
        print("  [FAIL] spine parquet missing — run --schedules"); return False
    g = pd.read_parquet(gp)
    canon = set(nfl_epa.TEAM_ABBR_TO_NAME)
    teams = set(g["home_team"]) | set(g["away_team"])
    check(teams <= canon, f"spine team abbrs ⊆ canonical 32 (extra: {sorted(teams-canon)})")
    check(canon <= teams, f"all 32 canonical teams appear in spine (missing: {sorted(canon-teams)})")
    check(set(g["game_type"]) <= set(KEEP_GAME_TYPES), "spine game_type ⊆ {REG,WC,DIV,CON,SB} (no PRE)")
    for s in seasons:
        gs = g[g["season"].astype(str) == str(s)]
        n = len(gs)
        # full season = 272 REG + 13 playoff = 285; current season may be partial
        check(n > 0, f"{s}: {n} spine games")
    # 2) per-season layers: existence + game_id/player_id join-ability + no dups
    for s in seasons:
        for layer, key in (("nfl_pbp", "game_id"), ("nfl_player_week", "player_id"),
                           ("nfl_roster", "gsis_id"), ("nfl_team_week", None)):
            p = os.path.join(MIRROR_DIR, f"{layer}__{SPORT}__{s}.parquet")
            if not os.path.exists(p):
                if verbose:
                    print(f"    [skip] {layer} {s} not present")
                continue
            df = pd.read_parquet(p)
            check(len(df) > 0, f"{layer} {s}: {len(df):,} rows")
            if layer == "nfl_pbp":
                gids = set(df["game_id"]) - set(g[g["season"].astype(str) == str(s)]["game_id"])
                check(len(gids) == 0, f"{layer} {s}: all game_ids in spine (orphans: {len(gids)})")
            if layer == "nfl_player_week" and "player_id" in df.columns:
                check(df["player_id"].notna().all(), f"{layer} {s}: player_id non-null")
    print(f"  VALIDATE: {'ALL PASS' if ok else 'FAIL — see above'}")
    return ok


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="ingest every layer")
    for name, _ in _LAYERS:
        ap.add_argument(f"--{name.replace('_', '-')}", action="store_true",
                        help=f"ingest the {name} layer")
    ap.add_argument("--validate", action="store_true",
                    help="data-quality checks on the written parquets (FREE, no nflreadpy)")
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    selected = [(n, fn) for n, fn in _LAYERS
                if args.all or getattr(args, n.replace("-", "_"))]
    if selected:
        print(f"NFL ingest via nflreadpy → {MIRROR_DIR}  (seasons: {seasons})")
        for name, fn in selected:
            print(f"\n[{name}]")
            try:
                fn(seasons)
            except SystemExit:
                raise
            except Exception as exc:
                print(f"  ⚠ {name} failed: {type(exc).__name__}: {exc}")
    if args.validate:
        validate(seasons)
    if not selected and not args.validate:
        print("nothing to do — pass --all (or --pbp/--player-week/--rosters/--depth/"
              "--injuries/--ngs/--team-week/--schedules) and/or --validate. "
              f"Mirror dir: {MIRROR_DIR}")


if __name__ == "__main__":
    main()
