"""nfl_ingest.py — OFFLINE NFL data ingestion via nflreadpy → mirror parquet.

The nflverse ingestion layer, run on a dev box (NOT the live app). Uses
``nflreadpy`` (the maintained successor to the deprecated ``nfl_data_py``; polars
backend, ``load_*`` API) to pull the canonical nflverse feeds and write them as
committed/LFS-shared **mirror parquets** in ``warehouse_mirror_data/``, keyed by
the canonical ``game_id`` spine.

Runtime (live app + backtests) NEVER imports nflreadpy/polars — they read the
mirror parquets dep-free (via ``nfl_schedule`` / pandas), exactly like the MLB
mirror. So polars lives only here, on the ingestion box.

Layers (built incrementally):
  * schedules  -> nfl_game__americanfootball_nfl.parquet         (the SPINE; game_id,
                  season, week, game_type, gameday, teams, scores, espn) — Phase 1.
  * player-wk  -> nfl_player_week__americanfootball_nfl__{season}.parquet — Phase 4.
  * nextgen    -> nfl_ngs_{type}__americanfootball_nfl__{season}.parquet — Phase 4.

Install (dev box only):  pip install nflreadpy
Run:                     python nfl_ingest.py --schedules --seasons 2023,2024,2025,2026
"""
import argparse
import os

# Mirror dir: reuse the SAME location warehouse_mirror uses, so NFL parquets sit
# alongside the MLB ones and ship via the same Git-LFS sharing.
try:
    import warehouse_mirror as _wm
    MIRROR_DIR = _wm.MIRROR_DIR
except Exception:
    MIRROR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "warehouse_mirror_data")

SPORT = "americanfootball_nfl"

# game_type kept in the spine: regular + all playoffs + Super Bowl; preseason
# ('PRE') is EXCLUDED (and isn't present in load_schedules anyway).
KEEP_GAME_TYPES = ("REG", "WC", "DIV", "CON", "SB")

# Spine columns we persist (subset of load_schedules' ~40).
_SCHEDULE_COLS = ["game_id", "season", "game_type", "week", "gameday", "gametime",
                  "home_team", "away_team", "home_score", "away_score",
                  "location", "result", "total", "espn"]


def _require_nflreadpy():
    try:
        import nflreadpy  # noqa: F401
        return nflreadpy
    except ImportError as exc:
        raise SystemExit(
            "nflreadpy is not installed. This is the OFFLINE ingestion box only:\n"
            "    pip install nflreadpy\n"
            "(the live app/backtests never import it — they read the mirror parquet).") from exc


def _to_pandas(df):
    """nflreadpy returns polars; normalize to pandas (already a dep)."""
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _game_file():
    return os.path.join(MIRROR_DIR, f"nfl_game__{SPORT}.parquet")


def ingest_schedules(seasons, verbose=True):
    """Pull nflverse schedules for ``seasons`` via nflreadpy → the nfl_game spine
    parquet. Filters to KEEP_GAME_TYPES. Writes ALL requested seasons in one file
    (like mlb_game__). Returns the number of games written."""
    nflreadpy = _require_nflreadpy()
    os.makedirs(MIRROR_DIR, exist_ok=True)
    seasons = [int(s) for s in seasons]
    df = _to_pandas(nflreadpy.load_schedules(seasons=seasons))
    # Keep only the columns we persist that actually exist in the feed.
    cols = [c for c in _SCHEDULE_COLS if c in df.columns]
    missing = [c for c in _SCHEDULE_COLS if c not in df.columns]
    df = df[cols].copy()
    df = df[df["game_type"].isin(KEEP_GAME_TYPES)]
    df = df[df["season"].astype(str).isin({str(s) for s in seasons})]
    # Normalize dtypes: season/scores as nullable ints, everything keyed by str.
    df["season"] = df["season"].astype(str)
    for c in ("home_score", "away_score"):
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    path = _game_file()
    df.to_parquet(path, index=False)
    if verbose:
        n = len(df)
        finals = int(df["home_score"].notna().sum()) if "home_score" in df else 0
        by = df.groupby("season").size().to_dict() if n else {}
        print(f"  [nfl_ingest] schedules -> {os.path.basename(path)}: {n} games "
              f"({finals} final)  by season: {by}")
        if missing:
            print(f"  [nfl_ingest] note: columns not in feed (skipped): {missing}")
    return len(df)


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedules", action="store_true",
                    help="ingest the nfl_game schedule/result SPINE (Phase 1)")
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    did = False
    if args.schedules:
        ingest_schedules(seasons)
        did = True
    if not did:
        print("  nothing to do — pass --schedules (player-week + NGS layers land in "
              "later phases). Mirror dir: " + MIRROR_DIR)


if __name__ == "__main__":
    main()
