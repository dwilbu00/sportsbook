"""Local parquet mirror of the read-only historical warehouse, for FAST OFFLINE
backtesting.

All four backtest tools (r2_backtest, coherence_backtest, f5_backtest,
scenario_backtest) read through this when `ODI_BACKTEST_MIRROR` is set AND the mirror
exists — one shared local columnar store instead of every tool re-hitting Azure on
every run. The LIVE APP and the CALIBRATION REFIT are untouched (they read Azure
directly and must — they need fresh data / the full ESPN-shape gamelogs).

Design:
- `sync()` pulls each read-only table from Azure and writes parquet (season-keyed).
  Odds are stored as the EXACT dict-rows `db_store.team_market_lines` /
  `player_prop_lines` return (per season × book), so shape parity is by construction.
  The MLB fact tables (mlb_game / mlb_pitcher_game / mlb_batter_game) are stored raw
  (needed columns only), and the reader rebuilds the same index structures the
  backtests consume.
- Readers mirror the db_store readers + r2_data / scenario index builders. Each
  returns `None` when its parquet is missing so the caller FALLS BACK to Azure per
  call — the mirror is safe-by-default (a partial sync never yields wrong data, just
  a slower Azure read for the missing slice).

Only the DEFAULT read path is mirrored (all snapshots, close-picking done downstream
on captured_at); `only_early` / `exclude_early` are NOT mirrored (the backtests don't
use them). 2024/2025 are immutable done seasons — re-sync only the current season.

Sync (needs Azure; run once per machine or copy the dir):
  python warehouse_mirror.py --sync --sport baseball_mlb --seasons 2024,2025,2026
Then for backtests:  export ODI_BACKTEST_MIRROR=1
"""
import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIRROR_DIR = os.environ.get(
    "ODI_MIRROR_DIR", os.path.join(_SCRIPT_DIR, "warehouse_mirror_data"))

# Books the backtests read (DK-parity 'draftkings' also captures legacy NULL rows,
# applied in SQL at sync time so the parquet already includes them).
BOOKS = ("draftkings", "pinnacle", "fanduel")


def enabled():
    """True iff mirror reads are turned on AND the mirror dir exists."""
    flag = str(os.environ.get("ODI_BACKTEST_MIRROR", "")).strip().lower()
    return flag in ("1", "true", "yes", "on") and os.path.isdir(MIRROR_DIR)


# ── parquet helpers ──────────────────────────────────────────────────────────

def _path(name):
    return os.path.join(MIRROR_DIR, name)


def _records(df):
    """DataFrame -> list[dict] with NaN coerced back to None (so a mirrored row is
    byte-identical to the SQL reader's dict — never NaN where the reader gives None)."""
    import pandas as pd
    if df is None or df.empty:
        return []
    obj = df.astype(object).where(pd.notna(df), None)
    return obj.to_dict("records")


def _read(name):
    """Read a mirror parquet -> DataFrame, or None if absent (caller falls back)."""
    import pandas as pd
    p = _path(name)
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


def _write(df, name):
    import pandas as pd
    os.makedirs(MIRROR_DIR, exist_ok=True)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_parquet(
        _path(name), index=False)


def _team_file(sport, book, season):
    return f"team__{sport}__{book}__{season}.parquet"


def _prop_file(sport, book, season):
    return f"props__{sport}__{book}__{season}.parquet"


def _game_file(sport):
    return f"mlb_game__{sport}.parquet"


def _pitcher_file(sport, season):
    return f"pitcher_game__{sport}__{season}.parquet"


def _batter_file(sport, season):
    return f"batter_game__{sport}__{season}.parquet"


# ── SYNC (Azure -> parquet) ──────────────────────────────────────────────────

def sync(sport, seasons, refresh=False, verbose=True):
    """Pull the read-only tables from Azure into the local parquet mirror. Skips a
    file that already exists unless ``refresh`` (2024/2025 are immutable; pass
    --refresh or --seasons <current> to refresh the live season)."""
    import db_store
    db_store.promote_secrets_from_toml()
    seasons = [str(s) for s in seasons]

    def _skip(name):
        return (not refresh) and os.path.exists(_path(name))

    # 1) Odds — store the EXACT db_store reader dicts, per season × book (parity by
    #    construction; the DK-parity NULL-book predicate is already applied in SQL).
    for s in seasons:
        df, dt = f"{s}-01-01", f"{s}-12-31"
        for book in BOOKS:
            tf = _team_file(sport, book, s)
            if not _skip(tf):
                rows = db_store.team_market_lines(
                    sport, date_from=df, date_to=dt, bookmaker=book)
                _write(rows, tf)
                if verbose:
                    print(f"  team  {s} {book:<10} {len(rows):>7,} rows -> {tf}")
            pf = _prop_file(sport, book, s)
            if not _skip(pf):
                rows = db_store.player_prop_lines(
                    sport, date_from=df, date_to=dt, bookmaker=book)
                _write(rows, pf)
                if verbose:
                    print(f"  props {s} {book:<10} {len(rows):>7,} rows -> {pf}")

    # 2) MLB fact tables (raw, needed columns only).
    _sync_facts(sport, seasons, refresh, verbose)


def _sync_facts(sport, seasons, refresh, verbose):
    import mlb_warehouse as wh
    import db_store
    from sqlalchemy import select as _select
    eng = db_store.get_engine()
    g = wh.mlb_game

    # mlb_game (all seasons in one file — a few thousand rows).
    gf = _game_file(sport)
    if refresh or not os.path.exists(_path(gf)):
        with eng.connect() as conn:
            rows = conn.execute(_select(
                g.c.game_pk, g.c.official_date, g.c.season, g.c.game_type,
                g.c.home_score, g.c.away_score, g.c.home_score_f5, g.c.away_score_f5,
                g.c.home_team_id, g.c.away_team_id)).fetchall()
        _write([dict(r._mapping) for r in rows], gf)
        if verbose:
            print(f"  mlb_game {'':<11} {len(rows):>7,} rows -> {gf}")

    # game_pk -> season / official_date, to season-scope + date-order the fact rows.
    gmeta = _read(gf)
    pk_season, pk_date = {}, {}
    if gmeta is not None:
        for r in _records(gmeta):
            gpk = r.get("game_pk")
            if gpk is None:
                continue
            pk_season[int(gpk)] = r.get("season")
            pk_date[int(gpk)] = r.get("official_date")

    pg, bg = wh.mlb_pitcher_game, wh.mlb_batter_game
    for s in seasons:
        si = int(s)
        pf = _pitcher_file(sport, s)
        if refresh or not os.path.exists(_path(pf)):
            with eng.connect() as conn:
                rows = conn.execute(_select(
                    pg.c.athlete_id, pg.c.game_pk, pg.c.team_id, pg.c.GS,
                    pg.c.IP, pg.c.ER, pg.c.K, pg.c.BB, pg.c.BF)).fetchall()
            out = []
            for r in rows:
                d = dict(r._mapping)
                gpk = d.get("game_pk")
                if gpk is None or pk_season.get(int(gpk)) != si:
                    continue
                d["official_date"] = pk_date.get(int(gpk))
                out.append(d)
            _write(out, pf)
            if verbose:
                print(f"  pitcher_game {s} {'':<6} {len(out):>7,} rows -> {pf}")
        bf = _batter_file(sport, s)
        if refresh or not os.path.exists(_path(bf)):
            with eng.connect() as conn:
                rows = conn.execute(_select(
                    bg.c.athlete_id, bg.c.game_pk, bg.c.H, bg.c.SO,
                    bg.c.TB, bg.c.RBI)).fetchall()
            out = []
            for r in rows:
                d = dict(r._mapping)
                gpk = d.get("game_pk")
                if gpk is None or pk_season.get(int(gpk)) != si:
                    continue
                d["official_date"] = pk_date.get(int(gpk))
                out.append(d)
            _write(out, bf)
            if verbose:
                print(f"  batter_game {s} {'':<7} {len(out):>7,} rows -> {bf}")


# ── READERS (mirror the db_store readers + index builders; None = fall back) ──

def _seasons_from(dates=None, date_from=None, date_to=None):
    """Infer the season set a query spans (files are season-keyed)."""
    yrs = set()
    for d in (dates or []):
        yrs.add(str(d)[:4])
    for d in (date_from, date_to):
        if d:
            yrs.add(str(d)[:4])
    return yrs


def _odds_lines(kind_file, sport, dates, date_from, date_to, bookmaker):
    """Shared team/prop reader: concat the season×book parquet(s), apply the same
    date filter the SQL reader would. Returns None if ANY needed season file is
    missing (so the caller falls back to Azure for the whole call)."""
    import pandas as pd
    seasons = _seasons_from(dates, date_from, date_to)
    if not seasons:
        return None
    frames = []
    for s in sorted(seasons):
        df = _read(kind_file(sport, bookmaker, s))
        if df is None:
            return None                       # partial -> fall back to Azure
        frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if dates:
        all_df = all_df[all_df["game_date"].isin([str(d) for d in dates])]
    else:
        if date_from is not None:
            all_df = all_df[all_df["game_date"] >= str(date_from)]
        if date_to is not None:
            all_df = all_df[all_df["game_date"] <= str(date_to)]
    return _records(all_df)


def team_market_lines(sport, dates=None, date_from=None, date_to=None,
                      only_early=False, bookmaker="draftkings"):
    if only_early:
        return None                           # not mirrored -> Azure
    return _odds_lines(_team_file, sport, dates, date_from, date_to, bookmaker)


def player_prop_lines(sport, dates=None, date_from=None, date_to=None,
                      exclude_early=False, only_early=False, prop_keys=None,
                      bookmaker="draftkings"):
    if only_early or exclude_early:
        return None                           # not mirrored -> Azure
    rows = _odds_lines(_prop_file, sport, dates, date_from, date_to, bookmaker)
    if rows is not None and prop_keys:
        keys = set(prop_keys)
        rows = [r for r in rows if r.get("prop_key") in keys]
    return rows


def _game_df(sport):
    return _read(_game_file(sport))


def build_team_scores_index(sport="baseball_mlb"):
    """{game_pk:(home_score, away_score)} for final games. None if unmirrored."""
    df = _game_df(sport)
    if df is None:
        return None
    df = df.dropna(subset=["home_score", "away_score"])
    return {int(r["game_pk"]): (float(r["home_score"]), float(r["away_score"]))
            for r in _records(df)}


def build_team_finals_index(sport="baseball_mlb"):
    """{game_pk: home_won 1.0/0.0} for final non-tie games. None if unmirrored."""
    idx = build_team_scores_index(sport)
    if idx is None:
        return None
    return {gpk: (1.0 if hs > as_ else 0.0)
            for gpk, (hs, as_) in idx.items() if hs != as_}


def build_f5_scores_index(sport="baseball_mlb"):
    """{game_pk:(home_score_f5, away_score_f5)}. None if unmirrored."""
    df = _game_df(sport)
    if df is None:
        return None
    df = df.dropna(subset=["home_score_f5", "away_score_f5"])
    return {int(r["game_pk"]): (float(r["home_score_f5"]), float(r["away_score_f5"]))
            for r in _records(df)}


def game_teams_index(sport="baseball_mlb"):
    """{game_pk:(home_team_id_str, away_team_id_str)}. None if unmirrored."""
    df = _game_df(sport)
    if df is None:
        return None
    return {int(r["game_pk"]): (str(r["home_team_id"]), str(r["away_team_id"]))
            for r in _records(df) if r.get("game_pk") is not None}


def pitcher_team_index(sport="baseball_mlb", seasons=None):
    """{(athlete_id_str, game_pk_int):(team_id_str, GS_float)}. None if no season
    file is present."""
    seasons = [str(s) for s in (seasons or [])]
    frames = [_read(_pitcher_file(sport, s)) for s in seasons]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    out = {}
    for f in frames:
        for r in _records(f):
            gpk = r.get("game_pk")
            if gpk is None:
                continue
            out[(str(r["athlete_id"]), int(gpk))] = (str(r["team_id"]), r.get("GS"))
    return out


def pitcher_game_index(season, sport="baseball_mlb"):
    """{athlete_id_str: [(official_date, outs, er, k, bb, bf), ...] asc} — matches
    mlb_warehouse._pitcher_game_index. None if the season file is absent."""
    import mlb_warehouse as wh
    df = _read(_pitcher_file(sport, str(season)))
    if df is None:
        return None
    out = {}
    for r in _records(df):
        aid = str(r.get("athlete_id"))
        ip = r.get("IP")
        outs = wh._ip_to_outs(float(ip)) if ip is not None else 0.0
        out.setdefault(aid, []).append((
            r.get("official_date"), outs, float(r.get("ER") or 0.0),
            float(r.get("K") or 0.0), float(r.get("BB") or 0.0),
            float(r.get("BF") or 0.0)))
    for aid in out:
        out[aid].sort(key=lambda t: (t[0] or ""))
    return out


def calib_gamelogs_bulk(role, season, sport="baseball_mlb"):
    """{athlete_id_str: [row dict, ...]} for outcome grading — carries the native
    stat columns + game_pk that r2_data.outcome_value reads (H/SO/TB/RBI for batters;
    K/ER/IP for pitchers). NOT the full ESPN shape (no opponent name/completed — the
    backtests don't read those). None if the season file is absent."""
    fpath = _pitcher_file if role == "pitcher" else _batter_file
    df = _read(fpath(sport, str(season)))
    if df is None:
        return None
    out = {}
    for r in _records(df):
        out.setdefault(str(r.get("athlete_id")), []).append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description="Local parquet mirror of the backtest warehouse.")
    ap.add_argument("--sync", action="store_true", help="pull Azure -> local parquet")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--refresh", action="store_true",
                    help="overwrite existing files (else skip; 2024/25 are immutable)")
    args = ap.parse_args()
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if args.sync:
        print(f"  syncing {args.sport} {seasons} -> {MIRROR_DIR}"
              f"  (refresh={args.refresh})")
        sync(args.sport, seasons, refresh=args.refresh)
        print("  done.")
    else:
        print(f"  mirror dir: {MIRROR_DIR}  enabled={enabled()}")
        print("  pass --sync to build it; set ODI_BACKTEST_MIRROR=1 to read from it.")


if __name__ == "__main__":
    main()
