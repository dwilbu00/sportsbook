"""Local parquet mirror of the read-only historical warehouse, for FAST OFFLINE
backtesting.

All four backtest tools (r2_backtest, coherence_backtest, f5_backtest,
scenario_backtest) read through this BY DEFAULT when the mirror exists — one shared
local columnar store instead of every tool re-hitting Azure on every run. Set
`ODI_BACKTEST_MIRROR=0` to force the live Azure path. The CALIBRATION REFIT reads
Azure directly (it needs the full ESPN-shape gamelogs).

The parquet blobs are stored in Git LFS (see `.gitattributes`) so they travel across
machines via `git pull`, but `.lfsconfig` EXCLUDES `warehouse_mirror_data/` from LFS
downloads by default — so a plain clone (production Streamlit Cloud, CI) gets only
tiny pointer stubs, never the ~40MB of blobs. `_read()` treats those stubs as absent,
so the readers fall back to Azure automatically there. To materialize the mirror on a
dev machine, clear the exclude locally once then pull:
`git config lfs.fetchexclude "" && git lfs pull` (the inline `-X ""` form is rejected
by some git-lfs versions).

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

Backtests use the mirror BY DEFAULT and auto-build it on first run (each tool calls
`autobuild()`), so a manual sync is optional:
  python warehouse_mirror.py --sync --sport baseball_mlb --seasons 2024,2025,2026
To force the live Azure path instead:  export ODI_BACKTEST_MIRROR=0
"""
import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIRROR_DIR = os.environ.get(
    "ODI_MIRROR_DIR", os.path.join(_SCRIPT_DIR, "warehouse_mirror_data"))

# In-process memo for the SMALL, immutable-within-a-run parquets (mlb_game dim + team
# dim) that get re-read many times per backtest — e.g. the 30-team × per-season defense
# lookup calls _team_final_games ~90×, each otherwise re-reading + re-parsing mlb_game
# from disk. One parse per process instead. `_write` clears it so a --sync sees fresh
# data. Odds/prop parquets are NOT memoized (large, read once per slice).
_SMALL_MEMO = {}


def _memo(key, builder):
    if key not in _SMALL_MEMO:
        _SMALL_MEMO[key] = builder()
    return _SMALL_MEMO[key]

# Books the backtests read (DK-parity 'draftkings' also captures legacy NULL rows,
# applied in SQL at sync time so the parquet already includes them).
BOOKS = ("draftkings", "pinnacle", "fanduel")

# Spring / all-star / exhibition — excluded from calib gamelogs (matches
# mlb_warehouse._NON_REGULAR_GAME_TYPES / get_calib_gamelogs_bulk). Postseason is KEPT.
_NON_REGULAR_GAME_TYPES = ("S", "A", "E")


def flag_on():
    """Mirror reads are ON BY DEFAULT — set `ODI_BACKTEST_MIRROR` to a falsy value
    (0/false/no/off) to force the live Azure path instead. Regardless of whether the
    dir exists yet: backtest tools gate the auto-build `ensure()` on this (it creates
    the dir); readers gate on `enabled()` (this AND dir present)."""
    return str(os.environ.get("ODI_BACKTEST_MIRROR", "")).strip().lower() not in (
        "0", "false", "no", "off")


def enabled():
    """True iff mirror reads are on (default) AND the mirror dir exists (readers gate on
    this — no dir, e.g. production Streamlit Cloud, means the Azure path is used)."""
    return flag_on() and os.path.isdir(MIRROR_DIR)


def source_label():
    """Human label of where a cold cache build will read from (for backtest logging)."""
    return "mirror (parquet)" if enabled() else "Azure SQL (live)"


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


# A file that PASSED --verify is renamed base.parquet -> base_valid.parquet, so
# ensure() can trust it and skip re-verifying on every backtest run. Invariant: at
# most one of {base, _valid} exists per slice. _write (sync) always produces base and
# drops any stale _valid (data changed -> must re-verify).

def _valid_name(name):
    return name[:-len(".parquet")] + "_valid.parquet" if name.endswith(".parquet") \
        else name + "_valid"


def _is_valid(name):
    return os.path.exists(_path(_valid_name(name)))


def _mark_valid(name):
    """Rename base -> _valid after a passing verify (no-op if base absent)."""
    base, vp = _path(name), _path(_valid_name(name))
    if os.path.exists(base):
        os.replace(base, vp)


def _demote(name):
    """Rename _valid -> base after a FAILING verify (no-op if _valid absent)."""
    base, vp = _path(name), _path(_valid_name(name))
    if os.path.exists(vp):
        os.replace(vp, base)


def _is_real_parquet(path):
    """True iff `path` is an actual parquet file (magic bytes 'PAR1'), not a Git-LFS
    pointer. A pointer-only checkout (e.g. Streamlit Cloud, or a fresh clone before
    `git lfs pull`) leaves ~130-byte text stubs where the blobs go; treating those as
    absent makes every reader fall back to Azure automatically instead of crashing on
    a bad parquet read."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"PAR1"
    except OSError:
        return False


def _read(name):
    """Read a mirror parquet -> DataFrame, preferring the _valid (verified) copy, then
    the unverified base, else None (caller falls back to Azure). LFS-pointer stubs are
    treated as absent (see `_is_real_parquet`)."""
    import pandas as pd
    for n in (_valid_name(name), name):
        p = _path(n)
        if os.path.exists(p) and _is_real_parquet(p):
            return pd.read_parquet(p)
    return None


def _write(df, name):
    import pandas as pd
    os.makedirs(MIRROR_DIR, exist_ok=True)
    (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_parquet(
        _path(name), index=False)
    vp = _path(_valid_name(name))       # fresh data invalidates any prior verification
    if os.path.exists(vp):
        os.remove(vp)
    _SMALL_MEMO.clear()                  # a fresh write invalidates the small-file memo


def _team_file(sport, book, season):
    return f"team__{sport}__{book}__{season}.parquet"


def _prop_file(sport, book, season):
    return f"props__{sport}__{book}__{season}.parquet"


def _game_file(sport):
    return f"mlb_game__{sport}.parquet"


def _statcast_file(season):
    return f"statcast__baseball_mlb__{season}.parquet"


def _team_dim_file(sport):
    return f"team_dim__{sport}.parquet"


def available_seasons(sport, kind="team"):
    """Seasons (as strings, sorted) for which a mirror ODDS parquet exists for this
    sport across ANY book. `kind`: 'team' or 'props'. Empty list if the dir is absent
    or nothing matches. Used to serve an UNSCOPED (dates=None) read from the mirror's
    own season files instead of blindly probing 2019..now into Azure. Counts pointer
    stubs too (the per-season read still falls back to Azure if a file is a stub)."""
    prefix = f"{'team' if kind == 'team' else 'props'}__{sport}__"
    seasons = set()
    try:
        for fn in os.listdir(MIRROR_DIR):
            if not (fn.startswith(prefix) and fn.endswith(".parquet")):
                continue
            stem = fn[:-len(".parquet")]
            if stem.endswith("_valid"):
                stem = stem[:-len("_valid")]
            seg = stem.rsplit("__", 1)
            if len(seg) == 2 and seg[1].isdigit():
                seasons.add(seg[1])
    except OSError:
        pass
    return sorted(seasons)


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
        return (not refresh) and (_is_valid(name) or os.path.exists(_path(name)))

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
    # The game/pitcher/batter fact tables are MLB-ONLY (mlb_game, mlb_pitcher_game,
    # mlb_batter_game). Without this guard a --sport nba/nfl sync would dump MLB data
    # into nba/nfl-named files (mlb_game__basketball_nba.parquet etc.). NBA/NFL fact
    # mirrors await their own stats warehouse ([[wishlist]]).
    if not str(sport).startswith("baseball"):
        if verbose:
            print(f"  [facts] skipped — {sport} has no MLB-style fact tables.")
        return
    import mlb_warehouse as wh
    import db_store
    from sqlalchemy import select as _select
    eng = db_store.get_engine()
    g = wh.mlb_game

    # mlb_game (all seasons in one file — a few thousand rows). game_date/status/
    # detailed_state added so the mirror can replicate _team_final_games' genuine-final
    # filter + return game_date (the defense lookup + calib gamelogs key on it).
    gf = _game_file(sport)
    if refresh or not (_is_valid(gf) or os.path.exists(_path(gf))):
        with eng.connect() as conn:
            rows = conn.execute(_select(
                g.c.game_pk, g.c.game_date, g.c.official_date, g.c.season,
                g.c.game_type, g.c.status, g.c.detailed_state,
                g.c.home_score, g.c.away_score, g.c.home_score_f5, g.c.away_score_f5,
                g.c.home_team_id, g.c.away_team_id)).fetchall()
        _write([dict(r._mapping) for r in rows], gf)
        if verbose:
            print(f"  mlb_game {'':<11} {len(rows):>7,} rows -> {gf}")

    # team dim (team_id -> name), ~30 rows — for opponent/venue NAME resolution in the
    # calib gamelogs + defense lookup (mirror of mlb_warehouse._team_name_map).
    tdf = _team_dim_file(sport)
    if refresh or not (_is_valid(tdf) or os.path.exists(_path(tdf))):
        t = wh.mlb_team
        with eng.connect() as conn:
            rows = conn.execute(_select(t.c.team_id, t.c.name)).fetchall()
        _write([dict(r._mapping) for r in rows], tdf)
        if verbose:
            print(f"  team_dim {'':<11} {len(rows):>7,} rows -> {tdf}")

    # Fact tables: join mlb_game for official_date + game_type, scope by
    # season_bucket EXACTLY as mlb_warehouse._game_log_bulk / _pitcher_game_index do
    # (NOT mlb_game.season) so the mirror indexes match. game_type is stored so
    # calib_gamelogs_bulk can exclude S/A/E (get_calib_gamelogs_bulk does; the pitcher
    # as-of index does NOT — each reader applies its own filter).
    # Mirror the FULL Azure stat set (pinned to mlb_warehouse's own tuples so it can't
    # drift) — a reduced set silently broke method D, which needs AB ("per-AB hit rate
    # + expected AB"): the batter facts previously carried only H/SO/TB/RBI, so
    # project_distributional returned None on mirror gamelogs and D was excluded. Full
    # set = the mirror gamelog is byte-shape-identical to get_calib_gamelog. team_id
    # added so the mirror can reconstruct opponent/is_home for the ESPN-shape readers.
    for tbl, mk_file, stat_cols in (
            (wh.mlb_pitcher_game, _pitcher_file,
             ("team_id",) + tuple(wh._PITCHER_GAME_STATS)),
            (wh.mlb_batter_game, _batter_file,
             ("team_id",) + tuple(wh._BATTER_GAME_STATS))):
        for s in seasons:
            f = mk_file(sport, s)
            if (not refresh) and (_is_valid(f) or os.path.exists(_path(f))):
                continue
            joined = tbl.join(g, tbl.c.game_pk == g.c.game_pk)
            cols = ([tbl.c.athlete_id, tbl.c.game_pk, tbl.c.season_bucket,
                     g.c.official_date, g.c.game_type] + [tbl.c[c] for c in stat_cols])
            with eng.connect() as conn:
                rows = conn.execute(
                    _select(*cols).select_from(joined)
                    .where(tbl.c.season_bucket == int(s))).fetchall()
            _write([dict(r._mapping) for r in rows], f)
            if verbose:
                print(f"  {mk_file(sport, s):<28} {len(rows):>7,} rows")


# ── Statcast (savant) mirror — OPT-IN (heavy: ~1 row/pitch). NOT part of the default
#    fact sync so a routine backtest autobuild doesn't pull ~700k rows/season it never
#    uses; build explicitly (`warehouse_mirror.py --statcast`). savant_history.load_days
#    reads it mirror-first when present, else Azure. ────────────────────────────────

def sync_statcast(seasons, refresh=False, verbose=True):
    """Mirror the Azure statcast_pitch rows to season-keyed parquet (one file per
    season, the exact savant_history.load_days shape). Heavy + opt-in. Existing files
    are skipped unless refresh (2024/25 immutable; refresh the live season)."""
    import db_store
    db_store.promote_secrets_from_toml()
    import savant_history as sh
    for s in [str(x) for x in seasons]:
        f = _statcast_file(s)
        if (not refresh) and (_is_valid(f) or os.path.exists(_path(f))):
            if verbose:
                print(f"  statcast {s}  (exists, skip)")
            continue
        rows = sh.load_days(f"{s}-01-01", f"{s}-12-31")   # Azure read (the sync step)
        _write(rows, f)
        if verbose:
            print(f"  statcast {s} {len(rows):>9,} pitches -> {f}")


def statcast_days(start, end):
    """savant_history.load_days served from the parquet mirror: all statcast rows in
    [start, end] (YYYY-MM-DD, inclusive), same shape/order as the Azure reader. Returns
    None if ANY season the range spans has no mirror file (caller falls back to Azure)."""
    import pandas as pd
    seasons = sorted({str(start)[:4], str(end)[:4]})
    # Fill any interior seasons (a range never realistically spans >1, but be safe).
    if len(seasons) == 2:
        seasons = [str(y) for y in range(int(seasons[0]), int(seasons[1]) + 1)]
    frames = []
    for s in seasons:
        df = _read(_statcast_file(s))
        if df is None:
            return None
        frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if all_df.empty or "game_date" not in all_df.columns:
        return []
    m = (all_df["game_date"] >= str(start)) & (all_df["game_date"] <= str(end))
    return _records(all_df[m].sort_values("game_date"))


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


def _odds_lines(kind_file, sport, dates, date_from, date_to, bookmaker,
                only_early=False, exclude_early=False, snapshot_source=None):
    """Shared team/prop reader: concat the season×book parquet(s), apply the same
    date + source filter the SQL reader would. Returns None if ANY needed season file
    is missing (so the caller falls back to Azure for the whole call).

    Source (snapshot-window) filtering mirrors db_store: `snapshot_source` keeps that
    exact window (early_12h/early_4h/closing); legacy only_early/exclude_early key on
    'backfill_early'. If a source filter is requested but the mirror parquet predates
    the `source` column (stale mirror), return None → fall back to Azure + signal a
    re-sync is needed."""
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
    if all_df.empty or "game_date" not in all_df.columns:
        return []                             # mirrored but 0 rows (e.g. wrong sport key)
    if dates:
        all_df = all_df[all_df["game_date"].isin([str(d) for d in dates])]
    else:
        if date_from is not None:
            all_df = all_df[all_df["game_date"] >= str(date_from)]
        if date_to is not None:
            all_df = all_df[all_df["game_date"] <= str(date_to)]
    if only_early or exclude_early or snapshot_source is not None:
        if "source" not in all_df.columns:
            return None                       # stale mirror (pre-source) -> re-sync/Azure
        if snapshot_source is not None:
            all_df = all_df[all_df["source"] == snapshot_source]
        elif only_early:
            all_df = all_df[all_df["source"] == "backfill_early"]
        elif exclude_early:
            all_df = all_df[all_df["source"].isna()
                            | (all_df["source"] != "backfill_early")]
    return _records(all_df)


def team_market_lines(sport, dates=None, date_from=None, date_to=None,
                      only_early=False, bookmaker="draftkings", snapshot_source=None):
    return _odds_lines(_team_file, sport, dates, date_from, date_to, bookmaker,
                       only_early=only_early, snapshot_source=snapshot_source)


def player_prop_lines(sport, dates=None, date_from=None, date_to=None,
                      exclude_early=False, only_early=False, prop_keys=None,
                      bookmaker="draftkings", snapshot_source=None):
    rows = _odds_lines(_prop_file, sport, dates, date_from, date_to, bookmaker,
                       only_early=only_early, exclude_early=exclude_early,
                       snapshot_source=snapshot_source)
    if rows is not None and prop_keys:
        keys = set(prop_keys)
        rows = [r for r in rows if r.get("prop_key") in keys]
    return rows


def _game_df(sport):
    return _memo(("game_df", sport), lambda: _read(_game_file(sport)))


def _game_records(sport):
    """Memoized list-of-dicts of the mlb_game dim (re-read ~90× by the defense lookup;
    parse once). None if the parquet is absent."""
    def _build():
        df = _game_df(sport)
        return _records(df) if df is not None else None
    return _memo(("game_records", sport), _build)


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
    return _memo(("cg_bulk", role, str(season), sport),
                 lambda: _calib_gamelogs_bulk(role, season, sport))


def _calib_gamelogs_bulk(role, season, sport="baseball_mlb"):
    """{athlete_id_str: [row dict, ...]} for outcome grading — carries the native
    stat columns + game_pk that r2_data.outcome_value reads (H/SO/TB/RBI for batters;
    K/ER/IP for pitchers). NOT the full ESPN shape (no opponent name/completed — the
    backtests don't read those). EXCLUDES spring/all-star/exhibition (S/A/E) to match
    mlb_warehouse.get_calib_gamelogs_bulk. None if the season file is absent."""
    fpath = _pitcher_file if role == "pitcher" else _batter_file
    df = _read(fpath(sport, str(season)))
    if df is None:
        return None
    out = {}
    for r in _records(df):
        if r.get("game_type") in _NON_REGULAR_GAME_TYPES:
            continue
        out.setdefault(str(r.get("athlete_id")), []).append(r)
    return out


def calib_gamelogs_bulk_full(role, season, sport="baseball_mlb"):
    return _memo(("cg_full", role, str(season), sport),
                 lambda: _calib_gamelogs_bulk_full(role, season, sport))


def _calib_gamelogs_bulk_full(role, season, sport="baseball_mlb"):
    """Full-shape variant of calib_gamelogs_bulk for the CALIBRATION actuals join
    (book_line_calibration.join_book_lines_to_actuals). Same rows PLUS ``game_date``
    (from official_date) and ``completed``=True — the only fields the join reads beyond
    the stat columns + game_pk. It intentionally omits opponent/is_home: the calibration
    fit takes game context from _attach_gamecontext (StatsAPI), NOT from the gamelog,
    and _match_rows_to_gamelog never reads them (matching is game_pk-dominant, date is a
    rare fallback). Excludes S/A/E to match mlb_warehouse.get_calib_gamelogs_bulk. None
    if the season file is absent (caller falls back to Azure)."""
    base = calib_gamelogs_bulk(role, season, sport)
    if base is None:
        return None
    out = {}
    for aid, rows in base.items():
        logs = []
        for r in rows:
            g = dict(r)
            if g.get("game_date") is None:
                g["game_date"] = r.get("official_date")
            g["completed"] = True
            logs.append(g)
        out[aid] = logs
    return out


def _build_team_name_map(sport):
    df = _read(_team_dim_file(sport))
    if df is None:
        return None
    return {str(r.get("team_id")): r.get("name") for r in _records(df)}


def team_name_map(sport="baseball_mlb"):
    """{team_id(str): name} from the mirrored team dim (mirror of mlb_warehouse.
    _team_name_map). None if the team-dim parquet is absent (caller -> Azure). Memoized;
    returns a fresh copy so callers may mutate freely."""
    cached = _memo(("team_name_map", sport), lambda: _build_team_name_map(sport))
    return dict(cached) if cached is not None else None


def team_final_games(team_id, as_of_date=None, season=None, limit=None,
                     sport="baseball_mlb"):
    """Mirror of mlb_warehouse._team_final_games (genuine-final games for a team, most-
    recent-first, {date,home_team,away_team,home_score,away_score,total_score,game_pk}).
    Replicates the Final + genuine-final(detailed_state) + regular/postseason filter
    from the mlb_game parquet + team dim. None if the parquet predates the game_date/
    status/detailed_state columns (stale mirror -> Azure)."""
    df = _game_df(sport)
    names = team_name_map(sport)
    if df is None or names is None:
        return None
    need = ("game_date", "status", "detailed_state", "home_team_id", "away_team_id",
            "home_score", "away_score", "game_type", "official_date", "season")
    if any(c not in df.columns for c in need):
        return None
    import mlb_starters
    non_final = [b.lower() for b in mlb_starters._NON_FINAL_DETAILED]
    tid = str(team_id)
    out = []
    for r in _game_records(sport):      # memoized parse (defense lookup calls this ~90×)
        if tid not in (str(r.get("home_team_id")), str(r.get("away_team_id"))):
            continue
        if r.get("status") != "Final":
            continue
        det = r.get("detailed_state")
        if det is not None and any(b in str(det).lower() for b in non_final):
            continue
        if r.get("game_type") in _NON_REGULAR_GAME_TYPES:
            continue
        if r.get("home_score") is None or r.get("away_score") is None:
            continue
        if season is not None and r.get("season") != int(season):
            continue
        if as_of_date is not None and str(r.get("official_date")) >= str(as_of_date):
            continue
        try:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
        except (TypeError, ValueError):
            continue
        out.append({"date": r.get("game_date"),
                    "home_team": names.get(str(r.get("home_team_id"))),
                    "away_team": names.get(str(r.get("away_team_id"))),
                    "home_score": hs, "away_score": as_, "total_score": hs + as_,
                    "game_pk": r.get("game_pk")})
    out.sort(key=lambda d: (d["date"] or "", d["game_pk"] or 0), reverse=True)
    return out[:int(limit)] if limit else out


def calib_gamelogs_bulk_espn(role, season, sport="baseball_mlb"):
    return _memo(("cg_espn", role, str(season), sport),
                 lambda: _calib_gamelogs_bulk_espn(role, season, sport))


def _calib_gamelogs_bulk_espn(role, season, sport="baseball_mlb"):
    """Full ESPN-shape bulk gamelogs (adds opponent NAME + is_home + game_date +
    completed) — mirror of mlb_warehouse.get_calib_gamelogs_bulk, for the SYNTHETIC
    sweep (which uses opponent for opp-defense/park weighting). Needs the fact parquet's
    team_id + the mlb_game (game_date/team ids) + team dim; None if any is absent/stale
    so the caller falls back to Azure (batter mirrors predating team_id -> None)."""
    fpath = _pitcher_file if role == "pitcher" else _batter_file
    fdf = _read(fpath(sport, str(season)))
    gdf = _game_df(sport)
    names = team_name_map(sport)
    if fdf is None or gdf is None or names is None:
        return None
    if "team_id" not in fdf.columns:
        return None
    if any(c not in gdf.columns for c in ("game_pk", "game_date", "home_team_id",
                                          "away_team_id")):
        return None
    gmeta = {r.get("game_pk"): (r.get("game_date"), str(r.get("home_team_id")),
                                str(r.get("away_team_id"))) for r in _game_records(sport)}
    out = {}
    for r in _records(fdf):
        if r.get("game_type") in _NON_REGULAR_GAME_TYPES:
            continue
        g = dict(r)
        tid = str(r.get("team_id"))
        meta = gmeta.get(r.get("game_pk"))
        if meta:
            gd, home_id, away_id = meta
            g["game_date"] = gd
            g["is_home"] = (tid == home_id)
            g["opponent"] = names.get(away_id if tid == home_id else home_id)
        elif g.get("game_date") is None:
            g["game_date"] = r.get("official_date")
        g["completed"] = True
        out.setdefault(str(r.get("athlete_id")), []).append(g)
    return out


def calib_gamelog(mlb_player_id, role, season, sport="baseball_mlb"):
    """Singular full ESPN-shape gamelog (mirror of mlb_warehouse.get_calib_gamelog).
    None if the mirror can't serve it (-> Azure); [] if the player simply has no games
    that season (matching the Azure reader)."""
    bulk = calib_gamelogs_bulk_espn(role, season, sport)
    if bulk is None:
        return None
    return bulk.get(str(mlb_player_id), [])


def _row_key(d):
    return tuple(sorted((k, v) for k, v in d.items()))


def _rows_eq(mrows, arows):
    return {_row_key(r) for r in (mrows or [])} == {_row_key(r) for r in (arows or [])}


def _calib_eq(role, season):
    """Functional parity of the outcome index: same (athlete, game_pk) keys + same
    stat-column values as get_calib_gamelogs_bulk (the extra ESPN fields the mirror
    drops aren't read by outcome_value)."""
    import mlb_warehouse as wh
    m = calib_gamelogs_bulk(role, season) or {}
    a = wh.get_calib_gamelogs_bulk(role, int(season)) or {}
    cols = (("IP", "ER", "K", "BB", "BF") if role == "pitcher"
            else ("H", "SO", "TB", "RBI"))

    def norm(d):
        out = {}
        for aid, games in d.items():
            for g in games:
                gpk = g.get("game_pk")
                if gpk is None:
                    continue
                out[(str(aid), int(gpk))] = tuple(g.get(c) for c in cols)
        return out
    return norm(m) == norm(a)


def verify(sport, seasons, verbose=True):
    """PARITY self-check (needs Azure): compare each mirror parquet to the live
    db_store / r2_data / mlb_warehouse reader on REAL data. Each file that passes ALL
    its checks is renamed base -> `_valid` (so ensure() trusts it and never re-verifies
    it); a failing file is demoted `_valid` -> base. Returns True iff every file passed.
    'mirror == Azure on your actual warehouse'."""
    import db_store
    db_store.promote_secrets_from_toml()
    import r2_data
    import mlb_warehouse as wh
    seasons = [str(x) for x in seasons]
    file_ok = {}

    def record(f, passed):
        file_ok[f] = file_ok.get(f, True) and passed

    # Odds: one file per (season, book).
    for s in seasons:
        df, dt = f"{s}-01-01", f"{s}-12-31"
        for book in BOOKS:
            tf = _team_file(sport, book, s)
            record(tf, _rows_eq(
                team_market_lines(sport, date_from=df, date_to=dt, bookmaker=book),
                db_store.team_market_lines(sport, date_from=df, date_to=dt, bookmaker=book)))
            pf = _prop_file(sport, book, s)
            record(pf, _rows_eq(
                player_prop_lines(sport, date_from=df, date_to=dt, bookmaker=book),
                db_store.player_prop_lines(sport, date_from=df, date_to=dt, bookmaker=book)))

    # Fact indexes: force the mirror OFF (=0) so r2_data hits Azure for the baseline
    # (the mirror is default-on, so popping the var would NOT disable it).
    gf = _game_file(sport)
    _saved = os.environ.get("ODI_BACKTEST_MIRROR")
    os.environ["ODI_BACKTEST_MIRROR"] = "0"
    try:
        record(gf, (build_team_scores_index() or {}) == r2_data.build_team_scores_index())
        record(gf, (build_f5_scores_index() or {}) == r2_data.build_f5_scores_index())
        record(gf, (build_team_finals_index() or {}) == r2_data.build_team_finals_index())
        for s in seasons:
            apix = {str(a): g for a, g in (wh._pitcher_game_index(int(s)) or {}).items()}
            record(_pitcher_file(sport, s), (pitcher_game_index(int(s)) or {}) == apix)
            record(_pitcher_file(sport, s), _calib_eq("pitcher", s))
            record(_batter_file(sport, s), _calib_eq("batter", s))
    finally:
        if _saved is None:
            os.environ.pop("ODI_BACKTEST_MIRROR", None)
        else:
            os.environ["ODI_BACKTEST_MIRROR"] = _saved

    all_ok = True
    for f in sorted(file_ok):
        passed = file_ok[f]
        all_ok = all_ok and passed
        (_mark_valid if passed else _demote)(f)
        if verbose:
            print(f"  [{'PASS -> _valid' if passed else 'FAIL -> demoted'}] {f}")
    print(f"  VERIFY: {'ALL PASS' if all_ok else 'MISMATCH — do not rely on the mirror'}")
    return all_ok


def mark_valid_trusted(sport, seasons, verbose=True):
    """OFFLINE trust-mark: rename every PRESENT real-parquet needed file base -> _valid
    WITHOUT the Azure parity check. For when the mirror is already trusted — it was
    parity-verified elsewhere and/or validated end-to-end by the calibration runs — and
    the full ``verify()`` Azure parity (esp. the large per-book prop reads) is
    prohibitively slow on a low-DTU tier, silently re-run by ``ensure`` every backtest.
    Skips LFS pointer stubs (never a real file) and already-_valid slices. Zero Azure.
    Returns the count marked. ⚠ Trust-only: run a real ``--verify`` at least once after a
    fresh ``--sync`` if you have DTU headroom; use this to (re)establish the _valid markers
    a slow/flaky verify passed but didn't persist, or to skip re-verifying immutable data."""
    seasons = [str(x) for x in seasons]
    marked = skipped = already = 0
    for f in _needed_files(sport, seasons):
        if _is_valid(f):
            already += 1
            continue
        base = _path(f)
        if os.path.exists(base) and _is_real_parquet(base):
            _mark_valid(f)
            marked += 1
            if verbose:
                print(f"  [marked _valid] {f}")
        else:
            skipped += 1
            if verbose:
                print(f"  [SKIP missing/stub] {f}")
    print(f"  MARK-VALID (trusted, no Azure): {marked} marked, {already} already valid, "
          f"{skipped} missing/stub.")
    return marked


def _needed_files(sport, seasons):
    """Logical filenames the backtest tools read for (sport, seasons)."""
    files = [_game_file(sport)]
    for s in [str(x) for x in seasons]:
        for book in BOOKS:
            files += [_team_file(sport, book, s), _prop_file(sport, book, s)]
        files += [_pitcher_file(sport, s), _batter_file(sport, s)]
    return files


def ensure(sport, seasons, refresh=False, verbose=False):
    """Make the mirror ready for (sport, seasons), called by each backtest tool when
    the mirror is on (the default; ODI_BACKTEST_MIRROR=0 disables). FAST PATH: if every needed file is already `_valid`, do
    nothing (no Azure, no verify). Otherwise sync missing files + verify (which marks
    passing files `_valid`), so verification happens ONCE per file, not per run.
    refresh=True re-syncs + re-verifies everything. Needs Azure to build/verify; on any
    failure (e.g. no DB) it leaves existing files as-is and returns False — the readers
    then fall back to Azure per call. Never raises."""
    seasons = [str(x) for x in seasons]
    if not refresh and all(_is_valid(f) for f in _needed_files(sport, seasons)):
        return True                                   # everything validated -> instant
    try:
        sync(sport, seasons, refresh=refresh, verbose=verbose)
        return verify(sport, seasons, verbose=verbose)
    except Exception as exc:                           # no Azure / transient -> leave as-is
        if verbose:
            print(f"  [warehouse_mirror] ensure() skipped (Azure unavailable?): {exc}")
        return False


def autobuild(sport, seasons, refresh=False, verbose=True):
    """One-liner for backtest tools: if the mirror flag is on, ensure the mirror is
    built + validated for (sport, seasons) (fast no-op once everything is `_valid`);
    do nothing if the flag is off. Never raises."""
    try:
        if flag_on():
            return ensure(sport, seasons, refresh=refresh, verbose=verbose)
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser(description="Local parquet mirror of the backtest warehouse.")
    ap.add_argument("--sync", action="store_true", help="pull Azure -> local parquet")
    ap.add_argument("--verify", action="store_true",
                    help="parity-check mirror vs Azure on real data (run after --sync)")
    ap.add_argument("--mark-valid", action="store_true",
                    help="OFFLINE: mark present real-parquet files _valid WITHOUT Azure "
                         "parity (trust the mirror — use when verify passed but didn't "
                         "persist, or the low-DTU Azure parity is too slow). No Azure.")
    ap.add_argument("--sport", default="baseball_mlb")
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--refresh", action="store_true",
                    help="overwrite existing files (else skip; 2024/25 are immutable)")
    ap.add_argument("--statcast", action="store_true",
                    help="ALSO mirror statcast_pitch to season parquet (heavy, opt-in) "
                         "so method-D/xBA + platoon refits read savant 0 DTU.")
    ap.add_argument("--timeout", type=int, default=600,
                    help="SQL query timeout (s) for the bulk pulls — 60s (live default) "
                         "intermittently times out ~200k-row prop reads on a low-DTU "
                         "tier under throttle. Default 600.")
    args = ap.parse_args()
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    # Raise the query timeout for the bulk reads, then reset any cached engine so it
    # rebuilds with it (get_engine reads SQL_TIMEOUT at build time).
    import os as _os
    _os.environ["SQL_TIMEOUT"] = str(args.timeout)
    try:
        import db_store as _dbs
        _dbs.configure_engine(None)
    except Exception:
        pass
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if args.sync:
        print(f"  syncing {args.sport} {seasons} -> {MIRROR_DIR}"
              f"  (refresh={args.refresh})")
        sync(args.sport, seasons, refresh=args.refresh)
        print("  done.")
    if args.statcast:
        print(f"  syncing statcast {seasons} -> {MIRROR_DIR} (refresh={args.refresh})")
        sync_statcast(seasons, refresh=args.refresh)
        print("  done.")
    if args.verify:
        # verify() reads mirror parquet directly (no read-flag needed) and compares
        # to the live Azure readers.
        print(f"  verifying mirror vs Azure ({args.sport} {seasons})...")
        verify(args.sport, seasons)
    if args.mark_valid:
        print(f"  trust-marking mirror _valid ({args.sport} {seasons}) — no Azure...")
        mark_valid_trusted(args.sport, seasons)
    if not (args.sync or args.verify or args.statcast or args.mark_valid):
        print(f"  mirror dir: {MIRROR_DIR}  enabled={enabled()}")
        print("  --sync to build · --verify to parity-check · reads are ON by default "
              "(ODI_BACKTEST_MIRROR=0 forces Azure) · backtests auto-build.")


if __name__ == "__main__":
    main()
