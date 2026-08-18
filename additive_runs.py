"""Pure, SQL-free additive expected-runs helpers shared by the OFFLINE bake-off
(backtest_starters) and the LIVE path (mlb_starters.live_additive_runs, Tier A #1d).

WHY THIS EXISTS: fit == serve BY CONSTRUCTION. The bake-off validated an additive
runs model; the live path must reproduce the SAME number for the same game/as-of.
Rather than a second implementation (which would drift), both paths import THESE
functions and run them on the SAME as-of rows — the offline caller supplies bulk
warehouse series, the live caller supplies single-entity on-demand series, but the
projection/blend/bullpen math is identical because it is the same code.

IMPORTS: xera_lite is a dependency-free leaf (safe at module import). mlb_starters is
imported LAZILY inside the projector — mlb_starters imports THIS module for the live
path, so a top-level `import mlb_starters` here would create an import cycle that breaks
mlb_starters at boot (i.e. the flag-OFF path). No SQL, no import-time side effects.
"""

import xera_lite


def exp_ip(v, default=5.2, lo=3.5, hi=7.0):
    """As-of avg innings/start, defaulted + clamped to a sane starter range."""
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def feat_from_row(row, feature_keys):
    feats = {k: row.get(k) for k in feature_keys}
    if any(v is None for v in feats.values()):
        return None, None
    return feats, row.get("n_bbe")


def window_diff(old, new, feature_keys):
    """Trailing-window features = new cumulative MINUS old cumulative. xwOBAcon via
    sum/count, k9 via K/IP; any other key falls back to the new cumulative value.
    None if the window added no batted balls / innings."""
    out = {}
    for k in feature_keys:
        if k == "xwobacon":
            so = (old.get("xwobacon") or 0.0) * (old.get("n_bbe") or 0)
            sn = (new.get("xwobacon") or 0.0) * (new.get("n_bbe") or 0)
            dn = (new.get("n_bbe") or 0) - (old.get("n_bbe") or 0)
            if dn <= 0:
                return None
            out[k] = (sn - so) / dn
        elif k == "k9":
            ko = (old.get("k9") or 0.0) * (old.get("ip") or 0.0) / 9.0
            kn = (new.get("k9") or 0.0) * (new.get("ip") or 0.0) / 9.0
            dip = (new.get("ip") or 0.0) - (old.get("ip") or 0.0)
            if dip <= 0:
                return None
            out[k] = (kn - ko) / dip * 9.0
        else:
            out[k] = new.get(k)
    return out


def make_feat_getter(series, mode, feature_keys, n_starts=10, blend_k=200.0):
    """feat_getter(pid, date) -> (feats, n) under a windowing `mode`:
    'cumulative' (season-to-date), 'window' (last n_starts via differencing), or
    'blend' (current season-to-date blended with the prior-season final, weight
    n/(n+blend_k)). `series` = {entity_id: [rows sorted by as_of_date]} (bulk offline
    or single-entity live — same shape)."""
    def _row_idx(rows, d):
        exact = [i for i, r in enumerate(rows) if r["as_of_date"] == d]
        if exact:
            return exact[0]
        prev = [i for i, r in enumerate(rows) if r["as_of_date"] < d]
        return prev[-1] if prev else None

    def getter(pid, date):
        rows = series.get(str(pid))
        if not rows:
            return None, None
        idx = _row_idx(rows, str(date)[:10])
        if idx is None:
            return None, None
        cur = rows[idx]
        feats, n = feat_from_row(cur, feature_keys)
        if feats is None:
            return None, None
        if mode == "window":
            back = idx - n_starts
            if back >= 0 and rows[back]["season_bucket"] == cur["season_bucket"]:
                wf = window_diff(rows[back], cur, feature_keys)
                if wf:
                    return wf, (cur.get("n_bbe") or 0) - (rows[back].get("n_bbe") or 0)
            return feats, n                          # not enough history -> cumulative
        if mode == "blend":
            prior = [r for r in rows
                     if r["season_bucket"] == cur["season_bucket"] - 1]
            if prior:
                pf, _pn = feat_from_row(prior[-1], feature_keys)
                if pf:
                    w = (n / (n + blend_k)) if n else 0.0
                    return ({k: w * feats[k] + (1.0 - w) * pf[k]
                             for k in feature_keys}, n)
            return feats, n
        return feats, n                              # cumulative
    return getter


def make_bp_getter(bp_series, resolve_id, league_rp_era, league_bp,
                   fatigue_weight=0.0, fatigue_window=3,
                   fatigue_baseline_ip=3.3, fatigue_cap=0.5):
    """bp_getter(team_key, date) -> the team's as-of bullpen rate9 on the TOTAL-runs
    scale used by the additive label. Computed LEAGUE-RELATIVE:
        rate9 = league_bp * clamp(team_rp_era / league_rp_era, 0.5, 2.0)
    so the earned-only RP era is rescaled and the earned-vs-total offset cancels.
    Falls back to the flat league_bp when the team/date has no prior relief line.

    ``resolve_id`` is a CALLABLE team_key -> team_id (the key `bp_series` is keyed by):
    the offline bake-off passes ``abbr_to_id.get`` (abbr -> MLBAM id); the live path
    passes an identity/str (team_id -> team_id). Generalizing the old abbr_to_id dict
    to a callable is what lets both paths share this one function.

    BULLPEN FATIGUE (Batch A #13, INERT at fatigue_weight=0): a recently over-worked
    pen prices WORSE. trailing_ip = the cumulative RP ``ip`` snapshot at prev[-1] MINUS
    the snapshot ``fatigue_window`` rows earlier. Both snapshots are strictly-before the
    game date (the same leakage-safe curve the era ratio uses), so trailing_ip covers
    the ``fatigue_window`` game-date intervals ENDING one game-date before the start
    (it excludes the freshest outing, matching the era term's strictly-before lag). It
    is compared to an expected ``fatigue_baseline_ip`` per game-date; the league-
    relative rate9 is scaled by
        1 + fatigue_weight * clamp(trailing_ip/(baseline*window) - 1, -cap, +cap).
    The two snapshots must share a ``season_bucket``: cumulative relief ip resets each
    season (built per-season), so a cross-season difference would read as maximally
    rested — guarded (mirrors make_feat_getter's window-mode season check). fatigue_
    weight=0, too little history, a cross-season boundary, or era-only rows missing
    ``ip`` -> factor is exactly 1.0 -> byte-identical to the pre-fatigue term (fit ==
    serve preserved).
    """
    def getter(team_key, date):
        if not (bp_series and league_rp_era):
            return league_bp
        try:
            tid = resolve_id(team_key)
        except Exception:                    # a bad resolver fails open to league_bp
            return league_bp
        rows = bp_series.get(str(tid)) if tid is not None else None
        if not rows:
            return league_bp
        d = str(date)[:10]
        prev = [r for r in rows if r["as_of_date"] < d]   # strictly before
        if not prev:
            return league_bp
        ratio = max(0.5, min(2.0, prev[-1]["era"] / league_rp_era))
        rate9 = league_bp * ratio
        if fatigue_weight and fatigue_baseline_ip and len(prev) > fatigue_window:
            ref, back = prev[-1], prev[-1 - fatigue_window]
            ref_ip, back_ip = ref.get("ip"), back.get("ip")
            # SAME season only — cumulative ip resets per season (see docstring).
            same_season = ref.get("season_bucket") == back.get("season_bucket")
            if ref_ip is not None and back_ip is not None and same_season:
                expected = fatigue_baseline_ip * fatigue_window
                if expected > 0:
                    excess = max(-fatigue_cap, min(fatigue_cap,
                                 (ref_ip - back_ip) / expected - 1.0))
                    rate9 *= (1.0 + fatigue_weight * excess)
        return rate9
    return getter


def make_additive_projector(feat_getter, xera_model, league_bp, feature_keys,
                            bp_getter=None, run_env_fn=None):
    """project_fn(row) -> (home_runs, away_runs) via mlb_starters.expected_runs_
    additive: home batting faces the AWAY starter (+ away exp-IP + home-lineup offense
    a_off_faced), away batting faces the HOME starter. Starter rate9 from xera_lite on
    feat_getter(pid,date); league_bp fallback when a starter's feats are missing.

    Bullpen term: the flat league_bp constant when bp_getter is None (v1 behavior),
    else the pitching team's as-of bullpen rate9 from bp_getter(team_key, date) — the
    AWAY bullpen backs the away starter for home_runs, the HOME bullpen for away_runs.
    The row carries home_abbr/away_abbr (offline: team abbrs; live: team_ids) as the
    bullpen keys.

    RUN ENVIRONMENT (Batch A park/weather, INERT when run_env_fn is None): a single
    per-GAME multiplier from run_env_fn(row) scales BOTH teams' expected runs equally
    (the park/weather environment is shared). None (or a returned falsy/1.0) -> the
    expected_runs_additive default run_env=1.0 -> byte-identical to the pre-run_env
    projector. The caller composes park x weather into one centered-on-1.0 factor."""
    def _bp(team_key, date):
        return bp_getter(team_key, date) if bp_getter else league_bp

    def project(row):
        import mlb_starters                   # lazy: breaks the import cycle (see header)
        date = row["date"]
        af, an = feat_getter(row["away_sp"], date)
        hf, hn = feat_getter(row["home_sp"], date)
        away_rate9 = xera_lite.predict(af, xera_model, n_sample=an) if af else None
        home_rate9 = xera_lite.predict(hf, xera_model, n_sample=hn) if hf else None
        away_rate9 = away_rate9 if away_rate9 is not None else league_bp
        home_rate9 = home_rate9 if home_rate9 is not None else league_bp
        run_env = (run_env_fn(row) if run_env_fn else 1.0) or 1.0
        home_runs = mlb_starters.expected_runs_additive(
            away_rate9, _bp(row.get("away_abbr"), date), exp_ip(row.get("a_ip")),
            offense_factor=row.get("a_off_faced") or 1.0, run_env=run_env)
        away_runs = mlb_starters.expected_runs_additive(
            home_rate9, _bp(row.get("home_abbr"), date), exp_ip(row.get("h_ip")),
            offense_factor=row.get("h_off_faced") or 1.0, run_env=run_env)
        return home_runs, away_runs
    return project
