"""One-off edit script (ESPN-for-NBA retirement, phase B): retire the nba_source shim, the four
ODI_NBA_* fallback switches, and every NBA fall-through to ESPN in live analysis + grading."""
ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


# ── the shim's only real importer ────────────────────────────────────────────────────────
edit("book_line_calibration.py", [
    ("""        import nba_source
        import nba_warehouse""",
     """        import nba_stats_source as nba_source
        import nba_warehouse"""),
])

# ── ODI_NBA_LIVE_SDV ─────────────────────────────────────────────────────────────────────
edit("nba_stats_source.py", [
    ('''def live_from_sdv():
    """NBA live analysis (player history + team form) is served from the warehouse/sportsdataverse
    by default; kill-switch ODI_NBA_LIVE_SDV=0 reverts to the ESPN path (burn-in safety,
    independent of the grading switch ODI_NBA_GRADE_SDV)."""
    import os
    return os.environ.get("ODI_NBA_LIVE_SDV", "1").strip().lower() not in ("0", "false", "no")


''', ""),
])
edit("espn_client.py", [
    ('''def _nba_warehouse_history(player_name, prop_key, n, team_ids=None):
    """Warehouse-first NBA player history (the Phase-4 live cutover): return the
    get_player_stat_history contract dict from the sportsdataverse per-player-game facts, or
    None so the caller falls open to the ESPN path. Gated on nba_source.live_from_sdv() +
    SQL enabled + a mapped prop; DNP games are excluded (a bet voids on a DNP, never a 0).
    Never raises."""
    try:
        import nba_stats_source as nba_source
        if not nba_source.live_from_sdv():
            return None
        cols = nba_source.PROP_STAT_COLS.get(prop_key)''',
     '''def _nba_warehouse_history(player_name, prop_key, n, team_ids=None):
    """NBA player history from the stats.nba.com WAREHOUSE (the only NBA source — ESPN retired
    2026-10): the get_player_stat_history contract dict from the per-player-game facts, or None
    (SQL off / unmapped prop / no played games) -> the caller fails closed. Odds-feed spellings
    resolve via nba_names (inside player_history); DNP games are excluded (a bet voids on a DNP,
    never a 0). Never raises."""
    try:
        import nba_stats_source as nba_source
        cols = nba_source.PROP_STAT_COLS.get(prop_key)'''),
    ('''    # NBA live-input cutover (gated via nba_source.live_from_sdv, fail-open): serve the
    # recent-game history from the sportsdataverse WAREHOUSE (per-player-game box) when
    # enabled + resolvable, else fall through to the ESPN path below unchanged. Mirrors the
    # MLB flip; stamps source="nba_warehouse" for per-prediction auditability.
    if allow_warehouse and sport == "basketball":
        nba = _nba_warehouse_history(player_name, prop_key, n, team_ids=team_ids)
        if nba is not None:
            return nba''',
     '''    # NBA: WAREHOUSE-ONLY (ESPN retired 2026-10). A miss (unresolvable name even after the
    # nba_names rules, SQL off, no played games) fails CLOSED — empty history drops the prop as
    # no_history, never an ESPN-sourced value. Stamps source="nba_warehouse".
    if sport == "basketball":
        nba = _nba_warehouse_history(player_name, prop_key, n, team_ids=team_ids)
        return nba if nba is not None else result'''),
    ('''    use_sql = (db_store is not None and db_store.enabled()
               and sport in ("basketball", "football"))''',
     '''    use_sql = (db_store is not None and db_store.enabled()
               and sport == "football")'''),
])
edit("app.py", [
    ('''def _nba_live_warehouse():
    """NBA live path is on the stats.nba.com warehouse (ODI_NBA_LIVE_SDV, default on). Off ->
    the legacy ESPN team dim / schedules (burn-in fallback)."""
    try:
        import nba_stats_source
        return nba_stats_source.live_from_sdv()
    except Exception:
        return False


''', ""),
    ('''    if sport_key == "basketball_nba" and _nba_live_warehouse():
        # NBA: the stats.nba.com warehouse team dim (no ESPN).''',
     '''    if sport_key == "basketball_nba":
        # NBA: the stats.nba.com warehouse team dim (no ESPN).'''),
    ('''    if sport_key == "basketball_nba" and _nba_live_warehouse():
        # NBA: recent games off the nba_game spine''',
     '''    if sport_key == "basketball_nba":
        # NBA: recent games off the nba_game spine'''),
    ('''    if sport["key"] in ("baseball_mlb", "americanfootball_nfl") or (
            sport["key"] == "basketball_nba" and _nba_live_warehouse()):
        espn_teams = {}''',
     '''    if sport["key"] in ("baseball_mlb", "americanfootball_nfl", "basketball_nba"):
        espn_teams = {}'''),
    ('''                if (not (wh_home and wh_away) and sport["key"] == "basketball_nba"):
                    import nba_stats_source as nba_source
                    if nba_source.live_from_sdv():
                        import nba_team_stats
                        wh_home = nba_team_stats.team_stats(home, recent_n)
                        wh_away = nba_team_stats.team_stats(away, recent_n)''',
     '''                if (not (wh_home and wh_away) and sport["key"] == "basketball_nba"):
                    import nba_team_stats
                    wh_home = nba_team_stats.team_stats(home, recent_n)
                    wh_away = nba_team_stats.team_stats(away, recent_n)'''),
    ('''    # Team data. MLB (P6, warehouse), NFL (nflverse games spine) and NBA (stats.nba.com
    # warehouse, unless ODI_NBA_LIVE_SDV=0) resolve + build team stats without ESPN, so skip
    # the ESPN get_all_teams fetch for them. NHL still fetches from ESPN.''',
     '''    # Team data. MLB (P6, warehouse), NFL (nflverse games spine) and NBA (stats.nba.com
    # warehouse) resolve + build team stats without ESPN, so skip the ESPN get_all_teams fetch
    # for them. NHL still fetches from ESPN.'''),
])

# ── ODI_NBA_GRADE_SDV (+ the ESPN warm-up for NBA) ───────────────────────────────────────
edit("recalibration.py", [
    ('''_NBA_GRADE_FROM_WAREHOUSE_DEFAULT = True


def _nba_grade_from_warehouse():
    v = os.environ.get("ODI_NBA_GRADE_SDV")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return _NBA_GRADE_FROM_WAREHOUSE_DEFAULT

''', ""),
    ('''        if sport_key == "basketball_nba" and _nba_grade_from_warehouse():
            v = _resolve_nba_actual(prop_key, player, game_date, commence)
            if v is not None:
                return float(v)
            if _nba_dnp_row(player, game_date) is True:
                return None
        gamelog, by_date = _load_player_gamelog(espn_sport, espn_league, player)''',
     '''        if sport_key == "basketball_nba":
            # WAREHOUSE-ONLY (ESPN retired 2026-10): a miss — a not-yet-ingested game or a
            # confirmed DNP — stays pending; maintenance ingests the game and a later pass
            # grades it (a DNP voids via _nba_is_dnp in the stale-void path).
            v = _resolve_nba_actual(prop_key, player, game_date, commence)
            return float(v) if v is not None else None
        gamelog, by_date = _load_player_gamelog(espn_sport, espn_league, player)'''),
    ('''    if sport_key != "baseball_mlb" and len(by_player) > 1:''',
     '''    if sport_key not in ("baseball_mlb", "basketball_nba") and len(by_player) > 1:'''),
])

# ── ODI_NBA_TEAM_GRADE_SDV ───────────────────────────────────────────────────────────────
edit("game_results.py", [
    ('''def _nba_team_grade_sdv():
    """NBA team-market scores come from the stats.nba.com WAREHOUSE spine by default; kill-switch
    ODI_NBA_TEAM_GRADE_SDV=0 reverts to the ESPN scoreboard (mirrors the prop cutover's
    ODI_NBA_GRADE_SDV and the NFL team switch)."""
    return os.environ.get("ODI_NBA_TEAM_GRADE_SDV", "1").strip().lower() \\
        not in ("0", "false", "no")


''', ""),
    ('''    if sport_key == "basketball_nba" and _nba_team_grade_sdv():
        sc = _nba_final_score(home_team, away_team, commence_time, game_date)
        if sc is not None:
            return sc''',
     '''    if sport_key == "basketball_nba":
        # WAREHOUSE-ONLY (ESPN retired 2026-10): a miss stays pending until maintenance
        # ingests the game.
        return _nba_final_score(home_team, away_team, commence_time, game_date)'''),
])

# ── ODI_NBA_SWEEP_SDV ────────────────────────────────────────────────────────────────────
edit("backtest.py", [
    ('''def _nba_sweep_mode(espn_sport):
    """True when the NBA player-props sweep should grade off nba_warehouse (all 16 props,
    real actuals) instead of ESPN (3 props). Auto-on when basketball + the warehouse is
    enabled; kill-switch ODI_NBA_SWEEP_SDV=0 (mirrors recalibration._nba_grade_from_warehouse).
    Gating on enabled() keeps the ESPN path when SQL is off (tests)."""
    if espn_sport != "basketball":
        return False
    try:
        import nba_warehouse
        if not nba_warehouse.enabled():
            return False
    except Exception:
        return False
    return os.environ.get("ODI_NBA_SWEEP_SDV", "1").strip().lower() \\
        not in ("0", "false", "off", "no")''',
     '''def _nba_sweep_mode(espn_sport):
    """True for basketball: the NBA player-props sweep always grades off nba_warehouse (all 16
    props, real actuals; mirror-first, so it works offline without SQL). ESPN retired 2026-10."""
    return espn_sport == "basketball"'''),
])
print("ok")
