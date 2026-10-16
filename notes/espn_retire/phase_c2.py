"""One-off edit script (ESPN-for-NBA retirement, phase C part 2): matchup / coherence / props-odds
backtests and the CLI (main.py) on the stats.nba.com warehouse for NBA."""
ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


edit("backtest.py", [
    # matchup-mode run_backtest
    ('''    if use_warehouse:
        print("=== team-market inputs: StatsAPI warehouse (ESPN bypassed) ===")
        espn_teams, schedules = _warehouse_team_schedules([season_year])
    else:
        espn_teams = get_all_teams(espn_sport, espn_league)
        print(f"Loaded {len(espn_teams)} teams")''',
     '''    if use_warehouse:
        print("=== team-market inputs: StatsAPI warehouse (ESPN bypassed) ===")
        espn_teams, schedules = _warehouse_team_schedules([season_year])
    elif espn_sport == "basketball":
        print("=== team-market inputs: stats.nba.com warehouse (ESPN bypassed) ===")
        espn_teams, schedules = _nba_warehouse_team_schedules([season_year])
    else:
        espn_teams = get_all_teams(espn_sport, espn_league)
        print(f"Loaded {len(espn_teams)} teams")'''),
    # coherence
    ('''    if use_warehouse:
        espn_teams, schedules = _warehouse_team_schedules(seasons_list)
    else:
        espn_teams = get_all_teams(espn_sport, espn_league)
        schedules = {}
        for sy in seasons_list:
            for tid, gms in build_schedules(
                    espn_sport, espn_league, espn_teams, season_year=sy).items():
                schedules.setdefault(tid, []).extend(gms)
    lookup, _unm = _build_odds_lookup(store, espn_teams)''',
     '''    if use_warehouse:
        espn_teams, schedules = _warehouse_team_schedules(seasons_list)
    elif espn_sport == "basketball":
        espn_teams, schedules = _nba_warehouse_team_schedules(seasons_list, store)
    else:
        espn_teams = get_all_teams(espn_sport, espn_league)
        schedules = {}
        for sy in seasons_list:
            for tid, gms in build_schedules(
                    espn_sport, espn_league, espn_teams, season_year=sy).items():
                schedules.setdefault(tid, []).extend(gms)
    lookup, _unm = _build_odds_lookup(store, espn_teams)'''),
    # props-odds player series
    ('''        role = "pitcher" if str(prop_key).startswith("pitcher_") else "batter"
        gamelog = mlb_warehouse.get_calib_gamelog(str(resolved[0]), role) or []
    else:
        aid = cached_athlete_id(espn_sport, espn_league, name)''',
     '''        role = "pitcher" if str(prop_key).startswith("pitcher_") else "batter"
        gamelog = mlb_warehouse.get_calib_gamelog(str(resolved[0]), role) or []
    elif espn_sport == "basketball":
        return _nba_player_stat_series(name, prop_key)
    else:
        aid = cached_athlete_id(espn_sport, espn_league, name)'''),
    ('''def _player_stat_series(espn_sport, espn_league, name, prop_key):''',
     '''_NBA_SERIES = {}                                        # {"bulk": .., "resolver": ..}


def _nba_player_stat_series(name, prop_key):
    """NBA dated per-game values for a prop from the stats.nba.com warehouse corpus (mirror-first;
    the current + three prior seasons, the span of the stored NBA odds), PLAYED games only, combo
    props summed. Odds-feed spellings resolve via nba_names. ESPN retired 2026-10."""
    import nba_names
    import nba_stats_source as nba_source
    import nba_warehouse
    cols = nba_source.PROP_STAT_COLS.get(prop_key)
    if cols is None:
        return []
    cols = cols if isinstance(cols, tuple) else (cols,)
    if "bulk" not in _NBA_SERIES:
        cur = nba_warehouse._current_season()
        _NBA_SERIES["bulk"] = nba_warehouse.calib_gamelogs_bulk([cur - 3, cur - 2, cur - 1, cur])
        _NBA_SERIES["resolver"] = nba_names.Resolver(_NBA_SERIES["bulk"])
    bulk = _NBA_SERIES["bulk"]
    games = bulk.get(nba_source._norm(name)) or bulk.get(
        _NBA_SERIES["resolver"].resolve(name) or "") or []
    out = []
    for g in games:
        if not g.get("played") or not g.get("game_date"):
            continue
        out.append((str(g["game_date"])[:10], float(sum(float(g.get(c) or 0.0) for c in cols))))
    out.sort(key=lambda x: x[0])
    return out


def _player_stat_series(espn_sport, espn_league, name, prop_key):'''),
    ('''    via mlb_starters.resolve_mlbam_id), current-season scoped. NBA/NFL use the ESPN
    path. (ESPN was fully removed for MLB in P4.)''',
     '''    via mlb_starters.resolve_mlbam_id), current-season scoped. NBA reads the stats.nba.com
    warehouse (_nba_player_stat_series); NFL uses the ESPN path. (ESPN removed for MLB in P4,
    for NBA 2026-10.)'''),
    ('''    # P3c/P4/P6: MLB player gamelogs come from the StatsAPI warehouse; NBA/NFL stay on
    # ESPN. (ESPN was fully removed for MLB in P4.)
    use_warehouse = espn_sport == "baseball"
    if use_warehouse:
        print("=== props-odds player logs: StatsAPI warehouse (ESPN bypassed) ===")''',
     '''    # P3c/P4/P6: MLB player gamelogs come from the StatsAPI warehouse, NBA from the stats.nba.com
    # warehouse (_player_stat_series); NFL stays on ESPN.
    use_warehouse = espn_sport == "baseball"
    if use_warehouse:
        print("=== props-odds player logs: StatsAPI warehouse (ESPN bypassed) ===")
    elif espn_sport == "basketball":
        print("=== props-odds player logs: stats.nba.com warehouse (ESPN bypassed) ===")'''),
])

edit("main.py", [
    ('''    # Team data. MLB (P6): team resolution is warehouse-only (no ESPN get_all_teams);
    # NBA/NFL/NHL still fetch from ESPN.
    if sport_key == "baseball_mlb":
        espn_teams = {}
        print("\\n  MLB team resolution: StatsAPI warehouse (no ESPN).")''',
     '''    # Team data. MLB (P6) and NBA (stats.nba.com, 2026-10): team resolution is warehouse-only
    # (no ESPN get_all_teams); NFL/NHL still fetch from ESPN.
    if sport_key == "baseball_mlb":
        espn_teams = {}
        print("\\n  MLB team resolution: StatsAPI warehouse (no ESPN).")
    elif sport_key == "basketball_nba":
        espn_teams = {}
        print("\\n  NBA team resolution: stats.nba.com warehouse (no ESPN).")'''),
    ('''            # P4 team-market flip: prefer the StatsAPI warehouse (MLB-only,
            # env-gated); fall open to the ESPN build. Require both sides.
            wh_home = mlb_warehouse_team_stats(espn_sport, home, recent_n)
            wh_away = mlb_warehouse_team_stats(espn_sport, away, recent_n)
            if wh_home and wh_away:
                home_stats, away_stats = wh_home, wh_away''',
     '''            # P4 team-market flip: prefer the StatsAPI warehouse (MLB-only,
            # env-gated); fall open to the ESPN build. Require both sides.
            wh_home = mlb_warehouse_team_stats(espn_sport, home, recent_n)
            wh_away = mlb_warehouse_team_stats(espn_sport, away, recent_n)
            matchup_features = None
            if sport_key == "basketball_nba":
                # NBA: warehouse team form + the efficiency team model (same as the app);
                # no ESPN fallback — a miss skips the game's team markets.
                import nba_team_model
                import nba_team_stats
                wh_home = nba_team_stats.team_stats(home, recent_n)
                wh_away = nba_team_stats.team_stats(away, recent_n)
                if not (wh_home and wh_away):
                    print("    WARNING: no warehouse team form. Skipping.")
                    continue
                matchup_features = nba_team_model.build_matchup_features(
                    home, away, _et_game_date(raw_game.get("commence_time")))
            if wh_home and wh_away:
                home_stats, away_stats = wh_home, wh_away'''),
    ('''            # Cache per-team avg_allowed for opponent-defense weighting on props.
            team_defense[home_espn["display_name"]] = home_stats["recent"]["avg_allowed"]
            team_defense[away_espn["display_name"]] = away_stats["recent"]["avg_allowed"]''',
     '''            # Cache per-team avg_allowed for opponent-defense weighting on props (not NBA:
            # its raw points-allowed defense is retired — nba_prop_context replaces it).
            if sport_key != "basketball_nba":
                team_defense[home_espn["display_name"]] = home_stats["recent"]["avg_allowed"]
                team_defense[away_espn["display_name"]] = away_stats["recent"]["avg_allowed"]'''),
    ('''                ml_candidates = analyze_moneyline_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key)''',
     '''                ml_candidates = analyze_moneyline_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key, matchup_features=matchup_features)'''),
    ('''                spread_candidates = analyze_spreads_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key)''',
     '''                spread_candidates = analyze_spreads_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key, matchup_features=matchup_features)'''),
    ('''                totals_candidates = analyze_totals_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key)''',
     '''                totals_candidates = analyze_totals_value(game_odds, home_stats, away_stats, threshold, sport_key=sport_key, matchup_features=matchup_features)'''),
    ('''            else:
                event_teams = [find_team(espn_teams, tn)
                               for tn in (prop_data.get("home_team"),
                                          prop_data.get("away_team")) if tn]''',
     '''            else:
                # NBA: stats.nba.com ids via the warehouse team dim (what the namesake
                # tie-break compares); ESPN ids for the other sports.
                event_teams = [_resolve_team_dim(sport_key, tn, espn_teams)
                               if sport_key == "basketball_nba" else find_team(espn_teams, tn)
                               for tn in (prop_data.get("home_team"),
                                          prop_data.get("away_team")) if tn]'''),
    ('''    if sport_key == "baseball_mlb":
        try:
            import mlb_warehouse
            wh = mlb_warehouse.warehouse_find_team(name)
            if wh:
                return wh
        except Exception:
            pass
    return find_team(espn_teams, name)''',
     '''    if sport_key == "baseball_mlb":
        try:
            import mlb_warehouse
            wh = mlb_warehouse.warehouse_find_team(name)
            if wh:
                return wh
        except Exception:
            pass
    if sport_key == "basketball_nba":
        # NBA: the stats.nba.com team dim (id = str(team_id)); None -> the game is skipped.
        try:
            import nba_warehouse
            tid = nba_warehouse.resolve_team_id(name)
            if tid is not None:
                return {"id": str(tid), "display_name": name, "record": None,
                        "wins": None, "losses": None, "win_pct": None}
        except Exception:
            pass
        return None
    return find_team(espn_teams, name)


def _et_game_date(commence_time):
    """US-Eastern calendar date of an event commence (the NBA team model's as-of date)."""
    try:
        from pricing_common import et_local_date
        return et_local_date(commence_time) or str(commence_time or "")[:10]
    except Exception:
        return str(commence_time or "")[:10]'''),
])
print("ok")
