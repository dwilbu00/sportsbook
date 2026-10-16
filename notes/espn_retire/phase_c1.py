"""One-off edit script (ESPN-for-NBA retirement, phase C part 1): calibration join, refit player
pool, sweep layoff schedules + name resolution, and the ESPN defense lookups — warehouse only."""
ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


edit("book_line_calibration.py", [
    # the join: NBA never falls back to ESPN
    ('''    if espn_sport == "basketball":
        nba = _nba_grade_book_lines(book_lines)
        if nba is not None:
            return nba''',
     '''    if espn_sport == "basketball":
        # WAREHOUSE-ONLY (ESPN retired 2026-10): mirror-first actuals; [] when none.
        return _nba_grade_book_lines(book_lines) or []'''),
    # the grader: mirror-first, so no SQL gate (an offline box with the parquet mirror works)
    ('''    if not db_store.enabled():
        return None
    prop_cols = nba_source.PROP_STAT_COLS''',
     '''    prop_cols = nba_source.PROP_STAT_COLS'''),
    # main(): the ESPN defense lookup is inert for NBA (int team ids) — skip it for basketball
    ('''        team_defense, _, league_avg_def = _team_defense_lookup(espn_sport, espn_league)''',
     '''        team_defense, _, league_avg_def = (
            ({}, {}, None) if espn_sport == "basketball"            # NBA: no ESPN (retired)
            else _team_defense_lookup(espn_sport, espn_league))'''),
    ('''            sm_team_def, _, sm_league_avg = _team_defense_lookup(espn_sport, espn_league)''',
     '''            sm_team_def, _, sm_league_avg = (
                ({}, {}, None) if espn_sport == "basketball"
                else _team_defense_lookup(espn_sport, espn_league))'''),
])

edit("refit_calibration.py", [
    ('''from espn_cache import seed_athlete_id
from espn_client import list_season_athletes
''', ''),
    ('''    if espn_sport == "baseball":
        return _mlb_warehouse_defense_lookup(season_year=season_year)
    return _team_defense_lookup(espn_sport, espn_league, season_year=season_year)''',
     '''    if espn_sport == "baseball":
        return _mlb_warehouse_defense_lookup(season_year=season_year)
    if espn_sport == "basketball":
        # NBA: no ESPN (retired 2026-10); the raw points-allowed defense is retired for NBA
        # (pace/defense context lives in nba_prop_context) -> an empty, inert lookup.
        return {}, {}, None
    return _team_defense_lookup(espn_sport, espn_league, season_year=season_year)'''),
    # the NBA pool from warehouse minutes (same selection rule as the ESPN listing)
    ('''    The authoritative ESPN ids from the listing are pinned into the id cache
    (`seed_athlete_id`) so the broad name list resolves to the RIGHT players
    instead of `search_athlete`'s first name match. Returns a de-duplicated list
    of names, or [] on failure (refit_sport then aborts rather than fitting from
    the 18-star fallback).
    """
    if not season:
        season = datetime.now(timezone.utc).year
    try:
        athletes = list_season_athletes("basketball", "nba", season)
    except Exception as exc:
        print(f"  [warn] NBA player pool unavailable: {exc}")
        return []

    eligible = [a for a in athletes
                if (a.get("games") or 0) >= min_games
                and (a.get("minutes") or 0) > 0]
    eligible.sort(key=lambda a: a.get("minutes") or 0.0, reverse=True)

    names, seen = [], set()
    for a in eligible[:max_players]:
        name = a.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            seed_athlete_id("basketball", "nba", name, a.get("id"))
        except OSError:
            pass  # best effort; cached_athlete_id falls back to name search
        names.append(name)
    return names''',
     '''    Source = the stats.nba.com warehouse player-game corpus (mirror-first; ESPN retired
    2026-10). Names are returned in the warehouse's own spelling, so fetch_player_data's
    name_norm lookup resolves them exactly. Returns a de-duplicated list of names, or [] on
    failure (refit_sport then aborts rather than fitting from the 18-star fallback).
    """
    try:
        import nba_warehouse
        if not season:
            season = nba_warehouse._current_season()
        bulk = nba_warehouse.calib_gamelogs_bulk([int(season)])
    except Exception as exc:
        print(f"  [warn] NBA player pool unavailable: {exc}")
        return []
    ranked = []
    for nn, games in bulk.items():
        played = [g for g in games if g.get("season") == int(season) and g.get("played")
                  and (g.get("MIN") or 0) > 0]
        if len(played) >= min_games:
            ranked.append((sum(float(g["MIN"]) for g in played), nn))
    ranked.sort(reverse=True)
    return [nn.title() for _mins, nn in ranked[:max_players]]'''),
])

edit("backtest.py", [
    # sweep layoff schedules: warehouse team games for NBA (was an ESPN call with foreign ids)
    ('''    team_schedules_for_filter = {}
    if unique_team_ids:
        print(f"\\n=== Fetching team schedules for reliability filter "
              f"({len(unique_team_ids)} teams) ===")
        for tid in unique_team_ids:
            try:
                if use_warehouse:''',
     '''    team_schedules_for_filter = {}
    if unique_team_ids:
        print(f"\\n=== Fetching team schedules for reliability filter "
              f"({len(unique_team_ids)} teams) ===")
        nba_team_sched = None
        if nba_wh_mode:
            # NBA: the warehouse team-game rows (stats.nba.com team_id keys — what the
            # gamelog rows carry); the filter reads only the game dates. No ESPN.
            import nba_warehouse
            nba_team_sched = {}
            for r in nba_warehouse.team_games(
                    [season_year or nba_warehouse._current_season()]):
                nba_team_sched.setdefault(str(r.get("team_id")), []).append(
                    {"date": r.get("game_date")})
        for tid in unique_team_ids:
            try:
                if nba_team_sched is not None:
                    team_schedules_for_filter[tid] = nba_team_sched.get(tid, [])
                elif use_warehouse:'''),
    # fetch_player_data: resolve odds-feed spellings against the warehouse corpus
    ('''            import nba_stats_source as nba_source
            gl = _nba_bulk_for().get(nba_source._norm(name))''',
     '''            import nba_stats_source as nba_source
            _bulk = _nba_bulk_for()
            gl = _bulk.get(nba_source._norm(name))
            if not gl and _bulk:
                import nba_names
                if "r" not in _nba_bulk:
                    _nba_bulk["r"] = nba_names.Resolver(_bulk)
                gl = _bulk.get(_nba_bulk["r"].resolve(name) or "")'''),
])
print("ok")
