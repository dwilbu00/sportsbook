"""One-off edit script (ESPN-for-NBA retirement, phase A): route every NBA player-name lookup
through nba_names.Resolver. Kept with the task notes per the repo's session-file convention."""
ROOT = r"c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/"


def edit(rel, pairs):
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:80])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


# nba_warehouse: cached resolver over the player dim; player_history + resolve_prop_ids use it
edit("nba_warehouse.py", [
    ('''def player_history(player_name, cols, n=20, as_of_date=None, team_ids=None):''',
     '''_NAME_RESOLVER = {}                                   # {"ts": t, "r": nba_names.Resolver}


def name_resolver(ttl=1800):
    """nba_names.Resolver over every nba_player name_norm (cached) — maps odds-feed spellings
    ("Jimmy Butler", "Nicolas Claxton") onto the warehouse's. None when disabled / on error."""
    hit = _NAME_RESOLVER.get("r")
    if hit is not None and (time.time() - _NAME_RESOLVER.get("ts", 0)) < ttl:
        return hit
    if not enabled():
        return None
    try:
        import nba_names
        with db_store.get_engine().connect() as conn:
            known = [r[0] for r in conn.execute(select(nba_player.c.name_norm)) if r[0]]
    except Exception:
        return None
    r = nba_names.Resolver(known)
    _NAME_RESOLVER.update({"ts": time.time(), "r": r})
    return r


def player_history(player_name, cols, n=20, as_of_date=None, team_ids=None):'''),
    ('''    cols = cols if isinstance(cols, tuple) else (cols,)
    try:
        nn = db_store.normalize_name(player_name)
        if not nn:
            return []
        pg, pl = nba_player_game, nba_player''',
     '''    cols = cols if isinstance(cols, tuple) else (cols,)
    try:
        nn = db_store.normalize_name(player_name)
        if not nn:
            return []
        res = name_resolver()
        nn = (res.resolve(player_name) if res is not None else None) or nn
        pg, pl = nba_player_game, nba_player'''),
    ('''def resolve_prop_ids(index, player_name, et_date):''',
     '''def resolve_prop_ids(index, player_name, et_date, resolver=None):'''),
    ('''    nn = db_store.normalize_name(player_name or "")
    cands = index.get((nn, str(et_date or "")[:10]))''',
     '''    nn = db_store.normalize_name(player_name or "")
    if resolver is not None:                       # odds spelling -> warehouse name_norm
        nn = resolver.resolve(player_name) or nn
    cands = index.get((nn, str(et_date or "")[:10]))'''),
])

# nba_stamp_odds: one resolver over the index's names
edit("nba_stamp_odds.py", [
    ('''    cache = {}                                           # (name_norm-input, et_date) -> (pid, gid)

    def _resolve(player, et_date):
        key = (player or "", et_date or "")
        if key not in cache:
            cache[key] = nba_warehouse.resolve_prop_ids(index, player, et_date)
        return cache[key]''',
     '''    cache = {}                                           # (name_norm-input, et_date) -> (pid, gid)
    import nba_names
    resolver = nba_names.Resolver({nn for nn, _d in index})

    def _resolve(player, et_date):
        key = (player or "", et_date or "")
        if key not in cache:
            cache[key] = nba_warehouse.resolve_prop_ids(index, player, et_date,
                                                        resolver=resolver)
        return cache[key]'''),
])

# recalibration grading: resolve against the season frame's names
edit("recalibration.py", [
    ('''def _nba_frame_match(df, pnorm, game_date):''',
     '''_NBA_FRAME_RESOLVERS = {}                             # id(frame) -> (frame, Resolver)


def _nba_canonical(df, player):
    """The frame's player_norm for an odds-feed name (nba_names rules: suffixes, given-name
    equivalents, aliases), else the plain normalized name. Resolver cached per frame object."""
    import nba_stats_source as _src
    plain = _src._norm(player)
    if df is None or "player_norm" not in getattr(df, "columns", []):
        return plain
    hit = _NBA_FRAME_RESOLVERS.get(id(df))
    if hit is None or hit[0] is not df:
        import nba_names
        hit = (df, nba_names.Resolver(set(df["player_norm"].dropna().astype(str))))
        _NBA_FRAME_RESOLVERS.clear()
        _NBA_FRAME_RESOLVERS[id(df)] = hit
    return hit[1].resolve(player) or plain


def _nba_frame_match(df, pnorm, game_date):'''),
    ('''        row = _nba_frame_match(df, _src._norm(player), game_date)''',
     '''        row = _nba_frame_match(df, _nba_canonical(df, player), game_date)'''),
    ('''        row = _nba_frame_match(_nba_player_game_frame(season), _src._norm(player), game_date)''',
     '''        _df = _nba_player_game_frame(season)
        row = _nba_frame_match(_df, _nba_canonical(_df, player), game_date)'''),
])

# calibration join: one resolver over the corpus names
edit("book_line_calibration.py", [
    ('''    by_player = defaultdict(list)
    for row in book_lines:
        by_player[row.get("player") or ""].append(row)''',
     '''    by_player = defaultdict(list)
    for row in book_lines:
        by_player[row.get("player") or ""].append(row)
    import nba_names
    resolver = nba_names.Resolver(bulk)                     # odds spelling -> warehouse norm'''),
    ('''        games = bulk.get(nba_source._norm(name))''',
     '''        games = bulk.get(resolver.resolve(name) or nba_source._norm(name))'''),
])

# bonus optimizer: team tag via a resolver over the player-team map
edit("nfl_bonus_optimizer.py", [
    ('''        tid = nba_teams.get(db_store.normalize_name(player or ""))''',
     '''        import nba_names
        res = nba_teams.get("__resolver__")
        nn = (res.resolve(player) if res is not None else None) \\
            or db_store.normalize_name(player or "")
        tid = nba_teams.get(nn)'''),
    ('''        try:
            import nba_warehouse
            nba_teams = nba_warehouse.player_teams()
        except Exception:
            nba_teams = {}''',
     '''        try:
            import nba_names
            import nba_warehouse
            nba_teams = dict(nba_warehouse.player_teams())
            if nba_teams:
                nba_teams["__resolver__"] = nba_names.Resolver(nba_teams)
        except Exception:
            nba_teams = {}'''),
])
print("ok")
