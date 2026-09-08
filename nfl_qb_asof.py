"""nfl_qb_asof.py — starting-QB as-of adjustment for the NFL EPA model.

The biggest gap in the team-EPA model (nfl_epa): it is TEAM-level and has no
notion of WHO is playing quarterback. NFL is QB-dominated — a backup or a
mid-season starter change makes the team's season-to-date offensive EPA stale,
and the market prices the QB change while our rating lags. This layer is the
NFL analog of MLB ``pitcher_asof``: it isolates the starting-QB signal the team
rating misses.

Three pieces, all off the pbp mirror (dep-free of nflreadpy), all leakage-safe:
  * qb_ratings(season, as_of)   per-passer dropback-EPA/play from games STRICTLY
                                before as_of, shrunk toward the league mean when
                                the sample is thin (a QB stabilises slowly).
  * game_starters(season)       {game_id: {team: passer_id}} — the passer with the
                                most dropbacks per team per game. Starter IDENTITY
                                is public pre-game (announced/inactives), so using
                                the game-actual starter is NOT outcome leakage;
                                his RATING is still as-of prior games only. This is
                                the optimistic CEILING test ("if we know the
                                starter, does QB info help?"). A live path would
                                swap in the depth-chart/injury-projected starter.
  * qb_edge_delta(...)          the EPA/play adjustment for one team-game: the
                                starter's as-of rating minus the team's as-of
                                BASELINE QB (the passer who built most of the
                                team's prior-games sample). ~0 when the usual QB
                                starts; negative when a weaker backup starts.

The consumer adds (delta_home - delta_away) to nfl_epa's net-EPA edge.
"""
import nfl_data

# A QB's dropback-EPA/play stabilises slowly; below this many as-of dropbacks we
# shrink his rating toward the league mean (weight = dropbacks / STABILIZE_DB).
STABILIZE_DB = 250
LEAGUE_FALLBACK_EPA = 0.0   # league-mean dropback EPA is ~0 by construction


def _truthy(v):
    try:
        return float(v) == 1.0
    except (TypeError, ValueError):
        return str(v) == "1"


_PBP_CACHE = {}


def _dropbacks(season):
    """List of {game_id, game_date, posteam, passer_id, epa} for REG dropbacks
    with a known passer + epa, from the pbp mirror. Cached per season."""
    if season in _PBP_CACHE:
        return _PBP_CACHE[season]
    df = nfl_data.pbp([str(season)])
    out = []
    if df is not None and len(df):
        cols = df.columns
        need = ("game_id", "game_date", "posteam", "passer_player_id",
                "qb_dropback", "epa", "season_type")
        if all(c in cols for c in need):
            for r in df.to_dict("records"):
                if str(r.get("season_type")) != "REG":
                    continue
                if not _truthy(r.get("qb_dropback")):
                    continue
                pid, epa = r.get("passer_player_id"), r.get("epa")
                # passer_player_id is NaN on unattributed dropbacks (some sacks/
                # scrambles); NaN is truthy in Python, so guard explicitly.
                if pid is None or (isinstance(pid, float) and pid != pid):
                    continue
                pid = str(pid)
                if pid in ("", "nan", "None"):
                    continue
                if epa is None or epa != epa:
                    continue
                try:
                    e = float(epa)
                except (TypeError, ValueError):
                    continue
                gd = r.get("game_date")
                out.append({
                    "game_id": r.get("game_id"),
                    "game_date": str(gd)[:10] if gd is not None and gd == gd else None,
                    "posteam": r.get("posteam"),
                    "passer_id": pid,
                    "epa": e,
                })
    _PBP_CACHE[season] = out
    return out


_RATINGS_CACHE = {}


def qb_ratings(season, as_of_date=None, prior_season=True):
    """{passer_id: {epa, dropbacks, raw_epa}} from dropbacks strictly before
    as_of_date (None = full season). When a passer has < STABILIZE_DB as-of
    dropbacks, his epa is shrunk toward the league mean; the prior season's
    dropbacks are folded in (also as-of-safe: they're all before this season) so
    a returning starter isn't treated as a rookie in week 1."""
    key = (season, as_of_date, prior_season)
    if key in _RATINGS_CACHE:
        return _RATINGS_CACHE[key]

    plays = list(_dropbacks(season))
    if prior_season:
        try:
            plays = list(_dropbacks(season - 1)) + plays   # prior season fully in the past
        except Exception:
            pass

    s_sum, s_n = {}, {}
    tot_sum = tot_n = 0
    for p in plays:
        if as_of_date and (p["game_date"] or "") >= as_of_date and \
                _same_season(p["game_date"], season):
            continue
        pid = p["passer_id"]
        s_sum[pid] = s_sum.get(pid, 0.0) + p["epa"]
        s_n[pid] = s_n.get(pid, 0) + 1
        tot_sum += p["epa"]
        tot_n += 1
    league = (tot_sum / tot_n) if tot_n else LEAGUE_FALLBACK_EPA

    out = {}
    for pid, n in s_n.items():
        raw = s_sum[pid] / n
        w = min(1.0, n / STABILIZE_DB)
        out[pid] = {"epa": w * raw + (1.0 - w) * league,
                    "raw_epa": raw, "dropbacks": n}
    _RATINGS_CACHE[key] = (out, league)
    return out, league


def _same_season(game_date, season):
    """True if a YYYY-MM-DD play date belongs to `season` (NFL season spans
    Sep..Feb+1). Used so the as_of cutoff only filters the CURRENT season's
    games — prior-season plays are always in the past and always kept."""
    if not game_date:
        return False
    import nfl_epa
    return nfl_epa.season_for_date(game_date) == season


_STARTERS_CACHE = {}


def game_starters(season):
    """{game_id: {team_abbr: passer_id}} — the passer with the most dropbacks for
    each team in each game (the de-facto starter)."""
    if season in _STARTERS_CACHE:
        return _STARTERS_CACHE[season]
    counts = {}   # (gid, team) -> {pid: n}
    for p in _dropbacks(season):
        k = (p["game_id"], p["posteam"])
        counts.setdefault(k, {})
        counts[k][p["passer_id"]] = counts[k].get(p["passer_id"], 0) + 1
    out = {}
    for (gid, team), c in counts.items():
        starter = max(c.items(), key=lambda kv: kv[1])[0]
        out.setdefault(gid, {})[team] = starter
    _STARTERS_CACHE[season] = out
    return out


def team_baseline_qb(season, as_of_date, team):
    """The passer who built most of `team`'s CURRENT-season sample strictly before
    as_of_date (the QB the team-EPA rating implicitly assumes). None in week 1
    (no prior current-season dropbacks) → the delta falls back to 0."""
    c = {}
    for p in _dropbacks(season):
        if p["posteam"] != team:
            continue
        if (p["game_date"] or "") >= (as_of_date or "9999"):
            continue
        c[p["passer_id"]] = c.get(p["passer_id"], 0) + 1
    if not c:
        return None
    return max(c.items(), key=lambda kv: kv[1])[0]


def qb_edge_delta(season, as_of_date, team, starter_id, ratings=None):
    """EPA/play adjustment for `team` in a game on as_of_date: the STARTER's
    as-of rating minus the team's BASELINE QB's as-of rating. ~0 when the usual
    starter plays; negative when a weaker backup starts (the signal team-EPA
    misses). Returns 0.0 when either QB is unknown/unrated (graceful degrade)."""
    if ratings is None:
        ratings, _lg = qb_ratings(season, as_of_date)
    if not starter_id:
        return 0.0
    base_id = team_baseline_qb(season, as_of_date, team)
    if base_id is None or base_id == starter_id:
        return 0.0
    sr = ratings.get(starter_id)
    br = ratings.get(base_id)
    if not sr or not br:
        return 0.0
    return sr["epa"] - br["epa"]


if __name__ == "__main__":
    import argparse
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--changes", action="store_true",
                    help="list games where the starter != the team's baseline QB "
                         "(the signal-bearing games)")
    args = ap.parse_args()

    ratings, league = qb_ratings(args.season, prior_season=False)
    print(f"=== {args.season} full-season QB dropback-EPA (league mean {league:+.4f}) ===")
    ranked = sorted(ratings.items(), key=lambda kv: -kv[1]["epa"])
    ranked = [r for r in ranked if r[1]["dropbacks"] >= 150]
    for pid, v in ranked[:args.top]:
        print(f"  {pid:<14} epa/db={v['epa']:+.4f} (raw {v['raw_epa']:+.4f}, "
              f"{v['dropbacks']} db)")

    if args.changes:
        print(f"\n=== {args.season} starter != baseline-QB games (as-of deltas) ===")
        starters = game_starters(args.season)
        import nfl_epa
        n = 0
        for gid in sorted(starters):
            parts = gid.split("_")
            # approximate as_of = game date via the spine
            for team, sid in starters[gid].items():
                # find the game date from any dropback of this game
                gd = next((p["game_date"] for p in _dropbacks(args.season)
                           if p["game_id"] == gid), None)
                if not gd:
                    continue
                base = team_baseline_qb(args.season, gd, team)
                if base and base != sid:
                    r, _l = qb_ratings(args.season, gd)
                    delta = qb_edge_delta(args.season, gd, team, sid, r)
                    if abs(delta) > 1e-9:
                        print(f"  {gid:<20} {team}: starter {sid} vs baseline {base}"
                              f"  delta={delta:+.4f} epa/play")
                        n += 1
        print(f"  ({n} team-games with a QB change + rating delta)")
