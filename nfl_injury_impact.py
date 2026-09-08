"""nfl_injury_impact.py — as-of injury-burden features (offense + DEFENSE).

Team net-EPA already carries team-level defense (off_epa - def_epa) and the QB
layer carries the starter; the gap is INJURIES to non-QB key players — especially
DEFENSIVE stars (pass-rusher/CB out) — which move NFL games hard and which the
team rating hasn't caught up to. This builds per-team, per-game injury-burden
features from the (reliable) injuries + player_week mirrors, all as-of (only the
current week's report + prior-week usage), leakage-free.

Valuation (kept as data-driven as our layers allow, minimal hand priors):
  * offense skill (RB/WR/TE/FB): the player's season-to-date total offensive EPA
    (passing+rushing+receiving) as-of — a principled "production missing" value.
  * o-line + DEFENSE: no per-player EPA in our data → COUNT of ruled-out regulars
    per unit and let the OOS regression assign the points value (no position
    priors baked in).
Only "regulars" count (appeared in >= REGULAR_WEEKS prior weeks this season) so a
never-active depth body on the report doesn't add phantom burden. QB is excluded
(handled by nfl_qb_asof — no double count).

Feature sign: edges are AWAY_burden - HOME_burden, so a positive value = the home
team is the healthier side (home margin should rise), matching starter_edge.
"""
import nfl_data

REGULAR_WEEKS = 3          # prior appearances to count as a rotation regular
OUT_STATUSES = {"Out", "Doubtful"}

SKILL_POS = {"RB", "WR", "TE", "FB"}
OLINE_POS = {"C", "G", "T", "OT", "OL", "G/T", "OG"}
DEF_POS = {"CB", "S", "FS", "SS", "SAF", "DB", "DE", "DT", "NT", "DL",
           "LB", "ILB", "OLB", "MLB", "EDGE"}

_SKILL_EPA_COLS = ("passing_epa", "rushing_epa", "receiving_epa")

_APPEAR_CACHE = {}
_EPA_CACHE = {}


def _build_indexes(season):
    """{gsis_id: {week: total_off_epa}} and {gsis_id: sorted[weeks appeared]} from
    player_week for one season."""
    if season in _EPA_CACHE:
        return _EPA_CACHE[season], _APPEAR_CACHE[season]
    pw = nfl_data.player_week([str(season)])
    epa_by = {}      # gsis -> {week: off_epa that week}
    appear = {}      # gsis -> set(weeks)
    if pw is not None and "player_id" in pw.columns and "week" in pw.columns:
        cols = [c for c in _SKILL_EPA_COLS if c in pw.columns]
        for r in pw.to_dict("records"):
            gid = r.get("player_id")
            if gid is None or (isinstance(gid, float) and gid != gid):
                continue
            gid = str(gid)
            try:
                wk = int(r.get("week"))
            except (TypeError, ValueError):
                continue
            appear.setdefault(gid, set()).add(wk)
            e = 0.0
            for c in cols:
                v = r.get(c)
                if v is not None and v == v:
                    try:
                        e += float(v)
                    except (TypeError, ValueError):
                        pass
            epa_by.setdefault(gid, {})[wk] = epa_by.get(gid, {}).get(wk, 0.0) + e
    _EPA_CACHE[season] = epa_by
    _APPEAR_CACHE[season] = appear
    return epa_by, appear


_INJ_BY_TW = {}


def _injuries_team_week(season):
    """{(team, week): [(gsis_id, position, status)]} for Out/Doubtful only."""
    if season in _INJ_BY_TW:
        return _INJ_BY_TW[season]
    out = {}
    df = nfl_data.injuries([str(season)])
    if df is not None and {"team", "week", "report_status"} <= set(df.columns):
        for r in df.to_dict("records"):
            if r.get("report_status") not in OUT_STATUSES:
                continue
            try:
                wk = int(r.get("week"))
            except (TypeError, ValueError):
                continue
            gid = r.get("gsis_id")
            gid = str(gid) if gid is not None and not (isinstance(gid, float) and gid != gid) else None
            out.setdefault((r.get("team"), wk), []).append(
                (gid, str(r.get("position") or "").strip(), r.get("report_status")))
    _INJ_BY_TW[season] = out
    return out


def team_injury_burden(season, week, team):
    """(off_epa_out, oline_out, def_out) for `team` in `week` (as-of: uses only
    prior-week usage to decide regular + value). All non-negative burdens."""
    epa_by, appear = _build_indexes(season)
    injuries = _injuries_team_week(season).get((team, week), [])
    off_epa_out = 0.0
    oline_out = def_out = 0
    for gid, pos, _status in injuries:
        if not gid:
            continue
        prior = [w for w in appear.get(gid, ()) if w < week]
        if len(prior) < REGULAR_WEEKS:
            continue                      # not an established regular → skip
        if pos in SKILL_POS:
            # season-to-date offensive EPA (prior weeks only) = production missing
            wk_epa = epa_by.get(gid, {})
            val = sum(v for w, v in wk_epa.items() if w < week)
            off_epa_out += abs(val)
        elif pos in OLINE_POS:
            oline_out += 1
        elif pos in DEF_POS:
            def_out += 1
    return off_epa_out, oline_out, def_out


def injury_edges(season, week, home, away):
    """(off_inj_edge, oline_inj_edge, def_inj_edge) = away_burden - home_burden
    for each unit (positive = home is the healthier side)."""
    ho, hol, hd = team_injury_burden(season, week, home)
    ao, aol, ad = team_injury_burden(season, week, away)
    return (ao - ho, aol - hol, ad - hd)


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
    ap.add_argument("--week", type=int, default=12)
    args = ap.parse_args()
    tw = _injuries_team_week(args.season)
    teams = sorted({t for (t, w) in tw if w == args.week})
    print(f"=== {args.season} week {args.week} injury burden (regulars Out/Doubtful) ===")
    print(f"  {'team':<5}{'off_epa_out':>12}{'oline_out':>10}{'def_out':>9}")
    for t in teams:
        o, ol, d = team_injury_burden(args.season, args.week, t)
        if o or ol or d:
            print(f"  {t:<5}{o:>12.1f}{ol:>10}{d:>9}")
