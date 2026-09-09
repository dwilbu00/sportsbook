"""nfl_injury_impact.py — as-of injury-burden features (offense + DEFENSE).

Team net-EPA already carries team-level defense (off_epa - def_epa) and the QB
layer carries the starter; the gap is INJURIES to non-QB key players — especially
DEFENSIVE stars (pass-rusher/CB out) — which move NFL games hard and which the
team rating hasn't caught up to. This builds per-team, per-game injury-burden
features from the injuries + snap_counts + player_week mirrors, all as-of (only
the current week's report + prior-week usage), leakage-free.

Regular-detection uses SNAP_COUNTS (offense_pct/defense_pct snap share) — reliable
for ALL positions incl. O-line and defense, which player_week can't see. A player
counts only if his mean prior-week snap share >= REGULAR_SHARE (a rotation
regular), so a never-active depth body on the report adds no phantom burden.

Valuation (kept as data-driven as our layers allow, minimal hand priors):
  * offense skill (RB/WR/TE/FB): season-to-date total offensive EPA (from
    player_week) as-of — a principled "production missing" value.
  * O-line + DEFENSE: no per-player EPA → COUNT of ruled-out regulars per unit;
    the OOS regression assigns the points value (no position priors baked in).
QB is excluded (handled by nfl_qb_asof — no double count).

Feature sign: edges are AWAY_burden - HOME_burden, so a positive value = the home
team is the healthier side (home margin should rise), matching starter_edge.
"""
import re

import nfl_data

REGULAR_SHARE = 0.5        # mean prior-week snap share to count as a rotation regular
OUT_STATUSES = {"Out", "Doubtful"}


_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _norm(name):
    if not name:
        return None
    s = str(name).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

SKILL_POS = {"RB", "WR", "TE", "FB"}
OLINE_POS = {"C", "G", "T", "OT", "OL", "G/T", "OG"}
DEF_POS = {"CB", "S", "FS", "SS", "SAF", "DB", "DE", "DT", "NT", "DL",
           "LB", "ILB", "OLB", "MLB", "EDGE"}

_SKILL_EPA_COLS = ("passing_epa", "rushing_epa", "receiving_epa")

_SNAP_CACHE = {}
_EPA_CACHE = {}


def _snap_index(season):
    """{(norm_name, team): {week: (offense_pct, defense_pct)}} from snap_counts —
    the reliable snap-share record for regular-detection at ALL positions."""
    if season in _SNAP_CACHE:
        return _SNAP_CACHE[season]
    sc = nfl_data.snap_counts([str(season)])
    idx = {}
    if sc is not None and {"player", "team", "week"} <= set(sc.columns):
        for r in sc.to_dict("records"):
            try:
                wk = int(r.get("week"))
            except (TypeError, ValueError):
                continue
            key = (_norm(r.get("player")), r.get("team"))
            op = r.get("offense_pct")
            dp = r.get("defense_pct")
            op = float(op) if op is not None and op == op else 0.0
            dp = float(dp) if dp is not None and dp == dp else 0.0
            idx.setdefault(key, {})[wk] = (op, dp)
    _SNAP_CACHE[season] = idx
    return idx


def _epa_index(season):
    """{gsis_id: {week: total_off_epa}} from player_week (offense skill value)."""
    if season in _EPA_CACHE:
        return _EPA_CACHE[season]
    pw = nfl_data.player_week([str(season)])
    epa_by = {}
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
    return epa_by


def _prior_share(snap_idx, norm_name, team, week):
    """(mean_offense_pct, mean_defense_pct) over this player's prior weeks, or
    (0,0) if never seen. Determines regular status + unit as-of."""
    rec = snap_idx.get((norm_name, team))
    if not rec:
        return 0.0, 0.0
    ops = [v[0] for w, v in rec.items() if w < week]
    dps = [v[1] for w, v in rec.items() if w < week]
    if not ops:
        return 0.0, 0.0
    return sum(ops) / len(ops), sum(dps) / len(dps)


_INJ_BY_TW = {}


def _injuries_team_week(season):
    """{(team, week): [(gsis_id, full_name, position, status)]} for Out/Doubtful."""
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
                (gid, r.get("full_name"), str(r.get("position") or "").strip(),
                 r.get("report_status")))
    _INJ_BY_TW[season] = out
    return out


def team_injury_burden(season, week, team):
    """(off_epa_out, oline_out, def_out) for `team` in `week` (as-of: uses only
    prior-week snap share + EPA to decide regular + value). All non-negative."""
    epa_by = _epa_index(season)
    snap_idx = _snap_index(season)
    injuries = _injuries_team_week(season).get((team, week), [])
    off_epa_out = 0.0
    oline_out = def_out = 0
    for gid, name, pos, _status in injuries:
        if pos == "QB":
            continue                      # QBs are handled by nfl_qb_asof — excluding
                                          # here avoids double-counting them (and their
                                          # passing EPA) in the injury burden. [review 2026-09-09]
        off_share, def_share = _prior_share(snap_idx, _norm(name), team, week)
        is_off_reg = off_share >= REGULAR_SHARE
        is_def_reg = def_share >= REGULAR_SHARE
        if not (is_off_reg or is_def_reg):
            continue                      # not a rotation regular → no phantom burden
        if is_def_reg and not is_off_reg:
            def_out += 1                  # defensive regular (snap-share based)
        elif pos in OLINE_POS:
            oline_out += 1                # offensive regular at an O-line spot
        elif gid and (pos in SKILL_POS or off_share >= REGULAR_SHARE):
            wk_epa = epa_by.get(gid, {})  # skill: production (EPA) missing
            off_epa_out += abs(sum(v for w, v in wk_epa.items() if w < week))
        else:
            oline_out += 1               # offensive regular, non-skill (OL/other)
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
