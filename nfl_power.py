"""nfl_power.py — OPPONENT-ADJUSTED EPA power ratings (leakage-safe, as-of).

Raw off/def EPA (nfl_epa.team_epa) is biased by strength of schedule: a team that
faced soft defenses looks better than it is. This solves for offensive and
defensive ratings that explain the observed per-play EPA as
    epa(off=t, def=d)  ~=  off_rating[t] - def_rating[d]
via iterative opponent adjustment (the EPA analog of SRS), so each team's rating
nets out who it played. Optionally restricts to EARLY DOWNS (1st/2nd), where EPA
is more stable and predictive of the future than late-down/garbage-time EPA.

Same leakage discipline as nfl_epa: only plays strictly before as_of_date count,
and the current-season rating is shrunk toward the (adjusted) prior season when
the sample is thin. Reads the same play list (mirror pbp) as nfl_epa.
"""
import nfl_epa
import nfl_data

N_ITER = 10                       # opponent-adjustment passes (converges fast)
STABILIZE_PLAYS = nfl_epa.STABILIZE_PLAYS


def _down_index(season):
    """{(game_id, posteam, defteam) -> not used}; we need per-play down, so build
    a parallel list of (game_id, posteam, defteam, epa, down, game_date) from the
    pbp mirror once, cached. Falls back to None (no down filter) if unavailable."""
    key = ("down", season)
    if key in _CACHE:
        return _CACHE[key]
    df = nfl_data.pbp([str(season)])
    out = None
    if df is not None and {"posteam", "defteam", "epa", "down", "game_date",
                           "pass", "rush", "season_type"} <= set(df.columns):
        out = []
        for r in df.to_dict("records"):
            if str(r.get("season_type")) != "REG":
                continue
            pt, dt, epa = r.get("posteam"), r.get("defteam"), r.get("epa")
            if not pt or not dt or epa is None or epa != epa:
                continue
            is_pr = False
            for k in ("pass", "rush"):
                v = r.get(k)
                try:
                    is_pr = is_pr or float(v) == 1.0
                except (TypeError, ValueError):
                    is_pr = is_pr or str(v) == "1"
            if not is_pr:
                continue
            try:
                dn = int(r.get("down")) if r.get("down") == r.get("down") else None
            except (TypeError, ValueError):
                dn = None
            gd = r.get("game_date")
            out.append((pt, dt, float(epa), dn,
                        str(gd)[:10] if gd is not None and gd == gd else None))
    _CACHE[key] = out
    return out


_CACHE = {}


def _adjust(pairs, n_iter=N_ITER):
    """pairs: {(off, def): (sum_epa, n)}. Returns opponent-adjusted
    {t: {off, def, net, off_plays, def_plays}} via iterative SRS-style passes."""
    if not pairs:
        return {}
    off_n, def_n = {}, {}
    off_sum, def_sum = {}, {}
    w = {}                                   # w[(t,d)] = n plays
    for (t, d), (s, n) in pairs.items():
        off_n[t] = off_n.get(t, 0) + n
        def_n[d] = def_n.get(d, 0) + n
        off_sum[t] = off_sum.get(t, 0.0) + s
        def_sum[d] = def_sum.get(d, 0.0) + s
        w[(t, d)] = n
    teams = set(off_n) | set(def_n)
    raw_off = {t: off_sum.get(t, 0.0) / off_n[t] for t in off_n}
    raw_def = {d: def_sum.get(d, 0.0) / def_n[d] for d in def_n}
    off_adj = dict(raw_off)
    def_adj = {d: 0.0 for d in def_n}
    for _ in range(n_iter):
        # offense = raw_off minus the avg (play-weighted) adjusted defense faced
        new_off = {}
        for t in off_n:
            corr = sum(w[(t, d)] * def_adj.get(d, 0.0)
                       for d in def_n if (t, d) in w) / off_n[t]
            new_off[t] = raw_off[t] + corr        # + because def_adj>0 = tough D suppressed us
        new_def = {}
        for d in def_n:
            corr = sum(w[(t, d)] * new_off.get(t, 0.0)
                       for t in off_n if (t, d) in w) / def_n[d]
            new_def[d] = raw_def[d] - corr        # def rating = EPA allowed vs adj offense
        # center to keep identifiable (league offense mean == raw league mean)
        lm_off = sum(new_off.values()) / len(new_off)
        lm_def = sum(new_def.values()) / len(new_def)
        off_adj = {t: v - lm_off for t, v in new_off.items()}
        def_adj = {d: v - lm_def for d, v in new_def.items()}
    out = {}
    for t in teams:
        o, dfn = off_adj.get(t, 0.0), def_adj.get(t, 0.0)
        out[t] = {"off_epa": o, "def_epa": dfn, "net_epa": o - dfn,
                  "off_plays": off_n.get(t, 0), "def_plays": def_n.get(t, 0)}
    return out


def _pairs(season, as_of_date, early_down=False):
    """{(off,def): (sum_epa, n)} from plays strictly before as_of_date."""
    pairs = {}
    if early_down:
        rows = _down_index(season) or []
        for pt, dt, epa, dn, gd in rows:
            if as_of_date and (gd or "") >= as_of_date:
                continue
            if dn not in (1, 2):
                continue
            s, n = pairs.get((pt, dt), (0.0, 0))
            pairs[(pt, dt)] = (s + epa, n + 1)
    else:
        for p in nfl_epa.load_plays(season):
            if as_of_date and (p["game_date"] or "") >= as_of_date:
                continue
            k = (p["posteam"], p["defteam"])
            s, n = pairs.get(k, (0.0, 0))
            pairs[k] = (s + p["epa"], n + 1)
    return pairs


_RATING_CACHE = {}


def team_power(season, as_of_date=None, prior_shrink=True, early_down=False,
               n_iter=N_ITER):
    """Opponent-adjusted ratings for `season` as-of `as_of_date`. Shrinks toward
    the adjusted prior season when the current sample is thin (like nfl_epa)."""
    key = (season, as_of_date, prior_shrink, early_down, n_iter)
    if key in _RATING_CACHE:
        return _RATING_CACHE[key]
    cur = _adjust(_pairs(season, as_of_date, early_down), n_iter)
    prior = None
    if prior_shrink:
        try:
            prior = _adjust(_pairs(season - 1, None, early_down), n_iter)
        except Exception:
            prior = None
    out = {}
    for t, c in cur.items():
        off, deff = c["off_epa"], c["def_epa"]
        if prior and t in prior:
            pn = min(c["off_plays"], c["def_plays"])
            wt = min(1.0, pn / STABILIZE_PLAYS)
            off = wt * off + (1 - wt) * prior[t]["off_epa"]
            deff = wt * deff + (1 - wt) * prior[t]["def_epa"]
        out[t] = {"off_epa": off, "def_epa": deff, "net_epa": off - deff,
                  "off_plays": c["off_plays"], "def_plays": c["def_plays"]}
    if prior:
        for t, pv in prior.items():
            if t not in out:
                out[t] = {"off_epa": pv["off_epa"], "def_epa": pv["def_epa"],
                          "net_epa": pv["net_epa"], "off_plays": 0, "def_plays": 0}
    _RATING_CACHE[key] = out
    return out


if __name__ == "__main__":
    import argparse
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--early-down", action="store_true")
    args = ap.parse_args()
    r = team_power(args.season, prior_shrink=False, early_down=args.early_down)
    raw = nfl_epa.team_epa(args.season, prior_shrink=False)
    print(f"=== {args.season} opponent-adjusted net EPA "
          f"({'early-down' if args.early_down else 'all-down'}) vs raw ===")
    for t, v in sorted(r.items(), key=lambda kv: -kv[1]["net_epa"])[:12]:
        rr = raw.get(t, {}).get("net_epa", 0)
        print(f"  {t:<4} adj_net={v['net_epa']:+.4f}  (raw {rr:+.4f}, "
              f"Δ{v['net_epa']-rr:+.4f})")
