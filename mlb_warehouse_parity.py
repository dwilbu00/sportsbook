"""mlb_warehouse_parity.py — P1 dual-run parity harness (read-only, NO writes).

Confidence gate for the MLB→StatsAPI cutover: diff the NEW StatsAPI-derived shapes
against what the LIVE ESPN path feeds the model TODAY, over a live window, before
anything is switched over (P4). This module is the ONLY P1 piece that touches
ESPN — it is a diagnostic, deleted at P6; the warehouse module itself stays
ESPN-free.

Two lenses:
  * standings_parity(season) — StatsAPI /standings win% vs ESPN get_all_teams
    win% per team. This is the riskiest net-new piece (§5: MLB records come only
    from ESPN today, no StatsAPI standings fetcher existed).
  * gamelog_parity(start, end) — StatsAPI boxscore-derived per-game stat vs the
    ESPN gamelog the app reads (get_player_stat_history), matched on
    (player, play-date). Batters compare hits; pitchers compare strikeouts.

Nothing here writes to SQL or calibration. It prints a diff report. The pure diff
helpers are unit-tested; the live ESPN accessors are best-effort and guarded.
"""

from __future__ import annotations

import argparse

import db_store
import mlb_starters
import mlb_warehouse

# ESPN sport/league identifiers for MLB (SPORTS[...]["espn_sport"/"espn_league"]).
_ESPN_SPORT = "baseball"
_ESPN_LEAGUE = "mlb"


def _norm(name):
    return db_store.normalize_name(name or "")


def _date10(v):
    """Normalize any date/timestamp string to YYYY-MM-DD (or None)."""
    if not v:
        return None
    return str(v)[:10]


def _prev_day(d):
    """The calendar day before an ISO YYYY-MM-DD string (or None)."""
    import datetime
    try:
        return (datetime.date.fromisoformat(str(d)[:10])
                - datetime.timedelta(days=1)).isoformat()
    except (TypeError, ValueError):
        return None


def _align_espn_to_official(espn_raw, statsapi_keys, start=None, end=None):
    """Collapse ESPN {(name_norm, UTC-date): value} rows onto the StatsAPI official
    play date, SUM games that land on one (player, official-date), and bound to
    [start, end]. Returns the aligned {(name_norm, official_date): value} map.

    ESPN's batter game_date is the UTC first-pitch date, which for non-Eastern
    night games (and late-Eastern games) is one calendar day AHEAD of the StatsAPI
    official (local) play date the StatsAPI side keys on — the same ±1-day
    UTC/local slippage the codebase already reconciles in mlb_starters.
    _candidate_dates and recalibration.resolve_one_prop. Without this collapse the
    identical game keys under two different dates and silently drops out of the
    diff, under-testing the DEFAULT batter role. Summing per (player, official)
    also makes a doubleheader compare day-total vs day-total instead of mis-pairing
    (or dropping) one of the two games.

    Residual caveat: two consecutive night games for one player, or a DH whose two
    games straddle the sampling window edge, can still be attributed imperfectly —
    but that surfaces as a visible mismatch/only_* count, never a false green.
    """
    out = {}
    for (nm, d), v in espn_raw.items():
        d_off = d
        if (nm, d) not in statsapi_keys:
            prev = _prev_day(d)
            if prev is not None and (nm, prev) in statsapi_keys:
                d_off = prev
        if start is not None and str(d_off) < str(start):
            continue
        if end is not None and str(d_off) > str(end):
            continue
        out[(nm, d_off)] = out.get((nm, d_off), 0.0) + (v or 0.0)
    return out


# ─────────────────────────────────────────────────────────────── pure diff core
def diff_value_maps(statsapi_map, espn_map, tol=1e-6):
    """Compare two {key: value} maps. Returns a report dict:

    matches / mismatches (both sides present, |Δ| > tol) / only_statsapi /
    only_espn, plus a small list of example mismatches. Pure + deterministic."""
    keys_a = set(statsapi_map)
    keys_b = set(espn_map)
    both = keys_a & keys_b
    matches = 0
    mismatches = []
    for k in sorted(both, key=str):
        a = statsapi_map[k]
        b = espn_map[k]
        if a is None or b is None:
            if a is None and b is None:
                matches += 1
            else:
                mismatches.append((k, a, b))
            continue
        if abs(float(a) - float(b)) <= tol:
            matches += 1
        else:
            mismatches.append((k, a, b))
    return {
        "compared": len(both),
        "matches": matches,
        "mismatches": len(mismatches),
        "only_statsapi": len(keys_a - keys_b),
        "only_espn": len(keys_b - keys_a),
        "match_rate": (matches / len(both)) if both else None,
        "examples": mismatches[:15],
    }


# ─────────────────────────────────────────────────────── StatsAPI side (tested)
def statsapi_standings_winpct(season):
    """{team_full_name_norm: win_pct} from a StatsAPI /standings snapshot.

    The /standings payload's team.name is only the NICKNAME ('Yankees', 'Dbacks'),
    which does not match ESPN's full displayName ('New York Yankees', 'Arizona
    Diamondbacks'). Resolve each team_id → full name via /teams so the two sides
    key on the same string (falls back to the nickname if /teams is unavailable)."""
    raw = mlb_warehouse.fetch_standings(season)
    try:
        id_to_name = {t["team_id"]: t["name"] for t
                      in mlb_warehouse.parse_teams(mlb_warehouse.fetch_teams(season))}
    except Exception:
        id_to_name = {}
    out = {}
    for rec in (raw or {}).get("records", []) or []:
        for tr in rec.get("teamRecords", []) or []:
            team = tr.get("team") or {}
            full = id_to_name.get(str(team.get("id"))) or team.get("name")
            wp = mlb_warehouse._f(tr.get("winningPercentage"))
            if full and wp is not None:
                out[_norm(full)] = wp
    return out


def statsapi_player_game_stats(box, game, role, stat_key):
    """{(name_norm, play_date): value} for one boxscore + game, for a role.

    role='batter' → derive_batter_rows; role='pitcher' → derive_pitcher_rows.
    Player names come from the boxscore person entries."""
    players = mlb_warehouse.parse_boxscore_players(box, game)
    name_by_id = {p["player_id"]: p["name_norm"] for p in players}
    date = _date10((game or {}).get("official_date") or (game or {}).get("game_date"))
    rows = (mlb_warehouse.derive_batter_rows(box, game) if role == "batter"
            else mlb_warehouse.derive_pitcher_rows(box, game))
    out = {}
    for r in rows:
        nm = name_by_id.get(r["athlete_id"])
        if nm and date and r.get(stat_key) is not None:
            out[(nm, date)] = r[stat_key]
    return out


# ─────────────────────────────────────────────────── ESPN side (live, best-effort)
def _espn_player_game_stats(name, prop_key, n=25):
    """{(name_norm, play_date): value} from the live ESPN gamelog the app reads."""
    try:
        import espn_client
        hist = espn_client.get_player_stat_history(
            _ESPN_SPORT, _ESPN_LEAGUE, name, prop_key, n=n)
    except Exception:
        return {}
    if not hist or not hist.get("found"):
        return {}
    out = {}
    nm = _norm(name)
    dates = hist.get("game_dates") or []
    values = hist.get("values") or []
    for d, v in zip(dates, values):
        d10 = _date10(d)
        if d10 is not None and v is not None:
            out[(nm, d10)] = v
    return out


def _espn_standings_winpct():
    """{team_name_norm: win_pct} from the live ESPN teams/standings merge."""
    try:
        import espn_client
        teams = espn_client.get_all_teams(_ESPN_SPORT, _ESPN_LEAGUE)
    except Exception:
        return {}
    out = {}
    for display_name, info in (teams or {}).items():
        wp = info.get("win_pct")
        if wp is not None:
            out[_norm(display_name)] = wp
    return out


# ─────────────────────────────────────────────────────────────────── the lenses
def standings_parity(season=None):
    """Diff StatsAPI vs ESPN team win% for a season. Returns a report dict."""
    season = int(season) if season else mlb_warehouse._current_season()
    a = statsapi_standings_winpct(season)
    b = _espn_standings_winpct()
    rep = diff_value_maps(a, b, tol=0.001)
    rep["season"] = season
    rep["statsapi_teams"] = len(a)
    rep["espn_teams"] = len(b)
    return rep


_ROLE_PROP = {
    "batter": ("H", "batter_hits"),
    "pitcher": ("K", "pitcher_strikeouts"),
}


def gamelog_parity(start, end, role="batter", sample=25, espn_n=40):
    """Diff StatsAPI boxscore-derived per-game stat vs the ESPN gamelog over a
    window. role ∈ {'batter','pitcher'}. Samples up to `sample` distinct players
    (keeps the ESPN fan-out bounded). Both sides are keyed on the official (local)
    play date and summed per (player, day) — see _align_espn_to_official for why
    the ESPN UTC first-pitch date must be collapsed onto it. Returns a report."""
    stat_key, prop_key = _ROLE_PROP[role]
    import datetime
    d0 = datetime.date.fromisoformat(str(start))
    d1 = datetime.date.fromisoformat(str(end))

    statsapi_map = {}          # (name_norm, official_date) -> summed value
    name_display = {}          # name_norm -> a display name for the ESPN lookup
    cur = d0
    while cur <= d1:
        date = cur.isoformat()
        cur += datetime.timedelta(days=1)
        try:
            raw = mlb_warehouse.fetch_schedule(date)
        except Exception:
            continue
        games, _teams = mlb_warehouse.parse_schedule(raw, int(date[:4]))
        for g in games:
            if not mlb_starters._is_genuine_final(
                    {"status": g["status"], "detailedState": g["detailed_state"]}):
                continue
            try:
                box = mlb_warehouse.fetch_boxscore(g["game_pk"])
            except Exception:
                continue
            for p in mlb_warehouse.parse_boxscore_players(box, g):
                if p["full_name"]:
                    name_display.setdefault(p["name_norm"], p["full_name"])
            # Sum per (player, official-date) so a split doubleheader is one
            # day-total on the StatsAPI side (mirrors the ESPN-side collapse).
            for k, v in statsapi_player_game_stats(box, g, role, stat_key).items():
                statsapi_map[k] = statsapi_map.get(k, 0.0) + v

    # Sample distinct players deterministically (sorted by name).
    players = sorted({nm for (nm, _d) in statsapi_map})[:sample]
    sampled = {(nm, d): v for (nm, d), v in statsapi_map.items() if nm in players}

    espn_raw = {}
    for nm in players:
        for k, v in _espn_player_game_stats(
                name_display.get(nm, nm), prop_key, n=espn_n).items():
            espn_raw[k] = espn_raw.get(k, 0.0) + v
    # Realign ESPN UTC dates onto the official play date, sum same-day games, and
    # bound to the window — kills both the batter UTC/local ±1-day misalignment
    # (which would drop most night games from `compared`) and the windowing-
    # inflated only_espn (espn_n reaches back past the window).
    espn_map = _align_espn_to_official(
        espn_raw, set(sampled), start=str(start), end=str(end))

    rep = diff_value_maps(sampled, espn_map, tol=1e-6)
    rep["role"] = role
    rep["window"] = f"{start}..{end}"
    rep["stat"] = stat_key
    rep["players_sampled"] = len(players)
    return rep


def _fmt_report(title, rep):
    lines = [f"── {title} ─────────────────────────────────────────"]
    for k in ("season", "role", "window", "stat", "players_sampled",
              "statsapi_teams", "espn_teams", "compared", "matches",
              "mismatches", "only_statsapi", "only_espn", "match_rate"):
        if k in rep and rep[k] is not None:
            v = rep[k]
            if k == "match_rate":
                v = f"{v:.1%}"
            lines.append(f"  {k:16s}: {v}")
    if rep.get("examples"):
        lines.append("  example mismatches (key, statsapi, espn):")
        for key, a, b in rep["examples"]:
            lines.append(f"    {key}  statsapi={a}  espn={b}")
    return "\n".join(lines)


def _main_cli():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="P1 dual-run parity harness: diff StatsAPI-derived MLB shapes "
                    "against the live ESPN path (read-only, no writes).")
    ap.add_argument("--standings", nargs="?", const=0, type=int, metavar="SEASON",
                    help="Standings win% parity for SEASON (default: current year).")
    ap.add_argument("--gamelog", nargs=2, metavar=("START", "END"),
                    help="Gamelog parity over an inclusive date window.")
    ap.add_argument("--role", choices=("batter", "pitcher"), default="batter",
                    help="Gamelog role to diff (default: batter → hits).")
    ap.add_argument("--sample", type=int, default=25,
                    help="Max distinct players to diff for --gamelog.")
    args = ap.parse_args()

    db_store.promote_secrets_from_toml()
    did = False
    if args.standings is not None:
        did = True
        season = args.standings if args.standings and args.standings > 0 else None
        print(_fmt_report("standings win% parity", standings_parity(season)))
    if args.gamelog:
        did = True
        rep = gamelog_parity(args.gamelog[0], args.gamelog[1],
                             role=args.role, sample=args.sample)
        print(_fmt_report(f"gamelog parity ({args.role})", rep))
    if not did:
        ap.error("nothing to do — pass --standings and/or --gamelog START END")


if __name__ == "__main__":
    _main_cli()
