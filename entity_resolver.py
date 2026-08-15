"""entity_resolver.py — odds-boundary MLB player identity resolver (envelope).

Resolves an odds-feed player NAME + game event → (MLBAM player_id, game_pk),
FAIL-CLOSED on ambiguity: it returns an *unresolved* result rather than guess,
because a false-positive identity is far more damaging than a false-negative
(binding a prop to the wrong player poisons both the bet and the fit).

This module is the ENVELOPE the identity STAMP is written through at prediction
time (the props / wagers / odds-line callers): it owns game_pk derivation
(find_game_pk_by_commence) + the only-on-miss schedule gap-fill, and returns the
{resolved, mlb_player_id, game_pk, is_pitcher, ...} dict the caller stamps and the
P4 circuit breaker reads. The id-CORE is delegated (Commit C) to the single
StatsAPI-native, game-context/role-verified resolver
``mlb_starters.resolve_mlbam_id`` — today's posted lineup/probables → statsapi
season-roster unique-exact → role-checked SFBB fallback (which rejects the
cross-role namesake drift the retired SFBB-only lookup could not see). P5 made this
delegation UNCONDITIONAL: the old ``ODI_MLB_STAMP_RESOLVER`` gate, the SFBB-only
id-core, and the write-only ``player_alias`` audit are retired (an OFF path would
only re-introduce that drift, and the delegated path fails closed).

game_pk = mlb_warehouse.find_game_pk_by_commence(home_id, away_id, commence) —
nearest game_date to the odds commence time (series- and split-DH-robust), derived
INDEPENDENTLY of the id (so it can be present even when the player is unresolved).

MLB only. Other sports return unresolved (the caller keeps current behavior).
Every path is defensive — the resolver NEVER raises into the live prediction path.
"""

from __future__ import annotations

import datetime

import mlb_warehouse

# The MLB sport-key prefix this resolver handles (the Odds API + app key is
# "baseball_mlb"). Other baseball keys (e.g. "baseball_kbo") and NBA/NFL stay on
# the ESPN path untouched.
_MLB_SPORT_PREFIX = "baseball_mlb"


_GAP_FILLED = set()   # official dates whose schedule we've gap-ingested this process


def _gap_fill_schedule(commence):
    """On a game_pk miss, ingest the SCHEDULE (no boxscores) for the date(s) around
    ``commence`` — the UTC date and the day before, covering the UTC/local official-
    date slippage find_game_pk_by_commence tolerates — so a same-day-added game
    becomes resolvable. Each date is ingested at most ONCE per process (a genuine
    odds/StatsAPI mismatch won't re-fetch forever). Cheap + fail-open; returns True
    if any date was (re)ingested."""
    try:
        base = datetime.date.fromisoformat(str(commence)[:10])
    except (TypeError, ValueError):
        return False
    did = False
    for delta in (0, -1):
        d = (base + datetime.timedelta(days=delta)).isoformat()
        if d in _GAP_FILLED:
            continue
        _GAP_FILLED.add(d)
        try:
            mlb_warehouse.ingest_date(d, with_boxscores=False)
            did = True
        except Exception:
            pass
    return did


def _season_of(when):
    """Season year from a game_date / commence ISO string (fallback: current)."""
    try:
        return int(str(when)[:4])
    except (TypeError, ValueError):
        return mlb_warehouse._current_season()


def _unresolved(reason, game_pk=None):
    return {"resolved": False, "mlb_player_id": None, "game_pk": game_pk,
            "is_pitcher": None, "confidence": 0.0, "method": "unresolved",
            "reason": reason}


def resolve(name, sport_key, home_team, away_team, game_date=None,
            commence=None, prop_key=None, season=None,
            confirmed_lineup=None, probable_starters=None):
    """Resolve an odds-feed player NAME + game event → identity. Returns a dict:

        {resolved, mlb_player_id, game_pk, is_pitcher, confidence, method, reason}

    ``resolved`` is True only when an MLBAM id was pinned (fail-closed otherwise).
    ``game_pk`` is derived independently and may be present even when the player is
    unresolved (or absent — e.g. a doubleheader-ambiguous commence). MLB only;
    other sports and a blank name return unresolved. NEVER raises.

    ``prop_key`` + ``confirmed_lineup`` / ``probable_starters`` (today's posted
    lineup / announced probables) role-partition + game-context the delegated
    id-core (mlb_starters.resolve_mlbam_id)."""
    try:
        if not str(sport_key or "").startswith(_MLB_SPORT_PREFIX) or not name:
            return _unresolved("non_mlb_or_no_name")
        season = season or _season_of(game_date or commence)

        # game_pk: nearest game to the odds commence time (independent of the id).
        home_id = mlb_warehouse.team_id_for_name(home_team)
        away_id = mlb_warehouse.team_id_for_name(away_team)
        game_pk = (mlb_warehouse.find_game_pk_by_commence(home_id, away_id, commence)
                   if commence and home_id and away_id else None)
        # Only-on-miss gap-fill: the odds feed drives the slate, so a same-day-added
        # game may not be in mlb_game yet → no game_pk to stamp. Ingest that date's
        # SCHEDULE once (cheap, no boxscores), then retry. Deduped per process +
        # fail-open, so it never re-fetches a genuine odds/StatsAPI mismatch nor
        # taxes the hot path in the common (already-ingested) case.
        if game_pk is None and commence:
            if _gap_fill_schedule(commence):
                home_id = home_id or mlb_warehouse.team_id_for_name(home_team)
                away_id = away_id or mlb_warehouse.team_id_for_name(away_team)
                if home_id and away_id:
                    game_pk = mlb_warehouse.find_game_pk_by_commence(
                        home_id, away_id, commence)

        # MLBAM id-core (Commit C / P5 — the SOLE path): the game-context/role-
        # verified resolver. It resolves TODAY'S posted game (lineup/probables) first —
        # trade-aware + namesake-safe — then statsapi season-roster unique-exact, then
        # a role-checked SFBB fallback (rejecting the cross-role drift the retired
        # SFBB-only stamp couldn't see). It fails CLOSED (None) on ambiguity/unknown
        # AND swallows infra errors to None, so a None here is an ordinary unresolved
        # (fail-open shadow row); a systemic outage is caught by the slate-level
        # ≥50%-unpinned circuit breaker in props, not conflated here. is_pitcher is
        # statsapi-authoritative.
        try:
            import mlb_starters
            found = mlb_starters.resolve_mlbam_id(
                name, season, prop_key=prop_key, teams=[home_team, away_team],
                confirmed_lineup=confirmed_lineup,
                probable_starters=probable_starters)
        except Exception:
            found = None
        if not found:
            return _unresolved("ambiguous_or_unknown", game_pk=game_pk)
        mlb_id, is_pitcher = found
        return {"resolved": True, "mlb_player_id": str(mlb_id),
                "game_pk": game_pk, "is_pitcher": is_pitcher,
                "confidence": 1.0, "method": "game_context_resolver",
                "reason": None}
    except Exception:
        return _unresolved("resolver_error")
