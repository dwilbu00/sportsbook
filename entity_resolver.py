"""entity_resolver.py — P3 odds-boundary MLB player identity resolver (additive).

Resolves an odds-feed player NAME + game event → (MLBAM player_id, game_pk),
FAIL-CLOSED on ambiguity: it returns an *unresolved* result rather than guess,
because a false-positive identity is far more damaging than a false-negative
(binding a prop to the wrong player poisons both the bet and the fit). Successful
GLOBALLY-unambiguous resolutions are recorded in the player_alias table (the
spec's "associations stored once" audit trail).

Additive / DUAL-RUN posture (P3): the caller STAMPS the returned (mlb_player_id,
game_pk) onto new prediction/wager/odds rows, but NOTHING is dropped yet — an
unresolved player still gets a prediction (fail-open preserved), it just lands
with NULL ids. Those NULLs ARE the shadow measurement: review unresolved MLB
props with `... WHERE sport_key='baseball_mlb' AND player_mlb_id IS NULL`. P4
flips enforcement (unresolved MLB → no prediction) once that blast radius is
understood.

Design (reuses the battle-tested pieces, no parallel ladder):
  * MLBAM resolution = mlb_starters.find_player_id(name, season, teams=[home, away])
    — SFBB id-map FIRST (disambiguates namesakes, folds accents, strips suffixes),
    then the statsapi season-roster unique-exact match; the two-team hint breaks a
    genuine namesake tie. It already returns None (fail-closed) on ambiguity — and
    P3 finally feeds it BOTH teams (the live enrichers pass only one today).
  * game_pk = mlb_warehouse.find_game_pk_by_commence(home_id, away_id, commence)
    — nearest game_date to the odds commence time (series- and split-DH-robust).
  * player_alias is written ONLY for names that resolve GLOBALLY unambiguously
    (mlb_id_for_name(name, teams=None) alone), so a bare-name alias can never later
    serve the wrong namesake; genuinely-shared names re-resolve per-event via the
    hint and are deliberately NOT bare-aliased.

MLB only. Other sports return unresolved (the caller keeps current behavior).
Every path is defensive — the resolver NEVER raises into the live prediction path.
"""

from __future__ import annotations

import db_store
import mlb_warehouse

# Sport keys this resolver handles. NBA/NFL stay on the ESPN path untouched.
_MLB_SPORT_KEYS = {"baseball_mlb"}


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
            commence=None, prop_key=None, season=None):
    """Resolve an odds-feed player NAME + game event → identity. Returns a dict:

        {resolved, mlb_player_id, game_pk, is_pitcher, confidence, method, reason}

    ``resolved`` is True only when an MLBAM id was pinned (fail-closed otherwise).
    ``game_pk`` is derived independently and may be present even when the player is
    unresolved (or absent — e.g. a doubleheader-ambiguous commence). MLB only;
    other sports and a blank name return unresolved. NEVER raises."""
    try:
        if sport_key not in _MLB_SPORT_KEYS or not name:
            return _unresolved("non_mlb_or_no_name")
        season = season or _season_of(game_date or commence)

        # game_pk: nearest game to the odds commence time (independent of the id).
        home_id = mlb_warehouse.team_id_for_name(home_team)
        away_id = mlb_warehouse.team_id_for_name(away_team)
        game_pk = (mlb_warehouse.find_game_pk_by_commence(home_id, away_id, commence)
                   if commence else None)

        # MLBAM id: fail-closed SFBB + roster ladder with the two-team hint.
        import mlb_starters
        res = mlb_starters.find_player_id(name, season, teams=[home_team, away_team])
        if not res:
            return _unresolved("ambiguous_or_unknown", game_pk=game_pk)
        mlb_id, is_pitcher = res

        # Record an audit alias ONLY when the name is globally unambiguous (safe to
        # key by bare name). A name that only resolves via the team hint is a shared
        # name → bare-aliasing it would later serve the wrong namesake, so skip it.
        method = "roster_or_hinted"
        try:
            import player_id_map
            if player_id_map.mlb_id_for_name(name, teams=None):
                method = "sfbb_unique"
                mlb_warehouse.record_player_alias(
                    "oddsapi", db_store.normalize_name(name), mlb_id,
                    confidence=1.0, method=method)
        except Exception:
            pass

        return {"resolved": True, "mlb_player_id": str(mlb_id), "game_pk": game_pk,
                "is_pitcher": is_pitcher, "confidence": 1.0, "method": method,
                "reason": None}
    except Exception:
        return _unresolved("resolver_error")
