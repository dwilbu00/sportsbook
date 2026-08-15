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

Design (SFBB-first, structurally fail-closed, no hot-path network):
  * MLBAM resolution = player_id_map.mlb_id_for_name — bare name first (globally
    unique → accept + bare-alias), else narrowed by the game's BOTH teams. The
    two-team want-set is structurally fail-closed: a name shared by two players IN
    THIS GAME stays None. Deliberately NOT find_player_id — its season-wide
    unique-exact fallback could bind an in-game namesake merely absent from an
    incomplete season index, and it fetches statsapi rosters (unwanted on the hot
    prediction path). An SFBB-missing player → None = the P3 shadow signal.
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

import datetime
import os

import db_store
import mlb_warehouse

# The MLB sport-key prefix this resolver handles (the Odds API + app key is
# "baseball_mlb"). Other baseball keys (e.g. "baseball_kbo") and NBA/NFL stay on
# the ESPN path untouched.
_MLB_SPORT_PREFIX = "baseball_mlb"


# ── Commit C — STAMP-resolver flip (env-gated, default OFF) ───────────────────
# When ON (+ MLB), the identity STAMP this envelope writes at prediction time
# delegates its id-CORE to the game-context/role-verified resolver
# (mlb_starters.resolve_mlbam_id: today's lineup/probables → statsapi season-roster
# unique-exact → role-checked SFBB) instead of the SFBB-only, two-team-hinted,
# role-BLIND player_id_map lookup below. The envelope is UNCHANGED either way — it
# still owns game_pk derivation, the schedule gap-fill, and the resolved/crash dict
# contract the P4 circuit breaker (props._enforce) reads. OFF (the default) is
# byte-identical to the pre-Commit-C behavior. Mirrors the espn_client warehouse
# gate helpers; boot-promoted from st.secrets in app.py and reported by
# espn_client.mlb_warehouse_gate_status (key "stamp_resolver").
_STAMP_RESOLVER_ENV = "ODI_MLB_STAMP_RESOLVER"


def _stamp_resolver_enabled():
    return os.environ.get(_STAMP_RESOLVER_ENV, "").strip().lower() in (
        "1", "true", "on", "yes")


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
    lineup / announced probables) feed the game-context/role-verified id-core only
    when the ``ODI_MLB_STAMP_RESOLVER`` gate is ON (Commit C); with the gate OFF
    (default) they are ignored and the SFBB-only lookup runs unchanged."""
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

        # Commit C id-core (env-gated): delegate to the game-context/role-verified
        # resolver. It resolves TODAY'S posted game (lineup/probables) first — trade-
        # aware + namesake-safe — then statsapi season-roster unique-exact, then a
        # role-checked SFBB fallback (rejects the cross-role drift the SFBB-only path
        # below can't see). It also fails CLOSED (None) on ambiguity/unknown AND
        # swallows infra errors to None, so a None here is treated as an ordinary
        # unresolved (fail-open shadow row); a systemic outage is caught by the
        # slate-level ≥50%-unpinned circuit breaker in props, not conflated here. The
        # bare-name audit alias is deliberately NOT written on this path: the id is
        # context-specific (role/team/lineup), not proven globally unique, so a bare
        # alias could later serve the wrong namesake — and the alias table is write-
        # only (never read) anyway. is_pitcher is statsapi-authoritative here.
        if _stamp_resolver_enabled():
            try:
                import mlb_starters
                found = mlb_starters.resolve_mlbam_id(
                    name, season, prop_key=prop_key,
                    teams=[home_team, away_team],
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

        # MLBAM id via the SFBB map, FAIL-CLOSED on namesake ambiguity. Try the
        # bare name first (globally unique → safe to bare-alias); else narrow by the
        # game's BOTH teams — structurally fail-closed: a name shared by two players
        # IN THIS GAME stays None rather than risk a mis-bind. Deliberately SFBB-only
        # (no statsapi roster fetch on the hot prediction path, and NO season-wide
        # unique fallback, which could bind an in-game namesake that is merely absent
        # from an incomplete season index). An SFBB-missing player → None = the P3
        # shadow signal (review it, update the map), never a guess.
        import player_id_map
        mid_bare = player_id_map.mlb_id_for_name(name, teams=None)
        mlb_id = mid_bare or player_id_map.mlb_id_for_name(
            name, teams=[home_team, away_team])
        if not mlb_id:
            return _unresolved("ambiguous_or_unknown", game_pk=game_pk)

        # Record an audit alias ONLY when the name is globally unambiguous (safe to
        # key by bare name). A name that only resolved via the team hint is shared →
        # bare-aliasing it would later serve the wrong namesake, so skip the write.
        method = "sfbb_unique" if mid_bare else "sfbb_hinted"
        if mid_bare:
            mlb_warehouse.record_player_alias(
                "oddsapi", db_store.normalize_name(name), mlb_id,
                confidence=1.0, method=method)

        return {"resolved": True, "mlb_player_id": str(mlb_id), "game_pk": game_pk,
                "is_pitcher": None, "confidence": 1.0, "method": method,
                "reason": None}
    except Exception:
        return _unresolved("resolver_error")
