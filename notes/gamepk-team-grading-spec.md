# Tier A #2 — game_pk team-market grading + DH fix (implementation spec)

Design workflow wf_e275cb96-045. Verdict GO_WITH_FIXES; NON_DESTRUCTIVE; DH-safety GAPS (2 medium fixes folded in). Winning = minimal-additive (51) hardened w/ dh-edge-first GRADE_PENDING + resolver-consistency.

## Contract
Additive; byte-identical when game_pk absent; MLB-only (sport_key=='baseball_mlb'); no DDL (odds_line.game_pk db_store.py:405, wagers.game_pk db_store.py:435 exist); normalize_name/_norm untouched; backfill dry-run-first, owner --apply; NEVER auto-regrade settled bets.

## Commit 1 — read primitive + grader (INERT, no caller)
### mlb_warehouse.final_game_by_pk (NEW, near get_game ~1024)
g=get_game(game_pk); None/SQL-off -> None. get_game returns SNAKE_CASE.
- terminal if str(detailed_state).lower() contains any of mlb_starters._NON_FINAL_DETAILED ('postpon','suspend','cancel')
- final if status=='Final' AND not terminal AND home_score is not None AND away_score is not None (use `is None` so 0-0 grades)
- else live
Return {'state':'final'|'live'|'terminal','home_score','away_score','home_team_id','away_team_id','commence_time'(=game_date)}.
MUST NOT call mlb_starters._is_genuine_final (camelCase status/detailedState; row is snake_case).
Fail: row missing/SQL off/bad ids -> None. Never raises.

### game_results.grade_team_bet_by_game_pk (NEW) + GRADE_PENDING sentinel
Module const GRADE_PENDING=object().
grader(sport_key, game_pk, bet_type, side, team, point) -> (status,'hs-as') | GRADE_PENDING | None.
- guard sport_key!='baseball_mlb' or not game_pk -> None
- lazy import mlb_warehouse; fg=final_game_by_pk(game_pk)
- fg None -> None (fallback)
- state=='live': **staleness fix** — if commence+6h elapsed -> None (fallback, provably DH-safe: stamped pk always commence-disambiguable); else GRADE_PENDING
- state=='terminal' -> None (fallback; next-day makeup grades)
- state=='final': total -> graded_side=side (orientation-invariant); ml/spread -> tid=team_id_for_name_tolerant(team); 'home' if str(tid)==str(home_team_id) elif 'away' if ==away_team_id else None. status=grade_team_bet(bet_type,graded_side,point,hs,as_); None->None; else (status, f'{hs:g}-{as_:g}')
- whole body try/except -> None
Uses team_id_for_name_TOLERANT (matches stamp resolver).

## Commit 2 — wire live grading
### wagers._grade_wager team branch (~443), BEFORE final_score(...)
game_pk=row.get('game_pk')
if row.get('sport_key')=='baseball_mlb' and game_pk:
  fast=game_results.grade_team_bet_by_game_pk(sport_key, game_pk, bet_type, row.get('side'), row.get('team'), row.get('point'))
  if fast is game_results.GRADE_PENDING: return None
  if fast is not None: return fast
Existing name+date block below UNCHANGED.

## Commit 3 — enrich-time stamping
### wagers._enrich_ids (~127): else branch to `if row.get('player')` for team bet_types
import mlb_warehouse; hid/aid=team_id_for_name_tolerant(home/away); if hid and aid and commence_time: row['game_pk']=find_game_pk_by_commence(hid,aid,commence_time). MLB-gated by startswith('baseball').
### warehouse._enrich_ids (~313): compute _team_gpk ONCE before per-line loop
hid/aid=team_id_for_name_tolerant(meta home/away); _team_gpk=find_game_pk_by_commence(...) if hid&aid&commence else None. team-line else branch add ln['game_pk']=_team_gpk. capture_odds_snapshot already persists ln['game_pk'].

## Commit 4 — team-anchor backfill (mirror P5 backfill_legacy_game_pk)
### mlb_warehouse.backfill_team_game_pk(dry_run=True, season=None) + CLI --backfill-team-game-pk (reuse --apply)
wagers: SELECT id,home_team,away_team,game_date,commence_time WHERE sport_key='baseball_mlb' AND game_pk IS NULL AND player IS NULL AND bet_type IN (ml,spread,total).
odds_line: join odds_snapshot; WHERE sport='baseball_mlb' AND bet_type IN(...) AND odds_line.game_pk IS NULL; resolve ONCE per snapshot_id, fan pk to its lines.
resolve per row (WRAP in try/except continue — tz-naive fix): hid/aid=team_id_for_name_tolerant; gpk=find_game_pk_by_commence(hid,aid,commence); if None and game_date: gpk=find_game_pk(game_date[:10],hid,aid) (unique-only). collect {rid,gpk} only when gpk not None.
apply (not dry_run): _WRITE_LOCK; update(table).where((id==rid)&game_pk.is_(None)).values(game_pk=gpk) bulk bindparam.
return {'dry_run','wagers':{'candidates','matched'},'odds_line':{'candidates','matched'}}.
Idempotent (game_pk IS NULL in SELECT+UPDATE); non-destructive (only writes game_pk).

## Deferred
#2b backtest DH-join consumption (backtest.py _build_odds_lookup/_lookup_game_odds/all_completed_games + db_store.team_market_lines SELECT game_pk + warehouse._assemble_team_entry/load_team_market_store key + mlb_warehouse._team_final_games dict). DATA half (seams 3-4) ships now. MUST also fix all_completed_games (date,home,away) dedup + load_team_market_store key.
#2c offline recalibration.resolve_pending_market_outcomes — market_prediction_log has NO game_pk col; needs ALTER + log-time stamp + backfill, then reuse grade_team_bet_by_game_pk.

## Owner-gated open question
After --apply, run a one-time regrade_wagers over legacy PENDING DH team bets? Default: NO auto-regrade (never-destroy); leave to owner.

## Low-sev deferred: actual-string orientation (cosmetic); split-DH-postponed-half name+date (== today); stamp freeze <20h (rare).
