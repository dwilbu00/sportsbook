# Step 2 Implementation Plan — MLB gamelogs ESPN → StatsAPI (true game-level)

> Reconstituted 2026-08-10 (the design-workflow output lost with the corrupted session).
> Grounded against the live tree by a 5-reader + 1-synth workflow (wf_faf463cf-e58).
> Incorporates the LOCKED key schema (surrogate id PK + NOT NULL(athlete_id,season_bucket,game_pk,team_id)
> + UNIQUE(athlete_id,season_bucket,game_pk); team_id NOT keyed) and the ordering fix.

All line refs verified against the live tree. NBA/NFL paths stay byte-identical throughout.
`db_store.reconcile` signature confirmed at `db_store.py:832`
(`reconcile(conn, table, desired, identity_cols, scope=None, ignore_cols=())`, raises `ValueError`
on dup identity per docstring `:872-877`). NOTE: suite is actually ~1055 test defs across 43 files
(grown past the ~1022 baseline).

---

## Commit 1 — `mlb_starters`: surface `game_pk` + `team_id` on the pitcher log; add `get_batter_gamelog`

**Why first:** the schema's `game_pk NOT NULL` / `team_id NOT NULL` contract is only satisfiable if the
fetch layer emits both. Landing this before the schema flip means no window where a live write can
produce a null-key row. Behavior-additive, does not touch the store → suite stays green on its own.

**Files touched**
- `mlb_starters.py:1506-1574` (`get_pitcher_gamelog`): stop popping `_gamePk` (local sort key, popped
  at `:1564`/`:1572-1573`); emit as `game_pk`. Add `team_id` from `sp.get("team",{}).get("id")`
  (MLBAM); update docstring `:1529-1530` that declares team_id intentionally omitted.
- `mlb_starters.py` (new, mirroring `get_pitcher_gamelog`): `get_batter_gamelog(player_name, season=None)`
  — resolve via `find_player_id` (`:1456`), **reject pitchers** → return `[]` so ESPN/synth fallback
  fires; iterate `_player_gamelog_splits(pid, "hitting", season)` (`:1478`); per split emit
  `AB/H/SO/BB/HBP/SF/SH` (= `_BATTER_STATS`) + `game_date, opponent, is_home, team_id (MLBAM),
  completed, game_pk (KEEP)`; sort newest-first by `(game_date, game_pk)` desc.
- `mlb_starters.py` (net-new Q2 boxscore pipeline): `get_boxscore(game_pk, max_age=CACHE_MAX_AGE)` over
  `GET game/{game_pk}/boxscore` via `_get` (`:144`); thin `_open_gamepks(date)` filtering
  `get_schedule_index(date)` (`:1342`) to not-`_is_genuine_final`. Bulk loader for the Q4 re-backfill;
  the per-athlete `_player_gamelog_splits` path is the cheaper per-player building block.

**Tests** — new `test_batter_gamelog.py` mirroring `test_pitcher_gamelog.py` (newest-first incl.
doubleheader game_pk tiebreak; pitcher-name → `[]`; game_pk/team_id present; completed rule);
new pitcher case (game_pk + team_id now present); Q2 pipeline test (mock boxscore _get).

**Risk / owner action**
- ⚠ **Verify against a live payload before coding:** exact hitting keys `hitByPitch`/`sacFlies`/
  `sacBunts`; `team` present in the gameLog split; boxscore endpoint version
  `/api/v1/game/{gamePk}/boxscore` vs `/api/v1.1/game/{gamePk}/feed/live` (confirm what the deployed
  `BASE_URL` at `mlb_starters.py:74` serves; boxscore used nowhere today).
- **team_id namespace change** (highest cross-module risk): batter team_id becomes MLBAM, but
  `props.py:1017/1033/1081` consume it as ESPN for venue + park factor. Interim fail-open loss until
  WS10 re-keys, unless Commit 4a lands.

---

## Commit 2 — schema: MLB gamelog tables gain `game_pk` + NOT NULLs + UNIQUE

**Files touched**
- `gamelog_store.py:102-116` (`_fact_table`): add `mlb=False`. When true: insert
  `Column("game_pk", String(32), nullable=False)`, set team_id `nullable=False`, append
  `UniqueConstraint("athlete_id","season_bucket","game_pk", name=f"uq_{name}_game")`. athlete_id/
  season_bucket already `nullable=False` (`:106-107`).
- `gamelog_store.py:119-122`: `mlb=True` on the two MLB tables only; NBA/NFL unchanged.
- `gamelog_store.py:165-166`: keep `_FACT_META_COLS` (NBA/NFL base); add
  `_MLB_FACT_META_COLS = (*_FACT_META_COLS, "game_pk")`.
- **Do NOT touch `_META_KEYS` (`:99`)** — `_reconstruct` (`:264-277`) is shared; adding game_pk there
  changes MLB dict shape (breaks roundtrip pins) and KeyErrors NBA/NFL. game_pk is SQL-layer only.
- `sql/schema.sql:391-405` / `:415-427`: fresh-DB form adds `game_pk NVARCHAR(32) NOT NULL`, flips
  team_id (`:401`/`:424`) to `NOT NULL`, adds `CONSTRAINT uq_mlb_{batter,pitcher}_gamelog_game
  UNIQUE (athlete_id, season_bucket, game_pk)`. Update header comment `:388`.
- `sql/schema.sql`: append **idempotent guarded ALTERs** (mirroring `prediction_log.player_key`
  pattern `:51-61`/`:94-122`): (i) ADD game_pk NULLABLE; (ii) guarded NOT NULL flip (NO-OP while any
  NULL); (iii) guarded UNIQUE add (NO-OP while any NULL/dup). All re-runnable.

**Tests** — SchemaParityTests `:396-404` point batter/pitcher at `_MLB_FACT_META_COLS`; nba/nfl
(`:406`/`:411`) literally unchanged; new UNIQUE + NOT NULL assertions on MLB only.

**Risk / owner action**
- ⚠ **PENDING OWNER ACTION — live Azure ALTER sequence.** `create_all()` (`:169`) touches only the
  SQLite test DB; prod DDL is hand-run. Live tables hold rows with no game_pk + possibly NULL team_id,
  so NOT NULL + UNIQUE can't fire until data is clean. Order **must be** exactly:
  **(1)** run only "ADD game_pk NULLABLE" now; **(2)** deploy Commits 1-3 (reconcile's ValueError
  fallback self-heals legacy NULL-game_pk partitions on first re-fetch); **(3)** run the Q4 re-backfill
  (current + prior season) to populate real game_pk/team_id on every row; **(4)** re-run schema.sql —
  guarded NOT NULL + UNIQUE now fire. If UNIQUE errors, a residual dup slipped through:
  `SELECT athlete_id, season_bucket, game_pk, COUNT(*) ... HAVING COUNT(*)>1`.

---

## Commit 3 — writer swap to `db_store.reconcile` + ordering fix (CRITICAL; both land together)

**Files touched**
- `gamelog_store.py:248-261` (`_row_params`): add `with_game_pk=False`; when true
  `params["game_pk"] = str(pk) if pk else "synth:" + _game_key(row)` (NOT NULL for dateless rows).
  NBA/NFL keep default → byte-identical.
- `gamelog_store.py` (new `_dedupe_by_game_pk`): drop dup `(athlete_id,season_bucket,game_pk)` keeping
  newest-first — real gamePks never collide, so only fires on two dateless synth rows sharing
  `"synth:"+game_key`. Guards UNIQUE + reconcile's dup-in-desired ValueError (`db_store.py:912-914`).
- `gamelog_store.py:458-467` (writer): branch on `"game_pk" in table.c`. **MLB:** build `desired` via
  `_dedupe_by_game_pk([_row_params(..., with_game_pk=True) ...])`; `db_store.reconcile(conn, table,
  desired, ("athlete_id","season_bucket","game_pk"), scope={"athlete_id":..., "season_bucket":...})`
  (mirrors WS15 `statcast_asof.put_rates` 433781e). Wrap in `except ValueError` → scoped delete+insert
  rebuild (documented fallback for legacy multi-NULL-game_pk partitions; desired is deduped so
  re-insert is UNIQUE-safe). **NBA/NFL:** existing delete+insert byte-identical. The
  `range(3)`/`OperationalError` retry (`:438-471`) + `_key_lock` (`:428`) unchanged.
- `gamelog_store.py:289` (`_read_rows`) — **REQUIRED CORRECTNESS FIX, this commit only:** replace
  `.order_by(table.c.id)` with MLB-conditional
  `order = (table.c.game_date.desc(), table.c.game_pk.desc()) if "game_pk" in table.c else (table.c.id,)`.
  Reconcile UPDATEs re-fetched games in place (keeps old id) and only APPENDs new games, so id no
  longer proxies recency; game_date is full ISO (lexical DESC == chronological), game_pk DESC breaks
  doubleheader ties, dateless synth rows sort last (SQL Server + SQLite). NBA/NFL keep id.

**Why the ordering fix is load-bearing** — public reader contract is "most-recent-first, caller slices
`[:n]`" (`:407-411`). Downstream consumers that **silently** regress (stale games / leakage) if order
stops tracking recency: `espn_client.get_player_stat_history:1068` (`gamelog[:n]`, feeds live
projection; `props.py:1008` asserts most-recent-first); `book_line_calibration.py:438/450`
(`gamelog[idx+1:]` as-of leakage slice needs newest-first); `backtest.py:2463/2488/2545`. **No exception
is raised on regression** → the suite would stay green while accuracy silently degrades. Hence the pin.

**Tests** — ordering-fix pin (scrambled insertion order; assert game_date DESC, game_pk tiebreak,
synth-last); surgical-upsert (changed game UPDATEd in place, new appended, vanished deleted); ValueError
→ scoped-rebuild fallback; synth game_pk; team_id in-place UPDATE (WS10-shape: changing only team_id
updates the row, no new row); confirm still-green roundtrip/rollover/clobber/concurrency.

---

## Commit 4 — batter StatsAPI-first fetch branch + consumer decoupling (home/away + layoff)

**Files touched**
- `gamelog_store.py:203-223` (`_fetch_espn`): batter-StatsAPI-first branch symmetric to pitcher,
  **gated on player_name known** — call `mlb_starters.get_batter_gamelog(player_name, season_year)`
  first, fall back to `get_athlete_gamelog` on empty, then existing synth. When `player_name is None`
  (most tests) the branch is skipped → byte-identical. season_year threaded (enables Q4 re-backfill).
- **Consumer decoupling for the team_id namespace flip — OWNER DECISION REQUIRED:**
  - **4a (recommended, name-based):** persist the player's OWN-team NAME (new stored column beyond the
    locked schema): `_META_KEYS`, `_fact_table`, `_row_params`, `_reconstruct`; propagate in
    `espn_client.get_player_stat_history:1086-1097` as `result["team_name"]`; `props.py:1077-1086`
    resolves `upcoming_is_home` from `history["team_name"]` via the tolerant matcher;
    `props.py:1032-1034` looks up `team_schedules` by name — all MLB-gated, id-path kept for NBA/NFL.
    Removes home/away + layoff from the id space entirely; immune to WS10's later re-key.
  - **4b (strict id lockstep):** store MLBAM team_id AND flip `id_to_name` (`props.py:945`) +
    `schedule_results` (`app.py:2748-2804`) to MLBAM in the SAME commit — contradicts Q3/WS10 deferral.

**Tests** — batter-StatsAPI-first with player_name set; player_name=None → ESPN byte-identical;
explicit anti-silent-regression: assert upcoming_is_home resolves + team_schedule found post-migration.

---

## Commit 5 — Q1 team listing + net-new StatsAPI runs-allowed `team_defense` feed (MLB-gated)

**Files touched**
- `mlb_starters.py` (net-new): `get_team_defense_statsapi(season, as_of=None)` over
  `GET schedule?sportId=1&season={season}&gameType=R&hydrate=linescore`, parsing
  `linescore.teams.{home,away}.runs` (pattern proven at `game_results.py:228-232`), keyed by StatsAPI
  club name. Must emit the **three structures** `backtest._team_defense_lookup` produces
  (`backtest.py:2247-2285`): `avg_lookup`, `series_lookup` (**sorted DESC**), `league_avg`. Optional
  `as_of` for leakage-safe slicing (mirror `_resolve_opp_pa_asof` `backtest.py:2288-2319`).
- `app.py:2696-2817` (MLB only): replace `fetch_espn_teams`/`get_all_teams` with
  `mlb_starters.get_team_index` (`:723`)/`_match_team_id` (`:743`); replace
  `get_team_schedule`→`build_team_defense_lookup` (`app.py:2817`) with the net-new feed; key
  `schedule_results` by team NAME for MLB. NBA/NFL keep ESPN.
- `app.py:2932-2964` (MLB spread/total path): `get_team_index` lacks
  `record/wins/losses/win_pct/display_name/short_name`, so this path + `annotate_opponent_strength`
  (`espn_client.py:382`) degrade — **larger blast radius than props**; rework or separate MLB path.
- `backtest.py:2247-2285`: MLB branch sources from the new StatsAPI helper (Q4 uniform basis).

**Tests** — runs-allowed helper shape parity (DESC series contract `test_calibration_refit.py:21`;
as-of variant matches `_resolve_opp_pa_asof`); keep green: `test_pitcher_ip_conversion.py:172-173/
235-236` patch `backtest.get_all_teams` + `backtest._team_defense_lookup`; `test_prediction_log.py:174`.

---

## Q4 re-backfill (operator run, not a commit)

Existing `refit_calibration.py` two-pass sweep (`refit_sport:662-721` — current `:693-702`, prior
`:711-721`) → `fetch_player_data(season_year=...)` → `cached_gamelog(season_year=...)` →
`gamelog_store.get_gamelog`, landing in the immutable per-year bucket (`:417`). After Commits 1-4 this
writes StatsAPI rows through reconcile with real game_pk/team_id. Must pass season_year and be
comprehensive — any partition never re-fetched keeps NULL game_pk and blocks the NOT NULL/UNIQUE flip.
Run current + prior season before the terminal `refit_calibration.py --sport mlb`. ⚠ INCUMBENT
HYSTERESIS: a bare re-sweep resets incumbents to A; splice back within-gate-band methods (batter_hits E)
first.

---

## URGENT — still-open owner action
🔴 **Rotate the leaked Odds API key** (present in git history; from the third-party audit eval).

---

## OPEN QUESTIONS FOR OWNER
1. **team_id namespace / consumer decoupling (4a vs 4b) — blocks Commit 4 and shapes the schema.**
   4a add stored own-team-NAME column (one column beyond locked schema; clean, WS10-immune, recommended)
   vs 4b strict id lockstep (contradicts Q3/WS10). Neither = interim silent fail-open loss of home/away
   + layoff + park for batters.
2. **Team-market path scope (Commit 5).** `get_team_index` drops `record/win_pct/display_name`,
   degrading MLB spread/total (`app.py:2932-2964`) + `annotate_opponent_strength`. Rework now / separate
   MLB path / accept interim degradation?
3. **Pitcher team_id source under NOT NULL.** Commit 1 emits `sp.team.id`; ok vs. backfilling from the
   athlete's season team?
4. **Live-migration confirmation.** Approve the 4-step Azure ALTER sequence + confirm the re-backfill
   covers every athlete/bucket (else belt-and-suspenders `UPDATE ... SET game_pk='synth:'+game_key
   WHERE game_pk IS NULL` for never-refetched rows — cannot fix NULL team_id).
5. **Endpoint verifications before coding (Commit 1):** boxscore path version; hitting keys; team in
   split; season-wide `schedule?season=` un-paged.
6. **identity_cols convention:** plan uses `("athlete_id","season_bucket","game_pk")` + `scope`
   (WS15-faithful, redundant-but-safe) vs. leaner `("game_pk",)` within scope (both correct).
