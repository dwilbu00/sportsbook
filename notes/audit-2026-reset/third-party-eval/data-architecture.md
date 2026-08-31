# Third-party audit evaluation — Cluster: Data architecture & game-centric MLB ingestion

Scope: audit Priorities **P1** (gamePk + MLBAM canonical player-game key), **P5** (Azure SQL as
sports-data warehouse; model reads local not external), **P8** (native SQL temporal types), and the
**entire `mlb_statsapi_architecture_summary.txt`** (game-centric boxscore ingestion, cache-by-gamePk,
incremental store, provider routing).

Verdict summary: the auditor's central thesis for this cluster is **substantially CORRECT and NOT
already done**. The MLB gamelog data layer is still player/season-centric and ESPN-shaped, keyed by
ESPN athlete_id + a synthetic `date|opp|home` game key. The SFBB MLBAM-id migration that the work-log
reports "complete" reached the *betting/prediction* tables (prediction_log, wagers, odds_line) but did
**not** reach the *gamelog fact tables*. gamePk exists in the codebase but only for grading, never for
storage. So P1 and the StatsAPI doc are genuine, high-value gaps. P5 is largely realized in principle
(model reads a local SQL cache) but not as a game-centric warehouse. P8 is absent but low-value churn.

---

## Evidence base (what the code actually does today)

### How batter gamelogs are fetched & stored — PLAYER/SEASON-CENTRIC, ESPN
- `espn_client.get_athlete_gamelog` (espn_client.py:683-707) hits
  `site.web.api.espn.com/.../athletes/{athlete_id}/gamelog?season=...` — **one player's whole season**,
  ESPN athlete-id keyed. This is the auditor's exact "Player -> ESPN -> entire season gamelog" premise.
- Stored via `gamelog_store.get_gamelog` into `mlb_batter_gamelog`, keyed on `athlete_id` +
  `season_bucket` (gamelog_store.py:102-119; schema.sql:376-389). Columns are ESPN labels
  (`AB,H,SO,BB,HBP,SF,SH`).

### How pitcher gamelogs are fetched & stored — StatsAPI but STILL PLAYER-CENTRIC
- On an empty ESPN gamelog, MLB falls back to `mlb_starters._pitcher_gamelog_or_synth`
  (gamelog_store.py:215-223; espn_client.py:1035-1038).
- That calls `get_pitcher_gamelog` (mlb_starters.py:1506-1574) which calls `_player_gamelog_splits`
  (mlb_starters.py:1478-1495) → StatsAPI `people/{pid}/stats?stats=gameLog&group=pitching&season=...`.
  **This is StatsAPI, but it is a per-PLAYER-per-SEASON gameLog pull — NOT a boxscore-by-gamePk fetch.**
  So even the "StatsAPI" path the work-log celebrates is player-centric ingestion, just from a better
  provider (real per-game rows with dates + variance).
- Notably, `get_pitcher_gamelog` reads `gamePk` from each split as a **local sort tiebreak** and then
  **`r.pop("_gamePk")`** — it is discarded, never stored (mlb_starters.py:1564,1570-1573). The
  identifier the auditor wants as canonical is fetched and thrown away.
- StatsAPI ids are MLBAM but pitcher rows deliberately omit `team_id` "StatsAPI ids are MLBAM, not
  ESPN" (mlb_starters.py:1529-1530) — an explicit acknowledgement that the two id-spaces are not
  reconciled in the gamelog layer.

### Is gamePk stored as canonical game id? NO — synthetic key
- `gamelog_store._game_key(row)` = `f"{game_date}|{opponent}|{is_home}"` (gamelog_store.py:244-245),
  stored in `game_key NVARCHAR(220)` and explicitly "**synthetic (not unique)**" (gamelog_store.py:107;
  schema.sql:381). This is *verbatim* the `game_date | opponent | home/away` key the StatsAPI doc §5 and
  P1 call out as the thing to replace.
- `grep boxscore` across the whole repo → matches ONLY in the two audit text files. There is **no
  boxscore-by-gamePk ingestion anywhere.**

### Is gamePk used at all? Yes — for GRADING only, never storage
- `mlb_starters.get_schedule_index(date)` returns `{gamePk: {...}}` (mlb_starters.py:1342-1420) and
  `resolve_player_game_stat` (mlb_starters.py:1611+) resolves outcomes via the gamePk hard id +
  commence_time (doubleheader disambiguation). So the game-centric *primitive* (schedule → gamePk) is
  already in the codebase and battle-tested for grading — it is simply not wired into ingestion/storage.

### Is there DELETE-then-INSERT churn? YES — replace-all per (athlete, season_bucket)
- Slow path: `delete(table).where(athlete_id==... & season_bucket==...)` then bulk `insert(...)`
  (gamelog_store.py:458-467). Module docstring states it plainly: "**Replace-all per (athlete,
  season_bucket). ESPN returns the whole current season with no 'since' filter, so there is no
  ESPN-level incremental fetch**" (gamelog_store.py:20-25). This is the auditor's
  "delete athlete/season cache → insert entire season again" (P1 diagram) exactly. It is guarded by a
  clobber check (`_should_replace`) but is still full-season replace, not incremental append.

### Provider routing? PARTIAL, and not the clean split the doc recommends
- Batters + NBA + NFL: ESPN (`get_athlete_gamelog`). MLB pitchers only: StatsAPI (player-centric) with
  ESPN-synth fallback. NHL/other: pass-through ESPN, no persistence (gamelog_store.py:199-223,412-415).
- So routing exists partially (MLB pitcher → StatsAPI), but MLB **batters are still ESPN**, and nothing
  is game-centric. The doc's target ("MLB → StatsAPI / other → ESPN, both normalized to one gamelog
  interface") is only ~⅓ realized.

### Does the model read local not external (P5)? PARTIAL — a TTL cache, not a warehouse-first read
- `get_player_stat_history` (espn_client.py:1008-1021) routes through `gamelog_store.get_gamelog` when
  `db_store.enabled()`. Fresh meta → served from SQL with **0 external calls** (gamelog_store.py:421-425).
  Past seasons get a 5-year TTL → fetched once, then permanent local (gamelog_store.py:59,386-388).
  `statcast_player_asof` is a derived local table read live by props (schema.sql:504-537). The
  `odds_snapshot`/`odds_line` warehouse exists (schema.sql:296-353).
- BUT it is a **TTL cache with replace-all refresh**, not a "warehouse is canonical, external only on
  miss" model. The current-season path re-fetches ESPN every LIVE_TTL_HOURS=4 and the batter source is
  still ESPN full-season. When SQL is OFF, the read silently drops to the ephemeral file cache /direct
  ESPN — the exact silent-fallback WS1 is meant to make fail loud.

### P8 — temporal types: ABSENT (confirmed by full schema scan)
- `grep -iE 'datetime2|datetimeoffset|\bdate\b|\bdatetime\b'` over sql/schema.sql → **zero hits.** Every
  timestamp is `NVARCHAR(40)` (commence_time, captured_at, resolved_at, fit_timestamp, ts) or `FLOAT`
  epoch (last_fetched_at, fetched_at) — schema.sql:16,148,257,308,441,585 etc.

### P7 (adjacent, not in-cluster) — no foreign keys
- `grep -iE 'foreign key|references'` → only a *comment* at schema.sql:335. `odds_line.snapshot_id` has
  no FK to `odds_snapshot.id`. Noted for context; not scored as a cluster finding.

---

## Per-priority evaluation

### P1 — Build the MLB data layer around gamePk + MLBAM player id
**current_state: ABSENT (for the gamelog layer).** Gamelog fact tables key on ESPN athlete_id +
synthetic `date|opp|home`; no `game_pk`, no `mlb_id` columns (schema.sql:376-411; gamelog_store.py:102-119,
244-245). The MLBAM migration reached prediction_log/wagers/odds_line (`player_key`, `player_mlb_id`,
schema.sql:65-73,171-179,355-357) but **stopped short of the gamelog fact tables**. gamePk is fetched
then discarded in the pitcher path (mlb_starters.py:1564,1573).

**verdict: ADAPT.** The recommendation is correct and valuable, but scope it to fit philosophy rather
than adopting the full rewrite verbatim:
- Make `(game_pk, player_mlb_id)` the canonical player-game key **for MLB**, sourced from StatsAPI
  boxscores (see StatsAPI-doc finding). Keep ESPN athlete_id + name as *evidence/display*, exactly the
  fail-closed identity philosophy already applied elsewhere.
- Do it **additively**: add `game_pk` + `player_mlb_id` columns to `mlb_batter_gamelog`/
  `mlb_pitcher_gamelog` (nullable at first, then unique `(game_pk, player_mlb_id, season_bucket)`),
  mirroring the *proven* prediction_log Phase-3→Phase-4 additive-then-swap pattern (schema.sql:62-121).
  This avoids a big-bang and keeps every step reversible.
- Leakage-safety actually *improves*: gamePk disambiguates doubleheaders exactly, which the synthetic
  key cannot; the as-of slice (`prior_games = gamelog[idx+1:]`) keeps working on game_date ordering.

**risks/traps:** (1) ESPN athlete_id ↔ MLBAM id bridge already exists (`player_id_map.espn_id_for_mlb_id`,
used at gamelog_store.py:488-490) so backfilling player_mlb_id onto existing ESPN-keyed rows is feasible
but must fail *closed* (leave NULL, don't guess) to honor the "false-positive identity is worse than a
skip" rule. (2) The synthetic `game_key` is currently used by tests/read-order; keep it during the
additive phase. (3) This priority is **coupled to and largely subsumed by** the StatsAPI-doc workstream
below — do them as one workstream, not two.

### StatsAPI architecture doc — game-centric boxscore ingestion + incremental store
**current_state: ABSENT (core), partial only in that the StatsAPI *dependency* and the schedule→gamePk
primitive already exist.** No boxscore-by-gamePk fetch (grep: audit files only); pitcher StatsAPI path
is player-centric (mlb_starters.py:1489-1490); ingestion is replace-all-per-athlete-season, not
incremental (gamelog_store.py:20-25,458-467); provider routing is partial (batters still ESPN).

**verdict: ADAPT (this is the highest-value item in the cluster).** Build game-centric MLB ingestion:
schedule(date) → gamePk set → `/game/{gamePk}/boxscore` → all-players' batting+pitching lines →
normalize to the existing gamelog dict shape (StatsAPI H/SO/IP/ER → the app's AB/H/SO/BB/IP/K/ER;
doc §7 mapping) → UPSERT into gamePk+MLBAM-keyed fact rows → served through the unchanged
`get_player_stat_history` interface. Adapt (not adopt-verbatim) because philosophy constraints must be
honored:
- **Keep ESPN as fallback** (doc Phase 3-4 agree) — never a hard cutover; fail-open to the current path
  so nothing regresses if StatsAPI is unavailable.
- **Reuse existing primitives:** `get_schedule_index` (mlb_starters.py:1342), the `_get`/cache plumbing,
  and the boxscore endpoint pattern already used for grading. This is an *extension of an existing
  dependency*, not a new provider.
- **Leakage-safe by construction:** ingest only Final games; game_date + gamePk give exact as-of
  ordering. This is materially *better* than today's synthetic key.
- **Incremental store** replaces the replace-all churn: append newly-Final games, never re-download
  history — the doc's central efficiency + reproducibility win (§9, §12), and it turns the SQL store
  into a true backtest warehouse (feeds P5).

**Why it matters for THIS app (beyond the doc's generic efficiency argument):**
- **Batters gain real per-game rows the same way pitchers already did.** Batters still come from ESPN
  full-season today; a boxscore ingest gives them dated, per-game, variance-preserving rows keyed to
  MLBAM ids — the same upgrade that made pitcher real-line calibration possible.
- **One reproducible warehouse** for backtests independent of ESPN uptime (the roadmap's recurring pain:
  thin pitcher n, stale synthetic calib) — more clean historical rows accrue automatically.
- Efficiency (15 boxscore calls vs ~180 player-season pulls per slate) is real but secondary here,
  because the SQL TTL cache already bounds redundant fetches; the identity/reproducibility wins dominate.

**risks/traps:** (1) StatsAPI boxscore field names differ from the gameLog splits used today — normalize
carefully and pin with tests (the app already has `test_pitcher_gamelog.py`/`test_gamelog_store.py`
harnesses to extend). (2) MLBAM id space ≠ ESPN id space; park/team lookups currently key off ESPN
team_id (mlb_starters.py:1529-1530) — the boxscore path must map MLBAM team_id → ESPN/park keys via the
existing `team_id_map` (schema.sql:605) or fail-open. (3) Scope is L; stage it exactly per the doc's
Phase 1-5 so each phase is shippable and reversible. (4) **Does NOT resolve WS9** — the
batter_strikeouts gap is an *odds-vendor/market-offered* gap (zero warehouse lines), not a gamelog-stat
gap; StatsAPI boxscore trivially yields batter SO if that market ever becomes active, but the WS9
blocker is upstream and unrelated.

### P5 — Treat Azure SQL as the sports-data warehouse; model reads local not external
**current_state: PARTIAL (largely realized in spirit).** Model reads gamelogs from local SQL when
TTL-fresh (0 external calls; espn_client.py:1008-1021, gamelog_store.py:421-425); past seasons are
fetched-once/permanent (5yr TTL); statcast_player_asof + odds warehouse are local derived tables. BUT
it's a TTL cache with replace-all refresh, not a warehouse-first read, and the batter source is still
ESPN full-season.

**verdict: ADAPT (mostly done; fold the delta into the game-centric workstream).** The explicit
"warehouse is canonical, external is an ingestion source" principle is worth stating and the concrete
delta is: (a) game-centric incremental ingest (above) makes the MLB gamelog store a real warehouse
rather than a cache, and (b) WS1's fail-loud-on-SQL-off directly implements the audit's "modeling layer
queries the local canonical DB" intent by refusing to silently fall back to ephemeral disk. No net-new
work beyond P1/StatsAPI + WS1; do not spin a separate effort.

**risks/traps:** Don't over-engineer toward a "pure warehouse, never touch external at predict time" —
the live path legitimately needs same-day games (doubleheader game-2 after game-1 finished); the
4h/negative-TTL design (gamelog_store.py:47-58) is a deliberate, correct compromise. Preserve it.

### P8 — Move important timestamps to native SQL temporal types
**current_state: ABSENT (confirmed).** All timestamps NVARCHAR(40)/FLOAT; zero DATETIME2/DATETIMEOFFSET
(schema scan: no hits).

**verdict: DEFER (lean toward reject-for-now).** ISO-8601 text sorts lexicographically = correct
ordering, and the app parses via `datetime.fromisoformat` (gamelog_store.py:297-301). Migrating every
timestamp touches all readers/writers + the SchemaParityTests contract for little modeling value on a
single-writer Streamlit + Azure SQL app. It violates nothing in the philosophy but is churn>value today.
**Trigger to revisit:** when native SQL analytics / retention policies / server-side range queries over
the warehouse become a real need (i.e., after the game-centric warehouse from P1/StatsAPI is in place
and someone actually wants to run T-SQL date analytics on it). If revisited, do it additively
(shadow DATETIME2 columns populated alongside the text) — never an in-place type swap.

**integration:** backlog.

---

## Integration recommendation
- **Propose one NEW workstream — "WS10: game-centric MLB player-game warehouse (gamePk + MLBAM)"** —
  that absorbs **P1 + the entire StatsAPI doc + the P5 delta**. They are one architecture, not three.
  Stage per the doc's Phase 1-5 (add StatsAPI boxscore ingest → add gamePk/mlb_id columns + UPSERT →
  route MLB reads to the local store → retire routine ESPN MLB gamelog pulls → optional PBP/live). Every
  phase additive + reversible + ESPN-fallback-preserved, mirroring the proven prediction_log identity
  migration.
- **WS1 synergy:** the "model reads local not external" (P5) intent is partly delivered by WS1's
  fail-loud-on-SQL-off. Cross-reference; don't duplicate.
- **P8 → backlog** with the trigger above.
- Effort: WS10 = **L** (stage it). P8 = **M** if ever done (broad but mechanical).

---

## Verifier verdict

Independent re-check against the actual code. Every cited file:line was opened and spot-checked;
the raw `current_state` labels are all CORRECT. Two rationales overstate the payoff and one
priority claim needs tempering — details below.

**Confirmed facts (all citations hold):**
- No boxscore ingestion anywhere — `grep boxscore` hits only the two audit text files.
- Gamelog fact tables key on `athlete_id + season_bucket + synthetic game_key` with no
  `game_pk`/`player_mlb_id` columns (schema.sql:376-418; gamelog_store.py:102-119,244-245). The
  MLBAM migration reached prediction_log/wagers/odds_line ONLY (schema.sql:65-73,172-173,356-357).
- Pitcher path is StatsAPI but player-per-season gameLog splits (mlb_starters.py:1489-1490); gamePk
  is read as a sort tiebreak then `r.pop("_gamePk")` — discarded, never stored (1564,1573).
- Store is replace-all delete+insert per (athlete, season_bucket) (gamelog_store.py:459-467); no
  incremental append.
- Read path is local-SQL-first with 0 external calls on fresh meta (espn_client.py:1008-1021;
  gamelog_store.py:421-425); past seasons permanent at 5yr TTL (gamelog_store.py:59,386-388);
  SQL-off silently drops to direct-ESPN + ephemeral file cache (espn_client.py:1022-1041).
- Zero DATETIME2/DATETIMEOFFSET in the schema; all timestamps NVARCHAR(40)/FLOAT epoch.

**Correction 1 — STATSAPI-DOC rationale overstates the batter payoff (and the "highest-value"
framing).** The memo says batters "gain real per-game rows the same way pitchers already did … the
same upgrade that made pitcher real-line calibration possible." That is FALSE. ESPN's batter gamelog
already returns real per-game, dated, variance-preserving rows (espn_client.py:776,792 populate
`game_date/opponent/is_home/team_id/completed` per game). The pitcher StatsAPI path exists for a
different reason: ESPN's gamelog endpoint returns EMPTY for pitchers ("The gamelog endpoint doesn't
support MLB pitchers", espn_client.py:833) — pitchers had NO ESPN per-game data, batters have full
ESPN per-game data. So WS10 delivers essentially NO new modeling signal for batters (same counting
stats, different provider) on a system the accuracy roadmap shows is already data-rich and
gate-saturated. Its genuine deltas are architectural: canonical MLBAM/gamePk identity in the fact
tables, provider independence for the *current-season* refresh (past seasons are already permanent
local), and cold-cache efficiency (which the memo itself already labels "secondary"). Keep WS10 as a
NEW workstream, but DOWNGRADE the "highest-value item in the cluster" framing and add a regression-
risk flag: this is the most load-bearing data path (feeds every projection + all calibration), so a
batter source swap risks silent stat-definition / calibration drift. Priority it AFTER WS1–WS9,
gated on a concrete reproducibility/identity need, not on an expected accuracy gain.

**Correction 2 — P1's identity risk is already substantially mitigated.** The memo frames P1 as
closing a name-matching hole in the gamelog layer, but the batter lookup already resolves the ESPN
athlete_id THROUGH the SFBB name→MLBAM→ESPN cross-map (`gamelog_store.get_athlete_id` baseball
branch, lines 530-534, via `_mlb_espn_id` 475-493) before ever touching `search_athlete`. So gamelog
rows are already MLBAM-anchored at lookup — they're merely STORED under the resolved ESPN id, not the
raw MLBAM id. P1 is therefore identity-HARDENING (removes that id-space seam + gives the store a hard
gamePk) rather than fixing an open false-positive-identity bug. Also note the synthetic `game_key` is
NOT a dedup/join key in the store (reads select by athlete_id+season_bucket ordered by `id`;
replace-all retains every row incl. both doubleheader games), and GRADING already disambiguates
doubleheaders via gamePk (`resolve_player_game_stat`, mlb_starters.py:1611+) — so the doubleheader
benefit of gamePk applies to the STORE's provenance/reproducibility, not to any data-loss bug today.
current_state "absent" (for the columns) and verdict "adapt" both stand; only the value framing is
tempered. Keep coupled to WS10; additive-then-swap is the right pattern.

**P5 — agree in full.** current_state "partial" and verdict "adapt (mostly done; fold into WS10 +
WS1)" are exactly right. No net-new effort; preserve the deliberate 4h live TTL for doubleheader
game-2. The "warehouse is canonical" intent is largely delivered today plus WS1's fail-loud-on-
SQL-off.

**P8 — agree in full.** current_state "absent" and verdict "defer" both stand. ISO-8601 text sorts
lexicographically and is parsed via `datetime.fromisoformat` (gamelog_store.py:297-301); migrating
every reader/writer + SchemaParityTests is churn>value on a single-writer app. Backlog with the
stated trigger; additive shadow DATETIME2 columns only, never an in-place swap.

**Net:** No false "already-done" hiding a real gap — the cluster's central thesis (gamelog layer is
still ESPN/player-centric, synthetic-keyed, replace-all) is genuinely true. The risk here is the
opposite: OVER-investing in WS10 on the strength of a modeling-benefit claim that does not hold for
batters. Ship WS10 as identity/reproducibility hardening if/when that need is concrete, staged and
reversible per the doc's Phase 1-5 — not as an accuracy play.
