# 11 — MLB → StatsAPI Medallion Re-Architecture (design)

**Status:** DESIGN — for owner review. No code written. Supersedes the phased
Step-2 / WS10 plan (`10-step2-concrete-commit-plan.md`) **for the MLB data
layer**; the calibration/scoring work-streams (docs 01–09) are unaffected.

**Scope:** MLB only. NBA/NFL stay on ESPN and are explicitly out of scope
("we'll worry about NFL and NBA later"). Every shared code path is dispatched by
`SPORTS[...]["espn_sport"/"espn_league"]` (`app.py:815-834`), so baseball can be
routed to a StatsAPI backend without disturbing the NBA/NFL branches.

**Grounded in:** the 4-agent current-state inventory (workflow
`weebpbh1e`), the entity-resolution spec
(`third_party_codebase_audit/mlb_player_identification_strategy.txt`), and the
locked gamelog key decision recorded in `mlb-2026-reset-audit-verdict.md`.

---

## 0. The one-paragraph statement of intent (owner's words)

> ESPN data is no longer used for MLB. Instead we use StatsAPI, which contains
> real, natural keys for entities (players, teams, games). When we're done, none
> of the old ESPN methods run for MLB. We store the natural keys and their
> associations once, then all subsequent work uses the stored/associated keys —
> minimizing wasted response data. Player identity resolves to the MLBAM id
> immediately at the odds boundary; everything downstream is pure ids.
> Team markets are in scope. We store the API response in its natural state
> (boxscore / schedule / rosters) in a semi-temporary landing table, use it to
> maintain durable dims (team, game, player) + facts, then serve the app from
> views. Clear-and-reload anything reproducible from StatsAPI; translate in place
> only when required. No separate DB schemas.

---

## 1. North-star principles (LOCKED)

These are decided. The rest of the doc is mechanism.

1. **MLBAM id is canonical player identity.** `game_pk`, `team_id`, date, and
   role are **evidence**, not identity. (Spec §1, §13.)
2. **Resolve name→MLBAM at the odds boundary, fail-closed.** The odds feed
   gives a NAME. We resolve it to an MLBAM id *once*, immediately, using
   game-roster context; if resolution is ambiguous we make **no prediction** for
   that player. A false positive (grading the wrong player) is strictly worse
   than a false negative. (Spec §12.)
3. **Game-centric ingestion.** One `/game/{gamePk}/boxscore` fetch yields the
   games-dim row, both rosters, and both teams' full stat lines. We never
   fan out one HTTP call per player when a box already has everyone.
4. **Medallion, one schema, naming convention only** (no `bronze.`/`silver.`
   schemas): BRONZE = transient raw-response landing; SILVER = durable dims +
   facts + connective tissue; GOLD = views the app reads.
5. **Bronze is semi-temporary.** A game's raw payload is purged once its stats
   are finalized *and* the dims/facts it feeds have been updated.
6. **Gold is views by default.** Materialize only when a measured cost/benefit
   says so.
7. **Games dim is the spine.** `game_pk` PK, `home_team_id` + `away_team_id`
   FKs to the team dim. Gamelog facts reference the games dim to get home/away →
   team dim (they do **not** each re-store the team pair).
8. **Migration = clear-and-reload the reproducible, translate-in-place the
   irreplaceable.** Reproducible (StatsAPI/SFBB/Statcast) tables get rebuilt;
   irreproducible durables (learned fits, predictions, wagers, bankroll, captured
   odds) are re-keyed in place, never dropped.
9. **The gamelog natural key is `(athlete_id, game_pk)` as a named UNIQUE, on a
   surrogate-`id` PK; `team_id` is NOT NULL but NOT in the key; `season_bucket`
   is a derived indexed attribute, NOT part of the key** (`game_pk` is globally
   unique, so it fully carries identity — see §6). This is the original narrow
   trigger, now folded in.

---

## 2. Target architecture — the medallion

```
                         The Odds API (event: date + home + away + player NAME)
                                          │
                                          ▼
                         ┌────────────────────────────────┐
                         │  ENTITY RESOLVER  (§4)          │
                         │  name + game context → MLBAM id │  fail-closed
                         └────────────────────────────────┘
                                          │  pure ids from here down
   statsapi.mlb.com                       ▼
   ─────────────                ┌───────────────────┐
   /schedule  ───┐              │  GOLD  (views)    │  ← app.py / props.py / backtest
   /boxscore  ───┼──► BRONZE ──►│  SILVER (durable) │──►│ read the wide shapes
   /standings ───┤   (transient │  dims + facts +   │
   /teams     ───┘    landing)  │  connective tissue│
                                └───────────────────┘
```

### 2A. BRONZE — transient raw-response landing

One table (or a small handful keyed by response kind). Purpose: land the
StatsAPI payload **in its natural JSON state**, so ingestion is a pure
DB→DB transform (no re-fetch) and is replayable.

Proposed shape (single table, response-kind discriminator):

| column | type | note |
|---|---|---|
| `id` | INT IDENTITY PK | surrogate |
| `kind` | NVARCHAR | `'schedule' \| 'boxscore' \| 'standings' \| 'teams'` |
| `natural_ref` | NVARCHAR | `game_pk` for boxscore, `YYYY-MM-DD` for schedule, `season` for standings/teams |
| `payload` | NVARCHAR(MAX) | raw JSON |
| `fetched_at` | DATETIME2 | |
| `processed_at` | DATETIME2 NULL | set when dims/facts updated |
| — | UNIQUE(`kind`,`natural_ref`) | one live payload per natural ref |

**Lifecycle (locked):** a boxscore row is deleted once (a) its game is a
genuine-final (`_is_genuine_final`, `mlb_starters.py:1322`) **and** (b) the
silver dims + gamelog facts it produced have been written. Schedule/standings
rows are TTL'd (they change until games finalize; a day's schedule row is purged
after all its games are final). This is the "semi-temporary" table the owner
described. Bronze is **never** read by the app — only by the silver-builder.

> Why land raw at all (vs. transform-on-fetch): it makes ingestion idempotent
> and replayable (re-derive silver from bronze without re-hitting the API),
> gives a natural retry/backfill unit, and cleanly separates "fetch" from
> "shape." Cost is one NVARCHAR(MAX) row per in-flight game — trivial, and
> purged on finalize. `mlb_starters.py` already has a file-cache layer
> (`_read_cache`/`_write_cache`, `mlb_starters.py:118-141`); bronze is the
> durable-SQL equivalent and can subsume it for the game-centric path.

### 2B. SILVER — durable dims + facts + connective tissue

**Dims (natural keys from StatsAPI):**

| dim | PK | key columns | source |
|---|---|---|---|
| `mlb_team` | `team_id` (MLBAM) | name, abbreviation, league_id, division | `/teams` (`get_team_index`, `mlb_starters.py:723`) |
| `mlb_game` | `game_pk` | `game_date`, `home_team_id`→team, `away_team_id`→team, `game_number` (DH), `venue_id`, `status`, `detailed_state`, `home_score`, `away_score` | `/schedule` (`get_schedule_index`, `mlb_starters.py:1342`) + linescore |
| `mlb_player` | `player_id` (MLBAM) | full_name, name_norm, primary_position, `is_pitcher`, `bats`, `throws` | `/boxscore` rosters + `/people` |

**Facts (reference dims by id — lean; no denormalized game attributes):**

| fact | grain | key | columns | references |
|---|---|---|---|---|
| `mlb_batter_gamelog` | player × game | UNIQUE(`athlete_id`,`game_pk`) | `id` PK, `athlete_id`, `game_pk`, `team_id`, stat cols (`AB,H,SO,BB,HBP,SF,SH`), `season_bucket` (derived index) | `game_pk`→game, `team_id`→team (NOT NULL, evidence) |
| `mlb_pitcher_gamelog` | player × game | same | same shape, stats `IP,K,ER` | same |
| `mlb_team_standings` (new) | team × season × as-of | UNIQUE(`team_id`,`season`,`as_of_date`) | records/win_pct | `team_id`→team |

> **Normalization (owner catch, 2026-08-10):** `game_date`, `opponent`, and
> `is_home` are all attributes of the game — FD `game_pk → game_date`;
> `opponent` = the other team in `game_pk`; `is_home` = `team_id ==
> game.home_team_id`. So the fact does **not** store them (today's ESPN store
> denormalizes all three onto every row). The fact carries only
> `(athlete_id, game_pk, team_id, stats)`; the GOLD view rejoins
> fact→`mlb_game`→`mlb_team` to reconstruct `opponents`/`home_aways`/
> `game_dates`. Benefit: a postponement updates `game_date` in **one** dim row,
> not ~30 denormalized fact copies — single source of truth, kills the
> postponement-drift bug class at the source.

**Connective tissue (the "associations" the owner wants stored once):**

| table | purpose |
|---|---|
| `player_alias` (new, or extend `player_id_map`) | provider NAME/id → MLBAM, with `confidence`, `resolution_method`, `valid_from/to` (spec §7) |
| `player_id_map` (exists) | SFBB name↔MLBAM↔ESPN cross-map — becomes a *seed* for `player_alias`, no longer a runtime ESPN bridge |
| `team_id_map` (exists) | SFBB team code cross-map — used only to translate legacy SFBB-coded durables (§8) into MLBAM `team_id` |

> **Key id-space shift:** today the gamelog fact tables key on **ESPN**
> `athlete_id`/`team_id` (`gamelog_store.py:459-467`, `schema.sql:392-405`) and
> the SFBB bridge maps *toward* ESPN (`_mlb_espn_id`,
> `gamelog_store.py:475-493`). The silver facts key on **MLBAM** ids and the
> games dim. `athlete_id_cache` (name→ESPN cache) and the toward-ESPN bridge
> become obsolete for MLB.

### 2C. GOLD — views the app reads

The app today reads two contract shapes (see §9). Gold reproduces them **exactly**
as views over silver so consumers don't change:

- `v_mlb_player_history` — reconstructs the `player_histories[name][prop_key]`
  dict shape (`espn_client.py:990-1002` / `gamelog_store._reconstruct:264-291`):
  `values`, `opponents`, `home_aways`, `minutes`, `game_dates`,
  `plate_appearances`, `at_bats`, `team_id`, most-recent-first. Ordering column:
  `ORDER BY game_date DESC, game_pk DESC` (§6).
- `v_mlb_team_form` — reproduces the ESPN team-market inputs (`win_pct`,
  `recent_games` scored/allowed, `opponent_win_pct`) from `mlb_game` +
  `mlb_team_standings` (replaces `get_all_teams`/`get_team_schedule` +
  `compute_recent_form`/`compute_team_defense`/`annotate_opponent_strength`).
- `v_mlb_team_defense` — `{team: avg_runs_allowed}` and the leakage-safe
  as-of series (`_team_defense_lookup`, `backtest.py:2247-2285`) from `mlb_game`
  linescores.

Materialize a view only if profiling shows the view is hot enough to matter
(cost/benefit per principle 6). Default = plain view.

---

## 3. Ingestion flow (game-centric, minimize wasted response)

Per day, per game — each StatsAPI response consumed once:

```
1. /schedule?sportId=1&date=D&hydrate=probablePitcher,linescore
       → land BRONZE(schedule, D)
       → upsert mlb_game rows (game_pk, home/away team_id, DH number, status, venue)
       → upsert mlb_team_standings snapshot cadence (or from /standings, §7)

2. for each gamePk that is genuine-final and not yet processed:
       /game/{gamePk}/boxscore
       → land BRONZE(boxscore, gamePk)
       → upsert mlb_player rows for everyone in the box (roster universe, §4)
       → upsert mlb_batter_gamelog / mlb_pitcher_gamelog fact rows
              (athlete_id=MLBAM, season_bucket, game_pk, team_id, opponent, is_home,
               stat line) via db_store.reconcile surgical upsert
       → set BRONZE.processed_at, then purge finalized bronze payload

3. /standings?leagueId=103,104&season=Y   (once per refresh cadence, §7)
       → land BRONZE(standings, Y) → upsert mlb_team_standings
```

The boxscore gives **both teams' stat lines + both rosters + home/away/final in
one call** — this is the single biggest efficiency win and the net-new
capability (`statsapi_capabilities` map §A.2: zero boxscore calls exist today;
per-player `gameLog` fan-out is what's used now).

**Writer:** the gamelog facts use `db_store.reconcile` (WS15,
`db_store.py:832`) — surgical natural-key upsert, not delete+insert — so a
re-ingested/backfilled game reconciles rather than churns. This is *why* §6's
ordering fix is mandatory (reconcile breaks id-as-recency).

---

## 4. Entity resolution (the odds boundary)

Per `third_party_codebase_audit/mlb_player_identification_strategy.txt`.

**Universe:** for an odds event `(date, home, away)`, resolve to `game_pk` via
the schedule index, then the resolver universe = the two rosters from that
game's boxscore (net-new capability; today only the ≤9 confirmed batting order
or a whole-season name index exist — `statsapi_capabilities` §A.6).

**Resolution hierarchy (fail-closed):**
1. Exact prior alias hit in `player_alias` (provider name/id → MLBAM) → use it.
2. Unique exact name match within the game's two rosters → accept, write alias
   with `resolution_method='roster_exact'`, high confidence.
3. Market-type as evidence (a pitcher-only prop narrows to the two probable
   starters; a batting prop narrows to position players). (Spec: market-type is
   evidence.)
4. Fuzzy match = **candidate generation only**, never auto-accept. If it yields
   exactly one candidate above threshold *and* no other candidate is close,
   accept with lower confidence + `resolution_method='fuzzy_single'`; otherwise
   → **fail closed** (no prediction). (Spec §12.)

**Fail-closed rule:** ambiguous (two plausible players, DH date collision that
can't be resolved, name not on either roster) → emit no prediction for that
player and log the miss. False positive ≫ false negative.

**Alias table** (`player_alias`, spec §7): `(provider, provider_key)` →
`mlb_player_id`, `confidence`, `resolution_method`, `valid_from`, `valid_to`.
Seeded from `player_id_map` (SFBB), grown at runtime. Replaces the runtime use
of `search_athlete` (ESPN) and the name-first `find_player_id`/`resolve_one_prop`
matching.

**Downstream contract:** once resolved, the prediction/wager/odds rows carry the
MLBAM `player_id` and the `game_pk`; **grading enters by (game_pk, MLBAM)**, not
by name (§9.3 shows grading is name-first today — this is the change).

---

## 5. Team markets (moneyline / spread / total) — the standings gap

**Confirmed gap** (`espn_touchpoints` gap #1, `statsapi_capabilities` §A.4):
MLB `wins/losses/win_pct/record` come *only* from ESPN's `/standings` merged
into `get_all_teams` (`espn_client.py:94-141`); **no StatsAPI `/standings` or
`/teams`-records fetcher exists anywhere.** This must be built from scratch.

Build:
- `/standings?leagueId=103,104&season=Y` → `mlb_team_standings`
  (`team_id`, `season`, `as_of_date`, `wins`, `losses`, `win_pct`).
- `v_mlb_team_form` view reproduces the `home_stats`/`away_stats` shape the
  analyzers consume (`app.py:2946-2965`; `analysis.analyze_moneyline_value:195`,
  `_predict_margin:77`, totals/spreads): `season.win_pct`, `recent.win_pct`,
  `recent_games` scored/allowed, `opponent_win_pct`.
- Recent form + team defense (runs allowed) come from `mlb_game` linescores
  (`_mlb_slate_for_date` already reads `linescore.teams.{home,away}.runs`,
  `game_results.py:197-245`) — replacing `get_team_schedule` +
  `compute_recent_form` + `compute_team_defense` for MLB.

> **Partial-today caveat:** the season day-by-day linescore loop exists
> (`backtest_starters.py:76`) but *discards* per-team runs (keeps
> margin/total only, `:93-95`). We must persist per-team runs-allowed as a
> durable fact (`mlb_game` home_score/away_score already covers this once we
> store them).

MLB margin/total is *already* further shifted by statsapi `matchup_features`
(`analysis.py:126-192`) — that stays.

---

## 6. The gamelog natural-key change (LOCKED — original trigger, folded in)

**Today** (`schema.sql:392-405`): `id INT IDENTITY PK`, `athlete_id` NOT NULL
(ESPN), `season_bucket` NOT NULL, `game_key` NVARCHAR **nullable / synthetic /
non-unique**, `team_id` nullable (ESPN), plus **denormalized** `game_date`,
`opponent`, `is_home` per row. No `game_pk`. No natural-key UNIQUE.
Refresh = delete+insert per `(athlete_id, season_bucket)`
(`gamelog_store.py:458-467`). Ordering relies on surrogate `id`
(`gamelog_store.py:289`, "insertion order == ESPN order recent-first").

**Target:**
- Add `game_pk INT NOT NULL` (from the games dim); drop the denormalized
  `game_date`/`opponent`/`is_home` — derive them by joining to `mlb_game`/
  `mlb_team` (see §2B normalization note).
- `athlete_id` becomes the **MLBAM** id (NOT NULL).
- `team_id` **NOT NULL** but **NOT in the key** — it's an attribute
  functionally dependent on (athlete, game); keeping it out of the key protects
  the re-key/backfill from creating orphan rows and lets a team correction not
  fork identity.
- Named `UNIQUE(athlete_id, game_pk)` = **gamePk + MLBAM**. `game_pk` is
  **globally unique across all of MLB history** (not season-scoped), and a
  player appears in a game at most once, so `(athlete_id, game_pk)` is the
  *minimal* natural key. `season_bucket` is **derived** (functionally dependent
  on the game) → keep it as a plain indexed attribute column for cheap season
  scans, but **NOT in the UNIQUE** (a redundant column inside the constraint
  would let a duplicate `(athlete_id, game_pk)` slip in under a mis-computed
  bucket). Surrogate `id` stays PK. Matches the sibling surrogate-PK +
  named-UNIQUE pattern (`gamelog_fetch_meta`/`athlete_id_cache`/
  `statcast_player_asof`).

**Mandatory ordering fix (silent-correctness hazard):** switching the writer to
`reconcile` (surgical upsert) breaks the "id ascending == recency" invariant the
reader depends on. The public reader contract is most-recent-first with callers
slicing `[:n]` (`gamelog_store.py:407-408`); real consumers of that order are
as-of leakage slices and recent-n windows:
`espn_client.get_player_stat_history[:n]` (`:1068`),
`book_line_calibration` as-of slices (`:438/:450`),
`backtest` prior-games slices (`:2463/:2488/:2545`). **Fix:** replace
`gamelog_store.py:289`'s `.order_by(table.c.id)` with an order over the joined
game dim — `ORDER BY mlb_game.game_date DESC, game_pk DESC` (the reader/gold
view joins fact→`mlb_game`, since `game_date` no longer lives on the fact). This
is a required correctness fix, not a nicety — it degrades accuracy with **zero
failing tests** if missed.

> **Why `game_date` leads, not `game_pk` (subtle).** `game_pk` is globally
> unique and *broadly* increasing, but it is assigned at **schedule-creation
> time, not play time** — it is NOT strictly chronological by when a game was
> actually played. A game scheduled in April (low `game_pk`) but rained out and
> made up in September keeps its original low `game_pk` while `game_date` moves
> to September. Ordering by `game_pk` would sort that recent game as if it were
> old, silently poisoning an as-of leakage slice. So the leakage-relevant
> chronology is `game_date` (play date); `game_pk DESC` is only the **same-date
> tiebreaker** (deterministically orders the two games of a doubleheader).
> `game_pk` transcends `game_date` for *identity/uniqueness*, but not for
> *chronology*.

---

## 7. Data migration — clear-and-reload vs translate-in-place

Two buckets (from `durable_tables` map). Nothing irreplaceable is dropped.

### 7A. CLEAR-AND-RELOAD (reproducible; rebuild covers whatever history StatsAPI has)

| table | today | action |
|---|---|---|
| `mlb_batter_gamelog` | ESPN athlete_id + ESPN team_id | drop rows, re-ingest from boxscores → MLBAM + game_pk |
| `mlb_pitcher_gamelog` | ESPN athlete_id + ESPN team_id | same |
| `gamelog_fetch_meta` | ESPN athlete_id | rebuild on next fetch (or retire — bronze/games dim subsumes the TTL gate) |
| `athlete_id_cache` | name→ESPN | **retire for MLB** (obsolete under MLBAM identity; `player_alias` replaces it) |
| `statcast_player_asof` | MLBAM (already) | leave; already correct id-space (`statcast_asof.py --build`) |
| `player_id_map` | SFBB/MLBAM/ESPN | keep as SFBB seed for `player_alias`; refetchable (`--refresh`) |
| `team_id_map` | SFBB/ESPN codes | keep for legacy-durable translation (§7B); refetchable |
| `id_map_meta` | TTL | regenerated |
| `nba_gamelog` / `nfl_gamelog` | ESPN | **untouched** (out of scope) |

New silver tables to create: `mlb_team`, `mlb_game`, `mlb_player`,
`mlb_team_standings`, `player_alias`, and the bronze landing table.

### 7B. TRANSLATE-IN-PLACE (irreproducible — re-key, never drop)

| table | irreplaceable because | what to reconcile |
|---|---|---|
| `prediction_log` | calibration training corpus + graded outcomes | already has `player_mlb_id` + `player_key`=`mlb:<id>` (`db_store.py:109-121`); **add `game_pk`** via retro-match (§7C) |
| `wagers` | real bets, ROI, CLV | has `player_mlb_id` + home/away/date; **add `game_pk`** (directly retro-matchable) |
| `market_prediction_log` | team-market forward tracking | home/away/date present; **add `game_pk`** |
| `odds_snapshot` / `odds_line` | captured closing lines (CLV/backtest, only partly re-backfillable) | home/away/date (snapshot) + `player_mlb_id` (line); add `game_pk` on snapshot |
| `recalibration_params`/`_folds`/`_meta` | learned Platt fits | **none** — prop-keyed, no entity refs; carry verbatim |
| `calibration/baseball_mlb.json` blocks | learned coefficients (`prob_shrink`, `starter_adjustment`, `expected_runs_challenger`, `lineup_adjustment`) | **none** — market/prop/slot-keyed, id-neutral; carry verbatim |
| `bankroll_ledger` | user financial ledger | keyed by `wager_id`; self-heals from wagers |
| `app_settings` | user settings | trivial carry |

> **Critical grounding:** `prediction_log` already stores `mlb:<id>` player_key
> and `player_mlb_id` (MLBAM) — the identity half of the migration is *already
> done* for the corpus. What's missing everywhere is a `game_pk` column
> (confirmed absent in `db_store.py`). Adding `game_pk` (nullable, backfilled) is
> the in-place translation; it does not disturb existing calibration reads.

### 7C. Historical `game_pk` backfill (retro-match — owner was right, it's recoverable)

Once `mlb_game` is fully backfilled (all seasons StatsAPI serves), retro-derive
`game_pk` for the durable rows:

- **Directly matchable** (`wagers`, `market_prediction_log`, `odds_snapshot`):
  they carry `home_team` + `away_team` + `game_date` → join to `mlb_game` on
  `(game_date, home_team_id, away_team_id)`. Team names → `team_id` via
  `team_id_map` / `_match_team_id` (`mlb_starters.py:743`, tolerant).
- **Indirectly matchable** (`prediction_log`, `odds_line`): they do **not** carry
  home/away (prediction_log stores only the player's *own* team). Reach `game_pk`
  by joining on `event_id` to a sibling that does carry home/away
  (odds_snapshot/wagers), or resolve player→game via roster+date.
- **Doubleheaders — disambiguate first, NULL only as last resort (owner,
  §11.6).** A `(date, home, away)` pair can map to two `game_pk`s (game_number
  1 & 2). Once the games dim is fully populated we usually **can** pick the right
  one from evidence now available: for **player-prop** rows, the game the player
  actually appeared in — via the gamelog facts / the existing nearest-`commence`
  gamePk locator in `resolve_player_game_stat` (`mlb_starters.py:1658-1691`); for
  **team-market** rows, `commence_time` → the specific game's `gameDate`
  timestamp. Keep that `game_pk` and backfill any required associated details.
  **Only if genuinely unresolvable** (traditional single-admission DH with
  identical timestamps and no player anchor) → leave `game_pk` NULL — harmless,
  those rows still grade by the existing name+date path; game_pk is additive.

---

## 8. ESPN teardown for MLB (what stops running)

From `espn_touchpoints`. Route baseball off these; NBA/NFL branches untouched.

| ESPN function | MLB replacement |
|---|---|
| `get_all_teams` (teams + `/standings`) | `mlb_team` dim + `mlb_team_standings` (§5,§7) |
| `get_team_schedule` | `mlb_game` dim (schedule+linescore already wired for scores) |
| `search_athlete` (name→ESPN id) | entity resolver → MLBAM (§4) |
| `get_athlete_gamelog` (**MLB batter source**) | boxscore ingestion → `mlb_batter_gamelog` |
| `get_pitcher_stats` (ESPN splits synth) | already a fallback; remove once StatsAPI batter+pitcher logs authoritative |
| `get_player_stat_history` (main prop entry) | route MLB to `v_mlb_player_history` gold view |
| `compute_recent_form`/`compute_team_defense`/`build_team_defense_lookup`/`annotate_opponent_strength`/`find_team` | recompute from `mlb_game`/`mlb_team_standings` (gold views) |
| `_mlb_espn_id` SFBB→ESPN bridge (`gamelog_store.py:475-493`) | delete — identity is MLBAM |

**No MLB action needed:** `get_team_pace_factor` (NBA/NHL only),
`list_season_athletes` (hardcoded NBA), `get_team_record_and_stats` (dead code).
**Pure helpers to keep/relocate:** `ip_to_outs`/`outs_to_ip`/`PROP_STAT_MAP`
(MLB math/constants, no network).

**Two call surfaces** to migrate together: `app.py` and the `main.py` CLI mirror
(`main.py:296,378,414,419,424`).

---

## 9. Consumer parity — the shapes gold MUST reproduce

The migration is invisible to consumers **iff** the gold views serve these exact
shapes (from `consumers_gold_grading`). This is the acceptance contract.

1. **`props.analyze_player_props_value`** (`props.py:862`, called
   `app.py:3013`): needs `player_histories[name][prop_key]` with `values`,
   `opponents`, `home_aways`, `minutes`, `game_dates`, `plate_appearances`,
   `at_bats`, `team_id`, `batting_order`, `lineup_status`, most-recent-first.
   Plus `team_defense` `{team: avg_allowed}`, `team_schedules` `{team_id: [...]}`,
   `id_to_name` reverse map for is_home resolution (`props.py:944-947,1079-1086`).
   **NB:** `team_id` here is consumed for venue/park resolution — the reverse-map
   and pitcher-team fallback (`backtest.py:2429-2438`) must key on the **new
   MLBAM team dim**, not ESPN.
2. **`backtest` / `book_line_calibration`**: gamelog rows with stat labels +
   `opponent`/`is_home`/`team_id`/`game_date`/`completed`, newest-first, so the
   as-of slices `gamelog[i+1:]` stay strictly older (`backtest.py:2488`,
   `book_line_calibration.py:450`). §6's ordering fix guarantees this.
   Column names must stay `AB,H,SO,BB,HBP,SF,SH` / `IP,K,ER` +
   `opponent,is_home,team_id,game_date,completed` (`gamelog_store.py:89-90,99`)
   or every consumer breaks.
3. **Grading/settlement**: today both predictions
   (`recalibration.resolve_one_prop`) and wagers (`_grade_wager`) enter **by
   player NAME**, then re-derive gamePk at grade time
   (`mlb_starters.resolve_player_game_stat:1611`). Target: enter by
   **(game_pk, MLBAM)** stored on the row — the hard-ID path already exists and
   does doubleheader disambiguation; we're moving it from grade-time-rederived to
   ingest-time-stored. Role gate (`_role_matches_gamelog`) stays.
4. **Team markets** (`analysis.py`): `season.win_pct`, `recent.win_pct`,
   `recent_games`, `opponent_win_pct`, per-game margins — from `v_mlb_team_form`.

---

## 10. Net-new StatsAPI capabilities to build (from `statsapi_capabilities`)

Already exist (reuse): schedule-by-date + gamePk (`get_schedule_index`),
`/teams` dim (`get_team_index`), name→MLBAM resolver
(`find_player_id`, fail-closed), per-date linescore/runs (`_mlb_slate_for_date`),
gamePk hard-ID grading + DH disambiguation (`resolve_player_game_stat`),
pitcher gamelog (`get_pitcher_gamelog`).

**Net-new:**
1. **`/game/{gamePk}/boxscore`** ingestion — the whole "one fetch → both stat
   lines + rosters + home/away/final" pattern. Biggest gap. Gives batter
   gamelogs (currently ESPN-only) *and* the roster resolver universe.
2. **`/standings`** — MLB records/win% for team markets (§5).
3. **roster-for-game** as the resolver universe (from the same boxscore).
4. **Durable per-team runs-allowed** persisted from linescores (today discarded).

---

## 11. Open owner decisions (before implementation)

**✅ ALL 7 RESOLVED (owner, 2026-08-10):** (1) backfill depth — `mlb_game` for
ALL seasons in any durable table; gamelog facts only for calibration-trained
seasons. (2) bronze — single polymorphic table. (3) `gamelog_fetch_meta` —
retire for MLB. (4) resolver store — new `player_alias` table; SFBB
`player_id_map` stays a pure seed. (5) gold views — plain, not materialized.
(6) DH retro-match — **REFINED, see §7C:** don't default to NULL; after the games
dim is populated, disambiguate the correct `game_pk` from now-available evidence
(player's actual game via gamelog facts / `resolve_player_game_stat`;
`commence_time`→game start), keep that `game_pk` + backfill associated details;
NULL only if genuinely unresolvable. (7) cutover — **push forward, go live
mid-season** (the P1 dual-run parity window still runs). **The app is now
officially ALPHA:** single user (owner), in-development, no external users → OK to
ship behavior changes live and iterate fast; the data-integrity /
never-destroy-irreplaceable constraint (§13) still holds fully.

_Original options + rationale retained below for the record._

1. **Backfill depth.** StatsAPI serves many seasons of boxscores. Clear-and-reload
   "covers whatever is available" (owner) — but how far back do we *ingest*?
   (Cheap: bounded by warehouse/prediction history that needs game_pk; full:
   everything StatsAPI has.) Recommend: backfill `mlb_game` for **all seasons
   present in any durable table** (so every legacy row can retro-match), and
   gamelog facts for the seasons the calibration corpus actually trains on.
2. **Bronze granularity.** One polymorphic table (kind + natural_ref + JSON) vs.
   per-kind tables. Recommend the single polymorphic table (simpler lifecycle,
   one purge path).
3. **`gamelog_fetch_meta` fate.** Retire (games dim + bronze subsume the TTL
   gate) or keep re-keyed to MLBAM? Recommend retire for MLB.
4. **`player_alias` vs. extend `player_id_map`.** New table (clean provider/
   confidence/validity per spec §7) vs. columns on the existing SFBB map.
   Recommend new `player_alias` (SFBB map stays a pure cross-map seed).
5. **Materialize any gold view?** Default no. Revisit if `v_mlb_player_history`
   profiles hot.
6. **NULL game_pk tolerance** on legacy durables where DH can't be resolved —
   confirm "leave NULL, keep grading by name+date" is acceptable (recommend yes).
7. **Sequencing vs. season.** Do we cut over mid-season (dual-run ESPN+StatsAPI
   for a validation window) or at an off-season boundary? Recommend a dual-run
   parity window regardless (see §12).

---

## 12. Suggested phasing (NOT a coding plan — review gate first)

Each phase is independently shippable and leaves the app working. **No code
until the owner approves this doc.**

- **P1 — Silver dims + bronze + ingestion (read-only, dual-run).** Create bronze,
  `mlb_team`, `mlb_game`, `mlb_player`, `mlb_team_standings`, `player_alias`.
  Build boxscore + standings ingestion. Populate in parallel with ESPN still
  live. Nothing consumes it yet. **Parity harness:** diff StatsAPI-derived
  gamelogs/team-form against the ESPN path for a live window.
- **P2 — Gamelog re-key + reconcile writer + ordering fix (§6).** Add `game_pk`,
  move to MLBAM ids, swap to `db_store.reconcile`, fix `.order_by`. This is the
  original narrow trigger; ship it behind the parity harness.
- **P3 — Entity resolver at the odds boundary (§4).** name→MLBAM fail-closed,
  `player_alias`. Store MLBAM + game_pk on new prediction/wager/odds rows.
- **P4 — Gold views + consumer cutover.** Route MLB `get_player_stat_history`,
  team-form, team-defense, and grading to the gold views / (game_pk, MLBAM)
  path. Retire ESPN for MLB.
- **P5 — Translate-in-place backfill (§7B/§7C).** Add `game_pk` to durables,
  retro-match, DH fail-closed. Verbatim-carry the learned fits.
- **P6 — ESPN teardown for MLB (§8).** Delete the dead MLB ESPN paths; keep the
  NBA/NFL branches.

---

## 13. Non-negotiables carried from prior work (do not regress)

- **Never delete irreplaceable data/JSON.** The 2021-24 team-market fits and
  method E/D params in `calibration/baseball_mlb.json` are irreproducible
  (`mlb-2026-reset-audit-verdict.md`). Translate-in-place = re-key, never drop.
- **Fail-closed identity** beats fail-open grading (spec §12).
- **Leakage-safe as-of ordering** must survive the reconcile switch (§6).
- **DK-only** for recommendations/prices (analysis may read other books).
- **Commit workflow:** commit proactively to `main`; never push (owner pushes).
- **Hand-run DDL** with idempotent guarded ALTERs (Azure SQL prod + SQLite
  tests); new tables + `game_pk` ALTERs follow the existing `schema.sql` pattern.

---

## 14. Still-open owner actions (unrelated, tracked elsewhere)

- 🔴 Rotate the leaked Odds API key (in git history).
- ⚠ Run the WS1c `fk_odds_line_snapshot` ALTER on Azure SQL.

These predate this doc; listed so they aren't lost.
