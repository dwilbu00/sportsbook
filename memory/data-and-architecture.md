---
name: data-and-architecture
description: THE SYSTEM: Azure SQL warehouse + MLB StatsAPI medallion (ESPN fully removed for MLB), MLBAM/game_pk identity, as-of feature stores, additive runs model, CLV. 5M-credit backfill + clean-slate relaunch + all prod DDL = DONE (2026-08-28).
metadata:
  type: project
---

## Bottom line (current architecture, 2026-08)

The app runs entirely on **Azure SQL** (Blob retired) with the **MLB data layer 100% off ESPN** (StatsAPI medallion warehouse). Identity is **MLBAM player_id + game_pk** (game-context/role-verified resolver), with SFBB cross-maps as a seed/fallback. NBA/NFL are still ESPN (correct — shared code dispatches on `SPORTS[...]["espn_sport"]`, never delete whole shared functions). Local-JSON stores are DEAD; everything reads via `db_store` / the warehouse. Facts (gamelogs, statcast) span 2023-2026; the team-market/odds backtest window is effectively 2024-2026 (2023 odds were SBR-poison-purged and are being re-backfilled in the credit-windfall). Betting reality: Doug bets **DraftKings AND FanDuel** (both executable); other books are analysis-only.

## Azure SQL warehouse (storage layer) — MIGRATION COMPLETE

- **DB:** `sportsbookstreamlitdb.database.windows.net:1433` / `sportsbook_value_finder_db`, 250 GB. App connects as least-privilege **`StreamlitApp`** (CRUD only). **Owner runs ALL DDL manually** from `sql/*.sql`; the app never issues DDL. Prod is **mssql+pymssql** (TDS wheel, no ODBC needed on Streamlit Cloud). Opt-in `SQL_DRIVER=pyodbc` on the desktop backfill box enables `fast_executemany` (~10x on UPDATE-heavy backfills; needs ODBC Driver 18 + `pip install pyodbc`).
- **Service tier (2026-09-02): Standard S1 = 20 DTUs, ALWAYS-ON (NOT serverless → no auto-pause / no resume latency).** ⚠ 20 DTUs is a LOW ceiling → concurrent-write tools THROTTLE fast: the precise-reload `--load` at `--workers 12` stalled on NBA's prop-heavy snapshots (DTU-saturated, looked like a hang); MLB's lighter team snapshots squeaked by at 12. **Keep write concurrency modest on this tier — `--workers 3-4` for prop-heavy loads.** A stall here = DTU throttle, not a hang or resume. The app login (`StreamlitApp`) is db_datawriter (SELECT/INSERT/UPDATE/DELETE) with **NO ALTER/DDL** → `TRUNCATE`/`ALTER` fail ("...or you do not have permissions"); use `DELETE` (batched) from the app, or run `TRUNCATE` as owner/admin in Azure. bet-book/DDL ops = owner in Azure portal.
- **Feature-flagged:** SQL used only when `SQL_SERVER/DATABASE/USER/PASSWORD` are set. `db_store._secret` is env-only (app.py promotes st.secrets→env at boot; CLIs call `db_store.promote_secrets_from_toml()` — required before any offline prod probe or it silently reads the empty LOCAL store). **⚠ Streamlit Cloud has NO OS env vars** — app boot promotes secrets (SQL_* + all gate flags) from `st.secrets`→os.environ; I cannot set Cloud secrets, only the owner can.
- **Schema is fully columnar** (no JSON-text columns), `sql/schema.sql` mirrors `db_store.py` SQLAlchemy metadata (`test_db_store.SchemaParityTests` guards drift). Phase A = prediction_log/wagers/recalibration_*; Phase B = odds warehouse (`odds_snapshot` UNIQUE(sport,game_date,event_id,kind,snapshot_hour) write-once + `odds_line` per-descriptor); Phase C = durable gamelog store. Cross-PROCESS concurrency is NOT guarded (in-process `_lock` only; Community Cloud single-replica) → run offline SQL-writing tools when the app is idle.
- Live counts prove active writes (SQL is source of truth). Rollback (re-add Blob SAS) is theoretical only; Blob is unconfigured. **Nothing pending on the migration.**

## MLB StatsAPI medallion — RE-ARCHITECTURE COMPLETE (ESPN fully removed for MLB)

The whole MLB data layer moved off ESPN onto MLB StatsAPI (grew out of the 2026-reset audit; superseded the old ESPN gamelog path). **Medallion, one schema, naming-convention only:** BRONZE = transient raw-JSON landing (`mlb_bronze`, purged after genuine-final; `--purge-boxscores`/`--no-bronze`); SILVER = durable dims+facts (`mlb_team`, `mlb_game` [spine: game_pk PK, home/away_team_id FKs], `mlb_player`, `mlb_team_standings` [incl. runs_scored/runs_allowed], `mlb_batter_game`, `mlb_pitcher_game`); GOLD = **Python-composed reads** (no SQL views in the codebase). Key files: `mlb_warehouse.py` (+`_parity.py`, since retired), `mlb_starters.py`, `entity_resolver.py`.

- **Game-centric ingestion:** one `/game/{gamePk}/boxscore` = games-dim row + both rosters + both teams' stat lines (concurrent fetch, `ODI_MLB_BOXSCORE_WORKERS` default 8; serial DB writes). Batter facts: AB,H,SO,BB,HBP,SF,SH,**HR,TB,RBI**; pitcher: IP,K,ER,**BB,BF,HR,HBP,GS**. Fact key = surrogate `id` PK + UNIQUE(athlete_id, game_pk); team_id NOT NULL but NOT in key; game_date/opponent/is_home are FD on game_pk → NOT stored (gold read rejoins mlb_game→mlb_team). **⚠ Ordering invariant:** readers sort `game_date DESC, game_pk DESC` (NOT by surrogate id — a rained-out low-pk game made up later would poison as-of slices; game_pk carries identity/uniqueness, NOT chronology).
- **Servable props warehouse-only:** batter_hits, batter_strikeouts (feed returns zero — see below), TB, RBI, pitcher_strikeouts/outs/earned_runs, via `_ACTUAL_STAT_SPEC` + `WAREHOUSE_ONLY_PROPS` (ESPN-excluded on all read paths). `batter_home_runs` captured but not a servable market. HR/TB/RBI columns were 99.7% NULL until the 2023-26 re-ingest; `get_player_history` skips NULL-stat games (no all-zeros corruption).
- **Self-maintaining:** hourly `recalibration.maintain_sport → mlb_warehouse.ingest_maintenance` (rolling recent window + 14-day straggler sweep) keeps schedule/facts current; `entity_resolver` gap-fills same-day-added games (schedule-only, per-process). Grading reads FROM the DB always via `mlb_warehouse.resolve_actual` (active games refresh a fresh boxscore then read; historical ≥48h freeze at 0-network — the freeze advances `fetched_at`); backlog resolves uncapped, only the recent live fallback is capped.
- **All offline backtests are warehouse-native for MLB** (P3/P3b/P3c → P4a retired the gate; the `ODI_MLB_WAREHOUSE_BACKTEST` flag NO LONGER EXISTS). `get_calib_gamelog`/`get_calib_gamelogs_bulk` reshape facts to the ESPN cached_gamelog shape; opponent keyed on canonical StatsAPI name (no spelling gaps). Market-consensus backtest was already StatsAPI-native.
- **Read-gate flags (all flipped ON + verified in prod 2026-08-13, then largely made unconditional):** `ODI_MLB_WAREHOUSE_HIST/_TEAM/_CALIB/_ENFORCE_IDENTITY`. `_CALIB` is vestigial (join is unconditionally warehouse-only). `mlb_warehouse_gate_status()` + `prediction_log.source` (warehouse|espn) prove the flip (full slate ~99.5% warehouse). Fail-closed enforcement: an MLB player the resolver can't uniquely pin → no prediction; slate-level circuit breaker fails OPEN if unpinned fraction ≥ 0.5. ~2.6% unresolved (prospects/rookies/namesakes) lose history entirely (fail-safe empty, never wrong) — monitor the rate.
- **Placeholder hygiene:** `--purge-non-franchise` removed all-star/postseason-seed teams (dim 40→30); ingest filters to real franchises. **Danny Jansen game_pk 746942** (only player officially listed for both clubs) is deduped per athlete_id in derive_*_rows.
- **batter_strikeouts:** VENDOR gap, not a code bug — The Odds API returns ZERO batter_strikeouts lines for MLB across all US books (verified via live probe; keeping the market key is free, auto-recovers). DK actively posts batter markets we don't yet request (total_bases/rbis/runs_scored/singles/doubles/walks/stolen_bases) — TB/RBI already have warehouse facts = cleanest expansion path if owner wants more.

**Remaining teardown tail (owner-gated, low priority):** delete the last dead ESPN MLB code was DONE (P4b-2, dab5372); a GOLDEN fixture harness to replace the retired `mlb_warehouse_parity.py` is the only additive cleanup. `game_results.py` is a SECOND ESPN client (NBA/NFL/NHL scores; MLB excluded).

## Identity: SFBB xmap + game-context resolver — BOTH COMPLETE

- **SFBB migration COMPLETE + verified:** hybrid `player_key = "mlb:<id>"` (MLBID resolves) else `"name:<normalized>"`, never NULL, fails open. `player_id_map`/`team_id_map` in Azure SQL. Read paths prefer canonical id, fall back to name. Prod: 0 NULL keys, ~97% `mlb:`. `normalize_name` = db_store.py (aligned with `mlb_starters._norm`). ⚠ SQLAlchemy footgun: `col==None` compiles to `IS NULL` — always gate `if pid:` in Python, spell the null arm `col.is_(None)`.
- **Commit C COMPLETE (code pushed, gate retired):** the live MLB identity STAMP is now written via the game-context/role-verified `mlb_starters.resolve_mlbam_id` (tier1 lineup/probables two-team-scoped → tier2 statsapi season-roster unique-exact → tier3 role-verified SFBB), UNCONDITIONALLY (no env flag). `entity_resolver.resolve` kept as the dict-envelope (owns game_pk derivation via `find_game_pk_by_commence` + gap-fill + the resolved/crash contract). Kills namesake drift (e.g. Luis Garcia Jr. pitcher-id-on-a-batter-prop). Shadow over 1,500 real inputs: agree 95.2%, 0 grading losses, 59 gains. **SFBB retirement is PARTIAL by design** — `team_code_for_name` is repo-wide + `mlb_id_for_name` is resolve_mlbam_id's tier-3 fallback; only `player_alias` (write-only, never read) is droppable.

## Statcast cache + as-of feature stores

- **Raw Statcast pitch cache is in Azure SQL** (`statcast_pitch` + `statcast_day` manifest) — machine-independent, self-updating (commit 58d13b2). `savant_history.py` is SQL-single-store (`ingest_day`, `load_days`, `missing_days`/`ensure_days`, `--migrate-to-sql`/`--ensure`). Verified populated 2023-26 (~800K pitches/season, ~173K xBA/season). Live-app freshness self-heals idle gaps via a durable `statcast_last_ensured` watermark in `app_settings` (widens lookback after burst-usage gaps; 5710208).
- **⚠ Refit-box gotcha (historical, now auto-handled):** a missing local Statcast day cache silently dropped method D + the xBA blend and mis-flipped batter_hits. `refit_sport_real_lines` auto-runs `ensure_days` (cap `STATCAST_GAPFILL_CAP=45`) over the obs seasons before the xBA build. **✅ Activated (DDL + seed done 2026-08-28)** via `savant_history.py --migrate-to-sql/--ensure`.
- **`pitcher_asof_daily`** (`pitcher_asof.py`) = durable per-(pitcher, as_of_date, role) as-of feature store: SP rows carry statcast (xwOBAcon/whiff/csw/barrel/hard_hit/gb + K%/BB%/BF once GS-backfilled) + warehouse (era/ip/avg_ip/k9/GS), RAW (fit in code). Team-bullpen RP rows keyed on GS==0. Lazy `get_or_fill` (no cron) + bulk `build_season`. Owner backfilled 2023-26 (~75k SP rows). Unifies fit==serve==grade + kills the per-run 3M-row statcast load.
- `statcast_asof.py` = single-cutoff live-app derived table (2026 serving-side). Key perf indexes (✅ created on prod 2026-08-28): `ix_statcast_pitch_offense (batting_team,p_throws,game_date) INCLUDE(xwoba)`, `ix_statcast_pitch_pitcher (pitcher,game_date) INCLUDE(xwoba)`, `ix_mlb_batter/pitcher_game_gamepk`, `ix_mlb_game_season_status`. **Perf lesson:** the paid Azure tier is NOT slow (full-season GROUP BY = 4.6s server-side); the ~1-file/min stalls were client-side — pymssql 60s query timeout + prewarm thread-pool concurrent connections. Fix = `mlb_starters.precompute_offense_cache([2024,2025,2026])` run ONCE (reads statcast in timeout-safe chunks, ~89s/season) before a backtest.

## Additive expected-runs model + game_pk grading (Tier-A) — LIVE, active build

- **`ODI_MLB_ADDITIVE_RUNS` + `ODI_MLB_WAREHOUSE_OFFENSE` are LIVE in prod** (owner-confirmed) — NOT inert. The additive model (`additive_runs.py` shared spine; fit==serve golden test to 1e-9) prices MLB **spreads** in production and has additive flags for **ML** (`_mlb_additive_ml_home_win`) and **totals** too. `expected_runs = [starter_rate9·(IP/9) + bullpen_rate9·((9-IP)/9)]·offense_factor·run_env`, rate9 from a fitted "xERA-lite" map (`xera_lite.py`) — dissolves the historical 3-way scale trap (xwOBAcon fit / est_woba serve / ERA grade). Offense factor is warehouse-native (`_warehouse_team_factors` from statcast_pitch, migrated off the live Savant HTTP endpoint that the backtest box couldn't reach). Bullpen from `load_rp_series` (GS-based team-RP), NOT the empty multiplicative `bullpen_xwoba` field (that field's warehouse derivation is WON'T-BUILD — additive replaced its only consumer).
- **Bake-off verdict (owner-run 2024-2026):** additive[BLEND]+team-RP beats the multiplicative incumbent on all 5 metrics (spreads Brier 0.2549→0.2456; calibSlope 0.37→0.58 = far less overconfident; additive's OOS-optimal spread shrink 0.6 == the served live shrink). The model is a VALUE/CLV edge, not an accuracy edge (BSS<0 vs close). Market-anchoring `prob_shrink` promoted (spreads 0.6, ML 0.35, totals 0.15). Blend = current + prior-season (fixes early-season starvation); pure rolling window adds nothing.
- **game_pk team-market grading (#2, code complete + reviewed):** `final_game_by_pk` + `grade_team_bet_by_game_pk` + `GRADE_PENDING` sentinel give team markets a positive identity so grading stops name+date DH over-matching; additive/byte-identical when game_pk absent. `--backfill-team-game-pk` (dry-run default) + #2b threads game_pk through the offline team backtest. **⚠ PARKED:** props odds_line game_pk backfill (`--backfill-prop-game-pk`) — a forward-stamp coverage gap (NULL falls back to player+date, not a correctness break), now retroactively fixable since the 2023-26 mlb_game re-ingest is complete.
- **Owner idea (endorsed, not built): `game_features` gold table** — materialized per-game_pk as-of feature hub (both starters' rates, bullpen, offense, park, weather), making game_pk the star-schema hub joining odds_line↔mlb_game↔game_features↔pitcher_asof_daily. Store RAW features (fit in code); build AFTER game_pk-on-odds + the feature set is finalized.

## CLV closing-line capture — COMPLETE

DK-vs-DK exact-line CLV via on-demand historical backfill (`backfill_dk_clv.py`), NOT a scheduler (Streamlit Cloud has none). Props (0906e73) + team markets (adbcdfe) both shipped; the warehouse-derived CLV path is fully retired. `dk_close_for_wager` fetches `bookmakers=['draftkings']` at `date=commence_time` (nearest at-or-before = the close); line-moved bets left honestly UNFILLED. Ledger is CLV-filled. **⚠ `--refresh` clears ALL CLV rows** (~400cr; dry-run without `--refresh` under-reports cost). Vendor facts: historical props only after 2023-05-03, team back to 2020-06; 5-min snapshots since Sept 2022; cost `10×markets×regions` (the `bookmakers=` filter does NOT reduce cost). Use Case B (corpus benchmark) + durable SQL cache table remain out of scope.

## Odds warehouse + the 5M-credit backfill (DONE 2026-08-28, owner-confirmed)

- **Table decision RESOLVED: keep ONE sport-keyed table** — `warehouse.capture_event_odds → db_store.odds_snapshot/odds_line` is already sport-agnostic (zero refactor). Ensure a `(sport,date,market)` leading index.
- **Tools:** `backfill_historical_odds.py` (T3) = multi-sport workhorse (`--sport {mlb,nfl,nba,nhl} --category {team,props} --snapshot {early,close}`, `--gap-fill` warehouse-coverage repair, `--dry-run` local cost); `topup_props_odds.py` (T1, MLB props CLOSE) + `topup_team_odds.py` (T2, MLB team CLOSE) = warehouse gap-diff. ⚠ Don't trust T2's MLB-team gap count (blind to `kind='seed'` snapshots → 4× over-reports).
- **Provenance:** `odds_snapshot.source` ('live'|'backfill_early'|'backfill_close'|'seed'|'sbr') lets backtests filter without affecting reads. `odds_provenance.py --retag`/`--prune-seed` (dry-run default, archives before delete). **Owner boundary: MLB live begins 2026-01-01; 2023-2025 = backfill corpus.** Early-time canonical = 13:00Z. **⚠ ORDERING: RETAG before GAP-FILL** (gap-fill matches on the exact source tag; untagged closes read as missing → wasted re-fetch).
- **Spend-review CLEARED:** 6 confirmed findings all fixed (the HIGH one: PROP_LABELS dropped ~70% of NFL/NBA prop markets into a black hole while claiming complete coverage — fixed + pre-flight guard). Backfill `--warehouse` asserts `storage_backend()=='Azure SQL'` before spending.
- **★ Deep re-refit (the backfill payoff):** season-aware join (grades each obs against its own season's gamelog), per-season opp_defense (fixed a future-season leak), and read-scale fixes for the ~1.5M-row prop warehouse — chunked `player_prop_lines` (SQL prop_keys filter, drop ORDER BY, `exclude_early`) + `get_calib_gamelogs_bulk` (ONE query/role/season, replacing thousands of per-(player,season) round-trips). PURE real-line path confirmed (`--real-lines` self-fits residuals on all obs; synthetic sweep + splice retired). ✅ The deep re-refit was RUN + `--promote`d (owner-confirmed 2026-08-28). Re-run recipe for future refits: `--discard → --real-lines --xstats-strength 0.5 --no-roi-tiebreak → --diff → --promote` (DK-only so `--no-roi-tiebreak`; do NOT pass `--store-label`).

## Clean-slate relaunch (DONE — executed 2026-08-28, owner-confirmed)

"So much has changed it's not even the same app." `archive_app_data.py` (built, f1f1dc0) archives prediction_log + market_prediction_log + wagers + bankroll_ledger to ONE timestamped JSON (fsync + round-trip verify), THEN id-bounded-deletes in one transaction, writes an `app_data_epoch` marker to app_settings. This is the non-destructive archive-then-epoch design (NOT the overturned wipe). **HARD LINE — PRESERVE, never touch:** calibration fits (`calibration/*.json`, `recalibration_params`), resolved facts (mlb_game/gamelogs/statcast), team-market blocks, the 2023-2025 odds corpus. ✅ Executed 2026-08-28 (owner-confirmed): bankroll ZEROED (re-enter via My Bets), thin ~3,000 pre-relaunch 2026 `live` odds pruned, clean early(13:00Z)+close corpus loaded, `app_data_epoch` marker set. (Design was archive-then-epoch, NOT the overturned wipe; run had app STOPPED + explicit confirm.)

## Backtest parquet mirror (offline speed + no-DB portability)

`warehouse_mirror.py` (shipped 2026-08-28) — a shared LOCAL parquet mirror of the
read-only historical tables that ALL backtest tools read instead of Azure when
`ODI_BACKTEST_MIRROR=1`. Ends per-tool re-pulling; a synced box backtests with ZERO
Azure round-trips (works with no DB at all — good for multi-machine). Odds stored as
the EXACT db_store reader dicts per season×book (shape parity by construction);
mlb_game/pitcher_game/batter_game raw, scoped by season_bucket + game_type stored so
`calib_gamelogs_bulk` excludes S/A/E (matches get_calib_gamelogs_bulk) while
`pitcher_game_index` keeps all types (matches _pitcher_game_index). NaN→None coercion
= byte-identical rows. Each reader returns None on a missing slice → r2_data /
scenario route to it with per-call **Azure fallback** (safe-by-default; OFF unless
the flag is set → zero live/refit impact). Data dir gitignored (code versioned,
data local).

**AUTO-BUILD + `_valid` marker (f83dce8; default-ON flip 172af30):** the mirror is ON
BY DEFAULT for backtests — set `ODI_BACKTEST_MIRROR=0` to force the live Azure path.
All four backtest tools call `warehouse_mirror.autobuild(sport, seasons)` in
`load_or_fetch`, so the manual `--sync` is OPTIONAL — the first run auto-syncs missing
files and verifies them. (Production Streamlit Cloud only gets LFS *pointer stubs* for
the mirror (see LFS note below) → `_read()` treats them as absent → Azure path, so live
serving is unaffected; the refit reads Azure directly.) Verification is expensive (queries
Azure), so a file that PASSES `--verify` is renamed `X.parquet → X_valid.parquet`;
readers prefer the `_valid` copy; a fresh sync drops any stale `_valid` (data changed
→ must re-verify); a failing verify demotes `_valid → base`. `ensure()` FAST-PATHs to
instant (no Azure) once every needed file is `_valid`, so verification happens ONCE
per file, not per run. No-DB box: leaves files as-is (readers fall back). CLIs:
`--sync`, `--verify`, `--refresh` (re-sync), per-tool `--refresh-mirror` (re-sync +
re-verify); `ODI_BACKTEST_MIRROR=0` to force Azure. Tests: test_warehouse_mirror.py (14).

**MIRROR IN GIT LFS (2026-09-03) — share across machines:** `warehouse_mirror_data/*.parquet`
(~42MB, 73 files: MLB/NBA/NFL odds + MLB fact tables) is tracked in Git LFS (`.gitattributes`)
and committed to `main` (was gitignored). `.lfsconfig` sets `fetchexclude=warehouse_mirror_data/**`
so a PLAIN clone/checkout (Streamlit Cloud, CI) pulls only ~130B pointer stubs — **zero LFS
download bandwidth** (GitHub free tier = 1GB storage + 1GB/mo bandwidth; blobs already live on
GitHub as of push of 79d9db3). `_read()` magic-byte-checks (`PAR1`) and treats pointer stubs as
absent → readers fall back to Azure automatically on a pointer-only checkout (no crash). **To
materialize the mirror on a dev machine:** `git lfs pull -I "warehouse_mirror_data/**" -X ""`
(or once: `git config lfs.fetchexclude ""` then normal `git lfs pull`). Verified via fresh local
clone: stubs by default, opt-in pull fetches real blobs. ⚠ LFS keeps EVERY version of each blob
— frequent full re-syncs accumulate against the 1GB storage cap; `git lfs prune` (local) helps.

**Two-layer backtest cache:** each tool also keeps a per-tool PICKLE of the assembled
blob (triads/indexes/prop rows) at `deploy/backtest_cache/` (project-local, gitignored;
55a4a45 moved it out of %TEMP%). Hierarchy: **pickle (fastest) → mirror (parquet) →
Azure**. ⚠ The pickle has NO TTL/staleness check — reused until its file is missing or
`--refresh` (cache key = sport+seasons [+prop_keys/kind/`v2`]); so after a mirror
refresh, `--refresh` the tool too or it serves stale data. A COLD build prints
"reading from mirror (parquet) / Azure SQL (live)" (`source_label()`); a warm pickle
just prints its path.

## Live-analysis performance

- **SHIPPED (pushed):** Phase-3 parallelization (`_analyze_one_event` in a 16-worker pool, workers return candidate lists merged in slate order → zero races, ~5-10x; ed9dbc7) + team-dim per-process cache (`_team_dim`, was ~half of all warehouse round-trips; a395ccf). Hosted-app crash fixed (config.json.example; 90c18e6). Verify picks LIVE (no unit coverage for the Streamlit handler).
- **Deferred wins** (chase only if still slow): #3 bulk-prewarm player gamelogs, #4 bulk statcast xBA, #5 team-stats result cache, #6 delete redundant fetches, #7 SQLAlchemy pool_size. #8 (calib memo) SKIPPED.

## Warehouse-integrity audit verdicts (2026-reset) — the durable conclusions

Four adversarial verifiers audited whether the forward-worse-than-backtest Brier gap is a data-integrity artifact fixable by a 2026-only reset. **NET: NO — do not reset (the reset premise is abandoned; see the 2026-reset-verdict topic).** Durable, still-valid findings:
- **No look-ahead leakage** in the as-of/backtest machinery — every primitive is strict-before-date (`savant_history`, `backtest.py` sorts newest-first then slices, `book_line_calibration`, xBA `asof_mean`). Confirmed independently.
- **No join/identity corruption** — all durable data is 2026-only (0 pre-2026 rows, 1 sport, 0 NULL player_key, ~3% name-keyed residual fails SAFE); real-line labels self-heal every refit (re-derived from box scores, stored `outcome` never read on that path); role + id gates cover every grading path; online-Platt is the only surface fed by stored outcomes and it's champion-gated.
- **The gap is genuine OOS degradation**, most plausibly (a) sweep selection-optimism / winner's curse (even the data-rich batter_hits n=3262 degrades forward, refuting "data-volume problem"), (b) 2026 ABS-season regime novelty, (c) a metric-definition asymmetry (forward = stored outcomes over ALL rows; backtest = re-derived labels over a curated subset). None fixed by a reset. **Takeaway: trust cv/forward Brier over fit_brier; treat forward Brier as the real objective.**
- One real-but-small structural defect: `combined_mult` train/serve skew — live folds matchup/lineup/weather/output_def multipliers that the offline fit omits, so params are learned matchup-free but served matchup-scaled. Magnitude ~0.0004 MAE (multiplier ≈1.0 avg); pitcher_outs/pitcher_strikeouts have ZERO gap (no matchup/park). Structural, not the root cause.

## Operator DDL / runbooks — ALL APPLIED (2026-08-28, owner-confirmed)

⭐ **I can verify prod state DIRECTLY** (read-only, no spend) by calling
`db_store.promote_secrets_from_toml()` first, then reading Azure SQL — so verify prod
myself rather than asking Doug. **Never commit `secrets.toml`.**

All prior operator actions are DONE: prod Azure DDL (statcast_pitch/statcast_day tables
+ seed; perf indexes ONLINE=ON; `odds_snapshot.source` ALTER + `v_odds_coverage`/
`v_mlb_team_coverage_vs_eligible` views; pitcher BB/BF/HR/HBP/GS ALTERs +
`mlb_pitcher_game`/`pitcher_asof --build` re-derive); the 5M-credit windfall backfill;
Commit C legacy re-stamp (`restamp_legacy_ids --apply --odds` → `--backfill-game-pk
--apply`); and the legacy team `--backfill-team-game-pk`.

Related domains: mlb-2026-reset-audit-verdict, calibration-candidate-workflow, edge-strategy-deep-dive, accuracy-improvement-roadmap.
