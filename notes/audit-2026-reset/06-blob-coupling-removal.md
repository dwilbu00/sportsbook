# Audit — Azure Blob Removal Plan (Area 06)

Date: 2026-08-07. Scope: READ-ONLY audit. No code edited, no DB writes, no vendor
calls. Every claim below is anchored to `file:line`. "PROVED" = read directly in
code; "INFER" = reasoned from code but not runtime-observed.

Repo root for all paths:
`c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/`

---

## 0. TL;DR

- Azure Blob is **fully bypassed in production**. Every durable-store dispatch
  checks `_sql()` FIRST; with the `SQL_*` secrets configured (prod is on Azure SQL
  per MEMORY), the Blob branches are never entered. PROVED by reading each
  dispatch site (see §2). The Blob path is dead-but-present ("dormant fallback"),
  not dead-and-unreachable — it re-activates the moment SQL is unconfigured.
- The real Azure-Blob surface is small and confined to **2 runtime modules**
  (`recalibration.py`, `warehouse.py`), **2 one-shot migration scripts** (already
  run, per MEMORY), **the app boot + 3 UI banners** (`app.py`), and **2 test
  classes + 2 test suites** that pin the Blob path.
- **Two false-positive traps** (do NOT touch): (a) `calibration_loader._load_blob`
  / `app._cached_calibration_blobs` / `backtest_props._load_blob` — these read the
  **local git-committed** `calibration/*.json`; "blob" is a variable name, not
  Azure. (b) `gamelog_store.py` header "Blob->SQL Phase C" is a stale docstring;
  the module is pure SQL + local file-cache, zero Azure coupling.
- **Recommended shape after removal:** collapse the 3-mode dispatch
  (SQL / Azure-Blob / local-disk) to **2 modes (SQL / local-disk)**. Delete only
  the Azure-Blob SAS branches; KEEP the local-disk branches (they are what local
  dev and ~dozens of hermetic tests rely on). A later, optional Stage can go fully
  SQL-only, but that one DOES break local dev / tests and must be deliberate.
- **Net safety win:** removing the Blob branch of `_load_recal_cached`
  (recalibration.py:2086-2113) deletes the documented "Blob/local-dev mode
  unhardened" hazard — that branch applies a learned fit with **no seed-as-prior
  per-key merge / champion gate** (unlike the SQL branch), i.e. it can wholesale
  replace the committed seed. See Finding F2.

---

## 1. Two distinct meanings of "blob" (read this first)

A naive grep for `blob` yields 36 files, but most are NOT Azure Blob.

**(A) Azure Blob storage (the thing that is shut down)** — SAS URL + `requests`
GET/PUT with `x-ms-blob-type` headers, gated on `PREDICTION_LOG_BLOB_URL`:
- `recalibration.py` (prediction log, wagers/bankroll/settings NDJSON, market
  prediction log, recalibration params).
- `warehouse.py` (odds snapshots + manifests).
- `migrate_blob_to_sql.py`, `migrate_warehouse_to_sql.py`.
- `app.py:23-29` (promotes the URL into env), plus display banners.
- Tests: `test_recalibration_durability.py`, `test_warehouse.py`.

**(B) The word "blob" as a local-dict / local-JSON variable (KEEP — load-bearing)**:
- `calibration_loader.py` `_load_blob` / `save_calibration` (calibration_loader.py:93-102,
  and every `blob` var 115-290) — reads/writes the **local git file**
  `calibration/<sport>.json`. This IS the calibration/recal read path core.
- `app.py:829-841` `_cached_calibration_blobs`, `app.py:1114-1116,1381-1452,1561-1586`
  — local calibration JSON, cached in-session.
- `backtest_props.py:73,457-458,1007-1069` `_load_blob` — same local calibration JSON.
- `historical_odds.py:61-65`, `mlb_starters.py:130-132`, `nfl_epa.py:331-336`,
  `props.py:647` — local JSON parse; "blob" is a local var.
- `recalibration._parse_recal_blob` (recalibration.py:1958-1979) and
  `_read_local_recal` (1982-1993) — despite the name, `_parse_recal_blob` parses a
  generic recal *dict shape* and is called by the **SQL** path too
  (`db_store.load_recal` -> parsed at recalibration.py:2050) and by the local-seed
  path (1993). NOT Azure-coupled. KEEP.
- `gamelog_store.py:1` docstring only.
- `db_store.py` — the SAS mentions are docstrings (db_store.py:5,7,341,729-810);
  the module has **zero** `requests`/SAS code (grep PROVED: only 1 docstring hit).

---

## 2. Inventory of Azure-Blob-coupled runtime code (the removal targets)

For each: DEAD-IN-PROD? = never entered when SQL configured. FALLBACK-IF-REMOVED?
= what still works if the branch is deleted but local-disk kept. TESTS = pinning.

### 2a. `recalibration.py`

| Symbol | Lines | Role | Dead in prod? | If removed (keep local) | Tests |
|---|---|---|---|---|---|
| `_prediction_log_blob_url` | 128-141 | reads env/secrets.toml SAS URL | n/a (returns "" when unset) | callers fall to local/SQL | patched everywhere |
| `prediction_log_storage` | 143-147 | UI string "Azure Blob"/"Local cache"/"Azure SQL" | returns "Azure SQL" in prod | display only | test_db_store.py:429 |
| `_read_log_snapshot` blob arm | 216-227 | GET prediction log | YES (SQL arm 214-215 first) | local file read 224-227 stays | durability tests |
| `_write_log_snapshot` blob arm | 233-247 | PUT prediction log | YES (only reached on non-SQL) | local atomic swap 248-252 stays | durability tests |
| `mutate_prediction_log` blob retry loop | 277-287 | ETag RMW | YES (SQL 261-268 first) | local lock path 269-276 stays | durability tests |
| `_read_ndjson_blob` blob arm | 370-387 | GET sibling NDJSON (wagers/bankroll/settings/market log) | YES (SQL 361-369 first) | local read 388-392 stays | NdjsonReadCacheTests, test_wagers, test_bankroll |
| `_write_ndjson_blob` blob arm | 398-412 | PUT sibling NDJSON | YES | local write 413-418 stays | same |
| `mutate_ndjson_log` blob loop | 448-459 | ETag RMW sibling | YES (SQL 433-438 first) | local lock 439-447 stays | same |
| `_blob_url_for` | 469-478 | derive sibling SAS URL | dead with callers | — | — |
| `_read_json_blob` / `_write_json_blob` | 481-531 | recal-params JSON GET/PUT | YES | — | durability tests |
| `save_recalibration` blob write | 1940-1951 | persist recal to Blob | YES (SQL write 1934-1939 first) | local write 1926-1930 stays | test_seed_save_stays_local_only |
| `_load_recal_cached` blob arm | 2086-2113 | applied-recal read via Blob | YES (SQL 2036-2069 first; local-only 2070-2084) | SQL + local-seed overlay stay | durability tests (F2) |

KEEP (not Azure-coupled): `_parse_recal_blob` (1958-1979), `_read_local_recal`
(1982-1993), `_blend_recal` (1996-2022), the SQL arm and local-only arm of
`_load_recal_cached`, all local-disk write/read arms, `_NDJSON_CACHE` (also serves
the SQL path — recalibration.py:361-369).

### 2b. `warehouse.py`

| Symbol | Lines | Role | Dead in prod? | If removed (keep local) | Tests |
|---|---|---|---|---|---|
| `_blob_base` | 100-112 | reads SAS URL | n/a | — | patched in tests |
| `storage_backend` | 115-119 | UI string | returns "Azure SQL" | display only | test_db_store.py:685 |
| `_blob_url_for` | 122-137 | container-root SAS URL | dead with callers | — | test_warehouse BlobStoreTests |
| `_get_blob` / `_put_blob` | 140-182 | JSON GET/PUT | YES | — | BlobStoreTests |
| `_read_json` / `_write_json` blob arm | 195-196, 208-210 | dispatch | YES (`if _blob_base()`) | local file arm 197-204, 211-222 stays | LocalFallbackTests |
| `capture_event_odds` blob arm | 392-422 | eager snapshot PUT + accumulator | YES (SQL 381-391 first) | (local writes via `_write_json`) | LocalFallbackTests |
| `_update_manifest` / `flush` | 425-472 | manifest RMW | YES in prod (SQL has no manifest) | local manifest stays | LocalFallbackTests |
| `list_snapshots` non-SQL | 483-488 | manifest / dir scan | YES (SQL 481-482 first) | `_scan_local_snapshots` 491-518 stays | LocalFallbackTests |
| `read_snapshot` + `_extract_line` | 521-601 | Blob/local snapshot parse for CLV | YES (used only by non-SQL `closing_line_for`) | local dir path stays | LocalFallbackTests |
| `closing_line_for` non-SQL arm | 652-666 | CLV via Blob/local | YES (SQL 640-650 first) | local stays | LocalFallbackTests |
| `seed_from_store` blob arm | 1059-1071 | seed warehouse to Blob | YES (SQL 1047-1058 first) | local write stays | SeedAndJoinTests |

KEEP (SQL-only already, no Azure coupling): `load_prop_lines` (950-982),
`load_team_market_store` (864-900), `_assemble_prop_entries` (903-947),
`team_market_lines`/`player_prop_lines` (via db_store). **These are the offline
real-line calibration read path and they are ALREADY SQL-only** — they return
`[]`/empty when `_sql()` is false (warehouse.py:876-877, 961-962). Blob removal
does not touch them.

### 2c. `db_store.py`
No Azure Blob code at all (PROVED: single docstring hit). It is the SQL backend
the Blob branches were migrated to. Nothing to remove.

### 2d. One-shot migration scripts (already executed per MEMORY blob-to-sql-migration)
- `migrate_blob_to_sql.py` — reads Blob prediction log / wagers / recal, writes
  SQL. Requires `_prediction_log_blob_url()` to be set (migrate_blob_to_sql.py:91-93)
  and `_read_ndjson_blob`/`_read_json_blob` (70-77). Dead weight once Blob is gone.
- `migrate_warehouse_to_sql.py` — reads Blob warehouse via `warehouse._read_json`
  / `read_snapshot` (66-67), writes SQL. Requires `warehouse._blob_base()`
  (108-109). Same.
- Both are the ONLY reason several Blob helpers must survive until they are
  deleted (`_read_ndjson_blob`, `_read_json_blob`, `warehouse.read_snapshot`,
  `warehouse._read_json`, `_blob_base`).

### 2e. `app.py`
- app.py:23-29 promotes `PREDICTION_LOG_BLOB_URL` from `st.secrets` into env at
  boot. Harmless when the secret is absent (which MEMORY says it now is).
- Display banners keyed on the storage-backend string: app.py:1148-1158 (prediction
  log), 1728-1732 (wagers), 2174-2184 (forward tracking). These print "consider
  setting PREDICTION_LOG_BLOB_URL" ONLY when the backend is "Local cache", i.e.
  never in prod (prod = "Azure SQL"). Copy is now stale advice (points users at a
  dead service).

### 2f. Config / docs
- `.streamlit/secrets.toml.example:7-10` documents `PREDICTION_LOG_BLOB_URL` as the
  legacy store. `README.md:380-418` has full Blob setup instructions and claims
  "Without `PREDICTION_LOG_BLOB_URL`, the app falls back to `cache/predictions/`"
  and "Durability ... requires the Azure Blob". Both are now wrong (SQL is the
  durable backend).
- `.gitignore:19` comment mentions the Blob copy; harmless.
- `requirements.txt`: `requests` CANNOT be dropped — used by odds_client,
  espn_client, game_results, savant_history, weather_factors, etc. (20 files).

---

## 3. SQL-only invariants to establish

1. **Prediction log, wagers, bankroll, app_settings, market prediction log,
   recalibration params, odds warehouse** are authoritative in Azure SQL. There is
   no Azure-Blob copy anymore. (MEMORY: migration complete + verified.)
2. **`_sql()` must be true in production.** Today, if `db_store` import fails
   (SQLAlchemy/pymssql missing) or the `SQL_*` secrets are unset, `_sql()` silently
   returns False (recalibration.py:105-106, warehouse.py:58-59) and the app falls
   back to **local disk** (ephemeral on Streamlit Cloud) — silent data loss, not a
   loud failure. This is the biggest latent operational risk uncovered here
   (Finding F1). A SQL-only posture should FAIL LOUD in prod instead of degrading.
3. **The recal READ path** has exactly two sources going forward: the git-committed
   seed (`calibration/*.json` + `recalibration_*.json` via
   `calibration_loader._load_blob` / `recalibration._read_local_recal`) and the
   SQL overlay (`recalibration_params`/`_folds`/`_meta` via `db_store.load_recal`).
   The Azure-Blob overlay (`_load_recal_cached` 2086-2113) is a redundant third
   source that should be removed.
4. **The offline real-line calibration input** (`warehouse.load_prop_lines` /
   `load_team_market_store`) is ALREADY SQL-only. Keep that invariant explicit.

---

## 4. Staged removal plan (safe -> aggressive)

**Stage 0 — Verify before deleting (READ-ONLY, do first).**
- Confirm prod `st.secrets` no longer contains `PREDICTION_LOG_BLOB_URL` (MEMORY
  says Blob unconfigured; verify in the deployed secrets, which this audit cannot
  read). If it is still set, removal is still safe (SQL wins), but the banners and
  boot promotion are the only things reading it.
- Confirm the SQL warehouse + ledgers are populated (MEMORY: pred 3649 / wagers 146
  / warehouse 545+13384 / gamelogs 35765+3880). If confirmed, the Blob copies are
  no longer a needed backup.

**Stage 1 — Delete the one-shot migration scripts (zero prod risk).**
- Remove `migrate_blob_to_sql.py` and `migrate_warehouse_to_sql.py`. They are run
  ONCE and MEMORY records both as done. They import nothing that other runtime code
  needs. This immediately frees several Blob helpers from having any live caller.
- No tests reference these two scripts (grep: only self-references).

**Stage 2 — Remove the Azure-Blob branches from runtime modules, KEEP local-disk.**
Turns the 3-mode dispatch into 2-mode (SQL else local-disk). Per-symbol:
- `recalibration.py`: delete the blob arms listed in §2a; in
  `_read_log_snapshot`/`_write_log_snapshot`/`mutate_prediction_log`/
  `_read_ndjson_blob`/`_write_ndjson_blob`/`mutate_ndjson_log`, replace
  `if _sql(): ... ; if not <blob_url>: <local> ; <blob loop>` with
  `if _sql(): ... ; else: <local>`. Delete `_blob_url_for`, `_read_json_blob`,
  `_write_json_blob`, and the blob write in `save_recalibration` (1940-1951) and
  the blob arm of `_load_recal_cached` (2086-2113). Reduce
  `_prediction_log_blob_url` to a no-op or delete it and its `prediction_log_storage`
  "Azure Blob" branch (collapse to "Azure SQL" / "Local cache").
- `warehouse.py`: delete `_blob_base`, `_blob_url_for`, `_get_blob`, `_put_blob`,
  the blob arms of `_read_json`/`_write_json`, the blob arm of
  `capture_event_odds` (392-422 collapses so non-SQL just calls the local
  `_write_json`), the blob branch of `seed_from_store` (1059-1071), and the
  `storage_backend` "Azure Blob" branch. `read_snapshot`/`_extract_line`/
  `_scan_local_snapshots`/`closing_line_for` local arms STAY (local-dev CLV).
- `app.py`: drop the boot promotion (23-29) and update the 3 banners (1148-1158,
  1728-1732, 2174-2184) to stop advising `PREDICTION_LOG_BLOB_URL`.
- **Tests to update/remove** (they will fail after Stage 2):
  - `test_recalibration_durability.py`: `BlobRecalibrationTests` (7 tests, lines
    54-147) and `NdjsonReadCacheTests` (2 tests, 384-442, they mock the Blob GET).
    The cache behavior they assert also runs on the SQL path — port them to a
    SQLite backend (`configure_engine("sqlite://")`) rather than deleting the
    coverage. `test_seed_save_stays_local_only` (137-147) asserts a local-only
    write; keep but drop the Blob-URL patch.
  - `test_warehouse.py`: `BlobStoreTests` (3 tests, 128-176). `LocalFallbackTests`
    (74-127), `SeedAndJoinTests`, `KindClassificationTests` STAY (local path).
  - Confirmed current green baseline: `python -m unittest
    test_recalibration_durability test_warehouse test_db_store` => 99 tests OK
    (ran this audit).
- Update docs (§2f) and `secrets.toml.example` / `README.md`.

**Stage 3 (optional, DELIBERATE) — go fully SQL-only (remove local-disk too).**
This is the "invariant is SQL-only" end state, but it is the ONLY stage that
breaks local dev and the hermetic suite, so treat it as a separate decision:
- Replace every remaining `else: <local>` with a loud failure when `_sql()` is
  false in a prod context, OR keep SQLite-in-tests as the sole non-prod backend.
- The ~dozens of hermetic tests that use the local NDJSON fixtures
  (`test_wagers._LocalLedger`, `test_bankroll._LedgerCM`, `test_modeling`
  patching `_prediction_log_blob_url` to "") would have to move to
  `configure_engine("sqlite://")`. This is substantial; do NOT bundle it with
  Stage 2.
- Recommendation: **stop at Stage 2** unless there is a concrete reason to forbid
  the local-disk dev backend. Stage 2 already removes 100% of the Azure coupling.

**Stage 4 — Harden the silent-fallback (independent of Blob, do regardless).**
Add a prod guard so a missing SQL config fails loudly instead of writing to
ephemeral local disk (Finding F1). This is the durability guarantee the Blob
removal otherwise quietly weakens.

---

## 5. Findings (ranked)

### F1 (HIGH) — Silent degradation to ephemeral local disk when SQL is misconfigured
`_sql()` returns False on any of: `db_store` import failure, unset `SQL_*`
(recalibration.py:97-106; warehouse.py:52-59). Every store then falls to local
disk, which on Streamlit Cloud is wiped on restart. Today Blob is a (dormant)
second fallback; once Blob is removed, a SQL outage/misconfig writes bets and
predictions straight to a disk that vanishes — with no error surfaced. Evidence:
dispatch order in recalibration.py:214-227, 261-276; warehouse.py:195-204. This is
the single most important thing to fix WHILE removing Blob (Stage 4). It also bears
on the 2026 reset: a reset run performed with SQL accidentally off would silently
train/write nothing durable.

### F2 (MEDIUM) — The Blob recal-read arm lacks the seed-as-prior hardening (the documented gap)
`_load_recal_cached` SQL arm (recalibration.py:2052-2064) does the per-key
merge + shrinkage blend (`_blend_recal`) so a learned fit never wholesale-replaces
the committed seed. The **Blob arm (2103-2108) does NOT** — it returns
`_parse_recal_blob(blob)` directly, i.e. the old all-or-nothing behavior. This is
exactly the MEMORY "Blob/local-dev mode unhardened (invariant is SQL-only)"
follow-up. Removing the Blob arm ELIMINATES the hazard (net safety improvement),
which is why Blob removal is desirable, not merely cleanup. (Note: the local-only
arm 2070-2084 also just returns the seed with no blend, but in local dev the seed
file IS the fit, so there is nothing to over-write.)

### F3 (MEDIUM) — Migration scripts are dead weight and keep Blob helpers alive
`migrate_blob_to_sql.py` / `migrate_warehouse_to_sql.py` are the only non-test live
callers of `_read_json_blob`, `_read_ndjson_blob(WAGERS_FILE)`, and
`warehouse.read_snapshot`/`_read_json`/`_blob_base` in a Blob-reading context.
Deleting them first (Stage 1) makes Stage 2 a clean excision. Both are one-shot and
recorded done in MEMORY.

### F4 (LOW) — Stale user-facing guidance points at a dead service
`app.py` banners (1148-1158, 1728-1732, 2174-2184), `README.md:380-418`, and
`secrets.toml.example:7-10` still tell the user to configure
`PREDICTION_LOG_BLOB_URL` for durability. Misleading now that Blob is shut down.

### F5 (LOW / INFER) — `NdjsonReadCacheTests` coverage is Blob-shaped but the code is shared
The `_NDJSON_CACHE` TTL/invalidation logic runs on BOTH the SQL and Blob arms
(recalibration.py:361-369 SQL, 379-387 Blob). The only tests that exercise it
(test_recalibration_durability.py:384-442) mock the Blob GET. If Blob is removed
without porting these, the cache path loses its dedicated test. Recommendation:
re-point them to a SQLite backend, don't just delete.

---

## 6. Do NOT touch (false positives verified)

- `calibration_loader.py` (`_load_blob`, `save_*`, all `blob` vars) — local
  `calibration/<sport>.json`; the calibration/recal read path core.
- `app._cached_calibration_blobs` (app.py:829-841) and all `calibration_blobs`
  usage — local calibration JSON.
- `backtest_props._load_blob` (backtest_props.py:73,457,1007,1068) — same.
- `historical_odds.py:61-65`, `mlb_starters.py:130-132`, `nfl_epa.py:331-336`,
  `props.py:647` — local JSON, "blob" is a variable name.
- `recalibration._parse_recal_blob` / `_read_local_recal` / `_blend_recal` —
  parse local/SQL recal dicts; used by the SQL path. KEEP.
- `gamelog_store.py` — pure SQL + espn file-cache; header docstring only.
- `db_store.py` — the SQL backend; no Azure code.
- `warehouse.load_prop_lines` / `load_team_market_store` — already SQL-only; the
  offline calibration read path. KEEP.

---

## 7. Bearing on the 2026 calibration reset

**Neutral-to-supporting.** Blob removal does not by itself argue for or against
resetting calibration to MLB-2026-only. But it MATTERS for executing a reset
cleanly:
- After removal, the authoritative calibration state lives in exactly two places
  the reset can reason about — the git seed JSON and the SQL `recalibration_*`
  tables. There is no dormant third Azure-Blob copy that could silently re-seed
  stale pre-2026 fits/labels back into the pipeline. That REMOVES A CONFOUNDER,
  supporting a clean reset.
- F1 is a direct execution risk for the reset: any offline refit / sweep run with
  SQL accidentally off would write nothing durable (or write to ephemeral disk)
  and could appear to "reset" while actually no-oping. Harden F1 before running a
  destructive reset.
- The forward-vs-backtest Brier gap that motivates the reset is unrelated to Blob;
  nothing in the Blob code affects prediction quality (it is pure persistence).

---

## 8. Open questions / could not determine from code

1. **Is `PREDICTION_LOG_BLOB_URL` still present in the deployed `st.secrets`?**
   Cannot read prod secrets. If present, the boot promotion (app.py:23-29) still
   sets the env var, but SQL still wins every dispatch, so it is inert. Removal is
   safe either way; worth confirming so the banners can be simplified.
2. **Does the local `.streamlit/secrets.toml` contain the SAS URL?** The Blob
   helpers also read `secrets.toml` directly (recalibration.py:133-140;
   warehouse.py:105-112), so a stale local file could make `_blob_base()` truthy in
   a *local* run even with SQL off — routing local dev to a dead Blob. Not read
   here (may contain live credentials). This is another reason Stage 2 (delete the
   Blob arms) de-risks local dev.
3. **Full-suite green count.** Verified the 3 storage modules (99 tests OK). Did
   not run the whole ~918-test suite this pass (pytest absent; unittest per-module
   works). Stage 2 test churn should be validated against the full suite.
4. **Any external ops tooling / scheduler still writing to Blob?** No scheduler
   found in-repo (MEMORY CLV notes "on-demand, NOT a scheduler"). Cannot rule out
   an out-of-repo cron. INFER: none, given MEMORY says Blob is shut down.
