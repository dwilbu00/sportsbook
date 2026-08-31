# Workstream 8 — Staged Azure Blob Removal (implementation spec)

**Date:** 2026-08-07 · **Mode:** DESIGN ONLY (this doc is read-only research; no
code/DB/vendor changes were made). · **Repo root:**
`c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/`

Grounds the prior audit (`c:/tmp/audit-2026-reset/06-blob-coupling-removal.md`)
against the CURRENT code. **Every line number below was re-verified this pass**
and matches the audit (the files have not shifted). Legend: PROVED = read directly
this session.

---

## 0. TL;DR / decision

- Azure Blob is fully bypassed in prod (`_sql()` first at every dispatch) and is
  reached over raw `requests` + SAS URLs — **there is no `azure.storage` SDK
  import anywhere** (grep PROVED: 0 hits in `*.py`/`requirements.txt`). So there
  is no dependency to drop; `requests` stays (used by ~20 non-blob modules).
- **Do Stage 1 + Stage 2 (delete migration scripts + excise the Azure-Blob
  branches, KEEP local-disk).** This removes 100% of Azure coupling, collapses the
  3-mode dispatch (SQL / Azure-Blob / local-disk) to **2-mode (SQL / local-disk)**,
  and as a *net safety win* deletes the un-hardened blob recal-read arm
  (Finding F2). **Do NOT do Stage 3** (remove local-disk / go fully SQL-only) here
  — it breaks ~dozens of hermetic tests and local dev; it belongs with Workstream
  1, not this workstream.
- **The SQL-only invariant is DATA (authoritative store), not code-topology.**
  Post-removal the code is "SQL when configured, else local-disk"; Workstream 1
  then adds the loud prod guard on the `else` branch. See §5 for the combined end
  state that keeps WS8 and WS1 from conflicting.

---

## 1. Verified inventory of removal targets (current line numbers)

### 1a. `recalibration.py` — Azure-Blob-coupled (DELETE)
| Symbol | Lines (verified) | Action |
|---|---|---|
| `REMOTE_LOG_URL_ENV = "PREDICTION_LOG_BLOB_URL"` | 48 | delete (unused after) |
| `_prediction_log_blob_url()` | 128–141 | delete |
| `prediction_log_storage()` "Azure Blob" branch | 143–147 | collapse to SQL/Local cache |
| `_read_log_snapshot` blob arm | 216–223 (local 224–227 stays) | delete blob arm |
| `_write_log_snapshot` blob arm | 233–247 (local 248–252 stays) | delete blob arm |
| `mutate_prediction_log` `if not _prediction_log_blob_url()` guard + blob retry loop | 269, 277–287 (SQL 261–268 / local 270–276 stay) | make local unconditional; delete loop |
| `_read_ndjson_blob` blob arm | 370–387 (SQL 361–369 / local 388–392 stay) | delete blob arm |
| `_write_ndjson_blob` blob arm | 398–412 (local 413–418 stays) | delete blob arm |
| `mutate_ndjson_log` `if not _blob_url_for()` guard + blob retry loop | 439, 448–459 (SQL 433–438 / local 440–447 stay) | make local unconditional; delete loop |
| `_blob_url_for(filename)` | 469–478 | delete |
| `_read_json_blob` / `_write_json_blob` | 481–531 | delete both |
| `save_recalibration` blob-write branch | `elif to_blob and _prediction_log_blob_url():` 1940–1951 (SQL 1934–1939 / local 1926–1930 stay) | delete `elif` branch |
| `_load_recal_cached` blob arm | 2086–2113 (SQL 2036–2069 / local-only 2070–2084 stay) | delete blob arm; make local-only unconditional (drop the `if not _prediction_log_blob_url():` at 2070) |

KEEP (NOT Azure — parse local/SQL recal dicts, serve the SQL path): `_parse_recal_blob`
(1958–1979), `_read_local_recal` (1982–1993), `_blend_recal` (1996–2022), the SQL
arm + local-only arm of `_load_recal_cached`, `_NDJSON_CACHE` logic (runs on the
SQL arm 361–369), all `_local_*_lock`/local read-write helpers.

### 1b. `warehouse.py` — Azure-Blob-coupled (DELETE)
| Symbol | Lines (verified) | Action |
|---|---|---|
| `BLOB_URL_ENV = "PREDICTION_LOG_BLOB_URL"` | 39 | delete (unused after) |
| `_blob_base()` | 100–112 | delete |
| `storage_backend()` "Azure Blob" branch | 115–119 | collapse to SQL/Local warehouse |
| `_blob_url_for(name)` | 122–137 | delete |
| `_get_blob` / `_put_blob` | 140–182 | delete both |
| `_read_json` blob arm | 195–196 (local 197–204 stays) | delete `if _blob_base(): return _get_blob(name)` |
| `_write_json` blob arm | 208–210 (local 211–222 stays) | delete `if _blob_base(): return _put_blob(...)` |
| `list_snapshots` blob-missing arm | 486–488 (`if not _blob_base(): return _scan_local_snapshots` / `return []`) | make `_scan_local_snapshots(...)` unconditional; drop `return []` |

**No edit needed** (blob-ness lives entirely inside `_read_json`/`_write_json`;
these functions never reference `_blob_base` directly): `capture_event_odds`
non-SQL arm (392–422), `_update_manifest`/`flush` (425–472), `read_snapshot`
(521–524), `_extract_line` (527–601), `_scan_local_snapshots` (491–518),
`closing_line_for` non-SQL arm (652–666), `seed_from_store` non-SQL arm
(1059–1071). All become local-only transitively once `_read_json`/`_write_json`
lose their blob dispatch. KEEP them.

KEEP (already SQL-only, the offline real-line calibration read path): `load_prop_lines`,
`load_team_market_store`, `_assemble_prop_entries`, `_emit_team_lines`,
`_emit_prop_lines`, `_enumerate_lines`, `_enrich_ids`.

### 1c. One-shot migration scripts (DELETE whole files — Stage 1)
- `migrate_blob_to_sql.py` (126 lines) — only live caller of
  `recalibration._read_json_blob`, `_read_ndjson_blob`, `_prediction_log_blob_url`
  in a blob-reading context. Recorded done in MEMORY (blob→SQL migration COMPLETE).
- `migrate_warehouse_to_sql.py` (140 lines) — only live caller of
  `warehouse._read_json`/`read_snapshot`/`_blob_base` in a blob-reading context.
- **No test references either** (grep PROVED: only self-references). Deleting them
  FIRST is what makes Stage 2's helper deletions a clean excision (else the helper
  deletions break these scripts' imports).

### 1d. `app.py` (F4 stale guidance)
- Boot promotion: 23–29 (`PREDICTION_LOG_BLOB_URL` → env) — delete this `try`
  block (keep the SQL-secret promotion at 30–39).
- Three banners keyed on `== "Local cache"` that advise setting the URL: 1148–1158
  (Model Guide), 1728–1733 (My Bets), 2173–2187 (sidebar). Keep the "local storage
  is ephemeral" warning; drop the "Set PREDICTION_LOG_BLOB_URL …" sentence. (These
  never fire in prod, since prod = "Azure SQL".)

### 1e. Docs / config (F4)
- `.streamlit/secrets.toml.example:7–10` — delete the `PREDICTION_LOG_BLOB_URL`
  block.
- `README.md:378–420` — remove blob lines from both secrets snippets (380, 403)
  and rewrite 384–390 / 418–420 to describe SQL (`SQL_SERVER/DATABASE/USER/PASSWORD`)
  as the durable backend; "without SQL, the app falls back to ephemeral
  `cache/predictions/`."

---

## 2. Exact code transformations (reversible; smallest correct change)

The pattern everywhere is **delete the blob middle-arm, keep SQL-first + local-last**.

**`_read_log_snapshot`** →
```python
def _read_log_snapshot(where=None):
    if _sql():
        return _db.read_rows(_PRED_TABLE, where=where), None
    if not os.path.exists(LOG_PATH):
        return [], None
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return _parse_log_text(f.read()), None
```

**`_write_log_snapshot(rows, version=None)`** → keep the `version` kwarg (signature
stability / reversibility; already ignored on the local path); body drops the blob
arm, leaving only the atomic-swap local write.

**`mutate_prediction_log(mutator, max_retries=5, where=None)`** → keep the
`max_retries` kwarg (inert now; avoids touching call sites); body:
```python
    if _sql():
        with _lock:
            return _db.mutate(_PRED_TABLE, mutator, where=where)
    with _lock:
        with _local_log_lock():
            rows, version = _read_log_snapshot()
            result = mutator(rows)
            if result:
                _write_log_snapshot(rows, version)
            return result
```

**`_read_ndjson_blob` / `_write_ndjson_blob` / `mutate_ndjson_log`** — identical
shape: SQL arm (with `_NDJSON_CACHE`) stays; delete the `_blob_url_for`-gated arm
and the retry loop; the local arm becomes unconditional. Keep `use_cache`,
`where`, `max_retries` kwargs.

**`save_recalibration`** — delete the `elif to_blob and _prediction_log_blob_url():`
branch (1940–1951). Result: SQL refit → `_db.save_recal` only; seed (`to_blob=False`)
→ local file; local dev (`to_blob=True`, no SQL) → local file (via the unchanged
`if not (to_blob and _sql()):` at 1926). Update the docstring (drop "Legacy blob").

**`_load_recal_cached`** — delete the blob arm 2086–2113 and the `if not
_prediction_log_blob_url():` at 2070 so the mtime-keyed local-only block becomes
the function tail. **This is the F2 fix**: the deleted blob arm returned
`_parse_recal_blob(blob)` directly with NO seed-as-prior blend (2103–2108), unlike
the SQL arm which does the per-key `_blend_recal` (2052–2064). Removing it
eliminates the documented "Blob/local-dev mode unhardened" hazard.

**`warehouse._read_json` / `_write_json`** — delete the leading `if _blob_base():`
dispatch line; local file read/write becomes the whole body.

**`warehouse.list_snapshots`** — after the manifest check, replace
`if not _blob_base(): return _scan_local_snapshots(...)` / `return []` with an
unconditional `return _scan_local_snapshots(sport, game_date)`.

**`prediction_log_storage` / `storage_backend`** →
`return "Azure SQL" if _sql() else "Local cache"` (resp. `"Local warehouse/"`).

---

## 3. SQL-only invariants to establish

1. **Durable state is authoritative in Azure SQL, with no Azure-Blob copy.**
   Prediction log, wagers, bankroll_ledger, app_settings, market prediction log,
   recalibration params, odds warehouse. (MEMORY: migration COMPLETE + verified.)
2. **The recal READ path has exactly two sources:** the git-committed seed
   (`calibration/*.json` + `recalibration_*.json` via `_read_local_recal`) and the
   SQL overlay (`db_store.load_recal`), merged per-key by `_blend_recal`. No third
   Azure-Blob overlay. (Removing 2086–2113 enforces this — and removes a confounder
   for the 2026 reset: no dormant blob copy can silently re-seed stale fits.)
3. **The offline real-line calibration input is already SQL-only** and stays so:
   `warehouse.load_prop_lines`/`load_team_market_store` return `[]` when `_sql()`
   is false. Keep explicit.
4. **Code topology is "SQL else local-disk" (2-mode).** The "SQL-only" wording is
   about the DATA, not about forbidding the dev/test local backend. Forbidding
   local is Workstream 1's loud-prod-guard job (Stage 4 / §5), NOT this workstream.

---

## 4. Tests: what pins blob today, what changes

Confirmed pinning tests (all verified this pass). `sqlite://` in-memory backend is
the established port target (`db_store.configure_engine("sqlite://")`, used by
`test_db_store`, `test_wagers`, `test_bankroll`).

### 4a. MUST update (break after Stage 2)
- **`test_recalibration_durability.py::BlobRecalibrationTests`** (54–147, 7 tests) —
  all patch `_prediction_log_blob_url` + mock `requests`. Port to SQLite (mirrors
  `test_bankroll._SqlBankroll`): `configure_engine("sqlite://")` + `create_all()`.
  Per test:
  - `test_save_then_load_blob_round_trip` → SQL round-trip (empty CALIB_DIR → no
    seed → `merged == loop_cfg`, so `a==0.5` still holds).
  - `test_unvalidated_fit_is_not_applied` → `_parse_recal_blob` still drops
    `validated!=True`; assert `{}`.
  - `test_load_falls_back_to_local_baseline_on_404` → rename to "SQL-empty falls
    back to local seed": empty recal table + a written local seed file → load
    returns the seed (exercises `_load_recal_cached` SQL arm `else` at 2065–2066).
  - `test_load_degrades_to_cache_on_network_error` → patch `_db.load_recal` to
    raise; assert degrade-to-cache (SQL arm 2045–2048).
  - `test_malformed_props_degrades_to_empty_without_raising` → patch `_db.load_recal`
    to return each malformed shape; assert `{}`.
  - `test_save_overwrites_an_unreadable_blob` → **DELETE** (blob-ETag-specific; SQL
    `save_recal` is replace-per-sport, the scenario cannot occur). Document.
  - `test_seed_save_stays_local_only` → **KEEP**, drop the blob patch: assert
    `to_blob=False` writes the local file AND does not call `_db.save_recal` (spy
    on it).
- **`test_recalibration_durability.py::NdjsonReadCacheTests`** (384–443, 2 tests) —
  mock blob GET; the `_NDJSON_CACHE` TTL/invalidation they cover ALSO runs on the
  SQL arm (361–369, 437). Port to SQLite: spy `_db.read_rows` and assert (a) a
  second `use_cache=True` read within TTL does NOT call `_db.read_rows` again, and
  (b) `mutate_ndjson_log` pops the cache so the next read re-fetches. Preserves the
  cache coverage on the surviving path.
- **`test_warehouse.py::BlobStoreTests`** (128–176, 3 tests) — **DELETE** (pure
  blob container fake). SQL-warehouse coverage already exists in
  `test_db_store.py` (`storage_backend()=="Azure SQL"` at 685, closing-line SQL
  path). `LocalFallbackTests` (74–126), `SeedAndJoinTests`, `KindClassificationTests`
  STAY (local path, patch `_blob_base` → "").  ⚠ `LocalFallbackTests.setUp` (81)
  patches `warehouse._blob_base`; after deletion that attribute is gone — **drop
  the `_blob_base` patch** (SCRIPT_DIR patch + `_sql()` False already force local).
- **Shared local-path fixtures that patch the deleted `_prediction_log_blob_url`**
  → drop that patch (local is the default when `_sql()` is False):
  - `test_wagers.py::_LocalLedger` (192–194) — remove `_p2`.
  - `test_bankroll.py::_LocalBankroll` (50–51) — remove `_p2`.
  - `test_modeling.py` (1207, 1271) — remove the `_prediction_log_blob_url`
    patch.object.
  These are the WS1 coordination hotspot (§5): they only work if a SQL-off local
  fallback is still permitted in tests.

### 4b. UNCHANGED (still valid)
- `test_wagers.py::ReadStatusTests` (515–528) — patches `_read_ndjson_blob`
  itself, which SURVIVES (only its blob arm is removed). No change.
- `test_db_store.py` SQL-dispatch tests (429 `prediction_log_storage()=="Azure SQL"`,
  685 `storage_backend()=="Azure SQL"`) — still hold.

### 4c. NEW tests to add
1. Storage-string collapse: with `_sql()` False and no blob,
   `prediction_log_storage() == "Local cache"` and `warehouse.storage_backend()
   == "Local warehouse/"` (nothing currently pins these; grep PROVED 0 literal
   asserts). Guards against a future re-introduction of an "Azure Blob" branch.
2. Recal seed-blend on the SURVIVING path: SQL fit for one prop + a multi-prop
   local seed → `load_recalibration` returns the blended prop AND the untouched
   seed-only props (pins the F2-motivating per-key merge now that the blob
   all-or-nothing arm is gone).

Baseline to re-run after edits: `python -m unittest test_recalibration_durability
test_warehouse test_db_store test_wagers test_bankroll test_modeling` then the
full suite (~918 tests).

---

## 5. Coordination with Workstream 1 (SQL-off hardening) — combined end state

Both workstreams edit the same dispatch functions, so define the end state
explicitly.

**WS8 (this) responsibility:** delete all Azure-Blob code, leaving each dispatch as
the clean 2-mode shape `if _sql(): <SQL> else: <local-disk>`. WS8 does **NOT** add
any prod guard.

**WS1 responsibility (F1):** make a SQL-off write fail LOUD in a production context
instead of silently degrading to ephemeral local disk. WS1 inserts its guard at the
`else` boundary via **one shared helper** (owned by WS1), e.g.:
```python
def _require_local_ok():
    """No-op in dev/test; raise in prod so a SQL misconfig fails loud."""
    if _prod_context():            # e.g. os.environ.get("REQUIRE_SQL") truthy
        raise RuntimeError("SQL backend required but not configured")
```
called as the first statement of every `else`/local branch. **Critical constraint:**
the guard must default to a **no-op** (prod marker absent) so the hermetic suite
and local dev keep working — this is exactly what makes the §4a fixture edits
(`_LocalLedger`, `_LocalBankroll`, test_modeling dropping the blob patch) safe.

**Recommended merge order: WS8 first, then WS1.** WS8 is pure deletion → produces
the 2-mode functions; WS1 then wraps the single `else` in each. This avoids WS1
having to add its guard around a 3-mode structure that WS8 subsequently reshapes.
If WS1 must land first, its guard sits on the current `else`/local-last branch and
WS8's blob-arm deletions are non-overlapping line ranges (middle of each function)
— resolvable, but messier. **Ownership split to avoid double edits:** WS8 owns the
shared test-fixture edits (it removes the patched `_prediction_log_blob_url`
symbol); WS1 owns `_require_local_ok`/`_prod_context` and must ensure they never
fire inside those fixtures.

---

## 6. Staging / reversibility

- **Stage 1 (one commit):** delete `migrate_blob_to_sql.py` +
  `migrate_warehouse_to_sql.py`. Zero prod risk; no test refs. Precedes Stage 2 so
  helper deletions are clean.
- **Stage 2 (one commit):** excise the Azure-Blob branches in `recalibration.py` +
  `warehouse.py` (§1a/§1b/§2), update `app.py` boot + 3 banners (§1d), update the
  tests (§4), update docs (§1e). Land WS1 after (or coordinate per §5).
- **Do NOT do Stage 3** (remove local-disk) in this workstream.
- **Reversibility:** every change is git-revertible; no DB writes, no schema
  change, no data migration. The local-disk backend (dev + tests) is untouched in
  behavior; only the Azure path is removed. `requests` stays in `requirements.txt`.

## 7. Open questions (cannot resolve from code)
1. Is `PREDICTION_LOG_BLOB_URL` still in the deployed `st.secrets`? Removal is safe
   either way (SQL wins every dispatch); confirming lets §1d/§1e proceed cleanly.
2. Any out-of-repo cron still writing Blob? None found in-repo; MEMORY says Blob is
   shut down. INFER: none.
3. WS1's exact prod-context signal (`REQUIRE_SQL` env vs Streamlit-Cloud detection)
   is WS1's call; §5 only requires it default to no-op.
