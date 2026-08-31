# Workstream 1 — Harden the silent SQL-off → ephemeral-disk fallback (Finding F1)

**Date:** 2026-08-07 · **Status:** design-only, READ-ONLY audit complete · **Reversible:** yes
**Repo root:** `c:/Users/Dwilburn/Documents/Git/ODI_SCRIPTS/SPORTSBOOK_ODDS/deploy/`

---

## 0. Goal (restated)

Make a **durable write** (bets, predictions, recal params, odds warehouse) and an
**offline refit** *fail loudly* when the SQL backend is misconfigured/off **in a
production context**, instead of silently degrading to ephemeral local disk (wiped
on Streamlit Cloud restart) and no-op'ing while appearing to succeed. Must NOT
break legitimate local-dev / hermetic-test usage.

This is audit Finding **F1** (see `00-SYNTHESIS.md` step 1 / `06-blob-coupling-removal.md` §5 F1).
It is a **prerequisite for any later refit** in the reset sequence.

---

## 1. Ground truth (verified this pass; audit line numbers re-checked)

The `_sql()` dispatch predicate is duplicated (by design — self-contained modules,
no import cycle) and both audit citations are still exact:

- `recalibration.py:105-106` — `def _sql(): return _db is not None and _db.enabled()` ✓
- `warehouse.py:58-59` — same ✓ (guarded `import db_store as _db`; `_db=None` when SQLAlchemy absent)

**SQL config detection (db_store.py):**
- `_SECRET_KEYS = ("SQL_SERVER","SQL_DATABASE","SQL_USER","SQL_PASSWORD")` — `db_store.py:51`
- `_secret(name)` reads the **environment only** (never secrets.toml) — `db_store.py:516-525`.
  Docstring: this keeps SQL OFF during tests "which never set these env vars."
- `_configured()` = all four secrets present — `db_store.py:548-549`
- `enabled()` = `_OVERRIDE_URL is not None or _configured()` — `db_store.py:552-554`
  (tests set `_OVERRIDE_URL` via `configure_engine("sqlite://")` → `enabled()` True even
  with no SQL_* env; `configure_engine(None)` → `enabled()` False)

**How SQL_* reach the env (the "production context" signal source):**
- App boot promotes `st.secrets` → env via `os.environ.setdefault` for the four keys —
  `app.py:33-39`.
- CLIs (refit, forward_tracker, backfills) call `db_store.promote_secrets_from_toml()`
  which `setdefault`s the four keys from `.streamlit/secrets.toml` — `db_store.py:528-545`;
  refit invokes it in `main()` at `refit_calibration.py:2732-2736`.
- Net: **SQL_* is in the environment iff the operator configured a SQL deployment.**
  A dev/test with no SQL deployment has *zero* SQL_* env vars. This is the clean,
  no-false-positive production signal to key the guard on.

**Durable WRITE choke points (complete inventory — every durable write funnels here):**
- `recalibration.mutate_prediction_log` — `recalibration.py:255` — prediction log (callers:
  `log_prediction`/upsert :663, compact :688, grading :1664, `mark_predictions_refit` :330,
  `migrate_blob_to_sql.py:64`)
- `recalibration.mutate_ndjson_log` — `recalibration.py:421` — wagers / bankroll / app_settings /
  market prediction log (callers: `wagers.py:302,328,368,395,566,621`; `bankroll.py:178,263,328`;
  `recalibration.py:892`)
- `recalibration.save_recalibration` — `recalibration.py:1909` — recal params. **Two modes:**
  `to_blob=False` → local git-committed seed file (INTENTIONAL local write, must NOT guard);
  `to_blob=True` + `_sql()` → SQL (`_db.save_recal` :1937); `to_blob=True` + not `_sql()` →
  local file / blob (:1926,1940) = the silent-loss case to guard.
- `warehouse.capture_event_odds` — `warehouse.py:381`(SQL)/`392`(local) — odds snapshots;
  **best-effort, wrapped in `try/except: pass` at :421** (a raise inside is swallowed → guard
  there is useless; covered by the boot guard instead).
- `warehouse.seed_from_store` — `warehouse.py:1047`(SQL)/`1059`(local) — one-shot seed.

**Refit-critical READ choke points (silent-`[]` = degenerate refit, the reset's risk):**
- `warehouse.load_prop_lines` — `warehouse.py:950`; `if not _sql(): return []` at :961-962.
  Sole non-test caller: `book_line_calibration.py:282` (offline real-line harvest).
- `warehouse.load_team_market_store` — `warehouse.py:864`; `if not _sql(): return empty` at :876-877.
  Sole non-test caller: `backtest.py:759` (team-market backtest).

**Test landscape (verified — guard is inert for the whole hermetic suite):**
- Tests NEVER set SQL_* env vars (grep confirmed; `test_db_store.py:6` documents it). They use
  `configure_engine("sqlite://")` (→ `enabled()` True → `_sql()` True → guard passes) or
  `configure_engine(None)` (→ `enabled()` False, **no SQL_* env** → `require_sql()` False → guard
  passes, local path preserved).
- `test_db_store.py:834 test_store_empty_when_sql_off` explicitly asserts
  `load_team_market_store` returns `{}` under `configure_engine(None)` — the read guard MUST stay
  inert here (it does: no SQL_* env → `require_sql()` False).
- Write tests either patch the mutator (`test_prediction_log.py`, `test_modeling.py:1243`,
  `test_recalibration_durability.py:214,236,462`) or run on sqlite/blob with no SQL_* env
  (`test_recalibration_durability.py` Blob + Ndjson tests, `test_wagers`, `test_bankroll`,
  `test_db_store.py:514`). None trip the guard.

---

## 2. Production-context detection design

Add a single predicate `require_sql()` — **environment-only**, so it is False in every
hermetic test and every no-SQL dev run, and True exactly when the operator has signalled a
SQL deployment. Precedence:

```
def require_sql():                       # canonical copy in db_store.py
    flag = os.environ.get("SPORTSBOOK_REQUIRE_SQL", "").strip().lower()
    if flag in ("1","true","yes","on"):  return True      # explicit opt-in (prod)
    if flag in ("0","false","no","off"): return False     # explicit escape hatch (dev)
    return any(_secret(k) for k in _SECRET_KEYS)           # infer: ANY SQL_* present ⇒ SQL intent
```

**Why "any SQL_* present" is safe (no false positives):**
- Whenever this is True, either `enabled()` is *also* True (all four present + importable →
  no raise, normal prod) OR the config is genuinely broken (partial secrets / SQLAlchemy
  missing / Azure unreachable → raise is correct).
- A dev/test with no SQL deployment sets zero SQL_* → False → **local fallback fully preserved.**
- The explicit `SPORTSBOOK_REQUIRE_SQL` flag is the **recommended prod belt-and-suspenders**
  (set it in the deployed `st.secrets`): it forces the guard ON even before secrets are read,
  and its `0/off` value is a clean local-dev escape hatch if a dev ever has partial SQL_* locally.

**Recommendation for the operator:** add `SPORTSBOOK_REQUIRE_SQL = "1"` to the deployed
Streamlit `secrets.toml` / app settings once this ships. Even without it, the inferred
"any SQL_* present" path already fires in prod (SQL_* are configured there).

---

## 3. The guard helper

Canonical `require_sql()` lives in **db_store.py** (owns `_SECRET_KEYS`/`_secret`/`enabled`).
Each of `recalibration.py` and `warehouse.py` gets a small `_ensure_durable(op)` that raises
a clear error, with a **self-contained env fallback for the `_db is None` case** (SQLAlchemy
absent but SQL_* configured = a prod misconfig we MUST still catch — cannot call
`_db.require_sql()` when `_db` is None):

```
# module-local (recalibration.py and warehouse.py both), near _sql():
_REQUIRE_SQL_ENV = "SPORTSBOOK_REQUIRE_SQL"
_SQL_SECRET_KEYS = ("SQL_SERVER","SQL_DATABASE","SQL_USER","SQL_PASSWORD")

def _require_sql():
    if _db is not None:
        return _db.require_sql()
    flag = os.environ.get(_REQUIRE_SQL_ENV, "").strip().lower()
    if flag in ("1","true","yes","on"):  return True
    if flag in ("0","false","no","off"): return False
    return any(os.environ.get(k, "").strip() for k in _SQL_SECRET_KEYS)

def _ensure_durable(op):
    if _sql() or not _require_sql():
        return
    raise RuntimeError(
        f"Refusing to {op}: the SQL backend is not enabled but a SQL deployment is "
        f"configured (SPORTSBOOK_REQUIRE_SQL or SQL_* secrets present). Writing to "
        f"ephemeral local disk would silently lose data. Fix the SQL_* secrets / the "
        f"db_store import, or set SPORTSBOOK_REQUIRE_SQL=0 for intentional local use.")
```

`_ensure_durable` is a no-op unless (SQL is off) AND (a SQL deployment is signalled) — so it
never fires in tests or no-SQL dev, and never fires in healthy prod.

---

## 4. Guard placement (layered, each edit tiny)

### Layer A — durable-write choke points (defense-in-depth, the data-loss last line)
- `recalibration.mutate_prediction_log` (:255): first statement `_ensure_durable("write the prediction log")`.
- `recalibration.mutate_ndjson_log` (:421): first statement `_ensure_durable(f"write {filename}")`.
- `recalibration.save_recalibration` (:1909): **only when `to_blob`** — insert
  `if to_blob: _ensure_durable("save recalibration params")` before the local-write branch (:1926).
  Leaves the `to_blob=False` offline-seed → local git file path untouched (it is intentionally local).

### Layer B — refit-critical warehouse reads (turn silent-`[]` into a loud refit abort)
- `warehouse.load_prop_lines` (:950): add `_ensure_durable("read prop lines from the odds warehouse")`
  at the top, before `if not _sql(): return []`.
- `warehouse.load_team_market_store` (:864): add `_ensure_durable("read team-market lines from the odds warehouse")`
  at the top, before `if not _sql(): return empty`.
  (Both stay inert in tests → `test_db_store.py:834` and the sqlite tests remain green.)

### Layer C — fail-early entry guards (friendliest; catch before any user action)
- **refit CLI** `refit_calibration.py:main()` right after `promote_secrets_from_toml()` (:2734):
  ```
  if not args.store_label and db_store.require_sql() and not db_store.enabled():
      p.error("SQL backend not reachable but a SQL deployment is configured; "
              "aborting so the refit does not silently train/write nothing. "
              "Fix SQL_* secrets or pass --store-label for an intentional local run.")
  ```
  `--store-label` (explicit local backfill read) is exempted from the *read* abort; Layer A
  still guards its writes (`mark_predictions_refit`). Wrap in try/except around the `import
  db_store` already present at :2733.
- **app boot** `app.py` after the SQL-secret promotion (:33-39): if `db_store.require_sql()` and
  not `db_store.enabled()`, `st.error(...)` + `st.stop()` so a misconfigured prod app halts at
  boot instead of writing bets to disk that vanishes. Keyed on `require_sql()` → local dev app
  (no SQL_*) is unaffected. **Coordinate with blob-removal** (which also edits app.py boot).

`capture_event_odds` (warehouse) is deliberately **not** guarded inline (its `try/except: pass`
at :421 would swallow the raise); the boot guard (Layer C) is its safety net in prod.

Minimum viable subset if "smallest change" is preferred: **Layer A + Layer C-refit**. Layer B
adds refit-read integrity for the non-refit caller (`backtest.py:759`); Layer C-app adds the
boot-time friendliness. Recommend shipping A + B + C together (all tiny, all reversible).

---

## 5. Reversibility & behavior preservation

- Pure additive: delete the helpers + the `_ensure_durable(...)` calls + `require_sql()` to
  fully revert; also git-reversible.
- The guard is a **no-op** in: all hermetic tests, no-SQL local dev, and healthy prod. It
  changes behavior ONLY in the misconfigured-prod case it is designed to catch.
- `save_recalibration(to_blob=False)` (offline seeding to the git file) is explicitly exempt →
  the seed-write workflow and `test_recalibration_durability.py:143 test_seed_save_stays_local_only`
  are unaffected.

## 6. Leakage / risk notes
- **No leakage surface** — this is persistence hardening only; touches no prediction/label math.
- Risk: a raise inside `mutate_prediction_log` could surface in a Streamlit page render in a
  misconfigured prod. That is the intended loud failure; crash-proof callers already swallow
  (`mark_predictions_refit` :329-332, `count_pending_refit` :305). The boot guard (Layer C)
  pre-empts it in the app case.
- Risk: forgetting to add `SPORTSBOOK_REQUIRE_SQL` in prod — mitigated because "any SQL_*
  present" already infers prod (SQL_* are configured in the deployed secrets).

## 7. Overlap with the Blob-removal workstream (COORDINATE)
Both workstreams edit `recalibration.py` (near `_sql`), `warehouse.py` (near `_sql`), and
`app.py` boot. They are **orthogonal and order-independent**:
- This guard sits at the *top* of each write/read function, *above* the SQL-vs-fallback
  dispatch. Blob-removal deletes the *blob arm* of the fallback *below* it (3-mode → 2-mode).
- Whichever lands first, the other rebases cleanly: the guard doesn't care whether the
  fallback is blob-then-local or local-only.
- app.py boot: this workstream adds a `require_sql()`/`st.stop()` block; blob-removal deletes
  the `PREDICTION_LOG_BLOB_URL` promotion (:23-29). Do them in one app.py edit if landing
  together. See `06-blob-coupling-removal.md` Stage 4 (it explicitly defers F1 to this workstream).
- Net safety: after blob removal there is no dormant Azure-Blob third fallback, so this guard
  becomes the *sole* thing standing between a SQL misconfig and ephemeral-disk data loss —
  making it more important, not less.

---

## 8. Tests

**Existing tests that pin the area (must stay green — verified inert):**
`test_db_store.py` (enabled/configure_engine/storage_backend/`test_store_empty_when_sql_off`:834,
load_* assemblers), `test_recalibration_durability.py` (save_recalibration, mutate paths, ndjson
cache), `test_warehouse.py` (LocalFallbackTests + capture), `test_wagers.py`, `test_bankroll.py`,
`test_prediction_log.py`, `test_modeling.py`. All either patch the mutator, use sqlite override,
or run local with no SQL_* env → guard never fires.

**New tests — `test_sql_off_hardening.py`** (setUp/tearDown snapshot+restore `os.environ` for the
4 SQL_* keys and `SPORTSBOOK_REQUIRE_SQL`, and `configure_engine(None)` in tearDown):
1. `mutate_prediction_log` raises `RuntimeError` when a partial SQL_* env is set (e.g. only
   `SQL_SERVER`) with `configure_engine(None)`; and does NOT raise (writes local) with clean env.
2. `mutate_ndjson_log("wagers.jsonl", …)` — same pair.
3. `save_recalibration(..., to_blob=True)` raises in prod-context; `save_recalibration(..., to_blob=False)`
   does NOT raise (writes the local seed) even in prod-context.
4. Flag semantics: `SPORTSBOOK_REQUIRE_SQL=1` with **no** SQL_* → raises (opt-in);
   `SPORTSBOOK_REQUIRE_SQL=0` with partial SQL_* present → does NOT raise (escape hatch).
5. `warehouse.load_prop_lines` / `load_team_market_store` raise in prod-context; return `[]`/empty
   with clean env (pins `test_store_empty_when_sql_off` behavior stays under the guard).
6. Import-failure fallback: monkeypatch `recalibration._db = None` (and `warehouse._db = None`),
   set full SQL_* env → `_ensure_durable`/`_require_sql` still raises via the env fallback.
   Restore `_db` in finally.
7. Healthy-SQL: `configure_engine("sqlite://")` → `enabled()` True → no write/read raises even
   with `SPORTSBOOK_REQUIRE_SQL=1` (guard only fires when SQL is actually off).
8. `db_store.require_sql()` unit table: (flag on/off/absent) × (0/partial/full SQL_* env).

**Validation command (post-implementation):**
`python -m unittest test_sql_off_hardening test_db_store test_recalibration_durability test_warehouse test_wagers test_bankroll`
then the full suite (~918 tests) to confirm zero regressions.

---

## 9. Files & anchors to edit
- `db_store.py:~554` — add `require_sql()` (+ `_REQUIRE_SQL_ENV`) after `enabled()`.
- `recalibration.py:~106` — add `_require_sql()`/`_ensure_durable()`; guard `mutate_prediction_log`:255,
  `mutate_ndjson_log`:421, `save_recalibration`:1909 (to_blob-gated).
- `warehouse.py:~59` — add `_require_sql()`/`_ensure_durable()`; guard `load_prop_lines`:950,
  `load_team_market_store`:864.
- `refit_calibration.py:2734` — CLI abort after `promote_secrets_from_toml()`.
- `app.py:~39` — boot `st.stop()` guard (coordinate with blob-removal).
- `test_sql_off_hardening.py` — new.
- Docs (optional, low pri): document `SPORTSBOOK_REQUIRE_SQL` in `.streamlit/secrets.toml.example`
  (coordinate with blob-removal's doc churn).
