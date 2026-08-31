# Third-party audit eval — Cluster: SQL integrity, infra, security & observability

Scope: audit P7 (foreign keys), P8 (temporal types), P20 (operational telemetry),
P21 (dependency pinning), P22 (static analysis in CI), P23 (API-key rotation / secrets).
Method: every audit claim treated as a hypothesis, confirmed/refuted against code with
file:line evidence. Repo root = `deploy/`. Evidence gathered 2026-08-10.

Key cross-cutting fact that reshapes P7/P8: **schema.sql is hand-run DDL by the admin;
the app runs least-privilege CRUD and issues NO DDL** (schema.sql:3-6, grants.sql:12).
`db_store.py` SQLAlchemy metadata must stay in lockstep with schema.sql, and
`test_db_store.py::SchemaParityTests` enforces that — **but the parity test only compares
column NAME sets, not types or constraints** (test_db_store.py:550-591). So type/FK
changes won't be caught by the parity guard (a silent-drift risk in both directions),
and any change must be applied to BOTH schema.sql (admin DDL) and db_store metadata by hand.

---

## P7 — Strengthen SQL relational integrity (explicit FKs)

**current_state: ABSENT (but scope is far smaller than the audit implies).**

Evidence:
- schema.sql contains ZERO `FOREIGN KEY` / `REFERENCES` clauses — the only hit is a
  comment: `-- ... snapshot_id references odds_snapshot.id.` (schema.sql:335). It has PKs,
  UNIQUE, CHECK, and indexes, but no FKs.
- db_store.py never imports `ForeignKey` (import block, db_store.py:40-44) and defines no
  `ForeignKey`/`relationship`. `odds_line.snapshot_id` is `Integer, nullable=False` with
  only an index (db_store.py:373, 385; schema.sql:353) — a logical reference, not enforced.

Reality check on the audit's framing: the auditor lists FKs "among game, player,
player_game, team, odds_event, odds_snapshot, prediction, wager." **Most of those tables
do not exist.** Full table inventory (schema.sql CREATE TABLE list): prediction_log,
wagers, market_prediction_log, recalibration_params/folds/meta, odds_snapshot, odds_line,
mlb_batter_gamelog, mlb_pitcher_gamelog, nba_gamelog, nfl_gamelog, gamelog_fetch_meta,
athlete_id_cache, statcast_player_asof, player_id_map, team_id_map, id_map_meta,
bankroll_ledger, app_settings. There is **no `game` dimension, no `player` dimension, no
`odds_event` table.** Gamelog tables are keyed by bare `athlete_id` strings, predictions by
`player_key` string — no dimension tables to FK against. The broad normalization vision
overlaps audit P5 (warehouse) / P1 (gamePk-canonical), which are outside this cluster and
depend on tables that must be built first.

Concrete FK candidates that actually exist today:
1. **`odds_line.snapshot_id` -> `odds_snapshot.id`** — the audit's own example, and the one
   clean win. The write path is atomic parent-then-child in a single transaction: insert the
   snapshot, read `result.inserted_primary_key[0]`, insert lines with that id
   (db_store.py:909-926). No orphan can be produced by the app; no snapshot/line DELETE
   exists in db_store. A FK (ideally `ON DELETE CASCADE`) is satisfied naturally and adds
   real integrity value + enables clean retention deletes later.
2. `recalibration_folds(sport_key, prop_key)` -> `recalibration_params(sport_key, prop_key)`
   (composite) — secondary, low value (child table is small, app-authoritative).
3. `bankroll_ledger.wager_id` -> `wagers.wager_id` — would need to stay nullable (adjustment
   rows have NULL wager_id, schema.sql:658-659); marginal value.

**Verdict: ADAPT.** Implement candidate #1 only (odds_line -> odds_snapshot, ON DELETE
CASCADE). Optionally #2. Reject the sweeping game/player/odds_event normalization here — it
belongs to the warehouse/gamePk workstreams and has no target tables yet.

Traps / deltas:
- Add the constraint in BOTH schema.sql (guarded `IF NOT EXISTS ... ADD CONSTRAINT
  fk_odds_line_snapshot FOREIGN KEY (snapshot_id) REFERENCES dbo.odds_snapshot(id) ON DELETE
  CASCADE`) AND db_store metadata (`Column("snapshot_id", Integer, ForeignKey(...))`).
- SQLite (hermetic tests) does NOT enforce FKs unless `PRAGMA foreign_keys=ON`; the current
  StaticPool test engine won't turn it on, so tests stay green — but that also means the
  hermetic suite won't *prove* the FK. Add one explicit test that enables the pragma and
  asserts an orphan line is rejected, else the FK is untested.
- Before adding the FK in prod, check for pre-existing orphan odds_line rows (needs live DB;
  the auditor had none). Cheap: `SELECT COUNT(*) FROM odds_line l LEFT JOIN odds_snapshot s
  ON l.snapshot_id=s.id WHERE s.id IS NULL`.

Integration: **WS1 (SQL-off / integrity hardening)** — closest existing home; it's a
schema.sql + db_store-metadata DDL delta. Effort: **S**.

---

## P8 — Native SQL temporal types (DATETIME2 / DATETIMEOFFSET)

**current_state: ABSENT (as the recommended migration); confirmed text/float everywhere.**

Evidence:
- Every timestamp in schema.sql is NVARCHAR or FLOAT-epoch: `ts NVARCHAR(40)` (line 18),
  `commence_time NVARCHAR(40)`, `captured_at NVARCHAR(40)`, `resolved_at NVARCHAR(40)`,
  `game_date NVARCHAR(10)`, `fit_timestamp NVARCHAR(40)` (line 291), and epoch floats
  `last_fetched_at FLOAT`/`fetched_at FLOAT` (lines 476, 498, 585, 622), `created_at
  NVARCHAR(40)` (bankroll, line 660).
- The ONLY `DATETIME2` in the whole SQL tree is the optional audit trigger
  `wager_status_audit.changed_at DATETIME2 ... DEFAULT (SYSUTCDATETIME())` (triggers.sql:19).
- db_store metadata mirrors the text choice: `Column("ts", String(40))` (db_store.py:134)
  and the import block includes **no `DateTime` type at all** (db_store.py:40-44); grep for
  `DateTime` in db_store.py = 0 hits.

Why this is genuinely a LARGE, risky migration (not a quick win):
- Timestamps are read as strings throughout the app (string slicing, ISO comparisons,
  `game_date[:10]` style joins). A SQLAlchemy `DateTime` column returns Python `datetime`,
  not `str` — every reader/formatter would need auditing.
- Mixed encodings: some are ISO text (`ts`, `commence_time`), some are **epoch floats used
  in TTL math** (`last_fetched_at`, `fetched_at`) — those must stay FLOAT or all TTL
  arithmetic breaks. Not a uniform "make them DATETIME2" sweep.
- SchemaParityTests only checks names, so a type mismatch between schema.sql and metadata
  would pass CI silently — the migration has no automated guard.
- Reproducibility/Brier-first philosophy doesn't need it; benefits (range predicates,
  retention) only bite at warehouse scale, which the app hasn't hit.

**Verdict: DEFER the bulk migration; ADAPT with a forward-only rule.** New timestamp
columns — especially the audit-P3 provenance fields (`prediction_created_at`,
`calibration_fit_timestamp`, `data_asof`) if/when that workstream lands — should be
`DATETIME2` from birth (and add `DateTime` to the db_store import block then). Cheap,
zero-migration, gets the benefit where it's free. Do NOT retrofit existing NVARCHAR/epoch
columns now.

Integration: **backlog** — bulk migration; the forward-only DATETIME2 rule attaches to the
provenance workstream (audit P3, different cluster). Trigger for the bulk work: warehouse
date-range query pain or a retention-policy requirement. Effort: **L** (bulk) / **S**
(forward-only rule).

---

## P20 — Operational observability / structured telemetry

**current_state: ABSENT.**

Evidence:
- `import logging` appears **nowhere** in the codebase (grep empty); 0 `getLogger`/`logger.`
  calls; 0 `warnings.warn`.
- **366 bare `except ...: pass`-style handlers** silently swallow fail-open/fail-closed
  events (grep count) — exactly the suppressed errors the auditor wants surfaced.
- User-facing surfacing is thin and transient: 16 `st.warning`/`st.error`/`st.exception` in
  app.py; these vanish on rerun and aren't aggregated.
- The only "diagnostics" are **offline model-quality CLI lenses** (`--feature-diag`,
  `--roi-diag`, `--negbin-diag` in refit_calibration.py / backtest.py) — these are Brier/ROI
  research tools, NOT operational telemetry of API/cache/identity/DB events.
- One partial signal exists: `odds_client._remaining_credits` module global (odds_client.py:27)
  updated from `x-requests-remaining`/`x-requests-last` headers (lines 191-193, 263-266,
  318-320). It's ephemeral (in-memory, per-process), not persisted, not event-structured.

**Verdict: ADAPT (valuable; scope down from the audit's event-bus list).** A full 11-event
telemetry bus is over-engineered for a single-user Streamlit app. Right-sized plan:
1. Add a lightweight `logging` config + a tiny structured-event helper (e.g.
   `ops_event(kind, **fields)`), and call it at the handful of bare-except sites that matter:
   `identity_failure`, `prediction_skipped`, `model_fallback`, `database_failure`,
   `api_failure`, `cache_stale`. Keep fail-open behavior — just make it observable.
2. Maintain a per-run counters dict and render a Streamlit **Diagnostics** expander:
   games analyzed, Odds/StatsAPI calls, credits remaining (already available via
   `_remaining_credits`), unresolved players, model fallbacks, DB errors — the auditor's
   example panel, minus the event-store ceremony.
3. (Optional, later) persist one per-run row to a new `ops_run` SQL table for trend — this
   is where DATETIME2 (P8) should be used on the new column.

Fits philosophy: yes — surfaces silent failures, changes no betting math. Directly supports
the "fail loud not silent-ephemeral" intent of WS1.

Integration: propose **new workstream "WS10 Ops Telemetry"**; the `database_failure` subset
overlaps WS1 (SQL-off failing loud) and should be built there first. Effort: **M** (logging
helper + counters + expander) / **L** (persisted event log).

---

## P21 — Fully pin production dependencies

**current_state: PARTIAL.**

Evidence (requirements.txt, 5 significant lines):
- `streamlit==1.56.0` — pinned exactly, with a long comment documenting a real breakage: a
  newer transitive `starlette>=0.48` broke Streamlit Cloud because uv resolved a bad combo
  on rebuild (requirements.txt:2-9). **Direct, documented proof that unpinned transitive
  deps have already bitten this project.**
- `SQLAlchemy>=2.0,<2.1`, `pymssql>=2.3,<2.4` — bounded ranges (not exact).
- `requests` — **fully unpinned** (requirements.txt:1).
- No `requirements.in`, no hashes, no pip-compile lock; no transitive pins.
- CI does `pip install -r requirements.txt` (tests.yml:16) — so CI does NOT test a frozen,
  reproducible dependency closure identical to prod.

**Verdict: ADAPT.** With only ~4 direct deps, a full lockfile is cheap and clearly justified
by the starlette incident. Concretely: create `requirements.in` (the 4 direct deps + the
streamlit rationale comment), generate a fully-pinned+hashed `requirements.txt` via
pip-compile (`--generate-hashes`), pin `requests` in the process, and have CI install the
locked file. Verify Streamlit Cloud honors the exact lock (it resolves with uv). Keep the
streamlit==1.56.0 comment verbatim.

Fits philosophy: yes — reproducibility is a stated goal. Effort: **S**. Integration:
**backlog / deploy-hardening** (adjacent to WS8 blob-removal deploy hygiene).

---

## P22 — Add static analysis (ruff + mypy/pyright) to CI

**current_state: ABSENT (in CI).**

Evidence:
- tests.yml runs only `python -m unittest discover -v` (tests.yml:18) — no lint/type step.
- No lint/type config anywhere: `pyproject.toml`, `ruff.toml`, `setup.cfg`, `.flake8`,
  `mypy.ini`, `tox.ini` all absent (ls errored — none exist).
- Pyright/pylance is configured only for the VS Code editor via `.devcontainer/devcontainer.json`
  (`ms-python.vscode-pylance`), never in CI.

**Verdict: ADOPT ruff now; DEFER mypy/pyright to CI.** Ruff (permissive or a curated rule
set: unused imports F401, undefined names F821, shadowing, star-imports) is a low-effort,
high-value safety net across a large, dict-heavy codebase and would run in seconds. mypy/
pyright yields low signal / high noise on today's `dict`-based inter-module contracts —
its value unlocks after audit-P10 typed domain objects land, and pylance already covers
dev-time typing. Add ruff as its own fast CI step (start non-blocking / warn-only if desired,
then enforce once clean).

Fits philosophy: yes — cheap correctness net, no math change. Effort: **S** (ruff) / **M**
(mypy later). Integration: add step to `.github/workflows/tests.yml`; propose small
workstream **"CI static analysis"** or backlog.

---

## P23 — Complete API-key security cleanup (rotate any committed key)

**current_state: PARTIAL — secrets ARCHITECTURE done, the actual ROTATION is NOT.**
This is the most urgent, lowest-effort item in the cluster.

Architecture DONE (confirmed):
- `config.json` gitignored (.gitignore:14) and not tracked (`git ls-files config.json`
  empty); `config.json.example` committed with placeholder.
- `.streamlit/secrets.toml` gitignored (.gitignore:11), **never committed** (`git log --all
  -- .streamlit/secrets.toml` empty); `secrets.toml.example` committed.
- Secrets read from env / Streamlit secrets; `promote_secrets_from_toml` copies SQL_* keys
  for offline CLIs (db_store.py:528-545).

Rotation NOT done — CRITICAL, live exposure:
- The live Odds API key `86e5622c3e84bcd0a48de5e49b24eb80` **is in committed git history**:
  `git show 12de96b:config.json` prints `"odds_api_key": "86e5622c3e84bcd0a48de5e49b24eb80"`
  (also present in earlier commits e.g. 04ae57b). It was un-tracked in commit 2fad6e7, whose
  own message says: *"Stop committing the live Odds API key ... (Key remains in history --"*.
- **The exact same key value is STILL the active key today** — present in both the live local
  `config.json` and `.streamlit/secrets.toml` (`grep 86e5622c...` matches secrets.toml).
  So a credential exposed in history is still valid and in use → it has not been rotated.

**Verdict: ADOPT — rotate immediately.** Regenerate the key at the-odds-api.com (invalidating
the exposed value), then update the local `secrets.toml`/`config.json` and the Streamlit Cloud
secret. History rewrite (BFG/git-filter-repo) is OPTIONAL/secondary: once the value is rotated
it's worthless, and rewriting shared history is disruptive — not required for the security fix,
though a scrub is nice-to-have if the repo is ever made public.

Fits philosophy: pure security hygiene, no conflict. Effort: **S** (owner action, ~minutes).
Integration: **backlog / immediate owner one-off** (not a code workstream).

---

## Quick-wins vs larger efforts (summary)

Low-risk high-value, do now:
- **P23 rotate the key** (S; minutes; exposed credential still live) — top priority.
- **P22 add ruff to CI** (S) — immediate dead-import/undefined-name coverage.
- **P21 lock requirements** (S) — pin `requests`, hashed lockfile; already proven necessary.
- **P7 odds_line -> odds_snapshot FK, ON DELETE CASCADE** (S) — one safe, atomic-write-backed FK.

Scoped-down / adapt:
- **P20 telemetry** (M) — logging helper + per-run counters + Diagnostics expander; skip the
  full event bus. New WS10; DB-failure subset overlaps WS1.

Defer:
- **P8 temporal types** (L) — bulk migration is risky (strings assumed everywhere, epoch-float
  TTLs, no type parity guard); adopt DATETIME2 only for NEW columns (forward-only, S).

Traps to remember:
- SchemaParityTests checks column NAMES only — it will NOT catch a wrong FK or wrong temporal
  type; any P7/P8 change needs a purpose-built test + hand-applied parity in schema.sql AND
  db_store metadata.
- SQLite hermetic tests don't enforce FKs without `PRAGMA foreign_keys=ON`.
- Streamlit Cloud resolves deps with uv — verify it honors the exact lockfile.

---

## Verifier verdict

Independent re-check against actual code (2026-08-10, repo `deploy/`). Every citation
in this memo was spot-checked; all six findings are well-supported. No false
"already-done"/"partial" and no over-eager adopt was found. Details + the two nuances I'd add:

**P23 (rotate key) — CONFIRMED, agree=yes (top priority).** Live exposure verified three
ways: `git show 12de96b:config.json` prints `86e5622c3e84bcd0a48de5e49b24eb80`; that exact
value is BOTH the current `config.json` and `.streamlit/secrets.toml` (`grep -c` = 1); commit
2fad6e7's own body says "Key remains in history -- rotate + purge out of band; this commit
does not fix history." Architecture-done half also verified (config.json gitignored .gitignore:14
+ `git ls-files config.json` empty; secrets.toml gitignored line 11 + `git log --all` empty).
"partial" is accurate. Pure hygiene, no philosophy conflict.

**P22 (ruff to CI) — CONFIRMED absent, agree=yes.** tests.yml runs only `pip install`
(line 16) + `unittest discover` (line 18); no pyproject/ruff/setup.cfg/.flake8/mypy.ini/
tox.ini exist; pylance is devcontainer.json:16 only (editor, not CI). Nuance: on a codebase
this size a *blocking* ruff step would likely red every PR on day one over pre-existing F401s
— adopt exactly as the finder hedges (curated ruleset, warn-only first, enforce once clean).

**P21 (pin deps) — CONFIRMED partial, agree=partial.** `requests` unpinned (requirements.txt:1);
streamlit==1.56.0 pinned w/ starlette-incident comment; SQLAlchemy/pymssql bounded ranges;
no lockfile/hashes; CI installs the loose file. Gap + direction are right. The nuance the
owner should weigh: the *full `pip-compile --generate-hashes` lockfile* is the risky/larger
part — Streamlit Cloud resolves with uv, and `--require-hashes` semantics across uv are the
unverified failure mode (the very class of problem the starlette pin already worked around by
pinning the DIRECT dep). The guaranteed cheap-S win is: pin `requests` + keep the bounded
ranges; treat the hashed lock as a separate, deploy-validated step, not a free S.

**P7 (odds_line->odds_snapshot FK) — CONFIRMED absent, agree=yes.** Zero FK/REFERENCES in
schema.sql (only the comment at :335); `ForeignKey` imported/used NOWHERE in the whole Python
tree; odds_line.snapshot_id is Integer NOT NULL + index only (db_store.py:373,385). Verified
the audit's game/player/odds_event FK targets genuinely don't exist (full CREATE TABLE
inventory). Safety of candidate #1 strengthened: `capture_odds_snapshot` (db_store.py:900) is
the SOLE odds_line writer — every caller (warehouse.py:390,1056; migrate_warehouse_to_sql.py:91)
routes through it, and it's atomic parent-then-child in one `engine.begin()` txn — so the app
cannot produce an orphan. Adapt-scope (implement #1 only, reject the normalization vision) is
correct. Load-bearing preconditions the memo already lists: (a) pre-check for existing orphan
rows before the ALTER (WITH CHECK validates existing data on Azure SQL — orphans would fail
the DDL); (b) SchemaParityTests compares column NAME sets only (verified test_db_store.py:579-591)
so it won't catch the FK — needs a purpose-built `PRAGMA foreign_keys=ON` test. Value is
mostly forward-looking (retention cascades) since orphans are already impossible today, but
it's low-risk/low-effort and touches no betting math.

**P20 (telemetry) — CONFIRMED absent, agree=yes.** `import logging`/`getLogger`/`warnings.warn`
= 0 real uses (the 2 "logging" grep hits are comments). ~347-371 fail-open `except` handlers
(count approximate; the finder's 366 is in-range and the point — hundreds of silent handlers —
holds); 16 st.warning/error in app.py; `_remaining_credits` is an in-memory module global
(odds_client.py:27). Scope-down (logging helper + per-run counters + Diagnostics expander;
skip the 11-event bus) is right for a single-user app and aligns with WS1 fail-loud. Caution:
many of those bare-excepts are DELIBERATE leakage-safe fail-opens — instrument selectively at
the handful that matter (identity/db/api), do not blanket-log or you'll drown the signal.

**P8 (temporal types) — CONFIRMED, agree=yes.** All schema timestamps are NVARCHAR
(ts/commence_time/captured_at/resolved_at/game_date/fit_timestamp/created_at) or FLOAT epoch
(last_fetched_at/fetched_at, 5 cols); only DATETIME2 is triggers.sql:19; db_store imports no
DateTime type. Verified the epoch-float TTL math (gamelog_store.py:383,510:
`(_now() - meta["last_fetched_at"]) < ttl*3600`) — those MUST stay FLOAT, confirming the
bulk sweep is genuinely non-uniform/risky with no type-parity guard. Defer-bulk +
forward-only-DATETIME2-on-new-columns is the correctly conservative call.

Net: adopt P23 (now), P22 (warn-only first), P7 (with orphan pre-check + pragma test);
adapt P20 (selective); pin `requests` under P21 with the hashed-lock treated as a separately
deploy-validated step; defer P8 bulk. Nothing to reject; nothing wrongly deferred.
