# Third-party audit eval — Cluster: Provenance, experiment registry & training/serving separation

Priorities evaluated: **P3** (immutable per-prediction model provenance), **P6** (uniform
source-data provenance), **P19** (formal model/experiment registry), **P4** (move automatic
model fitting out of the live request path).

All file:line references are against the repo at `SPORTSBOOK_ODDS/deploy` as of this eval
(git clean, main @ 5c4549b family per work-log).

---

## P4 — Move automatic model fitting OUT of the live request path

**current_state: PARTIAL (the concern is REAL and present; the loop exists and is gated).**

### What actually happens (traced end-to-end)
The live prop-analysis path DOES trigger training + a durable calibration write:

- `props.py:897` — inside `analyze_player_props_value` (the live analyze path, the same
  function the Streamlit prop pages call) → `maybe_auto_refit(sport_key)`.
- `recalibration.py:2278 maybe_auto_refit` → throttled to once/hour per process
  (`AUTO_MAINTENANCE_INTERVAL_SECONDS = 3600`, recalibration.py:75) → `maintain_sport`.
- `recalibration.py:2240 maintain_sport` → `resolve_pending_outcomes` (ESPN/StatsAPI box-score
  HTTP, capped at `MAX_RESOLVE_PER_LAUNCH = 80`, recalibration.py:54) then, if the gate opens,
  `refit_sport`.
- Refit gate (recalibration.py:2260-2265): `last_fit_ts is None` OR
  (`age_hours >= MIN_REFIT_INTERVAL_HOURS=12` AND `_count_resolved_since >= MIN_NEW_FOR_REFIT=25`).
- `recalibration.py:2127 refit_sport` → `fit_platt_chronological` (champion gate, see below) →
  `save_recalibration` (recalibration.py:1909) → `_db.save_recal(sport_key, blob)`
  (recalibration.py:1937) — a **durable Azure SQL write** to `recalibration_params`.

So a user opening/refreshing the app can, on the request path, cause: (a) external box-score
HTTP fetches, (b) a Platt refit, and (c) a durable SQL calibration mutation. This is precisely
the audit's scenario "calibration state changing because a user opened or refreshed the
application" (audit txt lines 293-295).

### Mitigations already in place (why this is not as bad as the audit assumes)
- **Champion gate**: `fit_platt_chronological(records, incumbent=...)` (recalibration.py:1796-1855)
  returns `None` unless the candidate beats BOTH the seed map AND raw on Brier AND log-loss in
  BOTH expanding chronological folds, and clears `MIN_OBS_FOR_OVERRIDE`. `refit_sport` seeds the
  incumbent from the committed book-line prior (`_read_local_recal`, recalibration.py:2149,2184-2186).
- **Validated-only application**: `_parse_recal_blob` (recalibration.py:1958-1979) only surfaces
  props with `validated is True`; unvalidated fits are stored but never applied.
- **Seed stays pristine**: for a SQL refit `save_recalibration` skips the local file write
  (recalibration.py:1924-1930), so the loop writes to SQL only and the committed seed remains the
  prior. This is the "online Platt seed-as-prior" redesign from memory.
- **Fail-open**: `maybe_auto_refit` swallows all exceptions (recalibration.py:2291-2294); the loop
  is throttled + bounded.

### Verdict: **adapt** (already a first-class Friday workstream)
The audit's architectural principle (live reads an approved calibration version; offline resolves
+ fits + validates + publishes) is correct and worth doing. But it must be adapted to this app's
existing champion-gate/fail-open machinery — not rebuilt. The clean move:

1. In the live path, keep `load_recalibration`/`load_calibration` (read-only) — unchanged.
2. Demote `maybe_auto_refit` on the request path to **resolve-only** (or remove it), and make
   `refit_sport`/`save_recal` fire ONLY from the offline entrypoint.
3. The offline home **already exists**: `forward_tracker.py --resolve` → `resolve_and_refit` →
   `maintain_sport` (forward_tracker.py:35-37, 21-22). It even promotes SQL secrets
   (forward_tracker.py:_main) so it writes to prod SQL correctly.

**TRAP (must flag):** there is **no scheduler/cron in the repo** (grep for
apscheduler/cron/BackgroundScheduler/azure-function = none; the only "schedule" hits are MLB game
schedules). Today the in-request `maybe_auto_refit` IS the de-facto cadence; `forward_tracker
--resolve` is manual. If we rip out the in-request trigger without establishing a scheduled
offline run (Task Scheduler / GitHub Action / Azure WebJob calling `forward_tracker --resolve`),
**calibration silently stops updating**. So P4 is a two-part deliverable: (i) remove/​demote the
in-request refit, (ii) stand up an offline cadence. Part (ii) is the actual new work.

**Integration:** WS6 (champion-gated pitcher online Platt) and WS1 (SQL-off hardening) already
touch this exact loop; fold P4 in there, plus a small "offline cadence" note. Also naturally
sits beside WS4 (like-for-like scoring). Effort **M** (wiring + establishing cadence; the fit
machinery already exists and is validated).

---

## P3 — Immutable per-prediction model provenance

**current_state: ABSENT for the version/provenance fields (only `created_at` present).**

### prediction_log columns today (schema.sql:14-122, incl. idempotent ALTERs)
`id, ts, sport_key, event_id, event_key, commence_time, prop_key, player, game_date, direction,
book, resolved_at, line, raw_prob, final_prob, projected, actual, price, outcome, is_value,
resolved, refit_performed, team, batting_order, player_mlb_id, team_code, player_key`.

Mapping to the audit's requested P3 fields:
| audit field | present? | evidence |
|---|---|---|
| prediction_created_at | **yes** | `ts` = `datetime.now(utc).isoformat()` (recalibration.py:570) |
| raw_prob / final_prob | yes (not asked but relevant) | schema.sql:28-29; row build recalibration.py:578-579 |
| model_version | **no** | not in schema; grep finds it only in audit txt |
| code_commit_sha / git_sha | **no** | grep: absent from all `.py` |
| calibration_version | **no** | absent (see note: `fit_timestamp` exists at the calibration-blob level, not per-row) |
| calibration_fit_timestamp | **no** (per-row) | only `recalibration_meta.fit_timestamp` (schema.sql:292) exists globally |
| feature_set_version | **no** | absent |
| data_asof | **no** | absent |
| odds_snapshot_id | **no** | absent as an FK on prediction_log (but `odds_snapshot.id` IDENTITY PK exists, schema.sql:302) |

`log_prediction`'s row dict (recalibration.py:569-590) writes none of the version fields.
`market_prediction_log` (schema.sql:195-245) is the same story: `ts`, `raw_prob`, `model_prob`,
but no version columns.

### What already exists that this can lean on
- Deployed calibration blob has a top-level `fit_timestamp` (calibration/baseball_mlb.json TOP
  KEYS include `fit_timestamp`) → a natural **calibration_version** value to stamp per row.
- `fit_basis` per-prop cfg ("synthetic_sweep"|"real_line", refit_calibration.py:656,996) →
  provenance of HOW each prop's calibration was fit.
- `player_key`/`player_mlb_id` already give durable identity provenance on each row.

### Verdict: **adapt** (fits philosophy; scope it to cheap high-value fields)
Reproducibility of an ADAPTIVE model is a genuine gap and matches the app's discipline ethos.
But don't add all 8 columns wholesale — grade by cost/value:
- **Cheap + high value (do first):** `git_sha` (a `subprocess`/`importlib.metadata` startup
  helper, cached module-global), `calibration_version` (stamp the loaded blob's `fit_timestamp`),
  `fit_basis` per row (already computed). These answer "which model state produced 63.7%".
- **Medium:** `data_asof` — the app has as-of timestamps (statcast `as_of_date`, gamelog
  `last_fetched_at`) but no single per-prediction data cutoff; would need to thread one value.
- **Hard / lower ROI here:** `feature_set_version` — features are code in `prop_features.py`, not
  an integer-versioned set; would need a registry hash. `odds_snapshot_id` — live analyze reads
  freshly-fetched odds, not necessarily a persisted `odds_snapshot` row, so the FK is often
  unavailable at prediction time (would need to persist the live snapshot first). Defer both.

**TRAPs:**
- `test_db_store.py::SchemaParityTests` enforces schema==SQLAlchemy metadata; any new column needs
  schema.sql + db_store.py metadata + the idempotent `IF COL_LENGTH ... ALTER` block + the writer.
- prediction_log is UPSERT-de-duped by forecast identity (recalibration.py:623-663,
  `_collapse_identity_rows`); "immutable" is aspirational — a re-log of the same
  (sport,event,prop,player_key,line) collapses rows. Provenance fields should be treated as
  "stamp on first write, don't overwrite on re-log" or they'll churn. Decide the merge policy.

**Integration:** propose new **WS10 "prediction provenance"** (or fold into WS4, which already
wants version-aware like-for-like scoring). Effort **M**.

---

## P6 — Uniform source-data provenance

**current_state: PARTIAL — provenance exists but is scattered/non-uniform.**

### What exists (per-table, ad hoc)
- `gamelog_fetch_meta.last_fetched_at` (schema.sql:475) — TTL/fetch time, but on the META table,
  not the fact rows.
- `odds_snapshot.captured_at` (schema.sql:307) — capture time, but no `source_provider`
  (implicitly The Odds API) and no `source_record_id`.
- `statcast_player_asof.as_of_date` (schema.sql:524) — as-of cutoff (leakage-safe), no provider.
- `player_id_map.source` + `fetched_at` (schema.sql:584-585); `id_map_meta.last_fetched_at`.
- `recalibration_meta.source` + `fit_timestamp` (schema.sql:291-292); `recalibration_params.source`.

### The gap
No uniform `source_provider` / `source_record_id` / `ingested_at` principle. Most notably the
durable gamelog fact tables `mlb_batter_gamelog` / `mlb_pitcher_gamelog` (schema.sql:376-418) key
on `athlete_id` (ESPN) and carry a **synthetic** `game_key` with **no `source_provider` column** —
you cannot tell from a row whether it came from ESPN or MLB StatsAPI. This matters precisely
because the work-log reports pitcher logs migrating to StatsAPI while batter logs may still be
ESPN-origin; today the schema can't distinguish them. `odds_snapshot` has no provider/record id
either.

### Verdict: **adapt + defer** (align with the P1 gamePk warehouse work)
Worth doing as a warehouse principle, but low urgency: the leakage-critical timestamps
(`as_of_date`, `last_fetched_at`) already exist where they matter. Adapt to targeted columns
rather than a blanket 5-field expansion:
- Add `source_provider` (and ideally `source_game_pk`/`source_record_id`) to the gamelog fact
  tables when the StatsAPI game-centric ingestion (audit P1, a different cluster) lands — that's
  when origins become genuinely mixed and the column earns its keep.
- Add `source_provider` to `odds_snapshot` (constant "THE_ODDS_API" today, but explicit).

**Integration:** backlog, gated on / bundled with the P1 gamePk canonical warehouse migration
(other cluster owns P1). Trigger: StatsAPI game-centric ingestion build. Effort **M**.

---

## P19 — Formal model/experiment registry

**current_state: PARTIAL — a deployed-calibration record exists; a candidate/rejected experiment
registry does not.**

### What exists
- `recalibration_params` (schema.sql:248-267): per (sport_key, prop_key) — a, b, n_fit,
  n_validation(_folds), holdout raw/calibrated Brier + log-loss, `holdout_metric_scope`,
  `deploy_fit_scope`, `validated` BIT, `source`. This is a real per-prop DEPLOYED calibration
  record WITH validation metrics — but PK is (sport_key, prop_key), so it holds only the current
  winner, no history, no candidate/rejected/retired lifecycle, no `git_sha`, no ROI/CLV.
- `recalibration_folds` (schema.sql:271-284): per-fold validation detail.
- `recalibration_meta` (schema.sql:288-293): `fit_timestamp` + `source`.

### The gap
No `model_run`/`experiment` table with `status ∈ {candidate, validated, deployed, retired,
rejected}` (grep: `model_run`/`experiment` absent from code). The app is doing extensive manual
MLOps — the whole accuracy roadmap is a sequence of candidate experiments (NegBin "E", rest,
platoon, gamecontext, xBA→hits, CSW→K) with rigorous forward-validation and deliberate rejection —
but that history lives in `MEMORY.md` / the roadmap doc + the ephemeral `--negbin-diag` /
`--feature-diag` / `--roi-diag` no-write lenses, NOT as structured SQL rows. When a lens is re-run
as the warehouse grows, the prior verdict isn't captured anywhere queryable.

### Verdict: **adapt (lightweight); defer the full form** — respects philosophy, don't over-build
The audit's full `model_run` schema (hyperparameters, training/validation windows, per-run ROI/CLV)
is over-engineered for a per-prop Platt + method-select system with no hyperparameter search. But a
small append-only **experiment/candidate log** that captures exactly what the diag lenses already
compute has real value and directly serves the "conservative rejection is a strength" ethos:
columns like `(ts, git_sha, sport_key, prop_key, line_bucket, kind[method|feature], name,
n_obs, delta_brier, delta_roi, decision[ship|reject], gate_threshold, notes)`. It turns the prose
roadmap into queryable rejection history and lets a re-run detect "this cleared the gate for the
first time."

**Integration:** propose new **WSx "experiment registry"** (lightweight, append-only), or backlog.
Naturally piggybacks on the existing `--*-diag` code paths (they already compute the deltas).
Effort **M**. Trigger: when the roadmap's "re-run the 3 free lenses as the warehouse accrues" loop
becomes frequent enough that prose tracking hurts.

---

## Summary table

| P | Title | current_state | verdict | integration | effort |
|---|---|---|---|---|---|
| P4 | Fitting off the live request path | partial (loop present + gated) | adapt | WS6/WS1 (+offline cadence) | M |
| P3 | Immutable per-prediction provenance | absent (only `ts`) | adapt | new WS10 (or WS4) | M |
| P19 | Model/experiment registry | partial (deployed-only) | adapt (lightweight)/defer | new WS / backlog | M |
| P6 | Uniform source-data provenance | partial (scattered) | adapt/defer | backlog w/ P1 warehouse | M |

### Cross-cutting notes / risks
- **P4 is the highest-value keeper in this cluster** and already maps onto Friday WS6/WS1; the only
  genuinely new work is standing up an offline cadence (no scheduler exists today).
- P3 + P19 + P4 compose: an offline "publish approved calibration version" step (P4) is the
  natural place to mint a `calibration_version` (P3) and write a `deployed` experiment row (P19).
  Sequence P4 → P3 → P19 if bundling.
- Every schema addition trips `SchemaParityTests` (test_db_store.py) — budget the metadata + DDL +
  writer + test churn into each estimate.
- None of these violate app philosophy (DK-only, Brier-first gate, conservative feature deployment).
  They are pure reproducibility/observability plumbing.

---

## Verifier verdict (independent re-check against live code)

Re-verified every citation against the actual repo (schema lives at `sql/schema.sql`, not repo
root — the memo's bare `schema.sql:NN` refs resolve there; **content matches**, and the
recalibration.py/props.py line numbers are exact). Prioritized falsifying the "partial/absent"
labels. Bottom line: **all four findings hold; no dangerous false-"already-done" and no
philosophy-violating over-adopt.** Details + one rationale correction:

- **P4 (partial / adapt) — AGREE.** Confirmed the live analyze path fits + writes durable SQL:
  `props.py:897` `maybe_auto_refit` (inside `analyze_player_props_value`) → `maintain_sport`
  (recalibration.py:2240, hourly throttle 2278-2289) → gate (2260-2265) → `refit_sport` (2127) →
  `save_recalibration` (1909) → `_db.save_recal` (1937, durable Azure SQL). All four mitigations
  verified: champion gate returns None unless the fit beats BOTH seed AND raw on Brier AND log-loss
  in BOTH folds + clears MIN_OBS_FOR_OVERRIDE (fit_platt_chronological 1820-1858); validated-only
  application (_parse_recal_blob 1977 `cfg.get("validated") is True`); seed stays pristine (SQL
  refit skips local write, save_recalibration 1926); fail-open (2291-2294). Offline home exists
  (`forward_tracker.py:35-37`). **TRAP independently confirmed: no scheduler anywhere** — grep for
  apscheduler/BackgroundScheduler/schedule.every/crontab/azure.functions = 0 hits; the only
  `.github/workflows` file is `tests.yml` (push/PR test runner, NO `schedule:` cron). So the
  in-request trigger IS the de-facto cadence and the real new work is standing up an offline run.
  Note: there is a SECOND in-request write path — `app.py:1177` a manual "Resolve all pending
  predictions now" button calls `maintain_sport(max_resolve=5000)` — but it's user-initiated/opt-in,
  so it's acceptable to keep; the finding correctly targets the *automatic* path.

- **P3 (absent / adapt) — AGREE (strongest keeper for reproducibility).** Falsification attempt
  FAILED to find any provenance field: a codebase-wide grep for
  git_sha/model_version/calibration_version/feature_set_version/feature_version/data_asof/
  odds_snapshot_id/model_config_hash/commit_sha returns **zero matches in all .py**. `log_prediction`'s
  row dict (recalibration.py:569-590) writes only ts + analytic fields; `_enrich_prediction_ids`
  adds mlb_id/team_code/player_key, still no version stamp. Leverage points confirmed: blob-level
  `fit_timestamp` (save_recalibration:1919; recalibration_meta.fit_timestamp schema:291) and per-prop
  `fit_basis` ("synthetic_sweep" refit_calibration.py:656 / "real_line" 996). Scoping to git_sha +
  calibration_version + fit_basis is right-sized and philosophy-consistent. The "immutable is
  aspirational" trap is REAL: `_collapse_identity_rows` (recalibration.py:738) + `compact_prediction_log`
  (668) collapse re-logged forecasts sharing (sport,event_key,prop,player_key,line) — so provenance
  must be stamp-on-first-write, don't-overwrite, or it churns. Merge policy must be decided.

- **P19 (partial / adapt-lightweight, defer-full) — AGREE.** `recalibration_params` (schema:248-267)
  is deployed-only: PK (sport_key, prop_key) = current winner only, no history/candidate/rejected/
  retired lifecycle, no git_sha/ROI/CLV; it does carry `validated` BIT + holdout Brier/log-loss +
  scope, so "partial" (a deployed slice) is fair, not overstated. No `model_run`/`experiment` table
  exists (whole schema read; grep absent). The diag lenses the memo proposes piggybacking on are
  real (`--negbin-diag`/`--feature-diag`/`--roi-diag` in refit_calibration.py; test_feature_diag.py,
  test_roi_diag.py). Lightweight append-only log is right-sized; the audit's full model_run schema
  (hyperparameters/training windows) is genuinely over-built for a per-prop Platt+method-select
  system with no hyperparameter search. Defer-the-full-form is correct.

- **P6 (partial / defer) — AGREE on state + verdict, with ONE rationale correction.** Confirmed
  no uniform source_provider: gamelog fact tables key on ESPN `athlete_id` with a SYNTHETIC
  `game_key` (`f"{game_date}|{opponent}|{is_home}"`, gamelog_store.py:244-245) and no provider
  column (schema:376-418); odds_snapshot has no provider (schema:299-316); scattered per-table
  provenance exists exactly as cited. **Correction:** the memo defers on the premise that origins
  become "genuinely mixed" only *when* the StatsAPI game-centric ingestion lands — but the mixing
  has ALREADY happened. `_fetch_espn` (gamelog_store.py:203-223) reroutes MLB pitchers to
  `mlb_starters._pitcher_gamelog_or_synth`, which returns **TRUE StatsAPI per-game data when the
  player name is known, else ESPN season-splits synth** — both land in the SAME `mlb_pitcher_gamelog`
  table under an ESPN athlete_id, indistinguishable by column, and the StatsAPI gamePk is DISCARDED
  in favor of the synthetic key. So the P6 gap is present today, not future. That said, origin is
  still partly derivable (batter table = ESPN; pitcher rows = StatsAPI-when-named), harm is low, and
  the clean fix is a `source_provider` (+ `source_game_pk`) column bundled with the P1 gamePk
  canonical migration — so **defer/bundle-with-P1 remains the right call**; only the "trigger hasn't
  fired yet" framing is wrong.

**Cross-cutting:** none of the four touch the calibration math, the Brier gate, DK-only pricing, or
feature deployment — they are observability/reproducibility plumbing, orthogonal to the modeling
philosophy. The only real regression surface is (a) SchemaParityTests churn on any new column
(P3/P6/P19) and (b) P4's demotion silently stopping calibration if the offline cadence isn't stood
up first — both already flagged in the memo.
