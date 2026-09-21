# F28 — Prediction provenance & replay: design proposal (for Doug's approval)

**Status:** proposal — *no schema or serving change until you approve.* F28 touches the
prediction corpus (irreplaceable calibration data), so per the audit + the
never-destroy rule this is a design-doc-first task.

## The problem (audit F28)
Prediction rows carry raw/final probabilities, event/player identity and a coarse
`source`, but **not** enough to *replay* a historical recommendation exactly or to
attribute a performance change:
- no model / config / calibration **hash**, no input **data version**, no
  **quote-as-of** envelope;
- the live NFL helper uses **wall-clock** season/week (and a `week=99` "all prior"
  sentinel), and its early-season history can spill into the prior season while the
  offline observation cohort is same-season gated — so live and backtest cohorts can
  differ for the same event;
- book calibration is keyed by sport/market only, with no reference-book / lead-time
  domain.

## What already exists to build on (don't reinvent)
- `calibration_loader.serving_fingerprint(sport_key)` → stable 12-char md5 of the
  serving calibration, **excluding** volatile `fit_timestamp`/`meta` (identical
  calibration → identical hash). This is our **calibration hash** already.
- `feature_store.FEATURE_SCHEMA_VERSION = 2` + `feature_store` cache key = model/feature
  schema version.
- `prediction_log` is a SQLAlchemy `Table` (db_store) → provenance can be **added as
  nullable columns**; existing rows stay valid as NULL/"unknown" provenance
  (never-destroy compliant, no rewrite of history).

## Proposed shape

### 1. An immutable `PredictionContext` (one value object, built at serve time)
Fields (all derivable from data we already have at serve time):
| field | source |
|---|---|
| `sport_key`, `event_id`, `commence_time` | already logged |
| `event_season`, `event_week` | derived from `commence_time` (NOT wall-clock) — fixes the week=99 gap |
| `as_of` | the quote/serve cutoff timestamp; features must use only rows strictly before it |
| `calibration_hash` | `calibration_loader.serving_fingerprint(sport_key)` |
| `model_schema` | `feature_store.FEATURE_SCHEMA_VERSION` (+ per-sport model-artifact version, e.g. the frozen `nfl_prop_models.json` build id) |
| `method` | the serving method actually used (A/C/D/E / nfl_model / independence) + fallback flag |
| `reference_book_policy` | which books fed the de-vig + lead-time bucket |
| `context_fingerprint` | md5 of the tuple above (stable id; changes iff any input changes) |

### 2. Storage (additive migration)
Add nullable columns to `prediction_log`: `event_season`, `event_week`, `as_of`,
`calibration_hash`, `model_schema`, `method`, `reference_book_policy`,
`context_fingerprint`. Old rows → NULL = "unknown provenance" (kept, still gradable).
Owner-run idempotent `ALTER TABLE ADD COLUMN` (like the parlay-tracker DDL).

### 3. Replay harness
`replay(prediction_row)` → recompute probability + selected offer using ONLY the row's
`as_of`/`event_season`/`event_week` + the artifacts named by its hashes; assert it
reproduces the stored `final_prob` (± tol) with no wall-clock reads. A fixture proves
replay; changing model/data/quote version changes `context_fingerprint`.

### 4. Validation-domain partitioning
Report Brier/ROI partitioned by `model_schema` / `calibration_hash` / lead-time /
market / fallback-status, and flag early-season prior-season spillover explicitly.

## Suggested phasing (each shippable, low-risk first)
- **P1 — stamp (additive, safe):** add the columns + populate on NEW rows from the
  context; derive `event_season/week` from `commence_time`; fix the NFL wall-clock
  week=99 serve path to use event-derived week. No replay yet. Old rows untouched.
- **P2 — replay harness + parity test** (early-season spill, artifact-parameter parity).
- **P3 — validation-domain partitioned reports.**

## Open decisions for you
1. **Schema:** nullable columns on `prediction_log` (proposed) vs a separate
   `prediction_context` table joined by a fingerprint. Columns are simpler; a table
   de-dupes identical contexts. **Recommend: columns.**
2. **Model-artifact version:** is `FEATURE_SCHEMA_VERSION` + `serving_fingerprint`
   enough, or add an explicit frozen-artifact build id (e.g. hash of
   `nfl_prop_models.json`)? **Recommend: add the artifact hash — cheap, precise.**
3. **Scope now:** ship **P1 only** (the safe additive stamp + week=99 fix) and defer
   P2/P3, or commit to all three? **Recommend: P1 now, P2/P3 as follow-ups.**
4. **Owner DDL:** OK to add an idempotent `sql/prediction_provenance.sql` you run on
   Azure (same pattern as `sql/parlay_tracker.sql`)?

Nothing is implemented pending your answers to 1–4.
