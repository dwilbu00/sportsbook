# Third-Party Audit — INTEGRATION DECISION (folds keepers into the Friday plan)

---
## ✅ CONFIRMED DECISIONS (Doug, 2026-08-10) — authoritative over the buckets below
1. **API key:** rotate only, no history scrub (owner action).
2. **Tier-1 + cheap fold-ins:** fold ALL in — WS10 entity-resolution; WS1 += FK/db-telemetry; WS4 += N_eff diag + CLV panel; WS6 += offline refit cadence.
3. **Tier-2:** adopt ALL four — WS11 units, WS12 provenance, WS13 staking haircut, WS14 telemetry.
4. **WS15:** adopt AND **promote to right after WS1**. Motivation is **integrity / best-practice / efficiency — explicitly NOT accuracy**. Build the **canonical-key** dimension/warehouse (gamePk for games, MLBAM id for players) on **natural keys**, delivering the dimension-entity robustness **without an enforced FK tree**.
   - **Row-level surgical UPSERT is a core principle** (update-in-place / insert-new / delete-only-genuine, via the proven `db_store.mutate` app-side-diff pattern, db_store.py:725-788 — NOT raw T-SQL MERGE).
   - **Upsert scope = convert EVERY replace-all site** (gamelogs gamelog_store.py:459-465; save_recal db_store.py:879-888; SFBB map player_id_map.py:309; statcast statcast_asof.py:144; caches). Broadest integrity win; stage carefully, tests green per site.
5. **WS16** (module split): stays deferred (never split while WS3/4/6 churn those files).
6. **Out-of-band hygiene:** P22 ruff warn-only curated ruleset; P21 pin `requests` now (hashed lock later).

**P7 clarification:** REJECT applies ONLY to the enforced FK tree / 3NF normalization (OLTP pattern guarding a concurrent-writer orphan race this single-writer app has engineered away, + unguarded by name-only SchemaParityTests, + fights fail-open NULL enrichment). The **dimension-entity / canonical-key intent is ADOPTED via WS15**. Integrity = stable natural keys + row-level upsert + occasional offline orphan-audit query. (Verify workflow wbdfh3z35, 4 agents.)

**FINALIZED SEQUENCE:** Phase A = WS8 → WS1 (+odds_line FK-or-audit +database_failure telemetry) → **WS15** → WS10 (file-disjoint, may overlap) → WS2..WS9, with WS11/12/13/14 at their noted homes.
No code changed yet; git clean; awaiting Doug's "start".
---

**Date:** 2026-08-10
**Author:** integration/sequencer pass (read-only; verifies every claim against live code)
**Inputs:** the 6 cluster eval memos + their adversarial verifier verdicts
(`third-party-eval/{data-architecture,entity-resolution,provenance-training,sql-infra-ops,maintainability,modeling-staking}.md`),
the internal Friday design (`memory/mlb-2026-reset-audit-verdict.md` + `design/00-IMPLEMENTATION-ORDER.md`,
workstreams WS1–WS9), and the 3 audit reports.
**Method:** finder-vs-verifier disagreements are reconciled by trusting concrete code evidence. Spot-checks
re-run this pass (all confirmed): ER5 no-team-hint (`gamelog_store.py:485-490`), P15 `eff_n = sum(weights)`
(`props.py:1125` + `stats._recency_weights:76-93`), P4 in-request refit (`props.py:897`), P3 zero provenance
fields (grep = 0 in all `.py`), P23 live key in history **and** current secrets.

**Rule respected throughout:** DK-only pricing; Brier-first gate (min gain 0.002); conservative feature
deployment (ship only after chronological forward validation, never on intuition); fail-closed identity;
leakage-safe as-of; reversible pure-code changes; **one commit per logical change; NEVER push (owner pushes).**

---

## 0. Executive answers

- **(a) SINGLE highest-value NEW idea → WS10 ER5+ER4: end-to-end fail-CLOSED entity resolution.**
  Today the id-*binder* fails closed but the *pipeline* fails **open**: an ambiguous MLB name drops to a
  `name:<norm>` key, still produces a prediction/bet, and the stat-fetch falls through to ESPN
  `search_athlete` → `candidates[0]` (`espn_client.py:610`) — a silent wrong-player bind that poisons
  projection → edge → recommendation (the money path). It is the ONE keeper that fixes a live defect, is
  small/reversible, exactly matches the stated "fail-closed identity" philosophy, and no existing WS covers
  identity. ER5 (thread home/away team **names** into the SFBB resolver) is the cheap do-first; ER4 (skip a
  genuine ≥2-id namesake collision as a counted diagnostic) is the payoff.

- **(b) BIGGEST TRAP / most over-rated → the game-centric StatsAPI/gamePk warehouse (P1 + StatsAPI doc)
  sold as an accuracy play.** The finder called it "highest-value in the cluster"; the verifier refuted the
  payoff with code: ESPN batter gamelogs **already** return real per-game, dated, variance-preserving rows
  (`espn_client.py:776,792`) — pitchers went to StatsAPI only because ESPN's gamelog is EMPTY for pitchers
  (`espn_client.py:833`). So the warehouse delivers **~zero new modeling signal for batters**; its genuine
  value is architectural (canonical MLBAM/gamePk identity, provider independence, reproducibility). It is an
  **L**-effort swap of the single most load-bearing data path feeding every projection + all calibration →
  real regression risk (silent stat-definition/calibration drift). Ship it as identity/reproducibility
  hardening AFTER WS1–WS9, gated on a concrete need — never as an accuracy bet.
  *Runner-up technical trap:* P15's shrinkage-denominator swap sold as "more conservative." The math is
  **inverted** — `sum(w) < Kish N_eff < n`, so swapping `sum(weights)` → Kish gives a **higher** pseudo-count
  = **less** shrinkage = mildly **anti**-conservative. Take the free diagnostic; route any swap through the
  gate with no "more correct = safer" presumption. (Also: P11's "~100× stake error" is overstated — the 5%
  Kelly cap clamps it; the real unbounded risk is prob>1 into the un-capped parlay copula.)

- **(c) Genuinely ALREADY DONE (moot — auditor had stale context):**
  - **ER1 (P2a) event-scoped identity key** — already stronger: `player_key` is id-first (`mlb:<id>` else
    `name:<norm>`, `db_store.py:108-120`); UNIQUE `(sport_key,event_key,prop_key,player_key,line)` with
    `event_key` derived on every write (`db_store.py:452-458`) → self-heals, can't drift.
  - **ER2 (P2c) team-as-evidence / fuzzy-as-candidate-gen / fail-closed id-binder** — the SFBB migration win.
    `_unique_id` returns None on >1 id, team only a canonicalized tiebreak, UNIQUE never team-filtered
    (`player_id_map.py:537-578`); zero similarity-score authority anywhere (grep difflib/fuzz/ratio = 0).
  - **P16 feature framework** — `prop_features.FEATURE_REGISTRY` genuinely shared by all three projection
    paths (`backtest.py:2662`, `book_line_calibration.py:800`, `props.py:1247`) and gated by
    `diagnose_features`. (The *program* of feeding it continues — that's the adopt.)
  - **P5 warehouse-first read** — model reads local SQL with 0 external calls on fresh meta
    (`espn_client.py:1008-1021`); past seasons permanent (5yr TTL). (Only the game-centric-ingest delta is new.)
  - **P18 wager-level DK-vs-DK CLV** — `apply_clv_updates:599`, `avg_clv_pct`, `backfill_dk_clv.py`. (Only
    the *model-eval* CLV panel is new.)
  - **P23 secrets architecture** — config.json/secrets.toml gitignored + never committed. (Only the actual
    key *rotation* is outstanding — and it is urgent.)

---

## 1. Reconciled finder-vs-verifier disagreements (code wins)

| Item | Finder | Verifier (trusted) | Resolution |
|---|---|---|---|
| P1 / StatsAPI-doc | "highest-value in cluster" | batters already have real ESPN per-game rows → ~0 new signal; identity/reproducibility only; regression-risk on most load-bearing path | **adapt, tier 3, NOT an accuracy play**; priority after WS1–9 |
| P15 shrinkage swap | Kish = more shrinkage, conservative, fewer thin-sample edges | math inverted: `sum(w) < Kish`; swap = *less* shrinkage, mildly anti-conservative | **diagnostic surface = clean adopt**; shrinkage swap = gate-only, defer enthusiasm |
| P11 severity | "~100× stake error" | 5% Kelly cap clamps it → "EV gate silently defeated" (bounded); unbounded risk is parlay copula fed prob>1 | **adapt WS11**; justify on bounded-bug + parlay-garbage, not 100× |
| P21 lockfile | full hashed lock = cheap S | uv on Streamlit Cloud + `--require-hashes` is the unverified risk | **pin `requests` now (S)**; hashed lock = separate deploy-validated step |
| P6 trigger | mixing happens "when StatsAPI lands (future)" | mixing already happened (pitcher StatsAPI vs ESPN synth share one table) | substance unchanged (defer/bundle w/ warehouse); framing corrected |
| ER3 label | JSON says "adapt" | memo body says "defer" | **defer/coordinate** (identical substance): reuse a WS15 gamePk→roster cache |
| ER4 damage path | contaminates calibration | statsapi hard-ID grader already fails closed; leaks only via ESPN fallback | primary damage = **live recommendation** (money path); calibration route partial. Sequence ER5→ER4. |

No false "already-done" was found hiding a real gap. The one dangerous tilt is the *opposite*:
over-investing in the warehouse on a modeling-benefit claim that doesn't hold for batters.

---

## 2. Bucketed verdict for every audit priority

### ADOPT (take it; only rollout tweaks)
- **P23: rotate the leaked Odds API key** → immediate owner one-off (regenerate at the-odds-api.com; update
  secrets.toml + config.json + Streamlit Cloud secret). Live key `86e5622c…` is in git history AND is the
  current active key. History scrub optional (rotation makes the leaked value worthless). Top urgency, ~minutes.
- **P22: add ruff to CI** → new fast step in `.github/workflows/tests.yml`; **warn-only / curated ruleset first**
  (F401/F821/star-imports) or it reds every PR day-one on pre-existing F401s; enforce once clean.
- **P16: feed new MLB/NBA candidates through the existing gate** → standing feature-candidate backlog, NO new
  plumbing; the framework is done and is the only proven edge-adder. NBA blocked on NBA season data + calib.
- **P15 (diagnostic half only): surface Kish N_eff `(Σw)²/Σw²`** beside `games_sampled` in prop/team
  diagnostics + calibration lenses → into **WS4**. Pure add, no projection change, free; natural P14 input.

### ADAPT (scoped/modified version is the keeper)
- **ER5 (P2 / do-first): thread home/away team NAMES into the projection stat-fetch resolvers**
  (`_mlb_espn_id`, `get_pitcher_gamelog` → `mlb_id_for_name/espn_id_for_name(..., teams=)`) → **WS10**.
  Plumbing already proven at `props.py:514,1146` (xBA path already passes teams). Fail-open. **S**.
- **ER4 (P2 / top keeper): MLB-gated ambiguity gate** — on a genuine ≥2-id namesake the team-hinted resolver
  can't break, **skip the prop** (counted diagnostic), don't fall to ESPN `candidates[0]`. Distinguish
  "ambiguous" (skip) from "unmapped callup" (keep degrading). MLB-only, fail-open for NBA/NFL → **WS10**. **M**.
- **ER7 (P2 / optional): role as a namesake tiebreak** inside `_unique_id` (pitcher_* vs batter_*) before
  failing closed → **WS10** (or defer). **S**.
- **P3: immutable per-prediction provenance** — cheap high-value fields only: `git_sha` (startup helper),
  `calibration_version` (loaded blob `fit_timestamp`), `fit_basis` per row (already computed). Stamp-on-first-write
  (rows collapse on re-log via `_collapse_identity_rows`). Defer `feature_set_version`/`odds_snapshot_id` →
  **WS12**. **M**.
- **P4: move auto-fitting off the live request path** — demote in-request `maybe_auto_refit` (props.py:897) to
  resolve-only; make refit fire from the existing offline home (`forward_tracker.py --resolve`); **stand up a
  scheduled offline cadence** (none exists — grep cron/scheduler = 0). Fold into **WS6 + WS1**. **M** (the
  cadence is the real new work). ⚠ removing the trigger without a scheduler silently freezes calibration.
- **P7: one FK `odds_line.snapshot_id → odds_snapshot.id` ON DELETE CASCADE** → **WS1**. Sole writer is atomic
  parent-then-child (`db_store.py:900-926`) so orphans are already impossible; value is forward (retention
  cascades). Pre-check orphans before ALTER; add a `PRAGMA foreign_keys=ON` test (SchemaParity checks names
  only). **Reject** the broad game/player/odds_event FK vision (those tables don't exist). **S**.
- **P10: minimal typed row DTO** for the persisted wager/prediction row only → rides **WS12/WS7**; incremental
  value is modest (persistence is already `_PREDICTION_SPEC`/`_WAGER_SPEC` spec-driven), so it earns its keep
  only alongside the provenance fields — not a standalone workstream. **Reject** the full frozen hierarchy
  (fights the load-bearing `.get(default)` idiom; no type-checker in CI). **S**.
- **P11: units normalization** — make candidate dicts fraction-native, single ×100 in one presentation helper,
  delete ~30 `/100.0` + `100.0-x` reconversions → **WS11**. No SQL migration (DB already fractions). Golden-test
  guarded (pixel/cent byte-identical). **M**.
- **P13: correlation-aware staking (haircut)** — per-bet Kelly haircut = f(Σ positive ρ to already-selected
  legs) via existing `parlay._pair_correlation`, applied BEFORE `scale_to_slate_cap` → **WS13**. Haircut-only
  (never increases). Reduces variance/drawdown, not ROI; bounded by existing half-Kelly + 5%/25% caps. **M**.
- **P17: market-structure features (dispersion / line-movement / time-to-start)** → **P16 candidate backlog**,
  routed through the same chronological gate. DK-only-safe (non-DK dispersion is a SIGNAL, never a price).
  Optional forward-only `odds_line` dispersion column if it graduates past research. **M** research + S–M column.
- **P18: aggregate CLV co-panel** (avg CLV + CLV hit-rate + n) alongside Brier/ROI on the model-eval surface →
  **WS4**. Brier stays PRIMARY. Per-model/version split blocked twice (data accrual + P3) → defer to WS12. **M**.
- **P19: lightweight append-only experiment registry** — one row per diag-lens verdict
  `(ts, git_sha, sport, prop, line_bucket, kind, name, n_obs, Δbrier, Δroi, decision, gate, notes)`, piggybacking
  the existing `--negbin/feature/roi-diag` paths → rides **WS12** (git_sha comes from the provenance helper) or
  backlog. **Reject** the audit's full `model_run` schema (hyperparameters/train-windows — over-built for a
  per-prop Platt+method-select system with no hyperparameter search). **M**.
- **P20: ops telemetry (scoped)** — a small `logging` config + `ops_event(kind, **fields)` helper called at the
  handful of bare-except sites that matter (identity/db/api/fallback), plus a per-run counters dict in a
  Streamlit Diagnostics expander → **WS14**. The `database_failure` subset builds inside **WS1** first. **Reject**
  the full 11-event bus; instrument selectively (most bare-excepts are deliberate leakage-safe fail-opens). **M**.
- **P1 + StatsAPI doc + P5 delta + P6 columns: game-centric MLB gamePk+MLBAM warehouse** → **WS15**, additive
  phased per the doc's Phase 1-5, ESPN fallback preserved. **Adapt, tier 3, identity/reproducibility hardening
  only** (see §0b trap). **L**.
- **P21: pin `requests`** (currently fully unpinned, `requirements.txt:1`); keep bounded ranges → deploy-hardening
  backlog. **Defer** the `pip-compile --generate-hashes` lockfile to a separate deploy-validated step (uv on
  Streamlit Cloud is the unverified risk). **S** (pin) / **M** (lock).

### DEFER (right idea, wrong time / needs a trigger)
- **ER3 (P2b): gamePk game-roster candidate universe** → until **WS15** builds a gamePk→boxscore roster cache;
  then layer as a tiebreak. Don't add a live boxscore fetch to the resolution hot path standalone. Two-team hint
  already breaks the dominant case.
- **ER6 (P2 strategy §6-8): provider_alias table with validity windows** → backlog (lean reject). Over-engineering
  for a single-provider, DK-only app; `dk_name` alias + per-row `player_mlb_id` + `athlete_id_cache` already cover
  the realized value. Trigger: a real logged namesake mis-resolution OR a 2nd odds provider.
- **P6: uniform `source_provider`/`source_game_pk`** → bundle with **WS15** (origins are already mixed today —
  pitcher StatsAPI vs ESPN synth share `mlb_pitcher_gamelog`; harm low, clean fix rides the warehouse migration).
- **P8: temporal-type migration** → backlog (bulk). Forward-only rule: **new** timestamp columns (e.g. WS12
  provenance) are `DATETIME2` from birth. Never in-place swap the existing NVARCHAR/epoch-float columns (TTL math
  needs FLOAT; SchemaParity checks names only). Trigger: server-side T-SQL date analytics/retention pain.
- **P9: split god modules** → **WS16**, AFTER WS3/WS4/WS6 stabilize (they actively rewrite logic inside those
  exact files — splitting during behavior change is the worst time). Then ONE module (backtest.py first) via the
  proven `test_module_split.py` facade+`assertIs` template. (Auditor missed app.py=3506.)
- **P12: empirical pairwise correlation → shrink toward heuristic prior → copula** → backlog. Impact narrow under
  DK-only + SGP-neutralization (positive ρ never credited as EV; cross-game legs forced ρ=0); joint-outcome data
  thin. Trigger: warehouse growth. Keep SGP neutralization; never estimate from own parlays.
- **P14: edge-uncertainty interval / P(true edge>0) gate** → backlog. Depends on P15 N_eff; small-sample concern
  largely handled by market-prior shrinkage; must beat the current gate under forward validation before shipping.
- **P15 (shrinkage-denominator swap)** → gate-only, speculative, mildly anti-conservative (inverted math);
  route through the props gate with no "more correct" presumption. (The diagnostic half is adopt, above.)
- **P21 (hashed lockfile)** → separate deploy-validated step (see adapt note).

### REJECT (do-not-do sub-recommendations; the scoped keeper lives in ADAPT above)
- **P7 broad normalization** (FK tree across game/player/odds_event) → target tables don't exist; belongs to WS15.
- **P10 full frozen Prediction/PlayerProp/Team hierarchy** → fights `.get(default)`; no CI type-checker; big
  behavior-sensitive diff.
- **P19 full `model_run` schema** (hyperparameters/train-windows/per-run ROI+CLV) → over-built for this system.
- **P20 full 11-event telemetry bus** → over-engineered for a single-user Streamlit app.
- **P8 in-place type swap** of existing columns → churn>value, breaks TTL math, no type-parity guard.

### ALREADY-DONE (moot — see §0c for evidence)
- **ER1 (P2a)** event-scoped identity key · **ER2 (P2c)** team-evidence/fuzzy-candidate-gen/fail-closed binder ·
  **P16** feature framework (registry+gate) · **P5** warehouse-first local read (delta only is new) ·
  **P18** wager-level DK-vs-DK CLV (model panel only is new) · **P23** secrets architecture (rotation only remains).

---

## 3. NEW workstreams (WS10–WS16)

> Numbering note: several cluster memos each independently proposed "WS10". This section supersedes those with a
> single non-colliding scheme. Tier 1 = do-soon, 2 = next, 3 = someday/triggered.

| WS | Scope (one line) | Tier | Effort | Priorities | Sequence / dependencies |
|---|---|---|---|---|---|
| **WS10** | Event-level entity-resolution hardening: ER5 (thread team names) → ER4 (skip ambiguous MLB namesakes, counted) → ER7 (role tiebreak, optional) | **1** | ER5 S, ER4 M, ER7 S | P2 (ER4/ER5/ER7) | File-disjoint from the SQL/calibration churn of WS1–9; can start alongside Phase A. **ER5 must land before ER4** (else ER4 skips cases ESPN's team-id would resolve). ER3 deferred to WS15. |
| **WS11** | Units normalization: candidate dicts fraction-native; single ×100 presentation helper; golden-test guarded | **2** | M | P11 | **After WS2 lands** (WS2 edits the same numeric pipeline — do not run concurrently). No SQL migration. |
| **WS12** | Prediction provenance & versioning: git_sha + calibration_version + fit_basis per row (stamp-on-first-write, DATETIME2 columns); minimal row DTO (P10); lightweight experiment registry (P19) | **2** | M | P3, P8-forward, P10-min, P19 | Sits beside WS4. **Provenance BEFORE experiment-registry and BEFORE P18 per-version CLV.** Budget SchemaParity churn (schema.sql + db_store metadata + idempotent ALTER + writer + test). |
| **WS13** | Correlation-aware staking: per-bet Kelly haircut from `parlay._pair_correlation`, applied before the slate cap; haircut-only | **2** | M | P13 | File-disjoint (wagers/bet_selector/app staking). Independent; can go any time in Phase B/C. Reuse the one correlation source. |
| **WS14** | Ops telemetry (scoped): `ops_event` helper at key bare-except sites + per-run counters + Diagnostics expander | **2** | M | P20 | `database_failure` subset builds **inside WS1** (fail-loud); the rest after WS1. Selective, not blanket. |
| **WS15** | Game-centric MLB player-game warehouse (gamePk + MLBAM): schedule→gamePk→boxscore→normalize→UPSERT; additive columns; ESPN fallback; source_provider | **3** | L | P1, StatsAPI-doc, P5-delta, P6 | **After WS1–WS9**, gated on a concrete reproducibility/identity need — NOT an accuracy play. Regression-risk flag: most load-bearing data path. Unlocks ER3 (roster cache) + P6 columns. |
| **WS16** | Targeted post-reset module split: ONE module (backtest.py first) via `test_module_split.py` facade+assertIs template | **3** | M (single slice) / L (ambitious) | P9 | **After WS3/WS4/WS6 stabilize.** Never split a module while a reset WS changes its behavior. |

**Items folded into EXISTING workstreams (no new WS):**
- **WS1**: P7 FK (odds_line→odds_snapshot); P20 `database_failure` subset; P4 partial (the SQL-off guard that
  makes demotion safe); P5 "reads canonical DB" intent.
- **WS4**: P15 N_eff diagnostic surface; P18 aggregate CLV panel.
- **WS6**: P4 (offline cadence + demote in-request refit) — WS6 already owns this loop.
- **P16 backlog** (existing gate): P17 market-structure candidates.

---

## 4. Sequencing narrative (relative to WS1–WS9)

**Now / out-of-band:** **P23 rotate the key** (minutes, non-code). **P22 ruff warn-only** + **P21 pin requests**
are cheap CI/deploy hygiene that can land any time.

**Phase A (WS8→WS1, persistence):** attach **P7 FK** and the **P20 database_failure** telemetry subset to WS1
(both are integrity/fail-loud work in the same files). WS1's fail-loud guard is also the precondition that makes
**P4**'s demotion safe.

**Alongside Phase A/B (file-disjoint):** **WS10** (entity resolution) can begin immediately — it touches
gamelog_store/player_id_map/mlb_starters/espn_client/app-analyze, none of which the calibration/persistence
workstreams are rewriting. Highest-value new idea; start with ER5, then ER4.

**Phase B/C:** **WS11** (units) only AFTER WS2 (shared numeric pipeline). **WS13** (staking haircut) is
independent and can slot in Phase B. **WS4** absorbs **P15 N_eff diagnostic** + **P18 CLV panel** (WS4 is already
Phase C, after WS2). **WS14** (rest of ops telemetry) after WS1.

**Phase D+ / cross-cutting:** **WS12** (provenance) sits beside WS4; it must precede **P18 per-version CLV** and
the **P19 experiment registry**. **WS6** absorbs **P4** (offline cadence). The **forward-only DATETIME2 rule (P8)**
attaches to WS12's new columns.

**Tier 3 / triggered (after WS1–9):** **WS15** warehouse (gated on a reproducibility/identity need, not accuracy)
→ which then unlocks **ER3** roster universe and **P6** source columns. **WS16** module split (after WS3/4/6
stabilize).

**Dependency chain to remember:** WS2 → WS11 · WS12 provenance → {P18 per-version, P19 registry} ·
WS15 gamePk warehouse → {ER3 roster universe, P6 source_provider} · P15 N_eff → P14 edge interval ·
WS1 fail-loud → P4 demotion safe.

---

## 5. Philosophy checks (flags)

None of the keepers violate the stated philosophy. Explicit confirmations for the tempting ones:
- **P17 (market dispersion)** stays DK-only-safe: non-DK disagreement is a SIGNAL routed through the gate, never
  an executable/recommended price. Keep `executed_price` DK-only (CLV invariant depends on it).
- **P13/P12** keep SGP neutralization; the haircut only ever reduces exposure (fail-safe).
- **P15 swap / P14 gate / P17 features** all must beat the current gate under chronological forward validation —
  never ship on "more correct in theory."
- **WS10 ER4** is the fail-closed-identity principle finally enforced end-to-end; skipping an ambiguous prop
  slightly shrinks the slate (intentional: one lost opportunity ≪ one poisoned observation/recommendation).
- Every new WS is reversible pure-code; none touches `calibration/baseball_mlb.json` or the Brier gate math.
