# Workstream 4 — Like-for-Like Forward vs Backtest Brier

**Status:** DESIGN ONLY (read-only pass). Reversible, additive, no-write.
**Depends on:** Workstream 2 (offline==online projection parity) landing first — see §7.
**Author:** WS4 design agent, 2026-08-07. All line numbers verified against the working tree this pass.

---

## 1. Objective

Forward Brier and backtest ("Chronological"/fit) Brier are today **different objects**, so the
gap between them (synthesis §1.2-B "metric-definition asymmetry") is not attributable to model
behavior. Produce an **aligned scoring path** where forward and backtest are the *same probability
object, same label derivation, same row subset*, so any residual gap is real OOS degradation.
Deliver it as a **no-write diagnostic** (mirrors the existing `diagnose_*` lenses), plus one tiny
additive field on the forward summary. Nothing that mutates calibration or the DB.

---

## 2. The three asymmetries (grounded)

### 2.1 Probability object
- **Forward** scores `final_prob` (fallback `raw_prob`): `recalibration.summarize_prediction_rows`,
  `recalibration.py:780-783` (pick `final_prob` else `raw_prob`) → Brier at `:818-819`.
  `final_prob` = deployed method **+ Platt recal + market-prior blend**.
- **Backtest** scores the **bare method** A/B/C/D/E (no Platt, no blend):
  `book_line_calibration._score_abc_real:1143-1164` (real-line path) and
  `backtest._score_calibration_methods:3099-3128` (synthetic sweep path).

**Enabling fact (the crux):** production already logs BOTH objects.
`props.py:1340` snapshots `raw_over_rate = over_rate` (the bare deployed-method P(over) at the
real line — *before* Platt at `:1359` and *before* the market-prior blend at `:1575-1581`), and
`props.py:1655-1656` stores it as `raw_prob`, with the post-Platt+blend `over_rate` as `final_prob`.
Both are **P(over)**, not picked-side (confirmed: `log_prediction(..., raw_prob=raw_over_rate,
final_prob=over_rate)`; the schema outcome is over-based). So **the pre-Platt object is already on
every forward row** — aligning on it costs a read, not a re-computation.

### 2.2 Label derivation
- **Forward:** stored `outcome`, set once at resolution: `outcome = 1 if actual > line else 0`,
  push (`actual == line`) → `None`/dropped (`recalibration.py:1639-1642`). Scored via
  `outcomes.append(int(row["outcome"]))` over `outcome in (0,1)` (`:773`, `:786-788`).
- **Backtest:** RE-DERIVES from the gamelog each run: `out.append(1 if r["actual"] > r["line"]
  else 0)` (`book_line_calibration.py:1144`), pushes excluded via `usable = [r ... if
  r["actual"] != r["line"]]` (`:1296`). Synthetic path identical (`backtest.py:3101,3104`).

**These are the SAME formula** (`1[actual>line]`, pushes dropped). The only divergence is the
data source: forward's `actual` was resolved once (statsapi hard-ID → ESPN fallback,
`resolve_one_prop`); backtest's `actual` is re-joined fresh from the ESPN gamelog in
`join_book_lines_to_actuals:439-444`. The aligned path uses the **re-derived** label as the single
source of truth and reports a forward-stored-vs-re-derived agreement rate as a data-integrity check.

### 2.3 Row subset
- **Forward:** ALL resolved graded rows, every line, every player (`summarize_prediction_rows`
  dedupes by identity, then scores all `outcome in (0,1)`). No prior-games / minutes filter at
  scoring; doubleheaders are *disambiguated* (`_pick_candidate`), not dropped.
- **Backtest (real-line):** curated. `join_book_lines_to_actuals` drops: player-not-found /
  no athlete_id (`:371,377`), cross-role gamelog (`:408-410`), doubleheader dates (`:433-435`;
  also dropped upstream in `harvest_real_line_book_lines:290-297`), `<10 min` played (`:446-448`),
  `<10 prior games` (`:451-453`). Plus pushes (`select_method_at_real_lines:1296`).

The subset is the hard asymmetry: forward rows carry no prior-games count or minutes, so they
cannot be filtered post-hoc. The design resolves this by computing BOTH numbers on the
**intersection** — the curated real-line obs that ALSO have a matching resolved forward row (§4).

---

## 3. Where each Brier lives (map)

| Number | Producer | Consumer / display |
|---|---|---|
| Forward Brier (post-Platt) | `recalibration.summarize_prediction_rows:818` | `app.py:1213-1215` "Probability Brier"; per-prop `:1241-1243` |
| fit_brier ("Chronological") | `refit_calibration._best_per_prop` / real-line `select_method_at_real_lines` → JSON | `app.py:1124-1127` "Chronological Brier" |
| Backtest bare-method Brier (real line) | `book_line_calibration._score_abc_real:1156-1164` | `select_method_at_real_lines`, `diagnose_*` |
| Backtest bare-method Brier (synthetic) | `backtest._score_calibration_methods:3106-3128` | sweep `_best_per_prop` |

`forward_tracker.py` only orchestrates resolve+refit (`resolve_and_refit`, `:35`); it computes no
Brier. `db_store.py` is pure I/O (no Brier). So the forward number is produced entirely in
`recalibration.py` and displayed in `app.py`.

---

## 4. Design

Two layered, independently-reversible parts.

### Part 1 — Forward-side pre-Platt Brier (tiny, additive)

Add a **parallel pre-Platt Brier** to `summarize_prediction_rows` so the forward number can be
displayed on the *same probability object* as the bare-method "Chronological Brier".

- In `metrics()` (`recalibration.py:771-836`), inside the existing `for row in graded:` block that
  already appends to `probabilities`/`outcomes` (guarded by the valid-probability check at
  `:786-788`), also append `raw_prob` (fallback to the already-computed `final`/`probability`):
  build a `raw_probabilities` list over the **identical row subset** so raw-vs-final is directly
  comparable.
- Compute `brier_raw = mean((p-y)^2)` and add key `"probability_brier_raw"` to the returned dict
  (alongside `"probability_brier"` at `:830`). Add it to each `by_prop` entry too (it flows through
  the same `metrics(group)` call at `:844`).
- **No existing key changes.** `probability_brier` stays post-Platt. Purely additive.

Optional display (`app.py`): a second metric/column "Probability Brier (pre-Platt)" next to the
existing one, and a caption noting it is the like-for-like partner of the bare-method
"Chronological Brier" (still a *different subset* — the true subset match is Part 2).

**Value:** removes asymmetry 2.1 at the dashboard level for free; also exposes the Platt effect
(`probability_brier` − `probability_brier_raw`) on live data.

### Part 2 — Aligned lens `diagnose_forward_parity` (the real deliverable)

A new **no-write** function in `refit_calibration.py` + a `--forward-parity-diag` CLI flag, modeled
byte-for-byte on `diagnose_negbin` (`refit_calibration.py:1281-1392`) and its dispatch
(`:2743-2745`). It scores all objects on **one shared subset, one label, one holdout**.

**Build (reuse existing primitives — zero new join/label code):**
1. `book_lines,_,_ = blc.harvest_real_line_book_lines(sport_key, props, store_label)` — already
   unions the resolved prediction log and drops doubleheaders.
2. `enriched = blc.join_book_lines_to_actuals(book_lines, espn_sport, espn_league)` — this IS the
   curated subset (no-id / <10min / <10 prior games / cross-role dropped) and re-derives `actual`.
3. Build a **forward index** from resolved prediction-log rows:
   `recalibration._read_log(where={"sport_key":sport_key,"resolved":True})`, keyed by
   `blc._book_line_key(r)` = `(player, prop_key, game_date, round(line,1))` with the id-key
   `blc._book_line_id_key(r)` as fallback — the SAME keys harvest already dedups on. Use the same
   ET-date derivation harvest uses (`gd = et_local_date(commence) or game_date[:10]`,
   `book_line_calibration.py:217`) so keys line up. Store `{raw_prob, final_prob, outcome}` per key.
4. Per prop: `rows = blc.build_real_line_obs(enriched, params, sport_key, prop_key, team_defense,
   league_avg_def, xstats_strength=0.0)` using the SHIPPED cfg's params (mirror `diagnose_negbin`
   `:1339-1350`). **NOTE:** `build_real_line_obs` (`:1076-1088`) currently drops the identity
   fields. Either (a) minimal additive change — also carry `event_id`/`player_mlb_id`/`player`
   through the row dict, or (b) join the forward index at the `enriched` level (each `enriched`
   obs retains `player`/`player_mlb_id`/`prop_key`/`line`/`game_date` via the `{**row,...}` spread
   at `join_book_lines_to_actuals:456`) and carry a stable key onto each `rows` entry. Prefer (b):
   compute `_book_line_key`/`_book_line_id_key` from the enriched obs and stamp `row["_fwd_key"]`
   / `row["_fwd_idkey"]` in `build_real_line_obs` (additive, no behavior change to existing callers).

**Score (shared subset S, shared holdout):**
- `usable = [r for r in rows if r.actual != r.line and (fwd_index has r._fwd_idkey or r._fwd_key)]`
  → S = curated obs that also have a resolved forward row. Drop pushes.
- Sort S by `game_date`; single chronological holdout `split = len(S)//2`, `test = S[split:]`
  (mirrors `select_method_at_real_lines:1303-1305` so the number equals the ship-path single_split).
- `scores,_,probs,out = blc._score_abc_real(S[:split], test, negbin_eligible)` → bare-method
  P(over) per test row for the DEPLOYED method (`cfg["method"]`); `label = out` (re-derived).
- For each test row, look up `raw_prob`/`final_prob`/`outcome` from the forward index by
  `_fwd_idkey`||`_fwd_key`.
- Report per prop, all on `test` (identical rows + identical `label` vector):
  - `backtest_bare` = `scores[method]` (bare offline method).
  - `forward_raw` = Brier(`raw_prob`, label) — pre-Platt production object.
  - `forward_final` = Brier(`final_prob`, label) — post-Platt production object.
  - `label_agreement` = fraction where forward stored `outcome` == re-derived `label`
    (data-integrity; expect ~1.0).
  - context (unaligned, for the reader): `forward_all` = the existing all-rows post-Platt Brier
    from `summarize_prediction_rows`, and `fit_brier` from the JSON cfg.

**Interpretation (the decomposition the task wants):**
- `forward_raw − backtest_bare` = **projection-parity residual** (offline vs online). ~0 once WS2
  lands; nonzero flags remaining combined_mult skew or actual/label disagreement. THIS lens is the
  instrument that verifies WS2 closed the gap.
- `forward_final − forward_raw` = **Platt + market-blend effect** (same rows, same label).
- `forward_all − forward_final` = **subset/selection effect** (all lines vs curated).
- `backtest_bare − fit_brier` = **selection optimism** (shared OOS subset vs the 576-variant
  argmax single-holdout headline; synthesis §1.2-A).

**Optional "Platt-adjusted in both" column:** apply the deployed recal map to `backtest_bare`
probs via `props._resolve_recal_cfg` + `recalibration.apply_platt` per test row →
`backtest_platt`, directly comparable to `forward_final`. Completes the task's "Platt-adjusted in
both" alignment. Keep behind the same lens, clearly labeled; it adds a read of the recal JSON only.

**CLI:** add `p.add_argument("--forward-parity-diag", action="store_true", ...)` near
`refit_calibration.py:2670`, and a dispatch block `if args.forward_parity_diag:
diagnose_forward_parity(args.sport, store_label=args.store_label); return` mirroring `:2743-2745`.

---

## 5. Join-key / subset precision

- Key = `_book_line_key` = `(player, prop_key, game_date, round(line,1))` (`blc:239-242`), id-key
  `_book_line_id_key` = `(player_mlb_id, prop_key, game_date, round(line,1))` (`blc:245-255`).
- Both harvest sources already dedup on these keys, so an obs and its forward row share the key by
  construction when the obs is prediction-log-sourced. Warehouse-sourced obs that also match a
  forward row join too; obs with no forward match are simply excluded from S (fail-safe) and
  counted (`n_curated` vs `n_shared` reported so coverage is visible).
- ET-date consistency: use `et_local_date(commence) or game_date[:10]` on the forward-index side
  (matches `harvest_book_lines_from_prediction_log:217`); `enriched` game_date is already ET.

---

## 6. Leakage safety

- Part 1: pure re-read of already-logged probabilities against already-stored outcomes. No fit,
  no new data. Leakage-free.
- Part 2: reuses `_score_abc_real` with the SAME chronological single-holdout the ship path uses
  (train strictly earlier half, score later half by `game_date` sort). `raw_prob`/`final_prob` are
  production values logged **before** each game (forward-tracked) — inherently OOS. The re-derived
  label reads only the test game's box score (never fed to the train fit). D's `p_dist` is
  split-independent (as-of, `_score_abc_real:1128-1129`). No write, no calibration mutation.
- Do NOT let the forward index leak into the bare-method fit — it is only a post-hoc lookup on the
  test rows.

---

## 7. Dependency on Workstream 2 (parity)

`forward_raw` uses the **production** projection (with `combined_mult`: matchup/lineup/weather/park,
`props.py:1251-1258`); `backtest_bare` uses the **offline** projection (`book_line_calibration
.project_and_empirical`; park added 8e76d86, matchup/lineup/weather still omitted per synthesis
§4.1). Until WS2 reconstructs (or disables) those offline, `forward_raw − backtest_bare` conflates
projection skew with model behavior. The lens is **buildable now** (it just reports the gap), but
its headline claim ("residual gap = model behavior") only holds after WS2. Sequence: WS2 lands →
run this lens → confirm `forward_raw ≈ backtest_bare`. Flag this prominently in the lens output.

---

## 8. Reversibility

- Part 1: one additive dict key + one additive loop append; delete to revert. No schema/DB change.
- Part 2: one new function + one CLI flag + one additive stamp in `build_real_line_obs`
  (`_fwd_key`/`_fwd_idkey`, ignored by every existing caller). No write path touched. Fully
  removable. Zero effect on the sweep, `--real-lines`, or shipped calibration.

---

## 9. Tests

**Existing pins (must stay green):**
- `test_modeling.py:1150-1198 test_forward_summary_deduplicates_and_scores_direction` — pins
  `probability_brier` = 0.05 (final_prob-first fallback). Part 1 must NOT change this; add a new
  assertion `probability_brier_raw ≈ 0.065` (raw_prob 0.8→out1 =0.04, 0.3→out0 =0.09, mean 0.065).
- `test_realline_calibration.py` — pins `select_method_at_real_lines`, `harvest_real_line_book_lines`
  (incl. doubleheader drop `:532`, dedup `:475/543`), `_real_line_folds`. Confirms the primitives
  the lens reuses; unchanged.
- `test_prediction_log.py` — pins raw/final_prob persistence + collapse. Unchanged.

**New tests:**
- `test_modeling.py`: extend the forward-summary test for `probability_brier_raw` (identical
  subset as final); add a case where a row has `raw_prob` but no `final_prob` to prove raw is
  scored over the same graded rows.
- New `test_forward_parity.py` (mirror `test_negbin.py`/`test_roi_diag.py` — monkeypatch
  `harvest_real_line_book_lines` / `join_book_lines_to_actuals` to a fixed fixture; patch
  `recalibration._read_log` to a matching forward log):
  1. Shared-subset selection: only curated obs with a matching resolved forward row enter S;
     pushes excluded; `n_shared ≤ n_curated`.
  2. Identical label vector: `backtest_bare`, `forward_raw`, `forward_final` scored over the
     same `test` rows and the same re-derived `label`.
  3. `label_agreement == 1.0` when stored outcome matches `1[actual>line]`; < 1.0 when a stored
     outcome is stale (integrity signal fires).
  4. When `raw_prob == backtest_bare`'s prob and Platt is identity, `forward_raw == backtest_bare`
     (parity assertion — the WS2 acceptance test).
  5. No-write: assert the calibration JSON and DB are untouched (no mutate calls).
  6. Empty / <20-usable → graceful "nothing to diagnose" (mirror `diagnose_negbin:1354-1357`).

---

## 10. Files touched (targets)

- `recalibration.py:771-836` — add `raw_probabilities` + `probability_brier_raw` in `metrics()`.
- `refit_calibration.py` — new `diagnose_forward_parity()` (model on `:1281`); CLI arg near
  `:2670` + dispatch near `:2743`.
- `book_line_calibration.py:1076-1088` — additive `_fwd_key`/`_fwd_idkey` stamp in
  `build_real_line_obs` (identity carry-through for the forward join).
- `app.py:1213-1243` (optional) — display pre-Platt forward Brier column/metric + caption.
- `test_modeling.py` (extend), `test_forward_parity.py` (new).

---

## 11. Risks / open questions

- **Subset coverage:** warehouse-sourced obs may not all have a forward row → S can be a fraction
  of the curated set for pitcher props (thin). Report `n_shared`; if too small (<20), the lens
  should say so rather than print a noisy number (reuse the `_real_line_folds`/min-20 guard).
- **ET-date/line-rounding key mismatch** between the forward index and enriched obs would silently
  shrink S. Mitigated by reusing the exact harvest keys + ET derivation; the `n_shared` vs
  `n_curated` counts make any mismatch visible. Add a test that a known row keys through.
- **Label integrity, not correctness:** `label_agreement < 1.0` surfaces stale/mis-sourced stored
  outcomes but this lens does not fix them (that is WS3/join territory) — it only reports.
- **Pre-WS2 headline caveat** (§7): do not interpret `forward_raw − backtest_bare` as pure model
  behavior until parity lands.
- **market-prior blend** is folded into `final_prob` alongside Platt; the lens reports their
  combined effect as `forward_final − forward_raw`. Splitting them would need `raw_prob` to also
  snapshot the post-Platt/pre-blend value, which production does not log — out of scope; note it.
