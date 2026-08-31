# Workstream 6 — Champion-gated online Platt for the pitcher props

**Date:** 2026-08-07
**Scope:** DESIGN ONLY (read-only investigation). No code/DB/sweep/vendor changes made.
**Target props:** `pitcher_strikeouts`, `pitcher_outs`, `pitcher_earned_runs`.
**Goal:** give the three pitcher props a forward Platt correction with the SAME
seed-as-prior / champion-gate-against-seed safety batter_hits was designed to have
(auto-reverts if a loop fit fails to beat the committed seed), so they stop deploying
with zero forward correction.

All line:file references below were re-verified against the working tree on 2026-08-07
(commit 5c4549b era). Where the audit's cited lines had shifted, the current lines are given.

---

## 0. TL;DR

The runtime apply, the online refit loop, and the champion gate are **already fully
prop-generic** — none of them special-cases batter_hits. The reason pitchers get zero
forward correction is purely **data/config**, in three gaps, plus one adjacent live bug:

1. **Seed target list is incomplete.** `PROPS_BY_SPORT["mlb"]`
   (`book_line_calibration.py:52`) = `["pitcher_strikeouts", "batter_hits"]`. `pitcher_outs`
   and `pitcher_earned_runs` are **never seeded**; `pitcher_strikeouts` is targeted but
   produced no passing seed on 2026-07-30 (thin data / failed beat-raw).
2. **No committed pitcher seeds.** `calibration/recalibration_baseball_mlb.json` holds
   exactly one key: `batter_hits@le_0.5`.
3. **Safety hole for the no-seed case.** If a pitcher prop has no seed, the online loop
   (`refit_sport`) fits it with `incumbent=None` → **beat-raw only, no champion-gate-
   against-seed** → violates the task's "SAME safety." A naked beat-raw fit on thin,
   forward-degraded pitcher data is exactly what we want to prevent.
4. **ADJACENT LIVE BUG (verified): batter_hits' online recal is orphaned.** The committed
   seed key is `batter_hits@le_0.5`, but the shipped `batter_hits` cfg now has
   `line_methods=None` (method E was re-spliced in 5c4549b, dropping the old le_0.5
   bucket). `_resolve_recal_cfg` therefore looks up the bare `batter_hits` key, which does
   not exist in the map, and returns `None`. **batter_hits currently gets ZERO online Platt
   correction in production** — I confirmed this by running the real resolver against the
   real files (see §2). The task's premise "batter_hits has one" is nominal only.

The fix is small, reversible, and mostly re-uses the existing generic machinery:
- **A.** Add `pitcher_outs`, `pitcher_earned_runs` to `PROPS_BY_SPORT["mlb"]` so the seed
  bootstrap targets them.
- **B.** Add a `REQUIRE_SEED_PROPS` guard in `refit_sport` so pitcher props are
  **champion-gate-against-seed only** (no naked beat-raw fit): no seed → skip → fail-safe
  to today's zero correction.
- **C.** Add a `--dry-run` to the `--seed` CLI, run it, inspect whether each pitcher
  passes the CV gate, then commit the seeds that do. This run **also re-keys batter_hits to
  the bare key**, incidentally fixing bug #4.
- **D.** Fix #4 (batter_hits orphan) — happens for free via C's re-seed (preferred), or by
  re-keying the committed entry by hand.
- **E.** Land the code (A/B/C) any time, but hold the prod **seed commit + enablement**
  until Workstream 1 hardens the SQL-off save path (else a degraded-mode refit overwrites
  the freshly committed pitcher seed).

**Would pitchers hit the same wall?** Not the identical one — see §6. They get their
forward correction from the committed **seed** (like batter_hits was meant to), not from
the online loop. The online loop stays dormant for pitchers for a long time, but for a
DIFFERENT reason than batter_hits (thin forward data vs an unbeatable strong seed). The
real near-term risk is the **seed-production** wall (§6.A): pitcher_strikeouts already
failed to seed once; whether the grown warehouse now yields passing pitcher seeds is
data-dependent and must be checked with C's `--dry-run` before promising anything.

---

## 1. How the existing loop works (grounding, all re-verified)

**Runtime apply — prop-generic.** `props.py`:
- `:897` `maybe_auto_refit(sport_key)` — runs the online maintenance loop on first analysis
  per process/sport.
- `:898` `recalibration = load_recalibration(sport_key)` — loads the applied Platt map.
- `:1344` `recal_cfg = _resolve_recal_cfg(recalibration, prop_key, line, prop_calib_cfg)`
  and `:1359` `over_rate = _apply_final_recalibration(over_rate, recal_cfg)` — applies the
  Platt map to the final over-rate. Nothing here is batter-specific; **a bare-keyed
  `pitcher_*` fit in the map applies automatically.**
- Safe-mode path mirrors it at `:1436`/`:1438`.

**Key resolution — `props.py`:**
- `_composite_recal_key(prop_key, line, line_methods)` `:615` — no `line_methods` ⇒ returns
  the **bare** `prop_key` (pitchers have no line_methods ⇒ bare key).
- `_resolve_recal_cfg(recal_map, prop_key, line, prop_calib_cfg)` `:635` — no `line_methods`
  on the cfg ⇒ `return recal_map.get(prop_key)` (`:663`) ⇒ a bare `pitcher_*` map key
  applies exactly.

**Online loop — prop-generic.** `recalibration.refit_sport` `:2127`:
- `:2149` reads the pristine committed seed: `_, seed_props = _read_local_recal(sport_key)`.
- `:2170-2179` groups resolved `prediction_log` rows by `rec_key = _composite_recal_key(...)`
  (bare `pitcher_*` for pitcher props). All 5 MLB props are in
  `PLAYER_PROPS_BY_SPORT["baseball_mlb"]` (`odds_client.py:796`) so all are predicted,
  logged, and seen here.
- `:2183-2186` the champion gate:
  ```python
  seed_cfg = seed_props.get(fit_key)
  incumbent = (seed_cfg["a"], seed_cfg["b"]) if seed_cfg else None
  result = fit_platt_chronological(records, incumbent=incumbent)
  ```
  So a seed ⇒ champion-gate-against-seed; **no seed ⇒ `incumbent=None` ⇒ beat-raw only**
  (the safety hole, gap #3).
- `:2212-2214` persists via `save_recalibration(...)` (to_blob default True).

**Champion gate — `fit_platt_chronological` `:1796`:** two expanding chronological folds
(`:1823-1832`, date-ordered, ties stay same side ⇒ leakage-safe). With `incumbent` set it
additionally requires `len(rows) >= MIN_OBS_FOR_OVERRIDE` (`:1820`, =300) **and** beating
the seed on Brier AND log-loss in **every** fold (`:1851-1858`) — else returns `None`
(auto-revert). `incumbent=None` keeps the original beat-raw behavior (`:1849-1850`).

**Seed bootstrap — `seed_from_book_line_cache` `:2301`:** targets
`target_props = PROPS_BY_SPORT.get(args.sport, [])` (`recalibration.py:2516`), harvests real
book lines (`harvest_real_line_book_lines`), joins to actuals, re-derives the raw prob with
the **current** per-prop knobs + method (`_params_for` `:2365`, `_method_cfg_for_line`
`:2402`, warmup via `count_current_season_games` `:2415`), fits Platt per `rec_key` with
`incumbent=None` (`:2442`, the seed is being *born*), and writes local-only via
`save_recalibration(..., to_blob=False)` (`:2473`). This is the ONLY producer of committed
seeds.

**Constants — `recalibration.py`:** `MIN_FIT_SAMPLES=50` (`:50`),
`MIN_VALIDATION_SAMPLES=20` (`:51`) ⇒ base fit floor = 50+2·20 = 90 rows;
`MIN_OBS_FOR_OVERRIDE=300` (`:58`); `RECAL_SEED_TRUST=1.0` (`:62`);
`MIN_NEW_FOR_REFIT=25` (`:52`), `MIN_REFIT_INTERVAL_HOURS=12` (`:53`).

---

## 2. The batter_hits orphan — verified against live code

Ran the real resolver against the real files (read-only):

```
recal_map keys: ['batter_hits@le_0.5']
batter_hits         line=0.5/1.5/4.5/5.5 : recal_applied=False
pitcher_strikeouts  all lines            : recal_applied=False
pitcher_outs        all lines            : recal_applied=False
pitcher_earned_runs all lines            : recal_applied=False
```

`baseball_mlb.json` props: `batter_hits.method="E"`, `line_methods=None`,
`mean_scale≈1.0036`, `fit_basis="real_line"`, `n_obs=3262`; the three pitchers have
`line_methods=None`. Because batter_hits lost its `line_methods` in the E resplice while the
committed seed key stayed `batter_hits@le_0.5`, `_resolve_recal_cfg` returns `None` for
every batter_hits line. **No prop currently receives an online Platt correction in prod**
(pitchers because there is no seed; batter_hits because of the key mismatch).

Consequence for WS6: keying pitcher seeds **bare** (`pitcher_strikeouts`, `pitcher_outs`,
`pitcher_earned_runs`) is correct and will apply, because those cfgs have no `line_methods`.
And the pitcher seed run (§4.C) naturally re-keys batter_hits to bare `batter_hits` (since
`_composite_recal_key("batter_hits", line, None)` = `"batter_hits"`), fixing the orphan.

---

## 3. Design principles

- **Re-use the generic machinery; change data + one guard, not the gate math.** The gate
  (`fit_platt_chronological`) and apply (`_resolve_recal_cfg`) already do the right thing
  for any prop.
- **Champion-gate-against-seed is mandatory for pitchers.** A pitcher fit is deployed ONLY
  as a refinement of a committed seed prior; never as a naked beat-raw fit. No seed ⇒ no
  correction (fail-safe = today's behavior).
- **Smallest reversible change.** Each part is a few lines and revertible independently.
- **Leakage-safe by construction.** Only the post-hoc Platt layer is touched; the fit uses
  chronological folds + as-of warmup already in place. No new features.

---

## 4. Concrete changes

### Part A — extend the seed target list  *(data coverage)*

`book_line_calibration.py:52`:
```python
"mlb":  ["pitcher_strikeouts", "pitcher_outs", "pitcher_earned_runs", "batter_hits"],
```
- **Consumers** (grep-verified): only `book_line_calibration.py:1675` (its own offline CLI
  harness) and `recalibration.py:2516` (the `--seed`/`--refit` CLI). Neither is on the app
  hot path. `PLAYER_PROPS_BY_SPORT` (the analysis/display list, `odds_client.py:793`) is a
  DIFFERENT constant and already lists all 5 props — unchanged.
- No test pins `PROPS_BY_SPORT["mlb"]` (grep-verified) ⇒ Part A breaks nothing.
- `batter_strikeouts` deliberately left out (out of scope; not negbin-eligible; dormant).
- **Reversible:** revert the list.
- *Alternative (more surgical, if the owner dislikes editing a shared constant):* add a
  `--props a,b,c` override to the `--seed` CLI (`recalibration._main_cli`) and pass it as
  `target_props`. Recommended primary = the constant edit (the list is genuinely
  incomplete); the `--props` override is a fine fallback.

### Part B — safety guard: pitcher props are seed-gated only  *(the "SAME safety")*

`recalibration.py`, near the constants (~`:58`):
```python
# Props whose online Platt loop may deploy ONLY as a champion-gated refinement of a
# committed seed prior — never as a naked beat-raw fit. Thin, forward-degraded pitcher
# data makes an un-anchored loop fit risky, so no seed ⇒ no correction (fail-safe to base
# calibration). batter_hits/NBA/NFL are unaffected (batter_hits has a seed; others absent).
REQUIRE_SEED_PROPS = {"pitcher_strikeouts", "pitcher_outs", "pitcher_earned_runs"}
```

`refit_sport` loop, at `:2183-2186`:
```python
for fit_key, records in by_prop_records.items():
    seed_cfg = seed_props.get(fit_key)
    if seed_cfg is None and fit_key.split("@", 1)[0] in REQUIRE_SEED_PROPS:
        continue  # champion-gate-against-seed mandatory; no naked beat-raw pitcher fit
    incumbent = (seed_cfg["a"], seed_cfg["b"]) if seed_cfg else None
    result = fit_platt_chronological(records, incumbent=incumbent)
    ...
```
- Only affects the **online loop** (`refit_sport`). The seed **producer**
  (`seed_from_book_line_cache`) is intentionally NOT guarded — that is where the seed is
  born with `incumbent=None`.
- With a committed pitcher seed ⇒ existing champion-gate path (unchanged math). Without ⇒
  skip. batter_hits (has seed once re-keyed), NBA/NFL (not in the set) unchanged.
- **Reversible:** delete the constant + the two lines.

### Part C — produce & commit pitcher seeds, gated by a dry-run  *(ops + tiny CLI add)*

Today `--seed` always writes the committed local file (a mutation with no preview). Add a
no-write preview so the implementer can see whether pitchers pass the gate before
committing.

`recalibration._main_cli` (`:2491`+): add `p.add_argument("--dry-run", action="store_true")`
and thread it into `seed_from_book_line_cache(..., dry_run=args.dry_run)`; in that function
guard the final save:
```python
if per_prop_params and not dry_run:
    save_recalibration(sport_key, per_prop_params,
                       meta={"source": "book_line_cache_seed"}, to_blob=False)
    _LOAD_CACHE.pop(sport_key, None)
```
(and always print the fits, as it does now). Default `dry_run=False` ⇒ behavior unchanged
when the flag is absent.

**Ops sequence (implementer, after A+B land and WS1 is done):**
1. `python recalibration.py --seed --sport mlb --dry-run` → inspect each pitcher's
   `(a, b, n_fit, holdout_raw_brier, holdout_calibrated_brier)`. Sane = `a<1` shrinking
   over-confidence, calibrated Brier < raw Brier, `n_fit ≥ 90`.
2. If ≥1 pitcher passes, run without `--dry-run` to write the local seed, then **git-diff**
   `calibration/recalibration_baseball_mlb.json`:
   - New bare `pitcher_*` entries appear for passers.
   - `batter_hits@le_0.5` is **replaced** by a bare `batter_hits` entry (the re-key — see D).
     Validate the new batter_hits `(a,b)` is sane before committing (it starts being applied,
     where the orphaned one was inert).
3. Commit the updated seed file.

### Part D — fix the batter_hits orphan  *(adjacent; free via C)*

- **D1 (preferred):** Part C's re-seed writes a bare `batter_hits` key that matches the
  current line_methods-less cfg ⇒ orphan fixed, no code change. Validate the fit in the diff.
- **D2 (only if the exact a=0.375 must be preserved and no re-seed is wanted):** hand-rename
  the committed key `batter_hits@le_0.5` → `batter_hits`. Do NOT instead broaden
  `_resolve_recal_cfg` to fall back from a composite-only map to the bare key — that changes
  its intentional "line_methods prop, bucket has no fit ⇒ return None" semantics (`:659-662`)
  and is a wider-blast, riskier edit. Prefer D1/D2 over a resolver change.

The owner may route D to a dedicated workstream, but since it is a **free side effect of the
required pitcher seed run**, treating it in-scope here is natural and avoids shipping
pitchers a "same as batter_hits" correction that is itself broken.

### Part E — Workstream 1 dependency (do before the prod seed commit)

`save_recalibration` guard (`:1926`) is `if not (to_blob and _sql()):` → when SQL is OFF, a
**runtime** refit (`to_blob=True`) still writes the **local committed seed**. That would
(a) overwrite/evict the freshly committed pitcher seed and (b) make the champion-gate
incumbent **self-referential** (it re-reads the loop's own last output as the "seed"),
defeating the whole safety. This is the open follow-up in `[[online-platt-seed-prior]]`.
WS1 must, before pitcher seeds go live in prod:
- Fail loudly (or no-op durably) when `_sql()` is False in a production refit context; and/or
  make the save guard mode-agnostic (`if not to_blob:` per the follow-up) so a runtime refit
  never rewrites the local seed.
- Mirror the per-key merge into the Blob read branch (`_load_recal_cached`, ~`:2103`,
  `status=="ok"`) so a first pitcher SQL/blob fit can't evict the rest of the committed seed
  (the old all-or-nothing bug, still unfixed on the blob path).

WS6 code (A/B/C) can LAND independently; only the **prod seed commit + enablement** waits on
WS1. (In current prod `_sql()` is True, so the immediate risk is a degraded/failover run.)

---

## 5. Leakage & correctness review

- **No leakage introduced.** Fits use `fit_platt_chronological`'s expanding chronological
  folds (`:1823-1832`); the seed run's raw prob uses as-of `prior_games` + current-season
  warmup (`count_current_season_games` `:2415`) exactly like batter_hits. The applied layer
  is a post-hoc 2-param Platt on the final over-rate.
- **Fit/apply key symmetry preserved.** Both sides derive the key from
  `_composite_recal_key` / `_resolve_line_bucket`; pitchers (no line_methods) ⇒ bare key on
  both sides. No collision (a real prop_key never contains `@`).
- **Fail-safe on every path.** No seed + require-seed ⇒ no fit (base calibration). Seed but
  loop can't beat it ⇒ gate returns None ⇒ seed stays. Malformed/absent map ⇒
  `_apply_final_recalibration` returns the probability unchanged (`:1348`).
- **Reversibility.** A: revert list. B: delete constant+2 lines. C: revert CLI + `git
  checkout` the seed JSON. D1: `git checkout` the seed JSON. All independently revertible;
  none touches `calibration/baseball_mlb.json` (the method/knob calibration).

---

## 6. Would pitchers hit the same wall? (explicit answer)

The open follow-up's wall is *"the loop never persisted a fresh SQL fit; the seed still
drives prod."* For batter_hits the mechanism is a **strong seed** (n=2144 book-line fit) the
live self-fit cannot beat OOS (dry-run 2026-07-30: seed wins Brier + log-loss in both
folds) ⇒ champion gate rejects forever. Pitchers face **different** walls:

- **A. Seed-production wall (the real near-term risk).** A pitcher gets a committed seed only
  if `seed_from_book_line_cache` produces a passing `fit_platt_chronological` (≥90 usable
  (raw,y) pairs after join AND beat-raw in both chronological folds). pitcher_strikeouts was
  targeted on 2026-07-30 and produced **nothing** (too thin then, or failed beat-raw);
  pitcher_outs/earned_runs were never even targeted. Real-line ceilings are ~252/359/419
  (SQL-derived in the synthesis, < the 500 real-line-refit gate). Whether the grown
  warehouse (≈4,253 lines) now yields passing pitcher seeds is **data-dependent and unknown
  from code** — it MUST be checked with Part C's `--dry-run`. If a pitcher still can't
  produce a seed, it stays at zero correction, and the require-seed guard (Part B) makes that
  **explicit and safe** instead of silently deploying a naked beat-raw fit.

- **B. Online-override wall (expected, and fine).** Even with a committed seed, the loop
  needs `MIN_OBS_FOR_OVERRIDE=300` resolved `prediction_log` rows **per pitcher prop** AND to
  beat the seed in both folds before it overrides. Pitcher forward rows accrue slowly (far
  fewer pitcher props/day than batter props), so the loop will stay dormant for pitchers for
  months. That is acceptable: **the committed seed IS the forward correction** in the interim
  (this is the seed-as-prior design working as intended, identical to batter_hits).

- **C. The original "loop never persists" wall.** Once a pitcher seed is committed and Wall B
  clears, the loop CAN persist a champion-gated override (in SQL, subject to WS1). So pitchers
  eventually escape it — but the immediate forward correction comes from the **seed**, not
  the loop.

**Bottom line:** Pitchers do NOT hit batter_hits' "unbeatable strong seed" wall. They get
their forward correction from the committed seed the moment Part C succeeds, and the online
self-learning loop remains dormant for them for a long time (thin data), which is expected
and safe. The one thing that can block pitchers entirely is **Wall A** — the seed simply not
fitting on current data — which is why Part C's dry-run is the go/no-go gate for the whole
workstream and its outcome cannot be promised from code.

---

## 7. Tests

### Existing pins (impact)
- `test_recal_blend.py` (14) — champion gate + blend math. **Unaffected** (B doesn't change
  gate math; A/C don't touch it).
- `test_recal_buckets.py::test_refit_flat_prop_unchanged` (`:191`) — uses
  `pitcher_strikeouts` with **no seed mock** (it reads the real committed seed, which has no
  pitcher key ⇒ `incumbent=None` ⇒ beat-raw ⇒ asserts `"pitcher_strikeouts" in params`).
  **This BREAKS under Part B** (guard skips the seedless pitcher fit). It is also fragile
  w.r.t. the committed seed file. **Update it:** either (a) switch its prop to a
  non-require-seed flat prop (e.g. an NBA `player_points`) to preserve the "flat prop
  unchanged" intent, or (b) mock `_read_local_recal` to return a weak `pitcher_strikeouts`
  seed so the champion-gate path yields the fit. Recommend (a) + add the new pitcher tests
  below. Either way, **mock `_read_local_recal` for determinism.**
- `test_recal_buckets.py::test_refit_buckets_by_composite_key` (`:163`) — already mocks
  `_read_local_recal=(None,{})` and uses batter_hits; unaffected.
- `test_recalibration_durability.py::test_seed_save_stays_local_only` (`:137`) — Part C's
  `--dry-run` must not regress it (dry-run only skips the save).

### New tests
1. **`test_refit_pitcher_requires_seed`** — `refit_sport` with ample `pitcher_outs` log rows
   + empty seed (`_read_local_recal=(None,{})`) ⇒ `"pitcher_outs" not in params` (guard
   fires). Contrast a non-require-seed flat prop in the same log still produces a fit.
2. **`test_refit_pitcher_with_seed_champion_gates`** — same rows + a weak seed
   (`{"pitcher_outs": {"a":1.0,"b":0.0,"n_fit":300}}`) ⇒ fit produced (beats identity
   seed); with a strong seed (the all-rows fit) ⇒ no override (reuses
   `test_gate_strong_incumbent_rejects` logic).
3. **`test_resolve_recal_pitcher_bare_key_applies`** —
   `props._resolve_recal_cfg({"pitcher_outs": fit}, "pitcher_outs", 4.5, {})` returns `fit`
   (bare-key pitcher apply works; extends `test_recal_buckets.py`).
4. **`test_seed_dry_run_writes_nothing`** — `seed_from_book_line_cache(..., dry_run=True)`
   never calls `save_recalibration` (mock + assert not called).
5. **`test_resolve_recal_batter_hits_bare_after_rekey`** (regression for the orphan) — with
   a bare `{"batter_hits": fit}` seed and a line_methods-less cfg, recal applies for all
   lines; documents the D1 fix and guards against re-introducing the `@le_0.5` mismatch.
6. **`test_props_by_sport_mlb_covers_pitchers`** — assert the 3 pitcher props + batter_hits
   are in `PROPS_BY_SPORT["mlb"]` (pins Part A).

---

## 8. Open questions / cannot-determine-from-code

- **Do pitchers actually seed on current data?** Unknown from code (Wall A). The `--dry-run`
  (Part C) is the empirical go/no-go and cannot be run in this read-only design pass.
- **Is the batter_hits orphan fix in WS6 or a separate workstream?** It is a free side
  effect of the required pitcher seed run; recommend handling here but flag for the owner.
- **Should the require-seed set also include `batter_strikeouts`?** It is analyzed
  (`PLAYER_PROPS_BY_SPORT`) but never seeded (`PROPS_BY_SPORT`), so today it too could get a
  naked beat-raw loop fit. Out of WS6 scope, but worth a follow-up decision.
