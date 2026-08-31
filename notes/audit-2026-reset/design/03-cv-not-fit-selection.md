# Workstream 3 — Select & report on `cv_brier` (2-fold), not single-holdout `fit_brier`

**Date:** 2026-08-07 · **Status:** DESIGN ONLY (read-only pass; no code/DB/sweep run)
**Scope:** `refit_calibration._best_per_prop` (the 576-variant synthetic sweep selector) + reporting.
**Anchor lines re-verified current (2026-08-07):** `MIN_CALIB_BRIER_GAIN = 0.002` `refit_calibration.py:39`; `_cv_brier` `:496`; `_best_per_prop` `:554`; grid `backtest._build_props_sweep_grid` `:1946` (comment "576" `:1965`); single 50/50 holdout `backtest.py:3053-3059`. Audit's cited numbers all still match.

---

## 1. Objective

`fit_brier` is the **argmax winner of a 576-variant grid on ONE 50/50 chronological holdout**, used as **both the selector and the advertised headline**. That is winner's-curse selection: the winning (variant × method) is the one that got lucky on that single split. The less-biased 2-fold `cv_brier` already exists (`_cv_brier` `:496`) and regresses it (JSON: pitcher_outs `fit 0.2201 → cv 0.2256`; pitcher_earned_runs `fit 0.2371 → cv 0.2362`; batter_hits synthetic `fit 0.2287 → cv 0.2301`).

Change:
1. **SELECTION** in `_best_per_prop` uses `cv_brier` (2-fold) as the argmax key, with a safe single-split fallback when folds can't form.
2. **REPORTING** advertises the less-optimistic cv figure as the headline (app "Chronological Brier" column), plus a cv-basis baseline (`baseline_cv_brier`), while **keeping** `fit_brier`/`baseline_brier` as honest single-split provenance.
3. **Confirmation-gate semantics preserved verbatim**: `MIN_CALIB_BRIER_GAIN = 0.002` margin + 2-fold `_confirms_over_baseline` + `_variant_confirms`. The gates decide ELIGIBILITY; cv only changes the argmax among the already-eligible candidates. This can only ever move a pick toward the safe baseline A — it never admits a method the gates rejected.

Applies uniformly to every prop in the synthetic sweep (naturally covers the tiny-n synthetic pitcher props the task targets; batter_hits' synthetic pick is subsequently overridden by the real-line path — see §6).

---

## 2. Current mechanics (grounded)

- `_best_per_prop` `:554-614`: for each prop, loops all 576 variants; per variant calls `_evaluate_calibration_methods(obs, k_values, holdout=True)` (single 50/50 split, `backtest.py:3053-3059`), keeps A/B/C. Non-A must clear the `0.002` margin vs single-split A **and** `_confirms_over_baseline(obs, method)` (2-fold). Non-baseline variants must also clear `_variant_confirms(obs, base_obs, method)` (2-fold vs baseline variant). **Winner = argmin of `e["brier"]` (single split)** via `if best is None or e["brier"] < best["brier"]` `:600`. `cv_brier` is computed via `_cv_brier(obs, method)` **only for the eventual winner** `:608`.
- `_cv_brier(obs, method)` `:496-508` and `_confirms_over_baseline(obs, method)` `:477-493` both call `_method_brier_by_fold(obs, method)` `:466`, which **ignores its `method` arg** and returns a per-fold `{method: eval}` list (method-agnostic). Folds = `_chronological_folds` `:438` (cuts 60%/80%, each fold ≥20, needs n≥60; strictly-earlier train — leakage-safe).
- `_build_prop_cfg` `:617-659` persists `fit_brier` (`round(winner["brier"],4)` `:640`), `baseline_brier` `:645`, `cv_brier` `:646`.
- App display: `app.py:1124-1127` shows `cfg['fit_brier']` as **"Chronological Brier"** (the headline). NFL/NBA legacy props carry `fit_brier` but **no `cv_brier`** (verified: `americanfootball_nfl.json` first prop has only `fit_brier`) → any display change MUST fall back to `fit_brier`.
- **Callers/tests audited:** `_cv_brier`, `_confirms_over_baseline`, `_method_brier_by_fold` are called ONLY inside `refit_calibration.py` (no test calls them directly) → safe to refactor. `_variant_confirms` IS unit-tested (`test_realline_calibration.py:392-410`) → keep signature/behavior. `_best_per_prop` is also called by `eval_min_streak.py:70` (reads `w['variant','method','brier','hit']`) and `test_calibration_refit.py:98` → keep those keys. Winner-dict fixture at `test_calibration_refit.py:234` is fed to `_build_prop_cfg` via `.get()` → additive keys are safe.

---

## 3. The change (smallest correct form)

### 3a. Two pure fold helpers (refactor, behavior-identical) — near `:466-508`

Add:
```python
def _cv_from_folds(fold_list, method):
    if not fold_list:
        return None
    briers = [bm[method]["brier"] for bm in fold_list if method in bm]
    if not briers or len(briers) < len(fold_list):
        return None
    return round(sum(briers) / len(briers), 4)

def _confirms_from_folds(fold_list, method):
    if not fold_list:
        return False
    for bm in fold_list:
        a, m = bm.get("A"), bm.get(method)
        if not a or not m or m["brier"] >= a["brier"]:
            return False
    return True
```
Refactor the existing public functions to thin wrappers (identical outputs; lets `_best_per_prop` compute folds ONCE per variant instead of recomputing them inside every confirm+cv call — a net efficiency win):
```python
def _cv_brier(obs, method):
    return _cv_from_folds(_method_brier_by_fold(obs, method), method)
def _confirms_over_baseline(obs, method):
    return _confirms_from_folds(_method_brier_by_fold(obs, method), method)
```

### 3b. Pure winner-selector (new, directly unit-testable)
```python
def _pick_winner(cands):
    """Choose the shipped candidate. Prefer the less-optimistic 2-fold cv when
    the safe empirical floor (method A) is itself cross-validatable — this
    guarantees A competes in cv-space, so cv can only pull a pick toward A, never
    admit a gate-rejected method. Fall back to the single-split holdout when folds
    can't form (thin data). min() keeps first-seen-wins on exact ties (preserves
    the rest0.0-before-rest1.0 tie-break, backtest.py:1974-1977)."""
    if not cands:
        return None
    cv_pool = [c for c in cands if c.get("cv_brier") is not None]
    floor_has_cv = any(c["method"] == "A" and c.get("cv_brier") is not None
                       for c in cands)
    if cv_pool and floor_has_cv:
        best = min(cv_pool, key=lambda c: c["cv_brier"])
        best["selected_on"] = "cv_brier"
    else:
        best = min(cands, key=lambda c: c["brier"])
        best["selected_on"] = "fit_brier"
    return best
```

### 3c. Rewrite `_best_per_prop` body `:564-614` (gates unchanged; collect → pick)
Per variant: keep the exact eligibility gates, compute `fold_list = _method_brier_by_fold(obs, None)` once, and append every eligible candidate with its cv:
```python
winners = {}
for prop_key in props:
    base_obs = _baseline_variant_obs(results, prop_key)
    cands = []
    for vname, by_prop in results.items():
        obs = by_prop[prop_key].get("calib_obs") or []
        if not obs:
            continue
        is_baseline = _is_baseline_variant(vname)
        evals = _evaluate_calibration_methods(obs, k_values, holdout=True)
        by_method = {}
        for e in evals:
            if e["brier"] is None or e["k"] not in (None, 0):
                continue
            if e["method"] in ("A", "B", "C"):
                by_method[e["method"]] = e
        baseline = by_method.get("A")
        fold_list = _method_brier_by_fold(obs, None)   # once per variant
        for method, e in by_method.items():
            if method != "A":
                if (baseline is None
                        or baseline["brier"] - e["brier"] < MIN_CALIB_BRIER_GAIN):
                    continue
                if not _confirms_from_folds(fold_list, method):
                    continue
            if (base_obs and not is_baseline
                    and not _variant_confirms(obs, base_obs, method)):
                continue
            cands.append({
                "variant": vname, "method": method,
                "brier": e["brier"], "hit": e["hit"],
                "baseline_brier": (round(baseline["brier"], 4) if baseline else None),
                "baseline_cv_brier": _cv_from_folds(fold_list, "A"),
                "cv_brier": _cv_from_folds(fold_list, method),
                "confirmed": method != "A",
                "variant_confirmed": not is_baseline,
            })
    best = _pick_winner(cands)
    if best:
        winners[prop_key] = best
return winners
```
Winner-dict keys are a **superset** of today's (`variant, method, brier, hit, baseline_brier, cv_brier, confirmed, variant_confirmed`) plus additive `baseline_cv_brier`, `selected_on`. Every existing consumer key is preserved.

### 3d. Persist the new provenance in `_build_prop_cfg` `:630-657` (additive)
Add two lines to the cfg dict (both via `.get()`):
```python
"baseline_cv_brier": winner.get("baseline_cv_brier"),
"selected_on": winner.get("selected_on"),   # "cv_brier" | "fit_brier"
```
Leave `fit_brier` (`:640`) and `baseline_brier` (`:645`) UNCHANGED — they remain the honest single-split provenance of the selection split; `cv_brier`/`baseline_cv_brier` are the advertised less-optimistic figures.

### 3e. Reporting (headline = cv) — `app.py`
Add a module-scope helper and use it:
```python
def _headline_brier(cfg):
    v = cfg.get("cv_brier")
    return v if v is not None else cfg.get("fit_brier")
```
- `app.py:1124-1127`: source **"Cross-validated Brier"** (relabel the existing "Chronological Brier") from `_headline_brier(cfg)` (fallback keeps NFL/NBA rows identical since they lack `cv_brier`).
- Add a secondary column **"Single-split (selection) Brier"** = `cfg['fit_brier']` so the optimistic selector number stays visible and labelled as such.
- Update the markdown copy `app.py:1103-1110` to state the headline is the **2-fold cross-validated** Brier (less optimistic than the single-split figure that selected the variant).
- Optional nicety: `refit_sport` print `:749-752` and `eval_min_streak.py:94-95` may append `cv` next to `brier`; not required (keys preserved).

**Deliberate non-change:** `book_line_calibration.select_method_at_real_lines` `:1263` selection stays single-split. It is NOT a 576-variant argmax (≤5 methods on one real-line variant), so its winner's-curse is far weaker, and switching it risks disturbing the shipped batter_hits E (within-0.002-band, held by incumbent protection / the E-splice recipe). It already persists `cv_brier` `:1393-1397`, so the app headline change surfaces cv for real-line props too — reporting is unified without touching real-line selection.

---

## 4. Gate-semantics preservation (explicit)

- `MIN_CALIB_BRIER_GAIN = 0.002` margin: unchanged, still measured on the single-split A-vs-method (eligibility filter).
- `_confirms_over_baseline` / `_confirms_from_folds` 2-fold confirm: unchanged.
- `_variant_confirms` 2-fold variant gate: unchanged (still called with same args).
- A always eligible (no gate); baseline-variant-A always a candidate (skips both gates) → a winner always exists.
- cv only re-orders the **already-eligible** set and always keeps A in the cv comparison (`floor_has_cv` guard) → the change is monotone-toward-safety: it can demote an optimistic B/C back to A but can never promote a rejected method.

---

## 5. Leakage / reversibility

- **Leakage: none added, net reduced.** `cv_brier` uses `_chronological_folds` (strictly-earlier train, tests on later 60-80%/80-100% windows) — the same OOS folds already used by the confirm gate. Selecting on cv REDUCES selection optimism (winner's curse across 576 candidates on one split is a form of selection leakage into the headline). No future data touches any fit.
- **Reversible:** pure code change in `refit_calibration.py` (+ `app.py` display). It writes NOTHING until someone runs the bare sweep (out of scope). The committed `baseball_mlb.json` is untouched by the code edit. `git revert` fully restores. New JSON fields are additive; no reader requires them.

---

## 6. Risks / interactions (flag for owner)

1. **Effect is deferred to the next bare sweep.** This change alters what a FUTURE `refit_calibration.py --sport mlb` selects; it does not modify the live JSON now. Materializes only when the owner runs the sweep.
2. **Pitcher props (tiny-n synthetic) may re-select toward A / different knobs** on cv — the intended, less-optimistic outcome. Shipped today: pitcher_strikeouts B, pitcher_outs A, pitcher_earned_runs A.
3. **batter_hits synthetic-sweep knobs may change under cv.** The real-line merge (`refit_sport_real_lines`) overwrites only method+residuals and PRESERVES the swept knobs (opp/venue/shrink/rest, comment `refit_calibration.py:963-964`). So a cv-changed synthetic knob pick for batter_hits would survive into production. This directly touches the open **opp0.5 reconciliation** the audit flags (`04-noop-gate-inventory.md §8`): cv would re-decide opp_defense on the less-optimistic basis. Expected/acceptable, but call it out before the sweep.
4. **Incumbent hysteresis unchanged.** E-splice recipe for batter_hits and `MIN_REAL_LINE_OVERRIDE_OBS=500` protection are in the real-line path, untouched.
5. **Degenerate all-push fold** in some variant → that variant's `cv_brier` is None and it drops out of cv competition (conservative; `floor_has_cv` keeps A). No crash; unlikely on real MLB data (synthetic lines rarely equal integer actuals).
6. **Runtime:** cv now computed per eligible candidate. Calibration scoring is O(n) and cheap vs the per-variant projection fetch the sweep already pays; the fold-list-once refactor offsets prior redundant recomputation. Negligible.
7. **NBA/NFL display:** rows lacking `cv_brier` fall back to `fit_brier` → identical to today.

---

## 7. Tests

### Existing tests that pin this area (must stay green)
- `test_calibration_refit.py::SelectionGateTests::test_confirmed_method_is_selected` `:101` and `::test_unconfirmed_method_falls_back_to_empirical` `:111` — perfect-separation / perfect-empirical fixtures where cv and single-split agree in direction; asserts `winner["method"]`, `confirmed`, `cv_brier` not None, `brier < baseline_brier`. All preserved (brier stays single-split; cv_brier still computed).
- `test_calibration_refit.py::BuildPropCfgKnobTests` `:228-267` — feeds a winner fixture (no `selected_on`/`baseline_cv_brier`) to `_build_prop_cfg`; additive `.get()` reads → None; unaffected.
- `test_calibration_refit.py::ChronologicalFoldsTests` `:79` — folds unchanged.
- `test_realline_calibration.py::*_variant_confirms*` `:392-410` — signature/behavior unchanged.

### New tests (add to `test_calibration_refit.py`)
1. **`test_pick_winner_prefers_cv_over_single_split`** — hand-built `cands`: `[{method:"A",brier:0.20,cv_brier:0.24,...},{method:"B",brier:0.18,cv_brier:0.26,...}]`. Single-split argmin = B (0.18) but cv argmin = A (0.24<0.26). Assert `_pick_winner` returns the A candidate and `selected_on=="cv_brier"`. (Crux: proves selection follows cv.)
2. **`test_pick_winner_falls_back_when_no_folds`** — all `cv_brier=None` → returns single-split argmin, `selected_on=="fit_brier"`.
3. **`test_pick_winner_floor_guard`** — A's `cv_brier=None` but a B has cv → `floor_has_cv` False → fallback to single-split argmin (never ships a cv-picked non-A without the A floor in cv-space).
4. **`test_best_per_prop_thin_data_fallback`** — `_make_obs(50,...)` (single-split evals exist at n≥40, folds need n≥60) → winner produced, `winner["cv_brier"] is None`, `winner["selected_on"]=="fit_brier"`; no crash.
5. **`test_build_prop_cfg_persists_cv_provenance`** — extend `BuildPropCfgKnobTests`: winner with `selected_on="cv_brier"`, `baseline_cv_brier=0.25` → cfg carries both.
6. **`test_cv_and_confirm_wrappers_unchanged`** — regression guard on 3a: `_cv_brier(obs,m)` == `_cv_from_folds(_method_brier_by_fold(obs,m),m)` and `_confirms_over_baseline` likewise, on `_make_obs(150,...)`.
7. **(app)** `test_headline_brier_prefers_cv` — `_headline_brier` returns cv when present, fit when cv None, None when neither.

---

## 8. Files touched
- `refit_calibration.py` — `:466-508` (add `_cv_from_folds`/`_confirms_from_folds`, wrapper refactor), new `_pick_winner`, `:564-614` (`_best_per_prop` rewrite), `:630-657` (`_build_prop_cfg` additive fields).
- `app.py` — `:1103-1110` (copy), `:1124-1139` (headline=cv + single-split column), add `_headline_brier` helper.
- `test_calibration_refit.py` — new tests §7.

All changes reversible via git; no JSON/DB write until a sweep is run (out of scope).
