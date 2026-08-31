# Third-party audit eval — Maintainability cluster (P9, P10, P11)

Cluster: module decomposition (P9), typed domain objects (P10), unit standardization (P11).
Repo root: `C:\Users\Dwilburn\Documents\Git\ODI_SCRIPTS\SPORTSBOOK_ODDS\deploy`
Evaluator method: every audit claim treated as a hypothesis; verified against code with file:line.

---

## TL;DR verdicts

| Prio | Topic | current_state | Verdict | Integration | Effort |
|------|-------|---------------|---------|-------------|--------|
| P9 | Split god modules | partial | **defer** (then adapt: 1 module, facade+contract-test) | new WS10 "targeted split", post-reset | L (min slice M) |
| P10 | Typed Prediction hierarchy | absent | **adapt** (typed DTO for the persisted wager/prediction row only; defer full hierarchy) | WS7 / provenance backlog | full L; min slice S–M |
| P11 | Fractions internal, ×100 at presentation | partial | **adapt** (candidate dicts → fraction-native, single presentation ×100; golden-test guarded) | new WS11 "units normalization" or backlog | M |

Order of value (highest first): **P11 > P10-minimal > P9**. P11 is the smallest scope with the most direct money-safety payoff; P9 is the largest scope with the most regression risk and collides with in-flight reset work.

---

## P9 — Split the remaining god modules

### Audit line-count claims are ACCURATE (verified `wc -l`)
| File | Audit est. | Actual |
|------|-----------|--------|
| backtest.py | ~3,600 | **3,888** |
| refit_calibration.py | ~2,600 | **2,803** |
| recalibration.py | ~2,400 | **2,553** |
| props.py | ~1,600 | **1,761** |
| odds_client.py | ~1,100 | **1,228** |
| db_store.py | ~1,100 | **1,169** |
| warehouse.py | ~1,100 | **1,149** |

The audit's list is **INCOMPLETE** — it omits the two/three largest maintainability liabilities:
- `app.py` = **3,506 lines** (a Streamlit UI monolith; 2nd largest file overall) — never mentioned.
- `book_line_calibration.py` = 1,777, `mlb_starters.py` = 1,746, `backtest_starters.py` = 1,765, `backtest_props.py` = 1,186 — none mentioned.

So if we ever prioritize by pure line-count/maintainability pain, `app.py` is arguably the biggest single target the auditor missed. (Streamlit page-splitting is its own pattern; flagged, not recommended now.)

### Precedent: the team has ALREADY done exactly this, well
`analysis.py` (now 940 lines) was decomposed into `stats.py` (271) / `pricing_common.py` (336) / `props.py` / `parlay.py` with a **backward-compatible re-export facade** and a **contract-lock test** `test_module_split.py` (lines 1–121). That test pins:
- clean import order / no cycles (subprocess `import <mod>` per leaf, lines 58–69),
- every name still resolves via `analysis.<name>` (facade completeness, 72–89),
- re-exports are the *same object* (`assertIs`, 80–89) incl. shared mutable caches (100–117),
- the `_norm_cdf` dedup (92–97).

This is the proven, low-risk template: **split behind a facade, lock the contract with `assertIs` re-export tests, change zero call sites.** Any future split should reuse it verbatim.

### backtest.py internal shape (why it's a god module)
87 top-level defs (`grep -c '^def \|^class '`). It mixes at least four unrelated statistical pathways in one file:
- team form/matchup backtest: `run_backtest` (379), `run_odds_backtest` (1060), `project_team_form` (225), `project_matchup` (282);
- **calibration WRITERS** (these mutate the money-gating JSON): `_write_shrink_calibration` (882), `_write_blend_calibration` (1385), `_write_market_prior_calibration` (1477);
- props backtest: `run_props_odds_backtest` (1515), `_props_p_over` (1463);
- sweep-grid builders, reliability printing, scoring/grading (`_grade` 614).

Note the backtesting concern is *already* spread across `backtest.py` + `backtest_props.py` + `backtest_starters.py` + `backtest_market_consensus.py` + `backtest_nfl_epa.py`. The audit's proposed `backtesting/` package would **consolidate** these — a large re-org, not just a split.

### Judgment (hard-nosed)
The stated goal — "reduce the chance that a change to one statistical pathway unintentionally affects another" — is legitimate and matches this app's regression-averse ethos. BUT:
1. **Timing collision.** The internal 2026-reset design is *actively rewriting logic inside these exact modules*: WS3 (cv-not-fit selection) and WS6 (pitcher online Platt) touch `refit_calibration.py`/`recalibration.py`; WS4 (like-for-like scoring) touches `backtest.py`. Splitting a file **while** you are changing its behavior stacks refactor risk on top of behavior change and maximizes merge pain. Worst possible time.
2. **Benefit is indirect** (no runtime behavior change, no accuracy/ROI change) on a working, well-tested system.
3. **Big-bang package re-org** (audit's `backtesting/`, `calibration/`, `storage/` trees) is high blast radius across imports and tests.

**Recommendation:** DEFER the package re-org. **Trigger:** after WS3/WS4/WS6 land and stabilize. THEN adapt to the *smallest* high-leverage slice — split exactly ONE module using the `test_module_split.py` facade template. Best first candidate is `backtest.py` along the seam that already half-exists: pull the three `_write_*_calibration` writers + team-vs-props runners apart, because that file has the loosest coupling to the active reset workstreams and the clearest internal seams. Do NOT attempt the full three-package tree in one pass. Leave `refit/recalibration` split until their logic churn (WS3/WS6) is done. `app.py` split is a separate, lower-priority track.

Integration: propose **new WS10 "targeted post-reset module split"** (or backlog). Effort: L for anything ambitious; M for the single-module slice.

---

## P10 — Canonical typed Prediction model

### current_state = ABSENT (verified)
Zero typed domain objects anywhere:
- `grep -rln "dataclass|TypedDict|NamedTuple"` → **no hits**.
- `grep -rln "namedtuple|import attr|pydantic"` → **no hits**.

Predictions flow as **loose dicts** end to end:
- team candidate dict built at `analysis.py:314-335` (`{"type":"moneyline","team":..,"model_prob":..,"edge_pct":..,"is_value":..}`);
- prop candidate dict built at `props.py:1520-1564` — **30+ keys** mixing identity (`player`/`team`/`prop`), prices, probabilities, display flags (`value_pending`, `safe_mode`), nested meta dicts (`calibration`, `recalibration`, `park_factor`, `weather`, `xstats`, `distributional`), and private raw arrays (`_values`, `_weights`).
- Consumed by `.get(...)` all over `app.py`, `wagers.py`, `parlay.py`, `bet_selector.py`, `backtest.py`.

So the audit is correct that this is dict-based. The benefits it lists (type checking, IDE help, fewer misspelled keys, explicit units, safer serialization) are real *in principle*.

### The enforcement infrastructure that would make P10 pay off does NOT exist
- No `mypy`/`pyright`/`pyre` in `requirements.txt`.
- CI (`.github/workflows/tests.yml`) runs only `python -m unittest discover -v`. No static type-check step.

Consequence: a `TypedDict`/annotation gives **documentation only** unless we also adopt a type-checker in CI. A **frozen dataclass** would give *runtime* value even without mypy (misspelled attribute → `AttributeError` instead of silent `None`; `frozen=True` blocks accidental mutation) — but converting the 30-key candidate dicts to dataclasses **fights the codebase's pervasive `.get(key, default)` idiom**, which is load-bearing: e.g. `bet_selector.py:58-70` and `wagers.py:159-225` rely on *absent* keys returning `None`/defaults. A dataclass forces every optional field to be declared with a default and every consumer to switch `c["x"]`/`c.get("x")` → `c.x`, a very large, behavior-sensitive diff across 5 modules.

### Judgment
Full frozen `Prediction`/`PlayerPropPrediction`/`TeamPrediction` hierarchy = **LARGE** refactor whose primary enforcement benefit (static checking) is not wired up and whose runtime benefit collides with the `.get`-default idiom. Not worth big-bang on a regression-averse system today.

**Adapt to a minimal, high-value slice:** introduce a typed object at the **persistence boundary only** — the wager / `prediction_log` **row** DTO. That boundary is already:
- fraction-based (no unit ambiguity — `db_store.py:279` `model_prob ... # picked side prob (0-1)`),
- a stable, small schema (`_PREDICTION_SPEC`/`_WAGER` field lists in `db_store.py:409-431`),
- the natural place a `frozen dataclass` earns its keep (serialization + schema-evolution + it dovetails with the provenance gap: per-prediction `git_sha`/`model_version`/`calibration_version` would be first-class typed fields).

Leave the *analyze-time candidate* dict as-is (or, if anything, cover it with a `TypedDict` **only if** a type-checker is added to CI — otherwise skip). Integration: **WS7** (wager/bankroll archive) or the provenance backlog. Effort: S–M for the row DTO; full hierarchy = L (defer/reject).

---

## P11 — Standardize probability/percentage units

### current_state = PARTIAL — the "fractions internal" rule is ALREADY followed everywhere EXCEPT one layer
Verified the rule holds in:
- the **math core** (all local vars: `model_prob`, `edge`, `final_prob`, `expected_roi` are 0-1 fractions — `analysis.py:280-312`);
- the **DB persistence layer** (`db_store.py:279` `model_prob Float # picked side prob (0-1)`);
- thresholds/config convert once at the edge (`threshold = threshold_pct / 100.0` — `analysis.py:209/355/557`, `props.py:888`, `backtest.py:1081/1525`, `refit_calibration.py:1693`).

The rule is **VIOLATED only in the transient candidate/analysis-result dicts**, which store probabilities ×100:
- `analysis.py:319-333`: `book_implied_prob`, `season_win_pct`, `model_prob`, `hist_prob`, `blended_prob`, `edge_pct`, `expected_roi_pct` all `round(x * 100, 2)`.
- `props.py:1530-1553`: `over_implied`, `over_rate`, `model_hit_at_safe`, `model_hit_at_line`, `model_delta` all `* 100`.

### This forces a whole decode layer and a genuine bug surface
Because the candidate dict is percentage-encoded, **every numeric consumer must re-divide by 100**:
- `parlay.py:190-271` — ~14 `/100.0` reconversions to rebuild fractions for the copula math, incl. `parlay.py:242` `"hist_prob": (100.0 - c["over_hit_rate"]) / 100.0` (the `100.0 -` complement is a direct artifact of the encoding; would be `1 - x` in fractions).
- `bet_selector.py:56` docstring "as a percent (0–100)"; `:65`/`:70` `100.0 - ohr` / `100.0 - over_rate` complements.
- `wagers.py:53-55` defines `_pct(value): return float(value)/100.0` — a **misnamed helper** (called `_pct` but returns a *fraction*), used at `:158/173/182/199/212` to decode `blended_prob`/`cover_rate`/`over_hit_rate`/`over_rate` before Kelly sizing. Comment at `:229-232` explicitly warns the reader that `row['model_prob'] is a 0-1 fraction via _pct`.
- `backtest.py:979-987` `/100.0` when grading candidate probs.
- `app.py`: 61 references to percentage-named candidate keys (`edge_pct` ×29, `expected_roi_pct` ×18, etc.), plus `app.py:140/150` re-dividing (`model_hit_at_safe/100.0`, `threshold_pct/100.0`).

**The latent bug:** the money path (Kelly stake in `wagers.py:238-241`) depends on a consumer remembering to `_pct()` a candidate prob to a fraction. A new consumer that forgets → feeds `63.7` where `0.637` is expected → a ~100× stake error. That is precisely the "important class of future mathematical bugs" the audit names, and it lands squarely on this app's money-safety priority.

### Judgment
Real, concrete, and the best value/scope ratio of the three. It's **partial** (rule already holds in math + DB), so the fix is bounded: make the candidate dicts **fraction-native**, delete the ~30 scattered `/100.0` and `100.0 - x` reconversions in `parlay.py`/`wagers.py`/`bet_selector.py`/`backtest.py`/`app.py`, and put the single ×100 in one presentation helper in `app.py`.

**Blast radius is nontrivial** (61 app.py display refs + 4 consumer modules) and the change must be **behavior-preserving to the pixel** (displayed numbers) and **to the cent** (bet sizing). So: adapt, guard with a **golden test** that asserts the rendered candidate values and Kelly stakes are byte-identical before/after. Because the DB already stores fractions, **no SQL schema or historical-row migration is required** — a big de-risker.

**Trap / sequencing:** P11 and P10-minimal overlap — a fraction-native candidate DTO would naturally be typed. If both are done, do **P11 first** (units), then wrap the now-clean fraction dict in the type. Do NOT do P11 concurrently with WS2 (offline==online multiplier parity) edits to the same numeric pipeline, or with any active props/analysis logic change — same worst-time-to-refactor caution as P9.

Integration: **new WS11 "units normalization"** (small, self-contained) or backlog. Effort: M.

---

## Cross-cutting notes / traps for the design plan
- **Reuse the proven template.** Any P9 split MUST copy `test_module_split.py`'s facade + `assertIs` re-export contract; that is the only reason the analysis.py split was safe.
- **Sequencing rule:** none of P9/P10/P11 should be done *inside* a module while a reset workstream is actively changing that module's behavior. All three are "quiet-period" refactors.
- **No type-checker in CI** is the hidden constraint on P10 — either accept documentation-only value, adopt a frozen dataclass for runtime protection at a narrow boundary, or add pyright to CI first (out of scope here).
- The auditor **missed app.py (3,506 lines)** as the largest UI monolith; note it but Streamlit-splitting is a separate track and low urgency vs. the money-path modules.

---

## Verifier verdict (adversarial re-check against code)

Independently re-verified every file:line citation. **All three findings' current_state labels and verdicts are confirmed correct.** One severity claim in P11 is overstated and corrected below.

### P9 — CONFIRMED (agree: yes)
- Line counts exact (`wc -l`): backtest.py 3888, refit_calibration.py 2803, recalibration.py 2553, props.py 1761, odds_client.py 1228, db_store.py 1169, warehouse.py 1149, app.py 3506.
- `analysis.py` split precedent real: `test_module_split.py` (121 lines) contains the exact facade+contract locks described — `assertIs` re-export tests (lines 80–89), shared-cache identity tests (105–113), clean-import subprocess cycle test (58–69), `_norm_cdf` dedup (92–97). This is a proven low-risk template.
- backtest.py internals confirmed: 87 top-level defs; writers at the cited lines exactly (`_write_shrink_calibration` 882, `_write_blend_calibration` 1385, `_write_market_prior_calibration` 1477, `run_props_odds_backtest` 1515). Backtest concern genuinely sprawls across backtest.py + backtest_props.py (1186) + backtest_starters.py (1765) + backtest_market_consensus.py (1097) + backtest_nfl_epa.py (128).
- "Auditor missed app.py / book_line_calibration.py (1777) / mlb_starters.py (1746)" — all confirmed by wc.
- **Verdict endorsed.** DEFER the big-bang re-org (collides with WS3/WS4/WS6 churning these exact modules; high blast radius; indirect benefit) then adapt ONE module via the facade template. Fully consistent with the regression-averse philosophy — do not refactor a module's structure while a reset workstream is changing its behavior.

### P10 — CONFIRMED (agree: yes)
- `grep -rln -E "dataclass|TypedDict|NamedTuple|namedtuple|import attr|pydantic"` over all *.py → **zero hits**. current_state=absent is exact.
- Enforcement infra absent: no mypy/pyright/pyre/pydantic/attrs in requirements.txt; CI (`.github/workflows/tests.yml`) runs ONLY `python -m unittest discover -v` — no type-check step. So a TypedDict/annotation is documentation-only; only a frozen dataclass would give runtime protection.
- The `.get(default)` idiom is load-bearing (bet_selector.py:57–70, wagers.py:150–226) — confirmed; converting 30-key dicts to dataclasses is a large behavior-sensitive diff.
- Provenance gap confirmed: NO git_sha/model_version/calibration_version in `_PREDICTION_SPEC` (db_store.py:391) or `_WAGER_SPEC` (402) — so P10-minimal's "row DTO dovetails with the provenance backlog" is well-founded.
- **Verdict endorsed** (minimal row DTO only; defer full hierarchy). Caveat: the persistence path is ALREADY spec-driven (`_PREDICTION_SPEC`/`_WAGER_SPEC` do centralized typed coercion), so even the minimal row DTO's incremental value over the existing spec lists is modest — it earns its keep mainly if it lands together with the provenance fields (WS7/provenance backlog), not as a standalone workstream. Do not spin up a dedicated workstream for it.

### P11 — CONFIRMED with a severity correction (agree: partial)
- current_state=partial is exact. Rule holds in math core (analysis.py:280–312 all fractions), DB (db_store.py:279 `model_prob # picked side prob (0-1)`), thresholds (`/100.0` at the edge). Violated only in transient candidate dicts (×100): analysis.py:319–333 and props.py:1530–1553 confirmed.
- Decode-layer artifacts confirmed verbatim: `_pct(value): float(value)/100.0` misnamed helper (wagers.py:53–55) used before Kelly sizing; the explicit warning comment "row['model_prob'] is a 0-1 fraction via _pct" (wagers.py:229–232); `100.0 - ohr`/`100.0 - over_rate` complements (bet_selector.py:65/70); `(100.0 - c["over_hit_rate"]) / 100.0` (parlay.py:242) and the ~14 `/100.0` reconversions in parlay.py:228–271.
- **CORRECTION — the "~100× stake error" headline is OVERSTATED.** `kelly_fraction` (pricing_common.py:102–125) gates on `er > 0` and returns `min(f, cap)` with `cap` defaulting to 0.05; `kelly_stake` (130–142) then does `bankroll * f`. If a consumer forgets `/100` and feeds 63.7 instead of 0.637, `er` is hugely positive so `f` is **clamped to the 5% cap** — the failure mode is "the bet always maxes to the cap and the EV gate is silently defeated," a real but BOUNDED bug, not a literal ~100× stake blow-up. The unbounded-units-bug risk is genuine but lives in the *non-capped* consumers (e.g. parlay copula math fed a probability >1 would produce garbage joint probs) rather than the Kelly path the memo cites.
- **Verdict endorsed** (adapt; new WS11 or backlog; golden-test-guarded; no SQL migration). It remains the best value/scope ratio of the three and the units inconsistency is a real latent-bug surface — just don't sell it on a "100× stake" figure the code's caps prevent. Respect the sequencing trap: not concurrent with WS2 or active props/analysis edits.

### Cross-cutting
No false "already-done." No over-eager adopt — all three are correctly gated (defer / minimal-slice / bounded-with-golden-test) and none violate DK-only, Brier-first, conservative-feature, or fail-closed philosophy. Priority order P11 > P10-minimal > P9 stands.
