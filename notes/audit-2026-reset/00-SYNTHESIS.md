# 2026-Only Calibration Reset — Decision-Grade Synthesis

**Date:** 2026-08-07
**Author:** Synthesizer agent (read-only audit)
**Inputs:** 7 investigator findings + 4 adversarial verdicts (leakage, join/identity, data-volume, redundancy/combined_mult)
**Rule applied:** where an adversarial verdict conflicts with the original claim, the verdict wins.

---

## 0. Bottom line up front

**Do NOT execute the reset as framed ("wipe calibration, train only on MLB-2026").** It is a no-op-to-regression and is contraindicated by every investigator plus every adversarial verdict.

- The premise — "stale pre-2026 seasons are diluting the model" — is **false**. Read-only prod SQL proves the entire durable corpus is already ~100% MLB-2026 (prediction_log 3,964 rows all 2026; odds_snapshot all 2026; ~36k batter + ~4k pitcher gamelogs all 2026 bar 11 stray 2025 pitcher rows; wagers all 2026). Every prop in `calibration/baseball_mlb.json` is `fit_season=2026`, `meta.warmup_season=null`, and the sweep already runs `season=2026 / cross_season=strict`. "Reset to 2026" changes **no training seasons**.
- A **destructive** reset is net-negative: deleting the JSON silently drops the historically-fit team-market / starter_adjustment / prob_shrink blocks (2021–2024, ~900–15k games) that 2026 volume **cannot rebuild**, reverting moneyline/spread/total to code defaults with no recovery path.
- The forward-vs-backtest Brier gap is **genuine out-of-sample degradation** (leakage and join/identity both ruled out), but its dominant drivers are **measurement/selection optimism in the backtest number** and **a pitcher-specific synthetic-line + no-Platt artifact** — none of which a 2026-only reset touches.

**What to do instead:** a sequence of reversible, non-destructive changes (offline==online parity, cv_brier reporting, a reversible `TRAIN_SEASON_MIN` knob, archived wager/bankroll reset, forward shadow A/B) with gates re-adjudicated through the existing `diagnose_*` lenses. Details in §3.

---

## 1. Diagnosis of the forward-vs-backtest Brier gap (reconciled)

### 1.1 What is RULED OUT (high confidence)

| Hypothesis | Verdict | Basis |
|---|---|---|
| Look-ahead leakage inflating the backtest | **RULED OUT** (CONFIRMED) | as-of primitives are strict-before-date: `savant_history.py:206–354` (`game_date < as_of`), `backtest_props.py:243` (`bisect_left`), newest-first gamelog + `prior_games=gamelog[i+1:]` in both fitters (`book_line_calibration.py:379/450`, `backtest.py:2160/2488`), doubleheaders dropped, strict-season. Leakage would make forward look *better*, not worse. |
| Join / identity corruption (pre-xmap name merges) | **RULED OUT** (CONFIRMED) | No pre-2026/pre-xmap corpus exists (100% 2026, post-dates the 2026-07-20 hard-ID group gate). Real-line labels **re-derive fresh from the gamelog every refit** (never trust stored outcomes) — labels self-heal. Residual name joins are ~3% and fail SAFE (drop, not mis-bind). |
| Regime change is what a reset would *fix* | **RULED OUT as a reset rationale** | Everything is already `fit_season=2026`, `cross_season=strict`; there is no pre-ABS training data to exclude. (ABS regime *novelty* survives as a residual-gap hypothesis — see 1.3 — but not as something a reset addresses.) |

### 1.2 Most-supported diagnosis (confidence: HIGH for the shape, MEDIUM-HIGH for the decomposition)

The gap is **real OOS degradation that is largely an artifact of how the backtest number is produced and selected**, compounded by a **pitcher-specific deployment mismatch**. Three mechanisms, in order of support:

**(A) Selection / measurement optimism in the reported `fit_brier` — cross-market.**
`fit_brier` is the argmax winner over a **576-variant** grid on a **single 50/50 holdout**, and is used as **both the selector and the headline number** (`refit_calibration._best_per_prop:554–614`; grid `backtest.py:1965`; single split `backtest.py:3053–3059`). The less-biased 2-fold `cv_brier` (`_cv_brier:496–508`) already regresses it (e.g. pitcher_outs fit 0.2201 → cv 0.2256 → forward 0.282). The data-volume adversarial verdict independently elevated this to a primary cause because **the deepest market (batter_hits, n=3,262) also degrades forward** (0.2432 vs fit 0.2409), which a data-volume story cannot explain but a selection-optimism story can.

**(B) Metric-definition asymmetry — the two Briers are different objects.**
Forward Brier scores `final_prob` = method + Platt + market-blend, on **STORED outcomes over ALL resolved rows** (`recalibration.summarize_prediction_rows:818/780`). Backtest scores the **bare method** on **RE-DERIVED labels over a curated subset** (join drops no-id, doubleheaders, <10 min, <10 prior games) (`_score_abc_real book_line_calibration.py:1135–1164`; `_score_calibration_methods backtest.py:3067`). This is a structural offset, not corruption, and is confirmed by the join adversarial verdict as a fair-comparison caveat.

**(C) Pitcher-specific compounding — explains why pitchers degrade MOST.**
All three pitcher props are (1) fit on **synthetic season-average lines** but graded forward on **real book lines** (`backtest.py:2555–2557,2712–2714`; `fit_basis=synthetic_sweep`), (2) selected off a thin **n=333** top-30-pitcher pool (pool-cap artifact, `max_pitchers=30`), and (3) deployed with **zero live Platt correction** (live recal table holds only 2 batter_hits fits). Result: forward outs 0.282 vs fit 0.2201 (+0.062), K +0.023, ER +0.022 — vs batter_hits (real-line + Platt) +0.002. batter_hits *tracks*; pitchers do not.

**(D) The offline==online multiplier skew is a SECONDARY, not primary, contributor.**
Adversarial verdict **downgraded** the "combined_mult omission is THE root cause across most markets" claim to PARTIAL: the mismatch applies to only **3 of 5 props** and is **ZERO for pitcher_strikeouts and pitcher_outs** (their `starter_adjustment` weight is 0.0; park kind excludes them), and its **sign is unproven** (MEMORY records these features as Brier-neutral). Real defect worth fixing for comparability — but not the dominant driver.

### 1.3 Residual / unresolved

- **ABS-2026 regime novelty** cannot be excluded from code (the app has only ~1 month of ABS-era data). It remains a live competing explanation for the *residual* gap after (A)–(C), but it is NOT addressable by a 2026-only reset because there is no non-2026 data to remove.
- **ROI / hit-rate is an EDGE problem, largely orthogonal to Brier.** batter_hits Brier is fine (0.243) yet directional hit is 46% and ROI −5% — near-zero real edge vs DK. Retraining the calibration window will not fix flat/negative ROI; audit the edge/EV gate and value thresholds separately.

---

## 2. Per-market GO / NO-GO on 2026-only calibration

Because all durable data is already 2026, "2026-only" is a no-op for training seasons everywhere. The verdicts below are about **feasibility and benefit of a 2026-basis refit per market** (data-volume adversarial verdict: anti-reset conclusion CONFIRMED; pitcher <500 ceilings are SQL-derived and were not re-verified from code).

| Market | Verdict | Rationale |
|---|---|---|
| **batter_hits** | GO_2026_ONLY (but a no-op) | Real-line n=3,262 > MIN_REAL_LINE_OVERRIDE_OBS=500; already ships real-line method E on 2026. A refit reproduces E. **Hazard:** a bare sweep resets it to A and `--real-lines` cannot rebuild E (beats A by ~0.0019 < 0.002 flip gate) — E must be spliced back as incumbent first. |
| **pitcher_strikeouts** | NEEDS_MORE_DATA | Real-line ceiling 419 < 500 gate → stays synthetic n=333. Worst-tracked group. Closest to the gate (~weeks of accrual). |
| **pitcher_earned_runs** | NEEDS_MORE_DATA | Real-line ceiling 359 < 500 → stays synthetic. |
| **pitcher_outs** | NEEDS_MORE_DATA | Real-line ceiling 252 < 500 → stays synthetic; furthest from the gate; worst forward Brier (0.282). |
| **batter_strikeouts** | NEEDS_MORE_DATA | Configured but dormant: ZERO warehouse lines, ZERO forward tracking → real-line is impossible, can only ever be synthetic. Confirm it is still offered/bet before spending any effort; candidate to drop. |
| **team markets (moneyline / spread / total)** | KEEP_MULTISEASON | Only 48–115 graded 2026 forward games vs ~900+ that fit `expected_runs_challenger` (2024) / `starter_adjustment` (2021–2024). 2026 cannot rebuild these. Keep the preserved historical blocks; **never delete the JSON.** |

---

## 3. Sequenced plan (diagnosis + reversible BEFORE anything destructive)

All steps are read-side or reversible until step 9. The "reset" is implemented as a **reversible `TRAIN_SEASON_MIN` knob** with gates **re-adjudicated via the sweep / `diagnose_*` lenses (not force-on)**, wager/bankroll **archived not deleted**, and a **forward shadow A/B** to measure success — exactly as the task requires.

1. **Harden the silent SQL-off → ephemeral-disk fallback (F1 guard).** Fail loudly when `_sql()` is false in a production context (`recalibration.py:105–106`, `warehouse.py:58–59`). *Non-destructive, reversible.* Depends on: nothing. **Prerequisite for any later refit** — a refit with SQL accidentally off would appear to succeed while persisting nothing.

2. **Fix offline==online multiplier parity.** Reconstruct matchup/lineup/weather offline (as park was added in 8e76d86), or disable the live-only signals, so backtest Brier and deployed Brier measure the same projection. *Non-destructive, reversible.* Depends on: nothing. **must_fix (see §4).**

3. **Report and select on `cv_brier`, not `fit_brier`, for tiny-n synthetic props.** Surface cv_brier as the headline expected-quality number; consider selecting on it. *Non-destructive, reversible.* Depends on: nothing.

4. **Score forward-vs-backtest like-for-like.** Either score the Platt-adjusted prob in backtest or expose the pre-Platt forward prob, and align label derivation + row subset (stored-on-all vs re-derived-on-curated). *Diagnostic only, reversible.* Depends on: 2.

5. **Add a reversible `TRAIN_SEASON_MIN` knob.** Read-side date filter threaded through the 3 real-line harvest hooks (`harvest_real_line_book_lines` → `warehouse.load_prop_lines(date_from=...)`; SQL layer already supports `date_from/date_to` at `db_store.py:1137–1140`, currently unused; plus pred-log + local-store harvests). **No deletes.** Also add `game_date>=floor` to the online-Platt resolved-row read. *Non-destructive, reversible.* Depends on: nothing. (Near-no-op today — excludes ~11 stray rows — but establishes the reversible lever the owner asked for.)

6. **Re-run the free `diagnose_*` / sweep lenses to re-adjudicate gates.** `--feature-diag`, `--negbin-diag`, `--roi-diag`, per-bucket lens — no write. Confirm whether parity fixes (step 2) or cv-selection (step 3) change any gate outcome. *Read-only.* Depends on: 2, 3, 4.

7. **Enable/seed online Platt for pitcher props** (champion-gated against the committed seed, as batter_hits already is). *Reversible* (gate reverts if it does not beat seed). Depends on: 1.

8. **Archive-then-zero the wager/bankroll ledgers for a clean forward ROI baseline.** `SELECT INTO` snapshot of `wagers` + `bankroll_ledger`, then Option A: an `app_settings` `wager_reset_at` epoch marker honored by `read_wagers`/`reconcile_bet_txns`/ROI views (**no deletes, fully reversible**). NOT deleting bankroll_ledger rows — balance is derived and `reconcile_bet_txns` regenerates bet txns from `wagers` (`bankroll.py:104–260`). *Reversible (archived + epoch-gated).* Depends on: nothing.

9. **ONLY IF steps 2–7 show a 2026-basis refit helps:** run the refit **in place** — bare `refit_sport` → `refit_sport_real_lines` → **splice batter_hits E back as incumbent** — and **never delete the JSON** (so team/starter_adjustment/prob_shrink blocks are preserved by `save_calibration`, `calibration_loader.py:105–132`). *Overwrites shipped calibration → mark destructive, but reversible via git + the E-splice recipe.* Depends on: 2, 3, 8, and a **forward shadow A/B** result.

10. **Forward shadow A/B.** Add a `shadow_prob` column on `prediction_log`; run the candidate basis in shadow against the shipped basis on live 2026 games and compare forward Brier/ROI before promoting. *Non-destructive, reversible.* Depends on: 9 (candidate basis) — run in parallel, promote only on a win.

---

## 4. must_fix_before_refit

A 2026 refit run without these first will be corrupted or misleading:

1. **Offline==online multiplier parity.** The offline fitter omits matchup/lineup/weather (and the synthetic sweep also omits park); a same-pipeline refit reproduces the skew and any post-refit backtest number stays non-comparable to live. Reconstruct or disable before trusting Brier. (`props.py:1251–1258` vs `book_line_calibration.py:854`, `backtest.py` sweep grid.)

2. **Never delete `calibration/baseball_mlb.json`.** `save_calibration` preserves the non-props blocks (starter_adjustment 2021–2024, expected_runs_challenger 2024, prob_shrink, lineup_adjustment) that 2026 volume cannot rebuild. A delete silently reverts team markets to code defaults with no recovery. Do an **in-place** bare sweep only.

3. **Splice batter_hits E back as incumbent before `--real-lines`.** Incumbent hysteresis: a bare sweep resets to A and `--real-lines` cannot rebuild within-gate-band E (Δ~0.0019 < 0.002). Without the splice the refit ships a strictly worse batter_hits — the one market with a real 2026 fit. (`refit_calibration.py:250,733`; `calibration_loader.py:119–121`.)

4. **Score forward-vs-backtest as the same object.** Forward = final_prob (method+Platt+blend) on STORED outcomes over ALL rows; backtest = bare method on RE-DERIVED labels over a curated subset. Any before/after judgment must control this, or a refit will be graded against a phantom baseline.

5. **Harden the silent SQL-off fallback (F1) first.** A refit with SQL misconfigured/off writes to ephemeral disk and no-ops durably while appearing to succeed.

6. **Do NOT select or advertise `fit_brier` for tiny-n synthetic props.** It is a 576-variant single-holdout argmax winner's-curse number; use cv_brier. Otherwise the refit re-enshrines optimistic pitcher fits.

---

## 5. Confidence and open questions

**Confidence.** HIGH: the reset-as-framed is a no-op-to-regression; the gap is not leakage/join/season-mixing. MEDIUM-HIGH: the primary drivers are selection-optimism + metric asymmetry + pitcher synthetic/no-Platt. MEDIUM: relative weighting of those three, and how much ABS-regime novelty contributes to the residual.

**Key open questions carried forward (from investigators + verdicts):**
- Pitcher <500-obs real-line ceilings and all forward-Brier magnitudes are SQL-derived and were **not independently re-verified from code** this pass (data-volume adversarial caveat).
- The counterfactual real-line pitcher Brier is unknown (pitchers never produced a real-line fit — obs < the 20-row gate), so the three-way pitcher decomposition (synthetic-line vs winner's-curse vs no-Platt) cannot be separated.
- Whether the residual gap is ABS-regime drift vs ordinary overfitting is not resolvable from code — measure per-market against a real forward/holdout after the parity fix.
- The sign of the combined_mult skew (does it make forward better or worse?) is unproven; MEMORY records the features as Brier-neutral.
- Row-level namesake examples (Luis Garcia Jr., Max Muncy) are mechanism-proven but not independently re-run (probe output deleted) — PARTIAL, not a refutation.
- Is `batter_strikeouts` still an offered/bet market? Dormant (zero warehouse lines, zero forward tracking).

---

## 6. Reconciliation notes (where verdicts overrode original claims)

- **Data-volume investigator** attributed the gap primarily to pitcher data-thinness. **Adversarial verdict = PARTIAL:** anti-reset conclusion CONFIRMED, but the root-cause is REFUTED (deep batter_hits also degrades forward) → reweighted toward selection-optimism / ABS-regime. Synthesis adopts the verdict.
- **Redundancy investigator** called combined_mult omission "the most parsimonious explanation across most markets." **Adversarial verdict = PARTIAL:** applies to only 3/5 props, ZERO for pitcher_strikeouts/outs, sign unproven → demoted to a secondary/comparability fix. Synthesis adopts the verdict.
- **Leakage and join/identity** claims were CONFIRMED with only scope refinements (no-leakage proven for cited primitives not exhaustively; namesake row-level examples PARTIAL). Synthesis adopts as-is.
