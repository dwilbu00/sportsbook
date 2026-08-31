# Audit 2026 Reset — No-op Gate / Disabled-Method / Off-by-Default Inventory

**Area:** Complete inventory of every dormant gate, disabled method, and off-by-default feature in the prediction arsenal.
**Scope:** READ-ONLY. No code/DB/sweep/vendor changes were made.
**Date:** 2026-08-07. Calibration file `fit_timestamp` = `2026-08-07T13:41:36Z` (commit 5c4549b era).
**Legend:** [PROVEN] = read directly from code/JSON in this session. [INFERRED] = from MEMORY.md / code comments; not re-measured (would require running the forbidden lenses/sweep).

---

## 0. TL;DR for the "reset the gates" step

- The **synthetic sweep** (`refit_calibration.py --sport mlb`, `refit_sport`) can only ever produce methods **A/B/C** and the six swept knobs (half_life, opp_defense, def_adj/output_def, shrink_k, venue, rest). Methods **D (xBA-distributional)** and **E (NegBin)** are **structurally impossible** in the sweep — they exist only in the `--real-lines` path. So a bare re-sweep DELETES E and D. [PROVEN]
- A bare sweep **overwrites** each prop's cfg wholesale (via `_build_prop_cfg`), dropping `line_methods`, `real_line_fit`, `mean_scale`, `dispersion`, `xstats_strength`, and resetting `fit_basis` to `synthetic_sweep`. [PROVEN — see §7]
- **gamecontext** and **platoon** features have **ZERO runtime wiring** in `props.py` (grep: no matches). Even if a gate wrote their knobs, production would ignore them. They are diagnostic-only forever until someone adds both a sweep/write path AND runtime threading. [PROVEN]
- The **currently-shipped `batter_hits` has `opp_defense_strength = 0.0`**, not 0.5 as MEMORY.md's §2.1b line implies. The opp0.5 that once "survived the gate" is NOT in the live file — most likely lost in the 2026-08-07 full-recal + E-resplice (incumbent hysteresis in action). [PROVEN — flag for reconciliation, §8]
- Incumbent-hysteresis gotcha is real and mechanically confirmed (§8): after a reset you cannot get E/D/knobs back automatically because (a) sweep can't make them and (b) `--real-lines` won't flip within the 0.002 gate band + `MIN_REAL_LINE_OVERRIDE_OBS=500` thin-guard.

---

## 1. Shipped calibration state (the baseline to reset FROM)

Source: `calibration/baseball_mlb.json` (5 MLB props). [PROVEN]

| prop | method | opp_def | def_adj (output_def) | shrink_k | venue | rest | half_life | n_obs | fit_basis | confirmed / variant_confirmed |
|---|---|---|---|---|---|---|---|---|---|---|
| batter_hits | **E** | **0.0** | 0.0 | 0 | 0.0 | (absent→0) | null | 3262 | real_line | True / False |
| pitcher_strikeouts | **B** | 0.0 | 0.0 | 0 | 0.25 | 0.0 | 5 | 333 | synthetic_sweep | True / True |
| pitcher_outs | **A** | 0.0 | 0.0 | 0 | 0.0 | 0.0 | null | 333 | synthetic_sweep | False / False |
| pitcher_earned_runs | **A** | 0.0 | 0.0 | 0 | 0.0 | 0.0 | null | 333 | synthetic_sweep | False / False |
| batter_strikeouts | **A** | **1.0** | 0.0 | 0 | 0.0 | 0.0 | null | 3143 | synthetic_sweep | False / **True** |

- batter_hits method E params: `mean_scale=1.0036`, `dispersion=0.0` → **NegBin has collapsed to Poisson** (no over-dispersion detected). [PROVEN]
- batter_hits `real_line_fit`: `n_obs=3262`, `xstats_strength=0.0`, `variant_label="none/opp0.0/defadj0.0/shrink0/ven0.0"`. [PROVEN]
- No prop carries `line_methods` (method D never active in production). No prop carries `gamecontext_strength`/`platoon_strength` (unwritable). [PROVEN]
- `batter_strikeouts` is calibrated but NOT in MEMORY.md's usual working list; it ships method A with **opp_defense_strength=1.0** (the only prop with weight-side defense ON) and variant_confirmed=True. [PROVEN — worth attention on reset]

---

## 2. The confirmation-gate constants (the "gates" themselves)

All in `refit_calibration.py`: [PROVEN]

| constant | value | line | role |
|---|---|---|---|
| `MIN_CALIB_BRIER_GAIN` | 0.002 | :39 | A non-A method (B/C/D/E) or a knob variant must beat the empirical baseline Brier by ≥ this on the single holdout AND confirm in 2 folds. The master gate. |
| `MIN_REAL_LINE_OVERRIDE_OBS` | 500 | :53 | A `--real-lines` flip away from the shipped method is suppressed below this obs count (anti-churn on thin pitcher props). |
| `MIN_BUCKET_OBS` | 100 | :84 | Per-line-bucket (LINE_CONDITIONAL) selection floor before a bucket can adopt its own method. |
| `ROI_TIEBREAK_MIN_BETS` | 15 | :66 | Min simulated value-bets for a method's ROI to count in the tiebreak. |
| `ROI_TIEBREAK_MIN_ROI_GAIN` | 0.02 | :67 | ROI winner must beat the Brier leader's ROI by ≥ this. |
| `ROI_TIEBREAK_THRESHOLD` | 0.05 | :68 | Edge threshold for the ROI sim (matches live gate). |
| `LINE_CONDITIONAL_PROPS` | {batter_hits} | :76 | Only prop eligible for per-line-bucket method selection. |
| `LINE_BUCKETS` | [0.5, None] | :77 | ≤0.5 and open-ended top. |
| `LINE_COND_XSTATS_STRENGTH` | 0.5 | :85 | xBA weight used when scoring method D per bucket. |

---

## 3. Method slots A / B / C / D / E — inventory per market

Definitions: `book_line_calibration._score_abc_real` (:1117) and `select_method_at_real_lines` (:1263). [PROVEN]

- **A** = empirical over-rate passthrough (the safe baseline; always eligible).
- **B** = pooled Gaussian residual (mu/sigma). Sweep- or real-line-selectable.
- **C** = pooled residual ECDF. Sweep- or real-line-selectable.
- **D** = §2.4b-2 distributional P(≥k hits) from contact-quality + xBA blend. **Real-line-ONLY**, and only via the per-line-bucket path (`_select_line_methods`, needs `p_dist` on rows + `MIN_BUCKET_OBS`). Whitelisted to batter_hits. **Currently shipped nowhere** (no `line_methods` in JSON). [PROVEN]
- **E** = §2.2 Negative-Binomial count model (`mean_scale` + `dispersion`). **Real-line-ONLY**, admitted only when `negbin_eligible` (prop ∈ `PROP_NEGBIN_ELIGIBLE`). Currently shipped on **batter_hits only** (disp=0.0 → Poisson). [PROVEN]

**Critical:** the synthetic sweep's `_best_per_prop` persists **only A/B/C** (`refit_calibration.py:581` — `if e["method"] in ("A","B","C")`). D and E are never in the sweep candidate set. [PROVEN]

Per-market current slot + why non-A methods are/aren't there:

| prop | shipped | negbin-eligible? | can hold D? | notes |
|---|---|---|---|---|
| batter_hits | E | yes | yes (line-cond) | Only prop that can hold D or E in production. E ships; D never has. |
| pitcher_strikeouts | B | yes | no | B from synthetic sweep (hl5/ven0.25); real-line kept it via incumbent protection (n≈333<500). [INFERRED why B held: MEMORY.md] |
| pitcher_outs | A | yes | no | Never cleared gate; n≈333 thin. [INFERRED] |
| pitcher_earned_runs | A | yes | no | Never cleared gate; C best-ROI but worse-Brier, tiebreak doesn't fire. [INFERRED: MEMORY.md] |
| batter_strikeouts | A | **no** (not in PROP_NEGBIN_ELIGIBLE) | no | opp1.0 knob confirmed; method stays A. |

---

## 4. Swept knobs — opp_defense, def_adj, shrink_k, venue, rest

Sweep grid: `backtest._build_props_sweep_grid` (:1946), 576 variants = half_life[None,5,10,15] × opp_defense[0.0,0.5,1.0] × def_adj[0.0,0.5,1.0] × shrink_k[0,5,10,15] × venue[0.0,0.25] × rest[0.0,1.0]. [PROVEN]

### 4a. opp_defense_strength (weight-side opponent-defense reweighting)
- **State:** OFF (0.0) on all props EXCEPT `batter_strikeouts`=1.0. [PROVEN]
- **Runtime knob:** `props.py:957` reads `opp_defense_strength`; default `DEFAULT_PLAYER_PROP_DEFENSE_STRENGTH=1.0` (props.py:217). **TRAP:** if a prop cfg is *missing* the key entirely, `_knob` (props.py:907) falls back to 1.0 → defense ON at full strength. All 5 props currently set it explicitly (0.0/1.0), so no live trap today, but a hand-edit or malformed reset could silently enable it. [PROVEN]
- **Why off:** [INFERRED, MEMORY.md] in the §2.1b gated re-sweep opp_defense survived only for a few props; the shipped batter_hits opp0.5 was later lost in reset (§8). Pitcher opp-matchup weights fit 0.0 (robust null, not thin data).
- **Re-adjudicate via:** the sweep (auto, if it clears the variant gate in both folds).

### 4b. def_adj / output_def_strength (output-side opponent-defense scaling)
- **State:** OFF (0.0) everywhere. [PROVEN]
- **Runtime knob:** `output_def_strength` (props.py:959).
- **Why off:** [INFERRED, MEMORY.md] "def_adj survive[s] nowhere."
- **Re-adjudicate via:** the sweep.

### 4c. shrinkage_k (Bayesian shrinkage toward season mean)
- **State:** OFF (0) everywhere. [PROVEN]
- **Runtime knob:** `shrinkage_k` (props.py:960). Note `_best_per_prop` deliberately drops per-player shrinkage method variants B*/C* (:578-582) — they overfit OOS.
- **Why off:** [INFERRED, MEMORY.md] "shrink_k ... survive[s] nowhere."
- **Re-adjudicate via:** the sweep (it is a parsed variant-label token; a swept 0 is honored, not overridden by `--shrinkage-k`, per `_build_prop_cfg` :627).

### 4d. venue_strength
- **State:** 0.25 on pitcher_strikeouts (ON, part of the B variant); 0.0 elsewhere. [PROVEN]
- **Re-adjudicate via:** the sweep.

### 4e. rest_strength (§2.6 rest/days-off feature — "rest tenant")
- **State:** OFF (0.0) everywhere it appears; absent on batter_hits E. [PROVEN]
- **Registry:** `prop_features.py:171` — applies to {pitcher_outs, pitcher_strikeouts, batter_hits}; runtime_knob `rest_strength`. Strengths swept {0.0, 1.0}.
- **Runtime wiring:** YES — `props.py:1246-1249` applies `projection_multiplier(prop_key, {"rest": rest_strength}, ...)` when knob>0. So rest is the ONE §2.6 feature that would actually work live if the gate wrote a non-zero knob. [PROVEN]
- **Why off:** [INFERRED, MEMORY.md] ships nowhere; pitcher_outs rest@0.5 single-split +0.0044 but folds=0/thin, str1.0→−0.0195, ROI worse = artifact.
- **Re-adjudicate via:** the sweep (rest is in the grid) OR `--feature-diag --feature rest` (no-write preview).

---

## 5. Off-by-default FEATURES (§2.6 / §3.1 candidate tenants)

### 5a. gamecontext (scalar mean multiplier) — INERT, NO RUNTIME PATH
- **What:** per-batter mean multiplier from `mlb_starters.build_game_context` (own offense vs opposing bullpen). Registry: `prop_features.py:188`, runtime_knob `gamecontext_strength`, whitelisted to batter_hits.
- **State:** OFF. **No sweep axis** (`_build_props_sweep_grid` has no gamecontext loop) AND **no runtime wiring** (grep of `props.py` for "gamecontext" = 0 matches; runtime only threads `{"rest": ...}`). [PROVEN]
- **Consequence:** cannot be auto-adopted by ANY gate and would be ignored by production even if a knob were written. Diagnostic-only.
- **Why off:** [INFERRED, MEMORY.md] +0.0002@1.0 on incumbent E, ~5–10× under the 0.002 gate; absorbed by line-0.5 saturation / shipped opp_defense.
- **Re-adjudicate via:** `--feature-diag --feature gamecontext` (no-write) only. **To SHIP requires code changes:** add gamecontext to the sweep grid or a write path AND thread `gamecontext_factors` into `props.projection_multiplier`.

### 5b. gamecontext variance ("Method G", context-conditional variance)
- **State:** **KILLED before build** [INFERRED, MEMORY.md 2026-08-05]: NegBin E already tests batter_hits variance and fits disp=0.0000 (Poisson floor = not over-dispersed). No code exists. [PROVEN: JSON shows batter_hits dispersion=0.0.]
- **Re-adjudicate via:** would need `--negbin-diag` to first show non-zero dispersion; currently disp re-fits to 0.0000.

### 5c. platoon (batter vs opposing-starter hand) — INERT, NO RUNTIME PATH
- **What:** per-batter vs-hand xwOBAcon residual over all-hands baseline. Registry: `prop_features.py:204`, runtime_knob `platoon_strength`, whitelisted to batter_hits. Offline factor from `book_line_calibration._attach_platoon`.
- **State:** OFF. Same as gamecontext — **no sweep axis, no runtime wiring** (grep of props.py for "platoon" = 0 matches). [PROVEN]
- **Why off:** [INFERRED, MEMORY.md] ships nowhere; +0.0000@0.5&1.0 on incumbent E, 10× under gate; absorbed at line-0.5 P(≥1 hit) saturation.
- **Re-adjudicate via:** `--feature-diag --feature platoon` (no-write) only; same "to SHIP requires code" caveat as gamecontext.

### 5d. xBA / method D / xstats_strength ("xBA feature")
- **What:** blend projection toward xBA-implied mean (`_xstats_blend`, props.py:400) and/or the distributional method D. `PROP_XSTATS_KIND={"batter_hits":"xba"}` (props.py:346).
- **State:** OFF. `xstats_strength=None`/0.0 on batter_hits; no `line_methods` → method D active nowhere. [PROVEN]
- **Runtime wiring:** YES — method D dispatched at props.py:1276 and xstats blend at :1140; both gated on cfg knobs currently 0/absent.
- **Why off:** [INFERRED, MEMORY.md §2.4a] xBA→batter_hits BUILT (4d53a24) but does not ship; saturates at line-0.5.
- **Re-adjudicate via:** `--dist-diag` (method D, no-write), `--real-lines --xstats-strength <w> --dry-run` (blend preview), and the per-line-bucket path can auto-adopt D only when a ≥1.5 bucket reaches `MIN_BUCKET_OBS=100` (`_lc_bucket_ready`, :108).

### 5e. CSW / whiff → pitcher_strikeouts ("CSW feature")
- **State:** NOT BUILT for shipping. `PROP_XSTATS_KIND` only maps batter_hits→xba; strikeouts→whiff/CSW is §2.4b "pending a raw re-pull" (props.py:345 comment; MEMORY.md 2.4b-1 BUILT 4d53a24 but ships nowhere). [PROVEN comment / INFERRED verdict]
- **Re-adjudicate via:** requires a raw Statcast CSW re-pull + a synthetic refit first; no free lens today.

### 5f. NegBin method E — the dormant-path detail
- **State:** SHIPS on batter_hits (disp=0.0 → Poisson). Ships NOWHERE ELSE. [PROVEN]
- **Eligibility:** `PROP_NEGBIN_ELIGIBLE = {pitcher_strikeouts, pitcher_outs, pitcher_earned_runs, batter_hits}` (props.py:366). batter_strikeouts is NOT eligible. [PROVEN]
- **Why off elsewhere:** [INFERRED, MEMORY.md --negbin-diag 2026-08-03] E ships nowhere new — batter_hits +0.0013<gate; pitcher_earned_runs +0.0027 single-split but fails 2-fold; pitcher_outs/strikeouts worse.
- **Re-adjudicate via:** `--negbin-diag` (no-write) or `--real-lines` (auto-adopt if it clears gate/tiebreak).

---

## 6. Team-market & auxiliary gates (secondary, but part of the arsenal)

From top-level blocks of `baseball_mlb.json` + consumers. [PROVEN state; consumption spot-checked]

- **expected_runs_challenger** (Pythagorean/independent-Poisson challenger): `enabled=true` but `live_markets = {moneyline:false, spreads:true, totals:false}`. Consumed at `analysis.py:155` (`if not enabled or not live_markets.get("spreads")`). → **Dormant for moneyline and totals; active only for spreads.** [PROVEN]
- **lineup_adjustment.props**: `batter_hits=0.75` (ON), `batter_strikeouts=0.0` (OFF — "_note: strikeout exposure failed forward validation and remains disabled"). [PROVEN]
- **starter_adjustment.props** (per-prop starter weight): batter_hits/batter_strikeouts=0.5, **all three pitcher props=0.0** (disabled). Also `bvp=0.0` (batter-vs-pitcher term disabled). [PROVEN]
- **prob_shrink** (team markets): spreads 0.25, totals 0.1, moneyline 0.3 — all ON. [PROVEN]
- **recalibration_baseball_mlb.json** (online Platt seed-as-prior): one entry `batter_hits@le_0.5` `validated=true`, `source=book_line_cache_seed`. Per MEMORY.md [online-platt-seed-prior] the loop has never persisted a *fresh SQL* fit — the committed seed still drives prod. Applied at runtime via `_composite_recal_key`/`_method_cfg_for_line` (props.py:615, 1341). [PROVEN file; INFERRED loop history]
- **SUPPRESS_UNDER_MAX_LINE = {batter_hits: 0.5}** (props.py:379): not a calibration gate but a live recommendation filter — model UNDER picks on batter_hits ≤0.5 are demoted from recommendations. Worth knowing for any ROI/hit-rate diagnosis. [PROVEN]

---

## 7. Diagnostic lenses & re-adjudication mechanisms (the toolbox)

CLI on `refit_calibration.py` (arg parser :2640-2797). [PROVEN]

| mechanism | flag / call | writes? | adjudicates |
|---|---|---|---|
| synthetic sweep | `--sport mlb` (bare) → `refit_sport` | **YES (overwrites props)** | A/B/C + 6 knobs, both confirmation gates |
| real-line re-select | `--real-lines [--xstats-strength w] [--dry-run] [--no-roi-tiebreak]` → `refit_sport_real_lines` | YES (merge) unless `--dry-run` | method @ real lines incl. D/E, line_methods, ROI tiebreak |
| NegBin lens | `--negbin-diag` → `diagnose_negbin` | no | method E vs A/B/C |
| distributional lens | `--dist-diag [--dist-xstats-strength]` → `diagnose_distributional` | no | method D vs C |
| feature lens | `--feature-diag [--feature] [--feature-prop]` → `diagnose_features` | no | rest / gamecontext / platoon (+ per-line-bucket, b6056c5) |
| ROI lens | `--roi-diag [--roi-threshold-pct] [--roi-xstats-strength]` → `diagnose_roi` | no | flat-1u ROI per method |
| center lens | `--center-diag [--center-prop]` → `diagnose_center` | no | recency-weighted MEDIAN vs MEAN center (a dormant alt-center method) |
| reliability | `--reliability [--min-cell-n]` | no | conditional calibration curve |
| recalibrate lens | `--recalibrate` | no | post-hoc Platt+isotonic OOS preview |

Note: `--feature-diag`, `--negbin-diag`, `--dist-diag`, `--roi-diag`, `--center-diag`, `--reliability`, `--recalibrate` are ALL no-write. Only the bare sweep and `--real-lines` (without `--dry-run`) mutate the JSON. [PROVEN]

---

## 8. Incumbent-hysteresis gotcha — mechanism & interaction with a full reset

**Mechanism (all PROVEN from code):**
1. Bare sweep (`refit_sport`) builds every prop cfg from `_build_prop_cfg` (:617), whose output dict contains ONLY: method (A/B/C), half_life, venue, opp_def, output_def, shrink_k, rest_strength, labels, briers, `fit_basis="synthetic_sweep"`, + fitted residuals. It carries **no** `mean_scale`/`dispersion` (E), **no** `line_methods` (D), **no** `real_line_fit`, **no** `xstats_strength`.
2. `save_calibration` is called with default `merge_props=True` (calibration_loader.py:105), but since the sweep passes ALL props, `existing_props.update(props_cfg)` **replaces each prop's whole cfg**. → E, D, line_methods, real-line provenance are WIPED; every incumbent effectively resets toward A/B/C with the gate's chosen knobs. [PROVEN]
3. `--real-lines` then re-selects at real lines but is throttled by:
   - `MIN_REAL_LINE_OVERRIDE_OBS=500`: pitcher props (n≈333) can't override their (now reset) incumbent (`_incumbent_protected`, :250).
   - the 0.002 gate band: [INFERRED, MEMORY.md] batter_hits E beats A by only ~0.0019 < 0.002, so the selector KEEPS the worse A. E is not recovered unless spliced back as incumbent FIRST (then incumbent-protection keeps it), or it wins the ROI cross-fold tiebreak.

**Observed live evidence of the hysteresis biting:** the shipped `batter_hits` `opp_defense_strength` is **0.0** with `variant_label="none/opp0.0/..."`, despite MEMORY.md §2.1b recording opp0.5 as a gate survivor. The 2026-08-07 full recal ("bare sweep → --real-lines" + manual E resplice, commit 5c4549b) appears to have **dropped the opp0.5 term**. [PROVEN discrepancy — recommend the owner reconcile whether batter_hits *should* carry opp0.5.]

**Interaction with the owner's 2026-only reset plan:**
- A reset that runs a bare sweep on MLB-2026-only data will: (a) lose method E on batter_hits (Poisson-E gone; reverts to the sweep's A/B/C pick), (b) lose any D/line_methods, (c) re-decide opp/venue/shrink/rest purely on 2026 obs, (d) reset all `fit_basis` to synthetic_sweep.
- To PRESERVE the current arsenal through a reset you must: run the bare sweep, THEN `--real-lines`, THEN manually resplice batter_hits E as incumbent before the real-line pass (the documented recipe) — otherwise E→A demotion recurs.
- If the reset's GOAL is to start clean on 2026 ABS-season data, the hysteresis is a *feature*: it forces every non-A method to re-earn its place on 2026 obs. But be aware D/E cannot be re-earned by the sweep at all — only `--real-lines` on sufficient 2026 real-line obs (batter_hits needs the within-band problem solved; pitchers need n≥500).

---

## 9. What I could NOT determine from code (adversarial notes)

- **The "WHY off" gate margins** (e.g. E +0.0013, gamecontext +0.0002) are all from MEMORY.md, not re-measured — re-running the lenses is out of scope (would read live SQL warehouse). If the owner wants the reset justified by *current* numbers, those lenses (`--negbin-diag`, `--feature-diag`, `--roi-diag`) must be re-run; they are no-write and free.
- **Whether the batter_hits opp0.5 drop was intentional** — the JSON says 0.0; MEMORY.md says 0.5 survived. I cannot tell from code alone whether the owner accepted the 5c4549b outcome. [Open question.]
- **Whether forward-Brier degradation is caused by any of these dormant items being wrongly off** — this inventory establishes STATE + re-enable mechanics; it does not measure predictive impact. That is the job of the diagnostic-lens re-runs.
- **NBA/NFL:** NFL props ship A/A/C (all `confirmed=None`, n=164, off-season/stale). NBA not inspected in depth here (out of the MLB-first scope) — its calibration is large and MEMORY.md flags it as stale/off-season.

---

## 10. Master "reset the gates" checklist (actionable)

For each item: [current state] → [re-enable mechanism].
1. Method E (batter_hits, Poisson) → resplice as incumbent before `--real-lines`, else lost. [PROVEN it's lost by a bare sweep.]
2. Method D (batter_hits, xBA-dist) → only via `--real-lines` once a ≥1.5 line bucket hits `MIN_BUCKET_OBS=100`; preview with `--dist-diag`.
3. opp_defense (batter_strikeouts=1.0 today; batter_hits=0.0) → sweep re-earns; watch the opp0.5-batter_hits reconciliation.
4. def_adj / shrink_k → sweep re-earns (currently survive nowhere).
5. rest → sweep grid OR `--feature-diag --feature rest`; runtime-ready.
6. gamecontext, platoon → `--feature-diag` only; NOT auto-adoptable and NOT runtime-wired — needs code to ship.
7. venue (pitcher_strikeouts=0.25) → sweep re-earns.
8. xstats/xBA blend → `--real-lines --xstats-strength w`; runtime-ready via method D.
9. CSW→strikeouts → needs raw Statcast re-pull + synthetic refit; not adoptable today.
10. Team-market: challenger (ML/totals off), lineup batter_strikeouts (off), starter pitcher-props + bvp (off) → these are fit by their own backtests (`backtest_starters`, `backtest_props`), not the prop sweep; a prop reset does NOT touch them (save_calibration preserves other top-level blocks).

---
*End of inventory. File is self-contained for post-crash/compaction reference.*
