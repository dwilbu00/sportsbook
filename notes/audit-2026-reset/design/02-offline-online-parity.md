# Workstream 2 — Offline==Online Multiplier Parity (implementation spec)

**Date:** 2026-08-07 · **Mode:** read-only design · **Status:** ready to implement (Part A); Part B is defer/attribute
**Goal:** make backtest Brier and deployed Brier measure the *same* projection, by aligning the projection-scaling multipliers the two offline fitters apply with the live `combined_mult`.
**Precedent to mirror:** commit `8e76d86` reconstructed `park_mult` offline in the real-line fitter (`book_line_calibration.py:811-858`).

---

## 0. TL;DR / decision

1. **Park is the one true, cheap, leakage-free parity bug still open — but it lives in the SYNTHETIC SWEEP, not the real-line fitter.** The real-line fitter already reconstructs park (`8e76d86`). The synthetic sweep has the park code (`backtest.py:2636-2650`) but it is **dead** because the sweep grid never sets `park_strength>0` (so `needs_park` is False and `park_s=0`). **FIX PART A: turn park on in the sweep at the sport default (1.0 for MLB), mirroring the real-line fitter + live, and fix the method-A line-shift to divide by `feat_mult*park_mult` (currently only `feat_mult`).** Cheap, leakage-free, reversible, byte-identical for park-neutral props and non-MLB.
2. **matchup / lineup / weather are NOT cheaply reconstructable** — the per-game inputs (opposing-starter identity + as-of stats; batter's batting-order slot; historical temp/wind) are **not in the durable gamelog** (`schema.sql:377-389, 400-411`), unlike park (static table). Weather is also **deliberately non-backtestable** and vendor-API-only. **PART B: do not reconstruct now and do not disable live by default** (they are independently validated / a deliberate physical prior); instead make the residual **measurable + future-reconstructable** by logging live `combined_mult` going forward, keep a **zero-code reversible disable fallback** (set the JSON weights to 0), and **defer** full reconstruction behind a schema add.
3. **pitcher_strikeouts and pitcher_outs have ZERO mismatch and MUST stay byte-identical** — every `combined_mult` factor is a no-op for them (proven below). Any change that perturbs them is a bug.

---

## 1. Verified live `combined_mult` and the per-prop applicability matrix

Live (`props.py:1251-1252`):
```
combined_mult = output_def_mult * matchup_mult * lineup_mult * park_mult * weather_mult * rest_mult
avg_stat      = base_proj * combined_mult                 # props.py:1254  (moves B/C/E)
effective_line= line / combined_mult                      # props.py:1258  (moves method A)
```

Per-factor gating (all confirmed from code + `calibration/baseball_mlb.json`):

| factor | fn / gate | applies to (prop) | shipped strength |
|---|---|---|---|
| `output_def_mult` | `_output_defense_multiplier` props.py:62; gate `output_def_strength` props.py:959,1112 | any | **0.0 on all 5 props → NO-OP everywhere** |
| `matchup_mult` | `_mlb_prop_matchup_mult` props.py:93; weight = `starter_adjustment.props[prop]` | batter_hits, batter_strikeouts (log5 vs opp starter); pitchers (vs opp lineup) | batter_hits 0.5, batter_strikeouts 0.5, **all 3 pitchers 0.0** |
| `lineup_mult` | `_mlb_lineup_exposure_mult` props.py:183 (`if prop_key != "batter_hits": return 1.0`) | **batter_hits only** | 0.75 (enabled) |
| `park_mult` | `_park_factor_mult` props.py:748; `PROP_PARK_KIND={batter_hits:hits, pitcher_earned_runs:runs}` park_factors.py:75 | batter_hits, pitcher_earned_runs | default 1.0 (MLB; props.py:299) |
| `weather_mult` | `_weather_factor_mult` props.py:804; same `PROP_PARK_KIND` | batter_hits, pitcher_earned_runs | default 0.5 (MLB; props.py:317) |
| `rest_mult` | `prop_features.projection_multiplier({"rest":s})` props.py:1246-1249 | {pitcher_outs, pitcher_strikeouts, batter_hits} | 0.0 everywhere → NO-OP |

**LIVE-ACTIVE multipliers per prop (the parity target):**

| prop | active live-only mults (beyond recency/venue/opp_def/shrink already in fitters) |
|---|---|
| **batter_hits** | matchup(0.5), lineup(0.75), park(1.0), weather(0.5) |
| **pitcher_earned_runs** | park(1.0), weather(0.5) |
| **batter_strikeouts** | matchup(0.5) *(dormant market)* |
| **pitcher_strikeouts** | **NONE** |
| **pitcher_outs** | **NONE** |

This confirms the audit: **the mismatch is exactly ZERO for pitcher_strikeouts and pitcher_outs** (matchup weight 0.0, not park/weather kind, rest 0.0, output_def 0.0).

---

## 2. What each offline fitter currently reconstructs (and the gaps)

### 2a. Real-line fitter — `book_line_calibration.project_and_empirical` (700-864)
Reconstructs: recency (`half_life`), `venue`, weight-side `opp_defense` (734-742), xBA blend (764-791), `feat_mult` (rest/gamecontext/platoon, 800-809), **`park_mult` (824-848, commit 8e76d86)**. Folds `combined_mult = feat_mult * park_mult` into projection AND line (854-862).
**Gaps vs live:** `matchup_mult`, `lineup_mult`, `weather_mult`, `output_def_mult`.

### 2b. Synthetic sweep — `backtest.run_player_props_backtest` (2329-…), grid `_build_props_sweep_grid` (1946-2001)
Per-variant projection loop (2560-2671) reconstructs: recency, `venue`, weight-side `opp_defense` (2564-2571), `shrink_k` (2591-2596), output-side `def_adj` (2605-2622, this IS `output_def` but as a **swept knob**, self-consistent), `feat_mult` (rest, 2662-2671). Park code exists (2636-2650) **but is dead**: the grid never sets `park_strength` (`_preset` default 0.0, `_build_props_sweep_grid` doesn't pass it), so `needs_park` (2415) is False and `park_s` (2636) is 0.0.
**Gaps vs live:** `park_mult` (**disabled**), `matchup_mult`, `lineup_mult`, `weather_mult`.
**Also a latent method-A inconsistency (2697-2707):** the calib_obs empirical-over line is shifted by `feat_mult` only, NOT by `park_mult` — so even if park were on, method A would disagree with B/C/E *within the sweep*. Contrast the real-line fitter which correctly uses `feat_mult*park_mult` (854).

### 2c. Which fitter actually fits which prop (per audit + JSON)
- **batter_hits** → real-line (n=3262 ≥ MIN_REAL_LINE_OVERRIDE_OBS=500), method **E**. Remaining gaps: **matchup, lineup, weather** (park OK).
- **pitcher_earned_runs** → synthetic (n≈333 < 500), method **A**. Gaps: **park (sweep-disabled), weather**.
- **pitcher_strikeouts** → synthetic, method **B**. Gaps: **none**.
- **pitcher_outs** → synthetic, method **A**. Gaps: **none**.
- **batter_strikeouts** → synthetic, dormant (0 warehouse lines), method A. Gap: matchup (moot, no real lines).

The synthetic sweep is **also stage 1 of every full refit** (it selects A/B/C that seeds the real-line stage) for ALL props, so the sweep park fix matters for both batter_hits and pitcher_earned_runs.

---

## 3. PART A — Synthetic-sweep park parity (implement now; mirror 8e76d86)

**Objective:** the sweep applies `park_mult` to batter_hits/pitcher_earned_runs exactly as live + the real-line fitter, and shifts the method-A empirical line by the same `feat_mult*park_mult`.

**Edits, all in `backtest.run_player_props_backtest` (`backtest.py`):**

1. **Resolve the sport park default (near 2409-2415).** Mirror the real-line fitter's fallback (`book_line_calibration.py:827-829`):
   ```python
   import props
   default_park_strength = props._player_prop_park_strength(sport_key)   # MLB→1.0, NBA/other→0.0
   needs_park = (default_park_strength or 0) > 0 or any(
       (p.get("park_strength", 0.0) or 0.0) > 0 for p in variants.values())
   ```
   (`needs_park=True` for MLB ⇒ `team_id_to_name`/`pitcher_team_name` get built, so home-game parks are not dropped — same reason the real-line fitter resolves the team.)

2. **Inject the default in the projection loop (2636).** Replace `park_s = params.get("park_strength", 0.0) or 0.0` with:
   ```python
   park_s = params.get("park_strength", 0.0) or 0.0
   if not park_s:
       park_s = default_park_strength or 0.0
   park_mult = 1.0                      # NEW: always defined (needed by the calib_obs shift)
   if park_s > 0:
       ... (unchanged 2637-2649) ...
       park_mult, _ = _park_factor_mult(prop_key, past_parks, weights, upcoming_park, park_s)
       projected *= park_mult
   ```
   (Initialize `park_mult=1.0` *before* the guard; today it is only assigned inside `if park_s>0`.)

3. **Fix the method-A line shift (2697-2707)** to mirror `book_line_calibration.py:854`:
   ```python
   combined_mult = feat_mult * park_mult
   line_eff = (synthetic_line / combined_mult
               if combined_mult and combined_mult != 1.0 else synthetic_line)
   empirical_over = _weighted_rate(prior_values, weights, lambda v: v > line_eff)
   ```

**Why this is correct + safe:**
- **Leakage-free.** Park is a static table (`park_factors.park_factor`); `opponent`/`is_home` are already in the gamelog; `upcoming_park` comes from `test_game` (known pre-game). Uses the SAME primitive (`props._park_factor_mult`) already shipped in the real-line fitter and live.
- **Byte-identical for the zero-mismatch props.** `PROP_PARK_KIND` excludes pitcher_strikeouts/pitcher_outs/batter_strikeouts ⇒ `_park_factor_mult`→1.0 ⇒ `projected` and `line_eff` unchanged (the pitcher_outs control the real-line test already pins).
- **Byte-identical for non-MLB / when disabled.** If `default_park_strength==0` (NBA, unknown sports), `park_s` stays 0, `park_mult=1.0` ⇒ old behavior exactly. This is the reversibility lever.
- **Consistent across all three surfaces.** After the fix, live, the real-line fitter, and the sweep all apply park at `props._player_prop_park_strength(sport_key)`.

**Expected effect:** the sweep's projection + method-A over-rate for batter_hits/pitcher_earned_runs now include the park delta, so the sweep's `fit_brier` and A/B/C selection become comparable to live and to the real-line stage. This *may* change A/B/C selection on a re-run — that is intended and is re-adjudicated through the diagnostic lenses (Workstream 6) before any write. The code change itself writes nothing.

---

## 4. PART B — matchup / lineup / weather (attribute + defer, don't reconstruct now)

### 4a. Feasibility (why these are NOT the park case)
| signal | live inputs | durable availability | verdict |
|---|---|---|---|
| **lineup_mult** (batter_hits) | batter's **upcoming batting-order slot** + recent AB (`_lineup_exposure_mult` props.py:165) | `mlb_batter_gamelog` has **no batting_order** (schema.sql:377-389); only `prediction_log` has it (schema.sql:44), and only for logged rows | needs schema add + boxscore backfill → **defer** |
| **matchup_mult** (batter_hits/strikeouts) | **opposing starter identity** + as-of `xba/k_pct/avg_ip/bf` (props.py:124-162) | gamelog stores only `opponent` team; no opposing-starter table; as-of pitcher-stat index not built for this | needs new opp-starter-per-game data + as-of index → **defer** |
| **weather_mult** (batter_hits/ER) | per-game **temp/wind** forecast (`weather_factors.get_game_weather`) | **not stored anywhere**; needs Open-Meteo *archive* API (vendor, slow, out of scope). Deliberately non-backtestable (props.py:305-317) | **do not reconstruct** |

### 4b. Recommendation (ranked)
1. **Do NOT reconstruct now.** The inputs are absent from the durable corpus; weather is vendor-only + non-backtestable by design.
2. **Do NOT disable them live by default.** matchup (`starter_adjustment`) was fit on 2024-25 with a chronological holdout + 2 folds (JSON `_note`); lineup was forward-validated; weather is a deliberate physical prior shipped conservatively and CLV-gated. Disabling to chase props-Brier parity discards validated signal — net-negative. (Audit already rates this mismatch SECONDARY, sign-unproven, Brier-neutral.)
3. **NEAR-TERM (reversible, no live behavior change): make the residual measurable + future-reconstructable.** Log the live `combined_mult` (or its components) onto `prediction_log` at prediction time (additive **nullable** column, e.g. `combined_mult FLOAT NULL`; populated where computed, NULL for legacy rows). This lets (a) the forward scorer divide the live-only mults out for a like-for-like comparison (shared with **Workstream 4**), and (b) a future refit reconstruct them from logged rows. Purely additive; revert = ignore/drop the column. *Optional within this workstream; the primary owner is WS4 — coordinate to avoid a double-add.*
4. **Reversible DISABLE fallback (zero code) if strict parity is ever mandated.** Set `starter_adjustment.props.batter_hits/batter_strikeouts = 0`, `lineup_adjustment.props.batter_hits = 0`, `weather_factor_strength = 0` in the JSON. No code change; instant revert; documented as trading validated signal for comparability.
5. **DEFERRED full reconstruction (spec only).** Add `batting_order` to `mlb_batter_gamelog` + backfill from boxscores; build an opposing-starter-per-game table + an as-of pitcher-stat index (mirror `savant_history` as-of); then reconstruct matchup/lineup in `project_and_empirical` exactly as park was (fold into `combined_mult`, divide the line). Large; not this workstream.

---

## 5. Leakage flags (explicit)
- **Part A (park):** none — static table, no future/vendor data.
- **Deferred matchup:** must use as-of pitcher stats **strictly before** the obs `game_date` (mirror `savant_history.py` `game_date < as_of`) and the **announced/probable** opposing starter, not the actual (using the actual is a minor look-ahead only when a probable is scratched).
- **Deferred lineup:** use the **announced** slot; the actual boxscore slot is a low-risk proxy (batting order is known pre-first-pitch).
- **Deferred weather:** archive reanalysis = **actual** conditions vs the live **forecast** ⇒ mild look-ahead; a faithful reconstruction would need the as-of forecast, not reanalysis. This is why weather stays live-only.

---

## 6. Reversibility
- Part A reverts via git, and is a runtime no-op whenever `props._player_prop_park_strength(sport_key)==0` (the sport-default knob) — so it is self-disabling for any sport without a park table.
- No calibration JSON is written by any code change here (the sweep/real-line *runs* that write are a separate, gated step — Workstream 9). This spec is code-parity only.
- Part B step 3 (logging) is an additive nullable column; step 4 (disable) is a JSON edit; both trivially reversible.

---

## 7. Tests

### Existing pins (must stay green)
- `test_realline_calibration.py::OfflineParkProjectionTests` (632-697): pins the real-line fitter park reconstruction (folds into projection + line; park-neutral byte-identical; pred-log row fails open). The new sweep tests mirror these three.
- `test_calibration_refit.py`: pins sweep/refit selection — **check for any fixture asserting a park-off sweep projection for an MLB park-kind prop** (batter_hits/pitcher_earned_runs); Part A changes those projections. Adjust the fixture or neutralize park in that test (patch `props._park_factor_mult` → `(1.0, None)`), do not weaken the assertion.

### New tests (Part A) — add near the sweep tests (backtest-level)
1. **Sweep folds park into projection AND method-A line.** Run `run_player_props_backtest(..., sweep=True, calibrate=True)` on a batter_hits fixture with all-home priors + a non-neutral upcoming park; assert the variant's `projected` scales by the patched `park_mult` and the `calib_obs` `empirical_over` reflects `line/(feat_mult*park_mult)` (analog of `test_park_folds_into_projection_and_line`).
2. **Zero-mismatch props are byte-identical.** Same run for pitcher_outs and pitcher_strikeouts: `projected` + `calib_obs` identical to a park-neutralized run (patch `_park_factor_mult`→`(1.0,None)`). Guards the "ZERO for pitcher_strikeouts/pitcher_outs" invariant.
3. **Sport default drives it (reversibility).** With `sport_key="baseball_mlb"` park is applied though the grid sets no `park_strength`; with a sport whose `_player_prop_park_strength==0` (or default forced to 0) the run is byte-identical to pre-fix.
4. **`needs_park` true for MLB.** Assert `team_id_to_name`/`pitcher_team_name` are populated so home-game parks aren't dropped (else home batter_hits would silently get no park).

### New tests (Part B, only if logging is added here vs WS4)
5. `prediction_log` round-trips `combined_mult`; NULL-safe for legacy rows; live path still succeeds when the column is absent (schema-tolerant write).

---

## 8. Explicit non-goals / invariants
- No change to `props.py` live projection math (Part A touches only the offline sweep).
- No calibration JSON write from this workstream's code.
- pitcher_strikeouts / pitcher_outs projections and over-rates must be **byte-identical** before/after Part A.
- `output_def_mult` needs no offline work: it is 0.0 on all props (no-op live) and the sweep already handles output-side defense self-consistently via the swept `def_adj` knob; if a prop ever ships `output_def_strength>0`, the real-line fitter would need the same treatment (flag, out of scope today).

---

## 9. File:line index (verified this pass)
- `props.py:1251-1252` combined_mult; `:1254` avg_stat; `:1258` effective_line; `:1284` method-D rate_mult (excludes lineup — separate handling)
- `props.py:62,93,183,748,804` the six multiplier fns; `:957-965` per-prop knob resolution; `:299/317` MLB park/weather defaults; `park_factors.py:75` PROP_PARK_KIND
- `book_line_calibration.py:800-864` real-line fitter feat+park reconstruction (`:854` combined_mult) — the pattern to mirror (8e76d86)
- `backtest.py:1946-2001` sweep grid (no park axis); `:2409-2415` needs_park guard; `:2560-2671` per-variant projection (park dead-code at 2636-2650); `:2697-2707` calib_obs method-A line shift (feat-only bug)
- `sql/schema.sql:377-389` mlb_batter_gamelog (no batting_order); `:400-411` mlb_pitcher_gamelog; `:44` prediction_log.batting_order
- `calibration/baseball_mlb.json`: starter_adjustment.props (batter 0.5, pitchers 0.0), lineup_adjustment.props (batter_hits 0.75), park/weather strengths None→sport default
