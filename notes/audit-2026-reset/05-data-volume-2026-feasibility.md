# Audit 05 — Is there ENOUGH 2026-only data to fit each market? (data-volume feasibility)

**Date:** 2026-08-07 · **Scope:** per-market go/no-go for a hypothetical "reset calibration to 2026-only" · **Mode:** READ-ONLY (live Azure SQL aggregates + calibration JSON + code). No writes, no vendor calls.

---

## TL;DR — headline verdict

1. **The "reset to 2026-only" premise is largely a MISNOMER.** Every data-driven input to the prop calibration is *already* 100% 2026: all 3,964 `prediction_log` rows, all 605 `odds_snapshot` rows, all 36k batter + ~4k pitcher gamelog rows are 2026; the synthetic sweep already runs `season_year=2026, cross_season="strict"`; `meta.warmup_season = null`; the real-line fit reads the (2026-only) warehouse. A bare re-sweep or `--real-lines` pass on today's store reproduces essentially the *same* fit it produced on 2026-08-07 (modulo ~1 day of warehouse growth). **A reset does not change the training seasons — because there is nothing but 2026 to train on.**

2. **The only non-2026 pieces are the fixed "adjustment" blocks**, fit by *separate backtest scripts* (not the sweep): `starter_adjustment` (props from 2024–2025; team from 2021–2024), `expected_runs_challenger` (2024), `lineup_adjustment`. A *bare sweep preserves these* (save_calibration merges other top-level blocks); a *destructive reset (deleting the JSON) DESTROYS them* and they cannot be rebuilt from 2026 data (see §6 team markets).

3. **Per-market data sufficiency for a 2026-only refit:**
   - `batter_hits` — **ENOUGH** (real-line, 3,262 obs, ceiling 4,312 and growing; clears the 500 override gate).
   - `batter_strikeouts` — enough for the SYNTHETIC sweep only (~3,143 obs); **ZERO real-line data ever** (market not captured in the warehouse).
   - `pitcher_strikeouts / pitcher_outs / pitcher_earned_runs` — **NOT ENOUGH for real-line** (ceilings 419 / 252 / 359 distinct player-games, all < 500). They stay on the thin, cap-limited synthetic sweep (n≈333). A reset reproduces this; it does NOT collapse them *below* today, but it cannot lift them off synthetic.
   - team markets (ML / spread / total) — **NOT ENOUGH** (48–115 graded forward games vs the ~900+ train games the 2024 fit used). Cannot be refit on 2026; a destructive reset would lose the working 2024 fit.

4. **The forward-vs-backtest Brier gap that motivates the reset is worst exactly where 2026 data is thinnest (pitcher props), and a reset cannot fix it** — those fits are already 2026-only and there isn't enough 2026 *real-line* pitcher data to replace synthetic with real. The remedy is warehouse accrual, not a reset. (Forward Brier corroboration in §7.)

---

## Method & caveats (proved vs inferred)

- **DB connectivity CONFIRMED.** `db_store.promote_secrets_from_toml()` → `_configured()=True` → `SELECT 1` returned 1. All numbers below are live counts from the production Azure SQL as of 2026-08-08.
- **PROVED** = a SQL aggregate or a literal in the calibration JSON / source. **INFERRED** = a reconstruction I could not fully execute (e.g. exact matched-obs count after the actuals-join + warmup filter — I use the distinct-(player,date) *ceiling* as an upper bound instead of re-running the join).
- I did NOT re-run the sweep or the actuals join (read-only rule). "Matched real-line obs" is bounded above by `COUNT(DISTINCT player+date)` in the warehouse per prop; the true matched/graded count is somewhat lower (unmatched actuals + warmup filter). The gate comparison (`< 500`) holds *a fortiori* for pitchers since even the ceiling is < 500.

---

## Data inventory — every durable table is 2026-only

Row counts (live):

| table | rows | seasons present |
|---|---|---|
| prediction_log (props, MLB) | 3,964 | **2026 only** (game_date 2026-07-24 → 2026-08-07) |
| market_prediction_log (team) | 284 | **2026 only** |
| odds_snapshot (warehouse) | 605 | **2026 only** (game_date 2026-07-10 → 2026-08-08; 23 distinct dates) |
| odds_line | 14,874 | 2026 only |
| mlb_batter_gamelog | 36,023 completed | **all game_date 2026** (season_bucket 0: 31,797; bucket 2026: 4,226 — same games, two buckets) |
| mlb_pitcher_gamelog | 4,010 | 3,985 in 2026; 11 in 2025; 14 NULL date → effectively 2026-only |
| nba_gamelog / nfl_gamelog | 0 / 0 | empty (off-season) |
| statcast_player_asof | 1,162 | derived; 2026 (+2024 build if present) |
| wagers | 156 | 2026 |

Local `historical_odds/` directory is **EMPTY** — there are no local pre-2026 odds artifacts either. The only historical seasons that ever touched this model live inside the *scalar* adjustment blocks in the JSON, fit long ago by backtest scripts against live-fetched (not durably stored) data.

`prediction_log` contains **only 4 prop_keys**, all 2026:

| prop_key | total | resolved | graded (outcome≠NULL) |
|---|---|---|---|
| batter_hits | 3,267 | 3,019 | 2,999 |
| pitcher_earned_runs | 223 | 199 | 196 |
| pitcher_outs | 221 | 198 | 196 |
| pitcher_strikeouts | 253 | 230 | 226 |

Note: **`batter_strikeouts` has ZERO prediction_log rows and ZERO warehouse lines**, despite having a calibration entry — it is a configured-but-dormant market (not fetched into the warehouse, not forward-tracked, not bet).

---

## Per-market data volume (the core table)

Warehouse `odds_line` (bet_type=player_prop), all 2026:

| prop_key | raw odds_line rows | **distinct (player,date)** = real-line obs CEILING | distinct dates | shipped fit_basis / method / n_obs | real-line override eligible? (needs ≥500) |
|---|---|---|---|---|---|
| batter_hits | 11,364 | **4,312** | 22 | real_line / **E** / n=3,262 | **YES** (already on real-line E) |
| pitcher_strikeouts | 952 | **419** | 22 | synthetic_sweep / B / n=333 | **NO** (419 < 500) |
| pitcher_earned_runs | 992 | **359** | 16 | synthetic_sweep / A / n=333 | **NO** (359 < 500) |
| pitcher_outs | 556 | **252** | 14 | synthetic_sweep / A / n=333 | **NO** (252 < 500) |
| batter_strikeouts | **0** | **0** | 0 | synthetic_sweep / A / n=3,143 | **NO** (never — market not captured) |

Gamelog raw material (2026, completed, dedup game_key):

| | distinct athletes | ≥11 g | ≥15 g | ≥20 g | post-warmup(>10) obs if ALL athletes used |
|---|---|---|---|---|---|
| pitchers | 193 | 166 | 142 | 111 | **1,890** |
| batters | 443 | 421 | 409 | 399 | **27,455** |

**Key reconciliation (INFERRED, strong):** the synthetic pitcher n_obs≈**333** is a *sweep-cap artifact, not a data ceiling*. `refit_sport` calls `_mlb_player_pool(season, max_batters=40, max_pitchers=30)` — it samples only **30 pitchers**, and warmup_games=10 discards each pitcher's first 10 starts. 30 pitchers × ~11 post-warmup starts ≈ 333. The store actually holds 193 pitchers / 1,890 post-warmup obs. So the pitcher synthetic sweep is deliberately capped, not starved — but raising the cap only adds *more 2026 synthetic* obs; it does **not** create real-line obs, which is what pitchers actually lack. Likewise `batter_strikeouts` n=3,143 is the 40-batter cap against 27,455 available.

---

## Gate thresholds (from refit_calibration.py)

- `MIN_REAL_LINE_OVERRIDE_OBS = 500` (refit_calibration.py:53) — a real-line method flip is suppressed (incumbent protected) below 500 obs, because the 2-fold confirmation gate is unreliable there.
- `MIN_BUCKET_OBS = 100` (:84) — per-line-bucket line-conditional selection floor.
- `MIN_CALIB_BRIER_GAIN = 0.002` (:39) — a non-baseline variant/method/feature must beat the baseline by ≥0.002 Brier to ship.
- Sweep pool caps: `max_batters=40, max_pitchers=30` (:281, :664); `games_per_player=80`, `warmup_games=10`.
- `save_calibration(..., merge_props=True)` (calibration_loader.py:105) starts from the existing blob and preserves every non-props top-level block (starter_adjustment, expected_runs_challenger, lineup_adjustment, prob_shrink, market_blend).

---

## Per-market go / no-go for a 2026-only refit

- **batter_hits — GO (but ≈ no-op).** 3,262 real-line obs (ceiling 4,312, growing daily) clears the 500 gate; method E already fit on 2026 real lines. A reset re-fits E on the *same* data. ⚠ Incumbent-hysteresis gotcha (per MEMORY): a *bare* sweep resets the method to A, and `--real-lines` can't rebuild E because E beats A by only ~0.0019 < the 0.002 flip-gate — so E must be spliced back as incumbent first, exactly as the last real refit did.
- **batter_strikeouts — synthetic-only GO.** ~3,143 synthetic obs is ample for the sweep; but it has **zero** real-line data (not in the warehouse), so it can never reach real-line calibration regardless of a reset. Also currently un-tracked (0 prediction_log rows) — verify it is even offered/bet before spending attention on it.
- **pitcher_strikeouts — NO real-line; stays synthetic (thin).** Real-line ceiling 419 < 500. Synthetic n=333 is marginal for the 2-fold gate (it currently confirms B). A reset reproduces ~333 synthetic; no improvement available from 2026 data yet. Closest of the pitcher props to crossing 500 (419 in ~1 month).
- **pitcher_earned_runs — NO real-line; stays synthetic (thin).** Ceiling 359 < 500. Method A (baseline, unconfirmed). Reset = same.
- **pitcher_outs — NO real-line; stays synthetic (thin).** Ceiling 252 < 500 (furthest away). Method A (baseline, unconfirmed). Reset = same.
- **team markets (moneyline / spread / total) — NO-GO for a 2026 refit.** Only 115 / 75 / 48 *graded* forward games exist (§7); `expected_runs_challenger` was validated on 916 train + 1,061 holdout (2024), and `prob_shrink`/`starter_adjustment` on thousands of 2021–2025 games. 2026 is ~10× too thin to refit these. A *bare* sweep leaves them intact; a *destructive* reset (delete JSON) **loses the working 2024 fit with no way to rebuild it from 2026** — this is the single biggest concrete risk of a naive reset.

---

## Forward-Brier corroboration (why this bears on the reset)

Forward Brier from graded rows (`AVG((prob − outcome)²)`) vs the JSON's fit/cv/baseline Brier:

| market | fwd n | **forward Brier** | fit_brier | cv_brier | baseline | forward gap vs fit |
|---|---|---|---|---|---|---|
| batter_hits | 2,999 | 0.2432 | 0.2409 | 0.2411 | 0.2425 | +0.0023 (forward ≈ *baseline*; E's edge didn't transfer) |
| pitcher_earned_runs | 196 | 0.2509 | 0.2287 | 0.2301 | 0.2287 | **+0.022** |
| pitcher_strikeouts | 226 | 0.2669 | 0.2439 | 0.2452 | 0.2583 | **+0.023** |
| pitcher_outs | 196 | **0.2820** | 0.2201 | 0.2256 | 0.2201 | **+0.062** (worse than a 0.25 coin flip) |
| moneyline (team) | 115 | 0.2429 | — | — | — | pick win-rate 61.7% |
| spread (team) | 75 | 0.2513 | — | — | — | pick win-rate 58.7% |
| total (team) | 48 | 0.2498 | — | — | — | pick win-rate 45.8% |

**Read:** the forward degradation is **smallest for batter_hits** (deep, real-line) and **largest for the pitcher props** (thin, synthetic, cap-limited, sub-500 real-line). This is exactly the fingerprint of *not enough real-line data*, not of *stale training seasons*. A 2026-only reset cannot help because (a) those fits are already 2026, and (b) there is no 2026 real-line pitcher data to switch them onto. `pitcher_outs` forward 0.2820 (n=196) is the loudest "broken forward" signal but is thin.

---

## Bearing on the 2026 reset decision

**UNDERMINES the reset as framed.** The decision to "retrain on 2026 only" rests on the belief that the current calibration is contaminated by older seasons. The evidence shows it is **not** — the per-prop calibration is already 2026-exclusive. Therefore:

- A **bare re-sweep + `--real-lines`** reset is at best a *no-op* (reproduces today's fits) and at worst a *regression* (demotes batter_hits E→A via the documented hysteresis unless E is re-spliced). It does not address the forward-Brier gap.
- A **destructive reset** (deleting `calibration/baseball_mlb.json`) is *actively harmful*: it discards the 2021–2025-fit `starter_adjustment`, `expected_runs_challenger`, and `lineup_adjustment` blocks, which drive the team markets and cannot be rebuilt from 2026's 48–115 graded games.
- The real, data-supported lever is **warehouse accrual**: pitcher props need their real-line obs to cross 500 (strikeouts at 419 is ~weeks away; ERA 359; outs 252 further) before they can escape the synthetic fallback that is producing the worst forward Brier. That is a *wait*, not a *reset*.

If the owner's actual goal is "stop the forward degradation," the data points at the pitcher synthetic-vs-real gap and the fixed 2024 adjustment blocks — neither of which a 2026 reset touches.

---

## Open questions / what I could NOT determine

1. **Exact matched/graded real-line obs after the actuals-join + warmup filter** — I used distinct-(player,date) ceilings (read-only rule; did not re-run `join_book_lines_to_actuals`). Conclusions are robust because pitcher ceilings are already < 500, but the precise batter_hits growth curve toward the next refit is only bounded (3,262 fit → 4,312 ceiling now).
2. **Whether `batter_strikeouts` is still an offered/bet market.** It has calibration but zero warehouse lines and zero forward tracking — appears dormant. Confirm before treating it as in-scope.
3. **Whether the sweep pool caps (30 pitchers / 40 batters) are intentionally low.** Raising `max_pitchers` toward the ~166 pitchers with ≥11 games would grow the synthetic pitcher n from ~333 toward ~1,890 — worth a separate look, but it is *synthetic* obs and may not fix the real-line forward gap. Out of scope for this audit; flagged.
4. **Pitcher forward Brier (esp. pitcher_outs 0.2820) is alarming but thin (n≈196).** Whether that is a genuine miscalibration or small-sample noise is another agent's area (forward-vs-backtest); it is consistent with the synthetic-fit-doesn't-transfer story here.
