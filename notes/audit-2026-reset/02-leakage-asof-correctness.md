# Audit 02 — Leakage / As-Of Correctness

**Date:** 2026-08-07
**Scope:** Verify feature values and labels are computed strictly from information
available BEFORE the game. Determine whether any leakage would make BACKTEST look
artificially good, explaining the forward-vs-backtest Brier gap.
**Verdict (headline):** NO look-ahead leakage found. The as-of machinery is
rigorously strict-before-date on every path I inspected. The one genuine
correctness gap is the KNOWN `combined_mult` omission, which is a TRAIN/SERVE
SKEW (not lookahead) and — for the terms that matter — small in magnitude.
Therefore the forward > backtest Brier gap is a GENUINE out-of-sample
degradation, not a leakage artifact.

---

## What I PROVED (leakage-safe)

### 1. Statcast as-of primitives — strict `game_date < as_of`
`savant_history.py`:
- `asof_pitcher_xwoba` (l.206), `asof_team_xwoba_vs_hand` (l.220),
  `asof_batter_xwoba_vs_hand` (l.235): all filter `r["game_date"] < as_of`.
- `batter_asof_rates` (l.251): `if not gd or gd >= as_of: continue` (l.269).
- `asof_rates` (l.341): same `gd >= as_of: continue` guard (l.354).
All use **strict** `<` (equal-date rows, e.g. same-game batted balls or a
doubleheader, are EXCLUDED). Tests confirm: `test_statcast_asof.py:88-96`
(row on the cutoff is excluded).

### 2. AsOfIndex / AsOfPitcherKIndex — bisect_left is strict-before
`backtest_props.py`:
- `AsOfIndex.asof_mean` (l.240): `bisect.bisect_left(dates, as_of)` → count of
  events strictly before `as_of` (l.243). Same for `asof_window_mean` (l.255),
  `AsOfPitcherKIndex.asof` (l.293), `BatterQualityIndex.asof` (l.339).
Prefix-sum construction sorts by date (l.232, l.278). Leakage-safe.

### 3. Gamelog-based projection — prior games are strictly OLDER
Both calibration paths select `prior_games` as the tail of a newest-first log:
- **Real-line join** (`book_line_calibration.join_book_lines_to_actuals`):
  `gamelog.sort(key=... reverse=True)` (l.379, newest-first) → `test_game =
  gamelog[idx]` (l.438) → `prior_games = gamelog[idx+1:]` (l.450) = strictly
  older. Doubleheader dates tracked (`dup_dates`) and SKIPPED (l.433) so a book
  line can't bind to the wrong same-date box score.
- **Synthetic sweep** (`backtest.run_player_props_backtest`):
  `gamelog.sort(... reverse=True)` (l.2160), `prior_games = gamelog[i+1:]`
  (l.2488), strict-season filter `_filter_to_current_season` (l.2495),
  eligibility rebuilt `as_of_date=test_date` (l.2519), in-progress games
  (`completed is False`) skipped as test (l.2481). Comment l.2498-2501 shows the
  intent is explicit.

### 4. xBA blend in the refit uses PER-GAME as-of, not the live SQL table
`book_line_calibration.project_and_empirical` (l.771-791): blends toward
`xba_index.asof_mean(pid, game_date)` — strict `< game_date` per obs. Docstring
l.696-698 states it "uses a per-GAME as-of estimate ... never the current-as-of
SQL table." Matches production's live blend shape. Leakage-safe.

### 5. Platoon attach (ships nowhere, but checked) — leakage-safe
`_attach_platoon` / `_build_platoon_indices` (l.536-657): quality legs use
`AsOfIndex.asof_mean(..., gd)` (strict `< gd`, l.648-649). The opposing
starter's HAND and the batter's team are read from the graded date but are
**pre-game facts** (lineup card), not outcomes — using them is not leakage.

### 6. Park / weather timing
- Park: static table (`park_factors`); reconstructed identically offline via
  `props._park_factor_mult` (book_line_calibration l.844). No future data.
- Weather: **live-only**, deliberately OMITTED from the backtest
  (`backtest.py:1878-1879` — "no historical per-game weather to reconstruct").
  So weather cannot leak INTO the backtest.

### 7. Live xBA table timing (not leakage)
`statcast_asof.build` writes ONE season-to-date row at `as_of = day after end`
(l.176-177). The live app reads that single row for TODAY's games — season-to-date
through build time, no future info. If anything it is slightly STALE (conservative),
the opposite of leakage.

---

## The one real correctness gap: `combined_mult` omission (TRAIN/SERVE SKEW)

**This is NOT leakage** (no future info enters the backtest). It is a mismatch
between the projection the calibration is FIT/SELECTED on and the projection it
is APPLIED to at serve. It was already flagged in MEMORY as a CONFIRMED audit
finding ("offline projection omits combined_mult; method-select on shifted
dist"). I verified it end-to-end and assessed magnitude.

**Production** (`props.py:1251-1258`):
```
combined_mult = output_def_mult * matchup_mult * lineup_mult
              * park_mult * weather_mult * rest_mult
avg_stat      = base_proj * combined_mult
effective_line = line / combined_mult
```
**Offline calibration basis** — reconstructs only `park_mult` (× inert features):
- `book_line_calibration.project_and_empirical:854`: `combined_mult = feat_mult * park_mult`
- `backtest.run_player_props_backtest`: applies `park_mult` (l.2648-2650) and
  `feat_mult` (l.2663-2671); has **no** `matchup_mult`, `lineup_mult`, or
  `weather_mult` (grep confirms `matchup_mult` never appears in backtest.py).
`feat_mult` = rest/gamecontext/platoon, all INERT (strength 0) in the shipped
JSON, so offline it is effectively **park only**.

**Terms omitted from the fit basis but ACTIVE at serve:**
- `matchup_mult` — **ACTIVE for batter_hits and batter_strikeouts**
  (`calibration/baseball_mlb.json` → `meta.starter_adjustment.props.results`:
  `selected_weight = 0.5` for both; `pitcher_outs`/`pitcher_earned_runs` = 0.0).
  Bounded `[0.7, 1.4]` (`props.py:162`).
- `weather_mult` — live-only; fires on weather-sensitive batter_hits /
  pitcher_earned_runs games.
- `lineup_mult` — `meta.lineup_adjustment.props = null` → effectively no-op now.
- `output_def_mult` — opp-defense output scaler (separate from the weight-side
  opp_defense that DOES ship); applied in production combined_mult.

**Where the skew bites at serve** (calibration applied to the shifted `avg_stat`):
- Method B: `apply_calibration_with_warmup(method_cfg, avg_stat, line, ...)`
  (`props.py:1320-1321`).
- Method E (batter_hits' shipped real-line method): `_negbin_over_rate(avg_stat,
  ...)` (`props.py:1307-1309`).
- Method A: empirical over-rate at `effective_line = line/combined_mult`
  (l.1258) — the served line includes matchup; the refit's method-SELECTION
  compared A/B/C/E at a line that did not.

So both the fit calibration PARAMETERS and the METHOD SELECTION were decided on a
projection distribution (and effective line) that production never actually serves.

**Magnitude (adversarial check):** For the only meaningfully-active term
(matchup on batter_hits), the signal is very weak: `signal_correlation = 0.0379`,
`baseline_mae 0.73377 → candidate_mae 0.73349` (a +0.04% MAE gain). matchup_mult
therefore sits very close to 1.0 with small dispersion, so the resulting
miscalibration is small. Weather's magnitude is unmeasured here (live-only) but
only fires on a subset of games. **This gap is real but is unlikely to be the
primary driver of a LARGE forward-vs-backtest divergence.**

---

## Bearing on the 2026 reset

- **No leakage inflating the backtest** ⇒ the forward > backtest Brier gap is a
  GENUINE out-of-sample degradation, not a look-ahead artifact. This rules out
  "the backtest was cheating" as the explanation. Mildly **SUPPORTS** the reset
  thesis in the sense that the gap is real and could be regime-driven (2026 ABS
  season), but my area CANNOT distinguish regime-change from ordinary
  overfitting — that is other agents' scope.
- The as-of machinery is **sport/season-generic and strict-before-date**, so a
  2026-only refit would be **equally leakage-safe**. No leakage obstacle to a reset.
- The `combined_mult` train/serve skew is **STRUCTURAL** — a 2026-only reset does
  NOT fix it. If any portion of the forward gap comes from this skew, the reset
  will not close that portion. Recommend (independent of the reset): either
  reconstruct `matchup_mult` offline in the calibration basis (the opposing
  starter's as-of xBA is already available via the Statcast as-of index, so it is
  reconstructable and leakage-safe), or drop matchup at serve. Given matchup's
  tiny measured effect, prioritize this only if batter_hits forward calibration
  is materially off.

**Net:** my area is NEUTRAL-to-mildly-SUPPORTIVE of the reset. It removes leakage
as a candidate cause of the gap and confirms a reset would stay leakage-safe, but
it also shows the reset won't fix the one genuine correctness defect (the skew),
which is small anyway.

---

## Open questions / limits of this audit

1. I could not MEASURE the live impact of the `combined_mult` skew on forward
   Brier (read-only; no refit/backtest run). Magnitude is INFERRED from the
   shipped matchup MAE numbers + clamp bounds, not measured directly.
2. Weather's serve-time dispersion is unquantifiable from code alone (live-only,
   no historical forecasts stored).
3. Whether the forward gap is regime-change (2026 ABS) vs overfitting is OUT OF
   SCOPE for leakage — see the calibration/overfitting and market-regime agents.
4. I did not exhaustively audit NBA/NFL as-of paths (MLB-first scope); the shared
   AsOfIndex/gamelog machinery is sport-generic, so the guarantees should carry,
   but that is INFERRED, not proven for those sports.
