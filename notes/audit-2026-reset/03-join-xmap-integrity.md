# Audit 03 — Join / Identity Integrity (2026 reset decision)

**Date:** 2026-08-07 · **Scope:** READ-ONLY. Player/team identity, id-map, gamelog
joins, props grading. Question: do residual name-based joins remain, could pre-xmap
rows poison a refit, is cross-role (batter/pitcher) contamination possible, quantify
residual risk. **Verdict: join/identity integrity is SOUND. Evidence UNDERMINES the
"corrupted historical labels" rationale for a reset.**

Ground rules honored: only reads, `COUNT`/`GROUP BY` SQL, test suite. SQL WAS
reachable (secrets.toml has SQL_SERVER/DATABASE/USER/PASSWORD; `promote_secrets_from_toml`
→ True). Probes: `c:/tmp/audit-2026-reset/probe{,2,3,4}.py`.

---

## Headline facts (PROVEN via read-only SQL)

### 1. There is NO pre-2026 / pre-xmap corpus in the live store to purge.
- `prediction_log`: 3,964 rows total, **all `baseball_mlb`, all game_date year 2026**
  (resolved rows span game_date **2026-07-24 → 2026-08-06**; resolved_at **2026-07-25
  → 2026-08-07**). 3,646 resolved.
- `odds_snapshot`: 605 rows, all baseball, game_date **2026-07-10 → 2026-08-08**.
- `odds_line`: baseball player_prop = **13,864**; team markets ml/spread/total = 420/348/242.
- `wagers`: baseball 2026-07-24 → 2026-08-07. `market_prediction_log`: 284, 2026-07-29 →.
- **Implication:** the premise "pre-xmap name-based joins corrupted *historical*
  calibration labels" has no substrate — the entire live corpus is 2026 and post-dates
  the id-map work. A "reset to 2026-only" would not change *which* data is used.

### 2. Every resolved row was graded AFTER the hard-ID position-group gate shipped.
- Group gate `mlb_starters.resolve_player_game_stat` (pitching prop → pitcher only,
  hitting → batter only; `mlb_starters.py:1623-1626`), fetching the gamelog by **MLBAM
  id** (`people/{pid}/stats`), not by name — shipped **2026-07-20 (e4a9689)**.
- `min(resolved_at)` = **2026-07-25** > 2026-07-20. So the primary grading path was
  role-gated + id-keyed for **100% of stored rows**. Cross-role contamination via the
  LIVE grading path is effectively impossible for this corpus.

### 3. Shipped real-line calibration RE-DERIVES labels fresh each refit.
- `refit_calibration.py` (8 call sites: lines 802/820, 1098/1105, 1307/1314, 1433/1440,
  1654/1661, 1873/1880, 2184/2191) always does
  `harvest_real_line_book_lines` → `join_book_lines_to_actuals`, which re-derives
  `actual` from the gamelog using CURRENT (post-xmap, SFBB-seeded, role-gated) id
  resolution (`book_line_calibration.py:339-468`). It never trusts stored
  `prediction_log.actual/outcome`.
- **Implication:** labels are not "stuck corrupted"; a from-scratch reset would use the
  SAME re-derived labels as a normal incremental refit. Join-integrity gives the reset
  no benefit.

### 4. Residual name-based (no MLBAM id) join surface is small and fail-SAFE.
- `prediction_log` resolved: **112 / 3,646 (3.1%)** are `name:%`-keyed (22 distinct
  players); 3,534 `mlb:%`; **0 NULL** keys. (name-keyed split: 98 batter, 14 pitcher.)
- `odds_line` player_prop (the calibration book-line source): **570 / 13,864 (4.1%)**
  carry no `player_mlb_id`.
- Fallback chain routes **SFBB-first** (`player_id_map._unique_id` refuses ambiguous
  names → None), then statsapi unique-exact (also refuses ambiguity), then lossy ESPN
  `search_athlete`. Dominant failure mode = **DROP (skipped, no obs)**, not mis-bind.

### 5. Namesake disambiguation works and fails safe — verified on live collisions.
- **"Luis Garcia Jr."** resolves to **two distinct MLBAM ids**, correctly SEPARATED by
  team hint: 671277 on Nationals games, 677651 on Yankees games. The 6 odds_line rows
  from the **Nationals-vs-Yankees** game (both teams have a "Luis Garcia") stayed
  **None / name-keyed** — correctly REFUSED because the hint can't disambiguate when
  both teams share the name. pred_log mirrors this: mlb:671277 (Nationals) + mlb:677651
  (Yankees) + 6 `name:luis garcia jr` (team=None).
- **"Max Muncy"**: 6 rows → 571970 (Dodgers, correct); 3 rows team=None → refused.
- This is exactly the designed refuse-when-ambiguous behavior (`_unique_id`,
  `player_id_map.py:537-578`; team-hint requires EVERY hinted team to canonicalize).

### 6. Role gate present on all grading/series paths + primary hard-ID path.
- Primary: group gate in `resolve_player_game_stat` (since 2026-07-20).
- Sweep: `backtest._player_stat_series` calls `_role_matches_gamelog` (`backtest.py:1443`).
- Book-line calibration: `book_line_calibration.py:408`.
- Recalibration ESPN fallback: `recalibration.py:1519` (ported 2026-08-06, 941ab5d).

### 7. Code health: identity/join tests GREEN.
- 184+ tests pass: `test_player_id_map`, `test_backfill_player_ids`, `test_gamelog_store`
  (85), `test_db_store`, `test_warehouse`, `test_recalibration_durability` (99),
  `test_prediction_log`.
- `test_pitcher_ip_conversion` incl. **`test_sweep_does_not_cross_contaminate_strikeout_pools`**
  passes (15/15) under `PYTHONIOENCODING=utf-8`. Its apparent "error" in a bare run is a
  Windows **cp1252 console crash on the `μ` glyph** in sweep print output
  (`backtest.py:2950`), NOT a logic failure. (Prod uses `cli_encoding.configure_stdio()`.)

---

## Residual RISKS / could-not-fully-determine

**[MEDIUM] The online Platt loop consumes STORED outcomes.**
`recalibration.refit_sport` (lines 2154-2179) fits Platt on `r["outcome"]`/`r["raw_prob"]`
from resolved rows — the ONE calibration surface fed by stored (name-graded-possible)
labels. Mitigations: champion gate vs the committed book-line seed + chronological
holdout (`fit_platt_chronological`, incumbent=seed); and per project memory the loop
"never persisted a validated SQL fit (seed drives prod)". BUT it CAN write
`per_prop_params` to SQL (line 2213) if it beats the seed. Bound: ≤112 name-keyed rows
(3.1%), most of which still grade correctly via statsapi unique-exact. Not a proven live
defect; a bounded theoretical exposure.

**[LOW] Pre-08-06 ESPN-fallback grading window (unquantified).**
3,182 rows were graded before 2026-08-06 (before `_role_matches_gamelog` was ported to
the ESPN fallback, 941ab5d). The ESPN fallback only fires when the hard-ID path (group-
gated since 07-20) returns None, and cross-role only bites on strikeout props (K/SO
label collision). Exposure window is narrow. **Could NOT determine** how many rows
actually took the ESPN fallback — it is not recorded on the row — so this residual is
inferred-small, not quantified.

**[LOW] "Luis Garcia Jr." id 677651 rows are batter props (role B).**
Could NOT verify 677651's true position from code/SQL. If 677651 is a Yankees *pitcher*,
those batter-prop rows carry a pitcher's id and would DROP at the grading/calibration
role gate (fail-safe), not mis-grade. Spot-check candidate; NOT proof of a wrong-player
mis-bind. I found **no evidence of any name binding to a demonstrably wrong id** — every
visible ambiguous case (Luis Garcia Jr., Max Muncy) was correctly separated or refused.

---

## Bearing on the 2026 reset decision

**UNDERMINES the join/identity rationale for a reset.**
- No pre-2026/pre-xmap corpus exists to purge (all data is already 2026 + post-group-gate).
- Real-line calibration re-derives labels fresh, so labels are not "stuck corrupted";
  a reset changes nothing about label provenance and confers no join-integrity benefit.
- Residual name-based joins are 3-4% and fail-SAFE (drop, not mis-bind); live namesake
  collisions are handled correctly.

Therefore the forward-worse-than-backtest Brier gap the owner observes is **unlikely to
be a join/identity artifact.** More plausible drivers (OUT OF SCOPE here): genuine
model/market drift; the ABS-season regime change; small-sample forward noise (~2 weeks,
3,646 graded obs); or a metric-definition mismatch (forward Brier uses STORED outcomes
while backtest uses RE-DERIVED labels — a fair-comparison issue, not corruption).

A fresh refit is SAFE from a join standpoint, but it will not close the forward/backtest
gap because there is no material join problem to fix.

---

## Evidence appendix (probe outputs)

- pred_log keyprefix×resolved: baseball_mlb mlb/False 310, mlb/True 3534, name/False 8,
  name/True 112. 0 NULL.
- pred_log resolved role×id: batter hasid 2921 / noid 98; pitcher hasid 613 / noid 14.
- odds_line prop year×id: 2026 hasid 13294 / noid 570.
- odds_line prop role×id: batter hasid 10896/noid 468; pitcher hasid 2398/noid 102.
- resolved_at months: 2026-07 mlb 1751/name 55; 2026-08 mlb 1783/name 57.
- outcome dist: 0→1714, 1→1903, NULL(push/void)→29.
- namesake collisions (odds_line, >1 distinct id per name): only "Luis Garcia Jr." (2).
- pred_log names carrying BOTH name+mlb keys: "Luis Garcia Jr." (7 mlb / 6 name),
  "Max Muncy" (6 mlb / 3 name).
- Timeline: e4a9689 group-gate 2026-07-20 · bd9529e lineup-gate 2026-07-31 · 9487284
  SFBB suffix/team 2026-08-04 · 941ab5d fallback role-gate 2026-08-06.
