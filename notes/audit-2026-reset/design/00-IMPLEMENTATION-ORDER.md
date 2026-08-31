# 2026-Reset Non-Destructive Fix Sequence — Implementation Order (Sequencer memo)

**Date:** 2026-08-07
**Author:** Sequencer agent (read-only design pass)
**Inputs:** the 9 workstream specs (design/01–09) + audit memos 00-SYNTHESIS, 04-noop-gate-inventory, 06-blob-coupling-removal.
**Rule:** single implementer, working sequentially, tests + commit between each change. This memo does NOT re-derive the specs — it orders them, names the file conflicts, and flags what is not ready. Read the per-workstream memo (design/0N-*.md) before implementing that step.

Scope note: the **conditional in-place refit** (synthesis step 9) and the **forward shadow A/B** (step 10), plus the free `diagnose_*` re-run that gates them (step 6), are OUT OF SCOPE of these 9 workstreams and are handled separately AFTER all 9. They are named only at the end for context.

All file:line anchors below were re-verified against the working tree this pass (see §6). The specs' cited lines still match to within a few lines.

---

## 1. TL;DR ordering

```
Phase A (persistence hardening — do first, strictly serial, close together):
  1. WS8  Blob removal        (recalibration.py, warehouse.py, app.py boot+banners, delete migrate_*)
  2. WS1  SQL-off hardening    (guards the now-2-mode dispatch; F1)      [depends on WS8 landing first]

Phase B (independent, file-disjoint from the cluster — can interleave with A or each other):
  3. WS2  Offline==online parity, Part A   (backtest.py)                 [unblocks WS4]
  4. WS7  Wager/bankroll archive epoch     (bankroll.py, wagers.py, app My-Bets)

Phase C (calibration read/quality + season floor — serial on shared files, all AFTER WS1):
  5. WS5  TRAIN_SEASON_MIN read-side floor (pricing_common, warehouse:950, book_line, recalibration:2154)
  6. WS3  cv_brier selection               (refit_calibration selection, app headline)
  7. WS4  like-for-like forward-parity lens (recalibration:758, refit_calibration lens, app forward)  [depends on WS2; NOT READY until WS2 lands + 3 Qs answered]

Phase D (pitcher online Platt — AFTER WS1 + WS8):
  8. WS6  champion-gated pitcher online Platt (book_line PROPS_BY_SPORT, recalibration:2127/58, seed CLI, recal JSON)

Phase E (blocked — do NOT start until owner unblocks):
  9. WS9  batter_strikeouts forward tracking (app odds fetch, odds_client)  [NOT READY — needs live vendor probe]

Then, separately (not part of these 9): free diagnose_* re-run → conditional in-place refit → forward shadow A/B.
```

Hard constraints honored: **WS8 → WS1** (shared `_sql()` dispatch); **WS6 after WS1**; **WS4 after WS2**.

---

## 2. Dependency graph (why this order)

### 2.1 WS8 before WS1 (the merge-order decision)
Both edit the SAME durable-write functions in `recalibration.py` (`mutate_prediction_log:255`, `mutate_ndjson_log:421`, `save_recalibration:1909`) and the SAME app boot block (`app.py:23-39`), and both touch `warehouse.py`.

- **WS8** collapses the 3-mode dispatch (SQL / Azure-Blob / local-disk) to **2-mode** (`if _sql(): SQL else: local`) by excising the blob arms.
- **WS1** then inserts the loud-prod guard at the TOP of the collapsed dispatch.

Land WS8 first because: (a) if WS1 landed first, its guards would be inside functions that still carry blob retry loops, and WS8 would then have to re-thread the guards through the excision → merge churn + double-guard risk (spec 8 explicitly calls this out); (b) WS8 removes the `_load_recal_cached` blob arm (2025→~2086-2113), which is the **F2** seed-blend hazard — removing it first means WS6 no longer has to "mirror per-key merge into the blob read branch" (that WS6 risk evaporates); (c) safety window is benign: Azure Blob is already fully bypassed in prod (audit 06 §0), so the interval between the WS8 and WS1 commits behaves exactly as prod does today (silent local-disk fallback) — WS8 does not worsen it, and WS1 closes it. Keep local-disk (do NOT do WS8's "Stage 3"; that is WS1's guard job, and Stage 3 would break dozens of hermetic tests).

**Coordination:** WS1 should introduce ONE canonical `db_store.require_sql()` + module-local `_ensure_durable(op)` wrappers. Because WS8 already deleted the alternate blob path, there is exactly one `else: <local>` per choke point to guard — no double-guard. Do WS8 and WS1 in close succession (same working session / adjacent commits) so the durability guarantee is never left half-built, and BOTH before WS6 and before any refit.

### 2.2 WS6 after WS1 (and after WS8)
WS6 risk: the `save_recalibration` local-write path (WS1 guards it at 1909) would, with SQL off, write the local seed and make the champion-gate incumbent self-referential + evict the committed pitcher seed. So WS1's guard must exist before pitcher online-Platt is enabled in prod. WS8-before-WS6 is also beneficial (removes the un-hardened blob recal-read arm WS6 would otherwise have to patch).

### 2.3 WS4 after WS2 (parity is the acceptance basis)
WS4's headline acceptance assertion is `forward_raw == backtest_bare` when Platt is identity — that only holds once WS2's offline projection matches the live projection. WS4 is the instrument that VERIFIES WS2. WS4 is also `ready_to_implement=false` largely because its acceptance shape depends on WS2's landing shape (offline-reconstruct vs disable). So WS2 lands, its open questions get answered, then WS4.

### 2.4 WS5 after WS1 (shared function)
WS5 adds `date_from/date_to` forwarding inside `warehouse.load_prop_lines:950`; WS1 adds a raise-guard at the top of that same function. Sequence WS5 after WS1 and rebase (guard at top, date params in body — non-conflicting hunks). WS5 also touches `recalibration.py` (online-Platt loop ~2154) which is a distinct region from WS1/WS8/WS6 edits there.

### 2.5 WS3, WS7, WS2 have no hard dependency
- **WS2 Part A** (`backtest.py` only, if Part B deferred) is file-disjoint from everything → can be commit #1.
- **WS7** (`bankroll.py`, `wagers.py`, `sql/schema.sql` doc, `app.py` My-Bets region) is file-disjoint from the SQL/blob cluster (shares only `app.py`, in a distinct hunk).
- **WS3** (`refit_calibration.py` selection region + `app.py` headline) has no hard dependency and is hunk-disjoint from WS1/WS8; placed in Phase C only for topical grouping. Freely movable earlier.

---

## 3. File-conflict matrix (serialize these; do not parallelize)

Source files touched by >1 workstream. "hunk-disjoint" = same file, different regions → sequential edit + line rebase is enough; no logical conflict.

| File | Workstreams | Nature | Order / handling |
|---|---|---|---|
| **recalibration.py** | WS8, WS1, WS4, WS5, WS6 | WS8 & WS1 overlap the SAME funcs (255/421/1909); WS4(758), WS5(~2154), WS6(2127/58) are distinct regions | WS8→WS1 first; then WS4/WS5/WS6 in any order (hunk-disjoint), respecting WS6-after-WS1 |
| **warehouse.py** | WS8, WS1, WS5 | WS8 blob helpers (100-182, _read/_write_json); WS1 & WS5 BOTH edit `load_prop_lines:950` | WS8→WS1→WS5; WS1 guard at top, WS5 date params in body |
| **app.py** | WS8, WS1, WS3, WS4, WS7, WS9 | WS8+WS1 overlap boot (23-39); others in distinct hunks (headline ~1103-1139, forward ~1213-1243, My-Bets ~1660-1839, odds ~796/2400/2724/2806) | WS8→WS1 for boot; rest hunk-disjoint, rebase line numbers |
| **refit_calibration.py** | WS1, WS3, WS4 | WS1 main abort (~2734), WS4 lens+dispatch (2670/2743) share the main/argparse region; WS3 selection (554-657) distinct | Serialize; WS1 & WS4 coordinate the argparse/dispatch tail |
| **book_line_calibration.py** | WS5, WS6, WS4 | WS5 import:42 + harvest:282 + join:450; WS6 PROPS_BY_SPORT:50-52; WS4 score:1076-1088 | Near-adjacent top-of-file (WS5 import vs WS6 PROPS_BY_SPORT); rebase |
| **sql/schema.sql** | WS7 (doc comment), WS2 (Part B, optional) | doc/DDL-comment only | trivial; if WS2 Part B ships, align with WS7 comment |
| **test_calibration_refit.py** | WS2, WS3 | both add/adjust sweep-fixture assertions | sequential |
| **test_realline_calibration.py** | WS2, WS5 | WS2 park reconstruction pins; WS5 date-floor | sequential |
| **test_modeling.py** | WS4 (probability_brier_raw), WS8 (drop blob patches) | distinct assertions | sequential |
| **test_recalibration_durability.py**, **test_wagers.py**, **test_bankroll.py** | WS1 (pins stay green), WS8 (port Blob tests → sqlite, drop blob patches) | WS8 rewrites; WS1 must keep them green after | do WS8's port first, then confirm WS1 leaves them green |

---

## 4. Independent batches (file-disjoint, safe to reorder/parallelize)

- **Batch I — {WS2 (Part A), WS7}:** share NO source file (backtest.py vs bankroll.py/wagers.py; WS2 Part A does not touch app.py). Both have no hard dependency. Can be done first, before or alongside Phase A.
- **Batch II — {WS3, WS5}:** file-disjoint from each other (WS3 = refit_calibration.py + app.py headline; WS5 = pricing_common.py + warehouse.py + book_line_calibration.py + recalibration.py + config). Valid as a pair ONLY after WS1 (WS5's warehouse:950/recalibration shared with WS1). WS3 alone has no such constraint.

**NOT a parallel batch:** {WS8, WS1} — they overlap the same functions and MUST be strictly serial (WS8 then WS1).

---

## 5. Readiness / blockers

| WS | ready_to_implement | Blocker / prerequisite |
|---|---|---|
| WS8 | yes | none (delete migrate_* before excising helpers, or their imports break) |
| WS1 | yes | land AFTER WS8; add SPORTSBOOK_REQUIRE_SQL to deployed secrets (belt-and-suspenders; guard also infers from any SQL_* present) |
| WS2 | yes (Part A) | Part B (combined_mult logging) deferred; owner Q: does Part B schema+logging live in WS2 or WS4 (avoid double schema add)? |
| WS7 | yes | owner Q: invocation surface (offline CLI recommended vs UI button); straddler pending-bet true-up |
| WS5 | yes | owner Qs: floor also gates LIVE online-Platt loop (Edit 6)? prior_games trim (Edit 4) wanted? — both default-OFF/no-op today |
| WS3 | yes | effect deferred to next bare sweep; flag opp0.5-batter_hits re-decide interacts with the refit step |
| **WS4** | **NO** | depends on WS2 landing SHAPE; 3 open Qs: (a) ship Platt-adjusted column in v1? (b) include batter_strikeouts (dormant)? (c) is `forward_raw==backtest_bare` the right acceptance assertion given WS2's chosen shape? Becomes ready once WS2 lands + owner answers. |
| WS6 | yes (code) | CODE is ready + reversible; the actual **seed JSON commit is data-gated**: run `recalibration.py --seed --dry-run` to see if each pitcher passes the CV champion gate (Wall A, empirical, cannot be run in a read-only design pass). Commit only the passers. Also owner Q: is the batter_hits orphan re-key in-scope for WS6 (it is a free side-effect), and should batter_strikeouts join REQUIRE_SEED_PROPS? |
| **WS9** | **NO** | needs one live vendor probe (markets=batter_strikeouts across regions us/us2/us_dfs) to decide Path A (add region, cost = markets×regions) vs Path B (retire the market). Path C (empty-market guardrail warning + the logging-side regression test) is ship-anytime and independent; the core A/B decision is blocked on the owner. |

---

## 6. Anchor re-verification (this pass)

- recalibration.py: `_sql`:105, `_prediction_log_blob_url`:128, `_read_log_snapshot`:208, `_write_log_snapshot`:230, `mutate_prediction_log`:255, `_read_ndjson_blob`:347, `_write_ndjson_blob`:395, `mutate_ndjson_log`:421, `_blob_url_for`:469, `_read_json_blob`:481, `_write_json_blob`:512, `log_prediction`:537, `log_prediction_rows`:623, `summarize_prediction_rows`:758, `save_recalibration`:1909, `_load_recal_cached`:2025 (blob arm ~2086-2113), `refit_sport`:2127, `MIN_OBS_FOR_OVERRIDE`:58. (Specs' 771-836/1940-1951/2070-2113/106/255/421 all consistent.)
- warehouse.py: `_sql`:58, `_blob_base`:100, `storage_backend`:115, `_blob_url_for`:122, `_get_blob`:140, `_put_blob`:160, `_read_json`:193, `_write_json`:207, `capture_event_odds`:363, `list_snapshots`:479, `load_team_market_store`:864, `load_prop_lines`:950, `seed_from_store`:1010. (Consistent with specs.)
- app.py boot: blob-URL promotion 23-29, SQL-secrets promotion 33-39 (WS8 removes 23-29; WS1 guard after 39). Confirmed.
- refit_calibration.py: `_confirms_over_baseline`:477, `_cv_brier`:496, `_best_per_prop`:554, `_build_prop_cfg`:617, `diagnose_negbin`:1281, `main`:2616, `--real-lines`:2643, `--store-label`:2648, `--negbin-diag`:2670, `promote_secrets_from_toml`:2734. Confirmed.
- book_line_calibration.py: `from pricing_common import ...`:42, `PROPS_BY_SPORT`:50, `harvest_real_line_book_lines`:258, `join_book_lines_to_actuals`:339, `_score_abc_real`:1117. pricing_common.py exists (WS5 adds helpers there). Confirmed.

---

## 7. Per-step test + commit checklist (for the implementer)

Commit after each step with tests green; run the full ~918-test suite at the end of each phase.

1. **WS8** — delete migrate_blob_to_sql.py + migrate_warehouse_to_sql.py FIRST; excise blob arms in recalibration.py/warehouse.py; collapse storage strings; drop app boot blob-promote + fix 3 banners; port BlobRecalibrationTests/NdjsonReadCacheTests/BlobStoreTests → `configure_engine('sqlite://')`, drop `_prediction_log_blob_url`/`_blob_base` patches from shared fixtures. Run test_recalibration_durability test_warehouse test_db_store test_wagers test_bankroll test_modeling, then full suite.
2. **WS1** — add `db_store.require_sql()` + module-local `_ensure_durable` in recalibration.py/warehouse.py; guard mutate_prediction_log/mutate_ndjson_log/save_recalibration(to_blob-gated)/load_prop_lines/load_team_market_store; add refit_calibration.main abort after promote_secrets and app boot st.stop. New test_sql_off_hardening.py. Verify all Phase-A pins still green (fixtures never set SQL_* env → guard inert).
3. **WS2 Part A** — backtest.py park default + calib_obs shift; keep pitcher_outs/pitcher_strikeouts byte-identical (zero-mismatch invariant test). OfflineParkProjectionTests must stay green.
4. **WS7** — epoch marker + baseline txn + Python-side placed_at filter through the 3 surfaces; keep read_wagers unfiltered. Pin lexicographic UTC-ISO compare.
5. **WS5** — pricing_common helpers (default None short-circuits everything → byte-identical today); thread date_from/date_to; floor online-Platt loop (confirm owner wants Edit 6/Edit 4).
6. **WS3** — `_pick_winner` cv-over-single-split with floor guard + fallback; persist selected_on/baseline_cv_brier; app headline prefers cv. Effect deferred to next sweep.
7. **WS4** — (only after WS2 + owner answers) add probability_brier_raw; new `--forward-parity-diag` no-write lens; new test_forward_parity.py. Acceptance: forward_raw==backtest_bare under identity Platt.
8. **WS6** — add pitcher_outs+pitcher_earned_runs to PROPS_BY_SPORT; REQUIRE_SEED_PROPS + seedless-skip guard; `--seed --dry-run`; fix test_refit_flat_prop_unchanged. RUN `--seed --dry-run` to decide which pitcher seeds (and the re-keyed batter_hits) to COMMIT.
9. **WS9** — blocked; if doing Path C only, add the empty-market guardrail + the logging-side regression test (forbids any future prop filter). Do NOT decide Path A/B without the live probe.

**Then, separately:** re-run the free `diagnose_*`/sweep lenses (no-write) to re-adjudicate gates → only if they show benefit, run the in-place refit (bare sweep → --real-lines → splice batter_hits E as incumbent, NEVER delete the JSON) → forward shadow A/B before promoting.

---

## 8. Leakage / reversibility posture (carried from specs)

Every one of the 9 is reversible (pure code / git-revert; WS7 via revert_epoch; WS6 seed JSON via git). None changes the calibration training basis: WS1/WS8 are persistence-only; WS2 Part A is leakage-free (static park table, opponent/is_home already in gamelog); WS5 only removes older rows (read-side floor); WS3/WS4 are selection/reporting; WS6 is post-hoc chronological Platt with as-of warmup; WS9 is request/logging-side. `calibration/baseball_mlb.json` is touched by NONE of the 9 (only the eventual separate refit touches it, and even then via merge-preserving `save_calibration`).
