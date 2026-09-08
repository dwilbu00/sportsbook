---
name: active
description: In-flight work only — what's being worked on now, linking to the domain files. Move an item to its domain when done; do not let this file accrete history.
metadata:
  type: project
---

# ACTIVE — in-flight work

One line per open item. When an item ships, delete it here and fold the durable outcome into its `[[domain]]` file. Last reconciled **2026-09-08** (folded the completed NFL Phases 1-3 + margin-model ship + granularity-law + prediction-ceiling into [[modeling-and-calibration]] and [[data-and-architecture]]; ACTIVE slimmed for an independent-review handoff).

> **▶▶ RESUME (2026-09-08):** **NFL is the ACTIVE sport; the TEAM model is COMPLETE, shipped live, and at its prediction ceiling.** Full arc done: clean nflverse data foundation → EPA + starting-QB + injuries(incl. defense) + rest + HFA margin model, OOS-validated and **promoted live** (`nfl_model.py`; OOS RMSE 13.185). Exhaustively established (7 tests) that **matchup sophistication does NOT beat the coarse aggregate OOS** and that we **cannot out-predict the sharp close** — margin 13.19 vs mkt 12.67, Brier 0.226 vs 0.211 (and worse on disagreement games). Details → [[modeling-and-calibration]] (NFL model section + granularity law + ceiling) and [[data-and-architecture]] (NFL data foundation). **⏸ PAUSED for an independent-model codebase review (Doug's call, 2026-09-08) — see `HANDOFF.md`.** After review, the two live frontiers are: (1) **NFL PROPS with the real modeling/calibration machinery** (where MLB's edge + math actually lived — only crudely scanned so far); (2) **calibration (market-shrunk break-even) + high-frequency line logger** for the transient-soft-line thesis. All work on `origin/main`.

## NFL — OPEN / next (team model is done; these are the live threads)
- **▶ NEXT FRONTIER A — NFL PROPS via the modeling machinery.** The heavy MLB stats work was PROPS-side (methods A-E, xStats blends, per-prop calibration); NFL props were only crudely flat-side scanned (`nfl_props_scan.py` → recreational OVER-bias, vig-eaten; the one apparent survivor `receptions OVER 0.5` was a DNP-survivorship artifact, killed). Untapped: real per-prop projection + calibration off the `nfl_player_week`/`nfl_snap` layers. ⚠ `calibration/americanfootball_nfl.json` props section is a TOY (`player_pass_yds` ≡ `player_rush_yds`, byte-identical — copy-paste placeholder) — rebuild from a real player-week layer with OOS splits.
- **▶ NEXT FRONTIER B — transient-soft-line thesis (Doug's).** We can't beat the sharp close, but can maybe catch soft/transient numbers. Two builds: (a) honest **market-shrunk break-even** — reliability diagram + shrink so `1/p_calibrated` is trustworthy (our big-edge disagreements are our OWN error; `nfl_edge_calibration.py` found NO reliable required-edge X*); (b) **high-frequency pre-game line logger** — poll DK/FD every few min through the pre-game window to detect transient dips vs consensus + measure live CLV. This is the only instrument that can SEE sub-hour staleness (our −12h/−4h/close snapshots can't). Backtest can only postulate; live logger verifies.
- **▶ PARKED lead — anchoring bias (Week-1 only).** Paper (Patterson/Shank/Fodor, SSRN 2025): fade the preseason-SB-anchored favorite ATS in Week 1. `nfl_anchoring.py` built; 2026 SB odds fetched (`calibration/nfl_preseason_sb_odds.json`). Week-1-only ≈ 16 games/yr → too thin to prove on 3-4 seasons; lean on the paper's 20-yr prior + forward-track. Best literature-backed lead we have.

## Open review findings
- **[OPEN, medium] NFL depth-chart 2025+ schema drift** — nflverse reshaped `load_depth_charts` to a dt-snapshot format (no week). `nfl_ingest.ingest_depth` is now schema-tolerant (stamps season, keeps union cols) but 2025+ is snapshot-based; a consumer that needs a 2025+ depth backup must pick the latest `dt` before the game date. Not blocking (the live QB projection uses recent-starter, not depth; depth is only the fallback). → [[data-and-architecture]].
- **[OPEN, medium] warehouse_mirror live-season staleness** (`warehouse_mirror.py`): `ensure()` fast-path returns instantly when all files `_valid` but never re-syncs the CURRENT season → backtests can read stale current-year data. MITIGATION: `--refresh-mirror`. FIX: exclude the live season from the `_is_valid` fast-path.
- **[RESOLVED] snap_counts layer** — ingested; `nfl_injury_impact` now uses snap share for all-position regular-detection.

## Ongoing monitors (MLB — parked/efficient, accumulate + watch)
- **⛔ Coherence run-line — PAUSED (dead on clean data).** Devigged bands + clean corpus → replication FAIL; the +11% was a raw-band artifact. Live `coherence_flags` cv_max=1.0 gate bets a −EV population → do not run. → [[edges-and-backtests]].
- **`batter_strikeouts UNDER 1.5` — the one clean-data MLB survivor** (+1.26% t=1.69, replicates; inverted public over-bias). Bet flat at DK/FD only when UNDER ≥ ~−110; small stakes, do NOT scale. → [[edges-and-backtests]].
- **⚠ cv_floor (earned_runs E + CV≥1.3) — SUSPECT.** Never re-validated on the clean corpus; re-run before any stake. → [[edges-and-backtests]].

## MLB — DONE + parked (tombstones; detail in domain files)
- **All MLB edges dissolved on the clean corpus (2026-09-05).** Steps 1-4 done: CLV, props recalibrated (batter_hits D+xBA(1.0)), team refit (well-calibrated, no edge), market-structure (coherence/under/f5 all DEAD). Lone survivor batter_K UNDER 1.5. → [[edges-and-backtests]] + [[modeling-and-calibration]].
- **Precise 5M-credit backfill + warehouse + mirror = DONE (2026-09-03).** Uniform −12h/−4h/close windows; backtests read the mirror 0-DTU. → [[data-and-architecture]].

## Parked calibration ideas (MLB, deferred while NFL active) → [[modeling-and-calibration]]
- value_gate auto-tune (advisory + `--promote`, never live-auto); EV floor 4%→3% for DK/FD; pitcher props seed Platt + exit synthetic-line fits; online Platt Blob-mode hardening (SQL-only today).
