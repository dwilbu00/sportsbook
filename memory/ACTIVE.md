---
name: active
description: In-flight work only — what's being worked on now, linking to the domain files. Move an item to its domain when done; do not let this file accrete history.
metadata:
  type: project
---

# ACTIVE — in-flight work

One line per open item. When an item ships, delete it here and fold the durable outcome into its `[[domain]]` file (result-supersedes-idea). Last reconciled 2026-08-28 (verified against code + owner-confirmed prod state).

## Ongoing monitors (not tasks — just accumulate + watch)
- **Coherence run-line forward-track** — the in-app forward log (`runline_coherence`, dog +1.5 at [0.60,0.70) fav band) is the confirmation instrument for the marginal t≈1.66 edge. Bet small, let the sample grow. [[edges-and-backtests]]
- **cv_floor forward-monitor** — LIVE (earned_runs method E + cv_floor 1.3); watch for decay + confirm DK/FanDuel book volatile SPs at size before scaling. [[edges-and-backtests]]

## [[edges-and-backtests]]
- **(next direction — open)** Both edge theses (sharp-staleness, variance→team) are now spent; the coherence run-line is the lone validated edge (forward-tracking). Options if resuming the hunt: sharpen coherence with a stable-favorite-SP filter (forward-track, don't backtest-ship), soft/niche markets, NBA/NFL (data-gated), or consolidate + operate. Monte Carlo sim = no-go on current evidence.

## [[data-and-architecture]]
- **Auto-create/refresh the parquet mirror from the backtest tools — QUEUED (Doug: do it once he verifies the manual sync/verify/run flow works).** Add `warehouse_mirror.ensure(sport, seasons, refresh=False)` (sync only MISSING files, or all if refresh); each backtest `load_or_fetch` calls it when `ODI_BACKTEST_MIRROR=1` so the manual `--sync` becomes optional (first flagged run auto-builds). Add a `--refresh-mirror` flag per tool (distinct from `--refresh` = pickle cache). Needs Azure to build; no-DB boxes still need a pre-synced/copied dir.

## [[modeling-and-calibration]]
- **value_gate auto-tune** — fold a confirmation-gated `--gate-diag` selection into the offline refit as advisory + `--promote` (never live-auto; threshold argmax overfits vs smooth Platt). `refit_calibration` still never calls `save_value_gate`.
- **EV floor 4%→3% for DK/FD** — `value_gate.ev_floor` is still 0.04 (chosen on optimistic consensus prices); A/B 0.04 vs 0.03 on DK/FanDuel fills via `--gate-diag`.
- **Pitcher props: seed Platt + exit synthetic-line fits** — grow pitcher REAL-line obs so they leave synthetic-line fits, and seed/enable online Platt for pitcher props (zero fits today) → closes the forward-Brier gap.
- **Online Platt Blob-mode hardening (low pri)** — seed-pristine invariant + per-key merge are SQL-only; mirror per-key merge into the Blob branch + make save guard mode-agnostic before any non-SQL deploy.
