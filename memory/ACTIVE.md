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
- **prop_roi RUN (2026-08-28) → over-bias confirmed but vig-eaten; 1 marginal cell.** Verdict in [[edges-and-backtests]]: only `batter_strikeouts UNDER 1.5` (+2.2%, t=1.93, replicates) crosses +EV; ER UNDER 2.5 breakeven; everything else vig-negative. OPEN owner decision: forward-track K-under-1.5 small (borderline, clears the coherence t≈1.66 precedent) or record + move on. Don't build a system on it.
- **Line-timing / CLV-decay (Doug's pick 2026-08-28) — STEP 1 probe BUILT (e719e46), awaiting run.** `scenario_backtest.py --scenario line_timing`: an EXECUTION-edge study (wait vs bet-now) to multiply the edges we already bet. FEASIBILITY GATE FIRST — the probe reports whether the warehouse has a real early→close price path per event; memory's prior is close-only (team) / tight-closes (props) → likely INFEASIBLE offline, needing Odds-API credits. If the "early+close path %" is meaningful → build the full CLV/ROI-by-lead-time study offline; if ~0 → it's a paid backfill decision (get owner spend-confirm first). Other parked directions: upset datamine, bet-disqualification, sharpen coherence w/ stable-fav-SP filter. Monte Carlo = no-go on current evidence.

## [[modeling-and-calibration]]
- **value_gate auto-tune** — fold a confirmation-gated `--gate-diag` selection into the offline refit as advisory + `--promote` (never live-auto; threshold argmax overfits vs smooth Platt). `refit_calibration` still never calls `save_value_gate`.
- **EV floor 4%→3% for DK/FD** — `value_gate.ev_floor` is still 0.04 (chosen on optimistic consensus prices); A/B 0.04 vs 0.03 on DK/FanDuel fills via `--gate-diag`.
- **Pitcher props: seed Platt + exit synthetic-line fits** — grow pitcher REAL-line obs so they leave synthetic-line fits, and seed/enable online Platt for pitcher props (zero fits today) → closes the forward-Brier gap.
- **Online Platt Blob-mode hardening (low pri)** — seed-pristine invariant + per-key merge are SQL-only; mirror per-key merge into the Blob branch + make save guard mode-agnostic before any non-SQL deploy.
