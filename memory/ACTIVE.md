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
- **Line-timing / CLV-decay (Doug's pick 2026-08-28) — probe RUN + STUDY REWRITTEN (30faed3), awaiting valid run.** Feasibility CONFIRMED for TEAM markets (25% / 1,504 events have an early→close path); PROPS too thin (7%/231) → skip. First study run had a BUG (fixed 30faed3): it picked ONE close snapshot per event for all markets, but the near-commence capture is ML-only → totals/RL close rows were empty and `summarize([])` faked +0.00%. Only VALID finding from that run = ML: **early lines softer, bet EARLY beats close ~2% both sides** (fav_ml ΔROI +2.03% +++, dog_ml +1.84% +++; dog_ml early +1.61% is +ROI). Rewrite: each side-category independently picks CLOSE = latest pre-commence snapshot CARRYING that market + EARLY = earliest ≥6h before it, both-or-neither paired, honest paired-n (n=0 ≠ 0% ROI; `no_early:` coverage counter). **rl_dog_+1.5 = the coherence edge's side → the direct execution answer.** RE-RUN: `python scenario_backtest.py --scenario line_timing --sport mlb --seasons 2024 2025 2026` (2023 purged; 2024/25 bulk-backfilled early+close, 2026 live rolling captures). VERDICT RULE: ΔROI>0 replicating → bet EARLY; <0 → wait. Other parked directions: upset datamine, bet-disqualification, sharpen coherence w/ stable-fav-SP filter. Monte Carlo = no-go.

## [[modeling-and-calibration]]
- **value_gate auto-tune** — fold a confirmation-gated `--gate-diag` selection into the offline refit as advisory + `--promote` (never live-auto; threshold argmax overfits vs smooth Platt). `refit_calibration` still never calls `save_value_gate`.
- **EV floor 4%→3% for DK/FD** — `value_gate.ev_floor` is still 0.04 (chosen on optimistic consensus prices); A/B 0.04 vs 0.03 on DK/FanDuel fills via `--gate-diag`.
- **Pitcher props: seed Platt + exit synthetic-line fits** — grow pitcher REAL-line obs so they leave synthetic-line fits, and seed/enable online Platt for pitcher props (zero fits today) → closes the forward-Brier gap.
- **Online Platt Blob-mode hardening (low pri)** — seed-pristine invariant + per-key merge are SQL-only; mirror per-key merge into the Blob branch + make save guard mode-agnostic before any non-SQL deploy.
