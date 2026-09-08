# NFL work — handoff for independent review (2026-09-08)

You're being asked to give an **impartial second opinion** on the NFL modeling work in this
repo. The author (an AI assistant, "Cal") and the owner (Doug) are both deep in it and want
fresh eyes — especially on whether the conclusions are *sound* and the methodology is *honest*,
not just whether the code runs. **Please be adversarial.** The most useful thing you can do is
find where we fooled ourselves.

## Project in one line
A sportsbook betting model (MLB, now pivoting to NFL). Bets execute at DraftKings + FanDuel only;
all other books are analysis-only. Persistent project memory lives in **`memory/`** — read
`memory/MEMORY.md` (index), `memory/ACTIVE.md` (current state), and the domain files
`memory/modeling-and-calibration.md` + `memory/data-and-architecture.md` + `memory/edges-and-backtests.md`.
Those contain the full reasoning trail; this file is the review-oriented summary.

## What was built this session (NFL team model)
Clean-slate NFL data foundation on **nflverse** + a dedicated game-margin model, now **shipped live**.

Key files (all in `deploy/`):
- `nfl_schedule.py` — canonical `game_id` spine + fail-closed odds→game join (837/837 clean).
- `nfl_ingest.py` — offline nflverse→mirror-parquet ingest (all layers; pbp 2016-2025).
- `nfl_data.py` — dep-free mirror readers + `game_context()` (rest/bye/division/primetime).
- `nfl_epa.py` — team EPA (leakage-safe as-of), now reads the pbp mirror; `build_matchup_features`.
- `nfl_qb_asof.py` — starting-QB EPA layer (per-passer dropback EPA, starter projection).
- `nfl_injury_impact.py` — snap-based injury burden (offense EPA-out + defense/O-line counts).
- `nfl_model.py` — assembles + fits + predicts the live margin model.
- `nfl_totals.py` — total-points model (diagnostic, not shipped).
- `nfl_power.py` — opponent-adjusted EPA (**tested and rejected** — kept as documented negative).
- `nfl_market_scan.py`, `nfl_situational_scan.py`, `nfl_props_scan.py`, `nfl_edge_calibration.py`,
  `nfl_qb_edge.py`, `nfl_injury_edge.py` — the diagnostic/edge-hunt harnesses.
- `test_nfl_epa.py` — leakage + abbreviation-parity guards.

**Live model** (`calibration/americanfootball_nfl.json` → `starter_adjustment.nfl_margin_model`):
`margin = 2.13(HFA) + 32.5·net_epa_edge + 10.9·qb_edge + 0.065·off_inj + 0.65·def_inj + 0.35·rest`,
sigma 13.12. Wired via `nfl_epa.build_matchup_features` → `analysis._predict_margin`.

## The central claims (verify these)
1. **The model is leakage-safe** — all features use only data strictly before each game.
   (`test_nfl_epa.py` checks EPA; please also scrutinize the QB, injury, and rest as-of paths.)
2. **Matchup granularity does not beat the coarse aggregate OOS** — tested 7 ways (opponent
   adjustment; offense/defense split; pass/rush phase split; position-aware injuries; interaction
   terms; a regularized λ-blend; Pythagorean/points). All neutral-to-worse. Conclusion: aggregate
   net-EPA + HFA + QB + aggregate-injuries + rest is the robust optimum.
3. **We cannot out-predict the sharp closing line** — margin RMSE 13.19 vs close 12.67; win-prob
   Brier 0.226 vs market 0.211, and *worse on the games where we disagree* (the conditional test).
4. Therefore the remaining edge, if any, is in **transient soft lines** (not out-predicting the
   close) and/or **NFL props** (the MLB machinery was props-side), not more team-model features.

## Please scrutinize (our likeliest self-deceptions)
- **OOS is thin.** Odds-graded tests are **3 seasons only** (Odds-API limit); leave-one-season-out
  = 3 folds. The 10-season tests are pbp/margin-only (no odds). Are conclusions over-claimed for
  the power available? Multiple-comparisons across ~a dozen tests — is any "win" just noise?
- **"Game-actual starter" in backtests.** The QB/model validation used the starter who actually
  played (public pre-game, so we argue it's not outcome leakage). Is that defensible, or does it
  flatter the model vs the live *projected* starter? (Live path = `projected_starter`; the gap we
  measured was small: 13.123 game-actual → 13.155 projected.)
- **One test had a real sign bug** (`nfl_power`/phase-matched, first pass showed 14.33 due to a
  flipped defense-EPA sign; corrected to ~13.33). We caught and fixed it — but it means the test
  harness deserves a careful read. Are the *other* edge/RMSE comparisons sign-correct?
- **The live wiring.** `analysis._predict_margin` now branches for NFL to use `nfl_pred_margin`.
  We smoke-tested `_predict_margin` directly and claim MLB is byte-identical — verify the full app
  path (`app.py` ~3137) and the backtest/`feature_store` fit==serve caching aren't broken.
- **Fixed sigma = 13.12** drives every win probability (Brier). Is a constant sigma appropriate,
  or should it vary (weather, spread magnitude)? Our Brier-vs-market comparison depends on it.
- **Devig + closing-line RMSE/Brier** — we compare to the DK closing line devigged via
  `r2_sharp.fair_two_way`. Is the "closing line RMSE 12.67 / Brier 0.211" benchmark computed fairly?
- **The props scan** killed a candidate as "DNP survivorship" (scratched low-line players dropped
  from grading inflate OVER). Verify that reasoning + the drill in `nfl_props_scan.py`.
- **Calibration props section is a known toy** (`player_pass_yds` ≡ `player_rush_yds`, byte-identical)
  — not yet rebuilt.

## How to run (Windows, `PYTHONIOENCODING=utf-8`)
- Tests: `python test_nfl_epa.py`
- Margin model fit + OOS: `python nfl_model.py --seasons 2023,2024,2025`
- QB OOS lift: `python backtest_nfl_epa.py --seasons 2023-2025 --oos-qb`
- Beat-the-close: `python nfl_qb_edge.py --seasons 2023,2024,2025`
- Injury OOS + beat-close: `python nfl_injury_edge.py --seasons 2023,2024,2025 --beat-close`
- Required-edge curve: `python nfl_edge_calibration.py --seasons 2023,2024,2025`
- Totals: `python nfl_totals.py --seasons 2023,2024,2025 --beat-close`
- Spine audit: `python nfl_schedule.py --audit --seasons 2023,2024,2025,2026`

nflverse data pulls are FREE (no paid API). Odds-API pulls cost credits — do not run those without
Doug's explicit go. The mirror parquets are in `warehouse_mirror_data/` (Git LFS).

## Bottom line we reached (challenge it)
The NFL team model is genuinely good and at its noise-limited ceiling (~0.5 RMSE behind the sharp
market, which is irreducible from box-score data). No amount of matchup cleverness closed the gap
(7 tests). This is offered as a *robust* finding — if you can break it, that's exactly the value
of this review.
