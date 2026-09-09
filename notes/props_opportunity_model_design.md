# Props opportunity-first modeling — design (MLB TB → RBIs; template for NFL props)

Status: DESIGN (2026-09-09). Approved direction: build opportunity-first distributional
models for the context-heavy MLB batter props, using the validated method-D machinery
(`props._dist_p_over`, now smoothed via the adjacent-count mixture) as the template.
Ignore BA-RISP (noise / non-repeatable — the "clutch" fallacy; regresses to overall BA).

## Why (the recurring lesson, applied)
Current methods leave signal on the table: `batter_total_bases`=E (NegBin on the count,
no opportunity structure), `batter_rbis`=A (raw over-rate passthrough, ignores EVERYTHING).
Both outcomes are opportunity-driven. The two-stage structure — model OPPORTUNITY, then the
CONDITIONAL outcome distribution — is already proven on `batter_hits` (method D = expected_AB
× per-AB hit prob → binomial). Port it. Honest caveat: better prediction ≠ betting edge (the
market prices lineups); but props are the recreational market where MLB's lone edge lived, and
either way it's a better-calibrated prediction. Validate OOS vs the incumbent (E/A); switch only
if it clears the real-line Brier gate. No RISP split (deliberately excluded — noise).

## Build 1 — Total Bases (clean proof-of-concept, NO lineup dependency)
Structure:  TB = sum over AB of bases-per-AB.
  opportunity  = expected_AB (have: `_lineup_exposure_mult` / batting-order exposure)
  conditional  = per-AB bases distribution from power/contact (have: xwOBA, barrels, hard-hit;
                 optionally add xSLG = estimated_slg_using_speedangle — NOT currently trimmed,
                 a data add if xwOBA/barrels prove insufficient)
  P(TB >= line) = mixture over ⌊m⌋/⌈m⌉ expected AB (reuse the smoothed opportunity primitive),
                 each AB drawing a bases outcome {0,1,2,3,4} from a per-AB multinomial fit to
                 the batter's power profile.
Validation: real-line Brier vs incumbent E, OOS, via `refit_calibration --real-lines`-style gate.

## Build 2 — RBIs (adds the runners-on context layer on Build-1 machinery)
Decompose:  E[RBI] ≈ E[HR]  +  E[runners-on-when-batting] × P(drive-in | runner, power)
  self-RBI     = HR rate × expected_AB (have: HR from stats/statcast)
  runners-on   = f(preceding batters' OBP, this batter's lineup slot)  ← NEW FEATURE to build:
                 from the posted lineup (batting_order + the 8 teammates), mean OBP of the
                 slots batting AHEAD of this batter. Needs the confirmed lineup (have via
                 mlb_starters) + per-player OBP (derivable from stats).
  drive-in     = power/contact conversion (xwOBA/barrels/xSLG — same as TB)
  P(RBI >= line) = distribution combining self (HR) + drive-in (runners × conversion).
Validation: same OOS real-line Brier gate vs incumbent A.

## Data inventory (what exists vs to-build)
- HAVE: batting_order, expected-AB-by-slot exposure, confirmed-lineup gating, xwOBA/barrels/
  hard-hit (per-pitch statcast), HR (stats), the smoothed `_dist_p_over` opportunity primitive.
- TO BUILD: preceding-batters'-OBP feature (lineup-order × teammate OBP); a per-AB bases/RBI
  conditional distribution (vs the current binary hit-or-not for hits).
- OPTIONAL data add (only if needed): xSLG (estimated_slg_using_speedangle) into the statcast
  trim (schema bump + re-ingest). NOT needed to start (xwOBA/barrels carry power).
- EXCLUDED: BA-RISP (base-state on_2b/on_3b not trimmed AND it's a non-repeatable clutch split).

## Sequence + gate
TB first (no lineup dep → validates the opportunity×conditional-distribution machinery cheaply),
THEN RBIs (adds preceding-OBP on proven code). Each ships OFF, activated per-prop only if it
beats the incumbent on OOS real-line Brier (the existing method-selection gate). This is the
MLB template that then ports to NFL props (receptions = targets × catch-rate; rush_yds =
carries × yds/carry) — item 3.
