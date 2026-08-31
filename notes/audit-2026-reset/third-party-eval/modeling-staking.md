# Third-party audit eval — Cluster: Modeling & staking enhancements (P12–P18)

Evaluator pass date: 2026-08-10. Repo cwd = `deploy/`. Every current_state claim below is grounded in
file:line evidence read this session. Framing: the third-party auditor had NO live DB and explicitly
flagged that the app "has already undergone a substantial audit-driven cleanup," so each claim was
treated as a hypothesis and verified against code, not accepted.

Philosophy constraints applied throughout: Brier-first calibration gate (min gain 0.002); features ship
ONLY after chronological forward validation (never on intuition); DK-only for prices/recommendations
(other books for analysis only); leakage-safe as-of features; conservative rejection of enhancements
that fail validation; no wholesale ML replacement; keep market blending.

---

## P12 — Empirical pairwise correlation → shrink toward heuristic prior → feed the copula

**current_state: PARTIAL (copula machinery done; correlation source is heuristic constants; empirical
estimation absent).**

Evidence:
- Gaussian copula + Cholesky + PSD shrinkage + degradation reporting all exist in `parlay.py`:
  `_cholesky` (34), `_make_psd_cholesky` (54, shrinks off-diagonals toward 0 until PSD, returns
  `applied_shrink`), `_gaussian_copula_joint_prob` (92), `_copula_joint_hit_prob` (461),
  and the surfaced `correlation_shrink`/`correlation_degraded` fields (871–872). The auditor's list of
  "already includes" (copula, Cholesky, PSD check, shrinkage, degradation reporting, SGP payout
  protection) is accurate.
- The SGP payout protection they credit is real: `_parlay_value_joint` (19) prices a same-game parlay
  against the INDEPENDENT joint so the copula's positive-correlation benefit is never credited as EV
  against the naive product payout (comment 812–833; gate 841).
- The remaining weakness they name is real: pairwise correlations are hard-coded heuristic constants in
  `_pair_correlation` (parlay.py:348–446) — e.g. MLB same-pitcher ER↔K = −0.35 (423), K-over↔total-under
  = +0.35 (424–429), ML↔batter_hits = +0.25 (436–443); generic same-game = +0.05 (446). No estimation
  from historical outcomes exists anywhere (grep for empirical correlation / covariance → only audit
  text).
- Note the constants are NOT parlay-only cosmetics: `bet_selector._pair_conflict` (bet_selector.py:198–205)
  blocks any co-selection whose `parlay._pair_correlation(...) <= -0.20` (`_CORR_BLOCK`, line 33). So the
  heuristic constants drive real slate composition (which single bets are co-picked), the safe-mode joint
  hit-probability ranking (`_score_parlay` safe branch, parlay.py:590; stage-2 `joint*1000`, 748), and the
  displayed `correlation_adjustment_pct`.

**verdict: adapt (and defer the build).** The idea is sound and the design the auditor sketches
(empirical → shrink toward the heuristic prior → copula) is exactly the right shape and matches the
codebase's Bayesian-shrinkage idiom (cf. `PLAYER_PROP_MARKET_PRIOR_K`, props.py:274–286). BUT the
realizable edge is narrower than the auditor implies, for two codebase-specific reasons:
1. Under DK-only + SGP-neutralization, positive correlation is NEVER credited as EV (SGP priced vs the
   independent joint; cross-game legs are forced ρ=0 at parlay.py:361–363). So better positive
   correlations do not unlock parlay value the book won't pay. The impact is confined to: (a) the L2
   anti-correlation *suppression* thresholds (getting the negative ρ's right so we neither wrongly block
   an independent pair nor co-select a truly self-cancelling one), (b) safe-mode joint hit-probability
   ranking, (c) displayed figures.
2. Estimating a pair like P(batter hit OVER ∧ team ML) requires many co-occurring GRADED outcomes in the
   same game. `prediction_log` (props) and `market_prediction_log` (team) hold the raw material, but
   per-pair volume is thin today — the same "blocked on data accrual" wall the roi/feature-diag lenses
   already hit (refit_calibration.py:1836).

Deltas if/when built: estimate ρ̂ per (sport, market-pair-type) from joined graded outcomes; Bayesian
blend ρ_used = (n·ρ̂ + k·ρ_prior)/(n+k) with ρ_prior = the current constant and k in pseudo-pairs;
leakage-safe (only pairs from games strictly before the projected game); forward-validate that the
empirical thresholds don't degrade slate ROI before flipping them on. Reuse `_pair_correlation` as the
prior lookup so there is ONE correlation source.

**integration: backlog** (a new "empirical-correlation" workstream when triggered; no current WS fits).
**effort: M–L.**

**Trap:** do not let empirical ρ leak into SGP EV crediting — the SGP neutralization in
`_parlay_value_joint` must stay. And do not estimate ρ from a bettor's own logged parlays (none exist,
and it would be circular).

---

## P13 — Portfolio-level Kelly / correlation haircut across simultaneously-held bets

**current_state: absent (per-bet Kelly + per-bet cap + proportional slate cap exist; no correlation
awareness in sizing).**

Evidence:
- Sizing is strictly per-leg fractional Kelly: `pricing_common.kelly_fraction` (102) = `f*·fraction`
  clamped to `cap`; `kelly_stake` (130) = `bankroll·f`. Slate control is `scale_to_slate_cap` (145) — a
  purely PROPORTIONAL shrink of the batch to ≤ `cap_fraction·bankroll`, with NO covariance term.
- The whole batch is sized independently then proportionally capped: `app.py::_selected_kelly_rows`
  (443–496) builds each row via `wagers.build_wager_row` in Kelly mode (238) then applies
  `scale_to_slate_cap` (490). Durable bankroll ledger + persisted knobs (half-Kelly, 5% per-bet, 25%
  slate) confirmed in `bankroll.py` (`_KELLY_SETTING_DEFAULTS`, 51–55) and `app.py` (`_KELLY_DEFAULTS`,
  409–413).
- No covariance/portfolio/haircut logic anywhere (grep covariance|portfolio|haircut|correlation.*kelly →
  only audit text + an unrelated SGP score comment). Matches the work-log note "NO correlation haircut /
  portfolio Kelly yet."

**verdict: adapt.** Genuinely new and fits the conservative philosophy: independently-sized correlated
bets over-bet the true joint Kelly, inflating variance and risk-of-ruin — precisely what half-Kelly +
caps are there to guard. The auditor's "simpler first implementation" (reduce individual Kelly by shared
exposure to same game/team/player/correlated-market-group) is the right scope; a full Markowitz optimizer
is not warranted. Reuse the EXISTING correlation source (`parlay._pair_correlation`) so there is one rule
set: compute a per-bet haircut = f(sum of positive ρ to already-selected legs), applied BEFORE
`scale_to_slate_cap`. This connects the correlation knowledge the app already has to the bankroll layer.

Caveat on ROI impact: this reduces *variance/drawdown*, not expected ROI — it is risk management, not an
edge finder. Partial protection already exists (per-bet 5% + slate 25% + half-Kelly), so the marginal
safety is real but bounded. Rank it above P12/P14 for value-per-effort because the correlation source is
already built and it touches real money.

**integration: new WS "correlation-aware staking" (or fold into a staking-hardening WS).**
**effort: M.**

**Trap:** keep it a *haircut* (never an *increase*) so it can only ever reduce exposure — a fail-safe
default. Do not double-count the slate cap; apply haircut first, then the existing proportional cap.

---

## P14 — Edge uncertainty: interval around estimated edge; gate on P(true edge > 0)

**current_state: absent.**

Evidence: no interval/uncertainty/`P(true edge>0)` machinery (grep edge_uncertainty|edge_interval|
P(true edge → only audit text). The value gate is a point-estimate double condition:
`pricing_common._prop_is_value` (167) requires `edge >= threshold AND expected_roi > 0`; team markets
use the same de-vig fair edge (`_devig_fair`, 185). Confidence today is implicit, folded into calibration
(Brier gate), `prob_shrink` overconfidence correction (`_apply_shrink`, 240), and market-as-prior
shrinkage (props.py:274–286).

**verdict: defer.** Philosophically aligned (conservative, confidence-aware) but weakest value-per-effort
in the cluster: (1) it depends on P15 (N_eff) as its primary input; (2) much of the "small-sample edges
are unreliable" concern is ALREADY handled by market-prior shrinkage (thin samples get pulled to the
market) and by min-obs gates; (3) per the philosophy, a `P(true edge>0)` gate must itself be shown to
beat the current point-estimate gate under chronological forward validation — it cannot ship on intuition,
so it needs a harness pass, and there is real risk it adds complexity without moving ROI.

Trigger to revisit: after P15 N_eff exists as a first-class quantity AND a no-write lens (like
--roi-diag/--feature-diag) shows an interval/`P(edge>0)` gate improves holdout ROI or Brier vs the current
gate.
**integration: backlog (depends on P15).**
**effort: L.**

---

## P15 — Effective sample size N_eff = (Σw)² / Σw² into confidence/gates/diagnostics

**current_state: partial — a *proxy* is used but it's the wrong estimator, and it's not surfaced.**

Evidence (this is the sharpest concrete finding in the cluster):
- The props projection already computes an "effective n" for Bayesian shrinkage, but it uses **`eff_n =
  sum(weights)`**, not the Kish N_eff: props.py:1125 `eff_n = sum(weights) if weights else 0.0`, then
  1126–1127 shrinks `base_proj = (eff_n·base_proj + k·unweighted_mean)/(eff_n + k)`
  (comment 260–264).
- The weights are UN-normalized exponential decay: `stats._recency_weights` (76–93) returns
  `exp(-decay·i)` (weight[0]=1.0, strictly decreasing, unequal). For geometric weights with ratio
  r=e^-decay, `sum(w) → 1/(1-r)` but Kish `N_eff = (Σw)²/Σw² → (1+r)/(1-r)`. So the current proxy
  **overstates** independent information by up to ~2× at long half-lives — exactly the failure mode the
  auditor names ("raw [weighted] count overstates the actual amount of independent information"). Using
  the true N_eff would shrink recency-weighted samples MORE toward the prior.
- N_eff is not reported anywhere (grep N_eff|effective sample → only audit text + this `sum(weights)`
  usage). The similar team paths (`analysis.py` `_recency_weights` at 108/265/389/441) have the same
  weight structure and no N_eff surface either.

**verdict: adopt (split into two moves of different risk).**
1. **Diagnostic surface (S, safe, no gate):** compute `N_eff = (Σw)²/Σw²` wherever recency weights exist
   and surface it beside `games_sampled` in prop/team diagnostics and the calibration lenses. This is a
   pure add — no projection change — so it needs no forward validation. It's also the cheapest way to make
   the "20 games observed / 8.7 effective" story the auditor wants concrete, and it's the natural input for
   P14.
2. **Shrinkage-denominator fix (M, gate-gated):** replace `eff_n = sum(weights)` with the Kish N_eff at
   props.py:1125. This CHANGES projections, so per philosophy it MUST go through the existing backtest gate
   (backtest.py props sweep), NOT be hand-swapped. It is correctly signed (more shrinkage for volatile
   thin/decayed samples) and low-risk, but "correct in theory" is not a ship criterion here.

Value: modest direct ROI (better min-data behavior / fewer thin-sample false edges), high value-per-effort
for move (1), and it's a foundation P14 needs. Genuinely useful; the diagnostic half is essentially free.

**integration: WS4 (like-for-like scoring/diagnostics) for the surface; the shrinkage fix rides the
existing props gate (feature/backtest harness).**
**effort: S (diagnostic) + M (validated shrinkage swap).**

**Trap:** don't silently swap the shrinkage denominator "because it's more correct" — that would violate
the never-ship-on-intuition rule the whole app is built on. Route it through the gate like any projection
change.

---

## P16 — Expand the feature-gating framework with new MLB/NBA candidates (through the existing gate)

**current_state: already-done (framework); ongoing (candidates).**

Evidence:
- `prop_features.py` is a real single-source registry (`FEATURE_REGISTRY`, 170–210) with `rest`,
  `gamecontext`, `platoon` tenants, each a bounded strength-scaled multiplier with leakage-safe as-of
  inputs and strength-0 byte-parity no-ops.
- It is genuinely consumed by all THREE projection paths (so runtime/synthetic/real-line can't drift, as
  the auditor praises): synthetic sweep `backtest.py:2662–2665`; offline real-line
  `book_line_calibration.py:800–803`; runtime `props.py:1247`. Helpers `strengths_from_params` (223),
  `projection_multiplier` (242), `feature_applies` (215). The gate itself lives in
  `refit_calibration.diagnose_features` (1826) with the 2-fold + `MIN_CALIB_BRIER_GAIN` confirmation and a
  consensus-ROI co-signal.
- The auditor's candidate lists are partly already-tested/registered (days rest = `rest`; the work-log
  shows platoon/gamecontext/rest all ran the gate and ship nowhere yet) and partly untested (expected PA,
  bullpen handedness composition, pitch-type matchup, projected pitch count, opponent chase/contact; NBA
  minutes/pace/usage-redistribution/implied-team-total/B2B/starting-status).

**verdict: adopt — as an ongoing program, NOT a build.** This is the app's ONLY proven mechanism for
adding real predictive edge (every shipped feature came through it), so per validated feature it has the
highest ROI potential in the cluster — but the framework is done and history shows most candidates FAIL
the gate, so the correct reading of P16 is exactly the caller's: "use the harness," add candidates through
the gate, never on intuition. No philosophy conflict (it IS the philosophy). NBA candidates are especially
worth queuing but are blocked on NBA season data + a current NBA calibration (cross-sport-parity memo).

**integration: a standing "feature-candidate backlog" (feeds the existing gate); no new plumbing.**
**effort: M per candidate (mostly the offline feature attach + a diag run).**

**Trap:** don't let the volume of candidate ideas pressure a lower gate. The gate's conservatism (0.002
Brier, 2-fold, min-obs incumbent protection) is the asset.

---

## P17 — Treat market info as first-class features (cross-book dispersion, line movement, time-to-start)

**current_state: partial — market blending exists; market STRUCTURE features (dispersion/movement/TTS)
are absent as model inputs, and the durable warehouse collapses the per-book granularity dispersion needs.**

Evidence:
- Blending model prob with de-vigged market prob exists and is calibrated per market:
  `pricing_common._blend_weight` (246) reads `market_blend` `w`; props edge is measured vs de-vigged
  consensus; `_devig_fair` (185); market-as-prior shrinkage (props.py:274–286). So the auditor's "already
  blends" is accurate.
- Cross-book dispersion / line movement / time-to-start are NOT features (grep dispersion|line_movement|
  time_to_start|cross_book → nothing in the model paths; the one hit,
  `backtest_market_consensus.py`, is a research STUDY of whether peer-consensus finds value at DK, not a
  shipped feature, and it de-vigs each peer book then takes the median — no dispersion signal retained).
- Data reality check: the durable warehouse does NOT preserve per-book prices. `odds_line` stores ONE
  extracted price per line — "best-across-books for team; consensus for props" (schema.sql:333–335;
  db_store.py:367). Per-book granularity (needed for dispersion) is collapsed at capture. `odds_snapshot`
  keeps only a comma-joined `bookmakers` string (schema.sql:313) and is "write-once per hour bucket"
  (306, 315) — so multiple hourly snapshots per event CAN support line-movement research, but the raw
  per-book prices survive only in odds_client's immutable historical CACHE (which
  backtest_market_consensus.py already reconstructs from), not in the queryable warehouse.

**verdict: adapt — and route through the P16 gate, DK-only-safe.** This is philosophically clean:
cross-book *disagreement* is a SIGNAL (analysis), not a PRICE — DK stays the only executable/recommended
price, so using FanDuel/MGM/Caesars dispersion as a feature does NOT violate the DK-only rule (explicitly
flagging this because it is the obvious place a reader might think it does). The right framing is not "add
market features now" but "make dispersion/movement/TTS candidate features that must pass the same
chronological gate as any other" — i.e. P17 is a specialization of P16.

Concrete prerequisite delta: to feed dispersion offline/leakage-safe, either (a) persist per-book prices
(or a precomputed dispersion stat) in `odds_line` going forward, or (b) reconstruct from the odds_client
historical cache the way backtest_market_consensus.py does. (a) is the warehouse-principle-aligned choice
if this graduates past research.

**integration: P16 feature backlog (as market-structure candidates); a small warehouse delta if
persisting per-book dispersion.**
**effort: M (research via gate) + S–M (optional odds_line dispersion column).**

**Trap:** never let a non-DK book's PRICE become a recommendation input — only its dispersion/movement as a
feature. Keep `executed_price` DK-only (the CLV/team invariant already depends on this).

---

## P18 — Elevate CLV to a primary model diagnostic alongside Brier/ROI, per model/version

**current_state: partial — wager-level CLV done; model-level per-version CLV diagnostic absent.**

Evidence:
- Wager-level CLV is real and DK-vs-DK: `wagers.apply_clv_updates` (599) writes
  close_price/close_line/clv_pct; `backfill_dk_clv.py` is the sole offline filler (wagers.py:8, 573–606);
  the warehouse render-time path was retired (app.py:1785–1790). It's already aggregated as a diagnostic:
  `wagers._metrics` computes `avg_clv_pct` (683, 698) and `summarize_wagers` splits it by sport and by
  bet-type/prop (740–749); the My Bets page shows per-bet CLV (app.py:1922–1923).
- The GAP the auditor targets: CLV is NOT a MODEL-evaluation metric. The model-selection/diagnostic
  pipelines score on Brier (+ consensus-priced ROI co-signal) and explicitly note true DK CLV is "blocked
  on data accrual": refit_calibration.py:1836; the roi/feature-diag lenses print ROI beside Brier but not
  CLV. There is no per-model/version CLV because there is no model_version/git_sha/calibration_version on
  predictions at all (grep model_version|git_sha|calibration_version → NO matches; prediction_log schema
  14–47 has none) — that's the P3 provenance gap (another cluster), and P18's "per model/version" clause
  DEPENDS on it.

**verdict: adapt (near-term, no per-version) + defer (the per-version split).** Keep Brier PRIMARY — the
philosophy is explicit and the work-log's own audit concluded deployed-A already shows +CLV/+ROI so
Brier-first should stay. The valuable, philosophy-consistent move is to add CLV (avg CLV + CLV hit-rate +
n_bets) as a co-reported diagnostic ALONGSIDE Brier/ROI on the model-eval surface (the same lenses that
already print ROI), so a run of unfavorable outcome variance can be distinguished from genuine model
decay. The per-MODEL/VERSION breakdown the auditor wants is blocked twice over: (1) data accrual (few
graded DK-vs-DK closes), (2) missing prediction provenance (P3). Do the aggregate CLV panel when volume
justifies; defer the version split until P3 lands.

**integration: WS4 (like-for-like forward/backtest scoring) for the aggregate CLV panel; the per-version
split is a dependent follow-on to the P3 provenance workstream.**
**effort: M (aggregate panel), blocked on data + P3 for per-version.**

**Trap:** don't let a thin-sample CLV number override the Brier gate — CLV is a co-signal, Brier decides
(same discipline as the existing ROI tiebreak, which only fires inside the Brier noise band).

---

## Ranking by expected value-per-effort (cluster-internal)

1. **P16 (adopt, ongoing)** — the only proven edge-adder; framework done; keep feeding the gate. Highest
   ROI potential per validated feature, but disciplined/incremental.
2. **P15 (adopt)** — cheap, foundational, fixes a real `sum(weights)`-vs-Kish overstatement; diagnostic
   half is free; shrinkage fix rides the gate. Best pure value-per-effort for a concrete change.
3. **P13 (adapt)** — genuine gap, reuses the existing correlation source, touches real money (variance/
   risk-of-ruin). Risk-mgmt not ROI, but high leverage for M effort.
4. **P18 (adapt/defer)** — mostly done at wager level; aggregate model-CLV panel is worthwhile; per-version
   blocked on data + P3.
5. **P12 (adapt/defer)** — copula done; empirical ρ is real but its impact is narrow under
   DK-only/SGP-neutralization and thin on data. Defer to warehouse growth.
6. **P14 (defer)** — aligned but complex, depends on P15, uncertain ROI, must beat the current gate under
   validation.

**What genuinely moves ROI/CLV vs nice-to-have:** Only P16 (and P17-as-P16-candidates) can *add edge*.
P13 improves risk-adjusted return (drawdown), not raw ROI. P15/P18/P12/P14 are accuracy-of-belief and
diagnostic refinements — valuable for discipline and for feeding future decisions, but not direct ROI
movers. Nothing here justifies relaxing the Brier-first / forward-validation / DK-only discipline; the
correct posture for all six new-ish ideas is "route through the existing gate / lenses," which is precisely
the app's established strength.

---

## Verifier verdict (adversarial re-check, 2026-08-10)

Every code citation in the memo was re-opened and spot-checked against the actual source. **All
file:line evidence is accurate** (P12 `_pair_correlation` constants + copula machinery + SGP
neutralization; P13 `kelly_fraction`/`scale_to_slate_cap`/`_prop_is_value`; P16 `prop_features`
registry consumed by all three paths + `diagnose_features` gate; P17 `odds_line` collapses per-book to
"best-across-books for team; consensus for props" at schema.sql:333-335; P18 `apply_clv_updates` +
`avg_clv_pct` + no `model_version`/`git_sha`/`calibration_version` anywhere in code; P14 absent). Six of
seven findings are **CONFIRMED as written** — verdicts and integration stand: P16 adopt-as-program, P13
adapt (genuine gap; risk-mgmt not ROI; bounded marginal benefit given existing half-Kelly + 5%/25%
caps), P18 adapt/defer, P12 adapt/defer, P17 adapt-as-P16-candidate, P14 defer.

**One material correction — P15's central math is inverted.** The code facts are right
(`eff_n = sum(weights)` at props.py:1125; un-normalized `exp(-decay*i)` weights with w[0]=1.0; N_eff not
surfaced). But the *direction* is backwards. For geometric weights (ratio r=e^-decay):

  sum(w) = 1/(1-r)   <   Kish N_eff = (Σw)²/Σw² = (1+r)/(1-r)   <   raw n

Numerically (MLB hl=7): n=15 → sum(w)=8.21, **Kish=12.75**, raw=15; n=20 → sum(w)=9.14, **Kish=15.31**.
Kish is 1.29–1.68× LARGER than sum(w), not smaller. So:

- The memo's "current proxy **overstates** independent information by ~2×" is wrong — sum(w)
  **understates** the Kish N_eff. sum(w) is the *smallest* of {sum(w), Kish, n}; there is no quantity it
  overstates here. The auditor's "raw count overstates" concern is about raw **n** vs N_eff, and the code
  doesn't use raw n — it uses the already-smaller sum(w).
- "Using the true N_eff would shrink recency-weighted samples **MORE** toward the prior" and "correctly
  signed (**more** shrinkage)" are both reversed. Kish > sum(w) ⇒ higher pseudo-count ⇒ **LESS**
  shrinkage ⇒ the projection trusts thin/decayed recency samples MORE. The swap is a **loosening**, not a
  conservative tightening, and the "fewer thin-sample false edges" ROI story is likely reversed (it could
  produce *more* thin-sample edges).

Consequence for the verdict: the **diagnostic-surface half stays a clean adopt** (free, no projection
change, correct precision measure, natural P14 input). The **shrinkage-denominator swap is downgraded in
enthusiasm** — it is speculative and mildly *anti*-conservative, so it must earn its place through the
existing props gate with no presumption that it's "more correct = safer." Keep it gate-routed (the memo's
own trap note is right), but do not sell it as a conservative correction or rank it as a near-free win.
Overall P15 remains adopt (for the diagnostic) but the rationale is partial.

Note this does not weaken P14: Kish N_eff is exactly the correct precision denominator for an
edge-uncertainty interval, so P14's stated dependency on P15 is if anything reinforced — P14 stays defer.

No finding was found over-eager enough to reject, and none was wrongly rejected/deferred (nothing to
upgrade). No DK-only / Brier-first / fail-closed conflicts introduced by any adopt/adapt. P17's DK-only
framing (non-DK dispersion is a signal, never a price) is correct and worth keeping explicit.
