---
name: bonus-strategy
description: The bonus/promo +EV engine — the project's ONE real edge. The thesis (boost flips high-P bets +EV on market-devig legs), the backtest that validated it, SGP correlation, the shipped system (engine/optimizer/tracker + app pages), the schema, and the harvest/bugfixes.
metadata:
  type: project
---

**THE STRATEGIC CORE (Doug 2026-09-15).** We do NOT beat the book on straights (MLB + NFL both efficient-for-us — see [[edges-and-backtests]]). The one real +EV mechanism = the book's OWN promos. A profit boost pays on the PAYOUT, so it flips high-probability −EV bets +EV (30% boost: −110 coin-flip → +9.1%; 3-leg 52% parlay → +23% vs −2.2% unboosted). Boosts target PARLAYS/SGPs (books assume longshots) → with ACCURATE high-P legs they flip decisively +EV. This is WHY calibrated prediction mattered all along (over-confident P = fake +EV).

## ★★ The load-bearing rule: leg P = the book's DE-VIGGED price, NOT our model
The backtest (`nfl_bonus_backtest.py`, 2023-25 DK+FD) proved our MODEL P is OVER-CONFIDENT in the high-P tail we bet (calls 80%→wins 61%; ECE 0.063) — aggregate Brier hid it. `--prob model` inflates predEV to fiction + the selection-curse compounds per leg. `--prob market` (devigged, ECE 0.008-0.016, calibrated) → predEV HONEST + POSITIVE every held-out season, both books: **50% parlay ~+20% predEV / +28-34% realized; 30% SGP 3-leg ~+6% predEV / +16-19% realized; 25% any ~+5% / +8-15%** (thinnest). ⇒ **the boost on a CALIBRATED (market-devig) favorite is +EV BY CONSTRUCTION (`mkt_P·(D−1)·boost`); no model edge needed.** The MODEL's ONLY job = the opportunity/trustworthiness ABSTAIN (field legs only where the book number is meaningful: count props above opportunity thresholds + anytime-TD via RZ; abstain yardage/pass_tds/completions). Bigger boost + fewer legs = safer (bias/vig compounds per leg). Sizing = ¼-Kelly, capped by each promo's max_wager.

## SGP correlation (copula) — real but sign-mixed; independence is the safe base
`nfl_sgp_correlation.py` (Gaussian copula on latent stat X per leg; OVER iff X>t; tetrachoric rho per relationship category; joint via MC). Correlations among trustworthy VOLUME props: **QB volume↔his pass-catcher +0.175; RB committee −0.55; pass-vs-rush (game script) −0.375; opposing volume −0.275; two same-team catchers ~0** (possession ~conserved → volume props TRADE OFF). **On aggregate 3-leg favorite tickets these ~CANCEL** → realized co-hit 15.96% vs independent product 14.77% (copula 15.01%); independence WINS overall log-loss ⇒ **independence is an adequate, slightly-conservative base joint.** ★ The copula's robust ACTIONABLE value = COMPOSITION SELECTION, not precise pricing: positively-correlated same-side stacks co-hit FAR more (pos-stack 19.3% vs weak/opposing 14.5%) → PREFER +corr same-side stacks (QB + his WR/TE), AVOID same-team script-conflict legs (two RBs, pass+rush). Frozen rho → `calibration/nfl_sgp_correlations.json`. ⚠ Historical SGP COMBINED prices don't exist in our data → realized SGP ROI can't be backtested; the copula is validated on co-HIT rates, and SGP EV is computed LIVE from the book's SGP price × our copula joint P.

## Shipped system
- **`bonus.py`** — EV engine. `Bonus` schema (Doug's): `bet_type` (single/parlay/sgp/any/any_parlay/sgp_sgpx) / `boost_pct` / `min_odds_leg` (default **-300**) / `min_odds_overall` (default no floor) / `min_legs` / `max_wager` / `min_wager` / `book` / `markets` (scope) / `bonus_id`. `evaluate(legs, bonus)` → boosted EV + ¼-Kelly + `qualifies` (legs_ok AND overall_ok AND type_ok AND legs_count_ok). `boosted_ev_per_dollar = P·(D−1)·(1+boost) − (1−P)`.
- **`nfl_bonus_optimizer.py`** — per bonus+book: `cross_game_plays` (one leg/game, greedy diverse frontier, gated on `evaluate()['qualifies']`) + `sgp_stacks`/`sgp_stacks_indep` (per-game +corr same-side stacks, copula [NFL] or independence [MLB] joint, required combined price for +EV). `evaluate_slate()` = the shared CLI+app core. `legs_from_board()` maps the app board → legs (opportunity gate + MARKET-devig P + the book's own price).
- **`nfl_opportunity_serving.py`** — live eligibility gate: fetches nflverse `stats_player_week` directly (no nflreadpy/polars), recreates the backtest gate byte-for-byte (VOL_STAT + SWEPT half-life; prior-season spill only early). 6h TTL cache, not Azure (replaceable data).
- **`parlay_store.py`** + **`bonus_store.py`** — durable Azure SQL (`parlays`/`parlay_legs`/`app_settings`; blobs are DEAD, local NDJSON dev-only). save/load/grade/settle; per-leg grading via `recalibration.resolve_one_prop`; boosted profit `stake·(D−1)·(1+boost)`. `build_manual_ticket()` = pure assembler behind the manual-add form.
- **App:** `💰 Bonuses` (promo CRUD + surfaced +EV cross-game portfolio + SGP stacks w/ builder-price input → exact boosted EV; one-click Log auto-consumes the bonus [single-use]) and `🎰 Parlays` (placed tickets, auto-graded per leg, + strategy analytics: realized-win vs our joint-P = live copula calibration, ROI by bonus/legcount/same-game/corr, "which leg broke it").

## Harvest + bugfixes (2026-09-22 → 09-24, all PUSHED)
- **Manual parlay-add** (`30c415a`) — the stub is now a real form; any book-built ticket (hand SGP, anytime-TD parlay) feeds the tracker + analytics.
- **Boosted-straight support** (`f49abbf`) — `pricing_common.boosted_profit`; wager grading applies it; My Bets pending editor has an editable **Boost %** column. (Column existed but nothing read it → boosted singles were under-graded.)
- **Anytime-TD grading** (`7106c8b`) — `recalibration._anytime_td_actual` sums rushing+receiving TDs via ESPN's unambiguous names, NOT the shared 'TD' label (which collided with a QB's passing TDs).
- **SGP min-total-odds bug** (`a56f5e6`) — `sgp_stacks`/`sgp_stacks_indep` ignored `min_odds_overall`; now the required price = max(+EV break-even, min_odds_overall) and the app SGP price box rejects a price below it. (Doug hit this: SGP boost with min-overall set surfaced un-placeable stacks.)
- **Default `min_odds_leg` → -300** (`93fc47c`) — dataclass default now matches the add-bonus form.

## MLB parity
MLB bonus arc wired (legs/gate/SGP) through `2fa8b2c` — MLB uses `sgp_stacks_indep` (independence; step-3 verdict = MLB same-game correlations weak, copula doesn't beat independence). `sport` field on bonuses; opportunity + threshold scripts exist for MLB.

## Open ideas (not built)
- Boosted-straight capture at Value-Finder SUBMIT time (currently only editable after, in My Bets).
- Retroactive boost on already-settled bets (currently pending-only).
- **NBA before tip-off (late Oct 2026):** NBA is recreational like NFL → prime bonus-leg territory; needs an `nba_api` warehouse + opportunity gate + calibrated de-vig legs (see [[wishlist]] uniform multi-sport warehouse). A natural next expansion of this engine.
