# Stage B design — app "Current Bonuses" section + live optimizer

Scoping doc (Doug to approve before any production code). Goal: a live app section where Doug
inputs the currently-active promos and the app surfaces the +EV plays, using the validated
bonus system (`bonus.py` → `nfl_bonus_optimizer.py`, backtested +EV, SGP-correlation-vetted).

Standing constraints honored: bets only at DK/FD; confirm before any credit spend; never destroy
irreplaceable data; production changes need explicit ok.

---

## 1. The one architectural decision that makes this cheap and honest

**Reuse the prop board the Value Finder already fetched — do NOT make new Odds API calls.**

- The Value Finder pipeline fetches the live DK/FD prop board via `get_event_odds(...)`
  (`app.py:2907`), parses it with `parse_player_props` (`app.py:3009`), and holds it in
  `prop_odds_results[eid]`. `get_event_odds` is 1h-cached (`odds_client.py:242`) — a re-read is
  **0 credits**.
- So the Bonuses section consumes the **same cached board** (same slate/markets the user just
  analyzed) → **zero marginal credit cost** for the default flow. A separate "refresh board"
  button is optional and would carry the usual one-beat spend confirmation.

**Leg probability = book de-vig** (`devig_two_way`, `odds_client.py:561`), NOT our model —
exactly as the backtest proved (our model is over-confident in the high-P tail; market P is
calibrated, ECE 0.008–0.016). This is *different* from the Value Finder's served prob (which is
our A/C model prob blended to market) — the Bonuses section deliberately uses raw market de-vig.

---

## 2. The only real gap: the eligibility gate (and how we close it without new plumbing)

The optimizer restricts legs to **trustworthy props above an opportunity threshold**
(`TRUSTWORTHY = {receptions:6.9 tgt, rush_attempts:8.5 car, pass_attempts:23.4 att}`). Offline
that threshold uses `nfl_data.player_week` expected volume — which is **NOT served in the live
app** (the app serves NFL props from ESPN gamelogs via `analyze_player_props_value`,
`props.py:1218`; our `player_week` opportunity models are offline-only).

Two ways to close it (recommend **A**):

- **(A) Compute the opportunity gate from the ESPN gamelog history the app already pulls.**
  The app already fetches per-player gamelogs via `get_player_stat_history` (`app.py:3057`) — those
  logs carry the same volume stats (targets/carries/attempts). A recency-weighted mean of the
  relevant volume stat is a faithful proxy for the opportunity gate. **No new serving feed, no
  Azure table, no credits.** Matches the gate's intent (workhorse, market-trustworthy).
- (B) Stand up a live serving feed for `nfl_data.player_week` opportunity projections (new Azure
  table + serving plumbing). Higher fidelity, but a big build for a gate that (A) approximates
  well. Defer unless (A) proves inadequate.

Honesty note: we validated *market-devig calibration* specifically on these 3 count props above
threshold. Staying inside that leg universe keeps Stage B within backtested territory. (Expanding
to yardage/other props later would need its own market-calibration check first.)

---

## 3. Freeze the SGP correlations into a config

`nfl_sgp_correlation.fit_correlations` needs historical actuals (offline). The app must NOT refit
live. Build step: run the fit once, **persist the rho-by-category table to
`calibration/nfl_sgp_correlations.json`** (loaded like other calibration blobs via
`calibration_loader`). `joint_prob()` then reads the frozen rho. Re-fit annually (offline).

---

## 4. Code changes (concrete)

**Refactor (small, no behavior change):** split `nfl_bonus_optimizer.py` so play-generation is
importable and UI-agnostic:
- `cross_game_plays(legs, bonus, bankroll)`, `sgp_stacks(legs, bonus, rho, bankroll)`,
  `_diversify(...)` are **already pure** — keep as-is.
- Add a **live leg adapter** `legs_from_board(prop_odds_results, histories, book) -> [leg]` that
  maps the parsed board + ESPN histories into the same leg dict the pure functions expect
  (`gid, prop, player, team, side, P=devig, odds, t_over, fair_over, line`), applying the
  gate from §2(A). (Offline `collect_week_legs` stays for CLI/backtests.)

**New UI (mirrors `render_my_bets`):**
- `app.py:2291` — add `"💰 Bonuses"` to the sidebar `st.radio` list.
- after `app.py:2301` — `if app_page == "💰 Bonuses": render_bonuses(); st.stop()`.
- `render_bonuses()` new function:
  - **Bonus manager**: a form for the 6 fields + book (`bet_type, boost_pct, min_odds_leg,
    min_odds_overall, min_legs, max_wager, min_wager, book`), add/edit/delete. Persist to
    `st.session_state["bonuses"]` and a small `calibration/active_bonuses.json` (durable, tiny,
    non-irreplaceable — safe to overwrite). Seed from `bonus.LIVE_BONUSES`.
  - **Plays**: if a slate board is in session (from a Value Finder run), build legs via the
    adapter and render, per active bonus/book: (a) the cross-game/single +EV portfolio table
    (EV%, stake, combined odds, P, legs) and (b) the SGP stacks table (joint P, maxCorr, need≥,
    legs). If no board cached, show "run Value Finder for a slate first" (avoids a surprise
    paid fetch).

**No changes** to: the Value Finder serving path, calibration serving, MLB, wagers/ledger.

---

## 5. Credit & data policy
- Default flow = **0 credits** (reuse cached board). Any explicit refresh = one "firing now — ~N
  credits, go?" beat.
- `active_bonuses.json` and `nfl_sgp_correlations.json` are small, re-creatable configs — safe to
  write/overwrite; they are NOT in the never-destroy class (wagers/ledger/calibration corpus).

---

## 6. SGP live pricing (honest limit)
The Odds API does not return SGP *combined* prices, and we have no historical SGP prices, so:
- Cross-game parlays/singles (bonuses 1 & 3): **fully priced and backtested +EV** → shown ready.
- SGP stacks (bonus 2, or 1/3 used same-game): shown as correlation-vetted **suggestions** with a
  `need ≥ <price>` threshold; Doug reads the actual SGP price from the DK/FD builder and bets only
  if it clears `need`. (Optionally: a tiny input to type the builder's price → app computes exact
  boosted EV + Kelly stake on the spot.)

---

## 7. Build order & test plan
1. Refactor optimizer: add `legs_from_board` adapter; keep pure play-gen. Unit-test the adapter
   maps a sample parsed board → correct leg dicts + gate.
2. Freeze rho → `calibration/nfl_sgp_correlations.json`; point `joint_prob` at it.
3. `render_bonuses()` + nav entry; bonus-manager CRUD with session + json persistence.
4. Wire plays off the cached board; render tables.
5. Manual test on a live (cached) slate; confirm 0 credits; sanity-check outputs vs the CLI.

## 8. Explicitly OUT of scope for v1
- Live serving feed for our opportunity models (§2B) — use the ESPN-gamelog gate instead.
- Auto-refresh / auto-bet — surfacing only; Doug places bets manually at DK/FD.
- Non-count props in the bonus leg universe (needs a market-calibration check first).
- Round-robins (can add later; the engine already evaluates arbitrary leg sets).

## 9. Rough size
Small–medium: ~1 optimizer adapter + 1 freeze script + ~1 Streamlit function (~150–250 lines) +
CRUD + 2 tables. No new heavy infra. The heaviest correctness risk (leg calibration, SGP joint)
is already solved and frozen upstream.
