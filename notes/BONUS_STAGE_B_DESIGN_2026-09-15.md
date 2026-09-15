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

## 2. The eligibility gate — DECISION: build the serving feed (Doug 2026-09-15)

The optimizer restricts legs to **trustworthy props above an opportunity threshold**
(`TRUSTWORTHY = {receptions:6.9 tgt, rush_attempts:8.5 car, pass_attempts:23.4 att}`). Offline
that threshold uses `nfl_data.player_week` expected volume — **NOT served in the live app** (the
app serves NFL props from ESPN gamelogs via `analyze_player_props_value`, `props.py:1218`; our
`player_week` opportunity models are offline-only). The ESPN-gamelog proxy was rejected in favor
of the higher-fidelity **serving feed** so the live gate is byte-for-byte the backtested gate.

**What the gate actually needs (scoped down):** only **expected volume per slate player** — a
recency-weighted mean of the relevant VOLUME stat (targets / carries / attempts) — compared to
the threshold. It does NOT need the full distributional model (leg P = market de-vig). So the
serving feed can be lightweight: recent weekly volume per player, nothing more.

**Build = an Azure `nfl_player_week` table (Azure is the only system of record; Streamlit Cloud
can't read local parquet).**
- **Table**: `nfl_player_week(player_norm, season, week, targets, carries, attempts, receptions,
  ...)` — the volume columns from `nfl_data.player_week`. Backfill 2012–2025 once from the parquet
  we already have (LFS `nfl_ladder_data`/player_week); this is FREE (nflreadpy, no credits).
- **Refresh — LAZY / SELF-HEALING in the app (Doug 2026-09-15):** no cron, no manual command. On
  entering the Bonuses section, the app checks whether Azure `nfl_player_week` already has the
  current season through the **most recent COMPLETED week**; if it's missing/stale, the app pulls
  that from nflreadpy and upserts it, then proceeds. Idempotent; the staleness check is cached per
  session so it fires at most once per app load. Historical rows are immutable.
  - Note on timing: for a live slate (games not yet played) the gate projects from PRIOR weeks, so
    "current data" = through the last completed week — which exists before kickoff. No dependency
    on same-day results.
- **Serving helper** (`nfl_opportunity_serving.py`): `ensure_current(season)` (the lazy refresh)
  + `expected_volume(player_norm, season, week, prop)` → recency-weighted mean over that player's
  prior weeks in the table, using the SAME SWEPT half-life the backtest used per prop. Returns the
  value the gate thresholds on. Cached per-slate in session.
- The gate in `legs_from_board` then = `expected_volume(...) >= TRUSTWORTHY[prop]`.

Honesty note: we validated *market-devig calibration* specifically on these 3 count props above
threshold — the serving feed keeps the live gate identical to that validated universe. (Expanding
to yardage/other props later would need its own market-calibration check first.)

**✅ DEPENDENCY RESOLVED — refresh via DIRECT nflverse URL fetch, no nflreadpy/polars.** The
architecture rule is explicit (`nfl_ingest.py`: "Runtime NEVER imports nflreadpy/polars"), and
nflreadpy drags in polars. So the in-app lazy refresh must NOT use nflreadpy. Instead it fetches
the nflverse release parquet DIRECTLY with pandas (pyarrow already present for mirror reads):
- URL: `https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet`
- **Verified** (2026-09-15): loads via `pd.read_parquet(url)`, carries every gate column
  (`targets, carries, attempts, receptions, week, season, player_display_name, team, position`).
- `ensure_current(season)`: check Azure max(week) for the season; if behind the release's max
  completed week, `pd.read_parquet` the release, select+rename to our `nfl_player_week` columns,
  upsert new (season,week) rows to Azure. pandas + requests/pyarrow only — honors the no-nflreadpy
  runtime rule AND Doug's self-healing-in-app intent. (b) Azure WRITE already works in the app
  (wagers/ledger). No new heavy cloud dependency.
- Backfill of 2012–2025 can use the SAME URL pattern per season (or the parquet we already have) —
  free, no nflreadpy.

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
  `need ≥ <price>` threshold. **DECISION (Doug 2026-09-15): add a price input** — a small field per
  stack to paste the DK/FD builder's combined SGP price → the app computes the exact boosted EV
  (using our copula joint P) + fractional-Kelly stake on the spot, and flags BET / SKIP. `need ≥`
  is still shown as the at-a-glance cutoff.

---

## 7. Build order & test plan
1. **Serving feed** (§2): create Azure `nfl_player_week` table; backfill 2012–2025 from parquet
   (free); write `nfl_opportunity_serving.py` = `ensure_current()` (lazy self-healing refresh) +
   `expected_volume()`. Verify `expected_volume` matches the offline `acc._obs_from_series`
   exp_vol for spot-checked players; verify `ensure_current` no-ops when the table is fresh.
2. Freeze rho → `calibration/nfl_sgp_correlations.json`; point `joint_prob` at it.
3. Refactor optimizer: add `legs_from_board(board, book)` adapter (gate via §1 serving feed);
   keep pure play-gen. Unit-test adapter maps a sample parsed board → correct leg dicts + gate.
4. `render_bonuses()` + nav entry; bonus-manager CRUD (session + `active_bonuses.json`).
5. Wire plays off the cached board; render cross-game portfolio + SGP tables; add the SGP price
   input → exact boosted EV + Kelly.
6. Manual test on a live (cached) slate; confirm 0 credits for the default flow; sanity-check
   outputs vs the CLI; verify the gate matches backtested eligibility.

## 7b. FOLLOW-ON — dedicated Parlay tracker (Doug 2026-09-15; build AFTER the optimizer section)

The `wagers` table is single-selection only — parlays aren't representable. Since SGP prices can't
be backtested, LIVE tracking is the only way to validate/refine the parlay+SGP thesis. Build a
dedicated **Parlays** section (separate from My Bets):
- **Storage — TWO normalized Azure tables (Doug's call; parlays are irreplaceable wagers → Azure):**
  - `parlays(parlay_id PK, placed_at, book, bonus_label, bet_type, boost_pct, is_same_game,
    n_legs, combined_american, stake, our_joint_prob, our_boosted_ev_pct, max_corr, status,
    settled_at, payout, profit)`
  - `parlay_legs(id PK, parlay_id FK, player, prop, line, side, price, our_leg_prob, team, opp,
    corr_category, actual, result)`
  - Owner-run DDL (I write it) + SQLAlchemy Table defs in db_store mirroring it.
- **Entry:** one-click "Log this parlay" from an optimizer-surfaced play (auto-fills our joint P,
  boosted EV, corr, legs) + manual add for book-built SGPs.
- **Grading:** reuse box-score per-leg grading → all legs win ⇒ ticket won; any lose ⇒ lost; voids
  handled. Combine, set payout/profit.
- **Analytics (the strategy payoff):** realized win rate vs our joint P (live copula calibration);
  ROI by bonus / leg count / same-game vs cross-game / correlation content (pos vs neg stack);
  leg hit rate vs market-devig P; "which leg broke it" attribution; boost value captured vs
  predicted.
- **STRAIGHT-bet boosts too (Doug 2026-09-15):** the 25% "any" bonus applies to singles, so add a
  nullable `boost_pct` column to the existing `wagers` table (additive owner-run ALTER, never
  drops). Store the RAW market price + boost_pct; compute boosted profit at grade time
  (`stake·(dec−1)·(1+boost)` on a win) so CLV/close comparisons stay honest (the boost is a promo,
  not a line move) and the boost is auditable. My Bets shows the boost; straight-bet analytics get
  boost-captured value. Do this in the same tracker phase as the parlay tables.

## 8. Explicitly OUT of scope for v1
- Auto-refresh / auto-bet — surfacing only; Doug places bets manually at DK/FD.
- Non-count props in the bonus leg universe (needs a market-calibration check first).
- Round-robins (can add later; the engine already evaluates arbitrary leg sets).
- Serving the full distributional model live — the gate needs only expected VOLUME (§2).

## 9. Rough size
**Medium** (up from small–medium, due to the serving feed): Azure table + backfill + weekly
refresh + `nfl_opportunity_serving.py` (§7.1) is the bulk of the new work; then the optimizer
adapter + freeze + ~1 Streamlit function (~200–300 lines) + CRUD + tables + SGP price input. The
heaviest correctness risk (leg calibration, SGP joint) is already solved and frozen upstream. One
owner-run step (Azure DDL/backfill) + a decision on where the weekly refresh runs.
