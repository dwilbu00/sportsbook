# Audit 2026-Reset — Area 08: Pipeline Map + Reversible-Cutoff / Wager-Reset Design

Date: 2026-08-07. Scope: READ-ONLY. Every claim is grounded in `file:line` or a
read-only SQL COUNT actually run against prod. "PROVED" = observed directly;
"INFER" = reasoned from code without a runtime confirmation.

---

## TL;DR (the single most important finding)

**The entire durable data corpus is ALREADY ~100% MLB-2026.** A "reset to train
only on 2026" is a near-no-op on the *training basis* — there is essentially no
pre-2026 data in any durable store to exclude. PROVED by SQL (below). This
**substantially UNDERMINES the destructive-reset premise** that stale pre-2026
data is diluting the model. The forward-worse-than-backtest gap is almost
certainly *not* a season-mixing problem. The genuinely useful "reset" levers are
narrower: (a) archive + zero-forward the wager/bankroll ledgers for a clean ROI
measurement, (b) refit the two pre-ABS `starter_adjustment` multipliers baked
into the calibration JSON, (c) reset the (barely-engaged) online recalibration
state, and (d) run a **forward shadow A/B** instead of a blind purge.

### PROVED data span (read-only SQL, prod, 2026-08-07)
| table | rows | season span |
|---|---|---|
| `prediction_log` | 3964 (3646 resolved) | **all 2026**, game_date 2026-07-24 .. 2026-08-07 |
| `odds_snapshot` | 605 | **all 2026**, game_date 2026-07-10 .. 2026-08-08 |
| `mlb_batter_gamelog` | 36023 | **all 2026** |
| `mlb_pitcher_gamelog` | 3985 / 11 / 14 | 2026 / 2025 / NULL-date |
| `market_prediction_log` | 284 | **all 2026** |
| `wagers` | 156 (146 settled) | **all 2026**; SUM(profit) = **-$12.13** |
| `bankroll_ledger` | 146 bet + 1 adjustment | adjustment total +$282.11 |
| `recalibration_params` | **1** (batter_hits online Platt only) | fit 2026-08-07 |

Only pre-2026 residue in durable stores: **11 pitcher_gamelog rows dated 2025**
(+14 NULL-date) — trivial. The real pre-ABS residue lives in the calibration
JSON meta, not in a table (see §5).

---

## 1. Calibration training pipeline — how season already flows

### 1a. Synthetic sweep = `refit_sport()`  (`refit_calibration.py:662`)
- Signature `refit_sport(sport, season=None, prior_season=None, ...)`.
  `season` defaults to the current UTC year when None (`:667`).
- CLI wiring: `--season` / `--prior-season` → `refit_sport(season=args.season,
  prior_season=args.prior_season, ...)` (`refit_calibration.py:2789`, args defined
  `:2620-2622`).
- Player pool: `_mlb_player_pool(season, ...)` (`refit_calibration.py:281`) already
  **season-scoped** — `frequent_batter_ids([season], ...)` + `starter_ids([season])`
  (`:289-290`). NBA sibling `_nba_player_pool(season,...)` (`:306`) likewise.
- Current-season fit calls `run_player_props_backtest(..., season_year=season,
  cross_season="strict")` (`:693-702`); optional warmup fit passes
  `season_year=prior_season, cross_season="all"` (`:713-722`).
- Writes `cfg["fit_season"] = season` and `meta["current_season"]=season`,
  `meta["warmup_season"]=prior_season` (`:735, 755-757`).
- **PROVED** in the shipped JSON: every prop cfg has `fit_season=2026`; meta
  `current_season=2026, warmup_season=null`. (`calibration/baseball_mlb.json`)

### 1b. `run_player_props_backtest()` season enforcement (`backtest.py:2329`)
- `season_year` threads into every data pull that could leak a wrong season:
  `fetch_player_data(..., season_year)` (`:2349`), `cached_schedule(..., season_year)`
  (`:2377`), `_team_defense_lookup(..., season_year)` (`:2393`),
  `_team_pace_lookup(..., season_year)` (`:2404`).
- `fetch_player_data` (`:2146`) → `cached_gamelog(..., season_year=season_year)`
  (`:2154`). `cached_gamelog` (`espn_cache.py:131`) keys the cache by season and
  asks ESPN/gamelog_store for that *specific* season (`:158-160`), so the sweep
  only ever sees that season's games. **PROVED by code.**
- Leakage guard: `cross_season="strict"` → `_filter_to_current_season(prior_games,
  test_date, sport_key)` (`backtest.py:2494`, def `:179`) drops any prior game
  before the test date's season start (`_season_start_iso` `:161`,
  `SPORT_SEASON_START_MONTH`). So even a multi-season gamelog is trimmed to the
  test game's season.

**Conclusion (1):** the synthetic sweep is already a clean 2026-only fit. A
season floor here is redundant with `--season 2026` and would change nothing.

### 1c. Real-line method re-selection = `refit_sport_real_lines()` (`refit_calibration.py:767`)
This is the ONLY training path that is **not** season-scoped today.
Chain:
```
refit_sport_real_lines
  -> blc.harvest_real_line_book_lines(sport_key, target_props, store_label)   [book_line_calibration.py:258]
       -> warehouse.load_prop_lines(sport_key)            [warehouse.py:950]  ← NO dates arg passed
            -> db_store.player_prop_lines(sport, dates=None)  [db_store.py:1104]
       -> harvest_book_lines_from_prediction_log(...)     [book_line_calibration.py:189]  ← no date filter
       -> harvest_book_lines_from_store(...)              [book_line_calibration.py:142]  ← no date filter (local JSON fallback)
  -> blc.join_book_lines_to_actuals(book_lines, ...)      [book_line_calibration.py:339]
       -> cached_gamelog(aid, ttl=6h, player_name=...)    [book_line_calibration.py:373]  ← NO season_year → current-season log
  -> statcast as-of index: years = {game_date[:4] for enriched obs}  [book_line_calibration.py:~859]  ← follows the obs
```
- **KEY, PROVED:** the DB layer *already supports* date filtering —
  `player_prop_lines(sport, dates=None, date_from=None, date_to=None)`
  (`db_store.py:1104`) applies `odds_snapshot.game_date >= date_from` /
  `<= date_to` (`:1137-1140`); `team_market_lines` is the identical sibling
  (`:1037`, filter `:1069-1072`); `warehouse.load_prop_lines(sport_key,
  dates=None)` already forwards a `dates` kwarg (`:964`). **The plumbing exists
  end-to-end at the DB tier and is simply never fed a value by the harvest.**
- Statcast window auto-follows the harvested obs' `game_date` years — filter the
  harvest to 2026 and the raw-Statcast load spans 2026 only (no separate knob).
  Statcast as-of table is season-bucketed independently (`statcast_asof.py:45`
  `season_bucket`; built per `--season`, `:157-202`).
- Real-line residuals are line-invariant; the batter_hits E fit
  (`fit_basis=real_line, n_obs=3262`) is drawn from this chain — and since the
  whole corpus is 2026, that n_obs is *already effectively 2026-only* today.

---

## 2. Where a season/date filter must apply for a TRUE 2026-only basis

| Stage | File:line | Season-scoped today? | Action for a 2026 floor |
|---|---|---|---|
| Synthetic sweep gamelog/schedule/defense/pace | backtest.py:2349-2404 | YES (`season_year`) | none (already 2026 via `--season`) |
| Prior-game leakage trim | backtest.py:2494 / :179 | YES (`cross_season=strict`) | none |
| Player pool | refit_calibration.py:281 | YES (`season`) | none |
| **Real-line warehouse harvest** | warehouse.py:950 → db_store.py:1104 | **NO (dates never passed)** | thread `date_from="2026-01-01"` (DB supports it) |
| **Real-line pred-log backstop** | book_line_calibration.py:189 | **NO** | filter `game_date[:4] >= floor` after `_read_log` |
| **Real-line local-JSON store** | book_line_calibration.py:142 | **NO** | filter `date >= floor` in the loop |
| Real-line actual-join gamelog | book_line_calibration.py:373 | de-facto current season (no `season_year`) | de-facto fine once obs are floored |
| Statcast raw-day load | book_line_calibration.py:~859 | follows obs years | none (auto) |
| **Online recalibration (Platt)** | recalibration.py resolved-row reads | **NO date floor** | filter resolved rows to `game_date >= floor` before fit |

---

## 3. Reversible `TRAIN_SEASON_MIN` design (NO row deletion)

Add ONE knob, default = None (== today's behavior), so it is trivially
reversible by lowering/removing it. Two placement options:

- **Config**: `config.json` new key `"train_season_min": 2026` (config already
  loaded app-wide; see `config.json`), OR
- **Env**: `ODI_TRAIN_SEASON_MIN` (mirrors `ODI_GAMELOG_TTL_HOURS` at
  `espn_cache.py:152`) so an offline refit can set it per-run without editing the
  committed config.

Thread it as a **read-side filter only** (no DELETE, no TRUNCATE):
1. `refit_sport_real_lines` derives `date_from = f"{floor}-01-01"` and passes it
   into `harvest_real_line_book_lines(..., date_from=date_from)`.
2. `harvest_real_line_book_lines` forwards `date_from` to
   `warehouse.load_prop_lines(sport_key, date_from=date_from)` (add the kwarg;
   `player_prop_lines` already honors `date_from`), and applies
   `if r["game_date"][:4] >= str(floor)` to the pred-log + local-store harvests.
3. Online recalibration: add `game_date >= floor` to the resolved-row read used
   by the Platt fit.
4. `refit_sport` needs nothing (already `--season 2026`), but for hygiene it can
   assert `season >= floor`.

**Reversibility guarantee:** every hook is a WHERE-clause / list-comprehension
filter on reads. Lower the floor or unset it → the full corpus reappears. No
archive/restore needed because nothing is destroyed. (Compare to the blunt
`sql/clear_tables.sql`, which TRUNCATEs *every* table — do NOT use it here.)

**Caveat / honest limitation:** because the corpus is already 2026-only (§TL;DR),
this floor will exclude ~0 rows today. Its value is *forward-proofing* (it keeps
2027+ or backfilled pre-2026 data out automatically) and *intent
documentation* — not an immediate accuracy change. I could not find any current
data it would remove beyond the 11 stray 2025 pitcher_gamelog rows.

---

## 4. Wager + bankroll tables — map + ARCHIVE-then-zero-forward reset

### 4a. Tables & storage mechanics (PROVED)
- `dbo.wagers` (`sql/schema.sql:125`): one row per bet; `status IN
  (pending,won,lost,push,void)`, `profit`, `stake`, CLV cols, `uq_wager_id`.
- `dbo.bankroll_ledger` (`sql/schema.sql:653`): signed `amount` txns,
  `txn_type IN (bet,adjustment)`, `txn_id` unique (`bet:<wager_id>` |
  `adj:<iso>#<n>`). **Balance is never stored — always derived** =
  `SUM(amount)` (`bankroll.py:104-108, 122`).
- `dbo.app_settings` (`sql/schema.sql:675`): KV; Kelly knobs today
  (`bankroll.py:_KELLY_SETTING_KEYS`).
- NDJSON→SQL mapping: `recalibration._table_for` strips the extension
  (`recalibration.py:109-112`), so `wagers.jsonl`→`dbo.wagers`,
  `bankroll_ledger.jsonl`→`dbo.bankroll_ledger`, `app_settings.jsonl`→
  `dbo.app_settings`. All I/O goes through `recalibration._read_ndjson_blob` /
  `mutate_ndjson_log` (`recalibration.py:347, 421`).
- **CRUCIAL coupling:** `bankroll.reconcile_bet_txns` (`bankroll.py:192`)
  *regenerates* a `bet:<wager_id>` txn for every currently-settled wager and
  *drops* bet txns whose wager is gone/unsettled. So **you cannot zero the
  bankroll by deleting bankroll rows** — the next reconcile sweep rebuilds them
  from `dbo.wagers`. The wagers table is the source of truth for bet-P/L.

### 4b. Archive-then-zero-forward design (do NOT delete the audit trail)

**Step 1 — SNAPSHOT (pure copy, originals untouched):**
```sql
SELECT * INTO dbo.wagers_archive_20260807          FROM dbo.wagers;
SELECT * INTO dbo.bankroll_ledger_archive_20260807 FROM dbo.bankroll_ledger;
-- optional, for a clean forward-Brier baseline too:
SELECT * INTO dbo.prediction_log_archive_20260807        FROM dbo.prediction_log;
SELECT * INTO dbo.market_prediction_log_archive_20260807 FROM dbo.market_prediction_log;
```
`SELECT ... INTO` creates a new table + copies rows; the source is read-only in
this operation. This is the real-money audit trail, preserved verbatim.

**Step 2 — ZERO-FORWARD. Two options:**

- **Option A — epoch marker (fully reversible, rows stay in place). RECOMMENDED.**
  Write `app_settings.wager_reset_at = <iso>` (via `bankroll.save_kelly_settings`'s
  KV machinery / a small `record_setting`). Thread it as a floor into the three
  readers so only wagers with `placed_at >= wager_reset_at` count:
    * `wagers.read_wagers` / the My-Bets ROI+hit-rate roll-ups,
    * `bankroll.reconcile_bet_txns` (skip wagers before the epoch when building
      desired bet txns — `bankroll.py:212-225`),
    * any forward-Brier view over `prediction_log`.
  Nothing is deleted; revert = delete the setting. This matches the codebase's
  "derive, never store" philosophy (balance is already derived) and is the most
  reversible.

- **Option B — move rows out (simpler, less reversible).**
  After Step 1, `DELETE FROM dbo.wagers;` (audit trail now lives in the archive
  table). `reconcile_bet_txns` then finds no settled wagers → drops all `bet`
  txns → bet-P/L component becomes 0 automatically. Then re-anchor the derived
  balance to the user's real starting bankroll with ONE call:
  `bankroll.record_adjustment(target=<real_$>)` (`bankroll.py:139`) — it writes a
  single signed `adjustment` so `SUM(amount)` == target. (Old `adjustment` txns
  can be archived+deleted from `bankroll_ledger` too, since they're preserved in
  the archive; or left — they're harmless once wagers are gone and one fresh
  anchoring adjustment is written.)

Recommend **A** for reversibility; **B** only if the user wants a genuinely empty
live `wagers` table. Either way the archive table is the untouched audit record.

**Do NOT** run `sql/clear_tables.sql` — it TRUNCATEs every table incl. gamelogs,
odds warehouse, id maps, and calibration state.

---

## 5. Pre-ABS residue that a reset SHOULD target (not in any table)
`calibration/baseball_mlb.json` `meta.starter_adjustment` (PROVED):
- `props`: `source="backtest_props:2024,2025"` (pre-ABS gamelogs), selected
  weights 0.5.
- `team_markets`: `source="backtest_starters:2021,2022,2023,2024"`, n_games 7915.
- `meta.prob_shrink`: `source="odds backtest --engine live"` (undated).

These multipliers encode a **pre-ABS regime** and are the most defensible
"reset" target if the 2026 ABS strike zone shifted the distribution. They are
small candidate-weighted adjustments, not the core A/B/C/E method choice, so the
blast radius is limited. Re-fitting them requires 2026 backtest data, which is
still only ~1 month deep (thin).

---

## 6. Forward SHADOW A/B — measure whether the reset actually helped

Purpose: answer the owner's real question (does a 2026-basis fix
forward<backtest?) on LIVE games, paired, before any destructive change.

Infrastructure that already exists:
- `dbo.prediction_log` (`schema.sql:14`) logs `raw_prob`, `final_prob`,
  `projected`, `line`, `outcome` per UNIQUE `(sport,event_key,prop,player,line)`
  (`uq_prediction_identity :45`). Grading via
  `recalibration.resolve_pending_outcomes` fills `actual/outcome`.
- `calibration_loader.py` loads a calibration file; a candidate file can sit
  beside the shipped one.

Sketch (minimal-invasive, non-destructive):
1. Produce **basis-B** = a 2026-floored real-line refit (+ optionally reset
   `starter_adjustment`) written to `calibration/baseball_mlb.candidate.json`.
   Keep the shipped file as **basis-A** (recommendation stays on A).
2. Add ONE nullable column `shadow_prob FLOAT` (and optionally `shadow_basis`) to
   `dbo.prediction_log`. In the analysis write path, compute the prop's
   probability under basis-B as well and store it in `shadow_prob`. The
   recommendation/pick is unchanged — basis-A drives everything the user sees.
3. Grading is shared: the same `outcome` grades both `final_prob` (A) and
   `shadow_prob` (B) against the same actual — an **apples-to-apples paired**
   comparison the current backtest-vs-forward mismatch cannot give.
4. After N resolved obs, compare A vs B Brier / log-loss / ROI with a single
   `GROUP BY` (paired on identical games). If B wins forward, ship it; if not,
   drop the candidate file + column — nothing was risked.

Heavier alternative (cleaner separation, more work): a sibling
`dbo.prediction_log_shadow` table (same schema + `basis` col) so multiple
candidate bases can run at once; but the single-column variant is enough for one
A/B and adds no new table.

This is strictly preferable to a blind purge because the ONLY ABS-era data that
exists is these ~3.6k resolved 2026 predictions — a destructive reset that
mis-fires would throw away the very data needed to judge it.

---

## 7. Adversarial self-check / what I could NOT prove
- I did NOT run the sweep/refit or any writer (rule-compliant). The "floor is a
  near-no-op today" claim rests on the SQL span counts (PROVED) — if there is a
  local historical_odds JSON store or Blob with pre-2026 lines, the real-line
  *local-store* harvest (`book_line_calibration.py:142`) could still pull older
  data; SQL is the prod primary (`harvest_real_line_book_lines:275-285` prefers
  the warehouse when `db_store.enabled()` and no `--store-label`), so the local
  path only fires with an explicit `--store-label`. I did not enumerate the local
  `historical_odds/` dir contents — flagged as an open question.
- I did NOT confirm the exact online-Platt read query filters (searched
  recalibration.py for the resolved-row read; the fit consumes resolved rows via
  `_read_log(where={resolved:True})` with no date predicate — INFER no season
  floor, consistent with the generic `_table_for` mapping).
- I did NOT verify *why* forward Brier < backtest Brier — out of scope here, but
  the SQL facts make "pre-2026 contamination" an unlikely cause; more likely
  expanding-fold optimism, thin synthetic pitcher props (n=333), online-Platt
  barely engaged (1 param), and early-2026 ABS non-stationarity.
