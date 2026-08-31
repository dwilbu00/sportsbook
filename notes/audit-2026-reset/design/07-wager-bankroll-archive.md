# Workstream 7 — Wager/Bankroll ARCHIVE-then-zero-forward reset (Option A: epoch marker, NO deletes)

**Date:** 2026-08-07 · **Mode:** DESIGN ONLY (read-only audit; no code/DB/sweep/vendor writes made).
**Grounding:** every claim below verified against current `file:line` this pass. Supersedes the sketch in `08-pipeline-map-and-reset-design.md §4` with a concrete, line-accurate build.

---

## 0. Bottom line

Implement the reset as a **display/accounting epoch**, not a data-lifecycle event. Add one KV marker `app_settings.wager_reset_at` plus one durable **opening-baseline** transaction, and thread a single Python-side `placed_at >= epoch` filter through exactly three surfaces: the My-Bets ROI/hit-rate/CLV rollup, the bankroll `reconcile_bet_txns` sweep, and the bankroll `summary()`. **No rows are ever deleted.** Grading, CLV backfill, and the raw wager rows keep operating on the full set. Fully reversible by deleting the marker + baseline txn (the next reconcile self-heals to the pre-reset balance).

Key facts that shape the design:
- **`where` is equality/IN only** (`db_store._where_clause` :643-653 builds `column == val` or `column.in_(...)`). There is **no range-comparison push-down**, so the epoch floor (`placed_at >= …`) **cannot** be a SQL `where` predicate. It must be a Python-side filter after the full read. This is why the filter lives in `wagers.py`/`bankroll.py`, not in the store layer.
- **Balance is derived, never stored** (`bankroll.current_balance` :104-108 = `SUM(amount)`; `summary` :121-122 = `bets_total + adj_total`).
- **`reconcile_bet_txns` regenerates `bet:<wager_id>` txns from settled wagers and DROPS bet txns not in its `desired` set** (`bankroll.py` :227-260). This is the landmine: any change that hands it an epoch-filtered wager list will silently drop the pre-epoch bet P/L from the live balance. The design compensates with the opening-baseline txn (below).
- **`placed_at` and the epoch marker share one generator.** Submit writes `placed_at = datetime.now(timezone.utc).isoformat()` (`app.py:461`); `bankroll._now_iso` (:63-64) is identical. Both are UTC ISO-8601 with `+00:00`, so **lexicographic string comparison == chronological comparison** — a valid, dependency-free epoch test. Inclusive lower bound (`>=`) → a bet at/after the reset instant is "current".
- **No schema change is required.** `app_settings` already exists (`schema.sql:674-681`); the marker is a new KV row. `bankroll_ledger` has **no CHECK on `txn_type`** (`schema.sql:652-662`), so a new `txn_type='baseline'` is legal with zero DDL.

---

## 1. Current-state map (verified line numbers)

### wagers.py
- `read_wagers(where=None)` :265-279 — unfiltered read returns ALL rows via `read_wagers_with_status()[0]` (:251-262). `where` path is SQL equality/IN only.
- `resolve_pending_wagers` :505-569 — grades pending rows (`where={"status":"pending"}`). **Must stay epoch-blind** (a pre-epoch pending bet that settles later must still record its status/profit for the archive).
- `summarize_wagers(rows)` :726-750 → `_metrics(group)` :671-699 — computes `roi`, `hit_rate`, `avg_clv_pct`, W-L-P, staked, `by_sport`, `by_bet_type`. **This is the ROI/hit-rate/CLV engine; per-epoch stats = call it on each epoch partition.**
- `apply_clv_updates` :599-624, `read_wagers` used by CLV backfill — see §5.
- `_blank_row` :68-106 sets `placed_at` from `meta['placed_at']`.

### bankroll.py
- `BANKROLL_FILE`='bankroll_ledger.jsonl' :38, `SETTINGS_FILE`='app_settings.jsonl' :39.
- `read_ledger` :84-90, `current_balance` :104-108 (`SUM(amount)`), `summary` :111-127 (`bets_total`+`adj_total`, `n_txns`, `txns` newest-first).
- `record_adjustment(target)` :139-181 — writes ONE `adjustment` txn = `target - current`, `txn_id='adj:<iso>#<n>'`.
- `reconcile_bet_txns(wager_rows=None)` :192-265 — builds `desired` from settled wagers (:212-225), upserts `bet:<wager_id>` txns, **drops `bet:` txns not in `desired`** (:254-259). Only touches `bet:`-prefixed txns (:229-230) — adjustments/baseline untouched.
- `load_kelly_settings` :272-288 / `save_kelly_settings` :291-330 — KV helpers, whitelisted to `_KELLY_SETTING_KEYS` (:50), parse to float. **No generic string-KV getter exists** → add one (§3).

### app.py
- `render_my_bets` :1711-… — reads `rows` (:1749), calls `reconcile_bet_txns(rows)` (:1768), renders `_render_bankroll_section()` (:1775), `summarize_wagers(rows)` (:1808), splits `settled`/`pending` (:1839-1840), editors (:1847-…).
- `_render_bankroll_section` :1649-1708 — bankroll metrics from `bankroll.summary()` (:1654): "Current bankroll" (:1661), "Realized bet P/L"=`bets_total` (:1662), "Manual adjustments"=`adjustments_total` (:1663), adjust expander (:1669-1689), history (:1690-1708).
- `_apply_bankroll_adjustment` :1624-1646 (adjust callback).
- Session prefetch reconcile :2094-2113 (`_bankroll.reconcile_bet_txns()` :2104).

### db_store.py / recalibration.py / schema.sql
- `_where_clause` :643-653 (equality/IN only — **no ranges**). `read_rows` :679-700, `mutate` :725-… (surgical diff by natural identity).
- `app_settings` table :247-254 / DDL :674-681; identity `{setting_key}` (:506-508). `bankroll_ledger` table :231-242 / DDL :652-668; identity `{txn_id}` (:503-505); **no txn_type CHECK**.
- `recalibration._read_ndjson_blob` :347-392, `mutate_ndjson_log` :421-459, `_table_for` :109 (strips `.jsonl`).

### backfill_dk_clv.py (must stay epoch-blind)
- `read_wagers()` :161, `read_wagers_with_status()` :346, `apply_clv_updates` :472 — CLV is per-bet; archived bets keep their CLV. **No change** (relies on `read_wagers` staying unfiltered).

---

## 2. Design overview (Option A)

Three durable artifacts, all in existing tables (zero DDL):
1. **`app_settings.wager_reset_at`** = ISO instant of the reset (the epoch boundary).
2. **One `bankroll_ledger` baseline txn**: `txn_id='epoch:<iso>'`, `txn_type='baseline'`, `amount = pre-epoch realized bet P/L` at reset time, `note='Epoch baseline — previous realized P/L locked in'`.
3. **(Optional insurance)** physical `SELECT INTO` snapshot tables (§6). With Option A the *logical* archive is the epoch partition itself (`placed_at < wager_reset_at` = "previous"); the physical snapshot is belt-and-suspenders before the ledger is mutated.

Epoch semantics: a wager is **current** iff `wager_reset_at is None` OR `placed_at >= wager_reset_at`; otherwise **previous** (archived). A row with missing/empty `placed_at` compares `"" >= epoch` → False → treated as previous (conservative; all real rows carry `placed_at`).

Balance carry-over identity (why the money is preserved):
- Before reset: `B = P_bet + A_adj` (all bet txns + all adjustments).
- Reset writes baseline `+P_bet` → sum `= B + P_bet`.
- Epoch-aware reconcile drops the now-previous `bet:` txns (`-P_bet`) → sum `= A_adj + P_bet(baseline) = B`. **Balance unchanged; money did not move.**
- Forward: only current-epoch bets create new `bet:` txns → balance moves by new P/L only; `summary.bets_total` starts at 0 (current epoch).

---

## 3. Concrete changes

### 3a. bankroll.py — generic KV + epoch marker (NEW)
Add generic string-KV helpers (do NOT fold into the float-only Kelly path):
```python
def read_setting(key, default=None):
    """Single string setting from the KV store; default on missing/error. Never raises."""
    try:
        rows, _ = recalibration._read_ndjson_blob(SETTINGS_FILE, use_cache=True)
    except Exception:
        return default
    for r in (rows or []):
        if r.get("setting_key") == key:
            v = r.get("setting_value")
            return v if v is not None else default
    return default

def write_setting(key, value):
    """Upsert one string setting (mirrors save_kelly_settings' upsert). Returns 1/0."""
    # same mutate_ndjson_log(SETTINGS_FILE, upsert) shape as save_kelly_settings :308-328
```
Add the epoch accessor:
```python
WAGER_RESET_KEY = "wager_reset_at"
def current_epoch_start():
    """ISO instant of the active tracking epoch, or None (== pre-reset behavior)."""
    return read_setting(WAGER_RESET_KEY)  # fail-open None
```

### 3b. wagers.py — epoch partition helpers (NEW; read stays unfiltered)
```python
def in_current_epoch(row, epoch):
    if not epoch:
        return True
    return (row.get("placed_at") or "") >= epoch

def partition_by_epoch(rows, epoch=None):
    """(current_rows, previous_rows). epoch defaults to bankroll.current_epoch_start()."""
    if epoch is None:
        try:
            import bankroll
            epoch = bankroll.current_epoch_start()
        except Exception:
            epoch = None
    if not epoch:
        return list(rows), []
    cur, prev = [], []
    for r in rows:
        (cur if in_current_epoch(r, epoch) else prev).append(r)
    return cur, prev
```
`read_wagers` / `read_wagers_with_status` are **unchanged** (still return all rows) so grading, CLV backfill, and delete/edit keep operating on the full set.

### 3c. bankroll.py — make `reconcile_bet_txns` epoch-aware (SURGICAL)
At the top of the `desired`-building loop (:212), skip pre-epoch wagers:
```python
epoch = current_epoch_start()
for w in (wager_rows or []):
    wid = w.get("wager_id")
    if not wid or w.get("status") not in _SETTLED:
        continue
    if epoch and (w.get("placed_at") or "") < epoch:
        continue          # pre-epoch bet P/L is frozen in the baseline txn
    ...
```
This alone makes the stale-drop logic (:254-259) remove pre-epoch `bet:` txns — which is exactly the intended drop, compensated by the baseline. When `epoch is None`, the branch is inert → **byte-identical to today**. Callers (`app.py:1768` passing full `rows`, `app.py:2104` reading the ledger itself, tests) need no change.

> **Requires** wager rows to carry `placed_at`. `app.py:1768` passes the full `read_wagers_with_status()` rows (have it). The self-reading path (`wager_rows=None`, :204-209 → `wagers.read_wagers()`) also has it. ✓

### 3d. bankroll.py — `summary()` surfaces the baseline (SMALL)
Recognize the baseline bucket so the UI can show it distinctly and the balance stays exact:
```python
baseline_total = round(sum(_amount(r) for r in rows if r.get("txn_type") == "baseline"), 2)
adj_total      = round(sum(_amount(r) for r in rows
                          if r.get("txn_type") == "adjustment"), 2)   # unchanged filter
...
"balance": round(bets_total + adj_total + baseline_total, 2),
"opening_baseline": baseline_total,
```
When no baseline txn exists (`baseline_total == 0`), every existing field is unchanged → existing test_bankroll assertions (`bets_total`, `adjustments_total`, `balance`, `n_txns`) still hold. `current_balance` (SUM of all) already includes baseline — no change there.

### 3e. bankroll.py — the reset + revert operations (NEW)
```python
def start_new_epoch(now=None):
    """Archive-then-zero-forward: lock in the prior realized bet P/L as an opening
    baseline, stamp the epoch marker, and zero the current-epoch bet P/L. The
    derived balance is UNCHANGED (money doesn't move). Idempotent per instant.
    Returns {"epoch": iso, "baseline": P_bet, "prev_pending": n} or None on error."""
    now = now or _now_iso()
    # 1) Guard/inform: count pre-epoch wagers still pending (straddlers — see Risks).
    # 2) Compute P_bet = SUM of current 'bet' txns (all pre-epoch at first reset).
    # 3) Write baseline txn FIRST  (txn_id='epoch:%s'%now, type='baseline', amount=P_bet).
    # 4) Write marker             (write_setting(WAGER_RESET_KEY, now)).
    # 5) reconcile_bet_txns()     (epoch-aware → drops the now-previous bet txns).
    # Ordering rule: baseline BEFORE marker so any app-triggered reconcile between
    # steps 4 and 5 cannot lose the balance (baseline already compensates the drop).

def revert_epoch():
    """Undo: delete the marker + all 'epoch:*' baseline txns, then reconcile
    (now epoch=None) → every settled wager's bet txn is rebuilt → balance == B.
    Fully restores the pre-reset state. Returns True on success."""
```
Invocation: recommend a thin CLI wrapper (e.g. `reset_wager_epoch.py --confirm` / `--revert`) run offline alongside the optional SQL snapshot (§6), because this is real-money accounting and pairs with a DBA snapshot step. A guarded UI button in `_render_bankroll_section` (inside an expander, requiring a typed confirmation) is an acceptable alternative; the core functions are identical either way. `db_store.promote_secrets_from_toml()` must be called first in any offline invocation (per the CLV runbook gotcha in MEMORY).

### 3f. app.py — read/reconcile/ROI views (SURGICAL)
In `render_my_bets` after the read (:1749) and reconcile (:1768, unchanged — reconcile self-filters now):
```python
current_rows, previous_rows = wagers.partition_by_epoch(rows)   # epoch from settings
summary = wagers.summarize_wagers(current_rows)                 # was summarize_wagers(rows)
```
- Headline metrics (:1808-1837) and the `settled`/`pending` splits (:1839-1840) derive from **`current_rows`** → editors show only current-epoch bets.
- Add an "🗄️ Archived (previous epoch)" `st.expander` rendering `wagers.summarize_wagers(previous_rows)` (ROI/hit-rate/CLV/W-L-P) and a read-only settled table, shown only when `previous_rows`. Reuses the exact same metric renderer → per-epoch ROI/hit-rate/CLV for free.
- When `wager_reset_at` is unset, `previous_rows == []` and `current_rows == rows` → the page is **pixel-identical to today**.

In `_render_bankroll_section` (:1660-1668): add a 4th metric / caption line for `bsummary["opening_baseline"]` when non-zero ("Opening baseline (locked-in prior P/L)"), so "Realized bet P/L" reads as current-epoch and the balance decomposition is legible.

### 3g. schema.sql / db_store.py — NO structural change
`app_settings` and `bankroll_ledger` already accommodate the marker and the `baseline` txn_type (no CHECK to amend). Add only doc comments: note `wager_reset_at` as a known `app_settings` key (near :674) and `baseline` as a third `txn_type` (near :652). Optional: ship `sql/archive_epoch.sql` template (§6).

---

## 4. What deliberately does NOT change (behavior preservation / leakage safety)
- `read_wagers` / `read_wagers_with_status` stay unfiltered → **grading, delete/edit, and CLV backfill are untouched** and continue covering both epochs.
- `resolve_pending_wagers` stays epoch-blind → straddling pre-epoch pending bets still settle into the archive.
- Calibration/forward-tracker read `prediction_log` / `market_prediction_log`, **not** `wagers` → **no leakage into the model.** The epoch is purely a P/L accounting boundary; it never gates training data. (Prediction-log archiving, if ever wanted, is a separate workstream — out of scope here.)
- `current_balance` (SUM all) is already correct with the baseline txn.
- `record_adjustment` counts `adj:` txn_ids (:161-162) — unaffected by `epoch:`/`baseline`.

---

## 5. Blast-radius audit (all consumers of the touched surfaces)
- `app.py` — render + reconcile (changed as above).
- `backfill_dk_clv.py` :161/:346/:472 — reads ALL wagers, writes CLV per-bet → **unaffected** (correct: archived bets keep CLV).
- `test_wagers.py`, `test_bankroll.py` — no epoch set in fixtures → all paths inert → **green unchanged** (verify).
- `migrate_warehouse_to_sql.py`, `test_db_store.py` — reference table names/specs only; `app_settings`/`bankroll_ledger` specs gain no columns → **unaffected**.

---

## 6. Optional physical snapshot (belt-and-suspenders, prod = Azure SQL)
Before running `start_new_epoch`, freeze the raw pre-reset rows (recoverable even if a later true-purge is ever done):
```sql
SELECT * INTO dbo.wagers_archive_20260807          FROM dbo.wagers;
SELECT * INTO dbo.bankroll_ledger_archive_20260807 FROM dbo.bankroll_ledger;
```
`SELECT ... INTO` copies rows; source is read-only. Revert = `DROP TABLE` the archives. **Do NOT run `sql/clear_tables.sql`** (it TRUNCATEs every table). This step is optional under Option A because no rows are deleted; the epoch partition is the working archive.

---

## 7. Reversibility summary
`revert_epoch()`: delete `wager_reset_at` + all `epoch:*` baseline txns → next reconcile (epoch=None) rebuilds every settled wager's `bet:` txn → balance returns to `B = P_bet + A_adj` exactly; My-Bets shows the full unified history again. Plus `git revert` of the code. Plus (if taken) `DROP` the snapshot tables. Three independent, composable undo levers; the durable data was never destroyed.

---

## 8. Test plan
**Existing pins (must stay green, epoch unset):** `test_bankroll.py` (`_BankrollBehaviorMixin` on SQL + Local: adjustments, reconcile upsert/drop/regrade, summary split, Kelly KV) — line 22-245; `test_wagers.py::SummaryTests` (:746-…), `_SqlLedger` (:658) / `_LocalLedger` (:185) round-trips, delete/edit/regrade, `apply_clv_updates` idempotency (:720).

**New tests (add to both SQL and Local backends via the existing mixins):**
1. `read_setting`/`write_setting` round-trip + default-on-missing; does not disturb Kelly keys.
2. `start_new_epoch` preserves `current_balance` exactly (B before == B after), writes one `baseline` txn = prior `bets_total`, sets the marker.
3. Post-reset `summary`: `bets_total == 0` for a fresh epoch, `opening_baseline == P_bet`, `balance == B`; a NEW current-epoch settled bet moves balance by exactly its profit (baseline not double-counted).
4. `reconcile_bet_txns` epoch-aware: with a marker set, a pre-epoch settled wager creates NO `bet:` txn (frozen in baseline); a post-epoch settled wager does. With no marker, behavior byte-identical to today (regression pin).
5. `partition_by_epoch`: correct current/previous split on `placed_at`; `epoch=None` → all current; missing `placed_at` → previous.
6. `summarize_wagers(current_rows)` vs `summarize_wagers(previous_rows)` give independent ROI/hit-rate/CLV (per-epoch stats).
7. `revert_epoch` restores pre-reset balance and re-creates all bet txns (idempotent).
8. Idempotency/ordering: calling `reconcile_bet_txns` between marker-set and final reconcile never loses the balance (simulate the crash window).
9. Straddler caveat (documented behavior): a pre-epoch bet still pending at reset, graded afterward, lands in the ARCHIVE summary and does NOT auto-move the live balance (assert, and assert the manual `record_adjustment` true-up path works).

---

## 9. Risks / open questions
- **Straddlers (main caveat).** A pre-epoch bet still *pending* at reset has no bet txn yet, so it is not in the baseline; when it later settles, the epoch-aware reconcile ignores it → its real profit won't auto-update the live balance. Mitigation: `start_new_epoch` reports `prev_pending` count so the caller can warn "settle open bets first"; recommend resetting at a quiet moment; the existing `record_adjustment(target=…)` gives a one-line manual true-up. (Corpus today: 156 wagers / ~10 pending — small, deliberate one-time action.)
- **Lexicographic `placed_at` compare** assumes the shared UTC-ISO generator (verified `app.py:461` == `bankroll._now_iso` :63). If a future writer ever stored a non-UTC/offset-varying `placed_at`, the compare could misbucket — pin with a test and keep the single generator.
- **Cross-file atomicity.** Marker (app_settings) and baseline (bankroll_ledger) are separate NDJSON mutations with no cross-file txn. Mitigated by ordering (baseline before marker) + reconcile idempotency + optional physical snapshot; a crash leaves a self-healing or trivially-revertible state, never lost money.
- **Multi-epoch.** The design chains (each reset adds another `epoch:*` baseline and advances the marker); the UI shows only "current vs everything-before-current". Sufficient for the owner's one previous/one current ask; multi-epoch history browsing is out of scope.
- **Local/Blob mode.** Prod is SQL-only (MEMORY). The physical snapshot (§6) is SQL-only; the epoch/baseline/partition logic is backend-agnostic (works on the NDJSON fallback too, exercised by the Local test mixin).
