"""nfl_ladder_backfill.py — dense 6h opener ladder for NFL props (+ team opener),
persisted DIRECTLY to the Azure warehouse as it fetches.

The opener probe proved NFL props post ~4-5 days out, and the −4h→close CLV test was
negative because −4h is already near the market consensus. To test the OPENER thesis
(the early line ≈ the book's own model, before sharp money corrects it → model-vs-model)
we capture a dense time ladder from the opener down toward kickoff, for all three books.

DURABILITY (Doug's rule): NOTHING local is durable — Azure SQL is the only system of
record. So every rung is written to the warehouse the instant it's fetched, via the same
write-once path the rest of the odds warehouse uses (db_store.capture_odds_snapshot with
ingest_multibook_cache._per_book_lines). No local-only parquet.

CAPTURE (one historical call per snapshot = 10 × #markets × 1 region):
  * books   : draftkings, fanduel (EXECUTABLE) + pinnacle (SHARP REFERENCE, analysis-only
              per the standing rule — stored like any book; the analysis-only convention is
              enforced at READ time as bookmaker=='pinnacle', not by a DB flag).
  * PROPS   : 8 player-prop markets at EVERY rung (80 credits) — kind='props'.
  * TEAM    : h2h/spreads/totals only at the −114h opener rung (+30) — kind='team'. We
              already hold the LATE team snapshots (early_12h/early_4h/close) in Azure, so
              teams only need the opener.
  * offsets : 6h ladder −6h → −114h. The CLOSE and −4h are NOT re-fetched — the warehouse
              already holds `closing` and `early_4h` for all three books (verified), so the
              CLV analysis uses those as the late endpoints. Ordered nearest-kickoff first.
  * source  : per-rung tag 'ladder_006h' … 'ladder_114h' (the write-once uq key already
              keeps rungs distinct via snapshot_hour; the tag is for WHERE-filtering).

RESUME/IDEMPOTENT: at start we load the set of (event_id, kind, snapshot_hour) already in
the warehouse under ladder sources; those rungs are SKIPPED with no fetch (0 credits), so a
resume after any interruption re-does only what's missing. capture_odds_snapshot is
write-once (duplicate → skipped). Fetch runs parallel; the SQL write runs serially in the
main thread (the 20-DTU tier throttles concurrent prop-heavy writes). --dry-run spends and
writes nothing. Recommend SQL_DRIVER=pyodbc for the bulk line insert (fast_executemany).
"""
import os as _os
import sys as _sys
# This is a bare CLI, not a `streamlit run` app, so Streamlit floods STDERR (one line per
# worker thread) with 'missing ScriptRunContext' / 'No runtime found' / cache-storage noise
# that drowns real progress. It bypasses Python logging levels (own handler), so we filter
# stderr in-process: drop ONLY those known-benign lines, pass everything else (real
# tracebacks etc.) through untouched. Installed before anything imports streamlit.
_os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")

_NOISE = ("missing ScriptRunContext", "No runtime found",
          "MemoryCacheStorageManager", "Session state does not function")


class _StderrFilter:
    def __init__(self, real):
        self._real = real

    def write(self, s):
        if any(tok in s for tok in _NOISE):
            return
        self._real.write(s)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


_sys.stderr = _StderrFilter(_sys.stderr)

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import text

import db_store
import ingest_multibook_cache as im
import warehouse as wh
import warehouse_mirror as wm
from odds_client import get_historical_event_odds, get_remaining_credits


SPORT = "americanfootball_nfl"
TEAM_MARKETS = ["h2h", "spreads", "totals"]
PROP_MARKETS = ["player_pass_yds", "player_pass_tds", "player_pass_attempts",
                "player_pass_completions", "player_rush_yds", "player_rush_attempts",
                "player_receptions", "player_reception_yds"]
TEAM_CSV = ",".join(TEAM_MARKETS)
PROP_ONLY = ",".join(PROP_MARKETS)
TEAM_PLUS = ",".join(TEAM_MARKETS + PROP_MARKETS)
BOOKS = ["draftkings", "fanduel", "pinnacle"]        # pinnacle = analysis-only reference
WORKERS = 10                                          # FETCH concurrency (writes stay serial)
COST_PROPS = 10 * len(PROP_MARKETS) * 1                       # 80
COST_TEAMPLUS = 10 * (len(TEAM_MARKETS) + len(PROP_MARKETS))  # 110

OFFSETS_H = list(range(6, 115, 6))    # −6h … −114h, nearest-kickoff first (NO close/−4h)
TEAM_OPENER_H = 114                   # team markets ride along only at this opener rung


def _src(offset_h):
    return f"ladder_{int(round(offset_h)):03d}h"


def _kinds_for(offset_h):
    return ["props", "team"] if int(round(offset_h)) == TEAM_OPENER_H else ["props"]


def _markets_for(offset_h):
    return TEAM_PLUS if int(round(offset_h)) == TEAM_OPENER_H else PROP_ONLY


def _rung_ts_sh(commence, offset_h):
    c0 = dt.datetime.fromisoformat(commence.replace("Z", "+00:00"))
    ts = (c0 - dt.timedelta(hours=offset_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ts, wh._hour_bucket(ts)


def _load_key():
    for mod in ("app", "config"):
        try:
            m = __import__(mod)
            if hasattr(m, "load_config"):
                k = m.load_config().get("odds_api_key")
                if k:
                    return k
        except Exception:
            pass
    import os
    return os.environ.get("ODDS_API_KEY", "")


def enumerate_games(seasons):
    """Distinct (event_id, commence, home, away, season) for NFL games in the warehouse."""
    games = {}
    for y in seasons:
        rows = wm.player_prop_lines(SPORT, date_from=f"{y}-01-01",
                                    date_to=f"{y}-12-31", bookmaker="draftkings") or []
        for r in rows:
            eid = r.get("event_id")
            if eid and eid not in games:
                games[eid] = {"event_id": eid, "commence": r.get("commence_time"),
                              "home": r.get("home"), "away": r.get("away"), "season": y}
    return [g for g in games.values() if g["commence"]]


def _preload_done():
    """{(event_id, kind, snapshot_hour)} already persisted under ladder_* sources — so a
    resume re-fetches nothing already durable in Azure."""
    eng = db_store.get_engine()
    done = set()
    with eng.connect() as c:
        for r in c.execute(text(
                "SELECT event_id, kind, snapshot_hour FROM odds_snapshot "
                "WHERE sport=:sp AND source LIKE 'ladder_%'"), {"sp": SPORT}):
            done.add((r[0], r[1], r[2]))
    return done


def _fetch_raw(key, game, offset_h, markets):
    ts, sh = _rung_ts_sh(game["commence"], offset_h)
    try:
        data, snap = get_historical_event_odds(
            key, SPORT, game["event_id"], ts, regions="us",
            markets=markets, bookmakers=BOOKS)
    except Exception:
        data, snap = None, None
    return game, data, snap, ts, sh


def _persist(game, data, snap, sh, offset_h, kinds, done):
    """Write the not-yet-durable kinds for one game/rung to Azure. Returns (written, skipped)."""
    written = skipped = 0
    commence = data.get("commence_time") or game["commence"]
    for kind in kinds:
        if (game["event_id"], kind, sh) in done:
            continue
        lines = im._per_book_lines(data, kind)
        if not lines:
            continue
        meta = {
            "sport": SPORT, "game_date": commence[:10], "event_id": game["event_id"],
            "kind": kind, "snapshot_hour": sh, "captured_at": snap,
            "commence_time": commence,
            "home": data.get("home_team") or game["home"],
            "away": data.get("away_team") or game["away"],
            "regions": "us", "markets": TEAM_CSV if kind == "team" else PROP_ONLY,
            "bookmakers": ",".join(BOOKS), "source": _src(offset_h),
        }
        ok = db_store.capture_odds_snapshot(meta, lines)
        if ok:
            written += 1
            done.add((game["event_id"], kind, sh))
        else:
            skipped += 1
    return written, skipped


def run(seasons, max_credits, dry_run):
    db_store.promote_secrets_from_toml()
    if not db_store.enabled():
        print("  Azure SQL is NOT configured — refusing to fetch without a durable sink "
              "(nothing local is durable). Set SQL_* secrets. Aborting, nothing spent.")
        return
    games = enumerate_games(seasons)
    ceiling = len(games) * ((len(OFFSETS_H) - 1) * COST_PROPS + COST_TEAMPLUS)
    print("=" * 96)
    print(f"  NFL LADDER BACKFILL → Azure — {len(games)} games × {len(OFFSETS_H)} rungs")
    print(f"  books={BOOKS}  props={COST_PROPS}/rung, team+props={COST_TEAMPLUS} at "
          f"−{TEAM_OPENER_H}h  (close/−4h already in warehouse — not re-fetched)")
    print(f"  ceiling ≈ {ceiling:,} credits   cap={max_credits:,}")
    print(f"  offsets (nearest-first): {OFFSETS_H}")
    print("=" * 96)

    done = _preload_done()
    print(f"  already-durable ladder snapshots in Azure: {len(done):,}")
    if dry_run:
        print("  (dry-run — no credits spent, nothing written)")
        return
    key = _load_key()
    if not key:
        print("  NO API KEY (app.load_config / ODDS_API_KEY). Aborting — nothing spent.")
        return

    spent_est = 0
    for offset in OFFSETS_H:
        kinds = _kinds_for(offset)
        markets = _markets_for(offset)
        per_call = COST_TEAMPLUS if "team" in kinds else COST_PROPS
        todo = []
        for g in games:
            _, sh = _rung_ts_sh(g["commence"], offset)
            if any((g["event_id"], k, sh) not in done for k in kinds):
                todo.append(g)
        if not todo:
            print(f"  −{offset:>3}h  fully durable already — skipping (0 credits)")
            continue
        est = len(todo) * per_call
        if spent_est + est > max_credits:
            print(f"  [cap] stopping before −{offset}h (est spent {spent_est:,}, "
                  f"next rung ~{est:,} would exceed cap {max_credits:,})")
            break
        written = skipped = empty = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(_fetch_raw, key, g, offset, markets) for g in todo]
            for f in as_completed(futs):
                game, data, snap, ts, sh = f.result()
                if not data:
                    empty += 1
                    continue
                w, s = _persist(game, data, snap, sh, offset, kinds, done)
                written += w
                skipped += s
        spent_est += est
        rem = get_remaining_credits()
        print(f"  −{offset:>3}h  todo={len(todo):>4}  written={written:>4}  "
              f"skipped={skipped:>3}  empty={empty:>3}  (est spent ≤ {spent_est:,}, "
              f"remaining {f'{rem:,}' if rem is not None else '?'})")
    rem = get_remaining_credits()
    print("=" * 96)
    print(f"  DONE. est spent ≤ {spent_est:,} credits; "
          f"remaining {f'{rem:,}' if rem is not None else '?'}.")
    print("  Next: refresh the local mirror so analysis sees the new rungs —")
    print("    python warehouse_mirror.py --sync --sport americanfootball_nfl "
          f"--seasons {','.join(seasons)} --refresh")


def main():
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2023,2024,2025,2026")
    ap.add_argument("--max-credits", type=int, default=1_450_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    run(seasons, args.max_credits, args.dry_run)


if __name__ == "__main__":
    main()
