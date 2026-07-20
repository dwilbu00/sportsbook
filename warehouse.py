"""Durable odds/line warehouse (roadmap 0.4).

Every odds snapshot the app fetches is otherwise ephemeral — Streamlit Cloud's
filesystem is wiped on restart, so there is no durable record of the lines the
model acted on. This module archives each fetched snapshot to Azure Blob (free —
it reuses payloads already fetched), giving us the closing-line history needed
for honest backtests, empirical correlations, and CLV on the bets actually
placed.

Design (locked)
---------------
* **Immutable, write-once snapshots.** Each capture PUTs one blob with
  ``If-None-Match:*`` and is never rewritten:
  ``warehouse/{sport}/{game_date}/{event_id}/{kind}/{YYYYMMDDTHHZ}.json``.
  The hour-bucketed, colon-free name is idempotent given the 1-hour fetch cache
  and safe on the local-fallback filesystem (Windows).
* **Manifest index (no list permission).** The container SAS has no ``list``
  right, so reads can't enumerate blobs. Each capture also appends to a
  per-``(sport, date)`` manifest blob (``.../_manifest.json``, a single
  deterministic GET) via a read-modify-write ETag loop. The prefix-structured
  names still enable native list-prefix later if ``l`` is granted.
* **Hot-path safe.** ``capture_event_odds`` does the immutable PUT eagerly
  (short timeout, swallowed; unique names → no contention) and appends to a
  thread-safe accumulator; ``flush`` (called once by the app after both fetch
  waves) updates the manifests.
* **Local fallback.** With no blob URL, everything writes under a gitignored
  ``warehouse/`` directory so tests and offline runs work unchanged.

Self-contained on purpose: it re-implements the ~40 lines of SAS plumbing rather
than importing recalibration, so the ``odds_client → warehouse`` capture hook
never risks an import cycle. Every public entry point fails closed.
"""
import json
import os
import threading
from datetime import date as _date, datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLOB_URL_ENV = "PREDICTION_LOG_BLOB_URL"

_TEAM_MARKETS = {"h2h", "spreads", "totals"}

# Thread-safe accumulator of manifest entries pending a flush().
_accumulator = []          # list of (sport, game_date, entry dict)
_acc_lock = threading.Lock()

# sport nickname -> The Odds API sport_key (CLI convenience).
_SPORT_KEYS = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
}


# ──────────────────────────────────────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hour_bucket(iso_ts):
    """'2026-07-16T14:23:01Z' -> '20260716T14Z' (colon-free, hour-bucketed)."""
    dt = _parse_utc(iso_ts) or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%HZ")


# ──────────────────────────────────────────────────────────────────────────────
# SAS plumbing (self-contained; container-scoped, no new secret)
# ──────────────────────────────────────────────────────────────────────────────

def _blob_base():
    """The prediction-log SAS URL (container-scoped), or ''."""
    url = os.environ.get(BLOB_URL_ENV, "").strip()
    if url:
        return url
    secrets_path = os.path.join(SCRIPT_DIR, ".streamlit", "secrets.toml")
    try:
        import tomllib
        with open(secrets_path, "rb") as f:
            value = tomllib.load(f).get(BLOB_URL_ENV)
        return str(value).strip() if value else ""
    except (ImportError, OSError, TypeError, ValueError):
        return ""


def storage_backend():
    """Human-readable active warehouse backend."""
    return "Azure Blob" if _blob_base() else "Local warehouse/"


def _blob_url_for(name):
    """Container-root-relative blob URL for ``name`` (e.g. 'warehouse/...').

    The SAS is container-scoped (``sr=c``), so any path under the container is
    reachable with the same token. We derive the container root from the base
    URL rather than assuming the log lives at the container root."""
    base = _blob_base()
    if not base:
        return ""
    from urllib.parse import urlsplit
    parts = urlsplit(base)
    segments = parts.path.split("/")
    container = segments[1] if len(segments) > 1 else ""
    root = f"{parts.scheme}://{parts.netloc}/{container}"
    url = f"{root}/{name}"
    return f"{url}?{parts.query}" if parts.query else url


def _get_blob(name):
    """(status, obj, etag): 'ok' | 'missing' | 'error'."""
    url = _blob_url_for(name)
    if not url:
        return "missing", None, None
    import requests
    try:
        resp = requests.get(url, timeout=10)
    except Exception:
        return "error", None, None
    if resp.status_code == 404:
        return "missing", None, None
    if resp.status_code != 200:
        return "error", None, None
    try:
        return "ok", json.loads(resp.text), resp.headers.get("ETag")
    except Exception:
        return "error", None, resp.headers.get("ETag")


def _put_blob(name, obj, if_none_match=False, version=None, timeout=10):
    """PUT a JSON blob. Returns True on success, False on conflict/failure.

    ``if_none_match`` → create-only (write-once). ``version`` → If-Match."""
    url = _blob_url_for(name)
    if not url:
        return False
    import requests
    headers = {
        "Content-Type": "application/json",
        "x-ms-blob-type": "BlockBlob",
        "x-ms-version": "2023-11-03",
    }
    if if_none_match:
        headers["If-None-Match"] = "*"
    elif version:
        headers["If-Match"] = version
    try:
        resp = requests.put(url, data=json.dumps(obj).encode("utf-8"),
                            headers=headers, timeout=timeout)
    except Exception:
        return False
    return resp.status_code in (200, 201)


# ──────────────────────────────────────────────────────────────────────────────
# Unified read/write (blob when configured, else local warehouse/)
# ──────────────────────────────────────────────────────────────────────────────

def _local_path(name):
    return os.path.join(SCRIPT_DIR, *name.split("/"))


def _read_json(name):
    """(status, obj, etag). Local files have no ETag (etag=None)."""
    if _blob_base():
        return _get_blob(name)
    path = _local_path(name)
    if not os.path.exists(path):
        return "missing", None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return "ok", json.load(f), None
    except Exception:
        return "error", None, None


def _write_json(name, obj, if_none_match=False, version=None, timeout=10):
    if _blob_base():
        return _put_blob(name, obj, if_none_match=if_none_match,
                         version=version, timeout=timeout)
    path = _local_path(name)
    if if_none_match and os.path.exists(path):
        return False  # write-once already present
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────────────────────

def _kind_for_markets(markets):
    ms = {m.strip() for m in (markets or "").split(",") if m.strip()}
    if any("alternate" in m for m in ms):
        return "alt"
    if ms and ms <= _TEAM_MARKETS:
        return "team"
    return "props"


def snapshot_name(sport, game_date, event_id, kind, captured_at):
    return (f"warehouse/{sport}/{game_date}/{event_id}/{kind}/"
            f"{_hour_bucket(captured_at)}.json")


def manifest_name(sport, game_date):
    return f"warehouse/{sport}/{game_date}/_manifest.json"


def capture_event_odds(sport, event_id, regions, markets, bookmakers, payload,
                       captured_at=None):
    """Archive one fetched event-odds payload. Best-effort; never raises.

    Eagerly PUTs the immutable snapshot (write-once, short timeout) and queues a
    manifest entry for the next flush(). A no-op when the payload lacks an event
    id or a commence date. ``captured_at`` overrides the timestamp (used by the
    historical backfill so past snapshots land under their true snapshot time)."""
    try:
        if not event_id or not isinstance(payload, dict):
            return
        commence = payload.get("commence_time")
        game_date = (commence or "")[:10]
        if not game_date:
            return
        captured_at = captured_at or _now_iso()
        kind = _kind_for_markets(markets)
        name = snapshot_name(sport, game_date, event_id, kind, captured_at)
        envelope = {
            "captured_at": captured_at,
            "sport": sport,
            "event_id": event_id,
            "regions": regions,
            "markets": markets,
            "bookmakers": bookmakers,
            "kind": kind,
            "commence_time": commence,
            "home": payload.get("home_team"),
            "away": payload.get("away_team"),
            "format": "the-odds-api-v4-event-odds",
            "payload": payload,
        }
        # Eager immutable snapshot (unique name → no contention; swallow 412).
        _write_json(name, envelope, if_none_match=True, timeout=5)
        entry = {
            "name": name,
            "event_id": event_id,
            "kind": kind,
            "commence_time": commence,
            "home": payload.get("home_team"),
            "away": payload.get("away_team"),
            "captured_at": captured_at,
            "markets": markets,
        }
        with _acc_lock:
            _accumulator.append((sport, game_date, entry))
    except Exception:
        pass


def _update_manifest(sport, game_date, entries, max_retries=5):
    """Merge ``entries`` into the per-(sport,date) manifest (RMW). Returns bool."""
    name = manifest_name(sport, game_date)
    for _ in range(max_retries):
        status, manifest, etag = _read_json(name)
        if status == "error":
            return False
        create = status == "missing" or not isinstance(manifest, dict)
        if create:
            manifest = {"sport": sport, "game_date": game_date, "snapshots": []}
        manifest.setdefault("snapshots", [])
        have = {s.get("name") for s in manifest["snapshots"]}
        added = False
        for entry in entries:
            if entry.get("name") not in have:
                manifest["snapshots"].append(entry)
                have.add(entry.get("name"))
                added = True
        if not added:
            return True  # manifest already current
        manifest["updated"] = _now_iso()
        if _write_json(name, manifest, if_none_match=create, version=etag):
            return True
        # Conflict (another writer won the race) — re-read and retry.
    return False


def flush():
    """Fold queued snapshot entries into their per-day manifests. Never raises.

    Called once by the app after both fetch waves. A no-op with an empty queue,
    so it is cheap to call on every rerun."""
    try:
        with _acc_lock:
            pending = list(_accumulator)
            _accumulator.clear()
        if not pending:
            return 0
        by_day = {}
        for sport, game_date, entry in pending:
            by_day.setdefault((sport, game_date), []).append(entry)
        flushed = 0
        for (sport, game_date), entries in by_day.items():
            if _update_manifest(sport, game_date, entries):
                flushed += len(entries)
        return flushed
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Readers
# ──────────────────────────────────────────────────────────────────────────────

def list_snapshots(sport, game_date):
    """Manifest entries for a (sport, date). Falls back to a local dir scan."""
    status, manifest, _ = _read_json(manifest_name(sport, game_date))
    if status == "ok" and isinstance(manifest, dict):
        return list(manifest.get("snapshots", []))
    if not _blob_base():
        return _scan_local_snapshots(sport, game_date)
    return []


def _scan_local_snapshots(sport, game_date):
    base = _local_path(f"warehouse/{sport}/{game_date}")
    out = []
    if not os.path.isdir(base):
        return out
    for root, _dirs, files in os.walk(base):
        for fn in files:
            if not fn.endswith(".json") or fn == "_manifest.json":
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, SCRIPT_DIR).replace(os.sep, "/")
            try:
                with open(full, "r", encoding="utf-8") as f:
                    env = json.load(f)
            except Exception:
                continue
            out.append({
                "name": rel,
                "event_id": env.get("event_id"),
                "kind": env.get("kind"),
                "commence_time": env.get("commence_time"),
                "home": env.get("home"),
                "away": env.get("away"),
                "captured_at": env.get("captured_at"),
                "markets": env.get("markets"),
            })
    out.sort(key=lambda e: e.get("captured_at") or "")
    return out


def read_snapshot(name):
    """Return one snapshot envelope, or None."""
    status, obj, _ = _read_json(name)
    return obj if status == "ok" else None


def _extract_line(env, bet_type, selection, point=None, player=None,
                  prop_key=None, direction=None):
    """Best price + implied prob for a bet descriptor within one snapshot, or
    None. Handles both raw the-odds-api payloads and seeded parsed stores."""
    payload = env.get("payload")
    if not isinstance(payload, dict):
        return None
    fmt = env.get("format")
    try:
        from odds_client import (parse_game_odds, parse_player_props,
                                 american_to_implied_prob)
    except Exception:
        return None

    def _implied(price):
        try:
            return american_to_implied_prob(int(price))
        except (TypeError, ValueError):
            return None

    bt = (bet_type or "").lower()
    try:
        if fmt == "historical_odds_store":
            parsed = payload
        elif bt == "player_prop":
            parsed = parse_player_props(payload)
        else:
            parsed = parse_game_odds(payload)
    except Exception:
        return None

    if bt in ("moneyline", "h2h"):
        offers = (parsed.get("moneyline") or {}).get(selection) or []
        prices = [o.get("price") for o in offers if o.get("price") is not None]
        if not prices:
            return None
        best = max(prices)
        return {"price": best, "implied_prob": _implied(best)}
    if bt in ("spread", "spreads"):
        offers = (parsed.get("spreads") or {}).get(selection) or []
        if point is not None:
            offers = [o for o in offers
                      if abs(float(o.get("spread", 1e9)) - float(point)) < 1e-9] or offers
        prices = [o.get("price") for o in offers if o.get("price") is not None]
        if not prices:
            return None
        best = max(prices)
        return {"price": best, "implied_prob": _implied(best)}
    if bt in ("total", "totals"):
        label = "Under" if (selection or "").lower() == "under" else "Over"
        offers = (parsed.get("totals") or {}).get(label) or []
        if point is not None:
            offers = [o for o in offers
                      if abs(float(o.get("line", 1e9)) - float(point)) < 1e-9] or offers
        prices = [o.get("price") for o in offers if o.get("price") is not None]
        if not prices:
            return None
        best = max(prices)
        return {"price": best, "implied_prob": _implied(best)}
    if bt == "player_prop":
        by_player = (parsed.get("props") or {}).get(prop_key) or {}
        info = by_player.get(player)
        if not info:
            return None
        if (direction or "OVER").upper() == "UNDER":
            price = info.get("under_price")
            implied = info.get("under_implied")
        else:
            price = info.get("over_price")
            implied = info.get("over_implied")
        if price is None:
            return None
        return {"price": price,
                "implied_prob": implied if implied is not None else _implied(price)}
    return None


def closing_line_for(sport, game_date, event_id, bet_type, selection=None,
                     commence_time=None, point=None, player=None,
                     prop_key=None, direction=None):
    """Closing line (best price + implied prob) for a bet, for CLV.

    Picks the snapshot for this event captured nearest at-or-before commence
    (else the nearest after), and extracts the bet's price. Returns
    ``{'price','implied_prob','captured_at'}`` or None. Best-effort."""
    try:
        snaps = [s for s in list_snapshots(sport, game_date)
                 if s.get("event_id") == event_id]
        if not snaps:
            return None
        target = _parse_utc(commence_time)

        def _order(snap):
            captured = _parse_utc(snap.get("captured_at"))
            if captured is None:
                return (2, 0.0)
            if target is None:
                return (0, -captured.timestamp())
            if captured <= target:
                return (0, -(target - captured).total_seconds())
            return (1, (captured - target).total_seconds())

        for snap in sorted(snaps, key=_order):
            env = read_snapshot(snap.get("name"))
            if not env:
                continue
            line = _extract_line(env, bet_type, selection, point=point,
                                 player=player, prop_key=prop_key,
                                 direction=direction)
            if line is not None:
                line["captured_at"] = snap.get("captured_at")
                return line
        return None
    except Exception:
        return None


def join_predictions_to_lines(sport, dates):
    """Join resolved prediction-log rows to their warehoused open/close lines.

    Report-only convenience (CLI ``--report``). Reads the prediction log lazily
    to keep the module free of an import-time recalibration dependency. Yields
    dicts with the logged prob/outcome plus the event's snapshot span."""
    try:
        import recalibration
        log_rows = recalibration.read_prediction_log()
    except Exception:
        log_rows = []
    want_dates = set(dates or [])
    # Index snapshots by (event_id) across the requested dates.
    snaps_by_event = {}
    for d in want_dates:
        for snap in list_snapshots(sport, d):
            snaps_by_event.setdefault(snap.get("event_id"), []).append(snap)
    joined = []
    for row in log_rows:
        if row.get("sport_key") != sport:
            continue
        if want_dates and (row.get("game_date") or "")[:10] not in want_dates:
            continue
        event_id = row.get("event_id")
        snaps = snaps_by_event.get(event_id) or []
        captured = sorted(s.get("captured_at") or "" for s in snaps)
        joined.append({
            "event_id": event_id,
            "prop_key": row.get("prop_key"),
            "player": row.get("player"),
            "line": row.get("line"),
            "prob": (row.get("final_prob")
                     if row.get("final_prob") is not None else row.get("raw_prob")),
            "outcome": row.get("outcome"),
            "resolved": row.get("resolved"),
            "n_snapshots": len(snaps),
            "open_captured_at": captured[0] if captured else None,
            "close_captured_at": captured[-1] if captured else None,
        })
    return joined


# ──────────────────────────────────────────────────────────────────────────────
# Seed from the existing historical_odds store
# ──────────────────────────────────────────────────────────────────────────────

def seed_from_store(sport_key, label=""):
    """Backfill the warehouse from historical_odds/<sport>.json (one snapshot
    per stored game). Returns the number of snapshots written. Best-effort."""
    try:
        import historical_odds as store_mod
    except Exception:
        return 0
    store = store_mod.load_store(sport_key, label)
    games = store.get("games") or {}
    written = 0
    for entry in games.values():
        commence = entry.get("commence_time")
        event_id = entry.get("event_id") or store_mod.game_key(
            commence, entry.get("home_team"), entry.get("away_team"))
        game_date = (commence or "")[:10]
        if not game_date:
            continue
        captured_at = (entry.get("props_snapshot_timestamp")
                       or entry.get("snapshot_timestamp") or commence
                       or _now_iso())
        payload = {
            "moneyline": entry.get("moneyline", {}),
            "spreads": entry.get("spreads", {}),
            "totals": entry.get("totals", {}),
            "props": entry.get("props", {}),
        }
        envelope = {
            "captured_at": captured_at,
            "sport": sport_key,
            "event_id": event_id,
            "kind": "seed",
            "commence_time": commence,
            "home": entry.get("home_team"),
            "away": entry.get("away_team"),
            "format": "historical_odds_store",
            "payload": payload,
        }
        name = snapshot_name(sport_key, game_date, event_id, "seed", captured_at)
        if _write_json(name, envelope, if_none_match=True):
            written += 1
        _update_manifest(sport_key, game_date, [{
            "name": name,
            "event_id": event_id,
            "kind": "seed",
            "commence_time": commence,
            "home": entry.get("home_team"),
            "away": entry.get("away_team"),
            "captured_at": captured_at,
            "markets": "seed",
        }])
    return written


# ──────────────────────────────────────────────────────────────────────────────
# CLI: report / seed
# ──────────────────────────────────────────────────────────────────────────────

def _report(sport_key, dates):
    print(f"=== Warehouse report — {sport_key} ({storage_backend()}) ===")
    if not dates:
        print("  No dates given (use --dates YYYY-MM-DD,YYYY-MM-DD). "
              "Nothing to enumerate without list permission.")
        return
    total_snaps = 0
    events = set()
    for d in dates:
        snaps = list_snapshots(sport_key, d)
        total_snaps += len(snaps)
        for s in snaps:
            events.add(s.get("event_id"))
        print(f"  {d}: {len(snaps)} snapshot(s), "
              f"{len({s.get('event_id') for s in snaps})} event(s)")
    print(f"  Total: {total_snaps} snapshot(s) across {len(events)} event(s).")
    joined = join_predictions_to_lines(sport_key, dates)
    resolved = [j for j in joined if j.get("resolved")]
    print(f"  Prediction-log rows for these dates: {len(joined)} "
          f"({len(resolved)} resolved).")
    for j in joined[:10]:
        print(f"    {j['player']} {j['prop_key']} line={j['line']} "
              f"prob={j['prob']} outcome={j['outcome']} "
              f"snaps={j['n_snapshots']} "
              f"[{j['open_captured_at']} .. {j['close_captured_at']}]")


def _main_cli():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", choices=list(_SPORT_KEYS.keys()), default="mlb")
    p.add_argument("--report", action="store_true",
                   help="Summarize warehoused snapshots for --dates.")
    p.add_argument("--seed-from-store", action="store_true",
                   help="Backfill from historical_odds/<sport>.json.")
    p.add_argument("--dates", default="",
                   help="Comma-separated game dates (YYYY-MM-DD) for --report.")
    p.add_argument("--label", default="",
                   help="historical_odds store label to seed from.")
    args = p.parse_args()

    sport_key = _SPORT_KEYS[args.sport]
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    if args.seed_from_store:
        print(f"Seeding warehouse for {sport_key} from historical_odds store "
              f"({storage_backend()})...")
        n = seed_from_store(sport_key, args.label)
        print(f"  Wrote {n} snapshot(s).")

    if args.report or not args.seed_from_store:
        _report(sport_key, dates)


if __name__ == "__main__":
    try:
        from cli_encoding import configure_stdio
        configure_stdio()
    except Exception:
        pass
    _main_cli()
