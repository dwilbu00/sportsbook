"""Durable odds/line warehouse (roadmap 0.4).

Every odds snapshot the app fetches is otherwise ephemeral — Streamlit Cloud's
filesystem is wiped on restart, so there is no durable record of the lines the
model acted on. This module archives each fetched snapshot to Azure SQL (or a
local ``warehouse/`` directory in dev), giving us the closing-line history needed
for honest backtests, empirical correlations, and CLV on the bets actually
placed.

Design (locked)
---------------
* **SQL when configured.** ``capture_event_odds`` parses each payload into
  normalized ``odds_snapshot`` + ``odds_line`` rows (Phase B). This is the
  durable store in production.
* **Local fallback (dev/tests).** With no SQL backend, everything writes under a
  gitignored ``warehouse/`` directory as immutable, write-once snapshots:
  ``warehouse/{sport}/{game_date}/{event_id}/{kind}/{YYYYMMDDTHHZ}.json``.
  The hour-bucketed, colon-free name is idempotent given the 1-hour fetch cache
  and safe on Windows.
* **Manifest index (local path).** A directory scan is the only local
  enumeration, so each capture also appends to a per-``(sport, date)`` manifest
  (``.../_manifest.json``): ``capture_event_odds`` queues entries on a
  thread-safe accumulator and ``flush`` (called once by the app after both fetch
  waves) writes the manifests.

Self-contained on purpose: no import of recalibration, so the
``odds_client → warehouse`` capture hook never risks an import cycle. Every
public entry point fails closed.
"""
import json
import os
import threading
from datetime import date as _date, datetime, timedelta, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_TEAM_MARKETS = {"h2h", "spreads", "totals"}

# Thread-safe accumulator of manifest entries pending a flush().
_accumulator = []          # list of (sport, game_date, entry dict)
_acc_lock = threading.Lock()

# ── SQL backend (Azure SQL, Phase B) ──
# When db_store is importable AND configured, the warehouse stores normalized
# snapshots + extracted lines in SQL (no _manifest.json). A missing SQLAlchemy
# install or unset secret leaves _sql() False → the local warehouse/ path is used
# unchanged. Guarded import (self-contained module, no import cycle).
try:
    import db_store as _db
except Exception:  # pragma: no cover - SQLAlchemy absent
    _db = None


def _sql():
    return _db is not None and _db.enabled()


# ── SQL-off hardening (WS1 Layer B) ──
# Reads that silently return [] when SQL is off would make a mis-deployed prod
# look like an empty warehouse instead of failing loud. _ensure_durable raises
# only when a SQL deployment is signalled but SQL is actually off; it stays inert
# in dev/tests (no SQL_* secrets) and in healthy prod (SQL on).
_REQUIRE_SQL_ENV = "SPORTSBOOK_REQUIRE_SQL"
_SQL_SECRET_KEYS = ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD")


def _require_sql():
    if _db is not None:
        return _db.require_sql()
    val = os.environ.get(_REQUIRE_SQL_ENV)
    if val is not None:
        return val.strip().lower() in ("1", "true", "yes", "on")
    return any(os.environ.get(k) for k in _SQL_SECRET_KEYS)


def _ensure_durable(op):
    """Raise if a durable ``op`` would hit ephemeral local disk in a prod context.
    No-op unless SQL is off AND a SQL deployment is signalled."""
    if _sql() or not _require_sql():
        return
    raise RuntimeError(
        f"Refusing to {op}: the SQL backend is not enabled but a SQL deployment "
        f"is configured (SPORTSBOOK_REQUIRE_SQL or SQL_* secrets present). The "
        f"local warehouse/ directory is ephemeral and would silently read empty. "
        f"Fix the SQL_* secrets / the db_store import, or set "
        f"SPORTSBOOK_REQUIRE_SQL=0 for intentional local use.")


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


def storage_backend():
    """Human-readable active warehouse backend."""
    return "Azure SQL" if _sql() else "Local warehouse/"


# ──────────────────────────────────────────────────────────────────────────────
# Unified read/write (local warehouse/; the SQL path is handled by the callers)
# ──────────────────────────────────────────────────────────────────────────────

def _local_path(name):
    return os.path.join(SCRIPT_DIR, *name.split("/"))


def _read_json(name):
    """(status, obj, etag). Local files have no ETag (etag=None)."""
    path = _local_path(name)
    if not os.path.exists(path):
        return "missing", None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return "ok", json.load(f), None
    except Exception:
        return "error", None, None


def _write_json(name, obj, if_none_match=False, version=None, timeout=10):
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


def _books_str(bookmakers):
    if isinstance(bookmakers, (list, tuple)):
        return ",".join(str(b) for b in bookmakers) or None
    return str(bookmakers) if bookmakers else None


def _emit_team_lines(parsed, lines):
    """Best-across-books price per team/side (mirrors _extract_line's max)."""
    from odds_client import american_to_implied_prob

    def _implied(price):
        try:
            return american_to_implied_prob(int(price))
        except (TypeError, ValueError):
            return None

    for team, offers in (parsed.get("moneyline") or {}).items():
        prices = [o.get("price") for o in offers if o.get("price") is not None]
        if prices:
            best = max(prices)
            lines.append({"bet_type": "moneyline", "selection": team,
                          "price": best, "implied_prob": _implied(best)})
    for market, key, pt_field in (("spread", "spreads", "spread"),
                                  ("total", "totals", "line")):
        for selection, offers in (parsed.get(key) or {}).items():
            by_point = {}
            for o in offers:
                if o.get("price") is None:
                    continue
                by_point.setdefault(o.get(pt_field), []).append(o["price"])
            for point, prices in by_point.items():
                best = max(prices)
                lines.append({"bet_type": market, "selection": selection,
                              "point": point, "price": best,
                              "implied_prob": _implied(best)})


def _emit_prop_lines(parsed, lines):
    """Per-player over/under consensus line (mirrors _extract_line's prop path)."""
    from odds_client import american_to_implied_prob

    def _implied(price):
        try:
            return american_to_implied_prob(int(price))
        except (TypeError, ValueError):
            return None

    for prop_key, players in (parsed.get("props") or {}).items():
        for player, info in (players or {}).items():
            if not isinstance(info, dict):
                continue
            point = info.get("line")
            for direction, price_key, imp_key in (
                    ("OVER", "over_price", "over_implied"),
                    ("UNDER", "under_price", "under_implied")):
                price = info.get(price_key)
                if price is None:
                    continue
                implied = info.get(imp_key)
                # Match _extract_line: fall back to raw implied when the
                # consensus implied is absent.
                if implied is None:
                    implied = _implied(price)
                lines.append({"bet_type": "player_prop", "selection": player,
                              "player": player, "prop_key": prop_key,
                              "direction": direction, "point": point,
                              "price": price, "implied_prob": implied})


def _enumerate_lines(payload, fmt, kind):
    """Parse-on-capture: extract every closing-line descriptor from a payload,
    reproducing _extract_line's best-price (team) / consensus (props) logic so
    closing_line_for is a plain SQL lookup. Best-effort; returns [] on failure."""
    lines = []
    try:
        if fmt == "historical_odds_store":
            _emit_team_lines(payload, lines)     # already parsed shape
            _emit_prop_lines(payload, lines)
        elif kind == "props":
            from odds_client import parse_player_props
            _emit_prop_lines(parse_player_props(payload), lines)
        else:  # team (alt yields little from parse_game_odds — metadata only)
            from odds_client import parse_game_odds
            _emit_team_lines(parse_game_odds(payload), lines)
    except Exception:
        pass
    return lines


def _enrich_ids(sport, meta, lines):
    """Populate SFBB id/code columns on the snapshot meta + its lines (in place).

    MLB-gated, O(1) in-process lookups, fully fail-open: a map miss / SQL-off /
    non-MLB sport leaves the fields unset (None), never breaking the never-raise
    capture contract. Team lines resolve ``team_code`` off the selection (a team
    name; "Over"/"Under" totals fall through to None); prop lines resolve
    ``player_mlb_id`` off the player name; the snapshot gets home/away codes."""
    try:
        if not (sport or "").startswith("baseball"):
            return meta, lines
        import player_id_map
        meta["home_code"] = player_id_map.team_code_for_name(meta.get("home"))
        meta["away_code"] = player_id_map.team_code_for_name(meta.get("away"))
        prop_teams = (meta.get("home"), meta.get("away"))
        for ln in lines:
            if (ln.get("bet_type") or "") == "player_prop":
                ln["player_mlb_id"] = player_id_map.mlb_id_for_name(
                    ln.get("player"), teams=prop_teams)
            else:
                ln["team_code"] = player_id_map.team_code_for_name(
                    ln.get("selection"))
    except Exception:
        pass
    return meta, lines


def capture_event_odds(sport, event_id, regions, markets, bookmakers, payload,
                       captured_at=None):
    """Archive one fetched event-odds payload. Best-effort; never raises.

    SQL backend: parse the payload into normalized snapshot + line rows
    (write-once). Local backend: eagerly write the immutable snapshot and
    queue a manifest entry for the next flush(). A no-op when the payload lacks an
    event id or a commence date. ``captured_at`` overrides the timestamp (used by
    the historical backfill so past snapshots land under their true time)."""
    try:
        if not event_id or not isinstance(payload, dict):
            return
        commence = payload.get("commence_time")
        game_date = (commence or "")[:10]
        if not game_date:
            return
        captured_at = captured_at or _now_iso()
        kind = _kind_for_markets(markets)
        if _sql():
            _snap, _lines = _enrich_ids(sport, {
                "sport": sport, "game_date": game_date, "event_id": event_id,
                "kind": kind, "snapshot_hour": _hour_bucket(captured_at),
                "captured_at": captured_at, "commence_time": commence,
                "home": payload.get("home_team"), "away": payload.get("away_team"),
                "regions": regions, "markets": markets,
                "bookmakers": _books_str(bookmakers),
            }, _enumerate_lines(payload, "the-odds-api-v4-event-odds", kind))
            _db.capture_odds_snapshot(_snap, _lines)
            return
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
    if _sql():
        return _db.list_odds_snapshots(sport, game_date)
    status, manifest, _ = _read_json(manifest_name(sport, game_date))
    if status == "ok" and isinstance(manifest, dict):
        return list(manifest.get("snapshots", []))
    return _scan_local_snapshots(sport, game_date)


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


def _closing_sort_key(target_dt, captured_at):
    """Order snapshots so the CLOSING one sorts first: nearest AT-OR-BEFORE the
    commence time wins (smallest gap first), else nearest after.

    ``target_dt`` is the parsed commence datetime (or None → newest first);
    ``captured_at`` is a snapshot's capture timestamp (ISO str or datetime).
    (Was -(gap) inside closing_line_for, which picked the farthest/opening
    snapshot — a latent CLV bug surfaced by the Phase B tests.)"""
    captured = (captured_at if isinstance(captured_at, datetime)
                else _parse_utc(captured_at))
    if captured is None:
        return (2, 0.0)
    if target_dt is None:
        return (0, -captured.timestamp())
    if captured <= target_dt:
        return (0, (target_dt - captured).total_seconds())
    return (1, (captured - target_dt).total_seconds())


def closing_line_for(sport, game_date, event_id, bet_type, selection=None,
                     commence_time=None, point=None, player=None,
                     prop_key=None, direction=None, player_mlb_id=None,
                     team_code=None):
    """Closing line (best price + implied prob) for a bet, for CLV.

    Picks the snapshot for this event captured nearest at-or-before commence
    (else the nearest after), and extracts the bet's price. Returns
    ``{'price','implied_prob','captured_at'}`` or None. Best-effort.

    ``player_mlb_id``/``team_code`` (SQL path only) let the odds-line lookup
    prefer the canonical id over the name; the local JSON fallback has no id
    columns and stays name-based."""
    try:
        target = _parse_utc(commence_time)
        _order = lambda snap: _closing_sort_key(target, snap.get("captured_at"))

        if _sql():
            snaps = _db.odds_snapshots_for_event(sport, game_date, event_id)
            for snap in sorted(snaps, key=_order):
                line = _db.odds_line_lookup(
                    snap["id"], bet_type, selection=selection, point=point,
                    player=player, prop_key=prop_key, direction=direction,
                    player_mlb_id=player_mlb_id, team_code=team_code)
                if line is not None:
                    line["captured_at"] = snap.get("captured_at")
                    return line
            return None

        snaps = [s for s in list_snapshots(sport, game_date)
                 if s.get("event_id") == event_id]
        if not snaps:
            return None
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
# Team-market backtest store (assembled from the SQL warehouse)
# ──────────────────────────────────────────────────────────────────────────────

# Per-offer book label. The warehouse stores the best price ACROSS books (it
# doesn't record which one), so there's no real book name — but the live
# analyzers (analyze_moneyline_value) read offer["book"], so every assembled
# offer must carry the field to reproduce odds_client.parse_game_odds's shape.
_WH_BOOK = "warehouse"


def _wh_implied(price):
    """American price → implied prob, or None. (Local import: no import cycle.)"""
    try:
        from odds_client import american_to_implied_prob
        return american_to_implied_prob(int(price))
    except (TypeError, ValueError, ImportError):
        return None


def _fill_moneyline(entry, snap_rows, home, away):
    """Populate entry['moneyline'] = {team: [{price, implied_prob}]} for both
    teams (best price per team), matching the historical_odds store shape."""
    ml = {}
    for r in snap_rows:
        if r.get("bet_type") != "moneyline":
            continue
        team, price = r.get("selection"), r.get("price")
        if team is None or price is None:
            continue
        cur = ml.get(team)
        if cur is None or price > cur["price"]:
            ml[team] = {"book": _WH_BOOK, "price": price,
                        "implied_prob": r.get("implied_prob")}
    if home in ml and away in ml:
        entry["moneyline"] = {home: [ml[home]], away: [ml[away]]}


def _fill_spreads(entry, snap_rows, home, away):
    """Populate entry['spreads'] with the mirrored home ``p`` / away ``-p`` pair.

    Groups each team's offers by point (best price per point), then picks the
    mirrored pair with the tightest two-way overround (tiebreak smallest |p|).
    Omits the market entirely when no mirrored pair exists — so _spread_market
    only ever devigs a genuine two-way line."""
    home_by_pt, away_by_pt = {}, {}
    for r in snap_rows:
        if r.get("bet_type") != "spread":
            continue
        team, price, point = r.get("selection"), r.get("price"), r.get("point")
        if price is None or point is None:
            continue
        bucket = (home_by_pt if team == home
                  else away_by_pt if team == away else None)
        if bucket is None:
            continue
        if bucket.get(point) is None or price > bucket[point]:
            bucket[point] = price
    best = None  # (overround, |point|, point, home_price, away_price)
    for point, hp in home_by_pt.items():
        ap = away_by_pt.get(-point)
        if ap is None:
            continue
        ih, ia = _wh_implied(hp), _wh_implied(ap)
        if ih is None or ia is None:
            continue
        cand = (ih + ia, abs(point), point, hp, ap)
        if best is None or cand < best:
            best = cand
    if best is not None:
        _, _, point, hp, ap = best
        entry["spreads"] = {
            home: [{"book": _WH_BOOK, "spread": point, "price": hp}],
            away: [{"book": _WH_BOOK, "spread": -point, "price": ap}]}


def _fill_totals(entry, snap_rows):
    """Populate entry['totals'] = {'Over'/'Under': [{line, price}]} for the same
    line L on both sides — tiebreak tightest two-way overround, then line."""
    over_by_pt, under_by_pt = {}, {}
    for r in snap_rows:
        if r.get("bet_type") != "total":
            continue
        sel, price, point = (r.get("selection") or ""), r.get("price"), r.get("point")
        if price is None or point is None:
            continue
        low = sel.lower()
        bucket = (over_by_pt if low == "over"
                  else under_by_pt if low == "under" else None)
        if bucket is None:
            continue
        if bucket.get(point) is None or price > bucket[point]:
            bucket[point] = price
    best = None  # (overround, line, over_price, under_price)
    for point, op in over_by_pt.items():
        up = under_by_pt.get(point)
        if up is None:
            continue
        io, iu = _wh_implied(op), _wh_implied(up)
        if io is None or iu is None:
            continue
        cand = (io + iu, point, op, up)
        if best is None or cand < best:
            best = cand
    if best is not None:
        _, line, op, up = best
        entry["totals"] = {
            "Over": [{"book": _WH_BOOK, "line": line, "price": op}],
            "Under": [{"book": _WH_BOOK, "line": line, "price": up}]}


def _assemble_team_entry(event_id, rows):
    """Build one historical_odds-store entry (moneyline/spreads/totals) for an
    event from its warehoused team-market line rows. Picks the closing snapshot
    (nearest at-or-before commence via _closing_sort_key) and reads that
    snapshot's lines. Returns the entry dict, or None if no market assembled."""
    if not rows:
        return None
    first = rows[0]
    commence = first.get("commence_time")
    home, away = first.get("home"), first.get("away")
    target = _parse_utc(commence)

    by_snap, snap_captured = {}, {}
    for r in rows:
        sid = r.get("snapshot_id")
        by_snap.setdefault(sid, []).append(r)
        snap_captured[sid] = r.get("captured_at")
    closing_sid = min(
        by_snap, key=lambda sid: _closing_sort_key(target, snap_captured.get(sid)))
    snap_rows = by_snap[closing_sid]

    # Initialize the three market keys to {} — the historical_odds shape this
    # reproduces (odds_client.parse_game_odds) ALWAYS carries them, and the
    # default engine="live" analyzers hard-subscript game_odds["moneyline"/
    # "spreads"/"totals"], so a missing key would KeyError-abort the backtest.
    # _fill_* overwrite when a genuine two-way market is present.
    entry = {"commence_time": commence, "home_team": home, "away_team": away,
             "home_code": first.get("home_code"),
             "away_code": first.get("away_code"),
             "event_id": event_id, "props": {},
             "moneyline": {}, "spreads": {}, "totals": {}}
    _fill_moneyline(entry, snap_rows, home, away)
    _fill_spreads(entry, snap_rows, home, away)
    _fill_totals(entry, snap_rows)
    if not (entry.get("moneyline") or entry.get("spreads")
            or entry.get("totals")):
        return None   # {} for all three is falsy → no usable market
    return entry


def load_team_market_store(sport_key, dates=None):
    """Assemble a historical_odds-shaped store from the SQL warehouse's captured
    team-market lines, for the team-market backtest.

    Returns the exact shape historical_odds.load_store produces
    ({'sport_key','bookmaker','games': {game_key: entry}}) so
    backtest._build_odds_lookup / _moneyline_market / _spread_market /
    _total_market consume it unchanged — one entry per event built from that
    event's closing snapshot. SQL-only and best-effort: returns an empty store
    when SQL is off or on any error."""
    empty = {"sport_key": sport_key, "games": {},
             "bookmaker": "warehouse (best-of-book, closing)"}
    _ensure_durable("read the team-market warehouse")
    if not _sql():
        return empty
    try:
        rows = _db.team_market_lines(sport_key, dates=dates)
    except Exception:
        return empty
    if not rows:
        return empty
    try:
        import historical_odds as store_mod
    except Exception:
        return empty
    by_event = {}
    for r in rows:
        by_event.setdefault(r.get("event_id"), []).append(r)
    games = {}
    for event_id, ev_rows in by_event.items():
        entry = _assemble_team_entry(event_id, ev_rows)
        if entry is None:
            continue
        key = store_mod.game_key(entry.get("commence_time"),
                                 entry.get("home_team"), entry.get("away_team"))
        games[key] = entry
    empty["games"] = games
    return empty


def _assemble_prop_entries(event_id, rows, sport_key):
    """One closing-line harvest row per (player, prop_key) for an event: pick the
    closing snapshot (nearest at-or-before commence via _closing_sort_key) and
    combine that snapshot's OVER/UNDER rows into {line, over_price, under_price}.
    ``game_date`` is left None here and set to the US-Eastern date by the caller.
    Returns a list of harvest-shaped dicts (may be empty)."""
    if not rows:
        return []
    first = rows[0]
    commence = first.get("commence_time")
    home, away = first.get("home"), first.get("away")
    target = _parse_utc(commence)

    by_snap, snap_captured = {}, {}
    for r in rows:
        sid = r.get("snapshot_id")
        by_snap.setdefault(sid, []).append(r)
        snap_captured[sid] = r.get("captured_at")
    closing_sid = min(
        by_snap, key=lambda sid: _closing_sort_key(target, snap_captured.get(sid)))

    combined = {}   # (player, prop_key) -> {line, over_price, under_price, mlb_id}
    for r in by_snap[closing_sid]:
        player, prop_key = r.get("player"), r.get("prop_key")
        point = r.get("point")
        if not player or not prop_key or point is None:
            continue
        e = combined.setdefault((player, prop_key),
                                {"line": point, "over_price": None,
                                 "under_price": None, "player_mlb_id": None})
        if e.get("player_mlb_id") is None:
            e["player_mlb_id"] = r.get("player_mlb_id")   # first non-None wins
        direction = (r.get("direction") or "").upper()
        if direction == "OVER":
            e["over_price"] = r.get("price")
        elif direction == "UNDER":
            e["under_price"] = r.get("price")

    return [{
        "sport_key": sport_key, "game_date": None, "commence_time": commence,
        "home_team": home, "away_team": away, "event_id": event_id,
        "player": player, "player_mlb_id": e.get("player_mlb_id"),
        "prop_key": prop_key, "line": e["line"],
        "over_price": e["over_price"], "under_price": e["under_price"],
    } for (player, prop_key), e in combined.items()]


def load_prop_lines(sport_key, dates=None):
    """Assemble player-prop closing-line rows from the SQL warehouse for the
    offline real-line calibration refit.

    Returns the shape book_line_calibration.harvest_book_lines_from_store emits,
    plus ``event_id``/``commence_time``:
    {sport_key, game_date(ET), commence_time, home_team, away_team, event_id,
     player, prop_key, line, over_price, under_price} — one CLOSING line per
    (event, player, prop_key). ``game_date`` is the US-Eastern calendar date
    (UTC at rest, ET on read; see pricing_common.et_local_date). SQL-only and
    best-effort: returns [] when SQL is off or on any error."""
    _ensure_durable("read the player-prop warehouse")
    if not _sql():
        return []
    try:
        rows = _db.player_prop_lines(sport_key, dates=dates)
    except Exception:
        return []
    if not rows:
        return []
    try:
        from pricing_common import et_local_date
    except Exception:
        def et_local_date(c):
            return (str(c)[:10] if c else None)
    by_event = {}
    for r in rows:
        by_event.setdefault(r.get("event_id"), []).append(r)
    out = []
    for event_id, ev_rows in by_event.items():
        for row in _assemble_prop_entries(event_id, ev_rows, sport_key):
            row["game_date"] = et_local_date(row.get("commence_time"))
            out.append(row)
    return out


def doubleheader_event_ids(rows):
    """The event ids belonging to a true same-day doubleheader — a
    (game_date, home_team, away_team) matchup carrying >1 distinct event_id on one
    calendar date (game_date must already be in US Eastern so consecutive-day
    series games, which differ by ET date, are NOT flagged). Calibration drops
    these: a doubleheader's two lines can't be cleanly attributed to the right box
    score, and game 2 is mis-projected (same pre-doubleheader inputs). Returns a
    set of event_ids."""
    by_matchup = {}
    for r in rows:
        key = (r.get("game_date"), r.get("home_team"), r.get("away_team"))
        eid = r.get("event_id")
        if eid:
            by_matchup.setdefault(key, set()).add(eid)
    dh = set()
    for ev_ids in by_matchup.values():
        if len(ev_ids) > 1:
            dh |= ev_ids
    return dh


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
        if _sql():
            _snap, _lines = _enrich_ids(sport_key, {
                "sport": sport_key, "game_date": game_date,
                "event_id": event_id, "kind": "seed",
                "snapshot_hour": _hour_bucket(captured_at),
                "captured_at": captured_at, "commence_time": commence,
                "home": entry.get("home_team"), "away": entry.get("away_team"),
                "regions": None, "markets": "seed", "bookmakers": None,
            }, _enumerate_lines(payload, "historical_odds_store", "seed"))
            if _db.capture_odds_snapshot(_snap, _lines):
                written += 1
            continue
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

    # Target the SQL backend when the SQL_* secrets are configured (mirrors the
    # app's boot promotion; outside Streamlit they aren't in the env yet). Falls
    # back to the local warehouse/ when SQL isn't configured or db_store is
    # unavailable.
    try:
        import db_store
        db_store.promote_secrets_from_toml()
    except Exception:
        pass

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
