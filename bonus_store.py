"""bonus_store.py — durable persistence for the user's active promos/boosts.

Stores the active-bonus list as one entry in the same generic KV store the Kelly knobs use
(``app_settings`` table → AZURE SQL, the system of record; durable across sessions and redeploys).
One JSON string under setting_key="active_bonuses". Seeds from bonus.LIVE_BONUSES the first time.
Best-effort; the UI still works off session_state if the store is unavailable.
"""
import json
from dataclasses import asdict

import bonus as bonuslib

_SETTINGS_FILE = "app_settings.jsonl"
_KEY = "active_bonuses"
_FIELDS = ("bet_type", "boost_pct", "min_odds_leg", "min_odds_overall",
           "min_legs", "max_wager", "min_wager", "book", "label", "sport",
           "markets")


def _clean(d):
    """Coerce a stored/edited dict into a valid Bonus kwargs dict."""
    out = {}
    for k in _FIELDS:
        if k not in d:
            continue
        v = d[k]
        if k in ("bet_type", "book", "label", "sport"):
            out[k] = str(v)
        elif k == "min_legs":
            out[k] = int(v)
        elif k == "markets":
            out[k] = tuple(str(m) for m in (v or ()) if m)   # scope; () = all
        else:
            out[k] = float(v)
    return out


def load_bonuses():
    """List[Bonus] from the KV store; seed with bonus.LIVE_BONUSES if nothing stored."""
    try:
        import recalibration
        rows, _ = recalibration._read_ndjson_blob(_SETTINGS_FILE, use_cache=False)
        for r in (rows or []):
            if r.get("setting_key") == _KEY:
                data = json.loads(r.get("setting_value") or "[]")
                return [bonuslib.Bonus(**_clean(d)) for d in data]
    except Exception:
        pass
    return list(bonuslib.LIVE_BONUSES)


def save_bonuses(bonuses):
    """Upsert the whole active-bonus list as one JSON KV entry. Returns 1 on write, 0 otherwise."""
    payload = json.dumps([{k: asdict(b)[k] for k in _FIELDS} for b in bonuses])

    def upsert(rows):
        import datetime
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for r in rows:
            if r.get("setting_key") == _KEY:
                if r.get("setting_value") == payload:
                    return 0
                r["setting_value"] = payload
                r["updated_at"] = ts
                return 1
        rows.append({"setting_key": _KEY, "setting_value": payload, "updated_at": ts})
        return 1

    try:
        import recalibration
        return recalibration.mutate_ndjson_log(_SETTINGS_FILE, upsert) or 0
    except Exception:
        return 0
