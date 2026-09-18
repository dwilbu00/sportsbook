"""book_calibration.py — per-market shrink on the BOOK's de-vigged probability.

The bonus system prices legs purely off book odds, so the boost edge is only as
honest as the book's probability. `refit_calibration.py --book-calib --save-recal`
measures the book's own calibration per market and, when a shrink beats the raw book
OOS, stores the winning map here. The bonus optimizer applies it to each leg's P so
boost picks price off the CALIBRATED book number.

Committed JSON per sport: calibration/book_calibration_<sport>.json =
  {market_key: {"kind": "platt", "a": .., "b": .., "n": ..}}
  {market_key: {"kind": "isotonic", "kx": [..], "ky": [..], "n": ..}}
Absent file / market = IDENTITY (trust the book P as-is; the measured markets so
far are calibrated, so most markets have no entry — that's the correct default).
"""
import json
import os


def _path(sport_key):
    return "calibration/book_calibration_%s.json" % sport_key


def load_maps(sport_key):
    """{market: map} for a sport, or {} when none is committed."""
    try:
        with open(_path(sport_key)) as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def save_map(sport_key, market, entry):
    """Merge one market's winning map into the committed per-sport file (preserving
    other markets). entry = {'kind': 'platt'|'isotonic', ...}. Returns the path."""
    path = _path(sport_key)
    data = load_maps(sport_key)
    data[market] = entry
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def apply(sport_key, market, p, maps=None):
    """Calibrated book probability for (sport, market). IDENTITY when no map exists
    (or on any error / bad input) — so an un-measured or already-calibrated market is
    served unchanged. ``maps`` lets a caller load once and reuse across many legs."""
    if p is None:
        return p
    m = (maps if maps is not None else load_maps(sport_key)).get(market)
    if not m:
        return p
    try:
        import recalibration as rc
        if m.get("kind") == "platt" and m.get("a") is not None:
            adj = rc.apply_platt(p, m["a"], m["b"])
        elif m.get("kind") == "isotonic" and m.get("kx") and m.get("ky"):
            adj = rc.apply_isotonic(p, m["kx"], m["ky"])
        else:
            return p
    except Exception:
        return p
    if adj is None:
        return p
    return max(0.0, min(1.0, adj))
