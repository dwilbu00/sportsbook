"""coherence_offset_store.py — durable, self-maintaining coherence run-line offset.

The offset is the mean of (implied_home_cover − DK run-line fair) over DK team
triads — a stable calibration constant. Computing it from scratch reads 3 FULL
seasons of triads (slow on 20-DTU Azure) and stalled the app's first team-market
slate of the day at "Loading team data…".

A mean is EXACTLY incrementally updatable:
    new_mean = (Σ_old + Σ_new) / (n_old + n_new)
so we store its SUFFICIENT STATISTICS (sum, n, watermark) in the same durable
app_settings KV the Kelly knobs / active bonuses use (AZURE SQL = system of record;
local app_settings.jsonl fallback for dev). The live app then maintains the offset
by reading ONLY the triads AFTER the stored watermark, folding them in, and writing
the advanced watermark back — so the daily read stays tiny forever. The first-ever
call BOOTSTRAPS once from a full-season read; every later call reads ~one day.

One KV row per sport (setting_key="coherence_offset_<sport>"). The stats are
re-derivable from the triad corpus, so a lost row just triggers a one-time rebuild.
"""
import datetime
import json

import coherence_flags

_SETTINGS_FILE = "app_settings.jsonl"


def _key(sport):
    return "coherence_offset_%s" % sport


def _cutoff(today=None):
    """The latest FULLY-PAST game date to include = yesterday (today's closing lines
    may still be moving pre-commence). ``today`` is 'YYYY-MM-DD' or None (= today)."""
    d = datetime.date.fromisoformat(today) if today else datetime.date.today()
    return (d - datetime.timedelta(days=1)).isoformat()


def load_stats(sport):
    """Baseline {sum, n, through_date, dispersion} for ``sport`` from the durable KV,
    or None when nothing is stored yet."""
    try:
        import recalibration
        rows, _ = recalibration._read_ndjson_blob(_SETTINGS_FILE, use_cache=False)
        for r in (rows or []):
            if r.get("setting_key") == _key(sport):
                return json.loads(r.get("setting_value") or "null")
    except Exception:
        pass
    return None


def save_stats(sport, stats):
    """Upsert the sport's baseline sufficient-stats as one JSON KV row. Returns 1 on
    write, 0 when unchanged / on failure."""
    payload = json.dumps({
        "sum": round(float(stats["sum"]), 8),
        "n": int(stats["n"]),
        "through_date": stats.get("through_date"),
        "dispersion": float(stats.get("dispersion") or 0.0),
    })
    k = _key(sport)

    def upsert(rows):
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for r in rows:
            if r.get("setting_key") == k:
                if r.get("setting_value") == payload:
                    return 0
                r["setting_value"] = payload
                r["updated_at"] = ts
                return 1
        rows.append({"setting_key": k, "setting_value": payload, "updated_at": ts})
        return 1

    try:
        import recalibration
        return recalibration.mutate_ndjson_log(_SETTINGS_FILE, upsert) or 0
    except Exception:
        return 0


def seed_stats(sport, seasons=None, dispersion=0.0):
    """Full-read (re)baseline: compute the offset sufficient-stats over full seasons
    and persist. Returns (offset, n, through_date). Used by the coherence_flags
    --freeze-offset CLI and the app's one-time auto-bootstrap."""
    seasons = seasons or coherence_flags.DEFAULT_OFFSET_SEASONS
    s, n, through = coherence_flags.compute_offset_stats(sport, seasons, dispersion)
    save_stats(sport, {"sum": s, "n": n, "through_date": through,
                       "dispersion": dispersion})
    return ((s / n) if n else 0.0), n, through


def current_offset(sport, today=None, seasons=None, dispersion=0.0):
    """The coherence offset for TODAY — durable + self-maintaining.

    Loads the stored baseline; if absent, BOOTSTRAPS once from a full-season read.
    Otherwise reads ONLY the triads after the stored watermark (through yesterday,
    the last fully-past game date), folds them into the running mean, and writes the
    advanced watermark back so the next read is again tiny. Returns (offset, n), or
    None on any failure (→ caller falls back to the live full compute). n == 0 → the
    caller skips coherence rather than betting a raw 0.0 offset.

    ⚠ The watermark advances strictly PAST its last date, so a triad captured/
    completed late for an already-passed date is skipped until a full re-seed
    (--freeze-offset). Over thousands of triads that omission is negligible."""
    seasons = seasons or coherence_flags.DEFAULT_OFFSET_SEASONS
    try:
        stats = load_stats(sport)
        if stats is None:                       # first ever → one-time full bootstrap
            off, n, _ = seed_stats(sport, seasons, dispersion)
            return off, n
        s = float(stats.get("sum") or 0.0)
        n = int(stats.get("n") or 0)
        through = stats.get("through_date")
        disp = float(stats.get("dispersion") or dispersion)
        cutoff = _cutoff(today)
        if through and through < cutoff:
            import r2_data
            df = (datetime.date.fromisoformat(through)
                  + datetime.timedelta(days=1)).isoformat()
            triads, _ = r2_data.load_team_triad_range(sport, df, cutoff)
            ds, dn = coherence_flags._triad_offset_stats(triads, disp)
            s += ds
            n += dn
            # Advance the watermark to the cutoff even when dn == 0 (all past game
            # dates in the window are now accounted for) so we never re-scan it.
            save_stats(sport, {"sum": s, "n": n, "through_date": cutoff,
                               "dispersion": disp})
        return ((s / n) if n else 0.0), n
    except Exception:
        return None
