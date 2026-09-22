"""F23 SQL Server concurrency smoke-test (audit T23).

Exercises db_store.mutate()'s optimistic compare-and-set (F23) on the REAL SQL Server
backend under concurrent writers — the check the audit said cannot be inferred from
SQLite (StaticPool can't reproduce SQL Server isolation/locking).

The workload: N threads each increment a single counter M times via db_store.mutate
(read value -> +1 -> write). With CAS ON, every increment is preserved (final == N*M);
with CAS OFF (ODI_DISABLE_MUTATE_CAS=1, identity-only UPDATEs), concurrent read-modify-
write races silently drop increments (final < N*M). Running both back-to-back makes the
F23 protection visible as a number.

SAFE: writes ONLY a namespaced scratch key in app_settings ("__f23_smoke_test__"), a
throwaway KV row — NEVER wagers / bankroll / prediction_log / parlays — and deletes it
afterward. Respects the never-destroy-irreplaceable-data rule.

RUN (on a machine whose .streamlit/secrets.toml has the Azure SQL_* keys), ideally when
the app is idle so the test's row locks don't contend with live reads:

    PYTHONIOENCODING=utf-8 python f23_sqlserver_smoke.py                    # default 6x25
    PYTHONIOENCODING=utf-8 python f23_sqlserver_smoke.py --threads 10 --increments 50
    PYTHONIOENCODING=utf-8 python f23_sqlserver_smoke.py --cas on           # ON only

WHAT TO LOOK FOR (paste the output back):
    * CAS ON  -> "MATCH"  (final == expected), failed 0, exceptions recovered.
    * CAS OFF -> "LOST UPDATES" (final <  expected)  <- proves CAS is what protects.
    * Any 'failed' worker or an unrecovered exception with CAS ON = a real problem the
      staging test is meant to catch (kill switch ODI_DISABLE_MUTATE_CAS=1 is the escape
      hatch if SQL Server misbehaves under load).

SCOPE: single-row write-write contention exercises the CAS conflict/retry path and SQL
Server row locking/blocking under READ COMMITTED. It does NOT force multi-row deadlock
cycles (unlikely with one row under READ COMMITTED); mutate() catches OperationalError
(how SQL Server surfaces deadlock-victim 1205) with backoff — ask if you want a dedicated
multi-row deadlock stressor.
"""
import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone

KEY = "__f23_smoke_test__"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _reset_counter(db):
    """Delete + re-create the scratch counter at 0 (fresh start per trial)."""
    import sqlalchemy as sa
    t = db.app_settings
    with db.get_engine().begin() as conn:
        conn.execute(sa.delete(t).where(t.c.setting_key == KEY))
        conn.execute(sa.insert(t).values(
            setting_key=KEY, setting_value="0", updated_at=_now()))


def _read_counter(db):
    rows = db.read_rows("app_settings", where={"setting_key": KEY})
    return int(rows[0]["setting_value"]) if rows else None


def _bump_once(db):
    """One read-modify-write of the scratch counter through the REAL mutate() path.
    where= scopes the read to just the scratch row (light on the 20-DTU DB)."""
    def mut(rows):
        for r in rows:
            if r.get("setting_key") == KEY:
                r["setting_value"] = str(int(r["setting_value"]) + 1)
                r["updated_at"] = _now()
                return 1
        return 0                                   # reset guarantees the row exists
    return db.mutate("app_settings", mut, where={"setting_key": KEY})


def _worker(db, n_increments, stats):
    conflicts = 0
    for _ in range(n_increments):
        # App-level retry: under heavy contention mutate() can exhaust its 3 internal
        # CAS retries and raise _LostUpdate; a real caller would retry. Each extra
        # attempt is one observed conflict (the contention signal).
        for attempt in range(200):
            try:
                _bump_once(db)
                break
            except Exception as exc:               # _LostUpdate or a DB error
                conflicts += 1
                stats["last_exc"] = repr(exc)
                if attempt == 199:
                    stats["failed"] += 1
                    stats["conflicts"] += conflicts
                    return
                time.sleep(0.02 * (attempt % 5 + 1))
    with stats["lock"]:
        stats["conflicts"] += conflicts


def _trial(db, threads, increments, cas_on):
    if cas_on:
        os.environ.pop("ODI_DISABLE_MUTATE_CAS", None)
    else:
        os.environ["ODI_DISABLE_MUTATE_CAS"] = "1"
    _reset_counter(db)
    stats = {"conflicts": 0, "failed": 0, "last_exc": None, "lock": threading.Lock()}
    ts = [threading.Thread(target=_worker, args=(db, increments, stats))
          for _ in range(threads)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    secs = time.time() - t0
    final = _read_counter(db)
    expected = threads * increments
    return {"cas": "ON" if cas_on else "OFF", "expected": expected, "final": final,
            "match": final == expected, "conflicts": stats["conflicts"],
            "failed": stats["failed"], "secs": round(secs, 1),
            "last_exc": stats["last_exc"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--increments", type=int, default=25)
    ap.add_argument("--cas", choices=["on", "off", "both"], default="both",
                    help="which trial(s) to run (default: both, for the contrast)")
    a = ap.parse_args()

    import db_store as db
    db.promote_secrets_from_toml()
    if not db.enabled():
        print("ABORT: SQL backend not configured — need the Azure SQL_* keys in "
              ".streamlit/secrets.toml.")
        sys.exit(2)
    dialect = db.get_engine().dialect.name
    print(f"backend dialect: {dialect}")
    if dialect == "sqlite":
        print("ABORT: engine is SQLite — F23 must be validated against SQL Server "
              "(the audit says CAS behavior can't be inferred from SQLite).")
        sys.exit(2)
    print(f"workload: {a.threads} threads x {a.increments} increments = "
          f"{a.threads * a.increments} writes contending on ONE row\n")

    want = ([True] if a.cas in ("on", "both") else []) + \
           ([False] if a.cas in ("off", "both") else [])
    results = []
    try:
        for cas_on in want:
            r = _trial(db, a.threads, a.increments, cas_on)
            results.append(r)
            verdict = "MATCH" if r["match"] else "LOST UPDATES"
            print(f"CAS {r['cas']:3} | expected {r['expected']:5} | final {r['final']!s:>5} "
                  f"| {verdict:12} | conflicts {r['conflicts']:5} | failed {r['failed']} "
                  f"| {r['secs']}s")
            if r["last_exc"]:
                print(f"           last exception seen: {r['last_exc']}")
    finally:
        os.environ.pop("ODI_DISABLE_MUTATE_CAS", None)
        try:
            import sqlalchemy as sa
            t = db.app_settings
            with db.get_engine().begin() as conn:
                conn.execute(sa.delete(t).where(t.c.setting_key == KEY))
            print(f"\ncleaned up scratch key {KEY!r}")
        except Exception as exc:
            print(f"\nWARN: could not clean up scratch key {KEY!r}: {exc!r} "
                  "(harmless — delete it manually if it lingers)")

    on = next((r for r in results if r["cas"] == "ON"), None)
    off = next((r for r in results if r["cas"] == "OFF"), None)
    print("\n--- verdict ---")
    if on:
        ok = on["match"] and on["failed"] == 0
        print(f"CAS ON correctness: {'PASS' if ok else 'FAIL'} — "
              + ("no lost updates, every conflict recovered."
                 if ok else f"final {on['final']} != expected {on['expected']} "
                            f"or {on['failed']} worker(s) failed. INVESTIGATE."))
    if on and off:
        demonstrated = off["final"] is not None and off["final"] < off["expected"] <= on["final"]
        print(f"F23 protection demonstrated: {'YES' if demonstrated else 'INCONCLUSIVE'} — "
              f"OFF dropped {off['expected'] - (off['final'] or 0)} update(s), ON dropped 0."
              + ("" if demonstrated else " (try more threads/increments for contention.)"))


if __name__ == "__main__":
    main()
