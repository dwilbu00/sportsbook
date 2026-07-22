"""SQL backend for the app's durable state (Azure SQL in prod; SQLite in tests).

When the ``SQL_*`` secrets are configured, the Blob/NDJSON stores in
``recalibration.py`` (prediction log, wagers ledger, recalibration params)
dispatch to the intent-level ops here instead of the hand-rolled SAS + ETag
read-modify-write path. Selection is a feature flag (``enabled()``): no SQL
secret → the app keeps using Blob/local exactly as before.

Design notes
------------
* **The pricing core never imports this.** Only ``recalibration.py`` (and, in
  Phase B, ``warehouse.py``) do — behind a guarded import so a missing
  SQLAlchemy install simply leaves SQL disabled.
* **Fully columnar (true relational) schema.** Every field of every store is its
  own typed column, so the data is directly queryable in SQL (ROI by prop,
  pending bets, CLV, etc.) — no JSON-text catch-all. The one nested structure,
  a recalibration prop's ``validation_folds`` list, is a child table.
  Consequence: an entirely new field would need a schema migration (a column
  add), which is normal relational practice.
* **Transactional replace, not ETag loops.** ``mutate()`` runs the caller's
  row-list mutator inside one transaction and, if it reports changes, replaces
  the store's rows atomically. CHECK/UNIQUE constraints reject bad data; the
  transaction gives ACID consistency in place of the optimistic-concurrency
  retry dance. These stores are small (personal bet + forecast history).
* **No DDL from the app.** Production tables are created MANUALLY by the user
  from ``sql/schema.sql`` (the app connects as a least-privilege CRUD-only user).
  ``create_all()`` here exists ONLY for the in-memory SQLite test harness and is
  kept in lockstep with ``sql/schema.sql`` (columns, types, constraints).

pymssql (the TDS driver — the Python analog of Azure's Go ``go-mssqldb`` sample)
is imported lazily by SQLAlchemy only when connecting to ``mssql+pymssql://``;
tests connect to ``sqlite://`` and never touch it.
"""

import os
import threading

from sqlalchemy import (
    Boolean, CheckConstraint, Column, Float, Index, Integer, MetaData,
    PrimaryKeyConstraint, String, Table, UniqueConstraint, create_engine,
    delete, insert, select,
)
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_SECRET_KEYS = ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD")
_SQL_PORT = 1433

_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_OVERRIDE_URL = None  # set by configure_engine() for tests


# ──────────────────────────────────────────────────────────────────────────────
# Value coercion (row dict value → typed DB param). SQLAlchemy's typed columns
# convert back to the right Python type on read, so reads need no coercion.
# ``actual`` in the wagers ledger is intentionally mixed (a float for props, a
# "home-away" score string for team markets) → stored as text.
# ──────────────────────────────────────────────────────────────────────────────

def _s(v):
    return None if v is None else str(v)


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _b(v):
    return None if v is None else bool(v)  # tri-state (None preserved)


def _bexact(v):
    return bool(v)  # NOT-NULL boolean: absent/None → False


# ──────────────────────────────────────────────────────────────────────────────
# Schema (SQLAlchemy Core metadata; mirrors sql/schema.sql exactly)
# ──────────────────────────────────────────────────────────────────────────────
_META = MetaData()

_WAGER_STATUSES = ("pending", "won", "lost", "push", "void")

# Prediction log — one row per forecast identity.
prediction_log = Table(
    "prediction_log", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", String(40)),
    Column("sport_key", String(64), nullable=False),
    Column("event_id", String(128)),
    Column("event_key", String(160), nullable=False),   # event_id or game_date
    Column("commence_time", String(40)),
    Column("prop_key", String(64), nullable=False),
    Column("player", String(160), nullable=False),
    Column("game_date", String(10)),
    Column("direction", String(8)),
    Column("book", String(64)),
    Column("resolved_at", String(40)),
    Column("line", Float, nullable=False),
    Column("raw_prob", Float),
    Column("final_prob", Float),
    Column("projected", Float),
    Column("actual", Float),
    Column("price", Integer),
    Column("outcome", Integer),                          # 1=over, 0=under, NULL
    Column("is_value", Boolean),                         # tri-state
    Column("resolved", Boolean, nullable=False, default=False),
    UniqueConstraint("sport_key", "event_key", "prop_key", "player", "line",
                     name="uq_prediction_identity"),
    Index("ix_prediction_sport_resolved", "sport_key", "resolved"),
)

# Actual bets ledger — one row per wager_id.
wagers = Table(
    "wagers", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("wager_id", String(64), nullable=False),
    Column("placed_at", String(40)),
    Column("sport_key", String(64)),
    Column("bet_type", String(32)),
    Column("event_id", String(128)),
    Column("commence_time", String(40)),
    Column("game_date", String(10)),
    Column("home_team", String(128)),
    Column("away_team", String(128)),
    Column("matchup", String(256)),
    Column("team", String(128)),
    Column("opponent", String(128)),
    Column("home_away", String(8)),
    Column("player", String(160)),
    Column("prop_key", String(64)),
    Column("prop_label", String(64)),
    Column("direction", String(8)),
    Column("side", String(16)),
    Column("book", String(64)),
    Column("status", String(16)),
    Column("actual", String(64)),                        # mixed float/str → text
    Column("resolved_at", String(40)),
    Column("point", Float),
    Column("line", Float),
    Column("stake", Float),
    Column("model_prob", Float),
    Column("model_edge", Float),
    Column("close_line", Float),
    Column("clv_pct", Float),
    Column("profit", Float),
    Column("executed_price", Integer),
    Column("model_price", Integer),
    Column("close_price", Integer),
    UniqueConstraint("wager_id", name="uq_wager_id"),
    CheckConstraint(
        "status IN ('pending','won','lost','push','void')",
        name="ck_wager_status"),
    CheckConstraint("stake IS NULL OR stake >= 0", name="ck_wager_stake"),
    Index("ix_wager_status", "status"),
)

# Learned Platt recalibration — scalar fit fields per (sport, prop) ...
recalibration_params = Table(
    "recalibration_params", _META,
    Column("sport_key", String(64), nullable=False),
    Column("prop_key", String(64), nullable=False),
    Column("a", Float),
    Column("b", Float),
    Column("n_fit", Integer),
    Column("n_validation", Integer),
    Column("n_validation_folds", Integer),
    Column("holdout_start", String(10)),
    Column("holdout_raw_brier", Float),
    Column("holdout_calibrated_brier", Float),
    Column("holdout_raw_log_loss", Float),
    Column("holdout_calibrated_log_loss", Float),
    Column("holdout_metric_scope", String(64)),
    Column("deploy_fit_scope", String(64)),
    Column("validated", Boolean),
    Column("source", String(64)),
    PrimaryKeyConstraint("sport_key", "prop_key", name="pk_recalibration_params"),
)

# ... and the nested per-fold cross-validation metrics (child of params).
recalibration_folds = Table(
    "recalibration_folds", _META,
    Column("sport_key", String(64), nullable=False),
    Column("prop_key", String(64), nullable=False),
    Column("fold_index", Integer, nullable=False),
    Column("holdout_start", String(10)),
    Column("n_validation", Integer),
    Column("raw_brier", Float),
    Column("calibrated_brier", Float),
    Column("raw_log_loss", Float),
    Column("calibrated_log_loss", Float),
    PrimaryKeyConstraint("sport_key", "prop_key", "fold_index",
                         name="pk_recalibration_folds"),
)

# Top-level per-sport recalibration metadata.
recalibration_meta = Table(
    "recalibration_meta", _META,
    Column("sport_key", String(64), primary_key=True),
    Column("fit_timestamp", String(40)),
    Column("source", String(64)),
)


# Column specs (name, write-coercer) for the NDJSON-style stores. Reads use the
# column names verbatim (SQLAlchemy typed columns already return the right type).
_PREDICTION_SPEC = [
    ("ts", _s), ("sport_key", _s), ("event_id", _s), ("commence_time", _s),
    ("prop_key", _s), ("player", _s), ("game_date", _s), ("direction", _s),
    ("book", _s), ("resolved_at", _s),
    ("line", _f), ("raw_prob", _f), ("final_prob", _f), ("projected", _f),
    ("actual", _f),
    ("price", _i), ("outcome", _i),
    ("is_value", _b), ("resolved", _bexact),
]

_WAGER_SPEC = [
    ("wager_id", _s), ("placed_at", _s), ("sport_key", _s), ("bet_type", _s),
    ("event_id", _s), ("commence_time", _s), ("game_date", _s),
    ("home_team", _s), ("away_team", _s), ("matchup", _s), ("team", _s),
    ("opponent", _s), ("home_away", _s), ("player", _s), ("prop_key", _s),
    ("prop_label", _s), ("direction", _s), ("side", _s), ("book", _s),
    ("status", _s), ("actual", _s), ("resolved_at", _s),
    ("point", _f), ("line", _f), ("stake", _f), ("model_prob", _f),
    ("model_edge", _f), ("close_line", _f), ("clv_pct", _f), ("profit", _f),
    ("executed_price", _i), ("model_price", _i), ("close_price", _i),
]

# Scalar recalibration-param fields (name, coercer) beyond the (sport, prop) key.
_RECAL_PARAM_SPEC = [
    ("a", _f), ("b", _f), ("n_fit", _i), ("n_validation", _i),
    ("n_validation_folds", _i), ("holdout_start", _s),
    ("holdout_raw_brier", _f), ("holdout_calibrated_brier", _f),
    ("holdout_raw_log_loss", _f), ("holdout_calibrated_log_loss", _f),
    ("holdout_metric_scope", _s), ("deploy_fit_scope", _s),
    ("validated", _b), ("source", _s),
]
_RECAL_FOLD_SPEC = [
    ("holdout_start", _s), ("n_validation", _i), ("raw_brier", _f),
    ("calibrated_brier", _f), ("raw_log_loss", _f), ("calibrated_log_loss", _f),
]


def _prediction_derive(row):
    return {"event_key": (row.get("event_id") or row.get("game_date") or "")}


_NDJSON_TABLES = {
    "prediction_log": {"table": prediction_log, "spec": _PREDICTION_SPEC,
                       "derive": _prediction_derive},
    "wagers": {"table": wagers, "spec": _WAGER_SPEC, "derive": lambda row: {}},
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration / engine
# ──────────────────────────────────────────────────────────────────────────────

def _secret(name):
    """Read a SQL_* secret from the ENVIRONMENT only.

    Deliberately env-only (NOT a secrets.toml fallback): the Streamlit app
    promotes st.secrets → env at boot, and CLI tools call
    promote_secrets_from_toml() explicitly. This keeps the SQL backend OFF during
    tests (which never set these env vars) even if a developer's local
    secrets.toml happens to contain SQL_* keys — so the hermetic storage tests
    can never accidentally hit the live Azure database."""
    return os.environ.get(name, "").strip()


def promote_secrets_from_toml(path=None):
    """Copy SQL_* keys from .streamlit/secrets.toml into os.environ (setdefault).

    For CLI tools (backfill, offline refit) that run outside Streamlit. The
    Streamlit app already promotes them at boot. Best-effort; returns True when
    all four keys ended up in the environment."""
    path = path or os.path.join(SCRIPT_DIR, ".streamlit", "secrets.toml")
    try:
        import tomllib
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        for key in _SECRET_KEYS:
            value = data.get(key)
            if value:
                os.environ.setdefault(key, str(value).strip())
    except (ImportError, OSError, TypeError, ValueError):
        pass
    return _configured()


def _configured():
    return all(_secret(key) for key in _SECRET_KEYS)


def enabled():
    """True when the SQL backend should be used (test override, or all secrets)."""
    return _OVERRIDE_URL is not None or _configured()


def configure_engine(url):
    """Point the backend at an explicit SQLAlchemy URL (tests: 'sqlite://').

    Resets any cached engine so the next call rebuilds it. Pass None to clear the
    override and fall back to the SQL_* secrets."""
    global _OVERRIDE_URL, _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.dispose()
        except Exception:
            pass
    _OVERRIDE_URL = url
    _ENGINE = None


def _connection_url():
    if _OVERRIDE_URL is not None:
        return _OVERRIDE_URL
    if not _configured():
        return None
    return URL.create(
        "mssql+pymssql",
        username=_secret("SQL_USER"),
        password=_secret("SQL_PASSWORD"),
        host=_secret("SQL_SERVER"),
        port=_SQL_PORT,
        database=_secret("SQL_DATABASE"),
    )


def get_engine():
    """Process-singleton SQLAlchemy Engine. Raises if SQL is not configured."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        url = _connection_url()
        if url is None:
            raise RuntimeError(
                "SQL backend not configured (SQL_SERVER/SQL_DATABASE/SQL_USER/"
                "SQL_PASSWORD missing).")
        if str(url).startswith("sqlite"):
            # Shared in-memory DB across the pool (tests): one persistent conn.
            _ENGINE = create_engine(
                "sqlite://", connect_args={"check_same_thread": False},
                poolclass=StaticPool)
        else:
            _ENGINE = create_engine(url, pool_pre_ping=True, pool_recycle=1500)
        return _ENGINE


def create_all():
    """Create tables. TEST-ONLY (SQLite). Prod schema is created manually from
    sql/schema.sql; the app never issues DDL against Azure."""
    _META.create_all(get_engine())


def storage_backend():
    return "Azure SQL" if enabled() else ""


# ──────────────────────────────────────────────────────────────────────────────
# NDJSON-style store ops (prediction log, wagers) — one column per field
# ──────────────────────────────────────────────────────────────────────────────

def _resolve(table_name):
    cfg = _NDJSON_TABLES.get(table_name)
    if cfg is None:
        raise KeyError(f"Unknown SQL store: {table_name!r}")
    return cfg


def _row_to_params(cfg, row):
    params = {name: fn(row.get(name)) for name, fn in cfg["spec"]}
    params.update(cfg["derive"](row))
    return params


def _select_rows(conn, cfg):
    """Ordered row-dict list (only the store's declared fields)."""
    names = [name for name, _ in cfg["spec"]]
    result = conn.execute(select(cfg["table"]).order_by(cfg["table"].c.id))
    return [{name: row._mapping[name] for name in names} for row in result]


def read_rows(table_name):
    """Return the row-dict list for an NDJSON store."""
    cfg = _resolve(table_name)
    with get_engine().connect() as conn:
        return _select_rows(conn, cfg)


def _replace_rows(conn, cfg, rows):
    conn.execute(delete(cfg["table"]))
    if rows:
        conn.execute(insert(cfg["table"]),
                     [_row_to_params(cfg, row) for row in rows])


def mutate(table_name, mutator, max_retries=3):
    """Transactionally read → mutate → replace an NDJSON store.

    The mutator receives the full row-dict list and mutates it in place, exactly
    as the Blob/local path expects; a falsy return skips the write. The whole
    thing runs in one transaction, so a constraint violation rolls back and
    propagates (matching the Blob path, where user-initiated writes surface
    failures). Returns the mutator's result."""
    cfg = _resolve(table_name)
    engine = get_engine()
    last_exc = None
    for _ in range(max_retries):
        try:
            with engine.begin() as conn:
                rows = _select_rows(conn, cfg)
                result = mutator(rows)
                if result:
                    _replace_rows(conn, cfg, rows)
                return result
        except OperationalError as exc:  # transient (lock/timeout) → retry
            last_exc = exc
    raise last_exc


# ──────────────────────────────────────────────────────────────────────────────
# Recalibration params (per (sport, prop) + validation-fold child rows)
# ──────────────────────────────────────────────────────────────────────────────

def _nonnull(mapping, spec):
    """Reconstruct a cfg dict from a row mapping, omitting NULL columns so it
    matches the original (which omitted absent keys) rather than filling nulls."""
    out = {}
    for name, _ in spec:
        value = mapping[name]
        if value is not None:
            out[name] = value
    return out


def load_recal(sport_key):
    """Return the recalibration cfg dict for a sport, or None when absent.

    Shape mirrors the Blob JSON: {sport_key, fit_timestamp, props:{prop:{...,
    validation_folds:[...]}}, meta}. The caller (recalibration._parse_recal_blob)
    applies the validated filter."""
    engine = get_engine()
    with engine.connect() as conn:
        param_rows = conn.execute(
            select(recalibration_params)
            .where(recalibration_params.c.sport_key == sport_key)
        ).all()
        if not param_rows:
            return None
        fold_rows = conn.execute(
            select(recalibration_folds)
            .where(recalibration_folds.c.sport_key == sport_key)
            .order_by(recalibration_folds.c.prop_key,
                      recalibration_folds.c.fold_index)
        ).all()
        meta_row = conn.execute(
            select(recalibration_meta)
            .where(recalibration_meta.c.sport_key == sport_key)
        ).first()

    folds_by_prop = {}
    for row in fold_rows:
        folds_by_prop.setdefault(row._mapping["prop_key"], []).append(
            _nonnull(row._mapping, _RECAL_FOLD_SPEC))

    props = {}
    for row in param_rows:
        prop_key = row._mapping["prop_key"]
        cfg = _nonnull(row._mapping, _RECAL_PARAM_SPEC)
        if prop_key in folds_by_prop:
            cfg["validation_folds"] = folds_by_prop[prop_key]
        props[prop_key] = cfg

    cfg = {"sport_key": sport_key, "props": props}
    if meta_row is not None:
        if meta_row._mapping["fit_timestamp"]:
            cfg["fit_timestamp"] = meta_row._mapping["fit_timestamp"]
        if meta_row._mapping["source"]:
            cfg["meta"] = {"source": meta_row._mapping["source"]}
    return cfg


def save_recal(sport_key, cfg):
    """Persist a recalibration cfg dict (replace this sport's rows atomically)."""
    props = (cfg or {}).get("props") or {}
    param_rows = []
    fold_rows = []
    for prop_key, prop_cfg in props.items():
        if not isinstance(prop_cfg, dict):
            continue
        row = {"sport_key": sport_key, "prop_key": prop_key}
        row.update({name: fn(prop_cfg.get(name))
                    for name, fn in _RECAL_PARAM_SPEC})
        param_rows.append(row)
        for index, fold in enumerate(prop_cfg.get("validation_folds") or []):
            if not isinstance(fold, dict):
                continue
            fold_row = {"sport_key": sport_key, "prop_key": prop_key,
                        "fold_index": index}
            fold_row.update({name: fn(fold.get(name))
                             for name, fn in _RECAL_FOLD_SPEC})
            fold_rows.append(fold_row)

    meta = (cfg or {}).get("meta") or {}
    engine = get_engine()
    with engine.begin() as conn:
        for table in (recalibration_params, recalibration_folds,
                      recalibration_meta):
            conn.execute(delete(table).where(table.c.sport_key == sport_key))
        if param_rows:
            conn.execute(insert(recalibration_params), param_rows)
        if fold_rows:
            conn.execute(insert(recalibration_folds), fold_rows)
        conn.execute(insert(recalibration_meta), {
            "sport_key": sport_key,
            "fit_timestamp": (cfg or {}).get("fit_timestamp"),
            "source": meta.get("source"),
        })
