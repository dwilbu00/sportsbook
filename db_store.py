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
import time

from sqlalchemy import (
    Boolean, CheckConstraint, Column, Float, Index, Integer, MetaData,
    PrimaryKeyConstraint, String, Table, UniqueConstraint, and_, create_engine,
    delete, func, insert, select, update,
)
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError, OperationalError
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
    # 0 when logged; set 1 after an offline calibration refit consumes the row.
    # Lets the app count new resolved-but-not-yet-refit records (the "time to
    # refit" banner) now that SQL rows are stable (surgical writes, not rewrites).
    Column("refit_performed", Boolean, nullable=False, default=False),
    UniqueConstraint("sport_key", "event_key", "prop_key", "player", "line",
                     name="uq_prediction_identity"),
    Index("ix_prediction_sport_resolved", "sport_key", "resolved"),
    Index("ix_prediction_refit_pending", "resolved", "refit_performed"),
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

# Team-market forward tracking — the MODEL's pick per (game, market). Sibling of
# prediction_log; one row per (sport, event, bet_type) natural identity.
market_prediction_log = Table(
    "market_prediction_log", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", String(40)),
    Column("sport_key", String(64), nullable=False),
    Column("event_id", String(128)),
    Column("event_key", String(160), nullable=False),   # event_id or game_date
    Column("commence_time", String(40)),
    Column("game_date", String(10)),
    Column("bet_type", String(16), nullable=False),      # moneyline|spread|total
    Column("home_team", String(128)),
    Column("away_team", String(128)),
    Column("team", String(128)),
    Column("opponent", String(128)),
    Column("home_away", String(8)),
    Column("side", String(16), nullable=False),          # home|away|over|under
    Column("matchup", String(256)),
    Column("book", String(64)),
    Column("actual", String(64)),                        # "home-away" score string
    Column("resolved_at", String(40)),
    Column("point", Float),                              # spread/total line; NULL for ML
    Column("model_prob", Float),                         # picked side prob (0-1)
    Column("raw_prob", Float),                           # pre-blend prob (0-1)
    Column("price", Integer),
    Column("outcome", Integer),                          # 1=won, 0=lost, NULL=push
    Column("is_value", Boolean),                         # tri-state
    Column("resolved", Boolean, nullable=False, default=False),
    UniqueConstraint("sport_key", "event_key", "bet_type",
                     name="uq_market_prediction_identity"),
    Index("ix_market_prediction_sport_resolved", "sport_key", "resolved"),
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

# Odds warehouse (Phase B) — normalized, replaces the Blob snapshot blobs +
# _manifest.json. One row per captured snapshot (write-once) ...
odds_snapshot = Table(
    "odds_snapshot", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sport", String(64), nullable=False),
    Column("game_date", String(10), nullable=False),
    Column("event_id", String(128), nullable=False),
    Column("kind", String(16), nullable=False),          # team|props|alt|seed
    Column("snapshot_hour", String(16), nullable=False),  # YYYYMMDDTHHZ bucket
    Column("captured_at", String(40)),
    Column("commence_time", String(40)),
    Column("home", String(128)),
    Column("away", String(128)),
    Column("regions", String(64)),
    Column("markets", String(256)),
    Column("bookmakers", String(256)),
    UniqueConstraint("sport", "game_date", "event_id", "kind", "snapshot_hour",
                     name="uq_odds_snapshot"),   # write-once per hour bucket
    Index("ix_odds_snapshot_event", "sport", "game_date", "event_id"),
)

# ... and one row per extracted line within a snapshot. The stored price/implied
# reproduce closing_line_for's extraction (best-across-books for team markets;
# de-vigged consensus for props), computed at capture — so CLV lookups are a
# plain query, not a re-parse. (Per-descriptor grain, not per-book raw.)
odds_line = Table(
    "odds_line", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("snapshot_id", Integer, nullable=False),
    Column("bet_type", String(16), nullable=False),   # moneyline|spread|total|player_prop
    Column("selection", String(160)),                 # team | Over | Under | player
    Column("point", Float),
    Column("player", String(160)),
    Column("prop_key", String(64)),
    Column("direction", String(8)),
    Column("price", Integer),
    Column("implied_prob", Float),
    Index("ix_odds_line_snapshot", "snapshot_id"),
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
    ("is_value", _b), ("resolved", _bexact), ("refit_performed", _bexact),
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

# Team-market forward-tracking columns (event_key is derived, like prediction).
_MARKET_PREDICTION_SPEC = [
    ("ts", _s), ("sport_key", _s), ("event_id", _s), ("commence_time", _s),
    ("game_date", _s), ("bet_type", _s), ("home_team", _s), ("away_team", _s),
    ("team", _s), ("opponent", _s), ("home_away", _s), ("side", _s),
    ("matchup", _s), ("book", _s), ("actual", _s), ("resolved_at", _s),
    ("point", _f), ("model_prob", _f), ("raw_prob", _f),
    ("price", _i), ("outcome", _i),
    ("is_value", _b), ("resolved", _bexact),
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


def _prediction_identity(row):
    # Mirrors uq_prediction_identity — the natural key used to diff rows for
    # surgical writes (event_key coalesced exactly as _prediction_derive stores it).
    return {
        "sport_key": row.get("sport_key"),
        "event_key": _prediction_derive(row)["event_key"],
        "prop_key": row.get("prop_key"),
        "player": row.get("player"),
        "line": _f(row.get("line")),
    }


def _market_prediction_derive(row):
    return {"event_key": (row.get("event_id") or row.get("game_date") or "")}


def _market_prediction_identity(row):
    # Mirrors uq_market_prediction_identity — one row per (sport, event, market).
    return {
        "sport_key": row.get("sport_key"),
        "event_key": _market_prediction_derive(row)["event_key"],
        "bet_type": row.get("bet_type"),
    }


# ``identity`` returns the natural-key {col: value} map used to diff before/after
# rows (surgical writes) and to build the UPDATE/DELETE WHERE — it matches each
# table's UNIQUE constraint.
_NDJSON_TABLES = {
    "prediction_log": {"table": prediction_log, "spec": _PREDICTION_SPEC,
                       "derive": _prediction_derive,
                       "identity": _prediction_identity},
    "market_prediction_log": {"table": market_prediction_log,
                              "spec": _MARKET_PREDICTION_SPEC,
                              "derive": _market_prediction_derive,
                              "identity": _market_prediction_identity},
    "wagers": {"table": wagers, "spec": _WAGER_SPEC, "derive": lambda row: {},
               "identity": lambda row: {"wager_id": row.get("wager_id")}},
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
            # login_timeout/timeout give the driver time to ride out an Azure SQL
            # serverless resume from auto-pause (the first connect after idle can
            # take tens of seconds) rather than failing instantly; pool_pre_ping
            # recycles connections the resume dropped.
            _ENGINE = create_engine(
                url, pool_pre_ping=True, pool_recycle=1500,
                connect_args={"login_timeout": 60, "timeout": 60})
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


def _where_clause(table, where):
    """ANDed equality/IN clause from a {col: value|list} map, or None if empty."""
    if not where:
        return None
    conds = []
    for col, val in where.items():
        column = table.c[col]
        conds.append(column.in_(list(val))
                     if isinstance(val, (list, tuple, set))
                     else column == val)
    return and_(*conds)


def _identity_where(cfg, row):
    """WHERE matching one row on its natural identity (for surgical UPDATE/DELETE)."""
    return and_(*(cfg["table"].c[col] == val
                  for col, val in cfg["identity"](row).items()))


def _key(cfg, row):
    """Hashable natural-identity key for diffing before/after rows."""
    return tuple(sorted(cfg["identity"](row).items()))


def _select_rows(conn, cfg, where=None):
    """Ordered row-dict list (only the store's declared fields), optionally
    filtered by an equality/IN ``where`` map."""
    names = [name for name, _ in cfg["spec"]]
    stmt = select(cfg["table"]).order_by(cfg["table"].c.id)
    clause = _where_clause(cfg["table"], where)
    if clause is not None:
        stmt = stmt.where(clause)
    result = conn.execute(stmt)
    return [{name: row._mapping[name] for name in names} for row in result]


def read_rows(table_name, where=None, max_retries=3):
    """Return the row-dict list for an NDJSON store, optionally filtered by an
    equality/IN ``where`` map (e.g. {"status": "pending"}) so a reconciliation
    caller pulls only the rows it needs instead of the whole table.

    Retries transient OperationalErrors with a short backoff — chiefly an Azure
    SQL serverless database resuming from auto-pause, whose first read after idle
    can time out. Without this the caller (e.g. the wagers ledger) would surface
    an empty store on the first page load and the user would have to click
    Refresh once the DB woke up."""
    cfg = _resolve(table_name)
    engine = get_engine()
    last_exc = None
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                return _select_rows(conn, cfg, where)
        except OperationalError as exc:  # transient (cold resume / lock / timeout)
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1 + 2 * attempt)
    raise last_exc


def count_rows(table_name, where=None, max_retries=3):
    """COUNT(*) for an NDJSON store's SQL table, optionally filtered by an
    equality/IN ``where`` map — cheap (no row egress), for the app's "time to
    refit" banner. Retries transient OperationalErrors like read_rows."""
    cfg = _resolve(table_name)
    engine = get_engine()
    last_exc = None
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                stmt = select(func.count()).select_from(cfg["table"])
                clause = _where_clause(cfg["table"], where)
                if clause is not None:
                    stmt = stmt.where(clause)
                return int(conn.execute(stmt).scalar() or 0)
        except OperationalError as exc:  # transient (cold resume / lock / timeout)
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1 + 2 * attempt)
    raise last_exc


def mutate(table_name, mutator, where=None, max_retries=3):
    """Transactionally read → mutate → write only the DELTA for an NDJSON store.

    The mutator receives the current row-dict list and mutates it in place
    (append / prune / edit), exactly as the Blob/local path expects; a falsy
    return skips the write. Rather than rewriting the whole table, we diff the
    before/after lists by each store's natural identity and emit only the changed
    rows as surgical INSERT/UPDATE/DELETE — all in one transaction, so a
    CHECK/UNIQUE violation rolls the whole thing back and propagates (matching the
    Blob path). Returns the mutator's result (its change count), not the DB-op
    count, so every caller is unchanged.

    ``where`` (an equality/IN {col: value} map) restricts the rows read into the
    mutator to a subset — used by update-only reconciliation passes (e.g. grade
    only status='pending') to avoid pulling the whole table out of the DB. A
    ``where``-filtered mutate MUST only update rows within that subset: it must not
    append a row whose identity could collide with an unread row, nor depend on
    rows outside the filter."""
    cfg = _resolve(table_name)
    table = cfg["table"]
    engine = get_engine()
    last_exc = None
    for attempt in range(max_retries):
        try:
            with engine.begin() as conn:
                before = _select_rows(conn, cfg, where)
                before_by_key = {_key(cfg, r): r for r in before}
                working = [dict(r) for r in before]   # rows are flat scalar dicts
                result = mutator(working)
                if not result:
                    return result
                after_by_key = {}
                for r in working:
                    k = _key(cfg, r)
                    if k in after_by_key:
                        # Two mutated rows share a natural identity. Delete-all +
                        # insert-all would have surfaced this via the UNIQUE
                        # constraint; the diff would otherwise silently drop one
                        # (last-writer-wins). Fail loudly to keep that backstop.
                        raise ValueError(
                            f"{table_name}: duplicate identity {k} in mutated rows")
                    after_by_key[k] = r
                inserts = []
                for k, row in after_by_key.items():
                    prior = before_by_key.get(k)
                    new_params = _row_to_params(cfg, row)
                    if prior is None:
                        inserts.append(new_params)
                    elif new_params != _row_to_params(cfg, prior):
                        conn.execute(update(table)
                                     .where(_identity_where(cfg, prior))
                                     .values(**new_params))
                if inserts:
                    conn.execute(insert(table), inserts)
                for k, row in before_by_key.items():
                    if k not in after_by_key:
                        conn.execute(delete(table)
                                     .where(_identity_where(cfg, row)))
                return result
        except OperationalError as exc:  # transient (cold resume/lock/timeout)
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1 + 2 * attempt)
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


# ──────────────────────────────────────────────────────────────────────────────
# Odds warehouse (Phase B) — snapshots + extracted lines
# ──────────────────────────────────────────────────────────────────────────────

_ODDS_LINE_COLS = ("bet_type", "selection", "point", "player", "prop_key",
                   "direction", "price", "implied_prob")


def capture_odds_snapshot(meta, lines):
    """Write-once insert of one snapshot + its extracted lines.

    Returns True if a new snapshot was written, False if it already existed
    (write-once, enforced by uq_odds_snapshot) or on a transient conflict. ``meta``
    carries the odds_snapshot columns; ``lines`` is a list of dicts over
    _ODDS_LINE_COLS."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            result = conn.execute(insert(odds_snapshot), {
                "sport": meta.get("sport"),
                "game_date": meta.get("game_date"),
                "event_id": meta.get("event_id"),
                "kind": meta.get("kind"),
                "snapshot_hour": meta.get("snapshot_hour"),
                "captured_at": _s(meta.get("captured_at")),
                "commence_time": _s(meta.get("commence_time")),
                "home": _s(meta.get("home")),
                "away": _s(meta.get("away")),
                "regions": _s(meta.get("regions")),
                "markets": _s(meta.get("markets")),
                "bookmakers": _s(meta.get("bookmakers")),
            })
            snapshot_id = result.inserted_primary_key[0]
            if lines:
                conn.execute(insert(odds_line), [{
                    "snapshot_id": snapshot_id,
                    "bet_type": _s(ln.get("bet_type")),
                    "selection": _s(ln.get("selection")),
                    "point": _f(ln.get("point")),
                    "player": _s(ln.get("player")),
                    "prop_key": _s(ln.get("prop_key")),
                    "direction": _s(ln.get("direction")),
                    "price": _i(ln.get("price")),
                    "implied_prob": _f(ln.get("implied_prob")),
                } for ln in lines])
        return True
    except IntegrityError:
        return False  # snapshot already captured this hour (write-once)


def odds_snapshots_for_event(sport, game_date, event_id):
    """(id, captured_at) for an event's snapshots — caller picks nearest-commence."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(odds_snapshot.c.id, odds_snapshot.c.captured_at,
                   odds_snapshot.c.kind)
            .where((odds_snapshot.c.sport == sport)
                   & (odds_snapshot.c.game_date == game_date)
                   & (odds_snapshot.c.event_id == event_id))
        ).all()
    return [{"id": r[0], "captured_at": r[1], "kind": r[2]} for r in rows]


def odds_line_lookup(snapshot_id, bet_type, selection=None, point=None,
                     player=None, prop_key=None, direction=None):
    """The stored line for a descriptor within one snapshot, or None.

    Reproduces _extract_line's matching: props key on (prop_key, player,
    direction); team markets on (selection[, point]) with a fall back to the best
    price across points when the exact point isn't stored."""
    table = odds_line
    bt = (bet_type or "").lower()
    with get_engine().connect() as conn:
        if bt == "player_prop":
            row = conn.execute(
                select(table.c.price, table.c.implied_prob).where(
                    (table.c.snapshot_id == snapshot_id)
                    & (table.c.bet_type == "player_prop")
                    & (table.c.prop_key == prop_key)
                    & (table.c.player == player)
                    & (table.c.direction == ((direction or "OVER").upper())))
            ).first()
            return {"price": row[0], "implied_prob": row[1]} if row else None

        norm = {"h2h": "moneyline", "moneyline": "moneyline",
                "spreads": "spread", "spread": "spread",
                "totals": "total", "total": "total"}.get(bt, bt)
        sel = ("Under" if norm == "total" and (selection or "").lower() == "under"
               else "Over" if norm == "total" else selection)
        base = ((table.c.snapshot_id == snapshot_id)
                & (table.c.bet_type == norm) & (table.c.selection == sel))
        if point is not None:
            row = conn.execute(
                select(table.c.price, table.c.implied_prob).where(
                    base & (func.abs(table.c.point - float(point)) < 1e-9))
            ).first()
            if row:
                return {"price": row[0], "implied_prob": row[1]}
        # No exact point (or point-less market) → best price for the selection.
        row = conn.execute(
            select(table.c.price, table.c.implied_prob)
            .where(base).order_by(table.c.price.desc())
        ).first()
        return {"price": row[0], "implied_prob": row[1]} if row else None


def list_odds_snapshots(sport, game_date):
    """Snapshot metadata rows for a (sport, date) — for reporting/enumeration."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(odds_snapshot)
            .where((odds_snapshot.c.sport == sport)
                   & (odds_snapshot.c.game_date == game_date))
            .order_by(odds_snapshot.c.captured_at)
        ).all()
    return [{
        "event_id": r._mapping["event_id"], "kind": r._mapping["kind"],
        "commence_time": r._mapping["commence_time"],
        "home": r._mapping["home"], "away": r._mapping["away"],
        "captured_at": r._mapping["captured_at"],
        "markets": r._mapping["markets"],
    } for r in rows]


def team_market_lines(sport, dates=None, date_from=None, date_to=None,
                      max_retries=3):
    """Bulk-read warehoused team-market lines (moneyline/spread/total) for a
    sport, joining each odds_line to its parent odds_snapshot.

    Feeds the team-market backtest's closing-line store
    (warehouse.load_team_market_store) from the growing Azure warehouse instead
    of the local historical_odds JSON. Player props are excluded. Optional date
    filter: ``dates`` (explicit game_date list) OR ``date_from``/``date_to``
    (inclusive range). Ordered by (event_id, captured_at) so the assembler can
    pick each event's closing snapshot. Retries transient OperationalErrors like
    read_rows (Azure SQL serverless cold-resume safety). Returns row dicts."""
    engine = get_engine()
    joined = odds_line.join(odds_snapshot,
                            odds_line.c.snapshot_id == odds_snapshot.c.id)
    stmt = (
        select(
            odds_snapshot.c.event_id, odds_snapshot.c.game_date,
            odds_snapshot.c.commence_time, odds_snapshot.c.home,
            odds_snapshot.c.away, odds_snapshot.c.captured_at,
            odds_snapshot.c.kind, odds_line.c.snapshot_id,
            odds_line.c.bet_type, odds_line.c.selection, odds_line.c.point,
            odds_line.c.price, odds_line.c.implied_prob,
        )
        .select_from(joined)
        .where((odds_snapshot.c.sport == sport)
               & odds_line.c.bet_type.in_(("moneyline", "spread", "total")))
    )
    if dates:
        stmt = stmt.where(odds_snapshot.c.game_date.in_(list(dates)))
    else:
        if date_from is not None:
            stmt = stmt.where(odds_snapshot.c.game_date >= date_from)
        if date_to is not None:
            stmt = stmt.where(odds_snapshot.c.game_date <= date_to)
    stmt = stmt.order_by(odds_snapshot.c.event_id, odds_snapshot.c.captured_at)

    last_exc = None
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                rows = conn.execute(stmt).all()
            return [{
                "event_id": r._mapping["event_id"],
                "game_date": r._mapping["game_date"],
                "commence_time": r._mapping["commence_time"],
                "home": r._mapping["home"], "away": r._mapping["away"],
                "captured_at": r._mapping["captured_at"],
                "kind": r._mapping["kind"],
                "snapshot_id": r._mapping["snapshot_id"],
                "bet_type": r._mapping["bet_type"],
                "selection": r._mapping["selection"],
                "point": r._mapping["point"],
                "price": r._mapping["price"],
                "implied_prob": r._mapping["implied_prob"],
            } for r in rows]
        except OperationalError as exc:  # transient (cold resume / lock / timeout)
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1 + 2 * attempt)
    raise last_exc
