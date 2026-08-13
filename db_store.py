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
import unicodedata

from sqlalchemy import (
    Boolean, CheckConstraint, Column, Float, ForeignKey, Index, Integer,
    MetaData, PrimaryKeyConstraint, String, Table, UniqueConstraint, and_,
    bindparam, create_engine, delete, event, func, insert, select, update,
)
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.pool import StaticPool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_SECRET_KEYS = ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD")
_REQUIRE_SQL_ENV = "SPORTSBOOK_REQUIRE_SQL"  # explicit prod opt-in / dev escape hatch
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
# Identity helpers (leaf functions importable everywhere, so db_store's surgical
# diff, recalibration's identity, player_id_map, and the backfill all compute the
# same key and can never drift). ``normalize_name`` == mlb_starters._norm.
# ──────────────────────────────────────────────────────────────────────────────

def normalize_name(name):
    """Cross-source name key: NFKD-fold accents to ASCII, lowercase, keep alnum +
    spaces, strip edges. Identical to mlb_starters._norm so a normalized odds-feed
    name matches the SFBB map's stored name_norm (which folds the same way)."""
    if not name:
        return ""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(c for c in n.lower() if c.isalnum() or c.isspace()).strip()


def player_key(row):
    """Hybrid, collision-proof player identity for a durable row: the MLBAM id when
    the row carries one ("mlb:<id>"), else the normalized name ("name:<norm>").

    The prefix guarantees an id can never collide with a name. ``player`` is
    NOT NULL upstream so the key is always non-NULL — important because SQL Server
    treats every NULL in a UNIQUE column as equal, which would collapse unrelated
    rows. This keeps the id-based identity total across NBA / historical / unmapped
    rows (which simply fall back to the name key, exactly today's behavior)."""
    mid = row.get("player_mlb_id")
    if mid:
        return f"mlb:{mid}"
    return f"name:{normalize_name(row.get('player') or '')}"


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
    # Rule inputs the pick-rules ROI lens re-derives the recommended slate from.
    # Nullable (pre-feature rows are NULL → team-based rules reported skipped).
    Column("team", String(160)),
    Column("batting_order", Integer),
    # SFBB cross-map enrichment (Phase 3, all nullable/best-effort): the MLBAM id +
    # canonical team code for id-based joins, and player_key = the hybrid identity
    # ("mlb:<id>" or "name:<norm>") that becomes the UNIQUE key in Phase 4.
    Column("player_mlb_id", String(32)),
    Column("team_code", String(16)),
    Column("player_key", String(200), nullable=False),
    # P3: the StatsAPI game_pk the prop belongs to (nullable/best-effort, stamped
    # by entity_resolver). Deliberately NOT in the UNIQUE key — identity stays
    # player_key-based; prop_key + line already distinguish a player's many props
    # in one game. NULL for NBA/NFL and for unresolved MLB rows (the P3 shadow
    # signal); P4 enforces MLB-scoped non-NULL, P5 backfills legacy rows.
    Column("game_pk", Integer),
    # Which data path served the model-input history at prediction time: "warehouse"
    # (StatsAPI facts) or "espn". Nullable/best-effort — makes a warehouse-gate flip's
    # effect auditable per prediction (NULL for pre-column rows / unknown source),
    # the durable signal for "prove the gate is carrying the load" before ESPN removal.
    Column("source", String(16)),
    UniqueConstraint("sport_key", "event_key", "prop_key", "player_key", "line",
                     name="uq_prediction_identity_v2"),
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
    # SFBB cross-map enrichment (Phase 3, nullable/best-effort): id-based joins.
    Column("player_mlb_id", String(32)),
    Column("team_code", String(16)),
    Column("opponent_code", String(16)),
    Column("home_code", String(16)),
    Column("away_code", String(16)),
    Column("game_pk", Integer),                           # P3: StatsAPI game (best-effort)
    UniqueConstraint("wager_id", name="uq_wager_id"),
    CheckConstraint(
        "status IN ('pending','won','lost','push','void')",
        name="ck_wager_status"),
    CheckConstraint("stake IS NULL OR stake >= 0", name="ck_wager_stake"),
    Index("ix_wager_status", "status"),
)

# Bankroll ledger — one signed transaction per row; the current bankroll is the
# SUM of all amounts. Two kinds: 'bet' (one per settled wager, amount = its
# realized profit, txn_id = 'bet:<wager_id>') and 'adjustment' (a manual
# deposit/withdrawal/correction, amount = the signed delta the user's typed
# target implies). The balance is never stored — it is derived — so a re-graded
# wager can't leave a stale running total behind.
bankroll_ledger = Table(
    "bankroll_ledger", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("txn_id", String(80), nullable=False),
    Column("txn_type", String(16)),                      # bet | adjustment
    Column("amount", Float),                             # signed dollars
    Column("wager_id", String(64)),                     # set for 'bet' txns
    Column("note", String(256)),
    Column("created_at", String(40)),
    UniqueConstraint("txn_id", name="uq_bankroll_txn"),
    Index("ix_bankroll_txn_type", "txn_type"),
)

# Durable per-user app settings — a generic key/value store. Currently the Kelly
# sizing knobs (fraction / per-bet cap % / slate-total cap %) so they persist
# across sessions, not just page switches.
app_settings = Table(
    "app_settings", _META,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("setting_key", String(64), nullable=False),
    Column("setting_value", String(256)),
    Column("updated_at", String(40)),
    UniqueConstraint("setting_key", name="uq_app_setting_key"),
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
    # SFBB cross-map enrichment (Phase 3, nullable/best-effort): canonical team codes.
    Column("team_code", String(16)),
    Column("opponent_code", String(16)),
    Column("home_code", String(16)),
    Column("away_code", String(16)),
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
    # SFBB cross-map enrichment (Phase 3, nullable/best-effort): canonical team codes.
    Column("home_code", String(16)),
    Column("away_code", String(16)),
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
    # Enforced parent reference (WS1c): a line cannot outlive its snapshot, and
    # deleting a snapshot cascades to its lines. Named so the prod DDL / ALTER
    # matches (see sql/schema.sql). SQLite enforces this only with the
    # PRAGMA foreign_keys=ON set on each connection in get_engine().
    Column("snapshot_id", Integer,
           ForeignKey("odds_snapshot.id", ondelete="CASCADE",
                      name="fk_odds_line_snapshot"),
           nullable=False),
    Column("bet_type", String(16), nullable=False),   # moneyline|spread|total|player_prop
    Column("selection", String(160)),                 # team | Over | Under | player
    Column("point", Float),
    Column("player", String(160)),
    Column("prop_key", String(64)),
    Column("direction", String(8)),
    Column("price", Integer),
    Column("implied_prob", Float),
    # SFBB cross-map enrichment (Phase 3, nullable/best-effort): id-based joins.
    Column("player_mlb_id", String(32)),
    Column("team_code", String(16)),
    Column("game_pk", Integer),                        # P3: StatsAPI game (best-effort)
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
    ("price", _i), ("outcome", _i), ("batting_order", _i), ("game_pk", _i),
    ("team", _s), ("player_mlb_id", _s), ("team_code", _s), ("player_key", _s),
    ("source", _s),
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
    ("player_mlb_id", _s), ("team_code", _s), ("opponent_code", _s),
    ("home_code", _s), ("away_code", _s), ("game_pk", _i),
]

_BANKROLL_SPEC = [
    ("txn_id", _s), ("txn_type", _s), ("amount", _f), ("wager_id", _s),
    ("note", _s), ("created_at", _s),
]

_APP_SETTINGS_SPEC = [
    ("setting_key", _s), ("setting_value", _s), ("updated_at", _s),
]

# Team-market forward-tracking columns (event_key is derived, like prediction).
_MARKET_PREDICTION_SPEC = [
    ("ts", _s), ("sport_key", _s), ("event_id", _s), ("commence_time", _s),
    ("game_date", _s), ("bet_type", _s), ("home_team", _s), ("away_team", _s),
    ("team", _s), ("opponent", _s), ("home_away", _s), ("side", _s),
    ("matchup", _s), ("book", _s), ("actual", _s), ("resolved_at", _s),
    ("point", _f), ("model_prob", _f), ("raw_prob", _f),
    ("price", _i), ("outcome", _i),
    ("team_code", _s), ("opponent_code", _s), ("home_code", _s), ("away_code", _s),
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
    # event_key + the hybrid player_key are both DERIVED (recomputed on every
    # write) so the stored columns can never drift from the row's source fields.
    return {
        "event_key": (row.get("event_id") or row.get("game_date") or ""),
        "player_key": player_key(row),
    }


def _prediction_identity(row):
    # Mirrors uq_prediction_identity_v2 — the natural key used to diff rows for
    # surgical writes. Keyed on the hybrid player_key (mlb:<id> else name:<norm>),
    # RECOMPUTED from the row (not read from the stored column) so before/after
    # rows key identically whether or not player_key is materialized yet — this
    # self-heals legacy rows that predate the backfill. event_key is coalesced
    # exactly as _prediction_derive stores it.
    return {
        "sport_key": row.get("sport_key"),
        "event_key": _prediction_derive(row)["event_key"],
        "prop_key": row.get("prop_key"),
        "player_key": player_key(row),
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
    "bankroll_ledger": {"table": bankroll_ledger, "spec": _BANKROLL_SPEC,
                        "derive": lambda row: {},
                        "identity": lambda row: {"txn_id": row.get("txn_id")}},
    "app_settings": {"table": app_settings, "spec": _APP_SETTINGS_SPEC,
                     "derive": lambda row: {},
                     "identity": lambda row: {"setting_key": row.get("setting_key")}},
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
        for key in (*_SECRET_KEYS, "SQL_DRIVER"):   # SQL_DRIVER optional (pyodbc opt-in)
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


def require_sql():
    """True when a SQL deployment is SIGNALLED, so a durable write/refit-read must
    fail loudly rather than silently degrade to ephemeral local disk.

    Environment-only (like _secret), so it is False in every hermetic test and
    every no-SQL dev run, and True exactly when the operator has signalled a SQL
    deployment. Precedence:
      SPORTSBOOK_REQUIRE_SQL=1/true/yes/on  → True   (explicit prod opt-in)
      SPORTSBOOK_REQUIRE_SQL=0/false/no/off → False  (explicit dev escape hatch)
      else                                   → any SQL_* secret present ⇒ SQL intent

    "Any SQL_* present" has no false positives: when True, either enabled() is
    also True (all four present → healthy prod, guard inert) or the config is
    genuinely broken (partial secrets / SQLAlchemy missing / Azure unreachable →
    the loud failure is correct). A dev/test with no SQL deployment sets zero
    SQL_* → False → local fallback fully preserved."""
    flag = os.environ.get(_REQUIRE_SQL_ENV, "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    return any(_secret(key) for key in _SECRET_KEYS)


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
    # Driver: pymssql by default — pure-Python TDS, no system ODBC driver needed
    # (the reason it was chosen; required on Streamlit Cloud). Set SQL_DRIVER=pyodbc
    # to use mssql+pyodbc, which unlocks fast_executemany (array-bound bulk
    # INSERT/UPDATE, ~10x+ vs pymssql's per-row round-trips) for the desktop bulk
    # backfill. That machine needs the "ODBC Driver 18 for SQL Server" +
    # `pip install pyodbc`; the cloud app leaves SQL_DRIVER unset and stays pymssql.
    common = dict(
        username=_secret("SQL_USER"),
        password=_secret("SQL_PASSWORD"),
        host=_secret("SQL_SERVER"),
        port=_SQL_PORT,
        database=_secret("SQL_DATABASE"),
    )
    if (_secret("SQL_DRIVER") or "pymssql").lower() == "pyodbc":
        return URL.create("mssql+pyodbc",
                          query={"driver": "ODBC Driver 18 for SQL Server"},
                          **common)
    return URL.create("mssql+pymssql", **common)


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
            # SQLite ignores FK constraints unless enabled per-connection, so the
            # odds_line -> odds_snapshot FK (WS1c) would be inert in tests. Turn
            # it on so the test harness enforces the same integrity Azure SQL does
            # natively. (Azure SQL/MSSQL enforces declared FKs by default.)
            @event.listens_for(_ENGINE, "connect")
            def _sqlite_fk_pragma(dbapi_conn, _rec):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        else:
            # pool_pre_ping recycles connections dropped during an Azure SQL
            # serverless resume from auto-pause; the timeout gives the first connect
            # after idle time to ride out that resume rather than failing instantly.
            # connect_args differ by driver: pymssql accepts login_timeout, pyodbc
            # does not (it would raise — pyodbc.connect has no such kwarg).
            #
            # fast_executemany (pyodbc only): pymssql runs executemany per-row, so a
            # bulk backfill of tens of thousands of fact rows is a round-trip PER ROW
            # against remote Azure SQL. pyodbc's array binding collapses each batch
            # into one parameterized round-trip (~10x+). Enabled only when the desktop
            # backfill opts into SQL_DRIVER=pyodbc; unset (cloud/pymssql) it is inert.
            is_pyodbc = "pyodbc" in str(url)
            eng_kwargs = dict(pool_pre_ping=True, pool_recycle=1500)
            if is_pyodbc:
                eng_kwargs["fast_executemany"] = True
                eng_kwargs["connect_args"] = {"timeout": 60}
            else:
                eng_kwargs["connect_args"] = {"login_timeout": 60, "timeout": 60}
            _ENGINE = create_engine(url, **eng_kwargs)
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


def reconcile(conn, table, desired, identity_cols, scope=None, ignore_cols=()):
    """App-side-diff reconcile of a *dimension/fact* table to a desired end-state
    — the natural-key analog of :func:`mutate` for the tables that aren't NDJSON
    stores (gamelogs, statcast as-of rates, the SFBB id maps + meta, the
    recalibration params/folds/meta, the id caches).

    Replaces a delete-all + insert-all rebuild with a diff on the caller-declared
    natural key: only rows that actually differ are written, as surgical INSERT /
    UPDATE / DELETE. Unchanged rows aren't churned, their surrogate ids stay
    stable, and the owned scope is never momentarily emptied — the WS15
    integrity/best-practice win, with no change to what data ends up stored.

    Parameters
    ----------
    conn : an OPEN transaction (from ``engine.begin()``). The diff runs inside it,
        so a CHECK/UNIQUE violation rolls the whole reconcile back — matching the
        atomic delete-all path it replaces. Every read happens before any DML, so
        a precondition ``ValueError`` (below) leaves the transaction clean.
    table : the SQLAlchemy ``Table``.
    desired : the end-state rows as column->value param dicts — exactly what the
        old ``insert(table), params`` received (every non-surrogate column
        present; a surrogate ``id`` key, if any, is ignored so INSERTs let the DB
        assign it and UPDATEs never touch it).
    identity_cols : the natural-key column names identifying a row *within the
        scope* (e.g. ``("player_id",)`` under a (season, split, role) scope, or
        ``("sport_key", "prop_key")``).
    scope : an equality ``{col: value}`` map bounding the rows this call owns — the
        partition the old delete-all targeted (``None`` = the whole table). Only
        in-scope rows are read, diffed, and deleted; rows outside it are untouched.
    ignore_cols : column names excluded from the "did this row change?" test but
        still WRITTEN on every INSERT and on any UPDATE a real change triggers.
        The audit-timestamp pattern: a per-row ``fetched_at``/``updated_at`` that
        ticks on every refresh would otherwise force an UPDATE of every row, so
        list it here — an unchanged row stays a no-op (its timestamp then records
        when its *data* last changed), while a genuinely changed row still gets a
        fresh timestamp. A column whose freshness you DO want persisted every call
        (e.g. a single meta row's ``last_fetched_at``) must NOT be listed.

    Returns ``(n_insert, n_update, n_delete)``.

    Raises ``ValueError`` if two desired rows — or two existing in-scope rows —
    share a natural identity (the diff can't tell them apart; a delete-all+insert
    would have surfaced this via the UNIQUE constraint, or silently kept dupes on
    an unconstrained table). Callers on an unconstrained table (the gamelog fact
    tables, whose ``game_key`` isn't unique) must guard against this and fall back
    to a scoped rebuild."""
    def _params(row):
        return {k: v for k, v in row.items() if k != "id"}

    scope_clause = _where_clause(table, scope)

    def _id_where(key):
        # AND the scope predicate in so an UPDATE/DELETE can only touch the
        # partition this call owns. identity_cols need not be globally unique
        # (e.g. ("player_id",) under a (season, split, role) scope), so keying on
        # them alone could otherwise hit an identical natural key in a sibling
        # partition — the leak this closes to honour the docstring's promise that
        # "rows outside [the scope] are untouched".
        parts = [table.c[col] == val for col, val in zip(identity_cols, key)]
        if scope_clause is not None:
            parts.append(scope_clause)
        return and_(*parts)

    # Read the current in-scope rows (SELECT only — no DML yet).
    stmt = select(table)
    if scope_clause is not None:
        stmt = stmt.where(scope_clause)
    before_by_key = {}
    for r in conn.execute(stmt):
        m = r._mapping
        key = tuple(m[c] for c in identity_cols)
        if key in before_by_key:
            raise ValueError(
                f"{table.name}: duplicate identity {key} in existing rows")
        before_by_key[key] = m

    after_by_key = {}
    for row in desired:
        params = _params(row)
        key = tuple(params.get(c) for c in identity_cols)
        if key in after_by_key:
            raise ValueError(
                f"{table.name}: duplicate identity {key} in desired rows")
        after_by_key[key] = params

    # Compute the whole diff before issuing any DML: a precondition raise above
    # then left only SELECTs on the transaction, so the caller can fall back.
    _ignore = set(ignore_cols)
    inserts, updates = [], []
    for key, params in after_by_key.items():
        prior = before_by_key.get(key)
        if prior is None:
            inserts.append(params)
        elif any(prior[k] != v for k, v in params.items() if k not in _ignore):
            updates.append((key, params))
    deletes = [key for key in before_by_key if key not in after_by_key]

    for key, params in updates:
        conn.execute(update(table).where(_id_where(key)).values(**params))
    if inserts:
        conn.execute(insert(table), inserts)
    for key in deletes:
        conn.execute(delete(table).where(_id_where(key)))
    return (len(inserts), len(updates), len(deletes))


def upsert_bulk(conn, table, rows, identity_cols, scope=None, ignore_cols=()):
    """Batched INSERT-or-UPDATE (NEVER delete) of ``rows`` keyed on identity_cols —
    the set-based analog of calling :func:`reconcile` once per row scoped to that
    row's own key (the warehouse ingestion idiom). Does ONE existing-read + ONE bulk
    INSERT (executemany) + one UPDATE per genuinely-changed row inside the caller's
    open transaction, instead of a SELECT+DML round-trip per row — the fix for the
    slow per-row backfill against remote SQL.

    Because it never deletes, it is safe for ACCUMULATING tables (mlb_player /
    mlb_team / mlb_game): a call only ever touches the rows it was handed, never a
    sibling. Two ways to bound the existing-read:
      * ``scope`` — an equality {col: value} map every desired row shares (a
        boxscore's facts share game_pk; a standings snapshot shares
        (season, as_of_date)); reads that partition, diffs by identity_cols.
      * else SINGLE-column identity → reads ``WHERE identity[0] IN (desired values)``.
        (Callers batch per game/day, well under the mssql ~2100-param IN limit; pass
        a ``scope`` instead for very large or composite-key sets.)
    Composite identity WITHOUT a scope is unsupported (raises). ``ignore_cols``
    matches reconcile (excluded from the change test, still written on insert/update).
    Raises ValueError on a duplicate identity in ``rows`` (as reconcile does).
    Returns ``(n_insert, n_update)``."""
    if not rows:
        return (0, 0)
    ident = tuple(identity_cols)

    after_by_key = {}
    for row in rows:
        params = {k: v for k, v in row.items() if k != "id"}
        key = tuple(params.get(c) for c in ident)
        if key in after_by_key:
            raise ValueError(f"{table.name}: duplicate identity {key} in rows")
        after_by_key[key] = params

    scope_clause = _where_clause(table, scope) if scope is not None else None
    stmt = select(table)
    if scope_clause is not None:
        stmt = stmt.where(scope_clause)
    elif scope is None and len(ident) == 1:
        stmt = stmt.where(table.c[ident[0]].in_([k[0] for k in after_by_key]))
    elif scope is None:
        raise ValueError(
            f"{table.name}: upsert_bulk needs a scope for composite identity")

    before_by_key = {}
    for r in conn.execute(stmt):
        m = r._mapping
        key = tuple(m[c] for c in ident)
        if key in before_by_key:
            raise ValueError(
                f"{table.name}: duplicate identity {key} in existing rows")
        before_by_key[key] = m

    _ignore = set(ignore_cols)
    inserts, updates = [], []
    for key, params in after_by_key.items():
        prior = before_by_key.get(key)
        if prior is None:
            inserts.append(params)
        elif any(prior[k] != v for k, v in params.items() if k not in _ignore):
            updates.append((key, params))

    # UPDATEs batched into one executemany per distinct SET column-set (array-bound
    # under fast_executemany) instead of a round-trip per changed row — the fix for
    # update-heavy backfills (e.g. filling new fact columns on rows that already
    # exist) against remote SQL. Identity columns are matched via "_k_"-prefixed
    # binds so they never collide with the SET-column binds (an identity col also
    # appears in the SET, exactly as the prior .values(**params) did). Grouping by
    # the exact column-set keeps each executemany's compiled statement uniform;
    # heterogeneous rows just form separate (usually one) groups. Byte-identical
    # result to the prior per-row loop.
    # A NULL in an identity value needs `col IS NULL` matching (SQLAlchemy renders
    # `col == None` as IS NULL); a bound `col = :p` with p=None renders `col = NULL`,
    # which matches nothing. Unreachable today (every upsert_bulk identity is a
    # NOT NULL column) but handle it per-row so behaviour stays byte-identical to the
    # prior loop; the common NOT-NULL case takes the batched executemany path.
    null_updates = [(k, p) for k, p in updates if None in k]
    batch_updates = [(k, p) for k, p in updates if None not in k]
    for key, params in null_updates:
        parts = [table.c[col] == val for col, val in zip(ident, key)]
        if scope_clause is not None:
            parts.append(scope_clause)
        conn.execute(update(table).where(and_(*parts)).values(**params))
    if batch_updates:
        key_where = [table.c[col] == bindparam("_k_" + col) for col in ident]
        if scope_clause is not None:
            key_where.append(scope_clause)
        where = and_(*key_where)
        by_cols = {}
        for key, params in batch_updates:
            by_cols.setdefault(tuple(sorted(params)), []).append((key, params))
        for set_cols, group in by_cols.items():
            stmt = update(table).where(where).values(
                {c: bindparam(c) for c in set_cols})
            payload = []
            for key, params in group:
                d = dict(params)
                d.update({"_k_" + col: val for col, val in zip(ident, key)})
                payload.append(d)
            conn.execute(stmt, payload)
    if inserts:
        conn.execute(insert(table), inserts)
    return (len(inserts), len(updates))


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
    """Persist a recalibration cfg dict for a sport (atomically).

    WS15: reconcile each of the three tables to its desired end-state via
    :func:`reconcile` (a surgical natural-key upsert on the composite PKs)
    instead of delete-all-by-sport_key + insert-all. A prop/fold whose cfg is
    unchanged across refits isn't rewritten, and the sport's rows are never
    momentarily emptied — the end-state is identical to the old rebuild
    (dropped props/shrunk fold sets are still DELETEd). All three writes share
    one transaction, so a CHECK/UNIQUE violation rolls the whole save back."""
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
    meta_row = {
        "sport_key": sport_key,
        "fit_timestamp": (cfg or {}).get("fit_timestamp"),
        "source": meta.get("source"),
    }
    scope = {"sport_key": sport_key}
    engine = get_engine()
    with engine.begin() as conn:
        reconcile(conn, recalibration_params, param_rows,
                  ("sport_key", "prop_key"), scope=scope)
        reconcile(conn, recalibration_folds, fold_rows,
                  ("sport_key", "prop_key", "fold_index"), scope=scope)
        reconcile(conn, recalibration_meta, [meta_row],
                  ("sport_key",), scope=scope)


# ──────────────────────────────────────────────────────────────────────────────
# Odds warehouse (Phase B) — snapshots + extracted lines
# ──────────────────────────────────────────────────────────────────────────────

_ODDS_LINE_COLS = ("bet_type", "selection", "point", "player", "prop_key",
                   "direction", "price", "implied_prob",
                   "player_mlb_id", "team_code", "game_pk")


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
                "home_code": _s(meta.get("home_code")),
                "away_code": _s(meta.get("away_code")),
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
                    "player_mlb_id": _s(ln.get("player_mlb_id")),
                    "team_code": _s(ln.get("team_code")),
                    "game_pk": _i(ln.get("game_pk")),
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
                     player=None, prop_key=None, direction=None,
                     player_mlb_id=None, team_code=None):
    """The stored line for a descriptor within one snapshot, or None.

    Reproduces _extract_line's matching: props key on (prop_key, player,
    direction); team markets on (selection[, point]) with a fall back to the best
    price across points when the exact point isn't stored.

    When a canonical id is supplied (``player_mlb_id`` for props, ``team_code``
    for moneyline/spread) the identity prefers the id — matching enriched rows by
    id (fixing accents/namesakes) while un-enriched historical rows (id IS NULL)
    still match by name. A None id degrades to the exact name-only behavior."""
    table = odds_line
    bt = (bet_type or "").lower()
    with get_engine().connect() as conn:
        if bt == "player_prop":
            if player_mlb_id:
                ident = ((table.c.player_mlb_id == player_mlb_id)
                         | (table.c.player_mlb_id.is_(None)
                            & (table.c.player == player)))
            else:
                ident = (table.c.player == player)
            row = conn.execute(
                select(table.c.price, table.c.implied_prob).where(
                    (table.c.snapshot_id == snapshot_id)
                    & (table.c.bet_type == "player_prop")
                    & (table.c.prop_key == prop_key)
                    & ident
                    & (table.c.direction == ((direction or "OVER").upper())))
            ).first()
            return {"price": row[0], "implied_prob": row[1]} if row else None

        norm = {"h2h": "moneyline", "moneyline": "moneyline",
                "spreads": "spread", "spread": "spread",
                "totals": "total", "total": "total"}.get(bt, bt)
        sel = ("Under" if norm == "total" and (selection or "").lower() == "under"
               else "Over" if norm == "total" else selection)
        if team_code and norm in ("moneyline", "spread"):
            ident = ((table.c.team_code == team_code)
                     | (table.c.team_code.is_(None) & (table.c.selection == sel)))
        else:
            ident = (table.c.selection == sel)
        base = ((table.c.snapshot_id == snapshot_id)
                & (table.c.bet_type == norm) & ident)
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
            odds_snapshot.c.kind, odds_snapshot.c.home_code,
            odds_snapshot.c.away_code, odds_line.c.snapshot_id,
            odds_line.c.bet_type, odds_line.c.selection, odds_line.c.point,
            odds_line.c.price, odds_line.c.implied_prob, odds_line.c.team_code,
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
                "home_code": r._mapping["home_code"],
                "away_code": r._mapping["away_code"],
                "captured_at": r._mapping["captured_at"],
                "kind": r._mapping["kind"],
                "snapshot_id": r._mapping["snapshot_id"],
                "bet_type": r._mapping["bet_type"],
                "selection": r._mapping["selection"],
                "point": r._mapping["point"],
                "price": r._mapping["price"],
                "implied_prob": r._mapping["implied_prob"],
                "team_code": r._mapping["team_code"],
            } for r in rows]
        except OperationalError as exc:  # transient (cold resume / lock / timeout)
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1 + 2 * attempt)
    raise last_exc


def player_prop_lines(sport, dates=None, date_from=None, date_to=None,
                      max_retries=3):
    """Bulk-read warehoused player-prop lines for a sport, joining each odds_line
    to its parent odds_snapshot.

    Sibling of team_market_lines: same join/retry, but selects bet_type=
    'player_prop' and carries the prop descriptor (player, prop_key, direction).
    Feeds the offline real-line calibration refit (warehouse.load_prop_lines)
    from the growing Azure warehouse instead of the local historical_odds JSON.
    Optional date filter: ``dates`` (explicit game_date list) OR
    ``date_from``/``date_to`` (inclusive range). Ordered by (event_id,
    captured_at) so the assembler can pick each event's closing snapshot. Retries
    transient OperationalErrors like read_rows. Returns row dicts."""
    engine = get_engine()
    joined = odds_line.join(odds_snapshot,
                            odds_line.c.snapshot_id == odds_snapshot.c.id)
    stmt = (
        select(
            odds_snapshot.c.event_id, odds_snapshot.c.game_date,
            odds_snapshot.c.commence_time, odds_snapshot.c.home,
            odds_snapshot.c.away, odds_snapshot.c.captured_at,
            odds_snapshot.c.kind, odds_line.c.snapshot_id,
            odds_line.c.selection, odds_line.c.player, odds_line.c.prop_key,
            odds_line.c.direction, odds_line.c.point, odds_line.c.price,
            odds_line.c.implied_prob, odds_line.c.player_mlb_id,
        )
        .select_from(joined)
        .where((odds_snapshot.c.sport == sport)
               & (odds_line.c.bet_type == "player_prop"))
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
                "selection": r._mapping["selection"],
                "player": r._mapping["player"],
                "player_mlb_id": r._mapping["player_mlb_id"],
                "prop_key": r._mapping["prop_key"],
                "direction": r._mapping["direction"],
                "point": r._mapping["point"],
                "price": r._mapping["price"],
                "implied_prob": r._mapping["implied_prob"],
            } for r in rows]
        except OperationalError as exc:  # transient (cold resume / lock / timeout)
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(1 + 2 * attempt)
    raise last_exc
