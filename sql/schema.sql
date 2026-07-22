-- Azure SQL schema for the sportsbook value-finder durable state.
--
-- Run this MANUALLY as the server admin (CloudSA…) BEFORE first app use.
-- The app connects only as the least-privilege user StreamlitApp (CRUD; see
-- grants.sql) and never issues DDL. Keep this in lockstep with db_store.py's
-- SQLAlchemy metadata (test_db_store.py::SchemaParityTests enforces it).
--
-- Fully columnar (one column per field) so the data is directly queryable in
-- SQL. The only nested structure — a recalibration prop's validation_folds —
-- is the child table recalibration_folds.

------------------------------------------------------------------- prediction_log
IF OBJECT_ID('dbo.prediction_log', 'U') IS NULL
CREATE TABLE dbo.prediction_log (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    ts            NVARCHAR(40),
    sport_key     NVARCHAR(64)  NOT NULL,
    event_id      NVARCHAR(128),
    event_key     NVARCHAR(160) NOT NULL,          -- event_id or game_date
    commence_time NVARCHAR(40),
    prop_key      NVARCHAR(64)  NOT NULL,
    player        NVARCHAR(160) NOT NULL,
    game_date     NVARCHAR(10),
    direction     NVARCHAR(8),
    book          NVARCHAR(64),
    resolved_at   NVARCHAR(40),
    line          FLOAT         NOT NULL,
    raw_prob      FLOAT,
    final_prob    FLOAT,
    projected     FLOAT,
    actual        FLOAT,
    price         INT,
    outcome       INT,                             -- 1=over, 0=under, NULL=push/unresolved
    is_value      BIT,                             -- tri-state (NULL allowed)
    resolved      BIT NOT NULL CONSTRAINT df_prediction_resolved DEFAULT (0),
    CONSTRAINT uq_prediction_identity
        UNIQUE (sport_key, event_key, prop_key, player, line)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_prediction_sport_resolved'
                 AND object_id = OBJECT_ID('dbo.prediction_log'))
CREATE INDEX ix_prediction_sport_resolved
    ON dbo.prediction_log (sport_key, resolved);
GO

--------------------------------------------------------------------------- wagers
IF OBJECT_ID('dbo.wagers', 'U') IS NULL
CREATE TABLE dbo.wagers (
    id             INT IDENTITY(1,1) PRIMARY KEY,
    wager_id       NVARCHAR(64) NOT NULL,
    placed_at      NVARCHAR(40),
    sport_key      NVARCHAR(64),
    bet_type       NVARCHAR(32),
    event_id       NVARCHAR(128),
    commence_time  NVARCHAR(40),
    game_date      NVARCHAR(10),
    home_team      NVARCHAR(128),
    away_team      NVARCHAR(128),
    matchup        NVARCHAR(256),
    team           NVARCHAR(128),
    opponent       NVARCHAR(128),
    home_away      NVARCHAR(8),
    player         NVARCHAR(160),
    prop_key       NVARCHAR(64),
    prop_label     NVARCHAR(64),
    direction      NVARCHAR(8),
    side           NVARCHAR(16),
    book           NVARCHAR(64),
    status         NVARCHAR(16),
    actual         NVARCHAR(64),                    -- mixed float/score-string → text
    resolved_at    NVARCHAR(40),
    point          FLOAT,
    line           FLOAT,
    stake          FLOAT,
    model_prob     FLOAT,
    model_edge     FLOAT,
    close_line     FLOAT,
    clv_pct        FLOAT,
    profit         FLOAT,
    executed_price INT,
    model_price    INT,
    close_price    INT,
    CONSTRAINT uq_wager_id UNIQUE (wager_id),
    CONSTRAINT ck_wager_status
        CHECK (status IN ('pending', 'won', 'lost', 'push', 'void')),
    CONSTRAINT ck_wager_stake CHECK (stake IS NULL OR stake >= 0)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_wager_status'
                 AND object_id = OBJECT_ID('dbo.wagers'))
CREATE INDEX ix_wager_status ON dbo.wagers (status);
GO

--------------------------------------------------------------- recalibration_params
IF OBJECT_ID('dbo.recalibration_params', 'U') IS NULL
CREATE TABLE dbo.recalibration_params (
    sport_key                   NVARCHAR(64) NOT NULL,
    prop_key                    NVARCHAR(64) NOT NULL,
    a                           FLOAT,
    b                           FLOAT,
    n_fit                       INT,
    n_validation                INT,
    n_validation_folds          INT,
    holdout_start               NVARCHAR(10),
    holdout_raw_brier           FLOAT,
    holdout_calibrated_brier    FLOAT,
    holdout_raw_log_loss        FLOAT,
    holdout_calibrated_log_loss FLOAT,
    holdout_metric_scope        NVARCHAR(64),
    deploy_fit_scope            NVARCHAR(64),
    validated                   BIT,
    source                      NVARCHAR(64),
    CONSTRAINT pk_recalibration_params PRIMARY KEY (sport_key, prop_key)
);
GO

---------------------------------------------------------------- recalibration_folds
IF OBJECT_ID('dbo.recalibration_folds', 'U') IS NULL
CREATE TABLE dbo.recalibration_folds (
    sport_key          NVARCHAR(64) NOT NULL,
    prop_key           NVARCHAR(64) NOT NULL,
    fold_index         INT          NOT NULL,
    holdout_start      NVARCHAR(10),
    n_validation       INT,
    raw_brier          FLOAT,
    calibrated_brier   FLOAT,
    raw_log_loss       FLOAT,
    calibrated_log_loss FLOAT,
    CONSTRAINT pk_recalibration_folds
        PRIMARY KEY (sport_key, prop_key, fold_index)
);
GO

----------------------------------------------------------------- recalibration_meta
IF OBJECT_ID('dbo.recalibration_meta', 'U') IS NULL
CREATE TABLE dbo.recalibration_meta (
    sport_key       NVARCHAR(64) NOT NULL PRIMARY KEY,
    fit_timestamp   NVARCHAR(40),
    source          NVARCHAR(64)
);
GO

------------------------------------------------------------------- odds_snapshot
-- Odds warehouse (Phase B): normalized, replaces the Blob snapshot blobs +
-- _manifest.json. One row per captured snapshot (write-once per hour bucket).
IF OBJECT_ID('dbo.odds_snapshot', 'U') IS NULL
CREATE TABLE dbo.odds_snapshot (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    sport         NVARCHAR(64)  NOT NULL,
    game_date     NVARCHAR(10)  NOT NULL,
    event_id      NVARCHAR(128) NOT NULL,
    kind          NVARCHAR(16)  NOT NULL,          -- team|props|alt|seed
    snapshot_hour NVARCHAR(16)  NOT NULL,          -- YYYYMMDDTHHZ bucket
    captured_at   NVARCHAR(40),
    commence_time NVARCHAR(40),
    home          NVARCHAR(128),
    away          NVARCHAR(128),
    regions       NVARCHAR(64),
    markets       NVARCHAR(256),
    bookmakers    NVARCHAR(256),
    CONSTRAINT uq_odds_snapshot
        UNIQUE (sport, game_date, event_id, kind, snapshot_hour)  -- write-once
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_odds_snapshot_event'
                 AND object_id = OBJECT_ID('dbo.odds_snapshot'))
CREATE INDEX ix_odds_snapshot_event
    ON dbo.odds_snapshot (sport, game_date, event_id);
GO

----------------------------------------------------------------------- odds_line
-- One row per extracted line within a snapshot. price/implied reproduce
-- closing_line_for's extraction (best-across-books for team; consensus for
-- props), computed at capture. snapshot_id references odds_snapshot.id.
IF OBJECT_ID('dbo.odds_line', 'U') IS NULL
CREATE TABLE dbo.odds_line (
    id           INT IDENTITY(1,1) PRIMARY KEY,
    snapshot_id  INT NOT NULL,
    bet_type     NVARCHAR(16) NOT NULL,            -- moneyline|spread|total|player_prop
    selection    NVARCHAR(160),                    -- team | Over | Under | player
    point        FLOAT,
    player       NVARCHAR(160),
    prop_key     NVARCHAR(64),
    direction    NVARCHAR(8),
    price        INT,
    implied_prob FLOAT
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_odds_line_snapshot'
                 AND object_id = OBJECT_ID('dbo.odds_line'))
CREATE INDEX ix_odds_line_snapshot ON dbo.odds_line (snapshot_id);
GO
