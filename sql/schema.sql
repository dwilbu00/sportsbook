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
    -- 0 when logged; set 1 after an offline calibration refit consumes the row
    -- (powers the app's "enough new data to refit" banner).
    refit_performed BIT NOT NULL CONSTRAINT df_prediction_refit DEFAULT (0),
    -- Rule inputs the pick-rules ROI lens (pickrules_roi.py) needs to RE-DERIVE
    -- the recommended slate from logged forecasts. Nullable: pre-feature rows
    -- have NULL and their team-based rules (Rule-of-3, opposing-team L3) are
    -- reported as skipped rather than replayed.
    team          NVARCHAR(160),
    batting_order INT,
    CONSTRAINT uq_prediction_identity
        UNIQUE (sport_key, event_key, prop_key, player, line)
);
GO
-- Idempotent add for an EXISTING prediction_log (the CREATE TABLE above only runs
-- on a fresh DB). Run this once against the live database.
IF COL_LENGTH('dbo.prediction_log', 'refit_performed') IS NULL
    ALTER TABLE dbo.prediction_log
        ADD refit_performed BIT NOT NULL
            CONSTRAINT df_prediction_refit DEFAULT (0);
GO
IF COL_LENGTH('dbo.prediction_log', 'team') IS NULL
    ALTER TABLE dbo.prediction_log ADD team NVARCHAR(160);
GO
IF COL_LENGTH('dbo.prediction_log', 'batting_order') IS NULL
    ALTER TABLE dbo.prediction_log ADD batting_order INT;
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_prediction_sport_resolved'
                 AND object_id = OBJECT_ID('dbo.prediction_log'))
CREATE INDEX ix_prediction_sport_resolved
    ON dbo.prediction_log (sport_key, resolved);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_prediction_refit_pending'
                 AND object_id = OBJECT_ID('dbo.prediction_log'))
CREATE INDEX ix_prediction_refit_pending
    ON dbo.prediction_log (resolved, refit_performed);
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

------------------------------------------------------- market_prediction_log
-- Forward tracking for TEAM markets (moneyline / spread / total): the MODEL's
-- pick per (game, market), logged independent of any wager and resolved against
-- final box scores. The sibling of prediction_log (player props) — kept a
-- separate table so the prop calibration/refit pipeline that reads
-- prediction_log stays uncontaminated. One row = the favored side (ML/spread) or
-- over/under lean (total) with its probability, price, and value flag.
IF OBJECT_ID('dbo.market_prediction_log', 'U') IS NULL
CREATE TABLE dbo.market_prediction_log (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    ts            NVARCHAR(40),
    sport_key     NVARCHAR(64)  NOT NULL,
    event_id      NVARCHAR(128),
    event_key     NVARCHAR(160) NOT NULL,          -- event_id or game_date
    commence_time NVARCHAR(40),
    game_date     NVARCHAR(10),
    bet_type      NVARCHAR(16)  NOT NULL,           -- moneyline|spread|total
    home_team     NVARCHAR(128),
    away_team     NVARCHAR(128),
    team          NVARCHAR(128),
    opponent      NVARCHAR(128),
    home_away     NVARCHAR(8),
    side          NVARCHAR(16)  NOT NULL,           -- home|away|over|under
    matchup       NVARCHAR(256),
    book          NVARCHAR(64),
    actual        NVARCHAR(64),                     -- "home-away" final score string
    resolved_at   NVARCHAR(40),
    point         FLOAT,                            -- spread/total line; NULL for ML
    model_prob    FLOAT,                            -- picked side prob (0-1)
    raw_prob      FLOAT,                            -- pre-blend prob (0-1)
    price         INT,
    outcome       INT,                              -- 1=pick won, 0=lost, NULL=push/unresolved
    is_value      BIT,                              -- tri-state (NULL allowed)
    resolved      BIT NOT NULL
        CONSTRAINT df_market_prediction_resolved DEFAULT (0),
    CONSTRAINT uq_market_prediction_identity
        UNIQUE (sport_key, event_key, bet_type)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_market_prediction_sport_resolved'
                 AND object_id = OBJECT_ID('dbo.market_prediction_log'))
CREATE INDEX ix_market_prediction_sport_resolved
    ON dbo.market_prediction_log (sport_key, resolved);
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

-- ═══════════════════════════════════════════════════════════════════════════
-- Durable rolling ESPN gamelog store (Phase C). Replaces the ephemeral file
-- cache (cache/backtest/*.json) so completed games survive Cloud restarts.
-- Per-sport dense fact tables: columns = ONLY the stats the app reads, so a read
-- reconstructs the exact get_athlete_gamelog dict shape. Mirrors gamelog_store.py
-- (test_gamelog_store.py::SchemaParityTests enforces it). MLB + NBA only; NFL and
-- other sports pass through to direct ESPN with no persistence. season_bucket 0 =
-- current/None season; a specific past year is its own immutable bucket.
-- game_date is the FULL ISO timestamp (its time component disambiguates
-- doubleheaders). Refresh is DELETE+INSERT per (athlete, season_bucket).
-- ═══════════════════════════════════════════════════════════════════════════

--------------------------------------------------------------- mlb_batter_gamelog
IF OBJECT_ID('dbo.mlb_batter_gamelog', 'U') IS NULL
CREATE TABLE dbo.mlb_batter_gamelog (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    athlete_id    NVARCHAR(32) NOT NULL,
    season_bucket INT NOT NULL,                     -- 0 = current/None season
    game_key      NVARCHAR(220),                    -- synthetic (not unique)
    game_date     NVARCHAR(40),                     -- FULL ISO timestamp
    opponent      NVARCHAR(160),
    is_home       BIT,
    team_id       NVARCHAR(32),
    completed     BIT,
    [AB]  FLOAT, [H]   FLOAT, [SO]  FLOAT, [BB] FLOAT,
    [HBP] FLOAT, [SF]  FLOAT, [SH]  FLOAT
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_batter_gamelog_athlete'
                 AND object_id = OBJECT_ID('dbo.mlb_batter_gamelog'))
CREATE INDEX ix_mlb_batter_gamelog_athlete
    ON dbo.mlb_batter_gamelog (athlete_id, season_bucket);
GO

-------------------------------------------------------------- mlb_pitcher_gamelog
IF OBJECT_ID('dbo.mlb_pitcher_gamelog', 'U') IS NULL
CREATE TABLE dbo.mlb_pitcher_gamelog (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    athlete_id    NVARCHAR(32) NOT NULL,
    season_bucket INT NOT NULL,
    game_key      NVARCHAR(220),
    game_date     NVARCHAR(40),
    opponent      NVARCHAR(160),
    is_home       BIT,
    team_id       NVARCHAR(32),
    completed     BIT,
    [IP] FLOAT, [K] FLOAT, [ER] FLOAT               -- IP raw (outs derived at runtime)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_pitcher_gamelog_athlete'
                 AND object_id = OBJECT_ID('dbo.mlb_pitcher_gamelog'))
CREATE INDEX ix_mlb_pitcher_gamelog_athlete
    ON dbo.mlb_pitcher_gamelog (athlete_id, season_bucket);
GO

--------------------------------------------------------------------- nba_gamelog
IF OBJECT_ID('dbo.nba_gamelog', 'U') IS NULL
CREATE TABLE dbo.nba_gamelog (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    athlete_id    NVARCHAR(32) NOT NULL,
    season_bucket INT NOT NULL,
    game_key      NVARCHAR(220),
    game_date     NVARCHAR(40),
    opponent      NVARCHAR(160),
    is_home       BIT,
    team_id       NVARCHAR(32),
    completed     BIT,
    [MIN] FLOAT, [PTS] FLOAT, [REB] FLOAT, [AST] FLOAT
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_nba_gamelog_athlete'
                 AND object_id = OBJECT_ID('dbo.nba_gamelog'))
CREATE INDEX ix_nba_gamelog_athlete
    ON dbo.nba_gamelog (athlete_id, season_bucket);
GO

--------------------------------------------------------------- gamelog_fetch_meta
-- TTL gate + which fact table an athlete lives in (player_type).
IF OBJECT_ID('dbo.gamelog_fetch_meta', 'U') IS NULL
CREATE TABLE dbo.gamelog_fetch_meta (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    sport           NVARCHAR(32) NOT NULL,
    league          NVARCHAR(32) NOT NULL,
    athlete_id      NVARCHAR(32) NOT NULL,
    season_bucket   INT NOT NULL,
    player_type     NVARCHAR(16),                   -- batter|pitcher|nba
    last_fetched_at FLOAT,                           -- epoch seconds
    game_count      INT,
    CONSTRAINT uq_gamelog_fetch_meta
        UNIQUE (sport, league, athlete_id, season_bucket)
);
GO

---------------------------------------------------------------- athlete_id_cache
-- Durable name->id (replaces the file cached_athlete_id); stores team_id so the
-- get_player_stat_history reroute keeps athlete.team_id. team_key = sorted
-- team_ids (or '') to preserve same-name disambiguation.
IF OBJECT_ID('dbo.athlete_id_cache', 'U') IS NULL
CREATE TABLE dbo.athlete_id_cache (
    id                INT IDENTITY(1,1) PRIMARY KEY,
    sport             NVARCHAR(32) NOT NULL,
    league            NVARCHAR(32) NOT NULL,
    player_name_lower NVARCHAR(160) NOT NULL,
    team_key          NVARCHAR(64) NOT NULL,
    athlete_id        NVARCHAR(32),                  -- NULL = not found
    name              NVARCHAR(160),
    team_id           NVARCHAR(32),
    fetched_at        FLOAT,
    CONSTRAINT uq_athlete_id_cache
        UNIQUE (sport, league, player_name_lower, team_key)
);
GO

----------------------------------------------------------- statcast_player_asof
-- P2.4a/b: small DERIVED per-player Statcast as-of rate table. Built OFFLINE from
-- the raw (gitignored, ephemeral) pitch cache by statcast_asof.py and read live by
-- props.py. P2.4a: batter xBA/xwOBA. P2.4b: adds plate-discipline / contact rates
-- (whiff/CSW/hard-hit/barrel) + a `role` ("bat"|"pit") discriminator (both share
-- the MLBAM id space). One row per (player_id, season_bucket, split, role);
-- replace-all per (season, split, role) refresh (fully rebuildable).
--
-- ⚠ P2.4b MIGRATION (existing §2.4a table): `role` joins the UNIQUE key, so an
--   already-created table must be rebuilt. This table is 100% derived, so the
--   simplest one-time migration is to DROP + recreate + re-run the build:
--       DROP TABLE dbo.statcast_player_asof;   -- run ONCE, then the CREATE below
--   (then `python statcast_asof.py --build --season 2026` [+ 2024] repopulates.)
IF OBJECT_ID('dbo.statcast_player_asof', 'U') IS NULL
CREATE TABLE dbo.statcast_player_asof (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    player_id     NVARCHAR(32) NOT NULL,             -- MLBAM id
    season_bucket INT NOT NULL,                       -- season year (e.g. 2026)
    split         NVARCHAR(16) NOT NULL,              -- all|vsL|vsR
    role          NVARCHAR(4) NOT NULL,               -- bat|pit
    as_of_date    NVARCHAR(16),                       -- YYYY-MM-DD build cutoff
    xba           FLOAT,                               -- expected BA (per AB) [bat]
    xwoba         FLOAT,                               -- xwOBAcon (per batted ball)
    n_ab          INT,                                 -- official ABs behind xba
    n_bbe         INT,                                 -- batted balls behind xwoba
    whiff_pct     FLOAT,                               -- whiffs / swings
    csw_pct       FLOAT,                               -- (called + whiff) / pitches
    hard_hit_pct  FLOAT,                               -- LS>=95 / batted balls
    barrel_pct    FLOAT,                               -- barrels / batted balls
    n_pitches     INT,                                 -- pitches behind whiff/csw
    n_bip         INT,                                 -- batted balls behind hh/brl
    CONSTRAINT uq_statcast_player_asof
        UNIQUE (player_id, season_bucket, split, role)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_statcast_player_asof_key'
                 AND object_id = OBJECT_ID('dbo.statcast_player_asof'))
CREATE INDEX ix_statcast_player_asof_key
    ON dbo.statcast_player_asof (season_bucket, split, role);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_statcast_player_asof_player'
                 AND object_id = OBJECT_ID('dbo.statcast_player_asof'))
CREATE INDEX ix_statcast_player_asof_player
    ON dbo.statcast_player_asof (player_id, season_bucket, role);
GO
