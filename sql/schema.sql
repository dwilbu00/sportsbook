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
-- SFBB cross-map enrichment (Phase 3): MLBAM id + canonical team code + the hybrid
-- player_key. All nullable now; player_key becomes the UNIQUE identity in Phase 4
-- (backfill populates it + merges collisions BEFORE the unique swap).
IF COL_LENGTH('dbo.prediction_log', 'player_mlb_id') IS NULL
    ALTER TABLE dbo.prediction_log ADD player_mlb_id NVARCHAR(32);
GO
IF COL_LENGTH('dbo.prediction_log', 'team_code') IS NULL
    ALTER TABLE dbo.prediction_log ADD team_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.prediction_log', 'player_key') IS NULL
    ALTER TABLE dbo.prediction_log ADD player_key NVARCHAR(200);
GO
-- P3: the StatsAPI game_pk the prop belongs to (nullable/best-effort, stamped by
-- entity_resolver). NOT in the UNIQUE key — prop_key + line already distinguish a
-- player's many props in one game. P4 enforces MLB-scoped non-NULL; P5 backfills.
IF COL_LENGTH('dbo.prediction_log', 'game_pk') IS NULL
    ALTER TABLE dbo.prediction_log ADD game_pk INT;
GO
-- P6 teardown verification: which data path served the model-input history at
-- prediction time — "warehouse" (StatsAPI facts) or "espn". Nullable/best-effort;
-- makes a warehouse-gate flip's effect auditable per prediction (the durable
-- "prove the gate is carrying the load" signal) before ESPN removal.
IF COL_LENGTH('dbo.prediction_log', 'source') IS NULL
    ALTER TABLE dbo.prediction_log ADD source NVARCHAR(16);
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
-- ── Phase 4: hybrid-identity swap (player name → player_key) ──
-- The hybrid player_key ("mlb:<id>" else "name:<norm>") becomes the UNIQUE forecast
-- identity, replacing the raw player name so accent variants / namesakes of one
-- player collapse to a single row. ORDERING IS THE CRUX: run backfill_player_ids.py
-- FIRST — it populates player_key on every row and merges any spelling collisions.
-- This block is a guarded NO-OP until then: it fires only when no prediction_log row
-- has a NULL player_key, which also prevents SQL Server's "every NULL is equal" rule
-- from failing the new UNIQUE. Safe + idempotent to re-run (fresh empty DB included).
IF NOT EXISTS (SELECT 1 FROM dbo.prediction_log WHERE player_key IS NULL)
BEGIN
    IF EXISTS (SELECT 1 FROM sys.columns
               WHERE object_id = OBJECT_ID('dbo.prediction_log')
                 AND name = 'player_key' AND is_nullable = 1)
        ALTER TABLE dbo.prediction_log
            ALTER COLUMN player_key NVARCHAR(200) NOT NULL;

    -- Swap the unique ATOMICALLY: create v2 BEFORE dropping the old one, both inside
    -- one XACT_ABORT transaction. If ADD v2 fails (e.g. an un-merged residual
    -- duplicate slipped past the backfill), the whole swap rolls back and the old
    -- uq_prediction_identity stays in place — the table is NEVER left with no forecast-
    -- identity guard. Both ALTERs stay existence-guarded, so re-running is idempotent.
    SET XACT_ABORT ON;
    BEGIN TRANSACTION;
        IF NOT EXISTS (SELECT 1 FROM sys.key_constraints
                       WHERE name = 'uq_prediction_identity_v2'
                         AND parent_object_id = OBJECT_ID('dbo.prediction_log'))
            ALTER TABLE dbo.prediction_log
                ADD CONSTRAINT uq_prediction_identity_v2
                    UNIQUE (sport_key, event_key, prop_key, player_key, line);

        IF EXISTS (SELECT 1 FROM sys.key_constraints
                   WHERE name = 'uq_prediction_identity'
                     AND parent_object_id = OBJECT_ID('dbo.prediction_log'))
            ALTER TABLE dbo.prediction_log DROP CONSTRAINT uq_prediction_identity;
    COMMIT TRANSACTION;
END
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
-- SFBB cross-map enrichment (Phase 3): id-based joins. All nullable/best-effort.
IF COL_LENGTH('dbo.wagers', 'player_mlb_id') IS NULL
    ALTER TABLE dbo.wagers ADD player_mlb_id NVARCHAR(32);
GO
IF COL_LENGTH('dbo.wagers', 'team_code') IS NULL
    ALTER TABLE dbo.wagers ADD team_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.wagers', 'opponent_code') IS NULL
    ALTER TABLE dbo.wagers ADD opponent_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.wagers', 'home_code') IS NULL
    ALTER TABLE dbo.wagers ADD home_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.wagers', 'away_code') IS NULL
    ALTER TABLE dbo.wagers ADD away_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.wagers', 'game_pk') IS NULL      -- P3: StatsAPI game (best-effort)
    ALTER TABLE dbo.wagers ADD game_pk INT;
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
-- SFBB cross-map enrichment (Phase 3): canonical team codes. All nullable.
IF COL_LENGTH('dbo.market_prediction_log', 'team_code') IS NULL
    ALTER TABLE dbo.market_prediction_log ADD team_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.market_prediction_log', 'opponent_code') IS NULL
    ALTER TABLE dbo.market_prediction_log ADD opponent_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.market_prediction_log', 'home_code') IS NULL
    ALTER TABLE dbo.market_prediction_log ADD home_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.market_prediction_log', 'away_code') IS NULL
    ALTER TABLE dbo.market_prediction_log ADD away_code NVARCHAR(16);
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
-- SFBB cross-map enrichment (Phase 3): canonical team codes. All nullable.
IF COL_LENGTH('dbo.odds_snapshot', 'home_code') IS NULL
    ALTER TABLE dbo.odds_snapshot ADD home_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.odds_snapshot', 'away_code') IS NULL
    ALTER TABLE dbo.odds_snapshot ADD away_code NVARCHAR(16);
GO
-- Capture provenance (2026-08-21): 'live' (live-analysis fetch) | 'backfill'
-- (historical-odds backfill) | 'seed' (bulk season seed) | 'sbr' (defunct).
-- Nullable; legacy rows retro-tagged via odds_provenance.py --retag. Does not
-- affect reads; lets backtests filter to a consistent source/snapshot.
IF COL_LENGTH('dbo.odds_snapshot', 'source') IS NULL
    ALTER TABLE dbo.odds_snapshot ADD source NVARCHAR(16);
GO

----------------------------------------------------------------------- odds_line
-- One row per extracted line within a snapshot. price/implied reproduce
-- closing_line_for's extraction (best-across-books for team; consensus for
-- props), computed at capture. snapshot_id references odds_snapshot.id, enforced
-- by fk_odds_line_snapshot ON DELETE CASCADE (WS1c) so a line cannot outlive its
-- snapshot and deleting a snapshot removes its lines.
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
    implied_prob FLOAT,
    bookmaker    NVARCHAR(64),                     -- per-book grain (multibook migration)
    region       NVARCHAR(16),                     -- us|eu (Pinnacle = eu)
    CONSTRAINT fk_odds_line_snapshot
        FOREIGN KEY (snapshot_id) REFERENCES dbo.odds_snapshot (id)
        ON DELETE CASCADE
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_odds_line_snapshot'
                 AND object_id = OBJECT_ID('dbo.odds_line'))
CREATE INDEX ix_odds_line_snapshot ON dbo.odds_line (snapshot_id);
GO
-- WS1c: add the FK to a pre-existing odds_line (created before the constraint).
-- Idempotent; WITH CHECK validates current rows, so clear any orphaned lines
-- (snapshot_id with no parent) first if this ever errors on legacy data.
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys
               WHERE name = 'fk_odds_line_snapshot'
                 AND parent_object_id = OBJECT_ID('dbo.odds_line'))
    ALTER TABLE dbo.odds_line WITH CHECK
        ADD CONSTRAINT fk_odds_line_snapshot
        FOREIGN KEY (snapshot_id) REFERENCES dbo.odds_snapshot (id)
        ON DELETE CASCADE;
GO
-- SFBB cross-map enrichment (Phase 3): id-based joins. All nullable/best-effort.
IF COL_LENGTH('dbo.odds_line', 'player_mlb_id') IS NULL
    ALTER TABLE dbo.odds_line ADD player_mlb_id NVARCHAR(32);
GO
IF COL_LENGTH('dbo.odds_line', 'team_code') IS NULL
    ALTER TABLE dbo.odds_line ADD team_code NVARCHAR(16);
GO
IF COL_LENGTH('dbo.odds_line', 'game_pk') IS NULL   -- P3: StatsAPI game (best-effort)
    ALTER TABLE dbo.odds_line ADD game_pk INT;
GO
-- Per-book grain (multibook migration, 2026-08-25): one odds_line row per
-- (bookmaker, point) within a snapshot, so a single snapshot holds every book's
-- price (incl. Pinnacle for the R2 sharp reference) instead of one collapsed row.
-- Nullable; legacy DK-only rows (NBA/NFL backfill) are retro-set to 'draftkings'
-- by the migration so the DK readers can filter bookmaker='draftkings' for strict
-- parity. region = us|eu (Pinnacle = eu). RUN THIS ALTER BEFORE deploying the
-- bookmaker-aware capture_odds_snapshot / ingester.
IF COL_LENGTH('dbo.odds_line', 'bookmaker') IS NULL
    ALTER TABLE dbo.odds_line ADD bookmaker NVARCHAR(64);
GO
IF COL_LENGTH('dbo.odds_line', 'region') IS NULL
    ALTER TABLE dbo.odds_line ADD region NVARCHAR(16);
GO
-- Snapshot-scoped bookmaker filtering (DK-parity readers + R2 sharp reads) now
-- that a snapshot holds ~40 books' rows rather than one collapsed row.
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_odds_line_snapshot_book'
                 AND object_id = OBJECT_ID('dbo.odds_line'))
CREATE INDEX ix_odds_line_snapshot_book ON dbo.odds_line (snapshot_id, bookmaker);
GO

-- Odds-coverage monitoring views (2026-08-21). v_odds_coverage: raw counts across
-- every dimension (all sports). v_mlb_team_coverage_vs_eligible: MLB team-market
-- completeness vs eligible completed games (MLB is the only sport with a warehouse
-- game-fact table; NBA/NFL show absolute counts in v_odds_coverage only).
CREATE OR ALTER VIEW dbo.v_odds_coverage AS
SELECT
    s.sport,
    s.kind                          AS category,     -- team | props | seed | alt
    s.source,                                        -- live | backfill_early/_close | seed | sbr
    l.bet_type,                                      -- moneyline | spread | total | player_prop
    l.prop_key,                                      -- NULL for team; market key for props
    LEFT(s.game_date, 4)            AS season_year,
    COUNT(DISTINCT CONCAT(s.event_id, '|', s.game_date)) AS games,
    COUNT(DISTINCT s.id)            AS snapshots,
    COUNT(l.id)                     AS lines
FROM dbo.odds_snapshot s
LEFT JOIN dbo.odds_line l ON l.snapshot_id = s.id
GROUP BY s.sport, s.kind, s.source, l.bet_type, l.prop_key, LEFT(s.game_date, 4);
GO
CREATE OR ALTER VIEW dbo.v_mlb_team_coverage_vs_eligible AS
WITH eligible AS (
    SELECT LEFT(official_date, 4) AS season_year, COUNT(*) AS eligible_games
    FROM dbo.mlb_game
    WHERE game_type IN ('R','F','D','L','W')
      AND status   IN ('Final','Completed Early','Game Over')
    GROUP BY LEFT(official_date, 4)
),
covered AS (
    SELECT LEFT(s.game_date, 4) AS season_year, l.bet_type,
           COUNT(DISTINCT CONCAT(s.event_id, '|', s.game_date)) AS games_with_line
    FROM dbo.odds_snapshot s
    JOIN dbo.odds_line l ON l.snapshot_id = s.id
    WHERE s.sport = 'baseball_mlb' AND l.bet_type IN ('moneyline','spread','total')
    GROUP BY LEFT(s.game_date, 4), l.bet_type
)
SELECT e.season_year, bt.bet_type, e.eligible_games,
       ISNULL(c.games_with_line, 0)                    AS games_with_line,
       e.eligible_games - ISNULL(c.games_with_line, 0) AS gap
FROM eligible e
CROSS JOIN (VALUES ('moneyline'),('spread'),('total')) AS bt(bet_type)
LEFT JOIN covered c ON c.season_year = e.season_year AND c.bet_type = bt.bet_type;
GO

-- ═══════════════════════════════════════════════════════════════════════════
-- Durable rolling ESPN gamelog store (Phase C). Replaces the ephemeral file
-- cache (cache/backtest/*.json) so completed games survive Cloud restarts.
-- Per-sport dense fact tables: columns = ONLY the stats the app reads, so a read
-- reconstructs the exact get_athlete_gamelog dict shape. Mirrors gamelog_store.py
-- (test_gamelog_store.py::SchemaParityTests enforces it). MLB + NBA + NFL; other
-- sports (e.g. NHL) pass through to direct ESPN with no persistence. season_bucket 0 =
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

--------------------------------------------------------------------- nfl_gamelog
-- ONE position-dependent row per game; the app reads only two labels (pass/rush
-- yds both -> [YDS]; anytime TD -> [TD]). [YDS] is passing yds for a QB, rushing
-- for a RB, receiving for a WR (see gamelog_store._NFL_STATS).
IF OBJECT_ID('dbo.nfl_gamelog', 'U') IS NULL
CREATE TABLE dbo.nfl_gamelog (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    athlete_id    NVARCHAR(32) NOT NULL,
    season_bucket INT NOT NULL,
    game_key      NVARCHAR(220),
    game_date     NVARCHAR(40),
    opponent      NVARCHAR(160),
    is_home       BIT,
    team_id       NVARCHAR(32),
    completed     BIT,
    [YDS] FLOAT, [TD] FLOAT
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_nfl_gamelog_athlete'
                 AND object_id = OBJECT_ID('dbo.nfl_gamelog'))
CREATE INDEX ix_nfl_gamelog_athlete
    ON dbo.nfl_gamelog (athlete_id, season_bucket);
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

----------------------------------------------------------------- statcast_pitch
-- Durable home for the RAW per-day Statcast pitch rows (previously the local,
-- machine-specific cache/statcast_days_v5/ files). This is the FULL per-date
-- as-of curve the OFFLINE backtest/refit reads via savant_history.load_days —
-- distinct from statcast_player_asof (a small single-cutoff DERIVED table the
-- live app reads). Moving it here makes the refit's xBA index + method D
-- available on ANY box and removes the silent "no Statcast cached → batter_hits
-- mis-selects" trap. One row per pitch (trimmed v5 shape); day-atomic replace
-- (DELETE WHERE game_date + bulk INSERT) via savant_history.ingest_day. ~700K
-- rows/season; fully rebuildable from Baseball Savant. Mirrors savant_history.py
-- (test_statcast_pitch.py::SchemaParityTests enforces the column set).
IF OBJECT_ID('dbo.statcast_pitch', 'U') IS NULL
CREATE TABLE dbo.statcast_pitch (
    id                 BIGINT IDENTITY(1,1) PRIMARY KEY,
    game_date          NVARCHAR(16) NOT NULL,           -- YYYY-MM-DD (official date)
    pitcher            NVARCHAR(16),                     -- MLBAM id
    batter             NVARCHAR(16),                     -- MLBAM id
    p_throws           NVARCHAR(2),                      -- L|R
    batting_team       NVARCHAR(8),                      -- Savant team abbr
    stand              NVARCHAR(2),                      -- batter side L|R
    xwoba              FLOAT,                             -- xwOBAcon (batted balls)
    xba                FLOAT,                             -- expected BA (per official AB)
    description        NVARCHAR(40),                     -- pitch outcome (whiff/CSW)
    [type]             NVARCHAR(2),                      -- S|B|X (X = batted ball)
    launch_speed       FLOAT,
    launch_speed_angle INT,                              -- 1..6 (6 = barrel)
    launch_angle       FLOAT,
    bb_type            NVARCHAR(20)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_statcast_pitch_date'
                 AND object_id = OBJECT_ID('dbo.statcast_pitch'))
CREATE INDEX ix_statcast_pitch_date ON dbo.statcast_pitch (game_date);
GO
-- Covering index for the warehouse team-offense aggregate (mlb_starters.
-- _warehouse_team_factors, ODI_MLB_WAREHOUSE_OFFENSE): GROUP BY batting_team, p_throws
-- over a game_date range with AVG/COUNT(xwoba). Group keys lead (stream aggregate, no
-- sort), game_date last (range seek per group), xwoba covered (index-only scan). Turns
-- the per-date backtest offense query from a growing base-table scan into an indexed
-- lookup. Add WITH (ONLINE = ON) on tiers that support it to avoid blocking the hourly
-- statcast maintenance during the build.
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_statcast_pitch_offense'
                 AND object_id = OBJECT_ID('dbo.statcast_pitch'))
CREATE INDEX ix_statcast_pitch_offense
    ON dbo.statcast_pitch (batting_team, p_throws, game_date) INCLUDE (xwoba);
GO
-- Per-PITCHER as-of query (pitcher_asof._asof_xwobacon_sql, the get_or_fill lazy-fill
-- path): WHERE pitcher=? AND game_date range with AVG/COUNT(xwoba). No pitcher-leading
-- index existed, so each get_or_fill MISS scanned the whole 3.1M-row table -> the
-- additive-backtest hang. Add WITH (ONLINE = ON) on tiers that support it.
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_statcast_pitch_pitcher'
                 AND object_id = OBJECT_ID('dbo.statcast_pitch'))
CREATE INDEX ix_statcast_pitch_pitcher
    ON dbo.statcast_pitch (pitcher, game_date) INCLUDE (xwoba);
GO

------------------------------------------------------------------- statcast_day
-- Per-day INGEST MANIFEST for statcast_pitch. One row per fetched game_date with
-- its row count — so an ingested-but-EMPTY offseason day (n_rows=0) is
-- distinguishable from a NOT-yet-fetched day. savant_history.missing_days /
-- ensure_days drive incremental gap-fill off this table (a bare game_date query
-- against statcast_pitch could not tell "empty" from "unfetched", re-pulling
-- empty days forever).
IF OBJECT_ID('dbo.statcast_day', 'U') IS NULL
CREATE TABLE dbo.statcast_day (
    game_date  NVARCHAR(16) NOT NULL PRIMARY KEY,        -- YYYY-MM-DD
    n_rows     INT NOT NULL,                             -- pitches ingested (0 = empty)
    fetched_at NVARCHAR(32)                              -- ISO-8601 UTC of the ingest
);
GO

-------------------------------------------------------------- pitcher_asof_daily
-- Durable per-(pitcher, game-date) as-of PITCHER feature store — the FULL per-DATE
-- as-of curve statcast_player_asof (a single season-to-date snapshot) does NOT hold.
-- One row per (entity_id, as_of_date, role); features are the cumulative line
-- STRICTLY BEFORE as_of_date (leakage-safe). role='SP' -> entity_id = pitcher MLBAM
-- id (one row per pitcher per game-date); role='RP' -> entity_id = team_id (team
-- relief aggregate; follow-up). Unifies the expected-runs pitcher signal across the
-- fitter / backtest / live onto ONE source (fit==serve==grade) and removes the
-- per-run in-memory load of the ~3M-row statcast_pitch corpus. RAW features only —
-- the fitted runs/9 ("xERA-lite") map is applied in code, so re-fitting never
-- rebuilds this table. Populated by pitcher_asof.build_season (bulk backfill) +
-- pitcher_asof.get_or_fill (lazy read-through; no cron). v1 columns are filled now;
-- the k_pct/bb_pct/n_bf columns need the mlb_pitcher_game BB/BF/HR unlock (v2) and
-- stay NULL until then. Mirrors pitcher_asof.py (SchemaParityTests enforces cols).
IF OBJECT_ID('dbo.pitcher_asof_daily', 'U') IS NULL
CREATE TABLE dbo.pitcher_asof_daily (
    id             INT IDENTITY(1,1) PRIMARY KEY,
    entity_id      NVARCHAR(32) NOT NULL,               -- MLBAM pitcher id (SP) | team_id (RP)
    as_of_date     NVARCHAR(16) NOT NULL,               -- the game date; features are < this
    role           NVARCHAR(4)  NOT NULL,               -- SP | RP
    season_bucket  INT,                                 -- derived; indexed, not in key
    -- warehouse mlb_pitcher_game as-of (v1)
    era            FLOAT,                               -- earned runs / 9, as-of
    ip             FLOAT,                               -- innings pitched, as-of (cumulative)
    avg_ip         FLOAT,                               -- ip / games (starter workload)
    k9             FLOAT,                               -- K / 9, as-of
    games          INT,                                 -- games behind the line
    -- statcast_pitch as-of contact quality (v1: xwobacon; rest nullable)
    xwobacon       FLOAT,                               -- mean xwOBAcon allowed (per BBE)
    n_bbe          INT,                                 -- batted balls behind xwobacon
    whiff_pct      FLOAT,
    csw_pct        FLOAT,
    barrel_pct     FLOAT,
    hard_hit_pct   FLOAT,
    gb_pct         FLOAT,
    n_pitches      INT,
    -- needs the mlb_pitcher_game BB/BF/HR unlock (v2; NULL until then)
    k_pct          FLOAT,
    bb_pct         FLOAT,
    n_bf           INT,
    fetched_at     FLOAT,
    CONSTRAINT uq_pitcher_asof_daily UNIQUE (entity_id, as_of_date, role)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_pitcher_asof_key'
                 AND object_id = OBJECT_ID('dbo.pitcher_asof_daily'))
CREATE INDEX ix_pitcher_asof_key
    ON dbo.pitcher_asof_daily (entity_id, role, as_of_date);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_pitcher_asof_season'
                 AND object_id = OBJECT_ID('dbo.pitcher_asof_daily'))
CREATE INDEX ix_pitcher_asof_season
    ON dbo.pitcher_asof_daily (season_bucket, role);
GO

-- ═══════════════════════════════════════════════════════════════════════════
-- Smart Fantasy Baseball ID cross-maps (data-integrity backbone). Two authoritative
-- maps, refreshed from smartfantasybaseball.com, that finally LINK the id-spaces the
-- app already stores but never connected: the betting `player` NAME, the ESPN
-- athlete_id (gamelog tables), and the MLBAM id (statcast_player_asof / statsapi).
-- Mirrors player_id_map.py (test_player_id_map.py::SchemaParityTests enforces it).
-- Refreshed by `python player_id_map.py --refresh` (and lazily, TTL-gated, in-app).
-- Replace-all per map (fully rebuildable); ids stored as text like the other tables.
-- ═══════════════════════════════════════════════════════════════════════════

------------------------------------------------------------------- player_id_map
-- One row per SFBB player. Anchor = sfbb_id (IDPLAYER; always present, unlike the
-- occasionally-blank MLBID). mlb_id = MLBID = the MLBAM id used by BOTH statsapi
-- and Statcast, so one numeric key covers the whole MLB player pipeline; espn_id
-- bridges to the ESPN gamelog tables. name_norm = normalize_name(PLAYERNAME).
IF OBJECT_ID('dbo.player_id_map', 'U') IS NULL
CREATE TABLE dbo.player_id_map (
    id           INT IDENTITY(1,1) PRIMARY KEY,
    sfbb_id      NVARCHAR(32)  NOT NULL,          -- IDPLAYER (SFBB stable key)
    mlb_id       NVARCHAR(32),                    -- MLBID = MLBAM (statsapi/Statcast)
    espn_id      NVARCHAR(32),                    -- ESPNID
    bref_id      NVARCHAR(32),                    -- BREFID
    fangraphs_id NVARCHAR(32),                    -- IDFANGRAPHS
    name         NVARCHAR(160),                   -- PLAYERNAME
    name_norm    NVARCHAR(160) NOT NULL,          -- normalize_name(PLAYERNAME)
    team         NVARCHAR(16),                    -- TEAM (SFBB code)
    pos          NVARCHAR(16),                    -- POS
    allpos       NVARCHAR(64),                    -- ALLPOS
    bats         NVARCHAR(8),
    throws       NVARCHAR(8),
    dk_name      NVARCHAR(160),                   -- DRAFTKINGSNAME (DK-only bettor)
    active       BIT,
    source       NVARCHAR(64),
    fetched_at   FLOAT,                           -- epoch seconds
    CONSTRAINT uq_player_id_map UNIQUE (sfbb_id)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_player_id_map_mlb'
                 AND object_id = OBJECT_ID('dbo.player_id_map'))
CREATE INDEX ix_player_id_map_mlb ON dbo.player_id_map (mlb_id);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_player_id_map_espn'
                 AND object_id = OBJECT_ID('dbo.player_id_map'))
CREATE INDEX ix_player_id_map_espn ON dbo.player_id_map (espn_id);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_player_id_map_name'
                 AND object_id = OBJECT_ID('dbo.player_id_map'))
CREATE INDEX ix_player_id_map_name ON dbo.player_id_map (name_norm);
GO

--------------------------------------------------------------------- team_id_map
-- 30 clubs; per-source abbreviations/nicknames (no numeric ids). Canonical key =
-- the stable 3-letter SFBBTEAM code. nickname (FANGRAPHSTEAM) is DISPLAY-ONLY and
-- can be stale (shows "Indians" for CLE) → canonicalize off the abbr columns, never
-- the nickname; the app layer curates full-name aliases (incl. a CLE→Guardians fix).
IF OBJECT_ID('dbo.team_id_map', 'U') IS NULL
CREATE TABLE dbo.team_id_map (
    id             INT IDENTITY(1,1) PRIMARY KEY,
    sfbb_code      NVARCHAR(16) NOT NULL,         -- SFBBTEAM (canonical)
    dk_code        NVARCHAR(16),                  -- DKTEAM
    espn_code      NVARCHAR(16),                  -- ESPNTEAM
    bbref_code     NVARCHAR(16),                  -- BBREFTEAM
    fangraphs_abbr NVARCHAR(16),                  -- FANGRAPHSABBR
    retrosheet     NVARCHAR(16),                  -- RETROSHEET
    nickname       NVARCHAR(64),                  -- FANGRAPHSTEAM (display; may be stale)
    name_norm      NVARCHAR(64) NOT NULL,         -- normalize_name(nickname)
    source         NVARCHAR(64),
    fetched_at     FLOAT,
    CONSTRAINT uq_team_id_map UNIQUE (sfbb_code)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_team_id_map_name'
                 AND object_id = OBJECT_ID('dbo.team_id_map'))
CREATE INDEX ix_team_id_map_name ON dbo.team_id_map (name_norm);
GO

--------------------------------------------------------------------- id_map_meta
-- TTL gate (mirror gamelog_fetch_meta): one row per map, last successful refresh +
-- row count. Lets the lazy in-app refresh serve a stale in-memory index when the
-- source is briefly unavailable rather than blocking.
IF OBJECT_ID('dbo.id_map_meta', 'U') IS NULL
CREATE TABLE dbo.id_map_meta (
    id              INT IDENTITY(1,1) PRIMARY KEY,
    map_name        NVARCHAR(16) NOT NULL,        -- 'player' | 'team'
    last_fetched_at FLOAT,                         -- epoch seconds
    row_count       INT,
    CONSTRAINT uq_id_map_meta UNIQUE (map_name)
);
GO

--------------------------------------------------------------- bankroll_ledger
-- One signed transaction per row; the current bankroll is the SUM of all amounts
-- (the balance is never stored -> a re-graded wager can't leave a stale running
-- total behind). Two kinds: 'bet' (one per settled wager, amount = its realized
-- profit, txn_id = 'bet:<wager_id>') and 'adjustment' (a manual deposit/withdrawal/
-- correction, amount = the signed delta the user's typed target implies).
IF OBJECT_ID('dbo.bankroll_ledger', 'U') IS NULL
CREATE TABLE dbo.bankroll_ledger (
    id          INT IDENTITY(1,1) PRIMARY KEY,
    txn_id      NVARCHAR(80) NOT NULL,           -- 'bet:<wager_id>' | 'adj:<iso>#<n>'
    txn_type    NVARCHAR(16),                    -- bet | adjustment
    amount      FLOAT,                           -- signed dollars
    wager_id    NVARCHAR(64),                    -- set for 'bet' txns
    note        NVARCHAR(256),
    created_at  NVARCHAR(40),
    CONSTRAINT uq_bankroll_txn UNIQUE (txn_id)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_bankroll_txn_type'
                 AND object_id = OBJECT_ID('dbo.bankroll_ledger'))
CREATE INDEX ix_bankroll_txn_type ON dbo.bankroll_ledger (txn_type);
GO

------------------------------------------------------------------ app_settings
-- Durable per-user app settings: a generic key/value store. Currently the Kelly
-- sizing knobs (fraction / per-bet cap % / slate-total cap %) so they persist
-- across sessions, not just page switches.
IF OBJECT_ID('dbo.app_settings', 'U') IS NULL
CREATE TABLE dbo.app_settings (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    setting_key   NVARCHAR(64) NOT NULL,
    setting_value NVARCHAR(256),
    updated_at    NVARCHAR(40),
    CONSTRAINT uq_app_setting_key UNIQUE (setting_key)
);
GO

-- ═══════════════════════════════════════════════════════════════════════════
-- MLB → StatsAPI medallion (P1). BRONZE transient raw-JSON landing + SILVER
-- durable dims (team/game/player) + a standings fact + a provider→MLBAM alias
-- scaffold. Populated read-only alongside the live ESPN path (dual-run); nothing
-- consumes it until the P4 cutover. Mirrors mlb_warehouse.py's SQLAlchemy Core
-- metadata (test_mlb_warehouse.py::SchemaParityTests enforces column parity).
-- Natural keys from StatsAPI are the PKs for the dims (team_id/game_pk/player_id);
-- the games dim is the spine (home/away FK → team dim). MLB only.
-- Order matters: mlb_team is created BEFORE mlb_game / mlb_team_standings so the
-- FKs resolve. game_date is the FULL ISO timestamp (its time disambiguates split
-- doubleheaders); official_date is the YYYY-MM-DD play date used for joins.
-- ═══════════════════════════════════════════════════════════════════════════

------------------------------------------------------------------------ mlb_bronze
-- Transient raw-response landing (one live payload per natural ref). A boxscore
-- row is purged once its game is genuine-final AND the silver it feeds is written;
-- schedule/standings rows are overwritten on next fetch (UNIQUE(kind,natural_ref)).
-- Never read by the app — only by the silver-builder. payload = raw JSON text.
IF OBJECT_ID('dbo.mlb_bronze', 'U') IS NULL
CREATE TABLE dbo.mlb_bronze (
    id           INT IDENTITY(1,1) PRIMARY KEY,
    kind         NVARCHAR(16)  NOT NULL,          -- schedule|boxscore|standings|teams
    natural_ref  NVARCHAR(64)  NOT NULL,          -- gamePk | YYYY-MM-DD | season
    payload      NVARCHAR(MAX) NOT NULL,          -- raw JSON
    fetched_at   FLOAT,                            -- epoch seconds
    processed_at FLOAT,                            -- set when dims/facts written; NULL=pending
    CONSTRAINT uq_mlb_bronze UNIQUE (kind, natural_ref)
);
GO

-------------------------------------------------------------------------- mlb_team
-- Team dim keyed on the MLBAM team_id (natural PK). league_id/division_id from
-- /teams; name_norm = normalize_name(name) for tolerant joins.
IF OBJECT_ID('dbo.mlb_team', 'U') IS NULL
CREATE TABLE dbo.mlb_team (
    team_id      NVARCHAR(32)  NOT NULL PRIMARY KEY,   -- MLBAM
    name         NVARCHAR(160),
    name_norm    NVARCHAR(160),
    abbreviation NVARCHAR(16),
    league_id    NVARCHAR(16),
    division_id  NVARCHAR(16),
    fetched_at   FLOAT
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_team_name'
                 AND object_id = OBJECT_ID('dbo.mlb_team'))
CREATE INDEX ix_mlb_team_name ON dbo.mlb_team (name_norm);
GO

-------------------------------------------------------------------------- mlb_game
-- Games dim = the spine. game_pk (MLBAM, globally unique across all MLB history)
-- is the natural PK; home/away FK → mlb_team. Facts reference this dim to derive
-- opponent/is_home/game_date rather than denormalizing them.
IF OBJECT_ID('dbo.mlb_game', 'U') IS NULL
CREATE TABLE dbo.mlb_game (
    game_pk        INT           NOT NULL PRIMARY KEY,  -- MLBAM (supplied, not IDENTITY)
    game_date      NVARCHAR(40),                        -- FULL ISO timestamp (UTC)
    official_date  NVARCHAR(10),                        -- YYYY-MM-DD play date
    season         INT,
    game_type      NVARCHAR(4),                         -- R|S|A|E|D|F|L|W|P (StatsAPI gameType)
    game_number    INT,                                 -- doubleheader game #
    double_header  NVARCHAR(4),                         -- N|Y (traditional)|S (split)
    home_team_id   NVARCHAR(32),
    away_team_id   NVARCHAR(32),
    venue_id       NVARCHAR(16),
    status         NVARCHAR(32),                         -- abstractGameState
    detailed_state NVARCHAR(64),                         -- detailedState
    home_score     FLOAT,
    away_score     FLOAT,
    fetched_at     FLOAT,
    CONSTRAINT fk_mlb_game_home
        FOREIGN KEY (home_team_id) REFERENCES dbo.mlb_team (team_id),
    CONSTRAINT fk_mlb_game_away
        FOREIGN KEY (away_team_id) REFERENCES dbo.mlb_team (team_id)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_game_official_date'
                 AND object_id = OBJECT_ID('dbo.mlb_game'))
CREATE INDEX ix_mlb_game_official_date ON dbo.mlb_game (official_date);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_game_teams'
                 AND object_id = OBJECT_ID('dbo.mlb_game'))
CREATE INDEX ix_mlb_game_teams
    ON dbo.mlb_game (official_date, home_team_id, away_team_id);  -- retro-match
GO
-- (season, status) COVERING scan (Azure missing-index rec): season-final-games reads
-- with scores/teams/dates. away_score moved from Azure's key into the INCLUDE (scores
-- aren't filter predicates). Add WITH (ONLINE = ON) on tiers that support it.
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_game_season_status'
                 AND object_id = OBJECT_ID('dbo.mlb_game'))
CREATE INDEX ix_mlb_game_season_status
    ON dbo.mlb_game (season, status)
    INCLUDE (away_score, home_score, away_team_id, home_team_id, detailed_state,
             game_date, game_type, official_date);
GO
-- Guarded ALTER for the existing prod mlb_game (added after the P1 create): the
-- StatsAPI schedule gameType (R=regular, A=all-star, S=spring, P/D/F/L/W=postseason).
-- Captured faithfully at silver; the P4 gold view uses it to exclude exhibitions.
IF COL_LENGTH('dbo.mlb_game', 'game_type') IS NULL
    ALTER TABLE dbo.mlb_game ADD game_type NVARCHAR(4);
GO
-- Guarded ALTER (Batch A #4 umpire): home-plate umpire parsed from the boxscore
-- `officials` block during ingest; joins to umpire_asof by hp_umpire_id. NULL on
-- schedule-only / pre-capture rows (umpire run_env fails open to neutral).
IF COL_LENGTH('dbo.mlb_game', 'hp_umpire_id') IS NULL
    ALTER TABLE dbo.mlb_game ADD hp_umpire_id NVARCHAR(32);
GO
IF COL_LENGTH('dbo.mlb_game', 'hp_umpire_name') IS NULL
    ALTER TABLE dbo.mlb_game ADD hp_umpire_name NVARCHAR(160);
GO

------------------------------------------------------------------------- mlb_venue
-- Venue dimension (Batch A run_env). venue_id is the physical park (already on
-- mlb_game, from the schedule); keying park + weather on venue_id — not the home
-- team name — is correct for neutral-site games, relocations, and the A's/Rays venue
-- limbo. Populated by mlb_warehouse.build_venue_dim (StatsAPI /venues hydrate=location
-- + the authored park_factors / weather_factors priors). park_hits/park_runs default
-- 1.0 (neutral) so an unmatched venue leaves run_env neutral.
IF OBJECT_ID('dbo.mlb_venue', 'U') IS NULL
CREATE TABLE dbo.mlb_venue (
    venue_id     NVARCHAR(16) NOT NULL PRIMARY KEY,  -- StatsAPI venue id
    name         NVARCHAR(160),
    team_id      NVARCHAR(32),                        -- canonical home team (NULL = neutral)
    team_name    NVARCHAR(160),                       -- for park/geo prior resolution
    lat          FLOAT,
    lon          FLOAT,
    cf_bearing   FLOAT,                               -- home plate -> CF compass degrees
    park_hits    FLOAT,                               -- authored prior (1.0 = neutral)
    park_runs    FLOAT,
    elevation_ft FLOAT,
    roof         NVARCHAR(16),                        -- open|retractable|dome
    fetched_at   FLOAT
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_venue_team'
                 AND object_id = OBJECT_ID('dbo.mlb_venue'))
CREATE INDEX ix_mlb_venue_team ON dbo.mlb_venue (team_id);
GO

---------------------------------------------------------------------- weather_game
-- Per-game weather (Batch A run_env), keyed by (venue_id, weather_date) so it joins
-- mlb_game on venue_id AND official_date = weather_date. First-pitch-hour conditions
-- from Visual Crossing (mlb_warehouse.build_weather). Used baseline-relative in the
-- weather run_env term. Actual weather is a pre-outcome game condition (not leakage).
IF OBJECT_ID('dbo.weather_game', 'U') IS NULL
CREATE TABLE dbo.weather_game (
    venue_id        NVARCHAR(16) NOT NULL,
    weather_date    NVARCHAR(10) NOT NULL,               -- YYYY-MM-DD (official_date)
    temp_f          FLOAT,
    humidity        FLOAT,                               -- relative %, 0-100
    pressure_mb     FLOAT,                               -- sea-level
    wind_mph        FLOAT,
    wind_dir_deg    FLOAT,                               -- direction wind blows FROM
    first_pitch_utc NVARCHAR(40),                        -- game hour sampled (audit)
    source          NVARCHAR(24),
    fetched_at      FLOAT,
    CONSTRAINT pk_weather_game PRIMARY KEY (venue_id, weather_date)
);
GO

------------------------------------------------------------------------ mlb_player
-- Player dim keyed on the MLBAM player_id (natural PK). bats/throws are nullable
-- in P1 (boxscore rosters give name/position/is_pitcher; handedness backfills from
-- /people later). name_norm for tolerant joins.
IF OBJECT_ID('dbo.mlb_player', 'U') IS NULL
CREATE TABLE dbo.mlb_player (
    player_id        NVARCHAR(32) NOT NULL PRIMARY KEY,  -- MLBAM
    full_name        NVARCHAR(160),
    name_norm        NVARCHAR(160),
    primary_position NVARCHAR(16),
    is_pitcher       BIT,
    bats             NVARCHAR(8),
    throws           NVARCHAR(8),
    fetched_at       FLOAT
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_player_name'
                 AND object_id = OBJECT_ID('dbo.mlb_player'))
CREATE INDEX ix_mlb_player_name ON dbo.mlb_player (name_norm);
GO

----------------------------------------------------------------- mlb_team_standings
-- Team win%/record snapshot fact (team × season × as-of). Feeds team markets +
-- opponent-strength; replaces the ESPN /standings merge for MLB.
IF OBJECT_ID('dbo.mlb_team_standings', 'U') IS NULL
CREATE TABLE dbo.mlb_team_standings (
    id          INT IDENTITY(1,1) PRIMARY KEY,
    team_id     NVARCHAR(32) NOT NULL,
    season      INT          NOT NULL,
    as_of_date  NVARCHAR(16) NOT NULL,               -- YYYY-MM-DD snapshot cutoff
    wins        INT,
    losses      INT,
    win_pct     FLOAT,
    fetched_at  FLOAT,
    CONSTRAINT uq_mlb_team_standings UNIQUE (team_id, season, as_of_date),
    CONSTRAINT fk_mlb_standings_team
        FOREIGN KEY (team_id) REFERENCES dbo.mlb_team (team_id)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_team_standings_asof'
                 AND object_id = OBJECT_ID('dbo.mlb_team_standings'))
CREATE INDEX ix_mlb_team_standings_asof
    ON dbo.mlb_team_standings (season, as_of_date);
GO
-- Additive: cumulative season runs (StatsAPI runsScored/runsAllowed) — the
-- StatsAPI-native team-defense input + a run-differential/Pythagorean signal.
-- Existing snapshots stay NULL until the next standings ingest re-populates them.
IF COL_LENGTH('dbo.mlb_team_standings', 'runs_scored') IS NULL
    ALTER TABLE dbo.mlb_team_standings ADD runs_scored INT;
GO
IF COL_LENGTH('dbo.mlb_team_standings', 'runs_allowed') IS NULL
    ALTER TABLE dbo.mlb_team_standings ADD runs_allowed INT;
GO

----------------------------------------------------------------------- player_alias
-- Provider NAME/id → MLBAM resolution store (the "associations" stored once).
-- Seeded from player_id_map (SFBB) + grown at runtime by the P3 entity resolver.
-- confidence/resolution_method/validity per the identity-resolution spec.
-- mlb_player_id is a value (validated at write time), NOT an enforced FK, so an
-- alias can be recorded before its player dim row lands.
IF OBJECT_ID('dbo.player_alias', 'U') IS NULL
CREATE TABLE dbo.player_alias (
    id                INT IDENTITY(1,1) PRIMARY KEY,
    provider          NVARCHAR(32)  NOT NULL,         -- e.g. 'oddsapi'|'sfbb'
    provider_key      NVARCHAR(200) NOT NULL,         -- provider name or id
    mlb_player_id     NVARCHAR(32)  NOT NULL,         -- MLBAM
    confidence        FLOAT,
    resolution_method NVARCHAR(32),                   -- alias|roster_exact|fuzzy_single|seed
    valid_from        NVARCHAR(40),
    valid_to          NVARCHAR(40),
    fetched_at        FLOAT,
    CONSTRAINT uq_player_alias UNIQUE (provider, provider_key)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_player_alias_mlb'
                 AND object_id = OBJECT_ID('dbo.player_alias'))
CREATE INDEX ix_player_alias_mlb ON dbo.player_alias (mlb_player_id);
GO

-- ═══════════════════════════════════════════════════════════════════════════
-- MLB → StatsAPI medallion (P2). Game-centric per-game batter/pitcher stat FACTS,
-- derived straight from the boxscore so athlete_id is the MLBAM id and game_pk
-- comes from the games dim. game_pk is ALWAYS present → the natural key is a plain
-- UNIQUE(athlete_id, game_pk) (no NULL / filtered-index dance). team_id is NOT
-- NULL but NOT in the key (attribute FD on (athlete, game)); season_bucket is
-- derived (FD on the game) → a plain indexed attribute, NOT in the key. The
-- denormalized game_date/opponent/is_home are NOT stored — the reader/gold view
-- rejoins fact→mlb_game→mlb_team. These live ALONGSIDE the ESPN-sourced
-- mlb_batter_gamelog/mlb_pitcher_gamelog (gamelog_store.py, untouched); nothing
-- app-facing consumes them until the P4 cutover. Constraint/index names mirror
-- mlb_warehouse.py (test_mlb_warehouse.py::SchemaParityTests enforces columns).
-- ═══════════════════════════════════════════════════════════════════════════

--------------------------------------------------------------------- mlb_batter_game
IF OBJECT_ID('dbo.mlb_batter_game', 'U') IS NULL
CREATE TABLE dbo.mlb_batter_game (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    athlete_id    NVARCHAR(32) NOT NULL,               -- MLBAM (natural key part)
    game_pk       INT          NOT NULL,               -- games dim (natural key part)
    team_id       NVARCHAR(32) NOT NULL,               -- MLBAM (attribute, NOT in key)
    season_bucket INT,                                  -- derived; indexed, NOT in key
    AB            FLOAT,
    H             FLOAT,
    SO            FLOAT,
    BB            FLOAT,
    HBP           FLOAT,
    SF            FLOAT,
    SH            FLOAT,
    HR            FLOAT,
    TB            FLOAT,
    RBI           FLOAT,
    fetched_at    FLOAT,
    CONSTRAINT uq_mlb_batter_game UNIQUE (athlete_id, game_pk),
    CONSTRAINT fk_mlb_batter_game_game
        FOREIGN KEY (game_pk) REFERENCES dbo.mlb_game (game_pk),
    CONSTRAINT fk_mlb_batter_game_team
        FOREIGN KEY (team_id) REFERENCES dbo.mlb_team (team_id)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_batter_game_athlete'
                 AND object_id = OBJECT_ID('dbo.mlb_batter_game'))
CREATE INDEX ix_mlb_batter_game_athlete
    ON dbo.mlb_batter_game (athlete_id, season_bucket);
GO
-- game_pk-first COVERING index (Azure missing-index rec): the uq is (athlete_id,
-- game_pk) so it can't seek game_pk-first; game_pk grading/as-of reads scan without it.
-- Covers all batter stat cols -> index-only. Add WITH (ONLINE = ON) where supported.
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_batter_game_gamepk'
                 AND object_id = OBJECT_ID('dbo.mlb_batter_game'))
CREATE INDEX ix_mlb_batter_game_gamepk
    ON dbo.mlb_batter_game (game_pk, season_bucket)
    INCLUDE (athlete_id, AB, H, SO, BB, HBP, SF, SH, HR, TB, RBI);
GO
-- Additive: HR/TB/RBI (StatsAPI homeRuns/totalBases/rbi). TB/RBI are fact-servable
-- (in _ACTUAL_STAT_SPEC) for the batter_total_bases / batter_rbis props; HR awaits an
-- odds market. Rows ingested before this landed have them NULL — get_player_history
-- SKIPS a NULL-stat game (never reads NULL-as-0), so a re-backfill widens coverage but
-- isn't a correctness prerequisite — see mlb_warehouse.py.
IF COL_LENGTH('dbo.mlb_batter_game', 'HR') IS NULL
    ALTER TABLE dbo.mlb_batter_game ADD HR FLOAT;
GO
IF COL_LENGTH('dbo.mlb_batter_game', 'TB') IS NULL
    ALTER TABLE dbo.mlb_batter_game ADD TB FLOAT;
GO
IF COL_LENGTH('dbo.mlb_batter_game', 'RBI') IS NULL
    ALTER TABLE dbo.mlb_batter_game ADD RBI FLOAT;
GO

-------------------------------------------------------------------- mlb_pitcher_game
IF OBJECT_ID('dbo.mlb_pitcher_game', 'U') IS NULL
CREATE TABLE dbo.mlb_pitcher_game (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    athlete_id    NVARCHAR(32) NOT NULL,               -- MLBAM (natural key part)
    game_pk       INT          NOT NULL,               -- games dim (natural key part)
    team_id       NVARCHAR(32) NOT NULL,               -- MLBAM (attribute, NOT in key)
    season_bucket INT,                                  -- derived; indexed, NOT in key
    IP            FLOAT,                                 -- base-3 float (6.1 == 6IP+1out)
    K             FLOAT,
    ER            FLOAT,
    BB            FLOAT,                                 -- Tier A #1c (walks)
    BF            FLOAT,                                 -- batters faced (K%/BB% denom)
    HR            FLOAT,                                 -- HR allowed (FIP)
    HBP           FLOAT,                                 -- HBP allowed (FIP)
    GS            FLOAT,                                 -- games started (1) / relief (0)
    fetched_at    FLOAT,
    CONSTRAINT uq_mlb_pitcher_game UNIQUE (athlete_id, game_pk),
    CONSTRAINT fk_mlb_pitcher_game_game
        FOREIGN KEY (game_pk) REFERENCES dbo.mlb_game (game_pk),
    CONSTRAINT fk_mlb_pitcher_game_team
        FOREIGN KEY (team_id) REFERENCES dbo.mlb_team (team_id)
);
GO
-- Tier A #1c UNLOCK for an EXISTING table (idempotent). ⚠ RUN THIS BEFORE deploying
-- the code that adds BB/BF/HR/HBP/GS to _PITCHER_GAME_STATS — _game_log SELECTs
-- *stat_cols, so the code expects these columns to exist. After the ALTER, re-derive
-- from the cached bronze boxscores to populate them (mlb_warehouse re-ingest).
IF COL_LENGTH('dbo.mlb_pitcher_game', 'BB') IS NULL
    ALTER TABLE dbo.mlb_pitcher_game
        ADD BB FLOAT, BF FLOAT, HR FLOAT, HBP FLOAT, GS FLOAT;
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_pitcher_game_athlete'
                 AND object_id = OBJECT_ID('dbo.mlb_pitcher_game'))
CREATE INDEX ix_mlb_pitcher_game_athlete
    ON dbo.mlb_pitcher_game (athlete_id, season_bucket);
GO
-- game_pk-first COVERING index (Azure missing-index rec): the uq is (athlete_id,
-- game_pk) so it can't seek game_pk-first; game_pk grading/as-of reads scan without it.
-- Covers all pitcher stat cols -> index-only. Add WITH (ONLINE = ON) where supported.
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_mlb_pitcher_game_gamepk'
                 AND object_id = OBJECT_ID('dbo.mlb_pitcher_game'))
CREATE INDEX ix_mlb_pitcher_game_gamepk
    ON dbo.mlb_pitcher_game (game_pk, season_bucket)
    INCLUDE (athlete_id, IP, K, ER, BB, BF, HR, HBP, GS);
GO
