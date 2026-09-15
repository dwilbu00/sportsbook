-- parlay_tracker.sql — owner-run migration for the dedicated parlay tracker (Stage B follow-on).
-- Idempotent: safe to run more than once. Additive only (never drops existing data).
-- Adds two normalized tables (parlays + parlay_legs) and a nullable boost_pct on wagers.
-- Run against the live Azure DB, then the 💰 Parlays app section becomes usable.

------------------------------------------------------------------ parlays (ticket-level)
IF OBJECT_ID('dbo.parlays', 'U') IS NULL
CREATE TABLE dbo.parlays (
    id                  INT IDENTITY(1,1) PRIMARY KEY,
    parlay_id           NVARCHAR(64) NOT NULL,
    placed_at           NVARCHAR(40),
    sport_key           NVARCHAR(64),
    book                NVARCHAR(64),
    bonus_label         NVARCHAR(128),
    bet_type            NVARCHAR(16),          -- parlay | sgp
    boost_pct           FLOAT,
    is_same_game        BIT,
    n_legs              INT,
    combined_american   INT,
    stake               FLOAT,
    our_joint_prob      FLOAT,                 -- our estimate at placement
    our_boosted_ev_pct  FLOAT,                 -- at placement
    max_corr            FLOAT,                 -- SGP: strongest pairwise correlation
    status              NVARCHAR(16),          -- pending|won|lost|push|void
    settled_at          NVARCHAR(40),
    payout              FLOAT,
    profit              FLOAT,
    game_date           NVARCHAR(10),
    CONSTRAINT uq_parlay_id UNIQUE (parlay_id),
    CONSTRAINT ck_parlay_status CHECK (status IN ('pending','won','lost','push','void')),
    CONSTRAINT ck_parlay_stake CHECK (stake IS NULL OR stake >= 0)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_parlay_status' AND object_id = OBJECT_ID('dbo.parlays'))
CREATE INDEX ix_parlay_status ON dbo.parlays (status);
GO

--------------------------------------------------------------------- parlay_legs (per leg)
IF OBJECT_ID('dbo.parlay_legs', 'U') IS NULL
CREATE TABLE dbo.parlay_legs (
    id             INT IDENTITY(1,1) PRIMARY KEY,
    parlay_id      NVARCHAR(64) NOT NULL,
    leg_index      INT NOT NULL,
    sport_key      NVARCHAR(64),
    player         NVARCHAR(160),
    prop_key       NVARCHAR(64),
    line           FLOAT,
    side           NVARCHAR(8),                -- OVER | UNDER
    price          INT,
    our_leg_prob   FLOAT,                      -- market de-vig (calibrated)
    team           NVARCHAR(16),
    opp            NVARCHAR(16),
    corr_category  NVARCHAR(32),
    event_id       NVARCHAR(128),
    commence_time  NVARCHAR(40),
    game_date      NVARCHAR(10),
    actual         NVARCHAR(64),
    result         INT,                        -- 1 win, 0 loss, NULL pending/push
    CONSTRAINT uq_parlay_leg UNIQUE (parlay_id, leg_index)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_parlay_leg_parlay' AND object_id = OBJECT_ID('dbo.parlay_legs'))
CREATE INDEX ix_parlay_leg_parlay ON dbo.parlay_legs (parlay_id);
GO

------------------------------------------------- wagers.boost_pct (boosted STRAIGHT bets)
-- Store the RAW market price + this boost fraction so CLV/close comparisons stay honest;
-- grading pays the boosted payout: profit = stake*(dec-1)*(1+boost) on a win.
IF COL_LENGTH('dbo.wagers', 'boost_pct') IS NULL
    ALTER TABLE dbo.wagers ADD boost_pct FLOAT;
GO
