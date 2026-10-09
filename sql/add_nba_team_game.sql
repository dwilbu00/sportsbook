-- add_nba_team_game.sql — ADDITIVE: create dbo.nba_team_game on an EXISTING NBA warehouse
-- (2026-10-09, pace-adjusted defense / team model, phase 1). Safe to re-run (guarded); touches
-- no other table. Mirrors the nba_team_game block in sql/schema.sql (keep in sync).
-- Run as owner/admin in Azure (the app login has no DDL), then backfill:
--   python nba_warehouse.py --seasons 2021-2026 --team-games-only
IF OBJECT_ID('dbo.nba_team_game', 'U') IS NULL
CREATE TABLE dbo.nba_team_game (
    id                  INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    game_id             NVARCHAR(16) NOT NULL,
    team_id             BIGINT NOT NULL,
    season              INT NOT NULL,
    season_type         NVARCHAR(16),
    game_date           NVARCHAR(10),
    opponent_team_id    BIGINT,
    home_away           NVARCHAR(8),
    game_minutes        FLOAT,
    points FLOAT, field_goals_made FLOAT, field_goals_attempted FLOAT,
    three_point_field_goals_made FLOAT, three_point_field_goals_attempted FLOAT,
    free_throws_made FLOAT, free_throws_attempted FLOAT, offensive_rebounds FLOAT,
    defensive_rebounds FLOAT, rebounds FLOAT, assists FLOAT, steals FLOAT, blocks FLOAT,
    turnovers FLOAT, fouls FLOAT, plus_minus FLOAT,
    opponent_points     FLOAT,
    possessions         FLOAT,
    fetched_at          FLOAT,
    CONSTRAINT uq_nba_team_game UNIQUE (game_id, team_id)
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_nba_team_game_team' AND object_id = OBJECT_ID('dbo.nba_team_game'))
CREATE INDEX ix_nba_team_game_team ON dbo.nba_team_game (team_id, season);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes
               WHERE name = 'ix_nba_team_game_season' AND object_id = OBJECT_ID('dbo.nba_team_game'))
CREATE INDEX ix_nba_team_game_season ON dbo.nba_team_game (season, game_date);
GO
