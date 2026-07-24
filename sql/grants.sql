-- Least-privilege CRUD grants for the app user (StreamlitApp).
--
-- Run MANUALLY as the server admin AFTER schema.sql. StreamlitApp must already
-- exist as a database user. On Azure SQL with SQL authentication a contained
-- user is created inside the database, e.g. (run once, in this database):
--
--   CREATE USER StreamlitApp WITH PASSWORD = '<strong-password>';
--
-- The app needs only SELECT/INSERT/UPDATE/DELETE — never DDL. Do NOT add it to
-- db_owner / db_ddladmin.

GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.prediction_log       TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.wagers               TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.recalibration_params TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.recalibration_folds  TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.recalibration_meta   TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.odds_snapshot        TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.odds_line            TO StreamlitApp;
-- Phase C: rolling ESPN gamelog store (replace-all refresh needs DELETE).
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.mlb_batter_gamelog   TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.mlb_pitcher_gamelog  TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.nba_gamelog          TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.gamelog_fetch_meta   TO StreamlitApp;
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.athlete_id_cache     TO StreamlitApp;
-- P2.4a: derived per-batter Statcast as-of rates (replace-all refresh needs DELETE).
GRANT SELECT, INSERT, UPDATE, DELETE ON dbo.statcast_player_asof TO StreamlitApp;
GO
