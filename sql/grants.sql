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
GO
