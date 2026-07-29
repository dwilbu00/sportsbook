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

GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA :: dbo TO StreamlitApp;