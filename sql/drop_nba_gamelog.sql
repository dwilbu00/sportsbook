-- drop_nba_gamelog.sql — OPTIONAL owner-run cleanup after the ESPN-for-NBA retirement
-- (2026-10-16). dbo.nba_gamelog was a RE-FETCHABLE ESPN cache (replace-all per athlete/season;
-- never irreplaceable data) that nothing reads anymore — NBA history + grading come from the
-- stats.nba.com warehouse. Safe to leave in place; run this only to reclaim the space.
-- Run as owner/admin in Azure (the app login has no DDL).
IF OBJECT_ID('dbo.nba_gamelog', 'U') IS NOT NULL
    DROP TABLE dbo.nba_gamelog;
GO
-- bookkeeping rows for the dropped cache (NFL rows in these shared tables are untouched)
DELETE FROM dbo.gamelog_fetch_meta WHERE sport = 'basketball';
DELETE FROM dbo.athlete_id_cache WHERE sport = 'basketball';
GO
