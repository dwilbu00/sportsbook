-- ============================================================================
-- nfl_gamelog: disambiguate stat columns (2026-09-25)  [owner-run, Azure SQL]
--
-- WHY: the NFL gamelog cache stored only [YDS] + [TD]. On the Cloud (SQL on),
-- forward grading reads THIS cache (via gamelog_store.get_gamelog), so:
--   * COUNT props (receptions / rushingAttempts / passingAttempts / completions)
--     had no column -> resolve_one_prop returned None -> stuck pending forever;
--   * [YDS] is the collision-prone display label = a QB row's RUSHING yards, not
--     passingYards (Stafford: passingYards 327 but YDS -1) -> yardage/TD props
--     graded to WRONG outcomes (poisoned calibration).
-- nfl_gamelog is a RE-FETCHABLE CACHE of ESPN gamelogs (NOT irreplaceable data),
-- so we DROP + recreate with the unambiguous machine-name columns and clear its
-- fetch-gate; it re-populates from ESPN on next use. Then reset the affected NFL
-- prediction rows so maintenance re-grades them correctly.
--
-- Run AFTER deploying the code change (gamelog_store._NFL_STATS + schema.sql).
-- ============================================================================

-- 1) Rebuild the cache table with disambiguated columns -----------------------
IF OBJECT_ID('dbo.nfl_gamelog', 'U') IS NOT NULL
    DROP TABLE dbo.nfl_gamelog;
GO
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
    passingYards FLOAT, rushingYards FLOAT, receivingYards FLOAT,
    receptions FLOAT, rushingAttempts FLOAT, passingAttempts FLOAT, completions FLOAT,
    passingTouchdowns FLOAT, rushingTouchdowns FLOAT, receivingTouchdowns FLOAT
);
GO
CREATE INDEX ix_nfl_gamelog_athlete ON dbo.nfl_gamelog (athlete_id, season_bucket);
GO

-- 2) Clear the NFL cache-gate so get_gamelog re-fetches from ESPN -------------
DELETE FROM dbo.gamelog_fetch_meta WHERE sport = 'football';
GO

-- 3) Reset already-resolved NFL prop rows so they re-grade on the correct data.
--    (The 50 stuck count props are already resolved=0; this fixes the yardage/TD
--    rows that were graded off the bad [YDS]/[TD] columns. Forecasts are kept —
--    only the outcome fields are cleared for a clean re-grade.)
UPDATE dbo.prediction_log
   SET resolved = 0, actual = NULL, outcome = NULL, resolved_at = NULL
 WHERE sport_key = 'americanfootball_nfl' AND resolved = 1;
GO

-- After running this: trigger maintenance to re-grade (offline is fastest):
--   python forward_tracker.py --resolve --max-resolve 5000
-- or the app's "Resolve all pending predictions now" button.
