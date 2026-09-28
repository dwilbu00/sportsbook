-- Widen app_settings.setting_value: NVARCHAR(256) -> NVARCHAR(MAX).
--
-- WHY: bonus_store.py persists the whole active-bonus list as ONE JSON string under
-- setting_key = 'active_bonuses'. At NVARCHAR(256) SQL Server rejected that write once
-- the JSON exceeded 256 chars ("String or binary data would be truncated"), so
-- save_bonuses() failed silently (returned 0) and the list reverted to seeds on every
-- reload. The Kelly knobs (tiny float strings) fit, which is why only bonuses broke.
--
-- SAFE / non-destructive: widening a column preserves every existing value (Kelly knobs
-- etc.) unchanged. Idempotent: only alters when the column is not already MAX.
-- Run once against the live Azure SQL database.

IF EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.app_settings')
      AND name = 'setting_value'
      AND max_length <> -1              -- -1 == NVARCHAR(MAX); a bounded col is 2*N bytes
)
BEGIN
    ALTER TABLE dbo.app_settings ALTER COLUMN setting_value NVARCHAR(MAX) NULL;
    PRINT 'app_settings.setting_value widened to NVARCHAR(MAX).';
END
ELSE
    PRINT 'app_settings.setting_value already NVARCHAR(MAX) — no change.';
