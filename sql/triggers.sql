-- OPTIONAL Azure-only triggers (robustness hardening). Run MANUALLY as admin.
--
-- The application is authoritative for all writes (it replaces a store's rows
-- inside a single transaction, and the full row is stored as JSON in `data`), so
-- NOTHING in the app depends on these triggers — they are pure DB-side extras.
-- They are intentionally kept out of db_store.py's metadata (which mirrors
-- schema.sql only) so the hermetic SQLite tests never diverge from prod.

-- ── Append-only audit of wager status changes ────────────────────────────────
-- Handy for after-the-fact debugging of grading ("when did this bet flip to
-- lost, and from what?"). Independent of the ledger itself.

IF OBJECT_ID('dbo.wager_status_audit', 'U') IS NULL
CREATE TABLE dbo.wager_status_audit (
    audit_id    INT IDENTITY(1,1) PRIMARY KEY,
    wager_id    NVARCHAR(64) NOT NULL,
    old_status  NVARCHAR(16),
    new_status  NVARCHAR(16),
    changed_at  DATETIME2 NOT NULL CONSTRAINT df_wager_audit_at DEFAULT (SYSUTCDATETIME())
);
GO

CREATE OR ALTER TRIGGER dbo.trg_wager_status_audit
ON dbo.wagers
AFTER UPDATE
AS
BEGIN
    SET NOCOUNT ON;
    INSERT INTO dbo.wager_status_audit (wager_id, old_status, new_status)
    SELECT i.wager_id, d.status, i.status
    FROM inserted i
    JOIN deleted  d ON d.id = i.id
    WHERE ISNULL(i.status, '') <> ISNULL(d.status, '');
END;
GO

-- If you enable the audit trigger, also grant the app user INSERT on the audit
-- table (the trigger runs in the caller's security context):
--   GRANT INSERT ON dbo.wager_status_audit TO StreamlitApp;
-- GO
