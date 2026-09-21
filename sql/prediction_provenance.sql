-- prediction_provenance.sql — F28 P1 (owner-run, idempotent).
--
-- Adds nullable PROVENANCE columns to dbo.prediction_log so a stored prediction can
-- be attributed and later replayed. All nullable: existing rows keep NULL = "unknown
-- provenance" (never rewritten). db_store.create_all() also creates these on a fresh
-- DB; this script brings an EXISTING Azure table up to date. Safe to run repeatedly.
--
-- Run once against the deployment's Azure SQL database (same as sql/parlay_tracker.sql).

IF COL_LENGTH('dbo.prediction_log', 'event_season') IS NULL
    ALTER TABLE dbo.prediction_log ADD event_season INT;

IF COL_LENGTH('dbo.prediction_log', 'event_week') IS NULL
    ALTER TABLE dbo.prediction_log ADD event_week INT;

IF COL_LENGTH('dbo.prediction_log', 'as_of') IS NULL
    ALTER TABLE dbo.prediction_log ADD as_of NVARCHAR(40);

IF COL_LENGTH('dbo.prediction_log', 'calibration_hash') IS NULL
    ALTER TABLE dbo.prediction_log ADD calibration_hash NVARCHAR(32);

IF COL_LENGTH('dbo.prediction_log', 'model_schema') IS NULL
    ALTER TABLE dbo.prediction_log ADD model_schema NVARCHAR(64);

IF COL_LENGTH('dbo.prediction_log', 'method') IS NULL
    ALTER TABLE dbo.prediction_log ADD method NVARCHAR(32);

IF COL_LENGTH('dbo.prediction_log', 'reference_book_policy') IS NULL
    ALTER TABLE dbo.prediction_log ADD reference_book_policy NVARCHAR(64);

IF COL_LENGTH('dbo.prediction_log', 'context_fingerprint') IS NULL
    ALTER TABLE dbo.prediction_log ADD context_fingerprint NVARCHAR(32);
