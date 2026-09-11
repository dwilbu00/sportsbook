/* ============================================================================
   nfl_ladder_separate.sql  —  move the NFL opener-ladder research rows OUT of the
   operational odds_snapshot/odds_line tables into dedicated *_research tables.

   WHY: the ~10M-row ladder backfill bloated odds_line, and at 20 DTU it now slows
   EVERY read (mirror sync, scoped queries all timing out). The ladder is a static
   research corpus queried differently from the live app's operational odds, so it
   belongs in its own table. This restores odds_line to its lean operational size.

   SAFETY (never-destroy-irreplaceable-data): the ladder's ONLY durable copy is here
   in Azure (local parquets are NOT durable). So this is strictly COPY -> VERIFY ->
   (only then) DELETE. DO NOT run STEP 3 until STEP 2's counts match exactly.

   RUN AS: owner/admin (StreamlitApp has no DDL). Run when the app is idle
   (cross-process writes aren't guarded). 20 DTU -> the delete is batched.
   ============================================================================ */

/* ---- STEP 1: COPY ladder rows into research tables (SELECT INTO copies data;
   heaps are fine for a research archive; original ids are preserved so
   odds_line_research.snapshot_id still matches odds_snapshot_research.id). ---- */

SELECT *
INTO   dbo.odds_snapshot_research
FROM   dbo.odds_snapshot
WHERE  sport = 'americanfootball_nfl' AND source LIKE 'ladder_%';

SELECT l.*
INTO   dbo.odds_line_research
FROM   dbo.odds_line l
JOIN   dbo.odds_snapshot s ON l.snapshot_id = s.id
WHERE  s.sport = 'americanfootball_nfl' AND s.source LIKE 'ladder_%';

/* helpful indexes for future research reads (SELECT INTO doesn't copy them) */
CREATE INDEX ix_osr_sport_source ON dbo.odds_snapshot_research (sport, source);
CREATE INDEX ix_osr_event        ON dbo.odds_snapshot_research (event_id);
CREATE INDEX ix_olr_snapshot     ON dbo.odds_line_research (snapshot_id);
CREATE INDEX ix_olr_prop         ON dbo.odds_line_research (prop_key);
GO

/* ---- STEP 2: VERIFY the copy is complete. The two pairs MUST match exactly.
   >>> STOP HERE. Only proceed to STEP 3 if main_snaps=research_snaps AND
       main_lines=research_lines. <<< ---- */

SELECT
  (SELECT COUNT(*) FROM dbo.odds_snapshot
     WHERE sport='americanfootball_nfl' AND source LIKE 'ladder_%')            AS main_snaps,
  (SELECT COUNT(*) FROM dbo.odds_snapshot_research)                            AS research_snaps,
  (SELECT COUNT(*) FROM dbo.odds_line l
     JOIN dbo.odds_snapshot s ON l.snapshot_id=s.id
     WHERE s.sport='americanfootball_nfl' AND s.source LIKE 'ladder_%')        AS main_lines,
  (SELECT COUNT(*) FROM dbo.odds_line_research)                               AS research_lines;
GO

/* ---- STEP 3: DELETE the ladder rows from the OPERATIONAL tables.
   ONLY run after STEP 2 verified. Batched (20 DTU + big log). Lines first
   (independent of whether the FK cascades), then the snapshots. ---- */

-- collect the ladder snapshot ids once (fast, ~tens of thousands of rows)
SELECT id INTO #ladder_ids
FROM   dbo.odds_snapshot
WHERE  sport='americanfootball_nfl' AND source LIKE 'ladder_%';
CREATE INDEX ix_li ON #ladder_ids (id);

-- batched line delete (5k/loop keeps the transaction log + DTU in check)
WHILE 1 = 1
BEGIN
    DELETE TOP (5000) l
    FROM   dbo.odds_line l
    JOIN   #ladder_ids t ON l.snapshot_id = t.id;
    IF @@ROWCOUNT = 0 BREAK;
END

-- then the snapshots (small; one statement is fine)
DELETE FROM dbo.odds_snapshot
WHERE  sport='americanfootball_nfl' AND source LIKE 'ladder_%';

DROP TABLE #ladder_ids;
GO

/* ---- STEP 4: POST-DELETE sanity. main_* should now be 0; research_* unchanged. ---- */

SELECT
  (SELECT COUNT(*) FROM dbo.odds_snapshot
     WHERE sport='americanfootball_nfl' AND source LIKE 'ladder_%')  AS main_snaps_should_be_0,
  (SELECT COUNT(*) FROM dbo.odds_snapshot_research)                  AS research_snaps_kept,
  (SELECT COUNT(*) FROM dbo.odds_line_research)                      AS research_lines_kept;
GO

/* Optional (reclaim space / refresh stats after the big delete, owner's discretion):
   ALTER INDEX ALL ON dbo.odds_line REBUILD;
   UPDATE STATISTICS dbo.odds_line;                                                  */
