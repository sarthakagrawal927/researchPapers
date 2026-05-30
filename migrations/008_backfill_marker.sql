ALTER TABLE papers ADD COLUMN IF NOT EXISTS references_backfilled_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS papers_refs_backfilled_idx ON papers (references_backfilled_at);
