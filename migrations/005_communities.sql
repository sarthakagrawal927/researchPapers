-- Community membership (Louvain) on the within-corpus citation subgraph.

ALTER TABLE papers ADD COLUMN IF NOT EXISTS community_id INTEGER;

CREATE INDEX IF NOT EXISTS papers_community_idx ON papers (community_id);
