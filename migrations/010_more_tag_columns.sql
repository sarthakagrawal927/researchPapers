-- Columns for additional tagger variants we're A/B'ing.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS web_lg_tags_json   JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS mlx_llm_tags_json  JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS mlx_llm_tldr       TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS web_lg_tagged_at   TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS mlx_llm_tagged_at  TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS papers_web_lg_idx ON papers (web_lg_tagged_at);
CREATE INDEX IF NOT EXISTS papers_mlx_idx    ON papers (mlx_llm_tagged_at);
