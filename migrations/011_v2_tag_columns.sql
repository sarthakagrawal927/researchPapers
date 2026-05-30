ALTER TABLE papers ADD COLUMN IF NOT EXISTS noun_tags_v2_json    JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS mlx_llm_v2_tags_json JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS mlx_llm_v2_tldr      TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS noun_v2_tagged_at    TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS mlx_llm_v2_tagged_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS papers_noun_v2_idx ON papers (noun_v2_tagged_at);
CREATE INDEX IF NOT EXISTS papers_mlx_v2_idx  ON papers (mlx_llm_v2_tagged_at);
