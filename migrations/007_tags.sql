ALTER TABLE papers ADD COLUMN IF NOT EXISTS tags_json JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS keywords_json JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS llm_tags_json JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS llm_tldr TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS llm_tagged_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS papers_tags_gin    ON papers USING GIN (tags_json);
CREATE INDEX IF NOT EXISTS papers_keywords_gin ON papers USING GIN (keywords_json);
CREATE INDEX IF NOT EXISTS papers_llm_tagged_idx ON papers (llm_tagged_at);
