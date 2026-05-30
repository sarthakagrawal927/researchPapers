-- Alternative tag approaches for head-to-head eval vs LLM tags.

ALTER TABLE papers ADD COLUMN IF NOT EXISTS noun_tags_json    JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS keybert_tags_json JSONB;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS noun_tagged_at    TIMESTAMPTZ;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS keybert_tagged_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS papers_noun_tagged_idx    ON papers (noun_tagged_at);
CREATE INDEX IF NOT EXISTS papers_keybert_tagged_idx ON papers (keybert_tagged_at);
