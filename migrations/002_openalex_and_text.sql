-- Adds OpenAlex-derived popularity signal + gzipped full text storage.

ALTER TABLE papers ADD COLUMN IF NOT EXISTS citation_count INTEGER;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS openalex_id    TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS doi            TEXT;

CREATE INDEX IF NOT EXISTS papers_citation_count_idx ON papers (citation_count DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS papers_openalex_id_idx    ON papers (openalex_id);

-- Gzipped UTF-8 PDF text. Kept in a side table so the hot `papers` table stays narrow.
CREATE TABLE IF NOT EXISTS paper_texts (
    arxiv_id      TEXT PRIMARY KEY REFERENCES papers (arxiv_id) ON DELETE CASCADE,
    content_gz    BYTEA NOT NULL,
    content_chars INTEGER NOT NULL,
    extracted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
