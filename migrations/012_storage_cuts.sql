-- Drop columns and tables we agreed not to use.
-- Idempotent: each DROP uses IF EXISTS.

-- Experimental tag columns superseded by v2 / eval-rejected variants.
ALTER TABLE papers DROP COLUMN IF EXISTS keybert_tags_json;
ALTER TABLE papers DROP COLUMN IF EXISTS keybert_tagged_at;
ALTER TABLE papers DROP COLUMN IF EXISTS web_lg_tags_json;
ALTER TABLE papers DROP COLUMN IF EXISTS web_lg_tagged_at;
ALTER TABLE papers DROP COLUMN IF EXISTS noun_tags_json;        -- v1, replaced by noun_tags_v2_json
ALTER TABLE papers DROP COLUMN IF EXISTS noun_tagged_at;
ALTER TABLE papers DROP COLUMN IF EXISTS mlx_llm_tags_json;     -- v1, replaced by mlx_llm_v2_tags_json
ALTER TABLE papers DROP COLUMN IF EXISTS mlx_llm_tldr;
ALTER TABLE papers DROP COLUMN IF EXISTS mlx_llm_tagged_at;

-- Deprecated body-text columns from the PDF pipeline.
ALTER TABLE papers DROP COLUMN IF EXISTS pdf_url;
ALTER TABLE papers DROP COLUMN IF EXISTS pdf_path;
ALTER TABLE papers DROP COLUMN IF EXISTS pdf_fetched_at;
ALTER TABLE papers DROP COLUMN IF EXISTS urls_extracted_at;

-- Tables we agreed not to use anymore.
DROP TABLE IF EXISTS paper_texts;
DROP TABLE IF EXISTS references_url;

-- Save 5 GB by NULLing abstracts on low-cited papers (will be done in a code step
-- after spaCy tagging is complete — included here as a placeholder query only).
-- See: scripts/prune_long_tail_abstracts.sql
