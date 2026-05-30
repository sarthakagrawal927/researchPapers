-- Paper -> paper edges from OpenAlex `referenced_works` (a list of OpenAlex IDs per work).
-- We store the cited side as a bare OpenAlex id and resolve a top-N subset to titles in a
-- second pass against the same API.

ALTER TABLE references_paper ADD COLUMN IF NOT EXISTS cited_openalex_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS references_paper_uq_oa
    ON references_paper (citing_arxiv_id, cited_openalex_id)
    WHERE cited_openalex_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS references_paper_cited_oa_idx
    ON references_paper (cited_openalex_id);

-- Resolved metadata for the most-cited works in our corpus. Filled lazily after the backfill.
CREATE TABLE IF NOT EXISTS cited_works (
    openalex_id     TEXT PRIMARY KEY,
    title           TEXT,
    cited_by_count  INTEGER,
    doi             TEXT,
    publication_year INTEGER,
    arxiv_id        TEXT,
    primary_topic   TEXT,
    resolved_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
