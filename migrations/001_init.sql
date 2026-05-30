CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    s2_paper_id       TEXT,
    title             TEXT NOT NULL,
    abstract          TEXT,
    primary_category  TEXT,
    categories        TEXT[],
    submitted_date    DATE,
    updated_date      DATE,
    authors_json      JSONB,
    pdf_url           TEXT,
    pdf_path          TEXT,
    pdf_fetched_at    TIMESTAMPTZ,
    urls_extracted_at TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS papers_primary_category_idx ON papers (primary_category);
CREATE INDEX IF NOT EXISTS papers_submitted_date_idx   ON papers (submitted_date);
CREATE INDEX IF NOT EXISTS papers_s2_paper_id_idx      ON papers (s2_paper_id);

-- Outgoing paper -> paper edges from Semantic Scholar.
CREATE TABLE IF NOT EXISTS references_paper (
    id                BIGSERIAL PRIMARY KEY,
    citing_arxiv_id   TEXT NOT NULL REFERENCES papers (arxiv_id) ON DELETE CASCADE,
    cited_s2_id       TEXT,
    cited_arxiv_id    TEXT,
    cited_doi         TEXT,
    cited_title       TEXT,
    fetched_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One edge per (citing, cited_s2_id). When s2_id is unknown, dedupe on doi or arxiv_id.
CREATE UNIQUE INDEX IF NOT EXISTS references_paper_uq_s2
    ON references_paper (citing_arxiv_id, cited_s2_id)
    WHERE cited_s2_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS references_paper_uq_arxiv
    ON references_paper (citing_arxiv_id, cited_arxiv_id)
    WHERE cited_s2_id IS NULL AND cited_arxiv_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS references_paper_uq_doi
    ON references_paper (citing_arxiv_id, cited_doi)
    WHERE cited_s2_id IS NULL AND cited_arxiv_id IS NULL AND cited_doi IS NOT NULL;

CREATE INDEX IF NOT EXISTS references_paper_cited_arxiv_idx ON references_paper (cited_arxiv_id);
CREATE INDEX IF NOT EXISTS references_paper_cited_s2_idx    ON references_paper (cited_s2_id);

-- Outgoing paper -> URL edges from PDF text extraction.
CREATE TABLE IF NOT EXISTS references_url (
    id                BIGSERIAL PRIMARY KEY,
    citing_arxiv_id   TEXT NOT NULL REFERENCES papers (arxiv_id) ON DELETE CASCADE,
    url_raw           TEXT NOT NULL,
    url_canonical     TEXT NOT NULL,
    scheme            TEXT,
    host              TEXT,
    context_snippet   TEXT,
    extracted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (citing_arxiv_id, url_canonical)
);

CREATE INDEX IF NOT EXISTS references_url_host_idx          ON references_url (host);
CREATE INDEX IF NOT EXISTS references_url_canonical_idx     ON references_url (url_canonical);

-- Tracks which migrations have been applied. The CLI checks this before running each file.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
