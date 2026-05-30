-- Graph centrality scores for papers + cycle records.

ALTER TABLE papers ADD COLUMN IF NOT EXISTS pagerank_score   DOUBLE PRECISION;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS katz_score       DOUBLE PRECISION;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS in_corpus_degree INTEGER;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS graph_scored_at  TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS papers_pagerank_idx ON papers (pagerank_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS papers_katz_idx     ON papers (katz_score DESC NULLS LAST);

-- Simple cycles found in the within-corpus subgraph. Citation cycles should be rare in
-- academic data, but they happen with mutual preprint references across versions.
CREATE TABLE IF NOT EXISTS citation_cycles (
    id            BIGSERIAL PRIMARY KEY,
    cycle_length  INTEGER NOT NULL,
    arxiv_ids     TEXT[] NOT NULL,
    detected_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS citation_cycles_length_idx ON citation_cycles (cycle_length);
