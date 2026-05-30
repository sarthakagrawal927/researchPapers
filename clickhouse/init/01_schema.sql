-- ClickHouse schema for researchPapers.
-- Designed for: multi-source ingest (arxiv, openreview, biorxiv, ...), append-only tagging,
-- citation graph at scale, time-series citation tracking.
--
-- All tables go into the `papers` database (created by docker env CLICKHOUSE_DB=papers).

------------------------------------------------------------------------
-- papers: canonical entity, deduplicated across sources at read time
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers
(
    paper_id          String,                          -- "arxiv:1412.6980", "openreview:abc123"
    source            LowCardinality(String),          -- 'arxiv' | 'openreview' | 'biorxiv' | ...
    source_id         String,                          -- the per-source identifier
    arxiv_id          Nullable(String),
    openalex_id       Nullable(String),
    doi               Nullable(String),
    title             String,
    abstract          Nullable(String),
    submitted_date    Nullable(Date),
    publication_year  Nullable(UInt16),
    citation_count    UInt32 DEFAULT 0,
    primary_category  LowCardinality(Nullable(String)),
    authors           Array(String),                   -- denormalized author names
    openalex_tags     Array(String),                   -- flat list of topic/keyword names
    openalex_keywords Array(String),

    -- analytical scores filled by background jobs
    pagerank_score    Nullable(Float64),
    katz_score        Nullable(Float64),
    community_id      Nullable(UInt32),
    semantic_cluster  Nullable(UInt16),
    in_corpus_degree  UInt32 DEFAULT 0,

    ingested_at       DateTime DEFAULT now(),
    updated_at        DateTime DEFAULT now(),

    -- vector embedding for semantic search (populated by separate job)
    abstract_embedding Array(Float32)                  -- size depends on encoder (384 or 768)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (source, source_id)
PARTITION BY toYear(coalesce(submitted_date, toDate('1970-01-01')))
SETTINGS index_granularity = 8192;

------------------------------------------------------------------------
-- paper_tags: append-only. Latest row per (paper_id, tagger) is the current truth.
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS paper_tags
(
    paper_id      String,
    tagger        LowCardinality(String),    -- 'spacy_v2' | 'mlx_qwen3b' | 'lm_qwen30b_oracle' | ...
    tags          Array(String),
    tldr          Nullable(String),
    model_version LowCardinality(Nullable(String)),
    computed_at   DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(computed_at)
ORDER BY (paper_id, tagger)
SETTINGS index_granularity = 8192;

------------------------------------------------------------------------
-- references_paper: append-only edges. LowCardinality on cited_openalex_id is HUGE for storage.
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS references_paper
(
    citing_paper_id     String,
    cited_openalex_id   LowCardinality(String),
    cited_arxiv_id      Nullable(String),
    cited_doi           Nullable(String),
    cited_title         Nullable(String),
    backfilled_at       DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (citing_paper_id, cited_openalex_id)
PARTITION BY toYYYYMM(backfilled_at)
SETTINGS index_granularity = 8192;

------------------------------------------------------------------------
-- citation_history: monthly snapshots of cited_by_count for trend detection.
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS citation_history
(
    paper_id        String,
    measured_at     Date,
    citation_count  UInt32
)
ENGINE = ReplacingMergeTree(measured_at)
ORDER BY (paper_id, measured_at);

------------------------------------------------------------------------
-- cited_works: resolved metadata for the most-cited OpenAlex IDs in our corpus.
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cited_works
(
    openalex_id        String,
    title              Nullable(String),
    cited_by_count     UInt32,
    doi                Nullable(String),
    publication_year   Nullable(UInt16),
    arxiv_id           Nullable(String),
    primary_topic      Nullable(String),
    resolved_at        DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(resolved_at)
ORDER BY openalex_id;

------------------------------------------------------------------------
-- citation_cycles: analytical artifact from graph job.
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS citation_cycles
(
    cycle_length  UInt8,
    paper_ids     Array(String),
    detected_at   DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (cycle_length, detected_at);

------------------------------------------------------------------------
-- openreview_reviews: where reviews/scores/decisions land. (Phase 2 ingest)
------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS openreview_reviews
(
    paper_id       String,                           -- joins to papers.paper_id where source='openreview'
    review_id      String,
    reviewer_id    Nullable(String),
    venue          LowCardinality(String),           -- 'NeurIPS-2024', 'ICLR-2025', etc.
    soundness      Nullable(UInt8),
    presentation   Nullable(UInt8),
    contribution   Nullable(UInt8),
    rating         Nullable(UInt8),
    confidence     Nullable(UInt8),
    summary        Nullable(String),
    strengths      Nullable(String),
    weaknesses     Nullable(String),
    questions      Nullable(String),
    decision       LowCardinality(Nullable(String)), -- 'Accept', 'Reject', 'Withdraw', etc.
    posted_at      Nullable(DateTime),
    ingested_at    DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(ingested_at)
ORDER BY (paper_id, review_id);
