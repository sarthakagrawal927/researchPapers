-- TF-IDF / KMeans cluster id per paper. Computed against abstract text.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS semantic_cluster INTEGER;
CREATE INDEX IF NOT EXISTS papers_semantic_cluster_idx ON papers (semantic_cluster);
