"""Migrate Postgres → ClickHouse for the canonical tables.

One-shot. Idempotent because the CH tables use ReplacingMergeTree on update timestamps,
so reruns just overwrite. Batches all inserts to keep RAM modest.

Run: uv run python scripts/migrate_pg_to_ch.py
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import clickhouse_connect
import psycopg
from psycopg.rows import dict_row

PG_DSN = "postgresql://papers:papers@localhost:5433/papers"
CH = dict(host="localhost", port=8123, database="papers", username="papers", password="papers")
BATCH = 5000


def _arxiv_pid(arxiv_id: str) -> str:
    return f"arxiv:{arxiv_id}"


def _safe_arr(v) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x]
    return []


def _flatten_tags(tags_json) -> list[str]:
    """OpenAlex tags_json is a list of {name, level, score}; flatten to names."""
    if not tags_json:
        return []
    return [t.get("name") for t in tags_json if isinstance(t, dict) and t.get("name")]


def _flatten_keywords(kw_json) -> list[str]:
    if not kw_json:
        return []
    return [k.get("name") for k in kw_json if isinstance(k, dict) and k.get("name")]


def _flatten_authors(authors_json) -> list[str]:
    if not authors_json:
        return []
    return [a.get("name") for a in authors_json if isinstance(a, dict) and a.get("name")]


def migrate_papers(pg, ch) -> int:
    """Pulls from PG papers → CH papers."""
    print("migrating papers...", flush=True)
    t0 = time.monotonic()
    n = 0
    with pg.cursor(name="papers_cursor", row_factory=dict_row) as cur:
        cur.itersize = BATCH
        cur.execute(
            """
            SELECT arxiv_id, openalex_id, doi, title, abstract,
                   submitted_date, citation_count, primary_category,
                   authors_json, tags_json, keywords_json,
                   pagerank_score, katz_score, community_id, semantic_cluster,
                   in_corpus_degree, references_backfilled_at
            FROM papers
            """
        )
        batch_rows: list[list] = []
        for r in cur:
            pid = _arxiv_pid(r["arxiv_id"])
            sub_date = r["submitted_date"]
            pub_year = sub_date.year if sub_date else None
            batch_rows.append([
                pid,
                "arxiv",
                r["arxiv_id"],
                r["arxiv_id"],
                r["openalex_id"],
                r["doi"],
                r["title"] or "",
                r["abstract"],
                sub_date,
                pub_year,
                int(r["citation_count"] or 0),
                r["primary_category"],
                _flatten_authors(r["authors_json"]),
                _flatten_tags(r["tags_json"]),
                _flatten_keywords(r["keywords_json"]),
                r["pagerank_score"],
                r["katz_score"],
                int(r["community_id"]) if r["community_id"] is not None else None,
                int(r["semantic_cluster"]) if r["semantic_cluster"] is not None else None,
                int(r["in_corpus_degree"] or 0),
                datetime.now(UTC),
                datetime.now(UTC),
                [],  # abstract_embedding placeholder
            ])
            if len(batch_rows) >= BATCH:
                ch.insert("papers", batch_rows, column_names=[
                    "paper_id", "source", "source_id", "arxiv_id", "openalex_id",
                    "doi", "title", "abstract", "submitted_date", "publication_year",
                    "citation_count", "primary_category", "authors",
                    "openalex_tags", "openalex_keywords",
                    "pagerank_score", "katz_score", "community_id", "semantic_cluster",
                    "in_corpus_degree", "ingested_at", "updated_at", "abstract_embedding",
                ])
                n += len(batch_rows)
                batch_rows = []
                if n % 25000 == 0:
                    elapsed = time.monotonic() - t0
                    print(f"  {n:,} papers ({n/elapsed:.0f}/sec)", flush=True)
        if batch_rows:
            ch.insert("papers", batch_rows, column_names=[
                "paper_id", "source", "source_id", "arxiv_id", "openalex_id",
                "doi", "title", "abstract", "submitted_date", "publication_year",
                "citation_count", "primary_category", "authors",
                "openalex_tags", "openalex_keywords",
                "pagerank_score", "katz_score", "community_id", "semantic_cluster",
                "in_corpus_degree", "ingested_at", "updated_at", "abstract_embedding",
            ])
            n += len(batch_rows)
    print(f"  done: {n:,} papers in {time.monotonic()-t0:.1f}s", flush=True)
    return n


def migrate_paper_tags(pg, ch) -> int:
    """Each non-NULL tag column becomes a row in paper_tags with the matching tagger name."""
    print("migrating paper_tags...", flush=True)
    t0 = time.monotonic()
    n = 0
    tagger_map = [
        ("noun_tags_v2_json",    "spacy_v2",       "noun_v2_tagged_at"),
        ("mlx_llm_v2_tags_json", "mlx_qwen3b_v2",  "mlx_llm_v2_tagged_at"),
        ("llm_tags_json",        "lm_qwen30b_oracle", "llm_tagged_at"),
    ]
    with pg.cursor(row_factory=dict_row) as cur:
        for col, tagger, ts_col in tagger_map:
            # Skip columns that don't exist after our cleanup migrations
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='papers' AND column_name=%s",
                (col,),
            )
            if not cur.fetchone():
                print(f"  skip {tagger}: column {col} no longer exists", flush=True)
                continue
            tldr_col = "llm_tldr" if col == "llm_tags_json" else None
            tldr_select = f", {tldr_col}" if tldr_col else ", NULL"
            cur.execute(
                f"SELECT arxiv_id, {col} AS tags, {ts_col} AS ts {tldr_select} AS tldr "
                f"FROM papers WHERE {col} IS NOT NULL"
            )
            batch: list[list] = []
            for r in cur:
                tags = r["tags"] if isinstance(r["tags"], list) else []
                tags = [str(t) for t in tags if t]
                batch.append([
                    _arxiv_pid(r["arxiv_id"]),
                    tagger,
                    tags,
                    r.get("tldr"),
                    None,
                    r["ts"] or datetime.now(UTC),
                ])
                if len(batch) >= BATCH:
                    ch.insert("paper_tags", batch, column_names=[
                        "paper_id", "tagger", "tags", "tldr", "model_version", "computed_at",
                    ])
                    n += len(batch)
                    batch = []
            if batch:
                ch.insert("paper_tags", batch, column_names=[
                    "paper_id", "tagger", "tags", "tldr", "model_version", "computed_at",
                ])
                n += len(batch)
            print(f"  {tagger}: cumulative {n:,}", flush=True)
    print(f"  done: {n:,} tag rows in {time.monotonic()-t0:.1f}s", flush=True)
    return n


def migrate_references(pg, ch) -> int:
    print("migrating references_paper...", flush=True)
    t0 = time.monotonic()
    n = 0
    with pg.cursor(name="refs_cursor", row_factory=dict_row) as cur:
        cur.itersize = BATCH
        cur.execute(
            """
            SELECT citing_arxiv_id, cited_openalex_id, cited_arxiv_id,
                   cited_doi, cited_title, fetched_at
            FROM references_paper
            WHERE cited_openalex_id IS NOT NULL
            """
        )
        batch: list[list] = []
        for r in cur:
            batch.append([
                _arxiv_pid(r["citing_arxiv_id"]),
                r["cited_openalex_id"],
                r["cited_arxiv_id"],
                r["cited_doi"],
                r["cited_title"],
                r["fetched_at"] or datetime.now(UTC),
            ])
            if len(batch) >= BATCH:
                ch.insert("references_paper", batch, column_names=[
                    "citing_paper_id", "cited_openalex_id", "cited_arxiv_id",
                    "cited_doi", "cited_title", "backfilled_at",
                ])
                n += len(batch)
                batch = []
                if n % 250_000 == 0:
                    elapsed = time.monotonic() - t0
                    print(f"  {n:,} edges ({n/elapsed:.0f}/sec)", flush=True)
        if batch:
            ch.insert("references_paper", batch, column_names=[
                "citing_paper_id", "cited_openalex_id", "cited_arxiv_id",
                "cited_doi", "cited_title", "backfilled_at",
            ])
            n += len(batch)
    print(f"  done: {n:,} edges in {time.monotonic()-t0:.1f}s", flush=True)
    return n


def migrate_cited_works(pg, ch) -> int:
    print("migrating cited_works...", flush=True)
    t0 = time.monotonic()
    n = 0
    with pg.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT openalex_id, title, cited_by_count, doi, publication_year,
                   arxiv_id, primary_topic, resolved_at
            FROM cited_works
            """
        )
        batch = []
        for r in cur:
            batch.append([
                r["openalex_id"], r["title"], int(r["cited_by_count"] or 0),
                r["doi"], r["publication_year"], r["arxiv_id"], r["primary_topic"],
                r["resolved_at"] or datetime.now(UTC),
            ])
        if batch:
            ch.insert("cited_works", batch, column_names=[
                "openalex_id", "title", "cited_by_count", "doi",
                "publication_year", "arxiv_id", "primary_topic", "resolved_at",
            ])
            n = len(batch)
    print(f"  done: {n:,} cited_works in {time.monotonic()-t0:.1f}s", flush=True)
    return n


def migrate_cycles(pg, ch) -> int:
    print("migrating citation_cycles...", flush=True)
    t0 = time.monotonic()
    n = 0
    with pg.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT cycle_length, arxiv_ids, detected_at FROM citation_cycles")
        batch = []
        for r in cur:
            ids = [_arxiv_pid(a) for a in (r["arxiv_ids"] or [])]
            batch.append([int(r["cycle_length"]), ids, r["detected_at"] or datetime.now(UTC)])
            if len(batch) >= BATCH:
                ch.insert("citation_cycles", batch, column_names=[
                    "cycle_length", "paper_ids", "detected_at",
                ])
                n += len(batch)
                batch = []
        if batch:
            ch.insert("citation_cycles", batch, column_names=[
                "cycle_length", "paper_ids", "detected_at",
            ])
            n += len(batch)
    print(f"  done: {n:,} cycles in {time.monotonic()-t0:.1f}s", flush=True)
    return n


def main() -> None:
    print("connecting...", flush=True)
    pg = psycopg.connect(PG_DSN, row_factory=dict_row)
    ch = clickhouse_connect.get_client(**CH)
    try:
        migrate_papers(pg, ch)
        migrate_paper_tags(pg, ch)
        migrate_cited_works(pg, ch)
        migrate_cycles(pg, ch)
        migrate_references(pg, ch)

        print("\n=== final ClickHouse row counts ===", flush=True)
        for tbl in ("papers", "paper_tags", "references_paper", "cited_works", "citation_cycles"):
            n = ch.query(f"SELECT count() FROM {tbl}").result_rows[0][0]
            print(f"  {tbl}: {n:,}", flush=True)
    finally:
        pg.close()
        ch.close()


if __name__ == "__main__":
    main()
