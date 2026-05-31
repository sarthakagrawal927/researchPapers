"""HTTP API over the researchPapers ClickHouse corpus.

Exa-style endpoints over the multi-source paper data. Backed entirely by ClickHouse —
no Postgres dependency.

Endpoints:
    GET /healthz                     — liveness
    GET /stats                       — corpus summary (papers by source, tag/review coverage)
    GET /search?q=...                — keyword search across title + abstract
    GET /papers/{paper_id}           — single paper with all tags + reviews
    GET /tags/top-rated              — tag×reviewer-rating leaderboard
    GET /tags/{tag}                  — papers tagged with a specific tag
    GET /reviews/top-rated           — highest-rated reviewed submissions

Launch with:  uv run papers api-serve --port 8000
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from researchpapers.ch_db import connect as ch_connect

log = logging.getLogger("researchpapers.api")

app = FastAPI(
    title="researchPapers API",
    description="Search and analytics over a 488k multi-source academic paper corpus (arxiv + openreview + biorxiv + medrxiv).",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


VALID_SOURCES = {"arxiv", "openreview", "biorxiv", "medrxiv", "chemrxiv"}


def _split_sources(s: str | None) -> list[str] | None:
    if not s:
        return None
    out = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in VALID_SOURCES]
    if bad:
        raise HTTPException(400, f"Unknown sources: {bad}. Valid: {sorted(VALID_SOURCES)}")
    return out or None


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/stats")
def stats() -> dict:
    """Corpus summary: papers by source, tag coverage, review count, edge count."""
    with ch_connect() as c:
        sources = c.query(
            "SELECT source, count() AS n FROM papers GROUP BY source ORDER BY n DESC"
        ).result_rows
        tag_coverage = c.query("""
            SELECT splitByChar(':', paper_id)[1] AS source, count() AS n
            FROM paper_tags FINAL WHERE tagger='spacy_v2' GROUP BY source
        """).result_rows
        n_reviews = c.query("SELECT count() FROM openreview_reviews").result_rows[0][0]
        n_edges = c.query("SELECT count() FROM references_paper").result_rows[0][0]
        on_disk = c.query(
            "SELECT formatReadableSize(sum(bytes_on_disk)) FROM system.parts WHERE active AND database='papers'"
        ).result_rows[0][0]
    return {
        "papers_by_source": [{"source": s, "n": int(n)} for s, n in sources],
        "spacy_tag_coverage": {s: int(n) for s, n in tag_coverage},
        "openreview_reviews": int(n_reviews),
        "paper_to_paper_edges": int(n_edges),
        "clickhouse_on_disk": on_disk,
    }


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=2, max_length=200, description="Search keyword(s)")],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    sources: Annotated[str | None, Query(description="Comma-separated sources to filter (e.g. arxiv,openreview)")] = None,
    min_citations: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Keyword search over titles + abstracts. Substring match (case-insensitive)."""
    src_filter = _split_sources(sources)
    src_clause = "AND source IN %(sources)s" if src_filter else ""
    with ch_connect() as c:
        rows = c.query(
            f"""
            SELECT paper_id, source, title, substring(abstract, 1, 400) AS abstract_preview,
                   submitted_date, citation_count, doi, arxiv_id
            FROM papers FINAL
            WHERE (positionCaseInsensitive(title, %(q)s) > 0
                   OR positionCaseInsensitive(abstract, %(q)s) > 0)
              AND citation_count >= %(min_citations)s
              {src_clause}
            ORDER BY citation_count DESC
            LIMIT %(limit)s
            """,
            parameters={
                "q": q,
                "limit": limit,
                "min_citations": min_citations,
                **({"sources": src_filter} if src_filter else {}),
            },
        ).result_rows
    return {
        "query": q,
        "count": len(rows),
        "results": [
            {
                "paper_id": r[0],
                "source": r[1],
                "title": r[2],
                "abstract_preview": r[3],
                "submitted_date": str(r[4]) if r[4] else None,
                "citation_count": int(r[5] or 0),
                "doi": r[6],
                "arxiv_id": r[7],
            }
            for r in rows
        ],
    }


@app.get("/papers/{paper_id:path}")
def get_paper(paper_id: str) -> dict:
    """Single paper detail. paper_id format: 'arxiv:1412.6980' or 'openreview:abc123'."""
    with ch_connect() as c:
        p = c.query(
            """SELECT p.paper_id, p.source,
                   coalesce(nullIf(m.title, ''), p.title) AS title,
                   p.abstract, p.submitted_date,
                   coalesce(nullIf(m.citation_count, 0), p.citation_count) AS citation_count,
                   p.doi, p.arxiv_id, p.authors
              FROM papers AS p FINAL
              LEFT JOIN paper_metadata_v2 AS m FINAL ON m.paper_id = p.paper_id
              WHERE p.paper_id = %(pid)s""",
            parameters={"pid": paper_id},
        ).result_rows
        if not p:
            raise HTTPException(404, f"paper not found: {paper_id}")
        row = p[0]
        tags = c.query(
            "SELECT tagger, tags, tldr, computed_at FROM paper_tags FINAL WHERE paper_id = %(pid)s ORDER BY computed_at DESC",
            parameters={"pid": paper_id},
        ).result_rows
        reviews = c.query(
            "SELECT venue, rating, confidence, decision, summary, strengths, weaknesses FROM openreview_reviews WHERE paper_id = %(pid)s",
            parameters={"pid": paper_id},
        ).result_rows
    return {
        "paper_id": row[0],
        "source": row[1],
        "title": row[2],
        "abstract": row[3],
        "submitted_date": str(row[4]) if row[4] else None,
        "citation_count": int(row[5] or 0),
        "doi": row[6],
        "arxiv_id": row[7],
        "authors": list(row[8] or []),
        "tags": [
            {
                "tagger": t[0],
                "tags": list(t[1] or []),
                "tldr": t[2],
                "computed_at": str(t[3]) if t[3] else None,
            }
            for t in tags
        ],
        "reviews": [
            {
                "venue": r[0],
                "rating": int(r[1]) if r[1] is not None else None,
                "confidence": int(r[2]) if r[2] is not None else None,
                "decision": r[3],
                "summary": r[4],
                "strengths": r[5],
                "weaknesses": r[6],
            }
            for r in reviews
        ],
    }


@app.get("/tags/top-rated")
def tags_top_rated(
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    min_papers: Annotated[int, Query(ge=1, le=200)] = 10,
) -> dict:
    """Tags ordered by mean OpenReview reviewer rating across papers tagged with them."""
    with ch_connect() as c:
        rows = c.query(
            """
            WITH paper_avg_rating AS (
                SELECT paper_id, avg(rating) AS avg_rating, count() AS n_reviews
                FROM openreview_reviews
                WHERE rating IS NOT NULL
                GROUP BY paper_id
                HAVING n_reviews >= 3
            )
            SELECT tag, round(avg(par.avg_rating), 2) AS mean_rating, count() AS n_papers,
                   round(quantile(0.9)(par.avg_rating), 2) AS p90_rating
            FROM paper_tags t FINAL
            ARRAY JOIN tags AS tag
            JOIN paper_avg_rating par ON par.paper_id = t.paper_id
            WHERE t.tagger = 'spacy_v2'
            GROUP BY tag
            HAVING n_papers >= %(min_papers)s
            ORDER BY mean_rating DESC
            LIMIT %(limit)s
            """,
            parameters={"min_papers": min_papers, "limit": limit},
        ).result_rows
    return {
        "count": len(rows),
        "tags": [
            {"tag": r[0], "mean_rating": float(r[1] or 0), "n_papers": int(r[2]), "p90_rating": float(r[3] or 0)}
            for r in rows
        ],
    }


@app.get("/tags/{tag}")
def tag_papers(
    tag: str,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> dict:
    """Papers tagged with the given (exact) tag, ordered by citation count."""
    with ch_connect() as c:
        rows = c.query(
            """
            SELECT p.paper_id, p.source, p.title, substring(p.abstract, 1, 300) AS abstract_preview,
                   p.citation_count, p.submitted_date
            FROM paper_tags t FINAL
            JOIN papers p ON p.paper_id = t.paper_id
            WHERE t.tagger = 'spacy_v2' AND has(t.tags, %(tag)s)
            ORDER BY p.citation_count DESC
            LIMIT %(limit)s
            """,
            parameters={"tag": tag, "limit": limit},
        ).result_rows
    return {
        "tag": tag,
        "count": len(rows),
        "results": [
            {
                "paper_id": r[0],
                "source": r[1],
                "title": r[2],
                "abstract_preview": r[3],
                "citation_count": int(r[4] or 0),
                "submitted_date": str(r[5]) if r[5] else None,
            }
            for r in rows
        ],
    }


@app.get("/reviews/top-rated")
def reviews_top_rated(
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    venue: Annotated[str | None, Query(description="Filter by venue, e.g. ICLR-2025")] = None,
) -> dict:
    """Highest-rated reviewed submissions (mean reviewer rating, ≥3 reviews)."""
    venue_clause = "AND venue = %(venue)s" if venue else ""
    with ch_connect() as c:
        rows = c.query(
            f"""
            SELECT r.paper_id, p.title, r.venue, avg(r.rating) AS avg_rating,
                   count() AS n_reviews, any(r.decision) AS decision
            FROM openreview_reviews r
            LEFT JOIN papers p ON p.paper_id = r.paper_id
            WHERE r.rating IS NOT NULL {venue_clause}
            GROUP BY r.paper_id, p.title, r.venue
            HAVING n_reviews >= 3
            ORDER BY avg_rating DESC
            LIMIT %(limit)s
            """,
            parameters={"limit": limit, **({"venue": venue} if venue else {})},
        ).result_rows
    return {
        "count": len(rows),
        "results": [
            {
                "paper_id": r[0],
                "title": r[1],
                "venue": r[2],
                "avg_rating": round(float(r[3] or 0), 2),
                "n_reviews": int(r[4]),
                "decision": r[5],
            }
            for r in rows
        ],
    }


@app.get("/semantic-search")
def semantic_search(
    q: Annotated[str, Query(min_length=3, max_length=300, description="Natural-language query")],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    sources: Annotated[str | None, Query(description="Comma-separated source filter")] = None,
    min_citations: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Semantic search over abstract+title embeddings (all-MiniLM-L6-v2, 384-dim, cosine).

    Computes the query embedding inline, then JOINs against paper_embeddings.
    """
    from sentence_transformers import SentenceTransformer
    # Cache the model on the app instance for warm-call latency
    if not hasattr(app.state, "embedder"):
        app.state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = app.state.embedder.encode([q], normalize_embeddings=True)[0].tolist()

    src_filter = _split_sources(sources)
    src_clause = "AND p.source IN %(sources)s" if src_filter else ""

    with ch_connect() as c:
        rows = c.query(
            f"""
            SELECT p.paper_id, p.source, p.title, substring(p.abstract, 1, 400) AS abstract_preview,
                   p.submitted_date, p.citation_count, p.doi, p.arxiv_id,
                   round(1 - cosineDistance(e.embedding, %(q_emb)s), 4) AS similarity
            FROM paper_embeddings AS e FINAL
            JOIN papers AS p FINAL ON p.paper_id = e.paper_id
            WHERE p.citation_count >= %(min_citations)s
              {src_clause}
            ORDER BY cosineDistance(e.embedding, %(q_emb)s) ASC
            LIMIT %(limit)s
            """,
            parameters={
                "q_emb": q_emb,
                "limit": limit,
                "min_citations": min_citations,
                **({"sources": src_filter} if src_filter else {}),
            },
        ).result_rows
    return {
        "query": q,
        "count": len(rows),
        "results": [
            {
                "paper_id": r[0], "source": r[1], "title": r[2],
                "abstract_preview": r[3],
                "submitted_date": str(r[4]) if r[4] else None,
                "citation_count": int(r[5] or 0),
                "doi": r[6], "arxiv_id": r[7],
                "similarity": float(r[8]),
            }
            for r in rows
        ],
    }


@app.get("/sleepers")
def sleepers(
    min_rating: Annotated[float, Query(ge=5.0, le=10.0)] = 7.0,
    max_citations: Annotated[int, Query(ge=0, le=1000)] = 20,
    since_year: Annotated[int, Query(ge=2000)] = 2024,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict:
    """Sleeper papers — reviewers loved them but they haven't accrued citations yet.

    The "look what the field is missing" leaderboard.
    """
    with ch_connect() as c:
        rows = c.query(
            """
            WITH par AS (
              SELECT paper_id, avg(rating) AS avg_rating, count() AS n_reviews,
                     any(decision) AS decision, any(venue) AS venue
              FROM openreview_reviews WHERE rating IS NOT NULL
              GROUP BY paper_id HAVING n_reviews >= 3
            )
            SELECT p.paper_id,
                   coalesce(nullIf(m.title, ''), p.title) AS title,
                   par.avg_rating, par.n_reviews,
                   coalesce(nullIf(m.citation_count, 0), p.citation_count) AS citation_count,
                   par.venue, par.decision, p.submitted_date
            FROM par
            JOIN papers p ON p.paper_id = par.paper_id
            LEFT JOIN paper_metadata_v2 AS m FINAL ON m.paper_id = p.paper_id
            WHERE par.avg_rating >= %(min_rating)s
              AND p.citation_count <= %(max_citations)s
              AND effective_year(p.source, p.arxiv_id, p.submitted_date) >= %(since_year)s
            ORDER BY par.avg_rating DESC, p.citation_count ASC
            LIMIT %(limit)s
            """,
            parameters={"min_rating": min_rating, "max_citations": max_citations,
                        "since_year": since_year, "limit": limit},
        ).result_rows
    return {
        "count": len(rows),
        "results": [
            {"paper_id": r[0], "title": r[1], "avg_rating": round(float(r[2]), 2),
             "n_reviews": int(r[3]), "citation_count": int(r[4] or 0),
             "venue": r[5], "decision": r[6], "submitted_date": str(r[7]) if r[7] else None}
            for r in rows
        ],
    }


@app.get("/similar/{paper_id:path}")
def similar_papers(
    paper_id: str,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> dict:
    """Papers similar to the given one, via embedding cosine distance.

    Falls back to tag+community overlap if the paper has no embedding yet
    (e.g. very short or missing abstract).
    """
    with ch_connect() as c:
        # Try embedding-based first.
        rows = c.query(
            """
            WITH (SELECT embedding FROM paper_embeddings FINAL WHERE paper_id = %(pid)s LIMIT 1) AS anchor_emb
            SELECT p.paper_id, p.title, p.source, p.citation_count, p.submitted_date,
                   round(1 - cosineDistance(e.embedding, anchor_emb), 4) AS similarity
            FROM paper_embeddings AS e FINAL
            JOIN papers AS p FINAL ON p.paper_id = e.paper_id
            WHERE p.paper_id != %(pid)s
              AND length(anchor_emb) > 0
            ORDER BY cosineDistance(e.embedding, anchor_emb) ASC
            LIMIT %(limit)s
            """,
            parameters={"pid": paper_id, "limit": limit},
        ).result_rows
        anchor_title_q = c.query(
            "SELECT title FROM papers FINAL WHERE paper_id = %(pid)s",
            parameters={"pid": paper_id},
        ).result_rows
        if not anchor_title_q:
            raise HTTPException(404, f"anchor not found: {paper_id}")
        anchor_title = anchor_title_q[0][0]

        if not rows:
            # No embedding for anchor — fall back to tag+community overlap.
            anchor = c.query(
                "SELECT community_id, openalex_tags FROM papers FINAL WHERE paper_id = %(pid)s",
                parameters={"pid": paper_id},
            ).result_rows
            if anchor:
                cid, tags = anchor[0]
                rows = c.query(
                    """
                    SELECT paper_id, title, source, citation_count, submitted_date, 0.0 AS similarity
                    FROM papers FINAL
                    WHERE paper_id != %(pid)s
                      AND community_id = %(cid)s
                      AND length(arrayIntersect(openalex_tags, %(tags)s)) >= 1
                    ORDER BY citation_count DESC
                    LIMIT %(limit)s
                    """,
                    parameters={"pid": paper_id, "cid": cid, "tags": list(tags or []), "limit": limit},
                ).result_rows
    return {
        "anchor": {"paper_id": paper_id, "title": anchor_title},
        "method": "embedding" if rows and rows[0][5] > 0 else "tag_overlap",
        "count": len(rows),
        "results": [
            {"paper_id": r[0], "title": r[1], "source": r[2],
             "citation_count": int(r[3] or 0),
             "submitted_date": str(r[4]) if r[4] else None,
             "similarity": float(r[5])}
            for r in rows
        ],
    }


@app.get("/hot")
def hot_papers(
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    since_year: Annotated[int, Query(ge=2010)] = 2023,
) -> dict:
    """Unified 'hotness' ranking: combines cites/year + PageRank + reviewer rating.

    Score = 0.5 * log1p(cites_per_year) + 0.3 * (avg_rating/10 or 0.5)
            + 0.2 * pagerank*10000
    """
    with ch_connect() as c:
        rows = c.query(
            """
            WITH par AS (
              SELECT paper_id, avg(rating) AS avg_rating
              FROM openreview_reviews WHERE rating IS NOT NULL
              GROUP BY paper_id HAVING count() >= 3
            )
            SELECT p.paper_id, p.source, p.title, p.citation_count,
                   p.submitted_date,
                   round(p.citation_count / greatest((today() - effective_date(p.source, p.arxiv_id, p.submitted_date)) / 365.25, 0.25), 1) AS cpy,
                   coalesce(par.avg_rating, 0) AS rating,
                   coalesce(p.pagerank_score, 0) AS pr,
                   round(
                     0.5 * log(1 + p.citation_count / greatest((today() - effective_date(p.source, p.arxiv_id, p.submitted_date)) / 365.25, 0.25))
                     + 0.3 * coalesce(par.avg_rating, 5.0) / 10
                     + 0.2 * coalesce(p.pagerank_score, 0) * 10000,
                   3) AS hotness
            FROM papers AS p FINAL
            LEFT JOIN par ON par.paper_id = p.paper_id
            WHERE p.submitted_date IS NOT NULL
              AND effective_year(p.source, p.arxiv_id, p.submitted_date) >= %(year)s
              AND p.citation_count >= 5
            ORDER BY hotness DESC
            LIMIT %(limit)s
            """,
            parameters={"year": since_year, "limit": limit},
        ).result_rows
    return {
        "count": len(rows),
        "results": [
            {"paper_id": r[0], "source": r[1], "title": r[2],
             "citation_count": int(r[3] or 0),
             "submitted_date": str(r[4]) if r[4] else None,
             "cites_per_year": float(r[5]),
             "avg_rating": round(float(r[6]), 2) if r[6] else None,
             "pagerank": float(r[7]),
             "hotness": float(r[8])}
            for r in rows
        ],
    }


@app.get("/authors/by-tag/{tag}")
def authors_by_tag(
    tag: str,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    min_papers: Annotated[int, Query(ge=1, le=20)] = 2,
) -> dict:
    """Top authors writing papers tagged with X, by avg reviewer rating + paper count.

    Returns `n_communities` and `n_semantic_clusters` per author — if these
    are ≥3, the name is almost certainly multiple distinct people merged
    together (no author disambiguation in OpenAlex source data).
    Also returns `disambiguated_buckets`: same author split by community for
    a "best guess" disambiguation.
    """
    with ch_connect() as c:
        rows = c.query(
            """
            WITH par AS (
              SELECT paper_id, avg(rating) AS avg_rating
              FROM openreview_reviews WHERE rating IS NOT NULL
              GROUP BY paper_id HAVING count() >= 3
            ),
            tagged AS (
              SELECT paper_id FROM paper_tags FINAL
              WHERE tagger = 'spacy_v2' AND has(tags, %(tag)s)
            )
            SELECT arrayJoin(p.authors) AS author,
                   count() AS n_papers,
                   round(avg(coalesce(par.avg_rating, 0)), 2) AS avg_rating_when_reviewed,
                   countIf(par.avg_rating IS NOT NULL) AS n_reviewed,
                   sum(p.citation_count) AS sum_citations,
                   length(groupUniqArray(p.community_id)) AS n_communities,
                   length(groupUniqArray(p.semantic_cluster)) AS n_semantic_clusters
            FROM tagged AS t
            JOIN papers AS p FINAL ON p.paper_id = t.paper_id
            LEFT JOIN par ON par.paper_id = p.paper_id
            WHERE length(p.authors) > 0
            GROUP BY author
            HAVING n_papers >= %(min_papers)s
            ORDER BY (avg_rating_when_reviewed * sqrt(n_reviewed) + log(1 + sum_citations)) DESC
            LIMIT %(limit)s
            """,
            parameters={"tag": tag, "min_papers": min_papers, "limit": limit},
        ).result_rows
    return {
        "tag": tag,
        "count": len(rows),
        "results": [
            {"author": r[0], "n_papers": int(r[1]),
             "avg_rating_when_reviewed": float(r[2]) if r[2] else None,
             "n_reviewed": int(r[3]),
             "sum_citations": int(r[4] or 0),
             "n_communities": int(r[5] or 0),
             "n_semantic_clusters": int(r[6] or 0),
             "likely_multiple_people": int(r[5] or 0) >= 3 and int(r[1]) >= 5}
            for r in rows
        ],
    }


@app.get("/authors/by-id/{openalex_id}")
def author_by_openalex_id(openalex_id: str) -> dict:
    """Look up an author's papers via OpenAlex author ID (proper disambiguation).

    Only works for papers in paper_metadata_v2 (top ~2000 most-cited as of now).
    """
    with ch_connect() as c:
        rows = c.query(
            """
            SELECT
              p.paper_id,
              coalesce(nullIf(m.title, ''), p.title) AS title,
              coalesce(nullIf(m.citation_count, 0), p.citation_count) AS citation_count,
              p.submitted_date,
              arrayFirst(a -> a.2 = %(oid)s, m.authors).1 AS display_name
            FROM paper_metadata_v2 AS m FINAL
            JOIN papers AS p FINAL ON p.paper_id = m.paper_id
            WHERE arrayExists(a -> a.2 = %(oid)s, m.authors)
            ORDER BY citation_count DESC
            LIMIT 50
            """,
            parameters={"oid": openalex_id},
        ).result_rows
    if not rows:
        raise HTTPException(404, f"no papers found for OpenAlex author {openalex_id}")
    return {
        "openalex_id": openalex_id,
        "display_name_sample": rows[0][4],
        "n_papers": len(rows),
        "papers": [
            {"paper_id": r[0], "title": r[1], "citation_count": int(r[2] or 0),
             "submitted_date": str(r[3]) if r[3] else None}
            for r in rows
        ],
    }


@app.get("/authors/{author}/disambiguate")
def disambiguate_author(
    author: str,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> dict:
    """Heuristic disambiguation: for a given author name, return one entry per
    distinct (community, top-cluster) tuple they appear in. Same name in 3
    different communities likely = 3 different people.
    """
    with ch_connect() as c:
        rows = c.query(
            """
            SELECT
              coalesce(community_id, 9999) AS cid,
              coalesce(semantic_cluster, 9999) AS sc,
              count() AS n_papers,
              sum(citation_count) AS sum_citations,
              arraySlice(groupArray((paper_id, title, citation_count)), 1, 5) AS samples
            FROM papers FINAL
            WHERE has(authors, %(author)s)
            GROUP BY cid, sc
            ORDER BY n_papers DESC
            LIMIT %(limit)s
            """,
            parameters={"author": author, "limit": limit},
        ).result_rows
    return {
        "author": author,
        "n_buckets": len(rows),
        "interpretation": f"This name appears in {len(rows)} community/cluster bucket(s). "
                          f"Buckets with very different topic mixes are likely different real people.",
        "buckets": [
            {
                "community_id": int(r[0]) if r[0] != 9999 else None,
                "semantic_cluster": int(r[1]) if r[1] != 9999 else None,
                "n_papers": int(r[2]),
                "sum_citations": int(r[3] or 0),
                "sample_papers": [
                    {"paper_id": s[0], "title": s[1], "citation_count": int(s[2] or 0)}
                    for s in r[4]
                ],
            }
            for r in rows
        ],
    }


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({
        "name": "researchPapers API",
        "version": "0.2.0",
        "endpoints": [
            "GET /healthz",
            "GET /stats",
            "GET /semantic-search?q=...&limit=20",
            "GET /search?q=transformer&limit=20&sources=arxiv,openreview&min_citations=10",
            "GET /papers/{paper_id}",
            "GET /similar/{paper_id}?limit=10",
            "GET /sleepers?min_rating=7&max_citations=20&since_year=2024",
            "GET /hot?since_year=2023&limit=25",
            "GET /tags/top-rated?limit=25&min_papers=10",
            "GET /tags/{tag}?limit=20",
            "GET /authors/by-tag/{tag}?limit=25",
            "GET /authors/{author}/disambiguate — split a name into community/cluster buckets",
            "GET /reviews/top-rated?limit=25&venue=ICLR-2025",
        ],
        "docs": "/docs",
    })
