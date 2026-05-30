"""Semantic Scholar reference fetcher. Wired but not run by default for v1.

S2's free tier without an API key is rate-limited well below 1 req/sec.
With a key (request at https://www.semanticscholar.org/product/api), the
limit lifts substantially. We prefer the batch endpoint with
`fields=references.title,references.externalIds` so one request returns
both the S2 id mapping AND the outgoing references for up to 500 papers.

Run `papers fetch-citations` to invoke this once you have a key.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from researchpapers.config import Settings
from researchpapers.db import connect
from researchpapers.http import build_client

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
BATCH_SIZE = 500
REQUEST_INTERVAL_SECONDS = 1.0  # 1 RPS without a key; loosen via config when one is set


def _chunks(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
)
def _post_batch(client: httpx.Client, ids: list[str], api_key: str | None) -> list[dict]:
    headers = {"x-api-key": api_key} if api_key else {}
    resp = client.post(
        S2_BATCH_URL,
        params={"fields": "externalIds,references.title,references.externalIds"},
        json={"ids": ids},
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_citations(settings: Settings, *, limit: int | None = None) -> tuple[int, int]:
    """Fetches references from S2 for papers without an s2_paper_id yet. Returns (papers, edges)."""
    interval = 0.1 if settings.semantic_scholar_api_key else REQUEST_INTERVAL_SECONDS
    papers_done = 0
    edges_written = 0
    with build_client(settings) as client, connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT arxiv_id FROM papers
                WHERE s2_paper_id IS NULL
                ORDER BY submitted_date DESC NULLS LAST
                """
                + (f" LIMIT {int(limit)}" if limit else "")
            )
            arxiv_ids = [r["arxiv_id"] for r in cur.fetchall()]
        last_request_at = 0.0
        for chunk in _chunks(arxiv_ids, BATCH_SIZE):
            elapsed = time.monotonic() - last_request_at
            if elapsed < interval:
                time.sleep(interval - elapsed)
            ids_for_batch = [f"ARXIV:{aid}" for aid in chunk]
            results = _post_batch(client, ids_for_batch, settings.semantic_scholar_api_key)
            last_request_at = time.monotonic()
            with conn.cursor() as cur:
                for arxiv_id, paper in zip(chunk, results, strict=True):
                    if paper is None:
                        continue
                    s2_id = paper.get("paperId")
                    if s2_id:
                        cur.execute(
                            "UPDATE papers SET s2_paper_id = %s WHERE arxiv_id = %s",
                            (s2_id, arxiv_id),
                        )
                    for ref in paper.get("references") or []:
                        ext = ref.get("externalIds") or {}
                        cur.execute(
                            """
                            INSERT INTO references_paper
                                (citing_arxiv_id, cited_s2_id, cited_arxiv_id, cited_doi, cited_title)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                arxiv_id,
                                ref.get("paperId"),
                                ext.get("ArXiv"),
                                ext.get("DOI"),
                                ref.get("title"),
                            ),
                        )
                        if cur.rowcount > 0:
                            edges_written += 1
                    papers_done += 1
            conn.commit()
    return papers_done, edges_written
