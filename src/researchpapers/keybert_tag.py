"""KeyBERT tagger.

Embeds title+abstract with a small sentence-transformer (all-MiniLM-L6-v2),
generates candidate n-gram phrases, scores each by cosine similarity to the
document embedding. MMR diversity to avoid near-duplicate tags.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

from researchpapers.config import Settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.keybert_tag")

KEYBERT_MODEL = "all-MiniLM-L6-v2"


def tag_papers(
    settings: Settings,
    *,
    limit: int | None = None,
    only_top_cited: bool = True,
    batch_size: int = 32,
) -> dict[str, int | float]:
    from keybert import KeyBERT

    counters: dict[str, int | float] = {"tagged": 0, "skipped": 0}
    log.info("loading KeyBERT model %s", KEYBERT_MODEL)
    model = KeyBERT(model=KEYBERT_MODEL)

    order_clause = "ORDER BY citation_count DESC NULLS LAST" if only_top_cited else ""
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT arxiv_id, title, abstract
            FROM papers
            WHERE keybert_tagged_at IS NULL
              AND abstract IS NOT NULL
              AND LENGTH(abstract) > 80
            {order_clause}
            """
            + (f" LIMIT {int(limit)}" if limit else "")
        )
        rows = cur.fetchall()

    log.info("queue: %d papers", len(rows))
    t0 = time.monotonic()

    texts = [f"{r['title']}\n\n{r['abstract']}" for r in rows]
    # KeyBERT supports batched extraction.
    all_results = model.extract_keywords(
        texts,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.5,
        top_n=12,
    )
    # When given a list of docs, KeyBERT returns a list of result lists.
    if all_results and not isinstance(all_results[0], list):
        all_results = [all_results]  # single-doc edge case

    now = datetime.now(UTC)
    with connect(settings) as conn, conn.cursor() as cur:
        for r, kw_list in zip(rows, all_results, strict=True):
            tags = [kw for kw, _score in (kw_list or [])][:12]
            cur.execute(
                "UPDATE papers SET keybert_tags_json = %s::jsonb, keybert_tagged_at = %s WHERE arxiv_id = %s",
                (json.dumps(tags), now, r["arxiv_id"]),
            )
            counters["tagged"] += 1
            if not tags:
                counters["skipped"] += 1
        conn.commit()

    elapsed = time.monotonic() - t0
    counters["elapsed_seconds"] = round(elapsed, 2)
    counters["papers_per_sec"] = round(int(counters["tagged"]) / elapsed, 1) if elapsed else 0
    return counters
