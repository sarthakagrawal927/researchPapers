"""Populate papers.abstract_embedding with sentence-transformers all-MiniLM-L6-v2.

384-dim L2-normalized vectors. Fast enough to embed the whole corpus on CPU
in 1-2 hours. Anti-join against existing embeddings so re-runs are incremental.
"""

from __future__ import annotations

import logging
import time

from researchpapers.ch_db import connect as ch_connect

log = logging.getLogger("researchpapers.embed")

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def embed_papers(
    *,
    source: str | None = None,
    batch_size: int = 256,
    limit: int | None = None,
) -> dict[str, int | float]:
    """Embed title+abstract → INSERT into paper_embeddings (separate table, no UPDATE)."""
    from sentence_transformers import SentenceTransformer

    log.info("loading %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    src_clause = "AND source = %(src)s" if source else ""
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    counters: dict[str, int | float] = {"embedded": 0}
    t0 = time.monotonic()

    with ch_connect() as ch:
        rows = ch.query(
            f"""
            SELECT p.paper_id, p.title, p.abstract
            FROM papers AS p FINAL
            WHERE length(p.abstract) > 80
              AND p.paper_id NOT IN (SELECT paper_id FROM paper_embeddings FINAL)
              {src_clause}
            {limit_clause}
            """,
            parameters={"src": source} if source else {},
        ).result_rows
    log.info("queue: %d papers", len(rows))
    if not rows:
        return counters

    total = len(rows)
    batch_idx = 0
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        texts = [f"{r[1] or ''}. {(r[2] or '')[:1000]}" for r in batch]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        payload = [
            [r[0], embeddings[i].tolist(), MODEL_NAME]
            for i, r in enumerate(batch)
        ]
        with ch_connect() as ch:
            ch.insert(
                "paper_embeddings",
                payload,
                column_names=["paper_id", "embedding", "model"],
            )
        counters["embedded"] += len(batch)
        batch_idx += 1
        if batch_idx % 4 == 0:
            elapsed = time.monotonic() - t0
            rate = counters["embedded"] / elapsed if elapsed else 0
            eta_sec = (total - counters["embedded"]) / rate if rate else 0
            log.info("progress: %d/%d (%.1f p/s, ETA %d min)",
                     counters["embedded"], total, rate, int(eta_sec / 60))

    elapsed = time.monotonic() - t0
    counters["elapsed_seconds"] = round(elapsed, 2)
    counters["papers_per_sec"] = round(counters["embedded"] / elapsed, 1) if elapsed else 0
    return counters
