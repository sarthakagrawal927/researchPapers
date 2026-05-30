"""Streaming ingest with parallel pdfminer extraction.

Per paper: download PDF (rate-limited to 3s/req) → submit bytes to worker pool
→ worker extracts text + URLs → main thread writes results to Postgres → temp PDF
is gone the moment the worker returns. No PDFs accumulate on disk.

The 3s/req download rate is the floor (arXiv's polite-scraping policy). pdfminer
extraction used to be serial after each download (2-5s per paper); pooling it
across 4 workers means extraction is no longer on the critical path and total
runtime is bounded by 8500 × 3s ≈ 7 hours instead of ~28.

  for each paper without urls_extracted_at:
    download PDF bytes (rate-limited)
    submit to ProcessPoolExecutor → returns (text, urls)
    write urls + gzipped text → mark done

Idempotent + resumable: re-runs only pick up papers with urls_extracted_at IS NULL.
"""

from __future__ import annotations

import gzip
import logging
import os
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from researchpapers.config import Settings
from researchpapers.db import connect
from researchpapers.http import build_client

REQUEST_INTERVAL_SECONDS = 3.0
PDF_BASE = "https://export.arxiv.org/pdf/"
LOG_EVERY = 50
DEFAULT_WORKERS = 4
DEFAULT_MAX_IN_FLIGHT = 8

log = logging.getLogger("researchpapers.ingest")


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(httpx.HTTPError),
)
def _download_bytes(client: httpx.Client, url: str) -> bytes:
    r = client.get(url)
    r.raise_for_status()
    return r.content


def _pdf_url_for(arxiv_id: str) -> str:
    return f"{PDF_BASE}{arxiv_id}"


def _extract_worker(pdf_bytes: bytes) -> tuple[str, list[tuple[str, str, str, str, str]]]:
    """Runs inside a pool worker. Takes PDF bytes, returns (text, url_tuples).

    Imports its own deps so the worker process doesn't inherit module state.
    """
    if not pdf_bytes:
        return "", []
    from pdfminer.high_level import extract_text

    from researchpapers.url_extract import extract_urls_from_text

    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="rp_ex_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(pdf_bytes)
        try:
            text = extract_text(path) or ""
        except Exception:  # noqa: BLE001 — pdfminer raises a broad family on malformed PDFs
            text = ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    urls = extract_urls_from_text(text) if text else []
    return text, urls


def _persist_result(
    conn: Any,
    arxiv_id: str,
    text: str,
    urls: list[tuple[str, str, str, str, str]],
    counters: dict[str, int],
) -> None:
    gz = gzip.compress(text.encode("utf-8")) if text else b""
    if not text:
        counters["empty_text"] += 1
    with conn.cursor() as cur:
        for raw, canonical, scheme, host, ctx in urls:
            cur.execute(
                """
                INSERT INTO references_url
                    (citing_arxiv_id, url_raw, url_canonical, scheme, host, context_snippet)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (citing_arxiv_id, url_canonical) DO NOTHING
                """,
                (arxiv_id, raw, canonical, scheme, host, ctx),
            )
            if cur.rowcount > 0:
                counters["urls_inserted"] += 1
        if gz:
            cur.execute(
                """
                INSERT INTO paper_texts (arxiv_id, content_gz, content_chars)
                VALUES (%s, %s, %s)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    content_gz    = EXCLUDED.content_gz,
                    content_chars = EXCLUDED.content_chars,
                    extracted_at  = now()
                """,
                (arxiv_id, gz, len(text)),
            )
        now = datetime.now(UTC)
        cur.execute(
            """
            UPDATE papers SET
                pdf_fetched_at    = COALESCE(pdf_fetched_at, %s),
                urls_extracted_at = %s
            WHERE arxiv_id = %s
            """,
            (now, now, arxiv_id),
        )
    conn.commit()
    counters["processed"] += 1


def _mark_failed(conn: Any, arxiv_id: str, counters: dict[str, int]) -> None:
    """Mark a paper as attempted (download failed) so the next run doesn't retry it forever."""
    counters["pdf_failed"] += 1
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE papers SET urls_extracted_at = %s WHERE arxiv_id = %s",
            (datetime.now(UTC), arxiv_id),
        )
    conn.commit()
    counters["processed"] += 1


def ingest_all(
    settings: Settings,
    *,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
) -> dict[str, int]:
    """Streams through every paper without urls_extracted_at. Returns counters."""
    counters: dict[str, int] = {
        "processed": 0,
        "pdf_failed": 0,
        "empty_text": 0,
        "urls_inserted": 0,
    }
    last_request_at = 0.0
    with build_client(settings, timeout=120.0) as client, connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT arxiv_id FROM papers
                WHERE urls_extracted_at IS NULL
                ORDER BY citation_count DESC NULLS LAST, submitted_date DESC NULLS LAST
                """
                + (f" LIMIT {int(limit)}" if limit else "")
            )
            arxiv_ids = [r["arxiv_id"] for r in cur.fetchall()]

        log.info("ingest queue: %d papers, %d workers, max_in_flight=%d",
                 len(arxiv_ids), workers, max_in_flight)

        in_flight: dict[Future, str] = {}
        idx = 0

        with ProcessPoolExecutor(max_workers=workers) as pool:
            while idx < len(arxiv_ids) or in_flight:
                # Top up the in-flight queue with new downloads, rate-limited.
                while idx < len(arxiv_ids) and len(in_flight) < max_in_flight:
                    arxiv_id = arxiv_ids[idx]
                    idx += 1
                    elapsed = time.monotonic() - last_request_at
                    if elapsed < REQUEST_INTERVAL_SECONDS:
                        time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
                    try:
                        pdf_bytes = _download_bytes(client, _pdf_url_for(arxiv_id))
                    except Exception as e:  # noqa: BLE001
                        log.warning("download failed for %s: %s", arxiv_id, e)
                        last_request_at = time.monotonic()
                        _mark_failed(conn, arxiv_id, counters)
                        if counters["processed"] % LOG_EVERY == 0:
                            _log_progress(counters, len(arxiv_ids))
                        continue
                    last_request_at = time.monotonic()
                    fut = pool.submit(_extract_worker, pdf_bytes)
                    in_flight[fut] = arxiv_id

                # Wait for at least one extraction to finish.
                if in_flight:
                    done, _pending = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        arxiv_id = in_flight.pop(fut)
                        try:
                            text, urls = fut.result()
                        except Exception as e:  # noqa: BLE001 — worker failure
                            log.warning("extract failed for %s: %s", arxiv_id, e)
                            text, urls = "", []
                        try:
                            _persist_result(conn, arxiv_id, text, urls, counters)
                        except Exception as e:  # noqa: BLE001 — bad URL, oversized text, etc.
                            log.warning("persist failed for %s: %s", arxiv_id, e)
                            conn.rollback()
                            _mark_failed(conn, arxiv_id, counters)
                        if counters["processed"] % LOG_EVERY == 0:
                            _log_progress(counters, len(arxiv_ids))

    return counters


def _log_progress(counters: dict[str, int], total: int) -> None:
    log.info(
        "progress: %d/%d done, %d url edges, %d pdf failures, %d empty",
        counters["processed"],
        total,
        counters["urls_inserted"],
        counters["pdf_failed"],
        counters["empty_text"],
    )
