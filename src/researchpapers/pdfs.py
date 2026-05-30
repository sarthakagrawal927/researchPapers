"""Polite arXiv PDF downloader. Idempotent — re-runs only fetch what's missing."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from tenacity import retry, stop_after_attempt, wait_exponential

from researchpapers.config import PDF_DIR, Settings
from researchpapers.db import connect
from researchpapers.http import build_client

REQUEST_INTERVAL_SECONDS = 3.0


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
def _download(client, url: str) -> bytes:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.content


def download_pdfs(settings: Settings, *, limit: int | None = None) -> int:
    """Downloads PDFs for papers with pdf_fetched_at IS NULL. Returns count downloaded."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    last_request_at = 0.0
    with build_client(settings, timeout=120.0) as client, connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT arxiv_id, pdf_url FROM papers
                WHERE pdf_fetched_at IS NULL AND pdf_url IS NOT NULL
                ORDER BY submitted_date DESC NULLS LAST
                """
                + (f" LIMIT {int(limit)}" if limit else "")
            )
            rows = cur.fetchall()
        for row in rows:
            arxiv_id = row["arxiv_id"]
            pdf_url = row["pdf_url"]
            target = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"
            if target.exists():
                # File on disk but DB doesn't know; reconcile and skip.
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE papers SET pdf_path = %s, pdf_fetched_at = %s WHERE arxiv_id = %s",
                        (str(target), datetime.now(UTC), arxiv_id),
                    )
                conn.commit()
                continue
            elapsed = time.monotonic() - last_request_at
            if elapsed < REQUEST_INTERVAL_SECONDS:
                time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
            content = _download(client, pdf_url)
            last_request_at = time.monotonic()
            target.write_bytes(content)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE papers SET pdf_path = %s, pdf_fetched_at = %s WHERE arxiv_id = %s",
                    (str(target), datetime.now(UTC), arxiv_id),
                )
            conn.commit()
            downloaded += 1
    return downloaded
