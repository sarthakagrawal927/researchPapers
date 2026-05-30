"""Extract URLs from downloaded PDFs and persist them to references_url.

PDF text is noisy: URLs wrap across lines (with or without a hyphen),
pick up trailing punctuation, and often run into the next word with no
whitespace. We do three passes:

1. De-hyphenate soft line breaks (`-\n`) and hard line breaks inside URLs.
2. Pull candidate URLs with a permissive regex.
3. Canonicalize each candidate (trim trailing punctuation, validate via
   urllib.parse, walk host right-to-left for the longest valid public-suffix
   match, drop tracking params).

The `extract_all` path reads PDFs from disk (used at ingest time).
The `re_extract_all_from_text` path reads gzipped text from paper_texts
(used after ingest, to iterate on URL canonicalization without re-downloading).
"""

from __future__ import annotations

import gzip
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import tldextract
from pdfminer.high_level import extract_text

from researchpapers.config import Settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.url_extract")

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?)]}>\"'`"
_CONTEXT_RADIUS = 80
# Postgres B-tree index can't hold values past ~2700 bytes, and any URL longer
# than this in our data is text-extraction noise anyway.
MAX_URL_LENGTH = 500

# When the rightmost host components don't form a valid public suffix (e.g. PDF
# text bled "scikit-learn.org.Keywords:" into a URL), we walk right-to-left to
# find the longest valid TLD and truncate everything after it.
_HOST_TRAILING_TRIM = re.compile(r"[^A-Za-z0-9.-]")

# Common tracking params to strip from the canonical form. Conservative — many
# academic links carry meaningful query params, so we only drop the obvious ones.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "ref_src", "ref_url", "gclid", "fbclid",
}


def _dehyphenate(text: str) -> str:
    """Join soft-hyphen line breaks and hard line breaks that fall inside a URL."""
    # `http-\nfoo` -> `httpfoo` (hyphen was the line-break artifact, not part of the URL)
    text = re.sub(r"-\n", "", text)
    # `https://foo.com/\nbar` -> `https://foo.com/bar`
    text = re.sub(r"(https?://\S*?)\n(\S+)", r"\1\2", text, flags=re.IGNORECASE)
    return text


def _normalize_host(raw_netloc: str) -> str | None:
    """Walks the host right-to-left and returns the longest prefix that has a valid public suffix.

    Returns None if no segment of the host has a recognized TLD. Drops anything past the suffix
    (e.g. 'scikit-learn.org.Keywords' -> 'scikit-learn.org', 'blog.and' -> None).
    """
    host = _HOST_TRAILING_TRIM.sub("", raw_netloc).strip(".").lower()
    if not host:
        return None
    parts = host.split(".")
    for drop in range(len(parts)):
        candidate = ".".join(parts[: len(parts) - drop])
        if not candidate or "." not in candidate:
            break
        extracted = tldextract.extract(candidate)
        if extracted.suffix and extracted.domain:
            return ".".join(
                p for p in (extracted.subdomain, extracted.domain, extracted.suffix) if p
            )
    return None


def _canonicalize(url: str) -> tuple[str, str, str] | None:
    """Returns (canonical_url, scheme, host) or None if the URL is invalid / not a real web URL."""
    cleaned = url.rstrip(_TRAILING_PUNCT)
    if len(cleaned) > MAX_URL_LENGTH:
        return None
    try:
        parsed = urlparse(cleaned)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    host = _normalize_host(parsed.netloc)
    if host is None:
        return None
    # If host normalization dropped trailing components, the rest of the URL beyond that point
    # was almost certainly bled-in text and shouldn't be kept either.
    netloc_lower = parsed.netloc.lower()
    if not netloc_lower.startswith(host):
        # The normalized host is a prefix of the raw netloc minus the bled-in noise.
        # Drop path/query/fragment because they came from after the noise boundary.
        return f"{parsed.scheme}://{host}", parsed.scheme, host
    # Drop the tracking params, keep everything else.
    if parsed.query:
        kept = "&".join(
            kv for kv in parsed.query.split("&")
            if "=" in kv and kv.split("=", 1)[0].lower() not in _TRACKING_PARAMS
        )
    else:
        kept = ""
    canonical = f"{parsed.scheme}://{host}{parsed.path}"
    if kept:
        canonical += f"?{kept}"
    if parsed.fragment:
        canonical += f"#{parsed.fragment}"
    return canonical, parsed.scheme, host


def extract_urls_from_text(text: str) -> list[tuple[str, str, str, str, str]]:
    """Returns (url_raw, url_canonical, scheme, host, context_snippet) tuples, deduped by canonical."""
    text = _dehyphenate(text)
    seen: dict[str, tuple[str, str, str, str, str]] = {}
    for m in _URL_RE.finditer(text):
        raw = m.group(0)
        canonical = _canonicalize(raw)
        if canonical is None:
            continue
        url_canonical, scheme, host = canonical
        if url_canonical in seen:
            continue
        ctx_start = max(0, m.start() - _CONTEXT_RADIUS)
        ctx_end = min(len(text), m.end() + _CONTEXT_RADIUS)
        ctx = " ".join(text[ctx_start:ctx_end].split())
        seen[url_canonical] = (raw, url_canonical, scheme, host, ctx)
    return list(seen.values())


def extract_urls_from_pdf(pdf_path: Path) -> list[tuple[str, str, str, str, str]]:
    text = extract_text(str(pdf_path))
    return extract_urls_from_text(text)


def re_extract_all_from_text(
    settings: Settings, *, limit: int | None = None, replace: bool = True
) -> tuple[int, int]:
    """Re-runs URL extraction from gzipped text in paper_texts. No PDF re-download.

    If `replace=True`, deletes existing references_url rows per paper before re-inserting.
    Use this after ingest completes to clean up the URL data with a refined canonicalizer.
    """
    papers_processed = 0
    urls_inserted = 0
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT arxiv_id, content_gz FROM paper_texts
                ORDER BY content_chars DESC
                """
                + (f" LIMIT {int(limit)}" if limit else "")
            )
            rows = cur.fetchall()
        log.info("re-extract queue: %d papers", len(rows))
        for row in rows:
            arxiv_id = row["arxiv_id"]
            try:
                text = gzip.decompress(row["content_gz"]).decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            urls = extract_urls_from_text(text)
            with conn.cursor() as cur:
                if replace:
                    cur.execute(
                        "DELETE FROM references_url WHERE citing_arxiv_id = %s", (arxiv_id,)
                    )
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
                        urls_inserted += 1
            conn.commit()
            papers_processed += 1
            if papers_processed % 500 == 0:
                log.info("re-extract progress: %d/%d", papers_processed, len(rows))
    return papers_processed, urls_inserted


def extract_all(settings: Settings, *, limit: int | None = None) -> tuple[int, int]:
    """Extracts URLs from all unprocessed PDFs. Returns (papers_processed, urls_inserted)."""
    papers_processed = 0
    urls_inserted = 0
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT arxiv_id, pdf_path FROM papers
                WHERE pdf_path IS NOT NULL AND urls_extracted_at IS NULL
                ORDER BY submitted_date DESC NULLS LAST
                """
                + (f" LIMIT {int(limit)}" if limit else "")
            )
            rows = cur.fetchall()
        for row in rows:
            arxiv_id = row["arxiv_id"]
            pdf_path = Path(row["pdf_path"])
            if not pdf_path.exists():
                continue
            try:
                urls = extract_urls_from_pdf(pdf_path)
            except Exception:  # noqa: BLE001 — pdfminer raises a broad family on malformed PDFs
                # Mark as processed so we don't keep retrying; the count just stays at 0.
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE papers SET urls_extracted_at = %s WHERE arxiv_id = %s",
                        (datetime.now(UTC), arxiv_id),
                    )
                conn.commit()
                papers_processed += 1
                continue
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
                        urls_inserted += 1
                cur.execute(
                    "UPDATE papers SET urls_extracted_at = %s WHERE arxiv_id = %s",
                    (datetime.now(UTC), arxiv_id),
                )
            conn.commit()
            papers_processed += 1
    return papers_processed, urls_inserted
