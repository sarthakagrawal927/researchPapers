"""bioRxiv / medRxiv / chemRxiv ingest into ClickHouse.

All three preprint servers expose JSON APIs with similar shapes:
  bioRxiv:  https://api.biorxiv.org/details/biorxiv/{from}/{to}/{cursor}
  medRxiv:  https://api.biorxiv.org/details/medrxiv/{from}/{to}/{cursor}
  chemRxiv: https://chemrxiv.org/engage/chemrxiv/public-api/v1/items?term=&skip=...&limit=...

Each call returns metadata for ~100 papers. Pagination is via cursor (bio/med) or skip (chem).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from researchpapers.ch_db import connect as ch_connect

log = logging.getLogger("researchpapers.biorxiv_ingest")

POLITE_INTERVAL = 0.3  # seconds between API calls
DEFAULT_USER_AGENT = "researchpapers/0.1 (sarthakagrawal927@gmail.com)"


def _client() -> httpx.Client:
    return httpx.Client(timeout=60.0, headers={"User-Agent": DEFAULT_USER_AGENT})


def _make_paper_id(server: str, doi_or_id: str) -> str:
    return f"{server}:{doi_or_id}"


def _coerce_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            return None


# ---------- bioRxiv / medRxiv (api.biorxiv.org) ----------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.RemoteProtocolError)),
)
def _fetch_one_page(client: httpx.Client, url: str) -> dict:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.json() or {}


def _bio_med_pages(client: httpx.Client, server: str, since: date, until: date) -> Iterator[list[dict]]:
    """Yields successive pages from bioRxiv/medRxiv. ~30 papers per page."""
    cursor = 0
    last_request_at = 0.0
    while True:
        elapsed = time.monotonic() - last_request_at
        if elapsed < POLITE_INTERVAL:
            time.sleep(POLITE_INTERVAL - elapsed)
        url = f"https://api.biorxiv.org/details/{server}/{since.isoformat()}/{until.isoformat()}/{cursor}"
        try:
            body = _fetch_one_page(client, url)
        except Exception as e:  # noqa: BLE001 — give up on this page after retries, log + stop
            log.warning("giving up at cursor=%d after retries: %s", cursor, e)
            return
        last_request_at = time.monotonic()
        coll = body.get("collection") or []
        if not coll:
            return
        yield coll
        msg = body.get("messages") or [{}]
        total = int(msg[0].get("total", 0) or 0)
        cursor += len(coll)
        if cursor >= total:
            return


def _to_paper_row(server: str, item: dict) -> list:
    pid = _make_paper_id(server, item.get("doi") or item.get("id") or "")
    sub_date = _coerce_date(item.get("date"))
    authors = [a.strip() for a in (item.get("authors") or "").split(";") if a.strip()]
    title = item.get("title") or ""
    abstract = item.get("abstract")
    return [
        pid,
        server,
        item.get("doi") or item.get("id") or "",
        None,
        None,
        item.get("doi"),
        title,
        abstract,
        sub_date,
        sub_date.year if sub_date else None,
        0,
        item.get("category"),
        authors,
        [],
        [],
        None, None, None, None, 0,
        datetime.now(UTC),
        datetime.now(UTC),
        [],
    ]


PAPER_COLS = [
    "paper_id", "source", "source_id", "arxiv_id", "openalex_id",
    "doi", "title", "abstract", "submitted_date", "publication_year",
    "citation_count", "primary_category", "authors",
    "openalex_tags", "openalex_keywords",
    "pagerank_score", "katz_score", "community_id", "semantic_cluster",
    "in_corpus_degree", "ingested_at", "updated_at", "abstract_embedding",
]


def ingest_biorxiv_medrxiv(
    server: str = "biorxiv",
    since: date | None = None,
    until: date | None = None,
) -> int:
    """`server` is 'biorxiv' or 'medrxiv'."""
    until = until or date.today()
    since = since or (until - timedelta(days=365))
    log.info("ingesting %s %s..%s", server, since, until)
    n = 0
    batch: list[list] = []
    ch = ch_connect()
    try:
        with _client() as client:
            for page in _bio_med_pages(client, server, since, until):
                for item in page:
                    batch.append(_to_paper_row(server, item))
                    if len(batch) >= 500:
                        ch.insert("papers", batch, column_names=PAPER_COLS)
                        n += len(batch)
                        batch = []
                        if n % 2000 == 0:
                            log.info("%s: %d papers", server, n)
            if batch:
                ch.insert("papers", batch, column_names=PAPER_COLS)
                n += len(batch)
    finally:
        ch.close()
    log.info("%s: %d papers ingested", server, n)
    return n


# ---------- chemRxiv (different API) ----------

def ingest_chemrxiv(limit: int | None = None, page_size: int = 50) -> int:
    log.info("ingesting chemrxiv")
    n = 0
    batch: list[list] = []
    last_request_at = 0.0
    skip = 0
    ch = ch_connect()
    try:
        with _client() as client:
            while True:
                elapsed = time.monotonic() - last_request_at
                if elapsed < POLITE_INTERVAL:
                    time.sleep(POLITE_INTERVAL - elapsed)
                url = (
                    f"https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
                    f"?term=&skip={skip}&limit={page_size}"
                )
                resp = client.get(url)
                last_request_at = time.monotonic()
                if resp.status_code != 200:
                    log.warning("chemrxiv non-200: %d", resp.status_code)
                    break
                body = resp.json() or {}
                items = body.get("itemHits") or []
                if not items:
                    break
                for hit in items:
                    item = hit.get("item") or {}
                    doi = item.get("doi") or item.get("id") or ""
                    sub_date = _coerce_date(item.get("publishedDate") or item.get("submittedDate"))
                    authors = [
                        f"{a.get('firstName','').strip()} {a.get('lastName','').strip()}".strip()
                        for a in (item.get("authors") or [])
                    ]
                    authors = [a for a in authors if a]
                    pid = _make_paper_id("chemrxiv", doi)
                    batch.append([
                        pid, "chemrxiv", doi, None, None, doi,
                        item.get("title") or "",
                        item.get("abstract"),
                        sub_date,
                        sub_date.year if sub_date else None,
                        0,
                        ((item.get("category") or {}).get("name")),
                        authors,
                        [], [], None, None, None, None, 0,
                        datetime.now(UTC), datetime.now(UTC), [],
                    ])
                    if len(batch) >= 500:
                        ch.insert("papers", batch, column_names=PAPER_COLS)
                        n += len(batch)
                        batch = []
                skip += len(items)
                if limit and n >= limit:
                    break
                if len(items) < page_size:
                    break
            if batch:
                ch.insert("papers", batch, column_names=PAPER_COLS)
                n += len(batch)
    finally:
        ch.close()
    log.info("chemrxiv: %d papers ingested", n)
    return n
