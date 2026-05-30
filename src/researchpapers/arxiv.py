"""Fetch metadata from the arXiv Atom API and upsert into the papers table.

arXiv's API policy: at most one request every 3 seconds, and identify yourself
in the User-Agent. Page size is capped at 2000 results per request. We page
through with start/max_results.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime

import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from researchpapers.config import Settings
from researchpapers.db import connect
from researchpapers.http import build_client

ARXIV_API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 200  # smaller pages = more progress visibility, well below the 2000 cap
REQUEST_INTERVAL_SECONDS = 3.0


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str | None
    categories: list[str]
    submitted_date: date | None
    updated_date: date | None
    authors: list[dict[str, str]]
    pdf_url: str | None


_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([^v\s]+)(?:v\d+)?")


def _parse_arxiv_id(entry_id: str) -> str:
    m = _ARXIV_ID_RE.search(entry_id)
    if not m:
        raise ValueError(f"could not parse arXiv id from {entry_id!r}")
    return m.group(1)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).date()


def _entry_to_paper(entry: dict) -> Paper:
    pdf_url = next(
        (link["href"] for link in entry.get("links", []) if link.get("type") == "application/pdf"),
        None,
    )
    primary_category = (entry.get("arxiv_primary_category") or {}).get("term")
    categories = [tag["term"] for tag in entry.get("tags", [])]
    return Paper(
        arxiv_id=_parse_arxiv_id(entry["id"]),
        title=" ".join(entry["title"].split()),
        abstract=" ".join(entry.get("summary", "").split()),
        primary_category=primary_category,
        categories=categories,
        submitted_date=_parse_date(entry.get("published")),
        updated_date=_parse_date(entry.get("updated")),
        authors=[{"name": a["name"]} for a in entry.get("authors", [])],
        pdf_url=pdf_url,
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
def _fetch_page(client: httpx.Client, search_query: str, start: int) -> list[dict]:
    resp = client.get(
        ARXIV_API_URL,
        params={
            "search_query": search_query,
            "start": start,
            "max_results": PAGE_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
    )
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    return feed.entries


def _build_query(category: str, since: date, until: date) -> str:
    # arXiv's query language uses YYYYMMDDHHMM ranges. Inclusive on both sides.
    since_s = since.strftime("%Y%m%d") + "0000"
    until_s = until.strftime("%Y%m%d") + "2359"
    return f"cat:{category} AND submittedDate:[{since_s} TO {until_s}]"


def fetch_papers(settings: Settings, *, category: str, since: date, until: date) -> int:
    """Hits the arXiv API for the given category + window. Returns the count of new/updated rows."""
    query = _build_query(category, since, until)
    upserted = 0
    last_request_at = 0.0
    with build_client(settings) as client, connect(settings) as conn:
        start = 0
        while True:
            elapsed = time.monotonic() - last_request_at
            if elapsed < REQUEST_INTERVAL_SECONDS:
                time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
            entries = _fetch_page(client, query, start)
            last_request_at = time.monotonic()
            if not entries:
                break
            papers = [_entry_to_paper(e) for e in entries]
            with conn.cursor() as cur:
                for p in papers:
                    cur.execute(
                        """
                        INSERT INTO papers (
                            arxiv_id, title, abstract, primary_category, categories,
                            submitted_date, updated_date, authors_json, pdf_url
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (arxiv_id) DO UPDATE SET
                            title             = EXCLUDED.title,
                            abstract          = EXCLUDED.abstract,
                            primary_category  = EXCLUDED.primary_category,
                            categories        = EXCLUDED.categories,
                            updated_date      = EXCLUDED.updated_date,
                            authors_json      = EXCLUDED.authors_json,
                            pdf_url           = EXCLUDED.pdf_url
                        """,
                        (
                            p.arxiv_id,
                            p.title,
                            p.abstract,
                            p.primary_category,
                            p.categories,
                            p.submitted_date,
                            p.updated_date,
                            json.dumps(p.authors),
                            p.pdf_url,
                        ),
                    )
                    upserted += 1
            conn.commit()
            if len(entries) < PAGE_SIZE:
                break
            start += PAGE_SIZE
    return upserted
