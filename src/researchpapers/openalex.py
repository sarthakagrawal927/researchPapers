"""OpenAlex selector — picks the top-N CS papers by citation count that have an arXiv preprint.

OpenAlex is free, requires no key, and is generous on rate limits when you
include your email in the `mailto` param ("polite pool"). We page with cursor
because deep offset pagination is disallowed past 10k.

Field id 17 = Computer Science. arXiv's OpenAlex source id is S4306400194.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from datetime import date, datetime
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from researchpapers.config import Settings
from researchpapers.db import connect
from researchpapers.http import build_client

log = logging.getLogger("researchpapers.openalex")

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
ARXIV_SOURCE_ID = "S4306400194"  # arXiv
CS_FIELD_ID = "17"               # Computer Science
PAGE_SIZE = 200
POLITE_INTERVAL_SECONDS = 0.5    # polite pool tolerates ~10 RPS; 2 RPS is safe and visible

_ARXIV_LANDING_RE = re.compile(r"arxiv\.org/abs/([^v\s/?#]+)(?:v\d+)?", re.IGNORECASE)


def _arxiv_id_from_work(work: dict[str, Any]) -> str | None:
    """Pulls the arxiv id from any landing_page_url, or from the doi if it is an arXiv doi."""
    for loc in work.get("locations") or []:
        url = loc.get("landing_page_url") or ""
        m = _ARXIV_LANDING_RE.search(url)
        if m:
            return m.group(1)
    doi = (work.get("doi") or "").lower()
    # arXiv-issued DOIs look like 10.48550/arxiv.<id>
    m = re.match(r"https?://(?:dx\.)?doi\.org/10\.48550/arxiv\.(\S+)", doi)
    if m:
        return m.group(1)
    return None


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _authors(work: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for a in work.get("authorships") or []:
        au = a.get("author") or {}
        if name := au.get("display_name"):
            out.append({"name": name})
    return out


def _tags(work: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten OpenAlex topics into a list of {name, level, score}.

    OpenAlex topic hierarchy: domain > field > subfield > topic. We capture each level so
    callers can filter at any granularity (e.g. all NLP papers vs all CS).
    """
    out: list[dict[str, Any]] = []
    for t in work.get("topics") or []:
        score = t.get("score")
        if name := t.get("display_name"):
            out.append({"name": name, "level": "topic", "score": score})
        for level in ("subfield", "field", "domain"):
            block = t.get(level) or {}
            if name := block.get("display_name"):
                out.append({"name": name, "level": level, "score": score})
    return out


def _keywords(work: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in work.get("keywords") or []:
        if name := k.get("display_name"):
            out.append({"name": name, "score": k.get("score")})
    return out


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def _fetch_page(client, cursor: str, *, cs_only: bool = True) -> dict[str, Any]:
    if cs_only:
        flt = (
            f"primary_topic.field.id:fields/{CS_FIELD_ID},"
            f"locations.source.id:{ARXIV_SOURCE_ID}"
        )
    else:
        flt = f"locations.source.id:{ARXIV_SOURCE_ID}"
    resp = client.get(
        OPENALEX_WORKS_URL,
        params={
            "filter": flt,
            "sort": "cited_by_count:desc",
            "per-page": PAGE_SIZE,
            "cursor": cursor,
            "select": (
                "id,doi,title,abstract_inverted_index,"
                "cited_by_count,publication_date,authorships,"
                "primary_topic,topics,keywords,concepts,"
                "locations,referenced_works"
            ),
        },
    )
    resp.raise_for_status()
    return resp.json()


def _reconstruct_abstract(inv: dict[str, list[int]] | None) -> str | None:
    """OpenAlex returns abstracts as inverted indexes for licensing reasons. Reverse it."""
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _iter_works(settings: Settings, target: int, *, cs_only: bool = True) -> Iterator[dict[str, Any]]:
    """Page through Works sorted by citation count. Yields raw work objects."""
    cursor = "*"
    seen = 0
    last_request_at = 0.0
    with build_client(settings) as client:
        # Polite pool: tell OpenAlex who's calling.
        client.params = {"mailto": settings.contact_email}
        while seen < target:
            elapsed = time.monotonic() - last_request_at
            if elapsed < POLITE_INTERVAL_SECONDS:
                time.sleep(POLITE_INTERVAL_SECONDS - elapsed)
            page = _fetch_page(client, cursor, cs_only=cs_only)
            last_request_at = time.monotonic()
            results = page.get("results") or []
            if not results:
                return
            for w in results:
                yield w
                seen += 1
                if seen >= target:
                    return
            cursor = (page.get("meta") or {}).get("next_cursor")
            if not cursor:
                return


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def _fetch_works_by_oa_ids(client, oa_ids: list[str]) -> list[dict[str, Any]]:
    short_ids = [i.rsplit("/", 1)[-1] for i in oa_ids]
    resp = client.get(
        OPENALEX_WORKS_URL,
        params={
            "filter": f"ids.openalex:{'|'.join(short_ids)}",
            "per-page": len(short_ids),
            "select": "id,referenced_works",
        },
    )
    resp.raise_for_status()
    return (resp.json() or {}).get("results", [])


def backfill_referenced_works(
    settings: Settings, *, batch_size: int = 50, limit: int | None = None
) -> tuple[int, int]:
    """Pulls referenced_works for papers that haven't been backfilled yet (incremental).

    Skips any paper with `references_backfilled_at IS NOT NULL`. The cited side is stored
    as a bare OpenAlex ID (resolved to titles in a follow-up pass).
    """
    papers_done = 0
    edges_written = 0
    last_request_at = 0.0
    with build_client(settings) as client, connect(settings) as conn:
        client.params = {"mailto": settings.contact_email}
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT arxiv_id, openalex_id FROM papers
                WHERE openalex_id IS NOT NULL
                  AND references_backfilled_at IS NULL
                """
                + (f" LIMIT {int(limit)}" if limit else "")
            )
            rows = cur.fetchall()
        oa_to_arxiv = {r["openalex_id"]: r["arxiv_id"] for r in rows}
        oa_ids = list(oa_to_arxiv)
        if not oa_ids:
            log.info("backfill: nothing to do (all caught up)")
            return papers_done, edges_written
        log.info("backfill queue: %d new papers", len(oa_ids))
        for i in range(0, len(oa_ids), batch_size):
            elapsed = time.monotonic() - last_request_at
            if elapsed < POLITE_INTERVAL_SECONDS:
                time.sleep(POLITE_INTERVAL_SECONDS - elapsed)
            chunk = oa_ids[i : i + batch_size]
            works = _fetch_works_by_oa_ids(client, chunk)
            last_request_at = time.monotonic()
            with conn.cursor() as cur:
                for w in works:
                    citing_oa = w.get("id")
                    citing_arxiv = oa_to_arxiv.get(citing_oa)
                    if not citing_arxiv:
                        continue
                    for ref in w.get("referenced_works") or []:
                        cur.execute(
                            """
                            INSERT INTO references_paper
                                (citing_arxiv_id, cited_openalex_id)
                            VALUES (%s, %s)
                            ON CONFLICT (citing_arxiv_id, cited_openalex_id)
                                WHERE cited_openalex_id IS NOT NULL
                                DO NOTHING
                            """,
                            (citing_arxiv, ref),
                        )
                        if cur.rowcount > 0:
                            edges_written += 1
                    cur.execute(
                        "UPDATE papers SET references_backfilled_at = now() WHERE arxiv_id = %s",
                        (citing_arxiv,),
                    )
                    papers_done += 1
            conn.commit()
            if papers_done % 1000 == 0:
                log.info("backfill referenced_works: %d/%d papers", papers_done, len(oa_ids))
    return papers_done, edges_written


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def _fetch_works_full(client, oa_ids: list[str]) -> list[dict[str, Any]]:
    short_ids = [i.rsplit("/", 1)[-1] for i in oa_ids]
    resp = client.get(
        OPENALEX_WORKS_URL,
        params={
            "filter": f"ids.openalex:{'|'.join(short_ids)}",
            "per-page": len(short_ids),
            "select": (
                "id,title,doi,cited_by_count,publication_year,primary_topic,locations"
            ),
        },
    )
    resp.raise_for_status()
    return (resp.json() or {}).get("results", [])


def resolve_top_cited(settings: Settings, *, top: int = 200) -> int:
    """Resolves the top-N most-frequently cited OpenAlex IDs in our references_paper into cited_works.

    Cheap because we only resolve the head, not the tail. Top 200 is enough for the leaderboard.
    """
    resolved = 0
    last_request_at = 0.0
    with build_client(settings) as client, connect(settings) as conn:
        client.params = {"mailto": settings.contact_email}
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cited_openalex_id, COUNT(DISTINCT citing_arxiv_id) AS n
                FROM references_paper
                WHERE cited_openalex_id IS NOT NULL
                GROUP BY cited_openalex_id
                ORDER BY n DESC
                LIMIT %s
                """,
                (top,),
            )
            oa_ids = [r["cited_openalex_id"] for r in cur.fetchall()]
        for i in range(0, len(oa_ids), 50):
            elapsed = time.monotonic() - last_request_at
            if elapsed < POLITE_INTERVAL_SECONDS:
                time.sleep(POLITE_INTERVAL_SECONDS - elapsed)
            chunk = oa_ids[i : i + 50]
            works = _fetch_works_full(client, chunk)
            last_request_at = time.monotonic()
            with conn.cursor() as cur:
                for w in works:
                    arxiv_id = _arxiv_id_from_work(w)
                    cur.execute(
                        """
                        INSERT INTO cited_works
                            (openalex_id, title, cited_by_count, doi, publication_year,
                             arxiv_id, primary_topic)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (openalex_id) DO UPDATE SET
                            title           = EXCLUDED.title,
                            cited_by_count  = EXCLUDED.cited_by_count,
                            doi             = EXCLUDED.doi,
                            publication_year = EXCLUDED.publication_year,
                            arxiv_id        = EXCLUDED.arxiv_id,
                            primary_topic   = EXCLUDED.primary_topic,
                            resolved_at     = now()
                        """,
                        (
                            w.get("id"),
                            w.get("title"),
                            w.get("cited_by_count"),
                            w.get("doi"),
                            w.get("publication_year"),
                            arxiv_id,
                            ((w.get("primary_topic") or {}).get("display_name")),
                        ),
                    )
                    resolved += 1
            conn.commit()
    return resolved


def select_top(settings: Settings, *, n: int, cs_only: bool = True) -> tuple[int, int]:
    """Selects up to n top-cited CS papers that have an arXiv preprint. Returns (scanned, inserted)."""
    scanned = 0
    inserted = 0
    # Scan a bit more than n so we still land near n after dropping non-arxiv-resolvable rows.
    # Empirically about 90% of CS Works with arXiv-source location resolve cleanly, but cap the
    # over-scan to keep the run bounded.
    target_scan = int(n * 1.25)
    with connect(settings) as conn:
        for w in _iter_works(settings, target_scan, cs_only=cs_only):
            scanned += 1
            arxiv_id = _arxiv_id_from_work(w)
            if not arxiv_id:
                continue
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO papers (
                        arxiv_id, openalex_id, doi, title, abstract,
                        citation_count, primary_category,
                        submitted_date, authors_json, tags_json, keywords_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                    ON CONFLICT (arxiv_id) DO UPDATE SET
                        openalex_id    = EXCLUDED.openalex_id,
                        doi            = COALESCE(EXCLUDED.doi, papers.doi),
                        citation_count = EXCLUDED.citation_count,
                        abstract       = COALESCE(EXCLUDED.abstract, papers.abstract),
                        authors_json   = EXCLUDED.authors_json,
                        tags_json      = EXCLUDED.tags_json,
                        keywords_json  = EXCLUDED.keywords_json
                    """,
                    (
                        arxiv_id,
                        w.get("id"),
                        w.get("doi"),
                        w.get("title") or "(untitled)",
                        _reconstruct_abstract(w.get("abstract_inverted_index")),
                        w.get("cited_by_count") or 0,
                        ((w.get("primary_topic") or {}).get("field") or {}).get("display_name"),
                        _parse_date(w.get("publication_date")),
                        json.dumps(_authors(w)),
                        json.dumps(_tags(w)),
                        json.dumps(_keywords(w)),
                    ),
                )
                # We won't have a pdf_url from OpenAlex; the ingest stage derives it from arxiv_id.
            conn.commit()
            inserted += 1
            if inserted >= n:
                break
    return scanned, inserted
