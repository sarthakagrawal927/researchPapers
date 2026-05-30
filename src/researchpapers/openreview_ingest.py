"""OpenReview ingest: NeurIPS / ICLR / ICML / COLM submissions + reviews + decisions.

Writes directly to ClickHouse. Two tables touched:
  papers              — one row per submission, source='openreview'
  openreview_reviews  — one row per review with scores + text

OpenReview API v2 is used for venues 2023+. Earlier venues live on v1 API.

Polite scraping: openreview.net has its own polite-pool conventions. We default to 0.5s
between batched queries.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Iterator

import openreview
import openreview.api

from researchpapers.ch_db import connect as ch_connect

log = logging.getLogger("researchpapers.openreview_ingest")

# Venues to ingest. Format: (display_name, venue_id, year)
DEFAULT_VENUES = [
    ("NeurIPS-2024", "NeurIPS.cc/2024/Conference", 2024),
    ("NeurIPS-2023", "NeurIPS.cc/2023/Conference", 2023),
    ("ICLR-2025",    "ICLR.cc/2025/Conference",    2025),
    ("ICLR-2024",    "ICLR.cc/2024/Conference",    2024),
    ("ICML-2024",    "ICML.cc/2024/Conference",    2024),
    # COLM venue id format unclear; verify before adding.
]


def _client() -> openreview.api.OpenReviewClient:
    return openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")


def _make_paper_id(submission_id: str) -> str:
    return f"openreview:{submission_id}"


def _flatten_authors(content: dict) -> list[str]:
    auths = (content.get("authors") or {}).get("value") or []
    return [str(a) for a in auths if a]


def _coerce_datetime(ts_ms: int | None) -> datetime | None:
    """ms-since-epoch → naive UTC datetime (ClickHouse DateTime expects naive)."""
    if not ts_ms:
        return None
    try:
        return datetime.utcfromtimestamp(ts_ms / 1000)
    except Exception:
        return None


def _coerce_date(ts_ms: int | None) -> date | None:
    """ms-since-epoch → date."""
    dt = _coerce_datetime(ts_ms)
    return dt.date() if dt else None


def _venue_papers(client, venue_id: str, year: int) -> Iterator[dict]:
    """Yield (submission_note, [review_notes], decision_note_or_None) per paper."""
    # Active venues use canonical Submission invitation; resolve via venue group.
    venue_group = client.get_group(venue_id)
    sub_inv = venue_group.content.get("submission_id", {}).get("value")
    if not sub_inv:
        log.warning("no submission_id for %s — skip", venue_id)
        return
    submissions = client.get_all_notes(invitation=sub_inv, details="directReplies")
    log.info("%s: %d submissions", venue_id, len(submissions))
    for sub in submissions:
        replies = sub.details.get("directReplies", []) or []
        def _inv(r):
            invs = _reply_attr(r, "invitations") if not isinstance(r, dict) else r.get("invitations")
            return (invs or [""])[0]
        # ICLR/NeurIPS use 'Official_Review'; ICML uses just 'Official_Review' inside a different
        # path; broaden the substring to include any 'Review' invitation, exclude meta-reviews.
        def _is_review(inv: str) -> bool:
            return ("Review" in inv) and ("Meta" not in inv) and ("Comment" not in inv)
        reviews = [r for r in replies if _is_review(_inv(r))]
        decisions = [r for r in replies if "Decision" in _inv(r)]
        yield {
            "submission": sub,
            "reviews": reviews,
            "decision": decisions[0] if decisions else None,
            "year": year,
        }


def _to_papers_row(record: dict) -> list:
    sub = record["submission"]
    c = sub.content or {}
    pid = _make_paper_id(sub.id)
    title = (c.get("title") or {}).get("value") or ""
    abstract = (c.get("abstract") or {}).get("value")
    venue_field = (c.get("venue") or {}).get("value")
    return [
        pid,
        "openreview",
        sub.id,
        None,                                       # arxiv_id
        None,                                       # openalex_id
        None,                                       # doi
        title,
        abstract,
        _coerce_date(sub.cdate),                    # submitted_date (Date)
        record["year"],                             # publication_year
        0,                                          # citation_count
        venue_field,                                # primary_category
        _flatten_authors(c),
        [],                                         # openalex_tags
        [],                                         # openalex_keywords
        None,                                       # pagerank_score
        None,                                       # katz_score
        None, None, 0,                              # community, semantic_cluster, in_corpus_degree
        datetime.utcnow(),
        datetime.utcnow(),
        [],                                         # abstract_embedding
    ]


def _reply_attr(r, key: str):
    """OpenReview API v2 returns directReplies as raw dicts, not Note objects."""
    if isinstance(r, dict):
        return r.get(key)
    return getattr(r, key, None)


def _to_review_rows(record: dict, venue_display: str) -> list[list]:
    sub_pid = _make_paper_id(record["submission"].id)
    rows = []
    decision = record.get("decision")
    decision_text = None
    if decision:
        dc = _reply_attr(decision, "content") or {}
        decision_text = (dc.get("decision") or {}).get("value")
    for r in record["reviews"]:
        rc = _reply_attr(r, "content") or {}
        def _v(key: str):
            return (rc.get(key) or {}).get("value")

        def _int_field(val):
            if val is None:
                return None
            try:
                return int(str(val).split(":")[0])
            except (ValueError, TypeError):
                return None

        rating_int = _int_field(_v("rating"))
        confidence_int = _int_field(_v("confidence"))
        soundness = _v("soundness")
        presentation = _v("presentation")
        contribution = _v("contribution")
        signatures = _reply_attr(r, "signatures") or []
        rows.append([
            sub_pid,
            _reply_attr(r, "id") or "",
            signatures[0] if signatures else None,
            venue_display,
            soundness if isinstance(soundness, int) else _int_field(soundness),
            presentation if isinstance(presentation, int) else _int_field(presentation),
            contribution if isinstance(contribution, int) else _int_field(contribution),
            rating_int,
            confidence_int,
            _v("summary"),
            _v("strengths"),
            _v("weaknesses"),
            _v("questions"),
            decision_text,
            _coerce_datetime(_reply_attr(r, "cdate")),
            datetime.utcnow(),
        ])
    return rows


PAPER_COLS = [
    "paper_id", "source", "source_id", "arxiv_id", "openalex_id",
    "doi", "title", "abstract", "submitted_date", "publication_year",
    "citation_count", "primary_category", "authors",
    "openalex_tags", "openalex_keywords",
    "pagerank_score", "katz_score", "community_id", "semantic_cluster",
    "in_corpus_degree", "ingested_at", "updated_at", "abstract_embedding",
]
REVIEW_COLS = [
    "paper_id", "review_id", "reviewer_id", "venue",
    "soundness", "presentation", "contribution", "rating", "confidence",
    "summary", "strengths", "weaknesses", "questions",
    "decision", "posted_at", "ingested_at",
]


def ingest(venues: list[tuple[str, str, int]] | None = None) -> dict[str, int]:
    venues = venues or DEFAULT_VENUES
    counters = {"submissions": 0, "reviews": 0}
    client = _client()
    ch = ch_connect()
    try:
        for display, venue_id, year in venues:
            try:
                papers_batch: list[list] = []
                reviews_batch: list[list] = []
                for record in _venue_papers(client, venue_id, year):
                    papers_batch.append(_to_papers_row(record))
                    reviews_batch.extend(_to_review_rows(record, display))
                    if len(papers_batch) >= 200:
                        ch.insert("papers", papers_batch, column_names=PAPER_COLS)
                        counters["submissions"] += len(papers_batch)
                        papers_batch = []
                    if len(reviews_batch) >= 500:
                        ch.insert("openreview_reviews", reviews_batch, column_names=REVIEW_COLS)
                        counters["reviews"] += len(reviews_batch)
                        reviews_batch = []
                if papers_batch:
                    ch.insert("papers", papers_batch, column_names=PAPER_COLS)
                    counters["submissions"] += len(papers_batch)
                if reviews_batch:
                    ch.insert("openreview_reviews", reviews_batch, column_names=REVIEW_COLS)
                    counters["reviews"] += len(reviews_batch)
                log.info("%s: cumulative %d subs, %d reviews", display,
                         counters["submissions"], counters["reviews"])
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed: %s", display, e)
            time.sleep(0.5)
    finally:
        ch.close()
    return counters
