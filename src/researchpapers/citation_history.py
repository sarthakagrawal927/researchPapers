"""Periodic citation_count snapshots into citation_history.

Run monthly. Pulls current cited_by_count from OpenAlex for every paper that has an
openalex_id, writes one row per paper into citation_history. ReplacingMergeTree on
measured_at means re-running on the same day is idempotent.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from typing import Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from researchpapers.ch_db import connect as ch_connect
from researchpapers.config import load_settings

log = logging.getLogger("researchpapers.citation_history")

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
BATCH_SIZE = 50           # ids per OpenAlex call
POLITE_INTERVAL = 0.2


def _client(mailto: str) -> httpx.Client:
    return httpx.Client(
        timeout=30.0,
        headers={"User-Agent": f"researchpapers/0.1 ({mailto})"},
        params={"mailto": mailto},
    )


def _chunked(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.RemoteProtocolError)),
)
def _fetch_batch(client: httpx.Client, short_ids: list[str]) -> dict:
    resp = client.get(
        OPENALEX_WORKS_URL,
        params={
            "filter": f"ids.openalex:{'|'.join(short_ids)}",
            "per-page": len(short_ids),
            "select": "id,cited_by_count",
        },
    )
    resp.raise_for_status()
    return resp.json() or {}


def snapshot_today(measured_at: date | None = None, *, limit: int | None = None) -> int:
    """Pull current cited_by_count for all papers with openalex_id; insert into citation_history."""
    measured_at = measured_at or date.today()
    settings = load_settings()
    log.info("snapshotting citation_history for %s", measured_at)
    ch = ch_connect()
    try:
        # Pull openalex_ids of all papers we know about.
        result = ch.query("""
            SELECT paper_id, openalex_id
            FROM papers FINAL
            WHERE isNotNull(openalex_id) AND length(openalex_id) > 0
        """)
        pairs = list(result.result_rows)
        if limit:
            pairs = pairs[: int(limit)]
        if not pairs:
            log.info("no papers with openalex_id; nothing to do")
            return 0
        log.info("snapshot queue: %d papers", len(pairs))

        oa_to_pid = {oa: pid for pid, oa in pairs}
        oa_ids = list(oa_to_pid)

        rows_to_write: list[list] = []
        last_at = 0.0
        with _client(settings.contact_email) as client:
            for chunk in _chunked(oa_ids, BATCH_SIZE):
                # rate-limit
                elapsed = time.monotonic() - last_at
                if elapsed < POLITE_INTERVAL:
                    time.sleep(POLITE_INTERVAL - elapsed)
                short_ids = [i.rsplit("/", 1)[-1] for i in chunk]
                body = _fetch_batch(client, short_ids)
                last_at = time.monotonic()
                for w in body.get("results", []):
                    oa = w.get("id")
                    pid = oa_to_pid.get(oa)
                    if not pid:
                        continue
                    rows_to_write.append([pid, measured_at, int(w.get("cited_by_count") or 0)])

        if rows_to_write:
            ch.insert(
                "citation_history",
                rows_to_write,
                column_names=["paper_id", "measured_at", "citation_count"],
            )
        log.info("wrote %d citation_history rows", len(rows_to_write))
        return len(rows_to_write)
    finally:
        ch.close()
