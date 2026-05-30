"""Leaderboards: top URLs by host, top exact URLs, top cited papers."""

from __future__ import annotations

import csv
import json
import sys
from typing import Literal

from researchpapers.config import Settings
from researchpapers.db import connect

Format = Literal["table", "csv", "json"]


def _emit(rows: list[dict], fmt: Format) -> None:
    if not rows:
        print("(no rows)", file=sys.stderr)
        return
    if fmt == "json":
        print(json.dumps(rows, indent=2, default=str))
        return
    if fmt == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return
    # table
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def top_hosts(settings: Settings, *, top: int, fmt: Format) -> None:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT host, COUNT(*) AS edges, COUNT(DISTINCT citing_arxiv_id) AS papers
            FROM references_url
            GROUP BY host
            ORDER BY papers DESC, edges DESC
            LIMIT %s
            """,
            (top,),
        )
        rows = cur.fetchall()
    _emit(rows, fmt)


def top_urls(settings: Settings, *, top: int, fmt: Format) -> None:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT url_canonical, host, COUNT(DISTINCT citing_arxiv_id) AS papers
            FROM references_url
            GROUP BY url_canonical, host
            ORDER BY papers DESC
            LIMIT %s
            """,
            (top,),
        )
        rows = cur.fetchall()
    _emit(rows, fmt)


def top_cited_papers(settings: Settings, *, top: int, fmt: Format) -> None:
    """Requires references_paper to be populated (run `papers fetch-citations` first)."""
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COALESCE(p.arxiv_id, r.cited_arxiv_id, r.cited_doi, r.cited_s2_id) AS identifier,
                COALESCE(p.title, MAX(r.cited_title)) AS title,
                COUNT(DISTINCT r.citing_arxiv_id) AS citing_papers
            FROM references_paper r
            LEFT JOIN papers p ON p.arxiv_id = r.cited_arxiv_id
            GROUP BY identifier, p.title
            ORDER BY citing_papers DESC
            LIMIT %s
            """,
            (top,),
        )
        rows = cur.fetchall()
    _emit(rows, fmt)
