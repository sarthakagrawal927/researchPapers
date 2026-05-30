"""ClickHouse connection helpers.

ClickHouse is becoming the primary store for analytical reads + multi-source data.
Postgres stays alive during the transition for legacy code.
"""

from __future__ import annotations

import os

import clickhouse_connect
from clickhouse_connect.driver.client import Client


def _settings() -> dict[str, object]:
    return {
        "host": os.environ.get("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.environ.get("CLICKHOUSE_PORT", "8123")),
        "database": os.environ.get("CLICKHOUSE_DB", "papers"),
        "username": os.environ.get("CLICKHOUSE_USER", "papers"),
        "password": os.environ.get("CLICKHOUSE_PASSWORD", "papers"),
    }


def connect() -> Client:
    """Returns a ClickHouse client. Use as a context manager or just close() after."""
    return clickhouse_connect.get_client(**_settings())


def ping() -> bool:
    """Returns True if the server responds."""
    try:
        with connect() as c:
            row = c.query("SELECT 1").result_rows
            return bool(row and row[0][0] == 1)
    except Exception:
        return False


def write_paper_tags(
    rows: list[tuple[str, str, list[str], str | None]],
    *,
    model_version: str | None = None,
) -> int:
    """Bulk insert into paper_tags. Each row is (paper_id, tagger, tags_list, tldr_or_None).

    ReplacingMergeTree(computed_at) means rewriting the same (paper_id, tagger) is fine —
    the latest row wins. Inserts are append-only and very cheap in ClickHouse.
    """
    if not rows:
        return 0
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    payload = [[pid, tagger, tags or [], tldr, model_version, now] for pid, tagger, tags, tldr in rows]
    with connect() as c:
        c.insert(
            "paper_tags",
            payload,
            column_names=["paper_id", "tagger", "tags", "tldr", "model_version", "computed_at"],
        )
    return len(payload)


def arxiv_paper_id(arxiv_id: str) -> str:
    """Canonical paper_id format. Keep in sync with migration code."""
    return f"arxiv:{arxiv_id}"


def table_sizes() -> list[dict]:
    """Returns [{table, rows, bytes_on_disk}] for our `papers` database."""
    sql = """
        SELECT
            name AS table,
            total_rows AS rows,
            total_bytes AS bytes_on_disk,
            formatReadableSize(total_bytes) AS pretty_size
        FROM system.tables
        WHERE database = currentDatabase()
        ORDER BY total_bytes DESC NULLS LAST
    """
    with connect() as c:
        return c.query(sql).named_results()
