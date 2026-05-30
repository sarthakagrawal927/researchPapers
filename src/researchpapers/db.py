from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from researchpapers.config import MIGRATIONS_DIR, Settings


@contextmanager
def connect(settings: Settings) -> Iterator[psycopg.Connection]:
    with psycopg.connect(settings.postgres_url, row_factory=dict_row) as conn:
        yield conn


def init_db(settings: Settings) -> list[str]:
    """Apply any migrations that have not been recorded in schema_migrations. Returns applied versions."""
    files = sorted(p for p in MIGRATIONS_DIR.glob("*.sql"))
    applied: list[str] = []
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("SELECT version FROM schema_migrations")
            done = {r["version"] for r in cur.fetchall()}
        for f in files:
            version = f.stem
            if version in done:
                continue
            sql = f.read_text()
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
                    (version,),
                )
            applied.append(version)
        conn.commit()
    return applied
