from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


@dataclass(frozen=True)
class Settings:
    # Postgres is the legacy store; kept optional for the few remaining tools that
    # still touch it (ingest writers, migration scripts). The dashboard / API / taggers
    # are ClickHouse-only.
    postgres_url: str | None
    contact_email: str
    semantic_scholar_api_key: str | None

    @property
    def user_agent(self) -> str:
        from researchpapers import __version__

        return f"researchpapers/{__version__} ({self.contact_email})"


def load_settings() -> Settings:
    return Settings(
        postgres_url=os.environ.get("POSTGRES_URL") or None,
        contact_email=os.environ.get("CONTACT_EMAIL", "anonymous@example.com"),
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
    )
