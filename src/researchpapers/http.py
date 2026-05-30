from __future__ import annotations

import httpx

from researchpapers.config import Settings


def build_client(settings: Settings, *, timeout: float = 30.0) -> httpx.Client:
    """Sync httpx client with the User-Agent arXiv and S2 ask for. One client per CLI command."""
    return httpx.Client(
        timeout=timeout,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    )
