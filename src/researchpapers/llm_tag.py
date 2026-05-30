"""Local LLM tagging via LM Studio (default) or Ollama: title+abstract → TLDR + specific tags.

LM Studio (recommended):
    Start the LM Studio app, load a model, click "Start Server" (port 1234 by default).
    Models to try: llama-3.2-3b-instruct, qwen2.5-3b-instruct, gemma-2-2b-it.
    LM Studio supports structured JSON output (response_format), more reliable than Ollama.

Ollama (alternative):
    ollama serve
    ollama pull qwen2.5:3b
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from enum import Enum

import httpx

from researchpapers.config import Settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.llm_tag")


class Backend(str, Enum):
    LM_STUDIO = "lm-studio"
    OLLAMA = "ollama"
    MLX = "mlx"


# (chat_url, default_model, storage_columns)
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_DEFAULT_MODEL = "llama-3.2-3b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"
MLX_URL = "http://localhost:8080/v1/chat/completions"
MLX_DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Each backend writes to its own pair of columns so we can A/B compare them.
BACKEND_COLUMNS = {
    Backend.LM_STUDIO: ("llm_tags_json", "llm_tldr", "llm_tagged_at"),
    Backend.OLLAMA:    ("llm_tags_json", "llm_tldr", "llm_tagged_at"),
    Backend.MLX:       ("mlx_llm_tags_json", "mlx_llm_tldr", "mlx_llm_tagged_at"),
}

# JSON schema we want the model to fill in. LM Studio honors this strictly; Ollama best-effort.
TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "tldr": {
            "type": "string",
            "description": "One sentence, max 25 words, describing what the paper does. Concrete, no fluff.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 12,
            "description": "Specific technical tags: method names, dataset names, sub-problems, application areas. Avoid generic terms like 'machine learning', 'deep learning', 'neural network'.",
        },
    },
    "required": ["tldr", "tags"],
}

SYSTEM_PROMPT = (
    "You are an expert at tagging academic computer science papers. "
    "Given a title and abstract, output a JSON object with a one-sentence TLDR and 5-12 specific technical tags. "
    "Tags should name concrete methods, datasets, sub-problems, or application areas. "
    "Avoid generic umbrella terms like 'machine learning', 'deep learning', or 'neural network'."
)

USER_PROMPT_TEMPLATE = """Title: {title}

Abstract: {abstract}"""


def _build_messages(paper: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                title=(paper.get("title") or "").strip(),
                abstract=((paper.get("abstract") or "").strip())[:1500],
            ),
        },
    ]


def _tag_one_lm_studio(client: httpx.Client, model: str, paper: dict) -> dict | None:
    resp = client.post(
        LM_STUDIO_URL,
        json={
            "model": model,
            "messages": _build_messages(paper),
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "paper_tags", "strict": True, "schema": TAG_SCHEMA},
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("message") or {}).get("content") or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _tag_one_ollama(client: httpx.Client, model: str, paper: dict) -> dict | None:
    messages = _build_messages(paper)
    prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
    resp = client.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = (resp.json() or {}).get("response", "")
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


async def _tag_one_openai_compat_async(
    client: httpx.AsyncClient, url: str, model: str, paper: dict, strict_schema: bool
) -> tuple[dict | None, dict]:
    """Generic OpenAI-Chat-Completions caller. Used for both LM Studio and MLX backends."""
    body: dict = {
        "model": model,
        "messages": _build_messages(paper),
        "temperature": 0.1,
        "max_tokens": 512,
    }
    if strict_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "paper_tags", "strict": True, "schema": TAG_SCHEMA},
        }
    else:
        body["response_format"] = {"type": "json_object"}
    resp = await client.post(url, json=body, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    choices = data.get("choices") or []
    if not choices:
        return None, usage
    content = (choices[0].get("message") or {}).get("content") or ""
    try:
        return json.loads(content), usage
    except json.JSONDecodeError:
        return None, usage


async def _tag_one_ollama_async(
    client: httpx.AsyncClient, model: str, paper: dict
) -> tuple[dict | None, dict]:
    """Ollama returns prompt_eval_count + eval_count. Normalize to OpenAI-shape usage."""
    messages = _build_messages(paper)
    prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
    resp = await client.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=180,
    )
    resp.raise_for_status()
    body = resp.json() or {}
    usage = {
        "prompt_tokens": body.get("prompt_eval_count", 0),
        "completion_tokens": body.get("eval_count", 0),
    }
    raw = body.get("response", "")
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return None, usage
    return (out if isinstance(out, dict) else None), usage




def _make_tag_fn(backend: Backend):
    if backend == Backend.LM_STUDIO:
        async def _fn(client, model, paper):
            return await _tag_one_openai_compat_async(client, LM_STUDIO_URL, model, paper, strict_schema=True)
        return _fn
    if backend == Backend.MLX:
        # MLX server does not honor strict json_schema; use json_object mode.
        async def _fn(client, model, paper):
            return await _tag_one_openai_compat_async(client, MLX_URL, model, paper, strict_schema=False)
        return _fn
    return _tag_one_ollama_async


_BACKEND_TAGGER_NAME = {
    Backend.LM_STUDIO: "lm_studio_llm",
    Backend.OLLAMA:    "ollama_llm",
    Backend.MLX:       "mlx_qwen3b_v2",
}


def _write_result(
    settings: Settings, columns: tuple[str, str, str], arxiv_id: str,
    tldr: str | None, tags: list[str],
) -> None:
    col_tags, col_tldr, col_at = columns
    # Postgres (legacy)
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE papers SET
                {col_tldr} = %s,
                {col_tags} = %s::jsonb,
                {col_at} = %s
            WHERE arxiv_id = %s
            """,
            (tldr, json.dumps(tags), datetime.now(UTC), arxiv_id),
        )
        conn.commit()
    # ClickHouse (new). Looking up the right tagger requires backend context — caller passes it.
    # See _write_result_with_backend below.


def _write_result_with_backend(
    settings: Settings, columns: tuple[str, str, str], backend: Backend,
    arxiv_id: str, tldr: str | None, tags: list[str],
) -> None:
    _write_result(settings, columns, arxiv_id, tldr, tags)
    try:
        from researchpapers.ch_db import arxiv_paper_id, write_paper_tags
        write_paper_tags(
            [(arxiv_paper_id(arxiv_id), _BACKEND_TAGGER_NAME.get(backend, "unknown_llm"), tags, tldr)],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("ClickHouse dual-write failed for %s: %s", arxiv_id, e)


async def _run_async(
    settings: Settings,
    backend: Backend,
    model: str,
    rows: list[dict],
    concurrency: int,
    counters: dict[str, int],
) -> None:
    tag_fn = _make_tag_fn(backend)
    columns = BACKEND_COLUMNS[backend]
    sem = asyncio.Semaphore(concurrency)

    async def handle_one(client: httpx.AsyncClient, paper: dict) -> None:
        async with sem:
            try:
                result, usage = await tag_fn(client, model, paper)
            except httpx.HTTPStatusError as e:
                print(f"[HTTP {e.response.status_code}] {paper['arxiv_id']}: {e.response.text[:300]}", flush=True)
                counters["failed"] += 1
                return
            except httpx.HTTPError as e:
                print(f"[httpx {type(e).__name__}] {paper['arxiv_id']}: {e}", flush=True)
                counters["failed"] += 1
                return
            except Exception as e:  # noqa: BLE001
                print(f"[{type(e).__name__}] {paper['arxiv_id']}: {e}", flush=True)
                counters["failed"] += 1
                return
            counters["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            counters["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            if not result:
                counters["skipped"] += 1
                return
            tldr = (result.get("tldr") or "")[:500] or None
            tags = result.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).strip() for t in tags if t][:20]
            try:
                await asyncio.to_thread(
                    _write_result_with_backend, settings, columns, backend,
                    paper["arxiv_id"], tldr, tags,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("write failed for %s: %s", paper["arxiv_id"], e)
                counters["failed"] += 1
                return
            counters["tagged"] += 1
            if counters["tagged"] % 50 == 0:
                log.info(
                    "llm-tag: %d tagged, %d failed, %d skipped",
                    counters["tagged"], counters["failed"], counters["skipped"],
                )

    # Bump httpx pool so concurrency up to ~100 doesn't bottleneck.
    limits = httpx.Limits(
        max_keepalive_connections=max(concurrency * 2, 20),
        max_connections=max(concurrency * 2, 20),
    )
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:
        await asyncio.gather(*(handle_one(client, r) for r in rows))


def tag_papers(
    settings: Settings,
    *,
    backend: Backend = Backend.LM_STUDIO,
    model: str | None = None,
    limit: int | None = None,
    only_top_cited: bool = True,
    concurrency: int = 4,
    premium_only: bool = False,
) -> dict[str, int | float]:
    import time
    counters: dict[str, int | float] = {
        "tagged": 0, "failed": 0, "skipped": 0,
        "prompt_tokens": 0, "completion_tokens": 0,
    }
    if model is None:
        if backend == Backend.LM_STUDIO:
            model = LM_STUDIO_DEFAULT_MODEL
        elif backend == Backend.MLX:
            model = MLX_DEFAULT_MODEL
        else:
            model = OLLAMA_DEFAULT_MODEL

    _, _, col_at = BACKEND_COLUMNS[backend]
    order_clause = "ORDER BY citation_count DESC NULLS LAST" if only_top_cited else ""
    premium_clause = ""
    if premium_only:
        # Hybrid filter: high absolute citation count OR high cites/year for recent papers.
        premium_clause = """
              AND (
                citation_count >= 100
                OR (EXTRACT(YEAR FROM submitted_date)::int = 2025 AND citation_count >= 20)
                OR (EXTRACT(YEAR FROM submitted_date)::int = 2024 AND citation_count >= 30)
                OR (EXTRACT(YEAR FROM submitted_date)::int = 2023 AND citation_count >= 50)
              )
        """
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT arxiv_id, title, abstract, tags_json
            FROM papers
            WHERE {col_at} IS NULL
              AND abstract IS NOT NULL
              AND LENGTH(abstract) > 80
              {premium_clause}
            {order_clause}
            """
            + (f" LIMIT {int(limit)}" if limit else "")
        )
        rows = cur.fetchall()
    log.info(
        "llm-tag queue: %d papers, backend=%s, model=%s, concurrency=%d",
        len(rows), backend.value, model, concurrency,
    )
    if not rows:
        return counters

    t0 = time.monotonic()
    asyncio.run(_run_async(settings, backend, model, rows, concurrency, counters))
    elapsed = time.monotonic() - t0

    counters["elapsed_seconds"] = round(elapsed, 2)
    counters["papers_per_sec"] = round(int(counters["tagged"]) / elapsed, 2) if elapsed else 0
    pt = int(counters["prompt_tokens"])
    ct = int(counters["completion_tokens"])
    counters["completion_tok_per_sec"] = round(ct / elapsed, 1) if elapsed else 0
    counters["total_tok_per_sec"] = round((pt + ct) / elapsed, 1) if elapsed else 0
    counters["avg_prompt_tokens"] = round(pt / int(counters["tagged"]), 0) if counters["tagged"] else 0
    counters["avg_completion_tokens"] = round(ct / int(counters["tagged"]), 0) if counters["tagged"] else 0
    return counters
