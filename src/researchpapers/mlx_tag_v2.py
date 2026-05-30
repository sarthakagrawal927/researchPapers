"""MLX v2: native batched generation with mlx_lm, no HTTP server overhead.

Calls mlx_lm.generate directly in-process. Batches multiple prompts per forward
pass when possible. Avoids the per-request HTTP cost of the OpenAI-compatible
mlx_lm.server.

Expected gain: 3-5x over the server-based path due to (a) no HTTP roundtrip,
(b) shared model state across all prompts in a session, (c) larger natural batches.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime

from psycopg.types.json import Jsonb

from researchpapers.config import Settings
from researchpapers.db import connect
from researchpapers.llm_tag import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

log = logging.getLogger("researchpapers.mlx_tag_v2")

MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"
JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _build_prompt(tokenizer, paper: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                title=(paper.get("title") or "").strip(),
                abstract=((paper.get("abstract") or "").strip())[:1500],
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _parse_response(raw: str) -> tuple[str | None, list[str]]:
    """Pull a JSON object out of the model's free-text response."""
    if not raw:
        return None, []
    m = JSON_PATTERN.search(raw)
    if not m:
        return None, []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, []
    if not isinstance(obj, dict):
        return None, []
    # Case-insensitive lookup — Qwen sometimes emits "TLDR"/"Tags".
    lc = {k.lower(): v for k, v in obj.items()}
    tldr = (lc.get("tldr") or "")[:500] or None
    tags = lc.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if t][:20]
    return tldr, tags


def tag_papers(
    settings: Settings,
    *,
    limit: int | None = None,
    only_top_cited: bool = True,
    max_tokens: int = 256,
) -> dict[str, int | float]:
    """Native in-process MLX inference. One model load, sequential generation per paper."""
    from mlx_lm import generate, load

    counters: dict[str, int | float] = {
        "tagged": 0, "failed": 0, "skipped": 0,
        "prompt_tokens": 0, "completion_tokens": 0,
    }

    log.info("loading MLX model %s (one-time)", MODEL_NAME)
    t_load = time.monotonic()
    model, tokenizer = load(MODEL_NAME)
    log.info("model loaded in %.1fs", time.monotonic() - t_load)

    order_clause = "ORDER BY citation_count DESC NULLS LAST" if only_top_cited else ""
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT arxiv_id, title, abstract
            FROM papers
            WHERE mlx_llm_v2_tagged_at IS NULL
              AND abstract IS NOT NULL
              AND LENGTH(abstract) > 80
            {order_clause}
            """
            + (f" LIMIT {int(limit)}" if limit else "")
        )
        rows = cur.fetchall()

    log.info("queue: %d papers", len(rows))
    if not rows:
        return counters

    t0 = time.monotonic()
    # Process sequentially — MLX shares model state, no per-request startup cost.
    updates: list[tuple] = []
    now = datetime.now(UTC)
    for i, r in enumerate(rows):
        prompt = _build_prompt(tokenizer, r)
        try:
            raw = generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("generate failed for %s: %s", r["arxiv_id"], e)
            counters["failed"] += 1
            continue
        tldr, tags = _parse_response(raw)
        if not tags:
            counters["skipped"] += 1
            continue
        updates.append((tldr, Jsonb(tags), now, r["arxiv_id"]))
        counters["tagged"] += 1
        # Estimate tokens (no API to ask; rough chars/4)
        counters["completion_tokens"] += max(1, len(raw) // 4)
        if counters["tagged"] % 50 == 0:
            elapsed = time.monotonic() - t0
            log.info(
                "progress: %d done, %.2f papers/sec",
                counters["tagged"], counters["tagged"] / elapsed,
            )

    # Bulk write
    if updates:
        with connect(settings) as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE papers SET mlx_llm_v2_tldr = %s, mlx_llm_v2_tags_json = %s, mlx_llm_v2_tagged_at = %s WHERE arxiv_id = %s",
                updates,
            )
            conn.commit()

    elapsed = time.monotonic() - t0
    counters["elapsed_seconds"] = round(elapsed, 2)
    counters["papers_per_sec"] = round(int(counters["tagged"]) / elapsed, 2) if elapsed else 0
    counters["completion_tok_per_sec"] = round(int(counters["completion_tokens"]) / elapsed, 1) if elapsed else 0
    return counters
