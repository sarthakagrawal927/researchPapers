"""MLX v3: grouped prompts — batch N papers per LLM call. CH-only, no Postgres.

Concept: pack 4 papers into a single chat prompt, ask for a JSON array, parse.
Same model load + same forward-pass cost ≈ 4× effective throughput, with the
quality of single-paper prompts (each paper still gets full title+abstract context).

Reads paper data + tracks "already tagged" state directly against ClickHouse
paper_tags (anti-join on tagger='mlx_qwen3b_v3'). Writes tags inline per batch
so killed runs lose at most one batch of work.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time

from researchpapers.ch_db import connect as ch_connect, write_paper_tags
from researchpapers.config import Settings


# RAM-aware throttling — yield CPU/GPU when other processes (e.g. an LLM
# training run) need the headroom.
_PAGE_SIZE = 4096
RAM_FULL_SPEED_MB = 6000   # > this much free → no throttle
RAM_PAUSE_MB = 3000        # < this much free → pause for RAM_PAUSE_SECONDS
RAM_PAUSE_SECONDS = 10


def _free_ram_mb() -> int:
    """Best-effort free RAM in MB on macOS (free + inactive + speculative)."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
    except Exception:
        return 8192  # safe fallback
    pages = {"free": 0, "inactive": 0, "speculative": 0}
    for line in out.splitlines():
        if "Pages free" in line:
            pages["free"] = int(line.rsplit(maxsplit=1)[-1].rstrip("."))
        elif "Pages inactive" in line:
            pages["inactive"] = int(line.rsplit(maxsplit=1)[-1].rstrip("."))
        elif "Pages speculative" in line:
            pages["speculative"] = int(line.rsplit(maxsplit=1)[-1].rstrip("."))
    return sum(pages.values()) * _PAGE_SIZE // (1024 * 1024)


def _ram_throttle() -> tuple[int, int]:
    """Sleep if free RAM is below threshold; return (free_mb_after_wait, total_paused_sec)."""
    paused = 0
    while True:
        free_mb = _free_ram_mb()
        if free_mb >= RAM_PAUSE_MB:
            return free_mb, paused
        log.info("RAM throttle: free=%d MB < %d MB, sleeping %ds", free_mb, RAM_PAUSE_MB, RAM_PAUSE_SECONDS)
        time.sleep(RAM_PAUSE_SECONDS)
        paused += RAM_PAUSE_SECONDS

log = logging.getLogger("researchpapers.mlx_tag_v3")

# Default to the smaller/faster 1.5B model. Override via --model flag if you
# need higher tag quality. Empirically the 1.5B handles short academic abstracts
# fine, just with slightly more uniform/less detailed tags.
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
SMALLER_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"  # 30% faster but 3× skip rate
GROUP_SIZE = 4
MAX_TOKENS = 700        # ~150 tokens × 4 papers + JSON overhead

GROUP_SYSTEM_PROMPT = """You are tagging academic papers for a research database.
For each paper provided, return a one-sentence TLDR (max 30 words) and 5-8 specific
technical tags (model names, methods, datasets, problem types — NOT generic words
like "deep learning" or "method").

Return ONLY a JSON array, one object per paper, in the exact same order:
[{"id": 1, "tldr": "...", "tags": ["tag1", "tag2", ...]}, {"id": 2, ...}]

No commentary, no markdown fences."""

JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def _build_group_prompt(tokenizer, papers: list[dict]) -> str:
    parts = []
    for i, p in enumerate(papers, start=1):
        title = (p.get("title") or "").strip()
        abstract = ((p.get("abstract") or "").strip())[:1200]
        parts.append(f"## Paper {i}\nTitle: {title}\nAbstract: {abstract}")
    user_msg = "\n\n".join(parts) + f"\n\nReturn a JSON array of {len(papers)} objects."
    messages = [
        {"role": "system", "content": GROUP_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _parse_group_response(raw: str, expected_n: int) -> list[tuple[str | None, list[str]] | None]:
    """Parse a JSON array of N objects. Returns list of (tldr, tags) or None per slot."""
    if not raw:
        return [None] * expected_n
    m = JSON_ARRAY_PATTERN.search(raw)
    if not m:
        return [None] * expected_n
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [None] * expected_n
    if not isinstance(arr, list):
        return [None] * expected_n

    out: list[tuple[str | None, list[str]] | None] = [None] * expected_n
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        lc = {k.lower(): v for k, v in obj.items()}
        idx = lc.get("id")
        if not isinstance(idx, int) or not (1 <= idx <= expected_n):
            continue
        tldr = (lc.get("tldr") or "")
        if isinstance(tldr, str):
            tldr = tldr[:500] or None
        else:
            tldr = None
        tags = lc.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip() for t in tags if t][:20]
        if tags:
            out[idx - 1] = (tldr, tags)
    return out


def tag_papers(
    settings: Settings,
    *,
    limit: int | None = None,
    only_top_cited: bool = True,
    premium_only: bool = False,
    group_size: int = GROUP_SIZE,
    max_tokens: int = MAX_TOKENS,
    tagger: str = "mlx_qwen3b_v3",
    shard: int = 0,
    total_shards: int = 1,
    model_name: str | None = None,
    throttle_seconds: float = 0.0,
) -> dict[str, int | float]:
    model_to_use = model_name or DEFAULT_MODEL
    """Group-prompted MLX inference. CH-only — reads from CH papers, writes to CH paper_tags.

    Tracks "already tagged" state by anti-joining against paper_tags WHERE tagger=...
    """
    from mlx_lm import generate, load

    counters: dict[str, int | float] = {
        "tagged": 0, "failed": 0, "skipped": 0, "groups": 0,
        "completion_tokens": 0,
    }

    log.info("loading MLX model %s (one-time)", model_to_use)
    t_load = time.monotonic()
    model, tokenizer = load(model_to_use)
    log.info("model loaded in %.1fs", time.monotonic() - t_load)

    premium_clause = ""
    if premium_only:
        premium_clause = """
            AND (
              citation_count >= 100
              OR (toYear(submitted_date) = 2025 AND citation_count >= 20)
              OR (toYear(submitted_date) = 2024 AND citation_count >= 30)
              OR (toYear(submitted_date) = 2023 AND citation_count >= 50)
            )
        """
    order_clause = "ORDER BY citation_count DESC" if only_top_cited else ""
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    shard_clause = ""
    if total_shards > 1:
        # %% escapes for clickhouse-connect's pyformat-style parameter substitution
        shard_clause = f"AND (cityHash64(paper_id) %% {total_shards}) = {shard}"

    with ch_connect() as ch:
        rows = ch.query(
            f"""
            SELECT p.paper_id, p.title, p.abstract
            FROM papers p FINAL
            WHERE p.source = 'arxiv'
              AND length(p.abstract) > 80
              {premium_clause}
              {shard_clause}
              AND p.paper_id NOT IN (
                SELECT paper_id FROM paper_tags FINAL WHERE tagger = %(tagger)s
              )
            {order_clause}
            {limit_clause}
            """,
            parameters={"tagger": tagger},
        ).result_rows

    log.info("queue: %d papers (group_size=%d, shard=%d/%d)", len(rows), group_size, shard, total_shards)
    if not rows:
        return counters

    # Convert tuples to dicts for the prompt builder (which expects .get())
    rows_dicts = [{"paper_id": r[0], "title": r[1], "abstract": r[2]} for r in rows]

    t0 = time.monotonic()
    pending: list[tuple] = []
    last_ram_check = 0.0
    paused_total = 0

    for start in range(0, len(rows_dicts), group_size):
        # RAM-aware throttle: every 5s, check free RAM. Pause until it recovers
        # if below the threshold (e.g. when an LLM training run starts elsewhere).
        if time.monotonic() - last_ram_check > 5:
            free_mb, paused = _ram_throttle()
            paused_total += paused
            if paused > 0:
                log.info("resumed: free=%d MB (total paused this run: %ds)", free_mb, paused_total)
            last_ram_check = time.monotonic()

        batch = rows_dicts[start : start + group_size]
        prompt = _build_group_prompt(tokenizer, batch)
        try:
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as e:  # noqa: BLE001
            log.warning("group generate failed (batch starting %s): %s", batch[0]["paper_id"], e)
            counters["failed"] += len(batch)
            continue
        counters["groups"] += 1
        counters["completion_tokens"] += max(1, len(raw) // 4)

        parsed = _parse_group_response(raw, len(batch))
        for r, slot in zip(batch, parsed, strict=True):
            if slot is None:
                counters["skipped"] += 1
                continue
            tldr, tags = slot
            pending.append((r["paper_id"], tagger, tags, tldr))
            counters["tagged"] += 1

        # Flush every 50 groups (~200 papers) so killed runs lose at most ~50s of work
        if counters["groups"] % 50 == 0 and pending:
            write_paper_tags(pending, model_version=model_to_use)
            pending = []

        # GPU-yield throttle — sleep between groups so other GPU consumers
        # (e.g. an LLM training run) get headroom.
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

        if counters["groups"] % 10 == 0:
            elapsed = time.monotonic() - t0
            rate = counters["tagged"] / elapsed if elapsed else 0
            log.info(
                "progress: %d tagged, %d skipped, %d groups, %.2f papers/sec",
                counters["tagged"], counters["skipped"], counters["groups"], rate,
            )

    if pending:
        write_paper_tags(pending, model_version=model_to_use)

    elapsed = time.monotonic() - t0
    counters["elapsed_seconds"] = round(elapsed, 2)
    counters["papers_per_sec"] = round(int(counters["tagged"]) / elapsed, 2) if elapsed else 0
    counters["completion_tok_per_sec"] = round(int(counters["completion_tokens"]) / elapsed, 1) if elapsed else 0
    return counters
