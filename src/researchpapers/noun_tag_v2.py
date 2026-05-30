"""spaCy v2 tagger: parser disabled, POS-only chunking, bulk DB writes.

The dependency parser eats ~60-70% of spaCy's CPU time. We don't actually need
parsed structure — only the POS tags. So we run a `tok2vec + tagger` pipeline
and extract noun-phrase candidates ourselves via a simple POS pattern match:

    (ADJ|NOUN|PROPN)+ ending with NOUN or PROPN

This matches "deep convolutional neural networks", "Adam optimizer",
"stochastic gradient descent", but not "the proposed" or "our model".

Expected: 3-5× faster than v1 on the same hardware.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections import Counter

import spacy

from researchpapers.config import Settings
from researchpapers.noun_tag import (
    BLACKLIST_PHRASE,
    BLACKLIST_SINGLE,
    _strip_leading,
)

log = logging.getLogger("researchpapers.noun_tag_v2")

SPACY_MODEL = "en_core_web_sm"


def _detect_cpu_count() -> int:
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


PAGE_SIZE = 4096
WORKER_RSS_MB = 1500          # observed per-spaCy-worker RSS at steady state
SAFETY_HEADROOM_MB = 1500     # leave this much free for the OS + other apps


def _free_ram_mb() -> int:
    """Best-effort free RAM in MB on macOS. Counts 'free' + 'inactive' + 'speculative' pages."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
    except Exception:
        return 4096  # safe fallback
    pages = {"free": 0, "inactive": 0, "speculative": 0}
    for line in out.splitlines():
        if "Pages free" in line:
            pages["free"] = int(line.rsplit(maxsplit=1)[-1].rstrip("."))
        elif "Pages inactive" in line:
            pages["inactive"] = int(line.rsplit(maxsplit=1)[-1].rstrip("."))
        elif "Pages speculative" in line:
            pages["speculative"] = int(line.rsplit(maxsplit=1)[-1].rstrip("."))
    return sum(pages.values()) * PAGE_SIZE // (1024 * 1024)


def _pick_n_process(cap: int | None = None) -> int:
    """RAM-aware worker count. cap clamps the upper bound."""
    cap = cap or min(8, max(1, _detect_cpu_count() - 1))
    free_mb = _free_ram_mb()
    budget = max(0, free_mb - SAFETY_HEADROOM_MB)
    n = max(1, min(cap, budget // WORKER_RSS_MB))
    log.info("RAM picker: free=%d MB → n_process=%d (cap=%d)", free_mb, n, cap)
    return int(n)


def _candidates_pos_only(doc) -> Counter:
    """Extract noun-phrase candidates via POS pattern, no dependency parsing."""
    from spacy.lang.en.stop_words import STOP_WORDS

    cnt: Counter[str] = Counter()
    n = len(doc)
    i = 0
    while i < n:
        if doc[i].pos_ in ("ADJ", "NOUN", "PROPN"):
            j = i
            while j < n and doc[j].pos_ in ("ADJ", "NOUN", "PROPN"):
                j += 1
            # Trim trailing ADJs — phrase must end with a noun
            end = j - 1
            while end >= i and doc[end].pos_ not in ("NOUN", "PROPN"):
                end -= 1
            if end > i:
                phrase = " ".join(doc[k].text for k in range(i, end + 1)).strip()
                cleaned = _strip_leading(phrase)
                n_words = len(cleaned.split())
                if 2 <= n_words <= 4:
                    if cleaned not in BLACKLIST_SINGLE and cleaned not in BLACKLIST_PHRASE:
                        tokens = cleaned.split()
                        if not all(t in BLACKLIST_SINGLE or t in STOP_WORDS for t in tokens):
                            if tokens[0] not in STOP_WORDS:
                                cnt[cleaned] += 1
            i = j
        else:
            i += 1

    # Proper nouns + capitalized acronyms.
    # Single PROPNs are noisy: in titles, spaCy frequently mis-tags ordinary nouns and
    # adjectives ("Deep", "Learning", "Network") as PROPN just because they're capitalized.
    # We keep singletons only if they have a clear "this is a real model/tool name" shape:
    #   - all uppercase + alphabetic + 2-8 chars (CNN, BERT, LLM, GPT)
    #   - OR mixed-case with internal capital after the first char (ImageNet, ResNet,
    #     PyTorch, OpenAI) or digits (GPT4, T5, Llama2)
    for tok in doc:
        t = tok.text
        n = len(t)
        if tok.pos_ == "PROPN" and 3 <= n <= 30:
            has_internal_upper_or_digit = any(c.isupper() or c.isdigit() for c in t[1:])
            if has_internal_upper_or_digit and t.lower() not in BLACKLIST_SINGLE:
                cnt[t] += 1
        elif t.isupper() and 2 <= n <= 8 and t.isalpha():
            cnt[t] += 1
    return cnt


def tag_multi_source(
    *,
    source: str,
    batch_papers: int = 5000,
    limit: int | None = None,
    n_process: int | None = None,
    max_procs: int | None = None,
) -> dict[str, int | float]:
    """Run spaCy v2 on any source in ClickHouse (e.g. 'openreview', 'biorxiv').

    Reads title+abstract from CH papers (source=...), writes tags to CH paper_tags.
    Does NOT touch Postgres. Skips papers already tagged by spacy_v2.
    """
    from researchpapers.ch_db import connect as ch_connect, write_paper_tags

    log.info("loading spaCy %s with parser DISABLED", SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner", "lemmatizer"])

    t0 = time.monotonic()
    total_tagged = 0
    total_skipped = 0

    with ch_connect() as ch:
        already = ch.query(
            "SELECT DISTINCT paper_id FROM paper_tags WHERE tagger='spacy_v2'"
        ).result_rows
        tagged_ids = {r[0] for r in already}
        log.info("already-tagged count: %d", len(tagged_ids))

        rows_q = ch.query(
            f"""
            SELECT paper_id, title, abstract
            FROM papers FINAL
            WHERE source = %(source)s
              AND length(abstract) > 80
            """,
            parameters={"source": source},
        ).result_rows
        rows = [
            {"paper_id": r[0], "title": r[1], "abstract": r[2]}
            for r in rows_q
            if r[0] not in tagged_ids
        ]
        if limit:
            rows = rows[: int(limit)]
        log.info("queue (untagged %s submissions): %d papers", source, len(rows))

    batch_idx = 0
    for start in range(0, len(rows), batch_papers):
        chunk = rows[start : start + batch_papers]
        batch_idx += 1
        n = n_process or _pick_n_process(cap=max_procs)
        log.info(
            "batch #%d: %d papers, n_process=%d (running total tagged=%d)",
            batch_idx, len(chunk), n, total_tagged,
        )
        texts = [f"{r['title']}\n\n{r['abstract']}" for r in chunk]
        results: list[list[str]] = []
        for doc in nlp.pipe(texts, batch_size=1024, n_process=n):
            cnt = _candidates_pos_only(doc)
            results.append([t for t, _ in cnt.most_common(12)])

        ch_rows = [
            (r["paper_id"], "spacy_v2", tags, None)
            for r, tags in zip(chunk, results, strict=True)
        ]
        write_paper_tags(ch_rows, model_version="en_core_web_sm")
        total_tagged += len(chunk)
        total_skipped += sum(1 for r in results if not r)

    elapsed = time.monotonic() - t0
    return {
        "tagged": total_tagged,
        "skipped": total_skipped,
        "elapsed_seconds": round(elapsed, 2),
        "papers_per_sec": round(total_tagged / elapsed, 1) if elapsed else 0,
    }


def tag_papers(
    settings: Settings,
    *,
    limit: int | None = None,
    only_top_cited: bool = True,
    batch_papers: int | None = None,
    n_process: int | None = None,
    max_procs: int | None = None,
) -> dict[str, int | float]:
    """Tag the arxiv slice with spaCy v2. Thin wrapper around tag_multi_source.

    only_top_cited is ignored (kept for CLI back-compat) — CH ORDER BY against
    a half-billion-row table is wasted work for what's effectively a bulk tag pass.
    """
    return tag_multi_source(
        source="arxiv",
        limit=limit,
        batch_papers=batch_papers or 25000,
        n_process=n_process,
        max_procs=max_procs,
    )
