"""spaCy noun-chunk tagger.

Pulls noun chunks + proper nouns from title+abstract, filters generic terms,
ranks by frequency. Designed for speed: 500-1000 papers/sec with n_process=4.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from datetime import UTC, datetime

import spacy


def _detect_cpu_count() -> int:
    # On macOS, os.cpu_count() returns logical cores; for P-cores only, query sysctl-style.
    # Just use logical count; spaCy workers spend most time on dep-parsing which is BLAS-heavy.
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4

from researchpapers.config import Settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.noun_tag")

SPACY_MODEL = "en_core_web_sm"

# Column + freshness-marker for each model variant we benchmark.
MODEL_COLUMNS = {
    "en_core_web_sm": ("noun_tags_json", "noun_tagged_at"),
    "en_core_web_lg": ("web_lg_tags_json", "web_lg_tagged_at"),
    "en_core_sci_lg": ("noun_tags_json", "noun_tagged_at"),  # reuses noun_tags_json for now
}

_LEADING_ARTICLES = ("a ", "an ", "the ", "this ", "that ", "these ", "those ",
                     "our ", "their ", "his ", "her ", "its ", "my ", "your ",
                     "some ", "many ", "much ", "every ", "any ", "all ", "no ",
                     "such ")
_LEADING_QUANTIFIERS = ("one ", "two ", "three ", "four ", "five ",
                        "first ", "second ", "third ", "next ", "last ",
                        "main ", "primary ", "novel ", "new ", "old ",
                        "recent ", "current ", "previous ", "future ",
                        "good ", "bad ", "better ", "best ", "worse ", "worst ",
                        "important ", "useful ", "general ", "specific ", "common ",
                        "different ", "similar ", "various ", "several ", "few ",
                        "single ", "double ", "multiple ", "wide ", "small ", "large ",
                        "high ", "low ", "fast ", "slow ", "key ", "great ")

# Generic terms from academic abstracts that aren't useful as topic tags.
BLACKLIST_SINGLE = {
    "paper", "study", "studies", "method", "methods", "approach", "approaches",
    "system", "systems", "model", "models", "framework", "frameworks", "work",
    "results", "result", "performance", "accuracy", "dataset", "datasets",
    "experiment", "experiments", "task", "tasks", "application", "applications",
    "technique", "techniques", "analysis", "evaluation", "comparison",
    "art", "use", "uses", "using", "case", "cases", "problem", "problems",
    "solution", "solutions", "input", "output", "data", "information",
    "process", "processes", "feature", "features", "function", "functions",
    "set", "sets", "way", "ways", "step", "steps", "type", "types", "kind", "kinds",
    "value", "values", "level", "levels", "factor", "factors", "term", "terms",
    "number", "numbers", "size", "sizes", "rate", "rates", "amount", "amounts",
    "i", "we", "our", "their", "his", "her", "its", "they", "us", "you",
    "abstract", "introduction", "conclusion", "background", "related",
    "context", "contexts", "instance", "instances", "example", "examples",
    "fact", "facts", "part", "parts", "section", "sections", "chapter",
    "respect", "regards", "addition", "order", "term", "course", "form", "forms",
    "name", "names", "scope", "field", "fields", "area", "areas",
    "author", "authors", "user", "users", "people", "person", "team", "world",
    "year", "years", "month", "day", "time", "times", "period", "moment",
    "table", "figure", "graph", "chart", "image", "images", "picture",
    "paper presents", "paper proposes", "paper introduces", "paper describes",
    "paper provides", "paper studies", "we present", "we propose", "we introduce",
    "we describe", "we provide", "we study", "we show", "we demonstrate",
    "we evaluate", "we use", "we apply",
}

# Trash multi-word phrases that survive article stripping.
BLACKLIST_PHRASE = {
    "main contribution", "main contributions", "key contribution", "key contributions",
    "thorough evaluation", "extensive experiments", "extensive experiment",
    "significant improvement", "significant improvements", "novel approach",
    "novel method", "novel framework", "novel architecture", "new approach",
    "new method", "new framework", "open question", "open questions",
    "good performance", "high accuracy", "low error", "wide range", "wide variety",
    "real world", "real-world", "real time", "real-time", "long time", "short time",
    "first time", "many years", "recent years", "past years", "future work",
    "future research", "future direction", "future directions",
    "increasing attention", "increasing interest", "growing interest",
    "fact", "context", "respect",
    "best performance", "best results", "state of the art", "state-of-the-art",
    "prior art", "prior work", "related work", "prior-art configurations",
    "previous work", "previous works", "existing work", "existing approaches",
    "existing methods", "existing systems", "existing models",
    "first-order gradient", "second-order gradient",
}

SKIP_ROOTS = {"thing", "kind", "type", "way", "case", "use", "set", "fact",
              "context", "respect", "part", "instance", "example", "term"}


def _strip_leading(text: str) -> str:
    """Strip leading articles + adjectival fluff repeatedly: 'a very good cnn' -> 'cnn'."""
    s = text.strip().lower()
    changed = True
    while changed:
        changed = False
        for prefix in (*_LEADING_ARTICLES, *_LEADING_QUANTIFIERS):
            if s.startswith(prefix):
                s = s[len(prefix):]
                changed = True
                break
    return s.strip()


def _candidates(doc) -> Counter:
    from spacy.lang.en.stop_words import STOP_WORDS
    cnt: Counter[str] = Counter()
    # Multi-word noun chunks
    for chunk in doc.noun_chunks:
        raw = chunk.text.strip()
        text = _strip_leading(raw)
        if not text:
            continue
        n_words = len(text.split())
        if n_words < 2 or n_words > 4:
            continue
        if chunk.root.is_stop or chunk.root.lemma_.lower() in SKIP_ROOTS:
            continue
        if text in BLACKLIST_SINGLE or text in BLACKLIST_PHRASE:
            continue
        # Skip if every token is generic.
        tokens = text.split()
        if all(t in BLACKLIST_SINGLE or t in STOP_WORDS for t in tokens):
            continue
        # Skip if the first token after stripping is still a stop word — extra safety.
        if tokens[0] in STOP_WORDS:
            continue
        cnt[text] += 1
    # Proper nouns + capitalized acronyms (BERT, ResNet, ImageNet, T5, RAG, etc.)
    for tok in doc:
        if tok.pos_ == "PROPN" and 2 <= len(tok.text) <= 30:
            text = tok.text
            if text.lower() in BLACKLIST_SINGLE:
                continue
            cnt[text] += 1
        elif tok.text.isupper() and 2 <= len(tok.text) <= 8 and tok.text.isalpha():
            cnt[tok.text] += 1
    return cnt


def tag_papers(
    settings: Settings,
    *,
    model_name: str = SPACY_MODEL,
    limit: int | None = None,
    only_top_cited: bool = True,
) -> dict[str, int | float]:
    counters: dict[str, int | float] = {"tagged": 0, "skipped": 0}
    if model_name not in MODEL_COLUMNS:
        raise ValueError(f"unknown spaCy model {model_name}; known: {list(MODEL_COLUMNS)}")
    col_tags, col_at = MODEL_COLUMNS[model_name]

    log.info("loading spaCy model %s -> col %s", model_name, col_tags)
    nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])

    order_clause = "ORDER BY citation_count DESC NULLS LAST" if only_top_cited else ""
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT arxiv_id, title, abstract
            FROM papers
            WHERE {col_at} IS NULL
              AND abstract IS NOT NULL
              AND LENGTH(abstract) > 80
            {order_clause}
            """
            + (f" LIMIT {int(limit)}" if limit else "")
        )
        rows = cur.fetchall()

    log.info("queue: %d papers", len(rows))
    t0 = time.monotonic()

    texts = [f"{r['title']}\n\n{r['abstract']}" for r in rows]
    results: list[list[str]] = []
    # Parallel processing — sm is light enough that more workers help; lg has higher per-doc cost
    # so subprocess overhead dominates beyond 1-2 workers.
    if model_name == "en_core_web_sm":
        n_process = max(1, _detect_cpu_count() - 1)  # leave 1 core for OS + DB writer
        batch_size = 512
    else:
        n_process = 2
        batch_size = 256

    log.info("nlp.pipe n_process=%d batch_size=%d", n_process, batch_size)
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        cnt = _candidates(doc)
        tags = [t for t, _ in cnt.most_common(12)]
        results.append(tags)

    import json as _json
    now = datetime.now(UTC)
    with connect(settings) as conn, conn.cursor() as cur:
        for r, tags in zip(rows, results, strict=True):
            cur.execute(
                f"UPDATE papers SET {col_tags} = %s::jsonb, {col_at} = %s WHERE arxiv_id = %s",
                (_json.dumps(tags), now, r["arxiv_id"]),
            )
            counters["tagged"] += 1
            if tags == []:
                counters["skipped"] += 1
        conn.commit()

    elapsed = time.monotonic() - t0
    counters["elapsed_seconds"] = round(elapsed, 2)
    counters["papers_per_sec"] = round(int(counters["tagged"]) / elapsed, 1) if elapsed else 0
    return counters
