"""Head-to-head eval: spaCy noun_tags vs KeyBERT tags vs LLM tags (treated as ground truth).

Metrics:
  - exact_overlap_at_k: count of tags also present (exact case-insensitive match) in LLM tags
  - fuzzy_overlap_at_k: count of tags whose normalized form is a substring of (or contains) any LLM tag
  - precision_at_k: exact_overlap / k
  - jaccard: |A ∩ B| / |A ∪ B| on full tag sets

Also prints a sample of N papers showing all three tag sets side-by-side for visual judgment.
"""

from __future__ import annotations

import logging
import re
import statistics
from typing import Any

from researchpapers.config import Settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.tag_eval")


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tags_from_llm(blob: list[Any]) -> list[str]:
    # llm_tags_json is a flat list of strings
    if not isinstance(blob, list):
        return []
    return [str(t) for t in blob if t]


def _exact_overlap(approach_tags: list[str], llm_tags: list[str]) -> int:
    a_norm = {_normalize(t) for t in approach_tags}
    l_norm = {_normalize(t) for t in llm_tags}
    return len(a_norm & l_norm)


def _fuzzy_overlap(approach_tags: list[str], llm_tags: list[str]) -> int:
    a_norm = [_normalize(t) for t in approach_tags]
    l_norm = [_normalize(t) for t in llm_tags]
    hits = 0
    for a in a_norm:
        if not a:
            continue
        for l in l_norm:
            if not l:
                continue
            # Either is substring of the other (handles "attention" matching "attention mechanism")
            if a in l or l in a:
                hits += 1
                break
    return hits


def _jaccard(approach_tags: list[str], llm_tags: list[str]) -> float:
    a = {_normalize(t) for t in approach_tags if _normalize(t)}
    b = {_normalize(t) for t in llm_tags if _normalize(t)}
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


APPROACH_COLUMNS = {
    "spacy_v1":    "noun_tags_json",
    "spacy_v2":    "noun_tags_v2_json",
    "spacy_lg":    "web_lg_tags_json",
    "keybert":     "keybert_tags_json",
    "mlx_llm":     "mlx_llm_tags_json",
    "mlx_llm_v2":  "mlx_llm_v2_tags_json",
}


def evaluate(settings: Settings, *, sample_n: int = 12) -> dict:
    # Pull every column; we'll require llm_tags_json + at least one alt to score per row.
    cols = list(APPROACH_COLUMNS.values())
    select_clause = ", ".join(cols)
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT arxiv_id, title, citation_count,
                   llm_tags_json, {select_clause}
            FROM papers
            WHERE llm_tags_json IS NOT NULL
            ORDER BY citation_count DESC NULLS LAST
            """
        )
        rows = cur.fetchall()

    if not rows:
        return {"error": "no LLM-tagged papers yet"}

    log.info("evaluating %d papers", len(rows))

    # per-approach: list of (exact, fuzzy, jaccard, n_tags) per paper, only counting papers
    # where that approach has tags populated.
    per_approach: dict[str, list[tuple[int, int, float, int]]] = {k: [] for k in APPROACH_COLUMNS}
    llm_n_tags: list[int] = []
    samples: list[dict] = []

    for r in rows:
        llm = _tags_from_llm(r["llm_tags_json"])
        llm_n_tags.append(len(llm))
        sample = {"arxiv_id": r["arxiv_id"], "title": r["title"], "citation_count": r["citation_count"], "llm": llm}
        for approach, col in APPROACH_COLUMNS.items():
            tags = _tags_from_llm(r.get(col))
            if not tags:
                continue
            per_approach[approach].append(
                (_exact_overlap(tags, llm), _fuzzy_overlap(tags, llm), _jaccard(tags, llm), len(tags))
            )
            sample[approach] = tags
        if len(samples) < sample_n:
            samples.append(sample)

    def stats(xs):
        if not xs:
            return {"mean": None, "median": None, "min": None, "max": None}
        return {
            "mean": round(statistics.mean(xs), 3),
            "median": statistics.median(xs),
            "min": min(xs),
            "max": max(xs),
        }

    report: dict = {
        "n_papers": len(rows),
        "llm_avg_tags": stats(llm_n_tags)["mean"],
        "samples": samples,
    }
    for approach, data in per_approach.items():
        if not data:
            report[approach] = {"n": 0}
            continue
        exact, fuzzy, jacc, n_tags = zip(*data, strict=True)
        report[approach] = {
            "n": len(data),
            "avg_tags": stats(list(n_tags))["mean"],
            "exact_overlap_per_paper": stats(list(exact)),
            "fuzzy_overlap_per_paper": stats(list(fuzzy)),
            "jaccard_vs_llm": stats(list(jacc)),
        }
    return report


def print_report(report: dict) -> None:
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return
    print(f"\n=== Tag Eval (LLM-tagged papers: {report['n_papers']}, avg LLM tags: {report['llm_avg_tags']}) ===\n")
    for approach in APPROACH_COLUMNS:
        b = report.get(approach, {})
        if not b or b.get("n") == 0:
            print(f"--- {approach}: no data ---\n")
            continue
        print(f"--- {approach} (n={b['n']}) ---")
        print(f"  avg tags/paper:       {b['avg_tags']}")
        print(f"  exact overlap w/ LLM: mean={b['exact_overlap_per_paper']['mean']}  median={b['exact_overlap_per_paper']['median']}  max={b['exact_overlap_per_paper']['max']}")
        print(f"  fuzzy overlap w/ LLM: mean={b['fuzzy_overlap_per_paper']['mean']}  median={b['fuzzy_overlap_per_paper']['median']}  max={b['fuzzy_overlap_per_paper']['max']}")
        print(f"  Jaccard vs LLM:       mean={b['jaccard_vs_llm']['mean']}  median={b['jaccard_vs_llm']['median']}")
        print()

    print("\n--- Sample papers (side-by-side) ---\n")
    for s in report["samples"]:
        print(f"[{s['arxiv_id']}] {(s['title'] or '')[:80]}")
        print(f"  cites: {s.get('citation_count')}")
        print(f"  LLM       : {', '.join(s.get('llm', [])[:10])}")
        for approach in APPROACH_COLUMNS:
            if approach in s:
                print(f"  {approach:10s}: {', '.join(s[approach][:10])}")
        print()
