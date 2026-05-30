"""TF-IDF + KMeans clustering of paper abstracts.

A different signal from citation-based communities (see graph.detect_communities).
Two papers can be in the same SEMANTIC cluster (talk about the same thing) without
citing each other, and vice versa. Comparing the two surfaces 'siloed work' (semantically
similar but disconnected research efforts).
"""

from __future__ import annotations

import logging
import re

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from researchpapers.config import Settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.clusters")

DEFAULT_N_CLUSTERS = 30
RANDOM_STATE = 42

# OpenAlex's reconstructed abstracts retain TeX math and XML formula markup,
# which TF-IDF was picking up as discriminative tokens (cluster anchored by
# physics/quantum papers got labeled 'tex, formula, formulatype'). Strip them.
_TEX_DOLLAR = re.compile(r"\$[^$\n]+\$")
_TEX_COMMAND = re.compile(r"\\[a-zA-Z]+(\{[^{}]*\})*")
_XML_TAG = re.compile(r"<[^<>]+>")
_WHITESPACE = re.compile(r"\s+")


def clean_abstract(text: str | None) -> str:
    if not text:
        return ""
    text = _TEX_DOLLAR.sub(" ", text)
    text = _TEX_COMMAND.sub(" ", text)
    text = _XML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def cluster_abstracts(settings: Settings, *, n_clusters: int = DEFAULT_N_CLUSTERS) -> dict[str, int]:
    counters = {"docs": 0, "clusters": 0}
    with connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT arxiv_id, abstract
                FROM papers
                WHERE abstract IS NOT NULL AND LENGTH(abstract) > 100
                """
            )
            rows = cur.fetchall()
        if not rows:
            log.warning("no abstracts found")
            return counters

        arxiv_ids = [r["arxiv_id"] for r in rows]
        texts = [clean_abstract(r["abstract"]) for r in rows]
        counters["docs"] = len(texts)

        log.info("vectorizing %d abstracts...", len(texts))
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.5,
            max_features=15000,
        )
        X = vec.fit_transform(texts)
        log.info("vocabulary size: %d", X.shape[1])

        log.info("fitting KMeans(n_clusters=%d)...", n_clusters)
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        counters["clusters"] = n_clusters

        with conn.cursor() as cur:
            cur.execute("UPDATE papers SET semantic_cluster = NULL")
            cur.executemany(
                "UPDATE papers SET semantic_cluster = %s WHERE arxiv_id = %s",
                [(int(cid), aid) for aid, cid in zip(arxiv_ids, labels, strict=True)],
            )
        conn.commit()

        sizes = [int((labels == c).sum()) for c in range(n_clusters)]
        log.info("done. cluster sizes top-10: %s", sorted(sizes, reverse=True)[:10])
    return counters
