"""Graph centrality (PageRank, Katz) + cycle detection on the citation subgraph.

The within-corpus subgraph: edges where both the citing arxiv_id and the cited
work are in our `papers` table. This is the only subgraph where transitive
flow is meaningful — cited works outside the corpus are sinks.

- PageRank: random-walk steady state with 0.85 damping. Naturally weights
  influential citers more.
- Katz: explicit α-decay over path length. α=0.05 means tertiary citations
  count at 0.25% of primary. The dial maps directly to the user's framing of
  "secondary < primary, tertiary < secondary, ..."
- Cycles: networkx.simple_cycles() with a length cap. Citation cycles are
  rare in academic data, so anything we find is interesting on its own.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import networkx as nx

from researchpapers.config import Settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.graph")

DEFAULT_KATZ_ALPHA = 0.05
DEFAULT_PAGERANK_DAMPING = 0.85
CYCLE_MAX_LENGTH = 8
LOUVAIN_SEED = 42


def _load_subgraph(conn) -> tuple[nx.DiGraph, dict[str, str]]:
    """Returns (DiGraph keyed by arxiv_id, mapping arxiv_id->openalex_id).

    Nodes = all our papers. Edges = (citing arxiv_id -> cited arxiv_id) where the cited
    work's openalex_id matches one of our papers' openalex_id.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT arxiv_id, openalex_id FROM papers")
        papers = cur.fetchall()
    oa_to_arxiv = {r["openalex_id"]: r["arxiv_id"] for r in papers if r["openalex_id"]}

    g = nx.DiGraph()
    for r in papers:
        g.add_node(r["arxiv_id"])

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.citing_arxiv_id, r.cited_openalex_id
            FROM references_paper r
            WHERE r.cited_openalex_id IS NOT NULL
            """
        )
        edges = cur.fetchall()
    skipped = 0
    for e in edges:
        cited_arxiv = oa_to_arxiv.get(e["cited_openalex_id"])
        if cited_arxiv is None:
            skipped += 1
            continue
        # Skip self-loops; they break Katz convergence and aren't meaningful
        if cited_arxiv == e["citing_arxiv_id"]:
            continue
        g.add_edge(e["citing_arxiv_id"], cited_arxiv)
    log.info(
        "subgraph: %d nodes, %d edges (skipped %d edges to out-of-corpus works)",
        g.number_of_nodes(),
        g.number_of_edges(),
        skipped,
    )
    return g, oa_to_arxiv


def _compute_katz(g: nx.DiGraph, alpha: float) -> dict[str, float]:
    """Katz centrality with safe alpha. If user-provided alpha breaks convergence, halve it."""
    try:
        return nx.katz_centrality(g, alpha=alpha, max_iter=2000, tol=1e-5)
    except nx.PowerIterationFailedConvergence:
        log.warning("Katz did not converge with alpha=%.4f, retrying with %.4f", alpha, alpha / 2)
        return nx.katz_centrality(g, alpha=alpha / 2, max_iter=2000, tol=1e-5)


def compute_scores(
    settings: Settings,
    *,
    katz_alpha: float = DEFAULT_KATZ_ALPHA,
    pagerank_damping: float = DEFAULT_PAGERANK_DAMPING,
) -> dict[str, int]:
    """Computes PageRank + Katz + in-corpus in-degree per paper, writes to papers table."""
    counters = {"nodes": 0, "edges": 0, "scored": 0, "cycles_found": 0}
    with connect(settings) as conn:
        g, _ = _load_subgraph(conn)
        counters["nodes"] = g.number_of_nodes()
        counters["edges"] = g.number_of_edges()

        # PageRank/Katz on g directly: edges are (citing -> cited), so the walker flows
        # from citers to cited papers and score accumulates at well-cited (foundational)
        # nodes. (Reversing the graph would give a "review-paper detector" instead —
        # interesting on its own, not what we want here.)
        log.info("computing PageRank (damping=%.2f)...", pagerank_damping)
        pagerank = nx.pagerank(g, alpha=pagerank_damping, max_iter=200, tol=1e-6)

        log.info("computing Katz centrality (alpha=%.4f)...", katz_alpha)
        katz = _compute_katz(g, katz_alpha)

        # In-corpus in-degree: how many of OUR papers cite this paper
        in_deg = dict(g.in_degree())

        now = datetime.now(UTC)
        with conn.cursor() as cur:
            for arxiv_id in g.nodes():
                cur.execute(
                    """
                    UPDATE papers SET
                        pagerank_score   = %s,
                        katz_score       = %s,
                        in_corpus_degree = %s,
                        graph_scored_at  = %s
                    WHERE arxiv_id = %s
                    """,
                    (
                        pagerank.get(arxiv_id, 0.0),
                        katz.get(arxiv_id, 0.0),
                        in_deg.get(arxiv_id, 0),
                        now,
                        arxiv_id,
                    ),
                )
                counters["scored"] += 1
        conn.commit()

        # Cycle detection on the within-corpus subgraph. No cap — at length_bound=8 the
        # whole enumeration finishes in well under a second on this graph; truncating to
        # an arbitrary 500 was hiding the real distribution.
        log.info("scanning for cycles up to length %d...", CYCLE_MAX_LENGTH)
        cycles_found = list(nx.simple_cycles(g, length_bound=CYCLE_MAX_LENGTH))
        cycles_found.sort(key=lambda c: (len(c), c))
        with conn.cursor() as cur:
            cur.execute("TRUNCATE citation_cycles RESTART IDENTITY")
            cur.executemany(
                "INSERT INTO citation_cycles (cycle_length, arxiv_ids) VALUES (%s, %s)",
                [(len(cyc), cyc) for cyc in cycles_found],
            )
        conn.commit()
        counters["cycles_found"] = len(cycles_found)
        by_len: dict[int, int] = {}
        for cyc in cycles_found:
            by_len[len(cyc)] = by_len.get(len(cyc), 0) + 1
        log.info("done. %d cycles persisted, by length: %s",
                 len(cycles_found), sorted(by_len.items()))
    return counters


def detect_communities(settings: Settings) -> dict[str, int]:
    """Louvain community detection on the within-corpus subgraph (undirected projection).

    Writes papers.community_id. Singletons (papers with no in-corpus edges) get NULL.
    """
    counters = {"nodes": 0, "edges": 0, "communities": 0, "assigned": 0}
    with connect(settings) as conn:
        g, _ = _load_subgraph(conn)
        g_und = g.to_undirected()
        g_active = g_und.subgraph(
            [n for n in g_und.nodes() if g_und.degree(n) > 0]
        ).copy()
        counters["nodes"] = g_active.number_of_nodes()
        counters["edges"] = g_active.number_of_edges()

        log.info("running Louvain on %d connected nodes...", g_active.number_of_nodes())
        communities = list(
            nx.community.louvain_communities(g_active, seed=LOUVAIN_SEED)
        )
        communities.sort(key=len, reverse=True)
        counters["communities"] = len(communities)

        paper_to_cid: dict[str, int] = {}
        for cid, members in enumerate(communities):
            for arxiv_id in members:
                paper_to_cid[arxiv_id] = cid
        counters["assigned"] = len(paper_to_cid)

        with conn.cursor() as cur:
            cur.execute("UPDATE papers SET community_id = NULL")
            cur.executemany(
                "UPDATE papers SET community_id = %s WHERE arxiv_id = %s",
                [(cid, aid) for aid, cid in paper_to_cid.items()],
            )
        conn.commit()
        log.info(
            "louvain: %d communities, top-10 sizes=%s",
            len(communities),
            [len(c) for c in communities[:10]],
        )
    return counters
