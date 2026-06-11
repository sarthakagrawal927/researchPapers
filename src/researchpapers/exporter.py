"""Export aggregations from ClickHouse → JSON for the Astro app.

Entirely ClickHouse-backed. The previous Postgres-based exporter is retired —
all analytics columns (pagerank_score, katz_score, community_id, semantic_cluster)
are populated in CH papers, and the multi-source corpus + paper_tags lives there too.

Maintains the same JSON output filenames as before so the Astro app needs no
changes. URL-leaderboard JSONs are stub-empty (those tables were dropped in
migration 012).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from researchpapers.ch_db import connect as ch_connect
from researchpapers.clusters import clean_abstract
from researchpapers.config import Settings


def _format_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(f) < 1024.0:
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024.0
    return f"{f:.1f} PB"


def _isodate(d) -> str | None:
    if d is None:
        return None
    if hasattr(d, "isoformat"):
        return d.isoformat()
    return str(d)


def export_all(settings: Settings, out_dir: Path, *, top: int = 200) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with ch_connect() as c:
        # ---------- 1. Summary headline counters ----------
        papers_total = c.query("SELECT count() FROM papers").result_rows[0][0]
        # Denominator for "ingest progress" is *taggable* papers — those with
        # an abstract long enough for spaCy. Papers with no/short abstract are
        # ingested but can't be tagged meaningfully.
        taggable = c.query(
            "SELECT count() FROM papers FINAL WHERE length(abstract) > 80"
        ).result_rows[0][0]
        paper_edges = c.query("SELECT count() FROM references_paper").result_rows[0][0]
        tagged = c.query(
            "SELECT count(DISTINCT paper_id) FROM paper_tags WHERE tagger='spacy_v2'"
        ).result_rows[0][0]
        bytes_on_disk = c.query(
            "SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND database='papers'"
        ).result_rows[0][0] or 0

        summary = {
            "papers_total": int(papers_total),
            "papers_ingested": int(tagged),
            "url_edges": 0,
            "unique_hosts": 0,
            "unique_urls": 0,
            "texts_stored": int(papers_total),
            "gz_size": _format_bytes(int(bytes_on_disk)),
            "gz_bytes": int(bytes_on_disk),
            "paper_edges": int(paper_edges),
        }
        # Compare tagged against TAGGABLE (filtered) rather than total — so the
        # headline reflects actual ingest progress, not papers with no abstract.
        summary["papers_total"] = int(taggable)
        summary["papers_remaining"] = max(int(taggable) - int(tagged), 0)
        summary["ingest_percent"] = round(100 * tagged / taggable, 1) if taggable else 0.0
        avg = (int(bytes_on_disk) / int(papers_total)) if papers_total else 0.0
        summary["avg_bytes_per_paper"] = int(avg)
        summary["projected_total_bytes"] = int(bytes_on_disk)
        summary["projected_total_human"] = _format_bytes(int(bytes_on_disk))
        summary["avg_per_paper_human"] = _format_bytes(int(avg))
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        written.append(out_dir / "summary.json")

        # ---------- 2/3/5/6/8. URL-leaderboard stubs (table dropped) ----------
        for fname in ("top_hosts.json", "top_urls.json", "host_categories.json", "urls_per_paper_hist.json"):
            (out_dir / fname).write_text("[]")
            written.append(out_dir / fname)
        (out_dir / "host_drilldowns.json").write_text("{}")
        written.append(out_dir / "host_drilldowns.json")

        # ---------- 4. Top papers (arxiv only, by PageRank) ----------
        today = date.today()
        rows = c.query(
            f"""
            SELECT
                p.arxiv_id,
                coalesce(nullIf(m.title, ''), p.title) AS title,
                coalesce(nullIf(m.citation_count, 0), p.citation_count) AS citation_count,
                p.primary_category,
                effective_date(p.source, p.arxiv_id, p.submitted_date) AS submitted_date,
                p.in_corpus_degree,
                coalesce(s.pagerank, p.pagerank_score) AS pagerank_score,
                p.katz_score,
                p.openalex_tags, p.openalex_keywords
            FROM papers AS p FINAL
            LEFT JOIN paper_metadata_v2 AS m FINAL ON m.paper_id = p.paper_id
            LEFT JOIN paper_scores_v2 AS s FINAL ON s.paper_id = p.paper_id
            WHERE p.source='arxiv' AND p.citation_count > 0
              AND (s.pagerank IS NOT NULL OR p.pagerank_score IS NOT NULL)
            ORDER BY coalesce(s.pagerank, p.pagerank_score) DESC
            LIMIT {top}
            """
        ).result_rows
        papers = []
        for r in rows:
            sd = r[4]
            year = sd.year if sd else None
            age = max((today - sd).days / 365.25, 0.25) if sd else 1.0
            cites_per_year = round(float(r[2] or 0) / age, 1) if sd else None
            papers.append({
                "arxiv_id": r[0],
                "title": r[1],
                "citation_count": int(r[2] or 0),
                "primary_category": r[3],
                "submitted_date": _isodate(sd),
                "in_corpus_degree": int(r[5] or 0),
                "pagerank_score": round(float(r[6]), 6) if r[6] is not None else None,
                "katz_score": round(float(r[7]), 6) if r[7] is not None else None,
                "n_urls": 0,
                "year": year,
                "cites_per_year": cites_per_year,
                "topic_tags": list(r[8] or [])[:6],
                "top_keywords": list(r[9] or [])[:5],
            })
        (out_dir / "top_papers.json").write_text(json.dumps(papers, indent=2))
        written.append(out_dir / "top_papers.json")

        # ---------- 7. Top cited works ----------
        rows = c.query(
            f"""
            SELECT
                c.title, c.cited_by_count, count(DISTINCT r.citing_arxiv_id) AS in_corpus_citations,
                c.arxiv_id, c.doi, c.publication_year, c.primary_topic
            FROM references_paper r
            JOIN cited_works c ON c.openalex_id = r.cited_openalex_id
            WHERE length(c.title) > 0
            GROUP BY c.openalex_id, c.title, c.cited_by_count, c.arxiv_id, c.doi, c.publication_year, c.primary_topic
            ORDER BY in_corpus_citations DESC
            LIMIT {top}
            """
        ).result_rows
        cited = [
            {
                "title": r[0],
                "global_citations": int(r[1] or 0),
                "in_corpus_citations": int(r[2] or 0),
                "arxiv_id": r[3],
                "doi": r[4],
                "publication_year": int(r[5]) if r[5] else None,
                "primary_topic": r[6],
            }
            for r in rows
        ]
        (out_dir / "top_cited_works.json").write_text(json.dumps(cited, indent=2))
        written.append(out_dir / "top_cited_works.json")

        # ---------- 7a. Communities ----------
        top_cids_rows = c.query(
            "SELECT community_id, count() AS n FROM papers FINAL WHERE source='arxiv' AND community_id IS NOT NULL GROUP BY community_id ORDER BY n DESC LIMIT 50"
        ).result_rows
        top_cids = [r[0] for r in top_cids_rows]

        communities: list[dict] = []
        cluster_texts: list[str] = []
        for cid in top_cids:
            members = c.query(
                "SELECT arxiv_id, title, pagerank_score, submitted_date FROM papers FINAL WHERE source='arxiv' AND community_id = %(cid)s ORDER BY pagerank_score DESC",
                parameters={"cid": cid},
            ).result_rows
            if not members:
                continue
            anchor = members[0]
            years = sorted({m[3].year for m in members if m[3]})
            top_papers = [
                {"arxiv_id": m[0], "title": m[1], "pagerank": round(float(m[2] or 0), 6)}
                for m in members[:5]
            ]
            cluster_texts.append(" ".join((m[1] or "") for m in members))
            communities.append({
                "id": int(cid),
                "size": len(members),
                "anchor_arxiv_id": anchor[0],
                "anchor_title": anchor[1],
                "year_range": [years[0], years[-1]] if years else None,
                "top_hosts": [],  # URL data dropped
                "top_papers": top_papers,
            })

        if cluster_texts:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000, min_df=2)
            tfidf = vec.fit_transform(cluster_texts)
            feature_names = vec.get_feature_names_out()
            for i, com in enumerate(communities):
                row = tfidf[i].toarray()[0]
                top_idx = row.argsort()[-6:][::-1]
                com["labels"] = [feature_names[j] for j in top_idx if row[j] > 0]
        (out_dir / "communities.json").write_text(json.dumps(communities, indent=2))
        written.append(out_dir / "communities.json")

        # ---------- 7c. Temporal evolution ----------
        ppy = c.query(
            "SELECT effective_year(source, arxiv_id, submitted_date) AS year, count() AS n FROM papers FINAL WHERE source='arxiv' AND submitted_date IS NOT NULL GROUP BY year ORDER BY year"
        ).result_rows
        papers_per_year = [{"year": int(r[0]), "n": int(r[1])} for r in ppy]

        top6_rows = c.query(
            "SELECT community_id, count() AS n FROM papers FINAL WHERE source='arxiv' AND community_id IS NOT NULL GROUP BY community_id ORDER BY n DESC LIMIT 6"
        ).result_rows
        top6_cids = [int(r[0]) for r in top6_rows]

        cy = c.query(
            "SELECT effective_year(source, arxiv_id, submitted_date) AS year, community_id, count() AS n FROM papers FINAL WHERE source='arxiv' AND submitted_date IS NOT NULL AND community_id IN %(cids)s GROUP BY year, community_id ORDER BY year, community_id",
            parameters={"cids": top6_cids},
        ).result_rows
        community_years = [{"year": int(r[0]), "community_id": int(r[1]), "n": int(r[2])} for r in cy]

        tpy = c.query(
            """
            SELECT year, argMax(arxiv_id, pagerank_score) AS arxiv_id,
                   argMax(title, pagerank_score) AS title, max(pagerank_score) AS max_pr,
                   argMax(citation_count, pagerank_score) AS citation_count
            FROM (
              SELECT effective_year(source, arxiv_id, submitted_date) AS year,
                     arxiv_id, title, pagerank_score, citation_count
              FROM papers FINAL
              WHERE source='arxiv' AND submitted_date IS NOT NULL AND pagerank_score IS NOT NULL
            )
            GROUP BY year ORDER BY year
            """
        ).result_rows
        top_paper_per_year = [
            {"year": int(r[0]), "arxiv_id": r[1], "title": r[2], "pagerank_score": round(float(r[3] or 0), 6), "citation_count": int(r[4] or 0)}
            for r in tpy
        ]

        cpy = c.query(
            """
            SELECT effective_year(source, arxiv_id, submitted_date) AS year, count() AS n,
                   round(avg(citation_count / greatest((today() - effective_date(source, arxiv_id, submitted_date)) / 365.25, 0.25)), 1) AS mean_cpy,
                   round(quantile(0.9)(citation_count / greatest((today() - effective_date(source, arxiv_id, submitted_date)) / 365.25, 0.25)), 1) AS p90_cpy
            FROM papers FINAL
            WHERE source='arxiv' AND submitted_date IS NOT NULL AND citation_count IS NOT NULL
              AND effective_year(source, arxiv_id, submitted_date) >= 2005
            GROUP BY year ORDER BY year
            """
        ).result_rows
        cites_per_year = [{"year": int(r[0]), "n": int(r[1]), "mean_cpy": float(r[2]), "p90_cpy": float(r[3])} for r in cpy]

        temporal = {
            "papers_per_year": papers_per_year,
            "top_communities": top6_cids,
            "community_years": community_years,
            "top_paper_per_year": top_paper_per_year,
            "cites_per_year_by_year": cites_per_year,
        }
        (out_dir / "temporal.json").write_text(json.dumps(temporal, indent=2))
        written.append(out_dir / "temporal.json")

        # ---------- 7d. Top authors ----------
        rows = c.query(
            f"""
            SELECT
                arrayJoin(authors) AS author,
                count() AS n_papers,
                sum(citation_count) AS sum_citations,
                round(sum(coalesce(pagerank_score, 0)) * 1000, 3) AS sum_pr,
                arraySlice(groupArray(arxiv_id), 1, 3) AS top_arxiv_ids
            FROM papers FINAL
            WHERE source='arxiv' AND length(authors) > 0
            GROUP BY author
            HAVING n_papers >= 2
            ORDER BY sum_citations DESC
            LIMIT 200
            """
        ).result_rows
        authors = [
            {
                "author": r[0],
                "n_papers": int(r[1]),
                "sum_citations": int(r[2] or 0),
                "sum_pr": float(r[3] or 0),
                "top_arxiv_ids": list(r[4] or []),
            }
            for r in rows
        ]
        (out_dir / "top_authors.json").write_text(json.dumps(authors, indent=2))
        written.append(out_dir / "top_authors.json")

        # ---------- 7d-ii. Author drilldowns (top 50) ----------
        author_drills: dict[str, dict] = {}
        for a in authors[:50]:
            name = a["author"]
            papers_for = c.query(
                """
                SELECT arxiv_id, title, citation_count, submitted_date, community_id, pagerank_score
                FROM papers FINAL
                WHERE source='arxiv' AND has(authors, %(name)s)
                ORDER BY citation_count DESC
                LIMIT 50
                """,
                parameters={"name": name},
            ).result_rows
            comm_for = c.query(
                "SELECT community_id, count() AS n FROM papers FINAL WHERE source='arxiv' AND has(authors, %(name)s) AND community_id IS NOT NULL GROUP BY community_id ORDER BY n DESC LIMIT 5",
                parameters={"name": name},
            ).result_rows
            author_drills[name] = {
                "author": name,
                "papers": [
                    {
                        "arxiv_id": r[0], "title": r[1], "citation_count": int(r[2] or 0),
                        "submitted_date": _isodate(r[3]),
                        "community_id": int(r[4]) if r[4] is not None else None,
                        "pagerank_score": round(float(r[5] or 0), 6),
                    }
                    for r in papers_for
                ],
                "top_hosts": [],  # URL data dropped
                "communities": [{"community_id": int(r[0]), "n": int(r[1])} for r in comm_for],
            }
        (out_dir / "author_drilldowns.json").write_text(json.dumps(author_drills, indent=2))
        written.append(out_dir / "author_drilldowns.json")

        # ---------- 7a-ii. Community drilldowns (top 30) ----------
        comm_drills: dict[str, dict] = {}
        for cid in top_cids[:30]:
            papers_for = c.query(
                "SELECT arxiv_id, title, citation_count, submitted_date, pagerank_score FROM papers FINAL WHERE source='arxiv' AND community_id = %(cid)s ORDER BY pagerank_score DESC LIMIT 50",
                parameters={"cid": cid},
            ).result_rows
            authors_for = c.query(
                "SELECT arrayJoin(authors) AS author, count() AS n FROM papers FINAL WHERE source='arxiv' AND community_id = %(cid)s AND length(authors) > 0 GROUP BY author ORDER BY n DESC LIMIT 10",
                parameters={"cid": cid},
            ).result_rows
            years_for = c.query(
                "SELECT effective_year(source, arxiv_id, submitted_date) AS year, count() AS n FROM papers FINAL WHERE source='arxiv' AND community_id = %(cid)s AND submitted_date IS NOT NULL GROUP BY year ORDER BY year",
                parameters={"cid": cid},
            ).result_rows
            comm_drills[str(cid)] = {
                "id": int(cid),
                "papers": [
                    {
                        "arxiv_id": r[0], "title": r[1], "citation_count": int(r[2] or 0),
                        "submitted_date": _isodate(r[3]),
                        "pagerank_score": round(float(r[4] or 0), 6),
                    }
                    for r in papers_for
                ],
                "top_hosts": [],
                "top_authors": [{"author": r[0], "n": int(r[1])} for r in authors_for],
                "years": [{"year": int(r[0]), "n": int(r[1])} for r in years_for],
            }
        (out_dir / "community_drilldowns.json").write_text(json.dumps(comm_drills, indent=2))
        written.append(out_dir / "community_drilldowns.json")

        # ---------- 7e. Semantic abstract clusters ----------
        sem_cids_rows = c.query(
            "SELECT semantic_cluster, count() AS size FROM papers FINAL WHERE source='arxiv' AND semantic_cluster IS NOT NULL GROUP BY semantic_cluster ORDER BY size DESC"
        ).result_rows
        sem_cids = [int(r[0]) for r in sem_cids_rows]
        sem_clusters: list[dict] = []
        sem_texts: list[str] = []
        for cid in sem_cids:
            members = c.query(
                "SELECT arxiv_id, title, abstract, pagerank_score FROM papers FINAL WHERE source='arxiv' AND semantic_cluster = %(cid)s ORDER BY pagerank_score DESC",
                parameters={"cid": cid},
            ).result_rows
            if not members:
                continue
            anchor = members[0]
            sem_texts.append(" ".join(clean_abstract(m[2] or "") for m in members[:50]))
            sem_clusters.append({
                "id": int(cid),
                "size": len(members),
                "anchor_arxiv_id": anchor[0],
                "anchor_title": anchor[1],
                "top_papers": [{"arxiv_id": m[0], "title": m[1]} for m in members[:5]],
            })

        if sem_texts:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000, min_df=2)
            tfidf = vec.fit_transform(sem_texts)
            feature_names = vec.get_feature_names_out()
            for i, com in enumerate(sem_clusters):
                row = tfidf[i].toarray()[0]
                top_idx = row.argsort()[-7:][::-1]
                com["labels"] = [feature_names[j] for j in top_idx if row[j] > 0]
        (out_dir / "abstract_clusters.json").write_text(json.dumps(sem_clusters, indent=2))
        written.append(out_dir / "abstract_clusters.json")

        # ---------- 7b. Citation cycles ----------
        cycle_counts_rows = c.query(
            "SELECT cycle_length, count() AS n FROM citation_cycles GROUP BY cycle_length ORDER BY cycle_length"
        ).result_rows
        cycle_counts = [{"cycle_length": int(r[0]), "n": int(r[1])} for r in cycle_counts_rows]
        sample_rows = c.query(
            "SELECT cycle_length, paper_ids FROM citation_cycles ORDER BY cycle_length, detected_at LIMIT 300"
        ).result_rows
        # paper_ids are 'arxiv:xxx' format in CH; strip prefix for backwards compat with Astro
        sample = [
            {
                "length": int(r[0]),
                "arxiv_ids": [pid.replace("arxiv:", "") for pid in r[1]],
            }
            for r in sample_rows
        ]
        cycles_doc = {"by_length": cycle_counts, "sample": sample}
        (out_dir / "cycles.json").write_text(json.dumps(cycles_doc, indent=2))
        written.append(out_dir / "cycles.json")

    return written
