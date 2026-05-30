"""Exports from ClickHouse → JSON for the Astro app.

Initial scope: OpenReview reviews aggregations. The Postgres-based exporter.py
keeps owning everything else for now; this is incremental.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from researchpapers.ch_db import connect as ch_connect

log = logging.getLogger("researchpapers.ch_exports")


def _row_to_dict(row, names: list[str]) -> dict:
    return dict(zip(names, row, strict=True))


def export_review_data(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with ch_connect() as c:
        # 1. Summary across venues
        result = c.query("""
            SELECT
                venue,
                count() AS n_reviews,
                countDistinct(paper_id) AS n_papers,
                avg(rating) AS avg_rating,
                avg(confidence) AS avg_confidence,
                countIf(decision = 'Accept (Oral)') AS oral_accepts,
                countIf(decision = 'Accept (Poster)') AS poster_accepts,
                countIf(decision = 'Reject') AS rejects
            FROM openreview_reviews
            GROUP BY venue
            ORDER BY n_reviews DESC
        """).result_rows
        cols = ["venue", "n_reviews", "n_papers", "avg_rating", "avg_confidence",
                "oral_accepts", "poster_accepts", "rejects"]
        venues = [
            {
                **_row_to_dict(r, cols),
                "avg_rating": round(float(r[3] or 0), 2),
                "avg_confidence": round(float(r[4] or 0), 2),
            }
            for r in result
        ]
        p = out_dir / "review_venues.json"
        p.write_text(json.dumps(venues, indent=2, default=str))
        written.append(p)

        # 2. Top-rated submissions: best reviewer-average per paper
        result = c.query("""
            SELECT
                r.paper_id,
                p.title,
                r.venue,
                avg(r.rating) AS avg_rating,
                avg(r.confidence) AS avg_confidence,
                count() AS n_reviews,
                any(r.decision) AS decision
            FROM openreview_reviews r
            LEFT JOIN papers p ON p.paper_id = r.paper_id
            WHERE r.rating IS NOT NULL
            GROUP BY r.paper_id, p.title, r.venue
            HAVING n_reviews >= 3
            ORDER BY avg_rating DESC, avg_confidence DESC
            LIMIT 200
        """).result_rows
        cols2 = ["paper_id", "title", "venue", "avg_rating", "avg_confidence",
                 "n_reviews", "decision"]
        top_papers = [
            {
                **_row_to_dict(r, cols2),
                "avg_rating": round(float(r[3] or 0), 2),
                "avg_confidence": round(float(r[4] or 0), 2),
            }
            for r in result
        ]
        p = out_dir / "review_top_papers.json"
        p.write_text(json.dumps(top_papers, indent=2, default=str))
        written.append(p)

        # 3. Rating distribution per venue
        result = c.query("""
            SELECT venue, rating, count() AS n
            FROM openreview_reviews
            WHERE rating IS NOT NULL
            GROUP BY venue, rating
            ORDER BY venue, rating
        """).result_rows
        distribution = [
            {"venue": r[0], "rating": int(r[1]), "n": int(r[2])} for r in result
        ]
        p = out_dir / "review_rating_distribution.json"
        p.write_text(json.dumps(distribution, indent=2))
        written.append(p)

        # 4. Source breakdown (papers across sources) — useful for the header card
        result = c.query("""
            SELECT source, count() AS n FROM papers GROUP BY source ORDER BY n DESC
        """).result_rows
        sources = [{"source": r[0], "n": int(r[1])} for r in result]
        p = out_dir / "ch_sources_summary.json"
        p.write_text(json.dumps(sources, indent=2))
        written.append(p)

        # 5a. Tag × reviewer rating cross-join — the HighSignal-shape insight.
        # Tags are normalized to lowercase (collapses "Deep Learning"/"deep learning"/"DEEP LEARNING")
        # and known plural pairs ("language models"/"language model") are merged via the CASE map.
        # For each spaCy-extracted tag, mean ICLR/NeurIPS reviewer rating across
        # papers tagged with it. Includes sample top-rated papers per tag for drilldown.
        result = c.query("""
            WITH paper_avg_rating AS (
                SELECT
                    r.paper_id,
                    avg(r.rating) AS avg_rating,
                    count() AS n_reviews,
                    any(r.venue) AS venue,
                    any(p.title) AS title
                FROM openreview_reviews r
                LEFT JOIN papers p ON p.paper_id = r.paper_id
                WHERE r.rating IS NOT NULL
                GROUP BY r.paper_id
                HAVING n_reviews >= 3
            )
            SELECT
                multiIf(
                  lower(tag) IN ('language model', 'large language model', 'large language models'), 'language models',
                  lower(tag) IN ('neural network'), 'neural networks',
                  lower(tag) IN ('diffusion model'), 'diffusion models',
                  lower(tag) IN ('transformer'), 'transformers',
                  lower(tag) IN ('vision transformer'), 'vision transformers',
                  lower(tag) IN ('graph neural network'), 'graph neural networks',
                  lower(tag) IN ('convolutional neural network'), 'convolutional neural networks',
                  lower(tag) IN ('llm', 'llms'), 'llms',
                  lower(tag)
                ) AS canonical_tag,
                round(avg(par.avg_rating), 2) AS mean_rating,
                count() AS n_papers,
                round(quantile(0.9)(par.avg_rating), 2) AS p90_rating,
                groupArray(50)((par.avg_rating, par.title, par.paper_id, par.venue)) AS samples
            FROM paper_tags t FINAL
            ARRAY JOIN tags AS tag
            JOIN paper_avg_rating par ON par.paper_id = t.paper_id
            WHERE t.tagger = 'spacy_v2'
            GROUP BY canonical_tag
            HAVING n_papers >= 10
            ORDER BY mean_rating DESC
            LIMIT 100
        """).result_rows
        tag_rating = []
        for r in result:
            samples = sorted(
                [{"avg_rating": float(s[0]), "title": s[1] or "", "paper_id": s[2], "venue": s[3]} for s in r[4]],
                key=lambda s: -s["avg_rating"],
            )[:5]
            tag_rating.append({
                "tag": r[0],
                "mean_rating": float(r[1] or 0),
                "n_papers": int(r[2]),
                "p90_rating": float(r[3] or 0),
                "samples": samples,
            })
        p = out_dir / "tag_rating.json"
        p.write_text(json.dumps(tag_rating, indent=2, default=str))
        written.append(p)

        # 6. Sleeper papers — reviewers loved them but they haven't accrued citations.
        result = c.query("""
            WITH par AS (
              SELECT paper_id, avg(rating) AS avg_rating, count() AS n_reviews,
                     any(decision) AS decision, any(venue) AS venue
              FROM openreview_reviews WHERE rating IS NOT NULL
              GROUP BY paper_id HAVING n_reviews >= 3
            )
            SELECT p.paper_id, p.title, par.avg_rating, par.n_reviews,
                   p.citation_count, par.venue, par.decision, p.submitted_date
            FROM par
            JOIN papers p ON p.paper_id = par.paper_id
            WHERE par.avg_rating >= 7.0 AND p.citation_count <= 20
              AND toYear(p.submitted_date) >= 2024
            ORDER BY par.avg_rating DESC, p.citation_count ASC
            LIMIT 100
        """).result_rows
        sleepers = [
            {"paper_id": r[0], "title": r[1], "avg_rating": round(float(r[2]), 2),
             "n_reviews": int(r[3]), "citation_count": int(r[4] or 0),
             "venue": r[5], "decision": r[6], "submitted_date": str(r[7]) if r[7] else None}
            for r in result
        ]
        p = out_dir / "sleepers.json"
        p.write_text(json.dumps(sleepers, indent=2, default=str))
        written.append(p)

        # 7. Hot right now — unified score across cites/year + rating + PageRank.
        result = c.query("""
            WITH par AS (
              SELECT paper_id, avg(rating) AS avg_rating
              FROM openreview_reviews WHERE rating IS NOT NULL
              GROUP BY paper_id HAVING count() >= 3
            )
            SELECT p.paper_id, p.source, p.title, p.citation_count, p.submitted_date,
                   round(p.citation_count / greatest((today() - p.submitted_date) / 365.25, 0.25), 1) AS cpy,
                   coalesce(par.avg_rating, 0) AS rating,
                   coalesce(p.pagerank_score, 0) AS pr,
                   round(
                     0.5 * log(1 + p.citation_count / greatest((today() - p.submitted_date) / 365.25, 0.25))
                     + 0.3 * coalesce(par.avg_rating, 5.0) / 10
                     + 0.2 * coalesce(p.pagerank_score, 0) * 10000,
                   3) AS hotness
            FROM papers AS p FINAL
            LEFT JOIN par ON par.paper_id = p.paper_id
            WHERE p.submitted_date IS NOT NULL
              AND toYear(p.submitted_date) >= 2023
              AND p.citation_count >= 5
            ORDER BY hotness DESC
            LIMIT 100
        """).result_rows
        hot = [
            {"paper_id": r[0], "source": r[1], "title": r[2],
             "citation_count": int(r[3] or 0),
             "submitted_date": str(r[4]) if r[4] else None,
             "cites_per_year": float(r[5]),
             "avg_rating": round(float(r[6]), 2) if r[6] else None,
             "pagerank": float(r[7]),
             "hotness": float(r[8])}
            for r in result
        ]
        p = out_dir / "hot.json"
        p.write_text(json.dumps(hot, indent=2, default=str))
        written.append(p)
    return written
