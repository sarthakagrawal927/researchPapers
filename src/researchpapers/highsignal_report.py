"""Sample HighSignal-style report queries against ClickHouse.

Outputs a markdown digest covering:
  - Top-rated peer-reviewed papers this cycle (from OpenReview)
  - Rising trends (would use citation_history when populated)
  - Tag frequency leaderboards (from paper_tags)
  - Source breakdown

Read as: this is the shape of queries HighSignal would run on the data we've built.
"""

from __future__ import annotations

from researchpapers.ch_db import connect as ch_connect


def render() -> str:
    out: list[str] = ["# HighSignal-shape weekly digest\n"]
    with ch_connect() as c:
        # Source breakdown
        rows = c.query(
            "SELECT source, count() AS n FROM papers GROUP BY source ORDER BY n DESC"
        ).result_rows
        out.append("## Corpus by source\n")
        for s, n in rows:
            out.append(f"- **{s}**: {n:,}")
        out.append("")

        # Top-rated peer-reviewed papers
        rows = c.query("""
            SELECT
                r.paper_id, p.title, r.venue,
                avg(r.rating) AS avg_rating,
                count() AS n_reviews,
                any(r.decision) AS decision
            FROM openreview_reviews r
            LEFT JOIN papers p ON p.paper_id = r.paper_id
            WHERE r.rating IS NOT NULL
            GROUP BY r.paper_id, p.title, r.venue
            HAVING n_reviews >= 3
            ORDER BY avg_rating DESC, count() DESC
            LIMIT 10
        """).result_rows
        if rows:
            out.append("## Top-rated peer-reviewed submissions\n")
            for pid, title, venue, rating, n, decision in rows:
                oid = pid.replace("openreview:", "")
                link = f"https://openreview.net/forum?id={oid}"
                title_clean = (title or "(no title)")[:100]
                out.append(
                    f"- **{rating:.2f}** ({venue}, {n} reviews, decision={decision or 'n/a'}) "
                    f"[{title_clean}]({link})"
                )
            out.append("")

        # Tag frequency leaderboards
        rows = c.query("""
            SELECT tagger, countDistinct(paper_id) AS n
            FROM paper_tags
            GROUP BY tagger
            ORDER BY n DESC
        """).result_rows
        if rows:
            out.append("## Tag coverage by tagger\n")
            for tagger, n in rows:
                out.append(f"- `{tagger}`: {n:,} papers tagged")
            out.append("")

        # Top tags overall (from any tagger)
        rows = c.query("""
            SELECT arrayJoin(tags) AS tag, countDistinct(paper_id) AS n
            FROM paper_tags
            GROUP BY tag
            ORDER BY n DESC
            LIMIT 25
        """).result_rows
        if rows:
            out.append("## Top tags overall\n")
            for tag, n in rows:
                out.append(f"- `{tag}` — {n} papers")
            out.append("")

        # Tag × reviewer rating: which research areas reviewers are most excited about.
        # Joins spaCy tags on OpenReview submissions to their reviewer ratings.
        rows = c.query("""
            WITH paper_avg_rating AS (
                SELECT paper_id, avg(rating) AS avg_rating, count() AS n_reviews
                FROM openreview_reviews
                WHERE rating IS NOT NULL
                GROUP BY paper_id
                HAVING n_reviews >= 3
            )
            SELECT
                tag,
                round(avg(par.avg_rating), 2) AS mean_rating,
                count() AS n_papers
            FROM paper_tags t FINAL
            ARRAY JOIN tags AS tag
            JOIN paper_avg_rating par ON par.paper_id = t.paper_id
            WHERE t.tagger = 'spacy_v2'
            GROUP BY tag
            HAVING n_papers >= 10
            ORDER BY mean_rating DESC
            LIMIT 25
        """).result_rows
        if rows:
            out.append("## Tags reviewers are most excited about\n")
            out.append("Mean ICLR/NeurIPS reviewer rating across papers tagged with each topic. "
                       "Min 10 papers per tag, min 3 reviews per paper.\n")
            out.append("| Tag | Avg rating | Papers |")
            out.append("|---|---:|---:|")
            for tag, rating, n in rows:
                out.append(f"| `{tag}` | {rating} | {n} |")
            out.append("")

        # Venue review activity (last cycle proxy)
        rows = c.query("""
            SELECT
                venue,
                count() AS n_reviews,
                countDistinct(paper_id) AS n_papers,
                round(avg(rating), 2) AS avg_rating,
                round(countIf(rating >= 8) * 100.0 / count(), 1) AS pct_strong_accept
            FROM openreview_reviews
            WHERE rating IS NOT NULL
            GROUP BY venue
            ORDER BY n_reviews DESC
        """).result_rows
        if rows:
            out.append("## Venue health\n")
            out.append("| Venue | Papers | Reviews | Avg rating | % strong (≥8) |")
            out.append("|---|---:|---:|---:|---:|")
            for v, n_rev, n_pap, avg, pct in rows:
                out.append(f"| {v} | {n_pap:,} | {n_rev:,} | {avg} | {pct}% |")
            out.append("")

    return "\n".join(out)


def main() -> None:
    print(render())


if __name__ == "__main__":
    main()
