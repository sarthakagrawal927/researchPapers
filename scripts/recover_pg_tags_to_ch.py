"""One-shot recovery: pull any Postgres-tagged rows not yet in ClickHouse paper_tags
and write them to CH. Run before refactoring tagging pipelines to be CH-only.
"""

from __future__ import annotations

from researchpapers.ch_db import arxiv_paper_id, connect as ch_connect, write_paper_tags
from researchpapers.config import load_settings
from researchpapers.db import connect


def main() -> None:
    settings = load_settings()

    # Pull mlx tags from Postgres
    with connect(settings) as pg, pg.cursor() as cur:
        cur.execute(
            "SELECT arxiv_id, mlx_llm_v2_tags_json, mlx_llm_v2_tldr "
            "FROM papers WHERE mlx_llm_v2_tagged_at IS NOT NULL"
        )
        mlx_rows = cur.fetchall()

    # Which paper_ids are already in CH paper_tags under mlx_qwen3b_v3?
    with ch_connect() as ch:
        existing = ch.query(
            "SELECT DISTINCT paper_id FROM paper_tags FINAL WHERE tagger='mlx_qwen3b_v3'"
        ).result_rows
    existing_ids = {r[0] for r in existing}

    to_write = []
    for r in mlx_rows:
        tags = r["mlx_llm_v2_tags_json"] or []
        if not tags:
            continue
        pid = arxiv_paper_id(r["arxiv_id"])
        if pid in existing_ids:
            continue
        to_write.append((pid, "mlx_qwen3b_v3", list(tags), r["mlx_llm_v2_tldr"]))

    print(f"Postgres mlx tags total: {len(mlx_rows)}")
    print(f"Already in CH (mlx_qwen3b_v3): {len(existing_ids)}")
    print(f"To write: {len(to_write)}")

    if to_write:
        n = write_paper_tags(to_write, model_version="mlx-community/Qwen2.5-3B-Instruct-4bit")
        print(f"Wrote {n} rows to CH paper_tags")


if __name__ == "__main__":
    main()
