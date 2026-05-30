from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Annotated

import typer

from pathlib import Path

from researchpapers import (
    analytics,
    arxiv,
    biorxiv_ingest as biorxiv_mod,
    charts,
    citation_history as citation_history_mod,
    clusters as clusters_mod,
    db,
    exporter,
    graph as graph_mod,
    ingest as ingest_mod,
    keybert_tag as keybert_tag_mod,
    llm_tag as llm_tag_mod,
    noun_tag as noun_tag_mod,
    mlx_tag_v2 as mlx_tag_v2_mod,
    noun_tag_v2 as noun_tag_v2_mod,
    openalex,
    openreview_ingest as openreview_mod,
    pdfs,
    semantic_scholar,
    tag_eval as tag_eval_mod,
    url_extract,
    watcher as watcher_mod,
)
from researchpapers.config import DATA_DIR, PROJECT_ROOT, load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)
rank_app = typer.Typer(no_args_is_help=True)
app.add_typer(rank_app, name="rank", help="Leaderboards over the ingested data.")


@app.command("init-db")
def init_db_cmd() -> None:
    """Apply any pending Postgres migrations."""
    settings = load_settings()
    applied = db.init_db(settings)
    if applied:
        typer.echo(f"applied: {', '.join(applied)}")
    else:
        typer.echo("schema already up to date")


@app.command()
def fetch(
    category: Annotated[str, typer.Option(help="arXiv category, e.g. cs.AI")] = "cs.AI",
    days: Annotated[int, typer.Option(help="Window size in days, ending today")] = 30,
    since: Annotated[str | None, typer.Option(help="Explicit start date (YYYY-MM-DD)")] = None,
    until: Annotated[str | None, typer.Option(help="Explicit end date (YYYY-MM-DD)")] = None,
) -> None:
    """Hit the arXiv Atom API and upsert papers into the DB."""
    settings = load_settings()
    end = date.fromisoformat(until) if until else date.today()
    start = date.fromisoformat(since) if since else end - timedelta(days=days)
    typer.echo(f"fetching {category} from {start} to {end}")
    n = arxiv.fetch_papers(settings, category=category, since=start, until=end)
    typer.echo(f"upserted {n} papers")


@app.command("download-pdfs")
def download_pdfs_cmd(
    limit: Annotated[int | None, typer.Option(help="Max PDFs to download this run")] = None,
) -> None:
    """Politely download PDFs for papers that don't have one cached yet."""
    settings = load_settings()
    n = pdfs.download_pdfs(settings, limit=limit)
    typer.echo(f"downloaded {n} PDFs")


@app.command("extract-urls")
def extract_urls_cmd(
    limit: Annotated[int | None, typer.Option(help="Max PDFs to process this run")] = None,
) -> None:
    """Extract URLs from downloaded PDFs into references_url."""
    settings = load_settings()
    papers_done, urls_inserted = url_extract.extract_all(settings, limit=limit)
    typer.echo(f"processed {papers_done} PDFs, inserted {urls_inserted} URL edges")


@app.command("ingest-openreview")
def ingest_openreview_cmd() -> None:
    """Pull NeurIPS / ICLR / ICML / COLM submissions + reviews from OpenReview into ClickHouse."""
    c = openreview_mod.ingest()
    typer.echo(f"submissions={c.get('submissions')} reviews={c.get('reviews')}")


@app.command("ingest-biorxiv")
def ingest_biorxiv_cmd(
    server: Annotated[str, typer.Option(help="biorxiv | medrxiv")] = "biorxiv",
    days: Annotated[int, typer.Option(help="Window size, days back from today")] = 365,
) -> None:
    """Pull bioRxiv or medRxiv preprints into ClickHouse."""
    from datetime import date, timedelta
    until = date.today()
    since = until - timedelta(days=days)
    n = biorxiv_mod.ingest_biorxiv_medrxiv(server=server, since=since, until=until)
    typer.echo(f"{server}: {n} papers ingested")


@app.command("ingest-chemrxiv")
def ingest_chemrxiv_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
) -> None:
    """Pull chemRxiv preprints into ClickHouse."""
    n = biorxiv_mod.ingest_chemrxiv(limit=limit)
    typer.echo(f"chemrxiv: {n} papers ingested")


@app.command("snapshot-citations")
def snapshot_citations_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
) -> None:
    """Pull current cited_by_count from OpenAlex; write into citation_history."""
    n = citation_history_mod.snapshot_today(limit=limit)
    typer.echo(f"wrote {n} citation_history rows")


@app.command("select-top")
def select_top_cmd(
    n: Annotated[int, typer.Option(help="Number of top-cited papers to select")] = 10000,
    all_fields: Annotated[bool, typer.Option(help="Drop the CS-only filter — pull ALL arxiv works")] = False,
) -> None:
    """Pick the top-N papers by citation count via OpenAlex; upsert into papers."""
    settings = load_settings()
    scanned, inserted = openalex.select_top(settings, n=n, cs_only=not all_fields)
    typer.echo(f"scanned {scanned} works, inserted/updated {inserted} arxiv-resolvable papers")


@app.command()
def ingest(
    limit: Annotated[int | None, typer.Option(help="Max papers to ingest this run")] = None,
    workers: Annotated[int, typer.Option(help="Parallel pdfminer worker processes")] = 4,
    max_in_flight: Annotated[int, typer.Option(help="Max submitted extracts in flight")] = 8,
) -> None:
    """Streaming: download PDFs (rate-limited 3s/req), extract text+URLs in parallel, gzip into DB."""
    settings = load_settings()
    c = ingest_mod.ingest_all(
        settings, limit=limit, workers=workers, max_in_flight=max_in_flight
    )
    typer.echo(
        f"processed={c['processed']} urls_inserted={c['urls_inserted']} "
        f"pdf_failed={c['pdf_failed']} empty_text={c['empty_text']}"
    )


@app.command("fetch-citations")
def fetch_citations_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers to query this run")] = None,
) -> None:
    """Pull paper->paper references from Semantic Scholar. Requires SEMANTIC_SCHOLAR_API_KEY for usable rates."""
    settings = load_settings()
    papers_done, edges = semantic_scholar.fetch_citations(settings, limit=limit)
    typer.echo(f"updated {papers_done} papers, wrote {edges} reference edges")


@rank_app.command("hosts")
def rank_hosts(
    top: Annotated[int, typer.Option(help="Top-N rows")] = 30,
    fmt: Annotated[str, typer.Option("--format", help="table|csv|json")] = "table",
) -> None:
    """Top hosts (domains) by number of citing papers."""
    analytics.top_hosts(load_settings(), top=top, fmt=fmt)  # type: ignore[arg-type]


@rank_app.command("urls")
def rank_urls(
    top: Annotated[int, typer.Option(help="Top-N rows")] = 30,
    fmt: Annotated[str, typer.Option("--format", help="table|csv|json")] = "table",
) -> None:
    """Top exact URLs by number of citing papers."""
    analytics.top_urls(load_settings(), top=top, fmt=fmt)  # type: ignore[arg-type]


@rank_app.command("papers")
def rank_papers(
    top: Annotated[int, typer.Option(help="Top-N rows")] = 30,
    fmt: Annotated[str, typer.Option("--format", help="table|csv|json")] = "table",
) -> None:
    """Top cited papers. Requires `papers fetch-citations` to have run."""
    analytics.top_cited_papers(load_settings(), top=top, fmt=fmt)  # type: ignore[arg-type]


@app.command("re-extract-urls-from-text")
def re_extract_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers to re-process")] = None,
    keep: Annotated[bool, typer.Option(help="Don't delete existing rows before re-insert")] = False,
) -> None:
    """Re-run URL extraction from the gzipped text already in paper_texts (no PDF re-download)."""
    settings = load_settings()
    papers_done, urls = url_extract.re_extract_all_from_text(
        settings, limit=limit, replace=not keep
    )
    typer.echo(f"re-processed {papers_done} papers, inserted {urls} URL rows")


@app.command("backfill-references")
def backfill_references_cmd() -> None:
    """Re-query OpenAlex for referenced_works (paper->paper edges) and store in references_paper."""
    settings = load_settings()
    papers_done, edges = openalex.backfill_referenced_works(settings)
    typer.echo(f"backfilled {papers_done} papers, wrote {edges} reference edges")


@app.command("cluster-abstracts")
def cluster_abstracts_cmd(
    n_clusters: Annotated[int, typer.Option(help="K for KMeans on TF-IDF vectors")] = 30,
) -> None:
    """TF-IDF + KMeans on paper abstracts. Orthogonal signal to citation-graph communities."""
    settings = load_settings()
    c = clusters_mod.cluster_abstracts(settings, n_clusters=n_clusters)
    typer.echo(f"docs={c['docs']} clusters={c['clusters']}")


@app.command("detect-communities")
def detect_communities_cmd() -> None:
    """Louvain community detection on the within-corpus subgraph. Writes papers.community_id."""
    settings = load_settings()
    c = graph_mod.detect_communities(settings)
    typer.echo(
        f"nodes={c['nodes']} edges={c['edges']} "
        f"communities={c['communities']} assigned={c['assigned']}"
    )


@app.command("watch")
def watch_cmd(
    threshold: Annotated[int, typer.Option(help="Re-run analytics chain every N new papers")] = 10000,
    force_boot: Annotated[bool, typer.Option(help="Force analytics run at boot even if no new papers")] = False,
) -> None:
    """Long-running loop: re-runs analytics + rebuilds web every THRESHOLD new papers ingested."""
    watcher_mod.watch_loop(threshold=threshold, force_boot_run=force_boot)


@app.command("spacy-tag")
def spacy_tag_cmd(
    model: Annotated[str, typer.Option(help="en_core_web_sm | en_core_web_lg | en_core_sci_lg")] = "en_core_web_sm",
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
    any_order: Annotated[bool, typer.Option(help="Don't prioritize high-citation papers")] = False,
) -> None:
    """Extract noun-chunk + PROPN tags via spaCy."""
    settings = load_settings()
    c = noun_tag_mod.tag_papers(settings, model_name=model, limit=limit, only_top_cited=not any_order)
    typer.echo(
        f"tagged={c.get('tagged')} skipped={c.get('skipped')} "
        f"elapsed={c.get('elapsed_seconds')}s papers/sec={c.get('papers_per_sec')}"
    )


@app.command("mlx-tag-v2")
def mlx_tag_v2_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
    any_order: Annotated[bool, typer.Option(help="Don't prioritize high-citation papers")] = False,
) -> None:
    """Direct in-process MLX inference (no HTTP). Loads model once, reuses across all prompts."""
    settings = load_settings()
    c = mlx_tag_v2_mod.tag_papers(settings, limit=limit, only_top_cited=not any_order)
    typer.echo(
        f"tagged={c.get('tagged')} failed={c.get('failed')} skipped={c.get('skipped')} "
        f"elapsed={c.get('elapsed_seconds')}s papers/sec={c.get('papers_per_sec')} "
        f"completion_tok/sec={c.get('completion_tok_per_sec')}"
    )


@app.command("spacy-tag-v2")
def spacy_tag_v2_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
    any_order: Annotated[bool, typer.Option(help="Don't prioritize high-citation papers")] = False,
    batch_papers: Annotated[int | None, typer.Option(help="If set, process in chunks of this size, re-sampling free RAM between chunks. Recommended: 25000.")] = None,
    n_process: Annotated[int | None, typer.Option(help="Override spaCy worker count (skips RAM picker)")] = None,
    max_procs: Annotated[int | None, typer.Option(help="Cap for the RAM-aware picker")] = None,
) -> None:
    """Faster spaCy: parser disabled, POS-only chunker. Stores in noun_tags_v2_json + CH paper_tags."""
    settings = load_settings()
    c = noun_tag_v2_mod.tag_papers(
        settings,
        limit=limit,
        only_top_cited=not any_order,
        batch_papers=batch_papers,
        n_process=n_process,
        max_procs=max_procs,
    )
    typer.echo(
        f"tagged={c.get('tagged')} skipped={c.get('skipped')} "
        f"elapsed={c.get('elapsed_seconds')}s papers/sec={c.get('papers_per_sec')}"
    )


@app.command("api-serve")
def api_serve_cmd(
    host: Annotated[str, typer.Option(help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes")] = False,
) -> None:
    """Serve the researchPapers HTTP API (FastAPI + uvicorn over ClickHouse)."""
    import uvicorn
    uvicorn.run("researchpapers.api:app", host=host, port=port, reload=reload)


@app.command("mlx-tag-v3")
def mlx_tag_v3_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
    premium_only: Annotated[bool, typer.Option(help="Restrict to hybrid-filter premium subset")] = False,
    group_size: Annotated[int, typer.Option(help="Papers per LLM call")] = 4,
    max_tokens: Annotated[int, typer.Option(help="Max output tokens per group")] = 700,
    any_order: Annotated[bool, typer.Option(help="Don't prioritize high-citation papers")] = False,
    shard: Annotated[int, typer.Option(help="Shard index (0..total_shards-1) for parallel runs")] = 0,
    total_shards: Annotated[int, typer.Option(help="Total parallel shards; >1 enables sharded queue")] = 1,
    model: Annotated[str | None, typer.Option(help="Override MLX model.")] = None,
    throttle_seconds: Annotated[float, typer.Option(help="Sleep between groups to yield GPU time")] = 0.0,
) -> None:
    """MLX v3: grouped-prompt batching for fast LLM tagging.

    For parallelism: run N processes with --shard i --total-shards N.
    They tag disjoint paper subsets via cityHash64(paper_id) % total_shards.
    """
    from researchpapers import mlx_tag_v3 as mod
    settings = load_settings()
    c = mod.tag_papers(
        settings,
        limit=limit,
        only_top_cited=not any_order,
        premium_only=premium_only,
        group_size=group_size,
        max_tokens=max_tokens,
        shard=shard,
        total_shards=total_shards,
        model_name=model,
        throttle_seconds=throttle_seconds,
    )
    typer.echo(f"tagged={c.get('tagged')} skipped={c.get('skipped')} failed={c.get('failed')} groups={c.get('groups')}")
    typer.echo(f"elapsed={c.get('elapsed_seconds')}s  papers/sec={c.get('papers_per_sec')}")
    typer.echo(f"completion_tok/sec={c.get('completion_tok_per_sec')}")


@app.command("embed")
def embed_cmd(
    source: Annotated[str | None, typer.Option(help="Limit to one source")] = None,
    limit: Annotated[int | None, typer.Option(help="Max papers")] = None,
    batch_size: Annotated[int, typer.Option(help="Encoder batch size")] = 128,
) -> None:
    """Embed papers (title+abstract) into CH paper_embeddings via all-MiniLM-L6-v2."""
    from researchpapers import embed
    c = embed.embed_papers(source=source, limit=limit, batch_size=batch_size)
    typer.echo(f"embedded={c.get('embedded')} elapsed={c.get('elapsed_seconds')}s "
               f"papers/sec={c.get('papers_per_sec')}")


@app.command("spacy-tag-source")
def spacy_tag_source_cmd(
    source: Annotated[str, typer.Option(help="Source to tag (e.g. openreview, biorxiv, medrxiv)")],
    limit: Annotated[int | None, typer.Option(help="Max papers")] = None,
    batch_papers: Annotated[int, typer.Option(help="Batch size for RAM re-sampling")] = 5000,
    max_procs: Annotated[int | None, typer.Option(help="Cap for RAM-aware worker picker")] = None,
) -> None:
    """Run spaCy v2 on a non-arxiv source in ClickHouse. Writes tags to CH paper_tags."""
    c = noun_tag_v2_mod.tag_multi_source(
        source=source,
        limit=limit,
        batch_papers=batch_papers,
        max_procs=max_procs,
    )
    typer.echo(
        f"tagged={c.get('tagged')} skipped={c.get('skipped')} "
        f"elapsed={c.get('elapsed_seconds')}s papers/sec={c.get('papers_per_sec')}"
    )


@app.command("keybert-tag")
def keybert_tag_cmd(
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
    any_order: Annotated[bool, typer.Option(help="Don't prioritize high-citation papers")] = False,
) -> None:
    """Extract tags via KeyBERT (sentence-transformer embedding + n-gram scoring)."""
    settings = load_settings()
    c = keybert_tag_mod.tag_papers(settings, limit=limit, only_top_cited=not any_order)
    typer.echo(
        f"tagged={c.get('tagged')} skipped={c.get('skipped')} "
        f"elapsed={c.get('elapsed_seconds')}s papers/sec={c.get('papers_per_sec')}"
    )


@app.command("eval-taggers")
def eval_taggers_cmd(
    sample_n: Annotated[int, typer.Option(help="Sample papers to print side-by-side")] = 12,
) -> None:
    """Compare spaCy + KeyBERT against LLM tags on papers tagged by all three."""
    settings = load_settings()
    report = tag_eval_mod.evaluate(settings, sample_n=sample_n)
    tag_eval_mod.print_report(report)


@app.command("llm-tag")
def llm_tag_cmd(
    backend: Annotated[str, typer.Option(help="lm-studio | ollama | mlx")] = "lm-studio",
    model: Annotated[str | None, typer.Option(help="Model name (e.g. qwen2.5-7b-instruct)")] = None,
    limit: Annotated[int | None, typer.Option(help="Max papers this run")] = None,
    any_order: Annotated[bool, typer.Option(help="Don't prioritize high-citation papers")] = False,
    concurrency: Annotated[int, typer.Option(help="Parallel in-flight requests to the LLM")] = 4,
    premium_only: Annotated[bool, typer.Option(help="Restrict to hybrid-filter premium subset (~102k papers)")] = False,
) -> None:
    """Tag papers via local LLM (LM Studio default). Needs the server running with parallelism enabled."""
    settings = load_settings()
    c = llm_tag_mod.tag_papers(
        settings,
        backend=llm_tag_mod.Backend(backend),
        model=model,
        limit=limit,
        only_top_cited=not any_order,
        concurrency=concurrency,
        premium_only=premium_only,
    )
    typer.echo("--- llm-tag benchmark report ---")
    typer.echo(f"tagged={c.get('tagged')} failed={c.get('failed')} skipped={c.get('skipped')}")
    typer.echo(f"elapsed={c.get('elapsed_seconds')}s  papers/sec={c.get('papers_per_sec')}")
    typer.echo(
        f"completion_tok/sec={c.get('completion_tok_per_sec')}  "
        f"total_tok/sec={c.get('total_tok_per_sec')}"
    )
    typer.echo(
        f"avg_prompt_tokens={c.get('avg_prompt_tokens')}  "
        f"avg_completion_tokens={c.get('avg_completion_tokens')}"
    )


@app.command("compute-graph-scores")
def compute_graph_scores_cmd(
    katz_alpha: Annotated[float, typer.Option(help="Katz decay (smaller = more weight on direct cites)")] = 0.05,
    pagerank_damping: Annotated[float, typer.Option(help="PageRank damping factor")] = 0.85,
) -> None:
    """Compute PageRank + Katz + in-corpus-degree per paper. Detect cycles. Stores in papers + citation_cycles."""
    settings = load_settings()
    c = graph_mod.compute_scores(
        settings, katz_alpha=katz_alpha, pagerank_damping=pagerank_damping
    )
    typer.echo(
        f"nodes={c['nodes']} edges={c['edges']} scored={c['scored']} cycles={c['cycles_found']}"
    )


@app.command("resolve-cited-works")
def resolve_cited_cmd(
    top: Annotated[int, typer.Option(help="Resolve the top-N most-cited works in our corpus")] = 200,
) -> None:
    """Resolves the head of the cited-works distribution to titles via OpenAlex (200 ≈ 4 batched calls)."""
    settings = load_settings()
    n = openalex.resolve_top_cited(settings, top=top)
    typer.echo(f"resolved {n} cited works")


@app.command("export-json")
def export_json_cmd(
    out: Annotated[str, typer.Option(help="Output directory")] = "",
    top: Annotated[int, typer.Option(help="Top-N rows in each leaderboard")] = 200,
) -> None:
    """Export aggregations as JSON files for the Astro app."""
    settings = load_settings()
    out_dir = Path(out) if out else PROJECT_ROOT / "web" / "public" / "data"
    paths = exporter.export_all(settings, out_dir, top=top)
    for p in paths:
        typer.echo(f"wrote {p}")


@app.command("highsignal-report")
def highsignal_report_cmd(
    out: Annotated[str | None, typer.Option(help="Write to this file (markdown). Stdout if omitted")] = None,
) -> None:
    """Render a sample HighSignal-style markdown digest from ClickHouse."""
    from researchpapers import highsignal_report
    md = highsignal_report.render()
    if out:
        Path(out).write_text(md)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(md)


@app.command("export-ch")
def export_ch_cmd(
    out: Annotated[str, typer.Option(help="Output directory")] = "",
) -> None:
    """Export review/source aggregations from ClickHouse to JSON for the Astro app."""
    from researchpapers import ch_exports
    out_dir = Path(out) if out else PROJECT_ROOT / "web" / "public" / "data"
    paths = ch_exports.export_review_data(out_dir)
    for p in paths:
        typer.echo(f"wrote {p}")


@app.command("refresh-web")
def refresh_web_cmd(
    top: Annotated[int, typer.Option(help="Top-N rows in each leaderboard")] = 200,
) -> None:
    """Re-export JSON + rebuild the Astro site. Run after ingest progresses to see fresh data."""
    import subprocess
    settings = load_settings()
    out_dir = PROJECT_ROOT / "web" / "public" / "data"
    exporter.export_all(settings, out_dir, top=top)
    typer.echo("exported JSON, building Astro site...")
    result = subprocess.run(
        ["npm", "run", "build"], cwd=str(PROJECT_ROOT / "web"), capture_output=True, text=True
    )
    if result.returncode != 0:
        typer.echo(result.stderr)
        raise typer.Exit(1)
    typer.echo("built. Preview at http://127.0.0.1:4321/")


@app.command("charts")
def charts_cmd(
    out: Annotated[str, typer.Option(help="Output directory for PNGs")] = "",
    top: Annotated[int, typer.Option(help="Top-N for the bar charts")] = 30,
) -> None:
    """Generate matplotlib charts from the references_url table."""
    settings = load_settings()
    out_dir = Path(out) if out else DATA_DIR / "charts"
    paths = charts.make_all(settings, out_dir, top=top)
    for p in paths:
        typer.echo(f"wrote {p}")


@app.command("status")
def status_cmd() -> None:
    """Quick progress check: how much of the ingest queue is done."""
    settings = load_settings()
    with db.connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              COUNT(*)                                    AS total,
              COUNT(urls_extracted_at)                    AS ingested,
              COUNT(*) FILTER (WHERE urls_extracted_at IS NULL) AS pending,
              (SELECT COUNT(*) FROM references_url)       AS url_edges,
              (SELECT COUNT(*) FROM paper_texts)          AS texts_stored,
              (SELECT pg_size_pretty(SUM(length(content_gz))::bigint) FROM paper_texts) AS gz_bytes
            FROM papers
            """
        )
        r = cur.fetchone()
    typer.echo(
        f"papers: {r['ingested']}/{r['total']} ingested  ({r['pending']} pending)\n"
        f"url edges: {r['url_edges']}\n"
        f"texts stored: {r['texts_stored']} ({r['gz_bytes']} gzipped)"
    )


if __name__ == "__main__":
    app()
