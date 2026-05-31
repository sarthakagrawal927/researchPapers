# researchPapers

Multi-source academic-paper data platform on ClickHouse.
**488k papers** across arxiv, OpenReview, bioRxiv, medRxiv — with semantic
search, citation graph PageRank, peer-review aggregations, MLX/spaCy
auto-tagging, and HighSignal-style insight surfaces (sleepers, hot now,
papers-like-this, authors-by-tag).

Stack: ClickHouse 24.10 (Dockerized) · FastAPI · Astro 5 + React + Tailwind
+ shadcn/ui · sentence-transformers · MLX (Qwen2.5-3B-4bit) · spaCy v2.

## Status

- 488,491 papers ingested, ~1.05M paper→paper edges, full-corpus PageRank
  computed, all papers embedded (all-MiniLM-L6-v2, 384-dim) and clustered
  into 64 semantic clusters.
- Runtime is **ClickHouse-only**. Postgres remains as an optional
  dependency for legacy CLI commands (e.g. `ingest`, `download-pdfs`) but
  is not required for the API, the frontend, or any current pipeline.
- The Astro frontend is fully wired to the FastAPI backend. See `DEPLOY.md`.
- GitHub: <https://github.com/sarthak-fleet/researchPapers>

## Quickstart (warm — from a dump)

If you have the `researchpapers_data_*.tar.gz` dump in hand, this is the
fastest path to a running system. Needs Docker + `uv`.

```bash
git clone https://github.com/sarthak-fleet/researchPapers
cd researchPapers
./scripts/deploy.sh /path/to/researchpapers_data_*.tar.gz
# CH on :8123, FastAPI on :8000

# Frontend (separate terminal)
cd web && npm install && npm run dev   # http://127.0.0.1:4321
```

For LAN/CDN deployments, see **[DEPLOY.md](DEPLOY.md)**.

## Quickstart (cold — rebuild from scratch)

A full rebuild takes hours. Outline:

```bash
docker compose up -d clickhouse                  # CH on :8123
uv sync                                          # python deps
uv run papers select-top --n 400000              # OpenAlex top-N CS metadata
uv run papers ingest-openreview                  # NeurIPS/ICLR + reviews
uv run papers ingest-biorxiv                     # bioRxiv + medRxiv
uv run papers backfill-references                # paper→paper edges
uv run papers refresh-metadata                   # arxiv API fixups for top papers
uv run papers pagerank-full                      # writes paper_scores_v2
uv run papers embed                              # all-MiniLM-L6-v2 (~17 min)
uv run papers cluster-embeddings                 # MiniBatchKMeans, 64 clusters
uv run papers spacy-tag-v2                       # noun-chunk tags (CPU)
uv run papers mlx-tag-v3 --shards 3              # premium subset, MLX on Apple Silicon
uv run papers export-ch                          # write web/public/data/*.json
uv run papers api-serve --host 0.0.0.0 --port 8000
```

## Architecture

```
                                   ┌──────────────────────┐
   arxiv API ──┐                   │  ClickHouse (papers) │
   OpenAlex ───┼──► ingesters ───► │  papers              │
   OpenReview ─┤                   │  paper_tags          │
   bioRxiv ────┘                   │  references_paper    │
                                   │  citation_history    │
                                   │  paper_embeddings    │
                                   │  paper_clusters      │
                                   │  paper_metadata_v2   │ ◄── overlay
                                   │  paper_scores_v2     │ ◄── overlay
                                   └──────────┬───────────┘
                                              │
                            ┌─────────────────┼─────────────────┐
                            ▼                 ▼                 ▼
                       ch_exports.py     pagerank_full.py   refresh_metadata.py
                       (JSON for FE)     (scipy.sparse)     (arxiv API title fix)
                            │                                   │
                            ▼                                   ▼
                       web/public/data/*.json            paper_metadata_v2
                            │
                            ▼
                       Astro 5 + React (web/)
                            ▲
                            │
                       FastAPI (src/researchpapers/api.py)
                       /search /papers /sleepers /hot /similar
                       /semantic-search /tags /authors /reviews
```

## API endpoints

All under the FastAPI server (`uv run papers api-serve`):

| Endpoint | What |
| --- | --- |
| `GET /healthz`, `GET /stats` | health + corpus stats |
| `GET /search?q=...` | full-text-ish search via CH `LIKE` |
| `GET /papers/{paper_id}` | canonical paper detail (joined w/ overlays) |
| `GET /semantic-search?q=...` | encodes q via MiniLM, `cosineDistance` over `paper_embeddings` |
| `GET /sleepers` | papers with late citation spikes |
| `GET /hot` | recent papers gaining attention |
| `GET /similar/{paper_id}` | nearest neighbours by embedding |
| `GET /tags/top-rated` | tag → mean rating cross-join (uses OpenReview scores) |
| `GET /tags/{tag}` | drilldown: papers under a tag |
| `GET /authors/by-tag/{tag}` | top authors per topic |
| `GET /authors/by-id/{openalex_id}` | full author profile |
| `GET /reviews/top-rated` | best-reviewed OpenReview papers |

## CLI cheatsheet

```bash
uv run papers api-serve                  # FastAPI on :8000
uv run papers export-ch                  # write JSON files for the static FE
uv run papers refresh-metadata           # pull arxiv API + OpenAlex into paper_metadata_v2
uv run papers pagerank-full              # full-corpus PageRank → paper_scores_v2
uv run papers embed                      # all-MiniLM-L6-v2 embed → paper_embeddings
uv run papers cluster-embeddings         # MiniBatchKMeans → paper_clusters
uv run papers snapshot-citations         # append to citation_history
uv run papers spacy-tag-v2               # POS-only noun-chunk tagger (arxiv)
uv run papers spacy-tag-source --source openreview
uv run papers mlx-tag-v3 --shards 3      # grouped-prompt MLX, premium subset
uv run papers status                     # progress / row counts
```

Full list: `uv run papers --help`.

## Data correction overlays

Some primary data has known defects: OpenAlex returns *latest revision dates*
for arxiv preprints (so "Attention Is All You Need" shows 2025 instead of
2017), and a handful of arxiv IDs have cross-contaminated titles/abstracts
in OpenAlex's index. We patch around these with two overlay tables and two
ClickHouse UDFs:

- **`paper_metadata_v2`** (ReplacingMergeTree) — corrected titles +
  author OpenAlex IDs pulled from the arxiv API. Populated by
  `papers refresh-metadata`. Joined in via `LEFT JOIN ... ON paper_id`
  with `COALESCE(nullIf(m.title, ''), p.title)`.
- **`paper_scores_v2`** (ReplacingMergeTree) — full-corpus PageRank,
  written by `papers pagerank-full`. Used because `papers.pagerank_score`
  can't be `ALTER UPDATE`d cheaply (partition key on `submitted_date`).
- **`effective_year(arxiv_id, submitted_date, publication_year)`** —
  parses the YYMM prefix of arxiv IDs (`2410.xxxxx` → 2024) and only
  falls back to OpenAlex's date if no arxiv prefix is present. Defined
  in `clickhouse/init/02_functions.sql` so it survives container
  restarts; `deploy.sh` re-applies it after restores.
- **`effective_date(arxiv_id, submitted_date)`** — same idea, returns a
  `Date` for chart axes.

## Repo layout

```
src/researchpapers/
  api.py                FastAPI server (all read endpoints)
  cli.py                typer entrypoint (uv run papers ...)
  ch_db.py              ClickHouse connection helper
  ch_exports.py         CH → JSON exports for the static frontend
  exporter.py           legacy multi-source exporter (CH-only)
  refresh_metadata.py   arxiv API + OpenAlex → paper_metadata_v2
  pagerank_full.py      scipy.sparse PageRank → paper_scores_v2
  embed.py              sentence-transformers → paper_embeddings
  cluster_embeddings.py MiniBatchKMeans → paper_clusters
  mlx_tag_v3.py         MLX grouped-prompt tagger (Apple Silicon)
  noun_tag_v2.py        spaCy v2 POS-only noun-chunk tagger
  llm_tag.py            LM Studio / Ollama / MLX HTTP tagger
  openalex.py           OpenAlex client + top-cited fetcher
  openreview_ingest.py  NeurIPS/ICLR API + reviews
  biorxiv_ingest.py     bioRxiv + medRxiv + chemRxiv
  citation_history.py   snapshots → citation_history
  highsignal_report.py  sample HighSignal-style report
  watcher.py            auto re-analytics loop
clickhouse/init/
  01_schema.sql         papers, paper_tags, references_paper, ...
  02_functions.sql      effective_year / effective_date UDFs
migrations/             legacy Postgres migrations (kept for cold restore)
scripts/
  deploy.sh             unpack dump, start CH, apply UDFs, serve API
  dump_data.sh          tar CH volume + web/public/data/ exports
  migrate_pg_to_ch.py   one-shot Postgres → ClickHouse migrator
web/
  src/pages/index.astro      dashboard with React islands
  src/pages/digest.astro     HighSignal-style digest page
  src/components/            shadcn/ui + TanStack tables + charts
  public/data/*.json         exported aggregations
  public/api-config.js       runtime API base override
DEPLOY.md                    three deployment shapes (host / LAN / CDN)
```

## Known issues / deferred

- **OpenAlex citation undercount.** Some preprints have lower
  `cited_by_count` in OpenAlex than they do on Google Scholar
  (Attention Is All You Need shows ~6,551). Could augment with
  Semantic Scholar `/paper/batch` for the top-1k papers; deferred.
- **Cross-contaminated OpenAlex records.** A small fraction of arxiv
  IDs return another paper's abstract/DOI/tags in OpenAlex. Title is
  fixed via `paper_metadata_v2`; abstracts and tags are not yet
  re-pulled from arxiv. Would require an `arxiv abstract refresh` job.
- **Author disambiguation** is only populated for the top-2000 papers
  (those refreshed via `papers refresh-metadata`). Non-top papers fall
  back to author name strings.
- **No Vercel/CF deploy yet.** The static FE builds clean (see
  `DEPLOY.md`) but hasn't been pushed to a CDN — the user prefers
  same-host deploy unless going public.
- **OrbStack 2.1.3 + macOS 26 instability.** Apple
  Virtualization.framework occasionally kills the OrbStack VM backend
  silently. Workaround: `~/.orbstack/bin/orb start` from the CLI
  (the GUI doesn't always re-spawn). Linux Docker daemons are fine.

## Environment

`.env` is optional. `CONTACT_EMAIL` is read by the polite-scraping headers;
`POSTGRES_URL` is only needed for the few legacy CLIs that still touch
Postgres (ingest, download-pdfs, extract-urls). See `.env.example`.
