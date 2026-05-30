# researchPapers

Ingests arXiv papers and ranks the **external URLs** they cite — datasets, code repos, blogs, RFCs, leaderboards. Paper→paper citation edges via Semantic Scholar are wired up but deferred to v2.

## Quickstart (popularity path — what's actually wired)

```bash
cp .env.example .env                                # edit if needed
docker compose up -d                                # Postgres on :5433
uv sync                                             # install python deps
uv run papers init-db                               # apply migrations 001-003

# 1. Pick the corpus: top 10k cited CS papers via OpenAlex (~3 min)
uv run papers select-top --n 10000

# 2. Stream-ingest: per paper, download PDF -> text -> URLs -> gzip text -> DELETE PDF (~10-12 hours)
uv run papers ingest

# 3. Paper -> paper graph from OpenAlex referenced_works (~5 min, runs anytime after select-top)
uv run papers backfill-references
uv run papers resolve-cited-works --top 200

# 4. Optionally re-extract URLs with a refined regex, no PDF re-download (~minutes)
uv run papers re-extract-urls-from-text

# 5. Build the data viewer
cd web && npm install && cd ..
uv run papers refresh-web    # export JSON + astro build
cd web && npm run preview    # http://127.0.0.1:4321/
```

## CLI

| Command | What it does |
| --- | --- |
| `papers init-db` | Apply pending migrations from `migrations/` |
| `papers select-top --n 10000` | OpenAlex: top-N CS papers by citation count → `papers` table |
| `papers ingest [--limit N]` | Streaming: download PDF → extract text+URLs → gzip into Postgres → delete PDF |
| `papers backfill-references` | OpenAlex referenced_works → `references_paper` (paper→paper graph) |
| `papers resolve-cited-works --top N` | Resolve the most-cited OpenAlex IDs in our corpus to titles |
| `papers re-extract-urls-from-text` | Re-extract URLs from stored gzipped text (no PDF re-download — iterate on canonicalization cheaply) |
| `papers export-json` | Dump aggregations to `web/public/data/*.json` |
| `papers refresh-web` | `export-json` + `npm run build` |
| `papers status` | Quick progress: how much of the ingest queue is done |
| `papers charts` | Matplotlib PNGs (alternative to the Astro viewer) |
| `papers rank hosts \| urls \| papers` | CLI leaderboards as `--format table\|csv\|json` |

Legacy commands (still wired, not on the recommended path): `papers fetch` (arXiv API by category+date), `papers download-pdfs` + `papers extract-urls` (kept-on-disk variant of ingest), `papers fetch-citations` (Semantic Scholar instead of OpenAlex for paper graph).

Every stage is rerunnable and idempotent via DB uniqueness constraints + `ON CONFLICT DO NOTHING`.

## Schema (at a glance)

- `papers(arxiv_id PK, s2_paper_id, title, abstract, primary_category, …, pdf_path, pdf_fetched_at, urls_extracted_at)`
- `references_url(citing_arxiv_id, url_raw, url_canonical, host, scheme, context_snippet)`, unique on `(citing, url_canonical)`
- `references_paper(citing_arxiv_id, cited_s2_id, cited_arxiv_id, cited_doi, cited_title)`, with partial unique indexes per identifier kind

## Polite-scraping notes

- arXiv API: ≥3s between requests, identify yourself in `User-Agent`. Both encoded in `arxiv.py` / `pdfs.py` / `http.py`.
- Semantic Scholar without a key is ~1 req/sec. With a key (request at https://www.semanticscholar.org/product/api) it loosens substantially. The batch endpoint with `fields=references.title,references.externalIds` returns up to 500 papers + their refs per request, so we don't need per-paper `/references`.

## Future

- **Bigger scope offline**: the Kaggle arXiv metadata dump (~4GB, weekly) is the drop-in for the scope picker when you want "all of cs.AI historically" without 1000s of arXiv API calls. Replace `arxiv.fetch_papers` with a Kaggle JSON reader; the rest of the pipeline doesn't change.
- **Paper→paper graph**: run `papers fetch-citations` once you have an S2 key. Then `papers rank papers` lights up.
- **Knowledgebase reuse**: PDFs live at `data/pdfs/{arxiv_id}.pdf` so `../knowledgebase`'s `upload` source can ingest the same files later, without runtime coupling between the two projects.
- **Viz**: top-N + force-directed graph (sigma.js / react-force-graph) once the paper→paper graph exists.

## Layout

```
src/researchpapers/
  cli.py            # typer entrypoint
  config.py         # .env loader + Settings dataclass
  db.py             # psycopg connect + migration runner
  http.py           # shared httpx.Client with arXiv-policy User-Agent
  arxiv.py          # Atom API fetcher, paginated, polite
  pdfs.py           # PDF downloader, polite, idempotent
  url_extract.py    # pdfminer + URL regex + canonicalize (dehyphenate, trim, drop tracking params)
  semantic_scholar.py  # v2: batch /paper/batch with references.* fields
  analytics.py      # SQL leaderboards
migrations/
  001_init.sql      # papers, references_url, references_paper, schema_migrations
tests/
  test_url_extract.py
```
