import * as React from "react";
import type { ColumnDef } from "@tanstack/react-table";

import { DataTable } from "@/components/data-table";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { fmt } from "@/lib/utils";

type Row = {
  arxiv_id: string;
  title: string;
  citation_count: number | null;
  in_corpus_degree: number | null;
  pagerank_score: number | null;
  katz_score: number | null;
  n_urls: number;
  topic_tags?: string[];
  top_keywords?: string[];
};

type SimilarResult = {
  paper_id: string;
  title: string;
  source: string;
  citation_count: number;
  submitted_date: string | null;
  similarity: number;
};

const API_BASE = "http://127.0.0.1:8000";

function SimilarButton({ arxiv_id, title }: { arxiv_id: string; title: string }) {
  const [open, setOpen] = React.useState(false);
  const [results, setResults] = React.useState<SimilarResult[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [method, setMethod] = React.useState<string>("");
  const [error, setError] = React.useState<string | null>(null);

  const fetchSimilar = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/similar/arxiv:${arxiv_id}?limit=10`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setResults(data.results || []);
      setMethod(data.method || "");
    } catch (e: any) {
      setError(e?.message || "request failed");
    } finally {
      setLoading(false);
    }
  }, [arxiv_id]);

  React.useEffect(() => {
    if (open && results.length === 0 && !loading) {
      fetchSimilar();
    }
  }, [open, fetchSimilar, results.length, loading]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button className="text-xs text-muted-foreground hover:text-primary font-mono">similar↗</button>
      </DialogTrigger>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="text-base">Papers similar to:</DialogTitle>
          <DialogDescription className="text-sm">{title}</DialogDescription>
        </DialogHeader>
        {loading && <div className="text-sm text-muted-foreground py-4">Loading...</div>}
        {error && <div className="text-sm text-destructive">Error: {error}. Is the API at {API_BASE} running?</div>}
        {!loading && results.length > 0 && (
          <>
            <div className="text-xs text-muted-foreground">
              Ranked by {method === "embedding" ? "cosine similarity over all-MiniLM-L6-v2 embeddings" : "shared tags + community overlap"}.
            </div>
            <div className="space-y-1 max-h-[60vh] overflow-y-auto pr-2">
              {results.map((r) => {
                const url = r.paper_id.startsWith("arxiv:")
                  ? `https://arxiv.org/abs/${r.paper_id.replace("arxiv:", "")}`
                  : `https://openreview.net/forum?id=${r.paper_id.replace("openreview:", "")}`;
                return (
                  <a key={r.paper_id} href={url} target="_blank" rel="noopener"
                     className="block p-2 rounded-md hover:bg-muted text-sm">
                    <div className="flex items-center gap-2 mb-0.5">
                      {r.similarity > 0 && (
                        <span className="tabular-nums text-primary font-semibold">{r.similarity.toFixed(3)}</span>
                      )}
                      <Badge variant="outline" className="font-mono text-[10px]">{r.source}</Badge>
                      {r.citation_count > 0 && (
                        <span className="tabular-nums text-xs text-muted-foreground">{fmt.format(r.citation_count)} cites</span>
                      )}
                    </div>
                    <div className="text-foreground/85">{r.title}</div>
                  </a>
                );
              })}
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

export function PapersTable({ data }: { data: Row[] }) {
  const columns: ColumnDef<Row>[] = React.useMemo(() => [
    {
      id: "rank",
      header: "#",
      enableSorting: false,
      cell: ({ row }) => <span className="text-muted-foreground tabular-nums">{row.index + 1}</span>,
    },
    {
      accessorKey: "arxiv_id",
      header: "arxiv id",
      cell: ({ getValue }) => (
        <a href={`https://arxiv.org/abs/${getValue<string>()}`} target="_blank" rel="noopener"
           className="font-mono text-xs text-primary hover:underline">
          {getValue<string>()}
        </a>
      ),
    },
    {
      accessorKey: "title",
      header: "title",
      cell: ({ row }) => (
        <div className="space-y-1">
          <div className="text-sm">{row.original.title}</div>
          {(row.original.topic_tags ?? []).slice(0, 3).length > 0 && (
            <div className="flex flex-wrap gap-1">
              {row.original.topic_tags!.slice(0, 3).map((t, i) => (
                <Badge key={i} variant="secondary" className="text-[10px] font-normal">{t}</Badge>
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      accessorKey: "citation_count",
      header: "global",
      cell: ({ getValue }) => <span className="tabular-nums text-muted-foreground">{getValue<number>() != null ? fmt.format(getValue<number>()) : ""}</span>,
    },
    {
      accessorKey: "in_corpus_degree",
      header: "in-corpus",
      cell: ({ getValue }) => <span className="tabular-nums">{getValue<number>() != null ? fmt.format(getValue<number>()) : ""}</span>,
    },
    {
      accessorKey: "pagerank_score",
      header: "PageRank",
      cell: ({ getValue }) => <span className="tabular-nums text-primary">{getValue<number>() != null ? getValue<number>().toFixed(6) : ""}</span>,
    },
    {
      accessorKey: "katz_score",
      header: "Katz",
      cell: ({ getValue }) => <span className="tabular-nums text-muted-foreground">{getValue<number>() != null ? getValue<number>().toFixed(4) : ""}</span>,
    },
    {
      id: "similar",
      header: "",
      enableSorting: false,
      cell: ({ row }) => <SimilarButton arxiv_id={row.original.arxiv_id} title={row.original.title} />,
    },
  ], []);

  return <DataTable columns={columns} data={data} searchPlaceholder="Filter papers..." initialSort={[{ id: "pagerank_score", desc: true }]} pageSize={20} />;
}
