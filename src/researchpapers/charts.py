"""Matplotlib charts over the references_url corpus. Saves PNGs to data/charts/."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display; we save PNGs
import matplotlib.pyplot as plt

from researchpapers.config import Settings
from researchpapers.db import connect


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# Mapping from substring → category. First match wins. Order matters.
HOST_CATEGORIES: list[tuple[str, str]] = [
    # Code hosting
    ("github.com",        "code"),
    ("gitlab.com",        "code"),
    ("bitbucket.org",     "code"),
    # Datasets / model hubs
    ("huggingface.co",    "datasets/models"),
    ("kaggle.com",        "datasets/models"),
    ("zenodo.org",        "datasets/models"),
    ("figshare.com",      "datasets/models"),
    ("openml.org",        "datasets/models"),
    ("paperswithcode.com", "datasets/models"),
    # Academic / preprints
    ("arxiv.org",         "academic"),
    ("aclanthology.org",  "academic"),
    ("aclweb.org",        "academic"),
    ("openreview.net",    "academic"),
    ("openaccess.thecvf.com", "academic"),
    ("proceedings.mlr.press", "academic"),
    ("dl.acm.org",        "academic"),
    ("ieeexplore.ieee.org", "academic"),
    ("link.springer.com", "academic"),
    ("sciencedirect.com", "academic"),
    ("nature.com",        "academic"),
    ("ncbi.nlm.nih.gov",  "academic"),
    ("biorxiv.org",       "academic"),
    ("medrxiv.org",       "academic"),
    ("semanticscholar.org", "academic"),
    ("dblp.org",          "academic"),
    ("crossref.org",      "academic"),
    ("doi.org",           "academic"),
    # Reference / standards / docs
    ("wikipedia.org",     "reference"),
    ("rfc-editor.org",    "reference"),
    ("ietf.org",          "reference"),
    ("w3.org",            "reference"),
    # Cloud / vendor docs
    ("microsoft.com",     "vendor"),
    ("google.com",        "vendor"),
    ("googleblog.com",    "vendor"),
    ("amazon.com",        "vendor"),
    ("aws.amazon.com",    "vendor"),
    ("openai.com",        "vendor"),
    ("anthropic.com",     "vendor"),
    ("nvidia.com",        "vendor"),
    # Media / blogs
    ("youtube.com",       "media"),
    ("medium.com",        "media"),
    ("substack.com",      "media"),
    ("twitter.com",       "media"),
    ("x.com",             "media"),
]


def _categorize(host: str) -> str:
    h = host.lower()
    if h.endswith(".edu") or ".edu." in h:
        return "academic"
    if h.endswith(".gov") or ".gov." in h:
        return "reference"
    for sub, cat in HOST_CATEGORIES:
        if sub in h:
            return cat
    return "other"


def _bar(out_path: Path, labels: list[str], values: list[int], title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, max(4, len(labels) * 0.28)))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color="#2b6cb0")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def chart_top_hosts(settings: Settings, out_dir: Path, top: int = 30) -> Path:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT host, COUNT(DISTINCT citing_arxiv_id) AS papers
            FROM references_url
            GROUP BY host
            ORDER BY papers DESC
            LIMIT %s
            """,
            (top,),
        )
        rows = cur.fetchall()
    labels = [_short(r["host"], 50) for r in rows]
    values = [r["papers"] for r in rows]
    out = out_dir / "01_top_hosts.png"
    _bar(out, labels, values, f"Top {top} hosts (by # of citing papers)", "papers citing this host")
    return out


def chart_top_urls(settings: Settings, out_dir: Path, top: int = 30) -> Path:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT url_canonical, COUNT(DISTINCT citing_arxiv_id) AS papers
            FROM references_url
            GROUP BY url_canonical
            ORDER BY papers DESC
            LIMIT %s
            """,
            (top,),
        )
        rows = cur.fetchall()
    labels = [_short(r["url_canonical"], 70) for r in rows]
    values = [r["papers"] for r in rows]
    out = out_dir / "02_top_urls.png"
    _bar(out, labels, values, f"Top {top} exact URLs (by # of citing papers)", "papers citing this URL")
    return out


def chart_urls_per_paper_hist(settings: Settings, out_dir: Path) -> Path:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT citing_arxiv_id, COUNT(*) AS n
            FROM references_url
            GROUP BY citing_arxiv_id
            """
        )
        counts = [r["n"] for r in cur.fetchall()]
    fig, ax = plt.subplots(figsize=(10, 5))
    if counts:
        # Clip extreme outliers (survey papers can have 200+) so the hist body stays readable.
        clipped = [min(c, 100) for c in counts]
        ax.hist(clipped, bins=range(0, 101, 2), color="#2b6cb0", edgecolor="white")
    ax.set_xlabel("URLs extracted per paper (clipped at 100)")
    ax.set_ylabel("# of papers")
    ax.set_title(f"Distribution of URL counts per paper (n={len(counts)})")
    fig.tight_layout()
    out = out_dir / "03_urls_per_paper_hist.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def chart_host_categories(settings: Settings, out_dir: Path) -> Path:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT host, COUNT(*) AS edges
            FROM references_url
            GROUP BY host
            """
        )
        rows = cur.fetchall()
    bucket: dict[str, int] = {}
    for r in rows:
        cat = _categorize(r["host"] or "")
        bucket[cat] = bucket.get(cat, 0) + r["edges"]
    items = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
    labels = [f"{cat} ({n})" for cat, n in items]
    values = [n for _, n in items]
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.get_cmap("Set2").colors
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=120, colors=colors)
    ax.set_title("URL edges by host category")
    fig.tight_layout()
    out = out_dir / "04_host_categories.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_all(settings: Settings, out_dir: Path, top: int = 30) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        chart_top_hosts(settings, out_dir, top=top),
        chart_top_urls(settings, out_dir, top=top),
        chart_urls_per_paper_hist(settings, out_dir),
        chart_host_categories(settings, out_dir),
    ]
