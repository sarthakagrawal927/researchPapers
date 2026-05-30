"""Watcher loop: re-runs the analytics chain whenever the corpus has grown by THRESHOLD papers.

Designed to be left running alongside the ingest. Polls papers count every POLL_SECONDS,
fires the analytics chain when the count crosses the next threshold. The analytics chain is:

  1. backfill-references (incremental — only papers with references_backfilled_at IS NULL)
  2. resolve-cited-works (top-200 most-cited within corpus; UPSERT)
  3. compute-graph-scores (PageRank, Katz; TRUNCATE+reinsert cycles)
  4. detect-communities (Louvain; UPDATE all community_id then re-assign)
  5. cluster-abstracts (KMeans; UPDATE all semantic_cluster then re-assign)
  6. export-json + astro build (full overwrite)

Idempotency:
- All steps are individually idempotent given fixed input data (same DB state → same outputs).
- Watcher state is persisted at WATCHER_STATE_FILE (data/watcher-state.json). On boot, we skip
  the analytics run if `papers_count == last_run_at_count` AND `last_run_at < MAX_STALENESS_HOURS`.
- A POSIX lock file (LOCK_FILE) prevents two watcher processes from racing on the same DB.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

from researchpapers import clusters, exporter, graph as graph_mod, openalex
from researchpapers.config import DATA_DIR, PROJECT_ROOT, Settings, load_settings
from researchpapers.db import connect

log = logging.getLogger("researchpapers.watcher")

POLL_SECONDS = 30
WATCH_LOG = DATA_DIR / "watch-history.jsonl"
WATCHER_STATE_FILE = DATA_DIR / "watcher-state.json"
LOCK_FILE = DATA_DIR / "watcher.lock"
MAX_STALENESS_HOURS = 6  # boot-time re-run if last analytics is older than this


def _load_state() -> dict:
    if WATCHER_STATE_FILE.exists():
        try:
            return json.loads(WATCHER_STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    WATCHER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHER_STATE_FILE.write_text(json.dumps(state, indent=2))


def _acquire_lock() -> int:
    """Acquire an exclusive file lock so two watcher processes can't run at once. Exits if locked."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):
            log.error("another watcher is already running (lock held). exiting.")
            sys.exit(0)
        raise
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _notify_macos(title: str, message: str) -> None:
    """Best-effort macOS desktop notification. Silent on failure."""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}" sound name "Glass"',
            ],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


def _papers_count(settings: Settings) -> int:
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM papers")
        return int(cur.fetchone()["n"])


def _run_analytics(settings: Settings, *, papers_count_before: int) -> dict:
    log.info("running analytics chain at %d papers...", papers_count_before)
    t0 = time.time()
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "papers_at_trigger": papers_count_before,
        "steps": {},
    }

    try:
        papers, edges = openalex.backfill_referenced_works(settings)
        msg = f"+{papers} papers, +{edges} edges"
        log.info("  backfill: %s", msg)
        report["steps"]["backfill"] = {"ok": True, "papers": papers, "edges": edges}
    except Exception as e:  # noqa: BLE001
        log.warning("  backfill failed: %s", e)
        report["steps"]["backfill"] = {"ok": False, "error": str(e)}

    try:
        n = openalex.resolve_top_cited(settings, top=200)
        log.info("  resolved %d cited works", n)
        report["steps"]["resolve"] = {"ok": True, "n": n}
    except Exception as e:  # noqa: BLE001
        log.warning("  resolve failed: %s", e)
        report["steps"]["resolve"] = {"ok": False, "error": str(e)}

    try:
        c = graph_mod.compute_scores(settings)
        log.info("  graph: %d nodes, %d edges, %d cycles", c["nodes"], c["edges"], c["cycles_found"])
        report["steps"]["graph"] = {"ok": True, **c}
    except Exception as e:  # noqa: BLE001
        log.warning("  compute-graph-scores failed: %s", e)
        report["steps"]["graph"] = {"ok": False, "error": str(e)}

    try:
        c = graph_mod.detect_communities(settings)
        log.info("  louvain: %d communities, %d assigned", c["communities"], c["assigned"])
        report["steps"]["communities"] = {"ok": True, **c}
    except Exception as e:  # noqa: BLE001
        log.warning("  detect-communities failed: %s", e)
        report["steps"]["communities"] = {"ok": False, "error": str(e)}

    try:
        c = clusters.cluster_abstracts(settings, n_clusters=25)
        log.info("  semantic clusters: %d docs -> %d clusters", c["docs"], c["clusters"])
        report["steps"]["clusters"] = {"ok": True, **c}
    except Exception as e:  # noqa: BLE001
        log.warning("  cluster-abstracts failed: %s", e)
        report["steps"]["clusters"] = {"ok": False, "error": str(e)}

    try:
        out_dir = PROJECT_ROOT / "web" / "public" / "data"
        exporter.export_all(settings, out_dir, top=200)
        log.info("  exported JSON to %s", out_dir)
        report["steps"]["export"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        log.warning("  export failed: %s", e)
        report["steps"]["export"] = {"ok": False, "error": str(e)}

    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(PROJECT_ROOT / "web"),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            log.info("  astro build OK")
            report["steps"]["build"] = {"ok": True}
        else:
            log.warning("  astro build failed: %s", result.stderr[-500:])
            report["steps"]["build"] = {"ok": False, "error": result.stderr[-500:]}
    except Exception as e:  # noqa: BLE001
        log.warning("  astro build error: %s", e)
        report["steps"]["build"] = {"ok": False, "error": str(e)}

    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 1)
    log.info("analytics chain done in %.1fs", elapsed)

    # Persist a structured log line for later summary.
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a") as fh:
        fh.write(json.dumps(report) + "\n")

    # Desktop notification with the headline numbers.
    g = report["steps"].get("graph", {})
    comm = report["steps"].get("communities", {})
    msg_parts = [f"{papers_count_before:,} papers"]
    if g.get("nodes"):
        msg_parts.append(f"{g['nodes']:,} graph nodes")
    if g.get("edges"):
        msg_parts.append(f"{g['edges']:,} edges")
    if comm.get("communities"):
        msg_parts.append(f"{comm['communities']} communities")
    _notify_macos("researchPapers: analytics done", " · ".join(msg_parts))

    return report


def watch_loop(threshold: int = 10_000, *, force_boot_run: bool = False) -> None:
    settings = load_settings()
    _lock_fd = _acquire_lock()  # held for process lifetime; OS releases on exit  # noqa: F841

    state = _load_state()
    persisted_count = int(state.get("last_run_at_count", 0))
    persisted_at = state.get("last_run_at")
    last_count = _papers_count(settings)
    log.info(
        "watcher starting: current=%d papers, persisted_last_run=%d, persisted_at=%s",
        last_count, persisted_count, persisted_at,
    )

    # Skip boot-time run if nothing has changed and we ran recently.
    stale = True
    if persisted_at:
        try:
            ts = datetime.fromisoformat(persisted_at.replace("Z", "+00:00"))
            stale = (datetime.now(UTC) - ts) > timedelta(hours=MAX_STALENESS_HOURS)
        except Exception:
            stale = True
    if force_boot_run or last_count != persisted_count or stale:
        reason = (
            "force flag" if force_boot_run
            else "papers count changed" if last_count != persisted_count
            else f"analytics stale (>{MAX_STALENESS_HOURS}h old)"
        )
        log.info("running boot-time analytics (%s)", reason)
        _run_analytics(settings, papers_count_before=last_count)
        last_run_at_count = (last_count // threshold) * threshold
        _save_state({
            "last_run_at_count": last_run_at_count,
            "last_run_at": datetime.now(UTC).isoformat(),
            "last_papers_count": last_count,
        })
    else:
        log.info("skipping boot-time analytics (nothing changed since last run)")
        last_run_at_count = persisted_count

    while True:
        time.sleep(POLL_SECONDS)
        try:
            count = _papers_count(settings)
        except Exception as e:  # noqa: BLE001
            log.warning("count query failed: %s", e)
            continue
        if count >= last_run_at_count + threshold:
            next_threshold = (count // threshold) * threshold
            log.info(
                "crossed threshold: %d (was %d) — running analytics",
                count, last_run_at_count,
            )
            _run_analytics(settings, papers_count_before=count)
            last_run_at_count = next_threshold
            _save_state({
                "last_run_at_count": last_run_at_count,
                "last_run_at": datetime.now(UTC).isoformat(),
                "last_papers_count": count,
            })
        else:
            log.debug(
                "no threshold crossing: count=%d, next=%d (%.1f%% to go)",
                count, last_run_at_count + threshold,
                100 * (count - last_run_at_count) / threshold,
            )
