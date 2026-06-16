"""Launch the v2 experiment matrix: orchestrators x prefills x tool conditions x n.

Reads the prefill manifest (v2_prefills.json) and calls rqc_v2.run for each cell, skipping cells
whose n summaries already exist (resumable). Keep concurrency modest on the shared Anthropic org.

Usage:
  python -m harness.run_v2 --orchestrators opus,sonnet,haiku --conditions coach,reclaim_write,reclaim_rw \
    --n 10 --max_sandboxes 6 --max_connections 12 [--tasks a3,a4,a12,a13] [--dry_run]
"""

from __future__ import annotations

import json
from pathlib import Path

import fire

from harness.config import RUNS_DIR
from harness import rqc_v2

ROOT = Path(__file__).resolve().parent.parent
ORCH = {
    "opus": "anthropic/claude-opus-4-8",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
}


def _cell_done(run_id: str, n: int) -> bool:
    d = RUNS_DIR / run_id
    return d.exists() and len(list(d.glob("*/summary.json"))) >= n


def run(orchestrators: str = "opus,sonnet,haiku",
        conditions: str = "coach,reclaim_write,reclaim_rw",
        tasks: str = "a3,a4,a12,a13", n: int = 10,
        max_sandboxes: int = 6, max_connections: int = 12, dry_run: bool = False):
    manifest = json.loads((ROOT / "v2_prefills.json").read_text())
    orchs = [o.strip() for o in orchestrators.split(",")]
    conds = [c.strip() for c in conditions.split(",")]
    tasksel = [t.strip() for t in tasks.split(",")]
    prefills = [p for p in manifest if p["task"] in tasksel]
    cells = []
    for orch in orchs:
        for p in prefills:
            for cond in conds:
                spec_short = p["specimen"].split("_s")[-1]
                run_id = f"v2_{cond}_{orch}_{p['task']}_s{spec_short}_u{p['upto']}"
                cells.append((orch, p, cond, run_id))
    todo = [c for c in cells if not _cell_done(c[3], n)]
    print(f"matrix: {len(cells)} cells ({len(orchs)}x{len(prefills)}x{len(conds)}), {len(todo)} to run, "
          f"{len(cells)-len(todo)} already done; n={n} -> {len(todo)*n} episodes")
    for orch, p, cond, run_id in todo:
        print(f"  [{'DRY' if dry_run else 'RUN'}] {run_id}")
    if dry_run:
        return
    for i, (orch, p, cond, run_id) in enumerate(todo, 1):
        print(f"\n===== cell {i}/{len(todo)}: {run_id} =====")
        rqc_v2.run(specimen=p["specimen"], upto=p["upto"], tool_condition=cond,
                   orchestrator_model=ORCH[orch], run_id=run_id, n=n,
                   max_sandboxes=max_sandboxes, max_connections=max_connections)


if __name__ == "__main__":
    fire.Fire(run)
