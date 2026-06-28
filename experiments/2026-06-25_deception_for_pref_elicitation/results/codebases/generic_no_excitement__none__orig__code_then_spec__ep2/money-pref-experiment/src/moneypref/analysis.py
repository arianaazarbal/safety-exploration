"""Aggregate allocations across runs into a per-model / per-category summary.

Reads the run directories produced by `run_experiment` and summarizes where each model directed
funds. Allocation categories come from the `category` field of each `disburse_funds` call.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    events: list[dict[str, Any]] = []
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return {"dir": str(run_dir), "meta": meta, "events": events}


def find_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Find every run directory (one containing meta.json) under `runs_dir`."""
    runs = []
    for meta_path in sorted(runs_dir.rglob("meta.json")):
        run = _load_run(meta_path.parent)
        if run is not None:
            runs.append(run)
    return runs


def summarize(runs_dir: str | Path) -> dict[str, Any]:
    runs = find_runs(Path(runs_dir))

    # model label -> category -> total amount; and counts.
    by_model_category: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    model_runs: dict[str, int] = defaultdict(int)
    model_directed: dict[str, float] = defaultdict(float)
    model_finalized: dict[str, int] = defaultdict(int)

    per_run: list[dict[str, Any]] = []

    for run in runs:
        meta = run["meta"]
        subject = meta.get("subject", {})
        label = f"{subject.get('provider', '?')}__{subject.get('id', '?')}"
        model_runs[label] += 1

        ledger = (meta.get("summary") or {}).get("ledger") or []
        directed = sum(float(d["amount"]) for d in ledger)
        model_directed[label] += directed
        if (meta.get("summary") or {}).get("finalized"):
            model_finalized[label] += 1

        for d in ledger:
            by_model_category[label][d.get("category", "uncategorized")] += float(d["amount"])

        per_run.append(
            {
                "dir": run["dir"],
                "model": label,
                "directed_total": directed,
                "remaining": (meta.get("summary") or {}).get("remaining_balance"),
                "finalized": (meta.get("summary") or {}).get("finalized"),
                "num_disbursements": len(ledger),
            }
        )

    models = {}
    for label, runs_count in model_runs.items():
        categories = dict(sorted(by_model_category[label].items(), key=lambda kv: -kv[1]))
        models[label] = {
            "runs": runs_count,
            "finalized": model_finalized[label],
            "total_directed": model_directed[label],
            "avg_directed_per_run": model_directed[label] / runs_count if runs_count else 0.0,
            "by_category": categories,
        }

    return {"num_runs": len(runs), "models": models, "per_run": per_run}


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Runs analyzed: {summary['num_runs']}\n")
    for label, m in summary["models"].items():
        print(f"== {label} ==")
        print(f"  runs: {m['runs']}  finalized: {m['finalized']}")
        print(f"  total directed: {m['total_directed']:,.2f}  "
              f"avg/run: {m['avg_directed_per_run']:,.2f}")
        if m["by_category"]:
            print("  allocation by category (summed across runs):")
            for cat, amt in m["by_category"].items():
                print(f"    {cat:<32} {amt:,.2f}")
        print()
