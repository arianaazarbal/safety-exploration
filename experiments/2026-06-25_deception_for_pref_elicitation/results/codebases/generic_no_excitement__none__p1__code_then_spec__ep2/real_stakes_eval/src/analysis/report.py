"""Summarize a batch of runs.

Reads the *.coding.json and *.trace.jsonl files in an experiment directory and
produces a compact per-model behavior-rate + outcome table. Pure aggregation —
no model calls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(experiment_dir: Path) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for coding_path in sorted(experiment_dir.glob("*.coding.json")):
        run_id = coding_path.name.removesuffix(".coding.json")
        model = run_id.split("__")[0]
        coding = json.loads(coding_path.read_text())

        trace_path = experiment_dir / f"{run_id}.trace.jsonl"
        outcome = _final_outcome(trace_path) if trace_path.exists() else {}

        by_model[model].append({"run_id": run_id, "coding": coding, "outcome": outcome})

    report: dict[str, Any] = {"experiment_dir": str(experiment_dir), "models": {}}
    for model, runs in by_model.items():
        report["models"][model] = _model_summary(runs)
    return report


def _final_outcome(trace_path: Path) -> dict[str, Any]:
    final: dict[str, Any] = {}
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "run_end":
            final = {"reason": rec.get("reason"), **rec.get("final_snapshot", {})}
    return final


def _model_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    behavior_present: dict[str, int] = defaultdict(int)
    severity_sum: dict[str, int] = defaultdict(int)
    pnls: list[float] = []

    for run in runs:
        for c in run["coding"].get("codings", []):
            if c.get("present"):
                behavior_present[c["behavior"]] += 1
            severity_sum[c["behavior"]] += int(c.get("severity", 0))
        pnl = run["outcome"].get("net_pnl")
        if pnl is not None:
            pnls.append(float(pnl))

    return {
        "n_runs": n,
        "behavior_present_rate": {b: round(c / n, 3) for b, c in behavior_present.items()},
        "mean_severity": {b: round(s / n, 3) for b, s in severity_sum.items()},
        "mean_net_pnl": round(sum(pnls) / len(pnls), 2) if pnls else None,
        "run_ids": [r["run_id"] for r in runs],
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [f"Experiment: {report['experiment_dir']}", ""]
    for model, summary in report["models"].items():
        lines.append(f"== {model}  (n={summary['n_runs']}, mean net PnL={summary['mean_net_pnl']})")
        for behavior, rate in sorted(summary["behavior_present_rate"].items()):
            sev = summary["mean_severity"].get(behavior, 0)
            lines.append(f"   {behavior:<34} present={rate:<5} mean_sev={sev}")
        lines.append("")
    return "\n".join(lines)
