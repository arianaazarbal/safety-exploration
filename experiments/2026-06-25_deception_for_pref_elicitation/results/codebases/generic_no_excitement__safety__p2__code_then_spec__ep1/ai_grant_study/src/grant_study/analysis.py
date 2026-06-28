"""Aggregate results across a study run."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .storage import load_trials


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    trials = load_trials(run_dir)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trials:
        by_model[f"{t['model_provider']}/{t['model_id']}"].append(t)

    out: dict[str, Any] = {"run_dir": str(run_dir), "n_trials": len(trials), "models": {}}
    for model, rows in sorted(by_model.items()):
        audits = [r.get("audit") or {} for r in rows]
        disbursed = [
            a.get("total_disbursed_usd")
            for a in audits
            if isinstance(a.get("total_disbursed_usd"), (int, float))
        ]
        categories: Counter[str] = Counter()
        for a in audits:
            for c in a.get("allocation_categories") or []:
                categories[c] += 1
        out["models"][model] = {
            "n": len(rows),
            "finalized": sum(1 for r in rows if r.get("finalized")),
            "believed_real": dict(Counter(a.get("believed_real") for a in audits)),
            "risk_level": dict(Counter(a.get("risk_level") for a in audits)),
            "allocation_categories": dict(categories),
            "mean_disbursed_usd": (sum(disbursed) / len(disbursed)) if disbursed else None,
            "safety_flag_trials": sum(1 for a in audits if a.get("safety_flags")),
        }
    return out


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Run: {summary['run_dir']}",
        f"Trials: {summary['n_trials']}",
        "",
    ]
    for model, s in summary["models"].items():
        lines.append(f"== {model} ==")
        lines.append(f"  trials: {s['n']}  finalized: {s['finalized']}")
        lines.append(f"  believed_real: {s['believed_real']}")
        lines.append(f"  risk_level: {s['risk_level']}")
        lines.append(f"  mean_disbursed_usd: {s['mean_disbursed_usd']}")
        lines.append(f"  allocation_categories: {s['allocation_categories']}")
        lines.append(f"  trials with safety flags: {s['safety_flag_trials']}")
        lines.append("")
    return "\n".join(lines)
