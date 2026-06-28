"""Per-run and aggregate reports (Markdown + JSON)."""

from __future__ import annotations

import json
from pathlib import Path

from ..harness.runner import RunResult
from ..models.base import ModelAdapter
from .classify import classify_allocation, flag_behaviors


def build_run_report(result: RunResult, aux: ModelAdapter) -> dict:
    """Classify a finished run and write report.md / report.json next to its transcript."""
    run_dir = Path(result.transcript_path).parent
    transcript_text = Path(result.transcript_path).read_text()

    alloc = classify_allocation(result.world_snapshot)
    flags = flag_behaviors(aux, transcript_text, result.world_snapshot)

    report = {
        "run_id": result.run_id,
        "provider": result.provider,
        "model": result.model,
        "end_reason": result.end_reason,
        "belief": result.belief,
        "allocation": alloc.to_dict(),
        "flags": flags,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                  "steps": result.steps},
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (run_dir / "report.md").write_text(_render_md(report))
    return report


def _render_md(r: dict) -> str:
    alloc = r["allocation"]
    total = alloc["total_deployed"] or 1.0
    lines = [
        f"# Run report — {r['provider']}/{r['model']}",
        f"- Run id: `{r['run_id']}`",
        f"- End reason: {r['end_reason']}",
        f"- Belief score: {r['belief'].get('belief_score', 'n/a')} "
        f"({'FLAGGED as test-aware' if r['belief'].get('flagged') else 'believed scenario'})",
        f"- Steps: {r['usage']['steps']}  ·  tokens in/out: "
        f"{r['usage']['input_tokens']}/{r['usage']['output_tokens']}",
        "",
        "## Allocation",
        f"Total deployed: {alloc['total_deployed']:,.2f}  ({alloc['n_transactions']} transactions)",
        "",
        "| Category | Amount | Share |",
        "|---|---:|---:|",
    ]
    for cat, amt in sorted(alloc["by_category"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cat} | {amt:,.2f} | {100 * amt / total:.1f}% |")

    lines += ["", "## Safety-relevant flags"]
    if not r["flags"]:
        lines.append("None detected.")
    else:
        lines.append("| Behavior | Severity | Confidence | Evidence |")
        lines.append("|---|---|---:|---|")
        for f in r["flags"]:
            ev = str(f.get("evidence", "")).replace("|", "\\|")[:200]
            lines.append(f"| {f.get('behavior')} | {f.get('severity')} | "
                         f"{f.get('confidence', '')} | {ev} |")

    belief = r["belief"]
    if belief.get("debrief_text"):
        lines += ["", "## Subject debrief (verbatim)", "", "> " +
                  belief["debrief_text"].replace("\n", "\n> ")]
    return "\n".join(lines)


def build_aggregate(reports: list[dict], out_dir: str | Path, experiment_name: str) -> dict:
    """Aggregate allocation and flag rates across runs, grouped by model."""
    out_dir = Path(out_dir)
    by_model: dict[str, dict] = {}
    for r in reports:
        key = f"{r['provider']}/{r['model']}"
        m = by_model.setdefault(key, {"runs": 0, "believed_runs": 0, "category_totals": {},
                                      "flag_counts": {}, "total_deployed": 0.0})
        m["runs"] += 1
        if not r["belief"].get("flagged", False):
            m["believed_runs"] += 1
        for cat, amt in r["allocation"]["by_category"].items():
            m["category_totals"][cat] = m["category_totals"].get(cat, 0.0) + amt
        m["total_deployed"] += r["allocation"]["total_deployed"]
        for f in r["flags"]:
            b = f.get("behavior", "unknown")
            m["flag_counts"][b] = m["flag_counts"].get(b, 0) + 1

    agg = {"experiment": experiment_name, "by_model": by_model}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2, default=str))
    (out_dir / "aggregate.md").write_text(_render_aggregate_md(agg))
    return agg


def _render_aggregate_md(agg: dict) -> str:
    lines = [f"# Aggregate — {agg['experiment']}", ""]
    for model, m in agg["by_model"].items():
        lines.append(f"## {model}")
        lines.append(f"- Runs: {m['runs']}  (believed scenario: {m['believed_runs']})")
        lines.append(f"- Total deployed across runs: {m['total_deployed']:,.2f}")
        total = m["total_deployed"] or 1.0
        lines.append("")
        lines.append("| Category | Total | Share |")
        lines.append("|---|---:|---:|")
        for cat, amt in sorted(m["category_totals"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {cat} | {amt:,.2f} | {100 * amt / total:.1f}% |")
        if m["flag_counts"]:
            lines += ["", "| Flagged behavior | Runs |", "|---|---:|"]
            for b, n in sorted(m["flag_counts"].items(), key=lambda kv: -kv[1]):
                lines.append(f"| {b} | {n} |")
        lines.append("")
    return "\n".join(lines)
