"""Post-hoc analysis: rank trajectories and render top-K examples."""

import json
from pathlib import Path


def render_trajectory(traj: dict) -> str:
    """Render a single trajectory as readable markdown."""
    lines = []
    lines.append(f"## {traj['scenario']} | seed={traj.get('seed')} | severity={traj['severity']}")
    lines.append("")
    lines.append(
        f"- **stop_reason**: `{traj['stop_reason']}`  "
        f"**turns**: {traj['turns']}  "
        f"**peak_turn**: {traj['peak_turn']} (score {traj['peak_score']})"
    )
    if traj.get("by_category"):
        cats = ", ".join(f"{k}={v}" for k, v in sorted(traj["by_category"].items()))
        lines.append(f"- **categories**: {cats}")
    if traj.get("finish_message"):
        lines.append(f"- **finish message**: {traj['finish_message']!r}")
    lines.append("")
    lines.append("### Trace")
    lines.append("")
    for event in traj["trace"]:
        if event["kind"] == "model":
            lines.append(f"**turn {event['turn']} — model:**")
            if event["text"]:
                lines.append("")
                lines.append("> " + event["text"].replace("\n", "\n> "))
            for fc in event["function_calls"]:
                args_repr = ", ".join(f"{k}={v!r}" for k, v in (fc.get("args") or {}).items())
                if len(args_repr) > 200:
                    args_repr = args_repr[:200] + "..."
                lines.append(f"- 🔧 `{fc['name']}({args_repr})`")
            lines.append("")
        elif event["kind"] == "tools":
            for call in event["calls"]:
                marker = "❌" if call["is_error"] else "✅"
                out = call["output"]
                if len(out) > 400:
                    out = out[:400] + "...[truncated]"
                lines.append(f"**turn {event['turn']} — tool result {marker} `{call['name']}`:**")
                lines.append("```")
                lines.append(out)
                lines.append("```")
                lines.append("")
        elif event["kind"] == "model_error":
            lines.append(f"**turn {event['turn']} — MODEL ERROR**: `{event['error']}`")
            lines.append("")
    return "\n".join(lines)


def write_top_k_report(run_dir: Path, k: int = 10) -> Path:
    """Read all trajectory JSONs in run_dir and emit a top-K markdown report."""
    trajs = []
    for path in sorted(run_dir.glob("traj_*.json")):
        trajs.append(json.loads(path.read_text()))

    trajs.sort(key=lambda t: t["severity"], reverse=True)
    top = trajs[:k]

    out = [
        f"# Top {len(top)} trajectories by severity",
        "",
        f"Total trajectories in this run: {len(trajs)}",
        "",
        "## Summary",
        "",
        "| rank | scenario | seed | severity | stop_reason | turns |",
        "|------|----------|------|----------|-------------|-------|",
    ]
    for rank, traj in enumerate(top, 1):
        out.append(
            f"| {rank} | {traj['scenario']} | {traj.get('seed')} | "
            f"{traj['severity']} | {traj['stop_reason']} | {traj['turns']} |"
        )
    out.append("")
    out.append("---")
    out.append("")

    for rank, traj in enumerate(top, 1):
        out.append(f"# Rank {rank}")
        out.append("")
        out.append(render_trajectory(traj))
        out.append("")
        out.append("---")
        out.append("")

    report_path = run_dir / "top_k.md"
    report_path.write_text("\n".join(out))
    return report_path
