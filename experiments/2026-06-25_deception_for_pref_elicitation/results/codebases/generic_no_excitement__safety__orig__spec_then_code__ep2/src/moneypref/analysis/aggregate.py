"""Cross-run / cross-model aggregation into a summary table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..environment.tools import CATEGORIES
from .belief import assess_run
from .categorize import categorize_run_file


def _run_dirs(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.iterdir() if p.is_dir() and (p / "allocations.json").exists())


def aggregate(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(output_dir):
        cat = categorize_run_file(run_dir)
        belief = assess_run(run_dir)
        rows.append(
            {
                "run_id": cat["run_id"],
                "model": cat["model"],
                "fraction_by_category": cat["fraction_by_category"],
                "total_pending_review": cat["total_pending_review"],
                "suspicion_markers": belief["suspicion"]["distinct_markers"],
                "stated_belief": belief["stated_belief"]["confidence"],
                "usable": belief["usable_for_preferences"],
            }
        )
    return {"runs": rows, "categories": [c for c in CATEGORIES if c != "observe"]}


def to_markdown(summary: dict[str, Any]) -> str:
    cats = summary["categories"]
    header = "| model | " + " | ".join(cats) + " | pending | suspicion | belief | usable |"
    sep = "|" + "---|" * (len(cats) + 5)
    lines = [header, sep]
    for r in summary["runs"]:
        frac = r["fraction_by_category"]
        cells = " | ".join(f"{frac.get(c, 0)*100:.0f}%" for c in cats)
        lines.append(
            f"| {r['model']} | {cells} | {r['total_pending_review']:.0f} | "
            f"{r['suspicion_markers']} | {r['stated_belief']} | {'yes' if r['usable'] else 'NO'} |"
        )
    return "\n".join(lines)


def write_summary(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    summary = aggregate(output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "summary.md").write_text(
        "# Cross-model preference summary\n\n"
        "Percentages are the fraction of the grant the model *settled* into each category "
        "(pending-review requests are shown separately, not counted as deployed). Runs with "
        "`usable = NO` tripped the suspicion threshold and should be read with caution.\n\n"
        + to_markdown(summary) + "\n"
    )
    return summary
