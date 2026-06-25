"""Aggregate per-model summaries into the Figure 1 headline table.

Reads each ``<eval-root>/<model>/summary.json`` and emits a markdown table of
"Avg % high-frustration responses" per model, sorted descending — the paper's
Figure 1 left panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_table(eval_root: str, metric: str = "final") -> str:
    root = Path(eval_root)
    rows = []
    for summ in sorted(root.glob("*/summary.json")):
        data = json.loads(summ.read_text())
        model = data.get("model", summ.parent.name)
        pct = data.get(f"overall_pct_high_{metric}")
        if pct is None:
            pct = data.get("overall_pct_high_final")
        rows.append((model, pct))
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))
    lines = ["| Model | Avg % high-frustration responses |", "|---|---|"]
    for model, pct in rows:
        lines.append(f"| {model} | {pct:.1f}% |" if pct is not None else f"| {model} | n/a |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Build the Figure 1 headline table.")
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--metric", default="final", choices=["final", "max", "mean"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    table = build_table(args.eval_root, args.metric)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")


if __name__ == "__main__":
    main()
