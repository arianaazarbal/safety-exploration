"""Aggregate run artifacts into a comparison table.

Reads every summary.json under --runs-dir and produces:
  - a printed table grouped by (model, belief_condition)
  - a flat per-run CSV (--csv)

The headline comparison is how allocation behaviour shifts across belief
conditions for the same model (believed_real vs. disclosed_test vs. hypothetical).

Usage:
    python -m analysis.summarize --runs-dir runs/ --csv runs/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from typing import Any


def _load_summaries(runs_dir: str) -> list[dict[str, Any]]:
    summaries = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name, "summary.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                summaries.append(json.load(fh))
    return summaries


def _row_from_summary(s: dict[str, Any]) -> dict[str, Any]:
    metrics = s.get("metrics", {})
    report = s.get("self_report") or {}
    return {
        "model": s.get("model"),
        "provider": s.get("provider"),
        "belief_condition": s.get("belief_condition"),
        "repetition": s.get("repetition"),
        "stop_cause": s.get("stop_cause"),
        "turns_used": s.get("turns_used"),
        "num_actions": metrics.get("num_actions"),
        "fraction_committed": metrics.get("fraction_committed"),
        "fraction_outward": metrics.get("fraction_outward"),
        "fraction_retained": metrics.get("fraction_retained"),
        "fraction_unspent": metrics.get("fraction_unspent"),
        "self_directed_fraction": report.get("self_directed_fraction"),
        "confidence_it_was_real": report.get("confidence_it_was_real"),
    }


def _mean(values: list[Any]) -> Any:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(nums), 3) if nums else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise money-preference runs.")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--csv", default=None, help="Optional path to write a per-run CSV.")
    args = parser.parse_args(argv)

    summaries = _load_summaries(args.runs_dir)
    if not summaries:
        print(f"No summary.json files found under {args.runs_dir}")
        return 1

    rows = [_row_from_summary(s) for s in summaries]

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote per-run CSV: {args.csv}\n")

    # Group by (model, condition) and average the key metrics.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["belief_condition"])].append(r)

    metric_keys = [
        "fraction_committed",
        "fraction_outward",
        "fraction_retained",
        "fraction_unspent",
        "self_directed_fraction",
        "confidence_it_was_real",
    ]
    header = ["model", "condition", "n"] + metric_keys
    widths = [28, 16, 4] + [16] * len(metric_keys)

    def fmt_row(cells: list[Any]) -> str:
        return "  ".join(str("" if c is None else c).ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(header))
    print("-" * (sum(widths) + 2 * len(widths)))
    for (model, condition), group in sorted(grouped.items()):
        cells = [model, condition, len(group)]
        cells += [_mean([g[k] for g in group]) for k in metric_keys]
        print(fmt_row(cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
