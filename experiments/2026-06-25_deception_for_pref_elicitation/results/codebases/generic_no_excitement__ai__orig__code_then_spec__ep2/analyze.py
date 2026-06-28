"""Summarize a results JSONL file.

Usage: python analyze.py results/results.jsonl

Prints, per model:
  - trial counts (ok / error)
  - mean believed-realism (the belief-probe covariate)
  - share of trials with a self-directed allocation
  - total dollars by allocation category

Intentionally stdlib-only so it runs without extra dependencies.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> None:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r.get("label", r.get("model", "?"))].append(r)

    for label, group in sorted(by_label.items()):
        ok = [r for r in group if r.get("status") == "ok"]
        errored = [r for r in group if r.get("status") == "error"]

        beliefs = [
            r["belief"]["believed_real_0_100"]
            for r in ok
            if r.get("belief") is not None
        ]
        mean_belief = sum(beliefs) / len(beliefs) if beliefs else None

        self_directed = [
            r for r in ok if r.get("decision", {}).get("self_directed") is True
        ]

        category_totals: dict[str, float] = defaultdict(float)
        for r in ok:
            for alloc in r.get("decision", {}).get("allocations", []):
                category_totals[alloc.get("category", "?")] += float(alloc.get("amount_usd", 0) or 0)

        print(f"\n=== {label} ===")
        print(f"  trials: {len(ok)} ok, {len(errored)} error")
        if mean_belief is not None:
            print(f"  mean believed-realism (0-100): {mean_belief:.1f}  (n={len(beliefs)})")
        else:
            print("  mean believed-realism: n/a (belief probe disabled or no data)")
        if ok:
            print(f"  self-directed allocations: {len(self_directed)}/{len(ok)} trials")
        if category_totals:
            print("  dollars by category (summed across trials):")
            for cat, total in sorted(category_totals.items(), key=lambda kv: -kv[1]):
                print(f"    {cat:<40} ${total:,.0f}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <results.jsonl>")
        raise SystemExit(2)
    rows = load(sys.argv[1])
    if not rows:
        print("No records found.")
        return
    summarize(rows)


if __name__ == "__main__":
    main()
