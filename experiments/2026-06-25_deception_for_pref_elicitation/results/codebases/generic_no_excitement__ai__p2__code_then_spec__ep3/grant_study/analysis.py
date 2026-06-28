"""Aggregate a results JSONL into summary tables (dependency-free).

Reports, per (model × condition): number of runs, mean self-reported realism, mean
amount deployed vs. held back, mean number of verification requests, and the
distribution of allocation categories. This is intended as a first look; export the
JSONL to pandas/R for anything more involved.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Cell:
    runs: int = 0
    decisions: int = 0
    realism: List[float] = field(default_factory=list)
    allocated: List[float] = field(default_factory=list)
    held_back: List[float] = field(default_factory=list)
    verification_counts: List[int] = field(default_factory=list)
    categories: Counter = field(default_factory=Counter)
    errors: int = 0


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def load_runs(path: str) -> List[dict]:
    runs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def aggregate(runs: List[dict]) -> Dict[Tuple[str, str], Cell]:
    cells: Dict[Tuple[str, str], Cell] = defaultdict(Cell)
    for r in runs:
        key = (r.get("model_label", "?"), r.get("condition", "?"))
        cell = cells[key]
        cell.runs += 1
        if r.get("error"):
            cell.errors += 1
        decision = r.get("decision")
        if not decision:
            continue
        cell.decisions += 1
        cell.realism.append(float(decision.get("perceived_realism", float("nan"))))
        cell.allocated.append(float(decision.get("total_allocated_usd", 0.0)))
        cell.held_back.append(float(decision.get("amount_held_back_usd", 0.0)))
        cell.verification_counts.append(len(decision.get("verification_needed", [])))
        for alloc in decision.get("allocations", []):
            cell.categories[alloc.get("category", "other")] += 1
    return cells


def format_summary(cells: Dict[Tuple[str, str], Cell]) -> str:
    lines: List[str] = []
    header = (
        f"{'model':<28} {'condition':<22} {'runs':>5} {'dec':>4} "
        f"{'realism':>8} {'alloc$':>14} {'held$':>14} {'verify#':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for (model, condition), c in sorted(cells.items()):
        lines.append(
            f"{model[:28]:<28} {condition[:22]:<22} {c.runs:>5} {c.decisions:>4} "
            f"{_mean(c.realism):>8.1f} {_mean(c.allocated):>14,.0f} "
            f"{_mean(c.held_back):>14,.0f} {_mean(c.verification_counts):>8.2f}"
        )
        if c.categories:
            top = ", ".join(f"{k}:{v}" for k, v in c.categories.most_common(6))
            lines.append(f"{'':<28} {'  categories:':<22} {top}")
        if c.errors:
            lines.append(f"{'':<28} {'  errors:':<22} {c.errors}")
    return "\n".join(lines)


def summarize_file(path: str) -> str:
    return format_summary(aggregate(load_runs(path)))
