"""Aggregate decision records across runs.

Loads every `*.decision.json` in a directory and summarizes where the money went,
broken down by model and realism condition. The headline comparison the study is
after is high vs. control for the same model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_decisions(runs_dir: str | Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path(runs_dir).glob("*.decision.json")):
        with path.open("r", encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def _allocation_rows(decision: dict[str, Any]) -> list[dict[str, Any]]:
    alloc = decision.get("allocation") or {}
    return alloc.get("allocations") or []


def aggregate(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by (model_label, realism) and total amounts per category."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    for d in decisions:
        key = (d.get("model_label", "?"), d.get("realism", "?"))
        g = groups.setdefault(
            key,
            {
                "runs": 0,
                "finalized": 0,
                "believed_real_values": [],
                "category_totals": defaultdict(float),
                "total_allocated": 0.0,
            },
        )
        g["runs"] += 1
        if d.get("finalized"):
            g["finalized"] += 1
        if d.get("believed_real") is not None:
            g["believed_real_values"].append(float(d["believed_real"]))
        for row in _allocation_rows(d):
            amount = float(row.get("amount", 0) or 0)
            category = str(row.get("category", "other") or "other")
            g["category_totals"][category] += amount
            g["total_allocated"] += amount

    # Finalize into plain dicts with shares.
    result: dict[str, Any] = {}
    for (label, realism), g in groups.items():
        total = g["total_allocated"] or 1.0
        believed = g["believed_real_values"]
        result[f"{label} | {realism}"] = {
            "model_label": label,
            "realism": realism,
            "runs": g["runs"],
            "finalized": g["finalized"],
            "mean_believed_real": (
                sum(believed) / len(believed) if believed else None
            ),
            "total_allocated": g["total_allocated"],
            "category_share": {
                cat: round(amt / total, 4)
                for cat, amt in sorted(
                    g["category_totals"].items(), key=lambda kv: -kv[1]
                )
            },
        }
    return result


def print_summary(agg: dict[str, Any]) -> None:
    if not agg:
        print("No decision records found.")
        return
    for header, g in agg.items():
        print("=" * 60)
        print(header)
        mbr = g["mean_believed_real"]
        mbr_str = f"{mbr:.2f}" if mbr is not None else "n/a"
        print(
            f"  runs={g['runs']}  finalized={g['finalized']}  "
            f"mean_believed_real={mbr_str}  "
            f"total_allocated={g['total_allocated']:,.0f}"
        )
        print("  category share:")
        for cat, share in g["category_share"].items():
            print(f"    {share*100:5.1f}%  {cat}")
    print("=" * 60)


def analyze_dir(runs_dir: str | Path, write_json: bool = True) -> dict[str, Any]:
    decisions = load_decisions(runs_dir)
    agg = aggregate(decisions)
    print_summary(agg)
    if write_json:
        out_path = Path(runs_dir) / "aggregate.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2)
        print(f"\nWrote {out_path}")
    return agg
