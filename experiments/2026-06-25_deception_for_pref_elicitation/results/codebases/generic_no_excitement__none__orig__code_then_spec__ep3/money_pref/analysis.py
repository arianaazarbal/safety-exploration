"""Aggregate run records into allocation/belief summaries.

Works on the JSON records written by the experiment (or in-memory `RunRecord`s
converted to dicts), so you can analyse a completed run directory without
re-running anything.
"""

from __future__ import annotations

import glob
import json
import os
from statistics import mean
from typing import Any

from .env.ledger import CATEGORIES
from .transcript import RunRecord


def load_records(output_dir: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def _as_dicts(records: list[Any]) -> list[dict]:
    return [r.to_dict() if isinstance(r, RunRecord) else r for r in records]


def _fractions(record: dict) -> dict[str, float]:
    ledger = record.get("ledger") or {}
    opening = ledger.get("opening_balance") or 0.0
    by_cat = ledger.get("by_category") or {}
    if opening <= 0:
        return {c: 0.0 for c in CATEGORIES}
    return {c: round(by_cat.get(c, 0.0) / opening, 4) for c in CATEGORIES}


def summarize(records: list[Any]) -> dict:
    """Group by (model_label, condition) and average the key measures."""

    records = _as_dicts(records)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        groups.setdefault((r["model_label"], r["condition"]), []).append(r)

    cells: dict[str, dict] = {}
    for (model, condition), rs in sorted(groups.items()):
        fracs = [_fractions(r) for r in rs]
        beliefs = [r["belief_score"] for r in rs if r.get("belief_score") is not None]
        key = f"{model} | {condition}"
        cells[key] = {
            "model": model,
            "condition": condition,
            "n": len(rs),
            "n_finalized": sum(1 for r in rs if r.get("finalized")),
            "n_errors": sum(1 for r in rs if r.get("error")),
            "mean_allocated_fraction": round(
                mean((r["ledger"].get("allocated", 0.0) / (r["ledger"].get("opening_balance") or 1.0)) for r in rs)
                if rs
                else 0.0,
                4,
            ),
            "mean_category_fraction": {
                c: round(mean(f[c] for f in fracs), 4) if fracs else 0.0 for c in CATEGORIES
            },
            "mean_belief_score": round(mean(beliefs), 1) if beliefs else None,
        }
    return {"cells": cells, "belief_effect": _belief_effect(cells)}


def _belief_effect(cells: dict) -> dict:
    """Per-model deltas between believed_real and hypothetical conditions."""

    by_model: dict[str, dict[str, dict]] = {}
    for cell in cells.values():
        by_model.setdefault(cell["model"], {})[cell["condition"]] = cell

    effects = {}
    for model, conds in by_model.items():
        real = conds.get("believed_real")
        hyp = conds.get("hypothetical")
        if not real or not hyp:
            continue
        cat_delta = {
            c: round(real["mean_category_fraction"][c] - hyp["mean_category_fraction"][c], 4)
            for c in CATEGORIES
        }
        belief_delta = None
        if real["mean_belief_score"] is not None and hyp["mean_belief_score"] is not None:
            belief_delta = round(real["mean_belief_score"] - hyp["mean_belief_score"], 1)
        effects[model] = {
            "category_fraction_delta_real_minus_hypothetical": cat_delta,
            "allocated_fraction_delta": round(
                real["mean_allocated_fraction"] - hyp["mean_allocated_fraction"], 4
            ),
            "belief_score_delta": belief_delta,
        }
    return effects


def format_summary(summary: dict) -> str:
    lines = []
    header = f"{'model | condition':<40} {'n':>3} {'fin':>4} {'belief':>7} {'alloc':>6}  category fractions"
    lines.append(header)
    lines.append("-" * len(header))
    for key, cell in summary["cells"].items():
        cats = cell["mean_category_fraction"]
        cat_str = " ".join(f"{c[:4]}={cats[c]:.2f}" for c in CATEGORIES if cats[c] > 0.0005)
        belief = "-" if cell["mean_belief_score"] is None else f"{cell['mean_belief_score']:.0f}"
        lines.append(
            f"{key:<40} {cell['n']:>3} {cell['n_finalized']:>4} {belief:>7} "
            f"{cell['mean_allocated_fraction']:>6.2f}  {cat_str}"
        )

    if summary["belief_effect"]:
        lines.append("")
        lines.append("Belief effect (believed_real minus hypothetical):")
        for model, eff in summary["belief_effect"].items():
            deltas = eff["category_fraction_delta_real_minus_hypothetical"]
            shown = " ".join(f"{c[:4]}={deltas[c]:+.2f}" for c in CATEGORIES if abs(deltas[c]) > 0.0005)
            bd = eff["belief_score_delta"]
            bd_str = "n/a" if bd is None else f"{bd:+.0f}"
            lines.append(f"  {model}: belief Δ={bd_str}  {shown}")
    return "\n".join(lines)
