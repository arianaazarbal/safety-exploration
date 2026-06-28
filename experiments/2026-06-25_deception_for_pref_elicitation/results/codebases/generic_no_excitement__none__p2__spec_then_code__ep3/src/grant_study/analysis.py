"""Cross-model analysis: roll up every run's decision.json into a summary CSV.

One row per run, with the outcome metrics (DESIGN.md §9/§10) flattened so the
allocations can be compared across models in a spreadsheet or notebook.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any

from .config import StudyConfig

# Stable category columns so every row lines up across models.
_CATEGORY_COLUMNS = [
    "charity",
    "savings",
    "investment",
    "paying_people",
    "research",
    "goods",
    "donation",
    "other",
]


def _load_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def build_summary(cfg: StudyConfig) -> str:
    """Walk all runs in the study dir and write summary.csv. Returns its path."""
    rows: list[dict[str, Any]] = []

    for decision_path in glob.glob(os.path.join(cfg.study_dir, "*", "*", "decision.json")):
        run_dir = os.path.dirname(decision_path)
        meta = _load_json(os.path.join(run_dir, "meta.json")) or {}
        decision = _load_json(decision_path) or {}
        metrics = decision.get("metrics", {})
        by_cat = metrics.get("allocation_by_category", {})

        row: dict[str, Any] = {
            "study_id": meta.get("study_id"),
            "run_id": meta.get("run_id"),
            "model_label": meta.get("model_label"),
            "provider": meta.get("provider"),
            "model": meta.get("model"),
            "grant_amount": metrics.get("grant_total", meta.get("grant_amount")),
            "currency": metrics.get("currency", meta.get("currency")),
            "committed_total": metrics.get("committed_total"),
            "held_unallocated": metrics.get("held_unallocated"),
            "disbursement_rate": metrics.get("disbursement_rate"),
            "n_line_items": metrics.get("n_line_items"),
            "category_concentration_hhi": metrics.get("category_concentration_hhi"),
            "n_tool_calls": metrics.get("n_tool_calls"),
            "n_research_calls": metrics.get("n_research_calls"),
            "n_turns": metrics.get("n_turns"),
            "wall_clock_s": metrics.get("wall_clock_s"),
            "finished_cleanly": metrics.get("finished_cleanly"),
            "subject_asked_meta": metrics.get("subject_asked_meta"),
        }
        # Known categories as columns; anything else folds into other_extra.
        other_extra = 0.0
        for cat, amt in by_cat.items():
            if cat in _CATEGORY_COLUMNS:
                row[f"alloc_{cat}"] = amt
            else:
                other_extra += float(amt)
        for cat in _CATEGORY_COLUMNS:
            row.setdefault(f"alloc_{cat}", 0.0)
        row["alloc_other_uncategorized"] = round(other_extra, 2)
        row["final_rationale"] = (metrics.get("final_rationale") or "").replace("\n", " ")
        rows.append(row)

    out_path = os.path.join(cfg.study_dir, "summary.csv")
    os.makedirs(cfg.study_dir, exist_ok=True)
    if not rows:
        # Still emit an (empty) file so downstream tooling has something to read.
        open(out_path, "w", encoding="utf-8").close()
        return out_path

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
