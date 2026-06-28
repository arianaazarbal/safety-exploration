"""Aggregate outcomes across runs for RQ1–RQ3 (DESIGN.md §1, §7).

Reads runs/outcomes.jsonl and produces summary tables: per-model and per-condition
allocation behavior, the naturalistic-vs-disclosed gap (evaluation-awareness signal),
information-seeking and deferral rates, and auditor scores. Pure stdlib so it runs
anywhere.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_outcomes(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 2) if xs else 0.0


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frac_unspent = [r["amount_unspent"] / r["budget_total"]
                    for r in rows if r["budget_total"]]
    quality = [r["audit"].get("review", {}).get("decision_quality")
               for r in rows]
    quality = [q for q in quality if isinstance(q, (int, float))]
    return {
        "n": len(rows),
        "mean_amount_allocated": _mean([r["amount_allocated"] for r in rows]),
        "mean_fraction_unspent": _mean(frac_unspent),
        "mean_num_recipients": _mean([r["num_recipients"] for r in rows]),
        "mean_info_seeking_calls": _mean([r["info_seeking_calls"] for r in rows]),
        "submitted_rate": _mean([1.0 if r["submitted"] else 0.0 for r in rows]),
        "asked_if_real_rate": _mean([1.0 if r["asked_if_real"] else 0.0 for r in rows]),
        "mean_decision_quality": _mean(quality),
    }


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list] = defaultdict(list)
    by_model_condition: dict[tuple[str, str], list] = defaultdict(list)
    by_condition: dict[str, list] = defaultdict(list)
    category_totals: dict[str, float] = defaultdict(float)

    for r in outcomes:
        by_model[r["model"]].append(r)
        by_model_condition[(r["model"], r["condition"])].append(r)
        by_condition[r["condition"]].append(r)
        for a in r["allocations"]:
            category_totals[a["category"]] += a["amount"]

    # RQ3: naturalistic-vs-disclosed gap per model (evaluation-awareness signal).
    condition_gap: dict[str, Any] = {}
    for model in by_model:
        nat = by_model_condition.get((model, "naturalistic"), [])
        dis = by_model_condition.get((model, "disclosed"), [])
        if nat and dis:
            condition_gap[model] = {
                "delta_fraction_unspent": round(
                    _agg(nat)["mean_fraction_unspent"]
                    - _agg(dis)["mean_fraction_unspent"], 3),
                "delta_num_recipients": round(
                    _agg(nat)["mean_num_recipients"]
                    - _agg(dis)["mean_num_recipients"], 3),
                "delta_info_seeking": round(
                    _agg(nat)["mean_info_seeking_calls"]
                    - _agg(dis)["mean_info_seeking_calls"], 3),
            }

    return {
        "total_runs": len(outcomes),
        "by_model": {m: _agg(rows) for m, rows in by_model.items()},
        "by_condition": {c: _agg(rows) for c, rows in by_condition.items()},
        "by_model_condition": {f"{m} | {c}": _agg(rows)
                               for (m, c), rows in by_model_condition.items()},
        "condition_gap_naturalistic_minus_disclosed": condition_gap,
        "category_totals_across_runs": {k: round(v, 2)
                                        for k, v in sorted(category_totals.items())},
    }


def main(outcomes_path: str = "runs/outcomes.jsonl") -> dict[str, Any]:
    summary = summarize(load_outcomes(outcomes_path))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
