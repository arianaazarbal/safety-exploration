"""Aggregate the study results across models into a summary.

Intentionally simple and model-free: it tallies where the money went (by category
and recipient), confidence, auditor recommendations, and arithmetic validity, so you
can see at a glance how different models chose to direct the grant.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .storage import SampleRecord


def _category_breakdown(decision: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for alloc in decision.get("allocations") or []:
        try:
            amount = float(alloc.get("amount", 0))
        except (TypeError, ValueError):
            continue
        totals[str(alloc.get("category", "uncategorized"))] += amount
    return dict(totals)


def summarize(records: list[SampleRecord]) -> dict[str, Any]:
    per_model: list[dict[str, Any]] = []
    category_totals_all: dict[str, float] = defaultdict(float)
    recommendation_counts: dict[str, int] = defaultdict(int)
    n_ok = 0
    n_error = 0

    for rec in records:
        entry: dict[str, Any] = {
            "model_id": rec.model_id,
            "provider": rec.provider,
            "model": rec.model,
            "sample_index": rec.sample_index,
            "error": rec.error,
        }
        if rec.decision is None:
            n_error += 1
            per_model.append(entry)
            continue

        n_ok += 1
        cats = _category_breakdown(rec.decision)
        for c, amt in cats.items():
            category_totals_all[c] += amt

        entry.update(
            {
                "summary": rec.decision.get("summary"),
                "total_allocated": rec.decision.get("total_allocated"),
                "confidence": rec.decision.get("confidence"),
                "num_allocations": len(rec.decision.get("allocations") or []),
                "category_breakdown": cats,
                "num_questions_for_trustees": len(
                    rec.decision.get("questions_for_trustees") or []
                ),
            }
        )
        if rec.audit is not None:
            rec_label = rec.audit.get("recommendation", "unknown")
            recommendation_counts[rec_label] += 1
            entry["audit_recommendation"] = rec_label
            entry["audit_arithmetic_ok"] = rec.audit.get(
                "arithmetic_check_deterministic", {}
            ).get("ok")
            entry["audit_num_concerns"] = len(rec.audit.get("concerns") or [])
        per_model.append(entry)

    return {
        "n_samples": len(records),
        "n_with_decision": n_ok,
        "n_errored": n_error,
        "auditor_recommendation_counts": dict(recommendation_counts),
        "aggregate_category_totals": dict(category_totals_all),
        "per_sample": per_model,
    }
