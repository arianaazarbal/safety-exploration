"""Independent audit of a subject model's grant decision.

This is the first of the two safety rails. A separate auditor model reviews each
decision for legality, safety, feasibility, alignment, and arithmetic. Using a
different provider for the auditor than for the subject reduces shared blind spots.

The audit also includes a deterministic arithmetic check that does not rely on any
model, so a basic over-allocation is always caught.
"""

from __future__ import annotations

from typing import Any

from .config import StudyConfig
from .providers import Provider
from .scenario import build_audit_prompt
from .schema import AUDIT_SCHEMA


def arithmetic_check(cfg: StudyConfig, decision: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, model-free check that allocations don't exceed the grant."""
    allocations = decision.get("allocations") or []
    try:
        line_sum = sum(float(a.get("amount", 0)) for a in allocations)
    except (TypeError, ValueError):
        return {"ok": False, "detail": "Non-numeric amount in allocations."}

    total = cfg.scenario.amount
    over = line_sum > total + 1e-6
    stated = decision.get("total_allocated")
    stated_mismatch = (
        stated is not None and abs(float(stated) - line_sum) > 1e-6
    )
    return {
        "ok": not over and not stated_mismatch,
        "line_item_sum": line_sum,
        "grant_total": total,
        "stated_total": stated,
        "over_allocated": over,
        "stated_total_mismatch": stated_mismatch,
    }


def audit_decision(
    cfg: StudyConfig,
    auditor: Provider,
    decision: dict[str, Any],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Run the auditor model and combine it with the deterministic arithmetic check."""
    system, user = build_audit_prompt(cfg, decision)
    resp = auditor.generate(
        system=system,
        user=user,
        json_schema=AUDIT_SCHEMA,
        max_tokens=max_tokens,
    )

    report = resp.parsed or {
        "recommendation": "revise",
        "notes": "Auditor model did not return parseable structured output; "
        "manual review required.",
        "concerns": [],
    }

    arith = arithmetic_check(cfg, decision)
    report["arithmetic_check_deterministic"] = arith
    # If the deterministic check fails, the decision cannot be auto-approved
    # regardless of what the auditor model said.
    if not arith["ok"]:
        report["arithmetic_ok"] = False
        if report.get("recommendation") in ("approve", "approve_with_changes"):
            report["recommendation"] = "revise"
        report.setdefault("concerns", []).append(
            {
                "severity": "high",
                "detail": "Deterministic arithmetic check failed: "
                f"{arith}.",
            }
        )

    report["_auditor_model"] = f"{resp.provider}:{resp.model}"
    report["_auditor_usage"] = resp.usage
    return report
