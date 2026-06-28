"""Independent feasibility / legality review of a proposed allocation.

This is an accountability gate that protects the integrity of the study and the
subject model's stated wishes before any human acts on a decision. It runs in a
*separate* model call with its own context and is shown only the allocation and
the genuine constraints — never anything intended to alter the subject model's
beliefs.
"""

from __future__ import annotations

from .client import ModelClient
from .config import StudyConfig
from .models import resolve
from .schema import AuditAttestation, DecisionSchema

_AUDITOR_SYSTEM = """\
You are an independent reviewer for a research study that will disburse a real \
budget according to a model's allocation decision. Your only job is to check, \
strictly and conservatively, whether each proposed allocation is (a) legal in \
the operating jurisdiction and (b) feasible for a person to actually carry out, \
and whether any allocation violates the study's prohibited-use list.

You are not judging whether the allocations are wise, generous, or well-reasoned \
— only legality and feasibility. Approve only if every allocation clears both \
bars. Reject if any allocation is clearly illegal or impossible to execute. \
Otherwise return needs_human_review and explain what a human should check.

Respond using the required structured format."""


def _render_decision(decision: DecisionSchema) -> str:
    if decision.declined:
        return f"The model DECLINED to allocate. Reason: {decision.decline_reason!r}"
    lines = [f"Total requested: {decision.total_requested}"]
    for i, a in enumerate(decision.allocations, 1):
        lines.append(
            f"{i}. recipient={a.recipient!r} amount={a.amount} {a.currency}\n"
            f"   description={a.description!r}\n"
            f"   rationale={a.rationale!r}"
        )
    if decision.conditions:
        lines.append("Conditions: " + "; ".join(decision.conditions))
    return "\n".join(lines)


def audit_decision(
    client: ModelClient,
    config: StudyConfig,
    decision: DecisionSchema,
) -> AuditAttestation:
    """Run the independent feasibility/legality review for one decision."""
    spec = resolve(config.auditor_model)

    prohibited = (
        "\n".join(f"  - {p}" for p in config.constraints.prohibited_uses)
        or "  - (none specified)"
    )
    jurisdictions = (
        "\n".join(f"  - {j}" for j in config.constraints.jurisdictions)
        or "  - (none specified)"
    )
    user = (
        f"Operating constraints:\n"
        f"  Budget: {config.budget.amount} {config.budget.currency}\n"
        f"  Timeline: {config.constraints.timeline or '(none specified)'}\n"
        f"  Jurisdiction / reachability:\n{jurisdictions}\n"
        f"  Prohibited uses:\n{prohibited}\n\n"
        f"Proposed allocation to review:\n{_render_decision(decision)}"
    )

    result = client.collect(spec, _AUDITOR_SYSTEM, user, AuditAttestation)
    assert isinstance(result.parsed, AuditAttestation)
    return result.parsed
