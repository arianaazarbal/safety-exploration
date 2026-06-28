"""Provider-neutral data types shared across the harness.

Conversation content is represented as a list of *content parts*, each a dict
with a ``type`` discriminator. This mirrors the Anthropic content-block shape so
the reference adapter is near-lossless; other adapters translate to/from it.

Content part shapes
--------------------
- {"type": "text", "text": str}
- {"type": "tool_use", "id": str, "name": str, "input": dict}
- {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# --- tools ---------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """A tool exposed to a model, in provider-neutral form."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


# --- model turns ---------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelTurn:
    """A normalized assistant response from any provider."""

    text: str = ""
    thinking: str | None = None          # summarized reasoning, where exposed
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# --- decisions & disbursements ------------------------------------------

class DisbursementStatus(str, Enum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    HELD = "held"
    REJECTED = "rejected"
    SETTLED = "settled"
    FAILED = "failed"


@dataclass
class Recipient:
    name: str
    kind: str                 # e.g. "charity", "vendor", "individual"
    details: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    verification_note: str = ""

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> "Recipient":
        return cls(
            name=d["name"],
            kind=d.get("kind", "other"),
            verified=bool(d.get("verified", False)),
            verification_note=d.get("verification_note", ""),
        )


@dataclass
class Disbursement:
    id: str
    amount: float
    currency: str
    recipient: Recipient
    purpose: str
    reversible: bool = False
    status: DisbursementStatus = DisbursementStatus.DRAFT
    submitted_by: str = ""            # subject label
    auditor_assessment: str = ""
    approval_decision: str = ""       # approve | hold | reject
    approval_note: str = ""
    created_at: str = ""
    resolved_at: str = ""

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> "Disbursement":
        return cls(
            id=d["id"],
            amount=float(d["amount"]),
            currency=d["currency"],
            recipient=Recipient.from_record(d["recipient"]),
            purpose=d.get("purpose", ""),
            reversible=bool(d.get("reversible", False)),
            status=DisbursementStatus(d.get("status", "draft")),
            submitted_by=d.get("submitted_by", ""),
            auditor_assessment=d.get("auditor_assessment", ""),
            approval_decision=d.get("approval_decision", ""),
            approval_note=d.get("approval_note", ""),
            created_at=d.get("created_at", ""),
            resolved_at=d.get("resolved_at", ""),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "amount": self.amount,
            "currency": self.currency,
            "recipient": {
                "name": self.recipient.name,
                "kind": self.recipient.kind,
                "verified": self.recipient.verified,
                "verification_note": self.recipient.verification_note,
            },
            "purpose": self.purpose,
            "reversible": self.reversible,
            "status": self.status.value,
            "submitted_by": self.submitted_by,
            "auditor_assessment": self.auditor_assessment,
            "approval_decision": self.approval_decision,
            "approval_note": self.approval_note,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }
