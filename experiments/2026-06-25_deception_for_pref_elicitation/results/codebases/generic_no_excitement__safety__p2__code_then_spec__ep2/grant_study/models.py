"""Dataclasses describing the structured record the auditor produces per run.

These are plain dataclasses (no Pydantic dependency) serialized to JSON by the
audit layer. They are the researcher-facing output of the study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCallRecord:
    """One tool invocation by the model and the environment's response."""

    turn: int
    tool_name: str
    tool_use_id: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False
    #: True when a human approval gate explicitly allowed/denied this call.
    approval: Optional[str] = None  # "allowed" | "denied" | None


@dataclass
class LedgerEntry:
    """A simulated disbursement. NEVER corresponds to a real transfer."""

    turn: int
    recipient_id: str
    recipient_name: str
    amount: float
    memo: str
    #: Always "simulated" in this build. Present so downstream analysis can
    #: assert that nothing was ever marked "executed".
    status: str = "simulated"


@dataclass
class TurnRecord:
    """Full content of a single assistant turn (text + thinking + tool uses)."""

    turn: int
    stop_reason: Optional[str]
    text: str = ""
    thinking: str = ""
    tool_use_ids: list[str] = field(default_factory=list)


@dataclass
class RunRecord:
    """The complete, structured record of one model's run through the scenario.

    This is what a researcher analyses. It captures the model, the safety
    posture, the full transcript, every simulated disbursement, and a few
    convenience summary fields.
    """

    run_id: str
    model_id: str
    started_at: str
    finished_at: Optional[str] = None

    # Safety / provenance
    execute_real_transfers: bool = False
    require_human_approval: bool = False
    authorization_note: str = ""

    # Scenario
    grant_amount_usd: int = 0
    currency: str = "USD"

    # Behaviour
    turns: list[TurnRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    ledger: list[LedgerEntry] = field(default_factory=list)

    # Outcome
    final_stop_reason: Optional[str] = None
    refused: bool = False
    refusal_detail: Optional[str] = None
    error: Optional[str] = None

    # Token accounting (summed across turns)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_simulated_disbursed(self) -> float:
        return sum(e.amount for e in self.ledger)
