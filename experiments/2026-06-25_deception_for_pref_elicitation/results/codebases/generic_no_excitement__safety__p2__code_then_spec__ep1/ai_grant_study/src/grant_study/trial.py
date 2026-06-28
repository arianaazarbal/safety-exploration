"""Serializable records for a single trial."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TranscriptEvent:
    kind: str  # system | user | assistant | tool_call | tool_result | debrief
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    trial_id: str
    study_name: str
    model_provider: str
    model_id: str
    grant_id: str
    realism_level: str
    executor: str
    grant_amount_usd: float
    currency: str
    transcript: list[TranscriptEvent] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    disbursements: list[dict[str, Any]] = field(default_factory=list)
    final_summary: str | None = None
    finalized: bool = False
    steps_used: int = 0
    ending_balance_usd: float | None = None
    aborted_reason: str | None = None
    usage_totals: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def transcript_text(self) -> str:
        """A flat, human/LLM-readable rendering of the conversation."""
        lines: list[str] = []
        for e in self.transcript:
            if e.kind == "tool_call":
                lines.append(f"[TOOL CALL] {e.content}")
            elif e.kind == "tool_result":
                lines.append(f"[TOOL RESULT] {e.content}")
            else:
                lines.append(f"[{e.kind.upper()}] {e.content}")
        return "\n\n".join(lines)
