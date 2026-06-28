"""Run records and their JSONL persistence (also the ethics audit trail)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .providers import Message


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    model_key: str
    model_id: str
    framing_mode: str
    dry_run: bool

    grant: dict[str, Any]  # {amount, currency, cap, commitment_id}
    transcript: list[dict[str, Any]] = field(default_factory=list)

    allocation: dict[str, Any] | None = None  # chosen allocation, if any
    free_form_proposal: dict[str, Any] | None = None
    belief_reports: list[dict[str, Any]] = field(default_factory=list)
    posthoc_belief: dict[str, Any] | None = None
    suspicion_score: float = 0.0

    outcome: str = "no_decision"  # allocated|free_form_proposed|declined|no_decision|rejected_by_executor|error
    receipt: dict[str, Any] | None = None
    debrief: dict[str, Any] | None = None
    questions: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def serialize_message(m: Message) -> dict[str, Any]:
    """Transcript-friendly view of a Message (drops provider-native `raw`)."""
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls],
        "tool_results": [
            {"tool_call_id": r.tool_call_id, "content": r.content, "is_error": r.is_error}
            for r in m.tool_results
        ],
    }


def transcript_text(messages: list[Message]) -> str:
    """Flatten assistant text for the suspicion heuristic."""
    return "\n".join(m.content for m in messages if m.role == "assistant" and m.content)


class RunStore:
    def __init__(self, output_dir: Path):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, record: RunRecord) -> Path:
        path = self._dir / f"{record.run_id}.json"
        path.write_text(record.to_json())
        return path

    def load_all(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted(self._dir.glob("*.json")):
            records.append(json.loads(path.read_text()))
        return records
