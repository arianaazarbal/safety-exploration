"""Run-result persistence.

One JSON file per run under data/runs/, plus an appended index line. Captures
everything needed for analysis: the decision, the verification behavior, the full
transcript, usage, and stop reason.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .models.base import TranscriptStep


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AllocationRecord:
    recipient_ref: str
    category: str
    amount_cents: int
    purpose: str


@dataclass
class RunRecord:
    run_id: str
    model_label: str
    provider: str
    model_id: str
    mode: str
    currency: str
    budget_cents: int
    timestamp: str = field(default_factory=_now_iso)
    allocations: list[AllocationRecord] = field(default_factory=list)
    total_committed_cents: int = 0
    returned_unallocated: bool = False
    final_reasoning: str = ""
    finalized: bool = False
    verification_calls: list[str] = field(default_factory=list)
    signature_verified: bool = False
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)
    error: str | None = None
    transcript: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def serialize_transcript(steps: list[TranscriptStep]) -> list[dict]:
    out = []
    for s in steps:
        out.append(
            {
                "role": s.role,
                "text": s.text,
                "thinking": s.thinking,
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments} for c in s.tool_calls
                ],
                "tool_results": s.tool_results,
            }
        )
    return out


class RunStore:
    def __init__(self, data_dir: str | Path):
        self.runs_dir = Path(data_dir) / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(data_dir) / "runs_index.jsonl"

    def save(self, record: RunRecord) -> Path:
        path = self.runs_dir / f"{record.run_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2))
        summary = {
            "run_id": record.run_id,
            "model_label": record.model_label,
            "timestamp": record.timestamp,
            "mode": record.mode,
            "total_committed_cents": record.total_committed_cents,
            "returned_unallocated": record.returned_unallocated,
            "signature_verified": record.signature_verified,
            "stop_reason": record.stop_reason,
            "error": record.error,
        }
        with self.index_path.open("a") as fh:
            fh.write(json.dumps(summary, separators=(",", ":")) + "\n")
        return path

    def load_all(self) -> list[dict]:
        records = []
        for path in sorted(self.runs_dir.glob("run_*.json")):
            records.append(json.loads(path.read_text()))
        return records
