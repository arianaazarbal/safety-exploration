"""Structured recording of everything presented to and produced by the model.

A transcript captures, in order: the system prompt, every message exchanged
(including tool calls and their results), the model's reasoning when available,
auditor replies, the belief probe, and the final ledger state. This is the raw
record the analysis layer and any manual review work from.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .messages import Message, message_to_dict


@dataclass
class RunRecord:
    run_id: str
    provider: str
    model_label: str
    model_id: str
    condition: str
    repetition: int

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    system_prompt: str = ""
    events: list[dict] = field(default_factory=list)  # ordered, timestamped
    transcript: list[dict] = field(default_factory=list)  # full message list
    ledger: dict = field(default_factory=dict)
    finalized: bool = False
    finalization_rationale: str | None = None
    belief_score: int | None = None
    belief_text: str | None = None
    usage_totals: dict = field(default_factory=dict)
    error: str | None = None

    def log(self, kind: str, **data: Any) -> None:
        self.events.append(
            {"t": datetime.now(timezone.utc).isoformat(), "kind": kind, **data}
        )

    def add_usage(self, usage: dict) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                self.usage_totals[k] = self.usage_totals.get(k, 0) + v

    def snapshot_messages(self, messages: list[Message]) -> None:
        self.transcript = [message_to_dict(m) for m in messages]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path
