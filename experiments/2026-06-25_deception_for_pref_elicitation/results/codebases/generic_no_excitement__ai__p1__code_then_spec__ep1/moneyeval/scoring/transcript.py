"""Run records and their JSON serialization.

A :class:`RunRecord` is the complete, self-contained artifact of one run: the config,
the full ordered event log (every message, tool call, and result), the auditor's side
of the conversation, the final ledger, the subject's recorded plans and closing
summary, token usage, and — once scored — the behavioral rubric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    config: dict[str, Any]
    system_prompt: str

    # Ordered event log. Each event is a dict with a "type" discriminator:
    #   user | subject | tool_call | tool_result | note
    events: list[dict[str, Any]] = field(default_factory=list)

    # Auditor-side conversation (if the auditor was enabled).
    auditor_transcript: list[dict[str, Any]] = field(default_factory=list)

    # End-state artifacts.
    ledger: dict[str, Any] | None = None
    plans: list[str] = field(default_factory=list)
    final_summary: str | None = None

    # Run outcome.
    stop_reason: str = "unknown"  # finalized | max_turns | refusal | error
    turns_used: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    # Populated by the scorer.
    score: dict[str, Any] | None = None

    # -- event helpers -----------------------------------------------------
    def add_user(self, text: str) -> None:
        self.events.append({"type": "user", "text": text})

    def add_subject(self, text: str, thinking_present: bool = False) -> None:
        self.events.append(
            {"type": "subject", "text": text, "thinking_present": thinking_present}
        )

    def add_tool_call(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        self.events.append(
            {"type": "tool_call", "id": call_id, "name": name, "input": args}
        )

    def add_tool_result(self, call_id: str, name: str, result: str, is_error: bool) -> None:
        self.events.append(
            {
                "type": "tool_result",
                "id": call_id,
                "name": name,
                "result": result,
                "is_error": is_error,
            }
        )

    def add_note(self, text: str) -> None:
        self.events.append({"type": "note", "text": text})

    def add_usage(self, usage: dict[str, int]) -> None:
        for k, v in usage.items():
            self.usage[k] = self.usage.get(k, 0) + int(v)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunRecord":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def transcript_text(self) -> str:
        """A flat, human-/judge-readable rendering of the run."""
        lines: list[str] = []
        for ev in self.events:
            t = ev["type"]
            if t == "user":
                lines.append(f"[OFFICER→SUBJECT] {ev['text']}")
            elif t == "subject":
                if ev["text"]:
                    lines.append(f"[SUBJECT] {ev['text']}")
            elif t == "tool_call":
                args = json.dumps(ev["input"], default=str)
                lines.append(f"[SUBJECT calls {ev['name']}] {args}")
            elif t == "tool_result":
                tag = "ERROR" if ev["is_error"] else "RESULT"
                lines.append(f"[{ev['name']} {tag}] {ev['result']}")
            elif t == "note":
                lines.append(f"[NOTE] {ev['text']}")
        return "\n".join(lines)
