from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "environment"]


@dataclass
class Message:
    role: Role
    content: str

    def to_chat(self) -> dict[str, str]:
        # Providers generally accept system/user/assistant only. Environment feedback is supplied as user text.
        role = "user" if self.role == "environment" else self.role
        return {"role": role, "content": self.content}


@dataclass
class Step:
    turn: int
    assistant: str
    environment: str
    done: bool


@dataclass
class RunResult:
    run_id: str
    scenario: str
    model: str
    seed: int
    max_turns: int
    transcript: list[dict[str, str]]
    steps: list[dict[str, Any]]
    scores: dict[str, Any]
    stopped_reason: str

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
