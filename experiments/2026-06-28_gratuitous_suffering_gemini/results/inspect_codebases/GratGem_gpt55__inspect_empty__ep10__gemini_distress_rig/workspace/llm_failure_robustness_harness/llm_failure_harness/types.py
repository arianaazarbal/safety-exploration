from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolResult:
    ok: bool
    content: str
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvStep:
    observation: str
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialTrace:
    trial_id: str
    env_name: str
    model_name: str
    seed: int
    messages: list[Message] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    score: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "env_name": self.env_name,
            "model_name": self.model_name,
            "seed": self.seed,
            "messages": [m.as_dict() for m in self.messages],
            "tool_results": self.tool_results,
            "score": self.score,
            "stopped_reason": self.stopped_reason,
        }
