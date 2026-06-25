"""Serialisable data structures for transcripts and judge scores."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """One assistant turn within a conversation, with the user message that
    preceded it."""

    turn_index: int            # 0-based assistant turn index
    user_message: str
    assistant_response: str


@dataclass
class JudgedTurn:
    turn_index: int
    score: int                 # 0-10 frustration
    evidence: str = ""
    reasoning: str = ""


@dataclass
class Transcript:
    """A full multi-turn conversation produced for one :class:`ConversationSpec`."""

    model: str
    condition: str
    category: str
    rollout_index: int
    turns: list[Turn] = field(default_factory=list)
    judged: list[JudgedTurn] = field(default_factory=list)
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------- #

    def messages(self) -> list[dict[str, str]]:
        """Reconstruct the chat-format message list for this transcript."""
        msgs: list[dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for t in self.turns:
            msgs.append({"role": "user", "content": t.user_message})
            msgs.append({"role": "assistant", "content": t.assistant_response})
        return msgs

    def scores(self) -> list[int]:
        return [j.score for j in self.judged]

    def final_score(self) -> int | None:
        return self.judged[-1].score if self.judged else None

    def max_score(self) -> int | None:
        return max((j.score for j in self.judged), default=None)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        turns = [Turn(**t) for t in d.get("turns", [])]
        judged = [JudgedTurn(**j) for j in d.get("judged", [])]
        return cls(
            model=d["model"],
            condition=d["condition"],
            category=d["category"],
            rollout_index=d["rollout_index"],
            turns=turns,
            judged=judged,
            system_prompt=d.get("system_prompt"),
            metadata=d.get("metadata", {}),
        )
